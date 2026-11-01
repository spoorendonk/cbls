"""The SCIP half of a full-library MINLPLib campaign, one shard per instance (#305).

`reference_solve.py` writes one CSV for a whole roster. The full-library
campaign (#256) instead solves every instance of
`<nl-dir>/bounds.csv` in its own process, `--jobs` at a time, and leaves one
shard per instance in `SHARD_DIR`. That shard layout is what
`gm_ratio.py --scip-shards` reads:

* `<instance>.csv`: `reference_solve.py`'s row for the instance. It is written
  as `<instance>.csv.tmp` and renamed unless the process was killed (exit 124
  or 137, as `sh` reports it), so a killed solve never leaves a half-written
  row. Any other exit with a row written keeps the row, as the #256 wrapper did.
* `<instance>.merged.csv`: the subset run's merged CSV (`--merged-out`);
  the scorer ignores it.
* `<instance>.failed`: one line, `exit <rc>`, when the process ran past the
  wrapper cap (`timeout` exit 124, or 137 after `-k`) or wrote no row.
* `<instance>.log`: the process's stdout and stderr.

SCIP runs at its defaults (`--seed 0`: no `randomseedshift`) for `--time-limit`
seconds, one thread per process. The budget covers the solve; reading the `.nl`
file is outside it, so the per-instance wrapper cap (`--cap`, 15 min by
default) is far above the budget and only stops a hung read or a solve that
ignores its limit.

Resumable: an instance whose `<instance>.csv` holds a whole row is skipped, so a re-run
retries exactly the `.failed` and unfinished ones. A shard that succeeds on
retry drops its `.failed` marker. The #256 rule for what is left after one
retry (hub, 2026-11-01): run it again once with the memory cap raised (the
campaign ran under a 110G cgroup cap), and a failure that remains counts as
"SCIP: no solution", named with its exit.

Usage, from any directory, under the machine lock (one lock for the whole run;
`--jobs` single-thread solves share the machine):

    xk lock -- .venv/bin/python3 benchmarks/minlplib/scip_shards.py \\
        --nl-dir NL_DIR --shards SHARD_DIR [--jobs 16] [--time-limit 60] \\
        [--cap 900] [--instances NAME ...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.minlplib.gm_ratio import (  # noqa: E402
    SCIP_FAILED_SUFFIX,
    SCIP_MERGED_SUFFIX,
    has_whole_row,
)
from benchmarks.minlplib.reference_solve import load_bounds  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

REFERENCE_SOLVE = Path(__file__).resolve().parent / "reference_solve.py"

#: Exit codes of `timeout`: the cap ran out (124), or the process ignored
#: SIGTERM and was killed after the grace period (137). Neither leaves a row.
CAP_EXITS = (124, 137)
#: Seconds `timeout -k` waits after SIGTERM before SIGKILL.
KILL_GRACE_SECONDS = 10


def reference_command(
    nl_dir: Path, time_limit: float, seed: int
) -> Callable[[str, Path, Path], list[str]]:
    """The real solve: `reference_solve.py` on one instance, SCIP at its defaults.

    The returned callable maps (instance, `.csv.tmp` path, `.merged.csv` path)
    to the command line; tests pass a stub of the same shape to `run_shards`.
    """

    def command(name: str, out: Path, merged: Path) -> list[str]:
        return [
            sys.executable,
            str(REFERENCE_SOLVE),
            "--inst-dir",
            str(nl_dir),
            "--time-limit",
            f"{time_limit:g}",
            "--seed",
            str(seed),
            "--instances",
            name,
            "--out",
            str(out),
            "--merged-out",
            str(merged),
        ]

    return command


def capped(cmd: Sequence[str], cap_seconds: float) -> list[str]:
    """`cmd` under coreutils `timeout`: SIGTERM at the cap, SIGKILL a grace later."""
    return ["timeout", "-k", str(KILL_GRACE_SECONDS), f"{cap_seconds:g}", *cmd]


def shell_exit(returncode: int) -> int:
    """`returncode` as `sh` reports it: a death by signal N is 128+N, not -N.

    A solve that aborts or is killed outside the cap (SIGABRT, an OOM SIGKILL)
    makes `timeout` re-raise that signal on itself, which `subprocess` reports
    as negative. The #256 markers were written by `sh` (`exit 134` for an
    abort), so a retry must record the same failure the same way, and a
    SIGKILL must read as 137 to count as a failure.
    """
    return 128 - returncode if returncode < 0 else returncode


def solve_shard(
    shards: Path,
    name: str,
    command: Callable[[str, Path, Path], list[str]],
    cap_seconds: float,
) -> int | None:
    """Solve one instance into its shard; None when its row is already there.

    Returns the process's exit code, as a shell reports it. The row is published
    (renamed from `.csv.tmp`) when the process wrote one whole row and was not
    stopped by the cap; otherwise `.failed` records the exit. "Whole row" is
    `gm_ratio.has_whole_row`, the scorer's own test, so a torn file is neither
    published nor skipped on resume.
    """
    csv_path = shards / f"{name}.csv"
    if csv_path.exists() and has_whole_row(csv_path):
        return None
    tmp = shards / f"{name}.csv.tmp"
    merged = shards / f"{name}{SCIP_MERGED_SUFFIX}"
    failed = shards / f"{name}{SCIP_FAILED_SUFFIX}"
    tmp.unlink(missing_ok=True)
    with (shards / f"{name}.log").open("w") as log:
        rc = shell_exit(
            subprocess.run(
                capped(command(name, tmp, merged), cap_seconds),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        )
    if tmp.exists() and has_whole_row(tmp) and rc not in CAP_EXITS:
        tmp.replace(csv_path)
        failed.unlink(missing_ok=True)
    else:
        failed.write_text(f"exit {rc}\n")
    return rc


def run_shards(
    shards: Path,
    names: Sequence[str],
    command: Callable[[str, Path, Path], list[str]],
    *,
    jobs: int,
    cap_seconds: float,
) -> dict[str, int | None]:
    """Every instance of `names`, `jobs` processes at a time; exit code by instance.

    On Ctrl-C no further solve starts. The running ones finish (at most the
    cap): `timeout` puts each in its own process group, out of the terminal's
    SIGINT. Their shards are written as usual, so a re-run resumes cleanly.
    """
    shards.mkdir(parents=True, exist_ok=True)

    def one(name: str) -> tuple[str, int | None]:
        rc = solve_shard(shards, name, command, cap_seconds)
        print(f"{name} {'skipped' if rc is None else f'exit {rc}'}", flush=True)
        return name, rc

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        # `Executor.map`'s iterator cancels its pending futures when an exception
        # (KeyboardInterrupt included) interrupts it.
        return dict(pool.map(one, names))


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve a MINLPLib roster in SCIP, one shard per instance."
    )
    parser.add_argument("--nl-dir", type=Path, required=True, help="holds bounds.csv and .nl")
    parser.add_argument("--shards", type=Path, required=True, help="shard directory")
    parser.add_argument("--jobs", type=int, default=16, help="parallel single-thread solves")
    parser.add_argument("--time-limit", type=float, default=60.0, help="SCIP budget, seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cap", type=float, default=900.0, help="per-instance wrapper cap, seconds"
    )
    parser.add_argument("--instances", nargs="+", default=[], help="subset; default roster")
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    # reference_solve.py refuses these with exit 2; caught here, they cannot turn
    # a whole roster into `.failed` markers that read as "SCIP: no solution".
    if args.time_limit <= 0:
        parser.error("--time-limit must be positive")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.cap <= args.time_limit:
        parser.error("--cap must exceed --time-limit: the cap stops hangs, not solves")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roster = [b.instance for b in load_bounds(args.nl_dir / "bounds.csv")]
    wanted = set(args.instances)
    unknown = sorted(wanted - set(roster))
    if unknown:
        print(f"not in the roster: {', '.join(unknown)}", file=sys.stderr)
        return 2
    names = [n for n in roster if n in wanted] if wanted else roster
    print(f"[{_now()}] start: {len(names)} instance(s), {args.jobs} at a time", flush=True)
    exits = run_shards(
        args.shards,
        names,
        reference_command(args.nl_dir, args.time_limit, args.seed),
        jobs=args.jobs,
        cap_seconds=args.cap,
    )
    failed = sorted(n for n in names if (args.shards / f"{n}{SCIP_FAILED_SUFFIX}").exists())
    solved = sum(1 for rc in exits.values() if rc is not None) - len(failed)
    skipped = sum(1 for rc in exits.values() if rc is None)
    print(
        f"[{_now()}] done: {solved} solved, {skipped} skipped (row present), "
        f"{len(failed)} failed{': ' + ', '.join(failed) if failed else ''}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
