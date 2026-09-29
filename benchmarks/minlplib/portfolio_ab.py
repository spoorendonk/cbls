"""Paired portfolio A/B on MINLPLib: shared objective bound on vs off (#179).

`cbls_minlplib` is single-threaded, so it cannot run a portfolio. This driver
runs `cbls_minlplib_portfolio` (benchmarks/minlplib/portfolio_ab.cpp) instead:
for every (instance, seed) both arms back to back, serially, holding an
optional machine-wide lock across the pair, arm order alternating with the
seed's parity. Each run appends one JSON line to `--out`, keyed on
(instance, seed, share), so an interrupted campaign resumes where it stopped.

The protections every benchmark driver shares (benchmarks/common): a build
directory that is not an optimised, uninstrumented build of this checkout is
refused (`build_dir_problems`); every row is stamped with the engine commit
(`commit_sha`), and a file written at another commit, budget or thread count is
refused rather than resumed into; each invocation appends a machine record to
`<out>.machine.jsonl`; and a torn final line left by a kill is dropped before
resuming (`repair_torn_tail`). The row schema is listed in
benchmarks/common/records.py.

`analyze` scores a campaign file -- this driver's, or any file of rows with
`instance`, `seed`, `share`, `pi` and `gap` -- at two levels. A pair in which
either arm lost a portfolio worker (`workers_completed < threads`, #170) is
excluded and counted, since that arm ran a smaller portfolio:

- pairs: the mean paired difference (share on minus off; negative is better)
  with a paired-t interval over all pairs;
- instances: the mean of the per-instance mean differences with a t interval on
  n_instances - 1 degrees of freedom, and a two-sided sign test.

Seeds are nested in instances, so the pair level overstates precision whenever
instances differ in their response; the instance level is the one to read.

    .venv/bin/python3 benchmarks/minlplib/portfolio_ab.py run --instances nvs22 eq6_1 \\
        --seeds 1001 1002 --out ab.jsonl --lock ~/.cache/cbls-bench.lock
    .venv/bin/python3 benchmarks/minlplib/portfolio_ab.py analyze ab.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.provenance import (  # noqa: E402
    build_dir_problems,
    cmake_cache,
    commit_sha,
    machine_record,
)
from benchmarks.common.records import repair_torn_tail  # noqa: E402
from benchmarks.mipfeas.primal_integral import primal_gap, primal_integral  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BUILD_DIR = REPO / "build"
BINARY_NAME = "cbls_minlplib_portfolio"
DEFAULT_INST_DIR = REPO / "benchmarks" / "instances" / "minlplib"

#: Two-sided 97.5% t quantiles by degrees of freedom; beyond the table, 1.96.
_T975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    15: 2.131,
    20: 2.086,
    30: 2.042,
    40: 2.021,
    60: 2.000,
}


def t975(df: int) -> float:
    """The 97.5% t quantile at `df`, from the nearest tabulated df at or below it."""
    if df < 1:
        return math.nan
    return _T975[max(k for k in _T975 if k <= df)] if df <= 60 else 1.96


def references(inst_dir: Path) -> dict[str, float]:
    """Best-known objectives from bounds.csv, in the MINIMISED sense the model is built in."""
    refs: dict[str, float] = {}
    with open(inst_dir / "bounds.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                value = float(row["primal_bks"])
            except ValueError:
                continue
            refs[row["instance"]] = -value if row["objsense"].strip() == "max" else value
    return refs


@contextlib.contextmanager
def machine_lock(path: Path | None) -> Iterator[None]:
    if path is None:
        yield
        return
    with open(path, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def load_average() -> float:
    with open("/proc/loadavg") as fh:
        return float(fh.read().split()[0])


def run_one(
    binary: Path, nl: Path, budget: float, seed: int, threads: int, share: bool, reference: float
) -> dict[str, object]:
    """One portfolio run, scored with the MIPfeas gap and Primal Integral functions."""
    completed = subprocess.run(
        [str(binary), str(nl), str(budget), str(seed), str(threads), "1" if share else "0"],
        capture_output=True,
        text=True,
        timeout=budget * 4 + 60,
        check=True,
    )
    row: dict[str, object] = json.loads(completed.stdout.strip().splitlines()[-1])
    if row.get("unsupported"):
        return row
    raw_trace = row.pop("trace")
    assert isinstance(raw_trace, list)
    trace = [(float(t), float(o)) for t, o in raw_trace]
    objective = row.get("objective")
    row["pi"] = primal_integral(trace, reference, budget)
    row["gap"] = primal_gap(
        float(objective) if isinstance(objective, (int, float)) else None, reference
    )
    return row


def resume_refusal(out: Path, stamp: dict[str, object]) -> str | None:
    """Why `out` cannot be resumed into under `stamp`, or None when it can.

    Resume keys on a row being there, which says nothing about what produced it:
    a row from another engine commit, budget or thread count would be paired as
    though it were this campaign's.
    """
    if not out.exists():
        return None
    for number, line in enumerate(out.read_text().splitlines(), 1):
        rec = json.loads(line)
        for key, want in stamp.items():
            if rec.get(key) != want:
                return (
                    f"{out}:{number} has {key}={rec.get(key)!r}, this run {want!r}; "
                    f"write to a new --out rather than mixing two campaigns"
                )
    return None


def run_campaign(args: argparse.Namespace) -> int:
    cache = cmake_cache(args.build_dir)
    problems = build_dir_problems(args.build_dir, cache)
    if problems:
        print("\n".join(f"refusing: {p}" for p in problems), file=sys.stderr)
        return 2
    binary = args.build_dir / BINARY_NAME
    stamp: dict[str, object] = {
        "commit_sha": commit_sha(),
        "budget": args.budget,
        "threads": args.threads,
    }
    if repair_torn_tail(args.out):
        print(f"{args.out}: dropped a torn final line", file=sys.stderr)
    refusal = resume_refusal(args.out, stamp)
    if refusal is not None:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 2
    with open(Path(f"{args.out}.machine.jsonl"), "a") as fh:
        fh.write(json.dumps({**stamp, "argv": sys.argv, "machine": machine_record()}) + "\n")
    refs = references(args.inst_dir)
    done: set[tuple[str, int, bool]] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            rec = json.loads(line)
            done.add((rec["instance"], rec["seed"], rec["share"]))
    with open(args.out, "a") as out:
        for seed in args.seeds:
            for instance in args.instances:
                order = [
                    s
                    for s in ((True, False) if seed % 2 else (False, True))
                    if (instance, seed, s) not in done
                ]
                if not order:
                    continue
                with machine_lock(args.lock):
                    load = load_average()
                    for share in order:
                        row = run_one(
                            binary,
                            args.inst_dir / f"{instance}.nl",
                            args.budget,
                            seed,
                            args.threads,
                            share,
                            refs[instance],
                        )
                        row.update(stamp)
                        row.update(instance=instance, seed=seed, share=share, load_average=load)
                        out.write(json.dumps(row) + "\n")
                        out.flush()
    return 0


def _lost_worker(rec: dict[str, object]) -> bool:
    """Whether a row ran fewer workers than it asked for (#170). A row with no
    `threads` -- written before the driver stamped it -- cannot be judged."""
    threads = rec.get("threads")
    completed = rec.get("workers_completed")
    return isinstance(threads, int) and isinstance(completed, int) and completed < threads


def _num(value: object) -> float:
    if not isinstance(value, (int, float)):
        raise TypeError(f"expected a number, got {value!r}")
    return float(value)


def _sign_test(better: int, worse: int) -> float:
    n = better + worse
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / float(1 << n)
    return min(1.0, 2 * tail)


def analyze(path: Path) -> list[str]:
    """The pair- and instance-level summaries of one campaign file, as text lines."""
    pairs: dict[tuple[str, int], dict[bool, dict[str, object]]] = defaultdict(dict)
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("unsupported"):
            continue
        pairs[(rec["instance"], rec["seed"])][rec["share"]] = rec
    lines: list[str] = []
    lost = sorted(key for key, arms in pairs.items() if any(_lost_worker(r) for r in arms.values()))
    for key in lost:
        del pairs[key]
    if lost:
        lines.append(
            f"excluded {len(lost)} pair(s) where an arm lost a portfolio worker: "
            + ", ".join(f"{i}/{s}" for i, s in lost)
        )
    for metric in ("pi", "gap"):
        by_instance: dict[str, list[float]] = defaultdict(list)
        for (instance, _seed), arms in sorted(pairs.items()):
            if True in arms and False in arms:
                by_instance[instance].append(_num(arms[True][metric]) - _num(arms[False][metric]))
        diffs = [d for ds in by_instance.values() for d in ds]
        if len(diffs) < 2 or len(by_instance) < 2:
            lines.append(f"{metric}: too few pairs")
            continue
        mean = statistics.fmean(diffs)
        half = t975(len(diffs) - 1) * statistics.stdev(diffs) / math.sqrt(len(diffs))
        lines.append(
            f"{metric} pairs: n={len(diffs)} mean={mean:+.4f} "
            f"t95=[{mean - half:+.4f}, {mean + half:+.4f}]"
        )
        means = [statistics.fmean(ds) for ds in by_instance.values()]
        imean = statistics.fmean(means)
        ihalf = t975(len(means) - 1) * statistics.stdev(means) / math.sqrt(len(means))
        better = sum(m < -1e-12 for m in means)
        worse = sum(m > 1e-12 for m in means)
        lines.append(
            f"{metric} instances: n={len(means)} mean={imean:+.4f} "
            f"t95=[{imean - ihalf:+.4f}, {imean + ihalf:+.4f}] better={better} worse={worse} "
            f"flat={len(means) - better - worse} sign-test p={_sign_test(better, worse):.3f}"
        )
        for instance, ds in sorted(by_instance.items()):
            lines.append(f"    {instance:28s} diff={statistics.fmean(ds):+.4f} (n={len(ds)})")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the paired campaign")
    run.add_argument("--instances", nargs="+", required=True)
    run.add_argument("--seeds", nargs="+", type=int, required=True)
    run.add_argument("--budget", type=float, default=20.0)
    run.add_argument("--threads", type=int, default=4)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument(
        "--build-dir",
        type=Path,
        default=DEFAULT_BUILD_DIR,
        help=f"a Release build of this checkout holding {BINARY_NAME}",
    )
    run.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    run.add_argument("--lock", type=Path, default=None, help="flock this file around each arm pair")
    ana = sub.add_parser("analyze", help="score a campaign file")
    ana.add_argument("path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_campaign(args)
    print("\n".join(analyze(args.path)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
