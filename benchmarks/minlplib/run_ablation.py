"""Run the MINLPLib ablation campaign of issue #143, serially and resumably.

Which parts of the engine the published MINLPLib result depends on: four arms
plus a gated fifth over the 50-instance roster at the published 60s budget,
three seeds each, **interleaved per instance** so machine drift hits every arm
equally instead of accumulating against the last one:

    control            (all defaults)
    no-float-hook      --no-float-hook
    unproductive-0     --unproductive-iters 0
    no-compound-moves  --no-compound-moves
    no-lns             --no-lns              (gated -- see LNS_GATE_* below)

Usage, from a configured Release build directory and a clean checkout:

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir /path/to/scratch
    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir ... --dry-run
    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir ... --report-only

The same invocation resumes: each completed run is appended to
`<out-dir>/results.csv` and flushed before the next starts, and a restart skips
exactly the `(instance, arm, seed)` triples already recorded.

Refusals, not warnings, since each would otherwise yield an untrustworthy number
hours later:

* **Writing near the published tables.** `--out-dir` must be outside
  `benchmarks/instances/`. The runner refuses `comparison.csv` and
  `anytime_trace.csv` too (#136), but its guard alone would let the default
  *control* arm through.
* **Sharing the machine.** `jobs.wallclock_lock` (shared with
  `run_benchmark.py`), an exclusive `<out-dir>/campaign.lock`, and a load check
  before the first solve: the budget is wall-clock.
* **Resuming into another configuration.** The stamp records commit, budget,
  seeds and arm set; a mismatch is refused, since a mix measures engines, not
  arms.
* Everything `run_benchmark.common_preflight` refuses: a dirty tree, a
  non-Release or sanitizer build, a build from another checkout, missing `.nl`
  files.

Changes no engine default and writes no published table; an arm arguing for
a new default is a separate change (#143). The removed
`transfer-145` (#145) and `presolve-322` (#331) campaigns are at `ef795b55`; an
out-dir stamped by either is refused.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Run as a script, only this directory is on sys.path; add the repo root so the
# `benchmarks.*` imports resolve as under `python3 -m`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.jobs import LockHeldError, run_process, wallclock_lock  # noqa: E402
from benchmarks.common.provenance import REPO_ROOT, commit_sha, modified_paths  # noqa: E402
from benchmarks.common.records import (  # noqa: E402
    append_csv_row,
    atomic_write,
    csv_header,
    csv_text,
    repair_torn_tail,
    stamp_refusal,
)
from benchmarks.minlplib import independent_check  # noqa: E402
from benchmarks.minlplib import runner as runner_contract  # noqa: E402
from benchmarks.minlplib.ablation_report import (  # noqa: E402
    CONTROL_ARM,
    PROBE_ARM_NAME,
    render_report,
)
from benchmarks.minlplib.run_benchmark import (  # noqa: E402
    DEFAULT_BUILD_DIR,
    DEFAULT_INST_DIR,
    DEFAULT_TIME_LIMIT,
    common_preflight,
    roster_from_bounds,
)
from benchmarks.minlplib.runner import (  # noqa: E402
    RUNNER_COLUMNS,
    RUNNER_EXIT_ERRORED,
    RUNNER_FAILED_NOTE,
    completed_search,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Nothing the campaign writes may land here: it holds every published table,
#: and replacing one is the one failure this driver cannot recover from.
PROTECTED_ROOT = REPO_ROOT / "benchmarks" / "instances"

#: The append-only record of completed runs. It is both the campaign's output
#: and its resume key, so it is never rewritten in place.
RESULTS_NAME = "results.csv"

#: Records which configuration the rows in an out-dir belong to.
STAMP_NAME = "stamp.txt"

#: Held for the whole campaign, so a second driver refuses instead of halving
#: both runs' iteration counts.
LOCK_NAME = "campaign.lock"

#: The gate's reading and verdict, so the decision is recoverable from the output.
GATE_NAME = "lns_gate.json"


@dataclass(frozen=True)
class Arm:
    """One ablation arm: a name for the rows and the flags that produce it."""

    name: str
    flags: tuple[str, ...]


#: The arms that always run. `control` runs in this sitting, not taken from the
#: published table, which was measured under other machine load (#143).
ARMS: tuple[Arm, ...] = (
    Arm(CONTROL_ARM, ()),
    Arm("no-float-hook", ("--no-float-hook",)),
    Arm("unproductive-0", ("--unproductive-iters", "0")),
    # Compound moves default on since 2026-10-07, so the arm turns them off (#143).
    Arm("no-compound-moves", ("--no-compound-moves",)),
)

#: The gated fifth arm.
GATED_ARM = Arm("no-lns", ("--no-lns",))

#: The gate probe: `control`'s flags and budget at one seed before the campaign,
#: only to read `lns_repairs`. Named apart so scoring refuses to average it with
#: the campaign's control, measured hours apart.
PROBE_ARM = Arm(PROBE_ARM_NAME, ())

#: The campaign every stamp records (see `campaign_stamp`, `campaign_refusal`).
ABLATION_CAMPAIGN = "ablation"

#: Three seeds, as the protocol requires. 1 is the published table's seed, so
#: control is comparable once that table is regenerated (it predates compound
#: moves defaulting on, 2026-10-07, and records `compound_moves=off`); 2 and 3
#: are the next two, unremarkable so they cannot look picked after the fact.
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3)

#: THE LNS GATE (#143): the `no-lns` arm runs only if the probe's `lns_repairs`
#: shows LNS working at this budget. The arm is skipped only at ZERO repairs
#: roster-wide, the point it becomes a provable no-op: `diversify()` in
#: `src/search.cpp` tests `lns_ != nullptr` first and otherwise perturbs, so with
#: no repair `--no-lns` takes the same branch and consumes the RNG identically.
#:
#: Gates on ATTEMPTS, not `lns_repairs_accepted` (#150): a rejected repair still
#: randomises, runs `fj_nl_initialize` and rolls back, consuming RNG and budget,
#: so it diverges, and that budget cost is what the arm prices. Acceptance is
#: reported beside attempts by `ablation_report.py`. Both thresholds are recorded
#: in `lns_gate.json`; `--lns-arm on|off` overrides the gate.
LNS_GATE_MIN_REPAIRS = 1
LNS_GATE_MIN_INSTANCES = 1

#: Share of probe runs that must carry an `lns_repairs` reading for a SKIP to
#: stand; below it the arm runs, since mostly-unsolved runs are not "the counter
#: reading that justified skipping" (#143). Fails toward spending machine time.
GATE_MIN_READABLE_FRACTION = 0.9

#: One-minute load average above which the campaign refuses to start.
#:
#: Well below one busy core: a single-threaded campaign oscillates around 1.00,
#: so 1.0 would admit a second one about half the time. `jobs.wallclock_lock`
#: excludes this user's timed drivers; this catches other load. `--allow-busy`
#: overrides, for an average still decaying from a campaign that just died.
MAX_LOAD_AVERAGE = 0.4

#: Runner columns copied onto a result row, in runner order, minus the two the
#: provenance columns already carry.
_RUNNER_COLUMNS: tuple[str, ...] = tuple(
    column for column in RUNNER_COLUMNS if column not in ("instance", "commit_sha")
)

#: `results.csv` columns: `instance,arm,seed` is the resume key, the provenance
#: columns make a row self-describing, the rest are the runner's row verbatim.
RESULT_COLUMNS: tuple[str, ...] = (
    "instance",
    "arm",
    "arm_flags",
    "seed",
    "time_limit",
    "commit_sha",
    *_RUNNER_COLUMNS,
)


@dataclass(frozen=True)
class Run:
    """One solve: an instance, an arm and a seed."""

    instance: str
    arm: Arm
    seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        """The resume key: what `results.csv` records and a restart skips on."""
        return (self.instance, self.arm.name, self.seed)

    @property
    def slug(self) -> str:
        return f"{self.instance}__{self.arm.name}__seed{self.seed}"


def campaign_plan(roster: Sequence[str], seeds: Sequence[int], arms: Sequence[Arm]) -> list[Run]:
    """The campaign's run order: instance-major, then seed, then arm.

    This ordering IS the protocol (#143): drift between control and an arm is
    minutes, not hours, and arms rotate innermost so each paired (control, arm)
    comparison at one seed is as close in time as possible. Arm-major order is
    forbidden: an hour of drift is indistinguishable from the arm effects this
    roster can resolve.
    """
    return [Run(instance, arm, seed) for instance in roster for seed in seeds for arm in arms]


def probe_plan(roster: Sequence[str], seed: int) -> list[Run]:
    """The gate probe: the control configuration over the roster at one seed.

    One seed: the gate asks whether LNS ever fires, not how often. The pass costs
    ~7% of the campaign and keeps the gated arm's interleave symmetric; gating per
    instance in the loop would put it last in every block.
    """
    return [Run(instance, PROBE_ARM, seed) for instance in roster]


def scratch_refusal(out_dir: Path) -> str | None:
    """Refuse an `--out-dir` from which the campaign could reach a published table.

    A path test, not a filename test: the whole tree is off limits, since the
    campaign and the runner write several files under it.
    """
    resolved = out_dir.resolve()
    protected = PROTECTED_ROOT.resolve()
    if resolved == protected or protected in resolved.parents:
        return (
            f"--out-dir {out_dir} is inside {PROTECTED_ROOT}, which holds every published "
            "results table in the repository; the campaign writes to scratch paths only"
        )
    if resolved in protected.parents or resolved == REPO_ROOT.resolve():
        return (
            f"--out-dir {out_dir} contains {PROTECTED_ROOT}; pass a scratch directory that "
            "is not an ancestor of the published tables"
        )
    return None


def load_refusal(allow_busy: bool) -> str | None:
    """Refuse to start on a machine that is already busy."""
    if allow_busy:
        return None
    load = os.getloadavg()[0]
    if load > MAX_LOAD_AVERAGE:
        return (
            f"one-minute load average is {load:.2f} (> {MAX_LOAD_AVERAGE:g}); these are "
            "wall-clock-budgeted solves and a shared machine measures a different engine. "
            "Wait for the machine to go quiet, or pass --allow-busy if this is the average "
            "still decaying from a campaign that has already exited."
        )
    return None


@contextlib.contextmanager
def campaign_lock(out_dir: Path) -> Iterator[None]:
    """Hold an exclusive lock for the whole campaign, or refuse.

    The serial guarantee against a second *driver*. `flock` is per open file
    description, so a second attempt fails from this process too.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LOCK_NAME
    # "a+", not "w": truncating before the flock fails would wipe the holder's
    # pid line, which the refusal names.
    handle = path.open("a+")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "pid unknown"
            raise RuntimeError(
                f"another campaign holds {path} ({holder}); timed comparisons must never "
                "share the machine. Wait for it to finish. Do NOT delete the lock file: the "
                "lock is the flock, which is held on the INODE and which the kernel drops "
                "when the holder exits -- so a file left by a crashed driver is already "
                "unlocked and deleting it gains nothing, while deleting a live one only lets "
                "this driver create a fresh inode, flock that, and run beside the first."
            ) from exc
        # Truncated only now that the lock is ours, so the file holds one line.
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        handle.close()


def arm_label(arm: Arm) -> str:
    """An arm as the stamp records it: its name and the flags that produce it."""
    flags = " ".join(arm.flags)
    return f"{arm.name}[{flags}]"


def roster_key(roster: Sequence[str]) -> str:
    """The stamp's hash of a roster, in run order; `all` for the whole roster."""
    return hashlib.sha256(",".join(roster).encode()).hexdigest()[:12] if roster else "all"


def campaign_stamp(
    sha: str,
    time_limit: float,
    seeds: Sequence[int],
    arms: Sequence[Arm],
    roster: Sequence[str] = (),
    lns_arm: str = "auto",
) -> str:
    """The configuration an out-dir's rows belong to, one field per line.

    Roster and gate setting are stamped like the commit: else `--instances a b`
    resumes into a partial campaign, and `--lns-arm on|off` overwrites the
    recorded gate reading that justified a skip. Arm FLAGS are stamped because
    a name does not identify an arm. The constant `campaign=ablation` line lets
    older out-dirs resume and lets `campaign_refusal` recognise other campaigns.
    """
    return (
        f"campaign={ABLATION_CAMPAIGN}\n"
        f"commit={sha}\n"
        f"time-limit={time_limit:g}\n"
        f"seeds={','.join(str(s) for s in seeds)}\n"
        f"arms={','.join(arm_label(arm) for arm in arms)}\n"
        f"roster={len(roster)}:{roster_key(roster)}\n"
        f"lns-arm={lns_arm}\n"
    )


def recorded_campaign(out_dir: Path) -> str | None:
    """The campaign an out-dir's stamp names, or None where it names none.

    None for an absent stamp or one predating the field; calling such an
    out-dir an ablation is the caller's decision.
    """
    path = out_dir / STAMP_NAME
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key == "campaign":
            return value
    return None


def stamp_conflict(out_dir: Path, stamp: str, *, resume: bool) -> str | None:
    """Refuse an out-dir written by a different engine, budget, seed set or arm set.

    A recorded resume triple says nothing about what produced it; without this,
    a resume at another commit reports engine differences as arm effects.
    """
    return stamp_refusal(
        out_dir / STAMP_NAME,
        stamp,
        resume=resume,
        label="recorded",
        advice="Use a fresh --out-dir, or pass --no-resume to start this one over (which moves "
        "the recorded rows aside rather than adding to them); mixing two configurations "
        "into one campaign measures the configurations, not the arms.",
    )


def recorded_keys(path: Path) -> set[tuple[str, str, int]]:
    """The `(instance, arm, seed)` triples already in `results.csv`.

    A row with an unparseable seed is not recorded, so the campaign re-runs it.
    """
    if not path.exists():
        return set()
    keys: set[tuple[str, str, int]] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                seed = int(row["seed"])
            except (TypeError, ValueError):
                continue
            keys.add((row["instance"], row["arm"], seed))
    return keys


def header_conflict(path: Path) -> str | None:
    """The reason an existing `results.csv` cannot be appended to, or None.

    Appending `RESULT_COLUMNS`-wide rows under another schema's header exits 0
    and makes `csv.DictReader` bind values to the wrong names (#150). The stamp
    usually catches a schema change via the commit, but can be absent. Checked
    here, not in `open_results`, so it reads like the other refusals and
    `execute_runs` keeps a create-if-missing call.
    """
    if not path.exists() or not path.read_bytes():
        return None
    header = csv_header(path)
    if header == list(RESULT_COLUMNS):
        return None
    missing = [c for c in RESULT_COLUMNS if c not in header]
    extra = [c for c in header if c not in RESULT_COLUMNS]
    return (
        f"{path} was written under a different row schema and appending to it would bind "
        f"values to the wrong columns (missing: {missing or 'none'}; unexpected: "
        f"{extra or 'none'}). Score it where it stands with --report-only, or start a new "
        "campaign with --no-resume or a fresh --out-dir."
    )


def open_results(path: Path) -> None:
    """Create `results.csv` with its header if it is not there yet."""
    if path.exists() and path.read_bytes():
        return
    with path.open("w", newline="") as fh:
        csv.writer(fh).writerow(RESULT_COLUMNS)


def append_result(path: Path, row: dict[str, str]) -> None:
    """Append one completed run, durably, before the next one starts."""
    append_csv_row(path, [row[column] for column in RESULT_COLUMNS])


def runner_command(args: argparse.Namespace, sha: str, run: Run, out_dir: Path) -> list[str]:
    """The `cbls_minlplib` invocation for one run.

    Its `--instance` keeps even the all-default control arm off `comparison.csv`:
    the runner refuses a subset run there.
    """
    return runner_contract.runner_command(
        args.build_dir,
        args.inst_dir,
        time_limit=args.time_limit,
        seed=run.seed,
        sha=sha,
        instance=run.instance,
        out=out_dir / "runs" / f"{run.slug}.csv",
        solution_dir=solution_dir(out_dir, run),
        extra=run.arm.flags,
    )


def locked_command(args: argparse.Namespace, sha: str, run: Run, out_dir: Path) -> list[str]:
    """The runner call as it is started: under `--lock-command`, if one is given."""
    return [*shlex.split(args.lock_command), *runner_command(args, sha, run, out_dir)]


class LockCommandError(Exception):
    """The lock prefix appears not to have started the solve: stop, record nothing.

    A recorded failed row enters the resume set, so a down lock would burn the
    rest of the plan in seconds into rows no resume retries.
    """


def solution_dir(out_dir: Path, run: Run) -> Path:
    """Where the runner leaves this run's verified assignment, one directory per run.

    Per run: every arm and seed writes `<instance>.sol`, and the check must read
    this run's.
    """
    return out_dir / "runs" / f"{run.slug}.sol.d"


def read_runner_row(path: Path, run: Run, sha: str) -> dict[str, str]:
    """The single result row the runner wrote, checked against what was asked for.

    Guards a header-only file from a dead runner, and a stale file for another
    instance or commit that would be recorded under this run's arm and seed.
    """
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise RuntimeError(f"{path} holds {len(rows)} result rows, expected exactly 1")
    row = rows[0]
    if row["instance"] != run.instance:
        raise RuntimeError(f"{path} is for {row['instance']}, expected {run.instance}")
    if row["commit_sha"] != sha:
        raise RuntimeError(f"{path} was written at {row['commit_sha']}, expected {sha}")
    # `is None`, not `not in`: `csv.DictReader` pads a truncated line with None
    # values under present keys. The runner writes "NaN", never an empty cell.
    missing = [column for column in _RUNNER_COLUMNS if row.get(column) is None]
    if missing:
        raise RuntimeError(f"{path} is missing column(s) {', '.join(missing)}")
    return row


def result_row(
    run: Run, args: argparse.Namespace, sha: str, runner: dict[str, str]
) -> dict[str, str]:
    """One campaign row: what was asked for, then what came back."""
    return {**_asked(run, args, sha), **{column: runner[column] for column in _RUNNER_COLUMNS}}


def _asked(run: Run, args: argparse.Namespace, sha: str) -> dict[str, str]:
    """The provenance columns of a campaign row: what this run asked the runner for."""
    return {
        "instance": run.instance,
        "arm": run.arm.name,
        "arm_flags": " ".join(run.arm.flags),
        "seed": str(run.seed),
        "time_limit": f"{args.time_limit:g}",
        "commit_sha": sha,
    }


def failed_row(
    run: Run, args: argparse.Namespace, sha: str, returncode: int, kind: str = ""
) -> dict[str, str]:
    """A row for a run whose process exited nonzero.

    Every measured cell is "NaN" and `feasible` "false": the report's
    `csv_number` keeps NaN out of means, and the gate's `_repairs_of` reads it as
    "no reading", not zero repairs.
    """
    row = {**_asked(run, args, sha), **dict.fromkeys(_RUNNER_COLUMNS, "NaN")}
    row["feasible"] = "false"
    # The prefix tells the report to hold the row out of every count, not read
    # `feasible=false` as lost feasibility.
    row["note"] = f"{RUNNER_FAILED_NOTE}-{kind or f'exit-{returncode}'}"
    return row


#: Everything `read_runner_row` can raise on a dead process's file: a torn header
#: (`KeyError`), invalid UTF-8, a malformed `csv` field. One escaping would wedge
#: the campaign. `UnicodeDecodeError`, not its `ValueError` base, which would
#: disguise a bug in this module as `runner-failed-unreadable-row`.
UNREADABLE_ROW = (RuntimeError, OSError, KeyError, UnicodeDecodeError, csv.Error)


def _failed_run_row(
    run: Run, args: argparse.Namespace, sha: str, returncode: int, out_dir: Path
) -> dict[str, str]:
    """The row to record for a run whose process exited nonzero.

    The runner's OWN row where it wrote one (exit `RUNNER_EXIT_ERRORED`, #153):
    it carries `primal_bks`/`dual_bound`/`n_int_vars` and a note naming which of
    read, build or solve threw, and is held out of counts like `failed_row`.
    Taken only when the exit is `RUNNER_EXIT_ERRORED`, `read_runner_row` accepts
    it and no completed search wrote its note; a crash, a signal or a row
    claiming a measurement fall back to `failed_row`.
    """
    if returncode != RUNNER_EXIT_ERRORED:
        return failed_row(run, args, sha, returncode)
    try:
        runner = read_runner_row(out_dir / "runs" / f"{run.slug}.csv", run, sha)
    except UNREADABLE_ROW:
        return failed_row(run, args, sha, returncode)
    if completed_search(runner["note"]):
        # A failed process cannot have completed a search; record the failure.
        return failed_row(run, args, sha, returncode, "row-claims-a-result")
    return result_row(run, args, sha, runner)


def _progress(index: int, total: int, run: Run, started: float, elapsed_each: list[float]) -> str:
    """The stderr line the orchestrator watches for hours."""
    done = len(elapsed_each)
    line = f"[{index}/{total}] {run.instance} {run.arm.name} seed={run.seed}"
    if done:
        mean = sum(elapsed_each) / done
        remaining = mean * (total - index + 1)
        spent = time.monotonic() - started
        line += f"  (elapsed {spent / 3600:.2f}h, eta {remaining / 3600:.2f}h)"
    return line


def block_of(run: Run) -> tuple[str, int]:
    """The interleave block a run belongs to: one (instance, seed) pair.

    Its arms run back to back, so they meet the same machine state.
    """
    return (run.instance, run.seed)


def drop_partial_block(
    results: Path, runs: Sequence[Run], done: set[tuple[str, str, int]]
) -> set[tuple[str, str, int]]:
    """Un-record the trailing block a resume would otherwise finish hours late.

    A plain resume breaks the interleave: an interruption lands inside a block
    with probability (k-1)/k (80% at five arms), and finishing it later is
    invisible downstream. So the trailing partial block re-runs whole, its rows
    moved aside, not deleted. Only the LAST recorded block: an earlier partial
    one (roster filter, gate) is not something to redo.
    """
    ordered_blocks = list(dict.fromkeys(block_of(r) for r in runs))
    with_rows = {block_of(r) for r in runs if r.key in done}
    recorded_blocks = [b for b in ordered_blocks if b in with_rows]
    if not recorded_blocks:
        return done
    last = recorded_blocks[-1]
    block_runs = [r for r in runs if block_of(r) == last]
    if all(r.key in done for r in block_runs):
        return done  # complete: nothing to redo
    dropped = {r.key for r in block_runs if r.key in done}
    if not dropped:
        return done
    kept: list[list[str]] = []
    moved: list[list[str]] = []
    with results.open(newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or list(RESULT_COLUMNS)
        for row in reader:
            try:
                key = (row["instance"], row["arm"], int(row["seed"]))
            except (KeyError, TypeError, ValueError):
                kept.append([row.get(f, "") for f in fields])
                continue
            (moved if key in dropped else kept).append([row.get(f, "") for f in fields])
    # Aside written and fsynced BEFORE rewriting results: the other order can
    # lose the rows from both files. This order can only duplicate them, which
    # is recoverable by hand.
    aside = results.with_name("results.split-block.csv")
    with aside.open("a", newline="") as fh:
        out = csv.writer(fh)
        if fh.tell() == 0:
            out.writerow(fields)
        out.writerows(moved)
        fh.flush()
        os.fsync(fh.fileno())
    atomic_write(results, csv_text([fields, *kept]))
    print(
        f"resume: the interrupted block {last[0]} seed {last[1]} was {len(dropped)} of "
        f"{len(block_runs)} run(s) complete. Those {len(dropped)} row(s) were moved to "
        f"{aside.name} and the block will be re-run whole, so its arms are compared at one "
        "point in the machine's life rather than across the interruption.",
        file=sys.stderr,
        flush=True,
    )
    return done - dropped


def execute_runs(
    args: argparse.Namespace, sha: str, runs: Sequence[Run], out_dir: Path, label: str
) -> None:
    """Run every plan entry that is not already recorded, one solve at a time.

    Serial by construction (a blocking loop over `run_process`); `campaign_lock`
    stops a second driver.
    """
    results = out_dir / RESULTS_NAME
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    # Here too, not only in `execute`: a headerless results.csv would make
    # `recorded_keys` read the first run as column names and resume past it.
    open_results(results)
    done = recorded_keys(results) if args.resume else set()
    if args.resume:
        done = drop_partial_block(results, runs, done)
    started = time.monotonic()
    elapsed_each: list[float] = []
    print(
        f"=== {label}: {len(runs)} run(s), {len(done & {r.key for r in runs})} already recorded",
        file=sys.stderr,
        flush=True,
    )
    for index, run in enumerate(runs, start=1):
        if run.key in done:
            print(
                f"[{index}/{len(runs)}] {run.slug}: recorded already, skipping",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(_progress(index, len(runs), run, started, elapsed_each), file=sys.stderr, flush=True)
        began = time.monotonic()
        # Per solve, so the campaign's hours do not hold the machine lock against
        # every other session's build; nothing holds it between solves (#331).
        cmd = locked_command(args, sha, run, out_dir)
        solution_dir(out_dir, run).mkdir(parents=True, exist_ok=True)
        runner_csv = out_dir / "runs" / f"{run.slug}.csv"
        if args.lock_command:
            # A row left by an earlier attempt must not read as this solve having run.
            runner_csv.unlink(missing_ok=True)
        completed = run_process(cmd, log=out_dir / "runs" / f"{run.slug}.log")
        # The runner opens its CSV before solving, so a dead solve (signal too)
        # leaves it. No file means the prefix refused or died queued, or a
        # failure repeating on every run (a bad flag).
        if args.lock_command and completed.returncode != 0 and not runner_csv.exists():
            raise LockCommandError(
                f"{run.slug}: exit {completed.returncode} with no runner row; the lock prefix "
                f"{args.lock_command!r} probably did not start the solve. Nothing was recorded "
                f"for it; fix the cause and resume. See {out_dir / 'runs' / f'{run.slug}.log'}"
            )
        if completed.returncode != 0:
            # Recorded, not raised: else a crash under one arm is retried on
            # every resume forever. The report buckets it apart.
            row = _failed_run_row(run, args, sha, completed.returncode, out_dir)
            print(
                f"    -> FAILED (exit {completed.returncode}); recorded as a failed run "
                f"(note={row['note']}), see {out_dir / 'runs' / f'{run.slug}.log'}",
                file=sys.stderr,
                flush=True,
            )
            append_result(results, row)
            elapsed_each.append(time.monotonic() - began)
            continue
        try:
            # The DAG-independent check (#205) before the row is read: a verified
            # row SCIP rejects is recorded as the VERIFY-FAILED it is.
            independent_check.check_rows(
                out_dir / "runs" / f"{run.slug}.csv", args.inst_dir, solution_dir(out_dir, run)
            )
        except ValueError as exc:
            # Uncheckable (unrecognised solution or SCIP model): no verdict, so
            # held out as failed rather than wedging the campaign.
            print(f"    -> UNCHECKABLE row ({exc}); recorded as a failed run", file=sys.stderr)
            append_result(results, failed_row(run, args, sha, 0, "independent-check-error"))
            elapsed_each.append(time.monotonic() - began)
            continue
        except UNREADABLE_ROW as exc:
            print(f"    -> UNREADABLE row ({exc}); recorded as a failed run", file=sys.stderr)
            append_result(results, failed_row(run, args, sha, 0, "unreadable-row"))
            elapsed_each.append(time.monotonic() - began)
            continue
        try:
            runner = read_runner_row(out_dir / "runs" / f"{run.slug}.csv", run, sha)
        except UNREADABLE_ROW as exc:
            # Exit 0 but unreadable row (e.g. a full disk). Raising would wedge
            # the campaign: every resume re-drops the block and dies here.
            print(f"    -> UNREADABLE row ({exc}); recorded as a failed run", file=sys.stderr)
            append_result(results, failed_row(run, args, sha, 0, "unreadable-row"))
            elapsed_each.append(time.monotonic() - began)
            continue
        append_result(results, result_row(run, args, sha, runner))
        elapsed_each.append(time.monotonic() - began)
        print(
            f"    -> {runner['note']} feasible={runner['feasible']} "
            f"gap={runner['gap_to_bks%']} lns_repairs={runner['lns_repairs']} "
            f"lns_accepted={runner['lns_repairs_accepted']}",
            file=sys.stderr,
            flush=True,
        )


@dataclass(frozen=True)
class GateDecision:
    """Whether the LNS arm runs, and the counter reading that decided it."""

    run_arm: bool
    reason: str
    instances_with_repairs: int
    total_repairs: int
    #: Probe runs with and without an lns_repairs reading: no solve is not zero.
    probed_runs: int
    unread_runs: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "run_arm": self.run_arm,
            "reason": self.reason,
            "instances_with_repairs": self.instances_with_repairs,
            "total_repairs": self.total_repairs,
            "unread_runs": self.unread_runs,
            "probed_runs": self.probed_runs,
            "min_repairs_per_instance": LNS_GATE_MIN_REPAIRS,
            "min_instances": LNS_GATE_MIN_INSTANCES,
        }


def _repairs_of(row: dict[str, str]) -> int | None:
    """A row's `lns_repairs` as a count, or None where the row has no reading.

    None, NOT zero: the runner writes "NaN" for an unsupported instance (exit 0)
    and a solve that threw (exit 3, #153, row kept by `_failed_run_row`); as 0
    such rows would vote for SKIP without a counter reading (#143).
    """
    try:
        value = float(row["lns_repairs"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return int(value)


def decide_lns_gate(rows: Sequence[dict[str, str]]) -> GateDecision:
    """Read the probe's `lns_repairs` counters and decide whether the arm runs.

    Recorded in the output: the campaign runs the arm or writes down the reading
    that says it would measure nothing.
    """
    per_instance: dict[str, int] = {}
    unread = 0
    for row in rows:
        repairs = _repairs_of(row)
        if repairs is None:
            unread += 1
            continue
        per_instance[row["instance"]] = per_instance.get(row["instance"], 0) + repairs
    hits = sum(1 for total in per_instance.values() if total >= LNS_GATE_MIN_REPAIRS)
    total = sum(per_instance.values())
    read = len(rows) - unread
    unread_note = (
        ""
        if not unread
        else f"; {unread} probe row(s) carried no reading (no solve ran) and "
        "were excluded from the denominator rather than counted as zero repairs"
    )
    if hits >= LNS_GATE_MIN_INSTANCES:
        reason = (
            f"{hits} instance(s) reached {LNS_GATE_MIN_REPAIRS}+ LNS repair(s) "
            f"({total} repairs over {read} probe run(s) with a reading); LNS is doing work at "
            f"this budget, so --no-lns is a real arm{unread_note}"
        )
        return GateDecision(True, reason, hits, total, read, unread)
    # A skip needs a reading: below the readable fraction the arm runs.
    if read < GATE_MIN_READABLE_FRACTION * len(rows):
        reason = (
            f"only {read} of {len(rows)} probe run(s) produced an lns_repairs reading, below "
            f"the {GATE_MIN_READABLE_FRACTION:.0%} the gate needs to stand on; running the arm "
            "rather than skipping it on a reading this thin"
        )
        return GateDecision(True, reason, hits, total, read, unread)
    reason = (
        f"only {hits} instance(s) reached {LNS_GATE_MIN_REPAIRS}+ LNS repair(s) "
        f"({total} repairs over {read} probe run(s) with a reading), below the "
        f"{LNS_GATE_MIN_INSTANCES}-instance threshold; with no repair the engine takes the "
        f"same branch at every kick, so --no-lns would measure nothing{unread_note}"
    )
    return GateDecision(False, reason, hits, total, read, unread)


def probe_rows(results: Path) -> list[dict[str, str]]:
    """The gate probe's rows out of `results.csv`."""
    if not results.exists():
        return []
    with results.open(newline="") as fh:
        return [row for row in csv.DictReader(fh) if row["arm"] == PROBE_ARM.name]


def resolve_gate(
    args: argparse.Namespace, sha: str, roster: Sequence[str], out_dir: Path
) -> GateDecision:
    """Run the probe if needed and decide the gate, recording the decision."""
    if args.lns_arm != "auto":
        forced = args.lns_arm == "on"
        decision = GateDecision(
            forced,
            f"--lns-arm {args.lns_arm} was passed; the gate was not consulted. Note that "
            "issue #143 asks for the counter reading whenever the arm is skipped, which "
            "only the probe produces.",
            0,
            0,
            0,
        )
    else:
        execute_runs(args, sha, probe_plan(roster, args.seeds[0]), out_dir, "gate probe")
        decision = decide_lns_gate(probe_rows(out_dir / RESULTS_NAME))
    gate_path = out_dir / GATE_NAME
    # A forced --lns-arm never overwrites a recorded probe reading: it is the
    # counter reading that justified a skip (#143).
    if args.lns_arm != "auto" and gate_path.exists():
        print(
            f"--lns-arm {args.lns_arm}: keeping the recorded gate reading in {gate_path.name} "
            "rather than overwriting it with a forced non-decision",
            file=sys.stderr,
            flush=True,
        )
    else:
        gate_path.write_text(json.dumps(decision.as_dict(), indent=2) + "\n")
    print(
        f"LNS gate: {'RUN' if decision.run_arm else 'SKIP'} -- {decision.reason}",
        file=sys.stderr,
        flush=True,
    )
    return decision


def estimate_hours(runs: int, time_limit: float) -> float:
    """Solving hours for a run count, from the budget alone.

    A floor: every roster instance uses its whole budget (committed
    `wall_seconds` 60.0 to 60.1 across all fifty); it omits only model building,
    seconds per run.
    """
    return runs * time_limit / 3600.0


def describe_plan(
    args: argparse.Namespace,
    sha: str,
    roster: Sequence[str],
    out_dir: Path,
    problems: Sequence[str],
) -> int:
    """`--dry-run`: print the plan and the cost, touch nothing."""
    for problem in problems:
        print(f"WOULD REFUSE: {problem}")
    probe = [] if args.lns_arm != "auto" else probe_plan(roster, args.seeds[0])
    arms = [*ARMS, GATED_ARM] if args.lns_arm != "off" else list(ARMS)
    main = campaign_plan(roster, args.seeds, arms)
    print(f"roster: {len(roster)} instance(s) from {args.inst_dir / 'bounds.csv'}")
    print(f"arms: {', '.join(arm_label(arm) for arm in arms)}")
    print(f"seeds: {', '.join(str(s) for s in args.seeds)}")
    print("order: instance-major, then seed, then arm (interleaved per instance)")
    if probe:
        command = " ".join(locked_command(args, sha, probe[0], out_dir))
        print(f"gate probe: {len(probe)} run(s) -- {command}")
    print(
        f"campaign: {len(main)} run(s) -- {' '.join(locked_command(args, sha, main[0], out_dir))}"
    )
    print(
        f"estimated {estimate_hours(len(probe), args.time_limit):.1f}h probe + "
        f"{estimate_hours(len(main), args.time_limit):.1f}h campaign = "
        f"{estimate_hours(len(probe) + len(main), args.time_limit):.1f}h of solving, "
        "less whichever arms the gate drops"
    )
    return 2 if problems else 0


def execute(args: argparse.Namespace, sha: str, roster: Sequence[str], out_dir: Path) -> int:
    """The campaign: probe, gate, interleaved arms, report."""
    conflict = stamp_conflict(
        out_dir,
        campaign_stamp(sha, args.time_limit, args.seeds, [*ARMS, GATED_ARM], roster, args.lns_arm),
        resume=args.resume,
    )
    if conflict:
        print(conflict, file=sys.stderr)
        return 2
    results = out_dir / RESULTS_NAME
    if not args.resume and results.exists():
        # `--no-resume` starts over: else `execute_runs` appends to the old rows,
        # `build_cells` averages two sittings (maybe two engines) into one cell,
        # and duplicate triples inflate `k` and deflate the control's stdev,
        # shrinking the noise floor by roughly a third at three seeds. Moved
        # aside, not deleted: hours of solving.
        superseded = results.with_suffix(f".superseded-{int(time.time())}.csv")
        results.rename(superseded)
        print(f"--no-resume: moved the previous rows to {superseded}", file=sys.stderr)
    if repair_torn_tail(results):
        print(
            f"dropped a torn final line from {results} (a previous run was killed mid-append)",
            file=sys.stderr,
        )
    schema = header_conflict(results)
    if schema:
        print(schema, file=sys.stderr)
        return 2
    open_results(results)
    decision = resolve_gate(args, sha, roster, out_dir)
    arms = [*ARMS, GATED_ARM] if decision.run_arm else list(ARMS)
    execute_runs(args, sha, campaign_plan(roster, args.seeds, arms), out_dir, "campaign")
    print(render_report(results, gate=decision.as_dict()))
    return 0


def usage_error(args: argparse.Namespace) -> str | None:
    """The reason to reject the argument combination outright, or None."""
    if args.time_limit <= 0.0:
        return f"--time-limit must be > 0 (got {args.time_limit})"
    if len(args.seeds) < 3:
        return (
            f"--seeds needs at least three seeds (got {len(args.seeds)}); a single-seed A/B on "
            "this roster cannot clear its own noise floor, and the floor is measured from the "
            "control's across-seed spread, which needs at least two per arm to exist at all"
        )
    if len(set(args.seeds)) != len(args.seeds):
        return f"--seeds must be distinct (got {args.seeds}); a repeated seed is a repeated run"
    return scratch_refusal(args.out_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--out-dir", type=Path, required=True, help="scratch directory (required)")
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--instances", nargs="+", default=[], help="subset; default whole roster")
    parser.add_argument(
        "--lns-arm",
        choices=("auto", "on", "off"),
        default="auto",
        help="auto: run the gate probe and let the lns_repairs counter decide",
    )
    parser.add_argument("--allow-busy", action="store_true", help="skip the load-average refusal")
    parser.add_argument(
        "--lock-command",
        default="",
        help="prefix taking the machine lock for one solve, e.g. 'xk lock --' (default: none)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="move the existing results.csv aside and re-run every triple",
    )
    parser.add_argument(
        "--no-build", dest="build", action="store_false", help="use the runner binary as it stands"
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument(
        "--report-only", action="store_true", help="score an existing results.csv and stop"
    )
    # `common_preflight` asks whether the comparison_all.csv merge would drop
    # SCIP rows; this campaign never merges, so fixed rather than a flag.
    parser.set_defaults(merge=False)
    return parser.parse_args(argv)


def campaign_refusal(out_dir: Path) -> str | None:
    """Refuse an out-dir whose stamp names a campaign other than the ablation.

    `transfer-145` (#145) and `presolve-322` (#331) rows have other arms, so
    neither resuming nor scoring them is meaningful (their code: `ef795b55`). A
    stamp with no campaign line predates the field and is an ablation.
    """
    stamped = recorded_campaign(out_dir)
    if stamped is None or stamped == ABLATION_CAMPAIGN:
        return None
    return (
        f"{out_dir / STAMP_NAME} records campaign={stamped}, which this driver no longer runs "
        "or scores; only the ablation is left. The removed campaign's driver is in git "
        "history at ef795b55."
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    refusal = campaign_refusal(out_dir) or usage_error(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    if args.report_only:
        results = out_dir / RESULTS_NAME
        if not results.exists():
            print(f"{results} not found; nothing to report on", file=sys.stderr)
            return 2
        # `load_rows` raises on a torn final row, so repair before scoring, but
        # on a COPY: a live campaign may be mid-append, and repairing in place
        # would permanently delete a run it already struck off `done`.
        with tempfile.TemporaryDirectory() as scratch:
            snapshot = Path(scratch) / RESULTS_NAME
            snapshot.write_bytes(results.read_bytes())
            if repair_torn_tail(snapshot):
                print(
                    f"{results} ends in a torn row (a campaign is probably still appending "
                    "to it); scoring a repaired COPY and leaving the original alone",
                    file=sys.stderr,
                )
            gate_path = out_dir / GATE_NAME
            gate = json.loads(gate_path.read_text()) if gate_path.exists() else None
            print(render_report(snapshot, gate=gate))
        return 0

    sha = commit_sha()
    roster = args.instances or roster_from_bounds(args.inst_dir / "bounds.csv")
    dirty = modified_paths() if sha.endswith("-dirty") else []
    problems = common_preflight(args, sha, roster, dirty)
    busy = load_refusal(args.allow_busy)
    if busy and not args.dry_run:
        problems = [*problems, busy]
    # Refused up front: else every run fails with exit 127 and is recorded failed.
    lock_program = shlex.split(args.lock_command)[:1]
    if lock_program and shutil.which(lock_program[0]) is None:
        problems = [*problems, f"--lock-command: {lock_program[0]} is not on PATH"]
    if args.dry_run:
        return describe_plan(args, sha, roster, out_dir, problems)
    if problems:
        for problem in problems:
            print(f"refusing to run: {problem}", file=sys.stderr)
        return 2

    print(
        f"campaign: commit {sha}, {args.time_limit:g}s/instance, seeds {args.seeds}, serial",
        file=sys.stderr,
    )
    print(f"roster {len(roster)} instance(s) from {args.inst_dir / 'bounds.csv'}", file=sys.stderr)
    print(f"out-dir {out_dir}", file=sys.stderr)
    return run_locked(args, sha, roster, out_dir)


def run_locked(args: argparse.Namespace, sha: str, roster: Sequence[str], out_dir: Path) -> int:
    """Build and run the campaign under both locks, or refuse with exit 2.

    `wallclock_lock` is shared with `run_benchmark.py`, so no two timed MINLPLib
    drivers share the machine; `campaign_lock` guards this out-dir's records.
    """
    try:
        with wallclock_lock("run_ablation.py"), campaign_lock(out_dir):
            if args.build:
                subprocess.run(runner_contract.build_command(args.build_dir, 4), check=True)
            return execute(args, sha, roster, out_dir)
    except LockHeldError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2
    except LockCommandError as exc:
        print(f"campaign stopped: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
