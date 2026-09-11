"""Run the MINLPLib ablation campaign of issue #143, serially and resumably.

The campaign asks which parts of the engine the published MINLPLib result
actually depends on. It runs four arms plus a gated fifth over the 50-instance
roster at the published 60s budget, three seeds each, with the arms
**interleaved per instance** so that slow drift in machine state hits every arm
equally rather than accumulating against whichever arm ran last:

    control            (all defaults)
    no-float-hook      --no-float-hook
    unproductive-0     --unproductive-iters 0
    compound-moves     --compound-moves
    no-lns             --no-lns              (gated -- see LNS_GATE_* below)

Usage, from a configured Release build directory and a clean checkout:

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir /path/to/scratch
    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir ... --dry-run
    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir ... --report-only

The same invocation resumes: every completed run is appended to
`<out-dir>/results.csv` and flushed to disk before the next one starts, and a
restart skips exactly the `(instance, arm, seed)` triples already recorded. A
ten-hour campaign that loses everything to one interruption is not usable, and
this one is expected to be interrupted.

WHAT THIS DRIVER REFUSES, and why each refusal is a refusal rather than a
warning -- every one of them would otherwise surface as a number nobody can
trust, hours after the mistake:

* **Writing anywhere near the published tables.** `--out-dir` must be outside
  `benchmarks/instances/`, so the campaign cannot regenerate or truncate
  `comparison.csv` or `anytime_trace.csv` however it is invoked. The runner
  refuses those files independently (#136), and neither guard is load-bearing
  alone: the runner's guard would let the *control* arm through, since control
  is by definition the default configuration.
* **Sharing the machine.** An exclusive lock on `<out-dir>/campaign.lock` for
  the whole run, plus a load-average check before the first solve. The budget
  is wall-clock, so a concurrent solve produces numbers comparable neither to
  the other arms nor to anything published.
* **Resuming into a different configuration.** A stamp file records the commit,
  budget, seeds and arm set; a mismatch is refused rather than merged, because
  an ablation assembled from two engines measures the engines, not the arms.
* Everything `run_benchmark.preflight` already refuses -- a dirty tree, a
  non-Release or sanitizer build directory, a build configured from another
  checkout, a roster with missing `.nl` files.

The campaign changes no engine default and writes no published table. If an arm
argues for a different default, that is a separate change with its own
justification (issue #143's own acceptance criteria).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Executed as a script (`python3 benchmarks/minlplib/run_ablation.py`) only this
# file's own directory lands on sys.path, so the sibling module whose preflight
# guards this driver reuses would not import. Adding the repository root keeps
# that invocation working alongside `python3 -m benchmarks.minlplib.run_ablation`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.minlplib.ablation_report import (  # noqa: E402
    CONTROL_ARM,
    PROBE_ARM_NAME,
    RUNNER_FAILED_NOTE,
    completed_search,
    render_report,
)
from benchmarks.minlplib.run_benchmark import (  # noqa: E402
    DEFAULT_BUILD_DIR,
    DEFAULT_INST_DIR,
    DEFAULT_TIME_LIMIT,
    REPO_ROOT,
    RUNNER_EXIT_ERRORED,
    RUNNER_TARGET,
    commit_sha,
    preflight,
    roster_from_bounds,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: Nothing the campaign writes may land under here. `benchmarks/instances/`
#: holds every published table in the repository, and the one failure mode this
#: driver cannot recover from is replacing one of them.
PROTECTED_ROOT = REPO_ROOT / "benchmarks" / "instances"

#: The append-only record of completed runs. It is both the campaign's output
#: and its resume key, so it is never rewritten in place.
RESULTS_NAME = "results.csv"

#: Records which configuration the rows in an out-dir belong to.
STAMP_NAME = "stamp.txt"

#: Held for the whole campaign, so a second driver refuses instead of quietly
#: halving both runs' iteration counts.
LOCK_NAME = "campaign.lock"

#: Where the gate's reading and its verdict are written, so the decision is
#: recoverable from the output rather than from this session's scrollback.
GATE_NAME = "lns_gate.json"


@dataclass(frozen=True)
class Arm:
    """One ablation arm: a name for the rows and the flags that produce it."""

    name: str
    flags: tuple[str, ...]


#: The arms that always run. `control` is first and is run in this same sitting
#: rather than taken from the published table, which was measured at another
#: time on a differently-loaded machine -- exactly the drift the interleaving
#: rule exists to defeat (issue #143's protocol).
ARMS: tuple[Arm, ...] = (
    Arm(CONTROL_ARM, ()),
    Arm("no-float-hook", ("--no-float-hook",)),
    Arm("unproductive-0", ("--unproductive-iters", "0")),
    Arm("compound-moves", ("--compound-moves",)),
)

#: The gated fifth arm.
GATED_ARM = Arm("no-lns", ("--no-lns",))

#: The gate probe's rows. Same flags as `control` and the same budget, run over
#: the roster at one seed before the campaign proper, for the sole purpose of
#: reading `lns_repairs`. Its rows are recorded under this name so that the
#: scoring can tell them apart from the campaign's own control and refuse to
#: average the two -- they were measured hours apart, which is the thing the
#: protocol says makes a control unusable as a comparison baseline.
PROBE_ARM = Arm(PROBE_ARM_NAME, ())

#: Three seeds, as the protocol requires. 1 is the seed the published table was
#: measured at, so the control arm stays comparable with it; 2 and 3 are the
#: next two, chosen for being unremarkable -- any seed is as good as any other
#: to this engine, and picking memorable ones invites the suspicion that they
#: were picked after looking.
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3)

#: THE LNS GATE. Arm 4 of issue #143 runs only if LNS is doing work at this
#: budget, read off the `lns_repairs` counter the runner now publishes.
#:
#: The threshold is 1 repair in 1 instance -- that is, the arm is skipped only
#: when the control probe recorded ZERO repairs across the whole roster. That is
#: not a timid threshold, it is the exact point at which the arm stops being an
#: experiment: `src/search.cpp`'s `diversify()` tests `lns_ != nullptr` before
#: anything else and otherwise perturbs, so with no repair anywhere the
#: `--no-lns` run takes the same branch at every kick and consumes the RNG
#: identically. Zero repairs means the arm is provably a no-op; one repair means
#: the trajectories genuinely diverge on that instance.
#:
#: The gate reads ATTEMPTS, deliberately, even though `lns_repairs_accepted`
#: (#150) now publishes the acceptance half the earlier version of this comment
#: said nothing could answer. Acceptance is a stronger reading of "was LNS
#: worth its budget", but it is NOT a stronger reading of the question this
#: gate asks, which is whether `--no-lns` is a distinguishable arm. A REJECTED
#: repair is not a no-op: it randomises variables, runs `fj_nl_initialize` and
#: rolls back, so it consumes the RNG and seconds of the budget that the
#: `--no-lns` run spends on search instead. An arm with 50 attempts and zero
#: acceptances therefore still diverges -- and the divergence is precisely the
#: budget cost the arm exists to price. Skipping on acceptance would skip the
#: measurement that matters most.
#:
#: So acceptance is reported, not gated on: `ablation_report.py` prints it
#: beside the attempt total for every arm it scores. Both numbers are flags, so
#: a campaign that comes back with two repairs in 50 instances can be re-gated
#: without editing this file.
LNS_GATE_MIN_REPAIRS = 1
LNS_GATE_MIN_INSTANCES = 1

#: The share of probe runs that must actually carry an `lns_repairs` reading
#: before a SKIP decision is allowed to stand on them. Below it the arm runs:
#: the acceptance criterion asks for "the counter reading that justified
#: skipping", and a reading assembled mostly from runs that never solved is not
#: one. Costs machine time in the ambiguous case, which is the right direction
#: to fail.
GATE_MIN_READABLE_FRACTION = 0.9

#: One-minute load average above which the campaign refuses to start.
#:
#: Well BELOW one busy core, deliberately. The runner is single-threaded, so a
#: running campaign holds the one-minute average at about 1.00 and oscillates
#: either side of it: a threshold of 1.0 would let a second campaign start
#: roughly half the time it was tried, and every timing in both would then be
#: invalid with nothing in either record saying so. The lock below is per
#: out-dir, so two campaigns writing to different directories are not otherwise
#: mutually excluded and this check is the only thing between them.
#:
#: Overridable with `--allow-busy`, which is there for the case this cannot
#: distinguish -- the average still decaying from the campaign that just died.
MAX_LOAD_AVERAGE = 0.4

#: The columns of `results.csv`. `instance,arm,seed` is the resume key; the
#: provenance columns after it are what let a row be read years later without
#: this file; the rest are the runner's own row, verbatim.
RESULT_COLUMNS: tuple[str, ...] = (
    "instance",
    "arm",
    "arm_flags",
    "seed",
    "time_limit",
    "commit_sha",
    "objective",
    "primal_bks",
    "dual_bound",
    "gap_to_bks%",
    "gap_to_dual%",
    "wall_seconds",
    "feasible",
    "note",
    "max_violation",
    "n_int_vars",
    "lns_repairs",
    "lns_repairs_accepted",
    "search_config",
)

#: The runner columns copied onto a result row, in `RESULT_COLUMNS` order.
_RUNNER_COLUMNS: tuple[str, ...] = RESULT_COLUMNS[6:]


@dataclass(frozen=True)
class Run:
    """One solve: an instance, an arm and a seed."""

    instance: str
    arm: Arm
    seed: int

    @property
    def key(self) -> tuple[str, str, int]:
        """The resume key -- what `results.csv` records and a restart skips on."""
        return (self.instance, self.arm.name, self.seed)

    @property
    def slug(self) -> str:
        return f"{self.instance}__{self.arm.name}__seed{self.seed}"


def campaign_plan(roster: Sequence[str], seeds: Sequence[int], arms: Sequence[Arm]) -> list[Run]:
    """The campaign's run order: instance-major, then seed, then arm.

    This ordering IS the protocol. Every arm at every seed for one instance runs
    back to back before the next instance is touched, so the machine drift
    between the control and an arm is minutes rather than hours; and the arms
    rotate innermost, so a paired (control, arm) comparison at one seed is as
    close together in time as this campaign can put it.

    The obvious alternative -- arm-major, one whole roster pass per arm -- is
    what the issue forbids: it makes every arm's comparison against the control
    a comparison across an hour of drift, which is not distinguishable from an
    arm effect at the size of effect this roster can resolve.
    """
    return [Run(instance, arm, seed) for instance in roster for seed in seeds for arm in arms]


def probe_plan(roster: Sequence[str], seed: int) -> list[Run]:
    """The gate probe: the control configuration over the roster at one seed.

    One seed, because the gate asks whether LNS ever fires at this budget, not
    how often on average. One roster pass is ~7% of the campaign and buys a
    symmetric interleave for the arm it gates -- deciding the gate per instance
    inside the campaign loop would instead put the gated arm systematically last
    in every block.
    """
    return [Run(instance, PROBE_ARM, seed) for instance in roster]


def scratch_refusal(out_dir: Path) -> str | None:
    """Refuse an `--out-dir` from which the campaign could reach a published table.

    A path test rather than a filename test: the campaign writes several files
    and the runner writes more underneath it, so the durable rule is that the
    whole tree is off limits, not that three names are.
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

    This is the serial guarantee, asserted rather than left to the caller: the
    driver never spawns a second solve itself, and this is what stops a second
    *driver* from doing it. `flock` is per open file description, so a second
    attempt fails whether it comes from another process or from this one.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / LOCK_NAME
    # "a+" and not "w": a truncating open would wipe the holder's pid line
    # BEFORE the flock below has even failed, so the refusal could not name the
    # process it lost to -- the one diagnostic the file exists to carry,
    # destroyed by the event that needs it.
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


def campaign_stamp(
    sha: str,
    time_limit: float,
    seeds: Sequence[int],
    arms: Sequence[Arm],
    roster: Sequence[str] = (),
    lns_arm: str = "auto",
) -> str:
    """The configuration an out-dir's rows belong to, one field per line.

    The roster and the gate setting are in here for the same reason the commit
    is: resuming with `--instances a b` otherwise passes every check and quietly
    produces a partial campaign, and resuming with `--lns-arm on|off` passes too
    and then overwrites the recorded gate reading with a forced non-decision --
    destroying exactly the "counter reading that justified skipping" the
    acceptance criterion asks for.
    """
    roster_key = hashlib.sha256(",".join(roster).encode()).hexdigest()[:12] if roster else "all"
    return (
        f"commit={sha}\n"
        f"time-limit={time_limit:g}\n"
        f"seeds={','.join(str(s) for s in seeds)}\n"
        f"arms={','.join(arm.name for arm in arms)}\n"
        f"roster={len(roster)}:{roster_key}\n"
        f"lns-arm={lns_arm}\n"
    )


def stamp_conflict(out_dir: Path, stamp: str, *, resume: bool) -> str | None:
    """Refuse an out-dir written by a different engine, budget, seed set or arm set.

    Resume matches on a `(instance, arm, seed)` triple being recorded, which says
    nothing about what produced it. Without this, a campaign interrupted at one
    commit and resumed at another reports arm effects that are partly engine
    differences, and only `wall_seconds` would hint at it.
    """
    path = out_dir / STAMP_NAME
    if resume and path.exists() and path.read_text() != stamp:
        return (
            f"{path} was written by a different configuration:\n"
            f"--- recorded ---\n{path.read_text()}--- now ---\n{stamp}"
            "Use a fresh --out-dir, or pass --no-resume to start this one over (which moves "
            "the recorded rows aside rather than adding to them); mixing two configurations "
            "into one campaign measures the configurations, not the arms."
        )
    path.write_text(stamp)
    return None


def repair_torn_tail(path: Path) -> bool:
    """Drop an unterminated final line from `results.csv`. True if one was dropped.

    A campaign killed mid-append leaves a partial row. It is a row nobody can
    read and, worse, one whose triple would be missing from the resume set while
    its bytes stay in the file, so the run would be redone and the file would
    then hold a half-row wedged between two whole ones.
    """
    if not path.exists():
        return False
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return False
    cut = data.rfind(b"\n")
    # Write-then-rename, not write_bytes: this is the one function whose job is
    # protecting the campaign record against a kill, and `write_bytes`
    # truncates to zero before writing, so a kill inside that window destroys
    # up to thirteen hours of solving outright.
    _atomic_write(path, data[: cut + 1] if cut >= 0 else b"")
    return True


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace `path`'s contents without ever leaving it truncated."""
    tmp = path.with_name(path.name + ".repair.tmp")
    with tmp.open("wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def recorded_keys(path: Path) -> set[tuple[str, str, int]]:
    """The `(instance, arm, seed)` triples already in `results.csv`.

    A row with an unparseable seed is not a recorded run: it is skipped here so
    that the campaign re-runs it rather than treating a corrupt line as done.
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

    `open_results` leaves a non-empty file alone, and `append_result` writes one
    value per `RESULT_COLUMNS` entry -- so a file whose header is a DIFFERENT
    schema gains rows that are wider or narrower than the header above them, at
    exit 0. `csv.DictReader` then binds each value to the wrong name (a row
    written after #150 read under a pre-#150 header hands the repair count to
    `search_config` and drops the last cell into `restkey`), and the campaign
    scores columns that never described it.

    The campaign stamp usually catches this first, because a schema change
    coincides with a commit change and the stamp records the commit. Usually is
    not always: the stamp file can be absent -- deleted, or an out-dir from a
    driver that predates it -- and then nothing between the old header and the
    new rows. Checked here rather than inside `open_results` so the refusal
    reads like the campaign's other refusals, and so `execute_runs`' own call
    stays a create-if-missing.
    """
    if not path.exists() or not path.read_bytes():
        return None
    with path.open(newline="") as fh:
        header = next(csv.reader(fh), [])
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
    """Append one completed run, durably, before the next one starts.

    `fsync` and not just a flush: the campaign runs for hours and the thing it
    is being protected from is the machine going away, which is exactly the case
    a buffered write in the kernel's page cache does not survive.
    """
    with path.open("a", newline="") as fh:
        csv.writer(fh).writerow([row[column] for column in RESULT_COLUMNS])
        fh.flush()
        os.fsync(fh.fileno())


def runner_command(args: argparse.Namespace, sha: str, run: Run, out_dir: Path) -> list[str]:
    """The `cbls_minlplib` invocation for one run.

    `--instance` is always present, which is itself a second lock on the
    published table: the runner refuses to write `comparison.csv` from a subset
    run whatever else is passed, so even the control arm -- whose flags are all
    default -- cannot reach it.
    """
    return [
        str(args.build_dir / RUNNER_TARGET),
        str(args.inst_dir),
        "--time-limit",
        f"{args.time_limit:g}",
        "--seed",
        str(run.seed),
        "--commit",
        sha,
        "--instance",
        run.instance,
        "--out",
        str(out_dir / "runs" / f"{run.slug}.csv"),
        *run.arm.flags,
    ]


def read_runner_row(path: Path, run: Run, sha: str) -> dict[str, str]:
    """The single result row the runner wrote, checked against what was asked for.

    The checks are cheap and each of them has a real failure behind it: a runner
    that wrote the header and died leaves no row; a stale file from an earlier
    invocation carries another instance or another commit and would be recorded
    under this run's arm and seed.
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
    # `row.get(...) is None`, not `column not in row`: `csv.DictReader` pads a
    # TRUNCATED data line by binding the missing columns to None, so the key is
    # present and a torn row would pass this check and then hand None to
    # everything downstream. The runner writes "NaN", never an empty cell.
    missing = [column for column in _RUNNER_COLUMNS if row.get(column) is None]
    if missing:
        raise RuntimeError(f"{path} is missing column(s) {', '.join(missing)}")
    return row


def result_row(
    run: Run, args: argparse.Namespace, sha: str, runner: dict[str, str]
) -> dict[str, str]:
    """One campaign row: what was asked for, then what came back."""
    row = {
        "instance": run.instance,
        "arm": run.arm.name,
        "arm_flags": " ".join(run.arm.flags),
        "seed": str(run.seed),
        "time_limit": f"{args.time_limit:g}",
        "commit_sha": sha,
    }
    row.update({column: runner[column] for column in _RUNNER_COLUMNS})
    return row


def failed_row(
    run: Run, args: argparse.Namespace, sha: str, returncode: int, kind: str = ""
) -> dict[str, str]:
    """A row for a run whose process exited nonzero.

    Every measured cell is "NaN" and `feasible` is "false", so nothing here can
    be read as a result: the report's `_number` maps "NaN" to NaN and a NaN
    never reaches a mean, and the gate's `_repairs_of` reads it as "no reading"
    rather than as zero repairs.
    """
    row = {
        "instance": run.instance,
        "arm": run.arm.name,
        "arm_flags": " ".join(run.arm.flags),
        "seed": str(run.seed),
        "time_limit": f"{args.time_limit:g}",
        "commit_sha": sha,
    }
    row.update({column: "NaN" for column in _RUNNER_COLUMNS})
    row["feasible"] = "false"
    # The prefix is the contract with the report, which holds these rows out of
    # every count rather than reading `feasible=false` as a lost feasibility.
    row["note"] = f"{RUNNER_FAILED_NOTE}-{kind or f'exit-{returncode}'}"
    return row


#: Everything `read_runner_row` can raise on a file a dead process left behind.
#: Wider than `(RuntimeError, OSError)`: a torn header reaches `row["instance"]`
#: as a `KeyError`, a half-written line can be invalid UTF-8 (`ValueError`), and
#: `csv` raises its own `Error` on a malformed field. Any of them escaping would
#: wedge a thirteen-hour campaign at the one point built to survive a bad file.
UNREADABLE_ROW = (RuntimeError, OSError, KeyError, ValueError, csv.Error)


def _failed_run_row(
    run: Run, args: argparse.Namespace, sha: str, returncode: int, out_dir: Path
) -> dict[str, str]:
    """The row to record for a run whose process exited nonzero.

    The runner's OWN row where it wrote one, `failed_row` otherwise.

    The runner exits nonzero on its error tally (#153), so a solve that threw --
    which used to exit 0 and be recorded as a plain result -- now arrives here.
    Substituting `failed_row` for the row the runner already wrote would LOSE
    information: the runner looked the instance's published bounds up and put
    `primal_bks`/`dual_bound`/`n_int_vars` on the row, which this driver never
    reads, and its note says `solve-error` rather than `exit-3`. Both rows are
    held out of every count identically, so keeping the richer one costs the
    scoring nothing and tells the reader which of read, build or solve threw.

    The runner's row is taken only when the exit status is the one the runner
    uses to REPORT its error tally, the row is READABLE (`read_runner_row` checks
    it is a single row for this instance at this commit) and its note is one no
    completed search writes. A crash, a signal, or a nonzero exit next to a row
    claiming a measurement all fall back to `failed_row` -- the driver must not
    let a process failure enter the campaign as a result, and a row left behind
    by a process that died is not evidence of anything.
    """
    if returncode != RUNNER_EXIT_ERRORED:
        return failed_row(run, args, sha, returncode)
    try:
        runner = read_runner_row(out_dir / "runs" / f"{run.slug}.csv", run, sha)
    except UNREADABLE_ROW:
        return failed_row(run, args, sha, returncode)
    if completed_search(runner["note"]):
        # The process failed, but the row it left behind says a search ran and
        # reported something. The two cannot both be true; record the failure.
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

    Every arm in a block runs back to back, which is what makes the comparison
    between them a comparison at one point in the machine's life.
    """
    return (run.instance, run.seed)


def drop_partial_block(
    results: Path, runs: Sequence[Run], done: set[tuple[str, str, int]]
) -> set[tuple[str, str, int]]:
    """Un-record the trailing block a resume would otherwise finish hours late.

    THE INTERLEAVE IS THE PROTOCOL, and a plain resume breaks it. The plan runs
    every arm for one (instance, seed) back to back precisely so the control
    and the arms meet the same machine; an interruption lands inside such a
    block with probability (k-1)/k -- 80% at five arms -- and a resume that
    simply skips what is recorded then finishes that block after however long
    the campaign was down. Nothing downstream can see it: the rows look like
    every other row.

    So the trailing partial block is dropped and re-run whole. Its rows are
    moved aside rather than deleted -- `--no-resume` already keeps what it
    replaces, and a discarded solve is still a measurement someone may want to
    look at.

    Only the LAST block with any recorded run is considered: earlier blocks were
    completed in one sitting, and a block that is partial for any other reason
    (a roster filter, a gate that ran the arm later) is not something to redo.
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
    # THE ASIDE IS WRITTEN AND FSYNCED FIRST, then the results file is
    # rewritten. The other order commits the truncated file before the rows
    # exist anywhere else, so a Ctrl-C, an OOM kill or a full disk in that
    # window loses those completed solves from BOTH files -- which is deletion,
    # not the "moved aside" this function documents. Doing it this way can
    # duplicate rows if the process dies between the two steps; a duplicate is
    # recoverable by hand and a loss is not.
    aside = results.with_name("results.split-block.csv")
    with aside.open("a", newline="") as fh:
        out = csv.writer(fh)
        if fh.tell() == 0:
            out.writerow(fields)
        out.writerows(moved)
        fh.flush()
        os.fsync(fh.fileno())
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(fields)
    writer.writerows(kept)
    _atomic_write(results, buffer.getvalue().encode())
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

    Serial by construction: a plain loop over `subprocess.run`, which blocks.
    There is no concurrency switch to get wrong, and `campaign_lock` above stops
    a second driver from supplying one.
    """
    results = out_dir / RESULTS_NAME
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    # Here and not only in `execute`: `append_result` opens the file for append,
    # so a results.csv that never got its header would gain data rows with no
    # header, and the next `recorded_keys` would read the first RUN as the
    # column names and resume past it.
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
        cmd = runner_command(args, sha, run, out_dir)
        completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
        (out_dir / "runs" / f"{run.slug}.log").write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0:
            # Recorded as a failed run rather than raised. An instance that
            # crashes only under one arm, at hour nine, otherwise makes the
            # campaign unfinishable except by hand-listing the rest of the
            # roster -- and a resume would retry the same crash forever. The
            # row enters the resume set so the campaign moves on, and the
            # report counts it in its own bucket instead of averaging it in.
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
            runner = read_runner_row(out_dir / "runs" / f"{run.slug}.csv", run, sha)
        except UNREADABLE_ROW as exc:
            # The runner exited 0 but its row cannot be read -- a full disk
            # reaches this. Raising instead would wedge the campaign: every
            # resume re-drops the block and dies at the same run, on an
            # unattended thirteen-hour job.
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
    #: Probe runs that produced an lns_repairs reading, and those that did not.
    #: Held apart because a row where no solve ran is not a reading of zero.
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

    None, NOT zero. The runner writes "NaN" for an instance it skipped as
    unsupported (exit 0) and for a solve that threw (exit 3 since #153, whose
    row `_failed_run_row` still records), so such rows reach the gate looking
    like completed runs. Folding them to 0 let a
    row where nothing ran vote for SKIP -- the opposite of what the runner's own
    comment on that cell says must happen, and a skip decided partly by
    instances that never solved would not be the "counter reading that
    justified skipping" the acceptance criterion asks for.
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

    A driver decision recorded in the output, not a human one made in a shell:
    the campaign either runs the arm or writes down the reading that says it
    would have measured nothing.
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
    # A skip has to be a reading, not an absence of one. If most of the probe
    # produced no counter at all there is nothing to skip on, so the arm runs:
    # the campaign spends the time rather than publishing a decision its own
    # data does not support.
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
    # A forced --lns-arm must never overwrite a recorded reading with a
    # non-reading. The acceptance criterion asks for the counter reading that
    # justified a skip, and only the probe produces one; a `probed_runs: 0,
    # total_repairs: 0` record written over it would destroy exactly that.
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

    A floor, not a forecast: every instance on this roster uses its whole budget
    (the committed table's `wall_seconds` are 60.0 to 60.1 across all fifty), so
    the only thing this leaves out is per-process model building, which is
    seconds per run against a sixty-second solve.
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
    print(f"arms: {', '.join(arm.name for arm in arms)}")
    print(f"seeds: {', '.join(str(s) for s in args.seeds)}")
    print("order: instance-major, then seed, then arm (interleaved per instance)")
    if probe:
        command = " ".join(runner_command(args, sha, probe[0], out_dir))
        print(f"gate probe: {len(probe)} run(s) -- {command}")
    print(
        f"campaign: {len(main)} run(s) -- {' '.join(runner_command(args, sha, main[0], out_dir))}"
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
    stamp_arms = [*ARMS, GATED_ARM]
    conflict = stamp_conflict(
        out_dir,
        campaign_stamp(sha, args.time_limit, args.seeds, stamp_arms, roster, args.lns_arm),
        resume=args.resume,
    )
    if conflict:
        print(conflict, file=sys.stderr)
        return 2
    results = out_dir / RESULTS_NAME
    if not args.resume and results.exists():
        # `--no-resume` means "start this one over", and it is what the
        # stamp-conflict message above offers as the way out of a configuration
        # mismatch. Without this it does the opposite: `open_results` will not
        # truncate a file that already holds rows and `execute_runs` appends, so
        # the old rows stay and `build_cells` averages two sittings -- across
        # commits, two engines -- into one cell. Duplicating a triple also
        # inflates `k` in the noise-floor scale while deflating the control's
        # stdev, which shrinks the floor by roughly a third at three seeds and
        # promotes effects from "inside the noise" to results. Moved aside
        # rather than deleted: they are hours of solving.
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
    # `preflight` is shared with run_benchmark.py, whose `_data_problems` asks
    # whether the comparison_all.csv merge would drop the SCIP rows. This
    # campaign never merges anything, so the answer is fixed here rather than
    # exposed as a flag nobody should touch.
    parser.set_defaults(merge=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    refusal = usage_error(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    if args.report_only:
        results = out_dir / RESULTS_NAME
        if not results.exists():
            print(f"{results} not found; nothing to report on", file=sys.stderr)
            return 2
        # A torn final row is the normal state of a campaign that was killed
        # mid-append, and `load_rows` raises on one rather than skipping it the
        # way `recorded_keys` does. So it has to be repaired before scoring --
        # but NOT IN PLACE. This is the way the README invites a reader to watch
        # a running campaign from a second terminal, and the row that looks torn
        # from here is very often one the running driver has already fsynced and
        # already struck off its in-memory `done` set: repairing the live file
        # would delete that run from the record permanently. Score a copy.
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
    problems = preflight(args, sha, roster)
    busy = load_refusal(args.allow_busy)
    if busy and not args.dry_run:
        problems = [*problems, busy]
    if args.dry_run:
        return describe_plan(args, sha, roster, out_dir, problems)
    if problems:
        for problem in problems:
            print(f"refusing to run: {problem}", file=sys.stderr)
        return 2

    print(
        f"commit {sha}, {args.time_limit:g}s/instance, seeds {args.seeds}, serial", file=sys.stderr
    )
    print(f"roster {len(roster)} instance(s) from {args.inst_dir / 'bounds.csv'}", file=sys.stderr)
    print(f"out-dir {out_dir}", file=sys.stderr)
    with campaign_lock(out_dir):
        if args.build:
            subprocess.run(
                ["cmake", "--build", str(args.build_dir), "--target", RUNNER_TARGET, "-j", "4"],
                check=True,
            )
        return execute(args, sha, roster, out_dir)


if __name__ == "__main__":
    sys.exit(main())
