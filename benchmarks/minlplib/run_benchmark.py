"""Re-run the MINLPLib CBLS rows at the current engine HEAD, in one command.

Whenever the engine moves, `comparison.csv`'s `commit_sha` column names an
engine that no longer exists; this regenerates the table against a current
build (#123) in one invocation:

    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py

which rebuilds the runner, solves the 50-instance roster serially at 60s each
(~50 minutes of solving), rewrites `comparison.csv` and `anytime_trace.csv`,
re-merges the **CBLS rows** of `comparison_all.csv`, and prints a summary. See
`benchmarks/instances/minlplib/README.md` ("Re-running the CBLS rows") for the
surrounding procedure and the post-run steps.

SEEDS (#141). `comparison.csv` is ONE pre-registered seed, `DEFAULT_SEED`, the
only seed that may write it or `anytime_trace.csv`, so the published seed is
fixed before the run, not chosen after it. Every whole-roster run onto the
default paths, that seed included, also publishes its rows into
`comparison_seeds.csv` beside it, seed on every row, replacing that seed's
earlier rows and keeping the others'. Another seed's own table and trace are
assembled inside its staging directory. The summary prints the spread across
the per-seed table's comparable seeds (`campaign_report.summarize_seeds`, the
single-table tally's aggregation rule). Each published set gets a run record
(commit, budget, seed, machine, concurrency) beside it (`comparison.run.json`,
`comparison_seeds.run.json`).

Why one instance at a time rather than one whole-roster command:

* **Resumable.** `cbls_minlplib` truncates its output CSV on open and writes
  rows as it goes, so a crash 40 minutes in leaves the published table
  half-replaced and the work lost. Here each instance lands in its own staging
  file, outside the checkout (`default_staging_root`), and a re-run skips the
  ones already complete.
* **Non-destructive.** `comparison.csv` and `anytime_trace.csv` are only touched
  at the end, by an atomic rename of a fully-assembled file, so a failed solve
  leaves the previous tables byte-for-byte intact. Except `comparison_all.csv`:
  the merge rewrites it in place, so a crash *there* can leave it half-written;
  a re-run repairs it, reusing the staged rows and going straight to the merge.
* **Faithful.** The runner seeds each solve from `--seed` directly
  (`cbls::solve(model, time_limit, seed, ...)`), so a per-instance process sees
  exactly the state a whole-roster process would. The budget is wall-clock
  either way, the dominant source of run-to-run spread; see the README's
  "Results".

Guards, because this file's output is published:

* checks every verified row OUTSIDE the engine's DAG before publishing it, in
  SCIP (`verify_staged`, #205); needs the `benchmarks` extra (pyscipopt);
* refuses a dirty working tree, its own published files excepted
  (`run_commit_sha`): a plain SHA from a modified checkout claims a
  reproducibility the numbers do not have;
* refuses a build directory that is not `Release`, or one configured from a
  different source tree than the SHA is read from;
* rebuilds the runner target itself, so the binary cannot lag the SHA it is
  about to be labelled with;
* refuses a subset run (`--instances`) without scratch output *and staging*
  paths, so a debug run can neither truncate a fifty-row table nor leave
  short-budget rows for a later resume to publish;
* refuses any whole-roster run that would write exactly one of the two published
  artifacts, which would pair a table and a trace from different engines (#149);
* refuses to resume a staging directory written by a different commit, budget,
  seed, thread count, host or roster, and re-solves any individual staged row
  whose `commit_sha` or thread count disagrees;
* sets aside (`.rejected`) the staged row of any solve whose runner exited
  nonzero, so a resume re-solves it (`quarantine_staged`, #170);
* leaves an instance with two recorded failed attempts alone on a resume, since
  a third that succeeds would turn a final no-solution into solved depending on
  how often someone re-runs (`final_no_solutions`, #323), and then assembles no
  table, which would lack its row; `--retry-final NAME` asks for another
  attempt, `--instances NAME` does not, since a campaign runs one name per call;
* marks each row the independent check has passed (or demoted) with
  `<instance>.checked`, its CSV's sha256, so a scorer can tell it from one a
  killed driver left (#307);
* never mixes thread counts in one table (`--threads`, default 1; campaigns run
  at `CAMPAIGN_THREADS`): each row's `search_config` and each run record carry
  the count, any other count may not publish, a per-seed table holding another
  count's seeds is refused, as is a count above the CPUs this process may use;
* refuses a seed other than `DEFAULT_SEED` writing either published artifact,
  by default or by name, and any scratch output (or its run record) resolving
  to a published file (#141);
* holds a machine-wide lock for the whole run, so two invocations never share
  the machine or race on the per-seed table.

Usage:
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --dry-run
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --seed 2   # into comparison_seeds.csv
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --instances nvs01 \
        --time-limit 3 --out /tmp/c.csv --trace-out /tmp/t.csv --staging-dir /tmp/stage
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Run as a script, only this file's directory is on sys.path; the repository root
# makes the shared modules importable from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.jobs import (  # noqa: E402
    LockHeldError,
    allowed_cpus,
    run_process,
    wallclock_lock,
)
from benchmarks.common.provenance import (  # noqa: E402
    REPO_ROOT,
    build_dir_problems,
    cmake_cache,
    commit_sha,
    machine_record,
    modified_paths,
)
from benchmarks.common.records import (  # noqa: E402
    append_attempt,
    atomic_write,
    attempts_exhausted,
    csv_header,
    stamp_refusal,
    write_json,
)
from benchmarks.minlplib import campaign_report, independent_check, runner  # noqa: E402
from benchmarks.minlplib.campaign_report import (  # noqa: E402
    RUN_RECORD_SCHEMA,
    SEED_COLUMN,
    SEEDS_TABLE_NAME,
    RunRecord,
    run_record_path,
)
from benchmarks.minlplib.runner import (  # noqa: E402
    RUNNER_EXIT_ERRORED,
    RUNNER_TARGET,
    stageable_note,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_INST_DIR = REPO_ROOT / "benchmarks" / "instances" / "minlplib"
#: Every published table and roster in the repository lives under here. A scratch
#: output must land outside it, and the only roster directory under it this
#: driver publishes into is `DEFAULT_INST_DIR` (#144: `heldout/` publishes nothing).
INSTANCES_ROOT = REPO_ROOT / "benchmarks" / "instances"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
REFERENCE_SOLVE = Path(__file__).resolve().parent / "reference_solve.py"

#: Seconds per instance. The runner's own documented default (issue #88), argued
#: from the committed anytime trace in the benchmark README ("Why 60s").
DEFAULT_TIME_LIMIT = 60.0

#: Seed of the published run: the PRE-REGISTERED seed (#141), so a re-run differs
#: from the old table only in the engine. The only seed allowed to write
#: `comparison.csv` / `anytime_trace.csv`; every seed, this one too, publishes
#: into the per-seed table (`campaign_report.SEEDS_TABLE_NAME`) beside it.
DEFAULT_SEED = 1

#: Processes at a time, recorded beside `--threads` (each solve's portfolio
#: size): a wall-clock-budgeted number means nothing without both.
PARALLEL_SOLVES = 1

#: `--threads` by default: the single-threaded `solve()` the committed tables
#: were measured with.
DEFAULT_THREADS = 1

#: The thread count every campaign runs cbls at: Simon's decision of 2026-11-01,
#: recorded on https://github.com/spoorendonk/cbls/issues/255 for the full
#: campaign, https://github.com/spoorendonk/cbls/issues/256. A table holds one
#: thread count: any other than this or `DEFAULT_THREADS` may not publish
#: (`threads_policy_error`), and `cbls_minlplib` refuses it the same way.
CAMPAIGN_THREADS = 16

#: Where a seed's assembled table and trace land when it is not the published
#: seed: inside its own staging directory, beside the per-instance files.
ASSEMBLED_TABLE = "comparison.assembled.csv"
ASSEMBLED_TRACE = "anytime_trace.assembled.csv"

#: Parallel jobs for the *build* only. The solves are always serial: the budget
#: is wall-clock, so a concurrent solve is not comparable to the committed table
#: or to the other rows of its own run.
DEFAULT_BUILD_JOBS = 4

#: Records which configuration a staging directory's rows belong to.
STAMP_NAME = "stamp.txt"


def default_staging_root() -> Path:
    """The parent of every seed's default staging directory, outside the checkout.

    `$XDG_STATE_HOME/cbls/minlplib-rerun` (`~/.local/state/...` when unset). Not
    under the build directory, which pre-push's ```clean fence (`rm -rf build`)
    deletes (#141), nor gitignored in the checkout, which the worktree workflow
    deletes after a merge. XDG *state*, not cache: state should survive a
    restart, cache may be deleted at any time. Two checkouts sharing it is safe:
    the stamp refuses to resume rows from another commit, budget, seed, thread
    count or host. Read at call time, so a test can point it elsewhere.
    """
    state = os.environ.get("XDG_STATE_HOME", "")
    # The XDG spec ignores a relative value; so does this.
    base = Path(state) if state and Path(state).is_absolute() else Path.home() / ".local" / "state"
    return base / "cbls" / "minlplib-rerun"


@dataclass(frozen=True)
class Paths:
    """Where this invocation reads the roster from and writes its results to."""

    published_out: Path
    out: Path
    trace_out: Path
    stage: Path
    #: The per-seed table this run's rows are published into, or None for a
    #: scratch or subset run, which publishes nothing.
    seeds_out: Path | None


def resolve_paths(args: argparse.Namespace, sha: str) -> Paths:
    """Every path, with the seed policy applied (#141).

    The pre-registered seed defaults to the published `comparison.csv` and
    `anytime_trace.csv`; any other seed defaults to its own staging directory,
    so leaving `--out` off can never publish a second seed into the published
    table. A whole-roster run onto the default paths, at any seed, also
    publishes into the per-seed table.

    The default staging directory is keyed by commit, budget and thread count as
    well as seed: it outlives every campaign, so keyed by seed alone the first
    run after a new commit would hit the stamp's refusal instead of starting fresh.
    """
    published_out = args.inst_dir / "comparison.csv"
    stage = args.staging_dir or (
        default_staging_root()
        / sha
        / f"{args.time_limit:g}s"
        / f"threads{args.threads}"
        / f"seed{args.seed}"
    )
    preregistered = args.seed == DEFAULT_SEED
    out = args.out or (published_out if preregistered else stage / ASSEMBLED_TABLE)
    trace_out = args.trace_out or (
        args.inst_dir / "anytime_trace.csv" if preregistered else stage / ASSEMBLED_TRACE
    )
    publishing = not args.instances and (
        args.out is None or args.out.resolve() == published_out.resolve()
    )
    return Paths(
        published_out=published_out,
        out=out,
        trace_out=trace_out,
        stage=stage,
        seeds_out=args.inst_dir / SEEDS_TABLE_NAME if publishing else None,
    )


def roster_from_bounds(bounds_csv: Path) -> list[str]:
    """Instance names in `bounds.csv` order — the roster of record.

    The runner's source and order without `--instance`, so an assembled table is
    row-for-row comparable with a whole-roster run's. An absent file yields an
    empty roster, which preflight refuses naming `download.py`.
    """
    if not bounds_csv.exists():
        return []
    with bounds_csv.open(newline="") as fh:
        return [row["instance"] for row in csv.DictReader(fh)]


def _build_problems(
    args: argparse.Namespace, sha: str, dirty_paths: Sequence[str] = ()
) -> list[str]:
    """Refusals about the binary this run would measure and the SHA it labels it with.

    `dirty_paths` names the files that made `sha` dirty, for the refusal to show.
    """
    problems: list[str] = []
    if sha.endswith("-dirty"):
        which = f": {', '.join(dirty_paths)}" if dirty_paths else ""
        problems.append(
            f"working tree is dirty ({sha}){which}; commit or stash first — a row labelled "
            "with a plain SHA must have been produced by that commit's code"
        )
    cache = cmake_cache(args.build_dir)
    problems += build_dir_problems(args.build_dir, cache)
    if not cache:
        return problems
    binary = args.build_dir / RUNNER_TARGET
    if not args.build and not binary.exists():
        problems.append(
            f"{binary} not found and --no-build was given; drop --no-build or build the "
            f"{RUNNER_TARGET} target first"
        )
    return problems


def _data_problems(args: argparse.Namespace, roster: Sequence[str]) -> list[str]:
    """Refusals about the roster and the files the run and the merge read."""
    problems: list[str] = []
    if not roster:
        problems.append(
            f"{args.inst_dir / 'bounds.csv'} is missing or has no rows; fetch the roster with "
            f"`{sys.executable} {args.inst_dir / 'download.py'}`"
        )
    missing = [name for name in roster if not (args.inst_dir / f"{name}.nl").exists()]
    if missing:
        shown = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        problems.append(
            f"{len(missing)} roster instance(s) have no .nl file ({shown}); fetch them with "
            f"`{sys.executable} {args.inst_dir / 'download.py'}`"
        )
    if args.merge and not (args.inst_dir / "scip_baseline.csv").exists():
        problems.append(
            f"{args.inst_dir / 'scip_baseline.csv'} not found; the merge rebuilds "
            "comparison_all.csv from it and would drop the SCIP rows. Pass --no-merge to "
            "regenerate comparison.csv only."
        )
    return problems


#: The per-seed table's header: the seed, then exactly the runner's columns.
SEEDS_TABLE_COLUMNS: tuple[str, ...] = (SEED_COLUMN, *runner.RUNNER_COLUMNS)


def _seeds_table_problems(args: argparse.Namespace) -> list[str]:
    """Refusals about a per-seed table this run would add its rows to.

    Up front, since `publish_seed_rows` refuses the same things AFTER the solves:
    a foreign header or an unparsable run record would cost fifty minutes to find.
    """
    seeds_out = resolve_paths(args, "").seeds_out
    if seeds_out is None:
        return []
    problems: list[str] = []
    if seeds_out.exists():
        header = csv_header(seeds_out)
        if tuple(header) != SEEDS_TABLE_COLUMNS:
            problems.append(
                f"{seeds_out} has columns {header}, not {list(SEEDS_TABLE_COLUMNS)}: its rows "
                "are from another runner version, so this run's seed could not be added beside "
                "them. Move it aside (the old campaign stays in the repository's history) and "
                "re-run"
            )
        else:
            problems += _seeds_row_problems(seeds_out)
    try:
        records = campaign_report.load_seed_run_records(seeds_out)
    except ValueError as exc:
        problems.append(str(exc))
    else:
        problems += _seeds_threads_problems(seeds_out, records, args)
    return problems


def _seeds_threads_problems(
    seeds_out: Path, records: dict[int, RunRecord], args: argparse.Namespace
) -> list[str]:
    """Refuse adding this run's seed beside seeds measured at another thread count.

    `publish_seed_rows` keeps every other seed's block, so a 16-thread seed beside
    1-thread ones would mix thread counts. This seed's own record is about to be
    replaced, so it does not count.
    """
    others = {
        seed: record.concurrency.get("threads_per_solve")
        for seed, record in records.items()
        if seed != args.seed
    }
    mixed = sorted(seed for seed, threads in others.items() if threads != args.threads)
    if not mixed:
        return []
    shown = ", ".join(f"seed {seed} at {others[seed]}" for seed in mixed)
    return [
        f"{seeds_out} holds {shown} thread(s), and this run is at --threads {args.threads}: "
        "one table holds one thread count. Publish into another --inst-dir, or move the table "
        "aside (the old campaign stays in the repository's history)"
    ]


#: A seed cell `int()` reads back exactly: ASCII digits only, so `--1` or `²` is
#: refused at preflight rather than after the solves.
SEED_CELL = re.compile(r"-?[0-9]+")


def _seeds_row_problems(seeds_out: Path) -> list[str]:
    """Rows `publish_seed_rows` would choke on after the solves: a short row or a bad seed."""
    with seeds_out.open(newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for line, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(SEEDS_TABLE_COLUMNS) or not SEED_CELL.fullmatch(row[0]):
                return [
                    f"{seeds_out} line {line} is not a per-seed row (an integer seed and "
                    f"{len(runner.RUNNER_COLUMNS)} runner cells); repair or move the table aside"
                ]
    return []


def common_preflight(
    args: argparse.Namespace, sha: str, roster: Sequence[str], dirty_paths: Sequence[str] = ()
) -> list[str]:
    """The refusals shared with `run_ablation.py`: the binary, the SHA and the roster.

    Apart from `preflight`: the ablation namespace has no `--out`/`--seed`/
    `--staging-dir`, which the per-seed-table check needs.
    """
    return _build_problems(args, sha, dirty_paths) + _data_problems(args, roster)


def preflight(
    args: argparse.Namespace, sha: str, roster: Sequence[str], dirty_paths: Sequence[str] = ()
) -> list[str]:
    """Every reason to refuse this invocation, checked before anything is spent.

    All cheap; each would otherwise surface as a wrong or half-written published
    table, some 50 minutes in.
    """
    return (
        common_preflight(args, sha, roster, dirty_paths)
        + _seeds_table_problems(args)
        + _cpu_problems(args)
        + _retry_final_problems(args, roster)
    )


def _retry_final_problems(args: argparse.Namespace, roster: Sequence[str]) -> list[str]:
    """Refuse a `--retry-final` name this run would not reach: it would quietly retry nothing."""
    unreached = sorted(set(args.retry_final) - set(roster))
    return [f"--retry-final names outside this run: {', '.join(unreached)}"] if unreached else []


def available_cpus() -> int:
    """How many CPUs this process may run on: its affinity mask on Linux, every CPU elsewhere."""
    return len(allowed_cpus())


def _cpu_problems(args: argparse.Namespace) -> list[str]:
    """Refuse more portfolio threads than CPUs this process may run on.

    Under `taskset -c 0-3` (the local-CI pin) or a unit with a smaller CPU set,
    sixteen workers time-slice on four cores: each gets a quarter of the budget,
    and every row still says `threads=16`.
    """
    cpus = available_cpus()
    if args.threads <= cpus:
        return []
    return [
        f"--threads {args.threads} but this process may run on {cpus} CPU(s): the workers "
        "would share them and each row would measure a fraction of its budget. Widen the "
        "CPU set (taskset, the unit's AllowedCPUs) or lower --threads"
    ]


def usage_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """The reason to reject the argument combination outright, or None."""
    if args.time_limit <= 0.0:
        return f"--time-limit must be > 0 (got {args.time_limit})"
    if args.build_jobs < 1:
        return f"--build-jobs must be >= 1 (got {args.build_jobs})"
    if args.threads < 1:
        return f"--threads must be >= 1 (got {args.threads})"
    seed_refusal = (
        roster_dir_error(args, published_out)
        or seed_policy_error(args, published_out)
        or budget_policy_error(args, published_out)
        or threads_policy_error(args, published_out)
    )
    if seed_refusal:
        return seed_refusal
    if args.instances:
        return _subset_error(args, published_out)
    return _whole_roster_error(args, published_out)


def _whole_roster_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """`usage_error` for a whole-roster run: never publish one of the two artifacts alone."""
    if args.seed != DEFAULT_SEED:
        # Neither published artifact is in reach: its table and trace default
        # into its staging directory, its rows into the per-seed table.
        return None
    # Publishing comparison.csv without the published trace (--no-trace, or a
    # scratch --trace-out) leaves anytime_trace.csv at the previous engine, and
    # the README's post-run step recomputes its budget table from that trace.
    if (not args.trace or args.trace_out is not None) and (
        args.out is None or args.out.resolve() == published_out.resolve()
    ):
        published_trace = args.inst_dir / "anytime_trace.csv"
        how = "--no-trace" if not args.trace else f"--trace-out {args.trace_out}"
        return (
            f"{how} on a whole-roster run would publish comparison.csv at this engine "
            f"while leaving {published_trace} at the previous one; drop it, or "
            "pass --out to write somewhere other than the published table"
        )
    # The converse (#149): `--out` moved off the published table with
    # `--trace-out` defaulted to the published `anytime_trace.csv`. `publish`
    # assembles traces into `trace_out` whatever `out` is, and the runner's
    # guard never sees the published path (each instance is staged), so a
    # scratch run would replace the published profile at exit 0.
    if (
        args.trace
        and args.trace_out is None
        and args.out is not None
        and args.out.resolve() != published_out.resolve()
    ):
        return (
            f"--out writes {args.out} but --trace-out is unset, so the trace would replace "
            f"the published {args.inst_dir / 'anytime_trace.csv'}; pass --trace-out, or "
            "--no-trace"
        )
    return None


def _subset_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """`usage_error` for a subset run, which must name scratch paths throughout.

    A subset rewrites the same whole files a full run does, and its rows would be
    resumed by the next full run.
    """
    if args.out is None:
        return (
            "--instances is a subset run; pass an explicit --out so it cannot replace the "
            f"published {published_out}"
        )
    if args.out.resolve() == published_out.resolve():
        return (
            f"--instances is a subset run and --out resolves to the published {published_out}; "
            "it would truncate the fifty-row table to the subset"
        )
    if args.trace and args.trace_out is None:
        return "--instances with tracing on requires an explicit --trace-out (or --no-trace)"
    if args.staging_dir is None:
        return (
            "--instances is a subset run; pass an explicit --staging-dir so its rows cannot be "
            "resumed into a later whole-roster run"
        )
    return None


#: Every table a publish owns. Each one's run record (`run_record_path`) is owned too.
PUBLISHED_NAMES: tuple[str, ...] = (
    "comparison.csv",
    "anytime_trace.csv",
    "comparison_all.csv",
    SEEDS_TABLE_NAME,
)


#: The instance directory's inputs -- and the report generator's committed
#: summary -- a scratch output must not land on them either. `analysis_notes.csv`
#: is read by the runner itself, so overwriting it changes every later row's note.
PROTECTED_INPUTS: tuple[str, ...] = (
    "bounds.csv",
    "scip_baseline.csv",
    "analysis_notes.csv",
    campaign_report.SUMMARY_JSON_NAME,
)


def published_paths(inst_dir: Path) -> tuple[Path, ...]:
    """Each published table in `inst_dir` and its run-record path.

    A superset of what a publish writes: `anytime_trace.run.json` and
    `comparison_all.run.json` are never written but are the driver's own names,
    so the dirty guard and the protected-output check treat them as output too.
    """
    tables = [inst_dir / name for name in PUBLISHED_NAMES]
    return tuple(f for t in tables for f in (t, run_record_path(t)))


def _published_files(inst_dir: Path) -> dict[Path, str]:
    owned = {f.resolve(): f.name for f in published_paths(inst_dir)}
    owned.update({(inst_dir / name).resolve(): name for name in PROTECTED_INPUTS})
    return owned


def run_commit_sha(inst_dir: Path, repo_root: Path = REPO_ROOT) -> str:
    """The SHA this run's rows are labelled with: `commit_sha`, blind to this driver's own output.

    The dirty guard is about the code that ran. A multi-seed campaign runs one
    seed per invocation, and seed 1's publish modifies tracked tables; counted,
    seed 2 would refuse that tree, and committing between seeds would split the
    campaign across two SHAs, which the seed summary drops (#123). Only
    `published_paths(inst_dir)` is ignored: a modified instance-directory input
    (`analysis_notes.csv` changes every later row's note), README or source file
    still marks the tree dirty.
    """
    return commit_sha(repo_root, ignore=published_paths(inst_dir))


def run_dirty_paths(inst_dir: Path, repo_root: Path = REPO_ROOT) -> list[str]:
    """The modified tracked files that make `run_commit_sha` dirty, for the refusal to name."""
    return modified_paths(repo_root, ignore=published_paths(inst_dir))


def _misdirected_output(args: argparse.Namespace, published_out: Path) -> str | None:
    """A scratch `--out`/`--trace-out` -- or `--out`'s run record -- landing on a published file.

    `seed_policy_error` handles the two names that ARE the published pair; this is
    every other spelling: the trace over `comparison.csv`, a table over the
    per-seed table (whose record would then replace every seed's), a `--out` whose
    stem makes its run record `comparison.run.json`, and so on.
    """
    published_trace = args.inst_dir / "anytime_trace.csv"
    protected = _published_files(args.inst_dir)
    written: list[tuple[str, Path]] = []
    if args.out is not None and args.out.resolve() != published_out.resolve():
        written += [("--out", args.out), ("--out's run record", run_record_path(args.out))]
    if (
        args.trace
        and args.trace_out is not None
        and args.trace_out.resolve() != published_trace.resolve()
    ):
        written.append(("--trace-out", args.trace_out))
        if args.out is not None and (
            args.trace_out.resolve() == run_record_path(args.out).resolve()
        ):
            return "--trace-out resolves to --out's run record, which the publish writes"
    for flag, path in written:
        if path.resolve() in protected:
            return (
                f"{flag} resolves to the published {protected[path.resolve()]}; a scratch "
                "output must not overwrite a published file or an instance-directory input"
            )
        # Another roster directory's tables are as published as this one's: with
        # --inst-dir heldout/, a scratch --out must not reach ../comparison.csv.
        if path.resolve().is_relative_to(INSTANCES_ROOT.resolve()):
            return (
                f"{flag} {path} is under {INSTANCES_ROOT}, which holds every published table "
                "and roster; a scratch output goes outside it"
            )
    return None


def roster_dir_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """Refuse publishing into a roster directory under `INSTANCES_ROOT` other than the default.

    `heldout/` (#144) is a roster no result is published against: HELDOUT.md
    requires every output to go outside `benchmarks/instances/`. A default-path
    run there would create `heldout/comparison.csv` and its per-seed table.
    """
    inst = args.inst_dir.resolve()
    if inst == DEFAULT_INST_DIR.resolve() or not inst.is_relative_to(INSTANCES_ROOT.resolve()):
        return None
    if (
        args.out is None
        or args.out.resolve() == published_out.resolve()
        or args.staging_dir is None
    ):
        return (
            f"--inst-dir {args.inst_dir} is not the published roster directory; pass scratch "
            "--out, --trace-out (or --no-trace) and --staging-dir outside "
            f"{INSTANCES_ROOT}"
        )
    return None


def budget_policy_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """Refuse a non-default budget writing any published table (#141's review).

    A whole-roster run onto the default paths publishes (`comparison.csv` at seed
    1, its seed's block of `comparison_seeds.csv` at any seed), so a five-second
    smoke run there would replace that seed's 60s block. As for the runners' "a
    non-default arm may never write a published table", a budget other than
    `DEFAULT_TIME_LIMIT` must name a scratch `--out`.
    """
    if args.time_limit == DEFAULT_TIME_LIMIT or args.instances:
        return None
    if args.out is None or args.out.resolve() == published_out.resolve():
        return (
            f"--time-limit {args.time_limit:g} is not the published protocol's "
            f"{DEFAULT_TIME_LIMIT:g}s, and this run would publish into "
            f"{published_out.name if args.seed == DEFAULT_SEED else SEEDS_TABLE_NAME}; pass a "
            "scratch --out (and --trace-out) for a run at another budget"
        )
    return None


def threads_policy_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """Refuse a thread count that is neither 1 nor `CAMPAIGN_THREADS` writing a published table.

    The runner's own guard never fires here, since every instance is staged. 1
    is the committed tables' count and 16 the campaign protocol's (#255, #256);
    any other count must name a scratch `--out`, as another budget must.
    """
    if args.threads in (DEFAULT_THREADS, CAMPAIGN_THREADS) or args.instances:
        return None
    if args.out is None or args.out.resolve() == published_out.resolve():
        return (
            f"--threads {args.threads} is neither the committed tables' {DEFAULT_THREADS} nor "
            f"the campaign protocol's {CAMPAIGN_THREADS} (#255, #256), and this run would "
            f"publish into "
            f"{published_out.name if args.seed == DEFAULT_SEED else SEEDS_TABLE_NAME}; pass a "
            "scratch --out (and --trace-out) for a run at another thread count"
        )
    return None


def seed_policy_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """Refuse any route by which a seed other than `DEFAULT_SEED` reaches a published file.

    The published table is ONE pre-registered seed (#141), an A/B against the
    previous table where only the engine differs; a second seed written there
    would make the published seed whichever ran last, a post-hoc selection. Also
    refused at seed 1: a `--trace-out` naming the published trace while `--out`
    is scratch, the explicit form of `usage_error`'s defaulted-trace guard.
    """
    misdirected = _misdirected_output(args, published_out)
    if misdirected:
        return misdirected
    published_trace = args.inst_dir / "anytime_trace.csv"
    out_published = args.out is not None and args.out.resolve() == published_out.resolve()
    trace_published = (
        args.trace
        and args.trace_out is not None
        and args.trace_out.resolve() == published_trace.resolve()
    )
    if args.seed != DEFAULT_SEED and (out_published or trace_published):
        return (
            f"--seed {args.seed}: {published_out.name} and {published_trace.name} hold the "
            f"pre-registered seed {DEFAULT_SEED} alone, so no other seed may write them. Drop "
            f"--out/--trace-out: a whole-roster run at seed {args.seed} publishes its rows into "
            f"{SEEDS_TABLE_NAME} beside them"
        )
    if trace_published and not (args.out is None or out_published):
        return (
            f"--trace-out names the published {published_trace} but --out writes {args.out}; "
            "the published trace would describe a run the published table does not"
        )
    return None


def staging_stamp(args: argparse.Namespace, sha: str) -> str:
    """The configuration a staging directory's rows belong to, one field per line.

    Host and roster (#141): a home directory can be shared between machines, and
    a campaign resumed on another would publish one table measured on two.
    Threads (#299): resumed at another `--threads`, one table would mix two
    portfolio sizes.
    """
    return (
        f"commit={sha}\ntime-limit={args.time_limit:g}\nseed={args.seed}\n"
        f"threads={args.threads}\n"
        + f"host={socket.gethostname()}\nroster={roster_fingerprint(args.inst_dir)}\n"
    )


def roster_fingerprint(inst_dir: Path) -> str:
    """A short hash of `bounds.csv`, so a staging dir is resumed only for its own roster.

    The default staging root is shared by every checkout and every `--inst-dir`;
    without this, a run over another roster directory at the same commit, budget
    and seed would resume rows staged for this one.
    """
    bounds = inst_dir / "bounds.csv"
    return hashlib.sha256(bounds.read_bytes()).hexdigest()[:16] if bounds.exists() else "none"


def staging_stamp_conflict(stage: Path, args: argparse.Namespace, sha: str) -> str | None:
    """Refuse a staging directory written by a different engine, budget or seed.

    Resume matches on a staged file being *complete*, not on what produced it:
    a run resumed at another commit, or a five-second smoke run over the roster,
    would publish one table from two configurations that only `wall_seconds` betrays.
    """
    stamp_path = stage / STAMP_NAME
    stamp = staging_stamp(args, sha)
    earlier = stamp_path.read_text() if stamp_path.exists() else None
    refused = args.resume and earlier is not None and earlier != stamp
    if not refused and earlier != stamp:
        # An attempt under another configuration is no retry of this one (#312).
        # Cleared before `stamp_refusal` writes the new stamp: a kill in between
        # leaves the old stamp, so the next run clears them again.
        for record in stage.glob(f"*{FAILURES_SUFFIX}"):
            record.unlink()
    return stamp_refusal(
        stamp_path,
        stamp,
        resume=args.resume,
        label="staged",
        advice="Delete the staging directory, pass a fresh --staging-dir, or pass --no-resume; "
        "reusing these rows would mix two configurations into one table.",
    )


def staged_row_complete(path: Path, sha: str) -> bool:
    """True when a staging CSV holds a header and a whole result row for `sha`.

    Three ways a staged file can look done without being usable:

    * the runner writes the header before it solves, so a killed job leaves a
      header-only file;
    * a job killed mid-write leaves a torn last line, still a line: hence the
      trailing-newline and field-count checks;
    * a row staged at an earlier commit would mix engines in one table (#123
      asks for one SHA in the column).
    """
    row = _staged_row(path)
    return row is not None and path.read_text().endswith("\n") and row.get("commit_sha") == sha


def staged_row_whole(path: Path) -> bool:
    """`staged_row_complete` without the commit test: a header and one whole, ended row."""
    return _staged_row(path) is not None and path.read_text().endswith("\n")


def staged_unpublishable(path: Path) -> bool:
    """Whether a staging CSV's row must NOT stand in for a fresh solve.

    Apart from `staged_row_complete`, which asks whether the FILE is whole: such
    a row is whole, and would otherwise stand in for a solve on the next resume.
    True for a row the runner wrote after throwing, and for any note
    `runner.STAGEABLE_NOTES` does not recognise, an absent `note` column included.
    An unreadable file is left to `staged_row_complete` (incomplete, not
    unpublishable).
    """
    row = _staged_row(path)
    return row is not None and not stageable_note(row.get("note", ""))


def _staged_row(path: Path) -> dict[str, str] | None:
    """A staging CSV's first row by column, or None when it has no whole one."""
    if not path.exists():
        return None
    rows = list(csv.reader(path.read_text().splitlines()))
    if len(rows) < 2 or len(rows[1]) != len(rows[0]):
        return None
    return dict(zip(rows[0], rows[1], strict=True))


def staged_complete(args: argparse.Namespace, sha: str, name: str, stage: Path) -> bool:
    """Whether `name`'s staged output can stand in for a fresh solve.

    Both files matter: a `--no-trace` run leaves a complete CSV and no trace, and
    skipping on the CSV alone would replace `comparison.csv` and only then fail
    to assemble the trace.

    A row the runner wrote after THROWING is refused (#153), as is any row whose
    note `runner.STAGEABLE_NOTES` does not recognise: structurally complete, it
    would be skipped on the default `--resume` and `publish` would assemble a row
    that measured nothing. A coverage gap (`unsupported`, or a missing `.nl`)
    exits 0 and is NOT refused; it is a documented row like any other.
    """
    if not staged_row_complete(stage / f"{name}.csv", sha):
        return False
    if staged_unpublishable(stage / f"{name}.csv"):
        return False
    if staged_row_threads(stage / f"{name}.csv") != args.threads:
        return False
    return not args.trace or (stage / f"{name}.trace.csv").exists()


#: The thread count `cbls_minlplib` appends to `search_config` off the
#: single-threaded default; a cell without it is a 1-thread row.
SEARCH_CONFIG_THREADS = re.compile(r"(?:^|;)threads=([0-9]+)(?:;|$)")


def staged_row_threads(path: Path) -> int | None:
    """The thread count a staged row was solved at, or None when it has no whole row.

    The stamp keys a directory to one count, but `--no-resume` rewrites it and
    leaves earlier rows in place, so an interrupted re-run at another count could
    resume onto them. The row's own `search_config` is what it was measured at.
    """
    row = _staged_row(path)
    if row is None:
        return None
    match = SEARCH_CONFIG_THREADS.search(row.get("search_config", ""))
    return int(match.group(1)) if match else 1


def runner_command(args: argparse.Namespace, sha: str, name: str, stage: Path) -> list[str]:
    """The `cbls_minlplib` invocation for one instance, staged under `stage`."""
    return runner.runner_command(
        args.build_dir,
        args.inst_dir,
        time_limit=args.time_limit,
        seed=args.seed,
        sha=sha,
        instance=name,
        out=stage / f"{name}.csv",
        solution_dir=stage,
        threads=args.threads,
        extra=(["--trace", str(stage / f"{name}.trace.csv")] if args.trace else []),
    )


def build_command(args: argparse.Namespace) -> list[str]:
    return runner.build_command(args.build_dir, args.build_jobs)


def verify_staged(args: argparse.Namespace, roster: Sequence[str], stage: Path) -> list[str]:
    """Check every staged verified row in SCIP, outside the DAG; demote what fails (#205).

    Over the WHOLE roster after `run_roster`, resumed rows included: a job killed
    between the runner's exit and the check would otherwise leave an unchecked
    row the next resume publishes. Idempotent (a demoted row is `feasible=false`
    and not looked at again), so a re-check costs one SCIP read.

    Each `.checked` marker is written after the check and any demotion, so its
    hash is of the row as it will be published.
    """
    demoted: list[str] = []
    for name in roster:
        demoted += check_and_mark(args.inst_dir, stage, name)
    for name in demoted:
        print(
            f"{name}: the independent SCIP check rejected the runner's verified row; "
            "it is published as VERIFY-FAILED",
            file=sys.stderr,
        )
    return demoted


def merge_command(inst_dir: Path) -> list[str]:
    """Rebuild `comparison_all.csv` from the CSVs on disk, solving nothing.

    `--merge-only` re-reads `scip_baseline.csv` and `bounds.csv` unchanged and
    takes only the CBLS rows from `comparison.csv`, keeping the `published-bks`
    and `scip` rows out of this run's reach.
    """
    return [sys.executable, str(REFERENCE_SOLVE), "--merge-only", "--inst-dir", str(inst_dir)]


def assemble(stage: Path, roster: Sequence[str], out: Path, suffix: str) -> None:
    """Concatenate the per-instance staging files into `out`, in roster order.

    Written to a sibling temporary file and renamed into place, so the published
    table is replaced atomically or not at all.
    """
    header: str | None = None
    body: list[str] = []
    for name in roster:
        path = stage / f"{name}{suffix}"
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"{path} is empty")
        if header is None:
            header = lines[0]
        elif lines[0] != header:
            raise RuntimeError(f"{path} header differs from {roster[0]}{suffix}:\n{lines[0]}")
        body.extend(lines[1:])
    if header is None:
        raise RuntimeError("nothing to assemble")
    atomic_write(out, "\n".join([header, *body]) + "\n")


def run_record(
    args: argparse.Namespace,
    sha: str,
    roster: Sequence[str],
    machine: dict[str, object],
    resumed: int = 0,
) -> RunRecord:
    """What produced this run's results: written beside every table it publishes (#141)."""
    return RunRecord(
        commit=sha,
        budget_seconds=args.time_limit,
        seed=args.seed,
        roster=len(roster),
        resumed=resumed,
        published_at=datetime.now(UTC).isoformat(timespec="seconds"),
        # Pinned to the table's bytes when it is written (`write_run_record`,
        # `publish_seed_rows`); nothing is published with this placeholder.
        table_sha256="",
        machine=machine,
        concurrency={
            "parallel_solves": PARALLEL_SOLVES,
            "threads_per_solve": args.threads,
            "build_jobs": args.build_jobs,
        },
    )


def write_run_record(table: Path, record: RunRecord) -> None:
    """The single-seed table's record, `<table stem>.run.json` beside it, pinned to its bytes."""
    pinned = replace(record, table_sha256=campaign_report.file_sha256(table))
    write_json(run_record_path(table), {"schema": RUN_RECORD_SCHEMA, **asdict(pinned)})


def publish_seed_rows(seeds_out: Path, seed: int, table: Path) -> str:
    """Replace `seed`'s rows in the per-seed table with `table`'s, keeping every other seed.

    Upsert, not append, so seeds published one invocation at a time accumulate.
    Rows are ordered by seed, each block in `table`'s (roster) order. Written
    atomically. Returns the block's `campaign_report.seed_block_sha256`, for its
    run record.
    """
    with table.open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        new_rows = [[str(seed), *row] for row in reader if row]
    if (SEED_COLUMN, *header) != SEEDS_TABLE_COLUMNS:
        raise RuntimeError(f"{table} has columns {header}, not {list(runner.RUNNER_COLUMNS)}")
    kept: list[list[str]] = []
    if seeds_out.exists():
        with seeds_out.open(newline="") as fh:
            reader = csv.reader(fh)
            existing = next(reader, [])
            if tuple(existing) != SEEDS_TABLE_COLUMNS:
                raise RuntimeError(
                    f"{seeds_out} has columns {existing}, not {list(SEEDS_TABLE_COLUMNS)}"
                )
            # Compared as integers: a hand-written "01" is seed 1's block too.
            kept = [row for row in reader if row and int(row[0]) != seed]
    rows = sorted([*kept, *new_rows], key=lambda row: int(row[0]))
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows([list(SEEDS_TABLE_COLUMNS), *rows])
    atomic_write(seeds_out, buffer.getvalue())
    return campaign_report.seed_block_sha256(new_rows)


def publish_seed_record(seeds_out: Path, record: RunRecord) -> None:
    """Add or replace `record.seed`'s entry in the per-seed table's run record."""
    records = campaign_report.load_seed_run_records(seeds_out)
    records[record.seed] = record
    write_json(
        run_record_path(seeds_out),
        {
            "schema": RUN_RECORD_SCHEMA,
            "seeds": {str(seed): asdict(records[seed]) for seed in sorted(records)},
        },
    )


def summarize(out: Path, bounds_csv: Path) -> str:
    """The written table's tally, under `campaign_report`'s aggregate definitions.

    No definitions of its own, so this and the README's tally share one
    denominator (#142): `campaign_report.summarize_results` applies
    `campaign_report.AGGREGATION_RULE` (documented failures count in the roster
    counts, out of the quality aggregates), printed beside the numbers. The full
    report, anytime scores and SCIP head-to-head included, is `campaign_report.py`.
    """
    rows = campaign_report.load_results(out, campaign_report.load_bounds_index(bounds_csv))
    summary = campaign_report.summarize_results(rows)
    counts, verdicts = summary.counts, summary.verdicts
    lines = [
        f"rule: {summary.rule}",
        "roster counts (every row):",
        f"  roster:               {counts.roster}",
        f"  built:                {counts.built}",
        f"  feasible:             {counts.feasible}",
        f"  infeasible:           {counts.infeasible}",
        f"  coverage gaps:        {counts.coverage_gaps}",
        f"  errors:               {counts.errors}",
        f"  non-finite:           {counts.non_finite}",
        f"  integrality mismatch: {counts.integrality_mismatches}",
        f"  verification failed:  {counts.verification_failures}",
        f"quality aggregates ({verdicts.denominator} feasible claim-set rows):",
        f"  matches-bks:          {verdicts.matches_bks}",
        f"  within-tolerance:     {verdicts.within_tolerance}",
        f"  worse than BKS:       {verdicts.worse}",
        f"  better than BKS:      {verdicts.better}",
        f"  no published bound:   {verdicts.no_bks}",
    ]
    if verdicts.unclassified:
        lines.append(f"  unclassified:         {', '.join(verdicts.unclassified)}")
    if counts.error_instances:
        lines.append(f"errors (no search completed): {', '.join(counts.error_instances)}")
    by_name = {r.instance: r for r in rows}
    lines += [
        f"excluded from quality aggregates: {name} -> {by_name[name].verdict}"
        for name in counts.documented_failures
    ]
    lines += [
        f"WARNING: documented failure {name} came back FEASIBLE -- check #110/#116 "
        "before publishing anything about it"
        for name in counts.documented_failures_feasible
    ]
    return "\n".join(lines)


def print_summary(out: Path, bounds_csv: Path) -> None:
    """Print `summarize`, or say why it could not be derived -- never raise.

    It runs after the table is written, so a refusal in the summary (an unknown
    instance, a duplicate row) must not read as a failed publish.
    """
    try:
        print(summarize(out, bounds_csv))
    except (ValueError, KeyError, OSError) as exc:
        print(
            f"WARNING: {out} is written, but its summary could not be derived: {exc}. "
            "Run benchmarks/minlplib/campaign_report.py over it to see the full report.",
            file=sys.stderr,
        )


def print_seeds_summary(
    seeds_out: Path, bounds_csv: Path, *, commit: str, budget: float, host: str | None
) -> None:
    """Print the multi-seed summary of the per-seed table -- never raise (see `print_summary`)."""
    try:
        summary = campaign_report.summarize_seeds(
            campaign_report.load_seed_results(
                seeds_out, campaign_report.load_bounds_index(bounds_csv)
            ),
            campaign_report.load_seed_run_records(seeds_out),
            commit=commit,
            budget=budget,
            block_hashes=campaign_report.seed_block_hashes(seeds_out),
            host=host,
        )
        print(campaign_report.render_seeds_text(summary))
    except (ValueError, KeyError, OSError) as exc:
        print(
            f"WARNING: {seeds_out} is written, but its multi-seed summary could not be derived: "
            f"{exc}. Run benchmarks/minlplib/campaign_report.py --seeds over it.",
            file=sys.stderr,
        )


def final_no_solutions(
    args: argparse.Namespace, sha: str, roster: Sequence[str], stage: Path
) -> list[str]:
    """The roster instances the #256 rule has settled, which a resume leaves alone (#323).

    Each has no row to resume from, a `.rejected` row and two recorded failed
    attempts (what `gm_ratio.py` scores as no-solution), and is not named by
    `--retry-final`. `--no-resume` does not override it, or with `--instances` it
    would give one instance a third attempt under its record's stamp. Read after
    `staging_stamp_conflict`: the attempts a new stamp clears do not count.
    """
    return [
        name
        for name in roster
        if name not in args.retry_final
        and not staged_complete(args, sha, name, stage)
        and (stage / f"{name}.csv{REJECTED_SUFFIX}").exists()
        and attempts_exhausted(stage / f"{name}{FAILURES_SUFFIX}")
    ]


def planned_final_no_solutions(
    args: argparse.Namespace, sha: str, roster: Sequence[str], stage: Path
) -> list[str]:
    """`final_no_solutions` as `--dry-run` sees it: another stamp clears or refuses the records."""
    stamp = stage / STAMP_NAME
    if not stamp.is_file() or stamp.read_text() != staging_stamp(args, sha):
        return []
    return final_no_solutions(args, sha, roster, stage)


def discard_staged(stage: Path, name: str) -> None:
    """Delete `name`'s staged row, trace and `.checked` marker; the `.rejected` ones stay."""
    checked_marker(stage, name).unlink(missing_ok=True)
    for suffix in (".csv", ".trace.csv"):
        (stage / f"{name}{suffix}").unlink(missing_ok=True)


def run_roster(args: argparse.Namespace, sha: str, roster: Sequence[str], stage: Path) -> int:
    """Solve every roster instance serially, skipping the ones already staged.

    Returns how many were skipped: their rows were solved by an EARLIER
    invocation, whose machine and load this one's run record did not see.
    """
    resumed = 0
    for index, name in enumerate(roster, start=1):
        if args.resume and staged_complete(args, sha, name, stage):
            print(f"[{index}/{len(roster)}] {name}: staged already, skipping")
            resumed += 1
            continue
        cmd = runner_command(args, sha, name, stage)
        print(f"[{index}/{len(roster)}] {name}: {' '.join(cmd)}", flush=True)
        # An old row and its check say nothing about this attempt: left here, a
        # runner dying before it opens its CSV would have it counted as this
        # attempt's (#312).
        discard_staged(stage, name)
        log = stage / f"{name}.log"
        completed = run_process(cmd, log=log)
        if completed.returncode != 0 or not staged_complete(args, sha, name, stage):
            # Exit 0 without a publishable row is a failed solve too, as it is
            # on the SCIP side: left in place it could never be scored.
            quarantine_staged(stage, name, completed.returncode)
            if attempts_exhausted(stage / f"{name}{FAILURES_SUFFIX}"):
                what_next = (
                    f"Its failed attempts make it a final cbls no-solution (#312): a re-run "
                    f"leaves it alone, unless --retry-final {name} asks for another attempt."
                )
            elif completed.returncode == RUNNER_EXIT_ERRORED:
                what_next = (
                    f"The runner threw on this instance; its row, set aside as {name}.csv"
                    f"{REJECTED_SUFFIX}, carries read-error/build-error/solve-error: no search "
                    "ran, so publishing the table would put a row in it that measures nothing. A "
                    "re-run solves it again rather than resuming past it: one more failed "
                    "attempt, resuming at this commit, makes it a final cbls no-solution (#312), "
                    "or fix the instance."
                )
            else:
                what_next = "Re-running resumes from here."
            raise RuntimeError(
                f"{name} failed (exit {completed.returncode}); see {log}. {what_next}"
            )
        # A solved instance has no failures to count; a later failure starts over.
        (stage / f"{name}{FAILURES_SUFFIX}").unlink(missing_ok=True)
    return resumed


#: The suffix a staged file gets when its solve exited nonzero; `staged_complete`
#: then finds no row, so a re-run solves the instance again, unless its attempts
#: have made it final (`final_no_solutions`).
REJECTED_SUFFIX = ".rejected"

#: The record of `name`'s failed attempts, one `exit <rc> <utc>` line each, which
#: `gm_ratio.py` counts: the `.rejected` row is overwritten by the next failure (#312).
FAILURES_SUFFIX = ".failures"

#: The suffix of the marker `verify_staged` writes beside a staged row it has
#: checked: `<instance>.checked`, holding the row file's sha256 (#307).
CHECKED_SUFFIX = ".checked"


def checked_marker(stage: Path, name: str) -> Path:
    return stage / f"{name}{CHECKED_SUFFIX}"


class RowChangedError(RuntimeError):
    """A staged row's bytes changed while `check_and_mark` was checking them."""


def check_and_mark(inst_dir: Path, stage: Path, name: str) -> list[str]:
    """Check `name`'s staged row in SCIP, demoting it if SCIP rejects it, then mark it.

    The marker is written after the check and any demotion, so it hashes the row
    as it will be published; a check that raises writes none. Returns the demoted
    instances, as `independent_check.check_rows` does.

    Unless the check demoted (rewrote) the row, its hash before and after must
    agree, else `RowChangedError` and no marker: the marker would vouch for bytes
    nobody checked. The wall-clock lock keeps this host's drivers out; this is
    for a writer the lock cannot see.
    """
    row = stage / f"{name}.csv"
    before = campaign_report.file_sha256(row)
    demoted = independent_check.check_rows(row, inst_dir, stage)
    after = campaign_report.file_sha256(row)
    if not demoted and after != before:
        raise RowChangedError(f"{row} changed while it was checked; check it again")
    mark_checked(stage, name, after)
    return demoted


def mark_checked(stage: Path, name: str, digest: str | None = None) -> None:
    """Write `name`'s `.checked` marker for `digest`, by default the row file's sha256 now."""
    if digest is None:
        digest = campaign_report.file_sha256(stage / f"{name}.csv")
    atomic_write(checked_marker(stage, name), digest + "\n")


def checked_digest(stage: Path, name: str) -> str | None:
    """The sha256 `name`'s `.checked` marker vouches for; None when it has none."""
    marker = checked_marker(stage, name)
    return marker.read_text().strip() if marker.is_file() else None


def staged_checked(stage: Path, name: str) -> bool:
    """Whether `name`'s staged row is the one `check_and_mark` checked, byte for byte.

    The hash counts, not the marker's presence: a re-solve rewrites the row, and
    `run_roster` also deletes the marker before one, so a marker never vouches
    for an unseen row.
    """
    row = stage / f"{name}.csv"
    return row.is_file() and checked_digest(stage, name) == campaign_report.file_sha256(row)


def quarantine_staged(stage: Path, name: str, returncode: int) -> None:
    """Set aside `name`'s staged row and trace after a nonzero runner exit, and record it.

    A whole row with a stageable note can still measure the wrong thing: a solve
    that lost portfolio workers (#170) says so only in its exit status, and the
    next resume would publish it as a full-thread row. Renamed, not deleted, so
    the evidence stays beside the log.

    The attempt is recorded last: a driver killed in between leaves one attempt
    too few, which a final score refuses, never one too many. Recorded only when
    the runner staged a row: one exiting before it opens its CSV (a usage error,
    an unwritable path, an unreadable roster) solved nothing, and counting it
    would make an earlier attempt's `.rejected` row read as retried.
    """
    staged_row = (stage / f"{name}.csv").exists()
    for suffix in (".csv", ".trace.csv"):
        path = stage / f"{name}{suffix}"
        if path.exists():
            path.replace(path.with_name(path.name + REJECTED_SUFFIX))
    if staged_row:
        append_attempt(stage / f"{name}{FAILURES_SUFFIX}", returncode)


def publish(
    args: argparse.Namespace, roster: Sequence[str], paths: Paths, record: RunRecord
) -> int:
    """Assemble the staged rows into the tables, record the run, publish the seed, re-merge.

    The run record goes beside `paths.out` whatever it is: a scratch table is a
    results set too, readable later only with its machine.
    """
    assemble(paths.stage, roster, paths.out, ".csv")
    print(f"wrote {paths.out}")
    if args.trace:
        assemble(paths.stage, roster, paths.trace_out, ".trace.csv")
        print(f"wrote {paths.trace_out}")
    write_run_record(paths.out, record)
    print(f"wrote {run_record_path(paths.out)}")
    if paths.seeds_out is not None:
        block = publish_seed_rows(paths.seeds_out, args.seed, paths.out)
        publish_seed_record(paths.seeds_out, replace(record, table_sha256=block))
        print(f"wrote seed {args.seed} into {paths.seeds_out}")
    merge_failed = False
    if args.merge:
        # check=False: `--merge-only` exits 2 on its own refusals, and a traceback
        # would leave comparison.csv rewritten with no summary or next step.
        merged = subprocess.run(merge_command(args.inst_dir), check=False)
        merge_failed = merged.returncode != 0
        if merge_failed:
            print(
                f"the comparison_all.csv merge failed (exit {merged.returncode}); "
                f"{paths.out} is written and the staged rows are kept, so a re-run goes "
                "straight back to the merge. comparison_all.csv still holds the PREVIOUS "
                "cbls rows until it succeeds.",
                file=sys.stderr,
            )
    print("\n=== Summary (derived from the written table) ===")
    print_summary(paths.out, args.inst_dir / "bounds.csv")
    if paths.seeds_out is not None:
        print(f"\n=== Across seeds (derived from {paths.seeds_out.name}) ===")
        print_seeds_summary(
            paths.seeds_out,
            args.inst_dir / "bounds.csv",
            commit=record.commit,
            budget=record.budget_seconds,
            host=socket.gethostname(),
        )
    return 1 if merge_failed else 0


def execute(args: argparse.Namespace, sha: str, roster: Sequence[str], paths: Paths) -> int:
    """Build, solve the roster, and publish.

    The machine is recorded BEFORE the build and the solves: its load average is
    the "was the box quiet" evidence, and read afterwards it would measure this
    run's own build.
    """
    machine = machine_record()
    with wallclock_lock("run_benchmark.py"):
        if args.build:
            subprocess.run(build_command(args), check=True)
        paths.stage.mkdir(parents=True, exist_ok=True)
        conflict = staging_stamp_conflict(paths.stage, args, sha)
        if conflict:
            print(conflict, file=sys.stderr)
            return 2
        final = final_no_solutions(args, sha, roster, paths.stage)
        for name in final:
            # What a killed further attempt left (no whole row, or `final_no_solutions`
            # would not hold) measures nothing, and the scorer would refuse it as unchecked.
            discard_staged(paths.stage, name)
            print(
                f"{name}: final cbls no-solution (two failed attempts), not solved again; "
                f"--retry-final {name} asks for another attempt"
            )
        solving = [name for name in roster if name not in final]
        resumed = run_roster(args, sha, solving, paths.stage)
        verify_staged(args, solving, paths.stage)
        if final:
            print(
                f"{len(final)} final no-solution(s) have no row, so no table was assembled: "
                f"{', '.join(final)}. Score the staging directory with gm_ratio.py "
                "--cbls-staging.",
                file=sys.stderr,
            )
            return 1
        if resumed:
            print(
                f"note: {resumed} of {len(roster)} row(s) were staged by an earlier invocation; "
                "the run record's machine and load are this invocation's, not theirs",
                file=sys.stderr,
            )
        return publish(args, roster, paths, run_record(args, sha, roster, machine, resumed))


def describe_plan(
    args: argparse.Namespace,
    sha: str,
    roster: Sequence[str],
    paths: Paths,
    problems: Sequence[str],
) -> int:
    """`--dry-run`: print exactly what would happen, touch nothing."""
    for problem in problems:
        print(f"WOULD REFUSE: {problem}")
    if args.build:
        print(f"build: {' '.join(build_command(args))}")
    final = planned_final_no_solutions(args, sha, roster, paths.stage)
    solving = [name for name in roster if name not in final]
    if solving:
        print(
            f"solve (x{len(solving)}, serial): "
            f"{' '.join(runner_command(args, sha, solving[0], paths.stage))}"
        )
    if final:
        print(f"WOULD ASSEMBLE NO TABLE: final no-solution(s), not solved: {', '.join(final)}")
    else:
        if args.merge:
            print(f"merge: {' '.join(merge_command(args.inst_dir))}")
        print(f"run record: {run_record_path(paths.out)}")
        if paths.seeds_out is not None:
            print(f"publish: seed {args.seed}'s rows into {paths.seeds_out}")
    print(
        f"estimated {len(solving) * args.time_limit / 60.0:.0f} min of solving on a quiet machine"
    )
    # Non-zero on a refusal or a run that would publish nothing, the exits the
    # run itself gives, so --dry-run is usable as a scriptable precheck.
    if problems:
        return 2
    return 1 if final else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--threads",
        type=int,
        default=DEFAULT_THREADS,
        help=f"portfolio threads per solve; {CAMPAIGN_THREADS} is the campaign protocol "
        "(#255, #256), and one table holds one thread count",
    )
    parser.add_argument("--build-jobs", type=int, default=DEFAULT_BUILD_JOBS)
    # nargs="+", not "*": a bare `--instances` would otherwise be an empty list,
    # slip past the subset guard, and run the roster into the published paths.
    parser.add_argument("--instances", nargs="+", default=[], help="subset; default whole roster")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"default <inst-dir>/comparison.csv at seed {DEFAULT_SEED}, "
        f"<staging-dir>/{ASSEMBLED_TABLE} at any other seed",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help=f"default <inst-dir>/anytime_trace.csv at seed {DEFAULT_SEED}, "
        f"<staging-dir>/{ASSEMBLED_TRACE} at any other seed",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "default $XDG_STATE_HOME/cbls/minlplib-rerun/<commit>/<budget>s/threads<T>/seed<N> "
            "(~/.local/state/...)"
        ),
    )
    parser.add_argument(
        "--no-trace", dest="trace", action="store_false", help="skip the anytime trace"
    )
    parser.add_argument(
        "--no-merge",
        dest="merge",
        action="store_false",
        help="write comparison.csv only; leave comparison_all.csv alone",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="re-solve every instance even if a staged result exists, final no-solutions "
        "excepted (--retry-final, or a new stamp)",
    )
    parser.add_argument(
        "--retry-final",
        nargs="+",
        default=[],
        metavar="NAME",
        help="solve these again though two failed attempts made them final no-solutions",
    )
    parser.add_argument(
        "--no-build",
        dest="build",
        action="store_false",
        help="use the runner binary as it stands (it may not match --commit)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="print the commands and the plan, run nothing"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    refusal = usage_error(args, args.inst_dir / "comparison.csv")
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    sha = run_commit_sha(args.inst_dir, REPO_ROOT)
    paths = resolve_paths(args, sha)
    # The merge rebuilds the published comparison_all.csv from the published
    # comparison.csv: destructive when this run's rows went elsewhere. Resolved,
    # so a relative --out naming the published file is not taken for scratch.
    skipping_merge = args.merge and paths.out.resolve() != paths.published_out.resolve()
    args.merge = args.merge and not skipping_merge

    roster = args.instances or roster_from_bounds(args.inst_dir / "bounds.csv")
    dirty = run_dirty_paths(args.inst_dir, REPO_ROOT) if sha.endswith("-dirty") else []
    problems = preflight(args, sha, roster, dirty)
    if problems and not args.dry_run:
        for problem in problems:
            print(f"refusing to run: {problem}", file=sys.stderr)
        return 2

    print(
        f"commit {sha}, {args.time_limit:g}s/instance, seed {args.seed}, "
        f"{args.threads} thread(s), serial"
    )
    print(f"roster {len(roster)} instance(s) from {args.inst_dir / 'bounds.csv'}")
    print(f"staging {paths.stage}")
    print(f"out {paths.out}")
    if args.trace:
        print(f"trace {paths.trace_out}")
    if paths.seeds_out is not None:
        print(f"per-seed table {paths.seeds_out}")
    if skipping_merge:
        why = (
            f"--out is not {paths.published_out}"
            if args.out is not None
            else f"seed {args.seed} is not the published seed {DEFAULT_SEED}"
        )
        print(
            f"note: {why}; skipping the comparison_all.csv merge and leaving the published "
            "tables alone"
        )
    if args.dry_run:
        return describe_plan(args, sha, roster, paths, problems)
    try:
        return execute(args, sha, roster, paths)
    except LockHeldError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
