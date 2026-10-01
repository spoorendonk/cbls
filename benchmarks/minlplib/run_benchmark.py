"""Re-run the MINLPLib CBLS rows at the current engine HEAD, in one command.

This is the driver for issue #123: `comparison.csv`'s `commit_sha` column names
an engine that no longer exists, and the table has to be regenerated against a
current build. The whole procedure is one invocation:

    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py

which rebuilds the runner, solves the 50-instance roster serially at 60s each
(~50 minutes of solving), rewrites `comparison.csv` and `anytime_trace.csv`,
re-merges the **CBLS rows** of `comparison_all.csv`, and prints a summary. See
`benchmarks/instances/minlplib/README.md` ("Re-running the CBLS rows") for the
surrounding procedure and the post-run steps.

SEEDS (#141). `comparison.csv` is ONE pre-registered seed, `DEFAULT_SEED`, and
only that seed may write it or `anytime_trace.csv`: the published seed is fixed
before the run by construction, not chosen after it. Every whole-roster run
onto the default paths -- that seed and any other -- also publishes its rows
into `comparison_seeds.csv` beside it, with the seed on every row, replacing
that seed's earlier rows and keeping the others'. Another seed's own table and
trace are assembled inside its staging directory. The summary then prints the
spread across the per-seed table's comparable seeds (`campaign_report.
summarize_seeds`, under the same aggregation rule as the single-table tally).
Each published set gets a run record -- commit, budget, seed, machine,
concurrency -- beside it (`comparison.run.json`, `comparison_seeds.run.json`).

Why it drives the runner one instance at a time rather than issuing the single
whole-roster command the README used to document:

* **Resumable.** `cbls_minlplib` truncates its output CSV on open and writes
  rows as it goes, so a crash 40 minutes in leaves the published table
  half-replaced and the work lost. Here each instance lands in its own staging
  file, outside the checkout (`default_staging_root`, so pre-push's
  `rm -rf build` cannot take it), and a re-run skips the ones already complete.
* **Non-destructive.** `comparison.csv` and `anytime_trace.csv` are only touched
  at the end, by an atomic rename of a fully-assembled file, so a failed solve
  leaves the previous tables byte-for-byte intact. `comparison_all.csv` is the
  exception: the merge step rewrites it in place, so a crash *there* can leave
  it half-written — re-run to repair it, which reuses the staged rows and goes
  straight back to the merge.
* **Faithful.** The runner seeds each instance's solve from `--seed` directly
  (`cbls::solve(model, time_limit, seed, ...)`), so a per-instance process sees
  exactly the state a whole-roster process would. The budget is wall-clock in
  either case, which is the dominant source of run-to-run spread; see the
  README's "These are single-sample numbers".

Guards, because this file's output is published:

* refuses a dirty working tree — a plain SHA from a modified checkout claims a
  reproducibility the numbers do not have;
* refuses a build directory that is not `Release`, or one configured from a
  different source tree than the SHA is read from;
* rebuilds the runner target itself, so the binary cannot lag the SHA it is
  about to be labelled with;
* refuses a subset run (`--instances`) that has not been given scratch output
  *and staging* paths, so a debug run can neither truncate a fifty-row table nor
  leave short-budget rows for a later run's resume to publish;
* refuses any whole-roster run that would write exactly one of the two published
  artifacts, in either direction — `--out` moved off `comparison.csv` with
  `--trace-out` left at its default would replace the published anytime trace at
  exit 0 while reporting a scratch table, and the converse publishes
  `comparison.csv` at this engine beside a trace from the previous one (#149);
* refuses to resume a staging directory written by a different commit, budget,
  seed or host, and re-solves any individual staged row whose `commit_sha`
  disagrees;
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
import contextlib
import csv
import fcntl
import hashlib
import io
import os
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

# Run as a script (`python3 benchmarks/minlplib/run_benchmark.py`, the documented
# form), only this file's own directory lands on sys.path. The repository root is
# what makes the shared modules importable from any working directory.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.jobs import run_process  # noqa: E402
from benchmarks.common.provenance import (  # noqa: E402
    REPO_ROOT,
    build_dir_problems,
    cmake_cache,
    commit_sha,
    machine_record,
)
from benchmarks.common.records import (  # noqa: E402
    atomic_write,
    csv_header,
    stamp_refusal,
    write_json,
)
from benchmarks.minlplib import campaign_report, runner  # noqa: E402
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
    from collections.abc import Iterator, Sequence

DEFAULT_INST_DIR = REPO_ROOT / "benchmarks" / "instances" / "minlplib"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
REFERENCE_SOLVE = Path(__file__).resolve().parent / "reference_solve.py"

#: Seconds per instance. The runner's own documented default (issue #88), argued
#: from the committed anytime trace in the benchmark README ("Why 60s").
DEFAULT_TIME_LIMIT = 60.0

#: Seed of the published run: the PRE-REGISTERED seed (#141). Kept so a re-run
#: differs from the old table only in the engine, not in the configuration, and
#: the only seed allowed to write `comparison.csv` / `anytime_trace.csv` -- so the
#: published seed is fixed before any run, never chosen after one. Every other
#: seed publishes into the per-seed table (`campaign_report.SEEDS_TABLE_NAME`)
#: beside it, as this one does too.
DEFAULT_SEED = 1

#: Solves per process and processes at a time, as the run record states them.
#: The runner is single-threaded and the driver runs one instance at a time;
#: both are recorded rather than implied, because a wall-clock-budgeted number
#: means nothing without them.
PARALLEL_SOLVES = 1
THREADS_PER_SOLVE = 1

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

    `$XDG_STATE_HOME/cbls/minlplib-rerun` (`~/.local/state/...` when unset). Until
    #141 it was `<build-dir>/minlplib-rerun`, which pre-push's ```clean fence
    (`rm -rf build`) deletes -- so a push in the middle of a fifty-minute campaign
    threw away every staged instance and its log. Outside the repository rather
    than in a gitignored directory inside it, because the worktree workflow
    deletes whole checkouts after a merge, and a gitignored `results/` would go
    with them. XDG's *state* directory and not its cache: state is defined as
    data that should survive a restart, cache as data that may be deleted at any
    time, and a resumable campaign is the former. Two checkouts sharing it is
    safe: the stamp refuses to resume rows from another commit, budget, seed or
    host. Read at call time, so a test can point it elsewhere.
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
    table. A whole-roster run onto the default paths -- at any seed -- also
    publishes its rows into the per-seed table.

    The default staging directory is keyed by commit as well as seed: it
    outlives every campaign now, and keyed by seed alone the first run after any
    new commit would hit the stamp's refusal instead of starting fresh.
    """
    published_out = args.inst_dir / "comparison.csv"
    stage = args.staging_dir or default_staging_root() / sha / f"seed{args.seed}"
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

    Same source and same order the runner uses when given no `--instance`, so an
    assembled table is row-for-row comparable with a whole-roster run's. An
    absent file yields an empty roster rather than raising, so preflight can turn
    it into the refusal that names `download.py`.
    """
    if not bounds_csv.exists():
        return []
    with bounds_csv.open(newline="") as fh:
        return [row["instance"] for row in csv.DictReader(fh)]


def _build_problems(args: argparse.Namespace, sha: str) -> list[str]:
    """Refusals about the binary this run would measure and the SHA it labels it with."""
    problems: list[str] = []
    if sha.endswith("-dirty"):
        problems.append(
            f"working tree is dirty ({sha}); commit or stash first — a row labelled with a "
            "plain SHA must have been produced by that commit's code"
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

    Checked up front because `publish_seed_rows` refuses the same things AFTER
    the roster has been solved: a table whose header is not this runner's, or a
    run record that does not parse, would otherwise cost fifty minutes to find.
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
    try:
        campaign_report.load_seed_run_records(seeds_out)
    except ValueError as exc:
        problems.append(str(exc))
    return problems


def common_preflight(args: argparse.Namespace, sha: str, roster: Sequence[str]) -> list[str]:
    """The refusals shared with `run_ablation.py`: the binary, the SHA and the roster.

    Kept apart from `preflight` because the ablation driver's namespace has no
    `--out`/`--seed`/`--staging-dir`, which the per-seed-table check needs.
    """
    return _build_problems(args, sha) + _data_problems(args, roster)


def preflight(args: argparse.Namespace, sha: str, roster: Sequence[str]) -> list[str]:
    """Every reason to refuse this invocation, checked before anything is spent.

    All of them are cheap and all of them would otherwise surface as a wrong or
    half-written published table — some of them 50 minutes in.
    """
    return common_preflight(args, sha, roster) + _seeds_table_problems(args)


def usage_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """The reason to reject the argument combination outright, or None."""
    if args.time_limit <= 0.0:
        return f"--time-limit must be > 0 (got {args.time_limit})"
    if args.build_jobs < 1:
        return f"--build-jobs must be >= 1 (got {args.build_jobs})"
    seed_refusal = seed_policy_error(args, published_out)
    if seed_refusal:
        return seed_refusal
    if args.instances:
        return _subset_error(args, published_out)
    return _whole_roster_error(args, published_out)


def _whole_roster_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """`usage_error` for a whole-roster run: never publish one of the two artifacts alone."""
    if args.seed != DEFAULT_SEED:
        # Its table and trace default into its own staging directory and its
        # rows into the per-seed table, so neither published artifact is in
        # reach and the two guards below have nothing to protect.
        return None
    # A whole-roster run replaces the published comparison.csv, so skipping
    # the trace would leave anytime_trace.csv describing the previous engine
    # with nothing in either file saying the two disagree — and the README's
    # post-run step recomputes its budget table from that stale trace.
    # EITHER way of not writing the published trace, not just --no-trace: a
    # scratch --trace-out with a defaulted --out leaves anytime_trace.csv at
    # the previous engine just as surely, and slips past the guard below too
    # (that one keys on --out having moved, and here it has not).
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
    # The converse hazard, and the one the #149 campaign walks straight into:
    # `--out` moved off the published table but `--trace-out` left to its
    # default, which is the published `anytime_trace.csv`. Nothing else stops
    # it -- `publish` assembles the staged traces into `trace_out` whatever
    # `out` is, and the runner's own guard never sees the published path
    # because each instance is staged. A scratch run would then replace the
    # published anytime profile with one measured at another seed or budget,
    # at exit 0, while reporting that it wrote a scratch table.
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


def _published_files(inst_dir: Path) -> dict[Path, str]:
    tables = [inst_dir / name for name in PUBLISHED_NAMES]
    return {f.resolve(): f.name for t in tables for f in (t, run_record_path(t))}


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
    for flag, path in written:
        if path.resolve() in protected:
            return (
                f"{flag} resolves to the published {protected[path.resolve()]}; a scratch "
                "output must not overwrite a published file"
            )
    return None


def seed_policy_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """Refuse any route by which a seed other than `DEFAULT_SEED` reaches a published file.

    The published table is ONE pre-registered seed (#141): an A/B against the
    previous table where only the engine differs. A second seed written there,
    by `--out` or by `--trace-out`, would make the published seed whichever one
    was run last -- a post-hoc selection by another name. Also refused at seed 1:
    a `--trace-out` naming the published trace while `--out` is scratch, the
    explicit spelling of the hazard `usage_error`'s defaulted-trace guard covers.
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

    The host and the roster are part of it since #141 moved the default staging directory out of
    the build tree: a home directory can be shared between machines, and a
    campaign resumed on another one would publish one table measured on two.
    """
    return (
        f"commit={sha}\ntime-limit={args.time_limit:g}\nseed={args.seed}\n"
        f"host={socket.gethostname()}\nroster={roster_fingerprint(args.inst_dir)}\n"
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

    Resume matches on a staged file being *complete*, which says nothing about
    what produced it. Without this, a run interrupted at one commit and resumed
    at another — or a five-second smoke run over the whole roster — publishes one
    table built from two configurations, and only `wall_seconds` would betray it.
    """
    return stamp_refusal(
        stage / STAMP_NAME,
        staging_stamp(args, sha),
        resume=args.resume,
        label="staged",
        advice="Delete the staging directory, pass a fresh --staging-dir, or pass --no-resume; "
        "reusing these rows would mix two configurations into one table.",
    )


def staged_row_complete(path: Path, sha: str) -> bool:
    """True when a staging CSV holds a header and a whole result row for `sha`.

    Three ways a staged file can look done without being usable:

    * the runner opens its CSV and writes the header before it solves anything,
      so a killed job leaves a header-only file;
    * a job killed mid-write leaves a torn last line, which still reads as a
      line — hence the trailing-newline and field-count checks;
    * a row staged by an earlier invocation carries *that* run's commit, and
      reusing it publishes a table whose rows disagree about which engine
      produced them (issue #123 asks for one SHA in the column).
    """
    row = _staged_row(path)
    return row is not None and path.read_text().endswith("\n") and row.get("commit_sha") == sha


def staged_unpublishable(path: Path) -> bool:
    """Whether a staging CSV's row must NOT stand in for a fresh solve.

    Read apart from `staged_row_complete`, which asks whether the FILE is whole.
    Such a row is whole -- header, one full line, the right commit -- and that is
    the problem: it would otherwise stand in for a solve on the next resume.

    True for a row the runner wrote after throwing, and for any note
    `runner.STAGEABLE_NOTES` does not recognise, an absent `note` column included. An
    unreadable file is left to `staged_row_complete` and reported as incomplete,
    not as unpublishable.
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

    Both files matter: a run made with `--no-trace` leaves a complete CSV and no
    trace at all, and skipping on the CSV alone would replace `comparison.csv`
    and only then fail to assemble the trace.

    A row the runner wrote after THROWING is refused (#153), as is any row whose
    note `runner.STAGEABLE_NOTES` does not recognise. It is complete by every structural
    check -- which is exactly why it has to be named here: the runner exits
    nonzero on it and `run_roster` aborts, but `--resume` is the default, so
    without this the next invocation would skip the instance and `publish` would
    assemble a row that measured nothing into `comparison.csv`. The abort would
    have delayed the bad publish by one invocation rather than preventing it. A
    coverage gap -- `unsupported`, or a missing `.nl` -- exits 0 and is NOT
    refused here; it is a documented row like any other.
    """
    if not staged_row_complete(stage / f"{name}.csv", sha):
        return False
    if staged_unpublishable(stage / f"{name}.csv"):
        return False
    return not args.trace or (stage / f"{name}.trace.csv").exists()


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
        extra=["--trace", str(stage / f"{name}.trace.csv")] if args.trace else [],
    )


def build_command(args: argparse.Namespace) -> list[str]:
    return runner.build_command(args.build_dir, args.build_jobs)


def merge_command(inst_dir: Path) -> list[str]:
    """Rebuild `comparison_all.csv` from the CSVs on disk, solving nothing.

    `--merge-only` re-reads `scip_baseline.csv` and `bounds.csv` unchanged and
    takes only the CBLS rows from the freshly written `comparison.csv`, which is
    what keeps the `published-bks` and `scip` rows out of this run's reach.
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
        machine=machine,
        concurrency={
            "parallel_solves": PARALLEL_SOLVES,
            "threads_per_solve": THREADS_PER_SOLVE,
            "build_jobs": args.build_jobs,
        },
    )


def write_run_record(table: Path, record: RunRecord) -> None:
    """The single-seed table's record, `<table stem>.run.json` beside it."""
    write_json(run_record_path(table), {"schema": RUN_RECORD_SCHEMA, **asdict(record)})


def publish_seed_rows(seeds_out: Path, seed: int, table: Path) -> None:
    """Replace `seed`'s rows in the per-seed table with `table`'s, keeping every other seed.

    Upsert, not append: a re-run of one seed replaces that seed's block and
    leaves the others' rows intact, so seeds published one invocation at a time
    accumulate without one clobbering another. Rows are ordered by seed, each
    seed's block in `table`'s (roster) order. Written atomically.
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
            kept = [row for row in reader if row and row[0] != str(seed)]
    rows = sorted([*kept, *new_rows], key=lambda row: int(row[0]))
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows([list(SEEDS_TABLE_COLUMNS), *rows])
    atomic_write(seeds_out, buffer.getvalue())


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

    Deliberately thin, and deliberately NOT its own definitions: until #142 this
    function held `elec` out of every count while the README's tally counted the
    whole roster, so a reader could not tell which denominator a number used. Both
    now come from `campaign_report.summarize_results`, which states the rule it
    applies (`campaign_report.AGGREGATION_RULE`) -- documented failures count in
    the roster counts and are held out of the quality aggregates -- and this
    prints that rule beside the numbers. The full report, anytime scores and SCIP
    head-to-head included, is `campaign_report.py`.
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

    It runs after the table has been atomically written, so a refusal in the
    summary (an instance `bounds.csv` does not know, a duplicate row) must not
    turn a successful publish into a traceback that reads as a failed one.
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
            host=host,
        )
        print(campaign_report.render_seeds_text(summary))
    except (ValueError, KeyError, OSError) as exc:
        print(
            f"WARNING: {seeds_out} is written, but its multi-seed summary could not be derived: "
            f"{exc}. Run benchmarks/minlplib/campaign_report.py --seeds over it.",
            file=sys.stderr,
        )


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
        log = stage / f"{name}.log"
        completed = run_process(cmd, log=log)
        if completed.returncode != 0 or not staged_complete(args, sha, name, stage):
            what_next = (
                "The runner threw on this instance and its staged row carries "
                "read-error/build-error/solve-error: no search ran, so publishing the table "
                "would put a row in it that measures nothing. `staged_complete` refuses that "
                "row, so a re-run hits the same throw rather than resuming past it -- fix the "
                "instance or drop it from the roster."
                if completed.returncode == RUNNER_EXIT_ERRORED
                else "Re-running resumes from here."
            )
            raise RuntimeError(
                f"{name} failed (exit {completed.returncode}); see {log}. {what_next}"
            )
    return resumed


def publish(
    args: argparse.Namespace, roster: Sequence[str], paths: Paths, record: RunRecord
) -> int:
    """Assemble the staged rows into the tables, record the run, publish the seed, re-merge.

    The run record goes beside `paths.out` whatever it is -- a scratch table is a
    results set too, and the machine is what makes it readable later.
    """
    assemble(paths.stage, roster, paths.out, ".csv")
    print(f"wrote {paths.out}")
    if args.trace:
        assemble(paths.stage, roster, paths.trace_out, ".trace.csv")
        print(f"wrote {paths.trace_out}")
    write_run_record(paths.out, record)
    print(f"wrote {run_record_path(paths.out)}")
    if paths.seeds_out is not None:
        publish_seed_rows(paths.seeds_out, args.seed, paths.out)
        publish_seed_record(paths.seeds_out, record)
        print(f"wrote seed {args.seed} into {paths.seeds_out}")
    merge_failed = False
    if args.merge:
        # check=False: `reference_solve.py --merge-only` exits 2 on its own
        # refusals, and a traceback here would leave comparison.csv rewritten
        # with no summary and no word about what to do next.
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
    with driver_lock():
        if args.build:
            subprocess.run(build_command(args), check=True)
        paths.stage.mkdir(parents=True, exist_ok=True)
        conflict = staging_stamp_conflict(paths.stage, args, sha)
        if conflict:
            print(conflict, file=sys.stderr)
            return 2
        resumed = run_roster(args, sha, roster, paths.stage)
        if resumed:
            print(
                f"note: {resumed} of {len(roster)} row(s) were staged by an earlier invocation; "
                "the run record's machine and load are this invocation's, not theirs",
                file=sys.stderr,
            )
        return publish(args, roster, paths, run_record(args, sha, roster, machine, resumed))


#: Held by every invocation for its whole run, in the shared staging root.
DRIVER_LOCK_NAME = "driver.lock"


@contextlib.contextmanager
def driver_lock() -> Iterator[None]:
    """One driver invocation at a time on this machine, or refuse.

    This is what makes the run record's `parallel_solves: 1` true rather than
    asserted, and what serialises the per-seed table's read-modify-write: two
    seeds started in two terminals would otherwise halve each other's iteration
    counts and could each publish over the other's block. In the default staging
    root whatever `--staging-dir` says, so a custom staging dir is not a way
    around it. `flock` is dropped by the kernel when the holder exits, so a lock
    file left by a crash is not stale.
    """
    root = default_staging_root()
    root.mkdir(parents=True, exist_ok=True)
    path = root / DRIVER_LOCK_NAME
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.seek(0)
            holder = handle.read().strip() or "pid unknown"
            raise RuntimeError(
                f"another run_benchmark.py holds {path} ({holder}); wall-clock-budgeted solves "
                "must not share the machine. Wait for it to finish."
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield


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
    if roster:
        print(
            f"solve (x{len(roster)}, serial): "
            f"{' '.join(runner_command(args, sha, roster[0], paths.stage))}"
        )
    if args.merge:
        print(f"merge: {' '.join(merge_command(args.inst_dir))}")
    print(f"run record: {run_record_path(paths.out)}")
    if paths.seeds_out is not None:
        print(f"publish: seed {args.seed}'s rows into {paths.seeds_out}")
    print(f"estimated {len(roster) * args.time_limit / 60.0:.0f} min of solving on a quiet machine")
    # Non-zero on a refusal, so --dry-run is usable as a scriptable precheck.
    return 2 if problems else 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
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
        help="default $XDG_STATE_HOME/cbls/minlplib-rerun/<commit>/seed<N> (~/.local/state/...)",
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
        help="re-solve every instance even if a staged result exists",
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
    sha = commit_sha()
    paths = resolve_paths(args, sha)
    # The merge reads and rewrites the published comparison_all.csv from the
    # published comparison.csv, so it is meaningless — and destructive — when
    # this run's rows went somewhere else. Resolved paths, so a relative --out
    # naming the published file is not mistaken for a scratch one.
    skipping_merge = args.merge and paths.out.resolve() != paths.published_out.resolve()
    args.merge = args.merge and not skipping_merge

    roster = args.instances or roster_from_bounds(args.inst_dir / "bounds.csv")
    problems = preflight(args, sha, roster)
    if problems and not args.dry_run:
        for problem in problems:
            print(f"refusing to run: {problem}", file=sys.stderr)
        return 2

    print(f"commit {sha}, {args.time_limit:g}s/instance, seed {args.seed}, serial")
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
    return execute(args, sha, roster, paths)


if __name__ == "__main__":
    sys.exit(main())
