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

Why it drives the runner one instance at a time rather than issuing the single
whole-roster command the README used to document:

* **Resumable.** `cbls_minlplib` truncates its output CSV on open and writes
  rows as it goes, so a crash 40 minutes in leaves the published table
  half-replaced and the work lost. Here each instance lands in its own staging
  file under the build directory and a re-run skips the ones already complete.
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
* refuses to resume a staging directory written by a different commit, budget or
  seed, and re-solves any individual staged row whose `commit_sha` disagrees.

Usage:
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --dry-run
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --instances nvs01 \
        --time-limit 3 --out /tmp/c.csv --trace-out /tmp/t.csv --staging-dir /tmp/stage
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INST_DIR = REPO_ROOT / "benchmarks" / "instances" / "minlplib"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
REFERENCE_SOLVE = Path(__file__).resolve().parent / "reference_solve.py"

#: The runner target and the executable `cmake --build` produces for it.
RUNNER_TARGET = "cbls_minlplib"

#: Seconds per instance. The runner's own documented default (issue #88), argued
#: from the committed anytime trace in the benchmark README ("Why 60s").
DEFAULT_TIME_LIMIT = 60.0

#: Seed of the published run. Kept so a re-run differs from the old table only
#: in the engine, not in the configuration.
DEFAULT_SEED = 1

#: Parallel jobs for the *build* only. The solves are always serial: the budget
#: is wall-clock, so a concurrent solve is not comparable to the committed table
#: or to the other rows of its own run.
DEFAULT_BUILD_JOBS = 4

#: Instances whose rows are published as documented failures and are excluded
#: from every aggregate and every quality claim, per issue #87 ("Do not publish
#: `elec` rows until #110 lands and #116's criterion can actually be checked").
#: They stay *in* the roster: #123 asks for 50 instances, the roster of record is
#: `bounds.csv`, and dropping the rows would make the table disagree with it.
CLAIM_EXCLUDED = ("elec25", "elec50")

#: Records which configuration a staging directory's rows belong to.
STAMP_NAME = "stamp.txt"


@dataclass(frozen=True)
class Paths:
    """Where this invocation reads the roster from and writes its results to."""

    published_out: Path
    out: Path
    trace_out: Path
    stage: Path


def resolve_paths(args: argparse.Namespace) -> Paths:
    published_out = args.inst_dir / "comparison.csv"
    return Paths(
        published_out=published_out,
        out=args.out or published_out,
        trace_out=args.trace_out or args.inst_dir / "anytime_trace.csv",
        stage=args.staging_dir or args.build_dir / "minlplib-rerun",
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


def _git(*argv: str) -> str:
    out = subprocess.run(["git", *argv], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    return out.stdout.strip()


def commit_sha() -> str:
    """The commit a run is attributed to, marked `-dirty` when the tree is modified.

    `git rev-parse --short=7` and not `git describe`, whose output format changes
    the moment the repository gains its first tag — the `commit_sha` column would
    silently change shape mid-history.

    Dirtiness comes from `git status --porcelain --untracked-files=no`, which
    covers modifications to tracked files. Untracked files are deliberately not
    counted: a scratch file beside the source says nothing about the code that
    ran. The corollary is that a *new*, never-added source file does not mark
    the tree dirty, so this is a guard against edited code, not against every
    difference from HEAD.
    """
    sha = _git("rev-parse", "--short=7", "HEAD")
    return f"{sha}-dirty" if _git("status", "--porcelain", "--untracked-files=no") else sha


def cmake_cache(build_dir: Path) -> dict[str, str]:
    """The `NAME:TYPE=VALUE` entries of a build directory's cache, by name."""
    cache = build_dir / "CMakeCache.txt"
    if not cache.exists():
        return {}
    entries: dict[str, str] = {}
    for line in cache.read_text().splitlines():
        name, sep, value = line.partition("=")
        if sep and ":" in name and not name.startswith(("#", "//")):
            entries[name.split(":", 1)[0]] = value.strip()
    return entries


def cmake_build_type(build_dir: Path) -> str | None:
    """`CMAKE_BUILD_TYPE` recorded in the build directory's cache, if any."""
    return cmake_cache(build_dir).get("CMAKE_BUILD_TYPE")


def _build_problems(args: argparse.Namespace, sha: str) -> list[str]:
    """Refusals about the binary this run would measure and the SHA it labels it with."""
    problems: list[str] = []
    if sha.endswith("-dirty"):
        problems.append(
            f"working tree is dirty ({sha}); commit or stash first — a row labelled with a "
            "plain SHA must have been produced by that commit's code"
        )
    cache = cmake_cache(args.build_dir)
    if not cache:
        problems.append(
            f"{args.build_dir}/CMakeCache.txt not found; configure first, e.g.\n"
            f'    cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"'
        )
        return problems
    if cache.get("CMAKE_BUILD_TYPE") != "Release":
        problems.append(
            f"{args.build_dir} is CMAKE_BUILD_TYPE={cache.get('CMAKE_BUILD_TYPE') or '(empty)'}, "
            "not Release; these are wall-clock-budgeted solves and an unoptimised build "
            "measures a different engine"
        )
    # Both are sticky cache entries, so a build dir configured once with either
    # keeps it through every later flag-less `cmake -B build` while
    # CMAKE_BUILD_TYPE still reads Release. A sanitizer binary runs several-fold
    # slower and -fno-omit-frame-pointer costs throughput, so either would
    # publish wall-clock-budgeted rows measured on an engine nobody runs. See
    # docs/profiling.md.
    if cache.get("CBLS_SANITIZE"):
        problems.append(
            f"{args.build_dir} is configured with CBLS_SANITIZE={cache['CBLS_SANITIZE']}; "
            "these are wall-clock-budgeted solves and a sanitizer build measures a "
            "different engine. Use a separate build directory for sanitizers."
        )
    if cache.get("CBLS_PROFILE", "OFF") not in ("OFF", "FALSE", "0", ""):
        problems.append(
            f"{args.build_dir} is configured with CBLS_PROFILE={cache['CBLS_PROFILE']}; "
            "frame pointers cost throughput and docs/profiling.md says a build-profile "
            "wall-clock is not a benchmark number. Use a separate build directory."
        )
    home = cache.get("CMAKE_HOME_DIRECTORY")
    if home and Path(home).resolve() != REPO_ROOT:
        problems.append(
            f"{args.build_dir} was configured from {home}, but the commit SHA is read from "
            f"{REPO_ROOT}; the rows would name one checkout and measure another"
        )
    runner = args.build_dir / RUNNER_TARGET
    if not args.build and not runner.exists():
        problems.append(
            f"{runner} not found and --no-build was given; drop --no-build or build the "
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


def preflight(args: argparse.Namespace, sha: str, roster: Sequence[str]) -> list[str]:
    """Every reason to refuse this invocation, checked before anything is spent.

    All of them are cheap and all of them would otherwise surface as a wrong or
    half-written published table — some of them 50 minutes in.
    """
    return _build_problems(args, sha) + _data_problems(args, roster)


def usage_error(args: argparse.Namespace, published_out: Path) -> str | None:
    """The reason to reject the argument combination outright, or None."""
    if args.time_limit <= 0.0:
        return f"--time-limit must be > 0 (got {args.time_limit})"
    if args.build_jobs < 1:
        return f"--build-jobs must be >= 1 (got {args.build_jobs})"
    if not args.instances:
        # A whole-roster run replaces the published comparison.csv, so skipping
        # the trace would leave anytime_trace.csv describing the previous engine
        # with nothing in either file saying the two disagree — and the README's
        # post-run step recomputes its budget table from that stale trace.
        if not args.trace and (args.out is None or args.out.resolve() == published_out.resolve()):
            published_trace = args.trace_out or args.inst_dir / "anytime_trace.csv"
            return (
                "--no-trace on a whole-roster run would publish comparison.csv at this engine "
                f"while leaving {published_trace} at the previous one; drop --no-trace, or "
                "pass --out to write somewhere other than the published table"
            )
        return None
    # A subset rewrites the same whole files a full run does, and its rows would
    # be resumed by the next full run, so it must name scratch paths throughout.
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


def staging_stamp(args: argparse.Namespace, sha: str) -> str:
    """The configuration a staging directory's rows belong to, one field per line."""
    return f"commit={sha}\ntime-limit={args.time_limit:g}\nseed={args.seed}\n"


def staging_stamp_conflict(stage: Path, args: argparse.Namespace, sha: str) -> str | None:
    """Refuse a staging directory written by a different engine, budget or seed.

    Resume matches on a staged file being *complete*, which says nothing about
    what produced it. Without this, a run interrupted at one commit and resumed
    at another — or a five-second smoke run over the whole roster — publishes one
    table built from two configurations, and only `wall_seconds` would betray it.
    """
    stamp = staging_stamp(args, sha)
    path = stage / STAMP_NAME
    if args.resume and path.exists() and path.read_text() != stamp:
        return (
            f"{path} was written by a different configuration:\n"
            f"--- staged ---\n{path.read_text()}--- now ---\n{stamp}"
            "Delete the staging directory, pass a fresh --staging-dir, or pass --no-resume; "
            "reusing these rows would mix two configurations into one table."
        )
    path.write_text(stamp)
    return None


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
    if not path.exists():
        return False
    text = path.read_text()
    if not text.endswith("\n"):
        return False
    rows = list(csv.reader(text.splitlines()))
    if len(rows) < 2 or len(rows[1]) != len(rows[0]):
        return False
    return dict(zip(rows[0], rows[1], strict=True)).get("commit_sha") == sha


def staged_complete(args: argparse.Namespace, sha: str, name: str, stage: Path) -> bool:
    """Whether `name`'s staged output can stand in for a fresh solve.

    Both files matter: a run made with `--no-trace` leaves a complete CSV and no
    trace at all, and skipping on the CSV alone would replace `comparison.csv`
    and only then fail to assemble the trace.
    """
    if not staged_row_complete(stage / f"{name}.csv", sha):
        return False
    return not args.trace or (stage / f"{name}.trace.csv").exists()


def runner_command(args: argparse.Namespace, sha: str, name: str, stage: Path) -> list[str]:
    """The `cbls_minlplib` invocation for one instance."""
    cmd = [
        str(args.build_dir / RUNNER_TARGET),
        str(args.inst_dir),
        "--time-limit",
        f"{args.time_limit:g}",
        "--seed",
        str(args.seed),
        "--commit",
        sha,
        "--instance",
        name,
        "--out",
        str(stage / f"{name}.csv"),
    ]
    if args.trace:
        cmd += ["--trace", str(stage / f"{name}.trace.csv")]
    return cmd


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        "cmake",
        "--build",
        str(args.build_dir),
        "--target",
        RUNNER_TARGET,
        "-j",
        str(args.build_jobs),
    ]


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
    tmp = out.with_name(out.name + ".partial")
    tmp.write_text("\n".join([header, *body]) + "\n")
    os.replace(tmp, out)


def verdict_of(note: str) -> str:
    """The runner's own verdict word, with its appended annotations stripped.

    The runner glues `analysis_notes.csv`'s curated root cause onto the note with
    ` | `, and an integrality remark with `; `. Without stripping them, one
    annotated row becomes its own histogram bucket.
    """
    return note.split("(")[0].split(" | ")[0].split(";")[0].strip()


def summarize(out: Path) -> str:
    """A tally derived from the written table, not recomputed from the solutions.

    Deliberately thin: the runner's own tally applies the tie and improvement
    bands and is captured per instance in the staging logs. This only counts the
    verdicts the table already carries, and holds `elec` apart because its rows
    are published as documented failures rather than as results (issue #87).
    Note that the README's Results tally counts over the *whole* roster, so take
    `rows written` — not `counted` — as its `roster` figure.
    """
    with out.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    counted = [r for r in rows if r["instance"] not in CLAIM_EXCLUDED]
    excluded = [r for r in rows if r["instance"] in CLAIM_EXCLUDED]
    verdicts: dict[str, int] = {}
    for row in counted:
        verdict = verdict_of(row["note"])
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
    lines = [
        f"rows written:         {len(rows)}",
        f"counted (excl. elec): {len(counted)}",
        f"feasible:             {sum(1 for r in counted if r['feasible'] == 'true')}",
    ]
    lines += [f"  {note:<20}{count}" for note, count in sorted(verdicts.items())]
    for row in excluded:
        lines.append(f"excluded from claims: {row['instance']} -> {verdict_of(row['note'])}")
    return "\n".join(lines)


def run_roster(args: argparse.Namespace, sha: str, roster: Sequence[str], stage: Path) -> None:
    """Solve every roster instance serially, skipping the ones already staged."""
    for index, name in enumerate(roster, start=1):
        if args.resume and staged_complete(args, sha, name, stage):
            print(f"[{index}/{len(roster)}] {name}: staged already, skipping")
            continue
        cmd = runner_command(args, sha, name, stage)
        print(f"[{index}/{len(roster)}] {name}: {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, capture_output=True, text=True)
        log = stage / f"{name}.log"
        log.write_text(completed.stdout + completed.stderr)
        if completed.returncode != 0 or not staged_complete(args, sha, name, stage):
            raise RuntimeError(
                f"{name} failed (exit {completed.returncode}); see {log}. Re-running resumes "
                "from here."
            )


def publish(args: argparse.Namespace, roster: Sequence[str], paths: Paths) -> int:
    """Assemble the staged rows into the published tables, then re-merge."""
    assemble(paths.stage, roster, paths.out, ".csv")
    print(f"wrote {paths.out}")
    if args.trace:
        assemble(paths.stage, roster, paths.trace_out, ".trace.csv")
        print(f"wrote {paths.trace_out}")
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
    print(summarize(paths.out))
    return 1 if merge_failed else 0


def execute(args: argparse.Namespace, sha: str, roster: Sequence[str], paths: Paths) -> int:
    """Build, solve the roster, and publish."""
    if args.build:
        subprocess.run(build_command(args), check=True)
    paths.stage.mkdir(parents=True, exist_ok=True)
    conflict = staging_stamp_conflict(paths.stage, args, sha)
    if conflict:
        print(conflict, file=sys.stderr)
        return 2
    run_roster(args, sha, roster, paths.stage)
    return publish(args, roster, paths)


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
    parser.add_argument("--out", type=Path, default=None, help="default <inst-dir>/comparison.csv")
    parser.add_argument(
        "--trace-out", type=Path, default=None, help="default <inst-dir>/anytime_trace.csv"
    )
    parser.add_argument(
        "--staging-dir", type=Path, default=None, help="default <build-dir>/minlplib-rerun"
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
    paths = resolve_paths(args)
    refusal = usage_error(args, paths.published_out)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    # The merge reads and rewrites the published comparison_all.csv from the
    # published comparison.csv, so it is meaningless — and destructive — when
    # this run's rows went somewhere else. Resolved paths, so a relative --out
    # naming the published file is not mistaken for a scratch one.
    skipping_merge = args.merge and paths.out.resolve() != paths.published_out.resolve()
    args.merge = args.merge and not skipping_merge

    sha = commit_sha()
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
    if args.dry_run:
        return describe_plan(args, sha, roster, paths, problems)
    if skipping_merge:
        print(f"note: --out is not {paths.published_out}; skipping the comparison_all.csv merge")
    return execute(args, sha, roster, paths)


if __name__ == "__main__":
    sys.exit(main())
