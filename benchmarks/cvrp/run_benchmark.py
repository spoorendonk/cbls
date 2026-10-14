"""CVRP campaign driver: one `cbls_cvrp` process per job, resumable (#199).

A job is one (instance, arm, budget, seed, threads). The driver plans the
product of `--roster`, `--arms`, `--budget`, `--seeds` and `--threads`, runs each
job as its own process through `benchmarks/common/jobs.py` and leaves one record
per job through `benchmarks/common/records.py`. `score.py` turns a results
directory into tables; this driver writes no table.

THE RESUME RULE. A job is done when its record path,

    <results>/<engine>/<arm>/t<threads>/b<budget>/<instance>.s<seed>.json

holds a readable record whose (engine, instance, arm, budget_seconds, seed,
threads) are that job's. A failure record -- a timeout, a crash, a run the runner itself
failed -- is readable and so is done: resume does not silently retry a job that
keeps failing. `--retry-failed` re-runs those, `--force` re-runs everything.

WHAT A JOB LEAVES. The runner writes its own JSON to `<stem>.runner.json` and,
when it found a verified solution, `<stem>.sol`; its output goes to `<stem>.log`.
The driver then writes the record: the runner's JSON plus `arm`, `arm_flags`,
`status` (`ok`, `failed`, `timeout`), `returncode` and `driver_seconds`. A job
the timeout killed, or one that left no readable runner JSON, gets a failure
record carrying only the key, `status`, and `feasible: false`. The record shape
is declared in `score.py` (`RECORD_FIELDS`), which scores PyVRP's records too.

WHAT IS REFUSED, before anything runs:

* a build directory that is not a Release, uninstrumented build of this
  checkout (`provenance.build_dir_problems`);
* a modified tree, whose commit names code that did not run, unless
  `--allow-dirty`, which stamps the directory not comparable (below);
* a results directory stamped by another commit or another arm table
  (`records.stamp_refusal` on `stamp.txt`);
* `--jobs` above 1 without `--not-comparable`. Jobs sharing the machine halve
  each other's iterations at a fixed wall-clock budget, so their numbers are
  comparable neither to each other nor to a one-up run. `--not-comparable` is
  stamped into the directory and `score.py` banners every table from it and
  refuses to write the published table.

The whole run holds `jobs.wallclock_lock`, the lock the MINLPLib drivers take,
so two timed drivers never share the machine. The machine record and the engine
commit are written once per results directory, to `machine.json`.

    .venv/bin/python benchmarks/cvrp/run_benchmark.py --roster smoke --budget 10 \\
        --seeds 1 --arms guided --results /tmp/cvrp-smoke
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common import jobs  # noqa: E402
from benchmarks.common.provenance import (  # noqa: E402
    REPO_ROOT,
    build_dir_problems,
    cmake_cache,
    commit_sha,
    machine_record,
)
from benchmarks.common.records import (  # noqa: E402
    read_json_object,
    stamp_refusal,
    write_json,
)
from benchmarks.instances.cvrp.download import read_roster  # noqa: E402

DEFAULT_INST_DIR = REPO_ROOT / "benchmarks" / "instances" / "cvrp"
DEFAULT_BUILD_DIR = REPO_ROOT / "build"
BINARY_NAME = "cbls_cvrp"

#: `--roster` names and the table each reads.
ROSTERS: dict[str, str] = {
    "smoke": "smoke.csv",
    "dev": "dev.csv",
    "report": "report.csv",
    "full": "roster.csv",
}

#: Seconds a job may run past its budget before it is killed: the model build
#: and the instance read sit outside the solve's own deadline, and both are well
#: under a second at n = 1000. A job that needs the margin is a defect, and the
#: kill records it as one.
TIMEOUT_MARGIN_SECONDS = 60.0

#: Each runner caps its own address space. CVRP at n = 1000 needs well under a
#: gigabyte single-threaded; the cap is there so a runaway takes one job, not
#: the machine.
DEFAULT_MEM_LIMIT_GB = 4.0

STAMP_NAME = "stamp.txt"
MACHINE_NAME = "machine.json"

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_TIMEOUT = "timeout"


@dataclass(frozen=True)
class Arm:
    """A named configuration of one engine: the flags its program runs with."""

    name: str
    flags: tuple[str, ...]
    engine: str = "cbls"


#: The arms this driver knows. `default` is the engine's own configuration, the
#: one a published table carries (`score.PUBLISHABLE_GROUPS`). `guided` is the arm
#: the smoke check uses (benchmarks/instances/cvrp/README.md, "Search arm for
#: the smoke check"); which arm a campaign runs is decided on `dev.csv` (#204).
ARMS: dict[str, Arm] = {
    arm.name: arm
    for arm in (
        Arm("default", ()),
        Arm("guided", ("--structural-selection", "violation_guided", "--neighbours", "20")),
    )
}


@dataclass(frozen=True)
class Job:
    instance: str
    arm: Arm
    budget: float
    seed: int
    threads: int

    def stem(self, results: Path) -> Path:
        """The path every file of this job shares, less its suffix: the resume key."""
        return (
            results
            / self.arm.engine
            / self.arm.name
            / f"t{self.threads}"
            / f"b{self.budget:g}"
            / f"{self.instance}.s{self.seed}"
        )

    def key(self) -> dict[str, object]:
        """The record fields that identify this job. A record whose fields differ
        is not this job's, whatever path it sits at."""
        return {
            "engine": self.arm.engine,
            "instance": self.instance,
            "arm": self.arm.name,
            "budget_seconds": self.budget,
            "seed": self.seed,
            "threads": self.threads,
        }


def _with_suffix(stem: Path, suffix: str) -> Path:
    # `Path.with_suffix` would replace the `.s<seed>` part of the stem.
    return stem.with_name(stem.name + suffix)


def record_path(job: Job, results: Path) -> Path:
    return _with_suffix(job.stem(results), ".json")


def runner_json_path(job: Job, results: Path) -> Path:
    return _with_suffix(job.stem(results), ".runner.json")


def solution_path(job: Job, results: Path) -> Path:
    return _with_suffix(job.stem(results), ".sol")


def log_path(job: Job, results: Path) -> Path:
    return _with_suffix(job.stem(results), ".log")


def timeout_seconds(job: Job) -> float:
    """The wall-clock limit on one job's process: its budget plus the margin."""
    return job.budget + TIMEOUT_MARGIN_SECONDS


def plan_jobs(
    instances: list[str],
    arms: list[Arm],
    budgets: list[float],
    seeds: list[int],
    threads: list[int],
) -> list[Job]:
    """Every job, arms innermost: the arms of one (instance, seed) run back to
    back, so a paired A/B difference is not also a difference in machine state."""
    return [
        Job(instance, arm, budget, seed, t)
        for budget in budgets
        for t in threads
        for seed in seeds
        for instance in instances
        for arm in arms
    ]


def read_record(job: Job, results: Path) -> dict[str, object] | None:
    """The job's record, or None when there is none or it is another job's."""
    record = read_json_object(record_path(job, results))
    if record is None:
        return None
    if any(record.get(k) != v for k, v in job.key().items()):
        return None
    return record


def needs_run(job: Job, results: Path, *, force: bool, retry_failed: bool) -> bool:
    """THE RESUME RULE (module doc)."""
    if force:
        return True
    record = read_record(job, results)
    if record is None:
        return True
    return retry_failed and record.get("status") != STATUS_OK


def build_command(job: Job, binary: Path, inst_dir: Path, results: Path, sha: str) -> list[str]:
    return [
        str(binary),
        "--instance",
        job.instance,
        "--inst-dir",
        str(inst_dir),
        "--out",
        str(runner_json_path(job, results)),
        "--solution-out",
        str(solution_path(job, results)),
        "--budget",
        f"{job.budget:g}",
        "--seed",
        str(job.seed),
        "--threads",
        str(job.threads),
        "--commit",
        sha,
        *job.arm.flags,
    ]


def failure_record(job: Job, status: str, outcome: jobs.Outcome, sha: str) -> dict[str, object]:
    """What a job that left no usable runner JSON records: its key and how it ended."""
    return {
        **job.key(),
        "arm_flags": list(job.arm.flags),
        "status": status,
        "returncode": outcome.returncode,
        "driver_seconds": outcome.elapsed,
        "feasible": False,
        "verified": False,
        "objective": None,
        "time_to_first_feasible": None,
        "commit_sha": sha,
    }


def job_record(
    job: Job, outcome: jobs.Outcome, runner: dict[str, object] | None, sha: str
) -> dict[str, object]:
    """The record for a finished process. Pure, so the wiring is testable."""
    if outcome.timed_out:
        return failure_record(job, STATUS_TIMEOUT, outcome, sha)
    if runner is None:
        return failure_record(job, STATUS_FAILED, outcome, sha)
    return {
        **runner,
        **job.key(),
        "arm_flags": list(job.arm.flags),
        "status": STATUS_OK if outcome.returncode == 0 else STATUS_FAILED,
        "returncode": outcome.returncode,
        "driver_seconds": outcome.elapsed,
    }


def run_job(job: Job, binary: Path, args: argparse.Namespace, sha: str) -> tuple[str, bool]:
    """Run one job and write its record. Returns the line to print and whether
    the job finished cleanly.

    Never raises on the job's own failure: `jobs.run_jobs` would cancel every
    job still queued.
    """
    results: Path = args.results
    stem = job.stem(results)
    stem.parent.mkdir(parents=True, exist_ok=True)
    # The record goes first: a kill between here and the new record leaves no
    # record, so resume re-runs the job rather than reading the old one.
    for stale in (
        record_path(job, results),
        runner_json_path(job, results),
        solution_path(job, results),
    ):
        stale.unlink(missing_ok=True)
    outcome = jobs.run_process(
        build_command(job, binary, args.inst_dir, results, sha),
        timeout=timeout_seconds(job),
        mem_limit_gb=args.mem_limit_gb,
        own_session=True,
        log=log_path(job, results),
    )
    runner = None if outcome.timed_out else read_json_object(runner_json_path(job, results))
    record = job_record(job, outcome, runner, sha)
    write_json(record_path(job, results), record)
    objective = record.get("objective")
    line = (
        f"{job.instance:14s} {job.arm.name:10s} b={job.budget:g} s={job.seed} t={job.threads} "
        f"{record['status']:7s} obj={objective if objective is not None else '-'} "
        f"{outcome.elapsed:.1f}s"
    )
    return line, record["status"] == STATUS_OK


def stamp_text(sha: str, comparable: bool) -> str:
    """What makes two invocations' records comparable: the commit, the arm table
    (every arm, not just this invocation's, so adding an arm to a later
    invocation does not refuse), and whether the jobs ran one at a time."""
    lines = [f"commit={sha}", f"comparable={'yes' if comparable else 'no'}"]
    lines += [f"arm {arm.engine}/{arm.name}={' '.join(arm.flags)}" for arm in ARMS.values()]
    return "\n".join(lines) + "\n"


def stamp_comparable(results: Path) -> bool:
    """Whether the directory's jobs ran one at a time; False when unstamped."""
    stamp = results / STAMP_NAME
    return stamp.exists() and "comparable=yes\n" in stamp.read_text()


def check_arguments(args: argparse.Namespace) -> str | None:
    if args.jobs < 1:
        return "--jobs must be >= 1"
    if args.jobs > 1 and not args.not_comparable:
        return (
            f"--jobs {args.jobs} runs timed solves side by side, and their numbers are "
            "comparable neither to each other nor to a one-up run. Pass --not-comparable "
            "if that is what this run is for (a tooling check); score.py will banner it."
        )
    if any(b <= 0 for b in args.budget):
        return "--budget must be > 0"
    if any(t < 1 for t in args.threads):
        return "--threads must be >= 1"
    unknown = [a for a in args.arms if a not in ARMS]
    if unknown:
        return f"unknown arm(s) {unknown}; known: {sorted(ARMS)}"
    return None


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--roster", choices=sorted(ROSTERS), default="smoke")
    parser.add_argument("--budget", type=float, nargs="+", default=[10.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--arms", nargs="+", default=["default"])
    parser.add_argument("--threads", type=int, nargs="+", default=[1])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--not-comparable", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--force", action="store_true", help="re-run every planned job")
    parser.add_argument("--retry-failed", action="store_true", help="re-run failed jobs")
    parser.add_argument("--mem-limit-gb", type=float, default=DEFAULT_MEM_LIMIT_GB)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_BUILD_DIR)
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> tuple[str, Path] | str:
    """The commit and binary to run with, or the refusal."""
    problems = build_dir_problems(args.build_dir, cmake_cache(args.build_dir))
    if problems:
        return "\n".join(problems)
    binary = args.build_dir / BINARY_NAME
    if not binary.is_file():
        return f"{binary} not found; build the {BINARY_NAME} target"
    sha = commit_sha()
    if sha.endswith("-dirty") and not args.allow_dirty:
        return (
            f"the tree is modified ({sha}), so the commit would name code that did not run; "
            "commit first, or pass --allow-dirty for a tooling check"
        )
    return sha, binary


def execute(args: argparse.Namespace, sha: str, binary: Path) -> int:
    results: Path = args.results
    results.mkdir(parents=True, exist_ok=True)
    refusal = stamp_refusal(
        results / STAMP_NAME,
        # A modified tree is never comparable: two different uncommitted edits of
        # one commit stamp alike, so their records could share a directory.
        stamp_text(sha, comparable=not args.not_comparable and not sha.endswith("-dirty")),
        resume=True,
        label="recorded",
        advice="\nUse a new --results directory.\n",
    )
    if refusal is not None:
        print(f"refusing: {refusal}", file=sys.stderr)
        return 2
    machine = results / MACHINE_NAME
    if not machine.exists():
        write_json(
            machine,
            {
                "commit_sha": sha,
                "argv": sys.argv,
                "started": time.time(),
                "machine": machine_record(),
            },
        )
    instances = [e.instance for e in read_roster(args.inst_dir / ROSTERS[args.roster])]
    planned = plan_jobs(
        instances, [ARMS[a] for a in args.arms], args.budget, args.seeds, args.threads
    )
    todo = [
        j
        for j in planned
        if needs_run(j, results, force=args.force, retry_failed=args.retry_failed)
    ]
    print(f"{len(planned)} job(s) planned, {len(todo)} to run", file=sys.stderr)
    failed = 0
    for line, ok in jobs.run_jobs(todo, lambda j: run_job(j, binary, args, sha), workers=args.jobs):
        print(line, flush=True)
        failed += not ok
    if failed:
        print(f"{failed} job(s) did not finish cleanly; see their .log", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    complaint = check_arguments(args)
    if complaint is not None:
        print(f"refusing: {complaint}", file=sys.stderr)
        return 2
    checked = preflight(args)
    if isinstance(checked, str):
        print(f"refusing: {checked}", file=sys.stderr)
        return 2
    sha, binary = checked
    try:
        with jobs.wallclock_lock("cvrp/run_benchmark.py"):
            return execute(args, sha, binary)
    except jobs.LockHeldError as exc:
        print(f"refusing to run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
