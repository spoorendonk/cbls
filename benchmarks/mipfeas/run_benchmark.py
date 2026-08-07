"""Drive a MIPfeas run: every roster instance against every engine.

Built for an unattended multi-hour run on a bigger machine, so:

* one process per (instance, engine) — a job that dies takes only itself;
* resumable — a job whose result file already exists is skipped, so an
  interrupted run continues where it stopped;
* size-aware — the largest instances run on their own after the rest, instead of
  four-up against a memory limit;
* every job is bounded by a wall-clock timeout and, optionally, an address-space
  limit, and a job killed by either leaves a result recording that.

Usage:
    python run_benchmark.py --roster smoke --budget 60 --jobs 2
    python run_benchmark.py --roster full --budget 600 --jobs 4 --mem-limit-gb 6
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INSTANCE_DIR = REPO_ROOT / "benchmarks" / "instances" / "mipfeas"
DEFAULT_CBLS_BIN = REPO_ROOT / "build" / "cbls_mipfeas"
CPSAT_SCRIPT = Path(__file__).resolve().parent / "cpsat_solve.py"

ENGINES = ("cbls", "cpsat")

#: Instances whose gzipped file is at least this large run alone rather than
#: alongside others. The MIPfeas roster spans four orders of magnitude in size.
DEFAULT_LARGE_BYTES = 5_000_000

#: Grace on top of the budget before a job is killed. Large, because a large model's
#: read and build happen before the search clock starts and the first search batch is
#: not interruptible: square47 spends ~170s on that before its first iteration. A job
#: killed here is scored as a failure, so the slack has to cover the worst case rather
#: than the typical one.
TIMEOUT_SLACK_SECONDS = 900.0


@dataclass(frozen=True)
class Job:
    engine: str
    instance: str

    def result_path(self, results_dir: Path) -> Path:
        return results_dir / self.engine / f"{self.instance}.json"


def read_roster(path: Path) -> list[str]:
    with open(path, newline="") as fh:
        return [row["instance"] for row in csv.DictReader(fh)]


def read_sizes(manifest: Path) -> dict[str, int]:
    if not manifest.exists():
        return {}
    with open(manifest, newline="") as fh:
        return {row["instance"]: int(row["bytes"]) for row in csv.DictReader(fh)}


def resolve_roster(value: str, inst_dir: Path) -> Path:
    named = {"smoke": inst_dir / "smoke.csv", "full": inst_dir / "roster.csv"}
    return named.get(value, Path(value))


def commit_sha() -> str:
    """The commit a run is attributed to, marked `-dirty` when the tree is modified.

    A plain SHA from a modified checkout claims a reproducibility the result does
    not have — the code that ran is not the code at that commit.
    """
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=7"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def build_command(job: Job, args: argparse.Namespace, results_dir: Path) -> list[str]:
    out_dir = str(results_dir / job.engine)
    if job.engine == "cbls":
        return [
            str(args.cbls_bin),
            "--instance",
            job.instance,
            "--inst-dir",
            str(args.inst_dir),
            "--out-dir",
            out_dir,
            "--budget",
            str(args.budget),
            "--seed",
            str(args.seed),
            "--inf-clamp",
            str(args.inf_clamp),
            "--compound-moves" if args.compound_moves else "--no-compound-moves",
            "--commit",
            args.commit,
        ]
    return [
        sys.executable,
        str(CPSAT_SCRIPT),
        "--instance",
        job.instance,
        "--inst-dir",
        str(args.inst_dir),
        "--out-dir",
        out_dir,
        "--budget",
        str(args.budget),
        "--seed",
        str(args.seed),
        "--workers",
        str(args.cpsat_workers),
    ]


def with_memory_limit(command: list[str], limit_gb: float | None) -> list[str]:
    """Wrap a command so the child caps its own address space before exec.

    Not `preexec_fn`: this driver runs jobs from a thread pool, and preexec_fn in a
    multithreaded parent can deadlock the child between fork and exec — the one
    failure mode an unattended multi-hour run must not have. `ulimit` in the
    intermediate shell does the same job with no fork-safety question. `"$0" "$@"`
    passes the argv through without re-quoting it.
    """
    if not limit_gb:
        return command
    limit_kb = int(limit_gb * 1024 * 1024)
    # `&&`, not `;`: if the limit cannot be set (a lower hard limit already in
    # force), the job must fail loudly rather than run uncapped.
    return ["/bin/sh", "-c", f'ulimit -v {limit_kb} && exec "$0" "$@"', *command]


def write_failure_result(
    job: Job, results_dir: Path, status: str, message: str, budget: float
) -> None:
    """Record a job the driver killed, so it scores as a failure rather than as unrun."""
    path = job.result_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "engine": job.engine,
                "instance": job.instance,
                "status": status,
                "message": message,
                "objective": None,
                "budget_seconds": budget,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


#: Markers run_job puts on a line that did not produce an honest search result.
FAILURE_MARKERS = ("TIMEOUT", "FAILED", "DRIVER-ERROR")


def run_job(job: Job, args: argparse.Namespace, results_dir: Path) -> str:
    """Run one job. Nothing escapes.

    `pool.map` re-raises a worker's exception where the caller iterates it, which
    cancels every queued job and aborts main() — including the large-instance serial
    tail — with a traceback. On an unattended multi-hour run a single transient
    OSError (a fork under memory pressure, a full disk) must cost one job, not the
    remainder of the roster.
    """
    try:
        return _run_job(job, args, results_dir)
    except Exception as exc:  # noqa: BLE001 - deliberate catch-all; see docstring
        with contextlib.suppress(OSError):
            # If even this fails there is nothing left to do; the returned line
            # still reports the failure.
            write_failure_result(job, results_dir, "killed", f"driver error: {exc!r}", args.budget)
        return f"{job.engine}/{job.instance}: DRIVER-ERROR {exc!r}"


def _run_job(job: Job, args: argparse.Namespace, results_dir: Path) -> str:
    (results_dir / job.engine).mkdir(parents=True, exist_ok=True)
    command = with_memory_limit(build_command(job, args, results_dir), args.mem_limit_gb)

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=args.budget + TIMEOUT_SLACK_SECONDS,
            # Own process group: without it a Ctrl-C reaches every in-flight child,
            # each of which then leaves a "killed" result that resume treats as done
            # — permanently converting those instances to a Primal Integral of 2.
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        write_failure_result(
            job,
            results_dir,
            "killed",
            f"exceeded {args.budget + TIMEOUT_SLACK_SECONDS}s wall clock",
            args.budget,
        )
        return f"{job.engine}/{job.instance}: TIMEOUT"

    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        if not job.result_path(results_dir).exists():
            # Non-zero with no result of its own: killed by the OOM killer or the
            # address-space limit, or the instance was absent.
            write_failure_result(
                job,
                results_dir,
                "killed",
                f"exit {completed.returncode}: {completed.stderr.strip()[:400]}",
                args.budget,
            )
        # A job that wrote a result and *then* died still died. Reporting that as
        # "done" is how a systematic crash goes unnoticed for a whole roster.
        return (
            f"{job.engine}/{job.instance}: FAILED (exit {completed.returncode}) "
            f"{completed.stderr.strip()[:200]}"
        )
    return f"{job.engine}/{job.instance}: {completed.stdout.strip() or 'done'} [{elapsed:.1f}s]"


def plan_jobs(
    instances: list[str], engines: tuple[str, ...], sizes: dict[str, int], large_bytes: int
) -> tuple[list[Job], list[Job]]:
    """Split jobs into the parallel batch and the large ones that run alone."""
    normal: list[Job] = []
    large: list[Job] = []
    for instance in instances:
        target = large if sizes.get(instance, 0) >= large_bytes else normal
        target.extend(Job(engine, instance) for engine in engines)
    return normal, large


def has_usable_result(job: Job, results_dir: Path) -> bool:
    """Whether `job` can be skipped on resume.

    A result truncated by an OOM kill or a reboot mid-write must not count: the
    driver would make the damage permanent, and scoring would later abort on the
    unparseable file.
    """
    path = job.result_path(results_dir)
    if not path.exists():
        return False
    try:
        json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        print(f"Re-running {job.engine}/{job.instance}: unreadable result file.")
        return False
    return True


def drop_completed(
    normal: list[Job], large: list[Job], results_dir: Path, force: bool
) -> tuple[list[Job], list[Job]]:
    """Filter out jobs already done, or clear their results when forcing."""
    if force:
        # Drop the old results first. A forced re-run that dies before writing would
        # otherwise leave the previous run's result in place — possibly from another
        # budget — with nothing downstream able to tell it apart from a fresh one.
        for job in normal + large:
            job.result_path(results_dir).unlink(missing_ok=True)
        return normal, large

    done = sum(1 for j in normal + large if has_usable_result(j, results_dir))
    if done:
        print(f"Resuming: {done} jobs already have results.")
    return (
        [j for j in normal if not has_usable_result(j, results_dir)],
        [j for j in large if not has_usable_result(j, results_dir)],
    )


def execute(jobs: list[Job], args: argparse.Namespace, results_dir: Path, workers: int) -> int:
    """Run `jobs`, printing one line each; returns how many did not succeed."""
    if not jobs:
        return 0
    failures = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in pool.map(lambda job: run_job(job, args, results_dir), jobs):
            if any(marker in line for marker in FAILURE_MARKERS):
                failures += 1
            print(line, flush=True)
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default="smoke", help="'smoke', 'full', or a path to a CSV")
    parser.add_argument("--budget", type=float, default=600.0, help="seconds per instance-solver")
    parser.add_argument("--jobs", type=int, default=1, help="concurrent jobs")
    parser.add_argument(
        "--mem-limit-gb", type=float, default=None, help="address-space cap per job"
    )
    parser.add_argument("--results-dir", default=str(REPO_ROOT / "results" / "mipfeas"))
    parser.add_argument("--inst-dir", default=str(DEFAULT_INSTANCE_DIR))
    parser.add_argument("--cbls-bin", default=str(DEFAULT_CBLS_BIN))
    parser.add_argument("--cpsat-workers", type=int, default=1)
    parser.add_argument(
        "--inf-clamp",
        type=float,
        default=1.0e7,
        help="finite box CBLS clamps infinite variable bounds to; a CBLS-side "
        "restriction the baseline does not share (CP-SAT does not truncate variable "
        "domains), recorded per result as n_clamped_bounds",
    )
    parser.add_argument(
        "--no-compound-moves",
        dest="compound_moves",
        action="store_false",
        help="disable CBLS Novelty Jump; on by default here because roughly half of "
        "CP-SAT's incumbents on this roster come from its compound-move subsolvers",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--engines", nargs="+", choices=ENGINES, default=list(ENGINES))
    parser.add_argument("--large-bytes", type=int, default=DEFAULT_LARGE_BYTES)
    parser.add_argument(
        "--force", action="store_true", help="re-run jobs that already have results"
    )
    args = parser.parse_args()
    if args.budget <= 0 or args.jobs < 1:
        # A non-positive budget makes every runner return instantly with a
        # "no_solution" result, which resume then treats as work completed.
        print(
            f"--budget must be > 0 and --jobs >= 1 (got {args.budget}, {args.jobs}).",
            file=sys.stderr,
        )
        return 2
    args.commit = commit_sha()
    args.inst_dir = Path(args.inst_dir)
    args.cbls_bin = Path(args.cbls_bin)

    roster_path = resolve_roster(args.roster, args.inst_dir)
    if not roster_path.exists():
        print(f"Roster {roster_path} not found; run download.py first.", file=sys.stderr)
        return 2
    if "cbls" in args.engines and not args.cbls_bin.exists():
        print(f"{args.cbls_bin} not found; build the cbls_mipfeas target first.", file=sys.stderr)
        return 2

    instances = read_roster(roster_path)
    missing = [i for i in instances if not (args.inst_dir / f"{i}.mps.gz").exists()]
    if missing:
        print(
            f"{len(missing)} of {len(instances)} roster instances are absent "
            f"(e.g. {missing[:3]}). Run:\n  python {args.inst_dir}/download.py",
            file=sys.stderr,
        )
        return 2

    results_dir = Path(args.results_dir)
    engines = tuple(args.engines)
    sizes = read_sizes(args.inst_dir / "manifest.csv")
    normal, large = plan_jobs(instances, engines, sizes, args.large_bytes)

    normal, large = drop_completed(normal, large, results_dir, force=args.force)

    total = len(normal) + len(large)
    print(
        f"Roster {roster_path.name}: {len(instances)} instances x {len(engines)} engines "
        f"= {total} jobs to run at {args.budget}s, {args.jobs} at a time "
        f"({len(large)} large jobs run alone at the end)."
    )
    started = time.monotonic()
    failures = execute(normal, args, results_dir, args.jobs)
    failures += execute(large, args, results_dir, 1)
    print(
        f"\nDone in {(time.monotonic() - started) / 60:.1f} min -> {results_dir} "
        f"({total - failures}/{total} jobs succeeded)"
    )
    if failures:
        # Non-zero exit, so an unattended run's wrapper can tell "finished" from
        # "finished having failed every job" — otherwise indistinguishable.
        print(f"{failures} of {total} jobs failed; see the lines above.", file=sys.stderr)
    # Score beside the results, not into the instance directory: both
    # comparison.csv and smoke_comparison.csv there are committed, README-cited
    # artifacts, and following a printed command must not be able to overwrite one
    # with a half-finished run (issue #103).
    out_name = "comparison.csv" if roster_path.name == "roster.csv" else "smoke_comparison.csv"
    print(
        "Score it with:\n"
        f"  python {Path(__file__).parent}/primal_integral.py "
        f"--results-dir {results_dir} --roster {roster_path} --budget {args.budget} "
        f"--out {results_dir}/{out_name}\n"
        f"Then copy it to {args.inst_dir}/{out_name} if it is the run you mean to publish."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
