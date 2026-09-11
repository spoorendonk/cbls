"""Drive a MIPfeas run: every roster instance against every engine.

Built for an unattended multi-hour run on a bigger machine, so:

* one process per (instance, engine) — a job that dies takes only itself;
* resumable — a job that already has a result *and* a verdict is skipped, so an
  interrupted run continues where it stopped, and a results directory that was
  solved but not checked is verified without re-solving;
* size-aware — the largest instances run on their own after the rest, instead of
  four-up against a memory limit;
* every job is bounded by a wall-clock timeout and, optionally, an address-space
  limit, and a job killed by either leaves a result recording that;
* every feasible solution is checked against the ORIGINAL instance file by
  `verify_solution.py`, which shares no reader with either engine (issue #138).
  The verdict lands beside the result as `<instance>.verify.json`, and a row
  without a passing verdict publishes no objective and no score.

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
VERIFY_SCRIPT = Path(__file__).resolve().parent / "verify_solution.py"

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

#: Wall clock a verification gets. Its own constant rather than the slack above,
#: which is a budget *overhead* justified by model build: this is an absolute
#: budget for reading the instance a second time and summing every nonzero in
#: Python, on models up to 27.4M nonzeros.
VERIFY_TIMEOUT_SECONDS = 900.0


@dataclass(frozen=True)
class Job:
    engine: str
    instance: str

    def result_path(self, results_dir: Path) -> Path:
        return results_dir / self.engine / f"{self.instance}.json"

    def solution_path(self, results_dir: Path) -> Path:
        return results_dir / self.engine / f"{self.instance}.sol"

    def verification_path(self, results_dir: Path) -> Path:
        return results_dir / self.engine / f"{self.instance}.verify.json"


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
    """The runner invocation for one job.

    The solution directory is the result directory: `verify_solution.py` reads
    `<instance>.sol` and `<instance>.json` from one place, and keeping them
    together is what makes a resumed run able to tell a verified row from an
    unverified one.
    """
    out_dir = str(results_dir / job.engine)
    solution_flags = ["--solution-dir", out_dir] if args.verify else []
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
            *([] if args.propagate_bounds else ["--no-propagate-bounds"]),
            *solution_flags,
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
        *solution_flags,
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


#: Reasons only the driver writes, for a checker that died rather than a solution
#: that was checked. Retried on resume up to MAX_VERIFY_ATTEMPTS; every verdict
#: the verifier writes itself is final.
DRIVER_WRITTEN_VERDICT_REASONS = ("verifier_timeout", "verifier_died")

#: How many times a driver-written verdict is retried before it sticks.
#:
#: The transient causes these reasons were introduced for -- a memory cap hit
#: under load, a timeout on a busy machine -- clear on a second pass. The
#: deterministic ones do not: a pyscipopt segfault, or an OOM on a model that
#: simply does not fit the cap, reproduces every time, and retrying it forever
#: re-pays the whole verification on every resume of a roster that can never
#: converge. Two attempts is the smallest number that still recovers a transient
#: failure; past that the verdict stands and the row stays withheld, which is the
#: honest outcome.
#:
#: A verdict written before this counter existed carries no `attempts` key and is
#: read as zero, so such a directory gets one extra pass. One-off and
#: self-healing: the first retry writes the key.
MAX_VERIFY_ATTEMPTS = 2


def write_failure_verdict(
    job: Job, results_dir: Path, reason: str, message: str, attempts: int = 1
) -> None:
    """Record a verification the driver could not complete.

    Mirrors `write_failure_result`: a verifier killed by the timeout or the
    memory cap writes nothing itself, and a row with no verdict file at all is
    indistinguishable from one nobody tried to check. The shape is the subset of
    `verify_solution.Verification` the scorer reads; it is written here rather
    than imported because this driver is run as a script, from any directory,
    and must not depend on the package being importable.
    """
    path = job.verification_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "instance": job.instance,
                "engine": job.engine,
                "verdict": "error",
                "reason": reason,
                "message": message,
                "marginal": False,
                "failed_checks": [],
                # How many times the driver has now tried and failed to check this
                # row. Read back by `needs_verification` so a deterministic failure
                # stops being retried; absent from the verifier's own verdicts,
                # which are final on the first pass.
                "attempts": attempts,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


#: Markers run_job puts on a line that did not produce an honest search result.
#: A rejected solution is one of them: this is the correctness benchmark, and a
#: run that published a point the checker refused must not exit 0.
FAILURE_MARKERS = ("TIMEOUT", "FAILED", "DRIVER-ERROR", "VERIFY-FAILED", "VERIFY-ERROR")


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
            # Only when the job has no result of its own. This catch-all now spans
            # the verification step too, and a failure there must not overwrite a
            # finished search: resume, seeing a "killed" record, would never redo
            # it, so a 600s solve would be discarded and the row scored 2.0. Left
            # alone, the result is simply re-verified on the next pass.
            if not job.result_path(results_dir).exists():
                write_failure_result(
                    job, results_dir, "killed", f"driver error: {exc!r}", args.budget
                )
        return f"{job.engine}/{job.instance}: DRIVER-ERROR {exc!r}"


def _run_job(job: Job, args: argparse.Namespace, results_dir: Path) -> str:
    """Solve the job if it still needs solving, then verify what it produced.

    Two steps rather than one because they resume independently: a results
    directory whose solves are done but whose verdicts are missing must be
    verifiable without paying for the search again.
    """
    (results_dir / job.engine).mkdir(parents=True, exist_ok=True)
    line = f"{job.engine}/{job.instance}: already solved"
    if needs_solve(job, results_dir, args.verify):
        line, solved = _run_solver(job, args, results_dir)
        if not solved:
            return line
        # The search just produced a new point, so any verdict sitting beside it
        # describes the previous one and would read as current.
        job.verification_path(results_dir).unlink(missing_ok=True)
    if needs_verification(job, results_dir, args.verify):
        line = f"{line} | {_verify(job, args, results_dir)}"
    return line


def _verify(job: Job, args: argparse.Namespace, results_dir: Path) -> str:
    """Check the job's solution against the original instance file.

    Runs out of process under the same memory cap as the solve: it reads the
    instance a second time, with a different reader, and on the largest models
    that is not a small allocation. A verifier that dies leaves an explicit error
    verdict, because the scorer must be able to tell "checked and rejected" from
    "never checked" from "the checker crashed".
    """
    command = with_memory_limit(
        [
            sys.executable,
            str(VERIFY_SCRIPT),
            "--instance",
            job.instance,
            "--inst-dir",
            str(args.inst_dir),
            "--result-dir",
            str(results_dir / job.engine),
        ],
        args.mem_limit_gb,
    )
    # The verdict this attempt is retrying, if any, and then out of the way. A
    # stale file left in place is indistinguishable from one this run wrote: the
    # verifier exits 1 both for "I checked it and it is wrong" and, as any Python
    # program does, for an uncaught traceback that writes nothing -- and a
    # traceback landing next to a previous attempt's `verifier_died` would be
    # reported as a rejected solution, which is this benchmark's loudest alarm
    # fired for a harness fault. Only a driver-written verdict can be here at all
    # -- `needs_verification` is false for every verdict the verifier reached
    # itself -- so a `fail` is never at risk of being removed.
    #
    # The attempt count lives in the file this removes, so a driver killed during
    # the check restarts the budget. The cost is bounded re-work on the next
    # resume, never a wrong verdict, which is the right way round.
    previous = _read_json(job.verification_path(results_dir)) or {}
    attempts = previous.get("attempts")
    attempt = (attempts if isinstance(attempts, int) else 0) + 1
    job.verification_path(results_dir).unlink(missing_ok=True)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=VERIFY_TIMEOUT_SECONDS,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        write_failure_verdict(
            job, results_dir, "verifier_timeout", f"exceeded {VERIFY_TIMEOUT_SECONDS}s", attempt
        )
        return "VERIFY-ERROR timeout"
    if completed.returncode == 0:
        return f"verified {completed.stdout.strip()[:200]}"
    if completed.returncode == 1 and job.verification_path(results_dir).exists():
        # Exit 1 is the verifier's "I checked it and it is wrong", and it wrote
        # the verdict itself -- this attempt's, since any earlier one was removed
        # above. A traceback exits 1 too and writes nothing, so it falls through
        # to the error path below instead of being reported as a rejected
        # solution.
        return f"VERIFY-FAILED {completed.stdout.strip()[:200]}"
    if not job.verification_path(results_dir).exists():
        # A negative return code is a signal: the address-space cap or the OOM
        # killer. Said in the message because it is the difference between "try
        # again with more memory" and "this checker crashes on this model".
        how = (
            f"signal {-completed.returncode}"
            if completed.returncode < 0
            else f"exit {completed.returncode}"
        )
        write_failure_verdict(
            job,
            results_dir,
            "verifier_died",
            f"{how} (attempt {attempt} of {MAX_VERIFY_ATTEMPTS}): {completed.stderr.strip()[:400]}",
            attempt,
        )
    return f"VERIFY-ERROR (exit {completed.returncode}) {completed.stderr.strip()[:200]}"


def _run_solver(job: Job, args: argparse.Namespace, results_dir: Path) -> tuple[str, bool]:
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
        return f"{job.engine}/{job.instance}: TIMEOUT", False

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
            f"{completed.stderr.strip()[:200]}",
            False,
        )
    return (
        f"{job.engine}/{job.instance}: {completed.stdout.strip() or 'done'} [{elapsed:.1f}s]",
        True,
    )


def count_rejected(jobs: list[Job], results_dir: Path) -> int:
    """How many of `jobs` carry a verdict that rejected the engine's solution."""
    return sum(
        1
        for job in jobs
        if (_read_json(job.verification_path(results_dir)) or {}).get("verdict") == "fail"
    )


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


def _read_json(path: Path) -> dict[str, object] | None:
    """The object at `path`, or None when it is absent or unreadable.

    A file truncated by an OOM kill or a reboot mid-write reads as absent: the
    driver would otherwise make the damage permanent, and scoring would later
    abort on the unparseable file.
    """
    if not path.exists():
        return None
    try:
        parsed = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return parsed if isinstance(parsed, dict) else None


def needs_solve(job: Job, results_dir: Path, verify: bool) -> bool:
    """Whether the search still has to run for `job`."""
    result = _read_json(job.result_path(results_dir))
    if result is None:
        return True
    if verify and result.get("status") == "solution_write_error":
        # The search found a point and only the dump failed (a full disk, a
        # read-only directory). Nothing else can produce the solution vector, and
        # the scorer withholds the row until one exists.
        print(f"Re-running {job.engine}/{job.instance}: the solution could not be written.")
        return True
    if (
        verify
        and result.get("status") == "feasible"
        and not job.solution_path(results_dir).exists()
    ):
        # Nothing else can produce the solution vector, and a feasible row
        # without one can never earn a verdict — so the search has to run again.
        # Reachable by ordinary use: a directory filled before #138, or one whose
        # earlier pass ran with --no-verify.
        print(f"Re-running {job.engine}/{job.instance}: feasible result with no solution file.")
        return True
    return False


def needs_verification(job: Job, results_dir: Path, verify: bool) -> bool:
    """Whether `job` still needs an independent verdict.

    Only a feasible row does: there is nothing to check about a run that found
    no solution, and its objective is already absent.
    """
    if not verify:
        return False
    result = _read_json(job.result_path(results_dir))
    if result is None or result.get("status") != "feasible":
        return False
    verdict = _read_json(job.verification_path(results_dir))
    if verdict is None:
        return True
    # The driver's own error verdicts describe a checker that died, not a solution
    # that was checked. Without a retry the row is withheld for good, recoverable
    # only by --force, which pays for the whole search again -- so a transient
    # cause (a memory cap hit under load, a timeout on a busy machine) gets
    # another pass. Only MAX_VERIFY_ATTEMPTS of them: the same reasons are written
    # for deterministic causes too, and retrying a segfault on every resume of a
    # 233-instance roster never converges. The verifier's own verdicts --
    # including its `error`s -- are sticky from the first pass.
    if verdict.get("reason") not in DRIVER_WRITTEN_VERDICT_REASONS:
        return False
    attempts = verdict.get("attempts")
    return (attempts if isinstance(attempts, int) else 0) < MAX_VERIFY_ATTEMPTS


def has_usable_result(job: Job, results_dir: Path, verify: bool = False) -> bool:
    """Whether `job` can be skipped on resume: solved, and verified if required."""
    return not needs_solve(job, results_dir, verify) and not needs_verification(
        job, results_dir, verify
    )


def drop_completed(
    normal: list[Job], large: list[Job], results_dir: Path, force: bool, verify: bool = False
) -> tuple[list[Job], list[Job]]:
    """Filter out jobs already done, or clear their results when forcing."""
    if force:
        # Drop the old results first. A forced re-run that dies before writing would
        # otherwise leave the previous run's result in place — possibly from another
        # budget — with nothing downstream able to tell it apart from a fresh one.
        # The solution and its verdict go with it: a stale verdict describing the
        # previous run's point is worse than none, since it reads as current.
        for job in normal + large:
            job.result_path(results_dir).unlink(missing_ok=True)
            job.solution_path(results_dir).unlink(missing_ok=True)
            job.verification_path(results_dir).unlink(missing_ok=True)
        return normal, large

    done = sum(1 for j in normal + large if has_usable_result(j, results_dir, verify))
    if done:
        print(f"Resuming: {done} jobs already have results.")
    return (
        [j for j in normal if not has_usable_result(j, results_dir, verify)],
        [j for j in large if not has_usable_result(j, results_dir, verify)],
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
        help="finite box CBLS clamps a variable bound to when no constraint implies "
        "one; a CBLS-side restriction the baseline does not share (CP-SAT does not "
        "truncate variable domains), recorded per result as n_clamped_bounds",
    )
    parser.add_argument(
        "--no-propagate-bounds",
        dest="propagate_bounds",
        action="store_false",
        help="disable implied-bound derivation, so every unbounded column falls back "
        "on --inf-clamp. Isolates propagation only: #120 also stopped the clamp "
        "narrowing finite bounds, and that half is unconditional. Not a "
        "configuration to publish, since the clamp it restores is not implied by "
        "the constraints",
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
    parser.add_argument(
        "--no-verify",
        dest="verify",
        action="store_false",
        help="skip the independent feasibility check of every reported solution "
        "against the original instance file. Off only for harness debugging: the "
        "scorer then publishes no objective for any feasible row unless it is run "
        "with --allow-unverified, because an unchecked row is what #138 exists to "
        "stop publishing",
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
    planned = normal + large

    normal, large = drop_completed(normal, large, results_dir, force=args.force, verify=args.verify)

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
    # Counted over every planned job rather than only the ones this invocation
    # ran, and kept out of the job tally above: a resume of a directory that
    # already holds a rejected solution runs nothing, and would otherwise exit 0
    # and tell an unattended wrapper the run was clean.
    rejected = count_rejected(planned, results_dir)
    if rejected:
        print(
            f"\nDEFECT: {rejected} solution(s) in {results_dir} were rejected by the "
            f"independent check against the instance file. Score the run to see "
            f"which, or read the .verify.json files.",
            file=sys.stderr,
        )
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
        + (
            ""
            if args.verify
            else "\nThis run skipped verification, so scoring it needs "
            "--allow-unverified and the table it writes is not publishable."
        )
    )
    return 1 if failures or rejected else 0


if __name__ == "__main__":
    sys.exit(main())
