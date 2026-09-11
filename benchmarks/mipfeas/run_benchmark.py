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
  without a passing verdict publishes no objective and no score;
* the run's preconditions are checked before the first job rather than trusted
  (issue #137): every roster instance and every reference file is hashed against
  its pin, and the CP-SAT baseline is preflighted on one tiny in-memory model. A
  substituted instance and an OR-Tools release that moved a subsolver flag or a
  log line are both silent failures that only show at scoring time;
* the machine is recorded. `run_record.json` lands beside the results holding one
  entry per invocation -- host, cores, memory, the concurrency and memory cap the
  run used, the engine commit and solver versions, the budget, and which
  yardstick file the gaps will be scored against. A budgeted comparison is a
  statement about a machine as much as about an algorithm, and a resumed run adds
  an entry rather than overwriting the one before it.

Usage:
    python run_benchmark.py --roster smoke --budget 60 --jobs 2
    python run_benchmark.py --roster full --budget 600 --jobs 4 --mem-limit-gb 6
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
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

#: Where the machine record for a results directory lives. One file per results
#: directory, holding one entry per invocation: a resumed run is a second machine
#: and a second concurrency, and overwriting the first would claim the whole set
#: came off one.
RUN_RECORD_FILENAME = "run_record.json"

#: Wall clock the CP-SAT preflight subprocess gets. Its solve is capped at 2s and
#: the rest is an OR-Tools import, so this is generous by an order of magnitude and
#: exists only so a hang cannot become the run's first six hours.
PREFLIGHT_TIMEOUT_SECONDS = 120.0

#: Pin tables read before a run starts. `manifest.csv` pins the instance bytes and
#: `references.csv` the yardstick every gap is scored against; both are written by
#: `benchmarks/instances/mipfeas/download.py`.
MANIFEST_FILENAME = "manifest.csv"
REFERENCES_FILENAME = "references.csv"

#: Files `references.csv` pins. Duplicated from the acquisition script rather than
#: imported, for the same reason `write_failure_verdict` duplicates a shape: this
#: driver is run as a script, from any directory, and `--inst-dir` may point at a
#: roster directory that is not the one in this repository -- so it must not
#: depend on the package being importable.
PINNED_REFERENCE_FILES: tuple[str, ...] = (
    "miplib2017-v36.solu",
    "roster.csv",
    "smoke.csv",
)


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


def read_pins(path: Path, key_column: str) -> dict[str, tuple[str, int]]:
    """`{key: (sha256, bytes)}` from a pin table, empty when the file is absent."""
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {row[key_column]: (row["sha256"], int(row["bytes"])) for row in csv.DictReader(fh)}


def _pin_complaint(label: str, path: Path, pinned: tuple[str, int] | None) -> str | None:
    """The complaint about `path` against its pin, or None when it matches.

    Both halves are compared: a size-only check passes a byte-for-byte substitution
    of the same length, and a hash-only one throws away the cheapest thing to say
    about a truncated download.
    """
    if pinned is None:
        return f"{label}: present but not pinned (no recorded hash)"
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual == pinned[0] and len(data) == pinned[1]:
        return None
    return f"{label}: pinned {pinned[0]} ({pinned[1]} bytes), found {actual} ({len(data)} bytes)"


def verify_preconditions(inst_dir: Path, instances: list[str]) -> list[str]:
    """Check the roster's bytes and its yardstick against the pins, before solving.

    A corrupted, truncated or upstream-revised instance is indistinguishable from
    the one a published row was measured on, and a revised reference file moves
    every gap in the table at once. Hashing the roster costs a few seconds against
    a run measured in CPU-days, so it is done unconditionally rather than trusted.
    """
    problems: list[str] = []
    # The yardstick is checked when the directory carries one. `--inst-dir` may
    # point at a bare collection of instances (the vendored miplib-fj set, say),
    # which has no roster tables to pin; a directory that holds them and no
    # `references.csv` is the unpinned-yardstick state and is refused.
    reference_pins = read_pins(inst_dir / REFERENCES_FILENAME, "file")
    present_references = [n for n in PINNED_REFERENCE_FILES if (inst_dir / n).exists()]
    if present_references and not reference_pins:
        problems.append(
            f"{inst_dir} holds {', '.join(present_references)} but no {REFERENCES_FILENAME}: "
            f"the reference values every gap is scored against are not pinned. "
            f"Run download.py --update-references."
        )
    elif reference_pins:
        for name in PINNED_REFERENCE_FILES:
            path = inst_dir / name
            if not path.exists():
                problems.append(f"{name}: pinned but absent")
                continue
            complaint = _pin_complaint(name, path, reference_pins.get(name))
            if complaint is not None:
                problems.append(complaint)

    instance_pins = read_pins(inst_dir / MANIFEST_FILENAME, "instance")
    if not instance_pins:
        problems.append(
            f"{MANIFEST_FILENAME} is absent from {inst_dir}: no instance bytes are pinned."
        )
        return problems
    for instance in instances:
        path = inst_dir / f"{instance}.mps.gz"
        if not path.exists():
            continue  # main() reports absent instances on its own, with a fetch command
        complaint = _pin_complaint(f"{instance}.mps.gz", path, instance_pins.get(instance))
        if complaint is not None:
            problems.append(complaint)
    return problems


def run_cpsat_preflight(workers: int) -> tuple[bool, str]:
    """Ask the baseline script to assert its own preconditions. `(ok, output)`.

    Out of process because that is how every CP-SAT job runs here, so the check
    exercises the same interpreter and the same import of OR-Tools the roster will.
    One tiny in-memory model; no instance and no network.

    At the worker count the roster will actually use: the announcement carries a
    multiplicity at two or more (`fj(2)`) and the thread-count assertion is about
    the share of CPU the baseline gets against CBLS's one thread, so preflighting a
    configuration the run does not use checks the wrong thing.
    """
    try:
        completed = subprocess.run(
            [sys.executable, str(CPSAT_SCRIPT), "--preflight", "--workers", str(workers)],
            capture_output=True,
            text=True,
            check=False,
            # The check exists to fail at second zero. Without a bound, a release
            # that ignores the solve deadline hangs the driver here instead --
            # a fail-open gate on the gate.
            timeout=PREFLIGHT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False, (
            f"the CP-SAT preflight did not finish within {PREFLIGHT_TIMEOUT_SECONDS}s. It "
            "solves one tiny in-memory model under a 2s limit, so a release that runs this "
            "long is not honouring the deadline the whole baseline is budgeted by."
        )
    return completed.returncode == 0, (completed.stdout + completed.stderr).strip()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def memory_total_kib() -> int | None:
    """Total RAM, or None off Linux. The record says what it could not measure."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def machine_record() -> dict[str, object]:
    """What produced a wall-clock-limited result, beyond the code that ran.

    A budgeted comparison is a statement about a machine as much as about an
    algorithm: the same roster on half the cores, or four-up instead of one-up, is
    a different measurement. Cores are reported twice because they differ under
    cgroup or taskset confinement, and that difference is exactly the sort of thing
    that makes two runs of "the same" benchmark disagree.
    """
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "memory_total_kib": memory_total_kib(),
        "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
    }


def reference_record(inst_dir: Path, roster_path: Path) -> dict[str, object]:
    """Which yardstick a table was scored against, by name and by hash.

    The roster is recorded by full path and by its own hash, not only through the
    pinned tables: `--roster` accepts any CSV, and the reference values a run is
    scored against are the ones in the file it actually read. Quoting the pinned
    hashes alone for a roster that is not one of them would name a yardstick the run
    never used.
    """
    pins = read_pins(inst_dir / REFERENCES_FILENAME, "file")
    manifest = inst_dir / MANIFEST_FILENAME
    return {
        "instance_dir": str(inst_dir),
        "solution_file": PINNED_REFERENCE_FILES[0],
        "pinned": {name: pins[name][0] for name in PINNED_REFERENCE_FILES if name in pins},
        "roster_path": str(roster_path),
        "roster_sha256": hashlib.sha256(roster_path.read_bytes()).hexdigest()
        if roster_path.exists()
        else None,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()
        if manifest.exists()
        else None,
    }


def build_run_record(
    args: argparse.Namespace,
    roster_path: Path,
    instances: list[str],
    engines: tuple[str, ...],
    planned: int,
    to_run: int,
) -> dict[str, object]:
    """The machine record published beside the results."""
    return {
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "finished_at": None,
        "status": "running",
        "machine": machine_record(),
        # The thing a wall-clock-limited result cannot be read without.
        "concurrency": {
            "jobs": args.jobs,
            "large_instance_jobs": 1,
            "cpsat_workers": args.cpsat_workers,
            "mem_limit_gb": args.mem_limit_gb,
        },
        "budget_seconds": args.budget,
        "run": {
            "roster": roster_path.name,
            "instances": len(instances),
            "engines": list(engines),
            "seed": args.seed,
            "verify": args.verify,
            # The one flag that makes a run unpublishable: with it off, the pinned
            # hashes above were never checked against the files on disk and the
            # baseline was never preflighted. Nothing else in the record would show
            # that, and stderr does not survive to whoever reads the table.
            "preconditions_checked": not args.skip_preconditions,
            "force": args.force,
            "inf_clamp": args.inf_clamp,
            "compound_moves": args.compound_moves,
            "propagate_bounds": args.propagate_bounds,
            "large_bytes": args.large_bytes,
            "jobs_planned": planned,
            "jobs_to_run": to_run,
        },
        "versions": {
            "engine_commit": args.commit,
            "cbls_binary": str(args.cbls_bin),
            "python": platform.python_version(),
            "ortools": package_version("ortools"),
            "pyscipopt": package_version("PySCIPOpt"),
        },
        "references": reference_record(args.inst_dir, roster_path),
        "outcome": None,
    }


def append_run_record(results_dir: Path, record: dict[str, object]) -> Path:
    """Add `record` to the results directory's run log, keeping earlier entries."""
    path = results_dir / RUN_RECORD_FILENAME
    existing = _read_json(path) or {}
    previous = existing.get("runs")
    runs = list(previous) if isinstance(previous, list) else []
    runs.append(record)
    _write_run_records(path, runs)
    return path


def close_run_record(results_dir: Path, record: dict[str, object]) -> None:
    """Replace the entry `append_run_record` added with its finished form."""
    path = results_dir / RUN_RECORD_FILENAME
    existing = _read_json(path) or {}
    previous = existing.get("runs")
    runs = list(previous) if isinstance(previous, list) else []
    if runs:
        runs[-1] = record
    else:
        runs = [record]
    _write_run_records(path, runs)


def _write_run_records(path: Path, runs: list[object]) -> None:
    # Temp-then-rename, like every other file this benchmark writes: a driver
    # killed mid-write must leave the previous record rather than a truncated one.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def check_preconditions(
    args: argparse.Namespace, instances: list[str], engines: tuple[str, ...]
) -> int | None:
    """Refuse to start a run whose inputs or baseline are not what they claim.

    Returns an exit code to stop on, or None to proceed. Both failures it looks
    for are silent: a substituted instance measures a different program under a
    published row's name, and an OR-Tools release that moved a subsolver flag or a
    log line produces a degraded baseline across all 233 instances at exit 0,
    discovered only at scoring time (issue #137).
    """
    if args.skip_preconditions:
        print(
            "WARNING: --skip-preconditions: instance bytes are unchecked and the CP-SAT "
            "baseline is unverified. This run is not publishable.",
            file=sys.stderr,
        )
        return None

    problems = verify_preconditions(args.inst_dir, instances)
    if problems:
        print(
            f"\n{len(problems)} pinned file(s) in {args.inst_dir} do not match what is recorded:",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(
            "\nA published row measured the pinned bytes, not these. Restore them, or "
            "re-pin deliberately -- `--verify` only re-reports this mismatch and exits "
            "1:\n"
            f"  python {args.inst_dir}/download.py --update-manifest      # instance bytes\n"
            f"  python {args.inst_dir}/download.py --update-references    # the yardstick",
            file=sys.stderr,
        )
        return 2
    print(
        f"Preconditions: {len(instances)} instance(s) and the reference files match "
        f"their pinned bytes."
    )
    if "cpsat" not in engines:
        return None
    ok, output = run_cpsat_preflight(args.cpsat_workers)
    print(output.splitlines()[0] if output else "CP-SAT preflight produced no output")
    if ok:
        return None
    print(
        "\n" + output + "\n\nRefusing to start: the CP-SAT baseline would be degraded or "
        "empty across the whole roster and would only show it at scoring time.",
        file=sys.stderr,
    )
    return 2


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


def count_unchecked(jobs: list[Job], results_dir: Path, verify: bool) -> int:
    """Feasible rows that never got a verdict saying they were actually checked.

    A row whose check the driver could not complete is retried up to
    `MAX_VERIFY_ATTEMPTS` and then stops being retried -- at which point
    `needs_verification` is false, `has_usable_result` is true and the job is
    dropped from every later resume. From the third pass onward such a directory
    prints "0 jobs to run" and the driver exits 0, reporting as a clean run a row
    that was never successfully checked and whose objective the scorer withholds.
    Counted over every planned job, like `count_rejected`, and for the same reason.

    `fail` is excluded because it *was* checked, and is counted (and shouted about)
    separately.
    """
    if not verify:
        return 0
    total = 0
    for job in jobs:
        result = _read_json(job.result_path(results_dir))
        if result is None or result.get("status") != "feasible":
            continue
        verdict = _read_json(job.verification_path(results_dir)) or {}
        if verdict.get("verdict") not in ("pass", "fail"):
            total += 1
    return total


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
        "--skip-preconditions",
        action="store_true",
        help="skip the pinned-bytes check and the CP-SAT preflight. For harness "
        "debugging only: the first is what stops a corrupted or substituted instance "
        "being measured as if it were the pinned one, and the second is what stops an "
        "OR-Tools release that moved a subsolver flag or a log line producing a "
        "degraded baseline across the whole roster at exit 0 (issue #137)",
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
    engines = tuple(args.engines)
    missing = [i for i in instances if not (args.inst_dir / f"{i}.mps.gz").exists()]
    if missing:
        print(
            f"{len(missing)} of {len(instances)} roster instances are absent "
            f"(e.g. {missing[:3]}). Run:\n  python {args.inst_dir}/download.py",
            file=sys.stderr,
        )
        return 2

    refusal = check_preconditions(args, instances, engines)
    if refusal is not None:
        return refusal

    results_dir = Path(args.results_dir)
    sizes = read_sizes(args.inst_dir / MANIFEST_FILENAME)
    normal, large = plan_jobs(instances, engines, sizes, args.large_bytes)
    planned = normal + large

    normal, large = drop_completed(normal, large, results_dir, force=args.force, verify=args.verify)

    total = len(normal) + len(large)
    print(
        f"Roster {roster_path.name}: {len(instances)} instances x {len(engines)} engines "
        f"= {total} jobs to run at {args.budget}s, {args.jobs} at a time "
        f"({len(large)} large jobs run alone at the end)."
    )
    # Written before the first job, not after the last: a run killed at hour six
    # still has to say what machine and what concurrency produced the results it
    # did leave behind.
    record = build_run_record(args, roster_path, instances, engines, len(planned), total)
    record_path = append_run_record(results_dir, record)
    print(f"Machine record -> {record_path}")

    started = time.monotonic()
    failures = execute(normal, args, results_dir, args.jobs)
    failures += execute(large, args, results_dir, 1)
    elapsed = time.monotonic() - started
    print(
        f"\nDone in {elapsed / 60:.1f} min -> {results_dir} "
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
    unchecked = count_unchecked(planned, results_dir, args.verify)
    record["status"] = "complete"
    record["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    record["outcome"] = {
        "jobs_run": total,
        "failures": failures,
        "rejected": rejected,
        "unchecked": unchecked,
        "wall_seconds": round(elapsed, 3),
    }
    close_run_record(results_dir, record)
    if rejected:
        print(
            f"\nDEFECT: {rejected} solution(s) in {results_dir} were rejected by the "
            f"independent check against the instance file. Score the run to see "
            f"which, or read the .verify.json files.",
            file=sys.stderr,
        )
    if unchecked:
        # Same shape as `rejected`, and counted over every planned job for the same
        # reason: a resume that runs nothing must not report a clean run.
        print(
            f"\nUNCHECKED: {unchecked} feasible row(s) in {results_dir} carry no verdict "
            f"saying they were checked -- the verification was exhausted after "
            f"{MAX_VERIFY_ATTEMPTS} attempts, refused to run, or never happened. Those "
            f"rows publish no objective. Read their .verify.json files, or re-run them "
            f"with --force once the cause is fixed.",
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
        f"Then copy it to {args.inst_dir}/{out_name} if it is the run you mean to "
        f"publish, along with {results_dir}/{RUN_RECORD_FILENAME} and the "
        f"{out_name.removesuffix('.csv')}_report.md beside it -- the report quotes the "
        f"machine record, and a published table without one is an anecdote."
        + (
            ""
            if args.verify
            else "\nThis run skipped verification, so scoring it needs "
            "--allow-unverified and the table it writes is not publishable."
        )
    )
    return 1 if failures or rejected or unchecked else 0


if __name__ == "__main__":
    sys.exit(main())
