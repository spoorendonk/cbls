"""Tests for the MIPfeas run driver.

The mechanics every driver shares -- the process runner, the memory cap, atomic
records -- are pinned once in `test_benchmark_common.py`. What is here is this
driver's policy: the invocations, what resume skips, what a failure records, and
what makes a run refuse to start or refuse to exit 0.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.instances.mipfeas.download import PINNED_REFERENCE_FILES
from benchmarks.mipfeas import run_benchmark
from benchmarks.mipfeas.primal_integral import NO_SOLUTION_GAP, score_instance, summarize
from benchmarks.mipfeas.run_benchmark import (
    FAILURE_MARKERS,
    MAX_VERIFY_ATTEMPTS,
    Job,
    build_command,
    count_rejected,
    count_unchecked,
    drop_completed,
    execute,
    has_usable_result,
    needs_solve,
    needs_verification,
    plan_jobs,
    read_roster,
    read_sizes,
    resolve_roster,
    write_failure_result,
)

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
BINARY = REPO_ROOT / "build" / "cbls_mipfeas"


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _binary() -> Path:
    if not BINARY.exists():
        pytest.skip("cbls_mipfeas not built")
    return BINARY


# --- the roster and the plan ----------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [("smoke", "smoke.csv"), ("full", "roster.csv"), ("mine.csv", "mine.csv")],
)
def test_resolve_roster_maps_the_named_rosters_and_passes_a_path_through(
    tmp_path: Path, value: str, expected: str
) -> None:
    given = value if value in ("smoke", "full") else str(tmp_path / value)
    assert resolve_roster(given, tmp_path) == tmp_path / expected


def test_read_roster_returns_instance_names(tmp_path: Path) -> None:
    path = tmp_path / "roster.csv"
    _write_csv(
        path,
        ["instance", "reference_value", "reference_kind"],
        [["a", 1.0, "opt"], ["b", 2.0, "best"]],
    )
    assert read_roster(path) == ["a", "b"]


def test_read_sizes_of_an_absent_manifest_is_empty(tmp_path: Path) -> None:
    assert read_sizes(tmp_path / "manifest.csv") == {}


@pytest.mark.parametrize(
    ("instances", "engines", "sizes", "normal", "large"),
    [
        (["a", "b"], ("cbls", "cpsat"), {}, ["cbls/a", "cpsat/a", "cbls/b", "cpsat/b"], []),
        # The roster spans four orders of magnitude in size; the big ones must not
        # run four-up against a memory limit.
        (
            ["small", "huge"],
            ("cbls",),
            {"small": 100, "huge": 50_000},
            ["cbls/small"],
            ["cbls/huge"],
        ),
        (["mystery"], ("cbls",), {}, ["cbls/mystery"], []),
    ],
    ids=["every-instance-with-every-engine", "large-set-aside", "unknown-size-is-small"],
)
def test_plan_jobs(
    instances: list[str],
    engines: tuple[str, ...],
    sizes: dict[str, int],
    normal: list[str],
    large: list[str],
) -> None:
    planned = plan_jobs(instances, engines, sizes, large_bytes=10_000)
    assert [[f"{j.engine}/{j.instance}" for j in part] for part in planned] == [normal, large]


def test_a_killed_job_scores_as_a_failure_not_as_unrun(tmp_path: Path) -> None:
    # A job the driver had to kill did happen, so it scores 2 like any other run
    # that produced nothing — but it must stay distinguishable from a job that was
    # never scheduled, which is excluded from the aggregate instead. The engine
    # directory does not exist yet: the failure record has to create it.
    job = Job("cbls", "inst")
    write_failure_result(job, tmp_path, "killed", "exceeded wall clock", budget=60.0)
    assert job.result_path(tmp_path) == tmp_path / "cbls" / "inst.json"

    scored = score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)
    assert scored.status == "killed"
    assert scored.primal_integral == pytest.approx(NO_SOLUTION_GAP)

    summary = summarize([scored], "cbls")
    assert summary.scored == 1
    assert summary.not_run == 0
    assert summary.feasible == 0
    assert summary.errored == 1


# --- the runner's process contract ----------------------------------------------


def test_the_runner_writes_nothing_for_an_absent_instance(tmp_path: Path) -> None:
    """The #103 guard: a missing instance must not be scored as 'found nothing'.

    Runs the real binary, because this is a property of the process contract the
    driver depends on — a non-zero exit and no result file — not of any function.
    """
    out_dir = tmp_path / "results"
    result = subprocess.run(
        [
            str(_binary()),
            *("--instance", "no-such-instance", "--inst-dir", str(tmp_path)),
            *("--out-dir", str(out_dir), "--budget", "1"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "not found" in result.stderr
    assert not (out_dir / "no-such-instance.json").exists()
    assert not (out_dir / "no-such-instance.trace.csv").exists()


def test_the_runner_rejects_a_nonpositive_budget(tmp_path: Path) -> None:
    # parse_double reports the bad flag and hands the guard a NaN rather than
    # exiting itself; solve() with no clock returns instantly, so unguarded this
    # would score a whole roster "no_solution" at exit 0. Both halves are pinned
    # deliberately: the parse layer names the flag, and the guard in run_benchmark
    # is what decides the exit code. Moving the exit into parse_double would keep
    # this test green only by accident.
    result = subprocess.run(
        [
            str(_binary()),
            *("--instance", "anything", "--out-dir", str(tmp_path), "--budget", "notanumber"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--budget: 'notanumber' is not a number" in result.stderr
    assert "--budget must be a positive" in result.stderr


# --- the invocations ------------------------------------------------------------


def _driver_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "cbls_bin": Path("/bin/cbls_mipfeas"),
        "inst_dir": Path("/instances"),
        "budget": 600.0,
        "seed": 42,
        "inf_clamp": 1.0e7,
        "compound_moves": True,
        "propagate_bounds": True,
        "cpsat_workers": 1,
        "cbls_threads": 1,
        "commit": "abc1234",
        "verify": True,
        "skip_preconditions": False,
        "jobs": 1,
        "mem_limit_gb": None,
        "force": False,
        "large_bytes": 5_000_000,
    }
    return argparse.Namespace(**{**defaults, **overrides})


@pytest.mark.parametrize(
    ("engine", "overrides", "stated", "absent"),
    [
        # Both are deliberate departures from the engine defaults, so the driver has
        # to state them: a run that silently inherited either would not be the run
        # the README describes. Propagation is the engine default, so the driver
        # says nothing about it. The thread count is always stated, 1 included: a
        # benchmark row records the concurrency it ran at, and the runner's own
        # default is not a record of anything.
        (
            "cbls",
            {},
            {"--inf-clamp": "10000000.0", "--compound-moves": None, "--threads": "1"},
            ["--no-propagate-bounds"],
        ),
        ("cbls", {"propagate_bounds": False}, {"--no-propagate-bounds": None}, []),
        ("cbls", {"compound_moves": False}, {"--no-compound-moves": None}, ["--compound-moves"]),
        ("cbls", {"cbls_threads": 4}, {"--threads": "4"}, []),
        ("cpsat", {}, {"--workers": "1"}, []),
        # Without a solution directory neither runner writes a solution vector, and
        # nothing downstream can check a reported point against its instance (#138).
        ("cbls", {}, {"--solution-dir": "/results/cbls"}, []),
        ("cpsat", {}, {"--solution-dir": "/results/cpsat"}, []),
        ("cbls", {"verify": False}, {}, ["--solution-dir"]),
    ],
    ids=[
        "cbls-defaults",
        "no-bound-propagation",
        "no-compound-moves",
        "cbls-threads",
        "cpsat-workers",
        "cbls-solution-dir",
        "cpsat-solution-dir",
        "no-solution-dir-without-verify",
    ],
)
def test_build_command_states_the_configuration(
    engine: str, overrides: dict[str, object], stated: dict[str, str | None], absent: list[str]
) -> None:
    command = build_command(Job(engine, "inst"), _driver_args(**overrides), Path("/results"))
    for flag, value in stated.items():
        assert flag in command
        if value is not None:
            assert command[command.index(flag) + 1] == value
    for flag in absent:
        assert flag not in command


def _main_with(argv: list[str]) -> int:
    saved = sys.argv
    sys.argv = ["run_benchmark.py", *argv]
    try:
        return run_benchmark.main()
    finally:
        sys.argv = saved


@pytest.mark.parametrize(
    ("argv", "said", "not_said"),
    [
        # CP-SAT at N workers runs N fj and N ls subsolvers, so an N-thread CBLS
        # against a 1-worker baseline is an N-fold CPU advantage the table would
        # read as an implementation gap. Refused before any instance is touched.
        (["--cbls-threads", "4"], "--cbls-threads 4 != --cpsat-workers 1", None),
        # The refusal guards the published claim; it is not a law of the harness.
        # An asymmetry that IS the measurement gets past it by saying so -- shown
        # by the run failing on the NEXT guard instead.
        (
            ["--cbls-threads", "4", "--allow-asymmetric-cpu", "--budget", "0"],
            "--budget must be > 0",
            "--cbls-threads 4 != --cpsat-workers 1",
        ),
    ],
    ids=["asymmetric-cpu-refused", "asymmetric-cpu-opted-into"],
)
def test_the_driver_refuses_a_run_whose_flags_cannot_be_read(
    capsys: pytest.CaptureFixture[str], argv: list[str], said: str, not_said: str | None
) -> None:
    assert _main_with(argv) == 2
    err = capsys.readouterr().err
    assert said in err
    assert not_said is None or not_said not in err


def test_resuming_at_a_different_thread_count_is_refused(tmp_path: Path) -> None:
    # Resume keys on file existence alone and a result's path carries no
    # concurrency, so re-running a 1-thread directory at 4 threads would skip
    # every job and score a uniform threads=1 table -- nothing for the scorer's
    # mixed-configuration guard to catch, and an operator who believes they
    # measured 4. Caught before the run rather than after it is paid for.
    (tmp_path / "cbls").mkdir()
    (tmp_path / "cbls" / "a.json").write_text(json.dumps({"instance": "a", "threads": 1}))
    jobs = [Job("cbls", "a")]

    def check(threads: int) -> int | None:
        args = _driver_args(cbls_threads=threads, cpsat_workers=threads)
        return run_benchmark.check_resume_configuration(jobs, tmp_path, args)

    assert check(4) == 2
    # The matching count resumes, and so does a result that predates the key:
    # it records nothing this run can contradict.
    assert check(1) is None
    (tmp_path / "cbls" / "a.json").write_text(json.dumps({"instance": "a"}))
    assert check(4) is None
    # The CP-SAT side keys on `workers`, not `threads`, and is checked separately.
    (tmp_path / "cpsat").mkdir()
    (tmp_path / "cpsat" / "a.json").write_text(json.dumps({"instance": "a", "workers": 1}))
    args = _driver_args(cbls_threads=1, cpsat_workers=4)
    assert run_benchmark.check_resume_configuration([Job("cpsat", "a")], tmp_path, args) == 2


# --- resume: what still needs a solve, and what still needs a verdict (#138) -----


def _row(
    results_dir: Path,
    status: str | None,
    *,
    solution: bool = False,
    verdict: dict[str, object] | str | None = None,
    job: Job | None = None,
) -> Job:
    """A job's files on disk: its result, its solution and its verdict, each optional."""
    job = job or Job("cbls", "inst")
    job.result_path(results_dir).parent.mkdir(parents=True, exist_ok=True)
    if status is not None:
        job.result_path(results_dir).write_text(json.dumps({"status": status, "objective": 1.0}))
    if solution:
        job.solution_path(results_dir).write_text("=obj= 1.0\nx 1\n")
    if verdict is not None:
        text = verdict if isinstance(verdict, str) else json.dumps(verdict)
        job.verification_path(results_dir).write_text(text)
    return job


def _driver_verdict(attempts: int | None) -> dict[str, object]:
    verdict: dict[str, object] = {"verdict": "error", "reason": "verifier_died"}
    return verdict if attempts is None else {**verdict, "attempts": attempts}


@pytest.mark.parametrize(
    ("status", "solution", "verdict", "verify", "solve", "check"),
    [
        (None, False, None, True, True, False),
        # A results directory filled before #138, or by a --no-verify pass: nothing
        # but the search can produce the solution vector, so resume cannot skip it.
        ("feasible", False, None, True, True, True),
        ("feasible", False, None, False, False, False),
        # ... but with its solution it still needs the verdict, and that step is
        # cheap and separate.
        ("feasible", True, None, True, False, True),
        ("no_solution", False, None, True, False, False),
        ("feasible", True, {"verdict": "pass"}, True, False, False),
        ("feasible", True, '{"verdict": "pa', True, False, True),
        # The search found a point and only the dump failed; nothing else can
        # produce the solution vector, and the scorer withholds the row until one
        # exists.
        ("solution_write_error", False, None, True, True, False),
        # Both driver-written causes are transient -- a memory cap under load, a
        # timeout. Treating the verdict as final would withhold the row for good,
        # recoverable only by --force, which pays for the whole search again...
        ("feasible", True, _driver_verdict(None), True, False, True),
        # ... but `verifier_died` is written for a deterministic cause too -- a
        # pyscipopt segfault, an OOM on a model that does not fit the cap -- and
        # retrying one on every resume of a 233-instance roster never converges.
        ("feasible", True, _driver_verdict(MAX_VERIFY_ATTEMPTS), True, False, False),
        # A verdict the checker itself reached is final: re-running a check that
        # cannot succeed never converges either.
        (
            "feasible",
            True,
            {"verdict": "error", "reason": "unsupported_constraint"},
            True,
            False,
            False,
        ),
    ],
    ids=[
        "no-result",
        "feasible-without-solution",
        "feasible-without-solution-no-verify",
        "feasible-with-solution-needs-a-verdict",
        "found-nothing",
        "verdict-not-recomputed",
        "truncated-verdict-recomputed",
        "solution-write-error",
        "driver-verdict-retried",
        "driver-verdict-attempts-exhausted",
        "checker-verdict-final",
    ],
)
def test_what_resume_still_has_to_run(
    tmp_path: Path,
    status: str | None,
    solution: bool,
    verdict: dict[str, object] | str | None,
    verify: bool,
    solve: bool,
    check: bool,
) -> None:
    job = _row(tmp_path, status, solution=solution, verdict=verdict)
    assert needs_solve(job, tmp_path, verify=verify) is solve
    assert needs_verification(job, tmp_path, verify=verify) is check
    assert has_usable_result(job, tmp_path, verify) is not (solve or check)


def test_forcing_a_rerun_drops_the_stale_solution_and_verdict(tmp_path: Path) -> None:
    # A verdict left behind describes the previous run's point, and reads as
    # current -- worse than no verdict at all.
    job = _row(tmp_path, "feasible", solution=True, verdict={"verdict": "pass"})

    drop_completed([job], [], tmp_path, force=True, verify=True)

    assert not job.result_path(tmp_path).exists()
    assert not job.solution_path(tmp_path).exists()
    assert not job.verification_path(tmp_path).exists()


# --- a job's outcome ------------------------------------------------------------


def test_a_rejected_solution_counts_as_a_failed_job(monkeypatch: pytest.MonkeyPatch) -> None:
    # This is the correctness benchmark: a run that published a point the checker
    # refused must not exit 0. Asserting the marker is in the list only restates a
    # constant, so drive the counter the exit code is actually computed from.
    assert "VERIFY-FAILED" in FAILURE_MARKERS
    assert "VERIFY-ERROR" in FAILURE_MARKERS
    monkeypatch.setattr(
        run_benchmark,
        "run_job",
        lambda job, args, results_dir: f"{job.engine}/{job.instance}: done | VERIFY-FAILED row",
    )
    assert execute([Job("cbls", "inst")], [], _driver_args(), Path("/results")) == 1


#: Stands in for the exit status of a verifier the timeout killed.
TIMED_OUT = 999


class _FakeRun:
    """Stands in for subprocess.run so _verify's return-code mapping can be driven.

    `writes` drops a verdict where the real verifier would, and it matters:
    `_verify` clears any verdict already on disk before starting, so a stand-in
    that writes nothing is a checker that crashed, and one that writes is a
    checker that reached a verdict. Conflating the two is the bug this
    distinction exists to keep out.
    """

    def __init__(self, returncode: int, writes: Callable[[], object] | None) -> None:
        self.returncode = returncode
        self.writes = writes
        self.stdout = "out"
        self.stderr = "err"

    def __call__(self, *args: object, **kwargs: object) -> _FakeRun:
        if self.returncode == TIMED_OUT:
            raise subprocess.TimeoutExpired(cmd="verify", timeout=1.0)
        if self.writes is not None:
            self.writes()
        return self


@pytest.mark.parametrize(
    ("returncode", "writes", "earlier", "line", "reason", "recorded"),
    [
        (TIMED_OUT, False, None, "VERIFY-ERROR", "verifier_timeout", {}),
        # Python exits 1 on any uncaught traceback, writing no verdict. Reporting
        # that as VERIFY-FAILED would fire this benchmark's loudest alarm -- "the
        # engine published an infeasible point" -- for a harness fault.
        (1, False, None, "VERIFY-ERROR", "verifier_died", {}),
        (1, True, None, "VERIFY-FAILED", "row_violation", {}),
        # A resumed run retries a driver-written verdict, so the previous attempt's
        # file is on disk when the verifier starts. A traceback next to it must not
        # read as a rejection, and the attempt is counted.
        (1, False, _driver_verdict(1), "VERIFY-ERROR", "verifier_died", {"attempts": 2}),
        # The address-space cap and the OOM killer both arrive as a signal, and
        # that is the difference between "try again with more memory" and "this
        # checker crashes on this model".
        (-9, False, None, "VERIFY-ERROR", "verifier_died", {}),
    ],
    ids=["timeout", "crash-is-not-a-rejection", "rejection", "crash-beside-earlier", "signal"],
)
def test_a_verification_outcome_leaves_the_verdict_that_describes_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    writes: bool,
    earlier: dict[str, object] | None,
    line: str,
    reason: str,
    recorded: dict[str, object],
) -> None:
    job = _row(tmp_path, None, verdict=earlier, job=Job("cpsat", "inst"))
    rejection: dict[str, object] = {"verdict": "fail", "reason": "row_violation"}

    def verifier_writes() -> Job:
        return _row(tmp_path, None, verdict=rejection, job=job)

    monkeypatch.setattr(
        subprocess, "run", _FakeRun(returncode, verifier_writes if writes else None)
    )
    reported = run_benchmark._verify(job, _driver_args(), tmp_path)

    assert line in reported
    assert line == "VERIFY-FAILED" or "VERIFY-FAILED" not in reported
    verdict = json.loads(job.verification_path(tmp_path).read_text())
    assert verdict["reason"] == reason
    if not writes:
        # The checker died, so the driver recorded it: a row with no verdict file
        # is indistinguishable from one nobody tried to check.
        assert (verdict["verdict"], verdict["engine"]) == ("error", "cpsat")
    if returncode < 0:
        assert f"signal {-returncode}" in verdict["message"]
    for key, value in recorded.items():
        assert verdict[key] == value


@pytest.mark.parametrize(
    ("step", "timeout"),
    [
        ("_run_solver", 600.0 + run_benchmark.TIMEOUT_SLACK_SECONDS),
        ("_verify", run_benchmark.VERIFY_TIMEOUT_SECONDS),
    ],
)
def test_every_job_process_is_bounded_capped_and_in_its_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, step: str, timeout: float
) -> None:
    # Unbounded, one hung job is the rest of an unattended run. Uncapped, one
    # large model takes the machine from its neighbours. And without its own
    # session a Ctrl-C at the driver reaches every in-flight child, each of which
    # leaves a "killed" result that resume treats as done.
    seen: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> _FakeRun:
        seen.update(kwargs, command=command)
        return _FakeRun(0, None)

    monkeypatch.setattr(subprocess, "run", run)
    getattr(run_benchmark, step)(Job("cbls", "inst"), _driver_args(mem_limit_gb=2.0), tmp_path)

    assert seen["timeout"] == timeout
    assert seen["start_new_session"] is True
    command = seen["command"]
    assert isinstance(command, list)
    assert command[:2] == ["/bin/sh", "-c"] and "ulimit -v 2097152" in command[2]


@pytest.mark.parametrize(
    ("returncode", "line", "message"),
    [
        (TIMED_OUT, "TIMEOUT", f"exceeded {600.0 + run_benchmark.TIMEOUT_SLACK_SECONDS}s"),
        # An OOM kill or the address-space cap writes no result of its own.
        (-9, "FAILED (exit -9)", "exit -9"),
    ],
    ids=["timeout", "died-without-a-result"],
)
def test_a_solve_the_driver_saw_die_leaves_a_killed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, returncode: int, line: str, message: str
) -> None:
    # Without the record the job reads as never scheduled, and without its budget
    # a resume at another budget skips it and the scorer's budget guard never fires.
    monkeypatch.setattr(subprocess, "run", _FakeRun(returncode, None))
    job = Job("cbls", "inst")
    reported, solved = run_benchmark._run_solver(job, _driver_args(), tmp_path)
    assert not solved and line in reported
    record = json.loads(job.result_path(tmp_path).read_text())
    assert (record["status"], record["budget_seconds"]) == ("killed", 600.0)
    assert message in record["message"]


def test_large_jobs_run_after_the_batch_and_their_failures_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The largest instances run alone at the end rather than four-up against one
    # memory limit; `execute` is the only place that schedules them at all.
    ran: list[str] = []
    batch = threading.Barrier(2, timeout=10)  # both batch jobs in flight at once

    def run_job(job: Job, args: argparse.Namespace, results_dir: Path) -> str:
        if job.instance != "big":
            batch.wait()
        ran.append(job.instance)
        return f"cbls/{job.instance}: FAILED (exit 1)" if job.instance == "big" else "ok"

    monkeypatch.setattr(run_benchmark, "run_job", run_job)
    normal = [Job("cbls", "a"), Job("cbls", "b")]
    failures = run_benchmark.execute(normal, [Job("cbls", "big")], _driver_args(jobs=2), tmp_path)
    assert sorted(ran[:2]) == ["a", "b"] and ran[2:] == ["big"]
    assert failures == 1


@pytest.mark.parametrize("finishes", [True, False], ids=["answers", "hangs"])
def test_the_cpsat_preflight_is_bounded(monkeypatch: pytest.MonkeyPatch, finishes: bool) -> None:
    # The check exists to fail at second zero; an OR-Tools release that ignores the
    # solve deadline must not turn it into the run's first six hours.
    seen: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> _FakeRun:
        seen.update(kwargs)
        return _FakeRun(0 if finishes else TIMED_OUT, None)(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    ok, output = run_benchmark.run_cpsat_preflight(1)

    assert seen["timeout"] == run_benchmark.PREFLIGHT_TIMEOUT_SECONDS
    assert ok is finishes
    assert ("did not finish within" in output) is not finishes


def test_a_verification_error_does_not_destroy_a_finished_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run_job's catch-all spans verification too. Overwriting the result with a
    # "killed" record there would discard a 600s search that already succeeded,
    # and resume -- seeing a non-feasible status -- would never redo it.
    job = _row(tmp_path, "feasible", solution=True)

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("cannot fork")

    monkeypatch.setattr(subprocess, "run", explode)
    line = run_benchmark.run_job(job, _driver_args(), tmp_path)

    assert "DRIVER-ERROR" in line
    assert json.loads(job.result_path(tmp_path).read_text())["status"] == "feasible"


def test_a_resolve_discards_the_verdict_on_the_previous_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The search produced a new point, so the verdict beside it describes the old
    # one -- and would read as current, which is worse than no verdict at all.
    job = _row(tmp_path, "solution_write_error", verdict={"verdict": "pass"})
    monkeypatch.setattr(
        run_benchmark, "_run_solver", lambda job, args, results_dir: ("solved", True)
    )
    monkeypatch.setattr(
        run_benchmark, "_verify", lambda job, args, results_dir: "verified tiny pass"
    )
    run_benchmark.run_job(job, _driver_args(), tmp_path)

    assert not job.verification_path(tmp_path).exists()


def _instance_dir(tmp_path: Path, name: str, mps: bytes, reference: float) -> tuple[Path, Path]:
    """A one-instance directory with its manifest pinned, and a roster naming it."""
    inst_dir = tmp_path / "instances"
    inst_dir.mkdir()
    data = gzip.compress(mps)
    (inst_dir / f"{name}.mps.gz").write_bytes(data)
    _write_csv(
        inst_dir / "manifest.csv",
        ["instance", "sha256", "bytes"],
        [[name, hashlib.sha256(data).hexdigest(), len(data)]],
    )
    roster = tmp_path / "roster.csv"
    _write_csv(
        roster, ["instance", "reference_value", "reference_kind"], [[name, reference, "opt"]]
    )
    return inst_dir, roster


def _run_driver(
    roster: Path, inst_dir: Path, results_dir: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "mipfeas" / "run_benchmark.py"),
            *("--roster", str(roster), "--inst-dir", str(inst_dir)),
            *("--results-dir", str(results_dir), "--cbls-bin", str(_binary())),
            *("--engines", "cbls", "--budget", "1"),
        ],
        capture_output=True,
        text=True,
    )


def test_the_driver_verifies_what_it_ran(tmp_path: Path) -> None:
    """End to end: solve, dump, verify, score — the chain #138 is about.

    Every other test here checks one joint of it in isolation, which is exactly
    how a chain ends up wired to nothing. This one runs the real driver over a
    real (tiny) instance and asserts a verdict landed and the row published.
    """
    pytest.importorskip("pyscipopt", reason="pyscipopt is in the 'benchmarks' extra, not 'dev'")
    from test_verify_solution import TINY_MPS  # the same hand-written instance

    # Pinned, because the driver refuses to run an instance whose bytes are not
    # recorded anywhere (issue #137) -- so this chain covers that joint too.
    inst_dir, roster = _instance_dir(tmp_path, "tiny", TINY_MPS.encode(), 9.0)
    results_dir = tmp_path / "results"

    completed = _run_driver(roster, inst_dir, results_dir)
    assert completed.returncode == 0, completed.stdout + completed.stderr

    verdict = json.loads((results_dir / "cbls" / "tiny.verify.json").read_text())
    assert verdict["verdict"] == "pass", verdict["message"]
    assert (results_dir / "cbls" / "tiny.sol").exists()

    scored = score_instance("tiny", "cbls", 9.0, "opt", results_dir, budget=1.0)
    assert scored.verification == "pass"
    assert not scored.withheld
    assert scored.objective is not None


# --- Preconditions: the bytes are what the pins say ---------------------------
#
# A corrupted or substituted instance measures a different program under a
# published row's name, and a revised reference file moves every gap in the table
# at once. Both are silent, so the driver refuses to start (issue #137). These
# fixtures are a few bytes each: the real roster is 546 MiB and re-fetching it is
# the very substitution the pins exist to catch.


def _pinned_dir(tmp_path: Path, instances: dict[str, bytes]) -> Path:
    """An instance directory whose manifest and references both match its files."""
    for name, data in instances.items():
        (tmp_path / f"{name}.mps.gz").write_bytes(data)
    _write_csv(
        tmp_path / "manifest.csv",
        ["instance", "sha256", "bytes"],
        [
            [name, hashlib.sha256(data).hexdigest(), len(data)]
            for name, data in sorted(instances.items())
        ],
    )
    reference_rows = []
    for name in PINNED_REFERENCE_FILES:
        (tmp_path / name).write_text(f"contents of {name}\n")
        data = (tmp_path / name).read_bytes()
        reference_rows.append([name, hashlib.sha256(data).hexdigest(), len(data)])
    _write_csv(tmp_path / "references.csv", ["file", "sha256", "bytes"], reference_rows)
    return tmp_path


def _pinned(tmp_path: Path) -> Path:
    return _pinned_dir(tmp_path, {"a": b"instance bytes"})


def _bare_dir(tmp_path: Path, pin: bool) -> Path:
    """An instance directory with no roster tables: `--inst-dir` may point at a
    vendored set that has none, pinned by its manifest or not at all."""
    (tmp_path / "a.mps.gz").write_bytes(b"instance bytes")
    if pin:
        digest = hashlib.sha256(b"instance bytes").hexdigest()
        _write_csv(tmp_path / "manifest.csv", ["instance", "sha256", "bytes"], [["a", digest, 14]])
    return tmp_path


@pytest.mark.parametrize(
    ("setup", "problem"),
    [
        (_pinned, None),
        (lambda d: (_pinned(d) / "a.mps.gz").write_bytes(b"different bytes"), "a.mps.gz"),
        # The yardstick every gap is scored against; a revision moves the whole table.
        (lambda d: (_pinned(d) / "roster.csv").write_text("edited\n"), "roster.csv:"),
        (lambda d: _bare_dir(d, pin=False), "manifest.csv"),
        # A directory holding the yardstick but nothing pinning it is the state
        # #137 exists to end; a bare directory with no yardstick is not.
        (lambda d: (_pinned(d) / "references.csv").unlink(), "references.csv"),
        (lambda d: _bare_dir(d, pin=True), None),
        # main() already names missing instances with the command that fetches
        # them; reporting them twice would bury the substitution this check is for.
        (lambda d: (_pinned(d) / "a.mps.gz").unlink(), None),
    ],
    ids=[
        "intact",
        "substituted-instance",
        "revised-reference-file",
        "unpinned-directory",
        "roster-tables-without-pins",
        "bare-directory-with-manifest",
        "absent-instance-left-to-main",
    ],
)
def test_verify_preconditions(
    tmp_path: Path, setup: Callable[[Path], object], problem: str | None
) -> None:
    setup(tmp_path)
    found = run_benchmark.verify_preconditions(tmp_path, ["a"])
    assert found == [] if problem is None else (len(found) == 1 and problem in found[0]), found


@pytest.mark.parametrize(
    ("engines", "overrides", "substituted", "preflight_ok", "rc", "preflights", "said"),
    [
        (("cbls",), {}, True, True, 2, [], ""),
        (("cbls",), {"skip_preconditions": True}, True, True, None, [], "not publishable"),
        # The preflight tests elsewhere pin what it concludes; this pins that the
        # driver acts on it. A release that moved a subsolver flag has to stop the
        # run, or the roster burns its budget on an empty baseline.
        (("cbls", "cpsat"), {}, False, False, 2, [1], ""),
        # The announcement carries a multiplicity at two or more, and the
        # thread-count assertion is about the CPU share the baseline gets -- so
        # preflighting one worker for an eight-worker run checks nothing that runs.
        (("cpsat",), {"cpsat_workers": 8}, False, True, None, [8], ""),
        # It asserts the CP-SAT baseline; a run without that engine has nothing to check.
        (("cbls",), {}, False, True, None, [], ""),
    ],
    ids=[
        "substituted-instance-stops-the-run",
        "skip-says-not-publishable",
        "broken-preflight-refuses",
        "preflight-at-the-roster-worker-count",
        "no-preflight-without-cpsat",
    ],
)
def test_check_preconditions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    engines: tuple[str, ...],
    overrides: dict[str, object],
    substituted: bool,
    preflight_ok: bool,
    rc: int | None,
    preflights: list[int],
    said: str,
) -> None:
    inst_dir = _pinned(tmp_path)
    if substituted:
        (inst_dir / "a.mps.gz").write_bytes(b"other")
    seen: list[int] = []

    def preflight(workers: int) -> tuple[bool, str]:
        seen.append(workers)
        return preflight_ok, "preflight OK" if preflight_ok else "PREFLIGHT FAILED: log broke."

    monkeypatch.setattr(run_benchmark, "run_cpsat_preflight", preflight)
    args = _driver_args(inst_dir=inst_dir, **overrides)

    assert run_benchmark.check_preconditions(args, ["a"], engines) == rc
    assert seen == preflights
    assert said in capsys.readouterr().err


# --- The machine record -------------------------------------------------------


def _record(tmp_path: Path, roster: Path | None = None, **overrides: object) -> dict[str, object]:
    args = _driver_args(**{"inst_dir": tmp_path, "jobs": 4, "mem_limit_gb": 6.0, **overrides})
    return run_benchmark.build_run_record(
        args, roster or tmp_path / "roster.csv", ["a"], ("cbls", "cpsat"), 2, 2
    )


def test_the_run_record_states_what_a_wall_clock_result_cannot_be_read_without(
    tmp_path: Path,
) -> None:
    # The same roster four-up is a different measurement from the same roster
    # one-up, and a budgeted comparison is a statement about a machine.
    record = _record(tmp_path)
    assert record["concurrency"] == {
        "jobs": 4,
        "large_instance_jobs": 1,
        "cpsat_workers": 1,
        # Per-solve CPU, the other half of what the machine was asked for.
        "cbls_threads": 1,
        "mem_limit_gb": 6.0,
    }
    machine = record["machine"]
    assert isinstance(machine, dict)
    assert machine["host"]
    assert isinstance(machine["cpu_count"], int)
    assert machine["memory_total_kib"] is None or isinstance(machine["memory_total_kib"], int)
    assert record["budget_seconds"] == 600.0
    versions = record["versions"]
    assert isinstance(versions, dict)
    assert versions["engine_commit"] == "abc1234"
    assert versions["python"]
    # --skip-preconditions is what makes a run unpublishable, and stderr does not
    # survive to whoever reads the table.
    assert record["run"]["preconditions_checked"] is True  # type: ignore[index]
    skipped = _record(tmp_path, skip_preconditions=True)
    assert skipped["run"]["preconditions_checked"] is False  # type: ignore[index]


def test_the_run_record_identifies_the_yardstick_the_gaps_will_be_scored_against(
    tmp_path: Path,
) -> None:
    # An upstream revision of the solution file moves every gap in the table, so a
    # later reader has to be able to tell which one a table was scored against --
    # and `--roster` accepts any CSV, so the file actually read is named too.
    inst_dir = _pinned(tmp_path)
    roster = tmp_path / "elsewhere.csv"
    roster.write_text("instance,reference_value,reference_kind\na,1.0,opt\n")

    references = _record(inst_dir, roster)["references"]

    assert isinstance(references, dict)
    pinned = references["pinned"]
    assert isinstance(pinned, dict)
    assert set(pinned) == set(PINNED_REFERENCE_FILES)
    assert references["manifest_sha256"]
    assert references["roster_path"] == str(roster)
    assert references["roster_sha256"] == hashlib.sha256(roster.read_bytes()).hexdigest()


def test_a_resumed_run_adds_a_record_rather_than_overwriting_the_first(tmp_path: Path) -> None:
    # A resumed directory was produced by two machines and two concurrencies;
    # keeping only the last would claim the whole set came off one.
    results = tmp_path / "results"
    run_benchmark.append_run_record(results, _record(tmp_path, jobs=1))
    run_benchmark.append_run_record(results, _record(tmp_path, jobs=8))

    runs = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"]
    assert [r["concurrency"]["jobs"] for r in runs] == [1, 8]
    # Closing the second invocation must not take the first one's entry with it.
    run_benchmark.close_run_record(results, {**_record(tmp_path, jobs=8), "status": "complete"})
    runs = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"]
    assert [(r["concurrency"]["jobs"], r["status"]) for r in runs] == [
        (1, "running"),
        (8, "complete"),
    ]


def test_closing_the_record_replaces_the_entry_this_run_opened(tmp_path: Path) -> None:
    # Written before the first job, not after the last: a run killed at hour six
    # still has to say what produced the results it did leave.
    results = tmp_path / "results"
    record = _record(tmp_path)
    run_benchmark.append_run_record(results, record)
    written = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"]
    assert [(r["status"], r["finished_at"]) for r in written] == [("running", None)]

    record["status"] = "complete"
    record["outcome"] = {"jobs_run": 1, "failures": 0, "rejected": 0, "wall_seconds": 1.0}
    run_benchmark.close_run_record(results, record)

    runs = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "complete"


# --- A resume that runs nothing must not report a clean run --------------------


@pytest.mark.parametrize(
    ("status", "verdict", "verify", "unchecked", "rejected"),
    [
        # After MAX_VERIFY_ATTEMPTS the row stops being retried, so every later
        # resume drops the job and runs nothing. Without this count the driver would
        # print "0 jobs to run" and exit 0 on a row that was never checked.
        ("feasible", _driver_verdict(MAX_VERIFY_ATTEMPTS), True, 1, 0),
        ("feasible", None, True, 1, 0),
        ("feasible", {"verdict": "pass"}, True, 0, 0),
        # It *was* checked. Conflating the two would bury the loudest signal this
        # benchmark has under a harness counter -- and a resume, which runs
        # nothing, must still count the rejection already on disk.
        ("feasible", {"verdict": "fail"}, True, 0, 1),
        ("feasible", None, False, 0, 0),
        ("no_solution", None, True, 0, 0),
    ],
    ids=["exhausted", "no-verdict", "checked", "rejected", "verification-off", "found-nothing"],
)
def test_the_rows_a_run_must_not_exit_zero_on_are_counted(
    tmp_path: Path,
    status: str,
    verdict: dict[str, object] | None,
    verify: bool,
    unchecked: int,
    rejected: int,
) -> None:
    job = _row(tmp_path, status, solution=True, verdict=verdict)
    other_engine = Job("cpsat", "inst")
    assert count_unchecked([job], tmp_path, verify=verify) == unchecked
    assert count_rejected([job, other_engine], tmp_path) == rejected


def test_the_machine_record_exists_before_the_first_job_is_dispatched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The point of writing it first: a run killed at hour six still has one.

    Nothing pinned this. Moving the write after `execute`, or deleting it from
    `main` outright, left the whole Python suite green -- criterion 4's central
    property held up by a comment. The spy asserts the file is on disk at the
    moment the first job would be dispatched, which is the only moment that
    distinguishes the two orderings.
    """
    inst_dir, roster = _instance_dir(tmp_path, "inst", b"NAME tiny\nENDATA\n", 1.0)
    results_dir = tmp_path / "results"
    binary = tmp_path / "cbls_mipfeas"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)

    seen: list[bool] = []

    def spy(jobs: list[run_benchmark.Job], *args: object, **kwargs: object) -> int:
        seen.append((results_dir / run_benchmark.RUN_RECORD_FILENAME).exists())
        return 0

    monkeypatch.setattr(run_benchmark, "execute", spy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmark.py",
            *("--roster", str(roster), "--inst-dir", str(inst_dir)),
            *("--results-dir", str(results_dir), "--cbls-bin", str(binary)),
            *("--engines", "cbls", "--budget", "1", "--skip-preconditions"),
        ],
    )
    run_benchmark.main()

    assert seen, "execute was never called, so the ordering was not exercised"
    assert all(seen), "a job was dispatched before the machine record was written"


def test_a_resume_with_nothing_left_to_run_still_exits_non_zero(tmp_path: Path) -> None:
    """End to end: the driver over a directory whose only row was never checked."""
    inst_dir, roster = _instance_dir(tmp_path, "inst", b"NAME tiny\nENDATA\n", 1.0)
    results_dir = tmp_path / "results"
    _row(results_dir, "feasible", solution=True, verdict=_driver_verdict(MAX_VERIFY_ATTEMPTS))

    completed = _run_driver(roster, inst_dir, results_dir)

    assert "0 jobs to run" in completed.stdout, completed.stdout
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "UNCHECKED" in completed.stderr
