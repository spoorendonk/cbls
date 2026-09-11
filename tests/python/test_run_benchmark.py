"""Tests for the MIPfeas run driver."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.mipfeas import run_benchmark
from benchmarks.mipfeas.primal_integral import NO_SOLUTION_GAP, score_instance, summarize
from benchmarks.mipfeas.run_benchmark import (
    FAILURE_MARKERS,
    Job,
    build_command,
    count_rejected,
    count_unchecked,
    drop_completed,
    execute,
    needs_solve,
    needs_verification,
    plan_jobs,
    read_roster,
    read_sizes,
    resolve_roster,
    with_memory_limit,
    write_failure_result,
    write_failure_verdict,
)


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_resolve_roster_maps_the_named_rosters(tmp_path: Path) -> None:
    assert resolve_roster("smoke", tmp_path) == tmp_path / "smoke.csv"
    assert resolve_roster("full", tmp_path) == tmp_path / "roster.csv"


def test_resolve_roster_passes_a_path_through(tmp_path: Path) -> None:
    custom = tmp_path / "mine.csv"
    assert resolve_roster(str(custom), tmp_path) == custom


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


def test_plan_jobs_pairs_every_instance_with_every_engine() -> None:
    normal, large = plan_jobs(["a", "b"], ("cbls", "cpsat"), {}, large_bytes=1000)
    assert large == []
    assert set(normal) == {Job("cbls", "a"), Job("cpsat", "a"), Job("cbls", "b"), Job("cpsat", "b")}


def test_plan_jobs_sets_large_instances_aside() -> None:
    # The roster spans four orders of magnitude in size; the big ones must not run
    # four-up against a memory limit.
    sizes = {"small": 100, "huge": 50_000}
    normal, large = plan_jobs(["small", "huge"], ("cbls",), sizes, large_bytes=10_000)
    assert normal == [Job("cbls", "small")]
    assert large == [Job("cbls", "huge")]


def test_plan_jobs_treats_an_unknown_size_as_small() -> None:
    normal, large = plan_jobs(["mystery"], ("cbls",), {}, large_bytes=10_000)
    assert normal == [Job("cbls", "mystery")]
    assert large == []


def test_result_path_is_per_engine(tmp_path: Path) -> None:
    assert Job("cbls", "inst").result_path(tmp_path) == tmp_path / "cbls" / "inst.json"


def test_a_killed_job_scores_as_a_failure_not_as_unrun(tmp_path: Path) -> None:
    # A job the driver had to kill did happen, so it scores 2 like any other run
    # that produced nothing — but it must stay distinguishable from a job that was
    # never scheduled, which is excluded from the aggregate instead.
    job = Job("cbls", "inst")
    write_failure_result(job, tmp_path, "killed", "exceeded wall clock", budget=60.0)

    scored = score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)
    assert scored.status == "killed"
    assert scored.primal_integral == pytest.approx(NO_SOLUTION_GAP)

    summary = summarize([scored], "cbls")
    assert summary.scored == 1
    assert summary.not_run == 0
    assert summary.feasible == 0
    assert summary.errored == 1


def test_the_runner_writes_nothing_for_an_absent_instance(tmp_path: Path) -> None:
    """The #103 guard: a missing instance must not be scored as 'found nothing'.

    Runs the real binary, because this is a property of the process contract the
    driver depends on — a non-zero exit and no result file — not of any function.
    """
    binary = Path(__file__).resolve().parents[2] / "build" / "cbls_mipfeas"
    if not binary.exists():
        pytest.skip("cbls_mipfeas not built")

    out_dir = tmp_path / "results"
    result = subprocess.run(
        [
            str(binary),
            "--instance",
            "no-such-instance",
            "--inst-dir",
            str(tmp_path),
            "--out-dir",
            str(out_dir),
            "--budget",
            "1",
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
    binary = Path(__file__).resolve().parents[2] / "build" / "cbls_mipfeas"
    if not binary.exists():
        pytest.skip("cbls_mipfeas not built")

    result = subprocess.run(
        [
            str(binary),
            "--instance",
            "anything",
            "--out-dir",
            str(tmp_path),
            "--budget",
            "notanumber",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--budget: 'notanumber' is not a number" in result.stderr
    assert "--budget must be a positive" in result.stderr


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
        "commit": "abc1234",
        "verify": True,
        "skip_preconditions": False,
        "jobs": 1,
        "mem_limit_gb": None,
        "force": False,
        "large_bytes": 5_000_000,
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_build_command_passes_the_cbls_configuration_through() -> None:
    # Both are deliberate departures from the engine defaults, so the driver has to
    # state them: a run that silently inherited either would not be the run the
    # README describes.
    command = build_command(Job("cbls", "inst"), _driver_args(), Path("/results"))
    assert "--inf-clamp" in command
    assert command[command.index("--inf-clamp") + 1] == "10000000.0"
    assert "--compound-moves" in command
    # Propagation is the engine default, so the driver says nothing about it.
    assert "--no-propagate-bounds" not in command


def test_build_command_can_disable_bound_propagation() -> None:
    command = build_command(
        Job("cbls", "inst"), _driver_args(propagate_bounds=False), Path("/results")
    )
    assert "--no-propagate-bounds" in command


def test_build_command_can_disable_compound_moves() -> None:
    command = build_command(
        Job("cbls", "inst"), _driver_args(compound_moves=False), Path("/results")
    )
    assert "--no-compound-moves" in command
    assert "--compound-moves" not in command


def test_build_command_gives_cpsat_its_worker_count() -> None:
    command = build_command(Job("cpsat", "inst"), _driver_args(), Path("/results"))
    assert command[command.index("--workers") + 1] == "1"


def test_with_memory_limit_is_a_no_op_without_a_limit() -> None:
    assert with_memory_limit(["prog", "--flag"], None) == ["prog", "--flag"]
    assert with_memory_limit(["prog"], 0) == ["prog"]


def test_with_memory_limit_actually_caps_the_child() -> None:
    # Asserting on the wrapper's shape would pass even if the quoting were wrong,
    # so run it and ask the child what its own limit is. 2 GB in KiB.
    command = with_memory_limit(["/bin/sh", "-c", "ulimit -v"], 2.0)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(2 * 1024 * 1024)


def test_with_memory_limit_preserves_arguments_containing_spaces() -> None:
    command = with_memory_limit(["/bin/echo", "two words", "--flag=a b"], 1.0)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.rstrip("\n") == "two words --flag=a b"


def test_write_failure_result_creates_the_engine_directory(tmp_path: Path) -> None:
    job = Job("cpsat", "inst")
    write_failure_result(job, tmp_path / "fresh", "killed", "oom", budget=60.0)
    assert (tmp_path / "fresh" / "cpsat" / "inst.json").exists()


# ---------------------------------------------------------------------------
# Verification (#138): the driver has to produce a verdict for every feasible row.


def test_build_command_hands_both_engines_a_solution_directory() -> None:
    # Without this neither runner writes a solution vector, and nothing
    # downstream can check a reported point against the instance it came from.
    for engine in ("cbls", "cpsat"):
        command = build_command(Job(engine, "inst"), _driver_args(), Path("/results"))
        assert command[command.index("--solution-dir") + 1] == f"/results/{engine}"


def test_build_command_omits_the_solution_directory_when_not_verifying() -> None:
    command = build_command(Job("cbls", "inst"), _driver_args(verify=False), Path("/results"))
    assert "--solution-dir" not in command


def _write_result_file(job: Job, results_dir: Path, status: str) -> None:
    path = job.result_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"status": status, "objective": 1.0}))


def test_a_job_with_no_result_still_needs_solving(tmp_path: Path) -> None:
    assert needs_solve(Job("cbls", "inst"), tmp_path, verify=True)


def test_a_feasible_result_without_its_solution_is_solved_again(tmp_path: Path) -> None:
    # A results directory filled before #138, or by a --no-verify pass: nothing
    # but the search can produce the solution vector, so resume cannot skip it.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    assert needs_solve(job, tmp_path, verify=True)
    assert not needs_solve(job, tmp_path, verify=False)


def test_a_feasible_result_with_its_solution_is_not_solved_again(tmp_path: Path) -> None:
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    assert not needs_solve(job, tmp_path, verify=True)
    # ... but it still needs the verdict, and that step is cheap and separate.
    assert needs_verification(job, tmp_path, verify=True)


def test_a_run_that_found_nothing_needs_no_verdict(tmp_path: Path) -> None:
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "no_solution")
    assert not needs_solve(job, tmp_path, verify=True)
    assert not needs_verification(job, tmp_path, verify=True)


def test_an_existing_verdict_is_not_recomputed(tmp_path: Path) -> None:
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    job.verification_path(tmp_path).write_text(json.dumps({"verdict": "pass"}))
    assert not needs_verification(job, tmp_path, verify=True)


def test_a_truncated_verdict_is_recomputed(tmp_path: Path) -> None:
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    job.verification_path(tmp_path).write_text('{"verdict": "pa')
    assert needs_verification(job, tmp_path, verify=True)


def test_forcing_a_rerun_drops_the_stale_solution_and_verdict(tmp_path: Path) -> None:
    # A verdict left behind describes the previous run's point, and reads as
    # current -- worse than no verdict at all.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    job.verification_path(tmp_path).write_text(json.dumps({"verdict": "pass"}))

    drop_completed([job], [], tmp_path, force=True, verify=True)

    assert not job.result_path(tmp_path).exists()
    assert not job.solution_path(tmp_path).exists()
    assert not job.verification_path(tmp_path).exists()


def test_write_failure_verdict_records_a_checker_that_died(tmp_path: Path) -> None:
    # A verifier killed by the timeout or the memory cap writes nothing itself,
    # and a row with no verdict file is indistinguishable from one nobody tried.
    job = Job("cpsat", "inst")
    write_failure_verdict(job, tmp_path, "verifier_timeout", "exceeded 900.0s")

    record = json.loads(job.verification_path(tmp_path).read_text())
    assert record["verdict"] == "error"
    assert record["reason"] == "verifier_timeout"
    assert record["engine"] == "cpsat"


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
    assert execute([Job("cbls", "inst")], _driver_args(), Path("/results"), workers=1) == 1


def _verdict_file(job: Job, results_dir: Path, verdict: str, reason: str = "") -> None:
    path = job.verification_path(results_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"verdict": verdict, "reason": reason}))


class _FakeRun:
    """Stands in for subprocess.run so _verify's return-code mapping can be driven.

    `writes` is the verdict the stand-in drops where the real verifier would, and
    it matters: `_verify` clears any verdict already on disk before starting, so a
    stand-in that writes nothing is a checker that crashed, and one that writes is
    a checker that reached a verdict. Conflating the two is the bug this
    distinction exists to keep out.
    """

    def __init__(
        self,
        returncode: int,
        raises: bool = False,
        writes: tuple[Job, Path, str, str] | None = None,
    ) -> None:
        self.returncode = returncode
        self.raises = raises
        self.writes = writes
        self.stdout = "out"
        self.stderr = "err"

    def __call__(self, *args: object, **kwargs: object) -> _FakeRun:
        if self.raises:
            raise subprocess.TimeoutExpired(cmd="verify", timeout=1.0)
        if self.writes is not None:
            job, results_dir, verdict, reason = self.writes
            _verdict_file(job, results_dir, verdict, reason)
        return self


def test_a_verifier_that_times_out_leaves_an_error_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = Job("cbls", "inst")
    monkeypatch.setattr(subprocess, "run", _FakeRun(0, raises=True))
    line = run_benchmark._verify(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert "VERIFY-ERROR" in line
    assert json.loads(job.verification_path(tmp_path).read_text())["reason"] == "verifier_timeout"


def test_a_verifier_that_crashed_is_not_reported_as_a_rejected_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Python exits 1 on any uncaught traceback, writing no verdict. Reporting that
    # as VERIFY-FAILED would fire this benchmark's loudest alarm -- "the engine
    # published an infeasible point" -- for a harness fault.
    job = Job("cbls", "inst")
    monkeypatch.setattr(subprocess, "run", _FakeRun(1))
    line = run_benchmark._verify(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert "VERIFY-ERROR" in line
    assert json.loads(job.verification_path(tmp_path).read_text())["reason"] == "verifier_died"


def test_a_rejected_solution_keeps_the_verifier_s_own_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = Job("cbls", "inst")
    monkeypatch.setattr(
        subprocess, "run", _FakeRun(1, writes=(job, tmp_path, "fail", "row_violation"))
    )
    line = run_benchmark._verify(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert "VERIFY-FAILED" in line
    assert json.loads(job.verification_path(tmp_path).read_text())["reason"] == "row_violation"


def test_a_crash_next_to_an_earlier_attempts_verdict_is_not_a_rejected_solution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A resumed run retries a driver-written verdict, so the previous attempt's
    # file is on disk when the verifier starts. A traceback exits 1 and writes
    # nothing, and finding that stale file would report the crash as "the engine
    # published an infeasible point" -- this benchmark's loudest alarm, fired for
    # a harness fault.
    job = Job("cbls", "inst")
    job.verification_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    job.verification_path(tmp_path).write_text(
        json.dumps({"verdict": "error", "reason": "verifier_died", "attempts": 1})
    )
    monkeypatch.setattr(subprocess, "run", _FakeRun(1))
    line = run_benchmark._verify(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert "VERIFY-FAILED" not in line
    assert "VERIFY-ERROR" in line
    record = json.loads(job.verification_path(tmp_path).read_text())
    assert record["reason"] == "verifier_died"
    assert record["attempts"] == 2


def test_a_verifier_that_a_signal_killed_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The address-space cap and the OOM killer both arrive as a signal, and that
    # is the difference between "try again with more memory" and "this checker
    # crashes on this model".
    job = Job("cbls", "inst")
    monkeypatch.setattr(subprocess, "run", _FakeRun(-9))
    run_benchmark._verify(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert "signal 9" in json.loads(job.verification_path(tmp_path).read_text())["message"]


def test_a_driver_verdict_stops_being_retried_once_the_attempts_run_out(
    tmp_path: Path,
) -> None:
    # `verifier_died` is written for a deterministic cause too -- a pyscipopt
    # segfault, an OOM on a model that does not fit the cap. Retrying one on every
    # resume of a 233-instance roster re-pays the whole verification and never
    # converges.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    job.verification_path(tmp_path).write_text(
        json.dumps(
            {
                "verdict": "error",
                "reason": "verifier_died",
                "attempts": run_benchmark.MAX_VERIFY_ATTEMPTS,
            }
        )
    )
    assert not needs_verification(job, tmp_path, verify=True)


def test_a_checker_the_driver_killed_is_checked_again_on_resume(tmp_path: Path) -> None:
    # Both causes are transient -- a memory cap under load, a timeout. Treating the
    # verdict as final would withhold the row for good, recoverable only by
    # --force, which pays for the whole search again.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    _verdict_file(job, tmp_path, "error", "verifier_died")
    assert needs_verification(job, tmp_path, verify=True)


def test_a_verdict_the_checker_itself_reached_is_final(tmp_path: Path) -> None:
    # Re-running a check that cannot succeed never converges.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")
    _verdict_file(job, tmp_path, "error", "unsupported_constraint")
    assert not needs_verification(job, tmp_path, verify=True)


def test_a_solution_the_runner_could_not_write_is_solved_again(tmp_path: Path) -> None:
    # The search found a point and only the dump failed; nothing else can produce
    # the solution vector, and the scorer withholds the row until one exists.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "solution_write_error")
    assert needs_solve(job, tmp_path, verify=True)


def test_a_resumed_run_still_reports_an_already_rejected_solution(tmp_path: Path) -> None:
    # A resume runs nothing, so without counting the verdicts already on disk the
    # driver would exit 0 and tell an unattended wrapper the run was clean.
    job = Job("cbls", "inst")
    _verdict_file(job, tmp_path, "fail", "row_violation")
    assert count_rejected([job], tmp_path) == 1
    assert count_rejected([Job("cpsat", "inst")], tmp_path) == 0


def test_the_driver_verifies_what_it_ran(tmp_path: Path) -> None:
    """End to end: solve, dump, verify, score — the chain #138 is about.

    Every other test here checks one joint of it in isolation, which is exactly
    how a chain ends up wired to nothing. This one runs the real driver over a
    real (tiny) instance and asserts a verdict landed and the row published.
    """
    pytest.importorskip("pyscipopt", reason="pyscipopt is in the 'benchmarks' extra, not 'dev'")
    repo_root = Path(__file__).resolve().parents[2]
    binary = repo_root / "build" / "cbls_mipfeas"
    if not binary.exists():
        pytest.skip("cbls_mipfeas not built")

    from test_verify_solution import TINY_MPS  # the same hand-written instance

    inst_dir = tmp_path / "instances"
    inst_dir.mkdir()
    instance_bytes = gzip.compress(TINY_MPS.encode())
    (inst_dir / "tiny.mps.gz").write_bytes(instance_bytes)
    # Pinned, because the driver now refuses to run an instance whose bytes are
    # not recorded anywhere (issue #137) -- so this chain covers that joint too.
    _write_csv(
        inst_dir / "manifest.csv",
        ["instance", "sha256", "bytes"],
        [["tiny", hashlib.sha256(instance_bytes).hexdigest(), len(instance_bytes)]],
    )
    roster = tmp_path / "roster.csv"
    _write_csv(roster, ["instance", "reference_value", "reference_kind"], [["tiny", 9.0, "opt"]])
    results_dir = tmp_path / "results"

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "mipfeas" / "run_benchmark.py"),
            "--roster",
            str(roster),
            "--inst-dir",
            str(inst_dir),
            "--results-dir",
            str(results_dir),
            "--cbls-bin",
            str(binary),
            "--engines",
            "cbls",
            "--budget",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    verdict = json.loads((results_dir / "cbls" / "tiny.verify.json").read_text())
    assert verdict["verdict"] == "pass", verdict["message"]
    assert (results_dir / "cbls" / "tiny.sol").exists()

    scored = score_instance("tiny", "cbls", 9.0, "opt", results_dir, budget=1.0)
    assert scored.verification == "pass"
    assert not scored.withheld
    assert scored.objective is not None


def test_a_verification_error_does_not_destroy_a_finished_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # run_job's catch-all now spans verification too. Overwriting the result with
    # a "killed" record there would discard a 600s search that already succeeded,
    # and resume -- seeing a non-feasible status -- would never redo it.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "feasible")
    job.solution_path(tmp_path).write_text("=obj= 1.0\nx 1\n")

    def explode(*args: object, **kwargs: object) -> None:
        raise OSError("cannot fork")

    monkeypatch.setattr(subprocess, "run", explode)
    line = run_benchmark.run_job(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert "DRIVER-ERROR" in line
    assert json.loads(job.result_path(tmp_path).read_text())["status"] == "feasible"


def test_a_resolve_discards_the_verdict_on_the_previous_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The search produced a new point, so the verdict beside it describes the old
    # one -- and would read as current, which is worse than no verdict at all.
    job = Job("cbls", "inst")
    _write_result_file(job, tmp_path, "solution_write_error")
    _verdict_file(job, tmp_path, "pass")
    monkeypatch.setattr(
        run_benchmark, "_run_solver", lambda job, args, results_dir: ("solved", True)
    )
    monkeypatch.setattr(
        run_benchmark, "_verify", lambda job, args, results_dir: "verified tiny pass"
    )
    run_benchmark.run_job(job, _driver_args(mem_limit_gb=None), tmp_path)

    assert not job.verification_path(tmp_path).exists()


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
    for name in run_benchmark.PINNED_REFERENCE_FILES:
        (tmp_path / name).write_text(f"contents of {name}\n")
        data = (tmp_path / name).read_bytes()
        reference_rows.append([name, hashlib.sha256(data).hexdigest(), len(data)])
    _write_csv(tmp_path / "references.csv", ["file", "sha256", "bytes"], reference_rows)
    return tmp_path


def test_an_intact_instance_directory_raises_no_precondition_problem(tmp_path: Path) -> None:
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    assert run_benchmark.verify_preconditions(inst_dir, ["a"]) == []


def test_a_substituted_instance_refuses_the_run(tmp_path: Path) -> None:
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    (inst_dir / "a.mps.gz").write_bytes(b"different bytes")

    problems = run_benchmark.verify_preconditions(inst_dir, ["a"])

    assert len(problems) == 1
    assert "a.mps.gz" in problems[0]


def test_a_revised_reference_file_refuses_the_run(tmp_path: Path) -> None:
    # The yardstick every gap is scored against; a revision moves the whole table.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    (inst_dir / "roster.csv").write_text("edited\n")

    problems = run_benchmark.verify_preconditions(inst_dir, ["a"])

    assert [p for p in problems if p.startswith("roster.csv:")]


def test_an_unpinned_instance_directory_is_refused_rather_than_trusted(tmp_path: Path) -> None:
    (tmp_path / "a.mps.gz").write_bytes(b"x")
    problems = run_benchmark.verify_preconditions(tmp_path, ["a"])
    assert len(problems) == 1, problems
    assert "manifest.csv" in problems[0]


def test_roster_tables_with_no_pins_beside_them_are_refused(tmp_path: Path) -> None:
    # A directory holding the yardstick but nothing pinning it is the state this
    # issue exists to end; a bare instance directory that has no yardstick at all
    # is not, and is checked only against manifest.csv.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    (inst_dir / "references.csv").unlink()

    problems = run_benchmark.verify_preconditions(inst_dir, ["a"])

    assert len(problems) == 1
    assert "references.csv" in problems[0]


def test_a_bare_instance_directory_is_checked_against_its_manifest_alone(
    tmp_path: Path,
) -> None:
    # `--inst-dir` may point at a vendored set with no roster tables of its own.
    (tmp_path / "a.mps.gz").write_bytes(b"instance bytes")
    _write_csv(
        tmp_path / "manifest.csv",
        ["instance", "sha256", "bytes"],
        [["a", hashlib.sha256(b"instance bytes").hexdigest(), len(b"instance bytes")]],
    )

    assert run_benchmark.verify_preconditions(tmp_path, ["a"]) == []


def test_an_absent_instance_is_left_to_the_drivers_own_report(tmp_path: Path) -> None:
    # main() already names missing instances with the command that fetches them;
    # reporting them twice would bury the substitution this check is for.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    (inst_dir / "a.mps.gz").unlink()

    assert run_benchmark.verify_preconditions(inst_dir, ["a"]) == []


def test_check_preconditions_stops_the_run_on_a_substituted_instance(tmp_path: Path) -> None:
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    (inst_dir / "a.mps.gz").write_bytes(b"other")
    args = _driver_args(inst_dir=inst_dir, skip_preconditions=False)

    assert run_benchmark.check_preconditions(args, ["a"], ("cbls",)) == 2


def test_skip_preconditions_proceeds_but_says_the_run_is_not_publishable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    (inst_dir / "a.mps.gz").write_bytes(b"other")
    args = _driver_args(inst_dir=inst_dir, skip_preconditions=True)

    assert run_benchmark.check_preconditions(args, ["a"], ("cbls",)) is None
    assert "not publishable" in capsys.readouterr().err


def test_a_broken_preflight_refuses_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The preflight tests elsewhere pin what it concludes; this pins that the driver
    # acts on it. A release that moved a subsolver flag has to stop the run, not
    # merely print about it, or the roster burns its budget on an empty baseline.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    monkeypatch.setattr(
        run_benchmark,
        "run_cpsat_preflight",
        lambda _workers: (False, "PREFLIGHT FAILED on ortools 9.16: log format broke."),
    )
    args = _driver_args(inst_dir=inst_dir, skip_preconditions=False)

    assert run_benchmark.check_preconditions(args, ["a"], ("cbls", "cpsat")) == 2


def test_the_preflight_runs_at_the_worker_count_the_roster_will_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The announcement carries a multiplicity at two or more, and the thread-count
    # assertion is about the CPU share the baseline gets -- so preflighting one
    # worker for an eight-worker run checks a configuration nothing will run.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    seen: list[int] = []

    def record_workers(workers: int) -> tuple[bool, str]:
        seen.append(workers)
        return True, "preflight OK"

    monkeypatch.setattr(run_benchmark, "run_cpsat_preflight", record_workers)
    args = _driver_args(inst_dir=inst_dir, skip_preconditions=False, cpsat_workers=8)

    assert run_benchmark.check_preconditions(args, ["a"], ("cpsat",)) is None
    assert seen == [8]


def test_the_preflight_is_not_paid_for_a_cbls_only_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # It asserts the CP-SAT baseline; a run without that engine has nothing to check.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    called: list[int] = []

    def record_workers(workers: int) -> tuple[bool, str]:
        called.append(workers)
        return True, "preflight OK"

    monkeypatch.setattr(run_benchmark, "run_cpsat_preflight", record_workers)
    args = _driver_args(inst_dir=inst_dir, skip_preconditions=False)

    assert run_benchmark.check_preconditions(args, ["a"], ("cbls",)) is None
    assert called == []


# --- The machine record -------------------------------------------------------


def _record_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    return _driver_args(**{"inst_dir": tmp_path, "jobs": 4, "mem_limit_gb": 6.0, **overrides})


def test_the_run_record_states_the_concurrency_the_run_used(tmp_path: Path) -> None:
    # The one thing a wall-clock-limited result cannot be read without: the same
    # roster four-up is a different measurement from the same roster one-up.
    record = run_benchmark.build_run_record(
        _record_args(tmp_path), tmp_path / "roster.csv", ["a"], ("cbls", "cpsat"), 2, 2
    )

    assert record["concurrency"] == {
        "jobs": 4,
        "large_instance_jobs": 1,
        "cpsat_workers": 1,
        "mem_limit_gb": 6.0,
    }


def test_the_run_record_names_the_machine_and_the_budget(tmp_path: Path) -> None:
    record = run_benchmark.build_run_record(
        _record_args(tmp_path), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )

    machine = record["machine"]
    assert isinstance(machine, dict)
    assert machine["host"]
    assert isinstance(machine["cpu_count"], int)
    assert machine["memory_total_kib"] is None or isinstance(machine["memory_total_kib"], int)
    assert record["budget_seconds"] == 600.0


def test_the_run_record_carries_the_engine_commit_and_solver_versions(tmp_path: Path) -> None:
    record = run_benchmark.build_run_record(
        _record_args(tmp_path), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )

    versions = record["versions"]
    assert isinstance(versions, dict)
    assert versions["engine_commit"] == "abc1234"
    assert versions["python"]


def test_the_run_record_identifies_the_yardstick_the_gaps_will_be_scored_against(
    tmp_path: Path,
) -> None:
    # An upstream revision of the solution file moves every gap in the table, so a
    # later reader has to be able to tell which one a table was scored against.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    record = run_benchmark.build_run_record(
        _record_args(inst_dir), inst_dir / "roster.csv", ["a"], ("cbls",), 1, 1
    )

    references = record["references"]
    assert isinstance(references, dict)
    pinned = references["pinned"]
    assert isinstance(pinned, dict)
    assert set(pinned) == set(run_benchmark.PINNED_REFERENCE_FILES)
    assert references["manifest_sha256"]


def test_the_run_record_names_the_roster_file_actually_read(tmp_path: Path) -> None:
    # `--roster` accepts any CSV, and the reference values a run is scored against
    # are the ones in the file it read -- so quoting only the pinned tables would
    # name a yardstick the run never used.
    inst_dir = _pinned_dir(tmp_path, {"a": b"instance bytes"})
    roster = tmp_path / "elsewhere.csv"
    roster.write_text("instance,reference_value,reference_kind\na,1.0,opt\n")

    record = run_benchmark.build_run_record(_record_args(inst_dir), roster, ["a"], ("cbls",), 1, 1)

    references = record["references"]
    assert isinstance(references, dict)
    assert references["roster_path"] == str(roster)
    assert references["roster_sha256"] == hashlib.sha256(roster.read_bytes()).hexdigest()


def test_the_run_record_says_whether_the_preconditions_were_checked(tmp_path: Path) -> None:
    # --skip-preconditions is what makes a run unpublishable, and stderr does not
    # survive to whoever reads the table.
    skipped = run_benchmark.build_run_record(
        _record_args(tmp_path, skip_preconditions=True),
        tmp_path / "roster.csv",
        ["a"],
        ("cbls",),
        1,
        1,
    )
    checked = run_benchmark.build_run_record(
        _record_args(tmp_path), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )

    assert skipped["run"]["preconditions_checked"] is False  # type: ignore[index]
    assert checked["run"]["preconditions_checked"] is True  # type: ignore[index]


def test_the_drivers_copy_of_the_pinned_reference_files_matches_acquisition() -> None:
    # Deliberately duplicated so the driver needs no import of the roster package,
    # but one of the three names carries a MIPLIB version -- so the copies are tied
    # together here rather than drifting the day the yardstick is revised.
    from benchmarks.instances.mipfeas import download

    assert download.PINNED_REFERENCE_FILES == run_benchmark.PINNED_REFERENCE_FILES


def test_a_resumed_run_adds_a_record_rather_than_overwriting_the_first(tmp_path: Path) -> None:
    # A resumed directory was produced by two machines and two concurrencies;
    # keeping only the last would claim the whole set came off one.
    results = tmp_path / "results"
    first = run_benchmark.build_run_record(
        _record_args(tmp_path, jobs=1), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )
    run_benchmark.append_run_record(results, first)
    second = run_benchmark.build_run_record(
        _record_args(tmp_path, jobs=8), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )
    run_benchmark.append_run_record(results, second)

    runs = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"]
    assert [r["concurrency"]["jobs"] for r in runs] == [1, 8]


def test_closing_the_record_replaces_the_entry_this_run_opened(tmp_path: Path) -> None:
    results = tmp_path / "results"
    record = run_benchmark.build_run_record(
        _record_args(tmp_path), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )
    run_benchmark.append_run_record(results, record)
    assert record["status"] == "running"

    record["status"] = "complete"
    record["outcome"] = {"jobs_run": 1, "failures": 0, "rejected": 0, "wall_seconds": 1.0}
    run_benchmark.close_run_record(results, record)

    runs = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "complete"


def test_an_unfinished_run_still_left_a_record_behind(tmp_path: Path) -> None:
    # Written before the first job, not after the last: a run killed at hour six
    # still has to say what produced the results it did leave.
    results = tmp_path / "results"
    record = run_benchmark.build_run_record(
        _record_args(tmp_path), tmp_path / "roster.csv", ["a"], ("cbls",), 1, 1
    )
    run_benchmark.append_run_record(results, record)

    written = json.loads((results / run_benchmark.RUN_RECORD_FILENAME).read_text())["runs"][0]
    assert written["status"] == "running"
    assert written["finished_at"] is None


# --- A resume that runs nothing must not report a clean run --------------------


def _feasible_row(results_dir: Path, verdict: dict[str, object] | None) -> list[Job]:
    job = Job("cbls", "inst")
    (results_dir / "cbls").mkdir(parents=True, exist_ok=True)
    job.result_path(results_dir).write_text(json.dumps({"status": "feasible", "objective": 1.0}))
    job.solution_path(results_dir).write_text("=obj= 1.0\n")
    if verdict is not None:
        job.verification_path(results_dir).write_text(json.dumps(verdict))
    return [job]


def test_an_exhausted_verification_is_counted_rather_than_forgotten(tmp_path: Path) -> None:
    # After MAX_VERIFY_ATTEMPTS the row stops being retried, so every later resume
    # drops the job and runs nothing. Without this count the driver would print
    # "0 jobs to run" and exit 0 on a row that was never successfully checked.
    jobs = _feasible_row(
        tmp_path,
        {
            "verdict": "error",
            "reason": "verifier_died",
            "attempts": run_benchmark.MAX_VERIFY_ATTEMPTS,
        },
    )

    assert all(run_benchmark.has_usable_result(j, tmp_path, True) for j in jobs)
    assert count_unchecked(jobs, tmp_path, verify=True) == 1


def test_a_feasible_row_with_no_verdict_at_all_is_counted_unchecked(tmp_path: Path) -> None:
    assert count_unchecked(_feasible_row(tmp_path, None), tmp_path, verify=True) == 1


def test_a_checked_row_is_not_counted_unchecked(tmp_path: Path) -> None:
    jobs = _feasible_row(tmp_path, {"verdict": "pass"})
    assert count_unchecked(jobs, tmp_path, verify=True) == 0


def test_a_rejected_row_is_counted_as_rejected_not_as_unchecked(tmp_path: Path) -> None:
    # It *was* checked. Conflating the two would bury the loudest signal this
    # benchmark has under a harness counter.
    jobs = _feasible_row(tmp_path, {"verdict": "fail"})
    assert count_unchecked(jobs, tmp_path, verify=True) == 0
    assert count_rejected(jobs, tmp_path) == 1


def test_nothing_is_unchecked_when_verification_is_off(tmp_path: Path) -> None:
    jobs = _feasible_row(tmp_path, None)
    assert count_unchecked(jobs, tmp_path, verify=False) == 0


def test_a_row_that_found_nothing_has_nothing_to_check(tmp_path: Path) -> None:
    job = Job("cbls", "inst")
    (tmp_path / "cbls").mkdir(parents=True)
    job.result_path(tmp_path).write_text(json.dumps({"status": "no_solution", "objective": None}))

    assert count_unchecked([job], tmp_path, verify=True) == 0


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
    inst_dir = tmp_path / "instances"
    inst_dir.mkdir()
    instance_bytes = gzip.compress(b"NAME tiny\nENDATA\n")
    (inst_dir / "inst.mps.gz").write_bytes(instance_bytes)
    _write_csv(
        inst_dir / "manifest.csv",
        ["instance", "sha256", "bytes"],
        [["inst", hashlib.sha256(instance_bytes).hexdigest(), len(instance_bytes)]],
    )
    roster = tmp_path / "roster.csv"
    _write_csv(roster, ["instance", "reference_value", "reference_kind"], [["inst", 1.0, "opt"]])
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
            "--roster",
            str(roster),
            "--inst-dir",
            str(inst_dir),
            "--results-dir",
            str(results_dir),
            "--cbls-bin",
            str(binary),
            "--engines",
            "cbls",
            "--budget",
            "1",
            "--skip-preconditions",
        ],
    )
    run_benchmark.main()

    assert seen, "execute was never called, so the ordering was not exercised"
    assert all(seen), "a job was dispatched before the machine record was written"


def test_a_resume_with_nothing_left_to_run_still_exits_non_zero(tmp_path: Path) -> None:
    """End to end: the driver over a directory whose only row was never checked."""
    repo_root = Path(__file__).resolve().parents[2]
    binary = repo_root / "build" / "cbls_mipfeas"
    if not binary.exists():
        pytest.skip("cbls_mipfeas not built")

    inst_dir = tmp_path / "instances"
    inst_dir.mkdir()
    instance_bytes = gzip.compress(b"NAME tiny\nENDATA\n")
    (inst_dir / "inst.mps.gz").write_bytes(instance_bytes)
    _write_csv(
        inst_dir / "manifest.csv",
        ["instance", "sha256", "bytes"],
        [["inst", hashlib.sha256(instance_bytes).hexdigest(), len(instance_bytes)]],
    )
    roster = tmp_path / "roster.csv"
    _write_csv(roster, ["instance", "reference_value", "reference_kind"], [["inst", 1.0, "opt"]])
    results_dir = tmp_path / "results"
    _feasible_row(
        results_dir,
        {
            "verdict": "error",
            "reason": "verifier_died",
            "attempts": run_benchmark.MAX_VERIFY_ATTEMPTS,
        },
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "benchmarks" / "mipfeas" / "run_benchmark.py"),
            "--roster",
            str(roster),
            "--inst-dir",
            str(inst_dir),
            "--results-dir",
            str(results_dir),
            "--cbls-bin",
            str(binary),
            "--engines",
            "cbls",
            "--budget",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert "0 jobs to run" in completed.stdout, completed.stdout
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "UNCHECKED" in completed.stderr
