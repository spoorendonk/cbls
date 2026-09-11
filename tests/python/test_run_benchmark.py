"""Tests for the MIPfeas run driver."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.mipfeas.primal_integral import NO_SOLUTION_GAP, score_instance, summarize
from benchmarks.mipfeas.run_benchmark import (
    FAILURE_MARKERS,
    Job,
    build_command,
    drop_completed,
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


def test_a_rejected_solution_counts_as_a_failed_job() -> None:
    # This is the correctness benchmark: a run that published a point the checker
    # refused must not exit 0.
    assert "VERIFY-FAILED" in FAILURE_MARKERS
    assert "VERIFY-ERROR" in FAILURE_MARKERS


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
    (inst_dir / "tiny.mps.gz").write_bytes(gzip.compress(TINY_MPS.encode()))
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
