"""Tests for the MIPfeas independent solution verifier.

The verifier is the only thing in this benchmark that reads the ORIGINAL
instance file, so a defect here is invisible everywhere else: it would report
`pass` on solutions nobody checked and the correctness benchmark would go on
publishing them. Every test therefore drives a real MPS file through a real
third-party reader, and half of them hand it a solution that must be rejected.

The instance is written by the test rather than taken from the roster: the 233
roster instances are gitignored (~546 MiB) and this suite runs in a fresh clone.
It is small but not trivial -- an integer and two continuous columns, an L row
with a RANGES entry (so the row has a lower bound the file never states
directly), a G row, an E row, and an objective constant carried on the objective
row's RHS. Each of those is a place a reader can quietly differ.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("pyscipopt", reason="pyscipopt is in the 'benchmarks' extra, not 'dev'")

from benchmarks.mipfeas.verify_solution import (  # noqa: E402
    ERROR,
    FAIL,
    INTEGRALITY_TOLERANCE,
    PASS,
    check,
    parse_solution,
    read_instance,
    verify_result,
    write_solution,
)

if TYPE_CHECKING:
    from benchmarks.mipfeas.verify_solution import Instance

REPO_ROOT = Path(__file__).resolve().parents[2]

#: min x + 2y - z + 5
#:   6 <= x + y <= 10     (L row at 10 with a RANGES entry of 4)
#:        x + z >= 2
#:       2x - z  = 4
#:   x integer in [0, 8], y in [-3, 6], z in [0, 5]
#:
#: Optimum: x=4, y=2, z=4, objective 9 -- derived by hand below, and reached by
#: both engines in the end-to-end tests at the bottom.
TINY_MPS = """NAME          TINY
ROWS
 N  COST
 L  C1
 G  C2
 E  C3
COLUMNS
    MARKER                 'MARKER'                 'INTORG'
    x         COST             1.0   C1               1.0
    x         C2               1.0   C3               2.0
    MARKER                 'MARKER'                 'INTEND'
    y         COST             2.0   C1               1.0
    z         COST            -1.0   C2               1.0
    z         C3              -1.0
RHS
    RHS       COST            -5.0   C1              10.0
    RHS       C2               2.0   C3               4.0
RANGES
    RNG       C1               4.0
BOUNDS
 UI BND       x                8.0
 LO BND       y               -3.0
 UP BND       y                6.0
 UP BND       z                5.0
ENDATA
"""

#: The known optimum of TINY_MPS, and its objective.
OPTIMUM = {"x": 4.0, "y": 2.0, "z": 4.0}
OPTIMUM_OBJECTIVE = 9.0


@pytest.fixture
def instance_file(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.mps"
    path.write_text(TINY_MPS)
    return path


@pytest.fixture
def instance(instance_file: Path) -> Instance:
    return read_instance(instance_file)


def test_the_reader_sees_the_program_the_file_declares(instance: Instance) -> None:
    # Not a tautology: RANGES turns an L row into a two-sided one, and the
    # objective constant lives on the objective row's RHS with its sign flipped.
    # A reader that dropped either would accept solutions this one rejects.
    assert instance.objective_offset == pytest.approx(5.0)
    assert instance.columns["x"].integral
    assert not instance.columns["y"].integral
    assert instance.columns["y"].lower == pytest.approx(-3.0)
    c1 = next(row for row in instance.rows if row.name == "C1")
    assert (c1.lower, c1.upper) == pytest.approx((6.0, 10.0))


def test_the_known_solution_passes(instance: Instance) -> None:
    verified = check(instance, OPTIMUM, OPTIMUM_OBJECTIVE)
    assert verified.verdict == PASS
    assert verified.reason == ""
    assert verified.failed_checks == []
    assert not verified.marginal
    # The objective is recomputed from the file, constant included, rather than
    # copied from what the engine reported.
    assert verified.objective_recomputed == pytest.approx(OPTIMUM_OBJECTIVE)


def test_a_corrupted_solution_is_rejected(instance: Instance) -> None:
    # One variable moved by 1: still integral, still inside every bound, and it
    # even keeps the equality row satisfied -- it violates only C1's RANGES-derived
    # lower bound, which is the half of the file a careless reader loses.
    corrupted = {**OPTIMUM, "y": 1.0}
    verified = check(instance, corrupted, OPTIMUM_OBJECTIVE)
    assert verified.verdict == FAIL
    assert verified.reason == "row_violation"
    assert verified.worst_row == "C1"
    assert verified.max_row_violation == pytest.approx(1.0)


def test_a_solution_outside_a_variable_bound_is_rejected(instance: Instance) -> None:
    verified = check(instance, {**OPTIMUM, "z": 5.5, "x": 4.75}, 8.25)
    assert verified.verdict == FAIL
    assert "bound_violation" in verified.failed_checks
    assert verified.worst_bound_column == "z"


def test_a_fractional_integer_is_rejected(instance: Instance) -> None:
    # x=4.5, z=5 keeps 2x - z = 4 exactly, so only integrality is violated.
    verified = check(instance, {"x": 4.5, "y": 1.5, "z": 5.0}, 7.5)
    assert verified.verdict == FAIL
    assert verified.reason == "integrality_violation"
    assert verified.worst_integral_column == "x"


def test_an_objective_that_is_not_the_point_s_objective_is_rejected(instance: Instance) -> None:
    # The feasible optimum, published under a better number. Nothing else in the
    # harness can catch this: the reference check only fires below a proven
    # optimum, and the engine's own drift check reads its own DAG.
    verified = check(instance, OPTIMUM, OPTIMUM_OBJECTIVE - 1.0)
    assert verified.verdict == FAIL
    assert verified.reason == "objective_mismatch"
    assert verified.objective_violation == pytest.approx(1.0)


def test_a_violation_inside_the_tolerance_passes_but_is_marked_marginal(
    instance: Instance,
) -> None:
    # Half the row tolerance: the threshold is the rule, so this passes -- and it
    # is flagged, so "just inside" is visible rather than invisible.
    verified = check(instance, {**OPTIMUM, "y": 2.0 - 5e-7}, OPTIMUM_OBJECTIVE - 1e-6)
    assert verified.verdict == PASS
    assert verified.marginal
    assert 0.1 <= verified.max_row_ratio <= 1.0


def test_a_violation_outside_the_tolerance_fails(instance: Instance) -> None:
    # Ten times the tolerance, on the same row as the marginal case above: the
    # verdict turns on the threshold and on nothing else.
    verified = check(instance, {**OPTIMUM, "y": 2.0 - 1e-5}, OPTIMUM_OBJECTIVE - 2e-5)
    assert verified.verdict == FAIL
    assert verified.reason == "row_violation"


def test_an_integer_just_off_by_more_than_the_tolerance_fails(instance: Instance) -> None:
    off = 10 * INTEGRALITY_TOLERANCE
    verified = check(instance, {"x": 4.0 + off, "y": 2.0 - off, "z": 4.0 + 2 * off}, 9.0 - off)
    assert verified.verdict == FAIL
    assert "integrality_violation" in verified.failed_checks


def test_a_solution_missing_a_column_is_an_error_not_a_pass(instance: Instance) -> None:
    # A partial dump must never read as "everything checked out".
    verified = check(instance, {"x": 4.0, "y": 2.0}, OPTIMUM_OBJECTIVE)
    assert verified.verdict == ERROR
    assert verified.reason == "solution_variable_mismatch"


def test_a_solution_naming_an_unknown_column_is_an_error(instance: Instance) -> None:
    verified = check(instance, {**OPTIMUM, "w": 1.0}, OPTIMUM_OBJECTIVE)
    assert verified.verdict == ERROR
    assert verified.reason == "solution_variable_mismatch"


def test_parse_solution_reads_the_format_the_runners_write() -> None:
    values, objective = parse_solution(
        "# instance tiny\n# engine cbls\n=obj= 9.0\n\nx 4\ny 2.0\nz 4e0\n"
    )
    assert values == OPTIMUM
    assert objective == pytest.approx(OPTIMUM_OBJECTIVE)


def test_parse_solution_rejects_a_repeated_variable() -> None:
    with pytest.raises(ValueError, match="appears twice"):
        parse_solution("x 1\nx 2\n")


def test_parse_solution_rejects_a_malformed_line() -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_solution("x 1 2\n")


def _write_job(directory: Path, values: dict[str, float], objective: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "tiny.json").write_text(
        json.dumps({"engine": "cbls", "status": "feasible", "objective": objective})
    )
    write_solution(directory / "tiny.sol", "tiny", "cbls", objective, values)
    return directory


def test_verify_result_reads_the_files_a_job_leaves_behind(tmp_path: Path) -> None:
    result_dir = _write_job(tmp_path / "cbls", OPTIMUM, OPTIMUM_OBJECTIVE)
    verified = verify_result("tiny", _gzipped_instance(tmp_path), result_dir)
    assert verified.verdict == PASS
    assert verified.engine == "cbls"
    assert verified.checker.startswith("SCIP ")


def test_verify_result_rejects_a_corrupted_solution_file(tmp_path: Path) -> None:
    result_dir = _write_job(tmp_path / "cbls", {**OPTIMUM, "y": 1.0}, OPTIMUM_OBJECTIVE)
    assert verify_result("tiny", _gzipped_instance(tmp_path), result_dir).verdict == FAIL


def test_verify_result_without_a_solution_file_is_an_error(tmp_path: Path) -> None:
    # The case the scorer must not read as "checked and fine": a feasible row
    # whose solution was never written cannot be checked at all.
    result_dir = tmp_path / "cbls"
    result_dir.mkdir()
    (result_dir / "tiny.json").write_text(json.dumps({"status": "feasible", "objective": 9.0}))
    verified = verify_result("tiny", _gzipped_instance(tmp_path), result_dir)
    assert verified.verdict == ERROR
    assert verified.reason == "missing_solution_file"


def test_verify_result_without_the_instance_is_an_error(tmp_path: Path) -> None:
    result_dir = _write_job(tmp_path / "cbls", OPTIMUM, OPTIMUM_OBJECTIVE)
    verified = verify_result("tiny", tmp_path / "nowhere", result_dir)
    assert verified.verdict == ERROR
    assert verified.reason == "missing_instance"


def test_verify_result_of_an_unreadable_instance_is_an_error(tmp_path: Path) -> None:
    inst_dir = tmp_path / "instances"
    inst_dir.mkdir()
    (inst_dir / "tiny.mps.gz").write_bytes(b"not a gzip stream")
    result_dir = _write_job(tmp_path / "cbls", OPTIMUM, OPTIMUM_OBJECTIVE)
    verified = verify_result("tiny", inst_dir, result_dir)
    assert verified.verdict == ERROR
    assert verified.reason == "instance_read_error"


def _gzipped_instance(tmp_path: Path) -> Path:
    inst_dir = tmp_path / "instances"
    inst_dir.mkdir(exist_ok=True)
    (inst_dir / "tiny.mps.gz").write_bytes(gzip.compress(TINY_MPS.encode()))
    return inst_dir


def _run_cli(inst_dir: Path, result_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "mipfeas" / "verify_solution.py"),
            "--instance",
            "tiny",
            "--inst-dir",
            str(inst_dir),
            "--result-dir",
            str(result_dir),
        ],
        capture_output=True,
        text=True,
    )


def test_the_cli_exits_zero_and_writes_a_verdict_for_a_good_solution(tmp_path: Path) -> None:
    inst_dir = _gzipped_instance(tmp_path)
    result_dir = _write_job(tmp_path / "cbls", OPTIMUM, OPTIMUM_OBJECTIVE)
    assert _run_cli(inst_dir, result_dir).returncode == 0
    verdict = json.loads((result_dir / "tiny.verify.json").read_text())
    assert verdict["verdict"] == PASS
    # The thresholds the verdict was reached under travel with it, so a table can
    # state its own rather than whatever the scorer was written against.
    assert verdict["tolerances"]["row_absolute"] == pytest.approx(1e-6)


def test_the_cli_exits_one_on_a_rejected_solution(tmp_path: Path) -> None:
    # The driver keys on this: exit 1 is "checked and wrong", exit 2 is "could
    # not check", and 0 must not cover either.
    inst_dir = _gzipped_instance(tmp_path)
    result_dir = _write_job(tmp_path / "cbls", {**OPTIMUM, "y": 1.0}, OPTIMUM_OBJECTIVE)
    completed = _run_cli(inst_dir, result_dir)
    assert completed.returncode == 1
    assert json.loads((result_dir / "tiny.verify.json").read_text())["verdict"] == FAIL


def test_the_cli_exits_two_when_it_cannot_check(tmp_path: Path) -> None:
    inst_dir = _gzipped_instance(tmp_path)
    result_dir = tmp_path / "cbls"
    result_dir.mkdir()
    (result_dir / "tiny.json").write_text(json.dumps({"status": "feasible", "objective": 9.0}))
    assert _run_cli(inst_dir, result_dir).returncode == 2


# ---------------------------------------------------------------------------
# End to end: what each runner actually writes, checked against the file it read.


def test_the_cbls_runner_writes_a_solution_that_verifies(tmp_path: Path) -> None:
    binary = REPO_ROOT / "build" / "cbls_mipfeas"
    if not binary.exists():
        pytest.skip("cbls_mipfeas not built")
    inst_dir = _gzipped_instance(tmp_path)
    out_dir = tmp_path / "cbls"
    completed = subprocess.run(
        [
            str(binary),
            "--instance",
            "tiny",
            "--inst-dir",
            str(inst_dir),
            "--out-dir",
            str(out_dir),
            "--solution-dir",
            str(out_dir),
            "--budget",
            "1",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "feasible" in completed.stdout
    verified = verify_result("tiny", inst_dir, out_dir)
    assert verified.verdict == PASS, verified.message
    # Solved to the known optimum, so the check above is checking the right point
    # rather than an easy one.
    assert verified.objective_recomputed == pytest.approx(OPTIMUM_OBJECTIVE, abs=1e-6)


def test_the_cpsat_baseline_writes_a_solution_that_verifies(tmp_path: Path) -> None:
    # Verification runs on both engines' solutions: the baseline's objective is
    # otherwise taken entirely on trust, and it reads the instance with its own
    # MPS reader, which is exactly what this check is independent of.
    pytest.importorskip("ortools", reason="ortools is in the 'benchmarks' extra, not 'dev'")
    inst_dir = _gzipped_instance(tmp_path)
    out_dir = tmp_path / "cpsat"
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "mipfeas" / "cpsat_solve.py"),
            "--instance",
            "tiny",
            "--inst-dir",
            str(inst_dir),
            "--out-dir",
            str(out_dir),
            "--solution-dir",
            str(out_dir),
            "--budget",
            "5",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    verified = verify_result("tiny", inst_dir, out_dir)
    assert verified.verdict == PASS, verified.message
    assert verified.engine == "cpsat"
    assert verified.objective_recomputed == pytest.approx(OPTIMUM_OBJECTIVE, abs=1e-6)
