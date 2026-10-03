"""The MINLPLib rows' DAG-independent feasibility check (#205).

`independent_check` has SCIP read an instance's `.nl` and check the assignment
the runner returned, so an evaluation bug this engine shares with its own
re-check cannot produce a `feasible` row. These tests drive real SCIP: a
stubbed SCIP would test nothing but the stub.
"""

from __future__ import annotations

import csv
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks.minlplib import independent_check
from benchmarks.minlplib.independent_check import (
    SOLUTION_MAGIC,
    Solution,
    check_rows,
    check_solution,
    read_solution,
)
from benchmarks.minlplib.runner import RUNNER_COLUMNS, completed_search

pyscipopt = pytest.importorskip("pyscipopt")

REPO = Path(__file__).resolve().parents[2]
INST = REPO / "benchmarks" / "instances" / "minlplib"
RUNNER = REPO / "build" / "cbls_minlplib"
RUNNER_SOURCE = REPO / "benchmarks" / "minlplib" / "minlplib.cpp"


def sqrt_nl(tmp_path: Path) -> Path:
    """`sqrt(x) >= 2`, x in [-10, 10], minimise x: the issue's repro shape, via SCIP's writer."""
    model = pyscipopt.Model()
    model.hideOutput()
    x = model.addVar("x", lb=-10, ub=10)
    model.addCons(pyscipopt.sqrt(x) >= 2)
    model.setObjective(x)
    path = tmp_path / "sqrtge.nl"
    model.writeProblem(str(path), verbose=False)
    # SCIP's writer leaves .col/.row name files beside it; without them the
    # reader names the column x0, as it does for every MINLPLib instance.
    for side in (".col", ".row"):
        path.with_suffix(side).unlink(missing_ok=True)
    return path


def test_the_solution_header_is_the_one_the_runner_writes() -> None:
    assert f'"{SOLUTION_MAGIC}\\n"' in RUNNER_SOURCE.read_text()


def test_a_domain_error_is_rejected_where_the_old_dag_accepted_it(tmp_path: Path) -> None:
    # The unfixed engine scored x = -10 as feasible: sqrt(-10) read 0.0 or +inf.
    nl = sqrt_nl(tmp_path)
    reason = check_solution(nl, Solution("sqrtge", -10.0, (-10.0,)))
    assert reason is not None and reason.startswith("SCIP rejects the assignment")
    assert check_solution(nl, Solution("sqrtge", 4.0, (4.0,))) is None


def test_a_linear_objective_must_match_scips(tmp_path: Path) -> None:
    nl = sqrt_nl(tmp_path)
    reason = check_solution(nl, Solution("sqrtge", 3.0, (4.0,)))
    assert reason is not None and "SCIP computes 4" in reason


# ex4_1_1: minimise a degree-6 polynomial in x0 in [-2, 11]. SCIP moves the whole
# objective into `objcons` over its `nlobjvar` column.
EX4_1_1_AT_1 = 1.0 - 2.08 + 0.4875 + 7.1 - 3.95 - 1.0 + 0.1


def test_a_nonlinear_minimisation_objective_is_bracketed() -> None:
    nl = INST / "ex4_1_1.nl"
    good = Solution("ex4_1_1", EX4_1_1_AT_1, (1.0,))
    assert check_solution(nl, good) is None
    better = check_solution(nl, replace(good, objective=EX4_1_1_AT_1 - 0.5))
    assert better is not None and "better than SCIP's value" in better
    worse = check_solution(nl, replace(good, objective=EX4_1_1_AT_1 + 0.5))
    assert worse is not None and "worse than SCIP's value" in worse
    outside = check_solution(nl, Solution("ex4_1_1", EX4_1_1_AT_1, (12.0,)))
    assert outside is not None and outside.startswith("SCIP rejects the assignment")


# alkylation: MAXIMISE, a linear objective part plus a nonlinear one in objcons.
# A feasible point the runner returned (engine commit ae704ad, 1s, seed 1).
ALKYLATION = Solution(
    "alkylation",
    494.13321496956542,
    (
        850.71115306966533,
        477.82558529724912,
        4653.9298916780508,
        14.450334798109838,
        560.04202144774263,
        10.91187260280809,
        1.5616363636363644,
        95.0,
        89.748875392964393,
        153.53535353535355,
    ),
)


def test_a_maximisation_objective_with_a_linear_part_is_bracketed() -> None:
    nl = INST / "alkylation.nl"
    assert check_solution(nl, ALKYLATION) is None
    better = check_solution(nl, replace(ALKYLATION, objective=ALKYLATION.objective + 1.0))
    assert better is not None and "better than SCIP's value" in better
    worse = check_solution(nl, replace(ALKYLATION, objective=ALKYLATION.objective - 1.0))
    assert worse is not None and "worse than SCIP's value" in worse


def test_a_column_count_mismatch_is_an_error_not_a_verdict() -> None:
    with pytest.raises(ValueError, match="columns"):
        check_solution(INST / "ex4_1_1.nl", Solution("ex4_1_1", 0.0, (1.0, 2.0)))


def test_read_solution_refuses_a_short_body(tmp_path: Path) -> None:
    path = tmp_path / "a.sol"
    path.write_text(f"{SOLUTION_MAGIC}\ninstance a\nobjective 1\ncolumns 2\n0\n")
    with pytest.raises(ValueError, match="1 values for 2 columns"):
        read_solution(path)


# --- check_rows ----------------------------------------------------------------


def _row(name: str, feasible: str, note: str, objective: str = "1.5") -> dict[str, str]:
    row = dict.fromkeys(RUNNER_COLUMNS, "0")
    row.update(
        instance=name, objective=objective, feasible=feasible, note=note, commit_sha="abc1234"
    )
    row["gap_to_bks%"] = "2"
    row["gap_to_dual%"] = "3"
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RUNNER_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fh:
        return {r["instance"]: r for r in csv.DictReader(fh)}


def test_check_rows_demotes_what_scip_rejects_and_leaves_the_rest(tmp_path: Path) -> None:
    inst = tmp_path / "inst"
    inst.mkdir()
    sqrt_nl(inst).rename(inst / "bad.nl")
    sqrt_nl(inst).rename(inst / "good.nl")
    sols = tmp_path / "sols"
    sols.mkdir()
    for name, x in (("bad", -10.0), ("good", 4.0)):
        (sols / f"{name}.sol").write_text(
            f"{SOLUTION_MAGIC}\ninstance {name}\nobjective {x!r}\ncolumns 1\n{x!r}\n"
        )
    out = tmp_path / "out.csv"
    _write_csv(
        out,
        [
            _row("bad", "true", "feasible"),
            _row("good", "true", "matches-bks"),
            _row("off", "false", "infeasible(residual=1)", objective="NaN"),
        ],
    )

    assert check_rows(out, inst, sols) == ["bad"]
    rows = _read_csv(out)
    bad = rows["bad"]
    assert bad["feasible"] == "false"
    for column in (
        "objective",
        "gap_to_bks%",
        "gap_to_dual%",
        "first_feasible_objective",
        "time_to_first_feasible",
    ):
        assert bad[column] == "NaN", column
    assert bad["note"].startswith("VERIFY-FAILED(independent: SCIP rejects the assignment")
    assert bad["note"].endswith("was feasible)")
    assert "," not in bad["note"]
    assert completed_search(bad["note"]), "a demoted row must still count as a completed search"
    assert rows["good"]["feasible"] == "true"
    assert rows["off"]["note"] == "infeasible(residual=1)"

    before = out.read_bytes()
    assert check_rows(out, inst, sols) == [], "the check is not idempotent"
    assert out.read_bytes() == before


def test_check_rows_refuses_a_verified_row_without_a_solution(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    _write_csv(out, [_row("a", "true", "feasible")])
    with pytest.raises(FileNotFoundError, match="cannot be checked"):
        check_rows(out, tmp_path, tmp_path)


def test_check_rows_refuses_a_solution_for_another_instance(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    _write_csv(out, [_row("a", "true", "feasible")])
    (tmp_path / "a.sol").write_text(f"{SOLUTION_MAGIC}\ninstance b\nobjective 1\ncolumns 1\n0\n")
    with pytest.raises(ValueError, match="is for b"):
        check_rows(out, tmp_path, tmp_path)


# --- the runner end to end ------------------------------------------------------


def test_the_runner_writes_a_solution_scip_accepts_for_a_verified_row(tmp_path: Path) -> None:
    if not RUNNER.exists():
        pytest.skip("cbls_minlplib not built")
    out = tmp_path / "out.csv"
    sols = tmp_path / "sols"
    sols.mkdir()
    result = subprocess.run(
        [
            str(RUNNER),
            str(INST),
            "--instance",
            "ex4_1_1",
            "--time-limit",
            "0.5",
            "--out",
            str(out),
            "--solution-dir",
            str(sols),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    row = _read_csv(out)["ex4_1_1"]
    assert row["feasible"] == "true", row["note"]
    solution = read_solution(sols / "ex4_1_1.sol")
    assert solution.instance == "ex4_1_1"
    assert len(solution.values) == 1
    # The CSV cell is rounded to six digits; the solution file is not.
    assert float(row["objective"]) == pytest.approx(solution.objective, rel=1e-5)
    assert re.fullmatch(r"-?[\d.e+-]+", repr(solution.objective))
    assert check_rows(out, INST, sols) == []
    assert not list(sols.glob("*.tmp")), "the temp file was left behind"


def test_the_runner_refuses_to_write_the_published_table_even_on_protocol(
    tmp_path: Path,
) -> None:
    """Only run_benchmark.py publishes: it is what runs the independent check."""
    if not RUNNER.exists():
        pytest.skip("cbls_minlplib not built")
    inst = tmp_path / "minlplib"
    inst.mkdir()
    (inst / "bounds.csv").write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound\nnosuch,NLP,1,1,min,1,1\n"
    )
    published = inst / "comparison.csv"
    published.write_text("published rows nobody may overwrite\n")
    result = subprocess.run(
        [str(RUNNER), str(inst), "--commit", "deadbee"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 2, result.stdout
    assert "written only by benchmarks/minlplib/run_benchmark.py" in result.stderr
    assert published.read_text() == "published rows nobody may overwrite\n"


def test_the_check_module_names_the_note_prefix_the_scorer_allowlists() -> None:
    assert completed_search(independent_check.DEMOTED_NOTE_PREFIX)


def test_columns_follow_the_nl_column_index_not_scips_type_order() -> None:
    # SCIP lists nvs08's variables integers first (i1, i2, x0); the solution
    # file is in NL column order, so mapping by position would load x0's value
    # into i1. Found on the first smoke run of this check.
    model = pyscipopt.Model()
    model.hideOutput()
    model.readProblem(str(INST / "nvs08.nl"))
    assert [v.name for v in model.getVars() if v.name != "nlobjvar"] == ["i1", "i2", "x0"]
    columns, _, _ = independent_check._columns(model, 3)
    assert [v.name for v in columns] == ["x0", "i1", "i2"]


def test_an_objective_constant_column_is_loaded_at_its_fixed_value(tmp_path: Path) -> None:
    """heldout/ex9_2_3: SCIP adds a fixed `objconstant` column (-60) for the constant."""
    nl = INST / "heldout" / "ex9_2_3.nl"
    model = pyscipopt.Model()
    model.hideOutput()
    model.readProblem(str(nl))
    columns, _, fixed = independent_check._columns(model, 16)
    assert len(columns) == 16
    assert [(v.name, value) for v, value in fixed] == [("objconstant", -60.0)]
    # A feasible point of SCIP's own, read back as the runner would write it,
    # passes; the same point claiming another objective does not.
    model.setParam("limits/time", 20)
    model.optimize()
    best = model.getBestSol()
    values = tuple(float(model.getSolVal(best, v)) for v in columns)
    objective = float(model.getSolObjVal(best))
    good = Solution("ex9_2_3", objective, values)
    assert check_solution(nl, good) is None
    assert check_solution(nl, replace(good, objective=objective - 5.0)) is not None


def test_a_column_name_without_an_index_is_refused_not_guessed(tmp_path: Path) -> None:
    model = pyscipopt.Model()
    model.hideOutput()
    x = model.addVar("x", lb=0, ub=1)
    model.addCons(x <= 1)
    model.setObjective(x)
    path = tmp_path / "named.nl"
    model.writeProblem(str(path), verbose=False)  # with its .col file: the column is "x"
    with pytest.raises(ValueError, match="carries no column index"):
        check_solution(path, Solution("named", 0.0, (0.0,)))
