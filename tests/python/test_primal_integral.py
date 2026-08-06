"""Tests for the MIPfeas Primal Integral scorer."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from benchmarks.mipfeas.primal_integral import (
    NO_SOLUTION_GAP,
    SIGN_FLIP_GAP,
    primal_gap,
    primal_integral,
    score_instance,
    shifted_geometric_mean,
    summarize,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_primal_gap_is_zero_at_the_reference() -> None:
    assert primal_gap(100.0, 100.0) == 0.0


def test_primal_gap_without_a_solution_is_two() -> None:
    assert primal_gap(None, 100.0) == NO_SOLUTION_GAP


def test_primal_gap_across_a_sign_change_is_one() -> None:
    assert primal_gap(5.0, -3.0) == SIGN_FLIP_GAP
    assert primal_gap(-5.0, 3.0) == SIGN_FLIP_GAP


def test_primal_gap_is_zero_when_both_values_are_near_zero() -> None:
    # Without the near-zero rule this would score 1.0 (a 100% relative gap) for
    # what is numerically the same answer.
    assert primal_gap(1e-9, 0.0) == 0.0


def test_primal_gap_normalizes_by_the_larger_magnitude() -> None:
    # |200 - 100| / max(200, 100)
    assert primal_gap(200.0, 100.0) == pytest.approx(0.5)


def test_primal_gap_of_a_zero_reference_against_a_nonzero_incumbent() -> None:
    assert primal_gap(5.0, 0.0) == pytest.approx(1.0)


def test_primal_integral_of_an_immediate_optimum_is_zero() -> None:
    assert primal_integral([(0.0, 10.0)], reference=10.0, budget=60.0) == pytest.approx(0.0)


def test_primal_integral_without_any_solution_is_two() -> None:
    assert primal_integral([], reference=10.0, budget=60.0) == NO_SOLUTION_GAP


def test_primal_integral_charges_two_until_the_first_solution() -> None:
    # No incumbent for the first 30s of 60, then the optimum: 2 * (30/60) = 1.0.
    assert primal_integral([(30.0, 10.0)], reference=10.0, budget=60.0) == pytest.approx(1.0)


def test_primal_integral_integrates_a_step_function() -> None:
    # 0-10s: no solution        -> 2.0 for 10s
    # 10-40s: obj 200 (gap 0.5) -> 0.5 for 30s
    # 40-60s: obj 100 (gap 0)   -> 0.0 for 20s
    trace = [(10.0, 200.0), (40.0, 100.0)]
    expected = (2.0 * 10 + 0.5 * 30 + 0.0 * 20) / 60
    assert primal_integral(trace, reference=100.0, budget=60.0) == pytest.approx(expected)


def test_primal_integral_holds_the_last_incumbent_to_the_budget() -> None:
    # A solution at t=0 that is never improved: constant gap over the whole budget.
    assert primal_integral([(0.0, 200.0)], reference=100.0, budget=60.0) == pytest.approx(0.5)


def test_primal_integral_clamps_entries_past_the_budget() -> None:
    # A solution logged after the deadline cannot retroactively improve the score.
    assert primal_integral([(90.0, 100.0)], reference=100.0, budget=60.0) == NO_SOLUTION_GAP


def test_primal_integral_sorts_an_out_of_order_trace() -> None:
    ordered = primal_integral([(10.0, 200.0), (40.0, 100.0)], reference=100.0, budget=60.0)
    shuffled = primal_integral([(40.0, 100.0), (10.0, 200.0)], reference=100.0, budget=60.0)
    assert shuffled == pytest.approx(ordered)


def test_primal_integral_holds_the_best_of_several_incumbents_at_one_timestamp() -> None:
    # CP-SAT logs to 0.01s and reports bursts of improvements inside one tick.
    # A plain tuple sort would apply the worst of the burst last and hold it.
    burst = primal_integral([(10.0, 200.0), (10.0, 100.0)], reference=100.0, budget=20.0)
    best_only = primal_integral([(10.0, 100.0)], reference=100.0, budget=20.0)
    assert burst == pytest.approx(best_only)


def test_primal_integral_stays_within_zero_and_two() -> None:
    assert 0.0 <= primal_integral([(5.0, -50.0)], reference=100.0, budget=60.0) <= 2.0


def test_primal_integral_rejects_a_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        primal_integral([(1.0, 1.0)], reference=1.0, budget=0.0)


def test_shifted_geometric_mean_tolerates_zero() -> None:
    # An instance solved immediately (PI = 0) must not collapse the mean to zero.
    assert shifted_geometric_mean([0.0, 1.0]) > 0.0


def test_shifted_geometric_mean_of_equal_values_is_that_value() -> None:
    assert shifted_geometric_mean([0.5, 0.5, 0.5]) == pytest.approx(0.5)


def _write_result(
    directory: Path,
    engine: str,
    instance: str,
    record: dict[str, object],
    trace: list[tuple[float, float]] | None = None,
) -> None:
    engine_dir = directory / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / f"{instance}.json").write_text(json.dumps(record))
    if trace is not None:
        lines = ["time_seconds,objective"] + [f"{t},{o}" for t, o in trace]
        (engine_dir / f"{instance}.trace.csv").write_text("\n".join(lines) + "\n")


def test_score_instance_reads_a_result_and_its_trace(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        "cbls",
        "inst",
        {"status": "feasible", "objective": 200.0, "wall_seconds": 60.0, "commit_sha": "abc1234"},
        trace=[(30.0, 200.0)],
    )
    scored = score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)
    assert scored.status == "feasible"
    assert scored.final_gap == pytest.approx(0.5)
    # 2.0 for the first 30s, then 0.5 for the rest.
    assert scored.primal_integral == pytest.approx((2.0 * 30 + 0.5 * 30) / 60)
    assert scored.provenance == "abc1234"


def test_score_instance_without_a_result_is_not_run(tmp_path: Path) -> None:
    scored = score_instance("absent", "cbls", 100.0, "opt", tmp_path, budget=60.0)
    assert scored.status == "not_run"
    assert math.isnan(scored.primal_integral)


def test_score_instance_falls_back_to_the_final_objective_without_a_trace(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        "cpsat",
        "inst",
        {"status": "feasible", "objective": 100.0, "ortools_version": "9.15"},
    )
    scored = score_instance("inst", "cpsat", 100.0, "opt", tmp_path, budget=60.0)
    # The solution is credited at the buzzer, so the gap is 2 for the whole budget.
    assert scored.primal_integral == pytest.approx(NO_SOLUTION_GAP)
    assert scored.provenance == "9.15"


def test_score_instance_ignores_a_trace_when_the_run_found_nothing(tmp_path: Path) -> None:
    # A stale trace from an earlier run must not score an infeasible result.
    _write_result(
        tmp_path,
        "cbls",
        "inst",
        {"status": "no_solution", "objective": None},
        trace=[(1.0, 100.0)],
    )
    scored = score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)
    assert scored.primal_integral == NO_SOLUTION_GAP


def test_score_instance_rejects_a_result_from_a_different_budget(tmp_path: Path) -> None:
    # The driver resumes on file existence and defaults to one results directory
    # whatever the budget, so a 60s smoke run and a 600s run land on top of each
    # other. Holding a 60s incumbent over 600s would score better than the run
    # earned, so this must fail rather than publish it.
    _write_result(
        tmp_path,
        "cbls",
        "inst",
        {"status": "feasible", "objective": 100.0, "budget_seconds": 60.0},
        trace=[(1.0, 100.0)],
    )
    with pytest.raises(ValueError, match="60.0s budget but is being scored at 600"):
        score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=600.0)


def test_score_instance_rejects_a_truncated_result_file(tmp_path: Path) -> None:
    engine_dir = tmp_path / "cbls"
    engine_dir.mkdir(parents=True)
    (engine_dir / "inst.json").write_text('{"status": "feasi')
    with pytest.raises(ValueError, match="not valid JSON"):
        score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)


def test_score_instance_rejects_a_non_finite_trace(tmp_path: Path) -> None:
    # One NaN would otherwise turn the geometric mean, the arithmetic mean and the
    # median all into NaN, with no warning anywhere.
    _write_result(
        tmp_path,
        "cbls",
        "inst",
        {"status": "feasible", "objective": 100.0},
        trace=[(1.0, float("nan"))],
    )
    with pytest.raises(ValueError, match="non-finite"):
        score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)


def test_summarize_excludes_not_run_instances_from_the_aggregates(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        "cbls",
        "solved",
        {"status": "feasible", "objective": 100.0},
        trace=[(0.0, 100.0)],
    )
    rows = [
        score_instance("solved", "cbls", 100.0, "opt", tmp_path, budget=60.0),
        score_instance("absent", "cbls", 100.0, "opt", tmp_path, budget=60.0),
    ]
    summary = summarize(rows, "cbls")
    assert summary.scored == 1
    assert summary.not_run == 1
    assert summary.feasible == 1
    assert summary.matched_reference == 1
    assert summary.arithmetic_mean == pytest.approx(0.0)
