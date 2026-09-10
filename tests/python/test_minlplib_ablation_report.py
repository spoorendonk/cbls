"""Unit tests for the MINLPLib ablation scoring (issue #143).

The arithmetic is hand-computed in the assertions rather than recomputed from
the module, because the point of these tests is the definitions: which instances
may be averaged, what the noise floor is derived from, and what "inside the
noise" means. A test that recomputed the formula would agree with any formula.
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib.ablation_report import (
    ARM_ONLY_FEASIBLE,
    BOTH_FEASIBLE,
    CONTROL_ARM,
    CONTROL_ONLY_FEASIBLE,
    NEITHER_FEASIBLE,
    NO_COMPARABLE_GAP,
    NOISE_Z,
    PROBE_ARM_NAME,
    Cell,
    build_cells,
    classify,
    compare,
    control_spreads,
    load_rows,
    noise_floor,
    render_report,
    scored_instances,
    summarize_arm,
)
from benchmarks.minlplib.run_benchmark import CLAIM_EXCLUDED

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

COLUMNS = (
    "instance",
    "arm",
    "arm_flags",
    "seed",
    "time_limit",
    "commit_sha",
    "objective",
    "primal_bks",
    "dual_bound",
    "gap_to_bks%",
    "gap_to_dual%",
    "wall_seconds",
    "feasible",
    "note",
    "max_violation",
    "n_int_vars",
    "lns_repairs",
    "search_config",
)


def cell(
    instance: str, arm: str, gaps: Sequence[float], *, runs: int = 3, feasible: int | None = None
) -> Cell:
    return Cell(
        instance=instance,
        arm=arm,
        runs=runs,
        feasible_runs=len(gaps) if feasible is None else feasible,
        gaps=tuple(gaps),
        repairs=(0.0,) * runs,
    )


def write_results(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        for row in rows:
            full = dict.fromkeys(COLUMNS, "")
            full.update({"wall_seconds": "60", "lns_repairs": "0", "seed": "1"})
            full.update({k: str(v) for k, v in row.items()})
            writer.writerow([full[column] for column in COLUMNS])
    return path


def run_rows(
    instance: str, arm: str, gaps: Sequence[float | None], *, repairs: int = 0
) -> list[dict[str, object]]:
    """One row per seed; a None gap is an infeasible run."""
    return [
        {
            "instance": instance,
            "arm": arm,
            "seed": seed,
            "feasible": "true" if gap is not None else "false",
            "gap_to_bks%": "NaN" if gap is None else gap,
            "lns_repairs": repairs,
        }
        for seed, gap in enumerate(gaps, start=1)
    ]


# --- buckets: nothing is silently dropped and no NaN reaches an aggregate ------


def test_a_both_feasible_delta_is_the_arm_mean_minus_the_control_mean() -> None:
    comparison = compare(
        cell("a", CONTROL_ARM, [10.0, 12.0, 14.0]), cell("a", "x", [20.0, 22.0, 24.0])
    )
    assert comparison.bucket == BOTH_FEASIBLE
    assert comparison.delta == pytest.approx(10.0)  # 22 - 12; positive is worse


def test_an_arm_feasible_where_the_control_is_not_has_no_delta_and_is_still_counted() -> None:
    """There is no control gap to subtract. The instance is a categorical win,
    reported as one -- not averaged in as a NaN and not dropped."""
    comparison = compare(cell("a", CONTROL_ARM, [], runs=3), cell("a", "x", [7.0, 8.0, 9.0]))
    assert comparison.bucket == ARM_ONLY_FEASIBLE
    assert comparison.delta is None
    assert comparison.feasibility_delta == 3


def test_a_control_only_feasible_instance_is_bucketed_not_dropped() -> None:
    comparison = compare(cell("a", CONTROL_ARM, [7.0, 8.0, 9.0]), cell("a", "x", [], runs=3))
    assert comparison.bucket == CONTROL_ONLY_FEASIBLE
    assert comparison.delta is None
    assert comparison.feasibility_delta == -3


def test_an_instance_neither_side_solved_is_its_own_bucket() -> None:
    comparison = compare(cell("a", CONTROL_ARM, [], runs=3), cell("a", "x", [], runs=3))
    assert comparison.bucket == NEITHER_FEASIBLE
    assert comparison.delta is None


def test_a_feasible_instance_with_no_published_bound_has_no_comparable_gap() -> None:
    """Feasible on both sides but no finite gap -- no BKS, or a non-finite
    objective. It is not `both-feasible` and must not be averaged."""
    control = Cell("a", CONTROL_ARM, runs=3, feasible_runs=3, gaps=(), repairs=(0.0, 0.0, 0.0))
    treatment = Cell("a", "x", runs=3, feasible_runs=3, gaps=(), repairs=(0.0, 0.0, 0.0))
    assert classify(control, treatment) == NO_COMPARABLE_GAP
    assert compare(control, treatment).delta is None


def test_a_partly_feasible_arm_is_compared_on_its_feasible_runs_and_the_counts_show_it() -> None:
    comparison = compare(
        cell("a", CONTROL_ARM, [10.0, 10.0, 10.0]), cell("a", "x", [4.0], runs=3, feasible=1)
    )
    assert comparison.bucket == BOTH_FEASIBLE
    assert comparison.delta == pytest.approx(-6.0)
    assert comparison.feasibility_delta == -2  # the loss is visible, not hidden in the mean


def test_a_nan_gap_never_reaches_a_cell(tmp_path: Path) -> None:
    results = write_results(tmp_path / "r.csv", run_rows("a", CONTROL_ARM, [10.0, None, 14.0]))
    cells = build_cells(load_rows(results))
    assert cells[("a", CONTROL_ARM)].gaps == (10.0, 14.0)
    assert cells[("a", CONTROL_ARM)].feasible_runs == 2
    assert cells[("a", CONTROL_ARM)].mean_gap == pytest.approx(12.0)


# --- the measured noise floor --------------------------------------------------


def test_the_floor_is_computed_from_the_control_s_own_across_seed_spread() -> None:
    """s = stdev([10, 12, 14]) = 2; scale = sqrt(1/3 + 1/3); floor = 2 * s * scale."""
    control = cell("a", CONTROL_ARM, [10.0, 12.0, 14.0])
    treatment = cell("a", "x", [20.0, 22.0, 24.0])
    spreads = control_spreads({("a", CONTROL_ARM): control}, ["a"])
    assert spreads["a"] == pytest.approx(2.0)
    floor = noise_floor([compare(control, treatment)], spreads)
    expected = NOISE_Z * 2.0 * math.sqrt(1 / 3 + 1 / 3)
    assert floor.per_instance["a"] == pytest.approx(expected)
    assert floor.aggregate == pytest.approx(expected)  # one instance: the mean is that instance
    assert floor.measured == 1
    assert floor.imputed == 0


def test_the_aggregate_floor_is_the_floor_on_the_mean_of_the_deltas() -> None:
    """Two identical instances: averaging two independent deltas halves the
    standard error by sqrt(2), so the aggregate floor is the per-instance one
    divided by sqrt(2) -- not equal to it, and not the sum."""
    comparisons = [
        compare(cell(name, CONTROL_ARM, [10.0, 12.0, 14.0]), cell(name, "x", [20.0, 22.0, 24.0]))
        for name in ("a", "b")
    ]
    spreads = {"a": 2.0, "b": 2.0}
    floor = noise_floor(comparisons, spreads)
    per_instance = NOISE_Z * 2.0 * math.sqrt(1 / 3 + 1 / 3)
    assert floor.aggregate == pytest.approx(per_instance / math.sqrt(2))


def test_an_instance_without_two_control_runs_borrows_the_median_spread() -> None:
    """Dropping it instead would shrink the floor by discarding exactly the
    instances the control found hardest."""
    comparisons = [
        compare(cell("a", CONTROL_ARM, [10.0, 12.0, 14.0]), cell("a", "x", [11.0, 13.0, 15.0])),
        compare(
            cell("b", CONTROL_ARM, [5.0], runs=3, feasible=1),
            cell("b", "x", [6.0], runs=3, feasible=1),
        ),
    ]
    floor = noise_floor(
        comparisons,
        control_spreads(
            {
                ("a", CONTROL_ARM): cell("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
                ("b", CONTROL_ARM): cell("b", CONTROL_ARM, [5.0], runs=3, feasible=1),
            },
            ["a", "b"],
        ),
    )
    assert floor.measured == 1
    assert floor.imputed == 1
    assert floor.median_spread == pytest.approx(2.0)
    # b has one run per side, so its scale is sqrt(1/1 + 1/1) = sqrt(2).
    assert floor.per_instance["b"] == pytest.approx(NOISE_Z * 2.0 * math.sqrt(2.0))


def test_a_campaign_with_no_control_spread_at_all_reports_no_floor() -> None:
    comparisons = [compare(cell("a", CONTROL_ARM, [5.0], runs=1), cell("a", "x", [6.0], runs=1))]
    floor = noise_floor(comparisons, {})
    assert math.isnan(floor.aggregate)


def test_the_floor_tracks_the_control_spread_rather_than_a_constant() -> None:
    """#143 cites a 3-4 gap-point floor from an earlier campaign; nothing here
    uses that number. Doubling the control's across-seed spread must double the
    floor, which no hard-coded constant would do."""
    tight = compare(cell("a", CONTROL_ARM, [10.0, 12.0, 14.0]), cell("a", "x", [1.0, 2.0, 3.0]))
    wide = compare(cell("a", CONTROL_ARM, [10.0, 14.0, 18.0]), cell("a", "x", [1.0, 2.0, 3.0]))
    narrow = noise_floor([tight], {"a": 2.0})
    broad = noise_floor([wide], {"a": 4.0})
    assert broad.aggregate == pytest.approx(2.0 * narrow.aggregate)


# --- verdicts ------------------------------------------------------------------


def test_an_effect_inside_the_floor_is_reported_as_such_with_the_floor_quoted(
    tmp_path: Path,
) -> None:
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
        *run_rows("a", "x", [10.5, 12.5, 14.5]),
        *run_rows("b", CONTROL_ARM, [20.0, 22.0, 24.0]),
        *run_rows("b", "x", [20.5, 22.5, 24.5]),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["a", "b"])
    assert summary.mean_delta == pytest.approx(0.5)
    assert "INSIDE THE NOISE" in summary.verdict
    assert f"{summary.floor.aggregate:.2f}" in summary.verdict


def test_an_effect_outside_the_floor_names_its_direction(tmp_path: Path) -> None:
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 10.1, 10.2]),
        *run_rows("a", "x", [40.0, 40.1, 40.2]),
        *run_rows("b", CONTROL_ARM, [20.0, 20.1, 20.2]),
        *run_rows("b", "x", [50.0, 50.1, 50.2]),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["a", "b"])
    assert summary.mean_delta == pytest.approx(30.0)
    assert "WORSE than" in summary.verdict
    assert "outside the measured floor" in summary.verdict


def test_an_arm_that_only_wins_on_feasibility_is_reported_on_the_counts(
    tmp_path: Path,
) -> None:
    rows = [
        *run_rows("a", CONTROL_ARM, [None, None, None]),
        *run_rows("a", "x", [3.0, 4.0, 5.0]),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["a"])
    assert summary.counts[ARM_ONLY_FEASIBLE] == 1
    assert summary.counts[BOTH_FEASIBLE] == 0
    assert summary.mean_delta is None
    assert summary.feasibility_delta == 3
    assert "feasibility counts" in summary.verdict


def test_an_instance_with_rows_on_one_side_only_is_counted_not_dropped(
    tmp_path: Path,
) -> None:
    """An interrupted campaign always ends mid-instance-block, so its last
    instance has control rows and not the arm's. Dropping it silently is the
    "instances scored" line disagreeing with the bucket counts and nothing
    saying why -- the silent drop the acceptance criterion names."""
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
        *run_rows("a", "x", [11.0, 13.0, 15.0]),
        *run_rows("b", CONTROL_ARM, [20.0, 22.0, 24.0]),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["a", "b"])
    assert summary.uncompared == 1
    assert sum(summary.counts.values()) + summary.uncompared == 2
    report = render_report(tmp_path / "r.csv")
    assert "rows on only one side" in report
    assert "incomplete for this arm" in report


def test_the_report_states_the_repair_counts_when_the_lns_arm_runs(tmp_path: Path) -> None:
    """#143 asks for the arm to be "run with its repair counts reported", not
    only for the reading that justifies skipping it. The gate JSON covers the
    skip half; this is the run half."""
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0], repairs=4),
        *run_rows("a", "no-lns", [11.0, 13.0, 15.0], repairs=0),
    ]
    report = render_report(write_results(tmp_path / "r.csv", rows))
    assert "LNS repairs over the roster: control 12, arm 0" in report


def test_a_row_with_no_reading_is_not_counted_as_zero_repairs(tmp_path: Path) -> None:
    """The runner writes NaN where no solve completed. Summing it as 0 would
    make "nothing ran" indistinguishable from "LNS ran and never repaired"."""
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0], repairs=2),
        *run_rows("a", "x", [11.0, 13.0, 15.0], repairs=0),
    ]
    rows[0]["lns_repairs"] = "NaN"
    report = render_report(write_results(tmp_path / "r.csv", rows))
    assert "control 4," in report  # the two readable rows, not three


# --- what is held out ----------------------------------------------------------


def test_the_probe_rows_are_never_averaged_into_the_control(tmp_path: Path) -> None:
    """The probe ran the control configuration hours earlier. Folding it in
    would make the control a two-sitting average, which is the drift the
    protocol exists to keep out of a comparison."""
    rows = [
        *run_rows("a", PROBE_ARM_NAME, [99.0]),
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
        *run_rows("a", "x", [10.0, 12.0, 14.0]),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    assert cells[("a", CONTROL_ARM)].gaps == (10.0, 12.0, 14.0)
    summary = summarize_arm("x", cells, ["a"])
    assert summary.mean_delta == pytest.approx(0.0)
    report = render_report(tmp_path / "r.csv")
    assert PROBE_ARM_NAME not in report.split("--- x ---")[1]
    assert "drift check" in report


def test_the_excluded_instances_are_held_out_of_the_scored_set(tmp_path: Path) -> None:
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
        *run_rows(CLAIM_EXCLUDED[0], CONTROL_ARM, [800.0, 900.0, 1000.0]),
    ]
    loaded = load_rows(write_results(tmp_path / "r.csv", rows))
    assert scored_instances(loaded) == ["a"]


def test_the_excluded_instances_are_reported_rather_than_dropped(tmp_path: Path) -> None:
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
        *run_rows("a", "x", [11.0, 13.0, 15.0]),
        *run_rows(CLAIM_EXCLUDED[0], CONTROL_ARM, [800.0, 900.0, 1000.0]),
        *run_rows(CLAIM_EXCLUDED[0], "x", [700.0, 800.0, 900.0]),
    ]
    report = render_report(write_results(tmp_path / "r.csv", rows))
    assert "excluded from every claim" in report
    assert CLAIM_EXCLUDED[0] in report.split("excluded from every claim")[1]
    # and its 900-point gap did not move the scored aggregate
    cells = build_cells(load_rows(tmp_path / "r.csv"))
    assert summarize_arm("x", cells, ["a"]).mean_delta == pytest.approx(1.0)


def test_the_report_states_the_sign_convention_and_the_gate(tmp_path: Path) -> None:
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0]),
        *run_rows("a", "x", [11.0, 13.0, 15.0]),
    ]
    report = render_report(
        write_results(tmp_path / "r.csv", rows),
        gate={"run_arm": False, "reason": "zero repairs across the roster"},
    )
    assert "POSITIVE IS WORSE" in report
    assert "skipped the arm" in report
    assert "zero repairs across the roster" in report


def test_an_empty_results_file_reports_rather_than_raises(tmp_path: Path) -> None:
    assert "no rows" in render_report(write_results(tmp_path / "r.csv", []))
