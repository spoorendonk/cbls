"""Unit tests for the MINLPLib ablation scoring (issue #143).

The arithmetic is hand-computed in the assertions rather than recomputed from
the module, because the point of these tests is the definitions: which instances
may be averaged, what the noise floor is derived from, and what "inside the
noise" means. A test that recomputed the formula would agree with any formula.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from typing import TYPE_CHECKING

import pytest

from benchmarks.common.provenance import REPO_ROOT
from benchmarks.minlplib.ablation_report import (
    ARM_ONLY_FEASIBLE,
    BOTH_FEASIBLE,
    CONTROL_ARM,
    CONTROL_ONLY_FEASIBLE,
    NEITHER_FEASIBLE,
    NO_COMPARABLE_GAP,
    NO_SEARCH_NOTES,
    PROBE_ARM_NAME,
    Cell,
    build_cells,
    compare,
    control_spreads,
    format_points,
    load_rows,
    min_move_points,
    noise_floor,
    render_report,
    scored_instances,
    summarize_arm,
    t_multiplier,
)
from benchmarks.minlplib.run_ablation import Arm, Run, failed_row
from benchmarks.minlplib.runner import CLAIM_EXCLUDED, COMPLETED_SEARCH_NOTES

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

#: A note `describe_infeasible` could have written, for a fixture row that says
#: `feasible=false`. Its exact text is never asserted on; what matters is that it
#: is on the completed-search allowlist, because a search DID run on such a row.
INFEASIBLE_NOTE = "infeasible(residual=0.5; 1 viol; worst row0 <=)"

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
    "lns_repairs_accepted",
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
        repairs_accepted=(0.0,) * runs,
    )


def _columns_without(dropped: str) -> tuple[str, ...]:
    """`COLUMNS` as an older driver wrote it, before `dropped` was a column."""
    return tuple(column for column in COLUMNS if column != dropped)


def write_results(
    path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str] = COLUMNS
) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            full = dict.fromkeys(columns, "")
            full.update(
                {
                    "wall_seconds": "60",
                    "lns_repairs": "0",
                    "lns_repairs_accepted": "0",
                    "seed": "1",
                }
            )
            full.update({k: str(v) for k, v in row.items() if k in full})
            # The scorer allowlists the notes a COMPLETED search writes (#153),
            # so a fixture row that leaves the cell empty is a row the runner
            # could never have written and is now held out of every count.
            # Filled from the row's own `feasible` cell, after the merge, so the
            # pair stays a combination the runner could actually produce. Keyed
            # on whether the row NAMED a note, not on whether the cell is empty:
            # a row that asks for an empty note is testing exactly that.
            if "note" not in row:
                full["note"] = "feasible" if full["feasible"] == "true" else INFEASIBLE_NOTE
            writer.writerow([full[column] for column in columns])
    return path


def run_rows(
    instance: str,
    arm: str,
    gaps: Sequence[float | None],
    *,
    repairs: int = 0,
    repairs_accepted: int = 0,
    primal_bks: float | None = None,
) -> list[dict[str, object]]:
    """One row per seed; a None gap is an infeasible run.

    `primal_bks` is left empty unless a test needs it: it reads back as NaN,
    which sizes the floor at its scale-free infimum and so leaves every test
    written before the floor became per-instance saying what it said.

    The `note` is the one the runner would have written for the row: a search
    completed either way, and since #153 that is what decides whether the row is
    scored at all.
    """
    return [
        {
            "instance": instance,
            "arm": arm,
            "seed": seed,
            "feasible": "true" if gap is not None else "false",
            "gap_to_bks%": "NaN" if gap is None else gap,
            "note": "feasible" if gap is not None else INFEASIBLE_NOTE,
            "lns_repairs": repairs,
            "lns_repairs_accepted": repairs_accepted,
            **({} if primal_bks is None else {"primal_bks": primal_bks}),
        }
        for seed, gap in enumerate(gaps, start=1)
    ]


# --- buckets: nothing is silently dropped and no NaN reaches an aggregate ------


@pytest.mark.parametrize(
    ("control", "treatment", "bucket", "delta", "feasibility_delta"),
    [
        (([10.0, 12.0, 14.0], 3, None), ([20.0, 22.0, 24.0], 3, None), BOTH_FEASIBLE, 10.0, 0),
        # There is no control gap to subtract. The instance is a categorical win,
        # reported as one -- not averaged in as a NaN and not dropped.
        (([], 3, None), ([7.0, 8.0, 9.0], 3, None), ARM_ONLY_FEASIBLE, None, 3),
        (([7.0, 8.0, 9.0], 3, None), ([], 3, None), CONTROL_ONLY_FEASIBLE, None, -3),
        (([], 3, None), ([], 3, None), NEITHER_FEASIBLE, None, 0),
        # Feasible on both sides but no finite gap -- no BKS, or a non-finite
        # objective. It is not `both-feasible` and must not be averaged.
        (([], 3, 3), ([], 3, 3), NO_COMPARABLE_GAP, None, 0),
        # A partly feasible arm is compared on its feasible runs, and the loss is
        # visible in the counts rather than hidden in the mean.
        (([10.0, 10.0, 10.0], 3, None), ([4.0], 3, 1), BOTH_FEASIBLE, -6.0, -2),
    ],
    ids=["both-feasible", "arm-only", "control-only", "neither", "no-comparable-gap", "partly"],
)
def test_every_instance_lands_in_exactly_one_bucket(
    control: tuple[list[float], int, int | None],
    treatment: tuple[list[float], int, int | None],
    bucket: str,
    delta: float | None,
    feasibility_delta: int,
) -> None:
    (control_gaps, control_runs, control_feasible) = control
    (arm_gaps, arm_runs, arm_feasible) = treatment
    comparison = compare(
        cell("a", CONTROL_ARM, control_gaps, runs=control_runs, feasible=control_feasible),
        cell("a", "x", arm_gaps, runs=arm_runs, feasible=arm_feasible),
    )
    assert comparison.bucket == bucket
    assert comparison.delta == (None if delta is None else pytest.approx(delta))  # positive: worse
    assert comparison.feasibility_delta == feasibility_delta


def test_a_nan_gap_never_reaches_a_cell(tmp_path: Path) -> None:
    results = write_results(tmp_path / "r.csv", run_rows("a", CONTROL_ARM, [10.0, None, 14.0]))
    cells = build_cells(load_rows(results))
    assert cells[("a", CONTROL_ARM)].gaps == (10.0, 14.0)
    assert cells[("a", CONTROL_ARM)].feasible_runs == 2
    assert cells[("a", CONTROL_ARM)].mean_gap == pytest.approx(12.0)


# --- the measured noise floor --------------------------------------------------


def test_the_floor_is_computed_from_the_control_s_own_across_seed_spread() -> None:
    """s = stdev([10, 12, 14]) = 2; scale = sqrt(1/3 + 1/3); floor = t(df=2) * s * scale."""
    control = cell("a", CONTROL_ARM, [10.0, 12.0, 14.0])
    treatment = cell("a", "x", [20.0, 22.0, 24.0])
    spreads = control_spreads({("a", CONTROL_ARM): control}, ["a"])
    assert spreads["a"] == pytest.approx(2.0)
    floor = noise_floor([compare(control, treatment)], spreads)
    expected = t_multiplier(2) * 2.0 * math.sqrt(1 / 3 + 1 / 3)
    assert floor.per_instance["a"] == pytest.approx(expected)
    assert floor.measured == 1
    assert floor.unmeasured == 0


def test_the_band_widens_when_the_control_has_fewer_seeds() -> None:
    """Three seeds is df = 2 and a multiplier of 4.30; two seeds is df = 1 and
    12.71. A flat 2.0 (an earlier cut) prints a band about half its nominal
    width at three seeds and a sixth of it at two."""
    three = noise_floor(
        [compare(cell("a", CONTROL_ARM, [10.0, 12.0, 14.0]), cell("a", "x", [1.0, 2.0, 3.0]))],
        {"a": 2.0},
    )
    two = noise_floor(
        [
            compare(
                cell("a", CONTROL_ARM, [10.0, 14.0], runs=2, feasible=2),
                cell("a", "x", [1.0, 2.0], runs=2, feasible=2),
            )
        ],
        {"a": 2.0},
    )
    assert three.per_instance["a"] < two.per_instance["a"]


def test_an_instance_without_two_control_runs_is_not_scored() -> None:
    """It is counted and listed, never given a borrowed floor.

    Imputing the median absolute spread onto it is unsound on this roster: the
    gaps span six orders of magnitude, so a ~1-point median spread lent to an
    instance whose gap is ~1e6 hands it a floor it clears automatically. That
    manufactures a significant result instead of measuring one -- and the
    instance most likely to have only one feasible control run is exactly the
    enormous one.
    """
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
    assert floor.unmeasured == 1
    assert "b" not in floor.per_instance


def test_a_campaign_with_no_control_spread_at_all_reports_no_floor() -> None:
    comparisons = [compare(cell("a", CONTROL_ARM, [5.0], runs=1), cell("a", "x", [6.0], runs=1))]
    floor = noise_floor(comparisons, {})
    assert not floor.per_instance
    assert math.isnan(floor.median_floor)


def test_the_floor_tracks_the_control_spread_rather_than_a_constant() -> None:
    """#143 cites a 3-4 gap-point floor from an earlier campaign; nothing here
    uses that number. Doubling the control's across-seed spread must double the
    floor, which no hard-coded constant would do."""
    tight = compare(cell("a", CONTROL_ARM, [10.0, 12.0, 14.0]), cell("a", "x", [1.0, 2.0, 3.0]))
    wide = compare(cell("a", CONTROL_ARM, [10.0, 14.0, 18.0]), cell("a", "x", [1.0, 2.0, 3.0]))
    narrow = noise_floor([tight], {"a": 2.0})
    broad = noise_floor([wide], {"a": 4.0})
    assert broad.median_floor == pytest.approx(2.0 * narrow.median_floor)


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
    assert summary.median_delta == pytest.approx(0.5)
    assert "INSIDE THE NOISE" in summary.verdict
    # The floor is quoted, which is the acceptance criterion. It is the typical
    # PER-INSTANCE floor: each instance is judged against its own, because an
    # aggregate gap-point floor is meaningless on a roster spanning six orders
    # of magnitude.
    assert format_points(summary.floor.median_floor, signed=False) in summary.verdict
    assert summary.moved_worse == 0
    assert summary.moved_better == 0


def _moving(worse: int, better: int, still: int) -> list[dict[str, object]]:
    """Instances whose arm moved worse, better, or not at all, far outside the noise."""
    rows: list[dict[str, object]] = []
    for count, (control, arm) in (
        (worse, (10.0, 40.0)),
        (better, (40.0, 10.0)),
        (still, (10.0, 10.0)),
    ):
        for _ in range(count):
            name = f"i{len(rows) // 6}"
            rows += run_rows(name, CONTROL_ARM, [control, control + 0.1, control + 0.2])
            rows += run_rows(name, "x", [arm, arm + 0.1, arm + 0.2])
    return rows


@pytest.mark.parametrize(
    ("rows", "expected", "said", "not_said"),
    [
        # Enough instances moving the same way IS a roster-level direction.
        (
            _moving(5, 0, 0),
            {"moved_worse": 5, "median_delta": 30.0},
            ["WORSE than", "moved outside their own floor"],
            [],
        ),
        # Instances moving in BOTH directions in comparable numbers is not a result:
        # once both directions are present the sign test decides, so a bare
        # majority is reported as mixed rather than as an arm effect.
        (
            _moving(3, 2, 0),
            {"moved_worse": 3, "moved_better": 2},
            ["MIXED, no consistent direction"],
            [],
        ),
        # A couple of instances clearing their own floors is a fact about those
        # instances, not an arm effect. An earlier cut named a direction off a
        # SINGLE mover, beside "sign test p = 1.000" and "median gap delta +0.00".
        (_moving(2, 0, 3), {"moved_worse": 2}, ["ISOLATED MOVERS"], ["WORSE than"]),
        # A control spread of exactly zero is three seeds landing on one value, not
        # a measurement that there is no noise. Scored with a floor of 0.0, ANY
        # nonzero delta clears it -- and the published roster has four instances at
        # gap exactly 0 and around fourteen more within 1e-7.
        (
            [
                *run_rows("flat", CONTROL_ARM, [0.0, 0.0, 0.0]),
                *run_rows("flat", "x", [1e-9, 1e-9, 1e-9]),
                *run_rows("real", CONTROL_ARM, [10.0, 10.1, 10.2]),
                *run_rows("real", "x", [10.0, 10.1, 10.2]),
            ],
            {"moved_worse": 0, "floor.unmeasured": 1},
            ["INSIDE THE NOISE"],
            [],
        ),
    ],
    ids=["direction", "split-decision", "isolated-movers", "control-never-varied"],
)
def test_the_verdict_names_a_direction_only_when_the_roster_has_one(
    tmp_path: Path,
    rows: list[dict[str, object]],
    expected: dict[str, float],
    said: list[str],
    not_said: list[str],
) -> None:
    loaded = load_rows(write_results(tmp_path / "r.csv", rows))
    summary = summarize_arm("x", build_cells(loaded), scored_instances(loaded))
    for path, value in expected.items():
        owner, _, name = path.rpartition(".")
        actual = getattr(summary.floor if owner else summary, name)
        assert actual == pytest.approx(value), path
    for text in said:
        assert text in summary.verdict, text
    for text in not_said:
        assert text not in summary.verdict, text
    if "floor.unmeasured" in expected:
        assert "flat" not in summary.floor.per_instance
    if len(said) == 1 and said[0] == "ISOLATED MOVERS":
        assert summary.sign_p > 0.05


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


@pytest.mark.parametrize(
    ("control", "arm", "unread", "said"),
    [
        # #143 asks for the arm to be "run with its repair counts reported", not
        # only for the reading that justifies skipping it.
        ((4, 1), (0, 0), None, "control 12 attempted, 3 accepted; arm 0 attempted, 0 accepted"),
        # #150: the attempt count alone cannot tell "LNS is working" from "LNS is
        # spending", and the two numbers must not be able to collapse onto each other.
        ((9, 0), (9, 9), None, "control 27 attempted, 0 accepted; arm 27 attempted, 27 accepted"),
        # The runner writes NaN where no solve completed. Summing it as 0 would
        # make "nothing ran" read as "LNS ran and never repaired" -- or, in the
        # accepted column, as "LNS repaired and kept nothing".
        ((2, 0), (0, 0), "lns_repairs", "control 4 attempted, 0 accepted; "),
        ((2, 2), (0, 0), "lns_repairs_accepted", "control 6 attempted, 4 accepted; "),
    ],
    ids=[
        "run-half-of-the-gate",
        "attempted-apart-from-accepted",
        "unread-attempts",
        "unread-accepts",
    ],
)
def test_the_report_states_the_repair_counts_it_read(
    tmp_path: Path,
    control: tuple[int, int],
    arm: tuple[int, int],
    unread: str | None,
    said: str,
) -> None:
    rows = [
        *run_rows(
            "a", CONTROL_ARM, [10.0, 12.0, 14.0], repairs=control[0], repairs_accepted=control[1]
        ),
        *run_rows("a", "no-lns", [11.0, 13.0, 15.0], repairs=arm[0], repairs_accepted=arm[1]),
    ]
    if unread is not None:
        rows[0][unread] = "NaN"
    report = render_report(write_results(tmp_path / "r.csv", rows))
    assert f"LNS repairs over the roster: {said}" in report


def test_a_side_with_no_reading_at_all_is_not_reported_as_zero(tmp_path: Path) -> None:
    """A campaign written before a counter existed has no column for it, so every
    row reads NaN and the sum is over an empty set.

    `math.fsum(())` is 0.0, and 0 is the STRONGEST claim either counter can make
    -- "LNS repaired nothing", "LNS kept nothing". Publishing that out of a file
    that never measured it is the same defect as folding a single NaN row to
    zero, one level up. `--report-only` re-scores exactly such files: #150's
    counter landed after the #143 campaign was already running.
    """
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 12.0, 14.0], repairs=4),
        *run_rows("a", "x", [11.0, 13.0, 15.0], repairs=0),
    ]
    for row in rows:
        del row["lns_repairs_accepted"]
    path = write_results(tmp_path / "r.csv", rows, columns=_columns_without("lns_repairs_accepted"))
    report = render_report(path)
    assert "control 12 attempted, no accepted reading; " in report
    assert "arm 0 attempted, no accepted reading" in report


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


def test_a_crashed_run_is_not_a_lost_feasibility(tmp_path: Path) -> None:
    """A row the DRIVER wrote for a process that exited nonzero carries
    feasible=false like any infeasible run, and nothing else distinguishes them.

    Three segfaults on one instance under one arm would otherwise bucket it
    `control-only-feasible` -- which means "this arm lost feasibility here", a
    claim about the search rather than about the process table.
    """
    rows = [
        *run_rows("a", CONTROL_ARM, [10.0, 10.1, 10.2]),
        *run_rows("a", "x", [10.0, 10.1, 10.2]),
    ]
    crashed = [
        {
            **rows[0],
            "arm": "x",
            "seed": 90 + i,
            "feasible": "false",
            "gap_to_bks%": "NaN",
            "note": "runner-failed-exit-139",
        }
        for i in range(3)
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", [*rows, *crashed])))
    summary = summarize_arm("x", cells, ["a"])

    assert summary.comparisons[0].bucket == "both-feasible"
    assert summary.comparisons[0].treatment.failed_runs == 3
    assert summary.comparisons[0].treatment.runs == 3
    assert summary.feasibility_delta == 0


# --- rows where no search completed (issue #151) --------------------------------


def crash_rows(instance: str, arm: str, seeds: Sequence[int]) -> list[dict[str, object]]:
    """Rows exactly as the DRIVER writes them for a process that exited nonzero.

    Built from `run_ablation.failed_row` rather than hand-rolled, so the note
    prefix the report keys on stays the driver's own.
    """
    args = argparse.Namespace(time_limit=60.0)
    return [
        dict(failed_row(Run(instance, Arm(arm, ()), seed), args, "deadbeef", 139)) for seed in seeds
    ]


def solve_error_rows(instance: str, arm: str, seeds: Sequence[int]) -> list[dict[str, object]]:
    """Rows as the RUNNER writes them when `cbls::solve` throws.

    `minlplib.cpp` catches the exception, bumps `Tally::errored`, writes an
    unsolved row and carries on; since #153 the process then exits 3, and
    `run_ablation._failed_run_row` records this very row rather than downgrading
    it. The row is well-formed and `feasible=false` -- objective, gap and both
    LNS counters are NaN, while `primal_bks`/`dual_bound` carry the published
    bounds and the wall is 0.0, since `write_unsolved_row` knows those without
    having solved. Nothing but the note says no search ran.
    """
    return [
        {
            "instance": instance,
            "arm": arm,
            "seed": seed,
            "feasible": "false",
            "gap_to_bks%": "NaN",
            "lns_repairs": "NaN",
            "lns_repairs_accepted": "NaN",
            "note": "solve-error",
        }
        for seed in seeds
    ]


@pytest.mark.parametrize(
    ("control", "arm", "expected"),
    [
        # `feasible_runs == 0` is true of a cell whose every row crashed, so a bucket
        # decided from it manufactures a verdict out of a process table.
        (
            lambda: run_rows("i", CONTROL_ARM, [7.0, 7.1, 7.2]),
            lambda: crash_rows("i", "x", [1, 2, 3]),
            {
                "counts.control-only-feasible": 0,
                "counts.no-runs-recorded": 1,
                "feasibility_delta": 0,
                "feasibility_delta_balanced": 0,
            },
        ),
        # The mirror case is the worse one: three segfaults on the CONTROL hand the
        # arm `arm-only-feasible`, an arm win produced by a process table.
        (
            lambda: crash_rows("i", CONTROL_ARM, [1, 2, 3]),
            lambda: run_rows("i", "x", [7.0, 7.1, 7.2]),
            {
                "counts.arm-only-feasible": 0,
                "feasibility_delta": 0,
                "feasibility_delta_balanced": 0,
            },
        ),
        # A `solve-error` row is a well-formed row no exclusion caught. Whether a
        # solve throws can depend on the search configuration, i.e. on the ARM, so
        # scoring it reads an exception as an arm losing feasibility.
        (
            lambda: run_rows("i", CONTROL_ARM, [7.0, 7.1, 7.2]),
            lambda: solve_error_rows("i", "x", [1, 2, 3]),
            {"counts.control-only-feasible": 0, "treatment.no_search_runs": 3, "treatment.runs": 0},
        ),
        # The denylist failed open: a seventh runner outcome carried
        # `feasible=false`, matched no held-out prefix, and was counted as this arm
        # losing feasibility -- #151 re-armed. Scoring is now decided by the notes a
        # COMPLETED search writes, so the unfamiliar note falls out of every count.
        (
            lambda: run_rows("i", CONTROL_ARM, [7.0, 7.1, 7.2]),
            lambda: unknown_note_rows("i", "x", [1, 2, 3], "budget-exhausted-before-init"),
            {"treatment.runs": 0, "treatment.no_search_runs": 3, "feasibility_delta_balanced": 0},
        ),
        # `minlplib.cpp` writes `unsupported: <reason>` with commas replaced by `;`,
        # so the cell is matched on its prefix rather than compared whole.
        (
            lambda: run_rows("i", CONTROL_ARM, [7.0, 7.1, 7.2]),
            lambda: unknown_note_rows("i", "x", [1], "unsupported: NL_UNKNOWN_OPCODE 42; at row 3"),
            {"treatment.no_search_notes": ("unsupported",)},
        ),
    ],
    ids=[
        "arm-crashed",
        "control-crashed",
        "arm-solve-error",
        "unrecognised-note",
        "long-unsupported",
    ],
)
def test_a_cell_with_no_completed_run_is_in_no_feasibility_bucket(
    tmp_path: Path,
    control: Callable[[], list[dict[str, object]]],
    arm: Callable[[], list[dict[str, object]]],
    expected: dict[str, object],
) -> None:
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", [*control(), *arm()])))
    summary = summarize_arm("x", cells, ["i"])
    comparison = summary.comparisons[0]

    assert comparison.bucket == "no-runs-recorded"
    for path, value in expected.items():
        owner, _, name = path.partition(".")
        if owner == "counts":
            assert summary.counts[name] == value, path
        elif owner == "treatment":
            assert getattr(comparison.treatment, name) == value, path
        else:
            assert getattr(summary, owner) == value, path


def test_a_solve_error_row_does_not_contaminate_the_balanced_feasibility_delta(
    tmp_path: Path,
) -> None:
    """`feasibility_delta_balanced` is the figure the report presents as the
    result rather than as bookkeeping.

    A `solve-error` row has the same run count as a successful one, so the
    instance stayed "balanced" and contributed a -2 that no line disclosed.
    """
    rows = [
        *run_rows("a", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *run_rows("a", "x", [7.0]),
        *solve_error_rows("a", "x", [2, 3]),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["a"])

    assert summary.feasibility_delta_balanced == 0
    assert summary.feasibility_unbalanced == 1


def test_the_report_discloses_both_kinds_of_held_out_row_and_agrees_with_them(
    tmp_path: Path,
) -> None:
    """No bucket count may contradict the disclosure printed beneath it."""
    rows = [
        *run_rows("crashy", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *crash_rows("crashy", "x", [1, 2, 3]),
        *run_rows("throws", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *solve_error_rows("throws", "x", [1, 2, 3]),
    ]
    report = render_report(write_results(tmp_path / "r.csv", rows))

    assert "control-only-feasible=0" in report
    assert "no-runs-recorded=2" in report
    assert "3 run(s) crashed" in report
    assert "3 run(s) not scored" in report
    assert "2 instance(s) recorded no completed run on at least one side" in report
    assert "crashy, throws" in report


# --- the near-zero floor (issue #151) -------------------------------------------


def near_zero_rows(
    names: Sequence[str], delta: float, *, primal_bks: float | None = None
) -> list[dict[str, object]]:
    """Controls that landed within 1e-7 of their bound on every seed.

    The published roster has four instances at gap exactly 0 and roughly
    fourteen more within 1e-7. Their measured band is ~3.5e-7 gap points.
    """
    rows: list[dict[str, object]] = []
    for name in names:
        control = [0.0, 1e-7, 2e-7]
        rows += run_rows(name, CONTROL_ARM, control, primal_bks=primal_bks)
        rows += run_rows(name, "x", [g + delta for g in control], primal_bks=primal_bks)
    return rows


def test_a_delta_below_the_resolution_of_its_own_gap_is_not_a_move(tmp_path: Path) -> None:
    """A spread of ~1e-7 gap points is a measurement of nothing the gap can
    resolve, and a 1e-5 delta against it is not a result.

    Four such instances moving 1e-5 the same way was enough to print "the arm is
    WORSE than the control" beside "median gap delta +0.00 points", with the
    quoted floor (the roster median, +/-3.51) three million times the movers'
    own. The floor now cannot fall below the resolution at which the campaign
    knows an objective at all.
    """
    rows = near_zero_rows(["p", "q", "r", "s"], 1e-5)
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["p", "q", "r", "s"])

    assert summary.moved_worse == 0
    assert "WORSE" not in summary.verdict
    assert "INSIDE THE NOISE" in summary.verdict
    assert summary.floor.floored == 4


def test_a_named_direction_carries_a_delta_the_table_can_show(tmp_path: Path) -> None:
    """The guard must not silence real movers, and a direction must come with a
    number in the units it is a claim about -- printed at a resolution that
    makes it visible, here and in the per-instance table."""
    rows = near_zero_rows(["p", "q", "r", "s"], 1e-3)
    results = write_results(tmp_path / "r.csv", rows)
    summary = summarize_arm("x", build_cells(load_rows(results)), ["p", "q", "r", "s"])

    assert summary.moved_worse == 4
    assert "the arm is WORSE than the control" in summary.verdict
    assert "over the 4 mover(s) +0.001 points" in summary.verdict
    # ... and the instances that drove it are identifiable in the table below.
    assert "+0.001" in render_report(results).split("--- x ---")[1]


def test_a_named_direction_never_quotes_a_median_delta_of_zero(tmp_path: Path) -> None:
    """Issue #151's criterion, as an invariant rather than as a readability aim:
    "a verdict naming a direction cannot coexist with a median delta of +0.00".

    Naming the denominators was not enough. Six scored instances holding at
    delta 0 and four movers at +0.001 put the roster median at exactly 0.00,
    and the verdict still read "the arm is WORSE than the control ... median gap
    delta over the 10 scored instance(s) +0.00 points" -- the forbidden sentence,
    reached through the fix for the floor rather than through the floor bug.

    The roster median is not suppressed; it moves to the report line that owns
    it, where it sits beside its denominator and no direction is claimed.
    """
    names = ["h1", "h2", "h3", "h4", "h5", "h6"]
    rows: list[dict[str, object]] = []
    for name in names:
        rows += run_rows(name, CONTROL_ARM, [10.0, 10.1, 10.2])
        rows += run_rows(name, "x", [10.0, 10.1, 10.2])
    movers = ["p", "q", "r", "s"]
    rows += near_zero_rows(movers, 1e-3)
    results = write_results(tmp_path / "r.csv", rows)
    summary = summarize_arm("x", build_cells(load_rows(results)), [*names, *movers])

    assert summary.moved_worse == 4
    assert summary.held == 6
    # The roster median really is zero -- this test would be vacuous otherwise.
    assert summary.median_delta == 0.0
    assert "the arm is WORSE than the control" in summary.verdict
    # No delta the verdict calls a median may render as +0.00. Matched as a
    # pattern rather than as `"+0.00" not in verdict`, which "+0.001" satisfies.
    assert not re.search(r"median gap delta[^;]*\+0\.00 points", summary.verdict)
    assert "median gap delta over the 4 mover(s) +0.001 points" in summary.verdict
    assert "the other 6 scored instance(s) held inside their own floor" in summary.verdict
    # Not suppressed: still reported where its denominator is named.
    assert (
        "median per-instance gap delta: +0.00 points over the 10 SCORED instance(s)"
        in render_report(results)
    )


def test_the_median_and_the_mean_each_state_their_own_denominator(tmp_path: Path) -> None:
    """They are statistics over different sets, and were printed as parallel
    lines with neither saying so: two scored instances at delta 0 beside two
    unscored at +900 prints median +0.00 next to mean +450.00."""
    rows = []
    for name in ("s1", "s2"):
        rows += run_rows(name, CONTROL_ARM, [10.0, 10.1, 10.2])
        rows += run_rows(name, "x", [10.0, 10.1, 10.2])
    for name in ("u1", "u2"):  # zero control spread -- comparable, never scored
        rows += run_rows(name, CONTROL_ARM, [5.0, 5.0, 5.0])
        rows += run_rows(name, "x", [905.0, 905.0, 905.0])
    report = render_report(write_results(tmp_path / "r.csv", rows))

    assert "median per-instance gap delta: +0.00 points over the 2 SCORED instance(s)" in report
    assert "mean per-instance gap delta: +450.00 points over the 4 COMPARABLE instance(s)" in report


def test_the_unmeasurable_floor_line_names_both_reasons(tmp_path: Path) -> None:
    """Every control here produced three feasible runs with finite gaps, so
    "no instance has two comparable control runs" is false. The verdict on the
    next line has given the two-part reason since the zero-spread guard went in;
    this string was not updated with it."""
    rows = [
        *run_rows("flat", CONTROL_ARM, [5.0, 5.0, 5.0]),
        *run_rows("flat", "x", [6.0, 6.0, 6.0]),
    ]
    report = render_report(write_results(tmp_path / "r.csv", rows))

    assert "no instance has two comparable control runs" not in report
    assert "returned an identical gap on every seed" in report


@pytest.mark.parametrize(
    ("df", "multiplier"),
    [
        # df = 2 at three seeds, so the two-sided 95% multiplier is 4.30, not 2.0.
        # An earlier cut used 2.0 flat and said it had "no degrees of freedom for"
        # a t-interval, which is wrong in both directions: there are two, and the
        # band it printed was about half its nominal width.
        (2, 4.303),
        (1, 12.71),
        (7, 2.365),
        # Off the tabulated points the NEXT LOWER df's multiplier is used, the
        # wider band. df 11-14 once fell off the table and returned 1.96 --
        # narrower than both the true t (2.20-2.15) and the tabulated df=10 value.
        (11, 2.228),
        (14, 2.228),
        (50, 2.042),
    ],
)
def test_the_student_multiplier_never_narrows_below_the_next_lower_df(
    df: int, multiplier: float
) -> None:
    assert t_multiplier(df) == pytest.approx(multiplier)
    assert t_multiplier(df) >= 1.96


def test_a_row_where_a_search_did_complete_is_never_held_out(tmp_path: Path) -> None:
    """The prefix match must not sweep in a note the runner writes AFTER a search.

    `non-finite`, `VERIFY-FAILED(...)` and `infeasible(...)` are measurements of
    the arm: a search ran to completion and reported something. `minlplib.cpp`
    appends `; <integrality note>` and ` | <analysis note>` to some of them, so
    the match is on the START of the cell -- which is also what makes the long
    `unsupported: <reason>` form still hold out.
    """
    rows = [
        *run_rows("a", CONTROL_ARM, [7.0, 7.1, 7.2]),
        {"instance": "a", "arm": "x", "seed": 1, "feasible": "false", "note": "non-finite"},
        {
            "instance": "a",
            "arm": "x",
            "seed": 2,
            "feasible": "false",
            "note": "infeasible(residual=1e-3; 2 viol; worst row4 <=); integrality-mismatch(nl=3)",
        },
        {
            "instance": "a",
            "arm": "x",
            "seed": 3,
            "feasible": "false",
            "note": "VERIFY-FAILED(residual=2e-05; 1 fractional int; obj drift 0)",
        },
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))
    summary = summarize_arm("x", cells, ["a"])

    assert summary.comparisons[0].treatment.runs == 3
    assert summary.comparisons[0].treatment.no_search_runs == 0
    assert summary.comparisons[0].bucket == CONTROL_ONLY_FEASIBLE
    assert summary.feasibility_delta_balanced == -3


def test_every_preread_note_the_runner_writes_is_held_out() -> None:
    """Since #153 the classification is an ALLOWLIST, so a preread note is held
    out whether or not this list knows it. What this guards now is the other
    half: every preread literal must collapse to a NAMED label under
    `no_search_label`, or the disclosure line degrades into one
    `unrecognised-note(...)` bucket per distinct reason string -- and none of
    them may be allowlisted, which would score it as a measurement.

    `write_preread_row` is the tight half of the contract: it exists only for a
    row where nothing is known about the instance yet, so EVERY literal it is
    called with must be covered. (`write_unsolved_row` is called with measured
    notes too, so it cannot be swept the same way.)
    """
    source = (REPO_ROOT / "benchmarks" / "minlplib" / "minlplib.cpp").read_text()
    # EVERY call site, not every string literal: counting only the literals
    # lets a fifth call site added with a variable note keep the count at four
    # and the test green, which is the vacuity the guard exists to avoid.
    call_sites = re.findall(r"write_preread_row\(csv, args, name, (.+?)\);", source, re.S)
    literals = re.findall(r'write_preread_row\(csv, args, name, "([^"]*)"', source)

    assert len(call_sites) == 4, call_sites  # the regex still finds the call sites
    assert len(literals) == len(call_sites), call_sites  # each note starts with a literal
    for literal in literals:
        assert any(literal.startswith(note) for note in NO_SEARCH_NOTES), literal
        assert not any(literal.startswith(note) for note in COMPLETED_SEARCH_NOTES), literal


# --- the classification is an allowlist, not a denylist (issue #153) ----------


def unknown_note_rows(
    instance: str, arm: str, seeds: Sequence[int], note: str
) -> list[dict[str, object]]:
    """Rows carrying a note neither list has heard of.

    The shape a SEVENTH runner outcome would arrive in: a well-formed row with
    every measured cell NaN and `feasible=false`, and nothing but the note cell
    to say that no search ran.
    """
    return [
        {
            "instance": instance,
            "arm": arm,
            "seed": seed,
            "feasible": "false",
            "gap_to_bks%": "NaN",
            "lns_repairs": "NaN",
            "lns_repairs_accepted": "NaN",
            "note": note,
        }
        for seed in seeds
    ]


def test_an_unrecognised_note_discloses_itself_rather_than_disappearing(
    tmp_path: Path,
) -> None:
    """Failing safe is only half of it: a row held out for a reason the scorer
    cannot name has to say so, or an allowlist that has fallen behind the runner
    silently discards real measurements instead of silently scoring fake ones."""
    rows = [
        *run_rows("a", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *unknown_note_rows("a", "x", [1, 2, 3], "budget-exhausted-before-init"),
    ]
    report = render_report(write_results(tmp_path / "r.csv", rows))

    assert "unrecognised-note(budget-exhausted-before-init)" in report
    assert "3 run(s) not scored (control 0, arm 3;" in report
    assert "does not recognise" in report


def test_an_empty_note_is_not_a_completed_search(tmp_path: Path) -> None:
    """The runner writes a note on every row it writes, so a blank one means the
    cell did not come from the runner -- a truncated line, a schema the driver
    misread. Not something to score."""
    rows = [
        *run_rows("a", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *unknown_note_rows("a", "x", [1, 2, 3], ""),
    ]
    cells = build_cells(load_rows(write_results(tmp_path / "r.csv", rows)))

    assert cells[("a", "x")].runs == 0
    assert cells[("a", "x")].no_search_notes == ("unrecognised-note(<empty>)",)


def _function_body(source: str, signature: str) -> str:
    """The braced body of the one function whose declaration starts `signature`.

    Brace-matched rather than regex-matched so a `{}` inside the body cannot end
    it early. None of these bodies has a brace inside a string literal.
    """
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening : index + 1]
    raise AssertionError(f"unbalanced braces after {signature}")


#: Every place `minlplib.cpp` composes a note for a row where a search RAN, and
#: the pattern that pulls the literal out of it. Enumerated by call site rather
#: than by scanning the file for strings: a note added at a new site is what the
#: allowlist has to keep up with, and a literal count alone would not notice one.
COMPLETED_NOTE_SITES: tuple[tuple[str, str, int], ...] = (
    ("std::string classify_against_bks(", r'return "([^"]+)";', 4),
    ("bool verify_assignment(", r'buf\.size\(\),\s*"([^"]+)"', 1),
    ("std::string describe_infeasible(", r'buf\.size\(\),\s*"([^"]+)"', 2),
    ("void run_instance(", r'note = [^;]*?"([^"]+)"', 2),
)


def test_every_completed_search_note_the_runner_writes_is_allowlisted() -> None:
    """The cost of inverting the polarity, paid here instead of in a campaign.

    An allowlist cannot score a note it has never heard of -- but it also cannot
    score one the runner grew and nobody added, and THAT failure discards real
    measurements. It is the exact mirror of the denylist hole, so it gets the
    same kind of guard: sweep the runner for the literals a completed search can
    put on a row and hold every one of them against the list.
    """
    source = (REPO_ROOT / "benchmarks" / "minlplib" / "minlplib.cpp").read_text()
    for signature, pattern, expected in COMPLETED_NOTE_SITES:
        literals = re.findall(pattern, _function_body(source, signature), re.S)
        # The count is asserted so a refactor that moves a note elsewhere fails
        # here rather than leaving the sweep quietly matching nothing.
        assert len(literals) == expected, (signature, literals)
        for literal in literals:
            assert any(literal.startswith(note) for note in COMPLETED_SEARCH_NOTES), literal
            # ... and the two lists must not overlap, or a measurement would be
            # held out by whichever match was tried first.
            assert not any(literal.startswith(note) for note in NO_SEARCH_NOTES), literal

    # The per-site sweep above is blind to a note composed in a FIFTH function,
    # which is the likeliest way the allowlist falls behind. So also pin the
    # file-wide set of literals a `note` can START with, wherever they are
    # written. `integrality-mismatch(` and `unsupported` are the two that are
    # not sentence-initial notes in their own right: the first is only ever
    # appended after `; `, and the second is a no-search note.
    file_wide = set(re.findall(r'note = [^;]*?"([^"]+)"', source, re.S))
    assert file_wide == {"integrality-mismatch(nl=", "unsupported", "non-finite", "feasible"}, (
        file_wide
    )


# --- the floor is sized in the units of the gap it bounds (issue #151) ---------


def test_the_floor_is_the_runners_own_tie_band_pushed_through_safe_gap() -> None:
    """`safe_gap` is `100*(obj-ref)/|ref|`, so a gap point is a percentage of the
    published bound and the same objective difference is a different number of
    points at every scale on this roster.

    1e-4 points is the runner's tie band only as |ref| grows; at `ex6_2_6`'s
    bound (-2.6e-6) the same band is ~38 POINTS, wider than that instance's
    entire recorded gap. A single constant is five orders of magnitude too
    permissive there.
    """
    assert min_move_points(1e9) == pytest.approx(1e-4, rel=1e-3)
    assert min_move_points(582.236) == pytest.approx(1.00172e-4, rel=1e-3)
    assert min_move_points(-2.60e-6) == pytest.approx(38.46, rel=1e-2)
    # `safe_gap`'s absolute-residual branch: a gap point IS an objective unit.
    assert min_move_points(0.0) == pytest.approx(1e-6)
    # A row with no recorded bound falls back to the scale-free infimum.
    assert min_move_points(math.nan) == pytest.approx(1e-4)


def test_the_floor_tracks_the_published_bound_and_not_a_constant(tmp_path: Path) -> None:
    """The same near-zero control and the same 1e-3 delta, judged against two
    different published bounds, must not get the same verdict."""
    names = ["p", "q", "r", "s"]
    tiny = write_results(tmp_path / "tiny.csv", near_zero_rows(names, 1e-3, primal_bks=-2.60e-6))
    large = write_results(tmp_path / "large.csv", near_zero_rows(names, 1e-3, primal_bks=582.236))

    assert summarize_arm("x", build_cells(load_rows(tiny)), names).moved_worse == 0
    assert summarize_arm("x", build_cells(load_rows(large)), names).moved_worse == 4


def test_a_named_direction_quotes_the_movers_own_floor_not_only_the_rosters(
    tmp_path: Path,
) -> None:
    """#151: "The quoted floor (+/-3.51) is the roster median, while those four
    movers' own floors are ~3.5e-7" -- so a reader could not tell that the
    instances driving the verdict were judged against a wholly different band."""
    rows = near_zero_rows(["p", "q", "r", "s"], 1e-3, primal_bks=1000.0)
    for name in ("h1", "h2", "h3", "h4", "h5"):  # wide floors, and they hold
        rows += run_rows(name, CONTROL_ARM, [10.0, 10.5, 11.0])
        rows += run_rows(name, "x", [10.0, 10.5, 11.0])
    names = ["p", "q", "r", "s", "h1", "h2", "h3", "h4", "h5"]
    summary = summarize_arm(
        "x", build_cells(load_rows(write_results(tmp_path / "r.csv", rows))), names
    )

    assert summary.moved_worse == 4
    assert summary.floor.median_floor == pytest.approx(1.757, rel=1e-2)
    assert "typical per-instance floor +/-1.76 points" in summary.verdict
    assert "against their own median floor +/-0.0001" in summary.verdict


def test_the_held_out_rows_are_reported_by_side(tmp_path: Path) -> None:
    """A total hides the one thing these rows are evidence of: whether a solve
    crashes or throws can depend on the configuration, so which SIDE produced
    them is the arm property being reported."""
    rows = [
        *run_rows("crashy", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *crash_rows("crashy", "x", [1, 2, 3]),
        *run_rows("throws", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *solve_error_rows("throws", "x", [1, 2, 3]),
    ]
    report = render_report(write_results(tmp_path / "r.csv", rows))

    assert "3 run(s) crashed (control 0, arm 3)" in report
    assert "3 run(s) not scored (control 0, arm 3; solve-error)" in report
    # ... and the notes listed are the ones that occurred, not the whole category
    assert "not-found" not in report


def test_the_wall_average_excludes_the_rows_that_recorded_no_search(tmp_path: Path) -> None:
    """`write_preread_row` and the `solve-error` `write_unsolved_row` both write
    a 0.0 wall, so averaging them in drags the headline toward zero while every
    other line insists those rows record nothing."""
    rows = [
        *run_rows("a", CONTROL_ARM, [7.0, 7.1, 7.2]),
        *run_rows("a", "x", [7.0, 7.1, 7.2]),
    ]
    for row in rows:
        row["wall_seconds"] = "60"
    zero_wall = solve_error_rows("a", "x", [4, 5, 6])
    for row in zero_wall:
        row["wall_seconds"] = "0"
    report = render_report(write_results(tmp_path / "r.csv", [*rows, *zero_wall]))

    assert "mean wall per run:    60.0s" in report
    assert "rows recorded:        9 (6 completed a search)" in report
