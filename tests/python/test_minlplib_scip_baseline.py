"""Tests for the MINLPLib SCIP baseline's scoring and merge logic.

The SCIP rows and the CBLS rows are only comparable if both are scored by the
same rules. `safe_gap` and `classify_vs_bks` are hand-ports of the C++ runner's
`safe_gap` and its two-band BKS classification, and a divergence would not
crash — it would quietly publish a gap column meaning two different things in
two different rows.

Most tests here pin the Python side's contract against hand-worked cases, which
does *not* by itself detect drift on the C++ side: editing `minlplib.cpp` leaves
them green. `test_safe_gap_reproduces_the_cpp_runners_published_gap_column` is
the one that closes that loop, by recomputing the port against the gap column the
C++ binary actually wrote into the committed `comparison.csv`. Its resolution is
bounded by that file's six-significant-digit objectives, so it catches a sign or
sense error but not sub-1e-4 relative drift.

No SCIP solve happens here: the solving path needs the optional `benchmarks`
extra, but the scoring and merge paths must stay testable without it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from benchmarks.minlplib.reference_solve import (
    BKS_METHOD,
    CBLS_METHOD,
    SCIP_METHOD,
    Bound,
    ScipResult,
    annotate,
    classify_vs_bks,
    load_bounds,
    maximizing,
    read_scip_csv,
    safe_gap,
    write_merged_csv,
    write_scip_csv,
)

FEAS_TOL = 1e-6

#: The committed roster, used by the C++ cross-check below.
INST_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "minlplib"


def _bound(instance: str = "inst", primal_bks: float = 10.0, dual_bound: float = 5.0) -> Bound:
    return Bound(
        instance=instance,
        structure="bilinear",
        objsense="min",
        primal_bks=primal_bks,
        dual_bound=dual_bound,
        n_disc_vars_bks=0,
    )


# --- safe_gap: sign orientation is the whole point -------------------------


def test_safe_gap_is_positive_when_worse_for_a_minimize_instance() -> None:
    assert safe_gap(11.0, 10.0, maximizing=False) == pytest.approx(10.0)


def test_safe_gap_is_positive_when_worse_for_a_maximize_instance() -> None:
    # Without the sense flip this reads as a 10% improvement rather than a miss.
    assert safe_gap(9.0, 10.0, maximizing=True) == pytest.approx(10.0)


def test_safe_gap_is_negative_when_better_than_the_reference() -> None:
    assert safe_gap(9.0, 10.0, maximizing=False) == pytest.approx(-10.0)
    assert safe_gap(11.0, 10.0, maximizing=True) == pytest.approx(-10.0)


def test_safe_gap_against_a_zero_reference_is_an_absolute_residual() -> None:
    # Documented in the README: five roster instances have |BKS| < 1e-12, and a
    # percentage against zero is meaningless, so the column holds a raw
    # difference. Consumers must not bucket those rows as percentages.
    assert safe_gap(0.25, 0.0, maximizing=False) == pytest.approx(0.25)
    assert safe_gap(0.25, 1e-15, maximizing=False) == pytest.approx(0.25)


def test_safe_gap_is_nan_when_either_side_is_unknown() -> None:
    assert math.isnan(safe_gap(math.nan, 1.0, maximizing=False))
    assert math.isnan(safe_gap(1.0, math.nan, maximizing=False))


# --- classify_vs_bks: two deliberately different bands ---------------------


def test_a_clear_improvement_is_claimed_as_a_win() -> None:
    assert classify_vs_bks(9.0, 10.0, maximizing=False, feas_tol=FEAS_TOL) == "better-than-bks"


def test_a_clear_miss_is_just_feasible() -> None:
    assert classify_vs_bks(11.0, 10.0, maximizing=False, feas_tol=FEAS_TOL) == "feasible"


def test_an_objective_inside_the_tie_band_matches_bks() -> None:
    # tie_band = 1e-6 * (|BKS| + 1) = 1.1e-5 here.
    assert classify_vs_bks(10.000001, 10.0, maximizing=False, feas_tol=FEAS_TOL) == "matches-bks"


def test_an_improvement_below_the_claim_threshold_is_neither_win_nor_loss() -> None:
    # Between tie_band (1.1e-5) and win_slack (max(1.1e-5, 1e-5) = 1.1e-5)... so
    # widen the gap between the bands with a larger feas_tol, which is exactly
    # the regime the third label exists for: the feasibility slack alone could
    # have bought this much objective, so it is not a win — but calling an
    # objective that is *better* than BKS "worse than BKS" would be false.
    assert (
        classify_vs_bks(9.999, 10.0, maximizing=False, feas_tol=1e-3) == "within-tolerance-of-bks"
    )


def test_the_claim_threshold_has_an_absolute_floor_at_ten_feas_tol() -> None:
    # A tiny BKS makes the relative band vanish; without the 10*feas_tol floor a
    # solution that merely exploits the feasibility slack would be published as
    # beating a published bound.
    assert classify_vs_bks(0.0, 3.07e-4, maximizing=False, feas_tol=1e-3) == (
        "within-tolerance-of-bks"
    )
    assert classify_vs_bks(0.0, 3.07e-4, maximizing=False, feas_tol=1e-12) == "better-than-bks"


def test_a_tiny_bks_does_not_let_the_absolute_floor_swallow_a_real_miss() -> None:
    # The regression the two-band rule exists for: ex8_4_5 (BKS 3.07e-4) was
    # published as "matches-bks" while being 1.38% worse, because one band with
    # a 1e-5 absolute floor dwarfed an objective that small.
    assert classify_vs_bks(3.115e-4, 3.07e-4, maximizing=False, feas_tol=FEAS_TOL) == "feasible"


def test_classification_follows_the_objective_sense() -> None:
    assert classify_vs_bks(11.0, 10.0, maximizing=True, feas_tol=FEAS_TOL) == "better-than-bks"
    assert classify_vs_bks(9.0, 10.0, maximizing=True, feas_tol=FEAS_TOL) == "feasible"


def test_an_unknown_bks_cannot_be_matched_or_beaten() -> None:
    assert classify_vs_bks(1.0, math.nan, maximizing=False, feas_tol=FEAS_TOL) == "feasible"


# --- Bound parsing ---------------------------------------------------------


def test_load_bounds_preserves_file_order_as_the_roster_of_record(tmp_path: Path) -> None:
    path = tmp_path / "bounds.csv"
    path.write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks\n"
        "zeta,bilinear,3,2,max,1.5,2.5,4\n"
        "alpha,other,1,0,min,-7.0,-7.1,\n"
    )
    bounds = load_bounds(path)
    assert [b.instance for b in bounds] == ["zeta", "alpha"]
    assert bounds[0].maximizing and not bounds[1].maximizing
    assert bounds[0].n_disc_vars_bks == 4
    # An absent count must read as "unknown" (-1), not as "zero integer
    # variables" — the latter would fabricate an integrality mismatch.
    assert bounds[1].n_disc_vars_bks == -1


def test_load_bounds_reads_a_missing_bound_as_nan(tmp_path: Path) -> None:
    path = tmp_path / "bounds.csv"
    path.write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks\n"
        "alpha,other,1,0,min,-7.0,,0\n"
    )
    assert math.isnan(load_bounds(path)[0].dual_bound)


# --- merged CSV: the labelling the issue's acceptance criteria asks for ----


def _merged_rows(tmp_path: Path) -> list[dict[str, str]]:
    bounds = [_bound(instance="alpha"), _bound(instance="beta")]
    cbls = {
        "alpha": {
            "instance": "alpha",
            "objective": "10.5",
            "feasible": "true",
            "wall_seconds": "60.0",
            "gap_to_bks%": "5.0",
            "commit_sha": "abc1234",
            "note": "feasible",
        }
    }
    scip = {
        "alpha": ScipResult(
            instance="alpha",
            status="timelimit",
            objective=10.0,
            dual_bound=9.0,
            feasible=True,
            wall_seconds=60.0,
            notes=["matches-bks"],
        )
    }
    path = tmp_path / "comparison_all.csv"
    write_merged_csv(path, bounds, cbls, scip, "SCIP 10.0 / PySCIPOpt 6.2.1")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_merged_row_is_labelled_with_its_method(tmp_path: Path) -> None:
    rows = _merged_rows(tmp_path)
    assert {r["method"] for r in rows} == {BKS_METHOD, CBLS_METHOD, SCIP_METHOD}
    assert all(r["method"] and r["version"] for r in rows)


def test_merged_rows_are_grouped_per_instance_in_roster_order(tmp_path: Path) -> None:
    rows = _merged_rows(tmp_path)
    assert [(r["instance"], r["method"]) for r in rows] == [
        ("alpha", BKS_METHOD),
        ("alpha", CBLS_METHOD),
        ("alpha", SCIP_METHOD),
        ("beta", BKS_METHOD),
    ]


def test_an_instance_only_one_method_ran_still_gets_its_published_row(tmp_path: Path) -> None:
    # `beta` has no CBLS and no SCIP result. Dropping it would shrink the roster
    # silently; the published bound must still appear.
    rows = _merged_rows(tmp_path)
    beta = [r for r in rows if r["instance"] == "beta"]
    assert [r["method"] for r in beta] == [BKS_METHOD]


def test_only_scip_and_the_catalogue_claim_a_dual_bound(tmp_path: Path) -> None:
    # CBLS is a primal heuristic and proves none. Repeating the published dual on
    # its row would read as if all three methods had proved the same thing.
    by_method = {r["method"]: r for r in _merged_rows(tmp_path) if r["instance"] == "alpha"}
    assert by_method[CBLS_METHOD]["dual_bound"] == "NaN"
    assert float(by_method[SCIP_METHOD]["dual_bound"]) == 9.0
    assert float(by_method[BKS_METHOD]["dual_bound"]) == 5.0


def test_the_cbls_row_records_the_commit_it_came_from(tmp_path: Path) -> None:
    by_method = {r["method"]: r for r in _merged_rows(tmp_path) if r["instance"] == "alpha"}
    assert by_method[CBLS_METHOD]["version"] == "cbls@abc1234"
    assert "SCIP 10.0" in by_method[SCIP_METHOD]["version"]


# --- scip_baseline.csv round-trip -----------------------------------------


def test_scip_csv_round_trips_through_read_scip_csv(tmp_path: Path) -> None:
    # `--merge-only` rebuilds the comparison from this file rather than
    # re-running a 50-minute solve, so the round-trip has to hold.
    bound = _bound(instance="alpha")
    result = ScipResult(
        instance="alpha",
        status="timelimit",
        objective=10.0,
        dual_bound=9.0,
        scip_gap=0.1,
        feasible=True,
        wall_seconds=60.0,
        notes=["matches-bks"],
    )
    path = tmp_path / "scip_baseline.csv"
    write_scip_csv(path, [(bound, result)], "SCIP 10.0 / PySCIPOpt 6.2.1")
    back = read_scip_csv(path)["alpha"]
    assert back.feasible
    assert back.objective == pytest.approx(10.0)
    assert back.dual_bound == pytest.approx(9.0)
    assert back.wall_seconds == pytest.approx(60.0)
    assert back.status == "timelimit"
    assert back.note == "matches-bks"


def test_an_infeasible_scip_row_publishes_no_objective_or_gap(tmp_path: Path) -> None:
    # A row we do not stand behind must not carry the numbers it was rejected
    # for; the C++ runner has the same rule.
    bound = _bound(instance="alpha")
    result = ScipResult(instance="alpha", status="timelimit", notes=["no-solution(timelimit)"])
    path = tmp_path / "scip_baseline.csv"
    write_scip_csv(path, [(bound, result)], "v")
    with path.open(newline="") as fh:
        row = next(iter(csv.DictReader(fh)))
    assert row["feasible"] == "false"
    assert row["objective"] == "NaN"
    assert row["gap_to_bks%"] == "NaN"
    assert row["gap_to_dual%"] == "NaN"


def test_a_rejected_objective_is_withheld_from_both_csvs(tmp_path: Path) -> None:
    # An objective present on a row flagged infeasible is the dangerous case: an
    # analysis that filters on the objective column rather than on `feasible`
    # would silently pick it up.
    bound = _bound(instance="alpha")
    result = ScipResult(
        instance="alpha",
        status="timelimit",
        objective=-1.5,  # SCIP had a value; the re-check rejected the solution
        feasible=False,
        notes=["CHECK-FAILED(SCIP could not re-validate its own solution)"],
    )
    scip_path = tmp_path / "scip_baseline.csv"
    write_scip_csv(scip_path, [(bound, result)], "v")
    merged_path = tmp_path / "comparison_all.csv"
    write_merged_csv(merged_path, [bound], {}, {"alpha": result}, "v")

    with scip_path.open(newline="") as fh:
        assert next(iter(csv.DictReader(fh)))["objective"] == "NaN"
    with merged_path.open(newline="") as fh:
        scip_row = next(r for r in csv.DictReader(fh) if r["method"] == SCIP_METHOD)
    assert scip_row["objective"] == "NaN"
    assert scip_row["feasible"] == "false"


def test_a_note_containing_commas_survives_the_csv_round_trip_intact(tmp_path: Path) -> None:
    # The note is the only free-text column, so it is the one that can break the
    # columns. Asserting only that the columns survive would be vacuous —
    # csv.writer quotes the field either way — so pin the text itself, which
    # `--merge-only` reads back and republishes. A reader exception carrying a
    # comma must not come back split at that comma: "; " is the note separator.
    bound = _bound(instance="alpha")
    result = ScipResult(
        instance="alpha", status="timelimit", notes=["read-error(ValueError: bad token, line 3)"]
    )
    path = tmp_path / "scip_baseline.csv"
    write_scip_csv(path, [(bound, result)], "v")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["status"] == "timelimit"
    assert rows[0]["note"] == "read-error(ValueError: bad token, line 3)"
    assert read_scip_csv(path)["alpha"].note == rows[0]["note"]


def test_merge_only_does_not_promote_an_infeasible_row_to_feasible(tmp_path: Path) -> None:
    # The dangerous round-trip direction: a row written `feasible=false` must not
    # come back feasible, or the merged CSV publishes a solution SCIP never found.
    bound = _bound(instance="alpha")
    result = ScipResult(instance="alpha", status="timelimit", notes=["no-solution(timelimit)"])
    path = tmp_path / "scip_baseline.csv"
    write_scip_csv(path, [(bound, result)], "v")
    assert not read_scip_csv(path)["alpha"].feasible


def test_a_results_file_spanning_two_runs_keeps_each_rows_own_provenance(tmp_path: Path) -> None:
    # A crashed run resumed after a SCIP upgrade is exactly the case where
    # stamping one version on every row asserts a provenance most rows lack.
    bounds = [_bound(instance="alpha"), _bound(instance="beta")]
    old = ScipResult(instance="alpha", status="optimal", objective=10.0, feasible=True)
    old.version = "SCIP 9.0.0 / PySCIPOpt 5.0.0 / 60s / seed 0"
    new = ScipResult(instance="beta", status="optimal", objective=10.0, feasible=True)
    path = tmp_path / "comparison_all.csv"
    write_merged_csv(path, bounds, {}, {"alpha": old, "beta": new}, "SCIP 10.0.2 / run-wide")
    with path.open(newline="") as fh:
        scip = {
            r["instance"]: r["version"] for r in csv.DictReader(fh) if r["method"] == SCIP_METHOD
        }
    assert scip["alpha"] == "SCIP 9.0.0 / PySCIPOpt 5.0.0 / 60s / seed 0"
    assert scip["beta"] == "SCIP 10.0.2 / run-wide"


def test_read_scip_csv_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_scip_csv(tmp_path / "nope.csv") == {}


# --- annotate: the cross-checks that exist to fail loudly ------------------


def test_annotate_labels_the_row_then_records_the_proof() -> None:
    result = ScipResult(instance="alpha", status="optimal", objective=10.0, feasible=True)
    annotate(result, _bound(instance="alpha", primal_bks=10.0))
    assert result.notes == ["matches-bks", "proved-optimal"]


def test_annotate_reports_an_integrality_mismatch_against_the_catalogue() -> None:
    result = ScipResult(instance="alpha", status="timelimit", n_int_vars=3)
    bound = _bound(instance="alpha")
    bound.n_disc_vars_bks = 4
    annotate(result, bound)
    assert any("integrality mismatch" in n for n in result.notes)


def test_annotate_invents_no_mismatch_when_the_catalogue_is_silent() -> None:
    # -1 means "unknown", not "zero integer variables".
    result = ScipResult(instance="alpha", status="timelimit", n_int_vars=3)
    bound = _bound(instance="alpha")
    bound.n_disc_vars_bks = -1
    annotate(result, bound)
    assert not any("integrality" in n for n in result.notes)


def test_annotate_flags_a_sense_disagreement_between_instance_and_catalogue() -> None:
    # A one-word error in bounds.csv inverts every gap and label on the row, and
    # the C++ runner takes the sense from the model instead of the catalogue.
    result = ScipResult(instance="alpha", status="optimal", objsense="minimize")
    bound = _bound(instance="alpha")
    bound.objsense = "max"
    annotate(result, bound)
    assert any("objsense mismatch" in n for n in result.notes)


def test_annotate_is_quiet_when_the_senses_agree() -> None:
    result = ScipResult(instance="alpha", status="optimal", objsense="minimize")
    annotate(result, _bound(instance="alpha"))
    assert not any("objsense" in n for n in result.notes)


def test_the_instance_sense_outranks_the_catalogue_when_they_disagree() -> None:
    # The catalogue is derived metadata from a drifting library; the instance is
    # ground truth, and the C++ runner scores from the model too. A stale
    # `bounds.csv` must not invert the gap before the mismatch note can report it.
    bound = _bound(instance="alpha", primal_bks=10.0)
    bound.objsense = "max"
    result = ScipResult(instance="alpha", objsense="minimize", objective=11.0, feasible=True)
    assert not maximizing(bound, result)
    annotate(result, bound)
    # Minimizing, 11 against a BKS of 10, is a miss. Scored as the catalogue's
    # "max" it would read as a win.
    assert result.notes[0] == "feasible"
    assert any("objsense mismatch" in n for n in result.notes)


def test_the_catalogue_sense_is_used_when_the_instance_was_never_read() -> None:
    # `not-found` rows and `read_scip_csv` reloads carry no instance sense.
    assert maximizing(_bound(), ScipResult(instance="alpha")) is False
    maxi = _bound()
    maxi.objsense = "max"
    assert maximizing(maxi, ScipResult(instance="alpha")) is True


def test_annotate_flags_a_dual_bound_that_crosses_the_published_primal() -> None:
    # For a minimize instance a dual above the published primal means one of the
    # two numbers is wrong. The direction of this comparison is the whole check.
    crossed = ScipResult(instance="alpha", status="timelimit", dual_bound=12.0)
    annotate(crossed, _bound(instance="alpha", primal_bks=10.0))
    assert any("crosses the published primal" in n for n in crossed.notes)

    ok = ScipResult(instance="alpha", status="timelimit", dual_bound=9.0)
    annotate(ok, _bound(instance="alpha", primal_bks=10.0))
    assert not any("crosses" in n for n in ok.notes)


def test_the_crossing_check_flips_with_the_objective_sense() -> None:
    bound = _bound(instance="alpha", primal_bks=10.0)
    bound.objsense = "max"
    below = ScipResult(instance="alpha", status="timelimit", dual_bound=9.0)
    annotate(below, bound)
    assert any("crosses the published primal" in n for n in below.notes)

    above = ScipResult(instance="alpha", status="timelimit", dual_bound=12.0)
    annotate(above, bound)
    assert not any("crosses" in n for n in above.notes)


def test_an_unproved_dual_bound_raises_no_crossing_alarm() -> None:
    # SCIP reports "no bound proved" as its 1e20 infinity sentinel, which is a
    # finite float. `_finite_or_nan` folds it to NaN at capture; if that ever
    # regressed, every unproved row would fabricate "one of the two published
    # numbers is wrong".
    result = ScipResult(instance="alpha", status="timelimit", dual_bound=math.nan)
    annotate(result, _bound(instance="alpha", primal_bks=10.0))
    assert not any("crosses" in n for n in result.notes)


# --- the one test that closes the C++/Python loop ---------------------------


def test_safe_gap_reproduces_the_cpp_runners_published_gap_column() -> None:
    """Recompute the port against the gap column `cbls_minlplib` actually wrote.

    Every other test here compares the Python port to hand-worked expectations,
    so a change to `minlplib.cpp` leaves them green. This one reads the committed
    `comparison.csv` — C++ output — and re-derives its `gap_to_bks%` from the
    objective and the catalogue bound. Resolution is set by that file's six
    significant digits, so it pins the sign convention, the sense flip and the
    zero-reference fallback, not sub-1e-4 relative drift.
    """
    comparison = INST_DIR / "comparison.csv"
    bounds = {b.instance: b for b in load_bounds(INST_DIR / "bounds.csv")}
    with comparison.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["feasible"] == "true"]
    assert rows, "no feasible CBLS rows to cross-check against"

    checked = 0
    for row in rows:
        bound = bounds[row["instance"]]
        published = float(row["gap_to_bks%"])
        # Below this the 6-digit objective has destroyed the information the gap
        # was computed from, so a disagreement says nothing about the port.
        if abs(published) <= 0.01:
            continue
        ours = safe_gap(float(row["objective"]), bound.primal_bks, bound.maximizing)
        assert ours == pytest.approx(published, rel=1e-3), row["instance"]
        checked += 1
    assert checked >= 20, f"only {checked} rows had enough resolution to cross-check"
