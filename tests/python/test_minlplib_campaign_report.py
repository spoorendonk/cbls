"""Tests for `benchmarks/minlplib/campaign_report.py` (issue #142).

Two kinds of fixture, and nothing here solves anything:

* The COMMITTED campaign tables under `benchmarks/instances/minlplib/`. The
  issue's acceptance criterion is that the generator reproduces the numbers the
  README states, so `test_the_committed_tables_reproduce_the_readme` pins every
  one of them. When a campaign regenerates the tables (#123) this test goes red
  ON PURPOSE: update the README from the generator's output and these
  expectations together, in the same commit -- a README number nothing checks is
  the drift this module exists to end.
* Small SYNTHETIC tables written to `tmp_path`, one per definition, so each
  aggregate's edge cases are pinned without depending on what the published run
  happened to produce.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib.campaign_report import (
    AGGREGATION_RULE,
    DEFAULT_INST_DIR,
    NOT_RECORDED,
    README_RENDERERS,
    CampaignReport,
    Row,
    TracePoint,
    anytime_scores,
    apply_readme_blocks,
    build_report,
    cbls_ahead_of_scip,
    feasibility_profile,
    free_variable_instances,
    free_variable_split,
    gap_buckets,
    improvement_times,
    improvement_timing,
    legacy_margin_ties,
    load_bounds_index,
    load_results,
    load_scip,
    load_trace,
    main,
    readme_blocks,
    render_markdown,
    single_band_false_ties,
    stale_readme_blocks,
    summarize_results,
    summarize_trace,
    to_json,
)
from benchmarks.minlplib.run_benchmark import print_summary
from benchmarks.minlplib.runner import CLAIM_EXCLUDED

if TYPE_CHECKING:
    from pathlib import Path

ELEC = CLAIM_EXCLUDED[0]

RESULTS_HEADER = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,wall_seconds,"
    "feasible,note,commit_sha,max_violation,n_int_vars"
)
BOUNDS_HEADER = "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks"
SCIP_HEADER = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,wall_seconds,feasible,"
    "note,scip_dual_bound,scip_gap%,status,n_int_vars,read_seconds,solving_seconds,scip_version"
)
SCIP_VERSION = "SCIP 10.0.2 / PySCIPOpt 6.2.1 / 60s / seed 0"


def row(
    instance: str,
    *,
    objective: float = 1.0,
    bks: float = 1.0,
    gap: float = 0.0,
    wall: float = 60.0,
    feasible: bool = True,
    note: str = "matches-bks",
    maximizing: bool = False,
    n_int: int = 0,
    n_disc: int = 0,
    sha: str = "abc1234",
) -> Row:
    return Row(
        instance=instance,
        objective=objective,
        primal_bks=bks,
        gap_pct=gap,
        wall_seconds=wall,
        feasible=feasible,
        note=note,
        commit_sha=sha,
        n_int_vars=n_int,
        maximizing=maximizing,
        n_disc_vars_bks=n_disc,
    )


def infeasible(instance: str) -> Row:
    return row(
        instance,
        objective=math.nan,
        gap=math.nan,
        feasible=False,
        note="infeasible(residual=1) | bug: x",
    )


# --- the committed tables reproduce the README --------------------------------


def test_the_committed_tables_reproduce_the_readme() -> None:
    """Every run-derived number `benchmarks/instances/minlplib/README.md` states.

    Section by section, in README order: these literals pin the GENERATOR, and
    `test_the_committed_readme_blocks_are_the_generators_rendering` pins the
    README to the generator, so the two together pin the README. When this was
    first written three README numbers had drifted from the committed tables and
    the README was corrected, not this test: `ex8_4_5` is 1.20% worse than BKS
    (not 1.38%), `eg_all_s` walks 15930 steps between 15931 incumbents (not
    "15931 steps"), and the "within 1%" sentence missed `prob09`.
    """
    report = build_report(DEFAULT_INST_DIR, budget=60.0, seed=1, machine=None, feas_tol=None)
    res = report.results
    c, v = res.counts, res.verdicts

    # Provenance: what the tables record, and "not recorded" for what they do not.
    assert report.provenance.engine_commit == "21086c2+107"
    assert report.provenance.machine is None
    assert report.provenance.seed == 1
    assert report.provenance.warnings == []
    assert report.provenance.scip_configuration == SCIP_VERSION

    # Results: the tally table.
    assert (c.roster, c.built, c.built_pct, c.mixed_integer) == (50, 50, 100.0, 15)
    assert c.feasible == 46
    assert (v.matches_bks, v.within_tolerance, v.worse, v.better) == (18, 1, 27, 0)
    assert v.denominator == 46
    assert c.infeasible == 4
    assert sorted(c.infeasible_instances) == ["elec25", "elec50", "nvs01", "st_e40"]
    assert (c.coverage_gaps, c.errors) == (0, 0)
    assert (c.integrality_mismatches, c.verification_failures) == (0, 0)
    assert c.documented_failures_feasible == []

    # Results: gap distribution and the zero-BKS paragraph.
    assert res.gap_buckets.counts == [21, 22, 26]
    assert sorted(res.zero_bks_instances) == ["ex14_2_4", "ex14_2_5", "least", "mathopt1", "prob09"]
    assert sorted(res.gap_buckets.excluded_zero_bks) == ["least", "mathopt1", "prob09"]
    assert sorted(res.gap_buckets.retained_zero_bks) == ["ex14_2_4", "ex14_2_5"]
    assert (res.gap_buckets_strict.counts, res.gap_buckets_strict.denominator) == ([19, 20, 24], 41)

    # Results: the two-band worked examples.
    legacy = {e.instance: -e.gap_pct for e in res.legacy_margin_ties}
    assert legacy == pytest.approx({"ex6_2_6": 8.3e-5, "prob06": 3.2e-4}, rel=0.05)
    [false_tie] = res.single_band_false_ties
    assert false_tie.instance == "ex8_4_5"
    assert false_tie.primal_bks == pytest.approx(3.07e-4, rel=0.01)
    assert round(false_tie.gap_pct, 2) == 1.20

    # Results: the anytime score.
    at = report.trace.anytime
    assert at.budget_seconds == 60.0
    assert at.denominator == 48
    assert round(at.mean, 3) == 0.473
    assert round(at.median, 3) == 0.331
    assert round(at.shifted_geometric_mean, 4) == 0.0793

    # Why 60s: the cumulative-feasibility table and the late-feasible instances.
    fp = report.trace.feasibility
    assert fp.checkpoints == [1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0]
    assert fp.counts == [41, 41, 41, 42, 44, 46, 46]
    assert {k: round(t, 1) for k, t in fp.late_feasible.items()} == {
        "chain50": 17.6,
        "ex8_4_5": 24.9,
        "tln2": 26.9,
        "spring": 31.4,
        "minlphi": 36.8,
    }

    # Why 60s: the 46% / 22% split and the eg_all_s bound-tightening walk.
    it = report.trace.improvement
    assert it.denominator == 46
    assert (it.stopped_early, it.still_improving) == (21, 10)
    assert (it.new_best_stopped_early, it.new_best_still_improving) == (18, 12)
    assert it.sub_resolution_new_best_rows == 1266
    assert round(100 * it.stopped_early / it.denominator) == 46
    assert round(100 * it.still_improving / it.denominator) == 22
    assert it.most_steps_instance == "eg_all_s"
    assert it.most_steps_incumbents == 15931
    assert round(it.most_steps_median_ratio, 7) == 0.9989993
    assert it.most_steps_first == pytest.approx(1e9)
    assert round(it.most_steps_last, 2) == 8.46

    # SCIP baseline: the head-to-head table.
    h = report.head_to_head
    assert (h.cbls.feasible, h.cbls.roster, h.scip.feasible, h.scip.roster) == (46, 50, 49, 50)
    assert h.scip.proved_optimal == 34
    assert (h.cbls.hit_limit, h.scip.hit_limit) == (50, 16)
    assert round(h.cbls.total_wall_seconds) == 3001
    assert round(h.scip.total_wall_seconds) == 1011
    assert round(h.scip.median_wall_seconds, 2) == 0.28
    assert h.scip.under_one_second == 31
    assert (h.scip.integrality_mismatches, h.scip.verification_failures) == (0, 0)

    # SCIP baseline: "the failures are almost disjoint".
    disjoint = {d.instance: d for d in h.disjoint_failures}
    assert sorted(disjoint) == ["elec25", "elec50", "nvs01", "st_e36", "st_e40"]
    assert [d.instance for d in h.disjoint_failures if d.cbls_feasible] == ["st_e36"]
    assert disjoint["st_e36"].cbls_objective == -147
    assert round(disjoint["st_e36"].scip_dual_bound, 1) == -304.5
    assert round(disjoint["nvs01"].scip_wall_seconds, 2) == 0.11
    assert round(disjoint["st_e40"].scip_wall_seconds, 2) == 0.22
    assert round(disjoint["elec25"].scip_objective, 3) == 243.859
    assert round(disjoint["elec50"].scip_gap_pct, 1) == 34.8

    # SCIP baseline: both-solved quality buckets and the five rows CBLS leads.
    assert h.quality_denominator == 38
    assert h.cbls_quality == [17, 18, 22]
    assert h.scip_quality == [32, 32, 33]
    ahead = {a.instance: a for a in h.cbls_ahead}
    assert sorted(ahead) == ["eg_all_s", "eq6_1", "ex8_1_5", "ex8_6_1", "maxmin"]
    assert all(a.scip_status == "timelimit" for a in ahead.values())
    assert round(ahead["eg_all_s"].scip_gap_pct) == 2324

    # What #107 accounted for: the AFTER column.
    fv = report.free_variables
    assert fv is not None
    assert (fv.with_free, fv.with_free_eligible, fv.with_free_within_10pct) == (16, 12, 3)
    assert (fv.without_free, fv.without_free_eligible, fv.without_free_within_10pct) == (34, 27, 19)


def test_the_report_states_its_rule_and_its_provenance() -> None:
    report = build_report(DEFAULT_INST_DIR, budget=60.0, seed=None, machine=None, feas_tol=None)
    text = render_markdown(report)
    assert AGGREGATION_RULE in text
    assert "engine commit: 21086c2+107" in text
    assert f"seed: {NOT_RECORDED}" in text
    assert f"machine: {NOT_RECORDED}" in text
    assert "comparison.csv does not record the budget" in text
    for line in report.not_regenerated:
        assert line in text


# --- the aggregation rule -------------------------------------------------------


def test_documented_failures_count_in_the_roster_and_not_in_quality_aggregates() -> None:
    """#142's denominator fix: one rule, applied the same way everywhere."""
    rows = [
        row("a"),
        row("b", objective=1.5, gap=50.0, note="feasible"),
        row(ELEC, objective=1.0, gap=0.0, note="matches-bks"),
    ]
    summary = summarize_results(rows)
    assert summary.counts.roster == 3
    assert summary.counts.feasible == 3
    assert summary.counts.documented_failures_feasible == [ELEC]
    assert summary.verdicts.denominator == 2
    assert summary.verdicts.matches_bks == 1
    assert summary.gap_buckets.denominator == 2
    assert summary.gap_buckets.counts == [1, 1, 1]


def test_an_infeasible_documented_failure_is_a_roster_infeasible() -> None:
    summary = summarize_results([row("a"), infeasible(ELEC)])
    assert summary.counts.infeasible == 1
    assert summary.counts.infeasible_instances == [ELEC]
    assert summary.counts.documented_failures == [ELEC]


def test_coverage_gaps_and_errors_are_not_built_and_not_infeasible() -> None:
    rows = [
        row("a"),
        row("u", feasible=False, objective=math.nan, gap=math.nan, note="unsupported(V segment)"),
        row("e", feasible=False, objective=math.nan, gap=math.nan, note="read-error(bad)"),
    ]
    counts = summarize_results(rows).counts
    assert (counts.roster, counts.built, counts.infeasible) == (3, 1, 0)
    assert (counts.coverage_gaps, counts.errors) == (1, 1)


def test_an_integrality_mismatch_is_counted_from_the_catalogue_column() -> None:
    counts = summarize_results([row("a", n_int=2, n_disc=2), row("b", n_int=1, n_disc=3)]).counts
    assert counts.mixed_integer == 2
    assert counts.integrality_mismatches == 1


def test_mixed_commit_shas_are_all_reported() -> None:
    summary = summarize_results([row("a", sha="aaa"), row("b", sha="bbb")])
    assert summary.commit_shas == ["aaa", "bbb"]


# --- gap buckets ----------------------------------------------------------------


def test_gap_buckets_are_signed_and_handle_a_zero_bks() -> None:
    rows = [
        row("better", gap=-5.0, note="better-than-bks"),
        row("tiny", gap=0.005),
        row("mid", gap=0.5, note="feasible"),
        row("far", gap=50.0, note="feasible"),
        row("zero-exact", objective=0.0, bks=0.0, gap=0.0),
        row("zero-residual", objective=1.0, bks=0.0, gap=1.0, note="feasible"),
    ]
    buckets = gap_buckets(rows)
    assert buckets.denominator == 5
    assert buckets.counts == [3, 4, 4]
    assert buckets.retained_zero_bks == ["zero-exact"]
    assert buckets.excluded_zero_bks == ["zero-residual"]
    strict = gap_buckets(rows, keep_exact_zero=False)
    assert strict.denominator == 4
    assert strict.counts == [2, 3, 3]


def test_the_single_band_example_needs_a_row_between_the_two_bands() -> None:
    # BKS 3e-4: tie band ~1e-6, claim band 1e-5. 3.04e-4 is 4e-6 worse.
    between = row("between", objective=3.04e-4, bks=3e-4, gap=1.33, note="feasible")
    outside = row("outside", objective=3.5e-4, bks=3e-4, gap=16.7, note="feasible")
    summary = summarize_results([between, outside])
    assert [e.instance for e in summary.single_band_false_ties] == ["between"]


# --- the trace ------------------------------------------------------------------


def test_an_improvement_is_a_strict_decrease_of_the_recorded_objective() -> None:
    points = [TracePoint(0.1, 5.0), TracePoint(0.5, 5.0), TracePoint(2.0, 4.0), TracePoint(9, 4.0)]
    assert improvement_times(points) == [0.1, 2.0]


def test_improvement_timing_counts_early_and_late_and_skips_excluded() -> None:
    rows = [row("early"), row("late"), row(ELEC)]
    trace = {
        "early": [TracePoint(0.2, 3.0), TracePoint(50.0, 3.0)],
        "late": [TracePoint(0.2, 3.0), TracePoint(50.0, 2.0)],
        ELEC: [TracePoint(55.0, 1.0)],
    }
    timing = improvement_timing(rows, trace, budget=60.0)
    assert timing.denominator == 2
    assert (timing.stopped_early, timing.still_improving) == (1, 1)
    assert timing.most_steps_instance == "late"
    assert timing.most_steps_median_ratio == pytest.approx(2.0 / 3.0)


def test_the_feasibility_profile_follows_the_budget() -> None:
    rows = [row("a"), row("b"), infeasible("c")]
    trace = {"a": [TracePoint(0.5, 1.0)], "b": [TracePoint(7.0, 1.0)]}
    profile = feasibility_profile(rows, trace, budget=8.0)
    assert profile.checkpoints == [1.0, 5.0, 8.0]
    assert profile.counts == [1, 1, 2]
    assert profile.late_feasible == {"b": 7.0}
    assert profile.roster == 3


# --- anytime scores -------------------------------------------------------------


def test_anytime_reference_is_negated_on_a_maximize_row() -> None:
    """The trace objective is internally minimised: -600 against a max BKS of 1800.

    Unnegated, the reference and the incumbent differ in sign and every point of a
    maximize instance scores the sign-flip penalty 1.0 instead of 2/3.
    """
    rows = [row("m", objective=600.0, bks=1800.0, gap=66.7, maximizing=True, note="feasible")]
    trace = {"m": [TracePoint(0.0, -600.0)]}
    scores = anytime_scores(rows, trace, budget=60.0)
    [m] = scores.per_instance
    assert m.primal_integral == pytest.approx(2.0 / 3.0)
    assert m.final_primal_gap == pytest.approx(2.0 / 3.0)


def test_anytime_scores_integrate_from_the_first_incumbent_and_exclude_failures() -> None:
    rows = [row("a", bks=10.0), infeasible("never"), infeasible(ELEC)]
    trace = {"a": [TracePoint(30.0, 10.0)]}
    scores = anytime_scores(rows, trace, budget=60.0)
    per = {a.instance: a for a in scores.per_instance}
    assert per["a"].primal_integral == pytest.approx(1.0)  # 2.0 for half the budget
    assert per["never"].primal_integral == pytest.approx(2.0)
    assert per[ELEC].excluded
    assert scores.denominator == 2
    assert scores.mean == pytest.approx(1.5)


# --- SCIP head-to-head ------------------------------------------------------------


def _scip_csv(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join([SCIP_HEADER, *lines]) + "\n")


def test_cbls_ahead_needs_more_than_the_cells_rounding(tmp_path: Path) -> None:
    csv_path = tmp_path / "scip.csv"
    _scip_csv(
        csv_path,
        [
            f"round,582.2361405,582.236,582.236,0,0,0.1,true,matches-bks,582.236,0,optimal,0,0,0,{SCIP_VERSION}",
            f"real,10.5,10,10,5,5,60,true,feasible,9,NaN,timelimit,0,0,0,{SCIP_VERSION}",
            f"maxrow,90,100,100,10,10,60,true,feasible,110,NaN,timelimit,0,0,0,{SCIP_VERSION}",
            f"bigcell,1234563.4,1234560,1234560,0,0,60,true,feasible,1e6,NaN,timelimit,0,0,0,{SCIP_VERSION}",
        ],
    )
    scip = load_scip(csv_path)
    rows = [
        # 6-significant-digit cell 582.236 vs SCIP's 582.2361405: rounding, not a win.
        row("round", objective=582.236, bks=582.236, gap=0.0),
        row("real", objective=10.0, bks=10.0, gap=0.0),
        row("maxrow", objective=95.0, bks=100.0, gap=5.0, maximizing=True, note="feasible"),
        # 3.4 below SCIP clears the claim band (~1.2) but not the cell's resolution
        # (~6.2): a seventh significant digit the table never wrote.
        row("bigcell", objective=1234560.0, bks=1234560.0, gap=0.0),
    ]
    ahead = cbls_ahead_of_scip(rows, scip, feas_tol=1e-6)
    assert [a.instance for a in ahead] == ["real", "maxrow"]


# --- inputs ---------------------------------------------------------------------


def _write_campaign(directory: Path) -> None:
    (directory / "bounds.csv").write_text(
        f"{BOUNDS_HEADER}\na,other,1,0,min,1.0,1.0,0\n{ELEC},other,1,0,min,2.0,1.0,0\n"
    )
    (directory / "comparison.csv").write_text(
        f"{RESULTS_HEADER}\n"
        "a,1,1,1,0,0,60,true,matches-bks,abc1234,0,0\n"
        f"{ELEC},NaN,2,1,NaN,NaN,60,false,infeasible(residual=1),abc1234,1,0\n"
    )
    (directory / "anytime_trace.csv").write_text(
        "instance,time_seconds,objective,new_best\na,0.5,1,1\n"
    )
    _scip_csv(
        directory / "scip_baseline.csv",
        [
            f"a,1,1,1,0,0,0.1,true,matches-bks; proved-optimal,1,0,optimal,0,0,0,{SCIP_VERSION}",
            f"{ELEC},2,2,1,0,0,60,true,matches-bks,1,1,timelimit,0,0,0,{SCIP_VERSION}",
        ],
    )


def test_a_results_row_without_a_bound_is_refused(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    bounds = load_bounds_index(tmp_path / "bounds.csv")
    del bounds["a"]
    with pytest.raises(ValueError, match="not in bounds.csv"):
        load_results(tmp_path / "comparison.csv", bounds)


def test_free_variables_come_from_the_nl_bounds_segment(tmp_path: Path) -> None:
    header = "g3 1 1 0\n 2 0 1 0 0\n"
    (tmp_path / "free.nl").write_text(header + "b\n0 0 1\n3\n")
    (tmp_path / "boxed.nl").write_text(header + "b\n0 0 1\n2 0\n")
    assert free_variable_instances(tmp_path, ["free", "boxed"]) == {"free": True, "boxed": False}
    assert free_variable_instances(tmp_path, ["free", "missing"]) is None


def test_the_cli_writes_strict_json_and_markdown(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    json_out, md_out = tmp_path / "summary.json", tmp_path / "report.md"
    code = main(
        [
            "--inst-dir",
            str(tmp_path),
            "--budget",
            "60",
            "--seed",
            "7",
            "--json",
            str(json_out),
            "--markdown",
            str(md_out),
        ]
    )
    assert code == 0
    # Strict: NaN is not JSON, and a consumer must not need Python to read this.
    data = json.loads(json_out.read_text(), parse_constant=lambda c: pytest.fail(c))
    assert data["results"]["counts"]["roster"] == 2
    assert data["results"]["verdicts"]["denominator"] == 1
    assert data["results"]["rule"] == AGGREGATION_RULE
    assert data["provenance"]["seed"] == 7
    assert data["provenance"]["machine"] is None
    assert data["provenance"]["machine_source"] == NOT_RECORDED
    assert data["free_variables"] is None  # no .nl files in this fixture
    assert "## SCIP head-to-head" in md_out.read_text()


def test_the_cli_refuses_a_non_positive_budget(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    assert main(["--inst-dir", str(tmp_path), "--budget", "0"]) == 2


def test_json_has_no_nan(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    report = build_report(tmp_path, budget=60.0, seed=None, machine=None, feas_tol=None)
    assert "NaN" not in to_json(report)


# --- review round: definitions that must not fail open ---------------------------


def verify_failed(instance: str) -> Row:
    return row(
        instance,
        objective=math.nan,
        gap=math.nan,
        feasible=False,
        note="VERIFY-FAILED(residual=1)",
    )


def test_a_verify_failed_rows_rejected_incumbents_count_nowhere() -> None:
    """The runner traces before its re-check, so a VERIFY-FAILED row has incumbents.

    They must not make it feasible in the profile, and the row is withheld from
    the anytime score (as `mipfeas` withholds it) rather than scored from them.
    """
    rows = [row("a"), verify_failed("v")]
    trace = {"a": [TracePoint(0.5, 1.0)], "v": [TracePoint(0.5, 1.0)]}
    summary = summarize_trace(rows, trace, budget=60.0)
    assert summary.feasibility.counts[-1] == 1
    per = {a.instance: a for a in summary.anytime.per_instance}
    assert math.isnan(per["v"].primal_integral)
    assert summary.anytime.unscored == ["v"]
    assert summary.anytime.denominator == 1


def test_a_trace_that_is_not_the_tables_run_is_refused() -> None:
    rows = [row("a", objective=5.0, bks=5.0), infeasible("b")]
    with pytest.raises(ValueError, match="not the table's"):
        summarize_trace(rows, {"a": [TracePoint(0.5, 6.0)]}, budget=60.0)
    with pytest.raises(ValueError, match="absent from the trace"):
        summarize_trace(rows, {}, budget=60.0)
    with pytest.raises(ValueError, match="infeasible in the table but traced"):
        summarize_trace(rows, {"a": [TracePoint(0.5, 5.0)], "b": [TracePoint(1, 1)]}, 60.0)
    with pytest.raises(ValueError, match="not in the results table"):
        summarize_trace(rows, {"a": [TracePoint(0.5, 5.0)], "z": [TracePoint(1, 1)]}, 60.0)


def test_a_maximize_rows_trace_is_matched_after_un_negation() -> None:
    rows = [row("m", objective=600.0, bks=1800.0, gap=66.7, maximizing=True, note="feasible")]
    summarize_trace(rows, {"m": [TracePoint(0.0, -600.0)]}, budget=60.0)


def test_a_non_finite_trace_entry_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "trace.csv"
    path.write_text("instance,time_seconds,objective,new_best\na,0.5,nan,1\n")
    with pytest.raises(ValueError, match="non-finite"):
        load_trace(path)


def test_an_unknown_verdict_on_a_feasible_row_is_unclassified_not_dropped() -> None:
    verdicts = summarize_results([row("a"), row("w", note="weird")]).verdicts
    assert verdicts.denominator == 2
    assert verdicts.unclassified == ["w"]


def test_every_row_is_built_a_coverage_gap_or_an_error() -> None:
    """The allowlist complement: a note nobody has heard of is an error, not nothing."""
    rows = [
        row("a"),
        row("nf", feasible=False, objective=math.nan, gap=math.nan, note="non-finite"),
        row("rf", feasible=False, objective=math.nan, gap=math.nan, note="runner-failed(9)"),
        row("u", feasible=False, objective=math.nan, gap=math.nan, note="unsupported(V)"),
    ]
    c = summarize_results(rows).counts
    assert c.built + c.coverage_gaps + c.errors == c.roster
    assert (c.built, c.coverage_gaps, c.errors, c.non_finite) == (2, 1, 1, 1)
    assert c.error_instances == ["rf"]


def test_still_improving_never_reaches_into_the_early_window() -> None:
    rows = [row("a")]
    trace = {"a": [TracePoint(0.2, 3.0)]}
    timing = improvement_timing(rows, trace, budget=10.0)
    assert (timing.stopped_early, timing.still_improving) == (1, 0)


def test_a_first_incumbent_just_past_the_budget_still_counts_at_the_budget() -> None:
    profile = feasibility_profile([row("a")], {"a": [TracePoint(60.02, 1.0)]}, budget=60.0)
    assert profile.counts[-1] == 1


def test_the_most_stepped_instances_values_are_in_its_own_sense() -> None:
    rows = [row("m", objective=3.0, bks=3.0, maximizing=True)]
    trace = {"m": [TracePoint(0.1, -1.0), TracePoint(0.2, -2.0), TracePoint(0.3, -3.0)]}
    timing = improvement_timing(rows, trace, budget=60.0)
    assert (timing.most_steps_first, timing.most_steps_last) == (1.0, 3.0)


def test_a_budget_the_tables_contradict_is_warned_about(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    report = build_report(tmp_path, budget=30.0, seed=None, machine=None, feas_tol=None)
    warnings = " ".join(report.provenance.warnings)
    assert "median CBLS wall time is 60.00s" in warnings
    assert "SCIP's configuration records 60s" in warnings
    assert "**WARNING:**" in render_markdown(report)
    clean = build_report(tmp_path, budget=60.0, seed=None, machine=None, feas_tol=None)
    assert clean.provenance.warnings == []


@pytest.mark.parametrize("budget", ["nan", "inf", "-1"])
def test_the_cli_refuses_a_non_finite_or_negative_budget(tmp_path: Path, budget: str) -> None:
    _write_campaign(tmp_path)
    assert main(["--inst-dir", str(tmp_path), "--budget", budget]) == 2


def test_the_cli_prints_markdown_without_output_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_campaign(tmp_path)
    assert main(["--inst-dir", str(tmp_path), "--budget", "60"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("# MINLPLib campaign report")
    assert "## Per instance" in out


def test_a_duplicate_instance_row_is_refused(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    path = tmp_path / "comparison.csv"
    path.write_text(path.read_text() + "a,1,1,1,0,0,60,true,matches-bks,abc1234,0,0\n")
    with pytest.raises(ValueError, match="appears twice"):
        load_results(path, load_bounds_index(tmp_path / "bounds.csv"))


def test_free_variable_split_groups_and_filters() -> None:
    rows = [
        row("f1", gap=5.0, note="feasible"),
        row("f2", gap=50.0, note="feasible"),
        row("f3", objective=1.0, bks=0.0, gap=1.0, note="feasible"),  # |BKS| too small
        row("n1", gap=0.0),
        infeasible("n2"),
    ]
    free = {"f1": True, "f2": True, "f3": True, "n1": False, "n2": False}
    split = free_variable_split(rows, free)
    assert (split.with_free, split.with_free_eligible, split.with_free_within_10pct) == (3, 2, 1)
    assert (split.without_free, split.without_free_eligible) == (2, 1)
    assert split.without_free_within_10pct == 1


# --- the README's derived blocks are the generator's ------------------------------

README = DEFAULT_INST_DIR / "README.md"

#: The command `README.md`'s "After the run" step 2 documents. Change both together.
README_BUDGET = 60.0
README_SEED = 1


def _committed_report() -> CampaignReport:
    return build_report(
        DEFAULT_INST_DIR, budget=README_BUDGET, seed=README_SEED, machine=None, feas_tol=None
    )


def test_the_committed_readme_blocks_are_the_generators_rendering() -> None:
    """Every `campaign_report` block in the README, byte for byte (#142).

    A hand edit inside a block, or a regenerated table without a regenerated
    README, names the stale block here. Fix with the README's "After the run"
    step 2 (`campaign_report.py ... --write-readme`), never by editing a block.
    """
    text = README.read_text()
    blocks = readme_blocks(_committed_report())
    assert stale_readme_blocks(text, blocks) == []
    assert apply_readme_blocks(text, blocks) == text


@pytest.mark.parametrize(
    ("block", "old", "new"),
    [
        ("improvement", "46% stop improving", "52% stop improving"),
        (
            "tally",
            "| — better than BKS, but inside the tolerance slack | 1 |",
            "| — better than BKS, but inside the tolerance slack | 2 |",
        ),
        ("two-band", "1.20% worse", "1.38% worse"),
        ("cbls-ahead", "| `eq6_1` | 20.4%", "| `eq6_1` | 25.4%"),
        (
            "free-variables",
            "| ≥1 free variable | 16 | 12 | 3 |",
            "| ≥1 free variable | 16 | 12 | 4 |",
        ),
        ("feasibility", "41 solved instead of 46", "40 solved instead of 46"),
        (
            "head-to-head",
            "| proved optimal | n/a (primal heuristic) | 34 / 50 |",
            "| proved optimal | n/a (primal heuristic) | 35 / 50 |",
        ),
        ("hardware", "proving optimality on 34 instances", "proving optimality on 35 instances"),
    ],
)
def test_editing_a_readme_number_makes_its_block_stale(block: str, old: str, new: str) -> None:
    """The cold review's edits, each of which every earlier test let through."""
    text = README.read_text()
    assert text.count(old) == 1, old
    blocks = readme_blocks(_committed_report())
    assert stale_readme_blocks(text.replace(old, new), blocks) == [block]


def test_a_readme_missing_or_inventing_a_block_is_refused() -> None:
    blocks = {"a": "x"}
    with pytest.raises(ValueError, match="missing \\['a'\\]"):
        apply_readme_blocks("no blocks here\n", blocks)
    invented = (
        "<!-- campaign_report:begin a -->\nx\n<!-- campaign_report:end a -->\n"
        "<!-- campaign_report:begin b -->\ny\n<!-- campaign_report:end b -->\n"
    )
    with pytest.raises(ValueError, match="unknown \\['b'\\]"):
        apply_readme_blocks(invented, blocks)


def test_the_cli_checks_and_writes_a_readme(tmp_path: Path) -> None:
    _write_campaign(tmp_path)
    readme = tmp_path / "README.md"
    readme.write_text(
        "intro\n"
        + "".join(
            f"<!-- campaign_report:begin {name} -->\n<!-- campaign_report:end {name} -->\n"
            for name in README_RENDERERS
        )
    )
    args = ["--inst-dir", str(tmp_path), "--budget", "60"]
    assert main([*args, "--check-readme", str(readme)]) == 1
    assert main([*args, "--write-readme", str(readme)]) == 0
    assert main([*args, "--check-readme", str(readme)]) == 0
    assert readme.read_text().startswith("intro\n")


# --- the aggregation rule, with a FEASIBLE documented failure ---------------------


def _feasible_elec_rows() -> list[Row]:
    """A claim-set row and a feasible documented failure that would move every aggregate."""
    return [
        row("a", objective=10.0, bks=10.0, gap=0.0),
        row(
            ELEC,
            objective=3.0004,
            bks=3.0,
            gap=0.0133,
            note="feasible",
        ),
    ]


def test_a_feasible_documented_failure_counts_in_the_feasibility_profile() -> None:
    """A roster count: elec counts here even though it is out of every quality aggregate."""
    rows = _feasible_elec_rows()
    trace = {"a": [TracePoint(0.5, 10.0)], ELEC: [TracePoint(0.5, 3.0004)]}
    profile = feasibility_profile(rows, trace, budget=60.0)
    assert profile.counts[-1] == 2


def test_a_feasible_documented_failure_is_not_ahead_of_scip(tmp_path: Path) -> None:
    csv_path = tmp_path / "scip.csv"
    _scip_csv(
        csv_path,
        [
            f"a,20,10,10,100,100,60,true,feasible,9,NaN,timelimit,0,0,0,{SCIP_VERSION}",
            f"{ELEC},9,3,3,200,200,60,true,feasible,2,NaN,timelimit,0,0,0,{SCIP_VERSION}",
        ],
    )
    ahead = cbls_ahead_of_scip(_feasible_elec_rows(), load_scip(csv_path), feas_tol=1e-6)
    assert [a.instance for a in ahead] == ["a"]


def test_a_feasible_documented_failure_is_not_eligible_in_the_free_variable_split() -> None:
    split = free_variable_split(_feasible_elec_rows(), {"a": False, ELEC: True})
    assert (split.with_free, split.with_free_eligible, split.with_free_within_10pct) == (1, 0, 0)
    assert split.without_free_eligible == 1


def test_a_feasible_documented_failure_is_not_a_band_example() -> None:
    rows = [
        row(ELEC, objective=1.0, bks=1.000002, gap=-2e-4, note="matches-bks"),
        row(ELEC + "x", objective=1.0, bks=1.000002, gap=-2e-4, note="matches-bks"),
    ]
    assert [e.instance for e in legacy_margin_ties(rows)] == [ELEC + "x"]
    worse = [
        row(ELEC, objective=3.04e-4, bks=3e-4, gap=1.33, note="feasible"),
        row("b", objective=3.04e-4, bks=3e-4, gap=1.33, note="feasible"),
    ]
    assert [e.instance for e in single_band_false_ties(worse, 1e-6)] == ["b"]


# --- anytime details ----------------------------------------------------------------


def test_the_final_primal_gap_is_the_last_incumbents() -> None:
    rows = [row("a", objective=10.0, bks=10.0)]
    trace = {"a": [TracePoint(1.0, 20.0), TracePoint(2.0, 10.0)]}
    [a] = anytime_scores(rows, trace, budget=60.0).per_instance
    assert a.final_primal_gap == 0.0
    # 2 for 1s, 0.5 for 1s, 0 for 58s.
    assert a.primal_integral == pytest.approx((2.0 + 0.5) / 60.0)


def test_each_anytime_row_carries_its_reference_and_source() -> None:
    rows = [
        Row(
            instance="m",
            objective=600.0,
            primal_bks=1800.0,
            gap_pct=66.7,
            wall_seconds=60.0,
            feasible=True,
            note="feasible",
            commit_sha="abc",
            n_int_vars=0,
            maximizing=True,
            n_disc_vars_bks=0,
            catalogue_bks=1800.123456,
        )
    ]
    [m] = anytime_scores(rows, {"m": [TracePoint(0.0, -600.0)]}, budget=60.0).per_instance
    assert m.reference == -1800.123456
    assert "bounds.csv" in m.reference_source
    assert "maximize" in m.reference_source


def test_the_new_best_reading_is_reported_beside_the_printed_one() -> None:
    rows = [row("a")]
    trace = {
        "a": [
            TracePoint(0.2, 3.0, True),
            TracePoint(50.0, 3.0, True),  # flagged, but prints the same objective
        ]
    }
    timing = improvement_timing(rows, trace, budget=60.0)
    assert (timing.stopped_early, timing.still_improving) == (1, 0)
    assert (timing.new_best_stopped_early, timing.new_best_still_improving) == (0, 1)
    assert timing.sub_resolution_new_best_rows == 1


# --- the driver's summary never turns a publish into a traceback ------------------


def test_the_driver_summary_warns_instead_of_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_campaign(tmp_path)
    bounds = tmp_path / "bounds.csv"
    bounds.write_text(f"{BOUNDS_HEADER}\na,other,1,0,min,1.0,1.0,0\n")  # elec missing
    print_summary(tmp_path / "comparison.csv", bounds)
    err = capsys.readouterr().err
    assert "is written, but its summary could not be derived" in err
    assert "not in bounds.csv" in err
