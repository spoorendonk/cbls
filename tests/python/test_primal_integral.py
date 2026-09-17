"""Tests for the MIPfeas Primal Integral scorer."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import TYPE_CHECKING

import pytest

from benchmarks.mipfeas.primal_integral import (
    ENGINES,
    FULL_ROSTER_SIZE,
    NO_SOLUTION_GAP,
    SIGN_FLIP_GAP,
    Scored,
    check_uniform_configuration,
    primal_gap,
    primal_integral,
    score_instance,
    shifted_geometric_mean,
    summarize,
    write_comparison,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


#: The verdict a run since #138 writes for a solution the independent checker
#: accepted. Feasible results in these tests carry it unless they say otherwise.
_PASSING: dict[str, object] = {
    "verdict": "pass",
    "reason": "",
    "marginal": False,
    "tolerances": {"row_absolute": 1e-06},
}


@pytest.mark.parametrize(
    ("incumbent", "reference", "gap"),
    [
        (100.0, 100.0, 0.0),
        (None, 100.0, NO_SOLUTION_GAP),
        (5.0, -3.0, SIGN_FLIP_GAP),
        (-5.0, 3.0, SIGN_FLIP_GAP),
        # Without the near-zero rule this would score 1.0 (a 100% relative gap) for
        # what is numerically the same answer.
        (1e-9, 0.0, 0.0),
        # |200 - 100| / max(200, 100)
        (200.0, 100.0, 0.5),
        (5.0, 0.0, 1.0),
    ],
    ids=[
        "at-the-reference",
        "no-solution",
        "sign-change",
        "sign-change-other-way",
        "both-near-zero",
        "larger-magnitude",
        "zero-reference",
    ],
)
def test_primal_gap(incumbent: float | None, reference: float, gap: float) -> None:
    assert primal_gap(incumbent, reference) == pytest.approx(gap)


@pytest.mark.parametrize(
    ("trace", "reference", "budget", "integral"),
    [
        ([(0.0, 10.0)], 10.0, 60.0, 0.0),
        ([], 10.0, 60.0, NO_SOLUTION_GAP),
        # No incumbent for the first 30s of 60, then the optimum: 2 * (30/60) = 1.0.
        ([(30.0, 10.0)], 10.0, 60.0, 1.0),
        # 2.0 for 10s with no solution, 0.5 for 30s at 200, 0.0 for 20s at 100.
        ([(10.0, 200.0), (40.0, 100.0)], 100.0, 60.0, (2.0 * 10 + 0.5 * 30) / 60),
        # A solution at t=0 that is never improved: constant gap over the budget.
        ([(0.0, 200.0)], 100.0, 60.0, 0.5),
        # A solution logged after the deadline cannot retroactively improve the score.
        ([(90.0, 100.0)], 100.0, 60.0, NO_SOLUTION_GAP),
        ([(5.0, -50.0)], 100.0, 60.0, (2.0 * 5 + SIGN_FLIP_GAP * 55) / 60),
    ],
    ids=[
        "immediate-optimum",
        "no-solution",
        "two-until-the-first-solution",
        "step-function",
        "last-incumbent-held-to-the-budget",
        "entries-past-the-budget-clamped",
        "sign-flip",
    ],
)
def test_primal_integral(
    trace: list[tuple[float, float]], reference: float, budget: float, integral: float
) -> None:
    value = primal_integral(trace, reference=reference, budget=budget)
    assert value == pytest.approx(integral)
    assert 0.0 <= value <= 2.0


@pytest.mark.parametrize(
    ("trace", "equivalent", "budget"),
    [
        ([(40.0, 100.0), (10.0, 200.0)], [(10.0, 200.0), (40.0, 100.0)], 60.0),
        # CP-SAT logs to 0.01s and reports bursts of improvements inside one tick.
        # A plain tuple sort would apply the worst of the burst last and hold it.
        ([(10.0, 200.0), (10.0, 100.0)], [(10.0, 100.0)], 20.0),
    ],
    ids=["out-of-order-trace-sorted", "best-of-a-burst-held"],
)
def test_primal_integral_reads_a_trace_as_the_incumbent_it_describes(
    trace: list[tuple[float, float]], equivalent: list[tuple[float, float]], budget: float
) -> None:
    assert primal_integral(trace, 100.0, budget) == pytest.approx(
        primal_integral(equivalent, 100.0, budget)
    )


def test_primal_integral_rejects_a_nonpositive_budget() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        primal_integral([(1.0, 1.0)], reference=1.0, budget=0.0)


def test_shifted_geometric_mean() -> None:
    # An instance solved immediately (PI = 0) must not collapse the mean to zero.
    assert shifted_geometric_mean([0.0, 1.0]) > 0.0
    assert shifted_geometric_mean([0.5, 0.5, 0.5]) == pytest.approx(0.5)


def _write_result(
    directory: Path,
    engine: str,
    instance: str,
    record: dict[str, object],
    trace: list[tuple[float, float]] | None = None,
    verification: dict[str, object] | None = _PASSING,
) -> None:
    """Write one job's outputs.

    A feasible result gets a passing verdict by default, because that is what a
    run since #138 produces and what the rest of these tests are about. Pass
    `verification=None` for a row nobody checked, or a dict for any other verdict.
    """
    engine_dir = directory / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / f"{instance}.json").write_text(json.dumps(record))
    if trace is not None:
        lines = ["time_seconds,objective"] + [f"{t},{o}" for t, o in trace]
        (engine_dir / f"{instance}.trace.csv").write_text("\n".join(lines) + "\n")
    if verification is not None and record.get("status") == "feasible":
        (engine_dir / f"{instance}.verify.json").write_text(json.dumps(verification))


def _score(
    directory: Path,
    instance: str = "inst",
    *,
    engine: str = "cbls",
    reference: float = 100.0,
    kind: str = "opt",
    budget: float = 60.0,
    require: bool = True,
) -> Scored:
    return score_instance(
        instance, engine, reference, kind, directory, budget, require_verification=require
    )


def test_score_instance_reads_a_result_and_its_trace(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        "cbls",
        "inst",
        {"status": "feasible", "objective": 200.0, "wall_seconds": 60.0, "commit_sha": "abc1234"},
        trace=[(30.0, 200.0)],
    )
    scored = _score(tmp_path)
    assert scored.status == "feasible"
    assert scored.final_gap == pytest.approx(0.5)
    # 2.0 for the first 30s, then 0.5 for the rest.
    assert scored.primal_integral == pytest.approx((2.0 * 30 + 0.5 * 30) / 60)
    assert scored.provenance == "abc1234"


def test_score_instance_without_a_result_is_not_run(tmp_path: Path) -> None:
    scored = _score(tmp_path, "absent")
    assert scored.status == "not_run"
    assert math.isnan(scored.primal_integral)


def test_score_instance_falls_back_to_the_final_objective_without_a_trace(tmp_path: Path) -> None:
    _write_result(
        tmp_path,
        "cpsat",
        "inst",
        {"status": "feasible", "objective": 100.0, "ortools_version": "9.15"},
    )
    scored = _score(tmp_path, engine="cpsat")
    # The solution is credited at the buzzer, so the gap is 2 for the whole budget.
    assert scored.primal_integral == pytest.approx(NO_SOLUTION_GAP)
    assert scored.provenance == "9.15"


def test_score_instance_ignores_a_trace_when_the_run_found_nothing(tmp_path: Path) -> None:
    # A stale trace from an earlier run must not score an infeasible result.
    _write_result(
        tmp_path, "cbls", "inst", {"status": "no_solution", "objective": None}, trace=[(1.0, 100.0)]
    )
    assert _score(tmp_path).primal_integral == NO_SOLUTION_GAP


def _other_budget(directory: Path) -> None:
    # The driver resumes on file existence and defaults to one results directory
    # whatever the budget, so a 60s smoke run and a 600s run land on top of each
    # other. Holding a 60s incumbent over 600s would score better than the run
    # earned -- beside a result that was produced at the budget being scored.
    for name, budget in (("inst", 60.0), ("long", 600.0)):
        record = {"status": "feasible", "objective": 100.0, "budget_seconds": budget}
        _write_result(directory, "cbls", name, record, trace=[(1.0, 100.0)])


def _truncated(directory: Path) -> None:
    (directory / "cbls").mkdir(parents=True)
    (directory / "cbls" / "inst.json").write_text('{"status": "feasi')


def _non_finite_trace(directory: Path) -> None:
    # One NaN would otherwise turn the geometric mean, the arithmetic mean and the
    # median all into NaN, with no warning anywhere.
    record = {"status": "feasible", "objective": 100.0}
    _write_result(directory, "cbls", "inst", record, trace=[(1.0, float("nan"))])


@pytest.mark.parametrize(
    ("setup", "match"),
    [
        (_other_budget, "60.0s budget but is being scored at 600"),
        (_truncated, "not valid JSON"),
        (_non_finite_trace, "non-finite"),
    ],
    ids=["another-budget", "truncated-result", "non-finite-trace"],
)
def test_score_instance_refuses_a_result_it_cannot_score_honestly(
    tmp_path: Path, setup: Callable[[Path], None], match: str
) -> None:
    setup(tmp_path)
    budget = 600.0 if setup is _other_budget else 60.0
    if setup is _other_budget:
        assert _score(tmp_path, "long", budget=budget).status == "feasible"
    with pytest.raises(ValueError, match=match):
        _score(tmp_path, budget=budget)


@pytest.mark.parametrize(
    ("records", "verdicts", "match"),
    [
        # The budget guard catches only the budget. Novelty Jump and the bound clamp
        # are CLI flags measured to move the aggregate, and the driver resumes on
        # file existence alone -- so two invocations into one results directory
        # would average two configurations into a single table.
        ({"compound_moves": True}, {"compound_moves": False}, "span 2 configurations"),
        ({"compound_moves": True}, {"compound_moves": True}, None),
        # The same hazard one layer down: verdicts reached under different
        # thresholds are two different claims and the table states only one.
        (
            {"tolerances": {"row_absolute": 1e-6}},
            {"tolerances": {"row_absolute": 1e-4}},
            "2 tolerance sets",
        ),
    ],
    ids=["two-configurations", "one-configuration", "two-tolerance-sets"],
)
def test_scoring_refuses_a_table_that_mixes_two_claims(
    tmp_path: Path, records: dict[str, object], verdicts: dict[str, object], match: str | None
) -> None:
    for instance, extra in (("a", records), ("b", verdicts)):
        tolerances = "tolerances" in extra
        _write_result(
            tmp_path,
            "cbls",
            instance,
            {"status": "feasible", "objective": 100.0, **({} if tolerances else extra)},
            trace=[(1.0, 100.0)],
            verification={**_PASSING, **extra} if tolerances else _PASSING,
        )
    rows = [_score(tmp_path, name) for name in ("a", "b")]
    if match is None:
        check_uniform_configuration(rows)  # must not raise
    else:
        with pytest.raises(ValueError, match=match):
            check_uniform_configuration(rows)


def test_summarize_excludes_not_run_instances_from_the_aggregates(tmp_path: Path) -> None:
    _write_result(
        tmp_path, "cbls", "solved", {"status": "feasible", "objective": 100.0}, trace=[(0.0, 100.0)]
    )
    summary = summarize([_score(tmp_path, "solved"), _score(tmp_path, "absent")], "cbls")
    assert summary.scored == 1
    assert summary.not_run == 1
    assert summary.feasible == 1
    assert summary.matched_reference == 1
    assert summary.arithmetic_mean == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        # They are what a reader uses to check both engines saw the same program, so
        # a silent None here would publish an unfalsifiable comparison.
        (
            {"n_unbounded_columns": 40, "n_clamped_bounds": 7, "n_bounds_tightened": 33},
            (40, 7, 33),
        ),
        # A result written before #120 carries none of the keys and must still score.
        ({}, (None, None, None)),
    ],
    ids=["carried", "predating-bound-propagation"],
)
def test_scored_carries_the_bound_propagation_columns(
    tmp_path: Path, counts: dict[str, object], expected: tuple[int | None, ...]
) -> None:
    _write_result(tmp_path, "cbls", "inst", {"status": "feasible", "objective": 10.0, **counts})
    scored = _score(tmp_path, reference=10.0)
    assert scored.status == "feasible"
    assert (scored.n_unbounded_columns, scored.n_clamped_bounds, scored.n_bounds_tightened) == (
        expected
    )


# ---------------------------------------------------------------------------
# Verification (#138): a row nobody could check publishes nothing.


def _rejected(reason: str = "row_violation") -> dict[str, object]:
    return {"verdict": "fail", "reason": reason, "marginal": False, "tolerances": {}}


#: `expected` value meaning "a NaN": the derived number was withheld.
NAN = "nan"


@pytest.mark.parametrize(
    ("record", "verification", "kind", "require", "expected", "summary"),
    [
        (
            {"status": "feasible", "objective": 110.0},
            _rejected(),
            "opt",
            True,
            {
                "verification": "fail",
                "verification_reason": "row_violation",
                "withheld": True,
                "objective": None,
                # And no derived score either: not the gap, not the Primal Integral.
                "final_gap": NAN,
                "primal_integral": NAN,
            },
            {"verification_failed": 1},
        ),
        # `below_reference` is a defect flag, not a published number, and a solution
        # the checker rejected is the likeliest place for one. Dropping it with the
        # objective would blind the cheapest gate the benchmark has (232 of the 233
        # references are proven optima) on exactly the rows that need it.
        (
            {"status": "feasible", "objective": 90.0},
            _rejected(),
            "opt",
            True,
            {"withheld": True, "objective": None, "below_reference": True},
            {"below_reference": 1},
        ),
        # primal_gap takes an absolute value, so an objective below a proven optimum
        # scores as an ordinary positive gap. It is a bug signal and must not
        # publish silently ...
        (
            {"status": "feasible", "objective": 90.0},
            _PASSING,
            "opt",
            True,
            {"withheld": False, "below_reference": True},
            {"below_reference": 1},
        ),
        # ... but only `opt` references are proofs. Beating a best-known value is a
        # real result.
        (
            {"status": "feasible", "objective": 90.0},
            _PASSING,
            "best",
            True,
            {"below_reference": False},
            {"below_reference": 0},
        ),
        # Acceptance criterion of #138: every row reported feasible carries an
        # independent verdict. Only a default-on rule can guarantee that.
        (
            {"status": "feasible", "objective": 100.0},
            None,
            "opt",
            True,
            {"verification": "unverified", "withheld": True, "objective": None},
            {"unverified": 1},
        ),
        # The escape hatch for a results directory filled before #138. It relaxes
        # "nobody checked", never "checked and rejected" -- and the row is still
        # counted: the counter once keyed on `withheld`, which this turns off, so
        # the one mode that publishes unchecked numbers reported none of them.
        (
            {"status": "feasible", "objective": 100.0},
            None,
            "opt",
            False,
            {"withheld": False, "objective": 100.0, "primal_integral": 0.0},
            {"unverified": 1},
        ),
        (
            {"status": "feasible", "objective": 100.0},
            _rejected(),
            "opt",
            False,
            {"withheld": True, "objective": None},
            {"verification_failed": 1},
        ),
        # `error` is "could not check", not "checked and fine", so it withholds like
        # a failure -- and is counted apart from one, since it is a harness fault.
        (
            {"status": "feasible", "objective": 100.0},
            {"verdict": "error", "reason": "missing_solution_file", "marginal": False},
            "opt",
            True,
            {"withheld": True, "verification_reason": "missing_solution_file"},
            {"verification_failed": 0, "unverified": 1},
        ),
        # "Just inside the tolerance" is a signal to look, never a reason to withhold.
        (
            {"status": "feasible", "objective": 100.0},
            {"verdict": "pass", "reason": "", "marginal": True, "tolerances": {}},
            "opt",
            True,
            {"withheld": False, "verification_marginal": True, "objective": 100.0},
            {"verification_marginal": 1},
        ),
        # Nothing to verify about a run with no solution, and its objective is
        # already absent -- so the requirement must not make it a second failure
        # mode, nor read as a row nobody verified to a counter grouping on it.
        (
            {"status": "no_solution", "objective": None},
            None,
            "opt",
            True,
            {
                "withheld": False,
                "primal_integral": NO_SOLUTION_GAP,
                "verification": "not_applicable",
            },
            {"unverified": 0},
        ),
        # The search found a point and only the dump failed. Scoring it 2.0 would
        # publish a derived number for a row nothing could check -- and charge a
        # disk error to the search.
        (
            {"status": "solution_write_error", "objective": None, "message": "disk full"},
            None,
            "opt",
            True,
            {"withheld": True, "primal_integral": NAN},
            {"unverified": 1},
        ),
    ],
    ids=[
        "rejected",
        "rejected-below-proven-optimum",
        "below-proven-optimum",
        "below-best-known",
        "no-verdict",
        "allow-unverified",
        "allow-unverified-rejected",
        "checker-error",
        "marginal-pass",
        "found-nothing",
        "solution-write-error",
    ],
)
def test_a_row_publishes_only_what_its_verdict_allows(
    tmp_path: Path,
    record: dict[str, object],
    verification: dict[str, object] | None,
    kind: str,
    require: bool,
    expected: dict[str, object],
    summary: dict[str, int],
) -> None:
    objective = record["objective"]
    trace = [(0.0, objective)] if isinstance(objective, float) else None
    _write_result(tmp_path, "cbls", "inst", record, trace=trace, verification=verification)
    scored = _score(tmp_path, kind=kind, require=require)

    for field, value in expected.items():
        actual = getattr(scored, field)
        if value == NAN:
            assert math.isnan(actual), field
        elif isinstance(value, float):
            assert actual == pytest.approx(value), field
        else:
            assert actual == value, field
    counts = summarize([scored], "cbls")
    for field, count in summary.items():
        assert getattr(counts, field) == count, field


def test_a_rejected_row_is_excluded_from_the_aggregates_not_scored_two(tmp_path: Path) -> None:
    # Scoring it 2.0 would be publishing a derived number of its own -- and a
    # wrong one: the run did find a point, it was rejected.
    for instance, verification in (("good", _PASSING), ("bad", _rejected())):
        record = {"status": "feasible", "objective": 100.0}
        _write_result(
            tmp_path, "cbls", instance, record, trace=[(0.0, 100.0)], verification=verification
        )
    summary = summarize([_score(tmp_path, name) for name in ("good", "bad")], "cbls")

    assert summary.scored == 1
    assert summary.feasible == 1
    assert summary.verification_failed == 1
    assert summary.arithmetic_mean == pytest.approx(0.0)


# --- the table ----------------------------------------------------------------


def _table(
    tmp_path: Path, scored: Scored, summaries: bool = True
) -> tuple[str, list[str], list[str]]:
    """Write `scored` as a comparison table: its text, its header and its one row."""
    out = tmp_path / "comparison.csv"
    write_comparison(
        out, [scored], [summarize([scored], "cbls")] if summaries else [], 60.0, tmp_path / "r.csv"
    )
    text = out.read_text()
    rows = [r for r in csv.reader(text.splitlines()) if r and not r[0].startswith("#")]
    return text, rows[0], rows[1]


def test_a_write_that_dies_leaves_the_previous_table_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`open(path, "w")` truncates before a single row is written.

    CLAUDE.md's rule for anything that writes a published table: a job killed
    mid-write must not replace it with a header and nothing else, at exit 0. The
    C++ runners already write to a temp path and rename; this one did not. The
    kill lands at the last instant -- the new table fully written beside the old
    one -- because a write that goes straight to the published path has already
    destroyed it by then, and an earlier kill cannot tell the two apart.
    """
    _write_result(tmp_path, "cbls", "inst", {"status": "feasible", "objective": 10.0})
    scored = _score(tmp_path, reference=10.0)
    out = tmp_path / "comparison.csv"
    out.write_text("the published table\n")

    def die(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "replace", die)
    with pytest.raises(KeyboardInterrupt):
        write_comparison(out, [scored], [], 60.0, tmp_path / "roster.csv")

    assert out.read_text() == "the published table\n"


@pytest.mark.parametrize(
    ("record", "verification", "require", "cells", "said", "not_said"),
    [
        # write_comparison builds its header and its rows as two parallel lists.
        # #120 inserted two entries into the middle of both, the most
        # misalignment-prone edit that function admits. Tolerances are read back
        # off the verdicts, not restated in the scorer: a table has to quote the
        # thresholds its own rows were judged by. A fully verified table carries no
        # --allow-unverified banner.
        (
            {"status": "feasible", "objective": 10.0, "n_unbounded_columns": 40},
            _PASSING,
            True,
            {
                "n_unbounded_columns": "40",
                "n_bounds_tightened": "",
                "verification": "pass",
                "verification_reason": "",
            },
            "row_absolute=1e-06",
            "--allow-unverified",
        ),
        (
            {"status": "feasible", "objective": 10.0},
            _rejected("integrality_violation"),
            True,
            {
                "objective": "",
                "primal_integral": "nan",
                "verification_reason": "integrality_violation",
            },
            None,
            None,
        ),
        # The "Verified:" note is unconditional, so --allow-unverified would
        # otherwise produce a table asserting the one thing that is not true of it,
        # with the evidence only in a per-row column.
        (
            {"status": "feasible", "objective": 10.0},
            None,
            False,
            {},
            "SCORED WITH --allow-unverified",
            None,
        ),
    ],
    ids=["aligned-verified-row", "withheld-row", "published-unverified-row"],
)
def test_the_table_says_what_its_rows_are(
    tmp_path: Path,
    record: dict[str, object],
    verification: dict[str, object] | None,
    require: bool,
    cells: dict[str, str],
    said: str | None,
    not_said: str | None,
) -> None:
    _write_result(tmp_path, "cbls", "inst", record, trace=[(0.0, 10.0)], verification=verification)
    text, header, row = _table(tmp_path, _score(tmp_path, reference=10.0, require=require))

    assert len(row) == len(header)
    for column, value in cells.items():
        assert row[header.index(column)] == value, column
    assert said is None or said in text
    assert not_said is None or not_said not in text
    if not require:
        assert "1 feasible row(s) are published with no independent" in text


def _full_roster_rows(scored: Scored) -> list[Scored]:
    """One copy of `scored` per instance-solver pair of a full roster.

    `write_comparison` derives the instance count as `len(rows) // len(summaries)`,
    so the roster size is all that distinguishes a full table from a partial one —
    the rows themselves need not be distinct. Both engines are present because the
    divisor is the summary count: a one-engine table would only reach the full
    roster by also being passed no summaries, which is not a shape the scorer ever
    produces.
    """
    return [scored._replace(engine=engine) for engine in ENGINES for _ in range(FULL_ROSTER_SIZE)]


def test_full_roster_table_at_any_budget_is_not_a_wiring_check(tmp_path: Path) -> None:
    # #126: the budget used to be gated against a hardcoded 600s constant, so a
    # full-roster run at any other budget was stamped "not a publishable result".
    # A short budget is a legitimate scoring choice; only a short *roster* is a
    # wiring check.
    _write_result(tmp_path, "cbls", "inst", {"status": "feasible", "objective": 10.0})
    rows = _full_roster_rows(_score(tmp_path, reference=10.0))
    summaries = [summarize(rows, engine) for engine in ENGINES]
    out = tmp_path / "comparison.csv"
    write_comparison(out, rows, summaries, 60.0, tmp_path / "roster.csv")

    text = out.read_text()
    assert "WIRING CHECK" not in text
    assert "INCOMPLETE RUN" not in text
    # The table states the budget it was scored at, rather than being validated
    # against a constant that lives in the scorer. Matched in full: "60" alone is
    # also a substring of "600.0s", so a containment check would go green on a
    # scorer that went back to printing the retired hardcoded budget.
    budget_line = next(line for line in text.splitlines() if line.startswith("# Budget:"))
    assert budget_line.startswith("# Budget:  60.0s per instance-solver pair")
    assert "scored at" in budget_line


def test_partial_roster_table_is_still_banner_stamped(tmp_path: Path) -> None:
    # Fewer than FULL_ROSTER_SIZE instances is a wiring check at any budget.
    _write_result(tmp_path, "cbls", "inst", {"status": "feasible", "objective": 10.0})
    text, _, _ = _table(tmp_path, _score(tmp_path, reference=10.0), summaries=False)

    assert "*** WIRING CHECK, NOT A PUBLISHABLE RESULT ***" in text
    # Pinned as a phrase, not a bare "233": that also appears in objectives, column
    # counts and peak RSS, so a substring search for the number proves nothing.
    assert f"is {FULL_ROSTER_SIZE} instances" in text
    assert "this table used 1" in text
