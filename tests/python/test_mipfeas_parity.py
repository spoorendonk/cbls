"""Tests for the MIPfeas parity report: feasibility sets, defects, cross-checks.

The benchmark is admitted as a head-to-head against the reference implementation
of the same jump-based algorithm, and the run it now carries is a correctness
sweep. These tests cover what that sweep has to report (issue #139) and, in
`test_regenerating_the_report_reproduces_the_committed_numbers`, pin the whole of
it against a frozen results directory.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.mipfeas.primal_integral import (
    ANYTIME_LABEL,
    ENGINES,
    PARITY_EXCLUDED,
    PARITY_FEASIBLE,
    SHAPE_RULE,
    collect_defects,
    compare_feasibility,
    cross_check_shapes,
    cross_checked_instances,
    headline_lines,
    job_failures,
    parity_verdict,
    read_roster,
    read_run_record,
    render_report,
    run_record_section,
    score_instance,
    shape_notes_by_instance,
    summarize,
    timing_summary,
    trace_health,
)

if TYPE_CHECKING:
    from benchmarks.mipfeas.primal_integral import Scored

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "benchmarks" / "mipfeas" / "testdata"
FIXTURE_BUDGET = 2.0

_PASSING: dict[str, object] = {
    "verdict": "pass",
    "reason": "",
    "marginal": False,
    "tolerances": {"row_absolute": 1e-06},
    "n_columns": 10,
    "n_rows": 5,
}


def _write(
    directory: Path,
    engine: str,
    instance: str,
    record: dict[str, object],
    trace: list[tuple[float, float]] | None = None,
    verification: dict[str, object] | None = _PASSING,
) -> None:
    engine_dir = directory / engine
    engine_dir.mkdir(parents=True, exist_ok=True)
    (engine_dir / f"{instance}.json").write_text(json.dumps(record))
    if trace is not None:
        lines = ["time_seconds,objective"] + [f"{t},{o}" for t, o in trace]
        (engine_dir / f"{instance}.trace.csv").write_text("\n".join(lines) + "\n")
    if verification is not None and record.get("status") == "feasible":
        (engine_dir / f"{instance}.verify.json").write_text(json.dumps(verification))


def _score(directory: Path, instances: list[tuple[str, float]]) -> list[Scored]:
    return [
        score_instance(name, engine, reference, "opt", directory, budget=60.0)
        for name, reference in instances
        for engine in ENGINES
    ]


# --- feasibility parity ------------------------------------------------------


def test_parity_names_both_asymmetric_difference_sets(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "mine", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "mine", {"status": "no_solution", "objective": None}, [])
    _write(tmp_path, "cbls", "theirs", {"status": "no_solution", "objective": None}, [])
    _write(tmp_path, "cpsat", "theirs", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    parity = compare_feasibility(_score(tmp_path, [("mine", 1.0), ("theirs", 1.0)]))
    assert parity.only["cbls"] == ["mine"]
    assert parity.only["cpsat"] == ["theirs"]
    assert parity.agreement == 0
    assert parity.considered == ["mine", "theirs"]


def test_parity_counts_agreement_in_both_directions(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "easy", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "easy", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    _write(tmp_path, "cbls", "hard", {"status": "no_solution", "objective": None}, [])
    _write(tmp_path, "cpsat", "hard", {"status": "no_solution", "objective": None}, [])
    parity = compare_feasibility(_score(tmp_path, [("easy", 1.0), ("hard", 1.0)]))
    assert parity.both_feasible == ["easy"]
    assert parity.neither_feasible == ["hard"]
    assert parity.agreement == 2
    assert parity.only == {"cbls": [], "cpsat": []}


def test_parity_counts_feasibility_over_the_whole_roster(tmp_path: Path) -> None:
    # The per-engine feasible count and the comparable set have different
    # denominators, and the report states both rather than conflating them.
    _write(tmp_path, "cbls", "solo", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    parity = compare_feasibility(_score(tmp_path, [("solo", 1.0)]))
    assert parity.feasible == {"cbls": 1, "cpsat": 0}
    assert parity.considered == []
    assert parity.roster_size == 1


def test_a_killed_job_cannot_answer_the_parity_question(tmp_path: Path) -> None:
    # Scoring it as "did not reach feasibility" would charge a harness failure to
    # the search -- the mistake `not_run` has always been kept out of the
    # aggregates to avoid.
    _write(tmp_path, "cbls", "dead", {"status": "killed", "message": "oom", "objective": None})
    _write(tmp_path, "cpsat", "dead", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    rows = _score(tmp_path, [("dead", 1.0)])
    parity = compare_feasibility(rows)
    assert parity.considered == []
    assert parity.excluded == [("dead", "cbls", "did not search (status killed)")]


def test_a_withheld_row_is_excluded_from_parity_rather_than_counted_infeasible(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path,
        "cbls",
        "bad",
        {"status": "feasible", "objective": 1.0},
        [(1.0, 1.0)],
        verification={"verdict": "fail", "reason": "row_violation", "marginal": False},
    )
    _write(tmp_path, "cpsat", "bad", {"status": "feasible", "objective": 1.0}, [(1.0, 1.0)])
    rows = _score(tmp_path, [("bad", 1.0)])
    cbls_row = next(r for r in rows if r.engine == "cbls")
    assert parity_verdict(cbls_row) == PARITY_EXCLUDED
    assert compare_feasibility(rows).considered == []


# --- job failure reasons -----------------------------------------------------


def test_the_drivers_kill_message_reaches_the_report(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cbls",
        "dead",
        {"status": "killed", "message": "exceeded 1500.0s wall clock", "objective": None},
    )
    failures = job_failures(_score(tmp_path, [("dead", 1.0)]))
    killed = next(f for f in failures if f.engine == "cbls")
    assert killed.kind == "killed"
    assert killed.reason == "exceeded 1500.0s wall clock"


def test_a_rejected_solutions_reason_reaches_the_report(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cbls",
        "bad",
        {"status": "feasible", "objective": 1.0},
        [(1.0, 1.0)],
        verification={
            "verdict": "fail",
            "reason": "row_violation",
            "message": "row 4.5 (4.5e+06x tol at R7)",
            "marginal": False,
        },
    )
    failures = job_failures(_score(tmp_path, [("bad", 1.0)]))
    rejected = next(f for f in failures if f.kind == "verification fail")
    assert "row_violation" in rejected.reason
    assert "R7" in rejected.reason


def test_a_missing_result_is_reported_as_a_reason_not_a_blank(tmp_path: Path) -> None:
    (tmp_path / "cbls").mkdir()
    failures = job_failures(_score(tmp_path, [("absent", 1.0)]))
    assert [f.kind for f in failures] == ["not_run", "not_run"]
    assert all(f.reason for f in failures)


# --- model-shape cross-check -------------------------------------------------


def _shaped(status: str, **extra: object) -> dict[str, object]:
    record: dict[str, object] = {
        "status": status,
        "objective": 1.0 if status == "feasible" else None,
    }
    record.update(extra)
    return record


def _verdict(columns: int, rows: int) -> dict[str, object]:
    """A passing verdict that also records what SCIP read: the third opinion."""
    return dict(_PASSING, n_columns=columns, n_rows=rows)


def test_free_rows_the_baseline_keeps_make_a_constraint_difference_benign(
    tmp_path: Path,
) -> None:
    # The one this benchmark actually has: OR-Tools' ModelBuilder holds an MPS `N`
    # row after the objective as an unconstrained linear constraint; the CBLS
    # adapter drops it, and so does SCIP. Same feasible set.
    checked = _verdict(220, 51)
    _write(
        tmp_path, "cbls", "mad", _shaped("feasible", n_vars=220, n_cons=51), [(1.0, 1.0)], checked
    )
    _write(
        tmp_path,
        "cpsat",
        "mad",
        _shaped("feasible", n_vars=220, n_cons=52, n_free_cons=1),
        [(1.0, 1.0)],
        checked,
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("mad", 1.0)]))
    assert disagreement.kind == "constraints"
    assert disagreement.benign
    assert "free row" in disagreement.explanation


def test_a_constraint_difference_the_free_rows_do_not_explain_is_flagged(tmp_path: Path) -> None:
    checked = _verdict(48, 40)
    _write(
        tmp_path, "cbls", "odd", _shaped("feasible", n_vars=48, n_cons=40), [(1.0, 1.0)], checked
    )
    _write(
        tmp_path,
        "cpsat",
        "odd",
        _shaped("feasible", n_vars=48, n_cons=44, n_free_cons=1),
        [(1.0, 1.0)],
        checked,
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("odd", 1.0)]))
    assert not disagreement.benign


def test_a_checker_that_read_a_third_count_outranks_the_free_row_excuse(
    tmp_path: Path,
) -> None:
    # The free rows excuse the baseline, never the third reader. Without this the
    # verdict on a checker that read a different program entirely is `benign` the
    # moment the two engines happen to differ by exactly the free-row count.
    checked = _verdict(220, 99)
    _write(
        tmp_path, "cbls", "mad", _shaped("feasible", n_vars=220, n_cons=51), [(1.0, 1.0)], checked
    )
    _write(
        tmp_path,
        "cpsat",
        "mad",
        _shaped("feasible", n_vars=220, n_cons=52, n_free_cons=1),
        [(1.0, 1.0)],
        checked,
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("mad", 1.0)]))
    assert not disagreement.benign
    assert "checker" in disagreement.explanation


def test_extra_constraints_on_the_cbls_side_are_never_benign(tmp_path: Path) -> None:
    # The rule is directional: only the baseline keeps free rows. CBLS holding
    # MORE rows than CP-SAT is a reader defect in the other direction, and the
    # free-row count can never excuse it.
    checked = _verdict(48, 41)
    _write(
        tmp_path, "cbls", "odd", _shaped("feasible", n_vars=48, n_cons=41), [(1.0, 1.0)], checked
    )
    _write(
        tmp_path,
        "cpsat",
        "odd",
        _shaped("feasible", n_vars=48, n_cons=40, n_free_cons=1),
        [(1.0, 1.0)],
        checked,
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("odd", 1.0)]))
    assert not disagreement.benign


def test_two_verdicts_disagreeing_about_the_same_file_are_flagged(tmp_path: Path) -> None:
    # The two verdicts are two SCIP readings of one file. Them differing from each
    # other is a finding of its own, and one that would otherwise vanish: a single
    # reading is all the rest of the cross-check consumes.
    _write(
        tmp_path,
        "cbls",
        "odd",
        _shaped("feasible", n_vars=10, n_cons=5, n_free_cons=0),
        [(1.0, 1.0)],
        _verdict(10, 5),
    )
    _write(
        tmp_path,
        "cpsat",
        "odd",
        _shaped("feasible", n_vars=10, n_cons=5, n_free_cons=0),
        [(1.0, 1.0)],
        _verdict(10, 77),
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("odd", 1.0)]))
    assert not disagreement.benign
    assert "two different shapes" in disagreement.explanation


def test_a_row_the_runner_could_not_dump_says_so_rather_than_reading_as_unchecked(
    tmp_path: Path,
) -> None:
    # `withholds` is true for a solution_write_error too, so testing `withheld`
    # first would report a disk failure as "nobody checked the solution".
    _write(
        tmp_path,
        "cbls",
        "nodisk",
        {"status": "solution_write_error", "objective": 1.0, "message": "no space left"},
        [(1.0, 1.0)],
        verification=None,
    )
    _write(tmp_path, "cpsat", "nodisk", _shaped("feasible"), [(1.0, 1.0)])
    rows = _score(tmp_path, [("nodisk", 1.0)])
    ((_, engine, why),) = compare_feasibility(rows).excluded
    assert engine == "cbls"
    assert "solution_write_error" in why
    # And it appears once in the failure table, not twice.
    assert [f.engine for f in job_failures(rows)] == ["cbls"]


def test_a_variable_count_difference_is_never_benign(tmp_path: Path) -> None:
    checked = _verdict(48, 10)
    _write(
        tmp_path, "cbls", "odd", _shaped("feasible", n_vars=48, n_cons=10), [(1.0, 1.0)], checked
    )
    _write(
        tmp_path,
        "cpsat",
        "odd",
        _shaped("feasible", n_vars=49, n_cons=10, n_free_cons=7),
        [(1.0, 1.0)],
        checked,
    )
    kinds = {(d.kind, d.benign) for d in cross_check_shapes(_score(tmp_path, [("odd", 1.0)]))}
    assert ("variables", False) in kinds


def test_the_checker_disagreeing_with_both_engines_is_flagged(tmp_path: Path) -> None:
    # SCIP drops free rows too, so it agreeing with neither engine is not
    # something the free-row rule can explain in either direction.
    verdict = dict(_PASSING, n_columns=10, n_rows=99)
    _write(
        tmp_path, "cbls", "odd", _shaped("feasible", n_vars=10, n_cons=10), [(1.0, 1.0)], verdict
    )
    _write(
        tmp_path,
        "cpsat",
        "odd",
        _shaped("feasible", n_vars=10, n_cons=10, n_free_cons=0),
        [(1.0, 1.0)],
        verdict,
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("odd", 1.0)]))
    assert not disagreement.benign
    assert "checker" in disagreement.explanation


def test_a_free_row_excuse_with_no_third_reading_is_flagged(tmp_path: Path) -> None:
    """Absence of a checker is not agreement with one.

    A verdict exists only for a row that reported a solution, so on every
    instance where NEITHER engine found one there is no third reading -- and
    `free_rows` is written only by the baseline, so without SCIP the excuse is
    certified by the reader under test. Those are exactly the instances where a
    translation defect best explains the double failure.
    """
    _write(tmp_path, "cbls", "dark", _shaped("no_solution", n_vars=10, n_cons=40), [])
    _write(
        tmp_path,
        "cpsat",
        "dark",
        _shaped("no_solution", n_vars=10, n_cons=44, n_free_cons=4),
        [],
    )
    (disagreement,) = cross_check_shapes(_score(tmp_path, [("dark", 1.0)]))
    assert not disagreement.benign
    assert "unchecked" in disagreement.explanation


def test_an_instance_with_no_shape_from_one_engine_is_not_reported_as_agreeing(
    tmp_path: Path,
) -> None:
    """A killed job carries no counts, so nothing about it was compared."""
    _write(tmp_path, "cbls", "gone", _shaped("killed"), [])
    _write(tmp_path, "cpsat", "gone", _shaped("feasible", n_vars=10, n_cons=5), [(1.0, 1.0)])
    rows = _score(tmp_path, [("gone", 1.0)])
    assert cross_checked_instances(rows) == set()
    assert shape_notes_by_instance(rows)["gone"] == "not_compared"


def test_an_instance_both_engines_shaped_is_reported_as_agreeing(tmp_path: Path) -> None:
    for engine in ENGINES:
        _write(
            tmp_path,
            engine,
            "same",
            _shaped("feasible", n_vars=10, n_cons=5, n_free_cons=0),
            [(1.0, 1.0)],
            _verdict(10, 5),
        )
    rows = _score(tmp_path, [("same", 1.0)])
    assert cross_checked_instances(rows) == {"same"}
    assert shape_notes_by_instance(rows)["same"] == "agree"


def test_every_fixture_file_is_tracked_by_git() -> None:
    """An untracked fixture file is green here and red nowhere else.

    The regeneration test reads the working tree, not the index.

    This already happened once: `.gitignore`'s blanket `results/` swallowed the
    whole fixture directory, so criterion 8 passed only on the machine that
    generated it, while the guard test beside it stayed green because it reads
    the tracked expected file.
    """
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "benchmarks/mipfeas/testdata"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split("\0")
    on_disk = {
        str(path.relative_to(REPO_ROOT))
        for path in (FIXTURE).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    missing = sorted(on_disk - {name for name in tracked if name})
    assert missing == [], f"fixture files not tracked by git: {missing}"


def test_a_roster_that_repeats_an_instance_is_refused(tmp_path: Path) -> None:
    """A duplicate double-counts the instance and makes the two artifacts
    disagree about the roster size."""
    roster = tmp_path / "roster.csv"
    roster.write_text("instance,reference_value,reference_kind\na,1.0,opt\nb,2.0,opt\na,1.0,opt\n")
    with pytest.raises(ValueError, match="repeats a"):
        read_roster(roster)


def test_agreeing_shapes_produce_no_finding(tmp_path: Path) -> None:
    for engine in ENGINES:
        _write(
            tmp_path,
            engine,
            "same",
            _shaped("feasible", n_vars=10, n_cons=5, n_free_cons=0),
            [(1.0, 1.0)],
            _verdict(10, 5),
        )
    assert cross_check_shapes(_score(tmp_path, [("same", 1.0)])) == []


# --- trace health ------------------------------------------------------------


def test_a_profile_that_collapsed_to_one_point_is_counted_as_degraded(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cpsat",
        "flat",
        _shaped("feasible", trace_source="final_only"),
        [(1.0, 1.0)],
    )
    health = trace_health(_score(tmp_path, [("flat", 1.0)]), "cpsat")
    assert health.degraded == 1
    assert health.degraded_instances == ["flat"]
    assert health.reported_feasible == 1


def test_both_engines_genuine_trace_sources_count_as_healthy(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "ok", _shaped("feasible", trace_source="callback"), [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "ok", _shaped("feasible", trace_source="log"), [(1.0, 1.0)])
    rows = _score(tmp_path, [("ok", 1.0)])
    assert [trace_health(rows, e).healthy for e in ENGINES] == [1, 1]


def test_a_run_that_found_nothing_is_not_in_the_trace_denominator(tmp_path: Path) -> None:
    # It has no incumbent profile to have; counting it would make every
    # no-solution row read as a harness fault.
    _write(tmp_path, "cbls", "empty", _shaped("no_solution", trace_source="final_only"), [])
    health = trace_health(_score(tmp_path, [("empty", 1.0)]), "cbls")
    assert health.reported_feasible == 0
    assert health.degraded == 0


# --- timing decomposition ----------------------------------------------------


def test_setup_and_solve_time_are_separate_columns(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cbls",
        "slow",
        _shaped("feasible", read_seconds=1.5, build_seconds=2.5, wall_seconds=61.0),
        [(1.0, 1.0)],
    )
    (row,) = [r for r in _score(tmp_path, [("slow", 1.0)]) if r.engine == "cbls"]
    assert row.setup_seconds == pytest.approx(4.0)
    assert row.solve_seconds == pytest.approx(61.0)


def test_an_overrun_and_a_long_setup_are_reported_apart(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cbls",
        "slow",
        _shaped("feasible", setup_seconds=4.0, wall_seconds=75.0),
        [(1.0, 1.0)],
    )
    summary = timing_summary(_score(tmp_path, [("slow", 1.0)]), "cbls", budget=60.0)
    assert summary.setup_max == pytest.approx(4.0)
    assert summary.overruns == 1
    assert summary.overrun_max == pytest.approx(15.0)


def test_a_results_directory_that_measured_no_setup_states_no_magnitude(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "old", _shaped("feasible", wall_seconds=60.0), [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "old", _shaped("feasible", wall_seconds=60.0), [(1.0, 1.0)])
    rows = _score(tmp_path, [("old", 1.0)])
    assert all(timing_summary(rows, e, 60.0).setup_measured == 0 for e in ENGINES)
    report = render_report(
        rows, [summarize(rows, e) for e in ENGINES], 60.0, tmp_path / "r.csv", tmp_path / "t.csv"
    )
    assert "No setup time was recorded" in report


# --- headline and labelling --------------------------------------------------


def test_the_headline_block_leads_with_defects_then_parity(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "x", _shaped("feasible"), [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "x", _shaped("no_solution"), [])
    rows = _score(tmp_path, [("x", 1.0)])
    lines = headline_lines(rows, [summarize(rows, e) for e in ENGINES])
    assert lines[0].startswith("# DEFECTS")
    body = "\n".join(lines)
    assert "# FEASIBILITY PARITY:" in body
    assert "# MODEL SHAPE" in body
    assert "# TRACE HEALTH" in body
    assert "cbls only (1, i.e. not cpsat): x" in body


def test_the_anytime_aggregate_names_the_worker_pairing_it_measures() -> None:
    # Not "solvers" in general: the only baseline is CP-SAT's fj + ls subsolvers
    # under num_violation_ls, and a table quoted out of context must say so.
    assert "fj + ls" in ANYTIME_LABEL
    assert "num_violation_ls" in ANYTIME_LABEL
    assert "Primal Integral" in ANYTIME_LABEL


def test_the_report_states_the_shape_rule_rather_than_only_the_verdicts(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "x", _shaped("feasible"), [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "x", _shaped("feasible"), [(1.0, 1.0)])
    rows = _score(tmp_path, [("x", 1.0)])
    report = render_report(
        rows, [summarize(rows, e) for e in ENGINES], 60.0, tmp_path / "r.csv", tmp_path / "t.csv"
    )
    assert SHAPE_RULE in report


def test_defect_total_counts_every_counter(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "cbls",
        "bad",
        _shaped("feasible"),
        [(1.0, 1.0)],
        verification={"verdict": "fail", "reason": "row_violation", "marginal": False},
    )
    _write(tmp_path, "cpsat", "bad", _shaped("feasible", trace_source="final_only"), [(1.0, 1.0)])
    rows = _score(tmp_path, [("bad", 1.0)])
    defects = collect_defects(rows, [summarize(rows, e) for e in ENGINES])
    assert defects.verification_failed["cbls"] == 1
    assert defects.trace_degraded == 1
    assert defects.total >= 2


def test_a_clean_run_reports_no_defects(tmp_path: Path) -> None:
    for engine in ENGINES:
        _write(
            tmp_path,
            engine,
            "x",
            _shaped("feasible", trace_source="callback", n_vars=10, n_cons=5, n_free_cons=0),
            [(1.0, 1.0)],
        )
    rows = _score(tmp_path, [("x", 1.0)])
    assert all(parity_verdict(r) == PARITY_FEASIBLE for r in rows)
    assert collect_defects(rows, [summarize(rows, e) for e in ENGINES]).total == 0


# --- the regeneration check --------------------------------------------------


def test_regenerating_the_report_reproduces_the_committed_numbers(tmp_path: Path) -> None:
    """Score the frozen results directory and demand the committed artifacts back.

    The criterion with teeth: every headline the report publishes -- the defect
    counters, both difference sets, the shape verdicts, the trace-health counts,
    the timing split and the anytime aggregate -- is derived from
    `testdata/results/` and compared byte for byte. A change to any of them has
    to be made deliberately, by regenerating the two files with the command in
    `benchmarks/mipfeas/testdata/README.md`.
    """
    table = tmp_path / "expected_comparison.csv"
    report = tmp_path / "expected_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "mipfeas" / "primal_integral.py"),
            "--results-dir",
            str(FIXTURE / "results"),
            "--roster",
            str(FIXTURE / "roster.csv"),
            "--budget",
            str(FIXTURE_BUDGET),
            "--out",
            str(table),
            "--report",
            str(report),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit 1 is the fixture's rejected solution, which is the point of having one:
    # a correctness benchmark whose checker refused a published point must not
    # score at exit 0.
    assert completed.returncode == 1, completed.stderr
    assert table.read_text() == (FIXTURE / "expected_comparison.csv").read_text()
    assert report.read_text() == (FIXTURE / "expected_report.md").read_text()


def test_the_fixture_exercises_every_branch_the_report_has(tmp_path: Path) -> None:
    """A fixture that lost its defects would still reproduce byte for byte.

    So this checks the fixture is still worth comparing against: each of the
    findings the report exists to surface has to be present in it.
    """
    report = (FIXTURE / "expected_report.md").read_text()
    for expected in (
        "**cbls only** (1)",
        "**cpsat only** (1)",
        "| verification_failed | 1 | 0 | 1 |",
        "| not_run | 0 | 1 | 1 |",
        "exceeded 902.0s wall clock",
        "| mad | constraints | cbls=51 cpsat=52 checker=51 free_rows=1 | benign |",
        "FLAGGED",
        "- cpsat degraded: degraded-trace",
        "3.204 (slow-start)",
    ):
        assert expected in report, expected


# --- Section 8: the machine record --------------------------------------------
#
# A wall-clock-limited comparison is a statement about a machine as much as about
# an algorithm, so the report quotes what the driver recorded rather than leaving
# it in a results directory nobody publishes (issue #137).

_RECORD: dict[str, object] = {
    "budget_seconds": 600.0,
    "concurrency": {
        "jobs": 4,
        "large_instance_jobs": 1,
        "cpsat_workers": 1,
        "mem_limit_gb": 6.0,
    },
    "machine": {
        "host": "bench-01",
        "platform": "Linux-6.8.0",
        "cpu_count": 16,
        "cpu_affinity": 8,
        "memory_total_kib": 33554432,
    },
    "versions": {
        "engine_commit": "deadbee",
        "ortools": "9.15.6755",
        "pyscipopt": "6.2.1",
        "python": "3.12.0",
    },
    "references": {
        "solution_file": "miplib2017-v36.solu",
        "pinned": {"miplib2017-v36.solu": "9236602294c1a5aca5248b6f7d03689a"},
    },
    "started_at": "2026-01-01T00:00:00+00:00",
    "finished_at": "2026-01-01T06:00:00+00:00",
    "status": "complete",
}


def test_the_record_section_states_the_concurrency_the_run_used() -> None:
    section = "\n".join(run_record_section(_RECORD, 1))

    assert "**4 job(s) at a time**" in section
    assert "large instances 1 at a time" in section
    assert "CP-SAT 1 worker(s)" in section
    assert "address-space cap 6.0 GB" in section


def test_the_record_section_names_the_host_cores_memory_and_budget() -> None:
    section = "\n".join(run_record_section(_RECORD, 1))

    assert "| host | bench-01 |" in section
    assert "16 (8 available to the process)" in section
    assert "32.0 GiB" in section
    assert "600.0s per instance-engine pair" in section


def test_the_record_section_names_the_engine_commit_and_solver_versions() -> None:
    section = "\n".join(run_record_section(_RECORD, 1))

    assert "| engine commit | deadbee |" in section
    assert "ortools 9.15.6755" in section
    assert "PySCIPOpt 6.2.1" in section


def test_the_record_section_names_the_yardstick_the_gaps_were_scored_against() -> None:
    # An upstream revision of the solution file moves every gap in the table at
    # once, so which one produced it is part of the result.
    section = "\n".join(run_record_section(_RECORD, 1))

    assert "`miplib2017-v36.solu`" in section
    assert "9236602294c1a5ac" in section


def test_a_resumed_results_directory_says_so_rather_than_claiming_one_machine() -> None:
    section = "\n".join(run_record_section(_RECORD, 3))
    assert "3 invocations" in section


def test_results_with_no_machine_record_are_called_an_anecdote() -> None:
    # Silence here is the failure mode: a published table whose concurrency nobody
    # recorded reads exactly like one whose concurrency was stated.
    section = "\n".join(run_record_section(None, 0))

    assert "## 8. Machine and run record" in section
    assert "No machine record was written" in section
    assert "concurrency" in section


def test_read_run_record_returns_the_last_invocation_and_the_count(tmp_path: Path) -> None:
    (tmp_path / "run_record.json").write_text(
        json.dumps({"runs": [{"status": "complete"}, {"status": "running"}]})
    )

    record, count = read_run_record(tmp_path)

    assert count == 2
    assert record == {"status": "running"}


def test_read_run_record_treats_a_truncated_file_as_absent(tmp_path: Path) -> None:
    # A driver killed mid-write must not make the scorer abort on the file.
    (tmp_path / "run_record.json").write_text('{"runs": [')

    assert read_run_record(tmp_path) == (None, 0)


def test_read_run_record_of_a_directory_that_has_none(tmp_path: Path) -> None:
    assert read_run_record(tmp_path) == (None, 0)


def test_the_report_of_the_frozen_fixture_carries_its_machine_record() -> None:
    # The fixture holds a (constructed) run record precisely so that section 8 is
    # pinned byte for byte like every other section.
    report = (FIXTURE / "expected_report.md").read_text()

    assert "## 8. Machine and run record" in report
    assert "| host | fixture-host |" in report
    assert "**2 job(s) at a time**" in report


def test_the_scorer_warns_when_no_machine_record_is_beside_the_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _write(results, "cbls", "only-cbls", {"status": "no_solution", "objective": None})
    _write(results, "cpsat", "only-cbls", {"status": "no_solution", "objective": None})
    roster = tmp_path / "roster.csv"
    roster.write_text("instance,reference_value,reference_kind\nonly-cbls,1.0,opt\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks" / "mipfeas" / "primal_integral.py"),
            "--results-dir",
            str(results),
            "--roster",
            str(roster),
            "--budget",
            "2",
            "--out",
            str(tmp_path / "out.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "no run_record.json" in completed.stderr
    assert "No machine record was written" in (tmp_path / "out_report.md").read_text()


def test_a_field_the_driver_could_not_measure_says_so_rather_than_none() -> None:
    # Off Linux there is no /proc/meminfo and no sched_getaffinity, and os.cpu_count()
    # can be None. `| cores | None |` in a published table reads as a scorer bug
    # rather than as a fact about the machine.
    section = "\n".join(run_record_section({}, 1))

    assert "None" not in section
    assert "| host | not recorded |" in section
    assert "| concurrency | not recorded |" in section


def test_a_resume_that_ran_nothing_does_not_claim_to_be_the_machine_that_measured() -> None:
    # The driver appends a record on every invocation, so confirming a finished
    # directory from a laptop would otherwise publish the laptop as the machine.
    record = dict(_RECORD, outcome={"jobs_run": 0, "failures": 0, "rejected": 0})
    section = "\n".join(run_record_section(record, 2))

    assert "This invocation ran no jobs" in section


def test_an_invocation_that_ran_jobs_carries_no_such_caveat() -> None:
    record = dict(_RECORD, outcome={"jobs_run": 26, "failures": 0, "rejected": 0})
    assert "This invocation ran no jobs" not in "\n".join(run_record_section(record, 1))


def test_a_run_that_skipped_its_preconditions_says_so_in_the_report() -> None:
    # The flag is what makes a run unpublishable, and terminal scrollback does not
    # reach whoever reads the table.
    record = dict(_RECORD, run={"preconditions_checked": False})
    section = "\n".join(run_record_section(record, 1))

    assert "NOT CHECKED" in section
    assert "not publishable" in section


def test_a_checked_run_carries_no_preconditions_row() -> None:
    record = dict(_RECORD, run={"preconditions_checked": True})
    assert "NOT CHECKED" not in "\n".join(run_record_section(record, 1))
