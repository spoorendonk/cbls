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
SCORER = REPO_ROOT / "benchmarks" / "mipfeas" / "primal_integral.py"

_PASSING: dict[str, object] = {
    "verdict": "pass",
    "reason": "",
    "marginal": False,
    "tolerances": {"row_absolute": 1e-06},
    "n_columns": 10,
    "n_rows": 5,
}
_REJECTED: dict[str, object] = {"verdict": "fail", "reason": "row_violation", "marginal": False}


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


def _score(directory: Path, instances: list[str]) -> list[Scored]:
    return [
        score_instance(name, engine, 1.0, "opt", directory, budget=60.0)
        for name in instances
        for engine in ENGINES
    ]


def _shaped(status: str, **extra: object) -> dict[str, object]:
    record: dict[str, object] = {
        "status": status,
        "objective": 1.0 if status == "feasible" else None,
    }
    record.update(extra)
    return record


def _job(directory: Path, engine: str, instance: str, state: str, **extra: object) -> None:
    """One job in one of the states the report distinguishes; `absent` writes nothing."""
    if state == "absent":
        (directory / engine).mkdir(parents=True, exist_ok=True)
        return
    status = "feasible" if state == "rejected" else state
    record = _shaped(status, **extra)
    verification = _REJECTED if state == "rejected" else _PASSING
    trace = [(1.0, 1.0)] if status == "feasible" else []
    _write(directory, engine, instance, record, trace, verification)


# --- feasibility parity ------------------------------------------------------


@pytest.mark.parametrize(
    ("jobs", "expected"),
    [
        (
            {"mine": ("feasible", "no_solution"), "theirs": ("no_solution", "feasible")},
            {
                "only": {"cbls": ["mine"], "cpsat": ["theirs"]},
                "agreement": 0,
                "considered": ["mine", "theirs"],
            },
        ),
        (
            {"easy": ("feasible", "feasible"), "hard": ("no_solution", "no_solution")},
            {
                "both_feasible": ["easy"],
                "neither_feasible": ["hard"],
                "agreement": 2,
                "only": {"cbls": [], "cpsat": []},
            },
        ),
        # The per-engine feasible count and the comparable set have different
        # denominators, and the report states both rather than conflating them.
        (
            {"solo": ("feasible", "absent")},
            {"feasible": {"cbls": 1, "cpsat": 0}, "considered": [], "roster_size": 1},
        ),
        # Scoring a killed job as "did not reach feasibility" would charge a harness
        # failure to the search -- the mistake `not_run` has always been kept out of
        # the aggregates to avoid.
        (
            {"dead": ("killed", "feasible")},
            {"considered": [], "excluded": [("dead", "cbls", "did not search (status killed)")]},
        ),
        # A withheld row is excluded, not counted infeasible.
        ({"bad": ("rejected", "feasible")}, {"considered": []}),
    ],
    ids=["asymmetric-difference-sets", "agreement-both-ways", "whole-roster", "killed", "withheld"],
)
def test_feasibility_parity(
    tmp_path: Path, jobs: dict[str, tuple[str, str]], expected: dict[str, object]
) -> None:
    for instance, states in jobs.items():
        for engine, state in zip(ENGINES, states, strict=True):
            _job(tmp_path, engine, instance, state)
    rows = _score(tmp_path, list(jobs))
    parity = compare_feasibility(rows)
    for field, value in expected.items():
        assert getattr(parity, field) == value, field
    for row in rows:
        if jobs.get(row.instance, ("", ""))[ENGINES.index(row.engine)] == "rejected":
            assert parity_verdict(row) == PARITY_EXCLUDED


# --- job failure reasons -----------------------------------------------------


@pytest.mark.parametrize(
    ("state", "extra", "verification", "kinds", "reason"),
    [
        (
            "killed",
            {"message": "exceeded 1500.0s wall clock"},
            None,
            ["killed", "not_run"],
            "exceeded 1500.0s wall clock",
        ),
        (
            "feasible",
            {},
            {**_REJECTED, "message": "row 4.5 (4.5e+06x tol at R7)"},
            ["verification fail", "not_run"],
            "row_violation",
        ),
        # A reason, not a blank.
        ("absent", {}, None, ["not_run", "not_run"], "no result file was written for this job"),
    ],
    ids=["driver-kill-message", "rejected-solution-reason", "missing-result"],
)
def test_a_failed_job_reaches_the_report_with_its_reason(
    tmp_path: Path,
    state: str,
    extra: dict[str, object],
    verification: dict[str, object] | None,
    kinds: list[str],
    reason: str,
) -> None:
    if state == "absent":
        (tmp_path / "cbls").mkdir()
    else:
        record = _shaped(state, **extra)
        _write(tmp_path, "cbls", "x", record, [(1.0, 1.0)], verification or _PASSING)
    failures = job_failures(_score(tmp_path, ["x"]))
    assert [(f.engine, f.kind) for f in failures] == list(zip(ENGINES, kinds, strict=True))
    assert all(f.reason for f in failures)
    if verification is None:
        # The driver's own words, not a paraphrase of them.
        assert failures[0].reason == reason
    else:
        assert reason in failures[0].reason and "R7" in failures[0].reason


def test_a_row_the_runner_could_not_dump_says_so_rather_than_reading_as_unchecked(
    tmp_path: Path,
) -> None:
    # `withholds` is true for a solution_write_error too, so testing `withheld`
    # first would report a disk failure as "nobody checked the solution".
    record = {"status": "solution_write_error", "objective": 1.0, "message": "no space left"}
    _write(tmp_path, "cbls", "nodisk", record, [(1.0, 1.0)], verification=None)
    _write(tmp_path, "cpsat", "nodisk", _shaped("feasible"), [(1.0, 1.0)])
    rows = _score(tmp_path, ["nodisk"])
    ((_, engine, why),) = compare_feasibility(rows).excluded
    assert engine == "cbls"
    assert "solution_write_error" in why
    # And it appears once in the failure table, not twice.
    assert [f.engine for f in job_failures(rows)] == ["cbls"]


# --- model-shape cross-check -------------------------------------------------


def _shape(variables: int, constraints: int, free: int | None = None) -> dict[str, object]:
    counts: dict[str, object] = {"n_vars": variables, "n_cons": constraints}
    return counts if free is None else {**counts, "n_free_cons": free}


@pytest.mark.parametrize(
    ("status", "cbls", "cpsat", "checked", "findings", "explanation"),
    [
        # The one this benchmark actually has: OR-Tools' ModelBuilder holds an MPS
        # `N` row after the objective as an unconstrained linear constraint; the
        # CBLS adapter drops it, and so does SCIP. Same feasible set.
        (
            "feasible",
            _shape(220, 51),
            _shape(220, 52, 1),
            ((220, 51), (220, 51)),
            [("constraints", True)],
            "free row",
        ),
        (
            "feasible",
            _shape(48, 40),
            _shape(48, 44, 1),
            ((48, 40), (48, 40)),
            [("constraints", False)],
            "",
        ),
        # The free rows excuse the baseline, never the third reader. Without this a
        # checker that read a different program entirely is `benign` the moment the
        # two engines happen to differ by exactly the free-row count.
        (
            "feasible",
            _shape(220, 51),
            _shape(220, 52, 1),
            ((220, 99), (220, 99)),
            [("constraints", False)],
            "checker",
        ),
        # The rule is directional: only the baseline keeps free rows. CBLS holding
        # MORE rows than CP-SAT is a reader defect the free-row count can never excuse.
        (
            "feasible",
            _shape(48, 41),
            _shape(48, 40, 1),
            ((48, 41), (48, 41)),
            [("constraints", False)],
            "",
        ),
        # The two verdicts are two SCIP readings of one file. Them differing from
        # each other is a finding of its own, and one that would otherwise vanish:
        # a single reading is all the rest of the cross-check consumes.
        (
            "feasible",
            _shape(10, 5, 0),
            _shape(10, 5, 0),
            ((10, 5), (10, 77)),
            [("constraints", False)],
            "two different shapes",
        ),
        (
            "feasible",
            _shape(48, 10),
            _shape(49, 10, 7),
            ((48, 10), (48, 10)),
            [("variables", False)],
            "",
        ),
        # SCIP drops free rows too, so it agreeing with neither engine is not
        # something the free-row rule can explain in either direction.
        (
            "feasible",
            _shape(10, 10),
            _shape(10, 10, 0),
            ((10, 99), (10, 99)),
            [("constraints", False)],
            "checker",
        ),
        # Absence of a checker is not agreement with one. A verdict exists only for
        # a row that reported a solution, so where NEITHER engine found one there is
        # no third reading -- and `free_rows` is written only by the baseline, so
        # without SCIP the excuse is certified by the reader under test. Those are
        # exactly the instances where a translation defect best explains the double
        # failure.
        (
            "no_solution",
            _shape(10, 40),
            _shape(10, 44, 4),
            None,
            [("constraints", False)],
            "unchecked",
        ),
        ("feasible", _shape(10, 5, 0), _shape(10, 5, 0), ((10, 5), (10, 5)), [], ""),
    ],
    ids=[
        "free-rows-benign",
        "unexplained-constraint-difference",
        "checker-third-count-outranks-free-rows",
        "extra-cbls-constraints",
        "two-verdicts-disagree",
        "variable-count-difference",
        "checker-disagrees-with-both",
        "free-row-excuse-without-a-checker",
        "agreeing-shapes",
    ],
)
def test_the_model_shape_cross_check(
    tmp_path: Path,
    status: str,
    cbls: dict[str, object],
    cpsat: dict[str, object],
    checked: tuple[tuple[int, int], tuple[int, int]] | None,
    findings: list[tuple[str, bool]],
    explanation: str,
) -> None:
    for index, (engine, counts) in enumerate((("cbls", cbls), ("cpsat", cpsat))):
        verdict = None
        if checked is not None:
            columns, rows = checked[index]
            verdict = dict(_PASSING, n_columns=columns, n_rows=rows)
        trace = [(1.0, 1.0)] if status == "feasible" else []
        _write(tmp_path, engine, "x", _shaped(status, **counts), trace, verdict)

    disagreements = cross_check_shapes(_score(tmp_path, ["x"]))

    found = {(d.kind, d.benign) for d in disagreements}
    assert set(findings) <= found, found
    if not findings:
        assert disagreements == []
    elif findings[0][0] == "constraints":
        assert len(disagreements) == 1
        assert explanation in disagreements[0].explanation


@pytest.mark.parametrize(
    ("states", "compared", "note"),
    [
        # A killed job carries no counts, so nothing about it was compared -- on
        # either side.
        (("killed", "feasible"), set(), "not_compared"),
        (("feasible", "killed"), set(), "not_compared"),
        (("feasible", "feasible"), {"x"}, "agree"),
    ],
    ids=["cbls-unshaped", "cpsat-unshaped", "both-shaped"],
)
def test_an_instance_is_reported_as_agreeing_only_when_both_shapes_were_read(
    tmp_path: Path, states: tuple[str, str], compared: set[str], note: str
) -> None:
    for engine, state in zip(ENGINES, states, strict=True):
        counts = _shape(10, 5, 0) if state == "feasible" else {}
        _job(tmp_path, engine, "x", state, **counts)
    rows = _score(tmp_path, ["x"])
    assert cross_checked_instances(rows) == compared
    assert shape_notes_by_instance(rows)["x"] == note


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


# --- trace health ------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "sources", "engine", "expected"),
    [
        (
            "feasible",
            ("callback", "final_only"),
            "cpsat",
            {"degraded": 1, "degraded_instances": ["x"], "reported_feasible": 1},
        ),
        ("feasible", ("callback", "log"), "cbls", {"healthy": 1}),
        ("feasible", ("callback", "log"), "cpsat", {"healthy": 1}),
        # A run that found nothing has no incumbent profile to have; counting it
        # would make every no-solution row read as a harness fault.
        (
            "no_solution",
            ("final_only", "final_only"),
            "cbls",
            {"reported_feasible": 0, "degraded": 0},
        ),
    ],
    ids=["collapsed-to-one-point", "cbls-callback-healthy", "cpsat-log-healthy", "found-nothing"],
)
def test_trace_health(
    tmp_path: Path,
    status: str,
    sources: tuple[str, str],
    engine: str,
    expected: dict[str, object],
) -> None:
    for name, source in zip(ENGINES, sources, strict=True):
        trace = [(1.0, 1.0)] if status == "feasible" else []
        _write(tmp_path, name, "x", _shaped(status, trace_source=source), trace)
    health = trace_health(_score(tmp_path, ["x"]), engine)
    for field, value in expected.items():
        assert getattr(health, field) == value, field


# --- timing decomposition ----------------------------------------------------


def test_setup_and_solve_time_are_separate_and_reported_apart(tmp_path: Path) -> None:
    # Setup is recorded in halves by a runner that threw between them, and whole
    # by one that did not; an overrun is the solve running past the budget, which
    # search initialisation not being bounded by the deadline makes possible.
    halves = _shaped("feasible", read_seconds=1.5, build_seconds=2.5, wall_seconds=61.0)
    _write(tmp_path, "cbls", "slow", halves, [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "slow", _shaped("feasible", setup_seconds=4.0, wall_seconds=75.0))
    rows = _score(tmp_path, ["slow"])

    assert [(r.setup_seconds, r.solve_seconds) for r in rows] == [(4.0, 61.0), (4.0, 75.0)]
    summary = timing_summary(rows, "cpsat", budget=60.0)
    assert summary.setup_max == pytest.approx(4.0)
    assert summary.overruns == 1
    assert summary.overrun_max == pytest.approx(15.0)


def _report(rows: list[Scored], tmp_path: Path) -> str:
    summaries = [summarize(rows, e) for e in ENGINES]
    return render_report(rows, summaries, 60.0, tmp_path / "r.csv", tmp_path / "t.csv")


def test_a_results_directory_that_measured_no_setup_states_no_magnitude(tmp_path: Path) -> None:
    for engine in ENGINES:
        _write(tmp_path, engine, "old", _shaped("feasible", wall_seconds=60.0), [(1.0, 1.0)])
    rows = _score(tmp_path, ["old"])
    assert all(timing_summary(rows, e, 60.0).setup_measured == 0 for e in ENGINES)
    assert "No setup time was recorded" in _report(rows, tmp_path)


# --- headline and labelling --------------------------------------------------


def test_the_headline_block_leads_with_defects_then_parity(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "x", _shaped("feasible"), [(1.0, 1.0)])
    _write(tmp_path, "cpsat", "x", _shaped("no_solution"), [])
    rows = _score(tmp_path, ["x"])
    lines = headline_lines(rows, [summarize(rows, e) for e in ENGINES])
    assert lines[0].startswith("# DEFECTS")
    body = "\n".join(lines)
    assert "# FEASIBILITY PARITY:" in body
    assert "# MODEL SHAPE" in body
    assert "# TRACE HEALTH" in body
    assert "cbls only (1, i.e. not cpsat): x" in body
    # The report states the shape rule, not only the verdicts it produced.
    assert SHAPE_RULE in _report(rows, tmp_path)


def test_the_anytime_aggregate_names_the_worker_pairing_it_measures() -> None:
    # Not "solvers" in general: the only baseline is CP-SAT's fj + ls subsolvers
    # under num_violation_ls, and a table quoted out of context must say so.
    assert "fj + ls" in ANYTIME_LABEL
    assert "num_violation_ls" in ANYTIME_LABEL
    assert "Primal Integral" in ANYTIME_LABEL


def test_the_defect_total_counts_every_counter_and_a_clean_run_has_none(tmp_path: Path) -> None:
    _write(tmp_path, "cbls", "bad", _shaped("feasible"), [(1.0, 1.0)], _REJECTED)
    _write(tmp_path, "cpsat", "bad", _shaped("feasible", trace_source="final_only"), [(1.0, 1.0)])
    rows = _score(tmp_path, ["bad"])
    defects = collect_defects(rows, [summarize(rows, e) for e in ENGINES])
    assert defects.verification_failed["cbls"] == 1
    assert defects.trace_degraded == 1
    assert defects.total >= 2

    clean = tmp_path / "clean"
    for engine in ENGINES:
        record = _shaped("feasible", trace_source="callback", **_shape(10, 5, 0))
        _write(clean, engine, "x", record, [(1.0, 1.0)])
    rows = _score(clean, ["x"])
    assert all(parity_verdict(r) == PARITY_FEASIBLE for r in rows)
    assert collect_defects(rows, [summarize(rows, e) for e in ENGINES]).total == 0


# --- the regeneration check --------------------------------------------------


def test_regenerating_the_report_reproduces_the_committed_numbers(tmp_path: Path) -> None:
    """Score the frozen results directory and demand the committed artifacts back.

    The criterion with teeth: every headline the report publishes -- the defect
    counters, both difference sets, the shape verdicts, the trace-health counts,
    the timing split, the anytime aggregate and the machine record -- is derived
    from `testdata/results/` and compared byte for byte. A change to any of them
    has to be made deliberately, by regenerating the two files with the command in
    `benchmarks/mipfeas/testdata/README.md`.
    """
    table = tmp_path / "expected_comparison.csv"
    report = tmp_path / "expected_report.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCORER),
            *("--results-dir", str(FIXTURE / "results"), "--roster", str(FIXTURE / "roster.csv")),
            *("--budget", str(FIXTURE_BUDGET), "--out", str(table), "--report", str(report)),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # Exit 1 is the fixture's rejected solution, which is the point of having one:
    # a correctness benchmark whose checker refused a published point must not
    # score at exit 0.
    assert completed.returncode == 1, completed.stderr
    # Bytes, not text: the table carries csv.writer's CRLF row endings after an LF
    # header block, and read_text() folds both into "\n" -- so a text comparison
    # would pass a regeneration that changed every line ending in the file.
    assert table.read_bytes() == (FIXTURE / "expected_comparison.csv").read_bytes()
    assert report.read_bytes() == (FIXTURE / "expected_report.md").read_bytes()


def test_the_fixture_exercises_every_branch_the_report_has() -> None:
    """A fixture that lost its defects would still reproduce byte for byte.

    So this checks the fixture is still worth comparing against: each of the
    findings the report exists to surface has to be present in it -- including a
    (constructed) run record, so that section 8 is pinned like every other.
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
        "## 8. Machine and run record",
        "| host | fixture-host |",
        "**2 job(s) at a time**",
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


@pytest.mark.parametrize(
    ("record", "invocations", "said", "not_said"),
    [
        (
            _RECORD,
            1,
            [
                # The concurrency the run used ...
                "**4 job(s) at a time**",
                "large instances 1 at a time",
                "CP-SAT 1 worker(s)",
                "address-space cap 6.0 GB",
                # ... the host, cores, memory and budget ...
                "| host | bench-01 |",
                "16 (8 available to the process)",
                "32.0 GiB",
                "600.0s per instance-engine pair",
                # ... the engine commit and solver versions ...
                "| engine commit | deadbee |",
                "ortools 9.15.6755",
                "PySCIPOpt 6.2.1",
                # ... and the yardstick: an upstream revision of the solution file
                # moves every gap in the table at once.
                "`miplib2017-v36.solu`",
                "9236602294c1a5ac",
            ],
            [],
        ),
        (_RECORD, 3, ["3 invocations"], []),
        # Silence is the failure mode: a published table whose concurrency nobody
        # recorded reads exactly like one whose concurrency was stated.
        (
            None,
            0,
            ["## 8. Machine and run record", "No machine record was written", "concurrency"],
            [],
        ),
        # Off Linux there is no /proc/meminfo and no sched_getaffinity, and
        # os.cpu_count() can be None. `| cores | None |` in a published table reads
        # as a scorer bug rather than as a fact about the machine.
        ({}, 1, ["| host | not recorded |", "| concurrency | not recorded |"], ["None"]),
        # The driver appends a record on every invocation, so confirming a finished
        # directory from a laptop would otherwise publish the laptop as the machine.
        (
            dict(_RECORD, outcome={"jobs_run": 0, "failures": 0, "rejected": 0}),
            2,
            ["This invocation ran no jobs"],
            [],
        ),
        (
            dict(_RECORD, outcome={"jobs_run": 26, "failures": 0, "rejected": 0}),
            1,
            [],
            ["This invocation ran no jobs"],
        ),
        # The all-or-nothing guard once fired only when all three were absent, and
        # a record carrying `jobs` alone published `large instances None at a time`.
        (dict(_RECORD, concurrency={"jobs": 4}), 1, ["**4 job(s) at a time**"], ["None"]),
        # `isinstance(kib, int)` discarded a float the run did measure ...
        (dict(_RECORD, machine={"memory_total_kib": 16777216.0}), 1, ["16.0 GiB"], []),
        # ... and `bool` is an `int` in Python, so `True` rendered as `0.0 GiB`.
        (dict(_RECORD, machine={"memory_total_kib": True}), 1, ["not recorded"], ["0.0 GiB"]),
        # --skip-preconditions is what makes a run unpublishable, and terminal
        # scrollback does not reach whoever reads the table.
        (
            dict(_RECORD, run={"preconditions_checked": False}),
            1,
            ["NOT CHECKED", "not publishable"],
            [],
        ),
        (
            dict(_RECORD, run={"preconditions_checked": True}),
            1,
            [],
            ["NOT CHECKED", "not recorded -- this record does not say either way"],
        ),
        # Three states, and `is False` alone fails open on the third: a record with
        # no such key rendered exactly like a properly-checked run.
        (
            dict(_RECORD, run={"jobs_planned": 26}),
            1,
            ["not recorded -- this record does not say either way"],
            ["NOT CHECKED"],
        ),
    ],
    ids=[
        "full-record",
        "resumed-directory",
        "no-record-is-an-anecdote",
        "unmeasured-fields",
        "resume-that-ran-nothing",
        "invocation-that-ran-jobs",
        "partial-concurrency",
        "memory-round-tripped-as-float",
        "boolean-memory",
        "preconditions-skipped",
        "preconditions-checked",
        "preconditions-unrecorded",
    ],
)
def test_the_record_section(
    record: dict[str, object] | None, invocations: int, said: list[str], not_said: list[str]
) -> None:
    section = "\n".join(run_record_section(record, invocations))
    for text in said:
        assert text in section, text
    for text in not_said:
        assert text not in section, text


@pytest.mark.parametrize(
    ("contents", "expected"),
    [
        (
            json.dumps({"runs": [{"status": "complete"}, {"status": "running"}]}),
            ({"status": "running"}, 2),
        ),
        # A driver killed mid-write must not make the scorer abort on the file.
        ('{"runs": [', (None, 0)),
        (None, (None, 0)),
    ],
    ids=["last-invocation-and-count", "truncated", "absent"],
)
def test_read_run_record(
    tmp_path: Path, contents: str | None, expected: tuple[dict[str, object] | None, int]
) -> None:
    if contents is not None:
        (tmp_path / "run_record.json").write_text(contents)
    assert read_run_record(tmp_path) == expected


def test_the_scorer_warns_when_no_machine_record_is_beside_the_results(tmp_path: Path) -> None:
    results = tmp_path / "results"
    for engine in ENGINES:
        _write(results, engine, "only-cbls", {"status": "no_solution", "objective": None})
    roster = tmp_path / "roster.csv"
    roster.write_text("instance,reference_value,reference_kind\nonly-cbls,1.0,opt\n")

    completed = subprocess.run(
        [
            sys.executable,
            str(SCORER),
            *("--results-dir", str(results), "--roster", str(roster)),
            *("--budget", "2", "--out", str(tmp_path / "out.csv")),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "no run_record.json" in completed.stderr
    assert "No machine record was written" in (tmp_path / "out_report.md").read_text()
