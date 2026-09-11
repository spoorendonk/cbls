"""Tests for the CP-SAT baseline's log parsing.

The CP-SAT incumbent trace exists only because a regex matches lines of a debug log
format that carries no stability guarantee. If an OR-Tools release changes it,
`parse_trace` silently returns nothing, every CP-SAT instance scores a Primal
Integral near 2, and the aggregate reads as "CP-SAT is bad" rather than as a broken
harness — after a 39 CPU-hour run. These are the golden lines that pin it down.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("ortools", reason="ortools is in the 'benchmarks' extra, not 'dev'")

from benchmarks.mipfeas.cpsat_solve import (  # noqa: E402
    LOG_FORMAT_CHECK,
    PREFLIGHT_COLUMNS,
    PREFLIGHT_ROWS,
    SUPPORTED_ORTOOLS_RANGE,
    WORKER_RESTRICTION_CHECK,
    PreflightFailure,
    build_parameters,
    build_preflight_model,
    check_preflight_log,
    parse_subsolvers,
    parse_trace,
    report_preflight,
    run_preflight,
    status_note,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

# Verbatim from an ortools 9.15 run of this harness.
GOLDEN_LOG = """\
Starting CP-SAT solver v9.15.6755
1 first solution subsolver: [fj]
1 interleaved subsolver: [ls]
#1       0.01s best:69     next:[0,68]     fj_restart(batch:1 lin{mvs:43 evals:287})
#2       3.30s best:6908.97 next:[5726.32999,6908.97] ls_restart_decay_compound(batch:1)
#3       8.25s best:-4734.18129325 next:[0,1] ls_restart_perturb(batch:1 lin{mvs:126})
#4      10.00s best:2.2000185e+09 next:[0,1] ls_restart(batch:1)
#Bound  12.00s best:inf   next:[0,10]     objective_shaving
#Model  12.10s var:1458/1500 constraints:900/1000
#Variables: 1'458
CpSolverResponse summary:
status: FEASIBLE
"""


def test_parse_trace_extracts_each_improving_solution() -> None:
    assert parse_trace(GOLDEN_LOG) == [
        (0.01, 69.0),
        (3.30, 6908.97),
        (8.25, -4734.18129325),
        (10.00, 2.2000185e09),
    ]


def test_parse_trace_ignores_bound_and_model_lines() -> None:
    # `#Bound ... best:inf` would otherwise enter the trace as an infinity and turn
    # every aggregate into NaN; `#Variables: 1'458` is the thousands-separator trap.
    times = [t for t, _ in parse_trace(GOLDEN_LOG)]
    assert 12.00 not in times
    assert 12.10 not in times


def test_parse_trace_of_a_log_without_solutions_is_empty() -> None:
    assert parse_trace("Starting CP-SAT solver\nstatus: INFEASIBLE\n") == []


def test_parse_trace_handles_negative_and_scientific_objectives() -> None:
    values = [obj for _, obj in parse_trace(GOLDEN_LOG)]
    assert -4734.18129325 in values
    assert 2.2000185e09 in values


def test_build_parameters_restricts_the_solve_to_the_fj_and_ls_workers() -> None:
    # `filter_subsolvers` is the only parameter that accepts these names, and `ls`
    # without `fj` never bootstraps a first solution — so both must be present.
    params = build_parameters(workers=1, seed=42)
    assert "filter_subsolvers:'fj'" in params
    assert "filter_subsolvers:'ls'" in params
    assert "num_violation_ls:1" in params
    assert "log_search_progress:true" in params


def test_build_parameters_does_not_set_the_time_limit() -> None:
    # set_time_limit_in_seconds already populates max_time_in_seconds and this
    # string merges on top of it; stating the budget twice invites the two to drift.
    assert "max_time_in_seconds" not in build_parameters(workers=1, seed=42)


# --- Preflight ----------------------------------------------------------------
#
# The baseline's whole configuration is empirical against one OR-Tools release,
# while the parsing runs against whatever is installed. These pin the two things
# that can break silently (issue #137) and, crucially, that the failure names
# which one -- so they are driven from captured log text rather than from a
# solve, which is what lets a broken release be tested without installing it.

#: The subsolver announcement and search header a correctly restricted run emits.
RESTRICTED_HEADER = """\
Starting search at 0.00s with 1 workers.
1 first solution subsolver: [fj]
1 interleaved subsolver: [ls]
3 helper subsolvers: [neighborhood_helper, synchronization_agent, update_gap_integral]
30 ignored subsolvers: [core, default_lp, fixed, fj_lin, ls_lin, no_lp, probing]
"""

#: One improving-solution line, the shape the incumbent trace is recovered from.
IMPROVING_LINE = "#1       0.00s best:181   next:[0,180]    fj_restart_compound(batch:1)\n"

GOOD_PREFLIGHT_LOG = RESTRICTED_HEADER + IMPROVING_LINE


def _checks(failures: list[PreflightFailure]) -> set[str]:
    return {failure.check for failure in failures}


def test_preflight_passes_a_log_from_the_release_it_was_written_against() -> None:
    assert (
        check_preflight_log(GOOD_PREFLIGHT_LOG, status="FEASIBLE", workers=1, found_solution=True)
        == []
    )


def test_parse_subsolvers_strips_the_multiplicity_a_second_worker_adds() -> None:
    # `num_workers: 2` announces `fj(2)` / `ls(2)`; the restriction is unchanged.
    announced = parse_subsolvers(
        "2 first solution subsolver: [fj(2)]\n2 interleaved subsolver: [ls(2)]\n"
    )
    assert announced["first solution"] == frozenset({"fj"})
    assert announced["interleaved"] == frozenset({"ls"})


def test_a_release_that_lost_the_worker_restriction_fails_and_says_so() -> None:
    # `filter_subsolvers` renamed: CP-SAT runs its default portfolio instead, which
    # is a different (and rejected) baseline — epic #87.
    log = GOOD_PREFLIGHT_LOG.replace(
        "1 first solution subsolver: [fj]",
        "4 first solution subsolvers: [fj, default_lp, no_lp, quick_restart]",
    )
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}
    assert "default_lp" in failures[0].message


def test_a_rejected_parameter_string_is_a_worker_restriction_failure() -> None:
    failures = check_preflight_log(
        "", status="INVALID_SOLVER_PARAMETERS", workers=1, found_solution=False
    )

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}
    assert "filter_subsolvers" in failures[0].message


def test_a_release_that_reformatted_the_solution_line_fails_as_a_log_format_break() -> None:
    # The line still exists, in a shape the trace regex no longer matches. Nothing
    # crashes: every CP-SAT row would score ~2.0 and read as "CP-SAT is bad".
    log = RESTRICTED_HEADER + "solution 1 at 0.00s objective 181\n"
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {LOG_FORMAT_CHECK}
    assert "improving-solution line" in failures[0].message


def test_the_full_portfolio_running_alongside_fj_and_ls_is_a_restriction_failure() -> None:
    # An unrestricted CP-SAT run announces `full problem subsolvers`, not `full`.
    # A role whitelist spelling it the short way never matches the line at all, so
    # the preflight passed clean while the baseline was CP-SAT's default portfolio
    # -- the comparison epic #87 rejects -- across the whole roster.
    log = (
        RESTRICTED_HEADER.replace(
            "3 helper subsolvers",
            "8 full problem subsolvers: [default_lp, no_lp, max_lp, core]\n3 helper subsolvers",
        )
        + IMPROVING_LINE
    )
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}
    assert "default_lp" in failures[0].message


def test_a_role_this_harness_has_never_seen_is_not_waved_through() -> None:
    # The general form of the case above: any announced role outside the fj + ls
    # pairing and the bookkeeping ones means workers are running that should not be.
    log = RESTRICTED_HEADER + "5 background subsolvers: [default_lp, core]\n" + IMPROVING_LINE
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}


def test_a_dropped_restriction_is_not_reported_as_a_log_format_break() -> None:
    # With `filter_subsolvers` gone, CP-SAT logs `1 full problem subsolver: [main]`
    # and neither of the two roles this harness expects. The log format is intact;
    # naming it would send the reader to check the parser instead of the parameter.
    log = (
        "Starting search at 0.00s with 1 workers.\n"
        "1 full problem subsolver: [main]\n"
        "3 helper subsolvers: [neighborhood_helper, synchronization_agent]\n"
    ) + IMPROVING_LINE
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}


def test_a_missing_subsolver_announcement_is_a_log_format_break() -> None:
    # Shape versus content: a line that is gone is the log format moving, and the
    # restriction simply cannot be read -- which is reported as such rather than
    # guessed at.
    log = IMPROVING_LINE + "Starting search at 0.00s with 1 workers.\n"
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {LOG_FORMAT_CHECK}
    assert len(failures) == 2, "both announcement lines are missing"


def test_both_breaks_at_once_are_both_named() -> None:
    log = "1 first solution subsolvers: [fj, default_lp]\n1 interleaved subsolver: [ls]\n"
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK, LOG_FORMAT_CHECK}


def test_a_thread_count_the_solver_did_not_honour_is_a_restriction_failure() -> None:
    # The baseline getting more CPU than CBLS in the same wall clock is not a
    # comparison, and `num_workers` is the parameter that would have moved.
    log = GOOD_PREFLIGHT_LOG.replace("with 1 workers", "with 8 workers")
    failures = check_preflight_log(log, status="FEASIBLE", workers=1, found_solution=True)

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}
    assert "8 workers" in failures[0].message


def test_a_restricted_configuration_that_cannot_search_fails() -> None:
    # `ls` without `fj` never bootstraps a first solution; so would a release in
    # which the filter leaves nothing runnable.
    failures = check_preflight_log(
        RESTRICTED_HEADER, status="UNKNOWN", workers=1, found_solution=False
    )

    assert _checks(failures) == {WORKER_RESTRICTION_CHECK}


def test_report_preflight_exits_non_zero_and_names_the_broken_check(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = report_preflight([PreflightFailure(LOG_FORMAT_CHECK, "the line moved")], workers=1)

    assert code == 3
    captured = capsys.readouterr()
    assert LOG_FORMAT_CHECK in captured.err
    assert "the line moved" in captured.err


def test_report_preflight_exits_zero_when_nothing_is_broken(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert report_preflight([], workers=1) == 0
    assert "preflight OK" in capsys.readouterr().out


def test_the_installed_ortools_passes_the_preflight_it_documents() -> None:
    # The whole point: run it against whatever is actually installed. One tiny
    # in-memory model, no instance file and no network.
    assert run_preflight() == []


def test_the_preflight_model_is_the_same_one_on_every_machine() -> None:
    # A check whose input drifts cannot say whether the solver moved.
    first, second = build_preflight_model(), build_preflight_model()
    assert first.num_variables == second.num_variables == PREFLIGHT_COLUMNS
    assert first.num_constraints == second.num_constraints == PREFLIGHT_ROWS


def test_the_documented_ortools_range_matches_the_declared_dependency() -> None:
    # The bound and the constant say the same thing, or the preflight's failure
    # message tells a reader to install a version the project would not resolve.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    assert f'"ortools{SUPPORTED_ORTOOLS_RANGE}"' in pyproject


# --- A withheld baseline row says why -----------------------------------------
#
# The scorer publishes one message per withheld row, and "no message recorded" is
# the least useful form of the most important defect class this baseline has. It
# is also the class the preflight exists to catch before a roster rather than
# after one.


def test_a_rejected_parameter_string_records_why_the_row_is_withheld() -> None:
    note = status_note("INVALID_SOLVER_PARAMETERS", False, build_parameters(1, 42))

    assert note is not None
    status, message = note
    assert status == "invalid_parameters"
    assert "filter_subsolvers" in message
    assert "--preflight" in message


def test_a_model_cp_sat_cannot_express_records_why_the_row_is_withheld() -> None:
    note = status_note("MODEL_INVALID", False, "")

    assert note is not None
    status, message = note
    assert status == "invalid_model"
    assert "MODEL_INVALID" in message


def test_a_solver_that_errored_out_is_reported_apart_from_finding_nothing() -> None:
    assert status_note("ABNORMAL", False, "")[0] == "invalid_model"  # type: ignore[index]


def test_a_time_limited_run_that_found_nothing_is_not_a_harness_fault() -> None:
    # NOT_SOLVED is the ordinary outcome the metric is asking about, so it stays
    # `no_solution` and carries no message.
    assert status_note("NOT_SOLVED", False, "") is None


def test_a_model_invalid_verdict_that_still_produced_a_solution_is_not_withheld() -> None:
    assert status_note("MODEL_INVALID", True, "") is None
