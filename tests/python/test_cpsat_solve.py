"""Tests for the CP-SAT baseline's log parsing.

The CP-SAT incumbent trace exists only because a regex matches lines of a debug log
format that carries no stability guarantee. If an OR-Tools release changes it,
`parse_trace` silently returns nothing, every CP-SAT instance scores a Primal
Integral near 2, and the aggregate reads as "CP-SAT is bad" rather than as a broken
harness — after a 39 CPU-hour run. These are the golden lines that pin it down.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ortools", reason="ortools is in the 'benchmarks' extra, not 'dev'")

from benchmarks.mipfeas.cpsat_solve import build_parameters, parse_trace  # noqa: E402

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
