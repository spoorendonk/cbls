"""`SearchResult.counters` from Python (#169).

The C++ side (`tests/test_counters.cpp`) owns the identities and the portfolio
aggregation. What is left for here is what only a binding can get wrong: a field
that was never exposed, one exposed under a name that does not match the C++
one, and the per-generator rows, which are a bound `std::vector` of a bound
struct and so the part most likely to come back as an opaque handle.

Every run is iteration-budgeted with `time_limit = 0.0`, which is also the
regime in which `inner_solver_seconds` is deliberately 0.0 -- asserted below, so
a future change that starts reading a clock there fails a test rather than
quietly costing the determinism claim.
"""

import _cbls_core as cbls

ITERATIONS = 5000


def _quadratic() -> "cbls.Model":
    """x + y >= 1 over [-5, 5]^2, minimizing x^2 + y^2. No structured variable."""
    m = cbls.Model()
    x = m.float_var(-5, 5)
    y = m.float_var(-5, 5)
    two = m.constant(2)
    neg1 = m.constant(-1.0)
    one = m.constant(1.0)
    m.add_constraint(m.sum([one, m.prod(neg1, x), m.prod(neg1, y)]))
    m.minimize(m.sum([m.pow_expr(x, two), m.pow_expr(y, two)]))
    m.close()
    return m


def _structured() -> "cbls.Model":
    """A List and a Set, both consumed, so the structural batch has work to do.

    An unconsumed structured variable would leave the sweep with nothing to
    improve and prove nothing about the counters.
    """
    import numpy as np

    m = cbls.Model()
    route = m.list_var(8, "route")
    chosen = m.set_var(10, 3, 6, "chosen")
    cost = np.array([[float((i * 7 + j * 3) % 11) for j in range(8)] for i in range(8)])
    weights = np.array([float((e * 5) % 7) + 1.0 for e in range(10)])
    tour = m.pair_table_sum(route, cost, cyclic=True)
    load = m.lambda_table_sum(chosen, weights)
    m.add_constraint(m.sum([m.constant(-12.0), load]))  # load >= 12
    m.minimize(m.sum([tour, load]))
    m.close()
    return m


def _config() -> "cbls.SearchConfig":
    config = cbls.SearchConfig()
    config.max_iterations = ITERATIONS
    return config


def _structural_config() -> "cbls.SearchConfig":
    """Every batch structural, so the assertions below need no luck.

    `batch_iterations` is not bound, so the default 1000 GLS iterations per batch
    would give a 5000-iteration run about five batches and a 0.33 structural
    probability could take none of them. Forcing the probability to 1.0 instead
    makes the batch count the budget (a structural batch charges no GLS
    iteration, so `max_iterations` binds on batches -- see
    `budget_exhausted()`), which is both deterministic and quick.
    """
    config = cbls.SearchConfig()
    config.max_iterations = 300
    config.structural_batch_probability = 1.0
    return config


def test_batch_buckets_sum_to_the_batch_count() -> None:
    """The identity a reader of the breakdown relies on, read through the binding."""
    result = cbls.solve(_quadratic(), 0.0, 42, config=_config())
    counters = result.counters
    assert counters.batches > 0
    assert (
        counters.fj_batches + counters.novelty_batches + counters.structural_batches
        == counters.batches
    )
    # A scalar model runs feasibility-jump batches and nothing else.
    assert counters.fj_batches == counters.batches


def test_per_generator_rows_are_readable() -> None:
    """`by_generator` comes back as a list of named rows, not an opaque handle."""
    result = cbls.solve(_structured(), 0.0, 42, config=_structural_config())
    counters = result.counters
    assert counters.structural_batches == counters.batches > 0
    rows = list(counters.by_generator)
    assert len(rows) == 2, "one built-in generator per structured variable"
    for row in rows:
        assert isinstance(row.name, str)
        assert row.name
        assert row.moves_accepted <= row.moves_tried
    assert counters.structural_moves_tried == sum(r.moves_tried for r in rows)
    assert counters.structural_moves_accepted == sum(r.moves_accepted for r in rows)
    assert counters.structural_moves_tried > 0


def test_inner_solver_is_counted_but_not_timed_without_a_clock() -> None:
    """The gate from include/cbls/counters.h, asserted from the side that reads it."""
    result = cbls.solve(_quadratic(), 0.0, 42, hook=cbls.FloatIntensifyHook(), config=_config())
    assert result.counters.inner_solver_calls > 0
    assert result.counters.inner_solver_seconds == 0.0


def test_a_single_solve_reports_no_portfolio_restarts() -> None:
    """A solve cannot restart itself; only the portfolio's worker loop can."""
    result = cbls.solve(_quadratic(), 0.0, 42, config=_config())
    assert result.counters.portfolio_restarts == 0


def test_batch_kind_enum_is_bound() -> None:
    """The names a Python reader uses to talk about the buckets."""
    members = {
        cbls.BatchKind.FeasibilityJump,
        cbls.BatchKind.NoveltyJump,
        cbls.BatchKind.Structural,
    }
    assert len(members) == 3, "a duplicated enum value would collapse this set"


def test_counters_are_read_only() -> None:
    """They report a finished run; a writable field would be a way to falsify it."""
    import pytest

    result = cbls.solve(_quadratic(), 0.0, 42, config=_config())
    with pytest.raises(AttributeError):
        result.counters.batches = 0
