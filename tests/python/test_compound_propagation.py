"""`SearchConfig.compound_move_propagation` and its counters from Python (#340).

The C++ side (`tests/test_fj_compound_propagation.cpp`) pins what a propagated
leg does. What is left for here is what only the binding can get wrong: the
field missing, misnamed, defaulting on, or not reaching the solve, and the two
counters not coming back under their C++ names.
"""

import _cbls_core as cbls


def _tight() -> "cbls.Model":
    """Two equality rows over Ints in [0, 4]: any leg narrows its row's others."""
    m = cbls.Model()
    x = [m.int_var(0, 4) for _ in range(6)]
    m.add_constraint(m.eq_expr(m.sum(x[0:3]), m.constant(9.0)))
    m.add_constraint(m.eq_expr(m.sum(x[3:6]), m.constant(9.0)))
    m.add_constraint(m.leq(m.sum([x[0], x[3]]), m.constant(2.0)))
    m.minimize(m.sum([x[1], x[4]]))
    m.close()
    return m


def _config(propagate: bool) -> "cbls.SearchConfig":
    config = cbls.SearchConfig()
    config.max_iterations = 3000
    config.use_compound_moves = True
    config.novelty_jump_probability = 1.0
    config.compound_move_propagation = propagate
    return config


def test_compound_move_propagation_defaults_off() -> None:
    assert cbls.SearchConfig().compound_move_propagation is False


def test_compound_move_propagation_round_trips() -> None:
    config = cbls.SearchConfig()
    config.compound_move_propagation = True
    assert config.compound_move_propagation is True
    config.compound_move_propagation = False
    assert config.compound_move_propagation is False


def test_compound_move_propagation_reaches_the_solve() -> None:
    """On narrows Novelty legs and counts it; off never does."""
    on = cbls.solve(_tight(), 0.0, 42, config=_config(True)).counters
    off = cbls.solve(_tight(), 0.0, 42, config=_config(False)).counters
    assert on.novelty_batches > 0
    assert on.novelty_propagation_tightenings > 0
    # x0 + x3 <= 2 empties a domain whenever a leg sets x0 or x3 above 2, so
    # this model has conflicts too; that the two counts differ is what catches
    # one field bound under the other's name (1434 vs 136 at seed 42).
    assert on.novelty_propagation_conflicts > 0
    assert on.novelty_propagation_conflicts != on.novelty_propagation_tightenings
    assert (off.novelty_propagation_tightenings, off.novelty_propagation_conflicts) == (0, 0)
