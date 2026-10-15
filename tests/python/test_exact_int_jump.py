"""`SearchConfig.fj_exact_int_jump` from Python (#216).

The C++ side (`tests/test_feasibility_jump.cpp`) pins #208's repro under both
settings, through solve() and the LNS repair. What is left for here is what only
the binding can get wrong: the field missing, misnamed, defaulting on, or not
reaching the solve it is handed to.
"""

import _cbls_core as cbls


def _repro() -> "cbls.Model":
    """#208's repro: x + y == 1e7 - 1 and |x - y| <= 1 as two linear rows."""
    m = cbls.Model()
    x = m.int_var(0, 10_000_000)
    y = m.int_var(0, 10_000_000)
    m.add_constraint(m.eq_expr(m.sum([x, y]), m.constant(9_999_999.0)))
    m.add_constraint(m.leq(m.sum([x, m.neg(y)]), m.constant(1.0)))
    m.add_constraint(m.leq(m.sum([y, m.neg(x)]), m.constant(1.0)))
    m.close()
    return m


def test_exact_int_jump_defaults_off() -> None:
    assert cbls.SearchConfig().fj_exact_int_jump is False


def test_exact_int_jump_reaches_the_solve() -> None:
    """Off is main's 72-iteration solve at seed 1; on is #208's known regression,
    not feasible inside the same 10k-iteration cap."""
    runs = {}
    for exact in (False, True):
        config = cbls.SearchConfig()
        config.max_iterations = 10_000
        config.fj_exact_int_jump = exact
        runs[exact] = cbls.solve(_repro(), 0.0, 1, config=config)
    assert runs[False].feasible
    assert runs[False].iterations == 72
    assert not runs[True].feasible
