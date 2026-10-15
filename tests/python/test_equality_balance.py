"""`SearchConfig.fj_equality_balance` from Python (#222).

The C++ side (`tests/test_fj_equality_balance.cpp`) pins the step rule and the
move. What is left for here is what only the binding can get wrong: the field
missing, misnamed, defaulting on, or not reaching the solve it is handed to.
"""

import _cbls_core as cbls


def vid(handle: int) -> int:
    return -(handle + 1)


def _model() -> "cbls.Model":
    """Twenty copies of 20x + 20y == 200 at (5, 5) with x^2 >= 64: x alone
    gains 39 on the square and loses 60 on the equality."""
    m = cbls.Model()
    for _ in range(20):
        x = m.int_var(0, 10)
        y = m.int_var(0, 10)
        m.add_constraint(
            m.eq_expr(
                m.sum([m.prod(m.constant(20.0), x), m.prod(m.constant(20.0), y)]),
                m.constant(200.0),
            )
        )
        m.add_constraint(m.geq(m.prod(x, x), m.constant(64.0)))
        m.var_mut(vid(x)).value = 5.0
        m.var_mut(vid(y)).value = 5.0
    m.close()
    return m


def test_equality_balance_defaults_off() -> None:
    assert cbls.SearchConfig().fj_equality_balance is False


def test_equality_balance_reaches_the_solve() -> None:
    """On, each copy's minimum is left by one balance move instead of a bump
    and two jumps, so the solve takes fewer iterations."""
    runs = {}
    for on in (False, True):
        config = cbls.SearchConfig()
        config.max_iterations = 20_000
        config.skip_init = True
        config.use_compound_moves = False
        config.fj_equality_balance = on
        runs[on] = cbls.solve(_model(), 0.0, 1, config=config)
    assert runs[False].feasible
    assert runs[True].feasible
    assert runs[True].iterations < runs[False].iterations
