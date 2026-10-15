"""`SearchConfig.fj_quadratic_roots` from Python (#221).

The C++ side (`tests/test_fj_quadratic_roots.cpp`) pins the root formula, the
row walk and the candidates. What is left for here is what only the binding can
get wrong: the field missing, misnamed, defaulting on, or not reaching the solve
it is handed to.
"""

import _cbls_core as cbls


def _model() -> "cbls.Model":
    """x^2 == 49 over an Int domain too wide to enumerate."""
    m = cbls.Model()
    x = m.int_var(-1000, 1000)
    m.add_constraint(m.eq_expr(m.prod(x, x), m.constant(49.0)))
    m.close()
    return m


def test_quadratic_roots_defaults_off() -> None:
    assert cbls.SearchConfig().fj_quadratic_roots is False


def test_quadratic_roots_reaches_the_solve() -> None:
    """Off, FJ walks to the root by +-1 steps; on, it jumps there."""
    runs = {}
    for on in (False, True):
        config = cbls.SearchConfig()
        config.max_iterations = 10_000
        config.fj_quadratic_roots = on
        runs[on] = cbls.solve(_model(), 0.0, 1, config=config)
    assert runs[False].feasible
    assert runs[True].feasible
    assert runs[True].iterations < runs[False].iterations
