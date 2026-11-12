"""DAG entry points that are bound as free functions, from Python.

`full_evaluate`, `delta_evaluate` (whose Python set crosses a caster of its own)
and `compute_partial` are reachable and hand back the value they computed, not a
neighbouring one. DAG semantics belong to `tests/test_dag.cpp`.
"""

import math

import _cbls_core as cbls
from support import vid


def test_delta_evaluate_takes_a_set_of_var_ids_and_returns_the_objective() -> None:
    # The set crosses a caster to C++'s pointer and count. An empty arrival
    # returns the objective unchanged, so the value is checked after a change
    # that must move it.
    m = cbls.Model()
    x = m.float_var(0, 10)
    y = m.float_var(0, 10)
    z = m.float_var(0, 10)
    f = m.sum([m.prod(x, y), z])
    m.minimize(f)
    m.close()
    m.var_mut(vid(x)).value = 2.0
    m.var_mut(vid(y)).value = 3.0
    m.var_mut(vid(z)).value = 1.0
    assert cbls.full_evaluate(m) == 7.0

    m.var_mut(vid(x)).value = 5.0
    assert cbls.delta_evaluate(m, {vid(x)}) == 16.0


def test_compute_partial_returns_the_derivative_it_was_asked_for() -> None:
    # Two variables and a chain, so a binding that ignored its `var_id` argument
    # or returned the wrong node's adjoint would be caught rather than
    # coincidentally right.
    m = cbls.Model()
    x = m.float_var(0, 10)
    y = m.float_var(0, 10)
    scaled_y = m.prod(m.constant(3.0), y)
    s = m.sum([x, scaled_y])
    m.minimize(s)
    m.close()
    m.var_mut(vid(x)).value = 3.0
    m.var_mut(vid(y)).value = 4.0
    cbls.full_evaluate(m)
    assert cbls.compute_partial(m, s, vid(x)) == 1.0
    assert cbls.compute_partial(m, s, vid(y)) == 3.0
    # A non-objective expr_id whose partial differs, so a binding that dropped
    # the argument is not coincidentally right.
    assert cbls.compute_partial(m, scaled_y, vid(x)) == 0.0

    m2 = cbls.Model()
    x2 = m2.float_var(0, 10)
    f = m2.sin_expr(m2.pow_expr(x2, m2.constant(2)))
    m2.minimize(f)
    m2.close()
    m2.var_mut(vid(x2)).value = 1.5
    cbls.full_evaluate(m2)
    assert abs(cbls.compute_partial(m2, f, vid(x2)) - 2.0 * 1.5 * math.cos(1.5**2)) < 1e-10
