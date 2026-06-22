"""Tests for the no-commit weighted-violation-delta API via Python bindings."""

import _cbls_core as cbls


def vid(handle):
    return -(handle + 1)


def _naive_total(m, vm, var_id, j):
    """Weighted total violation after setting var_id <- j, then restore."""
    old = m.var(var_id).value
    m.var_mut(var_id).value = j
    cbls.full_evaluate(m)
    vm.invalidate_cache()
    total = vm.total_violation()
    m.var_mut(var_id).value = old
    cbls.full_evaluate(m)
    vm.invalidate_cache()
    return total


def test_per_constraint_delta_roundtrips_to_list_of_tuples():
    m = cbls.Model()
    x = m.float_var(0, 10)
    m.add_constraint(m.leq(x, m.constant(2.0)))
    m.add_constraint(m.leq(x, m.constant(4.0)))
    m.minimize(m.sum([x]))
    m.close()
    m.var_mut(vid(x)).value = 1.0

    pcd = m.per_constraint_violation_delta(vid(x), 6.0)
    assert isinstance(pcd, list)
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pcd)
    # 6 > 4 > 2 -> both constraints become violated.
    assert {ci for ci, _ in pcd} == {0, 1}


def test_weighted_delta_matches_naive_recompute():
    m = cbls.Model()
    x = m.float_var(1, 5)
    y = m.float_var(1, 5)
    m.add_constraint(m.leq(m.prod(x, y), m.constant(6.0)))
    m.add_constraint(m.leq(m.sum([x, y, m.constant(-3.0)]), m.constant(0.0)))
    m.minimize(m.sum([x, y]))
    m.close()
    m.var_mut(vid(x)).value = 2.0
    m.var_mut(vid(y)).value = 2.0

    vm = cbls.ViolationManager(m)
    cbls.full_evaluate(m)
    vm.invalidate_cache()
    base = vm.total_violation()

    for j in (1.0, 3.0, 5.0):
        expected = _naive_total(m, vm, vid(x), j) - base
        assert abs(vm.weighted_violation_delta(vid(x), j) - expected) < 1e-9

    # No-commit invariant.
    vm.invalidate_cache()
    assert abs(vm.total_violation() - base) < 1e-12


def test_weights_are_applied():
    m = cbls.Model()
    x = m.float_var(0, 10)
    m.add_constraint(m.leq(x, m.constant(2.0)))
    m.add_constraint(m.leq(x, m.constant(4.0)))
    m.minimize(m.sum([x]))
    m.close()
    m.var_mut(vid(x)).value = 1.0

    vm = cbls.ViolationManager(m)
    # nanobind returns a list copy on read, so set the whole vector at once.
    vm.weights = [3.0, 5.0]
    vm.invalidate_cache()

    weights = vm.weights
    pcd = m.per_constraint_violation_delta(vid(x), 6.0)
    expected = sum(weights[ci] * d for ci, d in pcd)
    assert abs(vm.weighted_violation_delta(vid(x), 6.0) - expected) < 1e-12
