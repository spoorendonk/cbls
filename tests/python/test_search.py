"""Tests for C++ search via Python bindings."""

import pytest
import _cbls_core as cbls


def vid(handle):
    return -(handle + 1)


class TestSolver:
    def test_unconstrained(self):
        m = cbls.Model()
        x = m.float_var(-10, 10)
        y = m.float_var(-10, 10)
        two = m.constant(2)
        m.minimize(m.sum([m.pow_expr(x, two), m.pow_expr(y, two)]))
        m.close()
        result = cbls.solve(m, 2.0, 42)
        assert result.feasible
        assert result.objective < 1.0

    def test_constrained(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        neg1 = m.constant(-1.0)
        three = m.constant(3.0)
        m.add_constraint(m.sum([three, m.prod(neg1, x), m.prod(neg1, y)]))
        m.minimize(m.sum([x, y]))
        m.close()
        result = cbls.solve(m, 3.0, 42)
        assert result.feasible
        assert result.objective < 5.0

    def test_returns_result(self):
        m = cbls.Model()
        x = m.float_var(0, 1)
        m.minimize(m.sum([x]))
        m.close()
        result = cbls.solve(m, 0.5, 42)
        assert result.iterations > 0
        assert result.time_seconds > 0


class TestViolation:
    def test_feasible(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        neg5 = m.constant(-5.0)
        m.add_constraint(m.sum([x, neg5]))
        m.minimize(m.sum([x]))
        m.close()
        m.var_mut(vid(x)).value = 3.0
        cbls.full_evaluate(m)
        vm = cbls.ViolationManager(m)
        assert vm.total_violation() == 0.0
        assert vm.is_feasible()

    def test_infeasible(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        neg5 = m.constant(-5.0)
        m.add_constraint(m.sum([x, neg5]))
        m.minimize(m.sum([x]))
        m.close()
        m.var_mut(vid(x)).value = 8.0
        cbls.full_evaluate(m)
        vm = cbls.ViolationManager(m)
        assert vm.total_violation() == 3.0
        assert not vm.is_feasible()
