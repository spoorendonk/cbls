"""Tests for C++ DAG via Python bindings."""

import math
import pytest
import _cbls_core as cbls


def vid(handle):
    """Get internal var ID from handle."""
    return -(handle + 1)


class TestBasicEvaluation:
    def test_sum(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        s = m.sum([x, y])
        m.minimize(s)
        m.close()
        m.var_mut(vid(x)).value = 3.0
        m.var_mut(vid(y)).value = 4.0
        cbls.full_evaluate(m)
        assert m.node(s).value == 7.0

    def test_prod(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        p = m.prod(x, y)
        m.minimize(p)
        m.close()
        m.var_mut(vid(x)).value = 3.0
        m.var_mut(vid(y)).value = 4.0
        cbls.full_evaluate(m)
        assert m.node(p).value == 12.0

    def test_pow(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        two = m.constant(2)
        p = m.pow_expr(x, two)
        m.minimize(p)
        m.close()
        m.var_mut(vid(x)).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(p).value == 9.0

    def test_sin(self):
        m = cbls.Model()
        x = m.float_var(-10, 10)
        s = m.sin_expr(x)
        m.minimize(s)
        m.close()
        m.var_mut(vid(x)).value = math.pi / 2
        cbls.full_evaluate(m)
        assert abs(m.node(s).value - 1.0) < 1e-10

    def test_nested(self):
        m = cbls.Model()
        x = m.float_var(-10, 10)
        y = m.float_var(-10, 10)
        two = m.constant(2)
        x_sq = m.pow_expr(x, two)
        xy = m.prod(x, y)
        two_xy = m.prod(two, xy)
        sin_y = m.sin_expr(y)
        f = m.sum([x_sq, two_xy, sin_y])
        m.minimize(f)
        m.close()
        m.var_mut(vid(x)).value = 2.0
        m.var_mut(vid(y)).value = 1.0
        cbls.full_evaluate(m)
        expected = 4.0 + 4.0 + math.sin(1.0)
        assert abs(m.node(f).value - expected) < 1e-10


class TestDeltaEvaluation:
    def test_delta_matches_full(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        z = m.float_var(0, 10)
        xy = m.prod(x, y)
        f = m.sum([xy, z])
        m.minimize(f)
        m.close()
        m.var_mut(vid(x)).value = 2.0
        m.var_mut(vid(y)).value = 3.0
        m.var_mut(vid(z)).value = 1.0
        cbls.full_evaluate(m)
        assert m.node(f).value == 7.0

        m.var_mut(vid(x)).value = 5.0
        delta_result = cbls.delta_evaluate(m, {vid(x)})
        assert delta_result == 16.0


class TestAD:
    def test_sum_partials(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        s = m.sum([x, y])
        m.minimize(s)
        m.close()
        m.var_mut(vid(x)).value = 3.0
        m.var_mut(vid(y)).value = 4.0
        cbls.full_evaluate(m)
        assert cbls.compute_partial(m, s, vid(x)) == 1.0
        assert cbls.compute_partial(m, s, vid(y)) == 1.0

    def test_chain_rule(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        two = m.constant(2)
        x2 = m.pow_expr(x, two)
        f = m.sin_expr(x2)
        m.minimize(f)
        m.close()
        m.var_mut(vid(x)).value = 1.5
        cbls.full_evaluate(m)
        expected = 2.0 * 1.5 * math.cos(1.5**2)
        assert abs(cbls.compute_partial(m, f, vid(x)) - expected) < 1e-10
