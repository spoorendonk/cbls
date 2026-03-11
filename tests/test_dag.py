"""Tests for Expression DAG: construction, evaluation, delta evaluation, AD."""

import math

import pytest

from cbls.model import Model
from cbls.dag import (
    BoolVar, IntVar, FloatVar, ListVar, SetVar,
    full_evaluate, delta_evaluate, compute_partial, topological_sort,
)


class TestBasicEvaluation:
    """Test full evaluation of all operator types."""

    def test_sum(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        s = m.sum(x, y)
        m.minimize(s)
        m.close()
        x.value = 3.0
        y.value = 4.0
        full_evaluate(m)
        assert s.value == 7.0

    def test_prod(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        p = m.prod(x, y)
        m.minimize(p)
        m.close()
        x.value = 3.0
        y.value = 4.0
        full_evaluate(m)
        assert p.value == 12.0

    def test_div(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(1, 10)
        d = m.div(x, y)
        m.minimize(d)
        m.close()
        x.value = 6.0
        y.value = 3.0
        full_evaluate(m)
        assert d.value == 2.0

    def test_pow(self):
        m = Model()
        x = m.float_var(0, 10)
        p = m.pow(x, 2)
        m.minimize(p)
        m.close()
        x.value = 3.0
        full_evaluate(m)
        assert p.value == 9.0

    def test_sin(self):
        m = Model()
        x = m.float_var(-10, 10)
        s = m.sin(x)
        m.minimize(s)
        m.close()
        x.value = math.pi / 2
        full_evaluate(m)
        assert abs(s.value - 1.0) < 1e-10

    def test_cos(self):
        m = Model()
        x = m.float_var(-10, 10)
        c = m.cos(x)
        m.minimize(c)
        m.close()
        x.value = 0.0
        full_evaluate(m)
        assert abs(c.value - 1.0) < 1e-10

    def test_abs(self):
        m = Model()
        x = m.float_var(-10, 10)
        a = m.abs(x)
        m.minimize(a)
        m.close()
        x.value = -5.0
        full_evaluate(m)
        assert a.value == 5.0

    def test_min_max(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        mn = m.min(x, y)
        mx = m.max(x, y)
        total = m.sum(mn, mx)
        m.minimize(total)
        m.close()
        x.value = 3.0
        y.value = 7.0
        full_evaluate(m)
        assert mn.value == 3.0
        assert mx.value == 7.0

    def test_neg(self):
        m = Model()
        x = m.float_var(0, 10)
        n = m.neg(x)
        m.minimize(n)
        m.close()
        x.value = 5.0
        full_evaluate(m)
        assert n.value == -5.0

    def test_if_then_else(self):
        m = Model()
        x = m.float_var(-10, 10)
        y = m.float_var(0, 10)
        z = m.float_var(0, 10)
        ite = m.if_then_else(x, y, z)
        m.minimize(ite)
        m.close()

        x.value = 1.0  # cond > 0 → then branch
        y.value = 5.0
        z.value = 9.0
        full_evaluate(m)
        assert ite.value == 5.0

        x.value = -1.0  # cond ≤ 0 → else branch
        full_evaluate(m)
        assert ite.value == 9.0

    def test_constants(self):
        m = Model()
        x = m.float_var(0, 10)
        expr = m.sum(x, 5.0)
        m.minimize(expr)
        m.close()
        x.value = 3.0
        full_evaluate(m)
        assert expr.value == 8.0

    def test_nested_expression(self):
        """f(x,y) = x^2 + 2*x*y + sin(y)"""
        m = Model()
        x = m.float_var(-10, 10)
        y = m.float_var(-10, 10)
        x_sq = m.pow(x, 2)
        xy = m.prod(x, y)
        two_xy = m.prod(2, xy)
        sin_y = m.sin(y)
        f = m.sum(x_sq, two_xy, sin_y)
        m.minimize(f)
        m.close()

        x.value = 2.0
        y.value = 1.0
        full_evaluate(m)
        expected = 4.0 + 4.0 + math.sin(1.0)
        assert abs(f.value - expected) < 1e-10


class TestDeltaEvaluation:
    """Test incremental delta evaluation matches full evaluation."""

    def test_delta_matches_full(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        z = m.float_var(0, 10)
        xy = m.prod(x, y)
        f = m.sum(xy, z)
        m.minimize(f)
        m.close()

        x.value = 2.0
        y.value = 3.0
        z.value = 1.0
        full_evaluate(m)
        assert f.value == 7.0

        # Change only x
        x.value = 5.0
        delta_result = delta_evaluate(m, {x})
        assert delta_result == 16.0  # 5*3 + 1

        # Verify full eval gives same result
        full_result = full_evaluate(m)
        assert full_result == delta_result

    def test_delta_unchanged_vars(self):
        """Delta eval with no changes should return same value."""
        m = Model()
        x = m.float_var(0, 10)
        f = m.pow(x, 2)
        m.minimize(f)
        m.close()

        x.value = 4.0
        full_evaluate(m)

        # Delta with empty set should not change
        result = delta_evaluate(m, set())
        assert result == 16.0

    def test_delta_multiple_vars(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        f = m.sum(m.pow(x, 2), m.pow(y, 2))
        m.minimize(f)
        m.close()

        x.value = 3.0
        y.value = 4.0
        full_evaluate(m)
        assert f.value == 25.0

        x.value = 1.0
        y.value = 2.0
        result = delta_evaluate(m, {x, y})
        assert result == 5.0


class TestListVar:
    """Test ListVar with at() and lambda_sum()."""

    def test_at(self):
        m = Model()
        lv = m.list_var(5)
        idx_node = m.const(2)
        a = m.at(lv, idx_node)
        m.minimize(a)
        m.close()

        lv.elements = [10, 20, 30, 40, 50]
        full_evaluate(m)
        assert a.value == 30.0

    def test_lambda_sum(self):
        m = Model()
        lv = m.list_var(4)
        ls = m.lambda_sum(lv, lambda e: e * e)
        m.minimize(ls)
        m.close()

        lv.elements = [1, 2, 3, 4]
        full_evaluate(m)
        assert ls.value == 30.0  # 1+4+9+16

    def test_list_delta_eval(self):
        m = Model()
        lv = m.list_var(3)
        ls = m.lambda_sum(lv, lambda e: e)
        m.minimize(ls)
        m.close()

        lv.elements = [0, 1, 2]
        full_evaluate(m)
        assert ls.value == 3.0

        lv.elements = [2, 1, 0]
        result = delta_evaluate(m, {lv})
        assert result == 3.0  # sum unchanged for permutation


class TestSetVar:
    """Test SetVar with count()."""

    def test_count(self):
        m = Model()
        sv = m.set_var(10, min_size=0, max_size=10)
        c = m.count(sv)
        m.minimize(c)
        m.close()

        sv.elements = {1, 3, 5, 7}
        full_evaluate(m)
        assert c.value == 4.0


class TestAutomaticDifferentiation:
    """Test reverse-mode AD on DAG."""

    def test_sum_partials(self):
        """∂(x+y)/∂x = 1, ∂(x+y)/∂y = 1"""
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        s = m.sum(x, y)
        m.minimize(s)
        m.close()
        x.value = 3.0
        y.value = 4.0
        full_evaluate(m)

        assert compute_partial(m, s, x) == 1.0
        assert compute_partial(m, s, y) == 1.0

    def test_prod_partials(self):
        """∂(x*y)/∂x = y, ∂(x*y)/∂y = x"""
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        p = m.prod(x, y)
        m.minimize(p)
        m.close()
        x.value = 3.0
        y.value = 4.0
        full_evaluate(m)

        assert compute_partial(m, p, x) == 4.0
        assert compute_partial(m, p, y) == 3.0

    def test_pow_partial(self):
        """∂(x^2)/∂x = 2x"""
        m = Model()
        x = m.float_var(0, 10)
        p = m.pow(x, 2)
        m.minimize(p)
        m.close()
        x.value = 3.0
        full_evaluate(m)

        assert abs(compute_partial(m, p, x) - 6.0) < 1e-10

    def test_sin_partial(self):
        """∂sin(x)/∂x = cos(x)"""
        m = Model()
        x = m.float_var(-10, 10)
        s = m.sin(x)
        m.minimize(s)
        m.close()
        x.value = 1.0
        full_evaluate(m)

        assert abs(compute_partial(m, s, x) - math.cos(1.0)) < 1e-10

    def test_chain_rule(self):
        """∂sin(x^2)/∂x = 2x * cos(x^2)"""
        m = Model()
        x = m.float_var(0, 10)
        x2 = m.pow(x, 2)
        f = m.sin(x2)
        m.minimize(f)
        m.close()
        x.value = 1.5
        full_evaluate(m)

        expected = 2 * 1.5 * math.cos(1.5 ** 2)
        assert abs(compute_partial(m, f, x) - expected) < 1e-10

    def test_composite_expression(self):
        """∂(x^2 + 2*x*y)/∂x = 2x + 2y"""
        m = Model()
        x = m.float_var(-10, 10)
        y = m.float_var(-10, 10)
        x_sq = m.pow(x, 2)
        xy = m.prod(x, y)
        two_xy = m.prod(2, xy)
        f = m.sum(x_sq, two_xy)
        m.minimize(f)
        m.close()

        x.value = 3.0
        y.value = 2.0
        full_evaluate(m)

        expected = 2 * 3.0 + 2 * 2.0  # 10
        assert abs(compute_partial(m, f, x) - expected) < 1e-10
