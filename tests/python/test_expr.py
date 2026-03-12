"""Tests for Expr wrapper and operator overloading via Python bindings."""

import math
import pytest
import _cbls_core as cbls


def vid(handle):
    return -(handle + 1)


class TestNewOps:
    """Test the 8 new operators via int32_t API."""

    def test_tan(self):
        m = cbls.Model()
        x = m.float_var(-1, 1)
        t = m.tan_expr(x)
        m.minimize(t)
        m.close()
        m.var_mut(vid(x)).value = 0.5
        cbls.full_evaluate(m)
        assert abs(m.node(t).value - math.tan(0.5)) < 1e-10

    def test_exp(self):
        m = cbls.Model()
        x = m.float_var(-10, 10)
        e = m.exp_expr(x)
        m.minimize(e)
        m.close()
        m.var_mut(vid(x)).value = 1.0
        cbls.full_evaluate(m)
        assert abs(m.node(e).value - math.exp(1.0)) < 1e-10

    def test_log(self):
        m = cbls.Model()
        x = m.float_var(0.01, 10)
        l = m.log_expr(x)
        m.minimize(l)
        m.close()
        m.var_mut(vid(x)).value = math.e
        cbls.full_evaluate(m)
        assert abs(m.node(l).value - 1.0) < 1e-10

    def test_sqrt(self):
        m = cbls.Model()
        x = m.float_var(0, 100)
        s = m.sqrt_expr(x)
        m.minimize(s)
        m.close()
        m.var_mut(vid(x)).value = 9.0
        cbls.full_evaluate(m)
        assert abs(m.node(s).value - 3.0) < 1e-10

    def test_geq(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        g = m.geq(x, y)
        m.minimize(m.abs_expr(g))
        m.close()
        m.var_mut(vid(x)).value = 5.0
        m.var_mut(vid(y)).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(g).value <= 0.0

    def test_neq(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        n = m.neq(x, y)
        m.minimize(n)
        m.close()
        m.var_mut(vid(x)).value = 3.0
        m.var_mut(vid(y)).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(n).value == 1.0  # violated (equal)
        m.var_mut(vid(x)).value = 3.0
        m.var_mut(vid(y)).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(n).value == 0.0  # satisfied

    def test_lt(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        l = m.lt(x, y)
        m.minimize(m.abs_expr(l))
        m.close()
        m.var_mut(vid(x)).value = 2.0
        m.var_mut(vid(y)).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(l).value < 0.0  # satisfied

    def test_gt(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        g = m.gt(x, y)
        m.minimize(m.abs_expr(g))
        m.close()
        m.var_mut(vid(x)).value = 7.0
        m.var_mut(vid(y)).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(g).value < 0.0  # satisfied


class TestExprArithmetic:
    """Test Expr operator overloading."""

    def test_add(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        f = x + y
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        m.var_mut(y.var_id()).value = 4.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 7.0

    def test_sub(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        f = x - y
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 7.0
        m.var_mut(y.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - 4.0) < 1e-10

    def test_mul(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        f = x * y
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        m.var_mut(y.var_id()).value = 4.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 12.0

    def test_div(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(1, 10)
        f = x / y
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 6.0
        m.var_mut(y.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 2.0

    def test_neg(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = -x
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == -5.0

    def test_scalar_add(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = x + 3.0
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 2.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 5.0

    def test_radd(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = 2.0 + x
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 5.0

    def test_scalar_mul(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = 2.0 * x
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 4.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 8.0

    def test_rsub(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = 10.0 - x
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - 7.0) < 1e-10

    def test_pow_float(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = x ** 2.0
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 9.0

    def test_pow_int(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = x ** 2
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 9.0

    def test_rpow(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = 2.0 ** x
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 8.0


class TestExprScalarComparison:
    def test_le_scalar(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        c = x <= 5.0
        m.add_constraint(c)
        m.minimize(x + 0.0)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value <= 0.0

    def test_ge_scalar(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        c = x >= 2.0
        m.add_constraint(c)
        m.minimize(x + 0.0)
        m.close()
        m.var_mut(x.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value <= 0.0

    def test_lt_scalar(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        c = x < 5.0
        m.add_constraint(c)
        m.minimize(x + 0.0)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value < 0.0

    def test_gt_scalar(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        c = x > 2.0
        m.add_constraint(c)
        m.minimize(x + 0.0)
        m.close()
        m.var_mut(x.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value < 0.0


class TestExprComparison:
    def test_le(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        c = x <= y
        m.add_constraint(c)
        m.minimize(x + y)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        m.var_mut(y.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value <= 0.0

    def test_ge(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        c = x >= y
        m.add_constraint(c)
        m.minimize(x + y)
        m.close()
        m.var_mut(x.var_id()).value = 7.0
        m.var_mut(y.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value <= 0.0

    def test_lt(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        c = x < y
        m.add_constraint(c)
        m.minimize(x + y)
        m.close()
        m.var_mut(x.var_id()).value = 2.0
        m.var_mut(y.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value < 0.0

    def test_gt(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        c = x > y
        m.add_constraint(c)
        m.minimize(x + y)
        m.close()
        m.var_mut(x.var_id()).value = 8.0
        m.var_mut(y.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value < 0.0

    def test_eq(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        c = x.eq(y)
        m.add_constraint(c)
        m.minimize(x + y)
        m.close()
        m.var_mut(x.var_id()).value = 5.0
        m.var_mut(y.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value == 0.0

    def test_neq(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        c = x.neq(y)
        m.add_constraint(c)
        m.minimize(x + y)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        m.var_mut(y.var_id()).value = 5.0
        cbls.full_evaluate(m)
        assert m.node(c.handle).value == 0.0


class TestExprMathFunctions:
    def test_sin(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = cbls.sin(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = math.pi / 2
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - 1.0) < 1e-10

    def test_cos(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = cbls.cos(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 0.0
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - 1.0) < 1e-10

    def test_tan(self):
        m = cbls.Model()
        x = m.Float(0, 1)
        f = cbls.tan(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 0.5
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - math.tan(0.5)) < 1e-10

    def test_exp(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = cbls.exp(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 1.0
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - math.exp(1.0)) < 1e-10

    def test_log(self):
        m = cbls.Model()
        x = m.Float(0.01, 10)
        f = cbls.log(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = math.e
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - 1.0) < 1e-10

    def test_sqrt(self):
        m = cbls.Model()
        x = m.Float(0, 100)
        f = cbls.sqrt(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 9.0
        cbls.full_evaluate(m)
        assert abs(m.node(f.handle).value - 3.0) < 1e-10

    def test_abs(self):
        m = cbls.Model()
        x = m.Float(-10, 10)
        f = cbls.abs(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = -5.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 5.0


class TestExprNested:
    def test_complex_expression(self):
        m = cbls.Model()
        x = m.Float(-10, 10)
        y = m.Float(-10, 10)
        f = x * x + 2.0 * x * y + cbls.sin(y)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 2.0
        m.var_mut(y.var_id()).value = 1.0
        cbls.full_evaluate(m)
        expected = 4.0 + 4.0 + math.sin(1.0)
        assert abs(m.node(f.handle).value - expected) < 1e-10

    def test_expr_matches_int32_api(self):
        # Build same model with int32_t API
        m1 = cbls.Model()
        x1 = m1.float_var(-10, 10)
        y1 = m1.float_var(-10, 10)
        two = m1.constant(2)
        f1 = m1.sum([m1.pow_expr(x1, two), m1.prod(two, m1.prod(x1, y1)), m1.sin_expr(y1)])
        m1.minimize(f1)
        m1.close()
        m1.var_mut(vid(x1)).value = 2.0
        m1.var_mut(vid(y1)).value = 1.0
        cbls.full_evaluate(m1)

        # Build same model with Expr API
        m2 = cbls.Model()
        x2 = m2.Float(-10, 10)
        y2 = m2.Float(-10, 10)
        f2 = x2 * x2 + 2.0 * x2 * y2 + cbls.sin(y2)
        m2.minimize(f2)
        m2.close()
        m2.var_mut(x2.var_id()).value = 2.0
        m2.var_mut(y2.var_id()).value = 1.0
        cbls.full_evaluate(m2)

        assert abs(m2.node(f2.handle).value - m1.node(f1).value) < 1e-10


class TestExprFreeFunctions:
    def test_pow_free(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        two = m.Constant(2.0)
        f = cbls.pow(x, two)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 9.0

    def test_min(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        f = cbls.min([x, y])
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        m.var_mut(y.var_id()).value = 7.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 3.0

    def test_max(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        y = m.Float(0, 10)
        f = cbls.max([x, y])
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = 3.0
        m.var_mut(y.var_id()).value = 7.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 7.0

    def test_if_then_else(self):
        m = cbls.Model()
        cond = m.Float(-10, 10)
        a = m.Float(0, 10)
        b = m.Float(0, 10)
        f = cbls.if_then_else(cond, a, b)
        m.minimize(f)
        m.close()
        m.var_mut(cond.var_id()).value = 1.0
        m.var_mut(a.var_id()).value = 5.0
        m.var_mut(b.var_id()).value = 9.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 5.0

    def test_abs_builtin(self):
        m = cbls.Model()
        x = m.Float(-10, 10)
        f = abs(x)
        m.minimize(f)
        m.close()
        m.var_mut(x.var_id()).value = -5.0
        cbls.full_evaluate(m)
        assert m.node(f.handle).value == 5.0


class TestExprIsVar:
    def test_var_expr(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        assert x.is_var() is True

    def test_node_expr(self):
        m = cbls.Model()
        x = m.Float(0, 10)
        f = x + 1.0
        assert f.is_var() is False
