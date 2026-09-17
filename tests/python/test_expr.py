"""Expr wrapper, operator overloading and the int32 handle API, via the bindings.

What these tests are for: that every operation is REACHABLE from Python under
the name it is bound as, in both APIs, and that it dispatches to the operation
it names. The arithmetic itself belongs to the engine and is asserted in
`tests/test_expr.cpp` and `tests/test_dag.cpp`, at C++ speed and nearer the code
-- so a value here is a dispatch witness (it separates `tan` from `sin`), not a
second opinion on whether the DAG evaluates correctly.

Two surfaces are bound and both are covered, because they are separate binding
code and either can break alone:

  * the int32 handle API -- `m.float_var()`, `m.tan_expr(handle)`;
  * the `Expr` API -- `m.Float()`, `cbls.tan(expr)`, and Python's operators
    through `__add__`, `__radd__`, `__pow__`, `__abs__`, the comparison dunders.
"""

import math
import operator
from collections.abc import Callable
from typing import Any

import _cbls_core as cbls
import pytest


def vid(handle: int) -> int:
    """Variable id from an int32 API variable handle."""
    return -(handle + 1)


def eval_handle(m: Any, handle: int, values: dict[int, float]) -> float:
    """Close `m`, assign `values` by variable id, evaluate, return the node's value."""
    m.minimize(handle)
    m.close()
    for var_id, value in values.items():
        m.var_mut(var_id).value = value
    cbls.full_evaluate(m)
    node_value: float = m.node(handle).value
    return node_value


def eval_expr(m: Any, expr: Any, values: dict[Any, float]) -> float:
    """`eval_handle` for the Expr API, keyed by Expr rather than by variable id."""
    return eval_handle(m, expr.handle, {x.var_id(): v for x, v in values.items()})


#: name of the int32 API method, the Expr API free function, the reference
#: implementation, a domain the operation is defined on, and a point in it.
UNARY_OPS = [
    ("sin_expr", cbls.sin, math.sin, (0.0, 10.0), math.pi / 2),
    ("cos_expr", cbls.cos, math.cos, (0.0, 10.0), 0.0),
    ("tan_expr", cbls.tan, math.tan, (-1.0, 1.0), 0.5),
    ("exp_expr", cbls.exp, math.exp, (-10.0, 10.0), 1.0),
    ("log_expr", cbls.log, math.log, (0.01, 10.0), math.e),
    ("sqrt_expr", cbls.sqrt, math.sqrt, (0.0, 100.0), 9.0),
    ("abs_expr", cbls.abs, abs, (-10.0, 10.0), -5.0),
]


@pytest.mark.parametrize(("method", "free", "reference", "domain", "point"), UNARY_OPS)
def test_a_unary_op_is_reachable_and_dispatches_in_both_apis(
    method: str,
    free: Callable[[Any], Any],
    reference: Callable[[float], float],
    domain: tuple[float, float],
    point: float,
) -> None:
    expected = reference(point)

    m = cbls.Model()
    x = m.float_var(*domain)
    assert eval_handle(m, getattr(m, method)(x), {vid(x): point}) == pytest.approx(
        expected, abs=1e-10
    )

    m2 = cbls.Model()
    x2 = m2.Float(*domain)
    assert eval_expr(m2, free(x2), {x2: point}) == pytest.approx(expected, abs=1e-10)


#: Python operator, the two operands, and the result. Covers `Expr op Expr`.
BINARY_OPS = [
    (operator.add, 3.0, 4.0, 7.0),
    (operator.sub, 7.0, 3.0, 4.0),
    (operator.mul, 3.0, 4.0, 12.0),
    (operator.truediv, 6.0, 3.0, 2.0),
]


@pytest.mark.parametrize(("op", "left", "right", "expected"), BINARY_OPS)
def test_an_arithmetic_operator_builds_the_node_it_names(
    op: Callable[[Any, Any], Any], left: float, right: float, expected: float
) -> None:
    m = cbls.Model()
    x = m.Float(0.0, 10.0)
    y = m.Float(1.0, 10.0)  # lower bound 1 keeps the division case well defined
    assert eval_expr(m, op(x, y), {x: left, y: right}) == pytest.approx(expected, abs=1e-10)


#: An expression mixing an Expr with a Python scalar, and its value at x = 3.
#: Both orders, so the reflected dunders (__radd__, __rsub__, __rpow__) are
#: covered as well as the direct ones.
SCALAR_OPS = [
    (lambda x: x + 3.0, 6.0),
    (lambda x: 2.0 + x, 5.0),
    (lambda x: x - 1.0, 2.0),
    (lambda x: 10.0 - x, 7.0),
    (lambda x: x * 2.0, 6.0),
    (lambda x: 2.0 * x, 6.0),
    (lambda x: x**2.0, 9.0),
    (lambda x: x**2, 9.0),  # an int exponent, which is a separate overload
    (lambda x: 2.0**x, 8.0),
    (lambda x: -x, -3.0),
    (lambda x: abs(-x), 3.0),  # __abs__, i.e. Python's builtin
]


@pytest.mark.parametrize(("build", "expected"), SCALAR_OPS)
def test_an_operator_between_an_expr_and_a_scalar(
    build: Callable[[Any], Any], expected: float
) -> None:
    m = cbls.Model()
    x = m.Float(0.0, 10.0)
    assert eval_expr(m, build(x), {x: 3.0}) == pytest.approx(expected, abs=1e-10)


#: A comparison, an assignment that SATISFIES it, and the residual sign that
#: satisfaction is reported as. A residual is `lhs - rhs` shaped: `<= 0` for the
#: non-strict forms, `< 0` for the strict ones, and exactly 0 for equality.
COMPARISONS = [
    (lambda x, y: x <= y, 3.0, 5.0, "le"),
    (lambda x, y: x >= y, 7.0, 3.0, "le"),
    (lambda x, y: x < y, 2.0, 5.0, "lt"),
    (lambda x, y: x > y, 8.0, 3.0, "lt"),
    (lambda x, y: x.eq(y), 5.0, 5.0, "eq"),
    (lambda x, y: x.neq(y), 3.0, 5.0, "eq"),
    (lambda x, y: x <= 5.0, 3.0, 0.0, "le"),
    (lambda x, y: x >= 2.0, 5.0, 0.0, "le"),
    (lambda x, y: x < 5.0, 3.0, 0.0, "lt"),
    (lambda x, y: x > 2.0, 5.0, 0.0, "lt"),
]


@pytest.mark.parametrize(("build", "left", "right", "satisfied_as"), COMPARISONS)
def test_a_comparison_reports_a_satisfied_point_as_satisfied(
    build: Callable[[Any, Any], Any], left: float, right: float, satisfied_as: str
) -> None:
    m = cbls.Model()
    x = m.Float(0.0, 10.0)
    y = m.Float(0.0, 10.0)
    constraint = build(x, y)
    m.add_constraint(constraint)
    residual = eval_expr(m, constraint, {x: left, y: right})

    if satisfied_as == "eq":
        assert residual == 0.0
    elif satisfied_as == "lt":
        assert residual < 0.0
    else:
        assert residual <= 0.0


#: The int32 API's own comparison constructors, which are bound separately from
#: the operator dunders above and can break independently of them.
HANDLE_COMPARISONS = [
    ("geq", 5.0, 3.0, "le"),
    ("lt", 2.0, 5.0, "lt"),
    ("gt", 7.0, 3.0, "lt"),
]


@pytest.mark.parametrize(("method", "left", "right", "satisfied_as"), HANDLE_COMPARISONS)
def test_a_handle_api_comparison_reports_a_satisfied_point_as_satisfied(
    method: str, left: float, right: float, satisfied_as: str
) -> None:
    m = cbls.Model()
    x = m.float_var(0.0, 10.0)
    y = m.float_var(0.0, 10.0)
    node = getattr(m, method)(x, y)
    residual = eval_handle(m, node, {vid(x): left, vid(y): right})
    assert residual < 0.0 if satisfied_as == "lt" else residual <= 0.0


def test_handle_api_neq_is_violated_only_when_the_operands_are_equal() -> None:
    # Its own case because neq is the one comparison whose residual is a FLAG
    # rather than a magnitude: 1 when violated, 0 when satisfied.
    m = cbls.Model()
    x = m.float_var(0.0, 10.0)
    y = m.float_var(0.0, 10.0)
    node = m.neq(x, y)
    m.minimize(node)
    m.close()

    m.var_mut(vid(x)).value = 3.0
    m.var_mut(vid(y)).value = 3.0
    cbls.full_evaluate(m)
    assert m.node(node).value == 1.0

    m.var_mut(vid(y)).value = 5.0
    cbls.full_evaluate(m)
    assert m.node(node).value == 0.0


def test_free_functions_over_expr_lists() -> None:
    # min/max take a LIST of Expr, which is its own caster; if_then_else takes
    # three positional Expr. Each gets its own model because `eval_expr` closes
    # the one it is given.
    m2 = cbls.Model()
    a = m2.Float(0.0, 10.0)
    b = m2.Float(0.0, 10.0)
    assert eval_expr(m2, cbls.min([a, b]), {a: 3.0, b: 7.0}) == 3.0
    m3 = cbls.Model()
    a3 = m3.Float(0.0, 10.0)
    b3 = m3.Float(0.0, 10.0)
    assert eval_expr(m3, cbls.max([a3, b3]), {a3: 3.0, b3: 7.0}) == 7.0
    m4 = cbls.Model()
    cond = m4.Float(-10.0, 10.0)
    t = m4.Float(0.0, 10.0)
    f = m4.Float(0.0, 10.0)
    assert eval_expr(m4, cbls.if_then_else(cond, t, f), {cond: 1.0, t: 5.0, f: 9.0}) == 5.0


def test_pow_with_a_constant_expr() -> None:
    # cbls.pow takes two Exprs, so it needs m.Constant() rather than a scalar --
    # a different path from the `x ** 2.0` dunder covered above.
    m = cbls.Model()
    x = m.Float(0.0, 10.0)
    assert eval_expr(m, cbls.pow(x, m.Constant(2.0)), {x: 3.0}) == 9.0


def test_is_var_distinguishes_a_variable_from_an_expression() -> None:
    m = cbls.Model()
    x = m.Float(0.0, 10.0)
    assert x.is_var() is True
    assert (x + 1.0).is_var() is False


def test_the_two_apis_build_the_same_model() -> None:
    """The cross-API equivalence, which is what makes either surface trustworthy.

    Neither API is the other's wrapper on the Python side -- `Expr` builds nodes
    through its own bound operators -- so agreeing on a non-trivial expression is
    a real statement about both.
    """
    m1 = cbls.Model()
    x1 = m1.float_var(-10.0, 10.0)
    y1 = m1.float_var(-10.0, 10.0)
    two = m1.constant(2)
    f1 = m1.sum([m1.pow_expr(x1, two), m1.prod(two, m1.prod(x1, y1)), m1.sin_expr(y1)])
    handle_value = eval_handle(m1, f1, {vid(x1): 2.0, vid(y1): 1.0})

    m2 = cbls.Model()
    x2 = m2.Float(-10.0, 10.0)
    y2 = m2.Float(-10.0, 10.0)
    expr_value = eval_expr(m2, x2 * x2 + 2.0 * x2 * y2 + cbls.sin(y2), {x2: 2.0, y2: 1.0})

    assert expr_value == pytest.approx(handle_value, abs=1e-10)
    assert expr_value == pytest.approx(4.0 + 4.0 + math.sin(1.0), abs=1e-10)
