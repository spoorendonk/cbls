"""element, ceil/floor/round and the extra-reading lambdas, from Python (#186).

The engine's arithmetic is pinned in `tests/test_element_rounding.cpp`; what is
pinned here is the binding: every new builder is REACHABLE under the name it is
bound as, in both APIs (the int32 handle API and `Expr`, including Python's own
`math.ceil`/`math.floor`/`round` through the dunders), and a round trip -- built
in Python, evaluated in C++, read back -- gives the value it names.

Validation is pinned in a CHILD interpreter, following `test_model_handles.py`:
the table and every handle an `element` or an extra-lambda is handed are
indexed by the engine without a bounds test on the hot path, so an unvalidated
one is a crash rather than an exception, and a crash in-process would take the
whole pytest run down instead of failing one test.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from functools import partial
from typing import TYPE_CHECKING, Any

import _cbls_core as cbls
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

CHILD_TIMEOUT_SECONDS = 20.0


def vid(handle: int) -> int:
    """Variable id from a variable handle."""
    return -(handle + 1)


def _run_scenario(name: str) -> subprocess.CompletedProcess[str]:
    """Run one `__main__` scenario in a child interpreter, under a hard deadline."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONOPTIMIZE"] = "0"
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), name],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )


def _value_at(m: Any, handle: int, values: dict[int, float]) -> float:
    """Close `m` with `handle` as the objective, assign by var id, evaluate."""
    m.minimize(handle)
    m.close()
    for var_id, value in values.items():
        m.var_mut(var_id).value = value
    cbls.full_evaluate(m)
    out: float = m.node_value(handle)
    return out


# ---------------------------------------------------------------------------
# element
# ---------------------------------------------------------------------------


def test_element_in_the_handle_api_reads_the_table_by_an_int_index() -> None:
    m = cbls.Model()
    t = m.int_var(0, 5)
    e = m.element([5.0, 6.0, 7.0, 8.0], t)
    assert m.node(e).op == cbls.NodeOp.Element
    assert _value_at(m, e, {vid(t): 2}) == 7.0
    m.var_mut(vid(t)).value = 4  # outside the table
    cbls.full_evaluate(m)
    assert m.node_value(e) == 0.0


def test_element_accepts_a_numpy_table_and_a_two_index_form() -> None:
    m = cbls.Model()
    r = m.int_var(0, 1)
    c = m.int_var(0, 2)
    table = np.arange(6, dtype=np.float64).reshape(2, 3)
    e = m.element(table, r, c)
    assert _value_at(m, e, {vid(r): 1, vid(c): 2}) == 5.0


def test_element_in_the_expr_api_follows_an_index_expression() -> None:
    m = cbls.Model()
    t = m.Int(0, 5)
    e = cbls.element([10.0, 20.0, 30.0], t - 1.0)
    assert _value_at(m, e.handle, {t.var_id(): 3}) == 30.0

    m2 = cbls.Model()
    i = m2.Int(0, 1)
    j = m2.Int(0, 1)
    e2 = cbls.element([[1.0, 2.0], [3.0, 4.0]], i, j)
    assert _value_at(m2, e2.handle, {i.var_id(): 1, j.var_id(): 0}) == 3.0


# ---------------------------------------------------------------------------
# ceil / floor / round
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "op", "expected"),
    [("ceil_expr", "Ceil", 3.0), ("floor_expr", "Floor", 2.0), ("round_expr", "Round", 3.0)],
)
def test_a_rounding_op_is_reachable_in_the_handle_api(name: str, op: str, expected: float) -> None:
    m = cbls.Model()
    x = m.float_var(-10, 10)
    node = getattr(m, name)(x)
    assert m.node(node).op == getattr(cbls.NodeOp, op)
    assert _value_at(m, node, {vid(x): 2.5}) == expected


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (cbls.ceil, -2.0),
        (cbls.floor, -3.0),
        (cbls.round, -3.0),  # half-way away from zero
        (math.ceil, -2.0),
        (math.floor, -3.0),
        (round, -3.0),
    ],
)
def test_a_rounding_op_is_reachable_in_the_expr_api(
    build: Callable[[Any], Any], expected: float
) -> None:
    m = cbls.Model()
    x = m.Float(-10, 10)
    node = build(x)
    assert isinstance(node, cbls.Expr)
    assert _value_at(m, node.handle, {x.var_id(): -2.5}) == expected


def test_round_with_ndigits_is_refused_rather_than_rounded_to_an_integer() -> None:
    m = cbls.Model()
    x = m.Float(0, 1)
    with pytest.raises(ValueError, match="ndigits"):
        round(x, 2)


# ---------------------------------------------------------------------------
# lambda_sum / pair_lambda_sum over extra decisions
# ---------------------------------------------------------------------------


STOP_COST = [[5.0, 6.0, 8.0], [4.0, 5.0, 7.0], [6.0, 7.0, 9.0], [3.0, 6.0, 60.0]]


def test_lambda_sum_hands_the_functor_the_extras_current_values() -> None:
    m = cbls.Model()
    route = m.list_var(4)
    vtype = m.int_var(0, 2)
    seen: list[list[float]] = []

    def stop_cost(i: int, x: list[float]) -> float:
        seen.append(list(x))
        return STOP_COST[i][int(x[0])]

    node = m.lambda_sum(route, stop_cost, extra=[vtype])
    assert m.node(node).op == cbls.NodeOp.LambdaExtra
    m.var_mut(vid(route)).elements = [3, 1]
    assert _value_at(m, node, {vid(vtype): 2}) == 60.0 + 7.0
    assert seen[-1] == [2.0]

    # A change to the extra alone re-evaluates the node.
    m.var_mut(vid(vtype)).value = 0
    cbls.delta_evaluate(m, {vid(vtype)})
    assert m.node_value(node) == 3.0 + 4.0


def test_pair_lambda_sum_takes_extra_by_keyword_and_closes_the_cycle() -> None:
    m = cbls.Model()
    route = m.list_var(3)
    speed = m.int_var(1, 3)

    def leg(a: int, b: int, x: list[float]) -> float:
        return (10.0 * a + b) / x[0]

    open_node = m.pair_lambda_sum(route, leg, extra=[speed])
    cyclic_node = m.pair_lambda_sum(route, leg, extra=[speed], cyclic=True)
    assert m.node(cyclic_node).op == cbls.NodeOp.PairLambdaExtra
    m.var_mut(vid(route)).elements = [2, 0, 1]
    m.minimize(m.sum([open_node, cyclic_node]))
    m.close()
    m.var_mut(vid(speed)).value = 2
    cbls.full_evaluate(m)
    assert m.node_value(open_node) == (20.0 + 1.0) / 2
    assert m.node_value(cyclic_node) == (20.0 + 1.0 + 12.0) / 2


def test_extra_is_keyword_only_in_both_lambda_forms() -> None:
    m = cbls.Model()
    route = m.list_var(3)
    t = m.int_var(0, 2)
    with pytest.raises(TypeError):
        m.lambda_sum(route, lambda i, x: float(i), [t])
    with pytest.raises(TypeError):
        m.pair_lambda_sum(route, lambda a, b, x: float(a), [t])


def test_the_node_op_enum_names_every_op_including_pair_lambda_and_custom() -> None:
    m = cbls.Model()
    route = m.list_var(3)
    pair = m.pair_lambda_sum(route, lambda a, b: 0.0)
    assert m.node(pair).op == cbls.NodeOp.PairLambda
    assert cbls.NodeOp.Custom.name == "Custom"


def test_the_plain_lambda_forms_are_unchanged_by_the_extra_overloads() -> None:
    m = cbls.Model()
    route = m.list_var(3)
    plain = m.lambda_sum(route, lambda i: float(i))
    pair = m.pair_lambda_sum(route, lambda a, b: float(a + b), True)
    assert m.node(plain).op == cbls.NodeOp.Lambda
    m.minimize(m.sum([plain, pair]))
    m.close()
    cbls.full_evaluate(m)
    assert m.node_value(plain) == 3.0
    assert m.node_value(pair) == 1.0 + 3.0 + 2.0


# ---------------------------------------------------------------------------
# Validation, in a child interpreter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["bad_element", "bad_lambda_extra"])
def test_a_bad_table_or_handle_raises_instead_of_reaching_the_engine(scenario: str) -> None:
    proc = _run_scenario(scenario)
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} (negative = killed by that signal)\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def _expect(exc: type[BaseException], build: Callable[[], object], what: str) -> None:
    try:
        build()
    except exc:
        return
    raise AssertionError(f"{what}: accepted")


def _scenario_bad_element() -> None:
    """Every malformed table and index is refused, and the model stays usable."""
    m = cbls.Model()
    t = m.int_var(0, 3)
    lst = m.list_var(3)
    _expect(ValueError, lambda: m.element([], t), "empty table")
    _expect(ValueError, lambda: m.element([[]], t, t), "empty row")
    _expect(ValueError, lambda: m.element([[1.0, 2.0], [3.0]], t, t), "ragged table")
    _expect(ValueError, lambda: m.element([1.0, 2.0], lst), "List index")
    for bad in (5, 100_000, -3, -100_000):  # -3: first handle past the 2 variables
        _expect(IndexError, partial(m.element, [1.0, 2.0], bad), f"handle {bad}")
        _expect(IndexError, partial(m.element, [[1.0]], t, bad), f"col handle {bad}")
    _expect(IndexError, lambda: m.ceil_expr(100_000), "ceil handle")
    _expect(ValueError, lambda: m.ceil_expr(lst), "ceil of a List")
    _expect(ValueError, lambda: m.element([1.0, math.inf], t), "infinite entry")
    _expect(ValueError, lambda: m.element([[math.nan]], t, t), "NaN entry")
    e = m.element([1.0, 2.0, 3.0, 4.0], t)
    m.minimize(e)
    m.close()
    m.var_mut(vid(t)).value = 3
    assert cbls.full_evaluate(m) == 4.0
    print("OK")


def _scenario_bad_lambda_extra() -> None:
    """A non-List list, a structured or bogus extra, a None functor: all refused."""
    m = cbls.Model()
    lst = m.list_var(3)
    other = m.list_var(3)
    t = m.int_var(0, 2)

    def f1(i: int, x: list[float]) -> float:
        return i * x[0]

    def f2(a: int, b: int, x: list[float]) -> float:
        return (a + b) * x[0]

    _expect(ValueError, lambda: m.lambda_sum(t, f1, extra=[t]), "scalar as list")
    _expect(ValueError, lambda: m.lambda_sum(m.constant(1.0), f1, extra=[t]), "node as list")
    _expect(ValueError, lambda: m.lambda_sum(lst, f1, extra=[other]), "List as extra")
    _expect(ValueError, lambda: m.pair_lambda_sum(lst, f2, extra=[other]), "pair List extra")
    for bad in (5, 100_000, -100_000):
        _expect(IndexError, partial(m.lambda_sum, lst, f1, extra=[bad]), f"extra {bad}")
        _expect(IndexError, partial(m.pair_lambda_sum, lst, f2, extra=[bad]), f"pair {bad}")
    _expect(TypeError, lambda: m.lambda_sum(lst, None, extra=[t]), "None functor")
    node = m.lambda_sum(lst, f1, extra=[t])
    m.minimize(node)
    m.close()
    m.var_mut(vid(t)).value = 2
    assert cbls.full_evaluate(m) == 2.0 * (0 + 1 + 2)
    print("OK")


if __name__ == "__main__":
    scenario = sys.argv[1]
    if scenario == "bad_element":
        _scenario_bad_element()
    elif scenario == "bad_lambda_extra":
        _scenario_bad_lambda_extra()
    else:
        raise SystemExit(f"unknown scenario {scenario}")
