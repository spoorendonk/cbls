"""The ordinary builders refuse a closed model from Python (#173).

Before the refusal, `add_constraint` (or any other builder) on a closed model was
accepted, but the new node was never placed in the topological order, so no
evaluation computed it and `solve` reported feasible over a violated row -- a
silent wrong answer. `ModelExtension` + `Model.extend` is the one way to grow a
closed model, and the refusal names it.

The binding half of the `[closed]` cases in `tests/test_model_extend.cpp`: the
refusal is a C++ `std::logic_error`, which reaches Python as `RuntimeError`
with no binding code of its own, so this pins that it does.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import _cbls_core as cbls
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

REFUSAL = r"ModelExtension.*Model::extend"


def vid(handle: int) -> int:
    """Variable handle (negative, as the builders return) -> variable id."""
    return -(handle + 1)


def test_add_constraint_after_close_raises_instead_of_solving_wrong() -> None:
    # The issue's repro, verbatim up to the second add_constraint.
    m = cbls.Model()
    x = m.int_var(0, 10, "x")
    m.add_constraint(m.leq(x, m.constant(20)))
    five = m.constant(5)  # built before close(), so only the row is new below
    m.close()
    rows = len(m.constraint_ids())

    with pytest.raises(RuntimeError, match=REFUSAL):
        m.add_constraint(m.geq(x, five))
    with pytest.raises(RuntimeError, match=REFUSAL):
        m.constant(5)
    assert len(m.constraint_ids()) == rows

    # The route the message names grows the model and gets the row evaluated.
    ext = cbls.ModelExtension(m)
    ext.add_constraint(ext.geq(x, ext.constant(5)))
    m.extend(ext)
    r = cbls.solve(m, time_limit=0.2, seed=1)
    assert r.feasible
    assert m.var(vid(x)).value >= 5.0


def _closed() -> tuple[cbls.Model, int, int, int, int]:
    m = cbls.Model()
    x = m.int_var(0, 10, "x")
    y = m.int_var(0, 10, "y")
    s = m.set_var(4, 0, 4, "s")
    row = m.sum([x, y, m.lambda_sum(s, lambda e: float(e))])
    m.add_constraint(m.leq(row, m.constant(20.0)))
    m.close()
    return m, x, y, s, row


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda m, x, y, s, row: m.bool_var(), id="bool_var"),
        pytest.param(lambda m, x, y, s, row: m.int_var(0, 1), id="int_var"),
        pytest.param(lambda m, x, y, s, row: m.float_var(0.0, 1.0), id="float_var"),
        pytest.param(lambda m, x, y, s, row: m.list_var(3), id="list_var"),
        pytest.param(lambda m, x, y, s, row: m.set_var(3), id="set_var"),
        pytest.param(lambda m, x, y, s, row: m.constant(1.0), id="constant"),
        pytest.param(lambda m, x, y, s, row: m.neg(x), id="neg"),
        pytest.param(lambda m, x, y, s, row: m.sum([x, y]), id="sum"),
        pytest.param(lambda m, x, y, s, row: m.prod(x, y), id="prod"),
        pytest.param(lambda m, x, y, s, row: m.geq(x, y), id="geq"),
        pytest.param(lambda m, x, y, s, row: m.count(s), id="count"),
        pytest.param(lambda m, x, y, s, row: m.lambda_sum(s, lambda e: float(e)), id="lambda_sum"),
        pytest.param(
            lambda m, x, y, s, row: m.pair_lambda_sum(s, lambda a, b: float(a + b)),
            id="pair_lambda_sum",
        ),
        pytest.param(lambda m, x, y, s, row: m.add_constraint(row), id="add_constraint"),
        pytest.param(lambda m, x, y, s, row: m.minimize(row), id="minimize"),
        pytest.param(lambda m, x, y, s, row: m.maximize(row), id="maximize"),
        pytest.param(lambda m, x, y, s, row: m.add_var_sequence([x, y]), id="add_var_sequence"),
    ],
)
def test_every_builder_refuses_a_closed_model(
    call: Callable[[cbls.Model, int, int, int, int], object],
) -> None:
    m, x, y, s, row = _closed()
    nodes = m.num_nodes()
    variables = m.num_vars()
    rows = len(m.constraint_ids())
    objective = m.objective_id()

    with pytest.raises(RuntimeError, match=REFUSAL):
        call(m, x, y, s, row)

    # The refusal left nothing behind.
    assert m.num_nodes() == nodes
    assert m.num_vars() == variables
    assert len(m.constraint_ids()) == rows
    assert m.objective_id() == objective


def test_expr_operators_refuse_a_closed_model() -> None:
    m = cbls.Model()
    x = m.Int(0, 10, "x")
    y = m.Int(0, 10, "y")
    m.add_constraint(x + y <= 20)
    m.close()
    nodes = m.num_nodes()
    with pytest.raises(RuntimeError, match=REFUSAL):
        _ = x + y
    with pytest.raises(RuntimeError, match=REFUSAL):
        m.Int(0, 1)
    assert m.num_nodes() == nodes


def test_an_objective_model_still_solves_after_close() -> None:
    # The first solve on an objective model appends the `obj <= bound` row after
    # close() -- the one internal growth path, which the refusal must not catch.
    m = cbls.Model()
    x = m.int_var(0, 10, "x")
    y = m.int_var(0, 10, "y")
    total = m.sum([x, y])
    m.add_constraint(m.geq(total, m.constant(3.0)))
    m.minimize(total)
    m.close()
    r = cbls.solve(m, time_limit=0.2, seed=1)
    assert r.feasible
    assert r.objective == pytest.approx(3.0)
