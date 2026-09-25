"""A frozen Model refuses structural changes from Python too (#157).

This is the binding half of `tests/test_model_share.cpp`. It matters on its own
because Python is where a raw `int32_t` handle or an out-of-contract call arrives
with no type system in the way (#156): a frozen model's nodes live in storage
every portfolio replica reads, so a structural mutator reaching them from here
would be a write into a peer's DAG. The guard turns that into an ordinary
exception, which is what makes it testable at all rather than a crash.
"""

from __future__ import annotations

import _cbls_core as cbls
import pytest


def vid(handle: int) -> int:
    """Variable handle (negative, as the builders return) -> variable id."""
    return -(handle + 1)


def _closed_model() -> tuple[cbls.Model, int, int]:
    m = cbls.Model()
    x = m.int_var(0, 20, "x")
    y = m.int_var(0, 20, "y")
    total = m.sum([x, y])
    row = m.leq(total, m.constant(10.0))
    m.add_constraint(row)
    m.minimize(total)
    m.close()
    return m, x, row


def test_freeze_folds_in_the_objective_row_and_is_idempotent() -> None:
    m, _x, _row = _closed_model()
    assert not m.is_frozen()
    nodes_before = m.num_nodes()

    m.freeze()
    assert m.is_frozen()
    # The objective is folded into the constraint set at the freeze point, which
    # is where solve() would have done it.
    assert m.num_nodes() == nodes_before + 2

    m.freeze()
    assert m.num_nodes() == nodes_before + 2


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda m, x, row: m.bool_var(), id="bool_var"),
        pytest.param(lambda m, x, row: m.int_var(0, 1), id="int_var"),
        pytest.param(lambda m, x, row: m.float_var(0.0, 1.0), id="float_var"),
        pytest.param(lambda m, x, row: m.list_var(3), id="list_var"),
        pytest.param(lambda m, x, row: m.set_var(3), id="set_var"),
        pytest.param(lambda m, x, row: m.constant(1.0), id="constant"),
        pytest.param(lambda m, x, row: m.neg(row), id="neg"),
        pytest.param(lambda m, x, row: m.sum([row, row]), id="sum"),
        pytest.param(lambda m, x, row: m.add_constraint(row), id="add_constraint"),
        pytest.param(lambda m, x, row: m.minimize(row), id="minimize"),
        pytest.param(lambda m, x, row: m.maximize(row), id="maximize"),
        pytest.param(lambda m, x, row: m.add_var_sequence([x]), id="add_var_sequence"),
        pytest.param(lambda m, x, row: m.close(), id="close"),
        # The sharpest one to refuse from Python: `lambda_sum` grows a table every
        # worker reads, and the callable would be invoked concurrently.
        pytest.param(lambda m, x, row: m.lambda_sum(x, lambda e: float(e)), id="lambda_sum"),
    ],
)
def test_frozen_model_refuses_structural_change(call: object) -> None:
    m, x, row = _closed_model()
    m.freeze()
    nodes = m.num_nodes()
    variables = m.num_vars()

    with pytest.raises(RuntimeError, match="frozen"):
        call(m, x, row)  # type: ignore[operator]

    # The refusal left nothing behind.
    assert m.num_nodes() == nodes
    assert m.num_vars() == variables


def test_a_frozen_model_still_does_everything_a_search_does() -> None:
    m, x, row = _closed_model()
    m.freeze()
    xid = vid(x)

    m.var_mut(xid).value = 4.0
    cbls.delta_evaluate(m, [xid])
    assert m.node_value(row) == pytest.approx(4.0 - 10.0)

    state = m.copy_state()
    m.var_mut(xid).value = 1.0
    cbls.full_evaluate(m)
    m.restore_state(state)
    cbls.full_evaluate(m)
    assert m.var(xid).value == 4.0
    assert m.node_value(row) == pytest.approx(4.0 - 10.0)


def test_node_value_range_checks_its_index() -> None:
    """`Model.node_value(id)` replaced `ExprNode.value`, so it is a new raw-int32
    index reachable from Python. CLAUDE.md's nanobind rule is that such a guard
    gets a test: unchecked, these would be heap reads rather than exceptions.
    """
    m, _x, row = _closed_model()
    assert m.node_value(row) == pytest.approx(0.0 + 0.0 - 10.0)
    for bad in (-1, m.num_nodes(), m.num_nodes() + 1000, -(2**31)):
        with pytest.raises(IndexError):
            m.node_value(bad)
