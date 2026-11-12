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
from support import vid


def _closed_model() -> tuple[cbls.Model, int, int, int]:
    m = cbls.Model()
    x = m.int_var(0, 20, "x")
    y = m.int_var(0, 20, "y")
    # A structured variable, so lambda_sum gets a handle it ACCEPTS and is refused
    # for being frozen: it rejects a non-List/Set handle before the freeze check,
    # as its table-backed and pair siblings do.
    s = m.set_var(4, 0, 4, "s")
    total = m.sum([x, y, m.lambda_sum(s, lambda e: float(e))])
    row = m.leq(total, m.constant(10.0))
    m.add_constraint(row)
    m.minimize(total)
    m.close()
    return m, x, row, s


def test_freeze_folds_in_the_objective_row_and_is_idempotent() -> None:
    m, _x, _row, _s = _closed_model()
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
        pytest.param(lambda m, x, row, s: m.bool_var(), id="bool_var"),
        pytest.param(lambda m, x, row, s: m.int_var(0, 1), id="int_var"),
        pytest.param(lambda m, x, row, s: m.float_var(0.0, 1.0), id="float_var"),
        pytest.param(lambda m, x, row, s: m.list_var(3), id="list_var"),
        pytest.param(lambda m, x, row, s: m.set_var(3), id="set_var"),
        pytest.param(lambda m, x, row, s: m.constant(1.0), id="constant"),
        pytest.param(lambda m, x, row, s: m.neg(row), id="neg"),
        pytest.param(lambda m, x, row, s: m.sum([row, row]), id="sum"),
        pytest.param(lambda m, x, row, s: m.add_constraint(row), id="add_constraint"),
        pytest.param(lambda m, x, row, s: m.minimize(row), id="minimize"),
        pytest.param(lambda m, x, row, s: m.maximize(row), id="maximize"),
        pytest.param(lambda m, x, row, s: m.add_var_sequence([x]), id="add_var_sequence"),
        pytest.param(lambda m, x, row, s: m.close(), id="close"),
        # The sharpest one to refuse from Python: `lambda_sum` grows a table every
        # worker reads, and the callable would be invoked concurrently.
        pytest.param(lambda m, x, row, s: m.lambda_sum(s, lambda e: float(e)), id="lambda_sum"),
    ],
)
def test_frozen_model_refuses_structural_change(call: object) -> None:
    m, x, row, s = _closed_model()
    m.freeze()
    nodes = m.num_nodes()
    variables = m.num_vars()

    with pytest.raises(RuntimeError, match="frozen"):
        call(m, x, row, s)  # type: ignore[operator]

    # The refusal left nothing behind.
    assert m.num_nodes() == nodes
    assert m.num_vars() == variables


def test_a_frozen_model_still_does_everything_a_search_does() -> None:
    m, x, row, s = _closed_model()
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
    """`Model.node_value(id)` is a raw-int32 index reachable from Python, and
    CLAUDE.md's nanobind rule is that its guard gets a test: unchecked, these
    would be heap reads rather than exceptions.
    """
    m, _x, row, _s = _closed_model()
    # x + y + lambda_sum(s) - 10, with s empty so its term is 0.0.
    assert m.node_value(row) == pytest.approx(0.0 + 0.0 + 0.0 - 10.0)
    for bad in (-1, m.num_nodes(), m.num_nodes() + 1000, -(2**31)):
        with pytest.raises(IndexError):
            m.node_value(bad)


def test_freeze_caches_every_variables_cone() -> None:
    """`freeze()` caches each variable's cone (#271): every one under the
    default cap, as whole 4-byte node ids. An unfrozen model has none."""
    m, _x, _row, _s = _closed_model()
    assert m.num_cached_cones() == 0
    assert m.cone_cache_bytes() == 0

    m.freeze()
    assert m.num_cached_cones() == m.num_vars()
    assert m.cone_cache_bytes() > 0
    assert m.cone_cache_bytes() % 4 == 0


def test_the_first_freezes_cap_holds() -> None:
    """A cap of 0 caches nothing, and a later `freeze()` is a no-op: it does not
    build the cache the first call left out."""
    m, _x, _row, _s = _closed_model()
    m.freeze(cone_cache_bytes=0)
    assert m.is_frozen()
    assert (m.num_cached_cones(), m.cone_cache_bytes()) == (0, 0)

    m.freeze()
    assert (m.num_cached_cones(), m.cone_cache_bytes()) == (0, 0)


def test_a_cached_walk_computes_what_an_uncached_one_does() -> None:
    """The same one-variable walks on a model frozen with the cache and one
    frozen without it leave every node with the same value."""
    cached, x, _row, _s = _closed_model()
    uncached, _x2, _row2, _s2 = _closed_model()
    cached.freeze()
    uncached.freeze(cone_cache_bytes=0)
    xid = vid(x)

    for value in (3.0, 7.0, 0.0, 12.0):
        for m in (cached, uncached):
            m.var_mut(xid).value = value
            cbls.delta_evaluate(m, [xid])
        assert [cached.node_value(i) for i in range(cached.num_nodes())] == [
            uncached.node_value(i) for i in range(uncached.num_nodes())
        ]


def test_solve_freezes_the_model_it_is_handed() -> None:
    """`solve()` freezes its model (#271), with `SearchConfig.cone_cache_bytes`
    as the cap, so a structural change after it is refused."""
    m, _x, _row, _s = _closed_model()
    config = cbls.SearchConfig()
    assert config.cone_cache_bytes == 256 << 20
    config.cone_cache_bytes = 0
    assert config.cone_cache_bytes == 0
    config.max_iterations = 100

    cbls.solve(m, 0.0, 1, config=config)
    assert m.is_frozen()
    assert m.num_cached_cones() == 0
    with pytest.raises(RuntimeError, match="frozen"):
        m.int_var(0, 1)


def test_a_negative_cone_cache_cap_is_refused() -> None:
    """The cap is unsigned on the C++ side; a negative one is a TypeError, not
    a wrapped-around huge cap."""
    config = cbls.SearchConfig()
    with pytest.raises(TypeError):
        config.cone_cache_bytes = -1
    m, _x, _row, _s = _closed_model()
    with pytest.raises(TypeError):
        m.freeze(cone_cache_bytes=-1)
    assert not m.is_frozen()
