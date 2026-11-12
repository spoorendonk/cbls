"""Variable-length List variables and list partitions through the bindings (#164).

The variable form, the partition and its two enums are reachable, and every
invariant `add_list_partition` checks is checked from Python too.

`Variable.elements` is a `def_rw`, so Python can write any List (out-of-universe
ids, an element in two lists of a partition, a bad length) that move generators
index by element id: the #156 crash class, so those scenarios run in a child
interpreter (as `test_model_handles.py`).
"""

import sys

import _cbls_core as cbls
import numpy as np
import pytest
from support import run_scenario, vid

CHILD_TIMEOUT_SECONDS = 30.0


def test_list_var_of_one_argument_is_still_a_permutation() -> None:
    m = cbls.Model()
    h = m.list_var(5, "perm")
    v = m.var(vid(h))
    assert v.universe_size == 5
    assert v.min_size == 5
    assert v.max_size == 5
    assert v.list_init == cbls.ListInit.Identity
    assert list(v.elements) == [0, 1, 2, 3, 4]
    assert not v.partitioned


def test_list_var_accepts_a_length_window() -> None:
    m = cbls.Model()
    h = m.list_var(universe=8, min_len=2, max_len=5, init=cbls.ListInit.Random, name="seq")
    v = m.var(vid(h))
    assert (v.universe_size, v.min_size, v.max_size) == (8, 2, 5)
    assert v.list_init == cbls.ListInit.Random
    assert list(v.elements) == []


def test_list_expr_form_accepts_a_length_window() -> None:
    m = cbls.Model()
    e = m.List(universe=6, min_len=0, max_len=3)
    assert m.var(e.var_id()).max_size == 3


@pytest.mark.parametrize(
    ("universe", "min_len", "max_len", "init"),
    [
        (5, 3, 2, cbls.ListInit.Empty),
        (5, 0, 6, cbls.ListInit.Empty),
        (5, -1, 2, cbls.ListInit.Empty),
        (5, 0, 5, cbls.ListInit.Identity),
        (5, 1, 3, cbls.ListInit.Empty),
    ],
)
def test_list_var_rejects_an_impossible_window(
    universe: int, min_len: int, max_len: int, init: "cbls.ListInit"
) -> None:
    m = cbls.Model()
    with pytest.raises(ValueError):
        m.list_var(universe, min_len, max_len, init)


def test_add_list_partition_records_membership() -> None:
    m = cbls.Model()
    a = m.list_var(6, 0, 6, cbls.ListInit.Empty, "a")
    b = m.list_var(6, 0, 6, cbls.ListInit.Empty, "b")
    p = m.add_list_partition([a, b], cbls.Cover.Exact)
    assert p == 0
    assert m.partition_of_list(vid(a)) == 0
    assert m.partition_of_list(vid(b)) == 0
    assert m.partition_of_list(99) == -1
    assert m.var(vid(a)).partitioned
    parts = m.list_partitions
    assert len(parts) == 1
    assert list(parts[0].list_ids) == [vid(a), vid(b)]
    assert parts[0].cover == cbls.Cover.Exact
    assert parts[0].universe_size == 6


@pytest.mark.parametrize("cover", ["exact", "at_most_once"])
def test_add_list_partition_accepts_the_cover_as_a_string(cover: str) -> None:
    m = cbls.Model()
    a = m.list_var(4, 0, 4)
    b = m.list_var(4, 0, 4)
    m.add_list_partition([a, b], cover)
    expected = cbls.Cover.Exact if cover == "exact" else cbls.Cover.AtMostOnce
    assert m.list_partitions[0].cover == expected


def test_add_list_partition_rejects_an_unknown_cover_string() -> None:
    m = cbls.Model()
    a = m.list_var(4, 0, 4)
    with pytest.raises(ValueError):
        m.add_list_partition([a], "sometimes")


def test_add_list_partition_validates_its_group() -> None:
    """Each failed call must leave the model exactly as it was.

    Deliberately cumulative: `wide` is still unpartitioned when the successful
    call reaches it only if no rejected group set its `partitioned` flag.
    """
    m = cbls.Model()
    a = m.list_var(4, 0, 4)
    scalar = m.bool_var("b")
    with pytest.raises(ValueError):
        m.add_list_partition([a, scalar], cbls.Cover.AtMostOnce)
    wide = m.list_var(9, 0, 9)
    with pytest.raises(ValueError):
        m.add_list_partition([a, wide], cbls.Cover.AtMostOnce)
    small = m.list_var(9, 0, 2)
    with pytest.raises(ValueError):
        m.add_list_partition([wide, wide], cbls.Cover.Exact)  # named twice
    tiny = m.list_var(9, 0, 2)
    with pytest.raises(ValueError):
        m.add_list_partition([small, tiny], cbls.Cover.Exact)  # 2 + 2 < 9
    m.add_list_partition([wide, small], cbls.Cover.Exact)
    with pytest.raises(ValueError):
        m.add_list_partition([wide], cbls.Cover.AtMostOnce)  # already partitioned


def test_an_exact_partition_is_solved_without_breaking_its_cover() -> None:
    m = cbls.Model()
    routes = [m.list_var(10, 0, 10, cbls.ListInit.Empty, f"r{i}") for i in range(3)]
    m.add_list_partition(routes, cbls.Cover.Exact)
    terms = [m.lambda_table_sum(r, np.arange(10, dtype=np.float64)) for r in routes]
    for r in routes:
        m.add_constraint(m.leq(m.count(r), m.constant(5.0)))
    m.minimize(m.sum(terms))
    m.close()

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2000
    # solve() leaves the model holding the state it returns, so the lists can be
    # read straight off it.
    cbls.solve(m, 0.0, 7, None, None, 3, None, cfg)
    seen: list[int] = []
    for r in routes:
        seen.extend(m.var(vid(r)).elements)
    assert sorted(seen) == list(range(10))


def test_a_broken_invariant_from_python_does_not_crash_the_interpreter() -> None:
    proc = run_scenario(__file__, "broken_invariant", CHILD_TIMEOUT_SECONDS)
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_an_out_of_universe_element_does_not_crash_the_interpreter() -> None:
    proc = run_scenario(__file__, "out_of_universe", CHILD_TIMEOUT_SECONDS)
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    assert "OK" in proc.stdout


def _partitioned_model(*, tabulated: bool) -> tuple["cbls.Model", list[int]]:
    """Three routes over one universe, read through Count and optionally a table.

    `tabulated=False` keeps every node bounds-safe on any element id, which is
    what the out-of-universe scenario needs: `lambda_table_sum` deliberately
    REFUSES an element outside the tabulated universe (#156), so a model built on
    one reports the corruption instead of searching with it.
    """
    m = cbls.Model()
    routes = [m.list_var(12, 0, 12, cbls.ListInit.Empty, f"r{i}") for i in range(3)]
    m.add_list_partition(routes, cbls.Cover.AtMostOnce)
    terms = [m.count(r) for r in routes]
    if tabulated:
        terms += [m.lambda_table_sum(r, np.arange(12, dtype=np.float64)) for r in routes]
    for r in routes:
        m.add_constraint(m.leq(m.count(r), m.constant(6.0)))
    m.minimize(m.sum(terms))
    m.close()
    return m, routes


def _scenario_broken_invariant() -> None:
    """The same element in two lists of one partition, and a length out of range.

    The moves preserve the cover, not restore it, so the engine cannot repair
    this; but membership is recomputed per call with bounds-tested stamp
    writes, so it is a wrong search, not a heap write.
    """
    m, routes = _partitioned_model(tabulated=True)
    # Every route holds the same three elements, and one is longer than max_len
    # allows.
    for r in routes:
        m.var_mut(vid(r)).elements = [0, 1, 2]
    m.var_mut(vid(routes[0])).elements = list(range(12)) + [0]
    cbls.full_evaluate(m)

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2000
    cbls.solve(m, 0.0, 3, None, None, 3, None, cfg)
    print("OK")


def _scenario_out_of_universe() -> None:
    """Element ids past the universe, which the membership stamps index by.

    Every stamp write in `list_membership` and `partition_unassigned` is
    bounds-tested, so such an id is ignored, not a heap write; the search runs
    on an unrepairable state and the interpreter survives.
    """
    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2000

    m, routes = _partitioned_model(tabulated=False)
    m.var_mut(vid(routes[0])).elements = [0, 1, 1_000_000, -5]
    m.var_mut(vid(routes[1])).elements = [2, 2_000_000_000]
    cbls.full_evaluate(m)
    cbls.solve(m, 0.0, 4, None, None, 3, None, cfg)

    # The same on an unpartitioned variable-length List, whose `list_insert`
    # builds its own membership stamp.
    m2 = cbls.Model()
    seq = m2.list_var(10, 0, 10, cbls.ListInit.Empty, "seq")
    m2.minimize(m2.count(seq))
    m2.close()
    m2.var_mut(vid(seq)).elements = [3, 400_000, -1]
    cbls.full_evaluate(m2)
    cbls.solve(m2, 0.0, 5, None, None, 3, None, cfg)

    # A tabulated read of the same corruption reports it instead of indexing
    # past the table: the #156 guard, on a variable-length List.
    m3, spoiled = _partitioned_model(tabulated=True)
    m3.var_mut(vid(spoiled[0])).elements = [1_000_000]
    try:
        cbls.full_evaluate(m3)
    except IndexError:
        print("OK")
        return
    raise AssertionError("lambda_table_sum accepted an out-of-universe element")


if __name__ == "__main__":
    name = sys.argv[1]
    if name == "broken_invariant":
        _scenario_broken_invariant()
    elif name == "out_of_universe":
        _scenario_out_of_universe()
    else:
        raise SystemExit(f"unknown scenario {name}")
