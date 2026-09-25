"""pair_lambda_sum and the table-backed lambda forms, from Python (#163).

Three things are pinned here.

**Values.** Both functor forms and both table forms, against a hand
computation, including the short lists (n = 0, 1, 2) whose behaviour the C++
header defines rather than leaves to chance.

**No Python on the hot path.** `pair_table_sum`/`lambda_table_sum` exist
because a Python functor re-acquires the GIL once per pair per node evaluation,
which serialises every portfolio worker on the interpreter. `sys.setprofile`
counts the `call` events a solve produces: the functor form makes thousands,
the table form makes none at all. Without this test the table form could
silently degrade into a wrapper around a Python callable and still pass every
value assertion.

**Crashes, not exceptions.** An ndarray of the wrong shape and an element
outside the tabulated universe are both heap reads past the end of a C++
vector, which take the interpreter down rather than failing a test. Those run
in a CHILD interpreter, following `test_model_handles.py`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import _cbls_core as cbls
import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

CHILD_TIMEOUT_SECONDS = 20.0


def vid(handle: int) -> int:
    """Variable handle (negative, as the builders return) -> variable id."""
    return -(handle + 1)


def dist(a: int, b: int) -> float:
    """Asymmetric on purpose: a symmetric cost cannot tell a cyclic sum from twice an open one."""
    return 10.0 * a + b


DIST = np.array([[dist(a, b) for b in range(4)] for a in range(4)], dtype=np.float64)
HEAD = np.array([100.0 + e for e in range(4)], dtype=np.float64)
TAIL = np.array([1000.0 + e for e in range(4)], dtype=np.float64)


def _value(build: str, elements: list[int], cyclic: bool, endpoints: bool) -> float:
    """Evaluate one pair node at `elements`, which may be shorter than the list."""
    m = cbls.Model()
    lv = m.list_var(4, "seq")
    if build == "functor":
        node = m.pair_lambda_sum(
            lv,
            dist,
            cyclic=cyclic,
            head=(lambda e: 100.0 + e) if endpoints else None,
            tail=(lambda e: 1000.0 + e) if endpoints else None,
        )
    else:
        node = m.pair_table_sum(
            lv,
            DIST,
            cyclic=cyclic,
            head=HEAD if endpoints else None,
            tail=TAIL if endpoints else None,
        )
    m.minimize(node)
    m.close()
    m.var_mut(vid(lv)).elements = elements
    cbls.full_evaluate(m)
    return float(m.node_value(node))


@pytest.mark.parametrize("build", ["functor", "table"])
@pytest.mark.parametrize(
    ("elements", "cyclic", "endpoints", "expected"),
    [
        # n = 0: zero for every variant, endpoints included.
        ([], False, False, 0.0),
        ([], True, False, 0.0),
        ([], True, True, 0.0),
        # n = 1: no pair, and Cyclic adds none; endpoints both charge e_0.
        ([3], False, False, 0.0),
        ([3], True, False, 0.0),
        ([3], False, True, 103.0 + 1003.0),
        # n = 2: the one edge is traversed twice when cyclic.
        ([1, 2], False, False, 12.0),
        ([1, 2], True, False, 12.0 + 21.0),
        ([1, 2], False, True, 12.0 + 101.0 + 1002.0),
        # General n: d(2,0) + d(0,3) + d(3,1) = 20 + 3 + 31.
        ([2, 0, 3, 1], False, False, 54.0),
        ([2, 0, 3, 1], True, False, 54.0 + 12.0),
        ([2, 0, 3, 1], False, True, 54.0 + 102.0 + 1001.0),
        ([2, 0, 3, 1], True, True, 54.0 + 12.0 + 102.0 + 1001.0),
    ],
)
def test_pair_sum_matches_a_hand_computation(
    build: str, elements: list[int], cyclic: bool, endpoints: bool, expected: float
) -> None:
    got = _value(build, elements, cyclic=cyclic, endpoints=endpoints)
    assert got == expected


def test_the_functor_and_table_forms_agree_on_every_variant() -> None:
    elements = [3, 1, 0, 2]
    for cyclic in (False, True):
        for endpoints in (False, True):
            functor = _value("functor", elements, cyclic=cyclic, endpoints=endpoints)
            table = _value("table", elements, cyclic=cyclic, endpoints=endpoints)
            assert functor == table, (cyclic, endpoints)


def test_lambda_table_sum_matches_lambda_sum() -> None:
    table = np.array([1.0, 2.5, 4.0, 8.0], dtype=np.float64)
    m = cbls.Model()
    lv = m.list_var(4)
    by_table = m.lambda_table_sum(lv, table)
    by_functor = m.lambda_sum(lv, lambda e: float(table[e]))
    m.minimize(m.sum([by_table, by_functor]))
    m.close()
    m.var_mut(vid(lv)).elements = [2, 0, 3]
    cbls.full_evaluate(m)
    assert m.node_value(by_table) == 4.0 + 1.0 + 8.0
    assert m.node_value(by_table) == m.node_value(by_functor)


def test_a_table_is_copied_so_mutating_the_caller_s_array_does_not_change_the_model() -> None:
    table = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    m = cbls.Model()
    lv = m.list_var(4)
    node = m.lambda_table_sum(lv, table)
    m.minimize(node)
    m.close()
    table[:] = 99.0
    m.var_mut(vid(lv)).elements = [0, 1, 2, 3]
    cbls.full_evaluate(m)
    assert m.node_value(node) == 10.0


def _count_python_calls_during_solve(use_table: bool) -> tuple[int, float]:
    """Solve a small tour model and count the Python-level calls the search made."""
    cities = 9
    matrix = np.array(
        [[dist(a, b) for b in range(cities)] for a in range(cities)], dtype=np.float64
    )
    m = cbls.Model()
    lv = m.list_var(cities, "tour")
    if use_table:
        node = m.pair_table_sum(lv, matrix, cyclic=True)
    else:
        node = m.pair_lambda_sum(lv, dist, cyclic=True)
    m.minimize(node)
    m.close()

    config = cbls.SearchConfig()
    config.max_iterations = 50000

    calls = 0

    def profiler(frame: object, event: str, arg: object) -> None:
        nonlocal calls
        if event == "call":
            calls += 1

    sys.setprofile(profiler)
    try:
        # time_limit = 0 -> the iteration budget alone bounds the run, so the
        # two arms do the same amount of work and the counts are comparable.
        result = cbls.solve(m, 0.0, 42, config=config)
    finally:
        sys.setprofile(None)
    assert node >= 0
    return calls, float(result.objective)


def test_the_table_form_makes_no_python_calls_during_solve() -> None:
    functor_calls, functor_objective = _count_python_calls_during_solve(use_table=False)
    table_calls, table_objective = _count_python_calls_during_solve(use_table=True)

    # The functor arm is the control: if it did not call back into Python
    # either, the counter is measuring nothing.
    assert functor_calls > 1000, functor_calls
    assert table_calls == 0, table_calls
    # Same cost function, so the same trajectory and the same answer.
    assert table_objective == functor_objective


def _run_scenario(name: str) -> subprocess.CompletedProcess[str]:
    """Run one `__main__` scenario in a child interpreter, under a hard deadline."""
    env = dict(os.environ)
    # No conftest.py in the child to put the build directory on the path.
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


@pytest.mark.parametrize("scenario", ["bad_table", "out_of_universe_element"])
def test_a_bad_table_or_element_raises_instead_of_reading_past_the_table(scenario: str) -> None:
    proc = _run_scenario(scenario)
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} (negative = killed by that signal)\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def _scenario_bad_table() -> None:
    """Every way a mis-sized or mis-typed table reaches the engine is refused."""
    m = cbls.Model()
    lv = m.list_var(4)
    sv = m.set_var(5)
    b = m.bool_var()

    cases: list[tuple[str, Callable[[], object], type[Exception]]] = [
        ("short matrix", lambda: m.pair_table_sum(lv, np.zeros((3, 3))), ValueError),
        ("long matrix", lambda: m.pair_table_sum(lv, np.zeros((5, 5))), ValueError),
        ("non-square", lambda: m.pair_table_sum(lv, np.zeros((4, 3))), ValueError),
        ("1-D dist", lambda: m.pair_table_sum(lv, np.zeros(4)), TypeError),
        ("short head", lambda: m.pair_table_sum(lv, DIST, head=np.zeros(3)), ValueError),
        ("short tail", lambda: m.pair_table_sum(lv, DIST, tail=np.zeros(9)), ValueError),
        ("set universe", lambda: m.pair_table_sum(sv, DIST), ValueError),
        ("scalar var", lambda: m.pair_table_sum(b, DIST), ValueError),
        ("node handle", lambda: m.pair_table_sum(0, DIST), ValueError),
        ("stale handle", lambda: m.pair_table_sum(-100, DIST), IndexError),
        ("short lambda table", lambda: m.lambda_table_sum(lv, np.zeros(3)), ValueError),
        ("2-D lambda table", lambda: m.lambda_table_sum(lv, np.zeros((4, 4))), TypeError),
    ]
    for name, call, expected in cases:
        nodes_before = m.num_nodes()
        try:
            call()
        except expected:
            pass
        else:
            raise AssertionError(f"{name} was accepted")
        # A refused node leaves no trace, so the model is still usable.
        assert m.num_nodes() == nodes_before, name

    # The right shapes still work, on both variable types.
    m.pair_table_sum(lv, DIST, cyclic=True, head=HEAD, tail=TAIL)
    m.lambda_table_sum(sv, np.zeros(5))
    print("OK")


def _scenario_out_of_universe_element() -> None:
    """`Variable.elements` is writable from Python, so the lookup stays range-checked."""
    m = cbls.Model()
    lv = m.list_var(4, "seq")
    node = m.pair_table_sum(lv, DIST, cyclic=True, head=HEAD, tail=TAIL)
    m.minimize(node)
    m.close()

    for bad in ([0, 1, 9], [0, -3], [77]):
        m.var_mut(vid(lv)).elements = bad
        try:
            cbls.full_evaluate(m)
        except IndexError:
            continue
        raise AssertionError(f"elements {bad} were read past the table")

    # And the in-universe assignment still evaluates.
    m.var_mut(vid(lv)).elements = [0, 1, 2, 3]
    cbls.full_evaluate(m)
    assert m.node_value(node) == 1.0 + 12.0 + 23.0 + 30.0 + 100.0 + 1003.0
    print("OK")


if __name__ == "__main__":
    scenario = sys.argv[1]
    if scenario == "bad_table":
        _scenario_bad_table()
    elif scenario == "out_of_universe_element":
        _scenario_out_of_universe_element()
    else:
        raise SystemExit(f"unknown scenario {scenario}")
