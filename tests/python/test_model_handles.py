"""Handle validation and G_v through the Model bindings (#156).

Python hands the int32 handle API raw integers, so a typo'd or stale handle
reaches the engine unchecked by any type system. Since #156 flattened the DAG's
edges into CSR arrays, such a handle is no longer caught late by a throwing
accessor: the back-reference rebuild would count it into an offsets array and
write past the end -- silent heap corruption, a segfault on a far-out id. The
model now rejects it when the node naming it is built.

The rejection is asserted in a CHILD interpreter, following `test_pool.py`: the
failure being guarded against is a crash, and a crashing scenario run in-process
would take the whole pytest run down instead of failing one test.

`constraints_of_var` is custom binding code -- the C++ accessor returns a view,
which the binding copies out to a list -- so it is covered here too.
"""

import os
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import _cbls_core as cbls
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
    # No conftest.py in the child to put the build directory on the path.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    # The scenario's assertions live in the child; keep them.
    env["PYTHONOPTIMIZE"] = "0"
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), name],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )


# -2 is the first variable handle past a one-variable model: the boundary of the
# variable-id check, which the far-out ids alone would not pin.
@pytest.mark.parametrize("handle", [5, 100_000, -2, -100_000])
def test_an_out_of_range_child_handle_raises_instead_of_corrupting_the_model(
    handle: int,
) -> None:
    proc = _run_scenario(f"bad_child:{handle}")
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} (negative = killed by that signal)\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_an_out_of_range_constraint_or_objective_handle_raises() -> None:
    proc = _run_scenario("bad_root")
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_an_empty_min_or_max_raises_instead_of_reading_past_its_children() -> None:
    proc = _run_scenario("empty_min_max")
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_a_short_state_or_weight_vector_raises_instead_of_reading_past_it() -> None:
    proc = _run_scenario("short_state_or_weights")
    assert proc.returncode == 0, (
        f"child exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_constraints_of_var_lists_ascending_constraint_indices() -> None:
    m = cbls.Model()
    x = m.float_var(0, 1)
    y = m.float_var(0, 1)
    one = m.constant(1.0)
    m.add_constraint(m.leq(m.sum([x, y]), one))  # 0: x, y
    m.add_constraint(m.leq(y, one))  # 1: y
    m.add_constraint(m.leq(m.prod(x, x), one))  # 2: x
    m.close()
    assert m.constraints_of_var(vid(x)) == [0, 2]
    assert m.constraints_of_var(vid(y)) == [0, 1]
    with pytest.raises(IndexError):
        m.constraints_of_var(2)
    with pytest.raises(IndexError):
        m.constraints_of_var(-1)


def _scenario_bad_child(handle: int) -> None:
    """Every way a child handle enters the model rejects an out-of-range one."""
    builders: dict[str, Callable[[Any, int], object]] = {
        "neg": lambda m, h: m.neg(h),
        "sum": lambda m, h: m.sum([m.constant(1.0), h]),
        "min_expr": lambda m, h: m.min_expr([h]),
        "prod": lambda m, h: m.prod(m.constant(1.0), h),
        "leq": lambda m, h: m.leq(h, m.constant(1.0)),
        "if_then_else": lambda m, h: m.if_then_else(m.constant(1.0), h, h),
    }
    for name, build in builders.items():
        m = cbls.Model()
        x = m.float_var(0, 1)
        n_nodes_before = m.num_nodes()
        try:
            build(m, handle)
        except IndexError:
            pass
        else:
            raise AssertionError(f"{name}: handle {handle} was accepted")
        # A rejected node leaves no trace, so the model is still usable. The
        # sum's first child is built before the bad one is seen, hence <= 1.
        assert m.num_nodes() - n_nodes_before <= 1, name
        f = m.sum([x, m.constant(2.0)])
        m.minimize(f)
        m.close()
        m.var_mut(vid(x)).value = 1.0
        assert cbls.full_evaluate(m) == 3.0, name
    print("OK")


def _scenario_short_state_or_weights() -> None:
    """Two more vectors a caller sizes and the engine then indexes unchecked."""
    m = cbls.Model()
    x = m.float_var(0, 1)
    one = m.constant(1.0)
    m.add_constraint(m.leq(x, one))
    m.minimize(m.sum([x]))
    m.close()

    state = m.copy_state()
    state.elements = []  # values still has one entry per variable
    try:
        m.restore_state(state)
    except ValueError:
        pass
    else:
        raise AssertionError("restore_state accepted a short elements list")

    vm = cbls.ViolationManager(m)
    for bad in ([], [1.0, 1.0]):
        try:
            vm.weights = bad
        except ValueError:
            continue
        raise AssertionError(f"weights accepted {len(bad)} entries for 1 constraint")
    vm.weights = [2.0]  # the right length still works
    assert list(vm.weights) == [2.0]
    vm.weighted_violation_delta(vid(x), 0.9)
    print("OK")


def _scenario_bad_root() -> None:
    m = cbls.Model()
    x = m.float_var(0, 1)
    f = m.sum([x, m.constant(1.0)])
    for bad in (m.num_nodes(), 100_000):
        for name, call in (
            ("add_constraint", m.add_constraint),
            ("minimize", m.minimize),
            ("maximize", m.maximize),
        ):
            try:
                call(bad)
            except IndexError:
                continue
            raise AssertionError(f"{name}({bad}) was accepted")
    m.minimize(f)
    m.close()
    print("OK")


def _scenario_empty_min_max() -> None:
    """Min and Max read their first child unchecked, so neither may be empty."""
    builders: dict[str, Callable[[Any], object]] = {
        "min_expr": lambda m: m.min_expr([]),
        "max_expr": lambda m: m.max_expr([]),
        "cbls.min": lambda m: cbls.min([]),
        "cbls.max": lambda m: cbls.max([]),
    }
    # The two Model builders guard the DAG read (children[0] of an empty slice);
    # cbls.min/cbls.max guard an earlier one, args[0] on an empty vector, which
    # never reaches the DAG at all.
    for name, build in builders.items():
        m = cbls.Model()
        m.float_var(0, 1)
        try:
            build(m)
        except ValueError:
            continue
        raise AssertionError(f"{name}([]) was accepted")
    print("OK")


if __name__ == "__main__":
    scenario = sys.argv[1]
    if scenario.startswith("bad_child:"):
        _scenario_bad_child(int(scenario.split(":", 1)[1]))
    elif scenario == "bad_root":
        _scenario_bad_root()
    elif scenario == "empty_min_max":
        _scenario_empty_min_max()
    elif scenario == "short_state_or_weights":
        _scenario_short_state_or_weights()
    else:
        raise SystemExit(f"unknown scenario {scenario}")
