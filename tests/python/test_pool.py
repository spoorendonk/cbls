"""Tests for ParallelSearch via Python bindings (#128).

Every scenario here runs in a **child interpreter**, not in-process. That is
deliberate: the bug these tests cover is a GIL deadlock, and a deadlocked
interpreter cannot fail a test. `pytest-timeout` cannot rescue it either --
its `thread` method runs `timeout_timer` on a `threading.Timer`, which is a
Python thread and so needs the GIL it is waiting on, and its `signal` method
sets a flag that only takes effect at the next bytecode in the main thread,
which is parked inside the C++ `join()`. Both mechanisms are starved by
exactly the condition they would have to report. A child process with a
wall-clock timeout is the only kind of deadline that still fires, so a
regression here fails the suite in ~20s instead of hanging it forever.

The child re-executes this file as a script (`__main__` block at the bottom),
so the scenarios stay next to the assertions that consume them.
"""

import os
import subprocess
import sys
import threading

import _cbls_core as cbls

# Generous relative to a 0.5s solve: this is a deadlock detector, not a
# performance floor, so it only has to be shorter than a developer's patience.
CHILD_TIMEOUT_SECONDS = 20.0


def _feasible_model() -> "cbls.Model":
    """x + y >= 3 over [0, 10]^2, minimizing x + y."""
    m = cbls.Model()
    x = m.float_var(0, 10)
    y = m.float_var(0, 10)
    neg1 = m.constant(-1.0)
    three = m.constant(3.0)
    m.add_constraint(m.sum([three, m.prod(neg1, x), m.prod(neg1, y)]))
    m.minimize(m.sum([x, y]))
    m.close()
    return m


def _run_scenario(name: str) -> subprocess.CompletedProcess[str]:
    """Run one `__main__` scenario in a child interpreter, under a hard deadline."""
    env = dict(os.environ)
    # The child imports _cbls_core directly, without conftest.py to place the
    # build directory on the path -- hand it this process's search path instead.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), name],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )


def _assert_scenario_ok(name: str) -> str:
    try:
        proc = _run_scenario(name)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"scenario {name!r} did not finish within {CHILD_TIMEOUT_SECONDS}s -- "
            "ParallelSearch is holding the GIL across its worker join again (#128)"
        ) from exc
    assert proc.returncode == 0, f"scenario {name!r} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_solve_accepts_a_python_model_factory() -> None:
    """A Python factory must be callable from the worker threads, not deadlock them."""
    out = _assert_scenario_ok("solve")
    assert "OK" in out


def test_solve_calls_the_factory_on_every_worker_thread() -> None:
    """Two workers means two factory calls, on two distinct threads.

    This is what the binding's docstring warns about, so it is worth asserting
    rather than describing: releasing the GIL is what lets the workers run the
    factory concurrently in the first place.
    """
    out = _assert_scenario_ok("solve")
    assert "distinct_factory_threads=2" in out


def test_solve_parallel_accepts_a_python_model_factory() -> None:
    """The full-featured overload carries the same call guard as the simple one."""
    out = _assert_scenario_ok("solve_parallel")
    assert "OK" in out


def test_solve_surfaces_a_raising_python_factory() -> None:
    """A factory that fails in every worker propagates the original Python exception.

    The C++ contract (tests/test_search.cpp, "ParallelSearch propagates a factory
    that fails in every worker") rethrows the parked exception on the caller; on
    the Python side nanobind must turn that back into the original ValueError,
    not a generic RuntimeError -- and not a hang, and not std::terminate.
    """
    out = _assert_scenario_ok("raising")
    assert "OK" in out


# --- Scenarios, executed in the child interpreter ---


def _scenario_solve() -> None:
    threads: set[int] = set()
    lock = threading.Lock()

    def factory() -> "cbls.Model":
        with lock:
            threads.add(threading.get_ident())
        return _feasible_model()

    result = cbls.ParallelSearch(2).solve(factory, 0.5, 42)
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    assert result.objective < 5.0, result.objective
    print(f"distinct_factory_threads={len(threads)}")


def _scenario_solve_parallel() -> None:
    result = cbls.ParallelSearch(2).solve_parallel(_feasible_model, 0.5, 42)
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    assert result.objective < 5.0, result.objective


def _scenario_raising() -> None:
    def factory() -> "cbls.Model":
        raise ValueError("python model factory failed")

    try:
        cbls.ParallelSearch(2).solve(factory, 0.5, 42)
    except ValueError as exc:
        # Assert the original type *and* message: solve_portfolio's "no result
        # and no error" guard on the same path would otherwise pass a type-only
        # check with the rethrow loop deleted.
        assert str(exc) == "python model factory failed", str(exc)
    else:
        raise AssertionError("a factory raising in every worker should have propagated")


if __name__ == "__main__":
    _scenarios = {
        "solve": _scenario_solve,
        "solve_parallel": _scenario_solve_parallel,
        "raising": _scenario_raising,
    }
    _scenarios[sys.argv[1]]()
    print("OK")
