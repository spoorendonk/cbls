"""Tests for ParallelSearch via Python bindings (#128, #129).

Every scenario here runs in a **child interpreter**, not in-process. That is
deliberate: the bugs these tests cover are a GIL deadlock (#128) and a double
free of a factory's return value (#129), and neither is reportable from inside
the process that hits it -- a deadlocked interpreter cannot fail a test, and an
aborting one takes the whole pytest run down with it. `pytest-timeout` cannot rescue it either --
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
    # Nearly every scenario assertion lives in the child, so an inherited
    # PYTHONOPTIMIZE would strip them and leave a child that prints OK and
    # exits 0 without having checked anything.
    env["PYTHONOPTIMIZE"] = "0"
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


def test_solve_parallel_accepts_a_python_hook_factory() -> None:
    """A Python hook_factory runs once per worker and its object is freed once (#129).

    The factory type used to hand back a raw owning pointer that src/pool.cpp
    adopted into a unique_ptr, so a nanobind-owned return value was deleted by
    C++ and again by the interpreter -- an unconditional SIGABRT, not a race.
    It now returns a shared_ptr, whose nanobind caster holds a Python reference
    and drops it under the GIL. Both regression modes are fatal to the process
    that hits them -- an abort, or a deadlock if the GIL handling is wrong --
    so this runs in a child, like every other scenario in this file.
    """
    out = _assert_scenario_ok("hook_factory")
    assert "hook_calls=2 hooks_destroyed=2" in out


def test_solve_parallel_accepts_a_python_lns_factory() -> None:
    """The same round trip for lns_factory, which has its own lifetime (#129).

    Checked separately from hook_factory rather than folded into one scenario:
    the two arguments have the same shape but are built from different places --
    one per worker model, one per worker -- so a fix that only reached one of
    them would pass a combined test.
    """
    out = _assert_scenario_ok("lns_factory")
    assert "lns_calls=2 lns_destroyed=2" in out


def test_solve_parallel_deterministic_builds_models_on_the_calling_thread() -> None:
    """Deterministic mode's factory contract is documented, so it is asserted.

    The docstring promises the factory runs on the calling thread before any
    worker starts -- the reason this mode never deadlocked the way portfolio
    mode did. Without a test, a change moving those calls into the workers
    would reintroduce #128 on this path with a green suite.
    """
    out = _assert_scenario_ok("deterministic")
    assert "factory_on_calling_thread=True" in out


def test_solve_parallel_calls_a_python_callback_from_a_worker_thread() -> None:
    """The progress callback is the other Python path the GIL release unblocked.

    `SolveCallback` reaches C++ through a nanobind trampoline, which acquires the
    GIL from worker 0 -- so before the release it deadlocked for exactly the same
    reason `model_factory` did. The docstring asserts the callback runs on worker
    0; without this test a change that reverted the guard on `solve_parallel`
    alone would leave the other scenarios green.
    """
    out = _assert_scenario_ok("callback")
    assert "callback_calls_off_main_thread=1" in out


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
        # Assert the original type *and* message, so a rethrow that loses the
        # original exception and re-wraps it cannot pass. (A type-only check
        # would already catch deleting the rethrow loop outright -- that path
        # throws std::runtime_error, which arrives as RuntimeError.)
        assert str(exc) == "python model factory failed", str(exc)
    else:
        raise AssertionError("a factory raising in every worker should have propagated")


def _scenario_hook_factory() -> None:
    lock = threading.Lock()
    calls = 0
    destroyed = 0

    # _cbls_core is a compiled extension with no stubs, so mypy sees every symbol
    # in it as Any and strict mode refuses to subclass one -- same reason as the
    # Recorder callback below.
    class CountingHook(cbls.FloatIntensifyHook):  # type: ignore[misc]
        def __del__(self) -> None:
            nonlocal destroyed
            with lock:
                destroyed += 1

    def hook_factory(model: "cbls.Model") -> "cbls.FloatIntensifyHook":
        nonlocal calls
        # The Model& reaches Python as a COPY, not as a handle on the worker's
        # own model: nanobind casts an lvalue reference with rv_policy::copy.
        # So the argument is worth type-checking and worthless to identity-check.
        assert isinstance(model, cbls.Model), type(model)
        with lock:
            calls += 1
        return CountingHook()

    result = cbls.ParallelSearch(2).solve_parallel(
        _feasible_model, 0.5, 42, cbls.SearchConfig(), hook_factory
    )
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    # Every hook dies before solve_parallel returns -- at the end of the worker
    # lambda in portfolio mode -- so the counter is settled by the time it does,
    # and nothing here holds a reference that could keep one alive.
    with lock:
        n_calls, n_destroyed = calls, destroyed
    print(f"hook_calls={n_calls} hooks_destroyed={n_destroyed}")


def _scenario_lns_factory() -> None:
    lock = threading.Lock()
    calls = 0
    destroyed = 0

    class CountingLNS(cbls.LNS):  # type: ignore[misc]
        def __del__(self) -> None:
            nonlocal destroyed
            with lock:
                destroyed += 1

    def lns_factory() -> "cbls.LNS":
        nonlocal calls
        with lock:
            calls += 1
        return CountingLNS(0.3)

    result = cbls.ParallelSearch(2).solve_parallel(
        _feasible_model, 0.5, 42, cbls.SearchConfig(), None, lns_factory
    )
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    with lock:
        n_calls, n_destroyed = calls, destroyed
    print(f"lns_calls={n_calls} lns_destroyed={n_destroyed}")


def _scenario_deterministic() -> None:
    calling_thread = threading.get_ident()
    threads: set[int] = set()

    def factory() -> "cbls.Model":
        threads.add(threading.get_ident())
        return _feasible_model()

    par = cbls.ParallelConfig()
    par.deterministic = True
    par.n_threads = 2
    par.max_epochs = 1
    par.epoch_iterations = 200
    cbls.ParallelSearch(2).solve_parallel(
        factory, 0.5, 42, cbls.SearchConfig(), None, None, None, par
    )
    print(f"factory_on_calling_thread={threads == {calling_thread}}")


def _scenario_callback() -> None:
    main_thread = threading.get_ident()
    idents: list[int] = []

    # _cbls_core is a compiled extension with no stubs, so mypy sees every symbol
    # in it as Any and strict mode refuses to subclass one. Scoped to this line
    # rather than relaxed in pyproject.toml, which would drop the check for every
    # base class in the suite.
    class Recorder(cbls.SolveCallback):  # type: ignore[misc]
        def on_progress(self, progress: "cbls.SolveProgress") -> None:
            idents.append(threading.get_ident())

    result = cbls.ParallelSearch(2).solve_parallel(
        _feasible_model, 0.5, 42, cbls.SearchConfig(), None, None, Recorder()
    )
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    assert idents, "the progress callback was never invoked"
    off_main = [i for i in idents if i != main_thread]
    assert len(off_main) == len(idents), f"callback ran on the calling thread: {idents}"
    print(f"callback_calls_off_main_thread={len(off_main)}")


if __name__ == "__main__":
    _scenarios = {
        "solve": _scenario_solve,
        "solve_parallel": _scenario_solve_parallel,
        "raising": _scenario_raising,
        "hook_factory": _scenario_hook_factory,
        "lns_factory": _scenario_lns_factory,
        "deterministic": _scenario_deterministic,
        "callback": _scenario_callback,
    }
    _scenarios[sys.argv[1]]()
    print("OK")
