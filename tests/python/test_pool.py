"""Tests for ParallelSearch via Python bindings (#128, #129, #159).

Every scenario here runs in a **child interpreter**, not in-process. That is
deliberate: the bugs these tests cover are a GIL deadlock (#128), a double
free of a factory's return value (#129) and a Python exception released without
the GIL (#159), and none is reportable from inside
the process that hits it -- a deadlocked interpreter cannot fail a test, and an
aborting one takes the whole pytest run down with it. `pytest-timeout` cannot
rescue it either -- its `thread` method runs `timeout_timer` on a
`threading.Timer`, which is a Python thread and so needs the GIL it is waiting
on, and its `signal` method sets a flag that only takes effect at the next
bytecode in the main thread, which is parked inside the C++ `join()`. Both
mechanisms are starved by
exactly the condition they would have to report. A child process with a
wall-clock timeout is the only kind of deadline that still fires, so a
regression here fails the suite in ~20s instead of hanging it forever.

The child re-executes this file as a script (`__main__` block at the bottom),
so the scenarios stay next to the assertions that consume them.
"""

import gc
import os
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable

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
            "something is holding the GIL across ParallelSearch's worker join: "
            "either the call guard is gone (#128), or a factory's return value is "
            "being released on a thread that cannot acquire it (#129), or, for the "
            "callback-raising scenarios, a parked exception's release blocking on "
            "the GIL (#159)"
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
    assert "hook_calls=2 hooks_destroyed=2 hook_threads=2" in out


def test_solve_parallel_accepts_a_python_lns_factory() -> None:
    """The same round trip for lns_factory, which has its own lifetime (#129).

    Checked separately from hook_factory rather than folded into one scenario:
    the two arguments have the same shape but are built from different places --
    one per worker model, one per worker -- so a fix that only reached one of
    them would pass a combined test.
    """
    out = _assert_scenario_ok("lns_factory")
    assert "lns_calls=2 lns_destroyed=2 lns_threads=2" in out


def test_solve_parallel_builds_each_factory_once_per_worker_not_per_restart() -> None:
    """A worker restarts inside its own thread; its factories must not re-run.

    The portfolio no longer runs one `solve()` per worker -- a worker whose
    solve returns with budget left is restarted on the remaining clock, so the
    cores are never idle. The factory contract is the thing that quietly breaks
    under that change: moving the factory calls inside the restart loop would
    build a fresh Model, hook and LNS per restart, which is a deep copy of the
    model and, from Python, a GIL acquisition on every one. The scenario forces
    many restarts (a small max_iterations against a wall clock) and asserts the
    counts are still one per worker.

    Runs in a child like every other scenario here: the failure mode if the GIL
    handling regresses is a deadlock, which no in-process deadline can report.
    """
    out = _assert_scenario_ok("restarts")
    assert "factory_calls=2" in out
    assert "hook_builds=2" in out
    assert "lns_builds=2" in out
    # And the restarts actually happened -- without this the count assertions
    # above pass on a portfolio that ran one solve per worker and stopped.
    assert "restarted=True" in out


def test_solve_parallel_calls_a_python_callback_from_a_worker_thread() -> None:
    """The progress callback is the other Python path the GIL release unblocked.

    `SolveCallback` reaches C++ through a nanobind trampoline, which acquires the
    GIL from worker 0 -- so before the release it deadlocked for exactly the same
    reason `model_factory` did. Every worker now reports, through the portfolio's
    serializing wrapper, so what this pins is that no call lands on the CALLING
    thread; without this test a change that reverted the guard on `solve_parallel`
    alone would leave the other scenarios green.
    """
    out = _assert_scenario_ok("callback")
    assert "callback_all_calls_off_main_thread=True" in out


# The three tests below cover a Python `SolveCallback` whose `on_progress`
# raises during a portfolio run (#159). The portfolio parks the resulting
# `nb::python_error` and later rethrows or drops it with the GIL released; why
# that is safe is written above PySolveCallback in python/bindings.cpp. What
# these tests pin:
#
#   * where the exception ends up -- re-raised as the original object, or
#     discarded while the survivors' result is returned;
#   * that every exception raised is released EXACTLY once: the callback raises
#     a fresh `CallbackError` each time and its `__del__` records the thread it
#     died on, so a leaked `exception_ptr` shows as destroyed < raised, and a
#     double release as a crashed child -- which is also why these run in one;
#   * WHICH thread the last release happens on, because that is the path the
#     issue is about: `__del__` is Python code, so a release that did not hold
#     the GIL would run it with no thread state and crash the child rather than
#     record a thread.
#
# Red-checked in a throwaway copy, rebuilding `_cbls_core` once per break:
#   - nanobind's `~python_error` without its `gil_scoped_acquire`: all three
#     children segfault. Skipping the acquire ONLY on a thread that already owns
#     a (released) thread state -- i.e. only on the calling thread -- segfaults
#     `..._kills_one_worker` alone, which is the isolation that shows it is the
#     test reaching #159's calling-thread path;
#   - run_worker's retry removed (break on the first throw): all three red --
#     `..._raises_once` on `destroyed_on_calling_thread=`, the others on
#     `raises=`;
#   - the all-failed `std::rethrow_exception` dropped: `..._every_worker` red;
#   - each parked `exception_ptr` leaked in run_worker's catch: all three red,
#     on `destroyed=`.


def test_solve_parallel_surfaces_a_callback_that_raises_in_every_worker() -> None:
    """A callback that kills every worker re-raises the ORIGINAL exception.

    One worker, so "every worker" is deterministic. With more than one it is
    not: a peer invokes the callback only when it improves the portfolio-wide
    best, so on an easy model a callback that raises on every call typically
    kills worker 0 alone and the run returns normally -- which is the third test.

    The worker's solve raises on its first report, is restarted, and gives up
    after three consecutive failures: `raises=3` pins the documented retry bound.
    The first two exceptions are released on the worker thread as each retry
    replaces them; the third reaches the caller and dies there.
    """
    out = _assert_scenario_ok("callback_raises_always")
    assert "reraised_original=True" in out
    assert "raises=3 raising_threads=1" in out
    assert "destroyed=3 destroyed_on_calling_thread=1" in out


def test_solve_parallel_absorbs_a_callback_that_raises_once() -> None:
    """A single raise costs one worker one attempt; the run completes normally.

    The worker that raised restarts and finishes its run, so the exception is
    parked for the rest of that run and released on the worker's own thread
    when it returns -- never reaching the caller.
    """
    out = _assert_scenario_ok("callback_raises_once")
    assert "reraised_original=False" in out
    assert "raises=1 raising_threads=1" in out
    assert "destroyed=1 destroyed_on_calling_thread=0" in out


def test_solve_parallel_absorbs_a_callback_that_kills_one_worker() -> None:
    """One worker dead from its callback, one alive: no exception, a real result.

    This is the partial failure #159 is about. The dead worker's last exception
    is parked in the portfolio's per-worker failure slot, never rethrown, and
    released when that vector goes out of scope -- on the CALLING thread, whose
    GIL the binding's call guard has released. `destroyed_on_calling_thread=1`
    is what shows the scenario reached that path rather than a nearby one.

    Worker 0 is made the one that dies without knowing its thread: only the
    heartbeat worker delivers rows with `new_best=False` (a peer reports only a
    new portfolio-wide best), so the callback raises on the first such row and
    on every later row from the same thread. Worker 0 raises on its first
    non-improving row -- at once if the peer reached the best first, else at its
    ~1s heartbeat -- then again on the first report of each of its two retries;
    the peer never raises.
    """
    out = _assert_scenario_ok("callback_kills_one_worker")
    assert "reraised_original=False" in out
    assert "raises=3 raising_threads=1" in out
    assert "destroyed=3 destroyed_on_calling_thread=1" in out


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
    threads: set[int] = set()
    calls = 0
    destroyed = 0

    # _cbls_core is a compiled extension with no stubs, so mypy sees every symbol
    # in it as Any and strict mode refuses to subclass one -- same reason as the
    # Recorder callback below.
    # __del__ takes `lock`, which is safe only because no hook is ever dropped
    # by a thread already holding it: each worker builds its own and releases it
    # after the factory has returned. A future scenario that drops one inside a
    # `with lock:` block would self-deadlock and be reported as a 20s timeout.
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
            threads.add(threading.get_ident())
        return CountingHook()

    result = cbls.ParallelSearch(2).solve_parallel(
        _feasible_model, 0.5, 42, cbls.SearchConfig(), hook_factory
    )
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    # Every hook dies before solve_parallel returns -- at the end of the worker
    # lambda in portfolio mode -- so the counter is settled by the time it does,
    # and nothing here holds a reference that could keep one alive.
    with lock:
        n_calls, n_destroyed, n_threads = calls, destroyed, len(threads)
    # The thread count is what makes "once per worker" testable: two calls on a
    # single worker would satisfy hook_calls=2 on its own.
    print(f"hook_calls={n_calls} hooks_destroyed={n_destroyed} hook_threads={n_threads}")


def _scenario_lns_factory() -> None:
    lock = threading.Lock()
    threads: set[int] = set()
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
            threads.add(threading.get_ident())
        return CountingLNS(0.3)

    result = cbls.ParallelSearch(2).solve_parallel(
        _feasible_model, 0.5, 42, cbls.SearchConfig(), None, lns_factory
    )
    assert result.feasible, "portfolio found no feasible solution for x + y >= 3"
    with lock:
        n_calls, n_destroyed, n_threads = calls, destroyed, len(threads)
    print(f"lns_calls={n_calls} lns_destroyed={n_destroyed} lns_threads={n_threads}")


def _scenario_restarts() -> None:
    factory_calls = 0
    hook_builds = 0
    lns_builds = 0
    lock = threading.Lock()

    def factory() -> "cbls.Model":
        nonlocal factory_calls
        with lock:
            factory_calls += 1
        return _feasible_model()

    def hook_factory(model: "cbls.Model") -> "cbls.FloatIntensifyHook":
        nonlocal hook_builds
        with lock:
            hook_builds += 1
        return cbls.FloatIntensifyHook()

    def lns_factory() -> "cbls.LNS":
        nonlocal lns_builds
        with lock:
            lns_builds += 1
        return cbls.LNS(0.3)

    # A tight iteration cap against a real wall clock: each solve() returns
    # almost at once having exhausted max_iterations, and the worker is restarted
    # for the rest of the second. Without the restart loop the whole call returns
    # in milliseconds having spent 2 * 400 iterations.
    cap = 400
    n_threads = 2
    budget = 1.0
    config = cbls.SearchConfig()
    config.max_iterations = cap

    par = cbls.ParallelConfig()
    par.n_threads = n_threads

    result = cbls.ParallelSearch(n_threads).solve_parallel(
        factory, budget, 42, config, hook_factory, lns_factory, None, par
    )
    print(f"factory_calls={factory_calls}")
    print(f"hook_builds={hook_builds}")
    print(f"lns_builds={lns_builds}")
    print(f"restarted={result.iterations > 4 * n_threads * cap}")


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
    # A flag, not the count: how many rows arrive depends on which worker
    # improves first (a peer reports only a new portfolio-wide best), and a
    # count of 1 printed here failed under load when both workers reported.
    print("callback_all_calls_off_main_thread=True")


def _run_raising_callback(
    should_raise: Callable[[int, int, bool, set[int]], bool], n_threads: int, budget: float
) -> None:
    """Drive a portfolio whose callback raises, and report what became of it.

    `should_raise(call_index, thread, new_best, raising_threads)` decides each
    row. Prints the outcome for the test to assert on.
    """
    calling_thread = threading.get_ident()
    lock = threading.Lock()
    raising: list[int] = []
    # Appended from `__del__`, i.e. on whichever thread drops the last reference,
    # under the GIL -- list.append is atomic there, so no lock is taken (one
    # taken inside `__del__` could be re-entered by a release under that lock).
    destroyed: list[int] = []
    raised_ids: list[int] = []
    calls = 0

    class CallbackError(ValueError):
        def __init__(self, message: str) -> None:
            super().__init__(message)
            raised_ids.append(id(self))

        def __del__(self) -> None:
            destroyed.append(threading.get_ident())

    class Raiser(cbls.SolveCallback):  # type: ignore[misc]
        def on_progress(self, progress: "cbls.SolveProgress") -> None:
            nonlocal calls
            thread = threading.get_ident()
            with lock:
                decide = should_raise(calls, thread, progress.new_best, set(raising))
                calls += 1
                if decide:
                    raising.append(thread)
            if decide:
                raise CallbackError("python progress callback failed")

    par = cbls.ParallelConfig()
    par.n_threads = n_threads
    reraised_original = False
    try:
        result = cbls.ParallelSearch(n_threads).solve_parallel(
            _feasible_model, budget, 42, cbls.SearchConfig(), None, None, Raiser(), par
        )
    except CallbackError as caught:
        # The very object last raised (with one worker, the last raise is the one
        # re-raised), its message, and a traceback that still reaches into the
        # callback: a boundary that re-wrapped or re-created the exception must
        # not pass. Comparing `id`s is sound because `caught` is still alive.
        reraised_original = (
            id(caught) == raised_ids[-1]
            and str(caught) == "python progress callback failed"
            and any(f.name == "on_progress" for f in traceback.extract_tb(caught.__traceback__))
        )
    else:
        assert result.feasible, "the surviving worker should still have solved x + y >= 3"
    # `except ... as` unbinds `caught` on exit, so nothing here still holds one.
    # The collection is belt and braces: none of these exceptions is in a cycle
    # -- which is why `raise CallbackError(...)` stays inline above. Binding it to
    # a local first makes a frame <-> exception cycle and moves its release into
    # gc on the calling thread.
    gc.collect()
    on_calling = sum(1 for t in destroyed if t == calling_thread)
    print(f"reraised_original={reraised_original}")
    print(f"raises={len(raising)} raising_threads={len(set(raising))}")
    print(f"destroyed={len(destroyed)} destroyed_on_calling_thread={on_calling}")


def _scenario_callback_raises_always() -> None:
    _run_raising_callback(lambda _i, _t, _nb, _r: True, n_threads=1, budget=5.0)


def _scenario_callback_raises_once() -> None:
    # 1.0s, not 0.5s: the raising worker must still have budget for its retry, or
    # it completes no attempt and its exception moves to the calling thread.
    _run_raising_callback(lambda i, _t, _nb, _r: i == 0, n_threads=2, budget=1.0)


def _scenario_callback_kills_one_worker() -> None:
    # 2.5s: the row that identifies worker 0 comes at the latest ~1s after its
    # last report, and the peer needs clock left over to be the run's survivor.
    _run_raising_callback(
        lambda _i, t, new_best, raising: (not new_best) or t in raising,
        n_threads=2,
        budget=2.5,
    )


if __name__ == "__main__":
    _scenarios = {
        "solve": _scenario_solve,
        "solve_parallel": _scenario_solve_parallel,
        "raising": _scenario_raising,
        "hook_factory": _scenario_hook_factory,
        "lns_factory": _scenario_lns_factory,
        "restarts": _scenario_restarts,
        "callback": _scenario_callback,
        "callback_raises_always": _scenario_callback_raises_always,
        "callback_raises_once": _scenario_callback_raises_once,
        "callback_kills_one_worker": _scenario_callback_kills_one_worker,
    }
    _scenarios[sys.argv[1]]()
    print("OK")


def test_termination_reason_exposes_stopped() -> None:
    """`Stopped` is part of the enum a Python caller reads off a SearchResult.

    It was added when ParallelSearch gained a shared stop flag, and a binding
    that forgot it would leave Python unable to name the reason its own runs
    report -- silently, since nanobind simply would not create the attribute.

    Python cannot *produce* a Stopped result: SearchCoordination is not bound,
    by design, so the flag is reachable only from C++. The enum value is what
    Python needs, and it is what this pins.
    """
    assert hasattr(cbls.TerminationReason, "Stopped")
    members = {
        cbls.TerminationReason.TimeLimit,
        cbls.TerminationReason.IterationLimit,
        cbls.TerminationReason.Feasible,
        cbls.TerminationReason.NoBudget,
        cbls.TerminationReason.Stopped,
    }
    assert len(members) == 5, "a duplicated enum value would collapse this set"


def test_parallel_config_pool_capacity_round_trips() -> None:
    """The field is plumbed from here to the SolutionPool the workers share.

    Nothing else in the Python suite touches it, and a binding that dropped the
    property would fail at attribute-set time rather than anywhere visible.
    0 means auto (max(10, 2 * n_threads)); see include/cbls/pool.h.
    """
    par = cbls.ParallelConfig()
    assert par.pool_capacity == 0, "default is auto"
    par.pool_capacity = 4
    assert par.pool_capacity == 4


def test_adjacent_base_seeds_do_not_share_worker_streams() -> None:
    """Bumping --seed must actually give a different portfolio.

    The old scheme was `base + worker + restart * n_threads`: correct within a
    run, but at 12 workers seeds 42 and 43 shared 11 of their 12 base streams,
    so the standard way to draw an independent sample barely changed anything.
    Exposed to Python so the property is checked directly rather than inferred
    from two search trajectories.
    """
    workers = 12
    a = {cbls.portfolio_worker_seed(42, w, 0) for w in range(workers)}
    b = {cbls.portfolio_worker_seed(43, w, 0) for w in range(workers)}
    assert len(a) == workers, "no collisions within one run"
    assert len(b) == workers
    assert not (a & b), f"seeds 42 and 43 share {len(a & b)} of {workers} streams"
