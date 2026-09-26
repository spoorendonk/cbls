"""`cbls.StopToken`: cancelling a running solve from another Python thread (#169).

Every scenario here runs in a **child interpreter**, for the reason
`tests/python/test_pool.py` sets out at length and this file inherits: the
failure mode is a starved interpreter, and no in-process deadline can report
one. `cbls.solve` did not release the GIL before #169, so a Python thread that
wanted to call `StopToken.request()` mid-solve could not run at all -- the
regression this guards against is exactly a deadlocked child, where the solve
runs its whole budget while the canceller thread waits for a GIL it never gets.
`pytest-timeout` cannot see that (its `thread` method needs the same GIL; its
`signal` method only takes effect at the next bytecode in a main thread parked
inside a C++ call), so the deadline is a wall clock on a subprocess.

The C++ side of the same feature is `tests/test_stop.cpp`, which covers the
`TerminationReason::Cancelled` contract without an interpreter in the way.
"""

import os
import subprocess
import sys
import threading

import _cbls_core as cbls

# Generous against the ~1s solves below: this is a deadlock detector, not a
# performance floor.
CHILD_TIMEOUT_SECONDS = 30.0

# Far more GLS iterations than a cancelled run can reach, so "the stop ended it"
# is provable from the iteration count rather than from the wall clock -- the
# #104 discipline the C++ tests follow.
#
# Sized with a wide margin on purpose. `_scenario_cancel_running_solve` raises the
# token on a 0.5s timer, so a budget the search could EXHAUST inside that window
# would make the scenario report IterationLimit and fail with no other symptom --
# a flake. At this size an unpolled stop instead runs into the child's own
# wall-clock deadline, which `_assert_scenario_ok` reports by name.
UNREACHABLE_ITERATIONS = 200_000_000


def _quadratic() -> "cbls.Model":
    """x + y >= 1 over [-5, 5]^2, minimizing x^2 + y^2.

    An objective model, so the search never exits on `Feasible` and keeps
    tightening its bound instead of converging to a stop of its own.
    """
    m = cbls.Model()
    x = m.float_var(-5, 5)
    y = m.float_var(-5, 5)
    two = m.constant(2)
    neg1 = m.constant(-1.0)
    one = m.constant(1.0)
    m.add_constraint(m.sum([one, m.prod(neg1, x), m.prod(neg1, y)]))
    m.minimize(m.sum([m.pow_expr(x, two), m.pow_expr(y, two)]))
    m.close()
    return m


def _run_scenario(name: str) -> subprocess.CompletedProcess[str]:
    """Run one `__main__` scenario in a child interpreter, under a hard deadline."""
    env = dict(os.environ)
    # The child imports _cbls_core directly, without conftest.py to place the
    # build directory on the path -- hand it this process's search path instead.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    # Every scenario assertion lives in the child, so an inherited
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
            "cbls.solve is holding the GIL for the duration of the C++ call, so the "
            "thread that would raise the StopToken never runs (#169), or the stop is "
            "not being polled at all and the run is grinding its iteration budget"
        ) from exc
    assert proc.returncode == 0, f"scenario {name!r} failed:\n{proc.stdout}\n{proc.stderr}"
    return proc.stdout


def test_a_stop_token_cancels_a_running_solve() -> None:
    """A token raised from another Python thread ends `solve` with `Cancelled`.

    This is the whole feature: it needs the GIL release AND the stop poll, and
    it fails differently for each -- a hung child for the first, a
    `Cancelled`-less result for the second.
    """
    out = _assert_scenario_ok("cancel_running_solve")
    assert "OK" in out


def test_a_pre_raised_token_cancels_before_any_work() -> None:
    """The degenerate case, and the one that proves the reason is not a budget."""
    out = _assert_scenario_ok("cancel_before_start")
    assert "OK" in out


def test_an_unattached_stop_leaves_the_run_to_its_budget() -> None:
    """The control. Without it the two above pass on a solve that always cancels."""
    out = _assert_scenario_ok("no_stop")
    assert "OK" in out


def test_a_stop_token_cancels_a_portfolio() -> None:
    """`ParallelConfig.stop` reaches every worker."""
    out = _assert_scenario_ok("cancel_portfolio")
    assert "OK" in out


def test_stop_reads_back_as_attached() -> None:
    """The property round-trips as a bool, and None detaches.

    In-process: no thread and no solve, so nothing here can deadlock.
    """
    config = cbls.SearchConfig()
    assert config.stop is False
    token = cbls.StopToken()
    config.stop = token
    assert config.stop is True
    config.stop = None
    assert config.stop is False

    par = cbls.ParallelConfig()
    assert par.stop is False
    par.stop = token
    assert par.stop is True


def test_a_token_reports_and_resets_its_own_flag() -> None:
    """In-process: the token is a flag, and nothing about it needs a solve."""
    token = cbls.StopToken()
    assert not token.requested()
    token.request()
    assert token.requested()
    token.reset()
    assert not token.requested()


# --- child-process scenarios ------------------------------------------------


def _scenario_cancel_running_solve() -> None:
    token = cbls.StopToken()
    config = cbls.SearchConfig()
    config.max_iterations = UNREACHABLE_ITERATIONS
    config.stop = token

    # Raise the token shortly after the solve begins. A timer rather than a
    # sleep in the main thread: the main thread is inside the C++ call, which is
    # the point -- only a released GIL lets this one run at all.
    threading.Timer(0.5, token.request).start()

    result = cbls.solve(_quadratic(), 0.0, 7, config=config)
    assert result.termination == cbls.TerminationReason.Cancelled, result.termination
    assert 0 < result.iterations < UNREACHABLE_ITERATIONS, result.iterations


def _scenario_cancel_before_start() -> None:
    token = cbls.StopToken()
    token.request()
    config = cbls.SearchConfig()
    config.max_iterations = UNREACHABLE_ITERATIONS
    config.stop = token

    result = cbls.solve(_quadratic(), 0.0, 7, config=config)
    assert result.termination == cbls.TerminationReason.Cancelled, result.termination
    assert result.iterations < UNREACHABLE_ITERATIONS, result.iterations


def _scenario_no_stop() -> None:
    config = cbls.SearchConfig()
    config.max_iterations = 5000
    result = cbls.solve(_quadratic(), 0.0, 7, config=config)
    assert result.termination == cbls.TerminationReason.IterationLimit, result.termination
    assert result.iterations >= 5000, result.iterations


def _scenario_cancel_portfolio() -> None:
    token = cbls.StopToken()
    par = cbls.ParallelConfig()
    par.n_threads = 2
    par.stop = token

    threading.Timer(0.5, token.request).start()

    search = cbls.ParallelSearch(2)
    # A 60s clock the run must not reach: a portfolio that ignored the token
    # would come back TimeLimit a minute from now, or not at all.
    result = search.solve_parallel(
        _quadratic,
        60.0,
        11,
        cbls.SearchConfig(),
        None,
        None,
        None,
        par,
    )
    assert result.termination == cbls.TerminationReason.Cancelled, result.termination


if __name__ == "__main__":
    _scenarios = {
        "cancel_running_solve": _scenario_cancel_running_solve,
        "cancel_before_start": _scenario_cancel_before_start,
        "no_stop": _scenario_no_stop,
        "cancel_portfolio": _scenario_cancel_portfolio,
    }
    _scenarios[sys.argv[1]]()
    print("OK")
