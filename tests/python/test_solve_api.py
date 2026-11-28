"""The public `cbls` package (#464): `__all__` and `cbls.solve`.

`cbls.solve` with threads != 1 runs the portfolio, which calls Python factories
and callbacks from C++ worker threads: those scenarios run in a child
interpreter under a timeout, as `tests/python/test_pool.py` explains.
"""

from __future__ import annotations

import subprocess
import sys

import cbls
import pytest
from cbls import _cbls_core
from support import run_scenario

CHILD_TIMEOUT_SECONDS = 60.0
# The portfolio restarts its workers until the clock runs out, so an iteration
# cap does not end it: its scenarios run on a short budget instead.
PORTFOLIO_SECONDS = 1.0


def _quadratic() -> tuple[cbls.Model, cbls.Expr, cbls.Expr]:
    """x + y >= 3, x integer in [0, 10], y in [-1.5, 2.5]; min x^2 + y^2."""
    m = cbls.Model()
    x = m.Int(0, 10, "x")
    y = m.Float(-1.5, 2.5, "y")
    m.add_constraint(x + y >= 3.0)
    m.minimize(x * x + y * y)
    m.close()
    return m, x, y


def _capped(iterations: int = 20_000) -> cbls.SearchConfig:
    config = cbls.SearchConfig()
    config.max_iterations = iterations
    return config


def test_all_names_resolve_and_star_import_gives_exactly_them() -> None:
    assert len(set(cbls.__all__)) == len(cbls.__all__)
    for name in cbls.__all__:
        assert hasattr(cbls, name), name
    namespace: dict[str, object] = {}
    exec("from cbls import *", namespace)
    assert set(namespace) - {"__builtins__"} == set(cbls.__all__)


def test_internals_are_reachable_only_through_cbls_core() -> None:
    for name in ("ViolationManager", "apply_move", "undo_move", "compute_partial", "MoveChange"):
        assert hasattr(_cbls_core, name), name
        assert name not in cbls.__all__
        assert not hasattr(cbls, name), name


def test_the_public_surface_covers_model_solve_callbacks_and_io() -> None:
    expected = {
        "Model",
        "Variable",
        "Expr",
        "sin",
        "solve",
        "SolveResult",
        "SearchConfig",
        "SearchResult",
        "SolveCallback",
        "SolveProgress",
        "StopToken",
        "load_model",
        "save_model",
        "read_mps",
        "mps_to_model",
        "read_nl",
        "nl_to_model",
    }
    assert expected <= set(cbls.__all__)


def test_solve_single_thread_returns_values_by_variable() -> None:
    m, x, y = _quadratic()
    result = cbls.solve(m, time_limit=60.0, threads=1, config=_capped())
    assert result.feasible
    assert len(result.values) == m.num_vars() == 2
    assert result.value(x) == m.var(x.var_id()).value == result.values[x.var_id()]
    assert result.value(m.var(y.var_id())) == result.values[y.var_id()]
    assert result.value(x.handle) == result.value(x)
    assert result.elements == ((), ())
    assert result.objective == pytest.approx(result.value(x) ** 2 + result.value(y) ** 2)
    assert result.termination == cbls.TerminationReason.IterationLimit
    assert result.search.iterations == result.iterations


def test_solve_single_thread_is_reproducible_by_seed() -> None:
    first = cbls.solve(_quadratic()[0], time_limit=60.0, seed=7, threads=1, config=_capped())
    second = cbls.solve(_quadratic()[0], time_limit=60.0, seed=7, threads=1, config=_capped())
    assert first.values == second.values
    assert first.iterations == second.iterations


def test_solve_with_lns_and_intensify_on_one_thread() -> None:
    m, _, _ = _quadratic()
    result = cbls.solve(m, time_limit=60.0, threads=1, lns=0.5, intensify=True, config=_capped())
    assert result.feasible


def test_solve_honours_a_raised_stop_token() -> None:
    token = cbls.StopToken()
    token.request()
    result = cbls.solve(_quadratic()[0], time_limit=60.0, threads=1, stop=token)
    assert result.termination == cbls.TerminationReason.Cancelled


def test_solve_calls_the_callback() -> None:
    class Recorder(cbls.SolveCallback):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def on_progress(self, progress: cbls.SolveProgress) -> None:
            self.calls += 1

    recorder = Recorder()
    cbls.solve(_quadratic()[0], time_limit=60.0, threads=1, callback=recorder, config=_capped())
    assert recorder.calls > 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threads": -1}, "threads"),
        ({"lns": 0.0}, "lns"),
        ({"lns": -0.5}, "lns"),
    ],
)
def test_solve_rejects_bad_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cbls.solve(_quadratic()[0], **kwargs)  # type: ignore[arg-type]


def test_solve_attaches_stop_to_a_copy_not_the_callers_config() -> None:
    config = _capped()
    token = cbls.StopToken()
    token.request()
    for _ in range(2):  # a second call must not see the first call's token
        result = cbls.solve(_quadratic()[0], time_limit=60.0, threads=1, stop=token, config=config)
        assert result.termination == cbls.TerminationReason.Cancelled
    assert not config.stop
    unstopped = cbls.solve(_quadratic()[0], time_limit=60.0, threads=1, config=config)
    assert unstopped.termination == cbls.TerminationReason.IterationLimit


def test_solve_refuses_an_unclosed_model() -> None:
    m = cbls.Model()
    x = m.Int(0, 10, "x")
    m.minimize(x + 1.0)
    with pytest.raises(ValueError, match="not closed"):
        cbls.solve(m, threads=1)


def test_builtin_named_functions_stay_out_of_star_import() -> None:
    for name in ("abs", "max", "min", "pow", "round"):
        assert name not in cbls.__all__
        assert getattr(cbls, name) is getattr(_cbls_core, name)


def test_solve_refuses_two_stop_tokens() -> None:
    config = cbls.SearchConfig()
    config.stop = cbls.StopToken()
    with pytest.raises(ValueError, match="not both"):
        cbls.solve(_quadratic()[0], stop=cbls.StopToken(), config=config)


def test_solve_result_value_rejects_a_non_variable() -> None:
    m = cbls.Model()
    x = m.Int(0, 10, "x")
    total = x + 1.0
    m.minimize(total)
    m.close()
    result = cbls.solve(m, time_limit=60.0, threads=1, config=_capped(100))
    with pytest.raises(ValueError, match="not a variable"):
        result.value(total)
    with pytest.raises(ValueError, match="handle"):
        result.value(0)


def _assert_scenario_ok(name: str) -> None:
    try:
        proc = run_scenario(__file__, name, CHILD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(f"scenario {name!r} hung past {CHILD_TIMEOUT_SECONDS}s") from exc
    assert proc.returncode == 0, f"scenario {name!r} failed:\n{proc.stdout}\n{proc.stderr}"
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def test_solve_parallel_restores_the_winner_into_the_model() -> None:
    _assert_scenario_ok("parallel_restores_winner")


def test_solve_parallel_with_lns_intensify_and_callback() -> None:
    _assert_scenario_ok("parallel_with_factories")


def test_solve_with_threads_zero_takes_a_path_and_solves() -> None:
    _assert_scenario_ok("threads_zero")


def _scenario_parallel_restores_winner() -> None:
    m, x, y = _quadratic()
    result = cbls.solve(m, time_limit=PORTFOLIO_SECONDS, threads=2)
    assert result.feasible
    assert result.search.workers_launched == 2
    assert result.values == (m.var(x.var_id()).value, m.var(y.var_id()).value)
    assert result.objective == pytest.approx(result.value(x) ** 2 + result.value(y) ** 2)
    assert m.node_value(m.objective_id()) == pytest.approx(result.objective)


def _scenario_parallel_with_factories() -> None:
    class Recorder(cbls.SolveCallback):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def on_progress(self, progress: cbls.SolveProgress) -> None:
            self.calls += 1

    recorder = Recorder()
    m, _, _ = _quadratic()
    result = cbls.solve(
        m, time_limit=PORTFOLIO_SECONDS, threads=2, lns=0.5, intensify=True, callback=recorder
    )
    assert result.feasible
    assert result.search.workers_completed == 2
    assert recorder.calls > 0


def _scenario_threads_zero() -> None:
    result = cbls.solve(_quadratic()[0], time_limit=PORTFOLIO_SECONDS, threads=0)
    assert result.feasible


if __name__ == "__main__":
    _scenarios = {
        "parallel_restores_winner": _scenario_parallel_restores_winner,
        "parallel_with_factories": _scenario_parallel_with_factories,
        "threads_zero": _scenario_threads_zero,
    }
    _scenarios[sys.argv[1]]()
    print("OK")
