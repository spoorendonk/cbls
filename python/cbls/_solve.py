"""`cbls.solve`: the C++ CLI's solve as one call (#464)."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

from . import _cbls_core as _core


@dataclass(frozen=True)
class SolveResult:
    """What `cbls.solve` returns: the engine's report and the best point found.

    `objective` is in the model's own sense, as the CLI prints it. The engine
    minimises, so a maximisation model's `search.objective`, its objective
    node's value and `SolveProgress.objective` are the negation (#472).
    `values[i]` and `elements[i]` belong to the variable with id `i`; the model
    itself is also left at that point. `search` is the full `SearchResult`
    (counters, portfolio worker accounting, `best_state`).
    """

    objective: float
    feasible: bool
    iterations: int
    time_seconds: float
    termination: _core.TerminationReason
    values: tuple[float, ...]
    elements: tuple[tuple[int, ...], ...]
    search: _core.SearchResult

    def value(self, var: int | _core.Expr | _core.Variable) -> float:
        """The value of one variable: its handle, a variable `Expr` or a `Variable`."""
        return self.values[_variable_id(var)]


def _variable_id(var: int | _core.Expr | _core.Variable) -> int:
    if isinstance(var, _core.Variable):
        return var.id
    if isinstance(var, _core.Expr):
        if not var.is_var():
            raise ValueError("SolveResult.value: the expression is not a variable")
        return var.var_id()
    if var >= 0:
        raise ValueError(f"SolveResult.value: {var} is not a variable handle (handles are < 0)")
    return -(var + 1)


def _effective_threads(threads: int) -> int:
    """`--threads`: 0 is one per core, as `std::thread::hardware_concurrency`."""
    if threads < 0:
        raise ValueError(f"cbls.solve: threads must be >= 0, got {threads}")
    return threads if threads > 0 else (os.cpu_count() or 1)


def _search_config(
    config: _core.SearchConfig | None, stop: _core.StopToken | None
) -> _core.SearchConfig:
    """`config` with `stop` attached to a copy: the caller's config is never changed."""
    if stop is None:
        return config if config is not None else _core.SearchConfig()
    attached = copy.copy(config) if config is not None else _core.SearchConfig()
    if attached.stop:
        raise ValueError("cbls.solve: pass the StopToken as stop= or config.stop, not both")
    attached.stop = stop
    return attached


def solve(
    model: _core.Model,
    time_limit: float = 10.0,
    seed: int = 42,
    threads: int = 0,
    lns: float | None = None,
    intensify: bool = False,
    callback: _core.SolveCallback | None = None,
    stop: _core.StopToken | None = None,
    config: _core.SearchConfig | None = None,
) -> SolveResult:
    """Solve a closed `model`, as the `cbls` CLI does, with its defaults.

    `threads` 1 runs the single-threaded search, any other count the parallel
    portfolio (0: one worker per core). `lns` is the LNS destroy fraction
    (None: off); `intensify` adds the float intensification hook. `stop` is
    attached to a copy of `config` (a fresh `SearchConfig` when None), which
    also holds `lns_interval`. The model is left at the best point found, which
    the result also carries by variable id; a portfolio with no worker result
    (cancelled before one finished) leaves it at its start, as the CLI does.
    An unclosed model raises ValueError: `nl_to_model` leaves an unsupported
    one so.

    The GIL is released while the search runs; see `cbls._cbls_core.solve` and
    `ParallelSearch.solve_master` for what that permits.
    """
    if not model.is_closed():
        raise ValueError(
            "cbls.solve: the model is not closed (an unsupported nl_to_model result "
            "is left so); call close() on a model you built"
        )
    if lns is not None and not lns > 0.0:
        raise ValueError(f"cbls.solve: lns must be a fraction > 0 or None, got {lns}")
    n_threads = _effective_threads(threads)
    search_config = _search_config(config, stop)
    if n_threads == 1:
        hook = _core.FloatIntensifyHook() if intensify else None
        destroy = _core.LNS(lns) if lns is not None else None
        result = _core.solve(
            model,
            time_limit,
            seed,
            hook,
            destroy,
            search_config.lns_interval,
            callback,
            search_config,
        )
    else:
        result = _solve_parallel(
            model, time_limit, seed, n_threads, lns, intensify, callback, search_config
        )
    return _result(model, result)


def _solve_parallel(
    model: _core.Model,
    time_limit: float,
    seed: int,
    n_threads: int,
    lns: float | None,
    intensify: bool,
    callback: _core.SolveCallback | None,
    config: _core.SearchConfig,
) -> _core.SearchResult:
    """The CLI's portfolio path; the winner is restored into `model`."""

    def hook_factory(_: _core.Model) -> _core.InnerSolverHook:
        return _core.FloatIntensifyHook()

    def lns_factory() -> _core.LNS:
        assert lns is not None
        return _core.LNS(lns)

    par_config = _core.ParallelConfig()
    par_config.n_threads = n_threads
    result = _core.ParallelSearch(n_threads).solve_master(
        model,
        time_limit,
        seed,
        config,
        hook_factory if intensify else None,
        lns_factory if lns is not None else None,
        callback,
        par_config,
    )
    # The workers searched their own copies: the master's assignment is untouched.
    state = result.best_state
    if len(state.values) == model.num_vars() and len(state.elements) == model.num_vars():
        model.restore_state(state)
        _core.full_evaluate(model)
    return result


def _result(model: _core.Model, result: _core.SearchResult) -> SolveResult:
    variables = [model.var(i) for i in range(model.num_vars())]
    return SolveResult(
        objective=-result.objective if model.is_maximizing() else result.objective,
        feasible=result.feasible,
        iterations=result.iterations,
        time_seconds=result.time_seconds,
        termination=result.termination,
        values=tuple(v.value for v in variables),
        elements=tuple(tuple(v.elements) for v in variables),
        search=result,
    )
