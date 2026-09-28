"""Structural writes refused while a solve runs, and the lifetime fixes beside them.

`cbls.solve` and `ParallelSearch.solve_master` release the GIL for the whole
search and register the model while they run, so every structural write Python
can reach -- Model builders, Expr operators, close/freeze, a second solve --
raises RuntimeError until the solve returns. A solve writes structure itself
(the objective row, and solve_master's freeze), so the refusal protects every
solve.

Every scenario runs in a child process under a wall-clock timeout: a structural
write racing a running search, or an object outliving the model it points into,
CRASHES the interpreter rather than failing a test, and `solve_master`'s workers
acquire the GIL to call back into Python, so a mistake there DEADLOCKS it. The
child re-executes this file as a script (`__main__` block at the bottom).
"""

import os
import subprocess
import sys
import threading
from collections.abc import Callable
from typing import Any

import _cbls_core as cbls
import numpy as np

CHILD_TIMEOUT_SECONDS = 60.0


def _run_scenario(name: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    # The child has no conftest.py to put the build directory on the path.
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    # The assertions live in the child; an inherited -O would strip them.
    env["PYTHONOPTIMIZE"] = "0"
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), name],
        capture_output=True,
        text=True,
        timeout=CHILD_TIMEOUT_SECONDS,
        check=False,
        env=env,
    )


def _assert_scenario_ok(name: str) -> None:
    try:
        proc = _run_scenario(name)
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"scenario {name!r} did not finish within {CHILD_TIMEOUT_SECONDS}s: something "
            "holds the GIL across a search that needs it to call back into Python"
        ) from exc
    assert proc.returncode == 0, (
        f"scenario {name!r} exited {proc.returncode} (negative = killed by that signal)\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout
    # nanobind reports instances still alive at interpreter exit: a reference the
    # binding holds and the cycle collector cannot see.
    assert "nanobind: leaked" not in proc.stderr, proc.stderr


def _attempt(log: list[tuple[str, str]], name: str, fn: Callable[[], object]) -> None:
    """Run `fn`, logging the RuntimeError it raised -- or that it did not raise."""
    try:
        fn()
    except RuntimeError as e:
        log.append((name, str(e)))
        return
    log.append((name, "NOT REFUSED"))


def _two_var_model(with_list: bool = False) -> tuple[Any, Any, Any]:
    """x + y >= 3 over [0, 10]^2, minimizing x + y; plus an unused permutation List
    (variable 2) when asked, for the List-reading builders to name."""
    m = cbls.Model()
    x = m.Float(0, 10, "x")
    y = m.Float(0, 10, "y")
    if with_list:
        m.list_var(3)
    m.add_constraint(x + y >= 3.0)
    m.minimize(x + y)
    m.close()
    return m, x, y


def _scenario_structural_writes_are_refused_during_a_solve() -> None:
    """Model builders, close/freeze, Expr operators and a second solve all raise
    while cbls.solve runs on their model.

    The race this closes is a SECOND Python thread writing structure while the
    solve, GIL released, writes it too -- the objective row at the start. A
    thread race is not deterministic, so a
    SolveCallback is the stand-in: it runs on the search thread while the solve
    is registered, so every attempt from it takes exactly the path another
    thread's would -- the same registry check, under the GIL -- deterministically.
    """
    m, x, y = _two_var_model(with_list=True)
    lst = -3  # the List's handle: variable 2
    target = m.objective_id()
    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2_000
    # Solve once first so the objective row exists and the model is closed before
    # the probed solve: then the only thing that can refuse a write mid-solve is
    # the registry under test, and "nothing was written" below is exact.
    cbls.solve(m, time_limit=0.0, seed=1, config=cfg)
    attempts: list[tuple[str, str]] = []

    class Probe(cbls.SolveCallback):  # type: ignore[misc]
        done = False

        def on_progress(self, p: Any) -> None:
            if Probe.done:
                return
            Probe.done = True
            for name, fn in [
                ("Model.float_var", lambda: m.float_var(0, 1)),
                ("Model.constant", lambda: m.constant(1.0)),
                ("Model.add_constraint", lambda: m.add_constraint(target)),
                ("Model.freeze", lambda: m.freeze()),
                ("Model.Float", lambda: m.Float(0, 1)),
                ("Expr.__add__", lambda: x + y),
                ("Expr.__le__", lambda: x <= 1.0),
                ("cbls.sin", lambda: cbls.sin(x)),
                ("cbls.min", lambda: cbls.min([x, y])),
                ("Model.close", lambda: m.close()),
                ("Model.List", lambda: m.List(3)),
                ("Model.minimize", lambda: m.minimize(target)),
                ("Model.add_list_partition", lambda: m.add_list_partition([lst], "exact")),
                ("Model.lambda_sum", lambda: m.lambda_sum(lst, lambda i: 0.0)),
                ("Model.lambda_table_sum", lambda: m.lambda_table_sum(lst, np.zeros(3))),
                ("Model.pair_lambda_sum", lambda: m.pair_lambda_sum(lst, lambda i, j: 0.0)),
                ("Model.pair_table_sum", lambda: m.pair_table_sum(lst, np.zeros((3, 3)))),
                ("Expr.__rpow__", lambda: 2.0**x),
                ("cbls.if_then_else", lambda: cbls.if_then_else(x, x, y)),
                # A second solve writes structure too (objective row,
                # solve_master's freeze).
                ("cbls.solve", lambda: cbls.solve(m, time_limit=0.0, seed=2, config=cfg)),
                (
                    "solve_master",
                    lambda: cbls.ParallelSearch(1).solve_master(m, time_limit=0.0, seed=1),
                ),
            ]:
                _attempt(attempts, name, fn)

    n_vars, n_nodes = m.num_vars(), m.num_nodes()
    cbls.solve(m, time_limit=0.0, seed=1, callback=Probe(), config=cfg)
    assert len(attempts) == 21, attempts
    for name, message in attempts:
        assert "cbls.solve is running on this model" in message, (name, message)
    # Nothing was written: the objective row already existed.
    assert (m.num_vars(), m.num_nodes()) == (n_vars, n_nodes)
    # Scoped to the solve: afterwards the same writes work again.
    m.freeze()
    assert m.is_frozen()
    print("OK")


def _scenario_solve_master_registers_the_master() -> None:
    """Structural writes to the master raise while solve_master runs on it.

    Without the registration the master is merely frozen, and the refusal names
    that instead, so the message discriminates.
    """
    m, x, y = _two_var_model()
    attempts: list[tuple[str, str]] = []
    lock = threading.Lock()

    class Probe(cbls.SolveCallback):  # type: ignore[misc]
        def on_progress(self, p: Any) -> None:
            with lock:
                if attempts:
                    return
                _attempt(attempts, "Model.float_var", lambda: m.float_var(0, 1))
                _attempt(attempts, "Expr.__add__", lambda: x + y)

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2_000
    par = cbls.ParallelConfig()
    par.n_threads = 2
    cbls.ParallelSearch(2).solve_master(
        m, time_limit=0.0, seed=1, config=cfg, callback=Probe(), par_config=par
    )
    assert len(attempts) == 2, attempts
    for name, message in attempts:
        assert "cbls.solve is running on this model" in message, (name, message)
    print("OK")


def _scenario_an_expr_keeps_its_model_alive() -> None:
    """`cbls.Model().Float(0, 1)` used to leave the Expr pointing at a freed model.

    The Expr held a raw Model* and nothing tied the Python Model to it. Freed, the
    model's memory is reused by the next Model of the same size, so the stale
    pointer then reads (and writes) THAT model: `x.model` returned the new model
    and `x + 1.0` built a node in it. Checked on the direct result, on a chain of
    operators, and on the free functions.
    """
    import gc

    x = cbls.Model().Float(0, 1, "x")
    gc.collect()
    decoys = []
    for _ in range(64):
        d = cbls.Model()
        for _ in range(5):
            d.float_var(0, 1)
        decoys.append(d)
    assert x.model.num_vars() == 1, x.model.num_vars()
    before = [d.num_nodes() for d in decoys]
    e = (x + 1.0) * 2.0
    e = cbls.sin(e)
    e = cbls.max([e, x])
    assert [d.num_nodes() for d in decoys] == before, "a decoy model grew"
    del x
    gc.collect()
    decoys2 = [cbls.Model() for _ in range(64)]
    for d in decoys2:
        d.bool_var()
    assert e.model.num_vars() == 1
    assert e.model.num_nodes() >= 5
    print("OK")


def _scenario_a_violation_manager_keeps_its_model_alive() -> None:
    """ViolationManager holds a Model&; dropped, the model's arrays were read freed."""
    import gc

    def make() -> Any:
        m, _, _ = _two_var_model()
        return cbls.ViolationManager(m)

    vm = make()
    gc.collect()
    # Same shape, different right-hand side: a freed model's storage is reused by
    # one of these, and the dangling manager then reads ITS row (violation 100).
    decoys = []
    for _ in range(64):
        d = cbls.Model()
        dx = d.Float(0, 10)
        dy = d.Float(0, 10)
        d.add_constraint(dx + dy >= 100.0)
        d.minimize(dx + dy)
        d.close()
        cbls.full_evaluate(d)
        decoys.append(d)
    # x + y >= 3 at x = y = 0.
    assert vm.constraint_violation(0) == 3.0, vm.constraint_violation(0)
    print("OK")


def _scenario_solve_master_round_trips_and_reraises() -> None:
    """solve_master's own result and exception contract.

    A callback raising in the one worker comes out of the call as that
    exception, and the unwind releases the registration: a second solve_master
    on the same model runs, answers correctly, reports every worker completed
    and leaves the model frozen.
    """
    m, _, _ = _two_var_model()

    class CallbackError(ValueError):
        pass

    class Raiser(cbls.SolveCallback):  # type: ignore[misc]
        def on_progress(self, p: Any) -> None:
            raise CallbackError("every worker")

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 20_000
    par = cbls.ParallelConfig()
    par.n_threads = 1
    try:
        cbls.ParallelSearch(1).solve_master(
            m, time_limit=0.0, seed=1, config=cfg, callback=Raiser(), par_config=par
        )
    except CallbackError as e:
        assert "every worker" in str(e)
    else:
        raise AssertionError("solve_master swallowed an exception every worker raised")
    par.n_threads = 2
    r = cbls.ParallelSearch(2).solve_master(m, time_limit=0.0, seed=2, config=cfg, par_config=par)
    assert r.feasible, r
    assert abs(r.objective - 3.0) < 1e-6, r.objective
    assert r.workers_completed == 2, r.workers_completed
    assert m.is_frozen()
    print("OK")


SCENARIOS = {
    "master_round_trip": _scenario_solve_master_round_trips_and_reraises,
    "structural_writes": _scenario_structural_writes_are_refused_during_a_solve,
    "master_registers": _scenario_solve_master_registers_the_master,
    "expr_keeps_model": _scenario_an_expr_keeps_its_model_alive,
    "vm_keeps_model": _scenario_a_violation_manager_keeps_its_model_alive,
}


def test_structural_writes_are_refused_during_a_solve() -> None:
    _assert_scenario_ok("structural_writes")


def test_solve_master_refuses_structural_writes_to_the_master() -> None:
    _assert_scenario_ok("master_registers")


def test_an_expr_keeps_its_model_alive() -> None:
    _assert_scenario_ok("expr_keeps_model")


def test_a_violation_manager_keeps_its_model_alive() -> None:
    _assert_scenario_ok("vm_keeps_model")


def test_solve_master_round_trips_and_reraises() -> None:
    _assert_scenario_ok("master_round_trip")


if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
