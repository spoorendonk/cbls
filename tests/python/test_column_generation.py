"""Column generation from Python (#168), and the lifetime fixes it needed.

A `cbls.ColumnGenerator` subclass is called by the engine at safe points between
batches, from the search thread, with the GIL released by `cbls.solve` -- so it
is exactly the "Python callable handed to C++" case CLAUDE.md's testing rules
are about. Three kinds of test live here:

- in-process: the single-threaded `cbls.solve` paths, which spawn no thread and
  whose failure mode is an ordinary wrong answer or exception;
- child process: anything that could CRASH the interpreter (a view retained past
  the call it was lent for, an `Expr` outliving its model, a structural write
  racing a running search) or DEADLOCK it (`ParallelSearch.solve_master`, whose
  workers acquire the GIL to price and clone). A crashed or deadlocked
  interpreter cannot fail a test; a child with a wall-clock timeout can.

The child re-executes this file as a script (`__main__` block at the bottom).

The cutting-stock instance is Falkenauer (1996) class "u" u120_00, as in
`tests/test_column_generation.cpp`: 120 items, capacity 150, optimum 48. Copied
from the OR-Library distribution, J.E. Beasley, `binpack1.txt`
(https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/binpack1.txt, the first
of its 20 instances), in the file's order.
"""

import math
import os
import subprocess
import sys
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import _cbls_core as cbls
import pytest

CHILD_TIMEOUT_SECONDS = 60.0

U120_CAPACITY = 150
U120_OPTIMUM = 48
U120_ITEMS = (
    42, 69, 67, 57, 93, 90, 38, 36, 45, 42, 33, 79, 27, 57, 44, 84, 86, 92, 46, 38, 85, 33, 82, 73,
    49, 70, 59, 23, 57, 72, 74, 69, 33, 42, 28, 46, 30, 64, 29, 74, 41, 49, 55, 98, 80, 32, 25, 38,
    82, 30, 35, 39, 57, 84, 62, 50, 55, 27, 30, 36, 20, 78, 47, 26, 45, 41, 58, 98, 91, 96, 73, 84,
    37, 93, 91, 43, 73, 85, 81, 79, 71, 80, 76, 83, 41, 78, 70, 23, 42, 87, 43, 84, 60, 55, 49, 78,
    73, 62, 36, 44, 94, 69, 32, 96, 70, 84, 58, 78, 25, 80, 58, 66, 83, 24, 98, 60, 42, 43, 43, 39,
)  # fmt: skip


@dataclass(frozen=True)
class CuttingStock:
    """One demand row per DISTINCT item size, sizes descending."""

    capacity: int
    sizes: tuple[int, ...]
    demand: tuple[int, ...]


def u120_00() -> CuttingStock:
    counts = Counter(U120_ITEMS)
    sizes = tuple(sorted(counts, reverse=True))
    return CuttingStock(U120_CAPACITY, sizes, tuple(counts[s] for s in sizes))


@dataclass
class CuttingModel:
    """Int x_p per pattern, `sum_p a_ip x_p >= d_i` per size, minimize sum_p x_p.

    Row i is constraint index i. Built from the trivial pattern set: one item of
    one size per roll, so the only objective it admits is 120.
    """

    m: Any
    row_sums: list[int] = field(default_factory=list)
    objective_sum: int = -1


def build_trivial(cs: CuttingStock) -> CuttingModel:
    cm = CuttingModel(cbls.Model())
    m = cm.m
    xs = []
    for d in cs.demand:
        x = m.int_var(0, d)
        cm.row_sums.append(m.sum([m.prod(m.constant(1.0), x)]))
        xs.append(x)
    for i, d in enumerate(cs.demand):
        m.add_constraint(m.geq(cm.row_sums[i], m.constant(float(d))))
    cm.objective_sum = m.sum(xs)
    m.minimize(cm.objective_sum)
    m.close()
    return cm


def best_pattern(cs: CuttingStock, value: list[float]) -> list[int]:
    """Bounded knapsack: max sum_i v_i a_i s.t. sum_i w_i a_i <= capacity.

    Each unit of each size is a 0/1 item, as in the C++ test, so the tie-breaking
    (and the whole trajectory) is the same one.
    """
    units = [
        (i, cs.sizes[i], value[i])
        for i in range(len(cs.sizes))
        for _ in range(min(cs.demand[i], cs.capacity // cs.sizes[i]))
    ]
    cap = cs.capacity
    dp = [0.0] * (cap + 1)
    take = [bytearray(cap + 1) for _ in units]
    for u, (_, w, v) in enumerate(units):
        row = take[u]
        for c in range(cap, w - 1, -1):
            with_u = dp[c - w] + v
            if with_u > dp[c]:
                dp[c] = with_u
                row[c] = 1
    a = [0] * len(cs.sizes)
    c = cap
    for u in range(len(units) - 1, -1, -1):
        if take[u][c]:
            size_idx, w, _ = units[u]
            a[size_idx] += 1
            c -= w
    return a


class KnapsackPricer(cbls.ColumnGenerator):  # type: ignore[misc]
    """The C++ test's pricer, in Python.

    A pattern `a` of cost 1 changes the weighted violation by about
    `W_obj - sum_i W_i a_i`, so the column worth adding is the knapsack optimum
    under values `W_i` whenever it beats `W_obj`. Up to PER_CALL columns a call;
    after each, the values of the sizes it used are halved, so the next knapsack
    looks elsewhere.
    """

    PER_CALL = 4

    def __init__(self, cs: CuttingStock, row_sums: list[int], objective_sum: int) -> None:
        super().__init__()
        self.cs = cs
        self.row_sums = row_sums
        self.objective_sum = objective_sum
        self.seeded = False
        self.events: list[Any] = []

    def clone(self) -> "KnapsackPricer":
        return KnapsackPricer(self.cs, self.row_sums, self.objective_sum)

    def price(self, ctx: Any, why: Any, ext: Any) -> None:
        self.events.append(why)
        sigs = ctx.signatures
        if not self.seeded:
            # The base model's columns count as duplicates too.
            for i in range(len(self.cs.sizes)):
                sigs.insert([(i, 1.0)], 1.0)
            self.seeded = True
        weights = ctx.weights
        value = weights[: len(self.cs.sizes)]
        w_obj = weights[ctx.objective_constraint_idx]
        room = min(self.PER_CALL, ctx.columns_remaining)
        for _ in range(self.PER_CALL):
            if room <= 0:
                break
            a = best_pattern(self.cs, value)
            gain = sum(v * k for v, k in zip(value, a, strict=True))
            if gain <= w_obj:
                break
            for i, k in enumerate(a):
                if k > 0:
                    value[i] *= 0.5
            if not sigs.insert([(i, float(k)) for i, k in enumerate(a) if k], 1.0):
                continue  # already a column
            self.stage(a, ext)
            room -= 1

    def stage(self, a: list[int], ext: Any) -> None:
        ub = max((d + k - 1) // k for d, k in zip(self.cs.demand, a, strict=True) if k > 0)
        x = ext.int_var(0, ub)
        for i, k in enumerate(a):
            if k > 0:
                ext.append_to_sum(self.row_sums[i], ext.prod(ext.constant(float(k)), x))
        ext.append_to_sum(self.objective_sum, x)


def _iteration_budget(iterations: int) -> Any:
    cfg = cbls.SearchConfig()
    cfg.max_iterations = iterations
    return cfg


def _solve(m: Any, seed: int, cfg: Any, callback: Any = None) -> Any:
    # time_limit 0.0 with max_iterations: an iteration-budgeted, clockless run,
    # so the trajectory -- the pricer's decisions included -- is deterministic.
    return cbls.solve(m, time_limit=0.0, seed=seed, callback=callback, config=cfg)


# ---------------------------------------------------------------------------
# In-process: the single-threaded paths.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_python_knapsack_pricer_beats_the_trivial_pattern_set(seed: int) -> None:
    """Same instance, budget and pricer as the C++ acceptance test.

    Also the proof that the engine's OWN extend still runs while the Python
    registry refuses every structural write to this model from Python: every
    column counted here went in through it.
    """
    cs = u120_00()
    cap = 400
    plain = build_trivial(cs)
    without = _solve(plain.m, seed, _iteration_budget(30_000))

    grown = build_trivial(cs)
    base_vars = grown.m.num_vars()
    cfg = _iteration_budget(30_000)
    pricer = KnapsackPricer(cs, grown.row_sums, grown.objective_sum)
    cfg.column_generator = pricer
    cfg.pricing_period = 5
    cfg.max_generated_columns = cap
    with_pricer = _solve(grown.m, seed, cfg)

    assert without.feasible and with_pricer.feasible
    assert without.objective == 120  # the trivial set admits nothing else
    assert U120_OPTIMUM <= with_pricer.objective < without.objective
    # Held to what the C++ measurement reaches at this budget (77-84 over these
    # seeds, docs/architecture.md), not merely "below 120".
    assert with_pricer.objective <= 90
    c = with_pricer.counters
    assert 0 < c.columns_added <= cap
    assert grown.m.num_vars() - base_vars == c.columns_added
    assert c.pricing_calls > 0
    assert c.pricing_seconds == 0.0, "clockless run: no pricing clock either"
    assert c.extensions_refused == 0
    # The prototype is never priced with: the engine prices a clone.
    assert pricer.events == []


def test_the_pricer_sees_the_configured_schedule() -> None:
    """Periodic every pricing_period batches, NewBest on an improving batch, and a
    context whose fields describe the model at that point."""
    cs = u120_00()
    seen: list[tuple[Any, int, int, int, float, float, bool]] = []

    class Recorder(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> "Recorder":
            return Recorder()

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            seen.append(
                (
                    why,
                    ctx.batches,
                    len(ctx.weights),
                    len(ctx.constraint_ids()),
                    ctx.elapsed_seconds,
                    ctx.remaining_seconds,
                    ctx.incumbent is not None,
                )
            )
            assert ext.empty()
            assert 0 <= ctx.objective_constraint_idx < len(ctx.weights)
            assert ctx.num_vars() == len(cs.sizes)
            assert ctx.var_value(0) == ctx.var_value(0)
            assert ctx.constraint_violation(0) >= 0.0
            assert ctx.node_value(ctx.constraint_ids()[0]) == ctx.node_value(
                ctx.constraint_ids()[0]
            )

    # Periodic only.
    cm = build_trivial(cs)
    cfg = _iteration_budget(20_000)
    cfg.column_generator = Recorder()
    cfg.pricing_period = 3
    cfg.price_on_stagnation = False
    result = _solve(cm.m, 1, cfg)
    assert seen, "nothing priced"
    assert {s[0] for s in seen} == {cbls.PricingEvent.Periodic}
    assert [s[1] for s in seen] == list(range(3, 3 * len(seen) + 1, 3))
    for _, _, n_weights, n_rows, elapsed, remaining, _ in seen:
        assert n_weights == n_rows == len(cs.sizes) + 1  # the demand rows + the objective row
        assert math.isnan(elapsed) and remaining == math.inf  # clockless
    assert result.counters.pricing_calls == len(seen)

    # NewBest only: every call is on an improving batch, before its weight reset,
    # so an incumbent always exists by then.
    seen.clear()
    cm = build_trivial(cs)
    cfg = _iteration_budget(20_000)
    cfg.column_generator = Recorder()
    cfg.price_on_new_best = True
    cfg.price_on_stagnation = False
    _solve(cm.m, 1, cfg)
    assert seen and {s[0] for s in seen} == {cbls.PricingEvent.NewBest}
    assert all(s[6] for s in seen)


def test_a_timed_run_reports_a_clock_to_the_pricer() -> None:
    cs = u120_00()
    clocks: list[tuple[float, float]] = []

    class Clock(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> "Clock":
            return Clock()

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            clocks.append((ctx.elapsed_seconds, ctx.remaining_seconds))

    cm = build_trivial(cs)
    cfg = cbls.SearchConfig()
    cfg.column_generator = Clock()
    cfg.pricing_period = 1
    cfg.price_on_stagnation = False
    cbls.solve(cm.m, time_limit=0.3, seed=1, config=cfg)
    assert clocks
    for elapsed, remaining in clocks:
        assert 0.0 <= elapsed <= 0.3 + 0.1
        assert 0.0 < remaining <= 0.3


class PricerError(Exception):
    pass


def test_an_exception_in_price_propagates_as_that_exception() -> None:
    """Out of cbls.solve unchanged, with nothing the call staged applied."""
    cs = u120_00()
    cm = build_trivial(cs)

    class Raiser(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> "Raiser":
            return Raiser()

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            x = ext.int_var(0, 3)
            ext.append_to_sum(cm.objective_sum, x)
            raise PricerError("from price")

    n_vars = cm.m.num_vars()
    cfg = _iteration_budget(20_000)
    cfg.column_generator = Raiser()
    cfg.pricing_period = 1
    with pytest.raises(PricerError, match="from price"):
        _solve(cm.m, 1, cfg)
    assert cm.m.num_vars() == n_vars
    # The solve registration was released on the way out: the model is buildable
    # and solvable again.
    ext = cbls.ModelExtension(cm.m)
    ext.append_to_sum(cm.objective_sum, ext.int_var(0, 1))
    cm.m.extend(ext)
    cfg.column_generator = None
    assert _solve(cm.m, 1, cfg).feasible


def test_clone_is_checked_and_its_exceptions_propagate() -> None:
    cs = u120_00()

    class CloneRaises(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> Any:
            raise PricerError("from clone")

    class CloneSelf(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> Any:
            return self

    class CloneWrongType(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> Any:
            return object()

    class NoOverrides(cbls.ColumnGenerator):  # type: ignore[misc]
        pass

    cases: list[tuple[Any, type[BaseException], str]] = [
        (CloneRaises(), PricerError, "from clone"),
        (CloneSelf(), ValueError, "returned self"),
        (CloneWrongType(), TypeError, "must return a cbls.ColumnGenerator"),
        (NoOverrides(), NotImplementedError, "clone must be overridden"),
    ]
    for generator, exc, match in cases:
        cm = build_trivial(cs)
        cfg = _iteration_budget(1_000)
        cfg.column_generator = generator
        with pytest.raises(exc, match=match):
            _solve(cm.m, 1, cfg)


def test_search_config_pricing_fields_round_trip() -> None:
    cfg = cbls.SearchConfig()
    assert cfg.column_generator is None
    assert cfg.pricing_period == 0
    assert cfg.price_on_stagnation is True
    assert cfg.price_on_new_best is False
    assert cfg.max_generated_columns == 10_000
    assert cfg.column_retire_age == 0

    cs = u120_00()
    cm = build_trivial(cs)
    pricer = KnapsackPricer(cs, cm.row_sums, cm.objective_sum)
    cfg.column_generator = pricer
    assert cfg.column_generator is pricer
    cfg.pricing_period = 7
    cfg.price_on_stagnation = False
    cfg.price_on_new_best = True
    cfg.max_generated_columns = 12
    cfg.column_retire_age = 3
    assert (cfg.pricing_period, cfg.price_on_stagnation, cfg.price_on_new_best) == (7, False, True)
    assert (cfg.max_generated_columns, cfg.column_retire_age) == (12, 3)
    cfg.column_generator = None
    assert cfg.column_generator is None
    with pytest.raises(TypeError, match="cbls.ColumnGenerator or None"):
        cfg.column_generator = object()


def test_counters_report_zero_pricing_without_a_generator() -> None:
    cm = build_trivial(u120_00())
    c = _solve(cm.m, 1, _iteration_budget(2_000)).counters
    assert (c.pricing_calls, c.columns_added, c.rows_added, c.columns_retired) == (0, 0, 0, 0)
    assert (c.extensions_refused, c.incumbents_revalidated, c.pricing_seconds) == (0, 0, 0.0)


# ---------------------------------------------------------------------------
# Child process: scenarios that can crash or deadlock the interpreter.
# ---------------------------------------------------------------------------


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
            "holds the GIL across a search that needs it to price or clone"
        ) from exc
    assert proc.returncode == 0, (
        f"scenario {name!r} exited {proc.returncode} (negative = killed by that signal)\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


def _expect(exc: type[BaseException], match: str, fn: Any, *args: Any) -> None:
    try:
        fn(*args)
    except exc as e:
        assert match in str(e), f"{fn}: {e}"
        return
    raise AssertionError(f"{fn}{args} did not raise {exc.__name__}")


def _scenario_retained_views_raise_after_the_call() -> None:
    """A context, signature set or extension kept past price() raises, not crashes.

    They are the ENGINE's objects, on its stack for one call. Checked both in a
    later call of the same solve (the stale lease must not read the new call's
    objects) and after the solve has returned (the frame is gone).
    """
    cs = u120_00()
    cm = build_trivial(cs)
    kept: dict[str, Any] = {}
    stale_uses_in_later_call: list[str] = []
    calls: list[int] = []

    class Keeper(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> "Keeper":
            return Keeper()

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            calls.append(1)
            if not kept:
                kept.update(ctx=ctx, ext=ext, sigs=ctx.signatures)
                return
            try:
                kept["ext"].int_var(0, 1)
            except RuntimeError as e:
                stale_uses_in_later_call.append(str(e))

    cfg = _iteration_budget(5_000)
    cfg.column_generator = Keeper()
    cfg.pricing_period = 1
    cfg.price_on_stagnation = False
    _solve(cm.m, 1, cfg)
    assert len(calls) >= 2, "the pricer was called only once"
    # Without the lease the stale view is not merely stale: the next call's
    # engine extension sits at the same stack address, and the write lands in it.
    assert stale_uses_in_later_call, "a view kept from an earlier call was used without raising"
    assert "used after the ColumnGenerator.price call" in stale_uses_in_later_call[0]

    ctx, ext, sigs = kept["ctx"], kept["ext"], kept["sigs"]
    expired = "used after the ColumnGenerator.price call"
    for fn, args in [
        (lambda: ctx.weights, ()),
        (lambda: ctx.batches, ()),
        (lambda: ctx.incumbent, ()),
        (lambda: ctx.signatures, ()),
        (ctx.num_vars, ()),
        (ctx.constraint_ids, ()),
        (ctx.var_value, (0,)),
        (ctx.node_value, (0,)),
        (ctx.constraint_violation, (0,)),
        (ext.int_var, (0, 1)),
        (ext.constant, (1.0,)),
        (ext.append_to_sum, (cm.objective_sum, -1)),
        (ext.add_constraint, (0,)),
        (ext.empty, ()),
        (ext.num_new_vars, ()),
        (sigs.insert, ([(0, 1.0)], 1.0)),
        (sigs.contains, ([(0, 1.0)], 1.0)),
        (len, (sigs,)),
    ]:
        _expect(RuntimeError, expired, fn, *args)
    # Handing the expired extension to extend raises too, rather than applying a
    # recording the engine has since destroyed.
    _expect(RuntimeError, expired, cm.m.extend, ext)
    print("OK")


def _attempt(log: list[tuple[str, str]], name: str, fn: Callable[[], object]) -> None:
    """Run `fn`, logging the RuntimeError it raised -- or that it did not raise."""
    try:
        fn()
    except RuntimeError as e:
        log.append((name, str(e)))
        return
    log.append((name, "NOT REFUSED"))


def _two_var_model() -> tuple[Any, Any, Any]:
    m = cbls.Model()
    x = m.Float(0, 10, "x")
    y = m.Float(0, 10, "y")
    m.add_constraint(x + y >= 3.0)
    m.minimize(x + y)
    m.close()
    return m, x, y


def _scenario_structural_writes_are_refused_during_a_solve() -> None:
    """ModelExtension construction and builders, Model builders and Expr operators
    all raise while cbls.solve runs on their model.

    The race this closes is a SECOND Python thread writing structure while the
    solve, GIL released, writes it too -- the objective row at the start, a
    pricing extend at any batch. A thread race is not deterministic, so a
    SolveCallback is the stand-in: it runs on the search thread while the solve
    is registered, so every attempt from it takes exactly the path another
    thread's would -- the same registry check, under the GIL -- deterministically.
    """
    m, x, y = _two_var_model()
    target = m.objective_id()
    # One extension recorded BEFORE the solve, to check its builders mid-solve.
    early = cbls.ModelExtension(m)
    early_var = early.float_var(0, 1)
    attempts: list[tuple[str, str]] = []

    class Probe(cbls.SolveCallback):  # type: ignore[misc]
        done = False

        def on_progress(self, p: Any) -> None:
            if Probe.done:
                return
            Probe.done = True
            for name, fn in [
                ("ModelExtension()", lambda: cbls.ModelExtension(m)),
                ("early.float_var", lambda: early.float_var(0, 1)),
                ("early.append_to_sum", lambda: early.append_to_sum(target, early_var)),
                ("early.constant", lambda: early.constant(2.0)),
                ("Model.float_var", lambda: m.float_var(0, 1)),
                ("Model.constant", lambda: m.constant(1.0)),
                ("Model.add_constraint", lambda: m.add_constraint(target)),
                ("Model.freeze", lambda: m.freeze()),
                ("Model.Float", lambda: m.Float(0, 1)),
                ("Expr.__add__", lambda: x + y),
                ("Expr.__le__", lambda: x <= 1.0),
                ("cbls.sin", lambda: cbls.sin(x)),
                ("cbls.min", lambda: cbls.min([x, y])),
                ("Model.extend", lambda: m.extend(early)),
            ]:
                _attempt(attempts, name, fn)

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2_000
    n_vars, n_nodes = m.num_vars(), m.num_nodes()
    cbls.solve(m, time_limit=0.0, seed=1, callback=Probe(), config=cfg)
    assert len(attempts) == 14, attempts
    for name, message in attempts:
        assert "cbls.solve is running on this model" in message, (name, message)
    # A query of the recording reads no model and is not refused.
    assert early.num_new_vars() == 1
    # Nothing was written: solve adds its objective row (two nodes) and nothing else.
    assert (m.num_vars(), m.num_nodes()) == (n_vars, n_nodes + 2)
    # Scoped to the solve: afterwards the same writes work again.
    ext = cbls.ModelExtension(m)
    ext.add_constraint(ext.leq(ext.float_var(0, 1), ext.constant(1.0)))
    m.extend(ext)
    _ = x + y
    print("OK")


def _scenario_only_the_lent_extension_is_accepted_in_price() -> None:
    """Inside price(), on the solve thread: building another extension, a Model
    builder, and applying the lent extension by hand are all refused; building
    into the lent one works, and the engine applies it when price returns."""
    m, _, _ = _two_var_model()
    in_price: list[tuple[str, str]] = []

    class Pricer(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> "Pricer":
            return Pricer()

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            if in_price:
                return
            _attempt(in_price, "ModelExtension()", lambda: cbls.ModelExtension(m))
            _attempt(in_price, "Model.bool_var", lambda: m.bool_var())
            _attempt(in_price, "Model.extend(lent)", lambda: m.extend(ext))
            z = ext.int_var(0, 1)
            ext.add_constraint(ext.leq(z, ext.constant(1.0)))

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 2_000
    cfg.column_generator = Pricer()
    cfg.pricing_period = 1
    result = cbls.solve(m, time_limit=0.0, seed=1, config=cfg)
    assert len(in_price) == 3, in_price
    for name, message in in_price:
        assert "cbls.solve is running on this model" in message, (name, message)
    assert (result.counters.columns_added, result.counters.rows_added) == (1, 1)
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


def _scenario_solve_master_prices_with_one_clone_per_worker() -> None:
    """Under ParallelSearch.solve_master every worker clones the prototype once,
    on its own thread, and prices with that clone; the master comes back grown."""
    cs = u120_00()
    cm = build_trivial(cs)
    base_vars = cm.m.num_vars()
    lock = threading.Lock()
    clones: list[int] = []
    pricers: set[int] = set()
    threads: set[int] = set()

    class Tracked(KnapsackPricer):
        serial = 0
        serial_id = 0

        def clone(self) -> "Tracked":
            with lock:
                Tracked.serial += 1
                c = Tracked(self.cs, self.row_sums, self.objective_sum)
                c.serial_id = Tracked.serial
                clones.append(c.serial_id)
            return c

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            with lock:
                pricers.add(getattr(self, "serial_id", 0))
                threads.add(threading.get_ident())
            super().price(ctx, why, ext)

    workers = 3
    cfg = cbls.SearchConfig()
    cfg.max_iterations = 20_000
    cfg.column_generator = Tracked(cs, cm.row_sums, cm.objective_sum)
    cfg.pricing_period = 5
    cfg.max_generated_columns = 200
    par = cbls.ParallelConfig()
    par.n_threads = workers
    result = cbls.ParallelSearch(workers).solve_master(
        cm.m, time_limit=0.0, seed=3, config=cfg, par_config=par
    )
    assert result.feasible
    assert sorted(clones) == list(range(1, workers + 1)), clones
    assert pricers == set(clones), (pricers, clones)  # the prototype (0) never priced
    assert threading.get_ident() not in threads
    assert result.objective < 120
    grown = cm.m.num_vars() - base_vars
    assert 0 < grown <= result.counters.columns_added
    assert not cm.m.is_frozen(), "the winner's grown model replaces the master"
    # Registered for the call only.
    cbls.ModelExtension(cm.m)
    print("OK")


def _scenario_solve_master_reraises_when_every_worker_raises() -> None:
    cs = u120_00()
    cm = build_trivial(cs)

    class Raiser(cbls.ColumnGenerator):  # type: ignore[misc]
        def clone(self) -> "Raiser":
            return Raiser()

        def price(self, ctx: Any, why: Any, ext: Any) -> None:
            raise PricerError("every worker")

    cfg = cbls.SearchConfig()
    cfg.max_iterations = 20_000
    cfg.column_generator = Raiser()
    cfg.pricing_period = 1
    par = cbls.ParallelConfig()
    par.n_threads = 2
    try:
        cbls.ParallelSearch(2).solve_master(
            cm.m, time_limit=0.0, seed=1, config=cfg, par_config=par
        )
    except PricerError as e:
        assert "every worker" in str(e)
    else:
        raise AssertionError("solve_master swallowed an exception every worker raised")
    print("OK")


def _scenario_the_factory_overload_refuses_a_generator() -> None:
    cs = u120_00()
    cm = build_trivial(cs)
    cfg = cbls.SearchConfig()
    cfg.max_iterations = 1_000
    cfg.column_generator = KnapsackPricer(cs, cm.row_sums, cm.objective_sum)
    m = cm.m
    m.freeze()
    _expect(
        ValueError,
        "column_generator",
        lambda: cbls.ParallelSearch(2).solve_parallel(
            lambda: m, time_limit=0.0, seed=1, config=cfg
        ),
    )
    print("OK")


SCENARIOS = {
    "retained_views": _scenario_retained_views_raise_after_the_call,
    "structural_writes": _scenario_structural_writes_are_refused_during_a_solve,
    "lent_extension_only": _scenario_only_the_lent_extension_is_accepted_in_price,
    "expr_keeps_model": _scenario_an_expr_keeps_its_model_alive,
    "master_clones": _scenario_solve_master_prices_with_one_clone_per_worker,
    "master_reraises": _scenario_solve_master_reraises_when_every_worker_raises,
    "factory_refuses": _scenario_the_factory_overload_refuses_a_generator,
}


def test_a_view_retained_past_price_raises_instead_of_crashing() -> None:
    _assert_scenario_ok("retained_views")


def test_model_extension_is_refused_during_a_solve() -> None:
    _assert_scenario_ok("structural_writes")


def test_only_the_lent_extension_is_accepted_inside_price() -> None:
    _assert_scenario_ok("lent_extension_only")


def test_an_expr_keeps_its_model_alive() -> None:
    _assert_scenario_ok("expr_keeps_model")


def test_solve_master_prices_with_one_clone_per_worker() -> None:
    _assert_scenario_ok("master_clones")


def test_solve_master_reraises_a_pricer_exception_every_worker_raised() -> None:
    _assert_scenario_ok("master_reraises")


def test_the_factory_overload_refuses_a_generator() -> None:
    _assert_scenario_ok("factory_refuses")


if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
