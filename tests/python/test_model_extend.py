"""Growing a closed model from Python: ModelExtension and Model.extend (#167).

Two halves.

The refusals run in a CHILD interpreter, following `test_model_handles.py`: the
builder takes raw int32 handles, and the failure each check guards against is a
crash -- an unvalidated handle in the CSR splice, a cyclic append leaving nodes
out of the topological order, a ViolationManager indexed past its weights --
which in-process would take the whole pytest run down instead of failing one
test. Each scenario that refuses an operation on the model asserts the Python
exception AND that the model is still usable afterwards, since a refusal that
left the model half-grown would be a crash deferred to the next call.

The round trips run in-process: build a model whole, build the same model as a
closed base plus an extension, and compare everything Python can observe --
node values of every row and of the objective, `constraints_of_var` for every
variable, and the values again under fresh assignments through both
`full_evaluate` and `delta_evaluate`. The C++ property test compares the
internal structure (parents, dependents, topological validity); this pins that
the binding hands the same operations through.
"""

import math
import os
import random
import resource
import subprocess
import sys
from dataclasses import dataclass, field
from functools import partial
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


def _expect_raises(exc: type[BaseException], call: "Callable[[], object]", what: str) -> None:
    try:
        call()
    except exc:
        return
    except Exception as other:  # the wrong exception is a failure too
        raise AssertionError(
            f"{what}: raised {type(other).__name__}, not {exc.__name__}"
        ) from other
    raise AssertionError(f"{what}: was accepted")


# ---------------------------------------------------------------------------
# A small base model the scenarios share
# ---------------------------------------------------------------------------


@dataclass
class Base:
    m: Any
    x: int  # float var [0, 1]
    y: int  # float var [0, 1]
    row: int  # Sum(x, y), the body of constraint 0
    obj: int  # Sum(x), the objective


def _base() -> Base:
    m = cbls.Model()
    x = m.float_var(0, 1)
    y = m.float_var(0, 1)
    row = m.sum([x, y])
    m.add_constraint(m.leq(row, m.constant(1.5)))
    obj = m.sum([x])
    m.minimize(obj)
    m.close()
    return Base(m, x, y, row, obj)


def _still_usable(b: Base) -> None:
    """A refusal must leave the model whole: a fresh extension still applies."""
    n_vars, n_nodes = b.m.num_vars(), b.m.num_nodes()
    ext = cbls.ModelExtension(b.m)
    z = ext.float_var(0, 1)
    ext.set_initial(z, 0.25)
    ext.append_to_sum(b.row, z)
    res = b.m.extend(ext)
    assert res.first_new_var == n_vars
    assert res.first_new_node == n_nodes
    assert b.m.num_vars() == n_vars + 1
    assert b.m.node_value(b.row) == pytest.approx(
        b.m.var(vid(b.x)).value + b.m.var(vid(b.y)).value + 0.25
    )
    # And it evaluates: `row` reads x with coefficient 1, whatever else it gained.
    b.m.var_mut(vid(b.x)).value = 0.0
    cbls.full_evaluate(b.m)
    low = b.m.node_value(b.row)
    b.m.var_mut(vid(b.x)).value = 1.0
    cbls.full_evaluate(b.m)
    assert b.m.node_value(b.row) - low == 1.0


# ---------------------------------------------------------------------------
# Child-process scenarios
# ---------------------------------------------------------------------------


def _scenario_bad_handles() -> None:
    """Every builder that takes a handle rejects one that names nothing.

    The boundaries are the first node and the first variable past the base PLUS
    what this extension has recorded, which is what the check is against; the
    far-out ids are what used to reach the CSR splice as a wild write.
    """
    one_arg = [
        "neg", "abs_expr", "sin_expr", "cos_expr", "tan_expr", "exp_expr",
        "log_expr", "sqrt_expr", "tanh_expr", "count",
    ]  # fmt: skip
    two_arg = [
        "prod", "div_expr", "pow_expr", "signpower_expr", "leq", "eq_expr",
        "geq", "neq", "lt", "gt", "at",
    ]  # fmt: skip
    b = _base()
    ext = cbls.ModelExtension(b.m)
    ext.float_var(0, 1)  # the variable boundary moves with what was recorded
    ok = ext.constant(1.0)  # ... and so does the node boundary
    first_bad_node = b.m.num_nodes() + ext.num_new_nodes()
    first_bad_var = -(b.m.num_vars() + ext.num_new_vars() + 1)
    for bad in (first_bad_node, 100_000, 2**31 - 1, first_bad_var, -100_000, -(2**31)):
        cases: dict[str, Callable[[], object]] = {}
        for name in one_arg:
            cases[name] = partial(getattr(ext, name), bad)
        for name in two_arg:
            cases[f"{name}[0]"] = partial(getattr(ext, name), bad, ok)
            cases[f"{name}[1]"] = partial(getattr(ext, name), ok, bad)
        for pos in range(3):
            args = [ok, ok, ok]
            args[pos] = bad
            cases[f"if_then_else[{pos}]"] = partial(ext.if_then_else, *args)
        for name in ("sum", "min_expr", "max_expr"):
            cases[name] = partial(getattr(ext, name), [ok, bad])
        cases["append_to_sum term"] = partial(ext.append_to_sum, b.row, bad)
        if bad >= 0:
            cases["add_constraint"] = partial(ext.add_constraint, bad)
            cases["append_to_sum target"] = partial(ext.append_to_sum, bad, ok)
        n_nodes = ext.num_new_nodes()
        for what, call in cases.items():
            _expect_raises(IndexError, call, f"{what}({bad})")
        # A rejected node leaves nothing recorded.
        assert ext.num_new_nodes() == n_nodes
    # The extension itself is still good, and so is the model.
    ext.add_constraint(ext.leq(ok, ext.constant(2.0)))
    b.m.extend(ext)
    _still_usable(b)
    print("OK")


def _scenario_node_required() -> None:
    """A variable handle where a node is required, and appends that are not appends."""
    b = _base()
    ext = cbls.ModelExtension(b.m)
    z = ext.float_var(0, 1)
    new_sum = ext.sum([z, b.x])
    leq_node = b.m.constraint_ids()[0]  # an existing node that is not a Sum
    cases: dict[str, Callable[[], object]] = {
        "add_constraint(var)": lambda: ext.add_constraint(z),
        "append_to_sum(var target)": lambda: ext.append_to_sum(b.x, z),
        "append_to_sum(non-Sum)": lambda: ext.append_to_sum(leq_node, z),
        "append_to_sum(new Sum)": lambda: ext.append_to_sum(new_sum, z),
        "min_expr([])": lambda: ext.min_expr([]),
        "max_expr([])": lambda: ext.max_expr([]),
    }
    for what, call in cases.items():
        _expect_raises(ValueError, call, what)
    b.m.extend(ext)
    _still_usable(b)
    print("OK")


def _scenario_cyclic_append() -> None:
    """Every append that would close a loop is refused at record time.

    `extend` has no rollback, so this is the only place the refusal can be made
    without leaving the model unusable.
    """
    b = _base()
    constraint_root = b.m.constraint_ids()[0]  # leq(row, 1.5): reads `row`
    ext = cbls.ModelExtension(b.m)
    z = ext.float_var(0, 1)
    reads_row = ext.prod(ext.constant(2.0), b.row)  # a NEW node above `row`
    cases: dict[str, Callable[[], object]] = {
        "the target itself": lambda: ext.append_to_sum(b.row, b.row),
        "an existing parent of the target": lambda: ext.append_to_sum(b.row, constraint_root),
        "a new node reading the target": lambda: ext.append_to_sum(b.row, reads_row),
    }
    for what, call in cases.items():
        _expect_raises(ValueError, call, what)

    # Two appends that are each acyclic but close a loop TOGETHER: obj gains a
    # term reading row, then row gains a term reading obj.
    reads_obj = ext.prod(ext.constant(3.0), b.obj)
    ext.append_to_sum(b.obj, reads_row)
    _expect_raises(ValueError, lambda: ext.append_to_sum(b.row, reads_obj), "two-append loop")

    # What was accepted still applies, and evaluates in dependency order.
    ext.append_to_sum(b.row, z)
    b.m.extend(ext)
    b.m.var_mut(vid(b.x)).value = 1.0
    b.m.var_mut(vid(b.y)).value = 0.5
    cbls.full_evaluate(b.m)
    assert b.m.node_value(b.row) == 1.5  # x + y + z(=0)
    assert b.m.node_value(b.obj) == 1.0 + 2.0 * 1.5  # x + 2*row
    _still_usable(b)
    print("OK")


def _scenario_set_initial() -> None:
    b = _base()
    ext = cbls.ModelExtension(b.m)
    z = ext.int_var(2, 5)
    cases: dict[str, Callable[[], object]] = {
        "an existing variable": lambda: ext.set_initial(b.x, 0.5),
        "a node handle": lambda: ext.set_initial(b.row, 0.5),
        "a handle past the recording": lambda: ext.set_initial(z - 1, 3.0),
        "INT32_MIN": lambda: ext.set_initial(-(2**31), 3.0),
        "below lb": lambda: ext.set_initial(z, 1.0),
        "above ub": lambda: ext.set_initial(z, 6.0),
        "NaN": lambda: ext.set_initial(z, math.nan),
    }
    for what, call in cases.items():
        _expect_raises(ValueError, call, what)
    ext.set_initial(z, 4.0)
    res = b.m.extend(ext)
    assert list(res.new_var_initial) == [4.0]
    assert b.m.var(vid(z)).value == 4.0
    _still_usable(b)
    print("OK")


def _scenario_model_refusals() -> None:
    """Refusals of the model as a whole, each before anything is touched."""
    # Not closed: refused at construction.
    open_model = cbls.Model()
    open_model.float_var(0, 1)
    _expect_raises(RuntimeError, lambda: cbls.ModelExtension(open_model), "open model")

    # Frozen: the extension may be built, extend refuses it.
    b = _base()
    b.m.freeze()
    ext = cbls.ModelExtension(b.m)
    ext.append_to_sum(b.row, ext.float_var(0, 1))
    n_vars, n_nodes = b.m.num_vars(), b.m.num_nodes()
    _expect_raises(RuntimeError, lambda: b.m.extend(ext), "frozen model")
    assert (b.m.num_vars(), b.m.num_nodes()) == (n_vars, n_nodes)
    assert cbls.full_evaluate(b.m) == b.m.var(vid(b.x)).value

    # Built against a different model: refused even at identical counts, since a
    # handle validated against one model names nothing in particular in another.
    a, c = _base(), _base()
    ext = cbls.ModelExtension(a.m)
    ext.append_to_sum(a.row, ext.float_var(0, 1))
    _expect_raises(ValueError, lambda: c.m.extend(ext), "another model's extension")

    # Stale: applied once already, or the model grew since it was recorded.
    a.m.extend(ext)
    _expect_raises(ValueError, lambda: a.m.extend(ext), "replayed extension")
    stale = cbls.ModelExtension(a.m)
    stale.float_var(0, 1)
    grow = cbls.ModelExtension(a.m)
    grow.float_var(0, 1)
    a.m.extend(grow)
    _expect_raises(ValueError, lambda: a.m.extend(stale), "extension older than the model")
    _still_usable(a)
    _still_usable(c)
    print("OK")


def _scenario_extend_inside_evaluation() -> None:
    """A lambda_sum callable that grows the model it is evaluated in is refused.

    Without the refusal the outer walk's per-thread dirty flags stay sized for
    the smaller node count and are then indexed with the new ids.
    """
    m = cbls.Model()
    lst = m.list_var(3)
    x = m.float_var(0, 1)
    pending: list[Any] = []
    seen: list[str] = []

    def func(i: int) -> float:
        if pending:
            ext = pending.pop()
            ext.float_var(0, 1)
            try:
                m.extend(ext)
            except RuntimeError as exc:
                seen.append(str(exc))
                raise
        return float(i)

    obj = m.sum([m.lambda_sum(lst, func), x])
    m.minimize(obj)
    m.close()
    n_vars, n_nodes = m.num_vars(), m.num_nodes()
    m.var_mut(vid(lst)).elements = [0, 1, 2]
    pending.append(cbls.ModelExtension(m))
    _expect_raises(RuntimeError, lambda: cbls.full_evaluate(m), "extend in full_evaluate")
    assert seen and "inside an evaluation" in seen[0], seen
    assert (m.num_vars(), m.num_nodes()) == (n_vars, n_nodes)

    # delta_evaluate re-runs the callable only when the list moves.
    pending.append(cbls.ModelExtension(m))
    m.var_mut(vid(lst)).elements = [2, 1, 0]
    _expect_raises(
        RuntimeError, lambda: cbls.delta_evaluate(m, {vid(lst)}), "extend in delta_evaluate"
    )
    assert len(seen) == 2, seen
    assert (m.num_vars(), m.num_nodes()) == (n_vars, n_nodes)

    # Outside the walk, the same shape of extension applies.
    assert cbls.full_evaluate(m) == 3.0
    ext = cbls.ModelExtension(m)
    ext.append_to_sum(obj, ext.float_var(0, 1))
    m.extend(ext)
    assert cbls.full_evaluate(m) == 3.0
    print("OK")


def _scenario_on_extended_order() -> None:
    """Between Model.extend and on_extended every weight-indexed manager read raises.

    (`is_feasible` and `violated_constraints` read node values only, and are
    safe in the gap.) Each of these indexes the weight vector or the violation cache by the NEW
    row count; before the order was enforced the gap was a heap over-read, and
    `bump_weights` a heap write. FeasibilityJump is not bound, but two bound
    entry points build one over the manager, so they are covered too.
    """
    b = _base()
    vm = cbls.ViolationManager(b.m)
    vm.weights = [3.0]
    ext = cbls.ModelExtension(b.m)
    z = ext.float_var(0, 1)
    ext.append_to_sum(b.row, z)
    ext.add_constraint(ext.geq(z, ext.constant(0.5)))
    res = b.m.extend(ext)

    reads: dict[str, Callable[[], object]] = {
        "total_violation": vm.total_violation,
        "augmented_objective": vm.augmented_objective,
        "bump_weights": vm.bump_weights,
        "weighted_violation_delta": lambda: vm.weighted_violation_delta(vid(z), 1.0),
        "fj_nl_initialize": lambda: cbls.fj_nl_initialize(b.m, vm, 10),
        "LNS.destroy_repair": lambda: cbls.LNS().destroy_repair(b.m, vm, cbls.RNG(1), 0.1),
        "LNS.destroy_repair_cycle": lambda: cbls.LNS().destroy_repair_cycle(
            b.m, vm, cbls.RNG(1), 2, 0.1
        ),
    }
    before = list(b.m.copy_state().values)
    for what, call in reads.items():
        _expect_raises(RuntimeError, call, what)
        # Refused BEFORE anything moved: LNS destroys before it repairs, and the
        # engine's own refusal comes from the repair.
        assert list(b.m.copy_state().values) == before, what
    # The weights setter checks against the manager's own (old) size, so a
    # Python caller cannot paper over the gap by assigning the new length.
    _expect_raises(ValueError, lambda: setattr(vm, "weights", [1.0, 1.0]), "weights resize")

    # Bad arguments to on_extended itself.
    for bad_weight in (math.nan, math.inf, -1.0):
        _expect_raises(ValueError, partial(vm.on_extended, res, bad_weight), f"weight={bad_weight}")
    other = _base()
    other_ext = cbls.ModelExtension(other.m)
    other_ext.add_constraint(other_ext.leq(other.x, other_ext.constant(1.0)))
    other_ext.add_constraint(other_ext.leq(other.y, other_ext.constant(1.0)))
    other_res = other.m.extend(other_ext)  # two new rows, where this model has one
    _expect_raises(ValueError, lambda: vm.on_extended(other_res), "another model's result")

    vm.on_extended(res, 2.0)
    assert list(vm.weights) == [3.0, 2.0]  # the existing row keeps its GLS weight
    # Applying it twice is refused: the manager is already at the new count.
    _expect_raises(ValueError, lambda: vm.on_extended(res), "on_extended twice")
    # z starts at 0 < 0.5, so the new row is violated by 0.5 at weight 2.
    assert vm.total_violation() == pytest.approx(1.0)
    assert vm.violated_constraints() == [1]
    cbls.fj_nl_initialize(b.m, vm, 100)
    assert vm.is_feasible()
    print("OK")


def _scenario_pad_state() -> None:
    b = _base()
    before = b.m.copy_state()
    ext = cbls.ModelExtension(b.m)
    z = ext.int_var(1, 9)
    ext.set_initial(z, 7.0)
    ext.append_to_sum(b.row, z)
    res = b.m.extend(ext)

    # Unpadded, the old state is refused rather than read short.
    _expect_raises(ValueError, lambda: b.m.restore_state(before), "restore unpadded")
    # Padding refuses any state that is not exactly the pre-extension size.
    now = b.m.copy_state()
    _expect_raises(ValueError, lambda: cbls.pad_state(now, res), "pad a current state")
    short = cbls.ModelState()
    short.values = [0.0]
    short.elements = [[]]
    _expect_raises(ValueError, lambda: cbls.pad_state(short, res), "pad a short state")

    b.m.var_mut(vid(b.x)).value = 1.0
    cbls.pad_state(before, res)
    assert list(before.values) == [0.0, 0.0, 7.0]
    b.m.restore_state(before)
    assert b.m.var(vid(b.x)).value == 0.0
    assert b.m.var(vid(z)).value == 7.0
    cbls.full_evaluate(b.m)
    assert b.m.node_value(b.row) == 7.0
    # Padding twice is refused: the state is now the new size.
    _expect_raises(ValueError, lambda: cbls.pad_state(before, res), "pad twice")
    print("OK")


def _scenario_result_is_read_only() -> None:
    """Python cannot fabricate or edit an ExtensionResult for the engine to index by."""
    b = _base()
    ext = cbls.ModelExtension(b.m)
    ext.float_var(0, 1)
    res = b.m.extend(ext)
    _expect_raises(TypeError, cbls.ExtensionResult, "construct")
    for name in ("first_new_var", "first_new_constraint", "num_new_constraints"):
        _expect_raises(AttributeError, partial(setattr, res, name, 999), f"set {name}")
    _expect_raises(
        AttributeError, lambda: setattr(res, "touched_constraints", [999]), "set touched"
    )
    # The list read is a copy: mutating it does not reach the result.
    got = res.touched_constraints
    got.append(999)
    assert res.touched_constraints == []
    print("OK")


def _scenario_keep_alive() -> None:
    """The extension keeps its base model alive (it holds a raw pointer to it).

    Asserted through a weak reference rather than through a use-after-free
    happening to crash: freed memory often still reads fine.
    """
    import gc
    import weakref

    class Tracked(cbls.Model):  # type: ignore[misc]  # a subclass is weak-referenceable
        pass

    def make() -> tuple[Any, int, "weakref.ref[Any]"]:
        m = Tracked()
        x = m.float_var(0, 1)
        row = m.sum([x])
        m.add_constraint(m.leq(row, m.constant(1.0)))
        m.close()
        return cbls.ModelExtension(m), row, weakref.ref(m)

    kept, row, model_ref = make()
    gc.collect()
    assert model_ref() is not None
    # The cycle walk and the Sum check both read the base model.
    kept.append_to_sum(row, kept.float_var(0, 1))
    _expect_raises(ValueError, partial(kept.append_to_sum, row, row), "cycle on the kept model")
    # ... and it is released with the extension, not leaked.
    del kept
    gc.collect()
    assert model_ref() is None
    print("OK")


def _scenario_append_after_another_extend() -> None:
    """Recording into an extension started before another one was applied.

    The cycle walk reads the base model's current children; once another
    extension has grown the model, a base Sum can name a node this extension's
    own table does not cover, and the walk indexed past it -- a SIGSEGV.
    """
    b = _base()
    constraint_root = b.m.constraint_ids()[0]
    early = cbls.ModelExtension(b.m)
    other = cbls.ModelExtension(b.m)
    other.append_to_sum(b.row, other.prod(other.constant(2.0), b.y))
    b.m.extend(other)
    _expect_raises(ValueError, partial(early.append_to_sum, b.obj, constraint_root), "stale append")
    _still_usable(b)
    print("OK")


def _scenario_extend_during_solve() -> None:
    """A SolveCallback that extends the model it is being solved on is refused.

    Append-only is the case that matters: it changes neither the variable nor
    the row count, so the engine's own table checks see nothing, and the search
    used to carry on with stale tables and report feasible on a model it had
    left infeasible. A count-changing extension is refused the same way.

    The extension is recorded BEFORE the solve: building one during it is itself
    refused now (test_model_extension_is_refused_during_a_solve in
    test_column_generation.py), so this scenario isolates the extend check. It is
    recorded after a priming solve, which adds the objective row: recorded before
    that, the structure token would refuse it (ValueError) whatever the registry
    did, and the scenario would no longer show that the registry is what stops a
    VALID append-only extend mid-search.
    """
    for shape in ("append_only", "new_var_and_row"):
        b = _base()
        attempts: list[str] = []
        config = cbls.SearchConfig()
        config.max_iterations = 2_000
        cbls.solve(b.m, time_limit=5.0, seed=1, config=config)
        ext = cbls.ModelExtension(b.m)
        if shape == "append_only":
            ext.append_to_sum(b.row, ext.prod(ext.constant(3.0), b.y))
        else:
            z = ext.float_var(0, 1)
            ext.append_to_sum(b.row, z)
            ext.add_constraint(ext.leq(z, ext.constant(0.5)))

        class Grow(cbls.SolveCallback):  # type: ignore[misc]
            def on_progress(
                self, p: Any, b: Base = b, ext: Any = ext, attempts: list[str] = attempts
            ) -> None:
                try:
                    b.m.extend(ext)
                except RuntimeError as exc:
                    attempts.append(str(exc))
                    raise

        n_vars, n_nodes = b.m.num_vars(), b.m.num_nodes()
        _expect_raises(
            RuntimeError,
            partial(cbls.solve, b.m, time_limit=5.0, seed=1, callback=Grow(), config=config),
            f"extend from on_progress ({shape})",
        )
        assert attempts and "cbls.solve is running" in attempts[0], attempts
        # Nothing grew: the priming solve already added the objective row.
        assert b.m.num_vars() == n_vars, shape
        assert b.m.num_nodes() == n_nodes, shape
        # The refusal is scoped to the solve: afterwards the same extension applies.
        ext = cbls.ModelExtension(b.m)
        ext.append_to_sum(b.row, ext.prod(ext.constant(3.0), b.y))
        b.m.extend(ext)
        assert cbls.solve(b.m, time_limit=5.0, seed=1, config=config).feasible
    print("OK")


def _scenario_replay_refused() -> None:
    """An extension that adds no variable and no node cannot be applied twice.

    Neither count moves, so a count check waved the replay through and every
    term and row it carried landed twice.
    """
    b = _base()
    ext = cbls.ModelExtension(b.m)
    ext.append_to_sum(b.row, b.y)  # row = x + 2y
    b.m.extend(ext)
    _expect_raises(ValueError, lambda: b.m.extend(ext), "replayed append-only extension")
    b.m.var_mut(vid(b.x)).value = 0.0
    b.m.var_mut(vid(b.y)).value = 1.0
    cbls.full_evaluate(b.m)
    assert b.m.node_value(b.row) == 2.0, b.m.node_value(b.row)

    cut = b.m.constraint_ids()[0]  # an existing node, added again as a row
    n_rows = len(b.m.constraint_ids())
    ext = cbls.ModelExtension(b.m)
    ext.add_constraint(cut)
    b.m.extend(ext)
    _expect_raises(ValueError, lambda: b.m.extend(ext), "replayed add_constraint")
    assert len(b.m.constraint_ids()) == n_rows + 1
    print("OK")


def _scenario_same_base_cycle_refused() -> None:
    """Two extensions recorded against one base, closing a cycle between them.

    Each passes its own cycle check. The second used to reach extend's re-sort
    backstop after the growth had begun, and the model then evaluated to 0.0 and
    solved "feasible". It is refused now before anything changes.
    """
    b = _base()
    s2 = b.obj  # Sum(x); b.row is Sum(x, y)
    e1 = cbls.ModelExtension(b.m)
    e1.append_to_sum(b.row, s2)
    e2 = cbls.ModelExtension(b.m)
    e2.append_to_sum(s2, b.row)
    b.m.extend(e1)
    n_nodes = b.m.num_nodes()
    _expect_raises(ValueError, lambda: b.m.extend(e2), "second same-base extension")
    assert b.m.num_nodes() == n_nodes
    assert not b.m.extend_interrupted()
    b.m.var_mut(vid(b.x)).value = 0.25
    b.m.var_mut(vid(b.y)).value = 0.5
    cbls.full_evaluate(b.m)
    assert b.m.node_value(b.row) == 1.0  # x + y + s2
    # Still growable: a fresh extension over the current structure applies.
    e3 = cbls.ModelExtension(b.m)
    e3.append_to_sum(b.row, e3.float_var(0, 1))
    b.m.extend(e3)
    print("OK")


def _scenario_variable_handle_outlives_growth() -> None:
    """A Variable held across growth reads and writes the model, not freed heap.

    var()/var_mut() returned a reference into the variable array, which extend
    (and any builder before close) reallocates; writing `.value` through one
    held across that was a heap use-after-free.
    """
    b = _base()
    held = b.m.var_mut(vid(b.x))
    held_ro = b.m.var(vid(b.y))
    held.value = 0.75
    node = b.m.node(b.row)
    ext = cbls.ModelExtension(b.m)
    for _ in range(5000):  # enough to force the variable array to reallocate
        ext.float_var(0, 1)
    ext.append_to_sum(b.row, ext.constant(0.0))
    b.m.extend(ext)
    junk = [bytearray(64) for _ in range(20000)]  # reuse whatever was freed
    held.value = 0.25
    assert b.m.var(vid(b.x)).value == 0.25, b.m.var(vid(b.x)).value
    assert held.value == 0.25
    assert held.id == vid(b.x)
    assert (held_ro.lb, held_ro.ub, held_ro.type) == (0.0, 1.0, cbls.VarType.Float)
    assert node.id == b.row and node.op == cbls.NodeOp.Sum
    del junk

    # The same hazard before close(), through the ordinary builders.
    m = cbls.Model()
    h = m.float_var(0, 1)
    w = m.var_mut(vid(h))
    for _ in range(5000):
        m.float_var(0, 1)
    w.value = 0.5
    assert m.var(vid(h)).value == 0.5

    # A handle keeps its model alive, and still refuses an id that names nothing.
    lone = cbls.Model()
    lone_var = lone.var_mut(vid(lone.int_var(0, 3)))
    del lone
    lone_var.value = 2.0
    assert lone_var.value == 2.0
    _expect_raises(IndexError, lambda: m.var(10**6), "out-of-range var()")
    print("OK")


def _scenario_interrupted_extend_is_refused() -> None:
    """An extend that fails part-way leaves a model every entry point refuses.

    There is no rollback, so the half-grown model must not be evaluated or
    searched. The one way to reach that state from Python is to run out of memory
    mid-growth: cap the address space just above what the process holds, then
    extend by enough variables that the variable array's reallocation fails.
    """
    b = _base()
    ext = cbls.ModelExtension(b.m)
    for _ in range(1_000_000):
        ext.float_var(0, 1)
    fresh = cbls.ModelExtension(b.m)  # recorded before the failure
    with open("/proc/self/status", encoding="ascii") as status:
        vm_bytes = (
            next(int(line.split()[1]) for line in status if line.startswith("VmSize:")) * 1024
        )
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (vm_bytes + 16 * 1024 * 1024, hard))
    try:
        _expect_raises(MemoryError, lambda: b.m.extend(ext), "extend under a tight address space")
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))
    assert b.m.extend_interrupted()
    config = cbls.SearchConfig()
    config.max_iterations = 100
    refusals: dict[str, Callable[[], object]] = {
        "extend(fresh)": lambda: b.m.extend(fresh),
        "extend(ext)": lambda: b.m.extend(ext),
        "ModelExtension": lambda: cbls.ModelExtension(b.m),
        "full_evaluate": lambda: cbls.full_evaluate(b.m),
        "delta_evaluate": lambda: cbls.delta_evaluate(b.m, {vid(b.x)}),
        "ViolationManager": lambda: cbls.ViolationManager(b.m),
        "per_constraint_violation_delta": lambda: b.m.per_constraint_violation_delta(vid(b.x), 0.5),
        "compute_partial": lambda: cbls.compute_partial(b.m, b.row, vid(b.x)),
        "compute_all_partials": lambda: cbls.compute_all_partials(b.m, b.row),
        "solve": lambda: cbls.solve(b.m, time_limit=1.0, seed=1, config=config),
        "freeze": lambda: b.m.freeze(),
    }
    # The base has an objective and has never been solved, so solve and freeze
    # would first append the objective row -- a rebuild over the half-grown
    # arrays -- unless the refusal comes before it.
    n_nodes = b.m.num_nodes()
    for what, call in refusals.items():
        _expect_raises(RuntimeError, call, what)
    assert b.m.num_nodes() == n_nodes, "a refusal ran after mutating the corrupt model"
    print("OK")


SCENARIOS: dict[str, "Callable[[], None]"] = {
    "bad_handles": _scenario_bad_handles,
    "node_required": _scenario_node_required,
    "cyclic_append": _scenario_cyclic_append,
    "set_initial": _scenario_set_initial,
    "model_refusals": _scenario_model_refusals,
    "extend_inside_evaluation": _scenario_extend_inside_evaluation,
    "on_extended_order": _scenario_on_extended_order,
    "pad_state": _scenario_pad_state,
    "result_is_read_only": _scenario_result_is_read_only,
    "keep_alive": _scenario_keep_alive,
    "append_after_another_extend": _scenario_append_after_another_extend,
    "extend_during_solve": _scenario_extend_during_solve,
    "replay_refused": _scenario_replay_refused,
    "same_base_cycle_refused": _scenario_same_base_cycle_refused,
    "variable_handle_outlives_growth": _scenario_variable_handle_outlives_growth,
    "interrupted_extend_is_refused": _scenario_interrupted_extend_is_refused,
}


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_a_bad_extension_raises_instead_of_crashing(scenario: str) -> None:
    proc = _run_scenario(scenario)
    assert proc.returncode == 0, (
        f"child exited {proc.returncode} (negative = killed by that signal)\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip().endswith("OK"), proc.stdout


# ---------------------------------------------------------------------------
# Round trips: extended == built whole
# ---------------------------------------------------------------------------


@dataclass
class Spec:
    """A linear model as rows of (var index, coefficient) terms.

    Variables `[0, n_base)` and rows `[0, len(base_rows))` form the base; the
    rest is the extension. `appends[r]` are the terms the extension adds to BASE
    row `r` (and `obj_appends` to the objective) -- the column-generation shape.
    """

    n_base: int
    n_new: int
    bounds: list[tuple[int, int]]
    initial: dict[int, float]  # new-variable initial values
    base_rows: list[list[tuple[int, float]]]
    rhs: list[float]
    new_rows: list[list[tuple[int, float]]]
    new_rhs: list[float]
    appends: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    obj_terms: list[tuple[int, float]] = field(default_factory=list)
    obj_appends: list[tuple[int, float]] = field(default_factory=list)


def _random_spec(rng: random.Random) -> Spec:
    n_base = rng.randint(2, 6)
    n_new = rng.randint(1, 4)
    n = n_base + n_new
    bounds = [(rng.randint(-3, 0), rng.randint(1, 5)) for _ in range(n)]
    initial = {v: float(rng.randint(*bounds[v])) for v in range(n_base, n) if rng.random() < 0.5}

    def terms(pool: range, k: int) -> list[tuple[int, float]]:
        return [(rng.choice(pool), float(rng.randint(-4, 4))) for _ in range(k)]

    base_rows = [terms(range(n_base), rng.randint(1, 4)) for _ in range(rng.randint(1, 4))]
    appends = {
        r: terms(range(n_base, n), rng.randint(1, 3))
        for r in range(len(base_rows))
        if rng.random() < 0.6
    }
    new_rows = [terms(range(n), rng.randint(1, 4)) for _ in range(rng.randint(0, 3))]
    return Spec(
        n_base=n_base,
        n_new=n_new,
        bounds=bounds,
        initial=initial,
        base_rows=base_rows,
        rhs=[float(rng.randint(-2, 6)) for _ in base_rows],
        new_rows=new_rows,
        new_rhs=[float(rng.randint(-2, 6)) for _ in new_rows],
        appends=appends,
        obj_terms=terms(range(n_base), rng.randint(1, 3)),
        obj_appends=terms(range(n_base, n), rng.randint(0, 2)),
    )


def _term(b: Any, handles: list[int], v: int, coef: float) -> int:
    """`coef * x_v` as a node, the way both builders share."""
    return int(b.prod(b.constant(coef), handles[v]))


def _build_whole(spec: Spec) -> tuple[Any, list[int], int]:
    m = cbls.Model()
    n = spec.n_base + spec.n_new
    xs = [m.int_var(*spec.bounds[v]) for v in range(n)]
    for v, val in spec.initial.items():
        m.var_mut(vid(xs[v])).value = val
    for r, row in enumerate(spec.base_rows):
        full = row + spec.appends.get(r, [])
        m.add_constraint(
            m.leq(m.sum([_term(m, xs, v, c) for v, c in full]), m.constant(spec.rhs[r]))
        )
    for r, row in enumerate(spec.new_rows):
        body = m.sum([_term(m, xs, v, c) for v, c in row])
        m.add_constraint(m.leq(body, m.constant(spec.new_rhs[r])))
    obj = m.sum([_term(m, xs, v, c) for v, c in spec.obj_terms + spec.obj_appends])
    m.minimize(obj)
    m.close()
    return m, xs, obj


def _build_extended(spec: Spec) -> tuple[Any, list[int], int, Any]:
    m = cbls.Model()
    xs = [m.int_var(*spec.bounds[v]) for v in range(spec.n_base)]
    sums = []
    for r, row in enumerate(spec.base_rows):
        s = m.sum([_term(m, xs, v, c) for v, c in row])
        sums.append(s)
        m.add_constraint(m.leq(s, m.constant(spec.rhs[r])))
    obj = m.sum([_term(m, xs, v, c) for v, c in spec.obj_terms])
    m.minimize(obj)
    m.close()

    ext = cbls.ModelExtension(m)
    xs += [ext.int_var(*spec.bounds[v]) for v in range(spec.n_base, spec.n_base + spec.n_new)]
    for v, val in spec.initial.items():
        ext.set_initial(xs[v], val)
    for r, extra in spec.appends.items():
        for v, c in extra:
            ext.append_to_sum(sums[r], _term(ext, xs, v, c))
    for r, row in enumerate(spec.new_rows):
        body = ext.sum([_term(ext, xs, v, c) for v, c in row])
        ext.add_constraint(ext.leq(body, ext.constant(spec.new_rhs[r])))
    for v, c in spec.obj_appends:
        ext.append_to_sum(obj, _term(ext, xs, v, c))
    res = m.extend(ext)
    return m, xs, obj, res


def _observe(m: Any, obj: int) -> tuple[list[float], float]:
    return [m.node_value(c) for c in m.constraint_ids()], m.node_value(obj)


@pytest.mark.parametrize("seed", range(24))
def test_an_extended_model_matches_the_same_model_built_whole(seed: int) -> None:
    spec = _random_spec(random.Random(seed))
    whole, wx, wobj = _build_whole(spec)
    grown, gx, gobj, res = _build_extended(spec)

    n = spec.n_base + spec.n_new
    assert grown.num_vars() == whole.num_vars() == n
    assert len(grown.constraint_ids()) == len(whole.constraint_ids())
    assert [vid(h) for h in gx] == [vid(h) for h in wx]  # same variable ids
    assert res.first_new_var == spec.n_base and res.num_new_vars == spec.n_new
    assert res.first_new_constraint == len(spec.base_rows)
    assert res.num_new_constraints == len(spec.new_rows)
    assert list(res.touched_constraints) == sorted(spec.appends)
    for v in range(n):
        assert grown.constraints_of_var(v) == whole.constraints_of_var(v), v
        assert grown.var(v).value == whole.var(v).value, v
    # new_incidences is EXACTLY what G_v gained -- every incidence the base rows
    # did not already have -- sorted by constraint then variable.
    base_incidences = {(r, v) for r, row in enumerate(spec.base_rows) for v, _ in row}
    gained = {(c, v) for v in range(n) for c in grown.constraints_of_var(v)} - base_incidences
    assert list(res.new_incidences) == sorted(gained)

    # extend leaves the node values current, exactly as close() does.
    assert _observe(grown, gobj) == _observe(whole, wobj)

    rng = random.Random(1000 + seed)
    for _ in range(5):
        values = [float(rng.randint(*spec.bounds[v])) for v in range(n)]
        for m in (whole, grown):
            for v in range(n):
                m.var_mut(v).value = values[v]
            cbls.full_evaluate(m)
        assert _observe(grown, gobj) == _observe(whole, wobj)
        # And incrementally: move one variable (new ones included) on both.
        v = rng.randrange(n)
        new_value = float(rng.randint(*spec.bounds[v]))
        for m in (whole, grown):
            m.var_mut(v).value = new_value
            cbls.delta_evaluate(m, {v})
        assert _observe(grown, gobj) == _observe(whole, wobj)


def _knapsack_like(extend: bool) -> tuple[Any, int]:
    """min 2x + 3y + z  s.t.  x + y + z >= 4,  z <= 3;  optimum 5 at (1, 0, 3).

    With `extend`, z arrives as a column: it enters the existing covering row
    and the objective by append_to_sum, and brings its own bound row.
    """
    m = cbls.Model()
    x = m.int_var(0, 10)
    y = m.int_var(0, 10)
    z = m.int_var(0, 10) if not extend else None
    cover = m.sum([x, y] + ([z] if z is not None else []))
    m.add_constraint(m.geq(cover, m.constant(4.0)))
    obj_terms = [m.prod(m.constant(2.0), x), m.prod(m.constant(3.0), y)]
    if z is not None:
        obj_terms.append(z)
        m.add_constraint(m.leq(z, m.constant(3.0)))
    obj = m.sum(obj_terms)
    m.minimize(obj)
    m.close()
    if extend:
        ext = cbls.ModelExtension(m)
        z = ext.int_var(0, 10)
        ext.append_to_sum(cover, z)
        ext.append_to_sum(obj, z)
        ext.add_constraint(ext.leq(z, ext.constant(3.0)))
        m.extend(ext)
    return m, obj


def test_solve_on_an_extended_model_finds_the_same_optimum_as_the_whole_model() -> None:
    config = cbls.SearchConfig()
    config.max_iterations = 20_000
    results = []
    for extend in (False, True):
        m, obj = _knapsack_like(extend)
        r = cbls.solve(m, time_limit=5.0, seed=7, config=config)
        assert r.feasible, extend
        results.append(r.objective)
        cbls.full_evaluate(m)
        assert m.node_value(obj) == pytest.approx(r.objective)
    assert results == [pytest.approx(5.0), pytest.approx(5.0)]


def test_solve_resumes_from_a_padded_incumbent_after_extending() -> None:
    """The between-solves loop: solve, extend, pad the incumbent, solve again."""
    base = cbls.Model()
    x = base.int_var(0, 10)
    y = base.int_var(0, 10)
    cover = base.sum([x, y])
    base.add_constraint(base.geq(cover, base.constant(4.0)))
    obj = base.sum([base.prod(base.constant(2.0), x), base.prod(base.constant(3.0), y)])
    base.minimize(obj)
    base.close()
    config = cbls.SearchConfig()
    config.max_iterations = 20_000
    first = cbls.solve(base, time_limit=5.0, seed=3, config=config)
    assert first.feasible and first.objective == pytest.approx(8.0)  # x = 4
    incumbent = base.copy_state()

    ext = cbls.ModelExtension(base)
    z = ext.int_var(0, 10)
    ext.append_to_sum(cover, z)
    ext.append_to_sum(obj, z)
    ext.add_constraint(ext.leq(z, ext.constant(3.0)))
    res = base.extend(ext)
    cbls.pad_state(incumbent, res)
    base.restore_state(incumbent)
    cbls.full_evaluate(base)
    # The old incumbent is still feasible (z = 0) and scores what it did.
    assert base.node_value(obj) == pytest.approx(8.0)
    vm = cbls.ViolationManager(base)
    assert vm.is_feasible()

    # skip_init keeps the assignment solve() is handed; without it FJ re-seeds
    # every scalar and the padded incumbent is never the start.
    config.skip_init = True
    reported: list[float] = []

    class Recorder(cbls.SolveCallback):  # type: ignore[misc]
        def on_progress(self, p: Any) -> None:
            reported.append(p.objective)

    second = cbls.solve(base, time_limit=5.0, seed=3, config=config, callback=Recorder())
    assert reported and reported[0] == pytest.approx(8.0)  # started from the incumbent
    assert second.feasible and second.objective == pytest.approx(5.0)
    assert base.var(vid(z)).value == 3.0


_UNARY = ["neg", "abs_expr", "sin_expr", "cos_expr", "tan_expr", "exp_expr", "log_expr",
          "sqrt_expr", "tanh_expr"]  # fmt: skip
_BINARY = ["prod", "div_expr", "pow_expr", "signpower_expr", "leq", "eq_expr", "geq", "neq",
           "lt", "gt"]  # fmt: skip
_LIST = ["at", "count"]


def _op_value(name: str, use_ext: bool) -> float:
    """The value of one op over fixed inputs, built by Model or by ModelExtension."""
    m = cbls.Model()
    a, b = m.float_var(0.3, 2), m.float_var(0.2, 3)
    lst = m.list_var(4)
    m.var_mut(vid(a)).value = 0.7
    m.var_mut(vid(b)).value = 1.9
    m.var_mut(vid(lst)).elements = [3, 0, 2, 1]
    m.add_constraint(m.leq(a, m.constant(5.0)))
    m.add_constraint(m.leq(m.count(lst), m.constant(5.0)))  # the list is read
    builder: Any = m
    if use_ext:
        m.close()
        builder = cbls.ModelExtension(m)
    args: list[Any]
    if name in _UNARY:
        args = [a]
    elif name in _BINARY:
        args = [a, b]
    elif name == "if_then_else":
        args = [a, b, a]
    elif name == "at":
        args = [lst, builder.constant(2.0)]
    elif name == "count":
        args = [lst]
    else:
        args = [[a, b]]
    node = int(getattr(builder, name)(*args))
    builder.add_constraint(builder.leq(node, builder.constant(99.0)))
    if use_ext:
        m.extend(builder)
    else:
        m.close()
    return float(m.node_value(node))


@pytest.mark.parametrize(
    "name", _UNARY + _BINARY + _LIST + ["if_then_else", "min_expr", "max_expr", "sum"]
)
def test_each_extension_builder_builds_the_same_op_as_the_model(name: str) -> None:
    got, want = _op_value(name, True), _op_value(name, False)
    assert got == want


if __name__ == "__main__":
    SCENARIOS[sys.argv[1]]()
