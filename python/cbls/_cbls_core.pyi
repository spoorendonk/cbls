"""CBLS: Constraint-Based Local Search engine (C++ core)"""

from collections.abc import Callable, Sequence, Set
import enum
from typing import Annotated, overload

import numpy
from numpy.typing import NDArray


class VarType(enum.Enum):
    Bool = 0

    Int = 1

    Float = 2

    List = 3

    Set = 4

class NodeOp(enum.Enum):
    Const = 0

    Neg = 1

    Sum = 2

    Prod = 3

    Div = 4

    Pow = 5

    Min = 6

    Max = 7

    Abs = 8

    Sin = 9

    Cos = 10

    If = 17

    At = 18

    Count = 19

    Lambda = 20

    PairLambda = 21

    Leq = 22

    Eq = 23

    Tan = 11

    Exp = 12

    Log = 13

    Sqrt = 14

    SignPower = 15

    Tanh = 16

    Geq = 24

    Neq = 25

    Lt = 26

    Gt = 27

    Element = 28

    Ceil = 29

    Floor = 30

    Round = 31

    LambdaExtra = 32

    PairLambdaExtra = 33

class Variable:
    @property
    def id(self) -> int: ...

    @property
    def type(self) -> VarType: ...

    @property
    def value(self) -> float: ...

    @value.setter
    def value(self, arg: float, /) -> None: ...

    @property
    def lb(self) -> float: ...

    @property
    def ub(self) -> float: ...

    @property
    def name(self) -> str: ...

    @property
    def elements(self) -> list[int]: ...

    @elements.setter
    def elements(self, arg: Sequence[int], /) -> None: ...

    @property
    def universe_size(self) -> int: ...

    @property
    def min_size(self) -> int: ...

    @property
    def max_size(self) -> int: ...

    @property
    def list_init(self) -> ListInit: ...

    @property
    def partitioned(self) -> bool: ...

class ListInit(enum.Enum):
    Identity = 0

    Empty = 1

    Random = 2

class Cover(enum.Enum):
    Exact = 0

    AtMostOnce = 1

class ListPartition:
    @property
    def list_ids(self) -> list[int]: ...

    @property
    def cover(self) -> Cover: ...

    @property
    def universe_size(self) -> int: ...

class ExprNode:
    @property
    def id(self) -> int: ...

    @property
    def op(self) -> NodeOp: ...

class TerminationReason(enum.Enum):
    TimeLimit = 0

    IterationLimit = 1

    Feasible = 2

    NoBudget = 3

    Stopped = 4

    Cancelled = 5

class StopToken:
    def __init__(self) -> None: ...

    def request(self) -> None:
        """
        Ask the solve to stop. Safe to call from any thread, including while
        a solve is running -- which is the point: cbls.solve releases the GIL,
        so another Python thread can reach this. The run ends at its next
        batch boundary with termination == TerminationReason.Cancelled.
        """

    def reset(self) -> None:
        """
        Clear the flag so the token can be reused. A solve does NOT reset it:
        handing the same raised token to the next solve cancels that one too.
        """

    def requested(self) -> bool: ...

class StructuralSelection(enum.Enum):
    FirstImprovingSample = 0

    BestOfSample = 1

    ViolationGuided = 2

    GranularBest = 3

    Auto = 4

class NeighbourList:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, rows: Sequence[Sequence[int]]) -> None:
        """
        rows[e] are element e's neighbours, nearest first. Raises ValueError on an id outside [0, len(rows)).
        """

    def universe(self) -> int: ...

    def neighbours_of(self, element: int) -> list[int]:
        """
        Element's neighbours, nearest first. Empty for an out-of-range element.
        """

    @property
    def offsets(self) -> list[int]: ...

    @property
    def ids(self) -> list[int]: ...

    def __len__(self) -> int: ...

def nearest_neighbours(universe: int, k: int, cost: Callable[[int, int], float]) -> NeighbourList:
    """
    Each element's k nearest others under cost(a, b), nearest first, ties broken by ascending id.

    O(universe^2) cost calls, and `cost` is a Python callable invoked from C++, so this is a setup-time convenience for a universe of a few thousand. Build the rows yourself and use NeighbourList(rows) for anything larger.
    """

class BatchKind(enum.Enum):
    FeasibilityJump = 0

    NoveltyJump = 1

    Structural = 2

    FixPropagateRepair = 3

class GeneratorCounters:
    @property
    def name(self) -> str: ...

    @property
    def moves_tried(self) -> int: ...

    @property
    def moves_accepted(self) -> int: ...

class SearchCounters:
    @property
    def batches(self) -> int: ...

    @property
    def fj_batches(self) -> int: ...

    @property
    def novelty_batches(self) -> int: ...

    @property
    def novelty_moves(self) -> int: ...

    @property
    def novelty_commits(self) -> int: ...

    @property
    def novelty_weight_bumps(self) -> int: ...

    @property
    def structural_batches(self) -> int: ...

    @property
    def fpr_batches(self) -> int: ...

    @property
    def fpr_setup_seconds(self) -> float: ...

    @property
    def structural_moves_tried(self) -> int: ...

    @property
    def structural_moves_accepted(self) -> int: ...

    @property
    def by_generator(self) -> list[GeneratorCounters]: ...

    @property
    def inner_solver_calls(self) -> int: ...

    @property
    def inner_solver_seconds(self) -> float: ...

    @property
    def inner_solver_infeasible_calls(self) -> int: ...

    @property
    def portfolio_restarts(self) -> int: ...

    @property
    def cone_cache_walks(self) -> int: ...

    @property
    def cone_fallback_walks(self) -> int: ...

    @property
    def shared_bound_tightenings(self) -> int: ...

    @property
    def own_best_behind_global(self) -> int: ...

    @property
    def bound_behind_global_batches(self) -> int: ...

    @property
    def bound_behind_global_seconds(self) -> float: ...

class WorkerFailure:
    @property
    def worker(self) -> int: ...

    @property
    def produced_result(self) -> bool: ...

    @property
    def reason(self) -> str: ...

class SearchResult:
    @property
    def objective(self) -> float: ...

    @property
    def feasible(self) -> bool: ...

    @property
    def iterations(self) -> int: ...

    @property
    def time_seconds(self) -> float: ...

    @property
    def termination(self) -> TerminationReason: ...

    @property
    def workers_launched(self) -> int: ...

    @property
    def workers_completed(self) -> int: ...

    @property
    def worker_failures(self) -> list[WorkerFailure]: ...

    @property
    def counters(self) -> SearchCounters: ...

    @property
    def best_state(self) -> ModelState: ...

class Model:
    def __init__(self) -> None: ...

    def bool_var(self, name: str = '') -> int: ...

    def int_var(self, lb: int, ub: int, name: str = '') -> int: ...

    def float_var(self, lb: float, ub: float, name: str = '') -> int: ...

    @overload
    def list_var(self, n: int, name: str = '') -> int:
        """
        A fixed-length permutation of {0..n-1}: universe == min_len == max_len.
        """

    @overload
    def list_var(self, universe: int, min_len: int, max_len: int, init: ListInit = ListInit.Empty, name: str = '') -> int:
        """
        An ordered sequence of distinct elements of {0..universe-1} whose
        length stays within [min_len, max_len].
        """

    def set_var(self, n: int, min_size: int = 0, max_size: int = -1, name: str = '') -> int: ...

    @overload
    def add_list_partition(self, lists: Sequence[int], cover: Cover = Cover.Exact) -> int:
        """
        Declare that `lists` partition their shared universe, maintained by
        the moves rather than by a constraint row. Returns the partition index.
        `cover` accepts a Cover value or the strings 'exact' / 'at_most_once'.
        """

    @overload
    def add_list_partition(self, lists: Sequence[int], cover: str) -> int: ...

    def partition_of_list(self, var_id: int) -> int:
        """
        Index into list_partitions() of the partition this variable id belongs
        to, or -1. Takes a var ID, not a handle.
        """

    @property
    def list_partitions(self) -> list[ListPartition]: ...

    def constant(self, arg: float, /) -> int: ...

    def neg(self, arg: int, /) -> int: ...

    def sum(self, arg: Sequence[int], /) -> int: ...

    def prod(self, arg0: int, arg1: int, /) -> int: ...

    def div_expr(self, arg0: int, arg1: int, /) -> int: ...

    def pow_expr(self, arg0: int, arg1: int, /) -> int: ...

    def min_expr(self, arg: Sequence[int], /) -> int: ...

    def max_expr(self, arg: Sequence[int], /) -> int: ...

    def abs_expr(self, arg: int, /) -> int: ...

    def sin_expr(self, arg: int, /) -> int: ...

    def cos_expr(self, arg: int, /) -> int: ...

    def tan_expr(self, arg: int, /) -> int: ...

    def exp_expr(self, arg: int, /) -> int: ...

    def log_expr(self, arg: int, /) -> int: ...

    def sqrt_expr(self, arg: int, /) -> int: ...

    def signpower_expr(self, arg0: int, arg1: int, /) -> int: ...

    def tanh_expr(self, arg: int, /) -> int: ...

    def if_then_else(self, arg0: int, arg1: int, arg2: int, /) -> int: ...

    def at(self, arg0: int, arg1: int, /) -> int: ...

    def count(self, arg: int, /) -> int: ...

    @overload
    def element(self, table: Sequence[float], index: int) -> int:
        """
        table[index]: a table looked up by a decision. The index may be
        any scalar; its value is truncated toward zero, as `at` reads its index,
        and an index outside [0, len(table)) reads 0.0. The table is copied into
        the model, and its entries must be finite.
        """

    @overload
    def element(self, table: Sequence[Sequence[float]], row: int, col: int) -> int:
        """
        table[row][col] over a rectangular table; 0.0 when either index is out of range.
        """

    def ceil_expr(self, x: int) -> int: ...

    def floor_expr(self, x: int) -> int: ...

    def round_expr(self, x: int) -> int:
        """round half away from zero, as C's round()."""

    def leq(self, arg0: int, arg1: int, /) -> int: ...

    def eq_expr(self, arg0: int, arg1: int, /) -> int: ...

    def geq(self, arg0: int, arg1: int, /) -> int: ...

    def neq(self, arg0: int, arg1: int, /) -> int: ...

    def lt(self, arg0: int, arg1: int, /) -> int: ...

    def gt(self, arg0: int, arg1: int, /) -> int: ...

    @overload
    def lambda_sum(self, list_var: int, func: Callable[[int], float]) -> int:
        """
        Sum `func(e)` over the elements of a List or Set variable.

        A callable that reaches an Expr (or the Model) pins the model for the
        process lifetime: the model holds the callable where the cycle collector
        cannot see it. Capture handles (`x.handle`) or plain data instead.
        """

    @overload
    def lambda_sum(self, list_var: int, func: Callable[[int, Sequence[float]], float], *, extra: Sequence[int]) -> int:
        """
        Sum `func(e, x)` over the elements of a List or Set variable, where `x`
        is the list of the CURRENT values of the `extra` handles, in order -- a
        per-element term that depends on other decisions, e.g. a stop cost by the
        route's vehicle type: `func=lambda i, x: c[i][int(x[0])]`.

        A change to any extra, or any edit to the list, re-sums the list. The
        node cannot be written to a .cbls file: the C++ save_model (not bound in
        Python) refuses the model.
        Every call re-acquires the GIL, as a plain lambda_sum's does.

        `extra` is keyword-only, here and in pair_lambda_sum, whose third
        positional parameter is already `cyclic`.
        """

    def lambda_table_sum(self, list_var: int, table: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> int:
        """
        lambda_sum with the function given as a length-n float64 array over the
        variable's universe. Copied into the engine at node creation, so the
        search makes no Python call.
        """

    @overload
    def pair_lambda_sum(self, list_var: int, func: Callable[[int, int], float], cyclic: bool = False, head: Callable[[int], float] | None = None, tail: Callable[[int], float] | None = None) -> int:
        """
        Sum `func(e_k, e_{k+1})` over the consecutive pairs of a List or Set
        variable's elements.

        A callable that reaches an Expr (or the Model) pins the model for the
        process lifetime: the model holds the callable where the cycle collector
        cannot see it. Capture handles (`x.handle`) or plain data instead.

        cyclic=True adds the closing pair `func(e_{n-1}, e_0)` when n >= 2, which
        is a tour cost. head and tail, if given, add `head(e_0)` and
        `tail(e_{n-1})` -- the depot legs of a route, which a cyclic sum over the
        customers alone would get wrong.

        n == 0 is 0.0 for every variant; n == 1 is `head(e_0) + tail(e_0)`.

        A Set's elements have no modelled order, so a pair sum over one reads
        whatever order they are currently stored in.

        Every call re-acquires the GIL, so a Python func is a serialisation point
        for a portfolio. Use pair_table_sum where the function is a matrix.
        """

    @overload
    def pair_lambda_sum(self, list_var: int, func: Callable[[int, int, Sequence[float]], float], *, extra: Sequence[int], cyclic: bool = False) -> int:
        """
        Sum `func(e_k, e_{k+1}, x)` over consecutive pairs, `x` as in the extra
        form of lambda_sum. cyclic=True adds the closing pair when n >= 2. There
        are no head/tail terms in this form. `extra` is keyword-only.
        """

    def pair_table_sum(self, list_var: int, dist: Annotated[NDArray[numpy.float64], dict(shape=(None, None), order='C', device='cpu', writable=False)], cyclic: bool = False, head: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None, tail: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> int:
        """
        pair_lambda_sum with the function given as a distance matrix.

        dist is an (n, n) float64 array over the variable's universe_size, and
        head/tail are length-n arrays.
        All are COPIED into the engine at node creation, so the search makes no
        Python call at all and resizing or freeing the caller's array afterwards
        is harmless. A wrong shape raises here rather than being read past.

        Copied once per MODEL, not once per process. ParallelSearch's Python entry
        points take a model factory, so a portfolio builds one Model -- and one
        copy of this matrix -- per worker: an (n, n) float64 table costs
        8 * n * n bytes per thread.
        """

    @overload
    def add_constraint(self, arg: int, /) -> None: ...

    @overload
    def add_constraint(self, arg: Expr, /) -> None: ...

    @overload
    def minimize(self, arg: int, /) -> None: ...

    @overload
    def minimize(self, arg: Expr, /) -> None: ...

    @overload
    def maximize(self, arg: int, /) -> None: ...

    @overload
    def maximize(self, arg: Expr, /) -> None: ...

    def add_var_sequence(self, var_ids: Sequence[int], min_block_on: int = 1, min_block_off: int = 1) -> None: ...

    def var_sequence_for(self, arg: int, /) -> tuple[int, int]: ...

    def close(self) -> None: ...

    def freeze(self, cone_cache_bytes: int = 268435456) -> None: ...

    def is_frozen(self) -> bool: ...

    def num_cached_cones(self) -> int: ...

    def cone_cache_bytes(self) -> int: ...

    def var(self, arg: int, /) -> Variable:
        """
        A handle to variable `id` of this model. It holds the model and the id, not
        a pointer into the model's storage, so it stays valid across builders and
        always reads the variable's current state. Keeps the model alive. var() and
        var_mut() return the same kind of handle.
        """

    def var_mut(self, arg: int, /) -> Variable:
        """
        A handle to variable `id` of this model. It holds the model and the id, not
        a pointer into the model's storage, so it stays valid across builders and
        always reads the variable's current state. Keeps the model alive. var() and
        var_mut() return the same kind of handle.
        """

    def node(self, arg: int, /) -> ExprNode:
        """
        A COPY of node `id`'s id and op -- neither ever changes once the node
        exists, so the copy cannot go stale. Its current value is
        Model.node_value(id).
        """

    def node_value(self, id: int) -> float: ...

    def objective_id(self) -> int: ...

    def constraint_ids(self) -> list[int]: ...

    def num_user_constraints(self) -> int: ...

    def num_domain_rows(self) -> int: ...

    def is_domain_node(self, id: int) -> bool: ...

    def constraints_of_var(self, var_id: int) -> list[int]: ...

    def per_constraint_violation_delta(self, var_id: int, j: float) -> list[tuple[int, float]]: ...

    def num_vars(self) -> int: ...

    def num_nodes(self) -> int: ...

    def copy_state(self) -> ModelState: ...

    def restore_state(self, arg: ModelState, /) -> None: ...

    def Bool(self, name: str = '') -> Expr: ...

    def Int(self, lb: int, ub: int, name: str = '') -> Expr: ...

    def Float(self, lb: float, ub: float, name: str = '') -> Expr: ...

    @overload
    def List(self, n: int, name: str = '') -> Expr: ...

    @overload
    def List(self, universe: int, min_len: int, max_len: int, init: ListInit = ListInit.Empty, name: str = '') -> Expr: ...

    def Set(self, n: int, min_size: int = 0, max_size: int = -1, name: str = '') -> Expr: ...

    def Constant(self, arg: float, /) -> Expr: ...

class Expr:
    @property
    def model(self) -> Model: ...

    @property
    def handle(self) -> int: ...

    def var_id(self) -> int: ...

    @overload
    def __add__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __add__(self, arg: float, /) -> Expr: ...

    def __radd__(self, arg: float, /) -> Expr: ...

    @overload
    def __mul__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __mul__(self, arg: float, /) -> Expr: ...

    def __rmul__(self, arg: float, /) -> Expr: ...

    @overload
    def __sub__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __sub__(self, arg: float, /) -> Expr: ...

    def __rsub__(self, arg: float, /) -> Expr: ...

    @overload
    def __truediv__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __truediv__(self, arg: float, /) -> Expr: ...

    def __rtruediv__(self, arg: float, /) -> Expr: ...

    def __neg__(self) -> Expr: ...

    @overload
    def __pow__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __pow__(self, arg: int, /) -> Expr: ...

    @overload
    def __pow__(self, arg: float, /) -> Expr: ...

    def __rpow__(self, arg: float, /) -> Expr: ...

    @overload
    def __le__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __le__(self, arg: float, /) -> Expr: ...

    @overload
    def __ge__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __ge__(self, arg: float, /) -> Expr: ...

    @overload
    def __lt__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __lt__(self, arg: float, /) -> Expr: ...

    @overload
    def __gt__(self, arg: Expr, /) -> Expr: ...

    @overload
    def __gt__(self, arg: float, /) -> Expr: ...

    def __ceil__(self) -> Expr: ...

    def __floor__(self) -> Expr: ...

    def __round__(self, ndigits: int | None = None) -> Expr: ...

    def __abs__(self) -> Expr: ...

    def is_var(self) -> bool: ...

    def eq(self, arg: Expr, /) -> Expr: ...

    def neq(self, arg: Expr, /) -> Expr: ...

    def pow(self, arg: Expr, /) -> Expr: ...

def sin(arg: Expr, /) -> Expr: ...

def cos(arg: Expr, /) -> Expr: ...

def tan(arg: Expr, /) -> Expr: ...

def exp(arg: Expr, /) -> Expr: ...

def log(arg: Expr, /) -> Expr: ...

def sqrt(arg: Expr, /) -> Expr: ...

def abs(arg: Expr, /) -> Expr: ...

def pow(arg0: Expr, arg1: Expr, /) -> Expr: ...

def min(arg: Sequence[Expr], /) -> Expr: ...

def max(arg: Sequence[Expr], /) -> Expr: ...

def if_then_else(arg0: Expr, arg1: Expr, arg2: Expr, /) -> Expr: ...

def ceil(arg: Expr, /) -> Expr: ...

def floor(arg: Expr, /) -> Expr: ...

def round(arg: Expr, /) -> Expr: ...

@overload
def element(table: Sequence[float], index: Expr) -> Expr:
    """
    table[index]: a table looked up by a decision. The index may be
    any scalar; its value is truncated toward zero, as `at` reads its index,
    and an index outside [0, len(table)) reads 0.0. The table is copied into
    the model, and its entries must be finite.
    """

@overload
def element(table: Sequence[Sequence[float]], row: Expr, col: Expr) -> Expr: ...

class ModelState:
    def __init__(self) -> None: ...

    @property
    def values(self) -> list[float]: ...

    @values.setter
    def values(self, arg: Sequence[float], /) -> None: ...

    @property
    def elements(self) -> list[list[int]]: ...

    @elements.setter
    def elements(self, arg: Sequence[Sequence[int]], /) -> None: ...

class ViolationManager:
    def __init__(self, arg: Model, /) -> None: ...

    def constraint_violation(self, arg: int, /) -> float: ...

    def total_violation(self) -> float: ...

    def augmented_objective(self) -> float: ...

    def is_feasible(self, tol: float = 1e-06) -> bool: ...

    def violated_constraints(self, tol: float = 1e-06) -> list[int]: ...

    def bump_weights(self, factor: float = 1.0) -> None: ...

    def weighted_violation_delta(self, var_id: int, j: float) -> float: ...

    def invalidate_cache(self) -> None: ...

    @property
    def weights(self) -> list[float]: ...

    @weights.setter
    def weights(self, arg: Sequence[float], /) -> None: ...

class RNG:
    def __init__(self, seed: int = 42) -> None: ...

    def uniform(self, arg0: float, arg1: float, /) -> float: ...

    def integers(self, arg0: int, arg1: int, /) -> int: ...

    def normal(self, arg0: float, arg1: float, /) -> float: ...

    def random(self) -> float: ...

    def seed(self) -> int: ...

class EditKind(enum.Enum):
    None_ = 0

    Replace = 1

    Swap = 2

    Reverse = 3

    MoveSegment = 4

    Insert = 5

    Erase = 6

    Assign = 7

class ElementEdit:
    @property
    def kind(self) -> EditKind: ...

    @property
    def from_pos(self) -> int: ...

    @property
    def to_pos(self) -> int: ...

    @property
    def length(self) -> int: ...

    @property
    def element(self) -> int: ...

class MoveChange:
    def __init__(self) -> None: ...

    @property
    def var_id(self) -> int: ...

    @var_id.setter
    def var_id(self, arg: int, /) -> None: ...

    @property
    def new_value(self) -> float: ...

    @new_value.setter
    def new_value(self, arg: float, /) -> None: ...

    @property
    def edits(self) -> list[ElementEdit]: ...

    @property
    def replacement(self) -> list[int]: ...

    def elements_after(self, elements: Sequence[int]) -> list[int]:
        """
        The elements this change produces from `elements` -- the absolute
        vector a change carried before #164. The engine edits in place.
        """

    def is_noop(self, elements: Sequence[int]) -> bool: ...

class Move:
    def __init__(self) -> None: ...

    @property
    def changes(self) -> list[MoveChange]: ...

    @changes.setter
    def changes(self, arg: Sequence[MoveChange], /) -> None: ...

    @property
    def move_type(self) -> str: ...

    @property
    def delta_F(self) -> float: ...

    @delta_F.setter
    def delta_F(self, arg: float, /) -> None: ...

class SavedValues:
    def __init__(self) -> None: ...

    @property
    def values(self) -> list[float]: ...

    @values.setter
    def values(self, arg: Sequence[float], /) -> None: ...

    @property
    def elements(self) -> list[list[int]]: ...

    @elements.setter
    def elements(self, arg: Sequence[Sequence[int]], /) -> None: ...

class LNS:
    def __init__(self, destroy_fraction: float = 0.3) -> None: ...

    def destroy_repair(self, model: Model, vm: ViolationManager, rng: RNG, repair_time_limit: float = 2.0) -> bool: ...

    def destroy_repair_cycle(self, model: Model, vm: ViolationManager, rng: RNG, n_rounds: int = 10, repair_time_limit: float = 2.0) -> int: ...

class Solution:
    def __init__(self) -> None: ...

    @property
    def state(self) -> ModelState: ...

    @state.setter
    def state(self, arg: ModelState, /) -> None: ...

    @property
    def objective(self) -> float: ...

    @objective.setter
    def objective(self, arg: float, /) -> None: ...

    @property
    def feasible(self) -> bool: ...

    @feasible.setter
    def feasible(self, arg: bool, /) -> None: ...

    @property
    def violation(self) -> float: ...

    @violation.setter
    def violation(self, arg: float, /) -> None: ...

class SolutionPool:
    def __init__(self, capacity: int = 10) -> None: ...

    def submit(self, arg: Solution, /) -> bool: ...

    def best(self) -> Solution | None: ...

    def top_k(self, arg: int, /) -> list[Solution]: ...

    def size(self) -> int: ...

    def best_feasible_objective(self) -> float: ...

class ParallelConfig:
    def __init__(self) -> None: ...

    @property
    def n_threads(self) -> int: ...

    @n_threads.setter
    def n_threads(self, arg: int, /) -> None: ...

    @property
    def pool_capacity(self) -> int: ...

    @pool_capacity.setter
    def pool_capacity(self, arg: int, /) -> None: ...

    @property
    def share_objective_bound(self) -> bool: ...

    @share_objective_bound.setter
    def share_objective_bound(self, arg: bool, /) -> None: ...

    @property
    def stop(self) -> bool:
        """
        A cbls.StopToken whose request() cancels every worker, or None. Reads
        back as a bool (whether one is attached), not as the token: the C++
        side holds a view, not the object. Same accumulating keep-alive as
        SearchConfig.stop -- see its docstring.
        """

    @stop.setter
    def stop(self, token: StopToken | None) -> None: ...

class SearchConfig:
    def __init__(self) -> None: ...

    @property
    def skip_init(self) -> bool: ...

    @skip_init.setter
    def skip_init(self, arg: bool, /) -> None: ...

    @property
    def max_iterations(self) -> int: ...

    @max_iterations.setter
    def max_iterations(self, arg: int, /) -> None: ...

    @property
    def lns_interval(self) -> int: ...

    @lns_interval.setter
    def lns_interval(self, arg: int, /) -> None: ...

    @property
    def structural_batch_probability(self) -> float: ...

    @structural_batch_probability.setter
    def structural_batch_probability(self, arg: float, /) -> None: ...

    @property
    def use_compound_moves(self) -> bool: ...

    @use_compound_moves.setter
    def use_compound_moves(self, arg: bool, /) -> None: ...

    @property
    def novelty_jump_probability(self) -> float: ...

    @novelty_jump_probability.setter
    def novelty_jump_probability(self, arg: float, /) -> None: ...

    @property
    def fpr_batch_probability(self) -> float: ...

    @fpr_batch_probability.setter
    def fpr_batch_probability(self, arg: float, /) -> None: ...

    @property
    def fpr_worker(self) -> int: ...

    @fpr_worker.setter
    def fpr_worker(self, arg: int, /) -> None: ...

    @property
    def lns_repair_fpr(self) -> bool: ...

    @lns_repair_fpr.setter
    def lns_repair_fpr(self, arg: bool, /) -> None: ...

    @property
    def fpr_dag_propagation(self) -> bool: ...

    @fpr_dag_propagation.setter
    def fpr_dag_propagation(self, arg: bool, /) -> None: ...

    @property
    def structural_selection(self) -> StructuralSelection: ...

    @structural_selection.setter
    def structural_selection(self, arg: StructuralSelection, /) -> None: ...

    @property
    def structural_sample_size(self) -> int: ...

    @structural_sample_size.setter
    def structural_sample_size(self, arg: int, /) -> None: ...

    @property
    def structural_neighbours(self) -> NeighbourList | None: ...

    @structural_neighbours.setter
    def structural_neighbours(self, arg: NeighbourList | None, /) -> None: ...

    @property
    def feasibility_tolerance(self) -> float: ...

    @feasibility_tolerance.setter
    def feasibility_tolerance(self, arg: float, /) -> None: ...

    @property
    def cone_cache_bytes(self) -> int: ...

    @cone_cache_bytes.setter
    def cone_cache_bytes(self, arg: int, /) -> None: ...

    @property
    def fj_objective_tie_break(self) -> bool: ...

    @fj_objective_tie_break.setter
    def fj_objective_tie_break(self, arg: bool, /) -> None: ...

    @property
    def stop(self) -> bool:
        """
        A cbls.StopToken whose request() ends this solve at its next batch
        boundary, or None. Reads back as a bool (whether one is attached).

        The token is kept alive by this config for as long as the config lives,
        which is what stops the C++ side reading a collected object. That tie
        only ACCUMULATES: a long-lived config assigned several tokens retains
        every one of them, and `config.stop = None` detaches the view but does
        not release the tie. Harmless -- it errs toward keeping objects alive --
        but build a fresh SearchConfig per solve if that matters.
        """

    @stop.setter
    def stop(self, token: StopToken | None) -> None: ...

class ParallelSearch:
    def __init__(self, n_threads: int = 0) -> None: ...

    def solve(self, model_factory: Callable[[], Model], time_limit: float = 10.0, seed: int = 42) -> SearchResult:
        """
        Run the parallel search and return the best result the workers found.

        The GIL is released for the duration of the C++ call, which is what makes
        this callable at all: the workers have to acquire the GIL to invoke a
        Python callable, and a caller holding it across the join deadlocks them.

        model_factory is invoked ONCE PER WORKER -- not once per restart -- each
        from that worker's own thread, so several calls are in flight at once. A
        progress callback is called from EVERY worker, serialized on one mutex
        inside the portfolio: rows arrive one at a time and in order, but on
        whichever worker thread produced them, and a slow callback throttles
        every worker rather than only its own. nanobind re-acquires
        the GIL around every
        invocation, so the interpreter itself is safe; what that does not give
        you is atomicity across bytecodes, so a callable that mutates
        Python-side state must lock it.

        `solve_parallel` accepts a Python hook_factory and lns_factory. Both are
        invoked once per worker, on that worker's own thread, immediately after
        model_factory. Both return a shared_ptr, so
        the C++ side shares ownership with the interpreter instead of adopting a
        pointer Python still owns (issue #129). The last reference is dropped
        before solve_parallel returns, under the GIL, at the end of the worker
        that built it.

        Return a FRESH object from every call. Each worker searches its own model
        on its own thread and the search never locks the hook or the LNS, so one
        shared instance that carries state is a data race. The old raw-pointer
        signature caught that as a double free; a shared_ptr accepts it silently.

        A FROZEN Model is the one sanctioned exception, and the way to avoid
        duplicating a large DAG per worker (#157). model_factory is declared to
        return a Model by value, so nanobind COPIES whatever object it hands back --
        and copying a frozen model shares its immutable structure while giving the
        worker its own variables, node values and objective bound. So
        `m.freeze()` once and `lambda: m` is correct here, where returning a shared
        MUTABLE model would not be: `freeze()` is what makes the shared half
        unwritable, and every structural call on it raises instead.

        What a Python factory can usefully return is NARROW. Neither
        InnerSolverHook nor LNS has a nanobind trampoline, so C++ dispatches
        through the base vtable: a Python subclass is constructed and released
        correctly, but an override of `solve` or `destroy_repair` is never called.
        The two fail differently, and the second is the trap. `solve` is not bound
        at all, so a Python `solve` is merely a new attribute nothing reads.
        `LNS.destroy_repair` IS bound, so an override does shadow it when called
        from Python -- and silently does not when the search calls it from C++.
        In practice these arguments buy a separately configured FloatIntensifyHook
        or LNS per worker, not a Python implementation of either. InnerSolverHook
        itself is abstract and exposes no constructor.

        The Model handed to hook_factory is a COPY of the worker's model, not a
        handle on it: nanobind casts an lvalue reference by copying. Mutating it
        changes nothing the worker will search. What that copy costs depends on
        the model: a FROZEN one shares its structure, so the copy is the
        variables and the node values; an open one is deep-copied whole, once per
        worker (#157).

        An exception raised by the factory in every worker is re-raised here with
        its original type and message, while one that fails in only some workers
        is absorbed and the survivors' result is returned.

        An exception raised by the progress callback's on_progress ends the solve
        of the worker that made the call -- NOT the portfolio. That worker is
        restarted, and stops after three consecutive failed attempts, or earlier
        when the time limit runs out or a peer settles a model that has no
        objective. A worker that stops that way without ever completing an
        attempt has FAILED. An exception is re-raised here -- the original
        object, with its type, message and traceback; the lowest-numbered failed
        worker's last one -- only if every worker failed, or if some worker failed
        and no worker returned or shared any result, feasible or not. Otherwise
        it is DISCARDED, with nothing reported, and the best result found is
        returned. That includes a failed worker's incumbents shared before it
        died, and a worker that raised after completing an attempt
        (SearchConfig.max_iterations makes that possible), which does not count
        as failed.

        Which workers call the callback at all depends on timing: a worker other
        than the first reports only a new portfolio-wide best, so a callback that
        raises on every call may kill the first worker alone and let the run
        return normally -- after which the periodic no-improvement rows, which
        only the first worker sends, stop. Raising is therefore not a way to stop
        a run; catch inside on_progress if a failure there must be seen. (The
        module-level, single-threaded cbls.solve differs: there a raising callback
        ends the search and the exception propagates at once.)
        """

    def solve_parallel(self, model_factory: Callable[[], Model], time_limit: float = 10.0, seed: int = 42, config: SearchConfig = ..., hook_factory: Callable[[Model], InnerSolverHook] | None = None, lns_factory: Callable[[], LNS] | None = None, callback: SolveCallback | None = None, par_config: ParallelConfig = ...) -> SearchResult:
        """
        Run the parallel search and return the best result the workers found.

        The GIL is released for the duration of the C++ call, which is what makes
        this callable at all: the workers have to acquire the GIL to invoke a
        Python callable, and a caller holding it across the join deadlocks them.

        model_factory is invoked ONCE PER WORKER -- not once per restart -- each
        from that worker's own thread, so several calls are in flight at once. A
        progress callback is called from EVERY worker, serialized on one mutex
        inside the portfolio: rows arrive one at a time and in order, but on
        whichever worker thread produced them, and a slow callback throttles
        every worker rather than only its own. nanobind re-acquires
        the GIL around every
        invocation, so the interpreter itself is safe; what that does not give
        you is atomicity across bytecodes, so a callable that mutates
        Python-side state must lock it.

        `solve_parallel` accepts a Python hook_factory and lns_factory. Both are
        invoked once per worker, on that worker's own thread, immediately after
        model_factory. Both return a shared_ptr, so
        the C++ side shares ownership with the interpreter instead of adopting a
        pointer Python still owns (issue #129). The last reference is dropped
        before solve_parallel returns, under the GIL, at the end of the worker
        that built it.

        Return a FRESH object from every call. Each worker searches its own model
        on its own thread and the search never locks the hook or the LNS, so one
        shared instance that carries state is a data race. The old raw-pointer
        signature caught that as a double free; a shared_ptr accepts it silently.

        A FROZEN Model is the one sanctioned exception, and the way to avoid
        duplicating a large DAG per worker (#157). model_factory is declared to
        return a Model by value, so nanobind COPIES whatever object it hands back --
        and copying a frozen model shares its immutable structure while giving the
        worker its own variables, node values and objective bound. So
        `m.freeze()` once and `lambda: m` is correct here, where returning a shared
        MUTABLE model would not be: `freeze()` is what makes the shared half
        unwritable, and every structural call on it raises instead.

        What a Python factory can usefully return is NARROW. Neither
        InnerSolverHook nor LNS has a nanobind trampoline, so C++ dispatches
        through the base vtable: a Python subclass is constructed and released
        correctly, but an override of `solve` or `destroy_repair` is never called.
        The two fail differently, and the second is the trap. `solve` is not bound
        at all, so a Python `solve` is merely a new attribute nothing reads.
        `LNS.destroy_repair` IS bound, so an override does shadow it when called
        from Python -- and silently does not when the search calls it from C++.
        In practice these arguments buy a separately configured FloatIntensifyHook
        or LNS per worker, not a Python implementation of either. InnerSolverHook
        itself is abstract and exposes no constructor.

        The Model handed to hook_factory is a COPY of the worker's model, not a
        handle on it: nanobind casts an lvalue reference by copying. Mutating it
        changes nothing the worker will search. What that copy costs depends on
        the model: a FROZEN one shares its structure, so the copy is the
        variables and the node values; an open one is deep-copied whole, once per
        worker (#157).

        An exception raised by the factory in every worker is re-raised here with
        its original type and message, while one that fails in only some workers
        is absorbed and the survivors' result is returned.

        An exception raised by the progress callback's on_progress ends the solve
        of the worker that made the call -- NOT the portfolio. That worker is
        restarted, and stops after three consecutive failed attempts, or earlier
        when the time limit runs out or a peer settles a model that has no
        objective. A worker that stops that way without ever completing an
        attempt has FAILED. An exception is re-raised here -- the original
        object, with its type, message and traceback; the lowest-numbered failed
        worker's last one -- only if every worker failed, or if some worker failed
        and no worker returned or shared any result, feasible or not. Otherwise
        it is DISCARDED, with nothing reported, and the best result found is
        returned. That includes a failed worker's incumbents shared before it
        died, and a worker that raised after completing an attempt
        (SearchConfig.max_iterations makes that possible), which does not count
        as failed.

        Which workers call the callback at all depends on timing: a worker other
        than the first reports only a new portfolio-wide best, so a callback that
        raises on every call may kill the first worker alone and let the run
        return normally -- after which the periodic no-improvement rows, which
        only the first worker sends, stop. Raising is therefore not a way to stop
        a run; catch inside on_progress if a failure there must be seen. (The
        module-level, single-threaded cbls.solve differs: there a raising callback
        ends the search and the exception propagates at once.)
        """

    def solve_master(self, model: Model, time_limit: float = 10.0, seed: int = 42, config: SearchConfig = ..., hook_factory: Callable[[Model], InnerSolverHook] | None = None, lns_factory: Callable[[], LNS] | None = None, callback: SolveCallback | None = None, par_config: ParallelConfig = ...) -> SearchResult:
        """
        Run the parallel search on ONE model and return the best result.

        The C++ `Model& master` entry point, which the factory forms cannot reach.
        `model` is FROZEN on this thread first and each worker searches a copy that
        shares its immutable structure (#157), so the model comes back frozen: build
        any further structure before calling this. hook_factory, lns_factory,
        callback and par_config behave as in solve_parallel (see there, including
        that hook_factory is handed a COPY of its worker's model, the GIL release
        and the exception contract).

        While this runs, structural writes to `model` from Python -- builders,
        close, freeze -- raise RuntimeError, as under cbls.solve.
        """

def load_model(path: str) -> Model:
    """Read a .cbls (JSON lines) model file."""

def save_model(model: Model, path: str) -> None:
    """Write `model` as a .cbls (JSON lines) file."""

class BoundPropagationStats:
    @property
    def passes(self) -> int: ...

    @property
    def n_tightened(self) -> int: ...

    @property
    def n_finitized(self) -> int: ...

    @property
    def n_fixed(self) -> int: ...

    @property
    def infeasible(self) -> bool: ...

    @property
    def hit_pass_limit(self) -> bool: ...

class MpsProblem:
    @property
    def name(self) -> str: ...

    @property
    def objective_row_name(self) -> str: ...

    @property
    def objective_offset(self) -> float: ...

    @property
    def maximize(self) -> bool: ...

    @property
    def num_vars(self) -> int: ...

    @property
    def num_rows(self) -> int: ...

    @property
    def num_nonzeros(self) -> int: ...

def read_mps(filename: str) -> MpsProblem:
    """Read an MPS file, fixed or free format; `.gz` is decompressed."""

class MpsToModelOptions:
    def __init__(self) -> None: ...

    @property
    def inf_clamp(self) -> float: ...

    @inf_clamp.setter
    def inf_clamp(self, arg: float, /) -> None: ...

    @property
    def propagate_bounds(self) -> bool: ...

    @propagate_bounds.setter
    def propagate_bounds(self, arg: bool, /) -> None: ...

    @property
    def max_propagation_passes(self) -> int: ...

    @max_propagation_passes.setter
    def max_propagation_passes(self, arg: int, /) -> None: ...

class MpsToModelResult:
    @property
    def model(self) -> Model: ...

    @property
    def var_handles(self) -> list[int]: ...

    @property
    def constraint_node_ids(self) -> list[int]: ...

    @property
    def objective_node_id(self) -> int: ...

    @property
    def bound_stats(self) -> BoundPropagationStats: ...

    @property
    def n_clamped_columns(self) -> int: ...

    def original_values(self) -> list[float]:
        """The model's current point as one value per source column."""

def mps_to_model(problem: MpsProblem, options: MpsToModelOptions = ...) -> MpsToModelResult:
    """
    Build a closed Model from a read_mps problem: one variable per column, one linear constraint per row.
    """

class NlProblem:
    @property
    def name(self) -> str: ...

    @property
    def n_vars(self) -> int: ...

    @property
    def n_cons(self) -> int: ...

    @property
    def n_objs(self) -> int: ...

    @property
    def n_defined_vars(self) -> int: ...

    @property
    def n_discrete_vars(self) -> int: ...

def read_nl(filename: str) -> NlProblem:
    """Read an AMPL .nl file, text or binary."""

class NlToModelOptions:
    def __init__(self) -> None: ...

    @property
    def propagate_bounds(self) -> bool: ...

    @propagate_bounds.setter
    def propagate_bounds(self, arg: bool, /) -> None: ...

    @property
    def max_propagation_passes(self) -> int: ...

    @max_propagation_passes.setter
    def max_propagation_passes(self, arg: int, /) -> None: ...

    @property
    def inf_clamp(self) -> float: ...

    @inf_clamp.setter
    def inf_clamp(self, arg: float, /) -> None: ...

    @property
    def int_inf_clamp(self) -> float: ...

    @int_inf_clamp.setter
    def int_inf_clamp(self, arg: float, /) -> None: ...

class NlToModelResult:
    @property
    def model(self) -> Model: ...

    @property
    def var_handles(self) -> list[int]: ...

    @property
    def constraint_node_ids(self) -> list[int]: ...

    @property
    def objective_node_id(self) -> int: ...

    @property
    def bound_stats(self) -> BoundPropagationStats: ...

    @property
    def n_clamped_columns(self) -> int: ...

    def original_values(self) -> list[float]:
        """The model's current point as one value per source column."""

    @property
    def supported(self) -> bool: ...

    @property
    def skipped_reasons(self) -> list[str]: ...

def nl_to_model(problem: NlProblem, options: NlToModelOptions = ...) -> NlToModelResult:
    """
    Build a Model from a read_nl problem. An unsupported operator does not raise: `supported` is False, `skipped_reasons` says why, and the model is not closed and must not be solved.
    """

def portfolio_worker_seed(base_seed: int, worker: int, restart: int) -> int:
    """
    RNG seed for one portfolio worker's one run: a splitmix64 finalizer over (base_seed, worker, restart). Mixed rather than added so that adjacent base seeds give genuinely different portfolios.
    """

def full_evaluate(arg: Model, /) -> float: ...

def delta_evaluate(arg0: Model, arg1: Set[int], /) -> float: ...

def compute_partial(arg0: Model, arg1: int, arg2: int, /) -> float: ...

def compute_all_partials(arg0: Model, arg1: int, /) -> list[float]: ...

def generate_standard_moves(arg0: Variable, arg1: RNG, /) -> list[Move]: ...

def apply_move(arg0: Model, arg1: Move, /) -> list[int]: ...

def save_move_values(arg0: Model, arg1: Move, /) -> SavedValues: ...

def undo_move(arg0: Model, arg1: Move, arg2: SavedValues, /) -> None: ...

class InnerSolverHook:
    pass

class FloatIntensifyHook(InnerSolverHook):
    def __init__(self) -> None: ...

    @property
    def max_sweeps(self) -> int: ...

    @max_sweeps.setter
    def max_sweeps(self, arg: int, /) -> None: ...

    @property
    def initial_step_size(self) -> float: ...

    @initial_step_size.setter
    def initial_step_size(self, arg: float, /) -> None: ...

    @property
    def max_line_search_steps(self) -> int: ...

    @max_line_search_steps.setter
    def max_line_search_steps(self, arg: int, /) -> None: ...

    @property
    def max_multi_var_constraints(self) -> int: ...

    @max_multi_var_constraints.setter
    def max_multi_var_constraints(self, arg: int, /) -> None: ...

class SolveProgress:
    def __init__(self) -> None: ...

    @property
    def iteration(self) -> int: ...

    @property
    def time_seconds(self) -> float: ...

    @property
    def objective(self) -> float: ...

    @property
    def total_violation(self) -> float: ...

    @property
    def feasible(self) -> bool: ...

    @property
    def new_best(self) -> bool: ...

    @property
    def perturbations(self) -> int: ...

class SolveCallback:
    def __init__(self) -> None: ...

    def on_progress(self, arg: SolveProgress, /) -> None: ...

def solve(model: Model, time_limit: float = 10.0, seed: int = 42, hook: InnerSolverHook | None = None, lns: LNS | None = None, lns_interval: int = 3, callback: SolveCallback | None = None, config: SearchConfig = ...) -> SearchResult:
    """
    Single-threaded solve.

    The GIL is RELEASED for the whole C++ call (#169). That is what lets another Python thread raise a cbls.StopToken mid-solve -- and it makes touching `model` from another thread while this runs a DATA RACE: the search writes variables and node values throughout and nothing locks them. Read the model only after this returns. STRUCTURAL writes to `model` -- its builders, close, freeze and Expr operators over it -- raise RuntimeError while this runs, from any thread, a SolveCallback included. A raw Python function passed to lambda_sum or pair_lambda_sum also re-acquires the GIL on every evaluation now; use the *_table_sum forms where the function is a table.

    An exception raised by callback.on_progress ends the search and propagates out of this call unchanged -- no result is returned, and the model is left at the assignment the search had reached -- with its internal objective bound still tightened if a feasible point had been found (the next solve resets it). (ParallelSearch.solve_parallel absorbs one instead unless every worker fails, or a worker fails and no worker returned any result; see its docstring.) NOTE: a Python subclass of InnerSolverHook or LNS is constructed and destroyed correctly, but its overrides are never called -- neither class has a nanobind trampoline, so C++ dispatches through the base vtable. These arguments configure the built-in classes; they do not let you implement one in Python.
    """

def initialize_random(model: Model, rng: RNG) -> None: ...

def initialize_structured_random(model: Model, rng: RNG) -> None: ...

def fj_nl_initialize(model: Model, vm: ViolationManager, max_iterations: int = 10000, rng: RNG | None = None, time_limit: float = 2.0, tolerance: float = 1e-06) -> int: ...
