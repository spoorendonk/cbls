"""CBLS: constraint-based local search over an expression DAG.

The public surface is `__all__`: models, variables, expressions and their math
functions, `solve`, its configuration, result and callbacks, and file I/O. The
engine's internals stay reachable as `cbls._cbls_core` (#464).
"""

from ._cbls_core import (
    BoundPropagationStats,
    Cover,
    Expr,
    ExprNode,
    GeneratorCounters,
    ListInit,
    ListPartition,
    Model,
    ModelState,
    MpsProblem,
    MpsToModelOptions,
    MpsToModelResult,
    NeighbourList,
    NlProblem,
    NlToModelOptions,
    NlToModelResult,
    NodeOp,
    SearchConfig,
    SearchCounters,
    SearchResult,
    SolveCallback,
    SolveProgress,
    StopToken,
    StructuralSelection,
    TerminationReason,
    Variable,
    VarType,
    WorkerFailure,
    ceil,
    cos,
    element,
    exp,
    floor,
    if_then_else,
    load_model,
    log,
    mps_to_model,
    nearest_neighbours,
    nl_to_model,
    read_mps,
    read_nl,
    save_model,
    sin,
    sqrt,
    tan,
)
from ._cbls_core import (
    abs as abs,  # explicit re-export, outside __all__
)
from ._cbls_core import (
    max as max,  # explicit re-export, outside __all__
)
from ._cbls_core import (
    min as min,  # explicit re-export, outside __all__
)
from ._cbls_core import (
    pow as pow,  # explicit re-export, outside __all__
)
from ._cbls_core import (
    round as round,  # explicit re-export, outside __all__
)
from ._cbls_core import version as _version
from ._solve import SolveResult, solve

__version__: str = _version

# abs, max, min, pow and round are left out: `from cbls import *` would shadow
# the builtins. They are still cbls.max and so on.
__all__ = [
    "BoundPropagationStats",
    "Cover",
    "Expr",
    "ExprNode",
    "GeneratorCounters",
    "ListInit",
    "ListPartition",
    "Model",
    "ModelState",
    "MpsProblem",
    "MpsToModelOptions",
    "MpsToModelResult",
    "NeighbourList",
    "NlProblem",
    "NlToModelOptions",
    "NlToModelResult",
    "NodeOp",
    "SearchConfig",
    "SearchCounters",
    "SearchResult",
    "SolveCallback",
    "SolveProgress",
    "SolveResult",
    "StopToken",
    "StructuralSelection",
    "TerminationReason",
    "VarType",
    "Variable",
    "WorkerFailure",
    "ceil",
    "cos",
    "element",
    "exp",
    "floor",
    "if_then_else",
    "load_model",
    "log",
    "mps_to_model",
    "nearest_neighbours",
    "nl_to_model",
    "read_mps",
    "read_nl",
    "save_model",
    "sin",
    "solve",
    "sqrt",
    "tan",
]
