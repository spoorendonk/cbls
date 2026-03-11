"""Expression DAG: construction, topological sort, full and delta evaluation."""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbls.model import Model

import numpy as np


# ---------------------------------------------------------------------------
# Variable types
# ---------------------------------------------------------------------------

class Variable:
    """Base class for all decision variables."""

    __slots__ = ("var_id", "value", "lb", "ub", "dependents", "name")

    def __init__(self, var_id: int, lb: float, ub: float, name: str = ""):
        self.var_id = var_id
        self.value: float = lb
        self.lb = lb
        self.ub = ub
        self.dependents: list[ExprNode] = []
        self.name = name

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.var_id}, val={self.value})"


class BoolVar(Variable):
    """Binary variable: domain {0, 1}."""

    def __init__(self, var_id: int, name: str = ""):
        super().__init__(var_id, 0.0, 1.0, name)


class IntVar(Variable):
    """Integer variable: domain [lb, ub] ⊆ ℤ."""

    def __init__(self, var_id: int, lb: int, ub: int, name: str = ""):
        super().__init__(var_id, float(lb), float(ub), name)
        self.value = float(lb)


class FloatVar(Variable):
    """Continuous variable: domain [lb, ub] ⊆ ℝ."""

    def __init__(self, var_id: int, lb: float, ub: float, name: str = ""):
        super().__init__(var_id, lb, ub, name)
        self.value = lb


class ListVar:
    """Ordered subset/permutation of {0..n-1}."""

    __slots__ = ("var_id", "elements", "max_size", "dependents", "name")

    def __init__(self, var_id: int, n: int, name: str = ""):
        self.var_id = var_id
        self.elements: list[int] = list(range(n))
        self.max_size = n
        self.dependents: list[ExprNode] = []
        self.name = name

    def __repr__(self):
        return f"ListVar(id={self.var_id}, elems={self.elements})"


class SetVar:
    """Unordered subset of {0..n-1}."""

    __slots__ = ("var_id", "elements", "universe_size", "min_size", "max_size",
                 "dependents", "name")

    def __init__(self, var_id: int, n: int, min_size: int = 0,
                 max_size: int | None = None, name: str = ""):
        self.var_id = var_id
        self.elements: set[int] = set()
        self.universe_size = n
        self.min_size = min_size
        self.max_size = max_size if max_size is not None else n
        self.dependents: list[ExprNode] = []
        self.name = name

    def __repr__(self):
        return f"SetVar(id={self.var_id}, elems={self.elements})"


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

class ExprNode:
    """Base node in the expression DAG."""

    __slots__ = ("node_id", "value", "children", "parents")

    def __init__(self, node_id: int, children: list):
        self.node_id = node_id
        self.value: float = 0.0
        self.children: list = children  # ExprNode | Variable
        self.parents: list[ExprNode] = []

    def evaluate(self) -> float:
        raise NotImplementedError

    def local_derivative(self, child_index: int) -> float:
        """∂self/∂child_i evaluated at current values."""
        raise NotImplementedError

    def _child_val(self, i: int) -> float:
        c = self.children[i]
        if isinstance(c, (ExprNode,)):
            return c.value
        elif isinstance(c, (Variable,)):
            return c.value
        return float(c)  # numeric constant stored directly

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.node_id}, val={self.value})"


class ConstNode(ExprNode):
    """Constant value."""

    def __init__(self, node_id: int, const_value: float):
        super().__init__(node_id, [])
        self.value = const_value

    def evaluate(self) -> float:
        return self.value

    def local_derivative(self, child_index: int) -> float:
        return 0.0


class NegNode(ExprNode):
    """Negation: -child."""

    def evaluate(self) -> float:
        return -self._child_val(0)

    def local_derivative(self, child_index: int) -> float:
        return -1.0


class SumNode(ExprNode):
    """Sum of children."""

    def evaluate(self) -> float:
        return sum(self._child_val(i) for i in range(len(self.children)))

    def local_derivative(self, child_index: int) -> float:
        return 1.0


class ProdNode(ExprNode):
    """Product of two children."""

    def evaluate(self) -> float:
        return self._child_val(0) * self._child_val(1)

    def local_derivative(self, child_index: int) -> float:
        # ∂(a*b)/∂a = b, ∂(a*b)/∂b = a
        other = 1 - child_index
        return self._child_val(other)


class DivNode(ExprNode):
    """Division: child0 / child1."""

    def evaluate(self) -> float:
        denom = self._child_val(1)
        if abs(denom) < 1e-15:
            return 1e15 if self._child_val(0) >= 0 else -1e15
        return self._child_val(0) / denom

    def local_derivative(self, child_index: int) -> float:
        if child_index == 0:
            denom = self._child_val(1)
            return 1.0 / denom if abs(denom) > 1e-15 else 0.0
        else:
            denom = self._child_val(1)
            if abs(denom) < 1e-15:
                return 0.0
            return -self._child_val(0) / (denom * denom)


class PowNode(ExprNode):
    """Power: child0 ** child1 (child1 typically a constant)."""

    def evaluate(self) -> float:
        base = self._child_val(0)
        exp = self._child_val(1)
        try:
            return base ** exp
        except (ValueError, OverflowError):
            return 1e15

    def local_derivative(self, child_index: int) -> float:
        base = self._child_val(0)
        exp = self._child_val(1)
        if child_index == 0:
            # ∂(b^e)/∂b = e * b^(e-1)
            if abs(base) < 1e-15 and exp < 1:
                return 0.0
            try:
                return exp * base ** (exp - 1)
            except (ValueError, OverflowError):
                return 0.0
        else:
            # ∂(b^e)/∂e = b^e * ln(b)
            if base <= 0:
                return 0.0
            try:
                return (base ** exp) * math.log(base)
            except (ValueError, OverflowError):
                return 0.0


class MinNode(ExprNode):
    """Minimum of children."""

    def evaluate(self) -> float:
        return min(self._child_val(i) for i in range(len(self.children)))

    def local_derivative(self, child_index: int) -> float:
        min_val = self.value
        return 1.0 if abs(self._child_val(child_index) - min_val) < 1e-12 else 0.0


class MaxNode(ExprNode):
    """Maximum of children."""

    def evaluate(self) -> float:
        return max(self._child_val(i) for i in range(len(self.children)))

    def local_derivative(self, child_index: int) -> float:
        max_val = self.value
        return 1.0 if abs(self._child_val(child_index) - max_val) < 1e-12 else 0.0


class AbsNode(ExprNode):
    """Absolute value: |child|."""

    def evaluate(self) -> float:
        return abs(self._child_val(0))

    def local_derivative(self, child_index: int) -> float:
        v = self._child_val(0)
        if v > 0:
            return 1.0
        elif v < 0:
            return -1.0
        return 0.0


class SinNode(ExprNode):
    """sin(child)."""

    def evaluate(self) -> float:
        return math.sin(self._child_val(0))

    def local_derivative(self, child_index: int) -> float:
        return math.cos(self._child_val(0))


class CosNode(ExprNode):
    """cos(child)."""

    def evaluate(self) -> float:
        return math.cos(self._child_val(0))

    def local_derivative(self, child_index: int) -> float:
        return -math.sin(self._child_val(0))


class IfNode(ExprNode):
    """If cond > 0 then child1 else child2."""

    def evaluate(self) -> float:
        cond = self._child_val(0)
        return self._child_val(1) if cond > 0 else self._child_val(2)

    def local_derivative(self, child_index: int) -> float:
        if child_index == 0:
            return 0.0  # non-differentiable w.r.t. condition
        cond = self._child_val(0)
        if child_index == 1:
            return 1.0 if cond > 0 else 0.0
        return 0.0 if cond > 0 else 1.0


class AtNode(ExprNode):
    """Indexing into a ListVar: list_var.elements[index].
    children[0] = ListVar, children[1] = index expression."""

    def evaluate(self) -> float:
        list_var = self.children[0]
        idx = int(self._child_val(1))
        if isinstance(list_var, ListVar):
            if 0 <= idx < len(list_var.elements):
                return float(list_var.elements[idx])
            return 0.0
        return 0.0

    def local_derivative(self, child_index: int) -> float:
        return 0.0  # discrete — not differentiable


class CountNode(ExprNode):
    """Length of a ListVar or SetVar."""

    def evaluate(self) -> float:
        var = self.children[0]
        if isinstance(var, ListVar):
            return float(len(var.elements))
        elif isinstance(var, SetVar):
            return float(len(var.elements))
        return 0.0

    def local_derivative(self, child_index: int) -> float:
        return 0.0


class LambdaNode(ExprNode):
    """Sum over list elements: Σ f(list_var.elements[i]).
    children[0] = ListVar, the function is stored as self.func.
    func takes an element value and returns a float."""

    __slots__ = ("node_id", "value", "children", "parents", "func")

    def __init__(self, node_id: int, list_var: ListVar, func):
        super().__init__(node_id, [list_var])
        self.func = func

    def evaluate(self) -> float:
        list_var = self.children[0]
        if isinstance(list_var, ListVar):
            return sum(self.func(e) for e in list_var.elements)
        return 0.0

    def local_derivative(self, child_index: int) -> float:
        return 0.0  # discrete


# Comparison/logical nodes for constraints
class LeqNode(ExprNode):
    """child0 - child1 (returns value ≤ 0 when child0 ≤ child1)."""

    def evaluate(self) -> float:
        return self._child_val(0) - self._child_val(1)

    def local_derivative(self, child_index: int) -> float:
        return 1.0 if child_index == 0 else -1.0


class EqNode(ExprNode):
    """Equality violation: |child0 - child1| (returns 0 when equal)."""

    def evaluate(self) -> float:
        return abs(self._child_val(0) - self._child_val(1))

    def local_derivative(self, child_index: int) -> float:
        diff = self._child_val(0) - self._child_val(1)
        sign = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        return sign if child_index == 0 else -sign


# ---------------------------------------------------------------------------
# DAG operations
# ---------------------------------------------------------------------------

def _register_parents(nodes: list[ExprNode]):
    """Set up parent pointers from child→parent."""
    for node in nodes:
        for child in node.children:
            if isinstance(child, ExprNode):
                if node not in child.parents:
                    child.parents.append(node)
            elif isinstance(child, (Variable, ListVar, SetVar)):
                if node not in child.dependents:
                    child.dependents.append(node)


def topological_sort(model: Model) -> list[ExprNode]:
    """Sort all DAG nodes so children come before parents. Uses Kahn's algorithm."""
    all_nodes = list(model._all_nodes.values())
    _register_parents(all_nodes)

    # Compute in-degrees (only counting ExprNode children)
    in_degree: dict[int, int] = {}
    for node in all_nodes:
        in_degree[node.node_id] = 0
    for node in all_nodes:
        for child in node.children:
            if isinstance(child, ExprNode):
                in_degree.setdefault(node.node_id, 0)
                # node depends on child, so node has an in-edge from child
                # Actually: in topological sort, we want children before parents.
                # in_degree counts how many ExprNode children a node has.
                pass

    # Recompute: in_degree[n] = number of ExprNode children of n
    in_degree = {}
    child_to_parents: dict[int, list[ExprNode]] = {}
    for node in all_nodes:
        in_degree[node.node_id] = 0
        for child in node.children:
            if isinstance(child, ExprNode):
                in_degree[node.node_id] = in_degree.get(node.node_id, 0) + 1
                child_to_parents.setdefault(child.node_id, []).append(node)

    # Recount properly
    in_degree = {n.node_id: 0 for n in all_nodes}
    for node in all_nodes:
        for child in node.children:
            if isinstance(child, ExprNode):
                in_degree[node.node_id] += 1

    queue = deque()
    for node in all_nodes:
        if in_degree[node.node_id] == 0:
            queue.append(node)

    sorted_nodes: list[ExprNode] = []
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for parent in child_to_parents.get(node.node_id, []):
            in_degree[parent.node_id] -= 1
            if in_degree[parent.node_id] == 0:
                queue.append(parent)

    model.expressions = sorted_nodes
    return sorted_nodes


def full_evaluate(model: Model) -> float:
    """Evaluate all nodes bottom-up. O(total DAG size)."""
    for node in model.expressions:
        node.value = node.evaluate()
    if model.objective is not None:
        return model.objective.value
    return 0.0


def delta_evaluate(model: Model, changed_vars: set) -> float:
    """Recompute only dirty subgraph after variable changes. O(affected subgraph)."""
    # 1. Mark dependents of changed variables as dirty (BFS up the DAG)
    dirty: set[int] = set()
    queue: deque = deque()
    for var in changed_vars:
        deps = var.dependents if isinstance(var, (Variable, ListVar, SetVar)) else []
        for dep in deps:
            if dep.node_id not in dirty:
                dirty.add(dep.node_id)
                queue.append(dep)
    while queue:
        node = queue.popleft()
        for parent in node.parents:
            if parent.node_id not in dirty:
                dirty.add(parent.node_id)
                queue.append(parent)

    # 2. Recompute dirty nodes in topological order
    for node in model.expressions:
        if node.node_id in dirty:
            node.value = node.evaluate()

    if model.objective is not None:
        return model.objective.value
    return 0.0


def compute_partial(model: Model, expr: ExprNode, var: Variable) -> float:
    """Compute ∂expr/∂var via reverse-mode AD on the DAG."""
    adjoint: dict[int, float] = {expr.node_id: 1.0}

    for node in reversed(model.expressions):
        if node.node_id not in adjoint:
            continue
        adj = adjoint[node.node_id]

        for i, child in enumerate(node.children):
            local_d = node.local_derivative(i)
            if isinstance(child, ExprNode):
                adjoint[child.node_id] = adjoint.get(child.node_id, 0.0) + adj * local_d
            elif isinstance(child, Variable):
                key = ("var", child.var_id)
                adjoint[key] = adjoint.get(key, 0.0) + adj * local_d

    return adjoint.get(("var", var.var_id), 0.0)
