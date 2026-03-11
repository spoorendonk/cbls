"""Model: Hexaly-style builder API for constructing expression DAGs."""

from __future__ import annotations

from cbls.dag import (
    Variable, BoolVar, IntVar, FloatVar, ListVar, SetVar,
    ExprNode, ConstNode, NegNode, SumNode, ProdNode, DivNode, PowNode,
    MinNode, MaxNode, AbsNode, SinNode, CosNode, IfNode,
    AtNode, CountNode, LambdaNode, LeqNode, EqNode,
    topological_sort, full_evaluate,
)


class Model:
    """Constraint-Based Local Search model with Hexaly-style builder API."""

    def __init__(self):
        self.variables: list[Variable | ListVar | SetVar] = []
        self.expressions: list[ExprNode] = []  # topological order (set by close())
        self.constraints: list[ExprNode] = []  # expressions that must be ≤ 0
        self.objective: ExprNode | None = None
        self._all_nodes: dict[int, ExprNode] = {}
        self._next_var_id = 0
        self._next_node_id = 0
        self._closed = False

    # ------------------------------------------------------------------
    # Variable creation
    # ------------------------------------------------------------------

    def _new_var_id(self) -> int:
        vid = self._next_var_id
        self._next_var_id += 1
        return vid

    def _new_node_id(self) -> int:
        nid = self._next_node_id
        self._next_node_id += 1
        return nid

    def bool_var(self, name: str = "") -> BoolVar:
        v = BoolVar(self._new_var_id(), name)
        self.variables.append(v)
        return v

    def int_var(self, lb: int, ub: int, name: str = "") -> IntVar:
        v = IntVar(self._new_var_id(), lb, ub, name)
        self.variables.append(v)
        return v

    def float_var(self, lb: float, ub: float, name: str = "") -> FloatVar:
        v = FloatVar(self._new_var_id(), lb, ub, name)
        self.variables.append(v)
        return v

    def list_var(self, n: int, name: str = "") -> ListVar:
        v = ListVar(self._new_var_id(), n, name)
        self.variables.append(v)
        return v

    def set_var(self, n: int, min_size: int = 0, max_size: int | None = None,
                name: str = "") -> SetVar:
        v = SetVar(self._new_var_id(), n, min_size, max_size, name)
        self.variables.append(v)
        return v

    # ------------------------------------------------------------------
    # Expression building
    # ------------------------------------------------------------------

    def _wrap(self, x) -> ExprNode | Variable:
        """Wrap a numeric constant or variable into a node."""
        if isinstance(x, (ExprNode, Variable, ListVar, SetVar)):
            return x
        # Numeric constant
        node = ConstNode(self._new_node_id(), float(x))
        self._all_nodes[node.node_id] = node
        return node

    def _make_node(self, cls, children):
        """Create an expression node and register it."""
        wrapped = [self._wrap(c) for c in children]
        node = cls(self._new_node_id(), wrapped)
        self._all_nodes[node.node_id] = node
        return node

    def const(self, value: float) -> ConstNode:
        return self._wrap(value)

    def neg(self, x) -> NegNode:
        return self._make_node(NegNode, [x])

    def sum(self, *args) -> ExprNode:
        if len(args) == 1 and hasattr(args[0], '__iter__') and not isinstance(args[0], (ExprNode, Variable)):
            args = tuple(args[0])
        if len(args) == 0:
            return self._wrap(0.0)
        if len(args) == 1:
            # Always wrap in a SumNode so the result is an ExprNode
            return self._make_node(SumNode, list(args))
        return self._make_node(SumNode, list(args))

    def prod(self, a, b) -> ProdNode:
        return self._make_node(ProdNode, [a, b])

    def div(self, a, b) -> DivNode:
        return self._make_node(DivNode, [a, b])

    def pow(self, base, exp) -> PowNode:
        return self._make_node(PowNode, [base, exp])

    def min(self, *args) -> MinNode:
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            args = tuple(args[0])
        return self._make_node(MinNode, list(args))

    def max(self, *args) -> MaxNode:
        if len(args) == 1 and hasattr(args[0], '__iter__'):
            args = tuple(args[0])
        return self._make_node(MaxNode, list(args))

    def abs(self, x) -> AbsNode:
        return self._make_node(AbsNode, [x])

    def sin(self, x) -> SinNode:
        return self._make_node(SinNode, [x])

    def cos(self, x) -> CosNode:
        return self._make_node(CosNode, [x])

    def if_then_else(self, cond, then_, else_) -> IfNode:
        return self._make_node(IfNode, [cond, then_, else_])

    def at(self, list_var: ListVar, index) -> AtNode:
        """Index into a ListVar."""
        idx = self._wrap(index)
        node = AtNode(self._new_node_id(), [list_var, idx])
        self._all_nodes[node.node_id] = node
        return node

    def count(self, var) -> CountNode:
        node = CountNode(self._new_node_id(), [var])
        self._all_nodes[node.node_id] = node
        return node

    def lambda_sum(self, list_var: ListVar, func) -> LambdaNode:
        """Sum over list elements: Σ func(element)."""
        node = LambdaNode(self._new_node_id(), list_var, func)
        self._all_nodes[node.node_id] = node
        return node

    def leq(self, a, b) -> LeqNode:
        """a ≤ b constraint expression (returns a - b, must be ≤ 0)."""
        return self._make_node(LeqNode, [a, b])

    def eq(self, a, b) -> EqNode:
        """Equality constraint expression (returns |a - b|, must be = 0)."""
        return self._make_node(EqNode, [a, b])

    # ------------------------------------------------------------------
    # Constraint and objective
    # ------------------------------------------------------------------

    def add_constraint(self, expr) -> None:
        """Add constraint: expr ≤ 0."""
        wrapped = self._wrap(expr)
        self.constraints.append(wrapped)

    def minimize(self, expr) -> None:
        self.objective = self._wrap(expr)

    def maximize(self, expr) -> None:
        self.objective = self.neg(expr)

    # ------------------------------------------------------------------
    # Model finalization
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalize model: topological sort and initial evaluation."""
        topological_sort(self)
        full_evaluate(self)
        self._closed = True

    def copy_state(self) -> dict:
        """Snapshot current variable values."""
        state = {}
        for var in self.variables:
            if isinstance(var, ListVar):
                state[var.var_id] = var.elements.copy()
            elif isinstance(var, SetVar):
                state[var.var_id] = var.elements.copy()
            else:
                state[var.var_id] = var.value
        return state

    def restore_state(self, state: dict) -> None:
        """Restore variable values from snapshot."""
        for var in self.variables:
            if isinstance(var, ListVar):
                var.elements = state[var.var_id].copy()
            elif isinstance(var, SetVar):
                var.elements = state[var.var_id].copy()
            else:
                var.value = state[var.var_id]
