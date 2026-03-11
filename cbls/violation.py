"""Penalty objective F = f + λ·V with adaptive λ."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cbls.model import Model
    from cbls.dag import ExprNode


class ViolationManager:
    """Manages constraint violations and augmented objective."""

    def __init__(self, model: Model):
        self.model = model
        self.weights: list[float] = [1.0] * len(model.constraints)
        self.adaptive_lambda = AdaptiveLambda()

    @property
    def lambda_(self) -> float:
        return self.adaptive_lambda.lambda_

    def constraint_violation(self, i: int) -> float:
        """Violation of constraint i: max(0, expr_i.value)."""
        return max(0.0, self.model.constraints[i].value)

    def total_violation(self) -> float:
        """V(x) = Σ max(0, constraint_i.value) * weight_i."""
        return sum(
            max(0.0, c.value) * w
            for c, w in zip(self.model.constraints, self.weights)
        )

    def augmented_objective(self) -> float:
        """F(x) = f(x) + λ · V(x)."""
        obj = self.model.objective.value if self.model.objective is not None else 0.0
        return obj + self.lambda_ * self.total_violation()

    def is_feasible(self, tol: float = 1e-9) -> bool:
        return all(c.value <= tol for c in self.model.constraints)

    def violated_constraints(self, tol: float = 1e-9) -> list[int]:
        return [i for i, c in enumerate(self.model.constraints) if c.value > tol]

    def bump_weights(self, factor: float = 1.0):
        """Increase weights on violated constraints (FJ-style)."""
        for i in self.violated_constraints():
            self.weights[i] += factor


class AdaptiveLambda:
    """Adapt λ based on search progress."""

    def __init__(self, initial_lambda: float = 1.0):
        self.lambda_: float = initial_lambda
        self.consecutive_infeasible: int = 0
        self.consecutive_feasible_stuck: int = 0

    def update(self, is_feasible: bool, obj_improved: bool):
        if not is_feasible:
            self.consecutive_infeasible += 1
            self.consecutive_feasible_stuck = 0
            if self.consecutive_infeasible > 10:
                self.lambda_ *= 1.5
                self.consecutive_infeasible = 0
        else:
            self.consecutive_infeasible = 0
            if not obj_improved:
                self.consecutive_feasible_stuck += 1
                if self.consecutive_feasible_stuck > 20:
                    self.lambda_ *= 0.8
                    self.consecutive_feasible_stuck = 0
            else:
                self.consecutive_feasible_stuck = 0
