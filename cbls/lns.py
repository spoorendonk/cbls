"""Large Neighborhood Search: destroy/repair for escaping local optima."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cbls.dag import (
    Variable, BoolVar, IntVar, FloatVar, ListVar, SetVar,
    full_evaluate, delta_evaluate,
)
from cbls.violation import ViolationManager
from cbls.search import fj_nl_initialize, initialize_random

if TYPE_CHECKING:
    from cbls.model import Model


class LNS:
    """Large Neighborhood Search: destroy subset of variables, repair."""

    def __init__(self, destroy_fraction: float = 0.3):
        self.destroy_fraction = destroy_fraction

    def destroy_repair(self, model: Model, violation_mgr: ViolationManager,
                       rng: np.random.Generator) -> bool:
        """Destroy and repair. Returns True if improvement found."""
        old_F = violation_mgr.augmented_objective()
        saved_state = model.copy_state()

        # 1. Select variables to destroy
        n_destroy = max(1, int(len(model.variables) * self.destroy_fraction))
        destroyed = list(rng.choice(model.variables, n_destroy, replace=False))

        # 2. Randomize destroyed variables
        for var in destroyed:
            if isinstance(var, BoolVar):
                var.value = float(rng.integers(0, 2))
            elif isinstance(var, IntVar):
                var.value = float(rng.integers(int(var.lb), int(var.ub) + 1))
            elif isinstance(var, FloatVar):
                var.value = float(rng.uniform(var.lb, var.ub))
            elif isinstance(var, ListVar):
                rng.shuffle(var.elements)
            elif isinstance(var, SetVar):
                size = rng.integers(var.min_size, var.max_size + 1)
                var.elements = set(rng.choice(var.universe_size, size=size, replace=False))

        full_evaluate(model)

        # 3. Repair via FJ-NL
        fj_nl_initialize(model, violation_mgr, max_iterations=2000, rng=rng)

        new_F = violation_mgr.augmented_objective()

        if new_F < old_F:
            return True  # keep improvement
        else:
            # Restore
            model.restore_state(saved_state)
            full_evaluate(model)
            return False

    def destroy_repair_cycle(self, model: Model, violation_mgr: ViolationManager,
                             rng: np.random.Generator, n_rounds: int = 10) -> int:
        """Run multiple destroy-repair rounds. Returns number of improvements."""
        improvements = 0
        for _ in range(n_rounds):
            if self.destroy_repair(model, violation_mgr, rng):
                improvements += 1
        return improvements
