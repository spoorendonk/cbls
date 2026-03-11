"""SA core + FJ-NL initialization."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from cbls.dag import (
    Variable, BoolVar, IntVar, FloatVar, ListVar, SetVar,
    full_evaluate, delta_evaluate, compute_partial,
)
from cbls.violation import ViolationManager
from cbls.moves import (
    Move, generate_standard_moves, newton_tight_move, gradient_lift_move,
    apply_move, undo_move, save_move_values, MoveProbabilities,
)

if TYPE_CHECKING:
    from cbls.model import Model


@dataclass
class SearchResult:
    """Result of a search run."""
    objective: float = float("inf")
    feasible: bool = False
    best_state: dict = field(default_factory=dict)
    iterations: int = 0
    time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_random(model: Model, rng: np.random.Generator):
    """Random initialization of all variables."""
    for var in model.variables:
        if isinstance(var, BoolVar):
            var.value = float(rng.integers(0, 2))
        elif isinstance(var, IntVar):
            var.value = float(rng.integers(int(var.lb), int(var.ub) + 1))
        elif isinstance(var, FloatVar):
            var.value = float(rng.uniform(var.lb, var.ub))
        elif isinstance(var, ListVar):
            var.elements = list(rng.permutation(var.max_size))
        elif isinstance(var, SetVar):
            size = rng.integers(var.min_size, var.max_size + 1)
            var.elements = set(rng.choice(var.universe_size, size=size, replace=False))


def fj_nl_initialize(model: Model, violation_mgr: ViolationManager,
                     max_iterations: int = 10000, rng: np.random.Generator | None = None):
    """FJ-NL: Nonlinear Feasibility Jump — greedy violation descent."""
    if rng is None:
        rng = np.random.default_rng(42)

    full_evaluate(model)

    for iteration in range(max_iterations):
        violated = violation_mgr.violated_constraints()
        if not violated:
            break  # feasible

        best_var = None
        best_val = None
        best_reduction = 0.0

        for var in model.variables:
            candidates = _fj_candidate_values(var, model, violation_mgr, rng)
            for val in candidates:
                saved = _save_var(var)
                _set_var(var, val)
                delta_evaluate(model, {var})
                new_viol = violation_mgr.total_violation()
                _set_var(var, saved)
                delta_evaluate(model, {var})

                old_viol = violation_mgr.total_violation()
                reduction = old_viol - new_viol
                if reduction > best_reduction:
                    best_var, best_val, best_reduction = var, val, reduction

        if best_var is None:
            # Stagnation: bump weights
            violation_mgr.bump_weights()
            # Try random perturbation
            var = rng.choice(model.variables)
            if isinstance(var, (BoolVar, IntVar, FloatVar)):
                candidates = _fj_candidate_values(var, model, violation_mgr, rng)
                if candidates:
                    _set_var(var, candidates[0])
                    delta_evaluate(model, {var})
            continue

        _set_var(best_var, best_val)
        delta_evaluate(model, {best_var})


def _fj_candidate_values(var, model, violation_mgr, rng):
    """Generate candidate values for FJ-NL."""
    if isinstance(var, BoolVar):
        return [1.0 - var.value]
    elif isinstance(var, IntVar):
        domain_size = int(var.ub - var.lb) + 1
        if domain_size <= 20:
            return [float(v) for v in range(int(var.lb), int(var.ub) + 1) if v != var.value]
        else:
            # Sample a few values
            vals = set()
            if var.value > var.lb:
                vals.add(var.value - 1)
            if var.value < var.ub:
                vals.add(var.value + 1)
            for _ in range(8):
                vals.add(float(rng.integers(int(var.lb), int(var.ub) + 1)))
            vals.discard(var.value)
            return list(vals)
    elif isinstance(var, FloatVar):
        vals = list(np.linspace(var.lb, var.ub, 10))
        # Gradient-based candidate
        violated = violation_mgr.violated_constraints()
        if violated:
            for ci in violated[:3]:
                constraint = model.constraints[ci]
                dg = compute_partial(model, constraint, var)
                if abs(dg) > 1e-12:
                    step = -constraint.value / dg
                    vals.append(float(np.clip(var.value + step, var.lb, var.ub)))
        return vals
    return []


def _save_var(var):
    if isinstance(var, ListVar):
        return var.elements.copy()
    elif isinstance(var, SetVar):
        return var.elements.copy()
    return var.value


def _set_var(var, val):
    if isinstance(var, ListVar):
        var.elements = val if isinstance(val, list) else list(val)
    elif isinstance(var, SetVar):
        var.elements = val if isinstance(val, set) else set(val)
    else:
        var.value = val


# ---------------------------------------------------------------------------
# SA Core
# ---------------------------------------------------------------------------

def solve(model: Model, time_limit: float = 10.0, seed: int = 42,
          use_fj: bool = True) -> SearchResult:
    """Main solver entry point: FJ-NL init → SA with enriched moves."""
    rng = np.random.default_rng(seed)
    violation_mgr = ViolationManager(model)

    start_time = time.time()
    deadline = start_time + time_limit

    # Initialize
    initialize_random(model, rng)
    full_evaluate(model)

    if use_fj:
        fj_nl_initialize(model, violation_mgr, max_iterations=5000, rng=rng)

    current_F = violation_mgr.augmented_objective()
    best_F = current_F
    best_feasible_obj = float("inf")
    best_state = model.copy_state()

    if violation_mgr.is_feasible():
        best_feasible_obj = model.objective.value if model.objective else 0.0

    # SA parameters
    temperature = _initial_temperature(best_F)
    cooling_rate = 0.9999
    reheat_interval = 5000

    # Move probability tuning
    move_probs = MoveProbabilities([
        "flip", "int_dec", "int_inc", "int_rand",
        "float_perturb", "list_swap", "list_2opt",
        "set_add", "set_remove", "set_swap",
        "newton_tight", "gradient_lift",
    ])

    iteration = 0
    while time.time() < deadline:
        # Select variable
        var = rng.choice(model.variables)

        # Generate moves
        moves = generate_standard_moves(var, rng)

        # Enriched moves for FloatVar
        if isinstance(var, FloatVar):
            violated = violation_mgr.violated_constraints()
            if violated:
                ci = rng.choice(violated)
                moves.extend(newton_tight_move(var, model, ci))
            moves.extend(gradient_lift_move(var, model))

        if not moves:
            iteration += 1
            continue

        # Pick a move (uniform for now, could use move_probs)
        move = moves[rng.integers(0, len(moves))]

        # Evaluate via delta
        saved = save_move_values(move)
        old_F = violation_mgr.augmented_objective()
        changed = apply_move(move)
        delta_evaluate(model, changed)
        new_F = violation_mgr.augmented_objective()
        delta_F = new_F - old_F

        # SA acceptance
        accept = False
        if delta_F <= 0:
            accept = True
        elif temperature > 1e-15:
            p = math.exp(-delta_F / temperature)
            accept = rng.random() < p

        if accept:
            # Update tracking
            obj_improved = False
            if violation_mgr.is_feasible():
                obj_val = model.objective.value if model.objective else 0.0
                if obj_val < best_feasible_obj:
                    best_feasible_obj = obj_val
                    best_state = model.copy_state()
                    obj_improved = True

            if new_F < best_F:
                best_F = new_F
                if not violation_mgr.is_feasible():
                    best_state = model.copy_state()

            move_probs.update(move.move_type, True)
            violation_mgr.adaptive_lambda.update(
                violation_mgr.is_feasible(), obj_improved
            )
        else:
            # Reject: undo
            undo_move(move, saved)
            delta_evaluate(model, changed)
            move_probs.update(move.move_type, False)

        # Cooling
        temperature *= cooling_rate
        if iteration > 0 and iteration % reheat_interval == 0:
            temperature = _initial_temperature(best_F) * 0.5

        iteration += 1

    # Restore best
    model.restore_state(best_state)
    full_evaluate(model)

    return SearchResult(
        objective=best_feasible_obj if best_feasible_obj < float("inf") else best_F,
        feasible=best_feasible_obj < float("inf"),
        best_state=best_state,
        iterations=iteration,
        time_seconds=time.time() - start_time,
    )


def _initial_temperature(F: float) -> float:
    """Compute initial SA temperature based on current objective."""
    return max(abs(F) * 0.1, 1.0)
