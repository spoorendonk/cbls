"""Move types: standard + Newton tight + gradient lift."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cbls.model import Model

from cbls.dag import (
    Variable, BoolVar, IntVar, FloatVar, ListVar, SetVar,
    compute_partial,
)


@dataclass
class Move:
    """A candidate move: change variable(s) to new value(s)."""
    changes: list[tuple[Any, Any]]  # [(var, new_value), ...]
    move_type: str = ""
    delta_F: float = 0.0


# ---------------------------------------------------------------------------
# Standard move generators
# ---------------------------------------------------------------------------

def bool_moves(var: BoolVar, rng) -> list[Move]:
    return [Move([(var, 1.0 - var.value)], "flip")]


def int_moves(var: IntVar, rng) -> list[Move]:
    moves = []
    if var.value > var.lb:
        moves.append(Move([(var, var.value - 1)], "int_dec"))
    if var.value < var.ub:
        moves.append(Move([(var, var.value + 1)], "int_inc"))
    moves.append(Move([(var, float(rng.integers(int(var.lb), int(var.ub) + 1)))], "int_rand"))
    return moves


def float_moves(var: FloatVar, rng, sigma_frac: float = 0.1) -> list[Move]:
    sigma = (var.ub - var.lb) * sigma_frac
    new_val = float(np.clip(var.value + rng.normal(0, sigma), var.lb, var.ub))
    return [Move([(var, new_val)], "float_perturb")]


def list_moves(var: ListVar, rng) -> list[Move]:
    moves = []
    n = len(var.elements)
    if n < 2:
        return moves
    i, j = rng.choice(n, 2, replace=False)
    # Swap
    new_elems = var.elements.copy()
    new_elems[i], new_elems[j] = new_elems[j], new_elems[i]
    moves.append(Move([(var, new_elems)], "list_swap"))
    # 2-opt reverse
    lo, hi = min(i, j), max(i, j)
    new_elems = var.elements.copy()
    new_elems[lo:hi + 1] = reversed(new_elems[lo:hi + 1])
    moves.append(Move([(var, new_elems)], "list_2opt"))
    return moves


def set_moves(var: SetVar, rng) -> list[Move]:
    moves = []
    universe = set(range(var.universe_size))
    not_in = list(universe - var.elements)
    in_set = list(var.elements)
    if not_in and len(var.elements) < var.max_size:
        moves.append(Move([(var, var.elements | {rng.choice(not_in)})], "set_add"))
    if in_set and len(var.elements) > var.min_size:
        elem = rng.choice(in_set)
        moves.append(Move([(var, var.elements - {elem})], "set_remove"))
    if in_set and not_in:
        add = rng.choice(not_in)
        rem = rng.choice(in_set)
        moves.append(Move([(var, (var.elements - {rem}) | {add})], "set_swap"))
    return moves


def generate_standard_moves(var, rng) -> list[Move]:
    """Auto-derive moves based on variable type."""
    if isinstance(var, BoolVar):
        return bool_moves(var, rng)
    elif isinstance(var, IntVar):
        return int_moves(var, rng)
    elif isinstance(var, FloatVar):
        return float_moves(var, rng)
    elif isinstance(var, ListVar):
        return list_moves(var, rng)
    elif isinstance(var, SetVar):
        return set_moves(var, rng)
    return []


# ---------------------------------------------------------------------------
# Newton tight moves (constraint-aware)
# ---------------------------------------------------------------------------

def newton_tight_move(var: FloatVar, model: Model,
                      constraint_idx: int) -> list[Move]:
    """1D Newton step to satisfy a violated constraint. LP/NLP-free."""
    constraint = model.constraints[constraint_idx]
    g_x = constraint.value  # > 0 means violated
    dg_dxj = compute_partial(model, constraint, var)

    if abs(dg_dxj) < 1e-12:
        return []

    delta = -g_x / dg_dxj
    new_val = float(np.clip(var.value + delta, var.lb, var.ub))
    if abs(new_val - var.value) < 1e-15:
        return []
    return [Move([(var, new_val)], "newton_tight")]


# ---------------------------------------------------------------------------
# Gradient lift moves (objective-improving)
# ---------------------------------------------------------------------------

def gradient_lift_move(var: FloatVar, model: Model,
                       step_size: float = 0.1) -> list[Move]:
    """Gradient step on objective w.r.t. var. LP/NLP-free."""
    if model.objective is None:
        return []
    df_dxj = compute_partial(model, model.objective, var)

    if abs(df_dxj) < 1e-12:
        return []

    delta = -step_size * df_dxj  # minimize → negative gradient
    new_val = float(np.clip(var.value + delta, var.lb, var.ub))
    if abs(new_val - var.value) < 1e-15:
        return []
    return [Move([(var, new_val)], "gradient_lift")]


# ---------------------------------------------------------------------------
# Move application
# ---------------------------------------------------------------------------

def apply_move(move: Move) -> set:
    """Apply a move, returning the set of changed variables."""
    changed = set()
    for var, new_val in move.changes:
        if isinstance(var, ListVar):
            var.elements = new_val if isinstance(new_val, list) else list(new_val)
        elif isinstance(var, SetVar):
            var.elements = new_val if isinstance(new_val, set) else set(new_val)
        else:
            var.value = new_val
        changed.add(var)
    return changed


def undo_move(move: Move, saved_values: list) -> set:
    """Undo a move given saved original values."""
    changed = set()
    for (var, _), old_val in zip(move.changes, saved_values):
        if isinstance(var, ListVar):
            var.elements = old_val
        elif isinstance(var, SetVar):
            var.elements = old_val
        else:
            var.value = old_val
        changed.add(var)
    return changed


def save_move_values(move: Move) -> list:
    """Save current values of variables in a move (for undo)."""
    saved = []
    for var, _ in move.changes:
        if isinstance(var, ListVar):
            saved.append(var.elements.copy())
        elif isinstance(var, SetVar):
            saved.append(var.elements.copy())
        else:
            saved.append(var.value)
    return saved


# ---------------------------------------------------------------------------
# Move probability tuning
# ---------------------------------------------------------------------------

class MoveProbabilities:
    """Adaptive move selection probabilities based on acceptance rate."""

    def __init__(self, move_types: list[str]):
        self.move_types = move_types
        n = len(move_types)
        self.accept_counts = [0] * n
        self.total_counts = [0] * n
        self.probabilities = [1.0 / n] * n
        self._update_interval = 1000
        self._total_updates = 0

    def select(self, rng) -> str:
        return rng.choice(self.move_types, p=self.probabilities)

    def update(self, move_type: str, accepted: bool):
        if move_type not in self.move_types:
            return
        idx = self.move_types.index(move_type)
        self.total_counts[idx] += 1
        if accepted:
            self.accept_counts[idx] += 1
        self._total_updates += 1

        if self._total_updates % self._update_interval == 0:
            self._rebalance()

    def _rebalance(self):
        n = len(self.move_types)
        floor = 0.05
        rates = [
            a / max(t, 1)
            for a, t in zip(self.accept_counts, self.total_counts)
        ]
        total = sum(rates) + 1e-10
        probs = [r / total for r in rates]
        # Iteratively enforce floor and redistribute
        for _ in range(3):
            deficit = 0.0
            above_floor = 0
            for i in range(n):
                if probs[i] < floor:
                    deficit += floor - probs[i]
                    probs[i] = floor
                else:
                    above_floor += 1
            if deficit > 0 and above_floor > 0:
                # Redistribute deficit from above-floor entries
                for i in range(n):
                    if probs[i] > floor:
                        probs[i] -= deficit / above_floor
                        probs[i] = max(probs[i], floor)
        total = sum(probs)
        self.probabilities = [p / total for p in probs]
