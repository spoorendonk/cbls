"""The MIPfeas primal gap and primal integral, shared by every benchmark that
scores an anytime trace against a reference value (mipfeas, MINLPLib, and
UC-CHPED's equal-time comparison, #442)."""

from __future__ import annotations

import math
import statistics

#: Below this magnitude an objective counts as zero, so that 0 vs 1e-12 does not
#: score as a 100% gap.
ZERO_TOLERANCE = 1e-6

#: Penalty for a run that never found a feasible solution.
NO_SOLUTION_GAP = 2.0

#: Penalty when incumbent and reference have opposite signs; the relative formula
#: is not meaningful across a sign change.
SIGN_FLIP_GAP = 1.0

#: Shift for the geometric mean, so that a PI of exactly 0 does not collapse it.
GEOMETRIC_MEAN_SHIFT = 0.001


def primal_gap(incumbent: float | None, reference: float) -> float:
    """MIPfeas primal gap of one incumbent against the reference value."""
    if incumbent is None:
        return NO_SOLUTION_GAP
    if abs(incumbent) < ZERO_TOLERANCE and abs(reference) < ZERO_TOLERANCE:
        return 0.0
    if incumbent * reference < 0:
        return SIGN_FLIP_GAP
    denominator = max(abs(incumbent), abs(reference))
    if denominator == 0.0:
        return 0.0
    return abs(incumbent - reference) / denominator


def primal_integral(trace: list[tuple[float, float]], reference: float, budget: float) -> float:
    """Time-average the primal gap over [0, budget].

    `trace` is the incumbent step function as (seconds, objective) pairs. Before the
    first entry the gap is NO_SOLUTION_GAP; each entry holds until the next one, and
    the last holds to the budget.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")

    area = 0.0
    previous_time = 0.0
    current_gap = NO_SOLUTION_GAP
    # Sorted by time, then objective descending: CP-SAT logs to 0.01s and routinely
    # reports several improvements inside one tick. Plain tuple ordering would apply
    # the *worst* of a tied group last and hold it until the next distinct time — a
    # small, one-directional penalty against whichever engine has the denser trace,
    # which is systematically CP-SAT.
    for raw_time, objective in sorted(trace, key=lambda entry: (entry[0], -entry[1])):
        time = min(max(raw_time, 0.0), budget)
        if time > previous_time:
            area += current_gap * (time - previous_time)
            previous_time = time
        current_gap = primal_gap(objective, reference)
    area += current_gap * (budget - previous_time)
    return area / budget


def shifted_geometric_mean(values: list[float], shift: float = GEOMETRIC_MEAN_SHIFT) -> float:
    if not values:
        return math.nan
    return math.exp(statistics.fmean(math.log(v + shift) for v in values)) - shift
