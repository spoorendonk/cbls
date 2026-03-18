"""GLSP-RP instance generator (Goerler, Lalla-Ruiz & Voss 2020, Table 9).

Generates instances for the General Lot-Sizing and Scheduling Problem with
Rework and Lifetime constraints. Three classes (A, B, C) with 50 instances
each, plus two scaled classes (D, E) for stress-testing.
"""

import random
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class GLSPInstance:
    name: str
    cls: str  # class label (A, B, C, D, E)
    n_products: int  # J
    n_macro: int  # T
    n_micro_per_macro: int  # |M_t| (same for all macro-periods)
    demand: list  # [j][t] demand for product j in macro-period t
    setup_cost: list  # [i][j] changeover cost from product i to j
    setup_time: list  # [i][j] changeover time from product i to j
    process_time: list  # [j] processing time per unit
    rework_time: list  # [j] rework processing time per unit
    holding_cost: list  # [j] serviceable inventory holding cost
    rework_holding_cost: list  # [j] rework inventory holding cost per micro-period
    min_lot: list  # [j] minimum lot size (kappa_j)
    disposal_cost: list  # [j] disposal cost per unit (lambda_j)
    lifetime: list  # [j] rework lifetime in micro-periods (Omega_j)
    defect_rate: list  # [j][t] defect fraction Theta_{j,t}
    capacity: list  # [t] total capacity per macro-period (in time units)
    seed: int = 0


def _uniform_or_zero(rng, lo, hi, p_zero=0.3):
    """Return 0 with probability p_zero, else uniform [lo, hi]."""
    if rng.random() < p_zero:
        return 0.0
    return rng.uniform(lo, hi)


def generate_instance(cls: str, idx: int, seed: int) -> GLSPInstance:
    """Generate a single GLSP-RP instance following Table 9."""
    rng = random.Random(seed)

    if cls == "A":
        J, T, M = 5, 4, 7
    elif cls == "B":
        J, T, M = 4, 3, 6
    elif cls == "C":
        J, T, M = 6, 2, 8
    elif cls == "D":
        J, T, M = 10, 6, 12
    elif cls == "E":
        J, T, M = 20, 10, 15
    else:
        raise ValueError(f"Unknown class: {cls}")

    # Demand d_{j,t}: U[0; 40,120] for A,B,D,E; U[0; 600,1000] for C
    if cls == "C":
        demand = [[_uniform_or_zero(rng, 600, 1000) for _ in range(T)] for _ in range(J)]
    else:
        demand = [[_uniform_or_zero(rng, 40, 120) for _ in range(T)] for _ in range(J)]

    # Setup cost f_{i,j}: U[100, 400], diagonal = 0
    setup_cost = [[0.0] * J for _ in range(J)]
    for i in range(J):
        for j in range(J):
            if i != j:
                setup_cost[i][j] = rng.uniform(100, 400)

    # Setup time st_{i,j}
    if cls == "C":
        setup_time = [[0.0] * J for _ in range(J)]
        for i in range(J):
            for j in range(J):
                if i != j:
                    setup_time[i][j] = rng.uniform(10, 40)
    else:
        # f_{i,j} / 10
        setup_time = [[setup_cost[i][j] / 10.0 for j in range(J)] for i in range(J)]

    # Process time tp_j = 1
    process_time = [1.0] * J

    # Rework time tp^R_j
    if cls == "C":
        rework_time = [0.75] * J
    else:
        rework_time = [0.5] * J

    # Holding cost h_j
    if cls == "C":
        holding_cost = [rng.uniform(1, 5) for _ in range(J)]
    else:
        holding_cost = [rng.uniform(10, 20) for _ in range(J)]

    # Rework holding cost h^R_j
    if cls == "C":
        rework_holding_cost = [(h / M) * 0.75 for h in holding_cost]
    else:
        rework_holding_cost = [h / M for h in holding_cost]

    # Min lot kappa_j
    if cls == "C":
        min_lot = [50.0] * J
    else:
        min_lot = [10.0] * J

    # Disposal cost lambda_j
    disposal_cost = [1000.0] * J

    # Lifetime Omega_j (micro-periods)
    if cls == "C":
        lifetime = [2] * J
    else:
        lifetime = [3] * J

    # Defect rate Theta_{j,t}: U[0; 0.005, 0.03]
    defect_rate = [[_uniform_or_zero(rng, 0.005, 0.03) for _ in range(T)] for _ in range(J)]

    # Capacity b_t (per macro-period, not total)
    # Paper Table 9: b_t = factor × Σ_j Σ_t d_{j,t} for EACH macro-period
    total_demand = sum(sum(demand[j][t] for j in range(J)) for t in range(T))
    if cls == "C":
        cap_per_period = 0.6 * total_demand
    else:
        cap_per_period = 2.0 * total_demand
    capacity = [cap_per_period] * T

    name = f"glsp_{cls.lower()}_{idx:03d}"

    return GLSPInstance(
        name=name, cls=cls, n_products=J, n_macro=T,
        n_micro_per_macro=M, demand=demand, setup_cost=setup_cost,
        setup_time=setup_time, process_time=process_time,
        rework_time=rework_time, holding_cost=holding_cost,
        rework_holding_cost=rework_holding_cost, min_lot=min_lot,
        disposal_cost=disposal_cost, lifetime=lifetime,
        defect_rate=defect_rate, capacity=capacity, seed=seed,
    )


def generate_all(classes=("A", "B", "C"), n_per_class=50, base_seed=42):
    """Generate all instances for the given classes."""
    instances = []
    for cls in classes:
        for i in range(n_per_class):
            seed = base_seed + hash((cls, i)) % (2**31)
            inst = generate_instance(cls, i, seed)
            instances.append(inst)
    return instances
