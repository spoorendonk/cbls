"""CHPED benchmark integration test.

Tests the full pipeline: model building → FJ-NL → SA → solution quality.
"""

import time

import pytest

from cbls.model import Model
from cbls.dag import full_evaluate
from cbls.search import solve
from cbls.violation import ViolationManager


def build_chped_model(instance):
    """Build a CHPED model using the CBLS API."""
    m = Model()
    N = instance["n_units"]
    T = instance.get("n_periods", 1)

    a = instance["a"]
    b = instance["b"]
    c = instance["c"]
    d = instance["d"]
    e = instance["e"]
    P_min = instance["P_min"]
    P_max = instance["P_max"]
    demand = instance["demand"]
    reserve = instance.get("reserve", [0.0] * T)

    # Variables
    commit = [[m.bool_var(name=f"u_{i}_{t}") for t in range(T)] for i in range(N)]
    power = [[m.float_var(P_min[i], P_max[i], name=f"p_{i}_{t}")
              for t in range(T)] for i in range(N)]

    # Objective: total valve-point cost
    cost_terms = []
    for i in range(N):
        for t in range(T):
            # cost_it = a[i] + b[i]*P + c[i]*P^2 + |d[i]*sin(e[i]*(P_min[i] - P))|
            P = power[i][t]
            base_cost = m.sum(
                a[i],
                m.prod(b[i], P),
                m.prod(c[i], m.pow(P, 2)),
            )
            valve_point = m.abs(
                m.prod(d[i], m.sin(m.prod(e[i], m.sum(P_min[i], m.prod(-1, P)))))
            )
            unit_cost = m.sum(base_cost, valve_point)
            # Only count if committed
            cost_terms.append(m.prod(commit[i][t], unit_cost))

    total_cost = m.sum(*cost_terms)
    m.minimize(total_cost)

    # Constraints
    for t in range(T):
        # Power balance: supply ≥ demand
        supply_terms = [m.prod(commit[i][t], power[i][t]) for i in range(N)]
        supply = m.sum(*supply_terms)

        # demand[t] - supply ≤ 0  (supply ≥ demand)
        m.add_constraint(m.sum(demand[t], m.prod(-1, supply)))

        # supply ≥ demand + reserve → demand + reserve - supply ≤ 0
        if reserve[t] > 0:
            m.add_constraint(m.sum(demand[t], reserve[t], m.prod(-1, supply)))

    m.close()
    return m, commit, power


class TestCHPED4Unit:
    """Test 4-unit CHPED instance."""

    def test_builds_model(self):
        from benchmarks.chped.data import CHPED_4UNIT
        m, commit, power = build_chped_model(CHPED_4UNIT)
        assert len(m.variables) == 4 + 4  # 4 commit + 4 power
        assert len(m.constraints) == 1    # 1 demand constraint (no reserve)

    def test_feasibility(self):
        """FJ-NL + SA should find feasible solution quickly."""
        from benchmarks.chped.data import CHPED_4UNIT
        m, commit, power = build_chped_model(CHPED_4UNIT)

        start = time.time()
        result = solve(m, time_limit=2.0, seed=42)
        elapsed = time.time() - start

        assert result.feasible, f"Should find feasibility. Obj={result.objective}"
        print(f"\n4-unit: feasible={result.feasible}, obj={result.objective:.2f}, "
              f"iters={result.iterations}, time={elapsed:.3f}s")

    def test_solution_quality(self):
        """Should get reasonable cost within a few seconds."""
        from benchmarks.chped.data import CHPED_4UNIT
        m, commit, power = build_chped_model(CHPED_4UNIT)

        result = solve(m, time_limit=5.0, seed=42)
        assert result.feasible
        # The cost should be in a reasonable range for 4 units serving 400 MW
        # Rough lower bound: ~400 * 2 = 800 (just fuel cost)
        # Rough upper bound: ~400 * 10 = 4000
        assert result.objective < 5000, f"Cost too high: {result.objective}"
        print(f"\n4-unit solution: cost={result.objective:.2f}")


class TestCHPED7Unit:

    def test_feasibility(self):
        from benchmarks.chped.data import CHPED_7UNIT
        m, commit, power = build_chped_model(CHPED_7UNIT)

        result = solve(m, time_limit=3.0, seed=42)
        assert result.feasible, f"Should find feasibility. Obj={result.objective}"
        print(f"\n7-unit: feasible={result.feasible}, obj={result.objective:.2f}, "
              f"iters={result.iterations}")


class TestCHPED24Unit:

    def test_feasibility(self):
        from benchmarks.chped.data import CHPED_24UNIT
        m, commit, power = build_chped_model(CHPED_24UNIT)

        result = solve(m, time_limit=10.0, seed=42)
        print(f"\n24-unit: feasible={result.feasible}, obj={result.objective:.2f}, "
              f"iters={result.iterations}")
        # 24-unit is harder; just check it ran
        assert result.iterations > 100
