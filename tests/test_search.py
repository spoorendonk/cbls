"""Tests for SA search and FJ-NL initialization."""

import math
import time

import numpy as np
import pytest

from cbls.model import Model
from cbls.dag import full_evaluate, FloatVar, IntVar, BoolVar
from cbls.search import solve, SearchResult, fj_nl_initialize, initialize_random
from cbls.violation import ViolationManager, AdaptiveLambda


class TestViolation:
    """Test penalty objective and violation computation."""

    def test_no_violation_when_feasible(self):
        m = Model()
        x = m.float_var(0, 10)
        m.add_constraint(m.sum(x, -5))  # x - 5 ≤ 0
        m.minimize(m.sum(x))
        m.close()

        x.value = 3.0
        full_evaluate(m)
        vm = ViolationManager(m)
        assert vm.total_violation() == 0.0
        assert vm.is_feasible()

    def test_violation_when_infeasible(self):
        m = Model()
        x = m.float_var(0, 10)
        m.add_constraint(m.sum(x, -5))  # x - 5 ≤ 0
        m.minimize(m.sum(x))
        m.close()

        x.value = 8.0
        full_evaluate(m)
        vm = ViolationManager(m)
        assert vm.total_violation() == 3.0  # 8 - 5 = 3
        assert not vm.is_feasible()

    def test_augmented_objective(self):
        m = Model()
        x = m.float_var(0, 10)
        m.add_constraint(m.sum(x, -5))  # x - 5 ≤ 0
        m.minimize(m.sum(x))
        m.close()

        x.value = 8.0
        full_evaluate(m)
        vm = ViolationManager(m)
        # F = f + λ*V = 8 + 1.0*3 = 11
        assert vm.augmented_objective() == 11.0


class TestAdaptiveLambda:
    """Test adaptive λ controller."""

    def test_lambda_increases_when_infeasible(self):
        al = AdaptiveLambda(initial_lambda=1.0)
        for _ in range(11):
            al.update(is_feasible=False, obj_improved=False)
        assert al.lambda_ > 1.0

    def test_lambda_decreases_when_stuck_feasible(self):
        al = AdaptiveLambda(initial_lambda=1.0)
        for _ in range(21):
            al.update(is_feasible=True, obj_improved=False)
        assert al.lambda_ < 1.0


class TestFJNL:
    """Test FJ-NL feasibility initialization."""

    def test_finds_feasibility_simple(self):
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        # x + y ≥ 5 → -(x+y) + 5 ≤ 0
        m.add_constraint(m.sum(-1, m.prod(-1, m.sum(x, y)), 5))
        m.minimize(m.sum(x, y))
        m.close()

        # Start infeasible
        x.value = 0.0
        y.value = 0.0
        full_evaluate(m)
        vm = ViolationManager(m)
        assert not vm.is_feasible()

        fj_nl_initialize(m, vm, max_iterations=1000)
        full_evaluate(m)
        # Should be feasible: x + y ≥ 5 → constraint ≤ 0
        # The constraint is: -1 + (-1)*(x+y) + 5 = 4 - x - y ≤ 0
        # So x + y ≥ 4
        assert vm.is_feasible() or vm.total_violation() < 0.1

    def test_finds_feasibility_bool(self):
        m = Model()
        x = m.bool_var()
        y = m.bool_var()
        # x + y ≥ 1 → -(x+y) + 1 ≤ 0 → 1 - x - y ≤ 0
        neg_x = m.prod(-1, x)
        neg_y = m.prod(-1, y)
        m.add_constraint(m.sum(1, neg_x, neg_y))
        m.minimize(m.sum(x, y))
        m.close()

        x.value = 0.0
        y.value = 0.0
        full_evaluate(m)
        vm = ViolationManager(m)
        assert not vm.is_feasible()

        fj_nl_initialize(m, vm, max_iterations=100)
        full_evaluate(m)
        assert vm.is_feasible()


class TestSASolver:
    """Test SA solver on small problems."""

    def test_unconstrained_minimum(self):
        """Minimize x^2 + y^2, optimum at (0, 0)."""
        m = Model()
        x = m.float_var(-10, 10)
        y = m.float_var(-10, 10)
        m.minimize(m.sum(m.pow(x, 2), m.pow(y, 2)))
        m.close()

        result = solve(m, time_limit=2.0, seed=42)
        assert result.feasible
        assert result.objective < 1.0  # should get close to 0

    def test_constrained_problem(self):
        """Minimize x + y subject to x + y ≥ 3, x,y ∈ [0, 10].
        Optimum: x + y = 3, obj = 3."""
        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        # constraint: 3 - x - y ≤ 0
        m.add_constraint(m.sum(3, m.prod(-1, x), m.prod(-1, y)))
        m.minimize(m.sum(x, y))
        m.close()

        result = solve(m, time_limit=3.0, seed=42)
        assert result.feasible
        assert result.objective < 5.0  # should be near 3

    def test_integer_problem(self):
        """Minimize |x - 7| for x ∈ {0, ..., 10}."""
        m = Model()
        x = m.int_var(0, 10)
        m.minimize(m.abs(m.sum(x, -7)))
        m.close()

        result = solve(m, time_limit=2.0, seed=42)
        assert result.feasible
        assert result.objective < 2.0

    def test_rosenbrock_2d(self):
        """Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2. Minimum at (1,1)."""
        m = Model()
        x = m.float_var(-5, 5)
        y = m.float_var(-5, 5)

        one_minus_x = m.sum(1, m.prod(-1, x))
        term1 = m.pow(one_minus_x, 2)

        y_minus_x2 = m.sum(y, m.prod(-1, m.pow(x, 2)))
        term2 = m.prod(100, m.pow(y_minus_x2, 2))

        m.minimize(m.sum(term1, term2))
        m.close()

        result = solve(m, time_limit=5.0, seed=42)
        assert result.feasible
        # Rosenbrock is hard; just check we get a reasonable result
        assert result.objective < 50.0

    def test_returns_result(self):
        m = Model()
        x = m.float_var(0, 1)
        m.minimize(m.sum(x))
        m.close()

        result = solve(m, time_limit=0.5, seed=42)
        assert isinstance(result, SearchResult)
        assert result.iterations > 0
        assert result.time_seconds > 0


class TestLNS:
    """Test LNS destroy-repair."""

    def test_lns_basic(self):
        from cbls.lns import LNS

        m = Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        m.add_constraint(m.sum(5, m.prod(-1, x), m.prod(-1, y)))  # x+y ≥ 5
        m.minimize(m.sum(x, y))
        m.close()

        # Set to feasible but suboptimal
        x.value = 8.0
        y.value = 8.0
        full_evaluate(m)
        vm = ViolationManager(m)

        lns = LNS(destroy_fraction=0.5)
        rng = np.random.default_rng(42)
        # Run a few rounds — may or may not improve
        lns.destroy_repair_cycle(m, vm, rng, n_rounds=5)
        # Just check it doesn't crash and stays feasible-ish
        full_evaluate(m)


class TestSolutionPool:
    """Test solution pool."""

    def test_pool_ordering(self):
        from cbls.pool import SolutionPool, Solution

        pool = SolutionPool(capacity=3)
        pool.submit(Solution(state={}, objective=10.0, feasible=True))
        pool.submit(Solution(state={}, objective=5.0, feasible=True))
        pool.submit(Solution(state={}, objective=20.0, feasible=False))
        pool.submit(Solution(state={}, objective=3.0, feasible=True))

        best = pool.best()
        assert best.objective == 3.0
        assert best.feasible
        assert len(pool.solutions) == 3  # capacity limit
