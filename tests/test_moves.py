"""Tests for move generation: standard, Newton tight, gradient lift."""

import math

import numpy as np
import pytest

from cbls.model import Model
from cbls.dag import BoolVar, IntVar, FloatVar, ListVar, SetVar, full_evaluate, delta_evaluate
from cbls.moves import (
    generate_standard_moves, newton_tight_move, gradient_lift_move,
    apply_move, save_move_values, undo_move, MoveProbabilities,
)


class TestStandardMoves:
    """Test auto-derived moves for each variable type."""

    def test_bool_flip(self):
        m = Model()
        x = m.bool_var()
        m.minimize(m.sum(x))
        m.close()
        rng = np.random.default_rng(42)

        x.value = 0.0
        moves = generate_standard_moves(x, rng)
        assert len(moves) == 1
        assert moves[0].changes[0][1] == 1.0

        x.value = 1.0
        moves = generate_standard_moves(x, rng)
        assert moves[0].changes[0][1] == 0.0

    def test_int_moves(self):
        m = Model()
        x = m.int_var(0, 10)
        m.minimize(m.sum(x))
        m.close()
        rng = np.random.default_rng(42)

        x.value = 5.0
        moves = generate_standard_moves(x, rng)
        # Should have dec, inc, random
        assert len(moves) == 3
        vals = [mv.changes[0][1] for mv in moves]
        assert 4.0 in vals  # decrement
        assert 6.0 in vals  # increment

    def test_int_at_bounds(self):
        m = Model()
        x = m.int_var(0, 10)
        m.minimize(m.sum(x))
        m.close()
        rng = np.random.default_rng(42)

        x.value = 0.0
        moves = generate_standard_moves(x, rng)
        vals = [mv.changes[0][1] for mv in moves]
        assert -1.0 not in vals  # no decrement below lb

    def test_float_perturb(self):
        m = Model()
        x = m.float_var(0, 10)
        m.minimize(m.sum(x))
        m.close()
        rng = np.random.default_rng(42)

        x.value = 5.0
        moves = generate_standard_moves(x, rng)
        assert len(moves) == 1
        val = moves[0].changes[0][1]
        assert 0.0 <= val <= 10.0  # within bounds

    def test_list_moves(self):
        m = Model()
        lv = m.list_var(5)
        m.minimize(m.lambda_sum(lv, lambda e: e))
        m.close()
        rng = np.random.default_rng(42)

        lv.elements = [0, 1, 2, 3, 4]
        moves = generate_standard_moves(lv, rng)
        assert len(moves) == 2  # swap + 2-opt
        for mv in moves:
            new_elems = mv.changes[0][1]
            assert sorted(new_elems) == [0, 1, 2, 3, 4]  # permutation preserved

    def test_set_moves(self):
        m = Model()
        sv = m.set_var(5, min_size=1, max_size=4)
        m.minimize(m.count(sv))
        m.close()
        rng = np.random.default_rng(42)

        sv.elements = {0, 1, 2}
        moves = generate_standard_moves(sv, rng)
        assert len(moves) >= 2  # add + remove + swap


class TestNewtonTightMoves:
    """Test Newton tight moves satisfy violated constraints."""

    def test_linear_constraint(self):
        """On violated linear constraint g(x) = x - 5 ≤ 0, Newton should set x=5."""
        m = Model()
        x = m.float_var(0, 10)
        # constraint: x - 5 ≤ 0
        constraint = m.sum(x, -5)
        m.add_constraint(constraint)
        m.minimize(m.sum(x))
        m.close()

        x.value = 8.0  # violated: 8-5 = 3 > 0
        full_evaluate(m)
        assert constraint.value == 3.0

        moves = newton_tight_move(x, m, 0)
        assert len(moves) == 1
        new_val = moves[0].changes[0][1]
        assert abs(new_val - 5.0) < 1e-10  # should fix to x=5

    def test_quadratic_constraint(self):
        """On x^2 - 4 ≤ 0 with x=3, Newton step should move toward feasibility."""
        m = Model()
        x = m.float_var(0, 10)
        x_sq = m.pow(x, 2)
        constraint = m.sum(x_sq, -4)  # x^2 - 4 ≤ 0
        m.add_constraint(constraint)
        m.minimize(m.sum(x))
        m.close()

        x.value = 3.0  # violated: 9-4 = 5 > 0
        full_evaluate(m)

        moves = newton_tight_move(x, m, 0)
        assert len(moves) == 1
        new_val = moves[0].changes[0][1]
        # Newton: x - g(x)/g'(x) = 3 - 5/6 ≈ 2.167
        assert new_val < 3.0  # moved toward feasibility


class TestGradientLiftMoves:
    """Test gradient lift moves improve objective."""

    def test_minimize_x_squared(self):
        """On f(x)=x^2, gradient at x=3 is 6, so move in -gradient direction."""
        m = Model()
        x = m.float_var(-10, 10)
        m.minimize(m.pow(x, 2))
        m.close()

        x.value = 3.0
        full_evaluate(m)

        moves = gradient_lift_move(x, m, step_size=0.1)
        assert len(moves) == 1
        new_val = moves[0].changes[0][1]
        assert new_val < 3.0  # moved toward minimum at 0


class TestMoveApplication:
    """Test apply/undo moves."""

    def test_apply_undo(self):
        m = Model()
        x = m.float_var(0, 10)
        m.minimize(m.sum(x))
        m.close()

        x.value = 5.0
        from cbls.moves import Move
        move = Move([(x, 8.0)], "test")
        saved = save_move_values(move)
        assert saved == [5.0]

        apply_move(move)
        assert x.value == 8.0

        undo_move(move, saved)
        assert x.value == 5.0


class TestMoveProbabilities:
    """Test adaptive move probability tuning."""

    def test_initial_uniform(self):
        mp = MoveProbabilities(["a", "b", "c"])
        assert len(mp.probabilities) == 3
        for p in mp.probabilities:
            assert abs(p - 1 / 3) < 1e-10

    def test_rebalance_favors_accepted(self):
        mp = MoveProbabilities(["a", "b"])
        mp._update_interval = 10

        # Type "a" always accepted, "b" never
        for _ in range(10):
            mp.update("a", True)
        for _ in range(10):
            mp.update("b", False)

        # After rebalance, "a" should have higher probability
        assert mp.probabilities[0] > mp.probabilities[1]

    def test_floor_probability(self):
        mp = MoveProbabilities(["a", "b", "c"])
        mp._update_interval = 10

        for _ in range(10):
            mp.update("a", True)
        for _ in range(10):
            mp.update("b", False)
        for _ in range(10):
            mp.update("c", False)

        # All should have at least 5% (floor)
        for p in mp.probabilities:
            assert p >= 0.049
