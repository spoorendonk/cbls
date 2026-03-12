"""Tests for C++ moves/LNS/pool via Python bindings."""

import pytest
import _cbls_core as cbls


def vid(handle):
    return -(handle + 1)


class TestMoves:
    def test_generate_float_moves(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        m.minimize(m.sum([x]))
        m.close()
        m.var_mut(vid(x)).value = 5.0
        cbls.full_evaluate(m)
        rng = cbls.RNG(42)
        moves = cbls.generate_standard_moves(m.var(vid(x)), rng)
        assert len(moves) > 0
        assert moves[0].move_type == "float_perturb"

    def test_apply_undo_move(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        m.minimize(m.sum([x]))
        m.close()
        m.var_mut(vid(x)).value = 5.0
        cbls.full_evaluate(m)
        rng = cbls.RNG(42)
        moves = cbls.generate_standard_moves(m.var(vid(x)), rng)
        move = moves[0]
        saved = cbls.save_move_values(m, move)
        cbls.apply_move(m, move)
        assert m.var(vid(x)).value != 5.0
        cbls.undo_move(m, move, saved)
        assert m.var(vid(x)).value == 5.0


class TestCopyRestore:
    def test_copy_restore_state(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        m.minimize(m.sum([x, y]))
        m.close()
        m.var_mut(vid(x)).value = 3.0
        m.var_mut(vid(y)).value = 7.0
        state = m.copy_state()
        m.var_mut(vid(x)).value = 1.0
        m.var_mut(vid(y)).value = 2.0
        m.restore_state(state)
        assert m.var(vid(x)).value == 3.0
        assert m.var(vid(y)).value == 7.0


class TestErrorPaths:
    def test_var_out_of_range(self):
        m = cbls.Model()
        m.float_var(0, 1)
        with pytest.raises(IndexError):
            m.var(999)

    def test_add_constraint_rejects_var_handle(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        with pytest.raises(ValueError):
            m.add_constraint(x)


class TestLNS:
    def test_lns_destroy_repair(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        neg1 = m.constant(-1.0)
        five = m.constant(5.0)
        m.add_constraint(m.sum([five, m.prod(neg1, x), m.prod(neg1, y)]))
        m.minimize(m.sum([x, y]))
        m.close()
        m.var_mut(vid(x)).value = 8.0
        m.var_mut(vid(y)).value = 8.0
        cbls.full_evaluate(m)
        vm = cbls.ViolationManager(m)
        lns = cbls.LNS(0.5)
        rng = cbls.RNG(42)
        lns.destroy_repair(m, vm, rng)
        # Just check it doesn't crash

    def test_lns_destroy_repair_cycle(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        neg1 = m.constant(-1.0)
        five = m.constant(5.0)
        m.add_constraint(m.sum([five, m.prod(neg1, x), m.prod(neg1, y)]))
        m.minimize(m.sum([x, y]))
        m.close()
        m.var_mut(vid(x)).value = 8.0
        m.var_mut(vid(y)).value = 8.0
        cbls.full_evaluate(m)
        vm = cbls.ViolationManager(m)
        lns = cbls.LNS(0.5)
        rng = cbls.RNG(42)
        lns.destroy_repair_cycle(m, vm, rng, 3)
        # Just check it doesn't crash


class TestSolutionPool:
    def test_pool_ordering(self):
        pool = cbls.SolutionPool(3)
        s1 = cbls.Solution()
        s1.objective = 10.0
        s1.feasible = True
        pool.submit(s1)
        s2 = cbls.Solution()
        s2.objective = 3.0
        s2.feasible = True
        pool.submit(s2)
        best = pool.best()
        assert best is not None
        assert best.objective == 3.0
