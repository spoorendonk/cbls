"""Tests for C++ moves/LNS/pool via Python bindings."""

import _cbls_core as cbls
import pytest


def vid(handle: int) -> int:
    return -(handle + 1)


class TestMoves:
    def test_generate_float_moves(self) -> None:
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

    def test_apply_undo_move(self) -> None:
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

    def test_a_structured_change_carries_positional_edits(self) -> None:
        """A structured candidate is an edit, not the whole element vector (#164).

        `elements_after` is what a caller uses to see the absolute vector the
        change used to carry; the engine applies the edit in place instead.
        """
        m = cbls.Model()
        lv = m.list_var(5, "perm")
        m.minimize(m.count(lv))
        m.close()
        m.var_mut(vid(lv)).elements = [0, 1, 2, 3, 4]
        cbls.full_evaluate(m)

        rng = cbls.RNG(42)
        moves = cbls.generate_standard_moves(m.var(vid(lv)), rng)
        by_type = {mv.move_type: mv for mv in moves}
        assert "list_swap" in by_type

        change = by_type["list_swap"].changes[0]
        assert change.var_id == vid(lv)
        assert change.edits[0].kind == cbls.EditKind.Swap
        assert change.edits[1].kind == cbls.EditKind.None_
        assert change.replacement == []

        before = list(m.var(vid(lv)).elements)
        after = change.elements_after(before)
        assert sorted(after) == before
        assert after != before
        assert not change.is_noop(before)

        cbls.apply_move(m, by_type["list_swap"])
        assert list(m.var(vid(lv)).elements) == after


class TestCopyRestore:
    def test_copy_restore_state(self) -> None:
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
    def test_var_out_of_range(self) -> None:
        m = cbls.Model()
        m.float_var(0, 1)
        with pytest.raises(IndexError):
            m.var(999)

    def test_add_constraint_rejects_var_handle(self) -> None:
        m = cbls.Model()
        x = m.float_var(0, 10)
        with pytest.raises(ValueError):
            m.add_constraint(x)


class TestLNS:
    def test_lns_destroy_repair(self) -> None:
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

    def test_lns_destroy_repair_cycle(self) -> None:
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
    def test_pool_ordering(self) -> None:
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
