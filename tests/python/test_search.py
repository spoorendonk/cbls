"""Tests for C++ search via Python bindings."""

import pytest
import _cbls_core as cbls


def vid(handle):
    return -(handle + 1)


class TestSolver:
    def test_unconstrained(self):
        m = cbls.Model()
        x = m.float_var(-10, 10)
        y = m.float_var(-10, 10)
        two = m.constant(2)
        m.minimize(m.sum([m.pow_expr(x, two), m.pow_expr(y, two)]))
        m.close()
        result = cbls.solve(m, 2.0, 42)
        assert result.feasible
        assert result.objective < 1.0

    def test_constrained(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        neg1 = m.constant(-1.0)
        three = m.constant(3.0)
        m.add_constraint(m.sum([three, m.prod(neg1, x), m.prod(neg1, y)]))
        m.minimize(m.sum([x, y]))
        m.close()
        result = cbls.solve(m, 3.0, 42)
        assert result.feasible
        assert result.objective < 5.0

    def test_returns_result(self):
        m = cbls.Model()
        x = m.float_var(0, 1)
        m.minimize(m.sum([x]))
        m.close()
        result = cbls.solve(m, 0.5, 42)
        assert result.iterations > 0
        assert result.time_seconds > 0


class TestTermination:
    """SearchResult.termination — which budget ended the run (#104)."""

    @staticmethod
    def _quadratic() -> "cbls.Model":
        m = cbls.Model()
        x = m.float_var(-5, 5)
        y = m.float_var(-5, 5)
        two = m.constant(2)
        m.minimize(m.sum([m.pow_expr(x, two), m.pow_expr(y, two)]))
        m.close()
        return m

    def test_iteration_limit_wins_over_a_live_clock(self) -> None:
        config = cbls.SearchConfig()
        config.max_iterations = 1000
        # A live but unreachable clock: the iteration budget is what stops this.
        result = cbls.solve(self._quadratic(), 30.0, 42, config=config)
        assert result.termination == cbls.TerminationReason.IterationLimit
        assert result.iterations >= 1000

    def test_no_budget_at_all_returns_immediately(self) -> None:
        # Neither budget set: solve() must return rather than spin forever, and
        # must say so instead of claiming a limit it was never given.
        result = cbls.solve(self._quadratic(), 0.0, 42, config=cbls.SearchConfig())
        assert result.termination == cbls.TerminationReason.NoBudget
        assert result.iterations == 0


class TestFjNlInitialize:
    def test_returns_iterations_spent(self) -> None:
        """The count is what makes 'did the clock stop this?' answerable (#104)."""
        m = cbls.Model()
        variables = [m.int_var(0, 10) for _ in range(50)]
        # 50 variables capped at 10 sum to at most 500, so this is unreachable and
        # the pass can only ever be stopped by a budget.
        m.add_constraint(m.abs_expr(m.sum([*variables, m.constant(-2500.0)])))
        m.close()

        vm = cbls.ViolationManager(m)
        rng = cbls.RNG(42)
        cbls.initialize_random(m, rng)
        cbls.full_evaluate(m)

        spent = cbls.fj_nl_initialize(m, vm, 500, rng, 0.0)
        assert spent == 500


class TestViolation:
    def test_feasible(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        neg5 = m.constant(-5.0)
        m.add_constraint(m.sum([x, neg5]))
        m.minimize(m.sum([x]))
        m.close()
        m.var_mut(vid(x)).value = 3.0
        cbls.full_evaluate(m)
        vm = cbls.ViolationManager(m)
        assert vm.total_violation() == 0.0
        assert vm.is_feasible()

    def test_infeasible(self):
        m = cbls.Model()
        x = m.float_var(0, 10)
        neg5 = m.constant(-5.0)
        m.add_constraint(m.sum([x, neg5]))
        m.minimize(m.sum([x]))
        m.close()
        m.var_mut(vid(x)).value = 8.0
        cbls.full_evaluate(m)
        vm = cbls.ViolationManager(m)
        assert vm.total_violation() == 3.0
        assert not vm.is_feasible()
