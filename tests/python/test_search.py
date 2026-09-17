"""Tests for C++ search via Python bindings."""

import _cbls_core as cbls
import pytest


def vid(handle: int) -> int:
    return -(handle + 1)


class TestSolver:
    def test_solve_returns_a_populated_result_on_a_constrained_model(self) -> None:
        """One solve, reading every SearchResult field but `termination`.

        `termination` has its own cases in `TestTermination` below, which is
        where the enum conversion is asserted.

        Whether the search is any good is `tests/test_search.cpp`'s question, on
        far more shapes than this; here the question is whether `solve` is
        callable from Python and hands back a result whose fields are populated
        rather than default-constructed.
        A constrained model is used so `feasible` means something.
        """
        m = cbls.Model()
        x = m.float_var(0, 10)
        y = m.float_var(0, 10)
        neg1 = m.constant(-1.0)
        three = m.constant(3.0)
        m.add_constraint(m.sum([three, m.prod(neg1, x), m.prod(neg1, y)]))  # x + y >= 3
        m.minimize(m.sum([x, y]))
        m.close()

        result = cbls.solve(m, 3.0, 42)
        assert result.feasible
        assert result.objective < 5.0
        assert result.iterations > 0
        assert result.time_seconds > 0

    def test_solve_propagates_an_exception_raised_by_the_callback(self) -> None:
        """A raising on_progress ends a single-threaded solve with the original exception.

        The contract `solve`'s docstring states, and the one the portfolio does
        NOT share (tests/python/test_pool.py covers that side). In-process on
        purpose: no thread is involved, so nothing here can deadlock.
        """
        m = cbls.Model()
        x = m.float_var(0, 10)
        m.add_constraint(m.sum([m.constant(3.0), m.prod(m.constant(-1.0), x)]))  # x >= 3
        m.minimize(m.sum([x]))
        m.close()

        class ProgressError(KeyError):
            pass

        class Raiser(cbls.SolveCallback):  # type: ignore[misc]
            def on_progress(self, progress: "cbls.SolveProgress") -> None:
                raise ProgressError("logging failed")

        with pytest.raises(ProgressError, match="logging failed"):
            cbls.solve(m, 5.0, 42, callback=Raiser())


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
    def test_violation_manager_reports_both_verdicts_on_one_model(self) -> None:
        # x - 5 <= 0, read at a satisfying point and at a violating one. Both in
        # one test because the two used to be separate models differing only in
        # the value assigned, which is one assertion's worth of information.
        #
        # invalidate_cache() below is NOT load-bearing: total_violation() diffs
        # every constraint against its cache on each call and self-corrects
        # (src/violation.cpp), so both reads are right without it -- verified by
        # removing them. It stays because nothing else in the Python suite
        # touches that binding, so this is its only reachability witness.
        m = cbls.Model()
        x = m.float_var(0, 10)
        m.add_constraint(m.sum([x, m.constant(-5.0)]))
        m.minimize(m.sum([x]))
        m.close()
        vm = cbls.ViolationManager(m)

        m.var_mut(vid(x)).value = 3.0
        cbls.full_evaluate(m)
        vm.invalidate_cache()
        assert vm.total_violation() == 0.0
        assert vm.is_feasible()

        m.var_mut(vid(x)).value = 8.0
        cbls.full_evaluate(m)
        vm.invalidate_cache()
        assert vm.total_violation() == 3.0
        assert not vm.is_feasible()
