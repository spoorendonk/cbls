"""Inner solver hook interface for structured decomposition."""

from __future__ import annotations

from typing import Any


class InnerSolverHook:
    """Interface for structured decomposition.
    User provides: given fixed combinatorial vars → optimal continuous solution."""

    def __call__(self, fixed_state: dict[str, Any]) -> tuple[dict[str, float], float]:
        """
        Args:
            fixed_state: {var_name: value} for all non-float variables
        Returns:
            (float_values: {var_name: value}, objective: float)
        """
        raise NotImplementedError


class HiGHSInnerSolver(InnerSolverHook):
    """Convenience wrapper: solve LP sub-problem via HiGHS."""

    def __init__(self, build_lp_func):
        """
        build_lp_func: callable(fixed_state) -> highspy model ready to solve.
        Must return (model, var_name_map) where var_name_map maps
        var names to HiGHS variable indices.
        """
        self.build_lp = build_lp_func

    def __call__(self, fixed_state):
        try:
            import highspy
        except ImportError:
            raise ImportError("HiGHS not installed. Install with: pip install highspy")

        lp, var_map, obj_offset = self.build_lp(fixed_state)
        lp.run()

        info = lp.getInfoValue("primal_solution_status")
        if info[1] != 2:  # not feasible
            return {}, float("inf")

        sol = lp.getSolution()
        result = {}
        for name, idx in var_map.items():
            result[name] = sol.col_value[idx]

        obj = lp.getInfoValue("objective_function_value")[1] + obj_offset
        return result, obj


class ScipyInnerSolver(InnerSolverHook):
    """Convenience wrapper: solve small NLP via scipy.optimize.minimize."""

    def __init__(self, objective_func, bounds_func, x0_func=None):
        """
        objective_func: callable(x, fixed_state) -> float
        bounds_func: callable(fixed_state) -> list of (lb, ub)
        x0_func: callable(fixed_state) -> initial guess array
        """
        self.obj = objective_func
        self.bounds = bounds_func
        self.x0_func = x0_func

    def __call__(self, fixed_state):
        try:
            from scipy.optimize import minimize
        except ImportError:
            raise ImportError("scipy not installed. Install with: pip install scipy")

        bounds = self.bounds(fixed_state)
        if self.x0_func:
            x0 = self.x0_func(fixed_state)
        else:
            x0 = [(lb + ub) / 2 for lb, ub in bounds]

        result = minimize(
            self.obj, x0, args=(fixed_state,),
            bounds=bounds, method="L-BFGS-B",
        )

        if not result.success:
            return {}, float("inf")

        var_values = {f"x{i}": result.x[i] for i in range(len(result.x))}
        return var_values, result.fun
