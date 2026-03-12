"""Compare off-the-shelf solvers on CHPED benchmark instances.

Runs scipy differential_evolution and PySCIPOpt on 13-unit and 40-unit
instances, prints a comparison table with objective values and timings.

Usage:
    pip install scipy pyscipopt
    cd benchmarks/chped
    python reference_solve.py
"""

import math
import time

import numpy as np
from scipy.optimize import LinearConstraint, differential_evolution

from data import CHPED_13UNIT, CHPED_40UNIT


def cost_function(P, inst):
    """Total valve-point cost for dispatch vector P."""
    total = 0.0
    for i in range(inst["n_units"]):
        p = P[i]
        total += (
            inst["a"][i]
            + inst["b"][i] * p
            + inst["c"][i] * p * p
            + abs(inst["d"][i] * math.sin(inst["e"][i] * (inst["P_min"][i] - p)))
        )
    return total


def solve_scipy(inst):
    """Solve with scipy differential_evolution."""
    n = inst["n_units"]
    bounds = [(inst["P_min"][i], inst["P_max"][i]) for i in range(n)]
    A = np.ones((1, n))
    constraint = LinearConstraint(A, lb=inst["demand"][0], ub=np.inf)

    t0 = time.time()
    result = differential_evolution(
        cost_function,
        bounds,
        args=(inst,),
        constraints=constraint,
        seed=42,
        maxiter=1000,
        tol=1e-12,
        polish=True,
    )
    elapsed = time.time() - t0
    return result.fun, elapsed, result.success


def solve_scip(inst, time_limit):
    """Solve with PySCIPOpt (global optimizer)."""
    from pyscipopt import Model, quicksum, sin  # noqa: E401

    n = inst["n_units"]
    m = Model(inst["name"])
    m.setRealParam("limits/time", time_limit)
    m.hideOutput()

    P = [
        m.addVar(f"P_{i}", lb=inst["P_min"][i], ub=inst["P_max"][i])
        for i in range(n)
    ]

    # Demand constraint
    m.addCons(quicksum(P[i] for i in range(n)) >= inst["demand"][0])

    # SCIP doesn't support nonlinear objectives directly.
    # Model: min z  s.t.  z = sum(a_i + b_i*P_i + c_i*P_i^2 + t_i)
    #         t_i >= d_i*sin(e_i*(Pmin_i - P_i)),  t_i >= -d_i*sin(...)
    T = []
    for i in range(n):
        valve = inst["d"][i] * sin(inst["e"][i] * (inst["P_min"][i] - P[i]))
        t = m.addVar(f"t_{i}", lb=0.0)
        m.addCons(t >= valve)
        m.addCons(t >= -valve)
        T.append(t)

    z = m.addVar("z", lb=-1e20, ub=1e20)
    cost_expr = (
        sum(inst["a"])
        + quicksum(inst["b"][i] * P[i] for i in range(n))
        + quicksum(inst["c"][i] * P[i] * P[i] for i in range(n))
        + quicksum(T)
    )
    m.addCons(z >= cost_expr)
    m.setObjective(z)
    m.optimize()

    status = m.getStatus()
    if status in ("optimal", "bestsollimit", "timelimit") and m.getNSols() > 0:
        obj = m.getObjVal()
        gap = m.getGap()
    else:
        obj = float("inf")
        gap = float("inf")
    solve_time = m.getSolvingTime()
    return obj, solve_time, gap


# CBLS SA results from our solver (for comparison)
CBLS_RESULTS = {
    "13-unit": {"obj": 18727.0, "time": 5.0},
    "40-unit": {"obj": 128391.0, "time": 15.0},
}

TIME_LIMITS = {
    "13-unit": 30.0,
    "40-unit": 120.0,
}


def main():
    print(f"{'Instance':<12} {'Solver':<12} {'Objective':>12} {'Time(s)':>9} {'Gap%':>8} {'vs BKS':>8}")
    print("-" * 65)

    for inst in [CHPED_13UNIT, CHPED_40UNIT]:
        name = inst["name"]
        bks = inst["known_optimum"]
        time_limit = TIME_LIMITS[name]
        rows = []

        # scipy
        try:
            obj, elapsed, success = solve_scipy(inst)
            pct = (obj - bks) / bks * 100
            rows.append((name, "scipy DE", f"{obj:.2f}", f"{elapsed:.2f}", "-", f"+{pct:.2f}%"))
        except Exception as e:
            rows.append((name, "scipy DE", "FAILED", "-", "-", str(e)[:20]))

        # SCIP
        try:
            obj, elapsed, gap = solve_scip(inst, time_limit)
            pct = (obj - bks) / bks * 100
            gap_str = f"{gap * 100:.2f}%" if gap < float("inf") else "-"
            rows.append((name, "SCIP", f"{obj:.2f}", f"{elapsed:.2f}", gap_str, f"+{pct:.2f}%"))
        except ImportError:
            rows.append((name, "SCIP", "not installed", "-", "-", "-"))
        except Exception as e:
            rows.append((name, "SCIP", "FAILED", "-", "-", str(e)[:20]))

        # CBLS SA
        cbls = CBLS_RESULTS[name]
        pct = (cbls["obj"] - bks) / bks * 100
        rows.append((name, "CBLS SA", f"{cbls['obj']:.2f}", f"{cbls['time']:.2f}", "-", f"+{pct:.2f}%"))

        for row in rows:
            print(f"{row[0]:<12} {row[1]:<12} {row[2]:>12} {row[3]:>9} {row[4]:>8} {row[5]:>8}")
        print()


if __name__ == "__main__":
    main()
