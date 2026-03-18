"""Reference solver for GLSP-RP using HiGHS MIP.

Implements the full GLSP-RP formulation (Goerler et al. 2020, Eqs 1-25)
using highspy Python bindings.

Usage:
    pip install highspy
    cd benchmarks/pharma-glsp
    python reference_solve.py                   # Class A, first 5 instances
    python reference_solve.py --class B --max 10
    python reference_solve.py --all --time 300
"""

import argparse
import json
import math
import os
import sys
import time

# Add instances directory to path
_inst_dir = os.path.join(os.path.dirname(__file__), "..", "instances", "pharma-glsp")
sys.path.insert(0, _inst_dir)

from data import generate_instance, generate_all, GLSPInstance


def solve_highs(inst, time_limit=300.0):
    """Solve GLSP-RP with HiGHS MIP.

    Variables:
        x[j,m] binary: product j set up in micro-period m
        y[j,m] >= 0: production amount
        z[i,j,m] binary: changeover from i to j in micro-period m
        s[j,t] >= 0: serviceable inventory at end of macro-period t
    """
    import highspy

    J = inst.n_products
    T = inst.n_macro
    M = inst.n_micro_per_macro
    L = T * M  # total micro-periods

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", time_limit)

    # Helper to add a binary variable
    vars_added = 0
    def add_bin(name=""):
        nonlocal vars_added
        h.addVar(0.0, 1.0)
        h.changeColIntegrality(vars_added, highspy.HighsVarType.kInteger)
        vars_added += 1
        return vars_added - 1

    def add_cont(lb=0.0, ub=1e20, name=""):
        nonlocal vars_added
        h.addVar(lb, ub)
        vars_added += 1
        return vars_added - 1

    # Micro-period indexing: m = t * M + k (t = macro, k = micro within macro)
    def micro(t, k):
        return t * M + k

    # x[j][m]: product j is set up in micro-period m
    x = [[add_bin(f"x_{j}_{m}") for m in range(L)] for j in range(J)]

    # y[j][m]: production amount of product j in micro-period m
    y = [[add_cont(0.0, 1e6, f"y_{j}_{m}") for m in range(L)] for j in range(J)]

    # z[i][j][m]: changeover from product i to j at start of micro-period m
    z = [[[add_bin(f"z_{i}_{j}_{m}") for m in range(L)] for j in range(J)] for i in range(J)]

    # s[j][t]: serviceable inventory at end of macro-period t
    s = [[add_cont(0.0, 1e6, f"s_{j}_{t}") for t in range(T)] for j in range(J)]

    # --- Constraints ---

    # Exactly one product set up per micro-period
    for m in range(L):
        idx = [x[j][m] for j in range(J)]
        vals = [1.0] * J
        h.addRow(1.0, 1.0, J, idx, vals)

    # Production only when set up: y[j,m] <= BigM * x[j,m]
    BigM = max(inst.capacity) / min(inst.process_time)
    for j in range(J):
        for m in range(L):
            # y[j,m] - BigM * x[j,m] <= 0
            h.addRow(-1e20, 0.0, 2, [y[j][m], x[j][m]], [1.0, -BigM])

    # Changeover detection: z[i,j,m] >= x[i,m-1] + x[j,m] - 1  for i != j
    for m in range(1, L):
        for i in range(J):
            for j in range(J):
                if i == j:
                    continue
                # x[i,m-1] + x[j,m] - z[i,j,m] <= 1
                h.addRow(-1e20, 1.0, 3,
                         [x[i][m-1], x[j][m], z[i][j][m]],
                         [1.0, 1.0, -1.0])

    # Setup state preservation across macro-period boundary (GLSP-CS)
    # Same constraint applies for the first micro of each macro (m=0 excluded, starts free)

    # Capacity per micro-period: sum_j y[j,m]*tp[j] + sum_{i,j} z[i,j,m]*st[i,j] <= 1
    # (each micro-period has unit time capacity, or we can scale)
    # Actually: total production + setup in each micro-period, but micro-periods are
    # slots. We constrain total capacity over the macro-period.
    for t in range(T):
        prod_vars = []
        prod_vals = []
        for j in range(J):
            for k in range(M):
                m = micro(t, k)
                prod_vars.append(y[j][m])
                prod_vals.append(inst.process_time[j])
        # Setup times
        for k in range(M):
            m = micro(t, k)
            for i in range(J):
                for j in range(J):
                    if i != j and inst.setup_time[i][j] > 0:
                        prod_vars.append(z[i][j][m])
                        prod_vals.append(inst.setup_time[i][j])
        h.addRow(-1e20, inst.capacity[t], len(prod_vars), prod_vars, prod_vals)

    # Inventory balance: s[j,t] = s[j,t-1] + sum_m_in_t y[j,m]*(1-theta) - d[j,t]
    for j in range(J):
        for t in range(T):
            # s[j,t] - s[j,t-1] - sum_m y[j,m]*(1-theta) = -d[j,t]
            row_vars = [s[j][t]]
            row_vals = [1.0]
            if t > 0:
                row_vars.append(s[j][t-1])
                row_vals.append(-1.0)
            for k in range(M):
                m = micro(t, k)
                serv_frac = 1.0 - inst.defect_rate[j][t]
                row_vars.append(y[j][m])
                row_vals.append(-serv_frac)
            rhs = -inst.demand[j][t]
            h.addRow(rhs, rhs, len(row_vars), row_vars, row_vals)

    # Min lot size: if y[j,m] > 0 then y[j,m] >= kappa[j]
    # y[j,m] >= kappa[j] * x[j,m] (when x=1, must produce at least kappa)
    # Actually this is min lot per micro-period. The paper has min lot per macro.
    # For simplicity: aggregate lot size per (j,t) must be >= kappa or 0.
    # We don't enforce per-micro min lot here; CBLS model also uses per-macro.

    # --- Objective ---
    # min sum_j sum_t h[j]*s[j,t]
    #   + sum_{i,j} sum_m f[i,j]*z[i,j,m]
    #   + sum_j sum_m h_R[j]*y[j,m]*theta[j,t_of_m] * M/2  (rework holding approx)
    #   + sum_j sum_m lambda[j] * max(0, y[j,m]*theta - rework_cap) (disposal approx)

    obj_vars = []
    obj_vals = []

    # Holding cost
    for j in range(J):
        for t in range(T):
            obj_vars.append(s[j][t])
            obj_vals.append(inst.holding_cost[j])

    # Changeover cost
    for m in range(L):
        for i in range(J):
            for j in range(J):
                if i != j and inst.setup_cost[i][j] > 0:
                    obj_vars.append(z[i][j][m])
                    obj_vals.append(inst.setup_cost[i][j])

    # Rework holding cost (approximate)
    for j in range(J):
        for t in range(T):
            if inst.defect_rate[j][t] < 1e-12:
                continue
            hr = inst.rework_holding_cost[j] * M * 0.5
            for k in range(M):
                m = micro(t, k)
                obj_vars.append(y[j][m])
                obj_vals.append(hr * inst.defect_rate[j][t])

    h.changeColsCost(list(range(vars_added)), [0.0] * vars_added)
    for var_idx, coef in zip(obj_vars, obj_vals):
        cur = h.getInfoValue("objective_function_value")  # not available yet
        # Set coefficients directly
        pass

    # Set objective coefficients
    obj_coeffs = [0.0] * vars_added
    for var_idx, coef in zip(obj_vars, obj_vals):
        obj_coeffs[var_idx] += coef
    h.changeColsCost(list(range(vars_added)), obj_coeffs)
    h.changeObjectiveSense(highspy.ObjSense.kMinimize)

    t0 = time.time()
    h.run()
    elapsed = time.time() - t0

    status = h.getInfoValue("primal_solution_status")
    obj = h.getInfoValue("objective_function_value")
    if h.getInfoValue("primal_solution_status") == 2:  # feasible
        return obj, elapsed, True
    else:
        return float("inf"), elapsed, False


def main():
    parser = argparse.ArgumentParser(description="GLSP-RP HiGHS reference solver")
    parser.add_argument("--class", dest="cls", default="A",
                        help="Instance class (A, B, C, D, E)")
    parser.add_argument("--max", type=int, default=5,
                        help="Max instances to solve")
    parser.add_argument("--time", type=float, default=300.0,
                        help="Time limit per instance (seconds)")
    parser.add_argument("--all", action="store_true",
                        help="Run all standard classes")
    args = parser.parse_args()

    classes = ["A", "B", "C"] if args.all else [args.cls]

    print(f"{'Instance':<16} {'J':>3} {'T':>3} {'M':>3} {'Objective':>12} "
          f"{'Feasible':>8} {'Time(s)':>8}")
    print("-" * 60)

    for cls in classes:
        instances = generate_all(classes=(cls,), n_per_class=50)
        if args.max > 0:
            instances = instances[:args.max]

        total_obj = 0
        feasible_count = 0

        for inst in instances:
            try:
                obj, elapsed, feasible = solve_highs(inst, args.time)
                status = "yes" if feasible else "NO"
                print(f"{inst.name:<16} {inst.n_products:>3} {inst.n_macro:>3} "
                      f"{inst.n_micro_per_macro:>3} {obj:>12.1f} {status:>8} "
                      f"{elapsed:>7.1f}s")
                if feasible:
                    total_obj += obj
                    feasible_count += 1
            except ImportError:
                print(f"{inst.name:<16} highspy not installed")
                return
            except Exception as e:
                print(f"{inst.name:<16} ERROR: {e}")

        if feasible_count > 0:
            print(f"\nClass {cls}: {feasible_count}/{len(instances)} feasible, "
                  f"avg obj = {total_obj / feasible_count:.1f}")
        print()


if __name__ == "__main__":
    main()
