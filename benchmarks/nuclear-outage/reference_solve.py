"""Reference solver for nuclear outage scheduling using PySCIPOpt.

Formulates as a MIP with scenario sampling (sample average approximation).
Merit-order dispatch is linearized: production variables per unit per period
per scenario, with fuel cost as linear coefficient.

Constraints enforced: unit spacing (non-overlap), max outages per site,
min spacing between outages at the same site, demand satisfaction.

Usage:
    pip install pyscipopt numpy
    cd benchmarks/nuclear-outage
    python reference_solve.py              # mini instance
    python reference_solve.py --instance small
    python reference_solve.py --all
"""

import argparse
import json
import math
import os
import time

import numpy as np


def load_instance(name, inst_dir=None):
    """Load JSONL instance from disk."""
    if inst_dir is None:
        inst_dir = os.path.join(os.path.dirname(__file__), "..", "instances", "nuclear-outage")
    path = os.path.join(inst_dir, f"{name}.jsonl")
    with open(path) as f:
        return json.load(f)


def solve_mip(inst, time_limit=300.0, n_scenarios=None, seed=42):
    """Solve nuclear outage scheduling with PySCIPOpt MIP.

    Uses sample average approximation (SAA) with n_scenarios sampled
    demand scenarios. Each scenario contributes production variables
    p[u,t,s] dispatched by merit order.

    Returns (objective, solve_time, gap).
    """
    from pyscipopt import Model, quicksum

    rng = np.random.RandomState(seed)
    n_units = inst["n_units"]
    n_periods = inst["n_periods"]
    n_outages = inst["n_outages"]
    n_sites = inst["n_sites"]
    total_scenarios = inst["n_scenarios"]

    # Sample scenarios (SAA)
    if n_scenarios is None:
        n_scenarios = min(20, total_scenarios)
    scenario_idx = sorted(rng.choice(total_scenarios, size=n_scenarios, replace=False))
    demand = [inst["demand"][s] for s in scenario_idx]

    m = Model(f"nuclear-outage-{inst['name']}")
    m.setRealParam("limits/time", time_limit)
    m.hideOutput()

    # --- Decision variables ---
    # s[o] = start period of outage o (integer)
    s = {}
    for o in range(n_outages):
        s[o] = m.addVar(f"s_{o}",
                        lb=inst["outage_earliest"][o],
                        ub=inst["outage_latest"][o],
                        vtype="I")

    # a[u,t] = 1 if unit u is available at period t (binary, implied by outage schedule)
    # We linearize: a[u,t] = 1 - sum_{outages o of unit u} x[o,t]
    # where x[o,t] = 1 if outage o covers period t

    # x[o,t] = 1 if outage o is active at period t
    # x[o,t] = 1 iff s[o] <= t < s[o] + duration[o]
    # Linearize with big-M:
    # x[o,t] = 1 iff t - s[o] >= 0 and s[o] + duration[o] - 1 - t >= 0
    x = {}
    for o in range(n_outages):
        dur = inst["outage_duration"][o]
        e = inst["outage_earliest"][o]
        l = inst["outage_latest"][o]
        for t in range(n_periods):
            # x[o,t] can only be 1 if t is within possible range
            if t < e or t >= l + dur:
                x[o, t] = 0  # constant 0
            else:
                x[o, t] = m.addVar(f"x_{o}_{t}", vtype="B")
                # x[o,t] = 1 => s[o] <= t  => t - s[o] >= 0
                # x[o,t] = 1 => s[o] + dur - 1 >= t  => s[o] >= t - dur + 1
                M = l - e + dur  # big-M
                m.addCons(t - s[o] >= -M * (1 - x[o, t]))
                m.addCons(s[o] + dur - 1 - t >= -M * (1 - x[o, t]))
                # x[o,t] = 0 => s[o] > t or s[o] + dur - 1 < t
                # We enforce: x[o,t] >= 1 - (s[o] - t + dur - 1)/(dur) ... simplified
                # Instead use linking: sum_t x[o,t] = dur (outage lasts exactly dur periods)

        # Total duration constraint: outage o covers exactly dur periods
        active_vars = [x[o, t] for t in range(n_periods) if isinstance(x[o, t], type(s[0]))]
        if active_vars:
            m.addCons(quicksum(active_vars) == dur)

    # Unit availability
    # a[u,t] = 1 - sum_{outages o of unit u} x[o,t]
    # Group outages by unit
    unit_outages = [[] for _ in range(n_units)]
    for o in range(n_outages):
        unit_outages[inst["outage_unit"][o]].append(o)

    # --- Production dispatch per scenario ---
    # p[u,t,sc] = production of unit u in period t, scenario sc
    p = {}
    unserved = {}
    for sc in range(n_scenarios):
        for t in range(n_periods):
            for u in range(n_units):
                p[u, t, sc] = m.addVar(f"p_{u}_{t}_{sc}", lb=0.0, ub=inst["capacity"][u])

                # If unit is in outage, production = 0
                for o in unit_outages[u]:
                    if isinstance(x[o, t], type(s[0])):
                        m.addCons(p[u, t, sc] <= inst["capacity"][u] * (1 - x[o, t]))

            # Unserved energy
            unserved[t, sc] = m.addVar(f"unserved_{t}_{sc}", lb=0.0)

            # Demand constraint
            m.addCons(
                quicksum(p[u, t, sc] for u in range(n_units)) + unserved[t, sc]
                >= demand[sc][t]
            )

    # --- Spacing constraints (same unit) ---
    for u in range(n_units):
        outages = unit_outages[u]
        if len(outages) < 2:
            continue
        # Sort by earliest start
        outages_sorted = sorted(outages, key=lambda o: inst["outage_earliest"][o])
        for i in range(len(outages_sorted) - 1):
            o1, o2 = outages_sorted[i], outages_sorted[i + 1]
            m.addCons(s[o1] + inst["outage_duration"][o1] <= s[o2])

    # --- Site resource constraints ---
    for t in range(n_periods):
        site_outages = [[] for _ in range(n_sites)]
        for o in range(n_outages):
            site_id = inst["site"][inst["outage_unit"][o]]
            if isinstance(x[o, t], type(s[0])):
                site_outages[site_id].append(x[o, t])
        for site_id in range(n_sites):
            if len(site_outages[site_id]) > inst["max_outages_per_site"][site_id]:
                m.addCons(
                    quicksum(site_outages[site_id])
                    <= inst["max_outages_per_site"][site_id]
                )

    # --- Min spacing between outages at the same site ---
    min_spacing = inst["min_spacing_same_site"]
    site_outage_map = [[] for _ in range(n_sites)]
    for o in range(n_outages):
        site_id = inst["site"][inst["outage_unit"][o]]
        site_outage_map[site_id].append(o)
    for site_id in range(n_sites):
        outages_at_site = site_outage_map[site_id]
        for i in range(len(outages_at_site)):
            for j in range(i + 1, len(outages_at_site)):
                o1, o2 = outages_at_site[i], outages_at_site[j]
                # Either o1 ends + spacing <= o2 starts, or o2 ends + spacing <= o1 starts
                # Use big-M with binary indicator
                M_val = n_periods + max(inst["outage_duration"])
                order_var = m.addVar(f"ord_{o1}_{o2}", vtype="B")
                m.addCons(s[o1] + inst["outage_duration"][o1] + min_spacing
                          <= s[o2] + M_val * (1 - order_var))
                m.addCons(s[o2] + inst["outage_duration"][o2] + min_spacing
                          <= s[o1] + M_val * order_var)

    # --- Objective: expected cost ---
    m.setObjective(
        (1.0 / n_scenarios) * (
            quicksum(
                inst["fuel_cost"][u] * p[u, t, sc]
                for u in range(n_units) for t in range(n_periods)
                for sc in range(n_scenarios)
            )
            + quicksum(
                inst["penalty_unserved"] * unserved[t, sc]
                for t in range(n_periods) for sc in range(n_scenarios)
            )
        )
    )

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


def solve_greedy(inst):
    """Simple greedy heuristic: schedule outages as early as possible, spread across sites.

    Returns (objective, time).
    """
    t0 = time.time()

    n_outages = inst["n_outages"]
    n_periods = inst["n_periods"]
    n_units = inst["n_units"]
    n_sites = inst["n_sites"]
    n_scenarios = inst["n_scenarios"]

    # Sort outages by earliest start
    order = sorted(range(n_outages), key=lambda o: inst["outage_earliest"][o])

    starts = [0] * n_outages
    # Track site occupancy
    site_busy_until = [0] * n_sites  # earliest period site is free

    unit_busy_until = [0] * n_units

    for o in order:
        u = inst["outage_unit"][o]
        dur = inst["outage_duration"][o]
        site_id = inst["site"][u]
        earliest = max(inst["outage_earliest"][o], unit_busy_until[u])
        # Also respect site constraint
        earliest = max(earliest, site_busy_until[site_id])
        starts[o] = min(earliest, inst["outage_latest"][o])
        unit_busy_until[u] = starts[o] + dur
        site_busy_until[site_id] = starts[o] + dur + inst["min_spacing_same_site"]

    # Compute dispatch cost
    # Build availability
    avail = [[True] * n_units for _ in range(n_periods)]
    for o in range(n_outages):
        u_out = inst["outage_unit"][o]
        for t in range(starts[o], min(starts[o] + inst["outage_duration"][o], n_periods)):
            avail[t][u_out] = False

    # Merit order
    merit = sorted(range(n_units), key=lambda u: inst["fuel_cost"][u])

    total_cost = 0.0
    for s in range(n_scenarios):
        for t in range(n_periods):
            demand = inst["demand"][s][t]
            remaining = demand
            for u in merit:
                if remaining <= 0:
                    break
                if not avail[t][u]:
                    continue
                gen = min(inst["capacity"][u], remaining)
                total_cost += gen * inst["fuel_cost"][u]
                remaining -= gen
            if remaining > 0:
                total_cost += remaining * inst["penalty_unserved"]

    obj = total_cost / n_scenarios
    elapsed = time.time() - t0
    return obj, elapsed


def main():
    parser = argparse.ArgumentParser(description="Nuclear outage reference solver")
    parser.add_argument("--instance", default="mini", help="Instance name (mini, small, medium)")
    parser.add_argument("--all", action="store_true", help="Run all instances")
    parser.add_argument("--time-limit", type=float, default=300.0)
    parser.add_argument("--scenarios", type=int, default=None, help="Number of scenarios for SAA")
    args = parser.parse_args()

    instances = ["mini", "small"] if args.all else [args.instance]

    print(f"{'Instance':<12} {'Method':<12} {'Objective':>14} {'Time(s)':>9} {'Gap%':>8}")
    print("-" * 60)

    for name in instances:
        inst = load_instance(name)

        # Greedy heuristic
        obj_greedy, t_greedy = solve_greedy(inst)
        print(f"{name:<12} {'Greedy':<12} {obj_greedy:>14,.0f} {t_greedy:>9.1f} {'—':>8}")

        # MIP (SAA)
        try:
            n_sc = args.scenarios or min(20, inst["n_scenarios"])
            obj_mip, t_mip, gap = solve_mip(inst, args.time_limit, n_scenarios=n_sc)
            gap_str = f"{gap * 100:.1f}%" if gap < float("inf") else "no sol"
            obj_str = f"{obj_mip:,.0f}" if obj_mip < float("inf") else "no sol"
            print(f"{name:<12} {'MIP(SAA)':<12} {obj_str:>14} {t_mip:>9.1f} {gap_str:>8}")
        except ImportError:
            print(f"{name:<12} {'MIP(SAA)':<12} {'SCIP n/a':>14}")
        except Exception as e:
            print(f"{name:<12} {'MIP(SAA)':<12} {'FAILED':>14}  {str(e)[:40]}")

        print()


if __name__ == "__main__":
    main()
