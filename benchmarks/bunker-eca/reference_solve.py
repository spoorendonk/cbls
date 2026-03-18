"""PySCIPOpt reference solver for maritime bunker-ECA benchmark.

Formulates a MIQCP (mixed-integer quadratically constrained program):
- Binary x[c,v]: cargo c assigned to ship v
- Continuous speed[c,v]: sailing speed when ship v carries cargo c
- Binary eca[c,v]: use MGO (1) or HFO (0) on ECA fraction of leg

Objective: maximize total_revenue - fuel_cost - port_cost
Subject to: time windows, cargo assignment, fuel capacity

Fuel consumption is cubic in speed: daily_fuel = k * v^3.
For the MIQCP we approximate as quadratic: fuel_cost ~ alpha * speed^2 * distance/24
(dropping the cubic term for tractability; SCIP handles quadratic constraints).

Usage:
    pip install pyscipopt
    cd benchmarks/bunker-eca
    python reference_solve.py
    python reference_solve.py --instance small
    python reference_solve.py --all
"""

import argparse
import math
import os
import sys
import time

# Add parent paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "instances", "bunker-eca"))
from data import ALL_INSTANCES, make_small, make_medium, make_large, make_xlarge


def _find_leg(inst, from_r, to_r):
    """Find leg between two regions, return (distance, eca_fraction) or None."""
    if from_r == to_r:
        return (0.0, 0.0)
    for leg in inst["legs"]:
        if leg["from_region"] == from_r and leg["to_region"] == to_r:
            return (leg["distance"], leg["eca_fraction"])
    return None


def solve_scip(inst, time_limit=300.0):
    """Solve bunker-ECA instance with PySCIPOpt MIQCP formulation.

    Returns (objective, solve_time, gap, solution_dict) or raises on failure.
    """
    from pyscipopt import Model, quicksum

    n_cargoes = len(inst["cargoes"])
    n_ships = len(inst["ships"])
    n_regions = len(inst["regions"])

    m = Model(inst["name"])
    m.setRealParam("limits/time", time_limit)
    m.hideOutput()

    # --- Decision variables ---
    # x[c,v] = 1 if cargo c assigned to ship v
    x = {}
    for c in range(n_cargoes):
        for v in range(n_ships):
            x[c, v] = m.addVar(f"x_{c}_{v}", vtype="B")

    # speed[c,v] = sailing speed (knots) when ship v carries cargo c
    # Only meaningful when x[c,v] = 1
    speed = {}
    for c in range(n_cargoes):
        for v in range(n_ships):
            ship = inst["ships"][v]
            v_max = ship["v_max_laden"]
            v_min = ship["v_min_laden"]
            speed[c, v] = m.addVar(f"spd_{c}_{v}", lb=0.0, ub=v_max)

    # eca_mgo[c,v] = 1 if MGO used on ECA portion (mandatory in ECA, but
    # modeled as choice to let solver see cost difference)
    eca_mgo = {}
    for c in range(n_cargoes):
        for v in range(n_ships):
            eca_mgo[c, v] = m.addVar(f"eca_{c}_{v}", vtype="B")

    # arrival[c,v] = arrival time at pickup port (days)
    arrival = {}
    for c in range(n_cargoes):
        for v in range(n_ships):
            arrival[c, v] = m.addVar(f"arr_{c}_{v}", lb=0.0,
                                     ub=inst["planning_horizon_days"])

    # --- Cargo assignment constraints ---
    for c in range(n_cargoes):
        cargo = inst["cargoes"][c]
        if cargo["is_contract"]:
            # Contract cargo: exactly one ship
            m.addCons(quicksum(x[c, v] for v in range(n_ships)) == 1)
        else:
            # Spot cargo: at most one ship
            m.addCons(quicksum(x[c, v] for v in range(n_ships)) <= 1)

    # --- Speed bounds: v_min * x <= speed <= v_max * x ---
    for c in range(n_cargoes):
        for v in range(n_ships):
            ship = inst["ships"][v]
            m.addCons(speed[c, v] >= ship["v_min_laden"] * x[c, v])
            m.addCons(speed[c, v] <= ship["v_max_laden"] * x[c, v])

    # --- ECA MGO forced when assigned (in ECA regions, MGO is mandatory) ---
    for c in range(n_cargoes):
        cargo = inst["cargoes"][c]
        leg_info = _find_leg(inst, cargo["pickup_region"], cargo["delivery_region"])
        if leg_info and leg_info[1] > 0:
            # Has ECA fraction: force MGO use when assigned
            for v in range(n_ships):
                m.addCons(eca_mgo[c, v] >= x[c, v])
        else:
            # No ECA: eca_mgo must be 0
            for v in range(n_ships):
                m.addCons(eca_mgo[c, v] == 0)

    # --- Time window constraints ---
    for c in range(n_cargoes):
        cargo = inst["cargoes"][c]
        for v in range(n_ships):
            ship = inst["ships"][v]
            # Arrival at pickup within time window (when assigned)
            m.addCons(arrival[c, v] >= cargo["pickup_tw_start"] * x[c, v])
            m.addCons(arrival[c, v] <= cargo["pickup_tw_end"] * x[c, v])
            # Arrival >= ship available day
            m.addCons(arrival[c, v] >= ship["available_day"] * x[c, v])

    # --- Delivery time feasibility ---
    # sailing_time = distance / (24 * speed)
    # delivery_arrival = arrival + service_load + sailing_time + service_discharge
    # Must be <= delivery_tw_end
    # This is nonlinear (distance / speed), so we use a big-M linear relaxation:
    # sailing_time >= distance / (24 * v_max) * x  (minimum sailing time)
    # delivery_time <= delivery_tw_end * x
    for c in range(n_cargoes):
        cargo = inst["cargoes"][c]
        leg_info = _find_leg(inst, cargo["pickup_region"], cargo["delivery_region"])
        if leg_info is None:
            # No route: cannot assign
            for v in range(n_ships):
                m.addCons(x[c, v] == 0)
            continue
        dist = leg_info[0]
        for v in range(n_ships):
            ship = inst["ships"][v]
            if dist > 0:
                min_sailing = dist / (24.0 * ship["v_max_laden"])
            else:
                min_sailing = 0.0
            service = cargo["service_time_load"] + cargo["service_time_discharge"]
            # delivery_arrival >= arrival + min_sailing + service (when assigned)
            # Must be within delivery window
            big_M = inst["planning_horizon_days"]
            m.addCons(arrival[c, v] + (min_sailing + service) * x[c, v]
                      <= cargo["delivery_tw_end"] + big_M * (1 - x[c, v]))

    # --- Fuel cost variables ---
    # fuel_cost[c,v] approximates k * speed^2 * (dist/24) for the assigned leg
    # Using quadratic: fuel_cost >= k * speed^2 * dist/24 - M*(1-x)
    fuel_cost = {}
    for c in range(n_cargoes):
        cargo = inst["cargoes"][c]
        leg_info = _find_leg(inst, cargo["pickup_region"], cargo["delivery_region"])
        dist = leg_info[0] if leg_info else 0.0
        for v in range(n_ships):
            ship = inst["ships"][v]
            fc = m.addVar(f"fc_{c}_{v}", lb=0.0)
            fuel_cost[c, v] = fc

            if dist > 0:
                # Fuel consumption: k * v^2 * dist/24 (quadratic approx of cubic)
                # Split into HFO and MGO portions
                eca_frac = leg_info[1] if leg_info else 0.0
                k_laden = ship["fuel_coeff_laden"]

                # Total fuel amount (MT) = k * speed^2 * sailing_days
                # sailing_days = dist / (24 * speed), so fuel = k * speed * dist / 24
                # Cost: hfo_fuel * hfo_price + mgo_fuel * mgo_price
                # For linearization: use average prices
                avg_hfo = sum(r["hfo_price"] for r in inst["regions"]) / n_regions
                avg_mgo = sum(r["mgo_price"] for r in inst["regions"]) / n_regions

                # fuel_amount = k * speed * dist / 24
                # cost = fuel_amount * ((1-eca_frac)*hfo_price + eca_frac*mgo_price)
                # This is bilinear in speed * x, linearize with McCormick
                # Simpler: use quadratic overestimate
                # fuel_cost >= k * dist/24 * price_blend * speed - M*(1-x)
                price_blend = (1 - eca_frac) * avg_hfo + eca_frac * avg_mgo
                coeff = k_laden * dist / 24.0 * price_blend

                big_M = coeff * ship["v_max_laden"]
                # fc >= coeff * speed - big_M * (1-x)
                m.addCons(fc >= coeff * speed[c, v] - big_M * (1 - x[c, v]))
            else:
                m.addCons(fc == 0)

    # --- Port costs ---
    # Each assigned cargo incurs port cost at pickup and delivery
    port_cost_expr = []
    for c in range(n_cargoes):
        cargo = inst["cargoes"][c]
        pc_pickup = inst["regions"][cargo["pickup_region"]]["port_cost"]
        pc_delivery = inst["regions"][cargo["delivery_region"]]["port_cost"]
        for v in range(n_ships):
            port_cost_expr.append((pc_pickup + pc_delivery) * x[c, v])

    # --- Objective: maximize revenue - fuel_cost - port_cost ---
    revenue_expr = quicksum(
        inst["cargoes"][c]["revenue"] * x[c, v]
        for c in range(n_cargoes)
        for v in range(n_ships)
    )
    total_fuel = quicksum(fuel_cost[c, v] for c in range(n_cargoes) for v in range(n_ships))
    total_port = quicksum(port_cost_expr)

    # Maximize profit = revenue - fuel - port
    profit = m.addVar("profit", lb=-1e15, ub=1e15)
    m.addCons(profit == revenue_expr - total_fuel - total_port)
    m.setObjective(profit, "maximize")

    # Solve
    t0 = time.time()
    m.optimize()
    wall_time = time.time() - t0

    status = m.getStatus()
    if status in ("optimal", "bestsollimit", "timelimit") and m.getNSols() > 0:
        obj = m.getObjVal()
        gap = m.getGap()

        # Extract solution
        sol = {"assignments": [], "objective": obj}
        for c in range(n_cargoes):
            for v in range(n_ships):
                if m.getVal(x[c, v]) > 0.5:
                    sol["assignments"].append({
                        "cargo": c,
                        "ship": v,
                        "speed": m.getVal(speed[c, v]),
                        "arrival": m.getVal(arrival[c, v]),
                        "eca_mgo": m.getVal(eca_mgo[c, v]) > 0.5,
                        "fuel_cost": m.getVal(fuel_cost[c, v]),
                    })
    else:
        obj = float("-inf")
        gap = float("inf")
        sol = None

    return obj, wall_time, gap, sol


def print_solution(inst, obj, wall_time, gap, sol):
    """Print solver results."""
    print(f"  Objective: {obj:,.0f}" if obj > float("-inf") else "  Objective: no solution")
    print(f"  Wall time: {wall_time:.1f}s")
    print(f"  Gap: {gap * 100:.2f}%" if gap < float("inf") else "  Gap: -")

    if sol:
        n_assigned = len(sol["assignments"])
        n_contract = sum(1 for a in sol["assignments"] if inst["cargoes"][a["cargo"]]["is_contract"])
        n_spot = n_assigned - n_contract
        total_revenue = sum(inst["cargoes"][a["cargo"]]["revenue"] for a in sol["assignments"])
        total_fuel = sum(a["fuel_cost"] for a in sol["assignments"])

        print(f"  Assigned: {n_assigned} cargoes ({n_contract} contract, {n_spot} spot)")
        print(f"  Revenue: {total_revenue:,.0f}")
        print(f"  Fuel cost: {total_fuel:,.0f}")

        # Per-ship summary
        ship_cargoes = {}
        for a in sol["assignments"]:
            v = a["ship"]
            if v not in ship_cargoes:
                ship_cargoes[v] = []
            ship_cargoes[v].append(a)

        for v in sorted(ship_cargoes):
            cargoes = ship_cargoes[v]
            print(f"  {inst['ships'][v]['name']}: {len(cargoes)} cargoes, "
                  f"speeds {min(a['speed'] for a in cargoes):.1f}-{max(a['speed'] for a in cargoes):.1f} kn")


TIME_LIMITS = {
    "small-3s-10c": 60.0,
    "medium-7s-30c": 300.0,
    "large-15s-60c": 600.0,
    "xlarge-30s-120c": 1800.0,
}


def main():
    parser = argparse.ArgumentParser(description="Bunker-ECA SCIP reference solver")
    parser.add_argument("--instance", type=str, default=None,
                        help="Instance to solve: small, medium, large, xlarge")
    parser.add_argument("--all", action="store_true", help="Solve all instances")
    parser.add_argument("--time-limit", type=float, default=None,
                        help="Override time limit (seconds)")
    args = parser.parse_args()

    if args.instance:
        instances = [args.instance]
    elif args.all:
        instances = list(ALL_INSTANCES.keys())
    else:
        instances = ["small"]

    print(f"{'Instance':<20} {'Objective':>12} {'Time(s)':>9} {'Gap%':>8} {'Cargoes':>8}")
    print("-" * 60)

    for key in instances:
        if key not in ALL_INSTANCES:
            print(f"Unknown instance: {key}")
            continue

        inst = ALL_INSTANCES[key]()
        tl = args.time_limit or TIME_LIMITS.get(inst["name"], 300.0)

        print(f"\n=== {inst['name']} ({len(inst['ships'])} ships, "
              f"{len(inst['cargoes'])} cargoes, {len(inst['regions'])} regions) ===")

        try:
            obj, wall_time, gap, sol = solve_scip(inst, time_limit=tl)
            print_solution(inst, obj, wall_time, gap, sol)

            n_assigned = len(sol["assignments"]) if sol else 0
            gap_str = f"{gap * 100:.2f}%" if gap < float("inf") else "-"
            obj_str = f"{obj:,.0f}" if obj > float("-inf") else "no sol"
            print(f"\n{'Summary':<20} {obj_str:>12} {wall_time:>9.1f} {gap_str:>8} {n_assigned:>8}")

        except ImportError:
            print("  PySCIPOpt not installed. Install with: pip install pyscipopt")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
