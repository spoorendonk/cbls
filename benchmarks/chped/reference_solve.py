"""Compare off-the-shelf solvers on CHPED and UC-CHPED benchmark instances.

Runs scipy differential_evolution and PySCIPOpt on dispatch-only CHPED and
unit commitment UC-CHPED instances. Prints comparison tables with objective
values, timings, and gaps vs known bounds.

Usage:
    pip install scipy pyscipopt
    cd benchmarks/chped
    python reference_solve.py          # dispatch-only CHPED
    python reference_solve.py --uc     # UC-CHPED instances
    python reference_solve.py --all    # both
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from data import CHPED_13UNIT, CHPED_40UNIT
from scipy.optimize import LinearConstraint, differential_evolution

#: A benchmark instance, as the data modules define it: a heterogeneous mapping
#: of scalars and per-unit lists. `Any` rather than a TypedDict because the
#: UC-CHPED instances are loaded from a file path at runtime (the directory is
#: hyphenated and cannot be imported as a package), so nothing more precise
#: survives the boundary.
Instance = dict[str, Any]

#: A PySCIPOpt model or variable. `pyscipopt` ships no type stubs, so this names
#: the boundary rather than pretending to check across it.
ScipModel = Any


@dataclass(frozen=True)
class UCVars:
    """The unit-commitment decision variables, keyed by `(unit, period)`."""

    #: 1 if the unit is on
    y: dict[tuple[int, int], Any]
    #: dispatch
    p: dict[tuple[int, int], Any]
    #: 1 if the unit starts up
    su: dict[tuple[int, int], Any]
    #: 1 if the unit shuts down
    sd: dict[tuple[int, int], Any]
    #: startup cost
    sc: dict[tuple[int, int], Any]


_uc_path = os.path.join(os.path.dirname(__file__), "..", "instances", "uc-chped", "data.py")
_spec = importlib.util.spec_from_file_location("uc_chped_data", _uc_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - a broken checkout
    raise SystemExit(f"cannot load the UC-CHPED instance data from {_uc_path}")
_uc_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_uc_data)
UCP_13UNIT = _uc_data.UCP_13UNIT
UCP_40UNIT = _uc_data.UCP_40UNIT
make_subinstance = _uc_data.make_subinstance


def cost_function(P: Any, inst: Instance) -> float:
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


def solve_scipy(inst: Instance) -> tuple[float, float, bool]:
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


def solve_scip(inst: Instance, time_limit: float) -> tuple[float, float, float]:
    """Solve with PySCIPOpt (global optimizer)."""
    from pyscipopt import Model, quicksum, sin  # noqa: E401

    n = inst["n_units"]
    m = Model(inst["name"])
    m.setRealParam("limits/time", time_limit)
    m.hideOutput()

    P = [m.addVar(f"P_{i}", lb=inst["P_min"][i], ub=inst["P_max"][i]) for i in range(n)]

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


# ---------------------------------------------------------------------------
# UC-CHPED solver (MIP with piecewise-linear valve-point approximation)
# ---------------------------------------------------------------------------


def _add_uc_variables(m: ScipModel, inst: Instance) -> UCVars:
    """Commitment, dispatch, start/stop and startup-cost variables."""
    y, p, su, sd, sc = {}, {}, {}, {}, {}
    for u in range(inst["n_units"]):
        for t in range(inst["n_periods"]):
            y[u, t] = m.addVar(f"y_{u}_{t}", vtype="B")
            p[u, t] = m.addVar(f"p_{u}_{t}", lb=0.0, ub=inst["P_max"][u])
            su[u, t] = m.addVar(f"su_{u}_{t}", vtype="B")
            sd[u, t] = m.addVar(f"sd_{u}_{t}", vtype="B")
            sc[u, t] = m.addVar(f"sc_{u}_{t}", lb=0.0)
    return UCVars(y=y, p=p, su=su, sd=sd, sc=sc)


def _add_balance_constraints(m: ScipModel, inst: Instance, v: UCVars) -> None:
    """Commitment logic, dispatch bounds, demand and spinning reserve."""
    from pyscipopt import quicksum

    n, periods = inst["n_units"], inst["n_periods"]
    for u in range(n):
        for t in range(periods):
            y_prev = inst["y_prev"][u] if t == 0 else v.y[u, t - 1]
            m.addCons(v.su[u, t] - v.sd[u, t] == v.y[u, t] - y_prev)
            m.addCons(v.su[u, t] + v.sd[u, t] <= 1)
            m.addCons(v.p[u, t] >= inst["P_min"][u] * v.y[u, t])
            m.addCons(v.p[u, t] <= inst["P_max"][u] * v.y[u, t])
    for t in range(periods):
        m.addCons(quicksum(v.p[u, t] for u in range(n)) >= inst["demand"][t])
        m.addCons(
            quicksum(inst["P_max"][u] * v.y[u, t] for u in range(n))
            >= inst["demand"][t] + inst["reserve"][t]
        )


def _add_min_up_down_constraints(m: ScipModel, inst: Instance, v: UCVars) -> None:
    """Minimum up- and down-time, including the state carried into the horizon."""
    periods = inst["n_periods"]
    for u in range(inst["n_units"]):
        min_on = inst["min_on"][u]
        # Initial: if unit was ON, must respect remaining min_on
        if inst["y_prev"][u] == 1:
            remaining = max(0, min_on - inst["n_init"][u])
            for t in range(min(remaining, periods)):
                m.addCons(v.y[u, t] == 1)
        # Ongoing: if started up at t, must stay on for min_on periods
        for t in range(periods):
            for tau in range(t + 1, min(t + min_on, periods)):
                m.addCons(v.y[u, tau] >= v.su[u, t])

        min_off = inst["min_off"][u]
        # Initial: if unit was OFF, must respect remaining min_off
        if inst["y_prev"][u] == 0:
            remaining = max(0, min_off - inst["n_init"][u])
            for t in range(min(remaining, periods)):
                m.addCons(v.y[u, t] == 0)
        # Ongoing: if shut down at t, must stay off for min_off periods
        for t in range(periods):
            for tau in range(t + 1, min(t + min_off, periods)):
                m.addCons(1 - v.y[u, tau] >= v.sd[u, t])


def _hot_start_lookback(inst: Instance, v: UCVars, u: int, t: int) -> list[object]:
    """Terms that make a hot start possible for unit `u` at period `t`.

    `y[u, tau]` over the lookback window, plus a literal 1 when the state carried
    into the horizon already puts the unit inside `t_cold`.
    """
    t_cold = inst["t_cold"][u]
    lookback: list[object] = [v.y[u, tau] for tau in range(max(0, t - t_cold), t)]
    if t >= t_cold:
        return lookback
    if inst["y_prev"][u] == 1:
        # Was ON before the horizon -- hot start always possible
        lookback.append(1)
    elif inst["n_init"][u] < t_cold and inst["n_init"][u] + t < t_cold:
        # Was OFF for n_init periods but ON before that, and the total off time
        # is still inside t_cold
        lookback.append(1)
    return lookback


def _add_startup_cost_constraints(m: ScipModel, inst: Instance, v: UCVars) -> None:
    """Hot start if the unit ran within `t_cold`, cold otherwise.

    `sc >= a_hot * su` always; `sc >= a_cold * su - (a_cold - a_hot) * w`, where
    `w` may be 1 only if some unit-on term is in the lookback window.
    """
    from pyscipopt import quicksum

    periods = inst["n_periods"]
    for u in range(inst["n_units"]):
        t_cold = inst["t_cold"][u]
        a_hot = inst["a_hot"][u]
        a_cold = inst["a_cold"][u]
        if t_cold == 0:
            # No cold start distinction -- always hot cost
            for t in range(periods):
                m.addCons(v.sc[u, t] >= a_hot * v.su[u, t])
            continue
        for t in range(periods):
            w = m.addVar(f"w_{u}_{t}", vtype="B")
            lookback = _hot_start_lookback(inst, v, u, t)
            if lookback:
                # w <= sum(lookback): can only be hot if someone was on. The
                # optimizer sets w = 1 when beneficial, since it reduces cost.
                m.addCons(w <= quicksum(lookback))
            else:
                # No lookback info -- must be cold
                m.addCons(w == 0)
            m.addCons(v.sc[u, t] >= a_hot * v.su[u, t])
            m.addCons(v.sc[u, t] >= a_cold * v.su[u, t] - (a_cold - a_hot) * w)


def _unit_breakpoints(inst: Instance, u: int, n_pwl_segments: int) -> tuple[Any, list[float]]:
    """Breakpoints over [Pmin, Pmax] and the true cost at each of them."""
    pmin = inst["P_min"][u]
    a_coef, b_coef = inst["a"][u], inst["b"][u]
    c_coef, d_coef, e_coef = inst["c"][u], inst["d"][u], inst["e"][u]
    breakpoints = np.linspace(pmin, inst["P_max"][u], n_pwl_segments + 1)
    costs_at_bp = []
    for bp in breakpoints:
        f = a_coef + b_coef * bp + c_coef * bp**2
        if d_coef != 0:
            f += abs(d_coef * math.sin(e_coef * (pmin - bp)))
        costs_at_bp.append(f)
    return breakpoints, costs_at_bp


def _add_pwl_cost(
    m: ScipModel, inst: Instance, v: UCVars, n_pwl_segments: int
) -> dict[tuple[int, int], object]:
    """Piecewise-linear approximation of the valve-point cost, per (unit, period).

    Incremental formulation: `p = Pmin * y + sum_k delta_k`, with `delta_k` bounded
    by its segment length and orderable only once the previous segment is full.
    """
    from pyscipopt import quicksum

    cost_pwl: dict[tuple[int, int], object] = {}
    for u in range(inst["n_units"]):
        pmin = inst["P_min"][u]
        breakpoints, costs_at_bp = _unit_breakpoints(inst, u, n_pwl_segments)
        for t in range(inst["n_periods"]):
            cost_var = m.addVar(f"cpwl_{u}_{t}", lb=0.0)
            cost_pwl[u, t] = cost_var

            deltas, slopes, seg_lens = [], [], []
            for k in range(n_pwl_segments):
                seg_len = breakpoints[k + 1] - breakpoints[k]
                slope = (costs_at_bp[k + 1] - costs_at_bp[k]) / seg_len if seg_len > 0 else 0
                deltas.append(m.addVar(f"d_{u}_{t}_{k}", lb=0.0, ub=seg_len))
                slopes.append(slope)
                seg_lens.append(seg_len)

            # Ordering: delta_k can only be > 0 if delta_{k-1} is at its max.
            zz = []
            for k in range(n_pwl_segments):
                z = m.addVar(f"z_{u}_{t}_{k}", vtype="B")
                zz.append(z)
                m.addCons(deltas[k] <= seg_lens[k] * z)
            for k in range(1, n_pwl_segments):
                m.addCons(deltas[k - 1] >= seg_lens[k - 1] * zz[k])

            # All deltas are 0 when the unit is off.
            for k in range(n_pwl_segments):
                m.addCons(deltas[k] <= seg_lens[k] * v.y[u, t])

            m.addCons(v.p[u, t] == pmin * v.y[u, t] + quicksum(deltas))
            m.addCons(
                cost_var
                == costs_at_bp[0] * v.y[u, t]
                + quicksum(slopes[k] * deltas[k] for k in range(n_pwl_segments))
            )
    return cost_pwl


def solve_uc_scip(
    inst: Instance, time_limit: float, n_pwl_segments: int = 50
) -> tuple[float, float, float]:
    """Solve UC-CHPED with PySCIPOpt.

    Model:
    - Binary y[u,t] for unit commitment
    - Continuous p[u,t] for dispatch
    - Piecewise-linear approximation of valve-point cost (SOS2-like via
      incremental formulation with n_pwl_segments breakpoints)
    - Min up/down time constraints
    - Hot/cold startup cost modeled via auxiliary variables
    - Demand + spinning reserve constraints

    Built in five steps rather than one function, along the model's own
    boundaries: variables, balance, min up/down, startup cost, piecewise cost.
    The arithmetic is unchanged -- this solver's numbers are published in
    `benchmarks/instances/uc-chped/comparison.csv`.
    """
    from pyscipopt import Model, quicksum

    m = Model(inst["name"])
    m.setRealParam("limits/time", time_limit)
    m.hideOutput()

    v = _add_uc_variables(m, inst)
    _add_balance_constraints(m, inst, v)
    _add_min_up_down_constraints(m, inst, v)
    _add_startup_cost_constraints(m, inst, v)
    cost_pwl = _add_pwl_cost(m, inst, v, n_pwl_segments)

    n, periods = inst["n_units"], inst["n_periods"]
    m.setObjective(
        quicksum(cost_pwl[u, t] for u in range(n) for t in range(periods))
        + quicksum(v.sc[u, t] for u in range(n) for t in range(periods))
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


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------


def main_chped() -> None:
    """Original dispatch-only CHPED benchmark."""
    print(
        f"{'Instance':<12} {'Solver':<12} {'Objective':>12} "
        f"{'Time(s)':>9} {'Gap%':>8} {'vs BKS':>8}"
    )
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
        rows.append(
            (name, "CBLS SA", f"{cbls['obj']:.2f}", f"{cbls['time']:.2f}", "-", f"+{pct:.2f}%")
        )

        for row in rows:
            print(f"{row[0]:<12} {row[1]:<12} {row[2]:>12} {row[3]:>9} {row[4]:>8} {row[5]:>8}")
        print()


UC_TIME_LIMITS = {
    1: 60.0,
    3: 120.0,
    6: 300.0,
    12: 600.0,
    24: 3600.0,
}


def main_uc() -> None:
    """UC-CHPED benchmark."""
    print("\n=== UC-CHPED (Unit Commitment with Valve-Point Effects) ===\n")
    print(
        f"{'Instance':<16} {'Periods':>7} {'SCIP Obj':>12} {'Time(s)':>9} "
        f"{'Gap%':>8} {'vs LB':>8} {'vs UB':>8}"
    )
    print("-" * 75)

    for base_inst in [UCP_13UNIT, UCP_40UNIT]:
        for n_periods in [1, 3, 6, 12, 24]:
            inst = make_subinstance(base_inst, n_periods)
            name = inst["name"]
            time_limit = UC_TIME_LIMITS.get(n_periods, 600.0)
            bounds = base_inst["known_bounds"].get(n_periods)

            try:
                obj, elapsed, gap = solve_uc_scip(inst, time_limit)
                gap_str = f"{gap * 100:.2f}%" if gap < float("inf") else "-"

                if bounds and obj < float("inf"):
                    lb, ub = bounds
                    vs_lb = f"+{(obj - lb) / lb * 100:.2f}%" if lb > 0 else "-"
                    vs_ub = f"+{(obj - ub) / ub * 100:.2f}%" if ub > 0 else "-"
                else:
                    vs_lb = "-"
                    vs_ub = "-"

                obj_str = f"{obj:.0f}" if obj < float("inf") else "no sol"
                print(
                    f"{name:<16} {n_periods:>7} {obj_str:>12} {elapsed:>9.1f} "
                    f"{gap_str:>8} {vs_lb:>8} {vs_ub:>8}"
                )
            except ImportError:
                print(f"{name:<16} {n_periods:>7} {'SCIP n/a':>12}")
            except Exception as e:
                print(f"{name:<16} {n_periods:>7} {'FAILED':>12}  {str(e)[:40]}")

        print()

    print("\nPedroso et al. (2014) known bounds for reference:")
    print(f"{'Instance':<16} {'Periods':>7} {'LB':>12} {'UB':>12} {'Gap%':>8}")
    print("-" * 60)
    for base_inst in [UCP_13UNIT, UCP_40UNIT]:
        for n_periods in sorted(base_inst["known_bounds"]):
            lb, ub = base_inst["known_bounds"][n_periods]
            gap_pct = (ub - lb) / lb * 100 if lb > 0 else 0
            print(f"{base_inst['name']:<16} {n_periods:>7} {lb:>12,} {ub:>12,} {gap_pct:>7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="CHPED/UC-CHPED reference solvers")
    parser.add_argument("--uc", action="store_true", help="Run UC-CHPED instances")
    parser.add_argument("--all", action="store_true", help="Run both CHPED and UC-CHPED")
    args = parser.parse_args()

    if args.all:
        main_chped()
        main_uc()
    elif args.uc:
        main_uc()
    else:
        main_chped()


if __name__ == "__main__":
    main()
