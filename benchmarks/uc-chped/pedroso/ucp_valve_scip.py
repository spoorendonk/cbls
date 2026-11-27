# SPDX-License-Identifier: GPL-3.0-or-later
#
# A port to SCIP (PySCIPOpt) of UCP_VALVE's ucp_valve.py,
# Copyright (C) 2013 JP Pedroso, M Kubo, A Viana (Universidade do Porto),
# http://www.dcc.fc.up.pt/~jpp/code/valve/ucp_valve.py (2013-12-09, sha256
# c1f6e8d9ba316c6291e0eab447a3df65a773adacb1c996a610f68c62d46d9104).
# Port and the exact MINLP: Copyright (C) 2026 Simon Spoorendonk.
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version. It is distributed WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License in LICENSE beside this file for more details.
#
# This directory is GPL-3.0-or-later; the rest of cbls is MIT. Nothing outside
# this directory imports it: it reads the MIT instance data and writes JSON lines.
"""UC-CHPED references on SCIP: Pedroso's linearise-and-refine method, and the exact MINLP.

`--method pwl` is Pedroso, Kubo & Viana (2014), arXiv:1404.4944, Algorithm 1, as
their ucp_valve.py implements it: a piecewise-linear interpolation of the fuel
cost through the valve points, which never exceeds the cost while the cost is
concave between valve points (`c <= e*f^2/2` per unit, their Appendix B), so its
MIP optimum is a lower bound; the MIP's dispatch priced at the true cost is an
upper bound; breakpoints are added where the dispatch landed, and the MIP is
solved again until the bounds meet or the CPU budget runs out.

`--method exact` is not in the paper: the same unit-commitment model with the
true `|e*sin(f*(p_min - p))|` term, solved by SCIP's spatial branch-and-bound.

Both write one JSON line per solve. The deviations from ucp_valve.py are listed
in README.md beside this file.

`--backend pyscipopt` (the default) solves in process on the PySCIPOpt wheel's
SCIP, single-threaded, on a process-CPU budget. `--backend driver` (#442) keeps
the wheel as the modeller only: each MIP or MINLP goes out as CIP to
`cbls_scip_traj` (`benchmarks/common/scip_driver/`), the pinned SCIP 11 at
`--threads` N, on a wall budget, and the record gains the running best's
trajectory `[[wall_s, ub], ...]`, strictly falling, for a primal integral.

Coefficient names follow ucp_valve.py: the valve term is `|e*sin(f*(p_min-p))|`,
which cbls's data calls `|d*sin(e*(P_min-P))|`.

    .venv/bin/python benchmarks/uc-chped/pedroso/ucp_valve_scip.py \\
        --instance ucp13 --periods 3 --method pwl --cpu 3600 --out pwl.jsonl
"""

from __future__ import annotations

import argparse
import glob
import importlib.metadata
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

_uc_path = os.path.join(os.path.dirname(__file__), "..", "..", "instances", "uc-chped", "data.py")
_spec = importlib.util.spec_from_file_location("uc_chped_data", _uc_path)
if _spec is None or _spec.loader is None:  # pragma: no cover - a broken checkout
    raise SystemExit(f"cannot load the UC-CHPED instance data from {_uc_path}")
_uc_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_uc_data)

#: ucp_valve.py's EPS: the stopping tolerance on the relative gap, and the margin
#: `has_added_p` keeps from a valve point.
EPS = 1.0e-7
#: ucp_valve.py's local EPS in the breakpoint builders.
BRK_EPS = 1.0e-12
#: ucp_valve.py's starting number of breakpoints per refined valve interval.
N_INIT = 100
#: Gurobi's defaults that ucp_valve.py runs at outside strict mode. SCIP's
#: defaults are 0, which would turn every iteration into a full solve to optimality.
GUROBI_MIPGAP = 1.0e-4
GUROBI_MIPGAPABS = 1.0e-10

BASES: dict[str, Any] = {
    "ucp13": _uc_data.UCP_13UNIT,
    "ucp40": _uc_data.UCP_40UNIT,
    "ucp100": _uc_data.UCP_100UNIT,
    "ucp200": _uc_data.UCP_200UNIT,
}


@dataclass(frozen=True)
class Unit:
    """One generating unit, in ucp_valve.py's names."""

    a: float
    b: float
    c: float
    e: float
    f: float
    p_min: float
    p_max: float
    y_prev: int
    t_cold: int
    n_init: int
    min_on: int
    min_off: int
    a_hot: float
    a_cold: float


@dataclass(frozen=True)
class Problem:
    """An instance at a horizon. `dem[t - 1]` is period t's demand."""

    name: str
    units: list[Unit]
    dem: list[float]
    res: list[float]

    @property
    def horizon(self) -> int:
        return len(self.dem)


def load_problem(instance: str, periods: int) -> Problem:
    """Read cbls's instance data (identical to ucp_data.py for ucp13 and ucp40)."""
    base = BASES[instance]
    if periods <= base["n_periods"]:
        inst = _uc_data.make_subinstance(base, periods)
    else:
        inst = _uc_data.extend_horizon(base, periods)
    units = [
        Unit(
            a=float(inst["a"][u]),
            b=float(inst["b"][u]),
            c=float(inst["c"][u]),
            e=float(inst["d"][u]),
            f=float(inst["e"][u]),
            p_min=float(inst["P_min"][u]),
            p_max=float(inst["P_max"][u]),
            y_prev=int(inst["y_prev"][u]),
            t_cold=int(inst["t_cold"][u]),
            n_init=int(inst["n_init"][u]),
            min_on=int(inst["min_on"][u]),
            min_off=int(inst["min_off"][u]),
            a_hot=float(inst["a_hot"][u]),
            a_cold=float(inst["a_cold"][u]),
        )
        for u in range(int(inst["n_units"]))
    ]
    flat = [i for i, unit in enumerate(units) if unit.f <= 0]
    if flat:
        # ucp_valve.py substitutes f = pi/(p_max - p_min) in its breakpoint builders
        # but not in has_added_p, which divides by f; no instance here needs it.
        raise SystemExit(f"{instance}: units {flat} have valve frequency f <= 0")
    return Problem(
        name=str(inst["name"]),
        units=units,
        dem=[float(x) for x in inst["demand"]],
        res=[float(x) for x in inst["reserve"]],
    )


# ---------------------------------------------------------------------------
# The cost and its breakpoints (ucp_valve.py's cost, expand_brkpts, has_added_p)
# ---------------------------------------------------------------------------


def cost(u: Unit, p: float) -> float:
    """Operating cost with the valve-point effect (ucp_valve.py's `cost`, d = 0)."""
    return u.a + u.b * p + u.c * p * p + abs(u.e * math.sin(u.f * (u.p_min - p)))


def concave_between_valves(u: Unit) -> bool:
    """Pedroso's Appendix B condition for the interpolation to under-estimate."""
    return u.c <= u.e * u.f * u.f / 2.0


def expand_brkpts(u: Unit, p_list: list[float], n: int) -> list[tuple[float, float]]:
    """Breakpoints: every valve point, and `n` per valve interval holding a past dispatch."""
    f = u.f
    brk: list[tuple[float, float]] = []
    nvalve = math.ceil(f * (u.p_max - u.p_min) / math.pi)
    k = 0
    p_sol = p_list[k] if k < len(p_list) else math.inf
    for i in range(nvalve + 1):
        p0 = u.p_min + i * math.pi / f
        if p0 >= u.p_max - BRK_EPS:
            brk.append((u.p_max, cost(u, u.p_max)))
            break
        for j in range(n):
            p = p0 + j * math.pi / f / n
            if p >= u.p_max - BRK_EPS:
                break
            brk.append((p, cost(u, p)))
            left, right = u.p_min + i * math.pi / f, u.p_min + (i + 1) * math.pi / f
            if left >= p_sol - BRK_EPS or right <= p_sol + BRK_EPS:
                break
        while p_sol < u.p_min + (i + 1) * math.pi / f + BRK_EPS:
            k += 1
            p_sol = p_list[k] if k < len(p_list) else math.inf
    return brk


def has_added_p(u: Unit, prev: list[float], p: float) -> bool:
    """True when `p` sits inside a valve interval no past dispatch has refined yet."""
    f = u.f
    nvalve = math.ceil(f * (u.p_max - u.p_min) / math.pi)
    for i in range(nvalve + 1):
        left = u.p_min + i * math.pi / f
        right = u.p_min + (i + 1) * math.pi / f
        if left + EPS < p < right - EPS:
            for r in prev:
                if left + EPS < r < right - EPS:
                    return False
        elif left - EPS <= p <= right + EPS:  # a valve point
            return False
    return True


# ---------------------------------------------------------------------------
# The unit-commitment model (ucp_valve.py's ucp_base)
# ---------------------------------------------------------------------------


@dataclass
class UcpVars:
    """The model's variables, keyed (unit, period); period 0 is the initial state."""

    y: dict[tuple[int, int], Any] = field(default_factory=dict)
    on: dict[tuple[int, int], Any] = field(default_factory=dict)
    off: dict[tuple[int, int], Any] = field(default_factory=dict)
    cold: dict[tuple[int, int], Any] = field(default_factory=dict)
    hot: dict[tuple[int, int], Any] = field(default_factory=dict)
    p: dict[tuple[int, int], Any] = field(default_factory=dict)
    fuel: dict[tuple[int, int], Any] = field(default_factory=dict)
    startup: dict[tuple[int, int], Any] = field(default_factory=dict)


def ucp_base(pr: Problem) -> tuple[Any, UcpVars]:
    """The Kazarlis unit-commitment MIP, without its fuel cost."""
    from pyscipopt import Model, quicksum

    m = Model("Unit Commitment Problem")
    m.hideOutput()
    v = UcpVars()
    horizon = pr.horizon
    for u in range(len(pr.units)):
        v.y[u, 0] = m.addVar(vtype="B", name=f"y({u},0)")
        for t in range(1, horizon + 1):
            v.y[u, t] = m.addVar(vtype="B", name=f"y({u},{t})")
            v.on[u, t] = m.addVar(lb=0, ub=1, name=f"on({u},{t})")
            v.off[u, t] = m.addVar(lb=0, ub=1, name=f"off({u},{t})")
            v.cold[u, t] = m.addVar(lb=0, ub=1, name=f"cold({u},{t})")
            v.hot[u, t] = m.addVar(lb=0, ub=1, name=f"hot({u},{t})")
            v.p[u, t] = m.addVar(lb=0, ub=None, name=f"p({u},{t})")
            v.fuel[u, t] = m.addVar(lb=0, ub=None, name=f"fuel({u},{t})")
            v.startup[u, t] = m.addVar(lb=0, ub=None, name=f"startup({u},{t})")

    units = range(len(pr.units))
    for t in range(1, horizon + 1):
        m.addCons(quicksum(v.p[u, t] for u in units) == pr.dem[t - 1], name=f"demand({t})")
        m.addCons(
            quicksum(v.y[u, t] * pr.units[u].p_max for u in units) >= pr.dem[t - 1] + pr.res[t - 1],
            name=f"reserve({t})",
        )
        for u in units:
            m.addCons(v.y[u, t] * pr.units[u].p_min <= v.p[u, t], name=f"min_prod({u},{t})")
            m.addCons(v.y[u, t] * pr.units[u].p_max >= v.p[u, t], name=f"max_prod({u},{t})")

    for u, unit in enumerate(pr.units):
        _initial_conditions(m, v, unit, u, horizon)
        _commitment_rows(m, v, unit, u, horizon)
    return m, v


def _initial_conditions(m: Any, v: UcpVars, unit: Unit, u: int, horizon: int) -> None:
    if unit.y_prev == 1:
        _initially_on(m, v, unit, u, horizon)
    else:
        _initially_off(m, v, unit, u, horizon)


def _initially_on(m: Any, v: UcpVars, unit: Unit, u: int, horizon: int) -> None:
    m.chgVarUb(v.cold[u, 1], 0)
    m.chgVarUb(v.hot[u, 1], 0)
    for t in range(min(horizon, max(0, unit.min_on - unit.n_init)) + 1):
        m.chgVarLb(v.y[u, t], 1)
    # A unit on before the horizon with t_cold > min_off never starts cold in
    # periods 1..t_cold+1.
    if unit.t_cold > unit.min_off:
        for t in range(1, min(horizon, unit.t_cold + 1) + 1):
            m.chgVarUb(v.cold[u, t], 0)


def _initially_off(m: Any, v: UcpVars, unit: Unit, u: int, horizon: int) -> None:
    from pyscipopt import quicksum

    if unit.n_init > unit.t_cold:
        m.addCons(v.cold[u, 1] == v.y[u, 1])
        m.chgVarUb(v.hot[u, 1], 0)
    else:
        m.chgVarUb(v.cold[u, 1], 0)
        m.addCons(v.hot[u, 1] == v.y[u, 1])
    for t in range(min(horizon, max(0, unit.min_off - unit.n_init)) + 1):
        m.chgVarUb(v.y[u, t], 0)
    # A unit off before the horizon with t_cold > min_off may start hot or cold
    # in periods 1..t_cold+1.
    if unit.t_cold > unit.min_off:
        for t in range(2, min(horizon, unit.t_cold + 1) + 1):
            n = max(0, unit.t_cold - (t - 1) - unit.n_init + 1)
            m.addCons(
                v.y[u, t] - quicksum(v.y[u, i] for i in range(1, t)) - n * (1 - unit.y_prev)
                <= v.cold[u, t],
                name=f"cold_init({u},{t})",
            )


def _commitment_rows(m: Any, v: UcpVars, unit: Unit, u: int, horizon: int) -> None:
    from pyscipopt import quicksum

    if unit.t_cold > 0:
        for t in range(unit.t_cold + 2, horizon + 1):
            m.addCons(
                v.y[u, t] - quicksum(v.y[u, i] for i in range(t - unit.t_cold - 1, t))
                <= v.cold[u, t],
                name=f"c_cold({u},{t})",
            )
    for t in range(1, horizon + 1):
        # A unit with t_cold < min_off (or t_cold = 0) always starts cold.
        if unit.t_cold < unit.min_off or unit.t_cold == 0:
            m.chgVarUb(v.hot[u, t], 0)
        m.addCons(v.y[u, t] - v.y[u, t - 1] <= v.on[u, t], name=f"c_on({u},{t})")
        m.addCons(v.off[u, t] == v.on[u, t] + v.y[u, t - 1] - v.y[u, t], name=f"c_off({u},{t})")
        t0 = max(t - unit.min_on + 1, 1)
        m.addCons(
            quicksum(v.on[u, i] for i in range(t0, t + 1)) <= v.y[u, t], name=f"keep_on({u},{t})"
        )
        t0 = max(t - unit.min_off + 1, 1)
        m.addCons(
            quicksum(v.off[u, i] for i in range(t0, t + 1)) <= 1 - v.y[u, t],
            name=f"keep_off({u},{t})",
        )
        m.addCons(v.hot[u, t] + v.cold[u, t] == v.on[u, t], name=f"hot_or_cold({u},{t})")


def _startup_objective(m: Any, v: UcpVars, pr: Problem) -> Any:
    from pyscipopt import quicksum

    for u, unit in enumerate(pr.units):
        for t in range(1, pr.horizon + 1):
            m.addCons(v.startup[u, t] == unit.a_hot * v.hot[u, t] + unit.a_cold * v.cold[u, t])
    return quicksum(
        v.fuel[u, t] + v.startup[u, t]
        for u in range(len(pr.units))
        for t in range(1, pr.horizon + 1)
    )


def ucp_lin_obj(m: Any, v: UcpVars, pr: Problem, brk: dict[tuple[int, int], Any]) -> int:
    """The interpolated fuel cost as a convex combination under SOS2. Returns the breakpoints."""
    from pyscipopt import quicksum

    total = 0
    for u in range(len(pr.units)):
        for t in range(1, pr.horizon + 1):
            xs = [x for (x, _) in brk[u, t]]
            ys = [c for (_, c) in brk[u, t]]
            z = [m.addVar(lb=0, ub=1) for _ in xs]
            m.addCons(v.p[u, t] == quicksum(x * zk for x, zk in zip(xs, z, strict=True)))
            m.addCons(v.fuel[u, t] == quicksum(c * zk for c, zk in zip(ys, z, strict=True)))
            m.addCons(quicksum(z) == v.y[u, t])
            m.addConsSOS2(z)
            total += len(xs)
    m.setObjective(_startup_objective(m, v, pr), "minimize")
    return total


def ucp_exact_obj(m: Any, v: UcpVars, pr: Problem) -> None:
    """The true fuel cost. Off, `p = 0` and the valve auxiliary may drop to 0, as |e sin| <= e."""
    from pyscipopt import sin

    for u, unit in enumerate(pr.units):
        for t in range(1, pr.horizon + 1):
            y, p = v.y[u, t], v.p[u, t]
            valve = m.addVar(lb=0, ub=None, name=f"valve({u},{t})")
            ripple = unit.e * sin(unit.f * (unit.p_min - p))
            m.addCons(valve >= ripple - unit.e * (1 - y))
            m.addCons(valve >= -ripple - unit.e * (1 - y))
            m.addCons(v.fuel[u, t] >= unit.a * y + unit.b * p + unit.c * p * p + valve)
    m.setObjective(_startup_objective(m, v, pr), "minimize")


# ---------------------------------------------------------------------------
# Pricing and checking a solution
# ---------------------------------------------------------------------------


#: A solution as variable name -> value: what both backends hand back (#442).
Values = dict[str, float]


def solution_values(m: Any) -> Values:
    """The in-process model's best solution, by variable name."""
    return {var.name: float(m.getVal(var)) for var in m.getVars()}


def true_cost(x: Values, v: UcpVars, pr: Problem) -> float:
    """The solution's startup cost plus its dispatch priced at the true fuel cost."""
    fixed = sum(
        x[v.startup[u, t].name] for u in range(len(pr.units)) for t in range(1, pr.horizon + 1)
    )
    variable = sum(
        cost(unit, x[v.p[u, t].name])
        for u, unit in enumerate(pr.units)
        for t in range(1, pr.horizon + 1)
        if x[v.y[u, t].name] > 0.5
    )
    return float(fixed + variable)


def check_solution(x: Values, v: UcpVars, pr: Problem, tol: float = 1.0e-6) -> list[str]:
    """Demand, reserve and dispatch limits, re-checked outside SCIP. Empty when it passes."""
    errors: list[str] = []
    for t in range(1, pr.horizon + 1):
        on = [x[v.y[u, t].name] > 0.5 for u in range(len(pr.units))]
        p = [x[v.p[u, t].name] for u in range(len(pr.units))]
        dem = pr.dem[t - 1]
        if abs(sum(p) - dem) > tol * max(1.0, dem):
            errors.append(f"t={t}: dispatch {sum(p):.6f} != demand {dem}")
        cap = sum(unit.p_max for unit, o in zip(pr.units, on, strict=True) if o)
        if cap < dem + pr.res[t - 1] - tol * max(1.0, dem):
            errors.append(f"t={t}: committed capacity {cap} < demand + reserve")
        for u, unit in enumerate(pr.units):
            lo, hi = (unit.p_min, unit.p_max) if on[u] else (0.0, 0.0)
            if not lo - tol * max(1.0, hi) <= p[u] <= hi + tol * max(1.0, hi):
                errors.append(f"t={t} unit {u}: p {p[u]:.6f} outside [{lo}, {hi}]")
    return errors


# ---------------------------------------------------------------------------
# The two methods
# ---------------------------------------------------------------------------

SOLVED = ("optimal", "gaplimit")


def _finite_bound(m: Any) -> float | None:
    """SCIP's dual bound, or None while it is infinite (no root LP yet, or infeasible)."""
    lb = float(m.getDualbound())
    return None if m.isInfinity(abs(lb)) else lb


def _relative_gap(lb: float | None, ub: float | None) -> float | None:
    """(ub - lb) / lb, or None unless both exist and lb is positive."""
    if lb is None or ub is None or lb <= 0:
        return None
    return (ub - lb) / lb


Params = dict[str, float]


def _base_params() -> Params:
    # Wall, as ucp_valve.py's Gurobi; README item 4 (#441). Threads and the time
    # limit are the backend's.
    return {"timing/clocktype": 2}


@dataclass
class Solve:
    """One SCIP solve as both backends report it."""

    status: str  # PySCIPOpt's status names, which the driver writes too
    n_sols: int
    lb: float | None
    objective: float | None  # SCIP's, in the original space
    x: Values | None  # the best solution, None without one
    #: Driver only: (seconds since the run's start, SCIP objective) per incumbent
    #: with a finite objective, in time order; not yet a running best.
    trajectory: list[tuple[float, float]] = field(default_factory=list)
    #: Driver at N > 1 with a start: whether completesol completed it, else None.
    start_completed: bool | None = None


#: A backend's solve: model, parameters, a partial start or None, the deadline
#: on the backend's clock.
SolveFn = Callable[[Any, Params, Values | None, float], Solve]


def improving(points: list[tuple[float, float]]) -> list[list[float]]:
    """The step function of the running best: each point that beats every earlier one."""
    best = math.inf
    steps: list[list[float]] = []
    for t, obj in points:
        if obj < best:
            best = obj
            steps.append([t, obj])
    return steps


def solve_in_process(m: Any, params: Params, start: Values | None, deadline: float) -> Solve:
    """The wheel's SCIP, single-threaded, as #441's column ran it."""
    for name, value in params.items():
        m.setParam(name, value)
    m.setIntParam("parallel/maxnthreads", 1)
    m.setRealParam("limits/time", max(deadline - time.process_time(), 0.0))
    if start is not None:
        by_name = {var.name: var for var in m.getVars()}
        sol = m.createPartialSol()
        for name, value in start.items():
            m.setSolVal(sol, by_name[name], value)
        m.addSol(sol)
    m.optimize()
    n_sols = int(m.getNSols())
    return Solve(
        status=str(m.getStatus()),
        n_sols=n_sols,
        lb=_finite_bound(m),
        objective=float(m.getObjVal()) if n_sols else None,
        x=solution_values(m) if n_sols else None,
    )


#: SCIP's default `numerics/sumepsilon`.
SCIP_SUMEPSILON = 1.0e-6
#: The driver's pinned SCIP commit; a libscip reporting anything else is refused.
SCIP_COMMIT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "common", "scip_driver", "SCIP_COMMIT"
)
#: Seconds the driver may run past its --time-limit before it is killed: SCIP
#: checks its limit between steps, and reading and writing files take time too.
DRIVER_GRACE_S = 60.0
#: Where the root CMakeLists.txt puts the driver when built with -DCBLS_SCIP_DIR.
DEFAULT_DRIVER = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "build", "cbls_scip_traj"
)


def read_sol(path: str) -> Values:
    """A SCIP `.sol` file (zeros printed) as variable name -> value."""
    x: Values = {}
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith(("solution status:", "objective value:")):
                continue
            name, value = line.split()[:2]
            x[name] = float(value)
    return x


@dataclass
class Driver:
    """`cbls_scip_traj` at N threads; one subprocess per solve, files in `workdir`."""

    path: str
    threads: int
    workdir: str
    pinned: str
    origin: float = 0.0  # the method's start on time.monotonic(): trajectories count from it
    scip_version: str | None = None
    scip_git_hash: str | None = None
    calls: int = 0
    #: Opt-in (#442): a driver killed after it wrote its result, so in SCIPfree, counts
    #: as finished; off, any kill is an error.
    salvage_killed: bool = False
    killed_in_free: int = 0
    grace_s: float = DRIVER_GRACE_S

    def solve(self, m: Any, params: Params, start: Values | None, deadline: float) -> Solve:
        self.calls += 1
        base = os.path.join(self.workdir, f"solve{self.calls}")
        feastol = params.get("numerics/feastol", SCIP_SUMEPSILON)
        if feastol < SCIP_SUMEPSILON:
            # SCIP 11 refuses sumepsilon > feastol (set.c:681); 10 took it. Strict pwl only.
            params = {**params, "numerics/sumepsilon": feastol}
        m.writeProblem(base + ".cip", verbose=False)
        with open(base + ".set", "w") as fh:
            fh.writelines(f"{name} = {value!r}\n" for name, value in params.items())
        cmd = [self.path, base + ".cip", "--threads", str(self.threads), "--set", base + ".set"]
        cmd += ["--traj-out", base + ".jsonl", "--result-out", base + ".json"]
        cmd += ["--sol-out", base + ".sol"]
        if start is not None:
            with open(base + ".start.sol", "w") as fh:
                fh.writelines(f"{name} {value!r}\n" for name, value in start.items())
            cmd += ["--partial-sol", base + ".start.sol"]
        # Taken after the writes, so the budget counts them and the spawn.
        time_left = max(deadline - time.monotonic(), 0.0)
        cmd += ["--time-limit", repr(time_left)]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, check=False, timeout=time_left + self.grace_s
            )
        except subprocess.TimeoutExpired as e:
            # The driver renames its result into place before SCIPfree.
            if not (self.salvage_killed and os.path.exists(base + ".json")):
                raise RuntimeError(
                    f"cbls_scip_traj ran {e.timeout:.0f} s, killed: {base}.*\n"
                    f"stdout: {_tail(e.stdout)}\nstderr: {_tail(e.stderr)}"
                ) from e
            self.killed_in_free += 1
            print(f"cbls_scip_traj killed in SCIPfree, result kept: {base}.json", file=sys.stderr)
        else:
            if proc.returncode != 0:
                raise RuntimeError(
                    f"cbls_scip_traj exited {proc.returncode} ({base}.*)\n"
                    f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
                )
        with open(base + ".json") as fh:
            res = json.load(fh)
        # The driver refuses another libscip itself; this catches a stale binary.
        git_hash = str(res.get("scip_git_hash") or "")
        if not res.get("ok"):
            # A salvaged driver's exit code is lost; its retcode says what failed.
            raise RuntimeError(
                f"cbls_scip_traj failed, SCIP retcode {res.get('scip_retcode')} ({base}.*): {res}"
            )
        if not git_hash or not self.pinned.startswith(git_hash):
            raise RuntimeError(f"cbls_scip_traj ran SCIP {git_hash!r}, not {self.pinned}: {res}")
        self.scip_version, self.scip_git_hash = str(res["scip_version"]), git_hash
        with open(base + ".jsonl") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        n_sols = int(res["n_sols"])
        solve = Solve(
            status=str(res["status"]),
            n_sols=n_sols,
            lb=res["dual_bound"],
            objective=res["primal_bound"],
            x=read_sol(base + ".sol") if n_sols else None,
            trajectory=[
                (r["mono_s"] - self.origin, r["objective"])
                for r in records
                if r["objective"] is not None
            ],
            start_completed=res.get("start_completed"),
        )
        # Parsed: free the files now, so a long run's model copies do not pile up.
        for path in glob.glob(glob.escape(base) + ".*"):
            os.remove(path)
        return solve


def _tail(out: str | bytes | None) -> str:
    """The last 2000 characters of a killed child's captured output."""
    if isinstance(out, bytes):
        out = out.decode(errors="replace")
    return (out or "")[-2000:]


@dataclass
class Backend:
    """How the methods solve and keep time: the wheel on CPU, or the driver on wall."""

    name: str
    solve: SolveFn
    clock: Callable[[], float]
    time_key: str  # the record's elapsed-time field
    threads: int = 1
    driver: Driver | None = None

    def start(self) -> float:
        """The method's start on this clock; the driver's trajectories count from it."""
        now = self.clock()
        if self.driver is not None:
            self.driver.origin = now
        return now


IN_PROCESS = Backend("pyscipopt", solve_in_process, time.process_time, "cpu_s")


#: The variables ucp_valve.py passes from one iteration's solution to the next.
WARM_VARS = ("y", "hot", "cold", "on", "off")


def _pwl_model(
    pr: Problem,
    prev: dict[tuple[int, int], list[float]],
    n: int,
) -> tuple[Any, UcpVars, int]:
    """One iteration's MIP: the base model and the interpolated cost."""
    m, v = ucp_base(pr)
    brk = {
        (u, t): expand_brkpts(unit, prev[u, t], n)
        for u, unit in enumerate(pr.units)
        for t in range(1, pr.horizon + 1)
    }
    n_brk = ucp_lin_obj(m, v, pr, brk)
    return m, v, n_brk


def _pwl_params(strict: bool, warm: bool) -> Params:
    params = _base_params()
    if strict:
        params.update({"limits/gap": 1.0e-12, "limits/absgap": 1.0e-12, "numerics/feastol": 1.0e-9})
    else:
        params.update({"limits/gap": GUROBI_MIPGAP, "limits/absgap": GUROBI_MIPGAPABS})
    if warm:
        # The start fixes y, hot, cold, on and off only; every breakpoint weight,
        # p, fuel and startup is unknown, which is far past completesol's default
        # 0.85 unknown rate, so SCIP would drop the start without a word.
        params["heuristics/completesol/maxunknownrate"] = 1.0
    return params


def _refine(x: Values, v: UcpVars, pr: Problem, prev: dict[tuple[int, int], list[float]]) -> bool:
    """Record each committed dispatch that lands in an unrefined valve interval."""
    new = False
    for u, unit in enumerate(pr.units):
        for t in range(1, pr.horizon + 1):
            if x[v.y[u, t].name] <= 0.5:
                continue
            pt = float(x[v.p[u, t].name])
            if has_added_p(unit, prev[u, t], pt):
                if any(abs(pp - pt) <= 1.0e-6 for pp in prev[u, t]):
                    # ucp_valve.py asserts this before appending.
                    raise RuntimeError(f"unit {u} period {t}: dispatch {pt} repeats a point")
                new = True
                prev[u, t].append(pt)
                prev[u, t].sort()
    return new


def _start_fate(backend: Backend, solve: Solve) -> dict[str, Any]:
    """Whether SCIP completed the warm start: the driver reports it, the wheel cannot."""
    return {} if backend.driver is None else {"start_completed": solve.start_completed}


def _trajectory_step(trajectory: list[list[float]], elapsed: float, ub: float) -> None:
    """A step of the running best, if `ub` improves on it."""
    if not trajectory or ub < trajectory[-1][1]:
        trajectory.append([elapsed, ub])


def _wall_step(
    trajectory: list[list[float]],
    iteration: dict[str, Any],
    solve: Solve,
    budget: float,
    ub: float,
) -> None:
    """On the wall budget, a UB whose solution SCIP found only after it (its limit's
    overshoot) is not the run's (#442); the wheel's CPU budget is #441's and unchanged.
    Found, not returned: the exit, SCIPfree and the re-check come after the limit, so
    the trajectory point is stamped at the find too. With no logged incumbent the find
    time is unknown and the return time stands in: such a UB is not shown in budget."""
    steps = improving(solve.trajectory)
    elapsed = iteration["wall_s"]
    found = min(steps[-1][0], elapsed) if steps else elapsed
    iteration["found_s"] = found
    iteration["in_budget"] = found <= budget
    if found <= budget:
        _trajectory_step(trajectory, found, ub)


def solve_pwl(
    pr: Problem, budget: float, warm_start: bool = True, backend: Backend = IN_PROCESS
) -> dict[str, Any]:
    """Pedroso's Algorithm 1, the loop of ucp_valve.py's `__main__`."""
    clock, time_key = backend.clock, backend.time_key
    start = backend.start()
    prev: dict[tuple[int, int], list[float]] = {
        (u, t): [] for u in range(len(pr.units)) for t in range(1, pr.horizon + 1)
    }
    n = N_INIT
    strict = False
    incumbent: Values | None = None
    iterations: list[dict[str, Any]] = []
    trajectory: list[list[float]] = []  # the running best true-cost UB, strictly falling
    converged = False
    out_of_budget = False
    deadline = start + budget
    while True:
        m, v, n_brk = _pwl_model(pr, prev, n)
        warm = incumbent if warm_start else None
        if clock() >= deadline:
            out_of_budget = True
            break
        solve = backend.solve(m, _pwl_params(strict, warm is not None), warm, deadline)
        status = solve.status
        if solve.x is None:
            iterations.append(
                {"n": n, "breakpoints": n_brk, "status": status, "lb": solve.lb}
                | _start_fate(backend, solve)
            )
            # A proven-infeasible MIP did not run out of CPU.
            out_of_budget = status != "infeasible"
            break
        x = solve.x
        lb = solve.lb
        ub = true_cost(x, v, pr)
        gap = _relative_gap(lb, ub)
        error = gap if gap is not None else math.inf
        check = check_solution(x, v, pr)
        elapsed = clock() - start
        iterations.append(
            {
                "n": n,
                "breakpoints": n_brk,
                "status": status,
                "strict": strict,
                "lb": lb,
                "ub": ub,
                "error": error,
                "check": check,
                time_key: elapsed,
            }
            | _start_fate(backend, solve)
        )
        if backend.driver is not None:
            _wall_step(trajectory, iterations[-1], solve, budget, ub)
        print(
            f"  n={n} brk={n_brk} {status} LB={lb} UB={ub:.4f} err={error:.3e} "
            f"{time_key}={elapsed:.1f}",
            file=sys.stderr,
        )
        if strict and error < EPS:
            converged = True
            break
        gap_param = 1.0e-12 if strict else GUROBI_MIPGAP
        if error < EPS or error < gap_param:
            strict = True
        if status not in SOLVED:
            out_of_budget = True
            break
        incumbent = {
            var.name: x[var.name] for name in WARM_VARS for var in getattr(v, name).values()
        }
        if not _refine(x, v, pr, prev):
            n *= 2
            strict = True

    result = _pwl_result(iterations, converged, out_of_budget)
    result[time_key] = clock() - start
    if backend.driver is not None:
        result["trajectory"] = trajectory
    return result


def _pwl_result(
    iterations: list[dict[str, Any]], converged: bool, out_of_budget: bool
) -> dict[str, Any]:
    """The run's bounds and status from its iterations."""
    # Every iteration's LB is SCIP's at its stop, late or not (README item 8).
    with_lb = [it for it in iterations if it.get("lb") is not None]
    with_ub = [
        i
        for i, it in enumerate(iterations)
        if it.get("ub") is not None and it.get("in_budget", True)
    ]
    best = min(with_ub, key=lambda i: iterations[i]["ub"]) if with_ub else None
    any_ub = any(it.get("ub") is not None for it in iterations)
    last_status = iterations[-1]["status"] if iterations else None
    # Converging after the wall budget is running out of it (#442).
    late = bool(iterations) and not iterations[-1].get("in_budget", True)
    converged = converged and not late
    result: dict[str, Any] = {
        "iterations": iterations,
        "converged": converged,
        "out_of_budget": out_of_budget or late,
        "last_lb": iterations[-1].get("lb") if iterations else None,
        "last_ub": iterations[-1].get("ub") if iterations else None,
        # While every unit has c <= e*f^2/2, each iteration's LB and UB bound the
        # true optimum, so the best of each is the result; the last iteration may
        # have had too little CPU to say much. `check` is the best UB's solution's.
        "lb": max((it["lb"] for it in with_lb), default=None),
        "ub": iterations[best]["ub"] if best is not None else None,
        # `infeasible` only when no iteration found a UB: a later infeasible MIP
        # after one did (numerics in strict mode) is a stop, not an infeasible run.
        "status": (
            "converged"
            if converged
            else ("stopped: infeasible MIP after a UB" if any_ub else "infeasible")
            if last_status == "infeasible"
            else "time limit reached"
        ),
        "last_scip_status": last_status,
    }
    if best is not None:
        result["check"] = iterations[best]["check"]
    return result


def solve_exact(
    pr: Problem, budget: float, gap: float, backend: Backend = IN_PROCESS
) -> dict[str, Any]:
    """One SCIP MINLP solve with the true valve-point term."""
    clock = backend.clock
    start = backend.start()
    m, v = ucp_base(pr)
    ucp_exact_obj(m, v, pr)
    params = _base_params()
    params["limits/gap"] = gap
    solve = backend.solve(m, params, None, start + budget)
    result: dict[str, Any] = {"status": solve.status, backend.time_key: clock() - start}
    steps = improving(solve.trajectory)
    if backend.driver is not None:
        # SCIP's objective per incumbent: fuel and valve are >= relaxations, so it
        # bounds that incumbent's true cost from above, and the final `ub` (the
        # best solution's true cost) may sit below the last point. Only points
        # within the wall budget count; later ones (SCIP's overshoot) are kept apart.
        result["trajectory"] = [p for p in steps if p[0] <= budget]
        result["trajectory_after_budget"] = [p for p in steps if p[0] > budget]
    if solve.x is None:
        result.update(lb=solve.lb, ub=None)
        return result
    late = {
        "ub": true_cost(solve.x, v, pr),
        "scip_objective": solve.objective,
        "check": check_solution(solve.x, v, pr),
    }
    result["lb"] = solve.lb
    if not result.get("trajectory_after_budget"):
        result.update(late)
        return result
    # The best solution came after the budget (#442): it is kept apart as
    # `*_after_budget`, and `ub` is the last in-budget incumbent's SCIP objective,
    # a bound on its true cost; the driver keeps only the best solution, so that
    # incumbent is not re-checked.
    # SCIP's status is the late one too: the run itself stopped on its budget.
    result.update({f"{k}_after_budget": val for k, val in late.items()})
    result["status_after_budget"] = result["status"]
    result["status"] = "timelimit"
    result["ub"] = result["trajectory"][-1][1] if result["trajectory"] else None
    result["ub_is_scip_objective"] = True
    return result


def _budget_and_threads(ap: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[float, int]:
    """The backend's budget and threads; each refuses the other's flags."""
    if args.backend == "pyscipopt":
        driver_only = (args.time_limit, args.threads, args.driver_grace)
        if any(a is not None for a in driver_only) or args.salvage_killed:
            ap.error(
                "--time-limit, --threads, --driver-grace and --salvage-killed are the driver's; "
                "pyscipopt takes --cpu"
            )
        return (3600.0 if args.cpu is None else args.cpu), 1
    if args.cpu is not None:
        ap.error("--cpu is pyscipopt's; the driver takes --time-limit, a wall budget")
    threads = 10 if args.threads is None else args.threads
    if threads < 1:
        ap.error(f"--threads {threads} is not positive")
    budget = 3600.0 if args.time_limit is None else args.time_limit
    if not (math.isfinite(budget) and budget > 0):
        ap.error(f"--time-limit {budget} is not a positive number")
    if args.driver_grace is not None and not (
        math.isfinite(args.driver_grace) and args.driver_grace >= 0
    ):
        ap.error(f"--driver-grace {args.driver_grace} is not a non-negative number")
    if not os.access(args.driver, os.X_OK):
        ap.error(f"no cbls_scip_traj at {args.driver}: benchmarks/common/scip_driver/README.md")
    return budget, threads


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--instance", required=True, choices=sorted(BASES))
    ap.add_argument("--periods", type=int, required=True)
    ap.add_argument("--method", required=True, choices=["pwl", "exact"])
    ap.add_argument(
        "--backend",
        choices=["pyscipopt", "driver"],
        default="pyscipopt",
        help="pyscipopt: the wheel's SCIP in process (default); driver: cbls_scip_traj (#442)",
    )
    ap.add_argument(
        "--cpu",
        type=float,
        help="pyscipopt: process CPU budget in seconds; each SCIP solve gets what is left as a "
        "wall limit (default 3600)",
    )
    ap.add_argument(
        "--time-limit",
        type=float,
        help="driver: wall budget in seconds from the start, writing and reading included "
        "(default 3600)",
    )
    ap.add_argument("--threads", type=int, help="driver: SCIP's threads (default 10)")
    ap.add_argument(
        "--driver",
        default=os.environ.get("CBLS_SCIP_DRIVER", DEFAULT_DRIVER),
        help="driver: the cbls_scip_traj binary (default $CBLS_SCIP_DRIVER, else build/)",
    )
    ap.add_argument(
        "--exact-gap",
        type=float,
        default=EPS,
        help="SCIP's relative gap limit for --method exact (default ucp_valve.py's EPS)",
    )
    ap.add_argument(
        "--no-warm-start",
        action="store_true",
        help="pwl: do not pass the last solution on (ucp_valve.py's own start is inert)",
    )
    ap.add_argument(
        "--salvage-killed",
        action="store_true",
        help="driver: a driver killed past its limit and grace after writing its result "
        "(so in SCIPfree) counts as finished; default: an error",
    )
    ap.add_argument(
        "--driver-grace",
        type=float,
        help=f"driver: seconds past its limit before it is killed (default {DRIVER_GRACE_S:g})",
    )
    ap.add_argument("--out", help="append the result as one JSON line")
    args = ap.parse_args()
    budget, threads = _budget_and_threads(ap, args)

    import pyscipopt

    pr = load_problem(args.instance, args.periods)
    bad = [i for i, u in enumerate(pr.units) if not concave_between_valves(u)]
    backend, driver = IN_PROCESS, None
    if args.backend == "driver":
        with open(SCIP_COMMIT_FILE) as fh:
            pinned = fh.read().strip()
        # Removed only after a clean run: a failed one leaves its files to inspect.
        driver = Driver(args.driver, threads, tempfile.mkdtemp(prefix="ucp_valve_scip_"), pinned)
        driver.salvage_killed = args.salvage_killed
        if args.driver_grace is not None:
            driver.grace_s = args.driver_grace
        backend = Backend("driver", driver.solve, time.monotonic, "wall_s", threads, driver)
    try:
        if args.method == "pwl":
            res = solve_pwl(pr, budget, warm_start=not args.no_warm_start, backend=backend)
        else:
            res = solve_exact(pr, budget, args.exact_gap, backend)
    except BaseException:
        if driver is not None:
            print(f"kept {driver.workdir}", file=sys.stderr)
        raise
    if driver is not None:
        shutil.rmtree(driver.workdir)
    lb, ub = res.get("lb"), res.get("ub")
    record = {
        "instance": args.instance,
        "periods": args.periods,
        "method": args.method,
        **({"cpu_limit_s": budget} if driver is None else {"time_limit_s": budget}),
        "scip_clock": "wall",  # timing/clocktype 2; score.py refuses any other (#441)
        "lb": lb,
        "ub": ub,
        "gap_pct": (100.0 * g) if (g := _relative_gap(lb, ub)) is not None else None,
        "concavity_violated_units": bad,
        "backend": args.backend,
        "threads": threads,
        "pyscipopt": importlib.metadata.version("pyscipopt"),
        "scip": str(pyscipopt.Model().version()) if driver is None else driver.scip_version,
        "scip_git_hash": None if driver is None else driver.scip_git_hash,
        # 0 when the budget ran out before the first solve: scip/scip_git_hash are null then.
        **(
            {}
            if driver is None
            else {"driver_solves": driver.calls, "driver_killed_in_free": driver.killed_in_free}
        ),
        **{k: val for k, val in res.items() if k not in ("lb", "ub")},
    }
    shown = ("instance", "periods", "method", "lb", "ub", "gap_pct", "status", backend.time_key)
    print(json.dumps({k: record[k] for k in shown}))
    if args.out:
        with open(args.out, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    if bad and args.method == "pwl":
        print(
            f"warning: units {bad} break c <= e*f^2/2; the pwl LB is not a bound", file=sys.stderr
        )
    if res.get("check"):
        print(f"solution check failed: {res['check'][:5]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
