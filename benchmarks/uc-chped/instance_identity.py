"""Instance-identity check for UC-CHPED against Pedroso, Kubo & Viana (2014) (#148).

Answers "do we solve the instances Table 2's bounds describe?" three ways, none
of which trusts the others:

1. **Data diff.** Loads the authors' GPL `ucp_data.py` (pass its directory with
   `--upstream-dir`; it is not vendored, see FIDELITY.md section 7 for the URL and
   hash) and compares every field of `ucp13`/`ucp40` with
   `benchmarks/instances/uc-chped/data.py`.
2. **Re-pricing the authors' own solutions.** The same directory may carry the
   authors' `RESULTS/ucp{13,40}-{T}.txt` logs. Their final `prod levels:` block
   is re-priced with OUR coefficients and checked against OUR constraints; the
   result must reproduce the Table 2 upper bound to the logs' 3-decimal print
   rounding.
3. **Exact solves.** A SCIP MINLP with the true `|d*sin(e*(Pmin-P))|` term (no
   linearisation; SCIP's spatial branch-and-bound handles `sin` globally), plus
   the existing piecewise-linear reference (`benchmarks/chped/reference_solve.py`)
   at a fine segment count.

The MINLP follows the authors' `ucp_valve.py` semantics, which differ from ours
in one place that matters here: demand is an EQUALITY there and `>=` in our model
and in the SCIP reference. With a non-monotone valve-point cost the two optima can
differ, so both are solved (`--demand eq|ge|both`). Startup cost is priced cold
for every startup, which is exactly what both the authors' code and our model do
on these instances because every unit has `t_cold < min_off` (asserted below).

Usage (from the repository root):
    .venv/bin/python benchmarks/uc-chped/instance_identity.py \\
        --upstream-dir DIR --time-limit 600 --pwl-segments 200
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
import re
import sys
import time
import types
from dataclasses import dataclass
from typing import Any

Instance = dict[str, Any]

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, "..", ".."))

#: The three Table 2 rows whose LB equals UB, i.e. proven optima, at the two
#: decimals arXiv:1404.4944 prints (the repository's tables round them).
PROVEN_OPTIMA: dict[tuple[str, int], float] = {
    ("ucp13", 1): 11701.28,
    ("ucp13", 3): 38849.84,
    ("ucp40", 1): 55644.79,
}

#: Fields compared by the data diff, in our naming; the authors call d/e `e`/`f`.
_FIELDS = (
    "a",
    "b",
    "c",
    "d",
    "e",
    "P_min",
    "P_max",
    "y_prev",
    "t_cold",
    "n_init",
    "min_on",
    "min_off",
    "a_hot",
    "a_cold",
    "demand",
    "reserve",
)
_UPSTREAM_NAME = {"d": "e", "e": "f", "P_min": "p_min", "P_max": "p_max"}

#: sha256 of the upstream files this check was run against (FIDELITY.md 7.1),
#: keyed by path relative to `--upstream-dir`. A revised upstream file would
#: silently re-define the yardstick, so a mismatch is refused, not reported.
UPSTREAM_SHA256: dict[str, str] = {
    "ucp_data.py": "3d5b8f078550fea50c98a42389257db9f7cda891190a9676b43b29fcb18934c2",
    "RESULTS/ucp13-1.txt": "96660c8a6ede9671a32e29b22def1a5b0f04ee99835adc9198b338eaa19eea86",
    "RESULTS/ucp13-3.txt": "bc35a0d0f259b53a6096539993e9f06357b20a824128c85548fba2fa91da3920",
    "RESULTS/ucp40-1.txt": "e74c19e14cfc7637529605aa841d09c08d2ff0a5006e6c607b49535a9e8a7706",
}


def verified_upstream_path(upstream_dir: str, rel: str) -> str:
    """`upstream_dir/rel`, after checking it is the file the audit recorded."""
    path = os.path.join(upstream_dir, rel)
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    if digest != UPSTREAM_SHA256[rel]:
        raise SystemExit(
            f"{path}: sha256 {digest} is not the recorded {UPSTREAM_SHA256[rel]}; "
            "refusing to compare against a different upstream file"
        )
    return path


def _load_module(name: str, path: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve their module through this
    spec.loader.exec_module(module)
    return module


def load_ours() -> dict[str, Instance]:
    """The 13- and 40-unit instances as this repository builds them."""
    data = _load_module(
        "uc_chped_data", os.path.join(_REPO, "benchmarks", "instances", "uc-chped", "data.py")
    )
    return {"ucp13": data.UCP_13UNIT, "ucp40": data.UCP_40UNIT}


def _multidict(table: dict[Any, list[Any]]) -> list[Any]:
    """Stand-in for `gurobipy.multidict`: keys, then one dict per column."""
    keys = list(table)
    width = len(next(iter(table.values())))
    return [keys] + [{k: table[k][j] for k in keys} for j in range(width)]


def load_upstream(upstream_dir: str) -> dict[str, Instance]:
    """The authors' `ucp13(24)`/`ucp40(24)`, converted to our list layout.

    `ucp_data.py` does `from gurobipy import *` only for `multidict`, so a stub
    module supplies it and no Gurobi installation is needed.
    """
    stub = types.ModuleType("gurobipy")
    stub.multidict = _multidict  # type: ignore[attr-defined]
    stub.__all__ = ["multidict"]  # type: ignore[attr-defined]
    sys.modules.setdefault("gurobipy", stub)
    up = _load_module("pedroso_ucp_data", verified_upstream_path(upstream_dir, "ucp_data.py"))
    out: dict[str, Instance] = {}
    for name in ("ucp13", "ucp40"):
        fields = getattr(up, name)(24)
        units, periods = fields[0], fields[1]
        keys = [
            *("a", "b", "c", "e", "f", "p_min", "p_max", "y_prev", "t_cold", "n_init"),
            *("min_on", "min_off", "a_hot", "a_cold", "dem", "res"),
        ]
        raw = dict(zip(keys, fields[2:], strict=True))
        inst: Instance = {
            "name": f"{name}-upstream",
            "n_units": len(units),
            "n_periods": periods,
            "known_bounds": {},
        }
        for ours in _FIELDS:
            theirs = {"demand": "dem", "reserve": "res"}.get(ours, _UPSTREAM_NAME.get(ours, ours))
            column = raw[theirs]
            index = range(1, periods + 1) if ours in ("demand", "reserve") else sorted(units)
            # Keep the authors' Python types (int periods/flags, int or float
            # coefficients) so period counts stay usable as range bounds.
            inst[ours] = [column[k] for k in index]
        out[name] = inst
    return out


def diff_instances(ours: Instance, theirs: Instance) -> list[str]:
    """Every field where the two instances disagree (exact float comparison)."""
    problems = []
    for field in _FIELDS:
        a = [float(x) for x in ours[field]]
        b = [float(x) for x in theirs[field]]
        if len(a) != len(b):
            problems.append(f"{field}: length {len(a)} vs {len(b)}")
            continue
        bad = [(i + 1, x, y) for i, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
        if bad:
            problems.append(f"{field} differs at (1-indexed, ours, theirs) {bad}")
    return problems


def fuel(inst: Instance, u: int, p: float) -> float:
    """True operating cost of unit `u` at output `p` (committed)."""
    return float(
        inst["a"][u]
        + inst["b"][u] * p
        + inst["c"][u] * p * p
        + abs(inst["d"][u] * math.sin(inst["e"][u] * (inst["P_min"][u] - p)))
    )


@dataclass(frozen=True)
class Schedule:
    """A commitment/dispatch schedule over the first `len(y)` periods."""

    y: list[list[int]]  # y[t][u]
    p: list[list[float]]  # p[t][u]


def price(inst: Instance, s: Schedule) -> tuple[float, list[str]]:
    """Total cost (all startups cold) and the constraint violations of `s`."""
    n = inst["n_units"]
    cost = 0.0
    errors: list[str] = []
    for t, (yt, pt) in enumerate(zip(s.y, s.p, strict=True)):
        for u in range(n):
            before = inst["y_prev"][u] if t == 0 else s.y[t - 1][u]
            if yt[u] and not before:
                cost += inst["a_cold"][u]
            if yt[u]:
                cost += fuel(inst, u, pt[u])
                if not inst["P_min"][u] - 1e-3 <= pt[u] <= inst["P_max"][u] + 1e-3:
                    errors.append(f"t={t} u={u} p={pt[u]} outside [Pmin,Pmax]")
            elif abs(pt[u]) > 1e-3:
                errors.append(f"t={t} u={u} off but p={pt[u]}")
        # 3-decimal print rounding over up to 40 units
        if abs(sum(pt) - inst["demand"][t]) > 0.05:
            errors.append(f"t={t} supply {sum(pt):.3f} != demand {inst['demand'][t]}")
        cap = sum(inst["P_max"][u] for u in range(n) if yt[u])
        if cap < inst["demand"][t] + inst["reserve"][t]:
            errors.append(f"t={t} reserve short")
    return cost, errors + min_up_down_violations(inst, s)


def min_up_down_violations(inst: Instance, s: Schedule) -> list[str]:
    """Rolling min up/down windows over the horizon.

    `n_init >= min_on/min_off` on these instances (asserted by
    `_assert_all_starts_cold`), so the pre-horizon state forces nothing.
    """
    errors: list[str] = []
    for u in range(inst["n_units"]):
        col = [inst["y_prev"][u]] + [s.y[t][u] for t in range(len(s.y))]
        for t in range(1, len(col)):
            if col[t] != col[t - 1]:
                hold = inst["min_on"][u] if col[t] else inst["min_off"][u]
                if any(col[k] != col[t] for k in range(t, min(t + hold, len(col)))):
                    errors.append(f"u={u} switch at t={t - 1} breaks its {hold}-period minimum")
    return errors


def parse_upstream_solution(path: str, n_units: int) -> Schedule:
    """The final `printsol` block of one of the authors' RESULTS logs."""
    with open(path) as fh:
        text = fh.read()
    tail = text[text.rindex("\nObj:") :]
    y_block = tail[tail.index("y, on, off:") : tail.index("startup:\n")]
    p_block = tail[tail.index("prod levels:") :]
    y: list[list[int]] = []
    for line in y_block.splitlines()[1:]:
        cols = line.split("\t")
        if len(cols) >= 3 and cols[0].strip().isdigit() and int(cols[0]) > 0:
            y.append([int(v) for v in cols[1].split()])
    p: list[list[float]] = []
    for line in p_block.splitlines()[1:]:
        m = re.match(r"\s*(\d+)\s*\t(.*)\t\*", line)
        if m:
            p.append([float(v) for v in m.group(2).split()])
    if not y or len(y) != len(p) or any(len(r) != n_units for r in y + p):
        raise SystemExit(f"could not parse a {n_units}-unit schedule from {path}")
    return Schedule(y=y, p=p)


def _assert_all_starts_cold(inst: Instance) -> None:
    """Precondition under which every startup is cold in both codes."""
    for u in range(inst["n_units"]):
        assert inst["t_cold"][u] < inst["min_off"][u], u
        assert inst["y_prev"][u] == 1 or inst["n_init"][u] > inst["t_cold"][u], u
        assert inst["n_init"][u] >= max(inst["min_on"][u], inst["min_off"][u]), u


def solve_minlp(
    inst: Instance, periods: int, equality: bool, time_limit: float
) -> tuple[float, float, float, str]:
    """Global SCIP solve with the true valve-point term: (obj, dual bound, s, status).

    Fuel when off is forced to zero without a bilinear term: `p = 0` when off, and
    the valve auxiliary `v >= |d sin(e(Pmin-p))| - d(1-y)` is free to reach 0
    because `|d sin(.)| <= d`. When on, `v >= |...|` is tight at the optimum.
    """
    from pyscipopt import Model, quicksum, sin

    _assert_all_starts_cold(inst)
    n = inst["n_units"]
    m = Model(f"{inst['name']}-{periods}p-minlp")
    m.hideOutput()
    m.setRealParam("limits/time", time_limit)
    m.setRealParam("limits/gap", 0.0)
    m.setIntParam("parallel/maxnthreads", 1)
    y = {(u, t): m.addVar(vtype="B") for u in range(n) for t in range(periods)}
    p = {(u, t): m.addVar(lb=0.0, ub=inst["P_max"][u]) for u in range(n) for t in range(periods)}
    on = {(u, t): m.addVar(vtype="B") for u in range(n) for t in range(periods)}
    off = {(u, t): m.addVar(vtype="B") for u in range(n) for t in range(periods)}
    obj: list[Any] = []
    for t in range(periods):
        supply = quicksum(p[u, t] for u in range(n))
        if equality:
            m.addCons(supply == inst["demand"][t])
        else:
            m.addCons(supply >= inst["demand"][t])
        m.addCons(
            quicksum(inst["P_max"][u] * y[u, t] for u in range(n))
            >= inst["demand"][t] + inst["reserve"][t]
        )
        for u in range(n):
            d, e, pmin = inst["d"][u], inst["e"][u], inst["P_min"][u]
            m.addCons(p[u, t] >= pmin * y[u, t])
            m.addCons(p[u, t] <= inst["P_max"][u] * y[u, t])
            before = inst["y_prev"][u] if t == 0 else y[u, t - 1]
            m.addCons(on[u, t] - off[u, t] == y[u, t] - before)
            m.addCons(on[u, t] + off[u, t] <= 1)
            lo_on = max(0, t - inst["min_on"][u] + 1)
            m.addCons(quicksum(on[u, i] for i in range(lo_on, t + 1)) <= y[u, t])
            lo_off = max(0, t - inst["min_off"][u] + 1)
            m.addCons(quicksum(off[u, i] for i in range(lo_off, t + 1)) <= 1 - y[u, t])
            v = m.addVar(lb=0.0)
            valve = d * sin(e * (pmin - p[u, t]))
            m.addCons(v >= valve - d * (1 - y[u, t]))
            m.addCons(v >= -valve - d * (1 - y[u, t]))
            obj += [
                inst["a"][u] * y[u, t],
                inst["b"][u] * p[u, t],
                inst["c"][u] * p[u, t] * p[u, t],
                v,
                inst["a_cold"][u] * on[u, t],
            ]
    z = m.addVar(lb=0.0)
    m.addCons(z >= quicksum(obj))
    m.setObjective(z)
    m.optimize()
    best = m.getObjVal() if m.getNSols() > 0 else float("inf")
    return best, m.getDualbound(), m.getSolvingTime(), m.getStatus()


def pwl_envelope(inst: Instance, segments: int, samples: int = 4000) -> float:
    """Σ_u max |chord interpolation - F_u| on [Pmin, Pmax]: the reference's PWL error.

    `reference_solve.py` interpolates F_u between equally spaced breakpoints, so
    its objective is within this sum (per period) of the true cost of its own
    dispatch. Sampled densely; a bound for the comparison, not a proof.
    """
    total = 0.0
    for u in range(inst["n_units"]):
        lo, hi = inst["P_min"][u], inst["P_max"][u]
        h = (hi - lo) / segments
        worst = 0.0
        for k in range(segments):
            x0, x1 = lo + k * h, lo + (k + 1) * h
            f0, f1 = fuel(inst, u, x0), fuel(inst, u, x1)
            for j in range(1, samples // segments + 2):
                x = x0 + (x1 - x0) * j / (samples // segments + 2)
                chord = f0 + (f1 - f0) * (x - x0) / (x1 - x0)
                worst = max(worst, abs(chord - fuel(inst, u, x)))
        total += worst
    return total


def _reference_solver() -> Any:
    """`reference_solve.py`, loaded for its UC solver alone.

    It imports scipy at module scope for its dispatch-only CHPED path, which the
    `benchmarks` extra does not install; `solve_uc_scip` never touches it, so a
    stub stands in when scipy is absent rather than adding a dependency.
    """
    sys.path.insert(0, os.path.join(_REPO, "benchmarks", "chped"))
    if importlib.util.find_spec("scipy") is None:
        optimize = types.ModuleType("scipy.optimize")
        optimize.LinearConstraint = None  # type: ignore[attr-defined]
        optimize.differential_evolution = None  # type: ignore[attr-defined]
        sys.modules.setdefault("scipy", types.ModuleType("scipy"))
        sys.modules.setdefault("scipy.optimize", optimize)
    return _load_module(
        "reference_solve", os.path.join(_REPO, "benchmarks", "chped", "reference_solve.py")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--upstream-dir", help="directory holding ucp_data.py and RESULTS/")
    ap.add_argument("--time-limit", type=float, default=600.0, help="per-solve cap (s)")
    ap.add_argument("--pwl-segments", type=int, default=200)
    ap.add_argument("--demand", choices=("eq", "ge", "both"), default="both")
    ap.add_argument(
        "--data",
        choices=("ours", "upstream"),
        default="ours",
        help="solve this repository's instances, or the authors' (needs --upstream-dir)",
    )
    ap.add_argument("--cases", default="ucp13-1,ucp13-3,ucp40-1", help="comma-separated")
    ap.add_argument("--skip-solves", action="store_true")
    args = ap.parse_args()

    ours = load_ours()
    if args.upstream_dir:
        theirs = load_upstream(args.upstream_dir)
        for name in ("ucp13", "ucp40"):
            problems = diff_instances(ours[name], theirs[name])
            print(f"[data] {name}: {problems or 'IDENTICAL'}")

        for (name, periods), published in PROVEN_OPTIMA.items():
            rel = f"RESULTS/{name}-{periods}.txt"
            if not os.path.exists(os.path.join(args.upstream_dir, rel)):
                continue
            log = verified_upstream_path(args.upstream_dir, rel)
            sched = parse_upstream_solution(log, ours[name]["n_units"])
            cost, errors = price(ours[name], sched)
            print(
                f"[reprice] {name}-{periods}p authors' schedule at our data: {cost:.3f} "
                f"(published {published}, diff {100 * (cost - published) / published:+.4f}%) "
                f"violations={errors or 'none'}"
            )
    if args.skip_solves:
        return

    if args.data == "upstream" and not args.upstream_dir:
        raise SystemExit("--data upstream needs --upstream-dir")
    source = load_upstream(args.upstream_dir) if args.data == "upstream" else ours
    modes = {"eq": [True], "ge": [False], "both": [True, False]}[args.demand]
    ref = _reference_solver()
    wanted = set(args.cases.split(","))
    for (name, periods), published in PROVEN_OPTIMA.items():
        if f"{name}-{periods}" not in wanted:
            continue
        inst = source[name]
        for equality in modes:
            t0 = time.time()
            best, bound, secs, status = solve_minlp(inst, periods, equality, args.time_limit)
            print(
                f"[minlp] {name}-{periods}p demand {'==' if equality else '>='}: "
                f"obj {best:.3f} bound {bound:.3f} status {status} {secs:.1f}s "
                f"(wall {time.time() - t0:.1f}s) vs {published}: "
                f"{100 * (best - published) / published:+.4f}%"
            )
        sub = ref.make_subinstance(inst, periods)
        obj, secs, gap = ref.solve_uc_scip(sub, args.time_limit, args.pwl_segments)
        env = periods * pwl_envelope(inst, args.pwl_segments)
        print(
            f"[pwl] {name}-{periods}p reference_solve {args.pwl_segments} segments: obj {obj:.3f} "
            f"gap {gap:.2e} {secs:.1f}s envelope <= {env:.3f} vs {published}: "
            f"{100 * (obj - published) / published:+.4f}%"
        )


if __name__ == "__main__":
    main()
