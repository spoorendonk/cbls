"""Independent checker for UC-CHPED solutions (#445), against Pedroso 2014.

Written once from the source formulation (FIDELITY.md section 1), not from
cbls's model: demand with equality, spinning reserve on committed capacity,
dispatch limits, rolling-window min up/down with the initial conditions, and
fuel cost a + b p + c p^2 + |d sin(e (Pmin - p))| plus hot/cold startups.
The #445 original (sha256 5ca8285a...0a2b3c, #456) split into one function per
check and `T` renamed for the lint gate, the arithmetic unchanged (#462).

usage: check_solutions.py INSTANCE_DIR SOLUTIONS.jsonl [...]
Each solution line: {"instance", "periods", "seed", "objective", "y", "p"},
as `cbls_uc_chped --solution-out` writes them.
Prints one line per solution: verdict, recomputed cost, |reported - recomputed|.
Exit 1 if any solution fails, or if there is none to check.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

TOL = 1e-6  # absolute, on demand balance and dispatch limits
RESERVE_TOL = 1e-6


def base_file(inst_dir: Path, name: str) -> Path:
    """ucp13-6p -> ucp13.jsonl; ucp100-48p has its own file."""
    p = inst_dir / f"{name}.jsonl"
    if p.exists():
        return p
    stem = name.rsplit("-", 1)[0]
    return inst_dir / f"{stem}.jsonl"


def check_bounds(
    inst: dict[str, Any], y: list[list[int]], p: list[list[float]], horizon: int
) -> list[str]:
    err: list[str] = []
    for i in range(len(y)):
        for t in range(horizon):
            if y[i][t] not in (0, 1):
                err.append(f"y[{i}][{t}] not binary")
            lo = inst["P_min"][i] * y[i][t]
            hi = inst["P_max"][i] * y[i][t]
            if not (lo - TOL <= p[i][t] <= hi + TOL):
                err.append(f"p[{i}][{t}]={p[i][t]} outside [{lo},{hi}]")
    return err


def check_demand_reserve(
    inst: dict[str, Any], y: list[list[int]], p: list[list[float]], horizon: int
) -> list[str]:
    n = len(y)
    err: list[str] = []
    for t in range(horizon):
        d = inst["demand"][t]
        s = math.fsum(p[i][t] for i in range(n))
        if abs(s - d) > TOL:
            err.append(f"demand[{t}]: supply {s} != {d}")
        r = inst["reserve"][t]
        cap = math.fsum(inst["P_max"][i] * y[i][t] for i in range(n))
        if r > 0 and cap < d + r - RESERVE_TOL:
            err.append(f"reserve[{t}]: {cap} < {d + r}")
    return err


def check_min_runs(inst: dict[str, Any], y: list[list[int]], horizon: int) -> list[str]:
    err: list[str] = []
    for i in range(len(y)):
        yp = int(inst["y_prev"][i])

        def at(t: int, i: int = i, yp: int = yp) -> int:
            return yp if t < 0 else y[i][t]

        # initial conditions
        held_for = int(inst["min_on"][i] if yp == 1 else inst["min_off"][i])
        rem = max(0, held_for - int(inst["n_init"][i]))
        for t in range(min(rem, horizon)):
            if y[i][t] != yp:
                err.append(f"unit {i}: initial condition at t={t}")
        # rolling-window min up / min down
        for t in range(horizon):
            if y[i][t] == at(t - 1):
                continue
            run = int(inst["min_on"][i] if y[i][t] == 1 else inst["min_off"][i])
            for tau in range(t, min(t + run, horizon)):
                if y[i][tau] != y[i][t]:
                    err.append(f"unit {i}: min run from t={t} broken at {tau}")
                    break
    return err


def cost_of(inst: dict[str, Any], y: list[list[int]], p: list[list[float]], horizon: int) -> float:
    terms: list[float] = []
    for i in range(len(y)):
        a, b, c = inst["a"][i], inst["b"][i], inst["c"][i]
        d, e, pmin = inst["d"][i], inst["e"][i], inst["P_min"][i]
        tc = int(inst["t_cold"][i])
        yp = int(inst["y_prev"][i])
        for t in range(horizon):
            if y[i][t] != 1:
                continue
            q = p[i][t]
            terms.append(a + b * q + c * q * q + abs(d * math.sin(e * (pmin - q))))
            prev = yp if t == 0 else y[i][t - 1]
            if prev == 0:
                hot = any((yp if tau < 0 else y[i][tau]) == 1 for tau in range(t - tc, t))
                terms.append(inst["a_hot"][i] if hot else inst["a_cold"][i])
    return math.fsum(terms)


def check(inst: dict[str, Any], sol: dict[str, Any]) -> tuple[list[str], float]:
    n = int(inst["n_units"])
    horizon = int(sol["periods"])
    y: list[list[int]] = sol["y"]
    p: list[list[float]] = sol["p"]
    if len(y) != n or len(p) != n or any(len(r) != horizon for r in y + p):
        return ["shape"], math.nan
    err = check_bounds(inst, y, p, horizon)
    err += check_demand_reserve(inst, y, p, horizon)
    err += check_min_runs(inst, y, horizon)
    return err, cost_of(inst, y, p, horizon)


def main(argv: list[str]) -> int:
    inst_dir = Path(argv[1])
    cache: dict[str, dict[str, Any]] = {}
    bad = 0
    checked = 0
    for path in argv[2:]:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            checked += 1
            sol = json.loads(line)
            f = base_file(inst_dir, sol["instance"])
            if str(f) not in cache:
                cache[str(f)] = json.loads(f.read_text())
            err, cost = check(cache[str(f)], sol)
            diff = abs(cost - float(sol["objective"]))
            ok = not err and diff <= max(1e-6 * abs(cost), 1e-3)
            bad += 0 if ok else 1
            print(
                f"{'OK  ' if ok else 'FAIL'} {sol['instance']:<14} seed={sol['seed']} "
                f"cost={cost:.4f} |diff|={diff:.2e} {'; '.join(err[:3])}"
            )
    if checked == 0:
        # An empty file (every row vetoed or infeasible) is no evidence (#462).
        print("FAIL no solutions to check")
        return 1
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
