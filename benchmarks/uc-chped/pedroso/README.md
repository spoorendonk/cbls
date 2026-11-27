# UC-CHPED references on SCIP (GPL-3.0-or-later)

**This directory is licensed GPL-3.0-or-later** ([LICENSE](LICENSE)), because
`ucp_valve_scip.py` is a port of Pedroso, Kubo & Viana's GPL-3 `ucp_valve.py`.
The rest of cbls is MIT. Nothing outside this directory imports it: it reads the
MIT instance data in `benchmarks/instances/uc-chped/data.py` and writes JSON
lines, which other tools may read as data.

It replaces `benchmarks/uc-chped/reference_solve.py` (a 50-segment
piecewise-linear MIP that was not a bound in either direction; FIDELITY.md §4).

## Two methods

| `--method` | What it solves | Bounds |
|---|---|---|
| `pwl` | Pedroso et al. 2014 (arXiv:1404.4944), Algorithm 1, as `ucp_valve.py` implements it: interpolate the fuel cost through the valve points, solve the MIP, price its dispatch at the true cost, add breakpoints where the dispatch landed, repeat | LB: the MIP's dual bound, valid while every unit has `c <= e*f^2/2` (the paper's Appendix B; the runner names any unit that breaks it). UB: the dispatch priced at the true cost |
| `exact` | The same unit-commitment model with the true `\|e*sin(f*(p_min - p))\|` term, by SCIP's spatial branch-and-bound. Not in the paper | SCIP's dual bound and incumbent (re-priced at the true cost) |

Both use `ucp_valve.py`'s unit-commitment model (`ucp_base`): demand as an
equality, its hot/cold startup logic and initial conditions. That is the source's
problem, not cbls's model (which has demand `>=`, FIDELITY.md §1.4).

```sh
.venv/bin/python benchmarks/uc-chped/pedroso/ucp_valve_scip.py \
    --instance ucp13 --periods 3 --method pwl --cpu 3600 --out results.jsonl
```

Instances: `ucp13`, `ucp40` (identical to the authors' `ucp_data.py`,
FIDELITY.md §7), and cbls's own `ucp100`, `ucp200`; `--periods` above 24 uses
`extend_horizon()`. Timed runs go under `xk lock`.

## Deviations from `ucp_valve.py`

Source: <http://www.dcc.fc.up.pt/~jpp/code/valve/ucp_valve.py>, 2013-12-09,
17563 bytes, sha256 `c1f6e8d9ba316c6291e0eab447a3df65a773adacb1c996a610f68c62d46d9104`
(the copy FIDELITY.md §7.1 records).

1. **Solver.** SCIP through PySCIPOpt, not Gurobi 5.0.1. SOS2 by `addConsSOS2`.
2. **Gap limits.** Outside strict mode `ucp_valve.py` runs at Gurobi's defaults
   (MIPGap 1e-4, MIPGapAbs 1e-10), and its "go strict" test reads MIPGap.
   SCIP's defaults are 0, so they are set to Gurobi's values. Strict mode maps
   to `limits/gap` and `limits/absgap` 1e-12 and `numerics/feastol` 1e-9 (SCIP
   has no separate integrality tolerance).
3. **Warm start.** The paper passes each solution on as the next MIP's start.
   `ucp_valve.py` saves `.start` rather than `.x` after the solve, so its start is
   never a solution. The port passes the solution as a SCIP partial solution of
   `y, hot, cold, on, off`, and raises `heuristics/completesol/maxunknownrate` to
   1, because every other variable is unknown and SCIP's default (0.85) would
   drop the start silently. `--no-warm-start` turns it off.
4. **Clock.** `ucp_valve.py` budgets the loop in CPU time (`time.clock()`, from
   process start) and gives Gurobi the remainder as its TimeLimit, which is wall
   time. The port budgets from the start of the solve in process CPU time and
   gives SCIP the remainder in wall time too (`timing/clocktype 2`). One thread
   on its own core, so the two clocks barely differ. SCIP's CPU clock is not an
   option: it reads `times()`, a syscall SCIP made ~775k times a second here.
   That system time is a third of the process's CPU and its user-time clock
   never sees it: an `exact` solve at a 60 s limit took 93 CPU-s, and at
   3600 s none stopped before a 3900 s wall stop (#441).
5. **Data.** From cbls's `data.py`, not `ucp_data.py`; identical for `ucp13` and
   `ucp40` (FIDELITY.md §7). Off-horizon indices are clipped where `ucp_valve.py`
   would raise; no instance here reaches them.
6. **Reporting.** `ucp_valve.py` prints each iteration's LB and UB. The port
   records every iteration, and reports `lb`/`ub` as the best over them (each
   bounds the true optimum while every unit meets the concavity condition; the
   paper notes the UB can rise between iterations), with the last iteration's as
   `last_lb`/`last_ub`. `status` is `converged` or `time limit reached`, as the
   script prints, `infeasible` (no iteration found a UB), or `stopped: infeasible
   MIP after a UB`. Every iteration's solution is re-checked
   outside SCIP (demand, reserve, dispatch limits) into its iteration's `check`;
   the top-level `check` is the reported UB's (exact's late best: item 8). `out_of_budget` is false on a
   proven-infeasible MIP.
7. **Python 3**, typed; `lower_brkpts` and `upper_brkpts` are not ported
   (`ucp_valve.py` uses neither in its loop). A unit with valve frequency
   `f <= 0` is refused: the script substitutes `f` in its breakpoint builders
   but `has_added_p` would divide by zero, and no instance has one.
8. **`--backend driver`** (#442, opt-in; the default `pyscipopt` is items 1-7).
   The wheel only builds each model; `cbls_scip_traj`
   (`benchmarks/common/scip_driver/`) solves it on SCIP 11 at the pinned
   commit, at `--threads` N (concurrent at N > 1, defaults), on a wall
   `--time-limit` counted on `time.monotonic()` from the start, writing and
   reading included. The record carries `backend`, `threads`, `time_limit_s`
   (not `cpu_limit_s`), `wall_s` (not `cpu_s`), `scip_git_hash` and
   `trajectory`: `[[wall_s, objective], ...]`, the running best, strictly
   falling. For pwl it is the true-cost UB at each iteration's end that beats
   every earlier one; each iteration also records `start_completed` (N > 1;
   on ucp13 T=1 completesol mostly stops at its default node limit, at N = 1
   too, and the start is then lost). For exact
   it is SCIP's objective at each improving incumbent, an upper bound on that
   incumbent's true cost (the fuel and valve terms are `>=` relaxations), so
   the final `ub` may sit below the last point. Each solve's `--time-limit` is
   what is left just before the spawn; the driver is killed `--driver-grace`
   (60 s) past it, an error unless `--salvage-killed` finds the result it wrote
   before `SCIPfree` (`driver_killed_in_free` counts those). A pwl point past
   `time_limit_s` is a real time, which a score over `[0, time_limit_s]`
   ignores. A failed run keeps its work directory, named on stderr.
   **Budget:** `ub` counts only solutions SCIP found within `time_limit_s`. A
   pwl iteration records `found_s` (its best incumbent's logged time, else its
   return) and `in_budget`; its trajectory point is stamped at `found_s`, and a
   late one's UB does not count and cannot converge the run. In exact, a late
   best goes to `trajectory_after_budget` and `ub_after_budget`,
   `scip_objective_after_budget`, `check_after_budget`, `status_after_budget`;
   `status` is then `timelimit` and `ub` the last in-budget point's SCIP
   objective (`ub_is_scip_objective`, not re-checked; `score.py` lists those
   rows). Every `lb` is SCIP's at its stop, which can overshoot the limit by a
   step. Strict mode's `numerics/feastol`
   1e-9 also sets `numerics/sumepsilon` 1e-9: SCIP 11 refuses sumepsilon above
   feastol (`set.c:681`), and SCIP 10 did not check.

## Not validated yet

Before any table uses it, it must reproduce Table 2: the three proven optima
(`ucp13` 1p 11701.28, 3p 38849.84; `ucp40` 1p 55644.79) to the printed
decimals, and the other seven rows inside Pedroso's [LB, UB] at one CPU hour.
