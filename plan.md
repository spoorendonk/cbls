# CBLS Engine — Implementation Plan

## Context

Open-source **Constraint-Based Local Search engine** — SA on expression DAG with
Hexaly-style modeling API. Targets nonlinear, nonconvex, mixed discrete+continuous
problems. Enriched with FJ-NL warm-start, Newton/gradient moves, and LNS.

## Architecture

C++17 library (`libcbls`) with nanobind Python bindings (`_cbls_core`).

```
include/cbls/       # Public headers
  dag.h             # Variable, ExprNode, VarType, NodeOp
  dag_ops.h         # full/delta evaluate, reverse-mode AD
  model.h           # Model builder API
  violation.h       # ViolationManager, AdaptiveLambda
  moves.h           # Move generators, apply/undo, MoveProbabilities
  search.h          # SA solver, FJ-NL initialization
  lns.h             # LNS destroy-repair
  pool.h            # SolutionPool, ParallelSearch
  inner_solver.h    # InnerSolverHook interface
  rng.h             # RNG wrapper
  cbls.h            # Convenience header
src/                # Implementation
python/             # nanobind bindings + pyproject.toml
tests/              # Catch2 C++ tests + Python binding tests
benchmarks/chped/   # CHPED benchmark data + model builder
examples/           # simple.cpp, chped.cpp
```

## Status

All phases complete (0-8). Merged to `main` via PRs #1-#4.
93 C++ tests (Catch2), 67 Python binding tests (pytest).
Legacy Python implementation removed.
Repo: https://github.com/spoorendonk/cbls

### Completed

- Phase 0: Project setup (CMake, Catch2, nanobind)
- Phase 1: Expression DAG + delta evaluation + reverse-mode AD
- Phase 2: Penalty objective + adaptive lambda
- Phase 3: Move generation + SA core
- Phase 4: FJ-NL initialization
- Phase 5: LNS destroy-repair
- Phase 6: Multi-threading + SolutionPool
- Phase 7: CHPED benchmarks (4/7/24-unit)
- Code review: bounds checks, exception safety, Python error translation
- Legacy Python removal, build/test docs updated
- Phase 8: Expr wrapper + operator overloading + 8 new ops (PR #2)
  - New ops: tan, exp, log, sqrt, geq, neq, lt, gt (evaluate + AD)
  - Expr class: `x*x + 2*x*y + sin(y)` syntax in C++ and Python
  - Scalar-Expr mixed ops, scalar comparisons
  - Model::Float/Int/Bool/Constant Expr-returning methods
  - Full Python bindings with dunder methods + free functions

## Benchmark Suite

| # | Domain | Problem | Core nonlinearity | Key CBLS features | Status |
|---|--------|---------|-------------------|--------------------|--------|
| 1 | Energy | CHPED dispatch | Valve-point sinusoidal cost | FloatVar, delta eval | **Done** (PR #3, #4) |
| 2 | Energy | UC-CHPED | Valve-point + binary commitment | BoolVar + FloatVar, min up/down, startup costs | **Data + reference solver done** |
| 3 | Energy | ROADEF 2010 — Nuclear Outage | Stochastic production × binary outage dates | BoolVar/IntVar + FloatVar, inner solver hook, 500 scenarios | Planned |
| 4 | Maritime | Fleet Bunker + ECA | Cubic fuel (v³), binary fuel switching, tank dynamics | FloatVar + BoolVar, nonlinear state, inner solver | Planned |
| 5 | Manufacturing | Pharma GLSP + Shelf-Life | Coupled lot-sizing/sequencing, shelf-life cross-stage | ListVar + FloatVar, changeover, inner solver | Planned |

### Benchmark 2: UC-CHPED

Unit Commitment with valve-point effects. 13/40-unit instances from Pedroso et al.
(2014), 24-hour horizon with binary commitment, min up/down times, hot/cold startup
costs. Instance data in `benchmarks/instances/uc-chped/data.py`, MIP reference solver
in `benchmarks/chped/reference_solve.py --uc`.

**Next step:** Build CBLS model with BoolVar per unit per period, FloatVar dispatch,
violation constraints for min up/down, inner solver hook for dispatch optimization.

### Benchmark 3: ROADEF/EURO 2010 — Nuclear Outage Scheduling (planned)

Schedule maintenance/refueling outages for EDF's French nuclear fleet over multi-year
horizon. Binary outage dates × continuous production across 500 demand scenarios.
Two-stage stochastic: binary first-stage (outage dates) coupled to continuous
second-stage (production) via LP per scenario. Public instances at challenge.roadef.org.
Best competition results came from pure local search (Jost & Savourey, 1st prize).

**Key CBLS features tested:** Inner solver hook, stochastic evaluation (lambda over
scenario set), BoolVar/IntVar + FloatVar coupling.

### Benchmark 4: Maritime Fleet Bunker + ECA Optimization (planned)

Optimize sailing speed (float), bunkering quantity (float), fuel type at ECA
boundaries (bool). Cubic fuel consumption (v³), binary ECA fuel switching creates
discontinuities, nonlinear tank level dynamics. Instances from LINER-LIB (Brouer et
al. 2014) with ECA + fuel parameters from IMO zones. Vilhelmsen et al. (2024) ALNS
matheuristic is primary baseline (3.35% gap).

**Key CBLS features tested:** Nonlinear state evolution (tank dynamics), inner solver
hook (closed-form optimal speed per leg from calculus).

### Benchmark 5: Pharma Campaign Scheduling — GLSP + Shelf-Life (planned)

Multi-product, multi-line pharmaceutical production. Decide production sequence per
line (list) + lot sizes per campaign (float). Shelf-life coupling: batch expires unless
stage-2 starts within W days of stage-1 completion; gap depends on lot sizes of all
intervening campaigns — nonlinear feasibility condition coupling all float variables
across all lines. Instances from Haase & Kimms (2000) canonical GLSP dataset.

**Key CBLS features tested:** First real test of **ListVar** + FloatVar coupling,
auto-generated list moves (insert, remove, swap, reverse, relocate, 2-opt), inner
solver hook (given fixed sequence, solve lot sizes as LP).

## Future Work

- CI (GitHub Actions)
- ~~Standard benchmark instances with known optima~~ → Done: 13-unit and 40-unit (PR #3)
- ~~Reference solver comparison~~ → Done: scipy DE + PySCIPOpt (PR #4)
  - SCIP confirms known optima: 13-unit 0.02%, 40-unit 0.25% gap
  - Our SA gap: 13-unit 4.27%, 40-unit 5.75% — room for improvement
- 140-unit CHPED (replicated from 40-unit)
- SA quality improvements (close the 4-6% gap to known optima)
- Inner solver integration tests
