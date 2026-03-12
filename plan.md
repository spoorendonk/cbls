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

All phases complete (0-8). Merged to `main` via PRs #1, #2.
91 C++ tests (Catch2), 67 Python binding tests (pytest).
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

## Future Work

- CI (GitHub Actions)
- ~~Standard benchmark instances with known optima~~ → Done: 13-unit (Sinha et al. 2003) and 40-unit (Taipower)
- 140-unit CHPED (replicated from 40-unit)
- Reference solver comparison (SCIP or Couenne)
- Inner solver integration tests
