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

All phases complete (0-7). 62 C++ tests, 21 Python tests passing.

- Phase 0: Project setup (CMake, Catch2, nanobind)
- Phase 1: Expression DAG + delta evaluation + reverse-mode AD
- Phase 2: Penalty objective + adaptive lambda
- Phase 3: Move generation + SA core
- Phase 4: FJ-NL initialization
- Phase 5: LNS destroy-repair
- Phase 6: Multi-threading + SolutionPool
- Phase 7: CHPED benchmarks (4/7/24-unit)

## Future Work

- CI (GitHub Actions)
- Operator overloading on Python handles
- More benchmark instances
- Inner solver integration tests
