# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@.devkit/standards/nanobind.md

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build && cmake --build build -j$(nproc)
```

```test
ctest --test-dir build --output-on-failure -j$(nproc) && (pytest --tb=short -q; rc=$?; [ $rc -eq 0 ] || [ $rc -eq 5 ])
```

Run a single C++ test by name (Catch2 `-c` filter):
```bash
ctest --test-dir build -R "test_name_substring"
```

Run a single Python test:
```bash
pytest tests/python/test_foo.py::test_specific -x
```

Python bindings (off by default):
```bash
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
cmake --build build -j$(nproc)
```

## Architecture

CBLS = constraint-based local search. Simulated annealing over an expression DAG with penalty-method feasibility. Full details in `docs/architecture.md`.

### Core pipeline

1. **Model building** (`include/cbls/model.h`, `include/cbls/expr.h`) — Declare typed variables (Bool, Int, Float, List, Set), build expressions via operator overloading, add constraints, set objective, call `close()` which topologically sorts the DAG.

2. **Expression DAG** (`include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`) — Variables use negative handles `-(id+1)`, nodes use non-negative `id`. 23 operation types. Two evaluation modes:
   - `full_evaluate`: evaluate all nodes in topo order (initialization)
   - `delta_evaluate`: BFS dirty-marking from changed variables, recompute only affected nodes (moves)
   - Reverse-mode AD via `compute_all_partials` for gradient/Newton moves

3. **Search** (`src/search.cpp`) — Main SA loop. Phase 1 (20% time): Feasibility Jump construction heuristic. Phase 2 (80%): SA with Metropolis acceptance, periodic reheat, adaptive lambda penalty, adaptive move probabilities.

4. **Moves** (`src/moves.cpp`) — 12 move types by variable type (bool flip, int ±1/rand/neighbors, float perturb/newton_tight/gradient_lift, list swap/2opt, set add/remove/swap). `MoveProbabilities` rebalances every 1000 evaluations based on acceptance rates.

5. **Inner solver** (`src/inner_solver.cpp`) — `FloatIntensifyHook`: coordinate descent over float variables using Newton steps on violated constraints + backtracking line search on objective. Triggered every 10 discrete-variable accepts and on reheat.

6. **LNS** (`src/lns.cpp`) — Destroy 30% of variables, repair via FJ. Fires every N reheats.

7. **Violation & penalty** (`src/violation.cpp`) — `F = obj + λ * total_violation`. `AdaptiveLambda` increases λ when stuck infeasible, decreases when feasible-but-not-improving.

8. **Parallel search** (`src/pool.cpp`) — `SolutionPool` + `ParallelSearch` with opportunistic (independent seeds) or deterministic (epoch-sync) modes.

9. **I/O** (`src/io.cpp`) — JSONL `.cbls` model format. CLI in `src/cli.cpp`.

### Key extension points

- **`InnerSolverHook`** — subclass to provide domain-specific continuous optimization (see benchmark `*_hook.h` files for examples)
- **New operations** — add to `NodeOp` enum in `dag.h`, implement `evaluate()` in `dag.cpp`, `local_derivative()` for AD, `delta_evaluate` support in `dag_ops.cpp`

### Build targets

| Target | Source | Description |
|--------|--------|-------------|
| `cbls_lib` | `src/*.cpp` | Core library (static) |
| `cbls_cli` | `src/cli.cpp` | CLI executable |
| `cbls_tests` | `tests/*.cpp` | Catch2 test suite |
| `cbls_uc_chped` | `benchmarks/uc-chped/` | UC-CHPED benchmark runner |
| `cbls_nuclear_outage` | `benchmarks/nuclear-outage/` | Nuclear outage benchmark runner |
| `cbls_bunker_eca` | `benchmarks/bunker-eca/` | Bunker ECA benchmark runner |
| `cbls_pharma_glsp` | `benchmarks/pharma-glsp/` | Pharma GLSP benchmark runner |

### Dependencies

- **nlohmann/json** v3.11.3 — JSONL I/O (FetchContent)
- **Catch2** v3.5.2 — C++ tests (FetchContent)
- **nanobind** — Python bindings (optional, via scikit-build-core)

### C++ standard

C++17 (`CMAKE_CXX_STANDARD 17`). Note: the `.devkit/standards/cpp.md` says C++23 but this project uses C++17.

## Benchmarks

Each benchmark follows the same pattern:
- `data.h` — C++ data structures + constexpr data arrays
- `*_model.h` — model builder function
- `*_hook.h` — custom `InnerSolverHook` for domain-specific optimization
- `*.cpp` — runner executable
- `reference_solve.py` — SCIP/PySCIPOpt baseline
- `verify_*.h` — solution correctness checker

The `benchmarks/chped/` directory is reference-only (Benchmark 1 was dropped). Active benchmarks: uc-chped, nuclear-outage, bunker-eca, pharma-glsp.
