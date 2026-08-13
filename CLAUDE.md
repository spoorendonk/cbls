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

CBLS = constraint-based local search. ViolationLS (guided local search over single- and compound-variable jumps) on an expression DAG with penalty-method feasibility. Full details in `docs/architecture.md`.

### Core pipeline

1. **Model building** (`include/cbls/model.h`, `include/cbls/expr.h`) — Declare typed variables (Bool, Int, Float, List, Set), build expressions via operator overloading, add constraints, set objective, call `close()` which topologically sorts the DAG.

2. **Expression DAG** (`include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`) — Variables use negative handles `-(id+1)`, nodes use non-negative `id`. 23 operation types. Two evaluation modes:
   - `full_evaluate`: evaluate all nodes in topo order (initialization)
   - `delta_evaluate`: BFS dirty-marking from changed variables, recompute only affected nodes (moves)
   - Reverse-mode AD via `compute_all_partials` for the continuous (Newton) jump-value engine

3. **Search** (`src/search.cpp`) — ViolationLS batch outer loop (Davies et al. CPAIOR 2024, Algorithm 6). The objective is folded into the constraints as `obj <= bound`; each batch is a Feasibility Jump, Novelty Jump, or STRUCTURAL batch (selected by config probabilities). The objective bound is tightened on each new real-feasible solution; on stagnation the assignment is perturbed or diversified via LNS.

4. **Feasibility Jump** (`src/feasibility_jump.cpp`) — Generalised Feasibility Jump: a `JumpTable` of cached per-variable best jumps (score = `-W·δ_G`), best-of-N scan-set sampling, GLS weight dynamics (bump violated + ρ-decay), and Novelty Jump compound moves (Algorithms 4–5). Float jump values come from Newton-toward-violated-root candidates via reverse-mode AD.

5. **Moves** (`src/moves.cpp`) — typed move generators by variable type (bool flip, int ±1/rand, float perturb, list swap/2opt/relocate/or-opt, set add/remove/swap). Scalar moves are subsumed by FJ's jump values; the list/set moves feed the STRUCTURAL batch.

6. **Inner solver** (`src/inner_solver.cpp`) — `FloatIntensifyHook`: coordinate descent over float variables using Newton steps on violated constraints + backtracking line search on objective. Triggered on each new feasible solution.

7. **LNS** (`src/lns.cpp`) — Destroy 30% of variables, repair via FJ. Fires every N diversification kicks. Accepts on a lexicographic (real-violation, objective) key.

8. **Violation & penalty** (`src/violation.cpp`) — `total_violation = Σ W[c]·max(0, viol_c)` with per-constraint GLS weights `W`. `weighted_violation_delta` is the no-commit counterfactual δ_G. `augmented_objective() = obj + total_violation()` is the penalty-method metric the inner solver descends.

9. **Parallel search** (`src/pool.cpp`) — `SolutionPool` + `ParallelSearch` with opportunistic (independent seeds) or deterministic (epoch-sync) modes.

10. **I/O** (`src/io.cpp`) — JSONL `.cbls` model format. CLI in `src/cli.cpp`.

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
| `cbls_mipfeas` | `benchmarks/mipfeas/` | MIPfeas runner (one instance per process) |
| `cbls_minlplib` | `benchmarks/minlplib/` | MINLPLib benchmark runner |
| `cbls_setcover` | `benchmarks/setcover/` | OR-Library set-covering runner (Set-variable coverage check, #93) |

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

The `benchmarks/chped/` directory is reference-only (Benchmark 1 was dropped).

### Benchmark priority

Four benchmarks carry the comparative story, in this order. Work the list top
down; do not start lower-priority benchmark work while a higher one is open.

| # | Benchmark | Compared against | Purpose | Tracking |
|---|-----------|------------------|---------|----------|
| 1 | `mipfeas` | OR-Tools CP-SAT `num_violation_ls` worker **only** | head-to-head correctness *and* performance against the reference implementation of the same jump-based algorithm | #90 |
| 2 | `minlplib` | SCIP (nonlinear) | performance on non-convex MINLP — the regime CP-SAT's LS worker cannot express | #89, #87 |
| 3 | `uc-chped` | Pedroso 2014 | a good result on a published unit-commitment formulation | #25, #91, #92 |
| 4 | `pharma-glsp` | Goerler et al. 2020 | a List-variable use case on a published formulation | #28, #94 |

**Priority 1 is deliberately not a MIP shootout.** Compare only against CP-SAT's
`num_violation_ls` worker (same algorithm, different implementation). Do **not**
compare against CP-SAT's default full portfolio, Xpress, Gurobi or CPLEX — see
epic #87 for why that framing is rejected.

`bunker-eca` and `nuclear-outage` are **deprioritised, not dropped**: their code,
data, tests and open issues (epics #26, #27) all stay, but no new work starts on
them. Don't pick up one of their issues just because it looks tractable.

`setcover` is **not a fifth benchmark**. It is the scoped coverage check the
`Set` variable type had been missing (#93): ten small OR-Library set-covering
instances, run under both a `Set` and a Bool encoding, whose result is a
documented limitation rather than a comparative claim (see
`benchmarks/instances/setcover/README.md`). Don't grow it into an epic.

### Benchmark worktrees

Active work happens in sibling git worktrees under `~/code/my/cbls/`. Each session works ONLY on its assigned benchmark.

| Worktree | Problem | Epic |
|----------|---------|------|
| `uc-chped/` | UC-CHPED: unit commitment + valve-point dispatch | #25 |
| `pharma-glsp/` | Pharma GLSP + shelf-life campaign scheduling | #28 |
| `nuclear-outage/` | ROADEF 2010 nuclear outage scheduling — deprioritised | #26 |
| `bunker-eca/` | Maritime fleet bunker + ECA fuel switching — deprioritised | #27 |

Engine-wide (cross-cutting) work is tracked under epic #24.

### Common benchmark workflow

Each benchmark session must follow these steps in order:

0. **Plan first** — Read the benchmark's epic (linked above) and its open sub-issues, plus `docs/architecture.md` (solver internals). Investigate what already exists for your benchmark (data, reference solver, model code). Propose an approach and wait for approval before implementing. This is the Plan phase — do not skip it.

1. **Download/prepare instance data** — Write a download/generation script into `benchmarks/instances/{name}/`. Follow the pattern of `benchmarks/instances/uc-chped/` (Python data definitions + JSONL serialization). Source data from public benchmark libraries, competition archives, or papers.

2. **Find a reference solver** — Implement a reference solver in `benchmarks/{name}/reference_solve.py` using SCIP (PySCIPOpt) or another open-source solver if SCIP can't handle the formulation. Follow the pattern in `benchmarks/chped/reference_solve.py`.

3. **Collect best-known results** — Find published results from papers/competitions. Write to `benchmarks/instances/{name}/comparison.csv` with columns: instance, method, objective, gap, source.

4. **Implement CBLS model** — Create `benchmarks/{name}/data.h` (C++ data structs + loaders) and `benchmarks/{name}/{name}_model.h` (model builder). Follow the pattern of `benchmarks/chped/chped_model.h`. **Critical rules:**
   - Implement features generically in `include/cbls/` and `src/` — not benchmark-specific hacks
   - You may READ files in other worktree sibling folders (e.g., `../cbls/`, `../nuclear-outage/`) to understand patterns, but NEVER WRITE to them or to their git branches
   - If the solver needs new ops, moves, or hooks — implement them in the core library so all benchmarks benefit
   - Add a runner executable in `benchmarks/{name}/{name}.cpp`
   - Add Catch2 tests in `tests/test_{name}.cpp`
   - Update `CMakeLists.txt` for new executables and tests

5. **Run comparison** — Compare CBLS vs reference solver vs best-known results. Report objective, gap %, and solve time. Update `comparison.csv`.

6. **Verify correctness** — Check CBLS solutions against best-known solutions. Feasibility must be verified (all constraints satisfied). Objective should be within reasonable gap of BKS.

7. **Commit often** — Commit to your worktree branch (`bench/{name}`) after each meaningful step. Use descriptive commit messages.

8. **Self-review loop** — After each commit, review your own changes for issues/nits. Fix and commit again. Repeat until clean.

9. **Do not interrupt the user** — No exceptions. Keep going until the benchmark is fully implemented, running, and producing correct results. Only stop if you hit a fundamental blocker that requires API/architecture changes.

### Cross-worktree rules

- Each session works ONLY on its assigned benchmark
- READ other worktrees for reference — never write to them
- Merge to main via squash-merge when done
- Pull main into your branch before final merge to pick up other benchmarks' core changes
