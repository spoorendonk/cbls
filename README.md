# CBLS

Constraint-Based Local Search engine for mixed discrete-continuous optimization.
ViolationLS — guided local search over single- and compound-variable jumps on an
expression DAG, with per-constraint GLS weights carrying the feasibility pressure.

**Status: early-stage research solver, actively developed.**

## Example

```cpp
#include <cbls/cbls.h>
#include <cstdio>

int main() {
    cbls::Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    auto result = cbls::solve(m, 5.0);
    printf("objective = %f\n", result.objective);
}
```

Or with operator overloading:

```cpp
cbls::Model m;
auto x = m.Float(0, 10);
auto y = m.Float(0, 10);
m.add_constraint(x + y >= 3.0);
m.minimize(x * x + 2 * x * y + sin(y));
m.close();
```

## Features

- **Variable types**: Bool, Int, Float, List (permutation), Set
- **Nonlinear expressions**: arithmetic, trig, exp/log, comparisons, lambda functions
- **ViolationLS** guided local search (Davies et al. CPAIOR 2024): single-variable Feasibility Jump + compound-move Novelty Jump, GLS weight dynamics, objective-as-soft-constraint
- **Feasibility Jump** construction heuristic (greedy violation reduction)
- **Gradient-based intensification**: Newton steps and backtracking line search on continuous variables
- **Large neighborhood search**: destroy-repair diversification
- **Delta evaluation**: incremental DAG update via BFS dirty-marking
- **Reverse-mode AD**: sparse automatic differentiation for gradient moves
- **Multi-threaded** parallel search with solution pool (opportunistic and deterministic modes)
- **Python bindings** via nanobind

## Build

```bash
cmake -B build
cmake --build build
ctest --test-dir build    # 274 C++ tests, ~39s (add -LE slow for the fast 268, ~8s)
```

The build type defaults to `Release`; pass `-DCMAKE_BUILD_TYPE=Debug` to override
it. Install `ccache` and CMake will use it as a compiler launcher automatically,
which makes a clean rebuild of the same directory near-instant.

With Python bindings:

```bash
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build
.venv/bin/pytest          # 175 tests, 73 of them for the bindings
```

Or install as a Python package:

```bash
pip install .
```

## What works

- Small-to-medium nonlinear mixed-integer problems
- Problems where escaping local optima matters (nonconvex, discontinuous) — GLS reweighting reshapes the landscape on stagnation, and diversification kicks or LNS restart the search from a perturbed assignment
- Problems where exact solvers time out (CBLS finds feasible solutions on instances where SCIP cannot within time limits)

## Known limitations

- Tightly-coupled multi-period problems with long-range constraints (e.g., min up/down times spanning many periods) are hard for a jump-based search — a single-variable jump sees one period at a time, so reaching a first feasible solution is slow
- Solution quality gaps of 15-40% vs exact solvers on problems where MIP works well
- Benchmark models may be simplified relative to their source papers, so a "BKS gap" is not always apples-to-apples; where that applies (uc-chped) the deviation is audited equation by equation in the benchmark's `FIDELITY.md` and its comparison table states what it may and may not claim
- No constraint propagation, cutting planes, or LP relaxation — this is pure local search
- **Set variables are expressible, not yet competitive; List variables are unbenchmarked.** A model whose only variables are structured (List/Set) is searched by the structural batch alone — a first-improvement hill climber over a small random move sample, with no Feasibility Jump and no compound moves. Diversification reaches it, but only with the same unguided moves. On the weighted OR-Library set-covering instances the `Set` encoding lands at 8.6-11.0x the proven optimum while the same instance encoded with one Bool per column is within 9-20% (see `benchmarks/instances/setcover/`); on unicost instances the two nearly converge. `List` variables are **unbenchmarked** in this respect — they have unit-test coverage, but the only List *benchmark* was pharma GLSP, which has been retired, so there is no measured evidence either way. The mechanical cause is shared and has two halves: Feasibility Jump's jumpable-variable whitelist is scalar-only, so no structured variable gets a jump-table entry, and `local_derivative` returns `0.0` for every structural op, so there is no AD signal to build one from. That is reason to expect `List` behaves like `Set` here, not to assume it does not.

## Benchmarks

Three benchmarks carry the comparative story, in priority order. See individual
directories for details.

| # | Benchmark | Runner | Compared against | What it establishes |
|---|-----------|--------|------------------|---------------------|
| 1 | MIPfeas (MIPLIB 2017) | `cbls_mipfeas` | OR-Tools CP-SAT `num_violation_ls` worker only | that this implementation holds up against the reference implementation of the same algorithm. Deliberately **not** a MIP-competitiveness claim |
| 2 | MINLPLib subset | `cbls_minlplib` | SCIP (nonlinear) | non-convex MINLP — the regime CP-SAT's LS worker cannot express. This is the headline claim |
| 3 | UC-CHPED | `cbls_uc_chped` | Pedroso et al. 2014 | a published unit-commitment formulation whose valve-point `\|d·sin(e·(Pmin−P))\|` term is likewise inexpressible in CP-SAT |

A benchmark earns a place on that list on one of two grounds: it contains terms
CP-SAT cannot express (rows 2-3), or it is a same-algorithm head-to-head on a
formulation both sides express identically (row 1). Nothing else qualifies — a
model whose faithful form is a MILP belongs to CP-SAT, CPLEX and Gurobi.

Two further directories are **not** benchmarks in that sense:

| Directory | Runner | What it is |
|-----------|--------|------------|
| `benchmarks/setcover/` | `cbls_setcover` | the scoped coverage check the `Set` variable type was missing — ten OR-Library instances under both a `Set` and a Bool encoding. Its result is a documented limitation, not a comparative claim |
| `benchmarks/chped/` | none — consumed by the `cbls_chped` example | reference-only: model builder, data and a SCIP baseline for CHPED dispatch. Benchmark 1 was dropped as active work; `examples/chped.cpp` includes the model header, so it survives as a worked modelling example with no runner or comparison table |

## Architecture

See [docs/architecture.md](docs/architecture.md) for solver internals: expression DAG, ViolationLS outer loop, Generalised Feasibility Jump, move generation, inner solver, LNS, threading.

## License

[MIT](LICENSE)
