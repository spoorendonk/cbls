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
ctest --test-dir build    # 93 C++ tests
```

With Python bindings:

```bash
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
cmake --build build
pytest                    # 67 Python binding tests
```

Or install as a Python package:

```bash
pip install .
```

## What works

- Small-to-medium nonlinear mixed-integer problems
- Problems where escaping local optima matters (nonconvex, discontinuous) — GLS reweighting reshapes the landscape on stagnation, and diversification kicks or LNS restart the search from a perturbed assignment
- Problems where exact solvers time out (CBLS finds feasible solutions on instances where SCIP cannot within time limits)
- Stochastic scheduling with continuous inner optimization (nuclear outage benchmark beats MIP baselines)

## Known limitations

- Tightly-coupled multi-period problems with long-range constraints (e.g., min up/down times spanning many periods) are hard for a jump-based search — a single-variable jump sees one period at a time, so reaching a first feasible solution is slow
- Solution quality gaps of 15-40% vs exact solvers on problems where MIP works well
- Benchmark models are simplified relative to their source papers; comparison results are not directly comparable to published results
- No constraint propagation, cutting planes, or LP relaxation — this is pure local search
- **Set variables are expressible, not yet competitive.** A model whose only variables are structured (List/Set) is searched by the structural batch alone — a first-improvement hill climber over a small random move sample, with no Feasibility Jump and no compound moves. Diversification reaches it, but only with the same unguided moves. On the weighted OR-Library set-covering instances the `Set` encoding lands at 8.5-9.9x the proven optimum while the same instance encoded with one Bool per column is within 9-20% (see `benchmarks/instances/setcover/`); on unicost instances the two nearly converge. List variables, which appear alongside scalars in the pharma GLSP model, are not affected in the same way.

## Benchmarks

Benchmark domains exercise different solver features. See individual directories for details.

| Domain | Problem | Key features tested |
|--------|---------|-------------------|
| Energy | CHPED dispatch | Float variables, delta evaluation |
| Energy | UC-CHPED | Bool + Float, min up/down constraints |
| Energy | Nuclear outage scheduling | Inner solver hook, stochastic evaluation |
| Manufacturing | Pharma GLSP | List variables (sequencing), lot-sizing |
| Combinatorial | OR-Library set covering | Set variables vs. the Bool encoding of the same instance |

## Architecture

See [docs/architecture.md](docs/architecture.md) for solver internals: expression DAG, ViolationLS outer loop, Generalised Feasibility Jump, move generation, inner solver, LNS, threading.

## License

[MIT](LICENSE)
