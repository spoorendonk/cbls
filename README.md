# CBLS

Constraint-Based Local Search engine for mixed discrete-continuous optimization.
Simulated annealing over an expression DAG with penalty-method feasibility.

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
- **Simulated annealing** with Metropolis acceptance, exponential cooling, periodic reheat
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
- Problems where SA's ability to escape local optima matters (nonconvex, discontinuous)
- Problems where exact solvers time out (CBLS finds feasible solutions on instances where SCIP cannot within time limits)
- Stochastic scheduling with continuous inner optimization (nuclear outage benchmark beats MIP baselines)

## Known limitations

- Tightly-coupled multi-period problems with long-range constraints (e.g., min up/down times spanning many periods) are hard for SA — construction heuristic struggles to find feasible solutions
- Solution quality gaps of 15-40% vs exact solvers on problems where MIP works well
- Benchmark models are simplified relative to their source papers; comparison results are not directly comparable to published results
- No constraint propagation, cutting planes, or LP relaxation — this is pure local search

## Benchmarks

Five benchmark domains exercise different solver features. See individual directories for details.

| Domain | Problem | Key features tested |
|--------|---------|-------------------|
| Energy | CHPED dispatch | Float variables, delta evaluation |
| Energy | UC-CHPED | Bool + Float, min up/down constraints |
| Energy | Nuclear outage scheduling | Inner solver hook, stochastic evaluation |
| Maritime | Fleet bunker + ECA | Nonlinear fuel cost (v³), binary fuel switching |
| Manufacturing | Pharma GLSP | List variables (sequencing), lot-sizing |

## Architecture

See [docs/architecture.md](docs/architecture.md) for solver internals: expression DAG, SA loop, move generation, inner solver, LNS, threading.

## License

[MIT](LICENSE)
