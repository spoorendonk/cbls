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

- **Variable types**: Bool, Int, Float, List (a permutation, or an ordered subset of a universe with a length window, optionally partitioned across several Lists), Set
- **Nonlinear expressions**: arithmetic, trig, exp/log, comparisons, ceil/floor/round, table lookup by a decision (`element`, each index any scalar, truncated toward zero), lambda functions — including ones that read other decisions (`lambda_sum(list, f, extra)`)
- **ViolationLS** guided local search (Davies et al. CPAIOR 2024): single-variable Feasibility Jump + compound-move Novelty Jump, GLS weight dynamics, objective-as-soft-constraint
- **Feasibility Jump** construction heuristic (greedy violation reduction)
- **Gradient-based intensification**: Newton steps and backtracking line search on continuous variables
- **Large neighborhood search**: destroy-repair diversification
- **Delta evaluation**: a one-variable move re-evaluates a topologically sorted cone cached when the model is frozen; probes roll back from an undo log
- **Reverse-mode AD**: sparse automatic differentiation for gradient moves
- **Multi-threaded** cooperative portfolio: workers share incumbents through a solution pool and restart stalled searches from it
- **Python bindings** via nanobind

## Build

```bash
cmake -B build
cmake --build build
ctest --test-dir build    # 882 C++ tests, ~25s (add -LE slow for the fast 872, ~12s)
```

The build type defaults to `Release`; pass `-DCMAKE_BUILD_TYPE=Debug` to override
it. Install `ccache` and CMake will use it as a compiler launcher automatically,
which makes a clean rebuild of the same directory near-instant.

Two off-by-default measurement builds, each into its own directory:
`-DCBLS_PROFILE=ON` adds symbols and frame pointers on top of Release, and
`-DCBLS_SANITIZE=address,undefined,float-cast-overflow` builds under ASan/UBSan.
See [docs/profiling.md](docs/profiling.md) for the heap and CPU profiling
recipes that use them.

With Python bindings (the venv first, which also builds the package into it):

```bash
uv sync --extra dev
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build
.venv/bin/pytest          # 1506 tests, 263 of them for the bindings
```

Tests of the benchmark baselines skip themselves when `ortools` or `pyscipopt` is
absent, so install the extra before treating a green run as complete:

```bash
uv sync --extra dev --extra benchmarks   # names both: uv sync drops an unnamed extra
CBLS_REQUIRE_BENCHMARKS=1 .venv/bin/pytest   # turns that skip into an error
```

To use CBLS from another project, install it as a Python package
(this repository's own venv is managed by `uv sync` only):

```bash
pip install .
```

## Scope

CBLS is built for small-to-medium mixed discrete-continuous models whose terms an
exact MIP or CP solver cannot take as written: non-convex, transcendental or
discontinuous ones. GLS reweighting reshapes the landscape on stagnation, and
diversification kicks or LNS restart the search from a perturbed copy of the
point it is exploring around — its own incumbent, a peer's state under the
multi-threaded portfolio, or the live assignment until the first feasible
solution. How it fares against CP-SAT, SCIP and published results is what the
benchmarks below measure; their full-scale tables are not published yet (#253).

## Known limitations

- Pure local search: no constraint propagation, cutting planes or LP relaxation, so no bound and no proof of optimality or infeasibility.
- Benchmark models may be simplified relative to their source papers, so a "BKS gap" is not always apples-to-apples; where that applies (uc-chped) the deviation is audited equation by equation in [`benchmarks/uc-chped/FIDELITY.md`](benchmarks/uc-chped/FIDELITY.md), and the comparison table states what it may and may not claim.
- **`Set` variables are expressible, not competitive; `List` variables have no published result.** On weighted OR-Library set covering the `Set` encoding lands at several times the optimum, while the same model in Bools is within 4-9% of it. No `List` result is published until the CVRP campaign (#204). The evidence and its mechanical cause are in [docs/benchmarks/README.md](docs/benchmarks/README.md#structured-variables-the-claim-is-currently-unsupported).

## Benchmarks

Three benchmarks carry the comparative story, in priority order. See individual
directories for details.

| # | Benchmark | Runner | Compared against | What it establishes |
|---|-----------|--------|------------------|---------------------|
| 1 | MIPfeas (MIPLIB 2017) | `cbls_mipfeas` | OR-Tools CP-SAT at its default parameters | how this implementation of the jump-based algorithm fares against CP-SAT as it ships, whose `fj`/`ls` workers are the reference implementation of the same algorithm. Pure MILP, so no expressiveness claim |
| 2 | MINLPLib | `cbls_minlplib` | SCIP (nonlinear) | non-convex MINLP — the regime CP-SAT's LS worker cannot express |
| 3 | UC-CHPED | `cbls_uc_chped` | Pedroso et al. 2014 | a published unit-commitment formulation whose valve-point `\|d·sin(e·(Pmin−P))\|` term is likewise inexpressible in CP-SAT |

A benchmark earns a place on that list on one of two grounds: it contains terms
CP-SAT cannot express (rows 2-3), or CP-SAT carries the same algorithm on a
formulation both sides express identically (row 1). Nothing else qualifies — a
model whose faithful form is a MILP belongs to CP-SAT, CPLEX and Gurobi.
Every comparison solver runs at its own default parameters, CP-SAT included:
nothing is restricted or evened out, and the differences are reported, not
corrected for (Simon, 2026-11-01,
[#87](https://github.com/spoorendonk/cbls/issues/87#issuecomment-6558356494)).

The other directories under `benchmarks/` are **not** comparative benchmarks:

| Directory | Runner | What it is |
|-----------|--------|------------|
| `benchmarks/cvrp/` | `cbls_cvrp` | the worked example for `List` variables: CVRPLIB X set, scored as gap to the best-known values, with PyVRP and OR-Tools as context. Unpublished until #204 |
| `benchmarks/setcover/` | `cbls_setcover` | the worked example for `Set` variables: ten OR-Library instances under both a `Set` and a Bool encoding. Its result is a documented limitation, not a comparative claim |
| `benchmarks/chped/` | none | CHPED dispatch model, data and SCIP reference. `examples/chped.cpp` uses the model; `reference_solve.py --uc` is UC-CHPED's SCIP reference |
| `benchmarks/tune/` | none | `xk tune` parameter studies, one per benchmark class |
| `benchmarks/common/` | none | headers and Python helpers the runners share |
| `benchmarks/instances/miplib-fj/` | none | a retired benchmark's instances, kept because tests read them |

Each benchmark's data, rosters and results write-up live in
`benchmarks/instances/<name>/` (UC-CHPED's write-up is
[docs/benchmarks/uc-chped/README.md](docs/benchmarks/uc-chped/README.md)); [docs/benchmarks/README.md](docs/benchmarks/README.md)
is the index.

## Architecture

See [docs/architecture.md](docs/architecture.md) for solver internals: expression DAG, ViolationLS outer loop, Generalised Feasibility Jump, move generation, inner solver, LNS, threading.

## License

[MIT](LICENSE)
