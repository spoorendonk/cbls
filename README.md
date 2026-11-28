# CBLS

Constraint-Based Local Search engine for mixed discrete-continuous optimization.
ViolationLS — guided local search over single- and compound-variable jumps on an
expression DAG, with per-constraint GLS weights carrying the feasibility pressure.

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
- **Delta evaluation**: a one-variable move on a frozen model re-evaluates a topologically sorted cone cached at freeze (up to a memory cap); probes roll back from an undo log
- **Reverse-mode AD**: sparse automatic differentiation for gradient moves
- **Multi-threaded** cooperative portfolio: workers share incumbents through a solution pool and restart stalled searches from it
- **Python bindings** via nanobind

## Build

```bash
cmake -B build
cmake --build build
ctest --test-dir build    # 900 C++ tests with -DCBLS_BUILD_PYTHON=ON, 899 without (add -LE slow for the fast 888, ~12s)
```

CMake 3.15 or newer (3.20 for `ctest --test-dir`), a C++17 compiler and zlib
are required; bzip2 is
optional (`-DCBLS_USE_BZIP2=ON` reads `.mps.bz2`). The build fetches
nlohmann/json, and Catch2 for the tests, so the first configure needs network.

The build type defaults to `Release`; pass `-DCMAKE_BUILD_TYPE=Debug` to override
it. Install `ccache` and CMake will use it as a compiler launcher automatically,
which makes a clean rebuild of the same directory near-instant.

Two off-by-default measurement builds, each into its own directory:
`-DCBLS_PROFILE=ON` adds symbols and frame pointers on top of Release, and
`-DCBLS_SANITIZE=address,undefined,float-cast-overflow` builds under ASan/UBSan.
See [docs/profiling.md](docs/profiling.md) for the heap and CPU profiling
recipes that use them.

With Python bindings (the venv first: it holds the tools and dependencies, not
cbls, and the tests import the module the build puts in `build/python/cbls`):

```bash
uv sync --extra dev
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build build
.venv/bin/pytest          # 1813 tests, 294 of them for the bindings
```

Tests of the benchmark baselines skip themselves when `ortools` or `pyscipopt` is
absent, so install the extra before treating a green run as complete:

```bash
uv sync --extra dev --extra benchmarks   # names both: uv sync drops an unnamed extra
CBLS_REQUIRE_BENCHMARKS=1 .venv/bin/pytest   # turns that skip into an error
```

To use CBLS from another Python project (Python 3.12+), install it as a package.
pip compiles the extension, so it needs CMake 3.15+, a C++17 compiler and zlib:

```bash
pip install git+https://github.com/spoorendonk/cbls   # or `pip install .` in a checkout
```

The wheel holds the `cbls` Python package only. The C++ library, headers and
`cbls` binary come from a CMake install (below). `ctest -R wheel_smoke` builds a
wheel, installs it into a fresh venv and solves a model there.

## Use from Python

```python
import cbls

m = cbls.Model()
x = m.Float(0, 10, "x")
y = m.Float(0, 10, "y")
m.add_constraint(x + y >= 5.0)
m.minimize(x**2 + y**2)
m.close()

result = cbls.solve(m, time_limit=5.0)  # threads=0: one portfolio worker per core
print(result.feasible, result.objective, result.value(x), result.value(y))
```

`cbls.solve` takes the CLI's options and defaults (`time_limit=10.0`, `seed=42`,
`threads=0`, `lns=None`, `intensify=False`) plus a `SolveCallback`, a
`StopToken` and a `SearchConfig`, and leaves the model at the best point found.
Models also come from files: `load_model`/`save_model` (`.cbls`),
`mps_to_model(read_mps(path))` and `nl_to_model(read_nl(path))`, whose results
carry the model as `.model`. The public names are `cbls.__all__`, typed by the
shipped stub; the engine internals stay in `cbls._cbls_core`.
`examples/simple.py` is the Python twin of `examples/simple.cpp`.

## Use from C++

Either way, link `cbls::cbls`. It carries the include path, C++17 and the
static library's link dependencies (zlib, threads, bzip2 when built with it).

Against an installed copy:

```bash
cmake -B build -DCBLS_BUILD_TESTS=OFF -DCBLS_BUILD_BENCHMARKS=OFF
cmake --build build
cmake --install build --prefix /opt/cbls
```

```cmake
find_package(cbls 0.1 CONFIG REQUIRED)   # -DCMAKE_PREFIX_PATH=/opt/cbls
target_link_libraries(my_app PRIVATE cbls::cbls)
```

From source, with FetchContent (or `add_subdirectory` on a checkout):

```cmake
include(FetchContent)
FetchContent_Declare(cbls
    GIT_REPOSITORY https://github.com/spoorendonk/cbls.git
    GIT_TAG main)   # pin a commit
FetchContent_MakeAvailable(cbls)
target_link_libraries(my_app PRIVATE cbls::cbls)
```

As a subproject cbls builds the library and the `cbls` CLI only: tests,
examples, benchmarks and install rules default to OFF (`-DCBLS_INSTALL=ON` if
the parent exports an installed target that links cbls), and it neither sets the build
type nor touches the parent's git config. It compiles at the parent's build type, so set
one (`-DCMAKE_BUILD_TYPE=Release`): an unset one compiles cbls unoptimized.

The library is static only (`BUILD_SHARED_LIBS` does not apply) and 0.x: the
version check accepts 0.1.x for `0.1`, and nothing promises source or ABI
compatibility between releases. The supported API is what `<cbls/cbls.h>`
brings in (the model, `solve`, the portfolio, MPS reading, ...), plus
`<cbls/io_nl.h>` for AMPL `.nl` models. Headers outside that set, such as
`structural_batch.h` and `arg_parse.h`, are installed for this repository's own
runners and tests and can change without notice.
`tests/consumer/` builds both ways in the test suite (`ctest -R consumer_`).

## Scope

CBLS is built for small-to-medium mixed discrete-continuous models whose terms an
exact MIP or CP solver cannot take as written: non-convex, transcendental or
discontinuous ones. GLS reweighting reshapes the landscape on stagnation, and
diversification kicks or LNS restart the search from a perturbed copy of the
point it is exploring around — its own incumbent, a peer's state under the
multi-threaded portfolio, or the live assignment until the first feasible
solution. How it fares against CP-SAT, SCIP and published results is what the
benchmarks below measure.

## Known limitations

- Local search, with no LP relaxation and no cutting planes, so no bound and no proof of optimality or infeasibility. Propagation is limited to the linear rows: implied variable bounds at MPS and `.nl` load (on by default). Propagation inside compound moves and a load-time presolve with postsolve did not help on [#331](https://github.com/spoorendonk/cbls/issues/331)'s dev rosters and were removed (#380, in git history at `57cdbfba`; #379, at `b10da9d2`).
- Tightly coupled equality rows are slow to satisfy with single-variable jumps. The removed presolve (#379), which included doubleton-equality aggregation, did not fix this on #331's dev rosters (measured as a whole; aggregation was not run alone). On 22 MIPLIB instances its pooled primal-integral ratio was 0.93, inside the 1.30 noise floor. On 25 MINLPLib instances it solved one instance fewer at one seed (23 against 24 of 25; ex8_3_2, whose residuals are on equality rows), and ex8_3_4 stayed infeasible at two seeds in every arm.
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

### What the benchmarks show

The full tables, each with its engine commit, run record and caveats, are in
[docs/results.md](docs/results.md).

- **MIPfeas:** CP-SAT at its defaults is ahead. On 211 held-out MIPLIB instances at 60 s, it finds a feasible point on 175 against cbls's 140, at a primal integral of 0.168 against 0.390 (engine `8246efd`, since moved on).
- **MINLPLib:** SCIP at its defaults is ahead. On Mittelmann's 195 nontrivial instances at 60 s, cbls's GM(gap+1) is 1.68 times SCIP's, and it solves 169 against 187. On MINLPLib's open instances it improves the best known on 3 by more than a tolerance can buy (engine `02c78ad`).
- **UC-CHPED:** at equal time and threads, cbls has the lowest primal integral on 23 of 24 rows. Pedroso et al.'s method ported to SCIP has the best or tied final gap on 21, and cbls alone on the 3 largest (ucp100 T=168, ucp200 T=48 and T=168), where the reference is cbls's own result. cbls ends 0-6.5 % above the best known bounds on the 20 short rows and 6.4-7.8 % on the 4 long ones (engine `2eb27278`). The long rows are one cbls seed, the rest three.
- **CVRP:** not run, so no `List` result.

The other directories under `benchmarks/` are **not** comparative benchmarks:

| Directory | Runner | What it is |
|-----------|--------|------------|
| `benchmarks/cvrp/` | `cbls_cvrp` | the worked example for `List` variables: CVRPLIB X set, scored as gap to the best-known values, with PyVRP and OR-Tools as context. Its full campaign (#204) has not been run, so it has no published result |
| `benchmarks/setcover/` | `cbls_setcover` | the worked example for `Set` variables: ten OR-Library instances under both a `Set` and a Bool encoding. Its result is a documented limitation, not a comparative claim |
| `benchmarks/common/` | none | headers and Python helpers the runners share |
| `benchmarks/instances/miplib-fj/` | none | a retired benchmark's instances, kept because tests read them |

Each benchmark's data, rosters and results write-up live in
`benchmarks/instances/<name>/` (UC-CHPED's write-up is
[docs/benchmarks/uc-chped/README.md](docs/benchmarks/uc-chped/README.md)); [docs/benchmarks/README.md](docs/benchmarks/README.md)
is the index.

## Architecture

See [docs/architecture.md](docs/architecture.md) for solver internals: expression DAG, ViolationLS outer loop, Generalised Feasibility Jump, move generation, inner solver, LNS, threading.

## License

[MIT](LICENSE), except [`benchmarks/uc-chped/pedroso/`](benchmarks/uc-chped/pedroso/README.md),
which is GPL-3.0-or-later as a port of Pedroso et al.'s GPL code. Nothing outside
that directory imports it.
