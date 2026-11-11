# CBLS Solver Architecture

Constraint-Based Local Search: **ViolationLS** (Davies, Didier, Perron,
"ViolationLS: Constraint-Based Local Search in CP-SAT", CPAIOR 2024) over an
expression DAG. A Generalised Feasibility Jump (GFJ) engine drives the
assignment toward feasibility under Guided Local Search (GLS) constraint
weights; the objective is folded in as a soft constraint `obj <= bound` whose
bound is tightened on each new feasible solution. Continuous variables are
polished by a gradient-based inner solver; stagnation triggers perturbation or
large-neighborhood-search diversification.

## Table of Contents

1. [Overview](#overview)
2. [Source Map](#source-map)
3. [Expression DAG](#expression-dag)
4. [Model](#model)
5. [Violation & GLS Weights](#violation--gls-weights)
6. [Generalised Feasibility Jump](#generalised-feasibility-jump)
7. [Novelty Jump](#novelty-jump)
8. [Structural Batch](#structural-batch)
9. [ViolationLS Outer Loop](#violationls-outer-loop)
10. [Inner Solver (Continuous Intensification)](#inner-solver-continuous-intensification)
11. [Large Neighborhood Search](#large-neighborhood-search)
12. [Solution Pool & Parallel Search](#solution-pool--parallel-search)
13. [Design Decisions](#design-decisions)
14. [Control Flow Diagram](#control-flow-diagram)
15. [Parameters Table](#parameters-table)
16. [I/O & Logging](#io--logging)
17. [Threading & Determinism](#threading--determinism)
18. [GPU](#gpu)
19. [Reference](#reference)

---

## Overview

CBLS is a hybrid metaheuristic/mathematical-programming solver for constrained
optimization over mixed discrete-continuous variables. The core algorithm
follows ViolationLS (CP-SAT's CBLS worker), adapted to a nonlinear expression
DAG:

1. **Models** problems as an expression DAG with typed variables (Bool, Int,
   Float, List, Set) and nonlinear constraints/objectives.
2. **Folds the objective into the constraint set** as a soft constraint
   `obj <= bound`. The bound starts at `+inf` (inert) and is tightened every
   time a strictly better real-feasible solution is found. Optimization thus
   reduces to a sequence of feasibility problems.
3. **Searches** with batches of **Generalised Feasibility Jump** — a GLS-driven
   best-of-N greedy that repeatedly applies the single-variable "jump" that most
   reduces the weighted constraint violation, bumping per-constraint weights on
   stagnation. Batches may instead be **Novelty Jump** (bounded-backtracking
   compound moves) or **Structural** (typed moves over List/Set variables).
4. **Intensifies** continuous variables via a gradient-based inner solver
   (Newton steps on violated constraints, backtracking line search on the
   objective, multi-variable minimum-norm Newton), triggered **on each new
   feasible solution**.
5. **Diversifies** on stagnation, from the point the search is exploring around
   rather than from wherever the last kick left it (#158): a random perturbation
   of the scalars plus a run of random structural moves per List/Set variable,
   which always moves at least one variable, or — every `lns_interval`-th
   diversification kick — large neighborhood search (destroy + GFJ repair).
6. **Tracks** the best real-feasible solution found (objective bound tightened
   alongside it), with a solution pool for parallel multi-seed search.

There is no temperature, no Metropolis acceptance, no global penalty multiplier
`lambda`. Feasibility pressure comes entirely from the per-constraint GLS
weights `W`, and progress is greedy on weighted violation. The penalty-method
metric `obj + total_violation()` survives only as the *inner solver's* local
descent objective (`ViolationManager::augmented_objective`), not as the search's
acceptance rule.

---

## Source Map

A one-screen map of the pipeline, its files, the extension points, build targets and dependencies. The sections after it are the full account; defaults are in the [Parameters Table](#parameters-table).

### Core pipeline

1. **Model building** (`include/cbls/model.h`, `include/cbls/expr.h`) — typed variables (Bool, Int, Float, List, Set), expressions by operator overloading, constraints, an objective; `close()` sorts the DAG topologically.
2. **Expression DAG** (`include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`) — variables have negative handles `-(id+1)`, nodes non-negative ids; 35 `NodeOp` types. `full_evaluate` evaluates everything; `delta_evaluate` recomputes only the nodes a move reaches (see [Evaluation Modes](#evaluation-modes), including the incremental Sum's drift bound). Slopes for the continuous jump values come from a forward up-cone pass (`UpConePartials`), or reverse-mode AD (`compute_all_partials`) for a large cone.
3. **Search** (`src/search.cpp`) — the ViolationLS batch outer loop (Davies et al. CPAIOR 2024, Algorithm 6): each batch is Feasibility Jump, Novelty Jump or Structural, chosen by config probabilities; the `obj <= bound` row tightens on each new real-feasible solution; stagnation perturbs or runs LNS.
4. **Feasibility Jump** (`src/feasibility_jump.cpp`, `src/linear_jump.cpp`) — a `JumpTable` of cached best jumps (score `-W·δ_G`), best-of-N sampling, GLS weight dynamics, and Novelty Jump compound moves (Algorithms 4–5). Float candidates come from Newton steps toward violated roots; a variable that reaches a `Ceil`/`Floor`/`Round`/`Element` node also gets its breakpoints; all-linear columns are scored in closed form by `LinearJumpScorer`, others by `weighted_violation_delta`. The default-off arms are listed in the [Parameters Table](#parameters-table).
5. **Moves** (`src/moves.cpp`) — typed move generators (bool flip, int ±1/rand, float perturb, list and set moves, the inter-list moves of a `ListPartition`). Scalar moves are subsumed by FJ; list/set moves feed the Structural batch as positional `ElementEdit`s. See [Structural Batch](#structural-batch).
6. **Inner solver** (`src/inner_solver.cpp`) — `FloatIntensifyHook`: coordinate descent over floats (Newton on violated rows, line search on the objective), run on each new feasible solution.
7. **LNS** (`src/lns.cpp`) — every N-th diversification kick: SISR ruin-and-recreate on each `ListPartition`, destroy 30% and repair by FJ elsewhere; accepted on a lexicographic (real violation, objective) key.
8. **Violation & penalty** (`src/violation.cpp`) — `total_violation = Σ W[c]·max(0, viol_c)`; `weighted_violation_delta` is the no-commit counterfactual δ_G; `augmented_objective()` is the inner solver's descent metric.
9. **Parallel search** (`src/pool.cpp`) — `SolutionPool` + `ParallelSearch`, a cooperative portfolio that shares incumbents and the objective bound. Workers share the frozen model's immutable structure (#157) and copy only what a search writes. Nondeterministic; `--threads 1` is the reproducible run. See [Solution Pool & Parallel Search](#solution-pool--parallel-search); the benchmark driver's CPU policy is in `benchmarks/instances/mipfeas/README.md`.
10. **I/O** (`src/io.cpp`, `src/io/`) — JSONL `.cbls` format; the MPS reader (`mps_reader.cpp` + `mps_to_model.cpp`, gzip, optional bzip2), the AMPL `.nl` reader (`nl_reader.cpp` + `nl_to_model.cpp`) and the MIPLIB `.solu` reader (`solu_reader.cpp`). CLI in `src/cli.cpp`. Load-time reduction for the adapters: implied bounds (`src/bound_propagation.cpp`); see [Implied variable bounds](#implied-variable-bounds). Presolve was removed ([Presolve](#presolve)).

### Key extension points

- **`InnerSolverHook`** — subclass for domain-specific continuous optimization; `FloatIntensifyHook` (`include/cbls/inner_solver.h`) is the reference. Its `solve()` gets a `StopRef stop` and **must** poll it if its work can outlast the budget, and return leaving every committed value delta-evaluated: the call cannot be pre-empted.
- **New operations** — add to `NodeOp` in `dag.h`, implement `evaluate()` in `dag.cpp`, `local_derivative()` for AD, and `delta_evaluate` support in `dag_ops.cpp`.
- **`CustomInvariant`** (`include/cbls/custom_invariant.h`, #166) — user code inside the DAG via `Model::custom(inputs, std::move(inv), name)`: optional incremental `delta`, a `commit`/`rollback` bracket, `clone()`d per portfolio worker. Not writable to `.cbls`, not exposed to Python (#132). Contract in footnote ² of [Expression Nodes](#expression-nodes), after the AD section.

### Build targets

| Target | Source | Description |
|--------|--------|-------------|
| `cbls_lib` | `src/*.cpp`, `src/io/*.cpp` | Core library (static) |
| `cbls_cli` | `src/cli.cpp` | CLI executable (output name `cbls`) |
| `cbls_tests` | `tests/*.cpp` | Catch2 test suite |
| `cbls_simple` | `examples/simple.cpp` | Minimal modeling example |
| `cbls_chped` | `examples/chped.cpp` | CHPED modeling example over `benchmarks/chped/` |
| `cbls_uc_chped` | `benchmarks/uc-chped/` | UC-CHPED benchmark runner |
| `cbls_mipfeas` | `benchmarks/mipfeas/` | MIPfeas runner (one instance per process) |
| `cbls_minlplib` | `benchmarks/minlplib/` | MINLPLib benchmark runner |
| `cbls_minlplib_nl_check` | `benchmarks/minlplib/nl_check.cpp` | reads and models NL files, no solve; per-instance status for `status.py` (#283) |
| `cbls_setcover` | `benchmarks/setcover/` | OR-Library set-covering runner (the `Set`-variable example) |
| `cbls_cvrp` | `benchmarks/cvrp/` | CVRPLIB X-set CVRP runner (the `List`-variable example, one instance per process) |

Examples, tests, benchmarks and bindings are each behind an option:
`CBLS_BUILD_EXAMPLES`/`CBLS_BUILD_TESTS`/`CBLS_BUILD_BENCHMARKS` default ON,
`CBLS_BUILD_PYTHON` defaults OFF (but the gated build turns it on).

### Dependencies

- **zlib** — required (`find_package(ZLIB REQUIRED)`), gzip-compressed MPS input
- **bzip2** — optional, `-DCBLS_USE_BZIP2=ON`, adds `.bz2` MPS support
- **nlohmann/json** v3.11.3 — JSONL I/O (FetchContent)
- **Catch2** v3.5.2 — C++ tests (FetchContent)
- **nanobind** — Python bindings (optional, via scikit-build-core)

---

## Expression DAG

**Files:** `include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`

### Variable Types

```
enum class VarType : uint8_t { Bool, Int, Float, List, Set };
```

Each `Variable` stores: `id`, `type`, `value` (scalar), `lb`/`ub` (bounds),
`elements` (for List/Set), `universe_size`, `min_size`/`max_size` — for both
structured types `elements` is drawn from `{0..universe_size-1}` and its size
stays within `[min_size, max_size]` — plus,
for a List, `list_init` and `partitioned`. It owns no edge storage: the nodes
that use it are
`Model::dependents(id)`, a slice of one flat CSR array the model holds.

Bool, Int and Float are *scalar* (jumpable by GFJ). List and Set are
*structural* — GFJ leaves them untouched; they are moved only by the
[structural batch](#structural-batch) and LNS.

> **Neither structural type has a positive result behind it.** "Generalises to
> structured variables" is a claim about modelling reach only. `Set` is
> measured and negative against one Bool per column
> ([structure-only models](#structure-only-models),
> `benchmarks/instances/setcover/README.md`); `List` has no evidence yet, and
> the CVRP benchmark (`benchmarks/cvrp/`) is where #204 will produce it.

### Handle Encoding

Variables and expression nodes share a single `int32_t` handle space:

- **Variables**: negative handles, `handle = -(var_id + 1)`
- **Nodes**: non-negative handles, `handle = node_id`

`Model::wrap(handle)` decodes: if `handle < 0`, it refers to variable
`-(handle + 1)`; otherwise it refers to node `handle`.

### Expression Nodes

Each `ExprNode` has an `op` (operation), a `(child_begin, child_count)` slice
of the model's flat `ChildRef` array (`id` + `is_var` flag, read through
`Model::children(node)`), and optionally `const_value` or `lambda_func_id`. Its
back-references are `Model::parents(id)`, a CSR slice rebuilt with the
topological order. No node or variable owns a heap block of its own (#156).

A node carries **no cached value**: the evaluation result lives in the model's
own `node_values()` array, read through `Model::node_value(id)` (#157). That is
what lets the whole node array sit in the immutable structure portfolio replicas
share — a `value` field there would be one worker's search state in storage every
worker reads. `ExprNode` is 32 bytes as a result, and the per-worker cost of a
node is its 8-byte value. Each incremental Sum (#188, below) adds 16 bytes
of per-model state, which is roughly one per row on a MIP.

**Supported operations:**

| Category     | Operations                                           |
|------------- |------------------------------------------------------|
| Arithmetic   | Const, Neg, Sum, Prod, Div, Pow                      |
| Aggregation  | Min, Max, Abs                                        |
| Trigonometric| Sin, Cos, Tan                                        |
| Exponential  | Exp, Log, Sqrt, SignPower (`sign(x)·abs(x)^p`), Tanh |
| Conditional  | If                                                   |
| Rounding     | Ceil, Floor, Round (#186³)                           |
| Table lookup | Element: `table[i]` or `table[i][j]`, each index any scalar, truncated toward zero (#186³) |
| Collection   | At (indexing), Count, Lambda, PairLambda (functional aggregation¹), LambdaExtra, PairLambdaExtra (the same, with a functor that also reads scalar `extra` children, #186³) |
| Comparison   | Leq, Eq, Geq, Neq, Lt, Gt                           |
| User code    | Custom (a `CustomInvariant`², #166)                  |

35 enumerators in all. The six #186 ones are appended after `Custom`; SignPower
and Tanh (for MINLPLib) sit before `If`. Saved models name ops rather than
number them, so the order is not a file format.

Comparison nodes evaluate to a **violation measure** (0 when satisfied,
positive when violated):
- `Leq(a, b)` = `a - b` (satisfied when <= 0)
- `Eq(a, b)` = `|a - b|`
- `Geq(a, b)` = `b - a`
- `Lt(a, b)` = `a - b + epsilon`
- `Gt(a, b)` = `b - a + epsilon`

`Leq`/`Geq`/`Lt`/`Gt` resolve one `inf - inf` case specially (`comparison_residual`
in `dag.h`, issue #100): when both sides are infinite with the same sign *and* the
side that would make the row vacuous is a literal `Const` node, the residual is
`0.0` (satisfied) rather than NaN. That is the "absent bound" idiom — `a <= +inf`,
`-inf <= b` — and it is the state every solve opens in, since the folded-in
`obj <= bound` row starts with `bound = +inf`. An infinity an *expression*
computed is an overflow, not a sentinel, so it keeps the NaN and stays maximally
violated (`exp(1000) <= exp(720)` is genuinely violated). `Eq`/`Neq` deliberately
keep plain `|a - b|`.

**Domain errors are NaN (#205).** `Sqrt`/`Log` of a negative, a `Pow` whose
`std::pow` is NaN (`pow(-8, 1/3)`) and `0/0` evaluate to NaN, which every row
reads as maximally violated (`clamped_node_violation`). `Log(0)` is `-inf`, a
`Pow` overflow keeps its sign, and a `Div` whose denominator is below `1e-15` in
magnitude is an infinity signed by `num * denom` (`1/-0.0 = -inf`), or NaN when
the numerator is 0 -- below the threshold the denominator is a zero, so such a
`Div` is never finite, which `LinearJumpScorer` relies on to keep it off the
closed form. Reading them as ordinary numbers would let an infeasible model
report feasible. The ops that consume a child also keep a NaN NaN instead of
turning it back into a defined value: `Min`/`Max`, `Neq`, an `If` condition,
and an `At` or `Element` index (out of range still reads `0.0`).

**Implicit domain rows (#248).** A NaN reads as the same flat violation at every
point outside the domain, so it gives the search no way back in. `close()`
therefore appends a hidden row `arg >= 1e-9` (`kLogDomainFloor`, SCIP's
`expr/log/minzerodistance`) per `Log` argument and `arg >= 0` per `Sqrt` and per
`Pow` with a constant non-integer exponent (`Model::append_domain_rows`); a
negative such exponent has a pole at 0, so it gets the `Log` floor. Each RHS is
the bare floor. A row within the tolerance counts as satisfied, so `x >= 1e-9`
read plainly would hold at `x = 0`. A violated domain row therefore adds
`Model::domain_margin()`, the smallest double above the solve's
`feasibility_tolerance` (`solve()` and the `FeasibilityJump` constructor set it
through `set_domain_tolerance`). An argument below its floor then reads
infeasible at that tolerance, and one at the floor reads 0 (#258).
`verify_model`, `ViolationManager::is_feasible`/`violated_constraints` and
`lns_acceptance_key` hold domain rows to 0, not to their `tol`. The step is not continuous, so Newton
steps (FJ's and the inner solver's) aim at the floor with the step taken off,
the closed-form linear scorer takes a domain row only over a bare variable
(computing its residual as `evaluate` does) and keeps it out of the convex Int
argmin where the range straddles the floor, and the incremental Sums' drift
checks judge a domain row's verdict at 0. They sit after the modeller's rows in `constraint_ids()` and before the objective
row (`num_user_constraints()`, `num_domain_rows()`), count in `verify_model`
and the feasibility test, and are left out by `save_model`. None is made for an
op no row or objective reads, for one read only through an `If` branch, for a
Const argument, or for a variable whose lower bound already meets the floor.

### Evaluation Modes

**Full evaluation** (`full_evaluate`): evaluates all nodes in topological
order. Used at initialization, after perturbation/LNS, and after the
objective-bound soft constraint is appended.

**Delta evaluation** (`delta_evaluate`): given a set of changed variable IDs,
recomputes only the nodes they reach, in topological order. A one-variable walk
on a frozen model reads the variable's cached cone (**Cone cache** below); any
other walk BFS-marks dirty nodes upward through
`Model::dependents`/`Model::parents` and sorts them. This is the hot path during
GFJ — each jump changes one variable and touches a small subgraph, and each
jump *candidate* is scored by a no-commit delta probe (see
[`weighted_violation_delta`](#violation--gls-weights)) unless every weighted
row of the variable's column is linear (an affine comparison or, since #190,
an affine bare body), where `LinearJumpScorer`
(`include/cbls/linear_jump.h`) scores it in closed form instead. A committed
jump goes through `commit_scalar_move` -- `delta_evaluate`'s walk, also told the
variable's old value (below).

**Cone cache** (#271). On a frozen model a one-variable walk -- every FJ probe
and commit -- neither marks nor sorts its cone: `Model::freeze()` collected each
variable's cone once and stored it sorted by `topo_pos`
(`ModelStructure::cone_offsets`/`cone_ids`, CSR, shared by every replica).
`topo_pos` is a total order, so the cached order is the one the walk sorts to
and every value is bit-identical. A model with no incremental Sum and no custom
node sets no dirty flag at all on that path; the others set the cone's flags
from the span, since `collect_changed_inputs` and the incremental-Sum pushes
read them. The cache costs 4 bytes per (variable, cone node) pair, O(nodes ×
vars) on a deep shared chain, so it is capped (`SearchConfig::cone_cache_bytes`,
256 MiB): variables are cached in id order up to the first whose cone would
cross the cap, and that one and every later one walk as an unfrozen model's do
(a prefix, so the build is O(cap), not O(vars × cap)). A walk over several
variables, an unfrozen model's, and one past the prefix collect and sort as
above.
`SearchCounters::cone_cache_walks`/`cone_fallback_walks` give the hit rate;
`tests/test_cone_cache.cpp` pins the bit-identity against an unfrozen twin.
`solve()` freezes the model it is handed, so the single-thread path gets the
cache too; a portfolio factory that builds a fresh model per call builds one
cache per worker.

**Incremental Sum** (#177, #188). A committed FJ move changes one term of each
row it touches, and re-summing a row costs its whole length (36% of swath3's
run in #177). `commit_scalar_move` therefore moves each *incremental* `Sum` by the
differences of its changed terms instead of re-summing it.

- *Incremental* (`ExprNode::kIncSum`, set at `close()` by
  `Model::classify_incremental_sums`). No term is a Sum, no term appears
  twice, and the Sum is read only by top-level comparisons
  (`Leq`/`Geq`/`Lt`/`Gt`/`Eq` with no parent) against a variable or a Const.
  It is not itself a constraint root (#192): FJ finds a row's Sum through the
  row's children, so a Sum that is its own row would drift ungated.
  That covers every MPS row and the objective row. Because a row reads its
  Sum directly, FJ can bound what drift does to the row. A Sum under any other
  node keeps the plain re-sum. Each Sum's per-model state (`IncSumState`,
  16 bytes) is found through its slot in `lambda_func_id`.
- *Exactness, per update*. TwoSum checks `new - old` and the add.
  - An exact update adds no drift. On integral data within 2^53 every update
    is exact, so the values are the re-sum's to the bit, as under #177's
    exact-only rule.
  - An inexact update adds its two rounding errors, computed exactly, to the
    Sum's `drift_bound`.
  - A checked re-sum sets the bound to its own rounding errors, again
    computed exactly. That is 0 when the re-sum was exact and never more than
    gamma_{n-1} * sum(|t_i|).

  The bound therefore always bounds |value - real sum of the stored terms|.
  A Sum is *drifting* once it has been updated with a nonzero bound. That
  covers inexact updates, and also a re-sum's rounding carried forward by
  exact ones, which a re-sum of the new terms would not make.
- *Containment*.
  - A Sum carries at most `kIncSumPeriod - 1` = 63 inexact updates; the next
    one re-sums it instead. The period is a safety cap on how far the bound
    can grow between re-groundings, not a value shown to be better than the
    alternatives: a pre-registered rule picked it on held-out instances, where
    it cost 0.67% of throughput against never re-summing
    (`docs/benchmarks/incremental-sum-drift.md`).
  - A walk that does not know the old values (Novelty Jump's legs, the inner
    solver's multi-variable pass, the structural batch) re-sums plainly and leaves the Sum
    untracked. So does `full_evaluate`. The next commit then re-sums it,
    checked.
  - A Probe applies the commit's own updates, and logs its whole cone's values
    before its first write (`ProbeUndo`, #270). The Rollback writes the log
    back, Sums included, to the bit, without walking the cone again.
  - A model with no incremental Sum and no custom node evaluates its cone
    with plain `evaluate()` and builds no incremental-Sum rules; its Probe
    still arms the log, and its Rollback writes it back like any other.

FeasibilityJump acts on no verdict that drift could have flipped:

- `update_var` notes any row whose residual lies within its Sum's bound, plus
  the comparison's own rounding, of the tolerance (`GFJConfig::tolerance`), on either side.
- At a local minimum only the noted rows' Sums are re-summed, before any GLS
  bump or Feasible verdict. A row violated only by drift leaves V instead of
  being bumped. A row satisfied only by drift enters V and is bumped with the
  rest. If either happened, the iteration samples again.
- Every Feasible verdict (the loop's, the batch limit's, Novelty Jump's) and
  every batch end first re-ground all drifted Sums (`reground_drifted_rows`).
  No drift leaves a batch, and every Feasible verdict is read off re-summed
  rows.
- On integral data nothing is ever noted, so the gate costs one empty-list
  test.

**What "exact" means here.** A verdict on a drifting Sum is decided against
the real sum of its stored terms, within the bound; a Sum just re-summed is
decided on the re-sum's value, as every row was before #177. So "no phantom
bump" and "every Feasible verdict exact" hold relative to a re-sum, not to
exact real arithmetic.

The design's deviations from #188's proposal (an exact re-sum bound, no margin
factor), its A/B on #177's roster and its MINLPLib no-regression checks are in
`docs/benchmarks/incremental-sum-drift.md`.

### Reverse-Mode Automatic Differentiation

`compute_partial(model, expr_id, var_id)` computes `d(expr)/d(var)` via
reverse-mode AD; `compute_all_partials(model, expr_id)` returns every variable's
partial in one reverse pass, and `compute_partials_sparse(model, expr_id, out)`
only the nonzero ones, as `(var_id, partial)` pairs:

1. Initialize `adjoint[expr_id] = 1.0`
2. Traverse the **cone** of `expr_id` (the nodes reachable from it through
   children) in reverse topological order, skipping any node whose adjoint is
   0. A cone larger than `|nodes| / (log2|nodes| + 1)` falls back to the whole
   `topo_order()`; both routes perform the same operations in the same order,
   so the partials are bit-identical either way (`reverse_sweep` in
   `src/dag_ops.cpp`).
3. For each node, propagate: `adjoint[child] += adjoint[node] * local_derivative(node, child_index)`
4. Adjoints live in one flat per-thread scratch: `[0, num_nodes)` for nodes,
   `num_nodes + var_id` for variables. The three entry points share it, so none
   may be called from inside another -- in particular not from a
   `CustomInvariant::partial`; the nested call throws `std::logic_error`.

`local_derivative` computes per-operation partial derivatives (chain rule
components). Discrete operations (At, Count, Lambda, PairLambda, LambdaExtra,
PairLambdaExtra) and the piecewise-constant ones (Element, Ceil, Floor, Round)
return 0 — FJ gets the latter's plateaus from their breakpoints instead (see
[`compute_var_jump`](#compute_var_jump)). A
`Custom` node asks its invariant's `partial`, whose NaN *or infinity* ("unknown")
reads as 0 here — the same answer, so a custom node with no derivative is exactly
as differentiable as a structural op. An infinity has to be folded too, not just
propagated: the chain rule would multiply it into the partial of every variable
below that input, and two such paths meeting with opposite signs read
`inf - inf` — NaN. Every propagation pass, reverse and forward, takes its edges
through `add_chain_term` (`include/cbls/dag_ops.h`), which skips a zero edge
rather than multiplying it, so an infinite adjoint or slope never reads
`inf * 0` across an untaken If branch or a Min's unselected operand (#260).

The built-ins follow the same rule (#205): `Pow`, `SignPower`, `Div` and
`Tan` fold a non-finite partial to 0, `Log`/`Sqrt` return 0 on and below
their singular point, and an `If` on a NaN condition gives neither branch a slope. No op offers a finite slope where its *value* is NaN — a
Newton step built from one would aim into a region where the function is not
defined. What stays finite but huge near a pole (`1/x` at `x = 1e-15`) is left
alone; the rule is at the site in `src/dag.cpp`.

AD is used to generate Newton-toward-root jump candidates for Float variables
(in `compute_var_jump`) and by the inner solver.

¹ **Lambda serialization:** Lambda nodes store a C++ `std::function`, which
cannot be serialized directly. `save_model` tabulates the function over its
input domain and writes the resulting table. `load_model` reconstructs an
equivalent Lambda via table lookup, so round-tripping through JSONL is lossless
for finite-domain Lambda nodes. A PairLambda is tabulated the same way, as an
`n x n` matrix, plus the keys that carry its variant: `"mode": "cyclic"` for the
closing pair and `"head"`/`"tail"` for the fixed-endpoint tables. Each is
written only when it says something; the reader defaults an absent key to the
open chain with no endpoints.

² **Custom nodes: user code in the DAG (#166).**
`Model::custom(inputs, std::unique_ptr<CustomInvariant>, name)` appends a node
whose value is whatever that object says it is. It is an ordinary node
afterwards: usable inside an expression, as the objective, or as a constraint
body. The inputs are ordinary handles — variables or nodes, scalar or
structured, in any mix — and reach the invariant through `InvariantInputs`,
a view over the node's children (`value(i)`, `elements(i)`,
`is_structured_input(i)`).

**Where inside a structured input: positional edits (#172).** `changed` names
the inputs that moved, never *where* inside one. `in.edits(i)` is the where: for
a List/Set variable input, the ordered `PositionalEdit`s (Swap, Reverse,
MoveSegment, Insert, Erase, Assign — each with the element an Erase or Assign
displaced) that took it from the elements the invariant last **committed** to to
the ones `elements(i)` returns now. With them a delta over a List is O(edits)
rather than O(|elements|); `tests/test_custom_invariant_edits.cpp` carries an
open-route-cost invariant written that way.

*How the edits get there.* The structural batch records each candidate's
positional `ElementEdit`s as they took effect (an edit the range guard ignores
is not recorded, so the record replays and inverts exactly) into an
`EditJournal`, and describes each transition to `delta_evaluate` as *the
previous candidate's edits inverted, then this one's*, since the batch restores
the sample baseline before every candidate. The two restores that close a
sample are described the same way. `delta_evaluate` takes the journal as an
optional last argument; every other caller passes none.

*The fallback is unmistakable.* `edits(i)` returns an `InputEdits`; with no
positional information `available()` is false and `list()` **throws** rather
than returning an empty span that would read as "no changes". It is unavailable
in `evaluate`/`partial`, for a scalar or node input, for any call whose caller
recorded nothing (the diversification kick, LNS, the inner solver, Novelty
Jump, a restore, a direct `var_mut`), for a variable that saw a `Replace`, and
after an exception left a probe open. (A `delta()` that throws on a `Commit`
pass is not detected; `full_evaluate` is the recovery.) An input *not* in
`changed` reports an available, empty list, so a scalar move does not push a
List invariant into a re-read.

*Accumulation and lifetime.* Edits describe the change since the last committed
state: on the batch every leg is a `Commit`; a `Probe` needs no edits on its
`Rollback`; a stale probe withholds them. They are keyed per input, read only
during the pass, and, like `changed`, valid for the one `delta()` call.

*Opt in.* The batch records only while some invariant returns true from
`wants_positional_edits()`, checked once per sweep. Not opting in is still
correct and adds no allocation (#172 measured about 1% per structural
candidate for a custom node, not isolated from code placement); opting in costs
about 6% more and no allocation.

The interface is six calls plus a clone:

| Call | When | Contract |
|---|---|---|
| `evaluate(in)` | `full_evaluate` — `close()`, after `restore_state`, every restart | From scratch. Redefines the committed state. |
| `delta(in, changed)` | `delta_evaluate`, per dirtied pass | Incremental. `changed` is the input indices the engine recomputed since the last committed state — a *superset* of those that actually differ. Defaults to `evaluate(in)`. |
| `wants_positional_edits()` | once per structural sweep | Opt in to `in.edits(i)` being recorded by the structural batch. Default false; false is still correct. |
| `commit()` | after a `delta` whose assignment is kept | The staged state becomes committed. |
| `rollback()` | after a `delta` the caller has undone | Discard the staged state; the engine restores the node's cached *value* itself. |
| `partial(in, i)` | reverse-mode AD | `d(value)/d(input i)`, or NaN **or ±inf** for unknown — both fold to 0, so one infinite edge cannot NaN a sibling's partial. |
| `clone()` | `Model` copy, i.e. one per portfolio worker; entered CONCURRENTLY on the master | Must carry the current state — a copied `Model` needs no `full_evaluate`, so a caller may copy and `delta_evaluate` straight away. Must be safe to call from N threads on one instance: `ParallelSearch` replicates the master on each worker's own thread. |

**Where the instance lives.** On the per-model side of the `ModelStructure`
split: `ExprNode::lambda_func_id` indexes `Model::custom_invariant(id)`, a
per-model `std::vector<CustomInvariantSlot>` that the copy constructor
`clone()`s entry by entry. The *index* is structure and is shared; the
*instance* is not. That is the difference from `lambda_funcs`, which every
replica invokes concurrently and which therefore cannot carry state at all.

**The probe bracket, and what is not bracketed.** `delta_evaluate` takes a
`DeltaMode`: `Commit` (the default), `Probe` and `Rollback`. `Model::weighted_violation_delta` and
`per_constraint_violation_delta` — the per-candidate scalar probe Feasibility
Jump runs for every jump value — are bracketed as `Probe` then `Rollback`, so a
custom node in one of those cones costs **one `delta` and one `rollback`** per
candidate rather than two deltas. **Three** other sites score by evaluating
forwards and evaluating back, and every leg of all three stays `Commit`: the
structural batch (`src/structural_batch.cpp`), the inner solver's
multi-variable Newton step (`src/inner_solver.cpp`; its coordinate probes are
bracketed since #207), and Novelty Jump's backtracking compound-move search
(`src/feasibility_jump.cpp`), which is on the same per-candidate path the
bracketed probe is. All three are correct — each leg really is a new committed
assignment, and the `changed` set each passes is a complete superset of what it
moved — but **twice the cost** of the bracketed path, and an invariant caching a
List's prefix sums updates it on both legs (O(edits) per leg on the structural
batch for an invariant that opts in to positional edits, a re-read otherwise).
Bracketing them is a separate change.

**A model with no custom node is untouched.** `Model::has_custom_nodes()` is
tested once per `delta_evaluate`/`full_evaluate` call, not per node; the false
branch is the plain loop, and the dispatch is a template rather than a
`std::function`, so it inlines the same `evaluate` call. The fence is
`tests/test_structural_equivalence.cpp` and `tests/test_structured_trajectory.cpp`.

**Serialisation refuses.** A `CustomInvariant` is C++ code, not a table of
numbers the way a `Lambda` is, so `save_model` throws before writing its first
line, naming the node and the invariant. Checked before the stream is opened:
`std::ofstream` truncates on open, so a refusal after that point would replace
an existing file with a prefix of a model.

**One contract obligation the engine enforces, and two it only documents.** The
enforced one: none of the five calls may re-enter `delta_evaluate` or
`full_evaluate`. That throws `std::logic_error`, because the nested call clears the
`thread_local` dirty list the outer one is holding, which would leak the outer
call's flags and silently stop those nodes being recomputed for the rest of the
process.

The two that are documented and **not** checked anywhere in the engine. First, a
custom node's value must be a pure function of its declared inputs — a node with
no inputs, or only `Const` ones, never enters a dirty cone, so `delta()` is never
called on it and its value only ever moves at `full_evaluate`. Nothing detects an
impure one; `cbls::verify_model` is a verifier a caller opts into, not a gate, and
an impure invariant otherwise shows up as trajectory divergence. Second,
`Model::restore_state` writes variables only, so a `full_evaluate` before the next
`delta_evaluate` is mandatory. Both are contracts on the caller, and the cost of
breaking either is a wrong answer rather than an exception.

**Not exposed to Python.** A Python-subclassable invariant needs the trampoline
and GIL machinery of #132, so nothing here is bound: `Model::custom` is not
exposed. `NodeOp::Custom` is in the binding enum only so that every op has a
Python name.

³ **Element, rounding, and lambdas that read other decisions (#186):**

| Op | Builder | Semantics | Workaround it replaces |
|----|---------|-----------|------------------------|
| `Element` | `element(table, i)`, `element(table2d, i, j)` | `table[trunc(i)]`; 0.0 when an index is outside the table or non-finite — `At`'s rule for a position past its List. The table is copied into `ModelStructure::element_tables` and written to `.cbls` as data | an if-chain over every index value |
| `Ceil`/`Floor`/`Round` | `ceil_expr`, `floor_expr`, `round_expr` (`ceil`/`floor`/`round` on `Expr`, and `math.ceil`/`math.floor`/`round` in Python) | `std::ceil`/`std::floor`/`std::round`; `round` takes a half-way value away from zero. No tolerance | an auxiliary Int `n` and a row such as `headway * n >= cycle` |
| `LambdaExtra` | `lambda_sum(list, f, extra)` | `sum_{e in list} f(e, [value(x) for x in extra])` | one lambda per value of the decision, behind a selector |
| `PairLambdaExtra` | `pair_lambda_sum(list, f, mode, extra)` | the plain pair sum's open or cyclic chain, `f(a, b, extras)`; no head/tail terms | as above |

The `extra` handles are children of the node, so a change to one re-evaluates it;
any list edit does too. Both re-sum the list, exactly what a plain `Lambda` does
on an edit.

**A list edit is O(|list|), not O(edits)**, for every lambda form (a deviation
from #186): only a `CustomInvariant` gets the positional-edit journal, and an
incremental floating-point lambda sum would reopen the drift that #188 contains
only for a `Sum` a row reads directly. #189 (incremental lambdas) is closed as
not planned until a List benchmark asks for it.

The two lambda forms cannot be written to `.cbls` — the functor takes
continuous values and has no finite table — so `save_model` refuses a model
holding one before it opens the stream, the same refusal a custom node gets. The
builders refuse a List/Set handle where a scalar is read (an index, an extra, a
rounding argument), and an empty, ragged or non-finite table. In Python `extra`
is keyword-only in both forms, because `pair_lambda_sum`'s third positional
parameter is already `cyclic`.

All six have local derivative 0 and are not affine, so a row over one is never
scored by `LinearJumpScorer`. What FJ gets instead is their breakpoints — see
[`compute_var_jump`](#compute_var_jump).

---

## Model

**Files:** `include/cbls/model.h`, `src/model.cpp`

### Building a Model

The `Model` class provides a fluent API for constructing optimization problems:

```cpp
Model m;
auto x = m.int_var(0, 10, "x");     // returns handle -(0+1) = -1
auto y = m.float_var(0.0, 5.0, "y"); // returns handle -(1+1) = -2
auto s = m.sum({x, y});              // returns handle 0 (node)
m.add_constraint(m.leq(s, m.constant(8)));
m.minimize(x);
m.close();
```

**Variable creation**: `bool_var`, `int_var`, `float_var`, `list_var`,
`set_var`. Each returns a negative handle.

`list_var(n)` is a fixed-length permutation of `{0..n-1}`.
`list_var(universe, min_len, max_len, init)` is the general form (#164): an
ordered sequence of *distinct* elements drawn from `{0..universe-1}` whose length
moves within `[min_len, max_len]`, which is the ordered-subset structure a route
or a schedule-with-rejection actually has. `add_list_partition(lists, cover)`
declares that a group of Lists over one universe covers it exactly, or at most
once, and that invariant is maintained **by the moves** rather than by a penalty
row — see [Structural Batch](#structural-batch).

**Expression creation** (`Model` builders): arithmetic (`sum`, `prod`,
`div_expr`, `pow_expr`, `neg`, `abs_expr`, `min_expr`, `max_expr`),
trigonometric (`sin_expr`, `cos_expr`, `tan_expr`), other (`exp_expr`,
`log_expr`, `sqrt_expr`, `signpower_expr`, `tanh_expr`), conditional (`if_then_else`), rounding (`ceil_expr`, `floor_expr`,
`round_expr`), table lookup (`element`), collection (`at`, `count`,
`lambda_sum`, `pair_lambda_sum`, each also with an `extra` form — see footnote
³ above), and comparisons (`leq`, `eq_expr`, `geq`,
`neq`, `lt`, `gt`). Each returns a non-negative node handle.

`pair_lambda_sum` sums a function over a List's consecutive pairs and takes a
`PairMode`: `Open` (the chain alone) or `Cyclic` (plus the closing pair), with
optional `head`/`tail` terms charging the first and last element against a
fixed endpoint — a tour cost and a depot-anchored route cost respectively. The
closing rule and the two endpoint function ids live in a side table
(`ModelStructure::pair_lambda_specs`) keyed by the node's `lambda_func_id`,
rather than in extra `NodeOp` enumerators: every variant evaluates through the
same loop, so a wider dispatch table would buy nothing. The header defines
`n = 0, 1, 2` explicitly.

**Objective.** `minimize(e)` sets the objective node directly; `maximize(e)`
sets it to `neg(e)` and flips `is_maximizing_`. Internally the objective is
*always a quantity to minimize*, so the soft constraint `obj <= bound` is
uniformly correct for both senses.

### Finalization

`close()` rebuilds the back-references, computes the topological order,
builds the `var_id -> constraint-index` adjacency (`build_var_constraints`, the
paper's `G_v`), performs an initial full evaluation, and sets the `closed_`
flag. After close the ordinary builders — variable and expression creation,
`add_constraint`, `minimize`/`maximize`, `add_var_sequence`,
`add_list_partition` — throw `std::logic_error` (#173): a node or row
appended then would sit outside the topological order, never evaluated.
`solve()` freezes the model it is handed (`Model::freeze()`: close if open,
append the objective row, build the cone cache), so a solved model refuses the
builders too. A closed model does not grow, apart from the objective soft
constraint `freeze()` appends (see below).

### Structure vs. state: `ModelStructure` and `freeze()` (#157)

A `Model` is two things with different lifetimes, and they are separate types:

- **`ModelStructure`** — the nodes, the flat edges, both CSR back-reference
  arrays, the topological order and its inverse, the constraint list, `G_v`, the
  lambda tables and the variable sequences. Nothing a search writes.
- **the rest of `Model`** — the variables, `node_values()`, the objective bound,
  the delta probe's scratch, plus the structural scalars (`objective_id_`,
  `is_maximizing_`, the three objective-row node ids, `closed_`), which stay in
  `Model` because twenty bytes per worker is not worth an indirection.

`Model` holds the structure through `shared_ptr<const ModelStructure>` for
reading and `shared_ptr<ModelStructure>` for writing. `freeze()` drops the
writable one, after which every structural method throws — variable and
expression creation, `add_constraint`, `minimize`/`maximize`,
`add_var_sequence`, `reserve`, `close` — and a write to the shared side no longer
compiles. **`add_objective_soft_constraint` is the one intentional exception**: on
a frozen model it returns rather than throwing, because `freeze()` has already
run it and a caller may still call it on a frozen replica. Its
idempotent early return therefore comes before the frozen check, and
`tests/test_model_share.cpp` pins that order.

**Copying a frozen model shares its structure**; copying an open one deep-copies
it. That is the whole mechanism behind the portfolio's memory
behaviour: see [Parallel Search](#parallel-search).

**The freeze point is after the objective row, not at `close()`**, since the
row rebuilds the back-references, the topological order, `topo_pos` and `G_v`,
which N workers must not redo on one shared DAG. `freeze()` folds both steps in,
then builds the cone cache (see **Evaluation Modes**). A model comes back from
`solve()` frozen, so it newly refuses `reserve` and `close`.

Deliberately **not** copy-on-write: a silent detach would put a worker back on
its own full copy with every test green and the memory saving gone.

`lambda_sum`/`pair_lambda_sum` callables are shared too, so several workers
invoke the same callable object at once; one carrying mutable state of its own
is a data race.

The guarantee is one-sided: a write to the shared side from **outside** `Model`
does not compile, because every public accessor is `s()`-backed and const.
Inside `Model`, the private `mut()` hands out a mutable reference and refuses
only at *runtime*.

### Objective as a Soft Constraint

When the model has an objective, `freeze()` -- which `solve()` calls -- calls
`add_objective_soft_constraint()` once (idempotent). This:

- creates a constant node `objective_bound_node_` (initially `+inf`),
- adds the constraint `obj - bound <= 0` (`objective_constraint_node_`), inert
  while the bound is `+inf`,
- records its index in `constraint_ids()` (`ModelStructure::constraint_ids`) as `objective_constraint_idx_`,
- rebuilds the topo order and `G_v` (a node/constraint was appended after
  `close()`).

`set_objective_bound(bound)` records the bound in the model and recomputes the
constraint residual in place. It does **not** write the constant node's
`const_value`: the bound is per-worker state (each worker tightens its own, and
in a portfolio also to the pool's best objective), so it lives beside the node
values and `evaluate()`'s `Const` arm reads it back for the one node
`objective_bound_node()` names (#157). That leaf-only branch is needed:
`full_evaluate` runs after every `restore_state`, and without it a restart
would reset the bound to `+inf`. The search tightens the bound to `obj - eps`
on each new best feasible solution, and releases it to `+inf` before `solve()`
returns so post-solve verifiers don't see it violated.

`has_objective_constraint()`, `objective_constraint_idx()` and
`objective_bound()` expose this state. "Real feasibility" everywhere means *all
constraints except* `objective_constraint_idx()`.

### Mid-solve structural writes (Python)

A bound `cbls.solve` (and `ParallelSearch.solve_master`) releases the GIL for
the whole search, and the first solve of a model freezes it under that release.
A `SolveCallback` or another Python thread can reach the model mid-search,
while the search reads the node array, the CSR indices and the topological
order without a lock. So the binding keeps a registry of the models a bound
solve is running on, and every structural write Python can reach raises
`RuntimeError` mid-solve: the `Model` builders, `close`, `freeze`, the `Expr`
operators and free functions, and a second solve of the same model.
`cbls.solve` registers the model **before** it releases the GIL, in the body
rather than through a `call_guard`, so another thread's builder call cannot
pass its check in the gap. The engine's own freeze is a C++ call and never
passes through the registry. `tests/python/test_solve_guards.py` pins these in
child interpreters. Value writes (`Variable.value`, `restore_state`) are not
structural and stay the documented data race.

A `ViolationManager` built before the objective row exists is one weight short
once `freeze()` or the first `solve()` appends it. Its weight-indexed reads
refuse such a manager rather than read past the end, and `LNS.destroy_repair` /
`destroy_repair_cycle` refuse it before the destroy moves the assignment; build
the manager after the model's last row.

### Python handles and model lifetime

Python holds nothing a builder can invalidate. `Model.var()`/`var_mut()` return
a `(model, id)` handle bound as `cbls.Variable`, which resolves through the model
on every attribute access and keeps the model alive; `node()` returns a copy of
the node, whose two exposed fields never change (a `reference_internal` into
`vars_` would dangle when a builder reallocates it). An `Expr` holds a raw
`Model*`, so every `Expr` the binding returns keeps its model's Python object
alive -- the model itself, not the operand, so a `s = s + x` loop does not chain
every intermediate to the next. `ViolationManager` keeps its model alive too. The cost: a `lambda_sum` / `pair_lambda_sum` callable that reaches an `Expr`
(or the model) makes a cycle the collector cannot see -- the model holds the
callable in a C++ `std::function` -- so it pins the model for the process
lifetime. Capture handles or plain data in such callables.

### State Save/Restore

```cpp
struct State {
    std::vector<double> values;
    std::vector<std::vector<int32_t>> elements;
};
```

`copy_state()` snapshots all variable values and elements;
`restore_state(state)` restores them. `solve()` snapshots the best feasible
state and restores it at the end. LNS snapshots state before destruction for
rollback.

---

## Violation & GLS Weights

**Files:** `include/cbls/violation.h`, `src/violation.cpp`

### ViolationManager

Tracks per-constraint violation and the GLS weight vector.

- `weights` (public `std::vector<double>`) — the per-constraint GLS weight `W`,
  initialized to `1.0`. This is the **only** feasibility-pressure mechanism;
  there is no global `lambda`.
- `constraint_violation(i)` = `max(0, constraint_node_value)` — non-negative
  violation for constraint `i`.
- `total_violation()` = `sum_c W[c] * max(0, constraint_value[c])` — the
  weighted total the GFJ engine minimizes. Cached and updated incrementally
  (full recompute every 1000 updates to bound floating-point drift; call
  `invalidate_cache()` after a weight change or `full_evaluate`).
- `weighted_violation_delta(var_id, j)` — the **no-commit counterfactual**
  `deltaG`: the change in `total_violation()` if `var_id` were set to `j`,
  without keeping the change. (Delegates to the allocation-free
  `Model::weighted_violation_delta(var_id, j, weights)`; transiently mutates and
  restores node state, so it is not reentrant on a shared Model — each search
  thread owns its own Model.) The GFJ jump *score* is `-deltaG` (positive =
  improving). Scalar variables only.
- `snapshot_violations(out)` / `weighted_delta_from(snapshot[, rows[, differing]])` — the structural
  counterpart of the above, for moves on List/Set variables that the scalar-only
  probe cannot score. Snapshot the accepted assignment's per-constraint
  violations once, apply a candidate move, then read the weighted change against
  the snapshot. Like `weighted_violation_delta` it accumulates **per
  constraint**, which is what keeps a row clamped to `kInfPenalty` from
  swallowing the O(1) real rows (`1e30 + 3 == 1e30`, so differencing two whole
  sums loses them; `1e30 - 1e30` cancels exactly). See #100 and #118. The
  `rows` form, ascending constraint indices, reads only those rows and is exact
  when they cover every row the move can touch; `differing`, when non-null,
  gets the rows of `rows` whose violation changed (#273).
- `augmented_objective()` = `obj + total_violation()` — the penalty-method
  metric. **Used only as the inner solver's local descent objective**, not as
  any search acceptance rule. (When the objective is folded in as a soft
  constraint, the objective term is technically double-counted here; that is
  acceptable for the hook's local polish.)
- `is_feasible(tol=kDefaultFeasibilityTolerance)` / `violated_constraints(tol)` —
  convenience predicates over *all* constraints (including the objective soft
  constraint). The default is `1e-6` (matching SCIP's `numerics/feastol` and
  `verify_model`); it is an *absolute* residual, so a much tighter value is not
  achievable on models whose constraint bodies are large in magnitude.
  `SearchConfig::feasibility_tolerance` shares the same default.
- `bump_weights(factor=1.0)` — increments the weight of each currently-violated
  constraint (a simple additive scheme; the GFJ engine uses the
  decay-then-bump GLS update below instead).

### GLS Weight Dynamics

The Guided Local Search update fired on GFJ stagnation (paper Algorithm 3) is:

1. **Decay**: multiply every weight by `rho`.
2. **Bump**: add `1.0` to every currently-violated, *active* (weight > 0)
   constraint.

`rho` is sampled per batch from `{0.95, 1.0}` (decay or pure additive). Weights
masked to `0` (e.g. nonlinear constraints during the two-phase linear-first
pass) stay `0` under decay and are never bumped, so they remain inactive.

**The decay is lazy (#175).** `gls_update_weights(vm, rho)` is the eager form,
O(#constraints) per bump, and stays as the public reference. `FeasibilityJump`
applies the same update through `LazyWeightDecay`: it stores `w' = w / s`, a
decay is `s *= rho` (O(1)), and a bump adds `1/s` to the rows in V only, so a
bump costs O(|V|) plus the nonzeros of V's counted rows it requeues. `s` is
folded into the stored weights and cached jump scores when it would leave
`[1e-30, 1e30]` and on every exit from the GLS loop, so
`ViolationManager::weights` holds effective weights whenever FJ is not running.
The weights agree with the eager form to rounding (to the bit at `rho = 1`); a
positive weight is floored at the smallest subnormal, so only `rho = 0` zeroes
one. Unlike the eager form, a decay leaves every cached score exact, which
changes trajectories where a cached entry's rows were violated by at most the
tolerance. On cbs-cta, where the eager decay was the bottleneck, the lazy form
runs 31% more iterations; elsewhere it moves trajectories within seed noise
(the A/B tables are in this file's git history).

After a new best feasible solution, the search resets all weights to `1.0`
(`FeasibilityJump::reset_weights`) — a fresh penalty landscape per the paper.

---

## Generalised Feasibility Jump

**Files:** `include/cbls/feasibility_jump.h`, `src/feasibility_jump.cpp`

Generalised Feasibility Jump (GFJ; paper Algorithms 1–3) is *both* the
construction heuristic and the main search engine. It drives the model's current
assignment `X` toward feasibility by repeatedly applying the best of a sampled
set of improving single-variable jumps, with GLS weight bumping on stagnation.

The engine's state maps to the paper's `S = <G, X, W, V, Q, J>`:

| Symbol | Meaning | Implementation |
|--------|---------|----------------|
| `G` | constraint graph | `Model` |
| `X` | variable values | `Model` variable values |
| `W` | constraint weights | `ViolationManager::weights` |
| `V` | violated constraints | `violated_` flags, plus the dense list `violated_rows_` and per-variable active-violated counts, kept incrementally (#174) |
| `Q` | scan set of candidate vars | `queue_` / `in_queue_` |
| `J` | cached per-var best jump | `JumpTable` |

**V is kept incrementally (#174).** A dense list of violated rows and a
per-variable count of active violated rows are maintained wherever a row flips,
so the participation test is O(1), and the bump's requeue, the Novelty seed and
the feasibility test are O(|V|) plus the variable lists of the rows they queue
from, instead of a sweep of every row. A Novelty batch as a whole stays
O(#rows), through `init_novelty_weights` and the rebuild it ends with, once per
batch (see [Novelty Jump](#novelty-jump)). V is visited in list order (appends
plus swap-removes): a deterministic function of the flip history, so a seeded
run reproduces, but not ascending; `tests/test_fj_trajectory_fence.cpp` pins the
resulting hashes. The header comment on `violated_` states where it wins and
loses and what it costs in memory.

### JumpTable

A per-variable cache of the best jump found for that variable: the
`jump_value` to move to and the `score = -W.deltaG` (positive = improving). An
entry is lazily invalidated when a neighbouring variable changes (paper
Algorithm 1), so most iterations reuse cached scores.

### `compute_var_jump`

Computes the best jump for one scalar variable under a given weight vector — the
value minimising `weighted_violation_delta` over a small, type-specific
candidate set, plus its score:

| Type  | Candidates |
|-------|-----------|
| Bool  | the flip `1 - x` |
| Int (window width <= 256) | every integer in the sampling window — `domain_window(var)`, which is `[lb, ub]` verbatim whenever both bounds are finite and the width does not overflow (a domain as wide as `[-DBL_MAX, DBL_MAX]` is narrowed to the clamp). Taken only when both endpoints are within ±2^53; past that `v += 1.0` does not advance and the enumeration would not terminate |
| Int (otherwise) | window endpoints, neighbours `x±1` (clamped to the *declared* bounds, so a value that has drifted outside the window keeps a local move), and a 32-point rounded grid across the window |
| Float | Newton step toward the root of each violated constraint containing `v` (`x - residual/grad`, gradient from a forward up-cone pass, reverse-mode AD past its budget (#249); up to 4), then midpoint and endpoints. Once the search has stagnated, a Float at a *stationary* point of every violated constraint containing it additionally gets a two-sided local probe at `x ± {1e-6, 1e-2}·(|x|+1)` — see below |
| Int (otherwise) and Float, on a model with an `Element`/`Ceil`/`Floor`/`Round` node, for a variable `close()` marked as reaching one | additionally, the **breakpoints** (#186) of every such node — where `Ceil`/`Floor` change (the integers), where `Round` changes (the half-integers), and an `Element`'s index values — mapped back to `v` through the argument's slope at the current point (carried forward over the cone part `close()` marked, `breakpoint_flags`, through every op that `carries_slope`), or the exact reciprocal when the argument is `N / D` with `v` in `D` only. **Limitation:** any other nonlinearity is linearised, so a candidate can miss its edge, and a path through a stopping op (structural, user code, a breakpoint op) contributes nothing; candidates are still scored exactly. Each edge yields the values of `v` just either side of it: for an Int rounded outward, plus the edge itself when it maps to an integer; for a Float `nextafter`-stepped, plus the edge when it lies in its own plateau. An `Element` index gets `k` itself (Int) or the middle of index `k`'s plateau intersected with the range `u` covers (Float). A `Ceil`/`Floor`/`Round` argument offers every breakpoint while there are at most 32, an `Element` every index while there are no more than its capped path would offer; above that, the ones either side of the current argument, for an `Element` up to 32 representatives of the currently selected line, and a 31-point grid shared by every capped argument — **a deviation** from #186's "all index values", for cost (`kExhaustiveBreakpoints`). Candidates are deduplicated, and come after the Float's Newton candidates, so those still win a tie. Every other variable gets exactly the candidates above |

An exact Int jump over all-linear columns (#208, #216: `int_argmin`, a port of
OR-Tools' `ComputeLinearJump`, scoring `x ± 1` and a `RangeConvexMinimum` over
the side's kinks) replaced these candidates when `SearchConfig::fj_exact_int_jump`
was on. It was opt-in and off by default, and off was the candidate path bit for
bit, because it regressed #208's own repro (`x + y == 1e7 − 1`, `|x − y| ≤ 1` as
two rows, over `[0, 1e7]`): exact jumps land on the kinks and the pair
oscillates about the midpoint until the GLS weights break it, 261k-836k iterations to
feasible against 19-72 on the grid. #208 was closed as not planned and #381
removed the arm; the code, its tests and the `--exact-int-jump` runner flag are
in git history at `1330cd2c`.

Each candidate is scored with one `weighted_violation_delta` probe — or, when
every weighted row of `G_v` is a comparison (`Leq`/`Geq`/`Lt`/`Gt`/`Eq`) whose
two children are affine, or a bare affine body (`add_constraint(expr)`, read as
`expr <= 0`, scored as the residual `expr − 0` with a literal 0, which equals the
engine's `clamped(expr)` for every value, non-finite included — #190; a
non-constant comparison nested anywhere below the row disqualifies it, since its sentinel
residual is not affine), in closed form by `LinearJumpScorer`
(`include/cbls/linear_jump.h`): each such row has a constant slope
`r = ∂(p − q)/∂v`, cached lazily per row from `compute_partials_sparse` into a
table laid out parallel to `G_v` (a row's build writes each of its variables'
slots, so `prepare` reads the k-th row's slope at a fixed position, #176), and the candidate costs `Σ_c w_c·(clamped(cmp(p + rΔ, q)) − clamped(old))` — O(|G_v|)
instead of two `delta_evaluate`s over every row in the column. Same candidates,
same first-seen rule, same per-row differencing (#100); the scores agree with
the probe to rounding, not to the bit -- so the chosen jump is guaranteed the
probe's only on integral data, and an ulp can flip a near-tie on fractional
data -- and a committed jump still goes
through the DAG walk (`commit_scalar_move`). Any other weighted row, or a non-finite computed side,
takes the probe. The scorer's per-worker memory is recorded under the
portfolio section below. The Float Newton step reads the
same cache for a row's partial where it is bit-identical to `compute_partial`
(Leq/Geq/Lt/Gt always; Eq only against a literal side). Newton
candidates are considered first so that, on a tie in violation delta (a feasible
plateau), the gradient-informed point wins. Because the objective is a
constraint `obj <= bound`, when that constraint is violated its Newton candidate
pulls the objective *down* — this is how a hook-less continuous model still
descends the objective.

The probe is exactly antisymmetric and `LinearJumpScorer` is not: on a plateau
balanced over fractional coefficients a move and its reverse can both score
+1 ulp, and FJ can bounce between them (#178 found it on MINLPLib `alkylation`
with no feasibility or objective loss). The batch API bounds such a cycle, but
`gls()`/`run()` has no progress exit: the LNS repair and Python's
`fj_nl_initialize` end one only at their iteration or time caps, and a default
`FeasibilityJump::run()` has neither. Measurements:
`docs/benchmarks/linear-jump-pingpong.md`.

**The stagnation-gated escape probe (#107).** Every candidate above is either
a Newton step, which vanishes with the gradient, or a constant derived from the
box, so a Float at a point where every violated constraint containing it is
stationary has no candidate that moves it. (An Int always has `x ± 1`; its
bounds are read as doubles through `domain_window`, so infinite bounds still
give candidates, #114.) The probe supplies a two-sided local move, since at a
saddle a zero gradient cannot tell the descent direction. It is **off by
default and armed only once the search is stuck — after `perturbation_period`
batches without improvement, or, when the caller set a wall clock, after
`kEscapeArmFraction` (a quarter) of the budget with no new best (#117),
whichever comes first** — and disarmed on the next new best. The gate is
load-bearing: "stationary and nothing improving" is the steady state of local
search, and an always-on probe with no disarm was ~9x worse on `shiporig`,
because its drip of tiny improvements kept stagnation, and so diversification,
from ever registering.

The wall-clock route exists because `perturbation_period` counts batches, and a
batch can take seconds on an expensive model, so the count may never reach the
threshold. It reads the clock only when the caller set a wall-clock budget, so
a run bounded by `max_iterations` alone stays bit-reproducible. It arms without
a diversification kick, at most `1/kEscapeArmFraction` times per run, since each
improvement disarms it; diversification itself stays on the batch counter.
**Caveat:** with a wall clock, trajectories are budget-dependent (a 10s run arms
at 2.5s, a 600s run at 150s), and a run still hunting its first feasible point
has never improved, so the probe stays armed through the whole pre-feasible
phase, where it makes batches more expensive.

A single call is *not* a converged 1-D minimiser; the GLS loop iterates these
cheap jumps. The continuous heavy lifting is left to the
[inner solver](#inner-solver-continuous-intensification).

### The GLS Loop

`gls_loop(sample_size, batch_iter_limit)`:

```
loop:
    if apply_jump(sample_size) fails (no sampled var improves):
        if no active constraint is violated: return Feasible
        decay by rho, bump active violated   # lazily: s *= rho, w' += 1/s on V (#175)
        re-enqueue vars of violated active constraints, invalidate their jumps
    ++iterations
    stop if batch / global iteration budget reached
    every `stride` iterations: stop if the deadline has passed, else retune `stride`
```

The deadline is checked on a stride because reading the clock is not free; the
stride is sized in *time*, so it costs at most 1/64 of the budget (or one
iteration, whichever is larger) — see
[bound 1 of the wall-clock budget](#wall-clock-budget).

`apply_jump` samples up to `sample_size` **distinct** variables from the scan
set `Q` (best-of-N: `sample_size_general = 5` general,
`sample_size_linear = 5` linear phase), refreshes any stale `JumpTable` entries via `compute_var_jump`,
removes non-improving vars from `Q` permanently, and commits the best improving
jump via `update_var`. A removed var does not count toward `sample_size`, and
there is no draw cap: `apply_jump` returns false — the local minimum that
triggers a GLS bump — only once `Q` holds no positive-score var (ViolationLS
Algorithm 2; OR-Tools' `ScanRelevantVariables`). Distinctness comes from
drawing over `Q`'s unsampled suffix, so no redraw is ever needed. A cap that
counted the removals declared false minima on long rows of non-improving vars
and over-bumped their weights (#206). Novelty Jump's `select_novelty_var`
follows the same rule: only vars passing the filter F count toward its 3, and,
as in OR-Tools, a var that fails F or is on the move stack leaves `Q'` until a
move or undo on one of its rows re-queues it.

`update_var` writes `X[v]`, delta-evaluates, refreshes the violated set for
`v`'s constraints, invalidates neighbour jumps, and replenishes `Q` with vars now participating in active violated constraints.

**The objective row is skipped while inert (#210).** The folded `obj <= bound`
row spans every objective column, so walking it on each move of an objective
variable cost O(#columns) on a MIP. `update_var`, Novelty's
`nj_requeue_neighbours` and (since #209) its W' reset re-queue
`reset_changed_novelty_weights` skip it when it cannot change any neighbour's
score: the bound is `+inf` and the objective value stays finite (at `obj = -inf` a
candidate at `+inf` gives NaN, so the residual's `-inf` alone is not enough), or
the row is slack by more than `M = max_v |a_v| (ub_v - lb_v)` both before and
after the move (OR-Tools' `row_max_variations`;
`LinearJumpScorer::row_max_variation`). The finite-bound form also needs every
objective column inside its box, which the API does not guarantee, so each
rebuild of `V`/`Q` checks it and switches that form off while one is outside.
Only for an affine objective: a nonlinear one keeps the walk even at `+inf`,
since a jump can turn it NaN. The skip is exact for the jump table up to the
rounding the closed-form scorer already accepts; what changes is `Q`,
which no longer re-admits unchanged objective columns after every move, so
trajectories on models with an objective move. The bump's re-queue needs no
skip: it visits only violated rows, and an inert row never is. It pays most
where no feasible point is found and the bound stays `+inf`; the counts and
regimes are in the comment above `objective_row_inert` and this file's git history.

### Two-Phase Linear-First (construction only)

`FeasibilityJump::run()` (standalone construction) optionally runs GLS on the
*linear submodel* first: nonlinear constraint weights are masked to `0`, GLS
satisfies the affine constraints, then the general phase continues from the
weights the linear phase learned, with the masked rows unmasked at `1`
(ViolationLS Algorithm 6, lines 18-19, passes one W to both GLS calls). `compute_linear_constraints()` marks each constraint affine by a
single topo pass (Const/Neg/Sum affine in affine children; Prod/Div affine with
a constant factor/divisor; comparisons affine when both sides are). This is the
default for GFJ-as-solver. As a warm-start (LNS repair, SA-style seeding) the
two-phase pass over-commits the linear submodel to cost-pessimal boundary
values, so `fj_nl_initialize` and the outer loop use **single-phase**.

### Batch API (drives the outer loop)

The ViolationLS outer loop owns the iteration clock and calls:

- `begin(set_initial_x)` — optionally set each scalar var to the domain value
  closest to 0, full-evaluate, reset weights to 1, rebuild `V`/`Q`. This is the
  only thing that initialises scalars under `solve()` (#108); `set_initial_x =
  false` means "refine the assignment I already have" (LNS repair, warm starts,
  `skip_init`).
- `batch(batch_iterations)` — run `gls_loop` for at most `batch_iterations`
  GLS iterations; returns whether all active constraints are satisfied.
- `reset_weights()` — `W <- 1`, rebuild `V`/`Q` (called on a new best).
- `resync()` — rebuild `V`/`Q` from current state, keep weights (called after a
  hook/structural mutation outside the engine's bookkeeping).
- `perturb(probability)` — randomise each scalar var w.p. `probability` and
  apply `max(1, round(probability * |elements|))` random structural moves to
  each List/Set var, full-evaluate, reset weights. OR-Tools' no-restart kick
  (`PerturbateCurrentSolution`: each scalar w.p. 0.05 to a value near its
  current one, `RandomValueNearValue`) was an opt-in arm, off by default and
  kept off by #219 / #226's smoke A/B (#217); #382 removed it. The code
  (`random_near_value`), its tests and the `--perturb-near-value` runner flag
  are in git history at `8803b404`.

**The objective before the first feasible point (#218), on by default.**
Without help FJ ignores the objective until the first feasible point: the
folded row's bound is +inf, so the row is inert, and the start is the
closest-to-zero value. `GFJConfig::objective_tie_break` (`SearchConfig::fj_objective_tie_break`,
default **on**, #226; `GFJConfig`'s own default stays off) carries over
what OR-Tools does instead. It needs an affine
objective, whose coefficients FJ reads once at construction by one reverse
sweep; on any other objective it does nothing.

- `objective_tie_break`: in `compute_var_jump` an exact tie in the weighted
  violation delta goes to the candidate lowering the objective most, and a jump
  that wins on the objective alone scores `kObjectiveTieScore` (the smallest
  positive double); Q also keeps every variable not at its objective-better
  bound (`objective_scan`, OR-Tools' `HasBetterObjectiveValue`). OR-Tools adds
  `delta * coefficient / INT64_MAX` to the score instead, which in doubles over
  fractional weights can outweigh a small violation decrease; here the objective
  only breaks a tie, within a variable's candidates and between variables in
  `apply_jump`'s best-of-N (larger objective gain wins). A weight rescale leaves
  `kObjectiveTieScore` unscaled, since scaling the smallest double would round
  it to 0 and drop the column from Q. FJ batches
  only, throughout the run; Novelty batches and the LNS repair keep their own
  rules.
- `set_rho(rho)` — re-randomise the GLS decay between batches.

### Propagation inside compound moves (removed, #380)

Implied domains depend on bounds, not on current values, so a plain FJ jump
gains nothing from propagation beyond the load-time implied domains
([Implied variable bounds](#implied-variable-bounds)). The one place where
variables are held fixed for a while is the Novelty compound-move stack, and
#329/#339/#340 propagated each leg there over the linear rows (a
`DomainPropagator` with a trail; later legs drew their candidates inside the
narrowed box; a leg in conflict was not extended). It was opt-in, off by
default, and off was the Novelty path bit for bit. #331 measured it at
`ef1bbad` and kept it off: the MIPfeas dev roster's pooled PI ratio was 1.142,
and the MINLPLib dev sample's GM(gap+1) ratio 1.052, both worse and both
inside the control's noise floor. #380 removed it; the code, its counters
(`novelty_propagation_tightenings`, `novelty_propagation_conflicts`) and the
`--compound-propagation` runner flag are in git history at `57cdbfba`.

---

## Novelty Jump

**Files:** `include/cbls/feasibility_jump.h`, `src/feasibility_jump.cpp`
(`novelty_batch`, `apply_novelty_jump`)

Novelty Jump (paper Algorithms 4–5) is a bounded-backtracking **compound-move**
search that escapes local optima single-variable FJ cannot — chained-invariant
fixes where no single jump improves weighted violation but a short *sequence*
does.

### Novelty Weights

At the start of a Novelty batch it builds `W' = ` novelty weights from the GLS
weights `W`:

- `W'[c] = W[c]` for constraints **violated at entry**,
- `W'[c] = kCompoundDiscount * W[c]` (`kCompoundDiscount = 1/1024`, OR-Tools'
  epsilon) for constraints satisfied at entry.

Breaking a currently-satisfied constraint is therefore *cheap*, which lets the
search build chains that target the initially-broken constraints. When a move
breaks a satisfied constraint, that constraint is promoted to full weight and
its vars are enqueued. Algorithm 4 re-initialises `W'` this way at every
discrepancy level; only the rows noted since the last reset are redone
(`reset_changed_novelty_weights`, OR-Tools' `ResetChangedCompoundWeights`),
which reproduces the whole-row re-init exactly because every other row still
holds its re-init value — the comment above that function gives the argument,
and `novelty_weights_consistent()` checks it.

### Bounded-Backtracking Search

`novelty_jump_search(s_m, budget)` is a recursive DFS over compound moves with an
explicit move stack (the paper's set `T`). `s_m` is the cumulative
*original-weight* score of moves on the stack; `s_c` tracks the best child score
explored at the current level. A candidate var is selected (`select_novelty_var`)
as the best — by **novelty score**, as OR-Tools ranks its sample — of up
to 3 sampled vars in `Q\T` passing the filter
`(s_m + W'-score > 0) OR (W-score > s_c)`, scored by `compute_var_jump` under
`W'`; its W-score comes from the same pass (a dual-weight
`LinearJumpScorer::prepare` probe, #272), bit-identical to scoring it afresh
under `W`. The move is applied; if `s_m + W-score > 0` the compound move is
committed (left applied) and returns true; otherwise it recurses, and on failure
backtracks (reverting the move and consuming a `budget` discrepancy — at the
root too, as in the paper; OR-Tools does not count root backtracks, see the
comment above `novelty_jump_search`).

One ApplyNoveltyJump (`novelty_descent`, Algorithm 4) iterates `budget = 0, 1,
2`, committing improving compound moves and restarting at `budget = 0` from the
new state. A commit keeps `Q'` and `W'` as they are: the moves that made it up
already re-queued every var whose verdict they could have changed.

### The Novelty batch

`novelty_batch(batch_iterations)` is Algorithm 6 line 22, GLS with
ApplyNoveltyJump as its move: when a whole ApplyNoveltyJump fails at
`budget = 2` — a Novelty local minimum — the GLS weights are decayed and bumped
exactly as an FJ batch's are (`W'` follows on the bumped rows), and the search
goes on. `W'` is initialised and `Q'` seeded once per batch. Its work is
charged to the GLS iteration count, 3 per applied move and 1 per bump, and
bounded by `batch_iterations` of it — a Novelty move scores its whole sample of
3 afresh under `W'`, where an FJ iteration reads cached jumps, so this is the
charge closest to the equal per-batch effort Algorithm 6 assumes, and it keeps
`max_iterations` meaning comparable work with compound moves on or off. Its
wall clock is bounded by the GLS loop's deadline stride, polled per move and per
bump (a search that applies no move still bumps), and it ends by rebuilding
FJ's `V`, `Q` and jump table, so the caller needs no resync.

Like FJ's `update_var`, a move or undo re-queues into `Q'` only the neighbours
that sit in an active violated row (Algorithm 1 lines 16-17, OR-Tools'
`ShouldScan`), and like it ignores a row whose residual is in `(0, tol]`, which
counts as satisfied but still scores. A var whose rows all hold fails both arms
of the filter, so admitting it only costs scoring time.
`SearchCounters::novelty_moves`, `novelty_commits` and `novelty_weight_bumps`
record its work apart from FJ's. The MIPfeas measurements (cost per move,
profile, smoke roster) and the pre-registered A/B that settled when Algorithm
6's `A` is redrawn — every batch, a deviation from §5 that `pick_batch_kind`
documents — are in #209.

`apply_novelty_jump()` is one stand-alone ApplyNoveltyJump from a fresh `W'` and
`Q'`, capped at `kNoveltyWorkBudget = 256` applied moves; the caller must
`resync()` afterward. Tests and the trajectory fence's history use it; the
search runs `novelty_batch`.

> **Status:** Novelty Jump is **on by default**
> (`SearchConfig::use_compound_moves = true`), on a single-seed smoke that
> favoured it on MINLPLib and not on UC-CHPED, whose runner therefore turns it
> off; the smoke's numbers are in #209. When enabled,
> `novelty_jump_probability` (default 0.5, matching the paper) is the
> probability that a scalar batch is Novelty, drawn afresh every batch (see
> [Main Loop](#main-loop)).

---

## Structural Batch

**Files:** `src/structural_batch.cpp` (`StructuralBatch`),
`include/cbls/move_generator.h` (`MoveGenerator`, `NeighbourList`,
`StructuralSelection`), `src/moves.cpp` (the built-in typed moves)

FJ jumps only scalar variables, so List/Set-structured models cannot improve
their structural assignment through FJ alone. The structural batch is the
List/Set peer of an FJ/NJ batch: it sweeps the registered **move generators**,
has each propose candidate moves for the variables in its scope, and greedily
keeps any move that reduces total weighted violation (negative weighted `deltaG`
under the current GLS weights `W`).

By default the generators are the built-in ones — exactly one per List/Set
variable, in variable-id order. `tests/test_structural_equivalence.cpp` pins
their trajectory bit for bit at one thread on a given seed, against digests
recorded at `a805cb6`.

That `deltaG` is accumulated per constraint, via
`ViolationManager::snapshot_violations` / `weighted_delta_from`, and not by
subtracting two `total_violation()` readings, which fails twice over:

- **Clamped rows (#118).** Once any row is clamped to `kInfPenalty` (the
  objective row is, while the #116 sentinel bound is installed), both readings
  round to the same double and every move scores exactly 0.
- **Cache drift.** `total_violation()` is an incrementally maintained
  accumulator whose periodic from-scratch recompute bounds its rounding error
  without removing it, so two readings can differ in the last ulp for a move
  that changed nothing; a `- 1e-12` guard cannot filter that (`x - 1e-12 == x`
  for every double `x > 16384`, and GLS weights push setcover's total far
  above that). The old test committed such phantom zero-delta moves on
  setcover with no row clamped (measured in #118); `tests/test_nonfinite_guard.cpp`
  pins the property on a two-row fixture.

The delta is **restricted to the rows the move can have changed**, and is the
same `double` a full rescan gives, not merely the same answer: a row whose
violation equals the snapshot's is skipped by both, and ascending order keeps
the surviving terms in one summation sequence. The batch keeps `differing_`,
every row whose violation differs from the snapshot (empty at each snapshot),
and after each walk re-reads that list plus the walked variables' G_v
(`Model::constraints_of_var`), which re-derives it exactly — a row in neither
was equal before the walk and the walk did not touch it (#273). A partition
generator over every route of a CVRP is then scored over the one or two routes
a move touches. The walk's own rows would not be enough: under
`FirstImprovingSample` a kept candidate's rows differ from the re-taken
snapshot once the next candidate restores the sample base, and the candidate
after that does not walk them. Scoring reads no scope; the scope drives only
the skips (an empty one is still a contract breach, asserted in Debug).

Moves come from `generate_standard_moves` (`src/moves.cpp`):

| Type  | Moves |
|-------|-------|
| List  | `list_swap`, `list_2opt`, `list_relocate`, `list_or_opt_2`, `list_or_opt_3`, `list_insert`, `list_exchange`, `list_remove` |
| Set   | `set_add`, `set_remove`, `set_swap` |

`list_insert` and `list_remove` change the length and `list_exchange` changes the
membership at a fixed length, so all three are inert on a permutation (whose
length is pinned at its universe) and on a member of a partition, whose
membership is shared with its sibling lists. `list_exchange` is what makes
`min_len == max_len < universe` — a fixed-count selection model — searchable at
all: without it the membership drawn at initialisation would be the membership
for the whole run. The partition's own
moves — `partition_relocate`, `partition_swap`, `partition_2opt_star`, and under
`Cover::AtMostOnce` `partition_insert` / `partition_remove` — come from
`generate_partition_moves` and are registered by `default_move_generators` as one
generator per partition. Each of them preserves the cover by construction, which
is why the cover needs no constraint row; `generate_partition_moves` appends **at
most one candidate per call**, because several candidates from one call are all
built against the same assignment and `FirstImprovingSample` may commit more than
one of them.

A structured `Move::Change` carries **positional `ElementEdit`s** rather than the
whole element vector it wants (#164) — a swap of two positions, a reversal, a
`std::rotate` for a relocated segment, an insert, an erase, an assignment — or a
`replacement` vector for what positions cannot express. Positions are relative,
so the structural batch keeps the sample's starting assignment and puts back,
before each candidate, exactly what the previous one disturbed: committing
candidate k and then candidate k+1 leaves k's change gone. That is one
allocation-free `assign` per variable the previous candidate changed (one or
two for every built-in), where whole vectors cost at least two heap allocations and
three O(n) copies per candidate.

The scalar move generators `flip`, `int_dec`/`int_inc`/`int_rand` and
`float_perturb` also live here and are used by LNS randomization paths. There
is **no** sequence-aware block on/off generator; sequence structure is used by
LNS, whose destroy step can remove a whole registered sequence at once — see
the LNS section.

A batch is structural with probability `structural_batch_probability`: `< 0`
auto-selects `0.33` when the batch would build at least one move generator --
a registered `move_generators` entry, or a List/Set variable with
`default_structural_generators` on -- and `0.0` otherwise, so a model
whose batch would propose nothing never spends an iteration on one. When the
batch would build a generator but FJ has nothing to jump -- no Bool/Int/Float
variable with two values in its domain that some row reads, as on a List-only
CVRP model or a mixed one whose every scalar is fixed or unread -- the automatic
setting is `1.0` (#201): there an FJ batch would do no work, report itself stuck
and take the unproductive kick, which re-randomises every List. An explicit
probability is honoured, as the control arm, but on such a model its empty
FJ/Novelty batches neither kick nor count toward `perturbation_period`. A model
FJ has work in keeps its mix and RNG draws exactly -- which is not the same as
being safe: the folded objective counts as a row, so a List model whose only
movable scalar is read by the objective alone keeps the 0.33 mix, FJ settles
that scalar at once, and the empty-batch kicks return. Set the probability to
`1.0` explicitly there. After a structural batch commits anything, the engine
`resync()`s its scan set.

### Registering a move generator

`SearchConfig::move_generators` takes `shared_ptr<const MoveGenerator>`s. A
generator names the variables it moves (`scope()`), appends candidates
(`generate`), is told when one of its moves was committed (`on_commit`) and
hands out a per-worker copy (`clone`). The built-ins are peers registered the
same way, and `default_move_generators(model, neighbours)` returns them so a
caller assembling its own set can start from them. `default_structural_generators
= false` drops them.

Three contracts the batch relies on, all stated on the class:

- **`generate` must return in bounded time.** The deadline is checked *between*
  generators, never inside one (#105), so an unbounded `generate` is an
  unbounded overrun of `solve()`'s budget. One generator per variable keeps
  that check between variables.
- **`clone()` must return an independent object.** Every portfolio worker builds
  its own `StructuralBatch` and therefore its own clones, so a cache or a cursor
  a generator holds is per-worker state. A clone that shared it would be a data
  race across worker threads (#157); `tests/test_parallel.cpp` pins that the
  registered instance is never the one the search runs.
- **`scope()` must name every variable the moves can change.** It is what
  `ViolationGuided`'s and `GranularBest`'s skips are computed from, so a move
  touching a variable outside the scope can be skipped when it would have
  helped. Scoring does not read it: the batch scores the rows each move's own
  variables reach. Debug and sanitizer builds assert the contract on every
  move, so a move from an empty scope aborts there.

Not routed through the generator set: the diversification kick's structural half
(`FeasibilityJump::perturb_structural`). It draws from the same RNG as the rest
of the search, so changing what it draws shifts every later draw on every model
with a structured variable — and it wants an arbitrary legal move rather than a
good one, which is the opposite of what a cost-aware generator gives it. A List
in a `ListPartition` also offers that partition's inter-list moves, anchored on
the variable being kicked, which is what lets a kick move an all-empty
`AtMostOnce` partition at all.

### Selection policies and neighbour lists

`SearchConfig::structural_selection` chooses what the batch does with a
generator's candidates:

| Policy | Rule |
|---|---|
| `Auto` (**default**, #203) | `GranularBest` when `structural_neighbours` is set, `FirstImprovingSample` otherwise. Resolved once, when the batch is built (`resolve_structural_selection`) |
| `FirstImprovingSample` | one `generate` call, each candidate applied in turn and kept if it strictly improves |
| `BestOfSample` | call the generator until it has offered at least `structural_sample_size` candidates (one call always happens, and a call that appends several may overshoot), score them all, commit the best improving one |
| `ViolationGuided` | `BestOfSample`, restricted to generators whose scope can still change a **violated** row |
| `GranularBest` (#203) | `ViolationGuided`, except that each `ListPartition` offers the **granular complete neighbourhood** of one random element per `generate` call (Toth & Vigo 2003): every relocate (after/before), swap and adjacency-making 2-opt / 2-opt\* pairing it with each of its k nearest neighbours, a move into each empty member list (so an emptied list can be reopened), plus insert/remove under `AtMostOnce`. A call yielding fewer than `structural_sample_size` is topped up with a further element's neighbourhood. Partition members' own intra-list generators propose nothing, since same-list pairs cover them. Needs `structural_neighbours`; without one it is `ViolationGuided` |

`ViolationGuided`'s restriction is **exact rather than heuristic**: if no row in
the scope's G_v is violated at the accepted assignment, every one of those rows
contributes 0 to the weighted violation, so a move over that scope can only
leave the total alone or raise it. Skipping cannot skip an improving move.

It is, however, guidance at the **variable** level, not the element level:
the DAG tracks incidence per variable, and a `Set` read through a single
`Lambda` node puts every element of the universe in every row that node feeds.
Element-level guidance needs either a generator-supplied element scorer or the
custom-invariant work of #166; what is available today is the variable-level
choice plus whatever granularity a neighbour list supplies.

`NeighbourList` is that granularity — Toth & Vigo's granular neighbourhood
(INFORMS J. Computing 15(4), 2003; used throughout modern routing heuristics
such as Vidal's HGS-CVRP). For each element of a universe it names the `k`
others a move involving it may reach, nearest first under a cost the model
author supplies. Handed to the built-ins through
`SearchConfig::structural_neighbours`, it makes a List move's second position
the position of one of `elements[i]`'s neighbours, and a Set add/swap bring in
the nearest neighbour of a currently-selected element that is not already in.
Both fall back to the uniform draw when the list offers nothing usable, so a
partial list restricts where the search *looks* and never what it can *reach*.
It is validated once at construction and indexed unchecked thereafter; `of()` is
nonetheless total on an out-of-range element, which is what keeps the type safe
to expose to Python (#156).

`structural_sample_size` defaults to 8. That is a **neutral placeholder, not a
measured choice**, and no published result is derived from it. It feeds the
opt-in `BestOfSample` and `ViolationGuided`, and `GranularBest` — the `Auto`
default when a neighbour list is set — as the floor its one-element
neighbourhoods are topped up to. Sweeping it means driving
`cbls_setcover --structural-sample-size` directly.

### Structure-only models

In its **default configuration** the structural batch is a
**first-improvement hill climber over a 3-5 move random sample**, not a guided
search. Per pass and per variable, `set_moves` proposes exactly one random add,
one random remove and one random swap; nothing chooses *which* element on
violation grounds, the way FJ's jump table and best-of-N scan-set sampling
choose a scalar's value. On a mixed model GFJ hides that; on a model whose only
variables are structured it is the whole search, because everything else is
inert:

| Mechanism | On a structure-only model |
|---|---|
| FJ batch | not scheduled at the automatic mix: with no movable scalar every batch is structural |
| Novelty Jump | not scheduled at the automatic mix either — compound moves are chains of scalar jumps, nothing to chain |
| `perturb` kick | reaches them (each List/Set variable gets its own pass of `clamp(round(p*|elements|), 1, |elements|)` random structural moves), but the moves are the same unguided ones — see [Diversification](#diversification) |
| LNS | on an unpartitioned List/Set, destroys the variable wholesale (a List keeps its elements and is reshuffled), then repairs with an FJ that has nothing to jump. A `ListPartition` is ruined and recreated instead (see [LNS](#destroy-phase)): strings of elements leave a few lists and go back in one at a time at their best position, so the membership across lists moves |

The measured consequence is in `benchmarks/instances/setcover/README.md`: on
OR-Library set covering, one `Set` variable costs several times the optimum on
the weighted instances where one Bool per column is within a few percent, and
the opt-in guided policies have not established a separation from the default.
A 3-row, 4-column fixture in `tests/test_setcover.cpp` reproduces the stall
exactly: the Set encoding stops one move away from the optimum because every
single add/remove/swap from its incumbent is worse.

### Deadline bound

The sweep is bounded by the same wall-clock deadline as the rest of the loop,
checked **between generators** (`StructuralBatch::run`) — never mid-generator,
so a generator's drawn candidates are always evaluated whole (the reference move
set is never truncated for speed) and the overrun is capped at one generator's
work. The check is unconditional per generator, not strided.

The bound is needed because the sweep's cost grows with the model:
`O(#generators x #candidates x (delta_evaluate + |R_c| + |differing_|))`, where
a candidate is scored over the rows its walk reached (`R_c`, from each walked
variable's G_v) plus the rows that already differ from the pass's baseline
(`differing_`), not over every constraint (`StructuralBatch::delta_after_walk`).
`solve(model, time_limit)` is a library contract, and a large enough user model
overran it severalfold without the bound (#105, which has the measurements).

---

## ViolationLS Outer Loop

**File:** `src/search.cpp` (`solve`)

`solve()` implements the ViolationLS batch outer loop (paper Algorithm 6).

### Setup

1. If the model has an objective, add the `obj <= bound` soft constraint
   (idempotent) and reset its bound to `+inf`.
2. Construct the `ViolationManager`.
3. Unless `config.skip_init`, randomize the **List/Set** variables
   (`initialize_structured_random`).
4. Construct a single-phase `FeasibilityJump`; call `begin(set_initial_x)`, which
   sets every **scalar** to the domain value closest to 0.

### Who initialises what (#108)

Exactly one path initialises each variable, split by type:

| Variable type | Initialised by | To what |
|---------------|----------------|---------|
| Bool, Int, Float | `FeasibilityJump::begin(set_initial_x)` | the domain value closest to 0 (the published Feasibility Jump start) |
| List, Set | `initialize_structured_random` | a Set: a random subset of an admissible size. A List: whatever its `ListInit` says — a random permutation, empty, or a random subset of a random admissible length. A member of a `ListPartition` is laid out with its whole partition by `randomize_list_partition` instead, which under `Cover::Exact` ignores the per-list `ListInit` because the cover has to hold at the first assignment |

**The scalar starting point does not depend on the seed, and this is deliberate.**
Feasibility Jump specifies the closest-to-zero start, and `mipfeas` was built as
a head-to-head against CP-SAT's `fj` worker, so the start point was part of
what was compared; CP-SAT now runs at its defaults (#284), but the published
start stays. Two seeds on a
scalar-only model therefore begin at the *same* assignment — if you are reading
identical results across seeds as evidence that the search converged, check
whether it simply started there.

The seed still drives everything else: List/Set initialisation, best-of-N
scan-set sampling, diversification kicks, the per-batch `rho` draw, LNS destroy
sets and Novelty Jump.

Diversification is not a substitute for a randomised start: `perturb()`
randomises each jumpable variable only with probability `perturbation_probability`
(default **0.1**) and moves each List/Set variable by only
`round(p * |elements|)` structural moves, and it runs only after
`perturbation_period` (default **100**) stagnant batches or on the
unproductive-batch route (step 8 of the [Main Loop](#main-loop)).

`initialize_random` (which randomises scalars too) is a public utility for
callers who *want* a randomised scalar start; compose it with `skip_init`:

```cpp
RNG rng(seed);
initialize_random(model, rng);        // randomise everything, scalars included
SearchConfig cfg;
cfg.skip_init = true;                 // solve() keeps the assignment it is handed
solve(model, time_limit, seed, ..., cfg);
```

`initialize_random` is safe on any domain, unbounded ones included, because it
goes through the shared randomisation helper described below.

#### Randomising a variable (`include/cbls/randomize.h`)

Every uniform "randomise this variable" in the engine goes through one place:
`randomize_var` (scalar `value` or List/Set `elements`), which delegates to
`random_in_domain` for scalars and `randomize_structured_var` for the rest. Its
three callers are `initialize_random` / `initialize_structured_random`
(search.cpp), `LNS::destroy_repair`'s destroy step (lns.cpp) and
`FeasibilityJump::perturb`'s kick (feasibility_jump.cpp). Two more places use the
same *guard* (`domain_window`) without going through `randomize_var`: the move
generators in moves.cpp, which draw perturbations around a current value rather
than uniformly over the domain, and `int_jump_candidates` in feasibility_jump.cpp,
which is not a draw at all — it reads the window to decide which integers are
worth probing (#114). Nothing gates an unbounded Int *out* of an FJ
scan: `movable_domain` guards only the perturbation path, so a variable with no
jump candidates is still scanned, found scoreless, and left where it is.
Without the guard, one default-probability kick on a model with unbounded
Floats turned the assignment NaN for the rest of the run (#112).

`domain_window(var)` is the guard. It returns the finite `[lo, hi]` window the
draw actually samples, always a subset of the variable's domain:

- a **finite declared bound passes through untouched**, so a model with finite
  bounds keeps its exact draw sequence and its exact solve trajectory;
- an **infinite bound** is replaced by a clamp magnitude — `kRandomIntInfClamp`
  (1e6) for an Int, `kRandomInfClamp` (1e9) otherwise. These are the same
  magnitudes `NlToModelOptions` falls back on at load time, so a hand-built
  unbounded model lands in the same box a `.nl` one would have — on the columns
  the adapter cannot bound. The parity is `.nl`-only: `MpsToModelOptions` has no
  integer variant, so an unbounded MPS Int column falls back on `inf_clamp`
  (1e9) where a hand-built one gets `kRandomIntInfClamp` (1e6). The adapters
  reach that fallback only where bound propagation derives nothing. Unguarded,
  `uniform_real_distribution` returns NaN or +inf on an infinite bound
  (its precondition is `ub - lb <= DBL_MAX`), and an infinite Int bound casts
  to `INT64_MIN`;
- on a **half-infinite** domain the substituted end is pushed past the declared
  one where needed, so the window stays inside the domain even when the declared
  bound lies beyond the clamp magnitude;
- two finite bounds whose **width** overflows are narrowed to the clamp box.

Nothing else is rewritten, and in particular the window is **not** trimmed to the
`int64_t`-nameable range: clamping each Int bound independently is neither inert
nor a subset (`[-1e18, -1e17]` would become the single point `-2^53`, above the
declared `ub`). `int_sample_window(var)` does that
trim at the three places that actually cast — `random_in_domain`,
`random_different_in_domain` and `int_rand` — and reports *empty* when the domain
lies wholly past 2^53. Each caller decides for itself what that means:
`random_in_domain` draws from the untrimmed window (every double that large is
already an integer, so the draw is in-domain), `int_rand` drops the move, and
`movable_domain` reports immovable. `movable_domain` consults the window but does
not cast at all: `floor(hi) - ceil(lo) >= 1` is exact at any magnitude.

This closes the randomisation route into a non-finite assignment. It **does not
make the engine safe on unbounded domains**, and this section must not be read as
claiming it does: Float jump candidates are still built from `var.lb`/`var.ub`
directly, so a model with an unbounded Float can still settle on an infinite
value — and, because an infinite assignment can satisfy every row, be reported
`feasible = true` while carrying `±inf`. The *Int* jump path is closed (#114);
Float remains open.

`begin()`'s closest-to-zero start is likewise well defined on every domain — a
genuine advantage of letting FJ own the scalars, but **only for the starting
point**; the kick and LNS destroy need the window above.

`use_fj` is vestigial — GFJ is always the engine.

### Main Loop

While time and `max_iterations` remain, each pass:

1. **Pick the batch kind.** With probability `structural_batch_probability`,
   STRUCTURAL; else with probability `novelty_jump_probability` (only when
   `use_compound_moves`), NOVELTY JUMP; else FEASIBILITY JUMP. All drawn afresh
   every batch, where ViolationLS §5 keeps the algorithm until a new best or a
   perturbation; #209's pre-registered A/B chose this (see `pick_batch_kind`).
   Structural batches mutate state outside FJ's bookkeeping, so they set a
   `resync` flag; a Novelty batch rebuilds FJ's state itself.
2. **Run the batch.** `fj.batch(batch_iterations)`, `fj.novelty_batch(batch_iterations)`,
   or `StructuralBatch::run(...)`. Just before it, **portfolio only**, sync the
   shared bound: if the pool's best feasible objective earns a tighter
   bound than this worker holds, tighten to it (see
   [Parallel Search](#parallel-search), item 3). A no-op without a pool. Every
   batch kind is bounded by the wall-clock deadline from *inside*, not just at
   the loop top: FJ strides a check through its GLS loop, and the structural
   sweep checks between generators (see [Deadline bound](#deadline-bound)), so a
   batch entered just before the deadline cannot run to completion past it. LNS
   repair is bounded separately, by `remaining()` at its call site.
3. **On real feasibility**, `record_best()` *first*, banking the feasible point
   before anything can move off it; then run the inner solver hook (if any) to
   polish continuous variables, and `record_best()` again only if the polish
   stayed feasible.
4. **`record_best()`** keeps the assignment if it strictly improves
   `best_feasible_obj`, snapshots the state, and tightens the objective bound to
   the min of `obj - eps` and, in a portfolio, the shared cap
   (`eps = 1` on an integral objective, as CP-SAT's best - 1, else
   `eps = step*(|obj|+1)` with an **adaptive** relative `step` in
   `[1e-6, 1e-3]` (#241): it starts at 1e-3, doubles on each incumbent
   `record_best` records (twice in a batch whose float hook improves on FJ's
   point), and is divided by 10 at a full-period stall (step 8). The step doubles as the
   Newton step size for hook-less continuous descent, so a large one keeps the
   pace far from the optimum and a small one resolves near it -- see
   `objective_bound_below`). For pure-feasibility models (no objective)
   the first feasible solution ends the search.
5. **On a new best**, reset GLS weights to 1, resample `rho`, and **disarm the
   Float escape probe** — the latch `SearchResult::escape_probe_armed` reports.
6. **Otherwise** increment `stagnation`, and `resync()` if a structural or hook
   mutation happened.
7. **Arm the Float escape probe** if the search is stuck: either
   `stagnation >= perturbation_period` (armed at the same site as the kick in
   step 8), or — with a wall clock — a quarter of the budget elapsed since the
   last new best (#117).
8. **Diversify**, by either of two routes. Both restore `kick_origin()` first
   once there is one — the incumbent, or the state `adopt_from_pool` installed
   (see [Diversification](#diversification)).
   - **`stagnation >= perturbation_period`** — the ordinary one. On a
     continuous objective with an incumbent, first shrink the bound step
     tenfold (floor 1e-6) and re-ground the bound on the incumbent at it,
     capped by the shared bound: the one deliberate loosening, giving
     up a demand a whole period did not meet (the kick that follows rebuilds
     FJ's tables). Then diversify (below) and reset the stagnation counter.
   - **an unproductive Feasibility-Jump batch** (#102) — the batch reported that
     it stopped reducing the REAL rows' violation
     (`GFJConfig::unproductive_iterations`, default 300 iterations), and the
     outer loop's own count agrees at
     `stagnation >= perturbation_period / 20`. Diversify without arming the
     escape probe and **without** resetting the stagnation counter, so
     `perturbation_period` keeps meaning what it says. Once a feasible solution
     exists this route draws only a perturb, never an LNS destroy-repair: its
     progress measure cannot see the artificial `obj <= bound` row the search is
     then working against, so "no new all-time low" stops being evidence of a
     stall, and an LNS repair launched on that signal is bounded in *seconds* and
     is usually rejected. See `SearchResult::lns_repairs`, and
     `SearchResult::lns_repairs_accepted` for how many of those repairs the
     lexicographic accept rule actually kept.
9. Emit a progress callback (~1 s cadence, or immediately on a new best).

At the end, restore the best state — or, on an infeasible run, the *closest
approach* to the feasible region rather than the untouched initial assignment —
release the objective bound to `+inf`, full-evaluate, and return the best
feasible objective (or `+inf` / infeasible). `SearchResult::best_violation`
carries the largest real-constraint residual at the returned assignment.
`SearchResult::iterations` is the unit `max_iterations` is charged in: the
larger of the total GLS iteration count and the batch count, so a
structural-only run, which charges no GLS iteration, reports its batch count
rather than 0. The budget stops the run when either reaches the limit, so the
two agree by construction.
`SearchResult::termination` says which budget ended the run — see below.
`SearchResult::first_feasible_objective` and `time_to_first_feasible` record
where the run *arrived* in the feasible region, latched at the first
`record_best` on a feasible point, before that batch's inner-solver polish.

`SearchResult::counters` (a `SearchCounters`, `include/cbls/counters.h`) answers
the question the fields above cannot: not how much work a run did but **on what**.
Batches bucketed by `BatchKind` — and
`fj_batches + novelty_batches + structural_batches == batches` exactly, since
`pick_batch_kind` returns one of three and every batch is counted once — the
structural sweep's candidates tried and committed, one row per generator (keyed
by `MoveGenerator::name()` plus a `#<index>` suffix wherever one batch built two
of a kind, since the built-ins name themselves by type), the inner solver's call
count, and a portfolio's restart count. LNS stays on `lns_repairs` /
`lns_repairs_accepted` rather than being duplicated here.

Observational only. `inner_solver_seconds` reads **0.0** on a run without a
wall-clock budget, which reads no clock at all (see
[Wall-clock budget](#wall-clock-budget)); the call count is always filled.

### Wall-clock budget

`solve(model, time_limit)` is a promise to return within `time_limit`. Five
sub-steps could overrun it if entered just before the deadline; each is bounded
separately:

| # | Sub-step | Bound | Where |
|---|----------|-------|-------|
| 1 | Feasibility Jump batch | handed the same absolute deadline, checked inside the GLS loop on a stride bounded two ways: at most 64 iterations, and at most 1/64 of the *remaining* budget in predicted time (#113) | `gfj.time_limit = budget_seconds` |
| 2 | `InnerSolverHook` | not *started* when the budget is spent, and handed the loop's own `past_deadline()` as `solve()`'s `StopRef stop` (#191); `FloatIntensifyHook` polls it every `vars_per_poll()` Float variables (65 536 / max(rows, nodes), clamped to [1, 256]: a model-size proxy for a variable's cost, argued in `src/inner_solver.cpp`; 1 on rahue) and before every multi-variable Newton step. A custom hook that ignores `stop` is still unbounded — the call is synchronous | `if (hook && !past_deadline())`, `LoopStop` |
| 3 | LNS repair | handed `min(2.0, remaining())`, not its own independent 2s. Clock only: the repair does not see a peer worker's stop or a host cancel, so either can wait up to 2s for it | `diversify()` |
| 4 | STRUCTURAL sweep | checked between generators, one per structured variable by default; the overrun is one generator's candidates: a built-in `generate` call appends at most 5, `BestOfSample` and `ViolationGuided` call until they hold `structural_sample_size` (possibly overshooting), and `GranularBest` takes whole granular neighbourhoods topped up to that size; a registered generator's own count is not capped (#105, #165, #203) | `StructuralBatch::run` |
| 5 | diversification kick, structural half | checked between structural *moves*, on a stride bounded the same two ways as row 1: at most 64 moves, and at most 1/64 of the *remaining* budget in predicted time (#115) | `perturb_structural` |

Bound 3 has two halves, and only the lower one is about the deadline. The
`remaining()` half is the deadline bound proper: a kick taken near the end of
the budget gets only what is left. The `2.0` cap stops a single kick early in a
long run from monopolising it, and on any run whose whole budget is under 2s
(including every test here) it never binds. Only the `remaining()` half is
covered by a test. It is floored at `1e-9` rather than clamped to 0 while a
deadline exists: `remaining()` returns exactly `0.0` once the clock crosses the
deadline, and `0` means "no wall clock at all" downstream in
`fj_nl_initialize` — so clamping to zero would hand the repair an *unbounded*
run.

##### Bound 1: why the stride is sized in time (#113)

The GLS loop cannot read the clock every iteration: `steady_clock::now()` costs
~1.4 us on an **hpet** clocksource (~20–25 ns through the vDSO on **tsc**, so
the stride must not assume either), against GLS iterations that can be a few
microseconds. A *fixed* iteration stride is not a time bound either: one GLS
iteration is `O(sampled vars x candidate values x constraints touched)`, so 64
of them are microseconds on a small model and seconds on a large one.

Each check measures the interval since the previous one — the current cost of
an iteration — and sizes the next stride to `kStrideBudgetFraction` (1/64) of
the *remaining* budget (remaining, not total, so the bound tightens as the
deadline approaches). Growth is capped at `kStrideGrowth` (8x) per adjustment so
the ramp goes through progressively longer and more accurate measurements;
shrinking is uncapped, so an iteration that got more expensive is caught at the
very next check.

An uncapped shrink does **not** stop the stride ratcheting upward and going
silent: a shrink is only *applied* at a check, and the next check is a whole
stride away, so a stride grown while iterations were cheap is spent in full on
the first expensive one. That is what `kMaxDeadlineStride` (**64**) is for.

**Guarantee: a batch returns at most one stride past the deadline, where a stride
is at most 64 GLS iterations and at most 1/64 of the remaining budget in
predicted time — or one GLS iteration, whichever is larger.** The last clause is
irreducible: an iteration is atomic and cannot be pre-empted from the inside. The
prediction is from the last measurement, so a cost that rises mid-stride is
absorbed by the 64-iteration half of the bound, not the time half. The interval
between checks can span a batch boundary, so work the outer loop does between
batches (hook, LNS, structural sweep) is charged to the stride and shrinks it —
conservative, never the reverse.

Sizing the stride in time also bounds the clock overhead: one read per stride,
against a stride costing `budget/64`, is 1.4 us / (budget/64) even on the
expensive clocksource. It degrades only for budgets well under a millisecond.
The overrun and throughput measurements are on #113.

##### Bound 5: why the kick is checked between moves (#115)

The STRUCTURAL sweep (row 4) caps its overrun at one move-set evaluation per
variable, linear in that variable's size; the kick's per-variable cost is
`k = round(p * |elements|)` move-set *generations*, quadratic in it. A check
between variables therefore bounds nothing on a model whose structure lives in
one large List or Set, so the check sits *inside* the run, between moves. One
move — one move-set generation plus one apply, `O(|elements| + universe)`
element copies — is exactly the unit row 4 caps its overrun at.

A move is not cheap enough to check before every one (a move on a 100-element
List costs about one hpet clock read), so the check strides with bound 1's tuner,
`FeasibilityJump::next_deadline_stride`: sized in time from the last
measurement, growth capped at 8x, shrink uncapped, hard-capped at
`kMaxDeadlineStride`. The hard cap matters for the same reason as in bound 1: a
stride grown over many cheap small structures would otherwise be spent in full
on the first move of a large one. Bounding `k` against the remaining budget
instead would not work: `k` is chosen once, before the run, so a per-move cost
that rises *inside* the run is never re-observed.

**Guarantee: the structural pass applies at most 64 further moves after the
deadline passes, and a stride costs at most 1/64 of the remaining budget in
predicted time — or one move, whichever is larger.** The last clause is
irreducible for the same reason as bound 1's. The prediction is from the last
measurement, so a move whose cost jumps mid-stride is absorbed by the 64-move
half, not the time half.

The stride is armed at one move at the top of every kick rather than carried
across kicks, so the *first* variable is bounded too. It is kept separate from
the GLS tuner because the two loops advance in different units and interleave:
a kick runs between batches, so one shared tuner would have each mis-size the
other's stride. `arm_structural_kick()` runs on every kick, scalar-only models
included, so a deadline-armed run pays one `steady_clock::now()` per kick
(gated on `has_deadline_`, so determinism-safe).

Two deliberate exceptions to stopping. The check never fires before the pass has
applied a move, so a deadline already crossed on entry still leaves a kick
something to have done — the never-a-no-op contract of #109/#111, which `perturb`'s
fallback then completes if that one move happened to cancel itself out. And a
variable whose run was cut short still has its net effect recorded, so a
truncated kick is not mistaken for a no-op.

One gap, **deliberately left open**: the bound is stated in moves, but an
*attempted* move that finds no legal candidate is not free.
`generate_standard_moves` on a Set allocates a `vector<bool>` over the universe,
copies the membership and builds the complement — `O(|elements| + universe)` —
before discovering there is nothing legal. Because the check short-circuits on
"no move applied yet" without decrementing, a model of M saturated Sets
(`min_size == max_size == universe_size`) does `O(M * U)` work with zero clock
reads. That is linear in the model rather than quadratic in one variable.
Closing it means bounding failed *attempts* as well as moves.

##### No clock without a deadline

No clock read influences control flow when `time_limit <= 0` — `past_deadline()`,
`remaining()`, `StructuralBatch::run`, `perturb_structural` and FeasibilityJump's two
strided checks all short-circuit on their `has_deadline` flag — so with no
`SolveCallback` attached the loop reads no clock at all, and
iteration-budgeted runs stay bit-identical and deterministic. The reads that do
remain never reach the search trajectory: `solve()` timestamps entry and exit
for `time_seconds`; a callback's ~1s progress cadence reads once per batch;
`SearchResult::time_to_first_feasible` costs one read, latched at the first
feasible point. The `SearchCounters` timings (`inner_solver_seconds`;
`bound_behind_global_seconds`, portfolio control arm only) read the clock only
on a run with a wall clock, whether or not anyone reads them. A
`SearchConfig::tracer` is opt-in:
attached it costs one read per new best and two per inner-solver call.

#### How the bounds are tested

Benchmarks publish per-instance wall times, so a silently overrunning budget
corrupts published data; but the suite is deterministic (iteration-bounded,
`time_limit = 0`), and an `elapsed < X` assertion drags machine speed back in.
So bounds 1–3 and 5 are tested **without asserting on elapsed time** (#104):
each test observes its own bound in the unit the guarantee is stated in — work
done for 1, the call count of a test `InnerSolverHook` and the exact poll
sequence of `FloatIntensifyHook` under a counting stop for 2, the argument
handed to a test `LNS` for 3, and for 5 `structural_kick_moves()` /
`structural_kick_checks()`. The bound-1 tuner exposes
`deadline_check_stride()` / `deadline_checks()`, and its sizing rule
`FeasibilityJump::next_deadline_stride` is a pure function tested exhaustively
over a grid, plus two timing-free tests for the non-stationary (ratcheting)
case the grid cannot see. Bound 4's symptom really is duration, so its test
asserts on wall-clock duration, quarantined behind the `[timing]` tag as its own
labelled ctest entry with an explicit timeout; bound 2 has one such end-to-end
test too, `timing_inner_solver_hook_deadline`.

Every `[search][deadline]` test also asserts `SearchResult::termination`, so a test
whose model converged before the clock mattered cannot pass inertly. The FJ
batch and `fj_nl_initialize` tests fail in the *safe* direction: a slower machine does fewer
iterations and passes more comfortably. Why each test asserts what it does (the
margins, why the large-structure kicks assert exactly one move, why only the
small-structure test catches a countdown reloaded with a multiple of the stride)
is in the comments above the tests in `tests/`.

### Diversification

`diversify()` increments a `perturbations` counter and, every `lns_interval`-th
LNS-eligible kick (when an `LNS` is supplied), runs [LNS](#large-neighborhood-search)
destroy-repair (then resets FJ weights, since LNS mutates state outside the
engine). Otherwise it calls `fj.perturb(perturbation_probability)`. Either way
it resamples `rho`.

**The kick starts from the point the search is exploring around** once there is
one (#158) — `kick_origin()`: this worker's own incumbent, or the state
`adopt_from_pool` installed while the search is standing on a peer's point.
Before the first feasible solution, and while the only one on record is #100's
non-finite-objective witness, there is no origin and the kick fires in place.

**This is a deliberate divergence from the reference, not a bug fix.** Davies et
al., Algorithm 6 (the PDF in `docs/`) has exactly one restore, and it is not this
one:

```
 5   if a new best solution S is available in the shared pool then
 6       X ← S
11   if No new solutions found or imported for 100 iterations then
12       Perturb X, randomising each variable's value with probability 0.1
```

Line 12 perturbs `X` **in place**; line 6 is pool-sourced and conditioned on a
new *best* arriving, which is what `adopt_from_pool` already implements. Restoring
an origin first makes the outer loop elitist ILS with a strict-improvement
acceptance criterion instead. Since `perturb()` ends in `reset_weights()`,
neither the assignment nor the GLS landscape carries across a kick. The case for
it: #102's unproductive route kicks about 50x as often as Algorithm 6, so the
in-place walk is that much longer; measured before the restore, the search
typically left the feasible region after a kick and did not come back. Those
measurements, the held-out arm that admitted the change, and its LNS-acceptance
price (repairs now score against the origin, so far fewer are accepted) are on
#158.

**The restore is not gated on there being no pool**, so a portfolio worker kicks
from a restored point too. `adopt_from_pool` draws from the better half rather
than the best, so that workers stay spread, and leaves `best_state` alone when
the draw is not an improvement. Restoring `best_state` there would discard the
draw within a few batches, and the portfolio would degenerate into N elitist
searches. So adoption records the state it installed, under the same two rules
the improvement branch applies (feasible in *this* model, finite objective — the
pool also holds closest-approach states submitted with `feasible = false`, and
#100 witnesses with a non-finite objective). `tests/test_search.cpp`'s "a kick
after adoption departs from the adopted point, not our incumbent" pins it: it
goes red without the restore *and* with a `best_state`-only one.

`LNS::destroy_repair` snapshots the state it is handed, which once there is an
origin is the origin, so a rejected repair *moves* the search back to the origin and perturbs
nothing: not free, and not a diversification either, since FJ re-descends from a
point it has already converged on.

**Restart options (#239).** Two CP-SAT ViolationLS restart features, faithful
to OR-Tools main at `266f518`, were opt-in `SearchConfig` arms, off by default:
restart randomisation (the part of `LsOptions::Randomize` cbls lacked: weights
restarting at Uniform(1, 2) with probability ½, and a restore-only restart that
re-weights `kick_origin()` and moves nothing) and a Luby kick schedule in place
of #102's unproductive-batch trigger. Neither separated on the mipfeas smoke
A/B (the numbers are on #239), and #383 removed both. The code
(`restart_randomization`, `luby_restart_unit`, `luby()`), its tests and the
`--restart-randomization` / `--luby-restart-unit` runner flags are in git
history at `1f574b52`. The rest of `Randomize` is here: ρ is redrawn at each
kick and new best, and compound moves at each batch (#209).

**The scalar kick.** `perturb(p)` randomises each jumpable variable
independently with probability `p`, and then — **only if that moved nothing** —
forces one uniformly chosen variable to a *different* value, drawn from its
domain with the current value removed; a variable pinned by `lb == ub` cannot be
chosen. Without the fallback a kick on a small model is usually a no-op
(probability `(1-p)^n`: 81% on two variables at `p = 0.1`). As a fallback rather
than a forced variable, it almost never runs on a model large enough for `p` to
do its job, so the kick keeps the draw sequence it had without one.

**The structural kick (#111).** List and Set variables get their own pass in the
same kick; without it a kick on a structure-only model, such as `setcover`'s
`Set` encoding, randomises nothing. The pass applies
`k = max(1, round(p * |elements|))` random moves to **each** List/Set variable,
drawn from the [structural batch](#structural-batch)'s typed generators
(`generate_standard_moves`) plus, for a member of a `ListPartition`, that
partition's inter-list moves anchored on this variable
(`generate_partition_moves`, which is what lets a kick move an all-empty
`AtMostOnce` partition at all, #164). Its contracts:

- **Legal.** Every move is legal by construction: a List keeps its elements
  distinct and its length inside `[min_size, max_size]`, a partition member
  keeps its cover, and a Set stays inside `min_size`/`max_size`.
- **Net change.** No-op candidates (a relocate to the adjacent position) are
  filtered out before the draw, and a variable counts as moved only if its
  elements *net* changed.
- **Size.** `k` scales how much of each structure moves, since a structure has
  no "randomise the whole variable" analogue that is not a restart. It is at
  least 1 (even on an empty structure) and at most `|elements|` otherwise, so
  short of a deadline cut or a dead end every structure moves on every kick,
  `p = 0` included, and `p > 1` cannot make a kick unbounded. `|elements|` is the
  current membership: a 3-of-1000 Set is kicked on the 3 and cannot grow far in
  one kick; LNS, which resamples the cardinality, does that.
- **Local in adjacency.** `k` counts moves, not displaced slots: a 2-opt
  reversal rewrites a sub-range of positions but breaks two adjacent pairs, so
  the kick is local for a List the DAG reads pairwise (`pair_lambda_sum`) and
  much coarser than `p` suggests for one read positionally (`at`). Measurements
  are on #111.
- **Deadline-bounded.** Cost is `O(k * (|elements| + universe))` (a membership
  scan per candidate) per structure, superlinear in one structure's size, so
  the pass checks the deadline between *moves*, on a capped stride, and never before it has moved
  something; see [Bound 5](#bound-5-why-the-kick-is-checked-between-moves-115).
- **No draw on scalar-only models.** The pass tests the variable type before
  drawing, so a model without List/Set variables keeps its exact draw sequence,
  pinned by a test that runs the same scalar model with and without an
  immovable List and requires identical assignment sequences.

The kick's guarantee is model-wide: if anything can move, something moves. On a
mixed model the structures move unless dead-ended, so the scalar half is left to the
configured probability and the forced-scalar fallback fires only when nothing
moved at all. A model with no movable variable anywhere still changes nothing.

### SearchConfig

`SearchConfig` (`include/cbls/search.h`) carries every outer-loop knob. Its
fields and defaults are listed once, in the [Parameters Table](#parameters-table);
the header is the source of truth and carries each field's reasoning.

### SolveProgress

```cpp
struct SolveProgress {
    int64_t iteration = 0;        // batch count at emission
    double time_seconds = 0.0;
    double objective = inf;       // best feasible objective so far
    double total_violation = 0.0; // current weighted violation
    bool feasible = false;        // current assignment real-feasible
    bool new_best = false;
    int perturbations = 0;        // diversification kicks so far
};
```

---

## Inner Solver (Continuous Intensification)

**Files:** `include/cbls/inner_solver.h`, `src/inner_solver.cpp`

The `FloatIntensifyHook` implements `InnerSolverHook` — a constraint-directed NLP
sub-solver for continuous variables. GFJ explores the discrete/structural space
while the inner solver tightens continuous variables, avoiding treating them as
discretized jump candidates (which would need impractically fine granularity).

### Trigger

The hook fires **on each feasible batch**, immediately *after* `record_best()`
has banked the point and tightened `obj <= bound`, so it descends against the
tightened bound. Because the hook may move the assignment off the feasible
region, the loop re-checks `real_feasible()` before recording the polished
assignment; the pre-polish point is safe either way.

It descends `ViolationManager::augmented_objective()` = `obj + total_violation()`
and mutates Float variables directly.

### Per-variable cost (#207)

Each coordinate step on a Float variable v reads only what v can move:

- **Slopes:** one forward-mode pass over v's up-cone (`UpConePartials`, with
  no node budget) gives d(row)/dv for every row of G_v and d(objective)/dv. The
  pass costs O(u log u) for an up-cone of u nodes. A Sum's operand multiplicity
  comes from `close()`'s `kRepeatedChild` flag, so an objective Sum over n terms
  is not scanned once per variable.
- **Scoring:** a candidate is scored by the augmented objective's change over
  G_v plus the objective (`AugmentedDelta`), differenced per row the way
  `weighted_delta_from` does it, instead of by two full `augmented_objective()`
  readings.
- **Probes and commits:** these go through `probe_scalar_move` / `Rollback` and
  `commit_scalar_move`, so an incremental Sum (#188) above v moves by its one
  changed term rather than being re-summed, and the Rollback writes the probe's
  undo log back rather than walking the cone a second time (#270). `solve()` re-grounds every drifted
  Sum before the multi-variable pass, which reads raw, and again before it
  returns, early stop included, so the search's raw reads after the hook see
  re-grounded values.

A coordinate sweep is therefore O(Σ_v up-cone(v)), which wins over the
per-variable reverse sweep it replaced (O(Float vars × nodes)) on a separable
objective over many variables (UC-CHPED, any linear objective), where each
up-cone is a handful of nodes. It gains nothing on a dense objective in a few
variables (MINLPLib's least-squares shapes), where each up-cone is about the
whole objective cone. Mechanism 3 still costs O(model) per step, at most
`max_multi_var_constraints` steps per sweep.

### Mechanism 1 — Single-Variable Newton Steps

For each Float variable, examine its own violated rows (G_v, in ascending
order) and stop after 3 Newton candidates. For each row,
`g = constraint_value` and `dg = d(constraint)/d(var)`, taken from the forward
slopes; if `|dg| > 1e-12`, propose `clamp(var - g/dg, [lb, ub])` and keep it if
the augmented objective improves.

### Mechanism 2 — Backtracking Line Search on Objective

For each Float variable, `df = d(objective)/d(var)` (forward slopes); if `|df| > 1e-12`, try
`clamp(var - step*df, [lb, ub])` for `step = initial_step_size`, halving up to
`max_line_search_steps` times, keeping the best improvement (Armijo-style).

### Mechanism 3 — Multi-Variable Minimum-Norm Newton

For up to `max_multi_var_constraints` violated constraints, take one reverse AD
pass (`compute_all_partials`) and step *all* Float vars simultaneously by the
minimum-norm Newton solution to the linearized constraint `g + grad . dx = 0`:
`dx_j = (-g / ||grad||^2) * dg_j`, clamped to bounds. Requires >= 2 Float vars
with non-negligible gradients; accepted only if `augmented_objective` improves,
else fully rolled back.

Sweeps repeat up to `max_sweeps` times, stopping early when a sweep makes no
improvement, or when the search's `stop` is raised (#191: the deadline, a peer
worker's stop or a host cancel), polled every `vars_per_poll` variables
(`src/inner_solver.cpp`, sized by model size) and before each Mechanism 3 step.
The poll changes only whether the descent continues, so a pass that finishes within budget is unaffected by it.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_sweeps` | 3 | coordinate-descent sweeps over Float vars |
| `initial_step_size` | 0.1 | starting step for the line search |
| `max_line_search_steps` | 5 | max backtracking halvings |
| `max_multi_var_constraints` | 5 | max violated constraints for multi-var Newton |

---

## Large Neighborhood Search

**Files:** `include/cbls/lns.h`, `src/lns.cpp`

### Trigger

LNS fires from the outer loop's `diversify()` — every `lns_interval`-th
LNS-eligible diversification kick (a kick happens after `perturbation_period` stagnant
batches). When no `LNS` object is supplied, diversification is plain
perturbation instead.

### Destroy Phase

`destroy_repair` snapshots the state and its lexicographic key. Every
`ListPartition` is then **ruined and recreated** (#202), ASB-RR/SISR's
adjacent string removal and greedy insertion with blinks (Christiaens & Vanden
Berghe, KU Leuven technical report 2016; Transportation Science 54(2) 2020),
with the reference's parameters (c̄ = 10, L_max = 10, split string at 0.5,
split depth 0.01, blink rate 0.01):

- **Ruin.** A random assigned element is the seed; adj(seed) is the seed, its
  row of `SearchConfig::structural_neighbours` (handed over by `solve()` through
  `LNS::set_neighbours` before each kick), then every other assigned element in
  random order. Walking adj(seed), each element whose list is not yet ruined
  takes a string -- or a split string, which keeps a run inside its window --
  out of that list, until k lists are ruined, k drawn from the mean list
  length as the reference's Eq. 3-5 draw it. No list goes below its
  `min_size`.
- **Recreate.** The removed elements, in random order, each go to the best
  position over the lists with room (random list order, each position skipped
  with the blink rate), scored lexicographically on (real violation added on
  the list's rows, objective) -- the acceptance key's own rule -- by a DAG
  probe and rollback per position. Under `Cover::AtMostOnce` leaving the
  element out is a candidate too. At the repair deadline the rest go in
  unevaluated (`Exact`) or stay out (`AtMostOnce`).

Two departures from the reference, both because the engine does not know what
an element means: adj(seed) is a k-nearest row plus a random tail rather than a
full distance order, and the removed set is reinserted in the reference's
Random order only (its Demand, Far and Close orders are routing quantities).

Every variable **outside** a partition (with no partition, every variable, on the
same random draws) is then destroyed and repaired:

- **Sequence-aware** (when the model registers `var_sequences`): pick
  `ceil(n_seqs * destroy_fraction)` whole sequences plus a proportional fraction
  of non-sequence vars.
- **Uniform** (no sequences): `max(1, floor(num_vars * destroy_fraction))`
  random vars (default `destroy_fraction = 0.3`).

Destroyed variables are re-randomized by type through the shared `randomize_var`
(see "Randomising a variable" under [Who initialises what](#who-initialises-what-108)): Bool/Int/Float uniform over
the guarded domain window, List a reshuffle of its current elements
(`ListOrder::Perturb` — destroy/repair perturbs an incumbent, so the elements
present are preserved and only their order changes), Set a random
valid-cardinality subset.

### Repair Phase

When the scalar destroy chose anything, `full_evaluate`, then
`fj_nl_initialize(model, vm, 2000, &rng)` — a single-phase
GFJ repair (2000 iterations) refining the current (partially destroyed)
assignment toward feasibility, leaving a clean unit-weight penalty landscape.

### Acceptance — Lexicographic (real-violation, objective)

LNS computes a `state_key = (real_violation, objective)` where `real_violation`
**excludes the artificial `obj <= bound` soft constraint** (so the objective is
not double-counted). It accepts the repaired solution iff its key is
lexicographically smaller than the saved key — feasibility first, then objective
— mirroring `solve()`'s `record_best`. Otherwise it rolls back. This is
strictly improving in the (feasibility, objective) order.

`destroy_repair_cycle(n_rounds)` runs multiple rounds and returns the count of
accepted improvements.

---

## Solution Pool & Parallel Search

**Files:** `include/cbls/pool.h`, `src/pool.cpp`

### Solution Pool

A bounded, sorted collection of solutions, shared by every worker of a
`ParallelSearch` and guarded by one mutex. It is the only object the workers
touch concurrently.

**Sort order:** feasible solutions first; among same feasibility, ascending
objective. **Capacity:** `ParallelConfig::pool_capacity`, default 0 = auto =
`max(10, 2 * workers)` — the workers that actually run, so an `executor`
narrower than `n_threads` narrows this too — with the excess trimmed after each
insert. **Restart selection:** `get_restart_point()` samples
uniformly from the better half of the pool rather than the single best. That
reduces the pull toward one basin without preventing it: `submit` applies no
diversity criterion and no per-worker quota, so on an objective model the pool
tends to fill with one worker's successive refinements of a single point. Three
structural alternatives (a per-worker reserved slot, a per-worker tabu set, a
distance-weighted draw) could not be separated from this rule (#161); the
numbers and protocol are in the comment on `SolutionPool::get_restart_point`.

### Parallel Search

`ParallelSearch` is a **cooperative portfolio**: N threads (default
`hardware_concurrency()`), each with its own `Model` and running `solve()` on it.
The search itself is still single-threaded per solve; what is parallel is the
portfolio, and what is shared is solutions and the model's immutable structure,
never search state. Thread safety is by isolation plus one mutex, plus the
const-handle split described below.

#### Two ways in, and only one of them shares (#157)

- **`solve(Model& master, ...)`** — the overload to reach for. It `freeze()`s the
  master on the calling thread, before any worker exists, and hands each worker a
  copy. A copy of a frozen model shares the structure by reference and duplicates
  only what a search writes: the variables, the node values, the objective bound.
  The master itself is never searched, so it comes back frozen and carrying the
  `obj <= bound` row with its own assignment untouched — which is why `src/cli.cpp`
  and `benchmarks/mipfeas/` restore the winner into it before reporting.
- **`solve(std::function<Model()>, ...)`** — the factory overloads. These share
  whatever the returned model shares, so a factory returning a copy of a
  **frozen** model shares the structure too (the master overload is exactly that
  factory); `benchmarks/mipfeas/` does so to keep the per-worker copies out of
  the search's deadline. A copy of an **open** model, or a fresh one per call,
  deep-copies the DAG per worker: `freeze()` is the operative call, not the
  overload. From Python, freeze a master and return it from every call;
  nanobind's by-value cast copies it into the sharing copy constructor. Python
  binds the master overload as `ParallelSearch.solve_master` and the factory one
  as `solve_parallel`.

Callers: `src/cli.cpp` and the `cvrp`, `uc-chped` and `minlplib` runners at
`--threads > 1` use the master overload (#262); `benchmarks/mipfeas/` uses the
factory, as above.

Sharing roughly halves a worker's marginal memory on a large model, at no
throughput cost (#157; per-worker terms: #176). What remains per worker is
genuinely per-worker: the variables, the node values, at least three `Model::State` snapshots, FJ's per-variable and
per-constraint tables, `ViolationManager`'s two per-constraint vectors, FJ's
`LinearJumpScorer` (its slope table is structure, but each worker builds its
own) and `dag_ops.cpp`'s `thread_local` adjoint scratch, which that scorer's
slope build allocates on a pure MIP too. The shared pool's `max(10, 2N)`
`Solution`s grow with N too. So the criterion to hold this to is
"1-worker + N x a measured per-worker constant", not "1-worker + a small
constant" — and treat that constant as a **floor**: its largest term scales with
the column count.

Four things make it cooperative rather than N independent runs, and all four
are reached through one parameter -- `cbls::solve()`'s trailing
`SearchCoordination*`, which is null at every call site outside this class
(`setcover` always, the other runners at `--threads 1`), where null leaves
the single-threaded trajectory bit-identical:

1. **Submit when found.** `record_best()` shares each new incumbent into the
   pool the moment it records one, not once at the end of the run: one
   `Model::State` copy, made outside the lock, and one mutex acquisition per
   improving batch.
2. **Restart from the pool.** On the full-stagnation route
   (`stagnation >= perturbation_period`) a worker draws `get_restart_point()`
   and restarts from it instead of perturbing its own assignment, and that
   adoption *is* the diversification kick. It re-grounds: `restore_state` ->
   `full_evaluate` -> re-derive the objective bound from the adopted point ->
   `vm.invalidate_cache()` -> `fj.reset_weights()`. It is deliberately *not*
   wired to #102's unproductive-batch route, which can recur every batch.
   Three qualifications, each load-bearing:
   - every `lns_interval`-th kick stands adoption down, so the LNS cadence
     keeps running: `diversify()` is the only caller of `LNS::destroy_repair`
     and the only route that may draw LNS once a feasible solution exists;
   - adoption **disarms** the Float escape probe `maybe_diversify` just armed,
     since that diagnosis was about the assignment just left; so of #117's two
     arming routes, workers with a non-empty pool keep only the time-based one
     (#117's single-threaded roster does not measure this);
   - adoption is **refused** on an empty pool, a state whose shape does not
     match the local model, or a draw equal to the worker's assignment, and
     the kick falls through to the ordinary `diversify()`.
3. **Share the objective bound** (#179). The pool keeps its best feasible,
   finite objective `G` in an atomic, written under the mutex by `submit` and
   read without it. At each **batch boundary** a worker whose bound is looser
   than `G - eps(G)` -- the same `eps` rule `record_best` uses -- tightens to
   it, then `fj.resync()`, so a worker behind a peer stops searching for points
   the portfolio has already beaten. The rules, argued at
   `sync_shared_bound`, `record_best` and in `solution_pool.h`:
   - **never loosen past the shared cap.** The boundary test is strict, and
     every other writer of the bound during a solve (`record_best`, the
     adoption and the stall (#241) re-groundings) takes the min with the shared cap
     at the current step; the witness sentinel writes only over `+inf`. The one
     deliberate loosening is the stall re-grounding on a continuous objective,
     which gives up the part of a tightening that was the step, never a peer's
     improvement. (`solve()` opens at `+inf`, so a portfolio restart
     re-tightens at its first boundary.)
   - **only a peer moving tightens.** Each writer records the `G` its cap came
     from; the boundary tightens (and counts in `shared_bound_tightenings`)
     only for a `G` below it, never for the worker's own best, and
     `record_best` leaves the bound alone for an own best behind an unchanged
     `G`.
   - **the bound moves, nothing else does.** Not the assignment, not
     `best_feasible_obj` or `best_state` (`finish()` must return a point this
     worker found, so `record_best` still tests against its OWN best, and a
     point between `G` and it is still recorded, shared and returned), and not
     the GLS weights (`resync`, not `reset_weights`). With sharing on, such a
     point is no *improvement* to the loop unless the bound moved: no weight
     or stagnation reset, which would postpone the adoption that moves a
     lagging worker.
   - **it reaches workers that are not feasible yet.** The first peer to go
     feasible puts every worker's objective row in play, where single-threaded
     ViolationLS keeps it vacuous until the worker's own first feasible point.
     Deliberate; its cost can show only in the Primal Integral and gap, not in
     portfolio feasibility.
   - **only feasible, finite objectives count** toward `G`, never the pool's
     closest-approach states or #100's `+inf` witness. `G` is not re-evaluated
     on the reader's model; a factory returning different models misdirects
     the row's pressure but cannot corrupt an incumbent.
   - **the row's weight is never masked**: two-phase FJ would zero it, but
     `solve()` runs FJ single-phase, and the bound is only written when the
     model has an objective row.

   `ParallelConfig::share_objective_bound` (default on; `cbls_mipfeas
   --no-share-bound`) switches it off, which is the A/B control.
   `SearchCounters` reports its engagement in both arms
   (`shared_bound_tightenings`, `own_best_behind_global`,
   `bound_behind_global_batches` / `_seconds`; each documented in
   `include/cbls/counters.h`). A lone worker's single solve never fires any
   of it; a portfolio restart does, because the pool then holds the worker's
   own earlier solve, which the counters count as the global best.
   `--threads 1` passes no coordination at all.

   #179 holds the A/B (MINLPLib better, MIPfeas not separable) and one
   unmeasured risk: a large bound step can leave FJ stalled on a tie that
   weight bumps do not break, and sharing makes large steps routine.
4. **No idle workers.** Each worker is a restart loop over the *shared*
   deadline, not a single `solve()`: a run that returns with budget left (an
   exhausted `SearchConfig::max_iterations`) is restarted on the time its
   predecessor left, with `skip_init = true` so it keeps the assignment it
   converged to. Seeds come from `portfolio_worker_seed(base, worker, restart)`
   -- a splitmix64 finalizer over the triple, not a sum -- so a restart is never
   a replay of the run that just stalled AND adjacent `--seed` values give
   genuinely different portfolios. A worker whose `solve()` throws is retried
   a bounded number of times rather than left to idle its core; a factory that
   throws is not retried, since it will not succeed on a second ask. The one
   case where finishing early is correct -- a pure-feasibility model, whose first feasible
   solution *is* the answer -- raises `SearchCoordination::stop`, which every
   other worker observes at its next batch boundary and reports as
   `TerminationReason::Stopped`.

The best solution across threads is returned, prioritizing feasibility then
objective. Worker exceptions are caught (letting one escape a thread function
is `std::terminate`), but not discarded: if *every* worker throws, `solve()`
rethrows the lowest-indexed one rather than returning a default,
infeasible-looking result. A partial failure is absorbed -- the survivors'
best is returned -- but not silently (#170): the result carries
`workers_launched`, `workers_completed` and one `WorkerFailure` per worker that
did not complete, because an exception ended it or the shared deadline starved
it before its first solve (`SearchResult` in `include/cbls/search.h` defines
each case). The MIPfeas runner publishes the count per row and its scorer
refuses a row with fewer completed workers than `threads`.

`ParallelSearch::solve()` takes hook and LNS *factories* (these objects are
stateful and per-model); each worker gets its own instance, built on that
worker's own thread, **once per worker and not once per restart**. Return a
fresh object per call: the search never locks the hook or the LNS, so one
object shared by workers is a silent data race. Both return a
`std::shared_ptr` rather than a `unique_ptr` because nanobind cannot
relinquish a Python-constructed instance to a default-deleter one, so this one factory
type serves C++ and Python callers (#129; reasoning in
`include/cbls/pool.h`).

A *Python* factory is narrower than it looks. Neither
`InnerSolverHook` nor `LNS` has a nanobind trampoline, so C++ dispatches
through the base vtable and a Python override of `solve` or `destroy_repair`
is never called by the search. `LNS.destroy_repair` is the trap: it *is*
bound, so an override works when called from Python and silently does not
when the search calls it from C++. So these factories, like the `hook`
and `lns` arguments of the single-threaded `solve()`, buy a separately
configured `FloatIntensifyHook` or `LNS`, not a Python implementation (#132).

---

## Design Decisions

### Why ViolationLS instead of Simulated Annealing

ViolationLS replaces the tuning knobs of the simulated-annealing core it was
ported from (temperature, cooling, penalty multiplier) with one
self-adapting mechanism: per-constraint GLS weights driving a greedy
best-of-N jump. That gives a parameter-light core that adapts feasibility
pressure per constraint, and folds optimization into a sequence of feasibility
problems (objective-as-constraint) so the same engine handles both.

### Search core: Generalised FJ vs SA / tabu / WalkSAT

**Chosen:** GFJ — greedy on weighted violation, best-of-N sampled scan set, GLS
weights, gradient-informed Float jumps. Operates directly on the nonlinear
expression DAG via `delta_evaluate` and reverse-mode AD; no linear `Ax` form
required.

**Alternatives:** SA (now removed) trades determinism for uphill exploration via
temperature. WalkSAT-style focused repair (pick a violated constraint, fix its
best variable) is close in spirit to a single FJ jump but less general. Tabu
search would add short-term memory; GLS weighting already provides a longer-term
escape mechanism.

### Compound moves: Novelty Jump vs plain restarts

**Chosen:** Novelty Jump (bounded-backtracking compound moves with novelty
weights) to escape FJ local optima where only a sequence of moves improves.
On by default (see the Status note under Novelty Jump); the UC-CHPED runner
turns it off.

**Alternative:** rely solely on perturbation/LNS diversification. Simpler, but
cannot find the chained-invariant fixes Novelty Jump targets.

### Objective handling: soft-constraint bound tightening vs penalty multiplier

**Chosen:** fold the objective in as `obj <= bound`, tighten on each new best.
Optimization becomes a sequence of feasibility problems the same GFJ engine
solves; no penalty-multiplier tuning.

**Alternative:** the old `obj + lambda * violation` penalty with an adaptive
`lambda` (removed). Required balancing two adaptation thresholds and a global
multiplier against per-constraint pressure.

### Continuous variables: gradient jumps + inner solver vs discretization

**Chosen:** GFJ proposes Newton-toward-root Float jumps; an inner solver does
the heavy continuous polish on each feasible solution. Keeps Float handling
gradient-based without discretizing the domain.

**Alternative:** discretize Float domains into Int-like candidates — impractical
granularity, and loses the constraint-root information AD provides.

### Diversification: perturbation + LNS vs population/restarts

**Chosen:** random perturbation of scalars and structures alike (never a no-op)
by default; LNS (destroy + GFJ repair, lexicographic accept) every
`lns_interval`-th kick.

**Alternative:** population-based search (GA, scatter search) or systematic
restart schedules (Luby). The pool gives warm restarts *across* the workers of
a `ParallelSearch`; a single-threaded `solve()` gets its form of the same idea
from each kick, once there is an incumbent, restoring it first
([Diversification](#diversification)).
A Luby schedule was an opt-in arm until #383 removed it (in git history at
`1f574b52`): the smoke A/B (#239) found it no better than #102's stall trigger
and slower to a first feasible point.

---

## Control Flow Diagram

```
solve(model, time_limit, seed, use_fj, hook, lns, lns_interval, callback, config, coord)
│
├── model.freeze(config.cone_cache_bytes)   # close if open, objective row, cone cache
├── [if objective] set_objective_bound(+inf)
├── set_domain_tolerance(config.feasibility_tolerance)
├── ViolationManager vm(model)
├── [unless config.skip_init] initialize_structured_random(model)   # List/Set only
│
├── FeasibilityJump fj(model, vm, rng, single-phase)
├── fj.begin(set_initial_x = !skip_init)
│       └── set X near 0, full_evaluate, W <- 1, rebuild V/Q
│
└── outer loop (while time && max_iterations remain):
    │
    ├── pick batch kind:
    │     P(structural_batch_probability)        → STRUCTURAL
    │     elif use_compound_moves & P(nj_prob)   → NOVELTY JUMP
    │     else                                   → FEASIBILITY JUMP
    │
    ├── run batch:
    │     FJ:         fj.batch(batch_iterations)      # GLS: best-of-N jump + weight bump
    │     NOVELTY:    fj.novelty_batch(batch_iters)   # GLS over ApplyNoveltyJump
    │     STRUCTURAL: structural_.run()               # move generators; resync
    │
    ├── if max_real_violation() <= config.feasibility_tolerance:
    │     ├── record_best()                           # bank it BEFORE polishing
    │     │     └── tighten objective bound to obj - eps; snapshot best state;
    │     │        [if coord->pool] share the snapshot with the other workers
    │     └── [if hook] hook->solve(model, vm)        # continuous polish; resync
    │           └── if still real_feasible(): record_best()
    │
    ├── if new best:  stagnation=0; fj.set_escape_probe(false); fj.reset_weights()
    │                 resample rho
    │                 (pure feasibility → break)
    ├── else:         ++stagnation; if resync flag: fj.resync()
    │
    ├── [if wall clock] arm the Float escape probe once 25% of the budget has
    │     elapsed since the last new best (#117) — the batch-count arming
    │     below is unreachable when a batch costs seconds
    │
    ├── if stagnation >= perturbation_period:
    │     arm the Float escape probe; shrink the continuous bound step and
    │     re-ground the bound on the incumbent (#241), then EITHER
    │       [not an lns_interval-th kick, and coord->pool holds a usable state]
    │       adopt_from_pool():
    │         restore_state; full_evaluate; re-derive the objective bound;
    │         fj.reset_weights; disarm the probe
    │         resample rho; ++perturbations; stagnation=0
    │       OR diversify():
    │         [if there is one] restore kick_origin()   # incumbent, or the
    │                                                   # adopted state (#158)
    │         every lns_interval-th kick → full_evaluate; lns->destroy_repair();
    │                                      fj.reset_weights()
    │         else                       → fj.perturb(perturbation_probability)
    │         resample rho; ++perturbations; stagnation=0
    │
    └── emit progress (~1s cadence, or on new best)

    ── end loop ──

    restore best_state; release objective bound to +inf; full_evaluate
    return SearchResult{ best_feasible_obj, feasible, best_state,
                         iterations = fj.iterations(), time_seconds }
```

---

## Parameters Table

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `time_limit` | 10.0 | `solve()` arg | total search time (seconds); `<= 0` disables the wall clock entirely, leaving `max_iterations` as the only budget (deterministic) |
| `seed` | 42 | `solve()` arg | RNG seed; drives List/Set init, sampling, kicks, `rho`, LNS — **not** the scalar starting point (#108) |
| `use_fj` | true | `SearchConfig` | vestigial (GFJ always the engine) |
| `max_iterations` | 0 | `SearchConfig` | GLS-iteration cap (0 = use time_limit) |
| `skip_init` | false | `SearchConfig` | keep the current assignment whole — suppresses both List/Set randomisation and FJ's scalar start (portfolio restarts, caller-supplied starts) |
| `batch_iterations` | 1000 | `SearchConfig` | GLS iterations per FJ batch; also sizes a Novelty batch, charged 3 per move and 1 per bump |
| `perturbation_period` | 100 | `SearchConfig` | stagnant batches before a diversification kick |
| `perturbation_probability` | 0.1 | `SearchConfig` | per-var scalar randomisation probability on perturb; also scales the List/Set moves per kick (a no-op kick moves one var anyway) |
| `structural_batch_probability` | -1 (auto) | `SearchConfig` | P(structural batch); auto 0.33 when a generator would be built, else 0; auto is 1 when FJ has no movable, row-read variable (#201) |
| `use_compound_moves` | true | `SearchConfig` | enable Novelty Jump batches (the UC-CHPED runner turns it off) |
| `novelty_jump_probability` | 0.5 | `SearchConfig` | P(Novelty Jump batch) when enabled |
| `lns_interval` | 3 | `SearchConfig` / arg | LNS fires every Nth diversification kick |
| `move_generators` | empty | `SearchConfig` | extra structural move generators, cloned per worker (#165) |
| `default_structural_generators` | true | `SearchConfig` | also register the built-in List/Set generators |
| `structural_selection` | `Auto` | `SearchConfig` | how a generator's candidates become a commit; `Auto` is `GranularBest` with `structural_neighbours`, else `FirstImprovingSample` (#165, #203) |
| `structural_sample_size` | 8 | `SearchConfig` | candidates per generator under the sampling policies; a neutral placeholder, not measured |
| `structural_neighbours` | null | `SearchConfig` | k-nearest neighbour list for the built-in generators; setting it makes `Auto` `GranularBest` (#203) |
| `cone_cache_bytes` | 256 MiB (`Model::kDefaultConeCacheBytes`) | `SearchConfig` | cap on the frozen model's cone cache (#271) |
| `unproductive_iterations` | 300 | `SearchConfig` → `GFJConfig` | end an FJ batch after this many GLS iterations without a new real-violation low, and take the kick as due (#102); `<= 0` = fixed cadence |
| `fj_objective_tie_break` | true | `SearchConfig` | OR-Tools' affine-objective tie-break before the first feasible point (#218, #226) |
| `stop` | none | `SearchConfig` | caller's cancellation flag, read at batch boundaries (#169) |
| `tracer` | null | `SearchConfig` | caller's event sink, per worker (#169) |
| `auditor` | null | `SearchConfig` | test-facing consistency hook (#225) |
| `rho` (GLS decay) | {0.95, 1.0} | sampled per batch | GLS weight decay factor |
| `sample_size_general` | 5 | `GFJConfig` | best-of-N scan-set sample, general phase (ViolationLS §3, #211) |
| `sample_size_linear` | 5 | `GFJConfig` | best-of-N scan-set sample, linear phase |
| `two_phase` | true (solver) / false (outer loop, warm-start) | `GFJConfig` | linear-first GLS pass |
| `set_initial_x` | true | `GFJConfig` | set X to domain value nearest 0 first |
| `kCompoundDiscount` | 1/1024 | `feasibility_jump.cpp` | Novelty-weight discount on satisfied constraints |
| `kNoveltyWorkBudget` | 256 | `feasibility_jump.cpp` | max moves per `apply_novelty_jump` |
| `kEscapeArmFraction` | 0.25 | `search.cpp` `solve` | fraction of the wall-clock budget without a new best that arms the Float escape probe (#117). Not a `SearchConfig` knob |
| objective bound step (not a knob) | 1 (integral objective) / step·(abs(obj)+1), step adaptive in [1e-6, 1e-3] (#241) | `search.cpp` `objective_bound_below`, `kObjStep*` | bound tightening / hook Newton step |
| `destroy_fraction` | 0.3 | `LNS` ctor | fraction of variables/sequences destroyed |
| `repair_time_limit` | 2.0 s | `LNS::destroy_repair` | FJ repair budget; `<= 0` = iteration-bounded only |
| `feasibility_tolerance` | 1e-6 | `SearchConfig` / `kDefaultFeasibilityTolerance` | violation below which a constraint counts as satisfied |
| LNS repair iterations (not a knob) | 2000 | `lns.cpp`, literal in `fj_nl_initialize` call | GFJ iterations in LNS repair |
| `max_sweeps` | 3 | `inner_solver.h` | inner-solver coordinate-descent sweeps |
| `initial_step_size` | 0.1 | `inner_solver.h` | line-search starting step |
| `max_line_search_steps` | 5 | `inner_solver.h` | max backtracking halvings |
| `max_multi_var_constraints` | 5 | `inner_solver.h` | max constraints for multi-var Newton |
| `pool_capacity` | 0 = max(10, 2·workers) | `ParallelConfig` | max solutions in the shared pool |
| `share_objective_bound` | true | `ParallelConfig` | workers tighten `obj <= bound` to the pool's best at batch boundaries (#179) |
| `n_threads` | 0 = hw_concurrency | CLI / `ParallelConfig` | portfolio workers |

---

## I/O & Logging

### CLI

**File:** `src/cli.cpp`

The `cbls` executable loads a JSONL model file, solves it, and prints results.

```
cbls [OPTIONS] MODEL.cbls
```

| Option | Description |
|--------|-------------|
| `--time-limit SECS` | maximum solve time (default: 10.0) |
| `--seed INT` | RNG seed (default: 42) |
| `--no-fj` | set `SearchConfig::use_fj = false` (vestigial; GFJ still runs) |
| `--lns FRACTION` | enable LNS with given destroy fraction (e.g. 0.3) |
| `--lns-interval INT` | LNS fires every N diversification kicks (default: 3) |
| `--intensify` | enable the Float intensification hook |
| `--threads N` | number of search threads (default: 0 = one per core) |
| `--format human\|jsonl` | output format (default: human) |
| `--quiet` | suppress progress, print only the final result |
| `--help` / `--version` | usage / `cbls::kVersion` (`include/cbls/cbls.h`) |

Any other option exits 1.

### What the parallel path does to the output

`--threads` defaults to 0, so **this is the default reading of `cbls`'s output**.
Five things differ from a `--threads 1` run, in both the human and the JSONL
format:

- **`iterations` is a sum over workers; `time_seconds` is a max.** The two are
  not a rate.
- **Every progress record is a hybrid.** All workers report through one
  mutex-guarded wrapper (`PortfolioProgress`, `src/pool.cpp`). It rewrites
  `time_seconds` onto the **portfolio** clock and `objective` onto the
  **portfolio's** incumbent, with `new_best` set when that incumbent improved,
  so the stream is monotone in both and its last row matches the reported
  result. `iteration`, `total_violation`, `feasible` and `perturbations` stay
  the *reporting* worker's own, and consecutive rows come from different
  workers, so those four jump around and are not a rate. Worker 0 alone carries
  the periodic no-new-best tick that keeps a liveness row flowing.
- **One `SearchResult` field is dropped; the rest are pooled.** `best_violation`
  comes from the pooled solution, which carries the residual of the state it
  holds. `perturbations`, `lns_repairs`, `lns_repairs_accepted` and `counters`
  are **sums** over the workers, as `iterations` is, so not comparable to a
  single run's counts at the same wall time; `counters` goes through
  `SearchCounters::merge` at **both** aggregation sites (a worker's restarts in
  `WorkerAccumulator::absorb`, the workers in `solve_portfolio`), matching
  per-generator rows by name (`include/cbls/counters.h`).
  `first_feasible_objective` and `time_to_first_feasible` are the pair from the
  earliest worker *and restart* to reach feasibility, on the portfolio clock.
  Only `escape_probe_armed` reads its "not recorded" value: it latches one
  worker's end state, and a pooled solution carries no worker identity.
- **The model is read once, whatever `--threads` says.** `run_cli` loads it and
  hands it to `ParallelSearch::solve(Model&, ...)`, which freezes it and gives
  each worker a replica sharing one immutable structure (#157).
- **The run is not reproducible.** `--seed` still seeds every worker (through
  `portfolio_worker_seed`, which decorrelates adjacent base seeds), but which
  solution a stalled worker adopts depends on thread interleaving. `--threads 1`
  is the reproducible run, with [Determinism](#determinism)'s wall-clock caveat.

### Implied variable bounds

**Files:** `include/cbls/bound_propagation.h`, `src/bound_propagation.cpp`

CBLS variables need finite bounds, and a MIP column need not have any. A fixed
substitute magnitude is **not implied by the constraints** and can put the
optimum outside the box the search ever looks at, so `propagate_bounds` derives
bounds that *are* implied (#120), by standard activity-based tightening over
linear rows. For a row `lo <= Σ aⱼxⱼ <= hi`, the min/max activity of every term
but one bounds the remaining one:

```
aₖxₖ <= hi − minactivity(rest)      aₖxₖ >= lo − maxactivity(rest)
```

- **Infinite contributions are counted, not summed.** A row where exactly one
  term is unbounded still bounds that term; this is what finitizes a genuinely
  free column.
- **Derived bounds are relaxed outward** by the larger of `max(1e-9, 1e-12·|b|)`
  and a rounding bound scaled to the ACTIVITY the bound came from,
  `(nnz + 3)·DBL_EPSILON·(|row bound| + Σ|contributions|)/|coef|`, before being
  applied; integral columns round inward *after* that relaxation. A bound-scaled
  margin alone cut off a feasible point (#211; `tests/test_bound_propagation.cpp`
  carries the instance).
- **`1e20` and beyond is "no bound"** (the MPS/CPLEX/SCIP convention), read
  through `is_unbounded_below`/`is_unbounded_above`.
- **Fixed-point iteration is capped** (`max_passes`, default 10); each pass is
  O(nnz), and assembling the rows and propagating take about 1.1 s in the worst
  case on the MIPLIB roster (`square47`; timings on #120). Do not
  measure it by differencing two model builds: the difference is below the
  run-to-run noise.

A bound that **exists**, declared or derived, is honoured however wide; only a
missing one is invented, by `inf_clamp` (and the `.nl` side's `int_inf_clamp`),
and `MpsToModelResult::n_clamped_columns` reports how often. On the `.nl` side,
a missing bound whose other side is finite and at or beyond the clamp is placed
one clamp beyond that bound rather than at ±clamp, which would exclude it or
shrink the box to a point (#357). Propagation is on by
default in both adapters and can be turned off; that disables propagation only,
not the rule that the clamp supplies missing bounds rather than narrowing wide
ones.

`LinearRow` is a **view** into caller-owned arrays, so a large MIP's matrix is
never copied to hand it over; the MPS adapter builds one flat CSR that both the
expression builder and propagation read.

Deliberately **not** here: coefficient tightening, redundant-row removal,
aggregation, probing, dual reductions, and anything nonlinear. The `.nl` adapter
skips rows with a non-constant nonlinear part: omitting a row costs tightening,
never validity. Where propagation proves the linear system empty, the adapters
**discard** the derived bounds and build the model from the declared ones: an
infeasible instance should be reported by the search, not by the reader handing
it an empty box.

### Presolve

Removed (#379; in git history at `b10da9d2`). A load-time linear presolve with
postsolve (epic #322: fixed columns, singleton and redundant rows, coefficient
tightening #327, doubleton aggregation #328/#337; MPS adapter #325, `.nl`
adapter #326/#342) was opt-in behind `MpsToModelOptions`/`NlToModelOptions::presolve`
and `--presolve` in the MIPfeas and MINLPLib runners. #331 measured it at
`ef1bbad` and kept it off: the MIPfeas dev roster's pooled PI ratio was 0.934,
inside the 1.30 noise floor; on the MINLPLib dev sample it solved one instance
fewer at one seed (23 against 24 of 25), and its GM(gap+1) ratio of 1.025 was
inside the noise floor. Off, the model was node for node the plain build, so
removing it changes no result.

What outlived it: `cbls_mipfeas` still checks the point it writes against the
parsed `MpsProblem` (`benchmarks/mipfeas/original_check.h`: compensated row
sums, declared bounds, integrality, `c·x + offset` against the reported
objective); a point that fails is reported `original_mismatch` and carries no
objective (`original_size_mismatch`/`original_non_finite` say why when the
ratios cannot). `original_values(model)` on either adapter's result is the
model's point as one value per file column. `.nl` propagation reads the linear
rows (`linear_row` in `nl_to_model.cpp`: no nonlinear part or a lone constant
`k`, as AMPL writes `n0` for every linear row, read with the sides moved by
`-k`, #344).

### JSONL Model Format

**Files:** `include/cbls/io.h`, `src/io.cpp`

Models are serialized as `.cbls` files — one JSON object per line (JSONL),
describing a variable, a list partition, an expression node, a constraint, or an
objective. A List variable carries `n` (its universe) alone when it is a
permutation, and additionally `min_len`, `max_len` and `init` (`identity` /
`empty` / `random`) when it is not. A partition is
`{"partition": [<list names>], "cover": "exact" | "at_most_once"}`, written after
the variables and before the nodes so its members always resolve:

```jsonl
{"var":"x","type":"int","lb":0,"ub":10}
{"node":"s","op":"sum","children":["x","y"]}
{"constraint":"s_leq","op":"leq","children":["s","limit"]}
{"minimize":"x"}
```

**API:** `load_model(path|istream)` parses JSONL and returns a closed model;
`save_model(model, path|ostream)` serializes a closed model. Lambda nodes are
tabulated over their input domain on save and reconstructed on load, and a
PairLambda additionally carries its `mode` and its `head`/`tail` tables when
they are not the default (see the
[Lambda serialization note](#expression-dag)). The objective soft constraint is
*not* part of the serialized model — `freeze()`, which `solve()` calls, appends
it at runtime.

### NL Model Format (#280-#291)

`read_nl(path)` / `parse_nl(text)` (`include/cbls/io_nl.h`, `src/io/nl_reader.cpp`)
read AMPL `.nl` files, an original implementation of Gay's public spec: text
(`g` header) and binary (`b`, host byte order only). `nl_to_model`
(`src/io/nl_to_model.cpp`) builds a closed `Model`: one Int or Float per column,
each `V` defined variable inlined as one shared node, each constraint body as one
or two comparison rows from its `r` bound, the first objective with its sense.

- **Skipped:** `d` (initial dual guess) segments; the search does not use them.
- **Refused** with `std::runtime_error`: a malformed header (a block count
  outside `[0, n_vars]`, more discrete columns than variables, or block counts
  no valid variable ordering produces, #291); a discrete-column placement that
  does not match the declared count; out-of-range segment indices; a binary file
  in the other byte order; `S` suffixes (MINLPLib's are SOS declarations, and
  dropping them would solve a relaxation) and `F` imported functions.
- **Unsupported operators** do not throw: `nl_to_model` sets `supported = false`
  with a reason, and the model must not be solved.

`cbls_minlplib_nl_check` (`benchmarks/minlplib/nl_check.cpp`) models each file
named without solving and prints one status per file;
`benchmarks/instances/minlplib/status.py` runs it over the MINLPLib catalogue
(#283).

### SolveCallback

**File:** `include/cbls/search.h`

`SolveCallback::on_progress(const SolveProgress&)` fires on a ~1 s cadence and
immediately on any new best. Pass a `SolveCallback*` to `solve()` (the CLI
passes a formatter). Fields: [`SolveProgress`](#solveprogress).

### Formatters

**Files:** `include/cbls/formatter.h`, `src/formatter.cpp`

Two `SolveCallback` implementations, both writing to a configurable
`std::ostream` (default `std::cout`); the CLI selects one with `--format`, and
`--quiet` attaches neither:

- **`HumanFormatter`** — header (model path, var/constraint counts, objective
  sense, seed, time limit), progress lines (Time / Iter / Objective / Violation /
  Perturbs, `*` on a new best), and a final solution summary.
- **`JsonlFormatter`** — one JSON object per event (`start` / `progress` /
  `result`). `progress` carries iteration, time, objective (null when not
  feasible or not finite), violation, feasibility, perturbations and `new_best`;
  `result` carries time, iterations, `termination` (the `TerminationReason`
  token — `time_limit` / `iteration_limit` / `feasible` / `no_budget` /
  `stopped` / `cancelled`, which qualifies the two above), objective,
  `feasible`, `status` and `solution`.

### Parameter Locations

`solve()` takes `time_limit=10.0`, `seed=42`, `use_fj=true`, `hook=nullptr`,
`lns=nullptr`, `lns_interval=3`, `callback=nullptr`, `config={}` and
`coord=nullptr` (the portfolio's `SearchCoordination*`). Algorithmic tuning
lives in `SearchConfig` (outer loop) and `GFJConfig` (engine), inner-solver
knobs on `FloatIntensifyHook`, LNS's `destroy_fraction` on its constructor:
[Parameters Table](#parameters-table).

---

## Threading & Determinism

### Multi-threading

The core `solve()` is single-threaded; `ParallelSearch` is a cooperative
portfolio over it ([Parallel Search](#solution-pool--parallel-search)):
staggered seeds, a shared mutex-protected pool read and written mid-search, a
shared stop flag. No OpenMP, work-stealing or parallel DAG evaluation.

### Determinism

A **single-threaded** run is deterministic given the same seed, modulo
wall-clock. All randomness flows through a single `RNG` (`mt19937_64`);
`delta_evaluate` and AD keep their scratch in flat `thread_local` vectors and
recompute in topological order, so no container's iteration order reaches a
result. With `time_limit = 0` and
`SearchConfig::max_iterations` set, such a run reads no clock at all and is
bit-reproducible on any machine; with a wall clock it is reproducible for the
same seed on the same hardware under the same load, since `steady_clock`
determines where it stops.

**A portfolio run is not reproducible, by construction**: which pool solution a
stalled worker adopts depends on thread interleaving, and live sharing, pool
restarts and a global stop are all nondeterministic, so there is no
deterministic multi-threaded mode. Use `--threads 1` (or `cbls::solve()`) for
reproducibility.

### Caller-owned threads

`ParallelConfig::executor` is a `std::optional<ExecutorRef>`
(`include/cbls/executor.h`), unset by default. Unset, `ParallelSearch` creates one
`std::thread` per worker. Set, **it creates none**: worker `i` runs as index `i`
of `parallel_for_chunked(0, n_workers, ...)` on the caller's own pool, with
`n_workers = min(n_threads, executor->n_threads())`.

It lets a host run cbls on its own pool rather than oversubscribe its
cores. `ExecutorRef` is a type-erased, non-owning view of
anything with `parallel_for`, `parallel_for_chunked`, `parallel_invoke` and
`n_threads`; the caller owns and outlives the pool, as with `StopRef`. The
`cbls::Executor` concept in the same header compiles only in a C++20 (or later)
including translation unit.

- **The width cap is not cosmetic.** A worker holds its chunk for the whole
  shared deadline, so one queued behind another would search on no budget:
  asking for more workers than the pool is wide gets fewer workers, not slower
  ones.
- **The executor must run chunks concurrently**: workers share
  incumbents and restart from a peer's. A *sequential* executor is not
  rejected or detected: under a wall clock it degenerates to a one-worker
  portfolio (the first worker takes the whole deadline); with
  `time_limit <= 0` it runs each worker's full `max_iterations` budget in
  series, N times the wall time.
- **Chunk index is not worker index.** `src/pool.cpp` ignores `chunk_idx` (a
  chunk's number in `[0, n_threads())`), keys `results`, `failures` and
  `portfolio_worker_seed` on the loop index, and loops over
  `[chunk_begin, chunk_end)` for *any* chunking.

`launch_workers` (`src/pool.cpp`) is the one place the two paths differ; both
run one worker body, `run_one`, which swallows every exception a worker can
raise, since an executor may do anything with an escaping one, including abort.
An exception out of the *executor call itself* propagates: the caller reports
it. C++-only, deliberately: a Python executor would be called back under the
GIL, the serialisation a portfolio exists to avoid.

### External cancellation

`SearchConfig::stop` and `ParallelConfig::stop` are a `StopRef`
(`include/cbls/stop.h`): a type-erased, **non-owning** view of anything with
`bool requested() const`, such as the built-in `cbls::StopToken`. A host raises
it and the run ends at its next **batch** boundary with
`TerminationReason::Cancelled`. A C++20 `std::stop_token` needs the three-line
adaptor in that header, a **named lvalue** outliving the solve: the deleted
rvalue constructor refuses a temporary at compile time.

- **The granularity is the batch, not the iteration.** `requested()` is polled
  exactly where the wall clock is read (`past_deadline()`), so a running batch
  (`batch_iterations` GLS iterations: microseconds to seconds) finishes first.
- **`Cancelled` is not `Stopped`.** `Stopped` is a *peer worker* ending a
  portfolio run, which only `ParallelSearch` can cause; `Cancelled` is the host,
  and is reported when both hold, as the outer cause. `Feasible` outranks both.
- **The two `stop` fields are OR-ed, not alternatives.** `solve_portfolio`
  combines them into one `StopRef` on the calling thread's frame for the call;
  the objects both name must outlive the solve, and nothing in the engine owns
  or extends them.

**An attached stop is not a budget.** A run with no `time_limit` and no
`max_iterations` returns at once with `TerminationReason::NoBudget`, stop or no
stop, because "run until the host cancels" would hang forever the first time the
host forgot to; give it a generous `time_limit` and cancel inside it
(`StopRef::attached()` says the same). A portfolio cancelled before its first
worker ran reports `Cancelled`, not `NoBudget`.

From Python, `SearchConfig.stop` / `ParallelConfig.stop` accept a
`cbls.StopToken` (`request()`, `reset()`, `requested()`) or `None`, and read
back as a bool. The single-threaded `cbls.solve` releases the GIL for the
whole call, so a token raised from another Python thread reaches a running
solve (`tests/python/test_stop.py`, in a child process: a regression starves
the interpreter).

### Event tracing

`SearchConfig::tracer` is a `Tracer*` (`include/cbls/tracer.h`), null by
default: an unthrottled event stream on the search's own thread. It is **not**
`SolveCallback` (a throttled stream rewritten onto the portfolio's clock and
incumbent) and must not become it: N workers through one instance would need a
lock in the host's tracer or serialise the portfolio.

Under a portfolio each worker gets **its own** (`ParallelConfig::tracer_factory`,
called once per worker with its index, not once per restart). A factory, once
set, **decides**: a null return leaves that worker untraced, with *no* fallback
to `SearchConfig::tracer` (one shared, unsynchronised sink). A
`SearchConfig::tracer` with **no** factory is refused (`std::invalid_argument`,
on the calling thread, before any worker exists) whenever more than one worker
would run: dropping it loses events, keeping it is a race. One worker is
allowed through.

Five events, `batch_end`, `new_best`, `kick`, `lns` and `hook` (signatures and
iteration accounting in the header). **The granularity is the
batch, the kick or the inner-solver call — never the GLS iteration**, which
would put a virtual call on the engine's hottest loop. A consumer can rely on
`new_best` arriving before the `batch_end` of the batch that produced it, which
carries `improved = true`, and on a `KickKind::LNS` being followed by exactly
one `lns()` carrying the repair's outcome. `KickKind::Adopt`, the kick a pool
restart replaces the perturb with, is portfolio-only.

A null tracer changes nothing: `tests/test_tracer.cpp` compares objective,
feasibility, residual, iteration count, perturbation and LNS counters,
termination reason, escape-probe latch, every `SearchCounters` field
(per-generator rows included) and the **full final assignment** between a
traced and an untraced run of the same seed, except `inner_solver_seconds`, a
duration, which reads 0.0 in both arms of that clockless run.

---

## GPU

No GPU code: the bottleneck, `delta_evaluate` per jump candidate, is serial per
jump, and there is no batch-parallel evaluation of candidates.

---

## Reference

Davies, T. O., Didier, F., Perron, L. *ViolationLS: Constraint-Based Local
Search in CP-SAT.* CPAIOR 2024. (PDF in `docs/`.) Algorithm references
throughout this document (Algorithms 1–6) refer to this paper.
