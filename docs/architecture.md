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
2. [Expression DAG](#expression-dag)
3. [Model](#model)
4. [Violation & GLS Weights](#violation--gls-weights)
5. [Generalised Feasibility Jump](#generalised-feasibility-jump)
6. [Novelty Jump](#novelty-jump)
7. [Structural Batch](#structural-batch)
8. [ViolationLS Outer Loop](#violationls-outer-loop)
9. [Inner Solver (Continuous Intensification)](#inner-solver-continuous-intensification)
10. [Large Neighborhood Search](#large-neighborhood-search)
11. [Solution Pool & Parallel Search](#solution-pool--parallel-search)
12. [Design Decisions](#design-decisions)
13. [Control Flow Diagram](#control-flow-diagram)
14. [Parameters Table](#parameters-table)
15. [I/O & Logging](#io--logging)
16. [Threading & Determinism](#threading--determinism)
17. [GPU](#gpu)

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
5. **Diversifies** on stagnation: a random perturbation of the scalars plus a run
   of random structural moves per List/Set variable, which always moves at least
   one variable, or — every `lns_interval`-th diversification kick — large
   neighborhood search (destroy + GFJ repair).
6. **Tracks** the best real-feasible solution found (objective bound tightened
   alongside it), with a solution pool for parallel multi-seed search.

There is no temperature, no Metropolis acceptance, no global penalty multiplier
`lambda`. Feasibility pressure comes entirely from the per-constraint GLS
weights `W`, and progress is greedy on weighted violation. The penalty-method
metric `obj + total_violation()` survives only as the *inner solver's* local
descent objective (`ViolationManager::augmented_objective`), not as the search's
acceptance rule.

---

## Expression DAG

**Files:** `include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`

### Variable Types

```
enum class VarType : uint8_t { Bool, Int, Float, List, Set };
```

Each `Variable` stores: `id`, `type`, `value` (scalar), `lb`/`ub` (bounds),
`elements` (for List/Set), `universe_size`, `min_size`/`max_size` (Set
cardinality), and `dependent_ids` (nodes that use this variable).

Bool, Int and Float are *scalar* (jumpable by GFJ). List and Set are
*structural* — GFJ leaves them untouched; they are moved only by the
[structural batch](#structural-batch) and LNS.

> **Neither structural type has a positive result behind it.** List variables
> were once described here as validated on a published formulation; that claim
> is **withdrawn**. It rested on pharma-glsp, whose model was a macro-period
> relaxation of the source paper and which has since been retired (#28), so
> `List` now has no benchmark evidence in either direction. Set variables are
> validated for *expressiveness* only: a set-covering model over a `Set`
> variable produces verified solutions, but well short of the same instance
> encoded with one Bool per column — see
> [structure-only models](#structure-only-models-what-the-structural-batch-is-and-is-not)
> and `benchmarks/instances/setcover/README.md`. Treat "generalises to
> structured variables" as a claim about modelling reach only, for both types —
> beyond that `Set` is measured and negative, and `List` is unbenchmarked.

### Handle Encoding

Variables and expression nodes share a single `int32_t` handle space:

- **Variables**: negative handles, `handle = -(var_id + 1)`
- **Nodes**: non-negative handles, `handle = node_id`

`Model::wrap(handle)` decodes: if `handle < 0`, it refers to variable
`-(handle + 1)`; otherwise it refers to node `handle`.

### Expression Nodes

Each `ExprNode` has an `op` (operation), `children` (vector of `ChildRef`
with `id` + `is_var` flag), `parent_ids`, `value` (cached evaluation result),
and optionally `const_value` or `lambda_func_id`.

**Supported operations:**

| Category     | Operations                                           |
|------------- |------------------------------------------------------|
| Arithmetic   | Const, Neg, Sum, Prod, Div, Pow                      |
| Aggregation  | Min, Max, Abs                                        |
| Trigonometric| Sin, Cos, Tan                                        |
| Exponential  | Exp, Log, Sqrt                                       |
| Conditional  | If                                                   |
| Collection   | At (indexing), Count, Lambda (functional aggregation¹) |
| Comparison   | Leq, Eq, Geq, Neq, Lt, Gt                           |

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

### Evaluation Modes

**Full evaluation** (`full_evaluate`): evaluates all nodes in topological
order. Used at initialization, after perturbation/LNS, and after the
objective-bound soft constraint is appended.

**Delta evaluation** (`delta_evaluate`): given a set of changed variable IDs,
BFS-marks dirty nodes upward through `dependent_ids`/`parent_ids`, then
recomputes only dirty nodes in topological order. This is the hot path during
GFJ — each jump changes one variable and touches a small subgraph, and each
jump *candidate* is scored by a no-commit delta probe (see
[`weighted_violation_delta`](#violation--gls-weights)).

### Reverse-Mode Automatic Differentiation

`compute_partial(model, expr_id, var_id)` computes `d(expr)/d(var)` via
reverse-mode AD; `compute_all_partials(model, expr_id)` returns every variable's
partial in one reverse pass:

1. Initialize `adjoint[expr_id] = 1.0`
2. Traverse nodes in reverse topological order
3. For each node, propagate: `adjoint[child] += adjoint[node] * local_derivative(node, child_index)`
4. Variable adjoints use negative keys `-(var_id + 1)` to distinguish from
   node adjoints

`local_derivative` computes per-operation partial derivatives (chain rule
components). Discrete operations (At, Count, Lambda) return 0.

AD is used to generate Newton-toward-root jump candidates for Float variables
(in `compute_var_jump`) and by the inner solver.

¹ **Lambda serialization:** Lambda nodes store a C++ `std::function`, which
cannot be serialized directly. `save_model` tabulates the function over its
input domain and writes the resulting table. `load_model` reconstructs an
equivalent Lambda via table lookup, so round-tripping through JSONL is lossless
for finite-domain Lambda nodes.

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

**Expression creation**: arithmetic (`sum`, `prod`, `div_expr`, `pow_expr`,
`neg`, `abs`), trigonometric (`sin_`, `cos_`, `tan_`), other (`exp_`, `log_`,
`sqrt_`), conditional (`if_then_else`), collection (`at`, `count`,
`lambda_sum`), and comparisons (`leq`, `eq_expr`, `geq`, `neq`, `lt`, `gt`).
Each returns a non-negative node handle.

**Objective.** `minimize(e)` sets the objective node directly; `maximize(e)`
sets it to `neg(e)` and flips `is_maximizing_`. Internally the objective is
*always a quantity to minimize*, so the soft constraint `obj <= bound` is
uniformly correct for both senses.

### Finalization

`close()` computes the topological order, performs an initial full evaluation,
builds the `var_id -> constraint-index` adjacency (`build_var_constraints`, the
paper's `G_v`), and sets the `closed_` flag. The model is immutable in structure
after close — *except* for the objective soft constraint, which `solve()`
appends lazily (see below).

### Objective as a Soft Constraint

When the model has an objective, `solve()` calls `add_objective_soft_constraint()`
once (idempotent). This:

- creates a constant node `objective_bound_node_` (initially `+inf`),
- adds the constraint `obj - bound <= 0` (`objective_constraint_node_`), inert
  while the bound is `+inf`,
- records its index in `constraint_ids_` as `objective_constraint_idx_`,
- rebuilds the topo order and `G_v` (a node/constraint was appended after
  `close()`).

`set_objective_bound(bound)` updates the constant node and recomputes the
constraint residual in place. The search tightens the bound to `obj - eps` on
each new best feasible solution; the GFJ engine then treats meeting that bound
as just another constraint to satisfy. The bound is released back to `+inf`
before `solve()` returns so post-solve verifiers don't see it violated.

`has_objective_constraint()`, `objective_constraint_idx()` and
`objective_bound()` expose this state. "Real feasibility" everywhere means *all
constraints except* `objective_constraint_idx()`.

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
- `snapshot_violations(out)` / `weighted_delta_from(snapshot)` — the structural
  counterpart of the above, for moves on List/Set variables that the scalar-only
  probe cannot score. Snapshot the accepted assignment's per-constraint
  violations once, apply a candidate move, then read the weighted change against
  the snapshot. Like `weighted_violation_delta` it accumulates **per
  constraint**, which is what keeps a row clamped to `kInfPenalty` from
  swallowing the O(1) real rows (`1e30 + 3 == 1e30`, so differencing two whole
  sums loses them; `1e30 - 1e30` cancels exactly). See #100 and #118.
- `augmented_objective()` = `obj + total_violation()` — the penalty-method
  metric. **Used only as the inner solver's local descent objective**, not as
  any search acceptance rule. (When the objective is folded in as a soft
  constraint, the objective term is technically double-counted here; that is
  acceptable for the hook's local polish.)
- `is_feasible(tol=kDefaultFeasibilityTolerance)` / `violated_constraints(tol)` —
  convenience predicates. The default is `1e-6` (matching SCIP's
  `numerics/feastol` and `verify_model`); it is an *absolute* residual, so a much
  tighter value is not achievable on models whose constraint bodies are large in
  magnitude. `SearchConfig::feasibility_tolerance` shares the same default.
  over *all* constraints (including the objective soft constraint).
- `bump_weights(factor=1.0)` — increments the weight of each currently-violated
  constraint (a simple additive scheme; the GFJ engine uses the
  decay-then-bump `gls_update_weights` below instead).

### GLS Weight Dynamics

`gls_update_weights(vm, rho)` is the Guided Local Search update fired on GFJ
stagnation (paper Algorithm 3):

1. **Decay**: multiply every weight by `rho`.
2. **Bump**: add `1.0` to every currently-violated, *active* (weight > 0)
   constraint.

`rho` is sampled per batch from `{0.95, 1.0}` (decay or pure additive). Weights
masked to `0` (e.g. nonlinear constraints during the two-phase linear-first
pass) stay `0` under decay and are never bumped, so they remain inactive.

After a new best feasible solution, the search resets all weights to `1.0`
(`FeasibilityJump::reset_weights`) — a fresh penalty landscape per the paper.

> **Removed:** the old `AdaptiveLambda` global penalty multiplier (increase when
> stuck infeasible, decrease when feasible-not-improving) no longer exists. All
> feasibility pressure is now per-constraint GLS weighting.

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
| `V` | violated constraints | `violated_` bitset |
| `Q` | scan set of candidate vars | `queue_` / `in_queue_` |
| `J` | cached per-var best jump | `JumpTable` |

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
| Float | Newton step toward the root of each violated constraint containing `v` (`x - residual/grad`, gradient via reverse-mode AD; up to 4), then midpoint and endpoints. Once the search has stagnated, a Float at a *stationary* point of every violated constraint containing it additionally gets a two-sided local probe at `x ± {1e-6, 1e-2}·(|x|+1)` — see below |

Each candidate is scored with one `weighted_violation_delta` probe. Newton
candidates are considered first so that, on a tie in violation delta (a feasible
plateau), the gradient-informed point wins. Because the objective is a
constraint `obj <= bound`, when that constraint is violated its Newton candidate
pulls the objective *down* — this is how a hook-less continuous model still
descends the objective.

**The stagnation-gated escape probe (#107).** Every one of the candidates above
is either a Newton step, whose length is set by the target and which vanishes
with the gradient, or a constant derived from the box. So a Float sitting at a
point where every violated constraint containing it is stationary had *no
candidate that could move it at all* — an empty neighbourhood — and froze there
for the rest of the run. `Int` has no analogous *stationary-point* problem:
`int_jump_candidates` always offers `x ± 1`. Float was the one type with no local
move. `Int` did have its own freeze, from a different cause — `int_jump_candidates`
truncated the raw bounds with `std::lround`, in `long`, and glibc maps both
infinities to `LONG_MIN`. On `(-inf, +inf)` and `[lb, +inf)` that collapsed the
range and produced no candidates at all, `x ± 1` included. On `(-inf, ub]` it did
not collapse: with `ub >= 0` the width overflowed `long` and wrapped negative, so
the exhaustive arm ran up from `LONG_MIN` and wedged the solve, while with
`ub < 0` the width stayed a valid positive `long` and the grid arm returned
instantly with jumps near -9.2e18. Fixed by reading the bounds as doubles through
`domain_window` (**#114**); see the guard section below.

The probe supplies that local move, two-sided because at a saddle the descent
direction is precisely what a zero gradient cannot tell you. It is **off by
default and armed only once the search is stuck — after `perturbation_period`
batches without improvement, or, when the caller set a wall clock, after a
quarter of the budget with no new best (#117), whichever comes first** — then
disarmed on the next new best. That gating is load-bearing rather than a
throughput optimisation: "stationary and nothing improving" is the *steady
state* of local search, and an always-on probe was measured ~9x worse on
`shiporig` across every seed, because its drip of numerically tiny improvements
kept the search from ever registering stagnation, so diversification never
fired. The disarm is what separates this from that regime: every improvement
clears the flag, so an armed probe cannot drip indefinitely.

The wall-clock condition exists because `perturbation_period` counts *batches*,
and a batch is `batch_iterations` GLS iterations — microseconds on a small model,
seconds on an expensive one. Measured on MINLPLib `elec25` at a 60s budget, a
batch costs ~1.2s, so the run completed 52 batches against a threshold of 100 and
the probe was never armed at all: the gating above was dead code on exactly the
models it was meant to rescue, and 12 of the 50 MINLPLib instances complete fewer
than 100 batches at a 10s budget. *This condition* reads the clock **only** when
the caller set a wall-clock budget, so a run bounded by `max_iterations` alone
still lets no clock read influence the search trajectory and stays
bit-reproducible.

The clock route carries no diversification kick, unlike the stagnation route,
and an earlier revision of #117 gated it on `budget * batches / elapsed` falling
below `perturbation_period` for fear of starving diversification. **That gate was
removed.** The drip it defended against cannot run away: the improvement that
resets `stagnation` also disarms the probe, so re-arming costs another
`kEscapeArmFraction` of the budget and the clock route can arm at most
`1/kEscapeArmFraction` times in a run, each deferring at most one kick. #107's 9x
regression came from an always-on probe with *no* disarm. Measured on a
probe-sensitive model of Float double-wells, the gate cost objective quality
(-2.00 gated against -3.00 ungated at a threshold the batch route could not
reach) while preventing no starvation, and its predicate compared projected
*total* batches against a threshold the batch route only reaches after that many
*consecutive non-improving* ones. The tidier alternative — have the clock route
set `stagnation = perturbation_period` so the existing site arms and kicks
together — was considered and deferred: it changes the kick cadence, a tuned
parameter that #117 scopes out as wanting its own measurement. For the same
reason diversification itself stays on the batch counter.

Two properties to keep in mind when reading numbers off this. Trajectories become
*budget-dependent*: a 10s run arms at 2.5s and a 600s run at 150s, so anytime
traces are no longer comparable across budgets the way they were. And
`last_improvement` advances only on a real-feasibility improvement, so a run still
hunting for its first feasible point has by definition never improved — the probe
arms at 25% of budget and stays armed for the rest of the pre-feasible phase,
where it makes batches more expensive. #117's "52 batches at 60s" therefore
describes the *unpatched* engine and cannot be carried over. The measurements
quoted in that issue (10 of 12 batch-starved instances bit-identical, `eq6_1`
812.195 → 746.221, `maxmin` 0.11% worse, `shiporig` 2 of 3 seeds bit-identical)
describe this ungated arming condition, but were taken before the merge of #115
and #116 and have not been reproduced at this commit.

A single call is *not* a converged 1-D minimiser; the GLS loop iterates these
cheap jumps. The continuous heavy lifting is left to the
[inner solver](#inner-solver-continuous-intensification).

### The GLS Loop

`gls_loop(sample_size, batch_iter_limit)`:

```
loop:
    if apply_jump(sample_size) fails (no sampled var improves):
        if no active constraint is violated: return Feasible
        gls_update_weights(vm, rho)          # decay + bump violated
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
set `Q` (best-of-N: `sample_size_general = 3` general, `sample_size_linear = 5`
linear phase), refreshes any stale `JumpTable` entries via `compute_var_jump`,
removes non-improving vars from `Q` permanently, and commits the best improving
jump via `update_var`. `update_var` writes `X[v]`, delta-evaluates, refreshes
the violated set for `v`'s constraints, invalidates neighbour jumps, and
replenishes `Q` with vars now participating in active violated constraints.

### Two-Phase Linear-First (construction only)

`FeasibilityJump::run()` (standalone construction) optionally runs GLS on the
*linear submodel* first: nonlinear constraint weights are masked to `0`, GLS
satisfies the affine constraints, then all weights are restored to `1` for the
general phase. `compute_linear_constraints()` marks each constraint affine by a
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
  each List/Set var, full-evaluate, reset weights.
- `set_rho(rho)` — re-randomise the GLS decay between batches.

---

## Novelty Jump

**Files:** `include/cbls/feasibility_jump.h`, `src/feasibility_jump.cpp`
(`apply_novelty_jump`)

Novelty Jump (paper Algorithms 4–5) is a bounded-backtracking **compound-move**
search that escapes local optima single-variable FJ cannot — chained-invariant
fixes where no single jump improves weighted violation but a short *sequence*
does.

### Novelty Weights

On entry it builds `W' = ` novelty weights from the GLS weights `W`:

- `W'[c] = W[c]` for constraints **violated at entry**,
- `W'[c] = kCompoundDiscount * W[c]` (`kCompoundDiscount = 1/1024`, OR-Tools'
  epsilon) for constraints satisfied at entry.

Breaking a currently-satisfied constraint is therefore *cheap*, which lets the
search build chains that target the initially-broken constraints. When a move
breaks a satisfied constraint, that constraint is promoted to full weight and
its vars are enqueued.

### Bounded-Backtracking Search

`novelty_jump_search(s_m, budget)` is a recursive DFS over compound moves with an
explicit move stack (the paper's set `T`). `s_m` is the cumulative
*original-weight* score of moves on the stack; `s_c` tracks the best child score
explored at the current level. A candidate var is selected (`select_novelty_var`)
as the best of up to 3 sampled vars in `Q\T` passing the filter
`(s_m + W'-score > 0) OR (W-score > s_c)`, scored by `compute_var_jump` under
`W'`. The move is applied; if `s_m + W-score > 0` the compound move is committed
(left applied) and returns true; otherwise it recurses, and on failure backtracks
(reverting the move and consuming a `budget` discrepancy).

`apply_novelty_jump()` iterates `budget = 0, 1, 2` (iterated-deepening style),
committing improving compound moves and restarting from the new state; it stops
when feasibility is reached or `kNoveltyWorkBudget = 256` total applied moves are
exhausted. It commits its moves in place and returns whether it reached
feasibility; the caller must `resync()` afterward.

> **Status:** Novelty Jump is implemented, wired, and unit-tested, but **off by
> default** (`SearchConfig::use_compound_moves = false`). Its per-batch cost is
> not yet bounded tightly enough for the large continuous benchmarks. When
> enabled, `novelty_jump_probability` (default 0.5, matching the paper) sets the
> fraction of batches that are Novelty Jump.

---

## Structural Batch

**File:** `src/search.cpp` (`structural_pass`)

FJ jumps only scalar variables, so List/Set-structured models cannot improve
their structural assignment through FJ alone. The structural batch is the
List/Set peer of an FJ/NJ batch: it sweeps every List/Set variable, generates
the candidate structural moves for it, and greedily keeps any move that reduces
total weighted violation (negative weighted `deltaG` under the current GLS
weights `W`).

That `deltaG` is accumulated per constraint, via
`ViolationManager::snapshot_violations` / `weighted_delta_from`, and not by
subtracting two `total_violation()` readings. That removes **two** independent
defects, and only the first of them needs a clamped row.

**Clamped-row blindness (#118).** Once any row is clamped to `kInfPenalty`,
whole-sum differencing loses the real rows outright: 1e30 is fourteen orders of
magnitude above an O(1) row, so both readings round to the same double and every
move scores exactly 0. The objective soft constraint sits at that clamp for as
long as the #116 sentinel bound is installed, which used to freeze the structural
assignment of any model whose feasible region contains a non-finite objective.

**Phantom improvements from cache drift.** This one is older than #116 and needs
no clamped row at all. The old test was `after < before - 1e-12` against a
`before` threaded across candidate moves, and both readings come from
`total_violation()`'s incremental accumulator (`cached_total_ += (new - old) *
W`), whose 1000-call from-scratch recompute *bounds* the accumulated rounding
error but does not remove it. Two readings taken at different points in that
cycle can differ in the last ulp even when no constraint changed at all, and the
`- 1e-12` guard does not filter it. Above `2^14` it cannot: `x - 1e-12 == x` for
every double `x > 16384` (16384 itself is the last value the subtraction still
moves, because it lands in the binade below, whose ulp is 1.8e-12), and GLS
weights push setcover's weighted total to ~4.4e6, where one ulp is 9.3e-10.
Below `2^14` the guard is live but the drift simply outgrows it — observed
phantoms run 1.8e-12 to 1.2e-10.

Measured on `scp41` under the `Set` encoding (10s x 3 seeds, **no row clamped
anywhere in the run**): 99 of 39627 candidate moves were accepted by the old test
with a true weighted delta of exactly 0 — ~99 zero-delta moves committed per 30s
of search, each setting `changed` and forcing an `fj.resync()`. Unicost `scpe1`
shows the same at 12 of 166119. Per-constraint differencing scores those moves at
exactly 0 and rejects them. (A weighted delta of exactly 0 is not by itself proof
that *no row* moved — a `set_swap` shifting coverage +1/-1 across two rows of
equal GLS weight cancels exactly, and that is a genuine sideways move rather than
a no-op. The instrumentation counted the weighted delta only, so the no-op
reading is supported for the handful of disagreements whose changed-row count was
printed, not for all 99.) `tests/test_nonfinite_guard.cpp` pins the underlying
property deterministically, on a two-row fixture that reproduces the drift
without a search.

It is this second half, not the clamped-row half, that moved the `Set` numbers
below: setcover's objective is finite throughout, so no row is ever clamped
there.

Moves come from `generate_standard_moves` (`src/moves.cpp`):

| Type  | Moves |
|-------|-------|
| List  | `list_swap`, `list_2opt`, `list_relocate`, `list_or_opt_2`, `list_or_opt_3` |
| Set   | `set_add`, `set_remove`, `set_swap` |

(`generate_block_moves` provides sequence-aware block on/off moves for models
that register variable sequences; the scalar move generators `flip`,
`int_dec`/`int_inc`/`int_rand`, `float_perturb` also live here and are used by
LNS randomization paths.)

A batch is structural with probability `structural_batch_probability`: `< 0`
auto-selects `0.33` when the model has any List/Set variable and `0.0`
otherwise; scalar-only models always get `0.0`. After a structural batch commits
anything, the engine `resync()`s its scan set.

### Structure-only models: what the structural batch is and is not

The structural batch is a **first-improvement hill climber over a 3-5 move
random sample**, not a guided search. Per pass and per variable, `set_moves`
proposes exactly one random add, one random remove and one random swap; nothing
chooses *which* element on violation grounds, the way FJ's jump table and
best-of-N scan-set sampling choose a scalar's value.

That is invisible on a mixed model — where List/Set variables sit alongside
scalars that GFJ drives — but it is the whole search on a model whose
only variables are structured. There, everything else is inert:

| Mechanism | On a structure-only model |
|---|---|
| FJ batch | no jumpable variable: `apply_jump` fails every iteration and the batch degenerates into a pure GLS weight pump |
| Novelty Jump | compound moves are chains of scalar jumps — nothing to chain |
| `perturb` kick | reaches them since #111 (each List/Set variable gets its own pass of `clamp(round(p*|elements|), 1, |elements|)` random structural moves), but the moves are the same unguided ones — see [Diversification](#diversification) |
| LNS | destroys the structured variables wholesale, i.e. a random restart, then repairs with an FJ that has nothing to jump |

so progress is slow once the sampled neighbourhood stops improving — slow, not
finished: 6x the budget still buys ~18% on these instances (scp41 `set`, best of
seeds 42-44: 4739 at 10s, 3876 at 60s, both measured at the engine commit the
table below records), so the 10s numbers below are budget-limited rather than
neighbourhood-limited. Measured on OR-Library set covering
(`benchmarks/instances/setcover/`, issue #93): on the weighted instances the
same data modelled as one `Set` variable costs **8.6-11.0x the proven optimum**,
against **+9-20%** for one Bool per column — while on *unicost* instances,
where the objective is just cardinality, the two nearly converge. What the Set
search lacks is not reach but a violation-guided choice of *which* element to
move. A 3-row, 4-column fixture in `tests/test_setcover.cpp` reproduces it
exactly: the Set encoding stalls one move away from the optimum because every
single add/remove/swap from its incumbent is worse.

Note what the idle FJ batches cost. Their weight pump inflates `W` on the
persistently-violated objective row, and a skewed enough `W` is what lets the
structural pass accept a move that breaks a constraint — which is why the pump
was once thought to be load-bearing for a structure-only model. Setting
`structural_batch_probability = 1.0` removes the pump in exchange for more
structural passes, and at engine HEAD that trade measures *better* in both
regimes — unicost 7/7/7 -> 6/6/6, weighted 4917/4739/4902 -> 2593/2727/2916,
i.e. best-of-3 4739 -> 2593, a 45% improvement on scp41. The pump is not buying
anything that outweighs the passes it displaces.
Neither setting is a fix; both are symptoms of the structural batch having no
guidance of its own.

### Deadline bound

The sweep is bounded by the same wall-clock deadline as the rest of the loop,
checked **between variables** — never mid-variable, so a variable's move set is
always evaluated whole (the reference move set is never truncated for speed) and
the overrun is capped at one variable's work.

The bound is needed because the sweep's cost is unbounded in the model size:
`O(#structured vars x #moves x (delta_evaluate + O(#constraints)))`, since the
weighted delta rescans every constraint once per move. (The per-variable costs
below were measured at `95820e1`, where the pass hoisted `before =
vm.total_violation()` out of the move loop — one constraint scan per move plus
one per variable. The current form is one `weighted_delta_from` scan per move
plus a `snapshot_violations` per pass and per accepted move. Those numbers are
therefore close but not a strict upper bound: `total_violation()` is not O(1)
either — its "incremental" path diffs every constraint on every call — yet a
variable whose moves are all accepted costs one extra scan and one vector copy
apiece. Accepts are rare on a stalled structural search, and #118 makes them
rarer still by rejecting the phantom improvements it used to take. The genuinely
two-scans-per-move form — `invalidate_cache()` plus a full recompute on both
sides of each move — predates `8e17796` and none of these numbers were measured
against it.)
Both factors matter — a 1000-List x 800-element model costs 44.5us per variable
on its own, but **with ~40k constraints** the same model costs 407us per
variable. On a 1500-List x 100-element model with 40k constraints, a 0.5s budget
took **1.19–1.25s unbounded versus 0.502s bounded**. `solve(model, time_limit)`
is a library contract, and that was a violation for any user model of this shape
(issue #105).

Real benchmark models were nowhere near that scale. Measured on pharma-glsp
(the only List benchmark this repo ever had, retired in #28 — the numbers are
kept because they are the measurement that motivated the bound): it created one
List per macro-period, so its largest class (`glsp_e`, T=10) swept 10 structured
variables in **p50 499us, max 792us**, i.e. 0.03% of a 3s budget. The bound is
therefore about honouring the contract on large user models, not about the
benchmarks.

The check is **unconditional per variable**, not strided. An earlier self-tuning
stride was tried and deleted: because the stride persisted across passes while
its counter reset per pass, once it exceeded the model's structured-variable
count it could never fire again — it did nothing at all on 160 of the 170 real
pharma-glsp instances (2–6 List variables each; benchmark since retired,
#28). Cost of the plain check on real instances is ~0.75% of runtime at the
adversarial `structural_batch_probability
= 1.0`, and ~0.003% on the default 0.33 path.

> A per-variable `steady_clock::now()` costs ~1.4us only on an **HPET**
> clocksource (the machine these numbers came from); through the vDSO on a
> **TSC** clocksource it is ~20–25ns. On a synthetic model with 2000 tiny List
> variables the HPET cost does dominate — one sweep goes 1.54ms → 4.57ms — but
> that is a 60x-inflated constant that will not exist on most machines, which is
> why amortising it did not justify the complexity.

> **Removed:** the old SA "adaptive move probabilities" (per-move-type
> acceptance-rate tracking, 5% floor, rebalance every 1000 evaluations) and the
> SA-only Float moves `newton_tight` / `gradient_lift` no longer exist. The
> structural batch picks moves uniformly within each variable; Float steering is
> now handled by GFJ's gradient jump candidates and the inner solver.

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
| List, Set | `initialize_structured_random` | a random permutation / random subset |

**The scalar starting point does not depend on the seed, and this is deliberate.**
Feasibility Jump specifies the closest-to-zero start, and the priority-1
benchmark (`mipfeas`) is a head-to-head against another implementation of the
same algorithm, so the start point is part of what is being compared. Two seeds
on a scalar-only model therefore begin at the *same* assignment — if you are
reading identical results across seeds as evidence that the search converged,
check whether it simply started there.

The seed still drives everything else: List/Set initialisation, best-of-N
scan-set sampling, diversification kicks, the per-batch `rho` draw, LNS destroy
sets and Novelty Jump.

Do **not** over-read that as "a random start is reachable anyway". `perturb()`
randomises each jumpable variable only with probability `perturbation_probability`
(default **0.1**), not all of them, and moves each List/Set variable by only
`round(p * |elements|)` local structural moves (#111); it runs only after
`perturbation_period` (default **100**) stagnant batches, and when all of that
happens to move nothing it forces exactly one variable (#109). A kick is
therefore a sparse, occasional nudge, not a re-draw of the assignment, and it is
not a substitute for a randomised starting point.

`solve()` used to call `initialize_random` (which randomises scalars too) and
then overwrite every scalar a dozen lines later in `begin()`. The draws were dead
but still consumed. `initialize_random` remains as a public utility for callers
who *want* a randomised scalar start; compose it with `skip_init`:

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
worth probing (**#114**). Note nothing gates an unbounded Int *out* of an FJ
scan: `movable_domain` guards only the perturbation path, so a variable with no
jump candidates was still scanned, still found scoreless, and still left where it
was.

They used to hold three private copies of the same `switch (var.type)`, none of
them guarded against infinite bounds — so one default-probability kick on a model
with unbounded Floats turned the assignment NaN, `full_evaluate` propagated it,
`max_real_violation()` returned `+inf` permanently, and the run could not recover
while `solve()` still returned an ordinary-looking infeasible result restored
from the pre-kick closest-approach state (**#112**).

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
  (1e9) where a hand-built one gets `kRandomIntInfClamp` (1e6). Since #120 the adapters reach that
  fallback only where bound propagation (below) derives nothing, so a loaded
  model usually arrives here with finite bounds already. Unguarded, `uniform_real_distribution(lb, ub)`
  breaks its own precondition (`ub - lb <= DBL_MAX`) and libstdc++'s
  `lb + (ub - lb) * u` returns NaN on `(-inf, +inf)` and +inf on `[0, +inf)`,
  while an infinite Int bound casts to `INT64_MIN`;
- on a **half-infinite** domain the substituted end is pushed past the declared
  one where needed, so the window stays inside the domain even when the declared
  bound lies beyond the clamp magnitude;
- two finite bounds whose **width** overflows are narrowed to the clamp box.

Nothing else is rewritten, and in particular the window is **not** trimmed to the
`int64_t`-nameable range. Trimming it there clamped each Int bound independently,
which is neither inert (`[0, 1e17]` came back `[0, 2^53-1]`) nor a subset
(`[-1e18, -1e17]` came back as the single point `-2^53`, *above* the declared
`ub`, which `random_in_domain` then returned). `int_sample_window(var)` does that
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
`feasible = true` while carrying `±inf`. That path is independent of #112 and
predates it. The *Int* jump path was a second such route — it froze the variable
rather than infecting it — and is closed as of **#114**; Float remains open.

`begin()`'s closest-to-zero start is likewise well defined on every domain — a
genuine advantage of letting FJ own the scalars, but **only for the starting
point**; the kick and LNS destroy need the window above.

`use_fj` is now vestigial — GFJ is always the engine.

### Main Loop

While time and `max_iterations` remain, each pass:

1. **Pick the batch kind.** With probability `structural_batch_probability`,
   STRUCTURAL; else with probability `novelty_jump_probability` (only when
   `use_compound_moves`), NOVELTY JUMP; else FEASIBILITY JUMP. Structural and
   novelty batches mutate state outside FJ's bookkeeping, so they set a `resync`
   flag.
2. **Run the batch.** `fj.batch(batch_iterations)`, `fj.apply_novelty_jump()`,
   or `structural_pass(...)`. Every batch kind is bounded by the wall-clock
   deadline from *inside*, not just at the loop top: FJ strides a check through
   its GLS loop, and the structural sweep checks between variables (see
   [Deadline bound](#deadline-bound)), so a batch entered just before the
   deadline cannot run to completion past it. LNS repair is bounded separately,
   by `remaining()` at its call site.
3. **On real feasibility**, `record_best()` *first*, banking the feasible point
   before anything can move off it; then run the inner solver hook (if any) to
   polish continuous variables, and `record_best()` again only if the polish
   stayed feasible. Recording only after the hook silently discarded genuinely
   feasible solutions whenever the polish left the feasible region.
4. **`record_best()`** keeps the assignment if it strictly improves
   `best_feasible_obj`, snapshots the state, and tightens the objective bound to
   `obj - eps` (`eps = 1e-3*(|obj|+1)`; the step doubles as the Newton step size
   for hook-less continuous descent). For pure-feasibility models (no objective)
   the first feasible solution ends the search.
5. **On a new best**, reset GLS weights to 1, resample `rho`, and **disarm the
   Float escape probe** — the latch `SearchResult::escape_probe_armed` reports.
6. **Otherwise** increment `stagnation`, and `resync()` if a structural/novelty
   /hook mutation happened.
7. **Arm the Float escape probe** if the search is stuck: either
   `stagnation >= perturbation_period` (armed at the same site as the kick in
   step 8), or — with a wall clock — a quarter of the budget elapsed since the
   last new best (#117).
8. **On `stagnation >= perturbation_period`**, diversify (below) and reset the
   stagnation counter.
9. Emit a progress callback (~1 s cadence, or immediately on a new best).

At the end, restore the best state — or, on an infeasible run, the *closest
approach* to the feasible region rather than the untouched initial assignment —
release the objective bound to `+inf`, full-evaluate, and return the best
feasible objective (or `+inf` / infeasible). `SearchResult::best_violation`
carries the largest real-constraint residual at the returned assignment.
`SearchResult::iterations` is the total GLS iteration count, not the batch count.
`SearchResult::termination` says which budget ended the run — see below.

### Wall-clock budget

`solve(model, time_limit)` is a promise to return within `time_limit`. The
budget used to be checked only *between* batches, so any sub-step entered just
before the deadline could overrun it. Five can, and each is bounded separately:

| # | Sub-step | Bound | Where |
|---|----------|-------|-------|
| 1 | Feasibility Jump batch | handed the same absolute deadline, checked inside the GLS loop on a stride bounded two ways: at most 64 iterations, and at most 1/64 of the *remaining* budget in predicted time (#113) | `gfj.time_limit = budget_seconds` |
| 2 | `InnerSolverHook` | not *started* when the budget is spent — a hook is arbitrary user code, so its running time is unknowable | `if (hook && !past_deadline())` |
| 3 | LNS repair | handed `min(2.0, remaining())`, not its own independent 2s | `diversify()` |
| 4 | STRUCTURAL sweep | checked between variables (#105) | `structural_pass` |
| 5 | diversification kick, structural half | checked between structural *moves*, on a stride bounded the same two ways as row 1: at most 64 moves, and at most 1/64 of the *remaining* budget in predicted time (#115) | `perturb_structural` |

Bound 3 has two halves, and only the lower one is about the deadline. The
`remaining()` half is the deadline bound proper: a kick taken near the end of
the budget gets only what is left. The `2.0` cap is independent of the deadline
and predates it — it stops a single kick early in a long run from monopolising
it, and on any run whose whole budget is under 2s (including every test here) it
never binds. Only the `remaining()` half is covered by a test.

Bound 3 is floored at `1e-9` rather than clamped to 0 while a deadline exists:
`remaining()` returns exactly `0.0` once the clock crosses the deadline, and `0`
means "no wall clock at all" downstream in `fj_nl_initialize` — so clamping to
zero would hand the repair an *unbounded* run, the opposite of the intent.

##### Bound 1: why the stride is sized in time (#113)

The GLS loop cannot read the clock every iteration. `steady_clock::now()` costs
1408 ns/call on this project's reference machine, whose clocksource is **hpet**
(it is ~20–25 ns through the vDSO on a **tsc** clocksource, which is why the fix
must not assume either), against GLS iterations that can be a few microseconds.
Measured on a small Bool model, Release: **2991 → 4816 ns/iteration, a 1.75x
throughput loss**, for checking every iteration.

It used to check on a *fixed* stride of 64 iterations, which is not a time bound
at all: one GLS iteration is `O(sampled vars x candidate values x constraints
touched)`, so 64 of them are microseconds on a small model and seconds on a large
one — and the large ones are exactly the benchmark instances whose wall times
epic #87 publishes.

The stride is now sized in time. Each check measures the interval since the
previous one — which is the current cost of an iteration — and sizes the next
stride to `kStrideBudgetFraction` (1/64) of the *remaining* budget (remaining,
not total, so the bound tightens as the deadline approaches instead of permitting
`budget/64` right up to it). Growth is capped at `kStrideGrowth` (8x) per
adjustment so the ramp goes through progressively longer and more accurate
measurements; shrinking is uncapped, so an iteration that got more expensive is
caught at the very next check.

An uncapped shrink is **not** what stops the stride ratcheting upward and going
silent — the failure mode that got an earlier self-tuning stride removed from
this engine. It cannot be: a shrink is only *applied* at a check, and the next
check is a whole stride away, so a stride grown while iterations were cheap is
spent in full on the first expensive one. That is what `kMaxDeadlineStride` is
for, and it is set to **64** — the fixed stride this replaced — so the worst case
is no worse than the code being replaced while the time-based shrink still
delivers the case #113 filed. Uncapped, a model whose iteration cost rises
mid-run was measured at 9.9x over a 1 s budget (and 9.1x over 2 s, the absolute
overrun growing with the budget, because a larger budget buys a larger stride to
go silent with); capped, the same model runs 1.4x and 1.0x, against 2.4x and 1.2x
for the fixed stride.

**Guarantee: a batch returns at most one stride past the deadline, where a stride
is at most 64 GLS iterations and at most 1/64 of the remaining budget in
predicted time — or one GLS iteration, whichever is larger.** The last clause is
irreducible: an iteration is atomic and cannot be pre-empted from the inside. The
prediction is from the last measurement, so a cost that rises mid-stride is
absorbed by the 64-iteration half of the bound, not the time half.

Measured on 400 Int vars in 20 000 rows of 8 (Release, hpet, 12-core box under
concurrent load):

| budget | before | after | iterations before / after |
|--------|--------|-------|---------------------------|
| 0.05 s | 7.07 s | 0.22 s | 64 / 1 |
| 1.00 s | 7.11 s | 1.17 s | 64 / 6 |
| 3.00 s | 7.03 s | 3.10 s | 64 / 16 |

Before, the overrun was set by the model (always exactly 64 iterations, whatever
the budget); after, it is a handful of iterations. Note the relative overrun does
not shrink monotonically with the budget in general — that holds while iteration
cost is stable, which is the assumption the time half of the bound rests on.

Sizing the stride in time also bounds the clock overhead without measuring the
clock at all: one read per stride, against a stride costing `budget/64`, is
1.4 us / (budget/64) even on the expensive clocksource — 0.45% of a 20 ms budget,
0.009% of a 1 s one. It degrades only for budgets so small (well under a
millisecond) that the run is over before throughput matters. Measured on the
cheap-iteration model, Release, hpet: 2915–2994 ns/iteration adaptive against
2991–3064 fixed-64 and 2929–3008 with no wall clock at all — i.e. at the no-clock
floor, and no worse than the stride it replaces.

Two honest caveats. The stride is a *prediction* from a measurement, so an
iteration whose cost jumps mid-stride overruns the target and is corrected only
at the next check. And the interval between checks can span a batch boundary, so
work the outer loop does between batches (hook, LNS, structural sweep) is charged
to the stride and shrinks it — conservative, never the reverse.

##### Bound 5: why the kick is checked between moves (#115)

The kick's structural half used to check the clock only *between* structural
variables, and its comments claimed parity with the STRUCTURAL sweep's bound
(row 4). The parity did not hold, and the difference is the whole issue: the
sweep caps its overrun at one *move-set evaluation* per variable, which is linear
in that variable's size, while the kick's per-variable cost is
`k = round(p * |elements|)` move-set *generations*, quadratic in it. Both checked
between variables; only one had a per-variable cost small enough for that to
mean anything. A model whose structure lives in a single large List or Set had no
effective bound at all — it ran the entire quadratic pass and then consulted the
clock on its way to a variable that did not exist. Measured: one `perturb(0.1)`
took 2269 ms on a 41049-element Set and 1021 ms on a 30000-element List, and
`solve(model, 1.0)` returned in 1.29 s.

So the check moved *inside* the run, between moves. One move — one move-set
generation plus one apply, `O(|elements| + universe)` element copies — is
exactly the unit row 4 caps its overrun at, so the kick now has the bound its
comments used to claim.

A move is not cheap enough to check before every one of them. On the shape the
between-variables check already handled well, many small structures, a move on a
100-element List is ~1.5 us against 1408 ns for `steady_clock::now()` on the HPET
reference machine, so a per-move read would roughly double a kick. The check
therefore strides, and the stride is the one bound 1 already uses:
`FeasibilityJump::next_deadline_stride`, sized in time from the last measurement,
growth capped at 8x, shrink uncapped, hard-capped at `kMaxDeadlineStride`.

Sharing that tuner shares the lesson it encodes. A stride sized in time **alone**
goes silent exactly when it is needed, because a shrink can only be *applied* at
a check and the next check is a whole stride away — so a stride grown over many
cheap small structures would be spent in full on the first move of a large one.
The hard cap is what bounds that. It is also why the other direction #115 named,
bounding `k` against the remaining budget, was not taken: `k` is chosen once,
before the run starts, so a per-move cost that rises *inside* the run is never
re-observed at all. That is the same defect in a new place, not a fix for it.

**Guarantee: the structural pass applies at most 64 further moves after the
deadline passes, and a stride costs at most 1/64 of the remaining budget in
predicted time — or one move, whichever is larger.** The last clause is
irreducible for the same reason as bound 1's: a move is atomic and cannot be
pre-empted from the inside. The prediction is from the last measurement, so a
move whose cost jumps mid-stride is absorbed by the 64-move half, not the time
half.

The stride is armed at one move at the top of every kick rather than carried
across kicks, so the *first* variable is bounded too — the single large structure
is the case, and a stride inherited from a previous kick would spend itself
inside it. It is kept separate from the GLS tuner because the two loops advance
in different units and interleave: a kick runs between batches, so one shared
tuner would have each mis-size the other's stride.

Two deliberate exceptions to stopping. The check never fires before the pass has
applied a move, so a deadline already crossed on entry still leaves a kick
something to have done — the never-a-no-op contract of #109/#111, which `perturb`'s
fallback then completes if that one move happened to cancel itself out. And a
variable whose run was cut short still has its net effect recorded, so a
truncated kick is not mistaken for a no-op.

One gap the guarantee's unit hides, **pre-existing and deliberately left open**.
The bound is stated in moves, and the cost model behind it assumes cost is
proportional to moves — but an *attempted* move that finds no legal candidate is
not free. `generate_standard_moves` on a Set allocates a `vector<bool>` over the
universe, copies the membership and builds the complement — `O(|elements| +
universe)` — before discovering there is nothing legal, and the pass then leaves
that variable having applied nothing. Because the check short-circuits on "no
move applied yet" without decrementing, a model of M saturated Sets
(`min_size == max_size == universe_size`) does `O(M * U)` work with zero clock
reads. The old between-variables check was gated on `changed` and was equally
blind to it, so this is not a regression, and the cost is linear in the model
rather than quadratic in one variable — the shape #115 is about. Closing it means
bounding failed *attempts* as well as moves.

No clock read influences control flow when `time_limit <= 0` — `past_deadline()`,
`remaining()`, `structural_pass`, `perturb_structural` and FeasibilityJump's two
strided checks all short-circuit on their `has_deadline` flag — so with no
`SolveCallback` attached
the loop reads no clock at all, and iteration-budgeted runs stay bit-identical
and deterministic. (`solve()` still timestamps entry and exit to fill
`time_seconds`, and a callback's ~1s progress cadence reads the clock once per
batch; neither reaches the search trajectory.)

#### Why this is tested the way it is

A bug in this class made a 60s budget take 87s and was found only by reading a
benchmark's `wall_seconds` column by hand. Epic #87 publishes per-instance wall
times, so a silently overrunning budget corrupts published data — but the suite
was also deliberately converted to be deterministic (iteration-bounded,
`time_limit = 0`), and an `elapsed < X` assertion drags machine speed back in.

So bounds 1–3 are covered **without any assertion on elapsed time** (issue
#104). Each test observes its own bound directly: work done for 1 (one batch
cannot have run to completion), the call count of a test `InnerSolverHook` for
2, and the argument handed to a test `LNS` for 3. Bound 4 has no such seam —
its symptom really is duration — so its test is the suite's single
wall-clock-duration assertion, quarantined behind the `[timing]` tag and
registered as its own labelled ctest entry with an explicit timeout.

Every one of these tests also asserts `SearchResult::termination`. That is what
stops a test going quietly inert: a small time budget proves nothing if the
model converged before the clock ever mattered, which is exactly what had
happened to the old `fj_nl_initialize` time-limit test (its model converged in
~2ms against a 50ms cap, so it passed whether or not the limit was honoured).

The two tests that do bound work by counting — 1, and the reworked
`fj_nl_initialize` test — are not entirely free of machine speed either: they
need the machine to be unable to spend the whole iteration budget inside the
time budget. The trade-off is fixed at `margin = (time a regression takes to go
red) / (time budget)`, so buying margin costs time on the failing path. Both are
~20ms green and fail in the *safe* direction — a slower or loaded machine does
fewer iterations and passes more comfortably. They are different models, so
their measured numbers differ:

| Test | Iterations used | Margin | Time to go red |
|------|-----------------|--------|----------------|
| FJ batch bound | 128 of 40k | ~312x | ~10s |
| `fj_nl_initialize` time limit | 192 of 30k | ~156x | ~4.9s |

The stride sizing under bound 1 (#113) is tested the same way — nothing joins
`[timing]`. The sizing rule itself is a pure function of `(stride, elapsed,
target)`, `FeasibilityJump::next_deadline_stride`, so it is tested exhaustively
and deterministically: it grows only by the cap, shrinks without one, floors at
one iteration, floors again on a non-finite measurement, stops at
`kMaxDeadlineStride`, and — asserted over a grid of inputs — never predicts a
next stride costing more than the target.

That grid assumes iteration cost is *stationary*, which is exactly the assumption
that fails in the ratcheting case, so it is not the whole guarantee. Two further
tests cover what it misses, both timing-free: one ramps the stride through an
arbitrarily long cheap phase and asserts it never exceeds the cap, the other
walks the countdown arithmetic across a 1e6x cost jump and asserts the iterations
run past the deadline stay within the cap.

The two live tests then observe the tuner's own state through
`deadline_check_stride()` and `deadline_checks()` rather than any duration:

| Test | Observes | Margin |
|------|----------|--------|
| expensive iterations pin the stride to one | stride 1, one clock read per iteration, `< 64` iterations done | ~32x |
| cheap iterations let the stride grow | stride > 64, fewer reads than a fixed 64 would make | ~800x |
| a run with no wall clock reads no clock | `deadline_checks() == 0` | exact |

The first test's margin cannot be raised without lowering the work it does:
`(iterations that fit in the budget) x margin = 1 / kStrideBudgetFraction`, which
is 64 whatever budget the test picks. The last one is the determinism claim above
turned into an assertion — with `time_limit <= 0` the loop makes no clock read at
all, so nothing timing-derived can reach control flow.

All three go red on restoring a fixed stride of 64: verified by doing exactly
that (`deadline_check_stride() == 1` → 64, `> 64` → 64, and the end-to-end
`iterations < 64` → 64).

Bound 5 (#115) is observed the same way, in the unit its guarantee is stated in:
`structural_kick_moves()` counts the moves a kick applied and
`structural_kick_checks()` its clock reads, so no test asserts on a duration.

| Test | Observes | Fails if |
|------|----------|----------|
| one large List is bounded by the deadline, not by the List | exactly 1 move (unbounded: 10000) | the pass runs the variable out |
| one large Set is bounded by the deadline, not by the Set | exactly 1 move (unbounded: 8164) | as above |
| every kick re-arms the stride | 1 clock read per kick, on a 2-move kick | a kick inherits a grown stride |
| a deadline that expires mid-kick stops it within a stride | `moves <= 64 * checks + 1`, stopped short of 10000 | the pass runs a variable out past its budget, or never reloads the countdown |
| a kick with no wall clock reads no clock | `structural_kick_checks() == 0` | a clock read reaches control flow |
| many small structures are not cut short | all 400 moves run, 8 clock reads, `moves <= 64 * checks + 1` | the bound fires on structure count, reads per move, or the countdown is reloaded with a *multiple* of the stride |

Both large-structure models consume their structure through a constraint. An
unconsumed List or Set would let the engine leave it alone, and the test would
prove nothing while still passing.

Three of those deserve their reasoning recorded, because each covers a hole the
obvious version of the test leaves open.

The two large-structure cases assert **exactly one** move rather than "within the
bound of 65". The guarantee is one capped stride, but the observed value is
deterministic — the stride is re-armed to 1 at the top of every kick, so the
first check lands after a single move. Accepting 65 would let the re-arm be
deleted in silence, and a kick would then inherit a grown stride and spend 64
moves inside the first large variable (~35 ms rather than ~0.55 ms on #115's 41k
Set). That is not hypothetical: the same stride-persistence bug already shipped
once in `structural_pass`, where the stride outgrew the model's structured
variable count and the check went inert on 160 of 170 pharma-glsp instances
(the benchmark is gone in #28; the bug it exposed is not).

The re-arm has its own test because the two above only *depend* on it. On a
one-List model with `k = 2`, a re-armed kick reads the clock exactly once (move 1
is ungated; the check before move 2 finds countdown 1). A kick inheriting the
previous stride of 8 never exhausts its countdown in 2 moves and reads it zero
times. The discriminator is the countdown arithmetic, not a duration.

The mid-kick expiry test exists because the expired-deadline tests never reach
the guarantee's real scenario: with the deadline already gone, the first check
returns before `next_deadline_stride` is called at all, so the countdown reload
is dead code in them. Giving the pass a budget it outlives for a while exercises
the ramp, and `moves <= 64 * checks + 1` is the guarantee in the only form
observable without a clock.

That assertion is carried in *both* the mid-kick test and the small-structure
one, and only the second can fail on a countdown reloaded with a **multiple** of
the stride — an edit that multiplies the real overrun while leaving the reported
stride untouched. In the mid-kick test the time target is what binds, and the
tuner absorbs the mutation: measuring the interval it actually got, it sees `4x`
the elapsed time over `4 * stride` moves and shrinks the stride `4x` to match,
landing on the same moves-per-check. Only where the 64-move cap binds instead —
cheap moves, a distant deadline, which is the small-structure model — does the
mutation surface, as 256 moves between reads against a cap of 64.

The tuner's own ratcheting properties are not retested: bound 5 reuses
`next_deadline_stride` unchanged, so the exhaustive grid and the two timing-free
ratchet tests above already cover them. What is new in #115 is *where* the check
is placed and how the countdown is driven, which is what these observe.

One cost worth naming: `arm_structural_kick()` runs on every kick, including on
scalar-only models, so a deadline-armed run there now pays one
`steady_clock::now()` per kick where it paid none before. It is one read per
`perturbation_period` stagnant batches, and it is determinism-safe (the arm is
gated on `has_deadline_`), but it is not zero.

### Diversification

`diversify()` increments a `perturbations` counter and, every `lns_interval`-th
kick (when an `LNS` is supplied), runs [LNS](#large-neighborhood-search)
destroy-repair (then resets FJ weights, since LNS mutates state outside the
engine). Otherwise it calls `fj.perturb(perturbation_probability)`. Either way
it resamples `rho`.

`perturb(p)` randomises each jumpable variable independently with probability
`p`, and then — **only if that moved nothing** — forces one uniformly chosen
variable to a *different* value. The fallback is what makes the kick meaningful
on a small model: independent draws alone leave the assignment untouched with
probability `(1-p)^n`, which at the default `p = 0.1` is 81% on two variables —
so diversification would usually do nothing, burn the stagnation counter, and
let the search resume in the same basin.

Making it a fallback rather than a variable forced on every kick is deliberate.
On a model large enough for the per-variable probability to do its job, a no-op
kick is vanishingly rare, so the fallback never runs: no extra RNG draw, and the
kick keeps exactly the distribution *and the exact draw sequence* it had before.
Forcing a variable unconditionally would instead shift the draw sequence on every
model.

Resampling is not enough for the forced variable — a uniform redraw returns the
current value with probability `1/|domain|`, which on a Bool is one kick in two —
so the fallback draws from the domain with the current value removed. A variable
pinned by `lb == ub` cannot be chosen.

Both of those reach only *jumpable* (scalar) variables, so List and Set variables
get their own pass in the same kick (#111). Without it, a kick on a model whose
decision structure is structural randomised nothing at all, and since LNS fires
only every `lns_interval`-th kick, most kicks on such a model were no-ops — the
pharma-glsp campaign-scheduling formulation being the case in the roster at the
time. That benchmark has been retired (#28), but the guard is still
load-bearing: `setcover`'s `Set` encoding (`build_set_model`) allocates one
`Set` variable and no scalar at all, so every kick on it depends on this pass.
Its effect there is measured, not assumed — see
`benchmarks/instances/setcover/README.md`.

The structural pass applies `k = max(1, round(p * |elements|))` random moves to
**each** List/Set variable, drawn from the same typed generators the [structural
batch](#structural-batch) uses (`generate_standard_moves`), so every move is
legal by construction: a List stays a permutation of its elements, a Set stays
inside `min_size`/`max_size`. Candidates that happen to be no-ops (a relocate to
the adjacent position) are filtered out before the draw, and a variable counts as
moved only if its elements *net* changed — a run that adds an element and removes
it again has to fall through to the guarantee below like any other no-op.

Scaling `k` rather than sampling *which* structures move is the deliberate part.
A structure has no "randomise the whole variable" analogue that is not a restart,
so the probability sets how much of each structure moves instead of how many
structures move. It also stops a structure with fewer than `1/p` slots from never
moving.

`k` counts *moves*, not displaced slots, and for a List the two are far apart:
`list_2opt` reverses a random sub-range (mean ~n/3), so `k = 0.1n` rewrites ~98%
of positions on a 1000-element list while breaking ~26% of adjacent pairs. The
adjacency figure is the one that tracks `p`, so the scaling suits a List the DAG
reads pairwise (`pair_lambda_sum`) and is much coarser than `p` suggests for one
read positionally (`at`).
The floor of one move is the price: every structure moves on every kick, `p = 0`
included, where the scalar half moves exactly one variable.
`k` is clamped to at least one move and at most one per slot, already a full
scramble, so a misconfigured `p > 1` cannot turn a kick into unbounded work.
The size is the structure's *current membership*, which for a List is its whole
decision content but for a Set is not: a 3-of-1000 Set is kicked on the 3, so it
rewrites a `p` fraction of the set's state and cannot grow far in one kick — LNS,
which resamples the cardinality outright, is the mechanism for that.
The moves are local in the *adjacency* metric — a 2-opt reversal rewrites a whole
sub-range of positions but breaks only two adjacent pairs — so that is the metric
in which "not a restart" means anything. Measured on a 200-element List: the
default `p = 0.1` keeps 74% of its adjacent pairs, `p = 0.02` keeps 94%, and
`p = 0.5` keeps 23%. Cost follows the same scale, `O(k * (|elements| + universe))`
element copies per structure (the generator builds each candidate as a whole new
vector, and a Set's candidates scan its universe): 0.012 ms per kick on one
100-element List, 23 ms on the 1500-List model of #105 — paid once per
`perturbation_period` stagnant batches. Because that is superlinear in one
structure's size, checking the deadline between *variables* bounded nothing on a
model whose structure is one large List or Set: 2269 ms for a single kick on a
41049-element Set. The pass is therefore deadline-bounded between *moves*, on a
capped stride, and never before it has moved something — see
[Bound 5](#bound-5-why-the-kick-is-checked-between-moves-115) for the guarantee
and why a purely time-derived bound was rejected.

The pass tests the variable type before drawing, so it consumes no randomness on
a model without List/Set variables: scalar-only models keep their exact draw
sequence, verified bit-identical against the pre-#111 engine at the time of the
change over 12k kicks (6 model shapes x 4 probabilities x 25 seeds x 20 kicks,
comparing the assignment and the next two draws after each kick), and pinned in
the suite by a test that runs the same scalar model with and without an immovable
List and requires identical assignment sequences.

The kick's guarantee is model-wide: if anything can move, something moves. On a
mixed model that means the structures absorb it — they always move — so the
scalar half is left to the configured probability rather than being forced, and
the forced-scalar fallback only fires when nothing moved at all. A model with no
movable variable anywhere (every scalar pinned, every structure a dead end) still
correctly changes nothing.

### SearchConfig

```cpp
struct SearchConfig {
    bool skip_init = false;                 // keep the assignment whole: no List/Set
                                            // randomisation, no FJ scalar start
    int64_t max_iterations = 0;             // 0 = unlimited (use time_limit); counts GLS iterations
    bool use_fj = true;                     // vestigial: GFJ is always the engine
    int lns_interval = 3;                   // LNS fires every Nth diversification kick

    int64_t batch_iterations = 1000;        // GLS iterations per FJ batch
    int perturbation_period = 100;          // batches without improvement before diversifying
    double perturbation_probability = 0.1;  // scalar randomisation prob + List/Set
                                            // kick size (never a no-op)
    double structural_batch_probability = -1.0;  // <0 = auto (0.33 if List/Set vars, else 0)
    bool use_compound_moves = false;        // run Novelty Jump batches (else FJ only)
    double novelty_jump_probability = 0.5;  // P(a batch is Novelty Jump) when enabled
};
```

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
has banked the point. Because the hook may move the assignment off the feasible
region, the loop re-checks `real_feasible()` before recording the polished
assignment; the pre-polish point is safe either way. (This replaces the old SA
trigger of "every 10 discrete accepts / on every reheat".)

Note the bound tightening in `record_best()` now happens *before* the hook runs,
so the hook descends against an already-tightened `obj <= bound`.

It descends `ViolationManager::augmented_objective()` = `obj + total_violation()`
and mutates Float variables directly via `delta_evaluate`.

### Mechanism 1 — Single-Variable Newton Steps

For each Float variable, examine up to 3 violated constraints. For each,
`g = constraint_value`, `dg = d(constraint)/d(var)` (reverse-mode AD); if
`|dg| > 1e-12`, propose `clamp(var - g/dg, [lb, ub])` and keep it if
`augmented_objective` improves.

### Mechanism 2 — Backtracking Line Search on Objective

For each Float variable, `df = d(objective)/d(var)`; if `|df| > 1e-12`, try
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
improvement.

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
diversification kick (a kick happens after `perturbation_period` stagnant
batches). When no `LNS` object is supplied, diversification is plain
perturbation instead.

### Destroy Phase

`destroy_repair` snapshots the state and its lexicographic key, then selects
variables to destroy:

- **Sequence-aware** (when the model registers `var_sequences`): pick
  `ceil(n_seqs * destroy_fraction)` whole sequences plus a proportional fraction
  of non-sequence vars.
- **Uniform** (no sequences): `max(1, floor(num_vars * destroy_fraction))`
  random vars (default `destroy_fraction = 0.3`).

Destroyed variables are re-randomized by type through the shared `randomize_var`
(see "Randomising a variable" under Initialization): Bool/Int/Float uniform over
the guarded domain window, List a reshuffle of its current elements
(`ListOrder::Perturb` — destroy/repair perturbs an incumbent, so the elements
present are preserved and only their order changes), Set a random
valid-cardinality subset.

### Repair Phase

`full_evaluate`, then `fj_nl_initialize(model, vm, 2000, &rng)` — a single-phase
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

A bounded, sorted collection of solutions for tracking best results across
parallel searches.

**Sort order:** feasible solutions first; among same feasibility, ascending
objective. **Capacity:** default 10, excess trimmed after each insert.
**Restart selection:** `get_restart_point()` samples uniformly from the better
half of the pool.

### Parallel Search

`ParallelSearch::solve()` dispatches by `ParallelConfig::deterministic`:

**Opportunistic / portfolio mode** (default): launch N threads (default
`hardware_concurrency()`), each building its own `Model` via a factory and
calling `solve()` with a staggered seed (`seed + thread_index`). Only the
`SolutionPool` is shared (mutex-protected); thread safety is by isolation. The
best solution across threads is returned, prioritizing feasibility then
objective. Worker exceptions are caught (letting one escape a thread function
is `std::terminate`), but not discarded: if *every* worker throws, `solve()`
rethrows the lowest-indexed one rather than returning a default,
infeasible-looking result.
A partial failure is absorbed silently.

**Deterministic epoch-sync mode** (`deterministic = true`): the per-thread
models are built on the calling thread, so a failing model factory propagates
straight out of `solve()` -- it always has; only the epoch worker threads are
left unwrapped, so an exception raised inside one terminates the process.
Threads run
synchronized epochs of fixed GLS-iteration count (no wall-clock dependency).
Each epoch sets `SearchConfig::max_iterations = epoch_iterations`; after the
first epoch `skip_init = true` and FJ initialization is off. Per-epoch results
feed an elite `SolutionPool`; threads restart from elite states next epoch.
Thread seeds are `base_seed + epoch * n_threads + thread_id`. Repeats for
`max_epochs`.

`ParallelSearch::solve()` takes hook and LNS *factories* (these objects are
stateful and per-model); each thread builds its own instances. Both are C++-only
for now: a Python factory returns an object nanobind owns, which the adopt into
a `unique_ptr` here would free a second time, so the binding refuses it with a
`TypeError` pending #129.

---

## Design Decisions

### Why ViolationLS instead of Simulated Annealing

The engine was ported from a simulated-annealing core (Metropolis acceptance,
geometric cooling + reheat, adaptive `lambda`, adaptive move probabilities) to
ViolationLS. ViolationLS replaces all of SA's tuning knobs (temperature,
cooling rate, reheat interval, penalty multiplier) with a single, self-adapting
mechanism: per-constraint GLS weights driving a greedy best-of-N jump. In the
paper it is competitive with CP-SAT's other workers on CBLS-amenable problems;
here it gives a parameter-light core that adapts feasibility pressure
per-constraint rather than globally, and folds optimization into a sequence of
feasibility problems (objective-as-constraint) so the same engine handles both.

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
Currently off by default pending tighter per-batch cost bounds.

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
restart schedules (Luby). The solution pool supports multi-seed parallel search
but is not yet used for warm restarts within a single thread.

---

## Control Flow Diagram

```
solve(model, time_limit, seed, use_fj, hook, lns, lns_interval, callback, config)
│
├── [if objective] add_objective_soft_constraint(); set_objective_bound(+inf)
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
    │     NOVELTY:    fj.apply_novelty_jump()         # compound moves; resync
    │     STRUCTURAL: structural_pass()               # list/set moves; resync
    │
    ├── if max_real_violation() <= config.feasibility_tolerance:
    │     ├── record_best()                           # bank it BEFORE polishing
    │     │     └── tighten objective bound to obj - eps; snapshot best state
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
    │     arm the Float escape probe; diversify():
    │       every lns_interval-th kick → lns->destroy_repair(); fj.reset_weights()
    │       else                       → fj.perturb(perturbation_probability)
    │       resample rho; ++perturbations; stagnation=0
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
| `skip_init` | false | `SearchConfig` | keep the current assignment whole — suppresses both List/Set randomisation and FJ's scalar start (epoch restarts, caller-supplied starts) |
| `batch_iterations` | 1000 | `SearchConfig` | GLS iterations per FJ batch |
| `perturbation_period` | 100 | `SearchConfig` | stagnant batches before a diversification kick |
| `perturbation_probability` | 0.1 | `SearchConfig` | per-var scalar randomisation probability on perturb; also scales the List/Set moves per kick (a no-op kick moves one var anyway) |
| `structural_batch_probability` | -1 (auto) | `SearchConfig` | P(structural batch); auto 0.33 w/ List/Set, else 0 |
| `use_compound_moves` | false | `SearchConfig` | enable Novelty Jump batches |
| `novelty_jump_probability` | 0.5 | `SearchConfig` | P(Novelty Jump batch) when enabled |
| `lns_interval` | 3 | `SearchConfig` / arg | LNS fires every Nth diversification kick |
| `rho` (GLS decay) | {0.95, 1.0} | sampled per batch | GLS weight decay factor |
| `sample_size_general` | 3 | `GFJConfig` | best-of-N scan-set sample, general phase |
| `sample_size_linear` | 5 | `GFJConfig` | best-of-N scan-set sample, linear phase |
| `two_phase` | true (solver) / false (outer loop, warm-start) | `GFJConfig` | linear-first GLS pass |
| `set_initial_x` | true | `GFJConfig` | set X to domain value nearest 0 first |
| `kCompoundDiscount` | 1/1024 | `feasibility_jump.cpp` | Novelty-weight discount on satisfied constraints |
| `kNoveltyWorkBudget` | 256 | `feasibility_jump.cpp` | max moves per `apply_novelty_jump` |
| `kEscapeArmFraction` | 0.25 | `search.cpp` `solve` | fraction of the wall-clock budget without a new best that arms the Float escape probe (#117). Not a `SearchConfig` knob |
| `objective_bound_eps` | 1e-3·(|obj|+1) | `search.cpp` `record_best` | bound tightening / hook Newton step |
| `destroy_fraction` | 0.3 | `LNS` ctor | fraction of variables/sequences destroyed |
| `repair_time_limit` | 2.0 s | `LNS::destroy_repair` | FJ repair budget; `<= 0` = iteration-bounded only |
| `feasibility_tolerance` | 1e-6 | `SearchConfig` / `kDefaultFeasibilityTolerance` | violation below which a constraint counts as satisfied |
| `lns_repair_iters` | 2000 | `lns.cpp` | GFJ iterations in LNS repair |
| `max_sweeps` | 3 | `inner_solver.h` | inner-solver coordinate-descent sweeps |
| `initial_step_size` | 0.1 | `inner_solver.h` | line-search starting step |
| `max_line_search_steps` | 5 | `inner_solver.h` | max backtracking halvings |
| `max_multi_var_constraints` | 5 | `inner_solver.h` | max constraints for multi-var Newton |
| `pool_capacity` | 10 | `pool.h` | max solutions in pool |
| `n_threads` | 1 (CLI) / hw_concurrency (lib) | CLI / `ParallelConfig` | parallel search threads |
| `epoch_iterations` | 5000 | `ParallelConfig` | GLS iterations per epoch (deterministic) |
| `max_epochs` | 10 | `ParallelConfig` | epochs (deterministic mode) |

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
| `--threads N` | number of threads (0 = auto-detect, default: 1) |
| `--deterministic` | enable deterministic epoch-sync parallel mode |
| `--epoch-iters INT` | GLS iterations per epoch in deterministic mode (default: 5000) |
| `--max-epochs INT` | number of epochs in deterministic mode (default: 10) |
| `--format human\|jsonl` | output format (default: human) |
| `--quiet` | suppress progress, print only the final result |
| `--help` / `--version` | usage / `cbls::kVersion` |

> **Removed flags** (SA-era): `--cooling-rate`, `--reheat-interval`,
> `--hook-frequency`, `--fj-time-fraction`. These no longer exist; the
> corresponding mechanisms were deleted in the ViolationLS port.

### Implied variable bounds

**Files:** `include/cbls/bound_propagation.h`, `src/bound_propagation.cpp`

CBLS variables need finite bounds, and a MIP column need not have any. The
adapters used to substitute a fixed magnitude (`inf_clamp`), which is **not
implied by the constraints** and can therefore put the optimum outside the box
the search ever looks at.

`propagate_bounds` derives bounds that *are* implied, by standard activity-based
tightening over linear rows. For a row `lo <= Σ aⱼxⱼ <= hi`, the min/max activity
of every term but one bounds the remaining one:

```
aₖxₖ <= hi − minactivity(rest)      aₖxₖ >= lo − maxactivity(rest)
```

Four details carry the implementation:

- **Infinite contributions are counted, not summed.** A row where exactly one
  term is unbounded still bounds that term — the rest of the row is finite. This
  is the case that matters: it is what finitizes a genuinely free column.
- **Derived bounds are relaxed outward** by `max(1e-9, 1e-12·|b|)` before being
  applied, to absorb rounding in the activity sums. Integral columns round
  inward *after* that relaxation. The margin is scaled to the derived bound, not
  to the activity it came from, so it is a practical guard rather than a proof:
  a row summing ~1e5 terms of magnitude ~1e9 accumulates more error than it
  absorbs. No instance on the MIPLIB roster has reached that regime.
- **`1e20` and beyond is "no bound"** (the MPS/CPLEX/SCIP convention), read
  through `is_unbounded_below`/`is_unbounded_above`.
- **Fixed-point iteration is capped** (`max_passes`, default 10) and each pass
  is O(nnz). Timed in isolation on the MIPLIB roster: 0.13s to assemble the rows
  plus 0.13s to propagate on the largest instance by rows (710k columns, 961k
  rows, 4.9M nonzeros, 3 passes), and ~1.1s worst case on the densest
  (`square47`, 27.4M nonzeros). Do not measure this by differencing two model
  builds — the difference is smaller than the run-to-run noise.

The rule the adapters follow afterwards: a bound that **exists**, declared or
derived, is honoured however wide; only a missing one is invented. `inf_clamp`
(and the `.nl` side's `int_inf_clamp`) is the fallback, and
`MpsToModelResult::n_clamped_columns` reports how often it was still needed.
Propagation is on by default in both adapters and can be turned off. That
disables propagation only: the other half of #120 — the clamp supplying a bound
where none exists, rather than narrowing every bound wider than it — is
unconditional, so the switch does not restore the pre-#120 engine.

`LinearRow` is a **view**: `cols`/`coefs` point into arrays the caller owns.
The constraint matrix of a large MIP is hundreds of megabytes — `square47` alone
is 27.4M nonzeros — so the rows are never copied to hand them over. The MPS
adapter builds one flat CSR that both the expression builder and propagation
read, rather than grouping the matrix once and copying it again.

Deliberately **not** here: coefficient tightening, redundant-row removal,
aggregation, probing, dual reductions, and anything nonlinear. The `.nl` adapter
simply skips rows with a nonlinear part — omitting a row costs tightening, never
validity.

Where propagation proves the linear system empty, the adapters **discard** the
derived bounds and build the model from the declared ones: an infeasible
instance should be reported by the search, not by the reader silently handing it
an empty box.

### JSONL Model Format

**Files:** `include/cbls/io.h`, `src/io.cpp`

Models are serialized as `.cbls` files — one JSON object per line (JSONL),
describing a variable, expression node, constraint, or objective:

```jsonl
{"var":"x","type":"int","lb":0,"ub":10}
{"node":"s","op":"sum","children":["x","y"]}
{"constraint":"s_leq","op":"leq","children":["s","limit"]}
{"minimize":"x"}
```

**API:** `load_model(path|istream)` parses JSONL and returns a closed model;
`save_model(model, path|ostream)` serializes a closed model. Lambda nodes are
tabulated over their input domain on save and reconstructed on load (see the
[Lambda serialization note](#expression-dag)). The objective soft constraint is
*not* part of the serialized model — `solve()` appends it at runtime.

Models can also be built programmatically via the C++ or Python API.

### SolveCallback

**File:** `include/cbls/search.h`

`SolveCallback::on_progress(const SolveProgress&)` receives progress updates. It
fires on a ~1 s cadence and immediately on any new best. Pass a `SolveCallback*`
to `solve()` (the CLI passes a formatter). See
[`SolveProgress`](#solveprogress) for the fields.

### Formatters

**Files:** `include/cbls/formatter.h`, `src/formatter.cpp`

Two `SolveCallback` implementations format solver output:

- **`HumanFormatter`** — tabular output: header (model path, var/constraint
  counts, objective sense, seed, time limit), periodic progress lines
  (Time / Iter / Objective / Violation / Perturbs, `*` on a new best), and a
  final solution summary.
- **`JsonlFormatter`** — one JSON object per event (`start` / `progress` /
  `result`). `progress` carries iteration, time, objective (null when not
  feasible or not finite), violation, feasibility, perturbations and `new_best`;
  `result` carries time, iterations, `termination` (the `TerminationReason`
  token — `time_limit` / `iteration_limit` / `feasible` / `no_budget`, which
  qualifies the two above), objective, `feasible`, `status` and `solution`.

Both write to a configurable `std::ostream` (default `std::cout`). The CLI
selects via `--format` and suppresses both with `--quiet`.

### Version

`cbls::kVersion` is a `constexpr const char*` in `include/cbls/cbls.h`.

### Parameter Locations

Top-level `solve()` arguments carry defaults (`time_limit=10.0`, `seed=42`,
`use_fj=true`, `hook=nullptr`, `lns=nullptr`, `lns_interval=3`,
`callback=nullptr`, `config={}`). Algorithmic tuning lives in `SearchConfig`
(outer loop) and `GFJConfig` (engine); inner-solver knobs are fields on
`FloatIntensifyHook`; LNS takes `destroy_fraction`. See the
[Parameters Table](#parameters-table).

---

## Threading & Determinism

### Multi-threading

The core `solve()` is single-threaded. `ParallelSearch` provides two modes (see
[Parallel Search](#solution-pool--parallel-search)): opportunistic portfolio
(independent seeds, shared mutex-protected pool) and deterministic epoch-sync.
No OpenMP, no work-stealing, no parallel DAG evaluation.

### Determinism

The solver is deterministic given the same seed, modulo wall-clock. All
randomness flows through a single `RNG` (`mt19937_64`). Unordered containers in
`delta_evaluate` / AD are used only for membership/lookup — iteration order does
not affect results because recomputation follows topological order.

**Wall-clock caveat (opportunistic mode):** `steady_clock` determines when to
stop. Different machine speeds yield different GLS-iteration counts and thus
different solutions. Same seed + same hardware + same load = reproducible.

**Deterministic mode** removes the caveat: epochs stop by GLS-iteration count
(`epoch_iterations`), not wall-clock, so same seed + `n_threads` +
`epoch_iterations` + `max_epochs` = identical result on any machine. The
`SearchConfig::max_iterations` field is what enforces the per-epoch
iteration-count stop.

### CLI flags

```
--threads N           number of threads (0 = auto-detect, default: 1)
--deterministic       enable deterministic epoch-sync mode
--epoch-iters INT     GLS iterations per epoch in deterministic mode (default: 5000)
--max-epochs INT      number of epochs in deterministic mode (default: 10)
```

---

## GPU

No GPU code. All computation is CPU-only. The bottleneck (`delta_evaluate`,
invoked per jump candidate) is inherently serial per-jump. GPU acceleration
would require batch-parallel evaluation of multiple candidate jumps
simultaneously, which the current architecture does not support.

---

## Reference

Davies, T. O., Didier, F., Perron, L. *ViolationLS: Constraint-Based Local
Search in CP-SAT.* CPAIOR 2024. (PDF in `docs/`.) Algorithm references
throughout this document (Algorithms 1–6) refer to this paper.
