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
5. **Diversifies** on stagnation: a per-variable random perturbation, or — every
   `lns_interval`-th diversification kick — large neighborhood search (destroy +
   GFJ repair).
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
| Int (domain <= 256) | every value in `[lb, ub]` |
| Int (domain > 256)  | endpoints, neighbours `x±1`, and a 32-point rounded grid |
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
for the rest of the run. `Int` never had this problem: `int_jump_candidates`
always offers `x ± 1`. Float was the one type with no local move.

The probe supplies that local move, two-sided because at a saddle the descent
direction is precisely what a zero gradient cannot tell you. It is **off by
default and armed only after `perturbation_period` batches without improvement**,
then disarmed on the next new best. That gating is load-bearing rather than a
throughput optimisation: "stationary and nothing improving" is the *steady
state* of local search, and an always-on probe was measured ~9x worse on
`shiporig` across every seed, because its drip of numerically tiny improvements
kept the search from ever registering stagnation, so diversification never
fired. Armed only as a last resort, the same probe leaves productive searches
bit-identical and rescues frozen ones.

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
    stop if batch / global iteration budget or deadline reached
```

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
  closest to 0, full-evaluate, reset weights to 1, rebuild `V`/`Q`.
- `batch(batch_iterations)` — run `gls_loop` for at most `batch_iterations`
  GLS iterations; returns whether all active constraints are satisfied.
- `reset_weights()` — `W <- 1`, rebuild `V`/`Q` (called on a new best).
- `resync()` — rebuild `V`/`Q` from current state, keep weights (called after a
  hook/structural mutation outside the engine's bookkeeping).
- `perturb(probability)` — randomise each scalar var w.p. `probability`,
  full-evaluate, reset weights.
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
3. Unless `config.skip_init`, randomize the assignment (`initialize_random`).
4. Construct a single-phase `FeasibilityJump`; call `begin(set_initial_x)`.

`use_fj` is now vestigial — GFJ is always the engine.

### Main Loop

While time and `max_iterations` remain, each pass:

1. **Pick the batch kind.** With probability `structural_batch_probability`,
   STRUCTURAL; else with probability `novelty_jump_probability` (only when
   `use_compound_moves`), NOVELTY JUMP; else FEASIBILITY JUMP. Structural and
   novelty batches mutate state outside FJ's bookkeeping, so they set a `resync`
   flag.
2. **Run the batch.** `fj.batch(batch_iterations)`, `fj.apply_novelty_jump()`,
   or `structural_pass(...)`.
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
5. **On a new best**, reset GLS weights to 1 and resample `rho`.
6. **Otherwise** increment `stagnation`, and `resync()` if a structural/novelty
   /hook mutation happened.
7. **On `stagnation >= perturbation_period`**, diversify (below) and reset the
   stagnation counter.
8. Emit a progress callback (~1 s cadence, or immediately on a new best).

At the end, restore the best state — or, on an infeasible run, the *closest
approach* to the feasible region rather than the untouched initial assignment —
release the objective bound to `+inf`, full-evaluate, and return the best
feasible objective (or `+inf` / infeasible). `SearchResult::best_violation`
carries the largest real-constraint residual at the returned assignment.
`SearchResult::iterations` is the total GLS iteration count, not the batch count.

### Diversification

`diversify()` increments a `perturbations` counter and, every `lns_interval`-th
kick (when an `LNS` is supplied), runs [LNS](#large-neighborhood-search)
destroy-repair (then resets FJ weights, since LNS mutates state outside the
engine). Otherwise it calls `fj.perturb(perturbation_probability)`. Either way
it resamples `rho`.

### SearchConfig

```cpp
struct SearchConfig {
    bool skip_init = false;                 // keep current assignment (epoch restarts)
    int64_t max_iterations = 0;             // 0 = unlimited (use time_limit); counts GLS iterations
    bool use_fj = true;                     // vestigial: GFJ is always the engine
    int lns_interval = 3;                   // LNS fires every Nth diversification kick

    int64_t batch_iterations = 1000;        // GLS iterations per FJ batch
    int perturbation_period = 100;          // batches without improvement before diversifying
    double perturbation_probability = 0.1;  // per-variable randomisation probability
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

Destroyed variables are re-randomized by type (Bool/Int/Float uniform, List
shuffled, Set a random valid-cardinality subset).

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
objective.

**Deterministic epoch-sync mode** (`deterministic = true`): threads run
synchronized epochs of fixed GLS-iteration count (no wall-clock dependency).
Each epoch sets `SearchConfig::max_iterations = epoch_iterations`; after the
first epoch `skip_init = true` and FJ initialization is off. Per-epoch results
feed an elite `SolutionPool`; threads restart from elite states next epoch.
Thread seeds are `base_seed + epoch * n_threads + thread_id`. Repeats for
`max_epochs`.

`ParallelSearch::solve()` takes hook and LNS *factories* (these objects are
stateful and per-model); each thread builds its own instances.

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

**Chosen:** per-variable random perturbation by default; LNS (destroy + GFJ
repair, lexicographic accept) every `lns_interval`-th kick.

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
├── [unless config.skip_init] initialize_random(model)
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
    ├── if new best:  stagnation=0; fj.reset_weights(); resample rho
    │                 (pure feasibility → break)
    ├── else:         ++stagnation; if resync flag: fj.resync()
    │
    ├── if stagnation >= perturbation_period:
    │     diversify():
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
| `seed` | 42 | `solve()` arg | RNG seed |
| `use_fj` | true | `SearchConfig` | vestigial (GFJ always the engine) |
| `max_iterations` | 0 | `SearchConfig` | GLS-iteration cap (0 = use time_limit) |
| `skip_init` | false | `SearchConfig` | keep current assignment (epoch restarts) |
| `batch_iterations` | 1000 | `SearchConfig` | GLS iterations per FJ batch |
| `perturbation_period` | 100 | `SearchConfig` | stagnant batches before a diversification kick |
| `perturbation_probability` | 0.1 | `SearchConfig` | per-var randomisation probability on perturb |
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
| `--help` / `--version` | usage / `cbls::version` |

> **Removed flags** (SA-era): `--cooling-rate`, `--reheat-interval`,
> `--hook-frequency`, `--fj-time-fraction`. These no longer exist; the
> corresponding mechanisms were deleted in the ViolationLS port.

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
  `result`), each carrying iteration, time, objective (null when not feasible),
  violation, feasibility, perturbations, and `new_best`.

Both write to a configurable `std::ostream` (default `std::cout`). The CLI
selects via `--format` and suppresses both with `--quiet`.

### Version

`cbls::version` is a `constexpr const char*` in `include/cbls/cbls.h`.

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
