# CBLS Solver Architecture

Constraint-Based Local Search: simulated annealing over an expression DAG with
penalty-method feasibility, adaptive moves, gradient-based continuous
intensification, and large neighborhood search diversification.

## Table of Contents

1. [Overview](#overview)
2. [Expression DAG](#expression-dag)
3. [Model](#model)
4. [Violation & Penalty](#violation--penalty)
5. [Construction Heuristic: Feasibility Jump](#construction-heuristic-feasibility-jump)
6. [Simulated Annealing](#simulated-annealing)
7. [Adaptive Move Probabilities](#adaptive-move-probabilities)
8. [Inner Solver (Continuous Intensification)](#inner-solver-continuous-intensification)
9. [Large Neighborhood Search](#large-neighborhood-search)
10. [Solution Pool & Parallel Search](#solution-pool--parallel-search)
11. [Design Decisions](#design-decisions)
12. [Control Flow Diagram](#control-flow-diagram)
13. [Parameters Table](#parameters-table)
14. [I/O & Logging](#io--logging)
15. [Threading & Determinism](#threading--determinism)
16. [GPU](#gpu)

---

## Overview

CBLS is a hybrid metaheuristic/mathematical-programming solver for constrained
optimization over mixed discrete-continuous variables. The core algorithm:

1. **Models** problems as an expression DAG with typed variables (Bool, Int,
   Float, List, Set) and nonlinear constraints/objectives.
2. **Constructs** an initial feasible solution via Feasibility Jump (FJ), a
   greedy heuristic that iteratively reduces total constraint violation.
3. **Searches** via simulated annealing with type-specific moves, Metropolis
   acceptance, exponential cooling, and periodic reheat.
4. **Intensifies** continuous variables via a gradient-based inner solver
   (Newton steps, backtracking line search, multi-variable minimum-norm
   Newton), triggered periodically during SA.
5. **Diversifies** via large neighborhood search (destroy + FJ repair) fired
   every N reheats.
6. **Tracks** the best feasible solution found, with a solution pool for
   parallel multi-seed search.

The penalty method converts constrained optimization into unconstrained:
`F = obj + lambda * total_violation`, where lambda adapts during search.

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
| Collection   | At (indexing), Count, Lambda (functional aggregation) |
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
order. Used at initialization and after large state changes.

**Delta evaluation** (`delta_evaluate`): given a set of changed variable IDs,
BFS-marks dirty nodes upward through `dependent_ids`/`parent_ids`, then
recomputes only dirty nodes in topological order. This is the hot path during
SA — most moves change one variable and touch a small subgraph.

### Reverse-Mode Automatic Differentiation

`compute_partial(model, expr_id, var_id)` computes `d(expr)/d(var)` via
reverse-mode AD:

1. Initialize `adjoint[expr_id] = 1.0`
2. Traverse nodes in reverse topological order
3. For each node, propagate: `adjoint[child] += adjoint[node] * local_derivative(node, child_index)`
4. Variable adjoints use negative keys `-(var_id + 1)` to distinguish from
   node adjoints

`local_derivative` computes per-operation partial derivatives (chain rule
components). Discrete operations (At, Count, Lambda) return 0.

AD is used by Newton moves, gradient moves, and the inner solver.

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

### Finalization

`close()` computes the topological order, performs an initial full evaluation,
and sets the `closed_` flag. The model is immutable in structure after close
(variable values can still change).

### State Save/Restore

```cpp
struct State {
    std::vector<double> values;
    std::vector<std::vector<int32_t>> elements;
};
```

`copy_state()` snapshots all variable values and elements.
`restore_state(state)` restores them. Used for backtracking: SA saves the
best state found and restores it at the end. LNS saves state before
destruction for potential rollback.

---

## Violation & Penalty

**Files:** `include/cbls/violation.h`, `src/violation.cpp`

### ViolationManager

Tracks constraint satisfaction and computes the augmented objective.

- `constraint_violation(i)` = `max(0, constraint_node_value)` — non-negative
  violation for constraint `i`
- `total_violation()` = `sum(weight[i] * max(0, constraint_value[i]))` —
  weighted sum across all constraints
- `augmented_objective()` = `objective + lambda * total_violation()` — the
  value SA optimizes (called `F` throughout)
- `is_feasible(tol=1e-9)` — true when all constraint violations <= tolerance
- `violated_constraints(tol)` — returns indices of violated constraints

### Adaptive Lambda

The penalty multiplier `lambda` adapts during search via `AdaptiveLambda`:

| Condition | Action |
|-----------|--------|
| Infeasible for >10 consecutive steps | `lambda *= 1.5` (increase penalty) |
| Feasible but objective not improving for >20 steps | `lambda *= 0.8` (reduce penalty, explore more) |
| Feasible and objective improving | Reset both counters |

This balances exploration (low lambda allows infeasible moves) with
feasibility pressure (high lambda forces constraint satisfaction).

### Weight Bumping

`bump_weights(factor=1.0)` increments the weight of each currently-violated
constraint by `factor`. Used by FJ on stagnation to escape local minima by
changing the violation landscape.

---

## Construction Heuristic: Feasibility Jump

**File:** `src/search.cpp` (lines 50–181)

FJ is a greedy construction heuristic that initializes variable values to
reduce total constraint violation. It runs once before SA, allocated 20% of
the total time budget and at most 5000 iterations.

### Algorithm

```
initialize_random(model)
full_evaluate(model)
for iteration in [0, max_iterations):
    if no violated constraints: break
    for each variable:
        for each candidate value:
            trial-evaluate, measure violation reduction
    if best_reduction > 0:
        apply best (variable, value) pair
    else:
        bump_weights()  # stagnation escape
        perturb a random variable
```

### Candidate Generation by Type

| Type  | Candidates |
|-------|-----------|
| Bool  | Flip (1 candidate) |
| Int (domain <= 20) | All domain values except current |
| Int (domain > 20) | Neighbors (current +/- 1) + 8 random domain values |
| Float | 10 linspace values over [lb, ub] + Newton candidates from up to 3 violated constraints |
| List  | (not generated by FJ candidate function) |
| Set   | (not generated by FJ candidate function) |

**Float Newton candidates**: For each of the top 3 violated constraints,
compute `dg/dx` via AD. If `|dg/dx| > 1e-12`, propose
`x_new = clamp(x - g/dg, [lb, ub])`.

### Design Rationale vs FPR/LocalMip

FJ operates on the expression DAG directly, supporting nonlinear constraints
via `delta_evaluate`. FPR and LocalMip (from MIP heuristic literature) assume
linear constraints in sparse `Ax` form and use domain propagation / lift
bounds that require knowing constraint coefficients.

**Potential improvements from FPR/LocalMip ideas:**

- **WalkSAT-style repair**: pick a violated constraint, greedily fix the best
  variable for that constraint, with 30% random diversification. Better
  focused than current "bump weights + random perturb" stagnation handling.
- **Multi-attempt restarts**: FPR typically runs 10 attempts with different
  random seeds; FJ currently runs once.
- **Variable ranking by DAG connectivity**: prioritize high-fan-in variables
  (those appearing in many constraints) for earlier repair.

---

## Simulated Annealing

**File:** `src/search.cpp` (lines 207–374)

### Core Loop

Each iteration:

1. Select a random variable uniformly
2. Generate candidate moves (type-specific + enriched moves for floats)
3. Pick a move uniformly from candidates
4. Delta-evaluate the move to compute `delta_F` (change in augmented objective)
5. Accept/reject via Metropolis criterion
6. Update best tracking, adaptive lambda, move probabilities
7. Cool temperature; check for reheat

### Move Generation

Standard moves by type (see `generate_standard_moves` in `src/moves.cpp`):

| Type  | Moves |
|-------|-------|
| Bool  | `flip` — toggle 0/1 |
| Int   | `int_dec` (−1), `int_inc` (+1), `int_rand` (uniform in [lb,ub]) |
| Float | `float_perturb` — Gaussian with sigma = (ub−lb) * 0.1 |
| List  | `list_swap` (swap two elements), `list_2opt` (reverse a segment) |
| Set   | `set_add`, `set_remove`, `set_swap` |

**Enriched moves for Float variables:**

- `newton_tight` — pick a random violated constraint, compute Newton step
  `delta = -g / (dg/dx)`, clamp to bounds. Targets constraint satisfaction.
- `gradient_lift` — compute `df/dx` for the objective, step
  `delta = -0.1 * df/dx`, clamp to bounds. Targets objective improvement.

### Acceptance Criterion

```
if delta_F <= 0:
    accept (improvement)
else if temperature > 1e-15:
    accept with probability exp(-delta_F / temperature)
```

### Temperature Schedule

- **Initial**: `T_0 = max(|F| * 0.1, 1.0)` — scaled to problem magnitude
- **Cooling**: `T *= 0.9999` every iteration (exponential/geometric)
- **Reheat**: every 5000 iterations, `T = initial_temperature(best_F) * 0.5`

Reheat prevents premature convergence by periodically restoring acceptance of
uphill moves. The reheat temperature is 50% of the initial temperature
recomputed from the current best objective.

### Best Tracking

Two-tier tracking:
1. **Best feasible**: lowest objective among all feasible solutions seen
2. **Best overall**: lowest augmented objective `F` (used when no feasible
   solution has been found)

Feasible solutions always take priority. The best state is restored at the
end of search.

---

## Adaptive Move Probabilities

**File:** `src/moves.cpp` (lines 226–305)

The solver tracks 12 move types and adapts their selection probabilities based
on acceptance rates.

### Mechanism

- Initialized with uniform probability `1/12` per type
- On each move evaluation, record accept/reject for that type
- Every 1000 updates, **rebalance**:
  1. Compute acceptance rate per type: `rate = accepts / max(total, 1)`
  2. Normalize rates to probabilities
  3. Enforce 5% floor: redistribute deficit from below-floor types to
     above-floor types (3 iterations of redistribution)
  4. Final normalization

The 5% floor prevents move-type starvation — even rarely-accepted moves get
some exploration budget, which matters because acceptance rates change as
search progresses.

**Note:** Variable selection is uniform random. Move selection from the
generated candidate pool is also uniform. The adaptive probabilities influence
which move *types* are represented in the candidate pool via the `select()`
roulette wheel, though the current implementation generates all applicable
moves for the selected variable and picks uniformly.

---

## Inner Solver (Continuous Intensification)

**Files:** `include/cbls/inner_solver.h`, `src/inner_solver.cpp`

### Architecture

The `FloatIntensifyHook` implements `InnerSolverHook` — a constraint-directed
NLP sub-solver for continuous variables. It separates concerns: SA explores the
discrete variable space while the inner solver periodically tightens continuous
variables toward constraint satisfaction via gradient-based methods.

This hybrid metaheuristic/mathematical-programming approach avoids treating
continuous variables as discretized SA candidates, which would require
impractically fine granularity.

### Trigger Points

- **Every 10 discrete-variable acceptances** — after SA accepts a move
  involving Bool or Int variables
- **On every reheat** (every 5000 SA iterations)

### Mechanism 1: Single-Variable Newton Steps

For each Float variable, examine up to 3 violated constraints:

```
g = constraint_value
dg = d(constraint) / d(var)    # via reverse-mode AD
if |dg| > 1e-12:
    candidate = clamp(var - g/dg, [lb, ub])
    if augmented_objective improves: accept
```

This performs constraint tightening — one Newton step per variable per
constraint, zeroing the constraint by moving along its gradient.

### Mechanism 2: Backtracking Line Search on Objective

For each Float variable with a defined objective:

```
df = d(objective) / d(var)     # via reverse-mode AD
if |df| > 1e-12:
    step = 0.1
    for up to 5 iterations:
        candidate = clamp(var - step*df, [lb, ub])
        if augmented_objective improves: accept best
        step *= 0.5             # backtrack
```

This performs gradient descent on the objective with Armijo-style
backtracking.

### Mechanism 3: Multi-Variable Minimum-Norm Newton

For up to 5 violated constraints, simultaneously update all Float variables:

```
g = constraint_value
if |g| >= 1e-15:
    for each Float var j:
        dg_j = d(constraint) / d(var_j)
    grad_norm_sq = sum(dg_j^2)
    scale = -g / grad_norm_sq
    for each Float var j:
        var_j += scale * dg_j    # clamped to bounds
    if augmented_objective improves: accept
    else: rollback all changes
```

The update `delta_x = scale * gradient` is the minimum-norm solution to the
linearized constraint `g + gradient . delta_x = 0`. It distributes the
correction proportionally to gradient magnitude across all participating
variables (requires >= 2 Float variables with non-negligible gradients).

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_sweeps` | 3 | Coordinate-descent sweeps over all Float vars |
| `initial_step_size` | 0.1 | Starting step for backtracking line search |
| `max_line_search_steps` | 5 | Maximum backtracking halvings |
| `max_multi_var_constraints` | 5 | Max violated constraints for multi-var Newton |

---

## Large Neighborhood Search

**Files:** `include/cbls/lns.h`, `src/lns.cpp`

### Trigger

LNS fires every `lns_interval` reheats (i.e., every `lns_interval * 5000`
SA iterations). It provides diversification by partially destroying and
reconstructing the current solution.

### Destroy Phase

1. Save current state and augmented objective
2. Select `n_destroy = max(1, floor(num_vars * destroy_fraction))` random
   variables (default `destroy_fraction = 0.3`)
3. Randomize selected variables by type:
   - Bool: random 0/1
   - Int: uniform in [lb, ub]
   - Float: uniform in [lb, ub]
   - List: shuffle elements
   - Set: random subset with valid cardinality

### Repair Phase

1. Full evaluate (recompute all nodes after randomization)
2. Run FJ with 2000 iterations to reduce violation

### Acceptance

Pure improvement: accept if `new_F < old_F`. Otherwise rollback to saved
state. This is more conservative than SA acceptance — LNS only keeps
improvements.

### LNS Cycle

`destroy_repair_cycle(n_rounds=10)` runs multiple destroy-repair iterations,
returning the count of successful improvements.

---

## Solution Pool & Parallel Search

**Files:** `include/cbls/pool.h`, `src/pool.cpp`

### Solution Pool

A bounded, sorted collection of solutions for tracking best results across
parallel searches.

**Sorting order** (two criteria):
1. Feasible solutions first (`feasible > !feasible`)
2. Among same feasibility, ascending objective value

**Capacity**: default 10 solutions; excess trimmed after each insertion.

**Restart selection**: `get_restart_point()` picks uniformly from the first
half of the pool — biased toward better solutions but with diversity.

### Parallel Search

`ParallelSearch` runs `n_threads` (default 4) independent SA searches with
staggered seeds (`seed + thread_index`). Each thread:

1. Creates a model via a user-provided factory function
2. Calls `solve()` with the thread's unique seed
3. Submits the result to the shared (mutex-protected) pool

The best solution across all threads is returned, prioritizing feasibility
then objective value.

---

## Design Decisions

### Construction: FJ vs FPR/LocalMip

**Chosen:** FJ operates on the expression DAG directly, supporting nonlinear
constraints via delta evaluation and AD-based Newton candidates.

**Alternative:** FPR and LocalMip require linear `Ax` form but offer domain
propagation, WalkSAT-style constraint-directed repair, and multi-attempt
restarts. These ideas could be ported to the DAG setting — particularly
WalkSAT repair (pick violated constraint → fix best variable, 30%
diversification) and multi-attempt restarts.

### Acceptance Criterion: SA vs LAHC/Great Deluge/Threshold Accepting

**Chosen:** SA with exponential cooling + periodic reheat. Well-understood
theoretically (converges to global optimum under logarithmic cooling), with
reheat providing practical diversification.

**Alternative:** Late Acceptance Hill Climbing (LAHC) is parameter-free but
less understood theoretically. Great Deluge and Threshold Accepting avoid
probability-based acceptance but require water-level/threshold parameter
tuning.

### Move Selection: Adaptive Probabilities vs ALNS

**Chosen:** Per-move-type acceptance rate tracking with 5% floor and
rebalancing every 1000 evaluations. Simple, low overhead.

**Alternative:** Full Adaptive Large Neighborhood Search (ALNS) maintains
multiple destroy/repair operators with roulette-wheel selection, segment-based
scoring, and reaction factors. More powerful for operator selection but
significantly more complex.

### Temperature Schedule: Fixed Geometric vs Self-Adaptive

**Chosen:** Fixed `cooling_rate = 0.9999` with reheat every 5000 iterations.
Predictable behavior, easy to reason about.

**Alternative:** Self-adaptive schemes (e.g., record-to-record travel,
acceptance-rate targeting) avoid manual tuning. Luby restart sequences provide
theoretically optimal restart schedules.

### Constraint Weighting: Adaptive Lambda + Bump vs PAWS/SAPS

**Chosen:** Two-level adaptation: (1) global `lambda` multiplier that
increases after 10 consecutive infeasible steps (`*1.5`) and decreases when
feasible-stuck (`*0.8`); (2) per-constraint weight bumping on FJ stagnation.

**Alternative:** PAWS (Pure Additive Weighting Scheme) and SAPS (Scaling And
Probabilistic Smoothing) from SAT literature use more principled weight
update and decay mechanisms. PAWS adds weight to falsified clauses and
periodically smooths all weights.

### Diversification: LNS + Reheat vs Population/Restarts

**Chosen:** Single LNS operator (random destroy + FJ repair) with periodic
SA reheat. Simple and effective.

**Alternative:** Population-based approaches (genetic algorithms, scatter
search) maintain solution diversity explicitly. The solution pool exists but
isn't currently used for warm restarts during single-thread search. Luby or
geometric restart schemes from SAT provide systematic restart policies.

---

## Control Flow Diagram

```
solve(model, time_limit, seed, use_fj, hook, lns, lns_interval)
│
├── initialize_random(model)
├── full_evaluate(model)
│
├── [if use_fj]
│   └── fj_nl_initialize(model, 5000 iters, 20% time budget)
│       │
│       └── loop:
│           ├── find (variable, value) that most reduces violation
│           ├── if found: apply
│           └── if stagnated: bump_weights() + random perturb
│
├── T = max(|F|*0.1, 1.0)          # initial temperature
├── save best_state
│
└── SA main loop (while time remains):
    │
    ├── pick random variable
    ├── generate moves (standard + newton/gradient for floats)
    ├── pick one move uniformly
    ├── delta_evaluate → delta_F
    │
    ├── Metropolis accept?
    │   ├── yes: apply move, update adaptive_lambda
    │   │   ├── update best if improved
    │   │   ├── move_probs.update(type, accept=true)
    │   │   │
    │   │   └── [if hook && discrete var accepted]
    │   │       └── every 10 discrete accepts:
    │   │           hook->solve(model, vm)  # inner solver
    │   │
    │   └── no: undo move
    │       └── move_probs.update(type, accept=false)
    │
    ├── T *= 0.9999                 # cool
    │
    └── [every 5000 iters: REHEAT]
        ├── T = initial_temperature(best_F) * 0.5
        ├── hook->solve(model, vm)  # inner solver on reheat
        │
        └── [every lns_interval reheats: LNS]
            ├── destroy: randomize 30% of variables
            ├── repair: FJ with 2000 iterations
            └── accept if F improved, else rollback

    ── end loop ──

    restore best_state
    return SearchResult
```

---

## Parameters Table

| Parameter | Default | Location | Description |
|-----------|---------|----------|-------------|
| `time_limit` | (required) | `search.cpp:208` | Total search time in seconds |
| `seed` | (required) | `search.cpp:208` | RNG seed |
| `use_fj` | true | `search.cpp:208` | Enable Feasibility Jump initialization |
| `cooling_rate` | 0.9999 | `search.cpp:235` | SA geometric cooling factor |
| `reheat_interval` | 5000 | `search.cpp:236` | Iterations between reheats |
| `reheat_temp_factor` | 0.5 | `search.cpp:341` | Reheat temperature = initial_temp * factor |
| `initial_temp_scale` | 0.1 | `search.cpp:205` | `T_0 = max(abs(F)*scale, 1.0)` |
| `fj_time_fraction` | 0.2 | `search.cpp:221` | Fraction of time budget for FJ |
| `fj_max_iterations` | 5000 | `search.cpp:222` | Maximum FJ iterations |
| `hook_frequency` | 10 | `search.cpp:248` | Inner solver fires every N discrete accepts |
| `lns_interval` | (caller) | `search.cpp:209` | LNS fires every N reheats |
| `destroy_fraction` | 0.3 | `lns.cpp:11` | Fraction of variables randomized by LNS |
| `lns_repair_iters` | 2000 | `lns.cpp:48` | FJ iterations in LNS repair phase |
| `max_sweeps` | 3 | `inner_solver.h:21` | Inner solver coordinate-descent sweeps |
| `initial_step_size` | 0.1 | `inner_solver.h:22` | Line search starting step |
| `max_line_search_steps` | 5 | `inner_solver.h:23` | Max backtracking halvings |
| `max_multi_var_constraints` | 5 | `inner_solver.h:24` | Max constraints for multi-var Newton |
| `float_perturb_sigma` | (ub−lb)*0.1 | `moves.cpp:44` | Gaussian perturbation scale |
| `move_rebalance_interval` | 1000 | `moves.cpp:54` | Updates between probability rebalancing |
| `move_prob_floor` | 0.05 | `moves.cpp:261` | Minimum probability per move type |
| `adaptive_lambda_init` | 1.0 | `violation.h:10` | Initial penalty multiplier |
| `infeasible_threshold` | 10 | `violation.cpp:12` | Steps before lambda increase |
| `feasible_stuck_threshold` | 20 | `violation.cpp:19` | Steps before lambda decrease |
| `lambda_increase_factor` | 1.5 | `violation.cpp:13` | Lambda multiplier on infeasibility |
| `lambda_decrease_factor` | 0.8 | `violation.cpp:22` | Lambda multiplier when feasible-stuck |
| `pool_capacity` | 10 | `pool.h:22` | Maximum solutions in pool |
| `n_threads` | 4 | `pool.h:37` | Default parallel search threads |

---

## I/O & Logging

### Console Output

The solver library produces no output. There are no `std::cout` calls, no
logging framework, and no progress callbacks. Example programs use `printf`
after `solve()` returns to print results. Users wanting iteration-level
diagnostics must instrument externally (e.g., via the `InnerSolverHook`
interface or by inspecting `SearchResult` after completion). No verbosity
flag exists.

### File I/O

There is no file reading or writing anywhere in the library. Models are
constructed programmatically via the C++ or Python API. Benchmark data is
hardcoded as `constexpr` arrays (`benchmarks/chped/data.h`). There is no
serialization format — no MPS, LP, JSON, or protobuf support. To persist a
solution, users extract variable values from `SearchResult` and write them
out themselves.

### Parameters

Top-level parameters are arguments to `solve()` with defaults
(`search.h:27-31`): `time_limit=10.0`, `seed=42`, `use_fj=true`,
`hook=nullptr`, `lns=nullptr`, `lns_interval=3`. SA schedule parameters
(`cooling_rate`, `reheat_interval`, `fj_time_fraction`, `hook_frequency`)
are hardcoded constants in `search.cpp`. Inner solver parameters are fields
on `FloatIntensifyHook`. LNS takes `destroy_fraction`. There is no
centralized config struct — parameters are scattered across function
arguments, class fields, and local constants. See the
[Parameters Table](#parameters-table) for the full list.

---

## Threading & Determinism

### Multi-threading

The core solver (`solve()`) is single-threaded. `ParallelSearch` launches N
independent threads, each creating its own `Model` via a user-provided
factory function and calling `solve()` with staggered seeds
(`seed + thread_index`). Only the `SolutionPool` is shared
(mutex-protected). There is no shared state during search — thread safety is
achieved by isolation. No OpenMP, no work-stealing, no parallel evaluation
of the DAG.

### Determinism

The solver is deterministic given the same seed, with one caveat: wall-clock
time. All randomness flows through a single `RNG` (`mt19937_64`) seeded by
the `seed` parameter. Unordered containers (`unordered_set` in
`delta_evaluate`, `unordered_map` in AD) are used only for
membership/lookup — iteration order does not affect results because
recomputation follows topological order.

**Wall-clock caveat:** `std::chrono::steady_clock` determines when to stop
FJ and SA. Different machine speeds produce different iteration counts and
therefore different solutions. Same seed + same hardware + same load =
reproducible results.

---

## GPU

No GPU code. All computation is CPU-only. The bottleneck
(`delta_evaluate`) is inherently serial per-move due to sequential SA. GPU
acceleration would require batch-parallel evaluation (e.g., evaluating
multiple candidate moves simultaneously), which the current architecture
does not support.
