# Performance Profile: pharma-glsp (Benchmark 5)

**Date:** 2026-03-18
**Branch:** `bench/pharma-glsp`
**Build:** RelWithDebInfo, GCC, x86-64
**Instance:** Class A, single instance (`glsp_a_000`), 10s wall-clock

## Summary

| Metric | Value |
|---|---|
| Iterations (10s) | ~237K |
| Throughput | ~23.7K iter/s |
| Objective (10s) | 40,901 |
| Objective (30s) | 40,901 (no further improvement) |
| Variables | 24 (4 ListVar + 20 FloatVar) |
| DAG nodes | ~230 |

Throughput plateaus at 10s — the solver finds its best solution early and
doesn't escape. Increasing time limit from 10s to 30s yields zero improvement.

## Flat Profile (perf, self time)

| Self % | Function | Location |
|---|---|---|
| **20.5%** | `compute_partial` | `src/dag_ops.cpp:138` |
| **19.4%** | `delta_evaluate` | `src/dag_ops.cpp:92` |
| **12.9%** | `unordered_map::operator[]` | (inside `compute_partial`) |
| **9.6%** | `_int_malloc` | libc (hash node alloc) |
| **5.0%** | `malloc` | libc |
| **4.6%** | `_int_free` | libc (hash node dealloc) |
| **3.6%** | `local_derivative` | `src/dag.cpp` |
| **3.0%** | `child_val` | `src/dag.cpp` |
| **3.0%** | `malloc_consolidate` | libc |
| **2.6%** | `cfree` | libc |
| **1.9%** | `unordered_set::insert` | (inside `delta_evaluate`) |
| **1.9%** | `_Prime_rehash_policy` | libstdc++ |
| **1.8%** | `operator new` | libstdc++ |
| **1.6%** | `evaluate` | `src/dag.cpp` |

## Analysis

### Hotspot 1: `compute_partial` — 43% inclusive

`compute_partial` performs reverse-mode AD through the full DAG to compute
∂expr/∂var. It is called from two places per FloatVar iteration:

1. **`newton_tight_move`** (`moves.cpp:211`) — computes ∂constraint/∂var
2. **`gradient_lift_move`** (`moves.cpp:230`) — computes ∂objective/∂var

Each call allocates a fresh `std::unordered_map<int32_t, double>` for the
adjoint map, iterates the full topo order in reverse, and destroys the map on
return. For a ~230-node DAG this means ~230 hash insertions/lookups per call,
each requiring heap allocation of a hash node.

**Breakdown of the 43%:**
- 20.5% in `compute_partial` itself (loop + `local_derivative`)
- 12.9% in `unordered_map::operator[]` (hash probing)
- 9.6% in `malloc` / `_int_malloc` (hash node allocation)
- ~5% in `free` / `cfree` (hash node deallocation on destroy)

**Root cause:** `std::unordered_map` allocates one heap node per entry.
For a 230-entry map called ~23K times/sec (once per FloatVar iteration),
that's ~5.3M malloc/free pairs per second.

### Hotspot 2: `delta_evaluate` — 21% inclusive

`delta_evaluate` marks dirty nodes via BFS (using `unordered_set<int32_t>`)
then re-evaluates them in topo order. The `unordered_set` has the same
per-node-allocation problem but is smaller (only dirty nodes, not full DAG).

Called twice per iteration: once after apply, once after undo (if rejected).

### Hotspot 3: Inner solver hook — low frequency, high per-call cost

The `GLSPInnerSolverHook` runs every 10 discrete (list) move acceptances.
It writes all FloatVars and calls `delta_evaluate` once. Not a significant
fraction of total time given the low frequency.

## Proposed Optimizations (cross-benchmark)

These are generic engine improvements, not benchmark-specific hacks.

### P0: Replace `unordered_map` in `compute_partial` with flat array

**Impact:** ~40% of total CPU time recovered.

The adjoint map keys are node IDs (0..N-1) and variable IDs (encoded as
negative ints). Pre-allocate a `std::vector<double>` of size
`num_nodes + num_vars` once, zero-fill before each call, index directly.
Eliminates all hash overhead and heap allocation.

```cpp
// Before (current):
std::unordered_map<int32_t, double> adjoint;

// After (proposed):
thread_local std::vector<double> adjoint;
adjoint.assign(model.num_nodes() + model.num_vars(), 0.0);
// Use adjoint[nid] for nodes, adjoint[num_nodes + var_id] for vars
```

### P1: Replace `unordered_set` in `delta_evaluate` with bitset or flat vector

**Impact:** ~5% of total CPU time.

The dirty set keys are node IDs 0..N-1. Use a `std::vector<bool>` or a
`thread_local` bitset. Reset by clearing only the touched entries.

### P2: Cache `compute_partial` results across move candidates

**Impact:** ~10-20% for FloatVar-heavy models.

Currently `newton_tight_move` and `gradient_lift_move` each do a full
reverse-mode pass for the same variable. Factor into one call that returns
both ∂objective/∂var and ∂constraint/∂var (or cache the adjoint map across
the two calls).

### P3: Skip `compute_partial` for ListVar iterations

**Impact:** Minor for pharma-glsp, significant for list-heavy benchmarks.

When the selected variable is a ListVar, the enriched Float moves
(`newton_tight_move`, `gradient_lift_move`) are already skipped (line 282
checks `var.type == VarType::Float`). No issue here — just confirming.

### P4: Avoid full topo scan in `compute_partial`

Currently iterates all N nodes in reverse topo order even when only a subset
is reachable from `expr_id`. For partial derivatives of constraints (which
are leaves of sub-DAGs), most nodes are unreachable. A BFS-down from
`expr_id` to find relevant nodes, then reverse iterate only those, would
cut the inner loop significantly.

## Instance Scaling

| Class | J | T | Vars | ~Nodes | Expected iter/s | Notes |
|---|---|---|---|---|---|---|
| A | 5 | 4 | 24 | 230 | 23.7K | Profiled |
| B | 4 | 3 | 15 | 145 | ~35K | Smaller DAG |
| C | 6 | 2 | 14 | 140 | ~36K | Tight capacity |
| D | 10 | 6 | 66 | 640 | ~8K | 3× DAG → 3× slower |
| E | 20 | 10 | 210 | 2060 | ~2K | 9× DAG → 12× slower |

Class E is where the unordered_map overhead will dominate even more heavily —
2060 hash insertions per `compute_partial` call.

## Solution Quality Gap

| Class | LAHCM (paper) | CBLS | Gap |
|---|---|---|---|
| A | 5,588 | 38,789 | 6.9× |
| B | 2,668 | 8,892 | 3.3× |
| C | 31,335 | 6,505 | 0.2× (but only 60% feasible) |

The large gap suggests the solver is not exploring effectively, possibly due to:
1. Low iteration throughput limiting exploration
2. The backward-pass hook heuristic being too greedy (always latest periods)
3. Insufficient SA temperature / cooling tuning for this problem structure
4. LNS destroy-repair not well-suited to ListVar+FloatVar coupling

## Files Referenced

| File | Role |
|---|---|
| `src/dag_ops.cpp:92-136` | `delta_evaluate` — BFS dirty marking + topo recompute |
| `src/dag_ops.cpp:138-167` | `compute_partial` — reverse-mode AD |
| `src/search.cpp:273-410` | Main SA loop |
| `src/moves.cpp:205-241` | `newton_tight_move`, `gradient_lift_move` |
| `benchmarks/pharma-glsp/glsp_model.h` | DAG construction |
| `benchmarks/pharma-glsp/glsp_hook.h` | Inner solver hook |
