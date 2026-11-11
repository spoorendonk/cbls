#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cbls {

/// Which batch the ViolationLS outer loop ran (paper Algorithm 6 alternates
/// FJ/NJ; Structural is the List/Set peer). Public since #169 for
/// `SearchCounters` and `Tracer::batch_end`. A Structural batch commits outside
/// FJ's scan-set and jump-table, so the loop owes `fj.resync()` after it; a
/// Novelty batch rebuilds FJ's state itself (#209).
enum class BatchKind : std::uint8_t { FeasibilityJump, NoveltyJump, Structural };

/// Stable snake_case token for a `BatchKind` ("feasibility_jump",
/// "novelty_jump", "structural"). Returns a static string; never null.
const char* batch_kind_name(BatchKind kind);

/// One move generator's share of the structural work, keyed per generator
/// since a registered one may cover several variables (#165).
struct GeneratorCounters {
    /// `MoveGenerator::name()`, copied (the generator is a per-worker clone),
    /// with `#<index>` appended where one batch built several of that name:
    /// `merge`'s by-name path keys on it and built-ins name themselves by type.
    /// The suffix only matters when merged vectors differ in length or order;
    /// same-order vectors merge positionally. Not unique if a registered
    /// `name()` already ends in `#<n>` (documented limit, `StructuralBatch`).
    std::string name;
    /// Candidates APPLIED and SCORED, not generated (sampling may draw fewer).
    int64_t moves_tried = 0;
    /// Acceptances, not surviving changes: under `FirstImprovingSample` a later
    /// commit can undo an earlier one (see `MoveGenerator::on_commit`).
    int64_t moves_accepted = 0;
};

/// Where a run spent its work. Observational only: the search never reads
/// these back, so filling them leaves the trajectory unchanged. Fields
/// already on `SearchResult` (iterations, perturbations, LNS) stay there
/// (#169).
struct SearchCounters {
    /// `fj_batches + novelty_batches + structural_batches == batches` exactly
    /// (pinned by `tests/test_counters.cpp`).
    int64_t batches = 0;
    int64_t fj_batches = 0;
    int64_t novelty_batches = 0;
    /// Novelty batch work (#209): moves applied (each compound leg and each
    /// explored-then-undone move once; undos not), compound moves committed,
    /// and GLS weight bumps. Separates Novelty from FJ work, which the
    /// iteration count cannot (3 per move, 1 per bump).
    int64_t novelty_moves = 0;
    int64_t novelty_commits = 0;
    int64_t novelty_weight_bumps = 0;
    int64_t structural_batches = 0;

    /// Totals over `by_generator`, so callers need not sum the vector.
    int64_t structural_moves_tried = 0;
    int64_t structural_moves_accepted = 0;
    /// One row per GENERATOR the structural batch built, in its order (the
    /// `#<index>` suffix keeps same-name rows apart); a portfolio merges rows
    /// across workers. Empty when the batch builds no generator.
    std::vector<GeneratorCounters> by_generator;

    /// `InnerSolverHook::solve` calls (always counted; reads no clock) and
    /// their seconds, filled only under a wall-clock budget, else 0.0. Timing
    /// adds two clock reads per CALL, scaling with the run, where an
    /// iteration-budgeted run's other reads are O(1) per run (per batch with a
    /// `SolveCallback`); the gate keeps such runs bit-reproducible (#169), as
    /// `last_improvement_` in `src/search.cpp` does. Timed runs pay it whether
    /// read or not: the hook fires per new feasible point, so on the #125
    /// models at 1s (pk1 201, ucp13-1p 52, chain50 0 calls) at 1359ns/read
    /// (HPET, pessimistic; ~20-25ns via vDSO/TSC) it costs 0.055%, 0.014% and
    /// 0 of the budget (count x per-read cost, not differenced timings).
    int64_t inner_solver_calls = 0;
    double inner_solver_seconds = 0.0;

    /// Portfolio restarts beyond a worker's first solve, summed over workers;
    /// always 0 for `cbls::solve()`. Counts solves that produced a result, so
    /// a retry after a throwing solve is not a restart (`src/pool.cpp`).
    int64_t portfolio_restarts = 0;

    /// One-variable DAG walks served by the cone cache vs. collected and sorted
    /// because the variable was past the cache's cap (#271; `ConeCacheCounters`
    /// in dag_ops.h). Counted on the solving thread between `solve()`'s entry
    /// and exit, so user-callback walks are included. Summed over workers.
    int64_t cone_cache_walks = 0;
    int64_t cone_fallback_walks = 0;

    /// Shared objective bound engagement (#179): 0 for a single
    /// `cbls::solve()`; under `ParallelSearch` summed over workers. Filled with
    /// `ParallelConfig::share_objective_bound` on or off: the global best is
    /// read either way, and with sharing off only these counters depend on it.
    /// "Global" is the pool's, so after a restart it includes this worker's
    /// earlier solve (all four fields).
    ///
    /// Times the global best tightened this worker's bound (batch boundary, or
    /// record_best after a peer improved). Always 0 with sharing off. Excludes
    /// the cap from an adoption's re-grounding (an adoption is a
    /// `perturbations` event).
    int64_t shared_bound_tightenings = 0;
    /// New own-bests recorded while the pool already held a strictly better
    /// objective (a peer's, or after a restart this worker's earlier solve).
    /// Nonzero in both arms: sharing moves the bound, not the incumbent.
    int64_t own_best_behind_global = 0;
    /// Batches that RAN with this worker's bound looser than the global best,
    /// including those before its own first feasible point. 0 by construction
    /// with sharing on (tightened at the boundary; mid-batch peer gains are not
    /// charged), so it measures the control arm's wasted search.
    int64_t bound_behind_global_batches = 0;
    /// Wall seconds of those batches; same gate as `inner_solver_seconds`.
    double bound_behind_global_seconds = 0.0;

    /// Add `other` into this: every scalar adds, and `by_generator` merges by
    /// NAME (a present name adds in, a new one is appended). Used for both
    /// portfolio aggregations (across restarts and across workers).
    void merge(const SearchCounters& other);
};

}  // namespace cbls
