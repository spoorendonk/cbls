#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cbls {

/// Which kind of batch the ViolationLS outer loop ran (paper Algorithm 6
/// alternates FJ/NJ; STRUCTURAL is the List/Set peer added in P4).
///
/// Public because `SearchCounters` buckets by it and `Tracer::batch_end` reports
/// it. It was a detail of `src/search.cpp` until #169; nothing else about it
/// changed.
///
/// Structural and Novelty batches commit changes outside FeasibilityJump's
/// scan-set and jump-table, so the loop owes an `fj.resync()` after them.
enum class BatchKind : std::uint8_t { FeasibilityJump, NoveltyJump, Structural };

/// Stable snake_case token for a `BatchKind` ("feasibility_jump",
/// "novelty_jump", "structural"). Returns a static string; never null.
const char* batch_kind_name(BatchKind kind);

/// One registered move generator's share of the structural work.
///
/// Per generator rather than per variable because #165 made the generator the
/// unit the batch actually sweeps: a built-in covers one List/Set variable, but
/// a registered one may cover several, and `MoveGenerator::name()` is the only
/// identity either has.
struct GeneratorCounters {
    /// `MoveGenerator::name()`, copied. The generator is a per-worker clone that
    /// dies with its search, so the counters cannot borrow its `string_view`.
    std::string name;
    /// Candidates APPLIED and SCORED -- not candidates generated. The two differ
    /// when a generator offers more than the sampling policy draws.
    int64_t moves_tried = 0;
    /// The subset committed. Under `FirstImprovingSample` several candidates of
    /// one sample can be committed in turn, and a later one can undo an earlier
    /// one (see `MoveGenerator::on_commit`), so this counts ACCEPTANCES rather
    /// than surviving changes.
    int64_t moves_accepted = 0;
};

/// Where a run spent its work. Observational only: nothing in the search reads
/// any of these back, so a run that fills them takes the trajectory it would
/// have taken without them.
///
/// `SearchResult` already carried `iterations`, `perturbations`, `lns_repairs`
/// and `lns_repairs_accepted`; those stay where they are rather than being
/// duplicated here (#169 says "LNS as today"). What is new is the breakdown the
/// old fields could not answer: which KIND of batch the budget went to, how much
/// of the structural sweep was accepted, and what the inner solver cost.
struct SearchCounters {
    /// Batches run. `fj_batches + novelty_batches + structural_batches == batches`
    /// exactly -- `pick_batch_kind` returns one of the three and every batch is
    /// counted once, which `tests/test_counters.cpp` pins.
    int64_t batches = 0;
    int64_t fj_batches = 0;
    int64_t novelty_batches = 0;
    int64_t structural_batches = 0;

    /// Totals over `by_generator` below, kept as their own fields so a caller
    /// that only wants "how much structural work happened" does not have to sum
    /// a vector.
    int64_t structural_moves_tried = 0;
    int64_t structural_moves_accepted = 0;
    /// One entry per generator the structural batch built, in the batch's own
    /// order. Merged BY NAME across portfolio workers, so a portfolio reports
    /// one row per distinct generator name rather than one per worker per
    /// generator. Empty on a model with no structured variable and no
    /// registered generator, where the batch builds nothing at all.
    std::vector<GeneratorCounters> by_generator;

    /// `InnerSolverHook::solve` calls, and the seconds they took.
    ///
    /// THE SECONDS ARE ONLY FILLED WHEN THE RUN HAS A WALL-CLOCK BUDGET, and
    /// read 0.0 otherwise. That is not an oversight: `docs/architecture.md`
    /// guarantees an iteration-budgeted run (`time_limit <= 0`) reads no clock
    /// at all, which is what makes it bit-reproducible, and #169 asks for these
    /// counters to cost no additional clock read. The gate is exactly the one
    /// `last_improvement_` already carries in `src/search.cpp`. The CALL COUNT
    /// is always filled -- it reads no clock.
    int64_t inner_solver_calls = 0;
    double inner_solver_seconds = 0.0;

    /// Restarts a portfolio worker took beyond its first solve, summed over
    /// workers. Always 0 for a single `cbls::solve()`, which cannot restart
    /// itself. See `src/pool.cpp`'s restart loop for what a restart is and why
    /// one happens.
    int64_t portfolio_restarts = 0;

    /// Add `other` into this, as `ParallelSearch` sums `perturbations`: every
    /// scalar adds, and `by_generator` merges by NAME (an entry whose name is
    /// already present adds into it, a new name is appended). Used in both
    /// places the portfolio aggregates -- across a worker's restarts and across
    /// the workers -- so the two cannot drift apart.
    void merge(const SearchCounters& other);
};

}  // namespace cbls
