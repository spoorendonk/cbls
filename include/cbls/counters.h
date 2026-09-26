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
    /// `MoveGenerator::name()`, copied, with `#<index>` appended on every
    /// occurrence where one batch built more than one generator of that name.
    ///
    /// The suffix is there because `merge` below keys on this string and
    /// `MoveGenerator::name()` does not promise to be unique: the built-ins name
    /// themselves by TYPE, so two List variables both say "builtin_list".
    ///
    /// BE PRECISE ABOUT WHAT IT BUYS, because the obvious reading overstates it.
    /// `merge`'s two fast paths already handle the common case correctly without
    /// it: an empty target appends every row, and two same-length vectors whose
    /// names agree position-for-position are walked positionally -- and every
    /// worker builds the same generators in the same order, so that is what a
    /// portfolio actually produces. The collapse the suffix prevents needs the
    /// BY-NAME scan to run, i.e. two vectors that differ in length or ordering:
    /// merging a worker that built its batch against one that built a different
    /// set. So this is defensive work on a largely latent bug, not a fix for one
    /// that was happening.
    ///
    /// Unique within a batch UNLESS a registered generator's own `name()` already
    /// ends in `#<n>`, which `StructuralBatch`'s constructor records as a
    /// documented limit rather than defending against.
    ///
    /// The copy is needed either way: the generator is a per-worker clone that
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
    /// order -- and one row per GENERATOR, not per distinct `name()`, which is
    /// what the `#<index>` suffix on `GeneratorCounters::name` buys. A portfolio
    /// therefore reports the same rows a single `solve()` does, merged across its
    /// workers rather than duplicated per worker. Empty on a model with no
    /// structured variable and no registered generator, where the batch builds
    /// nothing at all.
    std::vector<GeneratorCounters> by_generator;

    /// `InnerSolverHook::solve` calls, and the seconds they took.
    ///
    /// THE SECONDS ARE ONLY FILLED WHEN THE RUN HAS A WALL-CLOCK BUDGET, and
    /// read 0.0 otherwise. That is not an oversight, and the argument is about
    /// SCALING rather than about a literal zero: an iteration-budgeted run does
    /// read the clock a bounded number of times already (`solve()`'s entry and
    /// exit, and `note_first_feasible` once; a `SolveCallback`, if one is
    /// attached, adds one per batch -- `docs/architecture.md` names them),
    /// but each of the three unconditional ones is O(1) per RUN, where timing the
    /// hook would add
    /// two reads per inner-solver CALL and so scale with the run. That is what
    /// #169's "no additional clock read" criterion protects, and it is what makes
    /// an iteration-budgeted run bit-reproducible. The gate is exactly the one
    /// `last_improvement_` already carries in `src/search.cpp`. The CALL COUNT is
    /// always filled -- it reads no clock.
    ///
    /// WHAT THE WALL-CLOCK SIDE COSTS, measured rather than asserted, because the
    /// gate is the RUN's and not the caller's: every timed run pays it whether or
    /// not it reads the field. The hook fires on each new feasible point, so the
    /// rate is the polish rate, not the iteration rate. On the three #125
    /// throughput models at their 1s budget: 201 calls (milp/pk1), 52
    /// (uc/ucp13-1p), 0 (minlp/chain50) -- so 402, 104 and 0 extra
    /// `steady_clock::now()` per second. At 1359ns per read, measured on an HPET
    /// clocksource (the pessimistic case; via the vDSO on a TSC one it is
    /// ~20-25ns, see structural_batch.h), that is 546us, 141us and 0 per second of
    /// budget -- 0.055%, 0.014% and 0. Derived from a call COUNT and a per-read
    /// cost rather than by differencing two whole-program timings, which under any
    /// load at all cannot resolve a number this size.
    int64_t inner_solver_calls = 0;
    double inner_solver_seconds = 0.0;

    /// Restarts a portfolio worker took beyond its first solve, summed over
    /// workers. Always 0 for a single `cbls::solve()`, which cannot restart
    /// itself. See `src/pool.cpp`'s restart loop for what a restart is and why
    /// one happens.
    ///
    /// Counted per SOLVE THAT PRODUCED A RESULT, so a retry after a throwing
    /// solve is not one: a worker whose first attempt threw and whose second
    /// succeeded has run one solve, not one solve and a restart.
    int64_t portfolio_restarts = 0;

    /// Add `other` into this, as `ParallelSearch` sums `perturbations`: every
    /// scalar adds, and `by_generator` merges by NAME (an entry whose name is
    /// already present adds into it, a new name is appended). Used in both
    /// places the portfolio aggregates -- across a worker's restarts and across
    /// the workers -- so the two cannot drift apart.
    void merge(const SearchCounters& other);
};

}  // namespace cbls
