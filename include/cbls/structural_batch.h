#pragma once

#include "counters.h"
#include "move_generator.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace cbls {

class Model;
class ViolationManager;
struct SearchConfig;

/// A STRUCTURAL batch, the List/Set peer of paper Algorithm 6's FJ/NJ.
/// Sweeps the registered `MoveGenerator`s and greedily commits candidates with
/// negative weighted delta_G under the current GLS weights W
/// (`total_violation()` is W-weighted). `FeasibilityJump`
/// jumps only scalars, so List/Set assignments improve only here. `run`
/// returns true if anything was committed: the caller owes `fj.resync()`.
///
/// Generators are built ONCE per search (built-ins plus a clone of each
/// `SearchConfig::move_generators` entry) and owned here, so each portfolio
/// worker's caches, cursors and counters are its own (#157).
///
/// DEADLINE: checked BETWEEN generators, never inside one, so a generator's
/// move set is never truncated and the overrun is one generator's work --
/// which the CALLER sizes (3-5 candidates for a built-in, sample size plus one
/// call's yield when sampling, unbounded for a registered generator under
/// `FirstImprovingSample`; see `MoveGenerator`'s `generate` contract). Needed
/// because sweep cost grows with the model and `solve(model, time_limit)` is
/// a contract (#105). Pinned by counting visited generators in
/// `a deadline that passes mid-sweep stops the sweep between generators`
/// (tests/test_structural_batch.cpp).
///
/// Cost: O(#constraints x (1 + #commits) + #generators x #candidates x
/// (delta_evaluate + |R_c|)), R_c being the rows a candidate can have moved
/// (its walked variables' G_v plus rows still differing from the snapshot,
/// #273). The per-commit term remains because `snapshot_violations` calls
/// `total_violation()`, which walks every row, on entry and after each
/// commit: cheap on a low-acceptance sweep, costly on a heavily committing
/// one; with large structures the second term dominates. Restricting the
/// refresh to moved rows would be exact but shifts `total_violation()`'s
/// 1000-call drift-resync schedule and so the digests
/// tests/test_structural_equivalence.cpp holds; re-record them deliberately or
/// not at all.
///
/// The check is unconditional per generator, not strided (the self-tuning
/// stride it replaced never fired once it exceeded the structured-variable
/// count): one clock read per generator, ~20-25ns via vDSO on a TSC
/// clocksource, ~1.4us on HPET -- too small to justify amortising.
class StructuralBatch {
public:
    /// `enabled` false builds nothing, skipping even `default_move_generators`'
    /// O(#variables) scan per worker; `solve()` passes false on a model with no
    /// structured variable, where the batch never runs.
    StructuralBatch(const Model& model, const SearchConfig& config, bool enabled);

    StructuralBatch(const StructuralBatch&) = delete;
    StructuralBatch& operator=(const StructuralBatch&) = delete;
    StructuralBatch(StructuralBatch&&) noexcept = default;
    StructuralBatch& operator=(StructuralBatch&&) noexcept = default;
    ~StructuralBatch() = default;

    /// One pass. Returns true if any move was committed.
    bool run(Model& model, ViolationManager& vm, RNG& rng, bool has_deadline,
             std::chrono::steady_clock::time_point deadline);

    [[nodiscard]] size_t generator_count() const noexcept { return generators_.size(); }
    /// The batch's OWN clone of generator `i`, not the registered instance
    /// (exposed so tests can check each worker got a distinct object).
    [[nodiscard]] const MoveGenerator& generator(size_t i) const { return *generators_.at(i); }

    /// Per generator since construction (#169): candidates applied-and-scored
    /// (not generated; sampling policies differ) and committed. Entry `i`
    /// belongs to `generator(i)`; names are copied since a clone's `name()`
    /// dies with it. Observational only: the sweep never reads them.
    [[nodiscard]] const std::vector<GeneratorCounters>& generator_counters() const noexcept {
        return counters_;
    }

private:
    bool take_first_improving(Model& model, ViolationManager& vm, MoveGenerator& gen,
                              GeneratorCounters& counters);
    bool take_best(Model& model, ViolationManager& vm, MoveGenerator& gen,
                   GeneratorCounters& counters);
    /// Snapshot the current violations into `baseline_`; no row differs then.
    void resnapshot(const ViolationManager& vm);
    /// After a walk over `walked`: the weighted change against `baseline_`
    /// over the rows that can differ (`differing_` plus the walked variables'
    /// G_v), updating `differing_`. See the note above `StructuralBatch::run`
    /// in structural_batch.cpp.
    double delta_after_walk(const Model& model, const ViolationManager& vm,
                            const std::vector<int32_t>& walked);
    /// Whether any row in the scope's G_v is violated at the baseline. See
    /// `StructuralSelection::ViolationGuided` for why skipping the rest is exact.
    [[nodiscard]] bool scope_can_improve(const Model& model, const MoveGenerator& gen) const;
    /// Fill `candidates_` from `gen`, up to the sample the policy asks for.
    void draw_candidates(MoveContext& ctx, MoveGenerator& gen);

    /// Record value and elements of every variable `candidates_` names before
    /// any is applied, once per sample; `apply_from_base` replays what the
    /// previous candidate disturbed. Scalars are included: the `MoveGenerator`
    /// contract allows them, and a missing one would be absent from
    /// `delta_evaluate`'s `touched` set.
    void snapshot_sample_base(const Model& model);
    /// Put the variables in `which` back to the baseline, via `assign` so
    /// nothing allocates.
    void restore_sample_base(Model& model, const std::vector<int32_t>& which) const;
    /// Copy the sampled variables into `accepted_*`, the sweep's fallback
    /// after a rejected last candidate.
    void record_accepted(const Model& model);
    /// Put the sampled variables at `accepted_*`.
    void restore_accepted(Model& model) const;
    /// Apply `move` to the sample baseline, as an absolute element vector
    /// would. Returns every variable whose value may have moved (the move's own
    /// plus those restored), which `delta_evaluate` needs in full.
    const std::vector<int32_t>& apply_from_base(Model& model, const Move& move);

    std::vector<std::unique_ptr<MoveGenerator>> generators_;
    /// One entry per generator, same order. See `generator_counters()`.
    std::vector<GeneratorCounters> counters_;
    StructuralSelection selection_;
    int sample_size_;
    /// Per-constraint violations of the last ACCEPTED assignment, which moves
    /// are judged against (why: note in structural_batch.cpp).
    std::vector<double> baseline_;
    /// Rows whose clamped violation differs bitwise from `baseline_`,
    /// ascending. Empty after each snapshot; kept exact by `delta_after_walk`.
    std::vector<int32_t> differing_;
    /// `delta_after_walk`'s scratch: the rows it scores, and a per-row
    /// dedup mark that is all zero between calls.
    std::vector<int32_t> score_rows_;
    std::vector<uint8_t> row_mark_;
    std::vector<Move> candidates_;
    /// The sample baseline, per variable `candidates_` names, before any
    /// candidate is applied (`snapshot_sample_base`).
    std::vector<int32_t> base_vars_;
    std::vector<double> base_values_;
    std::vector<std::vector<int32_t>> base_elements_;
    /// Assignment after this sample's last ACCEPTED candidate, parallel to
    /// `base_vars_`; what the sweep leaves when its last candidate is rejected.
    std::vector<double> accepted_values_;
    std::vector<std::vector<int32_t>> accepted_elements_;
    /// Variables the PREVIOUS candidate of this sample changed: exactly what
    /// the next must put back (candidates disturb nothing else). Doubles as
    /// `delta_evaluate`'s `touched` set, unioned with the current move's own.
    std::vector<int32_t> dirty_vars_;
    std::vector<int32_t> touched_;
};

}  // namespace cbls
