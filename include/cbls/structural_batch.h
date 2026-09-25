#pragma once

#include "move_generator.h"

#include <chrono>
#include <cstddef>
#include <memory>
#include <vector>

namespace cbls {

class Model;
class ViolationManager;
struct SearchConfig;

/// A STRUCTURAL batch (paper Algorithm 6 has FJ/NJ; this is the List/Set peer).
///
/// Sweeps the registered `MoveGenerator`s, has each propose candidate moves for
/// the variables in its scope, and greedily commits the ones that reduce total
/// weighted violation (a negative weighted delta_G under the current GLS weights
/// W, since `total_violation()` is W-weighted). `FeasibilityJump` only jumps
/// scalar variables, so a List/Set-structured model cannot improve its
/// structural assignment without this. `run` returns true if any move was
/// committed, which tells the caller it owes an `fj.resync()`.
///
/// The generator set is built ONCE per search -- the built-ins for every
/// List/Set variable plus a clone of each generator on
/// `SearchConfig::move_generators` -- and owned here. Cloning is what makes the
/// portfolio safe: every worker constructs its own batch and therefore its own
/// clones, so a generator's caches, cursors and counters are per-worker state
/// rather than shared across threads (#157).
///
/// DEADLINE. The sweep is bounded BETWEEN generators, never inside one: each
/// generator's candidates are evaluated whole, so the reference move set is
/// never truncated for speed and the overrun is capped at one generator's work.
///
/// "One generator's work" is a cap the CALLER sizes, not a constant. Between
/// two checks the batch applies, `delta_evaluate`s, scores and rolls back every
/// candidate on offer -- 3-5 for a built-in, up to
/// `SearchConfig::structural_sample_size` plus one call's yield under the
/// sampling policies, and whatever a registered generator chooses under
/// `FirstImprovingSample`, which caps nothing. See the `generate` contract on
/// `MoveGenerator`, which is where that obligation is stated.
/// The bound is needed because the sweep's cost is unbounded in the model size,
/// and `solve(model, time_limit)` is a library contract: on a
/// 1500-List x 100-element model with 40k constraints a 0.5s budget ran
/// 1.19-1.25s unbounded versus 0.502s bounded (#105). Real benchmark models were
/// nowhere near it -- pharma-glsp's largest class swept 10 List variables in
/// ~0.5ms (retired in #28; the measurement is what motivated the bound) -- so
/// this is about honouring the contract on large models, not about the
/// benchmarks.
///
/// THAT 1.19-1.25s IS A PRE-#165 NUMBER and does not describe this code. It was
/// the O(#constraints) full rescan PER CANDIDATE, which the G_v restriction
/// below removed: the same model's unbounded sweep re-measures at 0.024s.
///
/// Be precise about what that removed, because the obvious reading is wrong.
/// The filler rows are NOT "no longer read" -- an earlier draft of this comment
/// said so and it was false. `ViolationManager::snapshot_violations` calls
/// `total_violation()`, which walks EVERY constraint (src/violation.cpp), and
/// the batch snapshots once on entry and again after every committed move. So
/// the per-candidate O(#constraints) term is gone; a per-pass and a per-COMMIT
/// one remain, and on a low-acceptance sweep -- which is what the 0.024s
/// measurement is -- those are cheap enough to disappear into the total.
///
/// The remaining cost is therefore
/// O(#constraints x (1 + #commits) + #generators x #candidates x
/// (delta_evaluate + |R_g|)), where R_g is the deduped union of the
/// generator's scope's G_v -- equal to G_v for the one-variable built-ins, and
/// the reason this is written per GENERATOR rather than per structured
/// variable. Still unbounded in the model size, so the bound
/// still matters -- but it now takes a model with large STRUCTURES, or a sweep
/// that commits heavily, rather than merely many rows, to reach the same
/// overrun. Restricting the baseline refresh to the moved rows would close the
/// commit term and is exact by the same argument the restricted delta rests on;
/// it is not done here because `snapshot_violations` reaches `total_violation()`,
/// whose 1000-call drift-resync counter would then advance on a different
/// schedule -- which moves the trajectory digests that
/// tests/test_structural_equivalence.cpp exists to hold still. Re-record those
/// deliberately or not at all.
/// `a deadline that passes mid-sweep stops the sweep between generators`
/// (tests/test_structural_batch.cpp) is what pins the bound, by counting the
/// generators a sweep visited rather than by timing it.
///
/// The check is unconditional per generator rather than strided. An earlier
/// self-tuning stride was deleted: because the stride persisted across passes
/// while its counter reset per pass, once it exceeded the model's structured
/// variable count it could never fire again, so it did nothing at all on 160 of
/// the 170 real pharma-glsp instances (2-6 List variables each). A per-generator
/// clock read costs ~1.4us only on an HPET clocksource like the machine that was
/// measured on; via the vDSO on a TSC clocksource it is ~20-25ns. Amortising a
/// 60x-inflated constant did not justify the complexity.
class StructuralBatch {
public:
    /// `enabled` false builds nothing at all -- not even the per-variable scan
    /// that `default_move_generators` would do. `solve()` passes false on a
    /// model with no structured variable, where the batch never runs and the
    /// scan would be one more O(#variables) pass per portfolio worker on a
    /// model with 710 000 columns.
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
    /// The batch's OWN clone of generator `i` -- not the instance the caller
    /// registered. Exposed so a test can prove that each worker got a distinct
    /// object.
    [[nodiscard]] const MoveGenerator& generator(size_t i) const { return *generators_.at(i); }

private:
    bool take_first_improving(Model& model, ViolationManager& vm, MoveGenerator& gen,
                              ConstSpan<int32_t> rows, bool full_scan);
    bool take_best(Model& model, ViolationManager& vm, MoveGenerator& gen, ConstSpan<int32_t> rows,
                   bool full_scan);
    /// The constraint rows a move from `gen` can change: the union of its
    /// scope's G_v, ascending and deduplicated. Empty scope returns an empty
    /// span, which the caller reads as "score with the full scan".
    ConstSpan<int32_t> affected_rows(const Model& model, const MoveGenerator& gen);
    /// Whether any row in the scope's G_v is violated at the baseline. See
    /// `StructuralSelection::ViolationGuided` for why skipping the rest is exact.
    [[nodiscard]] bool scope_can_improve(const Model& model, const MoveGenerator& gen) const;
    /// Fill `candidates_` from `gen`, up to the sample the policy asks for.
    void draw_candidates(MoveContext& ctx, MoveGenerator& gen);

    std::vector<std::unique_ptr<MoveGenerator>> generators_;
    StructuralSelection selection_;
    int sample_size_;
    /// Per-constraint violations of the last ACCEPTED assignment. See the note
    /// in structural_batch.cpp for why a move is judged against this rather than
    /// by differencing two whole-sum `total_violation()` readings.
    std::vector<double> baseline_;
    std::vector<Move> candidates_;
    std::vector<int32_t> rows_;  // scratch for a multi-variable scope's G_v union
};

}  // namespace cbls
