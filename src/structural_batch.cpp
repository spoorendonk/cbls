#include "cbls/structural_batch.h"

#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include "cbls/moves.h"
#include "cbls/search.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace cbls {

namespace {

/// A candidate is kept only if it lowers weighted violation by more than this: a guard against
/// committing on rounding noise, not a tuned parameter (unchanged since before #165).
constexpr double kImprovementThreshold = -1e-12;

}  // namespace

StructuralBatch::StructuralBatch(const Model& model, const SearchConfig& config, bool enabled)
    : selection_(resolve_structural_selection(config.structural_selection,
                                              config.structural_neighbours != nullptr)),
      sample_size_(std::max(1, config.structural_sample_size)) {
    if (!enabled) {
        return;
    }
    if (config.default_structural_generators) {
        for (const std::shared_ptr<const MoveGenerator>& gen :
             default_move_generators(model, config.structural_neighbours)) {
            generators_.push_back(gen->clone());
        }
    }
    for (const std::shared_ptr<const MoveGenerator>& gen : config.move_generators) {
        if (gen != nullptr) {
            generators_.push_back(gen->clone());
        }
    }
    // Checked once, not left to surface mid-search: a null `clone()` would be dereferenced on
    // every sweep, and a `scope()` naming a missing variable would throw std::out_of_range out
    // of `solve()` mid-run, naming no generator. O(#generators x |scope|) once per search.
    for (const std::unique_ptr<MoveGenerator>& gen : generators_) {
        if (gen == nullptr) {
            throw std::invalid_argument("MoveGenerator::clone() returned null");
        }
        for (int32_t var_id : gen->scope()) {
            // Called for its throw; the [[nodiscard]] span is discarded.
            static_cast<void>(model.constraints_of_var(var_id));
        }
    }
    // Built eagerly, parallel to `generators_`, so an idle generator reads zero rather than
    // missing. A REPEATED NAME CARRIES ITS INDEX: `SearchCounters::merge` keys on the name, and
    // the built-ins are named by type ("builtin_list"/"builtin_set"), so two Lists would merge
    // into one row. Every worker builds the same generators in the same order (built-ins by
    // variable id, then `SearchConfig::move_generators`), so suffixes agree across workers.
    // Both occurrences are suffixed, for symmetry. Documented limit: a registered name already
    // ending in `#<n>` can collide (`foo`, `foo`, `foo#1` -> `foo#0`, `foo#1`, `foo#1`), and the
    // portfolio then reports a different row shape than a single `solve()`.
    counters_.reserve(generators_.size());
    std::vector<std::string> names;
    names.reserve(generators_.size());
    for (const std::unique_ptr<MoveGenerator>& gen : generators_) {
        names.emplace_back(gen->name());
    }
    // Occurrences counted once, O(G): one row per structured variable, and a `std::count` per
    // entry is ~2.25M string compares per solve on a 1500-List model. Loses a little on the
    // one-to-three-generator models every in-tree benchmark has, where a small hash table is
    // more work than three compares; keys borrow `names`' storage, so no string is allocated.
    std::unordered_map<std::string_view, int> occurrences;
    occurrences.reserve(names.size());
    for (const std::string& name : names) {
        ++occurrences[name];
    }
    for (size_t i = 0; i < names.size(); ++i) {
        const bool repeated = occurrences[names[i]] > 1;
        GeneratorCounters entry;
        entry.name = repeated ? names[i] + '#' + std::to_string(i) : names[i];
        counters_.push_back(std::move(entry));
    }
}

void StructuralBatch::resnapshot(const ViolationManager& vm) {
    vm.snapshot_violations(baseline_);
    differing_.clear();
}

// COST (#273): O(|R_c| + |differing_|) per call, plus a sort of R_c when the G_v runs arrive
// out of order. Wins over scanning the whole scope where a candidate walks a small part of a
// large one: a `ListPartitionGenerator` over every CVRP route, whose move touches one or two
// (14-15% of instructions on X-n251/X-n502 under GranularBest, callgrind). Loses on a
// one-variable scope,
// where R_c IS the scope: one more O(|G_v|) mark-and-copy pass next to the walk's O(cone).
double StructuralBatch::delta_after_walk(const Model& model, const ViolationManager& vm,
                                         const std::vector<int32_t>& walked) {
    score_rows_.clear();
    bool ascending = true;
    const auto add = [&](int32_t ci) {
        uint8_t& mark = row_mark_[static_cast<size_t>(ci)];
        if (mark != 0) {
            return;
        }
        mark = 1;
        ascending = ascending && (score_rows_.empty() || score_rows_.back() < ci);
        score_rows_.push_back(ci);
    };
    for (const int32_t var_id : walked) {
        for (const int32_t ci : model.constraints_of_var(var_id)) {
            add(ci);
        }
    }
    for (const int32_t ci : differing_) {
        add(ci);
    }
    for (const int32_t ci : score_rows_) {
        row_mark_[static_cast<size_t>(ci)] = 0;
    }
    // Ascending: the full scan's summation order (floating-point addition is not associative).
    if (!ascending) {
        std::sort(score_rows_.begin(), score_rows_.end());
    }
    differing_.clear();
    return vm.weighted_delta_from(baseline_, {score_rows_.data(), score_rows_.size()}, &differing_);
}

bool StructuralBatch::scope_can_improve(const Model& model, const MoveGenerator& gen) const {
    const ConstSpan<int32_t> scope = gen.scope();
    if (scope.empty()) {
        return true;  // unknown scope: never skipped
    }
    for (int32_t var_id : scope) {
        for (int32_t ci : model.constraints_of_var(var_id)) {
            if (baseline_[static_cast<size_t>(ci)] > 0.0) {
                return true;
            }
        }
    }
    return false;
}

void StructuralBatch::draw_candidates(MoveContext& ctx, MoveGenerator& gen) {
    candidates_.clear();
    gen.generate(ctx, candidates_);
    if (selection_ == StructuralSelection::FirstImprovingSample) {
        return;  // exactly one call, which is the pre-#165 sample
    }
    // Top the sample up, bounded by sample size AND call count, so a generator returning little
    // can neither spin nor overrun the deadline, which is checked only between generators.
    for (int extra = 0; static_cast<int>(candidates_.size()) < sample_size_ && extra < sample_size_;
         ++extra) {
        const size_t before = candidates_.size();
        gen.generate(ctx, candidates_);
        if (candidates_.size() == before) {
            break;  // exhausted
        }
    }
}

// A move outside its generator's `scope()` breaks `scope_can_improve`'s skip under
// ViolationGuided/GranularBest: a generator is passed over although its moves reach violated
// rows. Scoring is still right (#273), so the silent lost opportunity is asserted at hand-over.
// Debug/sanitizer only: O(|touched| x |scope|) on the hot path, 1x1 for every built-in.
static void assert_move_within_scope(const MoveGenerator& gen,
                                     const std::vector<int32_t>& touched) {
#ifndef NDEBUG
    const ConstSpan<int32_t> scope = gen.scope();
    for (int32_t var_id : touched) {
        assert(std::find(scope.begin(), scope.end(), var_id) != scope.end() &&
               "a MoveGenerator's move changed a variable outside its scope()");
    }
#else
    // Unused only in this branch.
    static_cast<void>(gen);
    static_cast<void>(touched);
#endif
}

// ---------------------------------------------------------------------------
// The sample baseline (#164)
// ---------------------------------------------------------------------------
//
// A structured `Move::Change` carries POSITIONAL EDITS, valid only on the assignment they
// were built against, and `FirstImprovingSample` may commit SEVERAL candidates of one sample.
// Every candidate applies to the sample's starting state, so candidate k+1 replaces k's
// change, as `MoveGenerator::generate` documents and tests/test_structural_equivalence.cpp
// pins; replaying k+1's edit on k's state would be a different move (an `Erase` past the
// shortened end, none at all). Before each candidate the batch puts back what its predecessor
// changed: one allocation-free `assign` per such variable, one or two for every built-in. The
// snapshot is taken once per sample into buffers reused across sweeps, so the steady state
// allocates nothing.
void StructuralBatch::snapshot_sample_base(const Model& model) {
    base_vars_.clear();
    for (const Move& move : candidates_) {
        for (const Move::Change& change : move.changes) {
            if (std::find(base_vars_.begin(), base_vars_.end(), change.var_id) ==
                base_vars_.end()) {
                base_vars_.push_back(change.var_id);
            }
        }
    }
    // Grown, never shrunk, so every `assign` below is allocation-free once warm.
    if (base_elements_.size() < base_vars_.size()) {
        base_values_.resize(base_vars_.size());
        base_elements_.resize(base_vars_.size());
        accepted_values_.resize(base_vars_.size());
        accepted_elements_.resize(base_vars_.size());
    }
    // The guard tests one buffer, the others are indexed unchecked here and in restore_*; they
    // stay equal because only this site resizes them. Asserted so a later resize elsewhere fails
    // here, not out of bounds. Free under NDEBUG.
    assert(base_values_.size() == base_vars_.size());
    assert(accepted_values_.size() == base_vars_.size());
    assert(accepted_elements_.size() == base_vars_.size());
    for (size_t i = 0; i < base_vars_.size(); ++i) {
        const Variable& var = model.var(base_vars_[i]);
        base_values_[i] = var.value;
        base_elements_[i].assign(var.elements.begin(), var.elements.end());
    }
    record_accepted(model);
    dirty_vars_.clear();
}

void StructuralBatch::restore_sample_base(Model& model, const std::vector<int32_t>& which) const {
    for (int32_t var_id : which) {
        const auto it = std::find(base_vars_.begin(), base_vars_.end(), var_id);
        if (it == base_vars_.end()) {
            continue;  // not part of this sample; nothing was recorded for it
        }
        const auto i = static_cast<size_t>(it - base_vars_.begin());
        Variable& var = model.var_mut(var_id);
        var.value = base_values_[i];
        var.elements.assign(base_elements_[i].begin(), base_elements_[i].end());
    }
}

void StructuralBatch::record_accepted(const Model& model) {
    for (size_t i = 0; i < base_vars_.size(); ++i) {
        const Variable& var = model.var(base_vars_[i]);
        accepted_values_[i] = var.value;
        accepted_elements_[i].assign(var.elements.begin(), var.elements.end());
    }
}

void StructuralBatch::restore_accepted(Model& model) const {
    for (size_t i = 0; i < base_vars_.size(); ++i) {
        Variable& var = model.var_mut(base_vars_[i]);
        var.value = accepted_values_[i];
        var.elements.assign(accepted_elements_[i].begin(), accepted_elements_[i].end());
    }
}

const std::vector<int32_t>& StructuralBatch::apply_from_base(Model& model, const Move& move) {
    // The model differs from the baseline only in what the PREVIOUS candidate changed, so the
    // restore is proportional to a move, not to the sample.
    restore_sample_base(model, dirty_vars_);
    touched_ = dirty_vars_;  // the restore moved these; a node reading one is dirty
    dirty_vars_.clear();
    // Inlined, not `apply_move`, whose discarded return vector would cost a malloc and free
    // per candidate scored.
    for (const Move::Change& change : move.changes) {
        Variable& var = model.var_mut(change.var_id);
        if (is_structured(var.type)) {
            apply_element_edits(change, var.elements);
        } else {
            var.value = change.new_value;
        }
        dirty_vars_.push_back(change.var_id);
        if (std::find(touched_.begin(), touched_.end(), change.var_id) == touched_.end()) {
            touched_.push_back(change.var_id);
        }
    }
    return touched_;
}

bool StructuralBatch::take_first_improving(Model& model, ViolationManager& vm, MoveGenerator& gen,
                                           GeneratorCounters& counters) {
    bool changed = false;
    snapshot_sample_base(model);
    // Up front because this loop cannot exit early. If a deadline check is ever added, move
    // this inside, or the try count over-counts and `moves_accepted <= moves_tried` loosens.
    counters.moves_tried += static_cast<int64_t>(candidates_.size());
    for (const Move& move : candidates_) {
        const std::vector<int32_t>& touched = apply_from_base(model, move);
        assert_move_within_scope(gen, touched);
        delta_evaluate(model, touched);
        const double delta = delta_after_walk(model, vm, touched);
        if (delta < kImprovementThreshold) {
            changed = true;  // improving: keep
            ++counters.moves_accepted;
            resnapshot(vm);
            gen.on_commit(move);
            record_accepted(model);
        }
        // A REJECTED candidate is deliberately not rolled back: the next one restores the
        // baseline first and nothing reads the rejected state; the restore below fixes the end.
    }
    // Land on the kept assignment: sample base + last ACCEPTED candidate. UNCONDITIONAL, not
    // keyed on `dirty_vars_`: an empty final candidate (registered generators only) leaves it
    // empty after restoring the base, while `baseline_` describes the accepted state, so a skip
    // would hand the next generator a mismatched snapshot. `take_best`'s same-shaped guard is
    // correct: it wants the base, which an empty last candidate leaves.
    restore_accepted(model);
    delta_evaluate(model, base_vars_);
    // Not a score: refreshes `differing_`, which the next generator's candidates need to name
    // every row differing from `baseline_`, including a restore not landing bit-for-bit.
    static_cast<void>(delta_after_walk(model, vm, base_vars_));
    dirty_vars_.clear();
    return changed;
}

bool StructuralBatch::take_best(Model& model, ViolationManager& vm, MoveGenerator& gen,
                                GeneratorCounters& counters) {
    std::ptrdiff_t best = -1;
    double best_delta = kImprovementThreshold;
    snapshot_sample_base(model);
    // Every candidate is scored, so tries = sample size, added up front as in
    // `take_first_improving`; at most ONE is committed.
    counters.moves_tried += static_cast<int64_t>(candidates_.size());
    for (size_t i = 0; i < candidates_.size(); ++i) {
        const Move& move = candidates_[i];
        const std::vector<int32_t>& touched = apply_from_base(model, move);
        assert_move_within_scope(gen, touched);
        delta_evaluate(model, touched);
        const double delta = delta_after_walk(model, vm, touched);
        if (delta < best_delta) {
            best_delta = delta;
            best = static_cast<std::ptrdiff_t>(i);
        }
    }
    if (best < 0) {
        if (!dirty_vars_.empty()) {
            restore_sample_base(model, dirty_vars_);
            delta_evaluate(model, dirty_vars_);
            // As at the end of `take_first_improving`: kept for `differing_`.
            static_cast<void>(delta_after_walk(model, vm, dirty_vars_));
            dirty_vars_.clear();
        }
        return false;
    }
    const Move& move = candidates_[static_cast<size_t>(best)];
    const std::vector<int32_t>& touched = apply_from_base(model, move);
    delta_evaluate(model, touched);
    resnapshot(vm);
    gen.on_commit(move);
    ++counters.moves_accepted;
    return true;
}

// `baseline_` holds the per-row violations of the last ACCEPTED assignment, and a move is
// judged by `weighted_delta_from` against it, not by differencing two `total_violation()`
// readings: a kInfPenalty-clamped row swallows every real row (#118), and the incremental
// total drifts by a last ulp that `- 1e-12` cannot filter above 2^14 (docs/architecture.md,
// "Structural Batch"). Per-row differencing cancels a clamped row exactly and gives an
// unchanged row an exact 0.
//
// Scoring only the rows that can differ is the same double as a full rescan (#165, #273): a
// row equal to the snapshot is skipped by both, and ascending order keeps the sum's sequence.
// `differing_` (empty at a snapshot) plus the walked variables' G_v re-derives it exactly; the
// walk alone would not, since under FirstImprovingSample the next candidate's restore moves a
// kept candidate's rows off the snapshot and the one after does not walk them.
//
// The two trailing `delta_after_walk` calls cost one pass over G(base_vars_) or G(dirty_vars_)
// per generator per sweep, small next to the walk just before; without them `differing_` is
// stale for the next generator. Every in-tree op lands a row bit-for-bit on a restore, so they
// leave it empty on in-tree models. Delta calls read constraint nodes directly, so no invalidate
// is needed across apply/undo; the baseline is re-snapshotted only when a move is kept.
bool StructuralBatch::run(Model& model, ViolationManager& vm, RNG& rng, bool has_deadline,
                          std::chrono::steady_clock::time_point deadline) {
    if (generators_.empty()) {
        return false;
    }
    bool changed = false;
    resnapshot(vm);
    // Sized to the rows `snapshot_violations` just checked against the weights; marks are
    // cleared after each use.
    row_mark_.resize(baseline_.size(), 0);
    MoveContext ctx{model, vm, rng, selection_, &baseline_};
    for (size_t i = 0; i < generators_.size(); ++i) {
        MoveGenerator& gen = *generators_[i];
        if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if ((selection_ == StructuralSelection::ViolationGuided ||
             selection_ == StructuralSelection::GranularBest) &&
            !scope_can_improve(model, gen)) {
            continue;
        }
        draw_candidates(ctx, gen);
        if (candidates_.empty()) {
            continue;
        }
        GeneratorCounters& counters = counters_[i];
        changed = (selection_ == StructuralSelection::FirstImprovingSample
                       ? take_first_improving(model, vm, gen, counters)
                       : take_best(model, vm, gen, counters)) ||
                  changed;
    }
    return changed;
}

}  // namespace cbls
