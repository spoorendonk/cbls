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

/// A candidate is kept only if it strictly lowers weighted violation by more
/// than this. Pre-#165 constant, unchanged: it is a guard against committing on
/// rounding noise, not a tuned parameter.
constexpr double kImprovementThreshold = -1e-12;

}  // namespace

StructuralBatch::StructuralBatch(const Model& model, const SearchConfig& config, bool enabled)
    : selection_(config.structural_selection),
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
    // Both contract violations a generator can commit before it has proposed
    // anything, checked once here rather than left to surface mid-search:
    //
    //  - a `clone()` that returns null would be dereferenced on every sweep;
    //  - a `scope()` naming a variable this model does not have would throw
    //    std::out_of_range out of `constraints_of_var` on the first sweep, i.e.
    //    out of `solve()` some way into a run, where the caller can no longer
    //    tell which generator did it.
    //
    // O(#generators x |scope|) once per search, and the span read is the same
    // one the sweep will read.
    for (const std::unique_ptr<MoveGenerator>& gen : generators_) {
        if (gen == nullptr) {
            throw std::invalid_argument("MoveGenerator::clone() returned null");
        }
        for (int32_t var_id : gen->scope()) {
            // Cast to void: the span is deliberately discarded -- this call is here
            // for its throw, and `constraints_of_var` is [[nodiscard]].
            static_cast<void>(model.constraints_of_var(var_id));
        }
    }
    // Built here rather than lazily, so `generator_counters()` is always
    // parallel to `generators_` and a generator that never proposed anything
    // still has a row reading zero -- which is the answer a caller asking "did
    // this generator do anything" wants, rather than a missing entry.
    // A NAME THAT REPEATS CARRIES ITS INDEX, and that is load-bearing rather
    // than cosmetic: `SearchCounters::merge` keys on this string, and
    // `MoveGenerator::name()` does not promise to be unique. The built-ins name
    // themselves by TYPE -- `StandardStructuralGenerator::name()` returns
    // "builtin_list" or "builtin_set" for every List or Set variable -- so a
    // model with two Lists gives two rows spelled alike, and merging them
    // un-disambiguated folds both of a peer's rows into the first of ours while
    // the second reads zero. A registered generator whose own `name()` already
    // ends in `#<n>` can still collide with a suffix generated here -- `foo`,
    // `foo`, `foo#1` yields `foo#0`, `foo#1`, `foo#1` -- which is contrived
    // enough to be left as a documented limit rather than a rule a registrant
    // has to know. A single `solve()` reports both correctly, so the
    // portfolio would report a different row SHAPE for the same model.
    //
    // Every worker builds the same generators in the same order (the built-ins
    // in variable-id order, then `SearchConfig::move_generators` in
    // registration order), so the suffixed names agree across workers, which is
    // what keeps the merge exact. Suffixed on BOTH occurrences rather than only
    // the second, so the rows of one kind are symmetric.
    counters_.reserve(generators_.size());
    std::vector<std::string> names;
    names.reserve(generators_.size());
    for (const std::unique_ptr<MoveGenerator>& gen : generators_) {
        names.emplace_back(gen->name());
    }
    // Occurrences counted ONCE, in O(G), rather than rescanned per entry.
    // `counters_` carries one row per structured variable, and
    // `SearchCounters::merge` reasons about a 1500-List model, where a
    // `std::count` per entry is ~2.25M string comparisons per solve -- the same
    // call frequency, and twice the size, of the cost merge's positional walk
    // exists to remove. It loses a little on the one-to-three-generator model
    // every benchmark in the tree has, where one small hash table is more work
    // than three comparisons; the keys borrow `names`' storage, so nothing here
    // allocates a string of its own.
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

ConstSpan<int32_t> StructuralBatch::affected_rows(const Model& model, const MoveGenerator& gen) {
    const ConstSpan<int32_t> scope = gen.scope();
    if (scope.empty()) {
        return {};
    }
    if (scope.size() == 1) {
        // The common case, the built-ins included: G_v is already a contiguous
        // ascending run of the model's CSR, so there is nothing to build.
        return model.constraints_of_var(scope[0]);
    }
    rows_.clear();
    for (int32_t var_id : scope) {
        const ConstSpan<int32_t> gv = model.constraints_of_var(var_id);
        rows_.insert(rows_.end(), gv.begin(), gv.end());
    }
    // Sort-then-unique rather than a stamp array. A stamp dedups in O(1) per
    // entry but leaves the result in scope order, and ASCENDING order is what
    // makes the restricted sum bit-identical to the full scan (floating-point
    // addition is not associative). Since the sort is needed either way, it may
    // as well do the dedup.
    std::sort(rows_.begin(), rows_.end());
    rows_.erase(std::unique(rows_.begin(), rows_.end()), rows_.end());
    return {rows_.data(), rows_.size()};
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
    // Top the sample up. Bounded twice over -- by the sample size and by a call
    // count -- so a generator that returns nothing (or one move at a time) can
    // neither spin nor blow the batch's share of the deadline, which is checked
    // only between generators.
    for (int extra = 0; static_cast<int>(candidates_.size()) < sample_size_ && extra < sample_size_;
         ++extra) {
        const size_t before = candidates_.size();
        gen.generate(ctx, candidates_);
        if (candidates_.size() == before) {
            break;  // exhausted
        }
    }
}

// A move that changes a variable outside its generator's `scope()` is scored
// against the wrong rows -- WRONGLY, not merely inefficiently: the rows it
// actually moved are absent from the restricted sum, so the batch can commit a
// move that raises the weighted violation and never notice. Unlike the #156
// hazards this one does not crash, which makes it harder to find rather than
// easier, so the contract is checked where the move is handed over. Debug and
// sanitizer builds only: `touched` is already materialised, but the check is
// O(|touched| x |scope|) on the hot path and it is 1x1 for every built-in.
static void assert_move_within_scope(const MoveGenerator& gen,
                                     const std::vector<int32_t>& touched) {
#ifndef NDEBUG
    const ConstSpan<int32_t> scope = gen.scope();
    for (int32_t var_id : touched) {
        assert(std::find(scope.begin(), scope.end(), var_id) != scope.end() &&
               "a MoveGenerator's move changed a variable outside its scope()");
    }
#else
    // Cast to void, as in the scope-validation loop above: these are unused
    // only in this branch.
    static_cast<void>(gen);
    static_cast<void>(touched);
#endif
}

// ---------------------------------------------------------------------------
// The sample baseline (#164)
// ---------------------------------------------------------------------------
//
// A structured `Move::Change` carries POSITIONAL EDITS rather than the absolute
// element vector it used to carry, which is what removes four O(n) copies and
// two allocations from every candidate scored. Positions are relative, so
// something has to guarantee that an edit lands on the assignment it was built
// against -- and under `FirstImprovingSample` that guarantee is not free, because
// the batch may commit SEVERAL candidates from one sample in turn.
//
// With absolute vectors, applying candidate k+1 after committing candidate k
// silently wiped k's change: the vector k+1 carried was built before k ran. That
// is the behaviour `MoveGenerator::generate` documents, and it is the behaviour
// every trajectory in tests/test_structural_equivalence.cpp was recorded under.
// Replaying k+1's EDIT on top of k's committed state would be a different move
// -- and for an `Erase` past the shortened end, not a move at all.
//
// So the batch keeps the sample's starting elements and puts back, before each
// candidate, exactly what the previous one disturbed. The ledger per candidate
// is then one allocation-free `assign` per variable THAT CANDIDATE'S PREDECESSOR
// changed -- one or two for every built-in -- against the absolute form's three
// copies (into the Move, into the undo snapshot, and back out again) and two
// allocations, whatever the sample size. The snapshot itself is taken once per
// sample, and its buffers are reused across sweeps, so the steady state
// allocates nothing at all.
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
    // Grown, never shrunk: the buffers are reused across samples and across
    // sweeps, which is what keeps every `assign` below allocation-free once the
    // search is warm.
    if (base_elements_.size() < base_vars_.size()) {
        base_values_.resize(base_vars_.size());
        base_elements_.resize(base_vars_.size());
        accepted_values_.resize(base_vars_.size());
        accepted_elements_.resize(base_vars_.size());
    }
    // The guard tests one of the four but resizes all four, and the three others
    // are indexed unchecked below and in restore_*. They stay equal because this
    // is the only place any of them is resized -- assert it, so a later resize
    // elsewhere fails here under the sanitizer build rather than writing out of
    // bounds. Free under NDEBUG.
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
    // Every candidate is applied to the baseline, so the model differs from it
    // only in what the PREVIOUS candidate changed -- putting exactly those back
    // is enough, and is what keeps the restore proportional to a move rather
    // than to the sample.
    restore_sample_base(model, dirty_vars_);
    touched_ = dirty_vars_;  // the restore moved these; a node reading one is dirty
    dirty_vars_.clear();
    // Inlined rather than calling `apply_move`, whose return value is a freshly
    // allocated vector this caller would discard -- one malloc and free per
    // candidate SCORED, on the path the positional representation exists to take
    // allocations off.
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
                                           ConstSpan<int32_t> rows, bool full_scan,
                                           GeneratorCounters& counters) {
    bool changed = false;
    snapshot_sample_base(model);
    // Added up front because this loop CANNOT exit early -- every candidate is
    // applied and scored. Move it inside the loop if a deadline check is ever
    // added there, or the try count silently over-counts and
    // `moves_accepted <= moves_tried` stops being the bound it is read as.
    counters.moves_tried += static_cast<int64_t>(candidates_.size());
    for (const Move& move : candidates_) {
        const std::vector<int32_t>& touched = apply_from_base(model, move);
        assert_move_within_scope(gen, touched);
        delta_evaluate(model, touched);
        const double delta =
            full_scan ? vm.weighted_delta_from(baseline_) : vm.weighted_delta_from(baseline_, rows);
        if (delta < kImprovementThreshold) {
            changed = true;  // improving: keep
            ++counters.moves_accepted;
            vm.snapshot_violations(baseline_);
            gen.on_commit(move);
            record_accepted(model);
        }
        // A REJECTED candidate is not rolled back here, deliberately. The next
        // candidate restores the baseline before applying its own edit, and the
        // delta it is then judged by is measured against `baseline_` from the
        // node values `delta_evaluate` derives from the variables -- so nothing
        // in between reads the rejected state. Only the sweep's final state has
        // to be right, which the restore below sees to.
    }
    // Put the sweep at the assignment it is keeping: the sample baseline plus
    // the last ACCEPTED candidate, which is what the absolute form's final
    // `undo_move` left behind. Cheap when nothing was accepted, since
    // `accepted_*` is then the baseline itself.
    //
    // UNCONDITIONAL, not keyed on `dirty_vars_` being non-empty. A final
    // candidate carrying NO changes leaves `dirty_vars_` empty -- `apply_from_base`
    // restores the baseline and returns early -- while `baseline_` and
    // `accepted_*` still describe base + the last accepted candidate. Skipping
    // the restore there hands the next generator in this same sweep a model at
    // the baseline and a violation snapshot for a different assignment, after
    // `changed` was already set. No built-in can emit an empty `Move`, so this
    // needs a registered generator; `take_best` has the same-shaped guard and is
    // correct with it, because take_best wants the baseline and that is exactly
    // what an empty last candidate leaves.
    restore_accepted(model);
    delta_evaluate(model, base_vars_);
    dirty_vars_.clear();
    return changed;
}

bool StructuralBatch::take_best(Model& model, ViolationManager& vm, MoveGenerator& gen,
                                ConstSpan<int32_t> rows, bool full_scan,
                                GeneratorCounters& counters) {
    std::ptrdiff_t best = -1;
    double best_delta = kImprovementThreshold;
    snapshot_sample_base(model);
    // Every candidate is applied and scored below, whether or not it wins, so
    // the try count is the sample size -- exactly as under FirstImprovingSample,
    // and added up front for the same reason: neither loop can exit early. At
    // most ONE of them is then committed, which is the policy difference the
    // accepted count makes visible.
    counters.moves_tried += static_cast<int64_t>(candidates_.size());
    for (size_t i = 0; i < candidates_.size(); ++i) {
        const Move& move = candidates_[i];
        const std::vector<int32_t>& touched = apply_from_base(model, move);
        assert_move_within_scope(gen, touched);
        delta_evaluate(model, touched);
        const double delta =
            full_scan ? vm.weighted_delta_from(baseline_) : vm.weighted_delta_from(baseline_, rows);
        if (delta < best_delta) {
            best_delta = delta;
            best = static_cast<std::ptrdiff_t>(i);
        }
    }
    if (best < 0) {
        if (!dirty_vars_.empty()) {
            restore_sample_base(model, dirty_vars_);
            delta_evaluate(model, dirty_vars_);
            dirty_vars_.clear();
        }
        return false;
    }
    const Move& move = candidates_[static_cast<size_t>(best)];
    const std::vector<int32_t>& touched = apply_from_base(model, move);
    delta_evaluate(model, touched);
    vm.snapshot_violations(baseline_);
    gen.on_commit(move);
    ++counters.moves_accepted;
    return true;
}

// Per-constraint violations of the last ACCEPTED assignment. A move is judged by
// ViolationManager::weighted_delta_from against this, not by differencing two
// whole-sum total_violation() values. That subtraction had TWO defects, and only
// the first one needs a clamped row.
//
// 1. Clamped-row blindness (#118). A row clamped to kInfPenalty swallows the
//    real rows: 1e30 is fourteen orders of magnitude above an O(1) row, so both
//    sums round to the same double and `after < before - 1e-12` reads
//    `before < before`. That is #100's defect in this pass, and #116's sentinel
//    objective bound put a permanently clamped row into every model whose
//    feasible region contains a non-finite objective -- so the pass rejected
//    every structural move for as long as the sentinel was installed, however
//    much it improved the real rows.
//
// 2. Phantom improvements, on ANY model, clamped row or not, and predating #116.
//    Both readings came from total_violation()'s incremental accumulator
//    (cached_total_ += (new - old) * W), whose 1000-call recompute bounds the
//    accumulated rounding error without removing it, and `before` was threaded
//    across candidate moves -- so two readings taken at different points in that
//    drift cycle differ in the last ulp even when no constraint changed at all.
//    `- 1e-12` cannot filter that: x - 1e-12 == x for every double x > 2^14
//    (16384 itself is the last value it still moves), and GLS weights put
//    setcover's weighted total at ~4.4e6, where one ulp is 9.3e-10. Measured on
//    scp41/Set with no row clamped anywhere: 99 of 39627 candidates were
//    accepted with a true weighted delta of exactly 0, each of them setting
//    `changed` and forcing a needless fj.resync().
//
// Differencing per constraint fixes both: the clamped row cancels exactly, and
// an unchanged row contributes an exact 0 instead of a drifted total.
//
// #165 restricted the differencing to the rows a candidate can actually have
// changed -- the union of the moved generator's scope's G_v -- which is the same
// double and not merely the same answer: every other row satisfies
// `now == snapshot[i]` bitwise and is skipped by both versions, and ascending
// order keeps the surviving terms in the same sequence. The full scan remains
// the fallback for a generator that declares no scope.
//
// Both delta calls self-correct to the current node values (they read the
// constraint nodes directly), so no explicit invalidate is needed across the
// apply/undo dance; the baseline is re-snapshotted only when a move is kept.
bool StructuralBatch::run(Model& model, ViolationManager& vm, RNG& rng, bool has_deadline,
                          std::chrono::steady_clock::time_point deadline) {
    if (generators_.empty()) {
        return false;
    }
    bool changed = false;
    vm.snapshot_violations(baseline_);
    MoveContext ctx{model, vm, rng, selection_, &baseline_};
    for (size_t i = 0; i < generators_.size(); ++i) {
        MoveGenerator& gen = *generators_[i];
        if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (selection_ == StructuralSelection::ViolationGuided && !scope_can_improve(model, gen)) {
            continue;
        }
        draw_candidates(ctx, gen);
        if (candidates_.empty()) {
            continue;
        }
        const ConstSpan<int32_t> rows = affected_rows(model, gen);
        const bool full_scan = rows.empty();
        GeneratorCounters& counters = counters_[i];
        changed = (selection_ == StructuralSelection::FirstImprovingSample
                       ? take_first_improving(model, vm, gen, rows, full_scan, counters)
                       : take_best(model, vm, gen, rows, full_scan, counters)) ||
                  changed;
    }
    return changed;
}

}  // namespace cbls
