#include "cbls/structural_batch.h"

#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include "cbls/search.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>

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
    (void)gen;
    (void)touched;
#endif
}

bool StructuralBatch::take_first_improving(Model& model, ViolationManager& vm, MoveGenerator& gen,
                                           ConstSpan<int32_t> rows, bool full_scan) {
    bool changed = false;
    for (const Move& move : candidates_) {
        SavedValues saved = save_move_values(model, move);
        std::vector<int32_t> touched = apply_move(model, move);
        assert_move_within_scope(gen, touched);
        delta_evaluate(model, touched);
        const double delta =
            full_scan ? vm.weighted_delta_from(baseline_) : vm.weighted_delta_from(baseline_, rows);
        if (delta < kImprovementThreshold) {
            changed = true;  // improving: keep
            vm.snapshot_violations(baseline_);
            gen.on_commit(move);
        } else {
            undo_move(model, move, saved);
            delta_evaluate(model, touched);
        }
    }
    return changed;
}

bool StructuralBatch::take_best(Model& model, ViolationManager& vm, MoveGenerator& gen,
                                ConstSpan<int32_t> rows, bool full_scan) {
    std::ptrdiff_t best = -1;
    double best_delta = kImprovementThreshold;
    for (size_t i = 0; i < candidates_.size(); ++i) {
        const Move& move = candidates_[i];
        SavedValues saved = save_move_values(model, move);
        std::vector<int32_t> touched = apply_move(model, move);
        assert_move_within_scope(gen, touched);
        delta_evaluate(model, touched);
        const double delta =
            full_scan ? vm.weighted_delta_from(baseline_) : vm.weighted_delta_from(baseline_, rows);
        undo_move(model, move, saved);
        delta_evaluate(model, touched);
        if (delta < best_delta) {
            best_delta = delta;
            best = static_cast<std::ptrdiff_t>(i);
        }
    }
    if (best < 0) {
        return false;
    }
    const Move& move = candidates_[static_cast<size_t>(best)];
    std::vector<int32_t> touched = apply_move(model, move);
    delta_evaluate(model, touched);
    vm.snapshot_violations(baseline_);
    gen.on_commit(move);
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
    for (const std::unique_ptr<MoveGenerator>& gen : generators_) {
        if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        if (selection_ == StructuralSelection::ViolationGuided && !scope_can_improve(model, *gen)) {
            continue;
        }
        draw_candidates(ctx, *gen);
        if (candidates_.empty()) {
            continue;
        }
        const ConstSpan<int32_t> rows = affected_rows(model, *gen);
        const bool full_scan = rows.empty();
        changed = (selection_ == StructuralSelection::FirstImprovingSample
                       ? take_first_improving(model, vm, *gen, rows, full_scan)
                       : take_best(model, vm, *gen, rows, full_scan)) ||
                  changed;
    }
    return changed;
}

}  // namespace cbls
