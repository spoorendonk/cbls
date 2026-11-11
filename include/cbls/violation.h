#pragma once

#include "model.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cbls {

/// The engine's blowup clamp: every non-finite or absurdly large violation
/// maps here, so inf/NaN stays well-ordered. It is also the largest objective
/// the violation machinery can tell from a blowup, so `record_best` installs it
/// as the sentinel bound for a feasible point with a non-finite objective
/// (#116); that is sound only while clamp and sentinel are one constant.
inline constexpr double kInfPenalty = 1.0e30;

/// A constraint node's value as a violation: max(0, value), with +inf, NaN
/// and values above kInfPenalty mapped to kInfPenalty (very bad, still
/// ordered). One definition for ViolationManager, `Model`'s jump scoring and
/// FJ's closed-form linear scorer, which must all agree on it.
inline double clamped_node_violation(double node_value) {
    // Before max(): std::max(0.0, NaN) is 0.0, which would mask NaN as
    // satisfied. NaN is a maximal violation; nothing says it holds.
    if (std::isnan(node_value)) {
        return kInfPenalty;
    }
    const double v = std::max(0.0, node_value);
    if (v > kInfPenalty) {  // also catches +inf
        return kInfPenalty;
    }
    return v;
}

class ViolationManager {
public:
    explicit ViolationManager(Model& model);

    double constraint_violation(int i) const;
    double total_violation() const;
    // Raw objective + unit-weighted total violation; InnerSolverHook descends
    // it. While solve() folds in `obj <= bound`, the objective counts twice:
    // fine for local polish, wrong for accept rules (LNS checks feasibility).
    double augmented_objective() const;
    bool is_feasible(double tol = kDefaultFeasibilityTolerance) const;
    std::vector<int> violated_constraints(double tol = kDefaultFeasibilityTolerance) const;
    void bump_weights(double factor = 1.0);

    // sum_c W[c] * delta_c if var_id <- j, uncommitted (the paper's -score).
    // Scalar variables only; throws on List/Set. `const` is logical only: it
    // mutates and restores the model, so it is NOT reentrant on a shared Model
    // (each search thread owns its own).
    double weighted_violation_delta(int32_t var_id, double j) const;

    // Per-constraint clamped violations, and the weighted change against such
    // a snapshot: the List/Set counterpart of weighted_violation_delta. They
    // difference PER CONSTRAINT because subtracting two totals (a) lets a
    // kInfPenalty row absorb every O(1) row (#100, #118) and (b) misses an
    // exact 0 for an unchanged row by an ulp of accumulator drift (#118).
    //
    // Snapshot the accepted state once, read weighted_delta_from() after each
    // applied candidate, re-snapshot only when one is kept. It throws unless
    // the snapshot has one entry per constraint. Both are O(#constraints);
    // snapshot_violations also refreshes the cached total, the other no cache.
    void snapshot_violations(std::vector<double>& out) const;
    double weighted_delta_from(const std::vector<double>& snapshot) const;

    /// The same quantity over `rows`: constraint indices, STRICTLY ASCENDING.
    /// Bit-identical to the full scan when `rows` covers the union of the
    /// moved variables' G_v (`Model::constraints_of_var`): rows outside it
    /// equal their snapshot and add nothing, and ascending order sums the same
    /// terms in the same sequence (pinned by
    /// `tests/test_structural_batch.cpp`).
    ///
    /// O(|rows|) against O(#constraints): the structural batch scores every
    /// candidate this way, and a List move touches only the rows reading it.
    ///
    /// `differing`, when non-null, gets each row of `rows` whose violation
    /// differs from the snapshot (the rows that contributed a term), appended
    /// in `rows` order (#273). Throws like the full scan on a wrong-size
    /// snapshot; an out-of-range entry in `rows` throws `std::out_of_range`.
    double weighted_delta_from(const std::vector<double>& snapshot, ConstSpan<int32_t> rows,
                               std::vector<int32_t>* differing = nullptr) const;

    // Invalidate cached total (call after weights change or full_evaluate)
    void invalidate_cache() { cache_valid_ = false; }

    std::vector<double> weights;

private:
    /// Throw unless `weights` and the violation cache have one entry per
    /// constraint of the model as it is NOW. `freeze()` / the first `solve()`
    /// of a model with an objective append the objective row, so a manager
    /// built before is one row short: reads would overread the heap and
    /// `bump_weights` write past it. Build a new manager then. Also catches a
    /// C++ caller shortening the public `weights` (Python's setter rejects a
    /// length change, #156, but against the manager's size, not the model's).
    /// One compare is noise against O(#constraints) bodies;
    /// `weighted_violation_delta` is not one, but FJ bypasses it (it calls
    /// `Model::weighted_violation_delta`), so only Python and tests pay.
    void require_row_count() const;
    void recompute_cache() const;

    Model& model_;
    mutable std::vector<double> cached_violations_;  // max(0, node_value(cid)) per constraint
    mutable double cached_total_ = 0.0;
    mutable bool cache_valid_ = false;
    mutable int incremental_updates_ = 0;  // counter to trigger periodic full recompute
};

}  // namespace cbls
