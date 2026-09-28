#pragma once

#include "model.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace cbls {

/// Default tolerance below which a constraint counts as satisfied.
///
/// Applied to the constraint node's violation value, which is an *absolute*
/// residual (for an equality row, the raw |lhs - rhs|). 1e-6 matches SCIP's
/// `numerics/feastol` default and `verify_model`'s tolerance. A far tighter
/// value is not meaningful on continuous/nonlinear models: on a row whose body
/// is of magnitude 1e4 it would demand ~13 significant digits, which double
/// precision cannot deliver.
inline constexpr double kDefaultFeasibilityTolerance = 1e-6;

/// The engine's blowup clamp. Every non-finite (or absurdly large) constraint
/// violation is mapped to this, so a search that wanders into inf/NaN stays
/// well-ordered instead of poisoning the violation cache and the structural
/// pass's move comparison.
///
/// It is therefore also the largest objective value the violation machinery can
/// still tell apart from a blowup, which is why `record_best` installs it as the
/// sentinel objective bound for a feasible point whose objective is not finite
/// (#116). That argument is only sound while the clamp and the sentinel are the
/// same number, so they read one constant rather than three copies.
inline constexpr double kInfPenalty = 1.0e30;

/// A constraint node's value as a violation: max(0, value), clamped.
///
/// Non-convex objectives/constraints (exp/pow/div blowups -- the MINLPLib
/// target) can drive a node value to +inf or NaN. Such a value would poison the
/// total_violation cache, the structural pass's move comparison and the
/// best-objective bookkeeping. Every non-finite (or absurdly large) violation is
/// mapped to kInfPenalty, so the search treats the point as very bad but still
/// well-ordered. One definition for ViolationManager, `Model`'s jump scoring
/// and FJ's closed-form linear scorer, which must all agree on it.
inline double clamped_node_violation(double node_value) {
    // NaN must be handled before max(): std::max(0.0, NaN) returns 0.0, which
    // would silently mask a NaN constraint as satisfied. NaN (e.g. inf-inf,
    // 0*inf) is treated as a maximal violation -- we have no evidence it holds.
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
    // Penalty-method objective: raw objective + unit-weighted total violation.
    // The continuous InnerSolverHook descends this. NOTE: when the objective is
    // folded in as the `obj <= bound` soft constraint (during solve()), the
    // objective term is double-counted; that is acceptable for the hook's local
    // polish but not for accept rules (LNS uses a real-feasibility comparison).
    double augmented_objective() const;
    bool is_feasible(double tol = kDefaultFeasibilityTolerance) const;
    std::vector<int> violated_constraints(double tol = kDefaultFeasibilityTolerance) const;
    void bump_weights(double factor = 1.0);

    // Change in total weighted violation if var_id <- j, without committing.
    // = sum_c W[c] * delta_c, the paper's -score (before negation). Scalar
    // variables only; throws on List/Set (see Model::per_constraint_violation_delta).
    // `const` is logical only: it transiently mutates and restores the model's
    // node/var state, so it is NOT reentrant on a shared Model (each search
    // thread owns its own Model, so this is safe in practice).
    double weighted_violation_delta(int32_t var_id, double j) const;

    // Per-constraint clamped violations of the current assignment, and the
    // weighted change against such a snapshot. Together they are the structural
    // (List/Set) counterpart of weighted_violation_delta: a move on a structured
    // variable cannot use that probe — it is scalar-only — but it needs the same
    // PER-CONSTRAINT differencing, for two reasons. (a) A row clamped to
    // kInfPenalty absorbs every O(1) real row when two whole sums are subtracted
    // instead (#100, and #118 where it blinded the structural pass under the
    // sentinel objective bound); differencing per constraint cancels it exactly.
    // (b) Even with no row clamped, two total_violation() readings disagree in
    // the last ulp for a move that changed nothing, because that value is an
    // incrementally maintained accumulator with a periodic resync; per-constraint
    // differencing reports an exact 0 for an unchanged row. See #118 and the
    // Structural Batch section of docs/architecture.md for the measurement.
    //
    // Intended use, from the caller that applies and rolls back candidate moves:
    // snapshot the accepted state once, then read weighted_delta_from() after
    // each candidate is applied, and re-snapshot only when one is kept.
    //
    // weighted_delta_from throws if the snapshot is not one constraint per
    // constraint of this model. Both are O(#constraints); snapshot_violations
    // also refreshes the cached total, weighted_delta_from touches no cache.
    void snapshot_violations(std::vector<double>& out) const;
    double weighted_delta_from(const std::vector<double>& snapshot) const;

    /// The same quantity restricted to `rows`, which must be constraint indices
    /// in STRICTLY ASCENDING order.
    ///
    /// Exact, not an approximation, whenever `rows` covers every constraint the
    /// caller's change can have touched -- the union of the moved variables' G_v
    /// (`Model::constraints_of_var`). A row outside that union has the same node
    /// value it had when the snapshot was taken, so `now == snapshot[i]` holds
    /// bitwise and the full-scan version skips it. Bit-identical rather than
    /// merely equal, on two counts: the same terms are summed, and ascending
    /// order makes them sum in the same sequence, which is what floating-point
    /// addition is sensitive to. `tests/test_structural_batch.cpp` pins the
    /// equality against the full scan on real structural moves.
    ///
    /// O(|rows|) where the full scan is O(#constraints). That is the whole
    /// reason it exists: the structural batch scores every candidate move this
    /// way, and a move on one List of a 40 000-row model can change only the
    /// handful of rows that read it.
    ///
    /// Throws, like the full scan, if the snapshot is not one entry per
    /// constraint. An out-of-range entry in `rows` throws `std::out_of_range`.
    double weighted_delta_from(const std::vector<double>& snapshot, ConstSpan<int32_t> rows) const;

    // Invalidate cached total (call after weights change or full_evaluate)
    void invalidate_cache() { cache_valid_ = false; }

    std::vector<double> weights;

private:
    /// Throw unless the weight vector and the violation cache are one entry per
    /// constraint of the model as it is NOW.
    ///
    /// Both are sized at construction, but the model can gain a row afterwards:
    /// `add_objective_soft_constraint`, which `freeze()` and the first `solve()`
    /// of a model with an objective run, appends the objective row. A manager
    /// built before that is one row short, and every read below would then be a
    /// heap overread and `bump_weights` a heap write -- reachable from Python as
    /// `ViolationManager(m)` followed by `m.freeze()`. Build a new manager then.
    /// `weights` is a public member, so it also catches a C++ caller that
    /// shortened it; Python cannot -- that setter is a `def_prop_rw` which
    /// rejects a length change (#156), though it checks against the manager's
    /// own size rather than the model's. One size compare is noise against
    /// bodies that are already O(#constraints) or that already compare a snapshot's
    /// size. `weighted_violation_delta` is the exception, and it is free for a
    /// different reason: FJ calls `Model::weighted_violation_delta` directly, so
    /// that overload is reached only from Python and the tests.
    void require_row_count() const;
    void recompute_cache() const;

    Model& model_;
    mutable std::vector<double> cached_violations_;  // max(0, node_value(cid)) per constraint
    mutable double cached_total_ = 0.0;
    mutable bool cache_valid_ = false;
    mutable int incremental_updates_ = 0;  // counter to trigger periodic full recompute
};

}  // namespace cbls
