#pragma once

#include "model.h"

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

    /// Grow with a model that `Model::extend` just grew (#167).
    ///
    /// EXISTING ROWS KEEP THEIR GLS WEIGHTS -- that is the whole point, and the
    /// reason this exists rather than a fresh `ViolationManager`: the weights are
    /// the search's accumulated knowledge of which rows are hard, and dropping
    /// them would restart the guided local search from scratch every time a
    /// column arrived. New rows start at `new_weight`, which defaults to 1 -- the
    /// value the constructor gives every row -- so an extension applied before
    /// the first batch leaves the weight vector where a whole-model construction
    /// would have.
    ///
    /// The cached total is invalidated rather than patched: an extension changes
    /// the node value of every row above a grown `Sum`, and `total_violation()`
    /// self-corrects against the node values on its next call.
    ///
    /// Throws `std::invalid_argument` if `ext` does not describe THIS model's
    /// constraint count, which is what stops a mismatched result silently sizing
    /// the weights to something the model does not have, or if `new_weight` is
    /// not finite and non-negative. Zero IS allowed and means the new rows start
    /// MASKED, since `FeasibilityJump::active` is `weight > 0` -- which is how
    /// `run()`'s linear-submodel phase masks the nonlinear rows.
    ///
    /// **Until this is called, the manager is out of step with the model and
    /// every read below throws** (`std::logic_error`): `Model::extend` grows the
    /// constraint list without touching the weights or the violation cache, and
    /// every loop here indexes both by constraint index -- `bump_weights` writes
    /// to them. Call it as soon as `extend` returns.
    void on_extended(const ExtensionResult& ext, double new_weight = 1.0);

    // Invalidate cached total (call after weights change or full_evaluate)
    void invalidate_cache() { cache_valid_ = false; }

    std::vector<double> weights;

private:
    /// Throw unless the weight vector and the violation cache are one entry per
    /// constraint of the model as it is NOW.
    ///
    /// The window this closes is the one between `Model::extend` returning and
    /// `on_extended` (#167), where every read below would be a heap overread and
    /// `bump_weights` a heap write. `weights` is public -- Python can set it
    /// (#156) -- so it also catches a caller that shortened it. One size compare
    /// against bodies that are already O(#constraints), or that already compare a
    /// snapshot's size.
    void require_row_count() const;
    void recompute_cache() const;

    Model& model_;
    mutable std::vector<double> cached_violations_;  // max(0, node_value(cid)) per constraint
    mutable double cached_total_ = 0.0;
    mutable bool cache_valid_ = false;
    mutable int incremental_updates_ = 0;  // counter to trigger periodic full recompute
};

}  // namespace cbls
