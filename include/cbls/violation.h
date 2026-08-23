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
/// pass's `after < before` test.
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

    // Invalidate cached total (call after weights change or full_evaluate)
    void invalidate_cache() { cache_valid_ = false; }

    std::vector<double> weights;

private:
    void recompute_cache() const;

    Model& model_;
    mutable std::vector<double> cached_violations_;  // max(0, node.value) per constraint
    mutable double cached_total_ = 0.0;
    mutable bool cache_valid_ = false;
    mutable int incremental_updates_ = 0;  // counter to trigger periodic full recompute
};

}  // namespace cbls
