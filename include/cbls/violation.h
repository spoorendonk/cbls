#pragma once

#include "model.h"

#include <vector>

namespace cbls {

struct AdaptiveLambda {
    double lambda_ = 1.0;
    int consecutive_infeasible = 0;
    int consecutive_feasible_stuck = 0;

    explicit AdaptiveLambda(double initial_lambda = 1.0) : lambda_(initial_lambda) {}

    void update(bool is_feasible, bool obj_improved);
};

class ViolationManager {
public:
    explicit ViolationManager(Model& model);

    double constraint_violation(int i) const;
    double total_violation() const;
    double augmented_objective() const;
    bool is_feasible(double tol = 1e-9) const;
    std::vector<int> violated_constraints(double tol = 1e-9) const;
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

    AdaptiveLambda adaptive_lambda;
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
