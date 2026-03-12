#pragma once

#include "model.h"
#include <vector>

namespace cbls {

struct AdaptiveLambda {
    double lambda_ = 1.0;
    int consecutive_infeasible = 0;
    int consecutive_feasible_stuck = 0;

    explicit AdaptiveLambda(double initial_lambda = 1.0)
        : lambda_(initial_lambda) {}

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

    AdaptiveLambda adaptive_lambda;
    std::vector<double> weights;

private:
    Model& model_;
};

}  // namespace cbls
