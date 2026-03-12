#include "cbls/violation.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cbls {

void AdaptiveLambda::update(bool is_feasible, bool obj_improved) {
    if (!is_feasible) {
        consecutive_infeasible++;
        consecutive_feasible_stuck = 0;
        if (consecutive_infeasible > 10) {
            lambda_ *= 1.5;
            consecutive_infeasible = 0;
        }
    } else {
        consecutive_infeasible = 0;
        if (!obj_improved) {
            consecutive_feasible_stuck++;
            if (consecutive_feasible_stuck > 20) {
                lambda_ *= 0.8;
                consecutive_feasible_stuck = 0;
            }
        } else {
            consecutive_feasible_stuck = 0;
        }
    }
}

ViolationManager::ViolationManager(Model& model) : model_(model) {
    weights.resize(model.constraint_ids().size(), 1.0);
}

double ViolationManager::constraint_violation(int i) const {
    if (i < 0 || i >= static_cast<int>(model_.constraint_ids().size()))
        throw std::out_of_range("constraint index out of range");
    int32_t cid = model_.constraint_ids()[i];
    return std::max(0.0, model_.node(cid).value);
}

double ViolationManager::total_violation() const {
    double total = 0.0;
    const auto& cids = model_.constraint_ids();
    for (size_t i = 0; i < cids.size(); ++i) {
        total += std::max(0.0, model_.node(cids[i]).value) * weights[i];
    }
    return total;
}

double ViolationManager::augmented_objective() const {
    double obj = 0.0;
    if (model_.objective_id() >= 0) {
        obj = model_.node(model_.objective_id()).value;
    }
    return obj + adaptive_lambda.lambda_ * total_violation();
}

bool ViolationManager::is_feasible(double tol) const {
    for (int32_t cid : model_.constraint_ids()) {
        if (model_.node(cid).value > tol) return false;
    }
    return true;
}

std::vector<int> ViolationManager::violated_constraints(double tol) const {
    std::vector<int> result;
    const auto& cids = model_.constraint_ids();
    for (size_t i = 0; i < cids.size(); ++i) {
        if (model_.node(cids[i]).value > tol) {
            result.push_back(static_cast<int>(i));
        }
    }
    return result;
}

void ViolationManager::bump_weights(double factor) {
    for (int i : violated_constraints()) {
        weights[i] += factor;
    }
}

}  // namespace cbls
