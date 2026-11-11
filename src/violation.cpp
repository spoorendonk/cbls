#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cbls {

// clamped_node_violation lives in violation.h: model.cpp's jump scoring and
// FJ's closed-form linear scorer must apply the identical clamp.

ViolationManager::ViolationManager(Model& model) : model_(model) {
    weights.resize(model.constraint_ids().size(), 1.0);
    cached_violations_.resize(model.constraint_ids().size(), 0.0);
}

void ViolationManager::require_row_count() const {
    const size_t nc = model_.constraint_ids().size();
    if (weights.size() != nc || cached_violations_.size() != nc) {
        throw std::logic_error(
            "ViolationManager: the weight vector is not one entry per constraint of this model. "
            "Either a row was added after this manager was built -- the objective row, which "
            "freeze() and the first solve() of a model with an objective add; build a new "
            "manager after them -- or a C++ caller resized the public weights vector");
    }
}

double ViolationManager::constraint_violation(int i) const {
    if (i < 0 || i >= static_cast<int>(model_.constraint_ids().size())) {
        throw std::out_of_range("constraint index out of range");
    }
    int32_t cid = model_.constraint_ids()[i];
    return clamped_node_violation(model_.node_value(cid));
}

void ViolationManager::recompute_cache() const {
    require_row_count();
    const auto& cids = model_.constraint_ids();
    cached_total_ = 0.0;
    for (size_t i = 0; i < cids.size(); ++i) {
        cached_violations_[i] = clamped_node_violation(model_.node_value(cids[i]));
        cached_total_ += cached_violations_[i] * weights[i];
    }
    cache_valid_ = true;
    incremental_updates_ = 0;
}

double ViolationManager::total_violation() const {
    require_row_count();
    if (!cache_valid_) {
        recompute_cache();
        return cached_total_;
    }

    // Periodically recompute from scratch to prevent floating-point drift
    if (++incremental_updates_ >= 1000) {
        recompute_cache();
        return cached_total_;
    }

    // Incremental update: check which constraints changed
    const auto& cids = model_.constraint_ids();
    for (size_t i = 0; i < cids.size(); ++i) {
        double new_viol = clamped_node_violation(model_.node_value(cids[i]));
        if (new_viol != cached_violations_[i]) {
            cached_total_ += (new_viol - cached_violations_[i]) * weights[i];
            cached_violations_[i] = new_viol;
        }
    }
    return cached_total_;
}

double ViolationManager::augmented_objective() const {
    double obj = 0.0;
    if (model_.objective_id() >= 0) {
        obj = model_.node_value(model_.objective_id());
        if (!std::isfinite(obj)) {
            obj = kInfPenalty;  // non-convex blowup: keep the metric ordered
        }
    }
    return obj + total_violation();
}

void ViolationManager::snapshot_violations(std::vector<double>& out) const {
    // total_violation() brings cached_violations_ up to date, so the snapshot is
    // a copy, not a second scan.
    total_violation();
    out = cached_violations_;
}

double ViolationManager::weighted_delta_from(const std::vector<double>& snapshot) const {
    require_row_count();
    const auto& cids = model_.constraint_ids();
    if (snapshot.size() != cids.size()) {
        throw std::invalid_argument("weighted_delta_from: snapshot size != constraint count");
    }
    // Per-constraint differencing, as in Model::weighted_violation_delta (a
    // clamped 1e30 row cancels exactly), and an unchanged row contributes an
    // exact 0, not cached_total_'s last-ulp drift (#118).
    //
    // Does not update the cache: the candidate is usually rolled back, and
    // total_violation() resyncs on its next call.
    double delta = 0.0;
    for (size_t i = 0; i < cids.size(); ++i) {
        const double now = clamped_node_violation(model_.node_value(cids[i]));
        if (now != snapshot[i]) {
            delta += weights[i] * (now - snapshot[i]);
        }
    }
    return delta;
}

double ViolationManager::weighted_delta_from(const std::vector<double>& snapshot,
                                             ConstSpan<int32_t> rows,
                                             std::vector<int32_t>* differing) const {
    require_row_count();
    const auto& cids = model_.constraint_ids();
    if (snapshot.size() != cids.size()) {
        throw std::invalid_argument("weighted_delta_from: snapshot size != constraint count");
    }
    // Must match the full scan term for term, `now != snapshot[i]` skip
    // included, to produce the SAME double; a divergence changes trajectories.
    double delta = 0.0;
    for (int32_t ci : rows) {
        if (ci < 0 || static_cast<size_t>(ci) >= cids.size()) {
            throw std::out_of_range("weighted_delta_from: constraint index out of range");
        }
        const auto i = static_cast<size_t>(ci);
        const double now = clamped_node_violation(model_.node_value(cids[i]));
        if (now != snapshot[i]) {
            delta += weights[i] * (now - snapshot[i]);
            if (differing != nullptr) {
                differing->push_back(ci);
            }
        }
    }
    return delta;
}

bool ViolationManager::is_feasible(double tol) const {
    const std::vector<int32_t>& cids = model_.constraint_ids();
    // Written as `<= tol` so NaN is rejected (`> tol` would pass it). A domain
    // row is held to 0, as in `verify_model` (#258).
    return std::all_of(cids.begin(), cids.end(), [&](int32_t cid) {
        return model_.node_value(cid) <= (model_.is_domain_node(cid) ? 0.0 : tol);
    });
}

std::vector<int> ViolationManager::violated_constraints(double tol) const {
    std::vector<int> result;
    const auto& cids = model_.constraint_ids();
    for (size_t i = 0; i < cids.size(); ++i) {
        if (!(model_.node_value(cids[i]) <= (model_.is_domain_node(cids[i]) ? 0.0 : tol))) {
            result.push_back(static_cast<int>(i));
        }
    }
    return result;
}

void ViolationManager::bump_weights(double factor) {
    require_row_count();
    for (int i : violated_constraints()) {
        weights[i] += factor;
    }
    cache_valid_ = false;  // weights changed, invalidate
}

double ViolationManager::weighted_violation_delta(int32_t var_id, double j) const {
    require_row_count();  // the probe indexes `weights` by the moved variable's G_v rows
    // Not the hot path: FJ calls `Model::weighted_violation_delta` directly, so
    // only bindings and tests come here and the size check costs nothing.
    return model_.weighted_violation_delta(var_id, j, weights);
}

}  // namespace cbls
