#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace cbls {

// Non-convex objectives/constraints (exp/pow/div blowups — the MINLPLib target)
// can drive a node value to +inf or NaN. Such a value would poison the
// total_violation cache, the structural pass's move comparison and the
// best-objective bookkeeping. Map every non-finite (or absurdly large) violation
// to a large but finite penalty so the search treats the point as very bad but
// still well-ordered. The clamp is one-sided on the violation (which is already
// max(0, .)), so a finite-but-huge value and a +inf value both become kInfPenalty.
namespace {

double clamped_node_violation(double node_value) {
    // NaN must be handled before max(): std::max(0.0, NaN) returns 0.0, which
    // would silently mask a NaN constraint as satisfied. NaN (e.g. inf-inf,
    // 0*inf) is treated as a maximal violation — we have no evidence it holds.
    if (std::isnan(node_value)) {
        return kInfPenalty;
    }
    double v = std::max(0.0, node_value);
    if (v > kInfPenalty) {  // also catches +inf
        return kInfPenalty;
    }
    return v;
}
}  // namespace

ViolationManager::ViolationManager(Model& model) : model_(model) {
    weights.resize(model.constraint_ids().size(), 1.0);
    cached_violations_.resize(model.constraint_ids().size(), 0.0);
}

double ViolationManager::constraint_violation(int i) const {
    if (i < 0 || i >= static_cast<int>(model_.constraint_ids().size())) {
        throw std::out_of_range("constraint index out of range");
    }
    int32_t cid = model_.constraint_ids()[i];
    return clamped_node_violation(model_.node(cid).value);
}

void ViolationManager::recompute_cache() const {
    const auto& cids = model_.constraint_ids();
    cached_total_ = 0.0;
    for (size_t i = 0; i < cids.size(); ++i) {
        cached_violations_[i] = clamped_node_violation(model_.node(cids[i]).value);
        cached_total_ += cached_violations_[i] * weights[i];
    }
    cache_valid_ = true;
    incremental_updates_ = 0;
}

double ViolationManager::total_violation() const {
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
        double new_viol = clamped_node_violation(model_.node(cids[i]).value);
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
        obj = model_.node(model_.objective_id()).value;
        if (!std::isfinite(obj)) {
            obj = kInfPenalty;  // non-convex blowup: keep the metric ordered
        }
    }
    return obj + total_violation();
}

void ViolationManager::snapshot_violations(std::vector<double>& out) const {
    // total_violation() brings cached_violations_ up to the current node values
    // (it diffs every constraint against the cache on each call), so the snapshot
    // is a copy of work that has to happen anyway rather than a second scan.
    total_violation();
    out = cached_violations_;
}

double ViolationManager::weighted_delta_from(const std::vector<double>& snapshot) const {
    const auto& cids = model_.constraint_ids();
    if (snapshot.size() != cids.size()) {
        throw std::invalid_argument("weighted_delta_from: snapshot size != constraint count");
    }
    // Per-constraint differencing, for the reason spelled out on
    // Model::weighted_violation_delta: `1e30 + 3` rounds back to `1e30`, so
    // subtracting two whole sums loses every real row the moment one row is
    // clamped, whereas `1e30 - 1e30` cancels to exactly 0 here.
    //
    // Deliberately does not update the cache: the caller's candidate move is
    // usually rolled back, and total_violation() self-corrects against whatever
    // the node values are when it is next called either way.
    double delta = 0.0;
    for (size_t i = 0; i < cids.size(); ++i) {
        const double now = clamped_node_violation(model_.node(cids[i]).value);
        if (now != snapshot[i]) {
            delta += weights[i] * (now - snapshot[i]);
        }
    }
    return delta;
}

bool ViolationManager::is_feasible(double tol) const {
    for (int32_t cid : model_.constraint_ids()) {
        // A NaN node value is not <= tol; treat it as infeasible (the `!(<=)`
        // form catches NaN, which a bare `> tol` would silently pass).
        if (!(model_.node(cid).value <= tol)) {
            return false;
        }
    }
    return true;
}

std::vector<int> ViolationManager::violated_constraints(double tol) const {
    std::vector<int> result;
    const auto& cids = model_.constraint_ids();
    for (size_t i = 0; i < cids.size(); ++i) {
        if (!(model_.node(cids[i]).value <= tol)) {
            result.push_back(static_cast<int>(i));
        }
    }
    return result;
}

void ViolationManager::bump_weights(double factor) {
    for (int i : violated_constraints()) {
        weights[i] += factor;
    }
    cache_valid_ = false;  // weights changed, invalidate
}

double ViolationManager::weighted_violation_delta(int32_t var_id, double j) const {
    // Delegate to the allocation-free Model probe (hot path: one call per jump
    // candidate). weights is the per-constraint GLS weight vector.
    return model_.weighted_violation_delta(var_id, j, weights);
}

}  // namespace cbls
