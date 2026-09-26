#pragma once

#include "dag.h"

#include <cstdint>
#include <set>
#include <vector>

namespace cbls {

double full_evaluate(Model& model);

/// What a `delta_evaluate` call means for a `CustomInvariant` (#166).
///
/// Built-in ops are identical under all three -- they recompute from the
/// assignment in front of them, whatever the caller intends to do next -- so a
/// model with no custom node evaluates the same way it always did whichever of
/// these is passed. Only a custom node reads the mode.
enum class DeltaMode : uint8_t {
    /// The assignment being evaluated is the new committed one. Each custom
    /// node in the dirty cone gets `delta()` then `commit()`. This is what an
    /// applied move, and both legs of an evaluate-forwards-evaluate-back probe,
    /// mean -- and it is the default, so every call site that predates #166
    /// keeps the behaviour it had.
    Commit,
    /// A counterfactual the caller WILL undo. Each custom node in the cone gets
    /// `delta()`, and its previous node value is stashed for the matching
    /// `Rollback` call. No `commit()`.
    Probe,
    /// The caller has put back the assignment the `Probe` was measured from.
    /// Each custom node with a probe pending gets `rollback()` and its stashed
    /// value back, and is NOT re-evaluated; one without a pending probe is
    /// treated as `Commit`, which is the honest reading of a caller that moved
    /// the assignment without probing first.
    ///
    /// Must follow a `Probe` over the same changed-variable set, or the node
    /// values it restores are not the ones the caller thinks they are.
    Rollback,
};

// Primary signature: accepts a contiguous range of var IDs
double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count,
                      DeltaMode mode = DeltaMode::Commit);

// Convenience overloads
inline double delta_evaluate(Model& model, const std::vector<int32_t>& changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit) {
    return delta_evaluate(model, changed_var_ids.data(), changed_var_ids.size(), mode);
}

inline double delta_evaluate(Model& model, const std::set<int32_t>& changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit) {
    std::vector<int32_t> ids(changed_var_ids.begin(), changed_var_ids.end());
    return delta_evaluate(model, ids.data(), ids.size(), mode);
}

inline double delta_evaluate(Model& model, std::initializer_list<int32_t> changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit) {
    return delta_evaluate(model, changed_var_ids.begin(), changed_var_ids.size(), mode);
}

double compute_partial(const Model& model, int32_t expr_id, int32_t var_id);

// Batch AD: compute partials of expr_id w.r.t. ALL variables in one reverse pass.
// Returns vector of size num_vars; entry[i] = ∂expr/∂var_i.
std::vector<double> compute_all_partials(const Model& model, int32_t expr_id);

}  // namespace cbls
