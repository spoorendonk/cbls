#pragma once

#include "dag.h"
#include <vector>
#include <set>

namespace cbls {

double full_evaluate(Model& model);

// Primary signature: accepts a contiguous range of var IDs
double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count);

// Convenience overloads
inline double delta_evaluate(Model& model, const std::vector<int32_t>& changed_var_ids) {
    return delta_evaluate(model, changed_var_ids.data(), changed_var_ids.size());
}

inline double delta_evaluate(Model& model, const std::set<int32_t>& changed_var_ids) {
    std::vector<int32_t> ids(changed_var_ids.begin(), changed_var_ids.end());
    return delta_evaluate(model, ids.data(), ids.size());
}

inline double delta_evaluate(Model& model, std::initializer_list<int32_t> changed_var_ids) {
    return delta_evaluate(model, changed_var_ids.begin(), changed_var_ids.size());
}

double compute_partial(const Model& model, int32_t expr_id, int32_t var_id);

// Batch AD: compute partials of expr_id w.r.t. ALL variables in one reverse pass.
// Returns vector of size num_vars; entry[i] = ∂expr/∂var_i.
std::vector<double> compute_all_partials(const Model& model, int32_t expr_id);

}  // namespace cbls
