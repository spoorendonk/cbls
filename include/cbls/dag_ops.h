#pragma once

#include "dag.h"
#include <vector>
#include <set>

namespace cbls {

void topological_sort(Model& model);
double full_evaluate(Model& model);
double delta_evaluate(Model& model, const std::set<int32_t>& changed_var_ids);
double compute_partial(const Model& model, int32_t expr_id, int32_t var_id);

}  // namespace cbls
