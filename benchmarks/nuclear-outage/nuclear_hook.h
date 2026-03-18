#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include "dispatch.h"
#include "nuclear_model.h"
#include <vector>
#include <algorithm>

namespace cbls {
namespace nuclear_outage {

/// Convert a var handle (negative, returned by int_var/float_var/etc.)
/// to a var ID (non-negative, used by model.var()/model.var_mut()).
inline int32_t handle_to_var_id(int32_t handle) {
    return -(handle + 1);
}

/// InnerSolverHook that evaluates production dispatch cost externally.
/// Reads outage start values from the model, runs merit-order dispatch
/// across demand scenarios, and writes the expected cost into the objective node.
class NuclearDispatchHook : public InnerSolverHook {
public:
    const NuclearInstance& inst;
    const NuclearModel& nm;

    // Scenario sampling: evaluate a subset of scenarios per move for speed.
    int scenarios_per_move = 50;

    NuclearDispatchHook(const NuclearInstance& inst_, const NuclearModel& nm_)
        : inst(inst_), nm(nm_) {}

    void solve(Model& model, ViolationManager& /*vm*/) override {
        // 1. Read current outage start values (convert var handles to var IDs)
        std::vector<int> starts(inst.n_outages);
        for (int o = 0; o < inst.n_outages; ++o) {
            int32_t vid = handle_to_var_id(nm.s[o]);
            starts[o] = static_cast<int>(model.var(vid).value);
        }

        // 2. Compute expected dispatch cost
        int n_sc = std::min(scenarios_per_move, inst.n_scenarios);
        double cost = expected_cost(inst, starts, n_sc);

        // 3. Add resource violation penalty
        cost += resource_violation_penalty(inst, starts);

        // 4. Write cost into the objective constant node
        // Update both value and const_value so full_evaluate won't reset it
        model.node_mut(nm.objective_node).value = cost;
        model.node_mut(nm.objective_node).const_value = cost;
    }
};

}  // namespace nuclear_outage
}  // namespace cbls
