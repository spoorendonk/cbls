#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include "dispatch.h"
#include "nuclear_model.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace cbls {
namespace nuclear_outage {

#ifndef CBLS_HANDLE_TO_VAR_ID_DEFINED
#define CBLS_HANDLE_TO_VAR_ID_DEFINED
/// Convert a var handle (negative, returned by int_var/float_var/etc.)
/// to a var ID (non-negative, used by model.var()/model.var_mut()).
inline int32_t handle_to_var_id(int32_t handle) {
    return -(handle + 1);
}
#endif

/// InnerSolverHook that evaluates production dispatch cost externally.
/// Reads outage start values from the model, runs merit-order dispatch
/// across demand scenarios, and writes the expected cost into the objective node.
///
/// Uses DispatchEvaluator for allocation-free evaluation (5a/5b/5c/5e)
/// and incremental dispatch (5d) with batched scenario rotation.
class NuclearDispatchHook : public InnerSolverHook {
public:
    const NuclearInstance& inst;
    const NuclearModel& nm;

    // Scenario sampling: evaluate a subset of scenarios per move for speed.
    int scenarios_per_move = 50;

    // 5d: how many hook calls to keep the same scenario window before rotating.
    // Within an epoch, incremental dispatch can reuse cached per-period costs.
    int epoch_size = 10;

    NuclearDispatchHook(const NuclearInstance& inst_, const NuclearModel& nm_)
        : inst(inst_), nm(nm_),
          evaluator_(inst_),
          starts_(inst_.n_outages, 0) {}

    void solve(Model& model, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/ = {}) override {
        // 1. Read current outage start values
        for (int o = 0; o < inst.n_outages; ++o) {
            int32_t vid = handle_to_var_id(nm.s[o]);
            starts_[o] = static_cast<int>(std::round(model.var(vid).value));
        }

        // 2. Determine scenario window (batched rotation for 5d)
        int n_sc = std::min(scenarios_per_move, inst.n_scenarios);
        if (++calls_in_epoch_ >= epoch_size) {
            calls_in_epoch_ = 0;
            scenario_offset_ += n_sc;
        }
        int offset = scenario_offset_ % std::max(1, inst.n_scenarios - n_sc + 1);

        // 3. Detect which outages changed vs the evaluator's cached state
        int n_changed = evaluator_.find_changes(starts_, changed_outages_);

        // 4. Compute dispatch cost (incremental if changes detected + same window)
        double cost = evaluator_.expected_cost(starts_, n_sc, offset, changed_outages_);

        // 5. Add resource violation penalty (delta for single change, full otherwise)
        int penalty_changed = (n_changed == 1) ? changed_outages_[0] : -1;
        cost += evaluator_.resource_penalty(starts_, 1e6, penalty_changed);

        // 6. Write cost into the objective constant node
        model.node_mut(nm.objective_node).value = cost;
        model.node_mut(nm.objective_node).const_value = cost;
    }

private:
    DispatchEvaluator evaluator_;
    std::vector<int> starts_;
    std::vector<int> changed_outages_;
    int scenario_offset_ = 0;
    int calls_in_epoch_ = 0;
};

}  // namespace nuclear_outage
}  // namespace cbls
