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

/// Convert a var handle (negative, returned by int_var/float_var/etc.)
/// to a var ID (non-negative, used by model.var()/model.var_mut()).
inline int32_t handle_to_var_id(int32_t handle) {
    return -(handle + 1);
}

/// InnerSolverHook that evaluates production dispatch cost externally.
/// Reads outage start values from the model, runs merit-order dispatch
/// across demand scenarios, and writes the expected cost into the objective node.
///
/// Uses DispatchEvaluator for allocation-free evaluation (5a/5b/5c/5e).
class NuclearDispatchHook : public InnerSolverHook {
public:
    const NuclearInstance& inst;
    const NuclearModel& nm;

    // Scenario sampling: evaluate a subset of scenarios per move for speed.
    // Rotates through scenario windows to avoid bias toward early scenarios.
    int scenarios_per_move = 50;

    NuclearDispatchHook(const NuclearInstance& inst_, const NuclearModel& nm_)
        : inst(inst_), nm(nm_),
          evaluator_(inst_),
          starts_(inst_.n_outages, 0),
          prev_starts_(inst_.n_outages, -1) {}

    void solve(Model& model, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/ = {}) override {
        // 1. Read current outage start values (convert var handles to var IDs)
        for (int o = 0; o < inst.n_outages; ++o) {
            int32_t vid = handle_to_var_id(nm.s[o]);
            starts_[o] = static_cast<int>(std::round(model.var(vid).value));
        }

        // 2. Compute expected dispatch cost with rotating scenario window
        int n_sc = std::min(scenarios_per_move, inst.n_scenarios);
        int offset = scenario_offset_ % std::max(1, inst.n_scenarios - n_sc + 1);
        double cost = evaluator_.expected_cost(starts_, n_sc, offset);
        scenario_offset_ += n_sc;

        // 3. Add resource violation penalty
        // 5e: detect which outage changed for delta penalty
        int changed = find_changed_outage();
        cost += evaluator_.resource_penalty(starts_, 1e6, changed);

        // 4. Write cost into the objective constant node
        model.node_mut(nm.objective_node).value = cost;
        model.node_mut(nm.objective_node).const_value = cost;

        // Save starts for next delta comparison
        prev_starts_ = starts_;
    }

private:
    DispatchEvaluator evaluator_;
    std::vector<int> starts_;
    std::vector<int> prev_starts_;
    int scenario_offset_ = 0;

    /// Find which outage changed (returns -1 if none or multiple changed).
    int find_changed_outage() const {
        int changed = -1;
        for (int o = 0; o < inst.n_outages; ++o) {
            if (starts_[o] != prev_starts_[o]) {
                if (changed >= 0) return -1;  // multiple changed → full recompute
                changed = o;
            }
        }
        return changed;
    }
};

}  // namespace nuclear_outage
}  // namespace cbls
