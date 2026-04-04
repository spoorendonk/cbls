#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include "bunker_eca_model.h"
#include <cmath>
#include <algorithm>

namespace cbls {
namespace bunker_eca {

// Inner solver hook: given fixed cargo assignments, optimize speeds analytically.
// Optimal speed = max(v_min, dist / (24 * available_time)) — minimum feasible speed
// minimizes quadratic fuel cost.
class BunkerSpeedHook : public InnerSolverHook {
public:
    FloatIntensifyHook float_hook;

    void solve(Model& model, ViolationManager& vm,
               const std::vector<int32_t>& last_changed_vars = {}) override {
        if (!inst_) {
            float_hook.solve(model, vm, last_changed_vars);
            return;
        }

        int C = (int)inst_->cargoes.size();
        std::vector<int32_t> changed;

        for (int c = 0; c < C; ++c) {
            int32_t speed_vid = handle_to_var_id(bec_model_->speed[c]);
            auto& speed_var = model.var_mut(speed_vid);

            int32_t assign_vid = handle_to_var_id(bec_model_->assign[c]);
            int assign_val = static_cast<int>(
                std::round(model.var(assign_vid).value));

            if (assign_val == 0) {
                // Unassigned cargo: set minimum speed
                if (std::abs(speed_var.value - speed_var.lb) > 1e-6) {
                    speed_var.value = speed_var.lb;
                    changed.push_back(speed_vid);
                }
                continue;
            }

            double dist = inst_->leg_distance(
                inst_->cargoes[c].pickup_region,
                inst_->cargoes[c].delivery_region);

            double available = inst_->cargoes[c].delivery_tw_end
                             - inst_->cargoes[c].pickup_tw_start
                             - inst_->cargoes[c].service_time_load
                             - inst_->cargoes[c].service_time_discharge;

            double v_opt;
            if (available > 0.0 && dist > 0.0) {
                v_opt = dist / (24.0 * available);
                v_opt = std::max(v_opt, (double)speed_var.lb);
                v_opt = std::min(v_opt, (double)speed_var.ub);
            } else {
                v_opt = speed_var.lb;
            }

            if (std::abs(speed_var.value - v_opt) > 1e-6) {
                speed_var.value = v_opt;
                changed.push_back(speed_vid);
            }
        }

        if (!changed.empty()) {
            delta_evaluate(model, changed);
        }

        float_hook.solve(model, vm, last_changed_vars);
    }

    void set_model(const BunkerECAModel* bec_model, const Instance* inst) {
        bec_model_ = bec_model;
        inst_ = inst;
    }

private:
    const BunkerECAModel* bec_model_ = nullptr;
    const Instance* inst_ = nullptr;
};

}  // namespace bunker_eca
}  // namespace cbls
