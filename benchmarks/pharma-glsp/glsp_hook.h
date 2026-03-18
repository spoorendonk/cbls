#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace cbls {
namespace glsp {

/// Convert a variable handle (negative) to a var_id (non-negative).
static inline int32_t handle_to_vid(int32_t handle) {
    return -(handle + 1);
}

/// Custom inner solver hook for GLSP-RP.
/// Optimizes FloatVar lot sizes given fixed ListVar sequences using a
/// just-in-time allocation with capacity spilling.
///
/// Strategy:
/// 1. Start with JIT: produce demand[j][t] / serv_frac in each period
/// 2. Fix over-capacity periods by spilling excess to earlier periods
///    (cheapest-to-hold products spilled first)
/// 3. Enforce minimum lot sizes
class GLSPInnerSolverHook : public InnerSolverHook {
public:
    GLSPInnerSolverHook(const GLSPInstance& inst,
                        const std::vector<int32_t>& seq_handles,
                        const std::vector<std::vector<int32_t>>& lot_handles)
        : inst_(inst) {
        for (auto h : seq_handles) seq_vars_.push_back(handle_to_vid(h));
        lot_vars_.resize(lot_handles.size());
        for (size_t j = 0; j < lot_handles.size(); ++j)
            for (auto h : lot_handles[j]) lot_vars_[j].push_back(handle_to_vid(h));
    }

    void solve(Model& model, ViolationManager& vm) override {
        int J = inst_.n_products;
        int T = inst_.n_macro;
        std::vector<int32_t> changed;

        // Compute setup times per macro-period from current sequences
        std::vector<double> setup_times(T, 0.0);
        for (int t = 0; t < T; ++t) {
            const auto& seq = model.var(seq_vars_[t]).elements;
            for (int k = 0; k < J - 1; ++k) {
                if (seq[k] != seq[k + 1])
                    setup_times[t] += inst_.setup_time[seq[k]][seq[k + 1]];
            }
        }

        std::vector<std::vector<double>> new_lots(J, std::vector<double>(T, 0.0));

        // Phase 1: JIT — produce exactly what's needed in each period,
        // accounting for defect rate
        for (int j = 0; j < J; ++j) {
            for (int t = 0; t < T; ++t) {
                double serv_frac = 1.0 - inst_.defect_rate[j][t];
                new_lots[j][t] = (serv_frac > 1e-9)
                    ? inst_.demand[j][t] / serv_frac
                    : inst_.demand[j][t];
            }
        }

        // Phase 2: fix over-capacity by spilling to earlier periods.
        // Spill cheapest-to-hold products first.
        std::vector<int> spill_order(J);
        std::iota(spill_order.begin(), spill_order.end(), 0);
        std::sort(spill_order.begin(), spill_order.end(), [&](int a, int b) {
            return inst_.holding_cost[a] < inst_.holding_cost[b];
        });

        for (int t = T - 1; t > 0; --t) {
            double used_time = setup_times[t];
            for (int j = 0; j < J; ++j)
                used_time += new_lots[j][t] * inst_.process_time[j];
            double excess_time = used_time - inst_.capacity[t];

            if (excess_time <= 1e-9) continue;

            for (int ji = 0; ji < J && excess_time > 1e-9; ++ji) {
                int j = spill_order[ji];
                if (new_lots[j][t] < 1e-9) continue;

                double spill_time = std::min(excess_time,
                    new_lots[j][t] * inst_.process_time[j]);
                double spill_units = spill_time / inst_.process_time[j];

                // Move production to earlier periods (latest first to
                // minimize additional holding)
                for (int s = t - 1; s >= 0 && spill_units > 1e-9; --s) {
                    double s_used = setup_times[s];
                    for (int jj = 0; jj < J; ++jj)
                        s_used += new_lots[jj][s] * inst_.process_time[jj];
                    double spare = std::max(0.0,
                        (inst_.capacity[s] - s_used) / inst_.process_time[j]);

                    // Adjust for different defect rates
                    double sf_t = 1.0 - inst_.defect_rate[j][t];
                    double sf_s = 1.0 - inst_.defect_rate[j][s];
                    double ratio = (sf_s > 1e-9) ? sf_t / sf_s : 1.0;

                    double move = std::min(spill_units, spare / std::max(ratio, 1e-9));
                    new_lots[j][s] += move * ratio;
                    new_lots[j][t] -= move;
                    spill_units -= move;
                    excess_time -= move * inst_.process_time[j];
                }
            }
        }

        // Phase 3: enforce min lot size
        for (int j = 0; j < J; ++j) {
            for (int t = 0; t < T; ++t) {
                if (new_lots[j][t] > 0 && new_lots[j][t] < inst_.min_lot[j]) {
                    if (inst_.demand[j][t] > 1e-9)
                        new_lots[j][t] = inst_.min_lot[j];
                    else
                        new_lots[j][t] = 0.0;
                }
            }
        }

        // Apply new lot sizes
        for (int j = 0; j < J; ++j) {
            for (int t = 0; t < T; ++t) {
                auto& var = model.var_mut(lot_vars_[j][t]);
                double clamped = std::clamp(new_lots[j][t], var.lb, var.ub);
                if (std::abs(var.value - clamped) > 1e-9) {
                    var.value = clamped;
                    changed.push_back(var.id);
                }
            }
        }

        if (!changed.empty()) {
            delta_evaluate(model, changed);
        }
    }

private:
    const GLSPInstance& inst_;
    std::vector<int32_t> seq_vars_;
    std::vector<std::vector<int32_t>> lot_vars_;
};

}  // namespace glsp
}  // namespace cbls
