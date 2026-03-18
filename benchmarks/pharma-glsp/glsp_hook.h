#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <set>

namespace cbls {
namespace glsp {

/// Convert a variable handle (negative) to a var_id (non-negative).
static inline int32_t handle_to_vid(int32_t handle) {
    return -(handle + 1);
}

/// Custom inner solver hook for GLSP-RP.
/// Optimizes FloatVar lot sizes given fixed ListVar sequences using a
/// backward-pass demand-driven heuristic.
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
        std::set<int32_t> changed;

        // Compute setup times per macro-period from current sequences
        std::vector<double> setup_times(T, 0.0);
        for (int t = 0; t < T; ++t) {
            const auto& seq = model.var(seq_vars_[t]).elements;
            for (int k = 0; k < J - 1; ++k) {
                if (seq[k] != seq[k + 1])
                    setup_times[t] += inst_.setup_time[seq[k]][seq[k + 1]];
            }
        }

        // Optimize lot sizes — backward-pass demand-driven heuristic
        std::vector<std::vector<double>> new_lots(J, std::vector<double>(T, 0.0));

        for (int j = 0; j < J; ++j) {
            double remaining = 0.0;
            for (int t = 0; t < T; ++t)
                remaining += inst_.demand[j][t];

            // Backward pass: place production in latest periods first
            for (int t = T - 1; t >= 0 && remaining > 1e-6; --t) {
                double avail = inst_.capacity[t] - setup_times[t];
                for (int jj = 0; jj < J; ++jj) {
                    if (jj != j) avail -= new_lots[jj][t] * inst_.process_time[jj];
                }
                double max_prod = std::max(0.0, avail / inst_.process_time[j]);

                double lot = std::min(remaining, max_prod);
                if (lot > 0 && lot < inst_.min_lot[j])
                    lot = std::min(inst_.min_lot[j], max_prod);
                if (lot < 0) lot = 0;

                new_lots[j][t] = lot;
                remaining -= lot;
            }

            // Forward pass for remaining demand
            for (int t = 0; t < T && remaining > 1e-6; ++t) {
                double avail = inst_.capacity[t] - setup_times[t];
                for (int jj = 0; jj < J; ++jj)
                    avail -= new_lots[jj][t] * inst_.process_time[jj];
                double additional = std::min(remaining, std::max(0.0, avail / inst_.process_time[j]));
                new_lots[j][t] += additional;
                remaining -= additional;
            }
        }

        // Apply new lot sizes
        for (int j = 0; j < J; ++j) {
            for (int t = 0; t < T; ++t) {
                auto& var = model.var_mut(lot_vars_[j][t]);
                double clamped = std::clamp(new_lots[j][t], var.lb, var.ub);
                if (std::abs(var.value - clamped) > 1e-9) {
                    var.value = clamped;
                    changed.insert(var.id);
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
