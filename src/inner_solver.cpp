#include "cbls/inner_solver.h"
#include "cbls/dag_ops.h"
#include <cmath>
#include <algorithm>

namespace cbls {

void FloatIntensifyHook::solve(Model& model, ViolationManager& vm) {
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        bool improved = false;

        for (const auto& var : model.variables()) {
            if (var.type != VarType::Float) continue;

            double old_val = var.value;
            double best_val = old_val;
            double best_aug = vm.augmented_objective();

            // Try Newton steps on violated constraints
            auto violated = vm.violated_constraints();
            int n_check = std::min(static_cast<int>(violated.size()), 3);
            for (int ci = 0; ci < n_check; ++ci) {
                int32_t cid = model.constraint_ids()[violated[ci]];
                double g = model.node(cid).value;
                double dg = compute_partial(model, cid, var.id);
                if (std::abs(dg) > 1e-12) {
                    double candidate = std::clamp(old_val + (-g / dg), var.lb, var.ub);
                    if (std::abs(candidate - old_val) > 1e-15) {
                        model.var_mut(var.id).value = candidate;
                        delta_evaluate(model, {var.id});
                        double new_aug = vm.augmented_objective();
                        if (new_aug < best_aug) {
                            best_val = candidate;
                            best_aug = new_aug;
                        }
                        model.var_mut(var.id).value = old_val;
                        delta_evaluate(model, {var.id});
                    }
                }
            }

            // Try gradient step on objective
            if (model.objective_id() >= 0) {
                double df = compute_partial(model, model.objective_id(), var.id);
                if (std::abs(df) > 1e-12) {
                    double candidate = std::clamp(old_val - step_size * df, var.lb, var.ub);
                    if (std::abs(candidate - old_val) > 1e-15) {
                        model.var_mut(var.id).value = candidate;
                        delta_evaluate(model, {var.id});
                        double new_aug = vm.augmented_objective();
                        if (new_aug < best_aug) {
                            best_val = candidate;
                            best_aug = new_aug;
                        }
                        model.var_mut(var.id).value = old_val;
                        delta_evaluate(model, {var.id});
                    }
                }
            }

            if (std::abs(best_val - old_val) > 1e-15) {
                model.var_mut(var.id).value = best_val;
                delta_evaluate(model, {var.id});
                improved = true;
            }
        }

        if (!improved) break;
    }
}

}  // namespace cbls
