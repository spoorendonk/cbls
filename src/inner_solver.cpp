#include "cbls/inner_solver.h"
#include "cbls/dag_ops.h"
#include <cmath>
#include <algorithm>
#include <set>
#include <vector>

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

            // Backtracking line search on objective
            if (model.objective_id() >= 0) {
                double df = compute_partial(model, model.objective_id(), var.id);
                if (std::abs(df) > 1e-12) {
                    double step = initial_step_size;
                    double prev_candidate = old_val;
                    for (int ls = 0; ls < max_line_search_steps; ++ls) {
                        double candidate = std::clamp(old_val - step * df, var.lb, var.ub);
                        if (std::abs(candidate - old_val) > 1e-15 &&
                            std::abs(candidate - prev_candidate) > 1e-15) {
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
                        prev_candidate = candidate;
                        step *= 0.5;
                    }
                }
            }

            if (std::abs(best_val - old_val) > 1e-15) {
                model.var_mut(var.id).value = best_val;
                delta_evaluate(model, {var.id});
                improved = true;
            }
        }

        // Multi-var Newton: minimum-norm step on violated constraints
        {
            auto violated = vm.violated_constraints();
            int n_mv = std::min(static_cast<int>(violated.size()), max_multi_var_constraints);
            for (int ci = 0; ci < n_mv; ++ci) {
                int32_t cid = model.constraint_ids()[violated[ci]];
                double g = model.node(cid).value;
                if (std::abs(g) < 1e-15) continue;

                // Collect float vars with non-trivial gradient
                struct VarGrad { int32_t id; double dg; double old_val; };
                std::vector<VarGrad> grads;
                for (const auto& v : model.variables()) {
                    if (v.type != VarType::Float) continue;
                    double dg = compute_partial(model, cid, v.id);
                    if (std::abs(dg) > 1e-12) {
                        grads.push_back({v.id, dg, v.value});
                    }
                }
                if (static_cast<int>(grads.size()) < 2) continue;

                double grad_norm_sq = 0.0;
                for (const auto& vg : grads) grad_norm_sq += vg.dg * vg.dg;
                double scale = -g / grad_norm_sq;

                // Capture baseline before applying step
                double old_aug = vm.augmented_objective();

                // Apply minimum-norm Newton step
                std::set<int32_t> changed_ids;
                for (const auto& vg : grads) {
                    const auto& v = model.var(vg.id);
                    double new_val = std::clamp(vg.old_val + scale * vg.dg, v.lb, v.ub);
                    model.var_mut(vg.id).value = new_val;
                    changed_ids.insert(vg.id);
                }
                delta_evaluate(model, changed_ids);
                double new_aug = vm.augmented_objective();

                if (new_aug < old_aug) {
                    improved = true;
                } else {
                    // Restore all vars
                    for (const auto& vg : grads) {
                        model.var_mut(vg.id).value = vg.old_val;
                    }
                    delta_evaluate(model, changed_ids);
                }
            }
        }

        if (!improved) break;
    }
}

}  // namespace cbls
