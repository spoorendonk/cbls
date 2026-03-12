#include "cbls/search.h"
#include "cbls/dag_ops.h"
#include <cmath>
#include <chrono>
#include <algorithm>
#include <vector>

namespace cbls {

void initialize_random(Model& model, RNG& rng) {
    for (auto& var : model.variables_mut()) {
        switch (var.type) {
        case VarType::Bool:
            var.value = static_cast<double>(rng.integers(0, 2));
            break;
        case VarType::Int:
            var.value = static_cast<double>(rng.integers(
                static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
            break;
        case VarType::Float:
            var.value = rng.uniform(var.lb, var.ub);
            break;
        case VarType::List:
            var.elements = rng.permutation(var.max_size);
            break;
        case VarType::Set: {
            int size = static_cast<int>(rng.integers(var.min_size, var.max_size + 1));
            auto chosen = rng.choice(var.universe_size, size);
            var.elements = chosen;
            break;
        }
        }
    }
}

// Helper: save a single variable's state
static void save_var(const Variable& var, double& val, std::vector<int32_t>& elems) {
    val = var.value;
    elems = var.elements;
}

static void set_var(Variable& var, double val, const std::vector<int32_t>& elems) {
    if (var.type == VarType::List || var.type == VarType::Set) {
        var.elements = elems;
    } else {
        var.value = val;
    }
}

// FJ candidate values for a variable
static std::vector<std::pair<double, std::vector<int32_t>>> fj_candidate_values(
    const Variable& var, Model& model, ViolationManager& vm, RNG& rng) {

    std::vector<std::pair<double, std::vector<int32_t>>> candidates;

    if (var.type == VarType::Bool) {
        candidates.push_back({1.0 - var.value, {}});
    } else if (var.type == VarType::Int) {
        int domain_size = static_cast<int>(var.ub - var.lb) + 1;
        if (domain_size <= 20) {
            for (int v = static_cast<int>(var.lb); v <= static_cast<int>(var.ub); ++v) {
                if (static_cast<double>(v) != var.value) {
                    candidates.push_back({static_cast<double>(v), {}});
                }
            }
        } else {
            std::set<double> vals;
            if (var.value > var.lb) vals.insert(var.value - 1);
            if (var.value < var.ub) vals.insert(var.value + 1);
            for (int k = 0; k < 8; ++k) {
                vals.insert(static_cast<double>(rng.integers(
                    static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1)));
            }
            vals.erase(var.value);
            for (double v : vals) {
                candidates.push_back({v, {}});
            }
        }
    } else if (var.type == VarType::Float) {
        // Linspace
        for (int k = 0; k < 10; ++k) {
            double v = var.lb + (var.ub - var.lb) * k / 9.0;
            candidates.push_back({v, {}});
        }
        // Gradient-based candidates from violated constraints
        auto violated = vm.violated_constraints();
        int n_check = std::min(static_cast<int>(violated.size()), 3);
        for (int ci = 0; ci < n_check; ++ci) {
            int32_t cid = model.constraint_ids()[violated[ci]];
            double dg = compute_partial(model, cid, var.id);
            if (std::abs(dg) > 1e-12) {
                double step = -model.node(cid).value / dg;
                double new_val = std::clamp(var.value + step, var.lb, var.ub);
                candidates.push_back({new_val, {}});
            }
        }
    }

    return candidates;
}

void fj_nl_initialize(Model& model, ViolationManager& vm,
                       int max_iterations, RNG* rng_ptr) {
    RNG local_rng(42);
    RNG& rng = rng_ptr ? *rng_ptr : local_rng;

    full_evaluate(model);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
        auto violated = vm.violated_constraints();
        if (violated.empty()) break;

        int best_var_id = -1;
        double best_val = 0.0;
        std::vector<int32_t> best_elems;
        double best_reduction = 0.0;

        for (const auto& var : model.variables()) {
            auto candidates = fj_candidate_values(var, model, vm, rng);
            for (const auto& [cand_val, cand_elems] : candidates) {
                // Save
                double old_val;
                std::vector<int32_t> old_elems;
                save_var(var, old_val, old_elems);

                // Apply candidate
                auto& mvar = model.var_mut(var.id);
                if (var.type == VarType::List || var.type == VarType::Set) {
                    mvar.elements = cand_elems;
                } else {
                    mvar.value = cand_val;
                }
                delta_evaluate(model, {var.id});
                double new_viol = vm.total_violation();

                // Restore
                mvar.value = old_val;
                mvar.elements = old_elems;
                delta_evaluate(model, {var.id});

                double old_viol = vm.total_violation();
                double reduction = old_viol - new_viol;
                if (reduction > best_reduction) {
                    best_var_id = var.id;
                    best_val = cand_val;
                    best_elems = cand_elems;
                    best_reduction = reduction;
                }
            }
        }

        if (best_var_id < 0) {
            // Stagnation: bump weights and perturb
            vm.bump_weights();
            if (!model.variables().empty()) {
                int idx = static_cast<int>(rng.integers(0, model.num_vars()));
                auto& var = model.var_mut(idx);
                if (var.type == VarType::Bool || var.type == VarType::Int || var.type == VarType::Float) {
                    auto candidates = fj_candidate_values(var, model, vm, rng);
                    if (!candidates.empty()) {
                        var.value = candidates[0].first;
                        delta_evaluate(model, {var.id});
                    }
                }
            }
            continue;
        }

        auto& mvar = model.var_mut(best_var_id);
        if (mvar.type == VarType::List || mvar.type == VarType::Set) {
            mvar.elements = best_elems;
        } else {
            mvar.value = best_val;
        }
        delta_evaluate(model, {best_var_id});
    }
}

static double initial_temperature(double F) {
    return std::max(std::abs(F) * 0.1, 1.0);
}

SearchResult solve(Model& model, double time_limit, uint64_t seed, bool use_fj) {
    RNG rng(seed);
    ViolationManager vm(model);

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::duration<double>(time_limit);

    // Initialize
    initialize_random(model, rng);
    full_evaluate(model);

    if (use_fj) {
        fj_nl_initialize(model, vm, 5000, &rng);
    }

    double current_F = vm.augmented_objective();
    double best_F = current_F;
    double best_feasible_obj = std::numeric_limits<double>::infinity();
    auto best_state = model.copy_state();

    if (vm.is_feasible() && model.objective_id() >= 0) {
        best_feasible_obj = model.node(model.objective_id()).value;
    }

    double temperature = initial_temperature(best_F);
    double cooling_rate = 0.9999;
    int reheat_interval = 5000;

    MoveProbabilities move_probs({
        "flip", "int_dec", "int_inc", "int_rand",
        "float_perturb", "list_swap", "list_2opt",
        "set_add", "set_remove", "set_swap",
        "newton_tight", "gradient_lift",
    });

    int64_t iteration = 0;
    while (std::chrono::steady_clock::now() < deadline) {
        // Select random variable
        int var_idx = static_cast<int>(rng.integers(0, model.num_vars()));
        const auto& var = model.var(var_idx);

        // Generate moves
        auto moves = generate_standard_moves(var, rng);

        // Enriched moves for FloatVar
        if (var.type == VarType::Float) {
            auto violated = vm.violated_constraints();
            if (!violated.empty()) {
                int ci = violated[rng.integers(0, violated.size())];
                auto nm = newton_tight_move(var.id, model, ci);
                moves.insert(moves.end(), nm.begin(), nm.end());
            }
            auto gm = gradient_lift_move(var.id, model);
            moves.insert(moves.end(), gm.begin(), gm.end());
        }

        if (moves.empty()) {
            iteration++;
            continue;
        }

        // Pick a move uniformly
        const auto& move = moves[rng.integers(0, moves.size())];

        // Evaluate via delta
        auto saved = save_move_values(model, move);
        double old_F = vm.augmented_objective();
        auto changed = apply_move(model, move);
        delta_evaluate(model, changed);
        double new_F = vm.augmented_objective();
        double delta_F = new_F - old_F;

        // SA acceptance
        bool accept = false;
        if (delta_F <= 0) {
            accept = true;
        } else if (temperature > 1e-15) {
            double p = std::exp(-delta_F / temperature);
            accept = rng.random() < p;
        }

        if (accept) {
            bool obj_improved = false;
            if (vm.is_feasible()) {
                double obj_val = model.objective_id() >= 0
                    ? model.node(model.objective_id()).value : 0.0;
                if (obj_val < best_feasible_obj) {
                    best_feasible_obj = obj_val;
                    best_state = model.copy_state();
                    obj_improved = true;
                }
            }

            if (new_F < best_F) {
                best_F = new_F;
                if (!vm.is_feasible()) {
                    best_state = model.copy_state();
                }
            }

            move_probs.update(move.move_type, true);
            vm.adaptive_lambda.update(vm.is_feasible(), obj_improved);
        } else {
            undo_move(model, move, saved);
            delta_evaluate(model, changed);
            move_probs.update(move.move_type, false);
        }

        temperature *= cooling_rate;
        if (iteration > 0 && iteration % reheat_interval == 0) {
            temperature = initial_temperature(best_F) * 0.5;
        }

        iteration++;
    }

    // Restore best
    model.restore_state(best_state);
    full_evaluate(model);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    SearchResult result;
    result.objective = best_feasible_obj < std::numeric_limits<double>::infinity()
        ? best_feasible_obj : best_F;
    result.feasible = best_feasible_obj < std::numeric_limits<double>::infinity();
    result.best_state = best_state;
    result.iterations = iteration;
    result.time_seconds = elapsed;
    return result;
}

}  // namespace cbls
