#include "cbls/search.h"

#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace cbls {

SolveCallback::~SolveCallback() = default;

void initialize_random(Model& model, RNG& rng) {
    for (auto& var : model.variables_mut()) {
        switch (var.type) {
            case VarType::Bool:
                var.value = static_cast<double>(rng.integers(0, 2));
                break;
            case VarType::Int:
                var.value = static_cast<double>(
                    rng.integers(static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
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

// Construction heuristic: Generalised Feasibility Jump (ViolationLS). Refines
// the model's current assignment toward feasibility. Delegates to
// FeasibilityJump; see src/feasibility_jump.cpp.
void fj_nl_initialize(Model& model, ViolationManager& vm, int max_iterations, RNG* rng_ptr,
                      double time_limit) {
    RNG local_rng(42);
    RNG& rng = rng_ptr ? *rng_ptr : local_rng;

    GFJConfig config;
    config.max_iterations = max_iterations;
    config.time_limit = time_limit;
    config.set_initial_x = false;  // refine the current (already-initialised) assignment
    // As an SA warm-start (not the full solver), the two-phase linear-first pass
    // over-commits the linear submodel to cost-pessimal feasibility-boundary
    // values the SA loop cannot recover from; single-phase gives SA a better
    // start. Two-phase stays the default for GFJ-as-solver (P2).
    config.two_phase = false;
    FeasibilityJump fj(model, vm, rng, config);
    fj.run();

    // Hand the SA loop a clean penalty landscape: GLS leaves per-constraint
    // weights skewed, which would distort the SA augmented objective.
    std::fill(vm.weights.begin(), vm.weights.end(), 1.0);
    vm.invalidate_cache();
}

// Update best tracking after hook runs
static void update_best_after_hook(Model& model, ViolationManager& vm, double& best_F,
                                   double& best_feasible_obj, Model::State& best_state) {
    double hook_F = vm.augmented_objective();
    if (vm.is_feasible()) {
        double hook_obj = model.objective_id() >= 0 ? model.node(model.objective_id()).value : 0.0;
        if (hook_obj < best_feasible_obj) {
            best_feasible_obj = hook_obj;
            best_state = model.copy_state();
        }
    }
    if (hook_F < best_F) {
        best_F = hook_F;
        if (!vm.is_feasible()) {
            best_state = model.copy_state();
        }
    }
}

static double initial_temperature(double F) {
    return std::max(std::abs(F) * 0.1, 1.0);
}

static SolveProgress make_progress(int64_t iteration, double elapsed, double best_feasible_obj,
                                   double total_viol, double temperature, bool feasible,
                                   bool new_best, int reheat_count) {
    SolveProgress p;
    p.iteration = iteration;
    p.time_seconds = elapsed;
    p.objective = best_feasible_obj;
    p.total_violation = total_viol;
    p.temperature = temperature;
    p.feasible = feasible;
    p.new_best = new_best;
    p.reheat_count = reheat_count;
    return p;
}

SearchResult solve(Model& model, double time_limit, uint64_t seed, bool use_fj,
                   InnerSolverHook* hook, LNS* lns, int lns_interval, SolveCallback* callback,
                   const SearchConfig& config) {
    RNG rng(seed);
    ViolationManager vm(model);

    auto start = std::chrono::steady_clock::now();
    auto deadline = start + std::chrono::duration<double>(time_limit);

    // Initialize
    if (!config.skip_init) {
        initialize_random(model, rng);
        full_evaluate(model);

        if (use_fj) {
            fj_nl_initialize(model, vm, 5000, &rng, time_limit * config.fj_time_fraction);
        }
    }

    double current_F = vm.augmented_objective();
    double best_F = current_F;
    double best_feasible_obj = std::numeric_limits<double>::infinity();
    auto best_state = model.copy_state();

    if (vm.is_feasible() && model.objective_id() >= 0) {
        best_feasible_obj = model.node(model.objective_id()).value;
    }

    double temperature = initial_temperature(best_F);
    double cooling_rate = config.cooling_rate;
    int reheat_interval = config.reheat_interval;

    MoveProbabilities move_probs({
        "flip",
        "block_on",
        "block_off",
        "int_dec",
        "int_inc",
        "int_rand",
        "float_perturb",
        "list_swap",
        "list_2opt",
        "list_relocate",
        "list_or_opt_2",
        "list_or_opt_3",
        "set_add",
        "set_remove",
        "set_swap",
        "newton_tight",
        "gradient_lift",
    });

    int64_t iteration = 0;
    int reheat_count = 0;
    int64_t discrete_accepts_since_hook = 0;
    const int64_t hook_frequency = config.hook_frequency;

    auto last_callback_time = start;
    constexpr double callback_interval_secs = 1.0;

    const int64_t max_iters = config.max_iterations;
    while ((max_iters > 0 ? iteration < max_iters : true) &&
           std::chrono::steady_clock::now() < deadline) {
        // Select random variable
        int var_idx = static_cast<int>(rng.integers(0, model.num_vars()));
        const auto& var = model.var(var_idx);

        // Generate moves
        auto moves = generate_standard_moves(var, rng);

        // Block moves for vars in sequences
        if (var.type == VarType::Bool) {
            auto bm = generate_block_moves(var.id, model, rng);
            moves.insert(moves.end(), bm.begin(), bm.end());
        }

        // Enriched moves for FloatVar
        if (var.type == VarType::Float) {
            auto violated = vm.violated_constraints();
            if (!violated.empty()) {
                int ci = violated[static_cast<size_t>(rng.integers(0, violated.size()))];
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
        const auto& move = moves[static_cast<size_t>(rng.integers(0, moves.size()))];

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
            double prev_best_feasible = best_feasible_obj;
            if (vm.is_feasible()) {
                double obj_val =
                    model.objective_id() >= 0 ? model.node(model.objective_id()).value : 0.0;
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

            // Fire callback on meaningful feasible objective improvement
            // Always fire for first feasible solution (prev was infinity);
            // otherwise require at least 1e-6 relative change to suppress float noise
            if (callback && obj_improved &&
                (prev_best_feasible == std::numeric_limits<double>::infinity() ||
                 best_feasible_obj == 0.0 ||
                 (prev_best_feasible - best_feasible_obj) / (std::abs(prev_best_feasible) + 1e-30) >
                     1e-6)) {
                auto now = std::chrono::steady_clock::now();
                double elapsed = std::chrono::duration<double>(now - start).count();
                callback->on_progress(make_progress(iteration, elapsed, best_feasible_obj,
                                                    vm.total_violation(), temperature,
                                                    vm.is_feasible(), true, reheat_count));
                last_callback_time = now;
            }

            // Run hook periodically after discrete variable acceptances
            if (hook) {
                bool has_discrete = false;
                for (const auto& ch : move.changes) {
                    auto t = model.var(ch.var_id).type;
                    if (t == VarType::Bool || t == VarType::Int || t == VarType::List) {
                        has_discrete = true;
                        break;
                    }
                }
                if (has_discrete && ++discrete_accepts_since_hook >= hook_frequency) {
                    discrete_accepts_since_hook = 0;
                    hook->solve(model, vm, changed);
                    update_best_after_hook(model, vm, best_F, best_feasible_obj, best_state);
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
            reheat_count++;

            // Run hook on reheat
            if (hook) {
                hook->solve(model, vm);
                update_best_after_hook(model, vm, best_F, best_feasible_obj, best_state);
            }

            // LNS diversification every lns_interval reheats (<=0 disables)
            if (lns && lns_interval > 0 && (reheat_count % lns_interval == 0)) {
                lns->destroy_repair(model, vm, rng);
                update_best_after_hook(model, vm, best_F, best_feasible_obj, best_state);
            }
        }

        // Periodic callback (~1s intervals)
        if (callback && iteration % 1000 == 0) {
            auto now = std::chrono::steady_clock::now();
            double since_last = std::chrono::duration<double>(now - last_callback_time).count();
            if (since_last >= callback_interval_secs) {
                double elapsed = std::chrono::duration<double>(now - start).count();
                callback->on_progress(make_progress(iteration, elapsed, best_feasible_obj,
                                                    vm.total_violation(), temperature,
                                                    vm.is_feasible(), false, reheat_count));
                last_callback_time = now;
            }
        }

        iteration++;
    }

    // Restore best
    model.restore_state(best_state);
    full_evaluate(model);

    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    SearchResult result;
    result.objective =
        best_feasible_obj < std::numeric_limits<double>::infinity() ? best_feasible_obj : best_F;
    result.feasible = best_feasible_obj < std::numeric_limits<double>::infinity();
    result.best_state = best_state;
    result.iterations = iteration;
    result.time_seconds = elapsed;
    return result;
}

}  // namespace cbls
