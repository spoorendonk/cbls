#include "cbls/search.h"

#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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
    // As an SA/LNS warm-start (not the full solver), the two-phase linear-first
    // pass over-commits the linear submodel to cost-pessimal feasibility-boundary
    // values; single-phase gives a better start. Two-phase stays the default for
    // GFJ-as-solver.
    config.two_phase = false;
    FeasibilityJump fj(model, vm, rng, config);
    fj.run();

    // Hand the caller a clean penalty landscape: GLS leaves per-constraint
    // weights skewed.
    std::fill(vm.weights.begin(), vm.weights.end(), 1.0);
    vm.invalidate_cache();
}

// Greedy structural pass over List/Set variables: try the candidate structural
// moves (swap / 2-opt / relocate / set add-remove-swap) for each and keep any
// that reduce total weighted violation. FeasibilityJump only jumps scalar
// variables, so without this list-structured models cannot improve their
// list/set assignment beyond the random initial one. Minimal by design; P4
// (#69) promotes this to a first-class structural batch. Returns true if any
// move was committed.
static bool structural_pass(Model& model, ViolationManager& vm, RNG& rng) {
    bool changed = false;
    for (const auto& var : model.variables()) {
        if (var.type != VarType::List && var.type != VarType::Set) {
            continue;
        }
        auto moves = generate_standard_moves(var, rng);
        // total_violation()'s incremental path self-corrects to the current node
        // state on each call (it diffs all constraints against its cache), so no
        // explicit invalidate is needed across the apply/undo dance; thread the
        // accepted baseline instead of recomputing it per move.
        double before = vm.total_violation();
        for (const auto& move : moves) {
            auto saved = save_move_values(model, move);
            auto touched = apply_move(model, move);
            delta_evaluate(model, touched);
            double after = vm.total_violation();
            if (after < before - 1e-12) {
                changed = true;  // improving: keep
                before = after;
            } else {
                undo_move(model, move, saved);
                delta_evaluate(model, touched);
            }
        }
    }
    return changed;
}

static SolveProgress make_progress(int64_t iteration, double elapsed, double best_feasible_obj,
                                   double total_viol, bool feasible, bool new_best,
                                   int perturbations) {
    SolveProgress p;
    p.iteration = iteration;
    p.time_seconds = elapsed;
    p.objective = best_feasible_obj;
    p.total_violation = total_viol;
    p.feasible = feasible;
    p.new_best = new_best;
    p.perturbations = perturbations;
    return p;
}

// ViolationLS (paper Algorithm 6): the objective is folded into the constraint
// set as `obj <= bound`; GFJ batches drive the assignment to feasibility while
// the bound is tightened on each new (real-)feasible solution. On stagnation the
// assignment is perturbed (or diversified via LNS). The InnerSolverHook polishes
// continuous variables (objective descent) on each feasible solution.
SearchResult solve(Model& model, double time_limit, uint64_t seed, bool use_fj,
                   InnerSolverHook* hook, LNS* lns, int lns_interval, SolveCallback* callback,
                   const SearchConfig& config) {
    (void)use_fj;  // GFJ is always the engine now; the flag is vestigial.
    RNG rng(seed);

    const bool has_obj = model.objective_id() >= 0;
    if (has_obj) {
        if (!model.has_objective_constraint()) {
            model.add_objective_soft_constraint();
        }
        model.set_objective_bound(std::numeric_limits<double>::infinity());  // reset for re-solves
    }
    ViolationManager vm(model);

    const auto start = std::chrono::steady_clock::now();
    const auto deadline = start + std::chrono::duration<double>(time_limit);

    if (!config.skip_init) {
        initialize_random(model, rng);
    }

    GFJConfig gfj;
    gfj.two_phase = false;  // batches reuse weights across calls; single-phase GLS
    gfj.time_limit = 0.0;   // the outer loop owns the wall clock
    gfj.max_iterations = 0;
    FeasibilityJump fj(model, vm, rng, gfj);
    fj.begin(/*set_initial_x=*/!config.skip_init);

    const int32_t obj_ci = model.objective_constraint_idx();
    const auto& cids = model.constraint_ids();

    // Real feasibility = every constraint except the artificial objective one.
    auto real_feasible = [&]() {
        for (size_t i = 0; i < cids.size(); ++i) {
            if (static_cast<int32_t>(i) == obj_ci) {
                continue;
            }
            if (model.node(cids[i]).value > 1e-9) {
                return false;
            }
        }
        return true;
    };
    auto current_obj = [&]() { return has_obj ? model.node(model.objective_id()).value : 0.0; };

    double best_feasible_obj = std::numeric_limits<double>::infinity();
    Model::State best_state = model.copy_state();
    bool have_feasible = false;
    int perturbations = 0;
    int stagnation = 0;
    int64_t batches = 0;
    auto last_callback = start;

    // Skip the structural pass entirely on scalar-only models.
    const bool has_structural = std::any_of(
        model.variables().begin(), model.variables().end(),
        [](const Variable& v) { return v.type == VarType::List || v.type == VarType::Set; });

    auto sample_rho = [&]() { fj.set_rho(rng.random() < 0.5 ? 0.95 : 1.0); };
    sample_rho();

    auto emit_progress = [&](bool new_best) {
        if (!callback) {
            return;
        }
        vm.invalidate_cache();
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        callback->on_progress(make_progress(batches, elapsed, best_feasible_obj,
                                            vm.total_violation(), real_feasible(), new_best,
                                            perturbations));
        last_callback = std::chrono::steady_clock::now();
    };

    // Record the current (real-feasible) assignment if it improves the best and
    // tighten the objective bound. Returns true on a new best.
    auto record_best = [&]() -> bool {
        double obj = current_obj();
        if (have_feasible &&
            obj >= best_feasible_obj - 1e-12 * (std::abs(best_feasible_obj) + 1.0)) {
            return false;
        }
        have_feasible = true;
        best_feasible_obj = obj;
        best_state = model.copy_state();
        if (has_obj) {
            // The bound step doubles as the Newton step size toward the objective
            // (the float jump chases obj <= bound), so it must be non-trivial for
            // hook-less continuous descent.
            double eps = 1e-3 * (std::abs(obj) + 1.0);
            model.set_objective_bound(obj - eps);
        }
        emit_progress(/*new_best=*/true);
        return true;
    };

    // On stagnation: LNS diversification every lns_interval-th time, else perturb.
    auto diversify = [&]() {
        if (lns && lns_interval > 0 && (perturbations % lns_interval == lns_interval - 1)) {
            lns->destroy_repair(model, vm, rng);
            fj.reset_weights();  // LNS mutated state outside GFJ
        } else {
            fj.perturb(config.perturbation_probability);  // self-resyncs
        }
        sample_rho();
        ++perturbations;
        stagnation = 0;
    };

    while (std::chrono::steady_clock::now() < deadline) {
        if (config.max_iterations > 0 &&
            batches * config.batch_iterations >= config.max_iterations) {
            break;
        }

        // Each batch is Feasibility Jump or Novelty Jump (paper Algorithm 6,
        // ~50/50). NJ commits compound moves outside the FJ scan-set/jump-table,
        // so it must be followed by a resync.
        bool resync = false;
        if (config.use_compound_moves && rng.random() < config.novelty_jump_probability) {
            fj.apply_novelty_jump();
            resync = true;
        } else {
            fj.batch(config.batch_iterations);
        }
        ++batches;

        resync = (has_structural && structural_pass(model, vm, rng)) || resync;

        bool improved = false;
        if (real_feasible()) {
            if (hook) {
                hook->solve(model, vm, {});  // continuous-objective polish (mutates floats)
                resync = true;
            }
            if (!hook || real_feasible()) {  // hook may have moved off the feasible region
                improved = record_best();
            }
        }

        if (improved) {
            stagnation = 0;
            fj.reset_weights();  // new best: fresh GLS weights (paper) + new rho
            sample_rho();
            if (!has_obj) {
                break;  // pure feasibility: first solution is the answer
            }
        } else {
            ++stagnation;
            if (resync) {
                fj.resync();  // re-sync after hook/structural mutation, keep GLS weights
            }
        }

        if (stagnation >= config.perturbation_period) {
            diversify();
        }

        // Periodic progress (~1s) even without improvement.
        if (callback &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - last_callback)
                    .count() >= 1.0) {
            emit_progress(/*new_best=*/false);
        }
    }

    model.restore_state(best_state);
    // Release the artificial objective bound so post-solve feasibility checks
    // (verifiers iterating model constraints) don't see it violated by ~eps.
    if (has_obj) {
        model.set_objective_bound(std::numeric_limits<double>::infinity());
    }
    full_evaluate(model);

    double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    SearchResult result;
    result.objective = have_feasible ? best_feasible_obj : std::numeric_limits<double>::infinity();
    result.feasible = have_feasible;
    result.best_state = best_state;
    result.iterations = fj.iterations();  // total GLS iterations (not batch count)
    result.time_seconds = elapsed;
    return result;
}

}  // namespace cbls
