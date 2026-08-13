#include "cbls/search.h"

#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"
#include "cbls/randomize.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace cbls {

SolveCallback::~SolveCallback() = default;

void initialize_random(Model& model, RNG& rng) {
    for (auto& var : model.variables_mut()) {
        randomize_var(var, rng);
    }
}

void initialize_structured_random(Model& model, RNG& rng) {
    for (auto& var : model.variables_mut()) {
        if (!is_structured(var.type)) {
            continue;
        }
        randomize_var(var, rng);
    }
}

const char* termination_reason_name(TerminationReason reason) {
    switch (reason) {
        case TerminationReason::TimeLimit:
            return "time_limit";
        case TerminationReason::IterationLimit:
            return "iteration_limit";
        case TerminationReason::Feasible:
            return "feasible";
        case TerminationReason::NoBudget:
            return "no_budget";
    }
    // Unreachable for any value of the enum; keeps the function total so a
    // caller can print the result unconditionally.
    return "unknown";
}

// Construction heuristic: Generalised Feasibility Jump (ViolationLS). Refines
// the model's current assignment toward feasibility. Delegates to
// FeasibilityJump; see src/feasibility_jump.cpp.
int64_t fj_nl_initialize(Model& model, ViolationManager& vm, int max_iterations, RNG* rng_ptr,
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
    return fj.iterations();
}

// A STRUCTURAL batch (paper Algorithm 6 has FJ/NJ; this is the list/set peer):
// sweep the List/Set variables, try the candidate structural moves (swap /
// 2-opt / relocate / or-opt / set add-remove-swap) for each, and greedily keep
// any that reduce total weighted violation (i.e. negative weighted delta_G under
// the current GLS weights W, since total_violation() is W-weighted).
// FeasibilityJump only jumps scalar variables, so list-structured models cannot
// improve their list/set assignment without this. Returns true if any move was
// committed (the caller must then resync the FJ scan-set/jump-table).
//
// The sweep is deadline-bounded *between variables*, never mid-variable: each
// variable's move set is evaluated whole, so the reference move set is never
// truncated for speed, and the overrun is capped at one variable's work.
//
// The bound is needed because the sweep's cost is unbounded in the model size:
// O(#structured vars x #moves x (delta_evaluate + O(#constraints))), since
// total_violation() rescans every constraint on each of the two calls per move.
// On a 1500-List x 100-element model with 40k constraints a 0.5s budget ran
// 1.19-1.25s unbounded versus 0.502s bounded. `solve(model, time_limit)` is a
// library contract, and that is a violation for any user model of this shape
// (issue #105). Real benchmark models are nowhere near it -- pharma-glsp's
// largest class sweeps 10 List variables in ~0.5ms -- so this bound is about
// honouring the contract on large models, not about the benchmarks.
//
// The check is unconditional per variable rather than strided. An earlier
// self-tuning stride was deleted: because the stride persisted across passes
// while its counter reset per pass, once it exceeded the model's structured
// variable count it could never fire again, so it did nothing at all on 160 of
// the 170 real pharma-glsp instances (2-6 List variables each). A per-variable
// clock read costs ~1.4us only on an HPET clocksource like the machine this was
// measured on; via the vDSO on a TSC clocksource it is ~20-25ns. Amortising a
// 60x-inflated constant did not justify the complexity.
static bool structural_pass(Model& model, ViolationManager& vm, RNG& rng, bool has_deadline,
                            std::chrono::steady_clock::time_point deadline) {
    bool changed = false;
    for (const auto& var : model.variables()) {
        if (!is_structured(var.type)) {
            continue;
        }
        if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
            break;
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
    // time_limit <= 0 means "no wall-clock budget": the run is bounded by
    // config.max_iterations alone and is therefore fully deterministic, which is
    // what tests need. Any positive limit is a hard deadline enforced at every
    // sub-step below, not just between batches.
    const bool has_deadline = time_limit > 0.0;
    // Saturate before converting to the clock's integer tick type: callers pass
    // very large limits to mean "effectively unbounded", and casting e.g.
    // double::max() seconds to nanoseconds overflows int64 and yields a deadline
    // already in the past, which would end the search immediately.
    constexpr double kMaxBudgetSeconds = 1.0e9;  // ~31 years
    // Saturate once and reuse for FJ's deadline below: FeasibilityJump::begin()
    // performs the same integer-tick duration_cast, so handing it the raw value
    // would reintroduce exactly the overflow this saturation prevents.
    const double budget_seconds = has_deadline ? std::min(time_limit, kMaxBudgetSeconds) : 0.0;
    const auto deadline =
        start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(budget_seconds));
    auto past_deadline = [&]() {
        return has_deadline && std::chrono::steady_clock::now() >= deadline;
    };
    auto remaining = [&]() {
        if (!has_deadline) {
            return 0.0;  // unbounded: sub-steps use their own iteration budgets
        }
        return std::max(
            0.0, std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count());
    };

    // Exactly one path initialises each variable (#108). FeasibilityJump owns the
    // scalar start — `begin(set_initial_x)` below sets every Bool/Int/Float to the
    // domain value closest to zero, per the published Feasibility Jump — so this
    // call covers only the types FJ cannot initialise (List, Set).
    //
    // Randomising the scalars here too, as this used to, was worse than merely
    // redundant: `begin()` overwrote every one of them a dozen lines later, so the
    // draws were dead but still consumed, and the code read as though the seed
    // varied the starting point when it did not. It also hid a live hazard --
    // `rng.uniform(lb, ub)` returns NaN on an infinite-width domain and +inf on a
    // half-infinite one, and `rng.integers` casts an infinite bound to INT64_MIN.
    //
    // The seed still drives everything else: List/Set initialisation here,
    // scan-set sampling, perturbation kicks, GLS rho, LNS destroy sets. Only the
    // initial *scalar* point is seed-independent, and deliberately so. A caller
    // who wants a randomised scalar start composes the two public pieces:
    // `initialize_random(model, rng)` followed by `solve(..., {.skip_init = true})`,
    // which keeps the assignment it was handed.
    if (!config.skip_init) {
        initialize_structured_random(model, rng);
    }

    GFJConfig gfj;
    gfj.two_phase = false;  // batches reuse weights across calls; single-phase GLS
    // Hand FJ the same absolute deadline (its own is armed in begin(), called
    // immediately below). A batch is 1000 GLS iterations, so without this a
    // batch entered just before the deadline runs to completion and overruns it
    // — measured at +45% on the largest MINLPLib instance.
    gfj.time_limit = budget_seconds;  // saturated: begin() casts to integer ticks
    gfj.max_iterations = 0;
    FeasibilityJump fj(model, vm, rng, gfj);
    fj.begin(/*set_initial_x=*/!config.skip_init);

    const int32_t obj_ci = model.objective_constraint_idx();
    const auto& cids = model.constraint_ids();

    // Largest violation over the *real* constraints (the artificial objective
    // constraint excluded); 0.0 when every real constraint holds. Unweighted, so
    // it is comparable across the run regardless of the GLS weight dynamics.
    //
    // NaN maps to +inf, not 0: a non-convex body can evaluate to NaN (inf-inf,
    // 0*inf, log of a negative), and a bare `value > tol` test would read that
    // as satisfied and hand back a "feasible" solution we have no evidence for.
    // This mirrors the guard in ViolationManager's clamped_node_violation.
    auto max_real_violation = [&]() {
        double worst = 0.0;
        for (size_t i = 0; i < cids.size(); ++i) {
            if (static_cast<int32_t>(i) == obj_ci) {
                continue;
            }
            double v = model.node(cids[i]).value;
            if (std::isnan(v)) {
                return std::numeric_limits<double>::infinity();
            }
            if (v > worst) {
                worst = v;
            }
        }
        return worst;
    };
    auto real_feasible = [&]() { return max_real_violation() <= config.feasibility_tolerance; };
    auto current_obj = [&]() { return has_obj ? model.node(model.objective_id()).value : 0.0; };

    double best_feasible_obj = std::numeric_limits<double>::infinity();
    Model::State best_state = model.copy_state();
    bool have_feasible = false;
    // Closest approach to the feasible region, tracked so an infeasible run
    // returns something diagnosable (which constraint is left violated, and by
    // how much) instead of the untouched initial assignment.
    double best_violation = std::numeric_limits<double>::infinity();
    Model::State closest_state = best_state;
    int perturbations = 0;
    int stagnation = 0;
    int64_t batches = 0;
    auto last_callback = start;

    // Skip the structural batch entirely on scalar-only models.
    const bool has_structural =
        std::any_of(model.variables().begin(), model.variables().end(),
                    [](const Variable& v) { return is_structured(v.type); });
    // Effective structural-batch probability: explicit config overrides; <0 means
    // auto (0.33 with list/set vars, 0 otherwise). Zeroed on scalar-only models.
    const double structural_probability = !has_structural ? 0.0
                                          : config.structural_batch_probability >= 0.0
                                              ? config.structural_batch_probability
                                              : 0.33;

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
    //
    // PRECONDITION: the caller has established real_feasible().
    auto record_best = [&]() -> bool {
        double obj = current_obj();
        // Feasibility is a property of the constraints alone. A non-convex
        // objective can overflow to +inf/NaN on part of the feasible region
        // (the Thomson problem's coincident-point configurations, say), and
        // such a point is still a feasible point of the model — refusing to
        // record it left have_feasible false and reported the whole instance
        // infeasible (issue #100).
        //
        // It cannot serve as an objective incumbent, though: it must never
        // become best_feasible_obj (nothing could ever beat +inf under the
        // relative-improvement test) and must never tighten the bound (the
        // `obj <= bound` row would go permanently unsatisfiable). So it is kept
        // strictly as the first feasibility witness, and any later
        // finite-objective feasible point displaces it.
        if (!std::isfinite(obj)) {
            if (have_feasible) {
                return false;  // already have a witness, and possibly a better one
            }
            have_feasible = true;
            best_state = model.copy_state();
            emit_progress(/*new_best=*/true);
            return true;
        }
        // isfinite(best_feasible_obj) guards the case where the incumbent is the
        // +inf witness above: the relative-improvement test would compute
        // `inf - inf` = NaN and decide by NaN comparison.
        if (have_feasible && std::isfinite(best_feasible_obj) &&
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
            // Bound the repair by whatever budget is left, so an LNS kick near
            // the deadline cannot run its own independent 2s.
            // Floored at a tiny positive value while a deadline exists:
            // remaining() returns exactly 0.0 if the clock crossed the deadline
            // since the past_deadline() check above, and 0 means "no wall-clock
            // limit" downstream in fj_nl_initialize — the opposite of intent.
            const double repair_limit =
                has_deadline ? std::max(1e-9, std::min(2.0, remaining())) : 0.0;
            lns->destroy_repair(model, vm, rng, repair_limit);
            fj.reset_weights();  // LNS mutated state outside GFJ
        } else {
            fj.perturb(config.perturbation_probability);  // self-resyncs
        }
        sample_rho();
        ++perturbations;
        stagnation = 0;
    };

    // Which budget ends the run. Assigned at every loop exit so it always
    // describes the exit actually taken; the `while` condition below is the only
    // exit that is not a `break`, so it seeds the value and each `break`
    // overwrites it. Reported on SearchResult so callers — and the regression
    // tests for the deadline bounds — can tell a budget-limited run from a
    // converged one without timing the call (#104).
    TerminationReason termination = TerminationReason::TimeLimit;
    while (!past_deadline()) {
        // Count *actual* GLS iterations, which is what the config documents and
        // what SearchResult::iterations reports. Using batches *
        // batch_iterations over-counts whenever a batch exits early on
        // feasibility, so the budget expired after far less work than asked for.
        if (config.max_iterations > 0 && fj.iterations() >= config.max_iterations) {
            termination = TerminationReason::IterationLimit;
            break;
        }
        // Structural and Novelty batches do not charge fj.iterations(), so on a
        // List/Set model with no wall clock the iteration budget alone cannot
        // guarantee termination (structural_batch_probability = 1.0 would spin
        // forever). Batches <= iterations by construction, so this only bites
        // when iterations have stalled.
        if (config.max_iterations > 0 && batches >= config.max_iterations) {
            termination = TerminationReason::IterationLimit;
            break;
        }
        if (!has_deadline && config.max_iterations <= 0) {
            // Neither budget set: nothing would ever stop the loop.
            termination = TerminationReason::NoBudget;
            break;
        }

        // Pick this batch's kind (paper Algorithm 6 alternates FJ/NJ; the
        // STRUCTURAL batch is the list/set peer added in P4). Structural and
        // Novelty batches commit changes outside the FJ scan-set/jump-table, so
        // they must be followed by a resync.
        enum class BatchKind { FeasibilityJump, NoveltyJump, Structural };
        BatchKind kind = BatchKind::FeasibilityJump;
        if (rng.random() < structural_probability) {
            kind = BatchKind::Structural;
        } else if (config.use_compound_moves && rng.random() < config.novelty_jump_probability) {
            kind = BatchKind::NoveltyJump;
        }

        bool resync = false;
        switch (kind) {
            case BatchKind::Structural:
                resync = structural_pass(model, vm, rng, has_deadline, deadline);
                break;
            case BatchKind::NoveltyJump:
                fj.apply_novelty_jump();
                resync = true;
                break;
            case BatchKind::FeasibilityJump:
                fj.batch(config.batch_iterations);
                break;
        }
        ++batches;

        double batch_violation = max_real_violation();
        // Only tracked until the first feasible solution: after that both the
        // final restore and the returned state use best_state, so the snapshot
        // would be pure allocation on every improving batch. The `batches == 1`
        // clause guarantees one capture even on an all-NaN run (violation stays
        // +inf, so `<` never fires) without re-snapshotting on every batch of a
        // violation plateau — the common infeasible case.
        if (!have_feasible && (batch_violation < best_violation || batches == 1)) {
            best_violation = batch_violation;
            closest_state = model.copy_state();
        }

        bool improved = false;
        if (batch_violation <= config.feasibility_tolerance) {
            // Record the feasible point we already have *before* polishing. The
            // hook descends the penalty-method objective and can land outside
            // the feasible region; recording only afterwards silently threw away
            // genuinely feasible solutions (an instance would be reported
            // infeasible despite the search having visited a feasible point).
            improved = record_best();
            // The hook is unbounded in *time* — a custom InnerSolverHook may do
            // arbitrary work, and even FloatIntensifyHook sweeps every Float
            // max_sweeps times. Don't start one we have no budget for.
            if (hook && !past_deadline()) {
                hook->solve(model, vm, {});  // continuous-objective polish (mutates floats)
                resync = true;
                if (real_feasible()) {  // keep the polish only if it stayed feasible
                    improved = record_best() || improved;
                }
            }
        }

        if (improved) {
            stagnation = 0;
            // Making progress: the Float escape probe is not needed and is not free.
            fj.set_escape_probe(false);
            fj.reset_weights();  // new best: fresh GLS weights (paper) + new rho
            sample_rho();
            if (!has_obj) {
                // Pure feasibility: first solution is the answer.
                termination = TerminationReason::Feasible;
                break;
            }
        } else {
            ++stagnation;
            if (resync) {
                fj.resync();  // re-sync after hook/structural mutation, keep GLS weights
            }
        }

        if (stagnation >= config.perturbation_period && !past_deadline()) {
            // Genuinely stuck. Arm the Float escape probe: a variable sitting at a
            // stationary point of every violated constraint has no other candidate
            // that can move it, and diversification alone cannot rescue it because
            // the search re-converges to the same point. Disarmed again on the next
            // improvement, so a productive search never pays for it.
            fj.set_escape_probe(true);
            diversify();
        }

        // Periodic progress (~1s) even without improvement.
        if (callback &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - last_callback)
                    .count() >= 1.0) {
            emit_progress(/*new_best=*/false);
        }
    }

    // On a feasible run the best-objective assignment is the answer; on an
    // infeasible one, hand back the closest approach rather than the initial
    // assignment (which carries no information about where the search got to).
    model.restore_state(have_feasible ? best_state : closest_state);
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
    result.best_state = have_feasible ? best_state : closest_state;
    // Residual of the assignment actually being returned, from the fresh
    // full_evaluate above rather than the incrementally-maintained node values.
    result.best_violation = max_real_violation();
    result.iterations = fj.iterations();  // total GLS iterations (not batch count)
    result.time_seconds = elapsed;
    result.termination = termination;
    return result;
}

}  // namespace cbls
