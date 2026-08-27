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
// O(#structured vars x #moves x (delta_evaluate + O(#constraints))), since the
// weighted delta rescans every constraint once per move.
// On a 1500-List x 100-element model with 40k constraints a 0.5s budget ran
// 1.19-1.25s unbounded versus 0.502s bounded. `solve(model, time_limit)` is a
// library contract, and that is a violation for any user model of this shape
// (issue #105). Real benchmark models were nowhere near it -- pharma-glsp's
// largest class swept 10 List variables in ~0.5ms (that benchmark has since
// been retired, #28; the measurement is what motivated this bound) -- so this
// bound is about honouring the contract on large models, not the benchmarks.
//
// The check is unconditional per variable rather than strided. An earlier
// self-tuning stride was deleted: because the stride persisted across passes
// while its counter reset per pass, once it exceeded the model's structured
// variable count it could never fire again, so it did nothing at all on 160 of
// the 170 real pharma-glsp instances (2-6 List variables each; the benchmark
// is gone in #28, the bug it exposed is not). A per-variable
// clock read costs ~1.4us only on an HPET clocksource like the machine this was
// measured on; via the vDSO on a TSC clocksource it is ~20-25ns. Amortising a
// 60x-inflated constant did not justify the complexity.
static bool structural_pass(Model& model, ViolationManager& vm, RNG& rng, bool has_deadline,
                            std::chrono::steady_clock::time_point deadline) {
    bool changed = false;
    // Per-constraint violations of the last ACCEPTED assignment. A move is judged
    // by ViolationManager::weighted_delta_from against this, not by differencing
    // two whole-sum total_violation() values. That subtraction had TWO defects,
    // and only the first one needs a clamped row.
    //
    // 1. Clamped-row blindness (#118). A row clamped to kInfPenalty swallows the
    //    real rows: 1e30 is fourteen orders of magnitude above an O(1) row, so
    //    both sums round to the same double and `after < before - 1e-12` reads
    //    `before < before`. That is #100's defect in this pass, and #116's
    //    sentinel objective bound put a permanently clamped row into every model
    //    whose feasible region contains a non-finite objective — so the pass
    //    rejected every structural move for as long as the sentinel was
    //    installed, however much it improved the real rows.
    //
    // 2. Phantom improvements, on ANY model, clamped row or not, and predating
    //    #116. Both readings came from total_violation()'s incremental
    //    accumulator (cached_total_ += (new - old) * W), whose 1000-call
    //    recompute bounds the accumulated rounding error without removing it, and
    //    `before` was threaded across candidate moves — so two readings taken at
    //    different points in that drift cycle differ in the last ulp even when no
    //    constraint changed at all. `- 1e-12` cannot filter that: x - 1e-12 == x
    //    for every double x >= 2^14, and GLS weights put setcover's weighted
    //    total at ~4.4e6 (1 ulp = 9.3e-10). Measured on scp41/Set with no row
    //    clamped anywhere: 99 of 39627 candidates were accepted with a true
    //    weighted delta of exactly 0 and zero rows changed, each of them setting
    //    `changed` and forcing a needless fj.resync().
    //
    // Differencing per constraint fixes both: the clamped row cancels exactly,
    // and an unchanged row contributes an exact 0 instead of a drifted total.
    //
    // Both calls self-correct to the current node values (they read the
    // constraint nodes directly), so no explicit invalidate is needed across the
    // apply/undo dance; the baseline is re-snapshotted only when a move is kept.
    std::vector<double> baseline;
    vm.snapshot_violations(baseline);
    for (const auto& var : model.variables()) {
        if (!is_structured(var.type)) {
            continue;
        }
        if (has_deadline && std::chrono::steady_clock::now() >= deadline) {
            break;
        }
        auto moves = generate_standard_moves(var, rng);
        for (const auto& move : moves) {
            auto saved = save_move_values(model, move);
            auto touched = apply_move(model, move);
            delta_evaluate(model, touched);
            if (vm.weighted_delta_from(baseline) < -1e-12) {
                changed = true;  // improving: keep
                vm.snapshot_violations(baseline);
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
    const auto deadline = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                      std::chrono::duration<double>(budget_seconds));
    auto past_deadline = [&]() {
        return has_deadline && std::chrono::steady_clock::now() >= deadline;
    };
    auto remaining = [&]() {
        if (!has_deadline) {
            return 0.0;  // unbounded: sub-steps use their own iteration budgets
        }
        return std::max(
            0.0,
            std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count());
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
        // relative-improvement test), and the bound must never be DERIVED from
        // it — `obj - eps` on a non-finite obj is +inf or NaN, and a NaN bound
        // makes the `obj <= bound` row permanently and unfixably violated. So
        // it is kept strictly as the first feasibility witness, and any later
        // finite-objective feasible point displaces it.
        if (!std::isfinite(obj)) {
            if (have_feasible) {
                return false;  // already have a witness, and possibly a better one
            }
            have_feasible = true;
            best_state = model.copy_state();
            // Leaving the bound at +inf as well, though, left the search with
            // no objective signal at all (issue #116). `obj <= +inf` is vacuous
            // by construction — comparison_residual reads a *written* +inf as
            // "this side is absent" and returns residual 0 (#100) — so with the
            // bound still at its initial value the objective row can never be
            // violated, no jump candidate scores anything through it, and every
            // later batch returns "feasible" having done no work.
            //
            // So install a finite bound that is NOT derived from obj: the
            // loosest one there is. Its only job is to make "the objective is
            // not a number" a violated row, so it sits at the violation
            // machinery's own blowup clamp — 1e30 is what clamped_node_violation
            // maps +inf and NaN to (kInfPenalty, shared from violation.h so the two
            // cannot drift apart), i.e. the largest objective value that machinery can
            // still tell apart from a blowup — and every finite objective under
            // it satisfies the row. A feasible point whose objective is finite
            // but *above* 1e30 is therefore indistinguishable from +inf here;
            // that is pre-existing kInfPenalty behaviour, not new.
            //
            // Why the loosest rather than something tighter (e.g. the largest
            // finite objective evaluated so far): a finite bound that some
            // feasible point can meet is the whole safety property here, and
            // this one is met by *every* finite-objective assignment, so the
            // only points it rules out are the ones with no objective value at
            // all. A tighter sentinel would keep pressure on after the
            // objective goes finite, but it can rule out feasible
            // finite-objective points, and it buys only the handful of batches
            // until the first finite-objective feasible point tightens the
            // bound properly through the path below.
            //
            // Guarded on the bound still being +inf, so this replaces "no bound
            // at all" and never overwrites one derived from a real incumbent.
            // (has_obj is implied — current_obj() returns a finite 0.0 when
            // there is no objective — but it is kept as the guard on
            // set_objective_bound's precondition.)
            //
            // Returning true below is load-bearing: the caller reads a new best
            // as an improvement and calls fj.reset_weights(), which rebuilds
            // FeasibilityJump's violated set. Without that rebuild the row just
            // installed stays invisible to the jump table. The baseline already
            // returned true from this same witness path, so nothing else on it
            // changes.
            //
            // One consequence this DOES introduce, confined to the window where
            // the sentinel is installed and the objective is still non-finite:
            //
            //   * progress reports pair feasible = true with total_violation
            //     ~1e30 until the objective goes finite. Documented on
            //     SolveProgress::total_violation rather than suppressed: that
            //     field is the weighted total over *all* rows including this
            //     artificial one, and feasible-with-positive-violation is
            //     already the steady state after any bound tightening, so no
            //     consumer can be reading it as "zero whenever feasible". Only
            //     the magnitude is new.
            //
            // The invariant this row imposes on the rest of the window: anything
            // that compares two assignments by violation must difference PER
            // CONSTRAINT, because a row clamped to 1e30 swallows every O(1) real
            // row when whole sums are subtracted instead. FJ's jump scoring
            // already did (#100); structural_pass did not, and was blind for the
            // whole window until #118 gave it the same treatment. LNS::state_key
            // and max_real_violation are safe by exclusion — neither looks at the
            // objective row at all.
            //
            // One deliberate interaction: with this row violated and its
            // gradient non-finite, float_jump_candidates reports "gradient
            // unusable" at a coincident-point configuration, which makes the
            // Float escape probe eligible there for the first time. When the
            // probe actually arms is #117's subject, so elec25 has to be
            // measured with both changes in.
            if (has_obj && !std::isfinite(model.objective_bound())) {
                model.set_objective_bound(kInfPenalty);  // the shared clamp; see violation.h
            }
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

    // Second arming condition for the Float escape probe (#117).
    // `perturbation_period` counts BATCHES, and a batch is `batch_iterations`
    // GLS iterations: microseconds on a small model, seconds on an expensive
    // one, so the threshold is a wall-clock duration that varies by orders of
    // magnitude across a roster. Measured on MINLPLib elec25 at a 60s budget a
    // batch costs ~1.2s, so the run gets 52 batches against a threshold of 100
    // and the probe is never armed at all — the stagnation gate that makes it a
    // last resort (#107) is dead code on any model whose batches cost seconds.
    // Arm on whichever comes first: the batch count, or this fraction of the
    // wall-clock budget with no new best — the latter only while the run is
    // projected to fall short of the batch count (see the gate at the arming
    // site below, which is what keeps this from starving diversification).
    //
    // Guarded on has_deadline below: with no wall-clock budget no clock read may
    // influence control flow, or iteration-budgeted runs stop being
    // bit-reproducible. Deliberately does NOT also diversify — the kick cadence
    // is a tuned parameter, and making it time-aware is a separate question that
    // wants its own measurement.
    constexpr double kEscapeArmFraction = 0.25;
    auto last_improvement = start;

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
            // Gated so an iteration-budgeted run reads no clock AT ALL, not merely
            // no clock that reaches control flow: this is the only writer of
            // last_improvement and its only reader is the has_deadline-gated
            // arming block below. Keeps architecture.md's "the loop reads no
            // clock at all" literally true.
            if (has_deadline) {
                last_improvement = std::chrono::steady_clock::now();
            }
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

        // Time-based arming (#117); see kEscapeArmFraction above. Tested on every
        // batch, not only stagnant ones, and costs ONE steady_clock::now() per
        // batch on top of the loop's own deadline reads — immaterial against a
        // batch of batch_iterations GLS iterations. Skipped entirely once the
        // probe is armed: arming again would be a no-op, and a new best clears
        // the flag and re-enables the check.
        //
        // This route carries no diversification kick, unlike the stagnation route
        // below, and an earlier revision gated it on the run being projected to
        // fall short of perturbation_period batches for fear of starving
        // diversify(). That gate was removed: the drip it defended against cannot
        // run away, because the improvement that resets `stagnation` also disarms
        // the probe (see the `improved` branch above), so re-arming costs another
        // kEscapeArmFraction of the budget and the route can arm at most
        // 1/kEscapeArmFraction times per run. #107's measured 9x regression came
        // from an always-on probe with no disarm, which is a different regime.
        // The gate was also measured to cost objective quality on a probe-
        // sensitive model while preventing nothing, and it is the ungated form
        // that #117's roster numbers describe.
        if (has_deadline && !fj.escape_probe()) {
            const auto now = std::chrono::steady_clock::now();
            if (now < deadline && std::chrono::duration<double>(now - last_improvement).count() >=
                                      kEscapeArmFraction * budget_seconds) {
                fj.set_escape_probe(true);
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
    result.escape_probe_armed = fj.escape_probe();
    return result;
}

}  // namespace cbls
