#include "cbls/search.h"

#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"
#include "cbls/randomize.h"
// search.h only forward-declares SearchCoordination, deliberately -- see the
// declaration there. This translation unit is one of the few that needs the
// definition, because ViolationLSLoop reads both of its channels.
#include "cbls/solution_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cbls {

SolveCallback::~SolveCallback() = default;

void initialize_random(Model& model, RNG& rng) {
    for (int32_t v = 0; v < static_cast<int32_t>(model.num_vars()); ++v) {
        randomize_var(model.var_mut(v), rng);
    }
}

void initialize_structured_random(Model& model, RNG& rng) {
    for (int32_t v = 0; v < static_cast<int32_t>(model.num_vars()); ++v) {
        Variable& var = model.var_mut(v);
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
        case TerminationReason::Stopped:
            return "stopped";
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
    RNG& rng = rng_ptr != nullptr ? *rng_ptr : local_rng;

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
    //    for every double x > 2^14 (16384 itself is the last value it still
    //    moves), and GLS weights put setcover's weighted total at ~4.4e6, where
    //    one ulp is 9.3e-10. Measured on scp41/Set with no row clamped anywhere:
    //    99 of 39627 candidates were accepted with a true weighted delta of
    //    exactly 0, each of them setting `changed` and forcing a needless
    //    fj.resync().
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

namespace {

// The wall-clock budget, computed in solve() because FeasibilityJump has to be
// handed it at construction, and then carried into the loop below.
struct Budget {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point deadline;
    bool has_deadline = false;
    double seconds = 0.0;  // saturated; see make_budget
};

Budget make_budget(double time_limit) {
    Budget b;
    b.start = std::chrono::steady_clock::now();
    // time_limit <= 0 means "no wall-clock budget": the run is bounded by
    // config.max_iterations alone and is therefore fully deterministic, which is
    // what tests need. Any positive limit is a hard deadline enforced at every
    // sub-step below, not just between batches.
    b.has_deadline = time_limit > 0.0;
    // Saturate before converting to the clock's integer tick type: callers pass
    // very large limits to mean "effectively unbounded", and casting e.g.
    // double::max() seconds to nanoseconds overflows int64 and yields a deadline
    // already in the past, which would end the search immediately.
    constexpr double kMaxBudgetSeconds = 1.0e9;  // ~31 years
    // Saturate once and reuse for FJ's deadline: FeasibilityJump::begin()
    // performs the same integer-tick duration_cast, so handing it the raw value
    // would reintroduce exactly the overflow this saturation prevents.
    b.seconds = b.has_deadline ? std::min(time_limit, kMaxBudgetSeconds) : 0.0;
    b.deadline = b.start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<double>(b.seconds));
    return b;
}

// Effective structural-batch probability: explicit config overrides; <0 means
// auto (0.33 with list/set vars, 0 otherwise). Zeroed on scalar-only models,
// which skip the structural batch entirely.
double effective_structural_probability(const Model& model, const SearchConfig& config) {
    const bool has_structural =
        std::any_of(model.variables().begin(), model.variables().end(),
                    [](const Variable& v) { return is_structured(v.type); });
    if (!has_structural) {
        return 0.0;
    }
    return config.structural_batch_probability >= 0.0 ? config.structural_batch_probability : 0.33;
}

// Pick this batch's kind (paper Algorithm 6 alternates FJ/NJ; the
// STRUCTURAL batch is the list/set peer added in P4). Structural and
// Novelty batches commit changes outside the FJ scan-set/jump-table, so
// they must be followed by a resync.
enum class BatchKind : std::uint8_t { FeasibilityJump, NoveltyJump, Structural };

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

// Second witness for #102's unproductive-batch exit. FeasibilityJump ends a
// batch on ITS measure -- the real rows' unweighted violation -- and that
// measure cannot see the artificial objective row at all. Before the first
// feasible solution that is the whole point. After it, the search's work is
// a trade between the two: the bound is tightened on every new best, FJ
// pulls the assignment off the real-feasible set to chase the objective row,
// and the real rows settle at a strictly positive equilibrium. Since the
// measure's reference is a running MINIMUM over the batch, "no new all-time
// low" is then the normal state of a search that is working, and the exit
// fires unconditionally -- on MINLPLib ex8_6_1 it fired on a run improving
// its incumbent on 152 of 162 batches, and the three LNS kicks that followed
// took 4.7s of a 10s budget (#102).
//
// So arm the exit only once THIS loop's own stagnation count agrees the
// search has stopped improving. That makes the mechanism an acceleration of
// the stagnation window rather than a replacement for it: the kick arrives
// after this many non-improving batches plus one unproductive one, instead
// of after `perturbation_period` batches -- a 20x shortening at the default,
// where leaving it unwitnessed shortened it by ~300x and turned a stall
// detector into a diversification schedule.
//
// A FRACTION of `perturbation_period` rather than a fresh constant, so the
// two windows keep their ratio when a caller retunes the one knob that
// already exists; floored at 1, since 0 is "always armed", the regime this
// is here to end.
//
// The divisor is the honest part to argue with, and it is tuned: a sweep of
// {5, 10, 20} batches on MINLPLib at a 10s budget over four paired seeds.
// All three remove the ex8_6_1 regression completely -- it matches or beats
// main on 4/4 seeds at every one of them, because a search that improves
// this often simply never reaches the threshold. They differ on the
// instance the mechanism exists for: st_e40 reaches its BKS on 4/4 seeds at
// 5 and on 2/4 at 10 or 20, since after the first feasible solution it needs
// the accelerated kick to move between its 52 feasible integer combinations.
// #158 was expected to cost some of that back and, measured properly, does
// not: at 10s over four PAIRED seeds st_e40 is 0.00 gap and 4/4 feasible both
// before and after kicks began departing from kick_origin(). An earlier pass
// here claimed 8/8 -> 6/8; that came from re-measuring the largest movers of a
// two-seed run, which selects for noise. See diversify() and
// SearchConfig::unproductive_iterations.
// nvs01 is feasible on 4/4 at all three (main solves it on none) with
// objective quality too noisy to separate them. So the smallest of the three
// is chosen, which is also the one closest to the unwitnessed behaviour on
// the instances that want it. Nothing here establishes that 20 transfers off
// MINLPLib; it is the same standing complaint GFJConfig::
// unproductive_iterations records against its own 300.
// A fraction of perturbation_period rather than a fresh knob, so the two
// windows keep their ratio when a caller retunes the one that already
// exists. The max(1, ...) floor is also where that ratio stops holding: at
// perturbation_period < kUnproductiveArmDivisor the window is 1 batch and
// the shortening is whatever perturbation_period happens to be, not 20x.
constexpr int kUnproductiveArmDivisor = 20;

// The ViolationLS outer loop: its state, and the steps of one pass over it. Every
// member below was a local of solve(), most of them closed over by one of its
// lambdas -- which is why the complexity metric charged the whole loop for each
// of them. They are gathered into one object rather than threaded through free
// functions because the steps genuinely share this state; what the split buys is
// that each step of docs/architecture.md's "Main Loop" can be read on its own,
// under the name that section already gives it.
class ViolationLSLoop {
public:
    ViolationLSLoop(Model& model, ViolationManager& vm, RNG& rng, FeasibilityJump& fj,
                    const SearchConfig& config, const Budget& budget, InnerSolverHook* hook,
                    LNS* lns, int lns_interval, SolveCallback* callback, SearchCoordination* coord);

    SearchResult run();

private:
    // ---- observations of the current assignment ----
    // Largest violation over the *real* constraints (the artificial objective
    // constraint excluded); 0.0 when every real constraint holds. Unweighted, so
    // it is comparable across the run regardless of the GLS weight dynamics.
    //
    // NaN maps to +inf, not 0: a non-convex body can evaluate to NaN (inf-inf,
    // 0*inf, log of a negative), and a bare `value > tol` test would read that
    // as satisfied and hand back a "feasible" solution we have no evidence for.
    // This mirrors the guard in ViolationManager's clamped_node_violation.
    [[nodiscard]] double max_real_violation() const;
    [[nodiscard]] bool real_feasible() const {
        return max_real_violation() <= config_.feasibility_tolerance;
    }
    [[nodiscard]] double current_obj() const {
        return has_obj_ ? model_.node_value(model_.objective_id()) : 0.0;
    }
    // Whether a peer worker has answered the question. Relaxed is the right
    // ordering: the flag guards no data -- the pool has its own mutex -- and
    // the only cost of observing it a batch late is that batch.
    [[nodiscard]] bool stop_requested() const {
        return coord_ != nullptr && coord_->stop != nullptr &&
               coord_->stop->load(std::memory_order_relaxed);
    }
    [[nodiscard]] bool clock_expired() const {
        return has_deadline_ && std::chrono::steady_clock::now() >= deadline_;
    }
    // Read by the loop condition and by every mid-batch "is there budget left"
    // guard, so a raised stop flag halts a worker everywhere the clock would.
    [[nodiscard]] bool past_deadline() const { return stop_requested() || clock_expired(); }
    [[nodiscard]] double remaining() const {
        if (!has_deadline_) {
            return 0.0;  // unbounded: sub-steps use their own iteration budgets
        }
        return std::max(
            0.0,
            std::chrono::duration<double>(deadline_ - std::chrono::steady_clock::now()).count());
    }

    // ---- the incumbent, and the kicks that leave it ----
    void sample_rho() { fj_.set_rho(rng_.random() < 0.5 ? 0.95 : 1.0); }
    void emit_progress(bool new_best);
    // Record the current (real-feasible) assignment if it improves the best and
    // tighten the objective bound. Returns true on a new best.
    //
    // PRECONDITION: the caller has established real_feasible().
    bool record_best();
    // Latch the first feasible point's objective and the time it took to reach
    // it (#149). Called from both of record_best's have_feasible_ transitions;
    // a no-op after the first. Observational only -- nothing reads the latched
    // values back, so the trajectory is unchanged by their being recorded.
    void note_first_feasible(double obj);
    // Seconds since solve() started. finish() reports the same quantity as
    // SearchResult::time_seconds.
    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
    // On stagnation: restore kick_origin() (#158), then LNS diversification
    // every lns_interval-th time, else perturb.
    // `allow_lns` is false only for #102's unproductive route once a feasible
    // solution exists. The kick itself is microseconds and st_e40 needs it to
    // hop between its 52 feasible integer combinations; what cost ex8_6_1 its
    // budget was the LNS half -- three repairs, each bounded by min(2.0,
    // remaining()), took 4.7s of a 10s run and the search never used a result.
    // Suppressing the repair rather than the kick keeps the cheap half of the
    // mechanism for the models it helps, and removes the expensive half from the
    // regime where the measure that triggers it has gone blind.
    void diversify(bool allow_lns = true);
    // Hand a new incumbent to the shared pool, if there is one. No-op for a
    // single-threaded solve. Called from both of record_best's recording arms.
    void share(double objective);
    // Restart from another worker's incumbent instead of perturbing our own.
    // Returns false -- leaving the assignment untouched -- when there is no
    // pool, the pool is empty, the drawn solution does not fit this model, or
    // the draw is the assignment we already hold.
    bool adopt_from_pool();
    // See the definition: the objective-bound rule an adoption applies.
    void reground_objective_bound_after_adoption(bool feasible_here, double obj);
    // Whether `state` is the assignment this model currently holds.
    [[nodiscard]] bool holds_assignment(const Model::State& state) const;
    // Whether the NEXT diversification kick is the one that draws LNS -- the
    // same test diversify() makes, asked before the fact. Adoption stands down
    // on those kicks, and advances `lns_slot_` on the ones it does take, so the
    // `lns_interval` cadence keeps running (see the call site in
    // maybe_diversify and the counter bump in adopt_from_pool).
    [[nodiscard]] bool lns_kick_due() const {
        return lns_ != nullptr && lns_interval_ > 0 &&
               (lns_slot_ % lns_interval_ == lns_interval_ - 1);
    }

    // The point a diversification kick departs from, or null when there is none
    // and the kick fires from wherever the search stands (#158).
    //
    // Normally our own incumbent. It is `adopted_origin_` instead while the
    // search is standing on a state adopt_from_pool installed that did not
    // become our incumbent -- a peer's point drawn from the better half rather
    // than the best, which is how the portfolio keeps its workers spread. The
    // next new incumbent clears it, because at that moment our own best IS where
    // the search got to.
    //
    // Both "is there an origin at all" tests live on the WRITE side rather than
    // here, and deliberately: `adopted_origin_` is only ever set from a draw
    // that this model found feasible with a finite objective, so the two rules
    // below cannot be short-circuited past by an adoption. Guarding on the read
    // instead would not even express the rule -- the witness test keys off
    // `best_feasible_obj_`, which says nothing about the adopted point.
    //
    // Null before the first feasible point: best_state_ is still the initial
    // assignment there, which is not a point to return to, so that regime keeps
    // the trajectory it always had. Also null while the only feasible point on
    // record is #100's non-finite-objective witness -- a feasibility witness
    // with no objective to descend, where the bound is the loosest one there is
    // and the useful thing is for the search to wander off and find a
    // finite-objective point rather than be pulled back to the degenerate
    // configuration every kick. Same rule adopt_from_pool applies to the bound:
    // never derive from a non-finite objective.
    [[nodiscard]] const Model::State* kick_origin() const {
        if (has_adopted_origin_) {
            return &adopted_origin_;
        }
        if (!have_feasible_ || (has_obj_ && !std::isfinite(best_feasible_obj_))) {
            return nullptr;
        }
        return &best_state_;
    }

    // ---- one pass of the main loop, in the order architecture.md lists it ----
    // Whether a budget has run out. Records which one in termination_.
    bool budget_exhausted();
    BatchKind pick_batch_kind();
    // Run the batch. Returns true if it committed changes outside FJ's scan-set
    // and jump-table, i.e. if the loop owes an fj_.resync().
    bool run_batch(BatchKind kind);
    void note_closest_approach(double batch_violation);
    // Steps 3-4: bank the feasible point, polish it, bank the polish. Returns
    // whether the batch produced a new best, and may set `resync`.
    bool polish_and_record(double batch_violation, bool& resync);
    // Steps 5-6. Returns false when the run is over (pure feasibility: solved).
    bool apply_batch_outcome(bool improved, bool resync);
    void maybe_arm_escape_probe();
    void maybe_diversify(BatchKind kind, bool improved);
    void maybe_emit_periodic_progress();
    SearchResult finish();

    Model& model_;
    ViolationManager& vm_;
    RNG& rng_;
    FeasibilityJump& fj_;
    const SearchConfig& config_;
    InnerSolverHook* hook_;
    LNS* lns_;
    int lns_interval_;
    SolveCallback* callback_;
    // Non-null only under ParallelSearch. Null here means the search touches
    // nothing shared, which is what keeps a single-threaded solve's trajectory
    // exactly what it was before this parameter existed.
    SearchCoordination* coord_;

    const std::chrono::steady_clock::time_point start_;
    const std::chrono::steady_clock::time_point deadline_;
    const bool has_deadline_;
    const double budget_seconds_;

    const bool has_obj_;
    const int32_t obj_ci_;
    const std::vector<int32_t>& cids_;
    const double structural_probability_;
    const int unproductive_arm_stagnation_;

    double best_feasible_obj_ = std::numeric_limits<double>::infinity();
    Model::State best_state_;
    bool have_feasible_ = false;
    // Closest approach to the feasible region, tracked so an infeasible run
    // returns something diagnosable (which constraint is left violated, and by
    // how much) instead of the untouched initial assignment.
    double best_violation_ = std::numeric_limits<double>::infinity();
    Model::State closest_state_;
    // The state adoption installed, held only while it is NOT best_state_; see
    // kick_origin(). Portfolio-only: adopt_from_pool is the sole writer and it
    // returns false immediately without a pool, so a single-threaded run never
    // sets the flag and never pays for the copy.
    Model::State adopted_origin_;
    bool has_adopted_origin_ = false;
    int perturbations_ = 0;
    int lns_repairs_ = 0;
    // The subset of lns_repairs_ that destroy_repair reported as accepted.
    // Instrumentation only: the value is recorded and never branched on, so the
    // trajectory is byte-identical to the run that discarded this return (#150).
    int lns_repairs_accepted_ = 0;
    // #149's pair. NaN until the first feasible point; see
    // SearchResult::first_feasible_objective for what each NaN means.
    double first_feasible_obj_ = std::numeric_limits<double>::quiet_NaN();
    double first_feasible_time_ = std::numeric_limits<double>::quiet_NaN();
    // Counts only the kicks ELIGIBLE for an LNS repair, which is what
    // `lns_interval` has always meant. Kept apart from `perturbations` because
    // #102's route can be refused its LNS half: letting a refused kick advance
    // the slot would shift -- and at some kick counts permanently freeze -- the
    // phase at which the perturbation_period route runs LNS, making that route's
    // cadence depend on how often the unproductive one fired.
    int lns_slot_ = 0;
    int stagnation_ = 0;
    int64_t batches_ = 0;
    std::chrono::steady_clock::time_point last_callback_;
    std::chrono::steady_clock::time_point last_improvement_;
    // Which budget ends the run. Assigned at every loop exit so it always
    // describes the exit actually taken; the `while` condition below is the only
    // exit that is not a `break`, so it seeds the value and each `break`
    // overwrites it. Reported on SearchResult so callers — and the regression
    // tests for the deadline bounds — can tell a budget-limited run from a
    // converged one without timing the call (#104).
    TerminationReason termination_ = TerminationReason::TimeLimit;
};

ViolationLSLoop::ViolationLSLoop(Model& model, ViolationManager& vm, RNG& rng, FeasibilityJump& fj,
                                 const SearchConfig& config, const Budget& budget,
                                 InnerSolverHook* hook, LNS* lns, int lns_interval,
                                 SolveCallback* callback, SearchCoordination* coord)
    : model_(model),
      vm_(vm),
      rng_(rng),
      fj_(fj),
      config_(config),
      hook_(hook),
      lns_(lns),
      lns_interval_(lns_interval),
      callback_(callback),
      coord_(coord),
      start_(budget.start),
      deadline_(budget.deadline),
      has_deadline_(budget.has_deadline),
      budget_seconds_(budget.seconds),
      has_obj_(model.objective_id() >= 0),
      obj_ci_(model.objective_constraint_idx()),
      cids_(model.constraint_ids()),
      structural_probability_(effective_structural_probability(model, config)),
      unproductive_arm_stagnation_(
          std::max(1, config.perturbation_period / kUnproductiveArmDivisor)),
      best_state_(model.copy_state()),
      closest_state_(best_state_),
      last_callback_(budget.start),
      last_improvement_(budget.start) {
    sample_rho();
}

double ViolationLSLoop::max_real_violation() const {
    double worst = 0.0;
    for (size_t i = 0; i < cids_.size(); ++i) {
        if (static_cast<int32_t>(i) == obj_ci_) {
            continue;
        }
        double v = model_.node_value(cids_[i]);
        if (std::isnan(v)) {
            return std::numeric_limits<double>::infinity();
        }
        worst = std::max(worst, v);
    }
    return worst;
}

void ViolationLSLoop::emit_progress(bool new_best) {
    if (callback_ == nullptr) {
        return;
    }
    vm_.invalidate_cache();
    SolveProgress p;
    p.iteration = batches_;
    p.time_seconds = elapsed();
    p.objective = best_feasible_obj_;
    p.total_violation = vm_.total_violation();
    p.feasible = real_feasible();
    p.new_best = new_best;
    p.perturbations = perturbations_;
    callback_->on_progress(p);
    last_callback_ = std::chrono::steady_clock::now();
}

void ViolationLSLoop::note_first_feasible(double obj) {
    // The time is the latch, not the objective: `obj` is allowed to be NaN here
    // (the #100 witness path can hand us one), so testing it would re-arm the
    // latch on every later feasible point and record the last one instead of
    // the first.
    if (!std::isnan(first_feasible_time_)) {
        return;
    }
    first_feasible_time_ = elapsed();
    first_feasible_obj_ = obj;
}

bool ViolationLSLoop::record_best() {
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
        if (have_feasible_) {
            return false;  // already have a witness, and possibly a better one
        }
        have_feasible_ = true;
        note_first_feasible(obj);
        best_state_ = model_.copy_state();
        // Both have_feasible_ transitions release the adopted origin, not just
        // the finite one below: this is still "the search got somewhere of its
        // own", and leaving it set would keep kicking from a peer's point while
        // best_state_ holds a witness this run found itself (#158).
        has_adopted_origin_ = false;
        adopted_origin_ = Model::State{};
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
        if (has_obj_ && !std::isfinite(model_.objective_bound())) {
            model_.set_objective_bound(kInfPenalty);  // the shared clamp; see violation.h
        }
        // Shared as +inf, not as `obj`: `obj` is non-finite here and the pool
        // sorts on the objective, where a NaN would make the comparator
        // inconsistent and the sort undefined. +inf is what best_feasible_obj_
        // still reads, and it puts the witness last among the feasible entries
        // -- which is exactly its standing.
        share(std::numeric_limits<double>::infinity());
        emit_progress(/*new_best=*/true);
        return true;
    }
    // isfinite(best_feasible_obj) guards the case where the incumbent is the
    // +inf witness above: the relative-improvement test would compute
    // `inf - inf` = NaN and decide by NaN comparison.
    if (have_feasible_ && std::isfinite(best_feasible_obj_) &&
        obj >= best_feasible_obj_ - (1e-12 * (std::abs(best_feasible_obj_) + 1.0))) {
        return false;
    }
    have_feasible_ = true;
    note_first_feasible(obj);
    best_feasible_obj_ = obj;
    best_state_ = model_.copy_state();
    // Wherever the search was told to explore from, it has now got somewhere
    // better, so the incumbent is the kick origin again (#158) and the adopted
    // one is released.
    has_adopted_origin_ = false;
    adopted_origin_ = Model::State{};
    if (has_obj_) {
        // The bound step doubles as the Newton step size toward the objective
        // (the float jump chases obj <= bound), so it must be non-trivial for
        // hook-less continuous descent.
        double eps = 1e-3 * (std::abs(obj) + 1.0);
        model_.set_objective_bound(obj - eps);
    }
    share(obj);
    emit_progress(/*new_best=*/true);
    return true;
}

// Requirement: submit when found, not at the end. The cost is ONE Model::State
// copy -- `best_state_` is the loop's own incumbent and must survive, so it is
// copied here and then MOVED into the pool's vector -- plus one
// uncontended-in-the-common-case mutex, plus the O(#constraints) residual scan
// the pooled solution's `violation` needs, per NEW BEST, i.e. per improving batch
// of 1000 GLS iterations. Nowhere near the hot path. The move matters at
// 32 workers on a large model: `submit` takes its argument by value precisely
// so the second copy happens out here rather than inside the critical section.
void ViolationLSLoop::share(double objective) {
    if (coord_ == nullptr || coord_->pool == nullptr) {
        return;
    }
    Solution sol;
    sol.state = best_state_;
    sol.objective = objective;
    sol.feasible = true;  // record_best's precondition
    // The model still holds `best_state_` -- record_best copies the state out of
    // it immediately above every call to this -- so the live residual is the
    // residual of the state being shared.
    sol.violation = max_real_violation();
    coord_->pool->submit(std::move(sol));
}

// Requirement: restart a stalled worker from the SHARED pool rather than only
// from its own assignment. Returns false having changed nothing when there is
// nothing to adopt, so the caller falls through to the ordinary kick.
bool ViolationLSLoop::holds_assignment(const Model::State& state) const {
    const auto& vars = model_.variables();
    for (size_t i = 0; i < vars.size(); ++i) {
        // Bit equality, not a tolerance: this asks "is this literally the state
        // we are sitting on", which is what a restored copy of our own snapshot
        // is. A near-miss is a different point and a legitimate restart.
        if (vars[i].value != state.values[i] || vars[i].elements != state.elements[i]) {
            return false;
        }
    }
    return true;
}

// The objective-bound half of an adoption, lifted out of adopt_from_pool: it is
// a self-contained responsibility with its own rule, and leaving it inline put
// that function over the cognitive-complexity threshold once the kick-origin
// bookkeeping joined it. `feasible_here` and `obj` are this model's verdict on
// the adopted point, not the submitter's.
void ViolationLSLoop::reground_objective_bound_after_adoption(bool feasible_here, double obj) {
    if (!has_obj_) {
        return;
    }
    // Never LOOSEN it. record_best is the only other writer and it rewrites
    // the bound only on a STRICT improvement over best_feasible_obj_ (it
    // returns early otherwise), so a bound loosened here can never tighten
    // back until this worker beats its own all-time best -- and the draw
    // that got us here is frequently one of our own earlier, worse
    // incumbents. Loosening on every such kick would leave the worker
    // searching with the objective row satisfied and no pressure at all.
    //
    // LNS is the precedent, and it is unambiguous: destroy_repair replaces
    // the assignment wholesale and does not touch the bound. A row reading
    // `obj <= best - eps` that is violated at the point we just arrived on
    // is ViolationLS's normal steady state, not a problem to fix.
    double target = std::numeric_limits<double>::infinity();
    if (feasible_here && std::isfinite(obj)) {
        target = obj;
    }
    if (have_feasible_ && std::isfinite(best_feasible_obj_)) {
        target = std::min(target, best_feasible_obj_);
    }
    if (std::isfinite(target)) {
        model_.set_objective_bound(target - (1e-3 * (std::abs(target) + 1.0)));
    } else if (!std::isfinite(model_.objective_bound())) {
        // No usable objective anywhere yet -- an infeasible or non-finite
        // adopted point with no incumbent of our own. Install the finite
        // sentinel so the row is not vacuous, under record_best's own guard
        // so it can never overwrite a real incumbent's bound.
        model_.set_objective_bound(kInfPenalty);  // the shared clamp; see violation.h
    }
}

bool ViolationLSLoop::adopt_from_pool() {
    if (coord_ == nullptr || coord_->pool == nullptr) {
        return false;
    }
    auto sol = coord_->pool->get_restart_point(rng_);
    if (!sol.has_value()) {
        return false;
    }
    // The pool is shared across workers whose models come from one factory, so
    // the shapes agree by construction -- but the factory is caller-supplied
    // and nothing makes it return the same model twice. A mismatched state
    // would be restored element-wise into the wrong variables and silently
    // searched from, so refuse it instead. Model::restore_state throws on a
    // width mismatch and elements is indexed without one, so this is the
    // difference between a refused restart and a worker lost to an exception.
    if (sol->state.values.size() != model_.num_vars() ||
        sol->state.elements.size() != model_.num_vars()) {
        return false;
    }

    // A draw equal to the assignment we already hold is not a kick: restoring
    // where you already are moves nothing, while zeroing stagnation_ and so
    // denying the ordinary kick its turn. Refusing here makes the caller fall
    // through to diversify().
    //
    // Against the LIVE assignment, deliberately, not against best_state_: the
    // question is whether this restore would move the search, and after
    // perturbation_period non-improving batches the model generally sits
    // somewhere FJ left it rather than on its own recorded incumbent. Comparing
    // against best_state_ would refuse draws that are genuine moves.
    //
    // How often it fires is model-dependent and NOT the common case, though an
    // earlier version of this comment claimed it was. Measured on a 20-variable
    // integer model over 12 seeds, a worker whose pool receives only its own
    // submissions declined the draw in 3 of 12 runs; on an easier asymmetric
    // variant it declined every time. So this is a guard against a degenerate
    // kick, not a routine path.
    if (holds_assignment(sol->state)) {
        return false;
    }

    model_.restore_state(sol->state);
    // Mandatory: restore_state writes the VARIABLES, leaving every DAG node at
    // the previous assignment. Everything below -- the violation manager, FJ's
    // violated set and scan set -- reads node values. Same sequence LNS uses
    // either side of a repair (src/lns.cpp:107,118).
    full_evaluate(model_);

    // THIS model's verdict, not the submitter's. `sol->feasible` describes
    // whatever model the submitting worker searched, and the factory is
    // caller-supplied -- nothing makes two workers' models identical, which is
    // the same reason the shape guard above exists. Trusting the flag would set
    // have_feasible_ (and with it best_state_, which finish() returns) on a
    // point that violates this model's constraints. The full_evaluate above is
    // what makes the local check both correct and cheap.
    const bool feasible_here = real_feasible();
    const double obj = current_obj();

    // The bound is a Const node, so Model::State does not carry it and we are
    // still holding the one our own incumbent earned. Left alone it would put
    // the artificial `obj <= bound` row in violation the instant we arrive on a
    // point worse than our own best, and FJ would spend the restart climbing
    // straight back to the basin we just left. So re-ground it on the adopted
    // point, exactly as record_best does -- under both of that function's
    // rules: a bound is derived only from a FEASIBLE point, and never from a
    // non-finite objective. An infeasible point (the pool also holds
    // closest-approach states, submitted at the end of a run that never reached
    // feasibility) typically has a *better* objective than any feasible one, so
    // deriving from it would install a bound nothing feasible can meet and
    // leave the artificial row violated for the rest of the run. The loosest
    // finite bound is the honest answer there.
    reground_objective_bound_after_adoption(feasible_here, obj);

    // Only when it IMPROVES. A worker ahead of the pool restarts from a
    // diverse point without losing the incumbent it will return: finish()
    // hands back best_state_, so overwriting it with a worse adopted point
    // would throw away this worker's own best work.
    if (feasible_here && std::isfinite(obj) &&
        (!have_feasible_ || !std::isfinite(best_feasible_obj_) || obj < best_feasible_obj_)) {
        have_feasible_ = true;
        note_first_feasible(obj);
        best_feasible_obj_ = obj;
        // Moved: `sol` is a by-value optional from get_restart_point and its
        // state was last read by the restore above, so this saves a whole
        // Model::State copy -- one vector<double> plus one vector<vector<int32_t>>
        // per variable -- on a model the size of atlanta-ip.
        best_state_ = std::move(sol->state);
        // It IS the incumbent now, so kick_origin() finds it there.
        has_adopted_origin_ = false;
        adopted_origin_ = Model::State{};
    } else if (feasible_here && std::isfinite(obj)) {
        // Not an improvement, so best_state_ is still ours -- but the search now
        // stands on the adopted point, and that is what a later kick must depart
        // from. Without this the next kick restores our own incumbent and the
        // draw is discarded within a few batches, which is the whole of what the
        // pool bought (#158, #135).
        //
        // Under the SAME two rules the improvement branch above applies, and
        // they are load-bearing rather than symmetry: the pool also holds
        // closest-approach states submitted with feasible = false at the end of
        // a run that never got there (see run_worker), and #100's witness is
        // shared with a non-finite objective. Either would anchor ~98% of this
        // worker's remaining kicks -- only the full-period route adopts, and
        // that route is 2% of kicks -- to a point that is infeasible or has no
        // objective, while the incumbent finish() returns would never be kicked
        // from again. That is the inverse of replacing the starting point with a
        // known-GOOD state.
        adopted_origin_ = std::move(sol->state);  // moved, as above
        has_adopted_origin_ = true;
    } else {
        // An infeasible or non-finite-objective draw is a legitimate diverse
        // restart -- adopt_from_pool has already installed it as the live
        // assignment -- but it is not a point to keep returning to. Release any
        // older origin so kick_origin() falls back to the incumbent.
        has_adopted_origin_ = false;
        adopted_origin_ = Model::State{};
    }

    if (!has_obj_ && feasible_here) {
        // Pure feasibility: the point we just adopted IS the answer, and this
        // worker can no longer reach apply_batch_outcome's Feasible exit --
        // have_feasible_ is now true and current_obj() is a constant 0.0, so
        // record_best's improvement test refuses every later batch. Say so
        // here, and raise the flag for the same reason run_worker does: the
        // question is settled for everyone.
        termination_ = TerminationReason::Feasible;
        if (coord_->stop != nullptr) {
            coord_->stop->store(true, std::memory_order_relaxed);
        }
    }

    vm_.invalidate_cache();
    // reset_weights, not resync: the GLS weights we hold were shaped by the
    // basin we are leaving, and this is the "state mutated outside GFJ" case
    // the LNS call site uses reset_weights for. NOT fj_.begin() -- that zeroes
    // iterations() (breaking the iteration budget and the reported count) and
    // re-arms a full fresh wall clock past solve()'s own deadline.
    fj_.reset_weights();
    // The caller armed the Float escape probe just before calling us, on the
    // diagnosis that some variable sits at a stationary point of every violated
    // constraint. That diagnosis was about the assignment we have just left; it
    // says nothing about the one we adopted, and the probe is not free. Disarm
    // it and let the two documented arming routes re-earn it from the new
    // point. (A kick that adoption declines keeps the armed probe. Before the
    // first feasible point that is diversify() perturbing the stuck assignment
    // itself, as it always did; after it, diversify() restores kick_origin()
    // first (#158), so the probe is armed on a diagnosis about the assignment
    // being left and applied from the point the search departs from. Harmless
    // rather than justified by the two being one basin -- the drift measurement
    // at diversify() says they are NOT: a post-kick assignment holds a
    // real-feasible point only 5.3% of the time. It is harmless because the flag
    // only ENABLES the probe: compute_var_jump re-tests `!gradient_usable` at the
    // live point, so the probe re-earns itself at the restored one.
    // test_parallel's self-draw case pins it staying armed there.)
    fj_.set_escape_probe(false);
    // ...and re-ground the clock the TIME-based route arms on, or the disarm
    // above lasts exactly one batch: last_improvement_ still describes the
    // stall we just left, so maybe_arm_escape_probe re-arms on the next batch.
    // That would also falsify the "at most 1/kEscapeArmFraction arms per run"
    // bound that route's own comment rests on, since adoption is a disarm with
    // no matching improvement.
    if (has_deadline_) {
        last_improvement_ = std::chrono::steady_clock::now();
    }
    sample_rho();
    ++perturbations_;
    // The slot counts kicks ELIGIBLE for an LNS repair, and this was one: the
    // full-period route always advances it, whether the kick was served by
    // adoption or by diversify(). Not advancing it starves the cadence rather
    // than preserving it -- lns_slot_ would sit at 0 forever, lns_kick_due()
    // would never come true, and the stand-down above would never fire. (This
    // is not the case the counter's separation from `perturbations` guards
    // against: that is about #102's route being REFUSED its LNS half, which is
    // a different route and still does not advance the slot.)
    ++lns_slot_;
    stagnation_ = 0;
    return true;
}

void ViolationLSLoop::diversify(bool allow_lns) {
    // KICK FROM THE INCUMBENT, not from wherever the last kick left the search
    // (#158).
    //
    // A diversification kick exists to leave the basin the search has settled
    // into, and both halves of it are written as though the assignment it acts
    // on were that settled point: perturb() randomises a fraction of it and
    // expects the rest to be worth keeping, and LNS::destroy_repair scores its
    // repair against the key of the state it was handed. That assumption holds
    // for the FIRST kick after an improvement and for nothing after it, because
    // nothing in a single-threaded run ever restores best_state_. Once a kick
    // fails to recover, the next one starts from the failure, and the kicks
    // compose into a random walk AWAY from the best solution found -- which is
    // drift, not diversification.
    //
    // Measured on the mipfeas smoke instances at 60s, 7 instances x 2 seeds
    // (#158): 13 487 kicks, of which 98% come from the unproductive-batch route
    // and land a median of 2 batches apart; over the batches following a kick
    // the assignment holds a real-feasible point only 5.3% of the time; and over
    // 1 216 windows of >= 5 batches the MEDIAN of (closest approach back to the
    // pre-kick point) / (kick distance) is 1.00 -- on at least half the windows
    // the assignment never got any nearer than the kick itself left it. So on
    // the typical kick the search left the feasible region and did not come
    // back: on binkar10_1 it reaches a real violation <= 1 at 26% of kicks and
    // lands feasible on 3 of 567 post-kick batches.
    //
    // THIS IS A DELIBERATE DIVERGENCE FROM THE REFERENCE, not a bug fix, and
    // CLAUDE.md's Reference Correctness rule governs it -- so read the paper
    // first. Davies et al., Algorithm 6 (docs/) has exactly one restore:
    //
    //     5   if a new best solution S is available in the shared pool then
    //     6       X <- S
    //    11   if No new solutions found or imported for 100 iterations then
    //    12       Perturb X, randomising each variable's value with probability 0.1
    //
    // Line 12 perturbs X IN PLACE. Line 6 is pool-sourced and conditioned on a
    // new BEST arriving, which is what adopt_from_pool implements. There is no
    // step that restores an incumbent before a kick, so kicks composing into a
    // walk is the reference's actual behaviour and not an oversight in this
    // port. What this makes the outer loop is elitist ILS with a
    // strict-improvement acceptance criterion -- and since perturb() ends in
    // reset_weights(), neither the assignment nor the GLS landscape carries
    // across a kick.
    //
    // The case for diverging is that the port ALREADY diverges in the direction
    // that makes the walk pathological. Algorithm 6 kicks once per 100 stagnant
    // batches; #102's unproductive route kicks at a median of 2, which is ~50x
    // the reference rate, and 98% of kicks come through it. A walk sampled 50x
    // more often is a much longer walk, and the measurement above is what it
    // does. `adopt_from_pool()` is the same replace-the-starting-point move in
    // the one place the paper sanctions it, and it needs a pool; this is that
    // move where there is none. Gated on have_feasible_ because before the
    // first feasible point best_state_ is still the initial assignment, which
    // is not a point to return to; that regime keeps the trajectory it had.
    //
    // What it costs is recorded rather than argued: see docs/architecture.md's
    // Diversification section for the MINLPLib arm, where the instances that
    // use post-feasible kicks to move between feasible integer combinations
    // pay for the elitism. If that cost ever outweighs the gain, the more
    // faithful thing to attack is the kick RATE, not this restore.
    //
    // WHICH point, in a portfolio, is `kick_origin()`: our own incumbent
    // normally, and the state adoption installed when the search is standing on
    // a peer's point instead. Restoring best_state_ unconditionally would undo
    // adopt_from_pool on the very next kick -- the pool draws from the better
    // HALF so that workers stay spread, and a non-improving draw deliberately
    // does not become best_state_, so an adopted point would survive about five
    // batches and the portfolio would degenerate to N elitist searches. See
    // kick_origin().
    //
    // The restore is cheap enough for #102's fast-recurring route, which is what
    // maybe_diversify's "not the unproductive one" note is about: what it prices
    // there is an ADOPTION -- a pool draw under a mutex, a restore, a
    // full_evaluate and an FJ rebuild. This is the restore alone. The
    // full_evaluate below belongs to the LNS branch and not to this one, and the
    // FJ rebuild was already being paid: perturb() ends with reset_weights().
    //
    // restore_state writes the VARIABLES and leaves every DAG node on the
    // previous assignment, so whoever reads node values next needs a
    // full_evaluate first. Only one of the two branches is such a reader:
    // LNS::destroy_repair keys the state it is about to destroy off node values.
    // perturb() reads none -- it draws from the DOMAINS, and apply_move only
    // writes -- and it ends with a full_evaluate of its own, so a sweep here too
    // would be a second whole-DAG walk per kick on the route that takes 98% of
    // them, computing values nothing reads. Skipping it changes no trajectory:
    // full_evaluate draws no random numbers and writes no variable.
    const Model::State* origin = kick_origin();
    if (origin != nullptr) {
        model_.restore_state(*origin);
    }
    if (allow_lns && lns_ != nullptr && lns_interval_ > 0 &&
        (lns_slot_ % lns_interval_ == lns_interval_ - 1)) {
        if (origin != nullptr) {
            full_evaluate(model_);  // see above: destroy_repair reads node values
            // violation.h's rule: the cached total is stale after a
            // full_evaluate. Immaterial to the built-in LNS, which keys off node
            // values directly, but destroy_repair is a documented extension
            // point and must not be handed a model and a cache that disagree.
            vm_.invalidate_cache();
        }
        // Bound the repair by whatever budget is left, so an LNS kick near
        // the deadline cannot run its own independent 2s.
        // Floored at a tiny positive value while a deadline exists:
        // remaining() returns exactly 0.0 if the clock crossed the deadline
        // since the past_deadline() check above, and 0 means "no wall-clock
        // limit" downstream in fj_nl_initialize — the opposite of intent.
        const double repair_limit =
            has_deadline_ ? std::max(1e-9, std::min(2.0, remaining())) : 0.0;
        // What the built-in LNS now scores against: the restore above means
        // `old_key` is the kick origin's key, so a repair is kept only if it
        // beats the point the kick departed from rather than a drifted one it
        // was handed. That is the rule SearchResult::lns_repairs_accepted has
        // always been documented as recording, and it is a strictly higher bar.
        //
        // What a REJECTION does is worth stating precisely, because it inverted
        // with this change. Before, destroy_repair snapshotted the drifted point
        // the kick started from, so a rejection put the search back exactly
        // where it already was: a genuine no-op. Now the restore above has
        // already happened, so the snapshot IS the origin, and a rejection moves
        // the search from wherever it had drifted back to the origin and
        // perturbs nothing. Not a no-op -- but not a diversification either: FJ
        // re-descends from a point it has already converged on. Whether such a
        // kick should fall through to a perturb is a trajectory change and so a
        // separate measurement, not a tidy-up.
        const bool accepted = lns_->destroy_repair(model_, vm_, rng_, repair_limit);
        ++lns_repairs_;
        lns_repairs_accepted_ += accepted ? 1 : 0;
        fj_.reset_weights();  // LNS mutated state outside GFJ
    } else {
        fj_.perturb(config_.perturbation_probability);  // self-resyncs
    }
    sample_rho();
    ++perturbations_;
    if (allow_lns) {
        ++lns_slot_;
    }
    stagnation_ = 0;
}

bool ViolationLSLoop::budget_exhausted() {
    // Count *actual* GLS iterations, which is what the config documents and
    // what SearchResult::iterations reports. Using batches *
    // batch_iterations over-counts whenever a batch exits early on
    // feasibility, so the budget expired after far less work than asked for.
    if (config_.max_iterations > 0 && fj_.iterations() >= config_.max_iterations) {
        termination_ = TerminationReason::IterationLimit;
        return true;
    }
    // Structural and Novelty batches do not charge fj.iterations(), so on a
    // List/Set model with no wall clock the iteration budget alone cannot
    // guarantee termination (structural_batch_probability = 1.0 would spin
    // forever). Batches <= iterations by construction, so this only bites
    // when iterations have stalled.
    if (config_.max_iterations > 0 && batches_ >= config_.max_iterations) {
        termination_ = TerminationReason::IterationLimit;
        return true;
    }
    if (!has_deadline_ && config_.max_iterations <= 0) {
        // Neither budget set: nothing would ever stop the loop.
        termination_ = TerminationReason::NoBudget;
        return true;
    }
    return false;
}

BatchKind ViolationLSLoop::pick_batch_kind() {
    if (rng_.random() < structural_probability_) {
        return BatchKind::Structural;
    }
    if (config_.use_compound_moves && rng_.random() < config_.novelty_jump_probability) {
        return BatchKind::NoveltyJump;
    }
    return BatchKind::FeasibilityJump;
}

bool ViolationLSLoop::run_batch(BatchKind kind) {
    switch (kind) {
        case BatchKind::Structural:
            return structural_pass(model_, vm_, rng_, has_deadline_, deadline_);
        case BatchKind::NoveltyJump:
            fj_.apply_novelty_jump();
            return true;
        case BatchKind::FeasibilityJump:
            fj_.batch(config_.batch_iterations);
            return false;
    }
    return false;
}

void ViolationLSLoop::note_closest_approach(double batch_violation) {
    // Only tracked until the first feasible solution: after that both the
    // final restore and the returned state use best_state, so the snapshot
    // would be pure allocation on every improving batch. The `batches == 1`
    // clause guarantees one capture even on an all-NaN run (violation stays
    // +inf, so `<` never fires) without re-snapshotting on every batch of a
    // violation plateau — the common infeasible case.
    if (!have_feasible_ && (batch_violation < best_violation_ || batches_ == 1)) {
        best_violation_ = batch_violation;
        closest_state_ = model_.copy_state();
    }
}

bool ViolationLSLoop::polish_and_record(double batch_violation, bool& resync) {
    if (batch_violation > config_.feasibility_tolerance) {
        return false;
    }
    // Record the feasible point we already have *before* polishing. The
    // hook descends the penalty-method objective and can land outside
    // the feasible region; recording only afterwards silently threw away
    // genuinely feasible solutions (an instance would be reported
    // infeasible despite the search having visited a feasible point).
    bool improved = record_best();
    // The hook is unbounded in *time* — a custom InnerSolverHook may do
    // arbitrary work, and even FloatIntensifyHook sweeps every Float
    // max_sweeps times. Don't start one we have no budget for.
    if (hook_ != nullptr && !past_deadline()) {
        hook_->solve(model_, vm_, {});  // continuous-objective polish (mutates floats)
        resync = true;
        if (real_feasible()) {  // keep the polish only if it stayed feasible
            improved = record_best() || improved;
        }
    }
    return improved;
}

bool ViolationLSLoop::apply_batch_outcome(bool improved, bool resync) {
    if (!improved) {
        ++stagnation_;
        if (resync) {
            fj_.resync();  // re-sync after hook/structural mutation, keep GLS weights
        }
        return true;
    }
    // Gated so an iteration-budgeted run reads no clock AT ALL, not merely
    // no clock that reaches control flow: this is the only writer of
    // last_improvement and its only reader is the has_deadline-gated
    // arming block below. Keeps architecture.md's "the loop reads no
    // clock at all" literally true.
    if (has_deadline_) {
        last_improvement_ = std::chrono::steady_clock::now();
    }
    stagnation_ = 0;
    // Making progress: the Float escape probe is not needed and is not free.
    fj_.set_escape_probe(false);
    fj_.reset_weights();  // new best: fresh GLS weights (paper) + new rho
    sample_rho();
    if (!has_obj_) {
        // Pure feasibility: first solution is the answer.
        termination_ = TerminationReason::Feasible;
        return false;
    }
    return true;
}

void ViolationLSLoop::maybe_arm_escape_probe() {
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
    if (has_deadline_ && !fj_.escape_probe()) {
        const auto now = std::chrono::steady_clock::now();
        if (now < deadline_ && std::chrono::duration<double>(now - last_improvement_).count() >=
                                   kEscapeArmFraction * budget_seconds_) {
            fj_.set_escape_probe(true);
        }
    }
}

void ViolationLSLoop::maybe_diversify(BatchKind kind, bool improved) {
    // A Feasibility-Jump batch that reported itself unproductive stopped
    // reducing the real rows' violation at all (GFJConfig::
    // unproductive_iterations). Waiting out the rest of perturbation_period
    // would be waiting for a batch that has already said it has nothing
    // left, so the kick is due now (#102).
    //
    // Gated on !improved. FeasibilityJump ends a batch on ITS measure, the
    // real rows; the outer loop's `improved` is a new best on the objective.
    // After the first feasible point those come apart routinely: a batch
    // satisfies every real row, keeps iterating because any_active_violated()
    // still sees the artificial objective row, plateaus on the real rows and
    // exits stuck -- having just recorded a new best. Without this guard that
    // batch is kicked anyway, three lines after the `improved` block set
    // stagnation to 0 and disarmed the escape probe. That would also falsify
    // the reasoning the time-based arming route above rests on ("the
    // improvement that resets stagnation also disarms the probe"), by
    // re-arming the probe on the very batch that just improved.
    //
    // batch_stuck() can now only be true when the batch started with
    // `stagnation >= unproductive_arm_stagnation`, so this is the second of
    // two gates rather than the only one; it is kept because it reads on the
    // state AFTER the batch, which the arming decision could not.
    const bool unproductive_kick =
        kind == BatchKind::FeasibilityJump && !improved && fj_.batch_stuck();
    if (stagnation_ >= config_.perturbation_period && !past_deadline()) {
        // Genuinely stuck. Arm the Float escape probe: a variable sitting at a
        // stationary point of every violated constraint has no other candidate
        // that can move it, and diversification alone cannot rescue it because
        // the search re-converges to the same point. Disarmed again on the next
        // improvement, so a productive search never pays for it.
        fj_.set_escape_probe(true);
        // Under ParallelSearch, restarting from a PEER's incumbent is MEANT to
        // be a better use of a full-period kick than re-perturbing our own
        // assignment: the pool holds points this worker has not seen, drawn
        // from the better half rather than the best so the workers stay spread.
        // Nothing here measures that, and it should not be read as settled --
        // "cooperative vs non-cooperative sharing at a fixed worker count" is
        // an open acceptance criterion on issue #135, and a null
        // SearchCoordination* is exactly how to run the control arm. adopt_from_pool() performs the
        // kick itself -- it disarms the probe it just armed, resamples rho, counts the perturbation
        // and zeroes stagnation -- so diversify() is skipped when it succeeds. With no pool it
        // returns false immediately and this is the same single-threaded code path it always was.
        //
        // Deliberately only this route, not the unproductive one below: that
        // kick fires on an iteration count, can recur every batch, and is
        // valuable precisely because it is cheap. An adoption is a pool draw
        // under a mutex plus a state restore plus a full_evaluate plus an FJ
        // rebuild, and putting it on the fast-recurring route would turn a stall
        // accelerator into a thrash.
        //
        // #158 put a restore on that route, which is NOT the same move and the
        // difference is the whole of this paragraph: it takes no lock, draws
        // nothing, needs no full_evaluate on the perturb branch, and the FJ
        // rebuild was already being paid by perturb()'s own reset_weights. What
        // is left is one per-variable copy. See diversify().
        //
        // `lns_kick_due()` FIRST, and it is load-bearing rather than a nicety.
        // diversify() is the only caller of LNS::destroy_repair and the only
        // writer of lns_slot_, and this route is the only one that may draw LNS
        // once a feasible solution exists (the other passes
        // allow_lns = !have_feasible_). So letting adoption take every kick
        // would silently switch LNS off for the whole of every portfolio
        // worker's life from its own first incumbent onward -- `cbls --lns 0.3`,
        // now multi-threaded by default, would build an LNS per worker and
        // never repair with it, with `lns_repairs` among the fields the parallel
        // compose drops so nothing would say so. Adoption replaces the PERTURB
        // half of the kick, not the destroy-repair half.
        if (lns_kick_due() || !adopt_from_pool()) {
            diversify();
        }
    } else if (unproductive_kick && !past_deadline()) {
        // Kick early, but do NOT arm the escape probe and do NOT reset the
        // stagnation counter.
        //
        // Not the probe: this route fires on an ITERATION count, so on a
        // model with no reachable feasible point the very first batch trips
        // it and every batch after it does too. Arming here would put the
        // probe on from ~300 iterations into the run onward and leave it on
        // -- the always-on regime #107 measured at a 9x regression -- and
        // the mitigation that makes arming safe elsewhere (disarm on the
        // next improvement) is worth nothing against a condition that
        // recurs every batch. The probe keeps its two documented routes.
        //
        // Not the counter: diversify() zeroes `stagnation`, and an
        // unproductive batch is by definition a non-improving one, so
        // letting it zero the counter would mean a model that is
        // unproductive every batch never reaches perturbation_period --
        // which is the probe's own arming clock, and on an iteration-
        // budgeted run (no wall clock) the only one it has. Carrying the
        // count across keeps "100 non-improving batches" meaning what it
        // says while still buying the early kick.
        const int carried = stagnation_;
        diversify(/*allow_lns=*/!have_feasible_);
        stagnation_ = carried;
    }
}

void ViolationLSLoop::maybe_emit_periodic_progress() {
    // Periodic progress (~1s) even without improvement.
    if (callback_ != nullptr &&
        std::chrono::duration<double>(std::chrono::steady_clock::now() - last_callback_).count() >=
            1.0) {
        emit_progress(/*new_best=*/false);
    }
}

SearchResult ViolationLSLoop::finish() {
    // On a feasible run the best-objective assignment is the answer; on an
    // infeasible one, hand back the closest approach rather than the initial
    // assignment (which carries no information about where the search got to).
    model_.restore_state(have_feasible_ ? best_state_ : closest_state_);
    // Release the artificial objective bound so post-solve feasibility checks
    // (verifiers iterating model constraints) don't see it violated by ~eps.
    if (has_obj_) {
        model_.set_objective_bound(std::numeric_limits<double>::infinity());
    }
    full_evaluate(model_);

    // Sampled here rather than at the assignment below, so the reported wall
    // time does not depend on how many cheap fields the record grows.
    const double run_seconds = elapsed();

    SearchResult result;
    result.objective =
        have_feasible_ ? best_feasible_obj_ : std::numeric_limits<double>::infinity();
    result.feasible = have_feasible_;
    result.best_state = have_feasible_ ? best_state_ : closest_state_;
    // Residual of the assignment actually being returned, from the fresh
    // full_evaluate above rather than the incrementally-maintained node values.
    result.best_violation = max_real_violation();
    result.iterations = fj_.iterations();  // total GLS iterations (not batch count)
    result.time_seconds = run_seconds;
    result.termination = termination_;
    result.escape_probe_armed = fj_.escape_probe();
    result.perturbations = perturbations_;
    result.lns_repairs = lns_repairs_;
    result.lns_repairs_accepted = lns_repairs_accepted_;
    result.first_feasible_objective = first_feasible_obj_;
    result.time_to_first_feasible = first_feasible_time_;
    return result;
}

SearchResult ViolationLSLoop::run() {
    while (!past_deadline()) {
        if (budget_exhausted()) {
            break;
        }

        const BatchKind kind = pick_batch_kind();
        // Re-decided every batch, from the count as it stands BEFORE the batch
        // runs; see unproductive_arm_stagnation above. Disarming also stops the
        // batch ENDING early, which matters on its own: on ex8_6_1 the early
        // exits cost about six gap points even with every kick suppressed.
        //
        // The rule is the stagnation count ALONE. Feasibility does not gate
        // arming or the kick; it decides only whether the kick may draw LNS (see
        // the kick site). Two alternatives were implemented and measured and both
        // are worse, so do not "restore" either from these notes:
        //
        //  - Gating the kick on !have_feasible bounds the misfire completely, but
        //    st_e40 then reaches its BKS on 2 of 4 seeds instead of 4 -- it uses
        //    post-feasible kicks to move between its 52 feasible combinations.
        //  - Rate-limiting the kick (arming on the advance since the last one)
        //    starves it the other way: infeasible at seed 2 on the 15 000-
        //    iteration regression budget.
        //
        // Note what this count does NOT do: it does not bound the number of
        // kicks. The kick site restores `stagnation` deliberately, so once the
        // count first crosses the threshold every later batch is armed until an
        // improvement or a full-period diversify resets it. That is a DELAY on
        // the first misfire, not a cap on the rate; what makes the misfire cheap
        // is that the kick drops its LNS half once a feasible solution exists.
        fj_.set_watch_progress(stagnation_ >= unproductive_arm_stagnation_);

        bool resync = run_batch(kind);
        ++batches_;

        const double batch_violation = max_real_violation();
        note_closest_approach(batch_violation);
        const bool improved = polish_and_record(batch_violation, resync);
        if (!apply_batch_outcome(improved, resync)) {
            break;
        }
        maybe_arm_escape_probe();
        maybe_diversify(kind, improved);
        maybe_emit_periodic_progress();
    }
    // The while-condition is the only exit that does not assign termination_,
    // so reaching here with the seeded TimeLimit means the loop condition ended
    // the run -- and that condition has two causes now. The clock is asked
    // first: a run whose budget genuinely expired is time-limited whether or
    // not a peer happened to raise the flag in the same instant. Every other
    // exit is a `break` that already wrote its own reason (Feasible above all:
    // a worker that solved the model did not stop, it finished).
    if (termination_ == TerminationReason::TimeLimit && stop_requested() && !clock_expired()) {
        termination_ = TerminationReason::Stopped;
    }
    return finish();
}

}  // namespace

// ViolationLS (paper Algorithm 6): the objective is folded into the constraint
// set as `obj <= bound`; GFJ batches drive the assignment to feasibility while
// the bound is tightened on each new (real-)feasible solution. On stagnation the
// assignment is perturbed (or diversified via LNS). The InnerSolverHook polishes
// continuous variables (objective descent) on each feasible solution.
SearchResult solve(Model& model, double time_limit, uint64_t seed, bool use_fj,
                   InnerSolverHook* hook, LNS* lns, int lns_interval, SolveCallback* callback,
                   const SearchConfig& config, SearchCoordination* coord) {
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

    const Budget budget = make_budget(time_limit);

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
    gfj.time_limit = budget.seconds;  // saturated: begin() casts to integer ticks
    gfj.max_iterations = 0;
    gfj.unproductive_iterations = config.unproductive_iterations;
    FeasibilityJump fj(model, vm, rng, gfj);
    fj.begin(/*set_initial_x=*/!config.skip_init);

    ViolationLSLoop loop(model, vm, rng, fj, config, budget, hook, lns, lns_interval, callback,
                         coord);
    return loop.run();
}

}  // namespace cbls
