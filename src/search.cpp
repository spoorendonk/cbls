#include "cbls/search.h"

#include "cbls/audit.h"
#include "cbls/dag_ops.h"
#include "cbls/feasibility_jump.h"
#include "cbls/randomize.h"
#include "cbls/structural_batch.h"
#include "cbls/tracer.h"
// The SearchCoordination definition search.h forward-declares: ViolationLSLoop
// reads both of its channels.
#include "cbls/solution_pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cbls {

SolveCallback::~SolveCallback() = default;
SearchAuditor::~SearchAuditor() = default;

namespace {

// Whether `v` is its partition's lowest-numbered member, where the sweep lays the
// whole partition out once. `list_ids` keeps the caller's order, hence
// `min_element`, not `front()`.
bool is_partition_anchor(const ListPartition& part, int32_t v) {
    return !part.list_ids.empty() &&
           *std::min_element(part.list_ids.begin(), part.list_ids.end()) == v;
}

// Both initialisers, over the variables `want` accepts. A partition member never
// goes through `randomize_var`, which ignores its siblings and can put an element
// in several lists or none: under `Cover::Exact` no move repairs that (#164). A
// model with no partition skips the lookup, so its draw sequence is unchanged.
template <typename Want>
void initialize_over(Model& model, RNG& rng, Want want) {
    const bool has_partitions = !model.list_partitions().empty();
    for (int32_t v = 0; v < static_cast<int32_t>(model.num_vars()); ++v) {
        Variable& var = model.var_mut(v);
        if (!want(var)) {
            continue;
        }
        const int part = has_partitions ? model.partition_of_list(v) : -1;
        if (part < 0) {
            randomize_var(var, rng);
            continue;
        }
        const ListPartition& partition = model.list_partitions()[static_cast<size_t>(part)];
        if (is_partition_anchor(partition, v)) {
            randomize_list_partition(model, partition, rng);
        }
    }
}

}  // namespace

void initialize_random(Model& model, RNG& rng) {
    initialize_over(model, rng, [](const Variable&) { return true; });
}

void initialize_structured_random(Model& model, RNG& rng) {
    initialize_over(model, rng, [](const Variable& var) { return is_structured(var.type); });
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
        case TerminationReason::Cancelled:
            return "cancelled";
    }
    // Unreachable for any enum value; keeps the function total.
    return "unknown";
}

// Refines the current assignment toward feasibility with FeasibilityJump.
int64_t fj_nl_initialize(Model& model, ViolationManager& vm, int max_iterations, RNG* rng_ptr,
                         double time_limit, double tolerance) {
    RNG local_rng(42);
    RNG& rng = rng_ptr != nullptr ? *rng_ptr : local_rng;

    GFJConfig config;
    config.max_iterations = max_iterations;
    config.time_limit = time_limit;
    config.set_initial_x = false;  // refine the current (already-initialised) assignment
    config.tolerance = tolerance;
    // As a warm start, two-phase over-commits the linear submodel to
    // cost-pessimal boundary values; single-phase starts better. Two-phase stays
    // GFJ-as-solver's default.
    config.two_phase = false;
    FeasibilityJump fj(model, vm, rng, config);
    fj.run();

    // Reset the GLS weights FJ skewed, so the caller starts unweighted.
    std::fill(vm.weights.begin(), vm.weights.end(), 1.0);
    vm.invalidate_cache();
    return fj.iterations();
}

namespace {

// The wall-clock budget, computed in solve() because FeasibilityJump takes it at
// construction, then carried into the loop.
struct Budget {
    std::chrono::steady_clock::time_point start;
    std::chrono::steady_clock::time_point deadline;
    bool has_deadline = false;
    double seconds = 0.0;  // saturated; see make_budget
};

Budget make_budget(double time_limit) {
    Budget b;
    b.start = std::chrono::steady_clock::now();
    // A positive limit is a hard deadline checked at every sub-step, not just
    // between batches; <= 0 leaves max_iterations alone, so runs are deterministic.
    b.has_deadline = time_limit > 0.0;
    // Saturate before the integer-tick cast: a huge limit (e.g. double::max())
    // overflows int64 nanoseconds into a past deadline that ends the run at once.
    constexpr double kMaxBudgetSeconds = 1.0e9;  // ~31 years
    // Reused for FJ's deadline: FeasibilityJump::begin() does the same cast, so
    // the raw value would overflow there.
    b.seconds = b.has_deadline ? std::min(time_limit, kMaxBudgetSeconds) : 0.0;
    b.deadline = b.start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                               std::chrono::duration<double>(b.seconds));
    return b;
}

// Whether an FJ/Novelty batch has anything to move: a Bool/Int/Float variable
// (FeasibilityJump::jumpable) with >= 2 domain values that some row reads; one in
// no row has an empty G_v, so FJ never scores it (#201).
//
// The folded `obj <= bound` row counts, since FJ is the only thing that optimises
// a scalar read only by the objective. Residual: once FJ settles such scalars,
// FJ batches report stuck, take the unproductive kick, and the List-thrash this
// prevents returns (see SearchConfig::structural_batch_probability).
bool fj_has_work(const Model& model) {
    const auto& vars = model.variables();
    for (size_t i = 0; i < vars.size(); ++i) {
        const Variable& v = vars[i];
        const bool jumpable =
            v.type == VarType::Bool || v.type == VarType::Int || v.type == VarType::Float;
        if (jumpable && movable_domain(v) &&
            !model.constraints_of_var(static_cast<int32_t>(i)).empty()) {
            return true;
        }
    }
    return false;
}

// Whether the structural batch will hold at least one generator.
bool structural_batch_has_work(const Model& model, const SearchConfig& config) {
    // A non-null registered generator counts on its own (#165): its moves may
    // touch only scalars, and keying on List/Set presence would silently drop it.
    // The built-ins count only when they will be built, else ~33% of batches
    // would be no-ops. Nulls are skipped, as StructuralBatch's constructor does.
    return std::any_of(
               config.move_generators.begin(), config.move_generators.end(),
               [](const std::shared_ptr<const MoveGenerator>& g) { return g != nullptr; }) ||
           (config.default_structural_generators &&
            std::any_of(model.variables().begin(), model.variables().end(),
                        [](const Variable& v) { return is_structured(v.type); }));
}

// Resolves SearchConfig::structural_batch_probability. 0 without structural work
// skips the batch and its generator scan. `fj_idle`: FJ has no work (fj_has_work).
double effective_structural_probability(bool has_structural, double configured, bool fj_idle) {
    if (!has_structural) {
        return 0.0;
    }
    if (configured >= 0.0) {
        return configured;
    }
    // Auto with nothing for FJ to jump: every batch is structural (#201). At 0.33
    // the empty FJ batches spun to their limit and kicked, re-randomising every
    // List; on CVRP X-n101-k25 (10 s, seed 1, engine 1304c43) 0.33 never reached
    // feasibility, 1.0 did at 0.006 s. Keyed on the model, evaluated once:
    // pick_batch_kind still draws once per batch, so other models keep their RNG
    // stream.
    if (fj_idle) {
        return 1.0;
    }
    return 0.33;
}

// Second arming condition for the Float escape probe (#117): this fraction of
// the wall-clock budget with no new best. Batches can cost seconds (elec25:
// ~1.2 s, 52 batches in 60 s), so there the batch count (perturbation_period =
// 100) never arms it. Only with a deadline: an iteration-budgeted run must not
// read the clock, to stay bit-reproducible. The kick cadence stays batch-counted
// (tuned; making it time-aware needs its own measurement; see
// maybe_arm_escape_probe).
constexpr double kEscapeArmFraction = 0.25;

// Second witness for #102's unproductive-batch exit: arm it only after
// perturbation_period / this non-improving batches (floored at 1; 0 would be
// "always armed"). FJ's measure ignores the objective row, so after feasibility
// it reports stuck on a search still improving (ex8_6_1: 152 of 162 batches).
// The kick then comes 20x sooner than perturbation_period at the default, not
// ~300x. A fraction, not a new knob, so the two windows keep their ratio when
// perturbation_period is retuned; integer division, so below 40 the window is 1
// batch and the ratio breaks.
// Tuned over windows of {5, 10, 20} batches on MINLPLib (10 s, four paired
// seeds): all fix ex8_6_1; st_e40 reaches its BKS on 4/4 seeds at 5, on 2/4 at 10
// or 20, so 5 (divisor 20) is chosen. Not shown to transfer off MINLPLib.
constexpr int kUnproductiveArmDivisor = 20;

// The bound a feasible objective `obj` earns, strictly below it. Every writer
// (record_best, adoption, the shared bound #179, the stall re-grounding) uses it,
// since "never loosen" needs the bound strictly increasing in `obj` at one
// `rel_step`; both branches are.
//  - Integral (Model::is_integral): `obj - 1`, as CP-SAT does; the next integer
//    is the only improvement and is reachable, which a relative step hid (#211).
//  - Otherwise `obj - rel_step * (|obj| + 1)`: the step is both the resolution
//    (a smaller improvement is never asked for) and the pace (FJ's Newton move
//    lands on the bound; eg_all_s takes 15,892 steps from 1e9 to 14.4, see
//    benchmarks/instances/minlplib/README.md). So it adapts (#241): starts at
//    kObjStepMax, a full-period stall divides it by kObjStepShrink down to
//    kObjStepMin, each recorded incumbent multiplies it by kObjStepGrow back up
//    to kObjStepMax (twice when the hook improves FJ's point; also on a best
//    behind a peer's, unmeasured). Held-out A/B (10 s, 3 seeds, 50 instances):
//    better than fixed 1e-3 on 2 beyond noise, worse on none; fixed 1e-4 and
//    1e-6 were worse on 10 and 16, their descents stalling.
double objective_bound_below(double obj, bool integral, double rel_step) {
    if (integral) {
        // nextafter keeps the bound strictly below obj past 2^53, where obj - 1
        // rounds back to obj.
        return std::min(obj - 1.0, std::nextafter(obj, -std::numeric_limits<double>::infinity()));
    }
    return obj - (rel_step * (std::abs(obj) + 1.0));
}

// The continuous step's range and moves (objective_bound_below). Fitted on
// nothing: the range is the two fixed steps #241 weighed, the factors its A/B's
// only setting.
constexpr double kObjStepMax = 1e-3;
constexpr double kObjStepMin = 1e-6;
constexpr double kObjStepShrink = 10.0;
constexpr double kObjStepGrow = 2.0;

// The ViolationLS outer loop: shared state, and one method per step of
// docs/architecture.md's "Main Loop".
class ViolationLSLoop {
public:
    ViolationLSLoop(Model& model, ViolationManager& vm, RNG& rng, FeasibilityJump& fj,
                    const SearchConfig& config, const Budget& budget, InnerSolverHook* hook,
                    LNS* lns, int lns_interval, SolveCallback* callback, SearchCoordination* coord);

    SearchResult run();

private:
    // ---- observations of the current assignment ----
    // Largest unweighted violation over the real constraints (objective row
    // excluded), so comparable across GLS weight changes; 0.0 when all hold. NaN
    // maps to +inf, else `value > tol` would call a NaN row satisfied (as
    // ViolationManager's clamped_node_violation guards).
    [[nodiscard]] double max_real_violation() const;
    [[nodiscard]] bool real_feasible() const {
        return max_real_violation() <= config_.feasibility_tolerance;
    }
    [[nodiscard]] double current_obj() const {
        return has_obj_ ? model_.node_value(model_.objective_id()) : 0.0;
    }
    // Whether the shared stop flag is raised. Relaxed: the flag guards no data
    // (the pool has its own mutex); seeing it a batch late costs that batch.
    [[nodiscard]] bool peer_stopped() const {
        return coord_ != nullptr && coord_->stop != nullptr &&
               coord_->stop->load(std::memory_order_relaxed);
    }
    // Whether the host cancelled (#169); apart from peer_stopped() because they
    // report different TerminationReasons. One null compare when unattached.
    [[nodiscard]] bool cancel_requested() const { return config_.stop.requested(); }
    [[nodiscard]] bool clock_expired() const {
        return has_deadline_ && std::chrono::steady_clock::now() >= deadline_;
    }
    // Read by the loop condition and every mid-batch budget guard, so a peer's
    // or the host's stop halts the search everywhere the clock would.
    [[nodiscard]] bool past_deadline() const {
        return cancel_requested() || peer_stopped() || clock_expired();
    }
    // past_deadline() in the shape StopRef adapts, handed to the inner-solver
    // hook so it stops on exactly what stops this loop (#191).
    struct LoopStop {
        const ViolationLSLoop* loop;
        [[nodiscard]] bool requested() const { return loop->past_deadline(); }
    };
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
    // Latch the first feasible point's objective and time (#149); a no-op after
    // the first. Nothing reads them back, so the trajectory is unchanged.
    void note_first_feasible(double obj);
    // Seconds since solve() started; SearchResult::time_seconds.
    [[nodiscard]] double elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    }
    // On stagnation: restore kick_origin() (#158), then LNS every
    // lns_interval-th time, else perturb. `allow_lns` is false only for #102's
    // unproductive route once feasible: the perturb costs microseconds and helps
    // (st_e40), while a repair costs up to min(2.0, remaining()) s and its trigger
    // is blind after feasibility (ex8_6_1 lost 4.7 s of 10 s to three).
    void diversify(bool allow_lns = true);
    // Hand a new incumbent to the shared pool, if there is one. No-op for a
    // single-threaded solve. Called from both of record_best's recording arms.
    void share(double objective);
    // The portfolio's best feasible finite objective, or +inf when there is no
    // pool or nothing feasible in it yet (#179). Read in both A/B arms.
    [[nodiscard]] double global_best() const {
        return coord_ != nullptr && coord_->pool != nullptr
                   ? coord_->pool->best_feasible_objective()
                   : std::numeric_limits<double>::infinity();
    }
    // objective_bound_below for this model, at the current step.
    [[nodiscard]] double bound_below(double obj) const {
        return objective_bound_below(obj, obj_integral_, obj_step_);
    }
    // On a full-period stall: shrink the continuous step and re-ground the bound
    // on the incumbent (#241). See the definition.
    void shrink_objective_step();
    // The bound the shared best `global` (a global_best() read) entitles us to,
    // or +inf when sharing is off or nothing is shared. Every bound writer takes
    // the min with it and records `global` in bound_global_, so our own later,
    // worse incumbent cannot undo a shared tightening (see record_best).
    [[nodiscard]] double shared_bound_cap(double global) const {
        return coord_ != nullptr && coord_->share_objective_bound && std::isfinite(global)
                   ? bound_below(global)
                   : std::numeric_limits<double>::infinity();
    }
    // At a batch boundary: tighten the bound to the shared cap if it is looser,
    // or -- sharing off -- note that the coming batch runs behind (#179).
    void sync_shared_bound();
    // After the batch: charge it to the behind-the-global counters if
    // sync_shared_bound said it ran behind.
    void account_behind_batch();
    // Restart from another worker's incumbent instead of perturbing our own.
    // Returns false -- leaving the assignment untouched -- when there is no
    // pool, the pool is empty, the drawn solution does not fit this model, or
    // the draw is the assignment we already hold.
    bool adopt_from_pool();
    // See the definition: the objective-bound rule an adoption applies.
    void reground_objective_bound_after_adoption(bool feasible_here, double obj);
    // Whether `state` is the assignment this model currently holds.
    [[nodiscard]] bool holds_assignment(const Model::State& state) const;
    // Whether the next kick draws LNS (diversify()'s test, asked beforehand).
    // Adoption stands down on those and advances `lns_slot_` on the ones it
    // takes, so the `lns_interval` cadence keeps running.
    [[nodiscard]] bool lns_kick_due() const {
        return lns_ != nullptr && lns_interval_ > 0 &&
               (lns_slot_ % lns_interval_ == lns_interval_ - 1);
    }

    // The point a kick departs from, or null to kick from where the search
    // stands (#158). Normally our incumbent; `adopted_origin_` while we stand on
    // an adopted peer point that did not become our incumbent (the next
    // incumbent clears it). `adopted_origin_` is only set from a draw feasible
    // here with a finite objective, so the null rules below need no read-side
    // check for it. Null before the first feasible point (best_state_ is the
    // initial assignment), and while the only feasible point is #100's
    // non-finite witness: never derive from a non-finite objective.
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
    // Bucket one batch into `counters_`. See the definition.
    void count_batch(BatchKind kind);
    // Run the batch. Returns true if it committed changes outside FJ's scan-set
    // and jump-table, i.e. if the loop owes an fj_.resync().
    bool run_batch(BatchKind kind);
    void note_closest_approach(double batch_violation);
    // Steps 3-4: bank the feasible point, polish it, bank the polish. Returns
    // whether the batch produced a new best, and may set `resync`.
    bool polish_and_record(double batch_violation, bool& resync);
    // Steps 5-6. Returns false when the run is over (pure feasibility: solved).
    bool apply_batch_outcome(bool improved, bool resync, bool count_stagnation);
    void maybe_arm_escape_probe();
    void maybe_diversify(BatchKind kind, bool improved);
    void maybe_emit_periodic_progress();
    // ---- the host's event stream (#169), no-ops without a tracer ------------
    // One null test per event, at batch, kick or inner-solver granularity, never
    // per GLS iteration.
    void trace_batch_end(BatchKind kind, bool improved) {
        if (tracer_ != nullptr) {
            tracer_->batch_end(kind, fj_.iterations(), improved);
        }
    }
    // `objective` is what share() was given, so both report the #100 witness as
    // +inf. The clock is read only with a tracer attached (determinism note in
    // docs/architecture.md).
    void trace_new_best(double objective) {
        if (tracer_ != nullptr) {
            tracer_->new_best(objective, elapsed());
        }
    }
    void trace_kick(KickKind kind) {
        if (tracer_ != nullptr) {
            tracer_->kick(kind);
        }
    }
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
    // SearchConfig::tracer, held so the null test is one load; per worker under
    // a portfolio.
    Tracer* tracer_;
    // Non-null only under ParallelSearch; null touches nothing shared, so a
    // single-threaded trajectory is unaffected.
    SearchCoordination* coord_;

    const std::chrono::steady_clock::time_point start_;
    const std::chrono::steady_clock::time_point deadline_;
    const bool has_deadline_;
    const double budget_seconds_;

    const bool has_obj_;
    // Whether the objective is integral, which sets objective_bound_below's step.
    const bool obj_integral_;
    // The continuous objective's relative bound step, adapted between
    // kObjStepMin and kObjStepMax (#241). Unused on an integral objective.
    double obj_step_ = kObjStepMax;
    // The global best the live bound last took its shared cap from; set by every
    // bound writer, +inf until the first. A shared tightening is made and counted
    // only past it: a step change re-deriving the cap from the same global best
    // is no peer moving the bound (#241).
    double bound_global_ = std::numeric_limits<double>::infinity();
    const int32_t obj_ci_;
    const std::vector<int32_t>& cids_;
    // The structural batch has work and FJ/Novelty have none (#201). False on
    // any model without structure, even one with no movable scalar.
    const bool fj_idle_;
    const double structural_probability_;
    const int unproductive_arm_stagnation_;
    // Owns this search's clone of every registered generator, so portfolio
    // workers share no mutable generator state (#157, #165).
    StructuralBatch structural_;

    double best_feasible_obj_ = std::numeric_limits<double>::infinity();
    Model::State best_state_;
    bool have_feasible_ = false;
    // Closest approach to feasibility, so an infeasible run returns a
    // diagnosable state instead of the initial assignment.
    double best_violation_ = std::numeric_limits<double>::infinity();
    Model::State closest_state_;
    // The state adoption installed, held only while it is not best_state_ (see
    // kick_origin()). Only adopt_from_pool writes it, so a run without a pool
    // never pays for the copy.
    Model::State adopted_origin_;
    bool has_adopted_origin_ = false;
    int perturbations_ = 0;
    int lns_repairs_ = 0;
    // Repairs destroy_repair accepted (#150); never branched on.
    int lns_repairs_accepted_ = 0;
    // #149's pair; NaN until the first feasible point.
    double first_feasible_obj_ = std::numeric_limits<double>::quiet_NaN();
    double first_feasible_time_ = std::numeric_limits<double>::quiet_NaN();
    // Kicks eligible for an LNS repair, the unit of `lns_interval`. Apart from
    // `perturbations`: if #102's LNS-refused kicks advanced it, the
    // perturbation_period route's LNS phase would shift, at some counts freeze.
    int lns_slot_ = 0;
    int stagnation_ = 0;
    int64_t batches_ = 0;
    // Where the run spent its work (#169). Never read by control flow, so it
    // cannot change the trajectory; hence `batches_` above, which
    // budget_exhausted() branches on, is kept separate.
    SearchCounters counters_;
    // Set by sync_shared_bound when the coming batch's bound is looser than the
    // portfolio's best (only with sharing off); consumed by account_behind_batch.
    // The start time is read only then and only with a wall clock (#179).
    bool batch_behind_global_ = false;
    std::chrono::steady_clock::time_point behind_batch_start_;
    std::chrono::steady_clock::time_point last_callback_;
    std::chrono::steady_clock::time_point last_improvement_;
    // Which budget ended the run (#104). The `while` condition's exit seeds it;
    // each `break` overwrites it, so it names the exit actually taken.
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
      tracer_(config.tracer),
      coord_(coord),
      start_(budget.start),
      deadline_(budget.deadline),
      has_deadline_(budget.has_deadline),
      budget_seconds_(budget.seconds),
      has_obj_(model.objective_id() >= 0),
      obj_integral_(has_obj_ && model.is_integral(model.objective_id())),
      obj_ci_(model.objective_constraint_idx()),
      cids_(model.constraint_ids()),
      fj_idle_(structural_batch_has_work(model, config) && !fj_has_work(model)),
      structural_probability_(effective_structural_probability(
          structural_batch_has_work(model, config), config.structural_batch_probability, fj_idle_)),
      unproductive_arm_stagnation_(
          std::max(1, config.perturbation_period / kUnproductiveArmDivisor)),
      structural_(model, config, structural_probability_ > 0.0),
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
    // Latch on the time, not `obj`: the #100 witness may pass NaN, which would
    // re-arm the latch and record the last point instead of the first.
    if (!std::isnan(first_feasible_time_)) {
        return;
    }
    first_feasible_time_ = elapsed();
    first_feasible_obj_ = obj;
}

bool ViolationLSLoop::record_best() {
    double obj = current_obj();
    // A feasible point whose objective is +inf/NaN is still feasible (#100), so
    // it is recorded as the first feasibility witness. It never becomes
    // best_feasible_obj_ (nothing beats +inf) and no bound is derived from it (a
    // NaN bound violates the objective row forever); any later finite-objective
    // feasible point displaces it.
    if (!std::isfinite(obj)) {
        if (have_feasible_) {
            return false;  // already have a witness, and possibly a better one
        }
        have_feasible_ = true;
        note_first_feasible(obj);
        best_state_ = model_.copy_state();
        // Both have_feasible_ transitions release the adopted origin: kicking
        // from a peer's point while best_state_ holds our own witness is wrong
        // (#158).
        has_adopted_origin_ = false;
        adopted_origin_ = Model::State{};
        // A +inf bound is vacuous (a written +inf reads as "side absent",
        // residual 0), leaving no objective signal (#116). So install the
        // loosest finite bound, kInfPenalty (1e30, violation.h's clamp for +inf
        // and NaN): it violates only a non-number objective and admits every
        // finite one (a finite objective above 1e30 reads as +inf, as before).
        // A tighter sentinel could exclude feasible finite points. Only when the
        // bound is still +inf, so it never overwrites a real incumbent's bound.
        //
        // The `return true` is load-bearing: the caller then calls
        // fj.reset_weights(), rebuilding FJ's violated set; without it the new
        // row stays invisible to the jump table.
        //
        // While the sentinel holds and the objective is non-finite: progress
        // reports feasible = true with total_violation ~1e30
        // (SolveProgress::total_violation); anything comparing assignments by
        // violation must difference per constraint, since a 1e30 row swallows
        // the real rows in a summed difference (FJ #100, structural batch #118);
        // and float_jump_candidates reports "gradient unusable", which makes the
        // Float escape probe eligible (#117).
        if (has_obj_ && !std::isfinite(model_.objective_bound())) {
            model_.set_objective_bound(kInfPenalty);  // the shared clamp; see violation.h
        }
        // Shared as +inf, not `obj`: a NaN would make the pool's sort undefined,
        // and +inf ranks the witness last among feasible entries.
        share(std::numeric_limits<double>::infinity());
        emit_progress(/*new_best=*/true);
        trace_new_best(std::numeric_limits<double>::infinity());
        return true;
    }
    // isfinite guards the +inf witness incumbent, where `inf - inf` is NaN. The
    // test is against our own best, not the shared bound (#179): best_state_ is
    // what finish() returns, so a point between the global best and ours is
    // recorded; the min below keeps it from loosening the bound.
    if (have_feasible_ && std::isfinite(best_feasible_obj_) &&
        obj >= best_feasible_obj_ - (1e-12 * (std::abs(best_feasible_obj_) + 1.0))) {
        return false;
    }
    // Read before share() folds `obj` into it; false without a pool (+inf).
    const bool behind_global = obj > global_best();
    const bool sharing = coord_ != nullptr && coord_->share_objective_bound;
    if (behind_global) {
        ++counters_.own_best_behind_global;
    }
    have_feasible_ = true;
    note_first_feasible(obj);
    best_feasible_obj_ = obj;
    best_state_ = model_.copy_state();
    bool bound_moved = true;  // no objective row: nothing to move, keep the old return
    // The new incumbent is the kick origin again (#158).
    has_adopted_origin_ = false;
    adopted_origin_ = Model::State{};
    if (has_obj_) {
        // Capped by the shared bound (#179), else a worker behind the portfolio
        // would raise a bound the pool tightened. The cap is +inf without a pool
        // or with sharing off, so those runs set `own` unchanged.
        const double own = bound_below(obj);
        const double global = global_best();
        const double cap = shared_bound_cap(global);
        const double previous = model_.objective_bound();
        // Counted only when a peer improved since the bound took its cap, not
        // when the sync already put it there or only the step changed.
        const bool peer_moved = global < bound_global_;
        if (cap < own && cap < previous && peer_moved) {
            ++counters_.shared_bound_tightenings;
        }
        // A behind own best, sharing on, no peer improvement since the cap was
        // taken: keep the bound. `own` and `cap` use the step record_best grew
        // since (#241), so either may sit below `previous`, and moving there
        // would be a step change read as a tightening (returning true below).
        // Otherwise the min. On an integral objective this differs only where an
        // adoption re-grounded below the cap, where the min would loosen.
        const double next = behind_global && sharing && !peer_moved ? previous : std::min(own, cap);
        bound_moved = next < previous;
        model_.set_objective_bound(next);
        bound_global_ = global;
        // Grow the step on every recorded incumbent (#241), after setting the
        // bound, so this best is bounded at the step that found it.
        obj_step_ = std::min(kObjStepMax, obj_step_ * kObjStepGrow);
    }
    share(obj);
    emit_progress(/*new_best=*/true);
    trace_new_best(obj);
    // Sharing on, an own best behind the portfolio's under an unmoved bound is
    // recorded and shared but not an improvement: true would buy
    // apply_batch_outcome's resets (weights, rho, probe, stagnation), which the
    // paper ties to a new bound, and would postpone the adoption kick. Without
    // sharing or a pool, true as before #179. Two asymmetries #179 measured
    // (re-measure before changing): a tie with the global best returns true
    // (pinned by "an own best that ties the shared bound still counts as an
    // improvement"); a behind best returns true when bound_moved, and that
    // reset_weights is then the only FJ rebuild after the move.
    return !behind_global || bound_moved || !sharing;
}

// Submits when found, not at the end. Per new best: one State copy (moved into
// the pool, so no copy inside the critical section), one mutex, one
// O(#constraints) residual scan; off the hot path.
void ViolationLSLoop::share(double objective) {
    if (coord_ == nullptr || coord_->pool == nullptr) {
        return;
    }
    Solution sol;
    sol.state = best_state_;
    sol.objective = objective;
    sol.feasible = true;  // record_best's precondition
    // The model still holds `best_state_` (record_best copies it just before
    // every call), so the live residual is the shared state's.
    sol.violation = max_real_violation();
    coord_->pool->submit(std::move(sol));
}

// #179: per batch, not per iteration, since a tightening costs an FJ rebuild and
// every other bound change lands per batch too. A tightening keeps the
// assignment, best_state_ (finish() returns only points we found) and the GLS
// weights (resync, not reset_weights: a reset per peer improvement is another
// mechanism). It also reaches a worker with no feasible point, putting its row
// in play early, as CP-SAT does; the per-worker cost is unmeasured. The row's
// weight is never zeroed: solve() runs FJ single-phase.
//
// Never loosens (`target < bound`, strictly). The other writers take the min
// with shared_bound_cap() at the current step, so shrink_objective_step gives up
// only the step, never a peer's improvement.
void ViolationLSLoop::sync_shared_bound() {
    batch_behind_global_ = false;
    if (!has_obj_) {
        return;
    }
    const double global = global_best();
    if (!std::isfinite(global)) {
        return;  // no pool, or nothing feasible in it yet
    }
    // The global best is no better than ours. Tested explicitly: record_best grew
    // the step after setting the bound (#241), so bound_below(global) sits below
    // it, and a lone worker would tighten against itself and leave its
    // sharing-off trajectory.
    if (have_feasible_ && global >= best_feasible_obj_) {
        return;
    }
    const double target = bound_below(global);
    if (!(target < model_.objective_bound())) {
        return;  // already at least as tight: our own best, or an earlier sync
    }
    if (coord_->share_objective_bound) {
        // No peer improved since the cap was taken: the gap is the grown step
        // (#241), not a tightening.
        if (!(global < bound_global_)) {
            return;
        }
        model_.set_objective_bound(target);
        bound_global_ = global;
        // FJ's violated set and jump table predate the rewritten row; without
        // the resync a previously satisfied row stays out of V and FJ can call
        // the batch feasible. No vm_.invalidate_cache(): total_violation()
        // re-diffs every row on each call, so its cache cannot be stale.
        fj_.resync();
        ++counters_.shared_bound_tightenings;
        return;
    }
    batch_behind_global_ = true;
    if (has_deadline_) {
        behind_batch_start_ = std::chrono::steady_clock::now();
    }
}

void ViolationLSLoop::account_behind_batch() {
    if (!batch_behind_global_) {
        return;
    }
    ++counters_.bound_behind_global_batches;
    if (has_deadline_) {
        counters_.bound_behind_global_seconds +=
            std::chrono::duration<double>(std::chrono::steady_clock::now() - behind_batch_start_)
                .count();
    }
}

bool ViolationLSLoop::holds_assignment(const Model::State& state) const {
    const auto& vars = model_.variables();
    for (size_t i = 0; i < vars.size(); ++i) {
        // Bit equality: a restored copy of our own snapshot matches exactly; a
        // near-miss is a different point and a legitimate restart.
        if (vars[i].value != state.values[i] || vars[i].elements != state.elements[i]) {
            return false;
        }
    }
    return true;
}

// #241: a full-period stall divides the step and re-grounds the bound on the
// incumbent, the one writer that loosens it. It stays strictly below
// best_feasible_obj_ and under the shared cap, so no found improvement is given
// back. No-op at kObjStepMin, without an incumbent, or on an integral objective.
// No FJ resync: the caller's kick follows at once, and every kick rebuilds FJ
// (reset_weights, or inside perturb()). Unmeasured under a portfolio, where an
// adopted point descends at the shrunk step until we record our own best.
void ViolationLSLoop::shrink_objective_step() {
    if (!has_obj_ || obj_integral_ || !have_feasible_ || !std::isfinite(best_feasible_obj_) ||
        obj_step_ <= kObjStepMin) {
        return;
    }
    // 1e-3 / 10^3 rounds just above kObjStepMin, so the floor takes a fourth
    // shrink (~1e-22 relative). Left as measured.
    obj_step_ = std::max(kObjStepMin, obj_step_ / kObjStepShrink);
    const double global = global_best();
    model_.set_objective_bound(std::min(bound_below(best_feasible_obj_), shared_bound_cap(global)));
    bound_global_ = global;
}

// The objective-bound half of an adoption. `feasible_here` and `obj` are this
// model's verdict on the adopted point, not the submitter's.
void ViolationLSLoop::reground_objective_bound_after_adoption(bool feasible_here, double obj) {
    if (!has_obj_) {
        return;
    }
    // Never loosen: only a strict own improvement or a peer's better best
    // tightens it again, and the draw is often one of our own worse incumbents,
    // so loosening would leave the row satisfied with no pressure. As with LNS,
    // a violated `obj <= best - eps` at the new point is the normal state.
    double target = std::numeric_limits<double>::infinity();
    if (feasible_here && std::isfinite(obj)) {
        target = obj;
    }
    if (have_feasible_ && std::isfinite(best_feasible_obj_)) {
        target = std::min(target, best_feasible_obj_);
    }
    // The shared cap (#179), as in record_best: re-grounding on our view must
    // not loosen a bound the pool tightened. +inf with sharing off.
    const double global = global_best();
    const double bound = std::min(
        std::isfinite(target) ? bound_below(target) : std::numeric_limits<double>::infinity(),
        shared_bound_cap(global));
    if (std::isfinite(bound)) {
        model_.set_objective_bound(bound);
        bound_global_ = global;
    } else if (!std::isfinite(model_.objective_bound())) {
        // No usable objective yet: install record_best's finite sentinel so the
        // row is not vacuous, only over a +inf bound.
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
    // The caller's factory may return differently shaped models; refuse a
    // mismatched state: restore_state throws on a width mismatch (losing the
    // worker) and `elements` is indexed unchecked.
    if (sol->state.values.size() != model_.num_vars() ||
        sol->state.elements.size() != model_.num_vars()) {
        return false;
    }

    // A draw equal to the live assignment moves nothing yet would zero
    // stagnation_, so refuse it and let the caller diversify(). Live, not
    // best_state_: after a stall the model usually sits where FJ left it, and
    // comparing with best_state_ would refuse genuine moves.
    if (holds_assignment(sol->state)) {
        return false;
    }

    model_.restore_state(sol->state);
    // Required: restore_state writes only the variables, and everything below
    // reads node values (as LNS does around a repair).
    full_evaluate(model_);

    // This model's verdict, not `sol->feasible`: workers' models may differ, and
    // trusting the flag could make finish() return a point violating ours.
    const bool feasible_here = real_feasible();
    const double obj = current_obj();

    // Model::State does not carry the bound (a Const node). Re-ground it under
    // record_best's rules: only from a feasible point, never from a non-finite
    // objective. The pool's infeasible closest-approach states often beat every
    // feasible objective, so a bound from one no feasible point could meet.
    reground_objective_bound_after_adoption(feasible_here, obj);

    // Only an improvement replaces best_state_, which finish() returns.
    if (feasible_here && std::isfinite(obj) &&
        (!have_feasible_ || !std::isfinite(best_feasible_obj_) || obj < best_feasible_obj_)) {
        have_feasible_ = true;
        note_first_feasible(obj);
        best_feasible_obj_ = obj;
        // Moved: `sol` is a by-value copy last read by the restore above.
        best_state_ = std::move(sol->state);
        // It is the incumbent now, so kick_origin() finds it there.
        has_adopted_origin_ = false;
        adopted_origin_ = Model::State{};
    } else if (feasible_here && std::isfinite(obj)) {
        // Not an improvement, but later kicks must depart from the adopted
        // point, or the next kick restores our incumbent and discards the draw
        // (#158, #135). The same feasible-and-finite rule is load-bearing: the
        // pool holds infeasible closest-approach states and #100's witness,
        // which would anchor most later kicks to an infeasible point or one with
        // no objective.
        adopted_origin_ = std::move(sol->state);  // moved, as above
        has_adopted_origin_ = true;
    } else {
        // A legitimate restart but not a point to return to: release any older
        // origin so kick_origin() falls back to the incumbent.
        has_adopted_origin_ = false;
        adopted_origin_ = Model::State{};
    }

    if (!has_obj_ && feasible_here) {
        // Pure feasibility: the adopted point is the answer. Say so here, since
        // record_best now refuses every later batch (objective constant 0.0) and
        // the Feasible exit is unreachable; stop the peers as run_worker does.
        termination_ = TerminationReason::Feasible;
        if (coord_->stop != nullptr) {
            coord_->stop->store(true, std::memory_order_relaxed);
        }
    }

    vm_.invalidate_cache();
    // reset_weights: the weights were shaped by the basin we leave. Not
    // fj_.begin(), which zeroes iterations() (breaking the budget and count) and
    // re-arms a fresh wall clock past solve()'s deadline.
    fj_.reset_weights();
    // The caller's probe arming diagnosed the assignment we left, so disarm and
    // let the arming routes re-earn it here. A kick adoption declines keeps it
    // armed, harmlessly: the flag only enables the probe, and compute_var_jump
    // re-tests `!gradient_usable` at the live point (pinned in test_parallel).
    fj_.set_escape_probe(false);
    // Re-ground the time route's clock too, or it re-arms next batch and breaks
    // its "at most 1/kEscapeArmFraction arms per run" bound.
    if (has_deadline_) {
        last_improvement_ = std::chrono::steady_clock::now();
    }
    sample_rho();
    ++perturbations_;
    // This was an LNS-eligible full-period kick, so advance the slot; otherwise
    // lns_slot_ stays 0 and lns_kick_due() never comes true.
    ++lns_slot_;
    stagnation_ = 0;
    return true;
}

void ViolationLSLoop::diversify(bool allow_lns) {
    // Kick from kick_origin(), not from where the last kick left the search
    // (#158): otherwise failed kicks compose into a walk away from the best
    // point (mipfeas smoke, 60 s: post-kick assignments were real-feasible 5.3%
    // of the time). A deliberate divergence from Davies et al. Algorithm 6,
    // whose line 12 perturbs X in place and whose only restore (line 6) is a new
    // pool best (adopt_from_pool). Justified because #102's route kicks ~50x the
    // reference rate (98% of kicks); cost on #158: MINLPLib within noise, LNS
    // acceptance 27.3% -> 7.5%. If that outweighs the gain, attack the kick rate,
    // not this restore. kick_origin(), not best_state_, so the next kick does not
    // undo an adoption. Null before the first feasible point: no restore.
    //
    // restore_state writes only the variables. Only the LNS branch reads node
    // values, so only it pays a full_evaluate; perturb() draws from domains and
    // ends with its own, and skipping one changes no trajectory (no RNG draw, no
    // variable write).
    const Model::State* origin = kick_origin();
    if (origin != nullptr) {
        model_.restore_state(*origin);
    }
    if (allow_lns && lns_ != nullptr && lns_interval_ > 0 &&
        (lns_slot_ % lns_interval_ == lns_interval_ - 1)) {
        if (origin != nullptr) {
            full_evaluate(model_);  // see above: destroy_repair reads node values
            // The cached total is stale after a full_evaluate (violation.h); an
            // overriding destroy_repair must not see the two disagree.
            vm_.invalidate_cache();
        }
        // At most the remaining budget, floored above 0 with a deadline: 0.0
        // means "no wall-clock limit" in fj_nl_initialize.
        const double repair_limit =
            has_deadline_ ? std::max(1e-9, std::min(2.0, remaining())) : 0.0;
        // After the restore, the built-in LNS keeps a repair only if it beats
        // the kick origin. A rejection leaves the search on the origin with
        // nothing perturbed; falling through to a perturb would be a trajectory
        // change needing its own measurement.
        trace_kick(KickKind::LNS);
        lns_->set_feasibility_tolerance(config_.feasibility_tolerance);
        lns_->set_neighbours(config_.structural_neighbours);
        const bool accepted = lns_->destroy_repair(model_, vm_, rng_, repair_limit);
        ++lns_repairs_;
        lns_repairs_accepted_ += accepted ? 1 : 0;
        if (tracer_ != nullptr) {
            tracer_->lns(accepted);
        }
        fj_.reset_weights();  // LNS mutated state outside GFJ
    } else {
        trace_kick(KickKind::Perturb);
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
    // Actual GLS iterations; batches * batch_iterations over-counts any batch
    // that exits early.
    if (config_.max_iterations > 0 && fj_.iterations() >= config_.max_iterations) {
        termination_ = TerminationReason::IterationLimit;
        return true;
    }
    // Batches that charge no iteration (Structural, or FJ/Novelty starting with
    // V empty) would otherwise spin forever without a clock; this bites only
    // when iterations stall.
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

// Exactly one arm per batch, so fj + novelty + structural == batches
// (tests/test_counters.cpp).
void ViolationLSLoop::count_batch(BatchKind kind) {
    ++counters_.batches;
    switch (kind) {
        case BatchKind::FeasibilityJump:
            ++counters_.fj_batches;
            break;
        case BatchKind::NoveltyJump:
            ++counters_.novelty_batches;
            break;
        case BatchKind::Structural:
            ++counters_.structural_batches;
            break;
    }
}

// The FJ/Novelty choice is redrawn every batch, a deliberate deviation from
// ViolationLS §5 / Algorithm 6, which holds it until a new best or perturbation.
// #209's pre-registered paired A/B (MIPfeas smoke, 30 s, seeds 1-10, engine
// a19b222, primal integral) favoured per-batch: sticky minus per-batch +0.031,
// 95% CI [+0.016, +0.046], 110 pairs. A sticky draw drifts toward Novelty, whose
// stretches run to perturbation_period while FJ's end at a new best. OR-Tools'
// ls worker likewise redraws use_compound_moves at each Luby restart and new
// best. No draw without compound moves; the structural draw is our extension.
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
            return structural_.run(model_, vm_, rng_, has_deadline_, deadline_);
        case BatchKind::NoveltyJump:
            // Leaves FJ's scan set and jump table current (#209).
            fj_.novelty_batch(config_.batch_iterations);
            return false;
        case BatchKind::FeasibilityJump:
            fj_.batch(config_.batch_iterations);
            return false;
    }
    return false;
}

void ViolationLSLoop::note_closest_approach(double batch_violation) {
    // Only until the first feasible point, after which finish() uses best_state_.
    // `batches == 1` captures once even on an all-NaN run (+inf never compares
    // `<`) without re-snapshotting on every batch of a plateau.
    if (!have_feasible_ && (batch_violation < best_violation_ || batches_ == 1)) {
        best_violation_ = batch_violation;
        closest_state_ = model_.copy_state();
    }
}

bool ViolationLSLoop::polish_and_record(double batch_violation, bool& resync) {
    if (batch_violation > config_.feasibility_tolerance) {
        return false;
    }
    // Record before polishing: the hook can leave the feasible region, and a
    // visited feasible point must not be lost.
    bool improved = record_best();
    // The hook's time is unbounded; don't start one with no budget left.
    if (hook_ != nullptr && !past_deadline()) {
        ++counters_.inner_solver_calls;
        // An iteration-budgeted run reads no clock at all (#169, the determinism
        // note in docs/architecture.md), so inner_solver_seconds reads 0.0 there.
        // An attached tracer widens the gate: Tracer::hook carries the duration.
        const bool time_the_hook = has_deadline_ || tracer_ != nullptr;
        const auto hook_started = time_the_hook ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
        // The hook gets the loop's stop condition (#191). past_deadline() reads
        // no clock on a clockless run, so it stays bit-reproducible.
        const LoopStop stop{this};
        // continuous-objective polish (mutates floats)
        hook_->solve(model_, vm_, {}, StopRef(stop));
        if (time_the_hook) {
            const double hook_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - hook_started)
                    .count();
            if (has_deadline_) {
                counters_.inner_solver_seconds += hook_seconds;
            }
            if (tracer_ != nullptr) {
                tracer_->hook(hook_seconds);
            }
        }
        resync = true;
        if (real_feasible()) {  // keep the polish only if it stayed feasible
            improved = record_best() || improved;
        }
    }
    return improved;
}

bool ViolationLSLoop::apply_batch_outcome(bool improved, bool resync, bool count_stagnation) {
    if (!improved) {
        if (count_stagnation) {
            ++stagnation_;
        }
        if (resync) {
            fj_.resync();  // re-sync after hook/structural mutation, keep GLS weights
        }
        return true;
    }
    // Gated so a clockless run reads no clock at all; the only reader is the
    // has_deadline_-gated arming route.
    if (has_deadline_) {
        last_improvement_ = std::chrono::steady_clock::now();
    }
    stagnation_ = 0;
    // Progress: the escape probe is not needed and not free.
    fj_.set_escape_probe(false);
    // new best: fresh GLS weights (paper) + new rho
    fj_.reset_weights();
    sample_rho();
    if (!has_obj_) {
        // Pure feasibility: first solution is the answer.
        termination_ = TerminationReason::Feasible;
        return false;
    }
    return true;
}

void ViolationLSLoop::maybe_arm_escape_probe() {
    // Time-based arming (#117; kEscapeArmFraction): one clock read per batch,
    // skipped while armed. It carries no kick and is not gated on a projected
    // batch shortfall: a new best disarms the probe and restarts the clock, so it
    // arms at most 1/kEscapeArmFraction times per run, unlike #107's always-on
    // probe (9x regression). #117's roster numbers describe this ungated form;
    // the removed gate was measured to cost objective quality while preventing
    // nothing.
    if (has_deadline_ && !fj_.escape_probe()) {
        const auto now = std::chrono::steady_clock::now();
        if (now < deadline_ && std::chrono::duration<double>(now - last_improvement_).count() >=
                                   kEscapeArmFraction * budget_seconds_) {
            fj_.set_escape_probe(true);
        }
    }
}

void ViolationLSLoop::maybe_diversify(BatchKind kind, bool improved) {
    // An FJ batch that reported itself unproductive (GFJConfig::
    // unproductive_iterations) has nothing left, so the kick is due now (#102).
    // Gated on !improved: after feasibility a batch can record a new best on the
    // objective yet exit stuck on the real rows, and kicking it would undo the
    // `improved` resets. That is the second gate after
    // unproductive_arm_stagnation, kept because it reads the post-batch state.
    // Gated on fj_idle_ (#201): an FJ batch with nothing to jump is stuck by
    // construction, and its kick would re-randomise every List. Reachable only
    // at an explicit structural probability; false wherever FJ has work.
    const bool unproductive_kick =
        kind == BatchKind::FeasibilityJump && !improved && fj_.batch_stuck() && !fj_idle_;
    if (stagnation_ >= config_.perturbation_period && !past_deadline()) {
        // Arm the Float escape probe: a variable at a stationary point of every
        // violated row has no other candidate, and a kick re-converges there.
        // The next improvement disarms it.
        fj_.set_escape_probe(true);
        // Before the kick, so an adoption re-grounds at the shrunk step too.
        shrink_objective_step();
        // Under ParallelSearch, try a peer's point instead (#135; whether that
        // beats perturbing is unmeasured, a null coord is the control arm).
        // adopt_from_pool() performs the whole kick; without a pool it returns
        // false at once. Only on this route: an adoption (lock, restore,
        // full_evaluate, FJ rebuild) on the every-batch unproductive route would
        // thrash, while #158's restore there costs one copy.
        //
        // `lns_kick_due()` first, load-bearing: this is the only route that may
        // draw LNS once feasible, so adoption taking every kick would switch LNS
        // off for every portfolio worker. Adoption replaces only the perturb.
        if (lns_kick_due() || !adopt_from_pool()) {
            diversify();
        } else {
            // Adoption was this batch's kick; never reported without a pool.
            trace_kick(KickKind::Adopt);
        }
    } else if (unproductive_kick && !past_deadline()) {
        // Kick early, but arm no probe and keep the stagnation count. This
        // route can fire every batch, so arming would be #107's always-on probe
        // (9x regression). Zeroing the count would keep an always-unproductive
        // model from ever reaching perturbation_period, the probe's only arming
        // clock on a clockless run.
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
    // The incumbent, or on an infeasible run the closest approach.
    model_.restore_state(have_feasible_ ? best_state_ : closest_state_);
    // Release the objective bound so post-solve checks over the model's
    // constraints don't see it violated by ~eps.
    if (has_obj_) {
        model_.set_objective_bound(std::numeric_limits<double>::infinity());
    }
    full_evaluate(model_);

    // Sampled before filling the record, so its size does not move the time.
    const double run_seconds = elapsed();

    SearchResult result;
    result.objective =
        have_feasible_ ? best_feasible_obj_ : std::numeric_limits<double>::infinity();
    result.feasible = have_feasible_;
    result.best_state = have_feasible_ ? best_state_ : closest_state_;
    // The returned assignment's residual, from the fresh full_evaluate above.
    result.best_violation = max_real_violation();
    // budget_exhausted() stops when either count reaches the limit, so this max
    // agrees with the budget (#201); a structural-only run reports batches, not 0.
    result.iterations = std::max(fj_.iterations(), batches_);
    result.time_seconds = run_seconds;
    result.termination = termination_;
    result.escape_probe_armed = fj_.escape_probe();
    result.perturbations = perturbations_;
    result.lns_repairs = lns_repairs_;
    result.lns_repairs_accepted = lns_repairs_accepted_;
    result.first_feasible_objective = first_feasible_obj_;
    result.time_to_first_feasible = first_feasible_time_;
    result.counters = counters_;
    result.counters.novelty_moves = fj_.novelty_moves();
    result.counters.novelty_commits = fj_.novelty_commits();
    result.counters.novelty_weight_bumps = fj_.novelty_weight_bumps();
    // Per-generator tallies live in the sweep; one copy at exit, not a merge per
    // batch.
    result.counters.by_generator = structural_.generator_counters();
    for (const GeneratorCounters& g : result.counters.by_generator) {
        result.counters.structural_moves_tried += g.moves_tried;
        result.counters.structural_moves_accepted += g.moves_accepted;
    }
    return result;
}

SearchResult ViolationLSLoop::run() {
    while (!past_deadline()) {
        if (budget_exhausted()) {
            break;
        }

        const BatchKind kind = pick_batch_kind();
        // From the pre-batch stagnation count alone (kUnproductiveArmDivisor);
        // disarmed, the batch also cannot end early (on ex8_6_1 early exits cost
        // ~6 gap points with kicks suppressed). Measured worse, don't restore:
        // gating the kick on !have_feasible (st_e40 BKS 2/4 seeds, not 4/4), and
        // rate-limiting it (infeasible at seed 2 on the 15 000-iteration budget).
        // A delay, not a rate cap: the kick keeps the count, so every batch after
        // the threshold is armed until a reset; dropping LNS once feasible is
        // what keeps a misfire cheap.
        fj_.set_watch_progress(stagnation_ >= unproductive_arm_stagnation_);

        // A no-op without a pool; draws no RNG, so the kind's draw is unchanged.
        sync_shared_bound();
        if (config_.auditor != nullptr) {
            config_.auditor->audit(AuditPoint::BeforeBatch, fj_);
        }
        bool resync = run_batch(kind);
        if (config_.auditor != nullptr && kind != BatchKind::Structural) {
            config_.auditor->audit(kind == BatchKind::FeasibilityJump
                                       ? AuditPoint::AfterFeasibilityJump
                                       : AuditPoint::AfterNoveltyJump,
                                   fj_);
        }
        account_behind_batch();
        ++batches_;
        count_batch(kind);

        const double batch_violation = max_real_violation();
        note_closest_approach(batch_violation);
        const bool improved = polish_and_record(batch_violation, resync);
        // Before apply_batch_outcome, which can end the run, so the last batch
        // is reported too.
        trace_batch_end(kind, improved);
        // An FJ/Novelty batch on an fj_idle_ model moved nothing (#201);
        // counting it would let empty batches alone fire full-period kicks.
        // Only at an explicit structural probability.
        const bool empty_batch = kind != BatchKind::Structural && fj_idle_;
        if (!apply_batch_outcome(improved, resync, /*count_stagnation=*/!empty_batch)) {
            break;
        }
        maybe_arm_escape_probe();
        maybe_diversify(kind, improved);
        maybe_emit_periodic_progress();
    }
    // A seeded TimeLimit means the loop condition ended the run; every `break`
    // wrote its own reason. Precedence: an expired clock, then the host's cancel
    // (the outer cause), then a peer's stop. The two flags are independent
    // (ParallelSearch does not mirror a cancel onto `coord.stop`), so both can
    // be true at once.
    if (termination_ == TerminationReason::TimeLimit && !clock_expired()) {
        if (cancel_requested()) {
            termination_ = TerminationReason::Cancelled;
        } else if (peer_stopped()) {
            termination_ = TerminationReason::Stopped;
        }
    }
    return finish();
}

}  // namespace

void validate_search_config(const SearchConfig& config) {
    if (config.batch_iterations <= 0) {
        throw std::invalid_argument("SearchConfig::batch_iterations must be positive, got " +
                                    std::to_string(config.batch_iterations));
    }
    if (!(config.feasibility_tolerance >= 0.0) || !std::isfinite(config.feasibility_tolerance)) {
        throw std::invalid_argument("SearchConfig::feasibility_tolerance must be finite and >= 0");
    }
}

SearchResult solve(Model& model, double time_limit, uint64_t seed, bool use_fj,
                   InnerSolverHook* hook, LNS* lns, int lns_interval, SolveCallback* callback,
                   const SearchConfig& config, SearchCoordination* coord) {
    (void)use_fj;  // GFJ is always the engine now; the flag is vestigial.
    validate_search_config(config);
    RNG rng(seed);

    // Frozen here (#271); an unclosed model is closed, not refused. freeze()
    // closes it (later builders are refused, #173; the variable-to-constraint
    // index is built), adds the objective row, then builds the cone cache.
    model.freeze(config.cone_cache_bytes);
    const ConeCacheCounters cones_at_entry = cone_cache_counters();
    const bool has_obj = model.objective_id() >= 0;
    if (has_obj) {
        model.set_objective_bound(std::numeric_limits<double>::infinity());  // reset for re-solves
    }
    // Before anything reads a row: domain rows use this solve's tolerance (#258).
    model.set_domain_tolerance(config.feasibility_tolerance);
    ViolationManager vm(model);

    const Budget budget = make_budget(time_limit);

    // One path per variable (#108): FJ's begin() sets the scalars to their
    // closest-to-zero value (published Feasibility Jump), so only List/Set are
    // randomised here. The initial scalar point is thus seed-independent; for a
    // random one, call initialize_random and pass skip_init = true. Don't
    // randomise scalars here: begin() overwrites them, the draws would still
    // shift the RNG stream, and rng.uniform/integers misbehave on infinite
    // domains.
    if (!config.skip_init) {
        initialize_structured_random(model, rng);
    }

    GFJConfig gfj;
    gfj.two_phase = false;  // batches reuse weights across calls; single-phase GLS
    // The same deadline (armed in begin() below), else a batch entered just
    // before it runs to completion and overruns (+45% on the largest MINLPLib
    // instance).
    gfj.time_limit = budget.seconds;  // saturated: begin() casts to integer ticks
    gfj.max_iterations = 0;
    gfj.unproductive_iterations = config.unproductive_iterations;
    gfj.tolerance = config.feasibility_tolerance;  // one meaning of "satisfied" (#211)
    gfj.objective_tie_break = config.fj_objective_tie_break;
    FeasibilityJump fj(model, vm, rng, gfj);
    fj.begin(/*set_initial_x=*/!config.skip_init);

    ViolationLSLoop loop(model, vm, rng, fj, config, budget, hook, lns, lns_interval, callback,
                         coord);
    SearchResult result = loop.run();
    // This run's share of the thread's walks (#271), from the freeze to here.
    const ConeCacheCounters& cones = cone_cache_counters();
    result.counters.cone_cache_walks =
        static_cast<int64_t>(cones.cached_walks - cones_at_entry.cached_walks);
    result.counters.cone_fallback_walks =
        static_cast<int64_t>(cones.fallback_walks - cones_at_entry.fallback_walks);
    return result;
}

}  // namespace cbls
