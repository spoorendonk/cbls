#pragma once

#include "counters.h"
#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "move_generator.h"
#include "moves.h"
#include "randomize.h"
#include "rng.h"
#include "stop.h"
#include "violation.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace cbls {

/// Defined in `solution_pool.h`; forward-declared so `search.h` does not pull
/// in `<atomic>`, `<mutex>` and `SolutionPool`.
struct SearchCoordination;

/// Defined in `tracer.h`; forward-declared for the same reason.
class Tracer;

/// Defined in `audit.h` (#225), forward-declared for the same reason.
class SearchAuditor;

struct SearchConfig {
    // Keep the caller's assignment whole: no List/Set randomisation and no
    // FJ closest-to-zero scalar start.
    bool skip_init = false;
    // Total GLS iterations (not batches); 0 = unlimited (bounded by time_limit).
    // Checked at batch boundaries, so it may be overrun by up to
    // batch_iterations - 1. A Novelty move charges 3, a weight bump 1 (#209).
    // The batch count is held to the same limit, since Structural batches charge
    // no GLS iteration: the run stops when either count reaches it and
    // SearchResult::iterations reports the larger (#201).
    int64_t max_iterations = 0;
    int lns_interval = 3;

    // ViolationLS batch outer loop (Algorithm 6).
    int64_t batch_iterations = 1000;  // GLS iterations per batch; must be > 0
    int perturbation_period = 100;    // batches without improvement before perturbing
    // Share of the model a diversification kick moves: each scalar is
    // randomised with this probability; each List/Set gets
    // clamp(round(p * |elements|), 1, |elements|) structural moves (#111), so
    // every structure moves on every kick, p = 0 included. A kick moves at
    // least one variable (#109); one that hits the deadline stops between
    // moves (#115).
    double perturbation_probability = 0.1;
    // P(a batch is Structural: typed List/Set moves kept when they reduce
    // weighted violation). < 0 = automatic: 0.33 when the batch would build a
    // generator (an entry in move_generators, or a List/Set variable with
    // default_structural_generators on), else 0.0; 1.0 when it would and FJ
    // has no jumpable work: no Bool/Int/Float variable with >= 2 domain values
    // that some row reads, the folded objective included (#201). An explicit
    // value is honoured; on such a model its empty FJ/Novelty batches neither
    // kick nor count toward perturbation_period. A scalar read only by the
    // objective, or one FJ settles early, still counts as FJ work: set 1.0
    // explicitly (#201).
    double structural_batch_probability = -1.0;

    // ---- structural batch: what proposes moves, and how one is chosen (#165) --
    //
    // Extra move generators, on top of the built-in List/Set ones. Each
    // portfolio worker clones every generator, so per-generator state is
    // per-worker; the registered instances are never mutated (hence `const`)
    // and may be shared between searches.
    std::vector<std::shared_ptr<const MoveGenerator>> move_generators;
    // Also register the built-in List/Set generators (off = custom moves only).
    bool default_structural_generators = true;
    // How candidates become a commit. Auto = GranularBest when
    // `structural_neighbours` is set, else the pre-#165 rule (#203).
    StructuralSelection structural_selection = StructuralSelection::Auto;
    // Candidates scored per generator under BestOfSample / ViolationGuided /
    // GranularBest (inert under FirstImprovingSample); GranularBest tops its
    // one-element neighbourhoods up to this floor. 8 is an unmeasured
    // placeholder (#203); sweep it per instance set.
    int structural_sample_size = 8;
    // Optional k-nearest neighbourhood for the built-in generators: a move's
    // target is a neighbour of the moved element instead of a uniform draw
    // (after Toth & Vigo 2003; see generate_granular_partition_moves). Null
    // keeps the uniform draw; set, Auto becomes GranularBest (#203). Shared,
    // not cloned: it is immutable (#157).
    std::shared_ptr<const NeighbourList> structural_neighbours;
    // Run Novelty Jump batches (else FJ only). On by default since #209
    // bounded a Novelty batch's cost; the UC-CHPED runner turns it off.
    bool use_compound_moves = true;  // run Novelty Jump batches (else FJ only)
    // P(a scalar batch is Novelty Jump), drawn every batch, not held until a new
    // best as ViolationLS §5 does (#209; see pick_batch_kind).
    double novelty_jump_probability = 0.5;

    // A constraint is satisfied when its violation is <= this: absolute, on the
    // constraint node's value (|lhs - rhs| for an equality). See
    // kDefaultFeasibilityTolerance.
    double feasibility_tolerance = kDefaultFeasibilityTolerance;

    // Byte cap on the per-variable cone cache solve() builds when it freezes
    // the model (#271; see Model::freeze); shared by all workers; 0 caches
    // nothing. Changes no trajectory. A pre-frozen model keeps its own cap.
    size_t cone_cache_bytes = Model::kDefaultConeCacheBytes;

    // End an FJ batch after this many GLS iterations without reducing the
    // unweighted violation of the real rows, and take the diversification kick
    // as due (#102); <= 0 = the fixed perturbation_period cadence. The kick
    // does not arm the Float escape probe or reset stagnation (#107).
    // Forwarded to GFJConfig::unproductive_iterations. Bounded twice:
    //   1. it fires only when `stagnation >= perturbation_period / 20`, since
    //      its measure cannot see the `obj <= bound` row;
    //   2. once a feasible solution exists, it draws only the perturb half of a
    //      kick, never an LNS repair: that measure is blind after feasibility
    //      and a repair costs up to 2 s (#158; see SearchResult::lns_repairs).
    int64_t unproductive_iterations = 300;

    // Use the objective to break ties before the first feasible point (#218);
    // forwarded to GFJConfig::objective_tie_break (affine objectives only). On
    // here since #226; GFJConfig's own default stays off.
    bool fj_objective_tie_break = true;

    // ---- host integration (#169): cancellation and tracing ------------------
    //
    // Both are non-owning: the caller keeps the objects alive for the whole
    // solve. Fields, not solve() parameters, so a binding cannot drop them and
    // they reach every portfolio restart (run_worker copies this struct).

    // Host cancellation; unset = bounded by time_limit and max_iterations
    // alone. A raised stop ends the run at the next batch boundary with
    // TerminationReason::Cancelled. See StopRef for the lifetime rule.
    StopRef stop;

    // Host event sink, or null (one null compare per event site). Per worker:
    // share a Tracer between workers only if it is thread-safe; otherwise use
    // ParallelConfig::tracer_factory. See Tracer for granularity and cost.
    Tracer* tracer = nullptr;

    // Test-facing consistency hook (#225), or null; non-owning. See
    // SearchAuditor.
    SearchAuditor* auditor = nullptr;
};

/// Why solve()'s outer loop stopped; exactly one ends every run. The qualifier
/// on `time_seconds`: it tells a run that exhausted its budget from one that
/// converged, and lets a test prove which budget bound it (#104).
enum class TerminationReason : std::uint8_t {
    /// The wall-clock deadline from `solve()`'s `time_limit` expired.
    TimeLimit,
    /// `SearchConfig::max_iterations` was reached (GLS iterations, or the batch
    /// count when structural/novelty batches stall the iteration counter).
    IterationLimit,
    /// Pure-feasibility model (no objective): the first feasible solution is the
    /// answer, so the search stopped on finding one.
    Feasible,
    /// Neither a wall-clock nor an iteration budget was set: returned at once,
    /// having done no work.
    NoBudget,
    /// A peer worker ended the run (`SearchCoordination::stop`) while budget
    /// remained; only from ParallelSearch, when a worker solved a
    /// pure-feasibility model. Not TimeLimit: the clock did not run out.
    Stopped,
    /// The host cancelled (`SearchConfig::stop` or `ParallelConfig::stop`)
    /// while budget remained. Outranks Stopped (the host's cancel is the outer
    /// cause); Feasible outranks both.
    Cancelled,
};

/// Stable snake_case token for a TerminationReason ("time_limit",
/// "iteration_limit", "feasible", "no_budget", "stopped", "cancelled"): the
/// CLI's JSONL `termination` field and its human output. Static; never null.
const char* termination_reason_name(TerminationReason reason);

/// One `ParallelSearch` worker that did not complete, and why (#170). See
/// `SearchResult::workers_completed` for what "complete" means.
struct WorkerFailure {
    /// The worker's index in `[0, workers_launched)`.
    int worker = 0;
    /// Whether the worker returned at least one solve before it died; those
    /// solves are in the aggregate's iterations and counters, the throwing
    /// attempt is not. False means "no counters in the aggregate", not
    /// "contributed nothing": an incumbent it pooled may still be the answer.
    bool produced_result = false;
    /// `what()` of the exception that ended the worker (nanobind's traceback
    /// for a Python one), a fixed string for a non-std exception, or a fixed
    /// sentence for a worker the deadline starved before its first solve.
    std::string reason;
};

struct SearchResult {
    /// Objective at `best_state`, or +inf when there is nothing to report. +inf
    /// does not imply infeasible: a feasible point whose objective overflows is
    /// returned with `feasible = true` and +inf here (#100); test `feasible`,
    /// then `std::isfinite`. Minimised sense: under `maximize(e)` this is -e
    /// (see Model::is_maximizing); SolveProgress::objective follows suit.
    double objective = std::numeric_limits<double>::infinity();
    /// Whether `best_state` satisfies every real constraint, whatever the
    /// objective is there.
    bool feasible = false;
    Model::State best_state;
    /// The work max_iterations is charged against: the larger of the GLS
    /// iterations FJ and Novelty batches charged (Novelty move 3, bump 1) and
    /// the outer batch count (#201). Never 0 for a run that ran a batch;
    /// `counters` splits batches by kind.
    int64_t iterations = 0;
    double time_seconds = 0.0;
    /// Which budget ended the run; qualifies `iterations` and `time_seconds`.
    TerminationReason termination = TerminationReason::NoBudget;
    /// Largest real-constraint violation at `best_state` (<= the tolerance when
    /// `feasible`). On an infeasible run, the closest approach found.
    double best_violation = std::numeric_limits<double>::infinity();
    /// Whether the Float escape probe was armed at exit (#117): a latch, not a
    /// count, so false does not mean "never armed". ParallelSearch cannot sum
    /// or attribute a latch, so it always reads false there.
    bool escape_probe_armed = false;

    /// Diversification kicks taken (both the perturbation_period and the #102
    /// route); SolveProgress::perturbations sampled at exit. ParallelSearch
    /// sums it over workers and restarts: work done, not comparable to a
    /// single run at the same wall time.
    int perturbations = 0;

    /// LNS destroy-repair cycles attempted; counted apart from `perturbations`
    /// because a repair costs up to min(2.0, remaining()) s against a perturb's
    /// microseconds. Summed by ParallelSearch like `perturbations`.
    int lns_repairs = 0;

    /// The subset of `lns_repairs` whose `LNS::destroy_repair` returned true
    /// (#150); never greater than `lns_repairs`. The built-in LNS returns true
    /// when the repaired state beat the kick origin on (real violation,
    /// objective) and was kept; an override defines its own true. A rejection
    /// still draws from the RNG, spends the budget and rolls the assignment back
    /// to `kick_origin()` (#158). Summed by ParallelSearch like `perturbations`.
    int lns_repairs_accepted = 0;

    /// Objective at the first feasible point this run recorded (#149), before
    /// that batch's inner-solver polish and any bound tightening.
    /// Observational: nothing reads it back; the cost is one clock read, latched.
    /// May be non-finite on a run that did reach feasibility (#100), so test
    /// `time_to_first_feasible` for "ever feasible". 0.0 on a model with no
    /// objective. ParallelSearch reports the pair from the earliest worker and
    /// restart, with the time shifted onto the portfolio's clock.
    double first_feasible_objective = std::numeric_limits<double>::quiet_NaN();
    /// Seconds from solve()'s start to the first feasible point; NaN if and
    /// only if none was recorded. See `first_feasible_objective`.
    double time_to_first_feasible = std::numeric_limits<double>::quiet_NaN();

    /// Where the run spent its work: batches by kind, structural acceptance,
    /// inner-solver cost, portfolio restarts (#169). Summed by ParallelSearch
    /// through SearchCounters::merge. Observational; see
    /// SearchCounters::inner_solver_seconds for the one clock read it adds.
    SearchCounters counters;

    /// Portfolio workers launched and completed (#170). A worker is completed
    /// when its last solve() attempt returned, or when it never started because
    /// a peer had solved the model or the host had cancelled. It is not when its
    /// last attempt threw (also after earlier results or a retry cut off), when
    /// its model, hook, LNS or tracer factory threw, or when the deadline passed
    /// before it started. So
    /// `workers_completed + worker_failures.size() == workers_launched`.
    /// `workers_launched` is after the executor's cap; compare `workers_completed`
    /// with the threads asked for. A single solve() reports 1 and 1 (its throw
    /// propagates); ParallelSearch throws when every worker died without a
    /// result.
    int workers_launched = 1;
    int workers_completed = 1;
    /// One entry per worker that did not complete -- it threw, or the deadline
    /// starved it -- in worker order. Empty on every non-portfolio result.
    std::vector<WorkerFailure> worker_failures;
};

/// One progress row. Under ParallelSearch, `time_seconds`, `objective` and
/// `new_best` are the portfolio's (monotone; the last row matches the result),
/// while `iteration`, `total_violation`, `feasible` and `perturbations` are the
/// reporting worker's, so not monotone. See PortfolioProgress in src/pool.cpp.
struct SolveProgress {
    int64_t iteration = 0;
    double time_seconds = 0.0;
    double objective = std::numeric_limits<double>::infinity();
    /// Weighted violation over all rows, the `obj <= bound` row included, so
    /// routinely positive when feasible; ~1e30 on a feasible point with a
    /// non-finite objective (#116).
    double total_violation = 0.0;
    bool feasible = false;
    bool new_best = false;
    int perturbations = 0;  // diversification kicks so far (ViolationLS)
};

class SolveCallback {
public:
    virtual ~SolveCallback();
    virtual void on_progress(const SolveProgress& p) = 0;
};

/// Randomise every variable, scalars included. solve() does not call it (FJ
/// owns the scalar start); pair it with `SearchConfig::skip_init = true`. Safe
/// on unbounded domains: draws a finite window (#112).
void initialize_random(Model& model, RNG& rng);

/// Randomise only the List/Set variables; what solve() calls, since FJ's
/// `begin(set_initial_x)` sets the scalars to their closest-to-zero value (#108).
void initialize_structured_random(Model& model, RNG& rng);

/// Returns the GLS iterations the repair pass spent; below `max_iterations`
/// it converged or `time_limit` cut it short (#104).
int64_t fj_nl_initialize(Model& model, ViolationManager& vm, int max_iterations = 10000,
                         RNG* rng = nullptr, double time_limit = 2.0,
                         double tolerance = kDefaultFeasibilityTolerance);

/// Throws std::invalid_argument on a config the search cannot run: today
/// `batch_iterations <= 0`, which would let a counter-only loop ignore the time
/// limit (#211). solve() and ParallelSearch call it on entry.
void validate_search_config(const SearchConfig& config);

/// `time_limit <= 0` disables the wall clock: bounded by `max_iterations`
/// alone and deterministic per seed; with neither budget it returns at once.
/// `coord` is ParallelSearch's cross-worker channel; null everywhere else, and
/// then the trajectory is bit-identical to a run without it
/// (tests/test_parallel.cpp).
///
/// Freezes `model` first (#271; Model::freeze), before the clock starts; the
/// cone-cache build is O(config.cone_cache_bytes). Copies of the frozen model
/// share its `lambda_sum`/`pair_lambda_sum` callables, so a callable with
/// mutable state races under parallel copies.
SearchResult solve(Model& model, double time_limit = 10.0, uint64_t seed = 42,
                   InnerSolverHook* hook = nullptr, LNS* lns = nullptr, int lns_interval = 3,
                   SolveCallback* callback = nullptr, const SearchConfig& config = {},
                   SearchCoordination* coord = nullptr);

}  // namespace cbls
