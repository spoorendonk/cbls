#include "cbls/pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>

namespace cbls {

// --- SolutionPool ---

SolutionPool::SolutionPool(int capacity) : capacity_(std::max(1, capacity)) {}

bool SolutionPool::submit(Solution sol) {
    std::scoped_lock lock(mutex_);
    solutions_.push_back(std::move(sol));
    std::sort(solutions_.begin(), solutions_.end(), [](const Solution& a, const Solution& b) {
        if (a.feasible != b.feasible) {
            return a.feasible > b.feasible;
        }
        return a.objective < b.objective;
    });
    if (static_cast<int>(solutions_.size()) > capacity_) {
        solutions_.resize(capacity_);
    }
    return true;
}

std::optional<Solution> SolutionPool::best() const {
    std::scoped_lock lock(mutex_);
    if (solutions_.empty()) {
        return std::nullopt;
    }
    return solutions_[0];
}

std::vector<Solution> SolutionPool::top_k(int k) const {
    std::scoped_lock lock(mutex_);
    int n = std::min(k, static_cast<int>(solutions_.size()));
    n = std::max(0, n);
    return {solutions_.begin(), solutions_.begin() + n};
}

std::optional<Solution> SolutionPool::get_restart_point(RNG& rng) const {
    std::scoped_lock lock(mutex_);
    if (solutions_.empty()) {
        return std::nullopt;
    }
    int n = std::max(1, static_cast<int>(solutions_.size()) / 2);
    int idx = static_cast<int>(rng.integers(0, n));
    return solutions_[idx];
}

size_t SolutionPool::size() const {
    std::scoped_lock lock(mutex_);
    return solutions_.size();
}

uint64_t portfolio_worker_seed(uint64_t base_seed, int worker, int restart) {
    // splitmix64's finalizer, over a combination that separates the three
    // inputs before mixing.
    //
    // For a FIXED base the combination is injective over every (worker, restart)
    // range that occurs: a collision needs A*dworker + B*drestart == 0 (mod
    // 2^64), and with these two odd multipliers the smallest such pair has
    // |dworker| ~ 2^58 at drestart <= 32, and still ~2^41 at drestart <= 2^20.
    //
    // Across DIFFERENT bases it is only a sum, so a triple CAN in principle be
    // matched by shifting the base -- portfolio_worker_seed(0, 1, 0) equals
    // portfolio_worker_seed(A, 0, 0). That needs a base difference of A*dworker,
    // which no adjacent --seed can reach, and it is not a property anything
    // relies on. The finalizer then decorrelates neighbours, which is the
    // property that was missing.
    uint64_t x = base_seed;
    x += 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(worker) + 1);
    x += 0xBF58476D1CE4E5B9ULL * (static_cast<uint64_t>(restart) + 1);
    x ^= x >> 30U;
    x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27U;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 31U;
    return x;
}

// --- ParallelSearch ---

ParallelSearch::ParallelSearch(int n_threads) : n_threads_(n_threads) {}

int effective_pool_capacity(int requested, int n_threads) {
    if (requested > 0) {
        return requested;
    }
    // 0 or negative means auto. Negative is folded in here rather than clamped
    // to 1: a negative capacity is a caller error, and silently giving them a
    // one-solution pool -- which is what SolutionPool's own max(1, ...) clamp
    // used to do -- is the least useful reading of it.
    return std::max(10, 2 * n_threads);
}

int ParallelSearch::effective_threads(const ParallelConfig& pc) const {
    int n = pc.n_threads > 0 ? pc.n_threads : n_threads_;
    if (n <= 0) {
        n = static_cast<int>(std::thread::hardware_concurrency());
    }
    return std::max(1, n);
}

// Backward-compatible simple solve
SearchResult ParallelSearch::solve(std::function<Model()> model_factory, double time_limit,
                                   uint64_t seed) {
    ParallelConfig pc;
    pc.n_threads = n_threads_;
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> no_hook;
    std::function<std::shared_ptr<LNS>()> no_lns;
    int n = effective_threads(pc);
    return solve_portfolio(model_factory, time_limit, seed, {}, no_hook, no_lns, nullptr, n,
                           effective_pool_capacity(pc.pool_capacity, n), pc);
}

// Full-featured solve
SearchResult ParallelSearch::solve(
    std::function<Model()> model_factory, double time_limit, uint64_t seed,
    const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
    std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
    const ParallelConfig& par_config) {
    int n = effective_threads(par_config);
    return solve_portfolio(model_factory, time_limit, seed, config, hook_factory, lns_factory,
                           callback, n, effective_pool_capacity(par_config.pool_capacity, n),
                           par_config);
}

// --- Master-model overloads: one structure, N workers ---
//
// `freeze()` runs here, on the calling thread and before any worker exists, so
// the one structural rebuild `add_objective_soft_constraint` performs happens
// once rather than N times on a DAG N workers are already reading.
//
// The factory then simply copies the frozen master. That copy shares the
// structure and duplicates only what a search writes (see Model's copy
// constructor), and it is safe to run on several worker threads at once:
// nothing here writes to `master`, and copying a shared_ptr is atomic.
// Both delegate to the matching factory overload rather than repeating its
// argument list: a `std::function<Model()>` prvalue cannot bind to `Model&`, so
// the delegation is unambiguous, and one place then decides the default
// SearchConfig, the thread count and the pool capacity.
SearchResult ParallelSearch::solve(Model& master, double time_limit, uint64_t seed) {
    master.freeze();
    return solve(std::function<Model()>([&master]() { return master; }), time_limit, seed);
}

SearchResult ParallelSearch::solve(
    Model& master, double time_limit, uint64_t seed, const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
    std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
    const ParallelConfig& par_config) {
    master.freeze();
    return solve(std::function<Model()>([&master]() { return master; }), time_limit, seed, config,
                 std::move(hook_factory), std::move(lns_factory), callback, par_config);
}

// Portfolio workers are homogeneous — same model, same budget, different seed —
// so their termination reasons almost always agree, and this only has to break
// ties. Precedence: a worker that actually finished the job (Feasible) outranks
// everything, because that is the one exit that answers the question rather than
// running out of something. The Stopped branch below is DEFENSIVE and
// unreachable as the code stands -- the flag is raised only by a worker whose
// own run ended Feasible, and Feasible short-circuits this loop before
// any_stopped is consulted. It is kept so that a future second reason to raise
// the flag surfaces as "a peer ended this" rather than being reported as a
// budget exit. Cancelled sits directly below Feasible: a host that cancelled
// needs to read that back whatever its workers' own budgets did in the same
// instant, and a cancelled portfolio's `time_seconds` must not be read as an
// expired budget. Among budget exits the shared wall clock outranks the per-worker
// iteration budget, because the portfolio's answer is clock-limited as soon as
// any worker ran the clock out. NoBudget is last: it is also what a worker that
// threw leaves behind, and one crashed thread should not relabel a run the
// others budget-limited.
static TerminationReason aggregate_termination(const std::vector<SearchResult>& results) {
    bool any_cancelled = false;
    bool any_stopped = false;
    bool any_time = false;
    bool any_iterations = false;
    for (const auto& r : results) {
        if (r.termination == TerminationReason::Feasible) {
            return TerminationReason::Feasible;
        }
        any_cancelled = any_cancelled || r.termination == TerminationReason::Cancelled;
        any_stopped = any_stopped || r.termination == TerminationReason::Stopped;
        any_time = any_time || r.termination == TerminationReason::TimeLimit;
        any_iterations = any_iterations || r.termination == TerminationReason::IterationLimit;
    }
    if (any_cancelled) {
        return TerminationReason::Cancelled;
    }
    if (any_stopped) {
        return TerminationReason::Stopped;
    }
    if (any_time) {
        return TerminationReason::TimeLimit;
    }
    return any_iterations ? TerminationReason::IterationLimit : TerminationReason::NoBudget;
}

// ~thread calls std::terminate on a thread that is still joinable, so every
// launch loop below has to join what it managed to create even when the loop
// itself fails: threads.emplace_back can throw std::system_error at the process
// thread limit, or bad_alloc. Unwinding past a half-filled vector would abort
// the process instead of reporting the failure.
static void join_all(std::vector<std::thread>& threads) {
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

namespace {

// One worker's running total across its restarts. A restart is a fresh
// `cbls::solve()` on the SAME model, so the honest aggregate of two solves is
// the sum of their iterations and the better of their solutions -- not the last
// one's, which may be a restart that spent its remaining milliseconds finding
// nothing.
struct WorkerAccumulator {
    SearchResult result;
    bool any_run = false;

    // `started_at` is when the restart being absorbed BEGAN, on the portfolio
    // clock. Every time on a worker's SearchResult is relative to its own
    // solve() start, so the pair below has to be shifted onto the shared clock
    // or a restart that began at t=19.9s reports "feasible at 0.001s".
    // `restarted` is whether the run being absorbed is a RESTART rather than
    // this worker's first solve -- the one counter the accumulator has to derive
    // rather than read, since a SearchResult cannot know it was one.
    void absorb(const SearchResult& r, double started_at, bool restarted) {
        result.iterations += r.iterations;
        // Summed, not maxed, for the same reason iterations are: a worker's
        // restarts run BACK TO BACK on its own thread, so the wall time it held
        // is their total. The max would report the longest single restart --
        // milliseconds of a multi-second run under a tight max_iterations --
        // and solve_portfolio's max ACROSS workers would then publish that as
        // the run's duration.
        result.time_seconds += r.time_seconds;
        // Every restart's reason overwrites the previous one: the reason that
        // matters is the one that ended the worker, and the loop below breaks
        // on exactly the reasons worth reporting (Feasible, Stopped, NoBudget).
        result.termination = r.termination;
        // Work done across the restarts, summed with the iterations above.
        result.perturbations += r.perturbations;
        result.lns_repairs += r.lns_repairs;
        result.lns_repairs_accepted += r.lns_repairs_accepted;
        // Same rule, one struct over. Both aggregation sites go through
        // SearchCounters::merge -- this one across a worker's restarts, and
        // solve_portfolio's loop across the workers -- so the two cannot drift
        // apart the way a hand-written sum per site would.
        result.counters.merge(r.counters);
        if (restarted) {
            ++result.counters.portfolio_restarts;
        }
        // The earliest restart to reach feasibility, with the objective it
        // reached -- both from that same restart.
        const double reached = started_at + r.time_to_first_feasible;
        if (!std::isnan(reached) && (std::isnan(result.time_to_first_feasible) ||
                                     reached < result.time_to_first_feasible)) {
            result.time_to_first_feasible = reached;
            result.first_feasible_objective = r.first_feasible_objective;
        }
        const bool better = !any_run || (r.feasible && !result.feasible) ||
                            (r.feasible == result.feasible && r.objective < result.objective);
        if (better) {
            result.objective = r.objective;
            result.feasible = r.feasible;
            result.best_state = r.best_state;
            // Belongs to `best_state`, so it moves with it. `escape_probe_armed`
            // deliberately does NOT: it is a latch on one worker's END state,
            // and the state being returned came from the pool, which carries no
            // worker identity to attribute it to. The portfolio leaves it false
            // and include/cbls/search.h says so.
            result.best_violation = r.best_violation;
        }
        any_run = true;
    }
};

// The portfolio's progress stream.
//
// Every worker reports through this, not only worker 0. A consumer's subject is
// the PORTFOLIO's incumbent, and worker 0's own trajectory is neither the best
// of them nor, once it restarts, monotone in time -- so a benchmark harness
// integrating the callback as a step function of wall time was integrating the
// wrong function, and the CLI could print a last progress row worse than the
// result it went on to report. Two things are corrected here and nowhere else:
//
//  - `time_seconds` is rewritten to the PORTFOLIO clock. A worker's solve()
//    times from its own start, so a restarted worker's rows otherwise walk
//    backwards through the stream.
//  - `objective` is the GLOBAL incumbent. A row is forwarded when it improves
//    on that -- from any worker -- or when it is worker 0's periodic
//    no-new-best tick, which is what keeps the CLI's once-a-second liveness
//    row. Either way it carries the portfolio's best rather than a stale peer
//    value, so the stream stays monotone.
//
// Everything else on the row (iteration, violation, perturbations, feasible)
// remains the REPORTING worker's own, and is read as such: no worker has a view
// of any other's counters.
//
// Serialized on its own mutex, and the consumer is called under it: workers
// report concurrently, and a callback writing to one stream or file is not
// thread-safe on its own.
class PortfolioProgress {
public:
    PortfolioProgress(SolveCallback* inner, std::function<double()> elapsed)
        : inner_(inner), elapsed_(std::move(elapsed)) {}

    void report(const SolveProgress& p, bool heartbeat) {
        std::scoped_lock lock(mutex_);
        const bool improved = p.objective < best_;
        if (improved) {
            best_ = p.objective;
        } else if (!heartbeat) {
            return;
        }
        SolveProgress q = p;
        q.objective = best_;
        q.new_best = improved;
        q.time_seconds = elapsed_();
        inner_->on_progress(q);
    }

private:
    SolveCallback* inner_;
    std::function<double()> elapsed_;
    std::mutex mutex_;
    double best_ = std::numeric_limits<double>::infinity();
};

// One worker's end of `PortfolioProgress`. `heartbeat` is set for exactly one
// worker, whose non-improving ticks are what keep a liveness row flowing.
class WorkerProgress : public SolveCallback {
public:
    WorkerProgress(PortfolioProgress& core, bool heartbeat) : core_(core), heartbeat_(heartbeat) {}

    void on_progress(const SolveProgress& p) override { core_.report(p, heartbeat_); }

private:
    PortfolioProgress& core_;
    bool heartbeat_;
};

// Everything one worker is and does, gathered so `solve_portfolio` below is
// about running N of them and combining their answers rather than about both at
// once. Held by reference: it lives on the calling thread's stack for the whole
// call, and every member is either shared on purpose (the pool, the stop flag)
// or read-only.
struct PortfolioContext {
    std::function<Model()>& model_factory;
    std::function<std::shared_ptr<InnerSolverHook>(Model&)>& hook_factory;
    std::function<std::shared_ptr<LNS>()>& lns_factory;
    const SearchConfig& config;
    // Worker 0's end of the portfolio stream, and every other worker's. Both
    // null when the caller passed no callback. See `PortfolioProgress`.
    SolveCallback* heartbeat_callback;
    SolveCallback* peer_callback;
    SearchCoordination& coord;
    // Seconds left on the SHARED deadline -- one clock for the whole portfolio,
    // so a worker that restarts gets the time its predecessor left rather than
    // a fresh full budget. Returns 0.0 when there is no wall clock at all, in
    // which case `has_deadline` is false and the value is not a budget.
    std::function<double()> remaining;
    /// Seconds since the portfolio started. One clock for every worker, so a
    /// time reported by one of them means the same thing as a time reported by
    /// another -- which a worker's own solve() clock does not, least of all
    /// across restarts.
    std::function<double()> elapsed;
    /// Builds one Tracer per worker, or empty. See
    /// ParallelConfig::tracer_factory for why it is per worker rather than one
    /// instance shared through the progress wrapper.
    const std::function<std::unique_ptr<Tracer>(int)>& tracer_factory;
    bool has_deadline;
    uint64_t seed;
    int n_threads;
};

// One worker: its own model, its own hook and LNS, and a restart loop over the
// shared deadline. Returns nullopt when it produced no result at all -- it was
// handed no budget, or every attempt threw; every other outcome is a result
// worth aggregating.
//
// Throws nothing of its own -- the caller's catch is what keeps an exception off
// a thread function -- but the factories it calls may.

// How many times in a row a worker's `solve()` may throw before the worker
// gives up. Small on purpose: the case this exists for is a transient failure
// (a bad_alloc under memory pressure that the next restart does not hit), and a
// deterministic one must not spend the portfolio's whole budget re-raising.
constexpr int kMaxWorkerRetries = 3;

// The two cancellation channels a caller may set -- ParallelConfig::stop and
// SearchConfig::stop -- behind one StopRef, OR-ed. A caller may set either or
// both; branching on which would put the question at every poll site instead of
// once at setup. An instance lives on solve_portfolio's frame for the whole
// call, and the workers are joined before it returns, so the StopRef the workers
// hold never outlives it.
struct CombinedStop {
    StopRef host;
    StopRef search;
    [[nodiscard]] bool requested() const { return host.requested() || search.requested(); }
};

// What a portfolio that produced no result at all reports.
//
// A cancelled portfolio reaches that path too, and must NOT be reported as
// NoBudget: a stop raised before the first worker got going leaves every worker
// breaking out of its restart loop without ever calling solve(), so nothing is
// submitted and nothing throws. That run did not lack a budget -- the host took
// it away (#169).
TerminationReason empty_portfolio_reason(bool cancelled) {
    return cancelled ? TerminationReason::Cancelled : TerminationReason::NoBudget;
}

// Whether this restart's exit ends the worker rather than earning it another
// one. A peer answered the question, or the host cancelled; either way there is
// nothing left for this worker to do.
bool ends_worker(TerminationReason reason) {
    return reason == TerminationReason::Stopped || reason == TerminationReason::Cancelled;
}

// Whether the worker should not start another restart at all: a peer raised the
// shared flag, or the host cancelled.
bool worker_should_stop(const PortfolioContext& ctx) {
    return ctx.coord.stop->load(std::memory_order_relaxed) || ctx.config.stop.requested();
}

// One tracer for this worker, or none. Called on the worker's own thread, ONCE
// rather than once per restart: a restarted worker carries on with the same
// tracer, exactly as it carries on with the same model, hook and LNS. See
// ParallelConfig::tracer_factory, including why it is per worker.
std::unique_ptr<Tracer> make_worker_tracer(const PortfolioContext& ctx, int index) {
    if (!ctx.tracer_factory) {
        return nullptr;
    }
    return ctx.tracer_factory(index);
}

// The SearchConfig for ONE of a worker's restarts.
//
// Two things differ from the portfolio's own config, and both are per restart
// rather than per worker, which is why this is not folded into worker_config:
//
//  - the worker's own tracer, which overrides whatever SearchConfig::tracer
//    held; a single tracer shared by N workers is the hazard
//    ParallelConfig::tracer_factory exists to remove;
//  - skip_init on every restart after the first, so the worker keeps the
//    assignment it already holds -- its own incumbent, or a peer's if it adopted
//    one -- instead of throwing the run away and starting from the
//    closest-to-zero point again.
SearchConfig restart_config(const PortfolioContext& ctx, Tracer* tracer, int restart) {
    SearchConfig cfg = ctx.config;
    if (tracer != nullptr) {
        cfg.tracer = tracer;
    }
    if (restart > 0) {
        cfg.skip_init = true;
    }
    return cfg;
}

// `failure` is set to the last exception the SEARCH raised, whether or not the
// worker went on to recover. The caller reports it only when the worker
// produced nothing at all, so a worker that threw once and then succeeded is
// not counted as failed.
std::optional<SearchResult> run_worker(const PortfolioContext& ctx, int index,
                                       std::exception_ptr& failure) {
    // Built ONCE per worker, not once per restart: a restart carries on with
    // the model it already holds, which is also what makes `skip_init` below
    // mean "keep the assignment this worker converged to".
    Model m = ctx.model_factory();

    // Both die at the end of this function, i.e. on the worker thread. For a
    // factory that came from Python that is where the last reference to the
    // returned object is dropped, and nanobind's shared_ptr deleter takes the
    // GIL to do it.
    std::shared_ptr<InnerSolverHook> hook;
    if (ctx.hook_factory) {
        hook = ctx.hook_factory(m);
    }
    std::shared_ptr<LNS> lns;
    if (ctx.lns_factory) {
        lns = ctx.lns_factory();
    }
    // Dies with this function, on this worker's thread. See make_worker_tracer.
    const std::unique_ptr<Tracer> tracer = make_worker_tracer(ctx, index);

    // Every worker reports, through the portfolio stream that serializes them
    // and reconciles their clocks; worker 0 additionally carries the heartbeat.
    SolveCallback* cb = (index == 0) ? ctx.heartbeat_callback : ctx.peer_callback;

    WorkerAccumulator acc;
    int consecutive_failures = 0;
    // No idle threads while budget remains. A single solve() can return with
    // time left -- an exhausted SearchConfig::max_iterations is the case that
    // exists today -- and the core it was using would then sit out the rest of
    // the run. Restart it instead, on the time its predecessor left.
    for (int restart = 0;; ++restart) {
        // A peer answered the question, or the HOST cancelled (#169). The second
        // is checked here as well as inside `cbls::solve` so that a cancel
        // arriving between two restarts ends the worker rather than launching one
        // more solve that immediately returns Cancelled.
        if (worker_should_stop(ctx)) {
            break;
        }
        // ONE read of the shared clock per restart, reused as this solve's
        // budget. Reading it twice lets the deadline pass between the guard and
        // the call: the second read then returns exactly 0.0, which
        // make_budget reads as "no wall clock at all" -- and a worker carrying
        // a SearchConfig::max_iterations budget would then run that budget
        // whole, past the deadline the portfolio set, with join_all waiting on
        // it.
        const double budget = ctx.remaining();
        if (ctx.has_deadline && budget <= 0.0) {
            break;
        }
        const SearchConfig cfg = restart_config(ctx, tracer.get(), restart);
        // Distinct per (worker, restart), and decorrelated ACROSS base seeds --
        // see portfolio_worker_seed.
        const uint64_t run_seed = portfolio_worker_seed(ctx.seed, index, restart);

        SearchResult r;
        const double started_at = ctx.elapsed();
        try {
            r = cbls::solve(m, budget, run_seed, cfg.use_fj, hook.get(), lns.get(),
                            cfg.lns_interval, cb, cfg, &ctx.coord);
        } catch (...) {
            // The model, hook and LNS are already built, so this throw came from
            // the SEARCH -- a bad_alloc on a large model, a throwing custom
            // hook -- not from a factory that would throw again identically.
            // Leaving the worker dead would idle its core for the rest of the
            // run while its peers continue, which is exactly the property this
            // class claims not to have. So retry, bounded: a search that fails
            // this many times consecutively is failing deterministically and
            // retrying it only burns the shared budget.
            //
            // The factory is NOT retried, deliberately -- it is called once per
            // worker, above this loop, and a factory that cannot build its model
            // will not build it on the second ask either.
            failure = std::current_exception();
            if (++consecutive_failures >= kMaxWorkerRetries) {
                break;
            }
            continue;
        }
        consecutive_failures = 0;
        acc.absorb(r, started_at, /*restarted=*/restart > 0);

        if (r.termination == TerminationReason::Feasible) {
            // A pure-feasibility model: the first feasible solution IS the
            // answer, so every other worker is now searching a settled
            // question. Stop them rather than leaving them to run their budget
            // out.
            //
            // Raised BEFORE the no-deadline break below, and the order is the
            // whole point: an iteration-budgeted portfolio has no clock to run
            // out, so its peers would otherwise grind their full iteration
            // budgets on a question already answered -- which is exactly what
            // pool.h promises does not happen, unqualified.
            ctx.coord.stop->store(true, std::memory_order_relaxed);
            break;
        }
        if (!ctx.has_deadline) {
            // No SHARED wall clock, so there is no "time the predecessor left"
            // for a restart to run on: the worker's iteration budget IS the
            // whole budget it was given, and restarting would hand it that
            // budget again, forever. An IterationLimit return is not one of the
            // exits below, so without this the loop never ends -- and
            // `time_limit <= 0` with SearchConfig::max_iterations set is a
            // supported call shape, reachable from C++ and from Python.
            break;
        }

        if (ends_worker(r.termination)) {
            break;
        }
        if (r.termination == TerminationReason::NoBudget) {
            // Neither a wall clock nor an iteration budget: restarting would
            // spin on a solve that does no work.
            break;
        }
    }

    if (!acc.any_run) {
        return std::nullopt;
    }
    return acc.result;
}

}  // namespace

// --- Cooperative portfolio ---

SearchResult ParallelSearch::solve_portfolio(
    std::function<Model()>& model_factory, double time_limit, uint64_t seed,
    const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)>& hook_factory,
    std::function<std::shared_ptr<LNS>()>& lns_factory, SolveCallback* callback, int n_threads,
    int pool_capacity, const ParallelConfig& par_config) {
    // One StopRef for every worker to poll; see CombinedStop.
    const CombinedStop combined{par_config.stop, config.stop};
    SearchConfig worker_config = config;
    worker_config.stop = combined;

    SolutionPool pool(pool_capacity);
    std::atomic<bool> stop{false};
    SearchCoordination coord{&pool, &stop};

    const bool has_deadline = time_limit > 0.0;
    const auto portfolio_start = std::chrono::steady_clock::now();
    // Saturated before the integer-tick cast, exactly as search.cpp's
    // make_budget does and for the same reason: callers pass a very large limit
    // to mean "effectively unbounded", and casting 1e12 seconds to nanoseconds
    // overflows int64 and yields a deadline already in the past. Here that
    // would make every worker break before its first solve and hand the caller
    // a NoBudget result with an empty state -- where make_budget's own
    // saturation would have run the search.
    constexpr double kMaxPortfolioSeconds = 1.0e9;  // ~31 years
    const auto deadline =
        portfolio_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(
                                  std::min(std::max(0.0, time_limit), kMaxPortfolioSeconds)));
    auto elapsed = [portfolio_start]() -> double {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - portfolio_start)
            .count();
    };
    auto remaining = [deadline, has_deadline]() -> double {
        if (!has_deadline) {
            return 0.0;  // no wall clock: a worker's own iteration budget bounds it
        }
        return std::max(
            0.0,
            std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count());
    };

    // The caller's callback reaches the workers only through this: one stream,
    // one clock, one monotone objective. Null in, null out -- a portfolio with
    // no callback pays for nothing.
    PortfolioProgress progress(callback, elapsed);
    WorkerProgress heartbeat(progress, /*heartbeat=*/true);
    WorkerProgress peer(progress, /*heartbeat=*/false);

    PortfolioContext ctx{model_factory,
                         hook_factory,
                         lns_factory,
                         worker_config,
                         callback != nullptr ? &heartbeat : nullptr,
                         callback != nullptr ? &peer : nullptr,
                         coord,
                         remaining,
                         elapsed,
                         par_config.tracer_factory,
                         has_deadline,
                         seed,
                         n_threads};

    std::vector<SearchResult> results(n_threads);
    // One slot per worker, left null unless that worker threw. Sized up front so
    // the lambdas below only ever write their own index.
    //
    // A slot may own a Python object -- a raising Python callback or factory
    // arrives as nanobind's python_error -- and this vector dies on the calling
    // thread, which reached here through a binding that released the GIL. That is
    // safe only because python_error takes the GIL in its own destructor; see
    // PySolveCallback in python/bindings.cpp before changing what gets parked.
    std::vector<std::exception_ptr> failures(n_threads);
    std::vector<std::thread> threads;

    try {
        for (int i = 0; i < n_threads; ++i) {
            threads.emplace_back([&, i]() {
                try {
                    std::exception_ptr worker_failure;
                    auto r = run_worker(ctx, i, worker_failure);
                    if (!r.has_value()) {
                        // Either handed no budget at all (worker_failure null,
                        // nothing to report) or every attempt threw.
                        failures[i] = worker_failure;
                        return;
                    }
                    results[i] = *r;

                    // The end-of-run submit still matters even though every
                    // incumbent was shared as it was found: on a run that never
                    // reached feasibility nothing was ever recorded, and this is
                    // what puts the closest approach in the pool so the
                    // aggregate below has something to return.
                    Solution sol;
                    sol.state = r->best_state;
                    sol.objective = r->objective;
                    sol.feasible = r->feasible;
                    sol.violation = r->best_violation;
                    pool.submit(sol);
                } catch (...) {
                    // A thread function must not let an exception escape -- that
                    // is std::terminate -- and one worker failing is not a
                    // reason to lose the others' work. Park it rather than drop
                    // it; whether it is rethrown is decided below, once every
                    // worker has reported.
                    failures[i] = std::current_exception();
                }
            });
        }
    } catch (...) {
        stop.store(true, std::memory_order_relaxed);
        join_all(threads);
        throw;
    }

    join_all(threads);

    // Every worker either submits a solution or parks its exception, and the
    // pool keeps the best `pool_capacity`, so an empty pool means every one of
    // them threw. Report that: a default SearchResult would say "searched, found
    // nothing feasible" about a run that never searched. A partial failure does
    // not reach here -- the survivors submitted, which is the point of catching.
    //
    // The lowest-index failure is the one reported. Portfolio workers are
    // homogeneous -- same model, same budget, different seed -- so when all of
    // them fail they have almost always failed the same way, and aggregating N
    // copies of one message would buy nothing.
    // "Every worker threw" is the contract, and the pool alone can no longer
    // decide it: a worker now shares incumbents DURING its run and may throw on
    // a later restart, leaving a non-empty pool behind. Ask the failures
    // directly, so a portfolio in which nothing survived reports that rather
    // than returning one dead worker's mid-run snapshot as a result.
    const bool all_failed =
        !failures.empty() && std::all_of(failures.begin(), failures.end(),
                                         [](const std::exception_ptr& f) { return f != nullptr; });
    auto best = pool.best();
    if (all_failed || !best) {
        for (const auto& f : failures) {
            if (f) {
                std::rethrow_exception(f);
            }
        }
        // Reached with no failure only when every worker found the shared
        // deadline already past before its first solve -- a positive time limit
        // so small it expired during thread creation -- so none of them ran and
        // none of them submitted. That is the same "did no work" case a single
        // solve() reports as NoBudget. Say so rather than returning a
        // feasible-looking default. (A NON-positive time limit does not reach
        // here: it disables the wall clock, the worker runs exactly one solve,
        // and that solve's result is submitted.)
        // A cancelled portfolio reaches here too; see empty_portfolio_reason.
        SearchResult empty;
        empty.termination = empty_portfolio_reason(combined.requested());
        return empty;
    }

    SearchResult result;
    result.objective = best->objective;
    result.feasible = best->feasible;
    result.best_state = best->state;
    // Of the state being returned, not of whatever worker 0 ended on. Left at
    // its +inf default this reported "the answer violates something by
    // infinity" for EVERY portfolio run, which any caller that gates publication
    // on the residual reads as a defective solution -- benchmarks/mipfeas does,
    // and refused every parallel row it was handed.
    result.best_violation = best->violation;
    // Sum iterations and take max time across threads
    int64_t total_iters = 0;
    double max_time = 0.0;
    // The counters below describe work DONE, so they sum over the portfolio the
    // way iterations do; a portfolio that reported zero perturbations and zero
    // LNS repairs while its workers ran thousands is an ablation measured wrong.
    for (const auto& r : results) {
        total_iters += r.iterations;
        max_time = std::max(max_time, r.time_seconds);
        result.perturbations += r.perturbations;
        result.lns_repairs += r.lns_repairs;
        result.lns_repairs_accepted += r.lns_repairs_accepted;
        result.counters.merge(r.counters);
        // The portfolio reached feasibility when its FIRST worker did, and the
        // objective reported beside that time is the one that worker reached --
        // the pair has to come from the same worker to mean anything.
        if (!std::isnan(r.time_to_first_feasible) &&
            (std::isnan(result.time_to_first_feasible) ||
             r.time_to_first_feasible < result.time_to_first_feasible)) {
            result.time_to_first_feasible = r.time_to_first_feasible;
            result.first_feasible_objective = r.first_feasible_objective;
        }
    }
    result.iterations = total_iters;
    result.time_seconds = max_time;
    result.termination = aggregate_termination(results);
    return result;
}

}  // namespace cbls
