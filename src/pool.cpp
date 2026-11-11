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
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cbls {

bool ranks_before(bool feasible_a, double objective_a, double violation_a, bool feasible_b,
                  double objective_b, double violation_b) {
    if (feasible_a != feasible_b) {
        return feasible_a;
    }
    if (!feasible_a) {
        const double va =
            std::isnan(violation_a) ? std::numeric_limits<double>::infinity() : violation_a;
        const double vb =
            std::isnan(violation_b) ? std::numeric_limits<double>::infinity() : violation_b;
        if (va != vb) {
            return va < vb;
        }
    }
    return objective_a < objective_b;
}

// --- SolutionPool ---

SolutionPool::SolutionPool(int capacity) : capacity_(std::max(1, capacity)) {}

bool SolutionPool::submit(Solution sol) {
    std::scoped_lock lock(mutex_);
    // Under the mutex, so concurrent submitters cannot interleave load and store
    // and lose the better value.
    if (sol.feasible && std::isfinite(sol.objective) &&
        sol.objective < best_feasible_objective_.load(std::memory_order_relaxed)) {
        best_feasible_objective_.store(sol.objective, std::memory_order_relaxed);
    }
    solutions_.push_back(std::move(sol));
    std::sort(solutions_.begin(), solutions_.end(), [](const Solution& a, const Solution& b) {
        return ranks_before(a.feasible, a.objective, a.violation, b.feasible, b.objective,
                            b.violation);
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
    // For a fixed base the pre-mix sum is injective over every (worker, restart)
    // range that occurs: a collision needs A*dworker + B*drestart == 0 (mod 2^64),
    // whose smallest solution has |dworker| ~ 2^58 at drestart <= 32, ~2^41 at
    // drestart <= 2^20. Across bases it is only a sum (seed(0, 1, 0) ==
    // seed(A, 0, 0)), unreachable by adjacent seeds and relied on by nothing.
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
    // <= 0 is auto; a negative capacity gets auto, not a one-solution pool.
    return std::max(10, 2 * n_threads);
}

int ParallelSearch::effective_threads(const ParallelConfig& pc) const {
    int n = pc.n_threads > 0 ? pc.n_threads : n_threads_;
    if (n <= 0) {
        n = static_cast<int>(std::thread::hardware_concurrency());
    }
    return std::max(1, n);
}

SearchResult ParallelSearch::solve(std::function<Model()> model_factory, double time_limit,
                                   uint64_t seed) {
    ParallelConfig pc;
    pc.n_threads = n_threads_;
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> no_hook;
    std::function<std::shared_ptr<LNS>()> no_lns;
    return solve_portfolio(model_factory, time_limit, seed, {}, no_hook, no_lns, nullptr,
                           effective_threads(pc), pc);
}

SearchResult ParallelSearch::solve(
    std::function<Model()> model_factory, double time_limit, uint64_t seed,
    const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
    std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
    const ParallelConfig& par_config) {
    return solve_portfolio(model_factory, time_limit, seed, config, hook_factory, lns_factory,
                           callback, effective_threads(par_config), par_config);
}

// --- Master-model overloads: one structure, N workers ---
//
// `freeze()` runs on the calling thread before any worker exists, so the
// objective-row rebuild happens once, not on a DAG workers are reading. The
// factory's copy shares structure and is safe on concurrent threads: nothing
// writes `master`, and copying a shared_ptr is atomic. Delegating to the factory
// overload (a prvalue cannot bind `Model&`) keeps the defaults in one place.
SearchResult ParallelSearch::solve(Model& master, double time_limit, uint64_t seed) {
    master.freeze();
    return solve(std::function<Model()>([&master]() { return master; }), time_limit, seed);
}

SearchResult ParallelSearch::solve(
    Model& master, double time_limit, uint64_t seed, const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
    std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
    const ParallelConfig& par_config) {
    master.freeze(config.cone_cache_bytes);
    return solve(std::function<Model()>([&master]() { return master; }), time_limit, seed, config,
                 std::move(hook_factory), std::move(lns_factory), callback, par_config);
}

// Breaks ties among workers' reasons. Feasible first: the one exit that answers
// the question. Cancelled next, so a host reads back its cancel whatever budgets
// did, and its `time_seconds` is not read as an expired budget. Stopped (a raised
// flag with no Feasible worker) is reported as a peer's stop, not a budget exit.
// TimeLimit over IterationLimit: one worker running out the shared clock makes
// the answer clock-limited. NoBudget last: a worker that threw leaves it too.
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

// ~thread on a joinable thread is std::terminate, so a failed launch loop must
// still join what it created (see launch_workers).
static void join_all(std::vector<std::thread>& threads) {
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }
}

namespace {

// One worker's total across its restarts (fresh solve()s on the same model):
// work summed, the better solution kept, not the last restart's.
struct WorkerAccumulator {
    SearchResult result;
    bool any_run = false;

    // `started_at`: when this restart began on the portfolio clock, to shift
    // the first-feasible time off the restart's own solve() clock.
    void absorb(const SearchResult& r, double started_at) {
        result.iterations += r.iterations;
        // Summed: restarts run back to back, so the worker's wall time is their
        // total; a max would publish the longest single restart as the run's.
        result.time_seconds += r.time_seconds;
        // Overwritten: the reason that matters is the one that ended the worker.
        result.termination = r.termination;
        result.perturbations += r.perturbations;
        result.lns_repairs += r.lns_repairs;
        result.lns_repairs_accepted += r.lns_repairs_accepted;
        // Both aggregation sites (restarts here, workers in solve_portfolio) use
        // SearchCounters::merge, so they cannot drift apart.
        result.counters.merge(r.counters);
        // Every absorbed run after the first is a restart; from `any_run`, not
        // the loop index, since a restart that threw never reaches here.
        if (any_run) {
            ++result.counters.portfolio_restarts;
        }
        // The earliest restart to reach feasibility, both values from it.
        const double reached = started_at + r.time_to_first_feasible;
        if (!std::isnan(reached) && (std::isnan(result.time_to_first_feasible) ||
                                     reached < result.time_to_first_feasible)) {
            result.time_to_first_feasible = reached;
            result.first_feasible_objective = r.first_feasible_objective;
        }
        const bool better =
            !any_run || ranks_before(r.feasible, r.objective, r.best_violation, result.feasible,
                                     result.objective, result.best_violation);
        if (better) {
            result.objective = r.objective;
            result.feasible = r.feasible;
            result.best_state = r.best_state;
            // Moves with `best_state`. `escape_probe_armed` does not: a latch on
            // one worker's end state cannot be attributed (search.h).
            result.best_violation = r.best_violation;
        }
        any_run = true;
    }
};

// The portfolio's progress stream; every worker reports through it. Rewrites
// `time_seconds` to the portfolio clock (a restarted worker's would run
// backwards) and `objective` to the global incumbent; forwards a row only when it
// improves that, or as worker 0's heartbeat tick, so the stream is monotone.
// Other fields stay the reporting worker's (see SolveProgress). The consumer is
// called under the mutex, since workers report concurrently.
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

// What every worker shares. Held by reference: it lives on the calling thread's
// stack for the whole call, and every member is shared on purpose (the pool,
// the stop flag) or read-only.
struct PortfolioContext {
    std::function<Model()>& model_factory;
    std::function<std::shared_ptr<InnerSolverHook>(Model&)>& hook_factory;
    std::function<std::shared_ptr<LNS>()>& lns_factory;
    const SearchConfig& config;
    // Worker 0's end of the stream and every other's; null without a callback.
    SolveCallback* heartbeat_callback;
    SolveCallback* peer_callback;
    SearchCoordination& coord;
    // Seconds left on the shared deadline, so a restart gets what its
    // predecessor left. 0.0 without a wall clock (`has_deadline` false), where
    // it is not a budget.
    std::function<double()> remaining;
    /// Seconds since the portfolio started: one clock for every worker.
    std::function<double()> elapsed;
    /// Builds one Tracer per worker, or empty (ParallelConfig::tracer_factory).
    const std::function<std::unique_ptr<Tracer>(int)>& tracer_factory;
    bool has_deadline;
    uint64_t seed;
};

// Consecutive solve() throws before a worker gives up. Small: it is for a
// transient failure (a bad_alloc the next restart does not hit), and a
// deterministic one must not spend the budget re-raising.
constexpr int kMaxWorkerRetries = 3;

// ParallelConfig::stop OR SearchConfig::stop behind one StopRef. Lives on
// solve_portfolio's frame, and workers are joined before it returns, so their
// StopRef never outlives it.
struct CombinedStop {
    StopRef host;
    StopRef search;
    [[nodiscard]] bool requested() const { return host.requested() || search.requested(); }
};

// Workers a portfolio runs: capped by an executor's width, since a worker holds
// its chunk for the whole deadline and a late one would get no budget. Floored at
// 1: a zero-worker portfolio returns no result.
int portfolio_workers(int requested, const std::optional<ExecutorRef>& executor) {
    if (!executor.has_value()) {
        return std::max(1, requested);
    }
    return std::max(1, std::min(requested, executor->n_threads()));
}

// Run `run_one(i)` for every worker index in `[0, n_workers)`, concurrently.
// `run_one` swallows worker exceptions, so only a launch failure reaches here.
//
// With an executor (#169): no thread is created; any chunking works (grouped
// indices run serially, each on the deadline as it stands); workers are keyed by
// index, not `chunk_idx`. An executor exception is the caller's, so uncaught.
//
// On our own threads, emplace_back can throw (thread limit, bad_alloc): raise
// the stop flag and join what exists before rethrowing, since ~thread on a
// joinable thread is std::terminate.
void launch_workers(int n_workers, const std::optional<ExecutorRef>& executor,
                    std::atomic<bool>& stop, detail::FunctionRef<void(int)> run_one) {
    if (executor.has_value()) {
        executor->parallel_for_chunked(
            0, n_workers, [n_workers, run_one](int begin, int end, int /*chunk_idx*/) {
                // Clamped to the requested range: `run_one` indexes two heap
                // vectors by `i`, and a mis-chunking executor would write past
                // them (#156's hazard class). One compare per worker turns that
                // into a dropped worker, not heap corruption.
                for (int i = std::max(0, begin); i < std::min(end, n_workers); ++i) {
                    run_one(i);
                }
            });
        return;
    }
    std::vector<std::thread> threads;
    try {
        for (int i = 0; i < n_workers; ++i) {
            threads.emplace_back([run_one, i]() { run_one(i); });
        }
    } catch (...) {
        stop.store(true, std::memory_order_relaxed);
        join_all(threads);
        throw;
    }
    join_all(threads);
}

// What a portfolio with no result reports. A cancel before any worker started
// lands here too and is Cancelled, not NoBudget (#169).
TerminationReason empty_portfolio_reason(bool cancelled) {
    return cancelled ? TerminationReason::Cancelled : TerminationReason::NoBudget;
}

// Whether this exit ends the worker rather than earning another restart: a
// peer's stop or the host's cancel.
bool ends_worker(TerminationReason reason) {
    return reason == TerminationReason::Stopped || reason == TerminationReason::Cancelled;
}

// A host cancel outranks an IterationLimit or NoBudget aggregate: a worker can
// finish an iteration budget and only then see the cancel, between restarts.
// Applied to the aggregate, not in the worker loop, because it covers strictly
// more (a cancel after the loop too) and is testable (tests/test_stop.cpp); no
// host code runs between restarts. TimeLimit stays (the clock is its own cause,
// as in ViolationLSLoop::run), and so does Feasible.
TerminationReason with_host_cancel(TerminationReason aggregate, bool cancelled) {
    if (!cancelled) {
        return aggregate;
    }
    if (aggregate == TerminationReason::IterationLimit ||
        aggregate == TerminationReason::NoBudget) {
        return TerminationReason::Cancelled;
    }
    return aggregate;
}

// Whether the worker should not start another restart at all: a peer raised the
// shared flag, or the host cancelled.
bool worker_should_stop(const PortfolioContext& ctx) {
    return ctx.coord.stop->load(std::memory_order_relaxed) || ctx.config.stop.requested();
}

// One tracer for this worker, or none; called once on its thread, spanning its
// restarts like its model, hook and LNS.
std::unique_ptr<Tracer> make_worker_tracer(const PortfolioContext& ctx, int index) {
    if (!ctx.tracer_factory) {
        return nullptr;
    }
    return ctx.tracer_factory(index);
}

// The SearchConfig for one restart:
//  - the worker's tracer whenever a factory exists, null included: otherwise a
//    declining worker would share the caller's `SearchConfig::tracer` across
//    threads, the race tracer_factory removes;
//  - skip_init after the first, so the worker keeps the assignment it holds.
SearchConfig restart_config(const PortfolioContext& ctx, Tracer* tracer, int restart) {
    SearchConfig cfg = ctx.config;
    if (ctx.tracer_factory) {
        cfg.tracer = tracer;
    }
    if (restart > 0) {
        cfg.skip_init = true;
    }
    return cfg;
}

// Whether the restart loop ends after the solve that returned `r`; raises the
// shared stop flag when it answered the question. The order of tests matters.
bool worker_finished(const PortfolioContext& ctx, const SearchResult& r) {
    if (r.termination == TerminationReason::Feasible) {
        // Pure feasibility: the answer is found, so stop every peer. Before the
        // no-deadline exit below, or an iteration-budgeted portfolio's peers
        // would run their full budgets (pool.h promises they stop).
        ctx.coord.stop->store(true, std::memory_order_relaxed);
        return true;
    }
    if (!ctx.has_deadline) {
        // No shared clock: the iteration budget is the whole budget, and a
        // restart would grant it again forever (IterationLimit is not an exit
        // below). `time_limit <= 0` with max_iterations is a supported call.
        return true;
    }
    // Neither a wall clock nor an iteration budget (NoBudget): restarting would
    // spin on a solve that does no work.
    return ends_worker(r.termination) || r.termination == TerminationReason::NoBudget;
}

// How a worker ended, for the result's worker accounting (#170, see
// SearchResult::workers_completed).
struct WorkerEnd {
    // What the last solve attempt threw, or null if it returned (or none ran).
    // A successful retry clears it; a later fatal restart sets it even after
    // earlier results.
    std::exception_ptr failure;
    // The deadline passed before the first attempt: work but no time (a serial
    // executor, or a budget that expired during launch), unlike a stop or cancel.
    bool starved = false;
};

// One worker: its own model, hook and LNS, and a restart loop over the shared
// deadline. nullopt when it produced no result (no budget, or every attempt
// threw). Throws nothing itself, but the factories it calls may.
std::optional<SearchResult> run_worker(const PortfolioContext& ctx, int index, WorkerEnd& end) {
    std::exception_ptr& failure = end.failure;
    // Once per worker: restarts continue on this model, which is what makes
    // `skip_init` keep the converged assignment.
    Model m = ctx.model_factory();

    // Both die on the worker thread; for a Python factory nanobind's
    // shared_ptr deleter takes the GIL there.
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

    // Worker 0 also carries the heartbeat.
    SolveCallback* cb = (index == 0) ? ctx.heartbeat_callback : ctx.peer_callback;

    WorkerAccumulator acc;
    int consecutive_failures = 0;
    // A solve() that returns with time left (e.g. max_iterations exhausted) is
    // restarted on the time it left, so no core idles while budget remains.
    for (int restart = 0;; ++restart) {
        // Checked here as well as in cbls::solve, so a cancel between restarts
        // launches no solve that returns Cancelled at once (#169).
        if (worker_should_stop(ctx)) {
            break;
        }
        // One clock read, reused as the budget: a second read could return 0.0,
        // which make_budget takes as "no wall clock", and the solve would run
        // its max_iterations whole past the deadline.
        const double budget = ctx.remaining();
        if (ctx.has_deadline && budget <= 0.0) {
            end.starved = restart == 0;
            break;
        }
        const SearchConfig cfg = restart_config(ctx, tracer.get(), restart);
        // Distinct per (worker, restart); see portfolio_worker_seed.
        const uint64_t run_seed = portfolio_worker_seed(ctx.seed, index, restart);

        SearchResult r;
        const double started_at = ctx.elapsed();
        try {
            r = cbls::solve(m, budget, run_seed, cfg.use_fj, hook.get(), lns.get(),
                            cfg.lns_interval, cb, cfg, &ctx.coord);
        } catch (...) {
            // The search threw (bad_alloc, a throwing hook), not a factory:
            // retry, bounded by kMaxWorkerRetries, so the core does not idle.
            // The factories ran once above and are not retried: one that failed
            // would fail again.
            failure = std::current_exception();
            if (++consecutive_failures >= kMaxWorkerRetries) {
                break;
            }
            continue;
        }
        consecutive_failures = 0;
        failure = nullptr;
        acc.absorb(r, started_at);

        if (worker_finished(ctx, r)) {
            break;
        }
    }

    if (!acc.any_run) {
        return std::nullopt;
    }
    return acc.result;
}

// The text a WorkerFailure carries; called after every worker joined. A parked
// python_error takes the GIL in its what() and destructor (PySolveCallback in
// python/bindings.cpp).
std::string describe_failure(const std::exception_ptr& failure) {
    try {
        std::rethrow_exception(failure);
    } catch (const std::exception& e) {
        return e.what();
    } catch (...) {
        return "an exception not derived from std::exception";
    }
}

// The reason a starved worker (WorkerEnd::starved) is listed with.
constexpr const char* kStarvedReason =
    "the shared deadline had passed before this worker's first solve";

// The result's worker accounting (#170): the one place that applies
// SearchResult::workers_completed's definition.
void report_workers(const std::vector<char>& produced,
                    const std::vector<std::exception_ptr>& ended_by,
                    const std::vector<char>& starved, SearchResult& result) {
    result.workers_launched = static_cast<int>(produced.size());
    result.workers_completed = 0;
    result.worker_failures.clear();
    for (size_t i = 0; i < produced.size(); ++i) {
        if (ended_by[i] != nullptr) {
            result.worker_failures.push_back(WorkerFailure{static_cast<int>(i), produced[i] != 0,
                                                           describe_failure(ended_by[i])});
        } else if (starved[i] != 0) {
            result.worker_failures.push_back(
                WorkerFailure{static_cast<int>(i), false, kStarvedReason});
        } else {
            ++result.workers_completed;
        }
    }
}

}  // namespace

// --- Cooperative portfolio ---

SearchResult ParallelSearch::solve_portfolio(
    std::function<Model()>& model_factory, double time_limit, uint64_t seed,
    const SearchConfig& config,
    std::function<std::shared_ptr<InnerSolverHook>(Model&)>& hook_factory,
    std::function<std::shared_ptr<LNS>()>& lns_factory, SolveCallback* callback, int n_threads,
    const ParallelConfig& par_config) {
    // Every per-worker vector and the aggregate are sized on this, not the
    // request: a never-run slot would read as a worker that found nothing.
    const int n_workers = portfolio_workers(n_threads, par_config.executor);
    validate_search_config(config);

    // One SearchConfig::tracer without a factory would be called from every
    // worker thread: refused, since dropping it loses events and keeping it races.
    // A one-worker portfolio has no peer to race, so it is allowed.
    if (config.tracer != nullptr && !par_config.tracer_factory && n_workers > 1) {
        throw std::invalid_argument(
            "ParallelSearch: SearchConfig::tracer is set with no ParallelConfig::tracer_factory; "
            "one tracer cannot be shared by several workers -- supply a factory");
    }

    // One StopRef for every worker to poll; see CombinedStop.
    const CombinedStop combined{par_config.stop, config.stop};

    SearchConfig worker_config = config;
    worker_config.stop = combined;

    // Auto capacity over the workers that run; an explicit one is honoured.
    SolutionPool pool(effective_pool_capacity(par_config.pool_capacity, n_workers));
    std::atomic<bool> stop{false};
    SearchCoordination coord{&pool, &stop, par_config.share_objective_bound};

    const bool has_deadline = time_limit > 0.0;
    const auto portfolio_start = std::chrono::steady_clock::now();
    // Saturated before the integer-tick cast, as in make_budget: a huge limit
    // would overflow into a past deadline, starving every worker into an empty
    // NoBudget result.
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

    // The callback's only route to the workers; unused without a callback.
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
                         seed};

    std::vector<SearchResult> results(n_workers);
    // Null unless that worker threw and left no result (or its end-of-run copy
    // or submit threw); decides the all-failed rethrow. Sized up front so each
    // worker writes only its own index. It may hold a python_error and dies on
    // this GIL-released thread, safe only because python_error takes the GIL in
    // its destructor (PySolveCallback in python/bindings.cpp).
    std::vector<std::exception_ptr> failures(n_workers);
    // Which workers produced a result (#170).
    std::vector<char> produced(n_workers, 0);
    // The exception that ended each worker, also after earlier results; what
    // the worker accounting reports (#170). A throw outside a solve attempt
    // reaches run_one's catch with `produced_result` false, its counters lost;
    // it still counts as failed.
    std::vector<std::exception_ptr> ended_by(n_workers);
    // Workers the deadline starved (WorkerEnd::starved); reported, not rethrown.
    std::vector<char> starved(n_workers, 0);

    // One worker, for both launch paths. Catches everything: an escaping
    // exception is std::terminate on a thread and anything at all on an
    // executor, so parking it keeps failure handling the same either way.
    auto run_one = [&ctx, &failures, &results, &pool, &produced, &ended_by, &starved](int i) {
        try {
            WorkerEnd end;
            auto r = run_worker(ctx, i, end);
            ended_by[i] = end.failure;
            starved[i] = end.starved ? 1 : 0;
            if (!r.has_value()) {
                // Handed no budget, stopped before its first attempt (failure
                // null: nothing to rethrow), or every attempt threw.
                failures[i] = end.failure;
                return;
            }
            results[i] = *r;
            produced[i] = 1;

            // Incumbents were shared as found; this puts an infeasible run's
            // closest approach in the pool, so the aggregate has a result.
            Solution sol;
            sol.state = r->best_state;
            sol.objective = r->objective;
            sol.feasible = r->feasible;
            sol.violation = r->best_violation;
            pool.submit(sol);
        } catch (...) {
            // Parked; whether to rethrow is decided once every worker reported.
            failures[i] = std::current_exception();
            ended_by[i] = failures[i];
        }
    };

    launch_workers(n_workers, par_config.executor, stop, run_one);

    // Rethrow when every worker failed (from `failures`, not the pool, which
    // may hold a dead worker's mid-run incumbent) or nothing was submitted: a
    // default SearchResult would read "searched, found nothing feasible". The
    // lowest-index failure is rethrown; homogeneous workers fail alike.
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
        // No failure: every worker was starved (a tiny positive limit expired
        // during launch) or cancelled first, so nothing ran. Report it as
        // empty_portfolio_reason says, not as a feasible-looking default.
        SearchResult empty;
        empty.termination = empty_portfolio_reason(combined.requested());
        report_workers(produced, ended_by, starved, empty);
        return empty;
    }

    SearchResult result;
    result.objective = best->objective;
    result.feasible = best->feasible;
    result.best_state = best->state;
    // The returned state's residual, not +inf: callers may gate on it.
    result.best_violation = best->violation;
    // Work done (iterations, kicks, repairs, counters) sums over workers; time is
    // the max.
    int64_t total_iters = 0;
    double max_time = 0.0;
    for (const auto& r : results) {
        total_iters += r.iterations;
        max_time = std::max(max_time, r.time_seconds);
        result.perturbations += r.perturbations;
        result.lns_repairs += r.lns_repairs;
        result.lns_repairs_accepted += r.lns_repairs_accepted;
        result.counters.merge(r.counters);
        // The earliest worker's pair, both values from that worker.
        if (!std::isnan(r.time_to_first_feasible) &&
            (std::isnan(result.time_to_first_feasible) ||
             r.time_to_first_feasible < result.time_to_first_feasible)) {
            result.time_to_first_feasible = r.time_to_first_feasible;
            result.first_feasible_objective = r.first_feasible_objective;
        }
    }
    result.iterations = total_iters;
    result.time_seconds = max_time;
    result.termination = with_host_cancel(aggregate_termination(results), combined.requested());
    report_workers(produced, ended_by, starved, result);
    return result;
}

}  // namespace cbls
