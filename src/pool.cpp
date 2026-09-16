#include "cbls/pool.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <memory>
#include <optional>
#include <thread>

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
                           effective_pool_capacity(pc.pool_capacity, n));
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
                           callback, n, effective_pool_capacity(par_config.pool_capacity, n));
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
// budget exit. Among budget exits the shared wall clock outranks the per-worker
// iteration budget, because the portfolio's answer is clock-limited as soon as
// any worker ran the clock out. NoBudget is last: it is also what a worker that
// threw leaves behind, and one crashed thread should not relabel a run the
// others budget-limited.
static TerminationReason aggregate_termination(const std::vector<SearchResult>& results) {
    bool any_stopped = false;
    bool any_time = false;
    bool any_iterations = false;
    for (const auto& r : results) {
        if (r.termination == TerminationReason::Feasible) {
            return TerminationReason::Feasible;
        }
        any_stopped = any_stopped || r.termination == TerminationReason::Stopped;
        any_time = any_time || r.termination == TerminationReason::TimeLimit;
        any_iterations = any_iterations || r.termination == TerminationReason::IterationLimit;
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

    void absorb(const SearchResult& r) {
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
        const bool better = !any_run || (r.feasible && !result.feasible) ||
                            (r.feasible == result.feasible && r.objective < result.objective);
        if (better) {
            result.objective = r.objective;
            result.feasible = r.feasible;
            result.best_state = r.best_state;
        }
        any_run = true;
    }
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
    SolveCallback* callback;
    SearchCoordination& coord;
    // Seconds left on the SHARED deadline -- one clock for the whole portfolio,
    // so a worker that restarts gets the time its predecessor left rather than
    // a fresh full budget. Returns 0.0 when there is no wall clock at all, in
    // which case `has_deadline` is false and the value is not a budget.
    std::function<double()> remaining;
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

    // Only thread 0 gets the callback, to avoid interleaved output.
    SolveCallback* cb = (index == 0) ? ctx.callback : nullptr;

    WorkerAccumulator acc;
    int consecutive_failures = 0;
    // No idle threads while budget remains. A single solve() can return with
    // time left -- an exhausted SearchConfig::max_iterations is the case that
    // exists today -- and the core it was using would then sit out the rest of
    // the run. Restart it instead, on the time its predecessor left.
    for (int restart = 0;; ++restart) {
        if (ctx.coord.stop->load(std::memory_order_relaxed)) {
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
        SearchConfig cfg = ctx.config;
        if (restart > 0) {
            // Keep the assignment this worker already holds -- its own
            // incumbent, or a peer's if it adopted one -- instead of throwing
            // the run away and starting from the closest-to-zero point again.
            cfg.skip_init = true;
        }
        // Distinct per (worker, restart), and decorrelated ACROSS base seeds --
        // see portfolio_worker_seed.
        const uint64_t run_seed = portfolio_worker_seed(ctx.seed, index, restart);

        SearchResult r;
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
        acc.absorb(r);

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

        if (r.termination == TerminationReason::Stopped) {
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
    int pool_capacity) {
    SolutionPool pool(pool_capacity);
    std::atomic<bool> stop{false};
    SearchCoordination coord{&pool, &stop};

    const bool has_deadline = time_limit > 0.0;
    // Saturated before the integer-tick cast, exactly as search.cpp's
    // make_budget does and for the same reason: callers pass a very large limit
    // to mean "effectively unbounded", and casting 1e12 seconds to nanoseconds
    // overflows int64 and yields a deadline already in the past. Here that
    // would make every worker break before its first solve and hand the caller
    // a NoBudget result with an empty state -- where make_budget's own
    // saturation would have run the search.
    constexpr double kMaxPortfolioSeconds = 1.0e9;  // ~31 years
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(
                                  std::min(std::max(0.0, time_limit), kMaxPortfolioSeconds)));
    auto remaining = [deadline, has_deadline]() -> double {
        if (!has_deadline) {
            return 0.0;  // no wall clock: a worker's own iteration budget bounds it
        }
        return std::max(
            0.0,
            std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count());
    };

    PortfolioContext ctx{model_factory, hook_factory, lns_factory,  config, callback,
                         coord,         remaining,    has_deadline, seed,   n_threads};

    std::vector<SearchResult> results(n_threads);
    // One slot per worker, left null unless that worker threw. Sized up front so
    // the lambdas below only ever write their own index.
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
        SearchResult empty;
        empty.termination = TerminationReason::NoBudget;
        return empty;
    }

    SearchResult result;
    result.objective = best->objective;
    result.feasible = best->feasible;
    result.best_state = best->state;
    // Sum iterations and take max time across threads
    int64_t total_iters = 0;
    double max_time = 0.0;
    for (const auto& r : results) {
        total_iters += r.iterations;
        max_time = std::max(max_time, r.time_seconds);
    }
    result.iterations = total_iters;
    result.time_seconds = max_time;
    result.termination = aggregate_termination(results);
    return result;
}

}  // namespace cbls
