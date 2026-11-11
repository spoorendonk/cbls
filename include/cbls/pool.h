#pragma once

#include "executor.h"
#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "search.h"
#include "solution_pool.h"
#include "stop.h"
#include "tracer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>

namespace cbls {

struct ParallelConfig {
    int n_threads = 0;  // 0 = hardware_concurrency()
    /// Solutions kept in the shared pool; <= 0 = auto, `max(10, 2 * workers)`
    /// over the workers that run (an `executor` cap narrows it). Below the
    /// worker count the pool cannot hold one point per worker, since `submit`
    /// has no per-worker quota.
    int pool_capacity = 0;

    /// Share the portfolio's best objective as every worker's `obj <= bound`
    /// at each batch boundary (#179; SearchCoordination::share_objective_bound).
    /// False is the A/B control arm.
    bool share_objective_bound = true;

    /// The host's cancellation channel for the whole portfolio (#169), OR-ed
    /// with `SearchConfig::stop`: either ends every worker at its next batch
    /// boundary with TerminationReason::Cancelled. Non-owning (see StopRef);
    /// polled from every worker thread, so the source must be safe to read
    /// concurrently (StopToken is).
    StopRef stop;

    /// Builds one Tracer per worker, given its index in `[0, n_workers)` (#169);
    /// null = no tracing. A null return means that worker is untraced; it never
    /// falls back to `SearchConfig::tracer`. A `SearchConfig::tracer` without a
    /// factory throws std::invalid_argument on the calling thread when more than
    /// one worker would run, since one tracer cannot serve N threads.
    /// Called once per worker, on its thread, before its first solve (a tracer
    /// spans the worker's restarts); called from N threads at once, so shared
    /// state in the factory needs its own lock. `unique_ptr` because a Tracer
    /// is C++-only.
    std::function<std::unique_ptr<Tracer>(int worker)> tracer_factory;

    /// The caller's thread pool, or unset for std::thread workers (#169).
    /// Worker `i` runs as index `i` of `parallel_for_chunked(0, n_workers, ...)`
    /// with `n_workers = min(n_threads, executor->n_threads())`: a worker holds
    /// its chunk for the whole deadline. The executor must run chunks
    /// concurrently; a sequential one is not detected and degenerates to one
    /// worker under a wall clock, or N serial solves without one (see
    /// ExecutorRef). Non-owning. C++-only: a Python executor would serialise on
    /// the GIL.
    std::optional<ExecutorRef> executor;
};

/// The pool capacity a portfolio uses: `requested` when positive, otherwise
/// `max(10, 2 * n_threads)` over the workers that run. See
/// `ParallelConfig::pool_capacity`. Exposed for testing.
[[nodiscard]] int effective_pool_capacity(int requested, int n_threads);

/// The RNG seed for one portfolio worker's one run: a splitmix64 finalizer over
/// (base_seed, worker, restart), so adjacent base seeds give unrelated
/// portfolios. Exposed for testing.
[[nodiscard]] uint64_t portfolio_worker_seed(uint64_t base_seed, int worker, int restart);

/// A cooperative portfolio: N workers, each searching its own Model on its own
/// thread, sharing solutions (and the best objective) through one mutex-guarded
/// SolutionPool, never search state. A worker submits every new incumbent at
/// once; follows the portfolio's best objective as its bound (#179); restarts
/// from the pool's better half when stalled (`get_restart_point`); and is
/// restarted on the remaining clock when its solve() returns early. A
/// pure-feasibility model's first feasible solution stops every worker. A
/// throwing solve() is retried up to `kMaxWorkerRetries` consecutive times
/// (src/pool.cpp); the model factory is called once per worker and never
/// retried.
class ParallelSearch {
public:
    explicit ParallelSearch(int n_threads = 0);

    // Every overload throws if every worker threw and none produced a result;
    // a partial failure returns the survivors' best and reports the dead workers
    // in `SearchResult::workers_completed` / `worker_failures` (#170).
    SearchResult solve(std::function<Model()> model_factory, double time_limit = 10.0,
                       uint64_t seed = 42);

    // Portfolio over one master model whose immutable structure every worker
    // shares (#157); implemented as a factory returning a copy of the frozen
    // master. Sharing follows from the model, not the overload: a factory
    // returning copies of a frozen model shares too; one returning an open or
    // fresh model deep-copies the DAG per worker.
    //
    // `master` is freeze()d on the calling thread before any worker exists
    // (see Model::freeze), so it comes back frozen with the `obj <= bound` row
    // and its assignment unchanged; this overload can therefore throw before
    // any worker runs. One master, one solve at a time; do not write the master
    // at all (values, node values, objective bound) while a solve runs: workers
    // copy it at arbitrary points and nothing detects the race.
    //
    // Each replica shares the structure and starts bit-for-bit from the
    // master's assignment. A one-worker portfolio is still not a single solve():
    // it seeds with `portfolio_worker_seed(seed, 0, 0)`, has a non-null
    // SearchCoordination and may restart with `skip_init`. A
    // `lambda_sum`/`pair_lambda_sum` callable is shared by all workers, so
    // mutable state in it is a data race.
    SearchResult solve(Model& master, double time_limit = 10.0, uint64_t seed = 42);
    SearchResult solve(Model& master, double time_limit, uint64_t seed, const SearchConfig& config,
                       std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
                       std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
                       const ParallelConfig& par_config);

    // Full-featured solve with hooks, LNS, config, and parallel config.
    //
    // Each factory is called once per worker (not per restart) and must return
    // a distinct object each call: the search never locks a hook or an LNS.
    // `shared_ptr` because nanobind cannot relinquish ownership of an in-place
    // instance to a `unique_ptr` (#129). A Python hook_factory receives a copy
    // of the worker's Model (deep for an open model, #157); a C++ one gets the
    // worker's own.
    //
    // `callback` sees the portfolio's stream through one mutex-guarded
    // wrapper, invoked under that mutex: it needs no lock of its own, a slow
    // callback throttles every worker, and it must not call back into the
    // portfolio. See SolveProgress for which fields are the portfolio's.
    SearchResult solve(std::function<Model()> model_factory, double time_limit, uint64_t seed,
                       const SearchConfig& config,
                       std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
                       std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
                       const ParallelConfig& par_config);

private:
    int n_threads_;

    [[nodiscard]] int effective_threads(const ParallelConfig& pc) const;

    static SearchResult solve_portfolio(
        std::function<Model()>& model_factory, double time_limit, uint64_t seed,
        const SearchConfig& config,
        std::function<std::shared_ptr<InnerSolverHook>(Model&)>& hook_factory,
        std::function<std::shared_ptr<LNS>()>& lns_factory, SolveCallback* callback, int n_threads,
        const ParallelConfig& par_config);
};

}  // namespace cbls
