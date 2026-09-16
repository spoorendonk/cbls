#pragma once

#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "search.h"
#include "solution_pool.h"

#include <cstdint>
#include <functional>
#include <memory>

namespace cbls {

struct ParallelConfig {
    int n_threads = 0;  // 0 = hardware_concurrency()
    /// Solutions kept in the shared pool. 0 (or any non-positive value) = auto,
    /// which is `max(10, 2 * n_threads)`.
    ///
    /// Auto rather than a fixed 10 because a pool smaller than the worker count
    /// cannot represent the portfolio at all: `submit` sorts globally by
    /// objective and applies no per-worker quota, so at 32 workers a capacity of
    /// 10 holds the ten best objectives ever submitted and nothing else. That
    /// does NOT make the pool diverse -- see `get_restart_point` -- it only
    /// stops the capacity itself from being the binding constraint.
    int pool_capacity = 0;
};

/// The pool capacity a portfolio actually uses: `requested` when positive,
/// otherwise auto -- `max(10, 2 * n_threads)`. See
/// `ParallelConfig::pool_capacity`. Exposed for the same reason
/// `portfolio_worker_seed` is: the rule is worth testing directly rather than
/// inferring from two search trajectories.
[[nodiscard]] int effective_pool_capacity(int requested, int n_threads);

/// The RNG seed for one portfolio worker's one run.
///
/// Mixed, not added. `base + worker + restart * n_threads` is a correct
/// bijection WITHIN a run, and that is all the old scheme claimed -- but across
/// runs it made adjacent seeds nearly the same portfolio: at 12 workers
/// `--seed 42` and `--seed 43` share 11 of their 12 base streams, so bumping the
/// seed, which is the standard way to draw an independent sample, barely changed
/// anything. Now the default worker count rather than an opt-in mode.
///
/// This is a splitmix64 finalizer over the triple, which decorrelates all three.
/// Exposed rather than kept static so the property can be tested directly
/// instead of inferred from two search trajectories.
[[nodiscard]] uint64_t portfolio_worker_seed(uint64_t base_seed, int worker, int restart);

/// A COOPERATIVE portfolio: N workers, each owning its own `Model` and searching
/// it on its own thread, sharing incumbents through one mutex-guarded
/// `SolutionPool`. The search itself is still single-threaded per solve; what is
/// parallel is the portfolio, and what is shared is solutions, never state.
///
/// Three properties distinguish it from a set of independent runs:
///
///  - a worker SUBMITS every new incumbent the moment it records one, not once
///    at the end;
///  - a stalled worker RESTARTS from the pool (`get_restart_point`, i.e. the
///    better half) instead of only perturbing its own assignment;
///  - no worker idles while budget remains. A worker whose `solve()` returns
///    early -- an exhausted iteration budget -- is restarted on the remaining
///    clock, and the one case where finishing early is correct (a
///    pure-feasibility model, whose first feasible solution is the answer)
///    stops every OTHER worker too rather than leaving them to run the clock
///    out on a settled question.
///
/// The last point covers a worker whose `solve()` THROWS too: it is retried up
/// to `kMaxWorkerRetries` consecutive times (see src/pool.cpp) rather than dying
/// and idling its core. A search that fails that many times in a row is failing
/// deterministically, and retrying it only burns the shared budget. The model
/// factory is deliberately NOT retried -- it is called once per worker, above
/// the restart loop, and a factory that cannot build its model will not build it
/// on the second ask either.
class ParallelSearch {
public:
    explicit ParallelSearch(int n_threads = 0);

    // Both overloads THROW if *every* worker threw AND none of them produced a
    // result -- a model factory that cannot build its model, say. Returning a
    // default SearchResult there would report "searched, found nothing
    // feasible" about a run that never searched. A partial failure is absorbed:
    // the survivors' best is returned and each dead worker contributes a
    // default SearchResult to the aggregate.
    //
    // The "and none of them produced a result" half is not pedantry: a worker
    // can now share incumbents, exhaust its retries on a later restart, and
    // still have a result worth keeping. Such a run returns normally and its
    // exceptions are not reported.
    //
    // Simple portfolio solve (backward-compatible)
    SearchResult solve(std::function<Model()> model_factory, double time_limit = 10.0,
                       uint64_t seed = 42);

    // Full-featured solve with hooks, LNS, config, and parallel config.
    //
    // Both factories hand back a SHARED pointer rather than a raw owning one.
    // The C++ side keeps each object alive for exactly one worker, so a
    // unique_ptr would say what it means -- but nanobind cannot relinquish an
    // in-place instance's ownership (nb_type_relinquish_ownership refuses
    // `state.internal`), so a Python factory could only satisfy a unique_ptr
    // signature through a nanobind-specific deleter, i.e. binding glue in this
    // header. shared_ptr costs one control block per worker and lets
    // nanobind's own caster keep the Python object alive and drop it under the
    // GIL. See issue #129.
    //
    // The cost of that choice: nothing now stops a factory from handing back
    // the SAME object every call. Don't. Each worker searches its own model on
    // its own thread and the search never locks the hook or the LNS, so a
    // shared stateful instance is a data race -- the failure the raw-pointer
    // signature used to catch loudly as a double free.
    //
    // One more asymmetry for a Python hook_factory: the Model& reaches it as a
    // COPY (nanobind demotes an lvalue reference to rv_policy::copy), so the
    // callee cannot see the model it will run against, and a large model is
    // deep-copied once per worker. Avoiding that would need a hand-written
    // binding wrapper, i.e. the glue this signature exists to avoid.
    //
    // Each factory is called ONCE per worker, not once per restart: a worker
    // that restarts keeps its model, hook and LNS and carries on with them.
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
        int pool_capacity);
};

}  // namespace cbls
