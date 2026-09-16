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
    int n_threads = 0;       // 0 = hardware_concurrency()
    int pool_capacity = 10;  // solutions kept in the shared pool
};

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
/// One exception to the last point, and it is not currently handled: a worker
/// that THROWS is gone for the rest of the run, its core idle, and unless every
/// worker threw the exception is not reported either. Restarting it would spin
/// on a factory that throws deterministically, so the fix is not obvious; see
/// issue #135.
class ParallelSearch {
public:
    explicit ParallelSearch(int n_threads = 0);

    // Both overloads THROW if *every* worker threw -- a model factory that
    // cannot build its model, say. Returning a default SearchResult there would
    // report "searched, found nothing feasible" about a run that never
    // searched. A partial failure is absorbed: the survivors' best is returned
    // and each dead worker contributes a default SearchResult to the aggregate.
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
