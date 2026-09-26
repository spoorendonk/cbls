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
    /// Solutions kept in the shared pool. 0 (or any non-positive value) = auto,
    /// which is `max(10, 2 * workers)` -- the workers that actually RUN, so an
    /// `executor` narrower than `n_threads` narrows this too.
    ///
    /// Auto rather than a fixed 10 because a pool smaller than the worker count
    /// cannot represent the portfolio at all: `submit` sorts globally by
    /// objective and applies no per-worker quota, so at 32 workers a capacity of
    /// 10 holds the ten best objectives ever submitted and nothing else. That
    /// does NOT make the pool diverse -- see `get_restart_point` -- it only
    /// stops the capacity itself from being the binding constraint.
    int pool_capacity = 0;

    /// The HOST's cancellation channel for the whole portfolio (#169).
    ///
    /// OR-ed with `SearchConfig::stop`, not a replacement for it: a caller may
    /// set either or both, and a raised stop on either ends every worker at its
    /// next batch boundary with `TerminationReason::Cancelled`. The combined
    /// source lives on `solve_portfolio`'s stack for the duration of the call,
    /// so nothing here extends any lifetime -- the objects BOTH refs name must
    /// outlive the solve, exactly as `StopRef` says.
    ///
    /// It is polled from every worker thread and from the restart loop, so the
    /// source has to be safe to read concurrently. `cbls::StopToken` is.
    StopRef stop;

    /// Builds ONE `Tracer` per worker, given that worker's index in
    /// `[0, n_workers)` -- which is `n_threads` unless an `executor` below caps it
    /// (#169). Null -- the default -- means no tracing at all.
    ///
    /// A factory, once set, DECIDES: a call that returns null means this worker is
    /// not traced, and does NOT fall back to `SearchConfig::tracer`. Falling back
    /// would hand the declining workers one shared sink, which is the race this
    /// field exists to remove.
    ///
    /// And a `SearchConfig::tracer` with NO factory here is REFUSED --
    /// `std::invalid_argument`, thrown on the calling thread before any worker
    /// exists -- whenever more than one worker would run. One tracer cannot serve N
    /// worker threads, and neither silent repair is defensible: dropping it loses
    /// events the caller asked for, keeping it is the race. A one-worker portfolio
    /// is allowed through, since there is no peer to race with.
    ///
    /// Per worker rather than one shared instance, and deliberately: a `Tracer`'s
    /// events arrive per batch on the reporting worker's own thread, and routing
    /// N of them through one object would either need a lock inside the host's
    /// tracer or serialise the portfolio on one -- which is exactly what
    /// `SolveCallback` pays for an ordered progress stream, and why a tracer is
    /// NOT that. A host wanting a portfolio-wide view keeps the tracers it built
    /// and aggregates across them after the solve; the worker index is handed
    /// over so it can tell them apart.
    ///
    /// `unique_ptr`, unlike the hook and LNS factories' `shared_ptr`: those are
    /// shaped by what nanobind can do with an in-place instance, and a `Tracer`
    /// is C++-only (see the note on `SearchConfig::tracer`), so the honest
    /// ownership is the one written here.
    ///
    /// Called ONCE per worker, on that worker's thread, before its first solve --
    /// not once per restart. A worker that restarts keeps its tracer, so a
    /// tracer's counts span its worker's restarts the way
    /// `SearchCounters::merge` makes the result's counters span them. The factory
    /// itself is called from N threads at once, so a factory that touches shared
    /// state of its own needs its own lock.
    std::function<std::unique_ptr<Tracer>(int worker)> tracer_factory;

    /// The CALLER's thread pool, or unset for today's `std::thread` workers
    /// (#169).
    ///
    /// With it set, `ParallelSearch` creates no thread of its own: worker `i`
    /// runs as index `i` of `parallel_for_chunked(0, n_workers, ...)`, with
    /// `n_workers = min(n_threads, executor->n_threads())`. The cap is not
    /// cosmetic -- a portfolio worker holds its chunk for the whole shared
    /// deadline, so a worker queued behind another would get no budget at all.
    ///
    /// THE EXECUTOR MUST RUN CHUNKS CONCURRENTLY. Workers are cooperative: they
    /// share incumbents through the pool as they find them and restart from a
    /// peer's, which is not a portfolio if they run one after another. A
    /// SEQUENTIAL executor is not rejected -- under a WALL CLOCK it degenerates to
    /// a one-worker portfolio, since the first worker takes the whole deadline, and
    /// with `time_limit <= 0` there is no deadline to take, so it runs every
    /// worker's full iteration budget in series instead: N solves back to back, N
    /// times the wall time. Nothing detects either. See `ExecutorRef`.
    ///
    /// NON-OWNING, like `stop`: the pool must outlive the solve.
    ///
    /// C++-only, and deliberately not bound to Python: a Python "executor" would
    /// have to be called back into from the engine under the GIL, which is the
    /// serialisation a portfolio exists to avoid.
    std::optional<ExecutorRef> executor;
};

/// The pool capacity a portfolio actually uses: `requested` when positive,
/// otherwise auto -- `max(10, 2 * n_threads)`, where `solve_portfolio` passes the
/// workers that actually run rather than the number requested. See
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

    // Portfolio over ONE master model, whose immutable structure every worker
    // SHARES (#157). The convenient overload, and the one to reach for when the
    // caller has nothing else to arrange.
    //
    // What decides whether the DAG is shared is the model, NOT which overload:
    // this one is implemented as exactly a factory returning a copy of a frozen
    // master. So a `std::function<Model()>` that returns a copy of a FROZEN model
    // shares the structure too -- `benchmarks/mipfeas/` does that deliberately, to
    // keep the per-worker copies in its own setup phase rather than in the search's
    // deadline. A factory returning a copy of an OPEN model, or building a fresh
    // one per call, deep-copies the DAG per worker; that is what cost ~0.75-0.86
    // GiB per worker on the largest MIPfeas instance, against ~0.34-0.39 GiB
    // shared (`docs/architecture.md` carries the table and its commit).
    // `freeze()` is the operative call.
    //
    // `master` is `freeze()`d here, BEFORE any thread exists: `Model::freeze`
    // explains why the freeze point is after the objective row rather than at
    // `close()`, and doing it on the calling thread is what keeps N workers from
    // re-sorting one shared DAG. So the master comes back frozen, carrying the
    // artificial `obj <= bound` row and refusing further structural changes,
    // while its ASSIGNMENT is left exactly as it was -- no worker searches the
    // master itself. (Its node-value array grows by that row's two entries, and
    // an open or unevaluated master is evaluated by the `close()`/objective-row
    // steps `freeze()` folds in.)
    //
    // ONE master, ONE solve at a time. `freeze()` writes the structure, so handing
    // the same master to two concurrent `solve` calls races. Reusing it for a
    // second solve after the first returns is fine -- `freeze()` is then a no-op
    // -- but the replicas start from whatever assignment the master holds then.
    //
    // And do not write the master AT ALL while a solve is running, structurally or
    // not. Workers copy it on their own threads at arbitrary points after the solve
    // starts, so a `SolveCallback` or `hook_factory` that captured the master and
    // set `var_mut().value`, wrote a node value or called `set_objective_bound`
    // races with whichever worker is mid-copy. A structural write throws; these do
    // not, and nothing detects them.
    //
    // Each worker gets a copy: the structure by reference, its own variables,
    // node values and objective bound. A replica therefore starts from the
    // master's assignment rather than from a re-evaluated one -- bit-for-bit the
    // state the master was in, which is what makes replication a memcpy with no
    // `full_evaluate`. That is NOT the same as saying a one-worker portfolio
    // reproduces a single `solve()`: a worker seeds with
    // `portfolio_worker_seed(seed, 0, 0)` rather than `seed`, runs with a non-null
    // `SearchCoordination`, and may restart with `skip_init`. The portfolio is a
    // different run at any thread count.
    //
    // One thing sharing changes: a `lambda_sum`/`pair_lambda_sum` callable is now
    // invoked by several workers at once, so a callable carrying mutable state of
    // its own is a data race. That is a NEW hazard for state captured by value --
    // each replica used to hold its own copy of the `std::function` -- and an old
    // one for state captured by reference.
    //
    // Same callback contract as the factory overloads. The throwing contract
    // differs in one way: `freeze()` runs on the CALLING thread, so this overload
    // can throw before any worker exists (a bad_alloc on the objective row, or a
    // topological sort that rejects an unclosed cyclic model), where a factory
    // overload only ever reports the aggregated worker failure.
    SearchResult solve(Model& master, double time_limit = 10.0, uint64_t seed = 42);
    SearchResult solve(Model& master, double time_limit, uint64_t seed, const SearchConfig& config,
                       std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory,
                       std::function<std::shared_ptr<LNS>()> lns_factory, SolveCallback* callback,
                       const ParallelConfig& par_config);

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
    // COPY (nanobind demotes an lvalue reference to rv_policy::copy), so the callee
    // cannot see the model it will run against. What that costs depends on the
    // model: a FROZEN one shares its structure, so the copy is the variables and
    // the node values; an open one is deep-copied whole, once per worker (#157).
    // Avoiding the copy at all would need a hand-written binding wrapper, i.e. the
    // glue this signature exists to avoid. A C++ hook_factory is handed the
    // worker's own `Model&` and copies nothing.
    //
    // Each factory is called ONCE per worker, not once per restart: a worker
    // that restarts keeps its model, hook and LNS and carries on with them.
    //
    // `callback` sees the PORTFOLIO's stream, not worker 0's. Every worker
    // reports through one mutex-guarded wrapper, and the consumer is invoked
    // UNDER that mutex -- three consequences, all deliberate: a consumer
    // writing to one stream or file needs no lock of its own; a consumer that
    // BLOCKS throttles the whole portfolio rather than only its own worker
    // (which is the price of an ordered stream, and why a callback here must
    // not do slow work); and a consumer must never call back into this
    // portfolio. The wrapper rewrites `time_seconds` onto the portfolio clock
    // and `objective`/`new_best` onto the global incumbent, leaving every other
    // field the reporting worker's own -- see SolveProgress in search.h.
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
