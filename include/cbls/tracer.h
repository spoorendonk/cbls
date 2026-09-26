#pragma once

#include "counters.h"

#include <cstdint>

namespace cbls {

/// Which diversification kick `Tracer::kick` is reporting.
enum class KickKind : std::uint8_t {
    /// A random perturbation of the assignment (`diversify`, the cheap half).
    Perturb,
    /// An LNS destroy-repair cycle was drawn. `Tracer::lns` reports its outcome
    /// immediately after.
    LNS,
    /// The search restarted from a peer's solution in the shared pool instead of
    /// perturbing its own. Portfolio-only: `adopt_from_pool` returns false at
    /// once without a pool.
    Adopt,
};

/// Stable snake_case token for a `KickKind` ("perturb", "lns", "adopt").
/// Returns a static string; never null.
const char* kick_kind_name(KickKind kind);

/// An event sink a host can wire to its own tracer.
///
/// SEPARATE FROM `SolveCallback` on purpose. That is a THROTTLED PROGRESS
/// STREAM -- one row about every second, rewritten onto the portfolio's clock
/// and the portfolio's incumbent by `ParallelSearch` -- and it must stay one.
/// This is an unthrottled event stream, delivered on the search's own thread,
/// with no rewriting: under a portfolio each worker gets ITS OWN `Tracer` (built
/// by `ParallelConfig::tracer_factory` with the worker's index) and reports its
/// own events, so a host that wants a portfolio-wide view aggregates across the
/// tracers it built. Routing it through the progress wrapper instead would
/// serialise every worker on one mutex for events that arrive per batch.
///
/// GRANULARITY IS THE BATCH, never the iteration. A per-iteration hook would put
/// a virtual call on the GLS inner loop, which runs `batch_iterations` (1000 by
/// default) times per batch on the engine's hottest path. Every method here
/// fires at most once per batch, once per kick, or once per inner-solver call.
///
/// The default is a NULL POINTER (`SearchConfig::tracer`), which costs one
/// predictable null compare per event site and nothing else -- no allocation,
/// no virtual dispatch, no clock read. Every method below is a no-op, so a
/// subclass overrides only the events it wants.
///
/// THE TIMESTAMPS COST A CLOCK READ, and only when a tracer is attached.
/// `docs/architecture.md` guarantees that an iteration-budgeted run
/// (`time_limit <= 0`) with no callback reads no clock at all; attaching a
/// tracer adds one read per new best and two per inner-solver call. Neither
/// reaches the search's control flow, so the trajectory is unchanged either
/// way -- but a host measuring iteration throughput on a clockless run should
/// know it is paying for them.
///
/// A method that THROWS propagates out of `solve()` exactly as a raising
/// `SolveCallback::on_progress` does, leaving the model at the assignment the
/// search had reached. Under `ParallelSearch` it is absorbed per worker, as a
/// raising callback is.
class Tracer {
public:
    Tracer() = default;
    Tracer(const Tracer&) = default;
    Tracer& operator=(const Tracer&) = default;
    Tracer(Tracer&&) = default;
    Tracer& operator=(Tracer&&) = default;
    /// Out-of-line so this class has a key function and its vtable is emitted
    /// once, the way `SolveCallback`'s is.
    virtual ~Tracer();

    /// One batch finished. `iterations` is the run's CUMULATIVE GLS iteration
    /// count as it stands after the batch -- the same quantity
    /// `SearchResult::iterations` reports, so a consumer can difference
    /// consecutive events to get the batch's own cost. Structural and Novelty
    /// batches do not charge the GLS counter, so the value repeats across them.
    /// `improved` is whether the batch produced a new best.
    virtual void batch_end(BatchKind kind, int64_t iterations, bool improved);

    /// A new best real-feasible solution was recorded, with the seconds since
    /// this `solve()` started. Fires BEFORE the inner-solver polish that may
    /// follow in the same batch, and again after it if the polish improved on
    /// it -- the two are different points and both are new bests.
    ///
    /// ONE IMPROVEMENT IS DELIBERATELY NOT REPORTED HERE: a portfolio worker
    /// adopting a peer's solution from the shared pool can install a better
    /// incumbent than its own (`adopt_from_pool`, src/search.cpp), and that path
    /// does not go through `record_best`. It reports `kick(KickKind::Adopt)`
    /// instead. The reason is the ordering rule above: an adoption happens in the
    /// diversification step, AFTER this batch's `batch_end`, so emitting it as a
    /// `new_best` would break "a new best arrives before the batch_end of the
    /// batch that produced it" -- which is the property that lets a consumer
    /// attribute an improvement to a batch at all. A host reconstructing the
    /// incumbent trajectory from this stream is seeing THIS WORKER's own
    /// improvements; the portfolio's incumbent is what `SolveCallback` reports.
    virtual void new_best(double objective, double seconds);

    /// A diversification kick was taken. `KickKind::LNS` is followed by exactly
    /// one `lns()` event carrying its outcome.
    virtual void kick(KickKind kind);

    /// An LNS destroy-repair cycle finished. `accepted` is what
    /// `LNS::destroy_repair` returned; see `SearchResult::lns_repairs_accepted`
    /// for what that does and does not mean.
    virtual void lns(bool accepted);

    /// An `InnerSolverHook::solve` call finished, and how long it took.
    virtual void hook(double seconds);
};

}  // namespace cbls
