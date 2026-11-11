#pragma once

#include "counters.h"

#include <cstdint>

namespace cbls {

/// Which diversification kick `Tracer::kick` is reporting.
enum class KickKind : std::uint8_t {
    /// A random perturbation of the assignment (`diversify`, the cheap half).
    Perturb,
    /// An LNS destroy-repair cycle was drawn; `Tracer::lns` reports its outcome
    /// immediately after.
    LNS,
    /// Restart from a peer's pool solution instead of perturbing. Portfolio
    /// only: `adopt_from_pool` returns false at once without a pool.
    Adopt,
};

/// Stable snake_case token for a `KickKind` ("perturb", "lns", "adopt").
/// Returns a static string; never null.
const char* kick_kind_name(KickKind kind);

/// An unthrottled event sink, delivered on the search's own thread with no
/// rewriting; distinct from `SolveCallback`, the throttled (~1/s) progress
/// stream on the portfolio's clock and incumbent. Under a portfolio each
/// worker gets its own `Tracer` (`ParallelConfig::tracer_factory`, by worker
/// index), so hosts aggregate themselves; routing through the progress
/// wrapper would serialise workers on one mutex per batch.
///
/// Granularity is the batch, never the iteration (a per-iteration virtual
/// call would sit on the GLS hot loop): each method fires at most once per
/// batch, kick or inner-solver call. Default `SearchConfig::tracer` is null:
/// one predictable null compare per site, nothing else. Methods are no-ops;
/// override only what you want.
///
/// Attaching a tracer adds clock reads (one per new best, two per
/// inner-solver call), even on an iteration-budgeted run (`time_limit <= 0`,
/// no callback) whose loop otherwise reads none (`docs/architecture.md`); the
/// trajectory is unchanged, but iteration throughput pays for them.
///
/// A throwing method propagates out of `solve()` like a raising
/// `SolveCallback::on_progress`, leaving the model at the reached assignment;
/// under `ParallelSearch` it is absorbed per worker.
class Tracer {
public:
    Tracer() = default;
    Tracer(const Tracer&) = default;
    Tracer& operator=(const Tracer&) = default;
    Tracer(Tracer&&) = default;
    Tracer& operator=(Tracer&&) = default;
    /// Out-of-line: the key function, so the vtable is emitted once.
    virtual ~Tracer();

    /// One batch finished. `iterations` is the run's CUMULATIVE GLS count
    /// after it (as `SearchResult::iterations`); difference consecutive events
    /// for a batch's cost. Structural batches charge none, so it repeats; a
    /// Novelty batch charges 3 per move and 1 per weight bump (#209).
    /// `improved`: the batch produced a new best.
    virtual void batch_end(BatchKind kind, int64_t iterations, bool improved);

    /// A new best real-feasible solution, with seconds since this `solve()`
    /// started. Arrives before the `batch_end` of the batch that produced it,
    /// which is what lets a consumer attribute it; fires before the inner
    /// polish and again after it if the polish improved. A pool adoption
    /// (`adopt_from_pool`) that installs a better incumbent is NOT reported
    /// here, since it happens after `batch_end`; it is `kick(KickKind::Adopt)`.
    /// So this stream is THIS worker's own improvements; the portfolio's
    /// incumbent is what `SolveCallback` reports.
    virtual void new_best(double objective, double seconds);

    /// A diversification kick was taken. `KickKind::LNS` is followed by exactly
    /// one `lns()` event carrying its outcome.
    virtual void kick(KickKind kind);

    /// An LNS destroy-repair cycle finished; `accepted` is what
    /// `LNS::destroy_repair` returned (see
    /// `SearchResult::lns_repairs_accepted`).
    virtual void lns(bool accepted);

    /// An `InnerSolverHook::solve` call finished, and how long it took.
    virtual void hook(double seconds);
};

}  // namespace cbls
