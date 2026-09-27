#pragma once

#include "model.h"
#include "rng.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace cbls {

struct Solution {
    Model::State state;
    double objective = std::numeric_limits<double>::infinity();
    bool feasible = false;
    /// Largest violation over the REAL constraints at `state` -- the artificial
    /// `obj <= bound` row excluded, exactly as `SearchResult::best_violation`
    /// defines it. Carried with the state rather than recomputed by the reader:
    /// a `ParallelSearch` assembles its result from the pool, on a thread that
    /// owns no `Model` to evaluate it against, and a result whose violation did
    /// not describe its own state reported +inf for every portfolio run.
    double violation = std::numeric_limits<double>::infinity();
    /// Portfolio worker index that submitted this point, or -1 (#161).
    int submitter = -1;
};

/// What the asking worker tells a restart draw (#161, A/B measurement only).
struct RestartRequest {
    /// The asking worker's portfolio index, or -1 when unknown.
    int worker = -1;
    /// Distance from the asker's LIVE assignment to a candidate state. Empty
    /// when the caller cannot supply one.
    std::function<double(const Model::State&)> distance;
};

/// TEMPORARY (#161): the restart rule under measurement, read once from the
/// environment variable CBLS_ISSUE161_ARM. Removed before landing.
enum class RestartRule : uint8_t { Control, ReservedSlot, Tabu, Distance };

/// A bounded, sorted store of the best solutions seen, shared across the workers
/// of a `ParallelSearch`. Every method takes the mutex, so it is the one object
/// the workers touch concurrently; everything else about a worker is private to
/// its own thread.
class SolutionPool {
public:
    explicit SolutionPool(int capacity = 10);
    ~SolutionPool();  // TEMPORARY (#161): prints draw counts when the arm is set
    SolutionPool(const SolutionPool&) = delete;
    SolutionPool& operator=(const SolutionPool&) = delete;
    SolutionPool(SolutionPool&&) = delete;
    SolutionPool& operator=(SolutionPool&&) = delete;

    /// By value, then moved into the store: the caller's copy is made outside
    /// the lock, so the critical section never copies a `Model::State`. That is
    /// the one path where N workers actually contend.
    bool submit(Solution sol);
    std::optional<Solution> best() const;
    std::vector<Solution> top_k(int k) const;
    /// Uniformly over the BETTER HALF of the pool -- not the single best. A
    /// stalled worker restarts from whatever this returns.
    ///
    /// This REDUCES the pull toward one basin; it does not prevent it, and the
    /// difference matters. `submit` applies no diversity criterion and no
    /// per-worker quota, so on an objective model the pool converges to the ten
    /// globally best objectives ever submitted -- which is typically the tail of
    /// one worker's monotone improving trajectory, i.e. ten refinements of a
    /// single point. Drawing from the better half of that is close to drawing
    /// the best. At the default worker count the capacity cannot even hold one
    /// entry per worker -- which `ParallelConfig::pool_capacity`'s auto mode now
    /// fixes, by scaling the capacity with the worker count.
    ///
    /// Capacity was the only part fixed, deliberately. The structural answer is
    /// a per-worker reserved slot, so each worker's own best is always drawable
    /// whatever the global ranking; that needs a submitter identity on
    /// `Solution` and a draw that knows which worker is asking, and it changes
    /// search behaviour in a way only a quality measurement could justify.
    /// Issue #135 scopes measurement out, so the slot is declined there rather
    /// than guessed at here. Raising the capacity needs no such justification:
    /// it only stops the capacity itself from being the binding constraint.
    std::optional<Solution> get_restart_point(RNG& rng, const RestartRequest& request = {});
    size_t size() const;

private:
    struct TabuKey {
        double objective;
        uint64_t hash;
    };
    int capacity_;
    RestartRule rule_ = RestartRule::Control;
    std::vector<Solution> solutions_;
    /// Per worker: that worker's best submission, kept whatever its global rank.
    std::vector<std::optional<Solution>> reserved_;
    /// Per worker: the start points that worker has already been handed.
    std::vector<std::vector<TabuKey>> tabu_;
    int64_t draws_ = 0;
    int64_t declines_ = 0;
    bool report_ = false;
    mutable std::mutex mutex_;
};

/// Cross-worker coordination, handed to `cbls::solve()` by `ParallelSearch`.
///
/// A null `SearchCoordination*` -- the default at every call site outside
/// `ParallelSearch` -- leaves the search bit-identical to the run without it.
/// That is the invariant every benchmark rests on (all four runners call
/// `cbls::solve()` directly), and `tests/test_parallel.cpp` pins it.
struct SearchCoordination {
    /// Shared incumbents: written on every new best, read on stagnation.
    SolutionPool* pool = nullptr;
    /// Global cancellation. Set when one worker has finished the job outright
    /// -- a pure-feasibility model whose first feasible solution IS the answer
    /// -- so the rest stop within a batch instead of running the clock out on a
    /// question already answered.
    std::atomic<bool>* stop = nullptr;
    /// The worker this coordination object belongs to, or -1 (#161).
    int worker = -1;
};

}  // namespace cbls
