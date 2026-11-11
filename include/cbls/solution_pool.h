#pragma once

#include "model.h"
#include "rng.h"

#include <atomic>
#include <cstddef>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace cbls {

struct Solution {
    Model::State state;
    double objective = std::numeric_limits<double>::infinity();
    bool feasible = false;
    /// Largest violation over the REAL constraints (no `obj <= bound` row), as
    /// `SearchResult::best_violation`. Carried, not recomputed:
    /// `ParallelSearch` builds its result on a thread that owns no `Model`.
    double violation = std::numeric_limits<double>::infinity();
};

/// Portfolio ranking: feasible before infeasible; two feasible by lower
/// objective; two infeasible by lower violation, then objective (their
/// objective is +inf, #211). A NaN violation ranks as +inf. Shared by
/// `SolutionPool`'s sort and a worker's across-restarts accumulator.
bool ranks_before(bool feasible_a, double objective_a, double violation_a, bool feasible_b,
                  double objective_b, double violation_b);

/// A bounded, sorted store of the best solutions seen, shared by the workers of
/// a `ParallelSearch`. Every method takes the mutex; it is the only object
/// workers touch concurrently.
class SolutionPool {
public:
    explicit SolutionPool(int capacity = 10);

    /// By value: the caller copies outside the lock and the store moves, so the
    /// critical section (the one contended path) never copies a `Model::State`.
    bool submit(Solution sol);
    std::optional<Solution> best() const;
    std::vector<Solution> top_k(int k) const;
    /// Uniform over the BETTER HALF of the pool; a stalled worker restarts from
    /// it. This reduces the pull toward one basin but does not prevent it:
    /// `submit` has no diversity criterion or per-worker quota, so on an
    /// objective model the pool tends to hold refinements of one point
    /// (`ParallelConfig::pool_capacity` auto scales capacity with workers so
    /// it is not the binding limit). #161 measured a per-worker reserved slot,
    /// a tabu on past start points and a distance-weighted whole-pool draw
    /// against this rule on `cbls_mipfeas`: every 95% CI spanned zero, so the
    /// rule stands (not separable at that power, not proven equal). Untested:
    /// diversity at `submit`, and a tabu by near-identity.
    std::optional<Solution> get_restart_point(RNG& rng) const;
    size_t size() const;

    /// Best FINITE objective of any FEASIBLE submission ever, else +inf (#179):
    /// what a worker tightens its bound towards (`share_objective_bound`).
    /// Monotone non-increasing; not `best()->objective`, which may be an
    /// infeasible closest-approach entry (often better, so unreachable as a
    /// bound) or #100's +inf witness (`objective - eps` undefined).
    ///
    /// Written under the mutex by `submit`, read lock-free with relaxed order:
    /// it guards no data, and a stale read costs one batch under a looser
    /// bound.
    ///
    /// The objective is the submitter's, not re-evaluated on the reader's model
    /// (an adoption is). A factory handing out different models gets a
    /// misdirected bound but no corrupt incumbent: a worker records only points
    /// its own model found feasible.
    [[nodiscard]] double best_feasible_objective() const {
        return best_feasible_objective_.load(std::memory_order_relaxed);
    }

private:
    int capacity_;
    std::vector<Solution> solutions_;
    std::atomic<double> best_feasible_objective_{std::numeric_limits<double>::infinity()};
    mutable std::mutex mutex_;
};

/// Cross-worker coordination, handed to `cbls::solve()` by `ParallelSearch`.
/// A null `SearchCoordination*` (the default everywhere else, including all
/// benchmark runners) leaves the search bit-identical to a run without it;
/// `tests/test_parallel.cpp` pins this.
struct SearchCoordination {
    /// Shared incumbents: written on every new best, read on stagnation.
    SolutionPool* pool = nullptr;
    /// Global cancellation, set when one worker answered a pure-feasibility
    /// model outright, so the rest stop within a batch.
    std::atomic<bool>* stop = nullptr;
    /// Whether a worker tightens its objective bound to the pool's global best
    /// at each batch boundary (#179). Read only when `pool` is set, so it
    /// cannot touch a single-threaded solve. False is the A/B control arm: the
    /// bound then moves only on own incumbents and adoptions.
    bool share_objective_bound = true;
};

}  // namespace cbls
