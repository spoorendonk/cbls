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
    /// Largest violation over the REAL constraints at `state` -- the artificial
    /// `obj <= bound` row excluded, exactly as `SearchResult::best_violation`
    /// defines it. Carried with the state rather than recomputed by the reader:
    /// a `ParallelSearch` assembles its result from the pool, on a thread that
    /// owns no `Model` to evaluate it against, and a result whose violation did
    /// not describe its own state reported +inf for every portfolio run.
    double violation = std::numeric_limits<double>::infinity();
};

/// A bounded, sorted store of the best solutions seen, shared across the workers
/// of a `ParallelSearch`. Every method takes the mutex, so it is the one object
/// the workers touch concurrently; everything else about a worker is private to
/// its own thread.
class SolutionPool {
public:
    explicit SolutionPool(int capacity = 10);

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
    /// Capacity was the only part fixed, deliberately: raising it only stops
    /// the capacity itself from being the binding constraint. Changing the DRAW
    /// changes search behaviour, and #161 measured three structural
    /// alternatives against this rule. None earned its place, so this rule
    /// stands.
    ///
    /// Protocol, pre-registered: engine `ea89d15` plus a temporary A/B switch,
    /// since removed; `cbls_mipfeas --threads 4 --budget 60`; seeds 101-108.
    /// The roster is the six instances whose single-threaded, pre-`0dc826b`
    /// last-improvement times in #158 fell at or before 75% of the budget:
    /// binkar10_1, neos5, gen-ip054, markshare2, mas76 and mad. 192 runs,
    /// serial, arm order rotated per (instance, seed), no worker lost. The
    /// metric is the MIPLIB primal gap to the proven optimum, paired per
    /// (instance, seed). An arm would land only if its mean paired difference
    /// had bootstrap and t 95% CIs both below zero, and it made no more
    /// instances worse than better. Figures are the mean paired difference
    /// against this rule (negative = better), with a stratified bootstrap 95%
    /// CI:
    ///
    ///  - a per-worker RESERVED SLOT, the answer #135 named: the better half
    ///    plus the asking worker's own best, whatever its rank.
    ///    -0.0024, CI [-0.0087, +0.0038].
    ///  - a per-worker TABU set of start points already handed to that worker,
    ///    by exact identity (objective plus an assignment hash): -0.0015,
    ///    CI [-0.0075, +0.0041]. 53% of its draws found the whole better half
    ///    tabu and fell back to an ordinary kick. That rate is pooled: 919 of
    ///    the 1,110 fallbacks were on markshare2, and none on binkar10_1.
    ///  - a draw over the whole pool WEIGHTED BY DISTANCE (Hamming, integer
    ///    variables) from the asker's live assignment: -0.0019,
    ///    CI [-0.0073, +0.0041].
    ///
    /// Every interval spans zero, so none is adopted. That is "could not be
    /// separated from this rule at this power", not "measured to make no
    /// difference". All three point estimates lean the alternative's way, and
    /// each CI's half-width exceeds its mean. The roster was also weak:
    ///
    ///  - after #158's restore-before-kick (`0dc826b`), it mostly no longer
    ///    stalls. Off neos5, only 19 of 40 control runs made their last
    ///    improvement by the 45s mark;
    ///  - binkar10_1 at four threads finished between 9,097 and 10,135 on all
    ///    8 control seeds, where #161's 3-seed pre-`0dc826b` table ran from 10K
    ///    to 3.0M;
    ///  - neos5 reached its optimum on every run;
    ///  - gen-ip054 drew only 0-3 times per run. Slot and tabu matched control
    ///    on all 8 seeds there;
    ///  - markshare2 sits at a gap of 0.98-0.99 on every run, where the metric
    ///    is saturated.
    ///
    /// The signal therefore comes mostly from binkar10_1 and mas76. Revisit
    /// only with a roster that stalls on the engine as it then stands; the
    /// distance draw, whose point estimate was best on binkar10_1, is the
    /// natural first arm.
    std::optional<Solution> get_restart_point(RNG& rng) const;
    size_t size() const;

private:
    int capacity_;
    std::vector<Solution> solutions_;
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
};

}  // namespace cbls
