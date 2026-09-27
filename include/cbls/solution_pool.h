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
    /// alternatives against this rule. None could be separated from it, so
    /// this rule stands.
    ///
    /// Engine: `ea89d15` plus the temporary switch in `8b7b37b`, selected by
    /// `CBLS_ISSUE161_ARM=slot|tabu|distance|wholepool` and removed in the
    /// commit after it. Pre-registered protocol:
    ///  - `cbls_mipfeas --threads 4 --budget 60`, seeds 201-208;
    ///  - roster admitted by a control-only pilot on disjoint seeds over the
    ///    smoke set plus 9 fixed-seed draws from the roster. An instance was
    ///    admitted only if its median last improvement was <= 45s, it was
    ///    optimal in <= 1 of 3 seeds, its median gap was < 0.9, its median
    ///    was >= 10 restart draws per run and no pilot run lost a worker. It
    ///    was re-derived rather than taken from #158's recorded times, because
    ///    that set no longer stalled on this engine (campaign 1, below). That
    ///    admitted binkar10_1, b1c1s1, mas76, pk1 and fastxgemm-n2r6s0t2, and
    ///    27 of their 40 control runs made their last improvement by 45s;
    ///  - 200 runs, serial, arm order rotated per (instance, seed), no worker
    ///    lost;
    ///  - metric: MIPLIB primal gap to the proven optimum, paired per (instance,
    ///    seed);
    ///  - an arm lands only if its bootstrap and t 95% CIs are both below zero
    ///    and it makes no more instances worse than better.
    ///
    /// Mean paired difference against this rule (negative = better), with the
    /// stratified bootstrap 95% CI:
    ///
    ///  - a per-worker RESERVED SLOT, the answer #135 named: the better half
    ///    plus the asking worker's own best, whatever its rank. +0.0007,
    ///    CI [-0.0154, +0.0150]. Its own best was missing from the better
    ///    half on 33% of draws (501/1530) and was drawn on 6% (89). 13 of 40
    ///    runs still ended on control's objective bit for bit, although the
    ///    slot changed the candidate set in every run -- on those runs the
    ///    change never reached the final objective.
    ///    Since #158 (`0dc826b`), diversify() also restores the worker's own
    ///    incumbent before kicking, unless it stands on an adopted peer point,
    ///    so the slot mostly duplicates the return to its own incumbent that
    ///    the kick already makes, minus the kick's perturbation.
    ///  - a per-worker TABU set of the start points a worker has actually
    ///    restarted from, by exact identity (objective plus an assignment
    ///    hash): +0.0033, CI [-0.0089, +0.0166]. It excluded >= 1 entry on 69%
    ///    of draws (1041/1517). On 39% (590, of which 447 were on pk1) it
    ///    excluded all of them and fell back to an ordinary kick. 13 of 40
    ///    runs were bit-identical to control, although it narrowed the draw
    ///    in every run.
    ///  - a draw over the whole pool WEIGHTED BY DISTANCE (Hamming, integer
    ///    variables) from the asker's live assignment: -0.0026,
    ///    CI [-0.0223, +0.0196]. Barely a distance draw in practice: its
    ///    distribution sat at a draw-weighted total-variation distance of 0.04
    ///    from uniform (0.002 on pk1, 0.24 on fastxgemm). The asker was about
    ///    equally far from every entry, as expected of a pool of refinements
    ///    of one point. It also declined 48 of 1521 draws (30 on fastxgemm)
    ///    where every candidate matched the asker on the integer variables.
    ///    So it could not separate "prefer far" from "draw from the
    ///    whole pool instead of the better half". A diagnostic fifth arm,
    ///    uniform over the same candidates, was -0.0056, CI [-0.0171, +0.0074],
    ///    and also indistinguishable.
    ///
    /// All intervals, bootstrap and t, span zero. The narrowest half-width of
    /// the three (0.013, tabu) is about four times the largest mean (0.0033).
    /// That means "not separable from this rule at this power", not "measured
    /// to make no difference". Untested here: a diversity criterion at
    /// `submit`, which is where the pool's sameness comes from, and a tabu by
    /// near-identity rather than exact identity.
    ///
    /// Campaign 1 (engine `0113c8f`, seeds 101-108, #158's recorded stall set)
    /// gave slot -0.0024, tabu -0.0015 and distance -0.0019, with every CI
    /// spanning zero. It is superseded because on that engine its roster
    /// mostly no longer stalled: neos5 was optimal on every run, gen-ip054 drew
    /// 0-3 times per run, and markshare2's gap was saturated at 0.98-0.99.
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
