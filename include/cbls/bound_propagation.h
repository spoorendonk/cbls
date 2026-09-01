#pragma once

// Activity-based bound propagation over linear rows.
//
// Model adapters (MPS, AMPL .nl) must hand the engine finite variable bounds,
// and historically did so by substituting a fixed magnitude for an infinite
// one. That substitution is unsound: the magnitude is not implied by the
// constraints, so it can cut off feasible points, including the optimum.
//
// The pass here derives bounds that *are* implied — for each linear row, the
// min/max activity of the other terms bounds the remaining one — so, up to the
// floating-point margin documented on `safety_absolute` below, it removes only
// points no feasible solution occupies. Where it cannot finitize a
// column, the caller's clamp remains as the fallback it always was.
//
// This is deliberately activity-based tightening only. Coefficient tightening,
// redundant-row removal, aggregation, probing and dual reductions are *not*
// here, and neither is anything nonlinear: a caller with nonlinear rows simply
// omits them, which costs tightening but never validity.

#include <cstdint>
#include <vector>

namespace cbls {

/// The magnitude at which a bound conventionally means "no bound" — the
/// MPS/CPLEX/SCIP reading, and what the model adapters treat as infinite.
inline constexpr double kBoundInfinity = 1.0e20;

inline bool is_unbounded_below(double lb) {
    return lb <= -kBoundInfinity;
}
inline bool is_unbounded_above(double ub) {
    return ub >= kBoundInfinity;
}

/// One linear row `lo <= sum_k coefs[k] * x[cols[k]] <= hi`, as a **view**.
///
/// `cols` and `coefs` are parallel arrays of `nnz` entries that the caller owns
/// and must keep alive for the call — propagation reads them once per pass and
/// never writes them. They are not copied: the constraint matrix of a large MIP
/// runs to hundreds of megabytes, and duplicating it to hand it over here would
/// be the single largest allocation the model build makes.
///
/// A one-sided row leaves the unused side infinite. A column may repeat within a
/// row, in which case the row is used as written (no term merging is performed).
struct LinearRow {
    const int32_t* cols = nullptr;
    const double* coefs = nullptr;
    int32_t nnz = 0;
    double lo = 0.0;
    double hi = 0.0;
};

struct BoundPropagationOptions {
    /// Fixed-point iteration cap. Each pass is O(nnz); propagation on a large
    /// instance must not eat the search budget, so this is a hard stop rather
    /// than a convergence tolerance.
    int max_passes = 10;

    /// Magnitude at or beyond which a bound is treated as infinite. A derived
    /// bound wider than this is not recorded — it says nothing the caller did
    /// not already know.
    double infinity = kBoundInfinity;

    /// A derived bound is applied only if it improves the incumbent by more
    /// than `max(min_absolute_improvement, min_relative_improvement * |b|)`.
    /// This keeps the fixed point from crawling on floating-point noise; the
    /// pass cap bounds the work either way.
    double min_absolute_improvement = 1e-7;
    double min_relative_improvement = 1e-9;

    /// Derived bounds are relaxed outward by
    /// `max(safety_absolute, safety_relative * |b|)` before being applied, to
    /// absorb rounding in the activity sums. Integral columns are rounded after
    /// this relaxation. Note the margin is scaled to the derived bound, not to
    /// the activity it came from, so it is a practical guard rather than a proof:
    /// a row summing ~1e5 terms of magnitude ~1e9 accumulates more error than
    /// this absorbs. No instance on the MIPLIB roster has reached that regime.
    double safety_absolute = 1e-9;
    double safety_relative = 1e-12;
};

struct BoundPropagationStats {
    int passes = 0;               ///< Passes actually run (<= max_passes).
    int n_tightened = 0;          ///< Columns whose lb or ub improved.
    int n_finitized = 0;          ///< Columns that had an infinite bound and no longer do.
    int n_fixed = 0;              ///< Columns whose box collapsed to a single point.
    bool infeasible = false;      ///< A column's derived box was empty: the system has no solution.
    bool hit_pass_limit = false;  ///< Stopped at `max_passes` with bounds still moving.
};

/// Tighten `lb`/`ub` in place using the implied bounds of `rows`.
///
/// `integral[j]` non-zero means column `j` takes integer values only, and its
/// derived bounds are rounded inward. `lb`, `ub` and `integral` must all have
/// one entry per column; rows referencing a column outside that range are
/// rejected. Each row's `cols`/`coefs` must remain valid for the whole call.
///
/// Never widens a bound, and (up to the `safety_absolute` margin) never removes
/// a point that satisfies every row — so a caller may apply the result
/// unconditionally. On `infeasible` the bounds
/// are left as far as propagation got; the caller decides what to report.
BoundPropagationStats propagate_bounds(const std::vector<LinearRow>& rows,
                                       const std::vector<uint8_t>& integral,
                                       std::vector<double>& lb, std::vector<double>& ub,
                                       const BoundPropagationOptions& opts = {});

}  // namespace cbls
