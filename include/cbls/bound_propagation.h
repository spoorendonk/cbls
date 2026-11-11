#pragma once

// Activity-based bound propagation over linear rows, so model adapters (MPS,
// .nl) can hand the engine finite bounds that are *implied* (a row's other
// terms' min/max activity bounds the remaining one) rather than a substituted
// magnitude that may cut off the optimum. Up to the `safety_absolute` margin
// it removes only infeasible points; a column it cannot finitize falls back to
// the caller's clamp. Activity-based tightening only: no coefficient
// tightening, row removal, aggregation, probing, dual reductions or nonlinear
// rows (a caller omits those, losing tightening, never validity).

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
/// `cols`/`coefs`: parallel arrays of `nnz` entries, caller-owned and alive for
/// the call, read once per pass, never written. Not copied: a large MIP's
/// matrix runs to hundreds of MB, and a copy would be the model build's largest
/// allocation. A one-sided row leaves the unused side infinite. A repeated
/// column is used as written (no term merging).
struct LinearRow {
    const int32_t* cols = nullptr;
    const double* coefs = nullptr;
    int32_t nnz = 0;
    double lo = 0.0;
    double hi = 0.0;
};

struct BoundPropagationOptions {
    /// Hard pass cap, not a convergence tolerance: each pass is O(nnz), and
    /// propagation must not eat a large instance's search budget.
    int max_passes = 10;

    /// Magnitude at or beyond which a bound is infinite; a derived bound wider
    /// than this is not recorded.
    double infinity = kBoundInfinity;

    /// A derived bound applies only if it improves by more than
    /// `max(min_absolute_improvement, min_relative_improvement * |b|)`, so the
    /// fixed point does not crawl on rounding noise.
    double min_absolute_improvement = 1e-7;
    double min_relative_improvement = 1e-9;

    /// Derived bounds are relaxed outward by
    /// `max(safety_absolute, safety_relative * |b|)` to absorb rounding in the
    /// activity sums, then rounded for integral columns. Scaled to the bound,
    /// not the activity, so a practical guard, not a proof: ~1e5 terms of ~1e9
    /// accumulate more error than it absorbs (no MIPLIB roster instance does).
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
/// `integral[j]` non-zero: column `j` is integer, its derived bounds rounded
/// inward. `lb`, `ub`, `integral` have one entry per column; a row naming a
/// column outside that range throws `std::invalid_argument`. Each row's arrays
/// must stay valid for the call.
///
/// Never widens a bound and (up to the safety margin) never removes a point
/// satisfying every row, so the result may be applied unconditionally. On
/// `infeasible` the bounds stay as far as propagation got; the caller reports.
BoundPropagationStats propagate_bounds(const std::vector<LinearRow>& rows,
                                       const std::vector<uint8_t>& integral,
                                       std::vector<double>& lb, std::vector<double>& ub,
                                       const BoundPropagationOptions& opts = {});

}  // namespace cbls
