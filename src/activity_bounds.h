#pragma once

// Row-activity arithmetic shared by bound propagation (src/bound_propagation.cpp)
// and the domain propagator (#329). Not installed.

#include "cbls/bound_propagation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace cbls::activity_detail {

/// Bounds are read through these, so that a sentinel magnitude (`1e20` and up,
/// the MPS/CPLEX/SCIP convention) behaves like the infinity it stands for.
/// Both map *outward* only — never turning a real bound into a tighter one.
inline double eff_lo(double b, double infinity) {
    return b <= -infinity ? -std::numeric_limits<double>::infinity() : b;
}
inline double eff_hi(double b, double infinity) {
    return b >= infinity ? std::numeric_limits<double>::infinity() : b;
}

/// Running min or max activity of a row, kept as the sum of its finite terms
/// plus a count of the infinite ones. The count is what lets a row tighten the
/// single column that made it infinite: with exactly one infinite contribution,
/// the activity of *everything else* is still finite and still bounds it.
struct Activity {
    double sum = 0.0;
    /// Sum of the finite terms' magnitudes: what the rounding error of `sum`
    /// (and of `without`) scales with, however much of it cancels.
    double magnitude = 0.0;
    int n_infinite = 0;

    void add(double contribution) {
        if (std::isfinite(contribution)) {
            sum += contribution;
            magnitude += std::abs(contribution);
        } else {
            ++n_infinite;
        }
    }

    /// Undoes `add(contribution)`. The finite sum is not restored bit for bit
    /// (floating-point subtraction); callers needing exact restores trail
    /// whole activities instead.
    void remove(double contribution) {
        if (std::isfinite(contribution)) {
            sum -= contribution;
            magnitude -= std::abs(contribution);
        } else {
            --n_infinite;
        }
    }

    /// The activity of the row excluding one term, or `unbounded` when two or
    /// more terms are infinite in that direction.
    [[nodiscard]] double without(double contribution, double unbounded) const {
        if (std::isfinite(contribution)) {
            return n_infinite == 0 ? sum - contribution : unbounded;
        }
        return n_infinite == 1 ? sum : unbounded;
    }
};

/// The term's contribution to the row's minimum activity (and its maximum),
/// given the term's effective column bounds.
inline double min_contribution(double coef, double lo, double hi) {
    return coef > 0.0 ? coef * lo : coef * hi;
}
inline double max_contribution(double coef, double lo, double hi) {
    return coef > 0.0 ? coef * hi : coef * lo;
}

/// The roundoff margin, per unit of `|row bound| + activity magnitude`, of a
/// bound derived on one term (coefficient `coef`) of an `nnz`-term row.
///
/// The margin a derived bound is relaxed by has to cover the error it was
/// computed with, and that error scales with the ACTIVITY, not with the
/// bound: a slack of 3 left over from activities of 1e8 carries the 1e8's
/// rounding (#211). A recursive sum of n terms is off by at most about
/// n * DBL_EPSILON * sum |terms|; the rest-of-row subtraction, the slack
/// and the division add three operations. So the margin in x-space is
/// `(nnz + 3) * DBL_EPSILON * (|row bound| + magnitude) / |coef|`, and this
/// returns everything but the parenthesised factor.
inline double roundoff_per_unit(std::size_t nnz, double coef) {
    return static_cast<double>(nnz + 3) * std::numeric_limits<double>::epsilon() / std::abs(coef);
}

/// The safety margin a derived bound `b` is relaxed by.
inline double relaxation(double b, const BoundPropagationOptions& opts) {
    return std::max(opts.safety_absolute, opts.safety_relative * std::abs(b));
}

/// True if `candidate` is a materially tighter upper bound than `incumbent`.
/// Lower bounds are compared through the same function with both signs flipped.
inline bool improves(double candidate, double incumbent, const BoundPropagationOptions& opts) {
    if (!std::isfinite(incumbent)) {
        return incumbent > 0.0;  // +inf: any finite candidate improves it.
    }
    const double threshold = std::max(opts.min_absolute_improvement,
                                      opts.min_relative_improvement * std::abs(incumbent));
    return candidate < incumbent - threshold;
}

}  // namespace cbls::activity_detail
