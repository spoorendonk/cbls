#pragma once

// The runner's check of a point against the MPS problem as read (#325).
//
// assess_result judges the built model, not the file; this checks the point
// against the `MpsProblem` itself: bounds, integrality, every row (compensated
// sum), and `c·x + offset` against the reported objective. Shares only the
// parsed file with the build; no second Model, which the driver's cap would count.
//
// Tolerances follow verify_solution.py with the runner's `--feas-tol` as the
// absolute term: rows `feas_tol + 1e-9 * scale` (scale: the larger of the finite
// sides and the sum of |a_ij x_j|), bounds `feas_tol + 1e-9 * |bound|`,
// integrality 1e-9 (assess_result's gate), the objective the runner's drift gate
// `1e-6 * (|objective| + 1)` plus `1e-9 * (sum |c_j x_j| + |offset|)`. The
// reported objective is the engine's plain sum, so terms that cancel to a small
// optimum can leave it further from the compensated sum here than the drift
// gate alone allows; the scale term, as for rows, absorbs that roundoff.

#include <algorithm>
#include <cbls/bound_propagation.h>
#include <cbls/io_mps.h>
#include <cmath>
#include <cstddef>
#include <vector>

namespace cbls::mipfeas {

struct OriginalCheck {
    double max_row_ratio = 0.0;    ///< Worst row violation over its tolerance.
    double max_bound_ratio = 0.0;  ///< Worst bound violation over its tolerance.
    double max_integrality = 0.0;  ///< Worst |x - round(x)| over integer columns.
    double objective = 0.0;        ///< c·x + offset at the point.
    double objective_drift = 0.0;  ///< |objective - reported|.
    double objective_scale = 0.0;  ///< sum |c_j x_j| + |offset|.
    bool size_mismatch = false;    ///< The point does not have one value per column.
    bool non_finite = false;       ///< Some value, or the objective, is not finite.

    [[nodiscard]] bool passes(double reported_objective) const {
        return !size_mismatch && !non_finite && max_row_ratio <= 1.0 && max_bound_ratio <= 1.0 &&
               max_integrality <= 1e-9 &&
               objective_drift <=
                   (1e-6 * (std::abs(reported_objective) + 1.0)) + (1e-9 * objective_scale);
    }
};

/// Neumaier's compensated sum, accumulated one term at a time.
struct CompensatedSum {
    double sum = 0.0;
    double comp = 0.0;
    void add(double v) {
        const double t = sum + v;
        comp += std::abs(sum) >= std::abs(v) ? (sum - t) + v : (v - t) + sum;
        sum = t;
    }
    [[nodiscard]] double value() const { return sum + comp; }
};

/// Two sides `lo <= v <= hi`; a NaN or infinite side is no side. Named fields
/// rather than two `double&`, which a transposed call would invert silently.
struct Sides {
    double lo = -kMpsInf;
    double hi = kMpsInf;
};

/// How far `v` lies outside `s`, and the magnitude of the side it violates.
struct SideViolation {
    double violation = 0.0;
    double side_scale = 0.0;
};

inline SideViolation side_violation(double v, const Sides& s) {
    SideViolation out;
    if (!std::isnan(s.lo) && !is_unbounded_below(s.lo) && v < s.lo) {
        out = {s.lo - v, std::abs(s.lo)};
    }
    if (!std::isnan(s.hi) && !is_unbounded_above(s.hi) && v - s.hi > out.violation) {
        out = {v - s.hi, std::abs(s.hi)};
    }
    return out;
}

/// A row's sides from its sense, rhs and range, re-derived from the MPS
/// definition rather than shared with the adapter, so a mistake there is not
/// repeated here.
inline Sides mps_row_sides(const MpsRow& r) {
    const double rng = std::abs(r.range);
    Sides s;
    if (r.sense == MpsRowSense::L) {
        s.hi = r.rhs;
        if (r.range != 0.0) {
            s.lo = r.rhs - rng;
        }
    } else if (r.sense == MpsRowSense::G) {
        s.lo = r.rhs;
        if (r.range != 0.0) {
            s.hi = r.rhs + rng;
        }
    } else if (r.range >= 0.0) {  // E: [rhs, rhs + range]
        s.lo = r.rhs;
        s.hi = r.rhs + r.range;
    } else {  // E, negative range: [rhs + range, rhs]
        s.lo = r.rhs + r.range;
        s.hi = r.rhs;
    }
    return s;
}

/// Check `x` (one value per MPS column) against `prob`. O(nnz) time, O(rows)
/// extra memory.
inline OriginalCheck check_original_point(const MpsProblem& prob, const std::vector<double>& x,
                                          double reported_objective, double feas_tol) {
    OriginalCheck out;
    if (x.size() != prob.vars.size()) {
        out.size_mismatch = true;
        return out;
    }
    for (std::size_t j = 0; j < x.size(); ++j) {
        const MpsVar& v = prob.vars[j];
        if (!std::isfinite(x[j])) {
            out.non_finite = true;
            return out;
        }
        const SideViolation sv = side_violation(x[j], Sides{v.lb, v.ub});
        out.max_bound_ratio =
            std::max(out.max_bound_ratio, sv.violation / (feas_tol + (1e-9 * sv.side_scale)));
        if (v.kind != MpsVarKind::Continuous) {
            out.max_integrality = std::max(out.max_integrality, std::abs(x[j] - std::round(x[j])));
        }
    }
    const std::size_t n_rows = prob.rows.size();
    std::vector<CompensatedSum> activity(n_rows);
    std::vector<double> abs_terms(n_rows, 0.0);
    CompensatedSum obj;
    obj.add(prob.objective_offset);
    out.objective_scale = std::abs(prob.objective_offset);
    for (const MpsNonzero& nz : prob.nonzeros) {
        const double term = nz.value * x[static_cast<std::size_t>(nz.col_idx)];
        if (nz.row_idx < 0) {
            obj.add(term);
            out.objective_scale += std::abs(term);
        } else if (static_cast<std::size_t>(nz.row_idx) < n_rows) {
            const auto r = static_cast<std::size_t>(nz.row_idx);
            activity[r].add(term);
            abs_terms[r] += std::abs(term);
        }
    }
    for (std::size_t r = 0; r < n_rows; ++r) {
        const double act = activity[r].value();
        // inf - inf is NaN, which no `<` or `>` flags; and finite terms near
        // DBL_MAX can sum |term| to inf while the activity stays finite, which
        // would make the tolerance infinite.
        if (!std::isfinite(act) || !std::isfinite(abs_terms[r])) {
            out.non_finite = true;
            return out;
        }
        const Sides s = mps_row_sides(prob.rows[r]);
        const SideViolation sv = side_violation(act, s);
        double scale = abs_terms[r];
        if (!is_unbounded_below(s.lo)) {
            scale = std::max(scale, std::abs(s.lo));
        }
        if (!is_unbounded_above(s.hi)) {
            scale = std::max(scale, std::abs(s.hi));
        }
        out.max_row_ratio = std::max(out.max_row_ratio, sv.violation / (feas_tol + (1e-9 * scale)));
    }
    out.objective = obj.value();
    out.non_finite = !std::isfinite(out.objective) || !std::isfinite(out.objective_scale);
    out.objective_drift = std::abs(out.objective - reported_objective);
    return out;
}

}  // namespace cbls::mipfeas
