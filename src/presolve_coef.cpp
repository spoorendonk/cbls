// Coefficient tightening (#327). See src/presolve_internal.h.
//
// Achterberg, Bixby, Gu, Rothberg, Weninger, "Presolve reductions in mixed
// integer programming", INFORMS J. Computing 32(2), 2020 (ZIB-Report 16-44),
// Section 3.3, eq. (3.6). For a row `A_S x_S + a_k x_k <= b` with x_k integral,
// `a_k > 0`, `u_k < inf` and `u_S = sup A_S x_S` over the column bounds:
//
//   if  a_k >= d := b - u_S - a_k (u_k - 1) > 0
//   then  A_S x_S + (a_k - d) x_k <= b - d u_k
//
// has the same integer solutions and dominates the original. The paper states
// only `a_k > 0`; `a_k < 0` follows from substituting y = -x_k (`y <= -l_k`):
// it needs `l_k > -inf` and `-a_k >= d := b - u_S - a_k (l_k + 1) > 0`, and
// gives `A_S x_S + (a_k + d) x_k <= b + d l_k`. A `>=` row is negated first.
//
// With M = u_S + max(a_k u_k, a_k l_k) the row's maximum activity, both cases
// read `d = (b - M) + |a_k|`, and the step moves a_k toward 0 by d and b by the
// same amount as M, so `gap = b - M` is invariant under it: computed once per
// row, it serves every column of the row in any order.

#include "presolve_internal.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cbls::presolve_detail {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// Row r's single finite side as `s * activity <= b`: s = +1 for `<= hi`,
/// s = -1 for `>= lo`. False when the row has both sides or none.
bool one_sided(const Work& w, std::size_t r, double& s, double& b) {
    const bool has_lo = std::isfinite(w.row_lo[r]);
    const bool has_hi = std::isfinite(w.row_hi[r]);
    if (has_lo == has_hi) {
        return false;
    }
    s = has_hi ? 1.0 : -1.0;
    b = has_hi ? w.row_hi[r] : -w.row_lo[r];
    return true;
}

/// The maximum of `s * activity` of row r over the column box, and the sum of
/// its terms' magnitudes. False when it is infinite.
bool max_activity(const Work& w, std::size_t r, double s, double& max_act, double& magnitude) {
    for (std::size_t p = w.row_begin[r]; p < w.row_end[r]; ++p) {
        const double a = s * w.coefs[p];
        if (a == 0.0) {
            continue;
        }
        const auto j = static_cast<std::size_t>(w.cols[p]);
        const double term = a > 0.0 ? a * w.ub[j] : a * w.lb[j];
        if (!std::isfinite(term)) {
            return false;
        }
        max_act += term;
        magnitude += std::abs(term);
    }
    return true;
}

/// Tighten the integral coefficients of row r, one-sided as `s * activity <= b`.
///
/// Margin: `gap = b - M` is computed with error at most `(nnz + 3) * eps *
/// (|b| + magnitude)`, the activity-scaled bound `reduce_implied_sides` uses
/// (#211), plus the side's own error from earlier folds (`row_roundoff`, which
/// grows with each step taken here). `d` is reduced by that, and by `eps * |a|`
/// for the rounding of the new coefficient `a' = fl(a -+ d)`, so the step
/// actually taken, `a - a'` exactly, never exceeds the true d: a smaller step
/// is still valid (the proof of (3.6) holds for any step in (0, min(d, |a|)]).
/// A step at or below `feasibility_tolerance` is not taken: it changes nothing
/// a search could see, and would only churn the fixed-point loop.
///
/// The new side `b + (a' - a) v` (v the bound the term's max uses) is folded
/// like a fixed column's shift: one more fold, `2 |shift|` added to `row_scale`.
void tighten_row(Work& w, std::size_t r, double s, double b) {
    double max_act = 0.0;
    double magnitude = 0.0;
    if (!max_activity(w, r, s, max_act, magnitude)) {
        return;  // M infinite: (3.6) needs u_S finite.
    }
    const std::size_t begin = w.row_begin[r];
    const std::size_t end = w.row_end[r];
    const double gap = b - max_act;
    const double per_unit = static_cast<double>(w.row_live[r] + 3) * kEps;
    const double gap_error = per_unit * (std::abs(b) + magnitude);
    if (gap + gap_error + w.row_roundoff(r) >= 0.0) {
        // The side is implied, or is within rounding of it (d >= |a_k| for
        // every k): `reduce_implied_sides` drops it if it can prove it, and a
        // step here would only leave a coefficient of rounding-noise size.
        return;
    }
    const double infinity = w.opts.propagation.infinity;
    const double min_step = w.opts.feasibility_tolerance;
    bool row_changed = false;
    for (std::size_t p = begin; p < end; ++p) {
        const double a = s * w.coefs[p];
        const auto j = static_cast<std::size_t>(w.cols[p]);
        // A frozen column is not skipped: its coefficient in a linear row
        // changes, but the column is neither removed nor substituted.
        if (a == 0.0 || w.integral[j] == 0 || w.lb[j] == w.ub[j]) {
            continue;
        }
        const double abs_a = std::abs(a);
        const double d_exact = gap + abs_a;
        const double d =
            d_exact - gap_error - w.row_roundoff(r) - (kEps * (abs_a + std::abs(d_exact)));
        // gap < 0 makes d < |a|; checked anyway, so the coefficient keeps its
        // sign whatever the rounding.
        if (!(d > min_step) || d >= abs_a) {
            continue;
        }
        const double v = a > 0.0 ? w.ub[j] : w.lb[j];
        const double a_new = a > 0.0 ? a - d : a + d;
        // The step taken, not the one asked for: at a large |a|, fl(a - d) can
        // round back to a, and a step that changes nothing must not count.
        if (!(std::abs(a - a_new) > min_step)) {
            continue;
        }
        const double shift = (a_new - a) * v;
        const double b_new = b + shift;
        if (!std::isfinite(shift) || !std::isfinite(b_new) || std::abs(b_new) >= infinity) {
            continue;
        }
        b = b_new;
        // Three roundings (a' - a, the product, the sum), against two in a
        // fold, so the shift counts twice toward the scale.
        w.row_scale[r] += 2.0 * std::abs(shift);
        ++w.row_folds[r];
        w.coefs[p] = s * a_new;  // nonzero: fl(a - d) for 0 < d < |a| keeps a's sign
        ++w.stats.coefs_tightened;
        row_changed = true;
    }
    if (row_changed) {
        if (s > 0.0) {
            w.row_hi[r] = b;
        } else {
            w.row_lo[r] = -b;
        }
        ++w.stats.rows_coef_tightened;
        w.changed = true;
    }
}

}  // namespace

void reduce_coef_tightening(Work& w) {
    for (std::size_t r = 0; r < w.n_rows(); ++r) {
        double s = 0.0;
        double b = 0.0;
        if (w.row_removed[r] != 0 || w.row_live[r] == 0 || !one_sided(w, r, s, b)) {
            continue;
        }
        tighten_row(w, r, s, b);
    }
}

}  // namespace cbls::presolve_detail
