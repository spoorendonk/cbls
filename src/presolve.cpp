// Presolve with postsolve. See include/cbls/presolve.h and src/presolve_internal.h.

#include "cbls/presolve.h"

#include "presolve_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cbls {

// ---------------------------------------------------------------------------
// Postsolve
// ---------------------------------------------------------------------------

void Postsolve::set_mapping(std::size_t n_original, std::vector<int32_t> reduced_to_col) {
    n_original_ = n_original;
    reduced_to_col_ = std::move(reduced_to_col);
}

std::vector<double> Postsolve::apply(const std::vector<double>& reduced_x) const {
    if (reduced_x.size() != reduced_to_col_.size()) {
        throw std::invalid_argument("Postsolve::apply: expected " +
                                    std::to_string(reduced_to_col_.size()) +
                                    " reduced values, got " + std::to_string(reduced_x.size()));
    }
    // NaN marks a column no step restored: a bug, and loud wherever it lands.
    std::vector<double> x(n_original_, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < reduced_x.size(); ++i) {
        x[static_cast<std::size_t>(reduced_to_col_[i])] = reduced_x[i];
    }
    for (auto it = fixed_.rbegin(); it != fixed_.rend(); ++it) {
        x[static_cast<std::size_t>(it->col)] = it->value;
    }
    return x;
}

namespace presolve_detail {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

double normalize_lo(double b, double infinity) {
    return b <= -infinity ? -kInf : b;
}
double normalize_hi(double b, double infinity) {
    return b >= infinity ? kInf : b;
}

void require(bool ok, const std::string& what) {
    if (!ok) {
        throw std::invalid_argument("presolve: " + what);
    }
}

bool any_nan(const std::vector<double>& v) {
    return std::any_of(v.begin(), v.end(), [](double d) { return std::isnan(d); });
}

bool all_finite(const std::vector<double>& v) {
    return std::all_of(v.begin(), v.end(), [](double d) { return std::isfinite(d); });
}

/// A lower bound at +infinity or an upper bound at -infinity (an empty side,
/// not a missing one) has no meaning the reductions could fold.
bool any_empty_side(const std::vector<double>& lo, const std::vector<double>& hi, double infinity) {
    for (std::size_t i = 0; i < lo.size(); ++i) {
        if (lo[i] >= infinity || hi[i] <= -infinity) {
            return true;
        }
    }
    return false;
}

void validate(const LinearProblem& p, double infinity) {
    const std::size_t n = p.n_cols();
    const std::size_t m = p.n_rows();
    require(
        p.col_ub.size() == n && p.integral.size() == n && p.frozen.size() == n && p.obj.size() == n,
        "column arrays differ in size");
    require(p.row_hi.size() == m && p.row_start.size() == m + 1,
            "row arrays differ in size (row_start needs n_rows + 1 entries)");
    require(p.row_start.front() == 0 && p.row_start.back() == p.cols.size() &&
                p.coefs.size() == p.cols.size(),
            "row_start does not span cols/coefs");
    for (std::size_t i = 0; i < m; ++i) {
        require(p.row_start[i] <= p.row_start[i + 1], "row_start is not non-decreasing");
    }
    for (const int32_t c : p.cols) {
        require(c >= 0 && static_cast<std::size_t>(c) < n,
                "column index " + std::to_string(c) + " out of range");
    }
    require(!any_nan(p.col_lb) && !any_nan(p.col_ub) && !any_nan(p.row_lo) && !any_nan(p.row_hi),
            "NaN in bounds");
    require(all_finite(p.coefs) && all_finite(p.obj) && std::isfinite(p.obj_offset),
            "non-finite coefficient or objective entry");
    require(!any_empty_side(p.col_lb, p.col_ub, infinity) &&
                !any_empty_side(p.row_lo, p.row_hi, infinity),
            "a lower bound at +infinity or an upper bound at -infinity");
}

/// Copy row i of `p` into `w`, merging repeated columns and dropping zeros.
/// Sorting the row costs O(len log len); it runs once per row, at load.
///
/// Repeats are summed with Neumaier's compensation, in a fixed (column,
/// coefficient) order: a plain sum of `1e16, 1, -1e16` gives 0 or 1 depending
/// on the order, and a 0 would drop the column from the row altogether.
void load_row(const LinearProblem& p, std::size_t i, std::vector<std::pair<int32_t, double>>& tmp,
              Work& w) {
    tmp.clear();
    for (std::size_t k = p.row_start[i]; k < p.row_start[i + 1]; ++k) {
        tmp.emplace_back(p.cols[k], p.coefs[k]);
    }
    std::sort(tmp.begin(), tmp.end());
    std::size_t k = 0;
    while (k < tmp.size()) {
        const int32_t col = tmp[k].first;
        double sum = tmp[k].second;
        double compensation = 0.0;
        std::size_t next = k + 1;
        for (; next < tmp.size() && tmp[next].first == col; ++next) {
            const double x = tmp[next].second;
            const double t = sum + x;
            compensation += std::abs(sum) >= std::abs(x) ? (sum - t) + x : (x - t) + sum;
            sum = t;
            ++w.stats.duplicates_merged;
        }
        sum += compensation;
        if (sum != 0.0) {
            w.cols.push_back(col);
            w.coefs.push_back(sum);
        }
        k = next;
    }
    w.row_start.push_back(w.cols.size());
}

void build_column_index(Work& w) {
    const std::size_t n = w.n_cols();
    const std::size_t m = w.n_rows();
    w.col_live.assign(n, 0);
    w.row_live.assign(m, 0);
    for (std::size_t r = 0; r < m; ++r) {
        w.row_live[r] = static_cast<int32_t>(w.row_start[r + 1] - w.row_start[r]);
        for (std::size_t p = w.row_start[r]; p < w.row_start[r + 1]; ++p) {
            ++w.col_live[static_cast<std::size_t>(w.cols[p])];
        }
    }
    w.col_start.assign(n + 1, 0);
    for (std::size_t j = 0; j < n; ++j) {
        w.col_start[j + 1] = w.col_start[j] + static_cast<std::size_t>(w.col_live[j]);
    }
    w.col_rows.resize(w.cols.size());
    w.col_offs.resize(w.cols.size());
    std::vector<std::size_t> fill(w.col_start.begin(), w.col_start.end() - 1);
    for (std::size_t r = 0; r < m; ++r) {
        for (std::size_t p = w.row_start[r]; p < w.row_start[r + 1]; ++p) {
            const std::size_t q = fill[static_cast<std::size_t>(w.cols[p])]++;
            w.col_rows[q] = static_cast<int32_t>(r);
            w.col_offs[q] = static_cast<int32_t>(p - w.row_start[r]);
        }
    }
}

/// Validate `p` and load it into `w`: bounds normalised to ±inf, rows merged.
/// The column index is not built here: only the reductions need it.
void load(const LinearProblem& p, Work& w) {
    const double infinity = w.opts.propagation.infinity;
    validate(p, infinity);
    require(p.n_rows() < static_cast<std::size_t>(std::numeric_limits<int32_t>::max()),
            "too many rows");
    const std::size_t n = p.n_cols();
    const std::size_t m = p.n_rows();
    w.lb.resize(n);
    w.ub.resize(n);
    for (std::size_t j = 0; j < n; ++j) {
        w.lb[j] = normalize_lo(p.col_lb[j], infinity);
        w.ub[j] = normalize_hi(p.col_ub[j], infinity);
    }
    w.obj = p.obj;
    w.integral = p.integral;
    w.frozen = p.frozen;
    w.col_removed.assign(n, 0);
    w.obj_offset = p.obj_offset;

    w.row_start.assign(1, 0);
    w.cols.reserve(p.cols.size());
    w.coefs.reserve(p.coefs.size());
    w.row_lo.resize(m);
    w.row_hi.resize(m);
    w.row_scale.resize(m);
    std::vector<std::pair<int32_t, double>> tmp;
    for (std::size_t i = 0; i < m; ++i) {
        load_row(p, i, tmp, w);
        w.row_lo[i] = normalize_lo(p.row_lo[i], infinity);
        w.row_hi[i] = normalize_hi(p.row_hi[i], infinity);
        const double lo_mag = std::isfinite(w.row_lo[i]) ? std::abs(w.row_lo[i]) : 0.0;
        const double hi_mag = std::isfinite(w.row_hi[i]) ? std::abs(w.row_hi[i]) : 0.0;
        w.row_scale[i] = std::max(lo_mag, hi_mag);
    }
    w.row_folds.assign(m, 0);
    w.row_removed.assign(m, 0);
}

/// Round integral columns' bounds inward and settle crossed bounds: within
/// `feasibility_tolerance` they collapse to a point, beyond it the problem is
/// infeasible. Column bounds are input, with no computed error, so no
/// roundoff term; row bounds get the `row_roundoff` term used everywhere else
/// (here only `DBL_EPSILON * |bound|`, since nothing is folded yet).
void prepare(Work& w) {
    const double tol = w.opts.feasibility_tolerance;
    for (std::size_t j = 0; j < w.n_cols(); ++j) {
        if (w.integral[j] != 0) {
            if (std::isfinite(w.lb[j])) {
                w.lb[j] = std::ceil(w.lb[j] - tol);
            }
            if (std::isfinite(w.ub[j])) {
                w.ub[j] = std::floor(w.ub[j] + tol);
            }
        }
        if (w.lb[j] > w.ub[j]) {
            if (w.lb[j] - w.ub[j] > tol) {
                w.infeasible = true;
                return;
            }
            w.ub[j] = w.lb[j];
        }
    }
    for (std::size_t r = 0; r < w.n_rows(); ++r) {
        if (w.row_lo[r] > w.row_hi[r]) {
            if (w.row_lo[r] - w.row_hi[r] > tol + w.row_roundoff(r)) {
                w.infeasible = true;
                return;
            }
            w.row_hi[r] = w.row_lo[r];
        }
    }
}

/// Error bound on the quotient `q = side / a` of row r: none when nothing was
/// folded into the side (it is input, exact) and the division was exact;
/// otherwise the side's roundoff over |a| plus one rounding of the division.
double quotient_error(const Work& w, std::size_t r, double side, double a, double q) {
    if (w.row_folds[r] == 0 && std::fma(q, a, -side) == 0.0) {
        return 0.0;
    }
    return (w.row_roundoff(r) / std::abs(a)) +
           (std::numeric_limits<double>::epsilon() * std::abs(q));
}

/// Singleton row r: its one live entry `a x` in [lo, hi] becomes bounds on x,
/// and the row goes.
///
/// Each quotient is relaxed outward by its error bound (`quotient_error`), so
/// it never cuts off a point the exact row admits. On top of that, `slack =
/// min(tol, tol / |a|)` is the most a point may sit outside the relaxed bound:
/// within it a point violates neither the row nor the column bound it widens
/// by more than `tol`. An integral column's bounds are rounded inward past
/// `slack`, and a crossing within it collapses to a point.
///
/// A quotient at or beyond `infinity`, or one that overflowed, would read as
/// "no bound" downstream, so the row is then kept as it is (as
/// `propagate_bounds` skips such a value).
void reduce_singleton(Work& w, std::size_t r) {
    std::size_t p = w.row_start[r];
    while (w.coefs[p] == 0.0) {
        ++p;
    }
    const double a = w.coefs[p];
    const auto j = static_cast<std::size_t>(w.cols[p]);
    const double lo_side = a > 0.0 ? w.row_lo[r] : w.row_hi[r];  // gives x's lower bound
    const double hi_side = a > 0.0 ? w.row_hi[r] : w.row_lo[r];
    double lo = lo_side / a;
    double hi = hi_side / a;
    if (std::isfinite(lo)) {
        lo -= quotient_error(w, r, lo_side, a, lo);
    }
    if (std::isfinite(hi)) {
        hi += quotient_error(w, r, hi_side, a, hi);
    }
    // Checked after the relaxation, which can itself carry a bound past it.
    const double infinity = w.opts.propagation.infinity;
    const auto unusable = [infinity](double side, double q) {
        return std::isfinite(side) && (!std::isfinite(q) || std::abs(q) >= infinity);
    };
    if (unusable(lo_side, lo) || unusable(hi_side, hi)) {
        return;
    }
    const double tol = w.opts.feasibility_tolerance;
    const double slack = std::min(tol, tol / std::abs(a));
    if (w.integral[j] != 0) {
        if (std::isfinite(lo)) {
            lo = std::ceil(lo - slack);
        }
        if (std::isfinite(hi)) {
            hi = std::floor(hi + slack);
        }
    }
    w.tighten_column(static_cast<int32_t>(j), lo, hi, slack);
    if (w.infeasible) {
        return;
    }
    w.remove_row(static_cast<int32_t>(r));
    ++w.stats.rows_singleton;
}

/// Running min or max activity of a row: the sum of its finite terms, their
/// magnitude (what the sum's rounding error scales with) and a count of the
/// infinite ones.
struct Activity {
    double sum = 0.0;
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
};

/// Drop each side of row r that every point in the column box satisfies, and
/// the row once it has no side left.
///
/// A side is dropped only when the computed activity clears it by its own
/// rounding bound, `(nnz + 3) * DBL_EPSILON * (|bound| + magnitude)` — the
/// same activity-scaled roundoff as `propagate_bounds` (#211) — plus the
/// bound's own error from earlier folds (`row_roundoff`), so a dropped side
/// is implied exactly, not merely to within floating-point noise.
void reduce_implied_sides(Work& w, std::size_t r) {
    Activity min_act;
    Activity max_act;
    for (std::size_t p = w.row_start[r]; p < w.row_start[r + 1]; ++p) {
        const double a = w.coefs[p];
        if (a == 0.0) {
            continue;
        }
        const auto j = static_cast<std::size_t>(w.cols[p]);
        min_act.add(a > 0.0 ? a * w.lb[j] : a * w.ub[j]);
        max_act.add(a > 0.0 ? a * w.ub[j] : a * w.lb[j]);
    }
    const double per_unit =
        static_cast<double>(w.row_live[r] + 3) * std::numeric_limits<double>::epsilon();
    const double bound_error = w.row_roundoff(r);
    double& hi = w.row_hi[r];
    double& lo = w.row_lo[r];
    if (std::isfinite(hi) && max_act.n_infinite == 0 &&
        max_act.sum + (per_unit * (std::abs(hi) + max_act.magnitude)) + bound_error <= hi) {
        hi = kInf;
        ++w.stats.row_sides_dropped;
        w.changed = true;
    }
    if (std::isfinite(lo) && min_act.n_infinite == 0 &&
        min_act.sum - (per_unit * (std::abs(lo) + min_act.magnitude)) - bound_error >= lo) {
        lo = -kInf;
        ++w.stats.row_sides_dropped;
        w.changed = true;
    }
    if (!std::isfinite(lo) && !std::isfinite(hi)) {
        w.remove_row(static_cast<int32_t>(r));
        ++w.stats.rows_redundant;
    }
}

/// Copy the live part of `w` out as the result.
PresolveResult build_result(Work& w, const std::vector<double>& lb0,
                            const std::vector<double>& ub0) {
    PresolveResult res;
    const std::size_t n = w.n_cols();
    const std::size_t m = w.n_rows();
    LinearProblem& out = res.reduced;
    res.col_to_reduced.assign(n, -1);
    for (std::size_t j = 0; j < n; ++j) {
        if (w.col_removed[j] != 0) {
            continue;
        }
        res.col_to_reduced[j] = static_cast<int32_t>(res.reduced_to_col.size());
        res.reduced_to_col.push_back(static_cast<int32_t>(j));
        out.col_lb.push_back(w.lb[j]);
        out.col_ub.push_back(w.ub[j]);
        out.integral.push_back(w.integral[j]);
        out.frozen.push_back(w.frozen[j]);
        out.obj.push_back(w.obj[j]);
        if (w.lb[j] != lb0[j] || w.ub[j] != ub0[j]) {
            ++w.stats.bounds_tightened;
        }
    }
    out.obj_offset = w.obj_offset;
    res.row_to_reduced.assign(m, -1);
    for (std::size_t r = 0; r < m; ++r) {
        if (w.row_removed[r] != 0) {
            continue;
        }
        res.row_to_reduced[r] = static_cast<int32_t>(out.row_lo.size());
        for (std::size_t p = w.row_start[r]; p < w.row_start[r + 1]; ++p) {
            if (w.coefs[p] != 0.0) {
                out.cols.push_back(res.col_to_reduced[static_cast<std::size_t>(w.cols[p])]);
                out.coefs.push_back(w.coefs[p]);
            }
        }
        out.row_start.push_back(out.cols.size());
        out.row_lo.push_back(w.row_lo[r]);
        out.row_hi.push_back(w.row_hi[r]);
    }
    w.postsolve.set_mapping(n, res.reduced_to_col);
    res.postsolve = std::move(w.postsolve);
    const PresolveStats& s = w.stats;
    const bool reduced = s.cols_fixed + s.cols_empty + s.rows_empty + s.rows_free +
                             s.rows_singleton + s.rows_redundant + s.row_sides_dropped +
                             s.bounds_tightened + s.coefs_tightened >
                         0;
    w.stats.status = reduced ? PresolveStatus::Reduced : PresolveStatus::Unchanged;
    res.stats = w.stats;
    return res;
}

/// The loaded problem with nothing removed and an identity postsolve.
PresolveResult identity_result(const LinearProblem& problem, const PresolveOptions& opts) {
    Work w(opts);
    load(problem, w);
    return build_result(w, w.lb, w.ub);
}

}  // namespace

// ---------------------------------------------------------------------------
// Work
// ---------------------------------------------------------------------------

double Work::row_roundoff(std::size_t r) const {
    return static_cast<double>(row_folds[r] + 1) * std::numeric_limits<double>::epsilon() *
           row_scale[r];
}

bool Work::can_fix(int32_t j, double value) const {
    const auto jj = static_cast<std::size_t>(j);
    const double infinity = opts.propagation.infinity;
    if (!std::isfinite(value) || !std::isfinite(obj[jj] * value)) {
        return false;
    }
    const auto in_range = [infinity](double bound, double shift) {
        return !std::isfinite(bound) || std::abs(bound - shift) < infinity;
    };
    for (std::size_t q = col_start[jj]; q < col_start[jj + 1]; ++q) {
        const auto r = static_cast<std::size_t>(col_rows[q]);
        if (row_removed[r] != 0) {
            continue;
        }
        const double shift = coefs[row_start[r] + static_cast<std::size_t>(col_offs[q])] * value;
        if (!std::isfinite(shift) || !in_range(row_lo[r], shift) || !in_range(row_hi[r], shift)) {
            return false;
        }
    }
    return true;
}

void Work::fix_column(int32_t j, double value) {
    const auto jj = static_cast<std::size_t>(j);
    for (std::size_t q = col_start[jj]; q < col_start[jj + 1]; ++q) {
        const auto r = static_cast<std::size_t>(col_rows[q]);
        if (row_removed[r] != 0) {
            continue;
        }
        const std::size_t p = row_start[r] + static_cast<std::size_t>(col_offs[q]);
        const double a = coefs[p];
        if (a == 0.0) {
            continue;
        }
        const double shift = a * value;
        if (std::isfinite(row_lo[r])) {
            row_lo[r] -= shift;
        }
        if (std::isfinite(row_hi[r])) {
            row_hi[r] -= shift;
        }
        row_scale[r] += std::abs(shift);
        ++row_folds[r];
        coefs[p] = 0.0;
        --row_live[r];
    }
    obj_offset += obj[jj] * value;
    col_removed[jj] = 1;
    col_live[jj] = 0;
    postsolve.push_fixed(j, value);
    changed = true;
}

void Work::remove_row(int32_t r) {
    const auto rr = static_cast<std::size_t>(r);
    for (std::size_t p = row_start[rr]; p < row_start[rr + 1]; ++p) {
        if (coefs[p] != 0.0) {
            --col_live[static_cast<std::size_t>(cols[p])];
        }
    }
    row_removed[rr] = 1;
    row_live[rr] = 0;
    changed = true;
}

void Work::tighten_column(int32_t j, double lo, double hi, double slack) {
    const auto jj = static_cast<std::size_t>(j);
    double new_lb = std::max(lb[jj], lo);
    double new_ub = std::min(ub[jj], hi);
    if (new_lb > new_ub) {
        if (new_lb - new_ub > slack) {
            infeasible = true;
            return;
        }
        new_ub = new_lb;
    }
    if (new_lb != lb[jj] || new_ub != ub[jj]) {
        lb[jj] = new_lb;
        ub[jj] = new_ub;
        changed = true;
    }
}

// ---------------------------------------------------------------------------
// Reductions
// ---------------------------------------------------------------------------

void reduce_propagate(Work& w) {
    std::vector<LinearRow> views;
    views.reserve(w.n_rows());
    for (std::size_t r = 0; r < w.n_rows(); ++r) {
        const std::size_t len = w.row_start[r + 1] - w.row_start[r];
        // A row longer than LinearRow can describe is skipped: it then
        // tightens nothing, which is sound.
        if (w.row_removed[r] != 0 || w.row_live[r] == 0 ||
            len > static_cast<std::size_t>(std::numeric_limits<int32_t>::max())) {
            continue;
        }
        // `propagate_bounds` covers the activity's roundoff, not the error
        // already in a folded side, so the sides are widened by it first.
        const double err = w.row_folds[r] > 0 ? w.row_roundoff(r) : 0.0;
        views.push_back(LinearRow{w.cols.data() + w.row_start[r], w.coefs.data() + w.row_start[r],
                                  static_cast<int32_t>(len), w.row_lo[r] - err, w.row_hi[r] + err});
    }
    const BoundPropagationStats s =
        propagate_bounds(views, w.integral, w.lb, w.ub, w.opts.propagation);
    if (s.infeasible) {
        w.infeasible = true;
    } else if (s.n_tightened > 0) {
        w.changed = true;
    }
}

void reduce_fixed_columns(Work& w) {
    for (std::size_t j = 0; j < w.n_cols(); ++j) {
        if (w.col_removed[j] != 0 || w.frozen[j] != 0 || w.lb[j] != w.ub[j]) {
            continue;
        }
        const double value = w.integral[j] != 0 ? std::round(w.lb[j]) : w.lb[j];
        if (!w.can_fix(static_cast<int32_t>(j), value)) {
            continue;  // kept, fixed by its bounds
        }
        w.fix_column(static_cast<int32_t>(j), value);
        ++w.stats.cols_fixed;
    }
}

void reduce_rows(Work& w) {
    for (std::size_t r = 0; r < w.n_rows() && !w.infeasible; ++r) {
        if (w.row_removed[r] != 0) {
            continue;
        }
        if (w.row_live[r] == 0) {
            // The bounds now hold minus the folded constant, up to its
            // rounding error; a residual within that plus the tolerance holds.
            const double tol = w.opts.feasibility_tolerance + w.row_roundoff(r);
            if (w.row_lo[r] > tol || w.row_hi[r] < -tol) {
                w.infeasible = true;
                return;
            }
            w.remove_row(static_cast<int32_t>(r));
            ++w.stats.rows_empty;
        } else if (!std::isfinite(w.row_lo[r]) && !std::isfinite(w.row_hi[r])) {
            w.remove_row(static_cast<int32_t>(r));
            ++w.stats.rows_free;
        } else if (w.row_live[r] == 1 && w.opts.singleton_rows) {
            reduce_singleton(w, r);
        } else if (w.opts.redundant_rows) {
            reduce_implied_sides(w, r);
        }
    }
}

void reduce_empty_columns(Work& w) {
    for (std::size_t j = 0; j < w.n_cols(); ++j) {
        if (w.col_removed[j] != 0 || w.frozen[j] != 0 || w.col_live[j] != 0) {
            continue;
        }
        const double c = w.obj[j];
        double value = 0.0;
        if (c == 0.0) {
            value = std::max(w.lb[j], std::min(0.0, w.ub[j]));
        } else if (c > 0.0 && std::isfinite(w.lb[j])) {
            value = w.lb[j];
        } else if (c < 0.0 && std::isfinite(w.ub[j])) {
            value = w.ub[j];
        } else {
            continue;  // Unbounded in the improving direction: left to the search.
        }
        if (!w.can_fix(static_cast<int32_t>(j), value)) {
            continue;
        }
        w.fix_column(static_cast<int32_t>(j), value);
        ++w.stats.cols_empty;
    }
}

}  // namespace presolve_detail

PresolveResult presolve(const LinearProblem& problem, const PresolveOptions& opts) {
    using presolve_detail::Work;
    const auto t0 = std::chrono::steady_clock::now();
    if (!opts.enabled) {
        return presolve_detail::identity_result(problem, opts);
    }
    int rounds = 0;
    {
        // Scoped so that on the infeasible path the working copy is freed
        // before the identity result is built.
        Work w(opts);
        presolve_detail::load(problem, w);
        presolve_detail::build_column_index(w);
        const std::vector<double> lb0 = w.lb;
        const std::vector<double> ub0 = w.ub;
        presolve_detail::prepare(w);
        for (int round = 0; round < opts.max_rounds && !w.infeasible; ++round) {
            w.changed = false;
            ++w.stats.rounds;
            if (opts.propagate_bounds) {
                presolve_detail::reduce_propagate(w);
            }
            if (!w.infeasible && opts.fixed_columns) {
                presolve_detail::reduce_fixed_columns(w);
            }
            if (!w.infeasible) {
                presolve_detail::reduce_rows(w);
            }
            // After the implied sides are dropped, so a redundant side is
            // removed rather than tightened.
            if (!w.infeasible && opts.coefficient_tightening) {
                presolve_detail::reduce_coef_tightening(w);
            }
            if (!w.infeasible && opts.empty_columns) {
                presolve_detail::reduce_empty_columns(w);
            }
            if (!w.changed) {
                break;
            }
        }
        if (!w.infeasible) {
            PresolveResult res = presolve_detail::build_result(w, lb0, ub0);
            res.stats.seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            return res;
        }
        rounds = w.stats.rounds;
    }
    PresolveResult res = presolve_detail::identity_result(problem, opts);
    res.stats.status = PresolveStatus::Infeasible;
    res.stats.rounds = rounds;
    res.stats.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
}

}  // namespace cbls
