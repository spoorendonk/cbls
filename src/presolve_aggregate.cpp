// Doubleton-equality aggregation (#328). See src/presolve_internal.h.
//
// A live row `a x + b y = c` (both sides equal and finite, two live entries)
// gives y = (c - a x)/b = d - k x, with k = a/b and d = c/b. Eliminating y:
//   - y's bounds become bounds on x: x = (c - b v)/a at each finite bound v;
//   - in each other row s holding y with coefficient e, x's coefficient gains
//     -e k, both sides lose e d, and y's entry goes;
//   - obj[x] gains -obj[y] k and the objective offset gains obj[y] d;
//   - the row and y go; postsolve recomputes y = (c - a x)/b from x, rounded
//     when y is integral and never clamped (clamping breaks the equality).
// Each aggregation removes y's k other entries, adds at most k for x and drops
// the row's two, so the total number of nonzeros only falls.
//
// Nothing changes until every check has passed, so an aggregation is all or
// nothing. Rows are scanned once per round in index order; a chain (x
// eliminated later through another row) is undone in reverse by postsolve.
//
// Cost: one aggregation walks each of y's other rows twice (x's slot looked
// up in `substitution_ok`, then in `add_entry`), so O(sum of their lengths),
// not O(nnz) per round: a dense row holding n eliminated columns costs O(n^2)
// per round. Sparse rows, the common case, stay linear; a column-to-slot map
// per row would buy the dense case back at 8 B per entry and its upkeep.

#include "presolve_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cbls::presolve_detail {

namespace {

constexpr double kEps = std::numeric_limits<double>::epsilon();

/// One way to eliminate a column of the row: y goes, x stays.
struct Candidate {
    int32_t y = -1;
    int32_t x = -1;
    double a = 0.0;  ///< x's coefficient in the row.
    double b = 0.0;  ///< y's coefficient in the row.
    double c = 0.0;  ///< The side; for an integral y, exactly q * b (see `eligible`).
    double k = 0.0;  ///< a / b as used in the substitution.
    double d = 0.0;  ///< c / b as used in the substitution.
};

/// Whether y can be eliminated through row r (`a x + b y = c`), and how.
///
/// An integral y needs y = d - k x integral at every integral x: x integral,
/// and k and d integers. k must be one exactly (`fma(round(k), b, -a) == 0`).
/// d may be one up to the side's error plus `feasibility_tolerance`: then
/// |q b - c| is within it, so taking c = q b moves the row by no more than
/// the tolerance (and a row with no integer point off by more is left alone).
/// With k and d integers the substitution and postsolve are exact.
bool eligible(const Work& w, std::size_t r, int32_t y, int32_t x, double a, double b, double c,
              Candidate& out) {
    const auto yy = static_cast<std::size_t>(y);
    const auto xx = static_cast<std::size_t>(x);
    if (w.frozen[yy] != 0 || w.frozen[xx] != 0) {
        return false;
    }
    double k = a / b;
    double d = c / b;
    if (!std::isfinite(k) || !std::isfinite(d) || std::abs(k) > w.opts.max_aggregation_ratio) {
        return false;
    }
    if (w.integral[yy] != 0) {
        const double kr = std::round(k);
        const double q = std::round(d);
        if (w.integral[xx] == 0 || std::fma(kr, b, -a) != 0.0 ||
            std::abs(std::fma(q, b, -c)) > w.opts.feasibility_tolerance + w.row_roundoff(r)) {
            return false;
        }
        k = kr;
        d = q;
        c = q * b;
    }
    out = Candidate{y, x, a, b, c, k, d};
    return true;
}

/// The better of two eligible candidates: a continuous y first (no
/// integrality to keep), then the y in fewer rows (less substitution and
/// fill), then the lower index.
bool better(const Work& w, const Candidate& p, const Candidate& q) {
    const auto py = static_cast<std::size_t>(p.y);
    const auto qy = static_cast<std::size_t>(q.y);
    if (w.integral[py] != w.integral[qy]) {
        return w.integral[py] == 0;
    }
    if (w.col_live[py] != w.col_live[qy]) {
        return w.col_live[py] < w.col_live[qy];
    }
    return p.y < q.y;
}

/// x's bounds implied by y's through the row, or false if one is unusable.
///
/// Each quotient x = (c - b v)/a is relaxed outward by its error bound: the
/// side's own error (`row_roundoff`), one rounding each of `b v` and of the
/// difference, all over |a|, plus one rounding of the division (the
/// singleton rule's `quotient_error`, with the numerator's roundings added).
/// A finite v whose quotient is not finite, or at or beyond `infinity`, would
/// read as "no bound" and leave y unbounded through x: the row is left alone.
bool transferred_bounds(const Work& w, std::size_t r, const Candidate& cand, double& lo,
                        double& hi) {
    const auto yy = static_cast<std::size_t>(cand.y);
    const double infinity = w.opts.propagation.infinity;
    const double side_error = w.row_roundoff(r);
    bool ok = true;
    const auto at = [&](double v, double outward) {
        if (!std::isfinite(v)) {
            return outward * std::numeric_limits<double>::infinity();
        }
        const double bv = cand.b * v;
        const double num = cand.c - bv;
        const double q = num / cand.a;
        const double err =
            ((side_error + (kEps * (std::abs(bv) + std::abs(num)))) / std::abs(cand.a)) +
            (kEps * std::abs(q));
        const double relaxed = q + (outward * err);
        if (!std::isfinite(relaxed) || std::abs(relaxed) >= infinity) {
            ok = false;
        }
        return relaxed;
    };
    // x = (c - b y)/a moves with y when -b/a > 0, i.e. a and b differ in sign.
    if ((cand.a > 0.0) != (cand.b > 0.0)) {
        lo = at(w.lb[yy], -1.0);
        hi = at(w.ub[yy], 1.0);
    } else {
        lo = at(w.ub[yy], -1.0);
        hi = at(w.lb[yy], 1.0);
    }
    return ok;
}

/// x's live coefficient in row s, or 0.
double coef_in_row(const Work& w, std::size_t s, int32_t x) {
    for (std::size_t p = w.row_begin[s]; p < w.row_end[s]; ++p) {
        if (w.cols[p] == x) {
            return w.coefs[p];
        }
    }
    return 0.0;
}

/// Whether row s's side `bound - shift` stays usable: a finite side must stay
/// finite and below `infinity`, where it would read as "no bound".
bool side_ok(double bound, double shift, double infinity) {
    if (!std::isfinite(bound)) {
        return true;
    }
    const double moved = bound - shift;
    return std::isfinite(moved) && std::abs(moved) < infinity;
}

/// Check every substitution of y into its other rows; on success `fill` is
/// the number of rows that gain an x entry.
bool substitution_ok(const Work& w, std::size_t r, const Candidate& cand, int& fill) {
    const auto yy = static_cast<std::size_t>(cand.y);
    const double infinity = w.opts.propagation.infinity;
    fill = 0;
    for (std::size_t q = w.col_begin[yy]; q < w.col_end[yy]; ++q) {
        const auto s = static_cast<std::size_t>(w.col_rows[q]);
        const double e = w.coefs[w.entry_pos(q)];
        if (s == r || w.row_removed[s] != 0 || e == 0.0) {
            continue;
        }
        const double delta = -e * cand.k;
        const double shift = e * cand.d;
        const double cur = coef_in_row(w, s, cand.x);
        if (!std::isfinite(delta) || !std::isfinite(cur + delta) || !std::isfinite(shift) ||
            !side_ok(w.row_lo[s], shift, infinity) || !side_ok(w.row_hi[s], shift, infinity)) {
            return false;
        }
        if (cur == 0.0 && delta != 0.0) {
            ++fill;
        }
    }
    return fill <= w.opts.max_fill;
}

/// Substitute y = d - k x into row s, where y has coefficient e at `p`.
///
/// The sides move by `e d`, accounted as a fold: one more fold, and the
/// scale gains `2 |shift|` (the product's and the difference's roundings,
/// plus d's own) and `|e/b| (row_folds[r] + 1) row_scale[r]`, which bounds
/// the eliminated row's side error carried over (`|e/b| row_roundoff(r)`;
/// s's fold count only grows, so it stays covered). x's new coefficient
/// `cur - e k` carries its rounding, which no margin models: exact for an
/// integral k and integral coefficients, the main target.
void substitute(Work& w, std::size_t r, const Candidate& cand, std::size_t s, std::size_t p) {
    const double e = w.coefs[p];
    const auto yy = static_cast<std::size_t>(cand.y);
    w.coefs[p] = 0.0;
    --w.row_live[s];
    --w.col_live[yy];
    const double shift = e * cand.d;
    if (std::isfinite(w.row_lo[s])) {
        w.row_lo[s] -= shift;
    }
    if (std::isfinite(w.row_hi[s])) {
        w.row_hi[s] -= shift;
    }
    w.row_scale[s] +=
        (2.0 * std::abs(shift)) +
        (std::abs(e / cand.b) * static_cast<double>(w.row_folds[r] + 1) * w.row_scale[r]);
    ++w.row_folds[s];
    const double delta = -e * cand.k;
    // Checked by `substitution_ok`: finite, nonzero, a finite sum. Only an
    // int32 row-length overflow is left, and that is not reachable here.
    if (delta != 0.0 && !w.add_entry(static_cast<int32_t>(s), cand.x, delta)) {
        throw std::logic_error("presolve aggregation: add_entry refused a checked entry");
    }
}

/// Eliminate cand.y through row r if every check passes; false if one fails
/// (nothing changed) or the transferred bounds proved infeasibility.
bool aggregate(Work& w, std::size_t r, const Candidate& cand) {
    const auto yy = static_cast<std::size_t>(cand.y);
    const auto xx = static_cast<std::size_t>(cand.x);
    double lo = 0.0;
    double hi = 0.0;
    int fill = 0;
    const double obj_x = w.obj[xx] - (w.obj[yy] * cand.k);
    const double offset = w.obj_offset + (w.obj[yy] * cand.d);
    if (!transferred_bounds(w, r, cand, lo, hi) || !substitution_ok(w, r, cand, fill) ||
        !std::isfinite(obj_x) || !std::isfinite(offset)) {
        return false;
    }
    // An x within `slack` of a transferred bound puts y within |k| slack <=
    // tol of its own, and x within tol of the bound the collapse widens: the
    // singleton rule with |k| in place of |a|.
    const double tol = w.opts.feasibility_tolerance;
    const double slack = std::min(tol, tol / std::abs(cand.k));
    if (w.integral[xx] != 0) {
        if (std::isfinite(lo)) {
            lo = std::ceil(lo - slack);
        }
        if (std::isfinite(hi)) {
            hi = std::floor(hi + slack);
        }
    }
    w.tighten_column(cand.x, lo, hi, slack);
    if (w.infeasible) {
        return false;
    }
    for (std::size_t q = w.col_begin[yy]; q < w.col_end[yy]; ++q) {
        const auto s = static_cast<std::size_t>(w.col_rows[q]);
        const std::size_t p = w.entry_pos(q);
        if (s != r && w.row_removed[s] == 0 && w.coefs[p] != 0.0) {
            substitute(w, r, cand, s, p);
        }
    }
    w.obj[xx] = obj_x;
    w.obj_offset = offset;
    w.obj[yy] = 0.0;
    w.remove_row(static_cast<int32_t>(r));
    w.col_removed[yy] = 1;
    w.col_live[yy] = 0;
    w.postsolve.push_aggregated(cand.y, cand.x, cand.a, cand.b, cand.c, w.integral[yy] != 0);
    ++w.stats.cols_aggregated;
    w.stats.aggregation_fill += fill;
    w.changed = true;
    return true;
}

}  // namespace

void reduce_doubleton_aggregation(Work& w) {
    for (std::size_t r = 0; r < w.n_rows() && !w.infeasible; ++r) {
        if (w.row_removed[r] != 0 || w.row_live[r] != 2 || w.row_lo[r] != w.row_hi[r] ||
            !std::isfinite(w.row_lo[r])) {
            continue;
        }
        std::array<std::size_t, 2> live = {0, 0};
        std::size_t n = 0;
        for (std::size_t p = w.row_begin[r]; p < w.row_end[r] && n < 2; ++p) {
            if (w.coefs[p] != 0.0) {
                live[n++] = p;
            }
        }
        const int32_t j0 = w.cols[live[0]];
        const int32_t j1 = w.cols[live[1]];
        const double a0 = w.coefs[live[0]];
        const double a1 = w.coefs[live[1]];
        const double c = w.row_lo[r];
        Candidate first;
        Candidate second;
        const bool ok0 = eligible(w, r, j0, j1, a1, a0, c, first);   // eliminate j0
        const bool ok1 = eligible(w, r, j1, j0, a0, a1, c, second);  // eliminate j1
        // The preferred orientation first; the other if its checks fail.
        const bool first_preferred = ok0 && (!ok1 || better(w, first, second));
        const Candidate& pick = first_preferred ? first : second;
        const Candidate& other = first_preferred ? second : first;
        if ((ok0 || ok1) && !aggregate(w, r, pick) && !w.infeasible && ok0 && ok1) {
            aggregate(w, r, other);
        }
    }
}

}  // namespace cbls::presolve_detail
