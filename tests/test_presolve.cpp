// Presolve and postsolve (#324, epic #322).

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <cbls/presolve.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Builder {
    LinearProblem p;

    int32_t col(double lb, double ub, bool integral = false, double obj = 0.0,
                bool frozen = false) {
        p.col_lb.push_back(lb);
        p.col_ub.push_back(ub);
        p.integral.push_back(integral ? 1 : 0);
        p.frozen.push_back(frozen ? 1 : 0);
        p.obj.push_back(obj);
        return static_cast<int32_t>(p.col_lb.size() - 1);
    }

    void row(const std::vector<std::pair<int32_t, double>>& terms, double lo, double hi) {
        for (const auto& [c, a] : terms) {
            p.cols.push_back(c);
            p.coefs.push_back(a);
        }
        p.row_start.push_back(p.cols.size());
        p.row_lo.push_back(lo);
        p.row_hi.push_back(hi);
    }
};

PresolveOptions only(bool propagate) {
    PresolveOptions o;
    o.enabled = true;
    o.propagate_bounds = propagate;
    return o;
}

double tol_of(double scale) {
    return 1e-9 * std::max(1.0, std::abs(scale));
}

/// x satisfies p's bounds, integrality and rows within 1e-9 (scaled).
bool satisfies(const LinearProblem& p, const std::vector<double>& x) {
    if (x.size() != p.n_cols()) {
        return false;
    }
    for (std::size_t j = 0; j < p.n_cols(); ++j) {
        if (!std::isfinite(x[j]) || x[j] < p.col_lb[j] - tol_of(x[j]) ||
            x[j] > p.col_ub[j] + tol_of(x[j])) {
            return false;
        }
        if (p.integral[j] != 0 && std::abs(x[j] - std::round(x[j])) > 1e-9) {
            return false;
        }
    }
    for (std::size_t i = 0; i < p.n_rows(); ++i) {
        double act = 0.0;
        double mag = 0.0;
        for (std::size_t k = p.row_start[i]; k < p.row_start[i + 1]; ++k) {
            const double t = p.coefs[k] * x[static_cast<std::size_t>(p.cols[k])];
            act += t;
            mag += std::abs(t);
        }
        const double lo = p.row_lo[i];
        const double hi = p.row_hi[i];
        const double scale = mag + (std::isfinite(lo) ? std::abs(lo) : 0.0) +
                             (std::isfinite(hi) ? std::abs(hi) : 0.0);
        if (act < lo - tol_of(scale) || act > hi + tol_of(scale)) {
            return false;
        }
    }
    return true;
}

double objective(const LinearProblem& p, const std::vector<double>& x) {
    double v = p.obj_offset;
    for (std::size_t j = 0; j < p.n_cols(); ++j) {
        v += p.obj[j] * x[j];
    }
    return v;
}

/// Values for `x + a*z1 - a*z2 <= c` (a = 0.1, z1 = v1, z2 = v2 fixed) where
/// the exact bound left on x, `c - a*v1 + a*v2`, is at least 3, but folding
/// z1 then z2 in floating point leaves less than 3 - 1e-8: the products
/// round to the same double while their exact values differ by 0.1 ulp(v).
struct FoldCut {
    double a = 0.1;
    double v1 = 0.0;
    double v2 = 0.0;
    double c = 0.0;
};

FoldCut find_fold_cut() {
    FoldCut f;
    f.c = 3.0 - 1.1e-8;
    for (int k = 0; k < 1000; ++k) {
        const double v = 1e9 + static_cast<double>(k);
        const double v2 = std::nextafter(v, kInf);
        const double s1 = f.a * v;
        const double s2 = f.a * v2;
        if (s1 != s2) {
            continue;
        }
        const double e1 = std::fma(f.a, v, -s1);  // exact product errors
        const double e2 = std::fma(f.a, v2, -s2);
        const double exact_gap = (f.c - 3.0) + (e2 - e1);  // exact bound - 3
        const double folded = (f.c - s1) + s2;             // presolve's fold order
        if (exact_gap >= 0.0 && folded < 3.0 - 1e-8) {
            f.v1 = v;
            f.v2 = v2;
            return f;
        }
    }
    FAIL("no fold-cut instance in the search range");
    return f;
}

std::vector<double> restrict_to(const PresolveResult& r, const std::vector<double>& x) {
    std::vector<double> out;
    out.reserve(r.reduced_to_col.size());
    for (const int32_t j : r.reduced_to_col) {
        out.push_back(x[static_cast<std::size_t>(j)]);
    }
    return out;
}

}  // namespace

TEST_CASE("presolve disabled returns the merged problem with an identity postsolve", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1);
    const int32_t y = b.col(5, 5);
    b.row({{x, 1.0}, {y, 1.0}, {x, 2.0}, {y, 0.0}}, -kInf, 10.0);
    const PresolveResult r = presolve(b.p);
    REQUIRE(r.stats.status == PresolveStatus::Unchanged);
    REQUIRE(r.stats.duplicates_merged == 2);
    REQUIRE(r.reduced.n_cols() == 2);
    REQUIRE(r.reduced.n_rows() == 1);
    // x's two entries merged to 3; y's (1 + 0) kept as 1.
    REQUIRE(r.reduced.cols == std::vector<int32_t>{0, 1});
    REQUIRE(r.reduced.coefs == std::vector<double>{3.0, 1.0});
    REQUIRE(r.postsolve.num_steps() == 0);
    REQUIRE(r.postsolve.apply({0.5, 5.0}) == std::vector<double>{0.5, 5.0});

    // Repeats are summed compensated, whatever their order: 1e16 + 1 - 1e16
    // is 1, not 0 (which would drop x from the row).
    Builder c;
    const int32_t u = c.col(0, 100);
    c.row({{u, 1e16}, {u, 1.0}, {u, -1e16}}, -kInf, 5.0);
    const PresolveResult rc = presolve(c.p);
    REQUIRE(rc.reduced.coefs == std::vector<double>{1.0});
}

TEST_CASE("presolve folds a fixed column into its rows and the objective", "[presolve]") {
    Builder b;
    const int32_t x = b.col(2, 2, false, 3.0);
    const int32_t y = b.col(0, 10, false, 1.0);
    const int32_t z = b.col(0, 10, false, 1.0);
    b.row({{x, 1.0}, {y, 1.0}, {z, 1.0}}, 1.0, 5.0);
    PresolveOptions o = only(false);
    o.redundant_rows = false;  // y + z >= -1 is implied; keep it to read it back
    const PresolveResult r = presolve(b.p, o);
    REQUIRE(r.stats.status == PresolveStatus::Reduced);
    REQUIRE(r.stats.cols_fixed == 1);
    REQUIRE(r.col_to_reduced[static_cast<std::size_t>(x)] == -1);
    REQUIRE(r.reduced.n_cols() == 2);
    REQUIRE(r.reduced.row_lo[0] == -1.0);
    REQUIRE(r.reduced.row_hi[0] == 3.0);
    REQUIRE(r.reduced.obj_offset == 6.0);
    REQUIRE(r.postsolve.apply({1.0, 2.0}) == std::vector<double>{2.0, 1.0, 2.0});

    // A fold that would move a row bound past the infinity threshold (here
    // 100 * -1e19 leaves y <= 1e21, read as no bound) keeps the column.
    Builder big;
    const int32_t bx = big.col(-1e19, -1e19);
    const int32_t by = big.col(0, kInf);
    const int32_t bz = big.col(0, kInf);
    big.row({{bx, 100.0}, {by, 1.0}, {bz, 1.0}}, -kInf, 0.0);
    const PresolveResult rb = presolve(big.p, o);
    REQUIRE(rb.stats.cols_fixed == 0);
    REQUIRE(rb.reduced.n_cols() == 3);
    REQUIRE(rb.reduced.row_hi[0] == 0.0);
}

TEST_CASE("presolve keeps a frozen column even when it is fixed", "[presolve]") {
    Builder b;
    const int32_t x = b.col(2, 2, false, 0.0, /*frozen=*/true);
    const int32_t y = b.col(0, 10);
    const int32_t z = b.col(0, 10);
    b.row({{x, 1.0}, {y, 1.0}, {z, 1.0}}, 1.0, 5.0);
    const PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.stats.cols_fixed == 0);
    REQUIRE(r.reduced.n_cols() == 3);
    REQUIRE(r.col_to_reduced[static_cast<std::size_t>(x)] == 0);

    // A singleton row on a frozen column tightens its bound and goes; the
    // column stays.
    Builder c;
    const int32_t f = c.col(0, 10, false, 0.0, /*frozen=*/true);
    c.row({{f, 2.0}}, -kInf, 6.0);
    const PresolveResult rc = presolve(c.p, only(false));
    REQUIRE(rc.stats.rows_singleton == 1);
    REQUIRE(rc.reduced.n_rows() == 0);
    REQUIRE(rc.reduced.n_cols() == 1);
    REQUIRE(rc.reduced.col_ub[0] == 3.0);
}

TEST_CASE("presolve removes a satisfied empty row and rejects a violated one", "[presolve]") {
    Builder b;
    b.col(0, 1);
    b.row({}, -1.0, 1.0);
    PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.stats.rows_empty == 1);
    REQUIRE(r.reduced.n_rows() == 0);
    REQUIRE(r.row_to_reduced[0] == -1);

    Builder c;
    c.col(0, 1);
    c.row({}, 1.0, 2.0);
    r = presolve(c.p, only(false));
    REQUIRE(r.stats.status == PresolveStatus::Infeasible);
    // The original problem comes back with an identity postsolve.
    REQUIRE(r.reduced.n_rows() == 1);
    REQUIRE(r.reduced.n_cols() == 1);
    REQUIRE(r.postsolve.num_steps() == 0);
    REQUIRE(r.stats.rows_empty == 0);  // nothing counted from the abandoned run
    REQUIRE(r.stats.rounds == 1);

    // At large magnitude the slack stays absolute: a residual of 0.05 left by
    // folding 1e4 * 5000 twice is a violation, not rounding, while an exact
    // fold leaves the row satisfied.
    for (const double rhs : {1e8, 1e8 + 0.05}) {
        CAPTURE(rhs);
        Builder d;
        const int32_t x = d.col(5000, 5000, true);
        const int32_t y = d.col(5000, 5000, true);
        d.row({{x, 1e4}, {y, 1e4}}, rhs, rhs);
        r = presolve(d.p, only(false));
        REQUIRE(r.stats.status ==
                (rhs == 1e8 ? PresolveStatus::Reduced : PresolveStatus::Infeasible));
    }
}

TEST_CASE("presolve removes a row with no finite side", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1);
    const int32_t y = b.col(0, 1);
    b.row({{x, 1.0}, {y, 1.0}}, -1e20, 1e30);
    const PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.stats.rows_free == 1);
    REQUIRE(r.reduced.n_rows() == 0);
}

TEST_CASE("presolve turns singleton rows into bounds", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 10, /*integral=*/true);
    const int32_t y = b.col(-5, 5);
    const int32_t z = b.col(0, 10);
    b.row({{x, 2.0}}, -kInf, 7.0);   // x <= 3.5 -> 3 (integral)
    b.row({{y, -2.0}}, -kInf, 3.0);  // y >= -1.5
    b.row({{x, 1.0}, {y, 1.0}, {z, 1.0}}, -kInf, 100.0);
    PresolveOptions o = only(false);
    o.redundant_rows = false;
    const PresolveResult r = presolve(b.p, o);
    REQUIRE(r.stats.rows_singleton == 2);
    REQUIRE(r.reduced.n_rows() == 1);
    REQUIRE(r.reduced.col_ub[static_cast<std::size_t>(x)] == 3.0);
    REQUIRE(r.reduced.col_lb[static_cast<std::size_t>(y)] == -1.5);
    REQUIRE(r.stats.bounds_tightened == 2);

    // 3x <= 3e9 + 1 over the integers is x <= 1e9: the quotient's fraction
    // (1/3) is rounded off, not absorbed by a slack scaled to 1e9.
    Builder big;
    const int32_t u = big.col(0, 4e9, /*integral=*/true);
    big.row({{u, 3.0}}, -kInf, 3e9 + 1);
    PresolveOptions keep_empty = only(false);
    keep_empty.empty_columns = false;  // u ends in no row; keep it to read it
    const PresolveResult rb = presolve(big.p, keep_empty);
    REQUIRE(rb.reduced.n_cols() == 1);
    REQUIRE(rb.reduced.col_ub[0] == 1e9);

    // A quotient at or beyond the infinity threshold would read as "no bound"
    // downstream: the row stays, and the column keeps its bounds.
    Builder huge;
    const int32_t h = huge.col(0, kInf);
    huge.row({{h, 1e-6}}, -kInf, 1e15);
    const PresolveResult rh = presolve(huge.p, only(false));
    REQUIRE(rh.stats.rows_singleton == 0);
    REQUIRE(rh.reduced.n_rows() == 1);
    REQUIRE(std::isinf(rh.reduced.col_ub[0]));

    // So is a quotient that overflowed: -1e19 / 1e-300 is -inf.
    Builder over;
    const int32_t ov = over.col(-kInf, kInf);
    over.row({{ov, 1e-300}}, -kInf, -1e19);
    const PresolveResult ro = presolve(over.p, only(false));
    REQUIRE(ro.stats.rows_singleton == 0);
    REQUIRE(ro.reduced.n_rows() == 1);

    // A quotient just under the threshold that its relaxation carries past it
    // keeps the row too; the bound never comes out as "no bound".
    Builder edge;
    const int32_t ev = edge.col(0, kInf);
    edge.row({{ev, 0.3}}, -kInf, std::nextafter(1e20, 0.0) * 0.3);
    const PresolveResult re = presolve(edge.p, keep_empty);
    REQUIRE(re.reduced.n_cols() == 1);
    REQUIRE((re.reduced.n_rows() == 1 || re.reduced.col_ub[0] < 1e20));

    // A singleton left by folds keeps their roundoff: x = 3 satisfies the
    // exact row though the folded bound is below 3.
    const FoldCut f = find_fold_cut();
    Builder fold;
    const int32_t fx = fold.col(0, 10, /*integral=*/true);
    const int32_t z1 = fold.col(f.v1, f.v1);
    const int32_t z2 = fold.col(f.v2, f.v2);
    fold.row({{fx, 1.0}, {z1, f.a}, {z2, -f.a}}, -kInf, f.c);
    const PresolveResult rf = presolve(fold.p, keep_empty);
    REQUIRE(rf.stats.rows_singleton == 1);
    REQUIRE(rf.reduced.col_ub[static_cast<std::size_t>(rf.col_to_reduced[0])] == 3.0);
}

TEST_CASE("presolve reports a singleton row outside the column box as infeasible", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1);
    b.row({{x, 1.0}}, 3.0, kInf);
    PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.stats.status == PresolveStatus::Infeasible);
    REQUIRE(r.reduced.n_rows() == 1);

    // Crossing by 0.05 at 1e8 is infeasible too: the slack is absolute.
    Builder c;
    const int32_t y = c.col(0, 1e8);
    c.row({{y, 1.0}}, 1e8 + 0.05, kInf);
    r = presolve(c.p, only(false));
    REQUIRE(r.stats.status == PresolveStatus::Infeasible);

    // A computed quotient one ulp above the column's ub is within its own
    // rounding error: the row goes and the column keeps a nonempty box.
    const double q = 1.0 / 3.0;
    Builder d;
    const int32_t z = d.col(0, std::nextafter(q, 0.0));
    d.row({{z, 3.0}}, 1.0, kInf);
    r = presolve(d.p, only(false));
    REQUIRE(r.stats.status == PresolveStatus::Reduced);
    REQUIRE(r.reduced.n_rows() == 0);
}

TEST_CASE("presolve drops implied row sides and removes a fully implied row", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1);
    const int32_t y = b.col(0, 1);
    b.row({{x, 1.0}, {y, 1.0}}, -kInf, 5.0);  // always holds: removed
    b.row({{x, 1.0}, {y, 1.0}}, 1.0, 5.0);    // upper side implied: dropped
    b.row({{x, 1.0}, {y, 1.0}}, 0.5, 2.0);    // max activity 2 == hi: kept
    const PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.stats.rows_redundant == 1);
    REQUIRE(r.row_to_reduced[0] == -1);
    REQUIRE(r.reduced.n_rows() == 2);
    REQUIRE(r.reduced.row_lo[0] == 1.0);
    REQUIRE(std::isinf(r.reduced.row_hi[0]));
    // A side the activity only meets, not clears by its rounding bound, stays:
    // dropping it would rest on the last ulp of the activity sum.
    REQUIRE(r.reduced.row_lo[1] == 0.5);
    REQUIRE(r.reduced.row_hi[1] == 2.0);
    REQUIRE(r.stats.row_sides_dropped == 2);
}

TEST_CASE("presolve fixes an empty column at its best bound", "[presolve]") {
    Builder b;
    const int32_t up = b.col(-2, 4, false, 1.0);       // c > 0: lb
    const int32_t down = b.col(-2, 4, false, -1.0);    // c < 0: ub
    const int32_t zero = b.col(1, 4, false, 0.0);      // c = 0: closest to 0
    const int32_t open = b.col(-kInf, 4, false, 1.0);  // unbounded improving: kept
    const int32_t frozen = b.col(-2, 4, false, 1.0, true);
    const PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.stats.cols_empty == 3);
    REQUIRE(r.reduced_to_col == std::vector<int32_t>{open, frozen});
    REQUIRE(r.reduced.obj_offset == -2.0 - 4.0);
    const std::vector<double> x = r.postsolve.apply({-7.0, 0.0});
    REQUIRE(x[static_cast<std::size_t>(up)] == -2.0);
    REQUIRE(x[static_cast<std::size_t>(down)] == 4.0);
    REQUIRE(x[static_cast<std::size_t>(zero)] == 1.0);
    REQUIRE(x[static_cast<std::size_t>(open)] == -7.0);
}

TEST_CASE("presolve rounds integral bounds inward and rejects an empty integral box",
          "[presolve]") {
    Builder b;
    const int32_t x = b.col(0.5, 2.7, true);
    const int32_t y = b.col(0, 10);
    b.row({{x, 1.0}, {y, 1.0}}, 1.0, 50.0);
    PresolveResult r = presolve(b.p, only(false));
    REQUIRE(r.reduced.col_lb[0] == 1.0);
    REQUIRE(r.reduced.col_ub[0] == 2.0);

    // At 1e9 a half is still a half: the bound rounds down, not up.
    Builder big;
    big.col(0, 1e9 + 0.5, true);
    PresolveOptions keep_empty = only(false);
    keep_empty.empty_columns = false;  // the column is in no row; keep it to read it
    r = presolve(big.p, keep_empty);
    REQUIRE(r.reduced.n_cols() == 1);
    REQUIRE(r.reduced.col_ub[0] == 1e9);

    Builder c;
    c.col(0.2, 0.8, true);
    r = presolve(c.p, only(false));
    REQUIRE(r.stats.status == PresolveStatus::Infeasible);
}

TEST_CASE("presolve with propagation fixes and removes a forced column", "[presolve]") {
    // x + y = 2 over binaries forces both to 1; everything then goes.
    Builder b;
    const int32_t x = b.col(0, 1, true, 1.0);
    const int32_t y = b.col(0, 1, true, 1.0);
    b.row({{x, 1.0}, {y, 1.0}}, 2.0, 2.0);
    const PresolveResult r = presolve(b.p, only(true));
    REQUIRE(r.stats.status == PresolveStatus::Reduced);
    REQUIRE(r.reduced.n_cols() == 0);
    REQUIRE(r.reduced.n_rows() == 0);
    REQUIRE(r.reduced.obj_offset == 2.0);
    REQUIRE(r.postsolve.apply({}) == std::vector<double>{1.0, 1.0});

    // Propagation over a folded row keeps the folds' roundoff: x = 3, w = 0
    // satisfies the exact row though the folded bound is below 3.
    const FoldCut f = find_fold_cut();
    Builder fold;
    const int32_t fx = fold.col(0, 10, /*integral=*/true);
    const int32_t fw = fold.col(0, 10, /*integral=*/true);
    const int32_t z1 = fold.col(f.v1, f.v1);
    const int32_t z2 = fold.col(f.v2, f.v2);
    fold.row({{fx, 1.0}, {fw, 1.0}, {z1, f.a}, {z2, -f.a}}, -kInf, f.c);
    const PresolveResult rf = presolve(fold.p, only(true));
    REQUIRE(rf.stats.status != PresolveStatus::Infeasible);
    REQUIRE(rf.reduced.col_ub[static_cast<std::size_t>(rf.col_to_reduced[0])] == 3.0);
}

TEST_CASE("presolve rejects a malformed problem", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1);
    b.row({{x, 1.0}}, 0.0, 1.0);
    LinearProblem bad = b.p;
    bad.obj.pop_back();
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.cols[0] = 4;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.col_ub[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.row_start.back() = 0;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.coefs[0] = kInf;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.obj[0] = -kInf;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    // An empty side, not a missing one: lb at +infinity, ub at -infinity.
    bad = b.p;
    bad.col_lb[0] = kInf;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.col_ub[0] = -1e20;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);
    bad = b.p;
    bad.row_lo[0] = 1e30;
    REQUIRE_THROWS_AS(presolve(bad), std::invalid_argument);

    const PresolveResult r = presolve(b.p, only(true));
    REQUIRE_THROWS_AS(r.postsolve.apply({1.0, 2.0}), std::invalid_argument);
}

TEST_CASE("presolve handles a problem with no rows or no columns", "[presolve]") {
    const PresolveResult empty = presolve(LinearProblem{}, only(true));
    REQUIRE(empty.reduced.n_cols() == 0);
    REQUIRE(empty.stats.status == PresolveStatus::Unchanged);

    Builder b;
    b.col(-1, 1, false, 0.0);
    const PresolveResult r = presolve(b.p, only(true));
    REQUIRE(r.reduced.n_cols() == 0);
    REQUIRE(r.postsolve.apply({}) == std::vector<double>{0.0});
}

namespace {

/// A random problem around a point it is built to contain. In `boxed` mode
/// every column is integral with a box of at most 5 values, so the reduced
/// problem's integer points can be enumerated.
struct Instance {
    LinearProblem p;
    std::vector<double> x_star;
};

double pick_slack(std::mt19937& rng) {
    const int s = std::uniform_int_distribution<int>(0, 4)(rng);
    return s == 4 ? kInf : static_cast<double>(s % 3);
}

void add_random_row(std::mt19937& rng, Builder& b, const std::vector<double>& x) {
    const int n = static_cast<int>(x.size());
    // Mostly rows of two or more terms: singletons are tested by hand above,
    // and a generator dominated by them leaves nothing for the other
    // reductions to get wrong.
    const int len = std::uniform_int_distribution<int>(std::min(n, 2), std::min(n, 4))(rng);
    std::vector<std::pair<int32_t, double>> terms;
    double act = 0.0;
    for (int k = 0; k < len; ++k) {
        const int32_t c = std::uniform_int_distribution<int32_t>(0, n - 1)(rng);
        int a = std::uniform_int_distribution<int>(-3, 2)(rng);
        a = a >= 0 ? a + 1 : a;  // in {-3..-1, 1..3}
        terms.emplace_back(c, static_cast<double>(a));
        act += a * x[static_cast<std::size_t>(c)];
    }
    switch (std::uniform_int_distribution<int>(0, 4)(rng)) {
        case 0:
            b.row(terms, act, act);
            break;
        case 1:
            b.row(terms, -kInf, act + pick_slack(rng));
            break;
        case 2:
            b.row(terms, act - pick_slack(rng), kInf);
            break;
        case 3:
            b.row(terms, act - pick_slack(rng), act + pick_slack(rng));
            break;
        default:
            b.row(terms, -kInf, kInf);
            break;
    }
}

Instance random_instance(uint32_t seed, bool boxed) {
    std::mt19937 rng(seed);
    Builder b;
    std::vector<double> x;
    const int n = std::uniform_int_distribution<int>(1, boxed ? 5 : 6)(rng);
    for (int j = 0; j < n; ++j) {
        const bool integral = boxed || std::uniform_int_distribution<int>(0, 1)(rng) == 1;
        const double v = integral ? std::uniform_int_distribution<int>(-3, 3)(rng)
                                  : 0.25 * std::uniform_int_distribution<int>(-12, 12)(rng);
        const double below =
            boxed ? std::uniform_int_distribution<int>(0, 1)(rng) : pick_slack(rng);
        const double above =
            boxed ? std::uniform_int_distribution<int>(1, 2)(rng) : pick_slack(rng);
        const double obj = std::uniform_int_distribution<int>(-2, 2)(rng);
        const bool frozen = std::uniform_int_distribution<int>(0, 7)(rng) == 0;
        b.col(v - below, v + above, integral, obj, frozen);
        x.push_back(v);
    }
    const int m = std::uniform_int_distribution<int>(0, 5)(rng);
    for (int i = 0; i < m; ++i) {
        add_random_row(rng, b, x);
    }
    return {b.p, x};
}

/// Every integer point of `p`'s box that satisfies `p`, if the box is finite.
template <typename F>
void for_each_point(const LinearProblem& p, F&& f) {
    const std::size_t n = p.n_cols();
    std::vector<double> x(n);
    for (std::size_t j = 0; j < n; ++j) {
        if (!std::isfinite(p.col_lb[j]) || !std::isfinite(p.col_ub[j])) {
            return;
        }
        x[j] = p.col_lb[j];
    }
    while (true) {
        if (satisfies(p, x)) {
            f(x);
        }
        std::size_t j = 0;
        for (; j < n; ++j) {
            if (x[j] + 1.0 <= p.col_ub[j]) {
                x[j] += 1.0;
                break;
            }
            x[j] = p.col_lb[j];
        }
        if (j == n) {
            return;
        }
    }
}

}  // namespace

TEST_CASE("presolve removes no feasible point and postsolve restores it", "[presolve]") {
    // With empty columns off every reduction is primal: the known point must
    // survive into the reduced problem and come back unchanged.
    PresolveOptions o = only(true);
    o.empty_columns = false;
    int n_tightened = 0;  // coefficient tightening (#327) is on by default
    for (uint32_t seed = 1; seed <= 1000; ++seed) {
        CAPTURE(seed);
        const Instance inst = random_instance(seed, seed % 2 == 0);
        REQUIRE(satisfies(inst.p, inst.x_star));
        const PresolveResult r = presolve(inst.p, o);
        REQUIRE(r.stats.status != PresolveStatus::Infeasible);
        n_tightened += r.stats.coefs_tightened;
        const std::vector<double> reduced_x = restrict_to(r, inst.x_star);
        REQUIRE(satisfies(r.reduced, reduced_x));
        const std::vector<double> back = r.postsolve.apply(reduced_x);
        for (std::size_t j = 0; j < back.size(); ++j) {
            CAPTURE(j);
            if (inst.p.integral[j] != 0) {
                REQUIRE(back[j] == inst.x_star[j]);  // exact on integral columns
            } else {
                REQUIRE_THAT(back[j],
                             Catch::Matchers::WithinAbs(inst.x_star[j], tol_of(inst.x_star[j])));
            }
        }
        REQUIRE_THAT(objective(r.reduced, reduced_x),
                     Catch::Matchers::WithinAbs(objective(inst.p, inst.x_star), 1e-9));
        for (std::size_t j = 0; j < r.reduced_to_col.size(); ++j) {
            if (inst.p.frozen[static_cast<std::size_t>(r.reduced_to_col[j])] != 0) {
                REQUIRE(r.reduced.frozen[j] != 0);
            }
        }
        for (std::size_t j = 0; j < inst.p.n_cols(); ++j) {
            if (inst.p.frozen[j] != 0) {
                REQUIRE(r.col_to_reduced[j] >= 0);
            }
        }
    }
    REQUIRE(n_tightened > 0);
}

TEST_CASE("every point of the reduced problem postsolves to a feasible one", "[presolve]") {
    // Empty columns on: the dual reduction keeps feasibility and does not
    // worsen the objective of the point it maps.
    const PresolveOptions o = only(true);
    int n_tightened = 0;
    for (uint32_t seed = 1; seed <= 1000; ++seed) {
        CAPTURE(seed);
        const Instance inst = random_instance(seed, /*boxed=*/true);
        const PresolveResult r = presolve(inst.p, o);
        REQUIRE(r.stats.status != PresolveStatus::Infeasible);
        n_tightened += r.stats.coefs_tightened;
        const std::vector<double> mapped = r.postsolve.apply(restrict_to(r, inst.x_star));
        REQUIRE(satisfies(inst.p, mapped));
        REQUIRE(objective(inst.p, mapped) <= objective(inst.p, inst.x_star) + 1e-9);
        int n_points = 0;
        for_each_point(r.reduced, [&](const std::vector<double>& y) {
            ++n_points;
            const std::vector<double> back = r.postsolve.apply(y);
            REQUIRE(satisfies(inst.p, back));
            REQUIRE_THAT(objective(inst.p, back),
                         Catch::Matchers::WithinAbs(objective(r.reduced, y), 1e-9));
        });
        REQUIRE(n_points >= 1);
    }
    REQUIRE(n_tightened > 0);
}

TEST_CASE("a postsolved point passes verify_model on the unpresolved MPS model", "[presolve]") {
    // x in [0, 10] integer, y >= 0 continuous, z fixed at 1.
    //   r0: x + y + z >= 2     r1: 2x <= 7 (singleton)     r2: x + z <= 100 (implied)
    MpsProblem mps;
    mps.vars = {{"x", 0.0, 10.0, MpsVarKind::Integer},
                {"y", 0.0, kMpsInf, MpsVarKind::Continuous},
                {"z", 1.0, 1.0, MpsVarKind::Binary}};
    mps.rows = {{"r0", MpsRowSense::G, 2.0, 0.0},
                {"r1", MpsRowSense::L, 7.0, 0.0},
                {"r2", MpsRowSense::L, 100.0, 0.0}};
    mps.nonzeros = {{-1, 0, 1.0}, {-1, 1, 2.0}, {0, 0, 1.0}, {0, 1, 1.0},
                    {0, 2, 1.0},  {1, 0, 2.0},  {2, 0, 1.0}, {2, 2, 1.0}};

    Builder b;
    b.col(0.0, 10.0, true, 1.0);
    b.col(0.0, kInf, false, 2.0);
    b.col(1.0, 1.0, true, 0.0);
    b.row({{0, 1.0}, {1, 1.0}, {2, 1.0}}, 2.0, kInf);
    b.row({{0, 2.0}}, -kInf, 7.0);
    b.row({{0, 1.0}, {2, 1.0}}, -kInf, 100.0);
    const PresolveResult r = presolve(b.p, only(true));
    REQUIRE(r.stats.status == PresolveStatus::Reduced);
    REQUIRE(r.col_to_reduced[2] == -1);  // z fixed
    REQUIRE(r.reduced.col_ub[static_cast<std::size_t>(r.col_to_reduced[0])] == 3.0);

    // The reduced problem's best point: x = 1, y = 0 (x + y >= 1).
    std::vector<double> y(r.reduced.n_cols(), 0.0);
    y[static_cast<std::size_t>(r.col_to_reduced[0])] = 1.0;
    REQUIRE(satisfies(r.reduced, y));
    const std::vector<double> x = r.postsolve.apply(y);

    MpsToModelResult built = mps_to_model(mps);
    for (std::size_t j = 0; j < x.size(); ++j) {
        built.model.var_mut(vid(built.var_handles[j])).value = x[j];
    }
    full_evaluate(built.model);
    REQUIRE(verify_model(built.model).ok);
}
