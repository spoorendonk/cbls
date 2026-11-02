// Coefficient tightening (#327): Achterberg et al. 2020, eq. (3.6), and its
// a < 0 mirror. Hand cases pin the formula; enumeration checks that the
// tightened row admits exactly the original row's integer points.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/presolve.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <utility>
#include <vector>

using namespace cbls;
using Catch::Matchers::WithinAbs;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct Builder {
    LinearProblem p;

    int32_t col(double lb, double ub, bool integral, bool frozen = false) {
        p.col_lb.push_back(lb);
        p.col_ub.push_back(ub);
        p.integral.push_back(integral ? 1 : 0);
        p.frozen.push_back(frozen ? 1 : 0);
        p.obj.push_back(0.0);
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

/// Coefficient tightening alone: every other reduction off, so the reduced
/// problem keeps every column and every row with a side.
PresolveOptions coef_only() {
    PresolveOptions o;
    o.enabled = true;
    o.propagate_bounds = false;
    o.fixed_columns = false;
    o.singleton_rows = false;
    o.redundant_rows = false;
    o.empty_columns = false;
    return o;
}

constexpr double kTol = 1e-12;

}  // namespace

TEST_CASE("coefficient tightening applies Savelsbergh's step to a binary", "[presolve]") {
    // 2x + y <= 2, x binary, y in [0, 1]: M = 3, d = 2 - 3 + 2 = 1,
    // giving x + y <= 1.
    Builder b;
    b.col(0, 1, true);
    b.col(0, 1, false);
    b.row({{0, 2.0}, {1, 1.0}}, -kInf, 2.0);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.status == PresolveStatus::Reduced);
    REQUIRE(r.stats.coefs_tightened == 1);
    REQUIRE(r.stats.rows_coef_tightened == 1);
    REQUIRE_THAT(r.reduced.coefs[0], WithinAbs(1.0, kTol));
    REQUIRE(r.reduced.coefs[1] == 1.0);  // continuous: untouched
    REQUIRE_THAT(r.reduced.row_hi[0], WithinAbs(1.0, kTol));
    REQUIRE(r.reduced.row_lo[0] == -kInf);
}

TEST_CASE("coefficient tightening follows eq. (3.6) on a general integer", "[presolve]") {
    // 5x + y <= 15.5, x in {0..3}, y in [0, 1]: M = 16, d = 15.5 - 16 + 5 =
    // 4.5, giving 0.5x + y <= 15.5 - 4.5 * 3 = 2.
    Builder b;
    b.col(0, 3, true);
    b.col(0, 1, false);
    b.row({{0, 5.0}, {1, 1.0}}, -kInf, 15.5);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 1);
    REQUIRE_THAT(r.reduced.coefs[0], WithinAbs(0.5, kTol));
    REQUIRE_THAT(r.reduced.row_hi[0], WithinAbs(2.0, kTol));
}

TEST_CASE("coefficient tightening mirrors (3.6) for a negative coefficient", "[presolve]") {
    // The case above with x -> -x: -5x + y <= 15.5, x in {-3..0}. The mirror
    // gives (a + d) x <= b + d l: -0.5x + y <= 15.5 + 4.5 * (-3) = 2.
    Builder b;
    b.col(-3, 0, true);
    b.col(0, 1, false);
    b.row({{0, -5.0}, {1, 1.0}}, -kInf, 15.5);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 1);
    REQUIRE_THAT(r.reduced.coefs[0], WithinAbs(-0.5, kTol));
    REQUIRE_THAT(r.reduced.row_hi[0], WithinAbs(2.0, kTol));
}

TEST_CASE("coefficient tightening negates a >= row and back", "[presolve]") {
    // -5x - y >= -15.5 is the general-integer case negated.
    Builder b;
    b.col(0, 3, true);
    b.col(0, 1, false);
    b.row({{0, -5.0}, {1, -1.0}}, -15.5, kInf);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 1);
    REQUIRE_THAT(r.reduced.coefs[0], WithinAbs(-0.5, kTol));
    REQUIRE(r.reduced.coefs[1] == -1.0);
    REQUIRE_THAT(r.reduced.row_lo[0], WithinAbs(-2.0, kTol));
    REQUIRE(r.reduced.row_hi[0] == kInf);
}

TEST_CASE("coefficient tightening leaves rows it does not cover untouched", "[presolve]") {
    // Each row would tighten x's coefficient as `5x + y <= 15.5` does, but:
    Builder b;
    b.col(0, 3, true);                                   // x
    b.col(0, 1, false);                                  // y
    b.col(0, kInf, false);                               // z: unbounded above
    b.col(0, 3, false);                                  // c: continuous
    b.row({{0, 5.0}, {1, 1.0}}, 15.5, 15.5);             // equality
    b.row({{0, 5.0}, {1, 1.0}}, 1.0, 15.5);              // two-sided
    b.row({{0, 5.0}, {1, 1.0}, {2, 1.0}}, -kInf, 15.5);  // M infinite
    b.row({{1, 1.0}, {3, 5.0}}, -kInf, 15.5);            // continuous column
    b.row({{0, 5.0}, {1, 1.0}}, -kInf, 16.0);            // side implied (d = |a|)
    b.row({{0, 5.0}, {1, 1.0}}, -kInf, 11.0);            // d = 0
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 0);
    REQUIRE(r.stats.status == PresolveStatus::Unchanged);
    REQUIRE(r.reduced.coefs == b.p.coefs);
    REQUIRE(r.reduced.row_hi == b.p.row_hi);
}

TEST_CASE("coefficient tightening skips a step that rounds to nothing", "[presolve]") {
    // 1e13 x + y <= 2e13 + 0.062, x binary, y in [0, 2e13]: the exact d is
    // 0.0625, and the rounding margins (gap error ~0.056, side roundoff ~0.004,
    // eps |a| ~0.002) cut it to ~3.4e-4: above the tolerance but below
    // ulp(1e13) / 2 ~ 9.8e-4, so fl(1e13 - d) == 1e13. Counting it would report
    // a change and spin every round. If the margins shrink by more than ~6e-4,
    // d no longer rounds away and this test needs a new instance.
    Builder b;
    b.col(0, 1, true);
    b.col(0, 2e13, false);
    b.row({{0, 1e13}, {1, 1.0}}, -kInf, 20000000000000.062);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 0);
    REQUIRE(r.stats.status == PresolveStatus::Unchanged);
    REQUIRE(r.stats.rounds == 1);
}

TEST_CASE("coefficient tightening applies to a frozen column", "[presolve]") {
    // Frozen keeps the column in the problem; its linear coefficient may move.
    Builder b;
    b.col(0, 3, true, /*frozen=*/true);
    b.col(0, 1, false);
    b.row({{0, 5.0}, {1, 1.0}}, -kInf, 15.5);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 1);
    REQUIRE(r.reduced.n_cols() == 2);
    REQUIRE_THAT(r.reduced.coefs[0], WithinAbs(0.5, kTol));
    REQUIRE_THAT(r.reduced.row_hi[0], WithinAbs(2.0, kTol));
}

TEST_CASE("coefficient tightening is off when its option is", "[presolve]") {
    Builder b;
    b.col(0, 3, true);
    b.col(0, 1, false);
    b.row({{0, 5.0}, {1, 1.0}}, -kInf, 15.5);
    PresolveOptions o = coef_only();
    o.coefficient_tightening = false;
    const PresolveResult r = presolve(b.p, o);
    REQUIRE(r.stats.coefs_tightened == 0);
    REQUIRE(r.reduced.coefs == b.p.coefs);
}

TEST_CASE("coefficient tightening tightens every integral column of a row", "[presolve]") {
    // 3x1 + 3x2 + y <= 6.5, binaries x1, x2, y in [0, 1]: M = 7, gap = -0.5,
    // d = 2.5 for each, giving 0.5x1 + 0.5x2 + y <= 6.5 - 2.5 - 2.5 = 1.5.
    Builder b;
    b.col(0, 1, true);
    b.col(0, 1, true);
    b.col(0, 1, false);
    b.row({{0, 3.0}, {1, 3.0}, {2, 1.0}}, -kInf, 6.5);
    const PresolveResult r = presolve(b.p, coef_only());
    REQUIRE(r.stats.coefs_tightened == 2);
    REQUIRE(r.stats.rows_coef_tightened == 1);
    REQUIRE_THAT(r.reduced.coefs[0], WithinAbs(0.5, kTol));
    REQUIRE_THAT(r.reduced.coefs[1], WithinAbs(0.5, kTol));
    REQUIRE_THAT(r.reduced.row_hi[0], WithinAbs(1.5, kTol));
}

namespace {

/// Row i of p at x, against its sides with a 1e-9 tolerance.
bool row_holds(const LinearProblem& p, std::size_t i, const std::vector<double>& x) {
    double act = 0.0;
    for (std::size_t k = p.row_start[i]; k < p.row_start[i + 1]; ++k) {
        act += p.coefs[k] * x[static_cast<std::size_t>(p.cols[k])];
    }
    return act >= p.row_lo[i] - 1e-9 && act <= p.row_hi[i] + 1e-9;
}

/// Calls f on every point of p's box: integral columns on their integers,
/// continuous ones on a quarter-step grid.
template <typename F>
void for_each_grid_point(const LinearProblem& p, F&& f) {
    const std::size_t n = p.n_cols();
    std::vector<double> x(p.col_lb);
    while (true) {
        f(x);
        std::size_t j = 0;
        for (; j < n; ++j) {
            const double step = p.integral[j] != 0 ? 1.0 : 0.25;
            if (x[j] + step <= p.col_ub[j]) {
                x[j] += step;
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

TEST_CASE("a tightened row admits exactly the original row's integer points", "[presolve]") {
    // One one-sided row over up to 4 columns with small boxes, some
    // continuous, coefficients in {-6..6} \ {0}, a side at a random
    // half-integer. Brute force over the box, original row vs reduced row.
    int n_tightened = 0;
    for (uint32_t seed = 1; seed <= 2000; ++seed) {
        CAPTURE(seed);
        std::mt19937 rng(seed);
        Builder b;
        const int n = std::uniform_int_distribution<int>(1, 4)(rng);
        std::vector<std::pair<int32_t, double>> terms;
        double max_act = 0.0;
        for (int j = 0; j < n; ++j) {
            const bool integral = std::uniform_int_distribution<int>(0, 3)(rng) != 0;
            const double lo = std::uniform_int_distribution<int>(-2, 1)(rng);
            const double hi = lo + std::uniform_int_distribution<int>(1, 3)(rng);
            b.col(lo, hi, integral);
            int a = std::uniform_int_distribution<int>(-6, 5)(rng);
            a = a >= 0 ? a + 1 : a;
            terms.emplace_back(j, static_cast<double>(a));
            max_act += a > 0 ? a * hi : a * lo;
        }
        // A side below the max activity by 0.5 to 8, so the row binds.
        const double side = max_act - (0.5 * std::uniform_int_distribution<int>(1, 16)(rng));
        const bool le = std::uniform_int_distribution<int>(0, 1)(rng) == 0;
        if (le) {
            b.row(terms, -kInf, side);
        } else {
            for (auto& t : terms) {
                t.second = -t.second;
            }
            b.row(terms, -side, kInf);
        }
        const PresolveResult r = presolve(b.p, coef_only());
        REQUIRE(r.stats.status != PresolveStatus::Infeasible);
        REQUIRE(r.reduced.n_cols() == b.p.n_cols());
        REQUIRE(r.reduced.n_rows() == 1);
        n_tightened += r.stats.coefs_tightened;
        for_each_grid_point(b.p, [&](const std::vector<double>& x) {
            CAPTURE(x);
            REQUIRE(row_holds(b.p, 0, x) == row_holds(r.reduced, 0, x));
        });
    }
    REQUIRE(n_tightened > 500);  // the generator exercises the reduction
}
