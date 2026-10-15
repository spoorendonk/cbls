// Exact univariate quadratic roots as FJ jump candidates (#221, LS-IQCQP):
// GFJConfig::quadratic_roots / SearchConfig::fj_quadratic_roots. Default off,
// and the default path is pinned by the trajectory fences; what is pinned here
// is the root formula, the DAG walk that recognises a quadratic row, the domain
// and Int handling of its candidates, and that the flag reaches every FJ.
#include "test_helpers.h"

#include <array>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cbls;

namespace {

std::vector<double> roots_of(double a, double b, double c) {
    std::array<double, 2> r{};
    const int n = quadratic_roots(UnivariateQuadratic{a, b, c}, r);
    return {r.begin(), r.begin() + n};
}

// Sets `x` and re-evaluates, so node values match the assignment.
void set_value(Model& m, int32_t handle, double v) {
    m.var_mut(vid(handle)).value = v;
    full_evaluate(m);
}

}  // namespace

TEST_CASE("quadratic roots are off by default (#221)", "[fj][quadratic_roots]") {
    REQUIRE_FALSE(GFJConfig{}.quadratic_roots);
    REQUIRE_FALSE(SearchConfig{}.fj_quadratic_roots);
    REQUIRE_FALSE(LNS{}.quadratic_roots());
}

TEST_CASE("quadratic_roots: two, one, none and the degenerate a = 0 (#221)",
          "[fj][quadratic_roots]") {
    SECTION("two roots, ascending") {
        REQUIRE(roots_of(1.0, -5.0, 6.0) == std::vector<double>{2.0, 3.0});
        REQUIRE(roots_of(-1.0, 5.0, -6.0) == std::vector<double>{2.0, 3.0});
    }
    SECTION("one root at a zero discriminant") {
        REQUIRE(roots_of(1.0, -4.0, 4.0) == std::vector<double>{2.0});
    }
    SECTION("no real root") {
        REQUIRE(roots_of(1.0, 0.0, 1.0).empty());
        REQUIRE(roots_of(std::nan(""), 0.0, 1.0).empty());
    }
    SECTION("a = 0 is the linear root, or none when b = 0 too") {
        REQUIRE(roots_of(0.0, 2.0, -4.0) == std::vector<double>{2.0});
        REQUIRE(roots_of(0.0, 0.0, -4.0).empty());
        REQUIRE(roots_of(0.0, 0.0, 0.0).empty());
    }
    SECTION("the small root survives cancellation") {
        // x^2 + 1e8 x + 1: the textbook formula loses the root near -1e-8.
        const std::vector<double> r = roots_of(1.0, 1e8, 1.0);
        REQUIRE(r.size() == 2);
        REQUIRE(r[1] == Catch::Approx(-1e-8).epsilon(1e-12));
        REQUIRE(r[0] == Catch::Approx(-1e8).epsilon(1e-12));
    }
}

TEST_CASE("row_quadratic_in reads a row's residual in one variable (#221)",
          "[fj][quadratic_roots]") {
    Model m;
    auto x = m.float_var(-10.0, 10.0);
    auto y = m.float_var(-10.0, 10.0);
    const auto two = m.constant(2.0);
    // 0: x^2 + 3x <= 10           residual  x^2 + 3x - 10
    m.add_constraint(
        m.leq(m.sum({m.pow_expr(x, two), m.prod(m.constant(3.0), x)}), m.constant(10.0)));
    // 1: x*y >= 4 (bilinear)      residual  -y x + 4
    m.add_constraint(m.geq(m.prod(x, y), m.constant(4.0)));
    // 2: (x*y)*x / 2 == exp(y)    residual  (y/2) x^2 - exp(y), under |.|
    m.add_constraint(m.eq_expr(m.div_expr(m.prod(m.prod(x, y), x), two), m.exp_expr(y)));
    // 3: x*x*x <= 1               cubic: not a quadratic
    m.add_constraint(m.leq(m.prod(m.prod(x, x), x), m.constant(1.0)));
    // 4: exp(x) <= 1              not a polynomial in x
    m.add_constraint(m.leq(m.exp_expr(x), m.constant(1.0)));
    // 5: x != y                   an indicator
    m.add_constraint(m.neq(x, y));
    // 6: -(x - y)^2 < 0           strict, with its margin; Neg and Pow of an affine base
    m.add_constraint(m.lt(m.neg(m.pow_expr(m.sum({x, m.neg(y)}), two)), m.constant(0.0)));
    // 7: x / y <= 1               a division by a quantity free of x
    m.add_constraint(m.leq(m.div_expr(x, y), m.constant(1.0)));
    // 8: y / x <= 1               a division by x: not a polynomial
    m.add_constraint(m.leq(m.div_expr(y, x), m.constant(1.0)));
    // 9: x^2 + 1e8 y - 1e8 <= 0.3 large cancelling terms, a modest residual
    m.add_constraint(m.leq(m.sum({m.prod(x, x), m.prod(m.constant(1e8), y), m.constant(-1e8)}),
                           m.constant(0.3)));
    m.close();
    set_value(m, x, 1.5);
    set_value(m, y, 3.0);

    UnivariateQuadratic q;
    REQUIRE(row_quadratic_in(m, 0, vid(x), q));
    REQUIRE((q.a == 1.0 && q.b == 3.0 && q.c == -10.0));

    REQUIRE(row_quadratic_in(m, 1, vid(x), q));
    REQUIRE((q.a == 0.0 && q.b == -3.0 && q.c == 4.0));

    REQUIRE(row_quadratic_in(m, 2, vid(x), q));
    REQUIRE(q.a == 1.5);
    REQUIRE(q.b == 0.0);
    REQUIRE(q.c == Catch::Approx(-std::exp(3.0)));
    // In y the same row is linear on the left but exp(y) on the right.
    REQUIRE_FALSE(row_quadratic_in(m, 2, vid(y), q));

    REQUIRE_FALSE(row_quadratic_in(m, 3, vid(x), q));
    REQUIRE_FALSE(row_quadratic_in(m, 4, vid(x), q));
    REQUIRE_FALSE(row_quadratic_in(m, 5, vid(x), q));

    // -(x - 3)^2 + margin = -x^2 + 6x - 9 + margin.
    REQUIRE(row_quadratic_in(m, 6, vid(x), q));
    REQUIRE((q.a == -1.0 && q.b == 6.0));
    REQUIRE(q.c == -9.0 + kStrictComparisonEps);

    REQUIRE(row_quadratic_in(m, 7, vid(x), q));
    REQUIRE((q.a == 0.0 && q.b == Catch::Approx(1.0 / 3.0) && q.c == -1.0));
    REQUIRE_FALSE(row_quadratic_in(m, 8, vid(x), q));
    // Recognised although the walk's sum and the DAG's round differently at 1e8.
    m.var_mut(vid(x)).value = 0.7;
    set_value(m, y, 1.0);
    REQUIRE(row_quadratic_in(m, 9, vid(x), q));
    REQUIRE(q.a == 1.0);
}

TEST_CASE("row_quadratic_in gives up past its node budget (#221)", "[fj][quadratic_roots]") {
    // x^2 + sum of 2000 c_i z_i <= 0: a quadratic in x, but one whose walk
    // costs more than the budget, so it falls back to Newton.
    for (const int n : {10, 2000}) {
        Model m;
        auto x = m.float_var(-10.0, 10.0);
        std::vector<int32_t> terms{m.prod(x, x)};
        for (int i = 0; i < n; ++i) {
            terms.push_back(m.prod(m.constant(1.0), m.float_var(0.0, 1.0)));
        }
        m.add_constraint(m.leq(m.sum(terms), m.constant(0.0)));
        m.close();
        set_value(m, x, 2.0);
        UnivariateQuadratic q;
        CAPTURE(n);
        REQUIRE(row_quadratic_in(m, 0, vid(x), q) == (n == 10));
    }
}

TEST_CASE("quadratic candidates keep in-domain roots only (#221)", "[fj][quadratic_roots]") {
    Model m;
    auto x = m.float_var(0.0, 10.0);
    // x^2 >= 4: roots -2 and 2; -2 is outside [0, 10].
    m.add_constraint(m.geq(m.prod(x, x), m.constant(4.0)));
    // x^2 - 5x + 6 <= 0: roots 2 and 3, both inside.
    m.add_constraint(m.leq(m.sum({m.prod(x, x), m.prod(m.constant(-5.0), x)}), m.constant(-6.0)));
    // x^2 >= 400: roots -20 and 20, both outside.
    m.add_constraint(m.geq(m.prod(x, x), m.constant(400.0)));
    // x + 1 <= 0: affine, so not a quadratic row.
    m.add_constraint(m.leq(m.sum({x, m.constant(1.0)}), m.constant(0.0)));
    m.close();
    set_value(m, x, 0.5);
    std::vector<double> out;
    REQUIRE(quadratic_row_candidates(m, 0, vid(x), 0.5, out));
    REQUIRE(out == std::vector<double>{2.0});
    REQUIRE(quadratic_row_candidates(m, 1, vid(x), 0.5, out));
    REQUIRE(out == std::vector<double>{2.0, 3.0});
    // Recognised, but nothing to offer: the caller takes the Newton step.
    REQUIRE(quadratic_row_candidates(m, 2, vid(x), 0.5, out));
    REQUIRE(out.empty());
    REQUIRE_FALSE(quadratic_row_candidates(m, 3, vid(x), 0.5, out));
    REQUIRE(out.empty());
    // The current value is never a candidate.
    set_value(m, x, 2.0);
    REQUIRE(quadratic_row_candidates(m, 1, vid(x), 2.0, out));
    REQUIRE(out == std::vector<double>{3.0});
}

TEST_CASE("an Int takes the floor and ceil of each root (#221)", "[fj][quadratic_roots]") {
    Model m;
    auto x = m.int_var(-2, 100);
    // x^2 >= 5: roots -/+2.236 -> -3 (outside), -2, 2, 3.
    m.add_constraint(m.geq(m.prod(x, x), m.constant(5.0)));
    // x^2 == 49: integral roots give one candidate each.
    m.add_constraint(m.eq_expr(m.prod(x, x), m.constant(49.0)));
    m.close();
    set_value(m, x, 0.0);
    std::vector<double> out;
    REQUIRE(quadratic_row_candidates(m, 0, vid(x), 0.0, out));
    REQUIRE(out == std::vector<double>{-2.0, 2.0, 3.0});
    REQUIRE(quadratic_row_candidates(m, 1, vid(x), 0.0, out));
    REQUIRE(out == std::vector<double>{7.0});
}

TEST_CASE("compute_var_jump offers the roots only with the flag (#221)", "[fj][quadratic_roots]") {
    SECTION("Float: the root replaces the Newton step") {
        // x^2 <= 2 at x = 5: Newton lands at 2.7, still violated; the box
        // midpoint 0 is feasible. Off, 0 wins; on, the root -sqrt(2) is
        // offered first and ties it.
        Model m;
        auto x = m.float_var(-10.0, 10.0);
        m.add_constraint(m.leq(m.prod(x, x), m.constant(2.0)));
        m.close();
        set_value(m, x, 5.0);
        const std::vector<double> w(1, 1.0);
        const JumpResult off = compute_var_jump(m, w, vid(x));
        const JumpResult on = compute_var_jump(m, w, vid(x), false, nullptr,
                                               kDefaultFeasibilityTolerance, false, nullptr, true);
        REQUIRE(off.jump_value == 0.0);
        REQUIRE(on.jump_value == Catch::Approx(-std::sqrt(2.0)));
        REQUIRE(on.score == off.score);
    }
    SECTION("Int: a wide domain's grid misses the root, the roots hit it") {
        Model m;
        auto x = m.int_var(-1000, 1000);
        m.add_constraint(m.eq_expr(m.prod(x, x), m.constant(49.0)));
        m.close();
        set_value(m, x, 0.0);
        const std::vector<double> w(1, 1.0);
        const JumpResult off = compute_var_jump(m, w, vid(x));
        const JumpResult on = compute_var_jump(m, w, vid(x), false, nullptr,
                                               kDefaultFeasibilityTolerance, false, nullptr, true);
        REQUIRE(off.score == 1.0);  // x = +-1: |1 - 49| = 48
        REQUIRE(std::abs(on.jump_value) == 7.0);
        REQUIRE(on.score == 49.0);
    }
}

TEST_CASE("the quadratic-roots flag reaches solve() and the LNS repair (#221)",
          "[fj][quadratic_roots]") {
    auto build = [](Model& m) {
        // x^2 == 49 over a domain too wide to enumerate: off, FJ walks to 7
        // by +-1 steps; on, the root is a candidate of the first jump.
        auto x = m.int_var(-1000, 1000);
        m.add_constraint(m.eq_expr(m.prod(x, x), m.constant(49.0)));
        m.close();
    };
    SECTION("solve()") {
        std::array<int64_t, 2> iterations{};
        for (const bool on : {false, true}) {
            Model m;
            build(m);
            SearchConfig config;
            config.max_iterations = 20'000;
            config.fj_quadratic_roots = on;
            const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/1, true, nullptr, nullptr,
                                         3, nullptr, config);
            CAPTURE(on, r.iterations);
            REQUIRE(r.feasible);
            iterations[static_cast<size_t>(on)] = r.iterations;
        }
        REQUIRE(iterations[1] < iterations[0]);
    }
    SECTION("fj_nl_initialize") {
        std::array<int64_t, 2> iterations{};
        for (const bool on : {false, true}) {
            Model m;
            build(m);
            full_evaluate(m);
            ViolationManager vm(m);
            RNG rng(1);
            iterations[static_cast<size_t>(on)] =
                fj_nl_initialize(m, vm, 20'000, &rng, 0.0, kDefaultFeasibilityTolerance, false, on);
            CAPTURE(on, iterations[static_cast<size_t>(on)]);
            REQUIRE(vm.total_violation() <= kDefaultFeasibilityTolerance);
        }
        REQUIRE(iterations[1] < iterations[0]);
    }
    SECTION("LNS::set_quadratic_roots reaches the repair FJ") {
        // Four copies of x^2 == 7000^2 over [-1e6, 1e6], all at 5e5: the
        // nearest grid point is 0 (step 62500), and the repair's 2000
        // iterations of +-1 steps cannot walk the 7000 from there; the roots
        // jump to it.
        for (const bool on : {false, true}) {
            Model m;
            for (int i = 0; i < 4; ++i) {
                auto x = m.int_var(-1'000'000, 1'000'000);
                m.add_constraint(m.eq_expr(m.prod(x, x), m.constant(49e6)));
                m.var_mut(vid(x)).value = 500'000.0;
            }
            m.close();
            full_evaluate(m);
            ViolationManager vm(m);
            RNG rng(1);
            LNS lns;
            lns.set_quadratic_roots(on);
            REQUIRE(lns.quadratic_roots() == on);
            lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0);
            full_evaluate(m);
            CAPTURE(on, vm.total_violation());
            REQUIRE((vm.total_violation() == 0.0) == on);
        }
    }
}
