#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include "test_helpers.h"
#include <cmath>

using namespace cbls;
using Catch::Matchers::WithinAbs;

// ===== New operator tests =====

TEST_CASE("Tan evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-1, 1);
    auto t = m.tan_expr(x);
    m.minimize(t);
    m.close();
    m.var_mut(vid(x)).value = 0.5;
    full_evaluate(m);
    REQUIRE_THAT(m.node(t).value, WithinAbs(std::tan(0.5), 1e-10));
}

TEST_CASE("Exp evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto e = m.exp_expr(x);
    m.minimize(e);
    m.close();
    m.var_mut(vid(x)).value = 1.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node(e).value, WithinAbs(std::exp(1.0), 1e-10));
}

TEST_CASE("Log evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0.01, 10);
    auto l = m.log_expr(x);
    m.minimize(l);
    m.close();
    m.var_mut(vid(x)).value = std::exp(1.0);
    full_evaluate(m);
    REQUIRE_THAT(m.node(l).value, WithinAbs(1.0, 1e-10));
}

TEST_CASE("Sqrt evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 100);
    auto s = m.sqrt_expr(x);
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = 9.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node(s).value, WithinAbs(3.0, 1e-10));
}

TEST_CASE("Geq evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto g = m.geq(x, y);
    m.minimize(m.abs_expr(g));
    m.close();

    // x >= y satisfied: violation = y - x <= 0
    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(g).value <= 0.0);  // satisfied

    m.var_mut(vid(x)).value = 2.0;
    m.var_mut(vid(y)).value = 7.0;
    full_evaluate(m);
    REQUIRE(m.node(g).value > 0.0);  // violated
}

TEST_CASE("Neq evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto n = m.neq(x, y);
    m.minimize(n);
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(n).value == 1.0);  // violated (equal)

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node(n).value == 0.0);  // satisfied (not equal)
}

TEST_CASE("Lt evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto l = m.lt(x, y);
    m.minimize(m.abs_expr(l));
    m.close();

    m.var_mut(vid(x)).value = 2.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node(l).value < 0.0);  // satisfied (2 < 5)

    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node(l).value > 0.0);  // violated (not strictly less)
}

TEST_CASE("Gt evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto g = m.gt(x, y);
    m.minimize(m.abs_expr(g));
    m.close();

    m.var_mut(vid(x)).value = 7.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(g).value < 0.0);  // satisfied (7 > 3)

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(g).value > 0.0);  // violated (not strictly greater)
}

// AD tests for new ops
TEST_CASE("AD: tan partial", "[dag]") {
    Model m;
    auto x = m.float_var(-1, 1);
    auto t = m.tan_expr(x);
    m.minimize(t);
    m.close();
    m.var_mut(vid(x)).value = 0.5;
    full_evaluate(m);
    double c = std::cos(0.5);
    REQUIRE_THAT(compute_partial(m, t, vid(x)), WithinAbs(1.0 / (c * c), 1e-10));
}

TEST_CASE("AD: exp partial", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto e = m.exp_expr(x);
    m.minimize(e);
    m.close();
    m.var_mut(vid(x)).value = 2.0;
    full_evaluate(m);
    REQUIRE_THAT(compute_partial(m, e, vid(x)), WithinAbs(std::exp(2.0), 1e-10));
}

TEST_CASE("AD: log partial", "[dag]") {
    Model m;
    auto x = m.float_var(0.01, 10);
    auto l = m.log_expr(x);
    m.minimize(l);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);
    REQUIRE_THAT(compute_partial(m, l, vid(x)), WithinAbs(1.0 / 3.0, 1e-10));
}

TEST_CASE("AD: sqrt partial", "[dag]") {
    Model m;
    auto x = m.float_var(0.01, 100);
    auto s = m.sqrt_expr(x);
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = 4.0;
    full_evaluate(m);
    REQUIRE_THAT(compute_partial(m, s, vid(x)), WithinAbs(1.0 / (2.0 * std::sqrt(4.0)), 1e-10));
}

TEST_CASE("AD: geq partial", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto g = m.geq(x, y);
    m.minimize(m.abs_expr(g));
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    // geq = child1 - child0, so d/dx = -1, d/dy = 1
    REQUIRE(compute_partial(m, g, vid(x)) == -1.0);
    REQUIRE(compute_partial(m, g, vid(y)) == 1.0);
}

TEST_CASE("AD: lt partial", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto l = m.lt(x, y);
    m.minimize(m.abs_expr(l));
    m.close();
    m.var_mut(vid(x)).value = 2.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);
    // lt = child0 - child1 + eps, so d/dx = 1, d/dy = -1
    REQUIRE(compute_partial(m, l, vid(x)) == 1.0);
    REQUIRE(compute_partial(m, l, vid(y)) == -1.0);
}

TEST_CASE("AD: gt partial", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto g = m.gt(x, y);
    m.minimize(m.abs_expr(g));
    m.close();
    m.var_mut(vid(x)).value = 7.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    // gt = child1 - child0 + eps, so d/dx = -1, d/dy = 1
    REQUIRE(compute_partial(m, g, vid(x)) == -1.0);
    REQUIRE(compute_partial(m, g, vid(y)) == 1.0);
}

TEST_CASE("AD: eq partial", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto e = m.eq_expr(x, y);
    m.minimize(e);
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    // eq = |child0 - child1|, diff > 0 so sign = 1
    REQUIRE(compute_partial(m, e, vid(x)) == 1.0);
    REQUIRE(compute_partial(m, e, vid(y)) == -1.0);
}

TEST_CASE("AD: neq partial", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto n = m.neq(x, y);
    m.minimize(n);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);
    // neq is non-differentiable, returns 0
    REQUIRE(compute_partial(m, n, vid(x)) == 0.0);
    REQUIRE(compute_partial(m, n, vid(y)) == 0.0);
}

TEST_CASE("Expr: scalar comparison", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);

    SECTION("Expr <= scalar") {
        auto c = x <= 5.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value <= 0.0);
    }

    SECTION("Expr >= scalar") {
        auto c = x >= 2.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value <= 0.0);
    }

    SECTION("Expr < scalar") {
        auto c = x < 5.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value < 0.0);
    }

    SECTION("Expr > scalar") {
        auto c = x > 2.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value < 0.0);
    }
}

// ===== Expr wrapper tests =====

TEST_CASE("Expr: basic arithmetic", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto y = m.Float(0, 10);
    auto f = x + y;
    m.minimize(f);
    m.close();
    x.var_mut().value = 3.0;
    y.var_mut().value = 4.0;
    full_evaluate(m);
    REQUIRE(m.node(f.handle).value == 7.0);
}

TEST_CASE("Expr: subtraction", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto y = m.Float(0, 10);
    auto f = x - y;
    m.minimize(f);
    m.close();
    x.var_mut().value = 7.0;
    y.var_mut().value = 3.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node(f.handle).value, WithinAbs(4.0, 1e-10));
}

TEST_CASE("Expr: multiplication", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto y = m.Float(0, 10);
    auto f = x * y;
    m.minimize(f);
    m.close();
    x.var_mut().value = 3.0;
    y.var_mut().value = 4.0;
    full_evaluate(m);
    REQUIRE(m.node(f.handle).value == 12.0);
}

TEST_CASE("Expr: division", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto y = m.Float(1, 10);
    auto f = x / y;
    m.minimize(f);
    m.close();
    x.var_mut().value = 6.0;
    y.var_mut().value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(f.handle).value == 2.0);
}

TEST_CASE("Expr: unary negation", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto f = -x;
    m.minimize(f);
    m.close();
    x.var_mut().value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node(f.handle).value == -5.0);
}

TEST_CASE("Expr: scalar mixed ops", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);

    SECTION("scalar + Expr") {
        auto f = 2.0 + x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 5.0);
    }

    SECTION("Expr + scalar") {
        auto f = x + 3.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 2.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 5.0);
    }

    SECTION("scalar * Expr") {
        auto f = 2.0 * x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 4.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 8.0);
    }

    SECTION("Expr * scalar") {
        auto f = x * 3.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 4.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 12.0);
    }

    SECTION("scalar - Expr") {
        auto f = 10.0 - x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(7.0, 1e-10));
    }

    SECTION("Expr - scalar") {
        auto f = x - 1.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(4.0, 1e-10));
    }

    SECTION("scalar / Expr") {
        auto f = 12.0 / x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 4.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 3.0);
    }

    SECTION("Expr / scalar") {
        auto f = x / 2.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 8.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 4.0);
    }
}

TEST_CASE("Expr: comparison operators", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto y = m.Float(0, 10);
    auto obj = x + y;  // need a node expression for minimize

    SECTION("<=") {
        auto c = x <= y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 3.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value <= 0.0);  // satisfied
    }

    SECTION(">=") {
        auto c = x >= y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 7.0;
        y.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value <= 0.0);  // satisfied
    }

    SECTION("<") {
        auto c = x < y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 2.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value < 0.0);  // satisfied
    }

    SECTION(">") {
        auto c = x > y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 8.0;
        y.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value < 0.0);  // satisfied
    }

    SECTION("eq") {
        auto c = x.eq(y);
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 5.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value == 0.0);  // satisfied
    }

    SECTION("neq") {
        auto c = x.neq(y);
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 3.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node(c.handle).value == 0.0);  // satisfied
    }
}

TEST_CASE("Expr: math functions", "[expr]") {
    Model m;
    auto x = m.Float(0.1, 10);

    SECTION("sin") {
        auto f = sin(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = M_PI / 2;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(1.0, 1e-10));
    }

    SECTION("cos") {
        auto f = cos(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 0.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(1.0, 1e-10));
    }

    SECTION("tan") {
        auto f = tan(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 0.5;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(std::tan(0.5), 1e-10));
    }

    SECTION("exp") {
        auto f = exp(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 1.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(std::exp(1.0), 1e-10));
    }

    SECTION("log") {
        auto f = log(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = std::exp(1.0);
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(1.0, 1e-10));
    }

    SECTION("sqrt") {
        auto f = sqrt(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 9.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node(f.handle).value, WithinAbs(3.0, 1e-10));
    }

    SECTION("abs") {
        auto y = m.Float(-10, 10);
        auto f = abs(y);
        m.minimize(f);
        m.close();
        y.var_mut().value = -5.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 5.0);
    }

    SECTION("pow") {
        auto two = m.Constant(2.0);
        auto f = x.pow(two);
        m.minimize(f);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node(f.handle).value == 9.0);
    }
}

TEST_CASE("Expr: nested expression x*x + 2*x*y + sin(y)", "[expr]") {
    Model m;
    auto x = m.Float(-10, 10);
    auto y = m.Float(-10, 10);
    auto f = x * x + 2.0 * x * y + sin(y);
    m.minimize(f);
    m.close();

    x.var_mut().value = 2.0;
    y.var_mut().value = 1.0;
    full_evaluate(m);
    double expected = 4.0 + 4.0 + std::sin(1.0);
    REQUIRE_THAT(m.node(f.handle).value, WithinAbs(expected, 1e-10));
}

TEST_CASE("Expr: same result as int32_t API", "[expr]") {
    // Build same model two ways and compare
    Model m1;
    auto x1 = m1.float_var(-10, 10);
    auto y1 = m1.float_var(-10, 10);
    auto two1 = m1.constant(2);
    auto x_sq1 = m1.pow_expr(x1, two1);
    auto xy1 = m1.prod(x1, y1);
    auto two_xy1 = m1.prod(two1, xy1);
    auto sin_y1 = m1.sin_expr(y1);
    auto f1 = m1.sum({x_sq1, two_xy1, sin_y1});
    m1.minimize(f1);
    m1.close();
    m1.var_mut(vid(x1)).value = 2.0;
    m1.var_mut(vid(y1)).value = 1.0;
    full_evaluate(m1);

    Model m2;
    auto x2 = m2.Float(-10, 10);
    auto y2 = m2.Float(-10, 10);
    auto f2 = x2 * x2 + 2.0 * x2 * y2 + sin(y2);
    m2.minimize(f2);
    m2.close();
    x2.var_mut().value = 2.0;
    y2.var_mut().value = 1.0;
    full_evaluate(m2);

    REQUIRE_THAT(m2.node(f2.handle).value, WithinAbs(m1.node(f1).value, 1e-10));
}

TEST_CASE("Expr: add_constraint and minimize with Expr", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto y = m.Float(0, 10);
    m.add_constraint(x <= y);
    m.minimize(x + y);
    m.close();
    // Just verify it builds and closes without error
    REQUIRE(m.constraint_ids().size() == 1);
    REQUIRE(m.objective_id() >= 0);
}
