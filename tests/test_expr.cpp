#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <cmath>

using namespace cbls;
using Catch::Matchers::WithinAbs;

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
        REQUIRE(m.node_value(c.handle) <= 0.0);
    }

    SECTION("Expr >= scalar") {
        auto c = x >= 2.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) <= 0.0);
    }

    SECTION("Expr < scalar") {
        auto c = x < 5.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) < 0.0);
    }

    SECTION("Expr > scalar") {
        auto c = x > 2.0;
        m.add_constraint(c);
        m.minimize(x + 0.0);
        m.close();
        x.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) < 0.0);
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
    REQUIRE(m.node_value(f.handle) == 7.0);
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
    REQUIRE_THAT(m.node_value(f.handle), WithinAbs(4.0, 1e-10));
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
    REQUIRE(m.node_value(f.handle) == 12.0);
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
    REQUIRE(m.node_value(f.handle) == 2.0);
}

TEST_CASE("Expr: unary negation", "[expr]") {
    Model m;
    auto x = m.Float(0, 10);
    auto f = -x;
    m.minimize(f);
    m.close();
    x.var_mut().value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node_value(f.handle) == -5.0);
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
        REQUIRE(m.node_value(f.handle) == 5.0);
    }

    SECTION("Expr + scalar") {
        auto f = x + 3.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 2.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 5.0);
    }

    SECTION("scalar * Expr") {
        auto f = 2.0 * x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 4.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 8.0);
    }

    SECTION("Expr * scalar") {
        auto f = x * 3.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 4.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 12.0);
    }

    SECTION("scalar - Expr") {
        auto f = 10.0 - x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(7.0, 1e-10));
    }

    SECTION("Expr - scalar") {
        auto f = x - 1.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(4.0, 1e-10));
    }

    SECTION("scalar / Expr") {
        auto f = 12.0 / x;
        m.minimize(f);
        m.close();
        x.var_mut().value = 4.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 3.0);
    }

    SECTION("Expr / scalar") {
        auto f = x / 2.0;
        m.minimize(f);
        m.close();
        x.var_mut().value = 8.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 4.0);
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
        REQUIRE(m.node_value(c.handle) <= 0.0);  // satisfied
    }

    SECTION(">=") {
        auto c = x >= y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 7.0;
        y.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) <= 0.0);  // satisfied
    }

    SECTION("<") {
        auto c = x < y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 2.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) < 0.0);  // satisfied
    }

    SECTION(">") {
        auto c = x > y;
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 8.0;
        y.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) < 0.0);  // satisfied
    }

    SECTION("eq") {
        auto c = x.eq(y);
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 5.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) == 0.0);  // satisfied
    }

    SECTION("neq") {
        auto c = x.neq(y);
        m.add_constraint(c);
        m.minimize(obj);
        m.close();
        x.var_mut().value = 3.0;
        y.var_mut().value = 5.0;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) == 0.0);  // satisfied
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
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(1.0, 1e-10));
    }

    SECTION("cos") {
        auto f = cos(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 0.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(1.0, 1e-10));
    }

    SECTION("tan") {
        auto f = tan(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 0.5;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(std::tan(0.5), 1e-10));
    }

    SECTION("exp") {
        auto f = exp(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 1.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(std::exp(1.0), 1e-10));
    }

    SECTION("log") {
        auto f = log(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = std::exp(1.0);
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(1.0, 1e-10));
    }

    SECTION("sqrt") {
        auto f = sqrt(x);
        m.minimize(f);
        m.close();
        x.var_mut().value = 9.0;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(f.handle), WithinAbs(3.0, 1e-10));
    }

    SECTION("abs") {
        auto y = m.Float(-10, 10);
        auto f = abs(y);
        m.minimize(f);
        m.close();
        y.var_mut().value = -5.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 5.0);
    }

    SECTION("pow") {
        auto two = m.Constant(2.0);
        auto f = x.pow(two);
        m.minimize(f);
        m.close();
        x.var_mut().value = 3.0;
        full_evaluate(m);
        REQUIRE(m.node_value(f.handle) == 9.0);
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
    REQUIRE_THAT(m.node_value(f.handle), WithinAbs(expected, 1e-10));
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

    REQUIRE_THAT(m2.node_value(f2.handle), WithinAbs(m1.node_value(f1), 1e-10));
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
