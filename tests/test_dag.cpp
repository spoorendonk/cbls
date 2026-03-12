#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include "test_helpers.h"
#include <cmath>

using namespace cbls;
using Catch::Matchers::WithinAbs;

TEST_CASE("Sum evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto s = m.sum({x, y});
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 4.0;
    full_evaluate(m);
    REQUIRE(m.node(s).value == 7.0);
}

TEST_CASE("Prod evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto p = m.prod(x, y);
    m.minimize(p);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 4.0;
    full_evaluate(m);
    REQUIRE(m.node(p).value == 12.0);
}

TEST_CASE("Div evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(1, 10);
    auto d = m.div_expr(x, y);
    m.minimize(d);
    m.close();
    m.var_mut(vid(x)).value = 6.0;
    m.var_mut(vid(y)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(d).value == 2.0);
}

TEST_CASE("Pow evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto two = m.constant(2);
    auto p = m.pow_expr(x, two);
    m.minimize(p);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(p).value == 9.0);
}

TEST_CASE("Sin evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto s = m.sin_expr(x);
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = M_PI / 2;
    full_evaluate(m);
    REQUIRE_THAT(m.node(s).value, WithinAbs(1.0, 1e-10));
}

TEST_CASE("Cos evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto c = m.cos_expr(x);
    m.minimize(c);
    m.close();
    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node(c).value, WithinAbs(1.0, 1e-10));
}

TEST_CASE("Abs evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto a = m.abs_expr(x);
    m.minimize(a);
    m.close();
    m.var_mut(vid(x)).value = -5.0;
    full_evaluate(m);
    REQUIRE(m.node(a).value == 5.0);
}

TEST_CASE("Min/Max evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto mn = m.min_expr({x, y});
    auto mx = m.max_expr({x, y});
    auto total = m.sum({mn, mx});
    m.minimize(total);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 7.0;
    full_evaluate(m);
    REQUIRE(m.node(mn).value == 3.0);
    REQUIRE(m.node(mx).value == 7.0);
}

TEST_CASE("Neg evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto n = m.neg(x);
    m.minimize(n);
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node(n).value == -5.0);
}

TEST_CASE("If-then-else evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(0, 10);
    auto z = m.float_var(0, 10);
    auto ite = m.if_then_else(x, y, z);
    m.minimize(ite);
    m.close();

    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 5.0;
    m.var_mut(vid(z)).value = 9.0;
    full_evaluate(m);
    REQUIRE(m.node(ite).value == 5.0);

    m.var_mut(vid(x)).value = -1.0;
    full_evaluate(m);
    REQUIRE(m.node(ite).value == 9.0);
}

TEST_CASE("Constants evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto five = m.constant(5.0);
    auto expr = m.sum({x, five});
    m.minimize(expr);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);
    REQUIRE(m.node(expr).value == 8.0);
}

TEST_CASE("Nested expression: x^2 + 2*x*y + sin(y)", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto two = m.constant(2);
    auto x_sq = m.pow_expr(x, two);
    auto xy = m.prod(x, y);
    auto two_xy = m.prod(two, xy);
    auto sin_y = m.sin_expr(y);
    auto f = m.sum({x_sq, two_xy, sin_y});
    m.minimize(f);
    m.close();

    m.var_mut(vid(x)).value = 2.0;
    m.var_mut(vid(y)).value = 1.0;
    full_evaluate(m);
    double expected = 4.0 + 4.0 + std::sin(1.0);
    REQUIRE_THAT(m.node(f).value, WithinAbs(expected, 1e-10));
}

// Delta evaluation tests
TEST_CASE("Delta evaluation matches full", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto z = m.float_var(0, 10);
    auto xy = m.prod(x, y);
    auto f = m.sum({xy, z});
    m.minimize(f);
    m.close();

    m.var_mut(vid(x)).value = 2.0;
    m.var_mut(vid(y)).value = 3.0;
    m.var_mut(vid(z)).value = 1.0;
    full_evaluate(m);
    REQUIRE(m.node(f).value == 7.0);

    m.var_mut(vid(x)).value = 5.0;
    double delta_result = delta_evaluate(m, {vid(x)});
    REQUIRE(delta_result == 16.0);

    double full_result = full_evaluate(m);
    REQUIRE(full_result == delta_result);
}

TEST_CASE("Delta eval with no changes", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto two = m.constant(2);
    auto f = m.pow_expr(x, two);
    m.minimize(f);
    m.close();

    m.var_mut(vid(x)).value = 4.0;
    full_evaluate(m);

    double result = delta_evaluate(m, {});
    REQUIRE(result == 16.0);
}

TEST_CASE("Delta eval multiple vars", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto two = m.constant(2);
    auto f = m.sum({m.pow_expr(x, two), m.pow_expr(y, two)});
    m.minimize(f);
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 4.0;
    full_evaluate(m);
    REQUIRE(m.node(f).value == 25.0);

    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 2.0;
    double result = delta_evaluate(m, {vid(x), vid(y)});
    REQUIRE(result == 5.0);
}

// ListVar tests
TEST_CASE("ListVar at()", "[dag]") {
    Model m;
    auto lv = m.list_var(5);
    auto idx = m.constant(2);
    auto a = m.at(lv, idx);
    m.minimize(a);
    m.close();

    auto& v = m.var_mut(vid(lv));
    v.elements = {10, 20, 30, 40, 50};
    full_evaluate(m);
    REQUIRE(m.node(a).value == 30.0);
}

TEST_CASE("ListVar lambda_sum", "[dag]") {
    Model m;
    auto lv = m.list_var(4);
    auto ls = m.lambda_sum(lv, [](int e) { return static_cast<double>(e * e); });
    m.minimize(ls);
    m.close();

    auto& v = m.var_mut(vid(lv));
    v.elements = {1, 2, 3, 4};
    full_evaluate(m);
    REQUIRE(m.node(ls).value == 30.0);  // 1+4+9+16
}

TEST_CASE("ListVar delta eval", "[dag]") {
    Model m;
    auto lv = m.list_var(3);
    auto ls = m.lambda_sum(lv, [](int e) { return static_cast<double>(e); });
    m.minimize(ls);
    m.close();

    auto& v = m.var_mut(vid(lv));
    v.elements = {0, 1, 2};
    full_evaluate(m);
    REQUIRE(m.node(ls).value == 3.0);

    v.elements = {2, 1, 0};
    double result = delta_evaluate(m, {vid(lv)});
    REQUIRE(result == 3.0);  // sum unchanged for permutation
}

// SetVar tests
TEST_CASE("SetVar count", "[dag]") {
    Model m;
    auto sv = m.set_var(10, 0, 10);
    auto c = m.count(sv);
    m.minimize(c);
    m.close();

    auto& v = m.var_mut(vid(sv));
    v.elements = {1, 3, 5, 7};
    full_evaluate(m);
    REQUIRE(m.node(c).value == 4.0);
}

// AD tests
TEST_CASE("AD: sum partials", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto s = m.sum({x, y});
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 4.0;
    full_evaluate(m);

    REQUIRE(compute_partial(m, s, vid(x)) == 1.0);
    REQUIRE(compute_partial(m, s, vid(y)) == 1.0);
}

TEST_CASE("AD: prod partials", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto p = m.prod(x, y);
    m.minimize(p);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 4.0;
    full_evaluate(m);

    REQUIRE(compute_partial(m, p, vid(x)) == 4.0);
    REQUIRE(compute_partial(m, p, vid(y)) == 3.0);
}

TEST_CASE("AD: pow partial", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto two = m.constant(2);
    auto p = m.pow_expr(x, two);
    m.minimize(p);
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);

    REQUIRE_THAT(compute_partial(m, p, vid(x)), WithinAbs(6.0, 1e-10));
}

TEST_CASE("AD: sin partial", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto s = m.sin_expr(x);
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = 1.0;
    full_evaluate(m);

    REQUIRE_THAT(compute_partial(m, s, vid(x)), WithinAbs(std::cos(1.0), 1e-10));
}

TEST_CASE("AD: chain rule sin(x^2)", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto two = m.constant(2);
    auto x2 = m.pow_expr(x, two);
    auto f = m.sin_expr(x2);
    m.minimize(f);
    m.close();
    m.var_mut(vid(x)).value = 1.5;
    full_evaluate(m);

    double expected = 2.0 * 1.5 * std::cos(1.5 * 1.5);
    REQUIRE_THAT(compute_partial(m, f, vid(x)), WithinAbs(expected, 1e-10));
}

TEST_CASE("AD: composite x^2 + 2*x*y", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto two = m.constant(2);
    auto x_sq = m.pow_expr(x, two);
    auto xy = m.prod(x, y);
    auto two_xy = m.prod(two, xy);
    auto f = m.sum({x_sq, two_xy});
    m.minimize(f);
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 2.0;
    full_evaluate(m);

    double expected = 2 * 3.0 + 2 * 2.0;  // 10
    REQUIRE_THAT(compute_partial(m, f, vid(x)), WithinAbs(expected, 1e-10));
}
