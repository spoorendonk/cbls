#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cbls;
using Catch::Matchers::ContainsSubstring;
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
    REQUIRE(m.node_value(s) == 7.0);
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
    REQUIRE(m.node_value(p) == 12.0);
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
    REQUIRE(m.node_value(d) == 2.0);
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
    REQUIRE(m.node_value(p) == 9.0);
}

TEST_CASE("Sin evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto s = m.sin_expr(x);
    m.minimize(s);
    m.close();
    m.var_mut(vid(x)).value = M_PI / 2;
    full_evaluate(m);
    REQUIRE_THAT(m.node_value(s), WithinAbs(1.0, 1e-10));
}

TEST_CASE("Cos evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto c = m.cos_expr(x);
    m.minimize(c);
    m.close();
    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node_value(c), WithinAbs(1.0, 1e-10));
}

TEST_CASE("Abs evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto a = m.abs_expr(x);
    m.minimize(a);
    m.close();
    m.var_mut(vid(x)).value = -5.0;
    full_evaluate(m);
    REQUIRE(m.node_value(a) == 5.0);
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
    REQUIRE(m.node_value(mn) == 3.0);
    REQUIRE(m.node_value(mx) == 7.0);
}

TEST_CASE("Neg evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto n = m.neg(x);
    m.minimize(n);
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    REQUIRE(m.node_value(n) == -5.0);
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
    REQUIRE(m.node_value(ite) == 5.0);

    m.var_mut(vid(x)).value = -1.0;
    full_evaluate(m);
    REQUIRE(m.node_value(ite) == 9.0);
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
    REQUIRE(m.node_value(expr) == 8.0);
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
    REQUIRE_THAT(m.node_value(f), WithinAbs(expected, 1e-10));
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
    REQUIRE(m.node_value(f) == 7.0);

    m.var_mut(vid(x)).value = 5.0;
    double delta_result = delta_evaluate(m, {vid(x)});
    REQUIRE(delta_result == 16.0);

    double full_result = full_evaluate(m);
    REQUIRE(full_result == delta_result);
}

TEST_CASE("back-references list each parent once, including a repeated child", "[dag]") {
    // `rebuild_back_references` deduplicates by a last-writer stamp rather than
    // by searching the list it is building, which is what took it from
    // O(sum of degree^2) to O(edges) -- 68% of square47's model build, whose
    // 95k columns appear in ~288 rows each.
    //
    // The stamp is only correct because a duplicate can come from ONE parent
    // naming the same child twice, a parent being visited once. `prod(x, x)` is
    // that case, so it is the case pinned: x must list the product once, and
    // both of x's genuinely distinct parents must survive.
    Model m;
    auto x = m.float_var(0, 10);
    auto sq = m.prod(x, x);  // names x twice
    auto lin = m.prod(x, m.constant(3.0));
    auto f = m.sum({sq, lin});
    m.minimize(f);
    m.close();

    const auto deps = m.dependents(vid(x));
    REQUIRE(std::count(deps.begin(), deps.end(), sq) == 1);
    REQUIRE(std::count(deps.begin(), deps.end(), lin) == 1);
    REQUIRE(deps.size() == 2);

    // And the dedup must not cost a dependency: a change to x still reaches f
    // through both arms.
    m.var_mut(vid(x)).value = 2.0;
    full_evaluate(m);
    REQUIRE(m.node_value(f) == 4.0 + 6.0);
    m.var_mut(vid(x)).value = 3.0;
    REQUIRE(delta_evaluate(m, {vid(x)}) == 9.0 + 9.0);
}

TEST_CASE("flat edge storage keeps child, parent, dependent and topological order", "[dag]") {
    // #156 moved every node's children and back-references out of per-node
    // vectors into flat arrays the model owns. That is a representation change
    // and must be neutral to search trajectories, and those trajectories depend
    // on ORDER: delta_evaluate's BFS enqueues in parents()/dependents() order,
    // and the topological sort's order is what full_evaluate and the AD sweep
    // walk. So the exact sequences are pinned here, on a DAG where a NODE is a
    // repeated child (`prod(n, n)`) -- the case where the sort now reads
    // deduplicated parent lists instead of counting both edges.
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto n = m.sum({x, y, x});       // node 0, names x twice
    auto p1 = m.prod(n, n);          // node 1, names n twice
    auto c = m.constant(3.0);        // node 2, a source made after a non-source
    auto p2 = m.sum({n, c});         // node 3
    auto p3 = m.prod(c, n);          // node 4
    auto top = m.sum({p1, p2, p3});  // node 5

    // Children are readable before close(), in the order given.
    const auto kids = m.children(m.node(n));
    REQUIRE(kids.size() == 3);
    REQUIRE((kids[0].is_var && kids[0].id == vid(x)));
    REQUIRE((kids[1].is_var && kids[1].id == vid(y)));
    REQUIRE((kids[2].is_var && kids[2].id == vid(x)));
    // Back-references are not built yet.
    REQUIRE(m.parents(n).empty());
    REQUIRE(m.dependents(vid(x)).empty());

    m.minimize(top);
    m.close();

    const auto as_vector = [](ConstSpan<int32_t> s) {
        return std::vector<int32_t>(s.begin(), s.end());
    };
    // Ascending parent id, each parent once however often it names the child.
    REQUIRE(as_vector(m.parents(n)) == std::vector<int32_t>{p1, p2, p3});
    REQUIRE(as_vector(m.parents(c)) == std::vector<int32_t>{p2, p3});
    REQUIRE(m.parents(top).empty());
    REQUIRE(as_vector(m.dependents(vid(x))) == std::vector<int32_t>{n});
    REQUIRE(as_vector(m.dependents(vid(y))) == std::vector<int32_t>{n});
    // Kahn's order: sources by id, then FIFO over the parent lists -- NOT id
    // order here (c, node 2, precedes p1, node 1), so an id-order or DFS sort
    // fails. The order matters beyond validity: the AD sweep accumulates in it.
    REQUIRE(m.topo_order() == std::vector<int32_t>{n, c, p1, p2, p3, top});

    // No node can be made after the rebuild by the ordinary builders -- they
    // refuse a closed model (#173), so no parent list goes stale. The refusal
    // moves no existing slice.
    REQUIRE_THROWS_AS(m.neg(top), std::logic_error);
    REQUIRE(m.parents(top).empty());
    REQUIRE(as_vector(m.parents(n)) == std::vector<int32_t>{p1, p2, p3});

    // A copy is a deep copy: its slices address its own arrays, so it evaluates
    // correctly after the original is gone -- which is how a portfolio worker
    // gets its model.
    auto copy = std::make_unique<Model>(m);
    m = Model();
    copy->var_mut(vid(x)).value = 1.0;
    copy->var_mut(vid(y)).value = 2.0;
    full_evaluate(*copy);
    // n = 4, p1 = 16, p2 = 7, p3 = 12
    REQUIRE(copy->node_value(top) == 16.0 + 7.0 + 12.0);
    copy->var_mut(vid(y)).value = 0.0;  // n = 2, p1 = 4, p2 = 5, p3 = 6
    REQUIRE(delta_evaluate(*copy, {vid(y)}) == 4.0 + 5.0 + 6.0);

    // G_v (constraints_of_var) is CSR too, in ascending constraint index.
    Model g;
    auto gx = g.float_var(0, 1);
    auto gy = g.float_var(0, 1);
    auto gc = g.constant(1.0);
    g.add_constraint(g.leq(g.sum({gx, gy}), gc));  // 0: x, y
    g.add_constraint(g.leq(gy, gc));               // 1: y
    g.add_constraint(g.leq(g.prod(gx, gx), gc));   // 2: x
    g.close();
    REQUIRE(as_vector(g.constraints_of_var(vid(gx))) == std::vector<int32_t>{0, 2});
    REQUIRE(as_vector(g.constraints_of_var(vid(gy))) == std::vector<int32_t>{0, 1});
    // The CSR bounds: one past the last id is out of range. A variable made
    // after the build used to sit past G_v's end too; the builders now refuse a
    // closed model instead (#173).
    REQUIRE_THROWS_AS(g.constraints_of_var(2), std::out_of_range);
    REQUIRE_THROWS_AS(g.parents(static_cast<int32_t>(g.num_nodes())), std::out_of_range);
    REQUIRE_THROWS_AS(g.dependents(2), std::out_of_range);
    REQUIRE_THROWS_AS(g.float_var(0, 1), std::logic_error);
    REQUIRE(g.num_vars() == 2);
}

TEST_CASE("Delta evaluation respects dependency order on a deep chain", "[dag]") {
    // delta_evaluate recomputes the dirty set in topological order. It used to
    // get that order by walking the WHOLE topo order and testing a flag, which
    // is O(all nodes) per call -- ~2M flag tests per move on the largest MIPfeas
    // instance, for a dirty set of a few dozen. It now sorts the dirty list by
    // topological position instead, so this pins the property the sort has to
    // preserve: a node must never be recomputed before the node it reads.
    //
    // A deep chain is what makes a wrong order observable. The BFS that marks
    // the dirty set enqueues parents in discovery order, which for a chain is
    // already topological -- so the check that bites is a DIAMOND, where the two
    // sides are discovered in one order and must be evaluated in another.
    Model m;
    auto x = m.float_var(0, 10);
    auto a = m.prod(x, m.constant(2.0));   // 2x
    auto b = m.sum({a, m.constant(1.0)});  // 2x + 1
    auto c = m.prod(b, b);                 // (2x + 1)^2
    auto d = m.sum({c, a});                // (2x + 1)^2 + 2x
    auto e = m.prod(d, m.constant(3.0));   // 3 * that
    m.minimize(e);
    m.close();

    m.var_mut(vid(x)).value = 1.0;
    full_evaluate(m);
    REQUIRE(m.node_value(e) == 3.0 * (9.0 + 2.0));

    m.var_mut(vid(x)).value = 4.0;
    const double delta_result = delta_evaluate(m, {vid(x)});
    // Every intermediate has to be recomputed before its reader, or the result
    // mixes the new x with a stale square.
    REQUIRE(delta_result == 3.0 * (81.0 + 8.0));
    REQUIRE(full_evaluate(m) == delta_result);
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
    REQUIRE(m.node_value(f) == 25.0);

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
    REQUIRE(m.node_value(a) == 30.0);
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
    REQUIRE(m.node_value(ls) == 30.0);  // 1+4+9+16
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
    REQUIRE(m.node_value(ls) == 3.0);

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
    REQUIRE(m.node_value(c) == 4.0);
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

    double expected = (2 * 3.0) + (2 * 2.0);  // 10
    REQUIRE_THAT(compute_partial(m, f, vid(x)), WithinAbs(expected, 1e-10));
}

TEST_CASE("Batch AD matches per-variable AD", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto z = m.float_var(-10, 10);
    auto two = m.constant(2);
    // f = x^2 + 2*x*y + sin(z)
    auto x_sq = m.pow_expr(x, two);
    auto xy = m.prod(x, y);
    auto two_xy = m.prod(two, xy);
    auto sin_z = m.sin_expr(z);
    auto f = m.sum({x_sq, two_xy, sin_z});
    m.minimize(f);
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 2.0;
    m.var_mut(vid(z)).value = 1.0;
    full_evaluate(m);

    auto all = compute_all_partials(m, f);
    REQUIRE(all.size() == 3);
    REQUIRE_THAT(all[vid(x)], WithinAbs(compute_partial(m, f, vid(x)), 1e-10));
    REQUIRE_THAT(all[vid(y)], WithinAbs(compute_partial(m, f, vid(y)), 1e-10));
    REQUIRE_THAT(all[vid(z)], WithinAbs(compute_partial(m, f, vid(z)), 1e-10));
}

// ---- SignPower / Tanh (issue #72: MINLPLib opsignpower / optanh) ----

// Central finite-difference of a single-output expr w.r.t. one variable.
static double fd_partial(Model& m, int32_t expr_id, int32_t var_id, double h) {
    double x0 = m.var(var_id).value;
    m.var_mut(var_id).value = x0 + h;
    full_evaluate(m);
    double fp = m.node_value(expr_id);
    m.var_mut(var_id).value = x0 - h;
    full_evaluate(m);
    double fm = m.node_value(expr_id);
    m.var_mut(var_id).value = x0;
    full_evaluate(m);
    return (fp - fm) / (2.0 * h);
}

TEST_CASE("SignPower evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto p = m.constant(3.0);
    auto sp = m.signpower_expr(x, p);
    m.minimize(m.sum({sp}));
    m.close();

    m.var_mut(vid(x)).value = 2.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node_value(sp), WithinAbs(8.0, 1e-10));  // sign(2)*|2|^3 = 8

    m.var_mut(vid(x)).value = -2.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node_value(sp), WithinAbs(-8.0, 1e-10));  // sign(-2)*|2|^3 = -8

    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    REQUIRE_THAT(m.node_value(sp), WithinAbs(0.0, 1e-10));
}

TEST_CASE("SignPower AD matches finite difference", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto p = m.constant(3.0);
    auto sp = m.signpower_expr(x, p);
    m.minimize(m.sum({sp}));
    m.close();

    for (double xv : {2.5, -2.5, 1.0, -1.0, 0.7}) {
        m.var_mut(vid(x)).value = xv;
        full_evaluate(m);
        // d/dx sign(x)|x|^3 = 3|x|^2
        REQUIRE_THAT(compute_partial(m, sp, vid(x)), WithinAbs(3.0 * xv * xv, 1e-8));
        REQUIRE_THAT(compute_partial(m, sp, vid(x)),
                     WithinAbs(fd_partial(m, sp, vid(x), 1e-5), 1e-4));
    }
}

TEST_CASE("Tanh evaluation", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto t = m.tanh_expr(x);
    m.minimize(m.sum({t}));
    m.close();

    for (double xv : {0.0, 1.0, -1.5, 3.0}) {
        m.var_mut(vid(x)).value = xv;
        full_evaluate(m);
        REQUIRE_THAT(m.node_value(t), WithinAbs(std::tanh(xv), 1e-10));
    }
}

TEST_CASE("Tanh AD matches finite difference", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto t = m.tanh_expr(x);
    m.minimize(m.sum({t}));
    m.close();

    for (double xv : {0.0, 1.0, -1.5, 2.0}) {
        m.var_mut(vid(x)).value = xv;
        full_evaluate(m);
        double th = std::tanh(xv);
        // d/dx tanh(x) = 1 - tanh^2(x)
        REQUIRE_THAT(compute_partial(m, t, vid(x)), WithinAbs(1.0 - (th * th), 1e-10));
        REQUIRE_THAT(compute_partial(m, t, vid(x)),
                     WithinAbs(fd_partial(m, t, vid(x), 1e-5), 1e-5));
    }
}

// ---- Cone-restricted reverse-mode AD ----
//
// compute_partial / compute_all_partials / compute_partials_sparse sweep only
// the cone of the expression, in reverse topological order, instead of the
// whole `topo_order()`. The claim is that this performs exactly the same
// floating-point operations in the same order, so every partial -- and every
// search trajectory built on them -- is bit-identical. The reference below is
// the full-order walk those functions used before, kept verbatim so the
// comparison is against the old arithmetic, not a re-derivation of it.

namespace {

std::vector<double> full_walk_all_partials(const Model& model, int32_t expr_id) {
    const size_t num_nodes = model.num_nodes();
    std::vector<double> adjoint(num_nodes + model.num_vars(), 0.0);
    adjoint[expr_id] = 1.0;
    const auto& order = model.topo_order();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const int32_t nid = *it;
        if (adjoint[nid] == 0.0) {
            continue;
        }
        const double adj = adjoint[nid];
        const auto& nd = model.node(nid);
        const ConstSpan<ChildRef> children = model.children(nd);
        for (int i = 0; i < static_cast<int>(children.size()); ++i) {
            const double ld = local_derivative(nd, i, model);
            const ChildRef& child = children[i];
            const size_t key = child.is_var ? num_nodes + static_cast<size_t>(child.id)
                                            : static_cast<size_t>(child.id);
            adjoint[key] += adj * ld;
        }
    }
    return {adjoint.begin() + static_cast<std::ptrdiff_t>(num_nodes), adjoint.end()};
}

uint64_t bits_of(double d) {
    uint64_t b = 0;
    std::memcpy(&b, &d, sizeof b);
    return b;
}

// Every entry point, against the reference, bit for bit, for one expression.
void require_bit_identical_partials(const Model& m, int32_t expr_id) {
    const std::vector<double> expected = full_walk_all_partials(m, expr_id);

    const std::vector<double> all = compute_all_partials(m, expr_id);
    REQUIRE(all.size() == expected.size());
    for (size_t v = 0; v < expected.size(); ++v) {
        INFO("expr " << expr_id << " var " << v);
        REQUIRE(bits_of(all[v]) == bits_of(expected[v]));
        REQUIRE(bits_of(compute_partial(m, expr_id, static_cast<int32_t>(v))) ==
                bits_of(expected[v]));
    }

    std::vector<std::pair<int32_t, double>> sparse{{-1, 99.0}};  // must be cleared
    compute_partials_sparse(m, expr_id, sparse);
    std::vector<uint8_t> seen(expected.size(), 0);
    for (const auto& [var, partial] : sparse) {
        INFO("expr " << expr_id << " sparse var " << var);
        REQUIRE(var >= 0);
        REQUIRE(static_cast<size_t>(var) < expected.size());
        REQUIRE(seen[static_cast<size_t>(var)] == 0);  // each variable at most once
        seen[static_cast<size_t>(var)] = 1;
        REQUIRE(partial != 0.0);
        REQUIRE(bits_of(partial) == bits_of(expected[static_cast<size_t>(var)]));
    }
    for (size_t v = 0; v < expected.size(); ++v) {
        INFO("expr " << expr_id << " var " << v << " missing from sparse");
        REQUIRE((seen[v] != 0) == (expected[v] != 0.0));
    }
}

// The sweep only sorts the cone when it is small against the DAG; on a toy
// model every cone is "large" and the full-order fallback runs instead. Pads the
// model with an unrelated constrained chain so the expressions under test take
// the sorted-cone route. Returns the chain's constraint, which is itself a
// whole-chain cone for the fallback route.
int32_t pad_with_unrelated_chain(Model& m, int length) {
    int32_t node = m.float_var(-1.0, 1.0);
    for (int i = 0; i < length; ++i) {
        node = m.sin_expr(node);
    }
    const int32_t row = m.leq(node, m.constant(1.0));
    m.add_constraint(row);
    return row;
}

}  // namespace

TEST_CASE("cone AD is bit-identical to the full-order walk on nonlinear shared DAGs", "[dag]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto z = m.float_var(0.1, 10);
    auto w = m.float_var(-10, 10);  // appears in no expression below but one
    auto three = m.constant(3.0);
    // Shared subexpression `xy` reached along three paths, and x along four.
    auto xy = m.prod(x, y);
    auto s = m.sin_expr(xy);
    auto e = m.exp_expr(m.div_expr(xy, z));
    auto l = m.log_expr(m.sum({z, m.prod(x, x)}));
    auto p = m.pow_expr(m.sum({x, y, xy}), three);
    auto f = m.sum({s, e, l, p, m.tanh_expr(m.neg(y)), m.sqrt_expr(z)});
    auto g = m.leq(m.sum({f, m.prod(xy, e)}), m.constant(1.0));
    auto h = m.leq(w, m.constant(0.0));
    m.add_constraint(g);
    m.add_constraint(h);
    pad_with_unrelated_chain(m, 400);
    m.minimize(f);
    m.close();

    for (const auto& pt : std::vector<std::vector<double>>{
             {0.3, -0.7, 1.1, 0.0}, {1.2, 0.4, 2.5, -1.0}, {-0.9, 1.3, 0.6, 2.0}}) {
        m.var_mut(vid(x)).value = pt[0];
        m.var_mut(vid(y)).value = pt[1];
        m.var_mut(vid(z)).value = pt[2];
        m.var_mut(vid(w)).value = pt[3];
        full_evaluate(m);
        for (int32_t node = 0; node < static_cast<int32_t>(m.num_nodes()); ++node) {
            require_bit_identical_partials(m, node);
        }
    }
}

TEST_CASE("cone AD keeps the zero-adjoint skip for cancelled and zero-derivative paths", "[dag]") {
    Model m;
    auto x = m.float_var(-1000, 1000);
    auto y = m.float_var(-10, 10);
    auto z = m.float_var(-10, 10);
    // u's adjoint is 1 + (-1) == 0.0 exactly, so nothing below u is propagated:
    // d/dx of (u - u) is 0 and x must be absent from the sparse result. The skip
    // is load-bearing, not just a saving: u = exp(800) overflows, so its local
    // derivative is +inf, and propagating the zero adjoint would write
    // 0 * inf = NaN into x's partial.
    auto u = m.exp_expr(x);
    auto cancelled = m.sum({u, m.neg(u)});
    // y * z at z == 0: the local derivative w.r.t. y is 0.0 -- y is written but
    // stays zero, so it too is absent from the sparse result.
    auto yz = m.prod(y, z);
    auto f = m.sum({cancelled, yz});
    const int32_t chain = pad_with_unrelated_chain(m, 400);
    m.minimize(f);
    m.close();
    m.var_mut(vid(x)).value = 800.0;
    m.var_mut(vid(y)).value = 2.0;
    m.var_mut(vid(z)).value = 0.0;
    full_evaluate(m);

    REQUIRE(compute_partial(m, f, vid(x)) == 0.0);
    REQUIRE(compute_partial(m, f, vid(y)) == 0.0);
    REQUIRE(compute_partial(m, f, vid(z)) == 2.0);
    std::vector<std::pair<int32_t, double>> sparse;
    compute_partials_sparse(m, f, sparse);
    REQUIRE(sparse == std::vector<std::pair<int32_t, double>>{{vid(z), 2.0}});
    require_bit_identical_partials(m, f);
    require_bit_identical_partials(m, cancelled);
    require_bit_identical_partials(m, chain);
}

TEST_CASE("cone AD lists a variable once when its adjoint cancels and is touched again", "[dag]") {
    // f = x - (x + 3x). Reverse order visits h = -k before q = 3x, so x's adjoint
    // goes 1 -> 0 (at k: 1 + -1) -> -3 (at q), and `written` lists x twice. All
    // the arithmetic is exact. The sparse result must still hold x once.
    Model d;
    const int32_t dx = d.float_var(-5.0, 5.0);
    const int32_t q = d.prod(d.constant(3.0), dx);
    const int32_t k = d.sum({dx, q});
    const int32_t dh = d.neg(k);
    const int32_t df = d.sum({dx, dh});
    pad_with_unrelated_chain(d, 400);
    d.minimize(df);
    d.close();
    d.var_mut(vid(dx)).value = 1.0;
    full_evaluate(d);
    std::vector<std::pair<int32_t, double>> dup;
    compute_partials_sparse(d, df, dup);
    REQUIRE(dup == std::vector<std::pair<int32_t, double>>{{vid(dx), -3.0}});
    require_bit_identical_partials(d, df);
}

TEST_CASE("cone AD on an unclosed model returns zeros instead of sorting by no position", "[dag]") {
    // Before close() there is no topological order and no position to sort a
    // cone by. The sweep must take the full-order walk -- over an empty order,
    // the all-zero answer it always gave -- rather than read `topo_pos` past its
    // end, which is what a small cone on the sorted route would do.
    Model m;
    const int32_t x = m.float_var(-1.0, 1.0);
    const int32_t y = m.float_var(-1.0, 1.0);
    const int32_t row = m.leq(m.sum({m.prod(x, y), m.sin_expr(x)}), m.constant(1.0));
    m.add_constraint(row);
    pad_with_unrelated_chain(m, 400);  // so the row's cone would fit the sorted route
    REQUIRE(m.topo_order().empty());
    REQUIRE(compute_partial(m, row, vid(x)) == 0.0);
    REQUIRE(compute_all_partials(m, row) == std::vector<double>(m.num_vars(), 0.0));
    std::vector<std::pair<int32_t, double>> sparse;
    compute_partials_sparse(m, row, sparse);
    REQUIRE(sparse.empty());
}

TEST_CASE("an AD call from inside a CustomInvariant partial is refused", "[dag]") {
    // The sweep that calls `partial` owns the thread's AD scratch; a nested sweep
    // would push onto the cone the outer one is iterating. It must throw, and the
    // outer sweep's guard must leave the scratch clean for the next call.
    class NestedAd : public CustomInvariant {
    public:
        NestedAd(const Model* inner, int32_t inner_expr) : inner_(inner), inner_expr_(inner_expr) {}
        double evaluate(const InvariantInputs& in) override { return in.value(0); }
        double partial(const InvariantInputs& /*in*/, int32_t /*i*/) override {
            return compute_partial(*inner_, inner_expr_, 0);
        }
        [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
            return std::make_unique<NestedAd>(*this);
        }

    private:
        const Model* inner_;
        int32_t inner_expr_;
    };

    Model inner;
    const int32_t ix = inner.float_var(-1.0, 1.0);
    const int32_t iexpr = inner.sin_expr(ix);
    inner.minimize(iexpr);
    inner.close();
    full_evaluate(inner);

    Model outer;
    const int32_t a = outer.float_var(-1.0, 1.0);
    const int32_t c = outer.custom({a}, std::make_unique<NestedAd>(&inner, iexpr), "nested");
    const int32_t row = outer.leq(outer.sum({c, outer.sin_expr(a)}), outer.constant(1.0));
    outer.add_constraint(row);
    pad_with_unrelated_chain(outer, 400);  // sorted-cone route: the cone is being iterated
    outer.close();
    full_evaluate(outer);

    // The message, not just the type: without the refusal the nested sweep
    // corrupts the outer one's cone and a garbage node id throws
    // std::out_of_range -- itself a std::logic_error.
    const auto refused = ContainsSubstring("re-entered from inside a reverse-mode AD sweep");
    REQUIRE_THROWS_WITH(compute_partial(outer, row, vid(a)), refused);
    REQUIRE_THROWS_WITH(compute_all_partials(outer, row), refused);
    std::vector<std::pair<int32_t, double>> sparse;
    REQUIRE_THROWS_WITH(compute_partials_sparse(outer, row, sparse), refused);
    // Outside a sweep the same call is fine, and the scratch was left clean.
    REQUIRE(compute_partial(inner, iexpr, 0) == std::cos(inner.var(0).value));
    require_bit_identical_partials(inner, iexpr);
}

TEST_CASE("cone AD is bit-identical on a large linear model for small and whole-DAG cones",
          "[dag]") {
    // A MIP-shaped model: many rows over a shared pool of variables, so a row's
    // cone is a handful of nodes against thousands -- the sorted-cone route --
    // while the objective, summing every row, has a cone of nearly the whole DAG
    // and takes the full-order fallback. Both must match the reference.
    constexpr int kVars = 400;
    constexpr int kRows = 600;
    Model m;
    std::vector<int32_t> vars;
    vars.reserve(kVars);
    for (int i = 0; i < kVars; ++i) {
        vars.push_back(m.float_var(-5.0, 5.0));
    }
    std::vector<int32_t> lhs;
    std::vector<int32_t> rows;
    for (int r = 0; r < kRows; ++r) {
        std::vector<int32_t> terms;
        for (int k = 0; k < 4; ++k) {
            // Deterministic, non-trivial coefficients. Every third row repeats
            // its first variable as the last term, so one variable collects its
            // partial along two paths.
            const int step = 13 * ((r % 5) + 1);
            const int v = (k == 3 && r % 3 == 0) ? (r * 7) % kVars : ((r * 7) + (k * step)) % kVars;
            const double coef = (0.1 * static_cast<double>(((r * 31) + (k * 17)) % 23)) - 1.05;
            terms.push_back(m.prod(m.constant(coef), vars[static_cast<size_t>(v)]));
        }
        const int32_t row_lhs = m.sum(terms);
        lhs.push_back(row_lhs);
        const int32_t row = m.leq(row_lhs, m.constant(1.0));
        m.add_constraint(row);
        rows.push_back(row);
    }
    const int32_t obj = m.sum(lhs);
    m.minimize(obj);
    m.close();
    for (int i = 0; i < kVars; ++i) {
        m.var_mut(vid(vars[static_cast<size_t>(i)])).value =
            (0.01 * static_cast<double>(i % 97)) - 0.3;
    }
    full_evaluate(m);

    for (int r = 0; r < kRows; r += 37) {
        require_bit_identical_partials(m, rows[static_cast<size_t>(r)]);
        require_bit_identical_partials(m, lhs[static_cast<size_t>(r)]);
    }
    require_bit_identical_partials(m, obj);
}

TEST_CASE("a partial that throws does not poison later AD calls on the thread", "[dag]") {
    // The sweep's scratch is thread_local and is restored by a guard. Without it,
    // a throw from user code would leave adjoints and cone marks set, and every
    // later call on this thread would start from them.
    class ThrowingPartial : public CustomInvariant {
    public:
        double evaluate(const InvariantInputs& in) override { return in.value(0) + in.value(1); }
        double partial(const InvariantInputs& /*in*/, int32_t /*i*/) override {
            throw std::runtime_error("partial unavailable");
        }
        [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
            return std::make_unique<ThrowingPartial>(*this);
        }
    };

    Model bad;
    const int32_t a = bad.float_var(-5.0, 5.0);
    const int32_t b = bad.float_var(-5.0, 5.0);
    const int32_t c = bad.custom({a, b}, std::make_unique<ThrowingPartial>(), "throws");
    const int32_t bad_row = bad.leq(bad.sum({c, bad.sin_expr(a)}), bad.constant(1.0));
    bad.add_constraint(bad_row);
    pad_with_unrelated_chain(bad, 400);  // sorted-cone route: the throw lands mid-cone
    bad.close();
    full_evaluate(bad);
    REQUIRE_THROWS_AS(compute_partial(bad, bad_row, vid(a)), std::runtime_error);
    REQUIRE_THROWS_AS(compute_all_partials(bad, bad_row), std::runtime_error);
    std::vector<std::pair<int32_t, double>> sparse;
    REQUIRE_THROWS_AS(compute_partials_sparse(bad, bad_row, sparse), std::runtime_error);

    // Same shape without the custom node, so its node ids overlap the ones the
    // throwing sweeps touched. Both models are padded onto the sorted-cone route:
    // there the whole cone is marked before the throw, so a leaked `in_cone` mark
    // would make this model's collection skip a node and its partials go wrong.
    Model good;
    const int32_t x = good.float_var(-5.0, 5.0);
    const int32_t y = good.float_var(-5.0, 5.0);
    const int32_t s = good.sum({x, y});
    const int32_t row = good.leq(good.sum({s, good.sin_expr(x)}), good.constant(1.0));
    good.add_constraint(row);
    pad_with_unrelated_chain(good, 400);
    good.close();
    good.var_mut(vid(x)).value = 0.5;
    full_evaluate(good);
    require_bit_identical_partials(good, row);  // first: before any call clears a stale mark
    for (int32_t node = 0; node < static_cast<int32_t>(good.num_nodes()); ++node) {
        require_bit_identical_partials(good, node);
    }
}

// ---------------------------------------------------------------------------
// Incremental Sum on commit (#177)
// ---------------------------------------------------------------------------

namespace {

// Every node value of `m` against a from-scratch evaluation of a copy, to the bit.
void require_matches_full_evaluate(const Model& m) {
    Model fresh(m);
    full_evaluate(fresh);
    const std::vector<double>& got = m.node_values();
    const std::vector<double>& want = fresh.node_values();
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (bits_of(got[i]) != bits_of(want[i])) {
            FAIL("node " << i << " holds " << got[i] << ", a full evaluation gives " << want[i]);
        }
    }
}

void require_same_bits(const std::vector<double>& got, const std::vector<double>& want) {
    REQUIRE(got.size() == want.size());
    for (size_t i = 0; i < got.size(); ++i) {
        if (bits_of(got[i]) != bits_of(want[i])) {
            FAIL("node " << i << " holds " << got[i] << ", expected " << want[i]);
        }
    }
}

// A MIP-shaped model: rows in the three shapes mps_to_model writes -- `x`,
// `neg(x)`, `prod(constant(a), x)` -- each long and sharing columns with the
// others. Integral coefficients over Int columns, or fractional ones over
// Float columns plus one big-M term per row, which is where drift lives.
struct Rows {
    Model m;
    std::vector<int32_t> cols;  // variable handles
    std::vector<int32_t> rows;  // the Sum nodes
};

Rows make_rows(int n_cols, int n_rows, int row_len, bool integral, uint64_t seed) {
    Rows r;
    RNG rng(seed);
    for (int j = 0; j < n_cols; ++j) {
        r.cols.push_back(integral ? r.m.int_var(-20, 20) : r.m.float_var(-20.0, 20.0));
    }
    const int32_t big = integral ? r.m.int_var(0, 1) : r.m.float_var(0.0, 1.0);
    for (int i = 0; i < n_rows; ++i) {
        std::vector<int32_t> terms;
        terms.reserve(static_cast<size_t>(row_len) + 1);
        for (int k = 0; k < row_len; ++k) {
            const int32_t x = r.cols[static_cast<size_t>(((i * 7) + k) % n_cols)];
            const int64_t shape = rng.integers(0, 3);
            if (shape == 0) {
                terms.push_back(x);
            } else if (shape == 1) {
                terms.push_back(r.m.neg(x));
            } else {
                const double a =
                    integral ? static_cast<double>(rng.integers(-9, 10)) : rng.uniform(-9.0, 9.0);
                terms.push_back(r.m.prod(r.m.constant(a), x));
            }
        }
        if (!integral) {
            terms.push_back(r.m.prod(r.m.constant(1e9 + rng.uniform(0.0, 1.0)), big));
        }
        const int32_t row = r.m.sum(terms);
        r.rows.push_back(row);
        r.m.add_constraint(r.m.leq(row, r.m.constant(static_cast<double>(rng.integers(-5, 6)))));
    }
    r.m.minimize(r.m.sum({r.cols[0], r.cols[1]}));
    r.m.close();
    if (!integral) {
        r.m.var_mut(vid(big)).value = 1.0;
        full_evaluate(r.m);
    }
    return r;
}

double random_value(RNG& rng, bool integral) {
    return integral ? static_cast<double>(rng.integers(-20, 21)) : rng.uniform(-20.0, 20.0);
}

void commit(Model& m, int32_t v, double value) {
    const double old_value = m.var(v).value;
    m.var_mut(v).value = value;
    commit_scalar_move(m, v, old_value);
}

// |row - its terms summed exactly|, the latter in long double from the terms'
// stored values (a term is a variable or a Neg/Prod, never itself drifted).
double row_drift(const Model& m, int32_t row) {
    long double exact = 0.0L;
    for (const ChildRef& c : m.children(m.node(row))) {
        exact += c.is_var ? m.var(c.id).value : m.node_values()[static_cast<size_t>(c.id)];
    }
    return static_cast<double>(std::fabs(static_cast<long double>(m.node_value(row)) - exact));
}

}  // namespace

TEST_CASE("commit_scalar_move moves a row by its changed term, exactly on integral data",
          "[dag][incremental_sum]") {
    // The regression test for the cost: once a row has been re-summed, a commit
    // must not re-sum it again until the period runs out. Red on the pre-#177
    // walk, which re-summed every dirty Sum -- `incremental` stays 0 there. And
    // on integral data every partial sum is exact, so the values are the
    // re-sum's to the bit however many updates they carry.
    Rows r = make_rows(40, 30, 25, /*integral=*/true, 5);
    Model& m = r.m;
    for (const int32_t row : r.rows) {
        REQUIRE(m.node(row).incremental_sum == 1);
    }
    RNG rng(11);
    incremental_sum_counters() = IncrementalSumCounters{};
    for (int step = 0; step < 3000; ++step) {
        commit(m, vid(r.cols[static_cast<size_t>(rng.integers(0, 40))]), random_value(rng, true));
        if (step % 250 == 0) {
            require_matches_full_evaluate(m);
        }
    }
    require_matches_full_evaluate(m);
    const IncrementalSumCounters counts = incremental_sum_counters();
    CHECK(counts.incremental > 10 * counts.resummed);
}

TEST_CASE("only a Sum with no Sum term and no repeated term is incremental",
          "[dag][incremental_sum]") {
    Model m;
    const int32_t a = m.int_var(-10, 10);
    const int32_t b = m.int_var(-10, 10);
    const int32_t f = m.float_var(-10.0, 10.0);
    const int32_t plain = m.sum({a, m.neg(b), m.prod(m.constant(0.3), f), m.sin_expr(f)});
    const int32_t nested = m.sum({plain, b});
    const int32_t repeated = m.sum({a, a, b});
    for (const int32_t row : {plain, nested, repeated}) {
        m.add_constraint(m.leq(row, m.constant(0.0)));
    }
    m.minimize(m.sum({a, b}));
    m.close();
    CHECK(m.node(plain).incremental_sum == 1);
    CHECK(m.node(nested).incremental_sum == 0);
    CHECK(m.node(repeated).incremental_sum == 0);
}

TEST_CASE("drift stays within the period's bound, and re-grounding removes it",
          "[dag][incremental_sum]") {
    // A big-M term keeps each row near 1e9, where one rounding is ~6e-8, and the
    // fractional terms keep every update inexact. Without the periodic re-sum
    // the error is a random walk over the whole run; with it, a row carries at
    // most kIncrementalSumPeriod roundings of a value this size.
    Rows r = make_rows(20, 8, 12, /*integral=*/false, 3);
    Model& m = r.m;
    // Every row stays in [2^29, 2^30), where one ulp is 2^-23 and one rounding at
    // most half that; a term's own difference rounds at a far smaller scale. So
    // a whole ulp per update is a safe per-update bound.
    const double ulp = std::ldexp(1.0, -23);
    const double bound = kIncrementalSumPeriod * ulp;
    RNG rng(8);
    double worst = 0.0;
    for (int step = 0; step < 1600000; ++step) {
        const int32_t v = vid(r.cols[static_cast<size_t>(rng.integers(0, 20))]);
        commit(m, v, random_value(rng, false));
        if (step % 64 == 0) {
            for (const int32_t row : r.rows) {
                REQUIRE(m.node_value(row) >= 536870912.0);  // 2^29
                REQUIRE(m.node_value(row) < 1073741824.0);  // 2^30
                worst = std::max(worst, row_drift(m, row));
            }
        }
    }
    INFO("worst drift " << worst << " against a bound of " << bound);
    CHECK(worst > 0.0);  // the data really does drift, or this proves nothing
    CHECK(worst <= bound);
    REQUIRE(reground_incremental_sums(m));
    require_matches_full_evaluate(m);
    CHECK_FALSE(reground_incremental_sums(m));  // nothing left to do
}

TEST_CASE("a probe on a drifted state: identity scores 0, and Rollback restores every bit",
          "[dag][incremental_sum]") {
    Rows r = make_rows(20, 8, 12, /*integral=*/false, 4);
    Model& m = r.m;
    RNG rng(2);
    for (int step = 0; step < 2000; ++step) {
        commit(m, vid(r.cols[static_cast<size_t>(rng.integers(0, 20))]), random_value(rng, false));
    }
    double drift = 0.0;
    for (const int32_t row : r.rows) {
        drift = std::max(drift, row_drift(m, row));
    }
    REQUIRE(drift > 0.0);  // a drifted state, which is the case that matters

    ViolationManager vm(m);
    const std::vector<double> before = m.node_values();
    for (const int32_t x : r.cols) {
        const int32_t v = vid(x);
        // The value it already holds: nothing moves, so exactly 0. A probe that
        // re-summed would compare the exact sum against the drifted one.
        REQUIRE(vm.weighted_violation_delta(v, m.var(v).value) == 0.0);
        (void)vm.weighted_violation_delta(v, random_value(rng, false));
        // Re-evaluating on the way back would re-sum the probed rows, snapping
        // their drift: the state would no longer be the one the caller had.
        require_same_bits(m.node_values(), before);
    }
}

TEST_CASE("every other writer interleaves with incremental commits consistently",
          "[dag][incremental_sum]") {
    // Plain deltas snap their cone, full passes snap everything, probes leave
    // nothing behind. After a re-grounding, whatever the interleaving, the
    // values are a full evaluation's.
    Rows r = make_rows(30, 20, 18, /*integral=*/false, 9);
    Model& m = r.m;
    ViolationManager vm(m);
    RNG rng(3);
    for (int step = 0; step < 3000; ++step) {
        const int32_t v = vid(r.cols[static_cast<size_t>(rng.integers(0, 30))]);
        const double value = random_value(rng, false);
        const int64_t what = rng.integers(0, 6);
        if (what == 0) {
            (void)vm.weighted_violation_delta(v, value);
        } else if (what == 1) {
            m.var_mut(v).value = value;
            delta_evaluate(m, &v, 1);
        } else if (what == 2 && step % 50 == 0) {
            full_evaluate(m);
        } else {
            commit(m, v, value);
        }
        if (step % 100 == 0) {
            (void)reground_incremental_sums(m);
            require_matches_full_evaluate(m);
        }
    }
}

TEST_CASE("a model copy carries the drift record with the node values", "[dag][incremental_sum]") {
    Rows r = make_rows(12, 6, 10, /*integral=*/false, 2);
    RNG rng(6);
    for (int step = 0; step < 200; ++step) {
        commit(r.m, vid(r.cols[static_cast<size_t>(rng.integers(0, 12))]),
               random_value(rng, false));
    }
    Model copy(r.m);
    REQUIRE(reground_incremental_sums(copy));
    require_matches_full_evaluate(copy);
}
