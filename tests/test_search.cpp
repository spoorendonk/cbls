#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <cmath>

using namespace cbls;

static int32_t vid(int32_t handle) { return -(handle + 1); }

// Violation tests
TEST_CASE("No violation when feasible", "[violation]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    m.add_constraint(m.sum({x, neg5}));  // x - 5 <= 0
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.total_violation() == 0.0);
    REQUIRE(vm.is_feasible());
}

TEST_CASE("Violation when infeasible", "[violation]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    m.add_constraint(m.sum({x, neg5}));
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.total_violation() == 3.0);
    REQUIRE_FALSE(vm.is_feasible());
}

TEST_CASE("Augmented objective", "[violation]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    m.add_constraint(m.sum({x, neg5}));
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    full_evaluate(m);
    ViolationManager vm(m);
    // F = f + lambda*V = 8 + 1.0*3 = 11
    REQUIRE(vm.augmented_objective() == 11.0);
}

// Adaptive lambda tests
TEST_CASE("Lambda increases when infeasible", "[violation]") {
    AdaptiveLambda al(1.0);
    for (int i = 0; i < 11; ++i) {
        al.update(false, false);
    }
    REQUIRE(al.lambda_ > 1.0);
}

TEST_CASE("Lambda decreases when stuck feasible", "[violation]") {
    AdaptiveLambda al(1.0);
    for (int i = 0; i < 21; ++i) {
        al.update(true, false);
    }
    REQUIRE(al.lambda_ < 1.0);
}

// FJ-NL tests
TEST_CASE("FJ-NL finds feasibility simple", "[search]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    // constraint: 4 - x - y <= 0 (i.e. x + y >= 4)
    auto neg1_node = m.constant(-1.0);
    auto xy_sum = m.sum({x, y});
    auto neg_xy = m.prod(neg1_node, xy_sum);
    auto c = m.sum({neg1_node, neg_xy, m.constant(5.0)});
    m.add_constraint(c);
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE_FALSE(vm.is_feasible());

    RNG rng(42);
    fj_nl_initialize(m, vm, 1000, &rng);
    full_evaluate(m);
    REQUIRE((vm.is_feasible() || vm.total_violation() < 0.1));
}

TEST_CASE("FJ-NL finds feasibility bool", "[search]") {
    Model m;
    auto x = m.bool_var();
    auto y = m.bool_var();
    auto neg1 = m.constant(-1.0);
    auto neg_x = m.prod(neg1, x);
    auto neg_y = m.prod(neg1, y);
    auto one = m.constant(1.0);
    m.add_constraint(m.sum({one, neg_x, neg_y}));  // 1 - x - y <= 0
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE_FALSE(vm.is_feasible());

    RNG rng(42);
    fj_nl_initialize(m, vm, 100, &rng);
    full_evaluate(m);
    REQUIRE(vm.is_feasible());
}

// SA solver tests
TEST_CASE("SA unconstrained minimum", "[search]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    auto result = solve(m, 2.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 1.0);
}

TEST_CASE("SA constrained problem", "[search]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto three = m.constant(3.0);
    m.add_constraint(m.sum({three, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    auto result = solve(m, 3.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 5.0);
}

TEST_CASE("SA integer problem", "[search]") {
    Model m;
    auto x = m.int_var(0, 10);
    auto neg7 = m.constant(-7.0);
    m.minimize(m.abs_expr(m.sum({x, neg7})));
    m.close();

    auto result = solve(m, 2.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 2.0);
}

TEST_CASE("SA Rosenbrock 2D", "[search]") {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto one = m.constant(1.0);
    auto neg1 = m.constant(-1.0);
    auto two = m.constant(2.0);
    auto hundred = m.constant(100.0);

    auto one_minus_x = m.sum({one, m.prod(neg1, x)});
    auto term1 = m.pow_expr(one_minus_x, two);

    auto y_minus_x2 = m.sum({y, m.prod(neg1, m.pow_expr(x, two))});
    auto term2 = m.prod(hundred, m.pow_expr(y_minus_x2, two));

    m.minimize(m.sum({term1, term2}));
    m.close();

    auto result = solve(m, 5.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 50.0);
}

TEST_CASE("SA returns result", "[search]") {
    Model m;
    auto x = m.float_var(0, 1);
    m.minimize(m.sum({x}));
    m.close();

    auto result = solve(m, 0.5, 42);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.time_seconds > 0);
}

// LNS test
TEST_CASE("LNS basic", "[lns]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto five = m.constant(5.0);
    m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    m.var_mut(vid(y)).value = 8.0;
    full_evaluate(m);
    ViolationManager vm(m);

    LNS lns(0.5);
    RNG rng(42);
    lns.destroy_repair_cycle(m, vm, rng, 5);
    full_evaluate(m);
    // Just check it doesn't crash
}

// Solution pool test
TEST_CASE("SolutionPool ordering", "[pool]") {
    SolutionPool pool(3);
    Model::State empty_state;

    pool.submit({empty_state, 10.0, true});
    pool.submit({empty_state, 5.0, true});
    pool.submit({empty_state, 20.0, false});
    pool.submit({empty_state, 3.0, true});

    auto best = pool.best();
    REQUIRE(best.has_value());
    REQUIRE(best->objective == 3.0);
    REQUIRE(best->feasible);
    REQUIRE(pool.size() == 3);
}
