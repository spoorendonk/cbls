#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include "test_helpers.h"
#include <cmath>

using namespace cbls;

TEST_CASE("solve with nullptr hook is regression-safe", "[inner_solver]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    auto result = solve(m, 2.0, 42, true, nullptr);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 1.0);
}

TEST_CASE("FloatIntensifyHook improves Float vars", "[inner_solver]") {
    // min x^2 + y^2 s.t. x + y >= 1  (i.e. 1 - x - y <= 0)
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);

    // constraint: 1 - x - y <= 0
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));
    // objective: x^2 + y^2
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    // Start at a feasible but suboptimal point
    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);

    ViolationManager vm(m);
    double before_aug = vm.augmented_objective();

    FloatIntensifyHook hook;
    hook.max_sweeps = 5;
    hook.step_size = 0.1;
    auto hr = hook.solve(m, vm);

    double after_aug = vm.augmented_objective();
    // Hook should improve the augmented objective
    REQUIRE(after_aug < before_aug);
    // Result should contain Float var assignments
    REQUIRE_FALSE(hr.assignments.empty());
}

TEST_CASE("FloatIntensifyHook with infeasible start (no objective)", "[inner_solver]") {
    // Pure feasibility: x + y >= 3 (no objective)
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto three = m.constant(3.0);

    m.add_constraint(m.sum({three, m.prod(neg1, x), m.prod(neg1, y)}));
    m.close();

    // Start infeasible: x=0, y=0, constraint = 3 > 0
    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);

    ViolationManager vm(m);
    REQUIRE_FALSE(vm.is_feasible());
    double before_aug = vm.augmented_objective();

    FloatIntensifyHook hook;
    hook.solve(m, vm);

    double after_aug = vm.augmented_objective();
    // Newton steps should reduce violation (no objective to counterbalance)
    REQUIRE(after_aug < before_aug);
}

TEST_CASE("solve with FloatIntensifyHook improves mixed problem", "[inner_solver]") {
    // Bool b, Float x in [0,10], constraint: b + x >= 3, min x
    Model m;
    auto b = m.bool_var();
    auto x = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto three = m.constant(3.0);

    m.add_constraint(m.sum({three, m.prod(neg1, b), m.prod(neg1, x)}));
    m.minimize(m.sum({x}));
    m.close();

    FloatIntensifyHook hook;
    auto result = solve(m, 2.0, 42, true, &hook);
    REQUIRE(result.feasible);
    // With hook, should find good solution (x=2 when b=1, or x=3 when b=0)
    REQUIRE(result.objective <= 3.5);
}
