#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <limits>

using namespace cbls;

TEST_CASE("JumpTable lazy invalidation", "[fj][jumptable]") {
    JumpTable jt(3);
    REQUIRE_FALSE(jt.valid(0));
    REQUIRE_FALSE(jt.valid(2));

    jt.set(1, 4.5, 2.0);
    REQUIRE(jt.valid(1));
    REQUIRE(jt.jump_value(1) == 4.5);
    REQUIRE(jt.score(1) == 2.0);

    jt.invalidate(1);
    REQUIRE_FALSE(jt.valid(1));

    jt.set(0, 1.0, 1.0);
    jt.set(2, 2.0, 3.0);
    jt.invalidate_all();
    REQUIRE_FALSE(jt.valid(0));
    REQUIRE_FALSE(jt.valid(2));
}

TEST_CASE("gls_update_weights: decay then bump violated", "[fj][gls]") {
    Model m;
    auto x = m.float_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(2.0)));  // c0: x <= 2
    m.add_constraint(m.leq(x, m.constant(9.0)));  // c1: x <= 9
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 5.0;  // c0 violated (3), c1 satisfied
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.weights[0] == 1.0);
    REQUIRE(vm.weights[1] == 1.0);

    gls_update_weights(vm, 0.9);
    // c0 violated: 1*0.9 + 1 = 1.9; c1 satisfied: 1*0.9 = 0.9
    REQUIRE(std::abs(vm.weights[0] - 1.9) < 1e-12);
    REQUIRE(std::abs(vm.weights[1] - 0.9) < 1e-12);

    SECTION("masked (weight 0) constraints stay 0 even if violated") {
        vm.weights[0] = 0.0;
        gls_update_weights(vm, 0.95);
        REQUIRE(vm.weights[0] == 0.0);  // violated but masked -> never bumped
    }
}

TEST_CASE("compute_var_jump: convex continuous converges to argmin", "[fj][jump]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto diff = m.sum({x, m.constant(-5.0)});                                     // x - 5
    m.add_constraint(m.leq(m.pow_expr(diff, m.constant(2.0)), m.constant(0.0)));  // (x-5)^2 <= 0
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;  // violation 25
    full_evaluate(m);
    ViolationManager vm(m);

    JumpResult r = compute_var_jump(m, vm, vid(x));
    REQUIRE(std::abs(r.jump_value - 5.0) < 1e-3);  // golden section finds the min at x=5
    REQUIRE(r.score > 24.0);                       // reduction ~25
}

TEST_CASE("compute_var_jump: linear / bool / int", "[fj][jump]") {
    SECTION("linear float") {
        Model m;
        auto x = m.float_var(0, 10);
        m.add_constraint(m.leq(x, m.constant(3.0)));  // x <= 3
        m.minimize(m.sum({x}));
        m.close();
        m.var_mut(vid(x)).value = 8.0;  // violation 5
        full_evaluate(m);
        ViolationManager vm(m);
        JumpResult r = compute_var_jump(m, vm, vid(x));
        REQUIRE(r.jump_value <= 3.0 + 1e-6);
        REQUIRE(std::abs(r.score - 5.0) < 1e-3);
    }
    SECTION("bool") {
        Model m;
        auto b = m.bool_var();
        m.add_constraint(m.leq(b, m.constant(0.0)));  // b must be 0
        m.minimize(m.sum({b}));
        m.close();
        m.var_mut(vid(b)).value = 1.0;
        full_evaluate(m);
        ViolationManager vm(m);
        JumpResult r = compute_var_jump(m, vm, vid(b));
        REQUIRE(r.jump_value == 0.0);
        REQUIRE(std::abs(r.score - 1.0) < 1e-12);
    }
    SECTION("int") {
        Model m;
        auto x = m.int_var(0, 10);
        m.add_constraint(m.leq(x, m.constant(3.0)));
        m.minimize(m.sum({x}));
        m.close();
        m.var_mut(vid(x)).value = 8.0;
        full_evaluate(m);
        ViolationManager vm(m);
        JumpResult r = compute_var_jump(m, vm, vid(x));
        REQUIRE(r.jump_value <= 3.0);
        REQUIRE(std::abs(r.score - 5.0) < 1e-9);
    }
}

TEST_CASE("compute_var_jump: score never negative, jump within domain", "[fj][jump]") {
    // Non-convex-ish violation; the result must still be a valid in-domain jump
    // with a non-negative score (never claims an improvement that does not exist).
    Model m;
    auto x = m.float_var(-3, 3);
    m.add_constraint(m.leq(m.sin_expr(x), m.constant(-0.5)));  // sin(x) <= -0.5
    m.minimize(m.sum({x}));
    m.close();
    m.var_mut(vid(x)).value = 0.0;  // sin(0)=0 > -0.5 -> violated
    full_evaluate(m);
    ViolationManager vm(m);
    JumpResult r = compute_var_jump(m, vm, vid(x));
    REQUIRE(r.jump_value >= -3.0);
    REQUIRE(r.jump_value <= 3.0);
    // sin(x) <= -0.5 is satisfiable in [-3,3] (e.g. x=-1 -> sin=-0.84), and from
    // x0=0 it is violated, so an improving jump exists and is found.
    REQUIRE(r.score > 0.0);
    REQUIRE(std::sin(r.jump_value) <= -0.5 + 1e-6);
}

TEST_CASE("FeasibilityJump finds feasible: continuous CSP", "[fj][run]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    m.add_constraint(m.leq(m.sum({x, y}), m.constant(10.0)));  // x + y <= 10
    m.add_constraint(m.geq(x, m.constant(3.0)));               // x >= 3
    m.add_constraint(m.geq(y, m.constant(3.0)));               // y >= 3
    m.minimize(m.sum({x, y}));
    m.close();

    ViolationManager vm(m);
    RNG rng(1);
    GFJConfig cfg;
    cfg.time_limit = 2.0;
    cfg.max_iterations = 100000;
    FeasibilityJump fj(m, vm, rng, cfg);
    REQUIRE(fj.run() == GFJStatus::Feasible);
    REQUIRE(vm.is_feasible());
}

TEST_CASE("FeasibilityJump finds feasible: boolean cardinality", "[fj][run]") {
    Model m;
    auto b1 = m.bool_var();
    auto b2 = m.bool_var();
    auto b3 = m.bool_var();
    auto total = m.sum({b1, b2, b3});
    m.add_constraint(m.geq(total, m.constant(2.0)));  // sum >= 2
    m.add_constraint(m.leq(total, m.constant(2.0)));  // sum <= 2  (exactly 2 on)
    m.minimize(m.sum({b1}));
    m.close();

    ViolationManager vm(m);
    RNG rng(7);
    GFJConfig cfg;
    cfg.max_iterations = 100000;
    cfg.time_limit = 2.0;
    FeasibilityJump fj(m, vm, rng, cfg);
    REQUIRE(fj.run() == GFJStatus::Feasible);
    REQUIRE(vm.is_feasible());
    double s = m.var(vid(b1)).value + m.var(vid(b2)).value + m.var(vid(b3)).value;
    REQUIRE(std::abs(s - 2.0) < 1e-9);
}

TEST_CASE("FeasibilityJump finds feasible: mixed linear + nonlinear (two-phase)", "[fj][run]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    m.add_constraint(m.geq(x, m.constant(2.0)));  // linear: x >= 2
    m.add_constraint(
        m.leq(m.pow_expr(y, m.constant(2.0)), m.constant(9.0)));  // nonlinear: y^2 <= 9
    m.add_constraint(m.leq(m.sum({x, y}), m.constant(12.0)));     // linear: x + y <= 12
    m.minimize(m.sum({x, y}));
    m.close();

    ViolationManager vm(m);
    RNG rng(3);
    GFJConfig cfg;
    cfg.two_phase = true;
    cfg.time_limit = 2.0;
    cfg.max_iterations = 100000;
    FeasibilityJump fj(m, vm, rng, cfg);
    REQUIRE(fj.run() == GFJStatus::Feasible);
    REQUIRE(vm.is_feasible());
}

TEST_CASE("FeasibilityJump returns Unsolved on contradiction within budget", "[fj][run]") {
    Model m;
    auto x = m.float_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(2.0)));  // x <= 2
    m.add_constraint(m.geq(x, m.constant(5.0)));  // x >= 5  (contradiction)
    m.minimize(m.sum({x}));
    m.close();

    ViolationManager vm(m);
    RNG rng(1);
    GFJConfig cfg;
    cfg.max_iterations = 500;  // bounded; must terminate, not hang
    FeasibilityJump fj(m, vm, rng, cfg);
    REQUIRE(fj.run() == GFJStatus::Unsolved);
    REQUIRE_FALSE(vm.is_feasible());
}

TEST_CASE("batch API drives objective minimization (Algorithm 6 mechanism)", "[fj][batch]") {
    // minimize x+y s.t. x+y >= 4, integer ; optimum obj = 4. An integer
    // objective makes the objective-as-constraint mechanism converge cleanly
    // (the continuous case relies on the InnerSolverHook for objective descent,
    // P4 — golden section sits on the feasible band rather than its lower edge).
    Model m;
    auto x = m.int_var(0, 5);
    auto y = m.int_var(0, 5);
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(4.0)));
    m.minimize(m.sum({x, y}));
    m.close();
    m.add_objective_soft_constraint();

    ViolationManager vm(m);
    RNG rng(5);
    GFJConfig cfg;
    cfg.time_limit = 3.0;  // safety bound
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);

    double best = std::numeric_limits<double>::infinity();
    bool have = false;
    int stagnation = 0;
    for (int b = 0; b < 3000; ++b) {
        bool feasible = fj.batch(200);
        if (feasible) {
            double obj = m.node(m.objective_id()).value;
            if (obj < best - 1e-9) {
                best = obj;
                have = true;
                double eps = 1e-6 * (std::abs(obj) + 1.0);
                m.set_objective_bound(obj - eps);
                vm.invalidate_cache();
                fj.reset_weights();
                stagnation = 0;
                continue;
            }
        }
        if (++stagnation >= 50) {
            fj.perturb(0.2);
            stagnation = 0;
        }
    }
    REQUIRE(have);
    REQUIRE(best >= 4.0 - 1e-6);  // cannot beat the true optimum
    REQUIRE(best < 4.0 + 0.05);   // reaches it
}
