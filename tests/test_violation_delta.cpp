#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <stdexcept>

using namespace cbls;

// Compares the no-commit delta API against the naive
// "set -> full_evaluate -> diff -> restore" recompute, and checks the
// no-commit invariant and sparsity. `handle` is a variable handle, `j` the
// candidate value.
static void check_delta(Model& m, ViolationManager& vm, int32_t handle, double j) {
    const int32_t id = vid(handle);
    const size_t nc = m.constraint_ids().size();

    // Baseline from current assignment.
    full_evaluate(m);
    vm.invalidate_cache();
    const double base_total = vm.total_violation();
    std::vector<double> base_viol(nc);
    for (size_t i = 0; i < nc; ++i) {
        base_viol[i] = vm.constraint_violation(static_cast<int>(i));
    }

    // API (must not commit).
    auto pcd = m.per_constraint_violation_delta(id, j);
    const double wdelta = vm.weighted_violation_delta(id, j);

    // No-commit invariant: state is exactly as before.
    vm.invalidate_cache();
    REQUIRE(std::abs(vm.total_violation() - base_total) < 1e-12);

    // Naive recompute.
    const double old_val = m.var(id).value;
    m.var_mut(id).value = j;
    full_evaluate(m);
    vm.invalidate_cache();
    const double new_total = vm.total_violation();
    std::vector<double> new_viol(nc);
    for (size_t i = 0; i < nc; ++i) {
        new_viol[i] = vm.constraint_violation(static_cast<int>(i));
    }
    m.var_mut(id).value = old_val;
    full_evaluate(m);
    vm.invalidate_cache();

    // Weighted delta matches naive weighted-total difference.
    REQUIRE(std::abs(wdelta - (new_total - base_total)) < 1e-9);

    // Per-constraint deltas match naive per-constraint diffs.
    for (size_t i = 0; i < nc; ++i) {
        double naive = new_viol[i] - base_viol[i];
        double got = 0.0;
        for (const auto& [ci, d] : pcd) {
            if (ci == static_cast<int32_t>(i)) {
                got = d;
            }
        }
        REQUIRE(std::abs(got - naive) < 1e-9);
    }

    // Sparsity: every reported constraint is in G_v and actually changed.
    const auto& gv = m.constraints_of_var(id);
    for (const auto& [ci, d] : pcd) {
        REQUIRE(std::find(gv.begin(), gv.end(), ci) != gv.end());
        REQUIRE(d != 0.0);
    }
}

TEST_CASE("delta: linear constraints, multiple sharing a var", "[violation][delta]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto c3 = m.constant(-3.0);
    auto c7 = m.constant(-7.0);
    m.add_constraint(m.leq(x, m.constant(3.0)));  // c0: x <= 3   (G_x)
    m.add_constraint(m.geq(y, m.constant(8.0)));  // c1: y >= 8   (G_y only)
    m.add_constraint(m.sum({x, y, c3}));          // c2: x+y-3<=0 (G_x, G_y)
    m.add_constraint(m.sum({x, c7}));             // c3: x-7<=0   (G_x)
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 2.0;
    ViolationManager vm(m);

    check_delta(m, vm, x, 0.0);
    check_delta(m, vm, x, 9.0);
    check_delta(m, vm, y, 9.0);
    check_delta(m, vm, y, 0.0);

    // x is not in c1 (y-only); never reported for an x-probe.
    auto pcd = m.per_constraint_violation_delta(vid(x), 9.0);
    for (const auto& [ci, d] : pcd) {
        REQUIRE(ci != 1);
    }
}

TEST_CASE("delta: nonlinear arithmetic (prod, div, pow)", "[violation][delta]") {
    Model m;
    auto x = m.float_var(1, 5);
    auto y = m.float_var(1, 5);
    m.add_constraint(m.leq(m.prod(x, y), m.constant(6.0)));                    // x*y <= 6
    m.add_constraint(m.leq(m.div_expr(x, y), m.constant(1.0)));                // x/y <= 1
    m.add_constraint(m.leq(m.pow_expr(x, m.constant(2.0)), m.constant(9.0)));  // x^2 <= 9
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 2.0;
    m.var_mut(vid(y)).value = 2.0;
    ViolationManager vm(m);

    check_delta(m, vm, x, 4.0);
    check_delta(m, vm, x, 1.0);
    check_delta(m, vm, y, 5.0);
}

TEST_CASE("delta: transcendental (sin/cos/tan/exp/log/sqrt/abs)", "[violation][delta]") {
    Model m;
    auto x = m.float_var(0.1, 3.0);
    m.add_constraint(m.leq(m.sin_expr(x), m.constant(0.5)));
    m.add_constraint(m.leq(m.cos_expr(x), m.constant(0.5)));
    m.add_constraint(m.leq(m.tan_expr(x), m.constant(1.0)));
    m.add_constraint(m.leq(m.exp_expr(x), m.constant(5.0)));
    m.add_constraint(m.leq(m.log_expr(x), m.constant(0.5)));
    m.add_constraint(m.leq(m.sqrt_expr(x), m.constant(1.2)));
    m.add_constraint(m.leq(m.abs_expr(x), m.constant(1.0)));
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 1.0;
    ViolationManager vm(m);

    check_delta(m, vm, x, 0.5);
    check_delta(m, vm, x, 2.5);
    check_delta(m, vm, x, 0.2);
}

TEST_CASE("delta: min/max and if-then-else", "[violation][delta]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    m.add_constraint(m.leq(m.min_expr({x, y}), m.constant(2.0)));
    m.add_constraint(m.leq(m.max_expr({x, y}), m.constant(5.0)));
    m.add_constraint(m.leq(m.if_then_else(m.leq(x, m.constant(3.0)), x, y), m.constant(4.0)));
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 4.0;
    ViolationManager vm(m);

    check_delta(m, vm, x, 1.0);
    check_delta(m, vm, x, 7.0);
    check_delta(m, vm, y, 6.0);
}

TEST_CASE("delta: comparison-root and step (eq/neq/lt/gt) constraints", "[violation][delta]") {
    Model m;
    auto x = m.int_var(0, 10);
    m.add_constraint(m.eq_expr(x, m.constant(4.0)));  // |x-4|
    m.add_constraint(m.neq(x, m.constant(2.0)));      // step: 1 if x==2
    m.add_constraint(m.lt(x, m.constant(6.0)));       // x < 6
    m.add_constraint(m.gt(x, m.constant(1.0)));       // x > 1
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 4.0;
    ViolationManager vm(m);

    check_delta(m, vm, x, 2.0);
    check_delta(m, vm, x, 7.0);
    check_delta(m, vm, x, 0.0);
}

TEST_CASE("delta: bool variable", "[violation][delta]") {
    Model m;
    auto b = m.bool_var();
    m.add_constraint(m.leq(b, m.constant(0.0)));  // b must be 0
    m.minimize(m.sum({b}));
    m.close();

    m.var_mut(vid(b)).value = 0.0;
    ViolationManager vm(m);
    check_delta(m, vm, b, 1.0);

    m.var_mut(vid(b)).value = 1.0;
    full_evaluate(m);
    check_delta(m, vm, b, 0.0);
}

TEST_CASE("delta: weights are applied", "[violation][delta]") {
    Model m;
    auto x = m.float_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(2.0)));  // c0
    m.add_constraint(m.leq(x, m.constant(4.0)));  // c1
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 1.0;
    ViolationManager vm(m);
    vm.weights[0] = 3.0;
    vm.weights[1] = 5.0;
    vm.invalidate_cache();

    auto pcd = m.per_constraint_violation_delta(vid(x), 6.0);
    double expected = 0.0;
    for (const auto& [ci, d] : pcd) {
        expected += vm.weights[ci] * d;
    }
    REQUIRE(std::abs(vm.weighted_violation_delta(vid(x), 6.0) - expected) < 1e-12);
    // both constraints become violated (6>4>2), weights differ -> nonzero
    REQUIRE(vm.weighted_violation_delta(vid(x), 6.0) > 0.0);
}

TEST_CASE("delta: throws on List/Set variables", "[violation][delta]") {
    Model m;
    auto lst = m.list_var(4);
    auto st = m.set_var(4);
    auto x = m.float_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(2.0)));
    m.minimize(m.sum({x}));
    m.close();

    REQUIRE_THROWS_AS(m.per_constraint_violation_delta(vid(lst), 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(m.per_constraint_violation_delta(vid(st), 0.0), std::invalid_argument);
}

TEST_CASE("delta: variable in no constraint returns empty", "[violation][delta]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);  // unconstrained
    m.add_constraint(m.leq(x, m.constant(2.0)));
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(y)).value = 3.0;
    ViolationManager vm(m);
    REQUIRE(m.per_constraint_violation_delta(vid(y), 9.0).empty());
    REQUIRE(vm.weighted_violation_delta(vid(y), 9.0) == 0.0);
    REQUIRE(m.constraints_of_var(vid(y)).empty());
}
