// Equality-balance moves (#222, LS-IQCQP): GFJConfig::equality_balance /
// SearchConfig::fj_equality_balance. Default off, and the default path is
// pinned by the trajectory fences; what is pinned here is the step rule
// (nullspace, integrality, gcd), that a committed move keeps its equality
// exact and its columns in their domains, and that the flag reaches every FJ.
#include "test_helpers.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cbls;

namespace {

Variable column(VarType type, double lb, double ub) {
    Variable v;
    v.type = type;
    v.lb = lb;
    v.ub = ub;
    return v;
}

// FJ over `m` from its current assignment, one GLS iteration.
struct OneIteration {
    ViolationManager vm;
    RNG rng{1};
    FeasibilityJump fj;
    OneIteration(Model& m, bool on) : vm(m), fj(m, vm, rng, config(on)) {
        fj.begin(/*set_initial_x=*/false);
    }
    static GFJConfig config(bool on) {
        GFJConfig c;
        c.equality_balance = on;
        c.two_phase = false;
        return c;
    }
};

}  // namespace

TEST_CASE("equality balance is off by default (#222)", "[fj][equality_balance]") {
    REQUIRE_FALSE(GFJConfig{}.equality_balance);
    REQUIRE_FALSE(SearchConfig{}.fj_equality_balance);
    REQUIRE_FALSE(LNS{}.equality_balance());
}

TEST_CASE("balance_steps stays in the row's nullspace (#222)", "[fj][equality_balance]") {
    const Variable f = column(VarType::Float, -100.0, 100.0);
    const Variable i = column(VarType::Int, -100.0, 100.0);
    const Variable b = column(VarType::Bool, 0.0, 1.0);
    double dv = 0.0;
    double du = 0.0;

    SECTION("two Floats: the target exactly, the partner on the ratio") {
        REQUIRE(balance_steps(f, 2.0, f, -4.0, 3.0, dv, du));
        REQUIRE(dv == 3.0);
        REQUIRE(du == 1.5);
    }
    SECTION("two Ints: the nearest nonzero multiple of |a_u| / gcd") {
        // 4 dv + 6 du = 0: dv in steps of 3, du in steps of 2.
        REQUIRE(balance_steps(i, 4.0, i, 6.0, 5.0, dv, du));
        REQUIRE((dv == 6.0 && du == -4.0));
        REQUIRE(balance_steps(i, 4.0, i, 6.0, 1.0, dv, du));  // rounds to 0: one step
        REQUIRE((dv == 3.0 && du == -2.0));
        REQUIRE(balance_steps(i, 4.0, i, -6.0, -5.0, dv, du));
        REQUIRE((dv == -6.0 && du == -4.0));
        REQUIRE((4.0 * dv) + (-6.0 * du) == 0.0);
        REQUIRE(balance_steps(b, 1.0, b, 1.0, 1.0, dv, du));
        REQUIRE((dv == 1.0 && du == -1.0));
    }
    SECTION("two Ints over fractional slopes: only an integral partner step") {
        REQUIRE(balance_steps(i, 0.5, i, 1.5, 3.0, dv, du));
        REQUIRE((dv == 3.0 && du == -1.0));
        REQUIRE_FALSE(balance_steps(i, 0.5, i, 1.5, 2.0, dv, du));
    }
    SECTION("an Int against a Float keeps the Int step integral") {
        REQUIRE(balance_steps(i, 2.0, f, 3.0, 2.4, dv, du));
        REQUIRE((dv == 2.0 && du == -4.0 / 3.0));
        REQUIRE(balance_steps(f, 3.0, i, 2.0, 0.1, dv, du));  // du rounds away from 0
        REQUIRE((du == -1.0 && dv == 2.0 / 3.0));
    }
    SECTION("a zero slope or target is no move") {
        REQUIRE_FALSE(balance_steps(f, 0.0, f, 1.0, 1.0, dv, du));
        REQUIRE_FALSE(balance_steps(f, 1.0, f, 0.0, 1.0, dv, du));
        REQUIRE_FALSE(balance_steps(f, 1.0, f, 1.0, 0.0, dv, du));
    }
}

TEST_CASE("a balance move escapes a minimum and keeps the equality exact (#222)",
          "[fj][equality_balance]") {
    // x + y == 10 holds at (5, 5); x >= 8 is violated by 3. Moving x alone
    // trades 3 on one row for 3 on the other: no positive jump. Moving y with
    // it along x + y == 10 fixes x >= 8 and keeps the equality.
    for (const bool on : {false, true}) {
        Model m;
        auto x = m.int_var(0, 10);
        auto y = m.int_var(0, 10);
        const int32_t eq = m.eq_expr(m.sum({x, y}), m.constant(10.0));
        m.add_constraint(eq);
        m.add_constraint(m.geq(x, m.constant(8.0)));
        m.close();
        m.var_mut(vid(x)).value = 5.0;
        m.var_mut(vid(y)).value = 5.0;
        full_evaluate(m);
        OneIteration run(m, on);
        run.fj.batch(1);
        CAPTURE(on, m.var(vid(x)).value, m.var(vid(y)).value);
        REQUIRE(run.fj.balance_moves() == (on ? 1 : 0));
        if (on) {
            REQUIRE(m.var(vid(x)).value == 8.0);
            REQUIRE(m.var(vid(y)).value == 2.0);
            REQUIRE(m.node_value(eq) == 0.0);
            REQUIRE(run.vm.total_violation() == 0.0);
        } else {
            REQUIRE(m.var(vid(x)).value == 5.0);  // the bump, no move
        }
    }
}

TEST_CASE("a balance move between Ints takes the gcd step (#222)", "[fj][equality_balance]") {
    // 4x + 6y == 60 at (3, 8); x >= 5 violated by 2. x alone cannot move by
    // 2 and keep the row; the smallest balanced step is x + 3, y - 2.
    Model m;
    auto x = m.int_var(0, 20);
    auto y = m.int_var(0, 20);
    const int32_t eq = m.eq_expr(m.sum({m.prod(m.constant(4.0), x), m.prod(m.constant(6.0), y)}),
                                 m.constant(60.0));
    m.add_constraint(eq);
    m.add_constraint(m.geq(x, m.constant(5.0)));
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 8.0;
    full_evaluate(m);
    OneIteration run(m, true);
    run.fj.batch(1);
    REQUIRE(run.fj.balance_moves() == 1);
    REQUIRE(m.var(vid(x)).value == 6.0);
    REQUIRE(m.var(vid(y)).value == 6.0);
    REQUIRE(m.node_value(eq) == 0.0);
}

TEST_CASE("a balance move never leaves a domain (#222)", "[fj][equality_balance]") {
    // x + y == 10 at (8, 2) with y <= 2; x <= 5 violated by 3. The balanced
    // move x - 3, y + 3 takes y to 5, outside its domain: no move.
    Model m;
    auto x = m.int_var(0, 10);
    auto y = m.int_var(0, 2);
    m.add_constraint(m.eq_expr(m.sum({x, y}), m.constant(10.0)));
    m.add_constraint(m.leq(x, m.constant(5.0)));
    m.close();
    m.var_mut(vid(x)).value = 8.0;
    m.var_mut(vid(y)).value = 2.0;
    full_evaluate(m);
    OneIteration run(m, true);
    run.fj.batch(1);
    REQUIRE(run.fj.balance_moves() == 0);
    REQUIRE(m.var(vid(y)).value <= 2.0);
}

TEST_CASE("the equality-balance flag reaches solve() and the LNS repair (#222)",
          "[fj][equality_balance]") {
    // Twenty copies of a nonlinear variant of the first case, which bound
    // propagation leaves alone: 20 x_i + 20 y_i == 200 at (5, 5), x_i^2 >= 64.
    // x alone gains 39 on the square and loses 60 on the equality.
    auto build = [](Model& m) {
        for (int i = 0; i < 20; ++i) {
            auto x = m.int_var(0, 10);
            auto y = m.int_var(0, 10);
            m.add_constraint(
                m.eq_expr(m.sum({m.prod(m.constant(20.0), x), m.prod(m.constant(20.0), y)}),
                          m.constant(200.0)));
            m.add_constraint(m.geq(m.prod(x, x), m.constant(64.0)));
            m.var_mut(vid(x)).value = 5.0;
            m.var_mut(vid(y)).value = 5.0;
        }
        m.close();
    };
    SECTION("solve()") {
        std::array<int64_t, 2> iterations{};
        for (const bool on : {false, true}) {
            Model m;
            build(m);
            SearchConfig config;
            config.max_iterations = 20'000;
            config.skip_init = true;
            config.use_compound_moves = false;  // FJ batches: the move lives there
            config.fj_equality_balance = on;
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
            iterations[static_cast<size_t>(on)] = fj_nl_initialize(
                m, vm, 20'000, &rng, 0.0, kDefaultFeasibilityTolerance, false, false, on);
            CAPTURE(on, iterations[static_cast<size_t>(on)]);
            REQUIRE(vm.total_violation() <= kDefaultFeasibilityTolerance);
        }
        REQUIRE(iterations[1] < iterations[0]);
    }
    SECTION("LNS::set_equality_balance reaches the repair FJ") {
        // Every copy has one solution in its domain, so both arms end at it;
        // what tells them apart is the work: the balance move's draws come
        // from the repair's RNG, so the stream it leaves behind differs only
        // if the repair's FJ took the flag.
        std::array<int64_t, 2> next_draw{};
        for (const bool on : {false, true}) {
            Model m;
            build(m);
            full_evaluate(m);
            ViolationManager vm(m);
            RNG rng(1);
            LNS lns;
            lns.set_equality_balance(on);
            REQUIRE(lns.equality_balance() == on);
            lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0);
            REQUIRE(vm.total_violation() <= kDefaultFeasibilityTolerance);
            next_draw[static_cast<size_t>(on)] = rng.integers(0, int64_t{1} << 40);
        }
        REQUIRE(next_draw[0] != next_draw[1]);
    }
}
