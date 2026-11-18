// Fix-propagate-repair as the LNS repair (#431): kept scalars fixed, one dive
// over the destroyed ones, then FJ.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/domain_propagation.h>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

// `pairs` rows x_i + y_i == 1 over Bools (a <= and a >= row, so every
// variable is locked both ways and no trivial rounding fixes it), x_i at `x`
// and y_i at `y`, closed and evaluated. A kept variable fixes its partner; a
// pair destroyed whole is branched. `minimize_x` adds the objective sum(x).
Model pair_model(int pairs, double x_value, double y_value, bool minimize_x = false) {
    Model m;
    std::vector<int32_t> xs;
    xs.reserve(pairs);
    for (int i = 0; i < pairs; ++i) {
        const int32_t x = m.bool_var();
        const int32_t y = m.bool_var();
        xs.push_back(x);
        m.add_constraint(m.leq(m.sum({x, y}), m.constant(1.0)));
        m.add_constraint(m.geq(m.sum({x, y}), m.constant(1.0)));
    }
    if (minimize_x) {
        m.minimize(m.sum(xs));
    }
    m.close();
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        m.var_mut(v).value = (v % 2 == 0) ? x_value : y_value;
    }
    full_evaluate(m);
    return m;
}

// Reaches the repair step itself, so a test can read what it wrote before FJ.
struct LnsProbe : LNS {
    using LNS::fpr_repair;
    using LNS::LNS;
};

std::vector<double> values(const Model& m) {
    std::vector<double> out;
    out.reserve(m.num_vars());
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        out.push_back(m.var(v).value);
    }
    return out;
}

}  // namespace

TEST_CASE("the LNS FPR repair dives from the kept values and rolls back a round that ties",
          "[lns][fpr]") {
    // x = 1, y = 0 is feasible with no objective: no repair can beat its key,
    // so every round is rejected and must restore the incumbent exactly.
    Model m = pair_model(40, 1.0, 0.0);
    ViolationManager vm(m);
    LNS lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);
    const std::vector<double> incumbent = values(m);

    for (int round = 0; round < 3; ++round) {
        CHECK_FALSE(lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0));
        CHECK(values(m) == incumbent);
    }
    CHECK(lns.fpr_repair_stats().dives == 3);
    CHECK(lns.fpr_repair_stats().kept_conflicts == 0);
    CHECK(lns.fpr_repair_stats().nodes > 0);
}

TEST_CASE("the LNS accepts a repair whose dive beats the incumbent's objective", "[lns][fpr]") {
    // Every x at 1 is the worst feasible point of min sum(x): with half the
    // variables destroyed, some whole pair comes back with x = 0.
    Model m = pair_model(40, 1.0, 0.0, /*minimize_x=*/true);
    ViolationManager vm(m);
    LNS lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);
    const std::pair<double, double> before = lns_acceptance_key(m, kDefaultFeasibilityTolerance);

    REQUIRE(lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0));
    CHECK(lns.fpr_repair_stats().dives == 1);
    const std::pair<double, double> after = lns_acceptance_key(m, kDefaultFeasibilityTolerance);
    CHECK(after.first == 0.0);
    CHECK(after.second < before.second);
}

TEST_CASE("a solve that throws still takes its repair store back from the LNS",
          "[search][lns][fpr]") {
    struct ThrowingLns : LNS {
        using LNS::LNS;
        bool destroy_repair(Model& /*model*/, ViolationManager& /*vm*/, RNG& /*rng*/,
                            double /*repair_time_limit*/) override {
            REQUIRE(repair_store() != nullptr);  // handed over before the kick
            // Widened by the solve's tolerance (#433).
            CHECK(repair_store()->row_tolerance() == 1e-5);
            throw std::runtime_error("repair failed");
        }
    };
    Model m = pair_model(20, 1.0, 1.0, /*minimize_x=*/true);
    SearchConfig cfg;
    cfg.batch_iterations = 1;
    cfg.perturbation_period = 1;
    cfg.max_iterations = 30;
    cfg.lns_repair_fpr = true;
    cfg.feasibility_tolerance = 1e-5;
    ThrowingLns lns(0.5);

    CHECK_THROWS_AS(
        solve(m, /*time_limit=*/0.0, 3, nullptr, &lns, /*lns_interval=*/1, nullptr, cfg),
        std::runtime_error);
    CHECK(lns.repair_store() == nullptr);
}

TEST_CASE("a kept value that breaks a linear row skips the dive, and FJ still repairs",
          "[lns][fpr]") {
    // All ones violates every row; with half destroyed some pair is kept
    // whole, so fixing the kept values conflicts. The round is then plain FJ,
    // which repairs the violation and is accepted on the key.
    Model m = pair_model(40, 1.0, 1.0);
    ViolationManager vm(m);
    LNS lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);

    CHECK(lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0));
    CHECK(lns.fpr_repair_stats().kept_conflicts == 1);
    CHECK(lns.fpr_repair_stats().dives == 0);
    CHECK(lns_acceptance_key(m, kDefaultFeasibilityTolerance).first == 0.0);
}

TEST_CASE("the LNS FPR dive stops at the repair deadline", "[lns][fpr][deadline]") {
    // Every pair destroyed whole is branched, so an unlimited dive pops many
    // nodes. A deadline already past when the dive starts stops it at its
    // first clock check (16 nodes).
    Model m = pair_model(400, 1.0, 0.0);
    ViolationManager vm(m);
    LNS lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);

    lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/1e-9);
    const int64_t limited = lns.fpr_repair_stats().nodes;
    CHECK(limited <= 16);
    // Limited first: a deadline left over from that round would stop this one.
    lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0);
    CHECK(lns.fpr_repair_stats().dives == 2);
    CHECK(lns.fpr_repair_stats().nodes - limited > 32);
}

TEST_CASE("the FPR repair writes propagated values and leaves no fixing behind", "[lns][fpr]") {
    // Pairs 0-9 lose only y; pairs 10-19 are destroyed whole. Every destroyed
    // value starts wrong, so a correct result can only come from the dive.
    Model m = pair_model(20, 1.0, 0.0);
    std::vector<int32_t> destroyed;
    for (int32_t pair = 0; pair < 20; ++pair) {
        if (pair >= 10) {
            destroyed.push_back(2 * pair);
            m.var_mut(2 * pair).value = 1.0;
        }
        destroyed.push_back((2 * pair) + 1);
        m.var_mut((2 * pair) + 1).value = 1.0;
    }
    full_evaluate(m);
    LnsProbe lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);

    REQUIRE(lns.fpr_repair(m, destroyed, rng, nullptr));
    for (int32_t pair = 0; pair < 20; ++pair) {
        INFO("pair " << pair);
        CHECK(m.var(2 * pair).value + m.var((2 * pair) + 1).value == 1.0);
        if (pair < 10) {
            CHECK(m.var(2 * pair).value == 1.0);
        }
    }

    // The opposite incumbent: a fixing left from the first round would refute it.
    std::vector<int32_t> ys;
    for (int32_t pair = 0; pair < 20; ++pair) {
        m.var_mut(2 * pair).value = 0.0;
        m.var_mut((2 * pair) + 1).value = 0.0;
        ys.push_back((2 * pair) + 1);
    }
    full_evaluate(m);
    REQUIRE(lns.fpr_repair(m, ys, rng, nullptr));
    CHECK(lns.fpr_repair_stats().kept_conflicts == 0);
    CHECK(lns.fpr_repair_stats().dives == 2);
    for (const int32_t y : ys) {
        CHECK(m.var(y).value == 1.0);
    }
}

TEST_CASE("kept floats hold within the tolerance, and a non-finite one skips the dive",
          "[lns][fpr]") {
    // a + b <= 1 overshot by 1e-7: feasible at the 1e-6 tolerance, refuted by
    // an exact fix of both unless the store is widened by it.
    Model m;
    const int32_t a = m.float_var(0.0, 10.0);
    const int32_t b = m.float_var(0.0, 10.0);
    const int32_t z = m.bool_var();
    // Past its upper bound by 5e-7, within the tolerance: held at the bound.
    const int32_t c = m.float_var(0.0, 1.0);
    m.add_constraint(m.leq(m.sum({a, b}), m.constant(1.0)));
    m.close();
    const int32_t a_id = handle_to_var_id(a);
    const std::vector<int32_t> destroyed = {handle_to_var_id(z)};
    m.var_mut(handle_to_var_id(c)).value = 1.0 + 5e-7;
    m.var_mut(a_id).value = 0.5;
    m.var_mut(handle_to_var_id(b)).value = 0.5 + 1e-7;
    full_evaluate(m);
    LnsProbe lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m, kDefaultFeasibilityTolerance));
    RNG rng(7);

    CHECK(lns.fpr_repair(m, destroyed, rng, nullptr));
    CHECK(lns.fpr_repair_stats().dives == 1);
    CHECK(lns.fpr_repair_stats().kept_conflicts == 0);

    // Past its bound by more than the tolerance: not held, a kept conflict.
    m.var_mut(handle_to_var_id(c)).value = 1.3;
    CHECK_FALSE(lns.fpr_repair(m, destroyed, rng, nullptr));
    CHECK(lns.fpr_repair_stats().kept_conflicts == 1);
    m.var_mut(handle_to_var_id(c)).value = 1.0;

    m.var_mut(a_id).value = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(lns.fpr_repair(m, destroyed, rng, nullptr));
    CHECK(lns.fpr_repair_stats().dives == 1);
    CHECK(lns.fpr_repair_stats().kept_conflicts == 2);
}

TEST_CASE("the FPR repair counts a refuted root and writes nothing", "[lns][fpr]") {
    // x + y >= 3 over two Bools: no assignment satisfies the row.
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(3.0)));
    m.close();
    full_evaluate(m);
    LnsProbe lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);

    CHECK_FALSE(lns.fpr_repair(m, {handle_to_var_id(x)}, rng, nullptr));
    CHECK(lns.fpr_repair_stats().root_refuted == 1);
    CHECK(lns.fpr_repair_stats().dives == 0);
}

TEST_CASE("the FPR repair keeps a destroyed integer's value though one side of it is unlocked",
          "[lns][fpr]") {
    // min sum(c_j x_j) over x_0 + ... + x_5 >= 1: no x has an up-lock, so a
    // trivial rounding would set every destroyed x to 1 whatever its cost (#433).
    Model m;
    std::vector<int32_t> xs;
    std::vector<int32_t> costed;
    for (int j = 0; j < 6; ++j) {
        xs.push_back(m.bool_var());
        costed.push_back(m.prod(m.constant(1.0 + j), xs.back()));
    }
    m.add_constraint(m.geq(m.sum(xs), m.constant(1.0)));
    m.minimize(m.sum(costed));
    m.close();
    const std::vector<double> randomised = {1.0, 0.0, 1.0, 0.0, 1.0, 0.0};
    std::vector<int32_t> destroyed;
    for (int j = 0; j < 6; ++j) {
        const int32_t v = handle_to_var_id(xs[static_cast<size_t>(j)]);
        m.var_mut(v).value = randomised[static_cast<size_t>(j)];
        if (j > 0) {
            destroyed.push_back(v);
        }
    }
    full_evaluate(m);
    LnsProbe lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);

    REQUIRE(lns.fpr_repair(m, destroyed, rng, nullptr));
    CHECK(values(m) == randomised);
}

TEST_CASE("the FPR repair keeps a destroyed float's value whatever its objective partial",
          "[lns][fpr]") {
    // min (x - 3)^2: the partial is 8 at x = 7, where the engine is built. A
    // cost-preferred fill would write x's lower bound every round (#433).
    Model m;
    const int32_t x = m.float_var(0.0, 10.0);
    m.minimize(m.pow_expr(m.sum({x, m.constant(-3.0)}), m.constant(2.0)));
    m.close();
    const int32_t xv = handle_to_var_id(x);
    LnsProbe lns(0.5);
    lns.set_repair_store(std::make_shared<const LinearRowStore>(m));
    RNG rng(7);
    // One engine for both rounds: the second runs on the first's costs.
    for (const double randomised : {7.0, 2.0}) {
        INFO("x randomised to " << randomised);
        m.var_mut(xv).value = randomised;
        full_evaluate(m);
        REQUIRE(lns.fpr_repair(m, {xv}, rng, nullptr));
        CHECK(m.var(xv).value == randomised);
    }
}

TEST_CASE("a kept integer a row accepts only within the tolerance holds on a widened store",
          "[lns][fpr]") {
    // 3.0000005 x <= 3 at x = 1: residual 5e-7, feasible at the 1e-6
    // tolerance, but fixing x = 1 against the exact row conflicts (#433).
    Model m;
    const int32_t x = m.int_var(0, 5);
    const int32_t z = m.bool_var();
    m.add_constraint(m.leq(m.prod(m.constant(3.0000005), x), m.constant(3.0)));
    m.close();
    m.var_mut(handle_to_var_id(x)).value = 1.0;
    full_evaluate(m);
    const std::vector<int32_t> destroyed = {handle_to_var_id(z)};
    RNG rng(7);
    for (const bool widened : {false, true}) {
        INFO("widened " << widened);
        LnsProbe lns(0.5);
        lns.set_repair_store(std::make_shared<const LinearRowStore>(
            m, widened ? kDefaultFeasibilityTolerance : 0.0));
        CHECK(lns.fpr_repair(m, destroyed, rng, nullptr) == widened);
        CHECK(lns.fpr_repair_stats().dives == (widened ? 1 : 0));
        CHECK(lns.fpr_repair_stats().kept_conflicts == (widened ? 0 : 1));
        CHECK(m.var(handle_to_var_id(x)).value == 1.0);
    }
}

TEST_CASE("solve hands the LNS a repair store only with lns_repair_fpr", "[search][lns][fpr]") {
    for (const bool on : {false, true}) {
        INFO("lns_repair_fpr " << on);
        // The objective keeps solve kicking past the first feasible point.
        Model m = pair_model(20, 1.0, 1.0, /*minimize_x=*/true);
        SearchConfig cfg;
        cfg.batch_iterations = 1;
        cfg.perturbation_period = 1;
        cfg.max_iterations = 30;
        cfg.lns_repair_fpr = on;
        LNS lns(0.5);
        const SearchResult r = solve(m, /*time_limit=*/0.0, 3, nullptr, &lns,
                                     /*lns_interval=*/1, nullptr, cfg);
        REQUIRE(r.lns_repairs > 0);
        // Cleared at the end of the solve: the engine reads this model.
        CHECK(lns.repair_store() == nullptr);
        const LNS::FprRepairStats& s = lns.fpr_repair_stats();
        const int64_t rounds = s.dives + s.kept_conflicts + s.root_refuted;
        CHECK(rounds == (on ? r.lns_repairs : 0));
    }
}
