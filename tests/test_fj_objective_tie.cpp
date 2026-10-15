// OR-Tools' use of the objective before the first feasible point (#218):
// GFJConfig::objective_tie_break (score tie-break + objective-improving scan)
// and GFJConfig::objective_start. Both default off, and the default path is
// pinned by the trajectory fences; what is pinned here is that the tie-break
// only ever breaks ties, that the scan reaches a column no violated row holds,
// that the start respects bounds, and that both flags reach solve().
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cbls;

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
}  // namespace

TEST_CASE("the objective flags are off by default (#218)", "[fj][objective_tie]") {
    REQUIRE_FALSE(GFJConfig{}.objective_tie_break);
    REQUIRE_FALSE(GFJConfig{}.objective_start);
    REQUIRE_FALSE(SearchConfig{}.fj_objective_tie_break);
    REQUIRE_FALSE(SearchConfig{}.fj_objective_start);
}

TEST_CASE("the objective tie-break picks among exact ties only (#218)", "[fj][objective_tie]") {
    // x in [0, 10] at 5, x <= 3 violated by 2: every x in 0..3 reduces the
    // violation by 2, every x above 5 raises it.
    Model m;
    auto x = m.int_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(3.0)));
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    const std::vector<double> w(m.constraint_ids().size(), 1.0);

    const JumpResult plain = compute_var_jump(m, w, vid(x));
    // Prefer larger x (minimise -x): of the tied 0..3 the tie-break takes 3, and
    // it never reaches 6..10, which would cost violation for objective.
    const double prefer_up = -1.0;
    const JumpResult up = compute_var_jump(m, w, vid(x), false, nullptr,
                                           kDefaultFeasibilityTolerance, false, &prefer_up);
    // Prefer smaller x: 0.
    const double prefer_down = 1.0;
    const JumpResult down = compute_var_jump(m, w, vid(x), false, nullptr,
                                             kDefaultFeasibilityTolerance, false, &prefer_down);
    CAPTURE(plain.jump_value, up.jump_value, down.jump_value);
    REQUIRE(up.jump_value == 3.0);
    REQUIRE(down.jump_value == 0.0);
    // The score is the violation decrease, unchanged by the tie-break.
    REQUIRE(plain.score == 2.0);
    REQUIRE(up.score == plain.score);
    REQUIRE(down.score == plain.score);
}

TEST_CASE("the objective tie-break never overrides a violation decrease (#218)",
          "[fj][objective_tie]") {
    // x in [0, 10] at 5, x >= 6 violated by 1 under a weight of 1e-300, against
    // an objective coefficient of 1e300 pulling x to 0. An additive epsilon (as
    // OR-Tools' 1/INT64_MAX * delta * coefficient) would let the objective win
    // here; the lexicographic tie-break takes the violation decrease, and of
    // the tied 6..10 the objective-best, 6.
    Model m;
    auto x = m.int_var(0, 10);
    m.add_constraint(m.geq(x, m.constant(6.0)));
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    const std::vector<double> w(m.constraint_ids().size(), 1e-300);
    const double coef = 1e300;
    const JumpResult r =
        compute_var_jump(m, w, vid(x), false, nullptr, kDefaultFeasibilityTolerance, false, &coef);
    CAPTURE(r.jump_value, r.score);
    REQUIRE(r.jump_value == 6.0);
    REQUIRE(r.score == 1e-300);
    REQUIRE(r.score > kObjectiveTieScore);
}

TEST_CASE("an objective-only jump scores kObjectiveTieScore (#218)", "[fj][objective_tie]") {
    // x in [0, 10] at 5 under a satisfied x <= 7: no candidate changes the
    // violation below 0, so without the tie-break there is no move; with it, x
    // goes to its objective-better bound at the smallest positive score.
    Model m;
    auto x = m.int_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(7.0)));
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    const std::vector<double> w(m.constraint_ids().size(), 1.0);
    const JumpResult plain = compute_var_jump(m, w, vid(x));
    REQUIRE(plain.score <= 0.0);
    const double coef = 1.0;
    const JumpResult tied =
        compute_var_jump(m, w, vid(x), false, nullptr, kDefaultFeasibilityTolerance, false, &coef);
    REQUIRE(tied.jump_value == 0.0);
    REQUIRE(tied.score == kObjectiveTieScore);
    REQUIRE(tied.score > 0.0);
    // Toward the worse side the same tie does not move it.
    const double up = -1.0;
    const JumpResult capped =
        compute_var_jump(m, w, vid(x), false, nullptr, kDefaultFeasibilityTolerance, false, &up);
    REQUIRE(capped.jump_value == 7.0);
    REQUIRE(capped.score == kObjectiveTieScore);
}

namespace {

// x >= 5 and x <= 4 cannot both hold, so FJ stays infeasible and keeps hitting
// local minima. z sits only in the objective (minimise -z): no violated row
// holds it, so only the objective scan can put it in Q, and only the tie-break
// can give it a positive score.
struct ScanFixture {
    Model m;
    int32_t x = 0;
    int32_t z = 0;
    ScanFixture() {
        x = m.int_var(0, 10);
        z = m.int_var(0, 10);
        m.add_constraint(m.geq(x, m.constant(5.0)));
        m.add_constraint(m.leq(x, m.constant(4.0)));
        m.minimize(m.neg(z));
        m.close();
        m.add_objective_soft_constraint();
    }
};

double z_after_batch(bool tie_break) {
    ScanFixture f;
    ViolationManager vm(f.m);
    RNG rng(3);
    GFJConfig cfg;
    cfg.objective_tie_break = tie_break;
    FeasibilityJump fj(f.m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    REQUIRE(f.m.var(vid(f.z)).value == 0.0);
    REQUIRE_FALSE(fj.batch(200));
    return f.m.var(vid(f.z)).value;
}

}  // namespace

TEST_CASE("objective_tie_break scans and moves an objective-only column (#218)",
          "[fj][objective_tie]") {
    REQUIRE(z_after_batch(false) == 0.0);
    REQUIRE(z_after_batch(true) == 10.0);
}

TEST_CASE("objective_start puts objective columns at their better bound (#218)",
          "[fj][objective_tie]") {
    for (const bool on : {false, true}) {
        Model m;
        auto up = m.int_var(-4, 10);             // minimised: lb -4
        auto down = m.int_var(-3, 7);            // maximised: ub 7
        auto flat = m.int_var(-5, 5);            // not in the objective: 0
        auto free_lb = m.float_var(-kInf, 4.0);  // minimised, lb infinite: 0
        auto b = m.bool_var();                   // maximised: 1
        auto fl = m.float_var(1.5, 2.5);         // minimised: 1.5
        m.add_constraint(m.leq(m.sum({up, down, flat, free_lb, b, fl}), m.constant(1e9)));
        m.minimize(m.sum({up, m.neg(down), free_lb, m.neg(b), fl}));
        m.close();
        m.add_objective_soft_constraint();
        ViolationManager vm(m);
        RNG rng(1);
        GFJConfig cfg;
        cfg.objective_start = on;
        FeasibilityJump fj(m, vm, rng, cfg);
        fj.begin(/*set_initial_x=*/true);
        CAPTURE(on);
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            REQUIRE(m.var(v).value >= m.var(v).lb);
            REQUIRE(m.var(v).value <= m.var(v).ub);
        }
        REQUIRE(m.var(vid(flat)).value == 0.0);
        REQUIRE(m.var(vid(free_lb)).value == 0.0);
        if (on) {
            REQUIRE(m.var(vid(up)).value == -4.0);
            REQUIRE(m.var(vid(down)).value == 7.0);
            REQUIRE(m.var(vid(b)).value == 1.0);
            REQUIRE(m.var(vid(fl)).value == 1.5);
        } else {
            REQUIRE(m.var(vid(up)).value == 0.0);
            REQUIRE(m.var(vid(down)).value == 0.0);
            REQUIRE(m.var(vid(b)).value == 0.0);
            REQUIRE(m.var(vid(fl)).value == 1.5);  // closest to zero is the lb
        }
    }
}

TEST_CASE("the objective flags do nothing on a non-affine objective (#218)",
          "[fj][objective_tie]") {
    Model m;
    auto x = m.int_var(-4, 10);
    auto y = m.int_var(-4, 10);
    m.add_constraint(m.leq(m.sum({x, y}), m.constant(100.0)));
    m.minimize(m.prod(x, y));
    m.close();
    m.add_objective_soft_constraint();
    ViolationManager vm(m);
    RNG rng(1);
    GFJConfig cfg;
    cfg.objective_start = true;
    cfg.objective_tie_break = true;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    REQUIRE(m.var(vid(x)).value == 0.0);
    REQUIRE(m.var(vid(y)).value == 0.0);
}

TEST_CASE("the objective flags reach solve() (#218)", "[fj][objective_tie]") {
    SECTION("objective_start: the first feasible point starts at the better bound") {
        // Any assignment is feasible, so the first feasible point is the start.
        for (const bool on : {false, true}) {
            Model m;
            std::vector<int32_t> xs;
            xs.reserve(5);
            for (int i = 0; i < 5; ++i) {
                xs.push_back(m.int_var(-5, 5));
            }
            m.add_constraint(m.leq(m.sum(xs), m.constant(100.0)));
            m.minimize(m.sum(xs));
            m.close();
            SearchConfig config;
            config.max_iterations = 1'000;
            config.fj_objective_start = on;
            const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/1, true, nullptr, nullptr,
                                         3, nullptr, config);
            CAPTURE(on, r.first_feasible_objective);
            REQUIRE(r.first_feasible_objective == (on ? -25.0 : 0.0));
        }
    }
    SECTION("objective_tie_break: the objective-only column moves while infeasible") {
        // ScanFixture's model through solve(): z is read only by the objective
        // row, which is inert while no feasible point exists, so only the
        // tie-break and its scan move it, and the run never becomes feasible.
        for (const bool on : {false, true}) {
            ScanFixture f;
            SearchConfig config;
            config.max_iterations = 2'000;
            config.use_compound_moves = false;
            config.fj_objective_tie_break = on;
            const SearchResult r = solve(f.m, /*time_limit=*/0.0, /*seed=*/1, true, nullptr,
                                         nullptr, 3, nullptr, config);
            REQUIRE_FALSE(r.feasible);
            CAPTURE(on, f.m.var(vid(f.z)).value);
            REQUIRE((f.m.var(vid(f.z)).value == 10.0) == on);
        }
    }
}
