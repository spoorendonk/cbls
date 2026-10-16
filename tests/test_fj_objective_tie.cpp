// OR-Tools' use of the objective before the first feasible point (#218):
// GFJConfig::objective_tie_break (score tie-break + objective-improving scan).
// Default off, and the default path is pinned by the trajectory fences; what is
// pinned here is that the tie-break only ever breaks ties, that the scan
// reaches a column no violated row holds, and that the flag reaches solve().
// (#218's objective-favourable start was removed after #226's A/B.)
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <vector>

using namespace cbls;

TEST_CASE("the objective tie-break is on by default in solve() (#226)", "[fj][objective_tie]") {
    // GFJConfig's own default stays off: only solve() turns it on.
    REQUIRE_FALSE(GFJConfig{}.objective_tie_break);
    REQUIRE(SearchConfig{}.fj_objective_tie_break);
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

TEST_CASE("a weight rescale keeps an objective-only score (#218)", "[fj][objective_tie]") {
    // Scaling the smallest positive double by <= 0.5 rounds it to 0, which
    // apply_jump reads as "no improving move" and drops from Q for good.
    JumpTable t(2);
    t.set(0, 1.0, kObjectiveTieScore);
    t.set(1, 1.0, 4.0);
    t.scale_scores(0.25, /*keep_objective_ties=*/true);
    REQUIRE(t.score(0) == kObjectiveTieScore);
    REQUIRE(t.score(1) == 1.0);
    t.scale_scores(0.25);  // the default path, unchanged
    REQUIRE(t.score(0) == 0.0);
}

TEST_CASE("objective-only columns stay scannable across weight folds (#218)",
          "[fj][objective_tie]") {
    // The review's repro: x >= 5 and x <= 4 never both hold, so every batch
    // bumps; at rho = 0.5 a fold is due every few bumps. Each z_i can rise by
    // the tie-break only while z_i + x <= 13 allows. After every batch, a
    // cached entry must not read 0 where a fresh computation finds an
    // objective-only jump.
    Model m;
    auto x = m.int_var(0, 10);
    std::vector<int32_t> zs;
    std::vector<int32_t> negs;
    for (int i = 0; i < 20; ++i) {
        zs.push_back(m.int_var(0, 20));
        negs.push_back(m.neg(zs.back()));
    }
    m.add_constraint(m.geq(x, m.constant(5.0)));
    m.add_constraint(m.leq(x, m.constant(4.0)));
    for (const int32_t z : zs) {
        m.add_constraint(m.leq(m.sum({z, x}), m.constant(13.0)));
    }
    m.minimize(m.sum(negs));
    m.close();
    m.add_objective_soft_constraint();
    ViolationManager vm(m);
    RNG rng(7);
    GFJConfig cfg;
    cfg.objective_tie_break = true;
    cfg.rho = 0.5;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    const double coef = -1.0;
    int checked = 0;
    for (int b = 0; b < 200; ++b) {
        fj.batch(37);
        for (const int32_t z : zs) {
            const auto cached = fj.cached_jump(vid(z));
            if (!cached.has_value()) {
                continue;
            }
            const JumpResult fresh = compute_var_jump(m, vm.weights, vid(z), false, nullptr,
                                                      kDefaultFeasibilityTolerance, false, &coef);
            if (fresh.score == kObjectiveTieScore) {
                ++checked;
                CAPTURE(b, vid(z));
                REQUIRE(cached->score == kObjectiveTieScore);
            }
        }
    }
    REQUIRE(checked > 0);
}

TEST_CASE("between variables the larger objective gain wins a score tie (#218)",
          "[fj][objective_tie]") {
    // Infeasible x as in ScanFixture; z1 can gain 1, z2 can gain 10, both only
    // by the tie-break. The sample sees all of Q, so at the first local minimum
    // both are offered at kObjectiveTieScore and z2 must move first. Without
    // the cross-variable key the first sampled wins, about half the seeds.
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        Model m;
        auto x = m.int_var(0, 10);
        auto z1 = m.int_var(0, 1);
        auto z2 = m.int_var(0, 10);
        m.add_constraint(m.geq(x, m.constant(5.0)));
        m.add_constraint(m.leq(x, m.constant(4.0)));
        m.minimize(m.neg(m.sum({z1, z2})));
        m.close();
        m.add_objective_soft_constraint();
        ViolationManager vm(m);
        RNG rng(seed);
        GFJConfig cfg;
        cfg.objective_tie_break = true;
        cfg.sample_size_general = 10;
        FeasibilityJump fj(m, vm, rng, cfg);
        fj.begin(/*set_initial_x=*/true);
        int first = 0;
        for (int it = 0; it < 100 && first == 0; ++it) {
            fj.batch(1);
            if (m.var(vid(z2)).value != 0.0) {
                first = 2;
            } else if (m.var(vid(z1)).value != 0.0) {
                first = 1;
            }
        }
        CAPTURE(seed);
        REQUIRE(first == 2);
    }
}

TEST_CASE("the objective tie-break does nothing on a non-affine objective (#218)",
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
    cfg.objective_tie_break = true;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    REQUIRE(m.var(vid(x)).value == 0.0);
    REQUIRE(m.var(vid(y)).value == 0.0);
}

TEST_CASE("the objective tie-break reaches solve() (#218)", "[fj][objective_tie]") {
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
