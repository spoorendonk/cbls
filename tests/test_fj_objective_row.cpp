// The folded objective row's neighbour walk (#210).
//
// Every move of an objective variable used to invalidate the cached jump of
// every other objective variable, because the `obj <= bound` row is in
// vars_of_constraint_ like any row. update_var and Novelty's re-queue now skip
// the row while it is inert -- bound +inf, or slack by more than any one
// variable's reach (OR-Tools' row_max_variations) -- and keep the walk
// otherwise. These tests pin both halves: a cached objective-variable jump that
// the move cannot affect survives it and equals a fresh computation, and one
// the move CAN affect is invalidated (which fails if the skip is applied
// unconditionally).

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

using namespace cbls;

namespace {

// a (Bool) is the only variable of r0: 2a >= 2, y (Bool) the only variable of
// r1: y >= 1, so from the all-zero start both flips improve -- a's by 2, y's by
// 1. One apply_jump scores both (Q holds two, the sample is three), keeps both
// in Q and applies a's; y's entry is then whatever the move left of it. a and y
// share no row but the objective's.
struct ObjectiveFixture {
    Model m;
    int32_t a = 0;
    int32_t y = 0;

    explicit ObjectiveFixture(bool nonlinear_objective) {
        a = m.bool_var();
        y = m.bool_var();
        m.add_constraint(m.geq(m.prod(m.constant(2.0), a), m.constant(2.0)));
        m.add_constraint(m.geq(y, m.constant(1.0)));
        if (nonlinear_objective) {
            m.minimize(m.sum({a, y, m.prod(a, y)}));
        } else {
            m.minimize(m.sum({a, y}));
        }
        m.close();
        m.add_objective_soft_constraint();
    }
};

struct Outcome {
    std::optional<JumpResult> cached;  // y's entry after a's move
    JumpResult fresh;                  // y's jump computed from scratch after it
    JumpResult before;                 // y's jump computed from scratch before it
};

// One GLS iteration from the all-zero start under `bound` (+inf: left as
// add_objective_soft_constraint opened it).
Outcome one_move(ObjectiveFixture& f, double bound) {
    if (bound != std::numeric_limits<double>::infinity()) {
        f.m.set_objective_bound(bound);
    }
    ViolationManager vm(f.m);
    RNG rng(7);
    GFJConfig cfg;
    cfg.two_phase = false;
    FeasibilityJump fj(f.m, vm, rng, cfg);
    fj.begin(true);
    Outcome out;
    out.before = compute_var_jump(f.m, vm.weights, vid(f.y), false, nullptr);
    REQUIRE_FALSE(fj.batch(1));
    REQUIRE(f.m.var(vid(f.a)).value == 1.0);  // the move under test was a's flip
    REQUIRE(f.m.var(vid(f.y)).value == 0.0);
    out.cached = fj.cached_jump(vid(f.y));
    out.fresh = compute_var_jump(f.m, vm.weights, vid(f.y), false, nullptr);
    return out;
}

}  // namespace

TEST_CASE(
    "FJ keeps an objective variable's cached jump across a move the objective row cannot "
    "pass on",
    "[fj][objective_row]") {
    ObjectiveFixture f(false);
    double bound = std::numeric_limits<double>::infinity();
    SECTION("no bound yet: the row is obj - inf throughout") {}
    SECTION("a finite bound with more slack than one variable can take") {
        // r goes -10 -> -9 and M = 1, so r + M < 0 on both sides.
        bound = 10.0;
    }
    const Outcome o = one_move(f, bound);
    if (!o.cached.has_value()) {
        FAIL("y's cached jump was invalidated by a move that cannot change it");
        return;
    }
    // Exactly what a fresh computation gives: the skip is exact, not a guess.
    CHECK(o.cached->jump_value == 1.0);
    CHECK(o.cached->score == 1.0);
    CHECK(o.cached->jump_value == o.fresh.jump_value);
    CHECK(o.cached->score == o.fresh.score);
}

TEST_CASE("FJ invalidates an objective variable's cached jump when the objective row is tight",
          "[fj][objective_row]") {
    // bound 1.5 and M = 1: before a's flip r = -1.5, and y's flip would leave
    // it at -0.5; after it r = -0.5, and y's flip takes obj to 2, violating the
    // row by 0.5. y's score really drops, from 1 to 0.5, so its entry must not
    // survive.
    ObjectiveFixture f(false);
    const Outcome o = one_move(f, 1.5);
    CHECK_FALSE(o.cached.has_value());
    CHECK(o.before.score == 1.0);
    CHECK(o.fresh.score == 0.5);  // the invalidation was needed
}

TEST_CASE("FJ keeps the objective row's walk for a non-affine objective, even with no bound",
          "[fj][objective_row]") {
    // A jump can turn a nonlinear objective NaN, which reads as violated, so no
    // bound on its contribution exists without a slope; the skip stays off.
    ObjectiveFixture f(true);
    const Outcome o = one_move(f, std::numeric_limits<double>::infinity());
    CHECK_FALSE(o.cached.has_value());
}

TEST_CASE("LinearJumpScorer::row_max_variation is the largest one-variable reach of a row",
          "[linear_jump][objective_row]") {
    Model m;
    const int32_t x = m.int_var(0, 4);
    const int32_t b = m.bool_var();
    const int32_t f = m.float_var(0.0, std::numeric_limits<double>::infinity());
    // r0: 2x + 3b <= 5 -> reach max(2 * 4, 3 * 1) = 8.
    m.add_constraint(
        m.leq(m.sum({m.prod(m.constant(2.0), x), m.prod(m.constant(3.0), b)}), m.constant(5.0)));
    // r1: f + x <= 3, f unbounded above -> +inf.
    m.add_constraint(m.leq(m.sum({f, x}), m.constant(3.0)));
    // r2: x * b <= 1 -- not affine, no bound.
    m.add_constraint(m.leq(m.prod(x, b), m.constant(1.0)));
    m.close();
    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng);
    LinearJumpScorer& sc = fj.linear_scorer();

    double out = -1.0;
    REQUIRE(sc.row_max_variation(0, {vid(x), vid(b)}, out));
    CHECK(out == 8.0);
    REQUIRE(sc.row_max_variation(1, {vid(f), vid(x)}, out));
    CHECK(out == std::numeric_limits<double>::infinity());
    CHECK_FALSE(sc.row_max_variation(2, {vid(x), vid(b)}, out));
    // A variable outside the row is a caller error, not a zero.
    CHECK_THROWS_AS(sc.row_max_variation(0, {vid(f)}, out), std::logic_error);
}
