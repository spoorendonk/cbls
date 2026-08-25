#include "test_helpers.h"

#include <algorithm>
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

    JumpResult r = compute_var_jump(m, vm.weights, vid(x));
    // The candidate set includes the domain midpoint (=5), the argmin of
    // (x-5)^2; Newton steps toward the constraint root also point there.
    REQUIRE(std::abs(r.jump_value - 5.0) < 1e-3);
    REQUIRE(r.score > 24.0);  // reduction ~25
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
        JumpResult r = compute_var_jump(m, vm.weights, vid(x));
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
        JumpResult r = compute_var_jump(m, vm.weights, vid(b));
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
        JumpResult r = compute_var_jump(m, vm.weights, vid(x));
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
    JumpResult r = compute_var_jump(m, vm.weights, vid(x));
    REQUIRE(r.jump_value >= -3.0);
    REQUIRE(r.jump_value <= 3.0);
    // From x0=0 the constraint is violated; the jump must strictly reduce
    // weighted violation (score > 0) and move sin(x) toward the feasible region
    // (the GLS loop iterates cheap steps to full feasibility — a single jump
    // need not land inside it).
    REQUIRE(r.score > 0.0);
    REQUIRE(std::sin(r.jump_value) < std::sin(0.0));
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

TEST_CASE("Novelty Jump escapes an FJ local optimum via a compound move", "[fj][novelty]") {
    // bool x, y with: x != y, x >= 1, y <= 0. Feasible only at (x=1, y=0).
    // From (0,1) every SINGLE flip is net-zero (fixes one constraint, breaks
    // x!=y), so Feasibility Jump is stuck; the 2-variable compound move
    // (x->1, y->0) reaches feasibility — exactly what Novelty Jump is for.
    Model m;
    auto x = m.bool_var();
    auto y = m.bool_var();
    m.add_constraint(m.neq(x, y));                // x != y
    m.add_constraint(m.geq(x, m.constant(1.0)));  // x >= 1
    m.add_constraint(m.leq(y, m.constant(0.0)));  // y <= 0
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 1.0;

    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng, GFJConfig{});
    fj.begin(/*set_initial_x=*/false);  // keep (0,1); evaluate + seed state

    // Single-variable jumps are non-improving here (score 0): FJ is at a local
    // optimum.
    REQUIRE(compute_var_jump(m, vm.weights, vid(x)).score <= 0.0);
    REQUIRE(compute_var_jump(m, vm.weights, vid(y)).score <= 0.0);

    // Novelty Jump finds the compound move and reaches feasibility.
    REQUIRE(fj.apply_novelty_jump());
    REQUIRE(vm.is_feasible());
    REQUIRE(m.var(vid(x)).value == 1.0);
    REQUIRE(m.var(vid(y)).value == 0.0);
}

TEST_CASE("Novelty Jump terminates and leaves consistent state on a contradiction",
          "[fj][novelty]") {
    Model m;
    auto x = m.int_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(2.0)));  // x <= 2
    m.add_constraint(m.geq(x, m.constant(5.0)));  // x >= 5  (contradiction)
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng, GFJConfig{});
    fj.begin(/*set_initial_x=*/false);

    REQUIRE_FALSE(fj.apply_novelty_jump());  // cannot reach feasibility; must return
    REQUIRE_FALSE(vm.is_feasible());
    // State left consistent: total violation is finite and the var is in-domain.
    fj.resync();
    REQUIRE(std::isfinite(vm.total_violation()));
    REQUIRE(m.var(vid(x)).value >= 0.0);
    REQUIRE(m.var(vid(x)).value <= 10.0);
}

// ---------------------------------------------------------------------------
// Escape from a stationary point (#107)
// ---------------------------------------------------------------------------

TEST_CASE("compute_var_jump: a stationary Float is frozen without the probe", "[fj][stationary]") {
    // f(y) = -4y^2 + 4y^4. f'(0) == 0 exactly, minima at y = +-1/sqrt(2).
    // Folded in as `obj <= bound` the objective row is violated at y = 0, but its
    // gradient there is 0 so no Newton candidate is emitted; the box is symmetric
    // so the midpoint candidate equals y and is skipped; and lb/ub are far worse.
    // Every candidate is dead and the variable cannot move at all — this is the
    // frozen state the escape probe exists for, and it is the default behaviour
    // because firing the probe in the steady state suppresses diversification.
    Model m;
    auto y = m.float_var(-10.0, 10.0, "y");
    m.minimize(m.sum({m.prod(m.constant(-4.0), m.pow_expr(y, m.constant(2.0))),
                      m.prod(m.constant(4.0), m.pow_expr(y, m.constant(4.0)))}));
    m.close();
    m.add_objective_soft_constraint();
    m.set_objective_bound(-1e-3);
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);

    REQUIRE(compute_var_jump(m, vm.weights, vid(y)).score == 0.0);

    SECTION("armed, it gets a local jump out") {
        JumpResult r = compute_var_jump(m, vm.weights, vid(y), /*allow_escape_probe=*/true);
        REQUIRE(r.score > 0.0);
        REQUIRE(r.jump_value != 0.0);
        REQUIRE(std::abs(r.jump_value) < 1.0);  // a local nudge, not a box endpoint
    }
}

TEST_CASE("compute_var_jump: the escape probe invents nothing", "[fj][stationary]") {
    // Control for the test above. At a true local minimum there is no improving
    // jump, and the probe must not manufacture one — otherwise an already
    // converged variable would churn forever and never let the search stagnate.
    Model m;
    auto z = m.float_var(-10.0, 10.0, "z");
    m.minimize(m.pow_expr(z, m.constant(2.0)));
    m.close();
    m.add_objective_soft_constraint();
    m.set_objective_bound(-1e-3);  // unreachable: z^2 >= 0, so the row stays violated
    m.var_mut(vid(z)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);

    REQUIRE(compute_var_jump(m, vm.weights, vid(z), /*allow_escape_probe=*/true).score == 0.0);
}

TEST_CASE("compute_var_jump: a fixed Float has nothing to escape", "[fj][stationary]") {
    // A variable whose domain is a single point has no jump, armed or not. The
    // probe candidates all clamp onto lb == ub == x0 and are discarded, so this
    // does not distinguish the `ub <= lb` early return's return value — it pins
    // the outcome, which is what callers depend on.
    Model m;
    auto f = m.float_var(3.0, 3.0, "f");
    m.minimize(m.pow_expr(f, m.constant(2.0)));
    m.close();
    m.add_objective_soft_constraint();
    m.set_objective_bound(-1e-3);
    m.var_mut(vid(f)).value = 3.0;
    full_evaluate(m);
    ViolationManager vm(m);

    REQUIRE(compute_var_jump(m, vm.weights, vid(f), /*allow_escape_probe=*/true).score == 0.0);
}

TEST_CASE("compute_var_jump: the escape probe is two-sided", "[fj][stationary]") {
    // f(y) = y^3 + y^4. f'(0) == 0, so the point is stationary and no Newton
    // candidate is emitted; the box is symmetric so the midpoint equals y; and
    // both endpoints are far worse (f(-10) = 9000, f(10) = 11000). The only
    // descent from the origin is NEGATIVE — f(+h) = +h^3 > 0 — so a probe that
    // only looked at x0 + h would find nothing and the variable would stay
    // frozen. The true minimum is -0.1055 at y = -0.75.
    Model m;
    auto y = m.float_var(-10.0, 10.0, "y");
    m.minimize(m.sum({m.pow_expr(y, m.constant(3.0)), m.pow_expr(y, m.constant(4.0))}));
    m.close();
    m.add_objective_soft_constraint();
    m.set_objective_bound(-1e-3);
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);

    JumpResult r = compute_var_jump(m, vm.weights, vid(y), /*allow_escape_probe=*/true);
    REQUIRE(r.score > 0.0);
    REQUIRE(r.jump_value < 0.0);  // the descent direction a one-sided probe misses
    REQUIRE(r.jump_value > -1.0);
}

// ---------------------------------------------------------------------------
// How the GLS loop observes the wall-clock deadline (#113)
//
// The loop cannot read the clock every iteration: steady_clock::now() measures
// 1408 ns/call on this project's HPET reference machine against GLS iterations
// of a few microseconds, and checking every time measured a 1.75x throughput
// loss (2991 -> 4816 ns/iteration, Release). It used to check on a FIXED stride
// of 64 iterations instead, which is not a time bound at all: one GLS iteration
// is O(sampled vars x candidate values x constraints touched), so 64 of them
// are microseconds on a small model and seconds on a large one. On 400 Int vars
// with 20 000 rows of 8, every budget from 0.05s to 3s took ~7s.
//
// The stride is now sized in *time*: each check measures the previous stride and
// sizes the next one to kStrideBudgetFraction of the budget. The guarantee is
//
//   one stride's work, where a stride costs at most 1/64 of the budget -- or one
//   GLS iteration, whichever is larger, an iteration being atomic.
//
// NONE OF THESE TESTS ASSERTS ON ELAPSED TIME (see the #104 block in
// test_search.cpp for why). The sizing rule is tested as a pure function, and
// the two live tests observe the tuner's own state: the stride it settled on and
// the number of clock reads it made.
// ---------------------------------------------------------------------------

TEST_CASE("the deadline-check stride is sized from the last measurement", "[fj][deadline]") {
    using FJ = FeasibilityJump;
    constexpr double kTarget = 0.001;  // one stride may cost 1ms

    SECTION("a stride well inside the target grows, but only by the growth cap") {
        // Not straight to target/elapsed = 1e6 iterations: the ramp goes through
        // progressively longer — and so more accurate — measurements instead of
        // extrapolating the whole way from one very short one.
        REQUIRE(FJ::next_deadline_stride(1, 1e-9, kTarget) == FJ::kStrideGrowth);
        REQUIRE(FJ::next_deadline_stride(8, 1e-6, kTarget) == 8 * FJ::kStrideGrowth);
    }

    SECTION("an interval below the clock's resolution counts as far inside") {
        REQUIRE(FJ::next_deadline_stride(8, 0.0, kTarget) == 8 * FJ::kStrideGrowth);
    }

    SECTION("a stride that landed on the target stays put") {
        REQUIRE(FJ::next_deadline_stride(32, kTarget, kTarget) == 32);
    }

    SECTION("a stride over the target shrinks, and the shrink is not capped") {
        // 10x over target shrinks 10x in ONE step, so an iteration that got more
        // expensive is caught on the very next check. Note this is NOT by itself
        // what stops the tuner ratcheting up and going silent — a shrink is only
        // applied AT a check, a whole stride later. kMaxDeadlineStride is what
        // bounds that; see "a grown stride cannot outrun its cap" below.
        REQUIRE(FJ::next_deadline_stride(64, 0.010, kTarget) == 6);  // 10x over -> 10x down
        REQUIRE(FJ::next_deadline_stride(64, 1.0, kTarget) == 1);    // 1000x over -> the floor
        // A stride from before the cap existed is clamped to it, not honoured.
        REQUIRE(FJ::next_deadline_stride(65536, 1e-9, kTarget) == FJ::kMaxDeadlineStride);
    }

    SECTION("a non-finite measurement floors rather than growing") {
        // A NaN fails `elapsed > 0.0`, which alone would take the growth cap —
        // i.e. treat "no information" as "far inside the target".
        const double nan = std::numeric_limits<double>::quiet_NaN();
        REQUIRE(FJ::next_deadline_stride(1000, nan, kTarget) == 1);
        REQUIRE(FJ::next_deadline_stride(1000, 1e-9, nan) == 1);
    }

    SECTION("one iteration costing more than the whole target floors at 1") {
        // The irreducible case: an iteration is atomic, so the loop cannot check
        // more often than once per iteration however expensive that iteration is.
        REQUIRE(FJ::next_deadline_stride(1, 10.0, kTarget) == 1);
        REQUIRE(FJ::next_deadline_stride(64, 10.0, kTarget) == 1);
    }

    SECTION("the stride never ratchets past its cap") {
        REQUIRE(FJ::next_deadline_stride(FJ::kMaxDeadlineStride, 1e-9, kTarget) ==
                FJ::kMaxDeadlineStride);
    }

    SECTION("a grown stride cannot outrun its cap, however cheap the ramp") {
        // The named hazard from #113: an earlier self-tuning stride was removed
        // from this engine for ratcheting past the loop bound and going silent.
        // A shrink cannot prevent that on its own, because a shrink is only
        // applied AT a check and the next check is a whole stride away — so
        // whatever the stride reached while iterations were cheap is spent in
        // full on the first expensive one. The iteration cap is what bounds it.
        int64_t stride = 1;
        for (int i = 0; i < 200; ++i) {  // an arbitrarily long cheap phase
            stride = FJ::next_deadline_stride(stride, 1e-9, kTarget);
            REQUIRE(stride <= FJ::kMaxDeadlineStride);
        }
        REQUIRE(stride == FJ::kMaxDeadlineStride);
    }

    SECTION("the work done past a deadline is bounded when iterations get expensive") {
        // Walk the countdown exactly as gls_loop does — cheap phase, then a
        // 1e6x cost jump — and count the iterations that run past the deadline.
        // Timing-free on purpose: the quantity the guarantee is about is
        // ITERATIONS PER CHECK, which is observable without a clock.
        constexpr double kCheap = 1e-9;
        constexpr double kExpensive = 1e-3;
        constexpr double kBudget = 1.0;

        double elapsed_total = 0.0;
        int64_t stride = 1;
        int64_t iterations_past_deadline = 0;
        int64_t worst_stride_at_check = 0;

        for (int64_t i = 0; i < 2000000; ++i) {
            // Cost jumps once the cheap phase has had ample time to ratchet.
            const double cost = i < 100000 ? kCheap : kExpensive;
            elapsed_total += cost;
            if (elapsed_total >= kBudget) {
                ++iterations_past_deadline;  // the loop cannot stop between checks
            }
            if (--stride > 0) {
                continue;
            }
            if (elapsed_total >= kBudget) {
                break;  // this is the check that observes the deadline and stops
            }
            const double remaining = kBudget - elapsed_total;
            // Reproduce the measurement the loop would have taken: the stride it
            // just finished, at the cost it was paying.
            stride = FJ::next_deadline_stride(stride == 0 ? 1 : stride, cost,
                                              remaining * FJ::kStrideBudgetFraction);
            worst_stride_at_check = std::max(worst_stride_at_check, stride);
        }

        // One stride's work is the whole of the overrun, and a stride is capped.
        REQUIRE(worst_stride_at_check <= FJ::kMaxDeadlineStride);
        REQUIRE(iterations_past_deadline <= FJ::kMaxDeadlineStride);
    }

    SECTION("the predicted cost of the next stride never exceeds the target") {
        // The guarantee itself, over the whole (stride, measurement) space: at
        // the per-iteration cost just measured, the next stride costs at most the
        // target — unless it is the one-iteration floor, which nothing can
        // subdivide.
        for (int64_t stride :
             {int64_t{1}, int64_t{7}, int64_t{64}, int64_t{1000}, FJ::kMaxDeadlineStride}) {
            for (double elapsed : {1e-7, 1e-6, 1e-4, 1e-3, 1e-2, 1.0}) {
                const double per_iteration = elapsed / static_cast<double>(stride);
                const int64_t next = FJ::next_deadline_stride(stride, elapsed, kTarget);
                REQUIRE(next >= 1);
                REQUIRE((next == 1 || static_cast<double>(next) * per_iteration <= kTarget));
            }
        }
    }
}

namespace {

// Expensive GLS iterations: 60 Int vars in 1500 rows of 8, so every variable
// sits in ~200 constraints and each of its 21 domain values has to be scored
// against all of them — ~2.5ms per iteration here (measured, Release). A
// scaled-down version of the model
// in #113 (400 vars, 20 000 rows), which is the shape of the largest benchmark
// instances: the ones whose wall times epic #87 publishes. Rows alternate
// `<= 3` and `>= 100` over overlapping variable sets, which is unsatisfiable, so
// the loop cannot exit early on feasibility.
void build_expensive_iterations(Model& m) {
    RNG rng(7);
    constexpr int kVars = 60;
    constexpr int kRows = 1500;
    constexpr int kPerRow = 8;
    std::vector<int32_t> vars;
    vars.reserve(kVars);
    for (int i = 0; i < kVars; ++i) {
        vars.push_back(m.int_var(0, 20));
    }
    for (int r = 0; r < kRows; ++r) {
        std::vector<int32_t> args;
        args.reserve(kPerRow);
        for (int k = 0; k < kPerRow; ++k) {
            args.push_back(vars[static_cast<size_t>(rng.integers(0, kVars))]);
        }
        auto row = m.sum(args);
        if (r % 2 == 0) {
            m.add_constraint(m.leq(row, m.constant(3.0)));
        } else {
            m.add_constraint(m.geq(row, m.constant(100.0)));
        }
    }
    m.close();
}

// Cheap GLS iterations: Bool vars (one candidate value per jump), tiny rows, and
// no objective — an objective row spans every variable, which would make every
// iteration invalidate every jump. ~3us per iteration here, i.e. the regime the
// stride exists to protect (a clock read is 1.4us of that). Unsatisfiable, so
// the loop never stops early on feasibility.
void build_cheap_iterations(Model& m) {
    RNG rng(7);
    constexpr int kVars = 8;
    constexpr int kRows = 8;
    std::vector<int32_t> vars;
    vars.reserve(kVars);
    for (int i = 0; i < kVars; ++i) {
        vars.push_back(m.bool_var());
    }
    for (int r = 0; r < kRows; ++r) {
        std::vector<int32_t> args;
        args.reserve(3);
        for (int k = 0; k < 3; ++k) {
            args.push_back(vars[static_cast<size_t>(rng.integers(0, kVars))]);
        }
        auto row = m.sum(args);
        // Overlapping triples forced both to 3 and to 0: unsatisfiable.
        if (r % 2 == 0) {
            m.add_constraint(m.geq(row, m.constant(3.0)));
        } else {
            m.add_constraint(m.leq(row, m.constant(0.0)));
        }
    }
    m.close();
}

}  // namespace

TEST_CASE("expensive iterations pin the deadline stride to one", "[fj][deadline]") {
    Model m;
    build_expensive_iterations(m);
    ViolationManager vm(m);
    RNG rng(42);

    GFJConfig cfg;
    cfg.two_phase = false;
    // One stride may cost 0.005/64 = 78us; one iteration of this model costs
    // ~2.5ms (measured, Release). So the tuner cannot afford a second iteration
    // per stride, settles on the floor, and the bound becomes the tightest one
    // available: a single atomic iteration.
    //
    // The budget is the one free parameter and it trades margin against work
    // done — and the trade is exactly fixed, because
    //
    //   (iterations that fit in the budget) x (margin) = 1 / kStrideBudgetFraction
    //
    // so at 1/64 no budget buys both. 0.005s puts it at ~2 iterations with a 32x
    // margin: a machine would have to run this model's GLS iterations 32x faster
    // than this one before the tuner could afford a stride of 2. The flake
    // direction is the safe one — a slower or busier machine does fewer, more
    // expensive iterations and passes more comfortably.
    cfg.time_limit = 0.005;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    fj.batch(/*batch_iterations=*/1000);

    REQUIRE(fj.iterations() > 0);                      // not inert: the batch really ran
    REQUIRE(fj.deadline_check_stride() == 1);          // one iteration per clock read...
    REQUIRE(fj.deadline_checks() == fj.iterations());  // ...and it really was read
    // And the batch stopped long before the 64 iterations the old fixed stride
    // let through unconditionally, which is the whole of #113.
    REQUIRE(fj.iterations() < 64);
}

TEST_CASE("cheap iterations let the deadline stride grow", "[fj][deadline]") {
    using FJ = FeasibilityJump;
    // The other half: the stride exists to protect throughput, and a tuner that
    // sat at 1 would reintroduce the 1.75x per-iteration-check cost it was
    // introduced to avoid. With ~3us iterations and a 10s budget (156ms per
    // stride) the tuner has room for tens of thousands of iterations per read.
    Model m;
    build_cheap_iterations(m);
    ViolationManager vm(m);
    RNG rng(42);

    GFJConfig cfg;
    cfg.two_phase = false;
    // Never reached — the batch limit below ends the run after ~60ms — so this
    // is a stride-sizing input, not a budget the test spends. It is generous on
    // purpose: the margin here is how much SLOWER the machine may be before the
    // tuner can no longer afford a stride over 64, and at 156ms per stride
    // against 3us iterations that is a factor of ~800.
    cfg.time_limit = 10.0;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    fj.batch(/*batch_iterations=*/20000);

    REQUIRE(fj.iterations() == 20000);  // not inert: the deadline never fired
    // The tuner ramps 1, 8, 64 and then holds at the cap, so on a cheap model it
    // ends up reading the clock exactly as often as the fixed stride it replaced
    // — no more. Growing past 64 was measured at ~1.2% throughput and cost an
    // unbounded overrun when iteration cost rose, so the cap keeps the shrink
    // (which is what #113 asked for) and drops the growth.
    REQUIRE(fj.deadline_check_stride() == FJ::kMaxDeadlineStride);
    // Still strictly fewer reads than a fixed 64 over the whole run, because the
    // ramp spends its first iterations at a coarser-than-1 stride only after
    // measuring: the ramp itself is 1, 8, 64.
    REQUIRE(fj.deadline_checks() <= fj.iterations() / 64 + 3);
}

TEST_CASE("a run with no wall clock reads no clock at all", "[fj][deadline]") {
    // Determinism, observed directly rather than inferred: with time_limit <= 0
    // the deadline branch short-circuits before the clock read, so nothing
    // timing-derived can reach control flow and an iteration-budgeted run stays
    // bit-identical from machine to machine.
    Model m;
    build_cheap_iterations(m);
    ViolationManager vm(m);
    RNG rng(42);

    GFJConfig cfg;
    cfg.two_phase = false;
    cfg.time_limit = 0.0;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    fj.batch(/*batch_iterations=*/500);

    REQUIRE(fj.iterations() == 500);  // not inert: iterations really ran
    REQUIRE(fj.deadline_checks() == 0);
}
