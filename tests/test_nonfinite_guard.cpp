// Non-finite guard (issue #72): a non-convex objective/constraint that overflows
// to +inf or evaluates to NaN must not poison the violation cache, the
// structural-pass comparison, or the best-objective bookkeeping. These tests
// drive node values to +inf/NaN and assert the search/violation machinery stays
// finite and well-ordered.
//
// The `[nonfinite-objective]` cases below cover issue #100: a non-finite
// *objective* must not destroy the feasibility signal. Three separable defects,
// one test each — a vacuous `obj <= +inf` row read as maximally violated, the
// 1e30 clamp absorbing the real rows inside the jump score, and feasibility
// bookkeeping refusing a feasible point because the objective is not finite.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <limits>

using namespace cbls;

TEST_CASE("violation cache clamps a +inf constraint to a finite penalty", "[nonfinite]") {
    Model m;
    // x in a wide range so exp(x) can overflow to +inf for large x.
    int32_t x = m.float_var(-1.0e9, 1.0e9, "x");
    // Constraint: exp(x) <= 1  (i.e. x <= 0). For large x, exp(x) = +inf.
    int32_t ex = m.exp_expr(x);
    int32_t c = m.leq(ex, m.constant(1.0));
    m.add_constraint(c);
    m.close();

    // Force the overflow point.
    m.var_mut(vid(x)).value = 1.0e6;
    full_evaluate(m);

    ViolationManager vm(m);
    vm.invalidate_cache();
    double tv = vm.total_violation();
    REQUIRE(std::isfinite(tv));
    REQUIRE(tv > 0.0);

    // is_feasible must report infeasible (not silently pass a +inf/NaN).
    REQUIRE_FALSE(vm.is_feasible());

    // augmented_objective stays finite even with a +inf objective term.
    REQUIRE(std::isfinite(vm.augmented_objective()));
}

TEST_CASE("weighted_violation_delta stays finite across an overflow probe", "[nonfinite]") {
    Model m;
    int32_t x = m.float_var(-1.0e9, 1.0e9, "x");
    int32_t ex = m.exp_expr(x);
    int32_t c = m.leq(ex, m.constant(1.0));
    m.add_constraint(c);
    m.close();

    m.var_mut(vid(x)).value = 0.0;  // feasible-ish baseline: exp(0)=1
    full_evaluate(m);
    ViolationManager vm(m);

    // Probe a candidate that overflows exp(x): the delta must be finite and
    // positive (the move is strongly discouraged, not NaN).
    double d = vm.weighted_violation_delta(vid(x), 1.0e6);
    REQUIRE(std::isfinite(d));
    REQUIRE(d > 0.0);

    // The probe must not have committed any value.
    REQUIRE(m.var(vid(x)).value == 0.0);
}

TEST_CASE("NaN constraint value is treated as infeasible, not feasible", "[nonfinite]") {
    Model m;
    int32_t x = m.float_var(-1.0e9, 1.0e9, "x");
    // exp(x) - exp(x) is +inf - +inf = NaN when x is large enough to overflow.
    int32_t ex = m.exp_expr(x);
    int32_t diff = m.sum({ex, m.neg(ex)});  // NaN at the overflow point
    int32_t c = m.leq(diff, m.constant(0.0));
    m.add_constraint(c);
    m.close();

    m.var_mut(vid(x)).value = 1.0e6;  // exp(1e6) = +inf; inf - inf = NaN
    full_evaluate(m);
    REQUIRE(std::isnan(m.node(c).value));

    ViolationManager vm(m);
    vm.invalidate_cache();
    REQUIRE_FALSE(vm.is_feasible());
    REQUIRE(std::isfinite(vm.total_violation()));
    REQUIRE(vm.total_violation() > 0.0);
}

TEST_CASE("solve does not NaN-poison on an unbounded-overflow direction", "[nonfinite]") {
    // minimize exp(x) s.t. x >= 0.5 ; x in [-1e9, 1e9].
    // The objective grows without bound as x->+inf (overflowing to +inf), but a
    // finite feasible optimum exists at x = 0.5. The guard must keep the search
    // finite and let it record a finite feasible objective.
    Model m;
    int32_t x = m.float_var(-1.0e9, 1.0e9, "x");
    int32_t ex = m.exp_expr(x);
    m.add_constraint(m.geq(x, m.constant(0.5)));
    m.minimize(ex);
    m.close();

    SearchResult r = solve_deterministic(m, 249000, 42);

    REQUIRE(r.feasible);
    REQUIRE(std::isfinite(r.objective));
    // exp(0.5) ~= 1.6487; the search should be in that neighbourhood, certainly
    // not +inf and not absurdly large.
    REQUIRE(r.objective < 100.0);
}

// ---------------------------------------------------------------------------
// Issue #100: a non-finite objective must not destroy the feasibility signal.
// ---------------------------------------------------------------------------

TEST_CASE("a row with an infinite bound is vacuous, not maximally violated",
          "[nonfinite][nonfinite-objective]") {
    // `a <= +inf` holds for every a, including a = +inf. Evaluating the residual
    // as a plain `a - b` gives inf - inf = NaN, which the violation machinery
    // reads as a maximal violation — so the row that is *least* informative in
    // the model becomes the one that dominates it. This is the state every solve
    // of a blown-up objective opens in: `obj <= bound` with the bound still at
    // its initial +inf.
    Model m;
    int32_t x = m.float_var(-1.0e9, 1.0e9, "x");
    int32_t ex = m.exp_expr(x);  // +inf for large x
    int32_t c = m.leq(ex, m.constant(std::numeric_limits<double>::infinity()));
    m.add_constraint(c);
    m.close();

    m.var_mut(vid(x)).value = 1.0e6;  // exp(1e6) = +inf
    full_evaluate(m);
    REQUIRE(std::isinf(m.node(ex).value));

    ViolationManager vm(m);
    vm.invalidate_cache();
    REQUIRE(vm.total_violation() == 0.0);
    REQUIRE(vm.is_feasible());

    // The same rule must hold on the objective-bound shortcut, which writes the
    // residual in place instead of going through evaluate().
    //
    // (The overflow counterpart — where the infinity is computed rather than
    // written — is pinned by the next test.)
    Model m2;
    int32_t y = m2.float_var(-1.0e9, 1.0e9, "y");
    m2.minimize(m2.exp_expr(y));
    m2.close();
    m2.var_mut(vid(y)).value = 1.0e6;
    full_evaluate(m2);
    m2.add_objective_soft_constraint();  // bound starts at +inf
    m2.set_objective_bound(std::numeric_limits<double>::infinity());
    REQUIRE(m2.node(m2.constraint_ids()[m2.objective_constraint_idx()]).value == 0.0);
}

TEST_CASE("an overflowed infinity is not mistaken for an absent bound",
          "[nonfinite][nonfinite-objective]") {
    // The counterpart to the vacuous-row rule, and the reason it is gated on the
    // bound being a literal Const node.
    //
    // `exp(1000) <= exp(720)` is a genuinely VIOLATED row — e^1000 > e^720 — but
    // both sides overflow to +inf in double. Resolving `inf - inf` by "equal
    // infinities are satisfied" would silently pass an assignment there is no
    // evidence for, breaking the invariant the rest of the engine defends (the
    // NaN guards in ViolationManager, LNS and the search loop all exist to stop
    // exactly that). An infinity a variable's expression *computed* means only
    // "left double range"; only an infinity the modeller *wrote* is the
    // "this side is absent" sentinel.
    Model m;
    int32_t x = m.float_var(-1.0e9, 1.0e9, "x");
    int32_t y = m.float_var(-1.0e9, 1.0e9, "y");
    int32_t c = m.leq(m.exp_expr(x), m.exp_expr(y));  // neither side is a Const
    m.add_constraint(c);
    m.close();

    m.var_mut(vid(x)).value = 1000.0;  // exp(1000) = +inf
    m.var_mut(vid(y)).value = 720.0;   // exp(720)  = +inf
    full_evaluate(m);
    REQUIRE(std::isnan(m.node(c).value));

    ViolationManager vm(m);
    vm.invalidate_cache();
    REQUIRE_FALSE(vm.is_feasible());
    REQUIRE(vm.total_violation() > 0.0);
    REQUIRE(std::isfinite(vm.total_violation()));
}

TEST_CASE("a clamped row does not absorb the real rows in a jump score",
          "[nonfinite][nonfinite-objective]") {
    // The jump score is the change in total weighted violation. Summing each
    // side of the probe into its own accumulator and subtracting destroys the
    // real constraints' contribution whenever any one row is clamped to
    // kInfPenalty (1e30): `1e30 + 3 == 1e30`, so both sums round to the same
    // value and every candidate scores exactly 0.
    //
    // Row A below is +inf regardless of x (y is pinned past the exp overflow),
    // so it contributes nothing but its clamp. Row B is `x <= 0`, violated by 3.
    // Moving x to 0 must therefore score -3, not 0.
    Model m;
    int32_t x = m.float_var(-10.0, 10.0, "x");
    int32_t y = m.float_var(-1.0e9, 1.0e9, "y");
    int32_t row_a = m.leq(m.sum({m.exp_expr(x), m.exp_expr(y)}), m.constant(1.0));
    int32_t row_b = m.leq(x, m.constant(0.0));
    m.add_constraint(row_a);
    m.add_constraint(row_b);
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 1.0e6;  // exp(1e6) = +inf: row A is +inf for any x
    full_evaluate(m);
    REQUIRE(std::isinf(m.node(row_a).value));
    REQUIRE(m.node(row_b).value == 3.0);

    ViolationManager vm(m);
    // Both rows are in x's adjacency, so the clamped one is inside the sum.
    REQUIRE(m.constraints_of_var(vid(x)).size() == 2);

    double delta = vm.weighted_violation_delta(vid(x), 0.0);
    REQUIRE(delta == -3.0);

    // And the signal survives all the way through FJ's scoring entry point: the
    // variable must come back with a real improving jump, not "no move".
    JumpResult jr = compute_var_jump(m, vm.weights, vid(x), /*allow_escape_probe=*/false);
    REQUIRE(jr.score > 0.0);
    REQUIRE(jr.jump_value <= 0.0);
}

TEST_CASE("a feasible point with a +inf objective is reported feasible",
          "[nonfinite][nonfinite-objective]") {
    // minimize exp(x) s.t. x >= 1000, x in [0, 1e9].
    //
    // The feasible region is [1000, 1e9] — trivially reachable — and exp
    // overflows past ~709.78, so the objective is +inf on *all* of it. There is
    // no finite-objective feasible point to fall back on, so the run is reported
    // feasible only if feasibility bookkeeping is independent of whether the
    // objective happens to be finite there. This is the shape of MINLPLib's
    // elec* family, whose feasible region likewise contains configurations where
    // the objective diverges.
    Model m;
    int32_t x = m.float_var(0.0, 1.0e9, "x");
    m.add_constraint(m.geq(x, m.constant(1000.0)));
    m.minimize(m.exp_expr(x));
    m.close();

    SearchResult r = solve_deterministic(m, 20000, 42);

    REQUIRE(r.feasible);
    REQUIRE(r.best_violation <= kDefaultFeasibilityTolerance);
    // Honest about what it cannot report: the assignment is feasible, but its
    // objective is not a number, so no objective value is claimed.
    REQUIRE_FALSE(std::isfinite(r.objective));
    // And the returned assignment really does satisfy the constraint.
    REQUIRE(m.var(vid(x)).value >= 1000.0);
}

TEST_CASE("a +inf-objective feasible point does not block a later improvement",
          "[nonfinite][nonfinite-objective]") {
    // A smoke test for the witness path, not a tight regression guard: it pins
    // the end-to-end property that recording a +inf-objective feasible point
    // still lets a later finite objective be found and reported. It passes on
    // pre-#100 code (which never recorded the witness at all) and would pass on
    // most natural mis-implementations of the witness logic too, so it catches
    // only the gross failure where the witness latches and blocks improvement.
    // The precise rules — never becoming best_feasible_obj, never tightening
    // the bound — are asserted by construction in record_best, not here.
    //
    // minimize 1/(x-y)^2 over x, y in [1, 10]: every point is feasible, and the
    // objective diverges exactly on the diagonal. The search starts *on* the
    // diagonal, so the witness is recorded first and must then be improved away
    // from.
    Model m;
    int32_t x = m.float_var(1.0, 10.0, "x");
    int32_t y = m.float_var(1.0, 10.0, "y");
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(2.0)));  // trivially true
    int32_t d = m.sum({x, m.neg(y)});
    m.minimize(m.div_expr(m.constant(1.0), m.prod(d, d)));
    m.close();

    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 1.0;
    full_evaluate(m);
    REQUIRE_FALSE(std::isfinite(m.node(m.objective_id()).value));

    SearchConfig cfg;
    // Start on the diagonal: skip_init keeps the assignment set above instead of
    // letting FJ's begin() reset both Floats to the domain value closest to zero.
    cfg.skip_init = true;
    cfg.max_iterations = 200000;
    SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                           nullptr, cfg);

    REQUIRE(r.feasible);
    // The witness was displaced: a real objective value is reported, not +inf.
    REQUIRE(std::isfinite(r.objective));
    REQUIRE(r.objective < 1.0);
}

// ---------------------------------------------------------------------------
// Issue #116: the objective bound must not stay +inf once the search is sitting
// on a feasible point whose objective is non-finite.
//
// `obj <= +inf` is vacuous (pinned by the first [nonfinite-objective] case
// above), so while the bound is at its initial value the objective row cannot
// be violated and contributes nothing to any jump score. A search that reaches
// its first feasible point there is feasible, has no objective row to descend,
// and spends the rest of its budget doing nothing — measured on MINLPLib's
// elec25 as 3614 batches yielding 13804 GLS iterations.
//
// The shared model below is the smallest thing with that shape: minimise
// 1/(x-y)^2 over the box [1,10]^2, which diverges exactly on the diagonal, with
// a trivially-true real constraint so the diagonal is *feasible*. The search is
// started on the diagonal, so its first feasible point has a +inf objective.
// The finite optimum is 1/81, at the opposite corners.
// ---------------------------------------------------------------------------

namespace {

// x, y in [1,10]; minimise 1/(x-y)^2 subject to x + y >= 2 (always true).
// Leaves the assignment on the diagonal (obj = +inf) and the model closed.
int32_t build_diagonal_blowup_model(Model& m) {
    int32_t x = m.float_var(1.0, 10.0, "x");
    int32_t y = m.float_var(1.0, 10.0, "y");
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(2.0)));
    int32_t d = m.sum({x, m.neg(y)});
    m.minimize(m.div_expr(m.constant(1.0), m.prod(d, d)));
    m.close();
    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 1.0;
    full_evaluate(m);
    return x;
}

// Captures the objective bound at the first feasible progress report. The
// callback fires from record_best, so this observes the state the search hands
// to the batch that follows.
class BoundAtFirstFeasible : public SolveCallback {
public:
    explicit BoundAtFirstFeasible(const Model& model) : model_(model) {}

    void on_progress(const SolveProgress& p) override {
        if (captured || !p.feasible) {
            return;
        }
        captured = true;
        bound = model_.objective_bound();
        objective = model_.node(model_.objective_id()).value;
        objective_row =
            model_.node(model_.constraint_ids()[model_.objective_constraint_idx()]).value;
    }

    bool captured = false;
    double bound = 0.0;
    double objective = 0.0;
    double objective_row = 0.0;

private:
    const Model& model_;
};

}  // namespace

TEST_CASE("a feasible point with a non-finite objective gets a finite bound",
          "[nonfinite][nonfinite-objective]") {
    Model m;
    build_diagonal_blowup_model(m);

    BoundAtFirstFeasible probe(m);
    SearchConfig cfg;
    cfg.skip_init = true;  // keep the diagonal start
    cfg.max_iterations = 20000;
    solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3, &probe, cfg);

    REQUIRE(probe.captured);
    // Precondition: this really is the state under test — feasible, objective
    // not a number. (True with or without the fix.)
    REQUIRE_FALSE(std::isfinite(probe.objective));
    // The fix: a finite bound, hence an objective row the search can descend.
    REQUIRE(std::isfinite(probe.bound));
    REQUIRE(probe.bound > 0.0);
    // And that row is genuinely violated, so it reaches the jump scores rather
    // than being a satisfied row nobody looks at.
    REQUIRE_FALSE(probe.objective_row <= 0.0);
}

TEST_CASE("a search starting feasible with a +inf objective still finds a finite one",
          "[nonfinite][nonfinite-objective]") {
    // Diversification is switched off deliberately. A random kick off the
    // diagonal lands on a feasible point with a finite objective, which
    // rescues the unfixed engine by luck — that is what makes the pre-existing
    // "does not block a later improvement" case above pass either way, and it
    // is not a signal the engine can rely on (elec25 has 3614 batches' worth of
    // kicks and still returns +inf). With the kick suppressed, the only route
    // to a finite objective is the objective row itself.
    Model m;
    build_diagonal_blowup_model(m);

    SearchConfig cfg;
    cfg.skip_init = true;              // keep the diagonal start
    cfg.max_iterations = 20000;
    cfg.perturbation_period = 1 << 30;  // no diversification within the budget

    SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3, nullptr,
              cfg);

    REQUIRE(r.feasible);
    REQUIRE(std::isfinite(r.objective));
    // The box optimum is 1/81 at the opposite corners; nothing can beat it.
    REQUIRE(r.objective >= 1.0 / 81.0 - 1e-9);
    REQUIRE(r.objective < 0.1);
}
