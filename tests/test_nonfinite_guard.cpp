// Non-finite guard (issue #72): a non-convex objective/constraint that overflows
// to +inf or evaluates to NaN must not poison the violation cache, the
// structural-pass comparison, or the best-objective bookkeeping. These tests
// drive node values to +inf/NaN and assert the search/violation machinery stays
// finite and well-ordered.

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
