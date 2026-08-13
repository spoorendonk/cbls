// Regression tests for #112: randomising a variable over an unbounded domain.
//
// `rng.uniform(lb, ub)` violates uniform_real_distribution's precondition
// (`ub - lb <= DBL_MAX`) on an infinite domain, and libstdc++'s `lb + (ub-lb)*u`
// then returns NaN on `(-inf, +inf)` and +inf on `[0, +inf)`; an infinite Int
// bound casts to INT64_MIN. That draw sits on the default `solve()` path via
// FeasibilityJump's diversification kick, so one kick replaced the assignment
// with NaN, `full_evaluate` propagated it through the DAG, and the search burned
// the rest of its budget on a dead assignment while `solve()` still returned an
// ordinary-looking infeasible result.
//
// The tests below are split by what they can catch:
//
//  - the `domain_window` / `random_in_domain` cases pin the guard itself,
//    including that it is INERT on a finite domain (existing runs must keep
//    their exact draw sequence);
//  - "a kick keeps the assignment finite" and "the search still improves after a
//    kick" are the discriminating ones — both fail on the pre-fix engine;
//  - the LNS case is required by the issue but is NOT discriminating on its own:
//    destroy/repair maps a NaN constraint to +inf in its acceptance key and
//    rolls back on non-improvement, so a NaN-poisoned repair could never win
//    even before the fix. It is made discriminating by also requiring that a
//    repair is *accepted*, which pre-fix never happens.
//
// No new NaN *detector* is added here. Three guards already absorb a non-finite
// constraint value safely (ViolationManager::clamped_node_violation, solve()'s
// max_real_violation, LNS's state_key — see tests/test_nonfinite_guard.cpp), and
// with randomisation no longer able to inject one there is no remaining path
// from the engine's own moves into a NaN assignment.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

#include <cmath>
#include <limits>
#include <vector>

using namespace cbls;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr uint64_t kSeeds = 200;

// The default from SearchConfig::perturbation_probability.
constexpr double kDefaultP = 0.1;

bool all_values_finite(const Model& m) {
    for (const auto& v : m.variables()) {
        if (!is_structured(v.type) && !std::isfinite(v.value)) {
            return false;
        }
    }
    return true;
}

bool all_constraints_finite(const Model& m) {
    for (int32_t cid : m.constraint_ids()) {
        if (!std::isfinite(m.node(cid).value)) {
            return false;
        }
    }
    return true;
}

// Two unbounded floats on the unit circle: the repro from the issue. `|x^2 +
// y^2 - 1| <= 0` is an equality written as a violation, so every constraint
// value is finite exactly when the assignment is.
Model unit_circle_model() {
    Model m;
    int32_t x = m.float_var(-kInf, kInf, "x");
    int32_t y = m.float_var(-kInf, kInf, "y");
    int32_t r = m.sum({m.prod(x, x), m.prod(y, y), m.constant(-1.0)});
    m.add_constraint(m.leq(m.abs_expr(r), m.constant(0.0)));
    m.close();
    return m;
}

// A scalar Variable with the given type and bounds, without going through the
// factories: `int_var` takes `int`, so an infinite Int bound is only reachable
// by writing one (which the .cbls/.nl/.mps readers can do).
Variable scalar_var(VarType type, double lb, double ub) {
    Variable v;
    v.id = 0;
    v.type = type;
    v.lb = lb;
    v.ub = ub;
    v.value = 0.0;
    return v;
}

}  // namespace

// ---------------------------------------------------------------------------
// The guard itself.
// ---------------------------------------------------------------------------

TEST_CASE("domain_window is inert on a finite domain", "[unbounded][randomize]") {
    // The determinism contract: a model whose bounds are finite must sample from
    // exactly the bounds it declared, so its RNG draws — and therefore its whole
    // solve trajectory — are unchanged by the guard.
    struct Case {
        VarType type;
        double lb;
        double ub;
    };
    const std::vector<Case> cases = {
        {VarType::Bool, 0.0, 1.0},        {VarType::Int, -3.0, 7.0},
        {VarType::Int, 0.0, 5.0e7},       {VarType::Float, -1.0e9, 1.0e9},
        {VarType::Float, 1.5, 1.5},       {VarType::Float, -2.5, 100.0},
        // Past the clamp magnitudes: a *declared* bound is honoured, exactly as
        // the .nl reader honours a declared finite integer bound rather than
        // narrowing it to int_inf_clamp.
        {VarType::Int, -1.0e8, 1.0e8},    {VarType::Float, -1.0e12, 3.0e11},
    };
    for (const Case& c : cases) {
        const DomainWindow w = domain_window(scalar_var(c.type, c.lb, c.ub));
        REQUIRE(w.lo == c.lb);
        REQUIRE(w.hi == c.ub);
    }
}

TEST_CASE("random_in_domain is finite and in-domain on an unbounded domain",
          "[unbounded][randomize]") {
    struct Case {
        const char* label;
        VarType type;
        double lb;
        double ub;
    };
    const std::vector<Case> cases = {
        {"float (-inf, +inf)", VarType::Float, -kInf, kInf},
        {"float [0, +inf)", VarType::Float, 0.0, kInf},
        {"float (-inf, 0]", VarType::Float, -kInf, 0.0},
        {"float [1, +inf)", VarType::Float, 1.0, kInf},
        // A declared bound past the clamp magnitude: the sampling window has to
        // move to stay inside the domain rather than sit at -1e9.
        {"float (-inf, -2e9]", VarType::Float, -kInf, -2.0e9},
        {"float [2e9, +inf)", VarType::Float, 2.0e9, kInf},
        // Finite bounds whose *width* overflows trip the same precondition.
        {"float [-1e308, 1e308]", VarType::Float, -1.0e308, 1.0e308},
        {"int (-inf, +inf)", VarType::Int, -kInf, kInf},
        {"int [0, +inf)", VarType::Int, 0.0, kInf},
        {"int (-inf, 10]", VarType::Int, -kInf, 10.0},
        {"bool with infinite bounds", VarType::Bool, -kInf, kInf},
    };

    for (const Case& c : cases) {
        INFO(c.label);
        const Variable var = scalar_var(c.type, c.lb, c.ub);
        for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
            RNG rng(seed);
            const double v = random_in_domain(var, rng);
            REQUIRE(std::isfinite(v));
            REQUIRE(v >= c.lb);
            REQUIRE(v <= c.ub);
            if (c.type != VarType::Float) {
                REQUIRE(v == std::floor(v));  // Int/Bool stay integral
            }
        }
    }
}

TEST_CASE("an infinite Int bound does not sample INT64_MIN", "[unbounded][randomize]") {
    // The specific pre-fix artefact: `static_cast<int64_t>(-inf)` is
    // INT64_MIN, so the draw ranged over the whole int64 line.
    const Variable var = scalar_var(VarType::Int, -kInf, kInf);
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        RNG rng(seed);
        const double v = random_in_domain(var, rng);
        REQUIRE(std::abs(v) <= kRandomIntInfClamp);
    }
}

TEST_CASE("randomize_var keeps an unbounded model's assignment finite",
          "[unbounded][randomize]") {
    // The entry point search.cpp's initialisers and LNS's destroy step share.
    Model m = unit_circle_model();
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        RNG rng(seed);
        initialize_random(m, rng);
        REQUIRE(all_values_finite(m));
    }
}

// ---------------------------------------------------------------------------
// The kick path. Both cases below FAIL on the pre-fix engine.
// ---------------------------------------------------------------------------

TEST_CASE("a diversification kick keeps an unbounded model finite", "[unbounded][perturb]") {
    // Pre-fix: the first kick writes NaN (either from a per-variable draw or,
    // when those move nothing, from the guaranteed-move fallback of #109 —
    // which is precisely the small models this one is), full_evaluate spreads it
    // over the DAG, and nothing recovers.
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        INFO("seed " << seed);
        Model m = unit_circle_model();
        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng);
        fj.begin(/*set_initial_x=*/true);
        for (int kick = 0; kick < 4; ++kick) {
            fj.perturb(kDefaultP);
            REQUIRE(all_values_finite(m));
            REQUIRE(all_constraints_finite(m));
        }
    }
}

TEST_CASE("solve keeps improving after a diversification kick", "[unbounded][search]") {
    // End-to-end on the default path. `perturbation_period = 1` makes the search
    // kick after every non-improving batch, so kicks are reached well inside the
    // iteration budget; everything else is default.
    //
    // Pre-fix this fails: after the first kick every constraint is NaN,
    // max_real_violation maps that to +inf, so no assignment is ever
    // real-feasible again, record_best never fires, and no progress callback
    // after the first kick ever carries new_best.
    struct LastBest : SolveCallback {
        int perturbations_at_last_best = -1;
        void on_progress(const SolveProgress& p) override {
            if (p.new_best) {
                perturbations_at_last_best = p.perturbations;
            }
        }
    };

    Model m;
    int32_t x = m.float_var(-kInf, kInf, "x");
    int32_t y = m.float_var(-kInf, kInf, "y");
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(4.0)));
    m.minimize(m.sum({m.prod(x, x), m.prod(y, y)}));  // optimum 8 at x = y = 2
    m.close();

    SearchConfig cfg;
    cfg.max_iterations = 200000;
    cfg.perturbation_period = 1;
    LastBest cb;
    SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr,
                           /*lns_interval=*/3, &cb, cfg);

    REQUIRE(cb.perturbations_at_last_best >= 1);
    REQUIRE(r.feasible);
    REQUIRE(std::isfinite(r.objective));
    REQUIRE(all_values_finite(m));
}

// ---------------------------------------------------------------------------
// The LNS path.
// ---------------------------------------------------------------------------

TEST_CASE("LNS destroy/repair works on an unbounded model", "[unbounded][lns]") {
    // Finiteness alone is green pre-fix (destroy/repair rolls back a
    // NaN-poisoned repair), so the load-bearing assertion is that a repair is
    // ACCEPTED: pre-fix the destroyed variable is NaN, the repair can never
    // improve the acceptance key, and every round is rolled back.
    int accepted = 0;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        INFO("seed " << seed);
        Model m;
        int32_t x = m.float_var(-kInf, kInf, "x");
        int32_t y = m.float_var(-kInf, kInf, "y");
        m.add_constraint(m.geq(m.sum({x, y}), m.constant(10.0)));
        m.close();
        m.var_mut(vid(x)).value = 0.0;  // violated by 10
        m.var_mut(vid(y)).value = 0.0;
        full_evaluate(m);

        ViolationManager vm(m);
        RNG rng(seed);
        LNS lns(0.3);
        // time_limit 0 = no wall clock; the repair's iteration budget binds, so
        // the outcome is fully determined by the seed.
        accepted += lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0) ? 1 : 0;

        REQUIRE(all_values_finite(m));
        REQUIRE(all_constraints_finite(m));
    }
    REQUIRE(accepted > 0);
}
