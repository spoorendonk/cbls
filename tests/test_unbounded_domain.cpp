// Regression tests for unbounded variable domains: #112 (randomising a variable
// over one) and #114 (generating FJ jump candidates for one — see the second
// block comment, below the #112 cases).
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
// with randomisation no longer able to inject one, that route is closed. It is
// not the only one: Float jump candidates are still drawn from `var.lb`/`var.ub`
// directly, so an unbounded model can still reach an infinite assignment by a
// path this fix does not touch. (The *Int* jump path was a second such route and
// is closed as of #114, below; Float remains open.)

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <limits>
#include <numeric>
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
        {VarType::Bool, 0.0, 1.0},
        {VarType::Int, -3.0, 7.0},
        {VarType::Int, 0.0, 5.0e7},
        {VarType::Float, -1.0e9, 1.0e9},
        {VarType::Float, 1.5, 1.5},  // pinned
        {VarType::Float, -2.5, 100.0},
        // Past the clamp magnitudes: a *declared* bound is honoured, exactly as
        // the .nl reader honours a declared finite integer bound rather than
        // narrowing it to int_inf_clamp.
        {VarType::Int, -1.0e8, 1.0e8},
        {VarType::Float, -1.0e12, 3.0e11},
        // Past 2^53, where an earlier attempt at #114 trimmed each Int bound
        // independently into the int64-nameable range. That was not inert:
        // [0, 1e17] came back as [0, 2^53-1] (halving the reachable jump), and
        // [-1e18, -1e17] collapsed to the single point -2^53 — which is not even
        // in the domain, so `random_in_domain` returned an out-of-domain value
        // (a #112 defect). The cast is trimmed at the cast instead; see
        // `int_sample_window`.
        {VarType::Int, 0.0, 1.0e17},
        {VarType::Int, -1.0e18, -1.0e17},
        {VarType::Int, 1.0e16, 1.0e16 + 5.0},
    };
    for (const Case& c : cases) {
        const DomainWindow w = domain_window(scalar_var(c.type, c.lb, c.ub));
        REQUIRE(w.lo == c.lb);
        REQUIRE(w.hi == c.ub);
    }
}

TEST_CASE("domain_window is a subset of the declared domain", "[unbounded][randomize]") {
    // The contract `randomize.h` states — "always a subset of the variable's own
    // domain, so a value drawn from it is in-domain by construction" — asserted
    // rather than assumed. It held for every case here until an Int-only trim
    // was added to `domain_window`: clamping each bound independently into
    // +/-2^53 moves a bound *outward* when the whole domain sits past it, so
    // [-1e18, -1e17] came back as [-2^53, -2^53], above the declared ub.
    struct Case {
        const char* label;
        VarType type;
        double lb;
        double ub;
    };
    const std::vector<Case> cases = {
        {"int [0, 10]", VarType::Int, 0.0, 10.0},
        {"int [0, 1e17]", VarType::Int, 0.0, 1.0e17},
        {"int [-1e18, -1e17]", VarType::Int, -1.0e18, -1.0e17},
        {"int [1e16, 1e16+5]", VarType::Int, 1.0e16, 1.0e16 + 5.0},
        {"int [1e18, 1e19]", VarType::Int, 1.0e18, 1.0e19},
        {"int [-1e300, 1e300]", VarType::Int, -1.0e300, 1.0e300},
        {"int [-1e308, 1e308] (width overflows)", VarType::Int, -1.0e308, 1.0e308},
        {"int (-inf, +inf)", VarType::Int, -kInf, kInf},
        {"int [0, +inf)", VarType::Int, 0.0, kInf},
        {"int (-inf, 0]", VarType::Int, -kInf, 0.0},
        {"int (-inf, -1e18]", VarType::Int, -kInf, -1.0e18},
        {"int [1e18, +inf)", VarType::Int, 1.0e18, kInf},
        {"float (-inf, +inf)", VarType::Float, -kInf, kInf},
        {"float (-inf, -2e9]", VarType::Float, -kInf, -2.0e9},
        {"float [2e9, +inf)", VarType::Float, 2.0e9, kInf},
        {"float [-1e308, 1e308] (width overflows)", VarType::Float, -1.0e308, 1.0e308},
        {"float [1.5, 1.5] (pinned)", VarType::Float, 1.5, 1.5},
        {"bool [0, 1]", VarType::Bool, 0.0, 1.0},
    };
    for (const Case& c : cases) {
        INFO(c.label);
        const DomainWindow w = domain_window(scalar_var(c.type, c.lb, c.ub));
        REQUIRE(std::isfinite(w.lo));
        REQUIRE(std::isfinite(w.hi));
        REQUIRE(w.lo <= w.hi);
        REQUIRE(w.lo >= c.lb);
        REQUIRE(w.hi <= c.ub);
    }
}

TEST_CASE("random_in_domain stays in domain past 2^53", "[unbounded][randomize]") {
    // `random_in_domain`'s Int arm casts the window to int64_t, so the window it
    // reads has to be nameable. Trimming it inside `domain_window` made these
    // domains *leave* the declared box; trimming at the cast keeps the draw
    // inside it, falling back to the untrimmed window when no int64_t range
    // names the domain at all. Measured before the fix: [-1e18, -1e17] returned
    // -9007199254740992, above the declared ub.
    struct Case {
        const char* label;
        double lb;
        double ub;
    };
    const std::vector<Case> cases = {
        {"int [-1e18, -1e17]", -1.0e18, -1.0e17},
        {"int [1e16, 1e16+5]", 1.0e16, 1.0e16 + 5.0},
        {"int [1e18, 1e19]", 1.0e18, 1.0e19},
        // Truncation toward zero, not magnitude: `static_cast<int64_t>(0.9)` is
        // 0, which is below the declared lb.
        {"int [0.9, 1.2]", 0.9, 1.2},
    };
    for (const Case& c : cases) {
        INFO(c.label);
        const Variable var = scalar_var(VarType::Int, c.lb, c.ub);
        for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
            RNG rng(seed);
            const double v = random_in_domain(var, rng);
            REQUIRE(std::isfinite(v));
            REQUIRE(v >= c.lb);
            REQUIRE(v <= c.ub);
        }
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

TEST_CASE("randomize_var keeps an unbounded model's assignment finite", "[unbounded][randomize]") {
    // The entry point search.cpp's initialisers and LNS's destroy step share.
    Model m = unit_circle_model();
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        RNG rng(seed);
        initialize_random(m, rng);
        REQUIRE(all_values_finite(m));
    }
}

TEST_CASE("standard moves stay finite on an unbounded domain", "[unbounded][moves]") {
    // `generate_standard_moves` is a public entry point (and a Python binding)
    // that read the raw bounds too. It is NOT on solve()'s default path —
    // structural_pass calls it only for List/Set — but `normal(0, (ub-lb)*0.1)`
    // is NaN when the width is infinite, and int_rand's cast hit INT64_MIN.
    Model m;
    int32_t x = m.float_var(-kInf, kInf, "x");
    int32_t n = m.int_var(0, 5, "n");
    m.add_constraint(m.leq(m.sum({x, n}), m.constant(1.0)));
    m.close();
    m.var_mut(vid(n)).ub = kInf;  // int_var takes `int`; a reader can write ±inf
    m.var_mut(vid(n)).value = 2.0;
    // A fresh Variable's value is its lower bound, so an unbounded-below Float
    // starts at -inf until FeasibilityJump::begin() replaces it with the
    // closest-to-zero point. Stand in for that here.
    m.var_mut(vid(x)).value = 0.0;

    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        RNG rng(seed);
        for (int32_t v : {vid(x), vid(n)}) {
            for (const Move& move : generate_standard_moves(m.var(v), rng)) {
                for (const Move::Change& ch : move.changes) {
                    INFO(move.move_type << " var=" << v);
                    REQUIRE(std::isfinite(ch.new_value));
                }
            }
        }
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

TEST_CASE("solve solves a model only a kick can escape", "[unbounded][search]") {
    // End-to-end, and the measurement from the issue. On `|x^2 + y^2 - 1| <= 0`
    // the closest-to-zero start (0, 0) is a stationary point of the only
    // constraint — both partials of |x^2 + y^2 - 1| vanish there — so no jump
    // value can move either variable and the search is stuck at violation 1
    // until a diversification kick moves it off the origin. The kick is
    // therefore load-bearing, not incidental.
    //
    // Pre-fix the kick writes NaN instead: `max_real_violation()` returns +inf
    // from then on, nothing is ever feasible again, and `solve()` returns the
    // pre-kick closest-approach state — `feasible = false`, `best_violation = 1`
    // — looking like an ordinary infeasible run.
    Model m = unit_circle_model();

    SearchConfig cfg;
    // Kick sooner than the default 100 stagnant batches so the budget below is
    // small enough to keep the test quick; everything else is default.
    cfg.perturbation_period = 4;
    cfg.max_iterations = 200000;
    SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr,
                           /*lns_interval=*/3, nullptr, cfg);

    REQUIRE(r.feasible);
    REQUIRE(r.best_violation <= cfg.feasibility_tolerance);
    REQUIRE(all_values_finite(m));
    REQUIRE(all_constraints_finite(m));
}

// ---------------------------------------------------------------------------
// The LNS path.
// ---------------------------------------------------------------------------

TEST_CASE("LNS destroy/repair works on an unbounded model", "[unbounded][lns]") {
    // The acceptance count is the load-bearing assertion. A NaN-poisoned repair
    // scores +inf in the lexicographic key and is always rolled back, so
    // "finiteness after the call" can be satisfied by never accepting anything —
    // requiring an accepted repair is what rules that out.
    //
    // (Finiteness turns out to fail pre-fix as well, for a second reason: with
    // the destroyed variable at NaN every partial derivative is NaN too, so the
    // repair finds no Newton candidate and falls back to the box constants —
    // which on an unbounded domain are +/-inf. The repair then "succeeds" at
    // x = +inf and is accepted.)
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

// ---------------------------------------------------------------------------
// List semantics are per-caller, not flattened by the shared helper.
// ---------------------------------------------------------------------------

TEST_CASE("ListOrder::Perturb keeps a List's elements, Regenerate need not",
          "[unbounded][randomize][list]") {
    // The three merged copies were NOT byte-equivalent: LNS shuffled a List's
    // current `elements`, the initialisers regenerated the order from iota.
    // Collapsing both onto Regenerate changed LNS's repair trajectory on every
    // List model (pharma-glsp among them; since retired) while the RNG draw
    // count stayed the same, so nothing caught it. Pin both semantics here.
    Model m;
    int32_t l = m.list_var(8, "l");
    m.close();
    Variable& var = m.var_mut(vid(l));

    // Drive the list away from iota so the two arms can be told apart at all.
    var.elements = {7, 0, 5, 2, 6, 1, 4, 3};
    const std::vector<int32_t> incumbent = var.elements;

    SECTION("Perturb preserves the multiset of elements") {
        for (uint64_t seed = 1; seed <= 50; ++seed) {
            var.elements = incumbent;
            RNG rng(seed);
            randomize_structured_var(var, rng, ListOrder::Perturb);

            REQUIRE(var.elements.size() == incumbent.size());
            std::vector<int32_t> got = var.elements;
            std::vector<int32_t> want = incumbent;
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            REQUIRE(got == want);
        }
    }

    SECTION("Perturb is exactly the pre-merge shuffle of the incumbent") {
        for (uint64_t seed = 1; seed <= 50; ++seed) {
            var.elements = incumbent;
            RNG a(seed);
            randomize_structured_var(var, a, ListOrder::Perturb);
            const std::vector<int32_t> via_helper = var.elements;

            std::vector<int32_t> direct = incumbent;
            RNG b(seed);
            b.shuffle(direct);  // what lns.cpp did before the merge

            REQUIRE(via_helper == direct);
        }
    }

    SECTION("Regenerate is exactly the pre-merge permutation") {
        for (uint64_t seed = 1; seed <= 50; ++seed) {
            var.elements = incumbent;
            RNG a(seed);
            randomize_structured_var(var, a, ListOrder::Regenerate);
            const std::vector<int32_t> via_helper = var.elements;

            RNG b(seed);
            const std::vector<int32_t> direct = b.permutation(8);  // what search.cpp did

            REQUIRE(via_helper == direct);
        }
    }

    SECTION("the two arms agree on a freshly built List and diverge after it moves") {
        // permutation(n) is iota-then-shuffle, so on a fresh List they coincide;
        // that is why initialisation is unaffected by the split.
        std::vector<int32_t> iota_order(8);
        std::iota(iota_order.begin(), iota_order.end(), 0);

        bool saw_divergence = false;
        for (uint64_t seed = 1; seed <= 50; ++seed) {
            var.elements = iota_order;
            RNG a(seed);
            randomize_structured_var(var, a, ListOrder::Perturb);
            const std::vector<int32_t> from_iota = var.elements;

            var.elements = iota_order;
            RNG b(seed);
            randomize_structured_var(var, b, ListOrder::Regenerate);
            REQUIRE(from_iota == var.elements);

            var.elements = incumbent;
            RNG c(seed);
            randomize_structured_var(var, c, ListOrder::Perturb);
            const std::vector<int32_t> moved = var.elements;

            var.elements = incumbent;
            RNG d(seed);
            randomize_structured_var(var, d, ListOrder::Regenerate);
            saw_divergence = saw_divergence || (moved != var.elements);
        }
        REQUIRE(saw_divergence);
    }
}

// ---------------------------------------------------------------------------
// FJ jump candidates on an unbounded Int (#114).
// ---------------------------------------------------------------------------
//
// The other half of the same hazard, and pre-existing rather than introduced by
// #112. `int_jump_candidates` truncated the *raw* declared bounds with
// `std::lround`, in `long`, and that failed three different ways:
//
//  - glibc's `lround` returns LONG_MIN for BOTH +inf and -inf. On
//    `(-inf, +inf)` the range collapsed (`ub - lb == 0`), the `ub <= lb`
//    early-out fired and the variable got no candidates at all — permanently
//    frozen in the jump table, never selected by a scan, unreachable by GLS
//    reweighting. On `[lb, +inf)` the same early-out fired (`LONG_MIN <= lb`).
//  - On `(-inf, ub]` with `ub >= 0` the early-out did not fire: `ub - lb`
//    overflowed `long` and wrapped negative, so the `<= 256` test chose the
//    EXHAUSTIVE arm and the loop ran up from LONG_MIN — ~9.2e18 candidates, each
//    a weighted_violation_delta. That case HANGS on the pre-fix engine rather
//    than returning a bad jump, so re-verify it under a timeout.
//  - On `(-inf, ub]` with `ub < 0` the width is a valid positive `long`, so the
//    grid arm ran instantly — off a `lb` of LONG_MIN, handing back jumps near
//    -9.2e18. Fast, and wrong.
//
// A *finite* bound past LONG_MAX (9.22e18) overflows `lround` the same way.
//
// The fix reads the bounds as doubles through `domain_window`, which substitutes
// only for an infinity, so a finite in-range domain keeps exactly the candidates
// it had.

TEST_CASE("an unbounded Int is offered jump candidates", "[unbounded][fj]") {
    struct Case {
        const char* label;
        double lb;
        double ub;
        bool at_least;  // constraint is `n >= bound`, else `n <= bound`
        double bound;
        double x0;
    };
    const std::vector<Case> cases = {
        {"int (-inf, +inf), n >= 5", -kInf, kInf, true, 5.0, 0.0},
        {"int [0, +inf), n >= 5", 0.0, kInf, true, 5.0, 0.0},
        // Finite, but past LONG_MAX (9.22e18): `lround` overflowed here exactly
        // as it did on an infinity.
        {"int [-1e19, 1e19], n >= 5", -1.0e19, 1.0e19, true, 5.0, 0.0},
        // Far from zero, so the sampling window sits past 2^53. An earlier
        // attempt at this issue trimmed the window inside `domain_window` and
        // collapsed both of these to a point, taking the early-out — the very
        // freeze this test exists to prevent, reintroduced one domain over. The
        // targets are inside the +/-1e6 window an infinite bound stands in for,
        // so a candidate that reaches them exists.
        {"int (-inf, -1e18], n <= -1e18-5e5", -kInf, -1.0e18, false, -1.0e18 - 5.0e5, -1.0e18},
        {"int [1e18, +inf), n >= 1e18+5e5", 1.0e18, kInf, true, 1.0e18 + 5.0e5, 1.0e18},
        // LAST deliberately: this is the one that HANGS pre-fix (see above), so
        // a re-verification run reports the others before it wedges.
        {"int (-inf, 0], n <= -5", -kInf, 0.0, false, -5.0, 0.0},
    };

    // A section per case rather than a bare loop: REQUIRE aborts the whole
    // TEST_CASE, so a plain loop would have reported only the first failure and
    // left the other cases unverified.
    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.label) {
            // `int_var` takes `int`, so the out-of-range bound is written
            // afterwards. No reader can produce one — .cbls goes through
            // `int_var(int, int)`, and the .nl/.mps readers clamp infinities to
            // 1e6/1e9 and then saturate to the `int` range — so this domain is
            // reachable only from the C++ API, which is what the issue is about.
            Model m;
            int32_t n = m.int_var(-10, 10, "n");
            m.add_constraint(c.at_least ? m.geq(n, m.constant(c.bound))
                                        : m.leq(n, m.constant(c.bound)));
            m.close();
            m.var_mut(vid(n)).lb = c.lb;
            m.var_mut(vid(n)).ub = c.ub;
            m.var_mut(vid(n)).value = c.x0;
            full_evaluate(m);

            const std::vector<double> weights(m.constraint_ids().size(), 1.0);
            const JumpResult r = compute_var_jump(m, weights, vid(n));

            // A positive score means some candidate strictly reduced weighted
            // violation, which is what "offered candidates" has to mean to be
            // worth anything: pre-fix these returned {x0, 0} — no candidates at
            // all — bar the hanging case and the two far-from-zero ones.
            REQUIRE(r.score > 0.0);
            REQUIRE(r.jump_value != c.x0);
            REQUIRE(std::isfinite(r.jump_value));
            REQUIRE(r.jump_value >= c.lb);
            REQUIRE(r.jump_value <= c.ub);
            // ...and it lands in the box randomisation samples, so the two paths
            // agree on this variable's searchable range. This is the half that
            // fails on the `lround` code for `(-inf, ub < 0]`: it returned a
            // jump near -9.2e18, in-domain but nowhere near the window.
            // Asserted against the window rather than against
            // `kRandomIntInfClamp`, because on a declared bound past the clamp
            // the window legitimately sits outside it. (Only the endpoints and
            // grid are bounded this way — the `x0 +/- 1` neighbours are clamped
            // to the raw domain, so a variable that has walked outside the
            // window keeps its local moves. Every x0 here is inside it.)
            const DomainWindow win = domain_window(m.var(vid(n)));
            REQUIRE(r.jump_value >= win.lo);
            REQUIRE(r.jump_value <= win.hi);
        }
    }
}

TEST_CASE("finite Int jump candidates are unchanged", "[unbounded][fj]") {
    // The determinism half of #114's acceptance criteria: a finite Int domain
    // must get the candidates the `lround` code gave it. `domain_window`
    // substitutes only for an infinity (pinned by "domain_window is inert on a
    // finite domain" above), so reading the bounds through it cannot move one.
    // These pin the resulting jump values for both arms of the branch — the
    // exhaustive one and the 32-point grid — against a future edit that would.
    SECTION("exhaustive arm (domain width <= 256)") {
        Model m;
        int32_t n = m.int_var(0, 10, "n");
        m.add_constraint(m.geq(n, m.constant(5.0)));
        m.close();
        m.var_mut(vid(n)).value = 0.0;
        full_evaluate(m);

        const std::vector<double> weights(m.constraint_ids().size(), 1.0);
        const JumpResult r = compute_var_jump(m, weights, vid(n));
        // Candidates are 0..10 in order; 5 is the first to reach violation 0.
        REQUIRE(r.jump_value == 5.0);
        REQUIRE(r.score == 5.0);
    }

    SECTION("grid arm (domain width > 256)") {
        Model m;
        int32_t n = m.int_var(0, 100000, "n");
        // |n - 3125| <= 0: exactly one feasible value, and it is grid point
        // k = 1 of `lb + (k/32) * (ub - lb)`. Nothing else in the candidate set
        // reaches it, so the assertion pins the grid formula itself.
        m.add_constraint(m.leq(m.abs_expr(m.sum({n, m.constant(-3125.0)})), m.constant(0.0)));
        m.close();
        m.var_mut(vid(n)).value = 0.0;
        full_evaluate(m);

        const std::vector<double> weights(m.constraint_ids().size(), 1.0);
        const JumpResult r = compute_var_jump(m, weights, vid(n));
        REQUIRE(r.jump_value == 3125.0);
        REQUIRE(r.score == 3125.0);
    }

    SECTION("grid arm past 2^53") {
        // The two sections above sit far inside 2^53, so they say nothing about
        // the range where determinism actually broke: trimming the window inside
        // `domain_window` clipped this domain's upper bound to 2^53-1, and the
        // jump went from 1e17 — which satisfies the row outright — to
        // 9007199254740991, leaving it violated by ~1e15.
        Model m;
        int32_t n = m.int_var(0, 10, "n");
        m.add_constraint(m.geq(n, m.constant(1.0e16)));
        m.close();
        m.var_mut(vid(n)).ub = 1.0e17;  // `int_var` takes `int`; see above
        m.var_mut(vid(n)).value = 0.0;
        full_evaluate(m);

        const std::vector<double> weights(m.constraint_ids().size(), 1.0);
        const JumpResult r = compute_var_jump(m, weights, vid(n));
        REQUIRE(r.jump_value == 1.0e17);
        REQUIRE(r.score == 1.0e16);
    }
}

TEST_CASE("solve moves a free Int that must move for feasibility", "[unbounded][fj][search]") {
    // End-to-end, and the acceptance criterion from the issue.
    // `set_initial_assignment` puts a free Int at the closest-to-zero point (0),
    // and `|n - 123457| <= 0` admits exactly one value, so the run reaches
    // feasibility only if the jump machinery actually walks the variable there.
    //
    // Pre-fix the variable is frozen at 0: the only thing that can move it is a
    // diversification kick, which draws uniformly over +/-1e6 and so hits the
    // one feasible value with probability 5e-7 per kick.
    struct Case {
        const char* label;
        double lb;
        double ub;
    };
    const std::vector<Case> cases = {
        {"int (-inf, +inf)", -kInf, kInf},
        {"int [0, +inf)", 0.0, kInf},
    };

    for (const Case& c : cases) {
        DYNAMIC_SECTION(c.label) {  // per-case reporting; see the test above
            Model m;
            int32_t n = m.int_var(0, 1, "n");
            m.add_constraint(m.leq(m.abs_expr(m.sum({n, m.constant(-123457.0)})), m.constant(0.0)));
            m.close();
            m.var_mut(vid(n)).lb = c.lb;
            m.var_mut(vid(n)).ub = c.ub;

            SearchConfig cfg;
            cfg.max_iterations = 200000;
            SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr,
                                   nullptr, /*lns_interval=*/3, nullptr, cfg);

            REQUIRE(r.feasible);
            REQUIRE(r.best_violation <= cfg.feasibility_tolerance);
            REQUIRE(m.var(vid(n)).value == 123457.0);
        }
    }
}

// #114, from independent review of b94cfe5: `movable_domain` truncated its
// window with `static_cast<int64_t>`, which rounds toward zero and so called
// `[0.9, 1.2]` movable. Only one integer lies in that domain, so nothing can
// move; the forced perturbation path then drew `rng.integers(0, 1) == 0` and
// wrote 0.0, BELOW the declared lb. `floor(hi) - ceil(lo) >= 1.0` is exact at
// any magnitude and calls the domain immovable, so the kick is a no-op instead.
//
// `perturb(0.0)` reaches that path deterministically: every per-variable draw
// fails `rng_.random() >= 0.0`, so nothing moves and the fallback runs.
TEST_CASE("a fractional Int domain holding one integer is not perturbed out of domain",
          "[unbounded][fj]") {
    for (uint64_t seed = 1; seed <= 5; ++seed) {
        DYNAMIC_SECTION("seed " << seed) {
            Model m;
            int32_t n = m.int_var(0, 2, "n");
            m.add_constraint(m.leq(n, m.constant(2.0)));
            m.close();
            m.var_mut(vid(n)).lb = 0.9;
            m.var_mut(vid(n)).ub = 1.2;
            m.var_mut(vid(n)).value = 1.0;
            full_evaluate(m);

            ViolationManager vm(m);
            RNG rng(seed);
            FeasibilityJump fj(m, vm, rng, GFJConfig{});
            for (int i = 0; i < 20; ++i) {
                fj.perturb(0.0);
                INFO("kick " << i);
                REQUIRE(m.var(vid(n)).value >= 0.9);
                REQUIRE(m.var(vid(n)).value <= 1.2);
            }
        }
    }
}
