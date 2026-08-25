// Who initialises what, and what the seed does and does not move (#108).
//
// `solve()` used to call `initialize_random` (randomising every variable) and
// then, a dozen lines later, `fj.begin(set_initial_x=true)`, which overwrote
// every Bool/Int/Float with the domain value closest to zero. The random scalar
// draws were therefore computed and immediately discarded, and the code read as
// though the seed varied the starting point when it did not.
//
// Resolution: FeasibilityJump owns the scalar start (the published Feasibility
// Jump initialisation) and `initialize_structured_random` covers only the types
// FJ cannot initialise. These tests pin both halves of that split, and the
// documented opt-in for callers who do want a randomised scalar start.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <limits>
#include <vector>

using namespace cbls;

namespace {

// Run `solve()` with neither a wall clock nor an iteration budget. The outer
// loop then breaks before its first batch, and `solve()` restores the state it
// snapshotted immediately after `fj.begin(...)` — so the model is left holding
// exactly the *starting* assignment, which is what these tests inspect.
SearchResult solve_no_budget(Model& model, uint64_t seed, const SearchConfig& base = {}) {
    SearchConfig cfg = base;
    cfg.max_iterations = 0;
    SearchResult r =
        solve(model, /*time_limit=*/0.0, seed, /*use_fj=*/true, nullptr, nullptr, 3, nullptr, cfg);
    // These tests are only meaningful if the outer loop really did no work — they
    // read the model expecting the *starting* assignment. That rests on solve()'s
    // "neither budget set, so nothing would ever stop the loop" guard breaking
    // before the first batch. Fail loudly here if that ever changes, rather than
    // silently inspecting a searched assignment.
    REQUIRE(r.iterations == 0);
    return r;
}

std::vector<double> scalar_values(const Model& model) {
    std::vector<double> out;
    for (const auto& var : model.variables()) {
        if (!is_structured(var.type)) {
            out.push_back(var.value);
        }
    }
    return out;
}

// A model whose scalar domains have distinct closest-to-zero points: one
// straddling zero, one strictly positive, one strictly negative, plus an Int and
// a Bool.
//
// The constraint asks for a sum >= 20 when the domains cap it at
// 4 + 5 - 2 + 7 + 1 = 15, so the model is in fact *infeasible*. That is
// deliberate and harmless here: every test below inspects the assignment before
// the search runs, and an infeasible model guarantees the constraint is violated
// at the start whatever the domains do. Don't reuse it for anything that expects
// to find a solution.
Model scalar_model() {
    Model m;
    auto straddling = m.float_var(-3.0, 4.0);  // clamp(0) =  0
    auto positive = m.float_var(2.0, 5.0);     // clamp(0) =  2
    auto negative = m.float_var(-5.0, -2.0);   // clamp(0) = -2
    auto integral = m.int_var(3, 7);           // clamp(0) =  3
    auto flag = m.bool_var();                  // clamp(0) =  0
    auto neg1 = m.constant(-1.0);
    auto twenty = m.constant(20.0);
    // 20 - (straddling + positive + negative + integral + flag) <= 0
    m.add_constraint(m.sum({twenty, m.prod(neg1, straddling), m.prod(neg1, positive),
                            m.prod(neg1, negative), m.prod(neg1, integral), m.prod(neg1, flag)}));
    m.minimize(m.sum({straddling, positive}));
    m.close();
    return m;
}

}  // namespace

TEST_CASE("solve starts each scalar at the domain value closest to zero", "[search][init]") {
    Model m = scalar_model();
    solve_no_budget(m, /*seed=*/1);

    // Order matches scalar_model()'s declaration order.
    REQUIRE(scalar_values(m) == std::vector<double>{0.0, 2.0, -2.0, 3.0, 0.0});
}

TEST_CASE("solve's scalar starting point does not vary with the seed", "[search][init]") {
    // This is the behaviour #108 settled on, and it is deliberate: the starting
    // point is the published Feasibility Jump one, not a random draw. Anyone
    // reading three identical results across three seeds on a scalar model is
    // seeing three runs that *began* at the same point.
    Model a = scalar_model();
    Model b = scalar_model();
    Model c = scalar_model();
    solve_no_budget(a, /*seed=*/1);
    solve_no_budget(b, /*seed=*/2);
    solve_no_budget(c, /*seed=*/98765);

    REQUIRE(scalar_values(a) == scalar_values(b));
    REQUIRE(scalar_values(b) == scalar_values(c));
}

TEST_CASE("solve still varies the structured starting point with the seed", "[search][init]") {
    // The other half of the split: FJ never touches List/Set, so
    // initialize_structured_random still owns them and the seed still moves them.
    int32_t order = 0;
    auto build = [&order]() {
        Model m;
        order = vid(m.list_var(10));
        auto x = m.float_var(0.0, 1.0);
        m.add_constraint(m.sum({m.prod(m.constant(-1.0), x)}));  // -x <= 0, holds at x = 0
        m.close();
        return m;
    };

    Model a = build();
    Model b = build();
    solve_no_budget(a, /*seed=*/1);
    solve_no_budget(b, /*seed=*/2);

    // Two independent permutations of 10 elements collide with probability
    // 1/10! ~ 2.8e-7; these are fixed seeds, so the check is deterministic.
    REQUIRE(a.var(order).elements != b.var(order).elements);
}

TEST_CASE("initialize_random plus skip_init gives a seed-varying scalar start", "[search][init]") {
    // The documented opt-in for callers who want a randomised scalar start:
    // randomise the assignment yourself, then tell solve() to keep it.
    SearchConfig cfg;
    cfg.skip_init = true;

    Model a = scalar_model();
    Model b = scalar_model();
    RNG rng_a(1);
    RNG rng_b(2);
    initialize_random(a, rng_a);
    initialize_random(b, rng_b);
    solve_no_budget(a, /*seed=*/1, cfg);
    solve_no_budget(b, /*seed=*/2, cfg);

    REQUIRE(scalar_values(a) != scalar_values(b));
    // And the random start survived rather than being clamped back to zero.
    REQUIRE(scalar_values(a) != std::vector<double>{0.0, 2.0, -2.0, 3.0, 0.0});
}

TEST_CASE("structured init consumes RNG independently of the scalar count", "[search][init]") {
    // The regression pin for #108. Correct behaviour is that solve() draws
    // *nothing* for scalars, so the number of scalars declared ahead of a List
    // cannot change which permutation that List receives at a given seed.
    //
    // Re-adding `initialize_random` to solve() fails this immediately: it draws
    // once per scalar first, so the List lands at a different position in the
    // stream and gets a different permutation for each N. That is what makes this
    // test discriminating where "does the start vary with the seed" is not — the
    // scalar *values* are identical either way, only the stream position differs.
    auto permutation_after_n_floats = [](int n_floats) {
        Model m;
        for (int i = 0; i < n_floats; ++i) {
            m.float_var(-3.0, 4.0);
        }
        auto order = m.list_var(8);
        m.add_constraint(m.sum({m.constant(-1.0)}));  // -1 <= 0, always satisfied
        m.close();
        solve_no_budget(m, /*seed=*/1);
        return m.var(vid(order)).elements;
    };

    const auto with_1 = permutation_after_n_floats(1);
    const auto with_50 = permutation_after_n_floats(50);

    REQUIRE(with_1.size() == 8);
    REQUIRE(with_1 == with_50);
}

TEST_CASE("solve's starting point is finite on an unbounded domain", "[search][init]") {
    // FJ's closest-to-zero start is well defined on any domain, and this pins
    // that solve() keeps using it: the specific values below are the
    // closest-to-zero point of each domain, so re-seeding the scalars from any
    // other rule (a uniform draw, say) fails here.
    //
    // Scope: the *starting point* only. Randomisation is safe on an unbounded
    // domain too, since #112 routed every draw through `domain_window`, but that
    // is pinned in tests/test_unbounded_domain.cpp, not here.
    const double inf = std::numeric_limits<double>::infinity();
    Model m;
    auto both = m.float_var(-inf, inf);                                          // clamp(0) = 0
    auto half = m.float_var(0.0, inf);                                           // clamp(0) = 0
    auto shifted = m.float_var(1.0, inf);                                        // clamp(0) = 1
    m.add_constraint(m.sum({m.constant(-1.0), m.prod(m.constant(0.0), both)}));  // -1 <= 0
    m.close();

    solve_no_budget(m, /*seed=*/1);

    REQUIRE(std::isfinite(m.var(vid(both)).value));
    REQUIRE(std::isfinite(m.var(vid(half)).value));
    REQUIRE(std::isfinite(m.var(vid(shifted)).value));
    REQUIRE(scalar_values(m) == std::vector<double>{0.0, 0.0, 1.0});
}

TEST_CASE("initialize_structured_random leaves every scalar untouched", "[search][init]") {
    Model m;
    auto order = m.list_var(6);
    auto chosen = m.set_var(6, 2, 4);
    auto x = m.float_var(-3.0, 4.0);
    auto n = m.int_var(3, 7);
    auto flag = m.bool_var();
    m.add_constraint(m.sum({m.constant(-1.0), m.prod(m.constant(0.0), x)}));  // -1 <= 0
    m.close();

    m.var_mut(vid(x)).value = 1.25;
    m.var_mut(vid(n)).value = 6.0;
    m.var_mut(vid(flag)).value = 1.0;

    RNG rng(7);
    initialize_structured_random(m, rng);

    REQUIRE(m.var(vid(x)).value == 1.25);
    REQUIRE(m.var(vid(n)).value == 6.0);
    REQUIRE(m.var(vid(flag)).value == 1.0);
    // The structured variables were populated.
    REQUIRE(m.var(vid(order)).elements.size() == 6);
    REQUIRE(m.var(vid(chosen)).elements.size() >= 2);
    REQUIRE(m.var(vid(chosen)).elements.size() <= 4);
}
