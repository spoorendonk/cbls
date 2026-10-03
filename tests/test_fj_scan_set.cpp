// Scan-set sampling must not declare a local minimum while Q still holds an
// improving variable (#206). ViolationLS (Davies et al. CPAIOR 2024) Algorithm 2
// loops while Q is non-empty and removes a non-positive variable without
// counting it; OR-Tools' ScanRelevantVariables has no cap either. A draw cap
// that also counted those removals gave up on a Q in which improving variables
// are rare, the GLS loop bumped the weights, and the bump re-queued the whole
// row, so the false minimum repeated. Novelty Jump's selection had the same
// shape: draws that failed the filter F counted toward its sample of 3.
//
// Each model below hides one or two useful variables among 200 that are not,
// in a single violated row. The capped code finds the useful one with
// probability ~0.28 (apply_jump) or ~0.09 (Novelty) per seed, so twenty seeds
// make the unfixed code fail with probability > 1 - 1e-10; the fixed code is
// deterministic here.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <vector>

using namespace cbls;

namespace {

constexpr int kFiller = 200;
constexpr uint64_t kSeeds = 20;

}  // namespace

TEST_CASE("apply_jump finds a rare improving variable without a weight bump", "[fj][scan_set]") {
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        CAPTURE(seed);
        // c0: sum(x) + y >= 1, violated at all-zero. ci: x_i <= 0, satisfied.
        // Flipping x_i fixes c0 (-1) and breaks ci (+1): score 0, so it is
        // removed from Q. Flipping y fixes c0 alone: score +1, the only
        // improving move among 201 queued variables.
        Model m;
        std::vector<int32_t> c0_terms;
        std::vector<int32_t> xs;
        for (int i = 0; i < kFiller; ++i) {
            xs.push_back(m.bool_var());
            c0_terms.push_back(xs.back());
        }
        const int32_t y = m.bool_var();
        c0_terms.push_back(y);
        m.add_constraint(m.geq(m.sum(c0_terms), m.constant(1.0)));
        for (const int32_t x : xs) {
            m.add_constraint(m.leq(x, m.constant(0.0)));
        }
        m.close();

        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng, GFJConfig{});
        fj.begin(/*set_initial_x=*/false);
        REQUIRE_FALSE(fj.all_satisfied());

        // One GLS iteration: it must apply y's jump, not bump.
        REQUIRE(fj.batch(/*batch_iterations=*/1));
        REQUIRE(m.var(vid(y)).value == 1.0);
        for (const double w : vm.weights) {
            REQUIRE(w == 1.0);
        }
    }
}

TEST_CASE("Novelty selection keeps scanning past filter-failing variables",
          "[fj][novelty][scan_set]") {
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        CAPTURE(seed);
        // The x != y local optimum of the Novelty Jump test, with 200 z_j added
        // to x's violated row with a NEGATIVE sign: flipping a z_j only worsens
        // that row, so its novelty score and its score are both negative and it
        // fails F. Q' = {x, y, z_*}; only x and y pass F, and the compound move
        // x->1, y->0 is the way out.
        Model m;
        const int32_t x = m.bool_var();
        const int32_t y = m.bool_var();
        std::vector<int32_t> row_terms{x};
        for (int j = 0; j < kFiller; ++j) {
            row_terms.push_back(m.prod(m.constant(-1.0), m.bool_var()));
        }
        m.add_constraint(m.neq(x, y));                               // x != y
        m.add_constraint(m.geq(m.sum(row_terms), m.constant(1.0)));  // x - sum(z) >= 1
        m.add_constraint(m.leq(y, m.constant(0.0)));                 // y <= 0
        m.close();

        m.var_mut(vid(x)).value = 0.0;
        m.var_mut(vid(y)).value = 1.0;
        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng, GFJConfig{});
        fj.begin(/*set_initial_x=*/false);

        REQUIRE(compute_var_jump(m, vm.weights, vid(x)).score <= 0.0);
        REQUIRE(compute_var_jump(m, vm.weights, vid(y)).score <= 0.0);

        REQUIRE(fj.apply_novelty_jump());
        REQUIRE(vm.is_feasible());
        REQUIRE(m.var(vid(x)).value == 1.0);
        REQUIRE(m.var(vid(y)).value == 0.0);
    }
}
