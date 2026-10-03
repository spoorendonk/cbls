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
#include <random>
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
        REQUIRE(fj.scan_sets_consistent());
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
        // The commit returns with Q' as the last selection left it: the chosen
        // var must have left Q' together with its flag.
        REQUIRE(fj.scan_sets_consistent());
    }
}

TEST_CASE("apply_jump's sample is distinct and reaches every queued variable", "[fj][scan_set]") {
    // Two improving variables among 200 zero-score ones: y2 (score 2, queued
    // first) and y1 (score 1, queued last). With fewer positives than the
    // sample size, apply_jump drains Q and must examine both, so it applies y2
    // on every seed. A sampler that counts a positive without moving it out of
    // the draw range both re-draws it and strands whatever sits at the prefix
    // slot it claimed -- y2 at slot 0 whenever y1 is found first.
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        CAPTURE(seed);
        Model m;
        const int32_t y2 = m.bool_var();
        std::vector<int32_t> c0_terms{m.prod(m.constant(2.0), y2)};
        std::vector<int32_t> xs;
        for (int i = 0; i < kFiller; ++i) {
            xs.push_back(m.bool_var());
            c0_terms.push_back(xs.back());
        }
        const int32_t y1 = m.bool_var();
        c0_terms.push_back(y1);
        // c0: 2*y2 + sum(x) + y1 >= 2, violated by 2 at all-zero.
        m.add_constraint(m.geq(m.sum(c0_terms), m.constant(2.0)));
        for (const int32_t x : xs) {
            m.add_constraint(m.leq(x, m.constant(0.0)));
        }
        m.close();

        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng, GFJConfig{});
        fj.begin(/*set_initial_x=*/false);
        REQUIRE(compute_var_jump(m, vm.weights, vid(y2)).score == 2.0);
        REQUIRE(compute_var_jump(m, vm.weights, vid(y1)).score == 1.0);

        REQUIRE(fj.batch(/*batch_iterations=*/1));
        REQUIRE(m.var(vid(y2)).value == 1.0);
        REQUIRE(m.var(vid(y1)).value == 0.0);
        REQUIRE(fj.scan_sets_consistent());
    }
}

TEST_CASE("scan-set bookkeeping survives batches and Novelty jumps", "[fj][novelty][scan_set]") {
    // A random over-constrained integer model, so batches bump, Novelty picks,
    // backtracks and re-queues, and both scan sets are swap-removed from many
    // times. After every step each set must agree with its membership flags.
    std::mt19937 gen(4242);
    std::uniform_int_distribution<int> pick_var(0, 59);
    std::uniform_int_distribution<int> pick_coef(-3, 3);
    std::uniform_int_distribution<int> pick_rhs(-2, 6);
    Model m;
    std::vector<int32_t> vars;
    for (int i = 0; i < 60; ++i) {
        vars.push_back(m.int_var(0, 4));
    }
    for (int r = 0; r < 130; ++r) {
        std::vector<int32_t> terms{m.constant(static_cast<double>(-pick_rhs(gen)))};
        for (int k = 0; k < 4; ++k) {
            const int coef = pick_coef(gen);
            if (coef != 0) {
                terms.push_back(m.prod(m.constant(static_cast<double>(coef)),
                                       vars[static_cast<size_t>(pick_var(gen))]));
            }
        }
        m.add_constraint(m.sum(terms));  // sum(a x) - rhs <= 0
    }
    m.close();

    ViolationManager vm(m);
    RNG rng(17);
    GFJConfig cfg;
    cfg.two_phase = false;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    int novelty_calls = 0;
    for (int b = 0; b < 40; ++b) {
        CAPTURE(b);
        fj.batch(200);
        REQUIRE(fj.scan_sets_consistent());
        if (fj.all_satisfied()) {
            break;
        }
        fj.apply_novelty_jump();
        ++novelty_calls;
        REQUIRE(fj.scan_sets_consistent());
        REQUIRE(fj.novelty_moves_last_call() <= FeasibilityJump::novelty_work_budget());
        fj.resync();
        REQUIRE(fj.scan_sets_consistent());
    }
    REQUIRE(novelty_calls > 0);  // the Novelty half actually ran
}

TEST_CASE("Novelty removes the var it picks from the scan set, not another",
          "[fj][novelty][scan_set]") {
    // c0: a + 2b >= 2, violated by 2 at (0, 0). Both flips pass F at the root on
    // their score alone (a: +1, b: +2), so a full drain samples both and picks
    // b, whose single move commits and reaches feasibility -- apply_novelty_jump
    // returns with b on the stack. Removing the var at the wrong prefix slot
    // (a's, whenever a was drawn first) would leave b in Q' while it is on the
    // stack, which scan_sets_consistent reports.
    for (uint64_t seed = 1; seed <= kSeeds; ++seed) {
        CAPTURE(seed);
        Model m;
        const int32_t a = m.bool_var();
        const int32_t b = m.bool_var();
        m.add_constraint(m.geq(m.sum({a, m.prod(m.constant(2.0), b)}), m.constant(2.0)));
        m.close();

        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng, GFJConfig{});
        fj.begin(/*set_initial_x=*/false);

        REQUIRE(fj.apply_novelty_jump());
        REQUIRE(m.var(vid(b)).value == 1.0);
        REQUIRE(m.var(vid(a)).value == 0.0);
        REQUIRE(fj.scan_sets_consistent());
    }
}

TEST_CASE("Novelty stops trying siblings once its budget is spent", "[fj][novelty][scan_set]") {
    // Infeasible by construction: A: v >= 1 is violated at entry; fixing it
    // breaks R: v - sum(u) <= 0, and each u_k that repairs R breaks its own
    // S_k: u_k <= 0. At the root v passes F (novelty 1 - eps, score 0). Below
    // it every u_k passes F once (novelty 1 - eps, score 0), never commits,
    // and fails F after its undo has promoted S_k. A level that only checked
    // its discrepancy budget and the work cap on entry tried all K of them,
    // so one call applied ~K moves against a cap of novelty_work_budget().
    constexpr int kSiblings = 600;
    static_assert(kSiblings > FeasibilityJump::novelty_work_budget());
    Model m;
    const int32_t v = m.bool_var();
    std::vector<int32_t> r_terms{v};
    std::vector<int32_t> us;
    for (int k = 0; k < kSiblings; ++k) {
        us.push_back(m.bool_var());
        r_terms.push_back(m.prod(m.constant(-1.0), us.back()));
    }
    m.add_constraint(m.geq(v, m.constant(1.0)));               // A
    m.add_constraint(m.leq(m.sum(r_terms), m.constant(0.0)));  // R
    for (const int32_t u : us) {
        m.add_constraint(m.leq(u, m.constant(0.0)));  // S_k
    }
    m.close();

    ViolationManager vm(m);
    RNG rng(5);
    FeasibilityJump fj(m, vm, rng, GFJConfig{});
    fj.begin(/*set_initial_x=*/false);
    REQUIRE_FALSE(fj.apply_novelty_jump());
    CAPTURE(fj.novelty_moves_last_call());
    REQUIRE(fj.novelty_moves_last_call() > 0);  // the search actually ran
    REQUIRE(fj.novelty_moves_last_call() <= FeasibilityJump::novelty_work_budget());
    REQUIRE(fj.scan_sets_consistent());
}
