// Novelty compound-move legs under propagation (B2, #330/#339):
// GFJConfig::compound_propagation fixes each leg in a DomainPropagator over the
// model's linear rows, and later legs draw their candidates from the implied
// domains (compute_var_jump's `box`). Undo and commit backtrack the domains.

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/domain_propagation.h>
#include <cstdint>
#include <random>
#include <vector>

using namespace cbls;

namespace {

// x + y <= 5 and y >= 8, with y's row weighted 10: at x = 4 every y in
// [2, 10] breaks the first row by less than its gain on the second, so the
// unboxed argmin leaves [0, 1] (the domain fixing x = 4 implies for y).
struct BoxFixture {
    Model m;
    int32_t x = -1;
    int32_t y = -1;
    std::vector<double> weights{1.0, 10.0};

    explicit BoxFixture(VarType y_type) {
        const int32_t xh = m.int_var(0, 10);
        const int32_t yh = y_type == VarType::Int ? m.int_var(0, 10) : m.float_var(0.0, 10.0);
        m.add_constraint(m.leq(m.sum({xh, yh}), m.constant(5.0)));
        m.add_constraint(m.geq(yh, m.constant(8.0)));
        m.close();
        x = vid(xh);
        y = vid(yh);
        m.var_mut(x).value = 4.0;
        m.var_mut(y).value = 0.0;
        full_evaluate(m);
    }
};

// A closed-form scorer with every row eligible, as FJ sets one up.
LinearJumpScorer make_scorer(const Model& m) {
    LinearJumpScorer sc(m);
    sc.resize_rows(m.constraint_ids().size());
    for (size_t c = 0; c < m.constraint_ids().size(); ++c) {
        sc.set_row_eligible(static_cast<int32_t>(c), true);
    }
    return sc;
}

// The trajectory fence's random over-constrained integer model (as in
// test_fj_novelty_incremental.cpp): Novelty reaches local minima, backtracks
// and bumps, so legs narrow domains and some fixes conflict.
void build_random_int_model(Model& m, uint32_t seed, int num_vars, int num_rows) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> pick_var(0, num_vars - 1);
    std::uniform_int_distribution<int> pick_coef(1, 3);
    std::uniform_int_distribution<int> pick_sign(0, 1);
    std::uniform_int_distribution<int> pick_rhs(-2, 6);
    std::vector<int32_t> handles;
    handles.reserve(static_cast<size_t>(num_vars));
    for (int i = 0; i < num_vars; ++i) {
        handles.push_back(m.int_var(0, 4));
    }
    for (int r = 0; r < num_rows; ++r) {
        std::vector<int> used;
        while (used.size() < 4) {
            const int v = pick_var(gen);
            if (std::find(used.begin(), used.end(), v) == used.end()) {
                used.push_back(v);
            }
        }
        std::vector<int32_t> terms;
        for (const int v : used) {
            const double coef = (pick_sign(gen) != 0 ? 1.0 : -1.0) * pick_coef(gen);
            terms.push_back(m.prod(m.constant(coef), handles[static_cast<size_t>(v)]));
        }
        const int32_t sum = m.sum(terms);
        const double rhs = pick_rhs(gen);
        m.add_constraint(r % 2 == 0 ? m.leq(sum, m.constant(rhs)) : m.geq(sum, m.constant(rhs)));
    }
}

// The same shape with every row nonlinear (a product of two vars plus a
// third): no row is affine, so the propagator stores nothing and narrows
// nothing. The third var lets a single move repair a row from all-zero.
void build_random_nonlinear_model(Model& m, uint32_t seed, int num_vars, int num_rows) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> pick_var(0, num_vars - 1);
    std::uniform_int_distribution<int> pick_rhs(0, 8);
    std::vector<int32_t> handles;
    handles.reserve(static_cast<size_t>(num_vars));
    for (int i = 0; i < num_vars; ++i) {
        handles.push_back(m.int_var(0, 4));
    }
    for (int r = 0; r < num_rows; ++r) {
        std::vector<int> used;
        while (used.size() < 3) {
            const int v = pick_var(gen);
            if (std::find(used.begin(), used.end(), v) == used.end()) {
                used.push_back(v);
            }
        }
        auto h = [&](int k) { return handles[static_cast<size_t>(used[static_cast<size_t>(k)])]; };
        const int32_t body = m.sum({m.prod(h(0), h(1)), h(2)});
        const double rhs = pick_rhs(gen);
        m.add_constraint(r % 2 == 0 ? m.leq(body, m.constant(rhs)) : m.geq(body, m.constant(rhs)));
    }
}

struct NoveltyRun {
    std::vector<double> values;
    int64_t moves = 0;
    int64_t commits = 0;
    int64_t bumps = 0;
    int64_t tightenings = 0;
    int64_t conflicts = 0;
    bool domains_at_root = true;
    std::vector<std::string> audit;
    bool scan_sets = false;
};

template <class Build>
NoveltyRun run_novelty(Build&& build, uint32_t seed, bool propagate, int batches) {
    Model m;
    build(m);
    m.close();
    ViolationManager vm(m);
    RNG rng(seed);
    GFJConfig cfg;
    cfg.compound_propagation = propagate;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/false);
    for (int b = 0; b < batches; ++b) {
        if (fj.novelty_batch(200)) {
            break;
        }
    }
    NoveltyRun out;
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        out.values.push_back(m.var(v).value);
    }
    out.moves = fj.novelty_moves();
    out.commits = fj.novelty_commits();
    out.bumps = fj.novelty_weight_bumps();
    out.tightenings = fj.novelty_propagation_tightenings();
    out.conflicts = fj.novelty_propagation_conflicts();
    if (const DomainPropagator* p = fj.novelty_propagator()) {
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            if (p->lo(v) != m.var(v).lb || p->hi(v) != m.var(v).ub) {
                out.domains_at_root = false;
            }
        }
    }
    out.audit = fj.audit();
    out.scan_sets = fj.scan_sets_consistent();
    return out;
}

}  // namespace

TEST_CASE("a narrowed box keeps an Int jump's candidates inside it",
          "[fj][novelty][compound_propagation]") {
    for (const bool exact : {false, true}) {
        for (const bool closed_form : {false, true}) {
            BoxFixture f(VarType::Int);
            LinearJumpScorer linear = make_scorer(f.m);
            LinearJumpScorer* scorer = closed_form ? &linear : nullptr;
            CAPTURE(exact, closed_form);
            const JumpResult free_jump = compute_var_jump(f.m, f.weights, f.y, false, scorer,
                                                          kDefaultFeasibilityTolerance, exact);
            REQUIRE(free_jump.jump_value > 1.0);  // the fixture's premise
            const JumpBox box{0.0, 1.0};
            const JumpResult boxed =
                compute_var_jump(f.m, f.weights, f.y, false, scorer, kDefaultFeasibilityTolerance,
                                 exact, nullptr, &box);
            REQUIRE(boxed.jump_value == 1.0);
            REQUIRE(boxed.score > 0.0);
            REQUIRE(f.m.var(f.y).lb == 0.0);  // the declared box is not written
            REQUIRE(f.m.var(f.y).ub == 10.0);
        }
    }
}

TEST_CASE("a narrowed box keeps a Float jump's candidates inside it",
          "[fj][novelty][compound_propagation]") {
    BoxFixture f(VarType::Float);
    LinearJumpScorer linear = make_scorer(f.m);
    const JumpResult free_jump = compute_var_jump(f.m, f.weights, f.y, false, &linear);
    REQUIRE(free_jump.jump_value > 1.0);
    const JumpBox box{0.0, 1.0};
    const JumpResult boxed = compute_var_jump(f.m, f.weights, f.y, false, &linear,
                                              kDefaultFeasibilityTolerance, false, nullptr, &box);
    REQUIRE(boxed.jump_value > 0.0);
    REQUIRE(boxed.jump_value <= 1.0);
}

TEST_CASE("a box narrowed to one value offers it even from outside the box",
          "[fj][novelty][compound_propagation]") {
    for (const VarType t : {VarType::Int, VarType::Float}) {
        BoxFixture f(t);
        LinearJumpScorer linear = make_scorer(f.m);
        CAPTURE(static_cast<int>(t));
        const JumpBox point{3.0, 3.0};  // y = 0 lies outside it
        const JumpResult r = compute_var_jump(f.m, f.weights, f.y, false, &linear,
                                              kDefaultFeasibilityTolerance, false, nullptr, &point);
        REQUIRE(r.jump_value == 3.0);
        // An Int box with no integer in it offers nothing.
        if (t == VarType::Int) {
            const JumpBox empty{2.2, 2.8};
            const JumpResult none =
                compute_var_jump(f.m, f.weights, f.y, false, &linear, kDefaultFeasibilityTolerance,
                                 false, nullptr, &empty);
            REQUIRE(none.jump_value == 0.0);  // x0: no jump
            REQUIRE(none.score == 0.0);
        }
    }
}

TEST_CASE("a box no narrower than the declared one changes no jump",
          "[fj][novelty][compound_propagation]") {
    BoxFixture f(VarType::Int);
    LinearJumpScorer linear = make_scorer(f.m);
    const JumpResult plain = compute_var_jump(f.m, f.weights, f.y, false, &linear);
    const JumpBox wide{-5.0, 20.0};
    const JumpResult boxed = compute_var_jump(f.m, f.weights, f.y, false, &linear,
                                              kDefaultFeasibilityTolerance, false, nullptr, &wide);
    REQUIRE(boxed.jump_value == plain.jump_value);
    REQUIRE(boxed.score == plain.score);
}

TEST_CASE("compound propagation narrows legs, and undo and commit restore the domains",
          "[fj][novelty][compound_propagation]") {
    // Over a few seeds, so that both counters are seen to move; on every seed
    // the domains are back at the declared boxes after each batch, and FJ's
    // own state passes the from-scratch audit.
    int64_t tightenings = 0;
    int64_t conflicts = 0;
    for (uint32_t seed = 1; seed <= 6; ++seed) {
        const NoveltyRun r = run_novelty(
            [seed](Model& m) { build_random_int_model(m, seed, 12, 14); }, seed, true, 8);
        CAPTURE(seed, r.audit);
        REQUIRE(r.moves > 0);
        REQUIRE(r.domains_at_root);
        REQUIRE(r.audit.empty());
        REQUIRE(r.scan_sets);
        REQUIRE(r.conflicts <= r.moves);
        tightenings += r.tightenings;
        conflicts += r.conflicts;
    }
    CAPTURE(tightenings, conflicts);
    REQUIRE(tightenings > 0);
    REQUIRE(conflicts > 0);
}

TEST_CASE("compound propagation with no linear row leaves the trajectory unchanged",
          "[fj][novelty][compound_propagation]") {
    // Nothing to narrow: the hook must then add nothing -- not a draw, not a
    // re-queue -- so the run is the flag-off run move for move.
    for (uint32_t seed = 1; seed <= 3; ++seed) {
        auto build = [seed](Model& m) { build_random_nonlinear_model(m, seed, 10, 12); };
        const NoveltyRun off = run_novelty(build, seed, false, 6);
        const NoveltyRun on = run_novelty(build, seed, true, 6);
        CAPTURE(seed);
        REQUIRE(off.moves > 0);
        REQUIRE(on.values == off.values);
        REQUIRE(on.moves == off.moves);
        REQUIRE(on.commits == off.commits);
        REQUIRE(on.bumps == off.bumps);
        REQUIRE(on.tightenings == 0);
        REQUIRE(on.conflicts == 0);
        REQUIRE(off.tightenings == 0);
    }
}
