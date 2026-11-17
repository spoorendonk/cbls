// Fix-propagate-repair's dive engine (#425): orders, value rules, modes.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/domain_propagation.h>
#include <cbls/fpr.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace cbls;

namespace {

// A closed, evaluated model's store, root-propagated propagator and engine.
struct Rig {
    explicit Rig(Model& model) : store(model), prop(store, model), fpr(model, store, prop) {
        REQUIRE(fpr.propagate_root());
    }
    LinearRowStore store;
    DomainPropagator prop;
    FprSearch fpr;
};

void finish(Model& m) {
    m.close();
    full_evaluate(m);
}

double val(const FprResult& r, int32_t handle) {
    return r.values[static_cast<size_t>(vid(handle))];
}

}  // namespace

TEST_CASE("diveprop solves a chain that fixing without propagation breaks", "[fpr]") {
    // x_i + x_{i+1} == 1: Up alone sets every x to 1; one fix propagates the rest.
    Model m;
    std::vector<int32_t> x;
    x.reserve(6);
    for (int i = 0; i < 6; ++i) {
        x.push_back(m.bool_var());
    }
    for (size_t i = 0; i + 1 < x.size(); ++i) {
        m.add_constraint(m.eq_expr(m.sum({x[i], x[i + 1]}), m.constant(1.0)));
    }
    finish(m);
    Rig rig(m);
    RNG rng(1);
    FprResult r;
    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Diveprop}, 0, rng, r);
    CHECK(r.complete);
    CHECK(r.wants_repair);
    CHECK(r.nodes == 1);
    CHECK(r.conflicts == 0);
    for (size_t i = 0; i < x.size(); ++i) {
        CHECK(val(r, x[i]) == (i % 2 == 0 ? 1.0 : 0.0));
    }
}

TEST_CASE("dfs backtracks out of a conflict to a feasible leaf; diveprop keeps the fixing",
          "[fpr]") {
    // a = 1 forces b = c = 0 against b + c >= 1: refuted only by the cascade,
    // so the root pass leaves a free.
    Model m;
    const int32_t a = m.bool_var();
    const int32_t b = m.bool_var();
    const int32_t c = m.bool_var();
    m.add_constraint(m.leq(m.sum({a, b}), m.constant(1.0)));
    m.add_constraint(m.leq(m.sum({a, c}), m.constant(1.0)));
    m.add_constraint(m.geq(m.sum({b, c}), m.constant(1.0)));
    m.add_constraint(m.geq(m.sum({a, b, c}), m.constant(1.0)));  // a down lock: no trivial rounding
    finish(m);
    Rig rig(m);
    REQUIRE(rig.prop.hi(vid(a)) == 1.0);
    RNG rng(1);
    FprResult r;
    const int32_t depth = rig.prop.depth();
    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
    CHECK(r.complete);
    CHECK_FALSE(r.wants_repair);
    CHECK(r.conflicts == 1);
    CHECK(val(r, a) == 0.0);
    CHECK(val(r, b) == 1.0);
    CHECK(val(r, c) == 1.0);
    CHECK(rig.prop.depth() == depth);

    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Diveprop}, 0, rng, r);
    CHECK(r.complete);
    CHECK(r.conflicts == 1);
    CHECK(val(r, a) == 1.0);  // kept outside the propagator; repair's job now
    CHECK(rig.prop.depth() == depth);

    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfsrep}, 0, rng, r);
    CHECK(r.complete);
    CHECK(r.wants_repair);
    CHECK(val(r, a) == 0.0);
}

TEST_CASE("an infeasible pigeonhole ends incomplete: siblings run out, or the node cap", "[fpr]") {
    // p pigeons in h holes, x[i][j] binary: each pigeon in one hole, each hole
    // holding at most one. The root pass proves nothing.
    const auto pigeonhole = [](Model& m, int p, int h) {
        std::vector<std::vector<int32_t>> x(static_cast<size_t>(p));
        for (auto& row : x) {
            for (int j = 0; j < h; ++j) {
                row.push_back(m.bool_var());
            }
            m.add_constraint(m.eq_expr(m.sum(row), m.constant(1.0)));
        }
        for (int j = 0; j < h; ++j) {
            std::vector<int32_t> col;
            col.reserve(x.size());
            for (auto& row : x) {
                col.push_back(row[static_cast<size_t>(j)]);
            }
            m.add_constraint(m.leq(m.sum(col), m.constant(1.0)));
        }
        finish(m);
    };
    RNG rng(3);
    FprResult r;
    {
        Model m;
        pigeonhole(m, 3, 2);
        Rig rig(m);
        rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
        CHECK_FALSE(r.complete);
        CHECK(r.nodes == 2);  // x11 = 1 and x11 = 0, each refuted by the cascade
        CHECK(r.conflicts == 2);
    }
    {
        Model m;
        pigeonhole(m, 6, 5);
        Rig rig(m);
        const int32_t depth = rig.prop.depth();
        rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
        CHECK_FALSE(r.complete);
        CHECK(r.nodes == static_cast<int64_t>(m.num_vars()) + 1);
        CHECK(rig.prop.depth() == depth);
        for (const double v : r.values) {
            CHECK((v == 0.0 || v == 1.0));
        }
    }
    {
        // a + b >= 2 refutes a + b <= 1 at the root without crossing a bound.
        Model m;
        const int32_t a = m.bool_var();
        const int32_t b = m.bool_var();
        m.add_constraint(m.geq(m.sum({a, b}), m.constant(2.0)));
        m.add_constraint(m.leq(m.sum({a, b}), m.constant(1.0)));
        finish(m);
        LinearRowStore store(m);
        DomainPropagator prop(store, m);
        FprSearch fpr(m, store, prop);
        REQUIRE_FALSE(fpr.propagate_root());
        fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
        CHECK_FALSE(r.complete);
        CHECK(r.nodes == 0);
    }
    {
        // An empty model domain on a variable in no row refutes the root too.
        Model m;
        m.int_var(3, 1);
        finish(m);
        LinearRowStore store(m);
        DomainPropagator prop(store, m);
        FprSearch fpr(m, store, prop);
        REQUIRE_FALSE(fpr.propagate_root());
        fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
        CHECK_FALSE(r.complete);
        CHECK(r.nodes == 0);
    }
}

TEST_CASE("static orders: LR, type buckets, locks within buckets, random within buckets", "[fpr]") {
    Model m;
    const int32_t b0 = m.bool_var();
    const int32_t i1 = m.int_var(0, 5);
    const int32_t f2 = m.float_var(0.0, 1.0);
    const int32_t b3 = m.bool_var();
    const int32_t i4 = m.int_var(0, 5);
    // b3 and i4 in two rows each, b0 and i1 in one.
    m.add_constraint(m.leq(m.sum({b0, i1, b3, i4, f2}), m.constant(9.0)));
    m.add_constraint(m.leq(m.sum({b3, i4}), m.constant(4.0)));
    finish(m);
    Rig rig(m);
    RNG rng(5);
    std::vector<int32_t> order;
    rig.fpr.static_order(FprVarOrder::LR, rng, order);
    CHECK(order == std::vector<int32_t>{vid(b0), vid(i1), vid(f2), vid(b3), vid(i4)});
    rig.fpr.static_order(FprVarOrder::Type, rng, order);
    CHECK(order == std::vector<int32_t>{vid(b0), vid(b3), vid(i1), vid(i4), vid(f2)});
    CHECK(rig.fpr.up_locks(vid(b3)) == 2);
    CHECK(rig.fpr.down_locks(vid(b3)) == 0);
    rig.fpr.static_order(FprVarOrder::Locks, rng, order);
    CHECK(order == std::vector<int32_t>{vid(b3), vid(b0), vid(i4), vid(i1), vid(f2)});
    rig.fpr.static_order(FprVarOrder::Random, rng, order);
    REQUIRE(order.size() == 5);
    const std::vector<int32_t> bins{order[0], order[1]};
    const std::vector<int32_t> ints{order[2], order[3]};
    CHECK((bins == std::vector<int32_t>{vid(b0), vid(b3)} ||
           bins == std::vector<int32_t>{vid(b3), vid(b0)}));
    CHECK((ints == std::vector<int32_t>{vid(i1), vid(i4)} ||
           ints == std::vector<int32_t>{vid(i4), vid(i1)}));
    CHECK(order[4] == vid(f2));
}

TEST_CASE("domain size branches on the smallest domain first", "[fpr]") {
    // x + 4y <= 12: LR fixes x = 9 first (so y = 0), DomainSize y = 2 first (so x = 4).
    Model m;
    const int32_t x = m.int_var(0, 9);
    const int32_t y = m.int_var(0, 2);
    m.add_constraint(m.leq(m.sum({x, m.prod(m.constant(4.0), y)}), m.constant(12.0)));
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(1.0)));  // down locks: no trivial rounding
    finish(m);
    Rig rig(m);
    RNG rng(1);
    FprResult r;
    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
    CHECK(r.complete);
    CHECK(val(r, x) == 9.0);
    CHECK(val(r, y) == 0.0);
    rig.fpr.run({FprVarOrder::DomainSize, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
    CHECK(r.complete);
    CHECK(val(r, x) == 4.0);
    CHECK(val(r, y) == 2.0);
}

TEST_CASE("value rules pick their documented values", "[fpr]") {
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    const int32_t z = m.int_var(0, 100);
    const int32_t w = m.int_var(-3, 7);
    m.add_constraint(m.leq(m.sum({x, y}), m.constant(1.0)));  // x, y: up locks
    m.add_constraint(m.geq(m.sum({z, w}), m.constant(-10.0)));
    m.minimize(m.sum({x, m.prod(m.constant(-2.0), z)}));  // cost x 1, z -2, w 0
    m.close();
    m.add_objective_soft_constraint();
    m.var_mut(vid(w)).value = 5.0;
    full_evaluate(m);
    Rig rig(m);
    RNG rng(9);
    FprSearch& f = rig.fpr;
    CHECK(f.cost(vid(x)) == 1.0);
    CHECK(f.cost(vid(z)) == -2.0);
    CHECK(f.cost(vid(w)) == 0.0);

    CHECK(f.choose_value(vid(x), 0, 1, FprValueRule::Up, rng) == 1.0);
    CHECK(f.choose_value(vid(z), 0, 1e30, FprValueRule::Up, rng) == 1e5);  // the paper's box
    CHECK(f.choose_value(vid(x), 0, 1, FprValueRule::GoodObj, rng) == 0.0);
    CHECK(f.choose_value(vid(x), 0, 1, FprValueRule::BadObj, rng) == 1.0);
    CHECK(f.choose_value(vid(z), 2, 6, FprValueRule::GoodObj, rng) == 6.0);
    CHECK(f.choose_value(vid(z), 2, 6, FprValueRule::BadObj, rng) == 2.0);
    CHECK(f.choose_value(vid(w), -3, 7, FprValueRule::GoodObj, rng) == -3.0);  // zero cost: lb
    CHECK(f.choose_value(vid(w), -3, 7, FprValueRule::BadObj, rng) == -3.0);
    CHECK(f.choose_value(vid(w), -3, 7, FprValueRule::Incumbent, rng) == 5.0);
    CHECK(f.choose_value(vid(w), -3, 2, FprValueRule::Incumbent, rng) == 2.0);
    for (int k = 0; k < 20; ++k) {
        const double v = f.choose_value(vid(w), -3, 7, FprValueRule::Random, rng);
        CHECK(v >= -3.0);
        CHECK(v <= 7.0);
        CHECK(v == std::round(v));
    }
    // x + y <= 1 is active at the root: x is locked upward only, so down.
    CHECK(f.choose_value(vid(x), 0, 1, FprValueRule::LooseDyn, rng) == 0.0);
    // z + w >= -10 is redundant at the root: no locks, so GoodObj (z's cost < 0).
    CHECK(f.choose_value(vid(z), 0, 4, FprValueRule::LooseDyn, rng) == 4.0);
}

TEST_CASE("trivially roundable variables go to their unlocked bound", "[fpr]") {
    Model m;
    const int32_t x = m.int_var(0, 5);
    const int32_t y = m.int_var(0, 5);
    const int32_t q = m.int_var(0, 5);
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(2.0)));  // x, y: down locks only
    m.add_constraint(m.leq(m.prod(q, q), m.constant(4.0)));   // q: a nonlinear row
    m.add_constraint(m.geq(q, m.constant(1.0)));
    finish(m);
    Rig rig(m);
    RNG rng(1);
    FprResult r;
    rig.fpr.run({FprVarOrder::LR, FprValueRule::GoodObj, FprMode::Dfs}, 0, rng, r);
    CHECK(r.complete);
    CHECK(val(r, x) == 5.0);
    CHECK(val(r, y) == 5.0);
    CHECK(val(r, q) == 1.0);  // not rounded: q's nonlinear row is unseen
}

TEST_CASE("a Float takes its current value clamped to its propagated box", "[fpr]") {
    Model m;
    const int32_t b = m.bool_var();
    const int32_t f = m.float_var(0.0, 10.0);
    const int32_t g = m.float_var(0.0, 10.0);
    m.add_constraint(m.leq(m.sum({m.prod(m.constant(6.0), b), f}), m.constant(10.0)));
    m.add_constraint(m.geq(m.sum({b, g}), m.constant(0.5)));  // b's down lock
    m.close();
    m.var_mut(vid(f)).value = 7.0;
    m.var_mut(vid(g)).value = 3.0;
    full_evaluate(m);
    Rig rig(m);
    RNG rng(1);
    FprResult r;
    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
    CHECK(r.complete);
    CHECK(val(r, b) == 1.0);
    CHECK(std::abs(val(r, f) - 4.0) < 1e-6);  // a derived Float bound keeps its safety margin
    CHECK(val(r, g) == 3.0);
}

TEST_CASE("a Float with a cost takes its cost-preferred bound", "[fpr]") {
    Model m;
    const int32_t b = m.bool_var();
    const int32_t up = m.float_var(0.0, 10.0);
    const int32_t down = m.float_var(0.0, 10.0);
    const int32_t free_side = m.float_var(2.0, std::numeric_limits<double>::infinity());
    m.add_constraint(m.geq(m.sum({b, up, down}), m.constant(0.5)));
    m.minimize(m.sum({m.prod(m.constant(3.0), up), m.prod(m.constant(-2.0), down),
                      m.prod(m.constant(-1.0), free_side)}));
    m.close();
    m.var_mut(vid(up)).value = 7.0;
    m.var_mut(vid(down)).value = 7.0;
    m.var_mut(vid(free_side)).value = 7.0;
    full_evaluate(m);
    Rig rig(m);
    RNG rng(1);
    FprResult r;
    rig.fpr.run({FprVarOrder::LR, FprValueRule::Up, FprMode::Dfs}, 0, rng, r);
    CHECK(val(r, up) == 0.0);               // positive cost: the lower bound
    CHECK(val(r, down) == 10.0);            // negative cost: the upper bound
    CHECK(val(r, free_side) == 2.0 + 1e5);  // an infinite bound is boxed
}

TEST_CASE("an attempt is deterministic under its seed and leaves the propagator as found",
          "[fpr]") {
    Model m;
    std::vector<int32_t> x;
    x.reserve(30);
    for (int i = 0; i < 30; ++i) {
        x.push_back(i % 3 == 0 ? m.int_var(0, 4) : m.bool_var());
    }
    RNG build(17);
    for (int c = 0; c < 20; ++c) {
        std::vector<int32_t> terms;
        terms.reserve(4);
        for (int k = 0; k < 4; ++k) {
            terms.push_back(x[static_cast<size_t>(build.integers(0, 30))]);
        }
        const int32_t rhs = m.constant(static_cast<double>(build.integers(1, 4)));
        m.add_constraint(c % 2 == 0 ? m.leq(m.sum(terms), rhs) : m.geq(m.sum(terms), rhs));
    }
    finish(m);
    Rig rig(m);
    std::vector<double> lo;
    std::vector<double> hi;
    for (int32_t v = 0; v < rig.store.num_vars(); ++v) {
        lo.push_back(rig.prop.lo(v));
        hi.push_back(rig.prop.hi(v));
    }
    std::vector<FprConfig> configs(kFprPortfolio.begin(), kFprPortfolio.end());
    // No portfolio entry uses the Random order; its first run must not differ
    // from its second either.
    configs.push_back({FprVarOrder::Random, FprValueRule::Random, FprMode::Dfs});
    for (const FprConfig& cfg : configs) {
        RNG r1(42);
        RNG r2(42);
        FprResult a;
        FprResult b;
        rig.fpr.run(cfg, 1, r1, a);
        rig.fpr.run(cfg, 1, r2, b);
        CHECK(a.values == b.values);
        CHECK(a.nodes == b.nodes);
        CHECK(a.conflicts == b.conflicts);
        CHECK(rig.prop.depth() == 0);
        for (int32_t v = 0; v < rig.store.num_vars(); ++v) {
            CHECK(rig.prop.lo(v) == lo[static_cast<size_t>(v)]);
            CHECK(rig.prop.hi(v) == hi[static_cast<size_t>(v)]);
        }
    }
}

TEST_CASE("the portfolio rotates on worker plus attempt", "[fpr]") {
    const auto same = [](const FprConfig& a, const FprConfig& b) {
        return a.order == b.order && a.value == b.value && a.mode == b.mode;
    };
    CHECK(same(fpr_portfolio_config(0, 0), kFprPortfolio[0]));
    CHECK(same(fpr_portfolio_config(2, 3), kFprPortfolio[5]));
    CHECK(same(fpr_portfolio_config(1, 6), kFprPortfolio[1]));
    CHECK(same(fpr_portfolio_config(0, -1), kFprPortfolio[5]));
}

namespace {

// x_0 >= 1 and x_{i+1} == x_i over Bools: FJ starts every x at 0 and repairs
// the chain one flip at a time; the root propagation fixes all of it.
Model fpr_chain(int n) {
    Model m;
    std::vector<int32_t> x;
    x.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        x.push_back(m.bool_var());
    }
    m.add_constraint(m.geq(x[0], m.constant(1.0)));
    for (int i = 0; i + 1 < n; ++i) {
        m.add_constraint(
            m.eq_expr(m.sum({x[static_cast<size_t>(i) + 1], m.neg(x[static_cast<size_t>(i)])}),
                      m.constant(0.0)));
    }
    return m;
}

SearchResult fpr_solve(Model& m, const SearchConfig& config) {
    return solve(m, /*time_limit=*/0.0, /*seed=*/7, nullptr, nullptr, 3, nullptr, config);
}

}  // namespace

TEST_CASE("an FPR batch solves a chain FJ alone cannot in the budget", "[fpr]") {
    SearchConfig config;
    config.max_iterations = 40;
    config.batch_iterations = 20;
    config.use_compound_moves = false;
    {
        Model m = fpr_chain(200);
        const SearchResult r = fpr_solve(m, config);
        CHECK_FALSE(r.feasible);
        CHECK(r.counters.fpr_batches == 0);
    }
    config.fpr_batch_probability = 1.0;
    Model m = fpr_chain(200);
    const SearchResult r = fpr_solve(m, config);
    CHECK(r.feasible);
    CHECK(r.counters.fpr_batches == r.counters.batches);
    CHECK(r.counters.fpr_batches >= 1);
}

TEST_CASE("a refuted root builds no FPR batch: every batch is FJ", "[fpr]") {
    Model m;
    const int32_t a = m.bool_var();
    const int32_t b = m.bool_var();
    m.add_constraint(m.geq(m.sum({a, b}), m.constant(2.0)));
    m.add_constraint(m.leq(m.sum({a, b}), m.constant(1.0)));
    SearchConfig config;
    config.max_iterations = 200;
    config.batch_iterations = 50;
    config.fpr_batch_probability = 1.0;
    const SearchResult r = fpr_solve(m, config);
    CHECK_FALSE(r.feasible);
    CHECK(r.counters.fpr_batches == 0);
    CHECK(r.counters.batches > 0);
    CHECK(r.counters.fj_batches + r.counters.novelty_batches == r.counters.batches);
}

TEST_CASE("FPR off draws nothing: fpr_worker cannot move the trajectory", "[fpr]") {
    // Probability 0 builds no FprBatch, so a worker index that keyed anything
    // would show here. That 0 also skips the draw (the pre-#426 trajectory)
    // is pinned by test_structured_trajectory.cpp's recorded signatures of
    // default solve() runs (libstdc++ hosts).
    SearchConfig base;
    base.max_iterations = 3000;
    base.batch_iterations = 100;
    SearchConfig moved = base;
    moved.fpr_worker = 5;
    Model m1 = fpr_chain(60);
    Model m2 = fpr_chain(60);
    const SearchResult r1 = fpr_solve(m1, base);
    const SearchResult r2 = fpr_solve(m2, moved);
    CHECK(r1.iterations == r2.iterations);
    CHECK(r1.counters.batches == r2.counters.batches);
    CHECK(r1.counters.novelty_batches == r2.counters.novelty_batches);
    CHECK(r1.best_state.values == r2.best_state.values);
    CHECK(r1.counters.fpr_batches == 0);
}

TEST_CASE("fpr_batch_probability outside [0, 1] is refused", "[fpr]") {
    for (const double p : {-0.1, 1.5, std::numeric_limits<double>::quiet_NaN()}) {
        SearchConfig config;
        config.fpr_batch_probability = p;
        Model m = fpr_chain(3);
        CHECK_THROWS_AS(fpr_solve(m, config), std::invalid_argument);
    }
}
