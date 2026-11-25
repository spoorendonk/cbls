// HC4 interval propagation over nonlinear monotone rows (#424):
// DagRowStore's row selection, exact narrowing on hand cases, soundness by
// sampling, conflicts, backtracking, and the FPR batch flag.

#include "test_helpers.h"

#include <catch2/catch_get_random_seed.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/dag_propagation.h>
#include <cbls/domain_propagation.h>
#include <cbls/violation.h>
#include <cmath>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

// A model, closed and evaluated, with its stores and propagator.
struct Rig {
    explicit Rig(Model& model)
        : m(model), linear(model), dag(model), prop(linear, model, {}, &dag) {}
    Model& m;
    LinearRowStore linear;
    DagRowStore dag;
    DomainPropagator prop;
};

void close(Model& m) {
    m.close();
    full_evaluate(m);
}

// Whether constraint c holds at `x` (one value per variable), as the engine
// reads it: violation 0 at tolerance 0.
bool holds(Model& m, int32_t c, const std::vector<double>& x) {
    for (size_t v = 0; v < x.size(); ++v) {
        m.var_mut(static_cast<int32_t>(v)).value = x[v];
    }
    full_evaluate(m);
    const int32_t nid = m.constraint_ids()[static_cast<size_t>(c)];
    return clamped_node_violation(m.node_value(nid)) == 0.0;
}

}  // namespace

TEST_CASE("DagRowStore keeps monotone nonlinear rows and skips the rest", "[dag_propagation]") {
    Model m;
    const int32_t x = m.float_var(0.5, 4.0);
    const int32_t y = m.float_var(1.0, 2.0);
    m.add_constraint(m.leq(m.exp_expr(x), m.constant(10.0)));                       // 0: kept
    m.add_constraint(m.leq(m.sum({x, y}), m.constant(5.0)));                        // 1: affine
    m.add_constraint(m.leq(m.sin_expr(x), m.constant(0.5)));                        // 2: Sin
    m.add_constraint(m.geq(m.prod(m.log_expr(x), y), m.constant(0.1)));             // 3: kept
    m.add_constraint(m.neq(m.exp_expr(x), y));                                      // 4: Neq
    m.add_constraint(m.lt(m.max_expr({m.neg(x), m.prod(x, y)}), m.constant(9.0)));  // 5: kept
    m.add_constraint(m.eq_expr(m.min_expr({x, y}), m.constant(1.5)));               // 6: kept
    m.minimize(m.exp_expr(y));
    m.close();
    m.add_objective_soft_constraint();  // not stored: soft
    full_evaluate(m);
    const DagRowStore dag(m);
    REQUIRE(dag.num_rows() == 4);
    CHECK(dag.row(0).constraint == 0);
    CHECK(dag.row(1).constraint == 3);
    CHECK(dag.row(2).constraint == 5);
    CHECK(dag.row(3).constraint == 6);
    // Lt carries its strictness margin; Eq is two-sided.
    CHECK(dag.row(2).hi < 0.0);
    CHECK(dag.row(3).lo == 0.0);
    CHECK(dag.row(3).hi == 0.0);
    // x's rows: 0, 1, 2, 3; y's: 1, 2, 3.
    CHECK(dag.var_row_start(vid(x) + 1) - dag.var_row_start(vid(x)) == 4);
    CHECK(dag.var_row_start(vid(y) + 1) - dag.var_row_start(vid(y)) == 3);
}

TEST_CASE("the DAG rows narrow each op's operands exactly on hand cases", "[dag_propagation]") {
    const double tol = 1e-6;
    SECTION("exp(x) <= e bounds x by 1, integral x rounds inward") {
        Model m;
        const int32_t x = m.float_var(-5.0, 5.0);
        const int32_t k = m.int_var(-5, 5);
        m.add_constraint(m.leq(m.exp_expr(x), m.constant(std::exp(1.0))));
        m.add_constraint(m.leq(m.exp_expr(k), m.constant(10.0)));
        close(m);
        Rig r(m);
        REQUIRE(r.prop.propagate_all());
        CHECK(std::abs(r.prop.hi(vid(x)) - 1.0) < tol);
        CHECK(r.prop.hi(vid(x)) >= 1.0);
        CHECK(r.prop.hi(vid(k)) == 2.0);
        CHECK(r.prop.lo(vid(x)) == -5.0);
        CHECK(r.prop.stats().dag_bounds_derived == 2);
    }
    SECTION("log(x) >= 0 bounds x below by 1") {
        Model m;
        const int32_t x = m.float_var(0.0, 5.0);
        m.add_constraint(m.geq(m.log_expr(x), m.constant(0.0)));
        close(m);
        Rig r(m);
        REQUIRE(r.prop.propagate_all());
        CHECK(std::abs(r.prop.lo(vid(x)) - 1.0) < tol);
        CHECK(r.prop.lo(vid(x)) <= 1.0);
    }
    SECTION("a product bounds a factor only through a sign-known other factor") {
        Model m;
        const int32_t x = m.float_var(0.0, 10.0);
        const int32_t y = m.float_var(1.0, 2.0);
        const int32_t z = m.float_var(-1.0, 1.0);
        const int32_t u = m.float_var(0.0, 10.0);
        const int32_t w = m.float_var(-1.0, 1.0);
        m.add_constraint(m.geq(m.prod(x, y), m.constant(4.0)));   // x >= 4 / 2
        m.add_constraint(m.leq(m.prod(x, z), m.constant(-4.0)));  // then z <= -4 / x
        m.add_constraint(m.geq(m.prod(u, w), m.constant(0.5)));   // both span 0: nothing
        close(m);
        Rig r(m);
        REQUIRE(r.prop.propagate_all());
        // x in [2, 10] gives z <= -0.4, which gives x >= 4 / 1.
        CHECK(std::abs(r.prop.lo(vid(x)) - 4.0) < tol);
        CHECK(std::abs(r.prop.hi(vid(z)) + 0.4) < tol);
        CHECK(r.prop.lo(vid(y)) == 1.0);
        CHECK(r.prop.lo(vid(u)) == 0.0);
        CHECK(r.prop.lo(vid(w)) == -1.0);
    }
    SECTION("min and max bound every operand, and the one that can still bind") {
        Model m;
        const int32_t a = m.float_var(0.0, 10.0);
        const int32_t b = m.float_var(0.0, 2.0);
        const int32_t c = m.float_var(0.0, 10.0);
        const int32_t d = m.float_var(0.0, 10.0);
        m.add_constraint(m.geq(m.min_expr({a, c}), m.constant(3.0)));
        m.add_constraint(m.geq(m.max_expr({b, d}), m.constant(5.0)));  // only d reaches 5
        m.add_constraint(m.leq(m.max_expr({a, d}), m.constant(8.0)));
        close(m);
        Rig r(m);
        REQUIRE(r.prop.propagate_all());
        CHECK(std::abs(r.prop.lo(vid(a)) - 3.0) < tol);
        CHECK(std::abs(r.prop.lo(vid(c)) - 3.0) < tol);
        CHECK(std::abs(r.prop.lo(vid(d)) - 5.0) < tol);
        CHECK(std::abs(r.prop.hi(vid(a)) - 8.0) < tol);
        CHECK(std::abs(r.prop.hi(vid(d)) - 8.0) < tol);
        CHECK(r.prop.lo(vid(b)) == 0.0);
    }
    SECTION("a nonlinear sum bounds its terms through Neg and Exp") {
        Model m;
        const int32_t x = m.float_var(-3.0, 3.0);
        const int32_t y = m.float_var(1.0, 5.0);
        m.add_constraint(m.leq(m.sum({m.exp_expr(x), m.neg(m.neg(y))}), m.constant(3.0)));
        close(m);
        Rig r(m);
        REQUIRE(r.prop.propagate_all());
        CHECK(std::abs(r.prop.hi(vid(x)) - std::log(2.0)) < tol);
        CHECK(std::abs(r.prop.hi(vid(y)) - (3.0 - std::exp(-3.0))) < tol);
    }
}

TEST_CASE("an empty implied DAG domain is a conflict, and backtrack restores bounds",
          "[dag_propagation]") {
    Model m;
    const int32_t x = m.float_var(0.0, 1.0);
    const int32_t y = m.float_var(0.0, 4.0);
    m.add_constraint(m.geq(m.sum({m.exp_expr(x), m.log_expr(y)}), m.constant(2.0)));
    close(m);
    Rig r(m);
    REQUIRE(r.prop.propagate_all());
    // exp(x) <= e, so log(y) >= 2 - e: y >= exp(2 - e).
    CHECK(std::abs(r.prop.lo(vid(y)) - std::exp(2.0 - std::exp(1.0))) < 1e-6);
    const double y_lo = r.prop.lo(vid(y));
    const int32_t level = r.prop.checkpoint();
    REQUIRE(r.prop.tighten(vid(y), y_lo, 1.0));  // log(y) <= 0: exp(x) >= 2
    CHECK(std::abs(r.prop.lo(vid(x)) - std::log(2.0)) < 1e-6);
    r.prop.backtrack(level);
    CHECK(r.prop.lo(vid(x)) == 0.0);
    CHECK(r.prop.hi(vid(y)) == 4.0);
    CHECK(r.prop.lo(vid(y)) == y_lo);

    r.prop.checkpoint();
    REQUIRE(r.prop.fix(vid(y), 1.0));
    CHECK_FALSE(r.prop.fix(vid(x), 0.2));  // exp(0.2) + 0 < 2
    CHECK(r.prop.stats().conflicts >= 1);
}

namespace {

using RowBuilder = std::function<int32_t(Model&, int32_t, int32_t, int32_t)>;

// One trial: x, y (Float or Int) and z in fixed boxes, one variable's upper
// bound cut at random, then the root pass; every sampled point of the cut
// box that satisfies the row must lie inside the narrowed box.
void check_sound(const RowBuilder& build, bool int_y, int pick_index, std::mt19937_64& gen) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    Model m;
    const int32_t x = m.float_var(0.1, 3.0);
    const int32_t y = int_y ? m.int_var(-2, 3) : m.float_var(-2.0, 2.0);
    const int32_t z = m.float_var(-1.5, 2.5);
    m.add_constraint(build(m, x, y, z));
    close(m);
    Rig r(m);
    REQUIRE(r.dag.num_rows() == 1);
    std::vector<std::pair<double, double>> box;
    box.reserve(3);
    for (int32_t v = 0; v < 3; ++v) {
        box.emplace_back(m.var(v).lb, m.var(v).ub);
    }
    auto& [plo, phi] = box[static_cast<size_t>(pick_index)];
    const double cut = plo + (unit(gen) * (phi - plo));
    phi = m.var(pick_index).type == VarType::Int ? std::floor(cut) : cut;
    const bool ok = r.prop.tighten(pick_index, plo, phi) && r.prop.propagate_all();
    std::vector<double> pt(3);
    for (int s = 0; s < 400; ++s) {
        for (int32_t v = 0; v < 3; ++v) {
            const auto [lo, hi] = box[static_cast<size_t>(v)];
            const double val = lo + (unit(gen) * (hi - lo));
            pt[static_cast<size_t>(v)] = m.var(v).type == VarType::Int ? std::round(val) : val;
        }
        if (!holds(m, 0, pt)) {
            continue;
        }
        REQUIRE(ok);  // a refuted box has no satisfying point
        for (int32_t v = 0; v < 3; ++v) {
            CHECK(pt[static_cast<size_t>(v)] >= r.prop.lo(v));
            CHECK(pt[static_cast<size_t>(v)] <= r.prop.hi(v));
        }
    }
}

}  // namespace

TEST_CASE("DAG propagation removes no point that satisfies the row", "[dag_propagation]") {
    // A shared subexpression appears in the last two rows.
    const std::vector<RowBuilder> rows = {
        [](Model& m, int32_t x, int32_t y, int32_t) {
            return m.leq(m.prod(m.exp_expr(x), y), m.constant(3.0));
        },
        [](Model& m, int32_t x, int32_t y, int32_t z) {
            return m.geq(m.sum({m.log_expr(x), m.prod(y, z), m.neg(z)}), m.constant(0.5));
        },
        [](Model& m, int32_t x, int32_t y, int32_t z) {
            return m.leq(m.max_expr({m.prod(x, y), z}), m.constant(1.0));
        },
        [](Model& m, int32_t x, int32_t y, int32_t z) {
            return m.lt(m.min_expr({m.exp_expr(y), m.sum({x, m.prod(x, z)})}), m.constant(0.7));
        },
        [](Model& m, int32_t x, int32_t y, int32_t z) {
            const int32_t s = m.prod(x, y);
            return m.leq(m.sum({s, m.exp_expr(s), m.prod(s, z)}), m.constant(2.0));
        },
        [](Model& m, int32_t x, int32_t y, int32_t z) {
            const int32_t s = m.sum({x, z});
            return m.geq(m.prod(m.log_expr(s), m.exp_expr(s)), m.prod(y, s));
        },
    };
    std::mt19937_64 gen(Catch::getSeed());  // printed per run; --rng-seed replays it
    for (size_t k = 0; k < rows.size(); ++k) {
        for (int trial = 0; trial < 24; ++trial) {
            INFO("row " << k << " trial " << trial);
            check_sound(rows[k], trial % 2 == 1, trial % 3, gen);
        }
    }
}

TEST_CASE("an empty DAG store changes no linear propagation", "[dag_propagation]") {
    Model m;
    const int32_t x = m.int_var(0, 10);
    const int32_t y = m.int_var(0, 10);
    const int32_t z = m.float_var(0.0, 10.0);
    m.add_constraint(m.leq(m.sum({x, y, z}), m.constant(12.0)));
    m.add_constraint(m.geq(m.sum({x, m.neg(y)}), m.constant(1.0)));
    close(m);
    const LinearRowStore linear(m);
    const DagRowStore dag(m);
    REQUIRE(dag.num_rows() == 0);
    DomainPropagator plain(linear, m);
    DomainPropagator with(linear, m, {}, &dag);
    for (DomainPropagator* p : {&plain, &with}) {
        REQUIRE(p->propagate_all());
        p->checkpoint();
        REQUIRE(p->fix(vid(x), 9.0));
        REQUIRE(p->tighten(vid(z), 2.5, 10.0));
    }
    for (int32_t v = 0; v < 3; ++v) {
        CHECK(plain.lo(v) == with.lo(v));
        CHECK(plain.hi(v) == with.hi(v));
    }
    CHECK(plain.stats().bounds_derived == with.stats().bounds_derived);
    CHECK(with.stats().dag_rows_scanned == 0);
}

TEST_CASE("the FPR batch with DAG propagation solves a nonlinear model", "[dag_propagation]") {
    for (const bool dag : {false, true}) {
        Model m;
        std::vector<int32_t> xs;
        xs.reserve(6);
        for (int i = 0; i < 6; ++i) {
            xs.push_back(m.int_var(0, 8));
        }
        for (int i = 0; i + 1 < 6; ++i) {
            m.add_constraint(
                m.geq(m.prod(m.exp_expr(xs[static_cast<size_t>(i)]),
                             m.log_expr(m.sum({xs[static_cast<size_t>(i) + 1], m.constant(1.0)}))),
                      m.constant(5.0)));
        }
        m.close();
        SearchConfig config;
        config.max_iterations = 400;
        config.batch_iterations = 50;
        config.fpr_batch_probability = 1.0;
        config.fpr_dag_propagation = dag;
        const SearchResult r = solve(m, 0.0, 7, nullptr, nullptr, 3, nullptr, config);
        INFO("dag " << dag);
        CHECK(r.feasible);
        CHECK(r.counters.fpr_batches >= 1);
    }
}
