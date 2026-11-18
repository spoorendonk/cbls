// LinearRowStore and DomainPropagator (#329, #420): propagation over linear
// rows, with an exact undo.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/domain_propagation.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

// Everything a backtrack must restore, read through the public accessors.
struct Snapshot {
    std::vector<double> lo;
    std::vector<double> hi;
    std::vector<double> min_sum;
    std::vector<int> min_inf;
    std::vector<double> max_sum;
    std::vector<int> max_inf;

    bool operator==(const Snapshot& o) const {
        // Bit for bit: == on doubles, and no NaN can occur in these.
        return lo == o.lo && hi == o.hi && min_sum == o.min_sum && min_inf == o.min_inf &&
               max_sum == o.max_sum && max_inf == o.max_inf;
    }
};

Snapshot snapshot(const DomainPropagator& p, const LinearRowStore& s) {
    Snapshot out;
    for (int32_t v = 0; v < s.num_vars(); ++v) {
        out.lo.push_back(p.lo(v));
        out.hi.push_back(p.hi(v));
    }
    for (int32_t r = 0; r < s.num_rows(); ++r) {
        out.min_sum.push_back(p.min_activity_sum(r));
        out.min_inf.push_back(p.min_activity_infinite(r));
        out.max_sum.push_back(p.max_activity_sum(r));
        out.max_inf.push_back(p.max_activity_infinite(r));
    }
    return out;
}

// A random model of sparse Leq/Geq/Eq rows over Int and Float variables.
void build_random(Model& m, RNG& rng) {
    std::vector<int32_t> xs;
    xs.reserve(12);
    for (int i = 0; i < 12; ++i) {
        xs.push_back(i % 2 == 0 ? m.int_var(0, 6) : m.float_var(-3.0, 4.0));
    }
    for (int c = 0; c < 10; ++c) {
        std::vector<int32_t> terms;
        const int64_t n = rng.integers(2, 6);
        for (int64_t k = 0; k < n; ++k) {
            const int32_t x = xs[static_cast<size_t>(rng.integers(0, 12))];
            terms.push_back(m.prod(m.constant(static_cast<double>(rng.integers(-4, 5)) + 0.5), x));
        }
        const int32_t lhs = m.sum(terms);
        const int32_t rhs = m.constant(rng.uniform(-6.0, 10.0));
        const int64_t kind = rng.integers(0, 3);
        if (kind == 0) {
            m.add_constraint(m.leq(lhs, rhs));
        } else if (kind == 1) {
            m.add_constraint(m.geq(lhs, rhs));
        } else {
            m.add_constraint(m.eq_expr(lhs, rhs));
        }
    }
    m.close();
    full_evaluate(m);
}

// A value inside variable `v`'s current domain, integral on an Int.
double random_value(const DomainPropagator& p, const Model& m, RNG& rng, int32_t v) {
    const double a = rng.uniform(p.lo(v), p.hi(v));
    return m.var(v).type == VarType::Int ? std::round(a) : a;
}

// One random operation: checkpoint, fix, tighten, fix_all or backtrack.
// Returns the propagating call's result, true otherwise. `levels` mirrors the
// checkpoint stack.
bool random_op(DomainPropagator& p, const Model& m, RNG& rng, std::vector<int32_t>& levels) {
    const int64_t op = rng.integers(0, 5);
    if (op == 4) {
        std::vector<std::pair<int32_t, double>> fixings;
        const int64_t n = rng.integers(1, 4);
        for (int64_t k = 0; k < n; ++k) {
            const auto v =
                static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(m.num_vars())));
            fixings.emplace_back(v, random_value(p, m, rng, v));
        }
        return p.fix_all(fixings);
    }
    if (op == 0) {
        levels.push_back(p.checkpoint());
        return true;
    }
    if (op == 3) {
        if (levels.empty()) {
            return true;
        }
        const auto i = static_cast<size_t>(rng.integers(0, static_cast<int64_t>(levels.size())));
        p.backtrack(levels[i]);
        levels.resize(i);
        return true;
    }
    const auto v = static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(m.num_vars())));
    const double lo = p.lo(v);
    const double hi = p.hi(v);
    const double a = random_value(p, m, rng, v);
    if (op == 1) {
        return p.fix(v, a);
    }
    return a - lo < hi - a ? p.tighten(v, a, hi) : p.tighten(v, lo, a);
}

}  // namespace

TEST_CASE("fixing one binary of x + y + z == 1 fixes the others", "[domain_propagation]") {
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    const int32_t z = m.bool_var();
    m.add_constraint(m.eq_expr(m.sum({x, y, z}), m.constant(1.0)));
    m.add_constraint(m.neq(x, y));  // not stored
    m.minimize(m.sum({x, y}));
    m.close();
    m.add_objective_soft_constraint();
    m.set_objective_bound(0.5);  // not stored: soft
    full_evaluate(m);
    const LinearRowStore store(m);
    REQUIRE(store.num_rows() == 1);
    CHECK(store.constraint(0) == 0);
    DomainPropagator p(store, m);

    const int32_t base = p.checkpoint();
    REQUIRE(p.fix(vid(x), 1.0));
    CHECK(p.hi(vid(y)) == 0.0);
    CHECK(p.hi(vid(z)) == 0.0);
    std::vector<int32_t> changed;
    p.changed_since(base, changed);
    CHECK(changed == std::vector<int32_t>{vid(x), vid(y), vid(z)});

    p.backtrack(base);
    CHECK(p.hi(vid(y)) == 1.0);
    p.checkpoint();
    REQUIRE(p.fix(vid(x), 0.0));
    CHECK(p.lo(vid(z)) == 0.0);  // y or z: nothing follows yet
    REQUIRE(p.fix(vid(y), 0.0));
    CHECK(p.lo(vid(z)) == 1.0);
}

TEST_CASE("a row with one infinite term bounds that term", "[domain_propagation]") {
    Model m;
    const int32_t x = m.float_var(0.0, 1.0);
    const int32_t y = m.float_var(0.0, 1e30);  // a sentinel infinity
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(3.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    REQUIRE(store.num_rows() == 1);
    DomainPropagator p(store, m);
    // Residual 3 - x - y <= 0: the min activity -x - y has y's -inf.
    CHECK(p.min_activity_infinite(0) == 1);
    CHECK(p.max_activity_infinite(0) == 0);
    REQUIRE(p.tighten(vid(x), 0.0, 0.5));
    CHECK(p.lo(vid(y)) >= 2.5 - 1e-6);
    CHECK(p.lo(vid(y)) <= 2.5);
    REQUIRE(p.tighten(vid(y), 0.0, 10.0));
    CHECK(p.min_activity_infinite(0) == 0);

    // A sentinel-sized row bound is no bound, as `propagate_bounds` reads it.
    Model s;
    const int32_t z = s.float_var(0.0, 1e30);
    s.add_constraint(s.leq(s.prod(s.constant(1e10), z), s.constant(1e25)));
    s.close();
    full_evaluate(s);
    const LinearRowStore sstore(s);
    REQUIRE(sstore.num_rows() == 1);
    DomainPropagator sp(sstore, s);
    REQUIRE(sp.propagate_all());
    CHECK(sp.hi(vid(z)) == 1e30);
}

TEST_CASE("a derived bound rounds inward on an integral variable only", "[domain_propagation]") {
    Model m;
    const int32_t x = m.int_var(0, 5);
    const int32_t f = m.float_var(0.0, 5.0);
    const int32_t y = m.int_var(0, 5);
    m.add_constraint(m.leq(m.sum({m.prod(m.constant(2.0), x), y}), m.constant(5.0)));
    m.add_constraint(m.leq(m.sum({m.prod(m.constant(2.0), f), y}), m.constant(5.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    DomainPropagator p(store, m);
    REQUIRE(p.fix(vid(y), 0.0));
    CHECK(p.hi(vid(x)) == 2.0);
    CHECK(p.hi(vid(f)) > 2.5);
    CHECK(p.hi(vid(f)) < 2.5 + 1e-6);
}

TEST_CASE("an empty implied domain is a conflict", "[domain_propagation]") {
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(2.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    DomainPropagator p(store, m);
    const int32_t base = p.checkpoint();
    CHECK_FALSE(p.fix(vid(x), 0.0));
    CHECK(p.stats().conflicts == 1);
    p.backtrack(base);
    REQUIRE(p.fix(vid(x), 1.0));
    CHECK(p.lo(vid(y)) == 1.0);
    // Outside the current domain: a conflict without propagation.
    CHECK_FALSE(p.fix(vid(y), 0.0));
}

// Ops go on after a conflict without a backtrack (#427): the state stays
// consistent, and a later backtrack still restores it bit for bit.
TEST_CASE("random fix and backtrack restore bounds and activities bit for bit",
          "[domain_propagation]") {
    int64_t continued = 0;  // ops run from a conflicted, unbacktracked state
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        RNG rng(seed);
        Model m;
        build_random(m, rng);
        const LinearRowStore store(m);
        DomainPropagationOptions opts;
        opts.cascade = seed % 2 == 0;
        DomainPropagator p(store, m, opts);
        const Snapshot fresh = snapshot(p, store);
        const int32_t root = p.checkpoint();
        std::vector<int32_t> levels;
        std::vector<Snapshot> at_level;
        bool conflicted = false;
        for (int step = 0; step < 300; ++step) {
            const size_t before = levels.size();
            continued += conflicted ? 1 : 0;
            const bool ok = random_op(p, m, rng, levels);
            conflicted = (conflicted || !ok) && levels.size() >= before;
            for (int32_t v = 0; v < store.num_vars(); ++v) {
                REQUIRE(p.lo(v) <= p.hi(v));  // never an empty domain
            }
            if (levels.size() > before) {
                at_level.push_back(snapshot(p, store));
            } else if (levels.size() < before) {
                // Backtracked to the checkpoint taken as at_level[levels.size()].
                REQUIRE(snapshot(p, store) == at_level[levels.size()]);
                at_level.resize(levels.size());
            }
        }
        p.backtrack(root);
        // Back at the start, the state is the one built from scratch.
        const DomainPropagator scratch(store, m, opts);
        REQUIRE(snapshot(p, store) == fresh);
        REQUIRE(snapshot(p, store) == snapshot(scratch, store));
    }
    CHECK(continued > 0);
}

TEST_CASE("fix_all propagates its fixings together and keeps them on a conflict",
          "[domain_propagation]") {
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    const int32_t z = m.bool_var();
    const int32_t u = m.bool_var();
    const int32_t w = m.bool_var();
    m.add_constraint(m.eq_expr(m.sum({x, y, z}), m.constant(1.0)));
    m.add_constraint(m.leq(m.sum({u, w}), m.constant(1.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    DomainPropagator p(store, m);
    const int32_t base = p.checkpoint();
    const Snapshot fresh = snapshot(p, store);

    // z = 1 follows from x = 0 and y = 0 together only.
    REQUIRE(p.fix_all({{vid(x), 0.0}, {vid(y), 0.0}}));
    CHECK(p.lo(vid(z)) == 1.0);
    p.backtrack(base);
    REQUIRE(snapshot(p, store) == fresh);

    // A propagation conflict keeps both fixings, and the state goes on.
    p.checkpoint();
    CHECK_FALSE(p.fix_all({{vid(x), 1.0}, {vid(y), 1.0}}));
    CHECK(p.lo(vid(x)) == 1.0);
    CHECK(p.lo(vid(y)) == 1.0);
    REQUIRE(p.fix(vid(u), 1.0));
    CHECK(p.hi(vid(w)) == 0.0);
    p.backtrack(base);
    REQUIRE(snapshot(p, store) == fresh);

    // A crossing fixing is skipped; the others are still fixed and propagated.
    p.checkpoint();
    REQUIRE(p.fix(vid(x), 1.0));
    const int64_t conflicts = p.stats().conflicts;
    CHECK_FALSE(p.fix_all({{vid(y), 1.0}, {vid(u), 1.0}}));
    CHECK(p.stats().conflicts == conflicts + 1);
    CHECK(p.hi(vid(y)) == 0.0);
    CHECK(p.hi(vid(w)) == 0.0);

    // A bad request throws before any change.
    const Snapshot before = snapshot(p, store);
    CHECK_THROWS_AS(p.fix_all({{vid(w), 0.0}, {store.num_vars(), 0.0}}), std::out_of_range);
    CHECK(snapshot(p, store) == before);
    p.backtrack(base);
    REQUIRE(snapshot(p, store) == fresh);
}

TEST_CASE("the slack filter skips no row a full scan would tighten", "[domain_propagation]") {
    int64_t filtered = 0;
    for (uint64_t seed = 101; seed <= 120; ++seed) {
        RNG build_rng(seed);
        Model m;
        build_random(m, build_rng);
        const LinearRowStore store(m);
        DomainPropagationOptions on;
        on.cascade = seed % 2 == 0;
        DomainPropagationOptions off = on;
        off.filter = false;
        DomainPropagator a(store, m, on);
        DomainPropagator b(store, m, off);
        RNG ra(seed * 7);
        RNG rb(seed * 7);
        std::vector<int32_t> la;
        std::vector<int32_t> lb;
        for (int step = 0; step < 300; ++step) {
            const bool ok_a = random_op(a, m, ra, la);
            const bool ok_b = random_op(b, m, rb, lb);
            REQUIRE(ok_a == ok_b);
            REQUIRE(la == lb);
            for (int32_t v = 0; v < store.num_vars(); ++v) {
                REQUIRE(a.lo(v) == b.lo(v));
                REQUIRE(a.hi(v) == b.hi(v));
            }
            if (!ok_a && !la.empty()) {
                a.backtrack(la.back());
                b.backtrack(lb.back());
                la.pop_back();
                lb.pop_back();
            }
        }
        filtered += a.stats().rows_filtered;
    }
    CHECK(filtered > 0);  // the filter was exercised
}

TEST_CASE("a big-M bound change leaves no rounding drift in derived bounds",
          "[domain_propagation]") {
    // max activity 0.3 + 1e15 rounds to 1e15 + 0.25; removing the 1e15
    // incrementally would leave 0.25, derive x >= 0.35 and call x = 0.3,
    // y = 0 infeasible.
    Model m;
    const int32_t x = m.float_var(0.0, 0.3);
    const int32_t y = m.float_var(0.0, 1e15);
    m.add_constraint(m.eq_expr(m.sum({x, y}), m.constant(0.3)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    REQUIRE(store.num_rows() == 1);
    DomainPropagator p(store, m);
    REQUIRE(p.fix(vid(y), 0.0));
    CHECK(p.lo(vid(x)) <= 0.3);
    CHECK(p.lo(vid(x)) >= 0.3 - 1e-6);
}

TEST_CASE("cascade rescans a row whose variable a later row tightened", "[domain_propagation]") {
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    const int32_t w = m.bool_var();
    m.add_constraint(m.leq(m.sum({x, y, w}), m.constant(2.0)));
    m.add_constraint(m.leq(m.sum({x, m.neg(w)}), m.constant(0.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    REQUIRE(store.num_rows() == 2);
    DomainPropagationOptions opts;
    opts.cascade = true;
    DomainPropagator p(store, m, opts);
    REQUIRE(p.fix(vid(x), 1.0));
    CHECK(p.lo(vid(w)) == 1.0);
    CHECK(p.hi(vid(y)) == 0.0);  // x + y + w <= 2 with x = w = 1
}

namespace {

// x_{i+1} <= x_i - 1 over Int x_0..x_5 in [0, 10]: upper bounds follow row
// order in one pass; lower bounds run against it and need the cascade.
std::vector<int32_t> build_chain(Model& m) {
    std::vector<int32_t> xs;
    xs.reserve(6);
    for (int i = 0; i < 6; ++i) {
        xs.push_back(m.int_var(0, 10));
    }
    for (size_t i = 0; i + 1 < xs.size(); ++i) {
        m.add_constraint(m.leq(m.sum({xs[i + 1], m.neg(xs[i])}), m.constant(-1.0)));
    }
    m.close();
    full_evaluate(m);
    return xs;
}

}  // namespace

TEST_CASE("propagate_all reaches the fixpoint and backtracks bit for bit", "[domain_propagation]") {
    Model m;
    const std::vector<int32_t> xs = build_chain(m);
    const LinearRowStore store(m);
    REQUIRE(store.num_rows() == 5);
    REQUIRE(store.nnz() == 10);
    DomainPropagator p(store, m);
    const Snapshot fresh = snapshot(p, store);
    const int32_t root = p.checkpoint();
    REQUIRE(p.propagate_all());
    for (size_t i = 0; i < xs.size(); ++i) {
        CHECK(p.lo(vid(xs[i])) == static_cast<double>(5 - i));
        CHECK(p.hi(vid(xs[i])) == static_cast<double>(10 - i));
    }
    CHECK(p.stats().propagate_all_calls == 1);
    // At least the 10 bounds that moved; x_0's lower bound rises one step per rescan.
    const int64_t derived = p.stats().propagate_all_bounds_derived;
    CHECK(derived >= 10);
    CHECK(derived == p.stats().bounds_derived);
    REQUIRE(p.propagate_all());  // at the fixpoint: nothing more
    CHECK(p.stats().propagate_all_bounds_derived == derived);
    p.backtrack(root);
    REQUIRE(snapshot(p, store) == fresh);
}

TEST_CASE("build_until is empty past its deadline and the full store before it",
          "[domain_propagation]") {
    Model m;
    build_chain(m);
    const LinearRowStore full(m);
    const auto now = std::chrono::steady_clock::now();
    CHECK_FALSE(LinearRowStore::build_until(m, 0.0, now - std::chrono::seconds(1)).has_value());
    const std::optional<LinearRowStore> timed =
        LinearRowStore::build_until(m, 0.0, now + std::chrono::hours(1));
    if (!timed.has_value()) {
        FAIL("a far deadline must build the store");
        return;
    }
    CHECK(timed->num_rows() == full.num_rows());
    CHECK(timed->cols() == full.cols());
    CHECK(timed->coefs() == full.coefs());
    CHECK(timed->col_entries() == full.col_entries());
    CHECK(timed->col_rows() == full.col_rows());
    for (int32_t r = 0; r < full.num_rows(); ++r) {
        CHECK(timed->row_start(r) == full.row_start(r));
        CHECK(timed->row_lo(r) == full.row_lo(r));
        CHECK(timed->row_hi(r) == full.row_hi(r));
    }
}

TEST_CASE("propagate_all reports an infeasible root as a conflict", "[domain_propagation]") {
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    m.add_constraint(m.geq(m.sum({x, y}), m.constant(3.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    DomainPropagator p(store, m);
    CHECK_FALSE(p.propagate_all());
    CHECK(p.stats().conflicts == 1);
}

TEST_CASE("the work valve stops a cascade early, and off is one pass", "[domain_propagation]") {
    Model m;
    const std::vector<int32_t> xs = build_chain(m);
    const LinearRowStore store(m);
    SECTION("a valve of one term stops after the call's own rows") {
        DomainPropagationOptions opts;
        opts.max_cascade_terms = 1;
        DomainPropagator p(store, m, opts);
        REQUIRE(p.propagate_all());
        CHECK(p.stats().rows_scanned == 5);
        CHECK(p.stats().valve_stops == 1);
        CHECK(p.hi(vid(xs[5])) == 5.0);  // with row order
        CHECK(p.lo(vid(xs[0])) == 1.0);  // against it: one step only
    }
    SECTION("cascade off scans each row once, and no valve trips") {
        DomainPropagationOptions opts;
        opts.cascade = false;
        DomainPropagator p(store, m, opts);
        REQUIRE(p.propagate_all());
        CHECK(p.stats().rows_scanned == 5);
        CHECK(p.stats().valve_stops == 0);
        CHECK(p.hi(vid(xs[5])) == 5.0);
        CHECK(p.lo(vid(xs[0])) == 1.0);
    }
    SECTION("the default valve lets the chain reach its fixpoint") {
        DomainPropagator p(store, m);
        REQUIRE(p.propagate_all());
        CHECK(p.stats().valve_stops == 0);
        CHECK(p.lo(vid(xs[0])) == 5.0);
    }
}

TEST_CASE("tighten rounds an integral variable inward and rejects NaN and empty requests",
          "[domain_propagation]") {
    Model m;
    const int32_t x = m.int_var(0, 10);
    const int32_t f = m.float_var(0.0, 10.0);
    m.add_constraint(m.leq(m.sum({x, f}), m.constant(20.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    DomainPropagator p(store, m);
    const int32_t base = p.checkpoint();
    REQUIRE(p.tighten(vid(x), 0.3, 4.7));
    CHECK(p.lo(vid(x)) == 1.0);
    CHECK(p.hi(vid(x)) == 4.0);
    REQUIRE(p.tighten(vid(f), 0.3, 4.7));
    CHECK(p.lo(vid(f)) == 0.3);
    CHECK(p.hi(vid(f)) == 4.7);
    CHECK_FALSE(p.fix(vid(x), 2.5));  // no integer in [2.5, 2.5]
    p.backtrack(base);
    CHECK_THROWS_AS(p.fix(vid(f), std::nan("")), std::invalid_argument);
    CHECK(p.lo(vid(f)) == 0.0);
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_FALSE(p.fix(vid(x), inf));
    CHECK_FALSE(p.tighten(vid(x), 0.0, -inf));
    CHECK_FALSE(p.fix(vid(f), inf));
    CHECK_FALSE(p.fix(vid(f), -inf));
    CHECK(p.lo(vid(x)) == 0.0);
    CHECK(p.hi(vid(f)) == 10.0);
    Model open;
    open.float_var(0.0, 1.0);
    CHECK_THROWS_AS(LinearRowStore(open), std::logic_error);
}

TEST_CASE("a row tolerance widens every row and lets a constant row exceed by it",
          "[domain_propagation]") {
    Model m;
    const int32_t f = m.float_var(0.0, 10.0);
    const int32_t g = m.float_var(0.0, 10.0);
    m.add_constraint(m.leq(m.sum({f, g}), m.constant(1.0)));
    m.add_constraint(m.eq_expr(f, m.constant(2.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m, 1e-6);
    REQUIRE(store.num_rows() == 2);
    CHECK(store.row_tolerance() == 1e-6);
    CHECK(store.row_hi(0) == 1.0 + 1e-6);
    CHECK(store.row_lo(0) == -std::numeric_limits<double>::infinity());
    CHECK(store.row_lo(1) == 2.0 - 1e-6);
    CHECK(store.row_hi(1) == 2.0 + 1e-6);
    CHECK_THROWS_AS(LinearRowStore(m, -1e-6), std::invalid_argument);
    CHECK_THROWS_AS(LinearRowStore(m, std::nan("")), std::invalid_argument);

    // h - h <= -5e-7: no terms, residual 5e-7, accepted only within 1e-6.
    Model c;
    const int32_t h = c.float_var(0.0, 1.0);
    c.add_constraint(c.leq(c.sum({h, c.neg(h)}), c.constant(-5e-7)));
    c.close();
    full_evaluate(c);
    for (const double tol : {0.0, 1e-6}) {
        INFO("tolerance " << tol);
        const LinearRowStore cs(c, tol);
        REQUIRE(cs.constant_rows().size() == 1);
        DomainPropagator p(cs, c);
        CHECK(p.propagate_all() == (tol > 0.0));
    }
}

TEST_CASE("a crossing within the margin collapses inside the current box", "[domain_propagation]") {
    Model m;
    const int32_t f = m.float_var(5.0, 10.0);
    const int32_t g = m.float_var(0.0, 10.0);
    m.add_constraint(m.leq(m.sum({f, g}), m.constant(20.0)));
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    DomainPropagator p(store, m);
    REQUIRE(p.tighten(vid(f), 5.0 + 1e-10, 5.0 - 1e-10));
    CHECK(p.lo(vid(f)) == 5.0);
    CHECK(p.hi(vid(f)) == 5.0);
}

TEST_CASE("propagate_all reports a violated constant row", "[domain_propagation]") {
    Model m;
    const int32_t x = m.float_var(0.0, 1.0);
    m.add_constraint(m.leq(m.sum({x, m.neg(x)}), m.constant(-1.0)));  // 0 <= -1
    m.close();
    full_evaluate(m);
    const LinearRowStore store(m);
    CHECK(store.num_rows() == 0);
    REQUIRE(store.constant_rows().size() == 1);
    CHECK(store.constant_rows()[0].residual == 1.0);
    DomainPropagator p(store, m);
    CHECK_FALSE(p.propagate_all());
    CHECK(p.stats().conflicts == 1);

    // A sentinel-sized bound on the wrong side is still a bound: 0 <= -1e25.
    Model big;
    const int32_t y = big.float_var(0.0, 1.0);
    big.add_constraint(big.leq(big.sum({y, big.neg(y)}), big.constant(-1e25)));
    big.close();
    full_evaluate(big);
    const LinearRowStore big_store(big);
    REQUIRE(big_store.constant_rows().size() == 1);
    DomainPropagator bp(big_store, big);
    CHECK_FALSE(bp.propagate_all());
}
