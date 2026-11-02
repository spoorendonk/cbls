// LinearRowStore and DomainPropagator (#329): propagation over linear rows,
// with an exact undo.

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/domain_propagation.h>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cbls;

namespace {

int32_t vid(int32_t handle) {
    return handle_to_var_id(handle);
}

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

// One random operation: checkpoint, fix, tighten or backtrack. Returns the
// tighten/fix result, true otherwise. `levels` mirrors the checkpoint stack.
bool random_op(DomainPropagator& p, const Model& m, RNG& rng, std::vector<int32_t>& levels) {
    const int64_t op = rng.integers(0, 4);
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
    const bool integral = m.var(v).type == VarType::Int;
    double a = rng.uniform(lo, hi);
    if (integral) {
        a = std::round(a);
    }
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

TEST_CASE("random fix and backtrack restore bounds and activities bit for bit",
          "[domain_propagation]") {
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
        for (int step = 0; step < 300; ++step) {
            const size_t before = levels.size();
            random_op(p, m, rng, levels);
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
