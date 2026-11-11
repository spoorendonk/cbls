// The per-variable cone cache `Model::freeze()` builds (#271): what it holds,
// its cap, and that a walk reading it computes what the collecting walk does,
// to the bit.

#include "test_models.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace cbls;
using cbls_test::bits_of;

namespace {

void require_same_values(const Model& a, const Model& b) {
    const std::vector<double>& va = a.node_values();
    const std::vector<double>& vb = b.node_values();
    REQUIRE(va.size() == vb.size());
    for (size_t i = 0; i < va.size(); ++i) {
        if (bits_of(va[i]) != bits_of(vb[i])) {
            FAIL("node " << i << ": " << va[i] << " against " << vb[i]);
        }
    }
}

// Every node above `var_id`, sorted by topological position: the cone as
// `collect_dirty_cone` and the sort would produce it, derived here from the
// public back-references alone.
std::vector<int32_t> expected_cone(const Model& m, int32_t var_id) {
    std::vector<uint8_t> seen(m.num_nodes(), 0);
    std::vector<int32_t> cone;
    for (const int32_t d : m.dependents(var_id)) {
        if (seen[d] == 0) {
            seen[d] = 1;
            cone.push_back(d);
        }
    }
    for (size_t i = 0; i < cone.size(); ++i) {
        for (const int32_t p : m.parents(cone[i])) {
            if (seen[p] == 0) {
                seen[p] = 1;
                cone.push_back(p);
            }
        }
    }
    std::sort(cone.begin(), cone.end(),
              [&m](int32_t a, int32_t b) { return m.topo_position(a) < m.topo_position(b); });
    return cone;
}

// Sums its inputs, and on a delta adds 1000 per input the walk reports as
// changed: so the two walks agree only if they report the same inputs.
class ChangedSensitiveSum : public CustomInvariant {
public:
    double evaluate(const InvariantInputs& in) override {
        double s = 0.0;
        for (int32_t i = 0; i < in.size(); ++i) {
            s += in.value(i);
        }
        return s;
    }
    double delta(const InvariantInputs& in, ConstSpan<int32_t> changed) override {
        return evaluate(in) + (1000.0 * static_cast<double>(changed.size()));
    }
    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        return std::make_unique<ChangedSensitiveSum>(*this);
    }
};

// Sums its inputs, and throws from `delta` while `*armed` is positive, counting
// it down: user code that refuses once, mid-walk.
class ThrowOnce : public CustomInvariant {
public:
    explicit ThrowOnce(std::shared_ptr<int> armed) : armed_(std::move(armed)) {}
    double evaluate(const InvariantInputs& in) override {
        double s = 0.0;
        for (int32_t i = 0; i < in.size(); ++i) {
            s += in.value(i);
        }
        return s;
    }
    double delta(const InvariantInputs& in, ConstSpan<int32_t> /*changed*/) override {
        if (*armed_ > 0) {
            --*armed_;
            throw std::runtime_error("invariant refused");
        }
        return evaluate(in);
    }
    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        return std::make_unique<ThrowOnce>(*this);
    }

private:
    std::shared_ptr<int> armed_;
};

enum class Shape : uint8_t { Plain, IncSum, Custom };

// A closed, unfrozen model of the given shape with its objective row in place,
// so that `freeze()` adds no node to one copy that the other lacks.
//
//  - Plain: products and sines under max/min rows, no incremental Sum.
//  - IncSum: integral rows, incremental Sums, a shared objective Sum.
//  - Custom: IncSum plus a custom node over several columns feeding a row.
Model build(Shape shape, int n) {
    Model m;
    RNG rng(5);
    std::vector<int32_t> x;
    x.reserve(static_cast<size_t>(n));
    for (int j = 0; j < n; ++j) {
        x.push_back(shape == Shape::Plain ? m.float_var(-2.0, 2.0) : m.int_var(-20, 20));
    }
    std::vector<int32_t> objective_terms;
    for (int i = 0; i < n; ++i) {
        const int32_t a = x[static_cast<size_t>(i)];
        const int32_t b = x[static_cast<size_t>((i + 1) % n)];
        const int32_t c = x[static_cast<size_t>((i + 3) % n)];
        if (shape == Shape::Plain) {
            const int32_t s = m.sin_expr(m.prod(a, b));
            const int32_t row = m.max_expr({s, m.prod(s, c)});
            m.add_constraint(m.leq(row, m.constant(0.5)));
            objective_terms.push_back(s);
        } else {
            const int32_t k = m.constant(static_cast<double>(rng.integers(-9, 10)));
            const int32_t row = m.sum({a, m.neg(b), m.prod(k, c)});
            m.add_constraint(m.leq(row, m.constant(static_cast<double>(rng.integers(-5, 6)))));
            objective_terms.push_back(m.prod(k, a));
        }
    }
    if (shape == Shape::Custom) {
        const int32_t cu = m.custom({x[0], m.prod(x[1], x[2]), m.neg(x[3]), x[5]},
                                    std::make_unique<ChangedSensitiveSum>(), "csum");
        m.add_constraint(m.leq(m.sum({cu, x[4]}), m.constant(30.0)));
    }
    m.minimize(shape == Shape::Plain ? m.max_expr(objective_terms) : m.sum(objective_terms));
    m.close();
    m.add_objective_soft_constraint();
    if (shape == Shape::Plain) {
        REQUIRE(m.inc_sum_nodes().empty());
    } else {
        REQUIRE_FALSE(m.inc_sum_nodes().empty());
    }
    REQUIRE(m.has_custom_nodes() == (shape == Shape::Custom));
    return m;
}

double draw(const Model& m, int32_t v, RNG& rng) {
    const Variable& var = m.var(v);
    if (var.type == VarType::Float) {
        return rng.uniform(var.lb, var.ub);
    }
    return static_cast<double>(
        rng.integers(static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
}

// One step of a random walk sequence, applied to `m` alone; `rng` is a copy
// per model so that both draw the same.
void step(Model& m, RNG& rng) {
    const auto n = static_cast<int64_t>(m.num_vars());
    const auto v = static_cast<int32_t>(rng.integers(0, n));
    const double old_value = m.var(v).value;
    const double new_value = draw(m, v, rng);
    switch (rng.integers(0, 5)) {
        case 0:  // FJ's scored probe and its rollback
            m.var_mut(v).value = new_value;
            probe_scalar_move(m, v, old_value);
            m.var_mut(v).value = old_value;
            delta_evaluate(m, &v, 1, DeltaMode::Rollback);
            break;
        case 1:  // FJ's commit
            m.var_mut(v).value = new_value;
            commit_scalar_move(m, v, old_value);
            break;
        case 2:  // a plain one-variable commit
            m.var_mut(v).value = new_value;
            delta_evaluate(m, &v, 1, DeltaMode::Commit);
            break;
        case 3:  // a plain probe, compared before its rollback
            m.var_mut(v).value = new_value;
            delta_evaluate(m, &v, 1, DeltaMode::Probe);
            m.var_mut(v).value = old_value;
            delta_evaluate(m, &v, 1, DeltaMode::Rollback);
            break;
        default: {  // two variables: always the collecting walk
            const auto w = static_cast<int32_t>((v + 1) % n);
            m.var_mut(v).value = new_value;
            m.var_mut(w).value = draw(m, w, rng);
            const std::vector<int32_t> both{v, w};
            delta_evaluate(m, both, DeltaMode::Commit);
            break;
        }
    }
}

// Runs the same walk sequence on an unfrozen model and on its copy frozen with
// `cap`, comparing every node value after each step.
void run_twins(Shape shape, size_t cap, int steps) {
    const Model built = build(shape, 12);
    Model open = built;
    Model frozen = built;
    frozen.freeze(cap);
    REQUIRE(frozen.num_nodes() == open.num_nodes());
    REQUIRE(open.num_cached_cones() == 0);
    RNG rng_open(17);
    RNG rng_frozen(17);
    for (int s = 0; s < steps; ++s) {
        step(open, rng_open);
        step(frozen, rng_frozen);
        require_same_values(open, frozen);
    }
}

}  // namespace

TEST_CASE("a frozen model caches each variable's cone in topological order", "[cone_cache]") {
    for (const Shape shape : {Shape::Plain, Shape::IncSum, Shape::Custom}) {
        CAPTURE(static_cast<int>(shape));
        Model m = build(shape, 12);
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            REQUIRE_FALSE(m.has_cached_cone(v));  // unfrozen: none
        }
        m.freeze();
        REQUIRE(m.num_cached_cones() == m.num_vars());
        size_t entries = 0;
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            REQUIRE(m.has_cached_cone(v));
            const ConstSpan<int32_t> got = m.cached_cone(v);
            const std::vector<int32_t> want = expected_cone(m, v);
            REQUIRE(std::vector<int32_t>(got.begin(), got.end()) == want);
            entries += want.size();
        }
        CHECK(m.cone_cache_bytes() == entries * sizeof(int32_t));
    }
}

TEST_CASE("one-variable walks on a frozen model match the unfrozen walk to the bit",
          "[cone_cache]") {
    // The cached cone is the collecting walk's sorted list, so every probe,
    // rollback and commit computes the same values in the same order -- on the
    // plain walk, under the incremental-Sum rules, and through a custom node
    // whose value depends on which inputs the walk reports as changed.
    for (const Shape shape : {Shape::Plain, Shape::IncSum, Shape::Custom}) {
        CAPTURE(static_cast<int>(shape));
        cone_cache_counters() = ConeCacheCounters{};
        run_twins(shape, Model::kDefaultConeCacheBytes, 2000);
        // Half the walks were the frozen model's, all from the cache; the
        // unfrozen model's all collected. A Rollback served from the undo log
        // walks nothing, and a two-variable walk is in neither count.
        const ConeCacheCounters counts = cone_cache_counters();
        CHECK(counts.cached_walks > 0);
        CHECK(counts.cached_walks == counts.fallback_walks);
    }
}

TEST_CASE("the cone cache's cap leaves the rest uncached and their walks unchanged",
          "[cone_cache]") {
    for (const Shape shape : {Shape::Plain, Shape::IncSum, Shape::Custom}) {
        CAPTURE(static_cast<int>(shape));
        Model full = build(shape, 12);
        full.freeze();
        const size_t all_bytes = full.cone_cache_bytes();
        REQUIRE(all_bytes > 0);

        Model capped = build(shape, 12);
        capped.freeze(all_bytes / 2);
        CHECK(capped.cone_cache_bytes() <= all_bytes / 2);
        CHECK(capped.num_cached_cones() > 0);
        CHECK(capped.num_cached_cones() < capped.num_vars());
        for (int32_t v = 0; v < static_cast<int32_t>(capped.num_vars()); ++v) {
            // A prefix: the first variable left out leaves every later one out.
            CHECK(capped.has_cached_cone(v) ==
                  (static_cast<size_t>(v) < capped.num_cached_cones()));
            if (capped.has_cached_cone(v)) {
                const ConstSpan<int32_t> got = capped.cached_cone(v);
                REQUIRE(std::vector<int32_t>(got.begin(), got.end()) == expected_cone(capped, v));
            }
        }

        Model none = build(shape, 12);
        none.freeze(0);
        CHECK(none.num_cached_cones() == 0);
        CHECK(none.cone_cache_bytes() == 0);

        cone_cache_counters() = ConeCacheCounters{};
        run_twins(shape, all_bytes / 2, 1000);
        const ConeCacheCounters counts = cone_cache_counters();
        CHECK(counts.cached_walks > 0);
        CHECK(counts.fallback_walks > counts.cached_walks);  // the twin's and the capped's

        cone_cache_counters() = ConeCacheCounters{};
        run_twins(shape, 0, 500);
        CHECK(cone_cache_counters().cached_walks == 0);
    }
}

TEST_CASE("an unused variable's empty cone is cached, and a cap of 0 caches nothing",
          "[cone_cache]") {
    // Its cone costs no bytes, so only the cap-0 early return keeps it out.
    const auto build_with_unused = [] {
        Model m;
        const int32_t x = m.int_var(0, 9);
        (void)m.int_var(0, 9);  // in no expression
        m.add_constraint(m.leq(x, m.constant(5.0)));
        m.close();
        return m;
    };
    Model all = build_with_unused();
    all.freeze();
    CHECK(all.num_cached_cones() == all.num_vars());
    CHECK(all.has_cached_cone(1));
    CHECK(all.cached_cone(1).empty());

    Model none = build_with_unused();
    none.freeze(0);
    CHECK(none.num_cached_cones() == 0);
    CHECK_FALSE(none.has_cached_cone(1));
}

TEST_CASE("a throw inside a cached-cone walk leaves no dirty flag behind", "[cone_cache]") {
    // The cached walk sets its whole cone's dirty flags before evaluating any
    // node. A flag left set after the throw would make the next collecting
    // walk skip that node as already seen, and leave it stale.
    auto armed = std::make_shared<int>(0);
    Model m;
    const int32_t x = m.int_var(0, 9);
    const int32_t y = m.int_var(0, 9);
    const int32_t c = m.custom({x, m.prod(x, y)}, std::make_unique<ThrowOnce>(armed), "throws");
    m.add_constraint(m.leq(m.sum({c, y}), m.constant(20.0)));
    m.minimize(m.sum({c, m.neg(y)}));
    m.close();
    m.freeze();
    const int32_t xid = handle_to_var_id(x);
    const int32_t yid = handle_to_var_id(y);
    REQUIRE(m.has_cached_cone(xid));
    REQUIRE(m.has_custom_nodes());
    Model twin = m;  // only ever fully evaluated, so never reaches `delta`

    *armed = 1;
    m.var_mut(xid).value = 4.0;
    cone_cache_counters() = ConeCacheCounters{};
    REQUIRE_THROWS_AS(delta_evaluate(m, &xid, 1, DeltaMode::Commit), std::runtime_error);
    REQUIRE(*armed == 0);
    REQUIRE(cone_cache_counters().cached_walks == 1);
    full_evaluate(m);  // the documented recovery

    m.var_mut(xid).value = 7.0;
    m.var_mut(yid).value = 2.0;
    const std::vector<int32_t> both{xid, yid};
    delta_evaluate(m, both, DeltaMode::Commit);  // the collecting walk

    twin.var_mut(xid).value = 7.0;
    twin.var_mut(yid).value = 2.0;
    full_evaluate(twin);
    require_same_values(twin, m);
}

TEST_CASE("solve freezes the model and reports its cone-cache walks", "[cone_cache]") {
    // The trajectory does not depend on the cache: a run with it off returns
    // the same assignment after the same iterations.
    SearchConfig config;
    config.max_iterations = 20000;
    Model with_cache = build(Shape::IncSum, 12);
    Model without = build(Shape::IncSum, 12);
    config.cone_cache_bytes = Model::kDefaultConeCacheBytes;
    const SearchResult a = solve(with_cache, 0.0, 3, true, nullptr, nullptr, 0, nullptr, config);
    config.cone_cache_bytes = 0;
    const SearchResult b = solve(without, 0.0, 3, true, nullptr, nullptr, 0, nullptr, config);

    REQUIRE(with_cache.is_frozen());
    REQUIRE(without.is_frozen());
    CHECK(a.counters.cone_cache_walks > 0);
    CHECK(a.counters.cone_fallback_walks == 0);
    CHECK(b.counters.cone_cache_walks == 0);
    CHECK(b.counters.cone_fallback_walks == a.counters.cone_cache_walks);

    CHECK(a.iterations == b.iterations);
    CHECK(bits_of(a.objective) == bits_of(b.objective));
    REQUIRE(a.best_state.values.size() == b.best_state.values.size());
    for (size_t i = 0; i < a.best_state.values.size(); ++i) {
        CHECK(bits_of(a.best_state.values[i]) == bits_of(b.best_state.values[i]));
    }
}
