// OR-Tools-style near-value perturbation (#217): `random_near_value`, and FJ's
// kick with GFJConfig::perturb_near_value on. The default kick is pinned by
// test_perturb.cpp and the trajectory fences; what is pinned here is the new
// sampler (in the domain, near the value, finite on unbounded domains) and that
// the flag reaches the kick and the search.
#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cbls;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kRatio = 0.2;  // OR-Tools' feasibility_jump_var_perburbation_range_ratio
constexpr int kDraws = 20'000;

Variable scalar(VarType type, double lb, double ub, double value) {
    Variable var;
    var.type = type;
    var.lb = lb;
    var.ub = ub;
    var.value = value;
    return var;
}

// Every draw in the domain (the clamp box on an infinite side), finite, and
// integral on an integral type.
void require_in_domain(const Variable& var, bool exclude_current) {
    RNG rng(7);
    const DomainWindow w = domain_window(var);
    for (int i = 0; i < kDraws; ++i) {
        const double x = random_near_value(var, rng, kRatio, exclude_current);
        CAPTURE(var.lb, var.ub, var.value, x);
        REQUIRE(std::isfinite(x));
        REQUIRE(x >= w.lo);
        REQUIRE(x <= w.hi);
        REQUIRE(x >= var.lb);
        REQUIRE(x <= var.ub);
        if (var.type != VarType::Float) {
            REQUIRE(x == std::round(x));
        }
        if (exclude_current) {
            REQUIRE(x != var.value);
        }
    }
}

double median_distance(const Variable& var) {
    RNG rng(11);
    std::vector<double> d;
    d.reserve(kDraws);
    for (int i = 0; i < kDraws; ++i) {
        d.push_back(std::abs(random_near_value(var, rng, kRatio) - var.value));
    }
    std::nth_element(d.begin(), d.begin() + kDraws / 2, d.end());
    return d[kDraws / 2];
}

}  // namespace

TEST_CASE("random_near_value stays in the domain (#217)", "[perturb][near_value]") {
    for (const bool exclude : {false, true}) {
        for (const double value : {-5.0, -4.0, 0.0, 37.0, 99.0, 100.0}) {
            require_in_domain(scalar(VarType::Int, -5, 100, value), exclude);
            require_in_domain(scalar(VarType::Float, -5, 100, value), exclude);
        }
        for (const double value : {0.0, 1.0}) {
            require_in_domain(scalar(VarType::Bool, 0, 1, value), exclude);
        }
        // A two-value Int domain and a value outside the domain.
        require_in_domain(scalar(VarType::Int, 3, 4, 3), exclude);
        require_in_domain(scalar(VarType::Int, 3, 4, 4), exclude);
        require_in_domain(scalar(VarType::Int, 0, 1'000, 5'000), exclude);
        require_in_domain(scalar(VarType::Float, 0, 1, -2.5), exclude);
        // A narrow Float far from zero: an offset below one ulp rounds back onto
        // the value, which exclude_current must still step off.
        require_in_domain(scalar(VarType::Float, 1e9, 1e9 + 1e-3, 1e9), exclude);
    }
}

TEST_CASE("random_near_value on a pinned domain returns the pin (#217)", "[perturb][near_value]") {
    RNG rng(1);
    REQUIRE(random_near_value(scalar(VarType::Int, 4, 4, 4), rng, kRatio) == 4.0);
    REQUIRE(random_near_value(scalar(VarType::Float, 2.5, 2.5, 2.5), rng, kRatio) == 2.5);
}

TEST_CASE("random_near_value stays near the value (#217)", "[perturb][near_value]") {
    // Strictly inside, the reference draws within ceil(20% of the chosen side)
    // of the value's neighbour; log-uniform, so the median distance is orders of
    // magnitude below the uniform draw's ~2.5e6 on this domain.
    for (const VarType type : {VarType::Int, VarType::Float}) {
        const Variable var = scalar(type, 0, 1e7, 5e6);
        RNG rng(3);
        for (int i = 0; i < kDraws; ++i) {
            const double x = random_near_value(var, rng, kRatio);
            CAPTURE(x);
            REQUIRE(x != var.value);
            REQUIRE(std::abs(x - var.value) <= (0.2 * 5e6) + 1.0);
        }
        CAPTURE(type);
        REQUIRE(median_distance(var) < 1e5);
    }
    // At the minimum, near the minimum: within 20% of the whole range.
    const Variable at_lb = scalar(VarType::Int, 0, 1'000, 0);
    RNG rng(5);
    for (int i = 0; i < kDraws; ++i) {
        REQUIRE(random_near_value(at_lb, rng, kRatio) <= 200.0);
    }
}

TEST_CASE("random_near_value flips a binary at the range ratio (#217)", "[perturb][near_value]") {
    for (const double value : {0.0, 1.0}) {
        const Variable var = scalar(VarType::Bool, 0, 1, value);
        RNG rng(9);
        int flips = 0;
        for (int i = 0; i < kDraws; ++i) {
            flips += random_near_value(var, rng, kRatio) != value ? 1 : 0;
        }
        const double rate = static_cast<double>(flips) / kDraws;
        CAPTURE(value, rate);
        REQUIRE(rate > 0.18);
        REQUIRE(rate < 0.22);
    }
}

TEST_CASE("random_near_value is finite and near on unbounded domains (#217)",
          "[perturb][near_value]") {
    // The clamp box stands in for an infinite side, as for random_in_domain, so
    // no width overflows and nothing lands at ±inf; and the draw stays near.
    for (const VarType type : {VarType::Int, VarType::Float}) {
        for (const auto& [lb, ub] :
             std::vector<std::pair<double, double>>{{-kInf, kInf}, {0, kInf}, {-kInf, 0}}) {
            for (const double value : {0.0, 12.0, -12.0}) {
                if (value < lb || value > ub) {
                    continue;
                }
                for (const bool exclude : {false, true}) {
                    require_in_domain(scalar(type, lb, ub, value), exclude);
                }
                CAPTURE(type, lb, ub, value);
                const double clamp = (type == VarType::Int) ? kRandomIntInfClamp : kRandomInfClamp;
                REQUIRE(median_distance(scalar(type, lb, ub, value)) < 0.01 * clamp);
            }
        }
    }
}

namespace {

// 1000 Int columns at the middle of [0, 1e7], each read by one row, as a
// MIPfeas column under the 1e7 inf-clamp would be.
void build_wide_ints(Model& m) {
    for (int i = 0; i < 1'000; ++i) {
        auto x = m.int_var(0, 10'000'000);
        m.add_constraint(m.leq(x, m.constant(9'000'000.0)));
    }
    m.close();
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        m.var_mut(v).value = 5e6;
    }
    full_evaluate(m);
}

}  // namespace

TEST_CASE("perturb_near_value is off by default (#217)", "[perturb][near_value]") {
    REQUIRE_FALSE(GFJConfig{}.perturb_near_value);
    REQUIRE_FALSE(SearchConfig{}.perturb_near_value);
    REQUIRE(GFJConfig{}.perturb_near_value_probability == 0.05);
    REQUIRE(GFJConfig{}.perturb_range_ratio == 0.2);
}

TEST_CASE("FJ's kick with perturb_near_value moves ~5% of columns, near (#217)",
          "[perturb][near_value]") {
    for (const bool near : {false, true}) {
        Model m;
        build_wide_ints(m);
        ViolationManager vm(m);
        RNG rng(1);
        GFJConfig config;
        config.perturb_near_value = near;
        FeasibilityJump fj(m, vm, rng, config);
        fj.begin(/*set_initial_x=*/false);
        // The caller's probability is the default 0.1; the near-value kick
        // ignores it for scalars and uses its own 0.05.
        fj.perturb(0.1);
        int moved = 0;
        double farthest = 0.0;
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            const double d = std::abs(m.var(v).value - 5e6);
            moved += d > 0.0 ? 1 : 0;
            farthest = std::max(farthest, d);
        }
        CAPTURE(near, moved, farthest);
        if (near) {
            REQUIRE(moved > 30);
            REQUIRE(moved < 75);
            REQUIRE(farthest <= 1e6 + 1.0);
        } else {
            REQUIRE(moved > 75);
            REQUIRE(farthest > 1e6 + 1.0);
        }
    }
}

TEST_CASE("FJ's near-value kick is never a no-op (#217)", "[perturb][near_value]") {
    // One Int column at its lower bound: the per-variable draw misses w.p. 0.95,
    // and when it hits it lands back on the bound w.p. 1/(bit width + 1). The
    // fallback must then move it, near the bound.
    for (uint64_t seed = 1; seed <= 200; ++seed) {
        Model m;
        auto x = m.int_var(0, 1'000);
        m.add_constraint(m.leq(x, m.constant(10.0)));
        m.close();
        full_evaluate(m);
        ViolationManager vm(m);
        RNG rng(seed);
        GFJConfig config;
        config.perturb_near_value = true;
        FeasibilityJump fj(m, vm, rng, config);
        fj.begin(/*set_initial_x=*/true);
        const double before = m.var(0).value;
        fj.perturb(0.1);
        CAPTURE(seed, before, m.var(0).value);
        REQUIRE(m.var(0).value != before);
        REQUIRE(std::abs(m.var(0).value - before) <= 201.0);
    }
}

TEST_CASE("SearchConfig::perturb_near_value reaches solve()'s kick (#217)",
          "[perturb][near_value]") {
    // An integer-infeasible model: every batch stalls, so the search kicks to
    // the iteration cap. There is no clock, so each arm is deterministic and the
    // two differ only in the kick. Where a batch ends (and so the final count
    // past the cap) follows the trajectory: if the flag were dropped on the way
    // to FJ, both arms would report the same count.
    for (const uint64_t seed : {1, 2, 3}) {
        std::array<int64_t, 2> iterations = {0, 0};
        for (const bool near : {false, true}) {
            Model m;
            std::vector<int32_t> xs;
            xs.reserve(50);
            for (int i = 0; i < 50; ++i) {
                xs.push_back(m.int_var(0, 10'000'000));
            }
            m.add_constraint(m.eq_expr(m.sum(xs), m.constant(0.5)));
            m.close();
            SearchConfig config;
            config.max_iterations = 20'000;
            config.use_compound_moves = false;
            config.perturb_near_value = near;
            const SearchResult r =
                solve(m, /*time_limit=*/0.0, seed, true, nullptr, nullptr, 3, nullptr, config);
            REQUIRE_FALSE(r.feasible);
            REQUIRE(r.perturbations > 0);
            iterations[near ? 1 : 0] = r.iterations;
        }
        CAPTURE(seed, iterations[0], iterations[1]);
        REQUIRE(iterations[0] != iterations[1]);
    }
}
