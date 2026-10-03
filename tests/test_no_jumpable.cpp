// A model with no Feasibility-Jump-jumpable variable (#201).
//
// FJ jumps Bool/Int/Float variables only. On a model whose decisions are all
// List/Set variables -- CVRP in the List encoding is the case that found this --
// an FJ batch has nothing to jump. At the automatic 0.33 structural probability
// two thirds of all batches were such empty FJ batches, and each one spun its
// GLS loop to the iteration limit, reported itself stuck and triggered #102's
// unproductive diversification kick, which re-randomises every List. The
// structural batches never kept their progress, and the default arm returned no
// feasible solution where `structural_batch_probability = 1.0` was feasible at
// once.
//
// The fix routes every batch to the structural sweep when FJ has no movable
// variable, and makes SearchResult::iterations report the unit max_iterations
// was actually charged in (the batch count, on such a run) rather than 0.
//
// Every model here CONSUMES its Lists: capacity reads each route through a
// `lambda_sum`, distance through a `pair_lambda_sum`, so a List that the search
// leaves badly placed is visible in both the rows and the objective. A probe
// over a List nothing reads would prove nothing.
//
// All runs are iteration-budgeted with time_limit = 0, so nothing reads a clock.

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/search.h>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cbls;

namespace {

// Deterministic pseudo-data from a splitmix-style mixer, so the instance does
// not depend on any RNG implementation.
uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

double unit(uint64_t key) {
    return static_cast<double>(mix(key) >> 11) * (1.0 / 9007199254740992.0);
}

struct Cvrp {
    int customers = 30;
    int routes = 6;
    double capacity = 0.0;
    std::vector<double> demand;  // per customer element 0..customers-1
    std::vector<double> dist;    // (customers + 1)^2, node 0 is the depot
};

// 30 customers, 6 routes, total demand within ~6% of the fleet's capacity: tight
// enough that a search whose Lists are re-randomised every few batches does not
// stumble into a packing, loose enough that a structural search finds one in a
// few hundred batches.
Cvrp make_cvrp() {
    Cvrp c;
    const int n = c.customers + 1;
    std::vector<double> x(n);
    std::vector<double> y(n);
    for (int i = 0; i < n; ++i) {
        x[i] = 100.0 * unit((3ULL * i) + 1);
        y[i] = 100.0 * unit((3ULL * i) + 2);
    }
    double total = 0.0;
    for (int e = 0; e < c.customers; ++e) {
        c.demand.push_back(1.0 + static_cast<double>(mix(1000ULL + e) % 9));
        total += c.demand.back();
    }
    c.capacity = std::ceil(total * 1.06 / c.routes);
    c.dist.resize(static_cast<size_t>(n) * n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c.dist[(static_cast<size_t>(i) * n) + j] = std::hypot(x[i] - x[j], y[i] - y[j]);
        }
    }
    return c;
}

// Hexaly-style List encoding: one List per route over the customer universe,
// partitioned with Cover::Exact, capacity as `lambda_sum <= Q`, distance as a
// depot-anchored `pair_lambda_sum`. `extra_scalar` optionally adds an Int
// variable read by the objective, with the given bounds.
struct Built {
    std::vector<int32_t> routes;
};

Built build_cvrp(Model& m, const Cvrp& c, bool extra_scalar = false, int scalar_lb = 0,
                 int scalar_ub = 0) {
    Built b;
    const int n = c.customers + 1;
    const double* dist = c.dist.data();
    const double* demand = c.demand.data();
    for (int r = 0; r < c.routes; ++r) {
        b.routes.push_back(m.list_var(c.customers, 0, c.customers, ListInit::Empty));
    }
    m.add_list_partition(b.routes, Cover::Exact);
    std::vector<int32_t> lengths;
    for (const int32_t r : b.routes) {
        const int32_t load = m.lambda_sum(r, [demand](int e) { return demand[e]; });
        m.add_constraint(m.leq(load, m.constant(c.capacity)));
        lengths.push_back(m.pair_lambda_sum(
            r, [dist, n](int a, int b2) { return dist[((a + 1) * n) + b2 + 1]; },
            [dist](int e) { return dist[e + 1]; },
            [dist, n](int e) { return dist[static_cast<size_t>(e + 1) * n]; }));
    }
    if (extra_scalar) {
        lengths.push_back(m.int_var(scalar_lb, scalar_ub));
    }
    m.minimize(m.sum(lengths));
    m.close();
    return b;
}

// Independent check of the returned assignment: every customer served exactly
// once, every route within capacity. Read off best_state, not the engine's
// `feasible` flag.
bool independently_feasible(const Cvrp& c, const Built& b, const SearchResult& r) {
    std::vector<int> seen(c.customers, 0);
    for (const int32_t h : b.routes) {
        const auto& elems = r.best_state.elements[static_cast<size_t>(handle_to_var_id(h))];
        double load = 0.0;
        for (const int32_t e : elems) {
            ++seen[static_cast<size_t>(e)];
            load += c.demand[static_cast<size_t>(e)];
        }
        if (load > c.capacity + 1e-9) {
            return false;
        }
    }
    for (const int s : seen) {
        if (s != 1) {
            return false;
        }
    }
    return true;
}

SearchResult run(Model& m, uint64_t seed, int64_t max_iterations, const SearchConfig& base = {}) {
    SearchConfig cfg = base;
    cfg.max_iterations = max_iterations;
    return solve(m, /*time_limit=*/0.0, seed, /*use_fj=*/true, /*hook=*/nullptr, /*lns=*/nullptr,
                 /*lns_interval=*/0, /*callback=*/nullptr, cfg);
}

}  // namespace

// The regression test. Shown red on the unfixed engine (1304c43): infeasible on
// every seed below at this budget -- and not merely because the old engine
// charged the budget in GLS iterations and so ran only a batch or two. Probed
// on this exact instance at max_iterations = 200000, the unfixed engine ran
// ~900 batches (~600 FJ, ~300 structural) and took ~560 kicks per run, and was
// still infeasible on all of seeds 1-6. The fixed engine is feasible within 200
// structural batches on all six.
TEST_CASE("a List-only CVRP model reaches feasibility at default settings",
          "[search][structural][no_jumpable]") {
    const Cvrp c = make_cvrp();
    for (const uint64_t seed : {1ULL, 2ULL, 3ULL}) {
        INFO("seed " << seed);
        Model m;
        const Built b = build_cvrp(m, c);
        const SearchResult r = run(m, seed, /*max_iterations=*/1000);
        REQUIRE(r.feasible);
        REQUIRE(independently_feasible(c, b, r));
        // The mechanism, not just the outcome: no FJ batch was scheduled, so no
        // empty batch could report itself stuck and kick.
        REQUIRE(r.counters.fj_batches == 0);
        REQUIRE(r.counters.novelty_batches == 0);
        REQUIRE(r.counters.structural_batches == r.counters.batches);
    }
}

TEST_CASE("a structural-only run reports its batches as iterations and stops on max_iterations",
          "[search][structural][no_jumpable]") {
    const Cvrp c = make_cvrp();
    Model m;
    build_cvrp(m, c);
    const SearchResult r = run(m, /*seed=*/1, /*max_iterations=*/1000);
    // Has an objective, so the run does not stop at its first feasible point:
    // it spends the whole budget, and the budget is exactly 1000 batches.
    REQUIRE(r.termination == TerminationReason::IterationLimit);
    REQUIRE(r.counters.batches == 1000);
    REQUIRE(r.iterations == 1000);
    REQUIRE(r.iterations == r.counters.batches);
}

TEST_CASE("an explicit structural probability cannot schedule empty FJ batches",
          "[search][structural][no_jumpable]") {
    // The probability apportions batches between FJ and the structural sweep;
    // with nothing for FJ to jump there is nothing to apportion, so even an
    // explicit 0 runs the sweep rather than a run of guaranteed no-op batches.
    const Cvrp c = make_cvrp();
    for (const double p : {0.0, 0.33}) {
        INFO("probability " << p);
        Model m;
        build_cvrp(m, c);
        SearchConfig cfg;
        cfg.structural_batch_probability = p;
        const SearchResult r = run(m, /*seed=*/1, /*max_iterations=*/300, cfg);
        REQUIRE(r.counters.fj_batches == 0);
        REQUIRE(r.counters.structural_batches == 300);
        REQUIRE(r.feasible);
    }
}

TEST_CASE("a mixed model counts as having no FJ work only when every scalar is fixed",
          "[search][structural][no_jumpable]") {
    const Cvrp c = make_cvrp();
    SECTION("a fixed scalar leaves FJ nothing to jump") {
        Model m;
        build_cvrp(m, c, /*extra_scalar=*/true, /*scalar_lb=*/3, /*scalar_ub=*/3);
        const SearchResult r = run(m, /*seed=*/1, /*max_iterations=*/1000);
        REQUIRE(r.counters.fj_batches == 0);
        REQUIRE(r.counters.structural_batches == r.counters.batches);
        REQUIRE(r.feasible);
    }
    SECTION("a movable scalar keeps the automatic batch mix") {
        Model m;
        build_cvrp(m, c, /*extra_scalar=*/true, /*scalar_lb=*/0, /*scalar_ub=*/5);
        const SearchResult r = run(m, /*seed=*/1, /*max_iterations=*/20000);
        REQUIRE(r.counters.fj_batches > 0);
        REQUIRE(r.counters.structural_batches > 0);
        // GLS iterations dominate the batch count whenever FJ runs, so the
        // reported count is the GLS count, as it always was.
        REQUIRE(r.iterations >= r.counters.batches);
    }
}
