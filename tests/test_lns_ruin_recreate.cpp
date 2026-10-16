// LNS ruin and recreate on a ListPartition (#202).
//
// The destroy step used to shuffle a List's order and hand the result to FJ,
// which cannot move a List, so on a partitioned-List model no repair was ever
// accepted (0 of every attempt on CVRP). These pin what replaced it: the cover
// and the length bounds survive, the operation is a function of the seed, and
// it finds improvements.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/lns.h>
#include <cbls/move_generator.h>
#include <cbls/randomize.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

using namespace cbls;

namespace {

/// A small capacitated routing model on a partition of Lists: `customers`
/// points on a deterministic spiral, unit demand, `n_routes` routes of capacity
/// `capacity` and minimum length 1, minimising route length from a depot at the
/// origin. Distances are Euclidean rounded to an integer, as TSPLIB's are, so a
/// reversed route costs exactly what the route did.
struct Routing {
    int customers;
    Model m;
    std::vector<int32_t> route_vars;  // variable ids
    std::vector<double> x;
    std::vector<double> y;

    explicit Routing(Cover cover, int customers = 30, int n_routes = 5, int capacity = 8)
        : customers(customers) {
        for (int i = 0; i < customers; ++i) {
            const double a = 0.7 * i;
            const double r = 1.0 + (0.3 * i);
            x.push_back(r * std::cos(a));
            y.push_back(r * std::sin(a));
        }
        std::vector<int32_t> routes;
        std::vector<int32_t> lengths;
        for (int r = 0; r < n_routes; ++r) {
            const int32_t route = m.list_var(customers, 1, customers, ListInit::Random);
            routes.push_back(route);
            route_vars.push_back(handle_to_var_id(route));
            const int32_t load = m.lambda_sum(route, [](int) { return 1.0; });
            m.add_constraint(m.leq(load, m.constant(capacity)));
            lengths.push_back(m.pair_lambda_sum(
                route, [this](int a, int b) { return dist(a, b); },
                [this](int e) { return depot(e); }, [this](int e) { return depot(e); }));
        }
        m.add_list_partition(routes, cover);
        m.minimize(m.sum(lengths));
        m.close();
    }
    [[nodiscard]] double dist(int a, int b) const {
        return std::round(10.0 * std::hypot(x[static_cast<size_t>(a)] - x[static_cast<size_t>(b)],
                                            y[static_cast<size_t>(a)] - y[static_cast<size_t>(b)]));
    }
    [[nodiscard]] double depot(int e) const {
        return std::round(10.0 * std::hypot(x[static_cast<size_t>(e)], y[static_cast<size_t>(e)]));
    }
    void lay_out(RNG& rng) {
        randomize_list_partition(m, m.list_partitions().front(), rng);
        full_evaluate(m);
    }
    [[nodiscard]] std::vector<std::vector<int32_t>> routes() const {
        std::vector<std::vector<int32_t>> out;
        out.reserve(route_vars.size());
        for (const int32_t v : route_vars) {
            out.push_back(m.var(v).elements);
        }
        return out;
    }
    /// How many routes hold each customer.
    [[nodiscard]] std::vector<int> coverage() const {
        std::vector<int> seen(static_cast<size_t>(customers), 0);
        for (const auto& r : routes()) {
            for (const int32_t e : r) {
                ++seen[static_cast<size_t>(e)];
            }
        }
        return seen;
    }
};

std::shared_ptr<const NeighbourList> neighbours_of(const Routing& r) {
    return std::make_shared<const NeighbourList>(
        nearest_neighbours(r.customers, 8, [&r](int a, int b) { return r.dist(a, b); }));
}

}  // namespace

TEST_CASE("ruin and recreate keeps an Exact cover and the length bounds (#202)",
          "[lns][partition]") {
    for (const bool related : {false, true}) {
        INFO("neighbours " << related);
        Routing r(Cover::Exact);
        RNG rng(7);
        r.lay_out(rng);
        ViolationManager vm(r.m);
        LNS lns;
        if (related) {
            lns.set_neighbours(neighbours_of(r));
        }
        for (int round = 0; round < 50; ++round) {
            lns.destroy_repair(r.m, vm, rng, /*repair_time_limit=*/0.0);
            for (const int seen : r.coverage()) {
                REQUIRE(seen == 1);
            }
            for (const int32_t v : r.route_vars) {
                const auto size = static_cast<int32_t>(r.m.var(v).elements.size());
                REQUIRE(size >= r.m.var(v).min_size);
                REQUIRE(size <= r.m.var(v).max_size);
            }
        }
    }
}

TEST_CASE("ruin and recreate places each element at most once under AtMostOnce (#202)",
          "[lns][partition]") {
    Routing r(Cover::AtMostOnce);
    RNG rng(11);
    r.lay_out(rng);
    ViolationManager vm(r.m);
    LNS lns;
    lns.set_neighbours(neighbours_of(r));
    for (int round = 0; round < 50; ++round) {
        lns.destroy_repair(r.m, vm, rng, 0.0);
        for (const int seen : r.coverage()) {
            REQUIRE(seen <= 1);
        }
    }
}

TEST_CASE("ruin and recreate is a function of the seed (#202)", "[lns][partition]") {
    std::vector<std::vector<std::vector<int32_t>>> runs;
    for (int rep = 0; rep < 2; ++rep) {
        Routing r(Cover::Exact);
        RNG rng(3);
        r.lay_out(rng);
        ViolationManager vm(r.m);
        LNS lns;
        lns.set_neighbours(neighbours_of(r));
        for (int round = 0; round < 20; ++round) {
            lns.destroy_repair(r.m, vm, rng, 0.0);
        }
        runs.push_back(r.routes());
    }
    REQUIRE(runs[0] == runs[1]);
}

TEST_CASE("ruin and recreate moves elements between lists (#202)", "[lns][partition]") {
    // The old destroy shuffled the order WITHIN a route and FJ could not move a
    // List, so it could never change which route serves whom. Here every route
    // holds two customers -- capacity 2, ten customers, five routes -- so a
    // shuffle is at best a reversal, which costs exactly the same on integral
    // distances: the old LNS accepted nothing on this model, ever. The pairing
    // is deliberately bad (i with i + 5, across the spiral), so moving
    // customers between routes is the only way to improve it.
    Routing r(Cover::Exact, /*customers=*/10, /*n_routes=*/5, /*capacity=*/2);
    for (int i = 0; i < 5; ++i) {
        r.m.var_mut(r.route_vars[static_cast<size_t>(i)]).elements = {i, i + 5};
    }
    full_evaluate(r.m);
    RNG rng(5);
    ViolationManager vm(r.m);
    LNS lns;
    lns.set_neighbours(neighbours_of(r));
    const auto start = lns_acceptance_key(r.m, kDefaultFeasibilityTolerance);
    int accepted = 0;
    auto previous = start;
    for (int round = 0; round < 100; ++round) {
        accepted += lns.destroy_repair(r.m, vm, rng, 0.0) ? 1 : 0;
        const auto now = lns_acceptance_key(r.m, kDefaultFeasibilityTolerance);
        REQUIRE_FALSE(previous < now);  // never worse: a rejection restores
        previous = now;
    }
    CAPTURE(start.first, start.second, previous.first, previous.second);
    REQUIRE(accepted > 0);
    REQUIRE(previous < start);
}

TEST_CASE("solve() accepts LNS repairs on a partitioned List model (#202)", "[lns][partition]") {
    Routing r(Cover::Exact);
    LNS lns;
    SearchConfig cfg;
    // FJ has nothing to jump here, so the batch cap is the budget.
    cfg.max_iterations = 400;
    cfg.batch_iterations = 20;
    cfg.perturbation_period = 2;  // kick often, so the LNS slot comes round
    cfg.structural_neighbours = neighbours_of(r);
    const SearchResult res = solve(r.m, /*time_limit=*/0.0, /*seed=*/1, true, nullptr, &lns,
                                   /*lns_interval=*/1, nullptr, cfg);
    CAPTURE(res.lns_repairs, res.lns_repairs_accepted, res.objective);
    REQUIRE(res.lns_repairs > 0);
    REQUIRE(res.lns_repairs_accepted > 0);
    for (const int seen : r.coverage()) {
        REQUIRE(seen == 1);
    }
}
