// Tests for the CVRP benchmark (#197): the `.vrp` reader, the in-runner route
// verifier, the model builder, and a tiny instance solved to its proven optimum.
//
// The solve is iteration-budgeted with the wall clock off (time_limit 0), so it
// is bit-reproducible for its seed and cannot flake under `ctest -j` load.

#include <algorithm>
#include <benchmarks/cvrp/cvrp_model.h>
#include <benchmarks/cvrp/data.h>
#include <benchmarks/cvrp/verify_cvrp.h>
#include <catch2/catch_test_macros.hpp>
#include <cbls/dag_ops.h>
#include <cbls/lns.h>
#include <cbls/search.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using cbls::cvrp::CvrpInstance;

namespace {

/// Seven customers in two clusters on either side of the depot, capacity 10,
/// demands summing to 21: at least three routes, and the clusters make the
/// optimum non-trivial to stumble on.
const char* const kTinyVrp =
    "NAME : tiny-n8-k3\n"
    "COMMENT : hand-built for tests/test_cvrp.cpp\n"
    "TYPE : CVRP\n"
    "DIMENSION : 8\n"
    "EDGE_WEIGHT_TYPE : EUC_2D\n"
    "CAPACITY : 10\n"
    "NODE_COORD_SECTION\n"
    "1 50 50\n"
    "2 10 20\n"
    "3 15 35\n"
    "4 5 45\n"
    "5 90 80\n"
    "6 80 95\n"
    "7 95 60\n"
    "8 50 5\n"
    "DEMAND_SECTION\n"
    "1 0\n"
    "2 3\n"
    "3 4\n"
    "4 2\n"
    "5 5\n"
    "6 3\n"
    "7 1\n"
    "8 3\n"
    "DEPOT_SECTION\n"
    "1\n"
    "-1\n"
    "EOF\n";

CvrpInstance parse(const std::string& text) {
    std::istringstream in(text);
    return cbls::cvrp::read_vrp(in);
}

CvrpInstance tiny() {
    return parse(kTinyVrp);
}

/// `kTinyVrp` with one substring replaced, for the refusal cases.
std::string tiny_with(const std::string& from, const std::string& to) {
    std::string s = kTinyVrp;
    const auto pos = s.find(from);
    REQUIRE(pos != std::string::npos);
    s.replace(pos, from.size(), to);
    return s;
}

/// The optimum by exhaustion: every customer order, cut into consecutive routes
/// at every subset of the n-1 gaps. That enumerates every solution with any
/// number of routes, so it does not depend on the model or the fleet size.
double brute_force_optimum(const CvrpInstance& inst) {
    const int n = inst.customers();
    std::vector<int> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    double best = std::numeric_limits<double>::infinity();
    do {
        for (unsigned cuts = 0; cuts < (1U << static_cast<unsigned>(n - 1)); ++cuts) {
            std::vector<std::vector<int>> routes(1);
            for (int i = 0; i < n; ++i) {
                if (i > 0 && ((cuts >> static_cast<unsigned>(i - 1)) & 1U) != 0U) {
                    routes.emplace_back();
                }
                routes.back().push_back(order[static_cast<std::size_t>(i)]);
            }
            const auto rc = cbls::cvrp::check_routes(inst, routes, 0.0);
            // check_routes reports the cost mismatch against 0 as an error; any
            // other error (an overload) makes the split infeasible.
            if (rc.errors.size() == 1) {
                best = std::min(best, rc.cost);
            }
        }
    } while (std::next_permutation(order.begin(), order.end()));
    return best;
}

bool mentions(const cbls::cvrp::RouteCheck& rc, const std::string& what) {
    return std::any_of(rc.errors.begin(), rc.errors.end(),
                       [&what](const std::string& e) { return e.find(what) != std::string::npos; });
}

}  // namespace

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

TEST_CASE("cvrp reader parses a TSPLIB CVRP file", "[cvrp]") {
    const CvrpInstance inst = tiny();
    REQUIRE(inst.name == "tiny-n8-k3");
    REQUIRE(inst.customers() == 7);
    REQUIRE(inst.capacity == 10);
    REQUIRE(inst.demand[0] == 0);
    REQUIRE(inst.demand[7] == 3);
    REQUIRE(inst.x[4] == 90);
    REQUIRE(inst.y[7] == 5);
    REQUIRE(cbls::cvrp::name_k(inst.name) == 3);
    REQUIRE(cbls::cvrp::name_k("X-n101-k25") == 25);
    REQUIRE(cbls::cvrp::name_k("no-k-here") == -1);
}

TEST_CASE("cvrp reader refuses what the BKS metric does not cover", "[cvrp]") {
    const std::vector<std::pair<std::string, std::string>> refusals = {
        {"TYPE : CVRP", "TYPE : TSP"},
        {"EDGE_WEIGHT_TYPE : EUC_2D", "EDGE_WEIGHT_TYPE : ATT"},
        {"1\n-1\n", "1\n2\n-1\n"},                     // two depots
        {"DEPOT_SECTION\n1\n", "DEPOT_SECTION\n2\n"},  // depot not node 1
        {"8 3\nDEPOT", "DEPOT"},                       // a node with no demand
        {"8 50 5\n", ""},                              // a node with no coordinates
        {"5 5\n", "5 11\n"},                           // one customer over capacity
        {"1 0\n2 3", "1 4\n2 3"},                      // a depot with demand
        {"CAPACITY : 10", "CAPACITY : 0"},
        {"2 10 20", "9 10 20"},  // node id out of range
    };
    for (const auto& [from, to] : refusals) {
        INFO(from << " -> " << to);
        REQUIRE_THROWS_AS(parse(tiny_with(from, to)), std::runtime_error);
    }
    REQUIRE_THROWS_AS(parse("NAME : empty\nTYPE : CVRP\n"), std::runtime_error);
}

TEST_CASE("cvrp distance is TSPLIB nint, not banker's rounding", "[cvrp]") {
    REQUIRE(cbls::cvrp::tsplib_euc_2d(0, 0, 3, 4) == 5.0);
    REQUIRE(cbls::cvrp::tsplib_euc_2d(0, 0, 1, 1) == 1.0);    // 1.414 -> 1
    REQUIRE(cbls::cvrp::tsplib_euc_2d(0, 0, 1, 2) == 2.0);    // 2.236 -> 2
    REQUIRE(cbls::cvrp::tsplib_euc_2d(0, 0, 2.5, 0) == 3.0);  // a half rounds up
}

// ---------------------------------------------------------------------------
// Verifier
// ---------------------------------------------------------------------------

TEST_CASE("cvrp verifier accepts a valid solution and its cost", "[cvrp]") {
    const CvrpInstance inst = tiny();
    const std::vector<std::vector<int>> routes = {{0, 1, 2}, {3, 4, 5}, {}, {6}};
    const auto first = cbls::cvrp::check_routes(inst, routes, 0.0);
    const auto rc = cbls::cvrp::check_routes(inst, routes, first.cost);
    REQUIRE(rc.ok);
    REQUIRE(rc.routes_used == 3);  // the empty route is not a route
    REQUIRE(rc.cost > 0.0);
}

TEST_CASE("cvrp verifier rejects a duplicated, missing or overloaded customer", "[cvrp]") {
    const CvrpInstance inst = tiny();
    auto cost_of = [&inst](const std::vector<std::vector<int>>& r) {
        return cbls::cvrp::check_routes(inst, r, 0.0).cost;
    };
    SECTION("duplicated") {
        const std::vector<std::vector<int>> r = {{0, 1, 2}, {3, 4, 5}, {6, 0}};
        const auto rc = cbls::cvrp::check_routes(inst, r, cost_of(r));
        REQUIRE_FALSE(rc.ok);
        REQUIRE(mentions(rc, "customer 0 is on 2"));
    }
    SECTION("missing") {
        const std::vector<std::vector<int>> r = {{0, 1, 2}, {3, 4, 5}};
        const auto rc = cbls::cvrp::check_routes(inst, r, cost_of(r));
        REQUIRE_FALSE(rc.ok);
        REQUIRE(mentions(rc, "customer 6 is on no route"));
    }
    SECTION("overloaded") {
        // 3 + 4 + 2 + 3 = 12 > 10.
        const std::vector<std::vector<int>> r = {{0, 1, 2, 6}, {3, 4, 5}};
        const auto rc = cbls::cvrp::check_routes(inst, r, cost_of(r));
        REQUIRE_FALSE(rc.ok);
        REQUIRE(mentions(rc, "exceeds capacity"));
    }
    SECTION("out of range") {
        const std::vector<std::vector<int>> r = {{0, 1, 2}, {3, 4, 5}, {6, 7}};
        const auto rc = cbls::cvrp::check_routes(inst, r, cost_of(r));
        REQUIRE_FALSE(rc.ok);
        REQUIRE(mentions(rc, "out of range"));
    }
    SECTION("wrong cost") {
        const std::vector<std::vector<int>> r = {{0, 1, 2}, {3, 4, 5}, {6}};
        const auto rc = cbls::cvrp::check_routes(inst, r, cost_of(r) - 1.0);
        REQUIRE_FALSE(rc.ok);
        REQUIRE(mentions(rc, "recomputed cost"));
    }
}

// ---------------------------------------------------------------------------
// Model and solve
// ---------------------------------------------------------------------------

TEST_CASE("cvrp default fleet leaves room above the name's minimum", "[cvrp]") {
    REQUIRE(cbls::cvrp::default_fleet(3) == 6);
    REQUIRE(cbls::cvrp::default_fleet(25) == 28);
    REQUIRE(cbls::cvrp::default_fleet(151) == 167);  // X-n936-k151's BKS uses 157
}

TEST_CASE("cvrp model's objective is the verifier's cost, empty routes included", "[cvrp]") {
    // Pins the builder against the independent recomputation: the endpoint
    // terms, an empty route costing 0, and the capacity rows reading demand.
    const CvrpInstance inst = tiny();
    cbls::cvrp::CvrpModel cm = cbls::cvrp::build_model(inst, 4);
    const std::vector<std::vector<int>> routes = {{2, 0, 1}, {}, {5, 3, 4}, {6}};
    for (std::size_t r = 0; r < routes.size(); ++r) {
        auto& el = cm.model.var_mut(cbls::handle_to_var_id(cm.routes[r])).elements;
        el.assign(routes[r].begin(), routes[r].end());
    }
    cbls::full_evaluate(cm.model);
    const double objective = cm.model.node_value(cm.model.objective_id());
    REQUIRE(cbls::cvrp::read_routes(cm) == routes);
    REQUIRE(cbls::cvrp::check_routes(inst, routes, objective).ok);
}

TEST_CASE("cvrp tiny instance is solved to its proven optimum", "[cvrp]") {
    const CvrpInstance inst = tiny();
    const double optimum = brute_force_optimum(inst);
    REQUIRE(std::isfinite(optimum));

    cbls::cvrp::CvrpModel cm = cbls::cvrp::build_model(inst, cbls::cvrp::default_fleet(3));
    cbls::SearchConfig cfg;
    cfg.max_iterations = 3000;
    cbls::LNS lns;
    // time_limit 0 switches the wall clock off: an iteration budget alone, so
    // the run is the same on every machine and under any load.
    const cbls::SearchResult result = cbls::solve(cm.model, /*time_limit=*/0.0, /*seed=*/7,
                                                  /*use_fj=*/true, nullptr, &lns, 3, nullptr, cfg);
    REQUIRE(result.feasible);
    cm.model.restore_state(result.best_state);
    cbls::full_evaluate(cm.model);
    const auto routes = cbls::cvrp::read_routes(cm);
    const auto rc = cbls::cvrp::check_routes(inst, routes, result.objective);
    INFO("optimum " << optimum << ", found " << result.objective);
    REQUIRE(rc.ok);
    REQUIRE(result.objective == optimum);
}
