#pragma once

// The in-runner CVRP check (#197). It does not trust the model: it takes the
// routes as plain customer lists and recomputes everything from the parsed
// instance -- coordinates, demands, capacity -- with its own distance, not the
// model's matrix. So a model builder that wired a wrong row, a wrong lambda or
// a wrong endpoint term is caught here rather than reported as a result.
//
// What it cannot catch is a reader bug, which both sides would share; the
// reader tests and the `.vrp` hash in benchmarks/instances/cvrp/manifest.csv
// cover that, and the independent Python verifier (#199) reads the instance
// file itself.

#include "data.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace cbls::cvrp {

struct RouteCheck {
    bool ok = false;
    std::vector<std::string> errors;
    double cost = 0.0;    // recomputed from the instance
    int routes_used = 0;  // non-empty routes
};

/// Checks `routes` (per route, 0-based customers in visiting order) against
/// `inst`: every customer exactly once, every load within capacity, and the
/// recomputed cost equal to `reported_objective`. The fleet is NOT capped at the
/// name's `k`, which is a minimum.
inline RouteCheck check_routes(const CvrpInstance& inst,
                               const std::vector<std::vector<int>>& routes,
                               double reported_objective) {
    RouteCheck rc;
    const int n = inst.customers();
    std::vector<int> seen(static_cast<std::size_t>(n), 0);
    for (std::size_t r = 0; r < routes.size(); ++r) {
        const auto& route = routes[r];
        if (route.empty()) {
            continue;
        }
        ++rc.routes_used;
        long long load = 0;
        int prev = 0;  // node index; 0 is the depot
        for (const int e : route) {
            if (e < 0 || e >= n) {
                rc.errors.push_back("route " + std::to_string(r) + ": customer " +
                                    std::to_string(e) + " out of range");
                continue;
            }
            ++seen[static_cast<std::size_t>(e)];
            const int node = e + 1;
            load += inst.demand[static_cast<std::size_t>(node)];
            rc.cost += tsplib_euc_2d(
                inst.x[static_cast<std::size_t>(prev)], inst.y[static_cast<std::size_t>(prev)],
                inst.x[static_cast<std::size_t>(node)], inst.y[static_cast<std::size_t>(node)]);
            prev = node;
        }
        rc.cost += tsplib_euc_2d(inst.x[static_cast<std::size_t>(prev)],
                                 inst.y[static_cast<std::size_t>(prev)], inst.x[0], inst.y[0]);
        if (load > inst.capacity) {
            rc.errors.push_back("route " + std::to_string(r) + ": load " + std::to_string(load) +
                                " exceeds capacity " + std::to_string(inst.capacity));
        }
    }
    for (int e = 0; e < n; ++e) {
        const int c = seen[static_cast<std::size_t>(e)];
        if (c != 1) {
            rc.errors.push_back(
                "customer " + std::to_string(e) +
                (c == 0 ? " is on no route" : " is on " + std::to_string(c) + " route positions"));
        }
    }
    // Every distance is an integer, so the sums are exact in a double and a
    // correct objective agrees to the last bit; the tolerance only absorbs the
    // order the engine summed in.
    if (!(std::abs(rc.cost - reported_objective) <= 1e-6 * std::max(1.0, std::abs(rc.cost)))) {
        rc.errors.push_back("recomputed cost " + std::to_string(rc.cost) +
                            " != reported objective " + std::to_string(reported_objective));
    }
    rc.ok = rc.errors.empty();
    return rc;
}

}  // namespace cbls::cvrp
