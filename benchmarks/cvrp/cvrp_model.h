#pragma once

// The CVRP model, in Hexaly's List encoding (#197,
// https://www.hexaly.com/templates/capacitated-vehicle-routing-problem-cvrp):
//
//   * `R` Lists over the customers 0..n-1 (the depot is not an element), each
//     starting empty, partitioned EXACTLY: every customer on exactly one route;
//   * per route, `lambda_sum(route, demand) <= Q`;
//   * per route, the fixed-endpoint `pair_lambda_sum` with the depot legs as
//     head and tail terms, so an empty route costs 0;
//   * minimise the sum of route lengths, in TSPLIB EUC_2D.
//
// `R` is a fleet SIZE the search may leave partly unused, not a vehicle count:
// the `k` of an X name is a minimum and the BKS may need more (X-n101-k25's
// uses 26). See default_fleet().

#include "data.h"

#include <algorithm>
#include <cbls/model.h>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cbls::cvrp {

/// The distance matrix over nodes 0..n (0 = depot), row-major. Shared by every
/// callable the model holds, so it outlives the instance it was built from and
/// is read, never written, by every portfolio worker.
struct DistanceMatrix {
    int nodes = 0;
    std::vector<double> d;

    explicit DistanceMatrix(const CvrpInstance& inst)
        : nodes(static_cast<int>(inst.x.size())),
          d(static_cast<std::size_t>(nodes) * static_cast<std::size_t>(nodes)) {
        for (int i = 0; i < nodes; ++i) {
            for (int j = 0; j < nodes; ++j) {
                d[index(i, j)] = tsplib_euc_2d(
                    inst.x[static_cast<std::size_t>(i)], inst.y[static_cast<std::size_t>(i)],
                    inst.x[static_cast<std::size_t>(j)], inst.y[static_cast<std::size_t>(j)]);
            }
        }
    }
    [[nodiscard]] std::size_t index(int i, int j) const {
        return (static_cast<std::size_t>(i) * static_cast<std::size_t>(nodes)) +
               static_cast<std::size_t>(j);
    }
    [[nodiscard]] double at(int i, int j) const { return d[index(i, j)]; }
    /// Between two customers, by List element (node = element + 1).
    [[nodiscard]] double customers(int a, int b) const { return at(a + 1, b + 1); }
    [[nodiscard]] double from_depot(int e) const { return at(0, e + 1); }
    [[nodiscard]] double to_depot(int e) const { return at(e + 1, 0); }
};

/// The default fleet size for an instance whose minimum is `k`: `k + max(3,
/// ceil(0.1 k))`. Room for the BKS's own excess over `k` (at most 6 routes on
/// the X set, X-n936-k151's 157), plus slack for the search to move through.
/// One rule for every instance; the runner records `R` on every row.
inline int default_fleet(int k) {
    return k + std::max(3, static_cast<int>(std::ceil(0.1 * k)));
}

struct CvrpModel {
    Model model;
    /// One List handle per route, in route order.
    std::vector<int32_t> routes;
    std::shared_ptr<const DistanceMatrix> dist;
};

/// Builds and closes the model. Requires `fleet >= 1`.
inline CvrpModel build_model(const CvrpInstance& inst, int fleet) {
    CvrpModel cm;
    cm.dist = std::make_shared<const DistanceMatrix>(inst);
    auto demand = std::make_shared<const std::vector<int>>(inst.demand);
    const int n = inst.customers();
    Model& m = cm.model;
    std::vector<int32_t> lengths;
    cm.routes.reserve(static_cast<std::size_t>(fleet));
    lengths.reserve(static_cast<std::size_t>(fleet));
    for (int r = 0; r < fleet; ++r) {
        cm.routes.push_back(m.list_var(n, 0, n, ListInit::Empty, "route" + std::to_string(r)));
    }
    m.add_list_partition(cm.routes, Cover::Exact);
    const std::shared_ptr<const DistanceMatrix>& dist = cm.dist;
    for (const int32_t route : cm.routes) {
        const int32_t load = m.lambda_sum(route, [demand](int e) {
            return static_cast<double>((*demand)[static_cast<std::size_t>(e) + 1]);
        });
        m.add_constraint(m.leq(load, m.constant(static_cast<double>(inst.capacity))));
        lengths.push_back(m.pair_lambda_sum(
            route, [dist](int a, int b) { return dist->customers(a, b); },
            [dist](int e) { return dist->from_depot(e); },
            [dist](int e) { return dist->to_depot(e); }));
    }
    m.minimize(m.sum(lengths));
    m.close();
    return cm;
}

/// The routes as the model holds them: per route, its customers (0-based
/// elements) in visiting order.
inline std::vector<std::vector<int>> read_routes(const CvrpModel& cm) {
    std::vector<std::vector<int>> out;
    out.reserve(cm.routes.size());
    for (const int32_t h : cm.routes) {
        const auto& el = cm.model.var(handle_to_var_id(h)).elements;
        out.emplace_back(el.begin(), el.end());
    }
    return out;
}

}  // namespace cbls::cvrp
