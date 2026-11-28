// Model a CVRP with List variables, solve it and read the routes
// (docs/examples/cvrp.md).
//
// Usage: build/cbls_example_cvrp benchmarks/instances/cvrp/X/X-n115-k10.vrp
#include <cbls/cbls.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

/// Node 0 is the depot; customer element e (0-based) is node e + 1.
struct Instance {
    std::string name;
    int capacity = 0;
    std::vector<std::pair<double, double>> coords;
    std::vector<int> demands;

    [[nodiscard]] int customers() const { return static_cast<int>(demands.size()) - 1; }
};

/// A header line `KEY : value`; `weight` takes EDGE_WEIGHT_TYPE.
void read_header_line(const std::string& line, Instance& inst, std::string& weight) {
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
        return;
    }
    std::istringstream key_stream(line.substr(0, colon));
    std::istringstream value(line.substr(colon + 1));
    std::string key;
    key_stream >> key;
    if (key == "NAME") {
        value >> inst.name;
    } else if (key == "CAPACITY") {
        value >> inst.capacity;
    } else if (key == "EDGE_WEIGHT_TYPE") {
        value >> weight;
    }
}

/// One `id x y` or `id demand` line, its id already read from `ss`; false
/// when the rest does not parse.
bool read_section_line(const std::string& section, std::istringstream& ss, Instance& inst) {
    if (section == "NODE_COORD_SECTION") {
        double x = 0.0;
        double y = 0.0;
        if (!(ss >> x >> y)) {
            return false;
        }
        inst.coords.emplace_back(x, y);
    } else if (section == "DEMAND_SECTION") {
        int d = 0;
        if (!(ss >> d)) {
            return false;
        }
        inst.demands.push_back(d);
    }
    return true;
}

/// A TSPLIB CVRP file as CVRPLIB's X set writes it: EUC_2D, depot at node 1.
Instance read_vrp(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("cannot open " + path);
    }
    Instance inst;
    std::string weight;
    std::string section;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string first;
        if (!(ss >> first)) {
            continue;
        }
        if (first == "NODE_COORD_SECTION" || first == "DEMAND_SECTION" ||
            first == "DEPOT_SECTION" || first == "EOF") {
            section = first;
        } else if (section.empty()) {
            read_header_line(line, inst, weight);
        } else if (!read_section_line(section, ss, inst)) {
            throw std::runtime_error("a coordinate or demand line does not parse in " + path);
        }
    }
    if (weight != "EUC_2D" || inst.capacity <= 0 || inst.demands.empty() || inst.demands[0] != 0 ||
        inst.coords.size() != inst.demands.size()) {
        throw std::runtime_error(
            path + ": expected EUC_2D distances, a positive capacity and the depot at node 1");
    }
    return inst;
}

/// TSPLIB's EUC_2D: the Euclidean distance rounded to the nearest integer.
std::vector<std::vector<double>> distances(const Instance& inst) {
    std::vector<std::vector<double>> d(inst.coords.size(), std::vector<double>(inst.coords.size()));
    for (std::size_t a = 0; a < inst.coords.size(); ++a) {
        for (std::size_t b = 0; b < inst.coords.size(); ++b) {
            d[a][b] = std::floor(std::hypot(inst.coords[a].first - inst.coords[b].first,
                                            inst.coords[a].second - inst.coords[b].second) +
                                 0.5);
        }
    }
    return d;
}

using Distances = std::shared_ptr<const std::vector<std::vector<double>>>;

/// `fleet` routes partitioning the customers, each within capacity, total
/// length minimised. Returns the route handles.
std::vector<int32_t> build(cbls::Model& m, const Instance& inst, const Distances& dist, int fleet) {
    const int n = inst.customers();
    // One List per route: distinct customers 0..n-1 in visiting order.
    std::vector<int32_t> routes;
    routes.reserve(static_cast<std::size_t>(fleet));
    for (int r = 0; r < fleet; ++r) {
        routes.push_back(m.list_var(n, 0, n, cbls::ListInit::Empty, "route" + std::to_string(r)));
    }
    // Every customer on exactly one route: the search starts from a random
    // complete partition and its moves keep it so.
    m.add_list_partition(routes, cbls::Cover::Exact);
    auto demands = std::make_shared<const std::vector<int>>(inst.demands);
    const int32_t capacity = m.constant(static_cast<double>(inst.capacity));
    std::vector<int32_t> lengths;
    for (const int32_t route : routes) {
        const int32_t load = m.lambda_sum(route, [demands](int e) {
            return static_cast<double>((*demands)[static_cast<std::size_t>(e) + 1]);
        });
        m.add_constraint(m.leq(load, capacity));
        // The legs between consecutive customers, plus the depot legs as head
        // and tail; an empty route costs 0.
        lengths.push_back(m.pair_lambda_sum(
            route,
            [dist](int a, int b) {
                return (*dist)[static_cast<std::size_t>(a) + 1][static_cast<std::size_t>(b) + 1];
            },
            [dist](int e) { return (*dist)[0][static_cast<std::size_t>(e) + 1]; },
            [dist](int e) { return (*dist)[static_cast<std::size_t>(e) + 1][0]; }));
    }
    m.minimize(m.sum(lengths));
    m.close();
    return routes;
}

/// Each customer on one route, each route within capacity, and the cost as stated.
std::vector<std::string> verify(const Instance& inst, const std::vector<std::vector<double>>& d,
                                const std::vector<std::vector<int>>& routes, double cost) {
    std::vector<std::string> errors;
    std::vector<int> seen(inst.demands.size(), 0);
    double length = 0.0;
    for (std::size_t r = 0; r < routes.size(); ++r) {
        int load = 0;
        std::size_t prev = 0;
        for (const int c : routes[r]) {
            const auto node = static_cast<std::size_t>(c);
            ++seen[node];
            load += inst.demands[node];
            length += d[prev][node];
            prev = node;
        }
        length += d[prev][0];
        if (load > inst.capacity) {
            errors.push_back("route " + std::to_string(r + 1) + " carries " + std::to_string(load) +
                             " > " + std::to_string(inst.capacity));
        }
    }
    for (std::size_t c = 1; c < seen.size(); ++c) {
        if (seen[c] != 1) {
            errors.push_back("customer " + std::to_string(c) + " is visited " +
                             std::to_string(seen[c]) + " times");
        }
    }
    if (length != cost) {
        errors.push_back("the routes are " + std::to_string(length) + " long, the solve reported " +
                         std::to_string(cost));
    }
    return errors;
}

int run(const char* path) {
    const Instance inst = read_vrp(path);
    auto dist = std::make_shared<const std::vector<std::vector<double>>>(distances(inst));
    const int n = inst.customers();
    // The fewest vehicles the total demand needs, plus room to spare: the
    // search may leave routes empty.
    const int total = std::accumulate(inst.demands.begin(), inst.demands.end(), 0);
    const int fleet = ((total + inst.capacity - 1) / inst.capacity) + 3;
    cbls::Model model;
    const std::vector<int32_t> routes = build(model, inst, dist, fleet);
    std::printf("%s: %d customers, capacity %d, fleet %d\n", inst.name.c_str(), n, inst.capacity,
                fleet);

    cbls::SearchConfig config;
    // Moves relocate a customer next to one of its 20 nearest customers.
    config.structural_neighbours = std::make_shared<const cbls::NeighbourList>(
        cbls::nearest_neighbours(n, 20, [&dist](int a, int b) {
            return (*dist)[static_cast<std::size_t>(a) + 1][static_cast<std::size_t>(b) + 1];
        }));
    cbls::LNS lns(/*destroy_fraction=*/0.2);
    const cbls::SearchResult result = cbls::solve(model, /*time_limit=*/5.0, /*seed=*/1,
                                                  /*hook=*/nullptr, &lns, /*lns_interval=*/3,
                                                  /*callback=*/nullptr, config);

    std::printf("feasible  = %s\n", result.feasible ? "true" : "false");
    std::printf("objective = %.0f\n", result.objective);
    // The solve leaves the model at the best point found. A List's elements
    // are its customers in visiting order, printed here as node numbers
    // (CVRPLIB's .sol numbering).
    std::vector<std::vector<int>> visited;
    for (const int32_t h : routes) {
        std::vector<int> route;
        for (const int e : model.var(cbls::handle_to_var_id(h)).elements) {
            route.push_back(e + 1);
        }
        if (!route.empty()) {
            visited.push_back(route);
        }
    }
    for (std::size_t r = 0; r < visited.size(); ++r) {
        std::printf("Route #%zu:", r + 1);
        for (const int c : visited[r]) {
            std::printf(" %d", c);
        }
        std::printf("\n");
    }
    const std::vector<std::string> errors = verify(inst, *dist, visited, result.objective);
    if (errors.empty()) {
        std::printf("verify: PASS\n");
    } else {
        std::printf("verify: FAIL (%zu errors)\n", errors.size());
    }
    for (const std::string& error : errors) {
        std::printf("  %s\n", error.c_str());
    }
    return result.feasible && errors.empty() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s INSTANCE.vrp\n", argv[0]);
        return 2;
    }
    try {  // a reader error (missing file, bad syntax) throws
        return run(argv[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
