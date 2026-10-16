// StructuralSelection::GranularBest (#203): the granular complete neighbourhood
// of one element over a ListPartition, committed best-improvement.

#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include "cbls/move_generator.h"
#include "cbls/moves.h"
#include "cbls/randomize.h"
#include "cbls/search.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace cbls;

namespace {

/// Capacitated routing on a partition of Lists: customers on a spiral, unit
/// demand, `n_routes` routes of capacity `capacity`, minimising rounded
/// Euclidean route length from a depot at the origin.
struct Routing {
    int customers;
    Model m;
    std::vector<int32_t> route_vars;
    std::vector<double> x;
    std::vector<double> y;

    explicit Routing(Cover cover, int customers = 30, int n_routes = 5, int capacity = 8,
                     int min_len = 1)
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
            const int32_t route = m.list_var(customers, min_len, customers, ListInit::Random);
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
    [[nodiscard]] std::vector<int> coverage() const {
        std::vector<int> seen(static_cast<size_t>(customers), 0);
        for (const int32_t v : route_vars) {
            for (const int32_t e : m.var(v).elements) {
                ++seen[static_cast<size_t>(e)];
            }
        }
        return seen;
    }
    [[nodiscard]] std::shared_ptr<const NeighbourList> neighbours(int k) const {
        return std::make_shared<const NeighbourList>(
            nearest_neighbours(customers, k, [this](int a, int b) { return dist(a, b); }));
    }
};

/// Whether `a` and `b` sit next to each other in some route.
bool adjacent(const Routing& r, int32_t a, int32_t b) {
    for (const int32_t v : r.route_vars) {
        const std::vector<int32_t>& el = r.m.var(v).elements;
        for (size_t p = 0; p + 1 < el.size(); ++p) {
            if ((el[p] == a && el[p + 1] == b) || (el[p] == b && el[p + 1] == a)) {
                return true;
            }
        }
    }
    return false;
}

/// Apply every candidate of many draws and check what each leaves behind,
/// putting the assignment back after each one.
void check_draws(Routing& r, int k, int draws, std::set<std::string>& kinds) {
    const auto nl = r.neighbours(k);
    RNG rng(5);
    std::vector<int64_t> scratch;
    for (int d = 0; d < draws; ++d) {
        std::vector<Move> out;
        generate_granular_partition_moves(r.m, 0, rng, *nl, out, scratch);
        REQUIRE(out.size() <= static_cast<size_t>((5 * k) + 1) + r.route_vars.size());
        const std::vector<int> start = r.coverage();
        for (const Move& move : out) {
            REQUIRE_FALSE(move.changes.empty());
            kinds.insert(move.move_type);
            const SavedValues saved = save_move_values(r.m, move);
            apply_move(r.m, move);
            const std::vector<int> seen = r.coverage();
            for (size_t e = 0; e < seen.size(); ++e) {
                if (r.m.list_partitions().front().cover == Cover::Exact) {
                    REQUIRE(seen[e] == 1);
                } else {
                    REQUIRE(seen[e] <= 1);
                    // Only an insert adds, only a remove drops, one element each.
                    if (move.move_type == "g_insert") {
                        REQUIRE(seen[e] >= start[e]);
                    } else if (move.move_type == "g_remove") {
                        REQUIRE(seen[e] <= start[e]);
                    } else {
                        REQUIRE(seen[e] == start[e]);
                    }
                }
            }
            for (const int32_t v : r.route_vars) {
                const auto size = static_cast<int32_t>(r.m.var(v).elements.size());
                REQUIRE(size >= r.m.var(v).min_size);
                REQUIRE(size <= r.m.var(v).max_size);
            }
            undo_move(r.m, move, saved);
        }
    }
}

}  // namespace

TEST_CASE("granular moves keep an Exact cover and the length bounds (#203)",
          "[structural][partition]") {
    Routing r(Cover::Exact);
    RNG rng(3);
    randomize_list_partition(r.m, r.m.list_partitions().front(), rng);
    std::set<std::string> kinds;
    check_draws(r, 6, 200, kinds);
    // Every pair kind shows up over enough draws.
    for (const char* kind : {"g_relocate", "g_swap", "g_2opt", "g_2opt_star"}) {
        CAPTURE(kind);
        REQUIRE(kinds.count(kind) == 1);
    }
    REQUIRE(kinds.count("g_insert") == 0);
    REQUIRE(kinds.count("g_remove") == 0);
}

TEST_CASE("granular moves insert and remove under AtMostOnce (#203)", "[structural][partition]") {
    Routing r(Cover::AtMostOnce);
    RNG rng(3);
    randomize_list_partition(r.m, r.m.list_partitions().front(), rng);
    // Leave a third of the customers unserved, so the insert branch is reachable.
    for (const int32_t v : r.route_vars) {
        auto& el = r.m.var_mut(v).elements;
        el.resize(std::max<size_t>(1, (2 * el.size()) / 3));
    }
    std::set<std::string> kinds;
    check_draws(r, 6, 200, kinds);
    REQUIRE(kinds.count("g_insert") == 1);
    REQUIRE(kinds.count("g_remove") == 1);
}

TEST_CASE("every granular relocate and 2-opt puts the drawn element next to a neighbour (#203)",
          "[structural][partition]") {
    // The neighbourhood is defined by the arc (e, f) it creates. The element is
    // the call's first draw, so a copy of the RNG names it. No list is empty
    // (min length 1), so every relocate lands next to some f.
    Routing r(Cover::Exact);
    RNG layout(9);
    randomize_list_partition(r.m, r.m.list_partitions().front(), layout);
    const auto nl = r.neighbours(5);
    RNG rng(11);
    std::vector<int64_t> scratch;
    int checked = 0;
    for (int d = 0; d < 100; ++d) {
        RNG probe = rng;
        const auto e = static_cast<int32_t>(probe.integers(0, r.customers));
        std::vector<Move> out;
        generate_granular_partition_moves(r.m, 0, rng, *nl, out, scratch);
        for (const Move& move : out) {
            if (move.move_type == "g_swap") {
                continue;  // a swap exchanges places; it creates no fixed arc
            }
            const SavedValues saved = save_move_values(r.m, move);
            apply_move(r.m, move);
            bool any = false;
            for (const int32_t f : nl->of(e)) {
                any = any || adjacent(r, e, f);
            }
            CAPTURE(d, e, move.move_type);
            REQUIRE(any);
            ++checked;
            undo_move(r.m, move, saved);
        }
    }
    REQUIRE(checked > 100);
}

TEST_CASE("granular moves reopen an empty list (#203)", "[structural][partition]") {
    for (const Cover cover : {Cover::Exact, Cover::AtMostOnce}) {
        Routing r(cover, 30, 5, 8, /*min_len=*/0);
        RNG layout(2);
        randomize_list_partition(r.m, r.m.list_partitions().front(), layout);
        // Empty the last route into the first.
        auto& first = r.m.var_mut(r.route_vars.front()).elements;
        auto& last = r.m.var_mut(r.route_vars.back()).elements;
        first.insert(first.end(), last.begin(), last.end());
        last.clear();
        const auto nl = r.neighbours(5);
        RNG rng(4);
        std::vector<int64_t> scratch;
        int reopened = 0;
        for (int d = 0; d < 50; ++d) {
            std::vector<Move> out;
            generate_granular_partition_moves(r.m, 0, rng, *nl, out, scratch);
            for (const Move& move : out) {
                for (const Move::Change& c : move.changes) {
                    reopened += c.var_id == r.route_vars.back() ? 1 : 0;
                }
            }
        }
        CAPTURE(static_cast<int>(cover));
        REQUIRE(reopened > 0);
        // ... and every candidate, the reopening ones included, keeps the cover.
        std::set<std::string> kinds;
        check_draws(r, 5, 50, kinds);
    }
}

TEST_CASE("solve() under GranularBest keeps the cover and reaches the engine (#203)",
          "[structural][partition]") {
    std::array<double, 2> objective{};
    int i = 0;
    for (const StructuralSelection sel :
         {StructuralSelection::ViolationGuided, StructuralSelection::GranularBest}) {
        Routing r(Cover::Exact);
        SearchConfig cfg;
        cfg.max_iterations = 300;
        cfg.batch_iterations = 20;
        cfg.structural_batch_probability = 1.0;
        cfg.structural_selection = sel;
        cfg.structural_neighbours = r.neighbours(8);
        const SearchResult res =
            solve(r.m, /*time_limit=*/0.0, /*seed=*/1, true, nullptr, nullptr, 0, nullptr, cfg);
        REQUIRE(res.feasible);
        for (const int seen : r.coverage()) {
            REQUIRE(seen == 1);
        }
        objective.at(static_cast<size_t>(i++)) = res.objective;
    }
    CAPTURE(objective[0], objective[1]);
    // The selection reaches the engine: the two runs differ.
    REQUIRE(objective[0] != objective[1]);
}

TEST_CASE("GranularBest leaves a scalar model's trajectory alone (#203)", "[structural]") {
    std::array<std::vector<double>, 2> values{};
    std::array<double, 2> objective{};
    int i = 0;
    for (const StructuralSelection sel :
         {StructuralSelection::FirstImprovingSample, StructuralSelection::GranularBest}) {
        Model m;
        std::vector<int32_t> xs;
        xs.reserve(6);
        for (int v = 0; v < 6; ++v) {
            xs.push_back(m.int_var(0, 10));
        }
        m.add_constraint(m.geq(m.sum(xs), m.constant(20.0)));
        m.add_constraint(m.leq(m.sum({xs[0], xs[1]}), m.constant(5.0)));
        std::vector<int32_t> weighted;
        weighted.reserve(6);
        for (int v = 0; v < 6; ++v) {
            weighted.push_back(m.prod(m.constant(1.0 + v), xs[static_cast<size_t>(v)]));
        }
        m.minimize(m.sum(weighted));
        m.close();
        SearchConfig cfg;
        cfg.max_iterations = 200;
        cfg.structural_selection = sel;
        const SearchResult res =
            solve(m, /*time_limit=*/0.0, /*seed=*/4, true, nullptr, nullptr, 0, nullptr, cfg);
        objective.at(static_cast<size_t>(i)) = res.objective;
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            values.at(static_cast<size_t>(i)).push_back(m.var(v).value);
        }
        ++i;
    }
    REQUIRE(objective[0] == objective[1]);
    REQUIRE(values[0] == values[1]);
}
