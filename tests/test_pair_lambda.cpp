// The PairLambda variants: cyclic closing and fixed endpoints (#163).
//
// Three things are pinned here that nothing else pins: the VALUE of each
// variant against a hand computation (short lists included, since a
// variable-length List makes n = 0, 1, 2 routine rather than exotic), the
// agreement between incremental and from-scratch evaluation after every list
// move generator, and the `.cbls` round-trip -- where an open chain must still
// serialise to the bytes it always did.

#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace cbls;

namespace {

// Asymmetric on purpose: a symmetric cost cannot distinguish a cyclic sum from
// twice an open one on n = 2, nor a tour from its reverse.
double d(int a, int b) {
    return (10.0 * a) + b;
}

double head_cost(int e) {
    return 100.0 + e;
}

double tail_cost(int e) {
    return 1000.0 + e;
}

// Build a one-node model over a List of `capacity` positions and evaluate it at
// `elements`. `elements` may be shorter than the list's capacity: that is how
// the n = 0, 1, 2 cases are reached without a separate model each.
double value_at(int capacity, const std::vector<int32_t>& elements, PairMode mode, bool endpoints) {
    Model m;
    auto lv = m.list_var(capacity, "seq");
    int32_t node = endpoints ? m.pair_lambda_sum(lv, d, head_cost, tail_cost, mode)
                             : m.pair_lambda_sum(lv, d, mode);
    m.minimize(node);
    m.close();
    m.var_mut(vid(lv)).elements = elements;
    full_evaluate(m);
    return m.node_value(node);
}

}  // namespace

TEST_CASE("pair_lambda_sum values for n = 0, 1, 2 and a general n", "[dag][pair_lambda]") {
    SECTION("n = 0 is zero for every variant, endpoints included") {
        REQUIRE(value_at(4, {}, PairMode::Open, false) == 0.0);
        REQUIRE(value_at(4, {}, PairMode::Cyclic, false) == 0.0);
        REQUIRE(value_at(4, {}, PairMode::Open, true) == 0.0);
        REQUIRE(value_at(4, {}, PairMode::Cyclic, true) == 0.0);
    }

    SECTION("n = 1 has no pair, and Cyclic adds none") {
        REQUIRE(value_at(4, {3}, PairMode::Open, false) == 0.0);
        REQUIRE(value_at(4, {3}, PairMode::Cyclic, false) == 0.0);
        // head(3) + tail(3) = 103 + 1003
        REQUIRE(value_at(4, {3}, PairMode::Open, true) == 1106.0);
        REQUIRE(value_at(4, {3}, PairMode::Cyclic, true) == 1106.0);
    }

    SECTION("n = 2 traverses its one edge twice when cyclic") {
        REQUIRE(value_at(4, {1, 2}, PairMode::Open, false) == 12.0);           // d(1,2)
        REQUIRE(value_at(4, {1, 2}, PairMode::Cyclic, false) == 12.0 + 21.0);  // + d(2,1)
        // d(1,2) + head(1) + tail(2)
        REQUIRE(value_at(4, {1, 2}, PairMode::Open, true) == 12.0 + 101.0 + 1002.0);
        REQUIRE(value_at(4, {1, 2}, PairMode::Cyclic, true) == 12.0 + 21.0 + 101.0 + 1002.0);
    }

    SECTION("general n") {
        const std::vector<int32_t> e = {2, 0, 3, 1};
        // d(2,0) + d(0,3) + d(3,1) = 20 + 3 + 31 = 54
        REQUIRE(value_at(4, e, PairMode::Open, false) == 54.0);
        REQUIRE(value_at(4, e, PairMode::Cyclic, false) == 54.0 + 12.0);  // + d(1,2)
        // + head(2) + tail(1) = 102 + 1001
        REQUIRE(value_at(4, e, PairMode::Open, true) == 54.0 + 102.0 + 1001.0);
        REQUIRE(value_at(4, e, PairMode::Cyclic, true) == 54.0 + 12.0 + 102.0 + 1001.0);
    }

    SECTION("Open is the default, so the pre-existing two-argument call is unchanged") {
        Model m;
        auto lv = m.list_var(4);
        int32_t node = m.pair_lambda_sum(lv, d);
        m.minimize(node);
        m.close();
        m.var_mut(vid(lv)).elements = {2, 0, 3, 1};
        full_evaluate(m);
        REQUIRE(m.node_value(node) == 54.0);
    }

    SECTION("an empty head or tail callable means no term") {
        Model m;
        auto lv = m.list_var(4);
        int32_t node = m.pair_lambda_sum(lv, d, head_cost, nullptr);
        m.minimize(node);
        m.close();
        m.var_mut(vid(lv)).elements = {2, 0, 3, 1};
        full_evaluate(m);
        REQUIRE(m.node_value(node) == 54.0 + 102.0);
    }
}

TEST_CASE("every list move keeps delta evaluation equal to a fresh full evaluation",
          "[dag][pair_lambda][moves]") {
    // One node per variant in ONE model, so a single move sequence exercises
    // all four and the fresh-copy comparison covers them together.
    Model m;
    auto lv = m.list_var(7, "seq");
    const int32_t open_node = m.pair_lambda_sum(lv, d);
    const int32_t cyclic_node = m.pair_lambda_sum(lv, d, PairMode::Cyclic);
    const int32_t open_ends = m.pair_lambda_sum(lv, d, head_cost, tail_cost);
    const int32_t cyclic_ends = m.pair_lambda_sum(lv, d, head_cost, tail_cost, PairMode::Cyclic);
    m.minimize(m.sum({open_node, cyclic_node, open_ends, cyclic_ends}));
    m.close();
    full_evaluate(m);

    RNG rng(1234);
    std::vector<std::string> seen;
    for (int round = 0; round < 60; ++round) {
        auto moves = generate_standard_moves(m.var(vid(lv)), rng);
        REQUIRE_FALSE(moves.empty());
        for (const Move& mv : moves) {
            seen.push_back(mv.move_type);
            std::vector<int32_t> changed = apply_move(m, mv);
            delta_evaluate(m, changed.data(), changed.size());

            // A fresh copy of the model at the same assignment, evaluated from
            // scratch: the reference the incremental value must match.
            Model fresh;
            auto flv = fresh.list_var(7, "seq");
            const int32_t f_open = fresh.pair_lambda_sum(flv, d);
            const int32_t f_cyclic = fresh.pair_lambda_sum(flv, d, PairMode::Cyclic);
            const int32_t f_open_ends = fresh.pair_lambda_sum(flv, d, head_cost, tail_cost);
            const int32_t f_cyclic_ends =
                fresh.pair_lambda_sum(flv, d, head_cost, tail_cost, PairMode::Cyclic);
            fresh.minimize(fresh.sum({f_open, f_cyclic, f_open_ends, f_cyclic_ends}));
            fresh.close();
            fresh.var_mut(vid(flv)).elements = m.var(vid(lv)).elements;
            full_evaluate(fresh);

            INFO("move " << mv.move_type);
            REQUIRE(m.node_value(open_node) == fresh.node_value(f_open));
            REQUIRE(m.node_value(cyclic_node) == fresh.node_value(f_cyclic));
            REQUIRE(m.node_value(open_ends) == fresh.node_value(f_open_ends));
            REQUIRE(m.node_value(cyclic_ends) == fresh.node_value(f_cyclic_ends));
        }
    }

    // The loop is only worth its cost if it really saw every generator.
    for (const char* want :
         {"list_swap", "list_2opt", "list_relocate", "list_or_opt_2", "list_or_opt_3"}) {
        INFO("generator " << want);
        REQUIRE(std::find(seen.begin(), seen.end(), want) != seen.end());
    }
}

TEST_CASE("a .cbls round-trip preserves the closing rule and the endpoint terms",
          "[io][pair_lambda]") {
    const std::vector<int32_t> elements = {2, 0, 3, 1};

    // head-only and tail-only are their own rows: the writer emits the two keys
    // through independent calls and the reader reads them through independent
    // calls, so a dropped or swapped one survives the both-endpoints row.
    struct Variant {
        const char* name;
        PairMode mode;
        bool head;
        bool tail;
        double expected;
    };
    const std::array<Variant, 6> variants = {{
        {"open", PairMode::Open, false, false, 54.0},
        {"cyclic", PairMode::Cyclic, false, false, 66.0},
        {"open+head", PairMode::Open, true, false, 54.0 + 102.0},
        {"open+tail", PairMode::Open, false, true, 54.0 + 1001.0},
        {"open+endpoints", PairMode::Open, true, true, 54.0 + 102.0 + 1001.0},
        {"cyclic+endpoints", PairMode::Cyclic, true, true, 66.0 + 102.0 + 1001.0},
    }};

    for (const Variant& v : variants) {
        INFO("variant " << v.name);
        Model m;
        auto lv = m.list_var(4, "seq");
        int32_t node = (v.head || v.tail) ? m.pair_lambda_sum(lv, d, v.head ? head_cost : nullptr,
                                                              v.tail ? tail_cost : nullptr, v.mode)
                                          : m.pair_lambda_sum(lv, d, v.mode);
        m.minimize(node);
        m.close();

        std::ostringstream out;
        save_model(m, out);
        const std::string saved = out.str();

        // An open chain with no endpoints is everything the format could say
        // before these variants existed, so it must still say exactly that.
        REQUIRE((saved.find("\"mode\"") != std::string::npos) == (v.mode == PairMode::Cyclic));
        REQUIRE((saved.find("\"head\"") != std::string::npos) == v.head);
        REQUIRE((saved.find("\"tail\"") != std::string::npos) == v.tail);

        std::istringstream in(saved);
        Model reloaded = load_model(in);
        reloaded.var_mut(0).elements = elements;
        full_evaluate(reloaded);
        REQUIRE(reloaded.node_value(reloaded.objective_id()) == v.expected);

        // Idempotent: a second save of the reloaded model is byte-identical.
        std::ostringstream out2;
        save_model(reloaded, out2);
        REQUIRE(out2.str() == saved);
    }
}

TEST_CASE("a .cbls round-trip of a Set child covers the whole universe", "[io][pair_lambda]") {
    // A Set's max_size is its cardinality bound, its universe_size the element
    // range. Tabulating by max_size wrote a matrix too narrow to index, and the
    // reload then threw on any element past it.
    Model m;
    auto sv = m.set_var(6, 2, 3, "chosen");
    int32_t node = m.pair_lambda_sum(sv, d, head_cost, tail_cost, PairMode::Cyclic);
    m.minimize(node);
    m.close();

    std::ostringstream out;
    save_model(m, out);
    std::istringstream in(out.str());
    Model reloaded = load_model(in);
    // Elements 4 and 5 are inside the universe but outside max_size == 3.
    reloaded.var_mut(0).elements = {5, 4};
    full_evaluate(reloaded);
    // d(5,4) + d(4,5) + head(5) + tail(4)
    REQUIRE(reloaded.node_value(reloaded.objective_id()) == 54.0 + 45.0 + 105.0 + 1004.0);
}

TEST_CASE("an unknown PairLambda mode is refused rather than read as open", "[io][pair_lambda]") {
    Model m;
    auto lv = m.list_var(3, "seq");
    m.minimize(m.pair_lambda_sum(lv, d, PairMode::Cyclic));
    m.close();
    std::ostringstream out;
    save_model(m, out);
    std::string corrupted = out.str();
    const auto at = corrupted.find("\"cyclic\"");
    REQUIRE(at != std::string::npos);
    corrupted.replace(at, std::string("\"cyclic\"").size(), "\"spiral\"");
    std::istringstream in(corrupted);
    REQUIRE_THROWS_AS(load_model(in), std::invalid_argument);
}

TEST_CASE("a frozen model shares the closing rule with its replicas", "[pair_lambda][share]") {
    // The side table is structure, not search state, so it belongs in
    // ModelStructure -- which a frozen model SHARES with every portfolio
    // replica rather than deep-copying (#157). A spec that had landed on the
    // per-worker side would leave each replica reading a default-constructed
    // entry, i.e. an open chain, while the master looked right.
    Model master;
    auto lv = master.list_var(4, "seq");
    const int32_t node = master.pair_lambda_sum(lv, d, head_cost, tail_cost, PairMode::Cyclic);
    master.minimize(node);
    master.close();
    master.var_mut(vid(lv)).elements = {2, 0, 3, 1};
    full_evaluate(master);
    master.freeze();
    REQUIRE(master.is_frozen());

    Model replica(master);
    REQUIRE(replica.is_frozen());
    replica.var_mut(vid(lv)).elements = {2, 0, 3, 1};
    full_evaluate(replica);
    REQUIRE(replica.node_value(node) == master.node_value(node));
    REQUIRE(replica.node_value(node) == 54.0 + 12.0 + 102.0 + 1001.0);

    // A replica's own assignment does not disturb the master's.
    replica.var_mut(vid(lv)).elements = {0, 1, 2, 3};
    full_evaluate(replica);
    REQUIRE(master.node_value(node) == 54.0 + 12.0 + 102.0 + 1001.0);
}

TEST_CASE("a cyclic pair_lambda_sum finds the optimal TSP tour", "[search][pair_lambda]") {
    // Seven cities on a circle, so the optimum is the convex-hull order and any
    // crossing costs more. The optimum is found by brute force in the test
    // rather than quoted, which keeps the assertion self-contained: 6! = 720
    // tours with city 0 pinned.
    constexpr int kCities = 7;
    std::vector<double> xs;
    std::vector<double> ys;
    for (int i = 0; i < kCities; ++i) {
        const double angle = 2.0 * 3.14159265358979323846 * i / kCities;
        xs.push_back(std::cos(angle));
        ys.push_back(std::sin(angle));
    }
    // Shuffle the city ids so the optimal tour is not the identity permutation,
    // which the List's initial order would otherwise hand the search for free.
    const std::vector<int> label = {0, 3, 6, 2, 5, 1, 4};
    auto dist = [&](int a, int b) {
        const double dx = xs[label[a]] - xs[label[b]];
        const double dy = ys[label[a]] - ys[label[b]];
        return std::sqrt((dx * dx) + (dy * dy));
    };

    std::vector<int> perm(kCities);
    std::iota(perm.begin(), perm.end(), 0);
    double best = std::numeric_limits<double>::infinity();
    // City 0 is pinned at the front: a cyclic tour is invariant under rotation,
    // so fixing it enumerates every distinct tour exactly once per direction.
    std::sort(perm.begin() + 1, perm.end());
    do {
        double total = 0.0;
        for (int i = 0; i < kCities; ++i) {
            total += dist(perm[i], perm[(i + 1) % kCities]);
        }
        best = std::min(best, total);
    } while (std::next_permutation(perm.begin() + 1, perm.end()));

    Model m;
    auto lv = m.list_var(kCities, "tour");
    int32_t length =
        m.pair_lambda_sum(lv, [&](int a, int b) { return dist(a, b); }, PairMode::Cyclic);
    m.minimize(length);
    m.close();

    // 100 000 iterations reaches the optimum on 10 of 10 seeds; three are
    // asserted here so the test is a property of the search rather than of one
    // lucky stream. The whole case costs well under a second -- a pure
    // objective model has no real row to satisfy, so a batch is cheap.
    for (uint64_t seed : {163U, 7U, 20240163U}) {
        INFO("seed " << seed);
        SearchResult r = solve_deterministic(m, 100000, seed);
        REQUIRE(r.feasible);
        REQUIRE_THAT(r.objective, Catch::Matchers::WithinAbs(best, 1e-9));
    }
}
