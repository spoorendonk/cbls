// Trajectory witnesses for models built on STRUCTURED variables (#163).
//
// Every other seed-pinned test in the suite is a scalar model: test_search's
// pinned runs, test_initialization, the throughput floors. None of them would
// notice a change in how a List or Set model searches, so "this change left the
// existing trajectories alone" had no witness at all for exactly the models a
// change to `Lambda`/`PairLambda` evaluation can move.
//
// These three pin the whole observable outcome of a seeded run -- iteration
// count, feasibility, objective bit pattern and the final element order -- on
// one List model, one Set model and one model mixing a List with scalars. They
// are a REGRESSION FENCE, not a quality assertion: the numbers below are
// whatever the engine produced, and any change to them is a trajectory change
// that has to be explained rather than re-recorded.
//
// Reproducibility rests on `solve_deterministic`: `time_limit = 0` removes the
// wall clock entirely, so an iteration budget alone bounds the run and no clock
// read can reach control flow (see the `has_deadline` note in src/search.cpp).

#include "test_helpers.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdio>
#include <string>
#include <vector>

using namespace cbls;

namespace {

// The whole observable outcome of one run, as one comparable line. A single
// string rather than four REQUIREs so a failure prints the entire signature it
// got, which is what a re-record would need.
std::string signature(const SearchResult& r, int32_t structured_var) {
    std::array<char, 128> buf{};
    std::snprintf(buf.data(), buf.size(), "iters=%lld feasible=%d obj=%.17g elements=",
                  static_cast<long long>(r.iterations), r.feasible ? 1 : 0, r.objective);
    std::string out(buf.data());
    const auto& elements = r.best_state.elements.at(static_cast<size_t>(structured_var));
    for (size_t i = 0; i < elements.size(); ++i) {
        out += (i == 0 ? "" : ",") + std::to_string(elements[i]);
    }
    return out;
}

// A deterministic, asymmetric "distance" over 0..n-1. Asymmetric on purpose:
// a symmetric matrix cannot tell a reversed tour from its original, so it would
// hide exactly the kind of trajectory change these witnesses are for.
double pair_cost(int a, int b) {
    return 1.0 + static_cast<double>(((a * 7) + (b * 3)) % 11);
}

double element_cost(int e) {
    return 1.0 + static_cast<double>((e * 5) % 7);
}

}  // namespace

TEST_CASE("structured trajectory witness: List with pair_lambda_sum", "[trajectory][structured]") {
    Model m;
    auto lv = m.list_var(7, "route");
    int32_t length = m.pair_lambda_sum(lv, pair_cost);
    m.add_constraint(m.leq(length, m.constant(40.0)));
    m.minimize(length);
    m.close();

    REQUIRE(signature(solve_deterministic(m, 4000, 20240163), vid(lv)) ==
            "iters=4000 feasible=1 obj=20 elements=6,1,0,4,2,3,5");
    REQUIRE(signature(solve_deterministic(m, 4000, 7), vid(lv)) ==
            "iters=4000 feasible=1 obj=30 elements=0,2,1,4,6,5,3");
}

TEST_CASE("structured trajectory witness: Set with lambda_sum", "[trajectory][structured]") {
    Model m;
    auto sv = m.set_var(9, 3, 6, "chosen");
    int32_t weight = m.lambda_sum(sv, element_cost);
    m.add_constraint(m.geq(weight, m.constant(12.0)));
    m.minimize(weight);
    m.close();

    REQUIRE(signature(solve_deterministic(m, 4000, 20240163), vid(sv)) ==
            "iters=4000 feasible=1 obj=15 elements=0,8,7,4");
    REQUIRE(signature(solve_deterministic(m, 4000, 7), vid(sv)) ==
            "iters=4000 feasible=1 obj=12 elements=3,8,6,7");
}

TEST_CASE("structured trajectory witness: List mixed with scalars", "[trajectory][structured]") {
    Model m;
    auto lv = m.list_var(6, "order");
    auto b = m.bool_var("use");
    auto x = m.int_var(0, 9, "level");
    int32_t length = m.pair_lambda_sum(lv, pair_cost);
    m.add_constraint(m.leq(m.sum({length, x}), m.constant(35.0)));
    m.add_constraint(m.geq(m.sum({x, b}), m.constant(3.0)));
    m.minimize(m.sum({length, m.prod(x, m.constant(0.5))}));
    m.close();

    REQUIRE(signature(solve_deterministic(m, 4000, 20240163), vid(lv)) ==
            "iters=4071 feasible=1 obj=13 elements=3,4,2,0,1,5");
    REQUIRE(signature(solve_deterministic(m, 4000, 7), vid(lv)) ==
            "iters=4000 feasible=1 obj=22 elements=0,4,3,1,2,5");
}
