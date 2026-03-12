#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include "benchmarks/chped/data.h"
#include "benchmarks/chped/chped_model.h"
#include <cstdio>

using namespace cbls;
using namespace cbls::chped;

TEST_CASE("CHPED 4-unit builds model", "[chped]") {
    auto inst = make_4unit();
    auto cm = build_chped_model(inst);
    auto& m = cm.model;
    REQUIRE(m.num_vars() == 8);  // 4 commit + 4 power
    REQUIRE(m.constraint_ids().size() == 1);  // 1 demand constraint
}

TEST_CASE("CHPED 4-unit feasibility", "[chped]") {
    auto inst = make_4unit();
    auto cm = build_chped_model(inst);
    auto& m = cm.model;

    auto result = solve(m, 2.0, 42);
    REQUIRE(result.feasible);
    printf("\n4-unit: feasible=%d, obj=%.2f, iters=%ld, time=%.3fs\n",
           result.feasible, result.objective, result.iterations, result.time_seconds);
}

TEST_CASE("CHPED 4-unit solution quality", "[chped]") {
    auto inst = make_4unit();
    auto cm = build_chped_model(inst);
    auto& m = cm.model;

    auto result = solve(m, 5.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 5000);
    printf("\n4-unit solution: cost=%.2f\n", result.objective);
}

TEST_CASE("CHPED 7-unit feasibility", "[chped]") {
    auto inst = make_7unit();
    auto cm = build_chped_model(inst);
    auto& m = cm.model;

    auto result = solve(m, 3.0, 42);
    REQUIRE(result.feasible);
    printf("\n7-unit: feasible=%d, obj=%.2f, iters=%ld\n",
           result.feasible, result.objective, result.iterations);
}

TEST_CASE("CHPED 24-unit feasibility", "[chped]") {
    auto inst = make_24unit();
    auto cm = build_chped_model(inst);
    auto& m = cm.model;

    auto result = solve(m, 10.0, 42);
    printf("\n24-unit: feasible=%d, obj=%.2f, iters=%ld\n",
           result.feasible, result.objective, result.iterations);
    REQUIRE(result.iterations > 100);
}
