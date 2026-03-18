#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include "benchmarks/uc-chped/data.h"
#include "benchmarks/uc-chped/uc_model.h"
#include <cstdio>

using namespace cbls;
using namespace cbls::uc_chped;

TEST_CASE("UC-CHPED 13-unit 1-period model builds", "[uc-chped]") {
    auto ucp13 = load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
    auto inst = make_subinstance(ucp13, 1);
    auto ucm = build_uc_model(inst);
    auto& m = ucm.model;
    // 13 bool (commitment) + 13 float (dispatch) = 26 vars
    REQUIRE(m.num_vars() == 26);
    printf("\n13-unit 1p: %ld vars, %ld nodes\n",
           (long)m.num_vars(), (long)m.num_nodes());
}

TEST_CASE("UC-CHPED 13-unit 1-period feasibility", "[uc-chped]") {
    auto ucp13 = load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
    auto inst = make_subinstance(ucp13, 1);
    auto ucm = build_uc_model(inst);

    FloatIntensifyHook hook;
    LNS lns(0.3);
    auto result = solve(ucm.model, 5.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);
    printf("\n13-unit 1p: obj=%.1f, iters=%ld, time=%.3fs\n",
           result.objective, (long)result.iterations, result.time_seconds);
}

TEST_CASE("UC-CHPED 13-unit 1-period quality", "[uc-chped]") {
    auto ucp13 = load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
    auto inst = make_subinstance(ucp13, 1);
    auto ucm = build_uc_model(inst);

    FloatIntensifyHook hook;
    LNS lns(0.3);
    auto result = solve(ucm.model, 10.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);
    // Known LB = 11701 from Pedroso. Allow up to ~25% gap for SA.
    REQUIRE(result.objective < 15000);
    printf("\n13-unit 1p quality: obj=%.1f (known LB=11701)\n", result.objective);
}

TEST_CASE("UC-CHPED 40-unit 1-period feasibility", "[uc-chped]") {
    auto ucp40 = load_jsonl("benchmarks/instances/uc-chped/ucp40.jsonl");
    auto inst = make_subinstance(ucp40, 1);
    auto ucm = build_uc_model(inst);

    FloatIntensifyHook hook;
    LNS lns(0.3);
    auto result = solve(ucm.model, 10.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);
    printf("\n40-unit 1p: obj=%.1f, iters=%ld, time=%.3fs\n",
           result.objective, (long)result.iterations, result.time_seconds);
}

TEST_CASE("UC-CHPED 100-unit 1-period builds and solves", "[uc-chped]") {
    auto ucp100 = load_jsonl("benchmarks/instances/uc-chped/ucp100.jsonl");
    auto inst = make_subinstance(ucp100, 1);
    auto ucm = build_uc_model(inst);
    auto& m = ucm.model;
    // 100 bool + 100 float = 200 vars
    REQUIRE(m.num_vars() == 200);

    FloatIntensifyHook hook;
    LNS lns(0.3);
    auto result = solve(ucm.model, 30.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);
    printf("\n100-unit 1p: obj=%.1f, %ld vars, iters=%ld, time=%.3fs\n",
           result.objective, (long)m.num_vars(), (long)result.iterations, result.time_seconds);
}
