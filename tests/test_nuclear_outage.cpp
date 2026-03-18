#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include "benchmarks/nuclear-outage/data.h"
#include "benchmarks/nuclear-outage/dispatch.h"
#include "benchmarks/nuclear-outage/nuclear_model.h"
#include "benchmarks/nuclear-outage/nuclear_hook.h"
#include <cstdio>

using namespace cbls;
using namespace cbls::nuclear_outage;

// Convert var handle (negative) to var ID (non-negative)
static int32_t h2v(int32_t handle) { return -(handle + 1); }

// Helper: load the mini instance
static NuclearInstance load_mini() {
    return load_jsonl("benchmarks/instances/nuclear-outage/mini.jsonl");
}

TEST_CASE("Nuclear outage mini instance loads", "[nuclear]") {
    auto inst = load_mini();
    REQUIRE(inst.n_units == 10);
    REQUIRE(inst.n_periods == 52);
    REQUIRE(inst.n_scenarios == 20);
    REQUIRE(inst.n_outages == 10);
    REQUIRE(inst.capacity.size() == 10);
    REQUIRE(inst.demand.size() == 20);
    REQUIRE(inst.demand[0].size() == 52);
}

TEST_CASE("Nuclear outage model builds", "[nuclear]") {
    auto inst = load_mini();
    auto nm = build_nuclear_model(inst);
    auto& m = nm.model;

    // 10 int vars (outage starts), no float vars (objective is a constant node)
    REQUIRE(m.num_vars() == 10);
    printf("\nNuclear model: %ld vars, %ld nodes, %ld constraints\n",
           (long)m.num_vars(), (long)m.num_nodes(),
           (long)m.constraint_ids().size());
}

TEST_CASE("Dispatch correctness - all units online", "[nuclear]") {
    auto inst = load_mini();

    // Schedule all outages at period 0 (they'll overlap but that's OK for this test)
    // With all units available at period 10, cost should just be fuel*demand
    std::vector<int> starts(inst.n_outages, 0);

    // At period 10 (after all outages end), all units should be available
    auto avail = compute_availability(inst, starts);

    // Check: at period 10, how many outages have ended?
    // Outage durations are [6,7,5,6,5,7,6,4,5,4], all start at 0
    // At t=10, all outages (max dur=7) have ended
    int online_count = 0;
    for (int u = 0; u < inst.n_units; ++u) {
        if (avail[10][u]) online_count++;
    }
    REQUIRE(online_count == inst.n_units);

    // At t=3, some should be offline (those with duration > 3)
    int offline_at_3 = 0;
    for (int u = 0; u < inst.n_units; ++u) {
        if (!avail[3][u]) offline_at_3++;
    }
    REQUIRE(offline_at_3 == inst.n_outages);  // all started at 0, all still in outage
}

TEST_CASE("Dispatch cost is positive", "[nuclear]") {
    auto inst = load_mini();

    // Spread outages across time
    std::vector<int> starts;
    for (int o = 0; o < inst.n_outages; ++o) {
        starts.push_back(inst.outage_earliest[o]);
    }

    double cost = expected_cost(inst, starts, 5);  // 5 scenarios
    REQUIRE(cost > 0.0);
    printf("\nDispatch cost (5 scenarios): %.2f\n", cost);
}

TEST_CASE("Resource violation penalty", "[nuclear]") {
    auto inst = load_mini();

    // Schedule all outages at the same time → should violate site constraints
    std::vector<int> starts(inst.n_outages, 5);
    double penalty = resource_violation_penalty(inst, starts);
    REQUIRE(penalty > 0.0);

    // Spread outages → should have less violation
    std::vector<int> spread_starts;
    for (int o = 0; o < inst.n_outages; ++o) {
        spread_starts.push_back(inst.outage_earliest[o] + o * 3);
    }
    // Clamp to valid range
    for (int o = 0; o < inst.n_outages; ++o) {
        spread_starts[o] = std::min(spread_starts[o], inst.outage_latest[o]);
    }
    double penalty2 = resource_violation_penalty(inst, spread_starts);
    REQUIRE(penalty2 < penalty);
}

TEST_CASE("Nuclear outage solver finds feasible solution", "[nuclear]") {
    auto inst = load_mini();
    auto sub = make_mini(inst, 5);  // 5 scenarios for speed
    auto nm = build_nuclear_model(sub);

    NuclearDispatchHook hook(sub, nm);
    hook.scenarios_per_move = 5;
    LNS lns(0.3);

    auto result = solve(nm.model, 5.0, 42, true, &hook, &lns);
    printf("\nMini solve: feasible=%d, obj=%.2f, iters=%ld, time=%.3fs\n",
           result.feasible, result.objective, (long)result.iterations,
           result.time_seconds);
    REQUIRE(result.feasible);
    REQUIRE(result.objective > 0.0);
}

TEST_CASE("Nuclear outage spacing constraints satisfied", "[nuclear]") {
    auto inst = load_mini();
    auto sub = make_mini(inst, 5);
    auto nm = build_nuclear_model(sub);

    NuclearDispatchHook hook(sub, nm);
    hook.scenarios_per_move = 5;
    LNS lns(0.3);

    auto result = solve(nm.model, 3.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);

    // Restore best state and check spacing constraints
    nm.model.restore_state(result.best_state);

    // Group outages by unit and verify no overlap
    std::vector<std::vector<int>> unit_outages(sub.n_units);
    for (int o = 0; o < sub.n_outages; ++o) {
        unit_outages[sub.outage_unit[o]].push_back(o);
    }

    for (int u = 0; u < sub.n_units; ++u) {
        auto& outages = unit_outages[u];
        if (outages.size() < 2) continue;

        // Sort by actual start time
        std::sort(outages.begin(), outages.end(),
                  [&](int a, int b) {
                      return nm.model.var(h2v(nm.s[a])).value < nm.model.var(h2v(nm.s[b])).value;
                  });

        for (size_t i = 0; i + 1 < outages.size(); ++i) {
            int o1 = outages[i];
            int o2 = outages[i + 1];
            double s1 = nm.model.var(h2v(nm.s[o1])).value;
            double s2 = nm.model.var(h2v(nm.s[o2])).value;
            // s1 + duration[o1] <= s2
            REQUIRE(s1 + sub.outage_duration[o1] <= s2 + 0.5);  // tolerance for int rounding
        }
    }
}
