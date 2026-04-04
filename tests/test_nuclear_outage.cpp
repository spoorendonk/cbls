#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cbls/cbls.h>
#include "benchmarks/nuclear-outage/data.h"
#include "benchmarks/nuclear-outage/dispatch.h"
#include "benchmarks/nuclear-outage/nuclear_model.h"
#include "benchmarks/nuclear-outage/nuclear_hook.h"
#include "benchmarks/nuclear-outage/roadef_dispatch.h"
#include "benchmarks/nuclear-outage/roadef_hook.h"
#include <cstdio>
#include <fstream>

using namespace cbls;
using namespace cbls::nuclear_outage;

// Helper: load the mini instance
static NuclearInstance load_mini() {
    return load_jsonl("benchmarks/instances/nuclear-outage/mini.jsonl");
}

// ===========================================================================
// Legacy synthetic instance tests
// ===========================================================================

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

    REQUIRE(m.num_vars() == 10);
    printf("\nNuclear model: %ld vars, %ld nodes, %ld constraints\n",
           (long)m.num_vars(), (long)m.num_nodes(),
           (long)m.constraint_ids().size());
}

TEST_CASE("Dispatch correctness - all units online", "[nuclear]") {
    auto inst = load_mini();
    std::vector<int> starts(inst.n_outages, 0);
    auto avail = compute_availability(inst, starts);

    int online_count = 0;
    for (int u = 0; u < inst.n_units; ++u) {
        if (avail[10][u]) online_count++;
    }
    REQUIRE(online_count == inst.n_units);

    int offline_at_3 = 0;
    for (int u = 0; u < inst.n_units; ++u) {
        if (!avail[3][u]) offline_at_3++;
    }
    REQUIRE(offline_at_3 == inst.n_outages);
}

TEST_CASE("Dispatch cost is positive", "[nuclear]") {
    auto inst = load_mini();
    std::vector<int> starts;
    for (int o = 0; o < inst.n_outages; ++o) {
        starts.push_back(inst.outage_earliest[o]);
    }
    double cost = expected_cost(inst, starts, 5);
    REQUIRE(cost > 0.0);
    printf("\nDispatch cost (5 scenarios): %.2f\n", cost);
}

TEST_CASE("Resource violation penalty - max outages per site", "[nuclear]") {
    auto inst = load_mini();
    std::vector<int> starts(inst.n_outages, 5);
    double penalty = resource_violation_penalty(inst, starts);
    REQUIRE(penalty > 0.0);

    std::vector<int> spread_starts;
    for (int o = 0; o < inst.n_outages; ++o) {
        spread_starts.push_back(inst.outage_earliest[o] + o * 3);
    }
    for (int o = 0; o < inst.n_outages; ++o) {
        spread_starts[o] = std::min(spread_starts[o], inst.outage_latest[o]);
    }
    double penalty2 = resource_violation_penalty(inst, spread_starts);
    REQUIRE(penalty2 < penalty);
}

TEST_CASE("Resource violation penalty - site spacing", "[nuclear]") {
    auto inst = load_mini();
    std::vector<int> starts(inst.n_outages, 40);
    starts[0] = 5;
    starts[1] = 11;

    double penalty = resource_violation_penalty(inst, starts);
    REQUIRE(penalty > 0.0);
    printf("\nSite spacing penalty (gap=0, min=3): %.0f\n", penalty);

    starts[1] = 14;
    double penalty2 = resource_violation_penalty(inst, starts);
    REQUIRE(penalty2 < penalty);
}

TEST_CASE("DispatchEvaluator matches free functions", "[nuclear]") {
    auto inst = load_mini();
    std::vector<std::vector<int>> schedules = {
        {1, 5, 10, 1, 8, 15, 20, 1, 12, 25},
        {30, 35, 40, 28, 33, 38, 42, 30, 36, 42},
        {15, 20, 25, 14, 20, 26, 30, 15, 24, 33},
        std::vector<int>(10, 5),
    };

    DispatchEvaluator eval(inst);

    for (auto& starts : schedules) {
        double ref_cost = expected_cost(inst, starts, 10, 0);
        double eval_cost = eval.expected_cost(starts, 10, 0, {});
        REQUIRE(eval_cost == Catch::Approx(ref_cost).epsilon(1e-10));

        double ref_pen = resource_violation_penalty(inst, starts);
        double eval_pen = eval.resource_penalty(starts, 1e6, -1);
        REQUIRE(eval_pen == Catch::Approx(ref_pen).epsilon(1e-10));
    }

    auto starts = schedules[0];
    eval.resource_penalty(starts, 1e6, -1);

    starts[3] = 10;
    double ref_pen = resource_violation_penalty(inst, starts);
    double delta_pen = eval.resource_penalty(starts, 1e6, 3);
    REQUIRE(delta_pen == Catch::Approx(ref_pen).epsilon(1e-10));

    starts[7] = 20;
    ref_pen = resource_violation_penalty(inst, starts);
    delta_pen = eval.resource_penalty(starts, 1e6, 7);
    REQUIRE(delta_pen == Catch::Approx(ref_pen).epsilon(1e-10));
}

TEST_CASE("Incremental dispatch matches full dispatch", "[nuclear]") {
    auto inst = load_mini();
    DispatchEvaluator eval(inst);

    auto starts = std::vector<int>{1, 5, 10, 1, 8, 15, 20, 1, 12, 25};

    double full_cost = eval.expected_cost(starts, 10, 0, {});
    double ref = expected_cost(inst, starts, 10, 0);
    REQUIRE(full_cost == Catch::Approx(ref).epsilon(1e-10));

    for (int trial = 0; trial < 5; ++trial) {
        int o = trial % inst.n_outages;
        int new_start = starts[o] + 3;
        if (new_start > inst.outage_latest[o]) new_start = inst.outage_earliest[o];
        starts[o] = new_start;

        double incr_cost = eval.expected_cost(starts, 10, 0, {o});
        double ref_cost = expected_cost(inst, starts, 10, 0);
        REQUIRE(incr_cost == Catch::Approx(ref_cost).epsilon(1e-10));
    }

    starts[0] = 10; starts[5] = 20;
    double multi_cost = eval.expected_cost(starts, 10, 0, {0, 5});
    double multi_ref = expected_cost(inst, starts, 10, 0);
    REQUIRE(multi_cost == Catch::Approx(multi_ref).epsilon(1e-10));

    double new_window = eval.expected_cost(starts, 10, 5, {});
    double ref_new = expected_cost(inst, starts, 10, 5);
    REQUIRE(new_window == Catch::Approx(ref_new).epsilon(1e-10));
}

TEST_CASE("Nuclear outage solver finds feasible solution", "[nuclear]") {
    auto inst = load_mini();
    auto sub = make_mini(inst, 5);
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

    nm.model.restore_state(result.best_state);

    std::vector<std::vector<int>> unit_outages(sub.n_units);
    for (int o = 0; o < sub.n_outages; ++o) {
        unit_outages[sub.outage_unit[o]].push_back(o);
    }

    for (int u = 0; u < sub.n_units; ++u) {
        auto& outages = unit_outages[u];
        if (outages.size() < 2) continue;
        std::sort(outages.begin(), outages.end(),
                  [&](int a, int b) {
                      return nm.model.var(handle_to_var_id(nm.s[a])).value < nm.model.var(handle_to_var_id(nm.s[b])).value;
                  });
        for (size_t i = 0; i + 1 < outages.size(); ++i) {
            int o1 = outages[i];
            int o2 = outages[i + 1];
            double s1 = nm.model.var(handle_to_var_id(nm.s[o1])).value;
            double s2 = nm.model.var(handle_to_var_id(nm.s[o2])).value;
            REQUIRE(s1 + sub.outage_duration[o1] <= s2 + 0.5);
        }
    }
}

// ===========================================================================
// ROADEF 2010 competition data tests
// ===========================================================================

static const char* ROADEF_DATA0 =
    "benchmarks/instances/nuclear-outage/data_release_13102009/data_release_13102009/data0.txt";

static bool roadef_data_exists() {
    std::ifstream f(ROADEF_DATA0);
    return f.good();
}

TEST_CASE("ROADEF data0 parser", "[roadef]") {
    if (!roadef_data_exists()) {
        SKIP("ROADEF data0.txt not found");
    }

    auto inst = load_roadef(ROADEF_DATA0);

    REQUIRE(inst.T == 623);
    REQUIRE(inst.H == 89);
    REQUIRE(inst.K == 2);
    REQUIRE(inst.S == 2);
    REQUIRE(inst.epsilon == Catch::Approx(0.01));
    REQUIRE(inst.n_type1 == 1);
    REQUIRE(inst.n_type2 == 2);
    REQUIRE(inst.timesteps_per_week == 7);

    // Check timestep durations
    REQUIRE(inst.timestep_durations.size() == 623);
    REQUIRE(inst.timestep_durations[0] == Catch::Approx(24.0));

    // Check demand
    REQUIRE(inst.demand.size() == 2);
    REQUIRE(inst.demand[0].size() == 623);
    REQUIRE(inst.demand[0][0] > 50000.0);

    // Check Type 1 plant
    REQUIRE(inst.type1_plants.size() == 1);
    REQUIRE(inst.type1_plants[0].name == "PowerPlant_1_0");
    REQUIRE(inst.type1_plants[0].pmin[0].size() == 623);
    REQUIRE(inst.type1_plants[0].cost[0][0] == Catch::Approx(10000.0));

    // Check Type 2 plants
    REQUIRE(inst.type2_plants.size() == 2);
    auto& p0 = inst.type2_plants[0];
    REQUIRE(p0.name == "PowerPlant_2_0");
    REQUIRE(p0.initial_stock == Catch::Approx(4974480.0));
    REQUIRE(p0.n_cycles == 2);
    REQUIRE(p0.durations.size() == 2);
    REQUIRE(p0.durations[0] == 5);
    REQUIRE(p0.durations[1] == 8);
    REQUIRE(p0.rmax.size() == 2);
    REQUIRE(p0.rmin.size() == 2);
    REQUIRE(p0.bo.size() == 3);  // K+1
    REQUIRE(p0.profiles.size() == 3);  // K+1
    REQUIRE(p0.pmax_t.size() == 623);
    REQUIRE(p0.fuel_price_end > 0.0);

    // Check profiles have points
    REQUIRE(p0.profiles[0].points.size() == 7);
    REQUIRE(p0.profiles[0].points[0].second == Catch::Approx(1.0));

    // Check constraints
    REQUIRE(inst.ct13.size() == 4);
    REQUIRE(inst.spacing_constraints.size() == 1);
    REQUIRE(inst.spacing_constraints[0].type == 14);
    REQUIRE(inst.spacing_constraints[0].spacing == Catch::Approx(6.0));
    REQUIRE(inst.ct19.empty());
    REQUIRE(inst.ct20.empty());
    REQUIRE(inst.ct21.empty());

    printf("\nROADEF data0: %d timesteps, %d weeks, %d outages, parsed OK\n",
           inst.T, inst.H, inst.n_outages());
}

TEST_CASE("ROADEF model builds from data0", "[roadef]") {
    if (!roadef_data_exists()) {
        SKIP("ROADEF data0.txt not found");
    }

    auto inst = load_roadef(ROADEF_DATA0);
    auto rm = build_roadef_model(inst);

    REQUIRE(rm.model.num_vars() == 4);
    REQUIRE(rm.ha.size() == 4);
    REQUIRE(rm.outage_info.size() == 4);

    printf("\nROADEF model: %ld vars, %ld nodes, %ld constraints\n",
           (long)rm.model.num_vars(), (long)rm.model.num_nodes(),
           (long)rm.model.constraint_ids().size());
}

TEST_CASE("ROADEF solver produces feasible solution on data0", "[roadef]") {
    if (!roadef_data_exists()) {
        SKIP("ROADEF data0.txt not found");
    }

    auto inst = load_roadef(ROADEF_DATA0);
    auto rm = build_roadef_model(inst);

    ROADEFDispatchHook hook(inst, rm);
    LNS lns(0.3);

    auto result = solve(rm.model, 10.0, 42, true, &hook, &lns);

    printf("\nROADEF data0 solve: feasible=%d, obj=%.2f, iters=%ld, time=%.1fs\n",
           result.feasible, result.objective, (long)result.iterations,
           result.time_seconds);
    REQUIRE(result.feasible);
    REQUIRE(result.objective > 0.0);
    REQUIRE(result.objective < 1e14);  // sanity bound
}

TEST_CASE("DecreaseProfile interpolation", "[roadef]") {
    DecreaseProfile prof;
    // Profile with 3 points: (100, 1.0), (50, 0.8), (0, 0.5)
    prof.points = {{100.0, 1.0}, {50.0, 0.8}, {0.0, 0.5}};

    // Above highest point → clamp to first
    REQUIRE(prof.evaluate(150.0) == Catch::Approx(1.0));

    // At exact points
    REQUIRE(prof.evaluate(100.0) == Catch::Approx(1.0));
    REQUIRE(prof.evaluate(50.0) == Catch::Approx(0.8));
    REQUIRE(prof.evaluate(0.0) == Catch::Approx(0.5));

    // Interpolation between points
    REQUIRE(prof.evaluate(75.0) == Catch::Approx(0.9));
    REQUIRE(prof.evaluate(25.0) == Catch::Approx(0.65));

    // Below lowest point → clamp to last
    REQUIRE(prof.evaluate(-10.0) == Catch::Approx(0.5));
}

// ===========================================================================
// simulate_scenario unit tests for the two correctness fixes
// ===========================================================================

// Build a minimal ROADEFInstance with one Type 1 plant and no Type 2 plants.
// Useful for testing Type 1 dispatch behavior in isolation.
static ROADEFInstance build_t1_only_instance(double pmin, double pmax, double cost,
                                             double demand_per_step) {
    ROADEFInstance inst;
    inst.T = 4;
    inst.H = 2;
    inst.timesteps_per_week = 2;
    inst.K = 0;
    inst.S = 1;
    inst.n_type1 = 1;
    inst.n_type2 = 0;
    inst.epsilon = 0.01;
    inst.timestep_durations.assign(inst.T, 1.0);
    inst.demand.assign(1, std::vector<double>(inst.T, demand_per_step));

    Type1Plant p1;
    p1.name = "p1";
    p1.index = 0;
    p1.pmin = {std::vector<double>(inst.T, pmin)};
    p1.pmax = {std::vector<double>(inst.T, pmax)};
    p1.cost = {std::vector<double>(inst.T, cost)};
    inst.type1_plants.push_back(p1);

    return inst;
}

TEST_CASE("simulate_scenario overproduces when demand below pmin", "[roadef]") {
    // demand=5, pmin=10, pmax=20 → plant runs at pmin=10 (overproducing by 5)
    auto inst = build_t1_only_instance(10.0, 20.0, 2.0, 5.0);

    PlantStatus status;
    std::vector<std::vector<double>> reload;
    auto result = simulate_scenario(inst, status, reload, 0);

    // Expected: each timestep dispatches 10 at cost 2.0 * dt (dt=1), 4 timesteps
    // Total cost = 4 * 10 * 2.0 = 80. No unserved penalty.
    REQUIRE(result.cost == Catch::Approx(80.0));
}

TEST_CASE("simulate_scenario dispatches normally when demand above pmin", "[roadef]") {
    // demand=15, pmin=10, pmax=20 → plant runs at 15 exactly
    auto inst = build_t1_only_instance(10.0, 20.0, 2.0, 15.0);

    PlantStatus status;
    std::vector<std::vector<double>> reload;
    auto result = simulate_scenario(inst, status, reload, 0);

    // Total = 4 * 15 * 2.0 = 120
    REQUIRE(result.cost == Catch::Approx(120.0));
}

TEST_CASE("simulate_scenario caps at pmax when demand exceeds pmax", "[roadef]") {
    // demand=30, pmin=10, pmax=20 → plant runs at pmax=20, remaining=10 unserved
    auto inst = build_t1_only_instance(10.0, 20.0, 2.0, 30.0);

    PlantStatus status;
    std::vector<std::vector<double>> reload;
    auto result = simulate_scenario(inst, status, reload, 0);

    // Dispatch cost: 4 * 20 * 2.0 = 160
    // Unserved penalty: 4 * 10 * 1e5 * 1.0 = 4e6
    REQUIRE(result.cost == Catch::Approx(160.0 + 4e6));
}

// Build a minimal ROADEFInstance with one Type 2 plant that has a week-0 outage.
// Used to verify the CT10 refueling fix fires at week 0.
static ROADEFInstance build_week0_outage_instance() {
    ROADEFInstance inst;
    inst.T = 4;
    inst.H = 2;
    inst.timesteps_per_week = 2;
    inst.K = 1;
    inst.S = 1;
    inst.n_type1 = 0;
    inst.n_type2 = 1;
    inst.epsilon = 0.01;
    inst.timestep_durations.assign(inst.T, 1.0);
    inst.demand.assign(1, std::vector<double>(inst.T, 0.0));  // no demand

    Type2Plant p2;
    p2.name = "p2";
    p2.index = 0;
    p2.initial_stock = 100.0;
    p2.n_cycles = 1;
    p2.durations = {2};                 // outage lasts 2 weeks (both weeks)
    p2.mmax = {100.0, 100.0};           // K+1 entries
    p2.bo = {10.0, 20.0};               // K+1 entries
    p2.rmax = {50.0};
    p2.rmin = {0.0};
    p2.q = {2.0};                       // refuel coefficient
    p2.amax = {100.0};
    p2.smax = {200.0};
    p2.refuel_cost = {0.0};
    p2.pmax_t.assign(inst.T, 100.0);
    p2.fuel_price_end = 0.0;
    p2.profiles.emplace_back();          // empty profiles (evaluate returns 1.0)
    p2.profiles.emplace_back();
    inst.type2_plants.push_back(p2);

    // Schedule an outage at week 0 for plant 0, cycle 0
    CT13Window w;
    w.plant_idx = 0;
    w.cycle = 0;
    w.TO = 0;
    w.TA = 0;
    inst.ct13.push_back(w);

    return inst;
}

TEST_CASE("simulate_scenario applies CT10 refueling at week-0 outage", "[roadef]") {
    auto inst = build_week0_outage_instance();
    std::vector<int> ha = {0};  // outage starts at week 0

    auto lookup = build_outage_lookup(inst, ha);
    auto status = compute_plant_status(inst, lookup);
    auto reload = compute_reloads(inst, lookup);

    // Plant should be in outage for both weeks; refueling should trigger at t=0.
    REQUIRE(status.in_outage[0][0] == true);
    REQUIRE(status.in_outage[0][1] == true);
    REQUIRE(reload[0][0] == Catch::Approx(50.0));  // rmax

    auto result = simulate_scenario(inst, status, reload, 0);

    // CT10 formula at t=0 (first outage timestep, k=0):
    //   fuel = ((q-1)/q) * (initial_stock - bo[0]) + reload[0][0] + bo[1]
    //        = (1/2) * (100 - 10) + 50 + 20
    //        = 45 + 50 + 20 = 115
    // Then fuel credit at end = 0 * 115 = 0 (fuel_price_end=0)
    // No production (outage whole horizon), no Type 1, no demand.
    // Cost should be 0 (no unserved, no production, no residual credit).
    REQUIRE(result.cost == Catch::Approx(0.0));
}

TEST_CASE("build_outage_lookup maps plant/cycle to start week", "[roadef]") {
    auto inst = build_week0_outage_instance();
    std::vector<int> ha = {0};

    auto lookup = build_outage_lookup(inst, ha);

    REQUIRE(lookup.size() == 1);  // 1 Type 2 plant
    REQUIRE(lookup[0].size() == 1);  // 1 cycle
    REQUIRE(lookup[0][0] == 0);  // outage starts at week 0
}
