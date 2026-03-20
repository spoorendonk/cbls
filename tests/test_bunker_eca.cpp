#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include "benchmarks/bunker-eca/data.h"
#include "benchmarks/bunker-eca/bunker_eca_model.h"
#include "benchmarks/bunker-eca/bunker_speed_hook.h"
#include "benchmarks/bunker-eca/verify_bunker_eca.h"
#include <cstdio>
#include <cmath>

using namespace cbls;
using namespace cbls::bunker_eca;

// For maximize models, solver stores -objective internally
static double actual_obj(const SearchResult& r, const Model& m) {
    if (!r.feasible) return r.objective;
    return m.is_maximizing() ? -r.objective : r.objective;
}

TEST_CASE("Bunker-ECA small instance data", "[bunker_eca]") {
    auto inst = make_small();
    REQUIRE(inst.ships.size() == 3);
    REQUIRE(inst.cargoes.size() == 10);
    REQUIRE(inst.regions.size() == 7);
    REQUIRE(inst.legs.size() > 0);
    REQUIRE(inst.bunker_options.size() > 0);

    int contracts = 0, spots = 0;
    for (auto& c : inst.cargoes) {
        if (c.is_contract) contracts++;
        else spots++;
    }
    REQUIRE(contracts == 6);
    REQUIRE(spots == 4);

    // Leg distances are symmetric
    for (auto& l : inst.legs) {
        double rev_dist = inst.leg_distance(l.to_region, l.from_region);
        REQUIRE(std::abs(l.distance - rev_dist) < 1e-6);
    }
}

TEST_CASE("Bunker-ECA small model builds", "[bunker_eca]") {
    auto inst = make_small();
    auto bec = build_bunker_eca_model(inst);
    auto& m = bec.model;

    REQUIRE(m.num_vars() > 0);
    REQUIRE(m.num_nodes() > 0);
    REQUIRE(m.is_closed());
    REQUIRE(m.is_maximizing());

    printf("\nSmall model: %ld vars, %ld nodes, %ld constraints\n",
           (long)m.num_vars(), (long)m.num_nodes(),
           (long)m.constraint_ids().size());
}

TEST_CASE("Bunker-ECA small feasibility", "[bunker_eca]") {
    auto inst = make_small();
    auto bec = build_bunker_eca_model(inst);

    BunkerSpeedHook hook;
    hook.set_model(&bec, &inst);

    auto result = solve(bec.model, 10.0, 42, true, &hook);
    double obj = actual_obj(result, bec.model);
    printf("\nSmall: feasible=%d, profit=%.0f, iters=%ld, time=%.3fs\n",
           result.feasible, obj, (long)result.iterations, result.time_seconds);

    REQUIRE(result.iterations > 100);
    if (result.feasible) {
        // Profit should be positive (revenue > costs)
        REQUIRE(obj > 0.0);
    }
}

TEST_CASE("Bunker-ECA small with LNS", "[bunker_eca]") {
    auto inst = make_small();
    auto bec = build_bunker_eca_model(inst);

    BunkerSpeedHook hook;
    hook.set_model(&bec, &inst);
    LNS lns(0.3);

    auto result = solve(bec.model, 15.0, 42, true, &hook, &lns);
    double obj = actual_obj(result, bec.model);
    printf("\nSmall+LNS: feasible=%d, profit=%.0f, iters=%ld, time=%.3fs\n",
           result.feasible, obj, (long)result.iterations, result.time_seconds);
}

TEST_CASE("Bunker-ECA fuel consumption formula", "[bunker_eca]") {
    // fuel = k * v^2 * d / 24
    double k = 0.0035;
    double v = 13.0;
    double d = 8300.0;
    double fuel = k * v * v * d / 24.0;
    REQUIRE(std::abs(fuel - 204.458) < 1.0);

    // travel_time = d / (24 * v) = 8300 / (24 * 13) = 26.6 days
    double travel = d / (24.0 * v);
    REQUIRE(std::abs(travel - 26.6) < 0.1);
}

TEST_CASE("Bunker-ECA medium model builds", "[bunker_eca]") {
    auto inst = make_medium();
    auto bec = build_bunker_eca_model(inst);
    auto& m = bec.model;

    REQUIRE(m.num_vars() > 0);
    REQUIRE(m.is_closed());

    printf("\nMedium model: %ld vars, %ld nodes, %ld constraints\n",
           (long)m.num_vars(), (long)m.num_nodes(),
           (long)m.constraint_ids().size());
}

TEST_CASE("Bunker-ECA small verify", "[bunker_eca][verify]") {
    auto inst = make_small();
    auto bec = build_bunker_eca_model(inst);

    BunkerSpeedHook hook;
    hook.set_model(&bec, &inst);
    LNS lns(0.3);

    auto result = solve(bec.model, 15.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);

    auto vr = verify_bunker_eca(bec, inst);
    vr.print_diagnostics(stdout);
    REQUIRE(vr.ok);
}

TEST_CASE("Bunker-ECA small noECA verify", "[bunker_eca][verify]") {
    auto inst = make_small();
    for (auto& leg : inst.legs) {
        leg.eca_fraction = 0.0;
    }
    auto bec = build_bunker_eca_model(inst);

    BunkerSpeedHook hook;
    hook.set_model(&bec, &inst);
    LNS lns(0.3);

    auto result = solve(bec.model, 15.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);

    auto vr = verify_bunker_eca(bec, inst);
    vr.print_diagnostics(stdout);
    REQUIRE(vr.ok);
}

TEST_CASE("Bunker-ECA medium verify", "[bunker_eca][verify]") {
    auto inst = make_medium();
    auto bec = build_bunker_eca_model(inst);

    BunkerSpeedHook hook;
    hook.set_model(&bec, &inst);
    LNS lns(0.3);

    auto result = solve(bec.model, 30.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);

    auto vr = verify_bunker_eca(bec, inst);
    vr.print_diagnostics(stdout);
    REQUIRE(vr.ok);
}

TEST_CASE("Bunker-ECA medium feasibility", "[bunker_eca]") {
    auto inst = make_medium();
    auto bec = build_bunker_eca_model(inst);

    BunkerSpeedHook hook;
    hook.set_model(&bec, &inst);
    LNS lns(0.3);

    auto result = solve(bec.model, 120.0, 42, true, &hook, &lns);
    double obj = actual_obj(result, bec.model);
    printf("\nMedium: feasible=%d, profit=%.0f, iters=%ld, time=%.3fs\n",
           result.feasible, obj, (long)result.iterations, result.time_seconds);

    REQUIRE(result.iterations > 50);
}
