#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include "benchmarks/pharma-glsp/data.h"
#include "benchmarks/pharma-glsp/glsp_model.h"
#include "benchmarks/pharma-glsp/glsp_hook.h"
#include "benchmarks/pharma-glsp/verify_glsp.h"
#include <cstdio>

using namespace cbls;
using namespace cbls::glsp;

TEST_CASE("GLSP tiny instance builds model", "[glsp]") {
    auto inst = make_tiny();
    auto gm = build_glsp_model(inst);
    auto& m = gm.model;

    // 3 products, 2 macro-periods
    // Variables: 2 ListVar(3) + 3*2 FloatVar = 8 vars
    REQUIRE(m.num_vars() == 8);
    // Should have constraints (capacity + demand satisfaction + min lot)
    REQUIRE(m.constraint_ids().size() > 0);
    // Should have objective
    REQUIRE(m.objective_id() >= 0);

    printf("\nGLSP tiny: %zu vars, %zu nodes, %zu constraints\n",
           m.num_vars(), m.num_nodes(), m.constraint_ids().size());
}

TEST_CASE("GLSP tiny instance feasibility", "[glsp]") {
    auto inst = make_tiny();
    auto gm = build_glsp_model(inst);
    auto& m = gm.model;

    GLSPInnerSolverHook hook(inst, gm.seq, gm.lot);
    auto result = solve(m, 5.0, 42, true, &hook);
    printf("\nGLSP tiny: feasible=%d, obj=%.2f, iters=%ld, time=%.3fs\n",
           result.feasible, result.objective, (long)result.iterations,
           result.time_seconds);
    REQUIRE(result.feasible);
    REQUIRE(result.objective >= 0);
}

TEST_CASE("GLSP tiny with LNS", "[glsp]") {
    auto inst = make_tiny();
    auto gm = build_glsp_model(inst);
    auto& m = gm.model;

    GLSPInnerSolverHook hook(inst, gm.seq, gm.lot);
    LNS lns(0.3);
    auto result = solve(m, 5.0, 42, true, &hook, &lns);
    printf("\nGLSP tiny+LNS: feasible=%d, obj=%.2f, iters=%ld\n",
           result.feasible, result.objective, (long)result.iterations);
    REQUIRE(result.feasible);
}

TEST_CASE("List moves produce valid permutations", "[glsp][moves]") {
    // Create a model with a single ListVar(5)
    Model m;
    auto lv = m.list_var(5, "test_list");
    int32_t var_id = -(lv + 1);  // handle to var_id
    m.minimize(m.constant(0.0));
    m.close();

    // Initialize
    RNG rng(42);
    initialize_random(m, rng);

    REQUIRE(m.var(var_id).elements.size() == 5);

    // Generate moves and verify they all produce valid permutations
    for (int iter = 0; iter < 100; ++iter) {
        auto moves = generate_standard_moves(m.var(var_id), rng);
        // Should generate: swap, 2opt, relocate, + possibly or_opt_2, or_opt_3
        REQUIRE(moves.size() >= 3);

        for (const auto& move : moves) {
            const auto& new_elems = move.changes[0].new_elements;
            REQUIRE(new_elems.size() == 5);

            // Check it's a valid permutation
            std::vector<int32_t> sorted_elems = new_elems;
            std::sort(sorted_elems.begin(), sorted_elems.end());
            for (int i = 0; i < 5; ++i) {
                REQUIRE(sorted_elems[i] == i);
            }
        }

        // Apply first move to evolve the list
        apply_move(m, moves[0]);
    }
}

TEST_CASE("GLSP loaded instance builds and solves", "[glsp]") {
    // Try to load class_a.jsonl and solve the first instance
    std::vector<GLSPInstance> instances;
    try {
        instances = load_jsonl("benchmarks/instances/pharma-glsp/class_a.jsonl");
    } catch (...) {
        // File might not exist in CI
        printf("\nSkipping: class_a.jsonl not found\n");
        return;
    }

    REQUIRE(!instances.empty());
    const auto& inst = instances[0];

    auto gm = build_glsp_model(inst);
    auto& m = gm.model;

    printf("\nGLSP %s: J=%d T=%d M=%d, %zu vars, %zu nodes, %zu constraints\n",
           inst.name.c_str(), inst.n_products, inst.n_macro,
           inst.n_micro_per_macro, m.num_vars(), m.num_nodes(),
           m.constraint_ids().size());

    GLSPInnerSolverHook hook(inst, gm.seq, gm.lot);
    LNS lns(0.3);
    auto result = solve(m, 10.0, 42, true, &hook, &lns);

    printf("GLSP %s: feasible=%d, obj=%.2f, iters=%ld, time=%.3fs\n",
           inst.name.c_str(), result.feasible, result.objective,
           (long)result.iterations, result.time_seconds);
    REQUIRE(result.iterations > 100);
}

TEST_CASE("GLSP tiny verify", "[glsp][verify]") {
    auto inst = make_tiny();
    auto gm = build_glsp_model(inst);

    GLSPInnerSolverHook hook(inst, gm.seq, gm.lot);
    LNS lns(0.3);
    auto result = solve(gm.model, 5.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);

    auto vr = verify_glsp(gm, inst);
    vr.print_diagnostics(stdout);
    REQUIRE(vr.ok);
}

TEST_CASE("GLSP loaded class-A verify", "[glsp][verify]") {
    std::vector<GLSPInstance> instances;
    try {
        instances = load_jsonl("benchmarks/instances/pharma-glsp/class_a.jsonl");
    } catch (...) {
        printf("\nSkipping: class_a.jsonl not found\n");
        return;
    }
    REQUIRE(!instances.empty());
    const auto& inst = instances[0];

    auto gm = build_glsp_model(inst);

    GLSPInnerSolverHook hook(inst, gm.seq, gm.lot);
    LNS lns(0.3);
    auto result = solve(gm.model, 15.0, 42, true, &hook, &lns);
    REQUIRE(result.feasible);

    auto vr = verify_glsp(gm, inst);
    vr.print_diagnostics(stdout);
    REQUIRE(vr.ok);
}
