// Set covering on a collection-typed `Set` variable (issue #93): the parser, the
// two encodings, the verifier, and a search on the vendored OR-Library roster.
//
// The Set-encoding solve tests are the point of the file: they are the only
// place in the suite where the Set move generators are exercised on a real
// instance rather than a 3-10 element toy.

#include "test_helpers.h"

#include "benchmarks/setcover/data.h"
#include "benchmarks/setcover/setcover_model.h"
#include "benchmarks/setcover/verify_setcover.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/dag_ops.h>

#include <cstdio>
#include <sstream>
#include <string>

using namespace cbls;
using namespace cbls::setcover;

namespace {

// 3 rows, 4 columns. Column 0 covers everything for 5; columns 1-3 cover one
// row each for 1. Optimum is 3 (columns 1,2,3); the coverage-greedy choice
// (column 0) costs 5, so the search cannot reach the optimum by accident.
const char* kTinyText = "3 4\n"
                        "5 1 1 1\n"
                        "2\n1 2\n"
                        "2\n1 3\n"
                        "2\n1 4\n";

SetCoverInstance tiny() {
    std::istringstream in(kTinyText);
    return parse_setcover(in, "tiny");
}

const char* kScpe1Path = "benchmarks/instances/setcover/scpe1.txt";
constexpr double kScpe1Optimum = 5.0;  // proven; see benchmarks/instances/setcover/README.md

}  // namespace

TEST_CASE("set cover parser reads the OR-Library format", "[setcover]") {
    SetCoverInstance inst = tiny();

    REQUIRE(inst.rows == 3);
    REQUIRE(inst.cols == 4);
    REQUIRE(inst.cost == std::vector<double>{5, 1, 1, 1});
    REQUIRE(inst.nonzeros() == 6);
    // Indices are stored 0-based; the file is 1-based.
    REQUIRE(inst.row_cols[0] == std::vector<int>{0, 1});
    REQUIRE(inst.row_cols[2] == std::vector<int>{0, 3});
    REQUIRE(inst.covers_row(0, 0));
    REQUIRE(inst.covers_row(0, 1));
    REQUIRE_FALSE(inst.covers_row(0, 2));
}

TEST_CASE("set cover parser rejects malformed instances", "[setcover]") {
    SECTION("truncated") {
        std::istringstream in("3 4\n5 1 1 1\n2\n1 2\n");
        REQUIRE_THROWS(parse_setcover(in, "truncated"));
    }
    SECTION("column index out of range") {
        std::istringstream in("1 2\n1 1\n1\n5\n");
        REQUIRE_THROWS(parse_setcover(in, "bad-index"));
    }
    SECTION("row covered by nothing") {
        std::istringstream in("1 2\n1 1\n0\n");
        REQUIRE_THROWS(parse_setcover(in, "empty-row"));
    }
}

TEST_CASE("set cover check recomputes cost and finds uncovered rows", "[setcover]") {
    SetCoverInstance inst = tiny();

    CoverCheck full = check_cover(inst, {1, 2, 3});
    REQUIRE(full.covered);
    REQUIRE(full.cost == 3.0);
    REQUIRE(full.uncovered_rows == 0);

    CoverCheck partial = check_cover(inst, {1});
    REQUIRE_FALSE(partial.covered);
    REQUIRE(partial.uncovered_rows == 2);
    REQUIRE(partial.cost == 1.0);

    // A repeated column is neither paid for twice nor treated as extra coverage.
    CoverCheck repeated = check_cover(inst, {0, 0});
    REQUIRE(repeated.covered);
    REQUIRE(repeated.cost == 5.0);
    REQUIRE(repeated.duplicate_columns == 1);
}

TEST_CASE("Set encoding expresses row coverage over the Set variable", "[setcover]") {
    SetCoverInstance inst = tiny();
    SetCoverModel scm = build_set_model(inst);
    Model& m = scm.model;

    // One Set variable, one constraint per row.
    REQUIRE(m.num_vars() == 1);
    const Variable& set_var = m.var(handle_to_var_id(scm.chosen));
    REQUIRE(set_var.type == VarType::Set);
    REQUIRE(set_var.universe_size == 4);
    // Cardinality bound is min(cols, rows): no minimal cover exceeds the row count.
    REQUIRE(set_var.max_size == 3);
    REQUIRE(m.constraint_ids().size() == 3);

    auto coverage_residual = [&](int row) { return m.node(m.constraint_ids()[row]).value; };

    // Only column 1 chosen: row 0 covered, rows 1 and 2 not. `geq` residuals are
    // <= 0 exactly when the row holds.
    m.var_mut(handle_to_var_id(scm.chosen)).elements = {1};
    full_evaluate(m);
    REQUIRE(coverage_residual(0) <= 0.0);
    REQUIRE(coverage_residual(1) > 0.0);
    REQUIRE(coverage_residual(2) > 0.0);
    REQUIRE(m.node(m.objective_id()).value == 1.0);
    REQUIRE(scm.selected_columns() == std::vector<int>{1});

    // The universal column covers every row on its own, for 5.
    m.var_mut(handle_to_var_id(scm.chosen)).elements = {0};
    full_evaluate(m);
    for (int row = 0; row < 3; ++row) {
        REQUIRE(coverage_residual(row) <= 0.0);
    }
    REQUIRE(m.node(m.objective_id()).value == 5.0);
}

TEST_CASE("Bool encoding expresses the same instance", "[setcover]") {
    SetCoverInstance inst = tiny();
    SetCoverModel scm = build_bool_model(inst);
    Model& m = scm.model;

    REQUIRE(m.num_vars() == 4);
    REQUIRE(m.constraint_ids().size() == 3);

    for (int32_t handle : scm.x) {
        m.var_mut(handle_to_var_id(handle)).value = 0.0;
    }
    m.var_mut(handle_to_var_id(scm.x[0])).value = 1.0;
    full_evaluate(m);
    for (int row = 0; row < 3; ++row) {
        REQUIRE(m.node(m.constraint_ids()[row]).value <= 0.0);
    }
    REQUIRE(m.node(m.objective_id()).value == 5.0);
    REQUIRE(scm.selected_columns() == std::vector<int>{0});
}

TEST_CASE("verifier rejects an uncovered assignment", "[setcover]") {
    SetCoverInstance inst = tiny();
    SetCoverModel scm = build_set_model(inst);

    scm.model.var_mut(handle_to_var_id(scm.chosen)).elements = {1};
    full_evaluate(scm.model);
    VerifyResult bad = verify_setcover(scm, inst);
    REQUIRE_FALSE(bad.ok);

    scm.model.var_mut(handle_to_var_id(scm.chosen)).elements = {1, 2, 3};
    full_evaluate(scm.model);
    REQUIRE(verify_setcover(scm, inst).ok);
}

// The Set encoding finds a cover but NOT the optimum, even on a 3x4 instance:
// from the universal column {0} no single add, remove or swap improves (each
// uncovers a row), and a Set-only model has no diversification — FJ, Novelty
// Jump and `perturb` all move scalar variables only. So the search stalls at 5
// where the Bool encoding below reaches 3. The assertion is therefore the
// invariant (a real cover, never below the optimum), not the optimum; the gap
// itself is measured in benchmarks/instances/setcover/README.md.
TEST_CASE("Set encoding covers the tiny instance", "[setcover]") {
    SetCoverInstance inst = tiny();
    SetCoverModel scm = build_set_model(inst);

    SearchResult result = solve_deterministic(scm.model, 20000);
    CoverCheck check = check_cover(inst, scm.selected_columns());

    REQUIRE(result.feasible);
    REQUIRE(check.covered);
    REQUIRE(verify_setcover(scm, inst).ok);
    REQUIRE(check.cost >= 3.0);  // 3 is proven optimal: below it means a broken model
}

TEST_CASE("Bool encoding solves the tiny instance to the optimum", "[setcover]") {
    SetCoverInstance inst = tiny();
    SetCoverModel scm = build_bool_model(inst);

    SearchResult result = solve_deterministic(scm.model, 20000);
    CoverCheck check = check_cover(inst, scm.selected_columns());

    REQUIRE(result.feasible);
    REQUIRE(check.covered);
    REQUIRE(verify_setcover(scm, inst).ok);
    REQUIRE(check.cost == 3.0);
}

// The real-instance test. Nothing here asserts a competitive objective — the Set
// encoding is well short of the optimum (see the benchmark README) — but a
// solution must be a genuine cover, and it must never come in BELOW a proven
// optimum, which would mean the model or the verifier is wrong.
TEST_CASE("Set encoding returns a verified cover on scpe1", "[setcover]") {
    SetCoverInstance inst = load_setcover(kScpe1Path);
    REQUIRE(inst.rows == 50);
    REQUIRE(inst.cols == 500);

    SetCoverModel scm = build_set_model(inst);
    // A Set-only model spends most GLS iterations in FJ batches that have no
    // scalar variable to jump (they only pump GLS weights), so an iteration
    // budget buys far more batches here than on a scalar model — 2e6 iterations
    // is well under a second.
    SearchResult result = solve_deterministic(scm.model, 2000000);
    CoverCheck check = check_cover(inst, scm.selected_columns());

    printf("\nscpe1 (Set encoding): obj=%.0f optimum=%.0f cols=%d iters=%lld\n", check.cost,
           kScpe1Optimum, check.num_columns, static_cast<long long>(result.iterations));

    REQUIRE(result.feasible);
    REQUIRE(check.covered);
    REQUIRE(verify_setcover(scm, inst).ok);
    REQUIRE(check.cost >= kScpe1Optimum);
}

TEST_CASE("Bool encoding returns a verified cover on scpe1", "[setcover]") {
    SetCoverInstance inst = load_setcover(kScpe1Path);
    SetCoverModel scm = build_bool_model(inst);

    // 500 Bool variables and a 500-term objective row: each GLS iteration costs
    // far more here than in the Set model above, so the budget is smaller.
    SearchResult result = solve_deterministic(scm.model, 2000);
    CoverCheck check = check_cover(inst, scm.selected_columns());

    printf("\nscpe1 (Bool encoding): obj=%.0f optimum=%.0f cols=%d iters=%lld\n", check.cost,
           kScpe1Optimum, check.num_columns, static_cast<long long>(result.iterations));

    REQUIRE(result.feasible);
    REQUIRE(check.covered);
    REQUIRE(verify_setcover(scm, inst).ok);
    REQUIRE(check.cost >= kScpe1Optimum);
}
