// Tests for the vendored MPS / .solu reader and the MPS-to-CBLS-Model
// adapter. These tests do not require network access — small MPS / .solu
// fixtures are written into a temporary directory at runtime.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using Catch::Matchers::WithinAbs;

namespace fs = std::filesystem;

namespace {

fs::path tmp_dir() {
    static fs::path dir = fs::temp_directory_path() / "cbls_mps_tests";
    std::error_code ec;
    fs::create_directories(dir, ec);
    return dir;
}

fs::path write_file(const std::string& name, const std::string& content) {
    fs::path p = tmp_dir() / name;
    std::ofstream f(p);
    f << content;
    return p;
}

// Tiny LP / IP, both feasible and bounded:
//   min  x + 2 y
//   s.t. x + y >= 3
//        x - y <= 5
//        0 <= x <= 10
//        0 <= y <= 10
// Optimum: x=3, y=0 -> obj 3.
const std::string kSmallLp =
    "NAME          SMALL\n"
    "ROWS\n"
    " N  COST\n"
    " G  C1\n"
    " L  C2\n"
    "COLUMNS\n"
    "    X    COST    1.0   C1    1.0\n"
    "    X    C2      1.0\n"
    "    Y    COST    2.0   C1    1.0\n"
    "    Y    C2     -1.0\n"
    "RHS\n"
    "    RHS  C1      3.0   C2    5.0\n"
    "BOUNDS\n"
    " UP BND  X      10.0\n"
    " UP BND  Y      10.0\n"
    "ENDATA\n";

// Pure-binary fixture exercising the MARKER 'INTORG'/'INTEND' path:
//   min   3 x1 + 2 x2 + 4 x3
//   s.t.  x1 + x2 + x3 >= 2
//   x_i in {0,1}
// Optimum: x1=0, x2=1, x3=1 (or x1=1,x2=1,x3=0) -> obj 5? let's check:
//   {x2=1,x3=1}: 0 + 2 + 4 = 6
//   {x1=1,x2=1}: 3 + 2 + 0 = 5  <- optimum
const std::string kSmallBinary =
    "NAME          BIN\n"
    "ROWS\n"
    " N  COST\n"
    " G  C1\n"
    "COLUMNS\n"
    "    MARKER1     'MARKER'                 'INTORG'\n"
    "    X1   COST   3.0   C1    1.0\n"
    "    X2   COST   2.0   C1    1.0\n"
    "    X3   COST   4.0   C1    1.0\n"
    "    MARKER2     'MARKER'                 'INTEND'\n"
    "RHS\n"
    "    RHS  C1     2.0\n"
    "BOUNDS\n"
    " BV BND  X1\n"
    " BV BND  X2\n"
    " BV BND  X3\n"
    "ENDATA\n";

// An E row carrying a RANGES entry, the only MPS shape that turns one row into
// TWO constraints. `range > 0` means [rhs, rhs + range]; `range < 0` means
// [rhs + range, rhs]. Nothing in the suite covered RANGES at all until the
// adapter's per-row builder was split out, and the split silently reordered
// this arm's node creation -- see the regression test below for why that
// matters.
//   min  x
//   s.t. 2 <= x + y <= 5      (E row, rhs 2, range +3)
//        0 <= x, y <= 10
const std::string kRangedEqualityRow =
    "NAME          RNGE\n"
    "ROWS\n"
    " N  COST\n"
    " E  R1\n"
    "COLUMNS\n"
    "    X    COST    1.0   R1    1.0\n"
    "    Y    R1      1.0\n"
    "RHS\n"
    "    RHS  R1      2.0\n"
    "RANGES\n"
    "    RNG  R1      3.0\n"
    "BOUNDS\n"
    " UP BND  X      10.0\n"
    " UP BND  Y      10.0\n"
    "ENDATA\n";

// The same row with a negative range: [rhs + range, rhs] = [-1, 2].
const std::string kRangedEqualityRowNegative =
    "NAME          RNGN\n"
    "ROWS\n"
    " N  COST\n"
    " E  R1\n"
    "COLUMNS\n"
    "    X    COST    1.0   R1    1.0\n"
    "    Y    R1      1.0\n"
    "RHS\n"
    "    RHS  R1      2.0\n"
    "RANGES\n"
    "    RNG  R1     -3.0\n"
    "BOUNDS\n"
    " UP BND  X      10.0\n"
    " UP BND  Y      10.0\n"
    "ENDATA\n";

}  // namespace

TEST_CASE("read_mps parses a tiny continuous LP", "[mps][reader]") {
    auto path = write_file("small.mps", kSmallLp);
    cbls::MpsProblem prob = cbls::read_mps(path.string());

    REQUIRE(prob.name == "SMALL");
    REQUIRE(prob.vars.size() == 2);
    REQUIRE(prob.rows.size() == 2);
    REQUIRE(prob.objective_row_name == "COST");

    // Variable bounds.
    for (const auto& v : prob.vars) {
        REQUIRE(v.kind == cbls::MpsVarKind::Continuous);
        REQUIRE(v.lb == 0.0);
        REQUIRE_THAT(v.ub, WithinAbs(10.0, 1e-12));
    }

    // Row senses + RHS.
    REQUIRE(prob.rows[0].sense == cbls::MpsRowSense::G);
    REQUIRE_THAT(prob.rows[0].rhs, WithinAbs(3.0, 1e-12));
    REQUIRE(prob.rows[1].sense == cbls::MpsRowSense::L);
    REQUIRE_THAT(prob.rows[1].rhs, WithinAbs(5.0, 1e-12));

    // Nonzero count: 4 in matrix + 2 in objective = 6.
    REQUIRE(prob.nonzeros.size() == 6);
}

TEST_CASE("read_mps marks INTORG-flagged columns as Integer", "[mps][reader]") {
    auto path = write_file("bin.mps", kSmallBinary);
    cbls::MpsProblem prob = cbls::read_mps(path.string());

    REQUIRE(prob.vars.size() == 3);
    for (const auto& v : prob.vars) {
        // BV bound also overrides Integer -> Binary.
        REQUIRE(v.kind == cbls::MpsVarKind::Binary);
        REQUIRE(v.lb == 0.0);
        REQUIRE(v.ub == 1.0);
    }
}

TEST_CASE("mps_to_model builds a closed CBLS model", "[mps][adapter]") {
    auto path = write_file("small_a.mps", kSmallLp);
    cbls::MpsProblem prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    REQUIRE(built.model.is_closed());
    REQUIRE(built.model.num_vars() == 2);
    REQUIRE(built.var_handles.size() == 2);
    REQUIRE(built.constraint_node_ids.size() == 2);
    REQUIRE(built.objective_node_id >= 0);
}

TEST_CASE("CBLS finds the optimum on a small continuous LP", "[mps][solve]") {
    auto path = write_file("small_b.mps", kSmallLp);
    auto prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    auto result = solve_deterministic(built.model, 1010000, 42, &hook, &lns);
    REQUIRE(result.feasible);
    // Optimum is 3.0 (x=3, y=0). SA may not hit exactly the LP optimum;
    // require a generous bound for the assertion (3.0 <= obj <= 6.0).
    REQUIRE(result.objective >= 3.0 - 1e-6);
    REQUIRE(result.objective <= 6.0);
}

TEST_CASE("CBLS finds a feasible point on a small binary IP", "[mps][solve]") {
    auto path = write_file("bin_b.mps", kSmallBinary);
    auto prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    auto result = solve_deterministic(built.model, 1486000, 42, &hook, &lns);
    REQUIRE(result.feasible);
    // Minimum is 5; any feasible binary solution is at most 9 (1+1+1=3 sum
    // with all costs); accept any feasible point satisfying the assertion.
    REQUIRE(result.objective >= 5.0 - 1e-6);
    REQUIRE(result.objective <= 9.0 + 1e-6);
}

TEST_CASE("model at known optimum has zero violation and matching objective", "[mps][adapter]") {
    // Issue #71 acceptance criterion: the MPS-to-Model adapter should produce
    // a closed CBLS model whose total_violation matches the LP residual when
    // fed the optimum from .solu.
    auto path = write_file("small_opt.mps", kSmallLp);
    auto prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    // Known optimum of kSmallLp: x = 3, y = 0, obj = 3.
    REQUIRE(built.var_handles.size() == 2);
    built.model.var_mut(vid(built.var_handles[0])).value = 3.0;
    built.model.var_mut(vid(built.var_handles[1])).value = 0.0;
    cbls::full_evaluate(built.model);

    cbls::ViolationManager vm(built.model);
    REQUIRE(vm.is_feasible());
    REQUIRE_THAT(vm.total_violation(), WithinAbs(0.0, 1e-9));

    REQUIRE(built.objective_node_id >= 0);
    REQUIRE_THAT(built.model.node(built.objective_node_id).value, WithinAbs(3.0, 1e-9));
}

TEST_CASE("read_mps applies MPS integer-default ub=1 for unbounded integers", "[mps][reader]") {
    // INTORG without any UP/UI/BV defaults to ub=1 (CPLEX/Gurobi/SCIP).
    const std::string content =
        "NAME          INTDEFAULT\n"
        "ROWS\n"
        " N  COST\n"
        " G  C1\n"
        "COLUMNS\n"
        "    MARKER1     'MARKER'                 'INTORG'\n"
        "    Z    COST   1.0   C1   1.0\n"
        "    MARKER2     'MARKER'                 'INTEND'\n"
        "RHS\n"
        "    RHS  C1     0.0\n"
        "ENDATA\n";
    auto path = write_file("intdefault.mps", content);
    auto prob = cbls::read_mps(path.string());
    REQUIRE(prob.vars.size() == 1);
    REQUIRE(prob.vars[0].kind == cbls::MpsVarKind::Integer);
    REQUIRE(prob.vars[0].lb == 0.0);
    REQUIRE_THAT(prob.vars[0].ub, WithinAbs(1.0, 1e-12));
}

TEST_CASE("read_mps leaves an LI-bounded integer column unbounded above", "[mps][reader]") {
    // The integer default is "no BOUNDS entry at all", not "no upper bound". `LI`
    // sets a lower bound and says nothing about the upper, so the column stays a
    // general integer. Applying ub=1 here binarises the model: on MIPLIB it makes
    // gen-ip054 and enlight_hard infeasible and moves gen-ip002's optimum, each of
    // which then reads as a solver failure rather than a reader bug.
    const std::string content =
        "NAME          LIBOUND\n"
        "ROWS\n"
        " N  COST\n"
        " G  C1\n"
        "COLUMNS\n"
        "    MARKER1     'MARKER'                 'INTORG'\n"
        "    Z    COST   1.0   C1   1.0\n"
        "    W    COST   1.0   C1   1.0\n"
        "    MARKER2     'MARKER'                 'INTEND'\n"
        "RHS\n"
        "    RHS  C1     0.0\n"
        "BOUNDS\n"
        " LI bnd  Z      0.0\n"
        "ENDATA\n";
    auto path = write_file("libound.mps", content);
    auto prob = cbls::read_mps(path.string());
    REQUIRE(prob.vars.size() == 2);

    // Z appeared in BOUNDS: no upper bound was given, so it keeps +inf.
    REQUIRE(prob.vars[0].name == "Z");
    REQUIRE(prob.vars[0].kind == cbls::MpsVarKind::Integer);
    REQUIRE(prob.vars[0].lb == 0.0);
    REQUIRE(prob.vars[0].ub == cbls::kMpsInf);

    // W never appeared in BOUNDS, so the ub=1 default still applies to it.
    REQUIRE(prob.vars[1].name == "W");
    REQUIRE(prob.vars[1].kind == cbls::MpsVarKind::Integer);
    REQUIRE_THAT(prob.vars[1].ub, WithinAbs(1.0, 1e-12));
}

TEST_CASE("read_mps records OBJSENSE and adapter rejects MAX", "[mps][reader]") {
    const std::string content =
        "NAME          MAX\n"
        "OBJSENSE\n"
        "    MAX\n"
        "ROWS\n"
        " N  COST\n"
        " L  C1\n"
        "COLUMNS\n"
        "    X    COST   1.0   C1   1.0\n"
        "RHS\n"
        "    RHS  C1     5.0\n"
        "BOUNDS\n"
        " UP BND  X      10.0\n"
        "ENDATA\n";
    auto path = write_file("max.mps", content);
    auto prob = cbls::read_mps(path.string());
    REQUIRE(prob.maximize);
    REQUIRE_THROWS_AS(cbls::mps_to_model(prob), std::runtime_error);
}

TEST_CASE("read_mps rejects non-finite coefficients", "[mps][reader]") {
    const std::string content =
        "NAME          NAN\n"
        "ROWS\n"
        " N  COST\n"
        " L  C1\n"
        "COLUMNS\n"
        "    X    COST   inf   C1   1.0\n"
        "RHS\n"
        "    RHS  C1     5.0\n"
        "ENDATA\n";
    auto path = write_file("nonfinite.mps", content);
    REQUIRE_THROWS_AS(cbls::read_mps(path.string()), std::runtime_error);
}

TEST_CASE("read_solu parses =opt= / =inf= / =best=", "[mps][reader]") {
    std::string content =
        "# comment line\n"
        "=opt=  inst_a   42.5\n"
        "=opt=  inst_b   -100\n"
        "=inf=  bad_inst\n"
        "=best= inst_c   7.0\n"
        "=feas= inst_d   3.14\n";
    auto path = write_file("test.solu", content);
    auto entries = cbls::read_solu(path.string());

    REQUIRE(entries.size() == 5);
    REQUIRE(entries[0].name == "inst_a");
    REQUIRE(entries[0].is_optimal);
    REQUIRE_THAT(entries[0].value, WithinAbs(42.5, 1e-12));
    REQUIRE(entries[1].name == "inst_b");
    REQUIRE(entries[1].is_optimal);
    REQUIRE_THAT(entries[1].value, WithinAbs(-100.0, 1e-12));
    REQUIRE(entries[2].name == "bad_inst");
    REQUIRE(entries[2].is_infeasible);
    REQUIRE_FALSE(entries[2].is_optimal);
    REQUIRE(entries[3].name == "inst_c");
    REQUIRE_FALSE(entries[3].is_optimal);
    REQUIRE_THAT(entries[3].value, WithinAbs(7.0, 1e-12));
    REQUIRE(entries[4].name == "inst_d");
    REQUIRE_FALSE(entries[4].is_optimal);
    REQUIRE_THAT(entries[4].value, WithinAbs(3.14, 1e-12));
}

TEST_CASE("RANGES on an E row becomes a two-sided constraint", "[mps][adapter]") {
    SECTION("a positive range gives [rhs, rhs + range]") {
        auto path = write_file("ranged_e.mps", kRangedEqualityRow);
        auto res = cbls::mps_to_model(cbls::read_mps(path.string()));
        REQUIRE(res.var_handles.size() == 2);
        cbls::ViolationManager vm(res.model);

        auto violation_at = [&](double x, double y) {
            res.model.var_mut(vid(res.var_handles[0])).value = x;
            res.model.var_mut(vid(res.var_handles[1])).value = y;
            cbls::full_evaluate(res.model);
            vm.invalidate_cache();
            return vm.total_violation();
        };

        // Both boundaries of [2, 5] are feasible, and so is a point inside.
        REQUIRE_THAT(violation_at(2.0, 0.0), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(violation_at(3.5, 0.0), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(violation_at(5.0, 0.0), WithinAbs(0.0, 1e-9));
        // Below the range and above it are both violations -- a plain E row
        // would have reported 3.5 and 5.0 as violations too.
        REQUIRE(violation_at(1.5, 0.0) > 1e-9);
        REQUIRE(violation_at(5.5, 0.0) > 1e-9);
    }

    SECTION("a negative range gives [rhs + range, rhs]") {
        auto path = write_file("ranged_e_neg.mps", kRangedEqualityRowNegative);
        auto res = cbls::mps_to_model(cbls::read_mps(path.string()));
        REQUIRE(res.var_handles.size() == 2);
        cbls::ViolationManager vm(res.model);

        auto violation_at = [&](double x, double y) {
            res.model.var_mut(vid(res.var_handles[0])).value = x;
            res.model.var_mut(vid(res.var_handles[1])).value = y;
            cbls::full_evaluate(res.model);
            vm.invalidate_cache();
            return vm.total_violation();
        };

        // [-1, 2]. The columns are bounded below at 0, so 0 is the lowest sum
        // reachable here; it is inside the range, and 2 is its upper boundary.
        REQUIRE_THAT(violation_at(0.0, 0.0), WithinAbs(0.0, 1e-9));
        REQUIRE_THAT(violation_at(2.0, 0.0), WithinAbs(0.0, 1e-9));
        REQUIRE(violation_at(3.0, 0.0) > 1e-9);
    }
}

TEST_CASE("a ranged E row creates its upper bound before the geq node", "[mps][adapter]") {
    // Node ids order the work the engine does, so the order in which a row's
    // nodes are created is part of the model, not an implementation detail.
    // Splitting the adapter's per-row builder out of mps_to_model moved this
    // arm's `m.constant(rhs + range)` to AFTER its `m.geq(...)`, which shifts
    // every later node id on such a row. Nothing caught it: the model is
    // otherwise identical and the whole suite stayed green. This pins the
    // order.
    //
    // The row's constraint node is the `leq`; the two nodes immediately before
    // it must be the `geq` and, before that, the constant holding rhs + range.
    auto path = write_file("ranged_e_order.mps", kRangedEqualityRow);
    auto res = cbls::mps_to_model(cbls::read_mps(path.string()));
    const auto& m = res.model;

    const int32_t leq_id = res.constraint_node_ids[0];
    REQUIRE(leq_id >= 2);
    REQUIRE(m.node(leq_id).op == cbls::NodeOp::Leq);
    REQUIRE(m.node(leq_id - 1).op == cbls::NodeOp::Geq);
    REQUIRE(m.node(leq_id - 2).op == cbls::NodeOp::Const);
    REQUIRE_THAT(m.node(leq_id - 2).const_value, WithinAbs(5.0, 1e-12));
}
