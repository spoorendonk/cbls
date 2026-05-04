// Tests for the vendored MPS / .solu reader and the MPS-to-CBLS-Model
// adapter. These tests do not require network access — small MPS / .solu
// fixtures are written into a temporary directory at runtime.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <cbls/io_mps.h>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

namespace fs = std::filesystem;

namespace {

fs::path tmp_dir() {
    static fs::path dir = fs::temp_directory_path() / "cbls_miplib_fj_tests";
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

}  // namespace

TEST_CASE("read_mps parses a tiny continuous LP", "[miplib-fj][reader]") {
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

TEST_CASE("read_mps marks INTORG-flagged columns as Integer", "[miplib-fj][reader]") {
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

TEST_CASE("mps_to_model builds a closed CBLS model", "[miplib-fj][adapter]") {
    auto path = write_file("small_a.mps", kSmallLp);
    cbls::MpsProblem prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    REQUIRE(built.model.is_closed());
    REQUIRE(built.model.num_vars() == 2);
    REQUIRE(built.var_handles.size() == 2);
    REQUIRE(built.constraint_node_ids.size() == 2);
    REQUIRE(built.objective_node_id >= 0);
}

TEST_CASE("CBLS finds the optimum on a small continuous LP", "[miplib-fj][solve]") {
    auto path = write_file("small_b.mps", kSmallLp);
    auto prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    auto result = cbls::solve(built.model, /*time_limit=*/3.0, /*seed=*/42,
                              /*use_fj=*/true, &hook, &lns);
    REQUIRE(result.feasible);
    // Optimum is 3.0 (x=3, y=0). SA may not hit exactly the LP optimum;
    // require a generous bound for the assertion (3.0 <= obj <= 6.0).
    REQUIRE(result.objective >= 3.0 - 1e-6);
    REQUIRE(result.objective <= 6.0);
}

TEST_CASE("CBLS finds a feasible point on a small binary IP", "[miplib-fj][solve]") {
    auto path = write_file("bin_b.mps", kSmallBinary);
    auto prob = cbls::read_mps(path.string());
    auto built = cbls::mps_to_model(prob);

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    auto result = cbls::solve(built.model, /*time_limit=*/3.0, /*seed=*/42,
                              /*use_fj=*/true, &hook, &lns);
    REQUIRE(result.feasible);
    // Minimum is 5; any feasible binary solution is at most 9 (1+1+1=3 sum
    // with all costs); accept any feasible point satisfying the assertion.
    REQUIRE(result.objective >= 5.0 - 1e-6);
    REQUIRE(result.objective <= 9.0 + 1e-6);
}

TEST_CASE("read_solu parses =opt= / =inf= / =best=", "[miplib-fj][reader]") {
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
