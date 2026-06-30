// Network-free tests for the AMPL NL text reader (issue #72, P3) and the
// NL->Model adapter (P4). Fixtures are tiny inline `g3` NL files; no instances
// are downloaded. Fixture layout mirrors a real minlplib `.nl` (header block,
// then b / x / r / J / O / C segments).

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <cmath>
#include <string>

using namespace cbls;
using Catch::Matchers::WithinAbs;

// A minimal but realistic header for `nv` vars, `nc` cons, `no` objs. The body
// segments follow. Counts beyond the first line are cosmetic for the reader
// (it only reads nvars/ncons/nobjs), so we keep them plausible but simple.
static std::string nl_header(int nv, int nc, int no) {
    std::string h = "g3 0 1 0\t# header\n";
    h += " " + std::to_string(nv) + " " + std::to_string(nc) + " " + std::to_string(no) +
         " 0 0\t# vars, cons, objs, ranges, eqns\n";
    h += " 0 0\t# nonlinear cons, objs\n";
    h += " 0 0\t# network\n";
    h += " 0 0 0\t# nonlinear vars\n";
    h += " 0 0 0 1\t# linear net, funcs, arith, flags\n";
    h += " 0 0 0 0 0\t# discrete\n";
    h += " 0 0\t# jacobian/gradient nonzeros\n";
    h += " 0 0\t# max name lengths\n";
    h += " 0 0 0 0 0\t# common exprs\n";
    return h;
}

TEST_CASE("NL reader rejects binary header", "[minlplib]") {
    std::string text = "b3 0 1 0\n 1 0 1 0 0\n";
    REQUIRE_THROWS_WITH(parse_nl(text, "bin"), Catch::Matchers::ContainsSubstring("binary"));
}

TEST_CASE("NL reader parses counts and variable bounds", "[minlplib]") {
    // 2 vars, 1 constraint, 1 objective.
    std::string text = nl_header(2, 1, 1);
    text += "b\n";        // variable bounds segment
    text += "0 -2 11\n";  // var0: range [-2, 11]
    text += "2 1.5\n";    // var1: lower bound >= 1.5  (type 2)
    text += "r\n";        // constraint bounds
    text += "1 7\n";      // con0: body <= 7  (type 1 = upper)
    text += "x1\n";       // initial guess, 1 entry
    text += "0 3.0\n";    // var0 := 3.0
    text += "O0 0\n";     // objective 0, sense min
    text += "n0\n";       // trivial objective expr: constant 0

    NlProblem p = parse_nl(text, "counts");
    REQUIRE(p.n_vars == 2);
    REQUIRE(p.n_cons == 1);
    REQUIRE(p.n_objs == 1);

    REQUIRE(p.var_bounds[0].type == NlBoundType::Range);
    REQUIRE_THAT(p.var_bounds[0].lower, WithinAbs(-2.0, 1e-12));
    REQUIRE_THAT(p.var_bounds[0].upper, WithinAbs(11.0, 1e-12));
    REQUIRE(p.var_bounds[1].type == NlBoundType::Lower);
    REQUIRE_THAT(p.var_bounds[1].lower, WithinAbs(1.5, 1e-12));

    REQUIRE(p.constraints[0].bound.type == NlBoundType::Upper);
    REQUIRE_THAT(p.constraints[0].bound.upper, WithinAbs(7.0, 1e-12));

    REQUIRE_THAT(p.initial_x[0], WithinAbs(3.0, 1e-12));

    REQUIRE(p.objectives[0].maximize == false);
}

TEST_CASE("NL reader maps range and equal constraint senses", "[minlplib]") {
    std::string text = nl_header(1, 3, 0);
    text += "b\n0 0 10\n";  // var0 in [0,10]
    text += "r\n";
    text += "0 1 5\n";  // con0: range [1, 5]
    text += "4 2\n";    // con1: == 2
    text += "3\n";      // con2: free
    NlProblem p = parse_nl(text, "senses");
    REQUIRE(p.constraints[0].bound.type == NlBoundType::Range);
    REQUIRE_THAT(p.constraints[0].bound.lower, WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(p.constraints[0].bound.upper, WithinAbs(5.0, 1e-12));
    REQUIRE(p.constraints[1].bound.type == NlBoundType::Equal);
    REQUIRE_THAT(p.constraints[1].bound.lower, WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(p.constraints[1].bound.upper, WithinAbs(2.0, 1e-12));
    REQUIRE(p.constraints[2].bound.type == NlBoundType::Free);
}

TEST_CASE("NL reader parses a nonlinear objective expression", "[minlplib]") {
    // minimize: o5 (pow) v0 n2  ==  v0 ^ 2.
    std::string text = nl_header(1, 0, 1);
    text += "b\n0 -5 5\n";
    text += "O0 0\n";
    text += "o5\n";  // OPPOW
    text += "v0\n";
    text += "n2\n";
    NlProblem p = parse_nl(text, "nlobj");
    REQUIRE(p.objectives[0].nonlinear.empty() == false);
    const NlExpr& e = p.objectives[0].nonlinear;
    const NlExprNode& root = e.nodes[e.root];
    REQUIRE(root.kind == NlNodeKind::Op);
    REQUIRE(root.opcode == 5);  // OPPOW
    REQUIRE(root.children.size() == 2);
    REQUIRE(e.nodes[root.children[0]].kind == NlNodeKind::Var);
    REQUIRE(e.nodes[root.children[0]].index == 0);
    REQUIRE(e.nodes[root.children[1]].kind == NlNodeKind::Num);
    REQUIRE_THAT(e.nodes[root.children[1]].num, WithinAbs(2.0, 1e-12));
}

TEST_CASE("NL reader parses linear J/G segments", "[minlplib]") {
    std::string text = nl_header(2, 1, 1);
    text += "b\n0 0 10\n0 0 10\n";
    text += "r\n1 8\n";           // con0 <= 8
    text += "J0 2\n0 3\n1 -1\n";  // con0 linear: 3*x0 - 1*x1
    text += "O0 0\n";
    text += "n0\n";              // obj nonlinear part: 0
    text += "G0 2\n0 1\n1 1\n";  // obj linear: x0 + x1
    NlProblem p = parse_nl(text, "lin");
    REQUIRE(p.constraints[0].linear.size() == 2);
    REQUIRE(p.constraints[0].linear[0].var == 0);
    REQUIRE_THAT(p.constraints[0].linear[0].coef, WithinAbs(3.0, 1e-12));
    REQUIRE(p.constraints[0].linear[1].var == 1);
    REQUIRE_THAT(p.constraints[0].linear[1].coef, WithinAbs(-1.0, 1e-12));
    REQUIRE(p.objectives[0].linear.size() == 2);
}

TEST_CASE("NL reader sums the discrete-variable header line", "[minlplib]") {
    // The 7th header line is "nbv niv nlvbi nlvci nlvoi"; n_discrete_vars is the
    // sum. nl_header() writes "0 0 0 0 0" on that line, so a default fixture has
    // zero discrete vars. Build a custom header with a nonzero discrete line.
    std::string text = "g3 0 1 0\t# header\n";
    text += " 3 0 1 0 0\t# vars, cons, objs, ranges, eqns\n";
    text += " 0 0\n";
    text += " 0 0\n";
    text += " 0 0 0\n";
    text += " 0 0 0 1\n";
    text += " 0 0 2 1 0\t# discrete: nbv niv nlvbi nlvci nlvoi -> 3 total\n";
    text += " 0 0\n";
    text += " 0 0\n";
    text += " 0 0 0 0 0\n";
    text += "b\n0 0 10\n0 0 10\n0 0 10\n";
    text += "O0 0\nn0\n";
    NlProblem p = parse_nl(text, "discrete");
    REQUIRE(p.n_discrete_vars == 3);

    // The default fixture (all-zero discrete line) reports zero.
    NlProblem q = parse_nl(nl_header(1, 0, 1) + "b\n0 0 1\nO0 0\nn0\n", "cont");
    REQUIRE(q.n_discrete_vars == 0);
}

TEST_CASE("NL reader throws a tagged marker on an unsupported opcode", "[minlplib]") {
    // OPPLTERM (64, piecewise) has a non-standard payload; the reader can't know
    // its arity and must throw the NL_UNKNOWN_OPCODE marker rather than crash.
    std::string text = nl_header(1, 0, 1);
    text += "b\n0 0 10\n";
    text += "O0 0\n";
    text += "o64\n";  // OPPLTERM
    text += "v0\n";
    REQUIRE_THROWS_WITH(parse_nl(text, "unsupported"),
                        Catch::Matchers::ContainsSubstring("NL_UNKNOWN_OPCODE"));
}

// ---- P4: NL -> Model adapter ----

TEST_CASE("nl_to_model builds a closed model", "[minlplib]") {
    // minimize x0^2 + x1 ; s.t. x0 + x1 >= 3 ; x in [0,10]^2.
    std::string text = nl_header(2, 1, 1);
    text += "b\n0 0 10\n0 0 10\n";
    text += "r\n2 3\n";          // con0: body >= 3
    text += "J0 2\n0 1\n1 1\n";  // con0 linear: x0 + x1
    text += "O0 0\n";
    text += "o5\nv0\nn2\n";  // obj nonlinear: x0^2
    text += "G0 1\n1 1\n";   // obj linear: x1
    NlProblem p = parse_nl(text, "build");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    REQUIRE(r.model.is_closed());
    REQUIRE(r.var_handles.size() == 2);
    REQUIRE(r.objective_node_id >= 0);
    REQUIRE(r.constraint_node_ids[0] >= 0);
}

TEST_CASE("nl_to_model: feasible point has zero violation and correct objective", "[minlplib]") {
    std::string text = nl_header(2, 1, 1);
    text += "b\n0 0 10\n0 0 10\n";
    text += "r\n2 3\n";  // x0 + x1 >= 3
    text += "J0 2\n0 1\n1 1\n";
    text += "O0 0\n";
    text += "o5\nv0\nn2\n";  // x0^2
    text += "G0 1\n1 1\n";   // + x1
    NlProblem p = parse_nl(text, "feas");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);

    // Set a feasible point: x0=2, x1=2 -> x0+x1=4 >= 3 OK; obj = 4 + 2 = 6.
    r.model.var_mut(0).value = 2.0;
    r.model.var_mut(1).value = 2.0;
    full_evaluate(r.model);

    ViolationManager vm(r.model);
    vm.invalidate_cache();
    REQUIRE_THAT(vm.total_violation(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(r.model.node(r.objective_node_id).value, WithinAbs(6.0, 1e-9));

    // Infeasible point: x0=0, x1=0 -> x0+x1=0 < 3, violation = 3.
    r.model.var_mut(0).value = 0.0;
    r.model.var_mut(1).value = 0.0;
    full_evaluate(r.model);
    vm.invalidate_cache();
    REQUIRE(vm.total_violation() > 0.0);
}

TEST_CASE("nl_to_model handles maximize sense", "[minlplib]") {
    // maximize x0 ; x0 in [0, 5]. CBLS minimizes the negated objective, so the
    // objective node value at x0=5 should be -5.
    std::string text = nl_header(1, 0, 1);
    text += "b\n0 0 5\n";
    text += "O0 1\n";       // sense 1 = maximize
    text += "n0\n";         // nonlinear part 0
    text += "G0 1\n0 1\n";  // linear: x0
    NlProblem p = parse_nl(text, "max");
    REQUIRE(p.objectives[0].maximize == true);
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);

    r.model.var_mut(0).value = 5.0;
    full_evaluate(r.model);
    REQUIRE_THAT(r.model.node(r.objective_node_id).value, WithinAbs(-5.0, 1e-9));
}

TEST_CASE("nl_to_model reports unsupported operator without throwing", "[minlplib]") {
    // OP_atan (49) is parseable (unary) but not in the CBLS supported set, so
    // the adapter must return supported=false with a reason, not throw.
    std::string text = nl_header(1, 0, 1);
    text += "b\n0 0 10\n";
    text += "O0 0\n";
    text += "o49\nv0\n";  // atan(x0)
    NlProblem p = parse_nl(text, "atan");
    NlToModelResult r = nl_to_model(p);
    REQUIRE_FALSE(r.supported);
    REQUIRE_FALSE(r.skipped_reasons.empty());
    REQUIRE(r.skipped_reasons[0].find("49") != std::string::npos);
}

TEST_CASE("nl_to_model maps OP1POW (base^const) to pow", "[minlplib]") {
    // OP1POW (76) is base ^ constant-exponent. With exponent 3 and x=2 -> 8.
    std::string text = nl_header(1, 0, 1);
    text += "b\n0 -5 5\n";
    text += "O0 0\n";
    text += "o76\nv0\nn3\n";  // x0 ^ 3
    NlProblem p = parse_nl(text, "p1pow");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    r.model.var_mut(0).value = 2.0;
    full_evaluate(r.model);
    REQUIRE_THAT(r.model.node(r.objective_node_id).value, WithinAbs(8.0, 1e-9));
}
