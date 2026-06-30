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
