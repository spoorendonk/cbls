// Network-free tests for the AMPL NL text reader (issue #72, P3), the
// NL->Model adapter (P4), and one end-to-end solve of a vendored non-convex
// instance (#124). Reader/adapter fixtures are tiny inline `g3` NL files; no
// instances are downloaded. Fixture layout mirrors a real minlplib `.nl`
// (header block, then b / x / r / J / O / C segments).

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <cmath>
#include <cstdint>
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

// Header with explicit nonlinear-variable counts ("nlvc nlvo nlvb") and discrete
// counts ("nbv niv nlvbi nlvci nlvoi") so integrality *placement* can be tested.
static std::string nl_header_disc(int nv, int nc, int no, int nlvc, int nlvo, int nlvb, int nbv,
                                  int niv, int nlvbi, int nlvci, int nlvoi) {
    std::string h = "g3 0 1 0\t# header\n";
    h += " " + std::to_string(nv) + " " + std::to_string(nc) + " " + std::to_string(no) +
         " 0 0\t# vars, cons, objs, ranges, eqns\n";
    h += " 0 0\t# nonlinear cons, objs\n";
    h += " 0 0\t# network\n";
    h += " " + std::to_string(nlvc) + " " + std::to_string(nlvo) + " " + std::to_string(nlvb) +
         "\t# nonlinear vars in cons, objs, both\n";
    h += " 0 0 0 1\t# linear net, funcs, arith, flags\n";
    h += " " + std::to_string(nbv) + " " + std::to_string(niv) + " " + std::to_string(nlvbi) + " " +
         std::to_string(nlvci) + " " + std::to_string(nlvoi) + "\t# discrete\n";
    h += " 0 0\t# jacobian/gradient nonzeros\n";
    h += " 0 0\t# max name lengths\n";
    h += " 0 0 0 0 0\t# common exprs\n";
    return h;
}

static std::string bounds_segment(int nv) {
    std::string s = "b\n";
    for (int i = 0; i < nv; ++i) {
        s += "0 0 10\n";
    }
    return s;
}

TEST_CASE("NL reader recovers discrete-variable count and positions", "[minlplib]") {
    SECTION("integers are the trailing columns of each nonlinear block") {
        // 4 vars. nlvb=2 (cols 0,1) with the last nlvbi=1 integer -> col 1.
        // nlvc-nlvb=1 (col 2) with the last nlvci=1 integer -> col 2.
        // col 3 is purely linear and continuous.
        std::string text = nl_header_disc(4, 0, 1, /*nlvc=*/3, /*nlvo=*/0, /*nlvb=*/2, /*nbv=*/0,
                                          /*niv=*/0, /*nlvbi=*/1, /*nlvci=*/1, /*nlvoi=*/0);
        text += bounds_segment(4) + "O0 0\nn0\n";
        NlProblem p = parse_nl(text, "disc-nonlinear");
        REQUIRE(p.n_discrete_vars == 2);
        REQUIRE(p.var_is_discrete == std::vector<uint8_t>{0, 1, 1, 0});
    }

    SECTION("purely-linear binary/integer columns are the last of the file") {
        // No nonlinear vars; nbv=1 and niv=1 -> the final two columns.
        std::string text = nl_header_disc(4, 0, 1, 0, 0, 0, /*nbv=*/1, /*niv=*/1, 0, 0, 0);
        text += bounds_segment(4) + "O0 0\nn0\n";
        NlProblem p = parse_nl(text, "disc-linear");
        REQUIRE(p.n_discrete_vars == 2);
        REQUIRE(p.var_is_discrete == std::vector<uint8_t>{0, 0, 1, 1});
    }

    SECTION("objective-only nonlinear block sits at [nlvc, nlvo)") {
        // nlvb=1 (col 0, continuous), nlvc-nlvb=1 (col 1, continuous). The
        // objective-only block is [nlvc, nlvo) = col 2 only, with the last
        // nlvoi=1 integer -> col 2. Col 3 is purely linear and continuous.
        std::string text = nl_header_disc(4, 0, 1, /*nlvc=*/2, /*nlvo=*/3, /*nlvb=*/1, 0, 0,
                                          /*nlvbi=*/0, /*nlvci=*/0, /*nlvoi=*/1);
        text += bounds_segment(4) + "O0 0\nn0\n";
        NlProblem p = parse_nl(text, "disc-obj");
        REQUIRE(p.n_discrete_vars == 1);
        REQUIRE(p.var_is_discrete == std::vector<uint8_t>{0, 0, 1, 0});
    }

    SECTION("objective-only integers survive a non-empty constraint-only block") {
        // Regression: `nlvo - nlvb` would place the block at [3, 7) on 8 columns
        // and mark col 6, while the header-count self-check still passed — so a
        // count check alone cannot catch a mis-placed block.
        // True layout: both=[0,1), cons-only=[1,3), obj-only=[3,5) -> integer col 4.
        std::string text = nl_header_disc(8, 0, 1, /*nlvc=*/3, /*nlvo=*/5, /*nlvb=*/1, 0, 0,
                                          /*nlvbi=*/0, /*nlvci=*/0, /*nlvoi=*/1);
        text += bounds_segment(8) + "O0 0\nn0\n";
        NlProblem p = parse_nl(text, "disc-obj-offset");
        REQUIRE(p.n_discrete_vars == 1);
        REQUIRE(p.var_is_discrete == std::vector<uint8_t>{0, 0, 0, 0, 1, 0, 0, 0});
    }

    SECTION("a fully continuous instance flags nothing") {
        NlProblem q = parse_nl(nl_header(1, 0, 1) + "b\n0 0 1\nO0 0\nn0\n", "cont");
        REQUIRE(q.n_discrete_vars == 0);
        REQUIRE(q.var_is_discrete == std::vector<uint8_t>{0});
    }
}

TEST_CASE("NL reader rejects a header whose discrete counts don't fit the layout", "[minlplib]") {
    // nlvbi=2 integer "nonlinear in both" variables, but nlvb=0 declares no such
    // block. Placement can't account for the declared count, so the reader must
    // fail loudly rather than build a model with silently wrong integrality.
    std::string text = nl_header_disc(3, 0, 1, /*nlvc=*/0, /*nlvo=*/0, /*nlvb=*/0, 0, 0,
                                      /*nlvbi=*/2, /*nlvci=*/1, /*nlvoi=*/0);
    text += bounds_segment(3) + "O0 0\nn0\n";
    REQUIRE_THROWS_WITH(parse_nl(text, "bad-layout"),
                        Catch::Matchers::ContainsSubstring("variable ordering"));
}

TEST_CASE("nl_to_model builds Int variables for discrete columns", "[minlplib]") {
    // cols 0,1 nonlinear-in-both with col 1 integer; col 2 nonlinear-in-cons and
    // integer; col 3 linear continuous.
    std::string text = nl_header_disc(4, 0, 1, 3, 0, 2, 0, 0, 1, 1, 0);
    text += bounds_segment(4) + "O0 0\nn0\n";
    NlProblem p = parse_nl(text, "int-model");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    REQUIRE(r.model.var(0).type == VarType::Float);
    REQUIRE(r.model.var(1).type == VarType::Int);
    REQUIRE(r.model.var(2).type == VarType::Int);
    REQUIRE(r.model.var(3).type == VarType::Float);
    // Bounds [0,10] survive the float->int narrowing exactly.
    REQUIRE_THAT(r.model.var(1).lb, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(r.model.var(1).ub, WithinAbs(10.0, 1e-12));
}

TEST_CASE("nl_to_model narrows fractional bounds inward for Int columns", "[minlplib]") {
    // A single integer column with bounds [-2.5, 3.5] admits integers [-2, 3]:
    // ceil the lower bound, floor the upper.
    std::string text = nl_header_disc(1, 0, 1, 1, 0, 1, 0, 0, /*nlvbi=*/1, 0, 0);
    text += "b\n0 -2.5 3.5\n";
    text += "O0 0\nn0\n";
    NlProblem p = parse_nl(text, "int-bounds");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    REQUIRE(r.model.var(0).type == VarType::Int);
    REQUIRE_THAT(r.model.var(0).lb, WithinAbs(-2.0, 1e-12));
    REQUIRE_THAT(r.model.var(0).ub, WithinAbs(3.0, 1e-12));
}

TEST_CASE("nl_to_model honours a large but finite Int bound", "[minlplib]") {
    // int_inf_clamp is a fallback for *infinite* bounds only. A declared finite
    // bound must survive verbatim — narrowing it would change the instance.
    std::string text = nl_header_disc(1, 0, 1, 1, 0, 1, 0, 0, /*nlvbi=*/1, 0, 0);
    text += "b\n0 0 50000000\n";  // [0, 5e7], well above int_inf_clamp (1e6)
    text += "O0 0\nn0\n";
    NlProblem p = parse_nl(text, "int-big");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    REQUIRE(r.model.var(0).type == VarType::Int);
    REQUIRE_THAT(r.model.var(0).ub, WithinAbs(5.0e7, 1e-6));
}

TEST_CASE("nl_to_model clamps an unbounded Int column to a searchable box", "[minlplib]") {
    // A free integer column must not inherit the ±1e9 float clamp.
    std::string text = nl_header_disc(1, 0, 1, 1, 0, 1, 0, 0, /*nlvbi=*/1, 0, 0);
    text += "b\n3\n";  // type 3 = free
    text += "O0 0\nn0\n";
    NlProblem p = parse_nl(text, "int-free");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    REQUIRE(r.model.var(0).type == VarType::Int);
    REQUIRE_THAT(r.model.var(0).lb, WithinAbs(-1.0e6, 1e-6));
    REQUIRE_THAT(r.model.var(0).ub, WithinAbs(1.0e6, 1e-6));
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

// ---------------------------------------------------------------------------
// End-to-end solve (issue #124).
//
// Everything above tests the reader and the adapter on inline fixtures; this
// runs the search on a real vendored instance, which is the only thing in the
// suite that would notice search quality collapsing on a non-convex MINLP.
//
// `ex4_1_8` is the instance: 2 continuous columns and one nonlinear EQUALITY
// row, which is what makes the feasible set non-convex —
//
//     min  x1^2 - 7 x1 - 12 x0    s.t.  2 x0^4 + x1 = 2,  x0 in [0,2], x1 in [0,3]
//
// so reaching feasibility means landing ON a quartic curve, not merely inside a
// box. Note what is NOT hard here: eliminating x1 leaves 4 x0^8 + 6 x0^4 -
// 12 x0 - 10 over x0 in [0,1], which is convex with a single minimum at
// x0 ~ 0.7175, so the difficulty is the equality rather than a multimodal
// objective. The instance is small enough to be fast (~0.06s for 20k GLS
// iterations) and stable: at that budget the engine lands within 0.02% of the
// published primal bound on every one of seeds 1-10 and on seed 42, reaching
// either the optimum (-16.73889) or a near-miss on the same curve (-16.73598).
//
// The gap bound is loose on purpose (5%, against a worst observed 0.017%): this
// is a floor, not a record. Be clear about what it does and does not catch —
// this instance is easy for the float jump values, so even a 100x smaller
// budget (200 iterations) stays within 0.4%, and dropping the intensification
// hook does not move it out of the band either. It is a collapse detector, not
// a quality gauge. The assertions with real teeth are feasibility, the
// objective-drift check, and the lower guard: an objective below the published
// DUAL bound cannot be produced by a correct model, so it catches a reader or
// objective-bookkeeping bug the way the mipfeas test's below_reference check
// does. Both bounds come from benchmarks/instances/minlplib/bounds.csv, which
// has no reader in the library (unlike the MIPLIB .solu the mipfeas test reads),
// so they are transcribed here with the row quoted.

TEST_CASE("MINLPLib ex4_1_8 solves within a loose gap of its published bound",
          "[minlplib][solve]") {
    // bounds.csv: ex4_1_8,other,2,1,min,-16.73889318,-16.7388932,0
    const double primal_bks = -16.73889318;
    const double dual_bound = -16.7388932;

    NlProblem prob = read_nl("benchmarks/instances/minlplib/ex4_1_8.nl");
    REQUIRE(prob.n_vars == 2);
    REQUIRE(prob.n_cons == 1);

    // `win_slack` from benchmarks/minlplib/minlplib.cpp — not its `tie_band`,
    // which that file keeps deliberately tighter — taken against the dual bound
    // rather than the primal, which differs only in the 8th digit. Below the
    // bound by less than this is feasibility-tolerance slack, not an impossible
    // objective: SCIP's own proven optimum (-16.738894589, scip_baseline.csv)
    // already sits 1.4e-6 below the 8-digit dual bound published here.
    const double band =
        std::max(1e-6 * (std::abs(dual_bound) + 1.0), 10.0 * kDefaultFeasibilityTolerance);
    const double loose_gap = 0.05 * std::abs(primal_bks);

    for (uint64_t seed : {1ULL, 2ULL, 42ULL}) {
        INFO("seed " << seed);
        NlToModelResult built = nl_to_model(prob);
        REQUIRE(built.supported);
        REQUIRE(built.objective_node_id >= 0);

        FloatIntensifyHook hook;
        // Inert at this budget (diversify() needs 100 stagnant batches of 1000
        // iterations); passed for parity with benchmarks/minlplib/minlplib.cpp.
        LNS lns(0.3);
        SearchResult result = solve_deterministic(built.model, 20000, seed, &hook, &lns);

        CAPTURE(result.best_violation, result.objective, result.iterations);
        REQUIRE(result.feasible);
        REQUIRE(result.best_violation <= kDefaultFeasibilityTolerance);
        REQUIRE(std::isfinite(result.objective));
        // The iteration budget, not a wall clock, is what stopped the run (#104).
        REQUIRE(result.termination == TerminationReason::IterationLimit);

        // The reported objective must be the one the returned assignment
        // evaluates to; solve() restores best_state before returning.
        const double model_objective = built.model.node(built.objective_node_id).value;
        REQUIRE(std::abs(model_objective - result.objective) <=
                1e-6 * (std::abs(result.objective) + 1.0));

        REQUIRE(result.objective <= primal_bks + loose_gap);
        REQUIRE(result.objective >= dual_bound - band);
    }
}
