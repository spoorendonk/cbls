// Network-free tests for the AMPL NL reader (issue #72, P3; binary format
// #282), the NL->Model adapter (P4), and one end-to-end solve of a vendored
// non-convex instance (#124). Reader/adapter fixtures are tiny inline `g3`/`b3`
// NL files, plus three vendored MINLPLib files in tests/data (one binary); no
// instances are downloaded. Fixture layout mirrors a real minlplib `.nl` (header block,
// then b / x / r / J / O / C segments).

#include "libstdcxx_rng.h"
#include "test_helpers.h"

#include <algorithm>
#include <array>
#include <benchmarks/minlplib/note_policy.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace cbls;
using Catch::Matchers::WithinAbs;

// A minimal but realistic header for `nv` vars, `nc` cons, `no` objs. The body
// segments follow. Counts beyond the first line are cosmetic for the reader
// (it reads nvars/ncons/nobjs, the discrete and the common-expression lines),
// so we keep them plausible but simple. `ndef` defined variables (V segments)
// are declared as `comb`.
static std::string nl_header(int nv, int nc, int no, int ndef = 0) {
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
    h += " " + std::to_string(ndef) + " 0 0 0 0\t# common exprs\n";
    return h;
}

// ---- Binary NL ('b' header), #282 ----

// This host's `arith` code: 1 = IEEE little-endian, 2 = big-endian.
static int host_arith() {
    const uint16_t probe = 1;
    unsigned char first = 0;
    std::memcpy(&first, &probe, 1);
    return first == 1 ? 1 : 2;
}

// The `g` fixture header turned into a `b` one: same counts, `arith` set
// (to the host's byte order unless given).
static std::string nl_header_binary(int nv, int nc, int no, int arith = 0) {
    std::string h = nl_header(nv, nc, no);
    h[0] = 'b';
    const std::string text_arith = " 0 0 0 1\t# linear net, funcs, arith, flags\n";
    const std::size_t at = h.find(text_arith);
    h.replace(at, text_arith.size(),
              " 0 0 " + std::to_string(arith == 0 ? host_arith() : arith) +
                  " 0\t# linear net, funcs, arith, flags\n");
    return h;
}

// Appends binary NL items in the host's byte order.
struct BinaryNl {
    std::string s;
    BinaryNl& tag(char c) {
        s += c;
        return *this;
    }
    template <typename T>
    BinaryNl& put(T v) {
        std::array<char, sizeof(T)> bytes{};
        std::memcpy(bytes.data(), &v, sizeof(T));
        s.append(bytes.data(), bytes.size());
        return *this;
    }
    BinaryNl& i32(int32_t v) { return put(v); }
    BinaryNl& f64(double v) { return put(v); }
};

// minimize x0^2 + x1 ; s.t. 2 x0 + 3 x1 >= 3 ; x in [0,10]^2 -- in both encodings.
static std::string text_twin() {
    std::string text = nl_header(2, 1, 1);
    text += "b\n0 0 10\n0 0 10\n";
    text += "r\n2 3\n";
    text += "J0 2\n0 2\n1 3\n";
    text += "O0 0\n";
    text += "o5\nv0\nn2\n";
    text += "G0 1\n1 1\n";
    return text;
}

static std::string binary_twin() {
    BinaryNl b;
    b.tag('b').tag('0').f64(0).f64(10).tag('0').f64(0).f64(10);
    b.tag('r').tag('2').f64(3);
    b.tag('J').i32(0).i32(2).i32(0).f64(2).i32(1).f64(3);
    b.tag('O').i32(0).i32(0);
    b.tag('o').i32(5).tag('v').i32(0).tag('n').f64(2);
    b.tag('G').i32(0).i32(1).i32(1).f64(1);
    return nl_header_binary(2, 1, 1) + b.s;
}

TEST_CASE("NL reader reads the binary format ('b' header)", "[minlplib]") {
    SECTION("the same problem as its text twin") {
        const NlProblem t = parse_nl(text_twin(), "text");
        const NlProblem b = parse_nl(binary_twin(), "binary");
        REQUIRE(b.n_vars == t.n_vars);
        REQUIRE(b.n_cons == t.n_cons);
        REQUIRE(b.n_objs == t.n_objs);
        for (std::size_t j = 0; j < 2; ++j) {
            REQUIRE(b.var_bounds[j].type == t.var_bounds[j].type);
            REQUIRE(b.var_bounds[j].lower == t.var_bounds[j].lower);
            REQUIRE(b.var_bounds[j].upper == t.var_bounds[j].upper);
        }
        REQUIRE(b.constraints[0].bound.type == t.constraints[0].bound.type);
        REQUIRE(b.constraints[0].bound.lower == t.constraints[0].bound.lower);
        // Term by term, so a swapped index and coefficient in binary J or G
        // (a column-major mix-up the objective value alone would not reveal
        // for J) fails here.
        const auto same_terms = [](const std::vector<NlLinTerm>& x,
                                   const std::vector<NlLinTerm>& y) {
            REQUIRE(x.size() == y.size());
            for (std::size_t i = 0; i < x.size(); ++i) {
                REQUIRE(x[i].var == y[i].var);
                REQUIRE(x[i].coef == y[i].coef);
            }
        };
        same_terms(b.constraints[0].linear, t.constraints[0].linear);
        same_terms(b.objectives[0].linear, t.objectives[0].linear);
        const NlExpr& e = b.objectives[0].nonlinear;
        REQUIRE(e.nodes.size() == t.objectives[0].nonlinear.nodes.size());
        REQUIRE(e.nodes[e.root].opcode == 5);

        // Same model, same value: at x = (2, 2) the objective is 4 + 2.
        NlToModelResult r = nl_to_model(b);
        REQUIRE(r.supported);
        r.model.var_mut(0).value = 2.0;
        r.model.var_mut(1).value = 2.0;
        full_evaluate(r.model);
        REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(6.0, 1e-12));
    }
    SECTION("short and long constants") {
        // minimize x0 * 3 + (-7): 's' is an int16, 'l' an int32.
        BinaryNl b;
        b.tag('b').tag('0').f64(-5).f64(5);
        b.tag('O').i32(0).i32(0).tag('o').i32(0).tag('o').i32(2).tag('v').i32(0);
        b.tag('s').put(static_cast<int16_t>(3));
        b.tag('l').i32(-7);
        NlToModelResult r = nl_to_model(parse_nl(nl_header_binary(1, 0, 1) + b.s, "sl"));
        REQUIRE(r.supported);
        r.model.var_mut(0).value = 2.0;
        full_evaluate(r.model);
        REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(-1.0, 1e-12));
    }
    SECTION("a truncated file is refused") {
        std::string cut = binary_twin();
        cut.resize(cut.size() - 3);  // into the last coefficient
        REQUIRE_THROWS_WITH(parse_nl(cut, "cut"),
                            Catch::Matchers::ContainsSubstring("unexpected end of file"));
    }
    SECTION("a foreign byte order is refused") {
        const int foreign = host_arith() == 1 ? 2 : 1;
        REQUIRE_THROWS_WITH(parse_nl(nl_header_binary(1, 0, 1, foreign) + "b", "endian"),
                            Catch::Matchers::ContainsSubstring("byte order"));
    }
    SECTION("a real MINLPLib file, kriging_peaks-red010") {
        // MINLPLib serves this instance only in binary. Its best-known point
        // (minlplib.org/sol/kriging_peaks-red010.p1.sol; columns x1, x2 per
        // its .col file) gives the catalogue's primal bound 0.291120782173534.
        const NlProblem prob = read_nl("tests/data/kriging_peaks-red010.nl");
        REQUIRE(prob.n_vars == 2);
        NlToModelResult r = nl_to_model(prob);
        if (!r.supported) {
            FAIL((r.skipped_reasons.empty() ? std::string("unsupported") : r.skipped_reasons[0]));
            return;
        }
        r.model.var_mut(0).value = -2.169611845910990;
        r.model.var_mut(1).value = 0.295734644415665;
        full_evaluate(r.model);
        REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(0.291120782173534, 1e-9));
    }
}

TEST_CASE("NL reader refuses a text tag missing its number", "[minlplib]") {
    // A bare "O" would otherwise take the next line's number as its index.
    const std::string text = nl_header(1, 0, 1) + "b\n0 0 1\nO\n0 0\nn0\n";
    REQUIRE_THROWS_WITH(parse_nl(text, "bare"),
                        Catch::Matchers::ContainsSubstring("missing its number"));
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
    // #291: each section's counts are within [0, n_vars] on their own (the
    // #290 range check passes), but no valid variable ordering produces them.
    // The reader refuses the header before placement rather than build a model
    // with silently wrong integrality.
    const auto refuses = [](const std::string& text, const std::string& relation) {
        REQUIRE_THROWS_WITH(parse_nl(text, "bad-layout"),
                            Catch::Matchers::ContainsSubstring("malformed header") &&
                                Catch::Matchers::ContainsSubstring(relation));
    };
    const std::string tail4 = bounds_segment(4) + "O0 0\nn0\n";
    SECTION("nonlinear-in-both block larger than the nonlinear-in-constraints one") {
        // The issue's scenario: one integer is placed and one declared, so the
        // placement self-check alone let column 2 become integer.
        refuses(
            nl_header_disc(4, 0, 1, /*nlvc=*/2, /*nlvo=*/0, /*nlvb=*/3, 0, 0, /*nlvbi=*/1, 0, 0) +
                tail4,
            "nlvb = 3 exceeds nlvc = 2");
    }
    SECTION("integers in an empty nonlinear-in-both block") {
        std::string text = nl_header_disc(3, 0, 1, /*nlvc=*/0, /*nlvo=*/0, /*nlvb=*/0, 0, 0,
                                          /*nlvbi=*/2, /*nlvci=*/1, /*nlvoi=*/0);
        refuses(text + bounds_segment(3) + "O0 0\nn0\n", "nlvbi = 2 exceeds nlvb = 0");
    }
    SECTION("more constraint-only integers than constraint-only columns") {
        refuses(nl_header_disc(4, 0, 1, /*nlvc=*/3, 0, /*nlvb=*/2, 0, 0, 0, /*nlvci=*/2, 0) + tail4,
                "nlvci = 2 exceeds nlvc - nlvb = 1");
    }
    SECTION("objective-only integers when nlvo does not pass nlvc") {
        refuses(nl_header_disc(4, 0, 1, /*nlvc=*/3, /*nlvo=*/2, /*nlvb=*/1, 0, 0, 0, 0,
                               /*nlvoi=*/1) +
                    tail4,
                "nlvoi = 1 exceeds max(nlvo - nlvc, 0) = 0");
    }
    SECTION("linear discrete block overlapping the nonlinear blocks") {
        refuses(nl_header_disc(4, 0, 1, /*nlvc=*/2, /*nlvo=*/3, 0, /*nbv=*/1, /*niv=*/1, 0, 0, 0) +
                    tail4,
                "max(nlvc, nlvo) + nbv + niv = 5 exceeds 4 variables");
    }
    SECTION("every block full to its bound still reads") {
        // nlvb=1 (col 0), constraint-only col 1, objective-only col 2, linear
        // binary col 3: each block's integer count at its maximum.
        const NlProblem p = parse_nl(nl_header_disc(4, 0, 1, /*nlvc=*/2, /*nlvo=*/3, /*nlvb=*/1,
                                                    /*nbv=*/1, 0, /*nlvbi=*/1, /*nlvci=*/1,
                                                    /*nlvoi=*/1) +
                                         tail4,
                                     "full");
        REQUIRE(p.var_is_discrete == std::vector<uint8_t>{1, 1, 1, 1});
    }
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

TEST_CASE("nl_to_model keeps a finite bound beyond the clamp inside the box", "[minlplib]") {
    // Regression for #357: one side finite at or past the clamp, the other
    // missing, put the fallback on the wrong side of the real bound, so the box
    // excluded it or collapsed to one point though the bound leaves room. The
    // box must hold the bound exactly and reach past it on the open side.
    // Default clamps: inf_clamp 1e9, int_inf_clamp 1e6.
    auto build = [](bool discrete, const std::string& bound) {
        const int nlvbi = discrete ? 1 : 0;
        std::string text = nl_header_disc(1, 0, 1, 1, 0, 1, 0, 0, nlvbi, 0, 0);
        text += "b\n" + bound + "\n";
        text += "O0 0\nn0\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "clamp-side"));
        REQUIRE(r.supported);
        REQUIRE(r.model.var(0).type == (discrete ? VarType::Int : VarType::Float));
        CHECK(r.n_clamped_columns == 1);
        return r;
    };
    SECTION("Int free below, ub -2e6: used to be fixed at -1e6, outside the bound") {
        NlToModelResult r = build(true, "1 -2000000");  // type 1: upper only
        CHECK_THAT(r.model.var(0).ub, WithinAbs(-2.0e6, 1e-9));
        CHECK(r.model.var(0).lb <= -2.0e6 - 1.0);
    }
    SECTION("Int lb 2e6, no ub: used to collapse to the point 2e6") {
        NlToModelResult r = build(true, "2 2000000");  // type 2: lower only
        CHECK_THAT(r.model.var(0).lb, WithinAbs(2.0e6, 1e-9));
        CHECK(r.model.var(0).ub >= 2.0e6 + 1.0);
    }
    SECTION("Float lb 5e9, no ub: used to swap to [1e9, 5e9]") {
        NlToModelResult r = build(false, "2 5e9");
        CHECK_THAT(r.model.var(0).lb, WithinAbs(5.0e9, 1e-6));
        CHECK(r.model.var(0).ub > 5.0e9 + 1.0);
    }
    SECTION("Float free below, ub -5e9: used to swap to [-5e9, -1e9]") {
        NlToModelResult r = build(false, "1 -5e9");
        CHECK_THAT(r.model.var(0).ub, WithinAbs(-5.0e9, 1e-6));
        CHECK(r.model.var(0).lb < -5.0e9 - 1.0);
    }
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

TEST_CASE("NL reader refuses a complementarity bound instead of misparsing", "[minlplib]") {
    // Regression for #211: bound type 5 (`5 k i`) matched no case and consumed
    // no tokens, so the rest of the file was misread from its payload on. It is
    // refused with the reader's "unsupported" wording, which the MINLPLib
    // runner buckets as a coverage gap.
    std::string text = nl_header(2, 1, 1);
    text += "b\n0 0 10\n0 0 10\n";
    text += "r\n5 1 1\n";  // con0: complementary with var 1
    text += "J0 1\n0 1\n";
    text += "O0 0\nn0\n";
    REQUIRE_THROWS_WITH(parse_nl(text, "compl"),
                        Catch::Matchers::ContainsSubstring("not supported by this reader"));
}

TEST_CASE("NL reader reads a defined variable (V segment)", "[minlplib]") {
    // v2 := 2*x0 + x1^2 ; minimize 3 * v2.
    std::string text = nl_header(2, 0, 1, /*ndef=*/1);
    text += "b\n0 -5 5\n0 -5 5\n";
    text += "V2 1 0\n0 2\no5\nv1\nn2\n";
    text += "O0 0\no2\nv2\nn3\n";
    NlProblem p = parse_nl(text, "defvar");
    REQUIRE(p.n_defined_vars == 1);
    REQUIRE(p.defined_vars.size() == 1);
    const NlDefinedVar& dv = p.defined_vars[0];
    REQUIRE(dv.defined);
    REQUIRE(dv.linear.size() == 1);
    REQUIRE(dv.linear[0].var == 0);
    REQUIRE_THAT(dv.linear[0].coef, WithinAbs(2.0, 1e-12));
    REQUIRE(dv.nonlinear.nodes[dv.nonlinear.root].opcode == 5);
}

TEST_CASE("NL reader refuses a V segment outside the declared range", "[minlplib]") {
    SECTION("index past the header's count") {
        std::string text = nl_header(2, 0, 1, /*ndef=*/1);
        text += "V3 0 0\nn1\nO0 0\nn0\n";
        REQUIRE_THROWS_WITH(parse_nl(text, "badv"),
                            Catch::Matchers::ContainsSubstring("defined-variable range"));
    }
    SECTION("index naming a column") {
        std::string text = nl_header(2, 0, 1, /*ndef=*/1);
        text += "V1 0 0\nn1\nO0 0\nn0\n";
        REQUIRE_THROWS_WITH(parse_nl(text, "badv"),
                            Catch::Matchers::ContainsSubstring("defined-variable range"));
    }
    SECTION("defined twice") {
        std::string text = nl_header(2, 0, 1, /*ndef=*/1);
        text += "V2 0 0\nn1\nV2 0 0\nn2\nO0 0\nn0\n";
        REQUIRE_THROWS_WITH(parse_nl(text, "twice"),
                            Catch::Matchers::ContainsSubstring("defined twice"));
    }
    SECTION("a common-expression count too large") {
        // Line 10's counts are each range-checked, and so is their sum.
        // INT64_MAX counts would wrap int64 if summed first; counts each
        // within int32 can still sum past it, and would wrap the cast.
        const std::string line =
            GENERATE(std::string(" 9223372036854775807 9223372036854775807 2 0 0"),
                     std::string(" 2000000000 2000000000 0 0 0"));
        std::string text = nl_header(2, 0, 1);
        const std::string line10 = " 0 0 0 0 0\t# common exprs";
        const std::size_t at = text.find(line10);
        REQUIRE(at != std::string::npos);
        text.replace(at, line10.size(), line);
        text += "O0 0\nn0\n";
        REQUIRE_THROWS_WITH(parse_nl(text, "overflow"),
                            Catch::Matchers::ContainsSubstring("malformed common-expression"));
    }
}

TEST_CASE("NL reader refuses header counts the file cannot back", "[minlplib]") {
    // #289: vectors are sized from these counts. ~2e9 of anything in a tiny
    // file must be a clean refusal, not a multi-GB resize or std::bad_alloc.
    constexpr int kHuge = 2000000000;
    const std::string body = "b\n0 0 1\nO0 0\nn0\n";
    SECTION("variables") {
        REQUIRE_THROWS_WITH(parse_nl(nl_header(kHuge, 0, 1) + body, "vars"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("constraints") {
        REQUIRE_THROWS_WITH(parse_nl(nl_header(1, kHuge, 1) + body, "cons"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("objectives") {
        REQUIRE_THROWS_WITH(parse_nl(nl_header(1, 0, kHuge) + body, "objs"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("defined variables") {
        REQUIRE_THROWS_WITH(parse_nl(nl_header(1, 0, 1, kHuge) + body, "defvars"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("binary format") {
        BinaryNl b;
        b.tag('b').tag('0').f64(0).f64(1).tag('O').i32(0).i32(0).tag('n').f64(0);
        REQUIRE_THROWS_WITH(parse_nl(nl_header_binary(kHuge, 0, 1) + b.s, "binvars"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("a negative count") {
        REQUIRE_THROWS_WITH(parse_nl(nl_header(-1, 0, 1) + body, "negative"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("counts the file does back still read") {
        REQUIRE(parse_nl(nl_header(1, 0, 1) + body, "ok").n_vars == 1);
    }
}

TEST_CASE("NL reader refuses nonlinear or discrete counts outside the variable count",
          "[minlplib]") {
    // #290: the placement counts of header lines 5 and 7 size `mark_tail`'s
    // loops and feed int64 sums. Before the check, the first section looped
    // ~4e18 times instead of refusing.
    const std::string body = "b\n0 0 1\n0 0 1\nO0 0\nn0\n";
    const auto with_lines = [&body](const std::string& line5, const std::string& line7) {
        std::string text = nl_header(2, 0, 1);
        const std::string old5 = " 0 0 0\t# nonlinear vars";
        const std::string old7 = " 0 0 0 0 0\t# discrete";
        const std::size_t at5 = text.find(old5);
        const std::size_t at7 = text.find(old7);
        REQUIRE(at5 != std::string::npos);
        REQUIRE(at7 != std::string::npos);
        text.replace(at7, old7.size(), line7);  // later line first: at5 stays valid
        text.replace(at5, old5.size(), line5);
        return text + body;
    };
    SECTION("a huge nonlinear block and integer count") {
        REQUIRE_THROWS_WITH(
            parse_nl(with_lines(" 0 0 4000000000000000000", " 0 0 4000000000000000000 0 0"),
                     "hang"),
            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("each count out of range on its own") {
        // One field at a time, the rest zero, so each field's check alone
        // decides: a negative value (which would wrap nlvc - nlvb or cat3's
        // subtraction), one past n_vars = 2, and a hang-sized one.
        const std::array<const char*, 8> fields = {"nlvc", "nlvo",  "nlvb",  "nbv",
                                                   "niv",  "nlvbi", "nlvci", "nlvoi"};
        const auto k = static_cast<std::size_t>(GENERATE(0, 1, 2, 3, 4, 5, 6, 7));
        const std::string value = GENERATE(as<std::string>{}, "-1", "3", "4000000000000000000");
        std::array<std::string, 8> v;
        v.fill("0");
        v.at(k) = value;
        const std::string line5 = " " + v[0] + " " + v[1] + " " + v[2];
        const std::string line7 = " " + v[3] + " " + v[4] + " " + v[5] + " " + v[6] + " " + v[7];
        CAPTURE(fields.at(k), value);
        REQUIRE_THROWS_WITH(
            parse_nl(with_lines(line5, line7), "range"),
            Catch::Matchers::ContainsSubstring("malformed header") &&
                Catch::Matchers::ContainsSubstring(std::string(fields.at(k)) + " = " + value));
    }
    SECTION("discrete counts each in range but summing past the variable count") {
        REQUIRE_THROWS_WITH(parse_nl(with_lines(" 0 0 0", " 2 2 0 0 0"), "sum"),
                            Catch::Matchers::ContainsSubstring("malformed header"));
    }
    SECTION("counts within the variable count still read") {
        const NlProblem p = parse_nl(with_lines(" 2 0 2", " 0 0 1 0 0"), "ok");
        REQUIRE(p.n_discrete_vars == 1);
        REQUIRE(p.var_is_discrete == std::vector<uint8_t>{0, 1});
    }
}

TEST_CASE("NL reader skips a dual initial guess (d segment)", "[minlplib]") {
    std::string text = nl_header(1, 1, 1);
    text += "b\n0 0 10\n";
    text += "r\n1 4\n";
    text += "d1\n0 0.5\n";
    text += "J0 1\n0 1\n";
    text += "O0 0\nn0\n";
    NlProblem p = parse_nl(text, "dual");
    REQUIRE(p.constraints[0].linear.size() == 1);
    REQUIRE_THAT(p.constraints[0].bound.upper, WithinAbs(4.0, 1e-12));
}

TEST_CASE("NL reader refuses a suffix segment and names the suffix", "[minlplib]") {
    // MINLPLib's suffixes are SOS declarations (`sosno`, `ref`); skipping them
    // would solve a relaxation, so the file is refused with the suffix named.
    std::string text = nl_header(2, 0, 1);
    text += "S0 2 sosno\n0 1\n1 1\n";
    text += "O0 0\nn0\n";
    REQUIRE_THROWS_WITH(parse_nl(text, "sos"),
                        Catch::Matchers::ContainsSubstring("suffix 'sosno'") &&
                            Catch::Matchers::ContainsSubstring("not supported by this reader"));
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

TEST_CASE("nl_to_model translates FLOOR and CEIL", "[minlplib]") {
    // Regression for #211: FLOOR (13) and CEIL (14) were parsed but not
    // translated, so an instance using them was skipped as unsupported though
    // the DAG has had Floor and Ceil since #186. minimize floor(x0) + ceil(x1).
    std::string text = nl_header(2, 0, 1);
    text += "b\n0 -10 10\n0 -10 10\n";
    text += "O0 0\n";
    text += "o0\no13\nv0\no14\nv1\n";
    NlProblem p = parse_nl(text, "floorceil");
    NlToModelResult r = nl_to_model(p);
    REQUIRE(r.supported);
    r.model.var_mut(0).value = 2.7;
    r.model.var_mut(1).value = -1.3;
    full_evaluate(r.model);
    REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(2.0 + -1.0, 1e-12));
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
    REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(6.0, 1e-9));

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
    REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(-5.0, 1e-9));
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
    REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(8.0, 1e-9));
}

// minimize <pow_expr> over one variable in [-10, 10]; the objective's value at x.
static double pow_objective_at(const std::string& pow_expr, double x) {
    std::string text = nl_header(1, 0, 1);
    text += "b\n0 -10 10\n";
    text += "O0 0\n" + pow_expr;
    NlToModelResult r = nl_to_model(parse_nl(text, "pow"));
    if (!r.supported) {
        FAIL((r.skipped_reasons.empty() ? std::string("unsupported") : r.skipped_reasons[0]));
    }
    r.model.var_mut(0).value = x;
    full_evaluate(r.model);
    return r.model.node_value(r.objective_node_id);
}

TEST_CASE("nl_to_model evaluates OPPOW with a constant base or exponent", "[minlplib]") {
    // MINLPLib's vcpower (x^c) and cvpower (c^x) both reach the NL file as a
    // plain o5 with one constant operand (#283); Pow takes std::pow's answer
    // unmodified (#205), NaN for a domain error included.
    SECTION("constant exponent: x^c") {
        CHECK_THAT(pow_objective_at("o5\nv0\nn.5\n", 4.0), WithinAbs(2.0, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nv0\nn3\n", -2.0), WithinAbs(-8.0, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nv0\nn-1\n", 4.0), WithinAbs(0.25, 1e-12));
    }
    SECTION("constant base: c^x") {
        CHECK_THAT(pow_objective_at("o5\nn2\nv0\n", 3.0), WithinAbs(8.0, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nn2\nv0\n", -1.0), WithinAbs(0.5, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nn.5\nv0\n", 2.5), WithinAbs(std::pow(0.5, 2.5), 1e-12));
    }
    SECTION("constant base c <= 0") {
        CHECK_THAT(pow_objective_at("o5\nn0\nv0\n", 2.0), WithinAbs(0.0, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nn0\nv0\n", 0.0), WithinAbs(1.0, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nn-2\nv0\n", 3.0), WithinAbs(-8.0, 1e-12));
        CHECK_THAT(pow_objective_at("o5\nn-2\nv0\n", 2.0), WithinAbs(4.0, 1e-12));
        CHECK(std::isnan(pow_objective_at("o5\nn-2\nv0\n", 0.5)));
        CHECK(std::isinf(pow_objective_at("o5\nn0\nv0\n", -1.0)));
    }
    SECTION("a negative base under a non-integer exponent is NaN") {
        CHECK(std::isnan(pow_objective_at("o5\nv0\nn.5\n", -4.0)));
        CHECK(std::isnan(pow_objective_at("o5\nv0\nn.86\n", -1.0)));
    }
}

// Reads and models a vendored MINLPLib file, sets its best-known point (values
// in NL column order), and checks the objective and that the point is feasible.
static void check_real_file_at_best_known(const std::string& path, const std::vector<double>& point,
                                          double objective) {
    const NlProblem prob = read_nl(path);
    REQUIRE(prob.n_vars == static_cast<int>(point.size()));
    NlToModelResult r = nl_to_model(prob);
    if (!r.supported) {
        FAIL((r.skipped_reasons.empty() ? std::string("unsupported") : r.skipped_reasons[0]));
        return;
    }
    for (std::size_t j = 0; j < point.size(); ++j) {
        r.model.var_mut(static_cast<int32_t>(j)).value = point[j];
    }
    full_evaluate(r.model);
    REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(objective, 1e-6));
    ViolationManager vm(r.model);
    vm.invalidate_cache();
    REQUIRE_THAT(vm.total_violation(), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Real MINLPLib files with x^c and c^x evaluate at their best-known points",
          "[minlplib]") {
    // Best-known points from minlplib.org/sol/<name>.p1.sol, mapped to NL
    // columns by each instance's .col file; the objectives are the catalogue's
    // primal bounds.
    SECTION("filter: (a quotient of x)^.86 (vcpower)") {
        // Columns x2, x1.
        check_real_file_at_best_known("tests/data/filter.nl", {10.0, 1290.226825443540065},
                                      8685.277077052489403);
    }
    SECTION("ex8_4_4: .157^x and friends (cvpower)") {
        // Columns x14 x15 x17 x1 .. x13 x16.
        check_real_file_at_best_known(
            "tests/data/ex8_4_4.nl",
            {0.556760596607749, 1.131808991890020, 0.621898732685481, 5.130141973432660,
             -4.935270010426710, 2.670723018129510, -2.070270651029460, 1.910958131972460,
             -1.103776175062120, 1.548384003739780, -0.580941739845044, 1.338122376788280,
             -0.228239587862993, 1.201670495795650, 0.042017456203428, 0.606297577433204,
             0.750199691215214},
            0.212459838698485);
    }
}

TEST_CASE("nl_to_model inlines a defined variable", "[minlplib]") {
    // v2 := 2*x0 + x1^2 ; minimize 3 * v2. At x = (1, 3): v2 = 11, obj = 33.
    std::string text = nl_header(2, 0, 1, /*ndef=*/1);
    text += "b\n0 -5 5\n0 -5 5\n";
    text += "V2 1 0\n0 2\no5\nv1\nn2\n";
    text += "O0 0\no2\nv2\nn3\n";
    NlToModelResult r = nl_to_model(parse_nl(text, "defvar"));
    REQUIRE(r.supported);
    REQUIRE(r.model.num_vars() == 2);  // a defined variable is not a column
    r.model.var_mut(0).value = 1.0;
    r.model.var_mut(1).value = 3.0;
    full_evaluate(r.model);
    REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(33.0, 1e-9));
}

TEST_CASE("nl_to_model resolves a chain of defined variables", "[minlplib]") {
    SECTION("V referencing V, in both parts") {
        // v2 := x0 + x1 ; v3 := v2 * v2 + 0.5 * v2.
        // minimize v3 + x0. At x = (1, 2): v2 = 3, v3 = 10.5, obj = 11.5.
        std::string text = nl_header(2, 0, 1, /*ndef=*/2);
        text += "b\n0 -5 5\n0 -5 5\n";
        text += "V2 2 0\n0 1\n1 1\nn0\n";
        text += "V3 1 0\n2 0.5\no2\nv2\nv2\n";
        text += "O0 0\nv3\n";
        text += "G0 1\n0 1\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "chain"));
        REQUIRE(r.supported);
        r.model.var_mut(0).value = 1.0;
        r.model.var_mut(1).value = 2.0;
        full_evaluate(r.model);
        REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(11.5, 1e-9));
    }
    SECTION("a chain too long to build by recursion") {
        // A recursive AMPL definition, s[t] := s[t-1] + 1, one V per step:
        // v1 := x0 + 1, v(k) := v(k-1) + 1. Building it one native frame per
        // link overflows the stack. minimize v(n); at x0 = 0.5 it is n + 0.5.
        constexpr int kLinks = 200000;
        std::string text = nl_header(1, 0, 1, kLinks);
        text += "b\n0 0 1\n";
        for (int k = 1; k <= kLinks; ++k) {
            text += "V" + std::to_string(k) + " 0 0\no0\nv" + std::to_string(k - 1) + "\nn1\n";
        }
        text += "O0 0\nv" + std::to_string(kLinks) + "\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "longchain"));
        REQUIRE(r.supported);
        r.model.var_mut(0).value = 0.5;
        full_evaluate(r.model);
        REQUIRE_THAT(r.model.node_value(r.objective_node_id), WithinAbs(kLinks + 0.5, 1e-6));
    }
}

TEST_CASE("nl_to_model builds a defined variable once for all its uses", "[minlplib]") {
    // v2 := x0 * x1 bounds two rows: both rows read the same DAG node.
    std::string text = nl_header(2, 2, 1, /*ndef=*/1);
    text += "b\n0 0 5\n0 0 5\n";
    text += "r\n1 4\n2 1\n";
    text += "V2 0 0\no2\nv0\nv1\n";
    text += "C0\nv2\nC1\nv2\n";
    text += "O0 0\nv2\n";
    NlToModelResult r = nl_to_model(parse_nl(text, "shared"));
    REQUIRE(r.supported);
    const ExprNode& c0 = r.model.node(r.constraint_node_ids[0]);
    const ExprNode& c1 = r.model.node(r.constraint_node_ids[1]);
    const ChildRef body0 = r.model.children(c0)[0];
    const ChildRef body1 = r.model.children(c1)[0];
    REQUIRE_FALSE(body0.is_var);
    REQUIRE(body0.id == body1.id);
    REQUIRE(r.model.node(body0.id).op == NodeOp::Prod);
}

TEST_CASE("nl_to_model reports an undefined or cyclic defined variable", "[minlplib]") {
    SECTION("declared but never defined") {
        std::string text = nl_header(1, 0, 1, /*ndef=*/1);
        text += "b\n0 0 5\nO0 0\nv1\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "undef"));
        REQUIRE_FALSE(r.supported);
        REQUIRE_THAT(r.skipped_reasons[0], Catch::Matchers::ContainsSubstring("no V segment"));
    }
    SECTION("refers to itself") {
        std::string text = nl_header(1, 0, 1, /*ndef=*/1);
        text += "b\n0 0 5\nV1 0 0\no0\nv1\nv0\nO0 0\nv1\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "cycle"));
        REQUIRE_FALSE(r.supported);
        REQUIRE_THAT(r.skipped_reasons[0], Catch::Matchers::ContainsSubstring("refers to itself"));
    }
    SECTION("an unsupported op inside a definition") {
        std::string text = nl_header(1, 0, 1, /*ndef=*/1);
        text += "b\n0 0 5\nV1 0 0\no49\nv0\nO0 0\nv1\n";  // atan
        NlToModelResult r = nl_to_model(parse_nl(text, "defatan"));
        REQUIRE_FALSE(r.supported);
        REQUIRE_THAT(r.skipped_reasons[0], Catch::Matchers::ContainsSubstring("49"));
    }
    SECTION("refers to itself through another") {
        // v1 := v2 + 1, v2 := v1 * x0: a cycle of two.
        std::string text = nl_header(1, 0, 1, /*ndef=*/2);
        text += "b\n0 0 5\nV1 0 0\no0\nv2\nn1\nV2 0 0\no2\nv1\nv0\nO0 0\nv1\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "cycle2"));
        REQUIRE_FALSE(r.supported);
        REQUIRE_THAT(r.skipped_reasons[0], Catch::Matchers::ContainsSubstring("refers to itself"));
    }
    SECTION("a linear J term that names no column") {
        // Index 1 is the defined variable (AMPL never writes one into J),
        // index 2 is past every variable. Dropping either term would relax
        // the row, so both are refused.
        const std::string idx = GENERATE(std::string("1"), std::string("2"));
        std::string text = nl_header(1, 1, 1, /*ndef=*/1);
        text += "b\n0 0 5\nr\n1 4\nV1 0 0\nn1\nC0\nn0\nJ0 1\n" + idx + " 1\nO0 0\nn0\n";
        NlToModelResult r = nl_to_model(parse_nl(text, "defjac"));
        REQUIRE_FALSE(r.supported);
        REQUIRE_THAT(r.skipped_reasons[0],
                     Catch::Matchers::ContainsSubstring("linear term names variable " + idx));
    }
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
        const double model_objective = built.model.node_value(built.objective_node_id);
        REQUIRE(std::abs(model_objective - result.objective) <=
                1e-6 * (std::abs(result.objective) + 1.0));

        REQUIRE(result.objective <= primal_bks + loose_gap);
        REQUIRE(result.objective >= dual_bound - band);
    }
}

// #102. st_e40 has 4 variables and, once C1-C3 restrict the three integers to
// {1,2,3,5,8,10,12}, exactly 343 integer combinations of which 52 are feasible.
// Pre-#102 the engine failed it at a 60s wall clock: a limit cycle within ~20
// GLS iterations (x1 flipping between two values, x3 hopping between the roots
// of its four rows) took 92% of a 155 000-iteration run in GLS weight bumps
// that changed nothing, visiting 3 of the 343 combinations and none of the 52
// feasible ones. Neither trapped value is in any feasible combination (every
// one of the 52 needs x0 >= 5 and x1 >= 5), so refining could not escape it.
//
// Diversification is gated on perturbation_period (100) non-improving BATCHES
// of batch_iterations (1000) GLS iterations, and before the first feasible
// solution no batch improves -- one kick per 100 000 iterations whatever the
// search does. GFJConfig::unproductive_iterations ends a batch that has
// stopped reducing the REAL rows' violation, and solve() treats such a batch
// as the diversification kick being due.
//
// On the iteration-bounded path so it is bit-deterministic and runs in seconds;
// the issue's wall-clock criterion (feasible within 60s at seed 42) is recorded
// in benchmarks/instances/minlplib/README.md instead, since a wall clock cannot
// be asserted on reproducibly.
//
// Shown red before the fix, per CLAUDE.md, rather than argued: a `git archive`
// of the branch with ONLY the four production files (src/feasibility_jump.cpp,
// src/search.cpp and their two headers) restored from main -- this test file
// left exactly as it is -- fails at the first seed with
//
//   REQUIRE( result.feasible ) ... false
//   seed 2; best_violation := 2.34203865; objective := inf; iterations := 15000
//
// i.e. the unfixed engine spends the whole budget and never reaches the feasible
// set at all, rather than reaching it and being scored differently.
//
// Budget calibration: the seeds below solve at an 8 000-iteration budget, so
// 15 000 carries ~1.9x. Raise the budget or re-pick seeds if the engine's
// trajectory changes; don't swap it for a wall clock. Re-picks, over seeds
// 1-30 at 15 000:
// - #206, whose fix to apply_jump's scan-set sampling moved every trajectory:
//   of the pair 2 and 42, 42 no longer solved inside 15 000; the pass rate
//   was 24/30 after the fix against 25/30 before it, so restart variance on a
//   343-point space, not a regression.
// - #226, when the objective tie-break became the default: 20/30 with the
//   tie-break on and 20/30 with it off, but seed 13 no longer among them.
//   Seeds 11 and 26 solved at 8 000 with it on (9/30 did).
// - #344, when propagation began reading this file's n0 rows and one column's
//   box tightened: 22/30 at 15 000 and 19/30 at 8 000. Seed 26 solves at
//   15 000 but no longer at 8 000; seeds 11 and 12 solve at both.
TEST_CASE("MINLPLib st_e40 reaches feasibility on a small iteration budget", "[minlplib][solve]") {
    skip_unless_libstdcxx_rng();
    // bounds.csv: st_e40,other,4,8,min,30.41421356,30.4142136,0
    const double primal_bks = 30.41421356;

    NlProblem prob = read_nl("benchmarks/instances/minlplib/st_e40.nl");
    REQUIRE(prob.n_vars == 4);
    REQUIRE(prob.n_cons == 8);

    for (uint64_t seed : {11ULL, 12ULL}) {
        INFO("seed " << seed);
        NlToModelResult built = nl_to_model(prob);
        REQUIRE(built.supported);
        REQUIRE(built.objective_node_id >= 0);

        FloatIntensifyHook hook;
        LNS lns(0.3);
        SearchResult result = solve_deterministic(built.model, 15000, seed, &hook, &lns);

        CAPTURE(result.best_violation, result.objective, result.iterations);
        REQUIRE(result.feasible);
        REQUIRE(result.best_violation <= kDefaultFeasibilityTolerance);
        REQUIRE(std::isfinite(result.objective));

        // The returned assignment must evaluate to the reported objective.
        const double model_objective = built.model.node_value(built.objective_node_id);
        REQUIRE(std::abs(model_objective - result.objective) <=
                1e-6 * (std::abs(result.objective) + 1.0));

        // No feasible point of this instance beats the proven optimum. A
        // strictly better objective would mean the "feasible" verdict is wrong.
        REQUIRE(result.objective >= primal_bks - (10.0 * kDefaultFeasibilityTolerance));
    }
}

// --- Curated-note policy ----------------------------------------------------
//
// A solved, verified instance with a curated note gets the "stale analysis
// note (now solved)" warning. Pre-fix the runner tested `verified` inside
// `if (... && !result.feasible)`; initialised from the feasibility result and
// only ever assigned false, it was always false there, so the warning could
// not fire: both instances that now solve kept their curated notes silently.

TEST_CASE("curated note is retired when the instance now solves", "[minlplib]") {
    using cbls::minlplib::note_action;
    using cbls::minlplib::NoteAction;

    // The case the old nesting made unreachable: solved, verified, note stale.
    REQUIRE(note_action(true, true, true) == NoteAction::kStale);

    // Still infeasible: the note describes the row, so it is merged.
    REQUIRE(note_action(true, false, false) == NoteAction::kMerge);

    // Feasible but verification failed: a bookkeeping mismatch, not the
    // infeasibility the note describes -- neither retire it nor paste it on.
    REQUIRE(note_action(true, true, false) == NoteAction::kNone);

    // No curated note: nothing to do in any state.
    REQUIRE(note_action(false, true, true) == NoteAction::kNone);
    REQUIRE(note_action(false, false, false) == NoteAction::kNone);
}
