// Domain errors evaluate as undefined, never as a satisfiable value (#205).
//
// The DAG used to map a point where a function is undefined to an ordinary
// number -- sqrt of a negative to 0.0, log of a negative to -inf, any non-finite
// pow to +inf, a near-zero denominator to an infinity signed by the numerator
// alone -- and a comparison row read that number as a finite-side difference.
// `geq(pow(x, 0.5), 2)` at x = -10 was `2 - inf = -inf`: satisfied. The search
// found and exploited such points because they improved the objective, and
// reported `feasible`.
//
// The four models below are the issue's repros. Each is solved
// deterministically (iteration budget, no wall clock); on the unfixed engine
// each reports a feasible point that is not one.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/io.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace cbls;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

double eval_at(Model& m, int32_t node) {
    full_evaluate(m);
    return m.node_value(node);
}

// A one-variable model x in [lb, ub] with `body(x)` as its only constraint and
// `minimize x`, loaded through the .cbls reader exactly as the issue ran it.
Model load(const std::string& jsonl) {
    std::istringstream in(jsonl);
    return load_model(in);
}

const char* const kPowModel = R"({"var":"x","type":"Float","lb":-10,"ub":1}
{"node":"h","op":"Const","value":0.5}
{"node":"p","op":"Pow","children":["x","h"]}
{"node":"c2","op":"Const","value":2.0}
{"node":"g","op":"Geq","children":["p","c2"]}
{"constraint":"g"}
{"node":"o","op":"Sum","children":["x"]}
{"minimize":"o"}
)";

const char* const kSqrtModel = R"({"var":"x","type":"Float","lb":-10,"ub":10}
{"node":"s","op":"Sqrt","children":["x"]}
{"node":"c1","op":"Const","value":1.0}
{"node":"g","op":"Leq","children":["s","c1"]}
{"constraint":"g"}
{"node":"o","op":"Sum","children":["x"]}
{"minimize":"o"}
)";

const char* const kLogModel = R"({"var":"x","type":"Float","lb":-10,"ub":10}
{"node":"l","op":"Log","children":["x"]}
{"node":"c1","op":"Const","value":1.0}
{"node":"g","op":"Leq","children":["l","c1"]}
{"constraint":"g"}
{"node":"o","op":"Sum","children":["x"]}
{"minimize":"o"}
)";

// -1/y <= -1 holds for y in (0, 1] and at the pole y = +0 (where -1/+0 = -inf);
// for every y < 0 the quotient is positive. The unfixed engine signed the
// near-zero quotient by the numerator alone, so every |y| < 1e-15 read as
// -inf -- satisfied -- including the negative ones the search then minimised to.
const char* const kDivModel = R"({"var":"y","type":"Float","lb":-10,"ub":10}
{"node":"m1","op":"Const","value":-1.0}
{"node":"d","op":"Div","children":["m1","y"]}
{"node":"g","op":"Leq","children":["d","m1"]}
{"constraint":"g"}
{"node":"o","op":"Sum","children":["y"]}
{"minimize":"o"}
)";

constexpr int64_t kIters = 20000;

}  // namespace

// ---------------------------------------------------------------------------
// The four repro models
// ---------------------------------------------------------------------------

TEST_CASE("sqrt of x >= 2 with x <= 1 is reported infeasible, not feasible at a negative x",
          "[dag][domain]") {
    // pow(x, 0.5) >= 2 needs x >= 4 > ub: the model has no feasible point.
    Model m = load(kPowModel);
    const SearchResult r = solve_deterministic(m, kIters);
    REQUIRE_FALSE(r.feasible);
}

TEST_CASE("sqrt(x) <= 1 minimising x reaches 0, not a negative x", "[dag][domain]") {
    Model m = load(kSqrtModel);
    const SearchResult r = solve_deterministic(m, kIters);
    REQUIRE(r.feasible);
    REQUIRE(r.objective >= -1e-9);
}

TEST_CASE("log(x) <= 1 minimising x reaches 0, not a negative x", "[dag][domain]") {
    // log(0) is -inf and satisfies the row; every x < 0 is undefined.
    Model m = load(kLogModel);
    const SearchResult r = solve_deterministic(m, kIters);
    REQUIRE(r.feasible);
    REQUIRE(r.objective >= 0.0);
}

TEST_CASE("minimising y under -1/y <= -1 never settles on a negative y", "[dag][domain]") {
    Model m = load(kDivModel);
    const SearchResult r = solve_deterministic(m, kIters);
    REQUIRE(r.feasible);
    REQUIRE(r.objective >= 0.0);
}

// ---------------------------------------------------------------------------
// Values
// ---------------------------------------------------------------------------

TEST_CASE("sqrt, log and pow of an argument outside their domain evaluate to NaN",
          "[dag][domain]") {
    Model m;
    const int32_t x = m.float_var(-10, 10, "x");
    const int32_t s = m.sqrt_expr(x);
    const int32_t l = m.log_expr(x);
    const int32_t p = m.pow_expr(x, m.constant(1.0 / 3.0));
    m.minimize(m.sum({s, l, p}));
    m.close();

    m.var_mut(vid(x)).value = -8.0;
    full_evaluate(m);
    CHECK(std::isnan(m.node_value(s)));
    CHECK(std::isnan(m.node_value(l)));
    CHECK(std::isnan(m.node_value(p)));

    // Inside the domain nothing changes.
    m.var_mut(vid(x)).value = 8.0;
    full_evaluate(m);
    CHECK(m.node_value(s) == std::sqrt(8.0));
    CHECK(m.node_value(l) == std::log(8.0));
    CHECK(m.node_value(p) == std::pow(8.0, 1.0 / 3.0));
}

TEST_CASE("log(0) stays -inf, at either sign of zero", "[dag][domain]") {
    Model m;
    const int32_t x = m.float_var(-1, 1, "x");
    const int32_t l = m.log_expr(x);
    m.minimize(l);
    m.close();
    m.var_mut(vid(x)).value = 0.0;
    CHECK(eval_at(m, l) == -kInf);
    m.var_mut(vid(x)).value = -0.0;
    CHECK(eval_at(m, l) == -kInf);
}

TEST_CASE("pow keeps the sign of an overflow and NaN for a domain error", "[dag][domain]") {
    Model m;
    const int32_t b = m.float_var(-1000, 1000, "b");
    const int32_t e = m.float_var(-1000, 1000, "e");
    const int32_t p = m.pow_expr(b, e);
    m.minimize(p);
    m.close();

    auto at = [&](double base, double exp) {
        m.var_mut(vid(b)).value = base;
        m.var_mut(vid(e)).value = exp;
        return eval_at(m, p);
    };
    CHECK(at(10.0, 400.0) == kInf);
    CHECK(at(-10.0, 401.0) == -kInf);  // odd integer power of a negative overflows to -inf
    CHECK(at(-10.0, 400.0) == kInf);
    CHECK(at(0.0, -1.0) == kInf);  // the pole
    CHECK(std::isnan(at(-2.0, 0.5)));
    CHECK(at(-2.0, 3.0) == -8.0);
}

TEST_CASE("a near-zero denominator signs the quotient by num * denom, and 0/0 is NaN",
          "[dag][domain]") {
    Model m;
    const int32_t n = m.float_var(-10, 10, "n");
    const int32_t d = m.float_var(-10, 10, "d");
    const int32_t q = m.div_expr(n, d);
    m.minimize(q);
    m.close();

    auto at = [&](double num, double den) {
        m.var_mut(vid(n)).value = num;
        m.var_mut(vid(d)).value = den;
        return eval_at(m, q);
    };
    CHECK(at(1.0, 1e-16) == kInf);
    CHECK(at(1.0, -1e-16) == -kInf);
    CHECK(at(-1.0, 1e-16) == -kInf);
    CHECK(at(-1.0, -1e-16) == kInf);
    CHECK(at(1.0, 0.0) == kInf);
    CHECK(at(1.0, -0.0) == -kInf);
    CHECK(std::isnan(at(0.0, 0.0)));
    CHECK(std::isnan(at(-0.0, -0.0)));
    // 0 over a tiny nonzero denominator has no pole to stand in for.
    CHECK(at(0.0, 1e-16) == 0.0);
    // Ordinary division is untouched.
    CHECK(at(3.0, 2.0) == 1.5);
}

TEST_CASE("an infinite numerator over an exact zero stays an infinity", "[dag][domain]") {
    Model m;
    const int32_t z = m.float_var(-1, 1, "z");
    const int32_t q = m.div_expr(m.constant(-kInf), z);
    m.minimize(q);
    m.close();
    m.var_mut(vid(z)).value = 0.0;
    CHECK(eval_at(m, q) == -kInf);
    m.var_mut(vid(z)).value = -0.0;
    CHECK(eval_at(m, q) == kInf);
}

TEST_CASE("a comparison over a domain error is maximally violated", "[dag][domain]") {
    // The issue's residual: geq(pow(x, 0.5), 2) at x = -10 was 2 - inf = -inf.
    Model m = load(kPowModel);
    m.var_mut(0).value = -10.0;
    full_evaluate(m);
    ViolationManager vm(m);
    CHECK(vm.constraint_violation(0) == kInfPenalty);
    CHECK_FALSE(vm.is_feasible());
}

// ---------------------------------------------------------------------------
// Ops that consume a NaN child must not turn it back into a defined value
// ---------------------------------------------------------------------------
//
// A NaN constant stands in for any undefined child. Each of these used to read
// it as a finite, satisfiable value; a domain error underneath one of them would
// otherwise reach a row as an ordinary number after all.

TEST_CASE("min and max propagate a NaN child wherever it sits", "[dag][domain]") {
    Model m;
    const int32_t nan = m.constant(kNaN);
    const int32_t five = m.constant(5.0);
    const int32_t mx_last = m.max_expr({five, nan});
    const int32_t mx_first = m.max_expr({nan, five});
    const int32_t mn_last = m.min_expr({five, nan});
    const int32_t mn_mid = m.min_expr({five, nan, m.constant(7.0)});
    const int32_t row = m.leq(mx_last, m.constant(6.0));
    m.add_constraint(row);
    m.close();
    full_evaluate(m);
    CHECK(std::isnan(m.node_value(mx_last)));
    CHECK(std::isnan(m.node_value(mx_first)));
    CHECK(std::isnan(m.node_value(mn_last)));
    CHECK(std::isnan(m.node_value(mn_mid)));
    ViolationManager vm(m);
    CHECK_FALSE(vm.is_feasible());
}

TEST_CASE("neq over a NaN side is violated, not satisfied", "[dag][domain]") {
    Model m;
    const int32_t row = m.neq(m.constant(kNaN), m.constant(1.0));
    m.add_constraint(row);
    m.close();
    full_evaluate(m);
    ViolationManager vm(m);
    CHECK_FALSE(vm.is_feasible());
}

TEST_CASE("if with a NaN condition is undefined, not its else branch", "[dag][domain]") {
    Model m;
    const int32_t ite = m.if_then_else(m.constant(kNaN), m.constant(1.0), m.constant(0.0));
    m.add_constraint(m.leq(ite, m.constant(0.5)));
    m.close();
    CHECK(std::isnan(eval_at(m, ite)));
    ViolationManager vm(m);
    CHECK_FALSE(vm.is_feasible());
}

TEST_CASE("at with a NaN index is undefined, and an out-of-range one still reads 0",
          "[dag][domain]") {
    Model m;
    const int32_t lst = m.list_var(3, "l");
    const int32_t i = m.float_var(-1e30, 1e30, "i");
    const int32_t a = m.at(lst, i);
    m.minimize(a);
    m.close();
    m.var_mut(vid(i)).value = kNaN;
    CHECK(std::isnan(eval_at(m, a)));
    m.var_mut(vid(i)).value = 1e30;  // used to be a cast past int range: UB
    CHECK(eval_at(m, a) == 0.0);
    m.var_mut(vid(i)).value = 1.7;  // truncates toward zero, as before
    CHECK(eval_at(m, a) == static_cast<double>(m.var(vid(lst)).elements[1]));
}

TEST_CASE("signpower of a NaN is NaN, not an infinity", "[dag][domain]") {
    Model m;
    const int32_t sp = m.signpower_expr(m.constant(kNaN), m.constant(2.0));
    m.minimize(sp);
    m.close();
    CHECK(std::isnan(eval_at(m, sp)));
}

// ---------------------------------------------------------------------------
// Reverse-mode AD
// ---------------------------------------------------------------------------

TEST_CASE("no slope is offered where log's value is NaN", "[dag][domain][ad]") {
    Model m;
    const int32_t x = m.float_var(-10, 10, "x");
    const int32_t l = m.log_expr(x);
    m.minimize(l);
    m.close();
    m.var_mut(vid(x)).value = -2.0;
    full_evaluate(m);
    CHECK(compute_partial(m, l, vid(x)) == 0.0);  // was 1/x = -0.5
    m.var_mut(vid(x)).value = 2.0;
    full_evaluate(m);
    CHECK(compute_partial(m, l, vid(x)) == 0.5);
}

TEST_CASE("a div partial that overflows does not poison a sibling's partial", "[dag][domain][ad]") {
    // f = (n / d) * y at n = 1e300, d = 1e-5, y = 0. The quotient is a finite
    // 1e305, but d(n/d)/dd = -n/d^2 = -1e310 overflows to -inf. The edge into the
    // quotient carries y = 0, and the sweep multiplies the two: 0 * -inf = NaN
    // in d's partial, although f does not move with d at all here.
    Model m;
    const int32_t n = m.float_var(-1e300, 1e300, "n");
    const int32_t d = m.float_var(-1, 1, "d");
    const int32_t y = m.float_var(-1, 1, "y");
    const int32_t q = m.div_expr(n, d);
    const int32_t f = m.prod(q, y);
    m.minimize(f);
    m.close();
    m.var_mut(vid(n)).value = 1e300;
    m.var_mut(vid(d)).value = 1e-5;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    const std::vector<double> g = compute_all_partials(m, f);
    for (double v : g) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("tan's partial is finite at a non-finite argument", "[dag][domain][ad]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    const int32_t t = m.tan_expr(x);
    m.minimize(t);
    m.close();
    m.var_mut(vid(x)).value = kInf;  // cos(inf) is NaN: sec^2 was NaN
    full_evaluate(m);
    CHECK(std::isfinite(compute_partial(m, t, vid(x))));
}

// ---------------------------------------------------------------------------
// LNS acceptance with a NaN objective
// ---------------------------------------------------------------------------

TEST_CASE("LNS accepts a repair that turns a NaN objective finite", "[lns][domain]") {
    // obj = e - e with e = exp(1000 x): inf - inf = NaN for x > ~0.71, 0 below.
    // state_key compared (violation, objective) lexicographically, and
    // `0 < NaN` is false, so no repair could ever displace a NaN-objective
    // incumbent at equal violation: LNS was stuck on it.
    int accepted = 0;
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        INFO("seed " << seed);
        Model m;
        const int32_t x = m.float_var(-10, 10, "x");
        const int32_t e = m.exp_expr(m.prod(m.constant(1000.0), x));
        m.minimize(m.sum({e, m.neg(e)}));
        m.close();
        m.var_mut(vid(x)).value = 5.0;
        full_evaluate(m);
        REQUIRE(std::isnan(m.node_value(m.objective_id())));

        ViolationManager vm(m);
        RNG rng(seed);
        LNS lns(1.0);
        if (lns.destroy_repair(m, vm, rng, /*repair_time_limit=*/0.0)) {
            ++accepted;
            CHECK(std::isfinite(m.node_value(m.objective_id())));
        }
    }
    REQUIRE(accepted > 0);
}
