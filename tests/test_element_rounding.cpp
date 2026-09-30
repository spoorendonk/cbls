// #186: `element` (a table looked up by Int expressions), `ceil`/`floor`/`round`,
// and the `lambda_sum`/`pair_lambda_sum` forms whose functor also reads other
// decisions (`extra`).
//
// Pinned here, per op: the VALUE against a hand computation (edge cases
// included -- out-of-range indices, half-way rounding, non-finite input), the
// agreement between incremental and from-scratch evaluation (after every list
// move generator for the lambda forms, and after an extra changes), the `.cbls`
// round-trip or refusal, and FJ's behaviour -- that the plateau edges and index
// values are offered as jump candidates where the zero derivative offers
// nothing. The last case is a line-planning-flavoured model written twice, once
// with these ops and once with the workarounds they replace (an auxiliary Int, an
// if-chain, one lambda per type), which is the reference the new encoding is
// checked against.

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <functional>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace cbls;

namespace {

double eval_at(Model& m, int32_t node) {
    full_evaluate(m);
    return m.node_value(node);
}

}  // namespace

// ---------------------------------------------------------------------------
// element
// ---------------------------------------------------------------------------

TEST_CASE("element reads a table by an Int index and 0.0 outside it", "[dag][element]") {
    Model m;
    const int32_t t = m.int_var(-5, 10, "t");
    const int32_t e = m.element({5.0, 6.0, 7.0, 8.0}, t);
    m.minimize(e);
    m.close();

    for (int i = 0; i < 4; ++i) {
        m.var_mut(vid(t)).value = i;
        REQUIRE(eval_at(m, e) == 5.0 + i);
    }
    // Out of range on either side reads 0.0 -- `at`'s rule for a position past
    // its List.
    m.var_mut(vid(t)).value = 4;
    REQUIRE(eval_at(m, e) == 0.0);
    m.var_mut(vid(t)).value = -1;
    REQUIRE(eval_at(m, e) == 0.0);
}

TEST_CASE("element truncates a fractional index toward zero, as at does", "[dag][element]") {
    Model m;
    const int32_t x = m.float_var(-10, 10, "x");
    const int32_t e = m.element({5.0, 6.0, 7.0}, x);
    m.minimize(e);
    m.close();

    m.var_mut(vid(x)).value = 2.7;
    REQUIRE(eval_at(m, e) == 7.0);
    // -0.5 truncates to 0, which is inside the table.
    m.var_mut(vid(x)).value = -0.5;
    REQUIRE(eval_at(m, e) == 5.0);
    m.var_mut(vid(x)).value = 3.0;
    REQUIRE(eval_at(m, e) == 0.0);
    // Non-finite and far out of int range are "outside", not undefined
    // behaviour in a cast.
    for (double v : {std::numeric_limits<double>::quiet_NaN(),
                     std::numeric_limits<double>::infinity(), 1e30, -1e30}) {
        m.var_mut(vid(x)).value = v;
        REQUIRE(eval_at(m, e) == 0.0);
    }
}

TEST_CASE("a two-index element reads table[row][col] and 0.0 if either is outside",
          "[dag][element]") {
    Model m;
    const int32_t r = m.int_var(-1, 3, "r");
    const int32_t c = m.int_var(-1, 4, "c");
    const std::vector<std::vector<double>> table = {{1, 2, 3}, {4, 5, 6}};
    const int32_t e = m.element(table, r, c);
    m.minimize(e);
    m.close();

    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            m.var_mut(vid(r)).value = i;
            m.var_mut(vid(c)).value = j;
            REQUIRE(eval_at(m, e) == table[i][j]);
        }
    }
    m.var_mut(vid(r)).value = 1;
    m.var_mut(vid(c)).value = 3;
    REQUIRE(eval_at(m, e) == 0.0);
    m.var_mut(vid(r)).value = 2;
    m.var_mut(vid(c)).value = 0;
    REQUIRE(eval_at(m, e) == 0.0);
    m.var_mut(vid(r)).value = -1;
    REQUIRE(eval_at(m, e) == 0.0);
}

TEST_CASE("element takes an index EXPRESSION and follows it under delta evaluation",
          "[dag][element]") {
    Model m;
    auto t = m.Int(0, 5, "t");
    auto e = element({10.0, 20.0, 30.0, 40.0}, t - 1.0);  // table[t - 1]
    m.minimize(e);
    m.close();
    m.var_mut(t.var_id()).value = 1;
    full_evaluate(m);
    REQUIRE(m.node_value(e.handle) == 10.0);

    m.var_mut(t.var_id()).value = 4;
    delta_evaluate(m, {t.var_id()});
    REQUIRE(m.node_value(e.handle) == 40.0);
    m.var_mut(t.var_id()).value = 0;  // index -1: outside
    delta_evaluate(m, {t.var_id()});
    REQUIRE(m.node_value(e.handle) == 0.0);
}

TEST_CASE("element refuses an empty or ragged table and a structured index", "[dag][element]") {
    Model m;
    const int32_t t = m.int_var(0, 3, "t");
    const int32_t lv = m.list_var(3, "l");
    REQUIRE_THROWS_AS(m.element(std::vector<double>{}, t), std::invalid_argument);
    REQUIRE_THROWS_AS(m.element(std::vector<std::vector<double>>{}, t, t), std::invalid_argument);
    REQUIRE_THROWS_AS(m.element(std::vector<std::vector<double>>{{}}, t, t), std::invalid_argument);
    REQUIRE_THROWS_AS(m.element(std::vector<std::vector<double>>{{1, 2}, {3}}, t, t),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(m.element({1.0, 2.0}, lv), std::invalid_argument);
    REQUIRE_THROWS_AS(m.element({1.0, 2.0}, 12345), std::out_of_range);
    // Nothing half-registered: the model still builds and evaluates.
    const int32_t ok = m.element({1.0, 2.0}, t);
    m.minimize(ok);
    m.close();
    REQUIRE(eval_at(m, ok) == 1.0);
}

// ---------------------------------------------------------------------------
// ceil / floor / round
// ---------------------------------------------------------------------------

TEST_CASE("ceil floor and round take std's rules, half-way away from zero", "[dag][rounding]") {
    Model m;
    auto x = m.Float(-10, 10, "x");
    auto c = ceil(x);
    auto f = floor(x);
    auto r = round(x);
    m.minimize(c + f + r);
    m.close();

    struct Row {
        double x, ceil, floor, round;
    };
    const std::vector<Row> rows = {
        {2.0, 2.0, 2.0, 2.0},     {2.3, 3.0, 2.0, 2.0},     {2.5, 3.0, 2.0, 3.0},
        {2.7, 3.0, 2.0, 3.0},     {-2.3, -2.0, -3.0, -2.0}, {-2.5, -2.0, -3.0, -3.0},
        {-0.5, -0.0, -1.0, -1.0}, {0.0, 0.0, 0.0, 0.0},
    };
    for (const Row& row : rows) {
        INFO("x = " << row.x);
        m.var_mut(x.var_id()).value = row.x;
        full_evaluate(m);
        REQUIRE(m.node_value(c.handle) == row.ceil);
        REQUIRE(m.node_value(f.handle) == row.floor);
        REQUIRE(m.node_value(r.handle) == row.round);
    }
    m.var_mut(x.var_id()).value = std::numeric_limits<double>::infinity();
    full_evaluate(m);
    REQUIRE(m.node_value(c.handle) == std::numeric_limits<double>::infinity());
    m.var_mut(x.var_id()).value = std::numeric_limits<double>::quiet_NaN();
    full_evaluate(m);
    REQUIRE(std::isnan(m.node_value(r.handle)));
}

TEST_CASE("the piecewise-constant ops have a zero local derivative", "[dag][rounding][ad]") {
    Model m;
    auto x = m.Float(0, 10, "x");
    auto t = m.Int(0, 3, "t");
    auto c = ceil(2.0 * x);
    auto f = floor(x);
    auto r = round(x);
    auto e = element({1.0, 5.0, 9.0, 13.0}, t);
    m.minimize(c + f + r + e);
    m.close();
    m.var_mut(x.var_id()).value = 1.3;
    m.var_mut(t.var_id()).value = 1;
    full_evaluate(m);
    for (const Expr& node : {c, f, r}) {
        REQUIRE(compute_partial(m, node.handle, x.var_id()) == 0.0);
    }
    REQUIRE(compute_partial(m, e.handle, t.var_id()) == 0.0);
}

// ---------------------------------------------------------------------------
// lambda_sum / pair_lambda_sum over extra decisions
// ---------------------------------------------------------------------------

namespace {

// A stop cost by vehicle type, the epic's `c[i][type_r]`, plus a second extra so
// the order of the extras is observable.
double stop_cost(int i, ConstSpan<double> x) {
    return (10.0 * i) + (100.0 * x[0]) + x[1];
}

// Asymmetric, so a reversed or rotated tour is distinguishable.
double leg_cost(int a, int b, ConstSpan<double> x) {
    return ((10.0 * a) + b) * (1.0 + x[0]);
}

struct ExtraModel {
    Model m;
    int32_t lv = 0;
    int32_t type = 0;
    int32_t scale = 0;
    int32_t stops = 0;
    int32_t open = 0;
    int32_t cyclic = 0;
};

// A variable-length List read by all three extra-lambda variants. `type` is an
// extra; `scale` feeds one through a node (2 * scale), so an extra that is a
// NODE is covered as well as one that is a variable.
void build_extra_model(ExtraModel& em) {
    Model& m = em.m;
    em.lv = m.list_var(6, 0, 6, ListInit::Empty, "route");
    em.type = m.int_var(0, 2, "type");
    em.scale = m.int_var(0, 3, "scale");
    const int32_t twice = m.prod(m.constant(2.0), em.scale);
    em.stops = m.lambda_sum(em.lv, stop_cost, {em.type, twice});
    em.open = m.pair_lambda_sum(em.lv, leg_cost, PairMode::Open, {em.type});
    em.cyclic = m.pair_lambda_sum(em.lv, leg_cost, PairMode::Cyclic, {em.type});
    m.minimize(m.sum({em.stops, em.open, em.cyclic}));
    m.close();
}

double expect_stops(const std::vector<int32_t>& el, double type, double scale) {
    double s = 0.0;
    for (int32_t e : el) {
        s += (10.0 * e) + (100.0 * type) + (2.0 * scale);
    }
    return s;
}

double expect_pairs(const std::vector<int32_t>& el, double type, bool cyclic) {
    double s = 0.0;
    for (size_t k = 0; k + 1 < el.size(); ++k) {
        s += ((10.0 * el[k]) + el[k + 1]) * (1.0 + type);
    }
    if (cyclic && el.size() >= 2) {
        s += ((10.0 * el.back()) + el.front()) * (1.0 + type);
    }
    return s;
}

void require_extra_values(const ExtraModel& em) {
    const auto& el = em.m.var(vid(em.lv)).elements;
    const double type = em.m.var(vid(em.type)).value;
    const double scale = em.m.var(vid(em.scale)).value;
    REQUIRE(em.m.node_value(em.stops) == expect_stops(el, type, scale));
    REQUIRE(em.m.node_value(em.open) == expect_pairs(el, type, false));
    REQUIRE(em.m.node_value(em.cyclic) == expect_pairs(el, type, true));
}

}  // namespace

TEST_CASE("the extra-lambda forms hand the functor the extras' current values",
          "[dag][lambda_extra]") {
    ExtraModel em;
    build_extra_model(em);
    Model& m = em.m;

    SECTION("n = 0, 1, 2 and a general n") {
        for (const std::vector<int32_t>& el :
             std::vector<std::vector<int32_t>>{{}, {3}, {1, 4}, {2, 0, 5, 1}}) {
            m.var_mut(vid(em.lv)).elements = el;
            m.var_mut(vid(em.type)).value = 2;
            m.var_mut(vid(em.scale)).value = 3;
            full_evaluate(m);
            INFO("n = " << el.size());
            require_extra_values(em);
        }
    }

    SECTION("a change to an extra alone re-sums the list under delta evaluation") {
        m.var_mut(vid(em.lv)).elements = {2, 0, 5, 1};
        full_evaluate(m);
        m.var_mut(vid(em.type)).value = 1;
        delta_evaluate(m, {vid(em.type)});
        require_extra_values(em);
        // Through a node child: scale feeds `2 * scale`.
        m.var_mut(vid(em.scale)).value = 2;
        delta_evaluate(m, {vid(em.scale)});
        require_extra_values(em);
    }
}

TEST_CASE("every list move keeps the extra-lambda forms equal to a full evaluation",
          "[dag][lambda_extra][moves]") {
    ExtraModel em;
    build_extra_model(em);
    Model& m = em.m;
    m.var_mut(vid(em.lv)).elements = {4, 1, 3};
    m.var_mut(vid(em.type)).value = 1;
    m.var_mut(vid(em.scale)).value = 2;
    full_evaluate(m);

    RNG rng(186);
    std::vector<std::string> seen;
    for (int round = 0; round < 80; ++round) {
        auto moves = generate_standard_moves(m.var(vid(em.lv)), rng);
        REQUIRE_FALSE(moves.empty());
        for (const Move& mv : moves) {
            seen.push_back(mv.move_type);
            std::vector<int32_t> changed = apply_move(m, mv);
            delta_evaluate(m, changed.data(), changed.size());
            INFO("move " << mv.move_type);
            require_extra_values(em);
        }
        // Interleave an extra change, so the list is also edited under an
        // extra that moved since the last full pass.
        m.var_mut(vid(em.type)).value = static_cast<double>(round % 3);
        delta_evaluate(m, {vid(em.type)});
        require_extra_values(em);
    }
    for (const char* want :
         {"list_swap", "list_2opt", "list_relocate", "list_insert", "list_remove"}) {
        INFO("generator " << want);
        REQUIRE(std::find(seen.begin(), seen.end(), want) != seen.end());
    }
}

TEST_CASE("the extra-lambda forms refuse a scalar list, a structured extra and an empty func",
          "[dag][lambda_extra]") {
    Model m;
    const int32_t lv = m.list_var(3, "l");
    const int32_t other = m.list_var(3, "other");
    const int32_t t = m.int_var(0, 2, "t");
    REQUIRE_THROWS_AS(m.lambda_sum(t, stop_cost, {t}), std::invalid_argument);
    REQUIRE_THROWS_AS(m.lambda_sum(m.constant(1.0), stop_cost, {t}), std::invalid_argument);
    REQUIRE_THROWS_AS(m.lambda_sum(lv, stop_cost, {other}), std::invalid_argument);
    REQUIRE_THROWS_AS(m.lambda_sum(lv, LambdaExtraFunc{}, {t}), std::invalid_argument);
    REQUIRE_THROWS_AS(m.lambda_sum(lv, stop_cost, {9999}), std::out_of_range);
    REQUIRE_THROWS_AS(m.pair_lambda_sum(lv, leg_cost, PairMode::Open, {other}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(m.pair_lambda_sum(lv, PairLambdaExtraFunc{}, PairMode::Open, {t}),
                      std::invalid_argument);
}

// ---------------------------------------------------------------------------
// .cbls I/O
// ---------------------------------------------------------------------------

TEST_CASE("a .cbls round-trip preserves element tables and the rounding ops", "[io][element]") {
    Model m;
    const int32_t t = m.int_var(0, 3, "t");
    const int32_t u = m.int_var(0, 2, "u");
    const int32_t x = m.float_var(-5, 5, "x");
    const int32_t e1 = m.element({1.5, 2.5, 3.5, 4.5}, t);
    const int32_t e2 = m.element(std::vector<std::vector<double>>{{1, 2, 3}, {4, 5, 6}}, u, t);
    const int32_t c = m.ceil_expr(x);
    const int32_t f = m.floor_expr(x);
    const int32_t r = m.round_expr(x);
    m.add_constraint(m.leq(e2, m.constant(5.0)));
    m.minimize(m.sum({e1, e2, c, f, r}));
    m.close();

    // A reload renumbers the nodes in the order the file lists them, so the
    // first save and the second can differ in node names alone; from the
    // second on the text is a fixed point. What must survive is every table and
    // every op, which the fixed point and the value check below pin together.
    std::ostringstream first;
    save_model(m, first);
    std::istringstream in(first.str());
    Model loaded = load_model(in);
    std::ostringstream second;
    save_model(loaded, second);
    std::istringstream in2(second.str());
    Model reloaded = load_model(in2);
    std::ostringstream third;
    save_model(reloaded, third);
    REQUIRE(second.str() == third.str());
    REQUIRE(second.str().find(R"("table":[[1.0,2.0,3.0],[4.0,5.0,6.0]])") != std::string::npos);
    REQUIRE(second.str().find(R"("table":[1.5,2.5,3.5,4.5])") != std::string::npos);

    // And the loaded model evaluates as the original, at a few assignments.
    for (const auto& [tv, uv, xv] : std::vector<std::tuple<int, int, double>>{
             {0, 0, -2.5}, {3, 1, 2.5}, {2, 1, 0.4}, {1, 0, -0.6}}) {
        for (Model* mm : {&m, &loaded}) {
            mm->var_mut(vid(t)).value = tv;
            mm->var_mut(vid(u)).value = uv;
            mm->var_mut(vid(x)).value = xv;
            full_evaluate(*mm);
        }
        REQUIRE(loaded.node_value(loaded.objective_id()) == m.node_value(m.objective_id()));
    }
}

TEST_CASE("save_model refuses an extra-lambda node before writing anything", "[io][lambda_extra]") {
    for (bool pair : {false, true}) {
        Model m;
        const int32_t lv = m.list_var(3, "l");
        const int32_t t = m.int_var(0, 2, "t");
        const int32_t node = pair ? m.pair_lambda_sum(lv, leg_cost, PairMode::Cyclic, {t})
                                  : m.lambda_sum(lv, stop_cost, {t, t});
        m.minimize(node);
        m.close();
        std::ostringstream out;
        INFO("pair = " << pair);
        REQUIRE_THROWS_AS(save_model(m, out), std::runtime_error);
        REQUIRE(out.str().empty());
    }
}

// ---------------------------------------------------------------------------
// FJ: breakpoint and index-value candidates
// ---------------------------------------------------------------------------

TEST_CASE("FJ offers every index value of an element as the index's jump candidates",
          "[fj][element]") {
    // A wide Int domain (1000 values), so `int_jump_candidates` gives only its
    // coarse grid and x +/- 1 -- neither of which contains 517, the one index
    // whose entry satisfies the row. The zero derivative offers nothing either.
    constexpr int kN = 1000;
    std::vector<double> table(kN, 0.0);
    table[517] = 10.0;
    Model m;
    const int32_t t = m.int_var(0, kN - 1, "t");
    m.add_constraint(m.geq(m.element(table, t), m.constant(5.0)));
    m.close();
    m.var_mut(vid(t)).value = 0;
    full_evaluate(m);
    ViolationManager vm(m);

    const JumpResult r = compute_var_jump(m, vm.weights, vid(t));
    REQUIRE(r.jump_value == 517.0);
    REQUIRE(r.score == 5.0);
}

TEST_CASE("FJ offers a ceil's plateau edges to a Float with no usable gradient", "[fj][rounding]") {
    // fleet = ceil(2.5 * f) must be exactly 8, i.e. 2.8 < f <= 3.2. From f = 0
    // the Newton step sees a zero gradient, and the three box candidates (0, 10,
    // 20) each make the fleet far too large -- so without the edge candidates
    // there is no improving jump at all.
    Model m;
    auto f = m.Float(0, 20, "f");
    auto fleet = ceil(2.5 * f);
    m.add_constraint(fleet >= 8.0);
    m.add_constraint(fleet <= 8.0);
    m.close();
    m.var_mut(f.var_id()).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);

    const JumpResult r = compute_var_jump(m, vm.weights, f.var_id());
    REQUIRE(r.score == 8.0);
    REQUIRE(r.jump_value > 2.8);
    REQUIRE(r.jump_value <= 3.2);
}

TEST_CASE("FJ offers a floor's edges and a round's half-way points as candidates",
          "[fj][rounding]") {
    SECTION("floor over a Float lands inside the target plateau") {
        Model m;
        auto x = m.Float(0, 10, "x");
        m.add_constraint(floor(x).eq(Expr{&m, m.constant(3.0)}));
        m.close();
        m.var_mut(x.var_id()).value = 0.0;
        full_evaluate(m);
        ViolationManager vm(m);
        // The box midpoint 5 alone would improve the row by 1; the edge at 3
        // clears it.
        const JumpResult r = compute_var_jump(m, vm.weights, x.var_id());
        REQUIRE(r.score == 3.0);
        REQUIRE(r.jump_value >= 3.0);
        REQUIRE(r.jump_value < 4.0);
    }
    SECTION("round over an Int past the grid's reach") {
        // round(x / 7) == 150 holds for x in 1047..1053; the Int grid over
        // [0, 1500] steps by ~47 and misses all seven.
        Model m;
        auto x = m.Int(0, 1500, "x");
        m.add_constraint(round(x / 7.0).eq(Expr{&m, m.constant(150.0)}));
        m.close();
        m.var_mut(x.var_id()).value = 0.0;
        full_evaluate(m);
        ViolationManager vm(m);
        const JumpResult r = compute_var_jump(m, vm.weights, x.var_id());
        REQUIRE(r.score == 150.0);
        REQUIRE(r.jump_value >= 1047.0);
        REQUIRE(r.jump_value <= 1053.0);
    }
}

TEST_CASE("a row over a piecewise-constant op is not scored in closed form",
          "[fj][linear_jump][rounding]") {
    // LinearJumpScorer must decline every row touching the new ops, so the
    // column takes the exact probe; the jump through the scorer is then the
    // probe's jump.
    Model m;
    auto x = m.Int(0, 1500, "x");
    auto y = m.Int(0, 10, "y");
    auto t = m.Int(0, 3, "t");
    m.add_constraint(x + y <= 1200.0);  // linear: eligible
    m.add_constraint(round(x / 7.0) + y >= 150.0);
    m.add_constraint(element({0.0, 1.0, 2.0, 3.0}, t) + y >= 4.0);
    m.add_constraint(ceil(1.5 * y) + floor(0.5 * x) <= 900.0);
    m.close();
    full_evaluate(m);
    ViolationManager vm(m);
    RNG rng(186);
    FeasibilityJump fj(m, vm, rng);
    REQUIRE(fj.linear_scorer().row_eligible(0));
    for (int32_t c = 1; c < 4; ++c) {
        INFO("row " << c);
        REQUIRE_FALSE(fj.linear_scorer().row_eligible(c));
    }
    for (int32_t v : {x.var_id(), y.var_id(), t.var_id()}) {
        const JumpResult probe = compute_var_jump(m, vm.weights, v);
        const JumpResult fast = compute_var_jump(m, vm.weights, v, false, &fj.linear_scorer());
        REQUIRE(fast.jump_value == probe.jump_value);
        REQUIRE(fast.score == probe.score);
    }
}

// ---------------------------------------------------------------------------
// A line-planning-flavoured model, in both encodings
// ---------------------------------------------------------------------------
//
// Two lines over four shared stops. Each line has a route (a variable-length
// List), an Int vehicle type and an Int frequency (vehicles per hour). Per line:
//
//   cycle  = round trip over the route, in minutes      (cyclic pair_lambda_sum)
//   fleet  = ceil(cycle * freq / 60)                     vehicles to run it
//   cost   = fleet * vehicle_cost[type]                  by vehicle type
//          + sum_{i in route} stop_cost[i][type]         by vehicle type
//   capacity[type] * freq >= demand(route)               seats per hour
//
// and every stop is served by at least one line. The new encoding writes the
// three rules directly: `ceil`, `element` and a `lambda_sum` reading the type
// through `extra`. The reference encoding is what a model had to write before
// #186: an auxiliary Int fleet with `60 * fleet >= cycle * freq`, an if-chain
// over the type values, and one lambda per type behind the same if-chain.

namespace {

constexpr int kStops = 4;
constexpr int kLines = 2;
constexpr int kTypes = 3;

// Minutes between stops; asymmetric only mildly, so a route and its reverse can
// still differ.
const std::vector<std::vector<double>> kTravel = {
    {0, 7, 12, 9}, {8, 0, 6, 11}, {12, 6, 0, 5}, {9, 12, 5, 0}};
const std::vector<double> kVehicleCost = {100.0, 160.0, 250.0};
const std::vector<double> kCapacity = {40.0, 70.0, 120.0};
const std::vector<double> kDemand = {60.0, 90.0, 50.0, 110.0};  // riders/hour per stop
// Stop 3 is a narrow street: the largest type pays heavily to call there.
const std::vector<std::vector<double>> kStopCost = {{5, 6, 8}, {4, 5, 7}, {6, 7, 9}, {3, 6, 60}};

double travel(int a, int b) {
    return kTravel[a][b];
}

struct LineVars {
    int32_t route = 0;
    int32_t type = 0;
    int32_t freq = 0;
    int32_t fleet_aux = -1;  // reference encoding only
};

// `If(cond, a, b)` picks `a` when cond > 0: the if-chain over type values.
int32_t by_type(Model& m, int32_t type, const std::vector<int32_t>& options) {
    int32_t out = options.back();
    for (int k = kTypes - 2; k >= 0; --k) {
        const int32_t cond = m.sum({m.constant(k + 0.5), m.neg(type)});  // type <= k
        out = m.if_then_else(cond, options[k], out);
    }
    return out;
}

struct LineModel {
    Model m;
    std::vector<LineVars> lines;
};

void build_line_model(LineModel& lm, bool new_ops) {
    Model& m = lm.m;
    std::vector<int32_t> cost_terms;
    std::vector<std::vector<int32_t>> serves(kStops);
    for (int l = 0; l < kLines; ++l) {
        LineVars v;
        v.route = m.list_var(kStops, 2, kStops, ListInit::Random, "route" + std::to_string(l));
        v.type = m.int_var(0, kTypes - 1, "type" + std::to_string(l));
        v.freq = m.int_var(1, 6, "freq" + std::to_string(l));
        const int32_t cycle = m.pair_lambda_sum(v.route, travel, PairMode::Cyclic);
        const int32_t veh_minutes = m.prod(cycle, v.freq);
        const int32_t demand =
            m.lambda_sum(v.route, [](int i) { return kDemand[static_cast<size_t>(i)]; });

        int32_t fleet = 0;
        int32_t veh_cost = 0;
        int32_t cap = 0;
        int32_t stop_cost_sum = 0;
        if (new_ops) {
            fleet = m.ceil_expr(m.div_expr(veh_minutes, m.constant(60.0)));
            veh_cost = m.element(kVehicleCost, v.type);
            cap = m.element(kCapacity, v.type);
            stop_cost_sum = m.lambda_sum(
                v.route,
                [](int i, ConstSpan<double> x) {
                    return kStopCost[static_cast<size_t>(i)][static_cast<size_t>(x[0])];
                },
                {v.type});
        } else {
            v.fleet_aux = m.int_var(0, 40, "fleet" + std::to_string(l));
            fleet = v.fleet_aux;
            m.add_constraint(m.geq(m.prod(m.constant(60.0), fleet), veh_minutes));
            std::vector<int32_t> vc;
            std::vector<int32_t> cp;
            std::vector<int32_t> sc;
            for (int k = 0; k < kTypes; ++k) {
                vc.push_back(m.constant(kVehicleCost[k]));
                cp.push_back(m.constant(kCapacity[k]));
                sc.push_back(m.lambda_sum(v.route, [k](int i) {
                    return kStopCost[static_cast<size_t>(i)][static_cast<size_t>(k)];
                }));
            }
            veh_cost = by_type(m, v.type, vc);
            cap = by_type(m, v.type, cp);
            stop_cost_sum = by_type(m, v.type, sc);
        }
        m.add_constraint(m.geq(m.prod(cap, v.freq), demand));
        cost_terms.push_back(m.prod(fleet, veh_cost));
        cost_terms.push_back(stop_cost_sum);
        for (int s = 0; s < kStops; ++s) {
            serves[s].push_back(m.lambda_sum(v.route, [s](int i) { return i == s ? 1.0 : 0.0; }));
        }
        lm.lines.push_back(v);
    }
    for (int s = 0; s < kStops; ++s) {
        m.add_constraint(m.geq(m.sum(serves[s]), m.constant(1.0)));
    }
    m.minimize(m.sum(cost_terms));
    m.close();
}

// The cost of one line at a given route/type/freq, straight from the data --
// the brute-force reference both encodings are held to. +inf when the
// capacity row fails.
double line_cost(const std::vector<int32_t>& route, int type, int freq) {
    double cycle = 0.0;
    double demand = 0.0;
    double stops = 0.0;
    for (size_t k = 0; k < route.size(); ++k) {
        cycle += travel(route[k], route[(k + 1) % route.size()]);
        demand += kDemand[route[k]];
        stops += kStopCost[route[k]][type];
    }
    if (kCapacity[type] * freq < demand) {
        return std::numeric_limits<double>::infinity();
    }
    const double fleet = std::ceil(cycle * freq / 60.0);
    return (fleet * kVehicleCost[type]) + stops;
}

// Every ordered route of 2..4 distinct stops.
std::vector<std::vector<int32_t>> all_routes() {
    std::vector<std::vector<int32_t>> out;
    std::function<void(std::vector<int32_t>&)> grow = [&](std::vector<int32_t>& cur) {
        if (cur.size() >= 2) {
            out.push_back(cur);
        }
        for (int s = 0; s < kStops; ++s) {
            if (std::find(cur.begin(), cur.end(), s) == cur.end()) {
                cur.push_back(s);
                grow(cur);
                cur.pop_back();
            }
        }
    };
    std::vector<int32_t> cur;
    grow(cur);
    return out;
}

double brute_force_optimum() {
    const auto routes = all_routes();
    // Best cost per route over type and frequency, then over route pairs that
    // cover every stop.
    std::vector<double> best(routes.size(), std::numeric_limits<double>::infinity());
    for (size_t r = 0; r < routes.size(); ++r) {
        for (int type = 0; type < kTypes; ++type) {
            for (int freq = 1; freq <= 6; ++freq) {
                best[r] = std::min(best[r], line_cost(routes[r], type, freq));
            }
        }
    }
    double opt = std::numeric_limits<double>::infinity();
    for (size_t a = 0; a < routes.size(); ++a) {
        for (size_t b = 0; b < routes.size(); ++b) {
            std::vector<bool> covered(kStops, false);
            for (int32_t s : routes[a]) {
                covered[s] = true;
            }
            for (int32_t s : routes[b]) {
                covered[s] = true;
            }
            if (std::all_of(covered.begin(), covered.end(), [](bool c) { return c; })) {
                opt = std::min(opt, best[a] + best[b]);
            }
        }
    }
    return opt;
}

}  // namespace

TEST_CASE("line planning: the new ops agree with their workarounds at every assignment",
          "[element][rounding][lambda_extra][line_planning]") {
    // Deterministic equivalence: at random assignments, with the reference's
    // auxiliary fleet set to the ceil it stands for, both encodings give the same
    // objective and the same feasibility.
    LineModel a;
    LineModel b;
    build_line_model(a, true);
    build_line_model(b, false);
    const auto routes = all_routes();
    RNG rng(18606);
    for (int trial = 0; trial < 300; ++trial) {
        for (int l = 0; l < kLines; ++l) {
            const auto& route =
                routes[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(routes.size())))];
            const auto type = static_cast<double>(rng.integers(0, kTypes));
            const auto freq = static_cast<double>(rng.integers(1, 7));
            for (LineModel* lm : {&a, &b}) {
                const LineVars& v = lm->lines[static_cast<size_t>(l)];
                lm->m.var_mut(vid(v.route)).elements = route;
                lm->m.var_mut(vid(v.type)).value = type;
                lm->m.var_mut(vid(v.freq)).value = freq;
            }
            double cycle = 0.0;
            for (size_t k = 0; k < route.size(); ++k) {
                cycle += travel(route[k], route[(k + 1) % route.size()]);
            }
            b.m.var_mut(vid(b.lines[static_cast<size_t>(l)].fleet_aux)).value =
                std::ceil(cycle * freq / 60.0);
        }
        full_evaluate(a.m);
        full_evaluate(b.m);
        REQUIRE(a.m.node_value(a.m.objective_id()) == b.m.node_value(b.m.objective_id()));
        ViolationManager va(a.m);
        ViolationManager vb(b.m);
        REQUIRE(va.is_feasible() == vb.is_feasible());
    }
}

TEST_CASE("line planning: both encodings solve to the brute-force optimum",
          "[search][element][rounding][lambda_extra][line_planning]") {
    const double optimum = brute_force_optimum();
    REQUIRE(std::isfinite(optimum));
    for (bool new_ops : {true, false}) {
        for (uint64_t seed : {186U, 7U, 2026U}) {
            LineModel lm;
            build_line_model(lm, new_ops);
            INFO("new_ops = " << new_ops << ", seed " << seed);
            const SearchResult r = solve_deterministic(lm.m, 60000, seed);
            REQUIRE(r.feasible);
            REQUIRE(r.objective == optimum);
        }
    }
}
