// Tests for activity-based bound propagation (#120) and its use by the MPS and
// `.nl` adapters. The property that matters throughout is soundness: a derived bound
// is entailed by the constraints, so it may shrink the box but must never put
// a feasible point outside it.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/bound_propagation.h>
#include <cbls/io_mps.h>
#include <cbls/io_nl.h>
#include <cbls/model.h>
#include <cbls/search.h>
#include <cmath>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace cbls;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

LinearRow row(std::vector<int32_t> cols, std::vector<double> coefs, double lo, double hi) {
    LinearRow r;
    r.cols = std::move(cols);
    r.coefs = std::move(coefs);
    r.lo = lo;
    r.hi = hi;
    return r;
}

}  // namespace

TEST_CASE("bound propagation derives an upper bound from a <= row") {
    // x + y <= 10, both non-negative and unbounded above.
    std::vector<double> lb{0.0, 0.0};
    std::vector<double> ub{kInf, kInf};
    std::vector<uint8_t> integral{0, 0};
    auto stats = propagate_bounds({row({0, 1}, {1.0, 1.0}, -kInf, 10.0)}, integral, lb, ub);

    CHECK(ub[0] <= 10.0 + 1e-6);
    CHECK(ub[0] >= 10.0);  // relaxed outward, never below the true implied bound
    CHECK(ub[1] <= 10.0 + 1e-6);
    CHECK(stats.n_tightened == 2);
    CHECK(stats.n_finitized == 2);
    CHECK_FALSE(stats.infeasible);
}

TEST_CASE("bound propagation finitizes the single unbounded term of a row") {
    // x + y <= 10 with y in [0, 4]: x is bounded above by 10 even though it was
    // declared free. This single-infinite-contribution case is the one that
    // matters on real instances -- with two unbounded terms nothing is implied.
    std::vector<double> lb{-kInf, 0.0};
    std::vector<double> ub{kInf, 4.0};
    std::vector<uint8_t> integral{0, 0};
    propagate_bounds({row({0, 1}, {1.0, 1.0}, -kInf, 10.0)}, integral, lb, ub);

    CHECK(ub[0] >= 10.0);
    CHECK(ub[0] <= 10.0 + 1e-6);
    CHECK(lb[0] == -kInf);  // nothing bounds x below
}

TEST_CASE("bound propagation leaves a row with two unbounded terms alone") {
    std::vector<double> lb{-kInf, -kInf};
    std::vector<double> ub{kInf, kInf};
    std::vector<uint8_t> integral{0, 0};
    auto stats = propagate_bounds({row({0, 1}, {1.0, 1.0}, -kInf, 10.0)}, integral, lb, ub);

    CHECK(lb[0] == -kInf);
    CHECK(ub[0] == kInf);
    CHECK(stats.n_tightened == 0);
    CHECK(stats.n_finitized == 0);
}

TEST_CASE("bound propagation handles negative coefficients") {
    // -2x + y <= -6, y in [0, 4]  =>  2x >= 6 + y_min  =>  x >= 3.
    std::vector<double> lb{0.0, 0.0};
    std::vector<double> ub{kInf, 4.0};
    std::vector<uint8_t> integral{0, 0};
    propagate_bounds({row({0, 1}, {-2.0, 1.0}, -kInf, -6.0)}, integral, lb, ub);

    CHECK(lb[0] <= 3.0);
    CHECK(lb[0] >= 3.0 - 1e-6);
    CHECK(ub[0] == kInf);
}

TEST_CASE("bound propagation rounds a derived bound inward on an integer column") {
    // 2x <= 5 with x integral => x <= 2, not 2.5.
    std::vector<double> lb{0.0};
    std::vector<double> ub{kInf};
    std::vector<uint8_t> integral{1};
    propagate_bounds({row({0}, {2.0}, -kInf, 5.0)}, integral, lb, ub);

    CHECK_THAT(ub[0], WithinAbs(2.0, 1e-12));
}

TEST_CASE("bound propagation does not round an integral bound off a feasible point") {
    // 3x <= 9: the implied bound is exactly 3, and floating-point slop in the
    // activity sum must not turn it into 2.
    std::vector<double> lb{0.0};
    std::vector<double> ub{kInf};
    std::vector<uint8_t> integral{1};
    propagate_bounds({row({0}, {3.0}, -kInf, 9.0)}, integral, lb, ub);

    CHECK_THAT(ub[0], WithinAbs(3.0, 1e-12));
}

TEST_CASE("bound propagation fixes a column whose box collapses to a point") {
    // x == 7 as a range row.
    std::vector<double> lb{-kInf};
    std::vector<double> ub{kInf};
    std::vector<uint8_t> integral{1};
    auto stats = propagate_bounds({row({0}, {1.0}, 7.0, 7.0)}, integral, lb, ub);

    CHECK_THAT(lb[0], WithinAbs(7.0, 1e-12));
    CHECK_THAT(ub[0], WithinAbs(7.0, 1e-12));
    CHECK(stats.n_fixed == 1);
    CHECK(stats.n_finitized == 1);
}

TEST_CASE("bound propagation never widens a bound that is already tighter") {
    // The row implies x <= 100, but x was declared with ub 5.
    std::vector<double> lb{0.0};
    std::vector<double> ub{5.0};
    std::vector<uint8_t> integral{0};
    auto stats = propagate_bounds({row({0}, {1.0}, -kInf, 100.0)}, integral, lb, ub);

    CHECK_THAT(ub[0], WithinAbs(5.0, 1e-12));
    CHECK(stats.n_tightened == 0);
}

TEST_CASE("bound propagation reaches a fixed point across chained rows") {
    // x <= 10; y <= x; z <= y. One pass in row order already resolves it; the
    // point is that the chain's bound reaches z rather than stopping at y.
    std::vector<double> lb{0.0, 0.0, 0.0};
    std::vector<double> ub{kInf, kInf, kInf};
    std::vector<uint8_t> integral{0, 0, 0};
    std::vector<LinearRow> rows{
        row({0}, {1.0}, -kInf, 10.0),
        row({1, 0}, {1.0, -1.0}, -kInf, 0.0),
        row({2, 1}, {1.0, -1.0}, -kInf, 0.0),
    };
    auto stats = propagate_bounds(rows, integral, lb, ub);

    CHECK(ub[2] <= 10.0 + 1e-6);
    CHECK(stats.n_finitized == 3);
    CHECK_FALSE(stats.hit_pass_limit);
}

TEST_CASE("bound propagation stops at the pass cap") {
    // Reversing the chain above forces one pass per link, so a cap of 1 leaves
    // the far end untightened and says so.
    std::vector<double> lb{0.0, 0.0, 0.0};
    std::vector<double> ub{kInf, kInf, kInf};
    std::vector<uint8_t> integral{0, 0, 0};
    std::vector<LinearRow> rows{
        row({2, 1}, {1.0, -1.0}, -kInf, 0.0),
        row({1, 0}, {1.0, -1.0}, -kInf, 0.0),
        row({0}, {1.0}, -kInf, 10.0),
    };
    BoundPropagationOptions opts;
    opts.max_passes = 1;
    auto stats = propagate_bounds(rows, integral, lb, ub, opts);

    CHECK(stats.passes == 1);
    CHECK(stats.hit_pass_limit);
    CHECK(ub[2] == kInf);
    CHECK_THAT(ub[0], WithinAbs(10.0, 1e-6));
}

TEST_CASE("bound propagation reports an empty derived box as infeasible") {
    // x >= 5 and x <= 3.
    std::vector<double> lb{-kInf};
    std::vector<double> ub{kInf};
    std::vector<uint8_t> integral{0};
    auto stats = propagate_bounds({row({0}, {1.0}, 5.0, kInf), row({0}, {1.0}, -kInf, 3.0)},
                                  integral, lb, ub);

    CHECK(stats.infeasible);
}

TEST_CASE("bound propagation treats a sentinel magnitude as infinite") {
    // 1e20 and beyond is "no bound" by the MPS/CPLEX/SCIP convention, so a
    // column declared that wide is finitized rather than left as-is.
    std::vector<double> lb{0.0};
    std::vector<double> ub{1e30};
    std::vector<uint8_t> integral{0};
    auto stats = propagate_bounds({row({0}, {1.0}, -kInf, 42.0)}, integral, lb, ub);

    CHECK(ub[0] <= 42.0 + 1e-6);
    CHECK(stats.n_finitized == 1);
}

TEST_CASE("bound propagation rejects malformed input") {
    std::vector<double> lb{0.0};
    std::vector<double> ub{1.0};
    std::vector<uint8_t> integral{0};
    LinearRow ragged;
    ragged.cols = {0};
    ragged.coefs = {1.0, 2.0};
    CHECK_THROWS_AS(propagate_bounds({ragged}, integral, lb, ub), std::invalid_argument);
    CHECK_THROWS_AS(propagate_bounds({row({3}, {1.0}, 0.0, 1.0)}, integral, lb, ub),
                    std::invalid_argument);
    std::vector<double> short_ub{};
    CHECK_THROWS_AS(propagate_bounds({}, integral, lb, short_ub), std::invalid_argument);
}

namespace {

// Two free columns whose only solution sits far outside any sane clamp:
//   x - y == 0
//   x     == 5e9
// The optimum is x = y = 5e9. A fixed clamp of 1e7 puts it outside the box;
// propagation derives [5e9, 5e9] for both and keeps it inside.
MpsProblem far_optimum_problem() {
    MpsProblem p;
    p.name = "FAR";
    p.vars = {
        MpsVar{"X", -kMpsInf, kMpsInf, MpsVarKind::Continuous},
        MpsVar{"Y", -kMpsInf, kMpsInf, MpsVarKind::Continuous},
    };
    p.rows = {
        MpsRow{"C1", MpsRowSense::E, 0.0, 0.0},
        MpsRow{"C2", MpsRowSense::E, 5.0e9, 0.0},
    };
    p.nonzeros = {
        MpsNonzero{0, 0, 1.0},
        MpsNonzero{0, 1, -1.0},  // x - y == 0
        MpsNonzero{1, 0, 1.0},   // x == 5e9
    };
    return p;
}

}  // namespace

TEST_CASE("mps adapter clamps the optimum away without propagation") {
    // The behaviour propagation replaces, pinned so the fix is visibly a fix.
    MpsToModelOptions opts;
    opts.propagate_bounds = false;
    opts.inf_clamp = 1.0e7;
    auto result = mps_to_model(far_optimum_problem(), opts);

    const Variable& x = result.model.var(0);
    CHECK(x.ub == 1.0e7);  // the only solution, at 5e9, is outside the box
    CHECK(result.n_clamped_columns == 2);
}

TEST_CASE("mps adapter keeps the optimum inside the box via propagation") {
    MpsToModelOptions opts;
    opts.propagate_bounds = true;
    opts.inf_clamp = 1.0e7;
    auto result = mps_to_model(far_optimum_problem(), opts);

    for (int j = 0; j < 2; ++j) {
        const Variable& v = result.model.var(j);
        CHECK(v.lb <= 5.0e9);
        CHECK(v.ub >= 5.0e9);
    }
    CHECK(result.bound_stats.n_finitized == 2);
    CHECK(result.n_clamped_columns == 0);
}

TEST_CASE("mps adapter propagation lets the search reach a far optimum") {
    MpsToModelOptions opts;
    opts.inf_clamp = 1.0e7;
    auto result = mps_to_model(far_optimum_problem(), opts);

    SearchConfig cfg;
    cfg.max_iterations = 200000;
    SearchResult r = solve(result.model, /*time_limit=*/-1.0, /*seed=*/7, /*use_fj=*/true, nullptr,
                           nullptr, 3, nullptr, cfg);

    REQUIRE(r.feasible);
    CHECK_THAT(r.best_state.values[0], WithinAbs(5.0e9, 1e-3));
    CHECK_THAT(r.best_state.values[1], WithinAbs(5.0e9, 1e-3));
}

TEST_CASE("mps adapter derives integer bounds through the clamp path") {
    // A free integer column bounded only by a row: 3z <= 21 with z >= 0.
    MpsProblem p;
    p.name = "INTB";
    p.vars = {MpsVar{"Z", 0.0, kMpsInf, MpsVarKind::Integer}};
    p.rows = {MpsRow{"C1", MpsRowSense::L, 21.0, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 3.0}};

    MpsToModelOptions opts;
    opts.inf_clamp = 1.0e7;
    auto result = mps_to_model(p, opts);

    const Variable& z = result.model.var(0);
    CHECK_THAT(z.lb, WithinAbs(0.0, 1e-12));
    CHECK_THAT(z.ub, WithinAbs(7.0, 1e-12));
    CHECK(result.n_clamped_columns == 0);
}

TEST_CASE("mps adapter falls back to the raw box when propagation proves infeasibility") {
    // x >= 5 and x <= 3 as two rows. Propagation empties the box; the adapter
    // must still build a model rather than throw on lb > ub.
    MpsProblem p;
    p.name = "INFEAS";
    p.vars = {MpsVar{"X", 0.0, 10.0, MpsVarKind::Continuous}};
    p.rows = {MpsRow{"C1", MpsRowSense::G, 5.0, 0.0}, MpsRow{"C2", MpsRowSense::L, 3.0, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 1.0}, MpsNonzero{1, 0, 1.0}};

    auto result = mps_to_model(p);

    CHECK(result.bound_stats.infeasible);
    // The discarded run's counts must not survive it: nothing was applied, so
    // reporting a tightening would credit the pass with work it undid.
    CHECK(result.bound_stats.n_tightened == 0);
    CHECK(result.bound_stats.n_finitized == 0);
    CHECK(result.bound_stats.passes == 0);
    const Variable& x = result.model.var(0);
    CHECK_THAT(x.lb, WithinAbs(0.0, 1e-12));
    CHECK_THAT(x.ub, WithinAbs(10.0, 1e-12));
}

TEST_CASE("mps adapter honours a finite bound wider than the clamp") {
    // Pre-#120 the clamp narrowed *any* column wider than its magnitude, a
    // declared one included. A declared bound is part of the instance, so
    // narrowing it can lose solutions exactly as substituting for an infinite
    // one can; only a missing bound is invented now.
    MpsProblem p;
    p.name = "WIDE";
    p.vars = {MpsVar{"X", -1.0e12, 1.0e12, MpsVarKind::Continuous}};
    p.rows = {MpsRow{"C1", MpsRowSense::G, -1.0e11, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 1.0}};

    MpsToModelOptions opts;
    opts.inf_clamp = 1.0e7;
    auto result = mps_to_model(p, opts);

    const Variable& x = result.model.var(0);
    CHECK_THAT(x.ub, WithinAbs(1.0e12, 1.0));
    CHECK(x.lb <= -1.0e11);  // the row's implied bound, not the declared -1e12
    CHECK(x.lb >= -1.0e11 - 1.0);
    CHECK(result.n_clamped_columns == 0);
}

namespace {

// One linear row `2*x0 + x1 <= 12` over two columns declared free. Nothing but
// the row bounds them, so what the engine ends up searching is entirely a
// question of whether implied bounds were derived.
NlProblem free_columns_problem(bool discrete) {
    NlProblem p;
    p.name = "NLFREE";
    p.n_vars = 2;
    p.n_cons = 1;
    p.var_is_discrete = {static_cast<uint8_t>(discrete ? 1 : 0),
                         static_cast<uint8_t>(discrete ? 1 : 0)};
    p.n_discrete_vars = discrete ? 2 : 0;
    p.var_bounds.resize(2);
    for (NlVarBound& b : p.var_bounds) {
        b.type = NlBoundType::Lower;  // x >= 0, unbounded above
        b.lower = 0.0;
        b.upper = kNlInf;
    }
    NlConstraint c;
    c.linear = {NlLinTerm{0, 2.0}, NlLinTerm{1, 1.0}};
    c.bound.type = NlBoundType::Upper;
    c.bound.lower = -kNlInf;
    c.bound.upper = 12.0;
    p.constraints = {c};
    return p;
}

}  // namespace

TEST_CASE("nl adapter derives implied bounds for a free continuous column") {
    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    auto result = nl_to_model(free_columns_problem(/*discrete=*/false), opts);

    REQUIRE(result.supported);
    CHECK_THAT(result.model.var(0).ub, WithinAbs(6.0, 1e-6));   // 2*x0 <= 12
    CHECK_THAT(result.model.var(1).ub, WithinAbs(12.0, 1e-6));  // x1 <= 12
    CHECK(result.n_clamped_columns == 0);
    CHECK(result.bound_stats.n_finitized == 2);
}

TEST_CASE("nl adapter derives implied bounds for a free integer column") {
    // The path that used to hand a free Int the ±1e6 int_inf_clamp box.
    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    opts.int_inf_clamp = 1.0e4;
    auto result = nl_to_model(free_columns_problem(/*discrete=*/true), opts);

    REQUIRE(result.supported);
    CHECK(result.model.var(0).type == VarType::Int);
    CHECK_THAT(result.model.var(0).ub, WithinAbs(6.0, 1e-9));
    CHECK_THAT(result.model.var(1).ub, WithinAbs(12.0, 1e-9));
    CHECK(result.n_clamped_columns == 0);
}

TEST_CASE("nl adapter falls back to the clamp where no row bounds the column") {
    NlProblem p = free_columns_problem(/*discrete=*/false);
    p.constraints.clear();
    p.n_cons = 0;

    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    auto result = nl_to_model(p, opts);

    REQUIRE(result.supported);
    CHECK_THAT(result.model.var(0).ub, WithinAbs(1.0e5, 1e-9));
    CHECK(result.n_clamped_columns == 2);
    CHECK(result.bound_stats.n_finitized == 0);
}

TEST_CASE("bound propagation is conservative when a column repeats in a row") {
    // `x + x <= 10` is written as two terms, not merged. The exact implied bound
    // is x <= 5; the pass derives the weaker x <= 10 because it treats the two
    // occurrences as independent. Weaker is sound — it keeps every feasible
    // point — and this pins that it never derives the *stronger* 5, which would
    // be right here but wrong for `x + y` with y aliased to x by other rows.
    std::vector<double> lb{0.0};
    std::vector<double> ub{kInf};
    std::vector<uint8_t> integral{0};
    propagate_bounds({row({0, 0}, {1.0, 1.0}, -kInf, 10.0)}, integral, lb, ub);

    CHECK(ub[0] >= 10.0);
    CHECK(ub[0] <= 10.0 + 1e-6);
}

TEST_CASE("bound propagation does not hand back a sub-tolerance crossed box") {
    // Two rows pin x from both sides 2.5e-9 apart. Each derived bound is relaxed
    // outward by 1e-9, so the box crosses by 5e-10 — too little to be called
    // infeasible, and exactly the shape accumulated rounding produces. A crossed
    // box must not escape: the MPS adapter throws on `lb > ub`, which would drop
    // the instance from a run with a build_error.
    std::vector<double> lb{-kInf};
    std::vector<double> ub{kInf};
    std::vector<uint8_t> integral{0};
    auto stats = propagate_bounds(
        {row({0}, {1.0}, 5.0, kInf), row({0}, {1.0}, -kInf, 5.0 - 2.5e-9)}, integral, lb, ub);

    CHECK_FALSE(stats.infeasible);
    CHECK(lb[0] <= ub[0]);
}

TEST_CASE("mps adapter survives a sub-tolerance crossed box") {
    // The same shape through the adapter, where a crossed box would throw.
    MpsProblem p;
    p.name = "CROSS";
    p.vars = {MpsVar{"X", -kMpsInf, kMpsInf, MpsVarKind::Continuous}};
    p.rows = {MpsRow{"C1", MpsRowSense::G, 5.0, 0.0},
              MpsRow{"C2", MpsRowSense::L, 5.0 - 2.5e-9, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 1.0}, MpsNonzero{1, 0, 1.0}};

    REQUIRE_NOTHROW(mps_to_model(p));
}

TEST_CASE("mps adapter clips an integer column to the int range and says so") {
    // A finite bound is honoured however wide, so a declared integer domain
    // beyond int32 now reaches variable creation. `Model::int_var` takes an int,
    // so the column is clipped — a narrowing, and counted as one.
    MpsProblem p;
    p.name = "BIGINT";
    p.vars = {MpsVar{"Z", 3.0e9, 4.0e9, MpsVarKind::Integer}};
    p.rows = {MpsRow{"C1", MpsRowSense::L, 1.0e10, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 1.0}};

    auto result = mps_to_model(p);

    // Both bounds sit above INT_MAX, so both clip to it. Clipping only the far
    // side of each bound leaves the lower one unclipped, and casting it to int
    // is undefined — in practice it wraps to INT_MIN and the domain inverts.
    const double kIntHi = static_cast<double>(std::numeric_limits<int>::max());
    const Variable& z = result.model.var(0);
    CHECK_THAT(z.lb, WithinAbs(kIntHi, 1.0));
    CHECK_THAT(z.ub, WithinAbs(kIntHi, 1.0));
    CHECK(result.n_clamped_columns == 1);
}

TEST_CASE("nl adapter ignores rows with a nonlinear part when deriving bounds") {
    // Activity arithmetic does not apply to a nonlinear body, so such a row must
    // contribute nothing. Here the *linear* part alone would imply x0 <= 6, but
    // the row is `2*x0 + x1 + (x0*x0) <= 12` and no bound may be derived from it.
    NlProblem p = free_columns_problem(/*discrete=*/false);
    NlExpr nl;
    nl.nodes.resize(3);
    nl.nodes[0].kind = NlNodeKind::Var;
    nl.nodes[0].index = 0;
    nl.nodes[1].kind = NlNodeKind::Var;
    nl.nodes[1].index = 0;
    nl.nodes[2].kind = NlNodeKind::Op;
    nl.nodes[2].opcode = 2;  // OPMULT
    nl.nodes[2].children = {0, 1};
    nl.root = 2;
    p.constraints[0].nonlinear = nl;

    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    auto result = nl_to_model(p, opts);

    REQUIRE(result.supported);
    CHECK(result.bound_stats.n_finitized == 0);
    CHECK_THAT(result.model.var(0).ub, WithinAbs(1.0e5, 1e-9));
    CHECK(result.n_clamped_columns == 2);
}

TEST_CASE("nl adapter skips a row naming a column that does not exist") {
    // `build_linear` drops an out-of-range term silently and io_nl.h promises to
    // skip a malformed file rather than throw, so propagation must not be the
    // one thing that throws. The whole row is dropped: no bound is derived.
    NlProblem p = free_columns_problem(/*discrete=*/false);
    p.constraints[0].linear.push_back(NlLinTerm{5, 1.0});  // n_vars is 2

    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    NlToModelResult result;
    REQUIRE_NOTHROW(result = nl_to_model(p, opts));
    CHECK(result.bound_stats.n_finitized == 0);
    CHECK_THAT(result.model.var(0).ub, WithinAbs(1.0e5, 1e-9));
}

TEST_CASE("nl adapter falls back to the raw box when propagation proves infeasibility") {
    // The twin of the MPS case. The two adapters hand-duplicate this block, so
    // testing only one lets them drift.
    NlProblem p = free_columns_problem(/*discrete=*/false);
    for (NlVarBound& b : p.var_bounds) {
        b.type = NlBoundType::Range;
        b.lower = 0.0;
        b.upper = 10.0;
    }
    NlConstraint lo;
    lo.linear = {NlLinTerm{0, 1.0}};
    lo.bound.type = NlBoundType::Lower;
    lo.bound.lower = 5.0;
    lo.bound.upper = kNlInf;
    NlConstraint hi;
    hi.linear = {NlLinTerm{0, 1.0}};
    hi.bound.type = NlBoundType::Upper;
    hi.bound.lower = -kNlInf;
    hi.bound.upper = 3.0;
    p.constraints = {lo, hi};
    p.n_cons = 2;

    auto result = nl_to_model(p);

    CHECK(result.bound_stats.infeasible);
    CHECK(result.bound_stats.n_tightened == 0);
    CHECK_THAT(result.model.var(0).lb, WithinAbs(0.0, 1e-12));
    CHECK_THAT(result.model.var(0).ub, WithinAbs(10.0, 1e-12));
}

TEST_CASE("nl adapter honours a NaN bound as no bound rather than throwing") {
    // propagate_bounds rejects NaN outright; the adapter must not let one reach
    // it, because io_nl.h promises to skip a malformed file, not to throw.
    NlProblem p = free_columns_problem(/*discrete=*/false);
    p.var_bounds[0].lower = std::numeric_limits<double>::quiet_NaN();

    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    NlToModelResult result;
    REQUIRE_NOTHROW(result = nl_to_model(p, opts));
    CHECK_THAT(result.model.var(0).lb, WithinAbs(-1.0e5, 1e-9));
}

TEST_CASE("nl adapter propagation can be turned off") {
    NlToModelOptions opts;
    opts.inf_clamp = 1.0e5;
    opts.propagate_bounds = false;
    auto result = nl_to_model(free_columns_problem(/*discrete=*/false), opts);

    CHECK_THAT(result.model.var(0).ub, WithinAbs(1.0e5, 1e-9));
    CHECK(result.bound_stats.passes == 0);
    CHECK(result.n_clamped_columns == 2);
}

TEST_CASE("mps adapter counts a doubly-narrowed column once") {
    // An unbounded-below Integer column whose upper bound is beyond int32: the
    // clamp supplies the lower bound and the int32 clip narrows the upper. One
    // column, so n_clamped_columns must be 1, not 2.
    MpsProblem p;
    p.name = "TWICE";
    p.vars = {MpsVar{"Z", -kMpsInf, 5.0e9, MpsVarKind::Integer}};
    p.rows = {MpsRow{"C1", MpsRowSense::L, 1.0e11, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 1.0}};

    MpsToModelOptions opts;
    opts.inf_clamp = 1.0e7;
    auto result = mps_to_model(p, opts);

    CHECK(result.n_clamped_columns == 1);
}

TEST_CASE("mps adapter honours a NaN bound as no bound rather than throwing") {
    // The twin of the `.nl` case. `propagate_bounds` rejects NaN outright, so
    // without sanitising here the same file would build under
    // propagate_bounds=false and throw under the default.
    MpsProblem p;
    p.name = "NANB";
    p.vars = {MpsVar{"X", std::numeric_limits<double>::quiet_NaN(), 10.0, MpsVarKind::Continuous}};
    p.rows = {MpsRow{"C1", MpsRowSense::L, 8.0, 0.0}};
    p.nonzeros = {MpsNonzero{0, 0, 1.0}};

    MpsToModelOptions opts;
    opts.inf_clamp = 1.0e7;
    MpsToModelResult result;
    REQUIRE_NOTHROW(result = mps_to_model(p, opts));
    CHECK_THAT(result.model.var(0).lb, WithinAbs(-1.0e7, 1e-9));
    CHECK_THAT(result.model.var(0).ub, WithinAbs(8.0, 1e-6));
}
