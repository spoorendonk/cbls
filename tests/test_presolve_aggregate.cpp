// Doubleton-equality aggregation (#328, #337): `a x + b y = c` eliminates y.

#include "presolve_test_util.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/presolve.h>
#include <cstdint>
#include <vector>

using namespace cbls;
using namespace cbls::presolve_test;

namespace {

/// Presolve with propagation and empty-column fixing off, so the expected
/// bounds come from aggregation (and the row reductions) alone, and the
/// column left in no row stays in the reduced problem.
PresolveResult run(const LinearProblem& p, int max_fill = PresolveOptions{}.max_fill,
                   double max_ratio = 1.0) {
    PresolveOptions o = only(false);
    o.empty_columns = false;
    o.max_fill = max_fill;
    o.max_aggregation_ratio = max_ratio;
    return presolve(p, o);
}

}  // namespace

TEST_CASE("aggregation eliminates the continuous column and moves its bounds", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 10, /*integral=*/true);
    const int32_t y = b.col(0, 4);
    b.row({{x, 1.0}, {y, 1.0}}, 5, 5);      // y = 5 - x
    b.row({{x, 1.0}, {y, 2.0}}, -kInf, 8);  // becomes -x <= -2
    const PresolveResult r = run(b.p);
    REQUIRE(r.stats.status == PresolveStatus::Reduced);
    CHECK(r.stats.cols_aggregated == 1);
    CHECK(r.stats.aggregation_fill == 0);  // x was already in the other row
    REQUIRE(r.reduced_to_col == std::vector<int32_t>{x});
    CHECK(r.reduced.col_lb[0] == 2.0);  // the substituted row, as a singleton
    CHECK(r.reduced.col_ub[0] == 5.0);  // y >= 0 moved onto x
    CHECK(r.postsolve.apply({3.0}) == std::vector<double>{3.0, 2.0});
}

TEST_CASE("aggregation keeps the ratio within max_aggregation_ratio", "[presolve]") {
    Builder b;
    const int32_t x = b.col(-10, 10);
    const int32_t y = b.col(-10, 10);
    b.row({{x, 3.0}, {y, 1.0}}, 3, 3);
    // Eliminating y needs |3/1| <= ratio; eliminating x needs |1/3|.
    const PresolveResult r = run(b.p);
    CHECK(r.stats.cols_aggregated == 1);
    CHECK(r.reduced_to_col == std::vector<int32_t>{y});
    const PresolveResult none = run(b.p, PresolveOptions{}.max_fill, 0.2);
    CHECK(none.stats.cols_aggregated == 0);
}

TEST_CASE("aggregation of an integral column needs integral ratio and side", "[presolve]") {
    SECTION("2x + 2y = 4: k = 1 and d = 2") {
        Builder b;
        const int32_t x = b.col(0, 5, true);
        const int32_t y = b.col(0, 5, true);
        b.row({{x, 2.0}, {y, 2.0}}, 4, 4);
        const PresolveResult r = run(b.p);
        CHECK(r.stats.cols_aggregated == 1);
        // A tie on integrality and row count: the lower index, x, goes.
        REQUIRE(r.reduced_to_col == std::vector<int32_t>{y});
        CHECK(r.postsolve.apply({0.0}) == std::vector<double>{2.0, 0.0});
        CHECK(r.postsolve.apply({5.0}) == std::vector<double>{-3.0, 5.0});
    }
    SECTION("2x + 2y = 3: d = 1.5") {
        Builder b;
        const int32_t x = b.col(0, 5, true);
        const int32_t y = b.col(0, 5, true);
        b.row({{x, 2.0}, {y, 2.0}}, 3, 3);
        CHECK(run(b.p).stats.cols_aggregated == 0);
        (void)x;
        (void)y;
    }
    SECTION("x + 2y = 4: k = 1/2 for y and |2| > 1 for x") {
        Builder b;
        const int32_t x = b.col(0, 5, true);
        const int32_t y = b.col(0, 5, true);
        b.row({{x, 1.0}, {y, 2.0}}, 4, 4);
        CHECK(run(b.p).stats.cols_aggregated == 0);
        // With ratio 2, x = 4 - 2y is eliminated: k = 2 and d = 4 are integers.
        const PresolveResult r = run(b.p, PresolveOptions{}.max_fill, 2.0);
        CHECK(r.stats.cols_aggregated == 1);
        CHECK(r.reduced_to_col == std::vector<int32_t>{y});
        CHECK(r.postsolve.apply({1.0}) == std::vector<double>{2.0, 1.0});
        (void)x;
    }
    SECTION("an integral y is not eliminated through a continuous x") {
        Builder b;
        const int32_t x = b.col(0, 5);
        const int32_t y = b.col(0, 5, true);
        const int32_t z = b.col(0, 5);
        b.row({{x, 1.0}, {y, 1.0}}, 3, 3);
        // Eliminating x would add y to both rows: fill 2, over max_fill 1,
        // so only y = 3 - x is left, which needs x integral.
        b.row({{x, 1.0}, {z, 1.0}}, -kInf, 3);
        b.row({{x, 1.0}, {z, -1.0}}, -kInf, 1);
        CHECK(run(b.p, 1).stats.cols_aggregated == 0);
        // Unlimited fill: the continuous x goes instead.
        const PresolveResult r = run(b.p);
        CHECK(r.stats.cols_aggregated == 1);
        CHECK(r.reduced_to_col == std::vector<int32_t>{y, z});
    }
}

TEST_CASE("aggregation never touches a frozen column", "[presolve]") {
    for (int frozen = 0; frozen < 2; ++frozen) {
        Builder b;
        const int32_t x = b.col(0, 5, false, 0.0, frozen == 0);
        const int32_t y = b.col(0, 5, false, 0.0, frozen == 1);
        b.row({{x, 1.0}, {y, 1.0}}, 3, 3);
        CHECK(run(b.p).stats.cols_aggregated == 0);
    }
}

TEST_CASE("aggregation reports bounds that cannot meet through the row as infeasible",
          "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 3, true);
    const int32_t y = b.col(1.25, 1.75);
    // The row's activity [1.25, 4.75] holds 3, so no row rule sees it, but
    // x = 3 - y in [1.25, 1.75] holds no integer.
    b.row({{x, 1.0}, {y, 1.0}}, 3, 3);
    CHECK(run(b.p).stats.status == PresolveStatus::Infeasible);
}

TEST_CASE("aggregation folds the eliminated column's objective", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 10, true, 1.0);
    const int32_t y = b.col(0, 10, false, 3.0);
    b.row({{x, 1.0}, {y, 1.0}}, 5, 5);  // y = 5 - x: x + 3y = 15 - 2x
    const PresolveResult r = run(b.p);
    REQUIRE(r.reduced_to_col == std::vector<int32_t>{x});
    CHECK(r.reduced.obj == std::vector<double>{-2.0});
    CHECK(r.reduced.obj_offset == 15.0);
}

TEST_CASE("a two-step aggregation chain postsolves in reverse order", "[presolve]") {
    Builder b;
    const int32_t a = b.col(0, 5, true);
    const int32_t u = b.col(-10, 10);
    const int32_t v = b.col(-10, 10);
    b.row({{u, 1.0}, {v, -1.0}}, 1, 1);  // u = 1 + v (u is in fewer rows)
    b.row({{v, 1.0}, {a, 1.0}}, 3, 3);   // then v = 3 - a
    const PresolveResult r = run(b.p);
    CHECK(r.stats.cols_aggregated == 2);
    REQUIRE(r.reduced_to_col == std::vector<int32_t>{a});
    CHECK(r.postsolve.apply({1.0}) == std::vector<double>{1.0, 3.0, 2.0});
}

TEST_CASE("aggregation respects max_fill", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1, true);
    const int32_t y = b.col(0, 1, true);
    const int32_t z = b.col(0, 3);
    b.row({{x, 1.0}, {y, 1.0}}, 1, 1);
    // Each of x and y is in two more rows the other is not: fill 2 either way.
    b.row({{y, 1.0}, {z, 1.0}}, -kInf, 3);
    b.row({{y, 2.0}, {z, 1.0}}, -kInf, 4);
    b.row({{x, 1.0}, {z, 1.0}}, -kInf, 3);  // none of the four is redundant
    b.row({{x, 2.0}, {z, 1.0}}, -kInf, 4);
    CHECK(run(b.p, 1).stats.cols_aggregated == 0);
    const PresolveResult r = run(b.p, 2);
    CHECK(r.stats.cols_aggregated == 1);
    CHECK(r.stats.aggregation_fill == 2);
}

TEST_CASE("x + y = 1 over binaries leaves y = 1 - x", "[presolve]") {
    Builder b;
    const int32_t x = b.col(0, 1, true);
    const int32_t y = b.col(0, 1, true);
    b.row({{x, 1.0}, {y, 1.0}}, 1, 1);
    const PresolveResult r = run(b.p);
    REQUIRE(r.stats.cols_aggregated == 1);
    REQUIRE(r.reduced.n_cols() == 1);
    CHECK(r.reduced.col_lb[0] == 0.0);
    CHECK(r.reduced.col_ub[0] == 1.0);
    for (const double v : {0.0, 1.0}) {
        const std::vector<double> back = r.postsolve.apply({v});
        CHECK(back[static_cast<std::size_t>(r.reduced_to_col[0])] == v);
        CHECK(back[0] + back[1] == 1.0);
    }
    (void)x;
    (void)y;
}
