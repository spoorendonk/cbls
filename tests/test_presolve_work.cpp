// Presolve working storage (#336): rows and columns that grow for fill-in.
// White-box: drives `presolve_detail::Work` directly, through the internal
// header, since no reduction adds entries yet.

#include "src/presolve_internal.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/presolve.h>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

using namespace cbls;
using cbls::presolve_detail::Work;

namespace {

/// `n` continuous columns in [0, 10] and the given rows, each `lo <= . <= hi`.
LinearProblem make(std::size_t n, const std::vector<std::vector<std::pair<int32_t, double>>>& rows,
                   double lo, double hi) {
    LinearProblem p;
    p.col_lb.assign(n, 0.0);
    p.col_ub.assign(n, 10.0);
    p.integral.assign(n, 0);
    p.frozen.assign(n, 0);
    p.obj.assign(n, 0.0);
    for (const auto& terms : rows) {
        for (const auto& [c, a] : terms) {
            p.cols.push_back(c);
            p.coefs.push_back(a);
        }
        p.row_start.push_back(p.cols.size());
        p.row_lo.push_back(lo);
        p.row_hi.push_back(hi);
    }
    return p;
}

void load_indexed(const LinearProblem& p, Work& w) {
    presolve_detail::load(p, w);
    presolve_detail::build_column_index(w);
}

/// Row r's entries (column, coefficient) in slot order, cleared ones included.
std::vector<std::pair<int32_t, double>> row_of(const Work& w, std::size_t r) {
    std::vector<std::pair<int32_t, double>> out;
    for (std::size_t p = w.row_begin[r]; p < w.row_end[r]; ++p) {
        out.emplace_back(w.cols[p], w.coefs[p]);
    }
    return out;
}

/// Column j's index entries as (row, coefficient the entry resolves to).
std::vector<std::pair<int32_t, double>> col_of(const Work& w, std::size_t j) {
    std::vector<std::pair<int32_t, double>> out;
    for (std::size_t q = w.col_begin[j]; q < w.col_end[j]; ++q) {
        const std::size_t p = w.entry_pos(q);
        REQUIRE(w.cols[p] == static_cast<int32_t>(j));  // every entry names its own slot
        out.emplace_back(w.col_rows[q], w.coefs[p]);
    }
    return out;
}

using Entries = std::vector<std::pair<int32_t, double>>;

}  // namespace

TEST_CASE("a full presolve row moves to the end to grow and then grows in place", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(make(4, {{{0, 1.0}, {1, 2.0}}, {{1, 4.0}}}, 0.0, 5.0), w);
    const std::size_t before = w.row_begin[0];

    REQUIRE(w.add_entry(0, 2, 3.0));
    CHECK(w.row_begin[0] != before);  // row 0 had no room: moved
    CHECK(row_of(w, 0) == Entries{{0, 1.0}, {1, 2.0}, {2, 3.0}});
    CHECK(row_of(w, 1) == Entries{{1, 4.0}});
    CHECK(w.row_live[0] == 3);
    CHECK(w.col_live[2] == 1);
    // The moved row's old entries still resolve through the column index.
    CHECK(col_of(w, 0) == Entries{{0, 1.0}});
    CHECK(col_of(w, 1) == Entries{{0, 2.0}, {1, 4.0}});
    CHECK(col_of(w, 2) == Entries{{0, 3.0}});

    const std::size_t moved = w.row_begin[0];
    REQUIRE(w.add_entry(0, 3, -1.0));
    CHECK(w.row_begin[0] == moved);  // room left over from the move
    CHECK(row_of(w, 0) == Entries{{0, 1.0}, {1, 2.0}, {2, 3.0}, {3, -1.0}});
    CHECK(col_of(w, 3) == Entries{{0, -1.0}});
}

TEST_CASE("a presolve row reuses its column's own cleared slot", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(make(3, {{{0, 1.0}, {1, 2.0}, {2, 3.0}}}, 0.0, 5.0), w);
    // Clear x1's entry, as a cancellation would.
    w.coefs[w.row_begin[0] + 1] = 0.0;
    --w.row_live[0];
    --w.col_live[1];

    const std::size_t begin = w.row_begin[0];
    REQUIRE(w.add_entry(0, 1, 7.0));
    CHECK(w.row_begin[0] == begin);
    CHECK(row_of(w, 0) == Entries{{0, 1.0}, {1, 7.0}, {2, 3.0}});
    CHECK(col_of(w, 1) == Entries{{0, 7.0}});  // listed once, not twice
    CHECK(w.row_live[0] == 3);
    CHECK(w.col_live[1] == 1);
}

TEST_CASE("a presolve column grows across many rows", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(
        make(2, {{{0, 1.0}}, {{0, 1.0}}, {{0, 1.0}}, {{0, 1.0}}, {{0, 1.0}}, {{0, 1.0}}}, 0.0, 5.0),
        w);
    Entries want;
    for (int32_t r = 0; r < 6; ++r) {
        REQUIRE(w.add_entry(r, 1, 1.0 + static_cast<double>(r)));
        want.emplace_back(r, 1.0 + static_cast<double>(r));
    }
    CHECK(col_of(w, 1) == want);
    CHECK(w.col_live[1] == 6);
    for (std::size_t r = 0; r < 6; ++r) {
        CHECK(row_of(w, r) == Entries{{0, 1.0}, {1, 1.0 + static_cast<double>(r)}});
    }
}

TEST_CASE("adding to a live presolve entry sums and a cancelling sum clears it", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(make(2, {{{0, 1.0}, {1, 2.0}}}, 0.0, 5.0), w);
    const std::size_t begin = w.row_begin[0];

    REQUIRE(w.add_entry(0, 1, 3.0));
    CHECK(w.row_begin[0] == begin);  // summed in place: no new slot
    CHECK(row_of(w, 0) == Entries{{0, 1.0}, {1, 5.0}});
    CHECK(col_of(w, 1) == Entries{{0, 5.0}});
    CHECK(w.row_live[0] == 2);
    CHECK(w.col_live[1] == 1);

    REQUIRE(w.add_entry(0, 1, -5.0));
    CHECK(row_of(w, 0) == Entries{{0, 1.0}, {1, 0.0}});
    CHECK(w.row_live[0] == 1);
    CHECK(w.col_live[1] == 0);
}

TEST_CASE("add_entry refuses a removed row or column and a zero or non-finite coefficient",
          "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(make(3, {{{0, 1e308}}, {{1, 1.0}}}, 0.0, 5.0), w);
    w.remove_row(1);
    w.col_removed[2] = 1;
    const std::vector<int32_t> row_live = w.row_live;
    const std::vector<int32_t> col_live = w.col_live;
    w.changed = false;

    CHECK_FALSE(w.add_entry(1, 0, 1.0));  // removed row
    CHECK_FALSE(w.add_entry(0, 2, 1.0));  // removed column
    CHECK_FALSE(w.add_entry(0, 1, 0.0));  // zero coefficient
    CHECK_FALSE(w.add_entry(0, 1, std::numeric_limits<double>::infinity()));
    CHECK_FALSE(w.add_entry(0, 1, std::numeric_limits<double>::quiet_NaN()));
    CHECK_FALSE(w.add_entry(0, 0, 1e308));  // the sum overflows
    CHECK(w.row_live == row_live);
    CHECK(w.col_live == col_live);
    CHECK(row_of(w, 0) == Entries{{0, 1e308}});
    CHECK_FALSE(w.changed);
}

TEST_CASE("a grown presolve row compacts into the result", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(make(4, {{{0, 1.0}, {1, 2.0}}, {{1, 4.0}}, {{3, 1.0}}}, 0.0, 5.0), w);
    const std::vector<double> lb0 = w.lb;
    const std::vector<double> ub0 = w.ub;
    REQUIRE(w.add_entry(0, 2, 3.0));   // row 0 moves to the end
    REQUIRE(w.add_entry(0, 0, -1.0));  // x0 cancels: a cleared slot to drop
    w.remove_row(1);

    const PresolveResult res = presolve_detail::build_result(w, lb0, ub0);
    const LinearProblem& out = res.reduced;
    REQUIRE(out.n_rows() == 2);
    CHECK(out.row_start == std::vector<std::size_t>{0, 2, 3});
    CHECK(out.cols == std::vector<int32_t>{1, 2, 3});
    CHECK(out.coefs == std::vector<double>{2.0, 3.0, 1.0});
    CHECK(res.row_to_reduced == std::vector<int32_t>{0, -1, 1});
}

TEST_CASE("bound propagation reads an entry added to a moved row", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    // x0 + x1 <= 4; adding x2 makes it x0 + x1 + x2 <= 4, so x2 <= 4.
    load_indexed(make(3, {{{0, 1.0}, {1, 1.0}}, {{1, 1.0}}}, 0.0, 4.0), w);
    const std::size_t before = w.row_begin[0];
    REQUIRE(w.add_entry(0, 2, 1.0));
    REQUIRE(w.row_begin[0] != before);
    w.changed = false;

    presolve_detail::reduce_propagate(w);
    CHECK_FALSE(w.infeasible);
    CHECK(w.changed);
    // Propagation keeps a small safety margin above the exact bound 4.
    CHECK(w.ub[2] >= 4.0);
    CHECK(w.ub[2] < 4.0 + 1e-6);
}

TEST_CASE("fixing a column folds an entry added to a moved row", "[presolve]") {
    const PresolveOptions opts;
    Work w(opts);
    load_indexed(make(3, {{{0, 1.0}, {1, 2.0}}}, 1.0, 9.0), w);
    REQUIRE(w.add_entry(0, 2, 3.0));
    REQUIRE(w.can_fix(2, 2.0));
    w.fix_column(2, 2.0);
    CHECK(w.row_lo[0] == -5.0);
    CHECK(w.row_hi[0] == 3.0);
    CHECK(row_of(w, 0) == Entries{{0, 1.0}, {1, 2.0}, {2, 0.0}});
    CHECK(w.row_live[0] == 2);
    CHECK(w.col_removed[2] == 1);
    CHECK(w.postsolve.num_steps() == 1);
}
