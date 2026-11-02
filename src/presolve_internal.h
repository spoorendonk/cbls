#pragma once

// Working storage shared by the presolve reductions (src/presolve*.cpp).
// Not installed: the public interface is include/cbls/presolve.h.
//
// Rows are a flat CSR whose deleted entries keep their column index and have
// their coefficient set to exactly 0 (`propagate_bounds` skips a zero term, so
// the live rows can be handed to it as views without compaction). Each column
// lists its entries as (row, offset within the row). Removed rows and columns
// are marked, and everything is compacted once, when the result is built.

#include "cbls/presolve.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cbls::presolve_detail {

struct Work {
    const PresolveOptions& opts;

    // Columns.
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<double> obj;
    std::vector<uint8_t> integral;
    std::vector<uint8_t> frozen;
    std::vector<uint8_t> col_removed;
    std::vector<int32_t> col_live;  ///< Live entries per column.
    double obj_offset = 0.0;

    // Rows (flat CSR; a deleted entry has coefficient 0).
    std::vector<std::size_t> row_start;
    std::vector<int32_t> cols;
    std::vector<double> coefs;
    std::vector<double> row_lo;  ///< -inf when absent.
    std::vector<double> row_hi;  ///< +inf when absent.
    /// Magnitude of the row's original bounds plus everything folded into
    /// them since: what the rounding error in `row_lo`/`row_hi` scales with.
    std::vector<double> row_scale;
    /// Number of fixed-column terms folded into the row's bounds since load.
    std::vector<int32_t> row_folds;
    std::vector<uint8_t> row_removed;
    std::vector<int32_t> row_live;  ///< Live entries per row.

    // Column-major index: column j's entries are
    // (col_rows[q], col_offs[q]) for q in [col_start[j], col_start[j+1]).
    std::vector<std::size_t> col_start;
    std::vector<int32_t> col_rows;
    std::vector<int32_t> col_offs;

    Postsolve postsolve;
    PresolveStats stats;
    bool changed = false;
    bool infeasible = false;

    explicit Work(const PresolveOptions& o) : opts(o) {}

    [[nodiscard]] std::size_t n_cols() const { return lb.size(); }
    [[nodiscard]] std::size_t n_rows() const { return row_lo.size(); }

    /// Bound on the floating-point error in row r's current `row_lo`/`row_hi`:
    /// each fold `bound -= a * v` adds at most one rounding of the product and
    /// one of the difference, each within `DBL_EPSILON` of a partial sum no
    /// larger than `row_scale`, so `(row_folds + 1) * DBL_EPSILON * row_scale`.
    [[nodiscard]] double row_roundoff(std::size_t r) const;

    /// Remove column j at `value`: fold it into its rows' bounds and the
    /// objective offset, and push the postsolve step. j must not be frozen.
    void fix_column(int32_t j, double value);

    /// Remove row r, releasing its live entries from their columns.
    void remove_row(int32_t r);

    /// Set column j's bounds to [lo, hi] intersected with the current ones,
    /// collapsing a crossing within `slack` to the point `lb` (rounding, not
    /// an empty box) and flagging `infeasible` beyond it. `slack` is the most
    /// the caller lets the collapsed point violate the bound it widens.
    void tighten_column(int32_t j, double lo, double hi, double slack);
};

// One function per reduction. Each sets `w.changed` when it changes anything
// and `w.infeasible` when it proves infeasibility.
void reduce_propagate(Work& w);
void reduce_fixed_columns(Work& w);
void reduce_rows(Work& w);  ///< Empty, free, singleton and redundant rows.
void reduce_empty_columns(Work& w);

}  // namespace cbls::presolve_detail
