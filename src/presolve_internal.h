#pragma once

// Working storage shared by the presolve reductions (src/presolve*.cpp).
// Not installed: the public interface is include/cbls/presolve.h.
//
// Row r's entries are `cols`/`coefs` in [row_begin[r], row_end[r]); deleted
// entries keep their column index and have their coefficient set to exactly 0
// (`propagate_bounds` skips a zero term, so the live rows can be handed to it
// as views without compaction). Each column lists its entries as (row, offset
// within the row) in `col_rows`/`col_offs` over [col_begin[j], col_end[j]).
// Removed rows and columns are marked, and everything is compacted once, when
// the result is built.
//
// A row can grow (`add_entry`). A slot's column never changes once the slot
// exists, so each slot is listed exactly once, by the column it holds, and a
// row moved elsewhere keeps its offsets (and so its column-index entries).

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

    // Rows (a deleted entry has coefficient 0). Row r owns [row_begin[r],
    // row_cap[r]) of `cols`/`coefs`, of which [row_begin[r], row_end[r]) is used.
    std::vector<std::size_t> row_begin;
    std::vector<std::size_t> row_end;
    std::vector<std::size_t> row_cap;
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

    // Column-major index: column j's entries are (col_rows[q], col_offs[q]) for
    // q in [col_begin[j], col_end[j]); column j owns up to col_cap[j]. An entry
    // of a removed row, or one whose coefficient is 0, is stale: skip it.
    std::vector<std::size_t> col_begin;
    std::vector<std::size_t> col_end;
    std::vector<std::size_t> col_cap;
    std::vector<int32_t> col_rows;
    std::vector<int32_t> col_offs;

    Postsolve postsolve;
    PresolveStats stats;
    bool changed = false;
    bool infeasible = false;

    explicit Work(const PresolveOptions& o) : opts(o) {}

    [[nodiscard]] std::size_t n_cols() const { return lb.size(); }
    [[nodiscard]] std::size_t n_rows() const { return row_lo.size(); }

    /// Position in `cols`/`coefs` of column-index entry q.
    [[nodiscard]] std::size_t entry_pos(std::size_t q) const {
        return row_begin[static_cast<std::size_t>(col_rows[q])] +
               static_cast<std::size_t>(col_offs[q]);
    }

    /// Add `a * x_j` to live row r, where j has no live entry (a != 0, finite).
    /// Reuses j's own cleared slot in r if it has one; otherwise appends, moving
    /// the row to the end of `cols`/`coefs` when it is full. Returns false, and
    /// changes nothing, if the row would outgrow an int32 offset.
    ///
    /// Cost: O(row length) to look for j's cleared slot, plus, when the row or
    /// column is full, a move of O(its length) into twice the room, so moves
    /// cost amortised O(1) per added entry. The vacated room is not reused
    /// before the result is built: at most about twice the entries ever added.
    [[nodiscard]] bool add_entry(int32_t r, int32_t j, double a);

    /// Bound on the floating-point error in row r's current `row_lo`/`row_hi`:
    /// each fold `bound -= a * v` adds at most one rounding of the product and
    /// one of the difference, each within `DBL_EPSILON` of a partial sum no
    /// larger than `row_scale`, so `(row_folds + 1) * DBL_EPSILON * row_scale`.
    [[nodiscard]] double row_roundoff(std::size_t r) const;

    /// Whether column j can be folded at `value`: every product `a * value`
    /// and `obj * value` is finite and no finite row bound moves to or past
    /// `infinity`, where it would read as "no bound". O(column length).
    [[nodiscard]] bool can_fix(int32_t j, double value) const;

    /// Remove column j at `value`: fold it into its rows' bounds and the
    /// objective offset, and push the postsolve step. j must not be frozen,
    /// and `can_fix(j, value)` must hold.
    void fix_column(int32_t j, double value);

    /// Remove row r, releasing its live entries from their columns.
    void remove_row(int32_t r);

    /// Set column j's bounds to [lo, hi] intersected with the current ones,
    /// collapsing a crossing within `slack` to the point `lb` (rounding, not
    /// an empty box) and flagging `infeasible` beyond it. `slack` is the most
    /// the caller lets the collapsed point violate the bound it widens.
    void tighten_column(int32_t j, double lo, double hi, double slack);
};

/// Validate `p` and load it into `w`: bounds normalised to ±inf, rows merged.
/// The column index is not built here: only the reductions need it.
void load(const LinearProblem& p, Work& w);
/// Build the column-major index and the live counts of a loaded `w`.
void build_column_index(Work& w);

// One function per reduction. Each sets `w.changed` when it changes anything
// and `w.infeasible` when it proves infeasibility.
void reduce_propagate(Work& w);
void reduce_fixed_columns(Work& w);
void reduce_rows(Work& w);  ///< Empty, free, singleton and redundant rows.
void reduce_empty_columns(Work& w);
/// Integral coefficients of one-sided rows (src/presolve_coef.cpp, #327).
void reduce_coef_tightening(Work& w);

}  // namespace cbls::presolve_detail
