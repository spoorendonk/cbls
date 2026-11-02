#pragma once

// Presolve over a linear problem, with postsolve back to the original columns.
//
// The model adapters (MPS, the linear part of AMPL .nl) hand a `LinearProblem`
// in, build the engine model from `PresolveResult::reduced`, and map a solution
// of the reduced problem back with `Postsolve::apply` before writing it (#322).
//
// Reductions, run round by round to a fixed point or `max_rounds`:
//   - activity bound tightening (`propagate_bounds`, see bound_propagation.h);
//   - fixed columns (lb == ub): folded into the row bounds and the objective
//     offset, then removed;
//   - empty rows (checked, then removed), free rows (removed);
//   - singleton rows: turned into bounds on their column, then removed;
//   - redundant row sides: a side the activity range over the column bounds
//     already implies is dropped, and a row left with no side is removed;
//   - empty columns (in no row): fixed at their best bound for the objective.
//     This is the only optimality-based (dual) reduction here: it keeps every
//     optimum's objective value, not every feasible point.
// Every other reduction removes no feasible point, up to the floating-point
// margins documented at each site in src/presolve.cpp.
//
// A `frozen` column is never removed: its bounds may tighten, but it stays in
// the reduced problem even when fixed. The NL adapter freezes every column that
// appears outside the purely linear rows, so its expressions need no remapping.

#include "bound_propagation.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cbls {

/// A linear problem in CSR form: `min obj·x + obj_offset` subject to
/// `row_lo[i] <= sum_k coefs[k] * x[cols[k]] <= row_hi[i]` for
/// `k in [row_start[i], row_start[i+1])`, and `col_lb <= x <= col_ub`.
/// Magnitudes at or beyond `kBoundInfinity` mean "no bound" (as do ±inf).
/// A column may repeat within a row; presolve merges the repeats.
struct LinearProblem {
    std::vector<double> col_lb;
    std::vector<double> col_ub;
    std::vector<uint8_t> integral;  ///< Non-zero: the column takes integer values only.
    std::vector<uint8_t> frozen;    ///< Non-zero: never removed or substituted.
    std::vector<double> obj;        ///< Minimisation coefficients, one per column.
    double obj_offset = 0.0;

    std::vector<std::size_t> row_start{0};  ///< n_rows + 1 entries.
    std::vector<int32_t> cols;
    std::vector<double> coefs;
    std::vector<double> row_lo;
    std::vector<double> row_hi;

    [[nodiscard]] std::size_t n_cols() const { return col_lb.size(); }
    [[nodiscard]] std::size_t n_rows() const { return row_lo.size(); }
};

struct PresolveOptions {
    /// Off by default: `presolve` then returns the problem with duplicate
    /// entries merged and zeros dropped, and an identity postsolve.
    bool enabled = false;
    bool fixed_columns = true;
    bool singleton_rows = true;
    bool redundant_rows = true;
    bool empty_columns = true;
    bool propagate_bounds = true;
    /// Hard stop on the number of rounds; each round is O(nnz).
    int max_rounds = 20;
    /// Passed to `propagate_bounds` each round.
    BoundPropagationOptions propagation;
    /// An empty row `lo <= 0 <= hi`, or a row whose bounds cross, is accepted
    /// within `feasibility_tolerance * max(1, scale)`, where `scale` is the
    /// magnitude of the row's bounds and of everything folded into them.
    double feasibility_tolerance = 1e-9;
};

enum class PresolveStatus : uint8_t {
    Unchanged,   ///< No reduction applied; `reduced` equals the (merged) input.
    Reduced,     ///< At least one reduction applied.
    Infeasible,  ///< Presolve proved the problem infeasible; see `presolve`.
};

struct PresolveStats {
    PresolveStatus status = PresolveStatus::Unchanged;
    int rounds = 0;
    int cols_fixed = 0;         ///< Fixed columns removed.
    int cols_empty = 0;         ///< Empty columns fixed at their best bound and removed.
    int rows_empty = 0;         ///< Empty rows removed.
    int rows_free = 0;          ///< Rows with no finite side removed.
    int rows_singleton = 0;     ///< Singleton rows turned into column bounds.
    int rows_redundant = 0;     ///< Rows removed because both sides were implied.
    int row_sides_dropped = 0;  ///< Implied row sides dropped (rows kept or not).
    int duplicates_merged = 0;  ///< Repeated (row, column) entries merged on load.
    int bounds_tightened = 0;   ///< Kept columns whose lb or ub moved.
    double seconds = 0.0;
};

/// Maps a solution of the reduced problem back to the original columns.
///
/// A LIFO stack of steps, each restoring the columns one reduction removed;
/// `apply` places the reduced values, then undoes the steps last to first, so a
/// later step may read a column an earlier one restores.
class Postsolve {
public:
    struct FixedColumn {
        int32_t col;
        double value;
    };

    /// `n_original` columns, of which `reduced_to_col[i]` is reduced column i.
    /// With no steps pushed, `apply` is then the identity on the survivors.
    void set_mapping(std::size_t n_original, std::vector<int32_t> reduced_to_col);

    void push_fixed(int32_t col, double value) { fixed_.push_back({col, value}); }

    /// One value per original column. Throws `std::invalid_argument` unless
    /// `reduced_x` has one entry per reduced column.
    [[nodiscard]] std::vector<double> apply(const std::vector<double>& reduced_x) const;

    [[nodiscard]] std::size_t num_steps() const { return fixed_.size(); }
    [[nodiscard]] std::size_t n_original() const { return n_original_; }
    [[nodiscard]] const std::vector<int32_t>& reduced_to_col() const { return reduced_to_col_; }

private:
    std::size_t n_original_ = 0;
    std::vector<int32_t> reduced_to_col_;
    std::vector<FixedColumn> fixed_;
};

struct PresolveResult {
    LinearProblem reduced;
    Postsolve postsolve;
    std::vector<int32_t> col_to_reduced;  ///< Per original column; -1 if removed.
    std::vector<int32_t> row_to_reduced;  ///< Per original row; -1 if removed.
    std::vector<int32_t> reduced_to_col;  ///< Per reduced column: its original index.
    PresolveStats stats;
};

/// Presolve `problem`.
///
/// Throws `std::invalid_argument` on a malformed problem (sizes that disagree,
/// a column index out of range, a NaN bound). On `Infeasible` the result holds
/// the original problem (duplicates merged) with an identity postsolve: as in
/// the adapters today, an infeasible instance is reported by the search, not
/// by the reader.
///
/// Cost: O(nnz) per round, `max_rounds` at most. Peak working memory is about
/// 20 bytes per nonzero (row-major entries 12 B, column-major index 8 B), plus
/// the reduced copy (12 B per kept nonzero), on top of the caller's input.
PresolveResult presolve(const LinearProblem& problem, const PresolveOptions& opts = {});

}  // namespace cbls
