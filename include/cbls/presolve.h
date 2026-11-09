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
//   - coefficient tightening: an integral column's coefficient in a one-sided
//     row moves toward 0 with the side, keeping the row's integer solutions
//     (Achterberg et al. 2020, eq. (3.6); src/presolve_coef.cpp);
//   - doubleton-equality aggregation: a row `a x + b y = c` with two live
//     entries eliminates y = (c - a x)/b (src/presolve_aggregate.cpp, #328).
//     y's bounds move onto x, y is substituted in its other rows and the
//     objective, and the row and y go; postsolve recomputes y from x. Only
//     |a/b| <= `max_aggregation_ratio`; an integral y only when x is
//     integral and a/b and c/b are integers. Limits: a frozen column is
//     never eliminated or kept in y's place, and no dual reasoning is used;
//     x's new coefficients carry their rounding (exact for integer ratios);
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
#include <limits>
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
    /// Coefficient tightening on one-sided rows (#327).
    bool coefficient_tightening = true;
    /// Doubleton-equality aggregation (#328): `a x + b y = c` eliminates y.
    bool doubleton_aggregation = true;
    /// Largest |a/b| for which y is eliminated: x's coefficients in y's other
    /// rows change by `-a_ry a/b`, so a large ratio inflates them.
    double max_aggregation_ratio = 1.0;
    /// Most entries one aggregation may add (fill). No limit by default: an
    /// aggregation removes y's k other entries and adds at most k, and drops
    /// the row, so the total never grows; the limit only caps how much x's
    /// column grows. Unmeasured: #331 measured presolve only as a whole
    /// and kept it off.
    int max_fill = std::numeric_limits<int>::max();
    /// Hard stop on the number of rounds; each round is O(nnz).
    int max_rounds = 20;
    /// Passed to `propagate_bounds` each round.
    BoundPropagationOptions propagation;
    /// Absolute slack, on top of a bound on the floating-point error of the
    /// value checked, for: an empty row `lo <= 0 <= hi`, crossed row or column
    /// bounds (collapsed to a point within it), and the inward rounding of an
    /// integral column's bounds. A point presolve admits therefore violates
    /// the original problem by at most about this much plus roundoff, far
    /// inside `verify_model`'s 1e-6. Anything more is reported `Infeasible`.
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
    int cols_fixed = 0;           ///< Fixed columns removed.
    int cols_empty = 0;           ///< Empty columns fixed at their best bound and removed.
    int rows_empty = 0;           ///< Empty rows removed.
    int rows_free = 0;            ///< Rows with no finite side removed.
    int rows_singleton = 0;       ///< Singleton rows turned into column bounds.
    int rows_redundant = 0;       ///< Rows removed because both sides were implied.
    int row_sides_dropped = 0;    ///< Implied row sides dropped (rows kept or not).
    int duplicates_merged = 0;    ///< Repeated (row, column) entries merged on load.
    int bounds_tightened = 0;     ///< Kept columns whose lb or ub moved.
    int coefs_tightened = 0;      ///< Coefficient-tightening steps (a column may count again).
    int rows_coef_tightened = 0;  ///< Row passes in which a coefficient was tightened.
    int cols_aggregated = 0;      ///< Columns eliminated through a doubleton equality.
    int aggregation_fill = 0;     ///< Entries added by those eliminations.
    double seconds = 0.0;
};

/// Maps a solution of the reduced problem back to the original columns.
///
/// A LIFO stack of steps, each restoring the columns one reduction removed;
/// `apply` places the reduced values, then undoes the steps last to first, so a
/// later step may read a column an earlier one restores.
class Postsolve {
public:
    /// One undo step, tagged by kind. Steps of every kind share one stack, so
    /// `apply` undoes them in exactly the reverse of the order presolve took.
    struct Step {
        enum class Kind : uint8_t {
            Fixed,       ///< x[col] = value.
            Aggregated,  ///< x[col] = (value - a * x[other]) / b, rounded if `integral`.
        };
        Kind kind = Kind::Fixed;
        uint8_t integral = 0;
        int32_t col = -1;
        int32_t other = -1;
        double value = 0.0;
        double a = 0.0;
        double b = 0.0;
    };

    /// `n_original` columns, of which `reduced_to_col[i]` is reduced column i.
    /// With no steps pushed, `apply` is then the identity on the survivors.
    void set_mapping(std::size_t n_original, std::vector<int32_t> reduced_to_col);

    void push_fixed(int32_t col, double value) {
        Step s;
        s.col = col;
        s.value = value;
        steps_.push_back(s);
    }

    /// y = (c - a x) / b, rounded when y is integral and never clamped:
    /// clamping would break the equality it came from.
    void push_aggregated(int32_t y, int32_t x, double a, double b, double c, bool y_integral) {
        Step s;
        s.kind = Step::Kind::Aggregated;
        s.integral = y_integral ? 1 : 0;
        s.col = y;
        s.other = x;
        s.value = c;
        s.a = a;
        s.b = b;
        steps_.push_back(s);
    }

    /// One value per original column. Throws `std::invalid_argument` unless
    /// `reduced_x` has one entry per reduced column.
    [[nodiscard]] std::vector<double> apply(const std::vector<double>& reduced_x) const;

    [[nodiscard]] std::size_t num_steps() const { return steps_.size(); }
    [[nodiscard]] std::size_t n_original() const { return n_original_; }
    [[nodiscard]] const std::vector<int32_t>& reduced_to_col() const { return reduced_to_col_; }

private:
    std::size_t n_original_ = 0;
    std::vector<int32_t> reduced_to_col_;
    std::vector<Step> steps_;
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
/// Throws `std::invalid_argument` on a malformed problem: sizes that disagree,
/// a column index out of range, a NaN bound, a non-finite coefficient or
/// objective entry, or a lower bound at +infinity / upper bound at -infinity.
/// On `Infeasible` the result holds the original problem (duplicates merged)
/// with an identity postsolve, and its stats count nothing but `rounds` and
/// `duplicates_merged`: as in the adapters today, an infeasible instance is
/// reported by the search, not by the reader.
///
/// Cost: O(nnz) per round, `max_rounds` at most, except doubleton aggregation:
/// O(length of y's other rows) per eliminated y, so quadratic in a dense row
/// that holds many eliminated columns. Peak working memory is about
/// 20 bytes per nonzero (row-major entries 12 B, column-major index 8 B), plus
/// the reduced copy (12 B per kept nonzero), on top of the caller's input,
/// plus 40 B per postsolve step (one per removed column).
/// Disabled, or once proven infeasible, the column index is not built.
PresolveResult presolve(const LinearProblem& problem, const PresolveOptions& opts = {});

}  // namespace cbls
