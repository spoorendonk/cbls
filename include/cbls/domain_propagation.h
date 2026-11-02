#pragma once

// Incremental, undoable domain propagation over a model's linear rows (#329).
//
// Implied domains depend on bounds, not on current values: a plain FJ jump
// changes no bound and gains nothing from this. The place where variables are
// held fixed for a while is the Novelty compound-move stack, which is what
// this is built for: GFJConfig::compound_propagation (B2, #330/#339).

#include "cbls/bound_propagation.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace cbls {

class Model;

namespace activity_detail {
struct Activity;  // src/activity_bounds.h
}  // namespace activity_detail

/// A model's linear rows in residual form, `lo <= sum_k a_k x_k <= hi`, built
/// once from `affine_row_terms` at the model's current node values.
///
/// Kept: every row `affine_row_terms` accepts (Leq, Geq, Lt, Gt and a bare
/// body read as `residual <= 0`; Eq as `residual = 0`) over scalar variables.
/// Skipped: the objective row (soft: propagating the incumbent cutoff would
/// make it hard), domain rows (#248/#258: `affine_row_terms` refuses them),
/// Neq, and every row that is not affine.
///
/// Memory: per nonzero one int32 column and one double coefficient in the row
/// CSR, plus one (row, position) entry in the column index -- about 24 B per
/// nonzero -- and O(1) per row and per variable.
class LinearRowStore {
public:
    explicit LinearRowStore(const Model& model);

    [[nodiscard]] int32_t num_rows() const noexcept { return static_cast<int32_t>(row_lo_.size()); }
    [[nodiscard]] int32_t num_vars() const noexcept {
        return static_cast<int32_t>(col_start_.size()) - 1;
    }
    /// The model's constraint index of stored row `r`.
    [[nodiscard]] int32_t constraint(int32_t r) const {
        return constraint_[static_cast<size_t>(r)];
    }
    [[nodiscard]] double row_lo(int32_t r) const { return row_lo_[static_cast<size_t>(r)]; }
    [[nodiscard]] double row_hi(int32_t r) const { return row_hi_[static_cast<size_t>(r)]; }
    /// Row `r`'s terms are `[row_start(r), row_start(r + 1))` of `cols()`/`coefs()`.
    [[nodiscard]] size_t row_start(int32_t r) const { return row_start_[static_cast<size_t>(r)]; }
    [[nodiscard]] const std::vector<int32_t>& cols() const noexcept { return cols_; }
    [[nodiscard]] const std::vector<double>& coefs() const noexcept { return coefs_; }
    /// Variable `v`'s entries are `[col_start(v), col_start(v + 1))` of `col_entries()`,
    /// each a position in `cols()`/`coefs()`, with `col_rows()` naming its row.
    [[nodiscard]] size_t col_start(int32_t v) const { return col_start_[static_cast<size_t>(v)]; }
    [[nodiscard]] const std::vector<size_t>& col_entries() const noexcept { return col_entries_; }
    [[nodiscard]] const std::vector<int32_t>& col_rows() const noexcept { return col_rows_; }

private:
    std::vector<int32_t> constraint_;
    std::vector<double> row_lo_;
    std::vector<double> row_hi_;
    std::vector<size_t> row_start_;
    std::vector<int32_t> cols_;
    std::vector<double> coefs_;
    std::vector<size_t> col_start_;
    std::vector<size_t> col_entries_;
    std::vector<int32_t> col_rows_;
};

struct DomainPropagationOptions {
    /// Infinity, the improvement thresholds and the safety margins, read as
    /// `propagate_bounds` reads them; `max_passes` is unused.
    BoundPropagationOptions bounds;
    /// Off (the default): one call scans only the rows of the variable it
    /// changed (depth 1); bounds it derives update activities but scan no
    /// further. On: derived bounds queue their rows too, until the queue is
    /// empty or `max_cascade_terms` terms have been scanned in the call.
    bool cascade = false;
    int64_t max_cascade_terms = 100000;
    /// Skip a row whose slack is at least its max variation (the filter).
    /// Off only to check, in tests, that the filter skips nothing a scan uses.
    bool filter = true;
};

struct DomainPropagationStats {
    int64_t calls = 0;          ///< fix/tighten calls.
    int64_t rows_scanned = 0;   ///< Rows whose terms were read.
    int64_t rows_filtered = 0;  ///< Rows the slack filter skipped in O(1).
    int64_t terms_scanned = 0;
    int64_t bounds_derived = 0;  ///< Bounds tightened by propagation.
    int64_t conflicts = 0;
};

/// Bounds of a model's scalar variables under `fix`/`tighten`, propagated over
/// a `LinearRowStore`'s rows, with checkpoints to undo to.
///
/// Each row keeps its min and max activity as a finite sum plus an infinite
/// count, updated incrementally, which decides only the slack filter. A row
/// that passes it is summed afresh from the current bounds, so its derived
/// bounds are `propagate_bounds`'s: an incremental sum keeps the rounding of a
/// contribution it removed. A derived bound is relaxed outward
/// by `max(safety margin, #211's activity-scaled roundoff)` and then rounded
/// inward on an integral (Bool/Int) variable.
///
/// The trail records old values -- bounds and whole activities -- never
/// differences, so `backtrack` restores state bit for bit, with no drift.
///
/// Costs, for d = rows of the changed variable:
/// - `fix`/`tighten`: O(d) activity updates, plus O(nnz) for each queued row
///   whose slack is below its max variation plus the drift margin (O(1) for
///   each other one), plus
///   O(rows of v) for every bound v so derived.
/// - `backtrack`: O(trail entries undone). `checkpoint`: O(1).
/// - Memory: O(rows + vars), plus the trail: one entry per bound change and
///   two activities per row update.
class DomainPropagator {
    struct BoundEntry {
        int32_t var;
        double lo;
        double hi;
    };
    struct Level {
        size_t bounds;
        size_t rows;
    };

public:
    /// Starts from the model's variable bounds. The store must be built from
    /// the same model, and outlive this.
    DomainPropagator(const LinearRowStore& store, const Model& model,
                     DomainPropagationOptions opts = {});
    ~DomainPropagator();  // out of line: Activity is incomplete here
    DomainPropagator(const DomainPropagator&) = delete;
    DomainPropagator& operator=(const DomainPropagator&) = delete;
    DomainPropagator(DomainPropagator&&) = delete;
    DomainPropagator& operator=(DomainPropagator&&) = delete;

    /// Pushes a checkpoint and returns its level, for `backtrack`.
    int32_t checkpoint();
    /// Restores the state as of `checkpoint()` returning `level`, and drops
    /// that checkpoint and every later one.
    void backtrack(int32_t level);
    /// Checkpoints held: 0 at the root.
    [[nodiscard]] int32_t depth() const noexcept { return static_cast<int32_t>(levels_.size()); }

    /// Fixes `var` to `value`: `tighten(var, value, value)`.
    bool fix(int32_t var, double value);
    /// Intersects `var`'s bounds with `[lo, hi]` and propagates. False on a
    /// conflict (an empty domain beyond the safety margin); the state is then
    /// partial, and the caller backtracks.
    bool tighten(int32_t var, double lo, double hi);

    [[nodiscard]] double lo(int32_t var) const { return lo_[static_cast<size_t>(var)]; }
    [[nodiscard]] double hi(int32_t var) const { return hi_[static_cast<size_t>(var)]; }
    /// The distinct variables whose bounds changed since `checkpoint()`
    /// returned `level`, in first-change order, into `out`.
    void changed_since(int32_t level, std::vector<int32_t>& out) const;

    /// Row `r`'s incremental min (max) activity: its finite sum and its
    /// infinite count. The sum may carry rounding drift; the count is exact.
    [[nodiscard]] double min_activity_sum(int32_t r) const;
    [[nodiscard]] int min_activity_infinite(int32_t r) const;
    [[nodiscard]] double max_activity_sum(int32_t r) const;
    [[nodiscard]] int max_activity_infinite(int32_t r) const;

    [[nodiscard]] const DomainPropagationStats& stats() const noexcept { return stats_; }

private:
    struct RowEntry;  // a trailed row: its index and both old activities

    void set_bounds(int32_t var, double lo, double hi);
    bool apply(int32_t var, double value, bool upper, double roundoff);
    void rebuild_activities(int32_t r, activity_detail::Activity& min_act,
                            activity_detail::Activity& max_act) const;
    [[nodiscard]] bool filtered(int32_t r) const;
    bool scan_row(int32_t r);
    void enqueue_rows_of(int32_t var);
    [[nodiscard]] bool conflict();

    const LinearRowStore& store_;
    DomainPropagationOptions opts_;
    std::vector<uint8_t> integral_;
    std::vector<double> lo_;
    std::vector<double> hi_;
    std::vector<activity_detail::Activity> min_act_;
    std::vector<activity_detail::Activity> max_act_;
    /// `max_k |a_k| * (hi_k - lo_k)` at the starting bounds: as bounds only
    /// shrink, no term can move row r's activity further, so a slack at least
    /// this large derives nothing (the filter).
    std::vector<double> max_variation_;
    /// `sum_k |a_k| * max(|lo_k|, |hi_k|)` at the starting bounds: no finite
    /// contribution exceeds its term, so it scales the rounding drift of the
    /// incremental sums, which the filter allows for.
    std::vector<double> drift_scale_;
    /// Live (trailed) activity updates per row: the drift's operation count.
    std::vector<int32_t> updates_;

    std::vector<BoundEntry> bound_trail_;
    std::vector<RowEntry> row_trail_;
    std::vector<Level> levels_;

    // Per call: the rows to scan, deduplicated by a stamp.
    std::vector<int32_t> queue_;
    std::vector<uint32_t> queued_stamp_;
    uint32_t stamp_ = 0;
    bool enqueue_derived_ = false;
    int64_t call_terms_ = 0;

    DomainPropagationStats stats_;
};

}  // namespace cbls
