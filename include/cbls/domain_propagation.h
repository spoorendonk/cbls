#pragma once

// Incremental, undoable domain propagation over a model's linear rows (#329,
// restored for fix-propagate-repair by #420).
//
// Implied domains depend on bounds, not on current values: a plain FJ jump
// changes no bound and gains nothing from this. It pays where variables are
// fixed one after another under propagation: fix-propagate-repair (#421).

#include "cbls/bound_propagation.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cbls {

class Model;

namespace activity_detail {
struct Activity;  // src/activity_bounds.h
}  // namespace activity_detail

/// A model's linear rows in residual form, `lo <= sum_k a_k x_k <= hi`, built
/// once from `affine_row_terms` at the model's current node values: the model
/// must be closed and fully evaluated (`full_evaluate`), or slopes and
/// constants read zero.
///
/// Kept: every row `affine_row_terms` accepts (Leq, Geq, Lt, Gt and a bare
/// body read as `residual <= 0`; Eq as `residual = 0`) over scalar variables.
/// Skipped: the objective row (soft: propagating the incumbent cutoff would
/// make it hard), domain rows (#248/#258: `affine_row_terms` refuses them),
/// Neq, and every row that is not affine. An affine row whose terms all
/// cancel is kept apart, in `constant_rows()`, for `propagate_all` to check.
///
/// Memory: per nonzero one int32 column and one double coefficient in the row
/// CSR, plus one (row, position) entry in the column index -- about 24 B per
/// nonzero -- and O(1) per row and per variable.
class LinearRowStore {
public:
    /// An affine row with no terms: `residual <= 0` (`== 0` when `eq`).
    struct ConstantRow {
        int32_t constraint;
        double residual;
        bool eq;
    };

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
    [[nodiscard]] int64_t nnz() const noexcept { return static_cast<int64_t>(cols_.size()); }
    [[nodiscard]] const std::vector<ConstantRow>& constant_rows() const noexcept {
        return constant_rows_;
    }

private:
    std::vector<ConstantRow> constant_rows_;
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
    /// On (the default): derived bounds queue their rows too, until the queue
    /// is empty (a fixpoint) or the call has scanned `max_cascade_terms` terms.
    /// Off: one call scans only the rows of the variable it changed (depth 1);
    /// bounds it derives update activities but scan no further.
    bool cascade = true;
    /// The per-call work valve: once a call has scanned this many terms, no
    /// further cascaded row is scanned (the rows the call queued itself always
    /// are). Negative means 100 times the store's nonzeros, mip-heuristics'
    /// FPR valve (#420).
    int64_t max_cascade_terms = -1;
    /// Skip a row whose slack is at least its max variation (the filter).
    /// Off only to check, in tests, that the filter skips nothing a scan uses.
    bool filter = true;
};

struct DomainPropagationStats {
    int64_t calls = 0;          ///< fix/tighten calls, and fix_all fixings.
    int64_t rows_scanned = 0;   ///< Rows whose terms were read.
    int64_t rows_filtered = 0;  ///< Rows the slack filter skipped in O(1).
    int64_t terms_scanned = 0;
    int64_t bounds_derived = 0;  ///< Bounds tightened by propagation.
    int64_t conflicts = 0;
    /// Calls the work valve stopped short of a fixpoint.
    int64_t valve_stops = 0;
    int64_t propagate_all_calls = 0;
    /// Bounds `propagate_all` derived: at depth 0, the root tightenings.
    int64_t propagate_all_bounds_derived = 0;
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
/// Changes made with no checkpoint held cannot be undone and are not trailed.
///
/// Costs, for d = rows of the changed variable:
/// - `propagate_all`: as `tighten` with every row queued, so at least O(rows).
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
    /// A temporary store would dangle.
    DomainPropagator(LinearRowStore&& store, const Model& model,
                     DomainPropagationOptions opts = {}) = delete;
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
    /// Intersects `var`'s bounds with `[lo, hi]` (rounded inward on an
    /// integral variable) and propagates. False on a conflict (an empty domain
    /// beyond the safety margin). The propagator never empties a domain: a
    /// crossing bound is refused, not written, and every change under a checkpoint is
    /// trailed. So after a conflict the state is partial but consistent, and
    /// the caller may backtrack or keep fixing and propagating from it (#427).
    /// Throws `std::invalid_argument` on a NaN bound.
    bool tighten(int32_t var, double lo, double hi);
    /// Fixes every `(var, value)` without propagating, then propagates the rows
    /// of every changed variable together: the reference FPR's trivial round
    /// (#427). A fixing that crosses its domain is a conflict and is skipped;
    /// the rest are kept and propagated. False on any conflict; the state is
    /// then as after `tighten`'s. Throws before any change on a bad request.
    /// Each fixing counts as one call in `stats().calls`, and a crossing one
    /// as one conflict, besides any the propagation finds.
    bool fix_all(const std::vector<std::pair<int32_t, double>>& fixings);
    /// Scans every row, cascading as `tighten` does: with cascade on, to a
    /// fixpoint under the work valve. The root propagation (#420). False on a
    /// conflict, a violated constant row included; the state is then as after
    /// `tighten`'s.
    bool propagate_all();

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
    [[nodiscard]] const DomainPropagationOptions& options() const noexcept { return opts_; }

private:
    struct RowEntry;  // a trailed row: its index and both old activities
    enum class Narrowed : uint8_t { kConflict, kUnchanged, kChanged };

    void check_request(int32_t var, double lo, double hi, const char* caller) const;
    Narrowed narrow(int32_t var, double lo, double hi, const char* caller);
    void set_bounds(int32_t var, double lo, double hi);
    bool apply(int32_t var, double value, bool upper, double roundoff);
    void rebuild_activities(int32_t r, activity_detail::Activity& min_act,
                            activity_detail::Activity& max_act) const;
    [[nodiscard]] bool filtered(int32_t r) const;
    bool scan_row(int32_t r);
    void enqueue_rows_of(int32_t var);
    void begin_call();
    bool run_queue();
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
    int64_t cascade_limit_ = 0;  // max_cascade_terms, resolved

    DomainPropagationStats stats_;
};

}  // namespace cbls
