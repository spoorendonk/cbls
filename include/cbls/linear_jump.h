#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cbls {

class Model;
enum class NodeOp : uint8_t;

/// Whether a row whose constraint node has this op is scored as a COMPARISON
/// (`Leq`/`Geq`/`Lt`/`Gt`/`Eq`: residual of its two children) rather than as a
/// bare body (`add_constraint(expr)`, read as `expr <= 0`). The scorer's build
/// and its owner's classification must split rows the same way.
[[nodiscard]] bool is_comparison_op(NodeOp op);

/// Closed-form scoring of a scalar jump over LINEAR comparison rows.
///
/// `Model::weighted_violation_delta` scores a candidate value by setting the
/// variable and running a Probe `delta_evaluate` and a Rollback one. A dirty
/// `Sum` re-sums all of its children, so on a MIP each candidate costs the
/// summed length of every row in the variable's column -- twice -- and an Int
/// can have ~260 candidates. On `gen-ip002`, `neos-860300` and `n2seq36q` that
/// was 65-86% of the solver's CPU (gprof, main `175e617`).
///
/// A row whose constraint node is a comparison (`Leq`/`Geq`/`Lt`/`Gt`/`Eq`)
/// with both children AFFINE in the variables has, for each variable v, a
/// constant r = d(p - q)/dv, where p and q are the two arguments of the row's
/// residual (`p = child0, q = child1` for Leq/Lt/Eq, the swap for Geq/Gt). So
/// moving v by D moves the residual from cmp(p, q) to cmp(p + r D, q) -- no DAG
/// walk, O(1) per row, O(|G_v|) per candidate.
///
/// A BARE body, `add_constraint(expr)` with an affine non-comparison node, means
/// `expr <= 0` (#190): the engine's violation is `clamped(value)`. It is built
/// as the residual `p - q` with p = the row's own node and q = a literal 0, read
/// through `comparison_residual(p, 0, p_literal, true)`. That is the node value
/// itself for EVERY input, non-finite included: the sentinel branch needs
/// `p == q` with q infinite, and 0 never is, and `v - 0.0` is `v` (NaN stays
/// NaN, +/-inf stays itself, -0.0 stays -0.0). The slope is the node's own
/// partial, the same sweep `compute_partial(row, v)` runs, so its Newton
/// partial is exact like a Leq row's.
///
/// **Exact in the arithmetic, not bit-identical.** `p + r D` is not the row the
/// DAG would re-sum, so a score can differ from `weighted_violation_delta` in its
/// last bits (the equivalence tests pin a tight relative tolerance). Nothing
/// accumulates here: a committed jump still goes through the DAG walk
/// (`commit_scalar_move`), and every prepare reads the node values fresh. Those
/// values are the DAG's own, which since #188 means an incremental Sum's: moved
/// by its terms' changes and, on fractional data, up to its drift bound off the
/// real sum until the batch's end re-sums it. The scorer reads them as they
/// are, which is what the commit it is scoring would move.
///
/// What is NOT changed: the candidate set, the selection rule and its first-seen
/// tie-breaking (`compute_var_jump`), and the per-constraint differencing of
/// issue #100 -- each row's `w * (clamped(new) - clamped(old))` is accumulated on
/// its own, so a row clamped to kInfPenalty cancels exactly instead of swallowing
/// the O(1) rows.
///
/// Rows are classified by the OWNER (FeasibilityJump, which already derives
/// per-node affineness) through `set_row_eligible`, and each row's slopes are
/// built LAZILY, on the first prepare that reads the row: a short-lived FJ (the
/// LNS repair builds one per call) sweeps only the rows it touches -- though its
/// slope table may be zero-filled whole on allocation; see Storage.
///
/// Storage: 4 bytes per row (a slot: ineligible, pending, or the index of its
/// built record), a 12-byte record per row BUILT, and one slope per entry of the
/// model's G_v incidence array -- 8 bytes per (variable, row) incidence, laid out
/// parallel to `Model::constraints_of_var`, so `prepare` reads the slope of the
/// k-th row of G_v at a fixed position instead of searching the row for the
/// variable (#176: that search was up to 49% of the MIPfeas runner's CPU). The
/// array is allocated on the first build by `calloc`. A block the allocator maps
/// fresh (glibc: above its dynamic mmap threshold, which tops out at 32 MiB)
/// comes back as untouched zero pages, so a short-lived FJ that builds a few rows
/// makes resident only the pages those rows' columns land on; a smaller block
/// reused from the heap is zero-filled, O(incidences) -- the same order as the
/// FJ constructor's own per-row tables, which the LNS repair already pays per
/// call. Measured, worst case (threshold raised, every call memset): calloc +
/// free of neos-860300's 3.1 MB table costs 0.04 ms warm and a 32 MB one 2.2 ms
/// (~15 GB/s, so ~0.2 ms for 3.1 MB cold); that instance runs 10 repairs in 20s,
/// under 0.01% of the budget. A build writes
/// the row's nonzero slopes into its variables' slots, O(row log |G_v|) once.
///
/// A row is classified once, before it is built, and its slopes never change.
/// The one thing that can move them is the model's incidence layout: the objective
/// row, appended by `add_objective_soft_constraint`, rebuilds every G_v and shifts
/// the offsets. `resize_rows` is how a caller follows a row added to the model,
/// so a strict growth drops every built row back to pending (the rows are
/// structure: rebuilding one reproduces its slopes bit for bit), and `prepare`
/// and `residual_partial_at` refuse to read while the row counts disagree. The
/// slopes are structure, but each portfolio worker's FJ builds its own; the build
/// also sizes `dag_ops.cpp`'s thread_local adjoint scratch, which a pure MIP
/// otherwise never allocated.
///
/// **Selection is unchanged to the bit only on integral data.** Scores there are
/// exact (small integers and dyadic coefficients sum without rounding), so the
/// first-seen minimum picks the same candidate as the probe. On fractional data a
/// score can differ by an ulp, which can flip a near-tie between candidates; the
/// candidate set and the rule are the same, the arithmetic is not.
///
/// One property the probe has and this does not: exact antisymmetry. The probe's
/// score of x -> j and of j -> x (after committing) are exact negations; here the
/// committed row is re-summed by the DAG, so on an exactly balanced Float plateau
/// both directions can score +1 ulp. Integral rows are exact and immune; a
/// threshold would change the selection rule, so none is applied.
class LinearJumpScorer {
public:
    explicit LinearJumpScorer(const Model& model);

    /// Size the per-row table to `n` rows. New rows start ineligible. Grow-only:
    /// a shrink throws `std::logic_error` -- a closed model's rows are never
    /// removed. A strict growth means the model gained a row, which rebuilt its
    /// G_v layout, so every built row returns to pending and is rebuilt on its
    /// next read (its classification is kept).
    void resize_rows(size_t n);
    [[nodiscard]] size_t num_rows() const { return slots_.size(); }

    /// Classify row `ci`. Throws `std::logic_error` if its slopes are already
    /// built: a row's body does not change once the model is closed, so a
    /// built row is never reclassified.
    void set_row_eligible(int32_t ci, bool eligible);
    /// As classified. A row not yet built can still be demoted by its build, on a
    /// non-finite slope.
    [[nodiscard]] bool row_eligible(int32_t ci) const;

    /// Snapshot the rows of G_v for closed-form scoring under `weights`. False
    /// means the caller must score with `weighted_violation_delta`: some row with
    /// a nonzero weight is not eligible, or a value the formula reads is not
    /// finite (where `p + r D` and the DAG's arithmetic part ways). Rows with a
    /// zero weight are skipped -- they contribute exactly 0 either way -- which
    /// is also what lets a variable touching a MASKED nonlinear row (the linear
    /// phase of two-phase GLS) score in closed form.
    ///
    /// The snapshot is valid until the assignment or `weights` changes. Throws
    /// `std::logic_error` if this scorer is not sized to the model's rows (a row
    /// added to the model after `resize_rows`).
    bool prepare(int32_t var_id, const std::vector<double>& weights);

    /// Weighted violation delta of moving the prepared variable to `j`, which
    /// must be finite (and so must `j - x0`); see `prepare`.
    [[nodiscard]] double delta(double j) const;

    /// d(residual of row ci)/d(var_id) for ci = `constraints_of_var(var_id)[k]`
    /// -- the row named by its position in G_v, which is how the Newton step
    /// walks it, so the slope is read in O(1). Bit-identical (up to the sign of a
    /// zero on a satisfied Eq row) to
    /// `compute_partial(model, constraint_ids()[ci], var_id)`, when the row's
    /// cached slope provably is (see the definition). False: call compute_partial
    /// -- also when k is past the end of G_v, or the model's row count is not
    /// this scorer's.
    bool residual_partial_at(int32_t var_id, size_t k, double& out);

    /// Prepares that took the closed form / fell back, and row partials served
    /// from the cache. Diagnostics, and the pins on the wiring in tests.
    [[nodiscard]] int64_t fast_prepares() const { return fast_prepares_; }
    [[nodiscard]] int64_t fallback_prepares() const { return fallback_prepares_; }
    [[nodiscard]] int64_t cached_partials() const { return cached_partials_; }
    /// Nonzero slopes written by the builds since the last layout reset -- a
    /// build counter, not a memory figure: the slope array costs 8 B per G_v
    /// incidence of the model however many rows are built.
    [[nodiscard]] size_t cached_slopes() const { return cached_slopes_; }
    /// Elements in the slope table: 0 before the first build, and after a layout
    /// change until the next one, then the model's G_v incidence count AT THAT
    /// BUILD. Observability for tests -- a table kept across a layout change would
    /// be the old, shorter length and overrun on the rebuild (#176).
    [[nodiscard]] size_t slope_table_size() const { return slope_len_; }

private:
    // slots_[ci]: kIneligible, kPending, or kFirstBuilt + index into built_.
    static constexpr uint32_t kIneligible = 0;
    static constexpr uint32_t kPending = 1;
    static constexpr uint32_t kFirstBuilt = 2;
    // BuiltRow::flags bits.
    static constexpr uint32_t kPIsVar = 1U << 0U;
    static constexpr uint32_t kQIsVar = 1U << 1U;
    static constexpr uint32_t kPLiteral = 1U << 2U;  // a Const node: the sentinel flag
    static constexpr uint32_t kQLiteral = 1U << 3U;
    static constexpr uint32_t kAbs = 1U << 4U;          // Eq: |p - q|
    static constexpr uint32_t kStrict = 1U << 5U;       // Lt/Gt: + the strictness epsilon
    static constexpr uint32_t kNewtonExact = 1U << 6U;  // see residual_partial_at
    static constexpr uint32_t kQZero = 1U << 7U;        // bare body: q is a literal 0, no node
    // A built row: its two residual arguments; its slopes are in slope_at_.
    struct BuiltRow {
        int32_t p_id = -1;  // first argument of the residual
        int32_t q_id = -1;  // second argument; -1 for a bare body's literal 0 (kQZero)
        uint8_t flags = 0;
    };
    // One row of G_v with a nonzero weight and slope, as prepare snapshots it.
    struct Term {
        double weight;
        double old_viol;  // clamped node value, exactly what the DAG holds
        double p;
        double q;
        double slope;
        bool p_literal;
        bool q_literal;
        bool is_abs;
        bool strict;
    };

    // Build row ci (must be pending); returns the row, or nullptr if the build
    // demoted it to ineligible. Appends to built_, so any BuiltRow pointer
    // held across a call is invalidated.
    const BuiltRow* build_row(int32_t ci);
    // The built record of row ci, building it first if pending; nullptr if the
    // row is (or becomes) ineligible.
    const BuiltRow* ready_row(int32_t ci);
    [[nodiscard]] double child_value(int32_t id, bool is_var) const;
    // The row's q argument: its node or variable, or the literal 0 of a bare body.
    [[nodiscard]] double q_value(const BuiltRow& row) const;
    // Row ci's partial from a built row and v's slope, as residual_partial_at reports it.
    [[nodiscard]] double row_partial(const BuiltRow& row, double r) const;
    // Every built row back to pending and the slope array released.
    void reset_built_rows();
    // The slope array, allocated on first use.
    double* slope_table();
    // Row ci's nonzero slopes (in merged_) into its variables' slots.
    void write_slopes(int32_t ci);

    struct FreeDeleter {
        void operator()(double* p) const;
    };

    const Model& model_;
    std::vector<uint32_t> slots_;  // per row
    std::vector<BuiltRow> built_;  // one record per built row
    // Parallel to the model's flat G_v incidence array: slope_at_[
    // constraints_of_var_offset(v) + k] is d(residual of row G_v[k])/dv, once that
    // row is built (0 until then, and for a zero or cancelled slope). calloc'd,
    // hence the deleter; null until the first build.
    std::unique_ptr<double, FreeDeleter> slope_at_;
    size_t slope_len_ = 0;  // elements in slope_at_, 0 while null
    size_t cached_slopes_ = 0;
    std::vector<Term> terms_;
    // build_row's scratch, kept so a build allocates only the row it keeps.
    std::vector<std::pair<int32_t, double>> merged_;
    std::vector<std::pair<int32_t, double>> side0_;
    std::vector<std::pair<int32_t, double>> side1_;
    double x0_ = 0.0;
    int64_t fast_prepares_ = 0;
    int64_t fallback_prepares_ = 0;
    int64_t cached_partials_ = 0;
};

}  // namespace cbls
