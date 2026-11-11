#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cbls {

class Model;

/// The affine residual of constraint `ci` as `sum(terms) + constant`, where the
/// row reads `residual <= 0` (Leq, Geq, Lt, Gt and a bare body) or
/// `residual = 0` (Eq). Geq/Gt are swapped to `q - p` as everywhere else, and a
/// strict row's margin (`ExprNode::const_value`) is in `constant`.
///
/// `terms`: each variable at most once, sorted by id, no zero entry, with
/// exactly the closed-form scorer's slopes (`build_row`'s
/// `compute_partials_sparse` sweep; for Eq, per side then merged). `constant`
/// is the exact residual at all variables zero: a variable-free subtree gives
/// its current value, carried through `Neg`/`Sum` and `Prod`/`Div` by one.
///
/// False, with outputs unspecified, for a non-affine row: `Neq`, a domain row
/// (#258: its residual adds a margin once positive), any other op on a variable
/// path (a nested comparison included), a divisor the DAG reads as zero, or a
/// non-finite slope or constant. Node values must be current.
///
/// Cost: two walks of the row's cone (one allocating a hash map) plus
/// O(k log k) for k terms; for a one-time load, not a per-move path.
bool affine_row_terms(const Model& model, int32_t ci,
                      std::vector<std::pair<int32_t, double>>& terms, double& constant);

/// Closed-form scoring of a scalar jump over LINEAR rows (affine comparisons
/// and affine bare bodies). The probe (`Model::weighted_violation_delta`)
/// re-sums every row of the column twice per candidate, 65-86% of CPU on
/// `gen-ip002`, `neos-860300`, `n2seq36q` (gprof, `175e617`). Here each row has
/// a constant slope r = d(p - q)/dv (p, q the residual's arguments, swapped for
/// Geq/Gt), so a move by D reads cmp(p + r D, q): O(1) per row, O(|G_v|) per
/// candidate.
///
/// A BARE body means `expr <= 0` (#190): p = the row's own node, q = a literal
/// 0, so the residual is the node value for every input, non-finite included.
/// That read is drift-free only because `classify_incremental_sums` (#192)
/// refuses incremental status to a Sum that is a row; relax that and a bare Sum
/// row is scored on a drifted value with no gate.
///
/// **Exact in the arithmetic, not bit-identical**: a score can differ from the
/// probe in its last bits. Nothing accumulates: commits go through the DAG
/// (`commit_scalar_move`) and every prepare reads node values fresh, as the
/// DAG holds them (an incremental Sum's within its drift bound, #188). The
/// candidate set, the first-seen rule (`compute_var_jump`) and #100's
/// per-constraint differencing are unchanged. So selection matches the probe
/// to the bit only on integral data (exact sums); on fractional data an ulp can
/// flip a near-tie. Nor is it exactly antisymmetric like the probe: on a
/// balanced Float plateau both directions can score +1 ulp. No threshold is
/// applied, since one would change the selection rule.
///
/// The OWNER (FeasibilityJump) classifies rows via `set_row_eligible`; slopes
/// are built LAZILY on the first prepare that reads a row, so a short-lived FJ
/// (one per LNS repair) sweeps only the rows it touches.
///
/// Storage: 4 B per row slot, 12 B per BUILT row, and 8 B per G_v incidence
/// for slopes laid out parallel to `Model::constraints_of_var`, so `prepare`
/// reads a slope at a fixed position rather than searching the row (#176: up to
/// 49% of the MIPfeas runner's CPU). The slope array is `calloc`ed on the first
/// build: a block mapped fresh (above glibc's dynamic mmap threshold, itself at most 32 MiB)
/// stays untouched zero pages beyond the rows built; a smaller heap block is
/// zero-filled, O(incidences), the order of the per-row tables each LNS
/// repair's FJ already pays. Measured worst case (every call memset): 3.1 MB
/// (neos-860300) 0.04 ms warm, ~0.2 ms cold; 10 repairs in 20 s is under
/// 0.01%. A build is O(row log |G_v|) once.
///
/// Slopes never change once built; only the incidence layout moves them (the
/// objective row rebuilds every G_v). A strict `resize_rows` growth sends every
/// built row back to pending (rebuilds are bit-identical), and `prepare` and
/// `residual_partial_at` refuse while row counts disagree. Each portfolio
/// worker's FJ builds its own; a build also sizes `dag_ops.cpp`'s thread_local
/// adjoint scratch, which a pure MIP otherwise never allocates.
class LinearJumpScorer {
public:
    explicit LinearJumpScorer(const Model& model);

    /// Size the per-row table to `n` rows; new rows start ineligible.
    /// Grow-only (a shrink throws `std::logic_error`). A strict growth sends
    /// every built row to pending, classification kept (see above).
    void resize_rows(size_t n);
    [[nodiscard]] size_t num_rows() const { return slots_.size(); }

    /// Classify row `ci`. Throws `std::logic_error` if it is already built: a
    /// closed model's row bodies never change.
    void set_row_eligible(int32_t ci, bool eligible);
    /// As classified; an unbuilt row can still be demoted by a non-finite
    /// slope.
    [[nodiscard]] bool row_eligible(int32_t ci) const;

    /// Snapshot G_v for closed-form scoring under `weights`. False: score
    /// with `weighted_violation_delta` (a nonzero-weight row is ineligible, or
    /// a value read is not finite, where `p + r D` and the DAG's arithmetic
    /// part ways). Zero-weight rows are skipped (exactly 0
    /// either way), so a MASKED nonlinear row (two-phase GLS) does not block
    /// the closed form.
    /// Valid until the assignment or `weights` changes. Throws
    /// `std::logic_error` if not sized to the model's rows.
    bool prepare(int32_t var_id, const std::vector<double>& weights);

    /// Returns `prepare(var_id, weights)` and, in the same pass, records each
    /// term's weight under `second` (#272: Novelty scores its W'-argmin under
    /// W). `second_ready()` is true when every row of G_v up to the first
    /// refusal is zero under `second` exactly where it is under `weights` (all
    /// prepare reads of a weight): `prepare(var_id, second)` would then succeed
    /// alike and `delta_second(j)` equals its `delta(j)` to the bit. Otherwise
    /// call `prepare(var_id, second)`. `second` is sized like `weights`.
    bool prepare(int32_t var_id, const std::vector<double>& weights,
                 const std::vector<double>& second);
    [[nodiscard]] bool second_ready() const { return second_ready_; }

    /// Weighted violation delta of moving the prepared variable to `j`, which
    /// must be finite (and so must `j - x0`); see `prepare`.
    [[nodiscard]] double delta(double j) const;
    /// `delta(j)` under the dual prepare's `second` weights; only after a dual
    /// prepare that returned true with `second_ready()` (else throws
    /// `std::logic_error`).
    double delta_second(double j);

    /// d(residual of row ci)/d(var_id), ci = `constraints_of_var(var_id)[k]`
    /// (by G_v position, as Newton walks it), in O(1). When true, bit-identical
    /// (up to a zero's sign on a satisfied Eq row) to
    /// `compute_partial(model, constraint_ids()[ci], var_id)`. False: call
    /// compute_partial -- also when k is past G_v or the row counts disagree.
    bool residual_partial_at(int32_t var_id, size_t k, double& out);

    /// OR-Tools' `row_max_variations` for row `ci` over `vars`: the largest
    /// |d(p - q)/dv| * (ub_v - lb_v), bounding how far ONE in-box jump moves
    /// the residual. Builds the row if pending; +inf for a nonzero slope on an
    /// infinite or NaN box. False when the row is (or is demoted) ineligible,
    /// or the scorer is unsized. Each of `vars` must have `ci` in its G_v, else
    /// `std::logic_error`. O(sum of log |G_v|); FJ calls it once, for the
    /// objective row (#210).
    bool row_max_variation(int32_t ci, const std::vector<int32_t>& vars, double& out);

    /// Closed-form / fallback prepares and cached row partials (diagnostics).
    [[nodiscard]] int64_t fast_prepares() const { return fast_prepares_; }
    [[nodiscard]] int64_t fallback_prepares() const { return fallback_prepares_; }
    [[nodiscard]] int64_t cached_partials() const { return cached_partials_; }
    /// delta_second calls: second weights scored from a dual prepare (#272).
    [[nodiscard]] int64_t second_deltas() const { return second_deltas_; }
    /// Nonzero slopes written since the last layout reset: a build counter, not
    /// memory (the array is 8 B per G_v incidence regardless).
    [[nodiscard]] size_t cached_slopes() const { return cached_slopes_; }
    /// Slope-table length: 0 before the first build and after a layout change
    /// until the next build, then the G_v incidence count AT THAT BUILD. Lets tests catch a stale
    /// table overrunning on rebuild (#176).
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
    static constexpr uint32_t kStrict = 1U << 5U;       // Lt/Gt: + the row's strictness margin
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
        double strict_margin;  // Lt/Gt: the node's margin (ExprNode::const_value), else 0
        // A domain row (#258): `Model::domain_margin()`, added to a positive
        // residual only, as `evaluate`'s Geq arm does; else 0.
        double domain_step;
    };

    // Both prepares: `second` null for the single-weight one.
    bool prepare_impl(int32_t var_id, const std::vector<double>& weights,
                      const std::vector<double>* second);
    // prepare's up-front pass over G_v (x0_ set): false if x0_ is not finite or
    // a weighted row is ineligible; sets second_ready_ from the zero patterns.
    bool refusal_scan(int32_t var_id, const std::vector<double>& weights,
                      const std::vector<double>* second);
    // delta's sum, each term weighted by Term::weight or by second_weights_.
    template <bool kSecond>
    [[nodiscard]] double delta_impl(double j) const;

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
    // A dual prepare's weights under `second`, parallel to terms_. Kept out of
    // Term so a single prepare's term stays one 64-byte line.
    std::vector<double> second_weights_;
    bool second_ready_ = false;
    bool second_terms_ = false;  // the last prepare was dual and succeeded
    // build_row's scratch, kept so a build allocates only the row it keeps.
    std::vector<std::pair<int32_t, double>> merged_;
    std::vector<std::pair<int32_t, double>> side0_;
    std::vector<std::pair<int32_t, double>> side1_;
    double x0_ = 0.0;
    int64_t fast_prepares_ = 0;
    int64_t fallback_prepares_ = 0;
    int64_t cached_partials_ = 0;
    int64_t second_deltas_ = 0;
};

}  // namespace cbls
