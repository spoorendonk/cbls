#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace cbls {

class Model;

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
/// **Exact in the arithmetic, not bit-identical.** `p + r D` is not the row the
/// DAG would re-sum, so a score can differ from `weighted_violation_delta` in its
/// last bits (the equivalence tests pin a tight relative tolerance). Nothing
/// accumulates: a committed jump still goes through `delta_evaluate`, so node
/// values stay exactly the DAG's and every prepare reads them fresh.
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
/// LNS repair builds one per call) pays only for the rows it touches.
///
/// Storage: 4 bytes per row (a slot: ineligible, pending, or the index of its
/// built record), plus, for each row BUILT, a 20-byte record and 12 bytes per
/// nonzero in one pooled CSR (ascending variable ids, parallel slopes). A row is
/// classified once, before it is built, and never rebuilt. The slopes are
/// structure, but each portfolio worker's FJ builds its own; the build also
/// sizes `dag_ops.cpp`'s thread_local adjoint scratch, which a pure MIP
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

    /// Size the per-row table to `n` rows. New rows start ineligible.
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

    /// d(residual of row ci)/d(var_id), bit-identical (up to the sign of a zero
    /// on a satisfied Eq row) to
    /// `compute_partial(model, constraint_ids()[ci], var_id)`, when the row's
    /// cached slope provably is (see the definition). False: call compute_partial.
    bool residual_partial(int32_t ci, int32_t var_id, double& out);

    /// Prepares that took the closed form / fell back, and row partials served
    /// from the cache. Diagnostics, and the pins on the wiring in tests.
    [[nodiscard]] int64_t fast_prepares() const { return fast_prepares_; }
    [[nodiscard]] int64_t fallback_prepares() const { return fallback_prepares_; }
    [[nodiscard]] int64_t cached_partials() const { return cached_partials_; }
    /// Slopes held in the pool: what the cache costs, 12 B each.
    [[nodiscard]] size_t pooled_slopes() const { return pool_vars_.size(); }

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
    static constexpr uint32_t kNewtonExact = 1U << 6U;  // see residual_partial
    // A built row: its slice of the pool and its two residual arguments.
    struct BuiltRow {
        uint32_t begin = 0;  // into pool_vars_ / pool_slopes_
        uint32_t count = 0;
        int32_t p_id = -1;  // first argument of the residual
        int32_t q_id = -1;  // second argument
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
    [[nodiscard]] double slope_of(const BuiltRow& row, int32_t var_id) const;

    const Model& model_;
    std::vector<uint32_t> slots_;  // per row
    std::vector<BuiltRow> built_;  // one record per built row
    std::vector<int32_t> pool_vars_;
    std::vector<double> pool_slopes_;
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
