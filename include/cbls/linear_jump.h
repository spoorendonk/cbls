#pragma once

#include "dag.h"

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
/// LNS repair builds one per call) pays only for the rows it touches. Storage is
/// per row -- a 64-byte header for every row, plus ascending variable ids and
/// slopes, 12 bytes per nonzero, for each row built -- so an extension (#167)
/// invalidates exactly the rows it changed, the same unit as FeasibilityJump's
/// other per-row tables. The slopes are structure, but each portfolio worker's FJ
/// builds its own; the build also sizes `dag_ops.cpp`'s thread_local adjoint
/// scratch, which a pure MIP otherwise never allocated.
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
    [[nodiscard]] size_t num_rows() const { return rows_.size(); }

    /// (Re)classify row `ci` and drop any slopes cached for it. Call for every
    /// row whose body changed -- a new row, or one a grown Sum sits inside.
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
    /// `std::logic_error` if this scorer is not sized to the model's rows (a
    /// `Model::extend` it was not told about).
    bool prepare(int32_t var_id, const std::vector<double>& weights);

    /// Weighted violation delta of moving the prepared variable to `j`, which
    /// must be finite (and so must `j - x0`); see `prepare`.
    [[nodiscard]] double delta(double j) const;

    /// d(residual of row ci)/d(var_id), BIT-IDENTICAL to
    /// `compute_partial(model, constraint_ids()[ci], var_id)`, when the row's
    /// cached slope provably is (see the definition). False: call compute_partial.
    bool residual_partial(int32_t ci, int32_t var_id, double& out);

    /// Prepares that took the closed form / fell back, and row partials served
    /// from the cache. Diagnostics, and the pins on the wiring in tests.
    [[nodiscard]] int64_t fast_prepares() const { return fast_prepares_; }
    [[nodiscard]] int64_t fallback_prepares() const { return fallback_prepares_; }
    [[nodiscard]] int64_t cached_partials() const { return cached_partials_; }

private:
    enum class RowState : uint8_t { Ineligible, Pending, Ready };
    // Every row carries one of these whether or not it is ever built, so it is
    // kept to 64 bytes: the two residual arguments as (id, is_var) fields rather
    // than padded ChildRefs.
    struct Row {
        std::vector<int32_t> vars;   // ascending
        std::vector<double> slopes;  // r = d(p - q)/dv, parallel to vars
        int32_t p_id = -1;           // first argument of the residual
        int32_t q_id = -1;           // second argument
        RowState state = RowState::Ineligible;
        bool p_is_var = false;
        bool q_is_var = false;
        bool p_literal = false;  // a Const node: comparison_residual's sentinel flag
        bool q_literal = false;
        bool is_abs = false;        // Eq: |p - q|
        bool strict = false;        // Lt/Gt: + the strictness epsilon
        bool newton_exact = false;  // see residual_partial
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

    void build_row(int32_t ci);
    [[nodiscard]] double child_value(int32_t id, bool is_var) const;
    [[nodiscard]] static double slope_of(const Row& row, int32_t var_id);

    const Model& model_;
    std::vector<Row> rows_;
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
