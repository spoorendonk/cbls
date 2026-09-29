#pragma once

#include "dag.h"

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace cbls {

double full_evaluate(Model& model);

/// What a `delta_evaluate` call means for a `CustomInvariant` (#166).
///
/// Built-in ops are identical under all three -- they recompute from the
/// assignment in front of them, whatever the caller intends to do next -- so a
/// model with no custom node evaluates the same way it always did whichever of
/// these is passed. Only a custom node reads the mode.
enum class DeltaMode : uint8_t {
    /// The assignment being evaluated is the new committed one. Each custom
    /// node in the dirty cone gets `delta()` then `commit()`. This is what an
    /// applied move, and both legs of an evaluate-forwards-evaluate-back probe,
    /// mean -- and it is the default, so every call site that predates #166
    /// keeps the behaviour it had.
    Commit,
    /// A counterfactual the caller WILL undo. Each custom node in the cone gets
    /// `delta()`, and its previous node value is stashed for the matching
    /// `Rollback` call. No `commit()`.
    Probe,
    /// The caller has put back the assignment the `Probe` was measured from.
    /// Each custom node with a probe pending gets `rollback()` and its stashed
    /// value back, and is NOT re-evaluated; one without a pending probe is
    /// treated as `Commit`, which is the honest reading of a caller that moved
    /// the assignment without probing first.
    ///
    /// Must follow a `Probe` over the same changed-variable set, or the node
    /// values it restores are not the ones the caller thinks they are.
    Rollback,
};

class EditJournal;

// Primary signature: accepts a contiguous range of var IDs.
//
// `journal`, when non-null, is WHERE the structured variables among
// `changed_var_ids` changed since the assignment their previous evaluation was
// measured at (#172): one record per such variable, which a `CustomInvariant`
// reads through `InvariantInputs::edits`. Null -- the default, and every caller
// but the structural batch -- means "no positional information", which an
// invariant reads as "re-read the input". Only custom nodes ever look at it, so
// a model without one evaluates identically either way. Read for the duration
// of this call only.
double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count,
                      DeltaMode mode = DeltaMode::Commit, const EditJournal* journal = nullptr);

/// `delta_evaluate(model, &var_id, 1)` -- a `Commit` of one scalar variable --
/// for a caller that still knows the variable's previous value, which it has
/// already overwritten with the new one (#177).
///
/// Leaves every node value bit-identical to what `delta_evaluate` would. What
/// the old value buys is the cost: a `Sum` the model classified as integral
/// (`ExprNode::kExactSum`) and that currently holds its exact sum
/// (`Model::sum_exact_state`) is moved by its terms' changes, O(1) per changed
/// term, instead of being re-summed over all of them. On a MIP row with
/// integral coefficients over Bool/Int columns that is the difference between
/// O(|G_v|) and O(sum of |row| over G_v) per committed move. Where the values
/// cannot be shown exact -- a fractional term, a term above 2^52 / (term count),
/// a NaN or an infinity -- the Sum is re-summed as before, so the result is the
/// re-sum's bits either way.
///
/// Precondition, stronger than `delta_evaluate`'s: every node value must
/// describe the assignment apart from `var_id`'s change. A variable written
/// without a walk (`restore_state` with no `full_evaluate`, say) leaves a Sum
/// marked exact on a stale base, which this would update rather than repair.
double commit_scalar_move(Model& model, int32_t var_id, double old_value);

/// How often the dirty `Sum`s of exact-sum eligibility (`ExprNode::kExactSum`)
/// were updated by their terms' changes rather than re-summed, on this thread,
/// since the caller last assigned it `ExactSumCounters{}`. Diagnostics for
/// tests and profiling (#177); nothing
/// in the engine reads them. Counted on the eligible Sums only, so the
/// re-summing path of every other node pays nothing for them.
struct ExactSumCounters {
    uint64_t incremental = 0;
    uint64_t resummed = 0;
};
ExactSumCounters& exact_sum_counters() noexcept;

// Convenience overloads
inline double delta_evaluate(Model& model, const std::vector<int32_t>& changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit,
                             const EditJournal* journal = nullptr) {
    return delta_evaluate(model, changed_var_ids.data(), changed_var_ids.size(), mode, journal);
}

inline double delta_evaluate(Model& model, const std::set<int32_t>& changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit) {
    std::vector<int32_t> ids(changed_var_ids.begin(), changed_var_ids.end());
    return delta_evaluate(model, ids.data(), ids.size(), mode);
}

inline double delta_evaluate(Model& model, std::initializer_list<int32_t> changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit) {
    return delta_evaluate(model, changed_var_ids.begin(), changed_var_ids.size(), mode);
}

// Reverse-mode AD over the cone of `expr_id` (the nodes reachable from it
// through children), visited in reverse topological order. Cost O(c log c) for
// a cone of c nodes, not O(|nodes|); a cone too large for the sort to pay falls
// back to the full-order walk. Results are bit-identical either way. The
// rationale and the cutover are at `reverse_sweep` in src/dag_ops.cpp.
//
// All three share one thread_local scratch, so none may be called from inside
// another on the same thread -- i.e. not from a `CustomInvariant::partial`. That
// is enforced: the nested call throws `std::logic_error`.
//
// `expr_id` must be a node id of `model` (not a variable handle); it is not
// range-checked here, on the hot path. The Python binding checks it.

/// ∂expr/∂var for one variable; 0.0 for a `var_id` outside the model.
double compute_partial(const Model& model, int32_t expr_id, int32_t var_id);

/// Batch AD: partials of `expr_id` w.r.t. ALL variables in one reverse pass.
/// Returns a vector of size `num_vars()`; entry[i] = ∂expr/∂var_i. The O(num_vars)
/// result is the floor here; prefer `compute_partials_sparse` on a large model.
std::vector<double> compute_all_partials(const Model& model, int32_t expr_id);

/// Sparse batch AD: fills `out` (cleared first) with (var_id, ∂expr/∂var_id) for
/// every variable whose partial is nonzero -- each variable at most once, in
/// the deterministic order the sweep first reached it. Values are bit-identical
/// to `compute_all_partials`' entries. Costs the sweep plus O(output) -- never
/// the O(num_vars) result `compute_all_partials` pays -- which is what a per-row
/// slope cache over a linear model wants. `out` is the caller's buffer so a hot
/// loop can reuse its capacity.
///
/// Partials are taken at the CURRENT node values, so a cached slope stays valid
/// only while every op on the path has a constant local derivative: `Sum`,
/// `Neg`, `Prod` or `Div` by a constant, and `Leq`/`Geq`/`Lt`/`Gt` do; `Eq` is
/// sign(residual), hence 0.0 when satisfied and sign-flipping otherwise; `Neq`
/// and the structural ops are always 0.0. A partial that is exactly 0.0 --
/// cancelled, or through a zero local derivative -- is absent, not listed.
void compute_partials_sparse(const Model& model, int32_t expr_id,
                             std::vector<std::pair<int32_t, double>>& out);

}  // namespace cbls
