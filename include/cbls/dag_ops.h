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
/// Built-in ops recompute from the assignment in front of them under all three,
/// whatever the caller intends to do next, with one exception: an incremental
/// `Sum` (#188) in the cone is stashed by a `Probe` and written back from the
/// stash by the matching `Rollback`, so that a committed value carrying drift
/// comes back to the bit instead of being re-summed. Otherwise only a custom
/// node reads the mode.
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

/// The inexact update that would be an incremental `Sum`'s kIncSumPeriod-th
/// since its last re-sum re-sums it instead (#188), so a Sum carries at most
/// kIncSumPeriod - 1 = 63. An exact update -- every one on integral data -- does
/// not count. A safety cap on how far the drift bound can grow between
/// re-groundings, not a tuned optimum: chosen by a pre-registered rule on
/// held-out instances, where it cost under 1% of throughput against never
/// re-summing (`docs/benchmarks/incremental-sum-drift.md`). 0 means never: the
/// drift is then bounded only by the batch-end re-grounding and the gate.
constexpr uint32_t kIncSumPeriod = 64;

/// A `Commit` of one scalar variable whose previous value was `old_value` (the
/// caller has already written the new one): `delta_evaluate(model, &var_id, 1)`,
/// except that every incremental `Sum` in the cone (`ExprNode::kIncSum`) is moved
/// by its terms' changes, O(1) per changed term, instead of being re-summed over
/// all of them (#177, #188). On a MIP row that is the difference between
/// O(|G_v|) and O(sum of |row| over G_v) per committed move.
///
/// Each update is tested for exactness (TwoSum on `new - old` and on the add).
/// An exact one changes no bits relative to the re-sum's arithmetic and adds no
/// drift. An inexact one leaves the Sum up to one rounding of each away from the
/// real sum of its stored terms, and adds those roundings -- exactly, as TwoSum
/// computes them -- to the Sum's `IncSumState::drift_bound`. After
/// `kIncSumPeriod - 1` inexact updates carried (at most 63), the next inexact
/// one re-sums the Sum instead. So does a non-finite value, and so does the
/// Sum's first commit after a `full_evaluate` or a plain `delta_evaluate` over
/// it, both of which re-sum without the check.
///
/// Precondition, stronger than `delta_evaluate`'s: every node value must
/// describe the assignment apart from `var_id`'s change. A variable written
/// without a walk (`restore_state` with no `full_evaluate`, say) leaves an
/// incremental Sum on a stale base, which this would update rather than repair.
double commit_scalar_move(Model& model, int32_t var_id, double old_value);

/// The `Probe` counterpart: the commit's updates, applied from the committed
/// values, so an identity move on a drifted state scores exactly 0. Where the
/// commit would re-sum instead (an untracked Sum, `kIncSumPeriod` reached, a
/// non-finite value) the probe still updates, so its score can differ from the
/// commit's by up to the Sum's drift bound plus the re-sum's rounding. The Sums
/// in the cone are stashed as the probe first writes each, and the matching
/// `delta_evaluate(..., Rollback)` writes them back, so a drifted committed
/// state comes back to the bit. No Sum's drift state changes.
double probe_scalar_move(Model& model, int32_t var_id, double old_value);

/// Re-sums, checked, the incremental Sum in `slot` and re-evaluates the rows
/// that read it -- by construction nothing else does. Afterwards the Sum
/// is a fresh re-sum (`IncSumState::drifting` is 0). O(|terms| + |readers|).
void reground_inc_sum(Model& model, int32_t slot);

/// `reground_inc_sum` for every Sum still drifting on `IncSums::drifted`,
/// appending each one's slot to `regrounded`, and empties the list. O(1) when
/// nothing drifted since the last call. FeasibilityJump calls it at the end of
/// every batch, which is what keeps drift inside one.
void reground_drifted_sums(Model& model, std::vector<int32_t>& regrounded);

/// Diagnostics for tests and profiling (#177, #188), on this thread since the
/// caller last assigned it `IncrementalSumCounters{}`; nothing in the engine
/// reads them.
struct IncrementalSumCounters {
    /// Commits that took an incremental Sum as its terms' updates left it.
    uint64_t incremental = 0;
    /// Commit-mode walks that re-summed one instead.
    uint64_t resummed = 0;
    /// Of the updates behind `incremental`, the inexact ones.
    uint64_t inexact = 0;
    /// Term updates a Probe applied to an incremental Sum: one per probe of a
    /// variable that is a term of it.
    uint64_t probe_pushes = 0;
    /// Sums re-summed by `reground_inc_sum`, from any caller.
    uint64_t regrounded = 0;
    /// FeasibilityJump's side. GLS iterations that found no improving jump.
    uint64_t local_minima = 0;
    /// Of those, the ones where some row's verdict was undecided by its Sum's
    /// drift bound, so the Sum was re-summed before any decision (the gate).
    uint64_t gated_minima = 0;
    /// Sums the gate re-summed.
    uint64_t gate_resums = 0;
    /// Gated minima where a re-sum moved a row into or out of V, so the
    /// iteration sampled again instead of bumping or stopping.
    uint64_t gate_flips = 0;
    /// Sums re-summed at the end of a batch, at the start of a Novelty Jump or
    /// before a Feasible verdict, because they drifted.
    uint64_t batch_end_resums = 0;
};
IncrementalSumCounters& incremental_sum_counters() noexcept;

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
