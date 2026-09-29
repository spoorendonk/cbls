#pragma once

#include "dag.h"

#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace cbls {

double full_evaluate(Model& model);

/// What a `delta_evaluate` call means for a `CustomInvariant` (#166), and for
/// the node values themselves (#177).
///
/// A built-in op recomputes from the assignment in front of it under `Commit`
/// and `Probe` alike. What differs is the way back: `Probe` stashes every dirty
/// node's value, and the matching `Rollback` writes the stash back rather than
/// re-evaluating, so the committed state -- including an incremental `Sum`'s
/// drift, see `commit_scalar_move` -- comes back bit for bit.
enum class DeltaMode : uint8_t {
    /// The assignment being evaluated is the new committed one. Each custom
    /// node in the dirty cone gets `delta()` then `commit()`. This is what an
    /// applied move, and both legs of an evaluate-forwards-evaluate-back probe,
    /// mean -- and it is the default, so every call site that predates #166
    /// keeps the behaviour it had.
    Commit,
    /// A counterfactual the caller WILL undo. Every node in the cone has its
    /// value stashed for the matching `Rollback` call; each custom node gets
    /// `delta()` and no `commit()`.
    Probe,
    /// The caller has put back the assignment the `Probe` was measured from.
    /// The probe's stash is written back and nothing is re-evaluated; each
    /// custom node with a probe pending also gets `rollback()`. With no probe
    /// pending it is treated as `Commit`, which is the honest reading of a
    /// caller that moved the assignment without probing first.
    ///
    /// Must follow its `Probe` with no other evaluation in between, or the node
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

/// How many term updates an incremental `Sum` may carry before a Commit
/// re-sums it (#177). The drift a Sum can hold is at most this many roundings,
/// and a re-sum costs its whole length once per this many updates. Chosen by
/// the sweep recorded on #177; a parameter, not a derived constant.
constexpr int kIncrementalSumPeriod = 64;
// IncrementalSumState::age counts in its low seven bits and climbs to
// kIncrementalSumPeriod - 1.
static_assert(kIncrementalSumPeriod >= 1 && kIncrementalSumPeriod <= 128,
              "IncrementalSumState::age counts in seven bits");

/// A `Commit` of one scalar variable whose previous value was `old_value` (the
/// caller has already written the new one): `delta_evaluate(model, &var_id,
/// 1)`, except that every `ExprNode::incremental_sum` Sum in the cone is moved
/// by its terms' changes, O(1) per changed term, instead of re-summed over all
/// of them (#177). The price is drift -- floating-point rounding of at most
/// `kIncrementalSumPeriod` updates per Sum, and none on integral data -- which
/// `reground_incremental_sums` removes. The rules and the consistency argument
/// are at the top of the incremental-Sum section of `src/dag_ops.cpp`.
double commit_scalar_move(Model& model, int32_t var_id, double old_value);

/// The `Probe` counterpart: the same updates, applied from the committed values
/// and stashed for the matching `delta_evaluate(..., DeltaMode::Rollback)`,
/// which writes the stash back bit for bit. A probe of the value the variable
/// already holds therefore changes no node value at all.
double probe_scalar_move(Model& model, int32_t var_id, double old_value);

/// Re-sums every incremental `Sum` that has drifted since the last call, and
/// re-evaluates their cones. Returns whether there was anything to do. Cost:
/// those cones, not the model. FeasibilityJump calls it at the end of every
/// batch, which is what keeps drift inside a batch.
bool reground_incremental_sums(Model& model);

/// The same, for only the drifted Sums in the cones below `roots` (node ids),
/// and what lies above those Sums. `rewritten` receives every node it
/// re-evaluated, with the value the node held before, so that a caller keeping
/// state derived from node values can settle what moved. FeasibilityJump calls
/// it on the rows in V before a GLS weight bump, so that no row is bumped whose
/// exact residual is within tolerance. Cost: those cones.
bool reground_incremental_sums_below(Model& model, const std::vector<int32_t>& roots,
                                     std::vector<std::pair<int32_t, double>>& rewritten);

/// How often the dirty incremental `Sum`s were moved by their terms' changes
/// rather than re-summed, on this thread, since the last reset. Diagnostics for
/// tests and profiling (#177); nothing in the engine reads them.
struct IncrementalSumCounters {
    uint64_t incremental = 0;
    uint64_t resummed = 0;
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
