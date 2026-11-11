#pragma once

#include "dag.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <utility>
#include <vector>

namespace cbls {

double full_evaluate(Model& model);

/// What a `delta_evaluate` call means. `Commit` and `Probe` recompute the cone
/// from the current assignment; a `Probe` also logs the cone's prior values
/// (`ProbeUndo`, #270), which `Rollback` writes back: O(cone) stores instead
/// of collect, sort and evaluate, and exact, so a drifting incremental `Sum`
/// (#188) returns to the bit.
enum class DeltaMode : uint8_t {
    /// The evaluated assignment is the new committed one: an applied move, and
    /// both legs of an evaluate-forwards-evaluate-back probe. The default.
    Commit,
    /// A counterfactual the caller WILL undo; the cone's values are logged for
    /// the matching `Rollback`.
    Probe,
    /// The caller restored the `Probe`'s starting assignment; the probe's cone
    /// gets its logged values back, nothing re-evaluated. Must directly follow
    /// a `Probe` over the same changed-variable set in the same order: any
    /// other walk, `full_evaluate`, `reground_inc_sum`,
    /// `Model::set_objective_bound` or `Model::set_domain_tolerance` between
    /// drops the log and this acts as `Commit`. Over a different set it writes
    /// the log back, then commits the union of both sets: correct either way.
    Rollback,
};

// Primary signature: accepts a contiguous range of var IDs.
double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count,
                      DeltaMode mode = DeltaMode::Commit);

/// An incremental Sum's kIncSumPeriod-th inexact update since its last re-sum
/// re-sums instead (#188), so it carries at most 63; exact updates (all, on
/// integral data) do not count. A safety cap on drift between re-groundings,
/// not a tuned optimum: picked by a pre-registered rule on held-out instances,
/// costing <1% throughput vs never re-summing
/// (`docs/benchmarks/incremental-sum-drift.md`). 0 means never: drift is then
/// bounded only by the batch-end re-grounding and the gate.
constexpr uint32_t kIncSumPeriod = 64;

/// `Commit` of one scalar whose previous value was `old_value` (new one
/// already written): `delta_evaluate(model, &var_id, 1)`, but each incremental
/// `Sum` in the cone (`ExprNode::kIncSum`) moves by its terms' changes, O(1)
/// each, rather than re-summing (#177, #188): O(|G_v|) instead of
/// O(sum of |row| over G_v) per MIP move.
///
/// Each update is TwoSum-tested: an exact one adds no rounding and no drift (on
/// integral data within 2^53 every update is exact, so values equal the
/// re-sum's bit for bit); an inexact one adds its exact roundings to
/// `IncSumState::drift_bound`. After
/// `kIncSumPeriod - 1` (63) inexact updates the next inexact one re-sums, as
/// do a non-finite value and the first commit after a `full_evaluate` or
/// plain `delta_evaluate` over the Sum (which re-sum unchecked).
///
/// Precondition, stronger than `delta_evaluate`'s: every node value describes
/// the assignment apart from `var_id`'s change. A write without a walk
/// (`restore_state` without `full_evaluate`) leaves a stale base this would
/// update rather than repair.
double commit_scalar_move(Model& model, int32_t var_id, double old_value);

/// The `Probe` counterpart: the commit's updates from the committed values,
/// so an identity move on a drifted state scores exactly 0. Where the commit
/// would re-sum (untracked Sum, `kIncSumPeriod` reached, non-finite) the
/// probe still updates, so scores can differ by the drift bound plus the
/// re-sum's rounding. `delta_evaluate(..., Rollback)` restores the state to
/// the bit (see `DeltaMode`). No drift state changes.
double probe_scalar_move(Model& model, int32_t var_id, double old_value);

/// Re-sums (checked) the incremental Sum in `slot` and re-evaluates the rows
/// that read it (by construction its only readers); afterwards
/// `IncSumState::drifting` is 0. O(|terms| + |readers|).
void reground_inc_sum(Model& model, int32_t slot);

/// `reground_inc_sum` for every Sum still drifting on `IncSums::drifted`,
/// appending each slot to `regrounded`, then empties the list; O(1) if nothing
/// drifted since the last call.
/// FeasibilityJump calls it at every batch end, keeping drift inside a batch.
void reground_drifted_sums(Model& model, std::vector<int32_t>& regrounded);

/// Diagnostics for tests and profiling (#177, #188), per thread since last
/// reset to `IncrementalSumCounters{}`; the engine never reads them.
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
    /// FeasibilityJump: local minima where some row's verdict was undecided by
    /// its Sum's drift bound, so the Sum was re-summed first (the gate).
    uint64_t gated_minima = 0;
    /// Gated minima where the re-sum moved a row into or out of V, so the
    /// iteration sampled again instead of bumping or stopping.
    uint64_t gate_flips = 0;
};
IncrementalSumCounters& incremental_sum_counters() noexcept;

/// One-variable walks on this thread since last reset to `ConeCacheCounters{}`
/// (#271): cone read from `Model::freeze()`'s cache, or collected and sorted
/// (unfrozen model, or variable past the cap). A Rollback from the undo log
/// and a multi-variable walk (always collects) count in neither. `solve()`
/// reports its own run's in `SearchCounters`.
struct ConeCacheCounters {
    uint64_t cached_walks = 0;
    uint64_t fallback_walks = 0;
};
ConeCacheCounters& cone_cache_counters() noexcept;

// Convenience overloads
inline double delta_evaluate(Model& model, const std::vector<int32_t>& changed_var_ids,
                             DeltaMode mode = DeltaMode::Commit) {
    return delta_evaluate(model, changed_var_ids.data(), changed_var_ids.size(), mode);
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

// Reverse-mode AD over the cone of `expr_id`, in reverse topological order:
// O(c log c) for a cone of c nodes, falling back to the full-order walk when
// the sort would not pay; bit-identical either way (cutover at `reverse_sweep`
// in src/dag_ops.cpp). All three share one thread_local scratch, so none may
// be called from inside another on one thread. `expr_id` must be a node id of
// `model` (not a variable handle); unchecked here (hot path), checked by the Python binding.

/// One chain-rule step: `acc += edge * carried`, `edge` the local derivative,
/// `carried` the adjoint or forward slope. Every propagation pass uses it
/// (reverse sweep, `UpConePartials`, FJ's pruned breakpoint slopes; #260).
/// A zero edge (untaken If branch, Min's unselected operand) is skipped, so
/// an inf/NaN `carried` cannot turn a true 0 partial into NaN; live paths
/// still propagate inf/NaN. For finite `carried` only a zero's sign changes.
inline void add_chain_term(double& acc, double edge, double carried) {
    if (edge != 0.0) {
        acc += edge * carried;
    }
}

/// d(parent)/d(child) for one DAG edge, summed over `parent`'s operands naming
/// `child` (the forward up-cone walk sees a parent once). For a Sum it is the
/// operand's multiplicity, 1 unless `Model::close()` marked it
/// `kRepeatedChild`, so an n-term Sum costs O(1) per edge, not O(n); other ops
/// scan their children. Requires a closed model.
double up_edge_derivative(const Model& model, int32_t parent, const ChildRef& child);

/// ∂expr/∂var for one variable; 0.0 for a `var_id` outside the model.
double compute_partial(const Model& model, int32_t expr_id, int32_t var_id);

/// d(node)/d(var) for every node above one variable, forward mode (#249):
/// collect the up-cone (via `Model::parents`), sort topologically, carry the
/// slope along each parent edge by `up_edge_derivative`. Same chain rule as
/// `compute_partial`, so `of(row)` matches it up to summation order on finite
/// paths; both skip zero edges (`add_chain_term`). Where infinite paths meet
/// the modes can differ (inf + -inf = NaN at different nodes), including a
/// parent naming a child twice (`z / z`): this pass sums the operand
/// derivatives before the zero test, the reverse sweep tests each.
///
/// COST: O(u log u) for an up-cone of u nodes plus their parent edges, giving
/// EVERY node's partial at once, vs `compute_partial`'s whole row cone per
/// row. Wins with a large row cone and small up-cone (separable `obj <=
/// bound`; UC-CHPED ucp200-168p: ~30 ms per reverse sweep over a 1.38M-node
/// cone, 42% of a 60 s run); loses with a large up-cone and small rows (a
/// variable in many nonlinear rows), so `build` gives up past a node budget
/// and the caller takes the reverse sweep.
///
/// Requires a closed model. Not thread-safe: one instance per thread, reused
/// (epoch-stamped entries, so a call never clears the arrays).
class UpConePartials {
public:
    /// Collects and sweeps `var_id`'s up-cone. False, every `of` at 0, when the
    /// model is not closed or the cone outgrows `max_nodes`; a throwing build
    /// also leaves every `of` at 0.
    bool build(const Model& model, int32_t var_id, size_t max_nodes);

    /// d(node)/dv at the point `build` saw; 0 for a node the variable cannot reach.
    [[nodiscard]] double of(int32_t node_id) const {
        const auto i = static_cast<size_t>(node_id);
        return i < stamp_.size() && stamp_[i] == epoch_ ? slope_[i] : 0.0;
    }

private:
    bool sweep(const Model& model, int32_t var_id, size_t max_nodes);
    void next_epoch();

    std::vector<uint32_t> stamp_;
    std::vector<double> slope_;
    std::vector<int32_t> cone_;
    uint32_t epoch_ = 0;
};

/// Batch AD: partials of `expr_id` w.r.t. ALL variables in one reverse pass;
/// entry[i] = ∂expr/∂var_i, size `num_vars()`. That O(num_vars) result is
/// the floor; prefer `compute_partials_sparse` on a large model.
std::vector<double> compute_all_partials(const Model& model, int32_t expr_id);

/// Sparse batch AD: fills `out` (cleared first; the caller's buffer, reused)
/// with (var_id, ∂expr/∂var_id) for each variable with a nonzero partial,
/// once each, in the sweep's deterministic first-reach order; values
/// bit-identical to `compute_all_partials`. Costs the sweep plus O(output),
/// never O(num_vars). Exactly-0.0 partials are absent.
///
/// Partials are at CURRENT node values, so a cached slope stays valid only
/// while every op on the path has a constant local derivative: `Sum`, `Neg`,
/// `Prod`/`Div` by a constant, `Leq`/`Geq`/`Lt`/`Gt`. `Eq` is sign(residual)
/// (0.0 when satisfied, sign-flipping otherwise); `Neq` and the structural
/// ops are always 0.0.
void compute_partials_sparse(const Model& model, int32_t expr_id,
                             std::vector<std::pair<int32_t, double>>& out);

}  // namespace cbls
