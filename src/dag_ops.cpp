#include "cbls/dag_ops.h"

#include "cbls/custom_invariant.h"
#include "cbls/model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cbls {

namespace detail {

// Kahn's algorithm over the node-to-node edges only (variable children are
// sources and carry no in-degree). Children come out before their parents.
//
// Walks the back-references `Model::rebuild_back_references` has just built,
// rather than a child->parents adjacency of its own: that was one more vector per
// node, allocated on each of the two sorts per build. Those lists hold each
// parent once, where the old adjacency listed a parent naming the same child
// twice (`prod(n, n)`) twice and counted both edges into its in-degree. The
// order is unchanged: those two entries sat next to each other, a parent's
// in-degree could only reach zero on the second, and nothing was queued between
// them -- so a parent is queued at the same point either way.
//
// `sorted` is its own FIFO: entries are appended at the back and consumed from
// `head`, the order a std::queue gave without the deque's block allocations.
std::vector<int32_t> compute_topo_order(const Model& model) {
    const auto n = static_cast<int32_t>(model.num_nodes());
    std::vector<int32_t> in_degree(static_cast<size_t>(n), 0);
    for (int32_t nid = 0; nid < n; ++nid) {
        for (const int32_t parent_id : model.parents(nid)) {
            ++in_degree[parent_id];
        }
    }

    std::vector<int32_t> sorted;
    sorted.reserve(static_cast<size_t>(n));
    for (int32_t nid = 0; nid < n; ++nid) {
        if (in_degree[nid] == 0) {
            sorted.push_back(nid);
        }
    }
    for (size_t head = 0; head < sorted.size(); ++head) {
        for (const int32_t parent_id : model.parents(sorted[head])) {
            if (--in_degree[parent_id] == 0) {
                sorted.push_back(parent_id);
            }
        }
    }
    return sorted;
}

}  // namespace detail

namespace {

// Is this thread already inside `full_evaluate` or `delta_evaluate`?
//
// `CustomInvariant` (#166) puts arbitrary user code inside the evaluation walk,
// and its motivating use case -- a black-box or simulation value -- is exactly
// the code that might reach for a sub-model. Re-entering from there is not merely
// unsupported, it is SILENT PERMANENT CORRUPTION of this thread: the nested call
// does `dirty_list.clear()` on the same `thread_local` vector that the outer
// call's `DirtyFlagGuard` holds a reference to, so the guard then clears the
// INNER call's ids and leaks the outer call's flags -- and a leaked flag makes
// `delta_evaluate`'s seeding loop skip that node for the life of the process (see
// `DirtyFlagGuard`). The symptom would be a node that quietly stops updating,
// long after and nowhere near the cause. So it is refused rather than documented.
//
// One `thread_local` test and one store per call, on both entry points -- the
// price `Model::has_custom_nodes()` already pays per call. It changes no value and
// draws no random number, so trajectories are unaffected; re-verified against main
// with the #166 witness.
thread_local bool in_evaluation = false;

// RAII, so that an exception out of user code inside the walk -- which
// `CustomInvariant` documents as possible -- clears the flag on the way out
// instead of poisoning every later call on this thread.
class EvaluationGuard {
public:
    explicit EvaluationGuard(const char* entry) {
        if (in_evaluation) {
            // Not an assert: this is reachable from user code in a Release build,
            // and the silent wrong answer above is the thing being prevented.
            throw std::logic_error(std::string(entry) +
                                   ": re-entered from inside an evaluation. A CustomInvariant's "
                                   "evaluate/delta/partial must not call full_evaluate or "
                                   "delta_evaluate, on this model or any other.");
        }
        in_evaluation = true;
    }
    EvaluationGuard(const EvaluationGuard&) = delete;
    EvaluationGuard& operator=(const EvaluationGuard&) = delete;
    EvaluationGuard(EvaluationGuard&&) = delete;
    EvaluationGuard& operator=(EvaluationGuard&&) = delete;
    ~EvaluationGuard() { in_evaluation = false; }
};

}  // namespace

double full_evaluate(Model& model) {
    const EvaluationGuard guard("full_evaluate");
    // A from-scratch pass is a `CustomInvariant`'s reset point (#166): every
    // custom node below is about to be told `evaluate()`, which redefines its
    // committed state, so a probe left open by a caller that never rolled back
    // is discarded here rather than firing against an unrelated assignment. One
    // predictable branch per call on a model that has no custom node.
    if (model.has_custom_nodes()) {
        model.clear_custom_probes();
    }
    // Re-sums without the rounding check, so no incremental Sum is tracked
    // afterwards and none carries drift (#188); its first commit re-sums it,
    // checked. Also what sizes the state: every close and every structural
    // rebuild ends in a full pass.
    IncSums& sums = model.inc_sums();
    sums.slots.assign(model.inc_sum_nodes().size(), IncSumState{});
    sums.drifted.clear();
    sums.probe_stash.clear();
    sums.probe_pending = false;
    for (int32_t nid : model.topo_order()) {
        model.set_node_value_unchecked(nid, evaluate(model.nodes()[nid], model));
    }
    if (model.objective_id() >= 0) {
        return model.node_values()[model.objective_id()];
    }
    return 0.0;
}

namespace {

// Recompute a marked dirty set in topological order, by whichever of the two
// routes is cheaper for THIS set. Its own function because it is a separate job
// from finding the set: the BFS above decides WHAT is stale, this decides how to
// walk it in dependency order.
//
// Sorting d entries costs O(d log d), with two scattered `topo_pos_` loads per
// comparison; scanning `topo_order()` and testing the flag costs O(|nodes|)
// sequential byte tests. Sorting wins by orders of magnitude in the regime that
// matters -- a few dozen dirty nodes against the 4.3M of the largest MIPfeas
// instance, where the scan was ~2M flag tests to recompute a handful, and delta
// evaluation was not sublinear in the model at all. It loses at the other end:
// as d approaches |nodes| the sort does ~log2(d) scattered comparisons per
// element where the scan does one sequential test, and a dense continuous model
// -- MINLPLib's regime, not this roster's -- can sit there.
//
// The condition is the cost model itself rather than a tuned constant: sort
// while d*log2(d) is under |nodes|, otherwise scan. Both routes produce the same
// order and the same values.
//
// `eval_node` is a template parameter rather than a `std::function` so that the
// built-in instantiation -- the one every model without a custom node takes --
// inlines the plain `evaluate()` call and compiles to what this loop was before
// #166. The custom-aware instantiation is a second, separate body.
template <typename EvalNode>
void evaluate_dirty_in_topo_order(Model& model, std::vector<int32_t>& dirty_list,
                                  const std::vector<uint8_t>& dirty_flags, size_t num_nodes,
                                  EvalNode eval_node) {
    const size_t dirty_count = dirty_list.size();
    size_t log2_dirty = 0;
    while ((size_t{1} << (log2_dirty + 1)) <= dirty_count) {
        ++log2_dirty;
    }
    if (dirty_count * (log2_dirty + 1) < num_nodes) {
        std::sort(dirty_list.begin(), dirty_list.end(), [&model](int32_t a, int32_t b) {
            return model.topo_position(a) < model.topo_position(b);
        });
        for (int32_t nid : dirty_list) {
            model.set_node_value_unchecked(nid, eval_node(nid));
        }
        return;
    }
    for (int32_t nid : model.topo_order()) {
        if (dirty_flags[nid] != 0) {
            model.set_node_value_unchecked(nid, eval_node(nid));
        }
    }
}

// Which of a custom node's inputs this pass recomputed, as indices into its
// children (#166).
//
// Derived from the dirty set the walk is already carrying rather than by
// comparing values against a cached copy: comparing would cost O(|elements|)
// per structured input, which is exactly the cost an incremental invariant
// exists to avoid. The result is therefore a SUPERSET of the inputs that really
// changed -- a node child that recomputed to the same value is still listed --
// which is what `CustomInvariant::delta` documents.
//
// Cost is O(arity * count), from the linear `std::find` over the changed-variable
// range per variable input. `count` is 1 on every path but the inner solver's
// multi-variable Newton step, where it is the number of Float variables with a
// usable partial -- so O(arity) in practice, against the O(sum of input sizes) a
// value comparison would cost. A var-id -> input-index map would beat it only at
// an arity and a `count` no caller has.
void collect_changed_inputs(const Model& model, const ExprNode& node,
                            const int32_t* changed_var_ids, size_t count,
                            const std::vector<uint8_t>& dirty_flags, std::vector<int32_t>& out) {
    out.clear();
    const ConstSpan<ChildRef> children = model.children(node);
    for (size_t i = 0; i < children.size(); ++i) {
        const ChildRef& ref = children[i];
        const bool changed = ref.is_var ? std::find(changed_var_ids, changed_var_ids + count,
                                                    ref.id) != changed_var_ids + count
                                        : dirty_flags[ref.id] != 0;
        if (changed) {
            out.push_back(static_cast<int32_t>(i));
        }
    }
}

// Clears the dirty flags the caller set, however the caller leaves.
//
// Not a tidiness wrapper: the flags are `thread_local`, and a LEAKED `1` is worse
// than stale, because `delta_evaluate`'s seeding loop skips a node whose flag is
// already set -- so the next call omits that node from its dirty list and never
// recomputes it, silently, for the rest of the process. Only user code inside the
// walk can throw (a `CustomInvariant`, a `lambda_sum` functor), so this was
// unreachable in practice before #166 and is a documented surface after it;
// `tests/test_custom_invariant.cpp`'s throwing-delta case fails without this.
// The destructor cannot throw: every id in `list` already indexed `flags` on the
// way in.
struct DirtyFlagGuard {
    DirtyFlagGuard(std::vector<uint8_t>& f, const std::vector<int32_t>& l) : flags(f), list(l) {}
    DirtyFlagGuard(const DirtyFlagGuard&) = delete;
    DirtyFlagGuard& operator=(const DirtyFlagGuard&) = delete;
    DirtyFlagGuard(DirtyFlagGuard&&) = delete;
    DirtyFlagGuard& operator=(DirtyFlagGuard&&) = delete;
    ~DirtyFlagGuard() {
        for (const int32_t nid : list) {
            flags[nid] = 0;
        }
    }

    std::vector<uint8_t>& flags;
    const std::vector<int32_t>& list;
};

// The custom-aware evaluator: one dirty node, under the caller's DeltaMode.
// Built-in ops take the same `evaluate()` they always did; only a Custom node
// reads the mode. See `DeltaMode` and `CustomInvariant` for the protocol.
double evaluate_dirty_node(Model& model, int32_t nid, DeltaMode mode,
                           const int32_t* changed_var_ids, size_t count,
                           const std::vector<uint8_t>& dirty_flags,
                           std::vector<int32_t>& changed_scratch, const EditJournal* journal) {
    const ExprNode& node = model.nodes()[nid];
    if (node.op != NodeOp::Custom) {
        return evaluate(node, model);
    }
    const int32_t slot = node.lambda_func_id;
    // Unreachable: `Model::custom` appends the slot and writes this id with nothing
    // that can throw in between. Asserted rather than assumed, for the symmetry
    // `custom_of` in src/dag.cpp keeps -- the three probe accessors below index the
    // slot vector unchecked, so -1 would be a heap read one entry before it.
    assert(slot >= 0);
    if (mode == DeltaMode::Rollback && model.custom_probe_pending(slot)) {
        // The engine restores the node's VALUE; the invariant discards only its
        // own staged state. Its parents recompute from the restored value below,
        // in topological order, so the whole cone lands back where the probe
        // found it.
        model.custom_invariant(slot).rollback();
        return model.custom_end_probe(slot);
    }
    // Whether this call's positional edits describe the change since the
    // invariant's committed state (#172). They do unless a probe is still open:
    // the invariant then never heard the commit or rollback that probe owed, so
    // its committed state and the caller's "since" are no longer the same
    // assignment, and neither `changed` nor any edit list bridges the gap.
    bool positional_ok = true;
    if (model.custom_probe_pending(slot)) {
        positional_ok = false;
        // A `Commit` or `Probe` pass reached a slot that still owes a rollback,
        // which only an exception out of user code mid-probe can produce -- the two
        // bracketed probes have nothing between their legs that can throw. Drop the
        // stale stash: the assignment has moved on, so the value it holds is no
        // longer anything to roll back TO, and leaving it would let a later
        // `Rollback` restore a value from a different assignment. Defensive, with
        // no observable effect on any non-throwing path.
        (void)model.custom_end_probe(slot);
    }
    collect_changed_inputs(model, node, changed_var_ids, count, dirty_flags, changed_scratch);
    CustomInvariant& inv = model.custom_invariant(slot);
    const ConstSpan<int32_t> changed(changed_scratch.data(), changed_scratch.size());
    // The two-argument form reports no positional information for any input,
    // which is the honest answer on the stale-probe path above.
    const InvariantInputs inputs =
        positional_ok ? InvariantInputs(model, model.children(node), changed, journal)
                      : InvariantInputs(model, model.children(node));
    const double value = inv.delta(inputs, changed);
    if (mode == DeltaMode::Probe) {
        // Read BEFORE the caller writes `value`: this is still the value the
        // probe is to be rolled back to.
        model.custom_begin_probe(slot, model.node_values()[nid]);
    } else {
        inv.commit();
    }
    return value;
}

}  // namespace

// ---------------------------------------------------------------------------
// Incremental Sum (#177, #188)
// ---------------------------------------------------------------------------
//
// A committed FJ move changes one term of each row it touches, and re-summing a
// row costs its whole length: on swath3 a committed dirty Sum averaged 1315
// terms, of which about one had changed, and the re-sum was 36% of the run.
// So `commit_scalar_move` moves each incremental Sum (`ExprNode::kIncSum`, see
// `Model::classify_incremental_sums`) by its terms' changes instead: a variable
// term's from the old value the caller passes, a node term's from the value it
// held just before the walk rewrote it. The update is applied to the Sum's
// value in place, ahead of the Sum's own turn in topological order, where it is
// then taken as it stands.
//
// THE UPDATE IS FLOATING POINT, AND EACH ONE IS TESTED. `new - old` and
// `value + (new - old)` are each checked with TwoSum, which computes the
// rounding error of an addition exactly. Both errors 0 means the update is
// exact: it adds no drift, and on integral data within 2^53 every update is,
// so there the values are the re-sum's to the bit, exactly as #177's
// exact-only rule had them. Otherwise the Sum's `IncSumState::drift_bound`
// grows by the two errors' magnitudes, so that it bounds |value - the real sum
// of the stored terms| at all times:
//
//   - a checked re-sum (`checked_resum`) sets it to the sum of the magnitudes
//     of its own n - 1 rounding errors, again computed exactly by TwoSum --
//     which is 0 when the re-sum was exact and never more than the textbook
//     bound gamma_{n-1} * sum(|t_i|);
//   - each inexact update adds |error of new - old| + |error of the add|,
//     each at most half an ulp of its result;
//   - every addition of the bound itself is rounded up (`round_bound_up`), so
//     rounding the bound can never make it an underestimate.
//
// Drift is contained three ways. A Sum re-sums, checked, on what would be its
// kIncSumPeriod-th inexact update since the last re-sum. `reground_inc_sum`
// re-sums one Sum on demand -- FeasibilityJump's local-minimum gate uses it on
// a row whose verdict the bound cannot decide -- and `reground_drifted_sums`
// every drifted one, which FeasibilityJump does at the end of every batch, so
// drift never leaves a batch: the search, the pool, LNS and the inner solver
// see only re-summed values. A non-finite value -- the Sum's, or a term's old
// or new one -- makes the Sum re-sum at its turn, since `inf - inf` is no
// update.
//
// Every other walk keeps the values consistent with a drifted Sum:
//
//   - a plain `Commit` (`delta_evaluate`) re-sums every incremental Sum in
//     its cone and leaves it untracked -- a snap, but only of rows the caller
//     moved and so re-reads; the next FJ commit to update the Sum re-sums it
//     again, checked, first;
//   - a `Probe` stashes every incremental Sum in its cone, and, when it knows
//     the old value (`probe_scalar_move`), applies the same updates the commit
//     would, so it scores exactly what the commit would produce -- including 0
//     for an identity move on a drifted state;
//   - the matching `Rollback` writes the stash back rather than re-summing,
//     which would snap the committed value, and change the violated-row
//     bookkeeping under FeasibilityJump's feet.
//
// The cost model. The update costs two TwoSums and a store per changed term,
// paid by the term, plus a parents scan for each dirty node term. It wins when
// rows are long and a move touches one term of each, which is the MIP regime.
// It loses on short rows, where the checks are a larger share of a small
// re-sum, and where walks that are not FJ commits -- Novelty Jump's legs, the
// structural batch, the inner solver -- alternate with FJ commits on the same
// Sum: each such walk leaves it untracked, so the next commit pays a checked
// re-sum (a TwoSum per term) where the plain walk paid a plain one.
namespace {

thread_local IncrementalSumCounters inc_sum_counts;

// dirty_flags values beyond 1 ("in the cone"): an incremental Sum that must be
// re-summed at its turn rather than taken as its updates left it, and one a
// Rollback has already written back from the probe's stash.
constexpr uint8_t kDirtyResum = 2;
constexpr uint8_t kDirtyRestored = 3;

// The rounding error of `s = fl(a + b)`, exactly: a + b == s + err in real
// arithmetic (Knuth's TwoSum; valid for any finite a, b whose sum does not
// overflow). Fast2Sum would need |a| >= |b|, which neither use can promise.
inline double two_sum_err(double a, double b, double s) {
    const double b_virtual = s - a;
    const double a_virtual = s - b_virtual;
    return (a - a_virtual) + (b - b_virtual);
}

// `x` -- the rounded sum of non-negative values -- nudged up past what one or
// two roundings to nearest can have taken off it: fl(a + b) >= (a + b)(1 - u),
// and x(1 + 2^-51), rounded, is still >= x(1 + 2^-52) > x / (1 - u)^2.
inline double round_bound_up(double x) {
    return x + (x * 0x1p-51);
}

// The Sum case of `evaluate()`, verbatim -- same start, same child order, so
// the same bits -- plus the exact rounding error of every addition, whose
// magnitudes it sums into the drift bound. The n - 1 additions of that sum
// round too; (1 + n 2^-52) covers them.
double checked_resum(const Model& model, const ExprNode& node, IncSumState& st) {
    const std::vector<double>& values = model.node_values();
    const std::vector<Variable>& vars = model.variables();
    double s = 0.0;
    double err = 0.0;
    for (const ChildRef& c : model.children(node)) {
        const double v = c.is_var ? vars[c.id].value : values[c.id];
        const double t = s + v;
        err += std::fabs(two_sum_err(s, v, t));
        s = t;
    }
    st.tracked = 1;
    st.inexact = 0;
    st.drifting = 0;
    // A finite result means no partial sum overflowed, so every error is
    // finite. A non-finite one carries no drift to bound: an update onto it is
    // refused, and it re-sums at its next turn.
    st.drift_bound =
        std::isfinite(s) ? err + (err * (static_cast<double>(node.child_count) * 0x1p-52)) : 0.0;
    return s;
}

// The incremental-Sum rules for one walk. `Plain` is an unclosed model, whose
// state no full_evaluate has sized: every node goes to the plain evaluator.
class IncSumWalk {
public:
    IncSumWalk(Model& model, DeltaMode mode, bool push, std::vector<uint8_t>& dirty_flags)
        : model_(model),
          nodes_(model.nodes()),
          sums_(model.inc_sums()),
          dirty_(dirty_flags),
          rule_(rule_for(model, mode)),
          push_(push && (rule_ == Rule::Commit || rule_ == Rule::Probe)) {
        if (rule_ != Rule::Restore) {
            // Any walk but the Rollback it belongs to moves the assignment on,
            // so a stash still pending -- only an exception out of a probe can
            // leave one -- no longer describes anything to roll back to. A
            // Probe re-arms it in prepare().
            sums_.probe_stash.clear();
            sums_.probe_pending = false;
        }
    }

    // Before the walk: a Rollback writes its probe's stash back; a Probe
    // stashes its cone's Sums; then the changed variables' own moves are pushed
    // into the Sums they are terms of.
    void prepare(const std::vector<int32_t>& dirty_list, const int32_t* changed_var_ids,
                 size_t count, const double* old_values) {
        switch (rule_) {
            case Rule::Plain:
                return;
            case Rule::Restore:
                restore_stash();
                return;
            case Rule::Probe:
                stash(dirty_list);
                break;
            case Rule::Commit:
                if (!push_ && model_.has_custom_nodes()) {
                    // A walk without pushes re-reads a term only at its Sum's
                    // turn, so user code throwing before that turn would leave
                    // the Sum stale while its state says it is tracked; the next
                    // commit would update the stale value. Untracked, it
                    // re-sums instead. Defensive: after a throw the contract
                    // already asks for a full_evaluate, which resets them all.
                    for (const int32_t nid : dirty_list) {
                        if ((nodes_[nid].inc_sum_flags & ExprNode::kIncSum) != 0) {
                            sums_.slots[slot_of(nid)].tracked = 0;
                        }
                    }
                }
                break;
        }
        if (!push_) {
            return;
        }
        const std::vector<Variable>& vars = model_.variables();
        for (size_t ci = 0; ci < count; ++ci) {
            const int32_t v = changed_var_ids[ci];
            for (const int32_t dep_id : model_.dependents(v)) {
                if ((nodes_[dep_id].inc_sum_flags & ExprNode::kIncSum) != 0) {
                    push(dep_id, vars[v].value, old_values[ci]);
                }
            }
        }
    }

    // One dirty node's new value. An incremental Sum is taken as its updates
    // left it, or re-summed; a dirty term node pushes its change into its
    // incremental parents.
    template <typename EvalOther>
    double eval(int32_t nid, EvalOther&& eval_other) {
        const uint8_t flags = nodes_[nid].inc_sum_flags;
        if (rule_ == Rule::Plain || flags == 0) {
            return eval_other(nid);
        }
        if ((flags & ExprNode::kIncSum) != 0) {
            return eval_sum(nid);
        }
        if (!push_) {
            return eval_other(nid);  // kFeedsIncSum: re-read at its Sum's turn
        }
        const double old_v = model_.node_values()[nid];
        const double new_v = eval_other(nid);
        for (const int32_t parent_id : model_.parents(nid)) {
            if ((nodes_[parent_id].inc_sum_flags & ExprNode::kIncSum) != 0) {
                push(parent_id, new_v, old_v);
            }
        }
        return new_v;
    }

private:
    enum class Rule : uint8_t { Plain, Commit, Probe, Restore };

    static Rule rule_for(const Model& model, DeltaMode mode) {
        if (model.inc_sums().slots.size() != model.inc_sum_nodes().size()) {
            return Rule::Plain;
        }
        if (mode == DeltaMode::Probe) {
            return Rule::Probe;
        }
        // A Rollback with no probe pending is a Commit, as it is for a custom
        // node (see DeltaMode).
        if (mode == DeltaMode::Rollback && model.inc_sums().probe_pending) {
            return Rule::Restore;
        }
        return Rule::Commit;
    }

    [[nodiscard]] int32_t slot_of(int32_t nid) const { return nodes_[nid].lambda_func_id; }

    void stash(const std::vector<int32_t>& dirty_list) {
        const std::vector<double>& values = model_.node_values();
        for (const int32_t nid : dirty_list) {
            if ((nodes_[nid].inc_sum_flags & ExprNode::kIncSum) != 0) {
                sums_.probe_stash.emplace_back(nid, values[nid]);
            }
        }
        sums_.probe_pending = true;
    }

    // Only a node in this walk's cone is flagged: the guard clears the flags
    // of the cone and nothing else, so flagging a stranger would leak.
    void restore_stash() {
        for (const auto& [nid, value] : sums_.probe_stash) {
            model_.set_node_value_unchecked(nid, value);
            if (dirty_[nid] != 0) {
                dirty_[nid] = kDirtyRestored;
            }
        }
        sums_.probe_stash.clear();
        sums_.probe_pending = false;
    }

    // One term of the incremental Sum `p` moved from `old_v` to `new_v`.
    void push(int32_t p, double new_v, double old_v) {
        uint8_t& flag = dirty_[p];
        if (flag == kDirtyResum) {
            return;  // re-summed at its turn whatever else arrives
        }
        const double cur = model_.node_values()[p];
        const double d = new_v - old_v;
        const double s = cur + d;
        if (rule_ == Rule::Probe) {
            // No drift state moves under a probe: it is rolled back.
            if (!std::isfinite(s) || !std::isfinite(d)) {
                flag = kDirtyResum;
                return;
            }
            ++inc_sum_counts.probe_pushes;
            model_.set_node_value_unchecked(p, s);
            return;
        }
        IncSumState& st = sums_.slots[slot_of(p)];
        if (st.tracked == 0 || !std::isfinite(s) || !std::isfinite(d)) {
            force_resum(flag, st);
            return;
        }
        const double err_d = two_sum_err(new_v, -old_v, d);
        const double err_s = two_sum_err(cur, d, s);
        if (err_d != 0.0 || err_s != 0.0) {
            if (kIncSumPeriod != 0 && st.inexact + 1 >= kIncSumPeriod) {
                force_resum(flag, st);
                return;
            }
            st.drift_bound = round_bound_up(st.drift_bound + (std::fabs(err_d) + std::fabs(err_s)));
            ++st.inexact;
            ++sums_.inexact_updates;
            ++inc_sum_counts.inexact;
        }
        // Exact or not, the value is no longer a re-sum of the current terms
        // unless it is the exact sum -- and a zero bound says it is.
        if (st.drift_bound != 0.0) {
            st.drifting = 1;
            if (st.listed == 0) {
                st.listed = 1;
                sums_.drifted.push_back(slot_of(p));
            }
        }
        model_.set_node_value_unchecked(p, s);
    }

    // Untracked as well as flagged: were user code to throw before the Sum's
    // turn, the flag would be cleared on the way out and the Sum -- which has
    // not taken this term's change -- must not be updated again from there.
    static void force_resum(uint8_t& flag, IncSumState& st) {
        flag = kDirtyResum;
        st.tracked = 0;
    }

    double eval_sum(int32_t nid) {
        const ExprNode& node = nodes_[nid];
        const uint8_t flag = dirty_[nid];
        switch (rule_) {
            case Rule::Plain:
                return evaluate(node, model_);
            case Rule::Restore:
                if (flag == kDirtyRestored) {
                    return model_.node_values()[nid];
                }
                break;  // not in the probe's cone: the caller broke the pairing
            case Rule::Probe:
                if (push_ && flag != kDirtyResum) {
                    return model_.node_values()[nid];
                }
                return evaluate(node, model_);  // no drift state moves under a probe
            case Rule::Commit: {
                IncSumState& st = sums_.slots[slot_of(nid)];
                if (push_ && flag != kDirtyResum && st.tracked != 0) {
                    ++inc_sum_counts.incremental;
                    return model_.node_values()[nid];
                }
                ++inc_sum_counts.resummed;
                if (push_) {
                    return checked_resum(model_, node, st);
                }
                // A walk without pushes -- Novelty Jump's legs, the inner
                // solver, the structural batch -- re-sums plainly and leaves
                // the Sum untracked, as full_evaluate does: the rounding check
                // is paid only by the FJ commit that next updates the Sum, and
                // never by a regime that re-sums on every walk.
                st.tracked = 0;
                st.drifting = 0;
                return evaluate(node, model_);
            }
        }
        ++inc_sum_counts.resummed;
        return checked_resum(model_, node, sums_.slots[slot_of(nid)]);
    }

    Model& model_;
    const std::vector<ExprNode>& nodes_;
    IncSums& sums_;
    std::vector<uint8_t>& dirty_;
    Rule rule_;
    bool push_;
};

// Marks the cone above `changed_var_ids` -- their dependents, then every
// ancestor -- in `dirty_flags`, listing each node once in `dirty_list`.
void collect_dirty_cone(const Model& model, const int32_t* changed_var_ids, size_t count,
                        std::vector<uint8_t>& dirty_flags, std::vector<int32_t>& dirty_list) {
    // Seed dirty set from changed variables' dependents
    for (size_t ci = 0; ci < count; ++ci) {
        for (const int32_t dep_id : model.dependents(changed_var_ids[ci])) {
            if (dirty_flags[dep_id] == 0) {
                dirty_flags[dep_id] = 1;
                dirty_list.push_back(dep_id);
            }
        }
    }

    // BFS upward through parents
    for (size_t i = 0; i < dirty_list.size(); ++i) {
        int32_t nid = dirty_list[i];
        for (const int32_t parent_id : model.parents(nid)) {
            if (dirty_flags[parent_id] == 0) {
                dirty_flags[parent_id] = 1;
                dirty_list.push_back(parent_id);
            }
        }
    }
}

double objective_value(const Model& model) {
    if (model.objective_id() >= 0) {
        return model.node_values()[model.objective_id()];
    }
    return 0.0;
}

// The one walk behind every entry point. `old_values`, when non-null, holds the
// previous value of each of `changed_var_ids` and switches the term updates on;
// null re-sums every dirty incremental Sum, as `delta_evaluate` always did.
double delta_walk(Model& model, const int32_t* changed_var_ids, size_t count, DeltaMode mode,
                  const EditJournal* journal, const double* old_values) {
    const EvaluationGuard guard("delta_evaluate");
    const size_t num_nodes = model.num_nodes();

    // Flat dirty flags + dirty list for O(dirty) cleanup
    // Use thread_local to avoid reallocation across calls
    thread_local std::vector<uint8_t> dirty_flags;
    thread_local std::vector<int32_t> dirty_list;

    if (dirty_flags.size() < num_nodes) {
        dirty_flags.resize(num_nodes, 0);
    }
    dirty_list.clear();

    // Before the early return: it is what drops a stale probe stash.
    IncSumWalk sums(model, mode, old_values != nullptr, dirty_flags);
    if (count == 0) {
        return objective_value(model);
    }

    // Armed BEFORE the seeding loop, so it covers every flag this call sets --
    // including the ones set before an exception out of the walk below.
    const DirtyFlagGuard flag_guard(dirty_flags, dirty_list);
    collect_dirty_cone(model, changed_var_ids, count, dirty_flags, dirty_list);
    sums.prepare(dirty_list, changed_var_ids, count, old_values);

    // One test per CALL, not per node: a model with no custom node takes the
    // pre-#166 loop verbatim, which is what keeps criterion 4's bit-identical
    // trajectories bit-identical (#166).
    if (model.has_custom_nodes()) {
        thread_local std::vector<int32_t> changed_inputs;
        evaluate_dirty_in_topo_order(model, dirty_list, dirty_flags, num_nodes, [&](int32_t nid) {
            return sums.eval(nid, [&](int32_t id) {
                return evaluate_dirty_node(model, id, mode, changed_var_ids, count, dirty_flags,
                                           changed_inputs, journal);
            });
        });
    } else {
        evaluate_dirty_in_topo_order(model, dirty_list, dirty_flags, num_nodes, [&](int32_t nid) {
            return sums.eval(nid,
                             [&model](int32_t id) { return evaluate(model.nodes()[id], model); });
        });
    }

    // The flags are cleared by `flag_guard` on the way out, which is also what
    // covers a throw from user code inside the walk.
    return objective_value(model);
}

}  // namespace

IncrementalSumCounters& incremental_sum_counters() noexcept {
    return inc_sum_counts;
}

double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count, DeltaMode mode,
                      const EditJournal* journal) {
    return delta_walk(model, changed_var_ids, count, mode, journal, nullptr);
}

double commit_scalar_move(Model& model, int32_t var_id, double old_value) {
    return delta_walk(model, &var_id, 1, DeltaMode::Commit, nullptr, &old_value);
}

double probe_scalar_move(Model& model, int32_t var_id, double old_value) {
    return delta_walk(model, &var_id, 1, DeltaMode::Probe, nullptr, &old_value);
}

void reground_inc_sum(Model& model, int32_t slot) {
    const EvaluationGuard guard("reground_inc_sum");
    IncSums& sums = model.inc_sums();
    // A write between a probe and its rollback would be undone by the stash.
    sums.probe_stash.clear();
    sums.probe_pending = false;
    const int32_t nid = model.inc_sum_nodes()[static_cast<size_t>(slot)];
    const std::vector<ExprNode>& nodes = model.nodes();
    model.set_node_value_unchecked(
        nid, checked_resum(model, nodes[nid], sums.slots[static_cast<size_t>(slot)]));
    // Its readers are top-level comparisons (classify_incremental_sums), so
    // they are all there is above it.
    for (const int32_t p : model.parents(nid)) {
        model.set_node_value_unchecked(p, evaluate(nodes[p], model));
    }
    ++inc_sum_counts.regrounded;
}

void reground_drifted_sums(Model& model, std::vector<int32_t>& regrounded) {
    IncSums& sums = model.inc_sums();
    for (const int32_t slot : sums.drifted) {
        IncSumState& st = sums.slots[static_cast<size_t>(slot)];
        st.listed = 0;
        if (st.drifting != 0) {
            reground_inc_sum(model, slot);
            regrounded.push_back(slot);
        }
    }
    sums.drifted.clear();
}

namespace {

// Scratch for the reverse sweep, shared by the three entry points below. Every
// buffer is left all-zero / empty between calls; `AdjointScratchGuard` restores
// that on the way out, including when a `CustomInvariant::partial` throws.
//
// `busy` refuses re-entry, the same way `EvaluationGuard` does for evaluation: a
// `CustomInvariant::partial` that differentiated a sub-model through one of the
// entry points would push onto `cone` while the outer sweep iterates it (UB),
// and its guard would then clear the outer sweep's bookkeeping and leak its
// adjoints into every later call on the thread. One thread_local test and store
// per call.
//
// `adjoint` is flat: [0, num_nodes) for nodes, [num_nodes, num_nodes + num_vars)
// for variables. It only ever grows, so a thread that sees a larger model keeps
// the larger buffer.
struct AdjointScratch {
    std::vector<double> adjoint;
    std::vector<int32_t> written;  // adjoint entries touched, for O(touched) cleanup
    std::vector<uint8_t> in_cone;  // node id -> collected into `cone`
    std::vector<int32_t> cone;     // nodes reachable from expr_id through children
    bool busy = false;             // a sweep is in progress on this thread
};

AdjointScratch& adjoint_scratch() {
    thread_local AdjointScratch scratch;
    return scratch;
}

class AdjointScratchGuard {
public:
    AdjointScratchGuard(AdjointScratch& scratch, const char* entry) : scratch_(scratch) {
        if (scratch_.busy) {
            // Thrown from the constructor, so the destructor -- which would clear
            // the OUTER sweep's state -- never runs for the refused call.
            throw std::logic_error(std::string(entry) +
                                   ": re-entered from inside a reverse-mode AD sweep. A "
                                   "CustomInvariant's partial must not call compute_partial, "
                                   "compute_all_partials or compute_partials_sparse.");
        }
        scratch_.busy = true;
    }
    AdjointScratchGuard(const AdjointScratchGuard&) = delete;
    AdjointScratchGuard& operator=(const AdjointScratchGuard&) = delete;
    AdjointScratchGuard(AdjointScratchGuard&&) = delete;
    AdjointScratchGuard& operator=(AdjointScratchGuard&&) = delete;
    ~AdjointScratchGuard() {
        for (const int32_t idx : scratch_.written) {
            scratch_.adjoint[idx] = 0.0;
        }
        scratch_.written.clear();
        for (const int32_t nid : scratch_.cone) {
            scratch_.in_cone[nid] = 0;
        }
        scratch_.cone.clear();
        scratch_.busy = false;
    }

private:
    AdjointScratch& scratch_;
};

// Push the adjoint of one node onto its children -- the body of the sweep,
// unchanged from the full-order walk it replaced.
void propagate_adjoint(const Model& model, int32_t nid, size_t num_nodes, AdjointScratch& s) {
    const double adj = s.adjoint[nid];
    const auto& nd = model.node(nid);
    const ConstSpan<ChildRef> children = model.children(nd);
    for (int i = 0; i < static_cast<int>(children.size()); ++i) {
        const double ld = local_derivative(nd, i, model);
        const ChildRef& child = children[i];
        const int32_t key = child.is_var ? static_cast<int32_t>(num_nodes) + child.id : child.id;
        if (s.adjoint[key] == 0.0) {
            s.written.push_back(key);
        }
        s.adjoint[key] += adj * ld;
    }
}

// Reverse-mode sweep from `expr_id`: on return `s.adjoint` holds d expr / d x
// for every node and variable x, and `s.written` lists every entry touched.
//
// Only the cone of `expr_id` -- the nodes reachable from it through children --
// can ever carry a nonzero adjoint, so the sweep visits the cone in reverse
// topological order rather than all of `topo_order()`. It keeps the full walk's
// `adjoint == 0.0` skip, so it performs exactly the same floating-point
// operations in exactly the same order: every node outside the cone had adjoint
// 0 in the full walk too, and the cone sorted by `topo_position` is the full
// order restricted to the cone. The results, and every search trajectory built
// on them, are bit-identical to the full walk.
//
// Cost: collecting the cone is O(cone + its edges), sorting it O(c log c) with
// two scattered `topo_position` loads per comparison, against the full walk's
// O(|nodes|) adjoint tests (a sequential read of `topo_order`, gathering the
// adjoint by id). That wins by orders of magnitude where the callers live -- a
// linear MIP row of a few dozen terms against the 765k nodes of MIPfeas'
// uccase12, where the full walk was 87% of CPU in a gprof profile at 175e617
// (~0.9 ms per call, from that profile's self time over its call count). It
// loses as the cone approaches the whole DAG (an objective over every node, a
// dense continuous model), where the sort does ~log2(c) scattered comparisons
// per node the scan tests once. So a cost model like
// `evaluate_dirty_in_topo_order`'s picks the route -- with log2|nodes| standing
// in for log2(c), since c is unknown while collecting: collect while
// c * (log2|nodes| + 1) <= |nodes|, and on crossing it abandon the cone and scan
// the full order exactly as before. The abandoned collection is at most
// ~|nodes| / log2|nodes| nodes, all of them reachable from `expr_id` and so
// nodes and edges the fallback walk visits anyway: the worst case is the old
// cost plus a comparable amount.
//
// A model whose `topo_order` does not cover every node -- one not yet closed
// has none -- has no position to sort by, so it takes the full-order walk as it
// always did (over an empty order: all zeros).
void reverse_sweep(const Model& model, int32_t expr_id, AdjointScratch& s) {
    const size_t num_nodes = model.num_nodes();
    const size_t total_size = num_nodes + model.num_vars();
    if (s.adjoint.size() < total_size) {
        s.adjoint.resize(total_size, 0.0);
    }
    if (s.in_cone.size() < num_nodes) {
        s.in_cone.resize(num_nodes, 0);
    }

    // Each push precedes the write it records, so a `bad_alloc` in the push
    // leaves nothing the guard does not know to undo.
    s.written.push_back(expr_id);
    s.adjoint[expr_id] = 1.0;

    size_t log2_nodes = 0;
    while ((size_t{1} << (log2_nodes + 1)) <= num_nodes) {
        ++log2_nodes;
    }
    const size_t max_sorted_cone = num_nodes / (log2_nodes + 1);

    // Breadth-first over node children, using `cone` itself as the queue.
    bool cone_fits = model.topo_order().size() == num_nodes;
    s.cone.push_back(expr_id);
    s.in_cone[expr_id] = 1;
    for (size_t head = 0; head < s.cone.size() && cone_fits; ++head) {
        for (const ChildRef& child : model.children(model.node(s.cone[head]))) {
            if (child.is_var || s.in_cone[child.id] != 0) {
                continue;
            }
            s.cone.push_back(child.id);
            s.in_cone[child.id] = 1;
            if (s.cone.size() > max_sorted_cone) {
                cone_fits = false;
                break;
            }
        }
    }

    if (cone_fits) {
        // Descending position = reverse topological order.
        std::sort(s.cone.begin(), s.cone.end(), [&model](int32_t a, int32_t b) {
            return model.topo_position(a) > model.topo_position(b);
        });
        for (const int32_t nid : s.cone) {
            if (s.adjoint[nid] != 0.0) {
                propagate_adjoint(model, nid, num_nodes, s);
            }
        }
        return;
    }
    const auto& order = model.topo_order();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        if (s.adjoint[*it] != 0.0) {
            propagate_adjoint(model, *it, num_nodes, s);
        }
    }
}

}  // namespace

double compute_partial(const Model& model, int32_t expr_id, int32_t var_id) {
    AdjointScratch& s = adjoint_scratch();
    const AdjointScratchGuard guard(s, "compute_partial");
    reverse_sweep(model, expr_id, s);
    if (var_id < 0 || static_cast<size_t>(var_id) >= model.num_vars()) {
        return 0.0;
    }
    return s.adjoint[model.num_nodes() + static_cast<size_t>(var_id)];
}

std::vector<double> compute_all_partials(const Model& model, int32_t expr_id) {
    AdjointScratch& s = adjoint_scratch();
    const AdjointScratchGuard guard(s, "compute_all_partials");
    reverse_sweep(model, expr_id, s);
    const size_t num_nodes = model.num_nodes();
    const auto first = s.adjoint.begin() + static_cast<std::ptrdiff_t>(num_nodes);
    return {first, first + static_cast<std::ptrdiff_t>(model.num_vars())};
}

void compute_partials_sparse(const Model& model, int32_t expr_id,
                             std::vector<std::pair<int32_t, double>>& out) {
    out.clear();
    AdjointScratch& s = adjoint_scratch();
    const AdjointScratchGuard guard(s, "compute_partials_sparse");
    reverse_sweep(model, expr_id, s);
    const auto num_nodes = static_cast<int32_t>(model.num_nodes());
    // `written` can list a variable twice -- its adjoint cancelled to 0.0 and was
    // touched again -- so each emitted entry is zeroed at once, which makes the
    // second listing read 0.0 and skip. The guard zeroes the rest.
    for (const int32_t key : s.written) {
        if (key >= num_nodes && s.adjoint[key] != 0.0) {
            out.emplace_back(key - num_nodes, s.adjoint[key]);
            s.adjoint[key] = 0.0;
        }
    }
}

}  // namespace cbls
