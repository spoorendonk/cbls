#include "cbls/dag_ops.h"

#include "cbls/model.h"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// The incremental Sum's drift bound (#188) rests on TwoSum computing each
// addition's rounding error exactly, which needs IEEE double arithmetic
// evaluated as written. -ffast-math (or -ffinite-math-only) lets the compiler
// fold `(a - (s - (s - a))) + ...` to 0, which would zero every error, make
// every inexact update read as exact, and silently switch the phantom-row gate
// off. Refused here rather than trusted to the build.
#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "src/dag_ops.cpp needs IEEE arithmetic: do not build it with -ffast-math"
#endif
static_assert(FLT_EVAL_METHOD == 0,
              "TwoSum needs double operations evaluated in double (FLT_EVAL_METHOD 0)");

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
// A `lambda_sum` functor is user code inside the evaluation walk. Re-entering
// from there is not merely unsupported, it is SILENT PERMANENT CORRUPTION of
// this thread: the nested call does `dirty_list.clear()` on the same
// `thread_local` vector that the outer call's `DirtyFlagGuard` holds a reference
// to, so the guard then clears the INNER call's ids and leaks the outer call's
// flags -- and a leaked flag makes `delta_evaluate`'s seeding loop skip that node
// for the life of the process (see `DirtyFlagGuard`). The symptom would be a node
// that quietly stops updating, long after and nowhere near the cause. So it is
// refused rather than documented.
//
// One `thread_local` test and one store per call, on both entry points. It
// changes no value and draws no random number, so trajectories are unaffected.
thread_local bool in_evaluation = false;

// RAII, so that an exception out of user code inside the walk clears the flag
// on the way out
// instead of poisoning every later call on this thread.
class EvaluationGuard {
public:
    explicit EvaluationGuard(const char* entry) {
        if (in_evaluation) {
            // Not an assert: this is reachable from user code in a Release build,
            // and the silent wrong answer above is the thing being prevented.
            throw std::logic_error(std::string(entry) +
                                   ": re-entered from inside an evaluation. A lambda functor "
                                   "must not call full_evaluate or delta_evaluate, on this "
                                   "model or any other.");
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
    // The probe's undo log (#270) is dropped: every value is about to be rewritten.
    model.probe_undo().pending = false;
    // Re-sums without the rounding check, so no incremental Sum is tracked
    // afterwards and none carries drift (#188); its first commit re-sums it,
    // checked. Also what sizes the state: every close and every structural
    // rebuild ends in a full pass.
    IncSums& sums = model.inc_sums();
    sums.slots.assign(model.inc_sum_nodes().size(), IncSumState{});
    sums.drifted.clear();
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
// plain instantiation inlines the `evaluate()` call.
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

// Clears the dirty flags the caller set, however the caller leaves.
//
// Not a tidiness wrapper: the flags are `thread_local`, and a LEAKED `1` is worse
// than stale, because `delta_evaluate`'s seeding loop skips a node whose flag is
// already set -- so the next call omits that node from its dirty list and never
// recomputes it, silently, for the rest of the process. Only user code inside the
// walk can throw (a `lambda_sum` functor). The destructor cannot throw: every id
// in `list` already indexed `flags` on the way in. `List` is the collected
// `std::vector`, read at destruction so that it covers every id pushed after the
// guard was armed, or a cached cone's span (#271), whose flags are all set
// before anything can throw.
template <typename List>
struct DirtyFlagGuard {
    DirtyFlagGuard(std::vector<uint8_t>& f, const List& l) : flags(f), list(l) {}
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
    const List& list;
};

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
//   - a `Probe`, when it knows the old value (`probe_scalar_move`), applies
//     the commit's update to the committed value, so an identity move on a
//     drifted state scores exactly 0. Where the commit would re-sum instead
//     (an untracked Sum, the period reached, a non-finite value) the probe
//     still updates, and its score differs from the commit's by at most the
//     Sum's drift bound plus the re-sum's rounding. That is a scoring
//     difference at the ulp level; FJ makes no feasibility or bump decision
//     on a probe's value;
//   - the matching `Rollback` writes the probe's undo log back (`ProbeUndo`,
//     #270) rather than re-summing, which would snap the committed value, and
//     change the violated-row bookkeeping under FeasibilityJump's feet.
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
thread_local ConeCacheCounters cone_cache_counts;

// The dirty_flags bit beyond 1 ("in the cone"), on an incremental Sum: re-sum
// it at its turn rather than take it as its updates left it.
constexpr uint8_t kDirtyResum = 2;

// The rounding error of `s = fl(a + b)`, exactly: a + b == s + err in real
// arithmetic (Knuth's TwoSum; valid for any finite a, b whose sum does not
// overflow). Fast2Sum would need |a| >= |b|, which neither use can promise.
inline double two_sum_err(double a, double b, double s) {
    const double b_virtual = s - a;
    const double a_virtual = s - b_virtual;
    return (a - a_virtual) + (b - b_virtual);
}

// `x` -- the result of at most two rounded additions of non-negative values,
// so the real y it stands for is at most x(1 + u)^2, u = 2^-53 -- nudged up past
// y. x * 2^-51 is exact (a power of two), and the one rounding of the add gives
// fl(x + x 2^-51) >= x(1 + 2^-51) / (1 + u), which is >= x(1 + u)^2 because
// (1 + 2^-51) = 1 + 4u >= (1 + u)^3. (Subnormal x is ignored: its absolute
// error, below 2^-1074, is nowhere near any tolerance the bound is read against.)
inline double round_bound_up(double x) {
    return x + (x * 0x1p-51);
}

// The Sum case of `evaluate()`, verbatim -- same start, same child order, so
// the same bits -- plus the exact rounding error of every addition, whose
// magnitudes it sums into the drift bound. The n - 1 additions of that sum
// round too, so the real sum of the magnitudes is at most err (1 + u)^(n-1),
// and the multiply and add below round once each: err + err n 2^-52, rounded,
// is at least err (1 + 2nu)(1 - u)^2, which covers (1 + u)^(n+1) for any n
// below 2^40.
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

// The incremental-Sum rules for one walk. `Plain` is a model with no incremental
// Sum, or an unclosed one whose state no full_evaluate has sized: every node goes
// to the plain evaluator.
class IncSumWalk {
public:
    IncSumWalk(Model& model, DeltaMode mode, bool push, std::vector<uint8_t>& dirty_flags)
        : model_(model),
          nodes_(model.nodes()),
          sums_(model.inc_sums()),
          dirty_(dirty_flags),
          rule_(rule_for(model, mode)),
          push_(push && (rule_ == Rule::Commit || rule_ == Rule::Probe)) {}

    // Before the walk: a plain commit untracks the Sums in its cone; then the
    // changed variables' own moves are pushed into the Sums they are terms of.
    //
    // This and `eval_flagged` are kept out of line
    // deliberately. Inlined, they grew `delta_walk` and made the probe-bound
    // MINLPLib walks 5-14% slower per iteration at a fixed iteration count
    // (eg_all_s 7.09 s against 6.20 s for the pre-#188 engine; maxmin, nvs01,
    // nvs14 likewise), on the same trajectory. Out of line, every one of them
    // timed at or below the pre-#188 engine (#188, review round 2). The cost is
    // one call per walk, and one per flagged node, which is a Sum or its term.
    [[gnu::noinline]] void prepare(ConstSpan<int32_t> dirty_list, const int32_t* changed_var_ids,
                                   size_t count, const double* old_values) {
        switch (rule_) {
            case Rule::Plain:
                return;
            case Rule::Probe:
                // Nothing to save: the walk's undo log holds every value the
                // probe is about to write (#270).
                break;
            case Rule::Commit:
                if (!push_) {
                    // A walk without pushes re-reads a term only at its Sum's
                    // turn, so anything throwing before that turn -- a plain
                    // or extra lambda (Python binds them as raw callables), or
                    // anything else -- would leave the Sum stale
                    // while its state says it is tracked; the next commit would
                    // update the stale value. Untracked, it re-sums instead.
                    // Defensive: after a throw the contract already asks for a
                    // full_evaluate, which resets them all.
                    //
                    // Unconditional (#192): keying it on the node kinds that run
                    // user code missed the plain lambdas once already. It changes
                    // no value -- every Sum in the cone is untracked at its own
                    // turn anyway (eval_sum, Rule::Commit without pushes), this
                    // only does it before the walk can throw. The cost is one
                    // pass over the dirty list per plain commit, a walk that
                    // already visits every node on that list; FJ's commits push
                    // and never take it.
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
    //
    // Split in two so that this, the part every dirty node runs, stays small
    // enough to inline into the walk's loop: on a probe-bound model most dirty
    // nodes are neither a Sum nor a term of one, and a call per node there cost
    // nvs14 ~8% of its run (gprof, #188).
    template <typename EvalOther>
    double eval(int32_t nid, EvalOther&& eval_other) {
        const uint8_t flags = nodes_[nid].inc_sum_flags;
        if (flags == 0 || rule_ == Rule::Plain) {
            return eval_other(nid);
        }
        return eval_flagged(nid, flags, eval_other);
    }

private:
    template <typename EvalOther>
    [[gnu::noinline]] double eval_flagged(int32_t nid, uint8_t flags, EvalOther& eval_other) {
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

    enum class Rule : uint8_t { Plain, Commit, Probe };

    static Rule rule_for(const Model& model, DeltaMode mode) {
        // A model with no incremental Sum -- most MINLPLib instances -- takes
        // the plain walk.
        if (model.inc_sum_nodes().empty() ||
            model.inc_sums().slots.size() != model.inc_sum_nodes().size()) {
            return Rule::Plain;
        }
        if (mode == DeltaMode::Probe) {
            return Rule::Probe;
        }
        // A Rollback reaches a walk only with no probe of its own pending,
        // and is then a Commit (see DeltaMode).
        return Rule::Commit;
    }

    [[nodiscard]] int32_t slot_of(int32_t nid) const { return nodes_[nid].lambda_func_id; }

    // One term of the incremental Sum `p` moved from `old_v` to `new_v`.
    void push(int32_t p, double new_v, double old_v) {
        uint8_t& flag = dirty_[p];
        if ((flag & kDirtyResum) != 0) {
            return;  // re-summed at its turn whatever else arrives
        }
        const double cur = model_.node_values()[p];
        const double d = new_v - old_v;
        const double s = cur + d;
        if (rule_ == Rule::Probe) {
            // No drift state moves under a probe: it is rolled back.
            if (!std::isfinite(s) || !std::isfinite(d)) {
                flag |= kDirtyResum;
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
    // (A throw mid-commit can also leave other Sums pushed but their readers
    // not yet re-evaluated; a full_evaluate after it resets every Sum.)
    static void force_resum(uint8_t& flag, IncSumState& st) {
        flag |= kDirtyResum;
        st.tracked = 0;
    }

    double eval_sum(int32_t nid) {
        const ExprNode& node = nodes_[nid];
        uint8_t& flag = dirty_[nid];
        switch (rule_) {
            case Rule::Plain:
                return evaluate(node, model_);
            case Rule::Probe:
                if (push_ && (flag & kDirtyResum) == 0) {
                    return model_.node_values()[nid];
                }
                return evaluate(node, model_);  // no drift state moves under a probe
            case Rule::Commit: {
                IncSumState& st = sums_.slots[slot_of(nid)];
                if (push_ && (flag & kDirtyResum) == 0 && st.tracked != 0) {
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

// A Probe's undo log (#270): its cone, with every value as it stood before the
// probe. Taken after the cone is collected and before anything in it is
// written -- an incremental Sum's term pushes included -- so the log holds the
// committed values, drift and all. O(cone) loads and stores, against the
// O(cone log cone) collect-sort-evaluate walk the Rollback no longer repeats.
void arm_probe_undo(Model& model, ConstSpan<int32_t> dirty_list, const int32_t* changed_var_ids,
                    size_t count) {
    ProbeUndo& undo = model.probe_undo();
    const std::vector<double>& values = model.node_values();
    undo.values.clear();
    for (const int32_t nid : dirty_list) {
        undo.values.emplace_back(nid, values[nid]);
    }
    undo.vars.assign(changed_var_ids, changed_var_ids + count);
    undo.pending = true;
}

// The Rollback of a pending probe (#270), with no cone collection, sort or
// evaluation: the log's values are written back, which is exact by
// construction, where re-evaluating restored them only as far as every node's
// value was what `evaluate()` gives -- not an incremental Sum carrying drift
// (#188), which needed a stash of its own.
void roll_back_probe(Model& model, const ProbeUndo& undo) {
    for (const auto& [nid, value] : undo.values) {
        model.set_node_value_unchecked(nid, value);
    }
}

// Whether a Rollback over `changed_var_ids` is the one the pending probe owes:
// the same set in the same order, which is how every caller passes it.
bool undo_matches(const ProbeUndo& undo, const int32_t* changed_var_ids, size_t count) {
    return undo.vars.size() == count &&
           std::equal(undo.vars.begin(), undo.vars.end(), changed_var_ids);
}

ConstSpan<int32_t> span_of(const std::vector<int32_t>& list) {
    return {list.data(), list.size()};
}

// Evaluates a walk's dirty nodes under the incremental-Sum rules, in the order
// `walk` visits them: `walk(eval_node)` writes `eval_node(nid)` to each node.
template <typename Walk>
void evaluate_with_sums(Model& model, IncSumWalk& sums, Walk walk) {
    walk([&](int32_t nid) {
        return sums.eval(nid, [&model](int32_t id) { return evaluate(model.nodes()[id], model); });
    });
}

// A one-variable walk over its cached cone (#271): the cone `freeze()` already
// collected and sorted, so no BFS and no sort. The same nodes in the same
// topological order `evaluate_dirty_in_topo_order` reaches by either route,
// so every value is the one the collecting walk computes.
//
// A model with no incremental Sum sets no dirty flag at all: only the scan
// route and the incremental-Sum pushes read them. The others set every flag of
// the cone up front, under a guard armed first, as the collecting walk's is.
double walk_cached_cone(Model& model, const int32_t* var_id, DeltaMode mode,
                        const double* old_values, std::vector<uint8_t>& dirty_flags) {
    const ConstSpan<int32_t> cone = model.cached_cone(*var_id);
    if (model.inc_sum_nodes().empty()) {
        if (mode == DeltaMode::Probe) {
            arm_probe_undo(model, cone, var_id, 1);
        }
        for (const int32_t nid : cone) {
            model.set_node_value_unchecked(nid, evaluate(model.nodes()[nid], model));
        }
        return objective_value(model);
    }
    IncSumWalk sums(model, mode, old_values != nullptr, dirty_flags);
    const DirtyFlagGuard<ConstSpan<int32_t>> flag_guard(dirty_flags, cone);
    for (const int32_t nid : cone) {
        dirty_flags[nid] = 1;
    }
    if (mode == DeltaMode::Probe) {
        arm_probe_undo(model, cone, var_id, 1);  // before `prepare`'s pushes
    }
    sums.prepare(cone, var_id, 1, old_values);
    evaluate_with_sums(model, sums, [&](auto eval_node) {
        for (const int32_t nid : cone) {
            model.set_node_value_unchecked(nid, eval_node(nid));
        }
    });
    return objective_value(model);
}

// The one walk behind every entry point. `old_values`, when non-null, holds the
// previous value of each of `changed_var_ids` and switches the term updates on;
// null re-sums every dirty incremental Sum, as `delta_evaluate` always did.
double delta_walk(Model& model, const int32_t* changed_var_ids, size_t count, DeltaMode mode,
                  const double* old_values) {
    const EvaluationGuard guard("delta_evaluate");
    // Any walk moves the assignment on from the pending probe's, so the log is
    // dropped here, whatever the mode; only the Rollback it belongs to spends it.
    ProbeUndo& undo = model.probe_undo();
    if (undo.pending) {
        undo.pending = false;
        if (mode == DeltaMode::Rollback) {
            if (undo_matches(undo, changed_var_ids, count)) {
                roll_back_probe(model, undo);
                return objective_value(model);
            }
            // A Rollback over a set the probe did not move: the caller broke
            // the pairing, so the log's values are not known to match the
            // assignment in front of us -- the probed variables may not have
            // been put back. The honest reading is a commit of BOTH sets (see
            // DeltaMode): the log is written back first, then the walk below
            // recomputes the union's cone from the assignment. Out of contract,
            // so sort-and-unique's O(n log n) is no concern.
            roll_back_probe(model, undo);
            thread_local std::vector<int32_t> merged;
            merged.assign(undo.vars.begin(), undo.vars.end());
            merged.insert(merged.end(), changed_var_ids, changed_var_ids + count);
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            // The walk below stays a Rollback, now with no log pending: a
            // Commit.
            changed_var_ids = merged.data();
            count = merged.size();
        }
    }
    const size_t num_nodes = model.num_nodes();

    // Flat dirty flags + dirty list for O(dirty) cleanup
    // Use thread_local to avoid reallocation across calls
    thread_local std::vector<uint8_t> dirty_flags;
    thread_local std::vector<int32_t> dirty_list;

    if (dirty_flags.size() < num_nodes) {
        dirty_flags.resize(num_nodes, 0);
    }
    dirty_list.clear();

    if (count == 0) {
        if (mode == DeltaMode::Probe) {
            arm_probe_undo(model, span_of(dirty_list), changed_var_ids, count);  // an empty cone
        }
        return objective_value(model);
    }

    // A one-variable walk on a frozen model reads its cone from the cache
    // (#271); a walk over several variables, an unfrozen model's, or one past
    // the cache's cap collects and sorts it below.
    if (count == 1) {
        if (model.has_cached_cone(changed_var_ids[0])) {
            ++cone_cache_counts.cached_walks;
            return walk_cached_cone(model, changed_var_ids, mode, old_values, dirty_flags);
        }
        ++cone_cache_counts.fallback_walks;
    }

    // A model with no incremental Sum -- most of MINLPLib -- evaluates its
    // cone with plain `evaluate()`, building no incremental-Sum rules (its
    // Probe still arms the log, and its paired Rollback returned above
    // without walking). Measured, not guessed: on
    // gear4, whose probe cones are a handful of nodes, merely constructing the
    // incremental-Sum rules per walk cost ~10% of the run (#188).
    if (model.inc_sum_nodes().empty()) {
        const DirtyFlagGuard<std::vector<int32_t>> plain_guard(dirty_flags, dirty_list);
        collect_dirty_cone(model, changed_var_ids, count, dirty_flags, dirty_list);
        if (mode == DeltaMode::Probe) {
            arm_probe_undo(model, span_of(dirty_list), changed_var_ids, count);
        }
        evaluate_dirty_in_topo_order(
            model, dirty_list, dirty_flags, num_nodes,
            [&model](int32_t id) { return evaluate(model.nodes()[id], model); });
        return objective_value(model);
    }
    IncSumWalk sums(model, mode, old_values != nullptr, dirty_flags);

    // Armed BEFORE the seeding loop, so it covers every flag this call sets --
    // including the ones set before an exception out of the walk below.
    const DirtyFlagGuard<std::vector<int32_t>> flag_guard(dirty_flags, dirty_list);
    collect_dirty_cone(model, changed_var_ids, count, dirty_flags, dirty_list);
    if (mode == DeltaMode::Probe) {
        // Before `prepare`, whose pushes are the walk's first writes.
        arm_probe_undo(model, span_of(dirty_list), changed_var_ids, count);
    }
    sums.prepare(span_of(dirty_list), changed_var_ids, count, old_values);
    evaluate_with_sums(model, sums, [&](auto eval_node) {
        evaluate_dirty_in_topo_order(model, dirty_list, dirty_flags, num_nodes, eval_node);
    });

    // The flags are cleared by `flag_guard` on the way out, which is also what
    // covers a throw from user code inside the walk.
    return objective_value(model);
}

}  // namespace

IncrementalSumCounters& incremental_sum_counters() noexcept {
    return inc_sum_counts;
}

ConeCacheCounters& cone_cache_counters() noexcept {
    return cone_cache_counts;
}

double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count, DeltaMode mode) {
    return delta_walk(model, changed_var_ids, count, mode, nullptr);
}

double commit_scalar_move(Model& model, int32_t var_id, double old_value) {
    return delta_walk(model, &var_id, 1, DeltaMode::Commit, &old_value);
}

double probe_scalar_move(Model& model, int32_t var_id, double old_value) {
    return delta_walk(model, &var_id, 1, DeltaMode::Probe, &old_value);
}

void reground_inc_sum(Model& model, int32_t slot) {
    const EvaluationGuard guard("reground_inc_sum");
    IncSums& sums = model.inc_sums();
    // A write between a probe and its rollback would be undone by the log.
    model.probe_undo().pending = false;
    const int32_t nid = model.inc_sum_nodes()[static_cast<size_t>(slot)];
    const std::vector<ExprNode>& nodes = model.nodes();
    model.set_node_value_unchecked(
        nid, checked_resum(model, nodes[nid], sums.slots[static_cast<size_t>(slot)]));
    // Its readers are top-level comparisons (classify_incremental_sums), so
    // they are all there is above it.
    for (const int32_t p : model.parents(nid)) {
        model.set_node_value_unchecked(p, evaluate(nodes[p], model));
    }
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
// that on the way out, including when an allocation throws.
//
// `adjoint` is flat: [0, num_nodes) for nodes, [num_nodes, num_nodes + num_vars)
// for variables. It only ever grows, so a thread that sees a larger model keeps
// the larger buffer.
struct AdjointScratch {
    std::vector<double> adjoint;
    std::vector<int32_t> written;  // adjoint entries touched, for O(touched) cleanup
    std::vector<uint8_t> in_cone;  // node id -> collected into `cone`
    std::vector<int32_t> cone;     // nodes reachable from expr_id through children
};

AdjointScratch& adjoint_scratch() {
    thread_local AdjointScratch scratch;
    return scratch;
}

class AdjointScratchGuard {
public:
    explicit AdjointScratchGuard(AdjointScratch& scratch) : scratch_(scratch) {}
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
        add_chain_term(s.adjoint[key], ld, adj);
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
    const AdjointScratchGuard guard(s);
    reverse_sweep(model, expr_id, s);
    if (var_id < 0 || static_cast<size_t>(var_id) >= model.num_vars()) {
        return 0.0;
    }
    return s.adjoint[model.num_nodes() + static_cast<size_t>(var_id)];
}

double up_edge_derivative(const Model& model, int32_t parent, const ChildRef& child) {
    const ExprNode& pn = model.node(parent);
    auto same = [&child](const ChildRef& r) {
        return r.is_var == child.is_var && r.id == child.id;
    };
    const ConstSpan<ChildRef> kids = model.children(pn);
    if (pn.op == NodeOp::Sum) {
        if ((model.breakpoint_flags()[static_cast<size_t>(parent)] &
             ModelStructure::kRepeatedChild) == 0) {
            return 1.0;
        }
        return static_cast<double>(std::count_if(kids.begin(), kids.end(), same));
    }
    double d = 0.0;
    for (size_t i = 0; i < kids.size(); ++i) {
        if (same(kids[i])) {
            d += local_derivative(pn, static_cast<int>(i), model);
        }
    }
    return d;
}

bool UpConePartials::build(const Model& model, int32_t var_id, size_t max_nodes) {
    try {
        return sweep(model, var_id, max_nodes);
    } catch (...) {
        next_epoch();  // the half-built cone must read 0, as a failed build does
        throw;
    }
}

void UpConePartials::next_epoch() {
    if (++epoch_ == 0) {
        std::fill(stamp_.begin(), stamp_.end(), 0);
        epoch_ = 1;
    }
}

bool UpConePartials::sweep(const Model& model, int32_t var_id, size_t max_nodes) {
    const size_t num_nodes = model.num_nodes();
    if (stamp_.size() < num_nodes) {
        stamp_.resize(num_nodes, 0);
        slope_.resize(num_nodes, 0.0);
    }
    next_epoch();
    cone_.clear();
    if (model.topo_order().size() != num_nodes || model.breakpoint_flags().size() != num_nodes) {
        return false;  // not closed: no parents, no order, no Sum flags
    }
    auto enter = [this](int32_t nid) {
        stamp_[static_cast<size_t>(nid)] = epoch_;
        slope_[static_cast<size_t>(nid)] = 0.0;
        cone_.push_back(nid);
    };
    for (const int32_t dep : model.dependents(var_id)) {
        enter(dep);
    }
    // `cone_` grows while it is walked, so it is indexed rather than iterated.
    size_t head = 0;
    while (head < cone_.size()) {
        if (cone_.size() > max_nodes) {
            next_epoch();  // a failed build reads 0 everywhere
            return false;
        }
        const int32_t nid = cone_[head++];
        for (const int32_t parent : model.parents(nid)) {
            if (stamp_[static_cast<size_t>(parent)] != epoch_) {
                enter(parent);
            }
        }
    }
    if (cone_.size() > max_nodes) {
        next_epoch();
        return false;
    }
    // Topological order, so a node's slope is complete before it is pushed.
    std::sort(cone_.begin(), cone_.end(), [&model](int32_t a, int32_t b) {
        return model.topo_position(a) < model.topo_position(b);
    });
    const ChildRef var_ref{var_id, true};
    for (const int32_t dep : model.dependents(var_id)) {
        slope_[static_cast<size_t>(dep)] += up_edge_derivative(model, dep, var_ref);
    }
    for (const int32_t nid : cone_) {
        const double s = slope_[static_cast<size_t>(nid)];
        if (s == 0.0) {
            continue;
        }
        const ChildRef ref{nid, false};
        for (const int32_t parent : model.parents(nid)) {
            add_chain_term(slope_[static_cast<size_t>(parent)],
                           up_edge_derivative(model, parent, ref), s);
        }
    }
    return true;
}

std::vector<double> compute_all_partials(const Model& model, int32_t expr_id) {
    AdjointScratch& s = adjoint_scratch();
    const AdjointScratchGuard guard(s);
    reverse_sweep(model, expr_id, s);
    const size_t num_nodes = model.num_nodes();
    const auto first = s.adjoint.begin() + static_cast<std::ptrdiff_t>(num_nodes);
    return {first, first + static_cast<std::ptrdiff_t>(model.num_vars())};
}

void compute_partials_sparse(const Model& model, int32_t expr_id,
                             std::vector<std::pair<int32_t, double>>& out) {
    out.clear();
    AdjointScratch& s = adjoint_scratch();
    const AdjointScratchGuard guard(s);
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
