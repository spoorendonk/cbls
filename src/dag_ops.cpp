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

// The incremental Sum's drift bound (#188) needs TwoSum's exact rounding error, i.e. IEEE double
// arithmetic as written. -ffast-math may fold the error term to 0, making every inexact update
// read as exact and silently disabling the phantom-row gate, so it is refused here.
#if defined(__FAST_MATH__) || defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "src/dag_ops.cpp needs IEEE arithmetic: do not build it with -ffast-math"
#endif
static_assert(FLT_EVAL_METHOD == 0,
              "TwoSum needs double operations evaluated in double (FLT_EVAL_METHOD 0)");

namespace cbls {

namespace detail {

// Kahn's algorithm over node-to-node edges (variables carry no in-degree); children come out
// before parents. Walks `Model::parents`, which lists a parent once even when it names a child
// twice (`prod(n, n)`); that queues each parent at the same point as counting both edges would.
// `sorted` is its own FIFO (append at the back, consume from `head`): a std::queue's order
// without the deque's block allocations.
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

// Is this thread already inside `full_evaluate` or `delta_evaluate`? Re-entry from a lambda
// functor (user code in the walk) is SILENT PERMANENT CORRUPTION: the nested call clears the
// `thread_local` dirty list the outer `DirtyFlagGuard` references, leaking the outer flags,
// and a leaked flag makes the seeding loop skip that node for the life of the process. So it
// is refused. One `thread_local` test and store per call; trajectories are unaffected.
thread_local bool in_evaluation = false;

// RAII, so an exception out of user code clears the flag instead of poisoning the thread.
class EvaluationGuard {
public:
    explicit EvaluationGuard(const char* entry) {
        if (in_evaluation) {
            // Not an assert: reachable from user code in a Release build.
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
    // Re-sums unchecked, so afterwards no incremental Sum is tracked or drifted (#188); its
    // first commit re-sums it, checked. Also sizes the state: every close and structural
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

// Recompute a marked dirty set in topological order, by the cheaper route for THIS set.
// Sorting d entries is O(d log d) with scattered `topo_pos_` loads; scanning `topo_order()` is
// O(|nodes|) sequential flag tests. Sorting wins by orders of magnitude on a few dozen dirty
// nodes of a 4.3M-node MIPfeas instance (the scan made delta evaluation non-sublinear); it
// loses as d approaches |nodes| (a dense continuous model, MINLPLib's regime). The condition
// is that cost model, not a tuned constant: sort while d*log2(d) < |nodes|. Both routes give
// the same order and values. `eval_node` is a template parameter so `evaluate()` inlines.
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

// Clears the dirty flags the caller set, however the caller leaves. The flags are
// `thread_local`, and a LEAKED `1` makes the seeding loop skip that node silently for the rest
// of the process. Only user code in the walk (a lambda functor) can throw; the destructor
// cannot, since every id in `list` already indexed `flags`. `List` is the collected vector,
// read at destruction to cover ids pushed after arming, or a cached cone's span (#271), whose
// flags are all set before anything can throw.
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
// Rules, exactness and containment: docs/architecture.md, "Incremental Sum". Here:
//
// `commit_scalar_move` moves each incremental Sum by its terms' changes: a variable term's
// from the caller's old value, a node term's from its value just before the walk rewrote it.
// The update is applied in place ahead of the Sum's topological turn, which then takes it as
// it stands. Each update adds its two TwoSum errors (each at most half an ulp of its result)
// to `drift_bound`, and every addition to the bound is rounded up (`round_bound_up`), so it
// never underestimates. A non-finite value (the Sum's, or a term's old or new) re-sums at its
// turn, since `inf - inf` is no update.
//
// A Probe that knows the old value applies the commit's update, so an identity move on a
// drifted state scores exactly 0. Where the commit would re-sum instead (untracked, period
// reached, non-finite) the probe still updates and its score differs by at most the drift
// bound plus the re-sum's rounding; FJ makes no feasibility or bump decision on a probe. The
// Rollback replays the undo log (#270): re-summing would snap the committed value and change
// the violated-row bookkeeping under FeasibilityJump's feet.
//
// Cost: two TwoSums and a store per changed term, plus a parents scan per dirty node term.
// Wins on long rows a move touches once (MIP). Loses on short rows, and where non-FJ walks
// (Novelty Jump's legs, the structural batch, the inner solver) alternate with FJ commits on
// one Sum: each leaves it untracked, so the next commit pays a checked re-sum.
namespace {

thread_local IncrementalSumCounters inc_sum_counts;
thread_local ConeCacheCounters cone_cache_counts;

// The dirty_flags bit beyond 1 ("in the cone"), on an incremental Sum: re-sum it at its turn
// rather than take it as its updates left it.
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

// The incremental-Sum rules for one walk. `Plain` (no incremental Sum, or state not yet sized
// by full_evaluate) sends every node to the plain evaluator.
class IncSumWalk {
public:
    IncSumWalk(Model& model, DeltaMode mode, bool push, std::vector<uint8_t>& dirty_flags)
        : model_(model),
          nodes_(model.nodes()),
          sums_(model.inc_sums()),
          dirty_(dirty_flags),
          rule_(rule_for(model, mode)),
          push_(push && (rule_ == Rule::Commit || rule_ == Rule::Probe)) {}

    // Before the walk: a plain commit untracks the Sums in its cone; then the changed
    // variables' moves are pushed into the Sums they are terms of.
    //
    // Out of line deliberately, as is `eval_flagged` (measured, #188): inlined, they grew
    // `delta_walk` and made probe-bound MINLPLib walks 5-14% slower per iteration on the same
    // trajectory (eg_all_s 7.09 s vs 6.20 s; maxmin, nvs01, nvs14 likewise); out of line all
    // timed at or below the pre-#188 engine. Cost: one call per walk and per flagged node.
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
                    // Untrack before the walk can throw: a throw (any lambda, or anything
                    // else) before a Sum's turn would leave it stale but tracked, and the
                    // next commit would update the stale value. Defensive, since a throw
                    // already requires a full_evaluate. Unconditional (#192): keying it on
                    // user-code node kinds missed plain lambdas. Changes no value (eval_sum
                    // untracks each at its turn anyway); costs one dirty-list pass per plain
                    // commit, which FJ's pushing commits never take.
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

    // One dirty node's new value: an incremental Sum as its updates left it, or re-summed; a
    // dirty term node pushes its change into its incremental parents. Split so this common
    // part inlines into the walk: a call per node cost nvs14 ~8% of its run (gprof, #188).
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
        // No incremental Sum (most MINLPLib instances): the plain walk.
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
        // A nonzero bound (from this or an earlier update) means the value may not be the
        // exact sum of the current terms.
        if (st.drift_bound != 0.0) {
            st.drifting = 1;
            if (st.listed == 0) {
                st.listed = 1;
                sums_.drifted.push_back(slot_of(p));
            }
        }
        model_.set_node_value_unchecked(p, s);
    }

    // Untracked as well as flagged: a throw before the Sum's turn clears the flag, and the Sum,
    // which has not taken this term's change, must not be updated from there. (A throw
    // mid-commit can leave other Sums pushed but unread; a full_evaluate after it resets all.)
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
                // A walk without pushes re-sums plainly and untracks, as full_evaluate does:
                // the rounding check is paid only by the next FJ commit to update the Sum.
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

// A Probe's undo log (#270): its cone's values before the probe, taken after collection and
// before any write (term pushes included), so it holds the committed values, drift and all.
// O(cone) loads and stores, against an O(cone log cone) collect-sort-evaluate Rollback walk.
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

// The Rollback of a pending probe (#270): the log written back, exact by construction, with no
// collection, sort or evaluation. Re-evaluating would not restore a drifted Sum (#188).
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

// A one-variable walk over the cone `freeze()` cached (#271): no BFS, no sort, the same nodes
// in the same topological order as the collecting walk, so the same values. Without
// incremental Sums no dirty flag is set (only the scan route and Sum pushes read them);
// otherwise every flag is set up front under a guard armed first.
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

// The one walk behind every entry point. Non-null `old_values` (previous value of each of
// `changed_var_ids`) switches term updates on; null re-sums every dirty incremental Sum.
double delta_walk(Model& model, const int32_t* changed_var_ids, size_t count, DeltaMode mode,
                  const double* old_values) {
    const EvaluationGuard guard("delta_evaluate");
    // Any walk moves on from the pending probe's assignment, so the log is dropped whatever the
    // mode; only its own Rollback spends it.
    ProbeUndo& undo = model.probe_undo();
    if (undo.pending) {
        undo.pending = false;
        if (mode == DeltaMode::Rollback) {
            if (undo_matches(undo, changed_var_ids, count)) {
                roll_back_probe(model, undo);
                return objective_value(model);
            }
            // A Rollback over a set the probe did not move (broken pairing; the probed
            // variables may not be back): commit BOTH sets (see DeltaMode). Write the log
            // back, then recompute the union's cone. Out of contract, so O(n log n) is fine.
            roll_back_probe(model, undo);
            thread_local std::vector<int32_t> merged;
            merged.assign(undo.vars.begin(), undo.vars.end());
            merged.insert(merged.end(), changed_var_ids, changed_var_ids + count);
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            // The walk below stays a Rollback, now with no log pending: a Commit.
            changed_var_ids = merged.data();
            count = merged.size();
        }
    }
    const size_t num_nodes = model.num_nodes();

    // Flat dirty flags + list for O(dirty) cleanup; thread_local to avoid reallocation.
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

    // A one-variable walk on a frozen model reads its cached cone (#271); several variables,
    // an unfrozen model, or a variable past the cache's cap collect and sort below.
    if (count == 1) {
        if (model.has_cached_cone(changed_var_ids[0])) {
            ++cone_cache_counts.cached_walks;
            return walk_cached_cone(model, changed_var_ids, mode, old_values, dirty_flags);
        }
        ++cone_cache_counts.fallback_walks;
    }

    // No incremental Sum (most of MINLPLib): plain `evaluate()`, no incremental-Sum rules (its
    // Probe still arms the log). Measured: on gear4, whose probe cones are a handful of nodes,
    // merely constructing the rules per walk cost ~10% of the run (#188).
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

    // Armed BEFORE seeding, so it covers every flag this call sets, a throw included.
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
    // Its readers are top-level comparisons (classify_incremental_sums): nothing else above.
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

// Reverse-sweep scratch for the three entry points below, all-zero / empty between calls
// (`AdjointScratchGuard` restores that, an allocation throw included). `adjoint` is flat:
// [0, num_nodes) nodes, then variables; it only grows.
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

// Push one node's adjoint onto its children: the sweep's body, shared by both routes.
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

// Reverse-mode sweep from `expr_id`: on return `s.adjoint` holds d expr / d x for every node
// and variable x, and `s.written` lists every entry touched.
//
// Only `expr_id`'s cone (reachable through children) can carry a nonzero adjoint, so the
// sweep visits it in reverse topological order. With the full walk's `adjoint == 0.0` skip
// kept, it performs the same floating-point operations in the same order: results and every
// trajectory are bit-identical to the full-order walk.
//
// Cost: collect O(cone + edges) and sort O(c log c) with scattered `topo_position` loads,
// against the full walk's O(|nodes|) tests. Wins by orders of magnitude on a few-dozen-term
// MIP row of a 765k-node model (MIPfeas uccase12: the full walk was 87% of CPU, gprof at
// 175e617); loses as the cone nears the whole DAG. So collect while c * (log2|nodes| + 1) <=
// |nodes| (log2|nodes| stands in for the unknown log2(c)), else abandon and scan the full
// order. The abandoned part (<= ~|nodes| / log2|nodes| nodes) is visited by the fallback
// anyway: worst case the full walk's cost plus a comparable amount.
//
// A model whose `topo_order` does not cover every node (an unclosed one has none) takes the
// full-order walk (over an empty order: all zeros).
void reverse_sweep(const Model& model, int32_t expr_id, AdjointScratch& s) {
    const size_t num_nodes = model.num_nodes();
    const size_t total_size = num_nodes + model.num_vars();
    if (s.adjoint.size() < total_size) {
        s.adjoint.resize(total_size, 0.0);
    }
    if (s.in_cone.size() < num_nodes) {
        s.in_cone.resize(num_nodes, 0);
    }

    // Each push precedes its write, so a `bad_alloc` leaves nothing the guard cannot undo.
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
