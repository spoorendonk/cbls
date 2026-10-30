#include "cbls/inner_solver.h"

#include "cbls/dag_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cbls {

namespace {

// d(node)/d(v) for every node above one variable, forward mode (#207).
//
// One pass over v's up-cone -- every node a change of v can reach through
// parents -- gives the partial of each row in G_v AND of the objective at once,
// which is everything a coordinate step on v reads. The cone is collected through
// `Model::parents`, sorted by topological position and swept once, each edge's
// factor being the parent's `local_derivative` for that operand (summed when the
// parent names it twice). That is the same chain rule the reverse sweep applies,
// so the slopes equal `compute_partial`'s up to summation order.
//
// COST, per variable: O(u log u) for an up-cone of u nodes, plus their parent
// edges. A probe's `delta_evaluate` dirties that same up-cone, so the slopes cost
// about one probe -- never more than a constant factor of the work the descent
// does anyway. Before #207 the descent took the objective partial by a reverse
// sweep over the objective's WHOLE cone and scanned every row for violated ones,
// O(nodes) per variable, O(Float vars x nodes) per sweep: on UC-CHPED ucp100-48p
// (9.6k variables, 195k nodes) one sweep outlasted a 10 s budget.
//
// Where it wins: a separable objective over many variables (UC-CHPED, every MIP
// with a linear objective), where u is a handful of nodes and the objective cone
// is the model. Where it loses: a dense objective in a few variables (MINLPLib's
// least-squares shapes), where every variable's up-cone is roughly the whole
// objective cone. There one `compute_all_partials(objective)` per sweep would be
// n times cheaper than n forward passes, but its gradient goes stale with every
// committed move of a nonlinear objective; the forward pass costs there exactly
// what the per-variable reverse sweep it replaces did, so that regime is no
// worse than before, only not better.
//
// A Sum's edge factor is its multiplicity of the operand, 1 unless the Sum
// names a child twice. `parents` lists a parent once however often it names the
// child, so the multiplicity has to come from the Sum's children; scanning them
// per edge would make an objective Sum over n terms cost O(n) for EVERY variable,
// the quadratic this class exists to remove. So each Sum is scanned once per
// hook call, the first time an edge into it is taken, and the verdict memoised.
// Any other op scans its children per edge, as `local_derivative` largely does
// itself; a wide Prod/Min/Max in many variables' cones is the regime that
// remains O(arity) per variable.
class UpConeSlopes {
public:
    explicit UpConeSlopes(const Model& model)
        : stamp_(model.num_nodes(), 0),
          slope_(model.num_nodes(), 0.0),
          sum_kind_(model.num_nodes(), kSumUnknown),
          child_seen_(model.num_nodes() + model.num_vars(), -1) {}

    void build(const Model& model, int32_t var_id) {
        if (++epoch_ == 0) {
            std::fill(stamp_.begin(), stamp_.end(), 0);
            epoch_ = 1;
        }
        cone_.clear();
        for (const int32_t dep : model.dependents(var_id)) {
            enter(dep);
        }
        // `cone_` grows while it is walked, so it is indexed rather than iterated.
        size_t head = 0;
        while (head < cone_.size()) {
            const int32_t nid = cone_[head++];
            for (const int32_t parent : model.parents(nid)) {
                if (stamp_[static_cast<size_t>(parent)] != epoch_) {
                    enter(parent);
                }
            }
        }
        // Topological order, so a node's slope is complete before it is pushed.
        std::sort(cone_.begin(), cone_.end(), [&model](int32_t a, int32_t b) {
            return model.topo_position(a) < model.topo_position(b);
        });
        const ChildRef var_ref{var_id, true};
        for (const int32_t dep : model.dependents(var_id)) {
            slope_[static_cast<size_t>(dep)] += edge_derivative(model, dep, var_ref);
        }
        for (const int32_t nid : cone_) {
            const double s = slope_[static_cast<size_t>(nid)];
            if (s == 0.0) {
                continue;
            }
            const ChildRef ref{nid, false};
            for (const int32_t parent : model.parents(nid)) {
                // A zero edge (an untaken If branch, a Min's unselected operand) is
                // skipped, not multiplied: an infinite `s` times 0 would read NaN
                // where the edge carries nothing, as in `UpConePartials::build`.
                const double d = edge_derivative(model, parent, ref);
                if (d != 0.0) {
                    slope_[static_cast<size_t>(parent)] += d * s;
                }
            }
        }
    }

    // d(node)/dv at the current point; 0 for a node v cannot reach.
    [[nodiscard]] double of(int32_t nid) const {
        const auto i = static_cast<size_t>(nid);
        return stamp_[i] == epoch_ ? slope_[i] : 0.0;
    }

private:
    static constexpr uint8_t kSumUnknown = 0;
    static constexpr uint8_t kSumDistinct = 1;
    static constexpr uint8_t kSumRepeats = 2;

    void enter(int32_t nid) {
        stamp_[static_cast<size_t>(nid)] = epoch_;
        slope_[static_cast<size_t>(nid)] = 0.0;
        cone_.push_back(nid);
    }

    [[nodiscard]] static size_t key(const Model& model, const ChildRef& r) {
        return r.is_var ? model.num_nodes() + static_cast<size_t>(r.id) : static_cast<size_t>(r.id);
    }

    // Does Sum `parent` name some child more than once? Scanned on first use.
    bool sum_repeats(const Model& model, int32_t parent) {
        uint8_t& kind = sum_kind_[static_cast<size_t>(parent)];
        if (kind == kSumUnknown) {
            kind = kSumDistinct;
            for (const ChildRef& c : model.children(model.node(parent))) {
                int32_t& seen = child_seen_[key(model, c)];
                if (seen == parent) {
                    kind = kSumRepeats;
                    break;
                }
                seen = parent;
            }
        }
        return kind == kSumRepeats;
    }

    // d(parent)/d(child), summed over the operands of `parent` naming `child`.
    double edge_derivative(const Model& model, int32_t parent, const ChildRef& child) {
        const ExprNode& pn = model.node(parent);
        const ConstSpan<ChildRef> kids = model.children(pn);
        auto same = [&child](const ChildRef& r) {
            return r.is_var == child.is_var && r.id == child.id;
        };
        if (pn.op == NodeOp::Sum && !sum_repeats(model, parent)) {
            return 1.0;
        }
        double d = 0.0;
        for (size_t i = 0; i < kids.size(); ++i) {
            if (same(kids[i])) {
                d += local_derivative(pn, static_cast<int>(i), model);
            }
        }
        return d;
    }

    std::vector<uint32_t> stamp_;
    std::vector<double> slope_;
    std::vector<uint8_t> sum_kind_;
    std::vector<int32_t> child_seen_;
    std::vector<int32_t> cone_;
    uint32_t epoch_ = 0;
};

// The objective as `augmented_objective()` reads it: 0 without one, and a
// non-finite value clamped so the metric stays ordered.
double clamped_objective(const Model& model) {
    if (model.objective_id() < 0) {
        return 0.0;
    }
    const double obj = model.node_value(model.objective_id());
    return std::isfinite(obj) ? obj : kInfPenalty;
}

// The augmented objective's change when one variable moves, measured over the
// only terms that CAN change: the rows of G_v and the objective (#207). Every
// other row keeps its node value bit for bit, so this is the difference of two
// `augmented_objective()` readings without their two O(rows) scans -- and
// without the last-ulp disagreement two readings of that incrementally kept
// total can show for a move that changed nothing (the reason
// `weighted_delta_from` differences per row, #118). Rows are differenced the
// same way, so a row clamped to kInfPenalty on both sides cancels exactly.
class AugmentedDelta {
public:
    void snapshot(const Model& model, const ViolationManager& vm, int32_t var_id) {
        rows_ = model.constraints_of_var(var_id);
        const auto& cids = model.constraint_ids();
        base_.resize(rows_.size());
        for (size_t k = 0; k < rows_.size(); ++k) {
            base_[k] =
                clamped_node_violation(model.node_value(cids[static_cast<size_t>(rows_[k])]));
        }
        base_objective_ = clamped_objective(model);
        weights_ = &vm.weights;
    }

    [[nodiscard]] double now(const Model& model) const {
        const auto& cids = model.constraint_ids();
        double delta = 0.0;
        for (size_t k = 0; k < rows_.size(); ++k) {
            const auto ci = static_cast<size_t>(rows_[k]);
            const double v = clamped_node_violation(model.node_value(cids[ci]));
            if (v != base_[k]) {
                delta += (*weights_)[ci] * (v - base_[k]);
            }
        }
        const double obj = clamped_objective(model);
        if (obj != base_objective_) {
            delta += obj - base_objective_;
        }
        return delta;
    }

    [[nodiscard]] ConstSpan<int32_t> rows() const { return rows_; }

private:
    ConstSpan<int32_t> rows_;
    std::vector<double> base_;
    double base_objective_ = 0.0;
    const std::vector<double>* weights_ = nullptr;
};

// Per-call scratch shared by every coordinate step.
struct Scratch {
    explicit Scratch(const Model& model) : slopes(model) {}
    UpConeSlopes slopes;
    AugmentedDelta delta;
};

// Best value found so far for the variable under consideration, paired with the
// change in augmented objective it achieved. Seeded with the variable's current
// value, at a change of 0.
struct Incumbent {
    double value;
    double delta;
};

// Apply `candidate` to `var_id`, measure the augmented objective's change there,
// then put the variable back. Records the candidate when it beats `best`. The
// model's values (and its node values) are left exactly as they were found, so
// the caller may probe any number of candidates against one common baseline.
//
// The probe moves each incremental Sum in v's cone by its one changed term
// (`probe_scalar_move`, #188) and the Rollback writes the probe's stash back, so
// a probe costs O(up-cone) rather than a re-sum of every wide Sum above v. On
// UC-CHPED the objective is a Sum over every unit x period cost term; before
// this, re-summing it twice per probe was what each probe cost.
void probe_candidate(Model& model, const AugmentedDelta& delta, int32_t var_id, double old_val,
                     double candidate, Incumbent& best) {
    model.var_mut(var_id).value = candidate;
    probe_scalar_move(model, var_id, old_val);
    const double d = delta.now(model);
    if (d < best.delta) {
        best.value = candidate;
        best.delta = d;
    }
    model.var_mut(var_id).value = old_val;
    delta_evaluate(model, &var_id, 1, DeltaMode::Rollback);
}

// Newton steps toward the root of the first few violated rows OF THIS VARIABLE
// (G_v, ascending): for a residual g with partial dg, the step that would zero g
// is -g/dg, clamped to the variable's domain. Before #207 these were the first
// three rows of the GLOBAL violated list, which for almost every variable of a
// large model named rows it is not in -- partial 0, no candidate.
void newton_candidates(Model& model, const Scratch& sc, const Variable& var, double old_val,
                       Incumbent& best) {
    constexpr int kMaxNewton = 3;
    int budget = kMaxNewton;
    const auto& cids = model.constraint_ids();
    for (const int32_t ci : sc.delta.rows()) {
        if (budget <= 0) {
            break;
        }
        const int32_t cid = cids[static_cast<size_t>(ci)];
        const double g = model.node_value(cid);
        // Not `<= tol`: a NaN residual (#205) has no root to step toward, and
        // fails this test as every comparison against NaN does.
        if (!(g > kDefaultFeasibilityTolerance)) {
            continue;
        }
        const double dg = sc.slopes.of(cid);
        // Negated form of `> 1e-12`, not `<= 1e-12`: a NaN partial must skip the
        // candidate. Same reasoning at the two other inverted guards below.
        if (!(std::abs(dg) > 1e-12)) {
            continue;
        }
        const double candidate = std::clamp(old_val + (-g / dg), var.lb, var.ub);
        if (std::abs(candidate - old_val) > 1e-15) {
            probe_candidate(model, sc.delta, var.id, old_val, candidate, best);
            --budget;
        }
    }
}

// Backtracking line search down the objective's gradient: halve the step size
// each time, and probe each point that is distinct both from where the variable
// started and from the previous point (clamping makes successive steps
// coincide once the domain boundary is reached).
void line_search_candidates(Model& model, const Scratch& sc, const Variable& var, double old_val,
                            double initial_step, int max_steps, Incumbent& best) {
    if (model.objective_id() < 0) {
        return;
    }
    const double df = sc.slopes.of(model.objective_id());
    if (!(std::abs(df) > 1e-12)) {  // NaN gradient: no step, see newton_candidates
        return;
    }
    double step = initial_step;
    double prev_candidate = old_val;
    for (int ls = 0; ls < max_steps; ++ls) {
        const double candidate = std::clamp(old_val - (step * df), var.lb, var.ub);
        if (std::abs(candidate - old_val) > 1e-15 && std::abs(candidate - prev_candidate) > 1e-15) {
            probe_candidate(model, sc.delta, var.id, old_val, candidate, best);
        }
        prev_candidate = candidate;
        step *= 0.5;
    }
}

// One coordinate-descent step on a single Float variable: collect the Newton and
// line-search candidates against a common baseline, then commit the best one if
// it beats staying put. Returns true if the variable moved. Cost: the slopes and
// each probe are O(up-cone of v), the scoring O(|G_v|) -- nothing scales with
// the model (#207).
bool descend_float_var(Model& model, const ViolationManager& vm, Scratch& sc, const Variable& var,
                       double initial_step, int max_line_search_steps) {
    const double old_val = var.value;
    sc.slopes.build(model, var.id);
    sc.delta.snapshot(model, vm, var.id);
    Incumbent best{old_val, 0.0};

    newton_candidates(model, sc, var, old_val, best);
    line_search_candidates(model, sc, var, old_val, initial_step, max_line_search_steps, best);

    if (!(std::abs(best.value - old_val) > 1e-15)) {  // NaN: nothing to commit
        return false;
    }
    model.var_mut(var.id).value = best.value;
    // Incremental, as FJ commits (#188); the drift it can leave is re-grounded
    // before the hook returns (FloatIntensifyHook::solve).
    commit_scalar_move(model, var.id, old_val);
    return true;
}

// Minimum-norm Newton step on ONE violated constraint, over every Float variable
// with a usable partial: the step -g/|grad|^2 * grad is the smallest move that
// would zero the residual under a linear model. Applied to all of them at once
// and rolled back wholesale unless the augmented objective improved. Needs at
// least two variables to be doing anything the per-variable sweep above did not.
// Returns true if the step was kept.
bool multi_var_newton_step(Model& model, ViolationManager& vm, int32_t cid) {
    double g = model.node_value(cid);
    // A non-finite residual (a domain error is NaN since #205) gives a NaN or
    // infinite step for every variable at once. Rejecting it must not hang on
    // two augmented_objective() readings comparing exactly: that total is an
    // incrementally maintained accumulator with a periodic resync, so a
    // last-ulp drift could accept an all-NaN assignment.
    if (!std::isfinite(g) || std::abs(g) < 1e-15) {
        return false;
    }

    // Batch AD: one reverse pass for all partials
    auto all_partials = compute_all_partials(model, cid);

    struct VarGrad {
        int32_t id;
        double dg;
        double old_val;
    };
    std::vector<VarGrad> grads;
    for (const auto& v : model.variables()) {
        if (v.type != VarType::Float) {
            continue;
        }
        double dg = all_partials[v.id];
        if (std::abs(dg) > 1e-12) {
            grads.push_back({v.id, dg, v.value});
        }
    }
    if (grads.size() < 2) {
        return false;
    }

    double grad_norm_sq = 0.0;
    for (const auto& vg : grads) {
        grad_norm_sq += vg.dg * vg.dg;
    }
    double scale = -g / grad_norm_sq;

    // Capture baseline before applying step
    double old_aug = vm.augmented_objective();

    // Apply minimum-norm Newton step
    std::vector<int32_t> changed_ids;
    changed_ids.reserve(grads.size());
    for (const auto& vg : grads) {
        const auto& v = model.var(vg.id);
        double new_val = std::clamp(vg.old_val + (scale * vg.dg), v.lb, v.ub);
        model.var_mut(vg.id).value = new_val;
        changed_ids.push_back(vg.id);
    }
    delta_evaluate(model, changed_ids);

    if (vm.augmented_objective() < old_aug) {
        return true;
    }
    // Restore all vars
    for (const auto& vg : grads) {
        model.var_mut(vg.id).value = vg.old_val;
    }
    delta_evaluate(model, changed_ids);
    return false;
}

// How one half of a sweep ended.
enum class PassOutcome : std::uint8_t { Improved, Unchanged, Stopped };

// The coordinate half of a sweep: one descent step per Float variable, polling
// `stop` before the first and then before every `vars_per_poll`-th one.
PassOutcome coordinate_pass(Model& model, const ViolationManager& vm, Scratch& sc,
                            double initial_step, int max_line_search_steps, StopRef stop,
                            int vars_per_poll) {
    bool improved = false;
    int until_poll = 0;
    for (const auto& var : model.variables()) {
        if (var.type != VarType::Float) {
            continue;
        }
        if (until_poll == 0) {
            if (stop.requested()) {
                return PassOutcome::Stopped;
            }
            until_poll = vars_per_poll;
        }
        --until_poll;
        if (descend_float_var(model, vm, sc, var, initial_step, max_line_search_steps)) {
            improved = true;
        }
    }
    return improved ? PassOutcome::Improved : PassOutcome::Unchanged;
}

// The multi-variable half: a minimum-norm Newton step on each of the first
// `max_constraints` violated rows, re-reading the violated set before each.
// Every step is polled: each is a whole reverse pass over the DAG plus a pass
// over every variable, which no poll cost comes near.
PassOutcome multi_var_pass(Model& model, ViolationManager& vm, int max_constraints, StopRef stop) {
    bool improved = false;
    for (int ci = 0; ci < max_constraints; ++ci) {
        auto violated = vm.violated_constraints();
        if (ci >= static_cast<int>(violated.size())) {
            break;
        }
        if (stop.requested()) {
            return PassOutcome::Stopped;
        }
        if (multi_var_newton_step(model, vm, model.constraint_ids()[violated[ci]])) {
            improved = true;
        }
    }
    return improved ? PassOutcome::Improved : PassOutcome::Unchanged;
}

// How often the coordinate pass polls (#191). Without any poll the hook ran
// MIPfeas rahue (12k Float columns, 77k rows) ~10 s past a 20 s budget.
//
// The stride is sized by the model's size, as a machine-independent proxy for a
// variable's cost: kPollUnits / max(rows, nodes), clamped to [1, 256]. Since
// #207 a descent costs O(up-cone of v): the forward slopes, a handful of
// `delta_evaluate` probes over that cone and a scoring pass over G_v. Before it,
// every descent scanned every row and ran a reverse sweep over the objective's
// whole cone, so the proxy was the descent's cost; now it is an upper bound on
// it -- an up-cone is at most every node, G_v at most every row -- and the
// proxy only ever polls MORE often than the work strictly needs. On a model
// whose up-cones are small (UC-CHPED, MIPfeas) that is one poll per cheap
// descent on large models, the regime discussed below.
//
// What a poll costs is the clock read inside the search's past_deadline():
// ~20-25 ns through the vDSO on a tsc clocksource, but 1408 ns on this project's
// reference machine (hpet; measured for #113, docs/architecture.md). The
// regimes, and where each constant loses:
//  - Large models (>= kPollUnits nodes): one variable per poll -- rahue
//    (352k nodes) is here. The overrun is one variable's descent, irreducible
//    from here: O(up-cone), which on a variable reaching most of a
//    multi-million-node DAG is milliseconds, and only a poll inside the probes
//    would cut it. Where the up-cones are small the poll may be a noticeable
//    share of a descent on hpet (unmeasured) -- the next regime.
//  - Where nodes overstate the work -- since #207 any large DAG whose
//    variables each reach few nodes -- a descent can be far cheaper than a node count suggests, and
//    polling every variable then costs up to a 1.4 us clock read per cheap descent on hpet. That is
//    the regime this proxy loses in, accepted because it can only cost hook throughput, never the
//    budget.
//  - Small models: up to kMaxVarsPerPoll variables per poll. Every descent there
//    is dominated by its partials and line-search probes (each a
//    delta_evaluate), so 256 of them are many times the 1.4 us poll, and the
//    overrun is 256 descents of a model with < 256 nodes -- sub-millisecond.
// A stride sized in time, as FJ's is (#113), is not open here: the hook cannot
// read the clock itself without breaking a clockless run's "reads no clock"
// guarantee, and the stop it is handed does not say whether a clock exists.
//
// The polls never change what the descent does, only whether it continues, so
// with `stop` never raised the trajectory is exactly what it was without them.
constexpr int64_t kPollUnits = int64_t{1} << 16;
constexpr int kMaxVarsPerPoll = 256;

}  // namespace

int FloatIntensifyHook::vars_per_poll(const Model& model) {
    const auto size =
        std::max<int64_t>({int64_t{1}, static_cast<int64_t>(model.constraint_ids().size()),
                           static_cast<int64_t>(model.num_nodes())});
    return static_cast<int>(std::clamp<int64_t>(kPollUnits / size, 1, kMaxVarsPerPoll));
}

void FloatIntensifyHook::solve(Model& model, ViolationManager& vm,
                               const std::vector<int32_t>& /*last_changed_vars*/, StopRef stop) {
    const int stride = vars_per_poll(model);
    Scratch scratch(model);
    std::vector<int32_t> regrounded;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        const PassOutcome coordinate = coordinate_pass(model, vm, scratch, initial_step_size,
                                                       max_line_search_steps, stop, stride);
        if (coordinate == PassOutcome::Stopped) {
            break;
        }
        // The multi-variable step reads rows and the objective raw and scores
        // by two absolute readings, the second after a plain delta_evaluate that
        // re-sums every incremental Sum it reaches. On a drifted state that
        // difference would include the drift the re-sum removed, so the pass
        // starts from re-grounded Sums. O(1) when nothing drifted.
        reground_drifted_sums(model, regrounded);
        const PassOutcome multi = multi_var_pass(model, vm, max_multi_var_constraints, stop);
        if (multi == PassOutcome::Stopped) {
            break;
        }
        if (coordinate != PassOutcome::Improved && multi != PassOutcome::Improved) {
            break;
        }
    }
    // The coordinate commits move incremental Sums by their terms' changes, which
    // can leave one within its drift bound of a re-sum. The search reads the
    // objective and the rows RAW after the hook returns (incumbent, bound,
    // feasibility), which is sound only on re-grounded Sums -- the contract FJ
    // keeps by re-grounding at every batch end (src/model.cpp, the incremental
    // Sum rules). So the hook leaves them as it found them: re-grounded.
    // O(1) when nothing drifted.
    reground_drifted_sums(model, regrounded);
    vm.invalidate_cache();
}

}  // namespace cbls
