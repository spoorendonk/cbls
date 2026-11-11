#include "cbls/inner_solver.h"

#include "cbls/dag_ops.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace cbls {

namespace {

// Node budget for the `UpConePartials` pass (#207, #249): none. One forward pass over v's
// up-cone gives every G_v row's and the objective's slope, equal to `compute_partial`'s up to
// summation order on finite paths, at about one probe's cost (cost argument:
// docs/architecture.md, "Per-variable cost"). Where it loses (a dense objective in few
// variables) one `compute_all_partials(objective)` per sweep would be n times cheaper but goes
// stale after each committed move of a nonlinear objective, so there is nothing cheaper to
// fall back to past a budget.
constexpr size_t kUnboundedUpCone = std::numeric_limits<size_t>::max();

// The objective as `augmented_objective()` reads it: 0 without one, and a
// non-finite value clamped so the metric stays ordered.
double clamped_objective(const Model& model) {
    if (model.objective_id() < 0) {
        return 0.0;
    }
    const double obj = model.node_value(model.objective_id());
    return std::isfinite(obj) ? obj : kInfPenalty;
}

// The augmented objective's change when one variable moves, over the only terms that CAN
// change: G_v's rows and the objective (#207). No O(rows) scans, and no last-ulp disagreement
// of two readings of the incrementally kept total for a no-op move (#118); differenced per
// row, so a row clamped to kInfPenalty on both sides cancels exactly.
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
    UpConePartials slopes;
    AugmentedDelta delta;
};

// Best value so far for the variable, with its augmented-objective change; seeded with the
// current value at a change of 0.
struct Incumbent {
    double value;
    double delta;
};

// Score `candidate` for `var_id` and record it when it beats `best`. Values and node values
// are left exactly as found, so any number of probes share one baseline. A probe is
// O(up-cone): incremental Sums move by their one changed term (#188) and the Rollback replays
// the undo log without a second walk (#270), instead of re-summing every wide Sum above v.
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

// Newton steps -g/dg, clamped to the domain, toward the roots of the first few violated rows
// OF THIS VARIABLE (G_v, ascending), not of the global violated list, which on a large model
// names rows v is not in (#207).
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
        const bool domain = model.is_domain_node(cid);
        double g = model.node_value(cid);
        // Negated so a NaN residual (#205), which has no root, fails it. A domain row is
        // violated at any g > 0 (#258) and is aimed at the floor itself, not floor + margin.
        if (!(g > (domain ? 0.0 : kDefaultFeasibilityTolerance))) {
            continue;
        }
        if (domain) {
            g -= model.domain_margin();
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

// Backtracking line search down the objective's gradient, halving the step; a point equal to
// the start or the previous point is not probed (clamping makes steps coincide at a bound).
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

// One coordinate-descent step on a Float variable: commit the best Newton or line-search
// candidate if it beats staying put; true if it moved. Cost O(up-cone of v) per slope pass and
// probe, O(|G_v|) per score: nothing scales with the model (#207).
bool descend_float_var(Model& model, const ViolationManager& vm, Scratch& sc, const Variable& var,
                       double initial_step, int max_line_search_steps) {
    const double old_val = var.value;
    // False only on a model not closed, which the hook never sees; every slope
    // then reads 0 and no candidate is probed.
    sc.slopes.build(model, var.id, kUnboundedUpCone);
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

// Minimum-norm Newton step -g/|grad|^2 * grad on ONE violated row over every Float variable
// with a usable partial, rolled back wholesale unless the augmented objective improves; true
// if kept. Needs two variables to do anything the per-variable sweep did not.
bool multi_var_newton_step(Model& model, ViolationManager& vm, int32_t cid) {
    double g = model.node_value(cid);
    // A non-finite residual (a domain error is NaN, #205) gives a NaN or infinite step for
    // every variable. Rejected here, not by the augmented-objective compare: that incremental
    // total can drift a last ulp and accept an all-NaN assignment.
    if (!std::isfinite(g)) {
        return false;
    }
    if (g > 0.0 && model.is_domain_node(cid)) {
        g -= model.domain_margin();  // aim at the floor, as newton_candidates does (#258)
    }
    if (std::abs(g) < 1e-15) {
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

// Coordinate-pass poll stride (#191): kPollUnits / max(rows, nodes), clamped to [1, 256], a
// machine-independent proxy for a descent's cost. A descent is O(up-cone of v) plus O(|G_v|)
// (#207), so the proxy is an upper bound and only ever polls MORE often than needed. A poll is
// the clock read in past_deadline(): ~20-25 ns via the vDSO on tsc, 1408 ns on the reference
// machine (hpet; measured for #113).
//  - Large models (>= kPollUnits nodes): one variable per poll. The overrun is one descent,
//    irreducible short of a poll inside the probes: milliseconds on a variable reaching most
//    of a multi-million-node DAG.
//  - Where nodes overstate the work (large DAGs whose variables each reach few nodes): up to a
//    1.4 us clock read per cheap descent on hpet. The regime this proxy loses in; accepted as
//    it costs hook throughput only, never the budget.
//  - Small models: up to 256 variables per poll, each descent many times the 1.4 us poll; the
//    overrun is 256 descents of a < 256-node model, sub-millisecond.
// A time-sized stride, as FJ's (#113), would break a clockless run's "reads no clock"
// guarantee, and `stop` does not say whether a clock exists. Polls only decide whether to
// continue, so a never-raised `stop` leaves the trajectory unchanged.
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
    Scratch scratch;
    std::vector<int32_t> regrounded;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        const PassOutcome coordinate = coordinate_pass(model, vm, scratch, initial_step_size,
                                                       max_line_search_steps, stop, stride);
        if (coordinate == PassOutcome::Stopped) {
            break;
        }
        // The multi-variable step scores by two raw readings, the second after a plain
        // delta_evaluate that re-sums the Sums it reaches; on drifted Sums the difference would
        // include the removed drift, so start re-grounded. O(1) when nothing drifted.
        reground_drifted_sums(model, regrounded);
        const PassOutcome multi = multi_var_pass(model, vm, max_multi_var_constraints, stop);
        if (multi == PassOutcome::Stopped) {
            break;
        }
        if (coordinate != PassOutcome::Improved && multi != PassOutcome::Improved) {
            break;
        }
    }
    // Coordinate commits can leave an incremental Sum within its drift bound of a re-sum; the
    // search reads rows and objective RAW after the hook, sound only on re-grounded Sums (FJ's
    // batch-end contract, src/model.cpp). So return them re-grounded; O(1) if none drifted.
    reground_drifted_sums(model, regrounded);
    vm.invalidate_cache();
}

}  // namespace cbls
