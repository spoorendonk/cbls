#include "cbls/inner_solver.h"

#include "cbls/dag_ops.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace cbls {

namespace {

// Best value found so far for the variable under consideration, paired with the
// augmented objective it achieved. Seeded with the variable's current value.
struct Incumbent {
    double value;
    double augmented;
};

// Apply `candidate` to `var_id`, measure the augmented objective there, then put
// the variable back. Records the candidate when it beats `best`. The model's
// values (and its node values) are left exactly as they were found, so the
// caller may probe any number of candidates against one common baseline.
void probe_candidate(Model& model, ViolationManager& vm, int32_t var_id, double old_val,
                     double candidate, Incumbent& best) {
    model.var_mut(var_id).value = candidate;
    delta_evaluate(model, {var_id});
    double new_aug = vm.augmented_objective();
    if (new_aug < best.augmented) {
        best.value = candidate;
        best.augmented = new_aug;
    }
    model.var_mut(var_id).value = old_val;
    delta_evaluate(model, {var_id});
}

// Newton steps toward the root of the first few violated constraints: for a
// residual g with partial dg, the step that would zero g is -g/dg, clamped to
// the variable's domain.
void newton_candidates(Model& model, ViolationManager& vm, const Variable& var, double old_val,
                       Incumbent& best) {
    auto violated = vm.violated_constraints();
    int n_check = std::min(static_cast<int>(violated.size()), 3);
    for (int ci = 0; ci < n_check; ++ci) {
        int32_t cid = model.constraint_ids()[violated[ci]];
        double g = model.node_value(cid);
        double dg = compute_partial(model, cid, var.id);
        // Negated form of the original `> 1e-12` guard, not `<= 1e-12`: a NaN
        // partial must skip the candidate, and every comparison against NaN is
        // false. Same reasoning at the two other inverted guards below.
        if (!(std::abs(dg) > 1e-12)) {
            continue;
        }
        double candidate = std::clamp(old_val + (-g / dg), var.lb, var.ub);
        if (std::abs(candidate - old_val) > 1e-15) {
            probe_candidate(model, vm, var.id, old_val, candidate, best);
        }
    }
}

// Backtracking line search down the objective's gradient: halve the step size
// each time, and probe each point that is distinct both from where the variable
// started and from the previous point (clamping makes successive steps
// coincide once the domain boundary is reached).
void line_search_candidates(Model& model, ViolationManager& vm, const Variable& var, double old_val,
                            double initial_step, int max_steps, Incumbent& best) {
    if (model.objective_id() < 0) {
        return;
    }
    double df = compute_partial(model, model.objective_id(), var.id);
    if (!(std::abs(df) > 1e-12)) {  // NaN gradient: no step, see newton_candidates
        return;
    }
    double step = initial_step;
    double prev_candidate = old_val;
    for (int ls = 0; ls < max_steps; ++ls) {
        double candidate = std::clamp(old_val - (step * df), var.lb, var.ub);
        if (std::abs(candidate - old_val) > 1e-15 && std::abs(candidate - prev_candidate) > 1e-15) {
            probe_candidate(model, vm, var.id, old_val, candidate, best);
        }
        prev_candidate = candidate;
        step *= 0.5;
    }
}

// One coordinate-descent step on a single Float variable: collect the Newton and
// line-search candidates against a common baseline, then commit the best one if
// it beats staying put. Returns true if the variable moved.
bool descend_float_var(Model& model, ViolationManager& vm, const Variable& var, double initial_step,
                       int max_line_search_steps) {
    const double old_val = var.value;
    Incumbent best{old_val, vm.augmented_objective()};

    newton_candidates(model, vm, var, old_val, best);
    line_search_candidates(model, vm, var, old_val, initial_step, max_line_search_steps, best);

    if (!(std::abs(best.value - old_val) > 1e-15)) {  // NaN: nothing to commit
        return false;
    }
    model.var_mut(var.id).value = best.value;
    delta_evaluate(model, {var.id});
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
PassOutcome coordinate_pass(Model& model, ViolationManager& vm, double initial_step,
                            int max_line_search_steps, StopRef stop, int vars_per_poll) {
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
        if (descend_float_var(model, vm, var, initial_step, max_line_search_steps)) {
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
// The stride is sized by the model's row count, as a machine-independent proxy
// for a variable's cost: each variable's descent reads every row's value once in
// its violated-row scan (newton_candidates), before any partial or probe. So
// between two polls lie at least kPollRowReads row reads, unless the
// kMaxVarsPerPoll cap binds.
//
// What a poll costs is the clock read inside the search's past_deadline():
// ~20-25 ns through the vDSO on a tsc clocksource, but 1408 ns on this project's
// reference machine (hpet; measured for #113, docs/architecture.md). The two
// regimes, and where each constant loses:
//  - Many rows (>= kPollRowReads): one variable per poll. The poll is then at
//    most ~1.4 us against a scan of >= 65 536 rows, tens of microseconds, so a
//    few percent of the hook at worst. The overrun is one variable's descent,
//    irreducible from here: on a multi-million-row model that one scan plus its
//    probes is milliseconds, and only a poll inside the probes would cut it.
//  - Few rows: up to kMaxVarsPerPoll variables per poll. Where rows are so few
//    that the cap binds, a variable's descent is dominated by its partials and
//    line-search probes (each a delta_evaluate), so 256 of them are many times
//    the 1.4 us poll, and the overrun is 256 such descents. A model with few rows
//    but very large per-variable cones is what this proxy undercounts: there
//    the overrun can reach 256 expensive descents.
// A stride sized in time, as FJ's is (#113), is not open here: the hook cannot
// read the clock itself without breaking a clockless run's "reads no clock"
// guarantee, and the stop it is handed does not say whether a clock exists.
//
// The polls never change what the descent does, only whether it continues, so
// with `stop` never raised the trajectory is exactly what it was without them.
constexpr int64_t kPollRowReads = int64_t{1} << 16;
constexpr int kMaxVarsPerPoll = 256;

}  // namespace

int FloatIntensifyHook::vars_per_poll(const Model& model) {
    const auto rows = std::max<int64_t>(1, static_cast<int64_t>(model.constraint_ids().size()));
    return static_cast<int>(std::clamp<int64_t>(kPollRowReads / rows, 1, kMaxVarsPerPoll));
}

void FloatIntensifyHook::solve(Model& model, ViolationManager& vm,
                               const std::vector<int32_t>& /*last_changed_vars*/, StopRef stop) {
    const int stride = vars_per_poll(model);
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        const PassOutcome coordinate =
            coordinate_pass(model, vm, initial_step_size, max_line_search_steps, stop, stride);
        if (coordinate == PassOutcome::Stopped) {
            return;
        }
        const PassOutcome multi = multi_var_pass(model, vm, max_multi_var_constraints, stop);
        if (multi == PassOutcome::Stopped) {
            return;
        }
        if (coordinate != PassOutcome::Improved && multi != PassOutcome::Improved) {
            break;
        }
    }
}

}  // namespace cbls
