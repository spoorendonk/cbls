#include "cbls/feasibility_jump.h"

#include "cbls/dag_ops.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cbls {

namespace {

constexpr double kTol = 1e-9;

// A constraint is violated if its residual exceeds the tolerance. Written so
// that non-finite residuals (NaN from inf-inf, or +inf) count as violated:
// !(x <= tol) is true for x > tol and for NaN. This prevents a NaN constraint
// from being silently treated as satisfied (a false "feasible").
bool is_violated(double residual) {
    return !(residual <= kTol);
}

double clamp_to_domain(const Variable& var, double value) {
    return std::min(std::max(value, var.lb), var.ub);
}

// Minimise f over [a, b] by golden-section search; returns the argmin. The
// paper assumes each violation is convex in a single variable, so a positive-
// weighted sum is convex and this converges to the global argmin; on a
// non-convex f it returns a local min (correctness unaffected, search just more
// restricted).
template <class F>
double golden_section_argmin(F&& func, double a, double b) {
    const double phi = 0.6180339887498949;  // (sqrt(5)-1)/2
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);
    double fc = func(c);
    double fd = func(d);
    for (int it = 0; it < 100 && (b - a) > 1e-12 * (std::abs(a) + std::abs(b) + 1.0); ++it) {
        if (fc < fd) {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = func(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = func(d);
        }
    }
    return 0.5 * (a + b);
}

}  // namespace

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

JumpResult compute_var_jump(Model& model, const ViolationManager& vm, int32_t var_id) {
    const Variable& var = model.var(var_id);
    const double x0 = var.value;

    // f(j) = weighted violation delta of moving var_id to j (0 at the current
    // value). The best jump minimises f; score is the reduction -min f.
    auto f = [&](double j) { return vm.weighted_violation_delta(var_id, j); };

    double best_j = x0;
    double best_f = 0.0;  // f(x0) == 0
    auto consider = [&](double j) {
        if (j == x0) {
            return;
        }
        double fv = f(j);
        if (fv < best_f) {
            best_f = fv;
            best_j = j;
        }
    };

    if (var.type == VarType::Bool) {
        consider(1.0 - x0);
    } else if (var.type == VarType::Int) {
        // Each consider() runs one weighted_violation_delta (two delta_evaluate
        // passes), so a small domain is scanned exhaustively and a large one on
        // a coarse grid. The closed-form linear-constraint argmin (paper) is a
        // deferred optimisation; the JumpTable cache means this is computed at
        // most once per variable between neighbour changes.
        const long lb = std::lround(var.lb);
        const long ub = std::lround(var.ub);
        if (ub > lb && ub - lb <= 256) {
            for (long v = lb; v <= ub; ++v) {
                consider(static_cast<double>(v));
            }
        } else if (ub > lb) {
            // Large domain: neighbours, endpoints, and a fixed coarse grid.
            consider(static_cast<double>(lb));
            consider(static_cast<double>(ub));
            consider(clamp_to_domain(var, x0 - 1));
            consider(clamp_to_domain(var, x0 + 1));
            const int grid = 32;
            for (int k = 1; k < grid; ++k) {
                double frac = static_cast<double>(k) / grid;
                consider(std::round(lb + frac * (ub - lb)));
            }
        }
    } else if (var.type == VarType::Float) {
        if (var.ub > var.lb) {
            consider(var.lb);
            consider(var.ub);
            consider(golden_section_argmin(f, var.lb, var.ub));
        }
    }

    return {best_j, -best_f};
}

void gls_update_weights(ViolationManager& vm, double rho) {
    const size_t nc = vm.weights.size();
    for (size_t c = 0; c < nc; ++c) {
        vm.weights[c] *= rho;
        // Bump only active (weight > 0) constraints that are currently violated;
        // masked constraints (weight 0) stay 0.
        if (vm.weights[c] > 0.0 && vm.constraint_violation(static_cast<int>(c)) > kTol) {
            vm.weights[c] += 1.0;
        }
    }
    vm.invalidate_cache();
}

// ---------------------------------------------------------------------------
// FeasibilityJump
// ---------------------------------------------------------------------------

FeasibilityJump::FeasibilityJump(Model& model, ViolationManager& vm, RNG& rng, GFJConfig config)
    : model_(model), vm_(vm), rng_(rng), config_(config), jumps_(model.num_vars()) {
    const size_t nc = model_.constraint_ids().size();
    violated_.assign(nc, 0);
    in_queue_.assign(model_.num_vars(), 0);
    is_linear_.assign(nc, 0);
    vars_of_constraint_.assign(nc, {});

    compute_linear_constraints();

    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v)) {
            continue;
        }
        for (int32_t c : model_.constraints_of_var(v)) {
            vars_of_constraint_[c].push_back(v);
        }
    }
}

bool FeasibilityJump::jumpable(int32_t var_id) const {
    auto t = model_.var(var_id).type;
    return t == VarType::Bool || t == VarType::Int || t == VarType::Float;
}

bool FeasibilityJump::active(int32_t constraint_idx) const {
    return vm_.weights[constraint_idx] > 0.0;
}

bool FeasibilityJump::participates_in_active_violated(int32_t var_id) const {
    for (int32_t c : model_.constraints_of_var(var_id)) {
        if (violated_[c] && active(c)) {
            return true;
        }
    }
    return false;
}

void FeasibilityJump::enqueue(int32_t var_id) {
    if (!in_queue_[var_id]) {
        in_queue_[var_id] = 1;
        queue_.push_back(var_id);
    }
}

void FeasibilityJump::compute_linear_constraints() {
    const auto& nodes = model_.nodes();
    const size_t nn = nodes.size();
    std::vector<uint8_t> is_const(nn, 0);
    std::vector<uint8_t> is_affine(nn, 0);

    auto child_const = [&](const ChildRef& c) -> bool {
        return c.is_var ? false : static_cast<bool>(is_const[c.id]);
    };
    auto child_affine = [&](const ChildRef& c) -> bool {
        return c.is_var ? true : static_cast<bool>(is_affine[c.id]);
    };

    // topo_order has children before parents.
    for (int32_t nid : model_.topo_order()) {
        const ExprNode& nd = nodes[nid];
        bool all_const = true;
        for (const auto& ch : nd.children) {
            if (!child_const(ch)) {
                all_const = false;
                break;
            }
        }
        bool affine = all_const;  // a constant subtree is affine
        if (!affine) {
            switch (nd.op) {
                case NodeOp::Const:
                case NodeOp::Neg:
                case NodeOp::Sum:
                    affine = true;
                    for (const auto& ch : nd.children) {
                        if (!child_affine(ch)) {
                            affine = false;
                            break;
                        }
                    }
                    break;
                case NodeOp::Prod:  // affine if at most one child non-constant
                    affine = (child_const(nd.children[0]) && child_affine(nd.children[1])) ||
                             (child_const(nd.children[1]) && child_affine(nd.children[0]));
                    break;
                case NodeOp::Div:  // affine / const
                    affine = child_affine(nd.children[0]) && child_const(nd.children[1]);
                    break;
                case NodeOp::Leq:
                case NodeOp::Geq:
                case NodeOp::Lt:
                case NodeOp::Gt:  // residual lhs-rhs is affine if both sides affine
                    affine = child_affine(nd.children[0]) && child_affine(nd.children[1]);
                    break;
                default:  // Eq (abs), Neq (step), Pow, Min, Max, trig, etc.
                    affine = false;
                    break;
            }
        }
        is_const[nid] = static_cast<uint8_t>(all_const);
        is_affine[nid] = static_cast<uint8_t>(affine);
    }

    const auto& cids = model_.constraint_ids();
    for (size_t c = 0; c < cids.size(); ++c) {
        is_linear_[c] = is_affine[cids[c]];
    }
}

void FeasibilityJump::set_initial_assignment() {
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v)) {
            continue;
        }
        Variable& var = model_.var_mut(v);
        double target = clamp_to_domain(var, 0.0);
        if (var.type == VarType::Bool || var.type == VarType::Int) {
            target = std::round(target);
        }
        var.value = target;
    }
}

void FeasibilityJump::rebuild_violated_and_scan_set() {
    const auto& cids = model_.constraint_ids();
    const size_t nc = cids.size();
    for (size_t c = 0; c < nc; ++c) {
        violated_[c] = is_violated(model_.node(cids[c]).value);
    }
    std::fill(in_queue_.begin(), in_queue_.end(), 0);
    queue_.clear();
    for (size_t c = 0; c < nc; ++c) {
        if (violated_[c] && active(static_cast<int32_t>(c))) {
            for (int32_t v : vars_of_constraint_[c]) {
                enqueue(v);
            }
        }
    }
    jumps_.invalidate_all();
}

void FeasibilityJump::update_var(int32_t var_id) {
    const double j = jumps_.jump_value(var_id);
    model_.var_mut(var_id).value = j;
    delta_evaluate(model_, &var_id, 1);
    jumps_.invalidate(var_id);

    const auto& cids = model_.constraint_ids();
    const auto& gv = model_.constraints_of_var(var_id);
    for (int32_t c : gv) {
        violated_[c] = is_violated(model_.node(cids[c]).value);
    }
    for (int32_t c : gv) {
        for (int32_t vp : vars_of_constraint_[c]) {
            if (vp == var_id) {
                continue;
            }
            jumps_.invalidate(vp);
            if (participates_in_active_violated(vp)) {
                enqueue(vp);
            }
        }
    }
}

bool FeasibilityJump::apply_jump(int sample_size) {
    // Sample up to `sample_size` DISTINCT variables from the scan set Q and
    // apply the best improving jump (paper Algorithm 2). Variables with a
    // non-positive score are removed from Q permanently (swap-remove); positive
    // ones stay. `examined` keeps the sample distinct; the draw cap is a
    // backstop when fewer than sample_size distinct positives remain.
    int32_t best_v = -1;
    double best_score = 0.0;
    int n = 0;
    int draws = 0;
    const int max_draws = sample_size * 8 + 16;
    examined_.clear();
    while (!queue_.empty() && n < sample_size && draws < max_draws) {
        ++draws;
        size_t idx = static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(queue_.size())));
        int32_t v = queue_[idx];
        if (std::find(examined_.begin(), examined_.end(), v) != examined_.end()) {
            continue;  // already sampled this call; redraw for a distinct var
        }
        if (!jumps_.valid(v)) {
            JumpResult r = compute_var_jump(model_, vm_, v);
            jumps_.set(v, r.jump_value, r.score);
        }
        double s = jumps_.score(v);
        if (s <= 0.0) {
            in_queue_[v] = 0;
            queue_[idx] = queue_.back();
            queue_.pop_back();
            continue;
        }
        examined_.push_back(v);
        if (s > best_score) {
            best_score = s;
            best_v = v;
        }
        ++n;
    }
    if (best_v < 0) {
        return false;
    }
    update_var(best_v);
    return true;
}

bool FeasibilityJump::any_active_violated() const {
    const size_t nc = violated_.size();
    for (size_t c = 0; c < nc; ++c) {
        if (violated_[c] && active(static_cast<int32_t>(c))) {
            return true;
        }
    }
    return false;
}

// The GLS inner loop (ApplyJump + stagnation weight bump), reusing the current
// state (Q, weights, jump table). Returns Feasible as soon as no active
// constraint is violated; Unsolved when the batch / global iteration budget or
// the deadline is hit. batch_iter_limit <= 0 means "no per-call limit".
GFJStatus FeasibilityJump::gls_loop(int sample_size, int64_t batch_iter_limit) {
    const size_t nc = model_.constraint_ids().size();
    int64_t batch_iters = 0;

    while (true) {
        if (!apply_jump(sample_size)) {
            if (!any_active_violated()) {
                return GFJStatus::Feasible;
            }
            gls_update_weights(vm_, config_.rho);
            for (size_t c = 0; c < nc; ++c) {
                if (violated_[c] && active(static_cast<int32_t>(c))) {
                    for (int32_t v : vars_of_constraint_[c]) {
                        jumps_.invalidate(v);
                        enqueue(v);
                    }
                }
            }
        }

        ++iterations_;
        ++batch_iters;
        if (batch_iter_limit > 0 && batch_iters >= batch_iter_limit) {
            return any_active_violated() ? GFJStatus::Unsolved : GFJStatus::Feasible;
        }
        if (config_.max_iterations > 0 && iterations_ >= config_.max_iterations) {
            return GFJStatus::Unsolved;
        }
        if (has_deadline_ && std::chrono::steady_clock::now() >= deadline_) {
            return GFJStatus::Unsolved;
        }
    }
}

GFJStatus FeasibilityJump::gls(int sample_size) {
    rebuild_violated_and_scan_set();
    return gls_loop(sample_size, 0);
}

// ---- Batch API (drives ViolationLS Algorithm 6 from an outer loop) ----

void FeasibilityJump::begin(bool set_initial_x) {
    iterations_ = 0;
    has_deadline_ = config_.time_limit > 0.0;
    if (has_deadline_) {
        deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(config_.time_limit));
    }
    if (set_initial_x) {
        set_initial_assignment();
    }
    full_evaluate(model_);
    std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
    vm_.invalidate_cache();
    rebuild_violated_and_scan_set();
}

bool FeasibilityJump::batch(int64_t batch_iterations) {
    return gls_loop(config_.sample_size_general, batch_iterations) == GFJStatus::Feasible;
}

void FeasibilityJump::reset_weights() {
    std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
    vm_.invalidate_cache();
    rebuild_violated_and_scan_set();
}

void FeasibilityJump::perturb(double probability) {
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v) || rng_.random() >= probability) {
            continue;
        }
        Variable& var = model_.var_mut(v);
        switch (var.type) {
            case VarType::Bool:
                var.value = static_cast<double>(rng_.integers(0, 2));
                break;
            case VarType::Int:
                var.value = static_cast<double>(
                    rng_.integers(static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
                break;
            default:  // Float
                var.value = rng_.uniform(var.lb, var.ub);
                break;
        }
    }
    full_evaluate(model_);
    reset_weights();
}

bool FeasibilityJump::all_satisfied() const {
    return !any_active_violated();
}

GFJStatus FeasibilityJump::run() {
    iterations_ = 0;
    has_deadline_ = config_.time_limit > 0.0;
    if (has_deadline_) {
        deadline_ = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(config_.time_limit));
    }

    if (config_.set_initial_x) {
        set_initial_assignment();
    }
    full_evaluate(model_);
    vm_.invalidate_cache();

    const size_t nc = model_.constraint_ids().size();
    bool has_nonlinear =
        std::any_of(is_linear_.begin(), is_linear_.end(), [](uint8_t lin) { return lin == 0; });

    if (config_.two_phase && has_nonlinear) {
        // Phase 1: GLS on the linear submodel. Non-linear constraint weights are
        // masked to 0 (a mask, not a learned weight); active() == weight>0 then
        // excludes them, and gls_update_weights leaves 0-weights at 0. Phase 2
        // restores all weights to 1 (the paper uses fresh weights per phase).
        for (size_t c = 0; c < nc; ++c) {
            vm_.weights[c] = is_linear_[c] ? 1.0 : 0.0;
        }
        vm_.invalidate_cache();
        gls(config_.sample_size_linear);
        // Restore full weights for the general phase.
        std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
        vm_.invalidate_cache();
    } else {
        std::fill(vm_.weights.begin(), vm_.weights.end(), 1.0);
        vm_.invalidate_cache();
    }

    GFJStatus status =
        gls(config_.two_phase ? config_.sample_size_general : config_.sample_size_linear);
    vm_.invalidate_cache();
    return status;
}

}  // namespace cbls
