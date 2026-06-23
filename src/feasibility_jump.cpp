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

// Integer jump candidates: exhaustive over a small domain, else a coarse grid
// plus neighbours/endpoints. Each consider() runs one weighted_violation_delta
// (two delta_evaluate passes); the JumpTable cache amortises this across the
// GLS loop. (Closed-form linear-constraint argmin is a deferred optimisation.)
template <class Consider>
void int_jump_candidates(const Variable& var, double x0, Consider&& consider) {
    const long lb = std::lround(var.lb);
    const long ub = std::lround(var.ub);
    if (ub <= lb) {
        return;
    }
    if (ub - lb <= 256) {
        for (long v = lb; v <= ub; ++v) {
            consider(static_cast<double>(v));
        }
        return;
    }
    consider(static_cast<double>(lb));
    consider(static_cast<double>(ub));
    consider(clamp_to_domain(var, x0 - 1));
    consider(clamp_to_domain(var, x0 + 1));
    const int grid = 32;
    for (int k = 1; k < grid; ++k) {
        double frac = static_cast<double>(k) / grid;
        consider(std::round(static_cast<double>(lb) + frac * static_cast<double>(ub - lb)));
    }
}

// Float jump candidates: cheap convex-ish descent — a Newton step toward the
// root of each violated constraint containing v (reverse-mode AD), plus
// midpoint/endpoints. The GLS loop iterates these to converge; the
// InnerSolverHook does the heavy continuous objective polish. This replaces a
// per-jump golden-section (~60 evals) with a handful — critical for throughput
// on continuous-heavy models. Newton candidates come FIRST so that, on a tie in
// violation delta (a feasible plateau), the gradient-informed point wins rather
// than an arbitrary endpoint (`consider` keeps the first-seen minimum). The
// objective enters here too: when `obj <= bound` is violated, chasing its root
// pulls the objective down.
template <class Consider>
void float_jump_candidates(Model& model, int32_t var_id, const Variable& var, double x0,
                           Consider&& consider) {
    if (var.ub <= var.lb) {
        return;
    }
    const auto& cids = model.constraint_ids();
    int budget = 4;
    for (int32_t c : model.constraints_of_var(var_id)) {
        if (budget <= 0) {
            break;
        }
        double residual = model.node(cids[c]).value;
        if (residual <= kTol) {
            continue;  // satisfied: no root to chase
        }
        double grad = compute_partial(model, cids[c], var_id);
        if (std::abs(grad) > 1e-12) {
            consider(clamp_to_domain(var, x0 - residual / grad));
            --budget;
        }
    }
    consider(0.5 * (var.lb + var.ub));
    consider(var.lb);
    consider(var.ub);
}

}  // namespace

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

JumpResult compute_var_jump(Model& model, const std::vector<double>& weights, int32_t var_id) {
    const Variable& var = model.var(var_id);
    const double x0 = var.value;

    // f(j) = weighted violation delta of moving var_id to j (0 at the current
    // value). The best jump minimises f; score is the reduction -min f.
    auto f = [&](double j) { return model.weighted_violation_delta(var_id, j, weights); };

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
        int_jump_candidates(var, x0, consider);
    } else if (var.type == VarType::Float) {
        float_jump_candidates(model, var_id, var, x0, consider);
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
            JumpResult r = compute_var_jump(model_, vm_.weights, v);
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

void FeasibilityJump::resync() {
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

// ---------------------------------------------------------------------------
// Novelty Jump (paper Algorithms 4-5)
// ---------------------------------------------------------------------------

void FeasibilityJump::init_novelty_weights() {
    // W'[c] = W[c] for constraints violated at entry, else kCompoundDiscount*W[c]
    // (the "novelty" weights make breaking a not-violated-since-best constraint
    // cheap, prioritising chains that fix the initially-broken constraints).
    const size_t nc = vm_.weights.size();
    novelty_weights_.resize(nc);
    for (size_t c = 0; c < nc; ++c) {
        novelty_weights_[c] = violated_[c] ? vm_.weights[c] : kCompoundDiscount * vm_.weights[c];
    }
}

void FeasibilityJump::nj_enqueue(int32_t var_id) {
    if (!nj_in_queue_[var_id]) {
        nj_in_queue_[var_id] = 1;
        nj_queue_.push_back(var_id);
    }
}

void FeasibilityJump::seed_novelty_scan_set() {
    std::fill(nj_in_queue_.begin(), nj_in_queue_.end(), 0);
    nj_queue_.clear();
    const auto& cids = model_.constraint_ids();
    for (size_t c = 0; c < cids.size(); ++c) {
        if (violated_[c] && active(static_cast<int32_t>(c))) {
            for (int32_t v : vars_of_constraint_[c]) {
                nj_enqueue(v);
            }
        }
    }
}

// Best of up to 3 sampled vars in Q\T satisfying the filter F (paper §4):
// F = (s_m + novelty_score > 0)  OR  (score > s_c). "Best" = highest original
// score. The chosen var is removed from Q (paper Algorithm 5 line 6).
FeasibilityJump::NoveltyPick FeasibilityJump::select_novelty_var(double s_m, double s_c) {
    NoveltyPick best;
    int sampled = 0;
    int draws = 0;
    const int max_draws = 32;
    examined_.clear();
    while (!nj_queue_.empty() && sampled < 3 && draws < max_draws) {
        ++draws;
        size_t idx = static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(nj_queue_.size())));
        int32_t v = nj_queue_[idx];
        if (on_stack_[v] || std::find(examined_.begin(), examined_.end(), v) != examined_.end()) {
            continue;  // on the stack (T) or already sampled this call
        }
        examined_.push_back(v);
        ++sampled;
        JumpResult nr = compute_var_jump(model_, novelty_weights_, v);  // W'-argmin
        double score = -model_.weighted_violation_delta(v, nr.jump_value, vm_.weights);
        bool passes = (s_m + nr.score > 0.0) || (score > s_c);
        if (passes && (best.var < 0 || score > best.score)) {
            best = {v, nr.jump_value, score, nr.score};
        }
    }
    if (best.var >= 0) {
        // Remove the chosen var from Q (swap-remove).
        for (size_t i = 0; i < nj_queue_.size(); ++i) {
            if (nj_queue_[i] == best.var) {
                nj_in_queue_[best.var] = 0;
                nj_queue_[i] = nj_queue_.back();
                nj_queue_.pop_back();
                break;
            }
        }
    }
    return best;
}

// NoveltyJumpSearch (Algorithm 5), recursive with the explicit move_stack_ for
// T-membership and commit/revert. s_m is the cumulative original-weight score of
// the moves currently on the stack. Returns true once a compound move with
// positive cumulative score is found (left applied); false leaves the assignment
// as it was on entry (every move it applied is reverted).
bool FeasibilityJump::novelty_jump_search(double s_m, int budget) {
    if (budget < 0 || nj_work_remaining_ <= 0) {
        return false;
    }
    double s_c = 0.0;  // best explored child score at this level
    const auto& cids = model_.constraint_ids();
    while (true) {
        NoveltyPick pick = select_novelty_var(s_m, s_c);
        if (pick.var < 0) {
            return false;
        }
        s_c = std::max(s_c, pick.score);

        const int32_t v = pick.var;
        const double old_value = model_.var(v).value;
        model_.var_mut(v).value = pick.jump;
        delta_evaluate(model_, &v, 1);
        move_stack_.push_back({v, old_value});
        on_stack_[v] = 1;
        --nj_work_remaining_;  // bound total moves applied per apply_novelty_jump

        // Refresh violated_ for v's constraints; promote any now-broken
        // constraint to full novelty weight and add its vars to the scan set.
        for (int32_t c : model_.constraints_of_var(v)) {
            violated_[c] = is_violated(model_.node(cids[c]).value);
            if (violated_[c] && novelty_weights_[c] != vm_.weights[c]) {
                novelty_weights_[c] = vm_.weights[c];
                for (int32_t vp : vars_of_constraint_[c]) {
                    nj_enqueue(vp);
                }
            }
        }

        if (s_m + pick.score > 0.0) {
            return true;  // commit (leave applied)
        }
        if (novelty_jump_search(s_m + pick.score, budget)) {
            return true;
        }

        // Backtrack: revert this move and try a sibling (consumes a discrepancy).
        on_stack_[v] = 0;
        move_stack_.pop_back();
        model_.var_mut(v).value = old_value;
        delta_evaluate(model_, &v, 1);
        for (int32_t c : model_.constraints_of_var(v)) {
            violated_[c] = is_violated(model_.node(cids[c]).value);
        }
        budget -= 1;
    }
}

bool FeasibilityJump::apply_novelty_jump() {
    const size_t nv = model_.num_vars();
    nj_in_queue_.assign(nv, 0);
    on_stack_.assign(nv, 0);
    move_stack_.clear();
    nj_work_remaining_ = kNoveltyWorkBudget;  // bound the compound-move search

    int b = 0;
    while (b <= 2) {
        init_novelty_weights();
        seed_novelty_scan_set();
        on_stack_.assign(nv, 0);
        move_stack_.clear();
        while (novelty_jump_search(0.0, b)) {
            if (!any_active_violated()) {
                return true;  // reached feasibility
            }
            // Committed a compound move; start a fresh one from the new state
            // (reset budget per Algorithm 4 line 8, keep evolving W').
            b = 0;
            seed_novelty_scan_set();
            on_stack_.assign(nv, 0);
            move_stack_.clear();
        }
        b += 1;
    }
    return false;
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
