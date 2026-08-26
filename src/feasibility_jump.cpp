#include "cbls/feasibility_jump.h"

#include "cbls/dag_ops.h"
#include "cbls/moves.h"
#include "cbls/randomize.h"

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

// --- perturbation helpers --------------------------------------------------
// `random_in_domain` itself lives in randomize.h, shared with search.cpp's
// initialisers and LNS's destroy step (#112). The two below are specific to the
// kick and stay here; they read the domain through the same `domain_window`, so
// the values they consider in-domain are exactly the ones it can draw.
//
// A variable can be moved by a perturbation only if its domain holds at least
// two values. Bool spans {0,1} today, but ask its bounds rather than assume so:
// should a Bool ever become pinnable, flipping it would put it outside its own
// domain. Int truncates its bounds the same way random_in_domain does, so both
// paths agree on which values the domain contains.
bool movable_domain(const Variable& var) {
    const DomainWindow w = domain_window(var);
    switch (var.type) {
        case VarType::Int:
            return static_cast<int64_t>(w.hi) > static_cast<int64_t>(w.lo);
        default:  // Bool, Float
            return w.hi > w.lo;
    }
}

// Draw a random value from the domain that DIFFERS from the current one, for
// the single variable a perturbation is guaranteed to move (#109). Plain
// resampling is not enough: it redraws the current value with probability
// 1/|domain|, which on a Bool is one kick in two. Returns the current value
// unchanged only for a pinned domain, where no move exists.
double random_different_in_domain(const Variable& var, RNG& rng) {
    if (!movable_domain(var)) {
        return var.value;
    }
    const DomainWindow w = domain_window(var);
    switch (var.type) {
        case VarType::Bool:
            return var.value != 0.0 ? 0.0 : 1.0;
        case VarType::Int: {
            const int64_t lb = static_cast<int64_t>(w.lo);
            const int64_t ub = static_cast<int64_t>(w.hi);
            const int64_t cur = static_cast<int64_t>(var.value);
            if (cur < lb || cur > ub) {
                return static_cast<double>(rng.integers(lb, ub + 1));  // any value differs
            }
            // Uniform over the domain minus the current value: draw from a
            // domain one value short, then step over the hole at `cur`.
            int64_t draw = rng.integers(lb, ub);  // [lb, ub-1]
            if (draw >= cur) {
                ++draw;
            }
            return static_cast<double>(draw);
        }
        default: {  // Float
            const double v = rng.uniform(w.lo, w.hi);
            if (v != var.value) {
                return v;
            }
            // Measure-zero in exact arithmetic, but a narrow enough domain makes
            // it reachable; fall back to the endpoint further from the current
            // value, which differs because the domain holds more than one point.
            return (var.value - w.lo >= w.hi - var.value) ? w.lo : w.hi;
        }
    }
}

// --- structural perturbation helpers ---------------------------------------
// List and Set variables are not jumpable, so none of the above can reach them:
// a kick on a model whose decisions live in structural variables randomised
// nothing, burned the stagnation counter and left the search exactly where it
// was (#111). They get their own pass, built out of the same typed move
// generators the STRUCTURAL batch uses (moves.cpp) rather than fresh mutation
// code — so the kick explores exactly the neighbourhood the search knows how to
// evaluate, and every move it applies is legal by construction: a List stays a
// permutation of its elements, a Set stays inside min_size/max_size.

// How many random structural moves a kick applies to one variable.
//
// `perturbation_probability` has to keep governing how much of the model moves,
// so scale with the variable's own size: k = round(p * |elements|) applies a p
// fraction of the structure's size in MOVES. That is not the same as displacing
// a p fraction of its slots, and the gap is large for a List: list_2opt
// reverses a random sub-range (mean ~n/3), so k = 0.1n moves rewrite ~98% of
// positions on n = 1000 while breaking ~26% of adjacent pairs. The adjacency
// figure is the one that tracks p, so the scaling is right for a List read
// pairwise (pair_lambda_sum) and much coarser than p
// suggests for one read positionally (`at`).
//
// A structure has no "randomise the whole variable" analogue that is not a
// restart, so the probability sets *how much* of each structure moves rather
// than *which* structures move — which is also what keeps a structure smaller
// than 1/p slots from never moving at all.
//
// The size is the CURRENT membership, which for a List is its whole decision
// content but for a Set is not: a 3-of-1000 Set gets a kick sized on the 3, not
// on the universe. That is deliberate — the kick then rewrites a p fraction of
// the set's *state*, and p = 1 rewrites all of it — but it does mean a sparse
// Set cannot grow far in a single kick. LNS, which resamples the cardinality
// outright, is the mechanism for that.
//
// Clamped to at least one move, so a kick on a model whose decisions are all
// structural is never a no-op (#111), and to at most one move per slot, which is
// already a full scramble, so a misconfigured p > 1 cannot turn a kick into
// unbounded work. The comparison is written to reject NaN.
int32_t structural_kick_size(const Variable& var, double probability) {
    const int32_t n = static_cast<int32_t>(var.elements.size());
    const double scaled = std::round(probability * static_cast<double>(n));
    if (!(scaled > 1.0)) {
        return 1;
    }
    // scaled > 1 implies probability * n > 1, hence n >= 1: an empty structure
    // always took the early return, so the clamp needs no guard for it.
    return static_cast<int32_t>(std::min(scaled, static_cast<double>(n)));
}

// Did a run of structural moves leave `var` somewhere the search can tell apart
// from `before`? "Somewhere else" is type-dependent, and comparing the raw
// vectors gets a Set wrong.
//
// For a List, order IS the decision content: the DAG reads it positionally
// (`at`) and pairwise (`pair_lambda_sum`), so vector inequality is exactly the
// question. For a Set, `elements` is unordered membership — Count and Lambda
// both read it order-insensitively — so a run that removes an element and adds
// it back lands on a permuted vector holding the identical set. Calling that
// "changed" hands back a kick that moved nothing and skips the never-a-no-op
// fallback, which is the defect #111 exists to prevent, just arrived at
// sideways. Measured at 0.5% of kicks on a universe-30 Set at the default p.
bool structure_moved(const Variable& var, const std::vector<int32_t>& before) {
    if (var.type != VarType::Set) {
        return var.elements != before;
    }
    if (var.elements.size() != before.size()) {
        return true;
    }
    std::vector<int32_t> a = var.elements;
    std::vector<int32_t> b = before;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    return a != b;
}

// Apply one uniformly chosen structural move to `var_id` and report whether the
// variable's elements actually changed. False means this draw offered nothing
// that moves the variable — in practice a structural dead end (a List shorter
// than two elements, a Set with no legal add, remove or swap), so the caller
// stops rather than redraw. That is a heuristic rather than a proof: the
// generator picks its positions at random, so on a List holding duplicate
// elements a later draw could in principle differ.
//
// Candidates that leave the elements as they are — a relocate to the adjacent
// position reinserts the element where it was — are dropped before the draw, so
// a kick cannot silently lose a move to one.
bool apply_random_structural_move(Model& model, int32_t var_id, RNG& rng) {
    std::vector<Move> moves = generate_standard_moves(model.var(var_id), rng);
    const std::vector<int32_t>& current = model.var(var_id).elements;
    moves.erase(std::remove_if(moves.begin(), moves.end(),
                               [&current](const Move& m) {
                                   return m.changes.size() != 1 ||
                                          m.changes.front().new_elements == current;
                               }),
                moves.end());
    if (moves.empty()) {
        return false;
    }
    const auto pick = static_cast<size_t>(rng.integers(0, static_cast<int64_t>(moves.size())));
    apply_move(model, moves[pick]);
    return true;
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
// Returns whether the gradient carried usable direction here. False means this
// variable takes part in at least one violated constraint yet none of them
// yielded a Newton candidate — every gradient was ~0 (or non-finite, which
// fails the same test), i.e. it sits at a
// *stationary* point — so the three remaining candidates are box constants
// unrelated to local geometry and the variable may have no move at all.
//
// A variable in no violated constraint reports true: nothing to escape.
template <class Consider>
bool float_jump_candidates(Model& model, int32_t var_id, const Variable& var, double x0,
                           Consider&& consider) {
    if (var.ub <= var.lb) {
        return true;  // fixed: nothing to escape
    }
    const auto& cids = model.constraint_ids();
    int budget = 4;
    bool any_newton = false;
    bool saw_violated = false;
    for (int32_t c : model.constraints_of_var(var_id)) {
        if (budget <= 0) {
            break;
        }
        double residual = model.node(cids[c]).value;
        if (residual <= kTol) {
            continue;  // satisfied: no root to chase
        }
        saw_violated = true;
        double grad = compute_partial(model, cids[c], var_id);
        if (std::abs(grad) > 1e-12) {
            consider(clamp_to_domain(var, x0 - residual / grad));
            any_newton = true;
            --budget;
        }
    }
    consider(0.5 * (var.lb + var.ub));
    consider(var.lb);
    consider(var.ub);
    return !saw_violated || any_newton;
}

// Relative probe steps. 1e-6 breaks an *exact* stationary point, where the
// first-order model says nothing and any nonzero step is information; 1e-2
// then covers ground, because the gain at a quadratic stationary point is
// O(h^2) and a 1e-6 step alone makes the search crawl.
constexpr double kEscapeRelSteps[] = {1e-6, 1e-2};

// The local move Float otherwise lacks entirely. `int_jump_candidates` always
// offers x0 +/- 1; Float had only a Newton step (length set by the target, and
// vanishing with the gradient) plus three box constants, so a Float at an
// interior stationary point had an empty neighbourhood and froze there.
//
// Two-sided because at a saddle the descent direction is exactly what a zero
// gradient cannot supply — it has to be sampled.
template <class Consider>
void float_escape_candidates(const Variable& var, double x0, Consider&& consider) {
    // Scaled on |x0|, deliberately NOT on box width: for an unbounded NL column
    // that width is the inf_clamp artifact rather than information. The +1 keeps
    // the step nonzero at x0 == 0 and keeps x0 +/- h distinct from x0.
    const double scale = std::abs(x0) + 1.0;
    for (double rel : kEscapeRelSteps) {
        const double h = rel * scale;
        const double up = clamp_to_domain(var, x0 + h);
        const double down = clamp_to_domain(var, x0 - h);
        if (up != var.ub) {  // lb/ub are already candidates; don't pay twice
            consider(up);
        }
        if (down != var.lb) {
            consider(down);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

JumpResult compute_var_jump(Model& model, const std::vector<double>& weights, int32_t var_id,
                            bool allow_escape_probe) {
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
        // The probe is a LAST RESORT and must stay one. Firing it whenever a
        // variable is stationary and nothing improved — which is the steady
        // state of local search — measured ~9x worse on shiporig across every
        // seed: the drip of tiny improvements suppresses stagnation, so
        // diversification never fires. Hence `allow_escape_probe`, which the
        // search loop arms only once it is genuinely stuck.
        const bool gradient_usable = float_jump_candidates(model, var_id, var, x0, consider);
        if (allow_escape_probe && !gradient_usable && best_f >= 0.0) {
            float_escape_candidates(var, x0, consider);
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

// The complement of `is_structured` (dag.h): between them the two partition
// VarType, and `solve()` relies on that to initialise every variable exactly once
// (#108) — FJ sets the scalars here, `initialize_structured_random` sets the rest.
// Deliberately a whitelist, not `!is_structured(t)`: a VarType added later must
// opt in to being jumped rather than default into it.
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
            JumpResult r = compute_var_jump(model_, vm_.weights, v, escape_probe_);
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

// Next deadline-check stride, given how long the last one actually took.
//
// Growth is capped at kStrideGrowth so the stride ramps up through
// progressively longer — and therefore more accurate — measurements instead of
// extrapolating the whole way from one short one. The cap never weakens the
// bound: an x8 growth is only taken when 8 x elapsed is still inside the
// target, so the predicted duration of the next stride is <= target either way.
//
// Shrinking is deliberately NOT capped: iterations that got more expensive must
// be caught on the very next check.
//
// An uncapped shrink is NOT, on its own, what stops the stride ratcheting upward
// and going silent — the failure mode that got an earlier self-tuning stride
// removed from this engine. It cannot be, because a shrink is only APPLIED at a
// check and the next check is a whole stride away: a stride grown while
// iterations were cheap is spent in full on the first expensive one, and the
// tuner learns nothing until after the damage. What bounds that is
// kMaxDeadlineStride, the hard iteration cap; see its comment.
int64_t FeasibilityJump::next_deadline_stride(int64_t stride, double elapsed_seconds,
                                              double target_seconds) {
    // A non-finite measurement carries no information; take the floor rather
    // than the growth cap, which is what an `elapsed > 0.0` test alone would
    // silently do with a NaN. Unreachable from a monotonic steady_clock, so this
    // is defensive, not load-bearing.
    if (std::isnan(elapsed_seconds) || std::isnan(target_seconds)) {
        return 1;
    }
    // elapsed <= 0 means the interval was below the clock's resolution, i.e.
    // far inside the target: grow by the cap.
    double scale = static_cast<double>(kStrideGrowth);
    if (elapsed_seconds > 0.0) {
        scale = std::min(scale, target_seconds / elapsed_seconds);
    }
    const double next = static_cast<double>(stride) * scale;
    if (!(next > 1.0)) {
        return 1;  // never fewer than one iteration per check
    }
    if (next >= static_cast<double>(kMaxDeadlineStride)) {
        return kMaxDeadlineStride;
    }
    return static_cast<int64_t>(next);
}

// Arm (or disarm) the wall-clock deadline and reset the stride tuner. Called
// from both entry points, begin() and run(), so the tuner state cannot be left
// stale from a previous run.
void FeasibilityJump::arm_deadline() {
    has_deadline_ = config_.time_limit > 0.0;
    deadline_checks_ = 0;
    if (!has_deadline_) {
        return;  // no clock is read, and no clock-derived state exists, at all
    }
    const auto now = std::chrono::steady_clock::now();
    deadline_ = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                          std::chrono::duration<double>(config_.time_limit));
    last_deadline_check_ = now;
    // Start at one iteration and let the tuner grow it. The first stride is the
    // one no measurement has bounded yet, and starting it at 64 on a model whose
    // iterations cost 100ms is the whole of #113.
    deadline_stride_ = 1;
    deadline_countdown_ = 1;
}

// The GLS inner loop (ApplyJump + stagnation weight bump), reusing the current
// state (Q, weights, jump table). Returns Feasible as soon as no active
// constraint is violated; Unsolved when the batch / global iteration budget or
// the deadline is hit. batch_iter_limit <= 0 means "no per-call limit".
//
// ---- How the deadline is observed (#113) ----
//
// Reading the clock is not free: steady_clock::now() measures 1408 ns/call on
// this project's reference machine, whose clocksource is HPET (it is ~20-25 ns
// through the vDSO on a TSC clocksource), against GLS iterations that can be a
// few microseconds. Checking every iteration measured 2996 -> 5228 ns per
// iteration, a 1.75x throughput loss, on a small Bool model — so the loop
// cannot simply check every time.
//
// It used to check on a fixed stride of 64 iterations, which is not a time
// bound at all: one GLS iteration is O(sampled vars x candidate values x
// constraints touched), so 64 of them are microseconds on a small model and
// seconds on a large one. Measured on 400 Int vars with 20 000 rows of 8: every
// budget from 0.05s to 3s took ~7s, always after exactly 64 iterations.
//
// So the stride is sized in *time* instead. Each check measures how long the
// previous stride took, which is the current cost of an iteration, and sizes
// the next stride to kStrideBudgetFraction of the total budget. The guarantee:
//
//   a batch returns at most one stride past the deadline, and a stride costs at
//   most 1/64 of the budget -- or one GLS iteration, whichever is larger,
//   because an iteration is atomic and cannot be pre-empted from the inside.
//
// Sizing the stride in time also bounds the clock overhead without having to
// measure the clock at all: one read per stride, against a stride costing
// budget/64, is 1.4us / (budget/64) even on the expensive clocksource — 0.45%
// of a 20ms budget, 0.009% of a 1s one. It degrades only for budgets so small
// (well under a millisecond) that the run is over before throughput matters.
//
// Two honest caveats. The prediction is a measurement, so an iteration whose
// cost jumps mid-stride overruns the target and is corrected only at the next
// check. And the interval between checks can span a batch boundary, so work the
// *outer* loop does between batches (hook, LNS, structural sweep) is charged to
// the stride, which shrinks it; that is conservative, never the reverse.
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
        // Short-circuited on has_deadline_, so a run with no wall clock neither
        // reads the clock nor touches any of the tuner state: iteration-budgeted
        // runs stay bit-identical.
        if (has_deadline_ && --deadline_countdown_ <= 0) {
            const auto now = std::chrono::steady_clock::now();
            ++deadline_checks_;  // every clock read, including the one that stops the run
            if (now >= deadline_) {
                return GFJStatus::Unsolved;
            }
            // Size the next stride against the budget that is LEFT, not the
            // budget that was given. A fraction of the total permits an overrun
            // of budget/64 right up to the deadline — 9.4 s on a 600 s
            // benchmark run, by design — whereas remaining/64 tightens as the
            // deadline approaches and costs nothing to compute.
            const double remaining = std::chrono::duration<double>(deadline_ - now).count();
            deadline_stride_ = next_deadline_stride(
                deadline_stride_, std::chrono::duration<double>(now - last_deadline_check_).count(),
                remaining * kStrideBudgetFraction);
            deadline_countdown_ = deadline_stride_;
            last_deadline_check_ = now;
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
    // A fresh run starts with the escape probe disarmed, alongside the iteration
    // count and the deadline. solve() constructs a FeasibilityJump per call so
    // this cannot matter today; it is here so a caller that reuses one instance
    // does not inherit the previous run's stagnation state (#117).
    escape_probe_ = false;
    arm_deadline();
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

int32_t FeasibilityJump::pick_forced_perturb_var() {
    const int32_t num_vars = static_cast<int32_t>(model_.num_vars());
    auto eligible = [this](int32_t v) { return jumpable(v) && movable_domain(model_.var(v)); };

    int32_t count = 0;
    for (int32_t v = 0; v < num_vars; ++v) {
        count += eligible(v) ? 1 : 0;
    }
    if (count == 0) {
        return -1;  // nothing can move; a no-op kick is the correct outcome
    }
    // Two passes rather than materialising the candidate list: the O(n) scan is
    // dwarfed by the full_evaluate the perturbation ends with.
    int64_t k = rng_.integers(0, count);
    for (int32_t v = 0; v < num_vars; ++v) {
        if (!eligible(v)) {
            continue;
        }
        if (k == 0) {
            return v;
        }
        --k;
    }
    return -1;  // unreachable: k < count eligible variables were skipped
}

// Reset the structural kick's move counter and stride tuner. Starting the stride
// at one move is the point of the whole bound: the first move of a kick is the
// one no measurement has sized yet, and on a model whose structure is a single
// large Set the entire quadratic run happens inside that first variable. A
// stride inherited from a previous kick would be spent there.
void FeasibilityJump::arm_structural_kick() {
    kick_moves_ = 0;
    kick_checks_ = 0;
    kick_stride_ = 1;
    kick_countdown_ = 1;
    if (has_deadline_) {
        last_kick_check_ = std::chrono::steady_clock::now();
    }
}

// Stop the structural pass? Checked between MOVES, which is what makes the
// bound mean anything on a model with one large structure (#115).
//
// The unit here is one structural move — one move-set generation plus one apply,
// O(|elements| + universe) element copies — which is the same unit the STRUCTURAL
// batch's sweep caps its overrun at (#105). Checking between *variables*, as this
// pass used to, capped nothing: a variable costs k = round(p * |elements|) of
// those moves, quadratic in its size, so a model whose structure lives in one
// List or Set ran the whole quadratic pass and then consulted the clock on its
// way to a variable that did not exist. Measured at 2.3 s for one kick on a
// 41k-element Set, with solve(time_limit=1.0) returning in 1.29 s.
//
// A move is not cheap enough to check the clock before every one of them: the
// shape this pass already handles well is many small structures, where a move on
// a 100-element List is ~1.5us against 1408 ns for steady_clock::now() on this
// project's HPET reference machine. So the check strides, and the stride is the
// one the GLS loop already uses — FeasibilityJump::next_deadline_stride, sized in
// time from the last measurement, growth capped at 8x, shrink uncapped, and hard
// capped at kMaxDeadlineStride moves.
//
// Sharing that tuner shares the lesson it encodes (#113): a stride sized in time
// ALONE goes silent exactly when it is needed, because the shrink can only be
// applied at a check and the next check is a whole stride away, so a stride grown
// over many cheap small structures would be spent in full on the first move of a
// large one. The hard cap is what bounds that, and it is why bounding k against
// the remaining budget instead — the other direction #115 named — was not taken:
// k is chosen once, before the run, so a per-move cost that rises inside the run
// is never re-observed at all.
//
//   Guarantee: perturb()'s structural pass applies at most kMaxDeadlineStride
//   (64) further moves after the deadline passes, and a stride costs at most
//   1/64 of the remaining budget in predicted time — or one move, whichever is
//   larger, since a move is atomic and cannot be pre-empted from the inside.
//
// The prediction is a measurement, so a move whose cost jumps mid-stride is
// absorbed by the 64-move half of the bound, not the time half.
//
// One honest gap, PRE-EXISTING and deliberately not closed here. The guarantee is
// stated in moves, and the cost model behind it assumes cost is proportional to
// moves — but a move that is never applied is not free. generate_standard_moves
// on a Set allocates a vector<bool> over the universe, copies the membership and
// builds the complement, O(|elements| + universe), before discovering there is no
// legal move; the pass then breaks out of that variable having applied nothing.
// Since the check short-circuits on kick_moves_ == 0 without decrementing, a
// model of M saturated Sets (min_size == max_size == universe_size) does O(M * U)
// work with zero clock reads. The old between-variables check was gated on
// `changed` and was equally blind to it, so this is not a regression, and the
// cost is linear in the model rather than quadratic in one variable — the shape
// #115 is about. Closing it means bounding failed ATTEMPTS as well as moves.
bool FeasibilityJump::kick_past_deadline() {
    // Short-circuited on has_deadline_, so a run with no wall clock neither reads
    // the clock nor touches any tuner state: iteration-budgeted runs stay
    // bit-identical. Gated on having applied a move as well, so a deadline
    // already crossed on entry still leaves the kick something to have done —
    // the never-a-no-op contract of #109/#111, which perturb()'s fallback then
    // completes if that one move happened to cancel out.
    if (!has_deadline_ || kick_moves_ == 0) {
        return false;
    }
    if (--kick_countdown_ > 0) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    // Counted before the deadline test — as the GLS loop also does — so the read
    // that stops the pass is included rather than dropped. The count is every
    // read made by THIS check, not by the kick as a whole: arm_structural_kick()
    // takes one more, uncounted. So `structural_kick_checks() == 0` states
    // exactly that the strided check never consulted the clock, and it is the
    // pairing with `has_deadline_ == false` — which also silences the arm — that
    // makes a no-wall-clock run read no clock at all.
    ++kick_checks_;
    if (now >= deadline_) {
        return true;
    }
    // Sized against the budget that is LEFT, as the GLS loop is: a fraction of
    // the total would permit budget/64 of overrun right up to the deadline.
    const double remaining = std::chrono::duration<double>(deadline_ - now).count();
    kick_stride_ = next_deadline_stride(
        kick_stride_, std::chrono::duration<double>(now - last_kick_check_).count(),
        remaining * kStrideBudgetFraction);
    kick_countdown_ = kick_stride_;
    last_kick_check_ = now;
    return false;
}

bool FeasibilityJump::perturb_structural(double probability) {
    // Cost per structural variable is O(k * (|elements| + universe)) element
    // copies, k = round(p * |elements|): the generator builds each candidate as a
    // whole new element vector, and a Set's candidates also scan its universe.
    // That is superlinear in a single structure's size, which is why the deadline
    // is checked between moves rather than between variables — see
    // kick_past_deadline() for the bound that buys and what it costs.
    arm_structural_kick();
    bool changed = false;
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!is_structured(model_.var(v).type)) {
            continue;  // no RNG draw: a scalar-only model keeps its draw sequence
        }
        // Whether the run moved the variable is a question about its NET effect,
        // not about how many moves were applied: a run of two can add an element
        // and remove it again, and calling that "changed" would hand back a kick
        // that changed nothing — the very thing #111 is about.
        const std::vector<int32_t> before = model_.var(v).elements;
        const int32_t k = structural_kick_size(model_.var(v), probability);
        bool out_of_time = false;
        for (int32_t i = 0; i < k; ++i) {
            if (kick_past_deadline()) {
                out_of_time = true;
                break;
            }
            if (!apply_random_structural_move(model_, v, rng_)) {
                break;  // nothing can move this variable; further tries cannot either
            }
            ++kick_moves_;
        }
        // Recorded even when the budget cut the run short: a truncated run still
        // moved the variable, and the caller's never-a-no-op fallback keys off it.
        changed = changed || structure_moved(model_.var(v), before);
        if (out_of_time) {
            break;
        }
    }
    return changed;
}

bool FeasibilityJump::force_structural_move() {
    std::vector<int32_t> structured;
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (is_structured(model_.var(v).type)) {
            structured.push_back(v);
        }
    }
    if (structured.empty()) {
        return false;
    }
    // Walk from a random structure and take the first that moves. Not uniform
    // over the movable ones — one sitting behind a run of dead ends is favoured
    // — but this runs only on a kick that would otherwise have changed nothing,
    // where any movable structure will do. A single applied move always changes
    // the variable: the no-op candidates were filtered out before the draw.
    const size_t n = structured.size();
    const size_t start = static_cast<size_t>(rng_.integers(0, static_cast<int64_t>(n)));
    for (size_t i = 0; i < n; ++i) {
        if (apply_random_structural_move(model_, structured[(start + i) % n], rng_)) {
            return true;
        }
    }
    return false;  // every structure is a dead end
}

void FeasibilityJump::perturb(double probability) {
    // Randomise each jumpable variable independently, then make sure the kick
    // actually moved something. Independent draws alone leave the assignment
    // untouched with probability (1-p)^n, which at the default p = 0.1 is 81% on
    // a two-variable model — exactly the small models most likely to be stuck in
    // one basin, where the kick then burned the stagnation counter and the
    // search resumed where it was (#109).
    //
    // The guarantee is a FALLBACK, not a variable forced on every kick. That
    // matters for fidelity: on a model big enough for the per-variable
    // probability to do its job a no-op kick is vanishingly rare, so the scan
    // below never runs, no extra RNG draw is taken, and the kick keeps exactly
    // the distribution — and the exact draw sequence — it had before. Forcing a
    // variable unconditionally instead would shift the draw sequence on every
    // model.
    bool changed = false;
    for (int32_t v = 0; v < static_cast<int32_t>(model_.num_vars()); ++v) {
        if (!jumpable(v) || rng_.random() >= probability) {
            continue;
        }
        Variable& var = model_.var_mut(v);
        const double previous = var.value;
        var.value = random_in_domain(var, rng_);
        changed = changed || var.value != previous;
    }
    // The loop above only reaches jumpable (scalar) variables. List and Set
    // variables get their own pass, so a kick on a model whose decision
    // structure is structural is a real kick rather than a no-op (#111). The
    // pass draws no random numbers on a model without List/Set variables, so
    // scalar-only models keep the exact draw sequence — and hence the exact
    // runs — they had before.
    const bool structural_changed = perturb_structural(probability);
    changed = changed || structural_changed;
    if (!changed) {
        // Nothing moved: pick one variable that CAN move and move it. Scalars
        // first, because that is the cheap answer and the common one; if every
        // scalar is pinned (-1), force a structure instead — the structural pass
        // reaching here means either that every structure is a dead end, in
        // which case this fails too and a kick that changes nothing is the
        // correct outcome, or that a run of moves cancelled itself out, which a
        // single further move undoes.
        const int32_t forced = pick_forced_perturb_var();
        if (forced >= 0) {
            Variable& var = model_.var_mut(forced);
            var.value = random_different_in_domain(var, rng_);
        } else {
            force_structural_move();
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
    arm_deadline();

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
