#include "cbls/model.h"

#include "cbls/dag_ops.h"
#include "cbls/expr.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace cbls {

// Forward declare from dag_ops.cpp
namespace detail {
std::vector<int32_t> compute_topo_order(Model& model);
}

namespace {
// Mirror ViolationManager's clamp: a non-convex node value that overflows to
// +inf or NaN is mapped to a large finite penalty so jump scoring stays ordered
// and never propagates NaN/inf into the search. Must match violation.cpp.
constexpr double kInfPenalty = 1.0e30;
double clamped_node_violation(double node_value) {
    // NaN before max(): std::max(0.0, NaN) == 0.0 would mask a NaN as satisfied.
    if (std::isnan(node_value)) {
        return kInfPenalty;
    }
    double v = std::max(0.0, node_value);
    if (v > kInfPenalty) {  // also catches +inf
        return kInfPenalty;
    }
    return v;
}
}  // namespace

int32_t Model::alloc_var(VarType type, double lb, double ub, const std::string& name) {
    Variable v;
    v.id = static_cast<int32_t>(vars_.size());
    v.type = type;
    v.value = lb;
    v.lb = lb;
    v.ub = ub;
    v.name = name;
    vars_.push_back(std::move(v));
    return vars_.back().id;
}

int32_t Model::alloc_node(NodeOp op, const std::vector<ChildRef>& children) {
    ExprNode nd;
    nd.id = static_cast<int32_t>(nodes_.size());
    nd.op = op;
    nd.children = children;
    nodes_.push_back(std::move(nd));
    return nodes_.back().id;
}

ChildRef Model::wrap(int32_t handle) {
    // Handle encoding: var handles = -(var_id + 1) (negative), node handles = node_id
    // (non-negative)
    ChildRef ref;
    if (handle < 0) {
        ref.id = -(handle + 1);
        ref.is_var = true;
    } else {
        ref.id = handle;
        ref.is_var = false;
    }
    return ref;
}

// Variable creation methods return negative handles: -(var_id + 1)
int32_t Model::bool_var(const std::string& name) {
    int32_t vid = alloc_var(VarType::Bool, 0.0, 1.0, name);
    return -(vid + 1);  // encode as var handle
}

int32_t Model::int_var(int lb, int ub, const std::string& name) {
    int32_t vid = alloc_var(VarType::Int, static_cast<double>(lb), static_cast<double>(ub), name);
    return -(vid + 1);
}

int32_t Model::float_var(double lb, double ub, const std::string& name) {
    int32_t vid = alloc_var(VarType::Float, lb, ub, name);
    return -(vid + 1);
}

int32_t Model::list_var(int n, const std::string& name) {
    int32_t vid = alloc_var(VarType::List, 0.0, 0.0, name);
    auto& v = vars_[vid];
    v.max_size = n;
    v.elements.resize(n);
    for (int i = 0; i < n; ++i) {
        v.elements[i] = i;
    }
    return -(vid + 1);
}

int32_t Model::set_var(int n, int min_size, int max_size, const std::string& name) {
    int32_t vid = alloc_var(VarType::Set, 0.0, 0.0, name);
    auto& v = vars_[vid];
    v.universe_size = n;
    v.min_size = min_size;
    v.max_size = (max_size < 0) ? n : max_size;
    return -(vid + 1);
}

// Expression creation methods return non-negative handles (node IDs)
int32_t Model::constant(double val) {
    ExprNode nd;
    nd.id = static_cast<int32_t>(nodes_.size());
    nd.op = NodeOp::Const;
    nd.const_value = val;
    nd.value = val;
    nodes_.push_back(std::move(nd));
    return nodes_.back().id;
}

int32_t Model::neg(int32_t x) {
    return alloc_node(NodeOp::Neg, {wrap(x)});
}

int32_t Model::sum(const std::vector<int32_t>& args) {
    if (args.empty()) {
        return constant(0.0);
    }
    std::vector<ChildRef> children;
    children.reserve(args.size());
    for (int32_t a : args) {
        children.push_back(wrap(a));
    }
    return alloc_node(NodeOp::Sum, children);
}

int32_t Model::prod(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Prod, {wrap(a), wrap(b)});
}

int32_t Model::div_expr(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Div, {wrap(a), wrap(b)});
}

int32_t Model::pow_expr(int32_t base, int32_t exp) {
    return alloc_node(NodeOp::Pow, {wrap(base), wrap(exp)});
}

int32_t Model::min_expr(const std::vector<int32_t>& args) {
    std::vector<ChildRef> children;
    for (int32_t a : args) {
        children.push_back(wrap(a));
    }
    return alloc_node(NodeOp::Min, children);
}

int32_t Model::max_expr(const std::vector<int32_t>& args) {
    std::vector<ChildRef> children;
    for (int32_t a : args) {
        children.push_back(wrap(a));
    }
    return alloc_node(NodeOp::Max, children);
}

int32_t Model::abs_expr(int32_t x) {
    return alloc_node(NodeOp::Abs, {wrap(x)});
}

int32_t Model::sin_expr(int32_t x) {
    return alloc_node(NodeOp::Sin, {wrap(x)});
}

int32_t Model::cos_expr(int32_t x) {
    return alloc_node(NodeOp::Cos, {wrap(x)});
}

int32_t Model::tan_expr(int32_t x) {
    return alloc_node(NodeOp::Tan, {wrap(x)});
}

int32_t Model::exp_expr(int32_t x) {
    return alloc_node(NodeOp::Exp, {wrap(x)});
}

int32_t Model::log_expr(int32_t x) {
    return alloc_node(NodeOp::Log, {wrap(x)});
}

int32_t Model::sqrt_expr(int32_t x) {
    return alloc_node(NodeOp::Sqrt, {wrap(x)});
}

int32_t Model::signpower_expr(int32_t base, int32_t exp) {
    return alloc_node(NodeOp::SignPower, {wrap(base), wrap(exp)});
}

int32_t Model::tanh_expr(int32_t x) {
    return alloc_node(NodeOp::Tanh, {wrap(x)});
}

int32_t Model::if_then_else(int32_t cond, int32_t then_, int32_t else_) {
    return alloc_node(NodeOp::If, {wrap(cond), wrap(then_), wrap(else_)});
}

int32_t Model::at(int32_t list_var_handle, int32_t index_expr) {
    return alloc_node(NodeOp::At, {wrap(list_var_handle), wrap(index_expr)});
}

int32_t Model::count(int32_t var_handle) {
    return alloc_node(NodeOp::Count, {wrap(var_handle)});
}

int32_t Model::leq(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Leq, {wrap(a), wrap(b)});
}

int32_t Model::eq_expr(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Eq, {wrap(a), wrap(b)});
}

int32_t Model::geq(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Geq, {wrap(a), wrap(b)});
}

int32_t Model::neq(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Neq, {wrap(a), wrap(b)});
}

int32_t Model::lt(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Lt, {wrap(a), wrap(b)});
}

int32_t Model::gt(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Gt, {wrap(a), wrap(b)});
}

int32_t Model::lambda_sum(int32_t list_var_handle, std::function<double(int)> func) {
    lambda_funcs_.push_back(std::move(func));
    int32_t func_id = static_cast<int32_t>(lambda_funcs_.size() - 1);

    int32_t nid = alloc_node(NodeOp::Lambda, {wrap(list_var_handle)});
    nodes_[nid].lambda_func_id = func_id;
    return nid;
}

int32_t Model::pair_lambda_sum(int32_t list_var_handle, std::function<double(int, int)> func) {
    pair_lambda_funcs_.push_back(std::move(func));
    int32_t func_id = static_cast<int32_t>(pair_lambda_funcs_.size() - 1);

    int32_t nid = alloc_node(NodeOp::PairLambda, {wrap(list_var_handle)});
    nodes_[nid].lambda_func_id = func_id;
    return nid;
}

// Expr-returning variable creation
Expr Model::Bool(const std::string& name) {
    return {this, bool_var(name)};
}

Expr Model::Int(int lb, int ub, const std::string& name) {
    return {this, int_var(lb, ub, name)};
}

Expr Model::Float(double lb, double ub, const std::string& name) {
    return {this, float_var(lb, ub, name)};
}

Expr Model::List(int n, const std::string& name) {
    return {this, list_var(n, name)};
}

Expr Model::Set(int n, int min_size, int max_size, const std::string& name) {
    return {this, set_var(n, min_size, max_size, name)};
}

Expr Model::Constant(double val) {
    return {this, constant(val)};
}

void Model::add_constraint(const Expr& e) {
    add_constraint(e.handle);
}

void Model::minimize(const Expr& e) {
    minimize(e.handle);
}

void Model::maximize(const Expr& e) {
    maximize(e.handle);
}

void Model::add_constraint(int32_t expr_id) {
    if (expr_id < 0) {
        throw std::invalid_argument(
            "add_constraint requires a node handle (non-negative), got var handle");
    }
    constraint_ids_.push_back(expr_id);
}

void Model::minimize(int32_t expr_id) {
    if (expr_id < 0) {
        throw std::invalid_argument(
            "minimize requires a node handle (non-negative), got var handle");
    }
    objective_id_ = expr_id;
}

void Model::maximize(int32_t expr_id) {
    // Maximize by negating
    objective_id_ = neg(expr_id);
    is_maximizing_ = true;
}

void Model::add_var_sequence(std::vector<int32_t> handles, int min_block_on, int min_block_off) {
    int seq_idx = static_cast<int>(var_sequences_.size());
    VarSequence seq;
    seq.min_block_on = min_block_on;
    seq.min_block_off = min_block_off;

    // Convert handles to internal var IDs
    seq.var_ids.reserve(handles.size());
    for (int32_t h : handles) {
        int32_t vid = (h < 0) ? -(h + 1) : h;  // decode var handle
        seq.var_ids.push_back(vid);
    }

    // Grow lookup table if needed
    for (size_t pos = 0; pos < seq.var_ids.size(); ++pos) {
        int32_t vid = seq.var_ids[pos];
        if (vid >= static_cast<int32_t>(var_to_seq_.size())) {
            var_to_seq_.resize(vid + 1, {-1, -1});
        }
        var_to_seq_[vid] = {seq_idx, static_cast<int>(pos)};
    }

    var_sequences_.push_back(std::move(seq));
}

std::pair<int, int> Model::var_sequence_for(int32_t var_id) const {
    if (var_id >= 0 && var_id < static_cast<int32_t>(var_to_seq_.size())) {
        return var_to_seq_[var_id];
    }
    return {-1, -1};
}

void Model::close() {
    topo_order_ = detail::compute_topo_order(*this);
    build_var_constraints();
    full_evaluate(*this);
    closed_ = true;
}

void Model::add_objective_soft_constraint() {
    if (objective_id_ < 0) {
        throw std::invalid_argument("add_objective_soft_constraint requires an objective");
    }
    if (objective_constraint_idx_ >= 0) {
        return;  // idempotent
    }

    objective_bound_ = std::numeric_limits<double>::infinity();
    objective_bound_node_ = constant(objective_bound_);
    // obj - bound <= 0; inert while bound is +inf, tightened during search.
    objective_constraint_node_ = leq(objective_id_, objective_bound_node_);
    objective_constraint_idx_ = static_cast<int32_t>(constraint_ids_.size());
    add_constraint(objective_constraint_node_);

    // Rebuild structure now that a node/constraint was appended after close().
    topo_order_ = detail::compute_topo_order(*this);
    build_var_constraints();
    full_evaluate(*this);
}

void Model::set_objective_bound(double bound) {
    if (objective_constraint_node_ < 0) {
        throw std::logic_error("set_objective_bound requires add_objective_soft_constraint first");
    }
    objective_bound_ = bound;
    ExprNode& bound_node = nodes_[objective_bound_node_];
    bound_node.const_value = bound;
    bound_node.value = bound;
    // Recompute the objective constraint residual in place (obj - bound). Must
    // use the same residual rule as evaluate()'s Leq case, or this shortcut and
    // the next delta_evaluate() would disagree on the row's value — in
    // particular on the `obj = +inf, bound = +inf` state that opens every solve
    // with a blown-up objective (issue #100). The bound side is
    // objective_bound_node_, a Const by construction (see
    // add_objective_soft_constraint), so it is a sentinel; the objective side is
    // a computed expression and never is.
    nodes_[objective_constraint_node_].value =
        comparison_residual(nodes_[objective_id_].value, bound,
                            /*a_is_const=*/false, /*b_is_const=*/true);
}

// Build var_id -> constraint-index adjacency (the paper's G_v) by walking down
// each constraint's subtree and recording every variable it reaches. Stamping
// gives O(1) per-constraint reset and dedups vars/nodes within a constraint.
void Model::build_var_constraints() {
    var_constraints_.assign(vars_.size(), {});
    std::vector<int32_t> node_stamp(nodes_.size(), -1);
    std::vector<int32_t> var_stamp(vars_.size(), -1);
    std::vector<int32_t> stack;
    for (int32_t ci = 0; ci < static_cast<int32_t>(constraint_ids_.size()); ++ci) {
        stack.clear();
        int32_t root = constraint_ids_[ci];
        node_stamp[root] = ci;
        stack.push_back(root);
        while (!stack.empty()) {
            int32_t nid = stack.back();
            stack.pop_back();
            for (const auto& child : nodes_[nid].children) {
                if (child.is_var) {
                    if (var_stamp[child.id] != ci) {
                        var_stamp[child.id] = ci;
                        var_constraints_[child.id].push_back(ci);
                    }
                } else if (node_stamp[child.id] != ci) {
                    node_stamp[child.id] = ci;
                    stack.push_back(child.id);
                }
            }
        }
    }
}

std::vector<std::pair<int32_t, double>> Model::per_constraint_violation_delta(int32_t var_id,
                                                                              double j) {
    const Variable& v = var(var_id);  // bounds-checked
    if (v.type == VarType::List || v.type == VarType::Set) {
        throw std::invalid_argument(
            "per_constraint_violation_delta: scalar variable required (Bool/Int/Float)");
    }

    const auto& affected = constraints_of_var(var_id);
    std::vector<std::pair<int32_t, double>> result;
    if (affected.empty()) {
        return result;
    }

    // Snapshot affected constraints' current violations.
    std::vector<double> old_viol(affected.size());
    for (size_t k = 0; k < affected.size(); ++k) {
        old_viol[k] = clamped_node_violation(node(constraint_ids_[affected[k]]).value);
    }

    // Probe: set candidate, recompute only the affected dirty cone.
    const double old_value = v.value;
    var_mut(var_id).value = j;
    delta_evaluate(*this, &var_id, 1);

    for (size_t k = 0; k < affected.size(); ++k) {
        double new_viol = clamped_node_violation(node(constraint_ids_[affected[k]]).value);
        double delta = new_viol - old_viol[k];
        if (delta != 0.0) {
            result.emplace_back(affected[k], delta);
        }
    }

    // Restore exactly: same inputs through deterministic evaluate() roll node
    // values back to where they were.
    var_mut(var_id).value = old_value;
    delta_evaluate(*this, &var_id, 1);

    return result;
}

double Model::weighted_violation_delta(int32_t var_id, double j,
                                       const std::vector<double>& weights) {
    const Variable& v = var(var_id);  // bounds-checked
    if (v.type == VarType::List || v.type == VarType::Set) {
        throw std::invalid_argument(
            "weighted_violation_delta: scalar variable required (Bool/Int/Float)");
    }
    const auto& affected = constraints_of_var(var_id);
    if (affected.empty()) {
        return 0.0;
    }

    // Accumulate the *per-constraint* differences, rather than differencing two
    // whole-sum accumulators.
    //
    // The two are equal in exact arithmetic but not in floating point, and the
    // difference is the whole ballgame once any one row's violation is large.
    // A non-convex blowup clamps to kInfPenalty = 1e30, some fourteen orders of
    // magnitude above the O(1) contributions of the real rows, so `1e30 + 1`
    // rounds back to `1e30`: both sums collapse to the same value, the
    // subtraction yields exactly 0, and every candidate jump scores identically.
    // Feasibility Jump is then blind — the search cannot tell which move reduces
    // real infeasibility (issue #100). Differencing per constraint makes the
    // huge term cancel exactly (1e30 - 1e30 == 0) and leaves the small terms at
    // full precision.
    //
    // probe_old_violation_ is a member so this stays allocation-free after
    // warm-up (one call per jump candidate). Safe for the same reason the
    // transient node mutation below is: each search thread owns its own Model.
    probe_old_violation_.resize(affected.size());
    for (size_t k = 0; k < affected.size(); ++k) {
        probe_old_violation_[k] = clamped_node_violation(node(constraint_ids_[affected[k]]).value);
    }

    const double old_value = v.value;
    var_mut(var_id).value = j;
    delta_evaluate(*this, &var_id, 1);

    double delta = 0.0;
    for (size_t k = 0; k < affected.size(); ++k) {
        const int32_t c = affected[k];
        const double new_viol = clamped_node_violation(node(constraint_ids_[c]).value);
        delta += weights[c] * (new_viol - probe_old_violation_[k]);
    }

    var_mut(var_id).value = old_value;
    delta_evaluate(*this, &var_id, 1);

    return delta;
}

Model::State Model::copy_state() const {
    State state;
    state.values.resize(vars_.size());
    state.elements.resize(vars_.size());
    for (size_t i = 0; i < vars_.size(); ++i) {
        state.values[i] = vars_[i].value;
        state.elements[i] = vars_[i].elements;
    }
    return state;
}

void Model::restore_state(const State& state) {
    if (state.values.size() != vars_.size()) {
        throw std::invalid_argument("state size does not match model");
    }
    for (size_t i = 0; i < vars_.size(); ++i) {
        vars_[i].value = state.values[i];
        vars_[i].elements = state.elements[i];
    }
}

}  // namespace cbls
