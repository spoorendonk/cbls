#include "cbls/model.h"

#include "cbls/dag_ops.h"
#include "cbls/expr.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace cbls {

// Forward declare from dag_ops.cpp
namespace detail {
std::vector<int32_t> compute_topo_order(const Model& model);
}

namespace {
// Mirror ViolationManager's clamp: a non-convex node value that overflows to
// +inf or NaN is mapped to a large finite penalty so jump scoring stays ordered
// and never propagates NaN/inf into the search. Must match violation.cpp.
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
    // The variable's empty dependents range goes in FIRST: if the push below
    // throws, the offsets are merely one entry longer than needed (still an
    // empty range) rather than one short.
    if (dependent_offsets_.empty()) {
        dependent_offsets_.push_back(0);
    }
    dependent_offsets_.push_back(dependent_offsets_.back());
    vars_.push_back(std::move(v));
    return vars_.back().id;
}

// Append a node whose children are `child_refs_[child_begin ..]`, already
// written by the caller. The one place a node is made, so the two invariants the
// flat storage rests on are kept in one place: the child slice fits the 32-bit
// offsets ExprNode carries, and parent_offsets_ stays one longer than nodes_.
int32_t Model::push_node(NodeOp op, size_t child_begin) {
    if (child_refs_.size() > std::numeric_limits<uint32_t>::max()) {
        child_refs_.resize(child_begin);  // leave the model as it was
        throw std::length_error("model has more than 2^32 - 1 child references");
    }
    ExprNode nd;
    nd.id = static_cast<int32_t>(nodes_.size());
    nd.op = op;
    nd.child_begin = static_cast<uint32_t>(child_begin);
    nd.child_count = static_cast<uint32_t>(child_refs_.size() - child_begin);
    // Offsets before the node, for the reason alloc_var gives.
    if (parent_offsets_.empty()) {
        parent_offsets_.push_back(0);
    }
    parent_offsets_.push_back(parent_offsets_.back());
    nodes_.push_back(nd);
    return nd.id;
}

int32_t Model::alloc_node(NodeOp op, std::initializer_list<ChildRef> children) {
    const size_t begin = child_refs_.size();
    child_refs_.insert(child_refs_.end(), children.begin(), children.end());
    return push_node(op, begin);
}

// The variadic ops' builder: children written straight into the flat array from
// the caller's handles, with no intermediate ChildRef vector.
int32_t Model::alloc_node_over_handles(NodeOp op, const std::vector<int32_t>& handles) {
    // No reserve(size + n) here: libstdc++ reserves exactly what is asked, so
    // doing it per node would defeat geometric growth and copy the whole array
    // on every call.
    const size_t begin = child_refs_.size();
    try {
        for (const int32_t h : handles) {
            child_refs_.push_back(wrap(h));
        }
    } catch (...) {
        child_refs_.resize(begin);  // a rejected node leaves no children behind
        throw;
    }
    return push_node(op, begin);
}

// Handle encoding: var handles = -(var_id + 1) (negative), node handles = node_id
// (non-negative).
//
// Validated here, when the node naming the handle is made, because nothing later
// can be: the back-reference rebuild counts every child into a CSR offsets array
// indexed by its id, so an id past the end is a silent heap write, not an
// exception (#156 -- the per-node vectors it replaced were reached through
// throwing accessors). Python passes raw integers, so this is reachable from a
// typo. It also rules out naming a node before it exists, which the topological
// sort never supported.
ChildRef Model::wrap(int32_t handle) const {
    ChildRef ref;
    if (handle < 0) {
        ref.id = -(handle + 1);
        ref.is_var = true;
        if (static_cast<size_t>(ref.id) >= vars_.size()) {
            throw std::out_of_range("variable handle out of range");
        }
    } else {
        ref.id = handle;
        ref.is_var = false;
        if (static_cast<size_t>(ref.id) >= nodes_.size()) {
            throw std::out_of_range("node handle out of range");
        }
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
    const int32_t nid = push_node(NodeOp::Const, child_refs_.size());
    nodes_[nid].const_value = val;
    nodes_[nid].value = val;
    return nid;
}

int32_t Model::neg(int32_t x) {
    return alloc_node(NodeOp::Neg, {wrap(x)});
}

int32_t Model::sum(const std::vector<int32_t>& args) {
    if (args.empty()) {
        return constant(0.0);
    }
    return alloc_node_over_handles(NodeOp::Sum, args);
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
    return alloc_node_over_handles(NodeOp::Min, args);
}

int32_t Model::max_expr(const std::vector<int32_t>& args) {
    return alloc_node_over_handles(NodeOp::Max, args);
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

int32_t Model::at(int32_t list_var_id, int32_t index_expr) {
    return alloc_node(NodeOp::At, {wrap(list_var_id), wrap(index_expr)});
}

int32_t Model::count(int32_t var_id) {
    return alloc_node(NodeOp::Count, {wrap(var_id)});
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

int32_t Model::lambda_sum(int32_t list_var_id, std::function<double(int)> func) {
    lambda_funcs_.push_back(std::move(func));
    auto func_id = static_cast<int32_t>(lambda_funcs_.size() - 1);

    int32_t nid = alloc_node(NodeOp::Lambda, {wrap(list_var_id)});
    nodes_[nid].lambda_func_id = func_id;
    return nid;
}

int32_t Model::pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func) {
    pair_lambda_funcs_.push_back(std::move(func));
    auto func_id = static_cast<int32_t>(pair_lambda_funcs_.size() - 1);

    int32_t nid = alloc_node(NodeOp::PairLambda, {wrap(list_var_id)});
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
    if (static_cast<size_t>(expr_id) >= nodes_.size()) {
        throw std::out_of_range("add_constraint: node handle out of range");
    }
    constraint_ids_.push_back(expr_id);
}

void Model::minimize(int32_t expr_id) {
    if (expr_id < 0) {
        throw std::invalid_argument(
            "minimize requires a node handle (non-negative), got var handle");
    }
    if (static_cast<size_t>(expr_id) >= nodes_.size()) {
        throw std::out_of_range("minimize: node handle out of range");
    }
    objective_id_ = expr_id;
}

void Model::maximize(int32_t expr_id) {
    // Maximize by negating
    objective_id_ = neg(expr_id);
    is_maximizing_ = true;
}

void Model::add_var_sequence(const std::vector<int32_t>& var_ids, int min_block_on,
                             int min_block_off) {
    int seq_idx = static_cast<int>(var_sequences_.size());
    VarSequence seq;
    seq.min_block_on = min_block_on;
    seq.min_block_off = min_block_off;

    // Callers pass var handles, as returned by bool_var() etc.; decode each to
    // the internal var id that seq.var_ids holds. The decode leaves a
    // non-negative value alone, so a raw var id survives it, but that is a
    // property of the encoding rather than a supported second calling
    // convention -- a non-negative value is indistinguishable from a node id
    // elsewhere in this API, so do not document it as one.
    seq.var_ids.reserve(var_ids.size());
    for (int32_t h : var_ids) {
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

// The inverse permutation of `topo_order_`, rebuilt wherever that is. Its own
// function because the two call sites must not drift: a stale `topo_pos_` would
// evaluate a dirty set out of dependency order, which is silent wrong values
// rather than a crash.
void Model::rebuild_topo_positions() {
    topo_pos_.assign(nodes_.size(), 0);
    for (size_t i = 0; i < topo_order_.size(); ++i) {
        topo_pos_[topo_order_[i]] = static_cast<int32_t>(i);
    }
}

// Rebuild the DAG's back-references: every node's parents and every variable's
// dependents. These are what delta_evaluate walks to find the nodes a changed
// variable dirties, and what the topological sort walks; they are pure derived
// state, so they are recomputed wholesale rather than patched.
//
// Two passes over the edges into CSR -- count, then fill -- so the arrays are
// sized exactly once and nothing is allocated per node. Both passes visit
// parents in ascending id and each parent's children in order, which is the
// order the per-node vectors this replaced were appended in, so every list is
// the same sequence it was.
//
// Deduplicated by a last-writer stamp, not by searching the list being built.
// A duplicate can only ever come from ONE parent naming the same child twice
// (`prod(x, x)`), because a parent is visited once -- so "already recorded by
// this parent" is the whole condition, and a stamp answers it in O(1) where the
// search was O(degree) per edge. That difference is not academic on a real
// matrix: the search made this O(sum of degree^2), which on square47 (95k
// columns in ~288 rows each) was ~3.9 BILLION comparisons and 68% of its model
// build. Each pass needs the stamps fresh, since both skip the same duplicates.
void Model::rebuild_back_references() {
    const size_t n_nodes = nodes_.size();
    const size_t n_vars = vars_.size();
    std::vector<int32_t> node_stamp(n_nodes, -1);
    std::vector<int32_t> var_stamp(n_vars, -1);

    // Pass 1: offsets[i + 1] = number of distinct parents of i.
    parent_offsets_.assign(n_nodes + 1, 0);
    dependent_offsets_.assign(n_vars + 1, 0);
    for (const ExprNode& nd : nodes_) {
        for (const ChildRef& child : children(nd)) {
            if (child.is_var) {
                if (var_stamp[child.id] != nd.id) {
                    var_stamp[child.id] = nd.id;
                    ++dependent_offsets_[child.id + 1];
                }
            } else if (node_stamp[child.id] != nd.id) {
                node_stamp[child.id] = nd.id;
                ++parent_offsets_[child.id + 1];
            }
        }
    }
    std::partial_sum(parent_offsets_.begin(), parent_offsets_.end(), parent_offsets_.begin());
    std::partial_sum(dependent_offsets_.begin(), dependent_offsets_.end(),
                     dependent_offsets_.begin());
    parent_ids_.resize(parent_offsets_.back());
    dependent_ids_.resize(dependent_offsets_.back());

    // Pass 2: fill. offsets[i] serves as i's write cursor, so when the pass is
    // done it has advanced to i's END -- which is offsets[i + 1]'s value -- and
    // one shift right restores the starts.
    std::fill(node_stamp.begin(), node_stamp.end(), -1);
    std::fill(var_stamp.begin(), var_stamp.end(), -1);
    for (const ExprNode& nd : nodes_) {
        for (const ChildRef& child : children(nd)) {
            if (child.is_var) {
                if (var_stamp[child.id] != nd.id) {
                    var_stamp[child.id] = nd.id;
                    dependent_ids_[dependent_offsets_[child.id]++] = nd.id;
                }
            } else if (node_stamp[child.id] != nd.id) {
                node_stamp[child.id] = nd.id;
                parent_ids_[parent_offsets_[child.id]++] = nd.id;
            }
        }
    }
    std::copy_backward(parent_offsets_.begin(), parent_offsets_.end() - 1, parent_offsets_.end());
    parent_offsets_.front() = 0;
    std::copy_backward(dependent_offsets_.begin(), dependent_offsets_.end() - 1,
                       dependent_offsets_.end());
    dependent_offsets_.front() = 0;
}

void Model::close() {
    rebuild_back_references();
    topo_order_ = detail::compute_topo_order(*this);
    rebuild_topo_positions();
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
    rebuild_back_references();
    topo_order_ = detail::compute_topo_order(*this);
    rebuild_topo_positions();
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
//
// Into CSR, by inversion: the walk writes the constraint -> variables incidence
// flat, in constraint order, and a counting pass turns it round. Visiting the
// constraints in ascending index is what gives each variable's list ascending
// constraint indices -- the order the per-variable push_back this replaced
// produced. The incidence array is transient and costs 4 bytes per entry, where
// a second walk to count first would cost the whole DFS again.
void Model::build_var_constraints() {
    const auto n_cons = static_cast<int32_t>(constraint_ids_.size());
    std::vector<int32_t> node_stamp(nodes_.size(), -1);
    std::vector<int32_t> var_stamp(vars_.size(), -1);
    std::vector<int32_t> stack;
    // Constraint order. Sized from the last build's incidence count, which the
    // rebuild after add_objective_soft_constraint() matches to within one row, and
    // that build's CSR is released first so the two are never both alive. Its
    // offsets go with it, so a throw below leaves constraints_of_var reporting
    // out_of_range rather than reading freed ids.
    std::vector<int32_t> incident_vars;
    incident_vars.reserve(var_constraint_ids_.size());
    std::vector<int32_t>().swap(var_constraint_ids_);
    var_constraint_offsets_.clear();
    std::vector<size_t> incident_begin(static_cast<size_t>(n_cons) + 1);  // per constraint
    for (int32_t ci = 0; ci < n_cons; ++ci) {
        incident_begin[ci] = incident_vars.size();
        stack.clear();
        int32_t root = constraint_ids_[ci];
        node_stamp[root] = ci;
        stack.push_back(root);
        while (!stack.empty()) {
            int32_t nid = stack.back();
            stack.pop_back();
            for (const ChildRef& child : children(nodes_[nid])) {
                if (child.is_var) {
                    if (var_stamp[child.id] != ci) {
                        var_stamp[child.id] = ci;
                        incident_vars.push_back(child.id);
                    }
                } else if (node_stamp[child.id] != ci) {
                    node_stamp[child.id] = ci;
                    stack.push_back(child.id);
                }
            }
        }
    }
    incident_begin[n_cons] = incident_vars.size();
    if (incident_vars.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("model has more than 2^32 - 1 variable-constraint incidences");
    }

    var_constraint_offsets_.assign(vars_.size() + 1, 0);
    for (const int32_t v : incident_vars) {
        ++var_constraint_offsets_[v + 1];
    }
    std::partial_sum(var_constraint_offsets_.begin(), var_constraint_offsets_.end(),
                     var_constraint_offsets_.begin());
    var_constraint_ids_.resize(incident_vars.size());
    // offsets[v] as v's write cursor, then shifted back -- as in
    // rebuild_back_references.
    for (int32_t ci = 0; ci < n_cons; ++ci) {
        for (size_t k = incident_begin[ci]; k < incident_begin[ci + 1]; ++k) {
            var_constraint_ids_[var_constraint_offsets_[incident_vars[k]]++] = ci;
        }
    }
    std::copy_backward(var_constraint_offsets_.begin(), var_constraint_offsets_.end() - 1,
                       var_constraint_offsets_.end());
    var_constraint_offsets_.front() = 0;
}

std::vector<std::pair<int32_t, double>> Model::per_constraint_violation_delta(int32_t var_id,
                                                                              double j) {
    const Variable& v = var(var_id);  // bounds-checked
    if (is_structured(v.type)) {
        throw std::invalid_argument(
            "per_constraint_violation_delta: scalar variable required (Bool/Int/Float)");
    }

    const ConstSpan<int32_t> affected = constraints_of_var(var_id);
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
    if (is_structured(v.type)) {
        throw std::invalid_argument(
            "weighted_violation_delta: scalar variable required (Bool/Int/Float)");
    }
    const ConstSpan<int32_t> affected = constraints_of_var(var_id);
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
