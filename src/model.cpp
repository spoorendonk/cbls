#include "cbls/model.h"
#include "cbls/expr.h"
#include "cbls/dag_ops.h"

namespace cbls {

// Forward declare from dag_ops.cpp
namespace detail {
std::vector<int32_t> compute_topo_order(Model& model);
}

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
    // Handle encoding: var handles = -(var_id + 1) (negative), node handles = node_id (non-negative)
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
    for (int i = 0; i < n; ++i) v.elements[i] = i;
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
    if (args.empty()) return constant(0.0);
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
    for (int32_t a : args) children.push_back(wrap(a));
    return alloc_node(NodeOp::Min, children);
}

int32_t Model::max_expr(const std::vector<int32_t>& args) {
    std::vector<ChildRef> children;
    for (int32_t a : args) children.push_back(wrap(a));
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
        throw std::invalid_argument("add_constraint requires a node handle (non-negative), got var handle");
    }
    constraint_ids_.push_back(expr_id);
}

void Model::minimize(int32_t expr_id) {
    if (expr_id < 0) {
        throw std::invalid_argument("minimize requires a node handle (non-negative), got var handle");
    }
    objective_id_ = expr_id;
}

void Model::maximize(int32_t expr_id) {
    // Maximize by negating
    objective_id_ = neg(expr_id);
    is_maximizing_ = true;
}

void Model::close() {
    topo_order_ = detail::compute_topo_order(*this);
    full_evaluate(*this);
    closed_ = true;
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
    if (state.values.size() != vars_.size())
        throw std::invalid_argument("state size does not match model");
    for (size_t i = 0; i < vars_.size(); ++i) {
        vars_[i].value = state.values[i];
        vars_[i].elements = state.elements[i];
    }
}

}  // namespace cbls
