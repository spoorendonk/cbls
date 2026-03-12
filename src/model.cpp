#include "cbls/model.h"
#include "cbls/dag_ops.h"
#include <algorithm>

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

ChildRef Model::wrap(int32_t id) {
    // Negative IDs or IDs that fall within var range but were returned by
    // var creation are var refs. IDs from expression creation are node refs.
    // The convention: var IDs and node IDs are in separate spaces.
    // Variables have IDs [0, num_vars), nodes have IDs [0, num_nodes).
    // We need to know if the user passed a var ID or node ID.
    // The Python version checks isinstance. In C++, we use a convention:
    // var IDs are encoded as negative: -(var_id + 1) in the user-facing API?
    // No, that's ugly. Better: use a flag.

    // Actually, looking at the Python code, Model.sum() etc take Variable or ExprNode.
    // In C++, we use int32_t IDs. We need to distinguish.
    // Convention: variable IDs are stored as -(id+1), node IDs as id.
    // OR: we use a separate "handle" type.

    // Simplest approach: The user-facing API returns "handles" that encode
    // whether they are vars or nodes. We'll use bit 31 as the flag.
    // Actually the simplest: we just check if id < num_vars and treat it
    // as a variable. But that's ambiguous when there are both vars and nodes.

    // Let me use a clean approach: user-facing IDs are globally unique.
    // Variables get IDs [0, N), nodes get IDs [N, N+M).
    // Then wrap(id) checks if id < num_vars.

    // Wait, this won't work because nodes are allocated dynamically and their
    // count changes. Let me think...

    // Better approach: we'll use a "handle" where the high bit indicates type.
    // Var handles: id | 0x80000000, Node handles: id (plain).
    // This is clean and efficient.

    // Actually, let's be even simpler. The vars and nodes are allocated
    // incrementally. We give them IDs in a single namespace:
    // vars get even IDs starting from 0, nodes get odd... no, that's silly.

    // Let me just do what the plan says: user-facing methods return int32_t,
    // and we use a global ID counter that assigns unique IDs across both
    // vars and nodes. Then wrap() can determine which.

    // OR: Most straightforward - keep vars and nodes in separate vectors
    // but return "typed handles". Since C++ API will be used by calling
    // bool_var(), float_var() etc which clearly return var handles, and
    // sum(), prod() etc which return expr handles, we can just use
    // a signed convention: var handles are negative, expr handles are non-negative.

    // Let's use: var handles = -(var_id + 1), node handles = node_id
    // So wrap(handle) decodes: if handle < 0 => var, else => node.

    ChildRef ref;
    if (id < 0) {
        ref.id = -(id + 1);  // decode var ID
        ref.is_var = true;
    } else {
        ref.id = id;
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

int32_t Model::lambda_sum(int32_t list_var_handle, std::function<double(int)> func) {
    lambda_funcs_.push_back(std::move(func));
    int32_t func_id = static_cast<int32_t>(lambda_funcs_.size() - 1);

    int32_t nid = alloc_node(NodeOp::Lambda, {wrap(list_var_handle)});
    nodes_[nid].lambda_func_id = func_id;
    return nid;
}

void Model::add_constraint(int32_t expr_id) {
    // expr_id should be a node handle (non-negative)
    constraint_ids_.push_back(expr_id);
}

void Model::minimize(int32_t expr_id) {
    objective_id_ = expr_id;
}

void Model::maximize(int32_t expr_id) {
    // Maximize by negating
    objective_id_ = neg(expr_id);
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
    for (size_t i = 0; i < vars_.size(); ++i) {
        vars_[i].value = state.values[i];
        vars_[i].elements = state.elements[i];
    }
}

}  // namespace cbls
