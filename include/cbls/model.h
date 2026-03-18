#pragma once

#include "dag.h"
#include <vector>
#include <string>
#include <functional>
#include <stdexcept>

namespace cbls {

// Forward declare Expr
class Expr;

class Model {
public:
    Model() = default;

    // Variable creation — returns var ID
    int32_t bool_var(const std::string& name = "");
    int32_t int_var(int lb, int ub, const std::string& name = "");
    int32_t float_var(double lb, double ub, const std::string& name = "");
    int32_t list_var(int n, const std::string& name = "");
    int32_t set_var(int n, int min_size = 0, int max_size = -1, const std::string& name = "");

    // Expression creation — returns node ID
    int32_t constant(double val);
    int32_t neg(int32_t x);
    int32_t sum(const std::vector<int32_t>& args);
    int32_t prod(int32_t a, int32_t b);
    int32_t div_expr(int32_t a, int32_t b);
    int32_t pow_expr(int32_t base, int32_t exp);
    int32_t min_expr(const std::vector<int32_t>& args);
    int32_t max_expr(const std::vector<int32_t>& args);
    int32_t abs_expr(int32_t x);
    int32_t sin_expr(int32_t x);
    int32_t cos_expr(int32_t x);
    int32_t tan_expr(int32_t x);
    int32_t exp_expr(int32_t x);
    int32_t log_expr(int32_t x);
    int32_t sqrt_expr(int32_t x);
    int32_t if_then_else(int32_t cond, int32_t then_, int32_t else_);
    int32_t at(int32_t list_var_id, int32_t index_expr);
    int32_t count(int32_t var_id);
    int32_t leq(int32_t a, int32_t b);
    int32_t eq_expr(int32_t a, int32_t b);
    int32_t geq(int32_t a, int32_t b);
    int32_t neq(int32_t a, int32_t b);
    int32_t lt(int32_t a, int32_t b);
    int32_t gt(int32_t a, int32_t b);
    int32_t lambda_sum(int32_t list_var_id, std::function<double(int)> func);
    int32_t pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func);

    void add_constraint(int32_t expr_id);
    void minimize(int32_t expr_id);
    void maximize(int32_t expr_id);

    // Expr-returning variable creation
    Expr Bool(const std::string& name = "");
    Expr Int(int lb, int ub, const std::string& name = "");
    Expr Float(double lb, double ub, const std::string& name = "");
    Expr List(int n, const std::string& name = "");
    Expr Set(int n, int min_size = 0, int max_size = -1, const std::string& name = "");
    Expr Constant(double val);

    // Overloaded constraint/objective accepting Expr
    void add_constraint(const Expr& e);
    void minimize(const Expr& e);
    void maximize(const Expr& e);

    void close();

    // Accessors
    const Variable& var(int32_t id) const {
        if (id < 0 || id >= static_cast<int32_t>(vars_.size()))
            throw std::out_of_range("var id out of range");
        return vars_[id];
    }
    Variable& var_mut(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(vars_.size()))
            throw std::out_of_range("var id out of range");
        return vars_[id];
    }
    const ExprNode& node(int32_t id) const {
        if (id < 0 || id >= static_cast<int32_t>(nodes_.size()))
            throw std::out_of_range("node id out of range");
        return nodes_[id];
    }
    ExprNode& node_mut(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(nodes_.size()))
            throw std::out_of_range("node id out of range");
        return nodes_[id];
    }
    int32_t objective_id() const noexcept { return objective_id_; }
    bool is_maximizing() const noexcept { return is_maximizing_; }
    const std::vector<int32_t>& constraint_ids() const noexcept { return constraint_ids_; }
    const std::vector<int32_t>& topo_order() const noexcept { return topo_order_; }
    const std::vector<Variable>& variables() const noexcept { return vars_; }
    std::vector<Variable>& variables_mut() noexcept { return vars_; }
    const std::vector<ExprNode>& nodes() const noexcept { return nodes_; }
    std::vector<ExprNode>& nodes_mut() noexcept { return nodes_; }
    size_t num_vars() const noexcept { return vars_.size(); }
    size_t num_nodes() const noexcept { return nodes_.size(); }
    bool is_closed() const noexcept { return closed_; }

    // Lambda function access
    const std::function<double(int)>& lambda_func(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(lambda_funcs_.size()))
            throw std::out_of_range("lambda func index out of range");
        return lambda_funcs_[idx];
    }

    const std::function<double(int, int)>& pair_lambda_func(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(pair_lambda_funcs_.size()))
            throw std::out_of_range("pair lambda func index out of range");
        return pair_lambda_funcs_[idx];
    }

    // State snapshot/restore
    struct State {
        std::vector<double> values;
        std::vector<std::vector<int32_t>> elements;
    };
    State copy_state() const;
    void restore_state(const State& state);

private:
    std::vector<Variable> vars_;
    std::vector<ExprNode> nodes_;
    std::vector<int32_t> topo_order_;
    std::vector<int32_t> constraint_ids_;
    int32_t objective_id_ = -1;
    bool is_maximizing_ = false;
    std::vector<std::function<double(int)>> lambda_funcs_;
    std::vector<std::function<double(int, int)>> pair_lambda_funcs_;
    bool closed_ = false;

    int32_t alloc_var(VarType type, double lb, double ub, const std::string& name);
    int32_t alloc_node(NodeOp op, const std::vector<ChildRef>& children);
    ChildRef wrap(int32_t id);  // auto-detect var vs node
};

}  // namespace cbls
