#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <limits>
#include <functional>

namespace cbls {

enum class VarType : uint8_t { Bool, Int, Float, List, Set };

struct Variable {
    int32_t id = -1;
    VarType type = VarType::Float;
    double value = 0.0;
    double lb = 0.0;
    double ub = 0.0;
    std::string name;
    std::vector<int32_t> elements;       // List/Set current elements
    int32_t universe_size = 0;           // Set: universe {0..n-1}
    int32_t min_size = 0;               // Set: minimum cardinality
    int32_t max_size = 0;               // Set/List: maximum cardinality
    std::vector<int32_t> dependent_ids;  // ExprNode IDs that depend on this var
};

enum class NodeOp : uint8_t {
    Const, Neg, Sum, Prod, Div, Pow, Min, Max, Abs,
    Sin, Cos, Tan, Exp, Log, Sqrt,
    If, At, Count, Lambda,
    Leq, Eq, Geq, Neq, Lt, Gt
};

struct ChildRef {
    int32_t id = -1;
    bool is_var = false;
};

struct ExprNode {
    int32_t id = -1;
    NodeOp op = NodeOp::Const;
    double value = 0.0;
    double const_value = 0.0;
    std::vector<ChildRef> children;
    std::vector<int32_t> parent_ids;
    int32_t lambda_func_id = -1;  // index into Model::lambda_funcs_
};

// Forward declaration
class Model;

double evaluate(const ExprNode& node, const Model& model);
double local_derivative(const ExprNode& node, int child_idx, const Model& model);

}  // namespace cbls
