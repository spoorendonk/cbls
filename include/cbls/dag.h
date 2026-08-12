#pragma once

#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

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
    int32_t min_size = 0;                // Set: minimum cardinality
    int32_t max_size = 0;                // Set/List: maximum cardinality
    std::vector<int32_t> dependent_ids;  // ExprNode IDs that depend on this var
};

enum class NodeOp : uint8_t {
    Const,
    Neg,
    Sum,
    Prod,
    Div,
    Pow,
    Min,
    Max,
    Abs,
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Sqrt,
    SignPower,
    Tanh,
    If,
    At,
    Count,
    Lambda,
    PairLambda,
    Leq,
    Eq,
    Geq,
    Neq,
    Lt,
    Gt
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

/// Residual of `a <= b`, i.e. `a - b`, with the IEEE `inf - inf` indeterminacy
/// resolved by the comparison the residual stands for.
///
/// An infinite bound marks a side as *absent*: `a <= +inf` and `-inf <= b` hold
/// for every a/b, including a non-finite one. Plain `a - b` yields NaN when both
/// sides are infinite with the same sign, and the violation machinery reads NaN
/// as a maximal violation (we have no evidence the row holds) — so a vacuous row
/// would be reported as the worst-violated row in the model. That is exactly
/// what destroys the feasibility signal when the objective is folded in as
/// `obj <= bound`, the objective overflows to +inf, and the bound is still at
/// its initial +inf (issue #100).
///
/// Equal infinities return 0.0: satisfied, sitting on the boundary. `Lt`/`Gt`
/// add their strictness epsilon on top, which correctly turns that boundary into
/// a violation (`+inf < +inf` is false). Every other input is the plain
/// difference, which is already right: +inf for (a=+inf, b=-inf), -inf for
/// (a=-inf, b=+inf), and NaN whenever either side is NaN — an unevaluable body
/// must stay a maximal violation.
inline double comparison_residual(double a, double b) {
    if (a == b && std::isinf(a)) {
        return 0.0;
    }
    return a - b;
}

// Forward declaration
class Model;

double evaluate(const ExprNode& node, const Model& model);
double local_derivative(const ExprNode& node, int child_idx, const Model& model);

}  // namespace cbls
