#include "cbls/dag.h"
#include "cbls/model.h"
#include <cmath>
#include <algorithm>
#include <limits>

namespace cbls {

static double child_val(const ChildRef& ref, const Model& model) {
    if (ref.is_var) {
        return model.var(ref.id).value;
    } else {
        return model.node(ref.id).value;
    }
}

static double list_element(const ChildRef& ref, const Model& model, int idx) {
    if (ref.is_var) {
        const auto& v = model.var(ref.id);
        if (idx >= 0 && idx < static_cast<int>(v.elements.size())) {
            return static_cast<double>(v.elements[idx]);
        }
        return 0.0;
    }
    return 0.0;
}

static int list_size(const ChildRef& ref, const Model& model) {
    if (ref.is_var) {
        return static_cast<int>(model.var(ref.id).elements.size());
    }
    return 0;
}

double evaluate(const ExprNode& node, const Model& model) {
    switch (node.op) {
    case NodeOp::Const:
        return node.const_value;

    case NodeOp::Neg:
        return -child_val(node.children[0], model);

    case NodeOp::Sum: {
        double s = 0.0;
        for (const auto& c : node.children) {
            s += child_val(c, model);
        }
        return s;
    }

    case NodeOp::Prod:
        return child_val(node.children[0], model) * child_val(node.children[1], model);

    case NodeOp::Div: {
        double denom = child_val(node.children[1], model);
        double num = child_val(node.children[0], model);
        if (std::abs(denom) < 1e-15) {
            return num >= 0 ? std::numeric_limits<double>::infinity()
                            : -std::numeric_limits<double>::infinity();
        }
        return num / denom;
    }

    case NodeOp::Pow: {
        double base = child_val(node.children[0], model);
        double exp = child_val(node.children[1], model);
        double result = std::pow(base, exp);
        if (std::isfinite(result)) return result;
        return std::numeric_limits<double>::infinity();
    }

    case NodeOp::Min: {
        double m = child_val(node.children[0], model);
        for (size_t i = 1; i < node.children.size(); ++i) {
            m = std::min(m, child_val(node.children[i], model));
        }
        return m;
    }

    case NodeOp::Max: {
        double m = child_val(node.children[0], model);
        for (size_t i = 1; i < node.children.size(); ++i) {
            m = std::max(m, child_val(node.children[i], model));
        }
        return m;
    }

    case NodeOp::Abs:
        return std::abs(child_val(node.children[0], model));

    case NodeOp::Sin:
        return std::sin(child_val(node.children[0], model));

    case NodeOp::Cos:
        return std::cos(child_val(node.children[0], model));

    case NodeOp::If: {
        double cond = child_val(node.children[0], model);
        return cond > 0 ? child_val(node.children[1], model)
                        : child_val(node.children[2], model);
    }

    case NodeOp::At: {
        // children[0] = list var, children[1] = index expr
        int idx = static_cast<int>(child_val(node.children[1], model));
        return list_element(node.children[0], model, idx);
    }

    case NodeOp::Count: {
        return static_cast<double>(list_size(node.children[0], model));
    }

    case NodeOp::Lambda: {
        // Sum over list elements using lambda function
        if (node.lambda_func_id < 0) return 0.0;
        const auto& func = model.lambda_func(node.lambda_func_id);
        const auto& ref = node.children[0];
        if (!ref.is_var) return 0.0;
        const auto& v = model.var(ref.id);
        double s = 0.0;
        for (int32_t e : v.elements) {
            s += func(e);
        }
        return s;
    }

    case NodeOp::Leq:
        // child0 - child1 (≤ 0 when child0 ≤ child1)
        return child_val(node.children[0], model) - child_val(node.children[1], model);

    case NodeOp::Eq:
        // |child0 - child1| (= 0 when equal)
        return std::abs(child_val(node.children[0], model) - child_val(node.children[1], model));
    }
    return 0.0;
}

double local_derivative(const ExprNode& node, int child_idx, const Model& model) {
    switch (node.op) {
    case NodeOp::Const:
        return 0.0;

    case NodeOp::Neg:
        return -1.0;

    case NodeOp::Sum:
        return 1.0;

    case NodeOp::Prod: {
        int other = 1 - child_idx;
        return child_val(node.children[other], model);
    }

    case NodeOp::Div: {
        if (child_idx == 0) {
            double denom = child_val(node.children[1], model);
            return std::abs(denom) > 1e-15 ? 1.0 / denom : 0.0;
        } else {
            double denom = child_val(node.children[1], model);
            if (std::abs(denom) < 1e-15) return 0.0;
            return -child_val(node.children[0], model) / (denom * denom);
        }
    }

    case NodeOp::Pow: {
        double base = child_val(node.children[0], model);
        double exp = child_val(node.children[1], model);
        if (child_idx == 0) {
            if (std::abs(base) < 1e-15 && exp < 1) return 0.0;
            try {
                double result = exp * std::pow(base, exp - 1);
                return std::isfinite(result) ? result : 0.0;
            } catch (...) {
                return 0.0;
            }
        } else {
            if (base <= 0) return 0.0;
            try {
                double result = std::pow(base, exp) * std::log(base);
                return std::isfinite(result) ? result : 0.0;
            } catch (...) {
                return 0.0;
            }
        }
    }

    case NodeOp::Min: {
        double min_val = node.value;
        return std::abs(child_val(node.children[child_idx], model) - min_val) < 1e-12 ? 1.0 : 0.0;
    }

    case NodeOp::Max: {
        double max_val = node.value;
        return std::abs(child_val(node.children[child_idx], model) - max_val) < 1e-12 ? 1.0 : 0.0;
    }

    case NodeOp::Abs: {
        double v = child_val(node.children[0], model);
        if (v > 0) return 1.0;
        if (v < 0) return -1.0;
        return 0.0;
    }

    case NodeOp::Sin:
        return std::cos(child_val(node.children[0], model));

    case NodeOp::Cos:
        return -std::sin(child_val(node.children[0], model));

    case NodeOp::If: {
        if (child_idx == 0) return 0.0;  // non-differentiable w.r.t. condition
        double cond = child_val(node.children[0], model);
        if (child_idx == 1) return cond > 0 ? 1.0 : 0.0;
        return cond > 0 ? 0.0 : 1.0;
    }

    case NodeOp::At:
    case NodeOp::Count:
    case NodeOp::Lambda:
        return 0.0;  // discrete — not differentiable

    case NodeOp::Leq:
        return child_idx == 0 ? 1.0 : -1.0;

    case NodeOp::Eq: {
        double diff = child_val(node.children[0], model) - child_val(node.children[1], model);
        double sign = diff > 0 ? 1.0 : (diff < 0 ? -1.0 : 0.0);
        return child_idx == 0 ? sign : -sign;
    }
    }
    return 0.0;
}

}  // namespace cbls
