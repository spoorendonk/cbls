#include "cbls/dag.h"

#include "cbls/model.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

namespace cbls {

// A child's value, read WITHOUT the range check `Model::var`/`Model::node` do.
//
// Safe because every ChildRef is validated once, when the node naming it is made
// (`Model::wrap` throws on an id past what exists), and a model never removes a
// variable or node -- so the check here could only ever pass. It is not free,
// though: evaluate() and local_derivative() read a child on nearly every DAG
// edge they touch, and the throwing branch kept the reads from being as cheap
// as a plain index. Measured for #156 (idle box; cbls_minlplib, 20k iterations,
// seed 7; serial, interleaved, median of 5; results identical), with the
// checked read -> this: nvs05 9.08 s -> 7.70, chain50 3.93 -> 2.31, ex8_6_1
// 3.95 -> 3.04, maxmin 2.99 -> 2.54. The asserts keep the check in a Debug build
// (-DCMAKE_BUILD_TYPE=Debug); a sanitizer build is Release unless configured
// otherwise, and loses them. Child counts are not re-checked either: Min and Max
// read children[0], so their builders refuse an empty argument list.
static double child_val(const ChildRef& ref, const Model& model) {
    if (ref.is_var) {
        assert(static_cast<size_t>(ref.id) < model.num_vars());
        return model.variables()[ref.id].value;
    }
    assert(static_cast<size_t>(ref.id) < model.num_nodes());
    return model.node_values()[ref.id];
}

// Whether a comparison's child is a literal Const node — i.e. a bound the
// modeller wrote, not a quantity the DAG computed. `comparison_residual` needs
// this to tell an "absent bound" +inf sentinel from an arithmetic overflow; a
// variable is never a sentinel, since its value is search state. Unchecked for
// the reason child_val gives.
static bool child_is_const(const ChildRef& ref, const Model& model) {
    assert(ref.is_var || static_cast<size_t>(ref.id) < model.num_nodes());
    return !ref.is_var && model.nodes()[ref.id].op == NodeOp::Const;
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

// A flat dispatch table over NodeOp's 28 cases, suppressed deliberately rather
// than split. What the score measures here is not compounded logic: it is the
// eighteen small guards the individual cases carry -- a loop over a variadic
// node's children, a divide-by-zero test, an overflow test -- each charged
// double because the metric adds the enclosing switch's nesting level to every
// one of them. The cases are siblings, not a hierarchy: none runs past six lines
// or nests deeper than two, and none can affect another.
//
// So the readable unit here is the table, and per-family helpers would hide it.
// The switch is also deliberately `default:`-free, so the compiler rather than
// this metric is what catches a NodeOp nobody handled, and delta_evaluate calls
// this once per dirtied node per candidate move -- any split would have to stay
// inlinable, which rules out the dispatch-through-a-table alternative outright.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
double evaluate(const ExprNode& node, const Model& model) {
    const ConstSpan<ChildRef> children = model.children(node);
    switch (node.op) {
        case NodeOp::Const:
            // One Const is not a literal: the objective row's RHS. Each
            // portfolio worker tightens its own bound on its own incumbents, so
            // that value is per-model state and lives in the model, not in the
            // shared node's `const_value` -- which keeps the +inf the row was
            // created with (#157).
            //
            // Cheap and safe to test here. A Const is a leaf, so it is never in a
            // `delta_evaluate` dirty cone and this arm runs only from
            // `full_evaluate`; and `objective_bound_node()` is -1 until the row
            // exists, so the comparison is inert on a model that has none. The
            // branch is also NEEDED rather than merely tidy: `full_evaluate` runs
            // after every `restore_state`, and without it a restart would reset
            // the worker's bound to +inf.
            if (node.id == model.objective_bound_node()) {
                return model.objective_bound();
            }
            return node.const_value;

        case NodeOp::Neg:
            return -child_val(children[0], model);

        case NodeOp::Sum: {
            double s = 0.0;
            for (const auto& c : children) {
                s += child_val(c, model);
            }
            return s;
        }

        case NodeOp::Prod:
            return child_val(children[0], model) * child_val(children[1], model);

        case NodeOp::Div: {
            double denom = child_val(children[1], model);
            double num = child_val(children[0], model);
            if (std::abs(denom) < 1e-15) {
                return num >= 0 ? std::numeric_limits<double>::infinity()
                                : -std::numeric_limits<double>::infinity();
            }
            return num / denom;
        }

        case NodeOp::Pow: {
            double base = child_val(children[0], model);
            double exp = child_val(children[1], model);
            double result = std::pow(base, exp);
            if (std::isfinite(result)) {
                return result;
            }
            return std::numeric_limits<double>::infinity();
        }

        case NodeOp::Min: {
            double m = child_val(children[0], model);
            for (size_t i = 1; i < children.size(); ++i) {
                m = std::min(m, child_val(children[i], model));
            }
            return m;
        }

        case NodeOp::Max: {
            double m = child_val(children[0], model);
            for (size_t i = 1; i < children.size(); ++i) {
                m = std::max(m, child_val(children[i], model));
            }
            return m;
        }

        case NodeOp::Abs:
            return std::abs(child_val(children[0], model));

        case NodeOp::Sin:
            return std::sin(child_val(children[0], model));

        case NodeOp::Cos:
            return std::cos(child_val(children[0], model));

        case NodeOp::Tan:
            return std::tan(child_val(children[0], model));

        case NodeOp::Exp:
            return std::exp(child_val(children[0], model));

        case NodeOp::Log: {
            double x = child_val(children[0], model);
            if (x <= 0) {
                return -std::numeric_limits<double>::infinity();
            }
            return std::log(x);
        }

        case NodeOp::Sqrt: {
            double x = child_val(children[0], model);
            if (x < 0) {
                return 0.0;
            }
            return std::sqrt(x);
        }

        case NodeOp::SignPower: {
            // sign(x) * |x|^p  (AMPL OPSIGNPOWER / MINLPLib opsignpower).
            double x = child_val(children[0], model);
            double p = child_val(children[1], model);
            double mag = std::pow(std::abs(x), p);
            if (!std::isfinite(mag)) {
                return x >= 0 ? std::numeric_limits<double>::infinity()
                              : -std::numeric_limits<double>::infinity();
            }
            return std::copysign(mag, x);  // sign(x) carried, 0 stays 0
        }

        case NodeOp::Tanh:
            return std::tanh(child_val(children[0], model));

        case NodeOp::If: {
            double cond = child_val(children[0], model);
            return cond > 0 ? child_val(children[1], model) : child_val(children[2], model);
        }

        case NodeOp::At: {
            // children[0] = list var, children[1] = index expr
            int idx = static_cast<int>(child_val(children[1], model));
            return list_element(children[0], model, idx);
        }

        case NodeOp::Count: {
            return static_cast<double>(list_size(children[0], model));
        }

        case NodeOp::Lambda: {
            // Sum over list elements using lambda function
            if (node.lambda_func_id < 0) {
                return 0.0;
            }
            const auto& func = model.lambda_func(node.lambda_func_id);
            const auto& ref = children[0];
            if (!ref.is_var) {
                return 0.0;
            }
            const auto& v = model.var(ref.id);
            double s = 0.0;
            for (int32_t e : v.elements) {
                s += func(e);
            }
            return s;
        }

        case NodeOp::PairLambda: {
            // Sum over consecutive pairs, with the closing rule and the two
            // optional fixed-endpoint terms that `PairLambdaSpec` records.
            // Contract for n = 0, 1, 2 is on Model::pair_lambda_sum.
            if (node.lambda_func_id < 0) {
                return 0.0;
            }
            const auto& func = model.pair_lambda_func(node.lambda_func_id);
            const auto& ref = children[0];
            if (!ref.is_var) {
                return 0.0;
            }
            const auto& spec = model.pair_lambda_spec(node.lambda_func_id);
            const auto& v = model.var(ref.id);
            const auto& el = v.elements;
            double s = 0.0;
            for (size_t k = 0; k + 1 < el.size(); ++k) {
                s += func(el[k], el[k + 1]);
            }
            if (spec.mode == PairMode::Cyclic && el.size() >= 2) {
                s += func(el.back(), el.front());
            }
            if (!el.empty()) {
                if (spec.head_id >= 0) {
                    s += model.lambda_func(spec.head_id)(el.front());
                }
                if (spec.tail_id >= 0) {
                    s += model.lambda_func(spec.tail_id)(el.back());
                }
            }
            return s;
        }

        case NodeOp::Leq:
            // child0 - child1 (≤ 0 when child0 ≤ child1)
            return comparison_residual(child_val(children[0], model), child_val(children[1], model),
                                       child_is_const(children[0], model),
                                       child_is_const(children[1], model));

        case NodeOp::Eq:
            // |child0 - child1| (= 0 when equal)
            return std::abs(child_val(children[0], model) - child_val(children[1], model));

        case NodeOp::Geq:
            // b - a (≤ 0 when a ≥ b)
            return comparison_residual(child_val(children[1], model), child_val(children[0], model),
                                       child_is_const(children[1], model),
                                       child_is_const(children[0], model));

        case NodeOp::Neq:
            // 1 when equal (violated), 0 when not equal (satisfied)
            return (std::abs(child_val(children[0], model) - child_val(children[1], model)) < 1e-9)
                       ? 1.0
                       : 0.0;

        case NodeOp::Lt: {
            // a - b + ε (≤ 0 when a < b strictly)
            constexpr double kEps = 1e-9;
            return comparison_residual(child_val(children[0], model), child_val(children[1], model),
                                       child_is_const(children[0], model),
                                       child_is_const(children[1], model)) +
                   kEps;
        }

        case NodeOp::Gt: {
            // b - a + ε (≤ 0 when a > b strictly)
            constexpr double kEps = 1e-9;
            return comparison_residual(child_val(children[1], model), child_val(children[0], model),
                                       child_is_const(children[1], model),
                                       child_is_const(children[0], model)) +
                   kEps;
        }
    }
    return 0.0;
}

// The AD peer of evaluate() above, and suppressed for the same reason and by the
// same argument: one flat `default:`-free dispatch table over the same 28 NodeOp
// cases, where the score is the sum of each case's own guards against a
// non-finite or non-differentiable point rather than any nesting between them.
// It scores higher than evaluate() only because a derivative needs more such
// guards, not because the cases interact. Reverse-mode AD calls it once per DAG
// edge, so it sits on the same hot path.
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
double local_derivative(const ExprNode& node, int child_idx, const Model& model) {
    const ConstSpan<ChildRef> children = model.children(node);
    switch (node.op) {
        case NodeOp::Const:
            return 0.0;

        case NodeOp::Neg:
            return -1.0;

        case NodeOp::Sum:
            return 1.0;

        case NodeOp::Prod: {
            int other = 1 - child_idx;
            return child_val(children[other], model);
        }

        case NodeOp::Div: {
            if (child_idx == 0) {
                double denom = child_val(children[1], model);
                return std::abs(denom) > 1e-15 ? 1.0 / denom : 0.0;
            }
            double denom = child_val(children[1], model);
            if (std::abs(denom) < 1e-15) {
                return 0.0;
            }
            return -child_val(children[0], model) / (denom * denom);
        }

        case NodeOp::Pow: {
            double base = child_val(children[0], model);
            double exp = child_val(children[1], model);
            if (child_idx == 0) {
                if (std::abs(base) < 1e-15 && exp < 1) {
                    return 0.0;
                }
                try {
                    double result = exp * std::pow(base, exp - 1);
                    return std::isfinite(result) ? result : 0.0;
                } catch (...) {
                    return 0.0;
                }
            } else {
                if (base <= 0) {
                    return 0.0;
                }
                try {
                    double result = std::pow(base, exp) * std::log(base);
                    return std::isfinite(result) ? result : 0.0;
                } catch (...) {
                    return 0.0;
                }
            }
        }

        case NodeOp::Min: {
            double min_val = model.node_values()[node.id];
            return std::abs(child_val(children[child_idx], model) - min_val) < 1e-12 ? 1.0 : 0.0;
        }

        case NodeOp::Max: {
            double max_val = model.node_values()[node.id];
            return std::abs(child_val(children[child_idx], model) - max_val) < 1e-12 ? 1.0 : 0.0;
        }

        case NodeOp::Abs: {
            double v = child_val(children[0], model);
            if (v > 0) {
                return 1.0;
            }
            if (v < 0) {
                return -1.0;
            }
            return 0.0;
        }

        case NodeOp::Sin:
            return std::cos(child_val(children[0], model));

        case NodeOp::Cos:
            return -std::sin(child_val(children[0], model));

        case NodeOp::Tan: {
            double c = std::cos(child_val(children[0], model));
            return 1.0 / (c * c);
        }

        case NodeOp::Exp:
            return std::exp(child_val(children[0], model));

        case NodeOp::Log: {
            double x = child_val(children[0], model);
            if (std::abs(x) < 1e-15) {
                return 0.0;
            }
            return 1.0 / x;
        }

        case NodeOp::Sqrt: {
            double x = child_val(children[0], model);
            if (x < 1e-15) {
                return 0.0;
            }
            return 1.0 / (2.0 * std::sqrt(x));
        }

        case NodeOp::SignPower: {
            // d/dx [sign(x)|x|^p] = p*|x|^(p-1). Exponent treated as constant (AD
            // w.r.t. the exponent child is not needed — NL emits a numeric power).
            if (child_idx == 1) {
                return 0.0;
            }
            double x = child_val(children[0], model);
            double p = child_val(children[1], model);
            double ax = std::abs(x);
            if (ax < 1e-15) {
                // Slope at 0 is finite only for p == 1 (value |x|); 0 otherwise so
                // the search doesn't chase an infinite gradient at the cusp.
                return p == 1.0 ? 1.0 : 0.0;
            }
            double result = p * std::pow(ax, p - 1.0);
            return std::isfinite(result) ? result : 0.0;
        }

        case NodeOp::Tanh: {
            double t = std::tanh(child_val(children[0], model));
            return 1.0 - (t * t);
        }

        case NodeOp::If: {
            if (child_idx == 0) {
                return 0.0;  // non-differentiable w.r.t. condition
            }
            double cond = child_val(children[0], model);
            if (child_idx == 1) {
                return cond > 0 ? 1.0 : 0.0;
            }
            return cond > 0 ? 0.0 : 1.0;
        }

        case NodeOp::At:
        case NodeOp::Count:
        case NodeOp::Lambda:
        case NodeOp::PairLambda:
            return 0.0;  // discrete — not differentiable

        case NodeOp::Leq:
            return child_idx == 0 ? 1.0 : -1.0;

        case NodeOp::Eq: {
            double diff = child_val(children[0], model) - child_val(children[1], model);
            // Both comparisons are false for diff == 0 and for NaN, so each keeps the
            // 0.0 initializer. std::copysign is NOT a substitute: it returns +/-1 for a
            // zero or NaN diff and would inject a spurious gradient on the AD path.
            double sign = 0.0;
            if (diff > 0) {
                sign = 1.0;
            } else if (diff < 0) {
                sign = -1.0;
            }
            return child_idx == 0 ? sign : -sign;
        }

        case NodeOp::Geq:
            // d/d(child0) of (child1 - child0) = -1, d/d(child1) = 1
            return child_idx == 0 ? -1.0 : 1.0;

        case NodeOp::Neq:
            return 0.0;  // non-differentiable

        case NodeOp::Lt:
            // d/d(child0) of (child0 - child1 + kEps) = 1, d/d(child1) = -1
            return child_idx == 0 ? 1.0 : -1.0;

        case NodeOp::Gt:
            // d/d(child0) of (child1 - child0 + kEps) = -1, d/d(child1) = 1
            return child_idx == 0 ? -1.0 : 1.0;
    }
    return 0.0;
}

}  // namespace cbls
