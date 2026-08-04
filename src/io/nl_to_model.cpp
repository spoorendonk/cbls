// Adapter: NlProblem -> closed CBLS Model.
//
// Maps AMPL NL operators (opcode.hd numbering) onto CBLS DAG ops. Operators
// outside the supported set do not throw: the result is marked unsupported with
// a human-readable reason, mirroring the "skip, don't crash" contract in
// io_nl.h. The supported set is the MINLPLib op subset CBLS can express today
// (+, -, *, /, pow, min, max, abs, sin, cos, tan, exp, log, log10, sqrt,
// signpower, tanh, unary minus) plus the linear J/G parts.

#include "cbls/expr.h"
#include "cbls/io_nl.h"
#include "cbls/model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace cbls {

namespace {

// AMPL opcodes we map (subset of opcode.hd). Anything else is unsupported.
enum AmplOp {
    OPPLUS = 0,
    OPMINUS = 1,
    OPMULT = 2,
    OPDIV = 3,
    OPPOW = 5,
    MINLIST = 11,
    MAXLIST = 12,
    ABS = 15,
    OPUMINUS = 16,
    OP_tanh = 37,
    OP_tan = 38,
    OP_sqrt = 39,
    OP_sin = 41,
    OP_log10 = 42,
    OP_log = 43,
    OP_exp = 44,
    OP_cos = 46,
    OPSUMLIST = 54,
    OP1POW = 76,  // base ^ constant exponent
    OP2POW = 77,  // x ^ 2
    OPCPOW = 78,  // constant ^ exponent
};

double clamp_lo(double lb, double inf_clamp) {
    if (!std::isfinite(lb) || lb < -inf_clamp) {
        return -inf_clamp;
    }
    return lb;
}
double clamp_hi(double ub, double inf_clamp) {
    if (!std::isfinite(ub) || ub > inf_clamp) {
        return inf_clamp;
    }
    return ub;
}

// Translate one NL expression tree into a CBLS expression node handle. On the
// first unsupported opcode, sets `ok=false`, records a reason, and returns a
// constant 0 (the caller bails out on !ok). `var_handles` maps NL var index to
// CBLS handle.
class ExprTranslator {
public:
    ExprTranslator(Model& m, const NlExpr& e, const std::vector<int32_t>& vh)
        : m_(m), e_(e), vh_(vh) {}

    bool ok() const { return ok_; }
    const std::string& reason() const { return reason_; }

    int32_t translate(int32_t node_idx) {
        if (!ok_) {
            return m_.constant(0.0);
        }
        const NlExprNode& n = e_.nodes[node_idx];
        switch (n.kind) {
            case NlNodeKind::Num:
                return m_.constant(n.num);
            case NlNodeKind::Var:
                if (n.index < 0 || n.index >= static_cast<int32_t>(vh_.size())) {
                    fail("var index out of range");
                    return m_.constant(0.0);
                }
                return vh_[n.index];
            case NlNodeKind::Op:
                return translate_op(n);
        }
        fail("unknown node kind");
        return m_.constant(0.0);
    }

private:
    void fail(const std::string& why) {
        if (ok_) {
            ok_ = false;
            reason_ = why;
        }
    }

    // Fold an n-ary child list into binary CBLS ops where needed.
    int32_t translate_op(const NlExprNode& n) {
        std::vector<int32_t> kids;
        kids.reserve(n.children.size());
        for (int32_t c : n.children) {
            kids.push_back(translate(c));
            if (!ok_) {
                return m_.constant(0.0);
            }
        }
        switch (n.opcode) {
            case OPPLUS:
                return m_.sum({kids[0], kids[1]});
            case OPMINUS:
                return m_.sum({kids[0], m_.neg(kids[1])});
            case OPMULT:
                return m_.prod(kids[0], kids[1]);
            case OPDIV:
                return m_.div_expr(kids[0], kids[1]);
            case OPPOW:
            case OP1POW:
                return m_.pow_expr(kids[0], kids[1]);
            case OP2POW:
                return m_.pow_expr(kids[0], m_.constant(2.0));
            case OPCPOW:
                // constant ^ x  ==  pow(base, x); base is kids[0] (a constant).
                return m_.pow_expr(kids[0], kids[1]);
            case OPUMINUS:
                return m_.neg(kids[0]);
            case ABS:
                return m_.abs_expr(kids[0]);
            case OP_sqrt:
                return m_.sqrt_expr(kids[0]);
            case OP_exp:
                return m_.exp_expr(kids[0]);
            case OP_log:
                return m_.log_expr(kids[0]);
            case OP_log10:
                // log10(x) = log(x) / log(10)
                return m_.div_expr(m_.log_expr(kids[0]), m_.constant(std::log(10.0)));
            case OP_sin:
                return m_.sin_expr(kids[0]);
            case OP_cos:
                return m_.cos_expr(kids[0]);
            case OP_tan:
                return m_.tan_expr(kids[0]);
            case OP_tanh:
                return m_.tanh_expr(kids[0]);
            case MINLIST:
                if (kids.empty()) {
                    fail("empty MINLIST");
                    return m_.constant(0.0);
                }
                return m_.min_expr(kids);
            case MAXLIST:
                if (kids.empty()) {
                    fail("empty MAXLIST");
                    return m_.constant(0.0);
                }
                return m_.max_expr(kids);
            case OPSUMLIST:
                return m_.sum(kids);  // empty -> constant 0 (Model::sum handles it)
            default:
                fail("unsupported NL opcode " + std::to_string(n.opcode));
                return m_.constant(0.0);
        }
    }

    Model& m_;
    const NlExpr& e_;
    const std::vector<int32_t>& vh_;
    bool ok_ = true;
    std::string reason_;
};

// Build the linear part sum_j coef_j * x_j as a CBLS node (or constant 0).
int32_t build_linear(Model& m, const std::vector<NlLinTerm>& terms,
                     const std::vector<int32_t>& vh) {
    std::vector<int32_t> parts;
    parts.reserve(terms.size());
    for (const NlLinTerm& t : terms) {
        if (t.var < 0 || t.var >= static_cast<int32_t>(vh.size())) {
            continue;
        }
        int32_t vhandle = vh[t.var];
        if (t.coef == 1.0) {
            parts.push_back(vhandle);
        } else if (t.coef == -1.0) {
            parts.push_back(m.neg(vhandle));
        } else {
            parts.push_back(m.prod(m.constant(t.coef), vhandle));
        }
    }
    if (parts.empty()) {
        return m.constant(0.0);
    }
    if (parts.size() == 1) {
        return parts[0];
    }
    return m.sum(parts);
}

// Combine a (possibly empty) nonlinear node and a linear node into the body
// expression. Returns a node handle.
int32_t combine_body(Model& m, int32_t nonlinear, bool has_nonlinear, int32_t linear,
                     bool has_linear) {
    if (has_nonlinear && has_linear) {
        return m.sum({nonlinear, linear});
    }
    if (has_nonlinear) {
        return nonlinear;
    }
    if (has_linear) {
        return linear;
    }
    return m.constant(0.0);
}

}  // namespace

NlToModelResult nl_to_model(const NlProblem& prob, const NlToModelOptions& opts) {
    NlToModelResult result;
    Model& m = result.model;

    // ---------- Variables ----------
    // Integer/binary NL columns become Int variables so the search respects
    // integrality; everything else is a Float. `var_is_discrete` comes from the
    // NL header counts plus Gay's variable ordering (see nl_reader.cpp).
    result.var_handles.reserve(prob.n_vars);
    for (int32_t j = 0; j < prob.n_vars; ++j) {
        double lb = -opts.inf_clamp;
        double ub = opts.inf_clamp;
        if (j < static_cast<int32_t>(prob.var_bounds.size())) {
            lb = clamp_lo(prob.var_bounds[j].lower, opts.inf_clamp);
            ub = clamp_hi(prob.var_bounds[j].upper, opts.inf_clamp);
        }
        if (lb > ub) {
            std::swap(lb, ub);  // defensive: degenerate bound ordering
        }
        const bool discrete = j < static_cast<int32_t>(prob.var_is_discrete.size()) &&
                              prob.var_is_discrete[static_cast<size_t>(j)] != 0;
        if (discrete) {
            // Tighten to the integers inside [lb, ub]. An unbounded integer
            // column would otherwise get the ±inf_clamp box, which is a useless
            // search domain; clamp it to `int_inf_clamp` instead.
            double ilb = std::ceil(std::max(lb, -opts.int_inf_clamp) - 1e-9);
            double iub = std::floor(std::min(ub, opts.int_inf_clamp) + 1e-9);
            if (ilb > iub) {
                // Bounds admit no integer (degenerate). Keep a single point so
                // the model still closes; the row's constraints will register
                // the violation rather than the reader silently dropping it.
                iub = ilb;
            }
            result.var_handles.push_back(m.int_var(static_cast<int>(ilb), static_cast<int>(iub),
                                                   "x" + std::to_string(j)));
        } else {
            result.var_handles.push_back(m.float_var(lb, ub, "x" + std::to_string(j)));
        }
    }

    // Seed initial values from the NL `x` segment where present.
    for (int32_t j = 0; j < prob.n_vars && j < static_cast<int32_t>(prob.initial_x.size()); ++j) {
        double x0 = prob.initial_x[j];
        if (std::isfinite(x0)) {
            double lb = m.var(j).lb;
            double ub = m.var(j).ub;
            m.var_mut(j).value = std::min(std::max(x0, lb), ub);
        }
    }

    auto record_unsupported = [&](const std::string& reason) {
        result.supported = false;
        result.skipped_reasons.push_back(reason);
    };

    // ---------- Constraints ----------
    result.constraint_node_ids.assign(prob.n_cons, -1);
    for (int32_t i = 0; i < prob.n_cons; ++i) {
        const NlConstraint& c = prob.constraints[i];

        int32_t nl_node = 0;
        bool has_nl = !c.nonlinear.empty();
        if (has_nl) {
            ExprTranslator tr(m, c.nonlinear, result.var_handles);
            nl_node = tr.translate(c.nonlinear.root);
            if (!tr.ok()) {
                record_unsupported("constraint " + std::to_string(i) + ": " + tr.reason());
                return result;
            }
        }
        int32_t lin_node = build_linear(m, c.linear, result.var_handles);
        bool has_lin = !c.linear.empty();
        int32_t body = combine_body(m, nl_node, has_nl, lin_node, has_lin);

        // Translate the bound into one/two CBLS comparison constraints.
        const NlConBound& b = c.bound;
        switch (b.type) {
            case NlBoundType::Upper: {
                int32_t cn = m.leq(body, m.constant(b.upper));
                m.add_constraint(cn);
                result.constraint_node_ids[i] = cn;
                break;
            }
            case NlBoundType::Lower: {
                int32_t cn = m.geq(body, m.constant(b.lower));
                m.add_constraint(cn);
                result.constraint_node_ids[i] = cn;
                break;
            }
            case NlBoundType::Equal: {
                int32_t cn = m.eq_expr(body, m.constant(b.lower));
                m.add_constraint(cn);
                result.constraint_node_ids[i] = cn;
                break;
            }
            case NlBoundType::Range: {
                // lower <= body <= upper -> two constraints.
                int32_t lo = m.geq(body, m.constant(b.lower));
                int32_t hi = m.leq(body, m.constant(b.upper));
                m.add_constraint(lo);
                m.add_constraint(hi);
                result.constraint_node_ids[i] = hi;  // primary = upper side
                break;
            }
            case NlBoundType::Free:
                // No constraint; leave node id at -1.
                break;
        }
    }

    // ---------- Objective (first objective only) ----------
    if (prob.n_objs > 0) {
        const NlObjective& o = prob.objectives[0];
        int32_t nl_node = 0;
        bool has_nl = !o.nonlinear.empty();
        if (has_nl) {
            ExprTranslator tr(m, o.nonlinear, result.var_handles);
            nl_node = tr.translate(o.nonlinear.root);
            if (!tr.ok()) {
                record_unsupported("objective: " + tr.reason());
                return result;
            }
        }
        int32_t lin_node = build_linear(m, o.linear, result.var_handles);
        bool has_lin = !o.linear.empty();
        int32_t obj = combine_body(m, nl_node, has_nl, lin_node, has_lin);

        // minimize/maximize reject a bare variable handle; wrap in a sum node.
        if (obj < 0) {
            obj = m.sum({obj});
        }
        if (o.maximize) {
            m.maximize(obj);  // negates internally; objective_id() is the neg node
        } else {
            m.minimize(obj);
        }
        result.objective_node_id = m.objective_id();
    }

    m.close();
    return result;
}

}  // namespace cbls
