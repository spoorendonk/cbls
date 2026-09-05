// Adapter: NlProblem -> closed CBLS Model.
//
// Maps AMPL NL operators (opcode.hd numbering) onto CBLS DAG ops. Operators
// outside the supported set do not throw: the result is marked unsupported with
// a human-readable reason, mirroring the "skip, don't crash" contract in
// io_nl.h. The supported set is the MINLPLib op subset CBLS can express today
// (+, -, *, /, pow, min, max, abs, sin, cos, tan, exp, log, log10, sqrt,
// signpower, tanh, unary minus) plus the linear J/G parts.

#include "cbls/bound_propagation.h"
#include "cbls/expr.h"
#include "cbls/io_nl.h"
#include "cbls/model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
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

// Supply a finite bound where the column has none. A bound that is finite —
// declared in the file or derived by propagation — is honoured as written,
// however wide; only "no bound" is replaced, because the replacement is not
// entailed by the constraints and so can cut off feasible points.
double clamp_lo(double lb, double inf_clamp) {
    return (std::isnan(lb) || is_unbounded_below(lb)) ? -inf_clamp : lb;
}
double clamp_hi(double ub, double inf_clamp) {
    return (std::isnan(ub) || is_unbounded_above(ub)) ? inf_clamp : ub;
}

// Translate one NL expression tree into a CBLS expression node handle. On the
// first unsupported opcode, sets `ok=false`, records a reason, and returns a
// constant 0 (the caller bails out on !ok). `var_handles` maps NL var index to
// CBLS handle.
class ExprTranslator {
public:
    ExprTranslator(Model& m, const NlExpr& e, const std::vector<int32_t>& vh)
        : m_(m), e_(e), vh_(vh) {}

    [[nodiscard]] bool ok() const { return ok_; }
    [[nodiscard]] const std::string& reason() const { return reason_; }

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

/// The row's body bounds `lo <= body <= hi`, read through the bound *type* so a
/// stale value on the unused side cannot be mistaken for a real bound. The
/// constraint builder is type-gated the same way; deriving a bound from a side
/// no constraint enforces would be exactly the unsoundness #120 exists to
/// remove. Returns false for a Free row, which builds no constraint at all.
bool linear_row_bounds(const NlConBound& b, double& lo, double& hi) {
    lo = -kNlInf;
    hi = kNlInf;
    switch (b.type) {
        case NlBoundType::Range:
            lo = b.lower;
            hi = b.upper;
            return true;
        case NlBoundType::Upper:
            hi = b.upper;
            return true;
        case NlBoundType::Lower:
            lo = b.lower;
            return true;
        case NlBoundType::Equal:
            lo = b.lower;
            hi = b.lower;
            return true;
        case NlBoundType::Free:
            return false;
    }
    return false;
}

/// Derive implied column bounds from the *purely linear* rows. A row with a
/// nonlinear part is skipped: activity arithmetic does not apply to it, and
/// omitting a row only costs tightening, never validity. Nonlinear presolve is
/// deliberately out of scope here.
///
/// Terms are packed into two flat arrays the caller owns for the duration, and
/// the rows are handed to propagation as views into them rather than as copies.
BoundPropagationStats tighten_column_bounds(const NlProblem& prob, const NlToModelOptions& opts,
                                            std::vector<double>& lb, std::vector<double>& ub,
                                            const std::vector<uint8_t>& integral) {
    std::vector<int32_t> cols;
    std::vector<double> coefs;
    std::vector<LinearRow> rows;
    rows.reserve(prob.constraints.size());
    // (start, nnz) per accepted row, resolved to pointers once `cols`/`coefs`
    // have stopped growing — they reallocate as rows are appended.
    std::vector<std::pair<std::size_t, int32_t>> spans;
    spans.reserve(prob.constraints.size());
    for (const NlConstraint& c : prob.constraints) {
        if (!c.nonlinear.empty() || c.linear.empty()) {
            continue;
        }
        LinearRow row;
        if (!linear_row_bounds(c.bound, row.lo, row.hi)) {
            continue;
        }
        // `propagate_bounds` rejects an out-of-range column outright, but
        // `build_linear` drops one silently and io_nl.h promises "skip, don't
        // throw" on a malformed file. Drop the whole row instead: omitting a row
        // costs tightening, never validity.
        bool in_range = true;
        for (const NlLinTerm& t : c.linear) {
            if (t.var < 0 || static_cast<std::size_t>(t.var) >= lb.size()) {
                in_range = false;
                break;
            }
        }
        if (!in_range) {
            continue;
        }
        const std::size_t start = cols.size();
        for (const NlLinTerm& t : c.linear) {
            cols.push_back(t.var);
            coefs.push_back(t.coef);
        }
        spans.emplace_back(start, static_cast<int32_t>(c.linear.size()));
        rows.push_back(row);
    }
    for (std::size_t i = 0; i < rows.size(); ++i) {
        rows[i].cols = cols.data() + spans[i].first;
        rows[i].coefs = coefs.data() + spans[i].first;
        rows[i].nnz = spans[i].second;
    }
    BoundPropagationOptions popts;
    popts.max_passes = opts.max_propagation_passes;
    return propagate_bounds(rows, integral, lb, ub, popts);
}

}  // namespace

NlToModelResult nl_to_model(const NlProblem& prob, const NlToModelOptions& opts) {
    NlToModelResult result;
    Model& m = result.model;

    // ---------- Variables ----------
    // Integer/binary NL columns become Int variables so the search respects
    // integrality; everything else is a Float. `var_is_discrete` comes from the
    // NL header counts plus Gay's variable ordering (see nl_reader.cpp).
    // ---------- Implied bounds ----------
    // Run before variable creation so the derived box is what the engine sees.
    // A bound propagation derives is entailed by the constraints, so from here
    // on it is treated exactly like one the file declared.
    const std::size_t n_cols = static_cast<std::size_t>(prob.n_vars);
    std::vector<double> col_lb(n_cols, -kNlInf);
    std::vector<double> col_ub(n_cols, kNlInf);
    std::vector<uint8_t> integral(n_cols, 0);
    for (std::size_t j = 0; j < n_cols; ++j) {
        if (j < prob.var_bounds.size()) {
            // A NaN bound is "no bound" here, as clamp_lo/clamp_hi already read
            // it. Handing one to propagate_bounds would throw, and io_nl.h
            // promises to skip a malformed file rather than throw out of the
            // adapter.
            const double declared_lb = prob.var_bounds[j].lower;
            const double declared_ub = prob.var_bounds[j].upper;
            col_lb[j] = std::isnan(declared_lb) ? -kNlInf : declared_lb;
            col_ub[j] = std::isnan(declared_ub) ? kNlInf : declared_ub;
        }
        integral[j] = (j < prob.var_is_discrete.size() && prob.var_is_discrete[j] != 0) ? 1 : 0;
    }
    if (opts.propagate_bounds) {
        const std::vector<double> raw_lb = col_lb;
        const std::vector<double> raw_ub = col_ub;
        result.bound_stats = tighten_column_bounds(prob, opts, col_lb, col_ub, integral);
        if (result.bound_stats.infeasible) {
            // Propagation proved the linear part empty. That is either a
            // genuinely infeasible instance or numerical trouble; either way the
            // honest thing is to hand the search the box the file declared and
            // let it report what it finds, rather than a derived empty one.
            col_lb = raw_lb;
            col_ub = raw_ub;
            // Nothing was applied, so the counts must not say otherwise; only
            // the verdict survives.
            result.bound_stats = BoundPropagationStats{};
            result.bound_stats.infeasible = true;
        }
    }

    result.var_handles.reserve(prob.n_vars);
    for (int32_t j = 0; j < prob.n_vars; ++j) {
        const std::size_t col = static_cast<std::size_t>(j);
        double lb = clamp_lo(col_lb[col], opts.inf_clamp);
        double ub = clamp_hi(col_ub[col], opts.inf_clamp);
        const bool clamp_used = lb != col_lb[col] || ub != col_ub[col];
        if (lb > ub) {
            std::swap(lb, ub);  // defensive: degenerate bound ordering
        }
        const bool discrete = j < static_cast<int32_t>(prob.var_is_discrete.size()) &&
                              prob.var_is_discrete[static_cast<size_t>(j)] != 0;
        if (discrete) {
            // Tighten to the integers inside [lb, ub]. A bound that exists —
            // declared or derived — is always honoured; only a genuinely
            // infinite one falls back to int_inf_clamp, since a ±1e9 integer box
            // is not a searchable domain.
            // Propagated, not as-declared: a derived bound is entailed, so it
            // rightly suppresses the unsound int_inf_clamp fallback.
            const double propagated_lb = col_lb[col];
            const double propagated_ub = col_ub[col];
            double ilb =
                is_unbounded_below(propagated_lb) ? -opts.int_inf_clamp : std::ceil(lb - 1e-9);
            double iub =
                is_unbounded_above(propagated_ub) ? opts.int_inf_clamp : std::floor(ub + 1e-9);
            const bool int_clamped =
                is_unbounded_below(propagated_lb) || is_unbounded_above(propagated_ub);
            // A finite bound is honoured however wide, so one beyond the int
            // range arrives here unclipped and would make the casts below UB.
            // `Model::int_var` takes an int; that representational limit narrows
            // the column, so it counts as clamped too.
            constexpr double kIntLo = static_cast<double>(std::numeric_limits<int>::min());
            constexpr double kIntHi = static_cast<double>(std::numeric_limits<int>::max());
            const double clipped_lb = std::min(std::max(ilb, kIntLo), kIntHi);
            const double clipped_ub = std::min(std::max(iub, kIntLo), kIntHi);
            // One column, one count: the fallback and the int32 clip can both
            // narrow the same column.
            if (int_clamped || clipped_lb != ilb || clipped_ub != iub) {
                ++result.n_clamped_columns;
            }
            ilb = clipped_lb;
            iub = clipped_ub;
            // Bounds that admit no integer (degenerate) collapse to the single
            // point ilb, so the model still closes; the row's constraints will
            // register the violation rather than the reader silently dropping it.
            iub = std::max(ilb, iub);
            result.var_handles.push_back(
                m.int_var(static_cast<int>(ilb), static_cast<int>(iub), "x" + std::to_string(j)));
        } else {
            if (clamp_used) {
                ++result.n_clamped_columns;
            }
            result.var_handles.push_back(m.float_var(lb, ub, "x" + std::to_string(j)));
        }
    }

    // Seed initial values from the NL `x` segment where present.
    for (int32_t j = 0; j < prob.n_vars && j < static_cast<int32_t>(prob.initial_x.size()); ++j) {
        double x0 = prob.initial_x[j];
        if (std::isfinite(x0)) {
            if (m.var(j).type == VarType::Int) {
                x0 = std::round(x0);  // an Int column must not start fractional
            }
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
