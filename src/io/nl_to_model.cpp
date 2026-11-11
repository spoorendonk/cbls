// Adapter: NlProblem -> closed CBLS Model.
//
// Maps AMPL NL operators (opcode.hd numbering) onto CBLS DAG ops. An operator
// outside the supported set marks the result unsupported with a reason rather
// than throwing (io_nl.h's "skip, don't crash"). Supported: +, -, *, /, pow,
// min, max, abs, floor, ceil, sin, cos, tan, exp, log, log10, sqrt, tanh,
// unary minus, plus the linear J/G parts.
//
// Not signpower: AMPL writes it as an imported function, whose `F` segment the
// reader refuses; `Model::signpower_expr` is C++/Python API only.

#include "cbls/bound_propagation.h"
#include "cbls/expr.h"
#include "cbls/io_nl.h"
#include "cbls/model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cbls {

namespace {

// AMPL opcodes we map (subset of opcode.hd). Anything else is unsupported.
enum AmplOp : std::uint8_t {
    OPPLUS = 0,
    OPMINUS = 1,
    OPMULT = 2,
    OPDIV = 3,
    OPPOW = 5,
    MINLIST = 11,
    MAXLIST = 12,
    FLOOR = 13,
    CEIL = 14,
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

// Whether a column side is "no bound": NaN or past the infinity convention.
// Only these are replaced, since the replacement is not entailed and can cut
// off feasible points; any finite bound is kept as is.
bool lo_missing(double lb) {
    return std::isnan(lb) || is_unbounded_below(lb);
}
bool hi_missing(double ub) {
    return std::isnan(ub) || is_unbounded_above(ub);
}

// The finite box edge that replaces a missing side, given the other side
// `other` (kNlInf / -kNlInf when missing too): ±clamp, unless `other` is at or
// beyond it, where ±clamp would exclude the real bound or shrink the box to a
// point (#357); then `clamp` beyond `other`.
double fallback_lo(double other, double clamp) {
    return (std::isfinite(other) && other <= -clamp) ? other - clamp : -clamp;
}
double fallback_hi(double other, double clamp) {
    return (std::isfinite(other) && other >= clamp) ? other + clamp : clamp;
}

// `coef * h`, without a multiply for the unit coefficients.
int32_t scaled(Model& m, double coef, int32_t h) {
    if (coef == 1.0) {
        return h;
    }
    if (coef == -1.0) {
        return m.neg(h);
    }
    return m.prod(m.constant(coef), h);
}

// The sum of already-built parts, or constant 0 for none.
int32_t sum_of(Model& m, const std::vector<int32_t>& parts) {
    if (parts.empty()) {
        return m.constant(0.0);
    }
    if (parts.size() == 1) {
        return parts[0];
    }
    return m.sum(parts);
}

// Translate NL expression trees into CBLS node handles, for one model. On the
// first failure, sets `ok=false`, records a reason and returns constant 0 (the
// caller bails out on !ok). An index past the columns names a defined variable
// (`V` segment), built on first use and shared, so it stays one DAG node.
class ExprTranslator {
public:
    ExprTranslator(Model& m, const NlProblem& prob, const std::vector<int32_t>& vh)
        : m_(m),
          prob_(prob),
          vh_(vh),
          def_handle_(prob.defined_vars.size(), 0),
          def_state_(prob.defined_vars.size(), DefState::Unbuilt) {}

    [[nodiscard]] bool ok() const { return ok_; }
    [[nodiscard]] const std::string& reason() const { return reason_; }

    int32_t translate(const NlExpr& e, int32_t node_idx) {
        if (!ok_) {
            return m_.constant(0.0);
        }
        const NlExprNode& n = e.nodes[node_idx];
        switch (n.kind) {
            case NlNodeKind::Num:
                return m_.constant(n.num);
            case NlNodeKind::Var:
                return var_handle(n.index);
            case NlNodeKind::Op:
                return translate_op(e, n);
        }
        fail("unknown node kind");
        return m_.constant(0.0);
    }

    /// The handle for NL variable `j`: a column, or a defined variable.
    int32_t var_handle(int32_t j) {
        if (j >= 0 && j < static_cast<int32_t>(vh_.size())) {
            return vh_[j];
        }
        const int64_t slot = slot_of(j);
        if (slot < 0) {
            fail("var index out of range");
            return m_.constant(0.0);
        }
        return defined_var(static_cast<std::size_t>(slot));
    }

private:
    enum class DefState : std::uint8_t { Unbuilt, Building, Built };

    /// One defined variable on the dependency walk, with the defined variables
    /// it names and how many of them have been visited.
    struct DefFrame {
        std::size_t slot;
        std::vector<std::size_t> deps;
        std::size_t next = 0;
    };

    /// The defined-variable slot NL variable `j` names, or -1 for a column or
    /// an index past the defined variables.
    [[nodiscard]] int64_t slot_of(int32_t j) const {
        const int64_t slot = static_cast<int64_t>(j) - static_cast<int64_t>(vh_.size());
        return (j >= 0 && slot >= 0 && slot < static_cast<int64_t>(def_state_.size())) ? slot : -1;
    }

    [[nodiscard]] std::string label(std::size_t slot) const {
        return "defined variable " + std::to_string(vh_.size() + slot);
    }

    // Build defined variable `slot` after its dependencies, deepest first, on
    // an explicit stack: a chain like `s[t] = s[t-1] + x[t]` would overflow the
    // native one on a long horizon. Each build then recurses only as deep as
    // its own expression.
    int32_t defined_var(std::size_t slot) {
        if (def_state_[slot] == DefState::Built) {
            return def_handle_[slot];
        }
        std::vector<DefFrame> stack;
        enter(slot, stack);
        while (ok_ && !stack.empty()) {
            DefFrame& top = stack.back();
            if (top.next < top.deps.size()) {
                const std::size_t dep = top.deps[top.next++];
                if (def_state_[dep] == DefState::Unbuilt) {
                    enter(dep, stack);  // invalidates `top`; not used again
                } else if (def_state_[dep] == DefState::Building) {
                    fail(label(dep) + " refers to itself, directly or through others");
                }
                continue;
            }
            build(top.slot);
            stack.pop_back();
        }
        return ok_ ? def_handle_[slot] : m_.constant(0.0);
    }

    // Mark `slot` in progress and push it with the defined variables it names.
    void enter(std::size_t slot, std::vector<DefFrame>& stack) {
        const NlDefinedVar& dv = prob_.defined_vars[slot];
        if (!dv.defined) {
            fail(label(slot) + " has no V segment");
            return;
        }
        def_state_[slot] = DefState::Building;
        DefFrame frame{slot, {}, 0};
        auto add = [&](int32_t j) {
            const int64_t s = slot_of(j);
            if (s >= 0) {
                frame.deps.push_back(static_cast<std::size_t>(s));
            }
        };
        for (const NlLinTerm& t : dv.linear) {
            add(t.var);
        }
        for (const NlExprNode& n : dv.nonlinear.nodes) {
            if (n.kind == NlNodeKind::Var) {
                add(n.index);
            }
        }
        stack.push_back(std::move(frame));
    }

    // Build `slot`, whose defined-variable dependencies are all built.
    void build(std::size_t slot) {
        const NlDefinedVar& dv = prob_.defined_vars[slot];
        std::vector<int32_t> parts;
        parts.reserve(dv.linear.size() + 1);
        for (const NlLinTerm& t : dv.linear) {
            const int32_t h = var_handle(t.var);
            if (!ok_) {
                return;
            }
            parts.push_back(scaled(m_, t.coef, h));
        }
        if (!dv.nonlinear.empty()) {
            parts.push_back(translate(dv.nonlinear, dv.nonlinear.root));
            if (!ok_) {
                return;
            }
        }
        def_handle_[slot] = sum_of(m_, parts);
        def_state_[slot] = DefState::Built;
    }

    void fail(const std::string& why) {
        if (ok_) {
            ok_ = false;
            reason_ = why;
        }
    }

    // Fold an n-ary child list into binary CBLS ops where needed.
    int32_t translate_op(const NlExpr& e, const NlExprNode& n) {
        std::vector<int32_t> kids;
        kids.reserve(n.children.size());
        for (int32_t c : n.children) {
            kids.push_back(translate(e, c));
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
            case FLOOR:
                return m_.floor_expr(kids[0]);
            case CEIL:
                return m_.ceil_expr(kids[0]);
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
    const NlProblem& prob_;
    const std::vector<int32_t>& vh_;
    std::vector<int32_t> def_handle_;
    std::vector<DefState> def_state_;
    bool ok_ = true;
    std::string reason_;
};

// Build the linear part sum_j coef_j * x_j as a CBLS node (or constant 0).
// Every term names a column; `build_body` refuses the body otherwise.
int32_t build_linear(Model& m, const std::vector<NlLinTerm>& terms,
                     const std::vector<int32_t>& vh) {
    std::vector<int32_t> parts;
    parts.reserve(terms.size());
    for (const NlLinTerm& t : terms) {
        parts.push_back(scaled(m, t.coef, vh[static_cast<std::size_t>(t.var)]));
    }
    return sum_of(m, parts);
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

/// A translated constraint/objective body, or the reason it could not be built:
/// data, not an exception, per the NL "skip, don't throw" contract.
struct BodyResult {
    int32_t node = 0;
    bool ok = true;
    std::string reason;
};

/// Translate the nonlinear part, build the linear part and combine them, for a
/// constraint or the objective; the caller labels the failure.
BodyResult build_body(Model& m, ExprTranslator& tr, const NlExpr& nonlinear,
                      const std::vector<NlLinTerm>& linear,
                      const std::vector<int32_t>& var_handles) {
    BodyResult out;
    int32_t nl_node = 0;
    const bool has_nl = !nonlinear.empty();
    if (has_nl) {
        nl_node = tr.translate(nonlinear, nonlinear.root);
        if (!tr.ok()) {
            out.ok = false;
            out.reason = tr.reason();
            return out;
        }
    }
    // A J/G term must name a column (AMPL never writes a defined variable
    // there); refused, since dropping it would relax the row.
    for (const NlLinTerm& t : linear) {
        if (t.var < 0 || t.var >= static_cast<int32_t>(var_handles.size())) {
            out.ok = false;
            out.reason = "linear term names variable " + std::to_string(t.var) + ", not a column";
            return out;
        }
    }
    const int32_t lin_node = build_linear(m, linear, var_handles);
    out.node = combine_body(m, nl_node, has_nl, lin_node, !linear.empty());
    return out;
}

/// Translate a constraint's bound into the one or two CBLS comparisons it stands
/// for, and return the primary constraint node — the upper side of a range. A
/// Free row builds no constraint and returns -1.
int32_t add_bound_constraints(Model& m, const NlConBound& b, int32_t body) {
    switch (b.type) {
        case NlBoundType::Upper: {
            const int32_t cn = m.leq(body, m.constant(b.upper));
            m.add_constraint(cn);
            return cn;
        }
        case NlBoundType::Lower: {
            const int32_t cn = m.geq(body, m.constant(b.lower));
            m.add_constraint(cn);
            return cn;
        }
        case NlBoundType::Equal: {
            const int32_t cn = m.eq_expr(body, m.constant(b.lower));
            m.add_constraint(cn);
            return cn;
        }
        case NlBoundType::Range: {
            // lower <= body <= upper -> two constraints.
            const int32_t lo = m.geq(body, m.constant(b.lower));
            const int32_t hi = m.leq(body, m.constant(b.upper));
            m.add_constraint(lo);
            m.add_constraint(hi);
            return hi;  // primary = upper side
        }
        case NlBoundType::Free:
            break;
    }
    return -1;
}

/// A row body's two-sided bounds, as named fields rather than two `double&`
/// out-parameters, whose silent transposition would invert the row's sense.
struct RowBounds {
    double lo = -kNlInf;
    double hi = kNlInf;
};

/// The row's body bounds `lo <= body <= hi`, read through the bound *type*, as
/// the constraint builder is, so a stale unused side never yields a bound no
/// constraint enforces (#120). Nullopt for a Free row.
std::optional<RowBounds> linear_row_bounds(const NlConBound& b) {
    RowBounds r;
    switch (b.type) {
        case NlBoundType::Range:
            r.lo = b.lower;
            r.hi = b.upper;
            return r;
        case NlBoundType::Upper:
            r.hi = b.upper;
            return r;
        case NlBoundType::Lower:
            r.lo = b.lower;
            return r;
        case NlBoundType::Equal:
            r.lo = b.lower;
            r.hi = b.lower;
            return r;
        case NlBoundType::Free:
            return std::nullopt;
    }
    return std::nullopt;
}

/// The bounds of a row's linear part, ignoring its nonlinear part, or nullopt
/// for an empty or Free row or one naming an out-of-range column, which
/// `propagate_bounds` would throw on (io_nl.h: "skip, don't throw"); omitting
/// it costs tightening, not validity, and `build_body` refuses the model.
std::optional<RowBounds> linear_part_bounds(const NlConstraint& c, std::size_t n_cols) {
    if (c.linear.empty()) {
        return std::nullopt;
    }
    for (const NlLinTerm& t : c.linear) {
        if (t.var < 0 || static_cast<std::size_t>(t.var) >= n_cols) {
            return std::nullopt;
        }
    }
    return linear_row_bounds(c.bound);
}

/// The constant an expression is: 0 when it is empty, the value of a lone
/// constant root, nullopt for anything else. AMPL writes a `C<i>` segment for
/// every row, `n0` for a linear one, and the reader keeps it as a Num node.
std::optional<double> constant_part(const NlExpr& e) {
    if (e.empty()) {
        return 0.0;
    }
    const auto root = static_cast<std::size_t>(e.root);
    if (root < e.nodes.size() && e.nodes[root].kind == NlNodeKind::Num) {
        return e.nodes[root].num;
    }
    return std::nullopt;
}

/// The rows propagation reads: `linear_part_bounds` of a row whose nonlinear
/// part is empty or a lone constant `k` (every linear row of an AMPL export,
/// #342, #344), as `lo - k <= linear <= hi - k`. The row is built verbatim.
std::optional<RowBounds> linear_row(const NlConstraint& c, std::size_t n_cols) {
    const std::optional<double> k = constant_part(c.nonlinear);
    if (!k.has_value()) {
        return std::nullopt;
    }
    std::optional<RowBounds> rb = linear_part_bounds(c, n_cols);
    if (!rb.has_value()) {
        return std::nullopt;
    }
    if (!is_unbounded_below(rb->lo)) {
        rb->lo -= *k;
    }
    if (!is_unbounded_above(rb->hi)) {
        rb->hi -= *k;
    }
    return rb;
}

/// Derive implied column bounds from the linear rows (`linear_row`); other rows
/// are skipped (activity arithmetic does not apply; costs tightening, not
/// validity). Rows are views into two flat local arrays, not copies.
BoundPropagationStats tighten_column_bounds(const NlProblem& prob, const NlToModelOptions& opts,
                                            std::vector<double>& lb, std::vector<double>& ub,
                                            const std::vector<uint8_t>& integral) {
    std::vector<int32_t> cols;
    std::vector<double> coefs;
    std::vector<LinearRow> rows;
    rows.reserve(prob.constraints.size());
    // (start, nnz) per row, resolved to pointers only once `cols`/`coefs` stop
    // growing, since they reallocate.
    std::vector<std::pair<std::size_t, int32_t>> spans;
    spans.reserve(prob.constraints.size());
    for (const NlConstraint& c : prob.constraints) {
        const std::optional<RowBounds> rb = linear_row(c, lb.size());
        if (!rb.has_value()) {
            continue;
        }
        LinearRow row;
        row.lo = rb->lo;
        row.hi = rb->hi;
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

/// The per-column search box handed to variable creation, and which columns are
/// integral. `lb`/`ub` are still in NL units here: the `inf_clamp` fallback is
/// applied per column when the variable itself is created.
struct ColumnBoxes {
    std::vector<double> lb;
    std::vector<double> ub;
    std::vector<uint8_t> integral;
};

/// The declared column boxes, tightened by propagation when it is enabled.
/// Runs before variable creation; a derived bound is entailed, so it is then
/// treated like a declared one.
ColumnBoxes column_boxes(const NlProblem& prob, const NlToModelOptions& opts,
                         BoundPropagationStats& stats) {
    const auto n_cols = static_cast<std::size_t>(prob.n_vars);
    ColumnBoxes box;
    box.lb.assign(n_cols, -kNlInf);
    box.ub.assign(n_cols, kNlInf);
    box.integral.assign(n_cols, 0);
    for (std::size_t j = 0; j < n_cols; ++j) {
        if (j < prob.var_bounds.size()) {
            // NaN means "no bound", as lo_missing/hi_missing read it; must be
            // mapped, else propagate_bounds throws (io_nl.h: skip, don't throw).
            const double declared_lb = prob.var_bounds[j].lower;
            const double declared_ub = prob.var_bounds[j].upper;
            box.lb[j] = std::isnan(declared_lb) ? -kNlInf : declared_lb;
            box.ub[j] = std::isnan(declared_ub) ? kNlInf : declared_ub;
        }
        box.integral[j] = (j < prob.var_is_discrete.size() && prob.var_is_discrete[j] != 0) ? 1 : 0;
    }
    if (!opts.propagate_bounds) {
        return box;
    }
    const std::vector<double> raw_lb = box.lb;
    const std::vector<double> raw_ub = box.ub;
    stats = tighten_column_bounds(prob, opts, box.lb, box.ub, box.integral);
    if (stats.infeasible) {
        // An empty linear part (infeasible or numerical trouble): search the
        // declared box rather than a derived empty one.
        box.lb = raw_lb;
        box.ub = raw_ub;
        // Nothing was applied: only the verdict survives.
        stats = BoundPropagationStats{};
        stats.infeasible = true;
    }
    return box;
}

/// One CBLS variable built from one NL column, and whether a clamp narrowed it;
/// the caller counts narrowed columns, not narrowings (the int_inf_clamp
/// fallback and the int32 clip can both bite on one column).
struct ColumnVar {
    int32_t handle = 0;
    bool clamped = false;
};

/// Map one NL column onto an Int (discrete, from the header counts and Gay's
/// ordering in nl_reader.cpp) or a Float variable.
ColumnVar add_column(Model& m, int32_t j, bool discrete, double box_lb, double box_ub,
                     const NlToModelOptions& opts) {
    ColumnVar out;
    const bool no_lb = lo_missing(box_lb);
    const bool no_ub = hi_missing(box_ub);
    double lb = no_lb ? fallback_lo(no_ub ? kNlInf : box_ub, opts.inf_clamp) : box_lb;
    double ub = no_ub ? fallback_hi(no_lb ? -kNlInf : box_lb, opts.inf_clamp) : box_ub;
    const bool clamp_used = no_lb || no_ub;
    if (lb > ub) {
        std::swap(lb, ub);  // defensive: degenerate bound ordering
    }
    const std::string name = "x" + std::to_string(j);
    if (!discrete) {
        out.clamped = clamp_used;
        out.handle = m.float_var(lb, ub, name);
        return out;
    }
    // Round inward to the integers. Only a missing side (after propagation,
    // whose bounds are entailed) falls back to int_inf_clamp, since a ±1e9
    // integer box is not searchable; placed relative to the rounded other
    // side, as for Float (#357).
    const double fin_lb = std::ceil(lb - 1e-9);
    const double fin_ub = std::floor(ub + 1e-9);
    const double ilb = no_lb ? fallback_lo(no_ub ? kNlInf : fin_ub, opts.int_inf_clamp) : fin_lb;
    const double iub = no_ub ? fallback_hi(no_lb ? -kNlInf : fin_lb, opts.int_inf_clamp) : fin_ub;
    const bool int_clamped = no_lb || no_ub;
    // A finite bound may lie beyond the int range: clip, else the casts below
    // are UB. A narrowing, so it counts as clamped.
    constexpr auto kIntLo = static_cast<double>(std::numeric_limits<int>::min());
    constexpr auto kIntHi = static_cast<double>(std::numeric_limits<int>::max());
    const double clipped_lb = std::min(std::max(ilb, kIntLo), kIntHi);
    double clipped_ub = std::min(std::max(iub, kIntLo), kIntHi);
    out.clamped = int_clamped || clipped_lb != ilb || clipped_ub != iub;
    // Bounds admitting no integer collapse to the point clipped_lb, so the
    // model still closes; the rows register the violation, not the reader.
    clipped_ub = std::max(clipped_lb, clipped_ub);
    out.handle = m.int_var(static_cast<int>(clipped_lb), static_cast<int>(clipped_ub), name);
    return out;
}

/// Seed variable `id`'s value from the NL `x` segment's entry for column `col`,
/// where present.
void seed_value(Model& m, const NlProblem& prob, int32_t id, int32_t col) {
    if (col < 0 || static_cast<std::size_t>(col) >= prob.initial_x.size()) {
        return;
    }
    double x0 = prob.initial_x[static_cast<std::size_t>(col)];
    if (!std::isfinite(x0)) {
        return;
    }
    if (m.var(id).type == VarType::Int) {
        x0 = std::round(x0);  // an Int column must not start fractional
    }
    m.var_mut(id).value = std::min(std::max(x0, m.var(id).lb), m.var(id).ub);
}

/// Seed initial values from the NL `x` segment where present. Variable `id` is
/// NL column `id`.
void seed_initial_values(Model& m, const NlProblem& prob) {
    for (int32_t j = 0; j < prob.n_vars; ++j) {
        seed_value(m, prob, j, j);
    }
}

/// The model: one variable per column, one or two constraints per row.
void build_plain(const NlProblem& prob, const NlToModelOptions& opts, NlToModelResult& result) {
    Model& m = result.model;

    const ColumnBoxes box = column_boxes(prob, opts, result.bound_stats);

    result.var_handles.reserve(prob.n_vars);
    for (int32_t j = 0; j < prob.n_vars; ++j) {
        const auto col = static_cast<std::size_t>(j);
        const ColumnVar cv =
            add_column(m, j, box.integral[col] != 0, box.lb[col], box.ub[col], opts);
        if (cv.clamped) {
            ++result.n_clamped_columns;
        }
        result.var_handles.push_back(cv.handle);
    }

    seed_initial_values(m, prob);

    auto record_unsupported = [&](const std::string& reason) {
        result.supported = false;
        result.skipped_reasons.push_back(reason);
    };

    // One translator for every body, so a defined variable used in several
    // rows is built once and shared.
    ExprTranslator tr(m, prob, result.var_handles);
    result.constraint_node_ids.assign(prob.n_cons, -1);
    for (int32_t i = 0; i < prob.n_cons; ++i) {
        const NlConstraint& c = prob.constraints[i];
        const BodyResult body = build_body(m, tr, c.nonlinear, c.linear, result.var_handles);
        if (!body.ok) {
            record_unsupported("constraint " + std::to_string(i) + ": " + body.reason);
            return;
        }
        result.constraint_node_ids[i] = add_bound_constraints(m, c.bound, body.node);
    }

    // First objective only.
    if (prob.n_objs > 0) {
        const NlObjective& o = prob.objectives[0];
        const BodyResult body = build_body(m, tr, o.nonlinear, o.linear, result.var_handles);
        if (!body.ok) {
            record_unsupported("objective: " + body.reason);
            return;
        }
        int32_t obj = body.node;
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
}

}  // namespace

NlToModelResult nl_to_model(const NlProblem& prob, const NlToModelOptions& opts) {
    NlToModelResult result;
    build_plain(prob, opts, result);
    return result;
}

std::vector<double> NlToModelResult::original_values(const Model& model) const {
    const auto& vars = model.variables();
    std::vector<double> x;
    x.reserve(var_handles.size());
    for (const int32_t h : var_handles) {
        const int32_t id = handle_to_var_id(h);
        if (id < 0 || static_cast<std::size_t>(id) >= vars.size()) {
            throw std::invalid_argument("original_values: model does not match this build");
        }
        x.push_back(vars[static_cast<std::size_t>(id)].value);
    }
    return x;
}

}  // namespace cbls
