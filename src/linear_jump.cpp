#include "cbls/linear_jump.h"

#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace cbls {

namespace {

bool is_literal(const Model& model, const ChildRef& ref) {
    // Exactly dag.cpp's `child_is_const`: comparison_residual's sentinel flag is
    // about a literal Const node, not about a constant-valued subtree.
    return !ref.is_var && model.nodes()[static_cast<size_t>(ref.id)].op == NodeOp::Const;
}

// Sparse partials of one side of an Eq row: a variable child is its own slope,
// a node is swept. Sorted by variable id for the merge in build_row.
void side_partials(const Model& model, const ChildRef& ref,
                   std::vector<std::pair<int32_t, double>>& out) {
    out.clear();
    if (ref.is_var) {
        out.emplace_back(ref.id, 1.0);
        return;
    }
    compute_partials_sparse(model, ref.id, out);
    std::sort(out.begin(), out.end());
}

// The sign `local_derivative` gives an Eq node: 0 for a zero OR NaN difference.
double eq_sign(double diff) {
    if (diff > 0.0) {
        return 1.0;
    }
    if (diff < 0.0) {
        return -1.0;
    }
    return 0.0;
}

}  // namespace

LinearJumpScorer::LinearJumpScorer(const Model& model) : model_(model) {}

void LinearJumpScorer::resize_rows(size_t n) {
    rows_.resize(n);
}

void LinearJumpScorer::set_row_eligible(int32_t ci, bool eligible) {
    Row& row = rows_.at(static_cast<size_t>(ci));
    row = Row{};  // drops the slopes and their capacity: the body changed
    row.state = eligible ? RowState::Pending : RowState::Ineligible;
}

bool LinearJumpScorer::row_eligible(int32_t ci) const {
    return rows_.at(static_cast<size_t>(ci)).state != RowState::Ineligible;
}

// Slopes of one eligible row, built on first use.
//
// Leq/Lt/Geq/Gt: the residual is p - q (with p = child1 for Geq/Gt), so r is the
// node's own partial -- one sweep of the node, the same `reverse_sweep` that
// `compute_partial(node, v)` runs, hence bit-identical to it (`newton_exact`).
//
// Eq: the node's partial is sign(p - q) * r, sign-dependent and 0 when
// satisfied, so it cannot be cached; r = s0 - s1 comes from sweeping each child.
// That equals the Eq node's sweep up to the factor sign exactly -- negation
// commutes with every rounding -- only when one side is a literal Const, so that
// no variable's adjoint mixes paths from both sides. Otherwise the Newton step
// keeps calling compute_partial for this row; the jump SCORE still uses r.
//
// Every slope must be finite, or the row goes back to ineligible: an affine cone
// through Prod by an infinite constant has no linear model. (Div by a constant
// below 1e-15 is the opposite case -- local derivative 0, value +/-inf -- and is
// caught in `prepare` by its non-finite side, not here.)
void LinearJumpScorer::build_row(int32_t ci) {
    Row& row = rows_[static_cast<size_t>(ci)];
    const int32_t nid = model_.constraint_ids()[static_cast<size_t>(ci)];
    const ExprNode& nd = model_.nodes()[static_cast<size_t>(nid)];
    const ConstSpan<ChildRef> children = model_.children(nd);
    const bool swap = nd.op == NodeOp::Geq || nd.op == NodeOp::Gt;
    const ChildRef p = children[swap ? 1 : 0];
    const ChildRef q = children[swap ? 0 : 1];
    row.p_id = p.id;
    row.p_is_var = p.is_var;
    row.q_id = q.id;
    row.q_is_var = q.is_var;
    row.p_literal = is_literal(model_, p);
    row.q_literal = is_literal(model_, q);
    row.is_abs = nd.op == NodeOp::Eq;
    row.strict = nd.op == NodeOp::Lt || nd.op == NodeOp::Gt;

    merged_.clear();
    if (row.is_abs) {
        side_partials(model_, p, side0_);
        side_partials(model_, q, side1_);
        size_t i = 0;
        size_t k = 0;
        while (i < side0_.size() || k < side1_.size()) {
            if (k == side1_.size() || (i < side0_.size() && side0_[i].first < side1_[k].first)) {
                merged_.push_back(side0_[i++]);
            } else if (i == side0_.size() || side1_[k].first < side0_[i].first) {
                merged_.emplace_back(side1_[k].first, -side1_[k].second);
                ++k;
            } else {
                merged_.emplace_back(side0_[i].first, side0_[i].second - side1_[k].second);
                ++i;
                ++k;
            }
        }
        row.newton_exact = row.p_literal || row.q_literal;
    } else {
        compute_partials_sparse(model_, nid, merged_);
        std::sort(merged_.begin(), merged_.end());
        row.newton_exact = true;
    }

    size_t n = 0;
    for (const auto& e : merged_) {
        if (!std::isfinite(e.second)) {
            set_row_eligible(ci, false);
            return;
        }
        n += e.second != 0.0 ? 1 : 0;
    }
    row.vars = std::make_unique<int32_t[]>(n);
    row.slopes = std::make_unique<double[]>(n);
    row.count = static_cast<uint32_t>(n);
    size_t k = 0;
    for (const auto& e : merged_) {
        if (e.second != 0.0) {
            row.vars[k] = e.first;
            row.slopes[k] = e.second;
            ++k;
        }
    }
    row.state = RowState::Ready;
}

double LinearJumpScorer::slope_of(const Row& row, int32_t var_id) {
    const int32_t* first = row.vars.get();
    const int32_t* last = first + row.count;
    const int32_t* it = std::lower_bound(first, last, var_id);
    if (it == last || *it != var_id) {
        return 0.0;
    }
    return row.slopes[static_cast<size_t>(it - first)];
}

double LinearJumpScorer::child_value(int32_t id, bool is_var) const {
    if (is_var) {
        return model_.variables()[static_cast<size_t>(id)].value;
    }
    return model_.node_values()[static_cast<size_t>(id)];
}

bool LinearJumpScorer::prepare(int32_t var_id, const std::vector<double>& weights) {
    const std::vector<int32_t>& cids = model_.constraint_ids();
    if (rows_.size() != cids.size()) {
        throw std::logic_error(
            "LinearJumpScorer::prepare: the model has a different row count than this scorer; "
            "an extension must be followed by resize_rows and set_row_eligible on its rows");
    }
    terms_.clear();
    x0_ = model_.var(var_id).value;
    const ConstSpan<int32_t> gv = model_.constraints_of_var(var_id);
    // w * (new - old) with both clamped, hence finite, is exactly 0 at w == 0, so
    // such a row cannot change the sum whatever its shape.
    //
    // Refused rows first, before anything is built: on a mixed model a variable
    // that falls back anyway should not pay slope sweeps for its linear rows.
    bool ok = std::isfinite(x0_);
    for (size_t k = 0; ok && k < gv.size(); ++k) {
        const auto c = static_cast<size_t>(gv[k]);
        ok = weights[c] == 0.0 || rows_[c].state != RowState::Ineligible;
    }
    const std::vector<double>& node_values = model_.node_values();
    for (size_t k = 0; ok && k < gv.size(); ++k) {
        const int32_t c = gv[k];
        const double w = weights[static_cast<size_t>(c)];
        if (w == 0.0) {
            continue;
        }
        Row& row = rows_[static_cast<size_t>(c)];
        if (row.state == RowState::Pending) {
            build_row(c);
        }
        if (row.state != RowState::Ready) {
            ok = false;  // demoted by its build: a non-finite slope
            break;
        }
        const double p = child_value(row.p_id, row.p_is_var);
        const double q = child_value(row.q_id, row.q_is_var);
        // A computed side must be finite for `p + r D` to be the DAG's value; a
        // literal side may be the +/-inf bound sentinel (the objective row's
        // bound opens at +inf) but not NaN.
        //
        // Checked BEFORE the zero-slope skip below: `Div` by a constant below
        // 1e-15 has local derivative 0 yet evaluates to +/-inf by the numerator's
        // sign, so a zero slope alone does not make the row constant -- but that
        // side is never finite, which this catches. (At exactly 1e-15 `evaluate`
        // divides while `local_derivative` still reports 0: a pre-existing AD
        // inconsistency this inherits, as every Newton step already does.)
        if ((row.p_literal ? std::isnan(p) : !std::isfinite(p)) ||
            (row.q_literal ? std::isnan(q) : !std::isfinite(q))) {
            ok = false;
            break;
        }
        const double r = slope_of(row, var_id);
        if (r == 0.0) {
            // Cancelled, or through a zero constant factor, with finite sides: the
            // DAG re-evaluates this row to the value it has, so its difference is
            // exactly 0.
            continue;
        }
        terms_.push_back(Term{w, clamped_node_violation(node_values[static_cast<size_t>(cids[c])]),
                              p, q, r, row.p_literal, row.q_literal, row.is_abs, row.strict});
    }
    if (!ok) {
        terms_.clear();
        ++fallback_prepares_;
        return false;
    }
    ++fast_prepares_;
    return true;
}

double LinearJumpScorer::delta(double j) const {
    const double d = j - x0_;
    double sum = 0.0;
    for (const Term& t : terms_) {
        // The whole move lands on p. Which side carries it matters only where a
        // side is infinite, and there it does not either: the computed sides are
        // finite (prepare), so an infinite side is a literal, and a literal
        // +/-inf plus any finite step is itself -- the residual comes out as the
        // DAG's whichever side moves. Elsewhere the two differ in rounding only.
        const double p = t.p + (t.slope * d);
        const double q = t.q;
        double residual = 0.0;
        if (t.is_abs) {
            residual = std::abs(p - q);
        } else {
            residual = comparison_residual(p, q, t.p_literal, t.q_literal);
            if (t.strict) {
                residual += kStrictComparisonEps;
            }
        }
        sum += t.weight * (clamped_node_violation(residual) - t.old_viol);
    }
    return sum;
}

bool LinearJumpScorer::residual_partial(int32_t ci, int32_t var_id, double& out) {
    Row& row = rows_[static_cast<size_t>(ci)];
    if (row.state == RowState::Pending) {
        build_row(ci);
    }
    if (row.state != RowState::Ready || !row.newton_exact) {
        return false;
    }
    const double r = slope_of(row, var_id);
    out =
        row.is_abs
            ? eq_sign(child_value(row.p_id, row.p_is_var) - child_value(row.q_id, row.q_is_var)) * r
            : r;
    ++cached_partials_;
    return true;
}

}  // namespace cbls
