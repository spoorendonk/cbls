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
// through Div by a zero constant, or Prod by an infinite one, has no linear model.
void LinearJumpScorer::build_row(int32_t ci) {
    Row& row = rows_[static_cast<size_t>(ci)];
    const int32_t nid = model_.constraint_ids()[static_cast<size_t>(ci)];
    const ExprNode& nd = model_.nodes()[static_cast<size_t>(nid)];
    const ConstSpan<ChildRef> children = model_.children(nd);
    const bool swap = nd.op == NodeOp::Geq || nd.op == NodeOp::Gt;
    row.p = children[swap ? 1 : 0];
    row.q = children[swap ? 0 : 1];
    row.p_literal = is_literal(model_, row.p);
    row.q_literal = is_literal(model_, row.q);
    row.is_abs = nd.op == NodeOp::Eq;
    row.strict = nd.op == NodeOp::Lt || nd.op == NodeOp::Gt;

    std::vector<std::pair<int32_t, double>> merged;
    if (row.is_abs) {
        std::vector<std::pair<int32_t, double>> s0;
        std::vector<std::pair<int32_t, double>> s1;
        side_partials(model_, row.p, s0);
        side_partials(model_, row.q, s1);
        size_t i = 0;
        size_t k = 0;
        while (i < s0.size() || k < s1.size()) {
            if (k == s1.size() || (i < s0.size() && s0[i].first < s1[k].first)) {
                merged.push_back(s0[i++]);
            } else if (i == s0.size() || s1[k].first < s0[i].first) {
                merged.emplace_back(s1[k].first, -s1[k].second);
                ++k;
            } else {
                merged.emplace_back(s0[i].first, s0[i].second - s1[k].second);
                ++i;
                ++k;
            }
        }
        row.newton_exact = row.p_literal || row.q_literal;
    } else {
        compute_partials_sparse(model_, nid, merged);
        std::sort(merged.begin(), merged.end());
        row.newton_exact = true;
    }

    row.vars.reserve(merged.size());
    row.slopes.reserve(merged.size());
    for (const auto& e : merged) {
        if (!std::isfinite(e.second)) {
            set_row_eligible(ci, false);
            return;
        }
        if (e.second != 0.0) {
            row.vars.push_back(e.first);
            row.slopes.push_back(e.second);
        }
    }
    row.state = RowState::Ready;
}

double LinearJumpScorer::slope_of(const Row& row, int32_t var_id) {
    const auto it = std::lower_bound(row.vars.begin(), row.vars.end(), var_id);
    if (it == row.vars.end() || *it != var_id) {
        return 0.0;
    }
    return row.slopes[static_cast<size_t>(it - row.vars.begin())];
}

double LinearJumpScorer::child_value(const ChildRef& ref) const {
    if (ref.is_var) {
        return model_.variables()[static_cast<size_t>(ref.id)].value;
    }
    return model_.node_values()[static_cast<size_t>(ref.id)];
}

bool LinearJumpScorer::prepare(int32_t var_id, const std::vector<double>& weights) {
    terms_.clear();
    x0_ = model_.var(var_id).value;
    bool ok = std::isfinite(x0_);
    const std::vector<int32_t>& cids = model_.constraint_ids();
    const std::vector<double>& node_values = model_.node_values();
    const ConstSpan<int32_t> gv = model_.constraints_of_var(var_id);
    for (size_t k = 0; ok && k < gv.size(); ++k) {
        const int32_t c = gv[k];
        const double w = weights[static_cast<size_t>(c)];
        // w * (new - old) with both clamped, hence finite: exactly 0 at w == 0,
        // so the row cannot change the sum whatever its shape.
        if (w == 0.0) {
            continue;
        }
        Row& row = rows_[static_cast<size_t>(c)];
        if (row.state == RowState::Pending) {
            build_row(c);
        }
        if (row.state != RowState::Ready) {
            ok = false;
            break;
        }
        const double r = slope_of(row, var_id);
        if (r == 0.0) {
            // Cancelled, or through a zero constant factor: the DAG re-evaluates
            // this row to the value it has, so its difference is exactly 0.
            continue;
        }
        const double p = child_value(row.p);
        const double q = child_value(row.q);
        // A computed side must be finite for `p + r D` to be the DAG's value; a
        // literal side may be the +/-inf bound sentinel (the objective row's
        // bound opens at +inf) but not NaN.
        if ((row.p_literal ? std::isnan(p) : !std::isfinite(p)) ||
            (row.q_literal ? std::isnan(q) : !std::isfinite(q))) {
            ok = false;
            break;
        }
        // D goes on a computed side. When both are computed they are finite, and
        // comparison_residual is then the plain difference, so the side is
        // immaterial up to rounding.
        const bool move_p = !row.p_literal;
        terms_.push_back(Term{w, clamped_node_violation(node_values[static_cast<size_t>(cids[c])]),
                              p, q, r, move_p, row.p_literal, row.q_literal, row.is_abs,
                              row.strict});
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
    double delta = 0.0;
    for (const Term& t : terms_) {
        double p = t.p;
        double q = t.q;
        if (t.move_p) {
            p += t.slope * d;
        } else {
            q -= t.slope * d;
        }
        double residual = 0.0;
        if (t.is_abs) {
            residual = std::abs(p - q);
        } else {
            residual = comparison_residual(p, q, t.p_literal, t.q_literal);
            if (t.strict) {
                residual += kStrictComparisonEps;
            }
        }
        delta += t.weight * (clamped_node_violation(residual) - t.old_viol);
    }
    return delta;
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
    out = row.is_abs ? eq_sign(child_value(row.p) - child_value(row.q)) * r : r;
    return true;
}

}  // namespace cbls
