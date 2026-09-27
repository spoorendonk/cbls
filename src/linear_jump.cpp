#include "cbls/linear_jump.h"

#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
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

// r = s0 - s1 per variable, for an Eq row: a union merge of two id-sorted lists.
void merge_sides(const std::vector<std::pair<int32_t, double>>& s0,
                 const std::vector<std::pair<int32_t, double>>& s1,
                 std::vector<std::pair<int32_t, double>>& out) {
    size_t i = 0;
    size_t k = 0;
    while (i < s0.size() || k < s1.size()) {
        if (k == s1.size() || (i < s0.size() && s0[i].first < s1[k].first)) {
            out.push_back(s0[i++]);
        } else if (i == s0.size() || s1[k].first < s0[i].first) {
            out.emplace_back(s1[k].first, -s1[k].second);
            ++k;
        } else {
            out.emplace_back(s0[i].first, s0[i].second - s1[k].second);
            ++i;
            ++k;
        }
    }
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
    for (size_t ci = n; ci < slots_.size(); ++ci) {
        release_row(slots_[ci]);
    }
    slots_.resize(n, kIneligible);
}

// A built row's pool entries and record become dead; counted, not freed, until
// compact_pool runs.
void LinearJumpScorer::release_row(uint32_t slot) {
    if (slot >= kFirstBuilt) {
        dead_entries_ += built_[slot - kFirstBuilt].count;
        ++dead_rows_;
    }
}

void LinearJumpScorer::set_row_eligible(int32_t ci, bool eligible) {
    uint32_t& slot = slots_.at(static_cast<size_t>(ci));
    release_row(slot);  // the body changed: its slopes are stale
    slot = eligible ? kPending : kIneligible;
}

bool LinearJumpScorer::row_eligible(int32_t ci) const {
    return slots_.at(static_cast<size_t>(ci)) != kIneligible;
}

// Rewrite the pool and the records with the live rows only, in row order.
// O(rows + live entries), run only once the dead outnumber the live -- so an
// extension-heavy run (column generation touches rows every batch) holds at
// most about twice its live slopes, and the copy is amortised over the builds
// that made the garbage.
void LinearJumpScorer::compact_pool() {
    std::vector<BuiltRow> built;
    std::vector<int32_t> vars;
    std::vector<double> slopes;
    built.reserve(built_.size() - dead_rows_);
    vars.reserve(pool_vars_.size() - dead_entries_);
    slopes.reserve(pool_slopes_.size() - dead_entries_);
    for (uint32_t& slot : slots_) {
        if (slot < kFirstBuilt) {
            continue;
        }
        BuiltRow row = built_[slot - kFirstBuilt];
        const auto first = static_cast<std::ptrdiff_t>(row.begin);
        const auto last = first + static_cast<std::ptrdiff_t>(row.count);
        row.begin = static_cast<uint32_t>(vars.size());
        vars.insert(vars.end(), pool_vars_.begin() + first, pool_vars_.begin() + last);
        slopes.insert(slopes.end(), pool_slopes_.begin() + first, pool_slopes_.begin() + last);
        slot = kFirstBuilt + static_cast<uint32_t>(built.size());
        built.push_back(row);
    }
    built_ = std::move(built);
    pool_vars_ = std::move(vars);
    pool_slopes_ = std::move(slopes);
    dead_entries_ = 0;
    dead_rows_ = 0;
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
const LinearJumpScorer::BuiltRow* LinearJumpScorer::build_row(int32_t ci) {
    const int32_t nid = model_.constraint_ids()[static_cast<size_t>(ci)];
    const ExprNode& nd = model_.nodes()[static_cast<size_t>(nid)];
    const ConstSpan<ChildRef> children = model_.children(nd);
    const bool swap = nd.op == NodeOp::Geq || nd.op == NodeOp::Gt;
    const ChildRef p = children[swap ? 1 : 0];
    const ChildRef q = children[swap ? 0 : 1];
    BuiltRow row;
    row.p_id = p.id;
    row.q_id = q.id;
    const bool p_literal = is_literal(model_, p);
    const bool q_literal = is_literal(model_, q);
    const bool is_abs = nd.op == NodeOp::Eq;
    bool newton_exact = true;

    merged_.clear();
    if (is_abs) {
        side_partials(model_, p, side0_);
        side_partials(model_, q, side1_);
        merge_sides(side0_, side1_, merged_);
        newton_exact = p_literal || q_literal;
    } else {
        compute_partials_sparse(model_, nid, merged_);
        std::sort(merged_.begin(), merged_.end());
    }
    size_t n = 0;
    for (const auto& e : merged_) {
        if (!std::isfinite(e.second)) {
            slots_[static_cast<size_t>(ci)] = kIneligible;
            return nullptr;
        }
        n += e.second != 0.0 ? 1 : 0;
    }

    if (dead_entries_ > pool_vars_.size() - dead_entries_) {
        compact_pool();
    }
    if (pool_vars_.size() + n > std::numeric_limits<uint32_t>::max() ||
        built_.size() + kFirstBuilt > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("LinearJumpScorer: more than 2^32 - 1 cached slopes");
    }
    row.flags = static_cast<uint8_t>((p.is_var ? kPIsVar : 0U) | (q.is_var ? kQIsVar : 0U) |
                                     (p_literal ? kPLiteral : 0U) | (q_literal ? kQLiteral : 0U) |
                                     (is_abs ? kAbs : 0U) |
                                     (nd.op == NodeOp::Lt || nd.op == NodeOp::Gt ? kStrict : 0U) |
                                     (newton_exact ? kNewtonExact : 0U));
    row.begin = static_cast<uint32_t>(pool_vars_.size());
    row.count = static_cast<uint32_t>(n);
    for (const auto& e : merged_) {
        if (e.second != 0.0) {
            pool_vars_.push_back(e.first);
            pool_slopes_.push_back(e.second);
        }
    }
    slots_[static_cast<size_t>(ci)] = kFirstBuilt + static_cast<uint32_t>(built_.size());
    built_.push_back(row);
    return &built_.back();
}

const LinearJumpScorer::BuiltRow* LinearJumpScorer::ready_row(int32_t ci) {
    const uint32_t slot = slots_[static_cast<size_t>(ci)];
    if (slot >= kFirstBuilt) {
        return &built_[slot - kFirstBuilt];
    }
    return slot == kPending ? build_row(ci) : nullptr;
}

double LinearJumpScorer::slope_of(const BuiltRow& row, int32_t var_id) const {
    const auto first = pool_vars_.begin() + static_cast<std::ptrdiff_t>(row.begin);
    const auto last = first + static_cast<std::ptrdiff_t>(row.count);
    const auto it = std::lower_bound(first, last, var_id);
    if (it == last || *it != var_id) {
        return 0.0;
    }
    return pool_slopes_[static_cast<size_t>(it - pool_vars_.begin())];
}

double LinearJumpScorer::child_value(int32_t id, bool is_var) const {
    if (is_var) {
        return model_.variables()[static_cast<size_t>(id)].value;
    }
    return model_.node_values()[static_cast<size_t>(id)];
}

bool LinearJumpScorer::prepare(int32_t var_id, const std::vector<double>& weights) {
    const std::vector<int32_t>& cids = model_.constraint_ids();
    if (slots_.size() != cids.size()) {
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
        ok = weights[c] == 0.0 || slots_[c] != kIneligible;
    }
    const std::vector<double>& node_values = model_.node_values();
    for (size_t k = 0; ok && k < gv.size(); ++k) {
        const int32_t c = gv[k];
        const double w = weights[static_cast<size_t>(c)];
        if (w == 0.0) {
            continue;
        }
        const BuiltRow* built = ready_row(c);
        if (built == nullptr) {
            ok = false;  // demoted by its build: a non-finite slope
            break;
        }
        const BuiltRow& row = *built;
        const bool p_literal = (row.flags & kPLiteral) != 0;
        const bool q_literal = (row.flags & kQLiteral) != 0;
        const double p = child_value(row.p_id, (row.flags & kPIsVar) != 0);
        const double q = child_value(row.q_id, (row.flags & kQIsVar) != 0);
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
        if ((p_literal ? std::isnan(p) : !std::isfinite(p)) ||
            (q_literal ? std::isnan(q) : !std::isfinite(q))) {
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
                              p, q, r, p_literal, q_literal, (row.flags & kAbs) != 0,
                              (row.flags & kStrict) != 0});
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
    const BuiltRow* built = ready_row(ci);
    if (built == nullptr || (built->flags & kNewtonExact) == 0) {
        return false;
    }
    const BuiltRow& row = *built;
    const double r = slope_of(row, var_id);
    if ((row.flags & kAbs) != 0) {
        // Read live: the sign follows the assignment, only r is structure.
        const double diff = child_value(row.p_id, (row.flags & kPIsVar) != 0) -
                            child_value(row.q_id, (row.flags & kQIsVar) != 0);
        out = eq_sign(diff) * r;
    } else {
        out = r;
    }
    ++cached_partials_;
    return true;
}

}  // namespace cbls
