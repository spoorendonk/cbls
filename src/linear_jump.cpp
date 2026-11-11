#include "cbls/linear_jump.h"

#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace cbls {

namespace {

bool is_literal(const Model& model, const ChildRef& ref) {
    // Exactly dag.cpp's `child_is_const`: comparison_residual's sentinel flag is
    // about a literal Const node, not about a constant-valued subtree.
    return !ref.is_var && model.nodes()[static_cast<size_t>(ref.id)].op == NodeOp::Const;
}

// The residual's sides p - q of row node `nd` (id `nid`): p = child1 for
// Geq/Gt, so that the residual reads p - q for every comparison. A bare body is
// the residual `body - 0`: p is the row's own node, q the literal zero, which
// has no node (id -1) and is never read from the DAG.
std::pair<ChildRef, ChildRef> residual_sides(const Model& model, const ExprNode& nd, int32_t nid) {
    if (!is_comparison_op(nd.op)) {
        return {ChildRef{nid, false}, ChildRef{-1, false}};
    }
    const ConstSpan<ChildRef> children = model.children(nd);
    const bool swap = nd.op == NodeOp::Geq || nd.op == NodeOp::Gt;
    return {children[swap ? 1 : 0], children[swap ? 0 : 1]};
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

// The divisor magnitude below which the DAG's `div_value` (dag.cpp) reads a
// quotient as a division by zero.
constexpr double kDivZero = 1e-15;

// One node of a side's cone, for `affine_row_terms`: its value at every variable
// zero, whether it reads no variable, and whether it is affine.
struct ConeConstant {
    double value = 0.0;
    bool var_free = true;
    bool affine = true;
};

ConeConstant child_constant(const ChildRef& ref,
                            const std::unordered_map<int32_t, ConeConstant>& memo) {
    if (ref.is_var) {
        return ConeConstant{0.0, false, true};
    }
    return memo.at(ref.id);
}

// The constant of one node, its children already in `memo`. A variable-free node
// is its own value: `const_value` for a literal, else the node's current value,
// which no variable can move. Above a variable, only the ops `node_is_affine`
// (feasibility_jump.cpp) admits carry a constant; a nested comparison does not
// (its residual has sentinel cases), so it is refused here.
ConeConstant node_constant(const Model& model, int32_t id,
                           const std::unordered_map<int32_t, ConeConstant>& memo) {
    const ExprNode& nd = model.nodes()[static_cast<size_t>(id)];
    const ConstSpan<ChildRef> children = model.children(nd);
    bool var_free = true;
    bool children_affine = true;
    for (const ChildRef& c : children) {
        const ConeConstant cc = child_constant(c, memo);
        var_free = var_free && cc.var_free;
        children_affine = children_affine && cc.affine;
    }
    if (var_free) {
        // The current value, never `const_value`: the objective bound's node
        // reads `objective_bound()`, which `set_objective_bound` writes only
        // into the node values.
        return ConeConstant{model.node_values()[static_cast<size_t>(id)], true, true};
    }
    ConeConstant out{0.0, false, false};
    if (!children_affine) {
        return out;
    }
    // An if-chain, not a NodeOp switch: every op not named here is refused,
    // which a switch could say only through a `default:`.
    if (nd.op == NodeOp::Neg) {
        out.value = -child_constant(children[0], memo).value;
        out.affine = true;
    } else if (nd.op == NodeOp::Sum) {
        for (const ChildRef& c : children) {
            out.value += child_constant(c, memo).value;
        }
        out.affine = true;
    } else if (nd.op == NodeOp::Prod) {
        const ConeConstant a = child_constant(children[0], memo);
        const ConeConstant b = child_constant(children[1], memo);
        out.value = a.value * b.value;
        out.affine = a.var_free || b.var_free;
    } else if (nd.op == NodeOp::Div) {
        // A divisor the DAG treats as zero (`div_value`, |b| < 1e-15) makes
        // the quotient +/-inf or NaN, not affine: refused.
        const ConeConstant b = child_constant(children[1], memo);
        out.value = child_constant(children[0], memo).value / b.value;
        out.affine = b.var_free && std::abs(b.value) >= kDivZero;
    }
    return out;
}

// The constant of side `ref` at every variable zero; false if the side is not
// affine or its constant is not finite. An explicit post-order walk of the cone,
// memoised, so a shared subtree is visited once and a deep one cannot overflow
// the call stack.
bool side_constant(const Model& model, const ChildRef& ref,
                   std::unordered_map<int32_t, ConeConstant>& memo, double& out) {
    if (ref.is_var) {
        out = 0.0;
        return true;
    }
    std::vector<std::pair<int32_t, bool>> stack{{ref.id, false}};
    while (!stack.empty()) {
        const auto [id, expanded] = stack.back();
        stack.pop_back();
        if (memo.count(id) != 0) {
            continue;
        }
        if (expanded) {
            memo.emplace(id, node_constant(model, id, memo));
            continue;
        }
        stack.emplace_back(id, true);
        for (const ChildRef& c : model.children(model.nodes()[static_cast<size_t>(id)])) {
            if (!c.is_var && memo.count(c.id) == 0) {
                stack.emplace_back(c.id, false);
            }
        }
    }
    const ConeConstant& cc = memo.at(ref.id);
    out = cc.value;
    return cc.affine && std::isfinite(cc.value);
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

bool affine_row_terms(const Model& model, int32_t ci,
                      std::vector<std::pair<int32_t, double>>& terms, double& constant) {
    const int32_t nid = model.constraint_ids()[static_cast<size_t>(ci)];
    const ExprNode& nd = model.nodes()[static_cast<size_t>(nid)];
    // A domain row (#258) adds domain_margin() where its residual turns
    // positive, so `p - q` is not the DAG's residual there.
    if (nd.op == NodeOp::Neq || model.is_domain_node(nid)) {
        return false;
    }
    const bool bare = !is_comparison_op(nd.op);
    const auto [p, q] = residual_sides(model, nd, nid);
    std::unordered_map<int32_t, ConeConstant> memo;
    double p_constant = 0.0;
    double q_constant = 0.0;
    if (!side_constant(model, p, memo, p_constant) ||
        (!bare && !side_constant(model, q, memo, q_constant))) {
        return false;
    }
    // The slopes exactly as build_row takes them.
    terms.clear();
    if (nd.op == NodeOp::Eq) {
        std::vector<std::pair<int32_t, double>> s0;
        std::vector<std::pair<int32_t, double>> s1;
        side_partials(model, p, s0);
        side_partials(model, q, s1);
        merge_sides(s0, s1, terms);
    } else {
        compute_partials_sparse(model, nid, terms);
        std::sort(terms.begin(), terms.end());
    }
    terms.erase(std::remove_if(terms.begin(), terms.end(),
                               [](const std::pair<int32_t, double>& t) { return t.second == 0.0; }),
                terms.end());
    for (const auto& t : terms) {
        if (!std::isfinite(t.second)) {
            return false;
        }
    }
    constant = p_constant - q_constant;
    if (nd.op == NodeOp::Lt || nd.op == NodeOp::Gt) {
        constant += nd.const_value;
    }
    return std::isfinite(constant);
}

void LinearJumpScorer::FreeDeleter::operator()(double* p) const {
    std::free(p);  // paired with build_row's calloc
}

LinearJumpScorer::LinearJumpScorer(const Model& model) : model_(model) {}

void LinearJumpScorer::resize_rows(size_t n) {
    if (n < slots_.size()) {
        throw std::logic_error("LinearJumpScorer::resize_rows: cannot shrink from " +
                               std::to_string(slots_.size()) + " to " + std::to_string(n) +
                               " rows; a closed model's rows are never removed");
    }
    if (n > slots_.size()) {
        // The model gained a row, and with it a rebuilt G_v layout: every slope
        // written so far may sit at a stale position.
        reset_built_rows();
    }
    slots_.resize(n, kIneligible);
}

void LinearJumpScorer::reset_built_rows() {
    for (uint32_t& slot : slots_) {
        if (slot >= kFirstBuilt) {
            slot = kPending;  // built means it was eligible; a rebuild gives the same slopes
        }
    }
    built_.clear();
    slope_at_.reset();
    slope_len_ = 0;
    cached_slopes_ = 0;
}

void LinearJumpScorer::set_row_eligible(int32_t ci, bool eligible) {
    uint32_t& slot = slots_.at(static_cast<size_t>(ci));
    if (slot >= kFirstBuilt) {
        throw std::logic_error("LinearJumpScorer::set_row_eligible: row " + std::to_string(ci) +
                               " is already built");
    }
    slot = eligible ? kPending : kIneligible;
}

bool LinearJumpScorer::row_eligible(int32_t ci) const {
    return slots_.at(static_cast<size_t>(ci)) != kIneligible;
}

// Slopes of one eligible row, built on first use.
//
// Leq/Lt/Geq/Gt: the residual is p - q (with p = child1 for Geq/Gt), so r is the
// node's own partial -- one sweep of the node, the same `reverse_sweep` that
// `compute_partial(node, v)` runs, hence bit-identical to it (`newton_exact`).
//
// Eq: the node's partial sign(p - q) * r is sign-dependent, so not cached;
// r = s0 - s1 from sweeping each child. It equals the Eq sweep up to sign
// exactly only when one side is a literal Const (no adjoint mixes both sides);
// otherwise Newton keeps calling compute_partial, while the jump score uses r.
//
// Bare body (#190): residual `body - 0`, built like Leq's with q the literal 0.
//
// Every slope must be finite, else the row is ineligible (Prod by an infinite
// constant). Div by a constant below 1e-15 (slope 0, value never finite) is
// caught in `prepare` by its non-finite side.
const LinearJumpScorer::BuiltRow* LinearJumpScorer::build_row(int32_t ci) {
    const int32_t nid = model_.constraint_ids()[static_cast<size_t>(ci)];
    const ExprNode& nd = model_.nodes()[static_cast<size_t>(nid)];
    const bool bare = !is_comparison_op(nd.op);
    const auto [p, q] = residual_sides(model_, nd, nid);
    // A domain row (#258) steps by domain_margin() where its residual turns
    // positive, so `p + r D` must match the DAG's sign exactly: only a bare
    // variable argument (q, after the Geq swap) allows that. Others probe.
    if (model_.is_domain_node(nid) && !q.is_var) {
        slots_[static_cast<size_t>(ci)] = kIneligible;
        return nullptr;
    }
    BuiltRow row;
    row.p_id = p.id;
    row.q_id = q.id;
    const bool p_literal = is_literal(model_, p);
    const bool q_literal = bare || is_literal(model_, q);
    const bool is_abs = nd.op == NodeOp::Eq;
    bool newton_exact = true;

    merged_.clear();
    if (is_abs) {
        side_partials(model_, p, side0_);
        side_partials(model_, q, side1_);
        merge_sides(side0_, side1_, merged_);
        newton_exact = p_literal || q_literal;
    } else {
        // Unsorted: write_slopes finds each variable's slot through its G_v.
        compute_partials_sparse(model_, nid, merged_);
    }
    size_t n = 0;
    for (const auto& e : merged_) {
        if (!std::isfinite(e.second)) {
            slots_[static_cast<size_t>(ci)] = kIneligible;
            return nullptr;
        }
        n += e.second != 0.0 ? 1 : 0;
    }

    if (built_.size() + kFirstBuilt > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("LinearJumpScorer: more than 2^32 - 2 built rows");
    }
    row.flags = static_cast<uint8_t>((p.is_var ? kPIsVar : 0U) | (q.is_var ? kQIsVar : 0U) |
                                     (p_literal ? kPLiteral : 0U) | (q_literal ? kQLiteral : 0U) |
                                     (is_abs ? kAbs : 0U) |
                                     (nd.op == NodeOp::Lt || nd.op == NodeOp::Gt ? kStrict : 0U) |
                                     (newton_exact ? kNewtonExact : 0U) | (bare ? kQZero : 0U));
    write_slopes(ci);
    // Append before publishing the slot: a throwing push_back must not leave the
    // slot naming a record that does not exist.
    built_.push_back(row);
    cached_slopes_ += n;
    slots_[static_cast<size_t>(ci)] = kFirstBuilt + static_cast<uint32_t>(built_.size() - 1);
    return &built_.back();
}

double* LinearJumpScorer::slope_table() {
    if (slope_at_) {
        return slope_at_.get();
    }
    // calloc, not a zeroed vector: a large block comes back as lazily mapped
    // zero pages, so unbuilt rows cost address space only (a small one may be
    // memset; see the header). At least one element, so the block is non-null.
    const size_t len = std::max<size_t>(model_.num_var_constraint_incidences(), 1);
    slope_at_.reset(static_cast<double*>(std::calloc(len, sizeof(double))));
    if (!slope_at_) {
        throw std::bad_alloc();
    }
    slope_len_ = len;
    return slope_at_.get();
}

void LinearJumpScorer::write_slopes(int32_t ci) {
    double* const slopes = slope_table();
    // Each nonzero slope into its variable's slot. A zero one stays the block's
    // +0.0, keeping -0.0 out of residual_partial_at's result.
    for (const auto& e : merged_) {
        if (e.second == 0.0) {
            continue;
        }
        const ConstSpan<int32_t> gv = model_.constraints_of_var(e.first);
        const auto* it = std::lower_bound(gv.begin(), gv.end(), ci);
        if (it == gv.end() || *it != ci) {
            throw std::logic_error("LinearJumpScorer: row " + std::to_string(ci) +
                                   " has a slope for variable " + std::to_string(e.first) +
                                   " but is not in its G_v");
        }
        slopes[model_.constraints_of_var_offset(e.first) + static_cast<size_t>(it - gv.begin())] =
            e.second;
    }
}

const LinearJumpScorer::BuiltRow* LinearJumpScorer::ready_row(int32_t ci) {
    const uint32_t slot = slots_[static_cast<size_t>(ci)];
    if (slot >= kFirstBuilt) {
        return &built_[slot - kFirstBuilt];
    }
    return slot == kPending ? build_row(ci) : nullptr;
}

double LinearJumpScorer::q_value(const BuiltRow& row) const {
    // The literal zero of a bare body has no node to read.
    return (row.flags & kQZero) != 0 ? 0.0 : child_value(row.q_id, (row.flags & kQIsVar) != 0);
}

double LinearJumpScorer::child_value(int32_t id, bool is_var) const {
    if (is_var) {
        return model_.variables()[static_cast<size_t>(id)].value;
    }
    return model_.node_values()[static_cast<size_t>(id)];
}

bool LinearJumpScorer::prepare(int32_t var_id, const std::vector<double>& weights) {
    return prepare_impl(var_id, weights, nullptr);
}

bool LinearJumpScorer::prepare(int32_t var_id, const std::vector<double>& weights,
                               const std::vector<double>& second) {
    return prepare_impl(var_id, weights, &second);
}

// w * (new - old) with both clamped, hence finite, is exactly 0 at w == 0, so
// such a row cannot change the sum whatever its shape.
//
// Refused rows first, before anything is built: on a mixed model a variable
// that falls back anyway should not pay slope sweeps for its linear rows.
//
// A dual prepare compares the two zero patterns in this same pass, and stops
// at the first refusal like a single one: a row that refuses with a nonzero
// weight under both vectors refuses `prepare(var_id, second)` too (as does a
// non-finite x0), so the patterns of the rows after it change nothing.
bool LinearJumpScorer::refusal_scan(int32_t var_id, const std::vector<double>& weights,
                                    const std::vector<double>* second) {
    const ConstSpan<int32_t> gv = model_.constraints_of_var(var_id);
    bool ok = std::isfinite(x0_);
    second_ready_ = second != nullptr;
    for (size_t k = 0; ok && k < gv.size(); ++k) {
        const auto c = static_cast<size_t>(gv[k]);
        const bool zero = weights[c] == 0.0;
        if (second_ready_) {
            second_ready_ = zero == ((*second)[c] == 0.0);
        }
        ok = ok && (zero || slots_[c] != kIneligible);
    }
    return ok;
}

bool LinearJumpScorer::prepare_impl(int32_t var_id, const std::vector<double>& weights,
                                    const std::vector<double>* second) {
    const std::vector<int32_t>& cids = model_.constraint_ids();
    if (slots_.size() != cids.size()) {
        throw std::logic_error(
            "LinearJumpScorer::prepare: the model has a different row count than this scorer; "
            "a row added to the model must be followed by resize_rows and set_row_eligible on it");
    }
    terms_.clear();
    second_weights_.clear();
    second_terms_ = false;
    x0_ = model_.var(var_id).value;
    const ConstSpan<int32_t> gv = model_.constraints_of_var(var_id);
    const size_t base = model_.constraints_of_var_offset(var_id);
    bool ok = refusal_scan(var_id, weights, second);
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
        const double q = q_value(row);
        // A computed side must be finite for `p + r D` to be the DAG's value; a
        // literal side may be the +/-inf bound sentinel (the objective row's
        // bound opens at +inf) but not NaN.
        //
        // Must precede the zero-slope skip: `Div` by a constant below 1e-15 has
        // slope 0 but evaluates to +/-inf or NaN (`div_value`, dag.cpp), never
        // finite, which this catches. At exactly 1e-15 both `evaluate` and
        // `local_derivative` divide (#205), giving an ordinary slope.
        if ((p_literal ? std::isnan(p) : !std::isfinite(p)) ||
            (q_literal ? std::isnan(q) : !std::isfinite(q))) {
            ok = false;
            break;
        }
        const double r = slope_at_.get()[base + k];  // row built above, so the array exists
        if (r == 0.0) {
            // Cancelled, or through a zero constant factor, with finite sides: the
            // DAG re-evaluates this row to the value it has, so its difference is
            // exactly 0.
            continue;
        }
        terms_.push_back(Term{w, clamped_node_violation(node_values[static_cast<size_t>(cids[c])]),
                              p, q, r, p_literal, q_literal, (row.flags & kAbs) != 0,
                              (row.flags & kStrict) != 0
                                  ? model_.nodes()[static_cast<size_t>(cids[c])].const_value
                                  : 0.0,
                              model_.is_domain_node(cids[c]) ? model_.domain_margin() : 0.0});
        if (second != nullptr) {
            second_weights_.push_back((*second)[static_cast<size_t>(c)]);
        }
    }
    if (!ok) {
        terms_.clear();
        second_weights_.clear();
        ++fallback_prepares_;
        return false;
    }
    second_terms_ = second != nullptr;
    ++fast_prepares_;
    return true;
}

double LinearJumpScorer::delta(double j) const {
    return delta_impl<false>(j);
}

double LinearJumpScorer::delta_second(double j) {
    if (!second_ready_ || !second_terms_) {
        throw std::logic_error(
            "LinearJumpScorer::delta_second: no successful dual prepare, or its second weights' "
            "zero pattern differs; prepare under the second weights instead");
    }
    ++second_deltas_;
    return delta_impl<true>(j);
}

template <bool kSecond>
double LinearJumpScorer::delta_impl(double j) const {
    const double d = j - x0_;
    double sum = 0.0;
    for (size_t i = 0; i < terms_.size(); ++i) {
        const Term& t = terms_[i];
        // The whole move lands on p. Computed sides are finite (prepare), so an
        // infinite side is a literal and absorbs any step: the residual matches
        // the DAG's either way, up to rounding.
        const double p = t.p + (t.slope * d);
        const double q = t.q;
        double residual = 0.0;
        if (t.is_abs) {
            residual = std::abs(p - q);
        } else {
            // + 0.0 on a non-strict row, which leaves every residual unchanged.
            residual = comparison_residual(p, q, t.p_literal, t.q_literal) + t.strict_margin;
            if (t.domain_step != 0.0) {
                // A domain row's argument is the variable itself (build_row), so
                // the DAG's residual is floor - j in one rounding; p - q above
                // takes two and may disagree in sign near the floor.
                residual = comparison_residual(t.p, j, t.p_literal, false);
                if (residual > 0.0) {
                    residual += t.domain_step;
                }
            }
        }
        // The same expression either way, so a dual prepare's second sum is
        // the double a single prepare under those weights would return.
        const double w = kSecond ? second_weights_[i] : t.weight;
        sum += w * (clamped_node_violation(residual) - t.old_viol);
    }
    return sum;
}

double LinearJumpScorer::row_partial(const BuiltRow& row, double r) const {
    if ((row.flags & kAbs) != 0) {
        // Read live: the sign follows the assignment, only r is structure.
        const double diff = child_value(row.p_id, (row.flags & kPIsVar) != 0) - q_value(row);
        return eq_sign(diff) * r;
    }
    return r;
}

bool LinearJumpScorer::residual_partial_at(int32_t var_id, size_t k, double& out) {
    // A row added to the model and not followed by resize_rows has moved G_v's
    // layout under the slope array, so nothing is read until it is.
    if (slots_.size() != model_.constraint_ids().size()) {
        return false;
    }
    const ConstSpan<int32_t> gv = model_.constraints_of_var(var_id);
    if (k >= gv.size()) {
        return false;
    }
    const BuiltRow* built = ready_row(gv[k]);
    if (built == nullptr || (built->flags & kNewtonExact) == 0) {
        return false;
    }
    out = row_partial(*built, slope_at_.get()[model_.constraints_of_var_offset(var_id) + k]);
    ++cached_partials_;
    return true;
}

bool LinearJumpScorer::row_max_variation(int32_t ci, const std::vector<int32_t>& vars,
                                         double& out) {
    if (slots_.size() != model_.constraint_ids().size()) {
        return false;  // the layout moved under the slope array; see residual_partial_at
    }
    if (ready_row(ci) == nullptr) {
        return false;
    }
    const double* const slopes = slope_at_.get();  // the row is built, so the array exists
    double m = 0.0;
    for (const int32_t v : vars) {
        const ConstSpan<int32_t> gv = model_.constraints_of_var(v);
        const auto* it = std::lower_bound(gv.begin(), gv.end(), ci);
        if (it == gv.end() || *it != ci) {
            throw std::logic_error("LinearJumpScorer::row_max_variation: variable " +
                                   std::to_string(v) + " is not in row " + std::to_string(ci));
        }
        const double slope =
            slopes[model_.constraints_of_var_offset(v) + static_cast<size_t>(it - gv.begin())];
        if (slope == 0.0) {
            continue;  // cancelled, or through a zero factor: v cannot move the row
        }
        const Variable& var = model_.var(v);
        const double reach = std::abs(slope) * (var.ub - var.lb);
        // NaN -- a NaN bound, or an inf - inf box -- bounds nothing.
        m = std::isnan(reach) ? std::numeric_limits<double>::infinity() : std::max(m, reach);
    }
    out = m;
    return true;
}

}  // namespace cbls
