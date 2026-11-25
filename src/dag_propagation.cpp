// HC4-style interval propagation over a model's nonlinear monotone rows (#424).
// See include/cbls/dag_propagation.h; the scan is `DomainPropagator`'s.

#include "cbls/dag_propagation.h"

#include "activity_bounds.h"
#include "cbls/domain_propagation.h"
#include "cbls/linear_jump.h"
#include "cbls/model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cbls {

using activity_detail::Activity;
using activity_detail::eff_hi;
using activity_detail::eff_lo;
using activity_detail::relaxation;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr double kPosInf = std::numeric_limits<double>::infinity();
constexpr double kEps = std::numeric_limits<double>::epsilon();

// The ops a row may use; any other op skips the whole row.
bool supported_op(NodeOp op) {
    switch (op) {
        case NodeOp::Const:
        case NodeOp::Neg:
        case NodeOp::Sum:
        case NodeOp::Prod:
        case NodeOp::Min:
        case NodeOp::Max:
        case NodeOp::Exp:
        case NodeOp::Log:
            return true;
        case NodeOp::Div:
        case NodeOp::Pow:
        case NodeOp::Abs:
        case NodeOp::Sin:
        case NodeOp::Cos:
        case NodeOp::Tan:
        case NodeOp::Sqrt:
        case NodeOp::SignPower:
        case NodeOp::Tanh:
        case NodeOp::If:
        case NodeOp::At:
        case NodeOp::Count:
        case NodeOp::Lambda:
        case NodeOp::PairLambda:
        case NodeOp::Leq:
        case NodeOp::Eq:
        case NodeOp::Geq:
        case NodeOp::Neq:
        case NodeOp::Lt:
        case NodeOp::Gt:
        case NodeOp::Element:
        case NodeOp::Ceil:
        case NodeOp::Floor:
        case NodeOp::Round:
        case NodeOp::LambdaExtra:
        case NodeOp::PairLambdaExtra:
            return false;
    }
    return false;
}

// One row's subtree, gathered from its residual sides: the non-Const nodes and
// the distinct variables, each once. False on an unsupported op, a structured
// variable or the objective bound's Const.
class RowCollector {
public:
    explicit RowCollector(const Model& model) : model_(model) {}

    bool collect(const ChildRef& p, const ChildRef& q) {
        nodes_.clear();
        vars_.clear();
        node_seen_.clear();
        var_slot_.clear();
        std::vector<ChildRef> stack{p, q};
        while (!stack.empty()) {
            const ChildRef ref = stack.back();
            stack.pop_back();
            if (ref.is_var) {
                if (is_structured(model_.var(ref.id).type)) {
                    return false;
                }
                if (var_slot_.emplace(ref.id, static_cast<int32_t>(vars_.size())).second) {
                    vars_.push_back(ref.id);
                }
                continue;
            }
            if (ref.id < 0) {
                continue;  // a bare body's literal zero
            }
            const ExprNode& nd = model_.nodes()[static_cast<size_t>(ref.id)];
            if (!supported_op(nd.op) || ref.id == model_.objective_bound_node()) {
                return false;
            }
            if (nd.op == NodeOp::Const || !node_seen_.emplace(ref.id, 0).second) {
                continue;
            }
            nodes_.push_back(ref.id);
            for (const ChildRef& c : model_.children(nd)) {
                stack.push_back(c);
            }
        }
        // Children before parents, as the model evaluates.
        std::sort(nodes_.begin(), nodes_.end(), [this](int32_t a, int32_t b) {
            return model_.topo_position(a) < model_.topo_position(b);
        });
        for (size_t k = 0; k < nodes_.size(); ++k) {
            node_seen_[nodes_[k]] = static_cast<int32_t>(k);
        }
        return true;
    }

    // The local reference of a model child: a slot, a row node or a literal.
    [[nodiscard]] DagRowStore::Ref ref(const ChildRef& c) const {
        if (c.is_var) {
            return {DagRowStore::RefKind::kVar, var_slot_.at(c.id), 0.0};
        }
        if (c.id < 0) {
            return {DagRowStore::RefKind::kConst, 0, 0.0};
        }
        const ExprNode& nd = model_.nodes()[static_cast<size_t>(c.id)];
        if (nd.op == NodeOp::Const) {
            return {DagRowStore::RefKind::kConst, 0, nd.const_value};
        }
        return {DagRowStore::RefKind::kNode, node_seen_.at(c.id), 0.0};
    }

    // Appends the collected subtree to the store's flat arrays; returns the row
    // with its sides and ranges set.
    DagRowStore::Row flatten(const ChildRef& p, const ChildRef& q,
                             std::vector<DagRowStore::Node>& nodes,
                             std::vector<DagRowStore::Ref>& children,
                             std::vector<int32_t>& row_vars) const {
        DagRowStore::Row row;
        row.p = ref(p);
        row.q = ref(q);
        row.node_begin = static_cast<uint32_t>(nodes.size());
        for (const int32_t id : nodes_) {
            const ExprNode& n = model_.nodes()[static_cast<size_t>(id)];
            DagRowStore::Node node{n.op, static_cast<uint32_t>(children.size()), 0};
            for (const ChildRef& ch : model_.children(n)) {
                children.push_back(ref(ch));
                ++node.child_count;
            }
            nodes.push_back(node);
        }
        row.node_end = static_cast<uint32_t>(nodes.size());
        row.var_begin = static_cast<uint32_t>(row_vars.size());
        row_vars.insert(row_vars.end(), vars_.begin(), vars_.end());
        row.var_end = static_cast<uint32_t>(row_vars.size());
        return row;
    }

    [[nodiscard]] size_t num_nodes() const { return nodes_.size(); }
    [[nodiscard]] size_t num_vars() const { return vars_.size(); }

private:
    const Model& model_;
    std::vector<int32_t> nodes_;
    std::vector<int32_t> vars_;
    std::unordered_map<int32_t, int32_t> node_seen_;  // model node -> local index
    std::unordered_map<int32_t, int32_t> var_slot_;   // model var -> slot
};

// Constraint `c`'s residual sides `p - q` if the store may keep it, as
// residual_sides (src/linear_jump.cpp) reads them: Geq/Gt swapped, a bare body
// against a literal zero. False for the rows the class comment skips before
// their subtree is read.
bool candidate_sides(const Model& model, int32_t c, std::vector<std::pair<int32_t, double>>& terms,
                     ChildRef& p, ChildRef& q) {
    if (c == model.objective_constraint_idx()) {
        return false;  // soft: propagating it would make the cutoff hard
    }
    const int32_t nid = model.constraint_ids()[static_cast<size_t>(c)];
    const ExprNode& nd = model.nodes()[static_cast<size_t>(nid)];
    double constant = 0.0;
    if (nd.op == NodeOp::Neq || model.is_domain_node(nid) ||
        affine_row_terms(model, c, terms, constant)) {
        return false;
    }
    p = ChildRef{nid, false};
    q = ChildRef{-1, false};
    if (is_comparison_op(nd.op)) {
        const ConstSpan<ChildRef> ch = model.children(nd);
        const bool swap = nd.op == NodeOp::Geq || nd.op == NodeOp::Gt;
        p = ch[swap ? 1 : 0];
        q = ch[swap ? 0 : 1];
    }
    return true;
}

}  // namespace

DagRowStore::DagRowStore(const Model& model, double row_tolerance) {
    if (!model.is_closed()) {
        throw std::logic_error("DagRowStore: the model must be closed and evaluated");
    }
    if (!std::isfinite(row_tolerance) || row_tolerance < 0.0) {
        throw std::invalid_argument("DagRowStore: row_tolerance must be finite and >= 0");
    }
    const auto n_cons = static_cast<int32_t>(model.constraint_ids().size());
    std::vector<std::pair<int32_t, double>> terms;
    RowCollector rc(model);
    for (int32_t c = 0; c < n_cons; ++c) {
        ChildRef p{};
        ChildRef q{};
        if (!candidate_sides(model, c, terms, p, q) || !rc.collect(p, q)) {
            continue;
        }
        Row row = rc.flatten(p, q, nodes_, children_, row_vars_);
        const ExprNode& nd =
            model.nodes()[static_cast<size_t>(model.constraint_ids()[static_cast<size_t>(c)])];
        row.constraint = c;
        const bool strict = nd.op == NodeOp::Lt || nd.op == NodeOp::Gt;
        row.hi = row_tolerance - (strict ? nd.const_value : 0.0);
        row.lo = nd.op == NodeOp::Eq ? -row_tolerance : kNegInf;
        max_row_nodes_ = std::max(max_row_nodes_, rc.num_nodes());
        max_row_vars_ = std::max(max_row_vars_, rc.num_vars());
        rows_.push_back(row);
    }
    index_rows_by_variable(static_cast<int32_t>(model.num_vars()));
}

void DagRowStore::index_rows_by_variable(int32_t n_vars) {
    // Counting sort of the row slots by variable.
    var_row_start_.assign(static_cast<size_t>(n_vars) + 1, 0);
    for (const int32_t v : row_vars_) {
        ++var_row_start_[static_cast<size_t>(v) + 1];
    }
    for (size_t v = 1; v <= static_cast<size_t>(n_vars); ++v) {
        var_row_start_[v] += var_row_start_[v - 1];
    }
    var_rows_.assign(row_vars_.size(), 0);
    std::vector<size_t> next(var_row_start_.begin(), var_row_start_.end() - 1);
    for (int32_t k = 0; k < num_rows(); ++k) {
        for (uint32_t s = rows_[static_cast<size_t>(k)].var_begin;
             s < rows_[static_cast<size_t>(k)].var_end; ++s) {
            var_rows_[next[static_cast<size_t>(row_vars_[s])]++] = k;
        }
    }
}

// ---------------------------------------------------------------------------
// The scan: `DomainPropagator::scan_dag_row`

namespace {

// Interval endpoints, widened outward by the safety margin plus `extra`. A NaN
// endpoint (inf - inf, inf / inf) bounds nothing on its side.
double down(double v, const BoundPropagationOptions& b, double extra = 0.0) {
    if (std::isnan(v)) {
        return kNegInf;
    }
    return std::isfinite(v) ? v - relaxation(v, b) - extra : v;
}
double up(double v, const BoundPropagationOptions& b, double extra = 0.0) {
    if (std::isnan(v)) {
        return kPosInf;
    }
    return std::isfinite(v) ? v + relaxation(v, b) + extra : v;
}

// A product endpoint with 0 * inf = 0: an infinite bound is unboundedness, a
// variable's value is finite, and a NaN value (an overflow times zero) violates.
double mul(double a, double b) {
    return (a == 0.0 || b == 0.0) ? 0.0 : a * b;
}

DagInterval product(const DagInterval& a, const DagInterval& b, const BoundPropagationOptions& o) {
    const std::array<double, 4> c = {mul(a.lo, b.lo), mul(a.lo, b.hi), mul(a.hi, b.lo),
                                     mul(a.hi, b.hi)};
    return {down(*std::min_element(c.begin(), c.end()), o),
            up(*std::max_element(c.begin(), c.end()), o)};
}

// z / b for a `b` that excludes 0: monotone in each argument over the box, so
// the corners span it. An inf / inf corner bounds nothing.
DagInterval quotient(const DagInterval& z, const DagInterval& b, const BoundPropagationOptions& o) {
    const std::array<double, 4> c = {z.lo / b.lo, z.lo / b.hi, z.hi / b.lo, z.hi / b.hi};
    if (std::any_of(c.begin(), c.end(), [](double x) { return std::isnan(x); })) {
        return {kNegInf, kPosInf};
    }
    return {down(*std::min_element(c.begin(), c.end()), o),
            up(*std::max_element(c.begin(), c.end()), o)};
}

bool excludes_zero(const DagInterval& b) {
    return b.lo > 0.0 || b.hi < 0.0;
}

// A sum's roundoff, `roundoff_per_unit` (#211) at a unit coefficient: `n`
// terms of total magnitude `mag`, read against `bound`.
double sum_roundoff(size_t n, double mag, double bound) {
    const double m = mag + (std::isfinite(bound) ? std::abs(bound) : 0.0);
    return static_cast<double>(n + 3) * kEps * m;
}

// A node's value over its children's intervals. Ops are only those
// `supported_op` admits; Const is never a node. False for a Log whose argument
// is nowhere positive: the row then derives nothing (its domain row owns it).
bool forward(NodeOp op, const std::vector<DagInterval>& kids, const BoundPropagationOptions& o,
             DagInterval& out) {
    if (op == NodeOp::Neg) {
        out = {-kids[0].hi, -kids[0].lo};
    } else if (op == NodeOp::Sum) {
        Activity lo;
        Activity hi;
        for (const DagInterval& c : kids) {
            lo.add(c.lo);
            hi.add(c.hi);
        }
        const size_t n = kids.size();
        out = {lo.n_infinite > 0 ? kNegInf : down(lo.sum, o, sum_roundoff(n, lo.magnitude, 0.0)),
               hi.n_infinite > 0 ? kPosInf : up(hi.sum, o, sum_roundoff(n, hi.magnitude, 0.0))};
    } else if (op == NodeOp::Prod) {
        out = product(kids[0], kids[1], o);
    } else if (op == NodeOp::Min || op == NodeOp::Max) {
        const bool take_max = op == NodeOp::Max;
        out = kids[0];
        for (const DagInterval& c : kids) {
            out.lo = take_max ? std::max(out.lo, c.lo) : std::min(out.lo, c.lo);
            out.hi = take_max ? std::max(out.hi, c.hi) : std::min(out.hi, c.hi);
        }
    } else if (op == NodeOp::Exp) {
        out = {down(std::exp(kids[0].lo), o), up(std::exp(kids[0].hi), o)};
    } else {  // Log: log(0) = -inf, NaN (a violation) below 0
        if (!(kids[0].hi > 0.0)) {
            return false;
        }
        out = {kids[0].lo > 0.0 ? down(std::log(kids[0].lo), o) : kNegInf,
               up(std::log(kids[0].hi), o)};
    }
    return true;
}

// Each child's projection, into `proj`, given the node's value lies in `z`
// and the children's forward intervals `kids`.
void sum_backward(const DagInterval& z, const std::vector<DagInterval>& kids,
                  const BoundPropagationOptions& o, std::vector<DagInterval>& proj) {
    Activity lo;
    Activity hi;
    for (const DagInterval& c : kids) {
        lo.add(c.lo);
        hi.add(c.hi);
    }
    const size_t n = kids.size();
    for (size_t i = 0; i < n; ++i) {
        // x_i <= z.hi - min(rest), x_i >= z.lo - max(rest).
        const double min_rest = lo.without(kids[i].lo, kNegInf);
        const double max_rest = hi.without(kids[i].hi, kPosInf);
        proj[i] = {down(z.lo - max_rest, o, sum_roundoff(n, hi.magnitude, z.lo)),
                   up(z.hi - min_rest, o, sum_roundoff(n, lo.magnitude, z.hi))};
    }
}

// Every child is past the node's near bound; the far bound binds only the one
// child that can still be the min (max). `proj` arrives unbounded.
void min_max_backward(bool is_max, const DagInterval& z, const std::vector<DagInterval>& kids,
                      std::vector<DagInterval>& proj) {
    size_t candidates = 0;
    size_t which = 0;
    for (size_t i = 0; i < kids.size(); ++i) {
        (is_max ? proj[i].hi : proj[i].lo) = is_max ? z.hi : z.lo;
        if (is_max ? kids[i].hi >= z.lo : kids[i].lo <= z.hi) {
            ++candidates;
            which = i;
        }
    }
    if (candidates == 1) {
        (is_max ? proj[which].lo : proj[which].hi) = is_max ? z.lo : z.hi;
    }
}

void backward(NodeOp op, const DagInterval& z, const std::vector<DagInterval>& kids,
              const BoundPropagationOptions& o, std::vector<DagInterval>& proj) {
    proj.assign(kids.size(), DagInterval{kNegInf, kPosInf});
    if (op == NodeOp::Neg) {
        proj[0] = {-z.hi, -z.lo};
    } else if (op == NodeOp::Sum) {
        sum_backward(z, kids, o, proj);
    } else if (op == NodeOp::Prod) {
        // Only through a sign-known factor: z / b needs b to exclude 0.
        for (size_t i = 0; i < 2; ++i) {
            if (excludes_zero(kids[1 - i])) {
                proj[i] = quotient(z, kids[1 - i], o);
            }
        }
    } else if (op == NodeOp::Min || op == NodeOp::Max) {
        min_max_backward(op == NodeOp::Max, z, kids, proj);
    } else if (op == NodeOp::Exp) {
        // exp(x) > 0 for every finite x; a z.hi <= 0 leaves x only -inf.
        proj[0] = {z.lo > 0.0 ? down(std::log(z.lo), o) : kNegInf,
                   z.hi > 0.0 ? up(std::log(z.hi), o) : kNegInf};
    } else {  // Log
        proj[0] = {down(std::exp(z.lo), o), up(std::exp(z.hi), o)};
    }
}

void meet(DagInterval& into, const DagInterval& with) {
    into.lo = std::max(into.lo, with.lo);
    into.hi = std::min(into.hi, with.hi);
}

// Empty beyond the margin; a crossing within it is widened back to its ends.
bool settle(DagInterval& iv, const BoundPropagationOptions& o) {
    if (iv.lo > iv.hi) {
        if (iv.lo > iv.hi + relaxation(iv.lo, o)) {
            return false;
        }
        std::swap(iv.lo, iv.hi);
    }
    return true;
}

}  // namespace

// One HC4 revise of DAG row k: forward intervals in topological order, the
// row's bounds on `p - q`, then each node's interval (its forward one met with
// every parent's projection, so shared nodes stay sound) projected onto its
// children in reverse order; each variable's meet is applied once through
// `apply`. Intervals are recomputed from the bounds, so nothing is trailed.
// O(row nodes + children) plus the derived bounds' set_bounds.
bool DomainPropagator::scan_dag_row(int32_t k) {
    const DagRowStore& dag = *dag_;
    const DagRowStore::Row& row = dag.row(k);
    const BoundPropagationOptions& o = opts_.bounds;
    const uint32_t n_nodes = row.node_end - row.node_begin;
    const uint32_t n_vars = row.var_end - row.var_begin;
    for (uint32_t s = 0; s < n_vars; ++s) {
        const auto j = static_cast<size_t>(dag.row_vars()[row.var_begin + s]);
        dag_var_fwd_[s] = {eff_lo(lo_[j], o.infinity), eff_hi(hi_[j], o.infinity)};
        dag_var_bwd_[s] = dag_var_fwd_[s];
    }
    const auto value = [&](const DagRowStore::Ref& r) -> DagInterval {
        switch (r.kind) {
            case DagRowStore::RefKind::kVar:
                return dag_var_fwd_[static_cast<size_t>(r.index)];
            case DagRowStore::RefKind::kNode:
                return dag_node_fwd_[static_cast<size_t>(r.index)];
            case DagRowStore::RefKind::kConst:
                break;
        }
        return {r.value, r.value};
    };
    const auto target = [&](const DagRowStore::Ref& r) -> DagInterval* {
        switch (r.kind) {
            case DagRowStore::RefKind::kVar:
                return &dag_var_bwd_[static_cast<size_t>(r.index)];
            case DagRowStore::RefKind::kNode:
                return &dag_node_bwd_[static_cast<size_t>(r.index)];
            case DagRowStore::RefKind::kConst:
                break;
        }
        return nullptr;  // a literal narrows nothing
    };
    const auto gather = [&](const DagRowStore::Node& node) {
        dag_kids_.clear();
        for (uint32_t i = 0; i < node.child_count; ++i) {
            dag_kids_.push_back(value(dag.children()[node.child_begin + i]));
        }
    };
    ++stats_.dag_rows_scanned;
    int64_t work = n_nodes;
    for (uint32_t i = 0; i < n_nodes; ++i) {
        const DagRowStore::Node& node = dag.nodes()[row.node_begin + i];
        gather(node);
        work += node.child_count;
        if (!forward(node.op, dag_kids_, o, dag_node_fwd_[i])) {
            call_terms_ += work;
            return true;
        }
        dag_node_bwd_[i] = dag_node_fwd_[i];
    }
    call_terms_ += work;
    // The root: p - q in [row.lo, row.hi].
    const DagInterval p = value(row.p);
    const DagInterval q = value(row.q);
    if (DagInterval* t = target(row.p)) {
        meet(*t, {down(row.lo + q.lo, o, sum_roundoff(1, std::abs(q.lo), row.lo)),
                  up(row.hi + q.hi, o, sum_roundoff(1, std::abs(q.hi), row.hi))});
    }
    if (DagInterval* t = target(row.q)) {
        meet(*t, {down(p.lo - row.hi, o, sum_roundoff(1, std::abs(p.lo), row.hi)),
                  up(p.hi - row.lo, o, sum_roundoff(1, std::abs(p.hi), row.lo))});
    }
    for (uint32_t i = n_nodes; i-- > 0;) {
        DagInterval& z = dag_node_bwd_[i];
        if (!settle(z, o)) {
            return false;
        }
        const DagRowStore::Node& node = dag.nodes()[row.node_begin + i];
        gather(node);
        backward(node.op, z, dag_kids_, o, dag_proj_);
        for (uint32_t c = 0; c < node.child_count; ++c) {
            if (DagInterval* t = target(dag.children()[node.child_begin + c])) {
                meet(*t, dag_proj_[c]);
            }
        }
    }
    const int64_t derived_before = stats_.bounds_derived;
    bool ok = true;
    for (uint32_t s = 0; s < n_vars && ok; ++s) {
        const int32_t var = dag.row_vars()[row.var_begin + s];
        DagInterval& iv = dag_var_bwd_[s];
        ok = settle(iv, o) && apply(var, iv.hi, /*upper=*/true, 0.0) &&
             apply(var, iv.lo, /*upper=*/false, 0.0);
    }
    stats_.dag_bounds_derived += stats_.bounds_derived - derived_before;
    return ok;
}

}  // namespace cbls
