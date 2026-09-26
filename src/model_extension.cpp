#include "cbls/model_extension.h"

#include "cbls/dag_ops.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace cbls {

// Forward declare from dag_ops.cpp, as src/model.cpp does.
namespace detail {
std::vector<int32_t> compute_topo_order(const Model& model);
}

// ===========================================================================
// ModelExtension -- pure recording. Nothing here touches the model.
// ===========================================================================

ModelExtension::ModelExtension(const Model& base)
    : base_(&base),
      base_num_vars_(static_cast<int32_t>(base.num_vars())),
      base_num_nodes_(static_cast<int32_t>(base.num_nodes())) {
    if (!base.is_closed()) {
        throw std::logic_error(
            "ModelExtension: the base model is not closed. An open model grows through the "
            "ordinary Model builders; extend() exists for a model whose derived indices are "
            "already built");
    }
}

void ModelExtension::validate_handle(int32_t handle) const {
    if (handle < 0) {
        const int32_t vid = -(handle + 1);
        if (vid >= base_num_vars_ + static_cast<int32_t>(new_vars_.size())) {
            throw std::out_of_range("ModelExtension: variable handle out of range");
        }
    } else if (handle >= base_num_nodes_ + static_cast<int32_t>(new_nodes_.size())) {
        throw std::out_of_range("ModelExtension: node handle out of range");
    }
}

int32_t ModelExtension::check_handle(int32_t handle) const {
    validate_handle(handle);
    return handle;
}

int32_t ModelExtension::check_node_handle(int32_t handle, const char* what) const {
    if (handle < 0) {
        throw std::invalid_argument(std::string(what) +
                                    " requires a node handle (non-negative), got a var handle");
    }
    return check_handle(handle);
}

int32_t ModelExtension::add_var(VarType type, double lb, double ub, const std::string& name) {
    NewVar v;
    v.type = type;
    v.lb = lb;
    v.ub = ub;
    // The lower bound, matching Model::alloc_var, so that a staged model and the
    // same model built whole start from the same assignment. set_initial()
    // overrides it.
    v.initial = lb;
    v.name = name;
    new_vars_.push_back(std::move(v));
    const auto vid = base_num_vars_ + static_cast<int32_t>(new_vars_.size()) - 1;
    return -(vid + 1);
}

int32_t ModelExtension::bool_var(const std::string& name) {
    return add_var(VarType::Bool, 0.0, 1.0, name);
}

int32_t ModelExtension::int_var(int lb, int ub, const std::string& name) {
    return add_var(VarType::Int, static_cast<double>(lb), static_cast<double>(ub), name);
}

int32_t ModelExtension::float_var(double lb, double ub, const std::string& name) {
    return add_var(VarType::Float, lb, ub, name);
}

void ModelExtension::set_initial(int32_t var, double value) {
    const int32_t idx = (var < 0) ? (-(var + 1) - base_num_vars_) : -1;
    if (idx < 0 || idx >= static_cast<int32_t>(new_vars_.size())) {
        throw std::invalid_argument(
            "ModelExtension::set_initial: not a variable this extension created. An existing "
            "variable's value is search state, and rewriting it here would silently move the "
            "assignment the model is already sitting on");
    }
    NewVar& nv = new_vars_[static_cast<size_t>(idx)];
    // Spelled with an explicit NaN test rather than as `!(lb <= v && v <= ub)`:
    // the negated form rejects NaN by accident, and the positive comparisons alone
    // would let it through.
    if (std::isnan(value) || value < nv.lb || value > nv.ub) {
        throw std::invalid_argument(
            "ModelExtension::set_initial: value is outside the variable's bounds");
    }
    nv.initial = value;
}

int32_t ModelExtension::push(NodeOp op, std::vector<int32_t> children, double const_value) {
    for (const int32_t h : children) {
        validate_handle(h);
    }
    NewNode n;
    n.op = op;
    n.const_value = const_value;
    n.children = std::move(children);
    new_nodes_.push_back(std::move(n));
    return base_num_nodes_ + static_cast<int32_t>(new_nodes_.size()) - 1;
}

int32_t ModelExtension::constant(double val) {
    return push(NodeOp::Const, {}, val);
}
int32_t ModelExtension::neg(int32_t x) {
    return push(NodeOp::Neg, {x});
}
int32_t ModelExtension::sum(const std::vector<int32_t>& args) {
    // Model::sum's own rule: an empty sum is the constant 0, not a childless Sum
    // node (which dag.cpp's Sum arm would read as 0 anyway, but Min/Max would not).
    if (args.empty()) {
        return constant(0.0);
    }
    return push(NodeOp::Sum, args);
}
int32_t ModelExtension::prod(int32_t a, int32_t b) {
    return push(NodeOp::Prod, {a, b});
}
int32_t ModelExtension::div_expr(int32_t a, int32_t b) {
    return push(NodeOp::Div, {a, b});
}
int32_t ModelExtension::pow_expr(int32_t base, int32_t exp) {
    return push(NodeOp::Pow, {base, exp});
}
int32_t ModelExtension::min_expr(const std::vector<int32_t>& args) {
    if (args.empty()) {
        throw std::invalid_argument("min_expr requires at least one argument");
    }
    return push(NodeOp::Min, args);
}
int32_t ModelExtension::max_expr(const std::vector<int32_t>& args) {
    if (args.empty()) {
        throw std::invalid_argument("max_expr requires at least one argument");
    }
    return push(NodeOp::Max, args);
}
int32_t ModelExtension::abs_expr(int32_t x) {
    return push(NodeOp::Abs, {x});
}
int32_t ModelExtension::sin_expr(int32_t x) {
    return push(NodeOp::Sin, {x});
}
int32_t ModelExtension::cos_expr(int32_t x) {
    return push(NodeOp::Cos, {x});
}
int32_t ModelExtension::tan_expr(int32_t x) {
    return push(NodeOp::Tan, {x});
}
int32_t ModelExtension::exp_expr(int32_t x) {
    return push(NodeOp::Exp, {x});
}
int32_t ModelExtension::log_expr(int32_t x) {
    return push(NodeOp::Log, {x});
}
int32_t ModelExtension::sqrt_expr(int32_t x) {
    return push(NodeOp::Sqrt, {x});
}
int32_t ModelExtension::signpower_expr(int32_t base, int32_t exp) {
    return push(NodeOp::SignPower, {base, exp});
}
int32_t ModelExtension::tanh_expr(int32_t x) {
    return push(NodeOp::Tanh, {x});
}
int32_t ModelExtension::if_then_else(int32_t cond, int32_t then_, int32_t else_) {
    return push(NodeOp::If, {cond, then_, else_});
}
int32_t ModelExtension::at(int32_t list_var_id, int32_t index_expr) {
    return push(NodeOp::At, {list_var_id, index_expr});
}
int32_t ModelExtension::count(int32_t var_id) {
    return push(NodeOp::Count, {var_id});
}
int32_t ModelExtension::leq(int32_t a, int32_t b) {
    return push(NodeOp::Leq, {a, b});
}
int32_t ModelExtension::eq_expr(int32_t a, int32_t b) {
    return push(NodeOp::Eq, {a, b});
}
int32_t ModelExtension::geq(int32_t a, int32_t b) {
    return push(NodeOp::Geq, {a, b});
}
int32_t ModelExtension::neq(int32_t a, int32_t b) {
    return push(NodeOp::Neq, {a, b});
}
int32_t ModelExtension::lt(int32_t a, int32_t b) {
    return push(NodeOp::Lt, {a, b});
}
int32_t ModelExtension::gt(int32_t a, int32_t b) {
    return push(NodeOp::Gt, {a, b});
}

void ModelExtension::add_constraint(int32_t expr) {
    new_constraints_.push_back(check_node_handle(expr, "ModelExtension::add_constraint"));
}

void ModelExtension::append_to_sum(int32_t sum_node, int32_t term) {
    check_node_handle(sum_node, "ModelExtension::append_to_sum");
    if (sum_node >= base_num_nodes_) {
        throw std::invalid_argument(
            "ModelExtension::append_to_sum: the target must be a node of the base model. A node "
            "this extension created has its children written by sum(), which already takes the "
            "whole list");
    }
    if (base_->node(sum_node).op != NodeOp::Sum) {
        throw std::invalid_argument(
            "ModelExtension::append_to_sum: the target node is not a Sum. Only a Sum's arity can "
            "grow without changing what the node means");
    }
    appends_.emplace_back(sum_node, check_handle(term));
}

// ===========================================================================
// Incremental structure maintenance
// ===========================================================================

namespace {

using Addition = std::pair<int32_t, int32_t>;  // (owner, value)

// Canonicalise a set of CSR additions: sort by owner then value, drop exact
// duplicates, and drop anything the owner already has.
//
// The duplicate rules are the ones `rebuild_back_references` enforces, so that a
// spliced array is bit-identical to a rebuilt one: a parent that names the same
// child twice (`prod(x, x)`) is listed once, and a term appended to a Sum that
// already names it adds no second back-reference. The "already has" test is a
// binary search because the existing lists are ascending -- O(log degree) per
// addition, where a scan would be O(degree).
void canonicalise(const std::vector<uint32_t>& offsets, const std::vector<int32_t>& ids,
                  std::vector<Addition>& additions) {
    std::sort(additions.begin(), additions.end());
    additions.erase(std::unique(additions.begin(), additions.end()), additions.end());
    additions.erase(std::remove_if(additions.begin(), additions.end(),
                                   [&](const Addition& a) {
                                       const auto owner = static_cast<size_t>(a.first);
                                       const int32_t* begin = ids.data() + offsets[owner];
                                       const int32_t* end = ids.data() + offsets[owner + 1];
                                       return std::binary_search(begin, end, a.second);
                                   }),
                    additions.end());
}

// Merge canonicalised `additions` into a CSR (offsets, ids) pair, in place.
//
// Only the suffix from the first touched owner moves: every owner below it keeps
// its offsets and its ids exactly where they were. That is what makes an
// extension whose additions all land on new owners O(k) instead of O(model), and
// one that touches an early owner O(tail) -- the two regimes `Model::extend`
// states.
//
// Each owner's list stays ASCENDING and distinct, which is contractual for
// `parents`, `dependents` and `constraints_of_var` alike:
// `ViolationManager::weighted_delta_from` requires strictly ascending rows, and
// FJ's scan order over G_v feeds the trajectory. An overflow list appended after
// the CSR base would NOT have that property, which is why this merges rather than
// appending.
void splice_csr(std::vector<uint32_t>& offsets, std::vector<int32_t>& ids,
                const std::vector<Addition>& additions) {
    if (additions.empty()) {
        return;
    }
    const size_t old_total = ids.size();
    const size_t new_total = old_total + additions.size();
    if (new_total > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("model has more than 2^32 - 1 entries in a back-reference index");
    }
    ids.resize(new_total);
    const auto n_owners = static_cast<int32_t>(offsets.size()) - 1;
    const int32_t first_owner = additions.front().first;
    size_t write = new_total;
    size_t a_hi = additions.size();
    auto old_end = static_cast<uint32_t>(old_total);
    for (int32_t owner = n_owners - 1; owner >= first_owner; --owner) {
        const uint32_t old_begin = offsets[static_cast<size_t>(owner)];
        const auto new_end = static_cast<uint32_t>(write);
        size_t a_lo = a_hi;
        while (a_lo > 0 && additions[a_lo - 1].first == owner) {
            --a_lo;
        }
        // Descending merge into the tail, so the read cursors are always at or
        // behind the write cursor and nothing is overwritten before it is read.
        size_t i = old_end;
        size_t j = a_hi;
        while (i > old_begin || j > a_lo) {
            const bool take_old =
                (j == a_lo) || (i > old_begin && ids[i - 1] > additions[j - 1].second);
            ids[--write] = take_old ? ids[--i] : additions[--j].second;
        }
        offsets[static_cast<size_t>(owner) + 1] = new_end;
        a_hi = a_lo;
        old_end = old_begin;
    }
    // Nothing below first_owner moved, so its start is where it always was.
    assert(write == offsets[static_cast<size_t>(first_owner)]);
    (void)write;
}

// Give one grown node a fresh contiguous child slice at the end of `child_refs`,
// leaving its old slice as a hole.
//
// Relocation rather than insertion because a node addresses its children by
// (child_begin, child_count): inserting in place would shift every later node's
// slice and invalidate half the offsets in the model. The cost is O(new arity)
// per grown node, so appending k terms one at a time to a row of arity a costs
// O(k*a). That is the regime this loses in -- many separate appends to one very
// wide row -- and it wins in the one column generation is actually in, where a
// row's arity is small next to the model and the alternative is an O(model)
// rebuild per append.
void relocate_grown_children(ModelStructure& st, int32_t node_id,
                             const std::vector<ChildRef>& terms) {
    const ExprNode& nd = st.nodes[static_cast<size_t>(node_id)];
    const auto begin = static_cast<size_t>(nd.child_begin);
    std::vector<ChildRef> slice(
        st.child_refs.begin() + static_cast<std::ptrdiff_t>(begin),
        st.child_refs.begin() + static_cast<std::ptrdiff_t>(begin + nd.child_count));
    slice.insert(slice.end(), terms.begin(), terms.end());
    const size_t new_begin = st.child_refs.size();
    if (new_begin + slice.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("model has more than 2^32 - 1 child references");
    }
    const uint32_t old_count = nd.child_count;
    st.child_refs.insert(st.child_refs.end(), slice.begin(), slice.end());
    ExprNode& target = st.nodes[static_cast<size_t>(node_id)];
    target.child_begin = static_cast<uint32_t>(new_begin);
    target.child_count = static_cast<uint32_t>(slice.size());
    st.child_ref_holes += old_count;
}

// Rewrite `child_refs` without its holes and rebase every node's child_begin.
//
// O(model), so it is amortised: triggered only once the holes are at least half
// the array, which bounds the total relocation waste at one extra copy of the
// edge list however many appends a run makes.
void compact_child_refs(ModelStructure& st) {
    std::vector<ChildRef> packed;
    packed.reserve(st.child_refs.size() - st.child_ref_holes);
    for (ExprNode& nd : st.nodes) {
        const auto begin = static_cast<std::ptrdiff_t>(nd.child_begin);
        nd.child_begin = static_cast<uint32_t>(packed.size());
        packed.insert(packed.end(), st.child_refs.begin() + begin,
                      st.child_refs.begin() + begin + static_cast<std::ptrdiff_t>(nd.child_count));
    }
    st.child_refs.swap(packed);
    st.child_ref_holes = 0;
}

// Every variable the subtree rooted at `root` reaches, distinct, in the order the
// walk finds them.
//
// The stamps are `unordered_map`s rather than the node-indexed arrays
// `build_var_constraints` uses, and deliberately: an array costs O(model) to
// allocate and clear, which is the cost this whole path exists to avoid. `epoch`
// separates roots so a node shared between two rows is visited once per row, as
// it is in the wholesale build.
void collect_cone_vars(const Model& model, int32_t root, int32_t epoch,
                       std::unordered_map<int32_t, int32_t>& node_stamp,
                       std::unordered_map<int32_t, int32_t>& var_stamp,
                       std::vector<int32_t>& out_vars, std::vector<int32_t>& stack) {
    stack.clear();
    node_stamp[root] = epoch;
    stack.push_back(root);
    while (!stack.empty()) {
        const int32_t nid = stack.back();
        stack.pop_back();
        for (const ChildRef& child : model.children(model.nodes()[static_cast<size_t>(nid)])) {
            std::unordered_map<int32_t, int32_t>& stamp = child.is_var ? var_stamp : node_stamp;
            int32_t& seen = stamp[child.id];
            if (seen == epoch) {
                continue;
            }
            seen = epoch;
            if (child.is_var) {
                out_vars.push_back(child.id);
            } else {
                stack.push_back(child.id);
            }
        }
    }
}

// Ancestors of `roots`, upward through `parents`, including the roots themselves.
std::unordered_set<int32_t> collect_ancestors(const Model& model,
                                              const std::vector<int32_t>& roots) {
    std::unordered_set<int32_t> seen;
    std::vector<int32_t> stack;
    for (const int32_t r : roots) {
        if (seen.insert(r).second) {
            stack.push_back(r);
        }
    }
    while (!stack.empty()) {
        const int32_t nid = stack.back();
        stack.pop_back();
        for (const int32_t parent : model.parents(nid)) {
            if (seen.insert(parent).second) {
                stack.push_back(parent);
            }
        }
    }
    return seen;
}

// Recompute the node values the extension invalidated, and only those: the new
// nodes, the grown nodes, and everything above them.
//
// The same shape as `delta_evaluate`'s walk, but seeded from NODES rather than
// from changed variables -- a grown Sum is dirty without any variable having
// moved. It is a separate body rather than a `delta_evaluate` overload because
// the mode protocol a `CustomInvariant` reads is defined in terms of a variable
// move, which this is not; `Model::extend` sends a model with custom nodes down
// `full_evaluate` instead.
void evaluate_extension_cone(Model& model, int32_t first_new_node, int32_t end_new_node,
                             const std::vector<int32_t>& grown_nodes) {
    std::unordered_set<int32_t> dirty;
    std::vector<int32_t> list;
    auto mark = [&](int32_t nid) {
        if (dirty.insert(nid).second) {
            list.push_back(nid);
        }
    };
    for (int32_t nid = first_new_node; nid < end_new_node; ++nid) {
        mark(nid);
    }
    for (const int32_t nid : grown_nodes) {
        mark(nid);
    }
    // `list` grows inside the walk, so it is indexed rather than iterated: a
    // range-based loop over a container the body appends to is undefined.
    size_t head = 0;
    while (head < list.size()) {
        const int32_t nid = list[head];
        ++head;
        for (const int32_t parent : model.parents(nid)) {
            mark(parent);
        }
    }
    std::sort(list.begin(), list.end(), [&model](int32_t a, int32_t b) {
        return model.topo_position(a) < model.topo_position(b);
    });
    for (const int32_t nid : list) {
        model.set_node_value_unchecked(nid,
                                       evaluate(model.nodes()[static_cast<size_t>(nid)], model));
    }
}

// Collect the (variable, constraint index) incidences of each row in `rows`, by
// walking that row's subtree. The splice's `canonicalise` then drops the ones the
// row already had, so this deliberately reports a row's FULL variable set rather
// than trying to work out which part of it is new -- which is both simpler and
// exactly the O(size of the touched rows) the extension is costed at.
std::vector<Addition> collect_incidence_additions(const Model& model,
                                                  const std::vector<int32_t>& rows) {
    std::vector<Addition> adds;
    std::unordered_map<int32_t, int32_t> node_stamp;
    std::unordered_map<int32_t, int32_t> var_stamp;
    std::vector<int32_t> vars;
    std::vector<int32_t> stack;
    const std::vector<int32_t>& cids = model.constraint_ids();
    for (size_t k = 0; k < rows.size(); ++k) {
        vars.clear();
        // k + 1: a missing key reads as 0 through operator[], so 0 must not be a
        // live epoch.
        collect_cone_vars(model, cids[static_cast<size_t>(rows[k])], static_cast<int32_t>(k) + 1,
                          node_stamp, var_stamp, vars, stack);
        for (const int32_t v : vars) {
            adds.emplace_back(v, rows[k]);
        }
    }
    return adds;
}

// Can the existing topological order absorb the new nodes as one contiguous
// block, or must it be recomputed?
//
// The block goes immediately before the earliest grown node, because a term
// appended to a Sum has to be evaluated before it. Two things can make that
// impossible, and both need an EXISTING node to be in the wrong place already:
// a new node whose existing child sits at or after the insertion point, and an
// existing node appended as a term to a Sum that precedes it. Neither is
// reachable by building a term out of a new variable and a new constant, which is
// what column generation does -- but a caller can write either, so the answer is
// a full re-sort rather than a refusal.
bool plan_topo_insert(const Model& model, const ModelStructure& st, const ExtensionResult& res,
                      const std::map<int32_t, std::vector<ChildRef>>& grown, int32_t& insert_pos) {
    insert_pos = static_cast<int32_t>(st.topo_order.size());
    for (const auto& g : grown) {
        insert_pos = std::min(insert_pos, st.topo_pos[static_cast<size_t>(g.first)]);
    }
    for (int32_t nid = res.first_new_node; nid < res.end_node(); ++nid) {
        for (const ChildRef& child : model.children(st.nodes[static_cast<size_t>(nid)])) {
            const bool late_existing_child =
                !child.is_var && child.id < res.first_new_node &&
                st.topo_pos[static_cast<size_t>(child.id)] >= insert_pos;
            if (late_existing_child) {
                return false;
            }
        }
    }
    for (const auto& g : grown) {
        const int32_t target_pos = st.topo_pos[static_cast<size_t>(g.first)];
        for (const ChildRef& term : g.second) {
            const bool late_existing_term = !term.is_var && term.id < res.first_new_node &&
                                            st.topo_pos[static_cast<size_t>(term.id)] >= target_pos;
            if (late_existing_term) {
                return false;
            }
        }
    }
    return true;
}

// Splice the new nodes into `topo_order` at `insert_pos`, in id order -- which is
// a valid order among themselves, because `ModelExtension` validates a node's
// children when it is recorded, so a child was always recorded first.
//
// `topo_pos` is then renumbered from the insertion point onward. Keeping the
// order DENSE rather than moving to a sparse position space is deliberate:
// `evaluate_dirty_in_topo_order` only compares positions and would not care, but
// `full_evaluate` walks the dense `topo_order` array itself.
void insert_topo_block(ModelStructure& st, const ExtensionResult& res, int32_t insert_pos) {
    std::vector<int32_t> block(static_cast<size_t>(res.num_new_nodes));
    for (int32_t i = 0; i < res.num_new_nodes; ++i) {
        block[static_cast<size_t>(i)] = res.first_new_node + i;
    }
    st.topo_order.insert(st.topo_order.begin() + insert_pos, block.begin(), block.end());
    st.topo_pos.resize(st.nodes.size());
    for (auto i = static_cast<size_t>(insert_pos); i < st.topo_order.size(); ++i) {
        st.topo_pos[static_cast<size_t>(st.topo_order[i])] = static_cast<int32_t>(i);
    }
}

// Every back-reference edge the extension adds: a new node names its children,
// and an appended term is named by the Sum it was appended to.
void extend_back_references(const Model& model, ModelStructure& st, const ExtensionResult& res,
                            const std::map<int32_t, std::vector<ChildRef>>& grown) {
    std::vector<Addition> parent_adds;
    std::vector<Addition> dependent_adds;
    for (int32_t nid = res.first_new_node; nid < res.end_node(); ++nid) {
        for (const ChildRef& child : model.children(st.nodes[static_cast<size_t>(nid)])) {
            (child.is_var ? dependent_adds : parent_adds).emplace_back(child.id, nid);
        }
    }
    for (const auto& g : grown) {
        for (const ChildRef& term : g.second) {
            (term.is_var ? dependent_adds : parent_adds).emplace_back(term.id, g.first);
        }
    }
    canonicalise(st.parent_offsets, st.parent_ids, parent_adds);
    splice_csr(st.parent_offsets, st.parent_ids, parent_adds);
    canonicalise(st.dependent_offsets, st.dependent_ids, dependent_adds);
    splice_csr(st.dependent_offsets, st.dependent_ids, dependent_adds);
}

// Grow G_v, and record which existing rows the extension reached.
void extend_var_constraints(const Model& model, ModelStructure& st, ExtensionResult& res,
                            const std::map<int32_t, std::vector<ChildRef>>& grown) {
    // New variables need an empty G_v range at the end. Unlike `dependent_offsets`
    // this array is NOT extended as variables are made (see `ModelStructure`), so
    // it is done here -- and before the splice, which addresses its owners
    // through it.
    if (st.var_constraint_offsets.empty()) {
        st.var_constraint_offsets.push_back(0);
    }
    if (res.num_new_vars > 0) {
        st.var_constraint_offsets.insert(st.var_constraint_offsets.end(),
                                         static_cast<size_t>(res.num_new_vars),
                                         st.var_constraint_offsets.back());
    }
    std::vector<int32_t> rows;
    if (!grown.empty()) {
        std::vector<int32_t> roots;
        roots.reserve(grown.size());
        for (const auto& g : grown) {
            roots.push_back(g.first);
        }
        // An extension reaches an existing row only through a grown Sum, so the
        // rows whose body changed are exactly the constraint roots above one.
        // Finding them costs one pass over the constraint list, which is why it is
        // skipped outright when nothing was appended.
        const std::unordered_set<int32_t> ancestors = collect_ancestors(model, roots);
        for (int32_t ci = 0; ci < res.first_new_constraint; ++ci) {
            if (ancestors.count(st.constraint_ids[static_cast<size_t>(ci)]) != 0) {
                res.touched_constraints.push_back(ci);
            }
        }
        rows = res.touched_constraints;
    }
    for (int32_t ci = res.first_new_constraint; ci < res.end_constraint(); ++ci) {
        rows.push_back(ci);
    }

    std::vector<Addition> adds = collect_incidence_additions(model, rows);
    canonicalise(st.var_constraint_offsets, st.var_constraint_ids, adds);
    res.new_incidences.reserve(adds.size());
    for (const Addition& a : adds) {
        res.new_incidences.emplace_back(a.second, a.first);
    }
    std::sort(res.new_incidences.begin(), res.new_incidences.end());
    splice_csr(st.var_constraint_offsets, st.var_constraint_ids, adds);
}

}  // namespace

// The new variables and nodes, in the order `ModelExtension` recorded them. Split
// out of `extend` because it is the one step that writes `Model`'s own per-model
// arrays (`vars_`, `node_values_`) rather than the structure.
void Model::append_extension_entities(const ModelExtension& ext, ExtensionResult& res) {
    ModelStructure& st = mut();
    res.new_var_initial.reserve(ext.new_vars_.size());
    for (const ModelExtension::NewVar& nv : ext.new_vars_) {
        const int32_t vid = alloc_var(nv.type, nv.lb, nv.ub, nv.name);
        vars_[static_cast<size_t>(vid)].value = nv.initial;
        res.new_var_initial.push_back(nv.initial);
    }
    res.num_new_vars = static_cast<int32_t>(ext.new_vars_.size());
    for (const ModelExtension::NewNode& nn : ext.new_nodes_) {
        const size_t begin = st.child_refs.size();
        for (const int32_t handle : nn.children) {
            st.child_refs.push_back(wrap(handle));
        }
        const int32_t nid = push_node(nn.op, begin);
        if (nn.op == NodeOp::Const) {
            // Mirrors Model::constant: the value is both structure and the node's
            // initial evaluation.
            st.nodes[static_cast<size_t>(nid)].const_value = nn.const_value;
            node_values_[static_cast<size_t>(nid)] = nn.const_value;
        }
    }
    res.num_new_nodes = static_cast<int32_t>(ext.new_nodes_.size());
}

ExtensionResult Model::extend(const ModelExtension& ext) {
    if (is_frozen()) {
        throw std::logic_error(
            "Model::extend: the model is frozen. freeze() publishes one ModelStructure to every "
            "portfolio replica, so growing it would mutate a peer's model under a running search "
            "-- ParallelSearch::solve(Model&) and the CLI at --threads > 1 both freeze, so growth "
            "is single-solve() only until #168 gives each worker its own extension overlay");
    }
    if (!closed_) {
        throw std::logic_error(
            "Model::extend: the model is not closed. Build it with the ordinary Model builders "
            "and call close(); extend() exists for a model whose derived indices already exist");
    }
    if (ext.base_ != this || ext.base_num_vars_ != static_cast<int32_t>(vars_.size()) ||
        ext.base_num_nodes_ != static_cast<int32_t>(s().nodes.size())) {
        throw std::invalid_argument(
            "Model::extend: the extension was built against a different model, or against this "
            "model before it grew. Its handles are absolute ids, so replaying it now would name "
            "the wrong entities");
    }

    ExtensionResult res;
    res.first_new_var = static_cast<int32_t>(vars_.size());
    res.first_new_node = static_cast<int32_t>(s().nodes.size());
    res.first_new_constraint = static_cast<int32_t>(s().constraint_ids.size());
    if (ext.empty()) {
        return res;
    }

    // Nothing below reserves exactly. Every array here grows through push_back or
    // insert, so libstdc++'s geometric policy amortises the reallocation over a
    // run of extends -- which is the regime a column-generation loop is in, and
    // the opposite of `add_objective_soft_constraint`'s, which reserves exactly
    // because it runs once and a doubling there is address space for nothing. The
    // price is one full copy of each array on the first extend after a build that
    // sized them exactly; `Model::extend`'s comment carries the measurement.
    append_extension_entities(ext, res);
    ModelStructure& st = mut();

    // Terms appended to existing Sum rows, grouped so a row relocates once
    // however many terms it gains. std::map rather than unordered_map because the
    // relocation order decides the layout of `child_refs`, and that has to be a
    // function of the extension alone.
    std::map<int32_t, std::vector<ChildRef>> grown;
    for (const std::pair<int32_t, int32_t>& append : ext.appends_) {
        grown[append.first].push_back(wrap(append.second));
    }
    for (const auto& g : grown) {
        relocate_grown_children(st, g.first, g.second);
    }
    if (st.child_ref_holes > 0 && st.child_ref_holes * 2 >= st.child_refs.size()) {
        compact_child_refs(st);
    }

    st.constraint_ids.insert(st.constraint_ids.end(), ext.new_constraints_.begin(),
                             ext.new_constraints_.end());
    res.num_new_constraints = static_cast<int32_t>(ext.new_constraints_.size());

    extend_back_references(*this, st, res, grown);

    int32_t insert_pos = 0;
    if (plan_topo_insert(*this, st, res, grown, insert_pos)) {
        insert_topo_block(st, res, insert_pos);
    } else {
        st.topo_order = detail::compute_topo_order(*this);
        rebuild_topo_positions();
        res.topo_order_rebuilt = true;
    }

    extend_var_constraints(*this, st, res, grown);

    if (has_custom_nodes()) {
        // A CustomInvariant's delta() is defined against a variable move, which
        // this is not; full_evaluate is the interface's documented reset point.
        // O(model), and the one evaluation regime extend does not improve on.
        full_evaluate(*this);
    } else {
        std::vector<int32_t> grown_nodes;
        grown_nodes.reserve(grown.size());
        for (const auto& g : grown) {
            grown_nodes.push_back(g.first);
        }
        evaluate_extension_cone(*this, res.first_new_node, res.end_node(), grown_nodes);
    }
    return res;
}

void pad_state(Model::State& state, const ExtensionResult& ext) {
    const auto base = static_cast<size_t>(ext.first_new_var);
    if (state.values.size() != base || state.elements.size() != base) {
        throw std::invalid_argument(
            "pad_state: the state is not the size the model had before this extension. Padding a "
            "state of any other size would make Model::restore_state's size check -- the guard "
            "that stops a short ModelState from Python becoming an unguarded read -- a check on "
            "nothing");
    }
    state.values.insert(state.values.end(), ext.new_var_initial.begin(), ext.new_var_initial.end());
    // A new variable is a scalar (see ModelExtension), so its elements stay empty.
    state.elements.resize(base + ext.new_var_initial.size());
}

}  // namespace cbls
