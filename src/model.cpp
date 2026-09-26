#include "cbls/model.h"

#include "cbls/dag_ops.h"
#include "cbls/expr.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cbls {

// Forward declare from dag_ops.cpp
namespace detail {
std::vector<int32_t> compute_topo_order(const Model& model);
}

// The structure is heap-allocated from the start, and both handles point at it,
// so every accessor is unconditional: there is no "not allocated yet" state to
// branch on, and `freeze()` is the one thing that changes which handles exist.
Model::Model() {
    auto structure = std::make_shared<ModelStructure>();
    open_structure_ = structure;
    structure_ = std::move(structure);
}

Model::Model(const Model& other)
    : vars_(other.vars_),
      node_values_(other.node_values_),
      objective_id_(other.objective_id_),
      is_maximizing_(other.is_maximizing_),
      objective_bound_node_(other.objective_bound_node_),
      objective_constraint_node_(other.objective_constraint_node_),
      objective_constraint_idx_(other.objective_constraint_idx_),
      objective_bound_(other.objective_bound_),
      closed_(other.closed_) {
    // probe_old_violation_ is deliberately left empty: it is resized and
    // overwritten before it is read on every call, so it carries no state.
    //
    // custom_invariants_ is the opposite: it IS state, and a per-worker copy of
    // it is the whole reason a custom node can be stateful at all (#166). Cloned
    // one by one, in order, because the index is `ExprNode::lambda_func_id` and
    // that lives in the structure the replicas share. A pending probe is not
    // carried over -- nothing copies a model mid-probe, and starting a replica
    // owing a rollback would be worse than starting it owing nothing.
    custom_invariants_.reserve(other.custom_invariants_.size());
    for (const CustomInvariantSlot& slot : other.custom_invariants_) {
        CustomInvariantSlot copy;
        copy.invariant = slot.invariant->clone();
        if (copy.invariant == nullptr) {
            throw std::invalid_argument("CustomInvariant::clone() returned null");
        }
        copy.name = slot.name;
        custom_invariants_.push_back(std::move(copy));
    }
    if (other.open_structure_ == nullptr) {
        // Frozen: share. This is the whole point -- a portfolio replica costs the
        // variables and the node values, not the DAG (#157). Reading `other`
        // concurrently from N worker threads is safe: nothing here writes to it,
        // and a shared_ptr copy is atomic.
        structure_ = other.structure_;
    } else {
        // Open: deep-copy, exactly as the implicit copy did before the split, so
        // that a half-built model can be copied and then extended on its own.
        auto structure = std::make_shared<ModelStructure>(*other.structure_);
        open_structure_ = structure;
        structure_ = std::move(structure);
    }
}

void Model::freeze() {
    if (is_frozen()) {
        return;
    }
    if (!closed_) {
        close();
    }
    if (objective_id_ >= 0) {
        // Idempotent, and the reason the freeze point is here rather than at
        // close(): see the comment on freeze() in the header.
        add_objective_soft_constraint();
    }
    // Dropping the writable handle is what makes every structural method throw,
    // and it leaves `structure_` -- a const handle -- as the only way in.
    open_structure_.reset();
}

void Model::require_open(const char* method) const {
    if (is_frozen()) {
        throw std::logic_error(std::string(method) +
                               ": model is frozen, its structure is shared and cannot change");
    }
}

Model& Model::operator=(const Model& other) {
    if (this != &other) {
        Model copy(other);
        *this = std::move(copy);
    }
    return *this;
}

namespace {
// Mirror ViolationManager's clamp: a non-convex node value that overflows to
// +inf or NaN is mapped to a large finite penalty so jump scoring stays ordered
// and never propagates NaN/inf into the search. Must match violation.cpp.
double clamped_node_violation(double node_value) {
    // NaN before max(): std::max(0.0, NaN) == 0.0 would mask a NaN as satisfied.
    if (std::isnan(node_value)) {
        return kInfPenalty;
    }
    double v = std::max(0.0, node_value);
    if (v > kInfPenalty) {  // also catches +inf
        return kInfPenalty;
    }
    return v;
}
}  // namespace

int32_t Model::alloc_var(VarType type, double lb, double ub, const std::string& name) {
    ModelStructure& st = mut();
    Variable v;
    v.id = static_cast<int32_t>(vars_.size());
    v.type = type;
    v.value = lb;
    v.lb = lb;
    v.ub = ub;
    v.name = name;
    // The variable's empty dependents range goes in FIRST: if the push below
    // throws, the offsets are merely one entry longer than needed (still an
    // empty range) rather than one short.
    if (st.dependent_offsets.empty()) {
        st.dependent_offsets.push_back(0);
    }
    st.dependent_offsets.push_back(st.dependent_offsets.back());
    vars_.push_back(std::move(v));
    return vars_.back().id;
}

// Append a node whose children are `st.child_refs[child_begin ..]`, already
// written by the caller. The one place a node is made, so the two invariants the
// flat storage rests on are kept in one place: the child slice fits the 32-bit
// offsets ExprNode carries, and st.parent_offsets stays one longer than st.nodes.
int32_t Model::push_node(NodeOp op, size_t child_begin) {
    ModelStructure& st = mut();
    if (st.child_refs.size() > std::numeric_limits<uint32_t>::max()) {
        st.child_refs.resize(child_begin);  // leave the model as it was
        throw std::length_error("model has more than 2^32 - 1 child references");
    }
    ExprNode nd;
    nd.id = static_cast<int32_t>(st.nodes.size());
    nd.op = op;
    nd.child_begin = static_cast<uint32_t>(child_begin);
    nd.child_count = static_cast<uint32_t>(st.child_refs.size() - child_begin);
    // Offsets before the node, for the reason alloc_var gives.
    if (st.parent_offsets.empty()) {
        st.parent_offsets.push_back(0);
    }
    st.parent_offsets.push_back(st.parent_offsets.back());
    // The value array is grown here, with the node, so the two can never drift:
    // every accessor indexes both by the same id. BEFORE the node, for the
    // reason alloc_var gives -- if this throws nothing was added at all, and if
    // the push below throws the array is merely one entry long for a node that
    // does not exist, which nothing can index.
    node_values_.push_back(0.0);
    st.nodes.push_back(nd);
    return nd.id;
}

int32_t Model::alloc_node(NodeOp op, std::initializer_list<ChildRef> children) {
    ModelStructure& st = mut();
    const size_t begin = st.child_refs.size();
    st.child_refs.insert(st.child_refs.end(), children.begin(), children.end());
    return push_node(op, begin);
}

// The variadic ops' builder: children written straight into the flat array from
// the caller's handles, with no intermediate ChildRef vector.
int32_t Model::alloc_node_over_handles(NodeOp op, const std::vector<int32_t>& handles) {
    ModelStructure& st = mut();
    // No reserve(size + n) here: libstdc++ reserves exactly what is asked, so
    // doing it per node would defeat geometric growth and copy the whole array
    // on every call.
    const size_t begin = st.child_refs.size();
    try {
        for (const int32_t h : handles) {
            st.child_refs.push_back(wrap(h));
        }
    } catch (...) {
        // A rejected node leaves no children behind. Nothing can observe the
        // difference -- every node addresses its own slice, so orphaned entries
        // are inert -- but a caller that catches and retries would otherwise
        // grow the array without bound.
        st.child_refs.resize(begin);
        throw;
    }
    return push_node(op, begin);
}

// Handle encoding: var handles = -(var_id + 1) (negative), node handles = node_id
// (non-negative).
//
// Validated here, when the node naming the handle is made, because nothing later
// can be: the back-reference rebuild counts every child into a CSR offsets array
// indexed by its id, so an id past the end is a silent heap write, not an
// exception (#156 -- the per-node vectors it replaced were reached through
// throwing accessors). Python passes raw integers, so this is reachable from a
// typo. It also rules out naming a node before it exists, which the topological
// sort never supported.
ChildRef Model::wrap(int32_t handle) const {
    const ModelStructure& st = s();
    ChildRef ref;
    if (handle < 0) {
        ref.id = -(handle + 1);
        ref.is_var = true;
        if (static_cast<size_t>(ref.id) >= vars_.size()) {
            throw std::out_of_range("variable handle out of range");
        }
    } else {
        ref.id = handle;
        ref.is_var = false;
        if (static_cast<size_t>(ref.id) >= st.nodes.size()) {
            throw std::out_of_range("node handle out of range");
        }
    }
    return ref;
}

// Variable creation methods return negative handles: -(var_id + 1)
int32_t Model::bool_var(const std::string& name) {
    int32_t vid = alloc_var(VarType::Bool, 0.0, 1.0, name);
    return -(vid + 1);  // encode as var handle
}

int32_t Model::int_var(int lb, int ub, const std::string& name) {
    int32_t vid = alloc_var(VarType::Int, static_cast<double>(lb), static_cast<double>(ub), name);
    return -(vid + 1);
}

int32_t Model::float_var(double lb, double ub, const std::string& name) {
    int32_t vid = alloc_var(VarType::Float, lb, ub, name);
    return -(vid + 1);
}

int32_t Model::list_var(int n, const std::string& name) {
    return list_var(n, n, n, ListInit::Identity, name);
}

int32_t Model::list_var(int universe, int min_len, int max_len, ListInit init,
                        const std::string& name) {
    if (universe < 0) {
        throw std::invalid_argument("list_var: negative universe");
    }
    if (min_len < 0 || min_len > max_len || max_len > universe) {
        throw std::invalid_argument("list_var: require 0 <= min_len <= max_len <= universe");
    }
    if (init == ListInit::Empty && min_len > 0) {
        // Empty is only a legal assignment when zero is a legal length. Allowing
        // it otherwise gives a List that starts -- and that every later
        // `randomize_var(Regenerate)` returns to -- below its own min_len, which
        // no constraint row prices and no move guard reports: `list_remove`
        // refuses to shrink it further and `list_insert` grows it one element
        // per candidate. `ListInit::Random` is the one that respects a minimum.
        throw std::invalid_argument("list_var: ListInit::Empty requires min_len == 0");
    }
    if (init == ListInit::Identity && (min_len != universe || max_len != universe)) {
        // Identity means "every element, in order", which is only a legal
        // assignment when the length is pinned at the universe. Rejecting it
        // here is what lets `randomize_structured_var` read Identity as exactly
        // the pre-#164 permutation draw.
        throw std::invalid_argument(
            "list_var: ListInit::Identity requires min_len == max_len == universe");
    }
    int32_t vid = alloc_var(VarType::List, 0.0, 0.0, name);
    auto& v = vars_[vid];
    v.universe_size = universe;
    v.min_size = min_len;
    v.max_size = max_len;
    v.list_init = init;
    if (init == ListInit::Identity) {
        v.elements.resize(universe);
        for (int i = 0; i < universe; ++i) {
            v.elements[i] = i;
        }
    } else {
        // Empty and Random both start empty; `initialize_structured_random`
        // fills a Random list (and any Exact-partition member) before the search
        // reads it. A model that never calls it -- a bare `full_evaluate` on a
        // freshly built model -- therefore sees an empty list, which is a legal
        // assignment whenever min_len is 0 and the honest answer otherwise.
        v.elements.clear();
    }
    return -(vid + 1);
}

int32_t Model::set_var(int n, int min_size, int max_size, const std::string& name) {
    int32_t vid = alloc_var(VarType::Set, 0.0, 0.0, name);
    auto& v = vars_[vid];
    v.universe_size = n;
    v.min_size = min_size;
    v.max_size = (max_size < 0) ? n : max_size;
    return -(vid + 1);
}

// Expression creation methods return non-negative handles (node IDs)
int32_t Model::constant(double val) {
    ModelStructure& st = mut();
    const int32_t nid = push_node(NodeOp::Const, st.child_refs.size());
    st.nodes[nid].const_value = val;
    node_values_[nid] = val;
    return nid;
}

int32_t Model::neg(int32_t x) {
    return alloc_node(NodeOp::Neg, {wrap(x)});
}

int32_t Model::sum(const std::vector<int32_t>& args) {
    if (args.empty()) {
        return constant(0.0);
    }
    return alloc_node_over_handles(NodeOp::Sum, args);
}

int32_t Model::prod(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Prod, {wrap(a), wrap(b)});
}

int32_t Model::div_expr(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Div, {wrap(a), wrap(b)});
}

int32_t Model::pow_expr(int32_t base, int32_t exp) {
    return alloc_node(NodeOp::Pow, {wrap(base), wrap(exp)});
}

// Min and Max evaluate children[0] unchecked, so an empty one would read
// another node's child slice (see child_val in dag.cpp).
int32_t Model::min_expr(const std::vector<int32_t>& args) {
    if (args.empty()) {
        throw std::invalid_argument("min_expr requires at least one argument");
    }
    return alloc_node_over_handles(NodeOp::Min, args);
}

int32_t Model::max_expr(const std::vector<int32_t>& args) {
    if (args.empty()) {
        throw std::invalid_argument("max_expr requires at least one argument");
    }
    return alloc_node_over_handles(NodeOp::Max, args);
}

int32_t Model::abs_expr(int32_t x) {
    return alloc_node(NodeOp::Abs, {wrap(x)});
}

int32_t Model::sin_expr(int32_t x) {
    return alloc_node(NodeOp::Sin, {wrap(x)});
}

int32_t Model::cos_expr(int32_t x) {
    return alloc_node(NodeOp::Cos, {wrap(x)});
}

int32_t Model::tan_expr(int32_t x) {
    return alloc_node(NodeOp::Tan, {wrap(x)});
}

int32_t Model::exp_expr(int32_t x) {
    return alloc_node(NodeOp::Exp, {wrap(x)});
}

int32_t Model::log_expr(int32_t x) {
    return alloc_node(NodeOp::Log, {wrap(x)});
}

int32_t Model::sqrt_expr(int32_t x) {
    return alloc_node(NodeOp::Sqrt, {wrap(x)});
}

int32_t Model::signpower_expr(int32_t base, int32_t exp) {
    return alloc_node(NodeOp::SignPower, {wrap(base), wrap(exp)});
}

int32_t Model::tanh_expr(int32_t x) {
    return alloc_node(NodeOp::Tanh, {wrap(x)});
}

int32_t Model::if_then_else(int32_t cond, int32_t then_, int32_t else_) {
    return alloc_node(NodeOp::If, {wrap(cond), wrap(then_), wrap(else_)});
}

int32_t Model::at(int32_t list_var_id, int32_t index_expr) {
    return alloc_node(NodeOp::At, {wrap(list_var_id), wrap(index_expr)});
}

int32_t Model::count(int32_t var_id) {
    return alloc_node(NodeOp::Count, {wrap(var_id)});
}

int32_t Model::leq(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Leq, {wrap(a), wrap(b)});
}

int32_t Model::eq_expr(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Eq, {wrap(a), wrap(b)});
}

int32_t Model::geq(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Geq, {wrap(a), wrap(b)});
}

int32_t Model::neq(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Neq, {wrap(a), wrap(b)});
}

int32_t Model::lt(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Lt, {wrap(a), wrap(b)});
}

int32_t Model::gt(int32_t a, int32_t b) {
    return alloc_node(NodeOp::Gt, {wrap(a), wrap(b)});
}

int32_t Model::lambda_sum(int32_t list_var_id, std::function<double(int)> func) {
    ModelStructure& st = mut();
    const ChildRef child = wrap(list_var_id);  // reject a bad handle before registering
    st.lambda_funcs.push_back(std::move(func));
    auto func_id = static_cast<int32_t>(st.lambda_funcs.size() - 1);

    int32_t nid = alloc_node(NodeOp::Lambda, {child});
    st.nodes[nid].lambda_func_id = func_id;
    return nid;
}

int32_t Model::pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func,
                               PairMode mode) {
    return pair_lambda_sum(list_var_id, std::move(func), nullptr, nullptr, mode);
}

int32_t Model::pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func,
                               std::function<double(int)> head, std::function<double(int)> tail,
                               PairMode mode) {
    ModelStructure& st = mut();
    const ChildRef child = wrap(list_var_id);  // reject a bad handle before registering

    // An empty callable means "no endpoint term", which is what -1 records.
    // Registering an empty std::function instead would cost a
    // std::bad_function_call on the evaluation path.
    PairLambdaSpec spec;
    spec.mode = mode;
    if (head) {
        st.lambda_funcs.push_back(std::move(head));
        spec.head_id = static_cast<int32_t>(st.lambda_funcs.size() - 1);
    }
    if (tail) {
        st.lambda_funcs.push_back(std::move(tail));
        spec.tail_id = static_cast<int32_t>(st.lambda_funcs.size() - 1);
    }

    // The two tables are addressed by one id, so they must stay the same
    // length. Reserving first is what makes the second push unable to throw
    // after the first has happened: a bad_alloc between them would leave every
    // LATER pair node indexing a spec table one short.
    st.pair_lambda_specs.reserve(st.pair_lambda_funcs.size() + 1);
    st.pair_lambda_funcs.push_back(std::move(func));
    st.pair_lambda_specs.push_back(spec);
    auto func_id = static_cast<int32_t>(st.pair_lambda_funcs.size() - 1);

    int32_t nid = alloc_node(NodeOp::PairLambda, {child});
    st.nodes[nid].lambda_func_id = func_id;
    return nid;
}

int32_t Model::custom(const std::vector<int32_t>& inputs, std::unique_ptr<CustomInvariant> inv,
                      const std::string& name) {
    ModelStructure& st = mut();  // rejects a frozen model before anything is registered
    if (inv == nullptr) {
        throw std::invalid_argument("custom: invariant must not be null");
    }
    const auto slot_id = static_cast<int32_t>(custom_invariants_.size());
    // The slot is BUILT and its space RESERVED before the node exists, so that
    // nothing between `alloc_node_over_handles` and the `lambda_func_id` write can
    // throw. `slot.name = name` allocates, and doing it on the far side of the node
    // would leave a Custom node carrying -1 for a slot that was never appended --
    // which throws out of the middle of the first evaluation, or out of the `.cbls`
    // writer's refusal, instead of out of this call. `CustomInvariantSlot`'s move is
    // noexcept, so the push_back into reserved capacity cannot throw either. The
    // reserve is the same argument, and the same shape, as the one in
    // pair_lambda_sum above; a throw from `alloc_node_over_handles` can still leave
    // it in place, which costs one pointer of capacity.
    CustomInvariantSlot slot;
    slot.invariant = std::move(inv);
    slot.name = name;
    custom_invariants_.reserve(custom_invariants_.size() + 1);
    const int32_t nid = alloc_node_over_handles(NodeOp::Custom, inputs);
    custom_invariants_.push_back(std::move(slot));
    st.nodes[nid].lambda_func_id = slot_id;
    return nid;
}

// Expr-returning variable creation
Expr Model::Bool(const std::string& name) {
    return {this, bool_var(name)};
}

Expr Model::Int(int lb, int ub, const std::string& name) {
    return {this, int_var(lb, ub, name)};
}

Expr Model::Float(double lb, double ub, const std::string& name) {
    return {this, float_var(lb, ub, name)};
}

Expr Model::List(int n, const std::string& name) {
    return {this, list_var(n, name)};
}

Expr Model::List(int universe, int min_len, int max_len, ListInit init, const std::string& name) {
    return {this, list_var(universe, min_len, max_len, init, name)};
}

Expr Model::Set(int n, int min_size, int max_size, const std::string& name) {
    return {this, set_var(n, min_size, max_size, name)};
}

Expr Model::Custom(const std::vector<Expr>& inputs, std::unique_ptr<CustomInvariant> inv,
                   const std::string& name) {
    std::vector<int32_t> handles;
    handles.reserve(inputs.size());
    for (const Expr& e : inputs) {
        handles.push_back(e.handle);
    }
    return {this, custom(handles, std::move(inv), name)};
}

Expr Model::Constant(double val) {
    return {this, constant(val)};
}

void Model::add_constraint(const Expr& e) {
    add_constraint(e.handle);
}

void Model::minimize(const Expr& e) {
    minimize(e.handle);
}

void Model::maximize(const Expr& e) {
    maximize(e.handle);
}

void Model::add_constraint(int32_t expr_id) {
    require_open("add_constraint");
    ModelStructure& st = mut();
    if (expr_id < 0) {
        throw std::invalid_argument(
            "add_constraint requires a node handle (non-negative), got var handle");
    }
    if (static_cast<size_t>(expr_id) >= st.nodes.size()) {
        throw std::out_of_range("add_constraint: node handle out of range");
    }
    st.constraint_ids.push_back(expr_id);
}

void Model::minimize(int32_t expr_id) {
    require_open("minimize");
    const ModelStructure& st = s();
    if (expr_id < 0) {
        throw std::invalid_argument(
            "minimize requires a node handle (non-negative), got var handle");
    }
    if (static_cast<size_t>(expr_id) >= st.nodes.size()) {
        throw std::out_of_range("minimize: node handle out of range");
    }
    objective_id_ = expr_id;
}

void Model::maximize(int32_t expr_id) {
    require_open("maximize");
    // Maximize by negating
    objective_id_ = neg(expr_id);
    is_maximizing_ = true;
}

void Model::add_var_sequence(const std::vector<int32_t>& var_ids, int min_block_on,
                             int min_block_off) {
    require_open("add_var_sequence");
    ModelStructure& st = mut();
    int seq_idx = static_cast<int>(st.var_sequences.size());
    VarSequence seq;
    seq.min_block_on = min_block_on;
    seq.min_block_off = min_block_off;

    // Callers pass var handles, as returned by bool_var() etc.; decode each to
    // the internal var id that seq.var_ids holds. The decode leaves a
    // non-negative value alone, so a raw var id survives it, but that is a
    // property of the encoding rather than a supported second calling
    // convention -- a non-negative value is indistinguishable from a node id
    // elsewhere in this API, so do not document it as one.
    seq.var_ids.reserve(var_ids.size());
    for (int32_t h : var_ids) {
        int32_t vid = (h < 0) ? -(h + 1) : h;  // decode var handle
        seq.var_ids.push_back(vid);
    }

    // Grow lookup table if needed
    for (size_t pos = 0; pos < seq.var_ids.size(); ++pos) {
        int32_t vid = seq.var_ids[pos];
        if (vid >= static_cast<int32_t>(st.var_to_seq.size())) {
            st.var_to_seq.resize(vid + 1, {-1, -1});
        }
        st.var_to_seq[vid] = {seq_idx, static_cast<int>(pos)};
    }

    st.var_sequences.push_back(std::move(seq));
}

// A partition is a structural declaration, so it lives in ModelStructure -- but
// it also flips `Variable::partitioned`, which lives in the per-model variable
// array. Both are written here, once, so a member can never be in the structure
// without the flag that stops `list_moves` proposing a move which would break
// the invariant.
int32_t Model::add_list_partition(const std::vector<int32_t>& lists, Cover cover) {
    require_open("add_list_partition");
    if (lists.empty()) {
        throw std::invalid_argument("add_list_partition: no lists");
    }
    ListPartition part;
    part.cover = cover;
    part.list_ids.reserve(lists.size());
    int64_t min_total = 0;
    int64_t max_total = 0;
    for (int32_t handle : lists) {
        // Var handles, as every other public entry point takes them. The decode
        // leaves a non-negative value alone, so a raw var id survives it -- but
        // that is a property of the encoding rather than a second calling
        // convention, exactly as `add_var_sequence` says of itself: a
        // non-negative value is indistinguishable from a NODE id elsewhere in
        // this API. `.cbls`'s partition record resolves names through a table
        // that yields node ids, which is where that would actually bite, so the
        // loader refuses a non-negative handle before it reaches here.
        const int32_t vid = (handle < 0) ? handle_to_var_id(handle) : handle;
        if (vid < 0 || vid >= static_cast<int32_t>(vars_.size())) {
            throw std::invalid_argument("add_list_partition: variable handle out of range");
        }
        const Variable& v = vars_[static_cast<size_t>(vid)];
        if (v.type != VarType::List) {
            throw std::invalid_argument("add_list_partition: '" + v.name +
                                        "' is not a List variable");
        }
        if (v.partitioned ||
            std::find(part.list_ids.begin(), part.list_ids.end(), vid) != part.list_ids.end()) {
            throw std::invalid_argument("add_list_partition: '" + v.name +
                                        "' is already in a partition");
        }
        if (part.list_ids.empty()) {
            part.universe_size = v.universe_size;
        } else if (v.universe_size != part.universe_size) {
            throw std::invalid_argument(
                "add_list_partition: every list must share one universe size");
        }
        if (v.list_init == ListInit::Identity && lists.size() > 1) {
            throw std::invalid_argument(
                "add_list_partition: ListInit::Identity puts every element in every list");
        }
        min_total += v.min_size;
        max_total += v.max_size;
        part.list_ids.push_back(vid);
    }
    if (min_total > part.universe_size) {
        throw std::invalid_argument("add_list_partition: the minimum lengths exceed the universe");
    }
    if (cover == Cover::Exact && max_total < part.universe_size) {
        throw std::invalid_argument(
            "add_list_partition: the maximum lengths cannot cover the universe");
    }

    ModelStructure& st = mut();
    const auto index = static_cast<int32_t>(st.list_partitions.size());
    for (int32_t vid : part.list_ids) {
        if (vid >= static_cast<int32_t>(st.var_to_partition.size())) {
            st.var_to_partition.resize(static_cast<size_t>(vid) + 1, -1);
        }
        st.var_to_partition[static_cast<size_t>(vid)] = index;
        vars_[static_cast<size_t>(vid)].partitioned = true;
    }
    st.list_partitions.push_back(std::move(part));
    return index;
}

int Model::partition_of_list(int32_t var_id) const {
    const ModelStructure& st = s();
    if (var_id >= 0 && var_id < static_cast<int32_t>(st.var_to_partition.size())) {
        return st.var_to_partition[static_cast<size_t>(var_id)];
    }
    return -1;
}

std::pair<int, int> Model::var_sequence_for(int32_t var_id) const {
    const ModelStructure& st = s();
    if (var_id >= 0 && var_id < static_cast<int32_t>(st.var_to_seq.size())) {
        return st.var_to_seq[var_id];
    }
    return {-1, -1};
}

// The inverse permutation of `st.topo_order`, rebuilt wherever that is. Its own
// function because the two call sites must not drift: a stale `st.topo_pos` would
// evaluate a dirty set out of dependency order, which is silent wrong values
// rather than a crash.
void Model::rebuild_topo_positions() {
    ModelStructure& st = mut();
    st.topo_pos.assign(st.nodes.size(), 0);
    for (size_t i = 0; i < st.topo_order.size(); ++i) {
        st.topo_pos[st.topo_order[i]] = static_cast<int32_t>(i);
    }
}

// Rebuild the DAG's back-references: every node's parents and every variable's
// dependents. These are what delta_evaluate walks to find the nodes a changed
// variable dirties, and what the topological sort walks; they are pure derived
// state, so they are recomputed wholesale rather than patched.
//
// Two passes over the edges into CSR -- count, then fill -- so the arrays are
// sized exactly once and nothing is allocated per node. Both passes visit
// parents in ascending id and each parent's children in order, which is the
// order the per-node vectors this replaced were appended in, so every list is
// the same sequence it was.
//
// Deduplicated by a last-writer stamp, not by searching the list being built.
// A duplicate can only ever come from ONE parent naming the same child twice
// (`prod(x, x)`), because a parent is visited once -- so "already recorded by
// this parent" is the whole condition, and a stamp answers it in O(1) where the
// search was O(degree) per edge. That difference is not academic on a real
// matrix: the search made this O(sum of degree^2), which on square47 (95k
// columns in ~288 rows each) was ~3.9 BILLION comparisons and 68% of its model
// build. Each pass needs the stamps fresh, since both skip the same duplicates.
void Model::rebuild_back_references() {
    ModelStructure& st = mut();
    const size_t n_nodes = st.nodes.size();
    const size_t n_vars = vars_.size();
    std::vector<int32_t> node_stamp(n_nodes, -1);
    std::vector<int32_t> var_stamp(n_vars, -1);

    // Pass 1: offsets[i + 1] = number of distinct parents of i.
    st.parent_offsets.assign(n_nodes + 1, 0);
    st.dependent_offsets.assign(n_vars + 1, 0);
    for (const ExprNode& nd : st.nodes) {
        for (const ChildRef& child : children(nd)) {
            if (child.is_var) {
                if (var_stamp[child.id] != nd.id) {
                    var_stamp[child.id] = nd.id;
                    ++st.dependent_offsets[child.id + 1];
                }
            } else if (node_stamp[child.id] != nd.id) {
                node_stamp[child.id] = nd.id;
                ++st.parent_offsets[child.id + 1];
            }
        }
    }
    std::partial_sum(st.parent_offsets.begin(), st.parent_offsets.end(), st.parent_offsets.begin());
    std::partial_sum(st.dependent_offsets.begin(), st.dependent_offsets.end(),
                     st.dependent_offsets.begin());
    // reserve() first because it allocates exactly, where a growing resize()
    // may double -- on a replica's rebuild that would be the whole array again.
    // clear() first because pass 2 overwrites every entry, and reserve() on a
    // non-empty vector copies the stale contents into the new block.
    st.parent_ids.clear();
    st.parent_ids.reserve(st.parent_offsets.back());
    st.parent_ids.resize(st.parent_offsets.back());
    st.dependent_ids.clear();
    st.dependent_ids.reserve(st.dependent_offsets.back());
    st.dependent_ids.resize(st.dependent_offsets.back());

    // Pass 2: fill. offsets[i] serves as i's write cursor, so when the pass is
    // done it has advanced to i's END -- which is offsets[i + 1]'s value -- and
    // one shift right restores the starts.
    std::fill(node_stamp.begin(), node_stamp.end(), -1);
    std::fill(var_stamp.begin(), var_stamp.end(), -1);
    for (const ExprNode& nd : st.nodes) {
        for (const ChildRef& child : children(nd)) {
            if (child.is_var) {
                if (var_stamp[child.id] != nd.id) {
                    var_stamp[child.id] = nd.id;
                    st.dependent_ids[st.dependent_offsets[child.id]++] = nd.id;
                }
            } else if (node_stamp[child.id] != nd.id) {
                node_stamp[child.id] = nd.id;
                st.parent_ids[st.parent_offsets[child.id]++] = nd.id;
            }
        }
    }
    std::copy_backward(st.parent_offsets.begin(), st.parent_offsets.end() - 1,
                       st.parent_offsets.end());
    st.parent_offsets.front() = 0;
    std::copy_backward(st.dependent_offsets.begin(), st.dependent_offsets.end() - 1,
                       st.dependent_offsets.end());
    st.dependent_offsets.front() = 0;
}

void Model::close() {
    require_open("close");
    ModelStructure& st = mut();
    rebuild_back_references();
    st.topo_order = detail::compute_topo_order(*this);
    rebuild_topo_positions();
    build_var_constraints();
    full_evaluate(*this);
    closed_ = true;
}

void Model::add_objective_soft_constraint() {
    if (objective_id_ < 0) {
        throw std::invalid_argument("add_objective_soft_constraint requires an objective");
    }
    // Idempotent, and the early return comes BEFORE the frozen check on purpose:
    // `solve()` calls this on every model it is handed, a frozen replica
    // included, and `freeze()` has already run it -- so on a frozen model this
    // must be a no-op rather than a throw, or no portfolio worker could start.
    if (objective_constraint_idx_ >= 0) {
        return;
    }
    require_open("add_objective_soft_constraint");
    ModelStructure& st = mut();

    // Exactly the room the row takes -- two nodes, two child refs, two node
    // values, one constraint. A copied vector's capacity is its size, so without
    // this a model at exact capacity -- every copy, and every model
    // `mps_to_model` sized exactly -- would DOUBLE each of those arrays to append
    // a handful of entries: address space the benchmark driver's `ulimit -v`
    // counts. On a model that already has the slack these are no-ops.
    //
    // `node_values_` is in the list for the same reason the others are, and is
    // the one that still matches the original argument literally: it is per-model,
    // so it is copied per worker, where the structural arrays are appended once on
    // the master before it is frozen (#157). Leaving it out doubled a 34 MB array
    // on the largest MIPfeas instance.
    st.nodes.reserve(st.nodes.size() + 2);
    st.parent_offsets.reserve(st.parent_offsets.size() + 2);
    st.child_refs.reserve(st.child_refs.size() + 2);
    st.constraint_ids.reserve(st.constraint_ids.size() + 1);
    node_values_.reserve(node_values_.size() + 2);

    objective_bound_ = std::numeric_limits<double>::infinity();
    objective_bound_node_ = constant(objective_bound_);
    // obj - bound <= 0; inert while bound is +inf, tightened during search.
    objective_constraint_node_ = leq(objective_id_, objective_bound_node_);
    objective_constraint_idx_ = static_cast<int32_t>(st.constraint_ids.size());
    add_constraint(objective_constraint_node_);

    // Rebuild structure now that a node/constraint was appended after close().
    rebuild_back_references();
    st.topo_order = detail::compute_topo_order(*this);
    rebuild_topo_positions();
    build_var_constraints();
    full_evaluate(*this);
}

void Model::set_objective_bound(double bound) {
    if (objective_constraint_node_ < 0) {
        throw std::logic_error("set_objective_bound requires add_objective_soft_constraint first");
    }
    objective_bound_ = bound;
    // The shared node's `const_value` is NOT touched -- it is structure, and a
    // frozen structure is every worker's. `evaluate()` reads the bound from
    // `objective_bound()` instead; see its `Const` arm.
    node_values_[objective_bound_node_] = bound;
    // Recompute the objective constraint residual in place (obj - bound). Must
    // use the same residual rule as evaluate()'s Leq case, or this shortcut and
    // the next delta_evaluate() would disagree on the row's value — in
    // particular on the `obj = +inf, bound = +inf` state that opens every solve
    // with a blown-up objective (issue #100). The bound side is
    // objective_bound_node_, a Const by construction (see
    // add_objective_soft_constraint), so it is a sentinel; the objective side is
    // a computed expression and never is.
    node_values_[objective_constraint_node_] =
        comparison_residual(node_values_[objective_id_], bound,
                            /*a_is_const=*/false, /*b_is_const=*/true);
}

// Build var_id -> constraint-index adjacency (the paper's G_v) by walking down
// each constraint's subtree and recording every variable it reaches. Stamping
// gives O(1) per-constraint reset and dedups vars/nodes within a constraint.
//
// Into CSR, by inversion: the walk writes the constraint -> variables incidence
// flat, in constraint order, and a counting pass turns it round. Visiting the
// constraints in ascending index is what gives each variable's list ascending
// constraint indices -- the order the per-variable push_back this replaced
// produced. The incidence array is transient and costs 4 bytes per entry, where
// a second walk to count first would cost the whole DFS again.
void Model::build_var_constraints() {
    ModelStructure& st = mut();
    const auto n_cons = static_cast<int32_t>(st.constraint_ids.size());
    std::vector<int32_t> node_stamp(st.nodes.size(), -1);
    std::vector<int32_t> var_stamp(vars_.size(), -1);
    std::vector<int32_t> stack;
    // Constraint order. Sized from the last build's incidence count, which the
    // rebuild after add_objective_soft_constraint() matches to within one row, and
    // that build's CSR is released first so the two are never both alive. Its
    // offsets go with it, so a throw below leaves constraints_of_var reporting
    // out_of_range rather than reading freed ids.
    std::vector<int32_t> incident_vars;
    incident_vars.reserve(st.var_constraint_ids.size());
    std::vector<int32_t>().swap(st.var_constraint_ids);
    st.var_constraint_offsets.clear();
    std::vector<size_t> incident_begin(static_cast<size_t>(n_cons) + 1);  // per constraint
    for (int32_t ci = 0; ci < n_cons; ++ci) {
        incident_begin[ci] = incident_vars.size();
        stack.clear();
        int32_t root = st.constraint_ids[ci];
        node_stamp[root] = ci;
        stack.push_back(root);
        while (!stack.empty()) {
            int32_t nid = stack.back();
            stack.pop_back();
            for (const ChildRef& child : children(st.nodes[nid])) {
                if (child.is_var) {
                    if (var_stamp[child.id] != ci) {
                        var_stamp[child.id] = ci;
                        incident_vars.push_back(child.id);
                    }
                } else if (node_stamp[child.id] != ci) {
                    node_stamp[child.id] = ci;
                    stack.push_back(child.id);
                }
            }
        }
    }
    incident_begin[n_cons] = incident_vars.size();
    if (incident_vars.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::length_error("model has more than 2^32 - 1 variable-constraint incidences");
    }

    st.var_constraint_offsets.assign(vars_.size() + 1, 0);
    for (const int32_t v : incident_vars) {
        ++st.var_constraint_offsets[v + 1];
    }
    std::partial_sum(st.var_constraint_offsets.begin(), st.var_constraint_offsets.end(),
                     st.var_constraint_offsets.begin());
    st.var_constraint_ids.resize(incident_vars.size());
    // offsets[v] as v's write cursor, then shifted back -- as in
    // rebuild_back_references.
    for (int32_t ci = 0; ci < n_cons; ++ci) {
        for (size_t k = incident_begin[ci]; k < incident_begin[ci + 1]; ++k) {
            st.var_constraint_ids[st.var_constraint_offsets[incident_vars[k]]++] = ci;
        }
    }
    std::copy_backward(st.var_constraint_offsets.begin(), st.var_constraint_offsets.end() - 1,
                       st.var_constraint_offsets.end());
    st.var_constraint_offsets.front() = 0;
}

std::vector<std::pair<int32_t, double>> Model::per_constraint_violation_delta(int32_t var_id,
                                                                              double j) {
    const Variable& v = var(var_id);  // bounds-checked
    if (is_structured(v.type)) {
        throw std::invalid_argument(
            "per_constraint_violation_delta: scalar variable required (Bool/Int/Float)");
    }

    const ConstSpan<int32_t> affected = constraints_of_var(var_id);
    std::vector<std::pair<int32_t, double>> result;
    if (affected.empty()) {
        return result;
    }

    // Snapshot affected constraints' current violations. Read through the value
    // array rather than the range-checking `node()`, for the reason `child_val`
    // in dag.cpp gives: `affected` holds constraint INDICES from
    // `constraints_of_var`, and `constraint_ids()` holds node ids validated when
    // the constraint was added, so the check could only ever pass.
    const std::vector<int32_t>& cids = constraint_ids();
    std::vector<double> old_viol(affected.size());
    for (size_t k = 0; k < affected.size(); ++k) {
        old_viol[k] = clamped_node_violation(node_values_[cids[affected[k]]]);
    }

    // Probe: set candidate, recompute only the affected dirty cone.
    const double old_value = v.value;
    var_mut(var_id).value = j;
    delta_evaluate(*this, &var_id, 1, DeltaMode::Probe);

    for (size_t k = 0; k < affected.size(); ++k) {
        double new_viol = clamped_node_violation(node_values_[cids[affected[k]]]);
        double delta = new_viol - old_viol[k];
        if (delta != 0.0) {
            result.emplace_back(affected[k], delta);
        }
    }

    // Restore exactly: same inputs through deterministic evaluate() roll node
    // values back to where they were. `Rollback` is what makes that true for a
    // custom node too -- see the note on weighted_violation_delta below.
    var_mut(var_id).value = old_value;
    delta_evaluate(*this, &var_id, 1, DeltaMode::Rollback);

    return result;
}

double Model::weighted_violation_delta(int32_t var_id, double j,
                                       const std::vector<double>& weights) {
    const Variable& v = var(var_id);  // bounds-checked
    if (is_structured(v.type)) {
        throw std::invalid_argument(
            "weighted_violation_delta: scalar variable required (Bool/Int/Float)");
    }
    const ConstSpan<int32_t> affected = constraints_of_var(var_id);
    if (affected.empty()) {
        return 0.0;
    }

    // Accumulate the *per-constraint* differences, rather than differencing two
    // whole-sum accumulators.
    //
    // The two are equal in exact arithmetic but not in floating point, and the
    // difference is the whole ballgame once any one row's violation is large.
    // A non-convex blowup clamps to kInfPenalty = 1e30, some fourteen orders of
    // magnitude above the O(1) contributions of the real rows, so `1e30 + 1`
    // rounds back to `1e30`: both sums collapse to the same value, the
    // subtraction yields exactly 0, and every candidate jump scores identically.
    // Feasibility Jump is then blind — the search cannot tell which move reduces
    // real infeasibility (issue #100). Differencing per constraint makes the
    // huge term cancel exactly (1e30 - 1e30 == 0) and leaves the small terms at
    // full precision.
    //
    // probe_old_violation_ is a member so this stays allocation-free after
    // warm-up (one call per jump candidate). Safe for the same reason the
    // transient node mutation below is: each search thread owns its own Model.
    const std::vector<int32_t>& cids = constraint_ids();
    probe_old_violation_.resize(affected.size());
    for (size_t k = 0; k < affected.size(); ++k) {
        probe_old_violation_[k] = clamped_node_violation(node_values_[cids[affected[k]]]);
    }

    // Probe/Rollback rather than two plain deltas, for the custom nodes of #166:
    // this pair is the per-candidate hot path (one call per jump value, per
    // variable, per GLS iteration), so a stateful invariant that saw it as two
    // deltas would rebuild its cache twice per candidate and never hold a
    // committed state for longer than one call. Under the bracket it sees one
    // `delta()` and one `rollback()`, and the node's cached value is put back by
    // the engine rather than recomputed.
    //
    // NARROWED DELIBERATELY: THREE other sites score by applying and then putting
    // back, and every leg of all three stays a plain `Commit` delta -- the
    // structural batch, the inner solver, and Novelty Jump's backtracking chain
    // (`novelty_jump_search`), which is on the same per-candidate path this probe
    // is. They are correct: each leg is a real assignment and the `changed` set
    // each passes is a complete superset of what it moved. But a custom node in
    // one of their cones costs two `delta()` + two `commit()` per candidate
    // instead of one `delta()` + one `rollback()`, and an invariant caching a
    // List's prefix sums rebuilds it on both legs. Bracketing them is a separate
    // change to `src/structural_batch.cpp`, `src/inner_solver.cpp` and
    // `src/feasibility_jump.cpp`.
    const double old_value = v.value;
    var_mut(var_id).value = j;
    delta_evaluate(*this, &var_id, 1, DeltaMode::Probe);

    double delta = 0.0;
    for (size_t k = 0; k < affected.size(); ++k) {
        const int32_t c = affected[k];
        const double new_viol = clamped_node_violation(node_values_[cids[c]]);
        delta += weights[c] * (new_viol - probe_old_violation_[k]);
    }

    var_mut(var_id).value = old_value;
    delta_evaluate(*this, &var_id, 1, DeltaMode::Rollback);

    return delta;
}

Model::State Model::copy_state() const {
    State state;
    state.values.resize(vars_.size());
    state.elements.resize(vars_.size());
    for (size_t i = 0; i < vars_.size(); ++i) {
        state.values[i] = vars_[i].value;
        state.elements[i] = vars_[i].elements;
    }
    return state;
}

void Model::restore_state(const State& state) {
    // Both vectors are written from Python (ModelState is default-constructible
    // with two writable fields), and `elements` is indexed unchecked below.
    if (state.values.size() != vars_.size() || state.elements.size() != vars_.size()) {
        throw std::invalid_argument("state size does not match model");
    }
    for (size_t i = 0; i < vars_.size(); ++i) {
        vars_[i].value = state.values[i];
        vars_[i].elements = state.elements[i];
    }
}

}  // namespace cbls
