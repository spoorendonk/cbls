#include "cbls/model.h"

#include "cbls/dag_ops.h"
#include "cbls/expr.h"
#include "cbls/violation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cbls {

// Forward declare from dag_ops.cpp
namespace detail {
std::vector<int32_t> compute_topo_order(const Model& model);
}

// Both handles point at a structure allocated here, so accessors never branch on
// "not allocated"; only `freeze()` changes which handles exist.
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
      domain_margin_(other.domain_margin_),
      closed_(other.closed_),
      inc_sums_(other.inc_sums_) {
    // probe_old_violation_ is left empty: every call resizes and overwrites it
    // before reading.
    if (other.open_structure_ == nullptr) {
        // Frozen: share the DAG, so a replica costs only variables and node
        // values (#157). Safe from N threads: `other` is only read, and a
        // shared_ptr copy is atomic.
        structure_ = other.structure_;
    } else {
        // Open: deep-copy, so the copy can be built further on its own.
        auto structure = std::make_shared<ModelStructure>(*other.structure_);
        open_structure_ = structure;
        structure_ = std::move(structure);
    }
}

void Model::freeze(size_t cone_cache_bytes) {
    if (is_frozen()) {
        return;
    }
    if (!closed_) {
        close();
    }
    if (objective_id_ >= 0) {
        // Idempotent; why it is here and not in close(): freeze() in the header.
        add_objective_soft_constraint();
    }
    // After the objective row, the last structural change (#271).
    build_cone_cache(cone_cache_bytes);
    // Without the writable handle every structural method throws; only the
    // const `structure_` remains.
    open_structure_.reset();
}

void Model::require_open(const char* method) const {
    if (is_frozen()) {
        throw std::logic_error(
            std::string(method) +
            ": model is frozen (by freeze() or solve()), its structure cannot change");
    }
}

void Model::require_buildable(const char* method) const {
    require_open(method);  // a frozen model keeps its own, more specific message
    if (closed_) {
        throw std::logic_error(std::string("Model::") + method +
                               ": model is closed, and a node or row added now would never be "
                               "evaluated; a closed model cannot grow, so build the whole "
                               "model before close()");
    }
}

Model& Model::operator=(const Model& other) {
    if (this != &other) {
        Model copy(other);
        *this = std::move(copy);
    }
    return *this;
}

int32_t Model::alloc_var(VarType type, double lb, double ub, const std::string& name) {
    ModelStructure& st = mut();
    Variable v;
    v.id = static_cast<int32_t>(vars_.size());
    v.type = type;
    v.value = lb;
    v.lb = lb;
    v.ub = ub;
    v.name = name;
    // Offsets first: if the push below throws they are one entry too long
    // (an empty range), never one short.
    if (st.dependent_offsets.empty()) {
        st.dependent_offsets.push_back(0);
    }
    st.dependent_offsets.push_back(st.dependent_offsets.back());
    vars_.push_back(std::move(v));
    return vars_.back().id;
}

// Append a node over `st.child_refs[child_begin ..]`, written by the caller. The
// only place a node is made, so it keeps both flat-storage invariants: the slice
// fits ExprNode's 32-bit offsets, and parent_offsets is one longer than nodes.
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
    // Values grow with the node (both indexed by one id), and before it, as in
    // alloc_var: a throw leaves at most one unindexable extra entry.
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
    // No reserve(size + n): libstdc++ reserves exactly, so per node it would
    // defeat geometric growth and copy the whole array each call.
    const size_t begin = st.child_refs.size();
    try {
        for (const int32_t h : handles) {
            st.child_refs.push_back(wrap(h));
        }
    } catch (...) {
        // A rejected node leaves no children behind, else a catch-and-retry
        // caller grows the array without bound (orphans are otherwise inert).
        st.child_refs.resize(begin);
        throw;
    }
    return push_node(op, begin);
}

// Handle encoding: var handles = -(var_id + 1) (negative), node handles = node_id
// (non-negative).
//
// Must be validated here, at node creation: the back-reference rebuild indexes a
// CSR array by id unchecked, so a bad id is a silent heap write (#156), and
// Python passes raw integers. Also rules out naming a not-yet-made node.
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
    require_buildable("bool_var");
    int32_t vid = alloc_var(VarType::Bool, 0.0, 1.0, name);
    return -(vid + 1);  // encode as var handle
}

int32_t Model::int_var(int lb, int ub, const std::string& name) {
    require_buildable("int_var");
    int32_t vid = alloc_var(VarType::Int, static_cast<double>(lb), static_cast<double>(ub), name);
    return -(vid + 1);
}

int32_t Model::float_var(double lb, double ub, const std::string& name) {
    require_buildable("float_var");
    int32_t vid = alloc_var(VarType::Float, lb, ub, name);
    return -(vid + 1);
}

int32_t Model::list_var(int n, const std::string& name) {
    return list_var(n, n, n, ListInit::Identity, name);
}

int32_t Model::list_var(int universe, int min_len, int max_len, ListInit init,
                        const std::string& name) {
    require_buildable("list_var");
    if (universe < 0) {
        throw std::invalid_argument("list_var: negative universe");
    }
    if (min_len < 0 || min_len > max_len || max_len > universe) {
        throw std::invalid_argument("list_var: require 0 <= min_len <= max_len <= universe");
    }
    if (init == ListInit::Empty && min_len > 0) {
        // Else the List starts (and `randomize_var(Regenerate)` returns it)
        // below min_len, which no row prices and no move guard reports.
        // `ListInit::Random` respects a minimum.
        throw std::invalid_argument("list_var: ListInit::Empty requires min_len == 0");
    }
    if (init == ListInit::Identity && (min_len != universe || max_len != universe)) {
        // Identity is "every element, in order": legal only at length ==
        // universe, which `randomize_structured_var` relies on (#164).
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
        // Empty and Random start empty; `initialize_structured_random` fills a
        // Random list (and any Exact-partition member) before the search reads
        // it. Without that call an evaluation sees an empty list.
        v.elements.clear();
    }
    return -(vid + 1);
}

int32_t Model::set_var(int n, int min_size, int max_size, const std::string& name) {
    require_buildable("set_var");
    int32_t vid = alloc_var(VarType::Set, 0.0, 0.0, name);
    auto& v = vars_[vid];
    v.universe_size = n;
    v.min_size = min_size;
    v.max_size = (max_size < 0) ? n : max_size;
    return -(vid + 1);
}

// Expression creation methods return non-negative handles (node IDs)
int32_t Model::constant(double val) {
    require_buildable("constant");
    return push_constant(val);
}

int32_t Model::push_constant(double val) {
    ModelStructure& st = mut();
    const int32_t nid = push_node(NodeOp::Const, st.child_refs.size());
    st.nodes[nid].const_value = val;
    node_values_[nid] = val;
    return nid;
}

int32_t Model::neg(int32_t x) {
    require_buildable("neg");
    return alloc_node(NodeOp::Neg, {wrap(x)});
}

int32_t Model::sum(const std::vector<int32_t>& args) {
    require_buildable("sum");
    if (args.empty()) {
        return constant(0.0);
    }
    return alloc_node_over_handles(NodeOp::Sum, args);
}

int32_t Model::prod(int32_t a, int32_t b) {
    require_buildable("prod");
    return alloc_node(NodeOp::Prod, {wrap(a), wrap(b)});
}

int32_t Model::div_expr(int32_t a, int32_t b) {
    require_buildable("div_expr");
    return alloc_node(NodeOp::Div, {wrap(a), wrap(b)});
}

int32_t Model::pow_expr(int32_t base, int32_t exp) {
    require_buildable("pow_expr");
    return alloc_node(NodeOp::Pow, {wrap(base), wrap(exp)});
}

// Min and Max evaluate children[0] unchecked, so an empty one would read
// another node's child slice (see child_val in dag.cpp).
int32_t Model::min_expr(const std::vector<int32_t>& args) {
    require_buildable("min_expr");
    if (args.empty()) {
        throw std::invalid_argument("min_expr requires at least one argument");
    }
    return alloc_node_over_handles(NodeOp::Min, args);
}

int32_t Model::max_expr(const std::vector<int32_t>& args) {
    require_buildable("max_expr");
    if (args.empty()) {
        throw std::invalid_argument("max_expr requires at least one argument");
    }
    return alloc_node_over_handles(NodeOp::Max, args);
}

int32_t Model::abs_expr(int32_t x) {
    require_buildable("abs_expr");
    return alloc_node(NodeOp::Abs, {wrap(x)});
}

int32_t Model::sin_expr(int32_t x) {
    require_buildable("sin_expr");
    return alloc_node(NodeOp::Sin, {wrap(x)});
}

int32_t Model::cos_expr(int32_t x) {
    require_buildable("cos_expr");
    return alloc_node(NodeOp::Cos, {wrap(x)});
}

int32_t Model::tan_expr(int32_t x) {
    require_buildable("tan_expr");
    return alloc_node(NodeOp::Tan, {wrap(x)});
}

int32_t Model::exp_expr(int32_t x) {
    require_buildable("exp_expr");
    return alloc_node(NodeOp::Exp, {wrap(x)});
}

int32_t Model::log_expr(int32_t x) {
    require_buildable("log_expr");
    return alloc_node(NodeOp::Log, {wrap(x)});
}

int32_t Model::sqrt_expr(int32_t x) {
    require_buildable("sqrt_expr");
    return alloc_node(NodeOp::Sqrt, {wrap(x)});
}

int32_t Model::signpower_expr(int32_t base, int32_t exp) {
    require_buildable("signpower_expr");
    return alloc_node(NodeOp::SignPower, {wrap(base), wrap(exp)});
}

int32_t Model::tanh_expr(int32_t x) {
    require_buildable("tanh_expr");
    return alloc_node(NodeOp::Tanh, {wrap(x)});
}

int32_t Model::if_then_else(int32_t cond, int32_t then_, int32_t else_) {
    require_buildable("if_then_else");
    return alloc_node(NodeOp::If, {wrap(cond), wrap(then_), wrap(else_)});
}

int32_t Model::at(int32_t list_var_id, int32_t index_expr) {
    require_buildable("at");
    return alloc_node(NodeOp::At, {wrap(list_var_id), wrap(index_expr)});
}

int32_t Model::count(int32_t var_id) {
    require_buildable("count");
    return alloc_node(NodeOp::Count, {wrap(var_id)});
}

int32_t Model::leq(int32_t a, int32_t b) {
    require_buildable("leq");
    return alloc_node(NodeOp::Leq, {wrap(a), wrap(b)});
}

int32_t Model::eq_expr(int32_t a, int32_t b) {
    require_buildable("eq_expr");
    return alloc_node(NodeOp::Eq, {wrap(a), wrap(b)});
}

int32_t Model::geq(int32_t a, int32_t b) {
    require_buildable("geq");
    return alloc_node(NodeOp::Geq, {wrap(a), wrap(b)});
}

int32_t Model::neq(int32_t a, int32_t b) {
    require_buildable("neq");
    return alloc_node(NodeOp::Neq, {wrap(a), wrap(b)});
}

int32_t Model::lt(int32_t a, int32_t b) {
    require_buildable("lt");
    const int32_t id = alloc_node(NodeOp::Lt, {wrap(a), wrap(b)});
    mut().nodes[id].const_value = strict_comparison_margin(is_integral(a) && is_integral(b));
    return id;
}

int32_t Model::gt(int32_t a, int32_t b) {
    require_buildable("gt");
    const int32_t id = alloc_node(NodeOp::Gt, {wrap(a), wrap(b)});
    mut().nodes[id].const_value = strict_comparison_margin(is_integral(a) && is_integral(b));
    return id;
}

namespace {

bool integral_var_type(VarType t) {
    return t == VarType::Bool || t == VarType::Int;
}

// What decides whether one node is integral: either the node alone (`leaf`,
// with its answer in `value`), or its children in [first, last), all of which
// must be.
struct IntegralRule {
    bool leaf = true;
    bool value = false;
    size_t first = 0;
    size_t last = 0;
};

IntegralRule integral_rule(const Model& model, const ExprNode& nd, ConstSpan<ChildRef> kids) {
    IntegralRule r;
    switch (nd.op) {
        case NodeOp::Const:
            r.value = std::isfinite(nd.const_value) && nd.const_value == std::trunc(nd.const_value);
            break;
        case NodeOp::Ceil:
        case NodeOp::Floor:
        case NodeOp::Round:
        case NodeOp::Count:
        case NodeOp::Neq:
            r.value = true;
            break;
        case NodeOp::Pow: {
            // The base decides, under a non-negative integer constant exponent.
            const ChildRef e = kids[1];
            const ExprNode* en = e.is_var ? nullptr : &model.node(e.id);
            if (en != nullptr && en->op == NodeOp::Const && en->const_value >= 0.0 &&
                en->const_value == std::trunc(en->const_value)) {
                r = IntegralRule{false, false, 0, 1};
            }
            break;
        }
        case NodeOp::If:
            r = IntegralRule{false, false, 1, kids.size()};  // the branches, not the condition
            break;
        case NodeOp::Neg:
        case NodeOp::Sum:
        case NodeOp::Prod:
        case NodeOp::Min:
        case NodeOp::Max:
        case NodeOp::Abs:
            r = IntegralRule{false, false, 0, kids.size()};
            break;
        // Not known integral. Listed, not `default:`, so a new op is a -Wswitch
        // finding.
        case NodeOp::Div:
        case NodeOp::Sin:
        case NodeOp::Cos:
        case NodeOp::Tan:
        case NodeOp::Exp:
        case NodeOp::Log:
        case NodeOp::Sqrt:
        case NodeOp::SignPower:
        case NodeOp::Tanh:
        case NodeOp::At:
        case NodeOp::Lambda:
        case NodeOp::PairLambda:
        case NodeOp::Leq:
        case NodeOp::Eq:
        case NodeOp::Geq:
        case NodeOp::Lt:
        case NodeOp::Gt:
        case NodeOp::Element:
        case NodeOp::LambdaExtra:
        case NodeOp::PairLambdaExtra:
            break;
    }
    return r;
}

}  // namespace

// Iterative, memoised per call: without the memo a shared subexpression is
// revisited once per path to it.
bool Model::is_integral(int32_t handle) const {
    if (handle < 0) {
        return integral_var_type(var(-(handle + 1)).type);
    }
    std::unordered_map<int32_t, bool> known;
    std::vector<int32_t> stack{handle};
    while (!stack.empty()) {
        const int32_t id = stack.back();
        if (known.count(id) != 0) {
            stack.pop_back();
            continue;
        }
        const ExprNode& nd = node(id);
        const ConstSpan<ChildRef> kids = children(nd);
        const IntegralRule rule = integral_rule(*this, nd, kids);
        bool all = rule.value;
        if (!rule.leaf) {
            all = true;
            bool pending = false;
            for (size_t k = rule.first; k < rule.last; ++k) {
                const ChildRef c = kids[k];
                if (c.is_var) {
                    all = all && integral_var_type(var(c.id).type);
                } else if (const auto it = known.find(c.id); it != known.end()) {
                    all = all && it->second;
                } else {
                    stack.push_back(c.id);
                    pending = true;
                }
            }
            if (pending) {
                continue;  // revisit once the children are known
            }
        }
        known.emplace(id, all);
        stack.pop_back();
    }
    return known.at(handle);
}

int32_t Model::lambda_sum(int32_t list_var_id, std::function<double(int)> func) {
    require_buildable("lambda_sum");
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
    require_buildable("pair_lambda_sum");
    ModelStructure& st = mut();
    const ChildRef child = wrap(list_var_id);  // reject a bad handle before registering

    // An empty callable is "no endpoint term", recorded as -1; registering it
    // would throw std::bad_function_call at evaluation.
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

    // Both tables share one id, so must stay the same length: reserving first
    // means the second push cannot throw once the first has happened.
    st.pair_lambda_specs.reserve(st.pair_lambda_funcs.size() + 1);
    st.pair_lambda_funcs.push_back(std::move(func));
    st.pair_lambda_specs.push_back(spec);
    auto func_id = static_cast<int32_t>(st.pair_lambda_funcs.size() - 1);

    int32_t nid = alloc_node(NodeOp::PairLambda, {child});
    st.nodes[nid].lambda_func_id = func_id;
    return nid;
}

// ---------------------------------------------------------------------------
// #186: Element, Ceil/Floor/Round, and the lambdas that read other decisions.
// ---------------------------------------------------------------------------

// A handle read as a SCALAR, validated by `wrap` before anything is registered.
// A List or Set is refused: its `value` is a constant 0.0, so the input would
// silently freeze.
ChildRef Model::wrap_scalar(int32_t handle, const char* what) const {
    const ChildRef ref = wrap(handle);
    if (ref.is_var && is_structured(vars_[ref.id].type)) {
        throw std::invalid_argument(std::string(what) +
                                    ": expected a scalar handle, got a List or Set variable");
    }
    return ref;
}

namespace {

// Representative indices per Element-table line (#186): one per distinct value
// when at most this many, else this many at evenly spaced ranks of the sorted
// distinct values, smallest and largest included. They are the jump candidates
// `compute_var_jump` offers for an index too large to enumerate. 32 matches
// `kJumpGridPoints` (src/feasibility_jump.cpp), the Int grid's probe budget.
constexpr size_t kElementRepresentatives = 32;

// The representatives of one line (`line[i]` is the value at index i). One sort
// of (value, index) pairs puts each distinct value's first index at the head of
// its run: O(m log m) for a line of m, no search per value.
std::vector<int32_t> line_representatives(const std::vector<double>& line) {
    std::vector<std::pair<double, int32_t>> cells;
    cells.reserve(line.size());
    for (size_t i = 0; i < line.size(); ++i) {
        cells.emplace_back(line[i], static_cast<int32_t>(i));
    }
    std::sort(cells.begin(), cells.end());
    std::vector<int32_t> first_of_value;  // in ascending value
    for (size_t i = 0; i < cells.size(); ++i) {
        if (i == 0 || cells[i].first != cells[i - 1].first) {
            first_of_value.push_back(cells[i].second);
        }
    }
    const size_t d = first_of_value.size();
    if (d <= kElementRepresentatives) {
        return first_of_value;
    }
    std::vector<int32_t> reps;
    reps.reserve(kElementRepresentatives);
    for (size_t r = 0; r < kElementRepresentatives; ++r) {
        const auto rank =
            static_cast<size_t>(std::llround(static_cast<double>(r) * static_cast<double>(d - 1) /
                                             static_cast<double>(kElementRepresentatives - 1)));
        reps.push_back(first_of_value[rank]);
    }
    return reps;
}

// Fill `tbl.row_reps_by_col` and, for a two-index table, `tbl.col_reps_by_row`.
// Once, at build: O(R C (log R + log C)) for R x C, at most 32 (R + C) indices
// stored, so a call can offer the line the OTHER index currently selects.
void compute_representatives(ElementTable& tbl, bool two_index) {
    const auto rows = static_cast<size_t>(tbl.rows);
    const auto cols = static_cast<size_t>(tbl.cols);
    std::vector<double> line;
    tbl.row_reps_by_col.resize(cols);
    for (size_t c = 0; c < cols; ++c) {
        line.clear();
        for (size_t r = 0; r < rows; ++r) {
            line.push_back(tbl.values[(r * cols) + c]);
        }
        tbl.row_reps_by_col[c] = line_representatives(line);
    }
    if (!two_index) {
        return;
    }
    tbl.col_reps_by_row.resize(rows);
    for (size_t r = 0; r < rows; ++r) {
        line.assign(tbl.values.begin() + static_cast<std::ptrdiff_t>(r * cols),
                    tbl.values.begin() + static_cast<std::ptrdiff_t>((r + 1) * cols));
        tbl.col_reps_by_row[r] = line_representatives(line);
    }
}

void require_finite_table(const std::vector<double>& values) {
    // A non-finite entry makes every delta through it inf - inf somewhere, and
    // `.cbls` writes it as null, which the reader then refuses.
    if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })) {
        throw std::invalid_argument("element: table entries must be finite");
    }
}

}  // namespace

// Register `tbl`, then make the Element node over `kids`: the node never names a
// missing table, and a failed node pops the table, leaving no orphan.
int32_t Model::push_element(ElementTable tbl, const std::vector<ChildRef>& kids) {
    ModelStructure& st = mut();
    require_finite_table(tbl.values);
    compute_representatives(tbl, kids.size() == 2);
    st.element_tables.push_back(std::move(tbl));
    const auto table_id = static_cast<int32_t>(st.element_tables.size() - 1);
    int32_t nid = -1;
    try {
        nid = alloc_node_over_refs(NodeOp::Element, kids);
    } catch (...) {
        st.element_tables.pop_back();
        throw;
    }
    st.nodes[nid].lambda_func_id = table_id;
    st.has_breakpoint_nodes = true;
    return nid;
}

int32_t Model::element(const std::vector<double>& table, int32_t index) {
    require_buildable("element");
    if (table.empty()) {
        throw std::invalid_argument("element: table must not be empty");
    }
    if (table.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::invalid_argument("element: table too large");
    }
    const ChildRef idx = wrap_scalar(index, "element");
    ElementTable tbl;
    tbl.rows = static_cast<int32_t>(table.size());
    tbl.cols = 1;
    tbl.values = table;
    return push_element(std::move(tbl), {idx});
}

int32_t Model::element(const std::vector<std::vector<double>>& table, int32_t row, int32_t col) {
    require_buildable("element");
    if (table.empty() || table.front().empty()) {
        throw std::invalid_argument("element: table must be non-empty in both dimensions");
    }
    const size_t cols = table.front().size();
    for (const auto& r : table) {
        if (r.size() != cols) {
            throw std::invalid_argument("element: table must be rectangular");
        }
    }
    constexpr auto kMax = static_cast<size_t>(std::numeric_limits<int32_t>::max());
    if (table.size() > kMax || cols > kMax / table.size()) {
        throw std::invalid_argument("element: table too large");
    }
    const ChildRef r = wrap_scalar(row, "element");
    const ChildRef c = wrap_scalar(col, "element");
    ElementTable tbl;
    tbl.rows = static_cast<int32_t>(table.size());
    tbl.cols = static_cast<int32_t>(cols);
    tbl.values.reserve(table.size() * cols);
    for (const auto& rw : table) {
        tbl.values.insert(tbl.values.end(), rw.begin(), rw.end());
    }
    return push_element(std::move(tbl), {r, c});
}

int32_t Model::ceil_expr(int32_t x) {
    require_buildable("ceil_expr");
    const int32_t nid = alloc_node(NodeOp::Ceil, {wrap_scalar(x, "ceil_expr")});
    mut().has_breakpoint_nodes = true;
    return nid;
}

int32_t Model::floor_expr(int32_t x) {
    require_buildable("floor_expr");
    const int32_t nid = alloc_node(NodeOp::Floor, {wrap_scalar(x, "floor_expr")});
    mut().has_breakpoint_nodes = true;
    return nid;
}

int32_t Model::round_expr(int32_t x) {
    require_buildable("round_expr");
    const int32_t nid = alloc_node(NodeOp::Round, {wrap_scalar(x, "round_expr")});
    mut().has_breakpoint_nodes = true;
    return nid;
}

// The children of an extra-lambda node: the List/Set, then every extra. Built in
// full -- every handle validated -- before the caller registers anything.
std::vector<ChildRef> Model::lambda_extra_children(int32_t list_var_id,
                                                   const std::vector<int32_t>& extra,
                                                   const char* what) const {
    const ChildRef list = wrap(list_var_id);
    if (!list.is_var || !is_structured(vars_[list.id].type)) {
        throw std::invalid_argument(std::string(what) + ": expected a List or Set variable handle");
    }
    std::vector<ChildRef> kids;
    kids.reserve(extra.size() + 1);
    kids.push_back(list);
    for (const int32_t h : extra) {
        kids.push_back(wrap_scalar(h, what));
    }
    return kids;
}

int32_t Model::alloc_node_over_refs(NodeOp op, const std::vector<ChildRef>& kids) {
    ModelStructure& st = mut();
    const size_t begin = st.child_refs.size();
    st.child_refs.insert(st.child_refs.end(), kids.begin(), kids.end());
    return push_node(op, begin);
}

int32_t Model::lambda_sum(int32_t list_var_id, LambdaExtraFunc func,
                          const std::vector<int32_t>& extra) {
    require_buildable("lambda_sum");
    ModelStructure& st = mut();
    if (!func) {
        throw std::invalid_argument("lambda_sum: func must not be empty");
    }
    const std::vector<ChildRef> kids = lambda_extra_children(list_var_id, extra, "lambda_sum");
    // Registered before the node, and popped if the node cannot be made, as
    // `push_element` does.
    st.lambda_extra_funcs.push_back(std::move(func));
    const auto func_id = static_cast<int32_t>(st.lambda_extra_funcs.size() - 1);
    int32_t nid = -1;
    try {
        nid = alloc_node_over_refs(NodeOp::LambdaExtra, kids);
    } catch (...) {
        st.lambda_extra_funcs.pop_back();
        throw;
    }
    st.nodes[nid].lambda_func_id = func_id;
    return nid;
}

int32_t Model::pair_lambda_sum(int32_t list_var_id, PairLambdaExtraFunc func, PairMode mode,
                               const std::vector<int32_t>& extra) {
    require_buildable("pair_lambda_sum");
    ModelStructure& st = mut();
    if (!func) {
        throw std::invalid_argument("pair_lambda_sum: func must not be empty");
    }
    const std::vector<ChildRef> kids = lambda_extra_children(list_var_id, extra, "pair_lambda_sum");
    // Reserved first, as in the plain form: the second push must not throw.
    st.pair_lambda_extra_modes.reserve(st.pair_lambda_extra_funcs.size() + 1);
    st.pair_lambda_extra_funcs.push_back(std::move(func));
    st.pair_lambda_extra_modes.push_back(mode);
    const auto func_id = static_cast<int32_t>(st.pair_lambda_extra_funcs.size() - 1);
    int32_t nid = -1;
    try {
        nid = alloc_node_over_refs(NodeOp::PairLambdaExtra, kids);
    } catch (...) {
        st.pair_lambda_extra_funcs.pop_back();
        st.pair_lambda_extra_modes.pop_back();
        throw;
    }
    st.nodes[nid].lambda_func_id = func_id;
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
    require_buildable("add_constraint");
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
    require_buildable("minimize");
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
    require_buildable("maximize");
    // Maximize by negating
    objective_id_ = neg(expr_id);
    is_maximizing_ = true;
}

void Model::add_var_sequence(const std::vector<int32_t>& var_ids, int min_block_on,
                             int min_block_off) {
    require_buildable("add_var_sequence");
    ModelStructure& st = mut();
    int seq_idx = static_cast<int>(st.var_sequences.size());
    VarSequence seq;
    seq.min_block_on = min_block_on;
    seq.min_block_off = min_block_off;

    // Takes var handles, decoded to var ids. A non-negative value passes
    // through unchanged, but that is not a supported convention (elsewhere it
    // means a node id); don't document it as one.
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

// Writes both the partition (ModelStructure) and `Variable::partitioned`
// (per-model), so no member lacks the flag that stops `list_moves` from
// proposing a move that breaks the partition.
int32_t Model::add_list_partition(const std::vector<int32_t>& lists, Cover cover) {
    require_buildable("add_list_partition");
    if (lists.empty()) {
        throw std::invalid_argument("add_list_partition: no lists");
    }
    ListPartition part;
    part.cover = cover;
    part.list_ids.reserve(lists.size());
    int64_t min_total = 0;
    int64_t max_total = 0;
    for (int32_t handle : lists) {
        // Var handles; a non-negative pass-through is not a convention (see
        // `add_var_sequence`). The `.cbls` loader, whose name table yields node
        // ids, must refuse a non-negative handle before reaching here.
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

// The inverse of `st.topo_order`, rebuilt wherever that is, by both call sites:
// a stale `topo_pos` evaluates a dirty set out of order, silently wrong.
void Model::rebuild_topo_positions() {
    ModelStructure& st = mut();
    st.topo_pos.assign(st.nodes.size(), 0);
    for (size_t i = 0; i < st.topo_order.size(); ++i) {
        st.topo_pos[st.topo_order[i]] = static_cast<int32_t>(i);
    }
    // Same order, same two sites (close() and the objective row), so it cannot
    // go stale either.
    classify_breakpoint_reach();
}

namespace {

// Collects `var_id`'s cone into `cone`, unsorted, as `collect_dirty_cone` does:
// dependents, then every ancestor, each once (`mark[n] == var_id` once taken).
// Returns false once it exceeds `room`, so an uncachable cone costs at most
// `room` nodes plus one node's parents.
bool collect_cone_within(const Model& model, int32_t var_id, size_t room,
                         std::vector<int32_t>& mark, std::vector<int32_t>& cone) {
    cone.clear();
    for (const int32_t dep : model.dependents(var_id)) {
        if (mark[dep] != var_id) {
            mark[dep] = var_id;
            cone.push_back(dep);
        }
    }
    if (cone.size() > room) {
        return false;
    }
    for (size_t i = 0; i < cone.size(); ++i) {
        for (const int32_t parent : model.parents(cone[i])) {
            if (mark[parent] != var_id) {
                mark[parent] = var_id;
                cone.push_back(parent);
            }
        }
        if (cone.size() > room) {
            return false;
        }
    }
    return true;
}

}  // namespace

// `ModelStructure::cone_offsets`/`cone_ids` (#271): per variable, the nodes a
// one-variable `delta_evaluate` recomputes (dependents and every ancestor, each
// once), sorted by `topo_pos`. Wins time: O(sum |cone| log |cone|) once instead
// of a collect-and-sort per probe and commit; `topo_pos` is a total order, so
// values are bit-identical to the walk's.
//
// Loses memory where cones are large and many (O(nodes * vars) on a deep shared
// chain), hence the cap, bounding build and cache:
//  - Variables in id order up to the first whose cone would cross the cap;
//    trying later ones could cost O(vars * cap).
//  - Pass 1 sizes that prefix, pass 2 collects into an exactly reserved array;
//    growing then shrinking would hold up to three times the cap.
// Build: O(cap) collection plus the sort; peak: the cache, one cone and 4 bytes
// per node of scratch.
void Model::build_cone_cache(size_t cap_bytes) {
    ModelStructure& st = mut();
    // Reset first, so a build that throws leaves no cached prefix behind.
    st.cone_prefix = 0;
    st.cone_offsets.assign(1, 0);
    st.cone_ids = {};
    // The offsets are 32-bit, as the back-references' are.
    const size_t cap_entries =
        std::min(cap_bytes / sizeof(int32_t), size_t{std::numeric_limits<uint32_t>::max()});
    if (cap_entries == 0) {
        return;  // not even a variable with an empty cone: 0 caches nothing
    }
    const size_t num_vars = vars_.size();
    std::vector<int32_t> mark(st.nodes.size(), -1);
    std::vector<int32_t> cone;
    size_t total = 0;
    size_t prefix = 0;
    while (prefix < num_vars && collect_cone_within(*this, static_cast<int32_t>(prefix),
                                                    cap_entries - total, mark, cone)) {
        total += cone.size();
        ++prefix;
    }
    // Pass 1's marks name the same variables pass 2's do.
    std::fill(mark.begin(), mark.end(), -1);
    st.cone_ids.reserve(total);
    st.cone_offsets.reserve(prefix + 1);
    for (size_t v = 0; v < prefix; ++v) {
        (void)collect_cone_within(*this, static_cast<int32_t>(v), total, mark, cone);
        std::sort(cone.begin(), cone.end(),
                  [&st](int32_t a, int32_t b) { return st.topo_pos[a] < st.topo_pos[b]; });
        st.cone_ids.insert(st.cone_ids.end(), cone.begin(), cone.end());
        st.cone_offsets.push_back(static_cast<uint32_t>(st.cone_ids.size()));
    }
    st.cone_prefix = prefix;
}

// `ModelStructure::breakpoint_flags` (#186). On a model with a breakpoint node,
// one reverse-topological pass: a node reaches a breakpoint op if it is one, or
// carries a slope and a parent reaches one. Then on every Sum a last-writer
// stamp flags a repeated child, which FJ's forward partials read on any model
// (#249). O(nodes + parent edges + children of Sums).
void Model::classify_breakpoint_reach() {
    ModelStructure& st = mut();
    std::vector<uint8_t>& flags = st.breakpoint_flags;
    flags.assign(st.nodes.size(), 0);
    for (auto it = st.topo_order.rbegin(); st.has_breakpoint_nodes && it != st.topo_order.rend();
         ++it) {
        const int32_t nid = *it;
        const NodeOp op = st.nodes[static_cast<size_t>(nid)].op;
        bool reaches = is_breakpoint_op(op);
        if (!reaches && carries_slope(op)) {
            const ConstSpan<int32_t> ps = parents(nid);
            reaches = std::any_of(ps.begin(), ps.end(), [&flags](int32_t p) {
                return (flags[static_cast<size_t>(p)] & ModelStructure::kReachesBreakpoint) != 0;
            });
        }
        if (reaches) {
            flags[static_cast<size_t>(nid)] = ModelStructure::kReachesBreakpoint;
        }
    }
    std::vector<int32_t> var_stamp(vars_.size(), -1);
    std::vector<int32_t> node_stamp(st.nodes.size(), -1);
    for (size_t nid = 0; nid < st.nodes.size(); ++nid) {
        const ExprNode& nd = st.nodes[nid];
        if (nd.op != NodeOp::Sum) {
            continue;
        }
        for (const ChildRef& c : children(nd)) {
            int32_t& stamp = c.is_var ? var_stamp[static_cast<size_t>(c.id)]
                                      : node_stamp[static_cast<size_t>(c.id)];
            if (stamp == static_cast<int32_t>(nid)) {
                flags[nid] |= ModelStructure::kRepeatedChild;
                break;
            }
            stamp = static_cast<int32_t>(nid);
        }
    }
}

// Rebuild the back-references (node parents, variable dependents) that
// delta_evaluate and the topological sort walk. Derived state: recomputed
// wholesale, not patched.
//
// Two CSR passes, count then fill, so arrays are sized once with no per-node
// allocation. Lists are in ascending parent id, each parent's children in order.
//
// Deduplicated by a last-writer stamp: a parent is visited once, so a duplicate
// only comes from one parent naming a child twice (`prod(x, x)`). O(1) per edge;
// searching the list is O(sum of degree^2), the bulk of a wide matrix's build
// (square47). Both passes need fresh stamps, as both skip the same duplicates.
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
    // reserve() allocates exactly where a growing resize() may double; clear()
    // first so reserve() does not copy stale contents pass 2 overwrites anyway.
    st.parent_ids.clear();
    st.parent_ids.reserve(st.parent_offsets.back());
    st.parent_ids.resize(st.parent_offsets.back());
    st.dependent_ids.clear();
    st.dependent_ids.reserve(st.dependent_offsets.back());
    st.dependent_ids.resize(st.dependent_offsets.back());

    // Pass 2: fill. offsets[i] is i's write cursor and ends at i's end, i.e.
    // offsets[i + 1]; one shift right restores the starts.
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

// Which Sums `commit_scalar_move` moves by their terms' changes (#177, #188).
//
// A Sum qualifies when:
//
//  - it has a term, no term is a Sum, none is named twice. A Sum term would
//    have to push its change further up; a repeated term is one
//    back-reference, so its change would count once.
//  - no term is the objective bound: a Const that `set_objective_bound`
//    rewrites without a walk, so it would change without pushing.
//  - it is read, and only by top-level comparisons (`Leq`/`Geq`/`Lt`/`Gt`/`Eq`
//    with no parent) whose other side is a variable or Const: every MPS row
//    and the objective row. A row reading the Sum directly is off by at most
//    the drift bound, and an undecided verdict re-sums that one Sum, so
//    FeasibilityJump sees all drift. A Sum under any other node keeps the
//    plain re-sum.
//  - it is not itself a constraint root (#192). Being a row is no parent edge,
//    and a row's slot is found through its children (here the terms), so its
//    drift would never be gated or re-grounded.
//    The objective node is NOT refused: it is no row, the rows over it read
//    it directly and so are gated, and refusing it would put every MPS
//    objective row back on the re-sum. This requires that the objective is read RAW only
//    between batches: every such reader (incumbent and bound tightening,
//    LNS's state key, inner solver, pool) runs after the GLS loop, whose end
//    re-grounds drifted Sums (reground_drifted_rows), and a throw out of a
//    batch is followed by solve()'s full_evaluate. A raw reader inside a
//    batch would break this. The inner solver is its own batch (#207): its
//    commits are incremental, its probes difference two readings over the
//    same drift, and it re-grounds every drifted Sum before its
//    multi-variable pass and before it returns.
//
// Terms may be anything else: every node term is flagged `kFeedsIncSum`, and a
// dirty one pushes its change into the Sum. Whether an update is exact is a
// property of the numbers, decided per update (see src/dag_ops.cpp).
namespace {

// The terms rule: at least one term, none a Sum or the objective bound, none
// named twice. `node_stamp`/`var_stamp` hold the last Sum that named each
// node/variable, so the repeat test is O(1) per term across the whole pass.
bool terms_qualify(const Model& m, int32_t nid, std::vector<int32_t>& node_stamp,
                   std::vector<int32_t>& var_stamp) {
    const ConstSpan<ChildRef> kids = m.children(m.nodes()[nid]);
    if (kids.empty()) {
        return false;
    }
    for (const ChildRef& ref : kids) {
        int32_t& stamp = ref.is_var ? var_stamp[ref.id] : node_stamp[ref.id];
        if (stamp == nid) {
            return false;  // named twice
        }
        stamp = nid;
        if (!ref.is_var &&
            (m.nodes()[ref.id].op == NodeOp::Sum || ref.id == m.objective_bound_node())) {
            return false;
        }
    }
    return true;
}

// The readers rule: read, and only by top-level comparisons against a
// variable or a Const.
bool readers_qualify(const Model& m, int32_t nid) {
    const ConstSpan<int32_t> readers = m.parents(nid);
    if (readers.empty()) {
        return false;
    }
    return std::all_of(readers.begin(), readers.end(), [&m, nid](int32_t p) {
        const ExprNode& cmp = m.nodes()[p];
        if (!is_comparison_op(cmp.op) || !m.parents(p).empty()) {
            return false;
        }
        const ConstSpan<ChildRef> sides = m.children(cmp);
        const ChildRef& other = (!sides[0].is_var && sides[0].id == nid) ? sides[1] : sides[0];
        return other.is_var || m.nodes()[other.id].op == NodeOp::Const;
    });
}

}  // namespace

// Not monotone (a Sum that gains another kind of reader drops out), so every
// flag and slot is recomputed from scratch. Both callers, close() and
// add_objective_soft_constraint, must follow it with the full_evaluate that
// sizes per-model state to the new slots.
void Model::classify_incremental_sums() {
    ModelStructure& st = mut();
    std::vector<ExprNode>& nodes = st.nodes;
    for (ExprNode& nd : nodes) {
        if ((nd.inc_sum_flags & ExprNode::kIncSum) != 0) {
            nd.lambda_func_id = -1;
        }
        nd.inc_sum_flags = 0;
    }
    st.inc_sum_nodes.clear();
    std::vector<int32_t> node_stamp(nodes.size(), -1);
    std::vector<int32_t> var_stamp(vars_.size(), -1);
    std::vector<uint8_t> is_row(nodes.size(), 0);
    for (const int32_t cid : st.constraint_ids) {
        is_row[static_cast<size_t>(cid)] = 1;
    }
    for (const int32_t nid : st.topo_order) {
        if (nodes[nid].op != NodeOp::Sum || is_row[static_cast<size_t>(nid)] != 0 ||
            !terms_qualify(*this, nid, node_stamp, var_stamp) || !readers_qualify(*this, nid)) {
            continue;
        }
        nodes[nid].inc_sum_flags = ExprNode::kIncSum;
        nodes[nid].lambda_func_id = static_cast<int32_t>(st.inc_sum_nodes.size());
        st.inc_sum_nodes.push_back(nid);
        for (const ChildRef& ref : children(nodes[nid])) {
            if (!ref.is_var) {
                nodes[ref.id].inc_sum_flags |= ExprNode::kFeedsIncSum;
            }
        }
    }
}

namespace {

// The floor of `nd`'s implicit domain row, or NaN if none: kLogDomainFloor for
// `Log`, 0 for `Sqrt` and for `Pow` with a finite non-integer Const exponent,
// the ops that are NaN outside their domain (#205). A negative such exponent
// also has a pole (+inf, as flat as NaN) at 0, so takes the `Log` floor. A
// variable exponent gets none: no constant decides a negative base's domain.
double domain_floor(const Model& m, const ExprNode& nd) {
    constexpr double kNone = std::numeric_limits<double>::quiet_NaN();
    if (nd.op == NodeOp::Log) {
        return kLogDomainFloor;
    }
    if (nd.op == NodeOp::Sqrt) {
        return 0.0;
    }
    if (nd.op != NodeOp::Pow) {
        return kNone;
    }
    const ChildRef& e = m.children(nd)[1];
    if (e.is_var || m.nodes()[e.id].op != NodeOp::Const) {
        return kNone;
    }
    const double p = m.nodes()[e.id].const_value;
    if (!std::isfinite(p) || std::floor(p) == p) {
        return kNone;
    }
    return p < 0.0 ? kLogDomainFloor : 0.0;
}

// 1 for nodes the rows or objective read, else 0. For domain rows an `If` reads
// only its condition (it evaluates one branch). Children precede parents
// (`wrap` refuses forward handles), so one sweep down the ids suffices.
std::vector<uint8_t> nodes_read(const Model& m) {
    std::vector<uint8_t> reached(m.num_nodes(), 0);
    for (const int32_t cid : m.constraint_ids()) {
        reached[static_cast<size_t>(cid)] = 1;
    }
    if (m.objective_id() >= 0) {
        reached[static_cast<size_t>(m.objective_id())] = 1;
    }
    for (size_t i = reached.size(); i-- > 0;) {
        if (reached[i] == 0) {
            continue;
        }
        const ExprNode& nd = m.nodes()[i];
        const ConstSpan<ChildRef> kids = m.children(nd);
        const size_t n_kids = nd.op == NodeOp::If ? 1 : kids.size();
        for (size_t k = 0; k < n_kids; ++k) {
            if (!kids[k].is_var) {
                reached[static_cast<size_t>(kids[k].id)] = 1;
            }
        }
    }
    return reached;
}

struct DomainRow {
    ChildRef arg;
    double floor;
};

// The domain rows the read nodes need, one per (argument, floor), leaving out
// a Const argument and a variable whose lower bound already meets the floor.
std::vector<DomainRow> domain_rows_needed(const Model& m) {
    const std::vector<uint8_t> reached = nodes_read(m);
    std::vector<DomainRow> rows;
    // Bit 1: a row at kLogDomainFloor exists for this argument; bit 2: at 0.
    std::vector<uint8_t> node_seen(m.num_nodes(), 0);
    std::vector<uint8_t> var_seen(m.num_vars(), 0);
    for (size_t i = 0; i < reached.size(); ++i) {
        const double floor = reached[i] != 0 ? domain_floor(m, m.nodes()[i])
                                             : std::numeric_limits<double>::quiet_NaN();
        if (std::isnan(floor)) {
            continue;
        }
        const ChildRef arg = m.children(m.nodes()[i])[0];
        if (arg.is_var ? m.var(arg.id).lb >= floor : m.nodes()[arg.id].op == NodeOp::Const) {
            continue;
        }
        const uint8_t bit = floor > 0.0 ? 1 : 2;
        uint8_t& seen = arg.is_var ? var_seen[arg.id] : node_seen[arg.id];
        if ((seen & bit) == 0) {
            seen |= bit;
            rows.push_back({arg, floor});
        }
    }
    return rows;
}

}  // namespace

// #248. Every domain-restricted op (`domain_floor`) the rows or objective read
// gets a hidden row `arg >= floor`, as SCIP's `expr/log/minzerodistance`.
// Outside the domain the op reads NaN, a flat 1e30 violation with no direction
// back; the row's graded violation gives FJ one.
//
// Costs: cuts off (0, 1e-9) under a `Log`, and adds rows the modeller did not
// write. They follow the modeller's in constraint_ids (`num_user_constraints`):
// counted by `verify_model` and the feasibility test, not by the modeller's
// row count, and left out by `save_model`.
//
// Not added:
//  - for an op nothing reads, or read only through an `If` branch:
//    `if(x > 0, log(x), 0)` is defined at x <= 0, and a row would cut it off;
//  - for a Const argument, which no move can change;
//  - for a variable whose lower bound already meets the floor (most of
//    MINLPLib), so those models keep their rows and trajectories.
// One row per (argument, floor). Runs once: the first `close()` or
// `add_objective_soft_constraint()`.
void Model::append_domain_rows() {
    ModelStructure& st = mut();
    if (st.num_user_constraints >= 0) {
        return;
    }
    // Collected before any node is made: alloc_node grows the arrays it reads.
    const std::vector<DomainRow> rows = domain_rows_needed(*this);
    st.num_user_constraints = static_cast<int64_t>(st.constraint_ids.size());
    st.domain_node_begin = static_cast<int32_t>(st.nodes.size());
    // RHS is the bare floor. Within tolerance, plain `arg >= floor` would pass
    // arg = 0 under a `Log`; `evaluate`'s `Geq` arm adds `domain_margin()` to a
    // violated domain row, so any arg below the floor exceeds the tolerance and
    // one at the floor scores 0 (#258).
    int32_t log_floor = -1;
    int32_t zero_floor = -1;
    for (const DomainRow& r : rows) {
        int32_t& rhs = r.floor > 0.0 ? log_floor : zero_floor;
        if (rhs < 0) {
            rhs = push_constant(r.floor);
        }
        ChildRef rhs_ref;
        rhs_ref.id = rhs;
        rhs_ref.is_var = false;
        st.constraint_ids.push_back(alloc_node(NodeOp::Geq, {r.arg, rhs_ref}));
    }
    st.num_domain_rows = static_cast<int32_t>(rows.size());
    st.domain_node_end = static_cast<int32_t>(st.nodes.size());
}

bool Model::set_domain_tolerance(double tol) {
    if (!(tol >= 0.0) || !std::isfinite(tol)) {
        throw std::invalid_argument("set_domain_tolerance: tolerance must be finite and >= 0");
    }
    const double margin = std::nextafter(tol, std::numeric_limits<double>::infinity());
    if (margin == domain_margin_) {
        return false;  // LNS builds an FJ per round at the solve's tolerance
    }
    domain_margin_ = margin;
    // A pending probe's Rollback would write old residuals back (#270): drop it.
    probe_undo_.pending = false;
    // Domain rows are unread roots, so re-evaluating them in place is the whole
    // update. Before close() they do not exist.
    const ModelStructure& st = s();
    for (int32_t id = std::max(st.domain_node_begin, 0); id < st.domain_node_end; ++id) {
        const ExprNode& nd = st.nodes[static_cast<size_t>(id)];
        if (nd.op == NodeOp::Geq) {
            node_values_[static_cast<size_t>(id)] = evaluate(nd, *this);
        }
    }
    return true;
}

void Model::close() {
    require_open("close");
    ModelStructure& st = mut();
    append_domain_rows();
    rebuild_back_references();
    st.topo_order = detail::compute_topo_order(*this);
    rebuild_topo_positions();
    classify_incremental_sums();
    build_var_constraints();
    full_evaluate(*this);
    closed_ = true;
}

void Model::add_objective_soft_constraint() {
    if (objective_id_ < 0) {
        throw std::invalid_argument("add_objective_soft_constraint requires an objective");
    }
    // Idempotent, returning BEFORE the frozen check: `freeze()` has run it, so on
    // a frozen replica a call must be a no-op, not a throw (#271).
    if (objective_constraint_idx_ >= 0) {
        return;
    }
    require_open("add_objective_soft_constraint");
    ModelStructure& st = mut();
    // On a model never closed, so the domain rows still come before this one.
    append_domain_rows();

    // Exactly the room the row takes (two nodes, two child refs, two values, one
    // constraint). A model at exact capacity (every copy, every model
    // `mps_to_model` sized) would otherwise DOUBLE each array, address space
    // `ulimit -v` counts. No-ops where slack exists. `node_values_` must be in the
    // list too: it is per-model, so copied per worker (#157), and leaving it
    // out doubled a 34 MB array on the largest MIPfeas instance.
    st.nodes.reserve(st.nodes.size() + 2);
    st.parent_offsets.reserve(st.parent_offsets.size() + 2);
    st.child_refs.reserve(st.child_refs.size() + 2);
    st.constraint_ids.reserve(st.constraint_ids.size() + 1);
    node_values_.reserve(node_values_.size() + 2);

    objective_bound_ = std::numeric_limits<double>::infinity();
    // Private allocators: the public builders refuse a closed model (#173), and
    // this is the one growth path that runs on one, made safe by the rebuild.
    objective_bound_node_ = push_constant(objective_bound_);
    // obj - bound <= 0; inert while bound is +inf, tightened during search.
    objective_constraint_node_ =
        alloc_node(NodeOp::Leq, {wrap(objective_id_), wrap(objective_bound_node_)});
    objective_constraint_idx_ = static_cast<int32_t>(st.constraint_ids.size());
    st.constraint_ids.push_back(objective_constraint_node_);

    // Rebuild structure now that a node/constraint was appended after close().
    rebuild_back_references();
    st.topo_order = detail::compute_topo_order(*this);
    rebuild_topo_positions();
    classify_incremental_sums();
    build_var_constraints();
    full_evaluate(*this);
    // The rebuild is close()'s, so mark it closed: #173's refusal keys on
    // closed_, else a later builder adds a row that is never evaluated.
    closed_ = true;
}

void Model::set_objective_bound(double bound) {
    if (objective_constraint_node_ < 0) {
        throw std::logic_error("set_objective_bound requires add_objective_soft_constraint first");
    }
    objective_bound_ = bound;
    // A pending probe's Rollback would write the old bound back (#270): drop it.
    probe_undo_.pending = false;
    // The shared node's `const_value` is NOT touched: it is structure, shared by
    // every worker once frozen. `evaluate()`'s `Const` arm reads
    // `objective_bound()`.
    node_values_[objective_bound_node_] = bound;
    // Residual in place (obj - bound). Must use evaluate()'s Leq rule, else this
    // and the next delta_evaluate() disagree, e.g. at obj = bound = +inf (#100).
    // The bound side is a Const (a sentinel); the objective side never is.
    node_values_[objective_constraint_node_] =
        comparison_residual(node_values_[objective_id_], bound,
                            /*a_is_const=*/false, /*b_is_const=*/true);
}

// Build var_id -> constraint-index adjacency (the paper's G_v) by walking down
// each constraint's subtree and recording every variable it reaches. Stamping
// gives O(1) per-constraint reset and dedups vars/nodes within a constraint.
//
// Into CSR by inversion: the walk writes constraint -> variables flat, in
// ascending constraint order, so each variable's list is ascending; a counting
// pass turns it round. The transient incidence costs 4 bytes per entry, where
// counting first would cost a second DFS.
void Model::build_var_constraints() {
    ModelStructure& st = mut();
    const auto n_cons = static_cast<int32_t>(st.constraint_ids.size());
    std::vector<int32_t> node_stamp(st.nodes.size(), -1);
    std::vector<int32_t> var_stamp(vars_.size(), -1);
    std::vector<int32_t> stack;
    // Sized from the last build's count (the objective-row rebuild matches it to
    // within one row), whose CSR is released first so both are never alive. Its
    // offsets go too, so after a throw constraints_of_var reports out_of_range
    // rather than reading freed ids.
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

    // Snapshot current violations, unchecked as in `child_val` (dag.cpp):
    // `constraint_ids()` holds node ids validated by add_constraint.
    const std::vector<int32_t>& cids = constraint_ids();
    std::vector<double> old_viol(affected.size());
    for (size_t k = 0; k < affected.size(); ++k) {
        old_viol[k] = clamped_node_violation(node_values_[cids[affected[k]]]);
    }

    // Probe: set candidate, recompute only the affected dirty cone.
    const double old_value = v.value;
    var_mut(var_id).value = j;
    probe_scalar_move(*this, var_id, old_value);

    for (size_t k = 0; k < affected.size(); ++k) {
        double new_viol = clamped_node_violation(node_values_[cids[affected[k]]]);
        double delta = new_viol - old_viol[k];
        if (delta != 0.0) {
            result.emplace_back(affected[k], delta);
        }
    }

    // Restore exactly: Rollback writes the probe's undo log back (#270), drifted
    // incremental Sums included.
    var_mut(var_id).value = old_value;
    delta_evaluate(*this, &var_id, 1, DeltaMode::Rollback);

    return result;
}

double Model::weighted_violation_delta(int32_t var_id, double j,
                                       const std::vector<double>& weights) {
    return probe_weighted_delta(var_id, j, weights, nullptr, nullptr);
}

double Model::weighted_violation_delta(int32_t var_id, double j, const std::vector<double>& weights,
                                       const std::vector<double>& second, double& second_delta) {
    return probe_weighted_delta(var_id, j, weights, &second, &second_delta);
}

double Model::probe_weighted_delta(int32_t var_id, double j, const std::vector<double>& weights,
                                   const std::vector<double>* second, double* second_delta) {
    const Variable& v = var(var_id);  // bounds-checked
    if (is_structured(v.type)) {
        throw std::invalid_argument(
            "weighted_violation_delta: scalar variable required (Bool/Int/Float)");
    }
    if (second_delta != nullptr) {
        *second_delta = 0.0;
    }
    const ConstSpan<int32_t> affected = constraints_of_var(var_id);
    if (affected.empty()) {
        return 0.0;
    }

    // Must accumulate per-constraint differences, not difference two whole sums:
    // a row clamped at kInfPenalty = 1e30 absorbs O(1) terms (1e30 + 1 == 1e30),
    // so every candidate would score 0 (#100). Per row, 1e30 - 1e30 cancels
    // exactly and the small terms keep full precision.
    //
    // probe_old_violation_ is a member so this is allocation-free after warm-up;
    // safe because each search thread owns its Model.
    const std::vector<int32_t>& cids = constraint_ids();
    probe_old_violation_.resize(affected.size());
    for (size_t k = 0; k < affected.size(); ++k) {
        probe_old_violation_[k] = clamped_node_violation(node_values_[cids[affected[k]]]);
    }

    // Probe/Rollback: the old value lets the probe move each incremental Sum by
    // its term's change, as the commit does (#188), so the score is against the
    // committed value, drift included. Rollback replays the undo log (#270),
    // restoring drifted state to the bit at O(cone) stores. Where the commit
    // would re-sum, the two can differ by the drift bound (`probe_scalar_move`).
    const double old_value = v.value;
    var_mut(var_id).value = j;
    probe_scalar_move(*this, var_id, old_value);

    // One probe, two weightings (#272): each row's difference is computed once,
    // and each sum is the single-weight call's, term for term and in order.
    double delta = 0.0;
    double delta2 = 0.0;
    for (size_t k = 0; k < affected.size(); ++k) {
        const int32_t c = affected[k];
        const double new_viol = clamped_node_violation(node_values_[cids[c]]);
        const double diff = new_viol - probe_old_violation_[k];
        delta += weights[c] * diff;
        if (second != nullptr) {
            delta2 += (*second)[c] * diff;
        }
    }

    var_mut(var_id).value = old_value;
    delta_evaluate(*this, &var_id, 1, DeltaMode::Rollback);

    if (second_delta != nullptr) {
        *second_delta = delta2;
    }
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
