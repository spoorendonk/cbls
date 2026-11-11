#include "cbls/io.h"

#include <cstddef>
#include <fstream>
#include <functional>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cbls {

using json = nlohmann::json;

static std::string op_to_string(NodeOp op) {
    switch (op) {
        case NodeOp::Const:
            return "Const";
        case NodeOp::Neg:
            return "Neg";
        case NodeOp::Sum:
            return "Sum";
        case NodeOp::Prod:
            return "Prod";
        case NodeOp::Div:
            return "Div";
        case NodeOp::Pow:
            return "Pow";
        case NodeOp::Min:
            return "Min";
        case NodeOp::Max:
            return "Max";
        case NodeOp::Abs:
            return "Abs";
        case NodeOp::Sin:
            return "Sin";
        case NodeOp::Cos:
            return "Cos";
        case NodeOp::Tan:
            return "Tan";
        case NodeOp::Exp:
            return "Exp";
        case NodeOp::Log:
            return "Log";
        case NodeOp::Sqrt:
            return "Sqrt";
        case NodeOp::SignPower:
            return "SignPower";
        case NodeOp::Tanh:
            return "Tanh";
        case NodeOp::If:
            return "If";
        case NodeOp::At:
            return "At";
        case NodeOp::Count:
            return "Count";
        case NodeOp::Lambda:
            return "Lambda";
        case NodeOp::PairLambda:
            return "PairLambda";
        case NodeOp::Leq:
            return "Leq";
        case NodeOp::Eq:
            return "Eq";
        case NodeOp::Geq:
            return "Geq";
        case NodeOp::Neq:
            return "Neq";
        case NodeOp::Lt:
            return "Lt";
        case NodeOp::Gt:
            return "Gt";
        case NodeOp::Element:
            return "Element";
        case NodeOp::Ceil:
            return "Ceil";
        case NodeOp::Floor:
            return "Floor";
        case NodeOp::Round:
            return "Round";
        case NodeOp::LambdaExtra:
            // Never written (`save_model` refuses it); named so a diagnostic
            // does not say "Unknown". No `default:`, so -Wswitch flags a new
            // op (#171; see evaluate() in src/dag.cpp).
            return "LambdaExtra";
        case NodeOp::PairLambdaExtra:
            return "PairLambdaExtra";
    }
    return "Unknown";
}

static NodeOp string_to_op(const std::string& s) {
    static const std::unordered_map<std::string, NodeOp> kMap = {
        {"Const", NodeOp::Const},     {"Neg", NodeOp::Neg},
        {"Sum", NodeOp::Sum},         {"Prod", NodeOp::Prod},
        {"Div", NodeOp::Div},         {"Pow", NodeOp::Pow},
        {"Min", NodeOp::Min},         {"Max", NodeOp::Max},
        {"Abs", NodeOp::Abs},         {"Sin", NodeOp::Sin},
        {"Cos", NodeOp::Cos},         {"Tan", NodeOp::Tan},
        {"Exp", NodeOp::Exp},         {"Log", NodeOp::Log},
        {"Sqrt", NodeOp::Sqrt},       {"SignPower", NodeOp::SignPower},
        {"Tanh", NodeOp::Tanh},       {"If", NodeOp::If},
        {"At", NodeOp::At},           {"Count", NodeOp::Count},
        {"Lambda", NodeOp::Lambda},   {"PairLambda", NodeOp::PairLambda},
        {"Leq", NodeOp::Leq},         {"Eq", NodeOp::Eq},
        {"Geq", NodeOp::Geq},         {"Neq", NodeOp::Neq},
        {"Lt", NodeOp::Lt},           {"Gt", NodeOp::Gt},
        {"Element", NodeOp::Element}, {"Ceil", NodeOp::Ceil},
        {"Floor", NodeOp::Floor},     {"Round", NodeOp::Round},
    };
    auto it = kMap.find(s);
    if (it == kMap.end()) {
        throw std::invalid_argument("unknown op: " + s);
    }
    return it->second;
}

static std::string vartype_to_string(VarType t) {
    switch (t) {
        case VarType::Bool:
            return "Bool";
        case VarType::Int:
            return "Int";
        case VarType::Float:
            return "Float";
        case VarType::List:
            return "List";
        case VarType::Set:
            return "Set";
    }
    return "Unknown";
}

static VarType string_to_vartype(const std::string& s) {
    if (s == "Bool") {
        return VarType::Bool;
    }
    if (s == "Int") {
        return VarType::Int;
    }
    if (s == "Float") {
        return VarType::Float;
    }
    if (s == "List") {
        return VarType::List;
    }
    if (s == "Set") {
        return VarType::Set;
    }
    throw std::invalid_argument("unknown variable type: " + s);
}

// Resolve a name to a handle (var handle or node id)
static int32_t resolve(const std::string& name,
                       const std::unordered_map<std::string, int32_t>& name_to_handle,
                       int line_num) {
    auto it = name_to_handle.find(name);
    if (it == name_to_handle.end()) {
        throw std::invalid_argument("line " + std::to_string(line_num) + ": unknown reference '" +
                                    name + "'");
    }
    return it->second;
}

namespace {

using NameMap = std::unordered_map<std::string, int32_t>;

// How a List record spells `ListInit`. Absent means Empty, which is
// `list_var`'s own default for the variable-length form.
ListInit parse_list_init(const json& j) {
    const auto init = j.value("init", std::string("empty"));
    if (init == "identity") {
        return ListInit::Identity;
    }
    if (init == "empty") {
        return ListInit::Empty;
    }
    if (init == "random") {
        return ListInit::Random;
    }
    throw std::invalid_argument("unknown List init '" + init + "'");
}

const char* list_init_name(ListInit init) {
    switch (init) {
        case ListInit::Identity:
            return "identity";
        case ListInit::Empty:
            return "empty";
        case ListInit::Random:
            return "random";
    }
    return "empty";
}

Cover parse_cover(const json& j) {
    const auto cover = j.value("cover", std::string("exact"));
    if (cover == "exact") {
        return Cover::Exact;
    }
    if (cover == "at_most_once") {
        return Cover::AtMostOnce;
    }
    throw std::invalid_argument("unknown partition cover '" + cover + "'");
}

// One `{"var": ...}` record, whose name the caller has already read. Returns the
// new variable's handle.
int32_t load_var_record(Model& m, const json& j, const std::string& name) {
    VarType vtype = string_to_vartype(j.value("type", "Float"));
    switch (vtype) {
        case VarType::Bool:
            return m.bool_var(name);
        case VarType::Int:
            return m.int_var(j.value("lb", 0), j.value("ub", 1), name);
        case VarType::Float:
            return m.float_var(j.value("lb", 0.0), j.value("ub", 1.0), name);
        case VarType::List: {
            const int n = j.at("n").get<int>();
            // None of the three optional keys means a permutation List, the
            // form every pre-#164 file holds; a missing one defaults.
            if (!j.contains("min_len") && !j.contains("max_len") && !j.contains("init")) {
                return m.list_var(n, name);
            }
            const int min_len = j.value("min_len", 0);
            const int max_len = j.value("max_len", n);
            return m.list_var(n, min_len, max_len, parse_list_init(j), name);
        }
        case VarType::Set: {
            int n = j.at("n").get<int>();
            int min_sz = j.value("min_size", 0);
            int max_sz = j.value("max_size", -1);
            return m.set_var(n, min_sz, max_sz, name);
        }
    }
    return 0;  // unreachable: the switch is exhaustive over VarType
}

// The (Pair)Lambda payload: a tabulated function, stored as a flat array for
// Lambda and a square matrix for PairLambda, keyed by list element value.
const json& require_table(const json& j, const char* op_name, int line_num) {
    if (!j.contains("table")) {
        throw std::invalid_argument("line " + std::to_string(line_num) + ": " + op_name +
                                    " node requires 'table' field");
    }
    return j["table"];
}

// The PairLambda closing rule, as the format spells it.
PairMode parse_pair_mode(const json& j, int line_num) {
    const auto mode = j.value("mode", std::string("open"));
    if (mode == "open") {
        return PairMode::Open;
    }
    if (mode == "cyclic") {
        return PairMode::Cyclic;
    }
    throw std::invalid_argument("line " + std::to_string(line_num) + ": unknown PairLambda mode '" +
                                mode + "'");
}

// One tabulated fixed-endpoint term, or an empty callable ("no term" to
// `pair_lambda_sum`) when the table is absent or empty. The writer never emits
// `[]`, so save(load(save)) == save holds.
std::function<double(int)> endpoint_func(const json& j, const char* field) {
    auto table = j.value(field, std::vector<double>{});
    if (table.empty()) {
        return nullptr;
    }
    return [table = std::move(table)](int e) -> double { return table.at(e); };
}

// An Element record (#186): `"table"` is a flat array for one index child and an
// array of equal-length rows for two. The model builder checks the shape; this
// only picks which of its two forms the child count asks for.
int32_t build_element(Model& m, const json& j, const std::vector<int32_t>& children, int line_num) {
    const json& table = require_table(j, "Element", line_num);
    if (children.size() == 1) {
        return m.element(table.get<std::vector<double>>(), children[0]);
    }
    if (children.size() == 2) {
        return m.element(table.get<std::vector<std::vector<double>>>(), children[0], children[1]);
    }
    throw std::invalid_argument("line " + std::to_string(line_num) +
                                ": Element node requires one or two index children");
}

// The one child of a rounding record (#186): any other count is refused, not
// silently read as a different model.
int32_t only_child(const std::vector<int32_t>& children, const char* op_name, int line_num) {
    if (children.size() != 1) {
        throw std::invalid_argument("line " + std::to_string(line_num) + ": " + op_name +
                                    " node requires exactly one child");
    }
    return children[0];
}

// Rebuild one non-Const node from its op and its already-resolved children:
// one line per NodeOp, no shared state, hence apart from the plumbing above.
int32_t build_node(Model& m, NodeOp op, const json& j, const std::vector<int32_t>& children,
                   int line_num) {
    switch (op) {
        case NodeOp::Neg:
            return m.neg(children.at(0));
        case NodeOp::Sum:
            return m.sum(children);
        case NodeOp::Prod:
            return m.prod(children.at(0), children.at(1));
        case NodeOp::Div:
            return m.div_expr(children.at(0), children.at(1));
        case NodeOp::Pow:
            return m.pow_expr(children.at(0), children.at(1));
        case NodeOp::Min:
            return m.min_expr(children);
        case NodeOp::Max:
            return m.max_expr(children);
        case NodeOp::Abs:
            return m.abs_expr(children.at(0));
        case NodeOp::Sin:
            return m.sin_expr(children.at(0));
        case NodeOp::Cos:
            return m.cos_expr(children.at(0));
        case NodeOp::Tan:
            return m.tan_expr(children.at(0));
        case NodeOp::Exp:
            return m.exp_expr(children.at(0));
        case NodeOp::Log:
            return m.log_expr(children.at(0));
        case NodeOp::Sqrt:
            return m.sqrt_expr(children.at(0));
        case NodeOp::SignPower:
            return m.signpower_expr(children.at(0), children.at(1));
        case NodeOp::Tanh:
            return m.tanh_expr(children.at(0));
        case NodeOp::If:
            return m.if_then_else(children.at(0), children.at(1), children.at(2));
        case NodeOp::At:
            return m.at(children.at(0), children.at(1));
        case NodeOp::Count:
            return m.count(children.at(0));
        case NodeOp::Lambda: {
            auto table = require_table(j, "Lambda", line_num).get<std::vector<double>>();
            return m.lambda_sum(children.at(0), [table](int e) -> double { return table.at(e); });
        }
        case NodeOp::PairLambda: {
            auto table =
                require_table(j, "PairLambda", line_num).get<std::vector<std::vector<double>>>();
            // Absent "mode"/endpoint tables mean the open chain with no
            // endpoint terms, what files predating those forms meant.
            const PairMode mode = parse_pair_mode(j, line_num);
            auto head = endpoint_func(j, "head");
            auto tail = endpoint_func(j, "tail");
            return m.pair_lambda_sum(
                children.at(0), [table](int a, int b) -> double { return table.at(a).at(b); },
                std::move(head), std::move(tail), mode);
        }
        case NodeOp::Leq:
            return m.leq(children.at(0), children.at(1));
        case NodeOp::Eq:
            return m.eq_expr(children.at(0), children.at(1));
        case NodeOp::Geq:
            return m.geq(children.at(0), children.at(1));
        case NodeOp::Neq:
            return m.neq(children.at(0), children.at(1));
        case NodeOp::Lt:
            return m.lt(children.at(0), children.at(1));
        case NodeOp::Gt:
            return m.gt(children.at(0), children.at(1));
        case NodeOp::Const:
            break;  // handled by the caller, which needs no children
        case NodeOp::Element:
            return build_element(m, j, children, line_num);
        case NodeOp::Ceil:
            return m.ceil_expr(only_child(children, "Ceil", line_num));
        case NodeOp::Floor:
            return m.floor_expr(only_child(children, "Floor", line_num));
        case NodeOp::Round:
            return m.round_expr(only_child(children, "Round", line_num));
        case NodeOp::LambdaExtra:
        case NodeOp::PairLambdaExtra:
            // Unreachable: `string_to_op` has no entry for either, so the
            // reader refuses them as unknown ops. A throw, not `break`, so a
            // future reader-side entry cannot silently return -1.
            throw std::invalid_argument("line " + std::to_string(line_num) +
                                        ": lambdas over extra decisions hold user code and "
                                        "cannot be loaded");
    }
    return -1;
}

// One `{"node": ...}` record. Returns the new node's id.
int32_t load_node_record(Model& m, const json& j, const NameMap& name_to_handle, int line_num) {
    std::string op_str = j.at("op").get<std::string>();
    NodeOp op = NodeOp::Const;
    try {
        op = string_to_op(op_str);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument("line " + std::to_string(line_num) + ": unknown op '" + op_str +
                                    "'");
    }

    if (op == NodeOp::Const) {
        return m.constant(j.at("value").get<double>());
    }

    std::vector<int32_t> children;
    if (j.contains("children")) {
        for (const auto& child_name : j["children"]) {
            children.push_back(resolve(child_name.get<std::string>(), name_to_handle, line_num));
        }
    }
    return build_node(m, op, j, children, line_num);
}

json parse_record(const std::string& line, int line_num) {
    try {
        return json::parse(line);
    } catch (const json::parse_error& e) {
        throw std::invalid_argument("line " + std::to_string(line_num) +
                                    ": invalid JSON: " + e.what());
    }
}

// Dispatch one already-parsed record to the loader for its kind, recording any
// name it defines. The first kind key matched wins; none of the six is refused.
void load_record(Model& m, const json& j, NameMap& name_to_handle, int line_num) {
    if (j.contains("var")) {
        std::string name = j["var"].get<std::string>();
        name_to_handle[name] = load_var_record(m, j, name);
    } else if (j.contains("node")) {
        // The node's own name is recorded only after its children are resolved,
        // so a record cannot refer to itself.
        std::string name = j["node"].get<std::string>();
        int32_t node_id = load_node_record(m, j, name_to_handle, line_num);
        name_to_handle[name] = node_id;
    } else if (j.contains("partition")) {
        // `{"partition": [<list names>], "cover": ...}` (#164): resolves names
        // like a constraint and adds nothing to the DAG.
        std::vector<int32_t> lists;
        for (const auto& list_name : j["partition"]) {
            const int32_t handle = resolve(list_name.get<std::string>(), name_to_handle, line_num);
            if (handle >= 0) {
                // A node id: `add_list_partition` reads a non-negative value as
                // a raw variable id, so it must be refused here, else the
                // variable carrying that id is silently partitioned.
                throw std::invalid_argument("line " + std::to_string(line_num) +
                                            ": partition member '" + list_name.get<std::string>() +
                                            "' is a node, not a variable");
            }
            lists.push_back(handle);
        }
        m.add_list_partition(lists, parse_cover(j));
    } else if (j.contains("constraint")) {
        m.add_constraint(resolve(j["constraint"].get<std::string>(), name_to_handle, line_num));
    } else if (j.contains("minimize")) {
        m.minimize(resolve(j["minimize"].get<std::string>(), name_to_handle, line_num));
    } else if (j.contains("maximize")) {
        m.maximize(resolve(j["maximize"].get<std::string>(), name_to_handle, line_num));
    } else {
        throw std::invalid_argument("line " + std::to_string(line_num) +
                                    ": unrecognized line type");
    }
}

}  // namespace

Model load_model(std::istream& input) {
    Model m;
    // name -> handle: var handles are negative -(var_id+1), node handles are node_id
    NameMap name_to_handle;
    int line_num = 0;
    std::string line;

    while (std::getline(input, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        load_record(m, parse_record(line, line_num), name_to_handle, line_num);
    }

    m.close();
    return m;
}

Model load_model(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("cannot open file: " + path);
    }
    return load_model(file);
}

// Writer

namespace {

// id -> serialisation name, one table per handle space.
using NameTable = std::unordered_map<int32_t, std::string>;

// A variable keeps its declared name, else a positional one; nodes are always
// positional (the DAG does not name them).
void build_name_tables(const Model& model, NameTable& var_names, NameTable& node_names) {
    for (const auto& var : model.variables()) {
        if (!var.name.empty()) {
            var_names[var.id] = var.name;
        } else {
            var_names[var.id] = "v" + std::to_string(var.id);
        }
    }
    for (const auto& node : model.nodes()) {
        node_names[node.id] = "n" + std::to_string(node.id);
    }
}

// operator[] on purpose: an id missing from build_name_tables writes an empty
// name rather than throwing mid-file.
std::string child_name(const ChildRef& ref, NameTable& var_names, NameTable& node_names) {
    if (ref.is_var) {
        return var_names[ref.id];
    }
    return node_names[ref.id];
}

json var_record(const Variable& var, const std::string& name) {
    json j;
    j["var"] = name;
    j["type"] = vartype_to_string(var.type);
    switch (var.type) {
        case VarType::Bool:
            break;
        case VarType::Int:
            j["lb"] = static_cast<int>(var.lb);
            j["ub"] = static_cast<int>(var.ub);
            break;
        case VarType::Float:
            j["lb"] = var.lb;
            j["ub"] = var.ub;
            break;
        case VarType::List:
            j["n"] = var.universe_size;
            // A permutation writes `n` alone, so pre-#164 files keep
            // save(load(save)) == save; otherwise all three keys are written.
            if (var.min_size != var.universe_size || var.max_size != var.universe_size ||
                var.list_init != ListInit::Identity) {
                j["min_len"] = var.min_size;
                j["max_len"] = var.max_size;
                j["init"] = list_init_name(var.list_init);
            }
            break;
        case VarType::Set:
            j["n"] = var.universe_size;
            j["min_size"] = var.min_size;
            j["max_size"] = var.max_size;
            break;
    }
    return j;
}

// A (Pair)Lambda's std::function is written tabulated over its child's
// universe: [func(0) .. func(n-1)], or the n x n matrix for PairLambda. The size
// caps refuse a table too large to write.
void tabulate_lambda(const Model& model, const ExprNode& node, json& j) {
    const auto& child_ref = model.children(node)[0];
    if (!child_ref.is_var) {
        throw std::runtime_error("Lambda node child must be a variable");
    }
    const auto& var = model.var(child_ref.id);
    // `universe_size` for List and Set alike (#164): elements range over the
    // universe; `max_size` is only the length/cardinality bound.
    int n = var.universe_size;
    if (n > 10000) {
        throw std::runtime_error("Lambda universe too large to tabulate (" + std::to_string(n) +
                                 " > 10000)");
    }
    const auto& func = model.lambda_func(node.lambda_func_id);
    j["table"] = json::array();
    for (int i = 0; i < n; ++i) {
        j["table"].push_back(func(i));
    }
}

void tabulate_pair_lambda(const Model& model, const ExprNode& node, json& j) {
    const auto& child_ref = model.children(node)[0];
    if (!child_ref.is_var) {
        throw std::runtime_error("PairLambda node child must be a variable");
    }
    const auto& var = model.var(child_ref.id);
    // Sized by universe_size, as in tabulate_lambda: sizing by max_size makes
    // a reload throw `table.at()` past it. A Set file written sized by max_size
    // (the pre-fix writer) thus throws on re-save, and a universe over the cap refuses to
    // serialise.
    int n = var.universe_size;
    if (n > 1000) {
        throw std::runtime_error("PairLambda universe too large to tabulate (" + std::to_string(n) +
                                 " > 1000)");
    }
    const auto& func = model.pair_lambda_func(node.lambda_func_id);
    j["table"] = json::array();
    for (int i = 0; i < n; ++i) {
        auto row = json::array();
        for (int jj = 0; jj < n; ++jj) {
            row.push_back(func(i, jj));
        }
        j["table"].push_back(row);
    }
    // Mode and endpoint terms must be written, else a cyclic tour reloads as an
    // open chain. Each key only when non-default: the format has no version
    // field, so an open chain without endpoints keeps its old bytes
    // (`tests/test_io.cpp` compares save(load(save)) as text).
    const PairLambdaSpec& spec = model.pair_lambda_spec(node.lambda_func_id);
    if (spec.mode == PairMode::Cyclic) {
        j["mode"] = "cyclic";
    }
    // n == 0 writes nothing, lossless (no element to charge): the reader reads
    // an empty table as "no term", so emitting one breaks save(load(save)).
    auto tabulate_endpoint = [&model, &j, n](const char* field, int32_t func_id) {
        if (func_id < 0 || n == 0) {
            return;
        }
        const auto& endpoint_fn = model.lambda_func(func_id);
        auto values = json::array();
        for (int i = 0; i < n; ++i) {
            values.push_back(endpoint_fn(i));
        }
        j[field] = values;
    };
    tabulate_endpoint("head", spec.head_id);
    tabulate_endpoint("tail", spec.tail_id);
}

json node_record(const Model& model, const ExprNode& node, NameTable& var_names,
                 NameTable& node_names) {
    json j;
    j["node"] = node_names[node.id];
    j["op"] = op_to_string(node.op);

    if (node.op == NodeOp::Const) {
        // The literal, not `model.node_value(node.id)`: they differ on the
        // objective row's RHS, whose value is the search's bound and whose
        // literal is +inf; `solve()` re-adds that row (#157).
        j["value"] = node.const_value;
        return j;
    }
    if (node.op == NodeOp::Lambda || node.op == NodeOp::PairLambda) {
        if (node.op == NodeOp::Lambda) {
            tabulate_lambda(model, node, j);
        } else {
            tabulate_pair_lambda(model, node, j);
        }
        // Only the tabulated child is written back; the tabulation subsumes any
        // index expression the node carried.
        j["children"] = json::array();
        j["children"].push_back(child_name(model.children(node)[0], var_names, node_names));
        return j;
    }
    if (node.op == NodeOp::Element) {
        // Written as the builder took it: a flat array for one index, rows for two.
        const ElementTable& tbl = model.element_table(node.lambda_func_id);
        if (model.children(node).size() == 1) {
            j["table"] = tbl.values;
        } else {
            j["table"] = json::array();
            for (int32_t r = 0; r < tbl.rows; ++r) {
                const auto first = tbl.values.begin() + (static_cast<std::ptrdiff_t>(r) * tbl.cols);
                j["table"].push_back(std::vector<double>(first, first + tbl.cols));
            }
        }
    }
    j["children"] = json::array();
    for (const ChildRef& ch : model.children(node)) {
        j["children"].push_back(child_name(ch, var_names, node_names));
    }
    return j;
}

json objective_record(const Model& model, NameTable& var_names, NameTable& node_names) {
    json j;
    if (model.is_maximizing()) {
        // Unwrap the auto-generated Neg node.
        const auto& neg_node = model.node(model.objective_id());
        j["maximize"] = child_name(model.children(neg_node)[0], var_names, node_names);
    } else {
        j["minimize"] = node_names[model.objective_id()];
    }
    return j;
}

// A `lambda_sum` / `pair_lambda_sum` over extra decisions (#186) has a functor of
// continuous values, so no finite table: refused. Must run before any line is
// written, else a throw leaves a truncated file that looks like a model.
void refuse_user_code_nodes(const Model& model) {
    // O(1) without such nodes (every in-tree model), so the `path` overload's
    // second call costs nothing; with one, a walk of all nodes.
    if (!model.has_lambda_extra_nodes()) {
        return;
    }
    for (const auto& node : model.nodes()) {
        if (node.op == NodeOp::LambdaExtra || node.op == NodeOp::PairLambdaExtra) {
            throw std::runtime_error("cannot serialise model: node n" + std::to_string(node.id) +
                                     " (" + op_to_string(node.op) +
                                     ") holds a functor over extra decisions, which the .cbls "
                                     "format cannot tabulate");
        }
    }
}

}  // namespace

void save_model(const Model& model, std::ostream& out) {
    refuse_user_code_nodes(model);
    NameTable var_names;
    NameTable node_names;
    build_name_tables(model, var_names, node_names);

    for (const auto& var : model.variables()) {
        out << var_record(var, var_names[var.id]).dump() << '\n';
    }

    // After the variables, before the nodes: members are variables only and
    // `add_list_partition` requires them to exist. None writes nothing, so
    // pre-#164 files round-trip unchanged.
    for (const ListPartition& part : model.list_partitions()) {
        json j;
        auto names = json::array();
        for (int32_t vid : part.list_ids) {
            names.push_back(var_names[vid]);
        }
        j["partition"] = names;
        j["cover"] = (part.cover == Cover::Exact) ? "exact" : "at_most_once";
        out << j.dump() << '\n';
    }

    // Nodes in topological order, so a reader sees every child before its parent.
    for (int32_t nid : model.topo_order()) {
        // Skip the auto-generated Neg wrapper when maximizing.
        if (model.is_maximizing() && nid == model.objective_id()) {
            continue;
        }
        // The implicit domain rows (#248): a reload's close() makes them again.
        if (model.is_domain_node(nid)) {
            continue;
        }
        out << node_record(model, model.node(nid), var_names, node_names).dump() << '\n';
    }

    for (int32_t cid : model.constraint_ids()) {
        if (model.is_domain_node(cid)) {
            continue;
        }
        json j;
        j["constraint"] = node_names[cid];
        out << j.dump() << '\n';
    }

    if (model.objective_id() >= 0) {
        out << objective_record(model, var_names, node_names).dump() << '\n';
    }
}

void save_model(const Model& model, const std::string& path) {
    // Before opening, which truncates: a later refusal destroys the old file.
    refuse_user_code_nodes(model);
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("cannot open file for writing: " + path);
    }
    save_model(model, file);
}

}  // namespace cbls
