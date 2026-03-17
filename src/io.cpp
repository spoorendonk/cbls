#include "cbls/io.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

namespace cbls {

using json = nlohmann::json;

static std::string op_to_string(NodeOp op) {
    switch (op) {
    case NodeOp::Const:  return "Const";
    case NodeOp::Neg:    return "Neg";
    case NodeOp::Sum:    return "Sum";
    case NodeOp::Prod:   return "Prod";
    case NodeOp::Div:    return "Div";
    case NodeOp::Pow:    return "Pow";
    case NodeOp::Min:    return "Min";
    case NodeOp::Max:    return "Max";
    case NodeOp::Abs:    return "Abs";
    case NodeOp::Sin:    return "Sin";
    case NodeOp::Cos:    return "Cos";
    case NodeOp::Tan:    return "Tan";
    case NodeOp::Exp:    return "Exp";
    case NodeOp::Log:    return "Log";
    case NodeOp::Sqrt:   return "Sqrt";
    case NodeOp::If:     return "If";
    case NodeOp::At:     return "At";
    case NodeOp::Count:  return "Count";
    case NodeOp::Lambda: return "Lambda";
    case NodeOp::Leq:    return "Leq";
    case NodeOp::Eq:     return "Eq";
    case NodeOp::Geq:    return "Geq";
    case NodeOp::Neq:    return "Neq";
    case NodeOp::Lt:     return "Lt";
    case NodeOp::Gt:     return "Gt";
    }
    return "Unknown";
}

static NodeOp string_to_op(const std::string& s) {
    static const std::unordered_map<std::string, NodeOp> map = {
        {"Const", NodeOp::Const}, {"Neg", NodeOp::Neg},
        {"Sum", NodeOp::Sum}, {"Prod", NodeOp::Prod},
        {"Div", NodeOp::Div}, {"Pow", NodeOp::Pow},
        {"Min", NodeOp::Min}, {"Max", NodeOp::Max},
        {"Abs", NodeOp::Abs}, {"Sin", NodeOp::Sin},
        {"Cos", NodeOp::Cos}, {"Tan", NodeOp::Tan},
        {"Exp", NodeOp::Exp}, {"Log", NodeOp::Log},
        {"Sqrt", NodeOp::Sqrt}, {"If", NodeOp::If},
        {"At", NodeOp::At}, {"Count", NodeOp::Count},
        {"Lambda", NodeOp::Lambda},
        {"Leq", NodeOp::Leq}, {"Eq", NodeOp::Eq},
        {"Geq", NodeOp::Geq}, {"Neq", NodeOp::Neq},
        {"Lt", NodeOp::Lt}, {"Gt", NodeOp::Gt},
    };
    auto it = map.find(s);
    if (it == map.end()) return static_cast<NodeOp>(255);  // sentinel for unknown
    return it->second;
}

static std::string vartype_to_string(VarType t) {
    switch (t) {
    case VarType::Bool:  return "Bool";
    case VarType::Int:   return "Int";
    case VarType::Float: return "Float";
    case VarType::List:  return "List";
    case VarType::Set:   return "Set";
    }
    return "Unknown";
}

static VarType string_to_vartype(const std::string& s) {
    if (s == "Bool")  return VarType::Bool;
    if (s == "Int")   return VarType::Int;
    if (s == "Float") return VarType::Float;
    if (s == "List")  return VarType::List;
    if (s == "Set")   return VarType::Set;
    throw std::invalid_argument("unknown variable type: " + s);
}

// Resolve a name to a handle (var handle or node id)
static int32_t resolve(const std::string& name,
                       const std::unordered_map<std::string, int32_t>& name_to_handle,
                       int line_num) {
    auto it = name_to_handle.find(name);
    if (it == name_to_handle.end()) {
        throw std::invalid_argument("line " + std::to_string(line_num) +
                                    ": unknown reference '" + name + "'");
    }
    return it->second;
}

Model load_model(std::istream& input) {
    Model m;
    // name -> handle: var handles are negative -(var_id+1), node handles are node_id
    std::unordered_map<std::string, int32_t> name_to_handle;
    int line_num = 0;
    std::string line;

    while (std::getline(input, line)) {
        line_num++;
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        json j;
        try {
            j = json::parse(line);
        } catch (const json::parse_error& e) {
            throw std::invalid_argument("line " + std::to_string(line_num) +
                                        ": invalid JSON: " + e.what());
        }

        if (j.contains("var")) {
            std::string name = j["var"].get<std::string>();
            std::string type_str = j.value("type", "Float");
            VarType vtype = string_to_vartype(type_str);
            int32_t handle = 0;

            switch (vtype) {
            case VarType::Bool:
                handle = m.bool_var(name);
                break;
            case VarType::Int:
                handle = m.int_var(j.value("lb", 0), j.value("ub", 1), name);
                break;
            case VarType::Float:
                handle = m.float_var(j.value("lb", 0.0), j.value("ub", 1.0), name);
                break;
            case VarType::List:
                handle = m.list_var(j.at("n").get<int>(), name);
                break;
            case VarType::Set: {
                int n = j.at("n").get<int>();
                int min_sz = j.value("min_size", 0);
                int max_sz = j.value("max_size", -1);
                handle = m.set_var(n, min_sz, max_sz, name);
                break;
            }
            }
            name_to_handle[name] = handle;

        } else if (j.contains("node")) {
            std::string name = j["node"].get<std::string>();
            std::string op_str = j.at("op").get<std::string>();
            NodeOp op = string_to_op(op_str);
            if (op == static_cast<NodeOp>(255)) {
                throw std::invalid_argument("line " + std::to_string(line_num) +
                                            ": unknown op '" + op_str + "'");
            }

            int32_t node_id = -1;

            if (op == NodeOp::Const) {
                node_id = m.constant(j.at("value").get<double>());
            } else {
                // Resolve children
                std::vector<int32_t> children;
                if (j.contains("children")) {
                    for (const auto& child_name : j["children"]) {
                        children.push_back(resolve(child_name.get<std::string>(),
                                                   name_to_handle, line_num));
                    }
                }

                switch (op) {
                case NodeOp::Neg:
                    node_id = m.neg(children.at(0));
                    break;
                case NodeOp::Sum:
                    node_id = m.sum(children);
                    break;
                case NodeOp::Prod:
                    node_id = m.prod(children.at(0), children.at(1));
                    break;
                case NodeOp::Div:
                    node_id = m.div_expr(children.at(0), children.at(1));
                    break;
                case NodeOp::Pow:
                    node_id = m.pow_expr(children.at(0), children.at(1));
                    break;
                case NodeOp::Min:
                    node_id = m.min_expr(children);
                    break;
                case NodeOp::Max:
                    node_id = m.max_expr(children);
                    break;
                case NodeOp::Abs:
                    node_id = m.abs_expr(children.at(0));
                    break;
                case NodeOp::Sin:
                    node_id = m.sin_expr(children.at(0));
                    break;
                case NodeOp::Cos:
                    node_id = m.cos_expr(children.at(0));
                    break;
                case NodeOp::Tan:
                    node_id = m.tan_expr(children.at(0));
                    break;
                case NodeOp::Exp:
                    node_id = m.exp_expr(children.at(0));
                    break;
                case NodeOp::Log:
                    node_id = m.log_expr(children.at(0));
                    break;
                case NodeOp::Sqrt:
                    node_id = m.sqrt_expr(children.at(0));
                    break;
                case NodeOp::If:
                    node_id = m.if_then_else(children.at(0), children.at(1), children.at(2));
                    break;
                case NodeOp::At:
                    node_id = m.at(children.at(0), children.at(1));
                    break;
                case NodeOp::Count:
                    node_id = m.count(children.at(0));
                    break;
                case NodeOp::Lambda: {
                    if (!j.contains("table")) {
                        throw std::invalid_argument("line " + std::to_string(line_num) +
                                                    ": Lambda node requires 'table' field");
                    }
                    auto table = j["table"].get<std::vector<double>>();
                    node_id = m.lambda_sum(children.at(0),
                        [table](int e) -> double { return table.at(e); });
                    break;
                }
                case NodeOp::Leq:
                    node_id = m.leq(children.at(0), children.at(1));
                    break;
                case NodeOp::Eq:
                    node_id = m.eq_expr(children.at(0), children.at(1));
                    break;
                case NodeOp::Geq:
                    node_id = m.geq(children.at(0), children.at(1));
                    break;
                case NodeOp::Neq:
                    node_id = m.neq(children.at(0), children.at(1));
                    break;
                case NodeOp::Lt:
                    node_id = m.lt(children.at(0), children.at(1));
                    break;
                case NodeOp::Gt:
                    node_id = m.gt(children.at(0), children.at(1));
                    break;
                default:
                    throw std::invalid_argument("line " + std::to_string(line_num) +
                                                ": unsupported op '" + op_str + "'");
                }
            }
            name_to_handle[name] = node_id;

        } else if (j.contains("constraint")) {
            std::string ref = j["constraint"].get<std::string>();
            int32_t handle = resolve(ref, name_to_handle, line_num);
            m.add_constraint(handle);

        } else if (j.contains("minimize")) {
            std::string ref = j["minimize"].get<std::string>();
            int32_t handle = resolve(ref, name_to_handle, line_num);
            m.minimize(handle);

        } else if (j.contains("maximize")) {
            std::string ref = j["maximize"].get<std::string>();
            int32_t handle = resolve(ref, name_to_handle, line_num);
            m.maximize(handle);

        } else {
            throw std::invalid_argument("line " + std::to_string(line_num) +
                                        ": unrecognized line type");
        }
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

void save_model(const Model& model, std::ostream& out) {
    // Build reverse maps: var_id -> name, node_id -> name
    // For unnamed entities, generate names
    std::unordered_map<int32_t, std::string> var_names;
    std::unordered_map<int32_t, std::string> node_names;

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

    // Check if any node has a user-supplied name via the name_to_handle map
    // We don't have that info, so we use generated names for nodes

    // Write variables
    for (const auto& var : model.variables()) {
        json j;
        j["var"] = var_names[var.id];
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
            j["n"] = var.max_size;
            break;
        case VarType::Set:
            j["n"] = var.universe_size;
            j["min_size"] = var.min_size;
            j["max_size"] = var.max_size;
            break;
        }
        out << j.dump() << '\n';
    }

    // Helper to get the name for a child reference
    auto child_name = [&](const ChildRef& ref) -> std::string {
        if (ref.is_var) {
            return var_names[ref.id];
        } else {
            return node_names[ref.id];
        }
    };

    // Write nodes in topological order
    for (int32_t nid : model.topo_order()) {
        const auto& node = model.node(nid);
        json j;
        j["node"] = node_names[node.id];
        j["op"] = op_to_string(node.op);

        if (node.op == NodeOp::Const) {
            j["value"] = node.const_value;
        } else if (node.op == NodeOp::Lambda) {
            // Tabulate: store [func(0), func(1), ..., func(n-1)]
            const auto& child_ref = node.children[0];
            if (!child_ref.is_var) {
                throw std::runtime_error("Lambda node child must be a variable");
            }
            const auto& var = model.var(child_ref.id);
            int n = (var.type == VarType::Set) ? var.universe_size : var.max_size;
            if (n > 10000) {
                throw std::runtime_error(
                    "Lambda universe too large to tabulate (" + std::to_string(n) + " > 10000)");
            }
            const auto& func = model.lambda_func(node.lambda_func_id);
            j["table"] = json::array();
            for (int i = 0; i < n; ++i) {
                j["table"].push_back(func(i));
            }
            j["children"] = json::array();
            j["children"].push_back(child_name(child_ref));
        } else {
            j["children"] = json::array();
            for (const auto& ch : node.children) {
                j["children"].push_back(child_name(ch));
            }
        }
        out << j.dump() << '\n';
    }

    // Write constraints
    for (int32_t cid : model.constraint_ids()) {
        json j;
        j["constraint"] = node_names[cid];
        out << j.dump() << '\n';
    }

    // Write objective
    if (model.objective_id() >= 0) {
        json j;
        j["minimize"] = node_names[model.objective_id()];
        out << j.dump() << '\n';
    }
}

void save_model(const Model& model, const std::string& path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::invalid_argument("cannot open file for writing: " + path);
    }
    save_model(model, file);
}

}  // namespace cbls
