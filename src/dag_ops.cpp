#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include <queue>
#include <algorithm>

namespace cbls {

namespace detail {

std::vector<int32_t> compute_topo_order(Model& model) {
    auto& nodes = model.nodes_mut();

    // Clear parent/dependent pointers and rebuild
    for (auto& nd : nodes) {
        nd.parent_ids.clear();
    }
    for (auto& v : model.variables_mut()) {
        v.dependent_ids.clear();
    }

    for (auto& nd : nodes) {
        for (const auto& child : nd.children) {
            if (child.is_var) {
                auto& v = model.var_mut(child.id);
                if (std::find(v.dependent_ids.begin(), v.dependent_ids.end(), nd.id) == v.dependent_ids.end()) {
                    v.dependent_ids.push_back(nd.id);
                }
            } else {
                auto& child_node = model.node_mut(child.id);
                if (std::find(child_node.parent_ids.begin(), child_node.parent_ids.end(), nd.id) == child_node.parent_ids.end()) {
                    child_node.parent_ids.push_back(nd.id);
                }
            }
        }
    }

    size_t n = nodes.size();
    std::vector<int> in_degree(n, 0);
    // Use flat vector instead of unordered_map for child->parents
    std::vector<std::vector<int32_t>> child_to_parents(n);

    for (auto& nd : nodes) {
        for (const auto& child : nd.children) {
            if (!child.is_var) {
                in_degree[nd.id]++;
                child_to_parents[child.id].push_back(nd.id);
            }
        }
    }

    std::queue<int32_t> queue;
    for (auto& nd : nodes) {
        if (in_degree[nd.id] == 0) {
            queue.push(nd.id);
        }
    }

    std::vector<int32_t> sorted;
    sorted.reserve(n);
    while (!queue.empty()) {
        int32_t nid = queue.front();
        queue.pop();
        sorted.push_back(nid);
        for (int32_t parent_id : child_to_parents[nid]) {
            in_degree[parent_id]--;
            if (in_degree[parent_id] == 0) {
                queue.push(parent_id);
            }
        }
    }

    return sorted;
}

}  // namespace detail

double full_evaluate(Model& model) {
    for (int32_t nid : model.topo_order()) {
        auto& nd = model.node_mut(nid);
        nd.value = evaluate(nd, model);
    }
    if (model.objective_id() >= 0) {
        return model.node(model.objective_id()).value;
    }
    return 0.0;
}

double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count) {
    if (count == 0) {
        if (model.objective_id() >= 0) {
            return model.node(model.objective_id()).value;
        }
        return 0.0;
    }

    const size_t num_nodes = model.num_nodes();

    // Flat dirty flags + dirty list for O(dirty) cleanup
    // Use thread_local to avoid reallocation across calls
    thread_local std::vector<uint8_t> dirty_flags;
    thread_local std::vector<int32_t> dirty_list;

    if (dirty_flags.size() < num_nodes) {
        dirty_flags.resize(num_nodes, 0);
    }
    dirty_list.clear();

    // Seed dirty set from changed variables' dependents
    for (size_t ci = 0; ci < count; ++ci) {
        const auto& v = model.var(changed_var_ids[ci]);
        for (int32_t dep_id : v.dependent_ids) {
            if (!dirty_flags[dep_id]) {
                dirty_flags[dep_id] = 1;
                dirty_list.push_back(dep_id);
            }
        }
    }

    // BFS upward through parents
    for (size_t i = 0; i < dirty_list.size(); ++i) {
        int32_t nid = dirty_list[i];
        const auto& nd = model.node(nid);
        for (int32_t parent_id : nd.parent_ids) {
            if (!dirty_flags[parent_id]) {
                dirty_flags[parent_id] = 1;
                dirty_list.push_back(parent_id);
            }
        }
    }

    // Recompute dirty nodes in topological order
    for (int32_t nid : model.topo_order()) {
        if (dirty_flags[nid]) {
            auto& nd = model.node_mut(nid);
            nd.value = evaluate(nd, model);
        }
    }

    // Clean up dirty flags (only touch entries we set)
    for (int32_t nid : dirty_list) {
        dirty_flags[nid] = 0;
    }

    if (model.objective_id() >= 0) {
        return model.node(model.objective_id()).value;
    }
    return 0.0;
}

// Sparse reverse-mode AD: only visit ancestors of expr_id
double compute_partial(const Model& model, int32_t expr_id, int32_t var_id) {
    const size_t num_nodes = model.num_nodes();
    const size_t num_vars = model.num_vars();

    // Flat adjoint vector: [0..num_nodes-1] for nodes, [num_nodes..num_nodes+num_vars-1] for vars
    thread_local std::vector<double> adjoint;
    thread_local std::vector<int32_t> written;  // dirty list for cleanup

    const size_t total_size = num_nodes + num_vars;
    if (adjoint.size() < total_size) {
        adjoint.resize(total_size, 0.0);
    }
    written.clear();

    adjoint[expr_id] = 1.0;
    written.push_back(expr_id);

    // Find ancestors of expr_id by walking topo_order in reverse,
    // only visiting nodes that have nonzero adjoint (i.e., are reachable from expr_id)
    const auto& order = model.topo_order();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        int32_t nid = *it;
        if (adjoint[nid] == 0.0) continue;
        double adj = adjoint[nid];

        const auto& nd = model.node(nid);
        for (int i = 0; i < static_cast<int>(nd.children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const auto& child = nd.children[i];
            if (child.is_var) {
                int32_t key = static_cast<int32_t>(num_nodes) + child.id;
                if (adjoint[key] == 0.0) written.push_back(key);
                adjoint[key] += adj * ld;
            } else {
                if (adjoint[child.id] == 0.0) written.push_back(child.id);
                adjoint[child.id] += adj * ld;
            }
        }
    }

    int32_t key = static_cast<int32_t>(num_nodes) + var_id;
    double result = (key < static_cast<int32_t>(adjoint.size())) ? adjoint[key] : 0.0;

    // Clean up only entries we wrote
    for (int32_t idx : written) {
        adjoint[idx] = 0.0;
    }

    return result;
}

// Batch AD: one reverse pass computing ∂expr/∂(all vars)
std::vector<double> compute_all_partials(const Model& model, int32_t expr_id) {
    const size_t num_nodes = model.num_nodes();
    const size_t num_vars = model.num_vars();

    thread_local std::vector<double> adjoint;
    thread_local std::vector<int32_t> written;

    const size_t total_size = num_nodes + num_vars;
    if (adjoint.size() < total_size) {
        adjoint.resize(total_size, 0.0);
    }
    written.clear();

    adjoint[expr_id] = 1.0;
    written.push_back(expr_id);

    const auto& order = model.topo_order();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        int32_t nid = *it;
        if (adjoint[nid] == 0.0) continue;
        double adj = adjoint[nid];

        const auto& nd = model.node(nid);
        for (int i = 0; i < static_cast<int>(nd.children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const auto& child = nd.children[i];
            if (child.is_var) {
                int32_t key = static_cast<int32_t>(num_nodes) + child.id;
                if (adjoint[key] == 0.0) written.push_back(key);
                adjoint[key] += adj * ld;
            } else {
                if (adjoint[child.id] == 0.0) written.push_back(child.id);
                adjoint[child.id] += adj * ld;
            }
        }
    }

    // Extract var partials
    std::vector<double> partials(num_vars);
    for (size_t i = 0; i < num_vars; ++i) {
        partials[i] = adjoint[num_nodes + i];
    }

    // Clean up
    for (int32_t idx : written) {
        adjoint[idx] = 0.0;
    }

    return partials;
}

}  // namespace cbls
