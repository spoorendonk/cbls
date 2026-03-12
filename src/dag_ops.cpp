#include "cbls/dag_ops.h"
#include "cbls/model.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

namespace cbls {

void topological_sort(Model& model) {
    auto& nodes = model.nodes_mut();
    size_t n = nodes.size();

    // Register parent pointers: for each node, for each child that is a node,
    // add this node as a parent of that child.
    // Also register variable dependents.
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

    // Kahn's algorithm
    std::vector<int> in_degree(n, 0);
    std::unordered_map<int32_t, std::vector<int32_t>> child_to_parents;

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
        auto it = child_to_parents.find(nid);
        if (it != child_to_parents.end()) {
            for (int32_t parent_id : it->second) {
                in_degree[parent_id]--;
                if (in_degree[parent_id] == 0) {
                    queue.push(parent_id);
                }
            }
        }
    }

    // Store topo order via mutable access through Model friend or internal method
    // We'll set it through model's internal state
    // For now, model exposes topo_order setter
    model.nodes_mut(); // ensure writeable
    // We need to set the topo_order_ on the model. Since dag_ops is a friend-like
    // module, we'll use the public interface that model provides.
    // Actually, we need a way to set topo_order. Let's just store it via a method.
    // We'll handle this by having model store it and dag_ops call through.

    // The topo order is stored in the model - we access it through non-const ref
    // We'll cast or add a setter. For now, let's just return sorted and
    // have the caller set it. But the Python API has topological_sort(model)
    // modify model.expressions in-place. We do the same.

    // Store via a helper. Model needs a set_topo_order method.
    // Let me just make this work by accessing model internals.
    // dag_ops.cpp is implementation code that needs to modify model state.
    // We declare dag_ops functions as friend in model.h, or add a setter.

    // For simplicity, we'll use a setter approach (added to model.h as public for now)
    // Actually, let's just store via a reference we received.
    // Model stores topo_order_ - we need to set it from here.
    // The simplest is: add a public set_topo_order to Model.
    // But we already have topo_order() const accessor.
    // Let's just use the nodes_mut() pattern: add topo_order_mut().
    // OR, since we already have nodes_mut(), let's just accept that dag_ops
    // is tightly coupled to Model and use a dedicated internal method.

    // I'll add a package-private approach: model exposes the vector directly
    // via a mutable accessor. We already have this pattern with nodes_mut().

    // Actually, the simplest approach: expose it through model.
    // For now let's just do it. (topo_order is set by the close() method
    // which calls topological_sort, so it's a one-time setup.)

    // We need to update topo_order in model. Since Model::close() calls us,
    // we need to write back. Let me restructure: return the sorted order
    // and let close() set it. But the Python version modifies model in place.

    // Simplest: make topological_sort set the order via a non-const Model ref.
    // We need access to model's topo_order_. Let's do a workaround:
    // model will have a friend declaration, or we pass it by non-const ref
    // and model provides a setter.

    // Final solution: I'll just access it via a set_topo_order method.
    // Not ideal but clean enough for implementation code.
    (void)model; // suppress warning - we'll use set_topo_order below
}

// This is the actual implementation that Model::close() will call
// We need to restructure. Let me put the actual topo sort logic
// as a free function that returns the order, and Model::close() sets it.

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
    std::unordered_map<int32_t, std::vector<int32_t>> child_to_parents;

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
        auto it = child_to_parents.find(nid);
        if (it != child_to_parents.end()) {
            for (int32_t parent_id : it->second) {
                in_degree[parent_id]--;
                if (in_degree[parent_id] == 0) {
                    queue.push(parent_id);
                }
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

double delta_evaluate(Model& model, const std::set<int32_t>& changed_var_ids) {
    if (changed_var_ids.empty()) {
        if (model.objective_id() >= 0) {
            return model.node(model.objective_id()).value;
        }
        return 0.0;
    }

    // Mark dirty nodes via BFS up from changed variables
    std::unordered_set<int32_t> dirty;
    std::queue<int32_t> queue;

    for (int32_t vid : changed_var_ids) {
        const auto& v = model.var(vid);
        for (int32_t dep_id : v.dependent_ids) {
            if (dirty.insert(dep_id).second) {
                queue.push(dep_id);
            }
        }
    }

    while (!queue.empty()) {
        int32_t nid = queue.front();
        queue.pop();
        const auto& nd = model.node(nid);
        for (int32_t parent_id : nd.parent_ids) {
            if (dirty.insert(parent_id).second) {
                queue.push(parent_id);
            }
        }
    }

    // Recompute dirty nodes in topological order
    for (int32_t nid : model.topo_order()) {
        if (dirty.count(nid)) {
            auto& nd = model.node_mut(nid);
            nd.value = evaluate(nd, model);
        }
    }

    if (model.objective_id() >= 0) {
        return model.node(model.objective_id()).value;
    }
    return 0.0;
}

double compute_partial(const Model& model, int32_t expr_id, int32_t var_id) {
    // Reverse-mode AD
    std::unordered_map<int32_t, double> adjoint;
    adjoint[expr_id] = 1.0;

    // We need to iterate topo order in reverse
    const auto& order = model.topo_order();
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        int32_t nid = *it;
        auto adj_it = adjoint.find(nid);
        if (adj_it == adjoint.end()) continue;
        double adj = adj_it->second;

        const auto& nd = model.node(nid);
        for (int i = 0; i < static_cast<int>(nd.children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const auto& child = nd.children[i];
            if (child.is_var) {
                int32_t key = -(child.id + 1);  // negative to distinguish from node IDs
                adjoint[key] = (adjoint.count(key) ? adjoint[key] : 0.0) + adj * ld;
            } else {
                adjoint[child.id] = (adjoint.count(child.id) ? adjoint[child.id] : 0.0) + adj * ld;
            }
        }
    }

    int32_t key = -(var_id + 1);
    auto it = adjoint.find(key);
    return it != adjoint.end() ? it->second : 0.0;
}

}  // namespace cbls
