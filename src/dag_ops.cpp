#include "cbls/dag_ops.h"

#include "cbls/model.h"

#include <algorithm>
#include <queue>
#include <vector>

namespace cbls {

namespace detail {

namespace {

// Rebuild the DAG's back-references: every node's parent_ids and every
// variable's dependent_ids. These are what delta_evaluate walks to find the
// nodes a changed variable dirties; they are pure derived state, so they are
// cleared and recomputed wholesale rather than patched.
void rebuild_back_references(Model& model) {
    for (auto& nd : model.nodes_mut()) {
        nd.parent_ids.clear();
    }
    for (auto& v : model.variables_mut()) {
        v.dependent_ids.clear();
    }

    // Deduplicated by a last-writer stamp, not by searching the list being
    // built. A duplicate can only ever come from ONE parent naming the same
    // child twice (`prod(x, x)`), because a parent is visited once -- so
    // "already appended by this parent" is the whole condition, and a stamp
    // answers it in O(1) where the search was O(degree) per edge.
    //
    // That difference is not academic on a real matrix. The search made this
    // O(sum of degree^2): square47's 95k columns appear in ~288 rows each, which
    // is ~3.9 BILLION comparisons, and `compute_topo_order` runs twice per build
    // (once at close(), once when the objective's soft constraint is added). It
    // was 68% of that instance's 6.4s model build.
    std::vector<int32_t> var_stamp(model.num_vars(), -1);
    std::vector<int32_t> node_stamp(model.nodes().size(), -1);

    for (auto& nd : model.nodes_mut()) {
        for (const auto& child : nd.children) {
            if (child.is_var) {
                if (var_stamp[child.id] == nd.id) {
                    continue;
                }
                var_stamp[child.id] = nd.id;
                model.var_mut(child.id).dependent_ids.push_back(nd.id);
            } else {
                if (node_stamp[child.id] == nd.id) {
                    continue;
                }
                node_stamp[child.id] = nd.id;
                model.node_mut(child.id).parent_ids.push_back(nd.id);
            }
        }
    }
}

// Kahn's algorithm over the node-to-node edges only (variable children are
// sources and carry no in-degree). Children come out before their parents.
std::vector<int32_t> kahn_sort(const std::vector<ExprNode>& nodes) {
    size_t n = nodes.size();
    std::vector<int> in_degree(n, 0);
    // Use flat vector instead of unordered_map for child->parents
    std::vector<std::vector<int32_t>> child_to_parents(n);

    for (const auto& nd : nodes) {
        for (const auto& child : nd.children) {
            if (!child.is_var) {
                in_degree[nd.id]++;
                child_to_parents[child.id].push_back(nd.id);
            }
        }
    }

    std::queue<int32_t> queue;
    for (const auto& nd : nodes) {
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

}  // namespace

std::vector<int32_t> compute_topo_order(Model& model) {
    rebuild_back_references(model);
    return kahn_sort(model.nodes());
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

namespace {

// Recompute a marked dirty set in topological order, by whichever of the two
// routes is cheaper for THIS set. Its own function because it is a separate job
// from finding the set: the BFS above decides WHAT is stale, this decides how to
// walk it in dependency order.
//
// Sorting d entries costs O(d log d), with two scattered `topo_pos_` loads per
// comparison; scanning `topo_order()` and testing the flag costs O(|nodes|)
// sequential byte tests. Sorting wins by orders of magnitude in the regime that
// matters -- a few dozen dirty nodes against the 4.3M of the largest MIPfeas
// instance, where the scan was ~2M flag tests to recompute a handful, and delta
// evaluation was not sublinear in the model at all. It loses at the other end:
// as d approaches |nodes| the sort does ~log2(d) scattered comparisons per
// element where the scan does one sequential test, and a dense continuous model
// -- MINLPLib's regime, not this roster's -- can sit there.
//
// The condition is the cost model itself rather than a tuned constant: sort
// while d*log2(d) is under |nodes|, otherwise scan. Both routes produce the same
// order and the same values.
void evaluate_dirty_in_topo_order(Model& model, std::vector<int32_t>& dirty_list,
                                  const std::vector<uint8_t>& dirty_flags, size_t num_nodes) {
    const size_t dirty_count = dirty_list.size();
    size_t log2_dirty = 0;
    while ((size_t{1} << (log2_dirty + 1)) <= dirty_count) {
        ++log2_dirty;
    }
    if (dirty_count * (log2_dirty + 1) < num_nodes) {
        std::sort(dirty_list.begin(), dirty_list.end(), [&model](int32_t a, int32_t b) {
            return model.topo_position(a) < model.topo_position(b);
        });
        for (int32_t nid : dirty_list) {
            auto& nd = model.node_mut(nid);
            nd.value = evaluate(nd, model);
        }
        return;
    }
    for (int32_t nid : model.topo_order()) {
        if (dirty_flags[nid] != 0) {
            auto& nd = model.node_mut(nid);
            nd.value = evaluate(nd, model);
        }
    }
}

}  // namespace

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
            if (dirty_flags[dep_id] == 0) {
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
            if (dirty_flags[parent_id] == 0) {
                dirty_flags[parent_id] = 1;
                dirty_list.push_back(parent_id);
            }
        }
    }

    evaluate_dirty_in_topo_order(model, dirty_list, dirty_flags, num_nodes);

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
        if (adjoint[nid] == 0.0) {
            continue;
        }
        double adj = adjoint[nid];

        const auto& nd = model.node(nid);
        for (int i = 0; i < static_cast<int>(nd.children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const auto& child = nd.children[i];
            if (child.is_var) {
                int32_t key = static_cast<int32_t>(num_nodes) + child.id;
                if (adjoint[key] == 0.0) {
                    written.push_back(key);
                }
                adjoint[key] += adj * ld;
            } else {
                if (adjoint[child.id] == 0.0) {
                    written.push_back(child.id);
                }
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
        if (adjoint[nid] == 0.0) {
            continue;
        }
        double adj = adjoint[nid];

        const auto& nd = model.node(nid);
        for (int i = 0; i < static_cast<int>(nd.children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const auto& child = nd.children[i];
            if (child.is_var) {
                int32_t key = static_cast<int32_t>(num_nodes) + child.id;
                if (adjoint[key] == 0.0) {
                    written.push_back(key);
                }
                adjoint[key] += adj * ld;
            } else {
                if (adjoint[child.id] == 0.0) {
                    written.push_back(child.id);
                }
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
