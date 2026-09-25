#include "cbls/dag_ops.h"

#include "cbls/model.h"

#include <algorithm>
#include <vector>

namespace cbls {

namespace detail {

// Kahn's algorithm over the node-to-node edges only (variable children are
// sources and carry no in-degree). Children come out before their parents.
//
// Walks the back-references `Model::rebuild_back_references` has just built,
// rather than a child->parents adjacency of its own: that was one more vector per
// node, allocated on each of the two sorts per build. Those lists hold each
// parent once, where the old adjacency listed a parent naming the same child
// twice (`prod(n, n)`) twice and counted both edges into its in-degree. The
// order is unchanged: those two entries sat next to each other, a parent's
// in-degree could only reach zero on the second, and nothing was queued between
// them -- so a parent is queued at the same point either way.
//
// `sorted` is its own FIFO: entries are appended at the back and consumed from
// `head`, the order a std::queue gave without the deque's block allocations.
std::vector<int32_t> compute_topo_order(const Model& model) {
    const auto n = static_cast<int32_t>(model.num_nodes());
    std::vector<int32_t> in_degree(static_cast<size_t>(n), 0);
    for (int32_t nid = 0; nid < n; ++nid) {
        for (const int32_t parent_id : model.parents(nid)) {
            ++in_degree[parent_id];
        }
    }

    std::vector<int32_t> sorted;
    sorted.reserve(static_cast<size_t>(n));
    for (int32_t nid = 0; nid < n; ++nid) {
        if (in_degree[nid] == 0) {
            sorted.push_back(nid);
        }
    }
    for (size_t head = 0; head < sorted.size(); ++head) {
        for (const int32_t parent_id : model.parents(sorted[head])) {
            if (--in_degree[parent_id] == 0) {
                sorted.push_back(parent_id);
            }
        }
    }
    return sorted;
}

}  // namespace detail

double full_evaluate(Model& model) {
    for (int32_t nid : model.topo_order()) {
        model.set_node_value_unchecked(nid, evaluate(model.nodes()[nid], model));
    }
    if (model.objective_id() >= 0) {
        return model.node_values()[model.objective_id()];
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
            model.set_node_value_unchecked(nid, evaluate(model.nodes()[nid], model));
        }
        return;
    }
    for (int32_t nid : model.topo_order()) {
        if (dirty_flags[nid] != 0) {
            model.set_node_value_unchecked(nid, evaluate(model.nodes()[nid], model));
        }
    }
}

}  // namespace

double delta_evaluate(Model& model, const int32_t* changed_var_ids, size_t count) {
    if (count == 0) {
        if (model.objective_id() >= 0) {
            return model.node_values()[model.objective_id()];
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
        for (const int32_t dep_id : model.dependents(changed_var_ids[ci])) {
            if (dirty_flags[dep_id] == 0) {
                dirty_flags[dep_id] = 1;
                dirty_list.push_back(dep_id);
            }
        }
    }

    // BFS upward through parents
    for (size_t i = 0; i < dirty_list.size(); ++i) {
        int32_t nid = dirty_list[i];
        for (const int32_t parent_id : model.parents(nid)) {
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
        return model.node_values()[model.objective_id()];
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
        const ConstSpan<ChildRef> children = model.children(nd);
        for (int i = 0; i < static_cast<int>(children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const ChildRef& child = children[i];
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
        const ConstSpan<ChildRef> children = model.children(nd);
        for (int i = 0; i < static_cast<int>(children.size()); ++i) {
            double ld = local_derivative(nd, i, model);
            const ChildRef& child = children[i];
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
