#pragma once

#include "dag.h"

#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <vector>

namespace cbls {

// Forward declare Expr
class Expr;

struct VarSequence {
    std::vector<int32_t> var_ids;  // ordered variable IDs in this sequence
    int min_block_on = 1;          // minimum consecutive vars to set to 1
    int min_block_off = 1;         // minimum consecutive vars to set to 0
};

/// Convert variable handle (negative, from int_var/float_var/etc.)
/// to var ID (non-negative, for model.var()/model.var_mut()).
inline int32_t handle_to_var_id(int32_t handle) {
    return -(handle + 1);
}

class Model {
public:
    Model() = default;

    // Variable creation — returns var ID
    int32_t bool_var(const std::string& name = "");
    int32_t int_var(int lb, int ub, const std::string& name = "");
    int32_t float_var(double lb, double ub, const std::string& name = "");
    int32_t list_var(int n, const std::string& name = "");
    int32_t set_var(int n, int min_size = 0, int max_size = -1, const std::string& name = "");

    // Expression creation — returns node ID
    int32_t constant(double val);
    int32_t neg(int32_t x);
    int32_t sum(const std::vector<int32_t>& args);
    int32_t prod(int32_t a, int32_t b);
    int32_t div_expr(int32_t a, int32_t b);
    int32_t pow_expr(int32_t base, int32_t exp);
    int32_t min_expr(const std::vector<int32_t>& args);
    int32_t max_expr(const std::vector<int32_t>& args);
    int32_t abs_expr(int32_t x);
    int32_t sin_expr(int32_t x);
    int32_t cos_expr(int32_t x);
    int32_t tan_expr(int32_t x);
    int32_t exp_expr(int32_t x);
    int32_t log_expr(int32_t x);
    int32_t sqrt_expr(int32_t x);
    // sign(base) * |base|^exp (AMPL/MINLPLib signpower). exp is typically a
    // constant node giving the power.
    int32_t signpower_expr(int32_t base, int32_t exp);
    int32_t tanh_expr(int32_t x);
    int32_t if_then_else(int32_t cond, int32_t then_, int32_t else_);
    int32_t at(int32_t list_var_id, int32_t index_expr);
    int32_t count(int32_t var_id);
    int32_t leq(int32_t a, int32_t b);
    int32_t eq_expr(int32_t a, int32_t b);
    int32_t geq(int32_t a, int32_t b);
    int32_t neq(int32_t a, int32_t b);
    int32_t lt(int32_t a, int32_t b);
    int32_t gt(int32_t a, int32_t b);
    int32_t lambda_sum(int32_t list_var_id, std::function<double(int)> func);
    int32_t pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func);

    void add_constraint(int32_t expr_id);
    void minimize(int32_t expr_id);
    void maximize(int32_t expr_id);

    // Expr-returning variable creation
    Expr Bool(const std::string& name = "");
    Expr Int(int lb, int ub, const std::string& name = "");
    Expr Float(double lb, double ub, const std::string& name = "");
    Expr List(int n, const std::string& name = "");
    Expr Set(int n, int min_size = 0, int max_size = -1, const std::string& name = "");
    Expr Constant(double val);

    // Overloaded constraint/objective accepting Expr
    void add_constraint(const Expr& e);
    void minimize(const Expr& e);
    void maximize(const Expr& e);

    // Variable sequences for block moves
    void add_var_sequence(const std::vector<int32_t>& var_ids, int min_block_on = 1,
                          int min_block_off = 1);
    [[nodiscard]] const std::vector<VarSequence>& var_sequences() const noexcept {
        return var_sequences_;
    }
    // Returns (seq_index, position) or (-1, -1) if not in any sequence
    [[nodiscard]] std::pair<int, int> var_sequence_for(int32_t var_id) const;

    void close();

    // ViolationLS objective-as-soft-constraint (paper §5, P2 #67). Folds the
    // objective into the constraint set as `objective_expr <= bound`, with the
    // bound a mutable RHS. Must be called after close() and only when an
    // objective is set; re-runs the topological sort and adjacency. The bound
    // starts at +inf (the constraint is inert until tightened), so search drives
    // the objective down by tightening it on each new feasible solution.
    void add_objective_soft_constraint();
    [[nodiscard]] bool has_objective_constraint() const noexcept {
        return objective_constraint_idx_ >= 0;
    }
    // Index of the objective constraint in constraint_ids(), or -1.
    [[nodiscard]] int32_t objective_constraint_idx() const noexcept {
        return objective_constraint_idx_;
    }
    // Tighten/relax the objective bound (RHS). Recomputes the objective
    // constraint node in place; caller invalidates any violation cache.
    void set_objective_bound(double bound);
    [[nodiscard]] double objective_bound() const noexcept { return objective_bound_; }

    // Accessors
    // Constraints (by index into constraint_ids()) that variable var_id can
    // affect. This is the paper's G_v, in ascending constraint index. Built by
    // close() and add_objective_soft_constraint() for the variables that existed
    // then; any other id -- every id, before close() -- is out of range.
    [[nodiscard]] ConstSpan<int32_t> constraints_of_var(int32_t var_id) const {
        if (var_id < 0 || static_cast<size_t>(var_id) + 1 >= var_constraint_offsets_.size()) {
            throw std::out_of_range("var id out of range");
        }
        const uint32_t begin = var_constraint_offsets_[var_id];
        return {var_constraint_ids_.data() + begin, var_constraint_offsets_[var_id + 1] - begin};
    }

    // Sparse per-constraint violation deltas if var_id <- j, WITHOUT committing.
    // Returns (constraint_index, delta) pairs for affected constraints whose
    // violation changes. Scalar variables only (Bool/Int/Float); throws on
    // List/Set. Does not clamp j to [lb, ub] — it is a pure counterfactual.
    // PRECONDITION: node values are consistent with the current assignment
    // (true after close(), full_evaluate(), or a committed move). The probe
    // restores exactly to that consistent state; it does not snapshot a dirty
    // mid-move state.
    std::vector<std::pair<int32_t, double>> per_constraint_violation_delta(int32_t var_id,
                                                                           double j);

    // Change in total WEIGHTED violation if var_id <- j: sum_c weights[c]*delta_c,
    // accumulated PER CONSTRAINT rather than as a difference of two whole sums, so
    // a row clamped to kInfPenalty cancels exactly instead of absorbing the O(1)
    // real rows (#100 — see the comment on the definition). Uses a member scratch
    // buffer, so it is allocation-free once warmed up but not on the first calls;
    // the per_constraint variant allocates on every call and is for sparse/tooling
    // use. `weights` is indexed by constraint index (constraint_ids()). Scalar
    // variables only; same no-commit / precondition contract as above.
    double weighted_violation_delta(int32_t var_id, double j, const std::vector<double>& weights);

    [[nodiscard]] const Variable& var(int32_t id) const {
        if (id < 0 || id >= static_cast<int32_t>(vars_.size())) {
            throw std::out_of_range("var id out of range");
        }
        return vars_[id];
    }
    Variable& var_mut(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(vars_.size())) {
            throw std::out_of_range("var id out of range");
        }
        return vars_[id];
    }
    [[nodiscard]] const ExprNode& node(int32_t id) const {
        if (id < 0 || id >= static_cast<int32_t>(nodes_.size())) {
            throw std::out_of_range("node id out of range");
        }
        return nodes_[id];
    }
    ExprNode& node_mut(int32_t id) {
        if (id < 0 || id >= static_cast<int32_t>(nodes_.size())) {
            throw std::out_of_range("node id out of range");
        }
        return nodes_[id];
    }
    /// `node`'s children, in the order they were given when it was created.
    /// Valid from creation, not only after `close()`: a node's children are
    /// written once, when it is made, and never change.
    ///
    /// `node` must be one of THIS model's nodes (from `node()`, `nodes()` or
    /// `node_mut()`), unmodified in `child_begin`/`child_count`. Its offsets are
    /// read against this model's array unchecked -- this is the evaluation hot
    /// path -- so a node taken from a different Model reads whatever that range
    /// holds in this one.
    [[nodiscard]] ConstSpan<ChildRef> children(const ExprNode& node) const noexcept {
        return {child_refs_.data() + node.child_begin, node.child_count};
    }
    /// The distinct nodes that name node `id` as a child, in ascending id order,
    /// each listed once however many times it names `id` (`prod(n, n)`).
    /// Rebuilt by `close()` and `add_objective_soft_constraint()`; empty for a
    /// node created since the last rebuild, and for every node before the first.
    [[nodiscard]] ConstSpan<int32_t> parents(int32_t id) const {
        if (id < 0 || id >= static_cast<int32_t>(nodes_.size())) {
            throw std::out_of_range("node id out of range");
        }
        const uint32_t begin = parent_offsets_[id];
        return {parent_ids_.data() + begin, parent_offsets_[id + 1] - begin};
    }
    /// The distinct nodes that name variable `var_id` as a child, with the same
    /// order, dedup and rebuild contract as `parents`.
    [[nodiscard]] ConstSpan<int32_t> dependents(int32_t var_id) const {
        if (var_id < 0 || var_id >= static_cast<int32_t>(vars_.size())) {
            throw std::out_of_range("var id out of range");
        }
        const uint32_t begin = dependent_offsets_[var_id];
        return {dependent_ids_.data() + begin, dependent_offsets_[var_id + 1] - begin};
    }
    [[nodiscard]] int32_t objective_id() const noexcept { return objective_id_; }
    [[nodiscard]] bool is_maximizing() const noexcept { return is_maximizing_; }
    [[nodiscard]] const std::vector<int32_t>& constraint_ids() const noexcept {
        return constraint_ids_;
    }
    /// Size the variable, node and child-reference arrays up front, when the
    /// caller already knows how big the model will be. `n_child_refs` is the
    /// total number of children over all nodes, i.e. the DAG's edge count.
    ///
    /// This is not a micro-optimisation on a large model; it is what stops a
    /// multi-gigabyte array being copied to grow it. A reader that appends a
    /// node per matrix entry grows `nodes_` and `child_refs_` by doubling, and
    /// every doubling copies everything built so far. Building the largest
    /// MIPfeas instance allocated 6.9 GB cumulatively against a 3.3 GB peak when
    /// nodes still owned their child vectors; the difference was that copying.
    /// Over-reserving costs address space and nothing else, so an estimate that
    /// is merely close is worth making.
    void reserve(size_t n_vars, size_t n_nodes, size_t n_child_refs = 0) {
        vars_.reserve(n_vars);
        dependent_offsets_.reserve(n_vars + 1);
        nodes_.reserve(n_nodes);
        parent_offsets_.reserve(n_nodes + 1);
        child_refs_.reserve(n_child_refs);
    }

    [[nodiscard]] const std::vector<int32_t>& topo_order() const noexcept { return topo_order_; }
    /// Where `id` sits in `topo_order()`, so a caller holding a handful of nodes
    /// can put them in evaluation order without walking the whole order to find
    /// them. `delta_evaluate` is the caller that matters: scanning `topo_order()`
    /// and testing a flag made it O(all nodes) per call on a model whose dirty
    /// set is typically a few dozen -- ~2M nodes walked per move on the largest
    /// MIPfeas instance.
    [[nodiscard]] int32_t topo_position(int32_t id) const noexcept { return topo_pos_[id]; }
    [[nodiscard]] const std::vector<Variable>& variables() const noexcept { return vars_; }
    [[nodiscard]] const std::vector<ExprNode>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] size_t num_vars() const noexcept { return vars_.size(); }
    [[nodiscard]] size_t num_nodes() const noexcept { return nodes_.size(); }
    [[nodiscard]] bool is_closed() const noexcept { return closed_; }

    // Lambda function access
    [[nodiscard]] const std::function<double(int)>& lambda_func(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(lambda_funcs_.size())) {
            throw std::out_of_range("lambda func index out of range");
        }
        return lambda_funcs_[idx];
    }

    [[nodiscard]] const std::function<double(int, int)>& pair_lambda_func(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(pair_lambda_funcs_.size())) {
            throw std::out_of_range("pair lambda func index out of range");
        }
        return pair_lambda_funcs_[idx];
    }

    // State snapshot/restore
    struct State {
        std::vector<double> values;
        std::vector<std::vector<int32_t>> elements;
    };
    [[nodiscard]] State copy_state() const;
    void restore_state(const State& state);

private:
    std::vector<Variable> vars_;
    std::vector<ExprNode> nodes_;
    // The DAG's edges, flat (#156). Per-node and per-variable vectors made model
    // build a few small allocations per node -- 2.66M on atlanta-ip's 540k nodes
    // -- and a portfolio replica a deep copy of all of them.
    //
    // `child_refs_` is append-only: a node's children are written when the node
    // is made and addressed by its (child_begin, child_count), so they are
    // readable before close(). The two back-reference arrays are CSR, rebuilt
    // wholesale by `rebuild_back_references`: the parents of node `i` are
    // `parent_ids_[parent_offsets_[i] .. parent_offsets_[i + 1])`, and likewise
    // for variables. Once its owner array is non-empty, each offsets array holds
    // at least one entry more -- creating a node or variable appends an empty
    // range before the element itself -- so the accessors, which range-check the id against the
    // owner first, need no "not built yet" branch, and a node made after a rebuild reads as
    // parentless, exactly as it did when it owned an empty vector.
    //
    // That is also why Model hands out no mutable nodes_ or vars_ vector: a
    // node or variable that bypassed push_node/alloc_var would have no range.
    std::vector<ChildRef> child_refs_;
    std::vector<uint32_t> parent_offsets_;
    std::vector<int32_t> parent_ids_;
    std::vector<uint32_t> dependent_offsets_;
    std::vector<int32_t> dependent_ids_;
    std::vector<int32_t> topo_order_;
    /// Inverse of `topo_order_`: node id -> its index there. Rebuilt with it,
    /// and only ever read through `topo_position`.
    std::vector<int32_t> topo_pos_;
    std::vector<int32_t> constraint_ids_;
    // var_id -> constraint indices (G_v), CSR like the back-references above: a
    // per-variable vector here was the largest allocation site left once those
    // were flat. Empty until close(), and then sized for the variables of the
    // last build only -- unlike dependent_offsets_ it is NOT extended as variables
    // are made, which is what keeps constraints_of_var's range check the same.
    std::vector<uint32_t> var_constraint_offsets_;
    std::vector<int32_t> var_constraint_ids_;
    int32_t objective_id_ = -1;
    bool is_maximizing_ = false;
    int32_t objective_bound_node_ = -1;       // Const node holding the objective RHS
    int32_t objective_constraint_node_ = -1;  // the `obj - bound` node
    int32_t objective_constraint_idx_ = -1;   // its index in constraint_ids_
    double objective_bound_ = 0.0;
    std::vector<std::function<double(int)>> lambda_funcs_;
    std::vector<std::function<double(int, int)>> pair_lambda_funcs_;
    bool closed_ = false;
    std::vector<VarSequence> var_sequences_;
    std::vector<std::pair<int, int>> var_to_seq_;  // var_id -> (seq_idx, pos), resized lazily
    // Scratch for weighted_violation_delta's pre-probe violations. A member so
    // the hot scoring path allocates only until it reaches the widest variable's
    // constraint count. Not reentrant — same single-thread-per-Model contract as
    // the probe's transient node mutation.
    std::vector<double> probe_old_violation_;

    void build_var_constraints();
    void rebuild_back_references();
    void rebuild_topo_positions();
    int32_t alloc_var(VarType type, double lb, double ub, const std::string& name);
    int32_t alloc_node(NodeOp op, std::initializer_list<ChildRef> children);
    int32_t alloc_node_over_handles(NodeOp op, const std::vector<int32_t>& handles);
    int32_t push_node(NodeOp op, size_t child_begin);
    // Decode a var or node handle, throwing std::out_of_range if it names
    // nothing this model has made yet.
    [[nodiscard]] ChildRef wrap(int32_t handle) const;
};

}  // namespace cbls
