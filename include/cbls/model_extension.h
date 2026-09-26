#pragma once

#include "dag.h"
#include "model.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace cbls {

/// What `Model::extend` added, and everything the search state needs in order to
/// grow with the model rather than be rebuilt (#167).
///
/// Every id range is contiguous and at the END of its array, which is what makes
/// "keep the old entries, append the new ones" a correct update for every table
/// indexed by a variable id or a constraint index.
struct ExtensionResult {
    /// New variables are `[first_new_var, first_new_var + num_new_vars)`.
    int32_t first_new_var = 0;
    int32_t num_new_vars = 0;
    /// New DAG nodes are `[first_new_node, first_new_node + num_new_nodes)`.
    int32_t first_new_node = 0;
    int32_t num_new_nodes = 0;
    /// New constraints are the indices
    /// `[first_new_constraint, first_new_constraint + num_new_constraints)`
    /// into `Model::constraint_ids()`.
    int32_t first_new_constraint = 0;
    int32_t num_new_constraints = 0;
    /// Each new variable's starting value, parallel to
    /// `[first_new_var, ...)`. This is what `pad_state` writes into a state
    /// captured before the extension, so an old incumbent stays a valid restart
    /// point (and `Model::restore_state` keeps its exact-size check).
    std::vector<double> new_var_initial;
    /// Indices of EXISTING constraints whose body changed, because a term was
    /// appended to a `Sum` somewhere inside them. Ascending and distinct.
    ///
    /// Two things follow for a caller holding search state: the row's node value
    /// is not the one it was, and the row's variable set may have grown. Every
    /// row whose value can have changed is either here or in the new-constraint
    /// range -- an extension reaches an existing row ONLY through
    /// `ModelExtension::append_to_sum`.
    std::vector<int32_t> touched_constraints;
    /// Every (constraint index, variable id) incidence the extension ADDED to
    /// `Model::constraints_of_var`, sorted by constraint and then by variable,
    /// distinct. Covers new and existing rows and new and existing variables.
    ///
    /// This is the pairing `FeasibilityJump::on_extended` needs: its
    /// `vars_of_constraint_` is the transpose of G_v, and recovering the pairing
    /// from the two id ranges alone is not possible.
    std::vector<std::pair<int32_t, int32_t>> new_incidences;
    /// True when the topological order had to be recomputed from scratch rather
    /// than spliced. See `Model::extend` for the condition; it is reported so a
    /// caller measuring the cost can tell the two regimes apart.
    bool topo_order_rebuilt = false;

    [[nodiscard]] int32_t end_var() const noexcept { return first_new_var + num_new_vars; }
    [[nodiscard]] int32_t end_node() const noexcept { return first_new_node + num_new_nodes; }
    [[nodiscard]] int32_t end_constraint() const noexcept {
        return first_new_constraint + num_new_constraints;
    }
};

/// A staged set of additions to a CLOSED model: new variables, new expression
/// nodes, new constraints, and terms appended to existing `Sum` rows (#167).
///
/// Nothing here touches the model. The recording is replayed by
/// `Model::extend`, which is the only point at which the DAG, the
/// back-references, the topological order and G_v change -- so a caller holding
/// a `ConstSpan` into any of those arrays is safe right up to that call, and
/// holds nothing across it.
///
/// Handles are the model's own: a node handle is a node id, a variable handle is
/// `-(var_id + 1)`, and the handles this class returns are the ids the entities
/// will have after `extend`. So the recording can name existing variables and
/// nodes freely, which is the point -- a new column enters existing rows, and a
/// lazily separated cut is a new row over existing variables.
///
/// Because the handles are absolute, an extension is tied to the model it was
/// built against: `extend` throws if that model has grown in the meantime.
///
/// WHAT IT DELIBERATELY DOES NOT OFFER, and why:
///
///  - **List and Set variables.** A structured variable's starting assignment is
///    laid out by `initialize_structured_random` inside `solve()`, and a member
///    of a `ListPartition` additionally has a cover invariant that only the
///    partition's own moves maintain. Both live above this layer, so a new
///    structured variable would arrive with no assignment and no way to get one.
///    Nodes reading EXISTING List/Set variables (`at`, `count`) are fine and are
///    offered.
///  - **`lambda_sum` / `pair_lambda_sum`.** Their callables live in the SHARED
///    `ModelStructure`, and `Model::freeze`'s contract around concurrent
///    invocation is the reason to think twice before growing those tables from a
///    running search.
///  - **`custom()` (#166).** `Model::custom_invariant` is indexed by
///    `ExprNode::lambda_func_id` into a PER-MODEL table, which an extension
///    could keep in step (it is append-only, exactly like `node_values_`), but
///    the slot's invariant instance would then have no committed state and no
///    `evaluate()` to establish one short of a `full_evaluate` -- which is the
///    O(model) cost `extend` exists to avoid. Left to #168.
///
/// Each refusal throws `std::invalid_argument` with the reason, rather than
/// silently producing something the engine cannot initialise.
class ModelExtension {
public:
    /// `base` must be closed and not frozen, and must still be at these counts
    /// when `extend` is called.
    explicit ModelExtension(const Model& base);

    // ---- New variables (scalars only; see the class comment) ----
    int32_t bool_var(const std::string& name = "");
    int32_t int_var(int lb, int ub, const std::string& name = "");
    int32_t float_var(double lb, double ub, const std::string& name = "");

    /// The value the new variable starts at, and the value `pad_state` writes
    /// into a state captured before the extension.
    ///
    /// Defaults to the variable's lower bound, which is what `Model`'s own
    /// variable creation does -- NOT to 0, so that a staged model and the same
    /// model built whole start from the same assignment.
    ///
    /// Throws `std::invalid_argument` unless `var` is a variable this extension
    /// created and `value` is within its bounds.
    void set_initial(int32_t var, double value);

    // ---- Expressions, over existing or new handles ----
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

    /// Add a new constraint row. `expr` must be a node handle (new or existing).
    void add_constraint(int32_t expr);

    /// Append `term` to an EXISTING `Sum` node -- the core operation of column
    /// generation, since a new column enters rows that are already there.
    ///
    /// `sum_node` must be a `NodeOp::Sum` node of the base model; a node created
    /// by this extension is rejected, because its children are not written yet
    /// and `sum()` already takes the full list. Appending the same term twice
    /// adds it twice, exactly as `sum({t, t})` would.
    void append_to_sum(int32_t sum_node, int32_t term);

    [[nodiscard]] bool empty() const noexcept {
        return new_vars_.empty() && new_nodes_.empty() && new_constraints_.empty() &&
               appends_.empty();
    }
    [[nodiscard]] size_t num_new_vars() const noexcept { return new_vars_.size(); }
    [[nodiscard]] size_t num_new_nodes() const noexcept { return new_nodes_.size(); }

private:
    friend class Model;

    struct NewVar {
        VarType type = VarType::Float;
        double lb = 0.0;
        double ub = 0.0;
        double initial = 0.0;
        std::string name;
    };
    struct NewNode {
        NodeOp op = NodeOp::Const;
        double const_value = 0.0;
        std::vector<int32_t> children;  // handles, as given
    };

    /// Validate a handle against the base model plus what this extension has
    /// recorded so far, and return it unchanged. Throws `std::out_of_range` if
    /// it names nothing.
    int32_t check_handle(int32_t handle) const;
    int32_t check_node_handle(int32_t handle, const char* what) const;
    int32_t push(NodeOp op, std::vector<int32_t> children, double const_value = 0.0);
    int32_t add_var(VarType type, double lb, double ub, const std::string& name);

    const Model* base_;
    int32_t base_num_vars_ = 0;
    int32_t base_num_nodes_ = 0;
    std::vector<NewVar> new_vars_;
    std::vector<NewNode> new_nodes_;
    std::vector<int32_t> new_constraints_;              // node handles
    std::vector<std::pair<int32_t, int32_t>> appends_;  // (existing sum node, term handle)
};

/// Grow a `Model::State` captured BEFORE `ext` was applied so that
/// `Model::restore_state` accepts it again, giving each new variable the initial
/// value the extension declared (#167).
///
/// Deliberately strict: it throws `std::invalid_argument` unless `state` is
/// exactly the size the model had before the extension. Accepting anything
/// wider would turn `restore_state`'s size check -- which is what stops Python
/// handing the engine a short `ModelState` and getting an unguarded read -- into
/// a check on nothing.
void pad_state(Model::State& state, const ExtensionResult& ext);

}  // namespace cbls
