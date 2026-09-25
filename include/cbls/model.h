#pragma once

#include "dag.h"

#include <cassert>
#include <functional>
#include <initializer_list>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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

/// Everything about a model that no search writes: the DAG's nodes and edges,
/// the derived indices over them, the constraint list, the lambda tables and the
/// variable sequences.
///
/// It is a type of its own so that a FROZEN model can hand it to every portfolio
/// worker by reference instead of being deep-copied per worker (#157). On the
/// largest MIPfeas instance that copy was the whole of the portfolio's memory
/// growth: ~0.8 GiB per extra worker, which is what capped the thread count on
/// ordinary hardware rather than the core count.
///
/// Reached only through `Model`'s accessors, and after `Model::freeze()` only
/// through a `shared_ptr<const ModelStructure>` -- so a write to it from a search
/// path does not compile rather than corrupting a peer's search.
///
/// What is deliberately NOT here: every node's current value (`Model::
/// node_values_`), every variable (`Model::vars_` -- see `Variable`, kept whole
/// and per worker), the objective bound and the delta probe's scratch. Those are
/// what a search writes. The structural SCALARS -- `objective_id_`,
/// `is_maximizing_`, the three objective-row node ids, `closed_` -- stay in
/// `Model` too: they are immutable after `freeze()` as well, but twenty bytes per
/// worker is not worth an indirection on the paths that read them.
struct ModelStructure {
    std::vector<ExprNode> nodes;
    // The DAG's edges, flat (#156). Per-node and per-variable vectors made model
    // build a few small allocations per node -- 2.66M on atlanta-ip's 540k nodes
    // -- and a portfolio replica a deep copy of all of them.
    //
    // `child_refs` is append-only: a node's children are written when the node
    // is made and addressed by its (child_begin, child_count), so they are
    // readable before close(). The two back-reference arrays are CSR, rebuilt
    // wholesale by `Model::rebuild_back_references`: the parents of node `i` are
    // `parent_ids[parent_offsets[i] .. parent_offsets[i + 1])`, and likewise
    // for variables. Once its owner array is non-empty, each offsets array holds
    // at least one entry more -- creating a node or variable appends an empty
    // range before the element itself -- so the accessors, which range-check the id against the
    // owner first, need no "not built yet" branch, and a node made after a rebuild reads as
    // parentless, exactly as it did when it owned an empty vector.
    //
    // That is also why Model hands out no mutable nodes or vars vector: a
    // node or variable that bypassed push_node/alloc_var would have no range.
    std::vector<ChildRef> child_refs;
    std::vector<uint32_t> parent_offsets;
    std::vector<int32_t> parent_ids;
    std::vector<uint32_t> dependent_offsets;
    std::vector<int32_t> dependent_ids;
    std::vector<int32_t> topo_order;
    /// Inverse of `topo_order`: node id -> its index there. Rebuilt with it,
    /// and only ever read through `Model::topo_position`.
    std::vector<int32_t> topo_pos;
    std::vector<int32_t> constraint_ids;
    // var_id -> constraint indices (G_v), CSR like the back-references above: a
    // per-variable vector here was the largest allocation site left once those
    // were flat. Empty until close(), and then sized for the variables of the
    // last build only -- unlike dependent_offsets it is NOT extended as variables
    // are made, which is what keeps constraints_of_var's range check the same.
    std::vector<uint32_t> var_constraint_offsets;
    std::vector<int32_t> var_constraint_ids;
    // Shared, and therefore invoked by every worker CONCURRENTLY once a model is
    // frozen and replicated. A callable carrying mutable state of its own is a
    // race: a NEW one where that state is captured by value, since each replica
    // used to deep-copy the `std::function`, and an old one where it is captured
    // by reference. `Model::freeze` says so where a caller will read it.
    std::vector<std::function<double(int)>> lambda_funcs;
    std::vector<std::function<double(int, int)>> pair_lambda_funcs;
    // Parallel to `pair_lambda_funcs` and the same length: entry i is the
    // closing rule and the fixed-endpoint terms of the node whose
    // `lambda_func_id` is i. See `PairLambdaSpec` in dag.h for why this is a
    // side table rather than more `NodeOp` enumerators.
    std::vector<PairLambdaSpec> pair_lambda_specs;
    std::vector<VarSequence> var_sequences;
    std::vector<std::pair<int, int>> var_to_seq;  // var_id -> (seq_idx, pos), resized lazily
};

class Model {
public:
    Model();

    /// Copying a model copies its per-model state -- the variables, the node
    /// values -- and then either SHARES or deep-copies the structure:
    ///
    ///  - a FROZEN model shares it, which is what makes a portfolio replica
    ///    cheap and is the point of `freeze()`. The copy is frozen too, so it
    ///    cannot change what its peers read.
    ///  - an OPEN model deep-copies it, exactly as it always did, so a model
    ///    still being built can be copied and then extended independently.
    ///
    /// The delta probe's scratch buffer is not copied: it is overwritten before
    /// it is read on every call, so its contents are not state.
    /// Every member of this class must be handled in the copy constructor -- it
    /// enumerates them explicitly, where the implicit one it replaced covered them
    /// all. A member added later and forgotten there is default-initialised in
    /// every portfolio replica while the master looks right, which shows up only
    /// as a trajectory divergence.
    Model(const Model& other);
    Model& operator=(const Model& other);
    /// A moved-from `Model` may only be destroyed or assigned to. It holds no
    /// structure at all, so every accessor that reads the structure is undefined
    /// on it -- unlike before the split, where it read as an empty model. `s()`
    /// asserts, so a Debug or sanitizer build names it. Note in particular that
    /// `is_frozen()` reports TRUE for one, so `if (!m.is_frozen()) m.freeze();`
    /// silently does nothing.
    Model(Model&&) noexcept = default;
    Model& operator=(Model&&) noexcept = default;
    ~Model() = default;

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

    /// Sum `func` over the consecutive pairs of a List (or Set) variable's
    /// elements, optionally closing the chain and optionally charging the first
    /// and last element against a fixed endpoint.
    ///
    /// With `e` the variable's `elements` and `n == e.size()`:
    ///
    ///     head(e_0) + sum_{k < n-1} func(e_k, e_{k+1}) + tail(e_{n-1})
    ///                + [mode == Cyclic && n >= 2] func(e_{n-1}, e_0)
    ///
    /// The short cases are DEFINED, not edge cases -- a List whose length
    /// varies makes them routine:
    ///
    ///  - `n == 0`: 0.0 for every variant, head and tail included. An unused
    ///    route costs nothing.
    ///  - `n == 1`: `head(e_0) + tail(e_0)`. There is no pair, and `Cyclic`
    ///    adds nothing: a single element is not a pair with itself.
    ///  - `n == 2`: `head(e_0) + func(e_0, e_1) + tail(e_1)`, and `Cyclic` adds
    ///    `func(e_1, e_0)` -- the two-city tour traverses its one edge twice,
    ///    which is the convention a distance matrix that need not be symmetric
    ///    requires.
    ///
    /// `Cyclic` is the tour cost of a TSP over one List. `head`/`tail` are the
    /// depot legs of a CVRP route, where a cyclic sum over the customers alone
    /// would wrongly add `func(e_{n-1}, e_0)`; the two are independent, so a
    /// cyclic sum with endpoint terms is expressible even though no use case
    /// here asks for it.
    ///
    /// A `Set` variable is accepted, but its `elements` carry no modelled
    /// order: the sum then reads whatever order they happen to be stored in,
    /// which the search moves arbitrarily. Consecutive pairs are a meaningful
    /// cost on a `List`.
    ///
    /// `func`, `head` and `tail` are invoked by EVERY portfolio worker
    /// concurrently once the model is frozen -- see `freeze()`.
    int32_t pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func,
                            PairMode mode = PairMode::Open);
    /// The fixed-endpoint form. Either callable may be empty, meaning "no term";
    /// passing both empty is the plain `mode` overload.
    int32_t pair_lambda_sum(int32_t list_var_id, std::function<double(int, int)> func,
                            std::function<double(int)> head, std::function<double(int)> tail,
                            PairMode mode = PairMode::Open);

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
        return s().var_sequences;
    }
    // Returns (seq_index, position) or (-1, -1) if not in any sequence
    [[nodiscard]] std::pair<int, int> var_sequence_for(int32_t var_id) const;

    void close();

    /// Make the structure immutable, so that copies of this model can SHARE it
    /// instead of deep-copying the DAG per portfolio worker (#157).
    ///
    /// `close()`s the model if it is still open, folds the objective into the
    /// constraint set if there is one, and then drops the writable handle on the
    /// structure. Idempotent.
    ///
    /// The freeze point is deliberately AFTER the objective row, not at
    /// `close()`. `solve()` calls `add_objective_soft_constraint()` on whatever
    /// model it is handed, which appends two nodes and a constraint and rebuilds
    /// the back-references, the topological order, `topo_pos` and G_v -- so a
    /// structure frozen at `close()` would have N workers re-sorting one shared
    /// DAG concurrently.
    ///
    /// Every structural method throws afterwards: variable and expression
    /// creation, `add_constraint`, `minimize`/`maximize`, `add_var_sequence`,
    /// `reserve` and `close`. Not copy-on-write, deliberately -- a silent detach
    /// would put that worker back on its own full copy, with every test still
    /// green and the memory saving gone.
    ///
    /// **`add_objective_soft_constraint` is the one intentional exception**: on a
    /// frozen model it RETURNS, it does not throw. `solve()` calls it on every
    /// model it is handed, a frozen replica included, and `freeze()` has already
    /// run it -- so its idempotent early return has to come before the frozen
    /// check or no portfolio worker could start. The `require_open` that follows
    /// that return is therefore unreachable by construction and kept only as a
    /// backstop against a future caller reaching it another way. Do not "fix" the
    /// order; `tests/test_model_share.cpp` pins it.
    ///
    /// It also does NOT re-derive a structure that was extended after `close()`.
    /// `close(); add_constraint(...); freeze();` on an objective-free model freezes
    /// the back-references, topological order, `topo_pos` and G_v as `close()` left
    /// them, and then shares them -- that is `close()`'s pre-existing contract,
    /// not something `freeze()` repairs. Close last, or add the objective, which
    /// makes the objective-row rebuild cover it.
    ///
    /// What a frozen model can still do is everything a search does: assign
    /// variables, evaluate, snapshot and restore state, tighten and release the
    /// objective bound.
    ///
    /// Two consequences of sharing, both about the copies:
    ///
    ///  - a copy starts from THIS model's node values, verbatim, so it is already
    ///    consistent with its variables and needs no `full_evaluate`;
    ///  - the `lambda_sum`/`pair_lambda_sum` callables are shared, so several
    ///    workers invoke the same callable object at once. One that carries
    ///    mutable state of its own is a data race -- a NEW one where that state is
    ///    captured by value, since each replica used to deep-copy the
    ///    `std::function`, and an old one where it is captured by reference.
    void freeze();
    [[nodiscard]] bool is_frozen() const noexcept { return open_structure_ == nullptr; }

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
    //
    // Per-model state, NOT structure: each portfolio worker tightens its own
    // bound on its own incumbents, so the value lives here and `evaluate()`'s
    // `Const` arm reads it back through `objective_bound_node()` rather than from
    // the shared node's `const_value` (#157). Allowed on a frozen model --
    // tightening the bound is what a search does.
    void set_objective_bound(double bound);
    [[nodiscard]] double objective_bound() const noexcept { return objective_bound_; }
    /// The `Const` node carrying the objective row's RHS, or -1 when there is no
    /// objective row. `evaluate()` compares against this to know which `Const` to
    /// read `objective_bound()` for instead of `const_value`.
    [[nodiscard]] int32_t objective_bound_node() const noexcept { return objective_bound_node_; }

    // Accessors
    // Constraints (by index into constraint_ids()) that variable var_id can
    // affect. This is the paper's G_v, in ascending constraint index. Built by
    // close() and add_objective_soft_constraint() for the variables that existed
    // then; any other id -- every id, before close() -- is out of range.
    [[nodiscard]] ConstSpan<int32_t> constraints_of_var(int32_t var_id) const {
        const ModelStructure& st = s();
        if (var_id < 0 || static_cast<size_t>(var_id) + 1 >= st.var_constraint_offsets.size()) {
            throw std::out_of_range("var id out of range");
        }
        const uint32_t begin = st.var_constraint_offsets[var_id];
        return {st.var_constraint_ids.data() + begin,
                st.var_constraint_offsets[var_id + 1] - begin};
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
        if (id < 0 || id >= static_cast<int32_t>(s().nodes.size())) {
            throw std::out_of_range("node id out of range");
        }
        return s().nodes[id];
    }
    /// Node `id`'s current value: the evaluation cache, which is per-model
    /// mutable state and therefore NOT part of the shared structure (#157).
    /// Range-checked, like `node()`.
    [[nodiscard]] double node_value(int32_t id) const {
        // Checked against the NODE array, not the value array: `push_node` appends
        // the value first, so a throwing node append can leave the value array one
        // entry longer, and that entry belongs to no node.
        if (id < 0 || id >= static_cast<int32_t>(s().nodes.size())) {
            throw std::out_of_range("node id out of range");
        }
        return node_values_[id];
    }
    /// Write node `id`'s value WITHOUT a range check -- the evaluation loops'
    /// write, paired with the unchecked reads `node_values()` hands out. Their
    /// ids come from `topo_order()`, so the check could only ever pass, and
    /// `full_evaluate` performs one write per node: 4.3M of them per call on the
    /// largest MIPfeas instance.
    ///
    /// Deliberately NOT bound to Python: an index supplied from there would be
    /// an unguarded heap write (#156). `node_value` is the checked reader.
    void set_node_value_unchecked(int32_t id, double value) noexcept { node_values_[id] = value; }
    /// `node`'s children, in the order they were given when it was created.
    /// Valid from creation, not only after `close()`: a node's children are
    /// written once, when it is made, and never change.
    ///
    /// `node` must be one of THIS model's nodes (from `node()` or `nodes()`),
    /// unmodified in `child_begin`/`child_count`. Its offsets are
    /// read against this model's array unchecked -- this is the evaluation hot
    /// path -- so a node taken from a different Model reads whatever that range
    /// holds in this one.
    [[nodiscard]] ConstSpan<ChildRef> children(const ExprNode& node) const noexcept {
        return {s().child_refs.data() + node.child_begin, node.child_count};
    }
    /// The distinct nodes that name node `id` as a child, in ascending id order,
    /// each listed once however many times it names `id` (`prod(n, n)`).
    /// Rebuilt by `close()` and `add_objective_soft_constraint()`; empty for a
    /// node created since the last rebuild, and for every node before the first.
    [[nodiscard]] ConstSpan<int32_t> parents(int32_t id) const {
        const ModelStructure& st = s();
        if (id < 0 || id >= static_cast<int32_t>(st.nodes.size())) {
            throw std::out_of_range("node id out of range");
        }
        const uint32_t begin = st.parent_offsets[id];
        return {st.parent_ids.data() + begin, st.parent_offsets[id + 1] - begin};
    }
    /// The distinct nodes that name variable `var_id` as a child, with the same
    /// order, dedup and rebuild contract as `parents`.
    [[nodiscard]] ConstSpan<int32_t> dependents(int32_t var_id) const {
        if (var_id < 0 || var_id >= static_cast<int32_t>(vars_.size())) {
            throw std::out_of_range("var id out of range");
        }
        const ModelStructure& st = s();
        const uint32_t begin = st.dependent_offsets[var_id];
        return {st.dependent_ids.data() + begin, st.dependent_offsets[var_id + 1] - begin};
    }
    [[nodiscard]] int32_t objective_id() const noexcept { return objective_id_; }
    [[nodiscard]] bool is_maximizing() const noexcept { return is_maximizing_; }
    [[nodiscard]] const std::vector<int32_t>& constraint_ids() const noexcept {
        return s().constraint_ids;
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
        ModelStructure& st = mut();
        vars_.reserve(n_vars);
        st.dependent_offsets.reserve(n_vars + 1);
        st.nodes.reserve(n_nodes);
        node_values_.reserve(n_nodes);
        st.parent_offsets.reserve(n_nodes + 1);
        st.child_refs.reserve(n_child_refs);
    }

    [[nodiscard]] const std::vector<int32_t>& topo_order() const noexcept { return s().topo_order; }
    /// Where `id` sits in `topo_order()`, so a caller holding a handful of nodes
    /// can put them in evaluation order without walking the whole order to find
    /// them. `delta_evaluate` is the caller that matters: scanning `topo_order()`
    /// and testing a flag made it O(all nodes) per call on a model whose dirty
    /// set is typically a few dozen -- ~2M nodes walked per move on the largest
    /// MIPfeas instance.
    [[nodiscard]] int32_t topo_position(int32_t id) const noexcept { return s().topo_pos[id]; }
    [[nodiscard]] const std::vector<Variable>& variables() const noexcept { return vars_; }
    [[nodiscard]] const std::vector<ExprNode>& nodes() const noexcept { return s().nodes; }
    /// Every node's current value, indexed by node id and always `num_nodes()`
    /// long. Unchecked indexing, like `nodes()`; `node_value(id)` is the checked
    /// reader. See `set_node_value_unchecked` for why the writer is unchecked.
    [[nodiscard]] const std::vector<double>& node_values() const noexcept { return node_values_; }
    [[nodiscard]] size_t num_vars() const noexcept { return vars_.size(); }
    [[nodiscard]] size_t num_nodes() const noexcept { return s().nodes.size(); }
    [[nodiscard]] bool is_closed() const noexcept { return closed_; }

    // Lambda function access
    [[nodiscard]] const std::function<double(int)>& lambda_func(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(s().lambda_funcs.size())) {
            throw std::out_of_range("lambda func index out of range");
        }
        return s().lambda_funcs[idx];
    }

    [[nodiscard]] const std::function<double(int, int)>& pair_lambda_func(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(s().pair_lambda_funcs.size())) {
            throw std::out_of_range("pair lambda func index out of range");
        }
        return s().pair_lambda_funcs[idx];
    }

    /// The closing rule and endpoint terms of the pair lambda `idx`. Same index
    /// space as `pair_lambda_func`, and the two tables are kept the same length.
    [[nodiscard]] const PairLambdaSpec& pair_lambda_spec(int32_t idx) const {
        if (idx < 0 || idx >= static_cast<int32_t>(s().pair_lambda_specs.size())) {
            throw std::out_of_range("pair lambda spec index out of range");
        }
        return s().pair_lambda_specs[idx];
    }

    // State snapshot/restore
    struct State {
        std::vector<double> values;
        std::vector<std::vector<int32_t>> elements;
    };
    [[nodiscard]] State copy_state() const;
    void restore_state(const State& state);

private:
    /// The structure, for reading -- const, which is what makes a write to the
    /// shared side a compile error (#157). Non-null on every `Model` that was not
    /// moved from; the assert is there for the one that was, on the same argument
    /// `ConstSpan::operator[]` makes -- it costs nothing under NDEBUG and turns an
    /// otherwise unreportable read into a named abort in the Debug and sanitizer
    /// builds.
    [[nodiscard]] const ModelStructure& s() const noexcept {
        assert(structure_ != nullptr);
        return *structure_;
    }
    /// The structure, for writing. Throws once `freeze()` has run: the structure
    /// is then shared with every replica, so a structural change would mutate a
    /// peer's model. Deliberately NOT copy-on-write -- a silent detach would put
    /// that worker back on its own full copy with every test still green and the
    /// memory saving gone, which is a failure that reports nothing.
    ModelStructure& mut() {
        if (open_structure_ == nullptr) {
            throw std::logic_error("model is frozen: its structure is shared and cannot change");
        }
        return *open_structure_;
    }

    std::vector<Variable> vars_;
    /// Node id -> its current value. Was `ExprNode::value`; moved out so that the
    /// node array holds nothing a search writes (#157). Kept exactly as long as
    /// `s().nodes` by `push_node`, the one place a node is made.
    std::vector<double> node_values_;
    /// The immutable side (#157). Both handles point at the same object while the
    /// model is open; `freeze()` drops `open_structure_`, after which `mut()`
    /// throws and the only handle left is a const one. Two pointers rather than
    /// one plus a flag, so that the defaulted move leaves no writable handle
    /// behind on the moved-from model.
    std::shared_ptr<const ModelStructure> structure_;
    std::shared_ptr<ModelStructure> open_structure_;
    int32_t objective_id_ = -1;
    bool is_maximizing_ = false;
    int32_t objective_bound_node_ = -1;       // Const node holding the objective RHS
    int32_t objective_constraint_node_ = -1;  // the `obj - bound` node
    int32_t objective_constraint_idx_ = -1;   // its index in constraint_ids()
    double objective_bound_ = 0.0;
    bool closed_ = false;
    // Scratch for weighted_violation_delta's pre-probe violations. A member so
    // the hot scoring path allocates only until it reaches the widest variable's
    // constraint count. Not reentrant — same single-thread-per-Model contract as
    // the probe's transient node mutation.
    std::vector<double> probe_old_violation_;

    /// Throws if the model is frozen. `mut()` is the backstop for anything that
    /// writes the structure; this is for the public mutators that would otherwise
    /// change something observable -- an objective id, say -- before reaching it,
    /// and it names the method in the message where `mut()` cannot.
    void require_open(const char* method) const;

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
