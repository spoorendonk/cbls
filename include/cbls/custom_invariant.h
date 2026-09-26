#pragma once

#include "dag.h"

#include <limits>
#include <memory>
#include <string>

namespace cbls {

class Model;

/// The inputs of one `NodeOp::Custom` node: its children, in the order they
/// were given to `Model::custom` (#166).
///
/// A VIEW, not a container. `children()` is a slice of the model's flat
/// `child_refs` array, so it carries the same invalidation rule every
/// `ConstSpan` in this codebase does: appending a node to the model may
/// reallocate that array. The engine builds one of these on the stack per call
/// and the invariant is handed it by const reference, so the only way to get
/// this wrong is to STORE it. Don't -- store the child ids, or nothing.
///
/// Reading through it is unchecked in the same places the DAG's own evaluation
/// is: an out-of-range child index asserts in a Debug build and is undefined
/// under NDEBUG, exactly as `ConstSpan::operator[]` is.
class InvariantInputs {
public:
    InvariantInputs(const Model& model, ConstSpan<ChildRef> children) noexcept
        : model_(&model), children_(children) {}

    /// The model being evaluated. Read-only: an invariant that wrote to it
    /// would be writing another worker's search state on the shared-structure
    /// path (#157), and would desynchronise the node-value cache besides.
    [[nodiscard]] const Model& model() const noexcept { return *model_; }
    [[nodiscard]] ConstSpan<ChildRef> children() const noexcept { return children_; }
    /// How many inputs the node has. `int32_t` rather than `size_t` because
    /// that is the type of a child index throughout this interface, `delta`'s
    /// `changed` list included.
    [[nodiscard]] int32_t size() const noexcept { return static_cast<int32_t>(children_.size()); }

    /// Input `i`'s current scalar value: the variable's `value` for a
    /// Bool/Int/Float child, the cached node value for a node child.
    ///
    /// A List/Set child has no scalar value; this returns its `value` field,
    /// which such a variable never sets and which is therefore 0.0. Read
    /// `elements(i)` instead -- `is_structured_input(i)` says which applies.
    [[nodiscard]] double value(int32_t i) const;

    /// Input `i`'s elements, for a List or Set variable child. Empty for every
    /// other input, including a node child -- a node has no elements.
    ///
    /// A List's order is the modelled one; a Set's is whatever order the moves
    /// happen to have left it in and carries no meaning (see `Variable`).
    [[nodiscard]] ConstSpan<int32_t> elements(int32_t i) const;

    /// Whether input `i` is a List or Set VARIABLE, i.e. whether `elements(i)`
    /// rather than `value(i)` is the input's content.
    [[nodiscard]] bool is_structured_input(int32_t i) const;

private:
    // A pointer rather than a reference so the class stays assignable; it is
    // never null, since the only constructor takes a reference.
    const Model* model_;
    ConstSpan<ChildRef> children_;
};

/// User code inside the DAG: a node whose value is whatever this object says it
/// is, with an optional incremental update path (#166).
///
/// The engine calls exactly one of `evaluate` and `delta` per pass over the
/// node, and never calls one instance from two threads -- every portfolio
/// worker holds its own `clone()`. See `Model::custom`.
///
/// **The probe protocol.** The search scores a candidate move by applying it,
/// evaluating, and then putting the old assignment back. A stateful invariant
/// would see that as two `delta` calls per candidate and would have to keep a
/// cache that thrashes between the two assignments. So for the scalar probe
/// that Feasibility Jump runs per candidate jump value
/// (`Model::weighted_violation_delta`), the engine instead brackets the pair:
///
///     delta(new inputs, changed)   // the candidate
///     ... the engine reads the node's value ...
///     rollback()                   // the caller restored the old assignment
///
/// and on the path that keeps a move,
///
///     delta(new inputs, changed)
///     commit()
///
/// So exactly one `commit()` or `rollback()` follows every `delta()`, and the
/// invariant's committed state only ever moves forward. The node's own cached
/// VALUE is restored by the engine on a rollback; the invariant is responsible
/// only for its own state.
///
/// **What is NOT bracketed**, deliberately and as a recorded narrowing of #166:
/// the structural batch (`src/structural_batch.cpp`) and the inner solver
/// (`src/inner_solver.cpp`) score their candidates by evaluating forwards and
/// then evaluating back, and both legs reach this object as an ordinary
/// `delta()` + `commit()`. They are CORRECT -- each leg really is a new
/// committed assignment, and `changed` is measured against the one before it --
/// but they cost two deltas and two commits per candidate rather than one delta
/// and one rollback, and an invariant caching a List's prefix sums will rebuild
/// that cache on both legs. Budget a structural candidate at twice a scalar
/// one until those two sites are bracketed too.
///
/// **Order of state.** `evaluate` is the reset point: it must return the node's
/// value for the inputs as they are, from scratch, and leave the object's
/// committed state describing exactly those inputs. The engine calls it from
/// `full_evaluate`, which is what runs after `Model::close()`, after
/// `restore_state()`, and on every restart -- so an invariant may assume that
/// whenever the assignment moves without a `delta()` telling it so, an
/// `evaluate()` follows before the next `delta()`.
class CustomInvariant {
public:
    CustomInvariant() = default;
    CustomInvariant(const CustomInvariant&) = default;
    CustomInvariant& operator=(const CustomInvariant&) = default;
    CustomInvariant(CustomInvariant&&) = default;
    CustomInvariant& operator=(CustomInvariant&&) = default;
    virtual ~CustomInvariant() = default;

    /// From scratch, over the inputs as they are. See "Order of state" above.
    virtual double evaluate(const InvariantInputs& in) = 0;

    /// Incremental. `changed` lists the indices of the inputs the engine
    /// recomputed since this object's last committed state, ascending.
    ///
    /// It is a SUPERSET of the inputs whose value actually differs: the engine
    /// derives it from the dirty cone it was about to recompute anyway, which
    /// is exact for variable inputs and conservative for node inputs (a node
    /// whose recomputation happened to land on the same value is still
    /// listed). Taking it as a hint rather than as a guarantee costs nothing
    /// and is what keeps it O(1) to build.
    ///
    /// The default re-evaluates from scratch, which is always correct and is
    /// what an invariant with no cheap incremental form should keep.
    virtual double delta(const InvariantInputs& in, ConstSpan<int32_t> changed) {
        (void)changed;
        return evaluate(in);
    }

    /// The assignment the last `delta()` was measured at is the new committed
    /// one.
    virtual void commit() {}

    /// The caller has put back the assignment the last `delta()` was measured
    /// FROM; discard whatever that call staged. The committed state is
    /// unchanged.
    virtual void rollback() {}

    /// d(node value)/d(input i), for the continuous jump candidates and the
    /// inner solver's Newton steps. NaN means "unknown", which the engine reads
    /// as 0.0 -- the same answer it gives for every structural op today, and
    /// one that leaves a Float input with Feasibility Jump's non-gradient
    /// candidates rather than no candidates.
    virtual double partial(const InvariantInputs& in, int32_t i) {
        (void)in;
        (void)i;
        return std::numeric_limits<double>::quiet_NaN();
    }

    /// One instance per worker. The copy must carry this object's CURRENT
    /// state, not a freshly constructed one: a portfolio replica starts from
    /// the master's node values verbatim and does not `full_evaluate` (see
    /// `Model::freeze`), so a clone that reset its cache would answer the first
    /// `delta()` against the wrong baseline.
    [[nodiscard]] virtual std::unique_ptr<CustomInvariant> clone() const = 0;
};

/// One custom node's PER-MODEL state (#166), held by `Model` and never by the
/// shared `ModelStructure`: the invariant is mutable state a search writes, so
/// it lives on the side of the #157 split that a portfolio replica copies.
///
/// `probe_*` is the bracket described on `CustomInvariant`. Only
/// `delta_evaluate` and `full_evaluate` touch them.
struct CustomInvariantSlot {
    std::unique_ptr<CustomInvariant> invariant;
    /// What `Model::custom` was told to call this node. Used by the `.cbls`
    /// writer's refusal message and nothing else, so it is duplicated per
    /// replica rather than given a table in the shared structure -- one short
    /// string per custom node, against a whole DAG the replicas already share.
    std::string name;
    /// The node's cached value as it stood before the pending probe's
    /// `delta()`, for `DeltaMode::Rollback` to put back.
    double probe_saved_value = 0.0;
    /// A `delta()` is awaiting its `commit()` or `rollback()`.
    bool probe_pending = false;
};

}  // namespace cbls
