#pragma once

#include "dag.h"
#include "element_edit.h"

#include <limits>
#include <memory>
#include <string>

namespace cbls {

class Model;

/// What the engine can say about WHERE a structured input changed, for one
/// `delta()` call (#172). See `InvariantInputs::edits`.
///
/// **"No positional information" cannot be mistaken for "no changes".** Those
/// are different answers and this type keeps them apart: `available()` is false
/// for the first, and `list()` then THROWS rather than returning an empty span
/// that would read as the second. An invariant that forgets the check fails
/// loudly on the first resynchronisation instead of returning a plausible,
/// wrong number from then on.
///
/// A VIEW into a buffer the caller of `delta_evaluate` owns (an `EditJournal`)
/// and rewrites for its next call. Valid for the duration of the `delta()` it
/// was obtained in; copy what you need, exactly as for `changed`.
class InputEdits {
public:
    /// No positional information.
    InputEdits() noexcept = default;
    /// This complete, ordered list of applied edits.
    explicit InputEdits(ConstSpan<PositionalEdit> edits) noexcept
        : edits_(edits), available_(true) {}

    /// Whether `list()` is a complete description of how this input moved since
    /// the invariant's last committed state. False means RE-READ `elements(i)`.
    [[nodiscard]] bool available() const noexcept { return available_; }

    /// The edits, in application order; each position is into the vector as it
    /// stood when that edit ran. Empty means "this input did not move".
    ///
    /// Throws `std::logic_error` when `available()` is false. Not an assert:
    /// the whole point is that the mistake is caught in a Release build, where
    /// the silent alternative is a wrong value.
    [[nodiscard]] ConstSpan<PositionalEdit> list() const;

private:
    ConstSpan<PositionalEdit> edits_;
    bool available_ = false;
};

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
    /// Outside a `delta()` -- `evaluate` and `partial` -- and for a `delta()`
    /// whose positional information is withheld (a stale probe). `edits(i)`
    /// reports none for every input.
    InvariantInputs(const Model& model, ConstSpan<ChildRef> children) noexcept
        : model_(&model), children_(children) {}

    /// Inside a `delta()`. `changed` is the same list the call is handed;
    /// `journal` is what the caller of `delta_evaluate` recorded, or null when
    /// it recorded nothing -- which is every caller but the structural batch.
    InvariantInputs(const Model& model, ConstSpan<ChildRef> children, ConstSpan<int32_t> changed,
                    const EditJournal* journal) noexcept
        : model_(&model),
          children_(children),
          changed_(changed),
          journal_(journal),
          in_delta_(true) {}

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
    /// which nothing in the engine ever writes for such a variable -- so 0.0,
    /// unless a caller put something there (`restore_state` writes `value` for
    /// every variable from a `State`, and `ModelState` is writable from Python).
    /// Read `elements(i)` instead -- `is_structured_input(i)` says which applies.
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

    /// WHERE structured input `i` changed since this invariant's last committed
    /// state: the ordered `PositionalEdit`s that took it from the elements the
    /// invariant last committed to the elements `elements(i)` returns now (#172).
    ///
    /// With them, a `delta()` over a List can be O(edits) rather than
    /// O(|elements|) -- an incremental route cost, a slack propagated from the
    /// first changed position.
    ///
    /// **Check `available()` first; when it is false, re-read `elements(i)`.**
    /// That is not a corner case. It is false:
    ///
    ///  - in `evaluate` and `partial`, which are not relative to anything;
    ///  - for an input that is not a List/Set variable;
    ///  - whenever the caller of `delta_evaluate` recorded nothing -- the
    ///    diversification kick, LNS, the inner solver, a restore, anything that
    ///    is not the structural batch, and any direct `var_mut` edit -- and in
    ///    the structural batch too unless some invariant in the model opts in
    ///    through `CustomInvariant::wants_positional_edits`;
    ///  - for a variable that saw a whole-vector `Replace` (a registered
    ///    generator's, or the inter-list tail exchange);
    ///  - after an exception left a probe open (see the class note).
    ///
    /// One case it does NOT detect: a `delta()` that threw on a `Commit` pass
    /// leaves the invariant's committed state behind the engine's, and neither
    /// `changed` nor these edits bridge the gap. That is the same obligation as
    /// for any exception out of the walk -- `full_evaluate` before the next
    /// `delta_evaluate`, which is the documented recovery.
    ///
    /// An input NOT listed in `changed` did not move, and reports an empty,
    /// available list.
    ///
    /// Accumulation: the list describes the change since the last COMMITTED
    /// state, not since the last call. A `rollback()` needs no edits -- the
    /// caller has put the assignment back -- and the next `delta()`'s list
    /// starts from the committed state again. Lifetime: as for `changed`, valid
    /// for this call only.
    ///
    /// Cost: a binary search of `changed` and a scan of the journal's records
    /// (see `EditJournal::lookup`).
    [[nodiscard]] InputEdits edits(int32_t i) const;

private:
    // A pointer rather than a reference so the class stays assignable; it is
    // never null, since the only constructor takes a reference.
    const Model* model_;
    ConstSpan<ChildRef> children_;
    ConstSpan<int32_t> changed_;
    const EditJournal* journal_ = nullptr;
    bool in_delta_ = false;
};

/// User code inside the DAG: a node whose value is whatever this object says it
/// is, with an optional incremental update path (#166).
///
/// The engine calls exactly one of `evaluate` and `delta` per pass over the
/// node, and never calls `evaluate`, `delta`, `commit`, `rollback` or `partial`
/// on one instance from two threads -- every portfolio worker holds its own
/// `clone()`. See `Model::custom`.
///
/// **`clone()` is the one exception, and it is not a small one.** `ParallelSearch`
/// replicates the master model on each WORKER's own thread (its factory is
/// `[&master]() { return master; }`, invoked from `run_worker` in `src/pool.cpp`),
/// so the master's single instance has `clone()` entered from N threads at once.
/// It must be safe to call concurrently on one object: read this object's state
/// and allocate a new one. Do not lazily fill a cache, bump a non-atomic counter
/// or draw from a shared RNG there -- which is exactly the hazard `Model::freeze`
/// documents for a stateful `lambda_sum` callable, arriving here by a different
/// route.
///
/// **The value must be a pure function of the declared inputs.** Two consequences,
/// neither obvious.
///
/// A node with NO inputs, or with only `Const` inputs, never enters a dirty cone,
/// because that cone is seeded from the changed VARIABLES' dependents -- so
/// `delta()` is never called on it and its value only ever moves at a
/// `full_evaluate`. Separately, and for its own reasons,
/// `FeasibilityJump::compute_linear_constraints` classifies such a node as a
/// constant subtree, hence affine, so a constraint containing it can be treated as
/// linear. Both readings are right for a pure value and both are silently wrong
/// for one that drifts; neither follows from the other.
///
/// And an `evaluate` that consumes a random draw, or folds a call count into its
/// result, makes `cbls::verify_model` report a spurious
/// `VerifyError::Kind::DagConsistency`, because that function re-derives every node
/// and compares against the cached value.
///
/// **No reentrancy, and it is ENFORCED.** None of these calls may invoke
/// `delta_evaluate` or `full_evaluate`, on this model or any other; doing so
/// throws `std::logic_error` from the nested call. `delta_evaluate` keeps its dirty
/// set -- and the `changed` span it hands to `delta` -- in `thread_local` buffers
/// that a nested call clears underneath the outer one, which would leak the outer
/// call's dirty flags and make those nodes stop being recomputed for the life of
/// the process. A refusal is the only failure mode that names the cause; the
/// alternative was a wrong answer somewhere else entirely, later. The same holds,
/// and is enforced the same way, for `partial` calling `compute_partial`,
/// `compute_all_partials` or `compute_partials_sparse`: the reverse sweep that
/// calls `partial` owns one `thread_local` scratch those share.
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
/// **What is NOT bracketed**, deliberately and as a recorded narrowing of #166.
/// THREE sites score a candidate by evaluating forwards and then evaluating back,
/// and every leg of all three reaches this object as an ordinary `delta()` +
/// `commit()`:
///
///  - the structural batch (`src/structural_batch.cpp`);
///  - the inner solver's `probe_candidate` and `multi_var_newton_step`
///    (`src/inner_solver.cpp`);
///  - Novelty Jump's backtracking compound-move search
///    (`FeasibilityJump::novelty_jump_search`, `src/feasibility_jump.cpp`), which
///    is on the same per-candidate path the bracketed scalar probe is.
///
/// All three are CORRECT -- each leg really is a new committed assignment, and the
/// `changed` set each passes is a complete superset of what it moved -- but they
/// cost two deltas and two commits per candidate rather than one delta and one
/// rollback, and an invariant caching a List's prefix sums updates that cache on
/// both legs. Budget a candidate on any of the three at twice a bracketed scalar
/// one. On the structural batch each leg carries positional edits for an
/// invariant that opts in (#172), so each leg is O(edits) there; on the other
/// two, and for an invariant that does not opt in, each leg is a re-read.
///
/// **Order of state.** `evaluate` is the reset point: it must return the node's
/// value for the inputs as they are, from scratch, and leave the object's
/// committed state describing exactly those inputs. The engine calls it from
/// `full_evaluate`, which is what runs after `Model::close()`, after every
/// `restore_state()` in the tree, and on every restart -- so an invariant may
/// assume that whenever the assignment moves without a `delta()` telling it so, an
/// `evaluate()` follows before the next `delta()`. (`Model::restore_state` states
/// the obligation; nothing enforces it.)
///
/// `cbls::verify_model` is the one caller that reaches `evaluate()` outside a
/// `full_evaluate` -- it re-derives every node to compare against the cached value
/// -- so it resets the committed state and pays a from-scratch pass. Harmless,
/// since it evaluates at the current assignment; worth knowing if `evaluate` is
/// expensive.
///
/// One more exit from the bracket: `full_evaluate` discards a pending probe with
/// NEITHER a `commit()` nor a `rollback()`, because `evaluate()` is about to
/// redefine the committed state anyway. Only an exception out of `delta()` can put
/// the engine in that position; the next `Commit` pass over the node does the same.
///
/// **So `evaluate()` must discard STAGED state as well as redefining committed
/// state.** It is the only call that closes the bracket without a `commit()` or a
/// `rollback()`, so an implementation that clears its staged entries only in those
/// two would carry a stale one across the reset and fold it into the next
/// `commit()`. Clear both in `evaluate()`.
class CustomInvariant {
public:
    CustomInvariant() = default;
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
    /// listed). Taking it as a hint rather than as a guarantee is what keeps
    /// building it O(arity), against the O(sum of input sizes) a value
    /// comparison would cost.
    ///
    /// **It says WHICH inputs moved; `in.edits(i)` says WHERE inside one** (#172).
    /// For a structured input, `changed` alone is "this List changed"; the
    /// positional edits, when the engine has them, are the positions. When it
    /// does not -- see `InvariantInputs::edits` for the list -- re-read
    /// `elements(i)`, which is O(n) and always correct.
    ///
    /// `changed` points into a `thread_local` buffer that the NEXT custom node
    /// in the same pass reuses. Valid for the duration of this call only; copy
    /// what you need. `in.edits(i)` has the same lifetime, for a different
    /// reason: it points into the caller's journal, which is not rewritten
    /// between custom nodes in one pass -- every node in the pass reads the same
    /// records, so no per-node buffer is needed -- but is rewritten for the
    /// caller's next `delta_evaluate`.
    ///
    /// The default re-evaluates from scratch, which is always correct and is
    /// what an invariant with no cheap incremental form should keep.
    virtual double delta(const InvariantInputs& in, ConstSpan<int32_t> changed) {
        (void)changed;
        return evaluate(in);
    }

    /// Whether this invariant reads `InvariantInputs::edits` (#172). OPT IN:
    /// the structural batch records positional edits only while some custom
    /// node in the model returns true here, so a model whose invariants all
    /// re-read records no journal. Measured idle on a 200-element List route
    /// invariant that re-reads: opting in costs ~6% per structural candidate,
    /// and a custom node that does not opt in measured +1.2% for the
    /// plumbing (not isolated: code placement alone moves these rows by more,
    /// in either direction). Allocation-free either way; docs/architecture.md has the runs and
    /// their load. The opt-in cost is why this is not simply always on.
    ///
    /// An invariant that returns false still gets a correct answer from
    /// `edits(i)` -- `available()` false whenever nothing was recorded -- so
    /// forgetting to opt in costs speed, never correctness. Queried once per
    /// structural sweep; the answer should not change over the object's life.
    [[nodiscard]] virtual bool wants_positional_edits() const { return false; }

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
    ///
    /// +/-inf reads as unknown as well. The reverse sweep accumulates
    /// `adjoint += adj * ld`, so one infinite edge would turn the next `ld == 0`
    /// into `inf * 0` -- a NaN partial for a SIBLING variable that has nothing to
    /// do with this node.
    virtual double partial(const InvariantInputs& in, int32_t i) {
        (void)in;
        (void)i;
        return std::numeric_limits<double>::quiet_NaN();
    }

    /// One instance per worker. The copy must carry this object's CURRENT state,
    /// not a freshly constructed one: a copied `Model` is already consistent with
    /// its node values and NEEDS no `full_evaluate` (see `Model::freeze`), so a
    /// caller may copy and then `delta_evaluate` straight away, and a clone that
    /// reset its cache would answer that first `delta()` against the wrong
    /// baseline. A portfolio worker happens to sweep anyway, inside
    /// `FeasibilityJump::begin`; that is a property of today's search, not of the
    /// copy, so do not lean on it.
    ///
    /// Called CONCURRENTLY on one instance -- see the note on this class.
    [[nodiscard]] virtual std::unique_ptr<CustomInvariant> clone() const = 0;

protected:
    // Copying is for a derived `clone()` -- `make_unique<Derived>(*this)` -- and
    // for nothing else. Public, they would let `CustomInvariant a = some_ref;`
    // slice an invariant down to its base, which is a silent loss of exactly the
    // state this class exists to hold.
    CustomInvariant(const CustomInvariant&) = default;
    CustomInvariant& operator=(const CustomInvariant&) = default;
    CustomInvariant(CustomInvariant&&) = default;
    CustomInvariant& operator=(CustomInvariant&&) = default;
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
    /// A `delta()` is awaiting its `commit()` or `rollback()`.
    bool probe_pending = false;
};

}  // namespace cbls
