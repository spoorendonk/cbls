#pragma once

#include "dag.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// The positional vocabulary of a structured change (#164), and the record of
// which of those changes actually happened (#172).
//
// Its own header rather than part of moves.h because `custom_invariant.h` needs
// it too, and moves.h includes model.h, which includes custom_invariant.h.

namespace cbls {

/// What one `ElementEdit` does to a structured variable's `elements` (#164).
///
/// POSITIONS RATHER THAN A WHOLE VECTOR. Every structured candidate used to
/// carry the complete element vector it wanted, which cost AT LEAST two heap
/// allocations (the generator's vector, and the copy of it into the candidate
/// list) and three O(n) copies (into the undo snapshot, into the variable, and
/// back out again) per candidate SCORED, on the batch's hot path, for a move
/// that touches two positions. "At least", because whether the generator's own
/// `push_back` counts depends on how it builds the vector -- so the ledger is
/// written as a floor here, in docs/architecture.md and in CLAUDE.md, and those
/// three must agree. All but the swap and the tail exchange are `std::rotate`,
/// `std::reverse` or a single insert/erase on the vector that is already there.
///
/// This is an allocation-count argument, not a micro-optimisation: it holds on
/// any machine and for any n, and it is what makes a larger candidate sample
/// affordable (see `StructuralSelection`). `Replace` keeps the old form for the
/// two things positions cannot express -- an inter-list tail exchange, and a
/// move from a generator the engine knows nothing about.
enum class EditKind : uint8_t {
    None,         ///< absent. A scalar change carries `new_value` and no edit.
    Replace,      ///< `elements = replacement`. The pre-#164 form.
    Swap,         ///< exchange positions `from` and `to`.
    Reverse,      ///< reverse the inclusive range [`from`, `to`].
    MoveSegment,  ///< erase `length` elements at `from`, reinsert at `to` of the
                  ///< SHORTENED vector. A `std::rotate`.
    Insert,       ///< insert `element` at `from`.
    Erase,        ///< erase position `from`.
    Assign,       ///< `elements[from] = element`.
};

/// One in-place rewrite of a structured variable's `elements`.
///
/// Which fields a kind reads:
///
///     Swap         from, to
///     Reverse      from, to
///     MoveSegment  from, to, length
///     Insert       from, element
///     Erase        from
///     Assign       from, element
///     Replace      the change's `replacement` vector
///
/// IT IS RELATIVE TO THE ASSIGNMENT THE MOVE WAS BUILT AGAINST. Applying an
/// edit to a different assignment is not merely a different move; the positions
/// name different elements and `Erase`/`Assign` can be out of range. The
/// structural batch is what makes that safe for the several-candidates-per-sample
/// rule -- see `MoveGenerator::generate`.
struct ElementEdit {
    EditKind kind = EditKind::None;
    int32_t from = 0;
    int32_t to = 0;
    int32_t length = 1;
    int32_t element = -1;
};

/// One edit AS IT WAS APPLIED to a structured variable's `elements` (#172) --
/// what a `CustomInvariant` is handed, through `InvariantInputs::edits`.
///
/// It differs from the `ElementEdit` a move carries in two ways, and both are
/// what make it usable by something that did not build the move:
///
///  - It records only edits that TOOK EFFECT. `apply_element_edits` ignores an
///    out-of-range position rather than indexing it (#156); such an edit is
///    absent here, so replaying this list, or inverting it, is exact.
///  - `removed` carries the element an `Erase` took out and the element an
///    `Assign` overwrote. The positions alone cannot say which one it was once
///    the vector has moved on, and an invariant pricing a removal needs it.
///
/// Fields by kind, all positions into the vector AS IT STOOD WHEN THIS EDIT RAN
/// -- i.e. after every earlier edit in the same list:
///
///     Swap         from, to                       exchanged
///     Reverse      from, to                       inclusive range reversed
///     MoveSegment  from, to, length               `length` elements at `from`
///                                                 moved to `to` of the
///                                                 shortened vector
///     Insert       from, element                  inserted at `from`
///     Erase        from, removed                  erased from `from`
///     Assign       from, element, removed         `removed` overwritten
///
/// `Replace` and `None` never appear: a whole-vector replacement has no
/// positional description, and a variable that saw one is reported as having
/// none (see `InputEdits::available`).
struct PositionalEdit {
    EditKind kind = EditKind::None;
    int32_t from = 0;
    int32_t to = 0;
    int32_t length = 1;
    int32_t element = -1;
    int32_t removed = -1;
};

/// The edit that undoes `edit` exactly, on the vector `edit` produced.
///
/// Swap and Reverse are their own inverses, a MoveSegment is moved back, an
/// Insert becomes an Erase of what it inserted and the other way round, and an
/// Assign puts `removed` back. Exact because a `PositionalEdit` only ever
/// describes an edit that took effect.
[[nodiscard]] PositionalEdit inverse_edit(const PositionalEdit& edit) noexcept;

/// Apply one recorded edit to `elements`, with the same range guards
/// `apply_element_edits` has. For an invariant that mirrors its input, and for
/// replaying an inverse.
void apply_positional_edit(const PositionalEdit& edit, std::vector<int32_t>& elements);

/// Per-variable lists of `PositionalEdit`s: what a caller of `delta_evaluate`
/// hands the engine so that a `CustomInvariant` can be told WHERE a structured
/// input changed, not only that it did (#172).
///
/// One RECORD per variable, holding the edits applied to that variable since
/// the assignment its previous evaluation was measured at, in application
/// order. A record is either KNOWN -- the edits are the complete description --
/// or UNKNOWN: something happened to the variable that has no positional
/// description (a `Replace`, or edits that could not be kept contiguous), and
/// the invariant must re-read it.
///
/// Owned by the CALLER and read by the engine for the duration of one
/// `delta_evaluate`; nothing retains a pointer into it past that call. Reusing
/// one journal across calls (`clear()` then record again) is what keeps the
/// steady state allocation-free: `clear()` keeps the capacity.
class EditJournal {
public:
    enum class Status : uint8_t {
        Absent,   ///< no record for this variable
        Unknown,  ///< a record, but no positional description of it
        Known,    ///< a record, and `edits` is the complete description
    };

    /// Forget every record. Keeps the capacity.
    void clear() noexcept {
        records_.clear();
        edits_.clear();
        open_ = kNone;
    }

    /// Open `var_id`'s record; `push` and `mark_unknown` apply to it until the
    /// next `begin`. Re-opening the most recent record continues it. Re-opening
    /// an EARLIER one cannot keep its edits contiguous, so that record becomes
    /// Unknown -- conservative, never wrong.
    void begin(int32_t var_id);

    /// Append one applied edit to the open record. Asserts a record is open.
    void push(const PositionalEdit& edit);

    /// The open record has no positional description.
    void mark_unknown() noexcept;

    /// Append to the open record the exact inverse of `var_id`'s record in
    /// `applied`: its edits inverted, in reverse order. An Unknown record makes
    /// the open one Unknown; an absent one appends nothing.
    ///
    /// `begin(v)`, `append_inverse(previous, v)`, `append_forward(current, v)`
    /// is how the structural batch describes "put the previous candidate back,
    /// then apply this one" without materialising either.
    void append_inverse(const EditJournal& applied, int32_t var_id);

    /// Append `var_id`'s record in `applied` to the open record, as it stands.
    /// An Unknown record makes the open one Unknown; an absent one appends
    /// nothing.
    void append_forward(const EditJournal& applied, int32_t var_id);

    /// Look up `var_id`. Fills `edits` only for `Status::Known`. Linear in the
    /// number of RECORDS: the variables one candidate touched (one or two for
    /// every built-in), or every variable a sample named on the structural
    /// batch's final restore.
    [[nodiscard]] Status lookup(int32_t var_id, ConstSpan<PositionalEdit>& edits) const noexcept;

    [[nodiscard]] bool empty() const noexcept { return records_.empty(); }

private:
    static constexpr size_t kNone = static_cast<size_t>(-1);

    struct Record {
        int32_t var_id = -1;
        uint32_t begin = 0;
        uint32_t end = 0;
        bool known = true;
    };
    std::vector<Record> records_;
    std::vector<PositionalEdit> edits_;
    size_t open_ = kNone;
};

}  // namespace cbls
