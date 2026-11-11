#pragma once

#include "inline_vector.h"
#include "model.h"
#include "rng.h"

#include <array>
#include <cstdint>
#include <string_view>
#include <vector>

namespace cbls {

class NeighbourList;

/// What one `ElementEdit` does to a structured variable's `elements` (#164).
///
/// POSITIONS RATHER THAN A WHOLE VECTOR: a whole-vector candidate costs AT
/// LEAST two heap allocations and three O(n) copies per candidate SCORED for a
/// move touching two positions (the floor docs/architecture.md and
/// docs/benchmarks/README.md also state; the three must agree). Every kind but
/// `Replace` edits the existing vector in place. An allocation-count
/// argument, true on any machine and n, that makes a larger candidate sample
/// affordable (`StructuralSelection`). `Replace` covers what positions cannot
/// express: an inter-list tail exchange and an unknown generator's move.
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
/// RELATIVE TO THE ASSIGNMENT THE MOVE WAS BUILT AGAINST: on another, the
/// positions name other elements and `Erase`/`Assign` can be out of range. The
/// structural batch's baseline keeps several candidates per sample safe (see
/// `MoveGenerator::generate`).
struct ElementEdit {
    EditKind kind = EditKind::None;
    int32_t from = 0;
    int32_t to = 0;
    int32_t length = 1;
    int32_t element = -1;
};

struct Move {
    /// One variable's part of a move: a scalar value, or up to two positional
    /// edits applied in ORDER (`set_swap` erases the dropped element, then
    /// appends the new one). A trailing `EditKind::None` means one edit.
    struct Change {
        int32_t var_id = -1;
        double new_value = 0.0;
        std::array<ElementEdit, 2> edits{};
        std::vector<int32_t> replacement;  // EditKind::Replace only
    };
    /// Two inline: every built-in move has one or two changes, so no malloc
    /// per candidate (#274).
    InlineVector<Change, 2> changes;
    /// The kind's name: a STRING LITERAL or other static-storage string only;
    /// a temporary std::string dangles. A view, so no per-candidate allocation
    /// for names past the small-string buffer (#274).
    std::string_view move_type;
    double delta_F = 0.0;
};

/// A scalar variable's change.
Move::Change scalar_change(int32_t var_id, double new_value);
/// A structured variable's change, as one or two positional edits.
Move::Change edit_change(int32_t var_id, const ElementEdit& first,
                         const ElementEdit& second = ElementEdit{});
/// A structured variable's change, as a whole replacement vector. The general
/// form, for what positions cannot express.
Move::Change replace_change(int32_t var_id, std::vector<int32_t> replacement);

inline ElementEdit swap_edit(int32_t a, int32_t b) {
    return {EditKind::Swap, a, b, 1, -1};
}
inline ElementEdit reverse_edit(int32_t lo, int32_t hi) {
    return {EditKind::Reverse, lo, hi, 1, -1};
}
inline ElementEdit segment_edit(int32_t from, int32_t length, int32_t to) {
    return {EditKind::MoveSegment, from, to, length, -1};
}
inline ElementEdit insert_edit(int32_t pos, int32_t element) {
    return {EditKind::Insert, pos, 0, 1, element};
}
inline ElementEdit erase_edit(int32_t pos) {
    return {EditKind::Erase, pos, 0, 1, -1};
}
inline ElementEdit assign_edit(int32_t pos, int32_t element) {
    return {EditKind::Assign, pos, 0, 1, element};
}

/// Apply `change`'s edits to `elements` in place. Out-of-range positions are
/// ignored, not indexed: `Variable.elements` is writable from Python, so an
/// edit can meet a shorter vector than it was built for (#156).
void apply_element_edits(const Move::Change& change, std::vector<int32_t>& elements);

/// `elements` with `change` applied, without touching the model (tests,
/// Python). The engine edits in place instead.
std::vector<int32_t> elements_after(const Move::Change& change,
                                    const std::vector<int32_t>& elements);

/// Does `change` leave `elements` exactly as it is? Decided per kind, without
/// materialising the result.
///
/// EXACT FOR A SINGLE EDIT OVER DISTINCT ELEMENTS (every change the engine
/// builds): `change_is_noop(c, e) == (elements_after(c, e) == e)`, pinned in
/// `tests/test_moves.cpp`. Distinct elements make any non-empty reversal or
/// displaced segment a reorder; an out-of-range edit reads as inert because
/// applying it is inert (#156).
///
/// OTHERWISE CONSERVATIVE: it may say "changed" for an unchanged result, never
/// the reverse, so the cost is a needless scored candidate. Reachable only via
/// Python-written `Variable::elements`: (1) edits are judged INDEPENDENTLY, so
/// `erase(1)` then `insert(1, 11)` on `{10, 11, 12}` reads as changed; (2) on
/// repeated elements `Reverse`/`MoveSegment` decide by position alone.
[[nodiscard]] bool change_is_noop(const Move::Change& change, const std::vector<int32_t>& elements);

/// The whole pre-move state of the variables a `Move` touches: absolute, not
/// an inverse edit list, since `undo_move` is off the hot path (the structural
/// batch restores from its sample baseline). The public and Python way to try
/// a move and take it back.
struct SavedValues {
    std::vector<double> values;
    std::vector<std::vector<int32_t>> elements;
};

// Move generators
std::vector<Move> generate_standard_moves(const Variable& var, RNG& rng);

/// Appending form of the above, called by the structural batch's built-in
/// generators (#165): same moves and RNG draws in the same order;
/// the overload above is this with `out` empty and `neighbours` null.
///
/// Non-null `neighbours` makes the TARGET granular: a List move pairs position
/// `i` with one of `elements[i]`'s nearest neighbours; a Set add/swap brings in
/// a selected element's nearest neighbour not already in. Both fall back to the
/// uniform draw when the list offers nothing, so a partial list limits where
/// the search looks, never what it reaches. Null (the default) keeps an
/// unconfigured run on its trajectory.
///
/// Cost: a guided List move adds an O(|elements|) element -> position index
/// per call, a constant factor on construction that is already O(|elements|);
/// it buys an asymptotic SEARCH win (k candidates from a k-nearest list, not
/// the universe), and is paid only when a list was supplied.
void generate_standard_moves(const Variable& var, RNG& rng, std::vector<Move>& out,
                             const NeighbourList* neighbours);

/// Append AT MOST ONE candidate inter-list move over `partition` (#164):
/// relocate, swap or 2-opt* between two of its lists, and -- under
/// `Cover::AtMostOnce` -- an insert from the unassigned elements or a removal.
/// The kind is drawn uniformly from those the partition admits; a kind whose
/// guards reject the draw appends nothing.
///
/// AT MOST ONE CANDIDATE PER CALL, deliberately. The batch's per-sample
/// baseline makes several safe (argument in `move_generator.h`), but one is
/// kept because a cover maintained by construction is costly to get wrong, and
/// one is safe under any batch that applies moves, not just this one's
/// discipline. Sampling policies call `generate` repeatedly for full samples.
///
/// `anchor`: a member list id every candidate must change, or -1 to draw both
/// lists; the diversification kick names one because it judges its effect on
/// that variable alone. `neighbours` makes the TARGET granular as for
/// intra-list moves (nearest neighbours in the other list); null by default.
void generate_partition_moves(const Model& model, int partition, int32_t anchor, RNG& rng,
                              std::vector<Move>& out, const NeighbourList* neighbours);

/// The GRANULAR COMPLETE neighbourhood of one element of `partition`, for
/// `StructuralSelection::GranularBest` (#203; Toth & Vigo 2003, INFORMS J.
/// Computing 15(4)). Draws one element `e` of the universe uniformly, then
/// appends every move pairing `e` with each of its `neighbours`:
///
///   - relocate `e` just after and just before `f` (within one list a segment
///     move, across two an erase and an insert);
///   - swap `e` and `f`;
///   - the 2-opt that makes `e` and `f` adjacent: within one list the two
///     reversals that do, across two lists the 2-opt* tail exchange in both
///     orientations.
///
/// Under `Cover::AtMostOnce` an assigned `e` is also offered for removal, and
/// an unassigned one inserted just after and before each assigned `f`. `e` is
/// also moved into every EMPTY member list, so an emptied list can reopen.
/// Length bounds hold and no-ops are dropped. At most
/// `5 * |neighbours.of(e)| + 1 + |empty lists|` candidates, so the batch's
/// bounded-yield contract holds for any fixed k and fleet.
///
/// Deviations from Toth & Vigo, as #203 specified: the sparse graph is each
/// element's k NEAREST neighbours, not edges below nu = beta * z / (n + K) plus
/// depot edges; and one call covers ONE drawn element, not the whole
/// neighbourhood.
///
/// SEVERAL CANDIDATES PER CALL, for a policy committing at most one
/// (take-best); all are built against the current assignment, which the
/// batch's per-sample baseline restores before each. `slot_scratch`: the
/// caller's reusable element -> (list, position) index, rebuilt every call.
void generate_granular_partition_moves(const Model& model, int partition, RNG& rng,
                                       const NeighbourList& neighbours, std::vector<Move>& out,
                                       std::vector<int64_t>& slot_scratch);

// Move application
std::vector<int32_t> apply_move(Model& model, const Move& move);
SavedValues save_move_values(const Model& model, const Move& move);
void undo_move(Model& model, const Move& move, const SavedValues& saved);

}  // namespace cbls
