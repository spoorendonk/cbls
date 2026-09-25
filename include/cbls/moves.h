#pragma once

#include "model.h"
#include "rng.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace cbls {

class NeighbourList;

/// What one `ElementEdit` does to a structured variable's `elements` (#164).
///
/// POSITIONS RATHER THAN A WHOLE VECTOR. Every structured candidate used to
/// carry the complete element vector it wanted, which cost one heap allocation
/// and one O(n) copy to build, a second pair to snapshot for the undo, and two
/// more copies to apply and roll back -- four copies and two allocations per
/// candidate SCORED, on the batch's hot path, for a move that touches two
/// positions. All but the swap and the tail exchange are `std::rotate`,
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

struct Move {
    /// One variable's part of a move: a scalar value, or up to two positional
    /// edits to its `elements`.
    ///
    /// Two rather than one because `set_swap` is genuinely two edits on one
    /// variable -- erase the dropped element where it sits, append the new one
    /// -- and that ORDER is what produces the element vector it has always
    /// produced. A trailing `EditKind::None` means "only one edit".
    struct Change {
        int32_t var_id = -1;
        double new_value = 0.0;
        std::array<ElementEdit, 2> edits{};
        std::vector<int32_t> replacement;  // EditKind::Replace only
    };
    std::vector<Change> changes;
    std::string move_type;
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
/// ignored rather than indexed: `Variable.elements` is writable from Python, so
/// an edit built against a longer vector can be replayed against a shorter one
/// without any check in the way (#156).
void apply_element_edits(const Move::Change& change, std::vector<int32_t>& elements);

/// `elements` with `change` applied -- the absolute vector the change used to
/// carry. For tests, for Python, and for anything that wants the result without
/// touching the model. The engine edits in place instead.
std::vector<int32_t> elements_after(const Move::Change& change,
                                    const std::vector<int32_t>& elements);

/// Does `change` leave `elements` exactly as it is? Decided per kind rather
/// than by materialising the result, which is the whole point of the
/// representation. Exact: a List's elements are distinct, so a reversal of a
/// non-empty range or a segment moved to a different position always changes
/// the order.
[[nodiscard]] bool change_is_noop(const Move::Change& change, const std::vector<int32_t>& elements);

/// The whole pre-move state of the variables a `Move` touches.
///
/// Deliberately absolute rather than an inverse edit list: `undo_move` is no
/// longer on the hot path -- the structural batch rolls a candidate back by
/// replaying it from the sample baseline instead (`StructuralBatch`) -- so the
/// simplest thing that cannot be subtly wrong is the right one here. It remains
/// the public and Python-facing way to try a move and take it back.
struct SavedValues {
    std::vector<double> values;
    std::vector<std::vector<int32_t>> elements;
};

// Move generators
std::vector<Move> generate_standard_moves(const Variable& var, RNG& rng);

/// Appending form of the above, and the one the structural batch's built-in
/// generators call (#165). Same moves, same RNG draws, in the same order;
/// `generate_standard_moves` is this with `out` empty and `neighbours` null.
///
/// `neighbours`, when non-null, makes the move's TARGET granular rather than
/// uniform: a List move relocates/swaps position `i` against the position of one
/// of `elements[i]`'s nearest neighbours, and a Set add/swap brings in the
/// nearest neighbour of a currently-selected element that is not already in.
/// Both fall back to the uniform draw when the list has nothing usable to offer,
/// so a partial list is a restriction on where the search looks and never a
/// restriction on what it can reach. Null is the default everywhere and is what
/// keeps an unconfigured run on the trajectory it had before #165.
///
/// A neighbour-guided List move builds an element -> position index, so it is
/// O(|elements|) per call on top of the move construction that is already
/// O(|elements|) per candidate. That is a constant-factor cost on a path that
/// only runs when a caller supplied a list; the win it pays for is asymptotic on
/// the SEARCH rather than on the call -- k candidates drawn from a k-nearest
/// list instead of from the whole universe.
void generate_standard_moves(const Variable& var, RNG& rng, std::vector<Move>& out,
                             const NeighbourList* neighbours);

/// Append AT MOST ONE candidate inter-list move over `partition` (#164):
/// relocate, swap or 2-opt* between two of its lists, and -- under
/// `Cover::AtMostOnce` -- an insert from the unassigned elements or a removal.
/// The kind is drawn uniformly from the ones the partition admits, so this list
/// is also the mix; a kind whose guards reject the draw appends nothing.
///
/// EXACTLY ONE CANDIDATE PER CALL IS A CORRECTNESS REQUIREMENT, not a budget.
/// Every candidate from one `MoveGenerator::generate` carries an ABSOLUTE
/// element vector built against the same pre-commit assignment, and
/// `StructuralSelection::FirstImprovingSample` may commit several of them in
/// turn -- so a second candidate built before the first was committed would
/// reinstate the list the first one moved an element out of, leaving that
/// element in two lists with nothing to notice. `move_generator.h` names the two
/// ways out; this is the first of them (emit one candidate per call), chosen
/// because it holds under EVERY selection policy rather than only under the two
/// that commit at most one. The sampling policies call `generate` repeatedly, so
/// they still get a sample of the size they asked for.
///
/// `anchor` is a member list id that every candidate must change, or -1 to draw
/// both lists. The diversification kick names one, because it asks "move THIS
/// variable" and decides whether the kick did anything by looking at that
/// variable alone.
///
/// `neighbours` makes the TARGET granular rather than uniform, exactly as it
/// does for the intra-list moves: an element is relocated next to, or swapped
/// against, one of its nearest neighbours in the other list. Null is the default
/// everywhere.
void generate_partition_moves(const Model& model, int partition, int32_t anchor, RNG& rng,
                              std::vector<Move>& out, const NeighbourList* neighbours);

// Move application
std::vector<int32_t> apply_move(Model& model, const Move& move);
SavedValues save_move_values(const Model& model, const Move& move);
void undo_move(Model& model, const Move& move, const SavedValues& saved);

}  // namespace cbls
