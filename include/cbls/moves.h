#pragma once

#include "model.h"
#include "rng.h"

#include <string>
#include <vector>

namespace cbls {

class NeighbourList;

struct Move {
    struct Change {
        int32_t var_id = -1;
        double new_value = 0.0;
        std::vector<int32_t> new_elements;
    };
    std::vector<Change> changes;
    std::string move_type;
    double delta_F = 0.0;
};

// Saved state for undo
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
