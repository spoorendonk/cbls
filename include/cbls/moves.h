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

// Move application
std::vector<int32_t> apply_move(Model& model, const Move& move);
SavedValues save_move_values(const Model& model, const Move& move);
void undo_move(Model& model, const Move& move, const SavedValues& saved);

}  // namespace cbls
