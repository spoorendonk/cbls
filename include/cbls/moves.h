#pragma once

#include "model.h"
#include "rng.h"

#include <string>
#include <vector>

namespace cbls {

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
std::vector<Move> generate_block_moves(int32_t var_id, const Model& model, RNG& rng);

// Move application
std::vector<int32_t> apply_move(Model& model, const Move& move);
SavedValues save_move_values(const Model& model, const Move& move);
void undo_move(Model& model, const Move& move, const SavedValues& saved);

}  // namespace cbls
