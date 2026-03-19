#pragma once

#include "model.h"
#include "rng.h"
#include <vector>
#include <string>

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
std::vector<Move> newton_tight_move(int32_t var_id, Model& model, int constraint_idx);
std::vector<Move> gradient_lift_move(int32_t var_id, Model& model, double step_size = 0.1);

// Move application
std::vector<int32_t> apply_move(Model& model, const Move& move);
SavedValues save_move_values(const Model& model, const Move& move);
void undo_move(Model& model, const Move& move, const SavedValues& saved);

class MoveProbabilities {
public:
    explicit MoveProbabilities(const std::vector<std::string>& move_types);

    std::string select(RNG& rng) const;
    void update(const std::string& move_type, bool accepted);

    const std::vector<double>& probabilities() const noexcept { return probs_; }

private:
    void rebalance();

    std::vector<std::string> move_types_;
    std::vector<int> accept_counts_;
    std::vector<int> total_counts_;
    std::vector<double> probs_;
    int update_interval_ = 1000;
    int total_updates_ = 0;
};

}  // namespace cbls
