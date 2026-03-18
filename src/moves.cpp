#include "cbls/moves.h"
#include "cbls/dag_ops.h"
#include <algorithm>
#include <cmath>

namespace cbls {

static std::vector<Move> bool_moves(const Variable& var) {
    Move m;
    m.move_type = "flip";
    m.changes.push_back({var.id, 1.0 - var.value, {}});
    return {m};
}

static std::vector<Move> int_moves(const Variable& var, RNG& rng) {
    std::vector<Move> moves;
    if (var.value > var.lb) {
        Move m;
        m.move_type = "int_dec";
        m.changes.push_back({var.id, var.value - 1.0, {}});
        moves.push_back(m);
    }
    if (var.value < var.ub) {
        Move m;
        m.move_type = "int_inc";
        m.changes.push_back({var.id, var.value + 1.0, {}});
        moves.push_back(m);
    }
    {
        Move m;
        m.move_type = "int_rand";
        double new_val = static_cast<double>(rng.integers(
            static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
        m.changes.push_back({var.id, new_val, {}});
        moves.push_back(m);
    }
    return moves;
}

static std::vector<Move> float_moves(const Variable& var, RNG& rng, double sigma_frac = 0.1) {
    double sigma = (var.ub - var.lb) * sigma_frac;
    double new_val = var.value + rng.normal(0, sigma);
    new_val = std::clamp(new_val, var.lb, var.ub);
    Move m;
    m.move_type = "float_perturb";
    m.changes.push_back({var.id, new_val, {}});
    return {m};
}

static std::vector<Move> list_moves(const Variable& var, RNG& rng) {
    std::vector<Move> moves;
    int n = static_cast<int>(var.elements.size());
    if (n < 2) return moves;

    int i = static_cast<int>(rng.integers(0, n));
    int j = static_cast<int>(rng.integers(0, n - 1));
    if (j >= i) j++;  // ensure i != j

    // Swap
    {
        Move m;
        m.move_type = "list_swap";
        auto new_elems = var.elements;
        std::swap(new_elems[i], new_elems[j]);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }

    // 2-opt reverse
    {
        Move m;
        m.move_type = "list_2opt";
        int lo = std::min(i, j);
        int hi = std::max(i, j);
        auto new_elems = var.elements;
        std::reverse(new_elems.begin() + lo, new_elems.begin() + hi + 1);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }

    // Relocate: remove element at position i, insert at position j
    {
        Move m;
        m.move_type = "list_relocate";
        auto new_elems = var.elements;
        int32_t elem = new_elems[i];
        new_elems.erase(new_elems.begin() + i);
        int insert_pos = (j > i) ? j - 1 : j;
        new_elems.insert(new_elems.begin() + insert_pos, elem);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }

    // Or-opt(2): relocate a consecutive pair
    if (n >= 3 && i < n - 1) {
        Move m;
        m.move_type = "list_or_opt_2";
        auto new_elems = var.elements;
        int32_t e0 = new_elems[i];
        int32_t e1 = new_elems[i + 1];
        new_elems.erase(new_elems.begin() + i, new_elems.begin() + i + 2);
        int insert_pos = j;
        if (j > i) insert_pos = std::max(0, j - 2);
        insert_pos = std::min(insert_pos, static_cast<int>(new_elems.size()));
        new_elems.insert(new_elems.begin() + insert_pos, e1);
        new_elems.insert(new_elems.begin() + insert_pos, e0);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }

    // Or-opt(3): relocate a consecutive triple
    if (n >= 4 && i < n - 2) {
        Move m;
        m.move_type = "list_or_opt_3";
        auto new_elems = var.elements;
        int32_t e0 = new_elems[i];
        int32_t e1 = new_elems[i + 1];
        int32_t e2 = new_elems[i + 2];
        new_elems.erase(new_elems.begin() + i, new_elems.begin() + i + 3);
        int insert_pos = j;
        if (j > i) insert_pos = std::max(0, j - 3);
        insert_pos = std::min(insert_pos, static_cast<int>(new_elems.size()));
        new_elems.insert(new_elems.begin() + insert_pos, e2);
        new_elems.insert(new_elems.begin() + insert_pos, e1);
        new_elems.insert(new_elems.begin() + insert_pos, e0);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }

    return moves;
}

static std::vector<Move> set_moves(const Variable& var, RNG& rng) {
    std::vector<Move> moves;

    // Build not_in and in_set lists
    std::vector<int32_t> in_set(var.elements.begin(), var.elements.end());
    std::vector<int32_t> not_in;
    // For set vars, elements stores the current set. Universe is {0..universe_size-1}
    std::vector<bool> in_flag(var.universe_size, false);
    for (int32_t e : var.elements) {
        if (e >= 0 && e < var.universe_size) in_flag[e] = true;
    }
    for (int32_t i = 0; i < var.universe_size; ++i) {
        if (!in_flag[i]) not_in.push_back(i);
    }

    int cur_size = static_cast<int>(var.elements.size());

    // Add
    if (!not_in.empty() && cur_size < var.max_size) {
        Move m;
        m.move_type = "set_add";
        int32_t add_elem = not_in[static_cast<size_t>(rng.integers(0, not_in.size()))];
        auto new_elems = var.elements;
        new_elems.push_back(add_elem);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }

    // Remove
    if (!in_set.empty() && cur_size > var.min_size) {
        Move m;
        m.move_type = "set_remove";
        int32_t rem_elem = in_set[static_cast<size_t>(rng.integers(0, in_set.size()))];
        auto new_elems = var.elements;
        auto it = std::find(new_elems.begin(), new_elems.end(), rem_elem);
        if (it != new_elems.end()) {
            new_elems.erase(it);
            m.changes.push_back({var.id, 0.0, new_elems});
            moves.push_back(m);
        }
    }

    // Swap
    if (!in_set.empty() && !not_in.empty()) {
        Move m;
        m.move_type = "set_swap";
        int32_t add_elem = not_in[static_cast<size_t>(rng.integers(0, not_in.size()))];
        int32_t rem_elem = in_set[static_cast<size_t>(rng.integers(0, in_set.size()))];
        auto new_elems = var.elements;
        auto it = std::find(new_elems.begin(), new_elems.end(), rem_elem);
        if (it != new_elems.end()) {
            new_elems.erase(it);
            new_elems.push_back(add_elem);
            m.changes.push_back({var.id, 0.0, new_elems});
            moves.push_back(m);
        }
    }

    return moves;
}

std::vector<Move> generate_standard_moves(const Variable& var, RNG& rng) {
    switch (var.type) {
    case VarType::Bool:  return bool_moves(var);
    case VarType::Int:   return int_moves(var, rng);
    case VarType::Float: return float_moves(var, rng);
    case VarType::List:  return list_moves(var, rng);
    case VarType::Set:   return set_moves(var, rng);
    }
    return {};
}

std::vector<Move> newton_tight_move(int32_t var_id, Model& model, int constraint_idx) {
    const auto& var = model.var(var_id);
    if (var.type != VarType::Float) return {};

    int32_t cid = model.constraint_ids()[constraint_idx];
    double g_x = model.node(cid).value;  // > 0 means violated
    double dg_dxj = compute_partial(model, cid, var_id);

    if (std::abs(dg_dxj) < 1e-12) return {};

    double delta = -g_x / dg_dxj;
    double new_val = std::clamp(var.value + delta, var.lb, var.ub);
    if (std::abs(new_val - var.value) < 1e-15) return {};

    Move m;
    m.move_type = "newton_tight";
    m.changes.push_back({var_id, new_val, {}});
    return {m};
}

std::vector<Move> gradient_lift_move(int32_t var_id, Model& model, double step_size) {
    const auto& var = model.var(var_id);
    if (var.type != VarType::Float) return {};
    if (model.objective_id() < 0) return {};

    double df_dxj = compute_partial(model, model.objective_id(), var_id);
    if (std::abs(df_dxj) < 1e-12) return {};

    double delta = -step_size * df_dxj;
    double new_val = std::clamp(var.value + delta, var.lb, var.ub);
    if (std::abs(new_val - var.value) < 1e-15) return {};

    Move m;
    m.move_type = "gradient_lift";
    m.changes.push_back({var_id, new_val, {}});
    return {m};
}

std::vector<int32_t> apply_move(Model& model, const Move& move) {
    std::vector<int32_t> changed;
    changed.reserve(move.changes.size());
    for (const auto& change : move.changes) {
        auto& var = model.var_mut(change.var_id);
        if (var.type == VarType::List || var.type == VarType::Set) {
            var.elements = change.new_elements;
        } else {
            var.value = change.new_value;
        }
        changed.push_back(change.var_id);
    }
    return changed;
}

SavedValues save_move_values(const Model& model, const Move& move) {
    SavedValues saved;
    for (const auto& change : move.changes) {
        const auto& var = model.var(change.var_id);
        saved.values.push_back(var.value);
        saved.elements.push_back(var.elements);
    }
    return saved;
}

void undo_move(Model& model, const Move& move, const SavedValues& saved) {
    for (size_t i = 0; i < move.changes.size(); ++i) {
        auto& var = model.var_mut(move.changes[i].var_id);
        var.value = saved.values[i];
        var.elements = saved.elements[i];
    }
}

// MoveProbabilities
MoveProbabilities::MoveProbabilities(const std::vector<std::string>& move_types)
    : move_types_(move_types) {
    size_t n = move_types.size();
    accept_counts_.resize(n, 0);
    total_counts_.resize(n, 0);
    probs_.resize(n, 1.0 / n);
}

std::string MoveProbabilities::select(RNG& rng) const {
    double r = rng.random();
    double cumulative = 0.0;
    for (size_t i = 0; i < move_types_.size(); ++i) {
        cumulative += probs_[i];
        if (r < cumulative) return move_types_[i];
    }
    return move_types_.back();
}

void MoveProbabilities::update(const std::string& move_type, bool accepted) {
    for (size_t i = 0; i < move_types_.size(); ++i) {
        if (move_types_[i] == move_type) {
            total_counts_[i]++;
            if (accepted) accept_counts_[i]++;
            break;
        }
    }
    total_updates_++;
    if (total_updates_ % update_interval_ == 0) {
        rebalance();
    }
}

void MoveProbabilities::rebalance() {
    size_t n = move_types_.size();
    double floor = 0.05;

    std::vector<double> rates(n);
    for (size_t i = 0; i < n; ++i) {
        rates[i] = static_cast<double>(accept_counts_[i]) /
                   std::max(total_counts_[i], 1);
    }

    double total = 0.0;
    for (double r : rates) total += r;
    total += 1e-10;

    std::vector<double> probs(n);
    for (size_t i = 0; i < n; ++i) {
        probs[i] = rates[i] / total;
    }

    // Iteratively enforce floor and redistribute
    for (int iter = 0; iter < 3; ++iter) {
        double deficit = 0.0;
        int above_floor = 0;
        for (size_t i = 0; i < n; ++i) {
            if (probs[i] < floor) {
                deficit += floor - probs[i];
                probs[i] = floor;
            } else {
                above_floor++;
            }
        }
        if (deficit > 0 && above_floor > 0) {
            for (size_t i = 0; i < n; ++i) {
                if (probs[i] > floor) {
                    probs[i] -= deficit / above_floor;
                    probs[i] = std::max(probs[i], floor);
                }
            }
        }
    }

    total = 0.0;
    for (double p : probs) total += p;
    for (size_t i = 0; i < n; ++i) {
        probs_[i] = probs[i] / total;
    }
}

}  // namespace cbls
