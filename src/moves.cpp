#include "cbls/moves.h"

#include "cbls/move_generator.h"
#include "cbls/randomize.h"

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
    // Through the shared window, so an infinite bound cannot cast to INT64_MIN
    // (#112). Inert on a finite domain within +/-2^53. An empty window means the
    // domain lies wholly past 2^53, where no int64_t range names it, so this move
    // is dropped rather than drawn from a range that does not exist (#114). The
    // int_dec/int_inc moves above are NOT dropped there and are no-ops when the
    // ulp exceeds 1 — pre-existing, and out of #114's scope.
    const DomainWindow w = int_sample_window(var);
    if (w.lo <= w.hi) {
        Move m;
        m.move_type = "int_rand";
        auto new_val = static_cast<double>(
            rng.integers(static_cast<int64_t>(w.lo), static_cast<int64_t>(w.hi) + 1));
        m.changes.push_back({var.id, new_val, {}});
        moves.push_back(m);
    }
    return moves;
}

static std::vector<Move> float_moves(const Variable& var, RNG& rng, double sigma_frac = 0.1) {
    // Scale the step by the shared sampling window rather than the raw bounds:
    // an unbounded domain gives an infinite width, and normal(0, inf) is NaN
    // (#112). Inert on a finite domain. The clamp still uses the real bounds —
    // it is what keeps the move in the actual domain.
    const DomainWindow w = domain_window(var);
    double sigma = (w.hi - w.lo) * sigma_frac;
    double new_val = var.value + rng.normal(0, sigma);
    new_val = std::clamp(new_val, var.lb, var.ub);
    Move m;
    m.move_type = "float_perturb";
    m.changes.push_back({var.id, new_val, {}});
    return {m};
}

// The second position of the move pair. Uniform over `{0..n-1} \ {i}` with no
// neighbour list -- which is the pre-#165 draw, verbatim, and the reason the
// uniform branch is written out here rather than reached through the granular
// one. With a list, `elements[i]`'s nearest neighbours are the candidates, and
// the uniform draw is the fallback when the list names nobody usable.
static int pick_second_position(const Variable& var, RNG& rng, int n,
                                const NeighbourList* neighbours, int i) {
    if (neighbours != nullptr && !neighbours->empty()) {
        const int32_t here = var.elements[static_cast<size_t>(i)];
        const ConstSpan<int32_t> nb = neighbours->of(here);
        if (!nb.empty()) {
            // elements -> position. Rebuilt per call: a List's elements are a
            // permutation that every accepted move rewrites, so a cached index
            // would have to be invalidated by the search rather than by this
            // function. O(n) against the O(n) each candidate move already costs.
            std::vector<int32_t> pos(static_cast<size_t>(n), -1);
            for (int p = 0; p < n; ++p) {
                const int32_t e = var.elements[static_cast<size_t>(p)];
                if (e >= 0 && e < n) {
                    pos[static_cast<size_t>(e)] = p;
                }
            }
            const int32_t f =
                nb[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(nb.size())))];
            const int32_t j = (f >= 0 && f < n) ? pos[static_cast<size_t>(f)] : -1;
            if (j >= 0 && j != i) {
                return j;
            }
        }
    }
    int j = static_cast<int>(rng.integers(0, n - 1));
    if (j >= i) {
        j++;  // ensure i != j
    }
    return j;
}

static void list_moves(const Variable& var, RNG& rng, std::vector<Move>& moves,
                       const NeighbourList* neighbours) {
    int n = static_cast<int>(var.elements.size());
    if (n < 2) {
        return;
    }

    int i = static_cast<int>(rng.integers(0, n));
    int j = pick_second_position(var, rng, n, neighbours, i);

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
        if (j > i) {
            insert_pos = std::max(0, j - 2);
        }
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
        if (j > i) {
            insert_pos = std::max(0, j - 3);
        }
        insert_pos = std::min(insert_pos, static_cast<int>(new_elems.size()));
        new_elems.insert(new_elems.begin() + insert_pos, e2);
        new_elems.insert(new_elems.begin() + insert_pos, e1);
        new_elems.insert(new_elems.begin() + insert_pos, e0);
        m.changes.push_back({var.id, 0.0, new_elems});
        moves.push_back(m);
    }
}

// The current subset, its complement and the membership flag, which all three
// Set moves read. Split out of set_moves so that function is three independent
// move constructions rather than one block that also owns this bookkeeping.
struct SetPartition {
    std::vector<int32_t> in_set;
    std::vector<int32_t> not_in;
    std::vector<bool> in_flag;  // indexed by element, over the universe
};

static SetPartition partition_set(const Variable& var) {
    SetPartition p;
    p.in_set.assign(var.elements.begin(), var.elements.end());
    // For set vars, elements stores the current set. Universe is {0..universe_size-1}
    p.in_flag.assign(static_cast<size_t>(var.universe_size), false);
    for (int32_t e : var.elements) {
        if (e >= 0 && e < var.universe_size) {
            p.in_flag[static_cast<size_t>(e)] = true;
        }
    }
    for (int32_t i = 0; i < var.universe_size; ++i) {
        if (!p.in_flag[static_cast<size_t>(i)]) {
            p.not_in.push_back(i);
        }
    }
    return p;
}

// The element a set_add / set_swap brings in. Uniform over the complement with
// no neighbour list -- the pre-#165 draw, verbatim. With a list, the nearest
// neighbour of a randomly chosen SELECTED element that is not already in the
// set, which is the granular form of "grow the subset where it already is"
// rather than anywhere in the universe. Falls back to the uniform draw when the
// list offers nothing usable.
//
// PRECONDITION: `part.not_in` is non-empty.
static int32_t pick_added_element(const Variable& var, RNG& rng, const SetPartition& part,
                                  const NeighbourList* neighbours) {
    if (neighbours != nullptr && !neighbours->empty() && !part.in_set.empty()) {
        const int32_t seed = part.in_set[static_cast<size_t>(
            rng.integers(0, static_cast<int64_t>(part.in_set.size())))];
        for (int32_t f : neighbours->of(seed)) {
            if (f >= 0 && f < var.universe_size && !part.in_flag[static_cast<size_t>(f)]) {
                return f;
            }
        }
    }
    return part
        .not_in[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(part.not_in.size())))];
}

// PRECONDITION: `part.in_set` is non-empty.
static int32_t pick_removed_element(RNG& rng, const SetPartition& part) {
    return part
        .in_set[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(part.in_set.size())))];
}

static void set_add_move(const Variable& var, RNG& rng, const SetPartition& part,
                         const NeighbourList* neighbours, std::vector<Move>& moves) {
    if (part.not_in.empty() || static_cast<int>(var.elements.size()) >= var.max_size) {
        return;
    }
    Move m;
    m.move_type = "set_add";
    auto new_elems = var.elements;
    new_elems.push_back(pick_added_element(var, rng, part, neighbours));
    m.changes.push_back({var.id, 0.0, new_elems});
    moves.push_back(m);
}

static void set_remove_move(const Variable& var, RNG& rng, const SetPartition& part,
                            std::vector<Move>& moves) {
    if (part.in_set.empty() || static_cast<int>(var.elements.size()) <= var.min_size) {
        return;
    }
    Move m;
    m.move_type = "set_remove";
    const int32_t rem_elem = pick_removed_element(rng, part);
    auto new_elems = var.elements;
    auto it = std::find(new_elems.begin(), new_elems.end(), rem_elem);
    if (it == new_elems.end()) {
        return;
    }
    new_elems.erase(it);
    m.changes.push_back({var.id, 0.0, new_elems});
    moves.push_back(m);
}

static void set_swap_move(const Variable& var, RNG& rng, const SetPartition& part,
                          const NeighbourList* neighbours, std::vector<Move>& moves) {
    if (part.in_set.empty() || part.not_in.empty()) {
        return;
    }
    Move m;
    m.move_type = "set_swap";
    // Draw order is add-then-remove, as it has always been: both draws come off
    // the search's RNG, so swapping them would shift every later draw.
    const int32_t add_elem = pick_added_element(var, rng, part, neighbours);
    const int32_t rem_elem = pick_removed_element(rng, part);
    auto new_elems = var.elements;
    auto it = std::find(new_elems.begin(), new_elems.end(), rem_elem);
    if (it == new_elems.end()) {
        return;
    }
    new_elems.erase(it);
    new_elems.push_back(add_elem);
    m.changes.push_back({var.id, 0.0, new_elems});
    moves.push_back(m);
}

static void set_moves(const Variable& var, RNG& rng, std::vector<Move>& moves,
                      const NeighbourList* neighbours) {
    const SetPartition part = partition_set(var);
    set_add_move(var, rng, part, neighbours, moves);
    set_remove_move(var, rng, part, moves);
    set_swap_move(var, rng, part, neighbours, moves);
}

void generate_standard_moves(const Variable& var, RNG& rng, std::vector<Move>& out,
                             const NeighbourList* neighbours) {
    switch (var.type) {
        case VarType::Bool: {
            std::vector<Move> m = bool_moves(var);
            out.insert(out.end(), m.begin(), m.end());
            return;
        }
        case VarType::Int: {
            std::vector<Move> m = int_moves(var, rng);
            out.insert(out.end(), m.begin(), m.end());
            return;
        }
        case VarType::Float: {
            std::vector<Move> m = float_moves(var, rng);
            out.insert(out.end(), m.begin(), m.end());
            return;
        }
        case VarType::List:
            list_moves(var, rng, out, neighbours);
            return;
        case VarType::Set:
            set_moves(var, rng, out, neighbours);
            return;
    }
}

std::vector<Move> generate_standard_moves(const Variable& var, RNG& rng) {
    std::vector<Move> moves;
    generate_standard_moves(var, rng, moves, nullptr);
    return moves;
}

std::vector<int32_t> apply_move(Model& model, const Move& move) {
    std::vector<int32_t> changed;
    changed.reserve(move.changes.size());
    for (const auto& change : move.changes) {
        auto& var = model.var_mut(change.var_id);
        if (is_structured(var.type)) {
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

}  // namespace cbls
