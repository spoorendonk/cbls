#include "cbls/moves.h"

#include "cbls/move_generator.h"
#include "cbls/randomize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <utility>

namespace cbls {

// ---------------------------------------------------------------------------
// The positional edit representation (#164)
// ---------------------------------------------------------------------------

Move::Change scalar_change(int32_t var_id, double new_value) {
    Move::Change change;
    change.var_id = var_id;
    change.new_value = new_value;
    return change;
}

Move::Change edit_change(int32_t var_id, const ElementEdit& first, const ElementEdit& second) {
    Move::Change change;
    change.var_id = var_id;
    change.edits[0] = first;
    change.edits[1] = second;
    return change;
}

Move::Change replace_change(int32_t var_id, std::vector<int32_t> replacement) {
    Move::Change change;
    change.var_id = var_id;
    change.edits[0].kind = EditKind::Replace;
    change.replacement = std::move(replacement);
    return change;
}

namespace {

/// True when `[lo, hi]` is a valid inclusive range of `elements`.
bool in_range(const std::vector<int32_t>& elements, int32_t lo, int32_t hi) {
    return lo >= 0 && hi >= lo && static_cast<size_t>(hi) < elements.size();
}

using Iter = std::vector<int32_t>::iterator;

Iter at(std::vector<int32_t>& elements, int32_t pos) {
    return elements.begin() + static_cast<std::ptrdiff_t>(pos);
}

/// Erase `length` elements at `from` and reinsert them, in order, at position
/// `to` of the shortened vector: one allocation-free `std::rotate`.
/// `to == from` is the identity.
void apply_move_segment(const ElementEdit& edit, std::vector<int32_t>& elements) {
    const auto n = static_cast<int32_t>(elements.size());
    if (edit.length <= 0 || edit.from < 0 || edit.to < 0 || edit.from + edit.length > n ||
        edit.to + edit.length > n) {
        return;
    }
    if (edit.to < edit.from) {
        std::rotate(at(elements, edit.to), at(elements, edit.from),
                    at(elements, edit.from + edit.length));
    } else if (edit.to > edit.from) {
        // Position `to` of the shortened vector is position `to + length` of
        // this one, since the erased run sits before it.
        std::rotate(at(elements, edit.from), at(elements, edit.from + edit.length),
                    at(elements, edit.to + edit.length));
    }
}

void apply_one_edit(const ElementEdit& edit, const std::vector<int32_t>& replacement,
                    std::vector<int32_t>& elements) {
    const auto n = static_cast<int32_t>(elements.size());
    switch (edit.kind) {
        case EditKind::None:
            return;
        case EditKind::Replace:
            elements = replacement;
            return;
        case EditKind::Swap:
            if (in_range(elements, 0, edit.from) && in_range(elements, 0, edit.to)) {
                std::swap(elements[static_cast<size_t>(edit.from)],
                          elements[static_cast<size_t>(edit.to)]);
            }
            return;
        case EditKind::Reverse:
            if (in_range(elements, edit.from, edit.to)) {
                std::reverse(at(elements, edit.from), at(elements, edit.to + 1));
            }
            return;
        case EditKind::MoveSegment:
            apply_move_segment(edit, elements);
            return;
        case EditKind::Insert:
            if (edit.from >= 0 && edit.from <= n) {
                elements.insert(at(elements, edit.from), edit.element);
            }
            return;
        case EditKind::Erase:
            if (in_range(elements, 0, edit.from)) {
                elements.erase(at(elements, edit.from));
            }
            return;
        case EditKind::Assign:
            if (in_range(elements, 0, edit.from)) {
                elements[static_cast<size_t>(edit.from)] = edit.element;
            }
            return;
    }
}

/// Whether ONE edit is inert on `elements`. See `change_is_noop`.
///
/// Every kind tests its range first, and an edit that does not fit is inert,
/// because `apply_one_edit` ignores it (#156); the two must agree. Reachable
/// when Python overwrites `Variable.elements` under a built change.
bool edit_is_noop(const ElementEdit& edit, const std::vector<int32_t>& replacement,
                  const std::vector<int32_t>& elements) {
    const auto n = static_cast<int32_t>(elements.size());
    switch (edit.kind) {
        case EditKind::None:
            return true;
        case EditKind::Replace:
            return replacement == elements;
        case EditKind::Swap:
            return edit.from == edit.to || !in_range(elements, 0, edit.from) ||
                   !in_range(elements, 0, edit.to) ||
                   elements[static_cast<size_t>(edit.from)] ==
                       elements[static_cast<size_t>(edit.to)];
        case EditKind::Reverse:
            return edit.from >= edit.to || !in_range(elements, edit.from, edit.to);
        case EditKind::MoveSegment:
            return edit.from == edit.to || edit.length <= 0 || edit.from < 0 || edit.to < 0 ||
                   edit.from + edit.length > n || edit.to + edit.length > n;
        case EditKind::Insert:
            return edit.from < 0 || edit.from > n;
        case EditKind::Erase:
            return !in_range(elements, 0, edit.from);
        case EditKind::Assign:
            return !in_range(elements, 0, edit.from) ||
                   elements[static_cast<size_t>(edit.from)] == edit.element;
    }
    return true;
}

}  // namespace

void apply_element_edits(const Move::Change& change, std::vector<int32_t>& elements) {
    for (const ElementEdit& edit : change.edits) {
        apply_one_edit(edit, change.replacement, elements);
    }
}

std::vector<int32_t> elements_after(const Move::Change& change,
                                    const std::vector<int32_t>& elements) {
    std::vector<int32_t> result = elements;
    apply_element_edits(change, result);
    return result;
}

bool change_is_noop(const Move::Change& change, const std::vector<int32_t>& elements) {
    // Inert only if both edits are (`set_swap`'s pair never is: it brings in an
    // element the set does not hold).
    return std::all_of(change.edits.begin(), change.edits.end(),
                       [&change, &elements](const ElementEdit& edit) {
                           return edit_is_noop(edit, change.replacement, elements);
                       });
}

static std::vector<Move> bool_moves(const Variable& var) {
    Move m;
    m.move_type = "flip";
    m.changes.push_back(scalar_change(var.id, 1.0 - var.value));
    return {m};
}

static std::vector<Move> int_moves(const Variable& var, RNG& rng) {
    std::vector<Move> moves;
    if (var.value > var.lb) {
        Move m;
        m.move_type = "int_dec";
        m.changes.push_back(scalar_change(var.id, var.value - 1.0));
        moves.push_back(m);
    }
    if (var.value < var.ub) {
        Move m;
        m.move_type = "int_inc";
        m.changes.push_back(scalar_change(var.id, var.value + 1.0));
        moves.push_back(m);
    }
    // Through the shared window, so an infinite bound cannot cast to INT64_MIN
    // (#112). An empty window (domain wholly past 2^53) drops this move (#114);
    // int_dec/int_inc above are kept there and are no-ops when the ulp exceeds 1.
    const DomainWindow w = int_sample_window(var);
    if (w.lo <= w.hi) {
        Move m;
        m.move_type = "int_rand";
        auto new_val = static_cast<double>(
            rng.integers(static_cast<int64_t>(w.lo), static_cast<int64_t>(w.hi) + 1));
        m.changes.push_back(scalar_change(var.id, new_val));
        moves.push_back(m);
    }
    return moves;
}

static std::vector<Move> float_moves(const Variable& var, RNG& rng, double sigma_frac = 0.1) {
    // Step scaled by the sampling window, not the raw bounds: normal(0, inf) is
    // NaN (#112). The clamp uses the real bounds, keeping the move in domain.
    const DomainWindow w = domain_window(var);
    double sigma = (w.hi - w.lo) * sigma_frac;
    double new_val = var.value + rng.normal(0, sigma);
    new_val = std::clamp(new_val, var.lb, var.ub);
    Move m;
    m.move_type = "float_perturb";
    m.changes.push_back(scalar_change(var.id, new_val));
    return {m};
}

// The second position of the move pair. Without a neighbour list, uniform over
// `{0..n-1} \ {i}` (the pre-#165 draw, kept verbatim). With one, a neighbour of
// `elements[i]`, falling back to the uniform draw if none is usable.
static int pick_second_position(const Variable& var, RNG& rng, int n,
                                const NeighbourList* neighbours, int i) {
    if (neighbours != nullptr && !neighbours->empty()) {
        const int32_t here = var.elements[static_cast<size_t>(i)];
        const ConstSpan<int32_t> nb = neighbours->of(here);
        if (!nb.empty()) {
            // element -> position, rebuilt per call (every accepted move
            // rewrites the List, so a cache would need search-side
            // invalidation): O(universe) against each candidate's O(n).
            // Sized by the UNIVERSE, not the length (#164), else ids >= n drop
            // to the uniform draw. `e < universe` is needed: Python can write
            // any id into `Variable.elements` (#156).
            const auto universe = static_cast<size_t>(std::max(var.universe_size, 0));
            std::vector<int32_t> pos(universe, -1);
            for (int p = 0; p < n; ++p) {
                const int32_t e = var.elements[static_cast<size_t>(p)];
                if (e >= 0 && static_cast<size_t>(e) < universe) {
                    pos[static_cast<size_t>(e)] = p;
                }
            }
            const int32_t f =
                nb[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(nb.size())))];
            const int32_t j =
                (f >= 0 && static_cast<size_t>(f) < universe) ? pos[static_cast<size_t>(f)] : -1;
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

// Which universe elements a List holds. O(universe + n), built only on
// length-changing paths.
//
// Writes are bounds-tested: Python can write any id into `Variable.elements`
// (#156). An out-of-universe id is not recorded, so the list looks shorter to
// the insert guard; at worst that yields a candidate the batch scores normally.
static std::vector<bool> list_membership(const Variable& var) {
    std::vector<bool> present(static_cast<size_t>(std::max(var.universe_size, 0)), false);
    for (int32_t e : var.elements) {
        if (e >= 0 && static_cast<size_t>(e) < present.size()) {
            present[static_cast<size_t>(e)] = true;
        }
    }
    return present;
}

/// The elements of `var`'s universe it does not hold: the membership flag over
/// the universe, and the complement as a list to draw from.
///
/// Built once per `generate_standard_moves` call, shared by the two moves that
/// need it: O(universe) with two allocations, so built only when one of them
/// can apply (never on a permutation List).
struct ListComplement {
    std::vector<bool> present;  // indexed by element, over the universe
    std::vector<int32_t> absent;
};

static ListComplement list_complement(const Variable& var) {
    ListComplement c;
    c.present = list_membership(var);
    for (int32_t e = 0; e < var.universe_size; ++e) {
        if (!c.present[static_cast<size_t>(e)]) {
            c.absent.push_back(e);
        }
    }
    return c;
}

// One element of the universe that `var` does not hold, or -1 if it holds them
// all.
//
// With a neighbour list, an absent neighbour of a held element, as
// `pick_added_element` does for a Set; else, or if none is absent, uniform. A
// partial list restricts where the search looks, never what it can reach.
static int32_t pick_absent_element(const Variable& var, RNG& rng, const NeighbourList* neighbours,
                                   const ListComplement& comp) {
    const auto n = static_cast<int32_t>(var.elements.size());
    if (neighbours != nullptr && !neighbours->empty() && n > 0) {
        const int32_t seed = var.elements[static_cast<size_t>(rng.integers(0, n))];
        for (int32_t f : neighbours->of(seed)) {
            if (f >= 0 && static_cast<size_t>(f) < comp.present.size() &&
                !comp.present[static_cast<size_t>(f)]) {
                return f;
            }
        }
    }
    if (comp.absent.empty()) {
        return -1;
    }
    return comp
        .absent[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(comp.absent.size())))];
}

// Insert one absent element at a random position (#164).
//
// Guards before draws, so a permutation List (`list_var(n)`, pinned at
// max_size) consumes no random numbers and keeps its trajectory bit-identical.
static void list_insert_move(const Variable& var, RNG& rng, const NeighbourList* neighbours,
                             const ListComplement& comp, std::vector<Move>& moves) {
    const auto n = static_cast<int32_t>(var.elements.size());
    if (n >= var.max_size) {
        return;
    }
    const int32_t chosen = pick_absent_element(var, rng, neighbours, comp);
    if (chosen < 0) {
        return;
    }
    const int64_t pos = rng.integers(0, n + 1);
    Move m;
    m.move_type = "list_insert";
    m.changes.push_back(edit_change(var.id, insert_edit(static_cast<int32_t>(pos), chosen)));
    moves.push_back(m);
}

// Exchange the element at a random position for one the List does not hold: the
// only move that changes a List's MEMBERSHIP without changing its length (#164).
//
// Without it a `min_len == max_len < universe` List keeps its initial
// membership all run. The List analogue of `set_swap`; guarded before it
// draws, so a permutation List consumes nothing.
//
// Not offered for a partition member (`list_moves` checks): `partition_swap` is
// the exchange that keeps the cover.
static void list_exchange_move(const Variable& var, RNG& rng, const NeighbourList* neighbours,
                               const ListComplement& comp, std::vector<Move>& moves) {
    const auto n = static_cast<int32_t>(var.elements.size());
    if (n <= 0 || var.universe_size <= n) {
        return;
    }
    const int32_t chosen = pick_absent_element(var, rng, neighbours, comp);
    if (chosen < 0) {
        return;
    }
    const auto pos = static_cast<int32_t>(rng.integers(0, n));
    Move m;
    m.move_type = "list_exchange";
    m.changes.push_back(edit_change(var.id, assign_edit(pos, chosen)));
    moves.push_back(m);
}

// Drop the element at a random position (#164). Guards before draws, as above:
// a permutation List sits at min_size and returns without drawing.
static void list_remove_move(const Variable& var, RNG& rng, std::vector<Move>& moves) {
    const auto n = static_cast<int32_t>(var.elements.size());
    if (var.partitioned || n <= var.min_size || n <= 0) {
        return;
    }
    const auto pos = static_cast<int32_t>(rng.integers(0, n));
    Move m;
    m.move_type = "list_remove";
    m.changes.push_back(edit_change(var.id, erase_edit(pos)));
    moves.push_back(m);
}

// The five length-preserving intra-list moves. Each guard requires the segment
// to fit strictly inside the list (relocating all of it is a no-op): or-opt(2)
// needs n >= 3, or-opt(3) n >= 4, and n < 2 has nothing to reorder.
static void list_reorder_moves(const Variable& var, RNG& rng, std::vector<Move>& moves,
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
        m.changes.push_back(edit_change(var.id, swap_edit(i, j)));
        moves.push_back(m);
    }

    // 2-opt reverse
    {
        Move m;
        m.move_type = "list_2opt";
        m.changes.push_back(edit_change(var.id, reverse_edit(std::min(i, j), std::max(i, j))));
        moves.push_back(m);
    }

    // Relocate: remove element at position i, insert at position j.
    //
    // One `segment_edit` each. The segment is erased first, so a target to its
    // right shifts left by its length, and the clamp is against the shortened
    // vector.
    {
        Move m;
        m.move_type = "list_relocate";
        const int insert_pos = (j > i) ? j - 1 : j;
        m.changes.push_back(edit_change(var.id, segment_edit(i, 1, insert_pos)));
        moves.push_back(m);
    }

    // Or-opt(2): relocate a consecutive pair
    if (n >= 3 && i < n - 1) {
        Move m;
        m.move_type = "list_or_opt_2";
        int insert_pos = j;
        if (j > i) {
            insert_pos = std::max(0, j - 2);
        }
        insert_pos = std::min(insert_pos, n - 2);
        m.changes.push_back(edit_change(var.id, segment_edit(i, 2, insert_pos)));
        moves.push_back(m);
    }

    // Or-opt(3): relocate a consecutive triple
    if (n >= 4 && i < n - 2) {
        Move m;
        m.move_type = "list_or_opt_3";
        int insert_pos = j;
        if (j > i) {
            insert_pos = std::max(0, j - 3);
        }
        insert_pos = std::min(insert_pos, n - 3);
        m.changes.push_back(edit_change(var.id, segment_edit(i, 3, insert_pos)));
        moves.push_back(m);
    }
}

// A List's typed moves: the five length-preserving reorderings first, then the
// three that can only apply to a List that is not a permutation (#164) -- insert,
// the fixed-length membership exchange, and remove.
//
// The tail moves must test their guards before touching the RNG: on a
// permutation List they then draw nothing, keeping the pre-#164 draw sequence.
// A draw ahead of a guard would break that.
static void list_moves(const Variable& var, RNG& rng, std::vector<Move>& moves,
                       const NeighbourList* neighbours) {
    list_reorder_moves(var, rng, moves, neighbours);
    const auto n = static_cast<int32_t>(var.elements.size());
    // Scanned once, only where insert or exchange can apply: never on a
    // permutation List (so it neither scans nor draws, #164) nor a partition
    // member (its membership belongs to the partition).
    if (!var.partitioned &&
        ((n < var.max_size && var.universe_size > 0) || (n > 0 && var.universe_size > n))) {
        const ListComplement comp = list_complement(var);
        list_insert_move(var, rng, neighbours, comp, moves);
        list_exchange_move(var, rng, neighbours, comp, moves);
    }
    list_remove_move(var, rng, moves);
}

// The current subset, its complement and the membership flag, which all three
// Set moves read.
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

// The element a set_add / set_swap brings in. Without a neighbour list, uniform
// over the complement (the pre-#165 draw). With one, the nearest absent
// neighbour of a random selected element, falling back to uniform.
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
    m.changes.push_back(
        edit_change(var.id, insert_edit(static_cast<int32_t>(var.elements.size()),
                                        pick_added_element(var, rng, part, neighbours))));
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
    const auto it = std::find(var.elements.begin(), var.elements.end(), rem_elem);
    if (it == var.elements.end()) {
        return;
    }
    m.changes.push_back(
        edit_change(var.id, erase_edit(static_cast<int32_t>(it - var.elements.begin()))));
    moves.push_back(m);
}

static void set_swap_move(const Variable& var, RNG& rng, const SetPartition& part,
                          const NeighbourList* neighbours, std::vector<Move>& moves) {
    if (part.in_set.empty() || part.not_in.empty()) {
        return;
    }
    Move m;
    m.move_type = "set_swap";
    // Draw order must stay add-then-remove: both use the search's RNG, so
    // swapping them shifts every later draw.
    const int32_t add_elem = pick_added_element(var, rng, part, neighbours);
    const int32_t rem_elem = pick_removed_element(rng, part);
    const auto it = std::find(var.elements.begin(), var.elements.end(), rem_elem);
    if (it == var.elements.end()) {
        return;
    }
    // Erase in place, then append to the SHORTENED vector: a Set read pairwise
    // (`pair_lambda_sum`) sees element order.
    const auto pos = static_cast<int32_t>(it - var.elements.begin());
    m.changes.push_back(
        edit_change(var.id, erase_edit(pos),
                    insert_edit(static_cast<int32_t>(var.elements.size()) - 1, add_elem)));
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
            out.insert(out.end(), std::make_move_iterator(m.begin()),
                       std::make_move_iterator(m.end()));
            return;
        }
        case VarType::Int: {
            std::vector<Move> m = int_moves(var, rng);
            out.insert(out.end(), std::make_move_iterator(m.begin()),
                       std::make_move_iterator(m.end()));
            return;
        }
        case VarType::Float: {
            std::vector<Move> m = float_moves(var, rng);
            out.insert(out.end(), std::make_move_iterator(m.begin()),
                       std::make_move_iterator(m.end()));
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

// ---------------------------------------------------------------------------
// Inter-list moves over a ListPartition (#164)
// ---------------------------------------------------------------------------
//
// The standard routing neighbourhood: relocate a customer to another route, swap
// two customers between routes, exchange two route tails (2-opt*), and -- under
// `Cover::AtMostOnce` only -- insert an unassigned element or drop one.
//
// Each preserves the cover by construction (as a penalty row, every repair would
// pass through a doubly-served or unserved state). They do not repair a cover
// Python broke via `Variable.elements` (#156): only the membership stamp
// indexes by element id, bounds-tested, so corruption is a wrong search, not a
// crash.

namespace {

/// The position in `dest` of the nearest neighbour of `e` that `dest` holds, or
/// -1 when there is no neighbour list or it names nobody `dest` holds. Draws no
/// random numbers; callers decide what a miss means.
int32_t neighbour_position(const std::vector<int32_t>& dest, int32_t e,
                           const NeighbourList* neighbours) {
    if (neighbours == nullptr || neighbours->empty()) {
        return -1;
    }
    for (int32_t f : neighbours->of(e)) {
        const auto it = std::find(dest.begin(), dest.end(), f);
        if (it != dest.end()) {
            return static_cast<int32_t>(it - dest.begin());
        }
    }
    return -1;
}

/// Insert position for `e` in `dest`: just after a nearest neighbour of `e` that
/// `dest` already holds, else uniform over the `|dest| + 1` gaps.
int32_t partition_insert_pos(const std::vector<int32_t>& dest, int32_t e, RNG& rng,
                             const NeighbourList* neighbours) {
    const int32_t near = neighbour_position(dest, e, neighbours);
    if (near >= 0) {
        return near + 1;
    }
    return static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(dest.size()) + 1));
}

/// Append `move` unless every change is a no-op: one would cost two
/// `delta_evaluate` passes to reject, and the kick would count it as made.
void push_if_changed(const Model& model, Move&& move, std::vector<Move>& out) {
    for (const Move::Change& change : move.changes) {
        if (!change_is_noop(change, model.var(change.var_id).elements)) {
            out.push_back(std::move(move));
            return;
        }
    }
}

/// Move one element out of `a` and into `b`.
void partition_relocate(const Model& model, int32_t a, int32_t b, RNG& rng,
                        const NeighbourList* neighbours, std::vector<Move>& out) {
    const Variable& va = model.var(a);
    const Variable& vb = model.var(b);
    if (va.elements.empty() || static_cast<int32_t>(va.elements.size()) <= va.min_size ||
        static_cast<int32_t>(vb.elements.size()) >= vb.max_size) {
        return;
    }
    const auto i = static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(va.elements.size())));
    const int32_t e = va.elements[static_cast<size_t>(i)];
    const int32_t pos = partition_insert_pos(vb.elements, e, rng, neighbours);
    Move m;
    m.move_type = "partition_relocate";
    m.changes.push_back(edit_change(a, erase_edit(i)));
    m.changes.push_back(edit_change(b, insert_edit(pos, e)));
    push_if_changed(model, std::move(m), out);
}

/// Exchange one element of `a` with one of `b`. Length-preserving on both, so it
/// is the move that still applies when every list sits at a length bound.
void partition_swap(const Model& model, int32_t a, int32_t b, RNG& rng,
                    const NeighbourList* neighbours, std::vector<Move>& out) {
    const Variable& va = model.var(a);
    const Variable& vb = model.var(b);
    if (va.elements.empty() || vb.elements.empty()) {
        return;
    }
    const auto i = static_cast<size_t>(rng.integers(0, static_cast<int64_t>(va.elements.size())));
    // With a neighbour list, the element of `b` nearest `a[i]`; on a miss,
    // uniform, so a partial list never restricts what the search can reach.
    const int32_t near = neighbour_position(vb.elements, va.elements[i], neighbours);
    const auto j =
        (near >= 0)
            ? static_cast<size_t>(near)
            : static_cast<size_t>(rng.integers(0, static_cast<int64_t>(vb.elements.size())));
    Move m;
    m.move_type = "partition_swap";
    // One assignment per side: across two variables, two independent writes.
    m.changes.push_back(edit_change(a, assign_edit(static_cast<int32_t>(i), vb.elements[j])));
    m.changes.push_back(edit_change(b, assign_edit(static_cast<int32_t>(j), va.elements[i])));
    push_if_changed(model, std::move(m), out);
}

/// `x[0, x_end)` then `y[y_from, end)`: one side of a 2-opt* tail exchange,
/// allocated once at its final size: one allocation per side, the whole cost of
/// the change's `replacement` (#274).
std::vector<int32_t> head_then_tail(const std::vector<int32_t>& x, size_t x_end,
                                    const std::vector<int32_t>& y, size_t y_from) {
    std::vector<int32_t> joined;
    joined.reserve(x_end + (y.size() - y_from));
    joined.insert(joined.end(), x.begin(), x.begin() + static_cast<std::ptrdiff_t>(x_end));
    joined.insert(joined.end(), y.begin() + static_cast<std::ptrdiff_t>(y_from), y.end());
    return joined;
}

/// Exchange the tails of `a` and `b` at independently drawn cut points -- the
/// 2-opt* of the routing literature. Membership over the pair is preserved, but
/// the two lengths are not, so both bounds are checked before the move is built.
void partition_two_opt_star(const Model& model, int32_t a, int32_t b, RNG& rng,
                            std::vector<Move>& out) {
    const Variable& va = model.var(a);
    const Variable& vb = model.var(b);
    const auto na = static_cast<int64_t>(va.elements.size());
    const auto nb = static_cast<int64_t>(vb.elements.size());
    const auto p = static_cast<size_t>(rng.integers(0, na + 1));
    const auto q = static_cast<size_t>(rng.integers(0, nb + 1));
    const auto len_a = static_cast<int32_t>(p + (static_cast<size_t>(nb) - q));
    const auto len_b = static_cast<int32_t>(q + (static_cast<size_t>(na) - p));
    if (len_a < va.min_size || len_a > va.max_size || len_b < vb.min_size || len_b > vb.max_size) {
        return;
    }
    Move m;
    m.move_type = "partition_2opt_star";
    m.changes.push_back(replace_change(a, head_then_tail(va.elements, p, vb.elements, q)));
    m.changes.push_back(replace_change(b, head_then_tail(vb.elements, q, va.elements, p)));
    push_if_changed(model, std::move(m), out);
}

/// The elements of the partition's universe that no member list holds.
///
/// Recomputed every time, never cached: Python, LNS, the kick and pool restarts
/// all move the assignment without notifying a generator
/// (`MoveGenerator::on_commit`). O(sum |lists| + universe); stamp writes are
/// bounds-tested, so an out-of-universe id is ignored (#156).
std::vector<int32_t> partition_unassigned(const Model& model, const ListPartition& part) {
    std::vector<bool> taken(static_cast<size_t>(std::max(part.universe_size, 0)), false);
    for (int32_t vid : part.list_ids) {
        for (int32_t e : model.var(vid).elements) {
            if (e >= 0 && static_cast<size_t>(e) < taken.size()) {
                taken[static_cast<size_t>(e)] = true;
            }
        }
    }
    std::vector<int32_t> free_elements;
    for (int32_t e = 0; e < part.universe_size; ++e) {
        if (!taken[static_cast<size_t>(e)]) {
            free_elements.push_back(e);
        }
    }
    return free_elements;
}

/// Bring one unassigned element into `a`. `Cover::AtMostOnce` only -- under
/// `Exact` there is nothing unassigned to bring in, by the invariant.
void partition_insert(const Model& model, const ListPartition& part, int32_t a, RNG& rng,
                      const NeighbourList* neighbours, std::vector<Move>& out) {
    const Variable& va = model.var(a);
    if (static_cast<int32_t>(va.elements.size()) >= va.max_size) {
        return;
    }
    const std::vector<int32_t> free_elements = partition_unassigned(model, part);
    if (free_elements.empty()) {
        return;
    }
    const int32_t e = free_elements[static_cast<size_t>(
        rng.integers(0, static_cast<int64_t>(free_elements.size())))];
    const int32_t pos = partition_insert_pos(va.elements, e, rng, neighbours);
    Move m;
    m.move_type = "partition_insert";
    m.changes.push_back(edit_change(a, insert_edit(pos, e)));
    push_if_changed(model, std::move(m), out);
}

/// Drop one element of `a`, leaving it unassigned. `Cover::AtMostOnce` only.
void partition_remove(const Model& model, int32_t a, RNG& rng, std::vector<Move>& out) {
    const Variable& va = model.var(a);
    if (va.elements.empty() || static_cast<int32_t>(va.elements.size()) <= va.min_size) {
        return;
    }
    const auto i = static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(va.elements.size())));
    Move m;
    m.move_type = "partition_remove";
    m.changes.push_back(edit_change(a, erase_edit(i)));
    push_if_changed(model, std::move(m), out);
}

/// The move kinds a partition admits, given how many lists it has and what its
/// cover permits. Drawn from uniformly, so this list is also the mix.
enum class PartitionMoveKind : std::uint8_t { Relocate, Swap, TwoOptStar, Insert, Remove };

/// Written into `out`; returns how many. A fixed array: this runs per proposed
/// candidate, and a candidate must not cost a heap allocation (nor does `Move`,
/// #274).
size_t applicable_kinds(const ListPartition& part, std::array<PartitionMoveKind, 5>& out) {
    size_t n = 0;
    if (part.list_ids.size() >= 2) {
        out[n++] = PartitionMoveKind::Relocate;
        out[n++] = PartitionMoveKind::Swap;
        out[n++] = PartitionMoveKind::TwoOptStar;
    }
    if (part.cover == Cover::AtMostOnce) {
        // Never under Exact: an insert would double-serve, a remove leave an
        // element unserved, and no later move could repair either.
        out[n++] = PartitionMoveKind::Insert;
        out[n++] = PartitionMoveKind::Remove;
    }
    return n;
}

/// A member of `part` other than `a`, uniformly (shift past `a`).
/// PRECONDITION: at least two members, `a` one of them. A non-member `a` draws
/// over all members, defined but unsupported (it would double-serve);
/// `generate_partition_moves` refuses one first.
int32_t pick_other_list(const ListPartition& part, int32_t a, RNG& rng) {
    const auto count = static_cast<int64_t>(part.list_ids.size());
    const auto it = std::find(part.list_ids.begin(), part.list_ids.end(), a);
    const auto skip =
        (it == part.list_ids.end()) ? count : static_cast<int64_t>(it - part.list_ids.begin());
    int64_t pick = rng.integers(0, count - (skip < count ? 1 : 0));
    if (skip < count && pick >= skip) {
        ++pick;
    }
    return part.list_ids[static_cast<size_t>(pick)];
}

}  // namespace

void generate_partition_moves(const Model& model, int partition, int32_t anchor, RNG& rng,
                              std::vector<Move>& out, const NeighbourList* neighbours) {
    const std::vector<ListPartition>& partitions = model.list_partitions();
    if (partition < 0 || partition >= static_cast<int>(partitions.size())) {
        return;
    }
    const ListPartition& part = partitions[static_cast<size_t>(partition)];
    std::array<PartitionMoveKind, 5> kinds{};
    const size_t num_kinds = applicable_kinds(part, kinds);
    if (num_kinds == 0 || part.list_ids.empty()) {
        return;
    }
    const PartitionMoveKind kind =
        kinds[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(num_kinds)))];

    // The first list is the caller's anchor if named (the kick asks to move THIS
    // variable). The second is drawn only by the three kinds that need one.
    const int32_t a = (anchor >= 0) ? anchor
                                    : part.list_ids[static_cast<size_t>(rng.integers(
                                          0, static_cast<int64_t>(part.list_ids.size())))];
    // A caller's anchor must be a member: this is public (moves.h), and the moves
    // below preserve the cover only over a pair of member lists. O(members).
    if (anchor >= 0 &&
        std::find(part.list_ids.begin(), part.list_ids.end(), a) == part.list_ids.end()) {
        return;
    }

    switch (kind) {
        case PartitionMoveKind::Relocate:
            partition_relocate(model, a, pick_other_list(part, a, rng), rng, neighbours, out);
            return;
        case PartitionMoveKind::Swap:
            partition_swap(model, a, pick_other_list(part, a, rng), rng, neighbours, out);
            return;
        case PartitionMoveKind::TwoOptStar:
            partition_two_opt_star(model, a, pick_other_list(part, a, rng), rng, out);
            return;
        case PartitionMoveKind::Insert:
            partition_insert(model, part, a, rng, neighbours, out);
            return;
        case PartitionMoveKind::Remove:
            partition_remove(model, a, rng, out);
            return;
    }
}

namespace {

/// Where element `e` sits: `list` indexes `part.list_ids`, -1 for unassigned.
struct ElementSlot {
    int32_t list = -1;
    int32_t pos = -1;
};

/// The granular 2-opt* pair between two different lists that makes `e` (at
/// `a[i]`) and `f` (at `b[j]`) adjacent: `a[..i] + b[j..]` with `b[..j) +
/// a(i..]`, and `b[..j] + a[i..]` with `a[..i) + b(j..]`. Both keep the pair's
/// membership; each is offered only when the two new lengths fit their bounds.
void granular_two_opt_star(const Variable& va, int32_t i, const Variable& vb, int32_t j,
                           const Model& model, std::vector<Move>& out) {
    const auto na = static_cast<int32_t>(va.elements.size());
    const auto nb = static_cast<int32_t>(vb.elements.size());
    const auto fits = [](const Variable& v, int32_t len) {
        return len >= v.min_size && len <= v.max_size;
    };
    // e then f: a keeps its head through e, then takes b's tail from f.
    const auto at = [](int32_t pos) { return static_cast<size_t>(pos); };
    if (fits(va, i + 1 + nb - j) && fits(vb, j + na - i - 1)) {
        Move m;
        m.move_type = "g_2opt_star";
        m.changes.push_back(
            replace_change(va.id, head_then_tail(va.elements, at(i + 1), vb.elements, at(j))));
        m.changes.push_back(
            replace_change(vb.id, head_then_tail(vb.elements, at(j), va.elements, at(i + 1))));
        push_if_changed(model, std::move(m), out);
    }
    // f then e: b keeps its head through f, then takes a's tail from e.
    if (fits(vb, j + 1 + na - i) && fits(va, i + nb - j - 1)) {
        Move m;
        m.move_type = "g_2opt_star";
        m.changes.push_back(
            replace_change(va.id, head_then_tail(va.elements, at(i), vb.elements, at(j + 1))));
        m.changes.push_back(
            replace_change(vb.id, head_then_tail(vb.elements, at(j + 1), va.elements, at(i))));
        push_if_changed(model, std::move(m), out);
    }
}

/// Every granular move pairing `e` with one neighbour `f`, both assigned.
void granular_pair_moves(const Model& model, const ListPartition& part, int32_t e, ElementSlot se,
                         int32_t f, ElementSlot sf, std::vector<Move>& out) {
    const int32_t a = part.list_ids[static_cast<size_t>(se.list)];
    const int32_t b = part.list_ids[static_cast<size_t>(sf.list)];
    const Variable& va = model.var(a);
    const Variable& vb = model.var(b);
    const int32_t i = se.pos;
    const int32_t j = sf.pos;
    if (a == b) {
        // Within one list. Relocate e next to f: positions in the vector
        // SHORTENED by e's removal, where f has moved down one if it was after e.
        const int32_t jf = j > i ? j - 1 : j;
        for (const int32_t to : {jf + 1, jf}) {
            Move m;
            m.move_type = "g_relocate";
            m.changes.push_back(edit_change(a, segment_edit(i, 1, to)));
            push_if_changed(model, std::move(m), out);
        }
        Move sw;
        sw.move_type = "g_swap";
        sw.changes.push_back(edit_change(a, swap_edit(i, j)));
        push_if_changed(model, std::move(sw), out);
        // The two 2-opt reversals that leave e and f adjacent.
        const int32_t lo = std::min(i, j);
        const int32_t hi = std::max(i, j);
        for (const auto& range :
             {std::pair<int32_t, int32_t>{lo + 1, hi}, std::pair<int32_t, int32_t>{lo, hi - 1}}) {
            Move m;
            m.move_type = "g_2opt";
            m.changes.push_back(edit_change(a, reverse_edit(range.first, range.second)));
            push_if_changed(model, std::move(m), out);
        }
        return;
    }
    // Across two lists. Relocate e just after f and just before it.
    if (static_cast<int32_t>(va.elements.size()) > va.min_size &&
        static_cast<int32_t>(vb.elements.size()) < vb.max_size) {
        for (const int32_t to : {j + 1, j}) {
            Move m;
            m.move_type = "g_relocate";
            m.changes.push_back(edit_change(a, erase_edit(i)));
            m.changes.push_back(edit_change(b, insert_edit(to, e)));
            push_if_changed(model, std::move(m), out);
        }
    }
    Move sw;
    sw.move_type = "g_swap";
    sw.changes.push_back(edit_change(a, assign_edit(i, f)));
    sw.changes.push_back(edit_change(b, assign_edit(j, e)));
    push_if_changed(model, std::move(sw), out);
    granular_two_opt_star(va, i, vb, j, model, out);
}

/// Element -> (list, position) over one partition, recomputed per call (see
/// `partition_unassigned`). Packed as list * stride + pos into the caller's
/// reused vector, so allocation-free once grown. Bounds-tested (#156).
class PartitionSlots {
public:
    PartitionSlots(const Model& model, const ListPartition& part, std::vector<int64_t>& packed)
        : packed_(packed) {
        packed_.assign(static_cast<size_t>(part.universe_size), -1);
        for (const int32_t vid : part.list_ids) {
            stride_ =
                std::max<int64_t>(stride_, static_cast<int64_t>(model.var(vid).elements.size()));
        }
        for (size_t li = 0; li < part.list_ids.size(); ++li) {
            const std::vector<int32_t>& el = model.var(part.list_ids[li]).elements;
            for (size_t p = 0; p < el.size(); ++p) {
                if (el[p] >= 0 && static_cast<size_t>(el[p]) < packed_.size()) {
                    packed_[static_cast<size_t>(el[p])] =
                        (static_cast<int64_t>(li) * stride_) + static_cast<int64_t>(p);
                }
            }
        }
    }

    [[nodiscard]] ElementSlot of(int32_t x) const {
        ElementSlot s;
        if (x < 0 || static_cast<size_t>(x) >= packed_.size()) {
            return s;
        }
        const int64_t packed = packed_[static_cast<size_t>(x)];
        if (packed >= 0) {
            s.list = static_cast<int32_t>(packed / stride_);
            s.pos = static_cast<int32_t>(packed % stride_);
        }
        return s;
    }

private:
    std::vector<int64_t>& packed_;
    int64_t stride_ = 1;
};

/// An unassigned `e` (AtMostOnce) brought in just after and just before `f`.
void granular_insert_moves(const Model& model, const ListPartition& part, int32_t e, ElementSlot sf,
                           std::vector<Move>& out) {
    const Variable& vb = model.var(part.list_ids[static_cast<size_t>(sf.list)]);
    if (static_cast<int32_t>(vb.elements.size()) >= vb.max_size) {
        return;
    }
    for (const int32_t to : {sf.pos + 1, sf.pos}) {
        Move m;
        m.move_type = "g_insert";
        m.changes.push_back(edit_change(vb.id, insert_edit(to, e)));
        push_if_changed(model, std::move(m), out);
    }
}

/// `e` moved into every EMPTY member list: relocated there when assigned, inserted
/// when not. Granular pairs land next to an assigned `f`, so without these an
/// emptied list never reopens (Toth & Vigo's depot arcs).
void granular_empty_list_moves(const Model& model, const ListPartition& part, int32_t e,
                               ElementSlot se, std::vector<Move>& out) {
    if (se.list >= 0) {
        const Variable& va = model.var(part.list_ids[static_cast<size_t>(se.list)]);
        if (static_cast<int32_t>(va.elements.size()) <= va.min_size) {
            return;
        }
    }
    for (const int32_t vid : part.list_ids) {
        const Variable& vb = model.var(vid);
        if (!vb.elements.empty() || vb.max_size < 1) {
            continue;
        }
        Move m;
        if (se.list >= 0) {
            m.move_type = "g_relocate";
            m.changes.push_back(
                edit_change(part.list_ids[static_cast<size_t>(se.list)], erase_edit(se.pos)));
        } else {
            m.move_type = "g_insert";
        }
        m.changes.push_back(edit_change(vid, insert_edit(0, e)));
        push_if_changed(model, std::move(m), out);
    }
}

}  // namespace

void generate_granular_partition_moves(const Model& model, int partition, RNG& rng,
                                       const NeighbourList& neighbours, std::vector<Move>& out,
                                       std::vector<int64_t>& slot_scratch) {
    const std::vector<ListPartition>& partitions = model.list_partitions();
    if (partition < 0 || partition >= static_cast<int>(partitions.size())) {
        return;
    }
    const ListPartition& part = partitions[static_cast<size_t>(partition)];
    if (part.universe_size <= 0 || part.list_ids.empty()) {
        return;
    }
    const PartitionSlots slots(model, part, slot_scratch);
    const auto e = static_cast<int32_t>(rng.integers(0, part.universe_size));
    const ElementSlot se = slots.of(e);
    if (se.list < 0 && part.cover != Cover::AtMostOnce) {
        return;  // a cover something else broke; these moves do not repair one
    }
    if (se.list >= 0 && part.cover == Cover::AtMostOnce) {
        const Variable& va = model.var(part.list_ids[static_cast<size_t>(se.list)]);
        if (static_cast<int32_t>(va.elements.size()) > va.min_size) {
            Move m;
            m.move_type = "g_remove";
            m.changes.push_back(edit_change(va.id, erase_edit(se.pos)));
            push_if_changed(model, std::move(m), out);
        }
    }
    for (const int32_t f : neighbours.of(e)) {
        const ElementSlot sf = slots.of(f);
        if (f == e || sf.list < 0) {
            continue;
        }
        if (se.list >= 0) {
            granular_pair_moves(model, part, e, se, f, sf, out);
        } else {
            granular_insert_moves(model, part, e, sf, out);
        }
    }
    granular_empty_list_moves(model, part, e, se, out);
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
            // In place, allocating only past capacity. The edit is relative to
            // the assignment it was built against; the caller must apply it to
            // THAT one (`ElementEdit`, `StructuralBatch`).
            apply_element_edits(change, var.elements);
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
