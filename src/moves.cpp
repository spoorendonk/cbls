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
/// `to` of the shortened vector -- which is one `std::rotate` and therefore
/// allocation-free. `to == from` is the identity.
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
/// EVERY KIND TESTS ITS RANGE FIRST, and an edit that does not fit reads as
/// inert -- because that is what applying it does: `apply_one_edit` ignores an
/// out-of-range position rather than indexing it (#156). The two have to agree
/// or the predicate is lying about the thing it exists to answer, and "does not
/// fit" is reachable exactly when a change is replayed against an assignment
/// other than the one it was built on -- which is Python overwriting
/// `Variable.elements`, the case the bounds test is there for in the first
/// place.
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
    // A pair of edits is inert only if both halves are -- `set_swap`'s erase and
    // append are never both inert, since the element it brings in is by
    // construction one the set does not hold.
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
        m.changes.push_back(scalar_change(var.id, new_val));
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
    m.changes.push_back(scalar_change(var.id, new_val));
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
            // element -> position. Rebuilt per call: a List's elements are
            // rewritten by every accepted move, so a cached index would have to
            // be invalidated by the search rather than by this function.
            // O(universe) against the O(n) each candidate move already costs.
            //
            // Sized by the UNIVERSE, not by the length (#164). They coincide on a
            // permutation, which is all a List could be before #164; on a
            // variable-length List every element id >= n would otherwise be
            // dropped, silently turning granular guidance back into the uniform
            // draw for most of the universe. The `e < universe` test is not
            // decoration either: `Variable.elements` is writable from Python, so
            // an out-of-universe id can reach here without passing any check
            // (#156).
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

// Which elements of a List's universe it currently holds. O(universe + n), and
// built only on the paths that change the LENGTH -- the five length-preserving
// moves never need it.
//
// Every write is bounds-tested against the universe: `Variable.elements` is
// writable from Python with nothing in the way, so an out-of-universe id reaches
// here without having passed any check (#156). Such an id is simply not
// recorded, which reads as "absent" -- the list then looks shorter than it is to
// the insert guard below, and the worst that follows is a candidate move, which
// the batch scores and may reject like any other.
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
/// Built ONCE per `generate_standard_moves` call and shared by the two moves
/// that need it, rather than once each. It is O(universe) with two allocations,
/// which is the shape `SetPartition` already has on the Set path -- but it is
/// also the cost this representation exists to keep off a candidate, so it is
/// built only when one of those moves can actually apply. On a permutation List
/// neither can, so nothing here runs at all.
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
// Granular where a neighbour list says so: grow the sequence next to what it
// already holds rather than anywhere in the universe, the same rule
// `pick_added_element` applies to a Set. Falls back to the uniform draw when the
// list is empty or the list names nobody absent, so a partial neighbour list
// restricts where the search looks and never what it can reach.
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
// GUARDS BEFORE DRAWS, which is what keeps a permutation List's trajectory
// bit-identical: on `list_var(n)` the length is pinned at max_size, so this
// returns having consumed no random numbers at all.
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
// Without it a List declared `min_len == max_len < universe` -- a fixed-count
// orienteering or selection model -- is a dead end. Insert is refused at
// max_len, remove at min_len, and the five reordering moves are all
// permutations of what is already there, so the membership drawn at
// initialisation is the membership for the whole run and the search silently
// explores one of C(universe, k) equivalence classes. It is the List analogue of
// `set_swap`, and it is guarded before it draws, so a permutation List
// (universe == n) still consumes nothing.
//
// Not offered for a partition member: membership there is shared with the
// sibling lists, and `partition_swap` is the length-preserving exchange that
// keeps the cover.
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

// The five length-preserving intra-list moves. Every guard here is about the
// SEGMENT fitting strictly inside the list -- relocating the whole list is a
// no-op, not a move -- so they read the same on a variable-length List as they
// always did on a permutation: or-opt(2) needs n >= 3 and or-opt(3) needs n >= 4
// for exactly that reason, and n < 2 leaves nothing to reorder at all.
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
    // The three relocating moves are one `segment_edit` each, and the insert
    // positions below are the ones the whole-vector construction computed -- the
    // segment is erased first, so a target to the RIGHT of it shifts left by the
    // segment length, and the clamp is against the shortened vector.
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
// ORDER MATTERS AND THE TAIL MUST STAY LAST. All three tail moves test their
// guards before touching the RNG, so on a permutation List -- where the length is
// pinned at min_size == max_size == universe_size -- they draw nothing and the
// resulting draw sequence is the pre-#164 one, move for move. Putting one of
// them ahead of the five would not change that, but putting a DRAW ahead of a
// guard would.
static void list_moves(const Variable& var, RNG& rng, std::vector<Move>& moves,
                       const NeighbourList* neighbours) {
    list_reorder_moves(var, rng, moves, neighbours);
    const auto n = static_cast<int32_t>(var.elements.size());
    // The membership scan is built ONCE and only where one of the two moves that
    // read it can apply. A permutation List can apply neither -- its length is
    // pinned at max_size and its universe is its length -- so it neither scans
    // nor draws, which is what keeps its trajectory the pre-#164 one. A partition
    // member applies neither either: its membership belongs to the partition.
    if (!var.partitioned &&
        ((n < var.max_size && var.universe_size > 0) || (n > 0 && var.universe_size > n))) {
        const ListComplement comp = list_complement(var);
        list_insert_move(var, rng, neighbours, comp, moves);
        list_exchange_move(var, rng, neighbours, comp, moves);
    }
    list_remove_move(var, rng, moves);
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
    // Draw order is add-then-remove, as it has always been: both draws come off
    // the search's RNG, so swapping them would shift every later draw.
    const int32_t add_elem = pick_added_element(var, rng, part, neighbours);
    const int32_t rem_elem = pick_removed_element(rng, part);
    const auto it = std::find(var.elements.begin(), var.elements.end(), rem_elem);
    if (it == var.elements.end()) {
        return;
    }
    // Two edits, in this order: the removal happens where the element sits, and
    // the addition lands at the end of the SHORTENED vector. That is the element
    // order the whole-vector construction produced, and a Set read pairwise
    // (`pair_lambda_sum`) can tell the difference.
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
// Every one of them preserves the cover BY CONSTRUCTION, which is the point:
// "each element served exactly once" as a penalty row is a poor landscape for a
// jump-based search, because every repair has to pass through a doubly-served or
// unserved state. What these moves do NOT do is REPAIR a cover that something
// else broke -- `Variable.elements` is writable from Python with no check in the
// way (#156) -- and they do not need to. Nothing below indexes an array BY an
// element id except the membership stamp, which bounds-tests every write, so a
// state Python corrupted is a wrong search rather than a crash.

namespace {

/// The position in `dest` of the nearest neighbour of `e` that `dest` holds, or
/// -1 when there is no neighbour list or it names nobody `dest` holds. Draws no
/// random numbers: the two callers below decide for themselves what a miss means.
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
/// `dest` already holds, else uniform over the `|dest| + 1` gaps. The uniform
/// draw is the only one taken when no neighbour list was supplied, which is the
/// default everywhere.
int32_t partition_insert_pos(const std::vector<int32_t>& dest, int32_t e, RNG& rng,
                             const NeighbourList* neighbours) {
    const int32_t near = neighbour_position(dest, e, neighbours);
    if (near >= 0) {
        return near + 1;
    }
    return static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(dest.size()) + 1));
}

/// Append `move` unless every change leaves its variable exactly as it is. A
/// no-op candidate is not wrong -- the batch scores it at delta 0 and rejects it
/// -- but it costs two `delta_evaluate` passes to learn that, and the
/// diversification kick would count it as a move it had made.
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
    // Granular where a neighbour list says so: exchange against the element of
    // `b` nearest `a[i]`, since swapping it against one on the far side of the
    // map can only be improving by accident. A miss -- no list, or no neighbour
    // of `a[i]` in `b` -- falls back to the uniform draw, so a partial list
    // restricts where the search looks and never what it can reach.
    const int32_t near = neighbour_position(vb.elements, va.elements[i], neighbours);
    const auto j =
        (near >= 0)
            ? static_cast<size_t>(near)
            : static_cast<size_t>(rng.integers(0, static_cast<int64_t>(vb.elements.size())));
    Move m;
    m.move_type = "partition_swap";
    // One position assigned on each side: an exchange ACROSS two variables is
    // two independent writes, where the intra-list swap is one.
    m.changes.push_back(edit_change(a, assign_edit(static_cast<int32_t>(i), vb.elements[j])));
    m.changes.push_back(edit_change(b, assign_edit(static_cast<int32_t>(j), va.elements[i])));
    push_if_changed(model, std::move(m), out);
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
    std::vector<int32_t> new_a(va.elements.begin(),
                               va.elements.begin() + static_cast<std::ptrdiff_t>(p));
    new_a.insert(new_a.end(), vb.elements.begin() + static_cast<std::ptrdiff_t>(q),
                 vb.elements.end());
    std::vector<int32_t> new_b(vb.elements.begin(),
                               vb.elements.begin() + static_cast<std::ptrdiff_t>(q));
    new_b.insert(new_b.end(), va.elements.begin() + static_cast<std::ptrdiff_t>(p),
                 va.elements.end());
    m.changes.push_back(replace_change(a, std::move(new_a)));
    m.changes.push_back(replace_change(b, std::move(new_b)));
    push_if_changed(model, std::move(m), out);
}

/// The elements of the partition's universe that no member list holds.
///
/// RECOMPUTED FROM THE LISTS EVERY TIME, never cached: `Variable.elements` is
/// writable from Python, and the assignment also moves under LNS, the
/// diversification kick and a restart from the solution pool, none of which
/// notifies a generator (see `MoveGenerator::on_commit`). O(sum |lists| +
/// universe), and every stamp write is bounds-tested, so an out-of-universe id
/// written from Python is ignored rather than indexing the heap (#156).
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

/// Written into `out`; the return is how many. A fixed array rather than a
/// vector because this runs once per candidate PROPOSED, and the argument for
/// the whole positional representation is that a candidate must not cost a heap
/// allocation.
///
/// Honesty about the residual: four of the five names this path assigns to
/// `Move::move_type` ("partition_relocate", "partition_2opt_star",
/// "partition_insert", "partition_remove") exceed libstdc++'s 15-character
/// small-string buffer, so each DOES cost one malloc. Every pre-existing move
/// name fits, which is why it never showed before. It is one allocation per
/// candidate proposed, against an O(universe) membership scan on the same path,
/// and the partition generator proposes at most one candidate per `generate` --
/// so the array-versus-vector point stands on its own and this is not worth
/// trading the readable identifiers for. Making `move_type` a `string_view`
/// is the real answer if it ever shows up in a profile.
size_t applicable_kinds(const ListPartition& part, std::array<PartitionMoveKind, 5>& out) {
    size_t n = 0;
    if (part.list_ids.size() >= 2) {
        out[n++] = PartitionMoveKind::Relocate;
        out[n++] = PartitionMoveKind::Swap;
        out[n++] = PartitionMoveKind::TwoOptStar;
    }
    if (part.cover == Cover::AtMostOnce) {
        // Under Exact these are the two halves of a relocate and never stand
        // alone: an insert would double-serve, a remove would leave an element
        // unserved, and no later move could repair either.
        out[n++] = PartitionMoveKind::Insert;
        out[n++] = PartitionMoveKind::Remove;
    }
    return n;
}

/// A member of `part` other than `a`, uniformly, by the shift-past trick the
/// intra-list move pair already uses. PRECONDITION: at least two members, and
/// `a` a member. A non-member `a` makes the shift inert and the draw uniform
/// over all of them, which is DEFINED but not supported: the resulting move
/// would take an element out of a list the partition does not own and put it in
/// one it does, double-serving it. `generate_partition_moves` refuses a
/// non-member anchor before reaching here rather than leaving that to the
/// caller.
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

    // The first list is the anchor when the caller named one -- the
    // diversification kick does, because it asks "move THIS variable" and reads
    // the answer off that variable alone. The second is drawn only by the three
    // kinds that need one, so an insert or a removal costs the draws it actually
    // uses rather than one more.
    const int32_t a = (anchor >= 0) ? anchor
                                    : part.list_ids[static_cast<size_t>(rng.integers(
                                          0, static_cast<int64_t>(part.list_ids.size())))];
    // A caller-supplied anchor must belong to this partition. Both live callers
    // honour it -- the kick derives the partition from the same variable, and
    // ListPartitionGenerator passes -1 -- but this is a public entry point in
    // moves.h, and every move below preserves the cover only over a PAIR of
    // member lists. One find over the member count, which is the route count.
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
            // In place, on the vector that is already there -- no allocation
            // unless the edit lengthens it past its capacity. The edit is
            // relative to the assignment the move was built against, so the
            // caller is responsible for applying it to THAT assignment; see
            // `ElementEdit` and, for the several-candidates-per-sample case,
            // `StructuralBatch`.
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
