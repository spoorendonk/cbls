#include "cbls/moves.h"

#include "cbls/move_generator.h"
#include "cbls/randomize.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <utility>

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

// Insert one absent element at a random position (#164).
//
// GUARDS BEFORE DRAWS, which is what keeps a permutation List's trajectory
// bit-identical: on `list_var(n)` the length is pinned at max_size, so this
// returns having consumed no random numbers at all.
static void list_insert_move(const Variable& var, RNG& rng, const NeighbourList* neighbours,
                             std::vector<Move>& moves) {
    const auto n = static_cast<int32_t>(var.elements.size());
    if (var.partitioned || n >= var.max_size || var.universe_size <= 0) {
        return;
    }
    const std::vector<bool> present = list_membership(var);
    // Granular where a neighbour list says so: grow the sequence next to what it
    // already holds rather than anywhere in the universe, the same rule
    // `pick_added_element` applies to a Set. Falls back to the uniform draw when
    // the list is empty or names nobody absent.
    int32_t chosen = -1;
    if (neighbours != nullptr && !neighbours->empty() && n > 0) {
        const int32_t seed = var.elements[static_cast<size_t>(rng.integers(0, n))];
        for (int32_t f : neighbours->of(seed)) {
            if (f >= 0 && static_cast<size_t>(f) < present.size() &&
                !present[static_cast<size_t>(f)]) {
                chosen = f;
                break;
            }
        }
    }
    if (chosen < 0) {
        std::vector<int32_t> absent;
        for (int32_t e = 0; e < var.universe_size; ++e) {
            if (!present[static_cast<size_t>(e)]) {
                absent.push_back(e);
            }
        }
        if (absent.empty()) {
            return;
        }
        chosen = absent[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(absent.size())))];
    }
    const auto pos = static_cast<int64_t>(rng.integers(0, n + 1));
    Move m;
    m.move_type = "list_insert";
    auto new_elems = var.elements;
    new_elems.insert(new_elems.begin() + static_cast<std::ptrdiff_t>(pos), chosen);
    m.changes.push_back({var.id, 0.0, new_elems});
    moves.push_back(m);
}

// Drop the element at a random position (#164). Guards before draws, as above:
// a permutation List sits at min_size and returns without drawing.
static void list_remove_move(const Variable& var, RNG& rng, std::vector<Move>& moves) {
    const auto n = static_cast<int32_t>(var.elements.size());
    if (var.partitioned || n <= var.min_size || n <= 0) {
        return;
    }
    const auto pos = static_cast<std::ptrdiff_t>(rng.integers(0, n));
    Move m;
    m.move_type = "list_remove";
    auto new_elems = var.elements;
    new_elems.erase(new_elems.begin() + pos);
    m.changes.push_back({var.id, 0.0, new_elems});
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

// A List's typed moves: the five length-preserving ones first, then the two that
// change the length (#164).
//
// ORDER MATTERS AND THE TAIL MUST STAY LAST. Both tail moves test their guard
// before touching the RNG, so on a permutation List -- where the length is
// pinned at min_size == max_size -- they draw nothing and the resulting draw
// sequence is the pre-#164 one, move for move. Putting either ahead of the five
// would not change that, but putting a DRAW ahead of a guard would.
static void list_moves(const Variable& var, RNG& rng, std::vector<Move>& moves,
                       const NeighbourList* neighbours) {
    list_reorder_moves(var, rng, moves, neighbours);
    list_insert_move(var, rng, neighbours, moves);
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

/// Insert position for `e` in `dest`: just after a nearest neighbour of `e` that
/// `dest` already holds, else uniform over the `|dest| + 1` gaps. The uniform
/// draw is the only one taken when no neighbour list was supplied, which is the
/// default everywhere.
int32_t partition_insert_pos(const std::vector<int32_t>& dest, int32_t e, RNG& rng,
                             const NeighbourList* neighbours) {
    if (neighbours != nullptr && !neighbours->empty()) {
        for (int32_t f : neighbours->of(e)) {
            const auto it = std::find(dest.begin(), dest.end(), f);
            if (it != dest.end()) {
                return static_cast<int32_t>(it - dest.begin()) + 1;
            }
        }
    }
    return static_cast<int32_t>(rng.integers(0, static_cast<int64_t>(dest.size()) + 1));
}

/// Append `move` unless every change leaves its variable exactly as it is. A
/// no-op candidate is not wrong -- the batch scores it at delta 0 and rejects it
/// -- but it costs two `delta_evaluate` passes to learn that, and the
/// diversification kick would count it as a move it had made.
void push_if_changed(const Model& model, Move&& move, std::vector<Move>& out) {
    for (const Move::Change& change : move.changes) {
        if (model.var(change.var_id).elements != change.new_elements) {
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
    const auto i = static_cast<size_t>(rng.integers(0, static_cast<int64_t>(va.elements.size())));
    const int32_t e = va.elements[i];
    const int32_t pos = partition_insert_pos(vb.elements, e, rng, neighbours);
    Move m;
    m.move_type = "partition_relocate";
    std::vector<int32_t> new_a = va.elements;
    new_a.erase(new_a.begin() + static_cast<std::ptrdiff_t>(i));
    std::vector<int32_t> new_b = vb.elements;
    new_b.insert(new_b.begin() + pos, e);
    m.changes.push_back({a, 0.0, std::move(new_a)});
    m.changes.push_back({b, 0.0, std::move(new_b)});
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
    // Granular where a neighbour list says so: exchange against an element of
    // `b` that lies near `a[i]`, since swapping it against one on the far side of
    // the map can only be improving by accident. `partition_insert_pos` returns
    // the gap AFTER the neighbour, so step back onto the neighbour itself, and
    // fall back to the uniform draw when it returned the last gap (no neighbour
    // of `a[i]` is in `b`, or there is no list at all).
    const int32_t gap = partition_insert_pos(vb.elements, va.elements[i], rng, neighbours);
    const auto j =
        (gap >= 1 && static_cast<size_t>(gap) <= vb.elements.size())
            ? static_cast<size_t>(gap - 1)
            : static_cast<size_t>(rng.integers(0, static_cast<int64_t>(vb.elements.size())));
    Move m;
    m.move_type = "partition_swap";
    std::vector<int32_t> new_a = va.elements;
    std::vector<int32_t> new_b = vb.elements;
    std::swap(new_a[i], new_b[j]);
    m.changes.push_back({a, 0.0, std::move(new_a)});
    m.changes.push_back({b, 0.0, std::move(new_b)});
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
    m.changes.push_back({a, 0.0, std::move(new_a)});
    m.changes.push_back({b, 0.0, std::move(new_b)});
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
    std::vector<int32_t> new_a = va.elements;
    new_a.insert(new_a.begin() + pos, e);
    m.changes.push_back({a, 0.0, std::move(new_a)});
    push_if_changed(model, std::move(m), out);
}

/// Drop one element of `a`, leaving it unassigned. `Cover::AtMostOnce` only.
void partition_remove(const Model& model, int32_t a, RNG& rng, std::vector<Move>& out) {
    const Variable& va = model.var(a);
    if (va.elements.empty() || static_cast<int32_t>(va.elements.size()) <= va.min_size) {
        return;
    }
    const auto i =
        static_cast<std::ptrdiff_t>(rng.integers(0, static_cast<int64_t>(va.elements.size())));
    Move m;
    m.move_type = "partition_remove";
    std::vector<int32_t> new_a = va.elements;
    new_a.erase(new_a.begin() + i);
    m.changes.push_back({a, 0.0, std::move(new_a)});
    push_if_changed(model, std::move(m), out);
}

/// The move kinds a partition admits, given how many lists it has and what its
/// cover permits. Drawn from uniformly, so this list is also the mix.
enum class PartitionMoveKind : std::uint8_t { Relocate, Swap, TwoOptStar, Insert, Remove };

std::vector<PartitionMoveKind> applicable_kinds(const ListPartition& part) {
    std::vector<PartitionMoveKind> kinds;
    if (part.list_ids.size() >= 2) {
        kinds.push_back(PartitionMoveKind::Relocate);
        kinds.push_back(PartitionMoveKind::Swap);
        kinds.push_back(PartitionMoveKind::TwoOptStar);
    }
    if (part.cover == Cover::AtMostOnce) {
        // Under Exact these are the two halves of a relocate and never stand
        // alone: an insert would double-serve, a remove would leave an element
        // unserved, and no later move could repair either.
        kinds.push_back(PartitionMoveKind::Insert);
        kinds.push_back(PartitionMoveKind::Remove);
    }
    return kinds;
}

}  // namespace

void generate_partition_moves(const Model& model, int partition, int32_t anchor, RNG& rng,
                              std::vector<Move>& out, const NeighbourList* neighbours) {
    const std::vector<ListPartition>& partitions = model.list_partitions();
    if (partition < 0 || partition >= static_cast<int>(partitions.size())) {
        return;
    }
    const ListPartition& part = partitions[static_cast<size_t>(partition)];
    const std::vector<PartitionMoveKind> kinds = applicable_kinds(part);
    if (kinds.empty() || part.list_ids.empty()) {
        return;
    }
    const PartitionMoveKind kind =
        kinds[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(kinds.size())))];

    // The first list is the anchor when the caller named one -- the
    // diversification kick does, because it asks "move THIS variable" and reads
    // the answer off that variable alone. Otherwise both are drawn.
    const auto count = static_cast<int64_t>(part.list_ids.size());
    int32_t a = anchor;
    if (a < 0) {
        a = part.list_ids[static_cast<size_t>(rng.integers(0, count))];
    }
    int32_t b = -1;
    if (count >= 2) {
        // Uniform over the members other than `a`, by the shift-past trick the
        // intra-list move pair already uses. `a` may be an anchor from outside
        // the id list, in which case the shift is inert and the draw is uniform
        // over all members.
        const auto it = std::find(part.list_ids.begin(), part.list_ids.end(), a);
        const auto skip =
            (it == part.list_ids.end()) ? count : static_cast<int64_t>(it - part.list_ids.begin());
        int64_t pick = rng.integers(0, count - (skip < count ? 1 : 0));
        if (skip < count && pick >= skip) {
            ++pick;
        }
        b = part.list_ids[static_cast<size_t>(pick)];
    }

    switch (kind) {
        case PartitionMoveKind::Relocate:
            partition_relocate(model, a, b, rng, neighbours, out);
            return;
        case PartitionMoveKind::Swap:
            partition_swap(model, a, b, rng, neighbours, out);
            return;
        case PartitionMoveKind::TwoOptStar:
            partition_two_opt_star(model, a, b, rng, out);
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
