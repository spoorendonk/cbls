#include "cbls/randomize.h"

#include "cbls/model.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace cbls {

DomainWindow domain_window(const Variable& var) {
    const double clamp = (var.type == VarType::Int) ? kRandomIntInfClamp : kRandomInfClamp;
    const bool lo_open = !std::isfinite(var.lb);
    const bool hi_open = !std::isfinite(var.ub);

    double lo = lo_open ? -clamp : var.lb;
    double hi = hi_open ? clamp : var.ub;

    if (lo_open && !hi_open) {
        // Anchor a clamp-wide window at the declared bound rather than trusting
        // ±clamp to be on the right side of it.
        lo = std::min(lo, hi - clamp);
    } else if (hi_open && !lo_open) {
        hi = std::max(hi, lo + clamp);
    }

    if (!std::isfinite(hi - lo)) {
        // Both bounds declared and finite, but the width overflows. Narrowing
        // only ever moves a bound inward, so the result stays a subset. Applies
        // to every scalar type: it used to be the `else` of an Int-only trim, so
        // Int skipped it.
        lo = std::max(lo, -clamp);
        hi = std::min(hi, clamp);
    }

    if (lo > hi) {
        std::swap(lo, hi);  // defensive: degenerate bound ordering, as in nl_to_model
    }
    return {lo, hi};
}

DomainWindow int_sample_window(const Variable& var) {
    const DomainWindow w = domain_window(var);
    if (var.type != VarType::Int) {
        return w;
    }
    // Inward-only, so still a subset of the window (and of the domain). Empty
    // when the whole window sits past 2^53; the bounds are NOT clamped
    // independently into the range, which is what made trimming inside
    // `domain_window` unsound.
    //
    // Rounding inward as well, because `static_cast<int64_t>` truncates toward
    // zero: on `[0.9, 1.2]` that named 0, and the draw then left the domain.
    // Both readers already round an Int column's bounds inward (`std::ceil` /
    // `std::floor` in nl_to_model.cpp and mps_to_model.cpp) and `int_var` takes
    // `int`, so no model the codebase can build reaches this with a fractional
    // bound — it cannot move an existing draw sequence.
    return {std::ceil(std::max(w.lo, -kExactIntMagnitude)),
            std::floor(std::min(w.hi, kExactIntMagnitude - 1.0))};
}

double random_in_domain(const Variable& var, RNG& rng) {
    const DomainWindow w = domain_window(var);
    switch (var.type) {
        case VarType::Bool:
            return static_cast<double>(rng.integers(0, 2));
        case VarType::Int: {
            const DomainWindow s = int_sample_window(var);
            if (s.lo > s.hi) {
                // Domain wholly past 2^53: no int64_t range to draw from. Draw
                // over the untrimmed window instead — in-domain by construction,
                // and integral for free, since every double that large already
                // is one.
                const double v = std::round(rng.uniform(w.lo, w.hi));
                return std::min(std::max(v, w.lo), w.hi);
            }
            return static_cast<double>(
                rng.integers(static_cast<int64_t>(s.lo), static_cast<int64_t>(s.hi) + 1));
        }
        default:  // Float (List/Set have no scalar value — see randomize_var)
            return rng.uniform(w.lo, w.hi);
    }
}

void randomize_structured_var(Variable& var, RNG& rng, ListOrder order) {
    switch (var.type) {
        case VarType::List:
            // `permutation(n)` is iota-then-shuffle, so on a freshly built List
            // (elements == iota) the two arms are bit-identical. They diverge
            // once the list has been moved: Regenerate discards that order,
            // Perturb keeps the same elements in a new arrangement. LNS needs
            // the latter — see ListOrder.
            //
            // Perturb is length-preserving whatever the List is, which is what
            // makes LNS destroy safe on a partition member: it rearranges the
            // elements that list already holds and can neither gain nor lose one.
            if (order == ListOrder::Perturb) {
                rng.shuffle(var.elements);
            } else if (var.list_init == ListInit::Identity) {
                // universe == min == max here (list_var enforces it), so this is
                // the pre-#164 draw on the pre-#164 variable, verbatim: one
                // `rng.permutation(max_size)` and nothing else. Keeping it a
                // distinct arm rather than a special case of the Random one below
                // is what makes every permutation trajectory bit-identical.
                var.elements = rng.permutation(var.max_size);
            } else if (var.list_init == ListInit::Empty) {
                var.elements.clear();  // no draw
            } else {
                // Random: a uniformly random admissible length, then that many
                // distinct elements of the universe in a uniformly random order.
                // `choice` shuffles the whole universe and truncates, so the
                // ORDER is random too -- which a List needs and a Set does not
                // care about.
                const int size = static_cast<int>(rng.integers(var.min_size, var.max_size + 1));
                var.elements = rng.choice(var.universe_size, size);
            }
            break;
        case VarType::Set: {
            const int size = static_cast<int>(rng.integers(var.min_size, var.max_size + 1));
            var.elements = rng.choice(var.universe_size, size);
            break;
        }
        default:  // Bool, Int, Float carry no elements
            break;
    }
}

// Lay `part` out as a uniformly random assignment that satisfies its cover:
// every list within its own [min_len, max_len], every element in at most one
// list, and -- for `Cover::Exact`, which `add_list_partition` has already
// checked is achievable -- every element in exactly one.
//
// Two passes over one shuffled universe. The first hands each list its minimum
// length, which is what makes the result feasible at all; the second offers each
// remaining element to a uniformly chosen list that is still under its TARGET.
//
// The target is where `ListInit` re-enters, and where the two covers differ:
//
//  - `Exact` ignores it and targets every list's `max_len`, so the second pass
//    places every element. It has to: the cover must hold at the first
//    assignment, because no `Exact` move can repair an incomplete one, and the
//    validated `sum(max_len) >= universe` is what guarantees some list always
//    has room.
//  - `AtMostOnce` targets what each member's `ListInit` asks for -- `Empty`
//    stops at the minimum (which `list_var` requires to be 0), `Random` draws a
//    uniform admissible length. So a prize-collecting model whose routes are
//    declared `Empty` starts with everything unassigned and its skip penalty at
//    its worst, which is a starting point the search can improve, rather than
//    fully assigned, which under a capacity bound it may not be able to leave.
//
// NOT uniform over the feasible assignments in general: once a list reaches its
// target the remaining elements are forced into the ones still open, which skews
// the size distribution. It is uniform when no target binds. This is a starting
// point rather than a sample, so that is enough.
//
// The lists are otherwise left in the order the draw produced: an inter-list
// move reorders them anyway, and imposing one here would only look tidier.
void randomize_list_partition(Model& model, const ListPartition& part, RNG& rng) {
    for (int32_t vid : part.list_ids) {
        model.var_mut(vid).elements.clear();
    }
    if (part.list_ids.empty() || part.universe_size <= 0) {
        return;
    }
    std::vector<int32_t> pool = rng.permutation(part.universe_size);
    size_t next = 0;
    for (int32_t vid : part.list_ids) {
        Variable& v = model.var_mut(vid);
        const auto want = static_cast<size_t>(v.min_size);
        v.elements.reserve(want);
        for (size_t k = 0; k < want && next < pool.size(); ++k, ++next) {
            v.elements.push_back(pool[next]);
        }
    }
    // Candidates are compacted in place as they fill up, so the draw below is
    // uniform over the lists that can still take an element rather than over all
    // of them -- otherwise a partition of one large and many tiny lists would
    // spend most of its draws on lists with no room.
    std::vector<int32_t> open_lists;
    std::vector<int32_t> targets;
    open_lists.reserve(part.list_ids.size());
    targets.reserve(part.list_ids.size());
    for (int32_t vid : part.list_ids) {
        const Variable& v = model.var(vid);
        int32_t target = v.max_size;
        if (part.cover == Cover::AtMostOnce) {
            target = (v.list_init == ListInit::Empty)
                         ? v.min_size
                         : static_cast<int32_t>(rng.integers(v.min_size, v.max_size + 1));
        }
        if (static_cast<int32_t>(v.elements.size()) < target) {
            open_lists.push_back(vid);
            targets.push_back(target);
        }
    }
    for (; next < pool.size() && !open_lists.empty(); ++next) {
        const auto pick =
            static_cast<size_t>(rng.integers(0, static_cast<int64_t>(open_lists.size())));
        Variable& v = model.var_mut(open_lists[pick]);
        v.elements.push_back(pool[next]);
        if (static_cast<int32_t>(v.elements.size()) >= targets[pick]) {
            open_lists[pick] = open_lists.back();
            open_lists.pop_back();
            targets[pick] = targets.back();
            targets.pop_back();
        }
    }
}

void randomize_var(Variable& var, RNG& rng, ListOrder order) {
    if (is_structured(var.type)) {
        randomize_structured_var(var, rng, order);
    } else {
        var.value = random_in_domain(var, rng);
    }
}

}  // namespace cbls
