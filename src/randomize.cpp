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
        // Both bounds finite, but the width overflows. Narrowing only moves a bound inward,
        // so the result stays a subset. Applies to every scalar type, Int included.
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
    // Inward-only, so a subset of the window and domain; empty when the whole window sits past
    // 2^53. The bounds are NOT clamped independently into the range: that made trimming inside
    // `domain_window` unsound.
    //
    // Rounded inward because `static_cast<int64_t>` truncates toward zero (on `[0.9, 1.2]` it
    // named 0, outside the domain). Both readers round Int bounds inward and `int_var` takes
    // `int`, so no buildable model has a fractional bound: this moves no existing draw sequence.
    return {std::ceil(std::max(w.lo, -kExactIntMagnitude)),
            std::floor(std::min(w.hi, kExactIntMagnitude - 1.0))};
}

bool movable_domain(const Variable& var) {
    switch (var.type) {
        case VarType::Int: {
            // Exact and UB-free at any magnitude, where an int64_t cast is neither: the window
            // holds two distinct integers iff floor(hi) - ceil(lo) >= 1.
            const DomainWindow w = int_sample_window(var);
            return std::floor(w.hi) - std::ceil(w.lo) >= 1.0;
        }
        default: {  // Bool, Float
            const DomainWindow w = domain_window(var);
            return w.hi > w.lo;
        }
    }
}

double random_in_domain(const Variable& var, RNG& rng) {
    const DomainWindow w = domain_window(var);
    switch (var.type) {
        case VarType::Bool:
            return static_cast<double>(rng.integers(0, 2));
        case VarType::Int: {
            const DomainWindow s = int_sample_window(var);
            if (s.lo > s.hi) {
                // Domain wholly past 2^53: draw over the untrimmed window, in-domain by
                // construction and integral, since every double that large is one.
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
            // `permutation(n)` is iota-then-shuffle, so on a fresh List (elements == iota) the
            // arms are bit-identical; once moved, Regenerate discards the order and Perturb
            // keeps the elements. Perturb preserves length and membership on any List, which
            // makes LNS destroy safe on a partition member (see ListOrder).
            if (order == ListOrder::Perturb) {
                rng.shuffle(var.elements);
            } else if (var.list_init == ListInit::Identity) {
                // universe == min == max (list_var enforces it): exactly one
                // `rng.permutation(max_size)`, the pre-#164 draw. A distinct arm, not a case of
                // Random below, so every permutation trajectory stays bit-identical.
                var.elements = rng.permutation(var.max_size);
            } else if (var.list_init == ListInit::Empty) {
                var.elements.clear();  // no draw
            } else {
                // Random: a uniform admissible length, then that many distinct elements.
                // `choice` shuffles the whole universe and truncates, so the ORDER is random
                // too, which a List needs.
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

// Two passes over one shuffled universe: the first gives each list its minimum length (what
// makes the result feasible); the second offers each remaining element to a uniformly chosen
// list still under its TARGET.
//  - `Exact` targets every `max_len`, placing every element: no `Exact` move can repair an
//    incomplete cover, and the validated `sum(max_len) >= universe` keeps some list open.
//  - `AtMostOnce` targets each member's `ListInit`: `Empty` stops at the minimum (0, per
//    `list_var`), `Random` draws a uniform admissible length. Routes declared `Empty` start
//    unassigned, which the search can improve; fully assigned under a capacity bound it may
//    not be able to leave.
// Uniform only when no target binds (a full list forces the rest into open ones); it is a
// starting point, not a sample. List order is left as drawn: inter-list moves reorder anyway.
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
    // Full lists are compacted out, so the draw is uniform over lists with room; otherwise one
    // large and many tiny lists would spend most draws on lists with no room.
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
