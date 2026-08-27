#include "cbls/randomize.h"

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
            if (order == ListOrder::Perturb) {
                rng.shuffle(var.elements);
            } else {
                var.elements = rng.permutation(var.max_size);
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

void randomize_var(Variable& var, RNG& rng, ListOrder order) {
    if (is_structured(var.type)) {
        randomize_structured_var(var, rng, order);
    } else {
        var.value = random_in_domain(var, rng);
    }
}

}  // namespace cbls
