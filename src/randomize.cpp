#include "cbls/randomize.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace cbls {

namespace {

// Largest magnitude at which a double still names every integer below it. An Int
// window is trimmed to it so `static_cast<int64_t>` at the call sites — and the
// `+ 1` that turns an inclusive upper bound into RNG::integers' exclusive one —
// stay defined for any finite bound. Nothing is lost: past 2^53 a double cannot
// represent consecutive integers anyway, so such an "integer domain" is already
// not one.
constexpr double kExactIntMagnitude = 9007199254740992.0;  // 2^53

}  // namespace

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

    if (var.type == VarType::Int) {
        lo = std::min(std::max(lo, -kExactIntMagnitude), kExactIntMagnitude - 1.0);
        hi = std::min(std::max(hi, -kExactIntMagnitude), kExactIntMagnitude - 1.0);
    } else if (!std::isfinite(hi - lo)) {
        // Both bounds declared and finite, but the width overflows.
        lo = std::max(lo, -clamp);
        hi = std::min(hi, clamp);
    }

    if (lo > hi) {
        std::swap(lo, hi);  // defensive: degenerate bound ordering, as in nl_to_model
    }
    return {lo, hi};
}

double random_in_domain(const Variable& var, RNG& rng) {
    const DomainWindow w = domain_window(var);
    switch (var.type) {
        case VarType::Bool:
            return static_cast<double>(rng.integers(0, 2));
        case VarType::Int:
            return static_cast<double>(
                rng.integers(static_cast<int64_t>(w.lo), static_cast<int64_t>(w.hi) + 1));
        default:  // Float (List/Set have no scalar value — see randomize_var)
            return rng.uniform(w.lo, w.hi);
    }
}

void randomize_structured_var(Variable& var, RNG& rng) {
    switch (var.type) {
        case VarType::List:
            // A fresh permutation of the whole universe rather than a shuffle of
            // the current `elements`: the result is then a well-formed List even
            // if `elements` was not, and it does not depend on what was there.
            var.elements = rng.permutation(var.max_size);
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

void randomize_var(Variable& var, RNG& rng) {
    if (is_structured(var.type)) {
        randomize_structured_var(var, rng);
    } else {
        var.value = random_in_domain(var, rng);
    }
}

}  // namespace cbls
