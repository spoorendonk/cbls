#pragma once

#include "dag.h"
#include "rng.h"

// Uniform randomisation of a single variable over its own domain.
//
// One implementation, three callers: `initialize_random` /
// `initialize_structured_random` (search.cpp), LNS's destroy step (lns.cpp) and
// FeasibilityJump's diversification kick (feasibility_jump.cpp). Those three
// each carried a private copy of the same Bool/Int/Float/List/Set switch, which
// meant a guard added to one of them fixed only that path (#112).
//
// The copies were not quite identical, and the difference is preserved rather
// than flattened: LNS shuffled a List's *current* `elements`, while the
// initialisers regenerated the order from scratch. See `ListOrder`.
//
// Every scalar entry point here is *total*: it returns a finite value inside the
// variable's domain for any domain the model can hold, including the unbounded
// ones ((-inf, +inf), [0, +inf), an Int with an infinite bound). That is the
// whole point of concentrating them — see `domain_window`.
//
// Note this makes the engine's own *randomisation* incapable of injecting a
// non-finite value. It does not make the engine safe on unbounded domains
// generally: a Float jump candidate is still drawn from `var.lb`/`var.ub`
// directly, so an unbounded model can still reach an infinite assignment (and
// report it as feasible) by a route that has nothing to do with this file.
//
// No non-finite *detector* comes with it, deliberately. Three guards already
// absorb a non-finite constraint value safely (ViolationManager's
// clamped_node_violation, solve()'s max_real_violation, LNS's state_key), and
// with the draws below unable to inject one there is no path from the engine's
// own randomisation into a NaN assignment left to detect.

namespace cbls {

/// Magnitude that stands in for an infinite Bool/Float bound when sampling.
///
/// Matches `NlToModelOptions::inf_clamp` (and `MpsToModelOptions::inf_clamp`),
/// which clamp infinite *variable bounds* to the same magnitude at load time.
/// A model read from `.nl`/`.mps` therefore never reaches the substitution
/// below, and a model built through the C++/Python API that does reach it lands
/// in exactly the same box the readers would have given it.
inline constexpr double kRandomInfClamp = 1.0e9;

/// Magnitude that stands in for an infinite Int bound when sampling.
///
/// Matches `NlToModelOptions::int_inf_clamp` and, verbatim from there: a ±1e9
/// integer domain is not a searchable space; unbounded integers in practice mean
/// "small non-negative count", so a far tighter box is the useful default.
inline constexpr double kRandomIntInfClamp = 1.0e6;

/// The finite `[lo, hi]` window that a uniform draw over a scalar variable's
/// domain actually samples from. Always a subset of the variable's own domain,
/// so a value drawn from it is in-domain by construction.
struct DomainWindow {
    double lo = 0.0;
    double hi = 0.0;
};

/// `var`'s bounds, made safe to sample and to cast.
///
/// A finite declared bound is returned untouched — the guard is inert on the
/// models that do not need it, so their RNG draws (and therefore their solve
/// trajectories) are bit-for-bit what they were. Only these cases are rewritten:
///
///  - an infinite bound becomes the clamp magnitude above (`kRandomIntInfClamp`
///    for Int, else `kRandomInfClamp`). Left alone,
///    `uniform_real_distribution(lb, ub)` violates its own precondition
///    (`ub - lb <= DBL_MAX`) and libstdc++'s `lb + (ub - lb) * u` yields NaN on
///    `(-inf, +inf)` and +inf on `[0, +inf)`, while an Int bound casts to
///    `INT64_MIN`;
///  - on a half-infinite domain the substituted end is pushed past the declared
///    one where needed, so the window stays *inside* the domain even when the
///    declared bound lies beyond the clamp magnitude (`(-inf, -2e9]` must not
///    sample -1e9);
///  - two finite bounds whose *width* overflows to +inf trip the same
///    distribution precondition, so they are narrowed to the clamp box;
///  - an Int window is kept within the integers a double can name exactly, which
///    is what makes the `int64_t` casts at the call sites defined.
///
/// Scalar types only (Bool/Int/Float). List/Set carry no `value`.
DomainWindow domain_window(const Variable& var);

/// Draw a uniformly random value from a scalar variable's domain. Finite and
/// in-domain for every domain — see `domain_window`.
///
/// Int and Float sample the window; Bool draws from {0, 1} without consulting it,
/// which is in-domain for every Bool the model can build (`bool_var` fixes the
/// bounds at [0, 1], and a fixed binary read from MPS becomes an Int). A Bool
/// pinned by `lb == ub` would not be respected here — unreachable today, but the
/// reason `movable_domain` in feasibility_jump.cpp checks the bounds anyway.
double random_in_domain(const Variable& var, RNG& rng);

/// How a List's new order relates to its current one.
///
/// Both draw a uniformly random permutation and consume identical RNG draws, so
/// this is not a distributional choice — it decides whether the incumbent order
/// survives, and the two call sites genuinely want different answers.
enum class ListOrder {
    /// Discard the current order and lay out a fresh permutation of the whole
    /// universe. What the initialisers want: there is no incumbent to respect,
    /// and the result is well-formed even if `elements` was not.
    Regenerate,
    /// Shuffle the current `elements` in place, preserving exactly which
    /// elements are present. What LNS destroy/repair wants: it perturbs an
    /// incumbent solution, so regenerating instead would silently change the
    /// repair trajectory on every List model (it did — that is why this
    /// parameter exists).
    Perturb,
};

/// Redraw a structured (List/Set) variable's `elements`: a uniformly random
/// permutation for a List, a uniformly random subset of an admissible size for a
/// Set. The structured counterpart of `random_in_domain`, and the hook for
/// anything that wants to randomise structure without going through
/// `randomize_var`'s type dispatch.
///
/// No-op on a scalar variable.
void randomize_structured_var(Variable& var, RNG& rng, ListOrder order = ListOrder::Regenerate);

/// Randomise one variable in place, whatever its type: `value` for a scalar,
/// `elements` for a List/Set. The single switch the three call sites share.
void randomize_var(Variable& var, RNG& rng, ListOrder order = ListOrder::Regenerate);

}  // namespace cbls
