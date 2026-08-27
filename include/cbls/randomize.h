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
// Only the kick's SCALAR half comes through here. Its List/Set half goes to the
// typed move generators instead (#111), because a kick wants k bounded local
// moves on an incumbent, and every option below is a full resample — a restart
// of the variable, which is what the kick is trying not to be.
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

/// Largest magnitude at which a double still names every integer below it.
///
/// Two uses, both about what a `double` can *say* rather than about sampling:
/// `int_sample_window` trims to it so `static_cast<int64_t>` and a trailing
/// `+ 1` stay defined, and `int_jump_candidates` (feasibility_jump.cpp) tests
/// against it before stepping a loop counter by 1.0 — past it, `v += 1.0` does
/// not advance and such a loop never terminates.
inline constexpr double kExactIntMagnitude = 9007199254740992.0;  // 2^53

/// A finite `[lo, hi]` interval over a scalar variable's domain.
///
/// `lo > hi` means *empty*, which only `int_sample_window` returns;
/// `domain_window` is total and always yields a non-empty subset of the
/// variable's own domain, so a value drawn from it is in-domain by construction.
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
///    distribution precondition, so they are narrowed to the clamp box.
///
/// Nothing else is rewritten. In particular the window is **not** trimmed to
/// what an `int64_t` can hold: that trim moved each Int bound independently and
/// so was neither inert on a finite domain (`[0, 1e17]` came back as
/// `[0, 2^53-1]`) nor a subset of one (`[-1e18, -1e17]` came back as the single
/// point `-2^53`, above the declared `ub`, which `random_in_domain` then
/// returned — a #112 defect). The cast is trimmed where the cast happens; see
/// `int_sample_window`. Pinned by "domain_window is inert on a finite domain"
/// and "domain_window is a subset of the declared domain".
///
/// Scalar types only (Bool/Int/Float). List/Set carry no `value`.
DomainWindow domain_window(const Variable& var);

/// The subset of `domain_window(var)` that `static_cast<int64_t>` and a
/// trailing `+1` can name. Rounds inward (`ceil`/`floor`) as well as clamping,
/// because the cast truncates toward zero and would leave the domain otherwise.
/// Empty (`lo > hi`) when the domain lies entirely past 2^53; each caller says
/// what it does then — `random_in_domain` draws from the untrimmed window,
/// `int_rand` drops the move, `movable_domain` reports immovable.
///
/// Int variables only; on any other type it is `domain_window` unchanged.
DomainWindow int_sample_window(const Variable& var);

/// Draw a uniformly random value from a scalar variable's domain. Finite and
/// in-domain for every domain — see `domain_window`. On the one Int case
/// `int_sample_window` cannot name (a domain wholly past 2^53) the draw falls
/// back to the untrimmed window, where every double is already an integer.
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
