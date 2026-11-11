#pragma once

#include "dag.h"
#include "rng.h"

#include <cstdint>

// Uniform randomisation of a single variable over its own domain: the one
// implementation behind initialize_random / initialize_structured_random,
// LNS's destroy step and the scalar half of FJ's kick (#112; the kick's
// List/Set half uses typed moves, #111).
//
// Every scalar entry point is total: a finite in-domain value for any domain
// the model can hold, unbounded ones included (see `domain_window`). This
// covers only randomisation: a Float jump candidate still reads var.lb/var.ub
// directly.

namespace cbls {

class Model;
struct ListPartition;

/// Magnitude standing in for an infinite Bool/Float bound when sampling; equal
/// to NlToModelOptions::inf_clamp and MpsToModelOptions::inf_clamp, so an API
/// model lands in the box the readers would give it.
inline constexpr double kRandomInfClamp = 1.0e9;

/// Magnitude standing in for an infinite Int bound when sampling; equal to
/// NlToModelOptions::int_inf_clamp.
inline constexpr double kRandomIntInfClamp = 1.0e6;

/// Largest magnitude at which a double names every integer below it (2^53).
/// `int_sample_window` trims to it so the int64_t cast and `+ 1` stay defined;
/// `int_jump_candidates` checks it before stepping by 1.0, which past it does
/// not advance.
inline constexpr double kExactIntMagnitude = 9007199254740992.0;  // 2^53

/// A finite `[lo, hi]` interval over a scalar variable's domain. `lo > hi`
/// means empty, which only `int_sample_window` returns; `domain_window` always
/// yields a non-empty subset of the domain.
struct DomainWindow {
    double lo = 0.0;
    double hi = 0.0;
};

/// `var`'s bounds, made safe to sample and to cast. A finite declared bound is
/// returned untouched, so models that need no guard keep their RNG draws bit
/// for bit. Rewritten only:
///  - an infinite bound, to the clamp magnitude (`kRandomIntInfClamp` for Int,
///    else `kRandomInfClamp`), since `uniform_real_distribution` needs
///    `ub - lb <= DBL_MAX`;
///  - on a half-infinite domain, the substituted end is pushed past the declared
///    one where needed, so the window stays inside the domain;
///  - two finite bounds whose width overflows, to the clamp box.
/// Not trimmed to int64_t range (that is `int_sample_window`'s job, #112).
/// Scalar types only.
DomainWindow domain_window(const Variable& var);

/// The subset of `domain_window(var)` that the int64_t cast and `+1` can name,
/// rounded inward. Empty (`lo > hi`) when the domain lies wholly past 2^53;
/// then `random_in_domain` draws from the untrimmed window, `int_rand` drops
/// the move and `movable_domain` reports immovable. Int only; otherwise
/// `domain_window`.
DomainWindow int_sample_window(const Variable& var);

/// Whether a scalar variable's domain holds at least two values (#109, #201).
/// Bool reads its bounds; Int reads `int_sample_window` (`[0.9, 1.2]` is
/// immovable); Float needs positive width. Scalar types only.
bool movable_domain(const Variable& var);

/// A uniform value from a scalar variable's domain, finite and in-domain (see
/// `domain_window`). Bool draws from {0, 1} without reading its bounds, so a
/// Bool pinned by `lb == ub` (unreachable today) would not be respected.
double random_in_domain(const Variable& var, RNG& rng);

/// How a List's new order relates to its current one. On a permutation List
/// both draw a uniform permutation with identical RNG use; on a variable-length
/// List (#164) only `Perturb` preserves length and membership, which keeps a
/// ListPartition's cover intact under LNS.
enum class ListOrder : std::uint8_t {
    /// Lay the List out from scratch per its `ListInit` (initialisers).
    Regenerate,
    /// Shuffle the current `elements` in place, keeping membership (LNS
    /// destroy/repair).
    Perturb,
};

/// Redraw a List/Set variable's `elements` per its `ListInit` (List) or as a
/// uniform admissible subset (Set). Per variable, so it cannot lay out a
/// ListPartition: use `randomize_list_partition`. No-op on a scalar.
void randomize_structured_var(Variable& var, RNG& rng, ListOrder order = ListOrder::Regenerate);

/// Randomise one variable in place, whatever its type: `value` for a scalar,
/// `elements` for a List/Set.
void randomize_var(Variable& var, RNG& rng, ListOrder order = ListOrder::Regenerate);

/// Lay a ListPartition out as a random assignment satisfying its cover (#164).
/// Not uniform once a list's length bound binds. The initialisers call it
/// instead of `randomize_var` on the members, so an `Exact` partition starts
/// complete (no `Exact` move can repair one that is not).
void randomize_list_partition(Model& model, const ListPartition& part, RNG& rng);

}  // namespace cbls
