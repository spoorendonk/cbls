#pragma once

#include "model.h"
#include "rng.h"
#include "violation.h"

#include <utility>

namespace cbls {

/// The key the built-in LNS accepts a repair on, compared lexicographically:
/// (real violation, objective). The real violation sums each row's residual
/// ABOVE `tol` -- a row at or below it is satisfied and adds nothing -- over
/// every row but the artificial `obj <= bound`; a NaN row makes it +inf, and a
/// NaN objective ranks as +inf. Public so the rule can be tested directly.
std::pair<double, double> lns_acceptance_key(const Model& model, double tol);

/// Destroy-and-repair diversification for the ViolationLS loop.
///
/// An extension point, like InnerSolverHook: `solve()` takes an `LNS*` and
/// `ParallelSearch` a factory returning `shared_ptr<LNS>`, so a C++ caller can
/// substitute its own destroy/repair strategy by overriding `destroy_repair`.
/// C++ only, for now: the Python binding registers `LNS` without a trampoline,
/// so an override there is never dispatched to and this base runs instead.
///
/// Hence the virtual destructor — a caller that deletes an override through a
/// base `LNS*` needs it, and nanobind keys its polymorphic downcast on it.
/// Note that `ParallelSearch`'s `shared_ptr<LNS>` is NOT the reason: a
/// shared_ptr captures its deleter from the most-derived type at construction,
/// so it would destroy an override correctly either way. That used to be the
/// reason, when the pointer was a `unique_ptr<LNS>` (issue #129).
class LNS {
public:
    explicit LNS(double destroy_fraction = 0.3);
    virtual ~LNS();

    // `repair_time_limit` seconds bound the FJ repair pass; <= 0 means the
    // repair is bounded by its iteration budget alone, which makes it
    // deterministic. `solve()` passes `min(2.0, remaining budget)` — capped so a
    // single kick early in a long run cannot monopolise it, and floored by the
    // remaining budget so a kick taken near the deadline cannot overrun it.
    //
    // An override must honour `repair_time_limit`: that is what keeps `solve()`
    // inside the wall-clock budget it promised its caller.
    //
    // NO DEFAULT ARGUMENT, deliberately. Default arguments are bound statically,
    // so on a virtual function the *base's* default wins on every call through
    // an `LNS*` — which is the only way `solve()` ever calls this — and an
    // override's own default would be silently ignored. Requiring the argument
    // removes the trap. `destroy_repair_cycle` below is non-virtual, so its
    // default is safe.
    virtual bool destroy_repair(Model& model, ViolationManager& vm, RNG& rng,
                                double repair_time_limit);
    int destroy_repair_cycle(Model& model, ViolationManager& vm, RNG& rng, int n_rounds = 10,
                             double repair_time_limit = 2.0);

    /// The residual a row may keep and still count as satisfied, both in the
    /// repair's FJ and in the acceptance key, which clamps each row's violation
    /// to 0 at or below it. `solve()` sets it to `SearchConfig::
    /// feasibility_tolerance` before each kick. Without the clamp two feasible
    /// states were ranked by sub-tolerance rounding noise on their equality
    /// rows (1e-12 against 3e-12), and the objective was never consulted (#211).
    void set_feasibility_tolerance(double tol) { feasibility_tolerance_ = tol; }
    [[nodiscard]] double feasibility_tolerance() const { return feasibility_tolerance_; }

    /// Whether the repair's FJ takes the exact Int jump (GFJConfig::
    /// exact_int_jump, #216). `solve()` sets it from `SearchConfig::
    /// fj_exact_int_jump` before each kick; off by default.
    void set_exact_int_jump(bool on) { exact_int_jump_ = on; }
    [[nodiscard]] bool exact_int_jump() const { return exact_int_jump_; }

    /// Whether the repair's FJ offers exact quadratic roots (GFJConfig::
    /// quadratic_roots, #221). `solve()` sets it from `SearchConfig::
    /// fj_quadratic_roots` before each kick; off by default.
    void set_quadratic_roots(bool on) { quadratic_roots_ = on; }
    [[nodiscard]] bool quadratic_roots() const { return quadratic_roots_; }

private:
    double destroy_fraction_;
    double feasibility_tolerance_ = kDefaultFeasibilityTolerance;
    bool exact_int_jump_ = false;
    bool quadratic_roots_ = false;
};

}  // namespace cbls
