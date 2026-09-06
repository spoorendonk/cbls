#pragma once

#include "model.h"
#include "rng.h"
#include "violation.h"

namespace cbls {

/// Destroy-and-repair diversification for the ViolationLS loop.
///
/// An extension point, like InnerSolverHook: `solve()` takes an `LNS*` and
/// `ParallelSearch` a factory returning `shared_ptr<LNS>`, so a caller can
/// substitute its own destroy/repair strategy by overriding `destroy_repair`.
/// Hence the virtual destructor — `ParallelSearch` holds one instance per
/// worker through that shared_ptr, and for a factory that came from Python the
/// pointee is an object the interpreter also references (issue #129).
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

private:
    double destroy_fraction_;
};

}  // namespace cbls
