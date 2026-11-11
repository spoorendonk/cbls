#pragma once

#include "model.h"
#include "rng.h"
#include "violation.h"

#include <memory>
#include <utility>

namespace cbls {

class NeighbourList;

/// The key the built-in LNS accepts a repair on, compared lexicographically:
/// (real violation, objective). The real violation sums each row's residual
/// ABOVE `tol` -- a row at or below it is satisfied and adds nothing -- over
/// every row but the artificial `obj <= bound`; a NaN row makes it +inf, and a
/// NaN objective ranks as +inf. Public so the rule can be tested directly.
std::pair<double, double> lns_acceptance_key(const Model& model, double tol);

/// Destroy-and-repair diversification for the ViolationLS loop.
///
/// Two destroy/repair pairs, by variable kind (#202):
///
///  - Every `ListPartition`'s members are ruined and recreated (ASB-RR/SISR,
///    Christiaens & Vanden Berghe, Transportation Science 54(2) 2020, with the
///    paper's parameters; `ruin_recreate_partition` in src/lns.cpp lists the
///    two departures): strings are removed from a few lists and reinserted one
///    by one at their best position under the acceptance key, probed through
///    the DAG. FJ cannot move a List, so perturb-then-FJ only worsens routes.
///  - Every other variable: a random `destroy_fraction` is randomised and an
///    FJ pass repairs it. A model with no partition draws as it always did.
///
/// An extension point, like InnerSolverHook: `solve()` takes an `LNS*` and
/// `ParallelSearch` a `shared_ptr<LNS>` factory, so C++ can override
/// `destroy_repair`. C++ only: the binding has no trampoline, so a Python
/// override is never dispatched to and this base runs instead. The virtual
/// destructor serves deletion through a base `LNS*` and nanobind's
/// polymorphic downcast (#129).
class LNS {
public:
    explicit LNS(double destroy_fraction = 0.3);
    virtual ~LNS();

    // `repair_time_limit` seconds bound the FJ repair; <= 0 leaves only its
    // iteration budget, which makes it deterministic. `solve()` passes
    // `min(2.0, remaining budget)`: no early kick monopolises a long run, no
    // late one overruns the deadline. An override must honour it, or `solve()`
    // breaks its wall-clock budget.
    //
    // NO DEFAULT ARGUMENT: defaults bind statically, so through an `LNS*` (how
    // `solve()` calls) an override's default would be silently ignored.
    // `destroy_repair_cycle` is non-virtual, so its default is safe.
    virtual bool destroy_repair(Model& model, ViolationManager& vm, RNG& rng,
                                double repair_time_limit);
    int destroy_repair_cycle(Model& model, ViolationManager& vm, RNG& rng, int n_rounds = 10,
                             double repair_time_limit = 2.0);

    /// The residual a row may keep and count as satisfied, in the repair's FJ
    /// and the acceptance key (which clamps each row to 0 at or below it, so
    /// feasible states rank by objective, not rounding noise, #211). `solve()`
    /// sets it to `SearchConfig::feasibility_tolerance` before each kick.
    void set_feasibility_tolerance(double tol) { feasibility_tolerance_ = tol; }
    [[nodiscard]] double feasibility_tolerance() const { return feasibility_tolerance_; }

    /// The relatedness the partition ruin draws strings by: each element's
    /// neighbours, nearest first (#202). `solve()` sets it from
    /// `SearchConfig::structural_neighbours` before each kick. Null (default):
    /// after the seed, strings are taken near randomly drawn elements.
    void set_neighbours(std::shared_ptr<const NeighbourList> neighbours) {
        neighbours_ = std::move(neighbours);
    }
    [[nodiscard]] const std::shared_ptr<const NeighbourList>& neighbours() const {
        return neighbours_;
    }

private:
    double destroy_fraction_;
    double feasibility_tolerance_ = kDefaultFeasibilityTolerance;
    std::shared_ptr<const NeighbourList> neighbours_;
};

}  // namespace cbls
