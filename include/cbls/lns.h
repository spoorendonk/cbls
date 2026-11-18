#pragma once

#include "model.h"
#include "rng.h"
#include "violation.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cbls {

class LinearRowStore;
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
    // Move-only: the FPR repair's engine is bound to one store (#431).
    LNS(LNS&&) noexcept;
    LNS& operator=(LNS&&) noexcept;

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

    /// Fix-propagate-repair before the scalar repair's FJ (#431): the linear
    /// rows of the model being repaired, built by `solve()` when
    /// `SearchConfig::lns_repair_fpr` is on, set before each kick and cleared
    /// when the solve ends. Null (default): plain FJ repair, drawing exactly as
    /// before. With a store, each round fixes the kept scalars at their
    /// incumbent values under propagation (a conflict skips FPR for the round),
    /// dives once over the destroyed ones from their randomised values, writes
    /// their values, then FJ repairs. The propagator and engine are built on
    /// the first round with a new store; a refuted root skips FPR for good. The
    /// store must be built from the model repaired, with the feasibility
    /// tolerance as its `row_tolerance` (as `solve()` does) so that a kept value
    /// a row only just accepts holds (#433); one of another width skips FPR.
    void set_repair_store(std::shared_ptr<const LinearRowStore> store);
    [[nodiscard]] const std::shared_ptr<const LinearRowStore>& repair_store() const {
        return repair_store_;
    }

    /// What the FPR repair did, summed over this LNS's rounds (#431).
    struct FprRepairStats {
        int64_t dives = 0;           ///< rounds that ran a dive and wrote its values
        int64_t kept_conflicts = 0;  ///< rounds whose kept values propagation refuted
        int64_t root_refuted = 0;    ///< rounds skipped: the linear rows refute the root
        int64_t nodes = 0;           ///< dive nodes, summed
    };
    [[nodiscard]] const FprRepairStats& fpr_repair_stats() const { return fpr_stats_; }

protected:
    // Holds the kept scalars and writes one dive's values into `destroyed`;
    // `deadline` (nullable) stops the dive. True when values were written.
    bool fpr_repair(Model& model, const std::vector<int32_t>& destroyed, RNG& rng,
                    const std::chrono::steady_clock::time_point* deadline);

private:
    struct FprRepair;

    double destroy_fraction_;
    double feasibility_tolerance_ = kDefaultFeasibilityTolerance;
    std::shared_ptr<const NeighbourList> neighbours_;
    std::shared_ptr<const LinearRowStore> repair_store_;
    std::unique_ptr<FprRepair> fpr_;  // over `repair_store_`, built lazily
    FprRepairStats fpr_stats_;
};

}  // namespace cbls
