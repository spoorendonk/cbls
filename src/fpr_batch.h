#pragma once

#include "cbls/dag_propagation.h"
#include "cbls/domain_propagation.h"
#include "cbls/fpr.h"

#include <chrono>
#include <cstdint>
#include <memory>

namespace cbls {

class Model;
class RNG;

/// ViolationLS's fix-propagate-repair batch (#426): one `FprSearch` over a
/// row store built once from the solve's closed, evaluated model, the root
/// propagated at construction. Only built when
/// `SearchConfig::fpr_batch_probability > 0`, so off costs nothing.
///
/// Cost: construction is the store (O(nnz), ~24 B per nonzero), the
/// propagator and the engine's O(vars + nnz); each `run` is one dive. With
/// `dag_propagation` (#424) also a `DagRowStore`: O(the nonlinear rows'
/// subtrees), each revised on every scan.
class FprBatch {
public:
    /// `worker` keys the portfolio rotation (`fpr_portfolio_config`); a dive
    /// stops past `deadline` when `has_deadline`. `dag_propagation` adds the
    /// nonlinear monotone rows to the propagator (#424).
    FprBatch(Model& model, int64_t worker, bool has_deadline,
             std::chrono::steady_clock::time_point deadline, bool dag_propagation = false);

    /// Whether root propagation left the linear rows (and, with
    /// `dag_propagation`, the monotone nonlinear rows) consistent. When false,
    /// no attempt can branch; the caller builds no batch (#429).
    [[nodiscard]] bool root_ok() const { return root_ok_; }

    struct Outcome {
        /// The attempt's scalars were written: a leaf, or any fill of a
        /// repairing mode. A Dfs dive that ended without a leaf (out of
        /// siblings, or past the deadline) writes nothing: a failed attempt.
        bool written = false;
        /// The attempt's mode asks for leaf repair (Diveprop, Dfsrep).
        bool wants_repair = false;
    };

    /// Runs the next portfolio attempt from the model's current point and,
    /// per `Outcome::written`, writes its scalar values into the model (node
    /// values are left stale: the caller re-evaluates). Needs `root_ok()`.
    Outcome run(RNG& rng);

private:
    Model& model_;
    LinearRowStore store_;
    std::unique_ptr<DagRowStore> dag_;  // null unless dag_propagation
    DomainPropagator prop_;
    FprSearch search_;
    FprResult result_;
    bool root_ok_;
    int64_t worker_;
    int64_t attempt_ = 0;
};

}  // namespace cbls
