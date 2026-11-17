#pragma once

#include "cbls/domain_propagation.h"
#include "cbls/fpr.h"

#include <cstdint>

namespace cbls {

class Model;
class RNG;

/// ViolationLS's fix-propagate-repair batch (#426): one `FprSearch` over a
/// row store built once from the solve's closed, evaluated model, the root
/// propagated at construction. Only built when
/// `SearchConfig::fpr_batch_probability > 0`, so off costs nothing.
///
/// Cost: construction is the store (O(nnz), ~24 B per nonzero), the
/// propagator and the engine's O(vars + nnz); each `run` is one dive.
class FprBatch {
public:
    /// `worker` keys the portfolio rotation (`fpr_portfolio_config`).
    FprBatch(Model& model, int64_t worker);

    struct Outcome {
        /// The root is consistent and the attempt's scalars were written.
        bool ran = false;
        /// The attempt's mode asks for leaf repair (Diveprop, Dfsrep).
        bool wants_repair = false;
    };

    /// Runs the next portfolio attempt from the model's current point and
    /// writes its scalar values into the model (node values are left stale: the
    /// caller re-evaluates). A refuted root writes nothing.
    Outcome run(RNG& rng);

private:
    Model& model_;
    LinearRowStore store_;
    DomainPropagator prop_;
    FprSearch search_;
    FprResult result_;
    bool root_ok_;
    int64_t worker_;
    int64_t attempt_ = 0;
};

}  // namespace cbls
