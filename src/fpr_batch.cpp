#include "fpr_batch.h"

#include "cbls/dag.h"
#include "cbls/model.h"

#include <cstddef>

namespace cbls {

FprBatch::FprBatch(Model& model, int64_t worker, bool has_deadline,
                   std::chrono::steady_clock::time_point deadline)
    : model_(model),
      store_(model),
      prop_(store_, model),
      search_(model, store_, prop_),
      root_ok_(search_.propagate_root()),
      worker_(worker) {
    if (has_deadline) {
        search_.set_deadline(deadline);
    }
}

FprBatch::Outcome FprBatch::run(RNG& rng) {
    const FprConfig cfg = fpr_portfolio_config(worker_, attempt_);
    search_.run(cfg, attempt_, rng, result_);
    ++attempt_;
    if (!result_.complete && !result_.wants_repair) {
        return {false, false};
    }
    for (size_t v = 0; v < result_.values.size(); ++v) {
        Variable& var = model_.var_mut(static_cast<int32_t>(v));
        if (!is_structured(var.type)) {
            var.value = result_.values[v];
        }
    }
    return {true, result_.wants_repair};
}

}  // namespace cbls
