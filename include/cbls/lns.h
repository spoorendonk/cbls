#pragma once

#include "model.h"
#include "violation.h"
#include "rng.h"

namespace cbls {

class LNS {
public:
    explicit LNS(double destroy_fraction = 0.3);

    // `repair_time_limit` seconds bound the FJ repair pass; <= 0 means the
    // repair is bounded by its iteration budget alone, which makes it
    // deterministic. The ViolationLS loop passes its own remaining budget so a
    // repair started near the deadline cannot overrun it.
    bool destroy_repair(Model& model, ViolationManager& vm, RNG& rng,
                        double repair_time_limit = 2.0);
    int destroy_repair_cycle(Model& model, ViolationManager& vm, RNG& rng, int n_rounds = 10,
                             double repair_time_limit = 2.0);

private:
    double destroy_fraction_;
};

}  // namespace cbls
