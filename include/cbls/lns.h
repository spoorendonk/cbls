#pragma once

#include "model.h"
#include "violation.h"
#include "rng.h"

namespace cbls {

class LNS {
public:
    explicit LNS(double destroy_fraction = 0.3);

    bool destroy_repair(Model& model, ViolationManager& vm, RNG& rng);
    int destroy_repair_cycle(Model& model, ViolationManager& vm, RNG& rng, int n_rounds = 10);

private:
    double destroy_fraction_;
};

}  // namespace cbls
