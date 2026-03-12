#pragma once

#include "model.h"
#include "violation.h"
#include "moves.h"
#include "rng.h"
#include <limits>

namespace cbls {

struct SearchResult {
    double objective = std::numeric_limits<double>::infinity();
    bool feasible = false;
    Model::State best_state;
    int64_t iterations = 0;
    double time_seconds = 0.0;
};

void initialize_random(Model& model, RNG& rng);

void fj_nl_initialize(Model& model, ViolationManager& vm,
                       int max_iterations = 10000, RNG* rng = nullptr);

SearchResult solve(Model& model, double time_limit = 10.0,
                   uint64_t seed = 42, bool use_fj = true);

}  // namespace cbls
