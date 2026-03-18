#pragma once

#include "model.h"
#include "violation.h"
#include "moves.h"
#include "inner_solver.h"
#include "lns.h"
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

struct SolveProgress {
    int64_t iteration = 0;
    double time_seconds = 0.0;
    double objective = std::numeric_limits<double>::infinity();
    double total_violation = 0.0;
    double temperature = 0.0;
    bool feasible = false;
    bool new_best = false;
    int reheat_count = 0;
};

class SolveCallback {
public:
    virtual ~SolveCallback();
    virtual void on_progress(const SolveProgress& p) = 0;
};

void initialize_random(Model& model, RNG& rng);

void fj_nl_initialize(Model& model, ViolationManager& vm,
                       int max_iterations = 10000, RNG* rng = nullptr,
                       double time_limit = 2.0);

SearchResult solve(Model& model, double time_limit = 10.0,
                   uint64_t seed = 42, bool use_fj = true,
                   InnerSolverHook* hook = nullptr,
                   LNS* lns = nullptr,
                   int lns_interval = 3,
                   SolveCallback* callback = nullptr,
                   bool skip_init = false);

}  // namespace cbls
