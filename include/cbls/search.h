#pragma once

#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "moves.h"
#include "rng.h"
#include "violation.h"

#include <limits>

namespace cbls {

struct SearchConfig {
    // SA-era knobs — unused by the ViolationLS solve loop. Kept until the
    // bindings/CLI that still reference them are swept (post-P2 cleanup).
    double cooling_rate = 0.9999;
    int reheat_interval = 5000;
    int hook_frequency = 10;
    double fj_time_fraction = 0.2;

    bool skip_init = false;
    int64_t max_iterations = 0;  // 0 = unlimited (use time_limit); counts GLS iterations
    bool use_fj = true;
    int lns_interval = 3;

    // ViolationLS batch outer loop (Algorithm 6).
    int64_t batch_iterations = 1000;        // GLS iterations per batch
    int perturbation_period = 100;          // batches without improvement before perturbing
    double perturbation_probability = 0.1;  // per-variable randomisation probability
    // Novelty Jump is implemented, wired, and unit-tested, but OFF by default:
    // its per-batch cost is not yet bounded tightly enough for the large
    // continuous benchmarks (it burns the time budget there). Enable + tune
    // (probability, work budget, when-stuck-only) in P5 (#70); the paper uses
    // 0.5 with deterministic-time-bounded batches.
    bool use_compound_moves = false;        // run Novelty Jump batches (else FJ only)
    double novelty_jump_probability = 0.5;  // P(a batch is Novelty Jump)
};

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

void fj_nl_initialize(Model& model, ViolationManager& vm, int max_iterations = 10000,
                      RNG* rng = nullptr, double time_limit = 2.0);

SearchResult solve(Model& model, double time_limit = 10.0, uint64_t seed = 42, bool use_fj = true,
                   InnerSolverHook* hook = nullptr, LNS* lns = nullptr, int lns_interval = 3,
                   SolveCallback* callback = nullptr, const SearchConfig& config = {});

}  // namespace cbls
