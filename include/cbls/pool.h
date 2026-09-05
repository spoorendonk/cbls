#pragma once

#include "inner_solver.h"
#include "lns.h"
#include "model.h"
#include "rng.h"
#include "search.h"

#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <vector>

namespace cbls {

struct Solution {
    Model::State state;
    double objective = std::numeric_limits<double>::infinity();
    bool feasible = false;
};

class SolutionPool {
public:
    explicit SolutionPool(int capacity = 10);

    bool submit(const Solution& sol);
    std::optional<Solution> best() const;
    std::vector<Solution> top_k(int k) const;
    std::optional<Solution> get_restart_point(RNG& rng) const;
    size_t size() const;

private:
    int capacity_;
    std::vector<Solution> solutions_;
    mutable std::mutex mutex_;
};

struct ParallelConfig {
    int n_threads = 0;                // 0 = hardware_concurrency()
    bool deterministic = false;       // epoch-sync mode
    int64_t epoch_iterations = 5000;  // iterations per epoch
    int max_epochs = 10;              // number of epochs in deterministic mode
    int elite_pool_size = 4;          // top solutions to share between epochs
};

class ParallelSearch {
public:
    explicit ParallelSearch(int n_threads = 0);

    // Both overloads THROW in portfolio mode if *every* worker threw -- a model
    // factory that cannot build its model, say. Returning a default
    // SearchResult there would report "searched, found nothing feasible" about
    // a run that never searched. A partial failure is absorbed: the survivors'
    // best is returned and each dead worker contributes a default SearchResult
    // to the aggregate.
    //
    // Deterministic mode builds every worker's model on the CALLING thread, so
    // a factory failure there propagates straight out of solve() -- it always
    // has, this is not new. Only its epoch worker threads are left unwrapped,
    // so an exception raised inside one terminates the process. Guard a
    // deterministic solve() the same way you guard a portfolio one.
    // Simple portfolio solve (backward-compatible)
    SearchResult solve(std::function<Model()> model_factory, double time_limit = 10.0,
                       uint64_t seed = 42);

    // Full-featured solve with hooks, LNS, config, and parallel config
    SearchResult solve(std::function<Model()> model_factory, double time_limit, uint64_t seed,
                       const SearchConfig& config,
                       std::function<InnerSolverHook*(Model&)> hook_factory,
                       std::function<LNS*()> lns_factory, SolveCallback* callback,
                       const ParallelConfig& par_config);

private:
    int n_threads_;

    [[nodiscard]] int effective_threads(const ParallelConfig& pc) const;

    static SearchResult solve_portfolio(std::function<Model()>& model_factory, double time_limit,
                                        uint64_t seed, const SearchConfig& config,
                                        std::function<InnerSolverHook*(Model&)>& hook_factory,
                                        std::function<LNS*()>& lns_factory, SolveCallback* callback,
                                        int n_threads);

    static SearchResult solve_deterministic(std::function<Model()>& model_factory, uint64_t seed,
                                            const SearchConfig& config,
                                            std::function<InnerSolverHook*(Model&)>& hook_factory,
                                            std::function<LNS*()>& lns_factory,
                                            SolveCallback* callback,
                                            const ParallelConfig& par_config, int n_threads);
};

}  // namespace cbls
