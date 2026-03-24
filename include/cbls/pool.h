#pragma once

#include "model.h"
#include "search.h"
#include "inner_solver.h"
#include "lns.h"
#include "rng.h"
#include <limits>
#include <mutex>
#include <vector>
#include <optional>
#include <functional>

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
    int n_threads = 0;               // 0 = hardware_concurrency()
    bool deterministic = false;      // epoch-sync mode
    int64_t epoch_iterations = 5000; // iterations per epoch
    int max_epochs = 10;             // number of epochs in deterministic mode
    int elite_pool_size = 4;         // top solutions to share between epochs
};

class ParallelSearch {
public:
    explicit ParallelSearch(int n_threads = 0);

    // Simple portfolio solve (backward-compatible)
    SearchResult solve(std::function<Model()> model_factory,
                       double time_limit = 10.0, uint64_t seed = 42);

    // Full-featured solve with hooks, LNS, config, and parallel config
    SearchResult solve(
        std::function<Model()> model_factory,
        double time_limit,
        uint64_t seed,
        const SearchConfig& config,
        std::function<InnerSolverHook*(Model&)> hook_factory,
        std::function<LNS*()> lns_factory,
        SolveCallback* callback,
        const ParallelConfig& par_config);

private:
    int n_threads_;

    int effective_threads(const ParallelConfig& pc) const;

    SearchResult solve_portfolio(
        std::function<Model()>& model_factory,
        double time_limit, uint64_t seed,
        const SearchConfig& config,
        std::function<InnerSolverHook*(Model&)>& hook_factory,
        std::function<LNS*()>& lns_factory,
        SolveCallback* callback,
        int n_threads);

    SearchResult solve_deterministic(
        std::function<Model()>& model_factory,
        uint64_t seed,
        const SearchConfig& config,
        std::function<InnerSolverHook*(Model&)>& hook_factory,
        std::function<LNS*()>& lns_factory,
        SolveCallback* callback,
        const ParallelConfig& par_config,
        int n_threads);
};

}  // namespace cbls
