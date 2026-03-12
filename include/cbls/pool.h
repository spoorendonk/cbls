#pragma once

#include "model.h"
#include "search.h"
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
    std::optional<Solution> get_restart_point(RNG& rng) const;
    size_t size() const;

private:
    int capacity_;
    std::vector<Solution> solutions_;
    mutable std::mutex mutex_;
};

class ParallelSearch {
public:
    explicit ParallelSearch(int n_threads = 4);

    SearchResult solve(std::function<Model()> model_factory,
                       double time_limit = 10.0, uint64_t seed = 42);

private:
    int n_threads_;
};

}  // namespace cbls
