#include "cbls/pool.h"
#include <thread>
#include <algorithm>
#include <numeric>

namespace cbls {

SolutionPool::SolutionPool(int capacity) : capacity_(capacity) {}

bool SolutionPool::submit(const Solution& sol) {
    std::lock_guard<std::mutex> lock(mutex_);
    solutions_.push_back(sol);
    // Sort: feasible first, then by objective ascending
    std::sort(solutions_.begin(), solutions_.end(),
              [](const Solution& a, const Solution& b) {
                  if (a.feasible != b.feasible) return a.feasible > b.feasible;
                  return a.objective < b.objective;
              });
    if (static_cast<int>(solutions_.size()) > capacity_) {
        solutions_.resize(capacity_);
    }
    return true;
}

std::optional<Solution> SolutionPool::best() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (solutions_.empty()) return std::nullopt;
    return solutions_[0];
}

std::optional<Solution> SolutionPool::get_restart_point(RNG& rng) const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (solutions_.empty()) return std::nullopt;
    int n = std::max(1, static_cast<int>(solutions_.size()) / 2);
    int idx = static_cast<int>(rng.integers(0, n));
    return solutions_[idx];
}

size_t SolutionPool::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return solutions_.size();
}

ParallelSearch::ParallelSearch(int n_threads) : n_threads_(n_threads) {}

SearchResult ParallelSearch::solve(std::function<Model()> model_factory,
                                    double time_limit, uint64_t seed) {
    SolutionPool pool;
    std::vector<SearchResult> results(n_threads_);
    std::vector<std::thread> threads;

    for (int i = 0; i < n_threads_; ++i) {
        threads.emplace_back([&, i]() {
            try {
                Model m = model_factory();
                results[i] = cbls::solve(m, time_limit, seed + i);

                Solution sol;
                sol.state = results[i].best_state;
                sol.objective = results[i].objective;
                sol.feasible = results[i].feasible;
                pool.submit(sol);
            } catch (...) {
                // Don't crash the whole search
            }
        });
    }

    for (auto& t : threads) t.join();

    auto best = pool.best();
    if (best) {
        SearchResult result;
        result.objective = best->objective;
        result.feasible = best->feasible;
        result.best_state = best->state;
        return result;
    }

    // Fallback
    SearchResult best_result;
    for (const auto& r : results) {
        if (r.feasible && !best_result.feasible) {
            best_result = r;
        } else if (r.feasible == best_result.feasible && r.objective < best_result.objective) {
            best_result = r;
        }
    }
    return best_result;
}

}  // namespace cbls
