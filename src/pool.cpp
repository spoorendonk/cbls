#include "cbls/pool.h"
#include "cbls/dag_ops.h"
#include <thread>
#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>

namespace cbls {

// --- SolutionPool ---

SolutionPool::SolutionPool(int capacity) : capacity_(capacity) {}

bool SolutionPool::submit(const Solution& sol) {
    std::lock_guard<std::mutex> lock(mutex_);
    solutions_.push_back(sol);
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

std::vector<Solution> SolutionPool::top_k(int k) const {
    std::lock_guard<std::mutex> lock(mutex_);
    int n = std::min(k, static_cast<int>(solutions_.size()));
    return std::vector<Solution>(solutions_.begin(), solutions_.begin() + n);
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

// --- ParallelSearch ---

ParallelSearch::ParallelSearch(int n_threads) : n_threads_(n_threads) {}

int ParallelSearch::effective_threads(const ParallelConfig& pc) const {
    int n = pc.n_threads > 0 ? pc.n_threads : n_threads_;
    if (n <= 0) {
        n = static_cast<int>(std::thread::hardware_concurrency());
    }
    return std::max(1, n);
}

// Backward-compatible simple solve
SearchResult ParallelSearch::solve(std::function<Model()> model_factory,
                                    double time_limit, uint64_t seed) {
    ParallelConfig pc;
    pc.n_threads = n_threads_;
    std::function<InnerSolverHook*(Model&)> no_hook;
    std::function<LNS*()> no_lns;
    int n = effective_threads(pc);
    return solve_portfolio(model_factory, time_limit, seed, {}, no_hook, no_lns, nullptr, n);
}

// Full-featured solve
SearchResult ParallelSearch::solve(
    std::function<Model()> model_factory,
    double time_limit,
    uint64_t seed,
    const SearchConfig& config,
    std::function<InnerSolverHook*(Model&)> hook_factory,
    std::function<LNS*()> lns_factory,
    SolveCallback* callback,
    const ParallelConfig& par_config) {

    int n = effective_threads(par_config);

    if (par_config.deterministic) {
        return solve_deterministic(model_factory, seed, config,
                                   hook_factory, lns_factory, callback, par_config, n);
    } else {
        return solve_portfolio(model_factory, time_limit, seed, config,
                               hook_factory, lns_factory, callback, n);
    }
}

// Portfolio workers are homogeneous — same model, same budget, different seed —
// so their termination reasons almost always agree, and this only has to break
// ties. Precedence: a worker that actually finished the job (Feasible) outranks
// any budget exit; among budget exits the shared wall clock outranks the
// per-worker iteration budget, because the portfolio's answer is clock-limited
// as soon as any worker ran the clock out. NoBudget is last: it is also what a
// worker that threw leaves behind, and one crashed thread should not relabel a
// run the others budget-limited.
static TerminationReason aggregate_termination(const std::vector<SearchResult>& results) {
    bool any_time = false;
    bool any_iterations = false;
    for (const auto& r : results) {
        if (r.termination == TerminationReason::Feasible) {
            return TerminationReason::Feasible;
        }
        any_time = any_time || r.termination == TerminationReason::TimeLimit;
        any_iterations = any_iterations || r.termination == TerminationReason::IterationLimit;
    }
    if (any_time) {
        return TerminationReason::TimeLimit;
    }
    return any_iterations ? TerminationReason::IterationLimit : TerminationReason::NoBudget;
}

// --- Portfolio (opportunistic) mode ---

SearchResult ParallelSearch::solve_portfolio(
    std::function<Model()>& model_factory,
    double time_limit, uint64_t seed,
    const SearchConfig& config,
    std::function<InnerSolverHook*(Model&)>& hook_factory,
    std::function<LNS*()>& lns_factory,
    SolveCallback* callback,
    int n_threads) {

    SolutionPool pool;
    std::vector<SearchResult> results(n_threads);
    std::vector<std::thread> threads;

    for (int i = 0; i < n_threads; ++i) {
        threads.emplace_back([&, i]() {
            try {
                Model m = model_factory();

                std::unique_ptr<InnerSolverHook> hook;
                if (hook_factory) hook.reset(hook_factory(m));

                std::unique_ptr<LNS> lns;
                if (lns_factory) lns.reset(lns_factory());

                // Only thread 0 gets the callback to avoid interleaved output
                SolveCallback* cb = (i == 0) ? callback : nullptr;

                results[i] = cbls::solve(m, time_limit, seed + i,
                                         config.use_fj, hook.get(),
                                         lns.get(), config.lns_interval,
                                         cb, config);

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

    // Aggregate results
    auto best = pool.best();
    if (best) {
        SearchResult result;
        result.objective = best->objective;
        result.feasible = best->feasible;
        result.best_state = best->state;
        // Sum iterations and take max time across threads
        int64_t total_iters = 0;
        double max_time = 0.0;
        for (const auto& r : results) {
            total_iters += r.iterations;
            max_time = std::max(max_time, r.time_seconds);
        }
        result.iterations = total_iters;
        result.time_seconds = max_time;
        result.termination = aggregate_termination(results);
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

// --- Deterministic epoch-sync mode ---

SearchResult ParallelSearch::solve_deterministic(
    std::function<Model()>& model_factory,
    uint64_t seed,
    const SearchConfig& config,
    std::function<InnerSolverHook*(Model&)>& hook_factory,
    std::function<LNS*()>& lns_factory,
    SolveCallback* callback,
    const ParallelConfig& par_config,
    int n_threads) {

    auto start = std::chrono::steady_clock::now();

    int elite_k = std::max(1, par_config.elite_pool_size);
    int64_t epoch_iters = std::max(int64_t(1), par_config.epoch_iterations);
    int max_epochs = std::max(1, par_config.max_epochs);

    // Each thread owns its model; initialize from factory
    std::vector<Model> models(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        models[i] = model_factory();
    }

    // Per-thread hooks and LNS
    std::vector<std::unique_ptr<InnerSolverHook>> hooks(n_threads);
    std::vector<std::unique_ptr<LNS>> lns_objs(n_threads);
    for (int i = 0; i < n_threads; ++i) {
        if (hook_factory) hooks[i].reset(hook_factory(models[i]));
        if (lns_factory) lns_objs[i].reset(lns_factory());
    }

    // Global best tracking
    SolutionPool pool(elite_k);
    SearchResult global_best;
    int64_t total_iterations = 0;

    // Per-thread results for each epoch
    std::vector<SearchResult> epoch_results(n_threads);

    // In deterministic mode, epochs stop by iteration count, not wall-clock.
    // 0 disables the wall clock entirely in solve(), so nothing can cut an epoch
    // short and make the run depend on machine speed. (The previous sentinel,
    // double::max(), was not itself broken - it produced a floating-rep +inf
    // deadline - but solve() now converts to the clock's integer tick type,
    // where such a value overflows.)
    const double epoch_time_limit = 0.0;

    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        // Configure per-epoch search: iteration-limited, no FJ after first epoch
        SearchConfig epoch_config = config;
        epoch_config.max_iterations = epoch_iters;
        if (epoch > 0) {
            epoch_config.skip_init = true;
        }

        // Launch threads for this epoch
        std::vector<std::thread> threads;
        for (int i = 0; i < n_threads; ++i) {
            threads.emplace_back([&, i, epoch]() {
                // Deterministic seed: base_seed + epoch * n_threads + thread_id
                uint64_t thread_seed = seed + static_cast<uint64_t>(epoch) * n_threads + i;

                SolveCallback* cb = (i == 0) ? callback : nullptr;

                epoch_results[i] = cbls::solve(
                    models[i], epoch_time_limit, thread_seed,
                    (epoch == 0) && config.use_fj,
                    hooks[i].get(), lns_objs[i].get(),
                    config.lns_interval, cb, epoch_config);
            });
        }

        for (auto& t : threads) t.join();

        // Collect results into pool
        for (int i = 0; i < n_threads; ++i) {
            Solution sol;
            sol.state = epoch_results[i].best_state;
            sol.objective = epoch_results[i].objective;
            sol.feasible = epoch_results[i].feasible;
            pool.submit(sol);
            total_iterations += epoch_results[i].iterations;
        }

        // Redistribute elite solutions to threads for next epoch
        auto elite = pool.top_k(elite_k);
        if (!elite.empty()) {
            for (int i = 0; i < n_threads; ++i) {
                const auto& sol = elite[i % static_cast<int>(elite.size())];
                models[i].restore_state(sol.state);
                full_evaluate(models[i]);
            }
        }
    }

    // Build final result from pool
    auto best = pool.best();
    if (best) {
        global_best.objective = best->objective;
        global_best.feasible = best->feasible;
        global_best.best_state = best->state;
    }
    global_best.iterations = total_iterations;
    global_best.time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    // Epoch-sync mode runs every epoch with epoch_time_limit = 0 (no wall clock
    // at all, by design — see above), so the whole run is iteration-bounded:
    // epoch_iterations per worker per epoch, max_epochs epochs. The only other
    // honest answer is a worker that finished outright, which aggregate_termination
    // reports; it can never be TimeLimit here.
    global_best.termination = aggregate_termination(epoch_results);

    return global_best;
}

}  // namespace cbls
