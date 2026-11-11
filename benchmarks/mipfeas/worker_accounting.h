#pragma once

// MIPfeas worker accounting (#170): row columns for workers that ran to the end
// (SearchResult::workers_completed), the lost-worker message and exit code.
// `threads` is what was asked for (a scorer key); ParallelSearch absorbs a dead
// worker (e.g. bad_alloc under the driver's cap) and returns the survivors'
// result, which these columns let the scorer refuse. Single-threaded reports 1
// of 1; a throw there is `solve_error`. Pure functions, as benchmarks/minlplib/note_policy.h, so a
// test can use a synthesised SearchResult without a production test seam.

#include <cbls/search.h>
#include <nlohmann/json.hpp>
#include <string>

namespace cbls::mipfeas {

/// Adds `workers_launched`, `workers_completed` and `worker_failures` (one
/// object per failure: worker, produced_result, reason) to `row`.
inline void add_worker_accounting(const SearchResult& result, nlohmann::json& row) {
    row["workers_launched"] = result.workers_launched;
    row["workers_completed"] = result.workers_completed;
    nlohmann::json failures = nlohmann::json::array();
    for (const WorkerFailure& f : result.worker_failures) {
        failures.push_back(
            {{"worker", f.worker}, {"produced_result", f.produced_result}, {"reason", f.reason}});
    }
    row["worker_failures"] = failures;
}

/// Whether the portfolio completed fewer workers than `threads` asked for.
inline bool lost_workers(const SearchResult& result, int threads) {
    return result.workers_completed < threads;
}

/// What the runner prints to stderr about a row that lost workers, one line for
/// the count and one per failure with its reason. Empty when nothing was lost.
inline std::string lost_workers_message(const std::string& instance, int threads,
                                        const SearchResult& result) {
    if (!lost_workers(result, threads)) {
        return {};
    }
    std::string out = instance + ": only " + std::to_string(result.workers_completed) + " of " +
                      std::to_string(threads) +
                      " portfolio workers completed; the row is refused\n";
    for (const WorkerFailure& f : result.worker_failures) {
        out += "  worker " + std::to_string(f.worker) +
               (f.produced_result ? " (after producing a result)" : "") + ": " + f.reason + "\n";
    }
    return out;
}

/// The exit code after the row is written: 1 when the solution could not be
/// written (unverifiable row) or workers were lost (row records why, but did
/// not measure what was asked); the driver reads 1 as a failed job.
inline int runner_exit_code(bool solution_write_failed, bool lost_workers) {
    return (solution_write_failed || lost_workers) ? 1 : 0;
}

}  // namespace cbls::mipfeas
