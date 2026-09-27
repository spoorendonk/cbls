#pragma once

// The MIPfeas runner's share of #170: how many portfolio workers actually ran to
// the end, as row columns, the message a lost worker earns, and what it does to
// the exit code.
//
// `threads` is what was ASKED for, and it is a scorer configuration key: a worker
// that died inside the solve bracket -- a bad_alloc on its FJ tables under the
// driver's address-space cap, after every replica fitted -- is absorbed by
// ParallelSearch, which returns the survivors' result. Without these columns such
// a row reads as a valid N-thread measurement; with them the scorer refuses it.
// "Completed" is defined on SearchResult::workers_completed. The single-threaded
// arm reports 1 of 1: cbls::solve has one worker, and a throw from it is a
// `solve_error` row instead.
//
// Kept as pure functions in a header, as benchmarks/minlplib/note_policy.h is,
// so they are testable on a synthesised SearchResult: a lost worker cannot be
// provoked in the runner binary without a test seam in production code.

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

/// The runner's exit code once it has written a row. Non-zero when the solution
/// could not be written -- the job ran, but its row cannot be verified -- and
/// when the portfolio lost workers: the row is written, so the failures are on
/// disk with their reasons, but the job did not measure what it was asked to.
/// The driver reads a non-zero exit as a failed job either way.
inline int runner_exit_code(bool solution_write_failed, bool lost_workers) {
    return (solution_write_failed || lost_workers) ? 1 : 0;
}

}  // namespace cbls::mipfeas
