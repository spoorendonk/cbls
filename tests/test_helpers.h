#pragma once

#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <cbls/arg_parse.h>
#include <cbls/model.h>
#include <cbls/search.h>
#include <cstdio>
#include <cstdlib>
#include <string>

// Short alias for tests — delegates to the canonical core function.
inline int32_t vid(int32_t handle) {
    return cbls::handle_to_var_id(handle);
}

// ---------------------------------------------------------------------------
// Deterministic solve for tests.
//
// A wall-clock budget makes a test's outcome depend on machine speed (how many
// batches ran), so tests bound the search by GLS *iterations* (`time_limit = 0`
// disables the wall clock): a given seed always gives exactly the same result.
// A test that keeps a wall clock asserts timing itself (e.g. "SA returns
// result" checks time_seconds > 0) and is commented as such.
//
// The call sites' iteration budgets were calibrated against the wall-clock
// budgets they replaced. Raise a budget if a quality assertion needs more
// search; never swap it back for a time limit.
// ---------------------------------------------------------------------------
inline cbls::SearchResult solve_deterministic(cbls::Model& model, int64_t max_iterations,
                                              uint64_t seed = 42,
                                              cbls::InnerSolverHook* hook = nullptr,
                                              cbls::LNS* lns = nullptr, int lns_interval = 3,
                                              cbls::SolveCallback* callback = nullptr,
                                              const cbls::SearchConfig& base = {}) {
    cbls::SearchConfig cfg = base;  // a test's protocol (e.g. uc-chped's); budget set here
    cfg.max_iterations = max_iterations;
    double time_limit = 0.0;  // 0 = no wall clock; the iteration budget binds

    // Recalibration escape hatch: CBLS_TEST_CALIBRATE=<seconds> runs each solve
    // under a wall clock instead and reports the iterations achieved, so the
    // budgets below can be re-derived when the engine's throughput changes.
    // Deliberately opt-in: with it set, tests are NOT deterministic.
    const char* calib = std::getenv("CBLS_TEST_CALIBRATE");
    if (calib != nullptr) {
        // cbls::try_parse_double, the one rule the CLI and the benchmark
        // runners parse their flags with: std::atof would read a typo as 0.0,
        // "no wall clock" (bugprone-unchecked-string-to-number-conversion), and
        // trailing characters are refused, so "5s" is not read as 5 seconds.
        // A failed parse resets time_limit to 0, so the guard below refuses.
        if (!cbls::try_parse_double(calib, time_limit)) {
            time_limit = 0.0;
        }
        if (!(time_limit > 0.0)) {
            std::fprintf(stderr,
                         "CBLS_TEST_CALIBRATE=\"%s\" is not a positive number of seconds; "
                         "every solve would return immediately. Refusing to run.\n",
                         calib);
            std::abort();
        }
        std::fprintf(stderr,
                     "WARNING: CBLS_TEST_CALIBRATE is set - this run is NOT deterministic "
                     "and its pass/fail result is meaningless.\n");
        cfg.max_iterations = 0;
    }

    cbls::SearchResult r = cbls::solve(model, time_limit, seed, /*use_fj=*/true, hook, lns,
                                       lns_interval, callback, cfg);
    if (calib != nullptr) {
        std::fprintf(stderr, "CALIB\t%s\t%lld\t%.2f\n",
                     Catch::getResultCapture().getCurrentTestName().c_str(),
                     static_cast<long long>(r.iterations), r.time_seconds);
    }
    return r;
}
