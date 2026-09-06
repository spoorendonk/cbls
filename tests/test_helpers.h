#pragma once

#include <catch2/interfaces/catch_interfaces_capture.hpp>
#include <cbls/model.h>
#include <cbls/search.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

// Short alias for tests — delegates to the canonical core function.
inline int32_t vid(int32_t handle) {
    return cbls::handle_to_var_id(handle);
}

// ---------------------------------------------------------------------------
// Deterministic solve for tests.
//
// A wall-clock budget makes a test's outcome depend on machine speed: the same
// assertion can pass on a fast machine and fail on a loaded CI box, because a
// different number of batches ran. Tests therefore bound the search by GLS
// *iterations* instead (`time_limit = 0` disables the wall clock entirely), so a
// given seed always produces exactly the same result.
//
// Not every call site is converted: a few tests deliberately keep a wall clock
// because the thing they assert *is* timing (e.g. "SA returns result" checks
// time_seconds > 0). Those are the exception and are commented as such.
//
// The iteration budgets at the call sites were calibrated against the wall-clock
// budgets they replaced — see tools note in the commit that introduced this.
// Raise a budget if a quality assertion needs more search; never swap it back
// for a time limit.
// ---------------------------------------------------------------------------
inline cbls::SearchResult solve_deterministic(cbls::Model& model, int64_t max_iterations,
                                              uint64_t seed = 42,
                                              cbls::InnerSolverHook* hook = nullptr,
                                              cbls::LNS* lns = nullptr, int lns_interval = 3,
                                              cbls::SolveCallback* callback = nullptr) {
    cbls::SearchConfig cfg;
    cfg.max_iterations = max_iterations;
    double time_limit = 0.0;  // 0 = no wall clock; the iteration budget binds

    // Recalibration escape hatch: CBLS_TEST_CALIBRATE=<seconds> runs each solve
    // under a wall clock instead and reports the iterations achieved, so the
    // budgets below can be re-derived when the engine's throughput changes.
    // Deliberately opt-in: with it set, tests are NOT deterministic.
    const char* calib = std::getenv("CBLS_TEST_CALIBRATE");
    if (calib != nullptr) {
        // std::atof has no error path -- a mistyped value would become 0.0 and
        // read as "no wall clock" (bugprone-unchecked-string-to-number-conversion).
        // std::stod throws instead, which the guard below turns into the message.
        // Trailing characters are rejected too, the same contract the benchmark
        // runners' parse_double got: CBLS_TEST_CALIBRATE="5s" must be refused
        // rather than quietly calibrate against 5 seconds.
        const std::string calib_text(calib);
        try {
            size_t used = 0;
            time_limit = std::stod(calib_text, &used);
            if (used != calib_text.size()) {
                time_limit = 0.0;  // trailing characters; reported by the guard below
            }
        } catch (const std::exception&) {
            time_limit = 0.0;  // reported by the guard below
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
