#pragma once

#include <cbls/model.h>
#include <cbls/search.h>

#include <catch2/interfaces/catch_interfaces_capture.hpp>

#include <cstdio>
#include <cstdlib>

// Short alias for tests — delegates to the canonical core function.
inline int32_t vid(int32_t handle) { return cbls::handle_to_var_id(handle); }

// ---------------------------------------------------------------------------
// Deterministic solve for tests.
//
// A wall-clock budget makes a test's outcome depend on machine speed: the same
// assertion can pass on a fast machine and fail on a loaded CI box, because a
// different number of batches ran. Tests therefore bound the search by GLS
// *iterations* instead (`time_limit = 0` disables the wall clock entirely), so a
// given seed always produces exactly the same result.
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
        time_limit = std::atof(calib);
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
