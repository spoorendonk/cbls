#pragma once

#include <cbls/arg_parse.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>

namespace cbls::bench {

// Flag-value parsing for the benchmark runners: `cbls::try_parse_*` supplies the
// rule, this header supplies the runners' reporting policy. It lives on the
// benchmark side rather than in `include/cbls/` precisely because it is policy
// -- writing to stderr and calling `std::exit` -- and because the drivers' tests
// pin it (`tests/python/test_run_benchmark.py`), so it belongs next to the
// programs those tests run.
//
// A bad double yields NaN rather than exiting here, deliberately: every double
// flag already has a `!(x > 0.0)` guard in the runner that reports it and exits
// 2, and NaN fails that guard, so the parse layer adds a diagnostic without
// moving where the failure is reported or changing the runner's exit code.
// Integer flags have no such guard, so those report and exit 2 directly --
// which is what parse_args already does for an unknown option.

inline double parse_double(const char* flag, const std::string& text) {
    double value = 0.0;
    if (!try_parse_double(text, value)) {
        std::fprintf(stderr, "%s: '%s' is not a number\n", flag, text.c_str());
        return std::numeric_limits<double>::quiet_NaN();
    }
    return value;
}

inline int64_t parse_int64(const char* flag, const std::string& text) {
    int64_t value = 0;
    if (!try_parse_int64(text, value)) {
        std::fprintf(stderr, "%s: '%s' is not an integer\n", flag, text.c_str());
        std::exit(2);
    }
    return value;
}

}  // namespace cbls::bench
