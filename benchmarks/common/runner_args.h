#pragma once

#include <cbls/arg_parse.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// The other half of the runners' shared argument handling: a flag that takes a
// value matches only when a value actually follows it. Written out per branch as
// `s == "--budget" && i + 1 < argc`, that rule was repeated some two dozen times
// across the runners with nothing naming it, and it is load-bearing in each one:
// `--budget` as the last argument on the command line must fall through to the
// runner's unknown-argument path -- which reports and exits 2 -- rather than
// read past argv's end or silently keep the default. A published results table
// is what a silently-kept default costs.
//
// The cursor drives the loop so that consuming a value is part of matching it:
//
//   ArgCursor c(argc, argv);
//   const char* v = nullptr;
//   while (c.advance()) {
//       const std::string s = c.arg();
//       if (c.value_flag("--budget", v)) { ... }
//       else if (s == "--no-propagate-bounds") { ... }
//   }
class ArgCursor {
public:
    ArgCursor(int argc, char** argv) : argc_(argc), argv_(argv) {}

    /// Moves onto the next argument. False once the command line is exhausted.
    bool advance() { return ++i_ < argc_; }

    /// The argument under the cursor.
    [[nodiscard]] std::string arg() const { return argv_[i_]; }

    /// True when the argument under the cursor is `flag` AND a value follows it,
    /// in which case the value is consumed: the cursor moves onto it, so the
    /// next advance() lands on the argument after. A trailing `flag` with no
    /// value does not match, leaving the caller's chain to reject it.
    bool value_flag(const char* flag, const char*& value) {
        if (i_ + 1 >= argc_ || std::strcmp(argv_[i_], flag) != 0) {
            return false;
        }
        value = argv_[++i_];
        return true;
    }

private:
    int argc_;
    char** argv_;
    int i_ = 0;
};

inline int64_t parse_int64(const char* flag, const std::string& text) {
    int64_t value = 0;
    if (!try_parse_int64(text, value)) {
        std::fprintf(stderr, "%s: '%s' is not an integer\n", flag, text.c_str());
        std::exit(2);
    }
    return value;
}

}  // namespace cbls::bench
