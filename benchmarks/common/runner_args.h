#pragma once

#include <cbls/arg_parse.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>

namespace cbls::bench {

// The runners' flag-value reporting policy over `cbls::try_parse_*`: stderr and
// `std::exit`, so it stays out of `include/cbls/`; pinned by
// tests/python/test_run_benchmark.py. A bad double returns NaN, which the
// runner's own guard reports with exit 2; a bad integer has no such guard, so
// parse_int64 exits 2 itself, as for an unknown option.

inline double parse_double(const char* flag, const std::string& text) {
    double value = 0.0;
    if (!try_parse_double(text, value)) {
        std::fprintf(stderr, "%s: '%s' is not a number\n", flag, text.c_str());
        return std::numeric_limits<double>::quiet_NaN();
    }
    return value;
}

// A value flag matches only when a value follows it: a trailing `--budget` must
// fall through to the runner's unknown-argument path (exit 2), not read past
// argv or silently keep the default a published table would then carry.
// Consuming the value is part of matching it:
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

    /// True when the argument is `flag` AND a value follows; the value is then
    /// consumed. A trailing `flag` does not match, so the caller rejects it.
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

/// Whether two paths name the same file, for the published-table guard, which
/// must compare the file, not which flags were typed (the documented command
/// spells `--out <published table>`). A path that does not exist yet cannot be
/// a published table, so the lexical fallback is safe.
inline bool same_file(const std::string& a, const std::string& b) {
    // Inode first: a hardlink is the same file under a name no path compare
    // (even `weakly_canonical`) equates. The path compare covers new files.
    std::error_code ec;
    if (std::filesystem::equivalent(a, b, ec)) {
        return true;
    }
    const std::filesystem::path pa = std::filesystem::weakly_canonical(a, ec);
    if (ec) {
        return a == b;
    }
    const std::filesystem::path pb = std::filesystem::weakly_canonical(b, ec);
    if (ec) {
        return a == b;
    }
    return pa == pb;
}

inline int64_t parse_int64(const char* flag, const std::string& text) {
    int64_t value = 0;
    if (!try_parse_int64(text, value)) {
        std::fprintf(stderr, "%s: '%s' is not an integer\n", flag, text.c_str());
        std::exit(2);
    }
    return value;
}

/// Upper bound on `--threads`, a memory limit: each worker owns its variables,
/// values and search tables (#157), 0.34-0.39 GiB on MIPfeas's
/// `neos-5114902-kasavu` at `06eb3e5`. A typo'd 1000 would be an OOM kill.
inline constexpr int kMaxThreads = 256;

/// `--threads N`; exits 2 outside [1, kMaxThreads]. 0 is not "one per core":
/// a machine-dependent default makes runs incomparable. `mipfeas` and `cvrp`
/// record the count per row; `minlplib` and `uc-chped` in `search_config` off
/// 1, and refuse a count above 1 onto their published tables.
inline int parse_thread_count(const std::string& text) {
    const int64_t threads = parse_int64("--threads", text);
    if (threads < 1 || threads > kMaxThreads) {
        std::fprintf(stderr, "--threads must be in [1, %d]\n", kMaxThreads);
        std::exit(2);
    }
    return static_cast<int>(threads);
}

}  // namespace cbls::bench
