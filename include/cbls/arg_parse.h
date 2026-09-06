#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

namespace cbls {

// Strict numeric parsing for command-line flag values.
//
// The parsing rule lives here; the diagnostic and the exit code do not. Those
// differ per program -- the benchmark runners' wording and exit 2 are pinned by
// their drivers' tests, while the CLI has its own `Error: ...`/exit 1
// convention -- so each caller keeps its own policy (see
// `benchmarks/common/runner_args.h` and `src/cli.cpp`) and only the rule is
// shared. That also keeps this header pure: no stream, no `std::exit`, nothing
// a library header has any business doing.
//
// The rule itself: `std::stod` / `std::stoll` rather than `std::atof` /
// `std::atoll`, because the ato* family has no error path at all, so a typo'd
// value silently became 0 and a run that never searched reported like a solver
// result (bugprone-unchecked-string-to-number-conversion). Trailing characters
// are rejected as well, which `std::stod` alone accepts, so `--time-limit 60s`
// cannot quietly mean 60. An out-of-range value fails like a malformed one.
//
// All three return false on failure and leave `out` untouched.

inline bool try_parse_double(const std::string& text, double& out) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        // Throws on empty or non-numeric text, and on overflow.
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        return false;
    }
    if (used != text.size()) {
        return false;  // trailing characters
    }
    out = value;
    return true;
}

inline bool try_parse_int64(const std::string& text, int64_t& out) {
    std::size_t used = 0;
    long long value = 0;
    try {
        // Throws on empty or non-numeric text, and on overflow.
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        return false;
    }
    if (used != text.size()) {
        return false;  // trailing characters
    }
    out = static_cast<int64_t>(value);
    return true;
}

// Unsigned, and separate from try_parse_int64 rather than a cast of it, because
// the two do not accept the same set: std::stoull spans the full 64-bit range
// and wraps a negative value instead of rejecting it. Both halves matter to a
// seed flag, which round-trips through its own printed output -- `--seed -1` is
// recorded as 18446744073709551615, and that has to parse back to the same
// generator state.
inline bool try_parse_uint64(const std::string& text, uint64_t& out) {
    std::size_t used = 0;
    unsigned long long value = 0;
    try {
        // Throws on empty or non-numeric text, and on overflow.
        value = std::stoull(text, &used);
    } catch (const std::exception&) {
        return false;
    }
    if (used != text.size()) {
        return false;  // trailing characters
    }
    out = static_cast<uint64_t>(value);
    return true;
}

}  // namespace cbls
