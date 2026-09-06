#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
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
// All three return false on failure and leave `out` untouched. The `*_status`
// forms additionally say *which* way the text failed, so a caller can tell a
// value that is not a number from one that is a number too large to hold --
// `--threads 99999999999999999999999` is the latter, and reporting it as a typo
// sends the reader hunting for one that is not there. The `try_parse_*` forms
// collapse the distinction, which is what the benchmark runners want: their
// wording and exit codes are pinned by their drivers' tests.

enum class ParseStatus { kOk, kMalformed, kOutOfRange };

inline ParseStatus parse_double_status(const std::string& text, double& out) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        // Throws on empty or non-numeric text, and separately on overflow.
        value = std::stod(text, &used);
    } catch (const std::out_of_range&) {
        return ParseStatus::kOutOfRange;
    } catch (const std::invalid_argument&) {
        return ParseStatus::kMalformed;
    }
    if (used != text.size()) {
        return ParseStatus::kMalformed;  // trailing characters
    }
    out = value;
    return ParseStatus::kOk;
}

inline bool try_parse_double(const std::string& text, double& out) {
    return parse_double_status(text, out) == ParseStatus::kOk;
}

inline ParseStatus parse_int64_status(const std::string& text, int64_t& out) {
    std::size_t used = 0;
    long long value = 0;
    try {
        // Throws on empty or non-numeric text, and separately on overflow.
        value = std::stoll(text, &used);
    } catch (const std::out_of_range&) {
        return ParseStatus::kOutOfRange;
    } catch (const std::invalid_argument&) {
        return ParseStatus::kMalformed;
    }
    if (used != text.size()) {
        return ParseStatus::kMalformed;  // trailing characters
    }
    out = static_cast<int64_t>(value);
    return ParseStatus::kOk;
}

inline bool try_parse_int64(const std::string& text, int64_t& out) {
    return parse_int64_status(text, out) == ParseStatus::kOk;
}

// Unsigned, and separate from try_parse_int64 rather than a cast of it, because
// the two do not accept the same set: std::stoull spans the full 64-bit range
// and wraps a negative value instead of rejecting it. Both halves matter to a
// seed flag, which round-trips through its own printed output -- `--seed -1` is
// recorded as 18446744073709551615, and that has to parse back to the same
// generator state.
inline ParseStatus parse_uint64_status(const std::string& text, uint64_t& out) {
    std::size_t used = 0;
    unsigned long long value = 0;
    try {
        // Throws on empty or non-numeric text, and separately on overflow.
        value = std::stoull(text, &used);
    } catch (const std::out_of_range&) {
        return ParseStatus::kOutOfRange;
    } catch (const std::invalid_argument&) {
        return ParseStatus::kMalformed;
    }
    if (used != text.size()) {
        return ParseStatus::kMalformed;  // trailing characters
    }
    out = static_cast<uint64_t>(value);
    return ParseStatus::kOk;
}

inline bool try_parse_uint64(const std::string& text, uint64_t& out) {
    return parse_uint64_status(text, out) == ParseStatus::kOk;
}

}  // namespace cbls
