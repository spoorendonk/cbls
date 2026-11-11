#pragma once

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/resource.h>
#include <system_error>

namespace cbls::bench {

// File and process helpers the benchmark runners share.

inline bool file_exists(const std::string& path) {
    const std::ifstream f(path);
    return f.good();
}

/// Seconds elapsed since `start`, on the monotonic clock.
inline double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// Peak resident set of this process so far, in KiB; 0 when getrusage fails.
inline long peak_rss_kib() {
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return usage.ru_maxrss;
}

/// Renames `tmp` onto `path`. On failure reports it and returns false,
/// leaving `tmp` for the caller to remove or keep.
inline bool rename_into_place(const std::string& tmp, const std::string& path) {
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::fprintf(stderr, "Failed to rename %s -> %s: %s\n", tmp.c_str(), path.c_str(),
                     ec.message().c_str());
        return false;
    }
    return true;
}

/// Writes `text` to `path` through `<path>.tmp` and a rename, so a killed job
/// leaves the previous file or none, never half of one. Reports and returns
/// false on failure.
inline bool write_atomically(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", tmp.c_str());
            return false;
        }
        out << text;
        out.flush();
        if (!out.good()) {
            std::fprintf(stderr, "Failed to write %s\n", tmp.c_str());
            return false;
        }
    }
    return rename_into_place(tmp, path);
}

}  // namespace cbls::bench
