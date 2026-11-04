#pragma once

#include <catch2/catch_test_macros.hpp>
#include <cstddef>  // any standard header defines __GLIBCXX__ under libstdc++

// Seeded trajectories, fingerprints and seed-built scenarios in some tests were
// recorded with libstdc++. <random>'s distributions (cbls::RNG draws through
// std::uniform_int_distribution and friends) are implementation-defined: libc++
// draws different values from the same engine state, so on libc++ (macOS) the
// pinned values cannot hold. Such a test calls this first and is skipped there
// rather than rewritten: Linux with libstdc++ is the platform that gates, and
// there this is a no-op (#355).
inline void skip_unless_libstdcxx_rng() {
#ifndef __GLIBCXX__
    SKIP(
        "pinned to libstdc++'s <random> distributions; this standard library draws "
        "differently (#355)");
#endif
}
