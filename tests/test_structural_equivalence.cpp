// Frozen-trajectory guard for the STRUCTURAL batch (#165).
//
// #165 replaced `structural_pass`'s hard-coded `generate_standard_moves` sweep
// with a registered `MoveGenerator` pipeline, a `StructuralSelection` policy and
// G_v-restricted candidate scoring. The acceptance criterion attached to that
// refactor is that the DEFAULT configuration -- no generator registered,
// `StructuralSelection::FirstImprovingSample` -- keeps the trajectory it had
// before, bit for bit, at one thread on a given seed.
//
// "Bit for bit" is not checkable by eyeballing an objective, because a
// structural search reaches many assignments of equal cost. So each scenario
// below is digested over the WHOLE returned state -- every scalar value's bit
// pattern and every element of every List/Set -- plus the iteration count, the
// termination reason and the objective's bits. Any change in the number or the
// order of RNG draws, in the order variables are swept, or in which candidates
// are accepted moves the digest.
//
// THE DIGESTS ARE A STORED BASELINE. They were recorded by running this file
// against the engine at a805cb6 -- the commit immediately before the generator
// API landed -- and they are not to be re-derived from the current build to
// turn a red test green. A mismatch means the refactor changed the search; that
// is the finding, not the test's problem.
//
// Every scenario is iteration-budgeted with `time_limit = 0`, so nothing here
// reads a clock and the run is fully deterministic (see the `time_limit <= 0`
// contract on `solve`).
//
// WHAT ELSE A MISMATCH CAN MEAN. The digests pin a trajectory, but a trajectory
// is a function of the toolchain as well as of the code. `RNG` is built on
// `std::uniform_real_distribution` / `std::normal_distribution` / `std::shuffle`,
// none of which is specified to produce the same sequence across standard
// library implementations, and `normal` reaches into libm. These were recorded
// with the repository's own build -- **GCC/libstdc++ on x86-64 at Release**,
// which is what `CMakeLists.txt` defaults to and what every gate builds. So on
// a different compiler or standard library, or in a `build/` that was first
// configured `Debug` or with `CBLS_SANITIZE` (CLAUDE.md warns that a build
// directory keeps whatever type it was first given, and that pre-commit gates
// on it), a mismatch is the TOOLCHAIN and not the engine. Check
// `CMAKE_BUILD_TYPE` in `build/CMakeCache.txt` before believing one.
//
// If the toolchain really has moved and the table has to be re-recorded, record
// it the way it was recorded the first time: build `a805cb6` in a scratch
// checkout on the NEW toolchain, run this file there, and copy the digests it
// prints. Never re-derive them from the current build -- that turns the guard
// into a tautology, which is the one failure mode it cannot survive.

#include "test_helpers.h"

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/search.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace cbls;

namespace {

// FNV-1a over the raw bytes, so two doubles that print alike but differ in the
// last ulp digest differently -- which is the point.
class Digest {
public:
    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const unsigned char*>(p);
        for (size_t i = 0; i < n; ++i) {
            h_ ^= b[i];
            h_ *= 1099511628211ULL;
        }
    }
    void add(double v) { bytes(&v, sizeof(v)); }
    void add(int64_t v) { bytes(&v, sizeof(v)); }
    void add(int32_t v) { bytes(&v, sizeof(v)); }
    [[nodiscard]] uint64_t value() const { return h_; }

private:
    uint64_t h_ = 14695981039346656037ULL;
};

uint64_t digest_run(const SearchResult& r) {
    Digest d;
    d.add(r.objective);
    d.add(static_cast<int32_t>(r.feasible ? 1 : 0));
    d.add(r.iterations);
    d.add(static_cast<int32_t>(r.termination));
    d.add(static_cast<int32_t>(r.best_state.values.size()));
    for (double v : r.best_state.values) {
        d.add(v);
    }
    for (const std::vector<int32_t>& elems : r.best_state.elements) {
        d.add(static_cast<int32_t>(elems.size()));
        for (int32_t e : elems) {
            d.add(e);
        }
    }
    return d.value();
}

// ---- the scenarios -------------------------------------------------------
//
// Deterministic pseudo-data, generated from a splitmix-style mixer rather than
// an RNG object, so the models are identical whatever the engine's RNG does.
Expr node_expr(Model& m, int32_t node_id) {
    return {&m, node_id};
}

uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// One Set variable over a 40-element universe, 12 coverage rows and a weighted
// objective: the shape benchmarks/setcover uses, small enough to run in
// milliseconds.
Model set_cover_model() {
    static std::vector<std::vector<int>> covers;  // row -> columns covering it
    static std::vector<double> cost;
    covers.assign(12, {});
    cost.assign(40, 0.0);
    for (int j = 0; j < 40; ++j) {
        cost[static_cast<size_t>(j)] =
            1.0 + static_cast<double>(mix(static_cast<uint64_t>(j)) % 20);
    }
    for (int r = 0; r < 12; ++r) {
        for (int j = 0; j < 40; ++j) {
            if (mix((static_cast<uint64_t>(r) * 131) + static_cast<uint64_t>(j)) % 5 == 0) {
                covers[static_cast<size_t>(r)].push_back(j);
            }
        }
        if (covers[static_cast<size_t>(r)].empty()) {
            covers[static_cast<size_t>(r)].push_back(r);
        }
    }

    Model m;
    Expr chosen = m.Set(40, 1, 12, "chosen");
    for (int r = 0; r < 12; ++r) {
        const std::vector<int>* row = &covers[static_cast<size_t>(r)];
        Expr covered = node_expr(m, m.lambda_sum(chosen.handle, [row](int e) {
            for (int j : *row) {
                if (j == e) {
                    return 1.0;
                }
            }
            return 0.0;
        }));
        m.add_constraint(covered >= m.Constant(1.0));
    }
    const std::vector<double>* c = &cost;
    m.minimize(m.lambda_sum(chosen.handle, [c](int e) {
        return (e >= 0 && e < 40) ? (*c)[static_cast<size_t>(e)] : 0.0;
    }));
    m.close();
    return m;
}

// One List variable read as a tour: a pair_lambda_sum objective and one
// positional (`at`) constraint, so both List readers are exercised.
Model list_tour_model() {
    static std::vector<double> xs;
    static std::vector<double> ys;
    xs.assign(14, 0.0);
    ys.assign(14, 0.0);
    for (int i = 0; i < 14; ++i) {
        xs[static_cast<size_t>(i)] = static_cast<double>(mix(7 * static_cast<uint64_t>(i)) % 100);
        ys[static_cast<size_t>(i)] =
            static_cast<double>(mix((11 * static_cast<uint64_t>(i)) + 3) % 100);
    }
    Model m;
    Expr tour = m.List(14, "tour");
    const std::vector<double>* px = &xs;
    const std::vector<double>* py = &ys;
    m.minimize(m.pair_lambda_sum(tour.handle, [px, py](int a, int b) {
        const double dx = (*px)[static_cast<size_t>(a)] - (*px)[static_cast<size_t>(b)];
        const double dy = (*py)[static_cast<size_t>(a)] - (*py)[static_cast<size_t>(b)];
        return std::sqrt((dx * dx) + (dy * dy));
    }));
    m.add_constraint(node_expr(m, m.at(tour.handle, m.Constant(0.0).handle)) <= m.Constant(3.0));
    m.close();
    return m;
}

// List + Set + scalars in one model, so the sweep visits several structured
// variables per pass and interleaves with Feasibility Jump batches.
Model mixed_model() {
    Model m;
    Expr route = m.List(9, "route");
    Expr pick = m.Set(16, 2, 6, "pick");
    Expr a = m.Int(0, 8, "a");
    Expr b = m.Int(0, 8, "b");
    Expr t = m.Float(0.0, 5.0, "t");

    m.add_constraint(node_expr(m, m.at(route.handle, m.Constant(0.0).handle)) + a >=
                     m.Constant(6.0));
    m.add_constraint(node_expr(m, m.at(route.handle, m.Constant(4.0).handle)) - b <=
                     m.Constant(2.0));
    Expr weight = node_expr(
        m, m.lambda_sum(pick.handle, [](int e) { return 1.0 + (0.25 * static_cast<double>(e)); }));
    m.add_constraint(weight <= m.Constant(9.0));
    m.add_constraint(weight + t >= m.Constant(4.0));
    m.minimize(weight + a + b + t + node_expr(m, m.pair_lambda_sum(route.handle, [](int x, int y) {
                   return static_cast<double>((x - y) * (x - y)) * 0.1;
               })));
    m.close();
    return m;
}

struct Scenario {
    const char* name;
    Model (*build)();
    uint64_t seed;
    int64_t max_iterations;
    double structural_probability;  // <0 = engine default (0.33 here)
    uint64_t expected;
};

// Recorded at a805cb6. See the header comment: do not regenerate.
const std::array<Scenario, 6> kScenarios = {{
    {"set_cover/seed42/auto", set_cover_model, 42, 4000, -1.0, 0x2afb90b15172f4d2ULL},
    {"set_cover/seed7/struct1", set_cover_model, 7, 4000, 1.0, 0x77c1525046818defULL},
    {"list_tour/seed42/auto", list_tour_model, 42, 4000, -1.0, 0xa25f848877a8ad48ULL},
    {"list_tour/seed7/struct1", list_tour_model, 7, 4000, 1.0, 0x24b5d55b1947f7c1ULL},
    {"mixed/seed42/auto", mixed_model, 42, 4000, -1.0, 0x20189d0215c20c7aULL},
    {"mixed/seed7/struct1", mixed_model, 7, 4000, 1.0, 0x7ff39ed3c0d80834ULL},
}};

uint64_t run_scenario(const Scenario& sc) {
    Model m = sc.build();
    SearchConfig config;
    config.max_iterations = sc.max_iterations;
    config.structural_batch_probability = sc.structural_probability;
    SearchResult r = solve(m, /*time_limit=*/0.0, sc.seed, /*use_fj=*/true, /*hook=*/nullptr,
                           /*lns=*/nullptr, /*lns_interval=*/3, /*callback=*/nullptr, config);
    return digest_run(r);
}

}  // namespace

TEST_CASE("default structural batch keeps its pre-generator trajectory", "[structural][moves]") {
    for (const Scenario& sc : kScenarios) {
        INFO("scenario " << sc.name);
        const uint64_t actual = run_scenario(sc);
        // Reported in hex so a mismatch is quotable; see the file header before
        // touching the table.
        INFO("actual digest 0x" << std::hex << actual);
        REQUIRE(actual == sc.expected);
    }
}

TEST_CASE("structural trajectory is reproducible within a build", "[structural][moves]") {
    // Guards the guard: if a scenario were nondeterministic, the table above
    // would be noise rather than a baseline.
    for (const Scenario& sc : kScenarios) {
        INFO("scenario " << sc.name);
        REQUIRE(run_scenario(sc) == run_scenario(sc));
    }
}
