// Trajectory fence for FeasibilityJump's scan-set bookkeeping (#174).
//
// #174 replaced the whole-row sweeps that decide which variables enter FJ's scan
// set -- after a weight bump, when seeding a Novelty jump, and in update_var's
// participation test -- with an incrementally maintained violated-row list. The
// ORDER in which variables enter the scan set feeds apply_jump's random draw, so
// a list visited in the wrong order changes the search while leaving every
// consistency check in tests/test_fj_violated_set.cpp green. This pins it.
//
// The hashes were first recorded at engine commit c19c982, the last commit
// before #174, and #174 reproduced both unchanged: its bookkeeping was
// bit-identical, which it bought by sorting V back into ascending row order
// before every scan that queues variables. Like
// tests/test_structured_trajectory.cpp, this is a REGRESSION FENCE and not a
// quality assertion. A change that moves an FJ trajectory on purpose re-records
// them and says why here and in its commit.
//
// #175 moved the batch-API hash on purpose: the GLS decay became lazy (a global
// scale, LazyWeightDecay), which agrees with the eager decay to rounding, not to
// the bit, and keeps cached jump scores in the same scaled space as fresh ones.
// The two-phase run() hash reproduced unchanged then.
//
// Both hashes were re-recorded once more when the ascending-order sort went
// (the #175 review round): with bit-identity already given up, the sort only
// made each bump O(|V| log |V|). The bump's requeue and the Novelty seed now
// visit V in its list order -- appends plus swap-removes, a deterministic
// function of the flip history, but not ascending -- so the order variables
// enter Q and the Novelty scan set changes, and with it apply_jump's and
// select_novelty_var's draws. That order is what this fence pins now: a
// swap-remove that mis-places a row still moves both hashes. Neither can be
// reproduced against c19c982 any more.
//
// The two-phase hash sees only the final assignment and weights, and run()
// refills every weight to 1 before its general phase, so phase-1 weights never
// reach it directly -- only through the assignment phase 1 ends on.

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using namespace cbls;

namespace {

uint64_t mix(uint64_t h, uint64_t x) {
    return h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

uint64_t bits(double d) {
    uint64_t u = 0;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

// A random sparse integer model, more rows than can hold at once. With
// `nonlinear`, every fifth row multiplies two of its variables, so run()'s
// two-phase path masks those rows out of the first phase.
void build_fence_model(Model& m, uint32_t seed, int num_vars, int num_rows, bool nonlinear) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> pick_var(0, num_vars - 1);
    std::uniform_int_distribution<int> pick_coef(1, 3);
    std::uniform_int_distribution<int> pick_sign(0, 1);
    std::uniform_int_distribution<int> pick_rhs(-2, 6);
    std::vector<int32_t> handles;
    handles.reserve(static_cast<size_t>(num_vars));
    for (int i = 0; i < num_vars; ++i) {
        handles.push_back(m.int_var(0, 4));
    }
    for (int r = 0; r < num_rows; ++r) {
        std::vector<int> used;
        while (used.size() < 4) {
            const int v = pick_var(gen);
            if (std::find(used.begin(), used.end(), v) == used.end()) {
                used.push_back(v);
            }
        }
        std::vector<int32_t> terms;
        for (const int v : used) {
            const double coef = (pick_sign(gen) != 0 ? 1.0 : -1.0) * pick_coef(gen);
            terms.push_back(m.prod(m.constant(coef), handles[static_cast<size_t>(v)]));
        }
        if (nonlinear && r % 5 == 0) {
            terms.push_back(m.prod(handles[static_cast<size_t>(used[0])],
                                   handles[static_cast<size_t>(used[1])]));
        }
        const int32_t sum = m.sum(terms);
        const double rhs = pick_rhs(gen);
        m.add_constraint(r % 2 == 0 ? m.leq(sum, m.constant(rhs)) : m.geq(sum, m.constant(rhs)));
    }
}

uint64_t hash_state(uint64_t h, const Model& m, const ViolationManager& vm) {
    for (size_t v = 0; v < m.num_vars(); ++v) {
        h = mix(h, bits(m.var(static_cast<int32_t>(v)).value));
    }
    for (const double w : vm.weights) {
        h = mix(h, bits(w));
    }
    return h;
}

// Batches with GLS bumps at both rho values, a Novelty jump every third batch
// and a kick every tenth, hashing the assignment and the weights after each.
uint64_t batch_api_trajectory() {
    Model m;
    build_fence_model(m, 101, 60, 130, false);
    m.close();
    ViolationManager vm(m);
    RNG rng(17);
    GFJConfig cfg;
    cfg.two_phase = false;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(true);
    uint64_t h = 0;
    for (int b = 0; b < 80; ++b) {
        fj.set_rho(b % 2 == 0 ? 0.95 : 1.0);
        h = mix(h, fj.batch(200) ? 1U : 0U);
        h = mix(h, static_cast<uint64_t>(fj.iterations()));
        h = hash_state(h, m, vm);
        if (b % 3 == 2) {
            h = mix(h, fj.apply_novelty_jump() ? 1U : 0U);
            h = hash_state(h, m, vm);
            fj.resync();
        }
        if (b % 10 == 9) {
            fj.perturb(0.1);
        }
    }
    return h;
}

// The single-shot run(), two-phase: the linear phase masks the non-linear rows
// to weight 0, the general phase unmasks them.
uint64_t two_phase_run_trajectory() {
    Model m;
    build_fence_model(m, 202, 40, 90, true);
    m.close();
    ViolationManager vm(m);
    RNG rng(29);
    GFJConfig cfg;
    cfg.max_iterations = 6000;
    FeasibilityJump fj(m, vm, rng, cfg);
    const GFJStatus status = fj.run();
    uint64_t h = mix(0, status == GFJStatus::Feasible ? 1U : 0U);
    h = mix(h, static_cast<uint64_t>(fj.iterations()));
    return hash_state(h, m, vm);
}

}  // namespace

TEST_CASE("FJ's batch-API trajectory matches its recorded fingerprint",
          "[fj][violated_set][trajectory]") {
    const uint64_t h = batch_api_trajectory();
    CAPTURE(h);
    REQUIRE(h == 0xf79597c62086247bULL);
}

TEST_CASE("FJ's two-phase run() trajectory matches its recorded fingerprint",
          "[fj][violated_set][trajectory]") {
    const uint64_t h = two_phase_run_trajectory();
    CAPTURE(h);
    REQUIRE(h == 0xac3e4c76b0aa79ecULL);
}
