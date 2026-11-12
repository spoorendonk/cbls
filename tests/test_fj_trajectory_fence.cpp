// Trajectory fence for FeasibilityJump's scan-set bookkeeping (#174).
//
// The ORDER in which variables enter FJ's scan set -- after a weight bump, when
// seeding a Novelty jump, and in update_var's participation test -- feeds
// apply_jump's random draw, so a wrong order changes the search while leaving
// every consistency check in tests/test_fj_violated_set.cpp green. The bump's
// requeue and the Novelty seed visit the violated-row list V in its list order
// (appends plus swap-removes: a deterministic function of the flip history, not
// ascending), so a swap-remove that mis-places a row moves the hashes. Like
// tests/test_structured_trajectory.cpp, this is a REGRESSION FENCE and not a
// quality assertion. A change that moves an FJ trajectory on purpose re-records
// them and says why here and in its commit.
//
// Provenance: first recorded at engine commit c19c982, the last before #174;
// #174 reproduced both unchanged (bit-identical, by sorting V back into
// ascending row order before every scan that queues variables). Re-recorded on
// purpose since, so neither reproduces against c19c982 any more:
// - #175, batch-API only (run() reproduced): lazy GLS decay (LazyWeightDecay)
//   agrees with the eager decay to rounding, not to the bit. Then both, in its
//   review round, when the ascending sort went (with bit-identity given up it
//   only made each bump O(|V| log |V|)): V's list order changes the order vars
//   enter Q and Q', and with it apply_jump's and select_novelty_var's draws.
// - #206, both: apply_jump samples until it holds sample_size positive-score
//   variables or Q has none, keeping the sample distinct by drawing from Q's
//   unsampled suffix and swapping each positive into the prefix, so the same
//   RNG value indexes a different variable, Q's order changes after every
//   call, and a call that returned false (bump) on a Q still holding an
//   improving variable now applies it. The two-phase hash moved for that
//   alone (run() never calls Novelty); the batch-API hash also because
//   select_novelty_var counts only filter-passing variables toward its 3,
//   draws from the unsampled suffix of Q' and has no 32-draw cap. Its review
//   round moved the batch-API hash only, twice: Novelty swap-removes a var
//   that fails F or is on the stack from Q' for good (as OR-Tools'
//   ScanRelevantVariables does) and re-queues a moved or undone var's
//   row-neighbours, so Q' holds different vars, in a different order, at
//   every draw; then the sibling loop re-checks the discrepancy budget and
//   work cap before every sibling, not on entry only, since the re-queue
//   means Q' no longer runs dry on its own (the fence model reaches levels
//   whose budget is spent).
// - #209, batch-API only: the fence drives a Novelty BATCH (novelty_batch, GLS
//   over ApplyNoveltyJump, which bumps weights at a Novelty local minimum and
//   needs no resync) where it drove one apply_novelty_jump() and a resync,
//   since that is what the search runs; Q' is no longer reseeded after each
//   commit or discrepancy level, W' is reset only on the rows that moved, and
//   select_novelty_var ranks its sample by novelty score rather than W score.
//   It moved twice more inside #209: Q' admitting only vars in an active
//   violated row with the batch sized at a third of its iterations in moves,
//   and a Novelty move charging 3 GLS iterations and a bump 1 (the hash reads
//   iterations()).
// - #211, both: sample_size_general 3 -> 5 (ViolationLS §3 samples 5 for a
//   Feasibility Jump), read by every batch() and run()'s general phase. The
//   two-phase hash also because run() carries the linear phase's learned
//   weights into the general phase, unmasking only the non-linear rows at 1, so
//   phase-1 weights reach the hash. FJ's tolerance moving from 1e-9 to 1e-6
//   touches neither: both fence models are integral, so every residual is an
//   integer.
// - #272 added the third, non-linear Novelty hash, recorded at engine 805cdb2
//   before #272 scored each Novelty candidate under W' and W in one pass; #272
//   reproduced all three unchanged.

#include "libstdcxx_rng.h"
#include "test_models.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <vector>

using namespace cbls;
using cbls_test::bits_of;
using cbls_test::build_random_int_model;

namespace {

uint64_t mix(uint64_t h, uint64_t x) {
    return h ^ (x + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
}

uint64_t hash_state(uint64_t h, const Model& m, const ViolationManager& vm) {
    for (size_t v = 0; v < m.num_vars(); ++v) {
        h = mix(h, bits_of(m.var(static_cast<int32_t>(v)).value));
    }
    for (const double w : vm.weights) {
        h = mix(h, bits_of(w));
    }
    return h;
}

// Batches with GLS bumps at both rho values, a Novelty batch every third batch
// and a kick every tenth, hashing the assignment and the weights after each.
uint64_t batch_api_trajectory() {
    Model m;
    build_random_int_model(m, 101, 60, 130, false);
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
            h = mix(h, fj.novelty_batch(200) ? 1U : 0U);
            h = hash_state(h, m, vm);
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
    build_random_int_model(m, 202, 40, 90, true);
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

// Novelty batches over a model whose every fifth row is non-linear and weighted
// (single phase), so select_novelty_var scores those rows' variables by the DAG
// probe -- the path the batch-API fence, all-linear, never takes.
uint64_t nonlinear_novelty_trajectory() {
    Model m;
    build_random_int_model(m, 303, 40, 90, true);
    m.close();
    ViolationManager vm(m);
    RNG rng(31);
    GFJConfig cfg;
    cfg.two_phase = false;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(true);
    uint64_t h = 0;
    for (int b = 0; b < 40; ++b) {
        fj.set_rho(b % 2 == 0 ? 0.95 : 1.0);
        h = mix(h, fj.batch(100) ? 1U : 0U);
        h = mix(h, fj.novelty_batch(200) ? 1U : 0U);
        h = mix(h, static_cast<uint64_t>(fj.iterations()));
        h = hash_state(h, m, vm);
        if (b % 10 == 9) {
            fj.perturb(0.1);
        }
    }
    // The fence is about the probe path: some prepare must have fallen back.
    REQUIRE(fj.linear_scorer().fallback_prepares() > 0);
    return h;
}

}  // namespace

TEST_CASE("FJ's batch-API trajectory matches its recorded fingerprint",
          "[fj][violated_set][trajectory]") {
    skip_unless_libstdcxx_rng();
    const uint64_t h = batch_api_trajectory();
    CAPTURE(h);
    REQUIRE(h == 0xe63589f3625faca1ULL);
}

TEST_CASE("FJ's two-phase run() trajectory matches its recorded fingerprint",
          "[fj][violated_set][trajectory]") {
    skip_unless_libstdcxx_rng();
    const uint64_t h = two_phase_run_trajectory();
    CAPTURE(h);
    REQUIRE(h == 0xfefd9a2f749d666bULL);
}

TEST_CASE("FJ's Novelty trajectory over non-linear rows matches its recorded fingerprint",
          "[fj][violated_set][trajectory]") {
    skip_unless_libstdcxx_rng();
    const uint64_t h = nonlinear_novelty_trajectory();
    CAPTURE(h);
    REQUIRE(h == 0x773677ae558b7b7eULL);
}
