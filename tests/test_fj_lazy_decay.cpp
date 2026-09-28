// The GLS weight decay kept lazily behind a global scale (#175).
//
// LazyWeightDecay stores w' = w / s and decays by moving s, so a decay is O(1)
// and a bump touches only the violated rows. The algorithm's weights are
// unchanged; these tests hold its EFFECTIVE weights s * w' against the eager
// update, `w <- rho * w; w += 1 on the active violated rows`
// (gls_update_weights), over long bump/decay sequences that cross the
// renormalisation bound several times and include masked rows.
//
// Tolerance: relative 1e-9. Neither side is exact -- the eager form rounds every
// weight twice per step, the lazy one rounds the scale once per step and each
// bumped weight once -- and over the ~12 000 steps here either accumulates at
// most ~1e4 roundings of 2^-53, about 1e-12 relative. 1e-9 is three orders
// above that and over seven below any scale error, since a neglected scale is off by
// a factor of rho^k. A masked (0) weight must be EXACTLY 0 on both sides, and a
// positive one positive.

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

using namespace cbls;

namespace {

constexpr double kRelTol = 1e-9;

bool close_rel(double got, double want) {
    return std::abs(got - want) <= kRelTol * std::abs(want);
}

// The eager reference and the lazy representation, driven in lockstep.
struct LockstepWeights {
    std::vector<double> eager;
    std::vector<double> stored;
    LazyWeightDecay lazy;
    int folds = 0;
    int materialisations = 0;
    bool saw_scaled = false;  // the stored vector was NOT the effective weights

    explicit LockstepWeights(size_t rows) : eager(rows, 1.0), stored(rows, 1.0) {}

    void set(size_t c, double w) {  // only 0 is scale-free; others need s = 1
        eager[c] = w;
        stored[c] = w;
    }

    // One GLS update: decay by rho, then bump the active violated rows.
    void update(const std::vector<uint8_t>& violated, double rho) {
        for (size_t c = 0; c < eager.size(); ++c) {  // gls_update_weights' body
            eager[c] *= rho;
            if (eager[c] > 0.0 && violated[c] != 0) {
                eager[c] += 1.0;
            }
        }
        if (lazy.decay(stored, rho) != 1.0) {
            ++folds;
        }
        for (size_t c = 0; c < stored.size(); ++c) {
            if (violated[c] != 0) {
                lazy.bump(stored, c);
            }
        }
    }

    // A batch boundary: fold the scale in, then do what a caller may do between
    // batches -- mask a live row, unmask a masked one.
    void boundary(int step) {
        lazy.materialise(stored);
        ++materialisations;
        const auto flip = static_cast<size_t>(10 + (step % 30));
        set(flip, stored[flip] > 0.0 ? 0.0 : 1.0);
        const auto unmask = static_cast<size_t>(step % 5);
        if (eager[unmask] == 0.0) {
            set(unmask, 1.0);
        }
    }

    void require_match() {
        for (size_t c = 0; c < eager.size(); ++c) {
            const double eff = lazy.effective(stored, c);
            CAPTURE(c, eff, eager[c], lazy.scale());
            if (eager[c] == 0.0) {
                REQUIRE(eff == 0.0);
                REQUIRE(stored[c] == 0.0);
                continue;
            }
            REQUIRE(eff > 0.0);
            REQUIRE(close_rel(eff, eager[c]));
            saw_scaled = saw_scaled || !close_rel(stored[c], eager[c]);
        }
    }
};

// Mostly the two rho values solve() draws; now and then 0.5, which moves the
// scale 13x faster toward the bound.
double draw_rho(RNG& rng) {
    const double r = rng.random();
    if (r < 0.7) {
        return 0.95;
    }
    return r < 0.995 ? 1.0 : 0.5;
}

}  // namespace

TEST_CASE("LazyWeightDecay's effective weights match the eager update through renormalisation",
          "[fj][gls][lazy_decay]") {
    constexpr size_t kRows = 40;
    constexpr int kSteps = 12000;
    RNG rng(175);
    LockstepWeights w(kRows);
    for (size_t c = 0; c < 5; ++c) {
        w.set(c, 0.0);  // masked from the start
    }
    std::vector<uint8_t> violated(kRows, 0);
    for (int step = 0; step < kSteps; ++step) {
        // Rows 5-9 are violated rarely, so they decay through long stretches
        // (still well inside the normal range); the rest about a third of the time.
        for (size_t c = 0; c < kRows; ++c) {
            violated[c] = rng.random() < (c < 10 ? 0.002 : 0.3) ? 1 : 0;
        }
        const double rho = draw_rho(rng);
        w.update(violated, rho);
        if (step % 2999 == 2998) {
            w.boundary(step);
            REQUIRE(w.lazy.scale() == 1.0);
        } else if (step % 331 == 330) {
            w.set(static_cast<size_t>(10 + (step % 30)), 0.0);  // 0 is 0 in either space
        }
        CAPTURE(step, rho);
        w.require_match();
    }
    // Not vacuous: the scale crossed the bound and was folded several times, and
    // for long stretches the stored vector was far from the effective weights --
    // so a reader that neglected the scale would have been caught above.
    CAPTURE(w.folds, w.materialisations);
    REQUIRE(w.folds >= 3);
    REQUIRE(w.materialisations >= 3);
    REQUIRE(w.saw_scaled);
}

TEST_CASE("LazyWeightDecay keeps 0 exactly 0 and a positive weight positive",
          "[fj][gls][lazy_decay]") {
    SECTION("a fold whose product underflows floors a positive weight") {
        std::vector<double> w = {1e-300, 0.0, 5.0};
        LazyWeightDecay lazy;
        REQUIRE(lazy.decay(w, 1e-20) == 1.0);  // s = 1e-20: inside the bound, lazy
        REQUIRE(w[0] == 1e-300);               // untouched: the decay was O(1)
        // s would be 1e-40 < kMinScale: folded, and 1e-300 * 1e-40 underflows.
        REQUIRE(lazy.decay(w, 1e-20) == 1e-40);
        REQUIRE(lazy.scale() == 1.0);
        REQUIRE(w[0] == std::numeric_limits<double>::denorm_min());
        REQUIRE(w[1] == 0.0);
        REQUIRE(close_rel(w[2], 5e-40));
    }
    SECTION("rho = 0 zeroes every weight, as the eager w * 0 does") {
        std::vector<double> w = {1.0, 0.0, 20.0};
        LazyWeightDecay lazy;
        (void)lazy.decay(w, 0.95);
        REQUIRE(lazy.decay(w, 0.0) == 0.0);
        REQUIRE(w == std::vector<double>{0.0, 0.0, 0.0});
        lazy.bump(w, 0);  // a zeroed row is masked: the bump skips it
        REQUIRE(w[0] == 0.0);
    }
    SECTION("materialise reports the factor it folded, which FJ rescales its scores by") {
        std::vector<double> w = {3.0, 0.0};
        LazyWeightDecay lazy;
        REQUIRE(lazy.materialise(w) == 1.0);  // s = 1: nothing to fold
        (void)lazy.decay(w, 0.5);
        lazy.bump(w, 0);                      // stored 3 + 1/0.5 = 5, effective 2.5
        REQUIRE(lazy.materialise(w) == 0.5);  // folded s = 0.5
        REQUIRE(w == std::vector<double>{2.5, 0.0});
        REQUIRE(lazy.scale() == 1.0);
    }
}

TEST_CASE("FJ's lazily decayed weights match an eager replay of every bump",
          "[fj][gls][lazy_decay]") {
    // x is pinned at 0, so no jump ever improves and EVERY GLS iteration is a
    // bump; V never changes. That makes the eager reference a plain replay:
    // gls_update_weights once per iteration, at the batch's rho. One batch of 3000
    // at rho = 0.95 folds the scale twice inside the loop (every 1347 decays) and
    // once more on the way out; the weights FJ leaves in the ViolationManager
    // must be the effective ones.
    Model m;
    const int32_t x = m.int_var(0, 0);
    m.add_constraint(m.geq(x, m.constant(1.0)));  // r0: violated, active
    m.add_constraint(m.geq(x, m.constant(2.0)));  // r1: violated, active
    m.add_constraint(m.leq(x, m.constant(5.0)));  // r2: satisfied, decays only
    m.add_constraint(m.geq(x, m.constant(3.0)));  // r3: violated, masked below
    m.add_constraint(m.leq(x, m.constant(0.0)));  // r4: satisfied, decays only
    m.close();

    ViolationManager vm(m);
    RNG rng(3);
    GFJConfig cfg;
    cfg.two_phase = false;
    cfg.unproductive_iterations = 0;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(true);
    vm.weights[3] = 0.0;
    vm.invalidate_cache();

    ViolationManager ref(m);
    ref.weights = vm.weights;

    struct Batch {
        double rho;
        int64_t iterations;
    };
    const std::vector<Batch> batches = {{0.95, 3000}, {1.0, 500}, {0.95, 2000}, {0.95, 7}};
    int64_t total = 0;
    for (const Batch& b : batches) {
        fj.set_rho(b.rho);
        REQUIRE_FALSE(fj.batch(b.iterations));
        total += b.iterations;
        REQUIRE(fj.iterations() == total);
        for (int64_t k = 0; k < b.iterations; ++k) {
            gls_update_weights(ref, b.rho);
        }
        for (size_t c = 0; c < vm.weights.size(); ++c) {
            CAPTURE(b.rho, b.iterations, c, vm.weights[c], ref.weights[c]);
            if (ref.weights[c] == 0.0) {
                REQUIRE(vm.weights[c] == 0.0);
            } else {
                REQUIRE(close_rel(vm.weights[c], ref.weights[c]));
            }
        }
    }
    // Not vacuous: the satisfied rows decayed by ~0.95^5000 and the violated ones
    // sit near the 1 / (1 - 0.95) = 20 fixed point.
    REQUIRE(ref.weights[2] < 1e-100);
    REQUIRE(ref.weights[0] > 19.0);
    REQUIRE(ref.weights[3] == 0.0);
}
