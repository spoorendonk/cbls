// Novelty Jump as a GLS batch with incremental W' and Q' (#209).
//
// Before #209 a Novelty batch was one ApplyNoveltyJump: it reseeded its scan
// set Q' from every violated row after every committed compound move and at
// every discrepancy level, rewrote W' over every row per level, ignored a
// failed search instead of bumping the GLS weights, ranked its sample by the W
// score, and the search redrew FJ-or-Novelty every batch. Each test below pins
// one of those changes against ViolationLS (Davies et al. CPAIOR 2024,
// Algorithms 3-6) and OR-Tools' compound-move mode
// (ortools/sat/feasibility_jump.cc).

#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <random>
#include <vector>

using namespace cbls;

namespace {

// The trajectory fence's random over-constrained integer model: rows that
// cannot all hold, so FJ and Novelty both reach local minima, backtrack, and
// bump.
void build_random_int_model(Model& m, uint32_t seed, int num_vars, int num_rows) {
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
        const int32_t sum = m.sum(terms);
        const double rhs = pick_rhs(gen);
        m.add_constraint(r % 2 == 0 ? m.leq(sum, m.constant(rhs)) : m.geq(sum, m.constant(rhs)));
    }
}

class KindRecorder : public Tracer {
public:
    struct Event {
        enum class Type : uint8_t { Batch, Kick };
        Type type;
        BatchKind kind;
        bool improved;
    };
    std::vector<Event> events;
    void batch_end(BatchKind kind, int64_t /*iterations*/, bool improved) override {
        events.push_back({Event::Type::Batch, kind, improved});
    }
    void kick(KickKind /*kind*/) override {
        events.push_back({Event::Type::Kick, BatchKind::FeasibilityJump, false});
    }
};

}  // namespace

TEST_CASE("a Novelty batch bumps the GLS weights at a Novelty local minimum",
          "[fj][novelty][novelty_batch]") {
    // A: x >= 1 and B: x <= 0 cannot both hold. Flipping x fixes one and breaks
    // the other (score 0), so no compound move ever has a positive score: every
    // ApplyNoveltyJump fails at its largest discrepancy budget. Algorithm 6 runs
    // Novelty under GLS (line 22, Algorithm 3), which answers that failure with
    // the decay-and-bump; before #209 the batch was one ApplyNoveltyJump whose
    // failure was ignored, so the weights never moved.
    Model m;
    const int32_t x = m.bool_var();
    m.add_constraint(m.geq(x, m.constant(1.0)));  // A
    m.add_constraint(m.leq(x, m.constant(0.0)));  // B
    m.close();

    ViolationManager vm(m);
    RNG rng(3);
    GFJConfig cfg;
    cfg.rho = 1.0;  // no decay: a bump is visible as an increment
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/false);
    REQUIRE(vm.weights == std::vector<double>{1.0, 1.0});

    REQUIRE_FALSE(fj.novelty_batch(50));
    CAPTURE(vm.weights, fj.novelty_weight_bumps(), fj.novelty_moves());
    REQUIRE(fj.novelty_moves() > 0);  // the search itself ran
    REQUIRE(fj.novelty_weight_bumps() > 0);
    // Only the violated row is ever bumped, so the weights must have moved off
    // their begin() value of 1.
    REQUIRE((vm.weights[0] > 1.0 || vm.weights[1] > 1.0));
    REQUIRE(fj.scan_sets_consistent());
    REQUIRE(fj.novelty_weights_consistent());
}

TEST_CASE("Novelty ranks its sample by novelty score, not by W score",
          "[fj][novelty][novelty_batch]") {
    // At x = y = 0, A: x + y >= 1 (weight 2) and C: y >= 1 (weight 1) are
    // violated; Bx: x <= 0 (1.5) and By: y <= 0 (2.9) hold, so their novelty
    // weight is W / 1024. Flipping x scores W 2 - 1.5 = 0.5 and novelty
    // 2 - 1.5/1024; flipping y scores W 3 - 2.9 = 0.1 and novelty 3 - 2.9/1024.
    // Q' is {x, y} and both pass F, so the 3-sample holds both on every seed.
    // OR-Tools' ScanRelevantVariables takes the best COMPOUND-weight score, y;
    // ranking by the W score took x. One unit of work lets exactly one move be
    // applied, and either one commits on its own positive score.
    for (uint64_t seed = 1; seed <= 20; ++seed) {
        CAPTURE(seed);
        Model m;
        const int32_t x = m.bool_var();
        const int32_t y = m.bool_var();
        m.add_constraint(m.geq(m.sum({x, y}), m.constant(1.0)));  // A
        m.add_constraint(m.geq(y, m.constant(1.0)));              // C
        m.add_constraint(m.leq(x, m.constant(0.0)));              // Bx
        m.add_constraint(m.leq(y, m.constant(0.0)));              // By
        m.close();

        ViolationManager vm(m);
        RNG rng(seed);
        FeasibilityJump fj(m, vm, rng, GFJConfig{});
        fj.begin(/*set_initial_x=*/false);
        vm.weights = {2.0, 1.0, 1.5, 2.9};
        vm.invalidate_cache();

        REQUIRE_FALSE(fj.novelty_batch(1));
        REQUIRE(fj.novelty_moves() == 1);
        REQUIRE(fj.novelty_commits() == 1);
        REQUIRE(m.var(vid(y)).value == 1.0);
        REQUIRE(m.var(vid(x)).value == 0.0);
    }
}

TEST_CASE("Novelty seeds its scan set once per batch, not per commit or level",
          "[fj][novelty][novelty_batch]") {
    // A chain of independent violated rows c_i: x_i >= 1, each fixed by one
    // flip: every flip commits on its own, so a batch commits once per row.
    // Before #209 every commit (and every failed level) reseeded Q' from all of
    // V, O(nnz(V)) each; the commits' own re-queues already hold what moved.
    constexpr int kRows = 50;
    Model m;
    for (int i = 0; i < kRows; ++i) {
        m.add_constraint(m.geq(m.bool_var(), m.constant(1.0)));
    }
    m.close();
    ViolationManager vm(m);
    RNG rng(11);
    FeasibilityJump fj(m, vm, rng, GFJConfig{});
    fj.begin(/*set_initial_x=*/false);

    REQUIRE(fj.novelty_batch(1000));
    REQUIRE(vm.is_feasible());
    REQUIRE(fj.novelty_commits() == kRows);
    REQUIRE(fj.novelty_scan_set_seeds() == 1);
    REQUIRE(fj.scan_sets_consistent());
}

TEST_CASE("Novelty's incremental W' matches the per-level re-init between calls",
          "[fj][novelty][novelty_batch]") {
    // Algorithm 4 re-initialises W' at every level: W on violated rows, W/1024
    // elsewhere. #209 redoes only the rows noted since the last reset; the
    // invariant that makes that exact is that every row NOT noted already holds
    // its re-init value. Checked after every Novelty batch and every
    // stand-alone ApplyNoveltyJump on a model that commits, backtracks, widens
    // levels and bumps.
    Model m;
    build_random_int_model(m, 303, 50, 120);
    m.close();
    ViolationManager vm(m);
    RNG rng(23);
    GFJConfig cfg;
    cfg.two_phase = false;
    FeasibilityJump fj(m, vm, rng, cfg);
    fj.begin(/*set_initial_x=*/true);
    for (int b = 0; b < 30; ++b) {
        CAPTURE(b);
        fj.set_rho(b % 2 == 0 ? 0.95 : 1.0);
        fj.novelty_batch(300);
        REQUIRE(fj.novelty_weights_consistent());
        REQUIRE(fj.scan_sets_consistent());
        fj.batch(100);
        fj.apply_novelty_jump();
        REQUIRE(fj.novelty_weights_consistent());
        REQUIRE(fj.scan_sets_consistent());
        fj.resync();
    }
    // The run exercised every path the invariant has to survive.
    REQUIRE(fj.novelty_commits() > 0);
    REQUIRE(fj.novelty_weight_bumps() > 0);
}

TEST_CASE("the search keeps its FJ-or-Novelty choice until a new best or a kick",
          "[search][novelty]") {
    // Algorithm 6 draws A in {FJ, NJ} at the start, on a new best and on a
    // perturbation, and otherwise "continue[s] with the same algorithm as last
    // batch" (paper section 5). So between two consecutive scalar batches the
    // kind may change only if the first produced a new best or a kick came
    // between them. Before #209 it was redrawn every batch.
    int switches = 0;
    for (uint64_t seed = 1; seed <= 6; ++seed) {
        CAPTURE(seed);
        Model m;
        build_random_int_model(m, static_cast<uint32_t>(100 + seed), 40, 100);
        m.close();
        SearchConfig cfg;
        cfg.use_compound_moves = true;
        cfg.novelty_jump_probability = 0.5;
        cfg.batch_iterations = 50;
        cfg.perturbation_period = 4;
        cfg.max_iterations = 6000;
        KindRecorder rec;
        cfg.tracer = &rec;
        solve(m, /*time_limit=*/0.0, seed, true, nullptr, nullptr, 3, nullptr, cfg);
        bool have_prev = false;
        BatchKind prev = BatchKind::FeasibilityJump;
        bool redraw_allowed = true;
        for (const auto& e : rec.events) {
            if (e.type == KindRecorder::Event::Type::Kick) {
                redraw_allowed = true;
                continue;
            }
            REQUIRE(e.kind != BatchKind::Structural);  // no structure in the model
            if (have_prev && e.kind != prev) {
                REQUIRE(redraw_allowed);
                ++switches;
            }
            have_prev = true;
            prev = e.kind;
            redraw_allowed = e.improved;
        }
    }
    REQUIRE(switches > 0);  // both kinds ran, so the check above had teeth
}

TEST_CASE("a Novelty batch that applies no move still stops at the deadline",
          "[search][novelty][novelty_batch]") {
    // x - y >= 1 and y - x >= 1 at x = y = 0: each flip fixes one row by 1 and
    // worsens the other by 1, both violated at full W', so every var fails F at
    // the root and ApplyNoveltyJump returns a local minimum having applied no
    // move. The batch bumps both rows equally and descends again, forever, and
    // with batch_iterations = 0 nothing but the deadline can stop it. The
    // deadline used to be polled only when a move was applied (#209 review).
    // Hand-registered in tests/CMakeLists.txt with a TIMEOUT, so a hang reports.
    Model m;
    const int32_t x = m.bool_var();
    const int32_t y = m.bool_var();
    m.add_constraint(m.geq(m.sum({x, m.prod(m.constant(-1.0), y)}), m.constant(1.0)));
    m.add_constraint(m.geq(m.sum({y, m.prod(m.constant(-1.0), x)}), m.constant(1.0)));
    m.close();
    SearchConfig cfg;
    cfg.use_compound_moves = true;
    cfg.novelty_jump_probability = 1.0;
    cfg.batch_iterations = 0;
    const SearchResult r = solve(m, /*time_limit=*/0.3, 7, true, nullptr, nullptr, 3, nullptr, cfg);
    REQUIRE_FALSE(r.feasible);
    REQUIRE(r.counters.novelty_batches > 0);
    REQUIRE(r.counters.novelty_moves == 0);  // the zero-move case, as intended
    REQUIRE(r.counters.novelty_weight_bumps > 0);
    REQUIRE(r.time_seconds < 5.0);
}
