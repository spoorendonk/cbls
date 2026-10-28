// #239's two restart arms after OR-Tools' ViolationLS: restart_randomization
// (random initial weights and the zero-perturbation restart) and
// luby_restart_unit (a Luby kick schedule in place of #102's unproductive
// kick). Off, both are today's search draw for draw, which the trajectory
// fences pin; what is pinned here is the sequence, the weight fill, and that
// each option reaches solve()'s kicks.
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cstdint>
#include <stdexcept>
#include <vector>

using namespace cbls;

namespace {

// Integer columns summing to 0.5: never feasible, so no batch improves and
// the kick schedule runs undisturbed by new bests to the iteration cap.
void build_infeasible(Model& m) {
    std::vector<int32_t> xs;
    xs.reserve(50);
    for (int i = 0; i < 50; ++i) {
        xs.push_back(m.int_var(0, 10'000'000));
    }
    m.add_constraint(m.eq_expr(m.sum(xs), m.constant(0.5)));
    m.close();
}

// Records the batch count at every kick, and the kinds.
struct KickLog : Tracer {
    int64_t batches = 0;
    std::vector<int64_t> kick_batches;
    std::vector<KickKind> kinds;
    void batch_end(BatchKind /*kind*/, int64_t /*iterations*/, bool /*improved*/) override {
        ++batches;
    }
    void kick(KickKind kind) override {
        kick_batches.push_back(batches);
        kinds.push_back(kind);
    }
    [[nodiscard]] int count(KickKind kind) const {
        int n = 0;
        for (const KickKind k : kinds) {
            n += k == kind ? 1 : 0;
        }
        return n;
    }
};

SearchResult solve_infeasible(SearchConfig config, uint64_t seed, KickLog& log) {
    Model m;
    build_infeasible(m);
    config.use_compound_moves = false;
    config.tracer = &log;
    return solve(m, /*time_limit=*/0.0, seed, true, nullptr, nullptr, 3, nullptr, config);
}

}  // namespace

TEST_CASE("luby is the Luby sequence (#239)", "[restart]") {
    const std::vector<int64_t> expected = {1, 1, 2, 1, 1, 2, 4, 1, 1, 2, 1, 1, 2, 4, 8, 1};
    for (size_t i = 0; i < expected.size(); ++i) {
        CAPTURE(i + 1);
        REQUIRE(luby(static_cast<int64_t>(i) + 1) == expected[i]);
    }
    // t_{2^k - 1} = 2^(k-1), deep into the sequence.
    REQUIRE(luby((int64_t{1} << 20) - 1) == (int64_t{1} << 19));
    REQUIRE(luby(int64_t{1} << 20) == 1);
    REQUIRE_THROWS_AS(luby(0), std::invalid_argument);
}

TEST_CASE("restart options are off by default (#239)", "[restart]") {
    const SearchConfig config;
    REQUIRE_FALSE(config.restart_randomization);
    REQUIRE(config.luby_restart_unit == 0);
}

TEST_CASE("a negative luby_restart_unit is refused (#239)", "[restart]") {
    SearchConfig config;
    config.luby_restart_unit = -1;
    REQUIRE_THROWS_AS(validate_search_config(config), std::invalid_argument);
    Model m;
    build_infeasible(m);
    REQUIRE_THROWS_AS(solve(m, 0.0, 1, true, nullptr, nullptr, 3, nullptr, config),
                      std::invalid_argument);
}

TEST_CASE("reset_weights(random) fills Uniform(1, 2) (#239)", "[restart]") {
    Model m;
    build_infeasible(m);
    ViolationManager vm(m);
    RNG rng(5);
    GFJConfig config;
    FeasibilityJump fj(m, vm, rng, config);
    fj.begin(/*set_initial_x=*/true);
    REQUIRE(!vm.weights.empty());
    bool off_one = false;
    for (int round = 0; round < 50; ++round) {
        fj.reset_weights(/*random=*/true);
        for (const double w : vm.weights) {
            REQUIRE(w >= 1.0);
            REQUIRE(w < 2.0);
            off_one = off_one || w != 1.0;
        }
    }
    REQUIRE(off_one);
    fj.reset_weights();
    for (const double w : vm.weights) {
        REQUIRE(w == 1.0);
    }
}

TEST_CASE("restart_randomization reaches solve()'s kicks (#239)", "[restart]") {
    for (const uint64_t seed : {1, 2, 3}) {
        SearchConfig config;
        config.max_iterations = 20'000;
        KickLog off;
        const SearchResult r_off = solve_infeasible(config, seed, off);
        config.restart_randomization = true;
        KickLog on;
        const SearchResult r_on = solve_infeasible(config, seed, on);
        CAPTURE(seed, off.kinds.size(), on.kinds.size(), on.count(KickKind::Restart));
        REQUIRE_FALSE(r_on.feasible);
        // Off, no kick is a restart; on, about half of the perturb kicks are.
        REQUIRE(off.count(KickKind::Restart) == 0);
        REQUIRE(on.count(KickKind::Restart) > 0);
        REQUIRE(on.count(KickKind::Perturb) > 0);
        REQUIRE(r_off.perturbations > 0);
        REQUIRE(r_on.perturbations > 0);
    }
}

namespace {

// A feasible model with an objective, whose rows 1..kGuards are `x_i <= 10`
// on a [0, 10] domain: never violated, so never bumped, and decay scales every
// row alike -- with W reset to 1 they stay equal, and only a random fill can
// part them. The hook runs on real-feasible batches with W materialised.
//
// Each of the three draw sites is told apart by what came last before a hook
// call. A new best runs the hook and THEN resets W, so with no LNS and no pool
// the last reset before a call is the kick's whenever a kick fell since the
// previous call, and the new best's otherwise.
constexpr int kGuards = 8;

struct GuardWeightProbe : InnerSolverHook {
    const KickLog* kicks = nullptr;
    size_t kicks_seen = 0;
    int calls = 0;
    bool parted = false;
    bool parted_last_call = false;
    bool parted_after_new_best = false;
    bool parted_after_restart = false;
    bool parted_after_perturb = false;
    void solve(Model& /*model*/, ViolationManager& vm,
               const std::vector<int32_t>& /*last_changed_vars*/, StopRef /*stop*/) override {
        const bool first = calls++ == 0;
        const size_t kicked = kicks->kinds.size();
        const bool after_kick = kicked > kicks_seen;
        const KickKind last = after_kick ? kicks->kinds.back() : KickKind::Perturb;
        kicks_seen = kicked;
        if (vm.weights.size() <= static_cast<size_t>(kGuards)) {
            return;
        }
        bool now = false;
        for (int i = 2; i <= kGuards; ++i) {
            now = now || vm.weights[static_cast<size_t>(i)] != vm.weights[1];
        }
        parted = parted || now;
        const bool was_parted = parted_last_call;
        parted_last_call = now;
        if (!now || first) {
            return;
        }
        if (!after_kick) {
            // Equal last call, parted now, no kick between: the only reset in
            // between is the new best's that followed the last call.
            parted_after_new_best = parted_after_new_best || !was_parted;
        } else if (last == KickKind::Restart) {
            parted_after_restart = true;
        } else if (last == KickKind::Perturb) {
            parted_after_perturb = true;
        }
    }
};

GuardWeightProbe solve_guarded(bool restart_randomization, uint64_t seed, KickLog& log) {
    Model m;
    std::vector<int32_t> xs;
    std::vector<int32_t> cost;
    for (int i = 0; i < kGuards; ++i) {
        xs.push_back(m.int_var(0, 10));
        cost.push_back(m.prod(m.constant(1.0 + i), xs.back()));
    }
    m.add_constraint(m.geq(m.sum(xs), m.constant(25.0)));
    for (const int32_t x : xs) {
        m.add_constraint(m.leq(x, m.constant(10.0)));
    }
    m.minimize(m.sum(cost));
    m.close();
    // Short batches and a short period, so kicks fall while the run is
    // feasible and the hook sees what each one left.
    SearchConfig config;
    config.batch_iterations = 100;
    config.perturbation_period = 3;
    config.max_iterations = 40'000;
    config.restart_randomization = restart_randomization;
    config.tracer = &log;
    GuardWeightProbe probe;
    probe.kicks = &log;
    solve(m, /*time_limit=*/0.0, seed, true, &probe, nullptr, 3, nullptr, config);
    return probe;
}

}  // namespace

TEST_CASE("restart_randomization's weight fill reaches solve() (#239)", "[restart]") {
    bool new_best = false;
    bool restart = false;
    bool perturb = false;
    for (const uint64_t seed : {1, 2, 3, 4, 5}) {
        KickLog off_log;
        const GuardWeightProbe off = solve_guarded(false, seed, off_log);
        CAPTURE(seed, off.calls, off_log.kinds.size());
        REQUIRE(off.calls > 1);
        REQUIRE_FALSE(off_log.kinds.empty());
        REQUIRE_FALSE(off.parted);
        KickLog on_log;
        const GuardWeightProbe on = solve_guarded(true, seed, on_log);
        new_best = new_best || on.parted_after_new_best;
        restart = restart || on.parted_after_restart;
        perturb = perturb || on.parted_after_perturb;
    }
    // Each site on its own: the new best's reset, the restore-only kick's, and
    // the perturb kick's.
    CHECK(new_best);
    CHECK(restart);
    CHECK(perturb);
}

TEST_CASE("luby_restart_unit kicks on the Luby schedule (#239)", "[restart]") {
    // With no new best the schedule never resets, so the gaps between kicks,
    // in counted non-improving batches, are unit * 1, 1, 2, 1, 1, 2, 4, ... up
    // to the perturbation_period kick, which takes the 100th batch.
    for (const int unit : {1, 2, 3}) {
        SearchConfig config;
        config.batch_iterations = 50;
        config.max_iterations = int64_t{50} * 150;
        config.luby_restart_unit = unit;
        KickLog log;
        const SearchResult r = solve_infeasible(config, 7, log);
        REQUIRE_FALSE(r.feasible);
        int64_t previous = 0;
        int64_t k = 1;
        int checked = 0;
        for (const int64_t at : log.kick_batches) {
            if (at >= config.perturbation_period) {
                break;
            }
            CAPTURE(unit, k, previous, at);
            REQUIRE(at - previous == unit * luby(k));
            previous = at;
            ++k;
            ++checked;
        }
        CAPTURE(unit, checked);
        REQUIRE(checked >= 5);
    }
}

TEST_CASE("luby_restart_unit changes the kick count against #102's kick (#239)", "[restart]") {
    for (const uint64_t seed : {1, 2, 3}) {
        SearchConfig config;
        config.max_iterations = 20'000;
        KickLog base;
        solve_infeasible(config, seed, base);
        config.luby_restart_unit = 4;
        KickLog luby_log;
        solve_infeasible(config, seed, luby_log);
        CAPTURE(seed, base.kinds.size(), luby_log.kinds.size());
        REQUIRE_FALSE(luby_log.kinds.empty());
        REQUIRE(base.kinds.size() != luby_log.kinds.size());
    }
}
