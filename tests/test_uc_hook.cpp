// UCHook (#461): the opt-in UC inner solver's write-back, rollback, stop and
// threading contracts on ucp13.

#include "benchmarks/uc-chped/data.h"
#include "benchmarks/uc-chped/greedy_init.h"
#include "benchmarks/uc-chped/uc_hook.h"
#include "benchmarks/uc-chped/uc_model.h"
#include "benchmarks/uc-chped/verify_uc_chped.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cbls/cbls.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>

using namespace cbls;
using namespace cbls::uc_chped;

namespace {

constexpr double kTol = 1e-6;
using Clock = std::chrono::steady_clock;

// ucp13 at its greedy start, fully evaluated.
struct Fixture {
    UCInstance inst;
    UCModel ucm;
    explicit Fixture(bool demand_equality = false,
                     const char* path = "benchmarks/instances/uc-chped/ucp13.jsonl")
        : inst(load_jsonl(path)), ucm(build_uc_model(inst, demand_equality)) {
        greedy_uc_initialize(ucm.model, inst, ucm);
        full_evaluate(ucm.model);
    }
};

double objective(const Model& m) {
    return m.node_value(m.objective_id());
}

std::vector<double> values(const Model& m) {
    std::vector<double> v(m.num_vars());
    for (size_t i = 0; i < v.size(); ++i) {
        v[i] = m.var(static_cast<int32_t>(i)).value;
    }
    return v;
}

// Raises after `polls` calls to requested().
struct CountdownStop {
    mutable int polls;
    bool requested() const { return polls-- <= 0; }
};

// Counts its polls, never raises.
struct CountingStop {
    mutable int polls = 0;
    bool requested() const {
        ++polls;
        return false;
    }
};

// Every node value equals a fresh full evaluation of the same assignment.
void require_evaluated(Model& m) {
    const double obj = objective(m);
    ViolationManager vm(m);
    const double viol = vm.max_real_violation();
    full_evaluate(m);
    CHECK(objective(m) == obj);
    CHECK(vm.max_real_violation() == viol);
}

}  // namespace

TEST_CASE("uc hook write-back is feasible, balanced and costs what the search reported",
          "[uc-hook]") {
    const bool equality = GENERATE(false, true);
    CAPTURE(equality);
    Fixture f(equality);
    Model& m = f.ucm.model;
    ViolationManager vm(m);
    REQUIRE(vm.max_real_violation() <= kTol);
    const double before = objective(m);

    UCHook hook(f.inst, f.ucm, UCHookConfig{}, 1);
    hook.solve(m, vm);
    REQUIRE(hook.stats().improvements == 1);
    CHECK(hook.stats().rejected == 0);

    const double after = objective(m);
    CHECK(after < before);
    CHECK(std::abs(after - hook.last_kept_total()) <= kTol * std::abs(after));
    CHECK(hook.best_seen() == after);
    CHECK(vm.max_real_violation() <= kTol);
    for (int t = 0; t < f.inst.n_periods; ++t) {
        double sum = 0.0;
        for (int u = 0; u < f.inst.n_units; ++u) {
            const double y = m.var(handle_to_var_id(f.ucm.y[u][t])).value;
            const double p = m.var(handle_to_var_id(f.ucm.p[u][t])).value;
            CHECK((y == 0.0 || y == 1.0));
            CHECK(p >= (y * f.inst.P_min[u]) - kTol);
            CHECK(p <= (y * f.inst.P_max[u]) + kTol);
            sum += p;
        }
        CHECK(std::abs(sum - f.inst.demand[t]) <= kTol * f.inst.demand[t]);
    }
    const VerifyResult vr = verify_uc_chped(f.ucm, f.inst);
    CHECK(vr.ok);
    require_evaluated(m);
}

TEST_CASE("uc hook repairs an all-off commitment into a feasible point", "[uc-hook]") {
    Fixture f;
    Model& m = f.ucm.model;
    for (int u = 0; u < f.inst.n_units; ++u) {
        for (int t = 0; t < f.inst.n_periods; ++t) {
            m.var_mut(handle_to_var_id(f.ucm.y[u][t])).value = 0.0;
            m.var_mut(handle_to_var_id(f.ucm.p[u][t])).value = 0.0;
        }
    }
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.max_real_violation() > kTol);

    UCHookConfig cfg;
    cfg.repair = true;
    cfg.on_infeasible = true;
    UCHook hook(f.inst, f.ucm, cfg, 2);
    CHECK(hook.wants_infeasible());
    hook.solve(m, vm);
    CHECK(hook.stats().infeasible_calls == 1);
    REQUIRE(hook.stats().improvements == 1);
    CHECK(vm.max_real_violation() <= kTol);
    CHECK(verify_uc_chped(f.ucm, f.inst).ok);
    require_evaluated(m);
}

TEST_CASE("uc hook without repair skips an infeasible commitment", "[uc-hook]") {
    Fixture f;
    Model& m = f.ucm.model;
    for (int u = 0; u < f.inst.n_units; ++u) {
        for (int t = 0; t < f.inst.n_periods; ++t) {
            m.var_mut(handle_to_var_id(f.ucm.y[u][t])).value = 0.0;
        }
    }
    full_evaluate(m);
    const std::vector<double> start = values(m);
    ViolationManager vm(m);
    UCHook hook(f.inst, f.ucm, UCHookConfig{}, 3);
    hook.solve(m, vm);
    CHECK(hook.stats().skipped == 1);
    CHECK(hook.stats().improvements == 0);
    CHECK(values(m) == start);
}

// A negative tolerance makes every point infeasible to the hook, so its
// write-back is refused: the model must come back value for value.
TEST_CASE("uc hook rolls back a write-back the model refuses, then skips the repeat", "[uc-hook]") {
    Fixture f;
    Model& m = f.ucm.model;
    const std::vector<double> start = values(m);
    const double obj = objective(m);
    ViolationManager vm(m);
    UCHookConfig cfg;
    cfg.feasibility_tolerance = -1.0;
    cfg.infeasible_share = 1.0;  // never throttled: own time <= lifetime
    UCHook hook(f.inst, f.ucm, cfg, 4);
    hook.solve(m, vm);
    CHECK(hook.stats().rejected == 1);
    CHECK(hook.stats().improvements == 0);
    CHECK(hook.best_seen() == kDispatchInf);
    CHECK(values(m) == start);
    CHECK(objective(m) == obj);
    require_evaluated(m);

    hook.solve(m, vm);
    CHECK(hook.stats().skipped == 1);
    CHECK(values(m) == start);
}

TEST_CASE("uc hook throttles infeasible calls past its time share", "[uc-hook]") {
    Fixture f;
    ViolationManager vm(f.ucm.model);
    UCHookConfig cfg;
    cfg.feasibility_tolerance = -1.0;
    cfg.infeasible_share = 0.0;  // the first call runs (no time used yet), then none
    UCHook hook(f.inst, f.ucm, cfg, 6);
    hook.solve(f.ucm.model, vm);
    hook.solve(f.ucm.model, vm);
    CHECK(hook.stats().calls == 1);
    CHECK(hook.stats().throttled == 1);
}

// Every phase polls `stop`; wherever it is raised, the model is the start or a
// kept, feasible, cheaper write-back, and fully evaluated.
TEST_CASE("uc hook stopped at any poll leaves the start or a kept point", "[uc-hook]") {
    UCHookConfig cfg;
    cfg.repair = true;
    cfg.ils_seconds = 10.0;  // the ILS ends on its stall count, not the clock
    cfg.search.ils_stall = 5;

    // Polls of an unstopped second call (the first runs no ILS).
    int total_polls = 0;
    {
        Fixture f;
        ViolationManager vm(f.ucm.model);
        UCHook hook(f.inst, f.ucm, cfg, 5);
        hook.solve(f.ucm.model, vm);
        const CountingStop counter;
        hook.solve(f.ucm.model, vm, {}, counter);
        total_polls = counter.polls;
    }
    REQUIRE(total_polls > 10);

    const int step = std::max(1, total_polls / 25);
    for (int polls = 0; polls <= total_polls; polls += step) {
        CAPTURE(polls);
        Fixture f;
        Model& m = f.ucm.model;
        ViolationManager vm(m);
        UCHook hook(f.inst, f.ucm, cfg, 5);
        hook.solve(m, vm);
        const std::vector<double> start = values(m);
        const double obj = objective(m);
        const int64_t kept = hook.stats().improvements;
        const CountdownStop stop{polls};
        hook.solve(m, vm, {}, stop);
        if (hook.stats().improvements == kept) {
            CHECK(values(m) == start);
        } else {
            CHECK(objective(m) < obj);
            CHECK(vm.max_real_violation() <= kTol);
        }
        require_evaluated(m);
    }
}

#ifdef NDEBUG
constexpr bool kOptimizedBuild = true;
#else
constexpr bool kOptimizedBuild = false;
#endif

// #460 review: iterate() reads its end between kicks only, and on ucp200 one
// full re-descent (ils_local = 0) runs far past a 20 ms ILS. Measured
// 2026-11-28 (Release, M-series Mac), ILS phase over 9 calls: with the deadline
// folded into the re-descent's stop the overrun is at most 0.4 ms; unfolded,
// 17-221 ms, median 52 ms. The 5 ms bound on the median sits between.
TEST_CASE("uc hook ILS stops at its deadline mid re-descent", "[timing][uc-hook]") {
    if (!kOptimizedBuild) {
        SKIP("the bound is measured at -O3; see the note above");
    }
    Fixture f(false, "benchmarks/instances/uc-chped/ucp200.jsonl");
    Model& m = f.ucm.model;
    ViolationManager vm(m);
    UCHookConfig cfg;
    cfg.ils_seconds = 0.02;
    cfg.search.ils_stall = 1 << 30;  // only the clock ends the ILS
    cfg.repair = true;               // the greedy start breaks capacity
    UCHook hook(f.inst, f.ucm, cfg, 7);
    hook.solve(m, vm);  // the first descent, no ILS
    REQUIRE(hook.stats().improvements == 1);
    std::vector<double> overrun;  // the ILS phase alone: the descent before it is not bounded
    for (int i = 0; i < 9; ++i) {
        const int64_t before = hook.stats().ils_nanos;
        hook.solve(m, vm);
        overrun.push_back((1e-9 * static_cast<double>(hook.stats().ils_nanos - before)) -
                          cfg.ils_seconds);
    }
    REQUIRE(hook.search_stats().ils_kicks >= 9);
    std::sort(overrun.begin(), overrun.end());
    CAPTURE(overrun);
    CHECK(overrun[overrun.size() / 2] <= 0.005);
}

TEST_CASE("uc hook one per worker under ParallelSearch keeps valid handles", "[uc-hook]") {
    Fixture f;
    std::mutex mu;
    std::vector<std::shared_ptr<UCHook>> hooks;
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> factory =
        [&](Model&) -> std::shared_ptr<InnerSolverHook> {
        const std::scoped_lock lock(mu);
        auto h = std::make_shared<UCHook>(f.inst, f.ucm, UCHookConfig{}, 100 + hooks.size());
        hooks.push_back(h);
        return h;
    };
    SearchConfig cfg;
    cfg.skip_init = true;
    ParallelConfig par;
    par.n_threads = 2;
    ParallelSearch ps(2);
    const SearchResult r = ps.solve(f.ucm.model, 1.0, 9, cfg, factory, {}, nullptr, par);
    REQUIRE(hooks.size() == 2);
    CHECK(hooks[0] != hooks[1]);
    int64_t calls = 0;
    for (const auto& h : hooks) {
        calls += h->stats().calls;
    }
    CHECK(calls > 0);
    REQUIRE(r.feasible);
    REQUIRE(r.best_state.values.size() == f.ucm.model.num_vars());
    f.ucm.model.restore_state(r.best_state);
    full_evaluate(f.ucm.model);
    CHECK(verify_uc_chped(f.ucm, f.inst).ok);
}
