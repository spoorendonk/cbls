// Commitment local search and ILS of the opt-in UC hook (#460). Properties
// only: no pinned values, as libstdc++ and libc++ draw differently.

#include "benchmarks/uc-chped/data.h"
#include "benchmarks/uc-chped/uc_commitment.h"
#include "benchmarks/uc-chped/uc_dispatch.h"
#include "benchmarks/uc-chped/uc_search.h"
#include "benchmarks/uc-chped/verify_uc_chped.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/rng.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace cbls;
using namespace cbls::uc_chped;

namespace {

constexpr double kTol = 1e-6;
using Clock = std::chrono::steady_clock;
using UnitDispatch = std::vector<std::vector<double>>;  // [u][t]

UCInstance ucp13() {
    return load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
}

UnitDispatch zero_dispatch(const UCInstance& in) {
    UnitDispatch p(in.n_units, std::vector<double>(in.n_periods, 0.0));
    return p;
}

// A repaired random commitment: ucp13's all-on covers demand + reserve and its
// sum Pmin is below every demand, so repair succeeds.
Commitment feasible_commitment(const UCInstance& in, RNG& rng, double on_prob) {
    Commitment y(in.n_units, CommitRow(in.n_periods, 0));
    for (auto& row : y) {
        for (auto& v : row) {
            v = rng.random() < on_prob ? 1 : 0;
        }
    }
    REQUIRE(repair_state(in, y));
    return y;
}

// The search loaded at y (no dispatch), costs cached.
CommitmentSearch loaded_search(const UCInstance& in, const Commitment& y,
                               const CommitmentSearchConfig& cfg, uint64_t seed) {
    CommitmentSearch s(in, cfg, seed);
    s.load(y, zero_dispatch(in));
    REQUIRE(s.init_costs());
    return s;
}

// pc_ and su_ equal a fresh computation for the current commitment.
void require_costs_consistent(const UCInstance& in, const CommitmentSearch& s,
                              const CommitmentSearchConfig& cfg) {
    const auto fresh = loaded_search(in, s.commitment(), cfg, 0);
    REQUIRE(s.period_costs().size() == fresh.period_costs().size());
    for (size_t t = 0; t < fresh.period_costs().size(); ++t) {
        CHECK(std::abs(s.period_costs()[t] - fresh.period_costs()[t]) <= kTol);
    }
    for (int u = 0; u < in.n_units; ++u) {
        CHECK(s.startup_costs()[u] == startup_cost(in, u, s.commitment()[u]));
    }
}

// Raises after `polls` calls to requested().
struct CountdownStop {
    mutable int polls;
    bool requested() const { return polls-- <= 0; }
};

}  // namespace

TEST_CASE("uc search try_rows leaves the commitment and its costs as they were", "[uc-search]") {
    const auto in = ucp13();
    RNG rng(5);
    const CommitmentSearchConfig cfg;
    for (int trial = 0; trial < 10; ++trial) {
        auto s = loaded_search(in, feasible_commitment(in, rng, 0.4), cfg, trial);
        const Commitment before = s.commitment();
        const auto pc = s.period_costs();
        std::vector<CommitRow> cands;
        std::vector<int> touched;
        std::vector<double> new_pc;
        for (int u = 0; u < in.n_units; ++u) {
            s.candidate_rows(u, cands);
            for (const auto& c : cands) {
                CHECK(row_feasible(in, u, c));
                (void)s.try_rows({u}, {c}, touched, new_pc);
            }
        }
        CHECK(s.commitment() == before);
        CHECK(s.period_costs() == pc);
    }
}

TEST_CASE("uc search try_rows into an undispatchable period is +inf, never NaN", "[uc-search]") {
    const auto in = ucp13();
    const CommitmentSearchConfig cfg;
    // All off: every period is undispatchable, so every cached cost is +inf.
    const Commitment off(in.n_units, CommitRow(in.n_periods, 0));
    CommitmentSearch s(in, cfg, 1);
    s.load(off, zero_dispatch(in));
    REQUIRE(s.init_costs());
    REQUIRE(s.period_costs()[0] == kDispatchInf);
    CommitRow r(in.n_periods, 0);
    r[0] = 1;  // one unit cannot meet period 0's demand alone
    std::vector<int> touched;
    std::vector<double> new_pc;
    const double d = s.try_rows({0}, {r}, touched, new_pc);
    CHECK_FALSE(std::isnan(d));
    CHECK(d == kDispatchInf);
}

TEST_CASE("uc search descent keeps costs consistent, feasible and never higher", "[uc-search]") {
    const auto in = ucp13();
    RNG rng(9);
    CommitmentSearchConfig cfg;
    for (int trial = 0; trial < 6; ++trial) {
        cfg.ls_polish_sweeps = trial % 2 == 0 ? 0 : 2;
        auto s = loaded_search(in, feasible_commitment(in, rng, 0.3), cfg, trial);
        const double start = s.total_cost();
        s.descend();
        CHECK(s.total_cost() <= start + kTol);
        CHECK(s.feasible());
        require_costs_consistent(in, s, cfg);
        std::vector<int> kicked;
        s.kick(kicked);
        CHECK(std::isfinite(s.total_cost()));
        CHECK(s.feasible());
        require_costs_consistent(in, s, cfg);
    }
}

TEST_CASE("uc search ILS ends on its best on every exit", "[uc-search]") {
    const auto in = ucp13();
    RNG rng(13);
    CommitmentSearchConfig cfg;
    cfg.ils_stall = 5;
    const Commitment y = feasible_commitment(in, rng, 0.3);

    auto check_exit = [&](CommitmentSearch& s, double start) {
        CHECK(s.total_cost() <= start + kTol);
        CHECK(s.feasible());
        require_costs_consistent(in, s, cfg);
    };
    SECTION("stall cap") {
        auto s = loaded_search(in, y, cfg, 3);
        s.descend();
        const double start = s.total_cost();
        s.iterate({}, Clock::time_point::max());
        CHECK(s.stats().ils_kicks >= cfg.ils_stall);
        check_exit(s, start);
    }
    SECTION("restricted descent") {
        cfg.ils_local = 2;
        auto s = loaded_search(in, y, cfg, 3);
        s.descend();
        const double start = s.total_cost();
        s.iterate({}, Clock::time_point::max());
        check_exit(s, start);
    }
    SECTION("deadline already passed") {
        auto s = loaded_search(in, y, cfg, 3);
        s.descend();
        const Commitment descended = s.commitment();
        const double start = s.total_cost();
        s.iterate({}, Clock::now());
        CHECK(s.stats().ils_kicks == 0);
        CHECK(s.commitment() == descended);
        check_exit(s, start);
    }
    SECTION("stop raised inside a re-descent") {
        for (int polls = 1; polls < 40; polls += 3) {
            auto s = loaded_search(in, y, cfg, 3);
            s.descend();
            const double start = s.total_cost();
            const CountdownStop stop{polls};
            s.iterate(stop, Clock::time_point::max());
            check_exit(s, start);
        }
    }
}

TEST_CASE("uc search memos at their cap change no result", "[uc-search]") {
    const auto in = ucp13();
    RNG rng(21);
    const Commitment y = feasible_commitment(in, rng, 0.3);
    CommitmentSearchConfig big;
    CommitmentSearchConfig tiny;
    tiny.memo_cap = 40;  // the polish memo holds 2
    auto a = loaded_search(in, y, big, 7);
    auto b = loaded_search(in, y, tiny, 7);
    a.descend();
    b.descend();
    CHECK(b.fast_memo_size() <= tiny.memo_cap);
    CHECK(a.commitment() == b.commitment());
    CommitmentSearch::Dispatches pa;
    CommitmentSearch::Dispatches pb;
    const double ca = a.final_dispatch(pa);
    const double cb = b.final_dispatch(pb);
    CHECK(b.polish_memo_size() <= tiny.memo_cap / 20);
    CHECK(ca == cb);
    CHECK(pa == pb);
}

TEST_CASE("uc search final dispatch balances, holds bounds and matches the verifier",
          "[uc-search]") {
    const auto in = ucp13();
    RNG rng(25);
    const CommitmentSearchConfig cfg;
    for (int trial = 0; trial < 5; ++trial) {
        auto s = loaded_search(in, feasible_commitment(in, rng, 0.4), cfg, trial);
        s.descend();
        CommitmentSearch::Dispatches p;
        const double total = s.final_dispatch(p);
        REQUIRE(std::isfinite(total));
        verify_detail::Assignment a;
        a.y.assign(in.n_units, std::vector<int>(in.n_periods, 0));
        a.p.assign(in.n_units, std::vector<double>(in.n_periods, 0.0));
        for (int t = 0; t < in.n_periods; ++t) {
            double sum = 0.0;
            for (int u = 0; u < in.n_units; ++u) {
                const double x = p[t][u];
                if (s.commitment()[u][t] != 0) {
                    CHECK(x >= in.P_min[u] - kTol);
                    CHECK(x <= in.P_max[u] + kTol);
                } else {
                    CHECK(x == 0.0);
                }
                sum += x;
                a.y[u][t] = s.commitment()[u][t];
                a.p[u][t] = x;
            }
            CHECK(std::abs(sum - in.demand[t]) < kTol);
        }
        const double recomputed = verify_detail::recompute_cost(a, in);
        CHECK(std::abs(total - recomputed) <= 1e-6 * std::abs(recomputed));
    }
}

TEST_CASE("uc search polish memo does not carry one call's loaded dispatch to the next",
          "[uc-search]") {
    const auto in = ucp13();
    RNG rng(31);
    // No polish sweeps: from scratch is the fast dispatch, so a polished
    // loaded p is cheaper wherever polish helps.
    CommitmentSearchConfig cfg;
    cfg.polish_sweeps = 0;
    const Commitment y = feasible_commitment(in, rng, 0.4);
    const Dispatcher ed(in);
    UnitDispatch good = zero_dispatch(in);
    for (int t = 0; t < in.n_periods; ++t) {
        std::vector<int> on;
        for (int u = 0; u < in.n_units; ++u) {
            if (y[u][t] != 0) {
                on.push_back(u);
            }
        }
        std::vector<double> p;
        REQUIRE(std::isfinite(ed.fast(on, in.demand[t], p)));
        (void)ed.polish(on, p, 20);
        for (int u = 0; u < in.n_units; ++u) {
            good[u][t] = p[u];
        }
    }
    CommitmentSearch s(in, cfg, 1);
    CommitmentSearch::Dispatches p;
    s.load(y, good);
    const double with_good = s.final_dispatch(p);
    s.load(y, zero_dispatch(in));  // does not balance: from scratch only
    const double after = s.final_dispatch(p);
    CommitmentSearch fresh(in, cfg, 1);
    fresh.load(y, zero_dispatch(in));
    const double from_scratch = fresh.final_dispatch(p);
    REQUIRE(with_good < from_scratch - kTol);  // the case is live
    CHECK(after == from_scratch);
}

TEST_CASE("uc search init_costs and final_dispatch stop when asked", "[uc-search]") {
    const auto in = ucp13();
    RNG rng(37);
    const CommitmentSearchConfig cfg;
    CommitmentSearch s(in, cfg, 1);
    s.load(feasible_commitment(in, rng, 0.4), zero_dispatch(in));
    const CountdownStop stop_init{2};
    CHECK_FALSE(s.init_costs(stop_init));
    REQUIRE(s.init_costs());
    const CountdownStop stop_final{2};
    CommitmentSearch::Dispatches p;
    CHECK(s.final_dispatch(p, stop_final) == kDispatchInf);
    CHECK(std::isfinite(s.final_dispatch(p)));
}
