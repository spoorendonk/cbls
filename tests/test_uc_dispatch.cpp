// The UC hook's pure pieces (#455): valve-point dispatch (uc_dispatch.h) and
// commitment repair (uc_commitment.h), checked against the independent
// UC-CHPED verifier's families rather than against themselves.
#include "benchmarks/uc-chped/data.h"
#include "benchmarks/uc-chped/uc_commitment.h"
#include "benchmarks/uc-chped/uc_dispatch.h"
#include "benchmarks/uc-chped/verify_uc_chped.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/rng.h>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

using namespace cbls;
using namespace cbls::uc_chped;

namespace {

constexpr double kTol = 1e-6;

UCInstance ucp13() {
    return load_jsonl("benchmarks/instances/uc-chped/ucp13.jsonl");
}

// The instance restricted to `units`, every per-unit vector filtered.
UCInstance subset_units(const UCInstance& in, const std::vector<int>& units) {
    UCInstance s = in;
    s.n_units = static_cast<int>(units.size());
    auto pick = [&](const auto& v) {
        std::decay_t<decltype(v)> out(units.size());
        for (size_t i = 0; i < units.size(); ++i) {
            out[i] = v[units[i]];
        }
        return out;
    };
    auto pick_d = [&](const std::vector<double>& v) { return pick(v); };
    auto pick_i = [&](const std::vector<int>& v) { return pick(v); };
    s.a = pick_d(in.a);
    s.b = pick_d(in.b);
    s.c = pick_d(in.c);
    s.d = pick_d(in.d);
    s.e = pick_d(in.e);
    s.P_min = pick_d(in.P_min);
    s.P_max = pick_d(in.P_max);
    s.a_hot = pick_d(in.a_hot);
    s.a_cold = pick_d(in.a_cold);
    s.min_on = pick_i(in.min_on);
    s.min_off = pick_i(in.min_off);
    s.t_cold = pick_i(in.t_cold);
    s.n_init = pick_i(in.n_init);
    s.y_prev = pick_i(in.y_prev);
    return s;
}

std::vector<int> all_units(const UCInstance& in) {
    std::vector<int> on(in.n_units);
    std::iota(on.begin(), on.end(), 0);
    return on;
}

std::vector<int> on_set(const Commitment& y, int t) {
    std::vector<int> on;
    for (int u = 0; u < static_cast<int>(y.size()); ++u) {
        if (y[u][t] != 0) {
            on.push_back(u);
        }
    }
    return on;
}

double sum_of(const std::vector<double>& p) {
    double s = 0.0;
    for (double x : p) {
        s += x;
    }
    return s;
}

// Σp = demand, every on unit within its bounds, every off unit at 0.
void require_dispatch_valid(const UCInstance& in, const std::vector<int>& on, double demand,
                            const std::vector<double>& p) {
    REQUIRE(static_cast<int>(p.size()) == in.n_units);
    CHECK(std::abs(sum_of(p) - demand) < kTol);
    std::vector<bool> is_on(in.n_units, false);
    for (int u : on) {
        is_on[u] = true;
    }
    for (int u = 0; u < in.n_units; ++u) {
        if (is_on[u]) {
            CHECK(p[u] >= in.P_min[u] - kTol);
            CHECK(p[u] <= in.P_max[u] + kTol);
        } else {
            CHECK(p[u] == 0.0);
        }
    }
}

Commitment random_commitment(const UCInstance& in, RNG& rng, double on_prob) {
    Commitment y(in.n_units, CommitRow(in.n_periods, 0));
    for (auto& row : y) {
        for (auto& v : row) {
            v = rng.random() < on_prob ? 1 : 0;
        }
    }
    return y;
}

verify_detail::Assignment to_assignment(const UCInstance& in, const Commitment& y,
                                        const std::vector<std::vector<double>>& p_by_period) {
    verify_detail::Assignment a;
    a.y.assign(in.n_units, std::vector<int>(in.n_periods, 0));
    a.p.assign(in.n_units, std::vector<double>(in.n_periods, 0.0));
    for (int u = 0; u < in.n_units; ++u) {
        for (int t = 0; t < in.n_periods; ++t) {
            a.y[u][t] = y[u][t];
            if (!p_by_period.empty()) {
                a.p[u][t] = p_by_period[t][u];
            }
        }
    }
    return a;
}

// The verifier's commitment families: min up, min down, initial conditions.
bool rows_pass_verifier(const UCInstance& in, const Commitment& y) {
    VerifyResult r;
    const auto a = to_assignment(in, y, {});
    verify_detail::check_min_run(r, a, in, /*on_block=*/true);
    verify_detail::check_min_run(r, a, in, /*on_block=*/false);
    verify_detail::check_initial_conditions(r, a, in);
    return r.ok;
}

// Fine 1-D scan of p_0 with p_1 = demand - p_0 (two units), every valve point
// of either unit included. Returns the best cost seen.
double brute_force_two(const Dispatcher& ed, const UCInstance& in, double demand, double h) {
    double best = kDispatchInf;
    auto eval = [&](double p0) {
        const double p1 = demand - p0;
        if (p0 < in.P_min[0] - 1e-12 || p0 > in.P_max[0] + 1e-12 || p1 < in.P_min[1] - 1e-12 ||
            p1 > in.P_max[1] + 1e-12) {
            return;
        }
        best = std::min(best, ed.fuel(0, p0) + ed.fuel(1, p1));
    };
    const int steps = static_cast<int>((in.P_max[0] - in.P_min[0]) / h);
    for (int k = 0; k <= steps; ++k) {
        eval(in.P_min[0] + (k * h));
    }
    for (double v : ed.valves(0)) {
        eval(v);
    }
    for (double v : ed.valves(1)) {
        eval(demand - v);
    }
    return best;
}

// Max |F'| over the unit's range, bounding a grid scan's error per unit.
double lipschitz(const UCInstance& in, int u) {
    return std::abs(in.b[u]) + (2.0 * std::abs(in.c[u]) * in.P_max[u]) +
           std::abs(in.d[u] * in.e[u]);
}

}  // namespace

TEST_CASE("uc dispatch sums to demand within bounds with off units at zero", "[uc-dispatch]") {
    const auto in = ucp13();
    Dispatcher ed(in);
    RNG rng(7);
    int dispatched = 0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<int> on;
        for (int u = 0; u < in.n_units; ++u) {
            if (rng.random() < 0.6) {
                on.push_back(u);
            }
        }
        double smin = 0.0;
        double smax = 0.0;
        for (int u : on) {
            smin += in.P_min[u];
            smax += in.P_max[u];
        }
        if (on.empty() || smax <= smin) {
            continue;
        }
        const double demand = rng.uniform(smin, smax);
        std::vector<double> p(in.n_units, 99.0);  // stale values must not survive
        const double fast = ed.fast(on, demand, p);
        REQUIRE(std::isfinite(fast));
        require_dispatch_valid(in, on, demand, p);
        CHECK(std::abs(fast - ed.cost_of(on, p)) < kTol);
        const double polished = ed.polish(on, p, 50);
        require_dispatch_valid(in, on, demand, p);
        CHECK(polished <= fast + kTol);
        ++dispatched;
    }
    CHECK(dispatched > 100);
}

TEST_CASE("uc dispatch polish never raises the cost under window and unit caps", "[uc-dispatch]") {
    const auto in = ucp13();
    Dispatcher ed(in);
    const auto on = all_units(in);
    for (double demand : {700.0, 1250.0, 1960.0, 2670.0}) {
        for (int window : {0, 1, 3}) {
            for (int jcap : {0, 2, 5}) {
                std::vector<double> p;
                const double fast = ed.fast(on, demand, p);
                REQUIRE(std::isfinite(fast));
                double prev = fast;
                for (int sweeps = 0; sweeps < 5; ++sweeps) {
                    const double now = ed.polish(on, p, 1, window, jcap);
                    CHECK(now <= prev + kTol);
                    require_dispatch_valid(in, on, demand, p);
                    prev = now;
                }
            }
        }
    }
}

TEST_CASE("uc dispatch of an on-set that cannot meet demand is infinite", "[uc-dispatch]") {
    const auto in = ucp13();
    Dispatcher ed(in);
    std::vector<double> p;
    // Units 9 and 10: Pmin 40 each, Pmax 120 each.
    CHECK(ed.fast({9, 10}, 300.0, p) == kDispatchInf);  // short of demand
    CHECK(ed.fast({9, 10}, 50.0, p) == kDispatchInf);   // sum Pmin above demand
    CHECK(ed.fast({}, 10.0, p) == kDispatchInf);
    REQUIRE(static_cast<int>(p.size()) == in.n_units);
    CHECK(sum_of(p) == 0.0);
}

TEST_CASE("uc dispatch on two units is never below a brute-force scan", "[uc-dispatch]") {
    const auto full = ucp13();
    constexpr double kH = 0.01;
    for (const auto& pair : std::vector<std::vector<int>>{{3, 4}, {1, 9}, {0, 11}}) {
        const auto in = subset_units(full, pair);
        Dispatcher ed(in);
        const std::vector<int> on = {0, 1};
        const double smin = in.P_min[0] + in.P_min[1];
        const double smax = in.P_max[0] + in.P_max[1];
        const double grid_err = (lipschitz(in, 0) + lipschitz(in, 1)) * kH;
        for (int k = 1; k < 10; ++k) {
            const double demand = smin + ((smax - smin) * k / 10.0);
            std::vector<double> p;
            const double fast = ed.fast(on, demand, p);
            const double polished = ed.polish(on, p, 50);
            const double brute = brute_force_two(ed, in, demand, kH);
            // A feasible dispatch cannot beat the optimum; the scan is within
            // grid_err of it.
            CHECK(polished >= brute - grid_err);
            CHECK(polished <= fast + kTol);
            // Polish moves one unit to a valve point; on two units that visits
            // the scan's valve candidates, so it lands on the scan's best.
            CHECK(polished <= brute + grid_err);
        }
    }
}

TEST_CASE("uc dispatch steps down the unit that saves most per MW", "[uc-dispatch]") {
    // Two linear units (c = 0, a negligible valve term) with valve points
    // every 25 MW on [0, 100]. Lambda dispatch at demand 150 lands both at
    // 100, so fast() must step 50 MW down in valve steps. Lowering unit 1
    // (50 $/MW) saves more than unit 0 (10 $/MW); the fix (review Medium 5)
    // steps unit 1 down twice. Before it, unit 0 was stepped down: {50, 100}.
    constexpr double kPi = 3.14159265358979323846;
    UCInstance in = subset_units(ucp13(), {3, 4});
    in.a = {0.0, 0.0};
    in.b = {10.0, 50.0};
    in.c = {0.0, 0.0};
    in.d = {1e-9, 1e-9};
    in.e = {kPi / 25.0, kPi / 25.0};
    in.P_min = {0.0, 0.0};
    in.P_max = {100.0, 100.0};
    Dispatcher ed(in);
    REQUIRE(ed.valves(0).size() == 5);
    std::vector<double> p;
    const double cost = ed.fast({0, 1}, 150.0, p);
    REQUIRE(std::isfinite(cost));
    require_dispatch_valid(in, {0, 1}, 150.0, p);
    CHECK(std::abs(p[0] - 100.0) < 1e-6);
    CHECK(std::abs(p[1] - 50.0) < 1e-6);
}

TEST_CASE("uc repair_row output is row-feasible per the verifier and idempotent",
          "[uc-commitment]") {
    auto in = ucp13();
    // Give units 0 (on before) and 2 (off before) a live initial-condition span.
    in.n_init[0] = 2;  // 6 periods forced on
    in.n_init[2] = 1;  // 4 periods forced off
    RNG rng(11);
    for (int trial = 0; trial < 300; ++trial) {
        Commitment y = random_commitment(in, rng, trial % 3 == 0 ? 0.2 : 0.5);
        for (int u = 0; u < in.n_units; ++u) {
            const CommitRow before = y[u];
            repair_row(in, u, y[u]);
            CHECK(row_feasible(in, u, y[u]));
            const CommitRow once = y[u];
            repair_row(in, u, y[u]);
            CHECK(y[u] == once);
            // Outside the forced span it only switches the unit on.
            int forced = 0;
            if (u == 0) {
                forced = 6;
            } else if (u == 2) {
                forced = 4;
            }
            for (int t = forced; t < in.n_periods; ++t) {
                CHECK(once[t] >= before[t]);
            }
        }
        CHECK(rows_pass_verifier(in, y));
        for (int t = 0; t < 6; ++t) {
            CHECK(y[0][t] == 1);
        }
        for (int t = 0; t < 4; ++t) {
            CHECK(y[2][t] == 0);
        }
    }
}

TEST_CASE("uc row_feasible agrees with the verifier on random rows", "[uc-commitment]") {
    auto in = ucp13();
    in.n_init[0] = 2;
    in.n_init[2] = 1;
    RNG rng(13);
    int feasible = 0;
    int infeasible = 0;
    for (int trial = 0; trial < 300; ++trial) {
        // Rows with long runs, so both outcomes occur.
        Commitment y(in.n_units, CommitRow(in.n_periods, 0));
        for (auto& row : y) {
            uint8_t v = rng.random() < 0.5 ? 1 : 0;
            for (auto& x : row) {
                if (rng.random() < 0.15) {
                    v = static_cast<uint8_t>(1 - v);
                }
                x = v;
            }
        }
        bool all = true;
        for (int u = 0; u < in.n_units; ++u) {
            Commitment one = y;
            for (int w = 0; w < in.n_units; ++w) {
                if (w != u) {
                    // A row that passes alone: held at y_prev throughout.
                    std::fill(one[w].begin(), one[w].end(), static_cast<uint8_t>(in.y_prev[w]));
                }
            }
            const bool mine = row_feasible(in, u, y[u]);
            CHECK(mine == rows_pass_verifier(in, one));
            all = all && mine;
        }
        (all ? feasible : infeasible) += 1;
    }
    CHECK(infeasible > 0);
}

TEST_CASE("uc repair_state yields a commitment the verifier and dispatch accept",
          "[uc-commitment]") {
    const auto in = ucp13();
    Dispatcher ed(in);
    RNG rng(17);
    for (int trial = 0; trial < 100; ++trial) {
        Commitment y = random_commitment(in, rng, trial % 2 == 0 ? 0.1 : 0.5);
        // ucp13: all units on cover every period's demand + reserve, and
        // sum Pmin of all units is below every demand, so repair succeeds.
        REQUIRE(repair_state(in, y));
        CHECK(commitment_feasible(in, y));
        std::vector<std::vector<double>> p_by_period(in.n_periods);
        double fuel = 0.0;
        for (int t = 0; t < in.n_periods; ++t) {
            const auto on = on_set(y, t);
            const double c = ed.fast(on, in.demand[t], p_by_period[t]);
            REQUIRE(std::isfinite(c));
            require_dispatch_valid(in, on, in.demand[t], p_by_period[t]);
            fuel += c;
        }
        VerifyResult r;
        const auto a = to_assignment(in, y, p_by_period);
        verify_detail::check_dispatch_bounds(r, a, in, 1e-4);
        verify_detail::check_demand(r, a, in, 1e-4, /*equality=*/true);
        verify_detail::check_reserve(r, a, in, 1e-4);
        verify_detail::check_min_run(r, a, in, true);
        verify_detail::check_min_run(r, a, in, false);
        verify_detail::check_initial_conditions(r, a, in);
        CHECK(r.ok);
        double startup = 0.0;
        for (int u = 0; u < in.n_units; ++u) {
            startup += startup_cost(in, u, y[u]);
        }
        const double total = verify_detail::recompute_cost(a, in);
        CHECK(std::abs(fuel + startup - total) <= 1e-6 * std::abs(total));
    }
}

TEST_CASE("uc capacity_ok requires demand even when reserve is not positive", "[uc-commitment]") {
    auto in = make_subinstance(ucp13(), 1);
    in.reserve[0] = 0.0;
    Commitment y(in.n_units, CommitRow(1, 0));
    y[10][0] = 1;  // 120 MW against 1250 MW demand
    CHECK_FALSE(capacity_ok(in, y, 0));
    CHECK_FALSE(commitment_feasible(in, y));
    in.reserve[0] = -50.0;
    CHECK_FALSE(capacity_ok(in, y, 0));
    // Repair turns units on until demand is covered (no reserve to add).
    REQUIRE(repair_state(in, y));
    Dispatcher ed(in);
    std::vector<double> p;
    CHECK(std::isfinite(ed.fast(on_set(y, 0), in.demand[0], p)));
}

TEST_CASE("uc capacity_ok rejects a sum Pmin above demand", "[uc-commitment]") {
    auto in = make_subinstance(ucp13(), 1);
    in.demand[0] = 100.0;
    in.reserve[0] = 0.0;
    Commitment y(in.n_units, CommitRow(1, 0));
    y[3][0] = 1;  // Pmin 60
    y[4][0] = 1;  // Pmin 60: 120 > 100
    CHECK_FALSE(capacity_ok(in, y, 0));
    // repair_state only turns units on, so it reports the failure.
    CHECK_FALSE(repair_state(in, y));
}

TEST_CASE("uc repair_state reports failure when capacity cannot be reached", "[uc-commitment]") {
    auto in = make_subinstance(ucp13(), 2);
    in.demand = {5000.0, 1000.0};  // above all 13 units' 2960 MW
    in.reserve = {0.0, 0.0};
    Commitment y(in.n_units, CommitRow(2, 0));
    CHECK_FALSE(repair_state(in, y));
    // The loop ran to the end: everything that can be on at t=0 is on.
    for (int u = 0; u < in.n_units; ++u) {
        CHECK(y[u][0] == 1);
    }
}
