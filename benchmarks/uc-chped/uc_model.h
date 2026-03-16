#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <algorithm>

namespace cbls {
namespace uc_chped {

struct UCModel {
    Model model;
    std::vector<std::vector<int32_t>> y;  // [unit][period] commitment (bool var handles)
    std::vector<std::vector<int32_t>> p;  // [unit][period] dispatch (float var handles)
};

inline UCModel build_uc_model(const UCInstance& inst) {
    UCModel result;
    auto& m = result.model;
    int N = inst.n_units;
    int T = inst.n_periods;

    // ---------- Variables ----------
    result.y.resize(N);
    result.p.resize(N);
    for (int u = 0; u < N; ++u) {
        result.y[u].resize(T);
        result.p[u].resize(T);
        for (int t = 0; t < T; ++t) {
            result.y[u][t] = m.bool_var("y_" + std::to_string(u) + "_" + std::to_string(t));
            // Bounds [0, Pmax]: when y=0, dispatch should be 0; Pmin enforced via constraint
            result.p[u][t] = m.float_var(0.0, inst.P_max[u],
                                          "p_" + std::to_string(u) + "_" + std::to_string(t));
        }
    }

    // ---------- Constants ----------
    auto zero = m.constant(0.0);
    auto neg1 = m.constant(-1.0);
    auto half = m.constant(0.5);
    auto neg_half = m.constant(-0.5);
    auto one_const = m.constant(1.0);

    // ---------- Objective: total cost ----------
    std::vector<int32_t> cost_terms;

    for (int u = 0; u < N; ++u) {
        auto ai = m.constant(inst.a[u]);
        auto bi = m.constant(inst.b[u]);
        auto ci = m.constant(inst.c[u]);
        auto di = m.constant(inst.d[u]);
        auto ei = m.constant(inst.e[u]);
        auto pmin_i = m.constant(inst.P_min[u]);
        auto two = m.constant(2.0);
        auto a_hot_c = m.constant(inst.a_hot[u]);
        auto a_cold_c = m.constant(inst.a_cold[u]);

        for (int t = 0; t < T; ++t) {
            auto P = result.p[u][t];
            auto Y = result.y[u][t];

            // --- Fuel cost: y * F(p) ---
            // F(p) = a + b*p + c*p^2 + |d*sin(e*(Pmin-p))|
            auto base_cost = m.sum({ai, m.prod(bi, P), m.prod(ci, m.pow_expr(P, two))});
            auto pmin_minus_p = m.sum({pmin_i, m.prod(neg1, P)});
            auto valve_point = m.abs_expr(m.prod(di, m.sin_expr(m.prod(ei, pmin_minus_p))));
            auto fuel_cost = m.prod(Y, m.sum({base_cost, valve_point}));
            cost_terms.push_back(fuel_cost);

            // --- Startup cost ---
            // Startup detection: su = max(0, y[t] - y_prev)
            int32_t y_prev_h;
            if (t == 0) {
                y_prev_h = m.constant(static_cast<double>(inst.y_prev[u]));
            } else {
                y_prev_h = result.y[u][t - 1];
            }
            auto su = m.max_expr({zero, m.sum({Y, m.prod(neg1, y_prev_h)})});

            // Hot/cold startup cost
            // Look back t_cold periods to see if unit was recently on
            // was_on = max(y[tau] for tau in lookback window)
            // If no lookback window (t_cold=0), always cold
            int lookback_start = (t == 0) ? -inst.t_cold[u] : t - inst.t_cold[u];
            std::vector<int32_t> lookback_ys;
            for (int tau = lookback_start; tau < t; ++tau) {
                if (tau < 0) {
                    // Before horizon: use y_prev if unit was on and n_init covers this
                    // Simplification: if y_prev=1, unit was on before horizon
                    lookback_ys.push_back(m.constant(static_cast<double>(inst.y_prev[u])));
                } else {
                    lookback_ys.push_back(result.y[u][tau]);
                }
            }

            int32_t startup_cost;
            if (lookback_ys.empty()) {
                // No lookback → always cold startup
                startup_cost = m.prod(a_cold_c, su);
            } else {
                auto was_on = m.max_expr(lookback_ys);
                // if_then_else(cond, then, else): cond > 0 → then, else → else
                // We want: was_on > 0.5 → hot, else → cold
                auto cond = m.sum({was_on, neg_half});
                startup_cost = m.if_then_else(cond, m.prod(a_hot_c, su), m.prod(a_cold_c, su));
            }
            cost_terms.push_back(startup_cost);
        }
    }

    m.minimize(m.sum(cost_terms));

    // ---------- Constraints ----------

    for (int t = 0; t < T; ++t) {
        // --- Demand constraint: demand[t] - sum(p[u][t]) <= 0 ---
        std::vector<int32_t> supply_terms;
        for (int u = 0; u < N; ++u) {
            supply_terms.push_back(result.p[u][t]);
        }
        auto supply = m.sum(supply_terms);
        auto demand_t = m.constant(inst.demand[t]);
        m.add_constraint(m.sum({demand_t, m.prod(neg1, supply)}));

        // --- Reserve constraint: (demand[t] + reserve[t]) - sum(Pmax[u]*y[u][t]) <= 0 ---
        if (inst.reserve[t] > 0) {
            std::vector<int32_t> cap_terms;
            for (int u = 0; u < N; ++u) {
                cap_terms.push_back(m.prod(m.constant(inst.P_max[u]), result.y[u][t]));
            }
            auto capacity = m.sum(cap_terms);
            auto reserve_t = m.constant(inst.reserve[t]);
            m.add_constraint(m.sum({demand_t, reserve_t, m.prod(neg1, capacity)}));
        }
    }

    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            auto Y = result.y[u][t];
            auto P = result.p[u][t];

            // --- Dispatch lower bound: Pmin*y - p <= 0 ---
            auto pmin_c = m.constant(inst.P_min[u]);
            m.add_constraint(m.sum({m.prod(pmin_c, Y), m.prod(neg1, P)}));

            // --- Dispatch upper bound: p - Pmax*y <= 0 ---
            auto pmax_c = m.constant(inst.P_max[u]);
            m.add_constraint(m.sum({P, m.prod(neg1, m.prod(pmax_c, Y))}));
        }

        // --- Min uptime constraints ---
        // If unit starts up at t (y[t]=1, y[t-1]=0), must stay on for min_on periods
        // For each pair (t, tau) where tau in [t+1, min(t+min_on-1, T-1)]:
        //   y[t] - y_prev - y[tau] <= 0  (if unit turned on at t, it must be on at tau)
        // Equivalently: y[t] - y[t-1] <= y[tau]
        for (int t = 0; t < T; ++t) {
            int32_t y_prev_h;
            if (t == 0) {
                y_prev_h = m.constant(static_cast<double>(inst.y_prev[u]));
            } else {
                y_prev_h = result.y[u][t - 1];
            }

            int end = std::min(t + inst.min_on[u], T);
            for (int tau = t + 1; tau < end; ++tau) {
                // y[t] - y_prev - y[tau] <= 0
                m.add_constraint(m.sum({result.y[u][t], m.prod(neg1, y_prev_h),
                                        m.prod(neg1, result.y[u][tau])}));
            }
        }

        // --- Min downtime constraints ---
        // If unit shuts down at t (y[t]=0, y[t-1]=1), must stay off for min_off periods
        // For each pair (t, tau) where tau in [t+1, min(t+min_off-1, T-1)]:
        //   y_prev - y[t] + y[tau] - 1 <= 0
        for (int t = 0; t < T; ++t) {
            int32_t y_prev_h;
            if (t == 0) {
                y_prev_h = m.constant(static_cast<double>(inst.y_prev[u]));
            } else {
                y_prev_h = result.y[u][t - 1];
            }

            int end = std::min(t + inst.min_off[u], T);
            for (int tau = t + 1; tau < end; ++tau) {
                // y_prev - y[t] + y[tau] - 1 <= 0
                m.add_constraint(m.sum({y_prev_h, m.prod(neg1, result.y[u][t]),
                                        result.y[u][tau], neg1}));
            }
        }

        // --- Initial conditions ---
        // If unit was ON (y_prev=1) and hasn't satisfied min_on yet,
        // force it on for the remaining periods
        if (inst.y_prev[u] == 1) {
            int remaining_on = std::max(0, inst.min_on[u] - inst.n_init[u]);
            for (int t = 0; t < std::min(remaining_on, T); ++t) {
                // y[t] >= 1  →  1 - y[t] <= 0
                m.add_constraint(m.sum({one_const, m.prod(neg1, result.y[u][t])}));
            }
        }
        // If unit was OFF (y_prev=0) and hasn't satisfied min_off yet,
        // force it off for the remaining periods
        if (inst.y_prev[u] == 0) {
            int remaining_off = std::max(0, inst.min_off[u] - inst.n_init[u]);
            for (int t = 0; t < std::min(remaining_off, T); ++t) {
                // y[t] <= 0  →  y[t] <= 0
                m.add_constraint(result.y[u][t]);
            }
        }
    }

    m.close();
    return result;
}

}  // namespace uc_chped
}  // namespace cbls
