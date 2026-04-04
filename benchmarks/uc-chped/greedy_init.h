#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include "uc_model.h"
#include <algorithm>
#include <numeric>
#include <vector>

namespace cbls {
namespace uc_chped {

// Greedy commitment initialization for UC-CHPED.
// For each period, commits cheapest units to meet demand+reserve,
// respecting min up/down times by committing in blocks.
// Sets dispatch to proportional share of demand among committed units.
inline void greedy_uc_initialize(Model& model, const UCInstance& inst,
                                  const UCModel& ucm) {
    int N = inst.n_units;
    int T = inst.n_periods;

    // Commitment state: y[u][t]
    std::vector<std::vector<int>> y(N, std::vector<int>(T, 0));
    // Dispatch state: p[u][t]
    std::vector<std::vector<double>> p(N, std::vector<double>(T, 0.0));

    // Sort units by average cost per MW at Pmax (cheapest first)
    // F(Pmax)/Pmax gives a better ranking than F(Pmin) for large systems
    std::vector<int> unit_order(N);
    std::iota(unit_order.begin(), unit_order.end(), 0);
    std::sort(unit_order.begin(), unit_order.end(), [&](int a, int b) {
        double Pa = inst.P_max[a];
        double cost_a = (inst.a[a] + inst.b[a] * Pa + inst.c[a] * Pa * Pa) / Pa;
        double Pb = inst.P_max[b];
        double cost_b = (inst.a[b] + inst.b[b] * Pb + inst.c[b] * Pb * Pb) / Pb;
        return cost_a < cost_b;
    });

    // Track how long each unit has been continuously on/off
    // Positive = consecutive ON periods, negative = consecutive OFF periods
    std::vector<int> run_length(N);
    for (int u = 0; u < N; ++u) {
        if (inst.y_prev[u] == 1) {
            run_length[u] = inst.n_init[u];  // was ON for this many periods
        } else {
            run_length[u] = -inst.n_init[u]; // was OFF for this many periods
        }
    }

    // Check if unit u can be turned ON at period t
    auto can_turn_on = [&](int u, int t) -> bool {
        if (y[u][t] == 1) return true;  // already on
        // Must have been off for at least min_off periods
        if (run_length[u] < 0 && -run_length[u] < inst.min_off[u]) return false;
        // Must have room for min_on consecutive periods
        // (soft check — we'll try even if horizon is short)
        return true;
    };

    // Apply initial conditions: forced on/off for early periods
    for (int u = 0; u < N; ++u) {
        if (inst.y_prev[u] == 1) {
            int remaining = std::max(0, inst.min_on[u] - inst.n_init[u]);
            for (int t = 0; t < std::min(remaining, T); ++t) {
                y[u][t] = 1;
            }
        }
        if (inst.y_prev[u] == 0) {
            int remaining = std::max(0, inst.min_off[u] - inst.n_init[u]);
            for (int t = 0; t < std::min(remaining, T); ++t) {
                y[u][t] = 0;  // forced off (already 0, but explicit)
            }
        }
    }

    // Greedy commitment: for each period, commit cheapest available units
    for (int t = 0; t < T; ++t) {
        // Update run lengths from previous period
        if (t > 0) {
            for (int u = 0; u < N; ++u) {
                if (y[u][t - 1] == 1) {
                    run_length[u] = (run_length[u] > 0) ? run_length[u] + 1 : 1;
                } else {
                    run_length[u] = (run_length[u] < 0) ? run_length[u] - 1 : -1;
                }
            }
        }

        // Calculate capacity from already-committed units (forced by min_on)
        double committed_capacity = 0.0;
        for (int u = 0; u < N; ++u) {
            if (y[u][t] == 1) {
                committed_capacity += inst.P_max[u];
            }
        }

        double target = inst.demand[t] + inst.reserve[t];

        // Commit additional units in merit order until target is met
        if (committed_capacity < target) {
            for (int u : unit_order) {
                if (y[u][t] == 1) continue;  // already committed
                if (!can_turn_on(u, t)) continue;

                // Commit this unit for min_on consecutive periods
                int block_end = std::min(t + inst.min_on[u], T);
                for (int tau = t; tau < block_end; ++tau) {
                    y[u][tau] = 1;
                }

                committed_capacity += inst.P_max[u];
                if (committed_capacity >= target) break;
            }
        }
    }

    // Set dispatch: proportional share of demand among committed units
    for (int t = 0; t < T; ++t) {
        double total_cap = 0.0;
        std::vector<int> on_units;
        for (int u = 0; u < N; ++u) {
            if (y[u][t] == 1) {
                total_cap += inst.P_max[u] - inst.P_min[u];
                on_units.push_back(u);
            }
        }

        // First assign Pmin to all on units
        double remaining_demand = inst.demand[t];
        for (int u : on_units) {
            p[u][t] = inst.P_min[u];
            remaining_demand -= inst.P_min[u];
        }

        // Distribute remaining demand proportionally to available range
        if (remaining_demand > 0 && total_cap > 0) {
            for (int u : on_units) {
                double range = inst.P_max[u] - inst.P_min[u];
                double share = range / total_cap * remaining_demand;
                p[u][t] = std::min(p[u][t] + share, inst.P_max[u]);
            }
        }
    }

    // Write to model variables (handles are negative-encoded: var_id = -(h+1))
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            int32_t yid = -(ucm.y[u][t] + 1);
            int32_t pid = -(ucm.p[u][t] + 1);
            model.var_mut(yid).value = static_cast<double>(y[u][t]);
            model.var_mut(pid).value = p[u][t];
        }
    }

    full_evaluate(model);
}

}  // namespace uc_chped
}  // namespace cbls
