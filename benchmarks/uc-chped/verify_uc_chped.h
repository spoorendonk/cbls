#pragma once

#include <cbls/verify.h>
#include "uc_model.h"
#include "data.h"
#include <cmath>
#include <vector>
#include <string>

namespace cbls {
namespace uc_chped {

inline VerifyResult verify_uc_chped(const UCModel& ucm, const UCInstance& inst,
                                     double tol = 1e-4) {
    VerifyResult result = verify_model(ucm.model);

    const auto& m = ucm.model;
    int N = inst.n_units;
    int T = inst.n_periods;

    // Extract variable values (handles are negative: var_id = -(handle + 1))
    auto val = [&](int32_t handle) -> double {
        return m.var(-(handle + 1)).value;
    };
    auto ival = [&](int32_t handle) -> int {
        return (int)std::round(val(handle));
    };

    // Extract all variable values
    std::vector<std::vector<int>> y_val(N, std::vector<int>(T));
    std::vector<std::vector<double>> p_val(N, std::vector<double>(T));
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            y_val[u][t] = ival(ucm.y[u][t]);
            p_val[u][t] = val(ucm.p[u][t]);
        }
    }

    // 1. Commitment integrality: y must be exactly 0 or 1
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            if (y_val[u][t] != 0 && y_val[u][t] != 1) {
                result.add_error({VerifyError::Kind::Custom,
                    "y[" + std::to_string(u) + "][" + std::to_string(t) + "]",
                    0.0, (double)y_val[u][t],
                    "commitment not 0 or 1"});
            }
        }
    }

    // 2. Dispatch bounds: Pmin*y <= p <= Pmax*y
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            double lb = inst.P_min[u] * y_val[u][t];
            double ub = inst.P_max[u] * y_val[u][t];
            if (p_val[u][t] < lb - tol) {
                result.add_error({VerifyError::Kind::Custom,
                    "p[" + std::to_string(u) + "][" + std::to_string(t) + "]",
                    lb, p_val[u][t],
                    "dispatch below Pmin*y"});
            }
            if (p_val[u][t] > ub + tol) {
                result.add_error({VerifyError::Kind::Custom,
                    "p[" + std::to_string(u) + "][" + std::to_string(t) + "]",
                    ub, p_val[u][t],
                    "dispatch above Pmax*y"});
            }
        }
    }

    // 3. Demand balance: sum_u(p[u][t]) >= demand[t]
    for (int t = 0; t < T; ++t) {
        double supply = 0.0;
        for (int u = 0; u < N; ++u) {
            supply += p_val[u][t];
        }
        if (supply < inst.demand[t] - tol) {
            result.add_error({VerifyError::Kind::Custom,
                "demand[" + std::to_string(t) + "]",
                inst.demand[t], supply,
                "supply does not meet demand"});
        }
    }

    // 4. Reserve margin: sum_u(Pmax[u]*y[u][t]) >= demand[t] + reserve[t]
    for (int t = 0; t < T; ++t) {
        if (inst.reserve[t] <= 0) continue;
        double capacity = 0.0;
        for (int u = 0; u < N; ++u) {
            capacity += inst.P_max[u] * y_val[u][t];
        }
        double required = inst.demand[t] + inst.reserve[t];
        if (capacity < required - tol) {
            result.add_error({VerifyError::Kind::Custom,
                "reserve[" + std::to_string(t) + "]",
                required, capacity,
                "committed capacity does not meet demand + reserve"});
        }
    }

    // 5. Min uptime: if y[u][t]=1 and y[u][t-1]=0, then y[u][tau]=1 for tau in [t+1, t+min_on-1]
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            int y_prev = (t == 0) ? inst.y_prev[u] : y_val[u][t - 1];
            if (y_val[u][t] == 1 && y_prev == 0) {
                // Startup at t: must stay on for min_on periods
                int end = std::min(t + inst.min_on[u], T);
                for (int tau = t + 1; tau < end; ++tau) {
                    if (y_val[u][tau] != 1) {
                        result.add_error({VerifyError::Kind::Custom,
                            "y[" + std::to_string(u) + "][" + std::to_string(tau) + "]",
                            1.0, (double)y_val[u][tau],
                            "min uptime violated (startup at t=" + std::to_string(t) +
                            ", min_on=" + std::to_string(inst.min_on[u]) + ")"});
                    }
                }
            }
        }
    }

    // 6. Min downtime: if y[u][t]=0 and y[u][t-1]=1, then y[u][tau]=0 for tau in [t+1, t+min_off-1]
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            int y_prev = (t == 0) ? inst.y_prev[u] : y_val[u][t - 1];
            if (y_val[u][t] == 0 && y_prev == 1) {
                // Shutdown at t: must stay off for min_off periods
                int end = std::min(t + inst.min_off[u], T);
                for (int tau = t + 1; tau < end; ++tau) {
                    if (y_val[u][tau] != 0) {
                        result.add_error({VerifyError::Kind::Custom,
                            "y[" + std::to_string(u) + "][" + std::to_string(tau) + "]",
                            0.0, (double)y_val[u][tau],
                            "min downtime violated (shutdown at t=" + std::to_string(t) +
                            ", min_off=" + std::to_string(inst.min_off[u]) + ")"});
                    }
                }
            }
        }
    }

    // 7. Initial conditions
    for (int u = 0; u < N; ++u) {
        if (inst.y_prev[u] == 1) {
            int remaining = std::max(0, inst.min_on[u] - inst.n_init[u]);
            for (int t = 0; t < std::min(remaining, T); ++t) {
                if (y_val[u][t] != 1) {
                    result.add_error({VerifyError::Kind::Custom,
                        "y[" + std::to_string(u) + "][" + std::to_string(t) + "]",
                        1.0, (double)y_val[u][t],
                        "initial on-condition violated (remaining_on=" + std::to_string(remaining) + ")"});
                }
            }
        }
        if (inst.y_prev[u] == 0) {
            int remaining = std::max(0, inst.min_off[u] - inst.n_init[u]);
            for (int t = 0; t < std::min(remaining, T); ++t) {
                if (y_val[u][t] != 0) {
                    result.add_error({VerifyError::Kind::Custom,
                        "y[" + std::to_string(u) + "][" + std::to_string(t) + "]",
                        0.0, (double)y_val[u][t],
                        "initial off-condition violated (remaining_off=" + std::to_string(remaining) + ")"});
                }
            }
        }
    }

    // 8. Objective recomputation
    double total_cost = 0.0;
    for (int u = 0; u < N; ++u) {
        for (int t = 0; t < T; ++t) {
            if (y_val[u][t] == 0) continue;

            double p = p_val[u][t];
            // Fuel cost: a + b*p + c*p^2 + |d*sin(e*(Pmin-p))|
            double fuel = inst.a[u] + inst.b[u] * p + inst.c[u] * p * p
                        + std::abs(inst.d[u] * std::sin(inst.e[u] * (inst.P_min[u] - p)));
            total_cost += fuel;

            // Startup cost
            int y_prev = (t == 0) ? inst.y_prev[u] : y_val[u][t - 1];
            int su = std::max(0, y_val[u][t] - y_prev);
            if (su > 0) {
                // Look back t_cold periods to determine hot/cold
                bool was_on = false;
                int lookback_start = t - inst.t_cold[u];
                for (int tau = lookback_start; tau < t; ++tau) {
                    if (tau < 0) {
                        if (inst.y_prev[u] == 1) { was_on = true; break; }
                    } else {
                        if (y_val[u][tau] == 1) { was_on = true; break; }
                    }
                }
                total_cost += was_on ? inst.a_hot[u] : inst.a_cold[u];
            }
        }
    }

    double dag_obj = m.node(m.objective_id()).value;
    if (std::abs(dag_obj - total_cost) > std::max(tol * std::abs(total_cost), 1.0)) {
        result.add_error({VerifyError::Kind::ObjectiveMismatch,
            "objective", total_cost, dag_obj,
            "independent cost recomputation mismatch (cost=" + std::to_string(total_cost) + ")"});
    }

    return result;
}

}  // namespace uc_chped
}  // namespace cbls
