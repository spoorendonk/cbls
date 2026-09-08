#pragma once

#include "data.h"
#include "uc_model.h"

#include <cbls/verify.h>
#include <cmath>
#include <string>
#include <vector>

namespace cbls::uc_chped {

// The verifier is the independent re-check of a published UC-CHPED row, so it is
// split along the constraint families of the formulation rather than into
// arbitrary halves: each helper below owns exactly one family, checks it against
// the instance data, and appends its own errors. They are called in the same
// order the families are numbered, which is the order the diagnostics come out
// in.
namespace verify_detail {

/// The assignment being checked, extracted once from the DAG so that every
/// family below reads plain values rather than variable handles.
struct Assignment {
    std::vector<std::vector<int>> y;     // [unit][period] commitment
    std::vector<std::vector<double>> p;  // [unit][period] dispatch
};

inline Assignment extract_assignment(const UCModel& ucm, const UCInstance& inst) {
    const auto& m = ucm.model;
    // Handles are negative: var_id = -(handle + 1)
    auto val = [&](int32_t handle) -> double { return m.var(-(handle + 1)).value; };

    Assignment a;
    a.y.assign(inst.n_units, std::vector<int>(inst.n_periods));
    a.p.assign(inst.n_units, std::vector<double>(inst.n_periods));
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            a.y[u][t] = static_cast<int>(std::round(val(ucm.y[u][t])));
            a.p[u][t] = val(ucm.p[u][t]);
        }
    }
    return a;
}

/// Commitment in the period before t, taken from the instance's initial state at
/// t == 0. Shared by every family that reasons about a transition.
inline int commitment_before(const Assignment& a, const UCInstance& inst, int u, int t) {
    return (t == 0) ? inst.y_prev[u] : a.y[u][t - 1];
}

inline std::string cell_name(const char* var, int u, int t) {
    return std::string(var) + "[" + std::to_string(u) + "][" + std::to_string(t) + "]";
}

/// 1. Commitment integrality: y must be exactly 0 or 1.
inline void check_integrality(VerifyResult& result, const Assignment& a, const UCInstance& inst) {
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            if (a.y[u][t] != 0 && a.y[u][t] != 1) {
                result.add_error({VerifyError::Kind::Custom, cell_name("y", u, t), 0.0,
                                  static_cast<double>(a.y[u][t]), "commitment not 0 or 1"});
            }
        }
    }
}

/// 2. Dispatch bounds: Pmin*y <= p <= Pmax*y.
inline void check_dispatch_bounds(VerifyResult& result, const Assignment& a, const UCInstance& inst,
                                  double tol) {
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            double lb = inst.P_min[u] * a.y[u][t];
            double ub = inst.P_max[u] * a.y[u][t];
            if (a.p[u][t] < lb - tol) {
                result.add_error({VerifyError::Kind::Custom, cell_name("p", u, t), lb, a.p[u][t],
                                  "dispatch below Pmin*y"});
            }
            if (a.p[u][t] > ub + tol) {
                result.add_error({VerifyError::Kind::Custom, cell_name("p", u, t), ub, a.p[u][t],
                                  "dispatch above Pmax*y"});
            }
        }
    }
}

/// 3. Demand balance: sum_u(p[u][t]) >= demand[t].
inline void check_demand(VerifyResult& result, const Assignment& a, const UCInstance& inst,
                         double tol) {
    for (int t = 0; t < inst.n_periods; ++t) {
        double supply = 0.0;
        for (int u = 0; u < inst.n_units; ++u) {
            supply += a.p[u][t];
        }
        if (supply < inst.demand[t] - tol) {
            result.add_error({VerifyError::Kind::Custom, "demand[" + std::to_string(t) + "]",
                              inst.demand[t], supply, "supply does not meet demand"});
        }
    }
}

/// 4. Reserve margin: sum_u(Pmax[u]*y[u][t]) >= demand[t] + reserve[t].
inline void check_reserve(VerifyResult& result, const Assignment& a, const UCInstance& inst,
                          double tol) {
    for (int t = 0; t < inst.n_periods; ++t) {
        if (inst.reserve[t] <= 0) {
            continue;
        }
        double capacity = 0.0;
        for (int u = 0; u < inst.n_units; ++u) {
            capacity += inst.P_max[u] * a.y[u][t];
        }
        double required = inst.demand[t] + inst.reserve[t];
        if (capacity < required - tol) {
            result.add_error({VerifyError::Kind::Custom, "reserve[" + std::to_string(t) + "]",
                              required, capacity,
                              "committed capacity does not meet demand + reserve"});
        }
    }
}

/// 5/6. Minimum up- and down-time. The two families differ only in which
/// transition starts the block and which value the block must hold, so they
/// share one implementation rather than two copies that can drift apart:
/// `on_block` selects a startup (y goes 0 -> 1, must stay on for min_on) or a
/// shutdown (y goes 1 -> 0, must stay off for min_off).
inline void check_min_run(VerifyResult& result, const Assignment& a, const UCInstance& inst,
                          bool on_block) {
    const int held = on_block ? 1 : 0;
    const int before = on_block ? 0 : 1;
    const std::vector<int>& min_run = on_block ? inst.min_on : inst.min_off;
    const char* what =
        on_block ? "min uptime violated (startup at t=" : "min downtime violated (shutdown at t=";
    const char* limit_name = on_block ? ", min_on=" : ", min_off=";
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            if (a.y[u][t] != held || commitment_before(a, inst, u, t) != before) {
                continue;
            }
            int end = std::min(t + min_run[u], inst.n_periods);
            for (int tau = t + 1; tau < end; ++tau) {
                if (a.y[u][tau] != held) {
                    result.add_error(
                        {VerifyError::Kind::Custom, cell_name("y", u, tau),
                         static_cast<double>(held), static_cast<double>(a.y[u][tau]),
                         what + std::to_string(t) + limit_name + std::to_string(min_run[u]) + ")"});
                }
            }
        }
    }
}

/// 7. Initial conditions: a unit that has not yet served out its minimum run at
/// the start of the horizon is pinned for the remainder of it.
inline void check_initial_conditions(VerifyResult& result, const Assignment& a,
                                     const UCInstance& inst) {
    for (int u = 0; u < inst.n_units; ++u) {
        const bool was_on = inst.y_prev[u] == 1;
        if (!was_on && inst.y_prev[u] != 0) {
            continue;
        }
        const int held = was_on ? 1 : 0;
        const int min_run = was_on ? inst.min_on[u] : inst.min_off[u];
        const char* what = was_on ? "initial on-condition violated (remaining_on="
                                  : "initial off-condition violated (remaining_off=";
        int remaining = std::max(0, min_run - inst.n_init[u]);
        for (int t = 0; t < std::min(remaining, inst.n_periods); ++t) {
            if (a.y[u][t] != held) {
                result.add_error({VerifyError::Kind::Custom, cell_name("y", u, t),
                                  static_cast<double>(held), static_cast<double>(a.y[u][t]),
                                  what + std::to_string(remaining) + ")"});
            }
        }
    }
}

/// Whether the startup at period t is a hot one: the unit ran at some point in
/// the t_cold periods before it.
inline bool was_recently_on(const Assignment& a, const UCInstance& inst, int u, int t) {
    for (int tau = t - inst.t_cold[u]; tau < t; ++tau) {
        const int y = (tau < 0) ? inst.y_prev[u] : a.y[u][tau];
        if (y == 1) {
            return true;
        }
    }
    return false;
}

/// 8. Objective recomputation, independent of the DAG: fuel cost with the
/// valve-point term plus hot/cold startup cost.
inline double recompute_cost(const Assignment& a, const UCInstance& inst) {
    double total_cost = 0.0;
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            if (a.y[u][t] == 0) {
                continue;
            }
            double p = a.p[u][t];
            // Fuel cost: a + b*p + c*p^2 + |d*sin(e*(Pmin-p))|
            total_cost += inst.a[u] + (inst.b[u] * p) + (inst.c[u] * p * p) +
                          std::abs(inst.d[u] * std::sin(inst.e[u] * (inst.P_min[u] - p)));

            int su = std::max(0, a.y[u][t] - commitment_before(a, inst, u, t));
            if (su > 0) {
                total_cost += was_recently_on(a, inst, u, t) ? inst.a_hot[u] : inst.a_cold[u];
            }
        }
    }
    return total_cost;
}

}  // namespace verify_detail

inline VerifyResult verify_uc_chped(const UCModel& ucm, const UCInstance& inst, double tol = 1e-4) {
    VerifyResult result = verify_model(ucm.model);
    const verify_detail::Assignment a = verify_detail::extract_assignment(ucm, inst);

    verify_detail::check_integrality(result, a, inst);
    verify_detail::check_dispatch_bounds(result, a, inst, tol);
    verify_detail::check_demand(result, a, inst, tol);
    verify_detail::check_reserve(result, a, inst, tol);
    verify_detail::check_min_run(result, a, inst, /*on_block=*/true);
    verify_detail::check_min_run(result, a, inst, /*on_block=*/false);
    verify_detail::check_initial_conditions(result, a, inst);

    const double total_cost = verify_detail::recompute_cost(a, inst);
    const double dag_obj = ucm.model.node(ucm.model.objective_id()).value;
    if (std::abs(dag_obj - total_cost) > std::max(tol * std::abs(total_cost), 1.0)) {
        result.add_error(
            {VerifyError::Kind::ObjectiveMismatch, "objective", total_cost, dag_obj,
             "independent cost recomputation mismatch (cost=" + std::to_string(total_cost) + ")"});
    }

    return result;
}

}  // namespace cbls::uc_chped
