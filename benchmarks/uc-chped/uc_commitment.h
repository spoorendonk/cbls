#pragma once

#include "data.h"

#include <algorithm>
#include <cstdint>
#include <vector>

// Commitment row checks and repair for the opt-in UC hook (#445, #455): pure
// functions of the instance and a commitment y[unit][period].
namespace cbls::uc_chped {

using CommitRow = std::vector<uint8_t>;  // 0 off, 1 on
using Commitment = std::vector<CommitRow>;

/// Exactly the model's commitment rows for unit u (uc_model.h): initial
/// conditions, then min up and min down.
[[nodiscard]] inline bool row_feasible(const UCInstance& in, int u, const CommitRow& r) {
    const int n_periods = in.n_periods;
    if (in.y_prev[u] == 1 || in.y_prev[u] == 0) {
        const int held = in.y_prev[u];
        const int min_run = held == 1 ? in.min_on[u] : in.min_off[u];
        const int rem = std::max(0, min_run - in.n_init[u]);
        for (int t = 0; t < std::min(rem, n_periods); ++t) {
            if (r[t] != held) {
                return false;
            }
        }
    }
    for (int t = 0; t < n_periods; ++t) {
        const int prev = t == 0 ? in.y_prev[u] : r[t - 1];
        if (r[t] == prev) {
            continue;
        }
        const int held = r[t];
        const int run = held == 1 ? in.min_on[u] : in.min_off[u];
        const int end = std::min(t + run, n_periods);
        for (int tau = t + 1; tau < end; ++tau) {
            if (r[tau] != held) {
                return false;
            }
        }
    }
    return true;
}

/// Forward pass making row r feasible for unit u (row_feasible holds after),
/// only ever switching the unit on outside the initial-condition span: that
/// span is forced to its held state; a startup's min_on run is extended; a
/// shutdown whose off run is cut short by a restart inside min_off is
/// cancelled (the gap filled). Idempotent.
inline void repair_row(const UCInstance& in, int u, CommitRow& r) {
    const int n_periods = in.n_periods;
    int forced = 0;
    if (in.y_prev[u] == 1 || in.y_prev[u] == 0) {
        const int held = in.y_prev[u];
        const int min_run = held == 1 ? in.min_on[u] : in.min_off[u];
        forced = std::min(n_periods, std::max(0, min_run - in.n_init[u]));
        std::fill(r.begin(), r.begin() + forced, static_cast<uint8_t>(held));
    }
    for (int t = forced; t < n_periods; ++t) {
        const int prev = t == 0 ? in.y_prev[u] : r[t - 1];
        if (r[t] == prev) {
            continue;
        }
        if (r[t] == 1) {
            const int end = std::min(t + in.min_on[u], n_periods);
            std::fill(r.begin() + t, r.begin() + end, 1);
        } else {
            const int end = std::min(t + in.min_off[u], n_periods);
            int restart = -1;
            for (int tau = t + 1; tau < end && restart < 0; ++tau) {
                if (r[tau] == 1) {
                    restart = tau;
                }
            }
            if (restart >= 0) {
                std::fill(r.begin() + t, r.begin() + restart, 1);
            }
        }
    }
}

/// Period t can be dispatched: committed sum Pmax covers demand plus any
/// positive reserve, and committed sum Pmin does not exceed demand. Exactly
/// when Dispatcher::fast is finite, so a commitment passing it for every t has
/// no +inf period cost (a reserve <= 0 no longer waives demand: #455, review
/// Low 9). Stricter than the `>=` model, which allows sum Pmin > demand.
[[nodiscard]] inline bool capacity_ok(const UCInstance& in, const Commitment& y, int t) {
    double cap = 0.0;
    double floor = 0.0;
    for (int u = 0; u < in.n_units; ++u) {
        if (y[u][t] != 0) {
            cap += in.P_max[u];
            floor += in.P_min[u];
        }
    }
    return cap >= in.demand[t] + std::max(0.0, in.reserve[t]) - 1e-9 &&
           floor <= in.demand[t] + 1e-9;
}

/// Every row feasible and every period dispatchable.
[[nodiscard]] inline bool commitment_feasible(const UCInstance& in, const Commitment& y) {
    for (int u = 0; u < in.n_units; ++u) {
        if (!row_feasible(in, u, y[u])) {
            return false;
        }
    }
    for (int t = 0; t < in.n_periods; ++t) {
        if (!capacity_ok(in, y, t)) {
            return false;
        }
    }
    return true;
}

namespace commitment_detail {

/// Committed sum Pmax in period t.
inline double committed_capacity(const UCInstance& in, const Commitment& y, int t) {
    double cap = 0.0;
    for (int u = 0; u < in.n_units; ++u) {
        cap += in.P_max[u] * y[u][t];
    }
    return cap;
}

/// The idle unit with the largest Pmax that can be on at t with a feasible
/// row, or -1.
inline int largest_startable(const UCInstance& in, const Commitment& y, int t) {
    int best = -1;
    for (int u = 0; u < in.n_units; ++u) {
        if (y[u][t] == 0 && (best < 0 || in.P_max[u] > in.P_max[best])) {
            CommitRow r = y[u];
            r[t] = 1;
            repair_row(in, u, r);
            if (r[t] == 1 && row_feasible(in, u, r)) {
                best = u;
            }
        }
    }
    return best;
}

}  // namespace commitment_detail

/// Each infeasible row repaired, then every period short of demand + reserve
/// gets the largest idle unit that may start there, run for min_on, until
/// covered. Only ever switches units on, so it cannot fix sum Pmin > demand.
/// The capacity loop terminates: each pass turns one more unit on at t, and a
/// repaired row never turns one off. Returns commitment_feasible(in, y).
inline bool repair_state(const UCInstance& in, Commitment& y) {
    for (int u = 0; u < in.n_units; ++u) {
        if (!row_feasible(in, u, y[u])) {
            repair_row(in, u, y[u]);
        }
    }
    for (int t = 0; t < in.n_periods; ++t) {
        const double need = in.demand[t] + std::max(0.0, in.reserve[t]) - 1e-9;
        while (commitment_detail::committed_capacity(in, y, t) < need) {
            const int best = commitment_detail::largest_startable(in, y, t);
            if (best < 0) {
                break;
            }
            y[best][t] = 1;
            repair_row(in, best, y[best]);
        }
    }
    return commitment_feasible(in, y);
}

/// Hot/cold startup cost of row r for unit u: hot when the unit was on in
/// the t_cold periods before the startup (y_prev before the horizon).
[[nodiscard]] inline double startup_cost(const UCInstance& in, int u, const CommitRow& r) {
    double c = 0.0;
    for (int t = 0; t < in.n_periods; ++t) {
        const int prev = t == 0 ? in.y_prev[u] : r[t - 1];
        if (r[t] == 1 && prev == 0) {
            bool hot = false;
            for (int tau = t - in.t_cold[u]; tau < t && !hot; ++tau) {
                hot = (tau < 0 ? in.y_prev[u] : r[tau]) == 1;
            }
            c += hot ? in.a_hot[u] : in.a_cold[u];
        }
    }
    return c;
}

}  // namespace cbls::uc_chped
