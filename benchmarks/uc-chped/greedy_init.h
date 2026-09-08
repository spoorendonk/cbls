#pragma once

#include "data.h"
#include "uc_model.h"

#include <algorithm>
#include <cbls/cbls.h>
#include <numeric>
#include <vector>

namespace cbls::uc_chped {

// The greedy warm start is five decisions taken in sequence, each of which can
// be read, changed and reasoned about on its own: which units are cheap, which
// commitments the initial state already forces, which further units to commit
// per period, how to spread demand over what is committed, and how to write the
// result back into the model. They are separate functions for that reason, not
// to shorten a body.
namespace greedy_detail {

/// Unit indices cheapest-first by average cost per MW at Pmax. F(Pmax)/Pmax
/// ranks large systems better than F(Pmin) does.
inline std::vector<int> merit_order(const UCInstance& inst) {
    auto unit_cost = [&](int u) {
        double pmax = inst.P_max[u];
        return (inst.a[u] + (inst.b[u] * pmax) + (inst.c[u] * pmax * pmax)) / pmax;
    };
    std::vector<int> order(inst.n_units);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return unit_cost(a) < unit_cost(b); });
    return order;
}

/// How long each unit has been continuously on or off entering the horizon.
/// Positive = consecutive ON periods, negative = consecutive OFF periods.
inline std::vector<int> initial_run_lengths(const UCInstance& inst) {
    std::vector<int> run_length(inst.n_units);
    for (int u = 0; u < inst.n_units; ++u) {
        run_length[u] = (inst.y_prev[u] == 1) ? inst.n_init[u] : -inst.n_init[u];
    }
    return run_length;
}

/// Commitments the initial state forces: a unit that has not yet served out its
/// minimum run is pinned for the remainder of it. The forced-off case writes the
/// zero the vector already holds, kept explicit because it is a decision.
inline void apply_initial_conditions(const UCInstance& inst, std::vector<std::vector<int>>& y) {
    for (int u = 0; u < inst.n_units; ++u) {
        if (inst.y_prev[u] != 1 && inst.y_prev[u] != 0) {
            continue;
        }
        const int held = inst.y_prev[u];
        const int min_run = (held == 1) ? inst.min_on[u] : inst.min_off[u];
        const int remaining = std::max(0, min_run - inst.n_init[u]);
        for (int t = 0; t < std::min(remaining, inst.n_periods); ++t) {
            y[u][t] = held;
        }
    }
}

/// Whether unit u may be switched on at period t: already-on units qualify, and
/// an off unit must have served its minimum downtime. Having room for min_on
/// consecutive periods is deliberately not required -- a short horizon should
/// still get a commitment, and the constraint model will price the shortfall.
inline bool can_turn_on(const UCInstance& inst, const std::vector<std::vector<int>>& y,
                        const std::vector<int>& run_length, int u, int t) {
    if (y[u][t] == 1) {
        return true;
    }
    return run_length[u] >= 0 || -run_length[u] >= inst.min_off[u];
}

/// Advances the on/off run lengths across the boundary into period t.
inline void advance_run_lengths(const std::vector<std::vector<int>>& y,
                                std::vector<int>& run_length, int t) {
    for (size_t u = 0; u < run_length.size(); ++u) {
        if (y[u][t - 1] == 1) {
            run_length[u] = (run_length[u] > 0) ? run_length[u] + 1 : 1;
        } else {
            run_length[u] = (run_length[u] < 0) ? run_length[u] - 1 : -1;
        }
    }
}

/// Period-by-period commitment: top up the capacity the initial conditions
/// already force with the cheapest eligible units until demand + reserve is
/// covered, committing each in a min_on-long block so the block constraints are
/// satisfied by construction.
inline void greedy_commitment(const UCInstance& inst, const std::vector<int>& unit_order,
                              std::vector<int>& run_length, std::vector<std::vector<int>>& y) {
    for (int t = 0; t < inst.n_periods; ++t) {
        if (t > 0) {
            advance_run_lengths(y, run_length, t);
        }

        double committed_capacity = 0.0;
        for (int u = 0; u < inst.n_units; ++u) {
            if (y[u][t] == 1) {
                committed_capacity += inst.P_max[u];
            }
        }
        const double target = inst.demand[t] + inst.reserve[t];

        for (int u : unit_order) {
            if (committed_capacity >= target) {
                break;
            }
            if (y[u][t] == 1 || !can_turn_on(inst, y, run_length, u, t)) {
                continue;
            }
            const int block_end = std::min(t + inst.min_on[u], inst.n_periods);
            for (int tau = t; tau < block_end; ++tau) {
                y[u][tau] = 1;
            }
            committed_capacity += inst.P_max[u];
        }
    }
}

/// Dispatch: every committed unit gets Pmin, and whatever demand is left over is
/// split across the committed units in proportion to their remaining range.
inline void proportional_dispatch(const UCInstance& inst, const std::vector<std::vector<int>>& y,
                                  std::vector<std::vector<double>>& p) {
    for (int t = 0; t < inst.n_periods; ++t) {
        double total_cap = 0.0;
        std::vector<int> on_units;
        double remaining_demand = inst.demand[t];
        for (int u = 0; u < inst.n_units; ++u) {
            if (y[u][t] != 1) {
                continue;
            }
            total_cap += inst.P_max[u] - inst.P_min[u];
            on_units.push_back(u);
            p[u][t] = inst.P_min[u];
            remaining_demand -= inst.P_min[u];
        }

        if (remaining_demand > 0 && total_cap > 0) {
            for (int u : on_units) {
                double share = (inst.P_max[u] - inst.P_min[u]) / total_cap * remaining_demand;
                p[u][t] = std::min(p[u][t] + share, inst.P_max[u]);
            }
        }
    }
}

/// Writes the assignment into the model's variables (handles are
/// negative-encoded: var_id = -(h+1)).
inline void write_assignment(Model& model, const UCModel& ucm, const UCInstance& inst,
                             const std::vector<std::vector<int>>& y,
                             const std::vector<std::vector<double>>& p) {
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            model.var_mut(-(ucm.y[u][t] + 1)).value = static_cast<double>(y[u][t]);
            model.var_mut(-(ucm.p[u][t] + 1)).value = p[u][t];
        }
    }
}

}  // namespace greedy_detail

// Greedy commitment initialization for UC-CHPED.
// For each period, commits cheapest units to meet demand+reserve,
// respecting min up/down times by committing in blocks.
// Sets dispatch to proportional share of demand among committed units.
inline void greedy_uc_initialize(Model& model, const UCInstance& inst, const UCModel& ucm) {
    std::vector<std::vector<int>> y(inst.n_units, std::vector<int>(inst.n_periods, 0));
    std::vector<std::vector<double>> p(inst.n_units, std::vector<double>(inst.n_periods, 0.0));

    const std::vector<int> unit_order = greedy_detail::merit_order(inst);
    std::vector<int> run_length = greedy_detail::initial_run_lengths(inst);

    greedy_detail::apply_initial_conditions(inst, y);
    greedy_detail::greedy_commitment(inst, unit_order, run_length, y);
    greedy_detail::proportional_dispatch(inst, y, p);
    greedy_detail::write_assignment(model, ucm, inst, y, p);

    full_evaluate(model);
}

}  // namespace cbls::uc_chped
