#pragma once

#include "data.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace cbls {
namespace nuclear_outage {

// Outage lookup table: [plant_idx][cycle] -> start_week, or -1 if unscheduled.
// Replaces O(n) linear scans with O(1) lookups.
using OutageLookup = std::vector<std::vector<int>>;

inline OutageLookup build_outage_lookup(const ROADEFInstance& inst,
                                        const std::vector<int>& ha) {
    int max_cycles = 0;
    for (int i = 0; i < inst.n_type2; ++i) {
        max_cycles = std::max(max_cycles, inst.type2_plants[i].n_cycles);
    }
    OutageLookup lookup(inst.n_type2, std::vector<int>(max_cycles, -1));
    for (int o = 0; o < inst.n_outages(); ++o) {
        lookup[inst.ct13[o].plant_idx][inst.ct13[o].cycle] = ha[o];
    }
    return lookup;
}

// Per-plant status arrays for ROADEF dispatch simulation.
struct PlantStatus {
    std::vector<std::vector<int>> cycle_at_week;   // [plant][week]
    std::vector<std::vector<bool>> in_outage;      // [plant][week]
};

inline PlantStatus compute_plant_status(const ROADEFInstance& inst,
                                        const OutageLookup& lookup) {
    PlantStatus status;
    status.cycle_at_week.assign(inst.n_type2, std::vector<int>(inst.H, -1));
    status.in_outage.assign(inst.n_type2, std::vector<bool>(inst.H, false));

    for (int i = 0; i < inst.n_type2; ++i) {
        auto& plant = inst.type2_plants[i];

        struct OutageEvent { int start; int cycle; int duration; };
        std::vector<OutageEvent> events;
        for (int k = 0; k < plant.n_cycles; ++k) {
            int start = lookup[i][k];
            if (start >= 0) {
                events.push_back({start, k, plant.durations[k]});
            }
        }
        std::sort(events.begin(), events.end(),
                  [](const OutageEvent& lhs, const OutageEvent& rhs) {
                      return lhs.start < rhs.start;
                  });

        int next_event = 0;
        int current_cycle = -1;  // -1 = current campaign before first outage

        for (int h = 0; h < inst.H; ++h) {
            if (next_event < (int)events.size() && h == events[next_event].start) {
                int ev_cycle = events[next_event].cycle;
                int ev_dur = events[next_event].duration;
                for (int w = h; w < std::min(h + ev_dur, inst.H); ++w) {
                    status.in_outage[i][w] = true;
                    status.cycle_at_week[i][w] = ev_cycle;
                }
                current_cycle = ev_cycle;
                h = std::min(h + ev_dur - 1, inst.H - 1);
                ++next_event;
            } else {
                status.in_outage[i][h] = false;
                status.cycle_at_week[i][h] = current_cycle;
            }
        }
    }
    return status;
}

// Compute reload amounts for each Type 2 plant/cycle.
inline std::vector<std::vector<double>> compute_reloads(
    const ROADEFInstance& inst,
    const OutageLookup& lookup) {
    std::vector<std::vector<double>> reload(inst.n_type2);
    for (int i = 0; i < inst.n_type2; ++i) {
        auto& plant = inst.type2_plants[i];
        reload[i].resize(plant.n_cycles, 0.0);
        for (int k = 0; k < plant.n_cycles; ++k) {
            if (lookup[i][k] >= 0) {
                reload[i][k] = plant.rmax[k];
            }
        }
    }
    return reload;
}

// Result of simulating one scenario.
struct ScenarioResult {
    double cost = 0.0;
    // Per-timestep production (only filled when record_production=true)
    std::vector<std::vector<double>> t1_prod;   // [type1_plant][timestep]
    std::vector<std::vector<double>> t2_prod;   // [type2_plant][timestep]
    std::vector<std::vector<double>> fuel_at_t; // [type2_plant][timestep+1]
};

// Simulate fuel dynamics + dispatch for one scenario.
// record_production=true fills per-timestep arrays (for solution output).
inline ScenarioResult simulate_scenario(
    const ROADEFInstance& inst,
    const PlantStatus& status,
    const std::vector<std::vector<double>>& reload,
    int scenario,
    bool record_production = false) {

    ScenarioResult result;
    double total_cost = 0.0;
    int tpw = inst.timesteps_per_week;

    std::vector<double> fuel(inst.n_type2);
    for (int i = 0; i < inst.n_type2; ++i) {
        fuel[i] = inst.type2_plants[i].initial_stock;
    }

    // Scratch space for per-timestep Type 2 production
    std::vector<double> t2_prod_step(inst.n_type2);

    if (record_production) {
        result.t1_prod.assign(inst.n_type1, std::vector<double>(inst.T, 0.0));
        result.t2_prod.assign(inst.n_type2, std::vector<double>(inst.T, 0.0));
        result.fuel_at_t.assign(inst.n_type2, std::vector<double>(inst.T + 1, 0.0));
        for (int i = 0; i < inst.n_type2; ++i) {
            result.fuel_at_t[i][0] = fuel[i];
        }
    }

    for (int t = 0; t < inst.T; ++t) {
        int week = t / tpw;
        double dt = inst.timestep_durations[t];
        double demand = inst.demand[scenario][t];

        double total_t2 = 0.0;
        for (int i = 0; i < inst.n_type2; ++i) {
            auto& plant = inst.type2_plants[i];

            if (week >= inst.H || status.in_outage[i][week]) {
                // CT3: offline during outage
                t2_prod_step[i] = 0.0;

                // CT10: refueling at first timestep of outage start
                bool prev_in_outage = (week > 0) ? status.in_outage[i][week - 1] : false;
                if (status.in_outage[i][week] && (t % tpw) == 0 && !prev_in_outage) {
                    int k = status.cycle_at_week[i][week];
                    if (k >= 0 && k < plant.n_cycles) {
                        double q_k = plant.q[k];
                        double bo_prev = (k > 0) ? plant.bo[k] : plant.bo[0];
                        double bo_k = plant.bo[k + 1];
                        fuel[i] = ((q_k - 1.0) / q_k) * (fuel[i] - bo_prev) +
                                  reload[i][k] + bo_k;
                        fuel[i] = std::min(fuel[i], plant.smax[k]);
                    }
                }
            } else {
                // Plant is online — determine max production
                int cycle_k = status.cycle_at_week[i][week];
                int prof_idx = (cycle_k < 0) ? 0 : (cycle_k + 1);
                double bo = plant.bo[prof_idx];
                double pmax_t = plant.pmax_t[t];

                double max_prod;
                if (fuel[i] >= bo) {
                    max_prod = pmax_t;
                } else {
                    auto& prof = plant.profiles[prof_idx];
                    double pb = prof.evaluate(fuel[i]);
                    max_prod = pb * pmax_t;
                    if (fuel[i] < pb * pmax_t * dt) {
                        max_prod = std::max(0.0, fuel[i] / dt);
                    }
                }

                double prod = std::max(0.0, max_prod);
                prod = std::min(prod, std::max(0.0, fuel[i] / dt));
                t2_prod_step[i] = prod;
                total_t2 += prod;

                fuel[i] -= prod * dt;
                fuel[i] = std::max(0.0, fuel[i]);
            }

            if (record_production) {
                result.t2_prod[i][t] = t2_prod_step[i];
                result.fuel_at_t[i][t + 1] = fuel[i];
            }
        }

        // Type 1 dispatch: fill remaining demand via merit order
        double remaining = demand - total_t2;

        if (remaining > 0.0 && inst.n_type1 > 0) {
            std::vector<int> order(inst.n_type1);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(),
                      [&](int lhs, int rhs) {
                          return inst.type1_plants[lhs].cost[scenario][t] <
                                 inst.type1_plants[rhs].cost[scenario][t];
                      });

            for (int j : order) {
                if (remaining <= 1e-6) break;
                auto& p1 = inst.type1_plants[j];
                double pmin_j = p1.pmin[scenario][t];
                double pmax_j = p1.pmax[scenario][t];
                if (pmax_j <= 0.0) continue;

                // Plant must run at >= pmin if committed; may overproduce when
                // remaining < pmin_j (acceptable for greedy dispatch).
                double gen = std::max(pmin_j, std::min(pmax_j, remaining));
                total_cost += gen * p1.cost[scenario][t] * dt;
                remaining -= gen;  // can go negative (overproduction)

                if (record_production) {
                    result.t1_prod[j][t] = gen;
                }
            }

            if (remaining > 1e-6) {
                total_cost += remaining * 1e5 * dt;
            }
        } else if (remaining > 1e-6) {
            total_cost += remaining * 1e5 * dt;
        }
    }

    // Residual fuel credit
    for (int i = 0; i < inst.n_type2; ++i) {
        total_cost -= inst.type2_plants[i].fuel_price_end * fuel[i];
    }

    result.cost = total_cost;
    return result;
}

}  // namespace nuclear_outage
}  // namespace cbls
