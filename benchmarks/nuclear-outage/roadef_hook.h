#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include "nuclear_model.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace cbls {
namespace nuclear_outage {

#ifndef CBLS_HANDLE_TO_VAR_ID_DEFINED
#define CBLS_HANDLE_TO_VAR_ID_DEFINED
inline int32_t handle_to_var_id(int32_t handle) {
    return -(handle + 1);
}
#endif

// ===========================================================================
// ROADEF dispatch hook: full fuel-aware second-stage evaluation
// ===========================================================================

class ROADEFDispatchHook : public InnerSolverHook {
public:
    const ROADEFInstance& inst;
    const ROADEFModel& rm;

    int scenarios_per_move = -1;  // -1 = all
    int epoch_size = 10;

    ROADEFDispatchHook(const ROADEFInstance& inst_, const ROADEFModel& rm_)
        : inst(inst_), rm(rm_),
          ha_(inst_.n_outages()),
          reload_(inst_.n_type2) {}

    void solve(Model& model, ViolationManager&,
               const std::vector<int32_t>& = {}) override {
        int O = inst.n_outages();

        // 1. Read current outage start weeks from model
        for (int o = 0; o < O; ++o) {
            int32_t vid = handle_to_var_id(rm.ha[o]);
            ha_[o] = static_cast<int>(std::round(model.var(vid).value));
        }

        // 2. Determine scenario window
        int n_sc = (scenarios_per_move > 0)
                   ? std::min(scenarios_per_move, inst.S)
                   : inst.S;
        if (++calls_in_epoch_ >= epoch_size) {
            calls_in_epoch_ = 0;
            scenario_offset_ += n_sc;
        }
        int offset = (inst.S > n_sc) ? (scenario_offset_ % (inst.S - n_sc + 1)) : 0;

        // 3. Compute plant status (which cycle each Type 2 plant is in per week)
        compute_plant_status();

        // 4. Optimize reload amounts
        optimize_reloads();

        // 5. Simulate fuel dynamics + dispatch for each scenario
        double total_obj = 0.0;
        for (int s = offset; s < offset + n_sc; ++s) {
            total_obj += dispatch_scenario(s);
        }

        // Average production cost across scenarios
        double avg_prod_cost = total_obj / n_sc;

        // Add refueling cost (scenario-independent)
        double refuel_cost = 0.0;
        for (int i = 0; i < inst.n_type2; ++i) {
            auto& plant = inst.type2_plants[i];
            for (int k = 0; k < plant.n_cycles; ++k) {
                if (is_outage_scheduled(i, k)) {
                    refuel_cost += plant.refuel_cost[k] * reload_[i][k];
                }
            }
        }

        double cost = refuel_cost + avg_prod_cost;

        // 6. Add scheduling constraint penalties
        cost += scheduling_penalty();

        // 7. Write into objective
        model.node_mut(rm.objective_node).value = cost;
        model.node_mut(rm.objective_node).const_value = cost;
    }

private:
    std::vector<int> ha_;  // outage start weeks [n_outages]
    std::vector<std::vector<double>> reload_;  // reload[plant][cycle]

    int scenario_offset_ = 0;
    int calls_in_epoch_ = 0;

    // Per-plant status: which cycle is active at each week, and whether in outage
    // cycle_at_week_[i][h] = cycle index k such that week h is in campaign k or outage k
    // in_outage_[i][h] = true if plant i is in outage at week h
    std::vector<std::vector<int>> cycle_at_week_;
    std::vector<std::vector<bool>> in_outage_;

    // Get outage start week for (plant i, cycle k), or -1 if not scheduled
    int outage_start(int plant_idx, int cycle) const {
        for (int o = 0; o < inst.n_outages(); ++o) {
            if (inst.ct13[o].plant_idx == plant_idx && inst.ct13[o].cycle == cycle) {
                return ha_[o];
            }
        }
        return -1;  // not schedulable
    }

    bool is_outage_scheduled(int plant_idx, int cycle) const {
        return outage_start(plant_idx, cycle) >= 0;
    }

    void compute_plant_status() {
        cycle_at_week_.assign(inst.n_type2, std::vector<int>(inst.H, -1));
        in_outage_.assign(inst.n_type2, std::vector<bool>(inst.H, false));

        for (int i = 0; i < inst.n_type2; ++i) {
            auto& plant = inst.type2_plants[i];

            // Start with current campaign (k=-1) until first outage
            int current_cycle = -1;  // -1 = current campaign before first outage

            // Build sorted list of (outage_start_week, cycle_index, duration)
            struct OutageEvent { int start; int cycle; int duration; };
            std::vector<OutageEvent> events;
            for (int k = 0; k < plant.n_cycles; ++k) {
                int start = outage_start(i, k);
                if (start >= 0) {
                    events.push_back({start, k, plant.durations[k]});
                }
            }
            std::sort(events.begin(), events.end(),
                      [](const OutageEvent& a, const OutageEvent& b) {
                          return a.start < b.start;
                      });

            // Fill in cycle_at_week and in_outage
            int next_event = 0;
            current_cycle = -1;  // before first outage = current campaign

            for (int h = 0; h < inst.H; ++h) {
                // Check if an outage starts this week
                if (next_event < (int)events.size() && h == events[next_event].start) {
                    // Start outage
                    int ev_cycle = events[next_event].cycle;
                    int ev_dur = events[next_event].duration;
                    for (int w = h; w < std::min(h + ev_dur, inst.H); ++w) {
                        in_outage_[i][w] = true;
                        cycle_at_week_[i][w] = ev_cycle;
                    }
                    // After outage ends, we're in campaign ev_cycle
                    current_cycle = ev_cycle;
                    h = std::min(h + ev_dur - 1, inst.H - 1);  // skip ahead
                    ++next_event;
                } else {
                    // In campaign
                    in_outage_[i][h] = false;
                    cycle_at_week_[i][h] = current_cycle;
                }
            }
        }
    }

    void optimize_reloads() {
        // For each Type 2 plant, optimize reload amounts.
        // Simple strategy: reload to max (RMAX) to maximize fuel availability.
        // This minimizes need for expensive Type 1 backup.
        // More sophisticated: consider refuel_cost vs fuel_price_end tradeoff.
        reload_.resize(inst.n_type2);
        for (int i = 0; i < inst.n_type2; ++i) {
            auto& plant = inst.type2_plants[i];
            reload_[i].resize(plant.n_cycles, 0.0);
            for (int k = 0; k < plant.n_cycles; ++k) {
                if (is_outage_scheduled(i, k)) {
                    // Reload to max — minimizes production cost at the expense of refueling cost
                    // TODO: optimize this tradeoff
                    reload_[i][k] = plant.rmax[k];
                }
            }
        }
    }

    // Dispatch a single scenario, returning production cost - residual fuel credit
    double dispatch_scenario(int s) const {
        int T = inst.T;
        int tpw = inst.timesteps_per_week;
        double total_cost = 0.0;

        // Simulate fuel dynamics for each Type 2 plant
        // fuel[i] = current fuel stock for plant i
        std::vector<double> fuel(inst.n_type2);
        for (int i = 0; i < inst.n_type2; ++i) {
            fuel[i] = inst.type2_plants[i].initial_stock;
        }

        // Track Type 2 production per timestep
        std::vector<double> t2_prod(inst.n_type2);

        for (int t = 0; t < T; ++t) {
            int week = t / tpw;
            double dt = inst.timestep_durations[t];
            double demand = inst.demand[s][t];

            // Determine Type 2 production for this timestep
            double total_t2 = 0.0;
            for (int i = 0; i < inst.n_type2; ++i) {
                auto& plant = inst.type2_plants[i];

                if (week >= inst.H || in_outage_[i][week]) {
                    // CT3: offline during outage
                    t2_prod[i] = 0.0;

                    // CT10: refueling at first timestep of outage
                    if (in_outage_[i][week] && (t % tpw) == 0 && week > 0 && !in_outage_[i][week - 1]) {
                        // This is the first week of an outage — apply refueling
                        int k = cycle_at_week_[i][week];
                        if (k >= 0 && k < plant.n_cycles) {
                            double q = plant.q[k];
                            // BO for previous campaign: if k==0, use bo[0] (current campaign)
                            double bo_prev = (k > 0) ? plant.bo[k] : plant.bo[0];
                            double x_before = fuel[i];
                            // CT10: x_after = ((Q-1)/Q) * (x_before - BO_{k-1}) + r(i,k) + BO_k
                            double bo_k = plant.bo[k + 1];
                            fuel[i] = ((q - 1.0) / q) * (x_before - bo_prev) + reload_[i][k] + bo_k;
                            // Clamp to SMAX
                            fuel[i] = std::min(fuel[i], plant.smax[k]);
                        }
                    }
                    continue;
                }

                // Plant is online — determine max production
                int cycle_k = cycle_at_week_[i][week];
                // Profile index: 0 = current campaign (k=-1), k+1 = cycle k
                int prof_idx = (cycle_k < 0) ? 0 : (cycle_k + 1);
                double bo = plant.bo[prof_idx];
                double pmax_t = plant.pmax_t[t];

                double max_prod;
                if (fuel[i] >= bo) {
                    // CT5: above BO threshold — produce up to PMAX
                    max_prod = pmax_t;
                } else {
                    // CT6: below BO — follow decreasing profile
                    auto& prof = plant.profiles[prof_idx];
                    double pb = prof.evaluate(fuel[i]);
                    max_prod = pb * pmax_t;
                    // Check if production would be less than the timestep energy
                    if (fuel[i] < pb * pmax_t * dt) {
                        // Not enough fuel even at reduced rate — produce what we can
                        max_prod = std::max(0.0, fuel[i] / dt);
                    }
                }

                // CT4: production >= 0 (always satisfied)
                // Produce at max feasible power (greedy: minimize Type 1 usage)
                double prod = std::max(0.0, max_prod);
                // Don't produce more than we have fuel for
                prod = std::min(prod, std::max(0.0, fuel[i] / dt));
                t2_prod[i] = prod;
                total_t2 += prod;

                // CT9: fuel consumption during campaign
                fuel[i] -= prod * dt;
                fuel[i] = std::max(0.0, fuel[i]);
            }

            // Type 1 dispatch: fill remaining demand
            double remaining = demand - total_t2;

            if (remaining > 0.0 && inst.n_type1 > 0) {
                // Merit-order dispatch of Type 1 plants
                // Build sorted order by cost for this scenario/timestep
                // (could cache this but keeping simple for now)
                std::vector<int> order(inst.n_type1);
                std::iota(order.begin(), order.end(), 0);
                std::sort(order.begin(), order.end(),
                          [&](int a, int b) {
                              return inst.type1_plants[a].cost[s][t] <
                                     inst.type1_plants[b].cost[s][t];
                          });

                for (int j : order) {
                    if (remaining <= 1e-6) break;
                    auto& p1 = inst.type1_plants[j];
                    double pmin_j = p1.pmin[s][t];
                    double pmax_j = p1.pmax[s][t];
                    if (pmax_j <= 0.0) continue;

                    double gen = std::min(pmax_j, remaining);
                    gen = std::max(gen, pmin_j);  // must produce at least pmin if started
                    gen = std::min(gen, remaining);  // don't overproduce

                    total_cost += gen * p1.cost[s][t] * dt;
                    remaining -= gen;
                }

                // Unserved energy penalty
                if (remaining > 1e-6) {
                    total_cost += remaining * 1e5 * dt;  // large penalty
                }
            } else if (remaining > 1e-6) {
                // No Type 1 plants and unmet demand
                total_cost += remaining * 1e5 * dt;
            }
            // If remaining < 0, Type 2 overproduction — reduce last plant
            // (acceptable approximation for greedy dispatch)
        }

        // Residual fuel credit: subtract C_{i,T+1} * x(i,T,s) for each Type 2 plant
        for (int i = 0; i < inst.n_type2; ++i) {
            total_cost -= inst.type2_plants[i].fuel_price_end * fuel[i];
        }

        return total_cost;
    }

    // Evaluate scheduling constraint penalties (CT14-CT21)
    double scheduling_penalty() const {
        double penalty = 0.0;
        const double PENALTY_WEIGHT = 1e13;

        // CT14: min spacing / max overlap
        for (auto& sc : inst.spacing_constraints) {
            if (sc.type == 14) {
                penalty += ct14_penalty(sc) * PENALTY_WEIGHT;
            } else if (sc.type == 15) {
                penalty += ct15_penalty(sc) * PENALTY_WEIGHT;
            } else if (sc.type == 16) {
                penalty += ct16_penalty(sc) * PENALTY_WEIGHT;
            } else if (sc.type == 17) {
                penalty += ct17_penalty(sc) * PENALTY_WEIGHT;
            } else if (sc.type == 18) {
                penalty += ct18_penalty(sc) * PENALTY_WEIGHT;
            }
        }

        // CT19: resource constraints
        for (auto& res : inst.ct19) {
            penalty += ct19_penalty(res) * PENALTY_WEIGHT;
        }

        // CT20: max overlap per week
        for (auto& ct : inst.ct20) {
            penalty += ct20_penalty(ct) * PENALTY_WEIGHT;
        }

        // CT21: max offline capacity
        for (auto& ct : inst.ct21) {
            penalty += ct21_penalty(ct) * PENALTY_WEIGHT;
        }

        return penalty;
    }

    // Helper: get all outage indices for a plant set
    std::vector<int> outages_for_plants(const std::vector<int>& plant_set) const {
        std::vector<int> result;
        for (int o = 0; o < inst.n_outages(); ++o) {
            for (int p : plant_set) {
                if (inst.ct13[o].plant_idx == p) {
                    result.push_back(o);
                    break;
                }
            }
        }
        return result;
    }

    // CT14: min spacing between outages
    double ct14_penalty(const SpacingConstraint& sc) const {
        auto outages = outages_for_plants(sc.plant_set);
        double viol = 0.0;
        for (size_t i = 0; i < outages.size(); ++i) {
            for (size_t j = i + 1; j < outages.size(); ++j) {
                int o1 = outages[i], o2 = outages[j];
                int h1 = ha_[o1], h2 = ha_[o2];
                int p1 = inst.ct13[o1].plant_idx, p2 = inst.ct13[o2].plant_idx;
                int k1 = inst.ct13[o1].cycle, k2 = inst.ct13[o2].cycle;
                int da1 = inst.type2_plants[p1].durations[k1];
                int da2 = inst.type2_plants[p2].durations[k2];

                // gap1 = h1 - h2 - da2, gap2 = h2 - h1 - da1
                double gap1 = h1 - h2 - da2;
                double gap2 = h2 - h1 - da1;
                double best_gap = std::max(gap1, gap2);
                if (best_gap < sc.spacing) {
                    viol += sc.spacing - best_gap;
                }
            }
        }
        return viol;
    }

    // CT15: min spacing during a specific period
    double ct15_penalty(const SpacingConstraint& sc) const {
        auto outages = outages_for_plants(sc.plant_set);
        double viol = 0.0;
        for (size_t i = 0; i < outages.size(); ++i) {
            for (size_t j = i + 1; j < outages.size(); ++j) {
                int o1 = outages[i], o2 = outages[j];
                int h1 = ha_[o1], h2 = ha_[o2];
                int p1 = inst.ct13[o1].plant_idx, p2 = inst.ct13[o2].plant_idx;
                int k1 = inst.ct13[o1].cycle, k2 = inst.ct13[o2].cycle;
                int da1 = inst.type2_plants[p1].durations[k1];
                int da2 = inst.type2_plants[p2].durations[k2];

                // Check if both outages intersect [ID, IF]
                bool o1_intersects = (h1 >= sc.period_start - da1 + 1) && (h1 <= sc.period_end);
                bool o2_intersects = (h2 >= sc.period_start - da2 + 1) && (h2 <= sc.period_end);
                if (!o1_intersects || !o2_intersects) continue;

                double gap1 = h1 - h2 - da2;
                double gap2 = h2 - h1 - da1;
                double best_gap = std::max(gap1, gap2);
                if (best_gap < sc.spacing) {
                    viol += sc.spacing - best_gap;
                }
            }
        }
        return viol;
    }

    // CT16: min spacing between decoupling dates
    double ct16_penalty(const SpacingConstraint& sc) const {
        auto outages = outages_for_plants(sc.plant_set);
        double viol = 0.0;
        for (size_t i = 0; i < outages.size(); ++i) {
            for (size_t j = i + 1; j < outages.size(); ++j) {
                double gap = std::abs(ha_[outages[i]] - ha_[outages[j]]);
                if (gap < sc.spacing) {
                    viol += sc.spacing - gap;
                }
            }
        }
        return viol;
    }

    // CT17: min spacing between coupling dates
    double ct17_penalty(const SpacingConstraint& sc) const {
        auto outages = outages_for_plants(sc.plant_set);
        double viol = 0.0;
        for (size_t i = 0; i < outages.size(); ++i) {
            for (size_t j = i + 1; j < outages.size(); ++j) {
                int o1 = outages[i], o2 = outages[j];
                int p1 = inst.ct13[o1].plant_idx, p2 = inst.ct13[o2].plant_idx;
                int k1 = inst.ct13[o1].cycle, k2 = inst.ct13[o2].cycle;
                int coupling1 = ha_[o1] + inst.type2_plants[p1].durations[k1];
                int coupling2 = ha_[o2] + inst.type2_plants[p2].durations[k2];
                double gap = std::abs(coupling1 - coupling2);
                if (gap < sc.spacing) {
                    viol += sc.spacing - gap;
                }
            }
        }
        return viol;
    }

    // CT18: min spacing between coupling and decoupling dates
    double ct18_penalty(const SpacingConstraint& sc) const {
        auto outages = outages_for_plants(sc.plant_set);
        double viol = 0.0;
        for (size_t i = 0; i < outages.size(); ++i) {
            for (size_t j = i + 1; j < outages.size(); ++j) {
                int o1 = outages[i], o2 = outages[j];
                int p1 = inst.ct13[o1].plant_idx, p2 = inst.ct13[o2].plant_idx;
                int k1 = inst.ct13[o1].cycle, k2 = inst.ct13[o2].cycle;
                int coupling1 = ha_[o1] + inst.type2_plants[p1].durations[k1];
                int coupling2 = ha_[o2] + inst.type2_plants[p2].durations[k2];

                // |coupling1 - ha2| and |coupling2 - ha1|
                double gap1 = std::abs(coupling1 - ha_[outages[j]]);
                double gap2 = std::abs(coupling2 - ha_[outages[i]]);
                if (gap1 < sc.spacing) viol += sc.spacing - gap1;
                if (gap2 < sc.spacing) viol += sc.spacing - gap2;
            }
        }
        return viol;
    }

    // CT19: resource constraints
    double ct19_penalty(const CT19Resource& res) const {
        double viol = 0.0;
        for (int h = 0; h < inst.H; ++h) {
            int count = 0;
            for (auto& usage : res.usages) {
                int plant = usage.plant_idx;
                for (int k = 0; k < (int)usage.start.size(); ++k) {
                    int ha = outage_start(plant, k);
                    if (ha < 0) continue;
                    int res_start = ha + usage.start[k];
                    int res_end = res_start + usage.duration[k];
                    if (h >= res_start && h < res_end) {
                        count++;
                    }
                }
            }
            if (count > res.quantity) {
                viol += count - res.quantity;
            }
        }
        return viol;
    }

    // CT20: max overlapping outages per week
    double ct20_penalty(const CT20MaxOverlap& ct) const {
        auto outages = outages_for_plants(ct.plant_set);
        int count = 0;
        for (int o : outages) {
            int h = ha_[o];
            int p = inst.ct13[o].plant_idx;
            int k = inst.ct13[o].cycle;
            int dur = inst.type2_plants[p].durations[k];
            if (ct.week >= h && ct.week < h + dur) {
                count++;
            }
        }
        return std::max(0, count - ct.max_allowed);
    }

    // CT21: max offline capacity
    double ct21_penalty(const CT21OfflineCap& ct) const {
        double viol = 0.0;
        for (int h = ct.time_start; h <= ct.time_end && h < inst.H; ++h) {
            double offline_cap = 0.0;
            for (int p : ct.plant_set) {
                if (p < inst.n_type2 && in_outage_[p][h]) {
                    // Sum PMAX of offline plants (use first timestep of week as representative)
                    int t = h * inst.timesteps_per_week;
                    if (t < inst.T) {
                        offline_cap += inst.type2_plants[p].pmax_t[t];
                    }
                }
            }
            if (offline_cap > ct.imax) {
                viol += offline_cap - ct.imax;
            }
        }
        return viol;
    }
};

}  // namespace nuclear_outage
}  // namespace cbls
