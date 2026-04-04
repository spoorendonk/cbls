#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include "nuclear_model.h"
#include "roadef_dispatch.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace cbls {
namespace nuclear_outage {

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
          ha_(inst_.n_outages()) {}

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

        // 3. Build lookup table and compute plant status
        lookup_ = build_outage_lookup(inst, ha_);
        auto status = compute_plant_status(inst, lookup_);
        auto reload = compute_reloads(inst, lookup_);

        // 4. Simulate fuel dynamics + dispatch for each scenario
        double total_obj = 0.0;
        for (int s = offset; s < offset + n_sc; ++s) {
            auto sr = simulate_scenario(inst, status, reload, s);
            total_obj += sr.cost;
        }

        double avg_prod_cost = total_obj / n_sc;

        // Add refueling cost (scenario-independent)
        double refuel_cost = 0.0;
        for (int i = 0; i < inst.n_type2; ++i) {
            auto& plant = inst.type2_plants[i];
            for (int k = 0; k < plant.n_cycles; ++k) {
                if (lookup_[i][k] >= 0) {
                    refuel_cost += plant.refuel_cost[k] * reload[i][k];
                }
            }
        }

        double cost = refuel_cost + avg_prod_cost;

        // 5. Add scheduling constraint penalties
        cost += scheduling_penalty(status);

        // 6. Write into objective
        model.node_mut(rm.objective_node).value = cost;
        model.node_mut(rm.objective_node).const_value = cost;
    }

private:
    std::vector<int> ha_;  // outage start weeks [n_outages]
    OutageLookup lookup_;

    int scenario_offset_ = 0;
    int calls_in_epoch_ = 0;

    // Evaluate scheduling constraint penalties (CT14-CT21)
    double scheduling_penalty(const PlantStatus& status) const {
        double penalty = 0.0;
        const double PENALTY_WEIGHT = 1e13;

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

        for (auto& res : inst.ct19) {
            penalty += ct19_penalty(res) * PENALTY_WEIGHT;
        }

        for (auto& ct : inst.ct20) {
            penalty += ct20_penalty(ct) * PENALTY_WEIGHT;
        }

        for (auto& ct : inst.ct21) {
            penalty += ct21_penalty(ct, status) * PENALTY_WEIGHT;
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

                double gap1 = std::abs(coupling1 - ha_[outages[j]]);
                double gap2 = std::abs(coupling2 - ha_[outages[i]]);
                if (gap1 < sc.spacing) viol += sc.spacing - gap1;
                if (gap2 < sc.spacing) viol += sc.spacing - gap2;
            }
        }
        return viol;
    }

    double ct19_penalty(const CT19Resource& res) const {
        double viol = 0.0;
        for (int h = 0; h < inst.H; ++h) {
            int count = 0;
            for (auto& usage : res.usages) {
                int plant = usage.plant_idx;
                if (plant < 0 || plant >= (int)lookup_.size()) continue;
                int max_k = (int)lookup_[plant].size();
                int n_cycles = (int)usage.start.size();
                for (int k = 0; k < std::min(n_cycles, max_k); ++k) {
                    int ha = lookup_[plant][k];
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

    double ct21_penalty(const CT21OfflineCap& ct, const PlantStatus& status) const {
        double viol = 0.0;
        for (int h = ct.time_start; h <= ct.time_end && h < inst.H; ++h) {
            double offline_cap = 0.0;
            for (int p : ct.plant_set) {
                if (p < inst.n_type2 && status.in_outage[p][h]) {
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
