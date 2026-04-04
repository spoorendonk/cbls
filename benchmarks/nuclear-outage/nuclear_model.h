#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <algorithm>
#include <cmath>

namespace cbls {
namespace nuclear_outage {

// ===========================================================================
// Legacy model for synthetic instances (NuclearInstance)
// ===========================================================================

struct NuclearModel {
    Model model;
    std::vector<int32_t> s;       // [outage] start period (int var handles)
    int32_t objective_node;       // constant node updated by hook (node ID)
};

inline NuclearModel build_nuclear_model(const NuclearInstance& inst) {
    NuclearModel result;
    auto& m = result.model;
    int O = inst.n_outages;

    result.s.resize(O);
    for (int o = 0; o < O; ++o) {
        result.s[o] = m.int_var(inst.outage_earliest[o], inst.outage_latest[o],
                                 "s_" + std::to_string(o));
    }

    result.objective_node = m.constant(1e15);
    m.minimize(result.objective_node);

    std::vector<std::vector<int>> unit_outages(inst.n_units);
    for (int o = 0; o < O; ++o) {
        unit_outages[inst.outage_unit[o]].push_back(o);
    }

    auto neg1 = m.constant(-1.0);
    for (int u = 0; u < inst.n_units; ++u) {
        auto& outages = unit_outages[u];
        if (outages.size() < 2) continue;
        std::sort(outages.begin(), outages.end(),
                  [&](int a, int b) {
                      return inst.outage_earliest[a] < inst.outage_earliest[b];
                  });
        for (size_t i = 0; i + 1 < outages.size(); ++i) {
            int o1 = outages[i];
            int o2 = outages[i + 1];
            auto dur = m.constant(static_cast<double>(inst.outage_duration[o1]));
            m.add_constraint(m.sum({result.s[o1], dur,
                                    m.prod(neg1, result.s[o2])}));
        }
    }

    m.close();
    return result;
}

// ===========================================================================
// ROADEF 2010 model
// ===========================================================================

struct ROADEFModel {
    Model model;

    // ha[o] = outage start week (int var handle) for outage o
    // Outage o corresponds to ct13[o] (plant_idx, cycle)
    std::vector<int32_t> ha;

    // Objective: constant node updated by hook
    int32_t objective_node;

    // Mapping from outage index to (plant_idx, cycle)
    std::vector<std::pair<int, int>> outage_info;
};

inline ROADEFModel build_roadef_model(const ROADEFInstance& inst) {
    ROADEFModel result;
    auto& m = result.model;
    int O = inst.n_outages();

    // ---------- Variables: ha[o] for each schedulable outage ----------
    result.ha.resize(O);
    result.outage_info.resize(O);

    for (int o = 0; o < O; ++o) {
        auto& w = inst.ct13[o];
        int lo = w.TO;
        int hi = w.TA;
        if (hi < 0) hi = inst.H - 1;  // if TA undefined, use full horizon

        result.ha[o] = m.int_var(lo, hi,
                                  "ha_" + std::to_string(w.plant_idx) + "_" +
                                  std::to_string(w.cycle));
        result.outage_info[o] = {w.plant_idx, w.cycle};
    }

    // Objective: constant updated by hook
    result.objective_node = m.constant(1e15);
    m.minimize(result.objective_node);

    auto neg1 = m.constant(-1.0);

    // ---------- CT13: cycle ordering ----------
    // For outages on the same plant, consecutive cycles must not overlap:
    // ha[i,k] >= ha[i,k-1] + DA[i,k-1]
    // Group outages by plant
    std::map<int, std::vector<int>> plant_outages;
    for (int o = 0; o < O; ++o) {
        plant_outages[inst.ct13[o].plant_idx].push_back(o);
    }

    for (auto& [plant_idx, outages] : plant_outages) {
        // Sort by cycle index
        std::sort(outages.begin(), outages.end(),
                  [&](int a, int b) {
                      return inst.ct13[a].cycle < inst.ct13[b].cycle;
                  });

        for (size_t i = 0; i + 1 < outages.size(); ++i) {
            int o1 = outages[i];
            int o2 = outages[i + 1];
            int k1 = inst.ct13[o1].cycle;
            // Duration of outage o1 in weeks
            int dur = inst.type2_plants[plant_idx].durations[k1];
            // ha[o1] + dur - ha[o2] <= 0
            auto dur_c = m.constant(static_cast<double>(dur));
            m.add_constraint(m.sum({result.ha[o1], dur_c,
                                    m.prod(neg1, result.ha[o2])}));
        }
    }

    // ---------- CT14-CT18: spacing constraints between outage sets ----------
    // For each constraint, enumerate all outage pairs in the plant set A_m
    // A_m = all outages of plants in C_m

    auto get_outages_for_plants = [&](const std::vector<int>& plant_set) {
        std::vector<int> outages;
        for (int o = 0; o < O; ++o) {
            for (int p : plant_set) {
                if (inst.ct13[o].plant_idx == p) {
                    outages.push_back(o);
                    break;
                }
            }
        }
        return outages;
    };

    for (auto& sc : inst.spacing_constraints) {
        auto outages = get_outages_for_plants(sc.plant_set);
        if (outages.size() < 2) continue;

        // For each pair of distinct outages
        for (size_t i = 0; i < outages.size(); ++i) {
            for (size_t j = i + 1; j < outages.size(); ++j) {
                int o1 = outages[i];
                int o2 = outages[j];
                int p1 = inst.ct13[o1].plant_idx;
                int p2 = inst.ct13[o2].plant_idx;
                int k1 = inst.ct13[o1].cycle;
                int k2 = inst.ct13[o2].cycle;
                int da1 = inst.type2_plants[p1].durations[k1];
                int da2 = inst.type2_plants[p2].durations[k2];

                if (sc.type == 14) {
                    // Min spacing/max overlap between outages:
                    // ha[o1] - ha[o2] - DA[o2] >= Se  OR  ha[o2] - ha[o1] - DA[o1] >= Se
                    // Encoded as penalty: min(Se - (ha1 - ha2 - da2), Se - (ha2 - ha1 - da1), 0)
                    // For the DAG, we use a disjunctive constraint approximation.
                    // Since CBLS uses SA with penalty, we can encode as:
                    // max(0, Se - gap1) + max(0, Se - gap2) where at least one must be 0
                    // But the DAG doesn't support disjunctions natively.
                    // Use penalty in hook instead — skip DAG constraints for CT14.
                    // (CT14-18 will be evaluated as penalties in the hook)
                }
                else if (sc.type == 16) {
                    // Min spacing between decoupling dates:
                    // |ha[o1] - ha[o2]| >= Se
                    // This is symmetric. Encode as penalty in hook.
                }
                // CT15, CT17, CT18 also involve disjunctions — handle in hook
            }
        }
    }

    // Note: CT14-CT18 are handled as penalties in the dispatch hook
    // because they involve disjunctive constraints that the DAG can't express.
    // CT19, CT20, CT21 are also evaluated in the hook.

    m.close();
    return result;
}

}  // namespace nuclear_outage
}  // namespace cbls
