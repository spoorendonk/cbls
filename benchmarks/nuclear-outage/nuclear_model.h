#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <algorithm>

namespace cbls {
namespace nuclear_outage {

struct NuclearModel {
    Model model;
    std::vector<int32_t> s;          // [outage] start period (int var handles)
    int32_t objective_node;           // constant node updated by hook (node ID, non-negative)
};

inline NuclearModel build_nuclear_model(const NuclearInstance& inst) {
    NuclearModel result;
    auto& m = result.model;
    int O = inst.n_outages;

    // ---------- Variables ----------
    result.s.resize(O);
    for (int o = 0; o < O; ++o) {
        result.s[o] = m.int_var(inst.outage_earliest[o], inst.outage_latest[o],
                                 "s_" + std::to_string(o));
    }

    // Objective: a constant node updated by the dispatch hook.
    // Using a constant (not a variable) prevents the SA loop from
    // directly minimizing it — only the hook sets its value.
    // Initialize to a large value so the first hook evaluation is an improvement
    result.objective_node = m.constant(1e15);
    m.minimize(result.objective_node);

    // ---------- Constraints ----------
    auto zero = m.constant(0.0);
    auto neg1 = m.constant(-1.0);

    // Spacing constraints: consecutive outages on the same unit must not overlap.
    // For outages o1, o2 on the same unit where o1 is earlier:
    //   s[o1] + duration[o1] - s[o2] <= 0
    // Group outages by unit
    std::vector<std::vector<int>> unit_outages(inst.n_units);
    for (int o = 0; o < O; ++o) {
        unit_outages[inst.outage_unit[o]].push_back(o);
    }

    for (int u = 0; u < inst.n_units; ++u) {
        auto& outages = unit_outages[u];
        if (outages.size() < 2) continue;

        // Sort by earliest start to establish ordering
        std::sort(outages.begin(), outages.end(),
                  [&](int a, int b) {
                      return inst.outage_earliest[a] < inst.outage_earliest[b];
                  });

        // Consecutive outages must not overlap
        for (size_t i = 0; i + 1 < outages.size(); ++i) {
            int o1 = outages[i];
            int o2 = outages[i + 1];
            // s[o1] + duration[o1] - s[o2] <= 0
            auto dur = m.constant(static_cast<double>(inst.outage_duration[o1]));
            m.add_constraint(m.sum({result.s[o1], dur,
                                    m.prod(neg1, result.s[o2])}));
        }
    }

    // Site spacing constraints are handled in the hook as penalty
    // (avoids O(n_periods * n_sites) DAG nodes)

    m.close();
    return result;
}

}  // namespace nuclear_outage
}  // namespace cbls
