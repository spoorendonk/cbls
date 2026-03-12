#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>

namespace cbls {
namespace chped {

struct CHPEDModel {
    Model model;
    std::vector<std::vector<int32_t>> commit;  // [unit][period] - var handles
    std::vector<std::vector<int32_t>> power;   // [unit][period] - var handles
};

inline CHPEDModel build_chped_model(const Instance& inst) {
    CHPEDModel result;
    auto& m = result.model;
    int N = inst.n_units;
    int T = inst.n_periods;

    // Variables
    result.commit.resize(N);
    result.power.resize(N);
    for (int i = 0; i < N; ++i) {
        result.commit[i].resize(T);
        result.power[i].resize(T);
        for (int t = 0; t < T; ++t) {
            result.commit[i][t] = m.bool_var("u_" + std::to_string(i) + "_" + std::to_string(t));
            result.power[i][t] = m.float_var(inst.P_min[i], inst.P_max[i],
                                              "p_" + std::to_string(i) + "_" + std::to_string(t));
        }
    }

    // Objective: total valve-point cost
    std::vector<int32_t> cost_terms;
    for (int i = 0; i < N; ++i) {
        for (int t = 0; t < T; ++t) {
            auto P = result.power[i][t];
            auto ai = m.constant(inst.a[i]);
            auto bi = m.constant(inst.b[i]);
            auto ci = m.constant(inst.c[i]);
            auto di = m.constant(inst.d[i]);
            auto ei = m.constant(inst.e[i]);
            auto pmin_i = m.constant(inst.P_min[i]);
            auto two = m.constant(2.0);
            auto neg1 = m.constant(-1.0);

            // base_cost = a[i] + b[i]*P + c[i]*P^2
            auto base_cost = m.sum({ai, m.prod(bi, P), m.prod(ci, m.pow_expr(P, two))});

            // valve_point = |d[i]*sin(e[i]*(P_min[i] - P))|
            auto pmin_minus_p = m.sum({pmin_i, m.prod(neg1, P)});
            auto valve_point = m.abs_expr(
                m.prod(di, m.sin_expr(m.prod(ei, pmin_minus_p)))
            );

            auto unit_cost = m.sum({base_cost, valve_point});
            cost_terms.push_back(m.prod(result.commit[i][t], unit_cost));
        }
    }

    auto total_cost = m.sum(cost_terms);
    m.minimize(total_cost);

    // Constraints
    auto neg1_c = m.constant(-1.0);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> supply_terms;
        for (int i = 0; i < N; ++i) {
            supply_terms.push_back(m.prod(result.commit[i][t], result.power[i][t]));
        }
        auto supply = m.sum(supply_terms);

        // demand[t] - supply <= 0
        auto demand_t = m.constant(inst.demand[t]);
        m.add_constraint(m.sum({demand_t, m.prod(neg1_c, supply)}));

        // reserve constraint
        if (inst.reserve[t] > 0) {
            auto reserve_t = m.constant(inst.reserve[t]);
            m.add_constraint(m.sum({demand_t, reserve_t, m.prod(neg1_c, supply)}));
        }
    }

    m.close();
    return result;
}

}  // namespace chped
}  // namespace cbls
