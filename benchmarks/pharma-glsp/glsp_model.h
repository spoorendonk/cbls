#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <cmath>

namespace cbls {
namespace glsp {

struct GLSPModel {
    Model model;
    // seq[t] = ListVar(J) handle — production sequence per macro-period
    std::vector<int32_t> seq;
    // lot[j][t] = FloatVar handle — lot size for product j in macro-period t
    std::vector<std::vector<int32_t>> lot;
    // setup_time_node[t] = PairLambda node for setup time (used in capacity constraint)
    std::vector<int32_t> setup_time_nodes;

    const GLSPInstance* instance = nullptr;
};

inline GLSPModel build_glsp_model(const GLSPInstance& inst) {
    GLSPModel result;
    result.instance = &inst;
    auto& m = result.model;
    int J = inst.n_products;
    int T = inst.n_macro;
    int M = inst.n_micro_per_macro;

    // ---------- Variables ----------

    result.seq.resize(T);
    for (int t = 0; t < T; ++t) {
        result.seq[t] = m.list_var(J, "seq_" + std::to_string(t));
    }

    result.lot.resize(J);
    for (int j = 0; j < J; ++j) {
        result.lot[j].resize(T);
        for (int t = 0; t < T; ++t) {
            double max_lot = inst.capacity[t] / inst.process_time[j];
            result.lot[j][t] = m.float_var(0.0, max_lot,
                "lot_" + std::to_string(j) + "_" + std::to_string(t));
        }
    }

    // ---------- Shared constants ----------
    auto zero = m.constant(0.0);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);

    // ---------- Objective terms ----------
    std::vector<int32_t> obj_terms;

    // 1. Changeover cost via pair_lambda_sum (single node per macro-period)
    for (int t = 0; t < T; ++t) {
        auto cost_matrix = inst.setup_cost;  // capture by value
        auto changeover = m.pair_lambda_sum(result.seq[t],
            [cost_matrix](int from, int to) -> double {
                return cost_matrix[from][to];
            });
        obj_terms.push_back(changeover);
    }

    // 2. Inventory holding cost
    std::vector<std::vector<int32_t>> inv_serv(J);
    for (int j = 0; j < J; ++j) {
        inv_serv[j].resize(T);
        int32_t cum_supply = zero;
        double cum_demand = 0.0;
        for (int t = 0; t < T; ++t) {
            double serv_frac = 1.0 - inst.defect_rate[j][t];
            auto production_serv = m.prod(m.constant(serv_frac), result.lot[j][t]);
            cum_supply = m.sum({cum_supply, production_serv});
            cum_demand += inst.demand[j][t];
            inv_serv[j][t] = m.sum({cum_supply, m.prod(neg1, m.constant(cum_demand))});

            auto pos_inv = m.max_expr({zero, inv_serv[j][t]});
            obj_terms.push_back(m.prod(m.constant(inst.holding_cost[j]), pos_inv));
        }
    }

    // 3. Rework/disposal cost
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            if (inst.defect_rate[j][t] < 1e-12) continue;
            auto defectives = m.prod(m.constant(inst.defect_rate[j][t]), result.lot[j][t]);

            double hr = inst.rework_holding_cost[j] * M * 0.5;
            obj_terms.push_back(m.prod(m.constant(hr), defectives));

            double rework_cap = inst.lifetime[j] * (inst.capacity[t] / T / M) / inst.rework_time[j];
            auto excess = m.max_expr({zero, m.sum({defectives, m.prod(neg1, m.constant(rework_cap))})});
            obj_terms.push_back(m.prod(m.constant(inst.disposal_cost[j]), excess));
        }
    }

    m.minimize(m.sum(obj_terms));

    // ---------- Constraints ----------

    // 1. Capacity: production time + setup time <= capacity
    result.setup_time_nodes.resize(T);
    for (int t = 0; t < T; ++t) {
        std::vector<int32_t> time_terms;
        for (int j = 0; j < J; ++j) {
            time_terms.push_back(m.prod(m.constant(inst.process_time[j]), result.lot[j][t]));
        }
        // Setup time via pair_lambda_sum
        auto time_matrix = inst.setup_time;
        auto setup_time = m.pair_lambda_sum(result.seq[t],
            [time_matrix](int from, int to) -> double {
                return time_matrix[from][to];
            });
        result.setup_time_nodes[t] = setup_time;
        time_terms.push_back(setup_time);
        auto total_time = m.sum(time_terms);
        m.add_constraint(m.sum({total_time, m.prod(neg1, m.constant(inst.capacity[t]))}));
    }

    // 2. Demand satisfaction: cumulative inventory >= 0
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            m.add_constraint(m.prod(neg1, inv_serv[j][t]));
        }
    }

    // 3. Min lot size
    for (int j = 0; j < J; ++j) {
        double kappa = inst.min_lot[j];
        if (kappa <= 0) continue;
        for (int t = 0; t < T; ++t) {
            auto gap = m.max_expr({zero, m.sum({m.constant(kappa), m.prod(neg1, result.lot[j][t])})});
            auto active = m.max_expr({zero, m.min_expr({one,
                m.prod(m.constant(1.0 / kappa),
                    m.max_expr({zero, m.sum({result.lot[j][t], neg1})}))})});
            m.add_constraint(m.prod(gap, active));
        }
    }

    m.close();
    return result;
}

}  // namespace glsp
}  // namespace cbls
