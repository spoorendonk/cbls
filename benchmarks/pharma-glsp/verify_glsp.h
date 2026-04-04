#pragma once

#include "data.h"
#include "glsp_model.h"

#include <algorithm>
#include <cbls/verify.h>
#include <cmath>
#include <string>
#include <vector>

namespace cbls {
namespace glsp {

inline VerifyResult verify_glsp(const GLSPModel& gm, const GLSPInstance& inst, double tol = 1e-4) {
    VerifyResult result = verify_model(gm.model);

    const auto& m = gm.model;
    int J = inst.n_products;
    int T = inst.n_macro;
    int M = inst.n_micro_per_macro;

    // Helper: read FloatVar value from handle
    auto val = [&](int32_t handle) -> double { return m.var(-(handle + 1)).value; };

    // Extract lot sizes: lot_val[j][t]
    std::vector<std::vector<double>> lot_val(J, std::vector<double>(T));
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            lot_val[j][t] = val(gm.lot[j][t]);
        }
    }

    // Extract sequences: seq_elems[t] = permutation of {0..J-1}
    std::vector<std::vector<int32_t>> seq_elems(T);
    for (int t = 0; t < T; ++t) {
        seq_elems[t] = m.var(-(gm.seq[t] + 1)).elements;
    }

    // 1. Sequence validity — each seq[t] must be a permutation of {0..J-1}
    for (int t = 0; t < T; ++t) {
        if ((int)seq_elems[t].size() != J) {
            result.add_error({VerifyError::Kind::Custom, "seq[" + std::to_string(t) + "]",
                              (double)J, (double)seq_elems[t].size(), "wrong sequence length"});
            continue;
        }
        std::vector<int32_t> sorted = seq_elems[t];
        std::sort(sorted.begin(), sorted.end());
        for (int k = 0; k < J; ++k) {
            if (sorted[k] != k) {
                result.add_error({VerifyError::Kind::Custom, "seq[" + std::to_string(t) + "]",
                                  (double)k, (double)sorted[k],
                                  "not a valid permutation of {0..J-1}"});
                break;
            }
        }
    }

    // 2. Lot bounds: 0 <= lot[j][t] <= capacity[t] / process_time[j]
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            double ub = inst.capacity[t] / inst.process_time[j];
            if (lot_val[j][t] < -tol || lot_val[j][t] > ub + tol) {
                result.add_error({VerifyError::Kind::Custom,
                                  "lot[" + std::to_string(j) + "][" + std::to_string(t) + "]", ub,
                                  lot_val[j][t],
                                  "lot size out of bounds [0," + std::to_string(ub) + "]"});
            }
        }
    }

    // 3. Min lot size: if lot[j][t] > eps then lot[j][t] >= min_lot[j]
    for (int j = 0; j < J; ++j) {
        double kappa = inst.min_lot[j];
        if (kappa <= 0) {
            continue;
        }
        for (int t = 0; t < T; ++t) {
            if (lot_val[j][t] > 1.0 && lot_val[j][t] < kappa - tol) {
                result.add_error({VerifyError::Kind::Custom,
                                  "lot[" + std::to_string(j) + "][" + std::to_string(t) + "]",
                                  kappa, lot_val[j][t], "positive lot below minimum"});
            }
        }
    }

    // Compute setup times independently
    std::vector<double> setup_time_val(T, 0.0);
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < J - 1; ++k) {
            int from = seq_elems[t][k];
            int to = seq_elems[t][k + 1];
            setup_time_val[t] += inst.setup_time[from][to];
        }
    }

    // 4. Capacity: sum_j lot[j][t]*process_time[j] + setup_time <= capacity[t]
    for (int t = 0; t < T; ++t) {
        double used = setup_time_val[t];
        for (int j = 0; j < J; ++j) {
            used += lot_val[j][t] * inst.process_time[j];
        }
        if (used > inst.capacity[t] + tol) {
            result.add_error({VerifyError::Kind::ConstraintViolation,
                              "capacity[" + std::to_string(t) + "]", inst.capacity[t], used,
                              "production time + setup exceeds capacity"});
        }
    }

    // 5. Cumulative demand satisfaction
    //    For each product j, at each period t:
    //    sum_{tau=0..t} (1-theta_{j,tau}) * lot[j][tau] >= sum_{tau=0..t} d_{j,tau}
    std::vector<std::vector<double>> cum_inv(J, std::vector<double>(T));
    for (int j = 0; j < J; ++j) {
        double cum_supply = 0.0;
        double cum_demand = 0.0;
        for (int t = 0; t < T; ++t) {
            double serv_frac = 1.0 - inst.defect_rate[j][t];
            cum_supply += serv_frac * lot_val[j][t];
            cum_demand += inst.demand[j][t];
            cum_inv[j][t] = cum_supply - cum_demand;
            if (cum_inv[j][t] < -tol) {
                result.add_error({VerifyError::Kind::ConstraintViolation,
                                  "demand[" + std::to_string(j) + "][" + std::to_string(t) + "]",
                                  0.0, cum_inv[j][t], "cumulative demand not satisfied"});
            }
        }
    }

    // 6. Objective recomputation
    double obj = 0.0;
    double obj_changeover = 0.0, obj_holding = 0.0;
    double obj_rework_hold = 0.0, obj_disposal = 0.0;

    // 6a. Changeover cost
    for (int t = 0; t < T; ++t) {
        for (int k = 0; k < J - 1; ++k) {
            int from = seq_elems[t][k];
            int to = seq_elems[t][k + 1];
            obj_changeover += inst.setup_cost[from][to];
        }
    }
    obj += obj_changeover;

    // 6b. Holding cost
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            obj_holding += inst.holding_cost[j] * std::max(0.0, cum_inv[j][t]);
        }
    }
    obj += obj_holding;

    // 6c. Rework holding cost + disposal cost
    for (int j = 0; j < J; ++j) {
        for (int t = 0; t < T; ++t) {
            if (inst.defect_rate[j][t] < 1e-12) {
                continue;
            }
            double defectives = inst.defect_rate[j][t] * lot_val[j][t];

            double hr = inst.rework_holding_cost[j] * M * 0.5;
            obj_rework_hold += hr * defectives;

            double rework_cap = inst.lifetime[j] * (inst.capacity[t] / T / M) / inst.rework_time[j];
            double excess = std::max(0.0, defectives - rework_cap);
            obj_disposal += inst.disposal_cost[j] * excess;
        }
    }
    obj += obj_rework_hold + obj_disposal;

    // Compare against DAG objective
    double dag_obj = m.node(m.objective_id()).value;
    double obj_tol = std::max(tol * std::abs(obj), 1.0);
    if (std::abs(dag_obj - obj) > obj_tol) {
        result.add_error({VerifyError::Kind::ObjectiveMismatch, "objective", obj, dag_obj,
                          "independent objective recomputation mismatch"});
    }

    return result;
}

}  // namespace glsp
}  // namespace cbls
