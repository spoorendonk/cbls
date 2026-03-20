#pragma once

#include <cbls/verify.h>
#include "bunker_eca_model.h"
#include "data.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

namespace cbls {
namespace bunker_eca {

inline VerifyResult verify_bunker_eca(const BunkerECAModel& bec, const Instance& inst,
                                       double tol = 1e-4) {
    // Start with generic model checks (use tighter DAG-level tolerance)
    VerifyResult result = verify_model(bec.model);

    const auto& m = bec.model;
    int V = (int)inst.ships.size();
    int C = (int)inst.cargoes.size();

    // Extract variable values (handles are negative: var_id = -(handle + 1))
    auto val = [&](int32_t handle) -> double {
        return m.var(-(handle + 1)).value;
    };
    auto ival = [&](int32_t handle) -> int {
        return (int)std::round(val(handle));
    };

    // Compute average fuel coefficient (same as model builder)
    double avg_fuel_coeff = 0.0;
    for (int v = 0; v < V; ++v) {
        avg_fuel_coeff += inst.ships[v].fuel_coeff_laden;
    }
    avg_fuel_coeff /= V;

    // Compute average prices (same as model builder)
    double avg_hfo_price = 0.0, avg_mgo_price = 0.0;
    int n_bo = (int)inst.bunker_options.size();
    if (n_bo > 0) {
        for (auto& bo : inst.bunker_options) {
            avg_hfo_price += bo.hfo_price;
            avg_mgo_price += bo.mgo_price;
        }
        avg_hfo_price /= n_bo;
        avg_mgo_price /= n_bo;
    } else {
        for (auto& r : inst.regions) {
            avg_hfo_price += r.hfo_price;
            avg_mgo_price += r.mgo_price;
        }
        avg_hfo_price /= inst.regions.size();
        avg_mgo_price /= inst.regions.size();
    }

    // Per-cargo extracted values
    std::vector<int> assign_val(C);
    std::vector<double> speed_val(C);
    std::vector<int> eca_fuel_val(C);

    for (int c = 0; c < C; ++c) {
        assign_val[c] = ival(bec.assign[c]);
        speed_val[c] = val(bec.speed[c]);
        eca_fuel_val[c] = (bec.eca_fuel[c] >= 0) ? ival(bec.eca_fuel[c]) : 0;
    }

    // 1. Contract coverage
    for (int c = 0; c < C; ++c) {
        if (inst.cargoes[c].is_contract && assign_val[c] < 1) {
            result.add_error({VerifyError::Kind::Custom,
                "cargo[" + std::to_string(c) + "]", 1.0, (double)assign_val[c],
                "contract cargo not assigned"});
        }
    }

    // 2. Assignment bounds
    for (int c = 0; c < C; ++c) {
        int lb = inst.cargoes[c].is_contract ? 1 : 0;
        if (assign_val[c] < lb || assign_val[c] > V) {
            result.add_error({VerifyError::Kind::Custom,
                "assign[" + std::to_string(c) + "]", (double)lb, (double)assign_val[c],
                "assignment out of range [" + std::to_string(lb) + "," + std::to_string(V) + "]"});
        }
    }

    // 3. Speed bounds (fleet-wide)
    double v_min = 1e9, v_max = 0.0;
    for (int v = 0; v < V; ++v) {
        v_min = std::min(v_min, inst.ships[v].v_min_laden);
        v_max = std::max(v_max, inst.ships[v].v_max_laden);
    }
    for (int c = 0; c < C; ++c) {
        if (speed_val[c] < v_min - tol || speed_val[c] > v_max + tol) {
            result.add_error({VerifyError::Kind::Custom,
                "speed[" + std::to_string(c) + "]", v_min, speed_val[c],
                "speed out of fleet bounds [" + std::to_string(v_min) + "," + std::to_string(v_max) + "]"});
        }
    }

    // 4. ECA fuel type: 0 or 1
    for (int c = 0; c < C; ++c) {
        if (bec.eca_fuel[c] >= 0) {
            if (eca_fuel_val[c] != 0 && eca_fuel_val[c] != 1) {
                result.add_error({VerifyError::Kind::Custom,
                    "eca_fuel[" + std::to_string(c) + "]", 0.0, (double)eca_fuel_val[c],
                    "ECA fuel choice not 0 or 1"});
            }
        }
    }

    // Compute per-cargo fuel consumption independently
    std::vector<double> fuel(C), hfo(C), mgo(C);
    std::vector<bool> active(C);

    for (int c = 0; c < C; ++c) {
        active[c] = (assign_val[c] >= 1);
        double dist = inst.leg_distance(inst.cargoes[c].pickup_region,
                                         inst.cargoes[c].delivery_region);
        double alpha = avg_fuel_coeff * dist / 24.0;
        fuel[c] = alpha * speed_val[c] * speed_val[c] * (active[c] ? 1.0 : 0.0);

        double eca_frac = inst.leg_eca_fraction(
            inst.cargoes[c].pickup_region, inst.cargoes[c].delivery_region);
        if (eca_frac > 0.0 && bec.eca_fuel[c] >= 0) {
            mgo[c] = fuel[c] * eca_frac * eca_fuel_val[c];
            hfo[c] = fuel[c] - mgo[c];
        } else {
            hfo[c] = fuel[c];
            mgo[c] = 0.0;
        }
    }

    // 5. Fuel capacity per ship
    for (int v = 0; v < V; ++v) {
        double fuel_cap = inst.ships[v].initial_hfo
                        + (inst.ships[v].hfo_tank_max - inst.ships[v].initial_hfo)
                        - inst.ships[v].hfo_safety;
        double ship_hfo = 0.0;
        for (int c = 0; c < C; ++c) {
            if (assign_val[c] == v + 1) {
                ship_hfo += hfo[c];
            }
        }
        if (ship_hfo > fuel_cap + tol) {
            result.add_error({VerifyError::Kind::Custom,
                "ship[" + std::to_string(v) + "] '" + inst.ships[v].name + "'",
                fuel_cap, ship_hfo,
                "HFO consumption exceeds fuel capacity"});
        }
    }

    // 6. Time windows
    for (int c = 0; c < C; ++c) {
        if (!active[c]) continue;
        double dist = inst.leg_distance(inst.cargoes[c].pickup_region,
                                         inst.cargoes[c].delivery_region);
        if (dist <= 0.0) continue;

        double travel_time = dist / (24.0 * speed_val[c]);
        double available = inst.cargoes[c].delivery_tw_end
                         - inst.cargoes[c].pickup_tw_start
                         - inst.cargoes[c].service_time_load
                         - inst.cargoes[c].service_time_discharge;
        if (available > 0 && travel_time > available + tol) {
            result.add_error({VerifyError::Kind::Custom,
                "cargo[" + std::to_string(c) + "] time_window",
                available, travel_time,
                "travel time exceeds available time"});
        }
    }

    // 7. ECA compliance: fully-ECA legs must use MGO (only for active cargoes)
    for (int c = 0; c < C; ++c) {
        if (!active[c]) continue;
        double eca_frac = inst.leg_eca_fraction(
            inst.cargoes[c].pickup_region, inst.cargoes[c].delivery_region);
        if (eca_frac >= 0.99 && bec.eca_fuel[c] >= 0 && eca_fuel_val[c] != 1) {
            result.add_error({VerifyError::Kind::Custom,
                "cargo[" + std::to_string(c) + "] eca_compliance",
                1.0, (double)eca_fuel_val[c],
                "fully-ECA leg must use MGO (eca_fuel=1)"});
        }
    }

    // 8. Workload balance
    int max_per_ship = std::max(4, (C + V - 1) / V * 3);
    for (int v = 0; v < V; ++v) {
        int count = 0;
        for (int c = 0; c < C; ++c) {
            if (assign_val[c] == v + 1) count++;
        }
        if (count > max_per_ship) {
            result.add_error({VerifyError::Kind::Custom,
                "ship[" + std::to_string(v) + "] workload",
                (double)max_per_ship, (double)count,
                "ship carries more than max cargoes"});
        }
    }

    // 9. Objective recomputation
    double profit = 0.0;
    for (int c = 0; c < C; ++c) {
        if (active[c]) {
            profit += inst.cargoes[c].revenue;
        }
        profit -= avg_hfo_price * hfo[c];
        profit -= avg_mgo_price * mgo[c];
        if (active[c]) {
            double port_cost = inst.regions[inst.cargoes[c].pickup_region].port_cost
                             + inst.regions[inst.cargoes[c].delivery_region].port_cost;
            profit -= port_cost;
        }
    }

    // Model stores -objective for maximization
    double dag_obj = m.node(m.objective_id()).value;
    double expected_dag_obj = -profit;  // negated for maximization

    if (std::abs(dag_obj - expected_dag_obj) > std::max(tol * std::abs(profit), 1.0)) {
        result.add_error({VerifyError::Kind::ObjectiveMismatch,
            "objective", expected_dag_obj, dag_obj,
            "independent profit recomputation mismatch (profit=" + std::to_string(profit) + ")"});
    }

    return result;
}

}  // namespace bunker_eca
}  // namespace cbls
