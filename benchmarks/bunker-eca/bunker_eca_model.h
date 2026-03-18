#pragma once

#include <cbls/cbls.h>
#include "data.h"
#include <vector>
#include <algorithm>
#include <string>
#include <cmath>

namespace cbls {
namespace bunker_eca {

struct BunkerECAModel {
    Model model;

    // Per cargo: which ship (0=none, 1..V)
    std::vector<int32_t> assign;        // IntVar handles

    // Per cargo: speed for the main transport leg (pickup -> delivery)
    std::vector<int32_t> speed;         // FloatVar handles

    // Per cargo: use MGO (1) or HFO (0) for the main leg (relevant if ECA fraction > 0)
    std::vector<int32_t> eca_fuel;      // BoolVar handles (-1 sentinel if no ECA)

    // Instance reference for solution extraction
    const Instance* inst = nullptr;
};

// Build indicator expression: 1 if assign[c] == v, else 0
inline int32_t make_indicator(Model& m, int32_t assign_var, int ship_val,
                               int32_t zero, int32_t one, int32_t half) {
    auto ship_const = m.constant(static_cast<double>(ship_val));
    auto diff = m.sum({assign_var, m.neg(ship_const)});
    auto abs_diff = m.abs_expr(diff);
    auto cond = m.sum({abs_diff, m.neg(half)});
    return m.if_then_else(cond, zero, one);
}

inline BunkerECAModel build_bunker_eca_model(const Instance& inst) {
    BunkerECAModel result;
    result.inst = &inst;
    auto& m = result.model;

    int V = (int)inst.ships.size();
    int C = (int)inst.cargoes.size();

    // ---------- Constants ----------
    auto zero = m.constant(0.0);
    auto one = m.constant(1.0);
    auto half = m.constant(0.5);
    auto two = m.constant(2.0);
    auto twenty_four = m.constant(24.0);

    // ---------- Variables ----------

    // Cargo assignment: contract = [1,V], spot = [0,V]
    result.assign.resize(C);
    for (int c = 0; c < C; ++c) {
        int lb = inst.cargoes[c].is_contract ? 1 : 0;
        result.assign[c] = m.int_var(lb, V, "assign_" + std::to_string(c));
    }

    // Speed per cargo
    result.speed.resize(C);
    for (int c = 0; c < C; ++c) {
        double v_min = 1e9, v_max = 0.0;
        for (int v = 0; v < V; ++v) {
            v_min = std::min(v_min, inst.ships[v].v_min_laden);
            v_max = std::max(v_max, inst.ships[v].v_max_laden);
        }
        result.speed[c] = m.float_var(v_min, v_max,
                                        "speed_" + std::to_string(c));
    }

    // ECA fuel choice per cargo
    result.eca_fuel.resize(C);
    for (int c = 0; c < C; ++c) {
        double eca_frac = inst.leg_eca_fraction(
            inst.cargoes[c].pickup_region, inst.cargoes[c].delivery_region);
        if (eca_frac > 0.0) {
            result.eca_fuel[c] = m.bool_var("eca_" + std::to_string(c));
        } else {
            result.eca_fuel[c] = -1;
        }
    }

    // ---------- Indicator expressions ----------
    // on_v[c][v] = 1 if cargo c is on ship v+1
    std::vector<std::vector<int32_t>> on_v(C, std::vector<int32_t>(V));
    for (int c = 0; c < C; ++c) {
        for (int v = 0; v < V; ++v) {
            on_v[c][v] = make_indicator(m, result.assign[c], v + 1,
                                         zero, one, half);
        }
    }

    // active[c] = 1 if cargo assigned to any ship
    std::vector<int32_t> active(C);
    for (int c = 0; c < C; ++c) {
        if (inst.cargoes[c].is_contract) {
            active[c] = m.constant(1.0);
        } else {
            auto is_zero = make_indicator(m, result.assign[c], 0, zero, one, half);
            active[c] = m.sum({one, m.neg(is_zero)});
        }
    }

    // ---------- Per-cargo fuel consumption ----------
    // Use average fuel coefficient across fleet (simplification for DAG size)
    // fuel[c] = alpha_avg * speed[c]^2 * active[c]
    double avg_fuel_coeff = 0.0;
    for (int v = 0; v < V; ++v) {
        avg_fuel_coeff += inst.ships[v].fuel_coeff_laden;
    }
    avg_fuel_coeff /= V;

    std::vector<int32_t> fuel_per_cargo(C);
    std::vector<int32_t> hfo_consumed(C);
    std::vector<int32_t> mgo_consumed(C);

    for (int c = 0; c < C; ++c) {
        double dist = inst.leg_distance(inst.cargoes[c].pickup_region,
                                         inst.cargoes[c].delivery_region);
        double alpha = avg_fuel_coeff * dist / 24.0;
        auto alpha_const = m.constant(alpha);
        auto speed_sq = m.pow_expr(result.speed[c], two);
        fuel_per_cargo[c] = m.prod(alpha_const, m.prod(speed_sq, active[c]));

        // Split into HFO/MGO by ECA fraction and fuel choice
        double eca_frac = inst.leg_eca_fraction(
            inst.cargoes[c].pickup_region, inst.cargoes[c].delivery_region);

        if (eca_frac > 0.0 && result.eca_fuel[c] >= 0) {
            auto eca_frac_const = m.constant(eca_frac);
            mgo_consumed[c] = m.prod(fuel_per_cargo[c],
                                      m.prod(eca_frac_const, result.eca_fuel[c]));
            hfo_consumed[c] = m.sum({fuel_per_cargo[c], m.neg(mgo_consumed[c])});
        } else {
            hfo_consumed[c] = fuel_per_cargo[c];
            mgo_consumed[c] = zero;
        }
    }

    // ---------- Fuel cost: use average prices ----------
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

    // ---------- Objective: maximize profit ----------
    std::vector<int32_t> profit_terms;

    // Revenue from carried cargoes
    for (int c = 0; c < C; ++c) {
        auto rev = m.constant(inst.cargoes[c].revenue);
        profit_terms.push_back(m.prod(rev, active[c]));
    }

    // Fuel cost: avg_price * sum(fuel consumed)
    auto hfo_price = m.constant(avg_hfo_price);
    auto mgo_price = m.constant(avg_mgo_price);
    for (int c = 0; c < C; ++c) {
        profit_terms.push_back(m.neg(m.prod(hfo_price, hfo_consumed[c])));
        profit_terms.push_back(m.neg(m.prod(mgo_price, mgo_consumed[c])));
    }

    // Port costs
    for (int c = 0; c < C; ++c) {
        double port_cost = inst.regions[inst.cargoes[c].pickup_region].port_cost
                         + inst.regions[inst.cargoes[c].delivery_region].port_cost;
        auto pc = m.constant(port_cost);
        profit_terms.push_back(m.neg(m.prod(pc, active[c])));
    }

    m.maximize(m.sum(profit_terms));

    // ---------- Constraints ----------

    // 1. Per-ship fuel capacity: total fuel consumed by cargoes on ship v
    //    must not exceed tank_max (initial + bunkered)
    for (int v = 0; v < V; ++v) {
        // Max usable fuel = initial + max_bunkerable - safety_reserve
        double fuel_cap = inst.ships[v].initial_hfo
                        + (inst.ships[v].hfo_tank_max - inst.ships[v].initial_hfo)
                        - inst.ships[v].hfo_safety;
        // fuel consumed by ship v: sum_c (on_v[c][v] * hfo_consumed[c])
        std::vector<int32_t> ship_fuel_terms;
        for (int c = 0; c < C; ++c) {
            ship_fuel_terms.push_back(m.prod(on_v[c][v], hfo_consumed[c]));
        }
        auto ship_fuel = m.sum(ship_fuel_terms);
        m.add_constraint(m.sum({ship_fuel, m.neg(m.constant(fuel_cap))}));
    }

    // 2. Time window constraints
    for (int c = 0; c < C; ++c) {
        double dist = inst.leg_distance(inst.cargoes[c].pickup_region,
                                         inst.cargoes[c].delivery_region);
        if (dist <= 0.0) continue;

        auto dist_const = m.constant(dist);
        auto travel_time = m.div_expr(dist_const,
                                       m.prod(twenty_four, result.speed[c]));

        double available = inst.cargoes[c].delivery_tw_end
                         - inst.cargoes[c].pickup_tw_start
                         - inst.cargoes[c].service_time_load
                         - inst.cargoes[c].service_time_discharge;

        if (available > 0) {
            auto avail_const = m.constant(available);
            m.add_constraint(m.sum({travel_time,
                                    m.neg(m.prod(avail_const, active[c]))}));
        }
    }

    // 3. ECA compliance: fully-ECA legs must use MGO
    for (int c = 0; c < C; ++c) {
        double eca_frac = inst.leg_eca_fraction(
            inst.cargoes[c].pickup_region, inst.cargoes[c].delivery_region);
        if (eca_frac >= 0.99 && result.eca_fuel[c] >= 0) {
            m.add_constraint(m.sum({one, m.neg(result.eca_fuel[c])}));
        }
    }

    // 4. Ship workload balance
    int max_per_ship = std::max(4, (C + V - 1) / V * 3);
    auto max_cargo_const = m.constant(static_cast<double>(max_per_ship));
    for (int v = 0; v < V; ++v) {
        std::vector<int32_t> ship_load;
        for (int c = 0; c < C; ++c) {
            ship_load.push_back(on_v[c][v]);
        }
        m.add_constraint(m.sum({m.sum(ship_load), m.neg(max_cargo_const)}));
    }

    m.close();
    return result;
}

}  // namespace bunker_eca
}  // namespace cbls
