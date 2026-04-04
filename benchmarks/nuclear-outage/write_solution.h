#pragma once

#include "data.h"
#include "nuclear_model.h"
#include "roadef_hook.h"
#include <fstream>
#include <cstdio>
#include <ctime>
#include <iomanip>

namespace cbls {
namespace nuclear_outage {

// Write a solution in ROADEF 2010 competition output format (§4.2)
inline bool write_roadef_solution(
    const ROADEFInstance& inst,
    const ROADEFModel& rm,
    Model& model,
    const std::string& data_file,
    const std::string& output_path,
    double objective,
    double solve_time_seconds)
{
    std::ofstream f(output_path);
    if (!f.is_open()) return false;

    int O = inst.n_outages();
    int tpw = inst.timesteps_per_week;

    // Read outage dates from model
    std::vector<int> ha(O);
    for (int o = 0; o < O; ++o) {
        int32_t vid = handle_to_var_id(rm.ha[o]);
        ha[o] = static_cast<int>(std::round(model.var(vid).value));
    }

    // ---- Main section ----
    f << "begin main\n";
    f << "team_identifier cbls\n";

    // Current date/time
    auto now = std::time(nullptr);
    auto tm = std::localtime(&now);
    char datebuf[64];
    std::strftime(datebuf, sizeof(datebuf), "%d/%m/%y %H:%M:%S", tm);
    f << "solution_time_date " << datebuf << "\n";

    int hours = (int)(solve_time_seconds / 3600);
    int mins = (int)(std::fmod(solve_time_seconds, 3600.0) / 60);
    int secs = (int)std::fmod(solve_time_seconds, 60.0);
    char timebuf[32];
    std::snprintf(timebuf, sizeof(timebuf), "%02d:%02d:%02d", hours, mins, secs);
    f << "solution_running_time " << timebuf << "\n";
    f << "data_set " << data_file << "\n";
    f << std::fixed << std::setprecision(6);
    f << "cost " << objective << "\n";
    f << "end main\n";

    // ---- Outages section ----
    f << "begin outages\n";

    // Compute reload amounts (same logic as hook)
    std::vector<std::vector<double>> reload(inst.n_type2);
    for (int i = 0; i < inst.n_type2; ++i) {
        auto& plant = inst.type2_plants[i];
        reload[i].resize(plant.n_cycles, 0.0);
        for (int k = 0; k < plant.n_cycles; ++k) {
            // Check if this outage is scheduled
            bool scheduled = false;
            for (int o = 0; o < O; ++o) {
                if (inst.ct13[o].plant_idx == i && inst.ct13[o].cycle == k) {
                    scheduled = true;
                    break;
                }
            }
            if (scheduled) {
                reload[i][k] = plant.rmax[k];  // same as hook
            }
        }
    }

    for (int i = 0; i < inst.n_type2; ++i) {
        auto& plant = inst.type2_plants[i];
        f << "name " << plant.name << "\n";
        f << "index " << plant.index << "\n";

        // outage_dates: ha(i,0), ha(i,1), ..., ha(i,K-1)
        f << "outage_dates";
        for (int k = 0; k < plant.n_cycles; ++k) {
            int start = -1;
            for (int o = 0; o < O; ++o) {
                if (inst.ct13[o].plant_idx == i && inst.ct13[o].cycle == k) {
                    start = ha[o];
                    break;
                }
            }
            f << " " << start;
        }
        f << "\n";

        // reloaded_fuel: r(i,0), r(i,1), ..., r(i,K-1)
        f << "reloaded_fuel";
        for (int k = 0; k < plant.n_cycles; ++k) {
            f << " " << std::fixed << std::setprecision(6) << reload[i][k];
        }
        f << "\n";
    }
    f << "end outages\n";

    // ---- Power output section ----
    f << "begin power_output\n";

    // Build plant status arrays (same as hook)
    std::vector<std::vector<bool>> in_outage(inst.n_type2,
        std::vector<bool>(inst.H, false));
    std::vector<std::vector<int>> cycle_at_week(inst.n_type2,
        std::vector<int>(inst.H, -1));

    for (int i = 0; i < inst.n_type2; ++i) {
        auto& plant = inst.type2_plants[i];
        int current_cycle = -1;
        struct Evt { int start, cycle, dur; };
        std::vector<Evt> events;
        for (int k = 0; k < plant.n_cycles; ++k) {
            for (int o = 0; o < O; ++o) {
                if (inst.ct13[o].plant_idx == i && inst.ct13[o].cycle == k) {
                    events.push_back({ha[o], k, plant.durations[k]});
                    break;
                }
            }
        }
        std::sort(events.begin(), events.end(),
                  [](const Evt& a, const Evt& b) { return a.start < b.start; });

        int next_ev = 0;
        current_cycle = -1;
        for (int h = 0; h < inst.H; ++h) {
            if (next_ev < (int)events.size() && h == events[next_ev].start) {
                int dur = events[next_ev].dur;
                for (int w = h; w < std::min(h + dur, inst.H); ++w) {
                    in_outage[i][w] = true;
                    cycle_at_week[i][w] = events[next_ev].cycle;
                }
                current_cycle = events[next_ev].cycle;
                h = std::min(h + dur - 1, inst.H - 1);
                ++next_ev;
            } else {
                cycle_at_week[i][h] = current_cycle;
            }
        }
    }

    for (int s = 0; s < inst.S; ++s) {
        f << "scenario " << s << "\n";

        // Simulate this scenario to get production and fuel levels
        std::vector<double> fuel(inst.n_type2);
        for (int i = 0; i < inst.n_type2; ++i) {
            fuel[i] = inst.type2_plants[i].initial_stock;
        }

        // Store per-timestep production for all plants
        std::vector<std::vector<double>> t1_prod(inst.n_type1,
            std::vector<double>(inst.T, 0.0));
        std::vector<std::vector<double>> t2_prod(inst.n_type2,
            std::vector<double>(inst.T, 0.0));
        std::vector<std::vector<double>> fuel_at_t(inst.n_type2,
            std::vector<double>(inst.T + 1, 0.0));

        for (int i = 0; i < inst.n_type2; ++i) {
            fuel_at_t[i][0] = fuel[i];
        }

        // Forward simulation
        for (int t = 0; t < inst.T; ++t) {
            int week = t / tpw;
            double dt = inst.timestep_durations[t];
            double demand = inst.demand[s][t];

            double total_t2 = 0.0;
            for (int i = 0; i < inst.n_type2; ++i) {
                auto& plant = inst.type2_plants[i];

                if (week >= inst.H || in_outage[i][week]) {
                    t2_prod[i][t] = 0.0;
                    // Refueling at first timestep of outage
                    if (in_outage[i][week] && (t % tpw) == 0 &&
                        week > 0 && !in_outage[i][week - 1]) {
                        int k = cycle_at_week[i][week];
                        if (k >= 0 && k < plant.n_cycles) {
                            double q = plant.q[k];
                            double bo_prev = (k > 0) ? plant.bo[k] : plant.bo[0];
                            double bo_k = plant.bo[k + 1];
                            fuel[i] = ((q - 1.0) / q) * (fuel[i] - bo_prev) +
                                      reload[i][k] + bo_k;
                            fuel[i] = std::min(fuel[i], plant.smax[k]);
                        }
                    }
                } else {
                    int ck = cycle_at_week[i][week];
                    int prof_idx = (ck < 0) ? 0 : (ck + 1);
                    double bo = plant.bo[prof_idx];
                    double pmax = plant.pmax_t[t];

                    double max_prod;
                    if (fuel[i] >= bo) {
                        max_prod = pmax;
                    } else {
                        auto& prof = plant.profiles[prof_idx];
                        double pb = prof.evaluate(fuel[i]);
                        max_prod = pb * pmax;
                        if (fuel[i] < max_prod * dt) {
                            max_prod = std::max(0.0, fuel[i] / dt);
                        }
                    }
                    double prod = std::max(0.0, std::min(max_prod, fuel[i] / dt));
                    t2_prod[i][t] = prod;
                    total_t2 += prod;
                    fuel[i] -= prod * dt;
                    fuel[i] = std::max(0.0, fuel[i]);
                }
                fuel_at_t[i][t + 1] = fuel[i];
            }

            // Type 1 dispatch
            double remaining = demand - total_t2;
            if (remaining > 0.0 && inst.n_type1 > 0) {
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
                    double pmax_j = p1.pmax[s][t];
                    if (pmax_j <= 0.0) continue;
                    double gen = std::min(pmax_j, remaining);
                    t1_prod[j][t] = gen;
                    remaining -= gen;
                }
            }
        }

        // Write Type 1 plants
        f << "begin type1_plants\n";
        for (int j = 0; j < inst.n_type1; ++j) {
            f << "name " << inst.type1_plants[j].name
              << " " << inst.type1_plants[j].index;
            for (int t = 0; t < inst.T; ++t) {
                f << " " << std::fixed << std::setprecision(6) << t1_prod[j][t];
            }
            f << "\n";
        }
        f << "end type1_plants\n";

        // Write Type 2 plants
        f << "begin type2_plants\n";
        for (int i = 0; i < inst.n_type2; ++i) {
            f << "name " << inst.type2_plants[i].name
              << " " << inst.type2_plants[i].index;
            for (int t = 0; t < inst.T; ++t) {
                f << " " << std::fixed << std::setprecision(6) << t2_prod[i][t];
            }
            f << "\n";

            // fuel_variation: x(i,0,s), x(i,1,s), ..., x(i,T-1,s) — T values
            f << "fuel_variation";
            for (int t = 0; t < inst.T; ++t) {
                f << " " << std::fixed << std::setprecision(6) << fuel_at_t[i][t];
            }
            f << "\n";

            // remaining_fuel_at_the_end
            f << "remaining_fuel_at_the_end "
              << std::fixed << std::setprecision(6) << fuel_at_t[i][inst.T] << "\n";
        }
        f << "end type2_plants\n";
    }
    f << "end power_output\n";

    return true;
}

}  // namespace nuclear_outage
}  // namespace cbls
