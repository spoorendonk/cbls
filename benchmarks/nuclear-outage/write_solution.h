#pragma once

#include "data.h"
#include "nuclear_model.h"
#include "roadef_dispatch.h"
#include <cbls/cbls.h>
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
    std::ofstream out(output_path);
    if (!out.is_open()) return false;

    int O = inst.n_outages();

    // Read outage dates from model
    std::vector<int> ha(O);
    for (int o = 0; o < O; ++o) {
        int32_t vid = handle_to_var_id(rm.ha[o]);
        ha[o] = static_cast<int>(std::round(model.var(vid).value));
    }

    // Build shared dispatch structures
    auto lookup = build_outage_lookup(inst, ha);
    auto status = compute_plant_status(inst, lookup);
    auto reload = compute_reloads(inst, lookup);

    // ---- Main section ----
    out << "begin main\n";
    out << "team_identifier cbls\n";

    auto now = std::time(nullptr);
    auto tm = std::localtime(&now);
    char datebuf[64];
    std::strftime(datebuf, sizeof(datebuf), "%d/%m/%y %H:%M:%S", tm);
    out << "solution_time_date " << datebuf << "\n";

    int hours = (int)(solve_time_seconds / 3600);
    int mins = (int)(std::fmod(solve_time_seconds, 3600.0) / 60);
    int secs = (int)std::fmod(solve_time_seconds, 60.0);
    char timebuf[32];
    std::snprintf(timebuf, sizeof(timebuf), "%02d:%02d:%02d", hours, mins, secs);
    out << "solution_running_time " << timebuf << "\n";
    out << "data_set " << data_file << "\n";
    out << std::fixed << std::setprecision(6);
    out << "cost " << objective << "\n";
    out << "end main\n";

    // ---- Outages section ----
    out << "begin outages\n";
    for (int i = 0; i < inst.n_type2; ++i) {
        auto& plant = inst.type2_plants[i];
        out << "name " << plant.name << "\n";
        out << "index " << plant.index << "\n";

        out << "outage_dates";
        for (int k = 0; k < plant.n_cycles; ++k) {
            out << " " << lookup[i][k];
        }
        out << "\n";

        out << "reloaded_fuel";
        for (int k = 0; k < plant.n_cycles; ++k) {
            out << " " << std::fixed << std::setprecision(6) << reload[i][k];
        }
        out << "\n";
    }
    out << "end outages\n";

    // ---- Power output section ----
    out << "begin power_output\n";
    for (int s = 0; s < inst.S; ++s) {
        out << "scenario " << s << "\n";

        auto sr = simulate_scenario(inst, status, reload, s, true);

        // Write Type 1 plants
        out << "begin type1_plants\n";
        for (int j = 0; j < inst.n_type1; ++j) {
            out << "name " << inst.type1_plants[j].name
                << " " << inst.type1_plants[j].index;
            for (int t = 0; t < inst.T; ++t) {
                out << " " << std::fixed << std::setprecision(6) << sr.t1_prod[j][t];
            }
            out << "\n";
        }
        out << "end type1_plants\n";

        // Write Type 2 plants
        out << "begin type2_plants\n";
        for (int i = 0; i < inst.n_type2; ++i) {
            out << "name " << inst.type2_plants[i].name
                << " " << inst.type2_plants[i].index;
            for (int t = 0; t < inst.T; ++t) {
                out << " " << std::fixed << std::setprecision(6) << sr.t2_prod[i][t];
            }
            out << "\n";

            out << "fuel_variation";
            for (int t = 0; t < inst.T; ++t) {
                out << " " << std::fixed << std::setprecision(6) << sr.fuel_at_t[i][t];
            }
            out << "\n";

            out << "remaining_fuel_at_the_end "
                << std::fixed << std::setprecision(6) << sr.fuel_at_t[i][inst.T] << "\n";
        }
        out << "end type2_plants\n";
    }
    out << "end power_output\n";

    return true;
}

}  // namespace nuclear_outage
}  // namespace cbls
