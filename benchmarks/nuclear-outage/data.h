#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <stdexcept>

namespace cbls {
namespace nuclear_outage {

struct NuclearInstance {
    std::string name;
    int n_units, n_periods, n_scenarios, n_outages, n_sites;

    // Unit data
    std::vector<double> capacity;       // MW per unit
    std::vector<double> min_power;      // MW minimum when online
    std::vector<double> fuel_cost;      // EUR/MWh marginal cost
    std::vector<int> site;              // site index per unit

    // Outage data (indexed by outage)
    std::vector<int> outage_unit;       // which unit this outage belongs to
    std::vector<int> outage_duration;   // weeks
    std::vector<int> outage_earliest;   // earliest start period
    std::vector<int> outage_latest;     // latest start period

    // Resource constraints
    int min_spacing_same_site;
    std::vector<int> max_outages_per_site;

    // Demand: [n_scenarios][n_periods] in MW
    std::vector<std::vector<double>> demand;

    // Cost
    double penalty_unserved;  // EUR/MWh for unserved energy

    // Known bounds: {method -> objective}
    std::map<std::string, double> known_bounds;
};

inline NuclearInstance load_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open instance file: " + path);
    }
    nlohmann::json j;
    f >> j;

    NuclearInstance inst;
    inst.name = j["name"];
    inst.n_units = j["n_units"];
    inst.n_periods = j["n_periods"];
    inst.n_scenarios = j["n_scenarios"];
    inst.n_outages = j["n_outages"];
    inst.n_sites = j["n_sites"];

    inst.capacity = j["capacity"].get<std::vector<double>>();
    inst.min_power = j["min_power"].get<std::vector<double>>();
    inst.fuel_cost = j["fuel_cost"].get<std::vector<double>>();
    inst.site = j["site"].get<std::vector<int>>();

    inst.outage_unit = j["outage_unit"].get<std::vector<int>>();
    inst.outage_duration = j["outage_duration"].get<std::vector<int>>();
    inst.outage_earliest = j["outage_earliest"].get<std::vector<int>>();
    inst.outage_latest = j["outage_latest"].get<std::vector<int>>();

    inst.min_spacing_same_site = j["min_spacing_same_site"];
    inst.max_outages_per_site = j["max_outages_per_site"].get<std::vector<int>>();

    // Demand: array of arrays
    inst.demand.resize(inst.n_scenarios);
    for (int s = 0; s < inst.n_scenarios; ++s) {
        inst.demand[s] = j["demand"][s].get<std::vector<double>>();
    }

    inst.penalty_unserved = j["penalty_unserved"];

    for (auto& [key, val] : j["known_bounds"].items()) {
        inst.known_bounds[key] = val.get<double>();
    }

    return inst;
}

/// Create a sub-instance with fewer scenarios (for fast dev/testing).
inline NuclearInstance make_mini(const NuclearInstance& inst, int n_scenarios) {
    if (n_scenarios < 1 || n_scenarios > inst.n_scenarios) {
        throw std::invalid_argument("n_scenarios out of range");
    }
    NuclearInstance sub = inst;
    sub.n_scenarios = n_scenarios;
    sub.demand.resize(n_scenarios);
    sub.name = inst.name + "-" + std::to_string(n_scenarios) + "sc";
    sub.known_bounds.clear();
    return sub;
}

}  // namespace nuclear_outage
}  // namespace cbls
