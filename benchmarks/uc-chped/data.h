#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <stdexcept>

namespace cbls {
namespace uc_chped {

struct UCInstance {
    std::string name;
    int n_units, n_periods;
    // Cost coefficients (valve-point)
    std::vector<double> a, b, c, d, e, P_min, P_max;
    // UC parameters
    std::vector<int> min_on, min_off, t_cold, n_init, y_prev;
    std::vector<double> a_hot, a_cold;
    // Multi-period
    std::vector<double> demand, reserve;
    // Known bounds: {n_periods -> (lb, ub)}
    std::map<int, std::pair<double, double>> known_bounds;
};

inline UCInstance load_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open instance file: " + path);
    }
    nlohmann::json j;
    f >> j;

    UCInstance inst;
    inst.name = j["name"];
    inst.n_units = j["n_units"];
    inst.n_periods = j["n_periods"];

    inst.a = j["a"].get<std::vector<double>>();
    inst.b = j["b"].get<std::vector<double>>();
    inst.c = j["c"].get<std::vector<double>>();
    inst.d = j["d"].get<std::vector<double>>();
    inst.e = j["e"].get<std::vector<double>>();
    inst.P_min = j["P_min"].get<std::vector<double>>();
    inst.P_max = j["P_max"].get<std::vector<double>>();

    inst.min_on = j["min_on"].get<std::vector<int>>();
    inst.min_off = j["min_off"].get<std::vector<int>>();
    inst.t_cold = j["t_cold"].get<std::vector<int>>();
    inst.n_init = j["n_init"].get<std::vector<int>>();
    inst.y_prev = j["y_prev"].get<std::vector<int>>();
    inst.a_hot = j["a_hot"].get<std::vector<double>>();
    inst.a_cold = j["a_cold"].get<std::vector<double>>();

    inst.demand = j["demand"].get<std::vector<double>>();
    inst.reserve = j["reserve"].get<std::vector<double>>();

    for (auto& [key, val] : j["known_bounds"].items()) {
        int np = std::stoi(key);
        inst.known_bounds[np] = {val[0].get<double>(), val[1].get<double>()};
    }

    return inst;
}

inline UCInstance make_subinstance(const UCInstance& inst, int n_periods) {
    if (n_periods < 1 || n_periods > inst.n_periods) {
        throw std::invalid_argument("n_periods out of range");
    }
    UCInstance sub = inst;
    sub.n_periods = n_periods;
    sub.demand.resize(n_periods);
    sub.reserve.resize(n_periods);
    sub.name = inst.name + "-" + std::to_string(n_periods) + "p";
    // Retain only the matching bound
    auto it = inst.known_bounds.find(n_periods);
    sub.known_bounds.clear();
    if (it != inst.known_bounds.end()) {
        sub.known_bounds[n_periods] = it->second;
    }
    return sub;
}

}  // namespace uc_chped
}  // namespace cbls
