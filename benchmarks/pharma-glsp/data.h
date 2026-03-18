#pragma once

#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <stdexcept>

namespace cbls {
namespace glsp {

struct GLSPInstance {
    std::string name;
    std::string cls;
    int n_products;       // J
    int n_macro;          // T
    int n_micro_per_macro; // |M_t|

    // demand[j][t]
    std::vector<std::vector<double>> demand;
    // setup_cost[i][j], setup_time[i][j]
    std::vector<std::vector<double>> setup_cost;
    std::vector<std::vector<double>> setup_time;
    // Per-product parameters
    std::vector<double> process_time;
    std::vector<double> rework_time;
    std::vector<double> holding_cost;
    std::vector<double> rework_holding_cost;
    std::vector<double> min_lot;
    std::vector<double> disposal_cost;
    std::vector<int> lifetime;  // Omega_j in micro-periods
    // defect_rate[j][t]
    std::vector<std::vector<double>> defect_rate;
    // capacity[t]
    std::vector<double> capacity;

    int total_micro_periods() const { return n_macro * n_micro_per_macro; }
};

inline GLSPInstance load_jsonl_line(const nlohmann::json& j) {
    GLSPInstance inst;
    inst.name = j["name"];
    inst.cls = j["cls"];
    inst.n_products = j["n_products"];
    inst.n_macro = j["n_macro"];
    inst.n_micro_per_macro = j["n_micro_per_macro"];

    inst.demand = j["demand"].get<std::vector<std::vector<double>>>();
    inst.setup_cost = j["setup_cost"].get<std::vector<std::vector<double>>>();
    inst.setup_time = j["setup_time"].get<std::vector<std::vector<double>>>();
    inst.process_time = j["process_time"].get<std::vector<double>>();
    inst.rework_time = j["rework_time"].get<std::vector<double>>();
    inst.holding_cost = j["holding_cost"].get<std::vector<double>>();
    inst.rework_holding_cost = j["rework_holding_cost"].get<std::vector<double>>();
    inst.min_lot = j["min_lot"].get<std::vector<double>>();
    inst.disposal_cost = j["disposal_cost"].get<std::vector<double>>();
    inst.lifetime = j["lifetime"].get<std::vector<int>>();
    inst.defect_rate = j["defect_rate"].get<std::vector<std::vector<double>>>();
    inst.capacity = j["capacity"].get<std::vector<double>>();

    return inst;
}

inline std::vector<GLSPInstance> load_jsonl(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open instance file: " + path);
    }
    std::vector<GLSPInstance> instances;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        auto j = nlohmann::json::parse(line);
        instances.push_back(load_jsonl_line(j));
    }
    return instances;
}

// Small inline instance for unit tests (no file I/O needed)
inline GLSPInstance make_tiny() {
    GLSPInstance inst;
    inst.name = "tiny_test";
    inst.cls = "test";
    inst.n_products = 3;
    inst.n_macro = 2;
    inst.n_micro_per_macro = 5;

    // demand[j][t]
    inst.demand = {{50, 60}, {40, 50}, {30, 40}};
    // setup_cost[i][j]
    inst.setup_cost = {
        {0, 200, 300},
        {250, 0, 150},
        {200, 180, 0}
    };
    // setup_time[i][j] = cost/10
    inst.setup_time = {
        {0, 20, 30},
        {25, 0, 15},
        {20, 18, 0}
    };
    inst.process_time = {1.0, 1.0, 1.0};
    inst.rework_time = {0.5, 0.5, 0.5};
    inst.holding_cost = {15.0, 12.0, 18.0};
    inst.rework_holding_cost = {3.0, 2.4, 3.6};
    inst.min_lot = {10.0, 10.0, 10.0};
    inst.disposal_cost = {1000.0, 1000.0, 1000.0};
    inst.lifetime = {3, 3, 3};
    inst.defect_rate = {{0.01, 0.02}, {0.0, 0.01}, {0.015, 0.0}};
    // capacity: generous for test
    inst.capacity = {500.0, 500.0};

    return inst;
}

}  // namespace glsp
}  // namespace cbls
