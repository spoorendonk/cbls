#pragma once

#include <vector>
#include <string>

namespace cbls {
namespace chped {

struct Instance {
    std::string name;
    int n_units;
    int n_periods;
    std::vector<double> a, b, c, d, e;
    std::vector<double> P_min, P_max;
    std::vector<double> demand;
    std::vector<double> reserve;
};

inline Instance make_4unit() {
    return {
        "4-unit", 4, 1,
        {25.0, 60.0, 100.0, 120.0},       // a
        {2.0, 1.8, 2.1, 2.0},              // b
        {0.008, 0.006, 0.009, 0.007},       // c
        {100.0, 140.0, 160.0, 180.0},       // d
        {0.042, 0.040, 0.038, 0.037},       // e
        {10.0, 20.0, 30.0, 40.0},           // P_min
        {75.0, 125.0, 175.0, 250.0},        // P_max
        {400.0},                             // demand
        {0.0},                               // reserve
    };
}

inline Instance make_7unit() {
    return {
        "7-unit", 7, 1,
        {25.0, 60.0, 100.0, 120.0, 40.0, 70.0, 110.0},
        {2.0, 1.8, 2.1, 2.0, 1.9, 2.2, 1.7},
        {0.008, 0.006, 0.009, 0.007, 0.008, 0.005, 0.006},
        {100.0, 140.0, 160.0, 180.0, 120.0, 150.0, 130.0},
        {0.042, 0.040, 0.038, 0.037, 0.041, 0.039, 0.043},
        {10.0, 20.0, 30.0, 40.0, 15.0, 25.0, 35.0},
        {75.0, 125.0, 175.0, 250.0, 100.0, 150.0, 200.0},
        {800.0},
        {0.0},
    };
}

inline Instance make_24unit() {
    auto base = make_4unit();
    Instance inst;
    inst.name = "24-unit";
    inst.n_units = 24;
    inst.n_periods = 1;
    inst.demand = {2000.0};
    inst.reserve = {100.0};

    // Simple deterministic perturbation (no RNG dependency)
    double scales[] = {
        0.95, 1.10, 0.85, 1.15, 1.00, 0.90, 1.05, 0.80,
        1.20, 0.88, 1.12, 0.92, 1.08, 0.83, 1.17, 0.97,
        1.03, 0.87, 1.13, 0.91, 1.09, 0.84, 1.16, 0.96
    };

    for (int i = 0; i < 24; ++i) {
        int j = i % 4;
        double s = scales[i];
        inst.a.push_back(base.a[j] * s);
        inst.b.push_back(base.b[j] * s);
        inst.c.push_back(base.c[j] * s);
        inst.d.push_back(base.d[j] * s);
        inst.e.push_back(base.e[j] * s);
        inst.P_min.push_back(base.P_min[j] * s);
        inst.P_max.push_back(base.P_max[j] * std::max(s, 1.0));
    }

    return inst;
}

}  // namespace chped
}  // namespace cbls
