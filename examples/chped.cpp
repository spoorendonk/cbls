#include <cbls/cbls.h>
#include "benchmarks/chped/data.h"
#include "benchmarks/chped/chped_model.h"
#include <cstdio>

int main() {
    auto inst = cbls::chped::make_4unit();
    auto cm = cbls::chped::build_chped_model(inst);
    auto& m = cm.model;

    printf("CHPED %s: %d units, demand=%.0f MW\n",
           inst.name.c_str(), inst.n_units, inst.demand[0]);

    auto result = cbls::solve(m, 5.0, 42);
    printf("feasible = %s\n", result.feasible ? "true" : "false");
    printf("objective = %.2f\n", result.objective);
    printf("iterations = %ld\n", result.iterations);
    printf("time = %.3fs\n", result.time_seconds);

    return 0;
}
