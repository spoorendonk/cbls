#include <cbls/cbls.h>
#include "data.h"
#include "bunker_eca_model.h"
#include "bunker_speed_hook.h"
#include <cstdio>
#include <string>
#include <vector>

// For maximize models, the solver stores -objective internally.
// Return the actual objective (positive for profit).
static double actual_objective(const cbls::SearchResult& result, const cbls::Model& model) {
    if (!result.feasible) return result.objective;
    return model.is_maximizing() ? -result.objective : result.objective;
}

int main(int argc, char** argv) {
    std::string inst_dir = "benchmarks/instances/bunker-eca";
    if (argc > 1) {
        inst_dir = argv[1];
    }

    struct InstanceSpec {
        std::string label;
        cbls::bunker_eca::Instance (*factory)();
        double time_limit;
    };

    std::vector<InstanceSpec> specs = {
        {"small",   cbls::bunker_eca::make_small,   30.0},
        {"medium",  cbls::bunker_eca::make_medium,  120.0},
        {"large",   cbls::bunker_eca::make_large,   300.0},
    };

    // Header
    printf("%-20s %5s %5s %5s %12s %8s %7s\n",
           "Instance", "Ships", "Cargo", "Regs", "Profit($)", "Feasbl", "Time(s)");
    printf("%-20s %5s %5s %5s %12s %8s %7s\n",
           "--------", "-----", "-----", "----", "---------", "------", "-------");

    for (auto& spec : specs) {
        auto inst = spec.factory();

        printf("%-20s %5d %5d %5d ",
               inst.name.c_str(),
               (int)inst.ships.size(),
               (int)inst.cargoes.size(),
               (int)inst.regions.size());
        fflush(stdout);

        auto bec = cbls::bunker_eca::build_bunker_eca_model(inst);

        cbls::bunker_eca::BunkerSpeedHook hook;
        hook.set_model(&bec, &inst);
        cbls::LNS lns(0.3);

        auto result = cbls::solve(bec.model, spec.time_limit, 42, true,
                                   &hook, &lns);

        double obj = actual_objective(result, bec.model);
        printf("%12.0f %8s %6.1fs",
               result.feasible ? obj : -1.0,
               result.feasible ? "yes" : "NO",
               result.time_seconds);

        printf("  (%ld vars, %ld nodes, %ld iters)\n",
               (long)bec.model.num_vars(),
               (long)bec.model.num_nodes(),
               (long)result.iterations);
    }

    // No-ECA mode comparison
    printf("\n--- No-ECA mode (comparison) ---\n");
    printf("%-20s %5s %5s %5s %12s %8s %7s\n",
           "Instance", "Ships", "Cargo", "Regs", "Profit($)", "Feasbl", "Time(s)");
    printf("%-20s %5s %5s %5s %12s %8s %7s\n",
           "--------", "-----", "-----", "----", "---------", "------", "-------");

    for (auto& spec : specs) {
        auto inst = spec.factory();

        for (auto& leg : inst.legs) {
            leg.eca_fraction = 0.0;
        }

        printf("%-20s %5d %5d %5d ",
               (inst.name + "-noECA").c_str(),
               (int)inst.ships.size(),
               (int)inst.cargoes.size(),
               (int)inst.regions.size());
        fflush(stdout);

        auto bec = cbls::bunker_eca::build_bunker_eca_model(inst);

        cbls::bunker_eca::BunkerSpeedHook hook;
        hook.set_model(&bec, &inst);
        cbls::LNS lns(0.3);

        auto result = cbls::solve(bec.model, spec.time_limit, 42, true,
                                   &hook, &lns);

        double obj = actual_objective(result, bec.model);
        printf("%12.0f %8s %6.1fs",
               result.feasible ? obj : -1.0,
               result.feasible ? "yes" : "NO",
               result.time_seconds);

        printf("  (%ld vars, %ld nodes, %ld iters)\n",
               (long)bec.model.num_vars(),
               (long)bec.model.num_nodes(),
               (long)result.iterations);
    }

    return 0;
}
