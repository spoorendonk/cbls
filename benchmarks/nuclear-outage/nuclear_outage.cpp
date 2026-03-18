#include <cbls/cbls.h>
#include "data.h"
#include "nuclear_model.h"
#include "nuclear_hook.h"
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    std::string inst_dir = "benchmarks/instances/nuclear-outage";
    if (argc > 1) {
        inst_dir = argv[1];
    }

    struct RunConfig {
        std::string file;
        double time_limit;
        int scenarios_per_move;
    };

    std::vector<RunConfig> configs = {
        {"mini.jsonl", 10.0, 5},
        {"small.jsonl", 60.0, 20},
    };

    printf("%-20s %6s %6s %6s %6s %12s %8s\n",
           "Instance", "Units", "Outag", "Prd", "Scen", "Objective", "Time(s)");
    printf("%-20s %6s %6s %6s %6s %12s %8s\n",
           "--------", "-----", "-----", "---", "----", "---------", "-------");

    for (auto& cfg : configs) {
        std::string path = inst_dir + "/" + cfg.file;
        cbls::nuclear_outage::NuclearInstance inst;
        try {
            inst = cbls::nuclear_outage::load_jsonl(path);
        } catch (const std::exception& e) {
            printf("%-20s  SKIP: %s\n", cfg.file.c_str(), e.what());
            continue;
        }

        printf("%-20s %6d %6d %6d %6d ",
               inst.name.c_str(), inst.n_units, inst.n_outages,
               inst.n_periods, inst.n_scenarios);
        fflush(stdout);

        auto nm = cbls::nuclear_outage::build_nuclear_model(inst);

        cbls::nuclear_outage::NuclearDispatchHook hook(inst, nm);
        hook.scenarios_per_move = cfg.scenarios_per_move;
        cbls::LNS lns(0.3);

        auto result = cbls::solve(nm.model, cfg.time_limit, 42, true, &hook, &lns);

        printf("%12.0f %7.1fs", result.objective, result.time_seconds);
        printf("  (%s, %ld vars, %ld iters)\n",
               result.feasible ? "feasible" : "INFEASIBLE",
               (long)nm.model.num_vars(), (long)result.iterations);

        // Print solution details if feasible
        if (result.feasible) {
            nm.model.restore_state(result.best_state);
            printf("  Outage schedule: ");
            for (int o = 0; o < std::min(inst.n_outages, 20); ++o) {
                int32_t vid = cbls::nuclear_outage::handle_to_var_id(nm.s[o]);
                printf("o%d@w%d ", o, (int)nm.model.var(vid).value);
            }
            if (inst.n_outages > 20) printf("...");
            printf("\n");
        }
    }

    return 0;
}
