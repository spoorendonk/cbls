#include <cbls/cbls.h>
#include "data.h"
#include "nuclear_model.h"
#include "nuclear_hook.h"
#include "roadef_hook.h"
#include "write_solution.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static void run_synthetic(const std::string& inst_dir) {
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
}

static void run_roadef(const std::string& data_file,
                        const std::string& solution_file,
                        double time_limit) {
    using namespace cbls::nuclear_outage;

    printf("Loading ROADEF instance: %s\n", data_file.c_str());
    auto inst = load_roadef(data_file);

    printf("  Instance: %s\n", inst.name.c_str());
    printf("  Timesteps: %d, Weeks: %d, Scenarios: %d\n", inst.T, inst.H, inst.S);
    printf("  Type 1 plants: %d, Type 2 plants: %d\n", inst.n_type1, inst.n_type2);
    printf("  Schedulable outages: %d\n", inst.n_outages());
    printf("  Constraints: CT13=%d, CT14-18=%d, CT19=%d, CT20=%d, CT21=%d\n",
           (int)inst.ct13.size(), (int)inst.spacing_constraints.size(),
           (int)inst.ct19.size(), (int)inst.ct20.size(), (int)inst.ct21.size());

    // Build model
    auto rm = build_roadef_model(inst);
    printf("  Model: %ld vars, %ld nodes, %ld constraints\n",
           (long)rm.model.num_vars(), (long)rm.model.num_nodes(),
           (long)rm.model.constraint_ids().size());

    // Create hook
    ROADEFDispatchHook hook(inst, rm);
    if (inst.S > 50) {
        hook.scenarios_per_move = 50;
    }
    cbls::LNS lns(0.3);

    printf("  Solving for %.0f seconds...\n", time_limit);
    fflush(stdout);

    auto result = cbls::solve(rm.model, time_limit, 42, true, &hook, &lns);

    printf("  Result: %s, objective=%.2f, iters=%ld, time=%.1fs\n",
           result.feasible ? "feasible" : "INFEASIBLE",
           result.objective, (long)result.iterations, result.time_seconds);

    if (result.feasible) {
        rm.model.restore_state(result.best_state);

        // Print outage schedule
        printf("  Outage schedule:");
        for (int o = 0; o < inst.n_outages(); ++o) {
            int32_t vid = handle_to_var_id(rm.ha[o]);
            int ha = (int)rm.model.var(vid).value;
            printf(" [p%d.k%d@w%d]", inst.ct13[o].plant_idx,
                   inst.ct13[o].cycle, ha);
        }
        printf("\n");

        // Write solution file
        if (!solution_file.empty()) {
            printf("  Writing solution to: %s\n", solution_file.c_str());
            bool ok = write_roadef_solution(inst, rm, rm.model, data_file,
                                            solution_file, result.objective,
                                            result.time_seconds);
            if (!ok) {
                fprintf(stderr, "ERROR: Failed to write solution file\n");
            }
        }
    }
}

int main(int argc, char** argv) {
    std::string data_file;
    std::string solution_file;
    double time_limit = 1800.0;  // default 30 minutes
    bool roadef_mode = false;

    // Parse arguments: -n INSTANCE -r SOLUTION -t TIME
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            data_file = argv[++i];
            roadef_mode = true;
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            solution_file = argv[++i];
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0) {
            printf("cbls\n");
            return 0;
        } else if (!roadef_mode) {
            // Legacy mode: first arg is instance directory
            data_file = argv[i];
        }
    }

    if (roadef_mode && !data_file.empty()) {
        run_roadef(data_file, solution_file, time_limit);
    } else {
        std::string inst_dir = data_file.empty()
            ? "benchmarks/instances/nuclear-outage"
            : data_file;
        run_synthetic(inst_dir);
    }

    return 0;
}
