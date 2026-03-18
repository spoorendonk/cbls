#include <cbls/cbls.h>
#include "data.h"
#include "glsp_model.h"
#include "glsp_hook.h"
#include <cstdio>
#include <string>
#include <map>

int main(int argc, char** argv) {
    std::string inst_dir = "benchmarks/instances/pharma-glsp";
    double time_limit = 30.0;
    int max_instances = 0;  // 0 = all
    std::string class_filter;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dir" && i + 1 < argc) {
            inst_dir = argv[++i];
        } else if (arg == "--time" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (arg == "--max" && i + 1 < argc) {
            max_instances = std::stoi(argv[++i]);
        } else if (arg == "--class" && i + 1 < argc) {
            class_filter = argv[++i];
        }
    }

    // Load instances
    std::vector<cbls::glsp::GLSPInstance> instances;
    std::vector<std::string> files = {"class_a.jsonl", "class_b.jsonl", "class_c.jsonl"};
    if (class_filter.empty()) {
        for (const auto& f : files) {
            try {
                auto loaded = cbls::glsp::load_jsonl(inst_dir + "/" + f);
                instances.insert(instances.end(), loaded.begin(), loaded.end());
            } catch (const std::exception& e) {
                fprintf(stderr, "Warning: %s\n", e.what());
            }
        }
    } else {
        std::string f = "class_" + class_filter + ".jsonl";
        instances = cbls::glsp::load_jsonl(inst_dir + "/" + f);
    }

    if (instances.empty()) {
        fprintf(stderr, "No instances loaded. Run gen_jsonl.py first.\n");
        return 1;
    }

    if (max_instances > 0 && static_cast<int>(instances.size()) > max_instances) {
        instances.resize(max_instances);
    }

    printf("%-20s %4s %4s %4s %12s %8s %8s %10s\n",
           "Instance", "J", "T", "M", "Objective", "Feasible", "Iters", "Time(s)");
    printf("%-20s %4s %4s %4s %12s %8s %8s %10s\n",
           "--------", "--", "--", "--", "---------", "--------", "-----", "-------");

    // Per-class statistics
    struct ClassStats {
        int count = 0;
        int feasible = 0;
        double total_obj = 0;
        double total_time = 0;
    };
    std::map<std::string, ClassStats> stats;

    for (const auto& inst : instances) {
        auto gm = cbls::glsp::build_glsp_model(inst);

        cbls::glsp::GLSPInnerSolverHook hook(inst, gm.seq, gm.lot);
        cbls::LNS lns(0.3);

        auto result = cbls::solve(gm.model, time_limit, 42, true, &hook, &lns);

        printf("%-20s %4d %4d %4d %12.1f %8s %8ld %9.1fs\n",
               inst.name.c_str(), inst.n_products, inst.n_macro,
               inst.n_micro_per_macro,
               result.feasible ? result.objective : -1.0,
               result.feasible ? "yes" : "NO",
               (long)result.iterations, result.time_seconds);

        auto& s = stats[inst.cls];
        s.count++;
        if (result.feasible) {
            s.feasible++;
            s.total_obj += result.objective;
        }
        s.total_time += result.time_seconds;
    }

    printf("\n%-10s %6s %6s %12s %10s\n",
           "Class", "N", "Feas%", "Avg Obj", "Avg Time");
    printf("%-10s %6s %6s %12s %10s\n",
           "-----", "--", "-----", "-------", "--------");
    for (const auto& [cls, s] : stats) {
        double feas_pct = 100.0 * s.feasible / std::max(s.count, 1);
        double avg_obj = s.feasible > 0 ? s.total_obj / s.feasible : 0;
        double avg_time = s.total_time / std::max(s.count, 1);
        printf("%-10s %6d %5.1f%% %12.1f %9.1fs\n",
               cls.c_str(), s.count, feas_pct, avg_obj, avg_time);
    }

    return 0;
}
