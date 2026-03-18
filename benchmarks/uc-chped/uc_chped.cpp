#include <cbls/cbls.h>
#include "data.h"
#include "uc_model.h"
#include <cstdio>
#include <map>
#include <string>
#include <vector>

struct InstanceSpec {
    std::string filename;
    std::vector<int> periods;
};

int main(int argc, char** argv) {
    // Default instance directory: relative to working directory
    std::string inst_dir = "benchmarks/instances/uc-chped";
    if (argc > 1) {
        inst_dir = argv[1];
    }

    // Time limits per number of periods
    std::map<int, double> time_limits = {
        {1, 10.0}, {3, 30.0}, {6, 60.0}, {12, 120.0}, {24, 300.0},
        {48, 600.0}, {168, 600.0},
    };

    // Instance specs: filename + period options
    std::vector<InstanceSpec> specs = {
        {"ucp13.jsonl",      {1, 3, 6, 12, 24}},
        {"ucp40.jsonl",      {1, 3, 6, 12, 24}},
        {"ucp100.jsonl",     {1, 3, 6, 12, 24}},
        {"ucp100-48p.jsonl", {48}},
        {"ucp100-168p.jsonl",{168}},
        {"ucp200.jsonl",     {1, 3, 6, 12, 24}},
        {"ucp200-48p.jsonl", {48}},
        {"ucp200-168p.jsonl",{168}},
    };

    printf("%-20s %6s %6s %12s %12s %8s %8s\n",
           "Instance", "Units", "Periods", "Objective", "Known LB", "Gap%", "Time(s)");
    printf("%-20s %6s %6s %12s %12s %8s %8s\n",
           "--------", "-----", "-------", "---------", "--------", "----", "-------");

    for (auto& spec : specs) {
        std::string path = inst_dir + "/" + spec.filename;
        cbls::uc_chped::UCInstance base;
        try {
            base = cbls::uc_chped::load_jsonl(path);
        } catch (const std::exception& e) {
            printf("Skipping %s: %s\n", spec.filename.c_str(), e.what());
            continue;
        }

        for (int T : spec.periods) {
            cbls::uc_chped::UCInstance inst;
            if (T <= base.n_periods) {
                inst = cbls::uc_chped::make_subinstance(base, T);
            } else {
                printf("%-20s %6d %6d  (skipped: T > n_periods)\n",
                       base.name.c_str(), base.n_units, T);
                continue;
            }
            auto ucm = cbls::uc_chped::build_uc_model(inst);

            double tlim = time_limits.count(T) ? time_limits[T] : 300.0;
            printf("%-20s %6d %6d ", inst.name.c_str(), inst.n_units, inst.n_periods);
            fflush(stdout);

            cbls::FloatIntensifyHook hook;
            cbls::LNS lns(0.3);
            auto result = cbls::solve(ucm.model, tlim, 42, true, &hook, &lns);

            // Compute gap vs known bounds
            auto it = inst.known_bounds.find(T);
            if (it != inst.known_bounds.end() && result.feasible) {
                double lb = it->second.first;
                double gap = 100.0 * (result.objective - lb) / lb;
                printf("%12.1f %12.1f %7.2f%% %7.1fs",
                       result.objective, lb, gap, result.time_seconds);
            } else {
                printf("%12.1f %12s %8s %7.1fs",
                       result.feasible ? result.objective : -1.0,
                       "—", result.feasible ? "—" : "INFEAS", result.time_seconds);
            }

            printf("  (%s, %ld vars, %ld nodes, %ld iters)\n",
                   result.feasible ? "feasible" : "INFEASIBLE",
                   (long)ucm.model.num_vars(), (long)ucm.model.num_nodes(),
                   (long)result.iterations);
        }
        printf("\n");
    }

    return 0;
}
