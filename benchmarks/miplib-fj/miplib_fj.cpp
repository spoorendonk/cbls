// MIPLIB-FJ benchmark runner.
//
// Loads each `.mps.gz` from `benchmarks/instances/miplib-fj/`, builds a
// CBLS model via the MPS-to-Model adapter, runs a fixed-time SA pass, and
// writes a per-instance row to `benchmarks/instances/miplib-fj/comparison.csv`.

#include <cbls/cbls.h>
#include <cbls/io_mps.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string inst_dir = "benchmarks/instances/miplib-fj";
    double time_limit = 30.0;
    std::vector<std::string> instances;  // optional override
    std::string commit_sha = "unknown";
    std::string out_csv;  // default: <inst_dir>/comparison.csv
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--time-limit" && i + 1 < argc) {
            a.time_limit = std::atof(argv[++i]);
        } else if (s == "--instance" && i + 1 < argc) {
            a.instances.emplace_back(argv[++i]);
        } else if (s == "--commit" && i + 1 < argc) {
            a.commit_sha = argv[++i];
        } else if (s == "--out" && i + 1 < argc) {
            a.out_csv = argv[++i];
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_miplib_fj [inst-dir] [--time-limit S]"
                " [--instance NAME ...] [--commit SHA] [--out CSV]\n");
            std::exit(0);
        } else {
            a.inst_dir = s;
        }
    }
    if (a.out_csv.empty()) {
        a.out_csv = a.inst_dir + "/comparison.csv";
    }
    return a;
}

// Default instance roster, mirrored from
// benchmarks/instances/miplib-fj/download.py.
std::vector<std::string> default_instances() {
    return {
        "enlight_hard", "markshare1", "markshare2", "gen-ip054", "gen-ip002",  "pk1",
        "mas76",        "neos5",      "flugpl",     "mad",       "binkar10_1",
    };
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

struct OptInfo {
    bool have = false;
    double value = 0.0;
    bool is_optimal = false;
    bool is_infeasible = false;
};

OptInfo lookup_solu(const std::vector<cbls::SoluEntry>& solu, const std::string& name) {
    OptInfo info;
    for (const auto& e : solu) {
        if (e.name == name) {
            info.have = true;
            info.value = e.value;
            info.is_optimal = e.is_optimal;
            info.is_infeasible = e.is_infeasible;
            return info;
        }
    }
    return info;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    std::vector<std::string> insts = args.instances.empty() ? default_instances() : args.instances;

    // Try to load .solu (optional — gap column is NaN if missing).
    std::vector<cbls::SoluEntry> solu;
    {
        std::string solu_path = args.inst_dir + "/miplib2017-v22.solu";
        if (file_exists(solu_path)) {
            try {
                solu = cbls::read_solu(solu_path);
                std::printf("Loaded %zu .solu entries from %s\n", solu.size(), solu_path.c_str());
            } catch (const std::exception& e) {
                std::printf("WARNING: failed to load %s: %s\n", solu_path.c_str(), e.what());
            }
        } else {
            std::printf("WARNING: %s missing; gap-to-opt will be NaN.\n", solu_path.c_str());
        }
    }

    std::ofstream csv(args.out_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", args.out_csv.c_str());
        return 2;
    }
    csv << "instance,objective,gap_to_opt%,wall_seconds,commit_sha\n";

    std::printf("\n%-24s %12s %12s %10s %10s\n", "Instance", "Objective", "Opt", "Gap%", "Time(s)");
    std::printf("%-24s %12s %12s %10s %10s\n", "--------", "---------", "---", "----", "-------");

    for (const std::string& name : insts) {
        std::string mps_path = args.inst_dir + "/" + name + ".mps.gz";
        if (!file_exists(mps_path)) {
            // Try plain .mps too.
            std::string alt = args.inst_dir + "/" + name + ".mps";
            if (file_exists(alt)) {
                mps_path = alt;
            } else {
                std::printf("%-24s  (skipped: %s not found)\n", name.c_str(), mps_path.c_str());
                continue;
            }
        }

        // Read MPS.
        cbls::MpsProblem prob;
        try {
            prob = cbls::read_mps(mps_path);
        } catch (const std::exception& e) {
            std::printf("%-24s  ERROR reading %s: %s\n", name.c_str(), mps_path.c_str(), e.what());
            continue;
        }
        if (prob.vars.empty() || prob.rows.empty()) {
            std::printf(
                "%-24s  (skipped: parsed 0 vars/rows from %s — "
                "probably a corrupted download)\n",
                name.c_str(), mps_path.c_str());
            continue;
        }

        // Build CBLS model.
        cbls::MpsToModelResult built;
        try {
            built = cbls::mps_to_model(prob);
        } catch (const std::exception& e) {
            std::printf("%-24s  ERROR building model: %s\n", name.c_str(), e.what());
            continue;
        }

        std::printf("%-24s ", name.c_str());
        std::fflush(stdout);

        // Run SA. Use FJ phase 1 to get an initial feasible point.
        auto t0 = std::chrono::steady_clock::now();
        cbls::FloatIntensifyHook hook;
        cbls::LNS lns(0.3);
        auto result = cbls::solve(built.model, args.time_limit, /*seed=*/42,
                                  /*use_fj=*/true, &hook, &lns);
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double>(t1 - t0).count();

        OptInfo opt = lookup_solu(solu, name);

        double obj = result.feasible ? result.objective : std::numeric_limits<double>::quiet_NaN();
        double gap = std::numeric_limits<double>::quiet_NaN();
        if (result.feasible && opt.have && !opt.is_infeasible && std::abs(opt.value) > 1e-12) {
            gap = 100.0 * (obj - opt.value) / std::abs(opt.value);
        } else if (result.feasible && opt.have && !opt.is_infeasible) {
            // optimum is exactly 0 — report absolute residual instead.
            gap = obj - opt.value;
        }

        // Console.
        if (result.feasible) {
            std::printf("%12.4g ", obj);
        } else {
            std::printf("%12s ", "INFEAS");
        }
        if (opt.have && !opt.is_infeasible) {
            std::printf("%12.4g ", opt.value);
        } else {
            std::printf("%12s ", "?");
        }
        if (std::isnan(gap)) {
            std::printf("%10s ", "N/A");
        } else {
            std::printf("%9.2f%% ", gap);
        }
        std::printf("%9.2fs\n", wall);

        // CSV row.
        csv << name << ",";
        if (result.feasible) {
            csv << obj;
        } else {
            csv << "NaN";
        }
        csv << ",";
        if (std::isnan(gap)) {
            csv << "NaN";
        } else {
            csv << gap;
        }
        csv << "," << wall << "," << args.commit_sha << "\n";
        csv.flush();
    }

    std::printf("\nWrote %s\n", args.out_csv.c_str());
    return 0;
}
