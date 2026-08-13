// Set-covering runner: the Set-variable coverage check of issue #93.
//
// Runs the vendored OR-Library roster under both encodings (see
// setcover_model.h) and scores each result against the instance's published
// optimum, verifying feasibility against the instance file rather than the DAG.
//
//   cbls_setcover                                   # whole roster, both encodings, 10s each
//   cbls_setcover --encoding set --time 30
//   cbls_setcover --instance benchmarks/instances/setcover/scpe1.txt --seeds 5
//   cbls_setcover --csv results.csv

#include "data.h"
#include "setcover_model.h"
#include "verify_setcover.h"

#include <cbls/cbls.h>
#include <cbls/search.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace {

// Proven optima, mirroring benchmarks/instances/setcover/download.py (which
// carries the provenance). Absent name => the result is reported without a gap.
const std::map<std::string, double>& published_optima() {
    static const std::map<std::string, double> optima = {
        {"scp41", 429}, {"scp42", 512}, {"scp43", 516}, {"scp44", 494}, {"scp45", 512},
        {"scpe1", 5},   {"scpe2", 5},   {"scpe3", 5},   {"scpe4", 5},   {"scpe5", 5},
    };
    return optima;
}

const std::vector<std::string>& default_roster() {
    static const std::vector<std::string> roster = {"scp41", "scp42", "scp43", "scp44", "scp45",
                                                    "scpe1", "scpe2", "scpe3", "scpe4", "scpe5"};
    return roster;
}

struct Options {
    std::string dir = "benchmarks/instances/setcover";
    std::string instance;  // explicit path; empty => the vendored roster
    double time_limit = 10.0;
    int seeds = 1;
    uint64_t first_seed = 42;
    double struct_prob = -1.0;  // <0 = engine auto (0.33 on a Set model)
    bool run_set = true;
    bool run_bool = true;
    std::string csv_path;
};

struct Run {
    std::string instance;
    cbls::setcover::Encoding encoding;
    uint64_t seed;
    double objective;
    bool feasible;
    bool verified;
    int columns;
    double seconds;
    int64_t iterations;
};

void print_usage() {
    printf("usage: cbls_setcover [--dir D] [--instance F] [--time S] [--seeds N]\n"
           "                     [--seed S0] [--encoding set|bool|both] [--struct-prob P]\n"
           "                     [--csv OUT]\n");
}

Options parse_args(int argc, char** argv, bool* ok) {
    Options opt;
    *ok = true;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "%s needs a value (%s)\n", arg.c_str(), what);
                *ok = false;
                return "";
            }
            return argv[++i];
        };
        if (arg == "--dir") {
            opt.dir = next("directory");
        } else if (arg == "--instance") {
            opt.instance = next("path");
        } else if (arg == "--time") {
            opt.time_limit = std::stod(next("seconds"));
        } else if (arg == "--seeds") {
            opt.seeds = std::stoi(next("count"));
        } else if (arg == "--seed") {
            opt.first_seed = std::strtoull(next("seed").c_str(), nullptr, 10);
        } else if (arg == "--struct-prob") {
            opt.struct_prob = std::stod(next("probability"));
        } else if (arg == "--csv") {
            opt.csv_path = next("path");
        } else if (arg == "--encoding") {
            std::string enc = next("set|bool|both");
            opt.run_set = (enc == "set" || enc == "both");
            opt.run_bool = (enc == "bool" || enc == "both");
            if (!opt.run_set && !opt.run_bool) {
                fprintf(stderr, "unknown encoding '%s'\n", enc.c_str());
                *ok = false;
            }
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            *ok = false;
        } else {
            fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
            *ok = false;
        }
        if (!*ok) {
            break;
        }
    }
    return opt;
}

Run solve_one(const cbls::setcover::SetCoverInstance& inst, cbls::setcover::Encoding encoding,
              uint64_t seed, const Options& opt) {
    cbls::setcover::SetCoverModel scm = cbls::setcover::build_model(inst, encoding);

    cbls::SearchConfig config;
    config.structural_batch_probability = opt.struct_prob;
    cbls::SearchResult result =
        cbls::solve(scm.model, opt.time_limit, seed, /*use_fj=*/true, /*hook=*/nullptr,
                    /*lns=*/nullptr, /*lns_interval=*/0, /*callback=*/nullptr, config);

    const cbls::VerifyResult verified = cbls::setcover::verify_setcover(scm, inst);
    const cbls::setcover::CoverCheck check =
        cbls::setcover::check_cover(inst, scm.selected_columns());

    Run run;
    run.instance = inst.name;
    run.encoding = encoding;
    run.seed = seed;
    // Report the recomputed cost, not the DAG objective: the two agreeing is
    // itself one of the verification checks.
    run.objective = check.cost;
    run.feasible = result.feasible && check.covered;
    run.verified = verified.ok;
    run.columns = check.num_columns;
    run.seconds = result.time_seconds;
    run.iterations = result.iterations;
    if (!verified.ok) {
        fprintf(stderr, "%s/%s seed %llu: VERIFY FAILED\n", inst.name.c_str(),
                cbls::setcover::encoding_name(encoding),
                static_cast<unsigned long long>(seed));
        verified.print_diagnostics();
    }
    return run;
}

double gap_percent(const std::string& instance, double objective) {
    auto it = published_optima().find(instance);
    if (it == published_optima().end() || it->second <= 0.0) {
        return -1.0;
    }
    return 100.0 * (objective - it->second) / it->second;
}

void write_csv(const std::string& path, const std::vector<Run>& runs) {
    FILE* out = std::fopen(path.c_str(), "w");
    if (!out) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    fprintf(out, "instance,encoding,seed,objective,optimum,gap_percent,feasible,verified,"
                 "columns,seconds,iterations\n");
    for (const Run& r : runs) {
        auto it = published_optima().find(r.instance);
        double optimum = (it == published_optima().end()) ? -1.0 : it->second;
        fprintf(out, "%s,%s,%llu,%.1f,%.1f,%.2f,%d,%d,%d,%.2f,%lld\n", r.instance.c_str(),
                cbls::setcover::encoding_name(r.encoding),
                static_cast<unsigned long long>(r.seed), r.objective, optimum,
                gap_percent(r.instance, r.objective), r.feasible ? 1 : 0, r.verified ? 1 : 0,
                r.columns, r.seconds, static_cast<long long>(r.iterations));
    }
    std::fclose(out);
    printf("\nwrote %s\n", path.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    bool ok = false;
    Options opt = parse_args(argc, argv, &ok);
    if (!ok) {
        return 1;
    }

    std::vector<std::string> paths;
    if (!opt.instance.empty()) {
        paths.push_back(opt.instance);
    } else {
        for (const std::string& name : default_roster()) {
            paths.push_back(opt.dir + "/" + name + ".txt");
        }
    }

    std::vector<cbls::setcover::Encoding> encodings;
    if (opt.run_set) {
        encodings.push_back(cbls::setcover::Encoding::Set);
    }
    if (opt.run_bool) {
        encodings.push_back(cbls::setcover::Encoding::Bool);
    }

    printf("%-10s %-5s %5s %10s %9s %8s %5s %8s %8s\n", "Instance", "Enc", "Seed", "Objective",
           "Optimum", "Gap%", "Cols", "Time(s)", "Feasible");
    printf("%-10s %-5s %5s %10s %9s %8s %5s %8s %8s\n", "--------", "---", "----", "---------",
           "-------", "----", "----", "-------", "--------");

    std::vector<Run> runs;
    int failures = 0;
    for (const std::string& path : paths) {
        cbls::setcover::SetCoverInstance inst;
        try {
            inst = cbls::setcover::load_setcover(path);
        } catch (const std::exception& e) {
            fprintf(stderr, "%s\n", e.what());
            fprintf(stderr, "run `python benchmarks/instances/setcover/download.py` first\n");
            return 1;
        }
        for (cbls::setcover::Encoding encoding : encodings) {
            for (int s = 0; s < opt.seeds; ++s) {
                const uint64_t seed = opt.first_seed + static_cast<uint64_t>(s);
                Run run = solve_one(inst, encoding, seed, opt);
                runs.push_back(run);
                if (!run.feasible || !run.verified) {
                    ++failures;
                }
                auto it = published_optima().find(run.instance);
                double optimum = (it == published_optima().end()) ? -1.0 : it->second;
                printf("%-10s %-5s %5llu %10.1f %9.1f %8.1f %5d %8.2f %8s\n", run.instance.c_str(),
                       cbls::setcover::encoding_name(run.encoding),
                       static_cast<unsigned long long>(run.seed), run.objective, optimum,
                       gap_percent(run.instance, run.objective), run.columns, run.seconds,
                       run.feasible ? (run.verified ? "yes" : "UNVERIFIED") : "NO");
                std::fflush(stdout);
            }
        }
    }

    if (!opt.csv_path.empty()) {
        write_csv(opt.csv_path, runs);
    }
    if (failures > 0) {
        fprintf(stderr, "\n%d run(s) infeasible or unverified\n", failures);
        return 1;
    }
    return 0;
}
