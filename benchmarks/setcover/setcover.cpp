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
//   cbls_setcover --encoding set --structural-selection violation_guided
//                 --structural-sample-size 8
//
// The last one's A/B ran through ab_selection.sh (git history, 047f2d4c).

#include "data.h"
#include "setcover_model.h"
#include "verify_setcover.h"

#include <cbls/arg_parse.h>
#include <cbls/cbls.h>
#include <cbls/search.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace {

// Proven optima, mirroring benchmarks/instances/setcover/download.py (which
// carries the provenance). Absent name => the result is reported without a gap.
const std::map<std::string, double>& published_optima() {
    static const std::map<std::string, double> kOptima = {
        {"scp41", 429}, {"scp42", 512}, {"scp43", 516}, {"scp44", 494}, {"scp45", 512},
        {"scpe1", 5},   {"scpe2", 5},   {"scpe3", 5},   {"scpe4", 5},   {"scpe5", 5},
    };
    return kOptima;
}

const std::vector<std::string>& default_roster() {
    static const std::vector<std::string> kRoster = {"scp41", "scp42", "scp43", "scp44", "scp45",
                                                     "scpe1", "scpe2", "scpe3", "scpe4", "scpe5"};
    return kRoster;
}

struct Options {
    bool help = false;
    std::string dir = "benchmarks/instances/setcover";
    std::string instance;  // explicit path; empty => the vendored roster
    double time_limit = 10.0;
    int seeds = 1;
    uint64_t first_seed = 42;
    double struct_prob = -1.0;  // the default, engine auto; a typed value is in [0, 1]
    // Structural-selection arm (#165). Default is the engine's, so an
    // unflagged run is the configuration the README reports.
    cbls::StructuralSelection selection = cbls::StructuralSelection::FirstImprovingSample;
    // 0 = leave SearchConfig's own default alone. Inert under the default
    // selection, which scores exactly one sample.
    int sample_size = 0;
    bool run_set = true;
    bool run_bool = true;
    std::string csv_path;
};

struct Run {
    std::string instance;
    cbls::setcover::Encoding encoding;
    // Read back off the SearchConfig the solve ran with, not the parsed flag,
    // so a row records what the engine was given.
    const char* selection = "";
    uint64_t seed;
    double objective;
    bool feasible;
    bool verified;
    int columns;
    double seconds;
    int64_t iterations;
};

void print_usage() {
    printf(
        "usage: cbls_setcover [--dir D] [--instance F] [--time S] [--seeds N]\n"
        "                     [--seed S0] [--encoding set|bool|both] [--structural-prob P]\n"
        "                     [--structural-selection "
        "first_improving|best_of_sample|violation_guided]\n"
        "                     [--structural-sample-size K] [--csv OUT]\n");
}

// Not runner_args.h, by policy: this runner has no `!(x > 0.0)` guard to catch
// a NaN, so it rejects a malformed value where it reads it and run_benchmark
// returns 1. Unsigned integers, NaN rejected, a failure flag instead of an
// exit: too different to share one helper. The strict parse names the flag
// (strtoull would turn `--seed abc` into 0 silently); NaN is a well-formed
// double that yields a solve that never searched.
//
// Not search_config_flags.h either (#198): the structural flags take its
// spellings and value rules, but its applicability mask is per runner, and one
// run here spans both encodings, the Bool one scalar. They apply to Set rows
// and are recorded on every row.
struct FlagValues {
    int argc = 0;
    char** argv = nullptr;
    int i = 0;       // index of the flag being read; advanced past its value
    bool ok = true;  // cleared by the first malformed or missing value

    std::string next(const char* what) {
        if (i + 1 >= argc) {
            fprintf(stderr, "%s needs a value (%s)\n", argv[i], what);
            ok = false;
            return "";
        }
        return argv[++i];
    }

    double number(const char* flag, const char* what) {
        const std::string text = next(what);
        double value = 0.0;
        if (!ok) {
            return 0.0;
        }
        if (!cbls::try_parse_double(text, value) || std::isnan(value)) {
            fprintf(stderr, "%s: '%s' is not a number\n", flag, text.c_str());
            ok = false;
            return 0.0;
        }
        return value;
    }

    uint64_t integer(const char* flag, const char* what) {
        const std::string text = next(what);
        uint64_t value = 0;
        if (!ok) {
            return 0;
        }
        if (!cbls::try_parse_uint64(text, value)) {
            fprintf(stderr, "%s: '%s' is not a non-negative integer\n", flag, text.c_str());
            ok = false;
            return 0;
        }
        return value;
    }
};

/// --seeds is an `int` loop bound, so an in-range uint64 can still overflow it.
int seed_count(FlagValues& v) {
    const uint64_t seeds = v.integer("--seeds", "count");
    if (v.ok && seeds > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "--seeds: '%llu' is out of range\n",
                static_cast<unsigned long long>(seeds));
        v.ok = false;
    }
    return static_cast<int>(seeds);
}

/// The structural-selection arm's two flags (#165). Returns false when `arg`
/// is not one of them.
bool parse_selection_flag(const std::string& arg, FlagValues& v, Options& opt) {
    if (arg == "--structural-selection") {
        const std::string name = v.next("first_improving|best_of_sample|violation_guided");
        if (v.ok && !cbls::try_parse_structural_selection(name, opt.selection)) {
            fprintf(stderr, "unknown selection '%s'\n", name.c_str());
            v.ok = false;
        }
        // `auto` is first_improving without a neighbour list; record it as the
        // policy that runs, so one run has one name.
        opt.selection = cbls::resolve_structural_selection(opt.selection, false);
        // No neighbour list here, so granular_best would be violation_guided
        // under a second name (#203).
        if (v.ok && opt.selection == cbls::StructuralSelection::GranularBest) {
            fprintf(stderr, "granular_best needs a neighbour list, which setcover has none of\n");
            v.ok = false;
        }
        return true;
    }
    if (arg == "--structural-sample-size") {
        const uint64_t k = v.integer("--structural-sample-size", "count");
        if (v.ok && (k == 0 || k > static_cast<uint64_t>(std::numeric_limits<int>::max()))) {
            fprintf(stderr, "--structural-sample-size must be a positive int\n");
            v.ok = false;
        }
        // Validate, then narrow: narrowing an out-of-range uint64_t is
        // implementation-defined (as seed_count).
        if (v.ok) {
            opt.sample_size = static_cast<int>(k);
        }
        return true;
    }
    return false;
}

/// `--encoding`, likewise: a value flag with its own validation.
void parse_encoding_flag(FlagValues& v, Options& opt) {
    const std::string enc = v.next("set|bool|both");
    opt.run_set = (enc == "set" || enc == "both");
    opt.run_bool = (enc == "bool" || enc == "both");
    if (v.ok && !opt.run_set && !opt.run_bool) {
        fprintf(stderr, "unknown encoding '%s'\n", enc.c_str());
        v.ok = false;
    }
}

/// The structural flags' cross-flag rules, from search_config_flags.h: a typed
/// flag read by nothing is refused. After the loop: flags come in any order.
bool structural_arm_is_read(const Options& opt) {
    // first_improving scores one sample, so a non-default size is read by nothing.
    if (opt.selection == cbls::StructuralSelection::FirstImprovingSample && opt.sample_size > 0 &&
        opt.sample_size != cbls::SearchConfig{}.structural_sample_size) {
        fprintf(stderr,
                "--structural-sample-size has no effect under --structural-selection "
                "first_improving\n");
        return false;
    }
    // At probability 0 no structural batch runs: policy and size are unread.
    if (opt.struct_prob == 0.0 &&
        (opt.selection != cbls::StructuralSelection::FirstImprovingSample || opt.sample_size > 0)) {
        fprintf(stderr,
                "--structural-prob 0 runs no structural batch, so --structural-selection and "
                "--structural-sample-size are read by nothing\n");
        return false;
    }
    return true;
}

Options parse_args(int argc, char** argv, bool* ok) {
    Options opt;
    FlagValues v{argc, argv, 0, true};
    for (v.i = 1; v.i < argc; ++v.i) {
        std::string arg = argv[v.i];
        if (arg == "--dir") {
            opt.dir = v.next("directory");
        } else if (arg == "--instance") {
            opt.instance = v.next("path");
        } else if (arg == "--time") {
            opt.time_limit = v.number("--time", "seconds");
        } else if (arg == "--seeds") {
            opt.seeds = seed_count(v);
        } else if (arg == "--seed") {
            opt.first_seed = v.integer("--seed", "seed");
        } else if (arg == "--structural-prob") {
            // `+ 0.0` folds -0 onto 0, so one arm has one recorded value.
            opt.struct_prob = v.number("--structural-prob", "probability") + 0.0;
            // "Automatic" is the default, spelled by omitting the flag.
            if (v.ok && (opt.struct_prob < 0.0 || opt.struct_prob > 1.0)) {
                fprintf(stderr, "--structural-prob must be in [0, 1]\n");
                v.ok = false;
            }
        } else if (parse_selection_flag(arg, v, opt)) {
            // handled, including its own validation
        } else if (arg == "--csv") {
            opt.csv_path = v.next("path");
        } else if (arg == "--encoding") {
            parse_encoding_flag(v, opt);
        } else if (arg == "--help" || arg == "-h") {
            opt.help = true;
            print_usage();
            break;
        } else {
            fprintf(stderr, "unknown argument '%s'\n", arg.c_str());
            v.ok = false;
        }
        if (!v.ok) {
            break;
        }
    }
    if (v.ok && !structural_arm_is_read(opt)) {
        v.ok = false;
    }
    *ok = v.ok;
    return opt;
}

Run solve_one(const cbls::setcover::SetCoverInstance& inst, cbls::setcover::Encoding encoding,
              uint64_t seed, const Options& opt) {
    cbls::setcover::SetCoverModel scm = cbls::setcover::build_model(inst, encoding);

    cbls::SearchConfig config;
    config.structural_batch_probability = opt.struct_prob;
    config.structural_selection = opt.selection;
    if (opt.sample_size > 0) {
        config.structural_sample_size = opt.sample_size;
    }
    cbls::SearchResult result =
        cbls::solve(scm.model, opt.time_limit, seed, /*hook=*/nullptr,
                    /*lns=*/nullptr, /*lns_interval=*/0, /*callback=*/nullptr, config);

    const cbls::VerifyResult verified = cbls::setcover::verify_setcover(scm, inst);
    const cbls::setcover::CoverCheck check =
        cbls::setcover::check_cover(inst, scm.selected_columns());

    Run run;
    run.instance = inst.name;
    run.encoding = encoding;
    run.selection = cbls::structural_selection_name(config.structural_selection);
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
                cbls::setcover::encoding_name(encoding), static_cast<unsigned long long>(seed));
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
    if (out == nullptr) {
        fprintf(stderr, "cannot write %s\n", path.c_str());
        return;
    }
    fprintf(out,
            "instance,encoding,selection,seed,objective,optimum,gap_percent,feasible,verified,"
            "columns,seconds,iterations\n");
    for (const Run& r : runs) {
        auto it = published_optima().find(r.instance);
        double optimum = (it == published_optima().end()) ? -1.0 : it->second;
        fprintf(out, "%s,%s,%s,%llu,%.1f,%.1f,%.2f,%d,%d,%d,%.2f,%lld\n", r.instance.c_str(),
                cbls::setcover::encoding_name(r.encoding), r.selection,
                static_cast<unsigned long long>(r.seed), r.objective, optimum,
                gap_percent(r.instance, r.objective), r.feasible ? 1 : 0, r.verified ? 1 : 0,
                r.columns, r.seconds, static_cast<long long>(r.iterations));
    }
    std::fclose(out);
    printf("\nwrote %s\n", path.c_str());
}

/// The instances to run: an explicit --instance path, else the vendored roster
/// resolved against --dir.
std::vector<std::string> instance_paths(const Options& opt) {
    if (!opt.instance.empty()) {
        return {opt.instance};
    }
    std::vector<std::string> paths;
    for (const std::string& name : default_roster()) {
        paths.push_back(opt.dir + "/" + name + ".txt");
    }
    return paths;
}

/// The encodings --encoding selected, in the order the table reports them.
std::vector<cbls::setcover::Encoding> selected_encodings(const Options& opt) {
    std::vector<cbls::setcover::Encoding> encodings;
    if (opt.run_set) {
        encodings.push_back(cbls::setcover::Encoding::Set);
    }
    if (opt.run_bool) {
        encodings.push_back(cbls::setcover::Encoding::Bool);
    }
    return encodings;
}

void print_table_header() {
    printf("%-10s %-5s %5s %10s %9s %8s %5s %8s %8s\n", "Instance", "Enc", "Seed", "Objective",
           "Optimum", "Gap%", "Cols", "Time(s)", "Feasible");
    printf("%-10s %-5s %5s %10s %9s %8s %5s %8s %8s\n", "--------", "---", "----", "---------",
           "-------", "----", "----", "-------", "--------");
}

/// One console row. The last column is the verifier's verdict: "NO" when the
/// search is infeasible, UNVERIFIED when feasible but failing verification.
void print_run_row(const Run& run) {
    auto it = published_optima().find(run.instance);
    double optimum = (it == published_optima().end()) ? -1.0 : it->second;
    const char* verified_label = "NO";
    if (run.feasible) {
        verified_label = run.verified ? "yes" : "UNVERIFIED";
    }
    printf("%-10s %-5s %5llu %10.1f %9.1f %8.1f %5d %8.2f %8s\n", run.instance.c_str(),
           cbls::setcover::encoding_name(run.encoding), static_cast<unsigned long long>(run.seed),
           run.objective, optimum, gap_percent(run.instance, run.objective), run.columns,
           run.seconds, verified_label);
    std::fflush(stdout);
}

int run_benchmark(int argc, char** argv) {
    bool ok = false;
    Options opt = parse_args(argc, argv, &ok);
    if (opt.help) {
        return 0;
    }
    if (!ok) {
        return 1;
    }

    const std::vector<std::string> paths = instance_paths(opt);
    const std::vector<cbls::setcover::Encoding> encodings = selected_encodings(opt);
    print_table_header();

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
                print_run_row(run);
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

}  // namespace

int main(int argc, char** argv) {
    // An escaping exception is a silent std::terminate; report it and exit
    // non-zero instead (bugprone-exception-escape).
    try {
        return run_benchmark(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
