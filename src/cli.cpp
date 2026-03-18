#include "cbls/cbls.h"
#include "cbls/io.h"
#include "cbls/formatter.h"
#include <iostream>
#include <string>

using namespace cbls;

static void print_help() {
    std::cout << R"(Usage: cbls [OPTIONS] MODEL

  Constraint-Based Local Search solver. Reads a JSONL model file (.cbls)
  and finds optimal or feasible variable assignments.

Arguments:
  MODEL                 Path to JSONL model file (.cbls)

Options:
  --time-limit SECS     Maximum solve time in seconds (default: 10.0)
  --seed INT            Random seed for reproducibility (default: 42)
  --no-fj               Disable feasibility jump initialization
  --lns FRACTION        Enable LNS with destroy fraction, e.g. 0.3
  --lns-interval INT    LNS fires every N reheats (default: 3)
  --intensify           Enable float intensification hook
  --cooling-rate FLOAT  SA cooling rate (default: 0.9999)
  --reheat-interval INT SA reheat interval in iterations (default: 5000)
  --hook-frequency INT  Run hook every N discrete acceptances (default: 10)
  --fj-time-fraction F  Fraction of time limit for FJ init (default: 0.2)
  --format human|jsonl  Output format (default: human)
  --quiet               Suppress progress, print only final result
  --help                Show this help message
  --version             Show version number
)";
}

int main(int argc, char* argv[]) {
    std::string model_path;
    double time_limit = 10.0;
    uint64_t seed = 42;
    bool use_fj = true;
    bool use_intensify = false;
    double lns_fraction = 0.0;
    int lns_interval = 3;
    SearchConfig config;
    std::string format = "human";
    bool quiet = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        } else if (arg == "--version") {
            std::cout << "cbls " << cbls::version << "\n";
            return 0;
        } else if (arg == "--time-limit" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoull(argv[++i]);
        } else if (arg == "--no-fj") {
            use_fj = false;
        } else if (arg == "--lns" && i + 1 < argc) {
            lns_fraction = std::stod(argv[++i]);
        } else if (arg == "--lns-interval" && i + 1 < argc) {
            lns_interval = std::stoi(argv[++i]);
        } else if (arg == "--intensify") {
            use_intensify = true;
        } else if (arg == "--cooling-rate" && i + 1 < argc) {
            config.cooling_rate = std::stod(argv[++i]);
        } else if (arg == "--reheat-interval" && i + 1 < argc) {
            config.reheat_interval = std::stoi(argv[++i]);
        } else if (arg == "--hook-frequency" && i + 1 < argc) {
            config.hook_frequency = std::stoi(argv[++i]);
        } else if (arg == "--fj-time-fraction" && i + 1 < argc) {
            config.fj_time_fraction = std::stod(argv[++i]);
        } else if (arg == "--format" && i + 1 < argc) {
            format = argv[++i];
            if (format != "human" && format != "jsonl") {
                std::cerr << "Error: --format must be 'human' or 'jsonl'\n";
                return 1;
            }
        } else if (arg == "--quiet") {
            quiet = true;
        } else if (arg[0] == '-') {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            return 1;
        } else {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: no model file specified. Use --help for usage.\n";
        return 1;
    }

    Model model;
    try {
        model = load_model(model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Set up optional components
    FloatIntensifyHook intensify_hook;
    InnerSolverHook* hook = use_intensify ? &intensify_hook : nullptr;

    LNS lns_obj(lns_fraction);
    LNS* lns_ptr = lns_fraction > 0.0 ? &lns_obj : nullptr;

    // Set up formatter
    HumanFormatter human_fmt(std::cout);
    JsonlFormatter jsonl_fmt(std::cout);

    SolveCallback* callback = nullptr;
    if (!quiet) {
        if (format == "human") {
            human_fmt.print_header(model_path, model, seed, time_limit);
            callback = &human_fmt;
        } else {
            jsonl_fmt.print_header(model_path, model, seed, time_limit);
            callback = &jsonl_fmt;
        }
    }

    auto result = solve(model, time_limit, seed, use_fj,
                        hook, lns_ptr, lns_interval, callback, config);

    if (format == "human") {
        human_fmt.print_result(result, model);
    } else {
        jsonl_fmt.print_result(result, model);
    }

    return result.feasible ? 0 : 1;
}
