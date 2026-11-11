#include "cbls/arg_parse.h"
#include "cbls/cbls.h"
#include "cbls/formatter.h"
#include "cbls/io.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <variant>

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
  --lns-interval INT    LNS fires every N diversification kicks (default: 3)
  --intensify           Enable float intensification hook
  --threads N           Number of search threads (default: 0 = one per core)
  --format human|jsonl  Output format (default: human)
  --quiet               Suppress progress, print only final result
  --help                Show this help message
  --version             Show version number
)";
}

namespace {

// A malformed numeric flag is reported at the parse and exits 1, the loop's
// `Error: ...` convention. (The benchmark runners instead return NaN for a
// later guard to exit 2, an exit code their tests pin; the CLI has no such
// guard.)
//
// Not rejected: `--time-limit 0` (no budget: solve() returns `no_budget`,
// exit 1) and `--time-limit inf` (saturates at make_budget's ~31-year
// ceiling); both are honest readings. NaN alone is rejected: nobody means it,
// and it silently disables --lns and the search.
//
// Every overload returns false having written the diagnostic; the caller only
// `return 1`s. The parsing rule is cbls/arg_parse.h, shared with
// benchmarks/common/runner_args.h.

// Overflow is reported as such, not as a typo the reader would hunt for.
void report_parse_failure(const char* flag, const char* text, cbls::ParseStatus status,
                          const char* kind) {
    if (status == cbls::ParseStatus::kOutOfRange) {
        std::cerr << "Error: " << flag << ": '" << text << "' is out of range\n";
    } else {
        std::cerr << "Error: " << flag << ": '" << text << "' is not " << kind << "\n";
    }
}

bool parse_flag(const char* flag, const char* text, double& out) {
    // std::stod accepts "nan"; reported as malformed, not out of range.
    double value = 0.0;
    const cbls::ParseStatus status = cbls::parse_double_status(text, value);
    if (status == cbls::ParseStatus::kOk && !std::isnan(value)) {
        out = value;
        return true;
    }
    report_parse_failure(flag, text, status, "a number");
    return false;
}

bool parse_flag(const char* flag, const char* text, int64_t& out) {
    const cbls::ParseStatus status = cbls::parse_int64_status(text, out);
    if (status == cbls::ParseStatus::kOk) {
        return true;
    }
    report_parse_failure(flag, text, status, "an integer");
    return false;
}

bool parse_flag(const char* flag, const char* text, int& out) {
    int64_t wide = 0;
    if (!parse_flag(flag, text, wide)) {
        return false;
    }
    if (wide < std::numeric_limits<int>::min() || wide > std::numeric_limits<int>::max()) {
        std::cerr << "Error: " << flag << ": '" << text << "' is out of range\n";
        return false;
    }
    out = static_cast<int>(wide);
    return true;
}

// --seed spans the full uint64 range: the header prints `--seed -1` as
// 18446744073709551615, and the CLI must be able to read back what it printed.
bool parse_flag(const char* flag, const char* text, uint64_t& out) {
    const cbls::ParseStatus status = cbls::parse_uint64_status(text, out);
    if (status == cbls::ParseStatus::kOk) {
        return true;
    }
    report_parse_failure(flag, text, status, "an integer");
    return false;
}

// Everything the option loop can set, with the CLI's defaults.
struct CliOptions {
    std::string model_path;
    double time_limit = 10.0;
    uint64_t seed = 42;
    bool use_fj = true;
    bool use_intensify = false;
    double lns_fraction = 0.0;
    SearchConfig config;  // also holds lns_interval, which --lns-interval writes
    std::string format = "human";
    bool quiet = false;
    // 0 = one worker per hardware thread. `--threads 1` gives a single-threaded,
    // exactly reproducible run (docs/profiling.md).
    int n_threads = 0;
};

// Whether an option consumed the argument, and if so whether it was well formed.
// kNotMine means "not this kind of option", so the loop keeps looking.
enum class FlagStatus : std::uint8_t { kNotMine, kOk, kError };

// The valueless switches. Returns true if `arg` was one of them.
bool take_toggle_option(const std::string& arg, CliOptions& opt) {
    if (arg == "--no-fj") {
        opt.use_fj = false;
        opt.config.use_fj = false;
        return true;
    }
    if (arg == "--intensify") {
        opt.use_intensify = true;
        return true;
    }
    if (arg == "--quiet") {
        opt.quiet = true;
        return true;
    }
    return false;
}

// The numeric options, each paired with the field it fills; the parse_flag
// overload varies with the type, the plumbing does not.
struct ValueOption {
    const char* flag;
    std::variant<double*, int*, int64_t*, uint64_t*> target;
};

std::array<ValueOption, 5> numeric_options(CliOptions& opt) {
    return {{{"--time-limit", &opt.time_limit},
             {"--seed", &opt.seed},
             {"--lns", &opt.lns_fraction},
             {"--lns-interval", &opt.config.lns_interval},
             {"--threads", &opt.n_threads}}};
}

// One `--flag VALUE` option. Advances `i` past the value when it takes one.
//
// A flag with nothing after it reports kNotMine, so the caller's unknown-option
// branch names it.
FlagStatus take_value_option(const std::string& arg, int& i, int argc, char** argv,
                             CliOptions& opt) {
    if (i + 1 >= argc) {
        return FlagStatus::kNotMine;
    }
    // --format is the one value-taking option whose value the parser must also
    // validate, so it is not in the table.
    if (arg == "--format") {
        opt.format = argv[++i];
        if (opt.format != "human" && opt.format != "jsonl") {
            std::cerr << "Error: --format must be 'human' or 'jsonl'\n";
            return FlagStatus::kError;
        }
        return FlagStatus::kOk;
    }
    for (const auto& option : numeric_options(opt)) {
        if (arg != option.flag) {
            continue;
        }
        const char* text = argv[++i];
        const bool ok = std::visit(
            [&](auto* field) { return parse_flag(option.flag, text, *field); }, option.target);
        return ok ? FlagStatus::kOk : FlagStatus::kError;
    }
    return FlagStatus::kNotMine;
}

// What the caller should do once the options are read.
enum class ParseOutcome : std::uint8_t { kRun, kDone, kFailed };

ParseOutcome parse_args(int argc, char** argv, CliOptions& opt) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return ParseOutcome::kDone;
        }
        if (arg == "--version") {
            std::cout << "cbls " << cbls::kVersion << "\n";
            return ParseOutcome::kDone;
        }
        if (take_toggle_option(arg, opt)) {
            continue;
        }
        const FlagStatus status = take_value_option(arg, i, argc, argv, opt);
        if (status == FlagStatus::kError) {
            return ParseOutcome::kFailed;
        }
        if (status == FlagStatus::kOk) {
            continue;
        }
        if (arg[0] == '-') {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            return ParseOutcome::kFailed;
        }
        opt.model_path = arg;
    }
    return ParseOutcome::kRun;
}

// The cooperative portfolio, over the Model run_cli loaded once: every worker
// is a replica sharing one immutable structure (#157).
//
// The model comes back frozen, with the `obj <= bound` row, and its assignment
// untouched: run_cli restores the winner before printing. Returns false having
// reported the failure.
bool solve_parallel(const CliOptions& opt, Model& model, int effective_threads,
                    SolveCallback* callback, SearchResult& result) {
    std::function<std::shared_ptr<InnerSolverHook>(Model&)> hook_factory;
    if (opt.use_intensify) {
        hook_factory = [](Model&) -> std::shared_ptr<InnerSolverHook> {
            return std::make_shared<FloatIntensifyHook>();
        };
    }

    std::function<std::shared_ptr<LNS>()> lns_factory;
    if (opt.lns_fraction > 0.0) {
        const double fraction = opt.lns_fraction;
        lns_factory = [fraction]() -> std::shared_ptr<LNS> {
            return std::make_shared<LNS>(fraction);
        };
    }

    ParallelConfig par_config;
    par_config.n_threads = effective_threads;

    ParallelSearch ps(effective_threads);
    // solve() throws when every worker threw; report it like a load failure.
    try {
        result = ps.solve(model, opt.time_limit, opt.seed, opt.config, hook_factory, lns_factory,
                          callback, par_config);
    } catch (const std::exception& e) {
        std::cerr << "Error: parallel search failed: " << e.what() << "\n";
        return false;
    }
    // A dead worker is absorbed and the survivors' answer returned (#170); say
    // so, since the run asked for is not the run that happened.
    if (result.workers_completed < effective_threads) {
        std::cerr << "Warning: " << result.workers_completed << " of " << effective_threads
                  << " portfolio workers completed\n";
        for (const WorkerFailure& f : result.worker_failures) {
            std::cerr << "  worker " << f.worker << ": " << f.reason << "\n";
        }
    }
    return true;
}

SearchResult solve_single(const CliOptions& opt, Model& model, SolveCallback* callback) {
    FloatIntensifyHook intensify_hook;
    InnerSolverHook* hook = opt.use_intensify ? &intensify_hook : nullptr;

    LNS lns_obj(opt.lns_fraction);
    LNS* lns_ptr = opt.lns_fraction > 0.0 ? &lns_obj : nullptr;

    return solve(model, opt.time_limit, opt.seed, opt.use_fj, hook, lns_ptr,
                 opt.config.lns_interval, callback, opt.config);
}

int run_cli(int argc, char** argv) {
    CliOptions opt;
    switch (parse_args(argc, argv, opt)) {
        case ParseOutcome::kDone:
            return 0;
        case ParseOutcome::kFailed:
            return 1;
        case ParseOutcome::kRun:
            break;
    }

    if (opt.model_path.empty()) {
        std::cerr << "Error: no model file specified. Use --help for usage.\n";
        return 1;
    }

    Model model;
    try {
        model = load_model(opt.model_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << "\n";
        return 1;
    }

    // Set up formatter
    HumanFormatter human_fmt(std::cout);
    JsonlFormatter jsonl_fmt(std::cout);

    SolveCallback* callback = nullptr;
    if (!opt.quiet) {
        if (opt.format == "human") {
            human_fmt.print_header(opt.model_path, model, opt.seed, opt.time_limit);
            callback = &human_fmt;
        } else {
            jsonl_fmt.print_header(opt.model_path, model, opt.seed, opt.time_limit);
            callback = &jsonl_fmt;
        }
    }

    // Determine effective thread count
    int effective_threads = opt.n_threads;
    if (effective_threads == 0) {
        // hardware_concurrency() is allowed to return 0 when it cannot tell.
        effective_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    SearchResult result;
    if (effective_threads > 1) {
        if (!solve_parallel(opt, model, effective_threads, callback, result)) {
            return 1;
        }
        // The formatters print this model's variables, not result.best_state;
        // the workers searched their own models, so restore the winner here
        // (single-threaded solve() does it itself).
        if (result.best_state.values.size() == model.num_vars() &&
            result.best_state.elements.size() == model.num_vars()) {
            model.restore_state(result.best_state);
            full_evaluate(model);
        }

    } else {
        result = solve_single(opt, model, callback);
    }

    if (opt.format == "human") {
        human_fmt.print_result(result, model);
    } else {
        jsonl_fmt.print_result(result, model);
    }

    return result.feasible ? 0 : 1;
}

}  // namespace

int main(int argc, char* argv[]) {
    // No exception may escape main (std::terminate, no diagnostic, #130): Model
    // accessors under solve() and the formatters can throw, and
    // bugprone-exception-escape cannot see across translation units. As in the
    // benchmark runners' main.
    try {
        return run_cli(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Error: unknown fatal error\n";
        return 1;
    }
}
