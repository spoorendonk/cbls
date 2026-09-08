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
  --threads N           Number of threads (0 = auto-detect, default: 1)
  --deterministic       Enable deterministic epoch-sync parallel mode
  --epoch-iters INT     Iterations per epoch in deterministic mode (default: 5000)
  --max-epochs INT      Number of epochs in deterministic mode (default: 10)
  --format human|jsonl  Output format (default: human)
  --quiet               Suppress progress, print only final result
  --help                Show this help message
  --version             Show version number
)";
}

namespace {

// How a malformed numeric flag behaves, decided once for the whole option loop.
//
// Report at the parse and return 1, matching the `Error: ...`/exit-1 convention
// the rest of this loop already uses for an unknown option, a bad --format and a
// missing model file. The benchmark runners deliberately do the opposite for
// doubles -- report, return NaN, and let a later positivity guard exit 2 -- but
// that split exists to keep an exit code their drivers' tests pin, and it needs
// a guard to land in. The CLI has neither: nothing pins its codes, and there is
// no downstream guard for a NaN to be caught by, so it reports here instead.
//
// What is deliberately NOT rejected, because the CLI's budget is not only the
// clock: `--time-limit 0` and `--time-limit inf` are working configurations
// under --deterministic, where --epoch-iters/--max-epochs bind instead. A
// runner-style `!(x > 0.0)` guard would break them. NaN is different -- it is
// never a request anyone can mean, and it silently turns --lns off and
// --time-limit into a solve that never searched -- so the double overload
// rejects it and nothing else.
//
// Every overload returns false having already written the diagnostic; the caller
// only has to `return 1`. The parsing rule itself is cbls/arg_parse.h, shared
// with benchmarks/common/runner_args.h.

// Overflow is reported as such rather than as a typo. Both arrive here as a
// failed parse, but "99999999999999999999999 is not an integer" sends the
// reader hunting for a mistyped digit that is not there.
void report_parse_failure(const char* flag, const char* text, cbls::ParseStatus status,
                          const char* kind) {
    if (status == cbls::ParseStatus::kOutOfRange) {
        std::cerr << "Error: " << flag << ": '" << text << "' is out of range\n";
    } else {
        std::cerr << "Error: " << flag << ": '" << text << "' is not " << kind << "\n";
    }
}

bool parse_flag(const char* flag, const char* text, double& out) {
    // Syntax alone is not enough: std::stod accepts "nan", so the NaN the
    // comment above rules out would otherwise arrive as a successful parse. It
    // is malformed for this flag's purposes, not out of range.
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

// --seed spans the full unsigned 64-bit range, not the signed one: the CLI
// prints the seed back in its own header, where `--seed -1` is recorded as
// 18446744073709551615. Narrowing to int64 would make the tool unable to read
// back the seed it just printed, which is the whole point of the flag.
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
    int n_threads = 1;
    bool deterministic = false;
    int64_t epoch_iters = 5000;
    int max_epochs = 10;
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
    if (arg == "--deterministic") {
        opt.deterministic = true;
        return true;
    }
    if (arg == "--quiet") {
        opt.quiet = true;
        return true;
    }
    return false;
}

// The numeric options, each paired with the field it fills. A table rather than
// a chain of branches because that is all they are: the plumbing around them --
// is there a value after the flag, did it parse, report and stop -- is identical
// for every one, and the parse_flag overload set is what varies with the type.
struct ValueOption {
    const char* flag;
    std::variant<double*, int*, int64_t*, uint64_t*> target;
};

std::array<ValueOption, 7> numeric_options(CliOptions& opt) {
    return {{{"--time-limit", &opt.time_limit},
             {"--seed", &opt.seed},
             {"--lns", &opt.lns_fraction},
             {"--lns-interval", &opt.config.lns_interval},
             {"--threads", &opt.n_threads},
             {"--epoch-iters", &opt.epoch_iters},
             {"--max-epochs", &opt.max_epochs}}};
}

// One `--flag VALUE` option. Advances `i` past the value when it takes one.
//
// A flag with nothing after it reports kNotMine rather than an error, so the
// caller's unknown-option branch names it -- which is what a trailing `--seed`
// has always produced.
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

// Portfolio or epoch-sync mode. The model is re-read per worker rather than
// copied, which is why this takes the path and not the loaded Model.
// Returns false having already reported the failure.
bool solve_parallel(const CliOptions& opt, int effective_threads, SolveCallback* callback,
                    SearchResult& result) {
    // Capture model_path for the factory (model is loaded once, factory re-loads)
    auto model_factory = [&opt]() { return load_model(opt.model_path); };

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
    par_config.deterministic = opt.deterministic;
    par_config.epoch_iterations = opt.epoch_iters;
    par_config.max_epochs = opt.max_epochs;

    ParallelSearch ps(effective_threads);
    // solve() throws when every portfolio worker threw -- the factory could
    // not re-read the model file, say. Report that the way the load failure
    // in run_cli is reported; letting it escape main is std::terminate.
    try {
        result = ps.solve(model_factory, opt.time_limit, opt.seed, opt.config, hook_factory,
                          lns_factory, callback, par_config);
    } catch (const std::exception& e) {
        std::cerr << "Error: parallel search failed: " << e.what() << "\n";
        return false;
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
    if (effective_threads > 1 || opt.deterministic) {
        if (!solve_parallel(opt, effective_threads, callback, result)) {
            return 1;
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
    // The argument parse above is no longer the only way an exception could
    // reach here: solve() and the formatters call into Model, whose accessors
    // throw std::out_of_range and std::logic_error. An exception escaping main
    // is std::terminate -- an abort with no diagnostic and no usable exit
    // status, which is the failure issue #130 was opened for. Naming the bad
    // flag fixed the common route; this closes the class
    // (bugprone-exception-escape, which cannot see across translation units and
    // so does not flag it). Same shape as both benchmark runners' main.
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
