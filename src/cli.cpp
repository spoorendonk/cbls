#include "cbls/arg_parse.h"
#include "cbls/cbls.h"
#include "cbls/formatter.h"
#include "cbls/io.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <thread>

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

int run_cli(int argc, char* argv[]) {
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
    int n_threads = 1;
    bool deterministic = false;
    int64_t epoch_iters = 5000;
    int max_epochs = 10;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help();
            return 0;
        }
        if (arg == "--version") {
            std::cout << "cbls " << cbls::kVersion << "\n";
            return 0;
        }
        if (arg == "--time-limit" && i + 1 < argc) {
            if (!parse_flag("--time-limit", argv[++i], time_limit)) {
                return 1;
            }
        } else if (arg == "--seed" && i + 1 < argc) {
            if (!parse_flag("--seed", argv[++i], seed)) {
                return 1;
            }
        } else if (arg == "--no-fj") {
            use_fj = false;
            config.use_fj = false;
        } else if (arg == "--lns" && i + 1 < argc) {
            if (!parse_flag("--lns", argv[++i], lns_fraction)) {
                return 1;
            }
        } else if (arg == "--lns-interval" && i + 1 < argc) {
            if (!parse_flag("--lns-interval", argv[++i], lns_interval)) {
                return 1;
            }
            config.lns_interval = lns_interval;
        } else if (arg == "--intensify") {
            use_intensify = true;
        } else if (arg == "--format" && i + 1 < argc) {
            format = argv[++i];
            if (format != "human" && format != "jsonl") {
                std::cerr << "Error: --format must be 'human' or 'jsonl'\n";
                return 1;
            }
        } else if (arg == "--threads" && i + 1 < argc) {
            if (!parse_flag("--threads", argv[++i], n_threads)) {
                return 1;
            }
        } else if (arg == "--deterministic") {
            deterministic = true;
        } else if (arg == "--epoch-iters" && i + 1 < argc) {
            if (!parse_flag("--epoch-iters", argv[++i], epoch_iters)) {
                return 1;
            }
        } else if (arg == "--max-epochs" && i + 1 < argc) {
            if (!parse_flag("--max-epochs", argv[++i], max_epochs)) {
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

    // Determine effective thread count
    int effective_threads = n_threads;
    if (effective_threads == 0) {
        // hardware_concurrency() is allowed to return 0 when it cannot tell.
        effective_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }

    SearchResult result;

    if (effective_threads > 1 || deterministic) {
        // Parallel mode: use ParallelSearch
        // Capture model_path for the factory (model is loaded once, factory re-loads)
        auto model_factory = [&model_path]() { return load_model(model_path); };

        std::function<InnerSolverHook*(Model&)> hook_factory;
        if (use_intensify) {
            hook_factory = [](Model&) -> InnerSolverHook* { return new FloatIntensifyHook(); };
        }

        std::function<LNS*()> lns_factory;
        if (lns_fraction > 0.0) {
            lns_factory = [lns_fraction]() -> LNS* { return new LNS(lns_fraction); };
        }

        ParallelConfig par_config;
        par_config.n_threads = effective_threads;
        par_config.deterministic = deterministic;
        par_config.epoch_iterations = epoch_iters;
        par_config.max_epochs = max_epochs;

        ParallelSearch ps(effective_threads);
        // solve() throws when every portfolio worker threw -- the factory could
        // not re-read the model file, say. Report that the way the load failure
        // above is reported; letting it escape main is std::terminate.
        try {
            result = ps.solve(model_factory, time_limit, seed, config, hook_factory, lns_factory,
                              callback, par_config);
        } catch (const std::exception& e) {
            std::cerr << "Error: parallel search failed: " << e.what() << "\n";
            return 1;
        }
    } else {
        // Single-thread mode: use solve() directly
        FloatIntensifyHook intensify_hook;
        InnerSolverHook* hook = use_intensify ? &intensify_hook : nullptr;

        LNS lns_obj(lns_fraction);
        LNS* lns_ptr = lns_fraction > 0.0 ? &lns_obj : nullptr;

        result =
            solve(model, time_limit, seed, use_fj, hook, lns_ptr, lns_interval, callback, config);
    }

    if (format == "human") {
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
