// UC-CHPED benchmark runner: loads each `ucp*.jsonl` from
// `benchmarks/instances/uc-chped/`, builds the unit-commitment-with-valve-point
// model, runs a fixed-time ViolationLS pass per (instance, horizon), and writes
// a per-row table to `benchmarks/instances/uc-chped/comparison.csv`.
//
// `--feas-tol` (default `cbls::kDefaultFeasibilityTolerance`) is stated and
// recorded on every row, not inherited: a published result must not change
// when an engine default moves (#103).
//
// Two tolerances, not one number: `feasible` is the engine's verdict at
// `--feas-tol`; `verified` is verify_uc_chped()'s re-check, its UC-semantic
// checks at an absolute 1e-4 and its leading cbls::verify_model() pass at 1e-6.

#include "data.h"
#include "greedy_init.h"
#include "trace_recorder.h"
#include "uc_model.h"
#include "verify_uc_chped.h"

#include <algorithm>
#include <array>
#include <benchmarks/common/published_guard.h>
#include <benchmarks/common/runner_args.h>
#include <benchmarks/common/runner_io.h>
#include <benchmarks/common/search_config_flags.h>
#include <benchmarks/mipfeas/worker_accounting.h>
#include <cbls/cbls.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

// The search-flag families this runner's scalar models can use (#198): with no
// List or Set variable a structural flag would be recorded and read by nothing,
// so validate_search_flags() refuses one.
constexpr unsigned kSearchFlagFamilies = cbls::bench::kScalarRunnerSearchFlags;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// verify_uc_chped()'s UC-semantic tolerance, distinct from `--feas-tol`,
/// passed at the call site so the two cannot drift. `verified` also covers its
/// cbls::verify_model() pass at 1e-6 and a relative-band objective recheck.
constexpr double kVerifierTolerance = 1e-4;

struct InstanceSpec {
    std::string filename;
    std::vector<int> periods;
};

/// The published search protocol: engine defaults with compound moves (Novelty
/// Jump, engine default since 2026-10-07) OFF. A single-seed 10s smoke at
/// b8f52f4 (load above 1.5: an indication, not a result) found them worse on 16
/// of the 19 (instance, horizon) rows that differed (typically +0.2-1%
/// objective, up to +2.4% on ucp13 3p), against better on MINLPLib.
/// `--compound-moves` is an ablation arm, refused onto the published table.
cbls::bench::SearchFlags published_search_flags() {
    cbls::bench::SearchFlags f;
    f.compound_moves = false;
    return f;
}

struct Args {
    std::string inst_dir = "benchmarks/instances/uc-chped";
    bool do_verify = false;
    // Unset: the per-horizon budget map. The uniform override is for smoke
    // tests; the budget applied is written to each row's time_limit_s, so a
    // short run is self-describing.
    bool time_limit_set = false;
    double time_limit = 0.0;
    // Seeds both the FJ warm-start RNG and solve(), so the two cannot drift.
    uint64_t seed = 42;
    // Absolute violation tolerance for "feasible", stated because it is a
    // published property; the default is the engine's, which matches SCIP's
    // numerics/feastol.
    double feas_tol = cbls::kDefaultFeasibilityTolerance;
    std::vector<std::string> instances;  // optional roster filter (base names)
    // Optional horizon filter on the selected files: a tuning study (#262)
    // scores one (instance, horizon) per run. Held as typed, so a refusal names
    // the value given.
    std::vector<int64_t> periods;
    // Portfolio workers: 1 is the single-threaded `solve()` the table measures;
    // >1 runs `ParallelSearch`, a different run, so not the published protocol.
    int threads = 1;
    bool commit_set = false;
    std::string commit_sha = "unknown";
    std::string out_csv;    // default: <inst_dir>/comparison.csv
    std::string trace_csv;  // optional: anytime profile (incumbent vs wall time, #147)
    // The ablation arm (#136); its canonical spelling is recorded on every
    // measured row, so a results file states its own configuration.
    cbls::bench::SearchFlags search = published_search_flags();
    // search_config_string(search), plus threads=N off 1; filled in by parse_args
    std::string search_config;
};

// Parse rule: include/cbls/arg_parse.h; report-and-exit policy:
// benchmarks/common/runner_args.h, shared rather than copied (#130). Numbers go
// through std::stod/std::stoll with trailing characters rejected, since std::ato*
// has no error path and a typo would run as 0
// (bugprone-unchecked-string-to-number-conversion). A bad double yields NaN for
// the `> 0.0` guards below to fail; a bad integer reports and exits 2 directly.
using cbls::bench::parse_double;
using cbls::bench::parse_int64;

/// True when a double flag names a usable positive quantity. Both failure modes
/// of the parse layer fall out here: NaN (the token was not a number) and inf
/// (std::stod accepts "inf" happily).
bool is_positive_finite(double v) {
    return v > 0.0 && std::isfinite(v);
}

/// The numeric flags' value guards, apart from the flag table: the parse layer
/// reports a bad token and hands over NaN, the guard decides the exit code
/// (tests/python/test_run_benchmark.py pins both halves).
void check_numeric_flags(Args& a) {
    std::string flag_error;
    if (!cbls::bench::validate_search_flags(a.search, kSearchFlagFamilies, a.time_limit_set,
                                            flag_error)) {
        std::fprintf(stderr, "%s\n", flag_error.c_str());
        std::exit(2);
    }
    a.search_config = cbls::bench::search_config_string(a.search, kSearchFlagFamilies);
    // The portfolio size is part of the row's configuration, so a campaign
    // scorer can refuse a mix (#363); written only off 1, so a `--threads 1`
    // row's cell stays as published.
    if (a.threads != 1) {
        a.search_config += (a.search_config.empty() ? "" : ";");
        a.search_config += "threads=" + std::to_string(a.threads);
    }
    // NaN, 0 or negative fails: solve() would return having searched nothing,
    // a full-looking table of empty results; isfinite too, as "inf" would never
    // terminate. The wall-clock-free arm (#136) is the one way past: solve()
    // disables its clock at `time_limit <= 0`, and validate_search_flags has
    // required a positive --max-iterations, so the run still has a budget.
    if (a.search.no_time_limit) {
        a.time_limit_set = true;
        a.time_limit = 0.0;
    } else if (a.time_limit_set && !is_positive_finite(a.time_limit)) {
        std::fprintf(stderr, "--time-limit must be > 0 (got %g)\n", a.time_limit);
        std::exit(2);
    }
    // isfinite too: std::stod accepts "inf", and an infinite tolerance calls
    // every assignment feasible, publishing rows the verifier would reject.
    if (!is_positive_finite(a.feas_tol)) {
        std::fprintf(stderr, "--feas-tol must be > 0 (got %g)\n", a.feas_tol);
        std::exit(2);
    }
}

/// The reason this run is not the full published measurement, or nullptr. One
/// answer for both published artifacts, results table and trace (#147), which
/// such a run must never overwrite: a partial roster, threads off 1, a
/// shortened budget (`--time-limit 2` would publish 2s results, #88), or an
/// ablation arm, whose rows look published but come from another search (#136).
/// The arm is tested before --time-limit because `--no-time-limit` forces a
/// zero budget; it reports --max-iterations (which --no-time-limit requires and
/// which sorts first among the arm's flags), not a flag never typed.
const char* non_published_protocol(const Args& a) {
    if (!a.instances.empty()) {
        return "--instance";
    }
    if (!a.periods.empty()) {
        return "--periods";
    }
    if (a.threads != 1) {
        return "--threads";
    }
    const char* arm =
        cbls::bench::first_non_default_search_flag(a.search, published_search_flags());
    if (arm != nullptr) {
        return arm;
    }
    if (a.time_limit_set) {
        return "--time-limit";
    }
    return nullptr;
}

#ifndef CBLS_RUNNER_BUILD_TYPE
#define CBLS_RUNNER_BUILD_TYPE ""  // built outside CMakeLists.txt: unknown, so refused
#endif

/// Why this binary must not write a published artifact, or nullptr. Read from
/// compile-time state because the runner is invoked directly and `preflight.py`'s
/// build-directory check can be skipped (#146): a Debug, sanitizer or
/// frame-pointer binary measures a different engine at a wall-clock budget.
const char* build_refusal() {
    if (std::string_view(CBLS_RUNNER_BUILD_TYPE) != "Release") {
        return "not a Release build";
    }
#ifndef NDEBUG
    return "built without NDEBUG";
#elif defined(CBLS_INSTRUMENTED_BUILD)
    return "built with CBLS_SANITIZE or CBLS_PROFILE";
#elif defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
    return "built with a sanitizer";
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
    __has_feature(memory_sanitizer) || __has_feature(undefined_behavior_sanitizer)
    return "built with a sanitizer";
#else
    return nullptr;
#endif
#else
    return nullptr;
#endif
}

/// Fills in `--out` and enforces what may overwrite this benchmark's published
/// artifacts: the shared guards (published_guard.h), then this runner's own
/// two, `--verify` and a Release binary.
void resolve_out_csv(Args& a) {
    if (a.out_csv.empty()) {
        a.out_csv = a.inst_dir + "/comparison.csv";
    }
    const std::optional<cbls::bench::PublishedTarget> target = cbls::bench::check_published_write(
        a.out_csv, a.trace_csv, a.inst_dir, non_published_protocol(a), a.commit_set);
    if (!target.has_value()) {
        return;
    }
    // An independent re-check is an acceptance criterion of the published run
    // (#146); without --verify every `verified` cell is empty. A trace-only
    // publish is held to it too, as only that run writes the trace: refusing
    // fails closed, and the message names the artifact actually given.
    if (!a.do_verify) {
        std::fprintf(
            stderr,
            "writing %s requires --verify%s "
            "(pass %s elsewhere for an unverified run)\n",
            target->path.c_str(),
            target->table ? "" : ": the published trace belongs to the verified published run",
            target->redirect());
        std::exit(2);
    }
    const char* const unfit = build_refusal();
    if (unfit != nullptr) {
        std::fprintf(stderr,
                     "this binary cannot write the published %s %s: it was %s "
                     "(rebuild in a Release build directory without CBLS_SANITIZE/CBLS_PROFILE)\n",
                     target->artifact(), target->path.c_str(), unfit);
        std::exit(2);
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    bool inst_dir_set = false;
    cbls::bench::ArgCursor c(argc, argv);
    const char* v = nullptr;
    while (c.advance()) {
        const std::string s = c.arg();
        if (s == "--verify") {
            a.do_verify = true;
        } else if (c.value_flag("--time-limit", v)) {
            a.time_limit = parse_double("--time-limit", v);
            a.time_limit_set = true;
        } else if (c.value_flag("--seed", v)) {
            const int64_t seed = parse_int64("--seed", v);
            // The seed is published on every measured row so a run can be
            // repeated; a negative one wraps to a uint64_t that --seed rejects,
            // so the recorded value would not be usable as an input.
            if (seed < 0) {
                std::fprintf(stderr, "--seed must be >= 0 (got %lld)\n",
                             static_cast<long long>(seed));
                std::exit(2);
            }
            a.seed = static_cast<uint64_t>(seed);
        } else if (c.value_flag("--feas-tol", v)) {
            a.feas_tol = parse_double("--feas-tol", v);
        } else if (c.value_flag("--instance", v)) {
            a.instances.emplace_back(v);
        } else if (c.value_flag("--periods", v)) {
            // No range check of its own: filter_specs refuses a horizon no
            // selected file has, which covers 0 and negatives too.
            a.periods.push_back(parse_int64("--periods", v));
        } else if (c.value_flag("--threads", v)) {
            a.threads = cbls::bench::parse_thread_count(v);
        } else if (c.value_flag("--commit", v)) {
            a.commit_sha = v;
            a.commit_set = true;
        } else if (c.value_flag("--out", v)) {
            a.out_csv = v;
        } else if (c.value_flag("--trace", v)) {
            a.trace_csv = v;
        } else if (cbls::bench::match_search_flag(c, s, a.search)) {
            // A search-configuration flag (#136); the table, the parse and the
            // recorded spelling all live in benchmarks/common/search_config_flags.h.
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_uc_chped [inst-dir] [--verify] [--time-limit S] [--seed N]"
                " [--feas-tol T] [--instance NAME ...] [--periods N ...] [--threads N]"
                " [--commit SHA] [--out CSV] [--trace CSV]\n"
                "       %s\n",
                cbls::bench::search_flags_usage(kSearchFlagFamilies).c_str());
            std::exit(0);
        } else if (!s.empty() && s[0] == '-') {
            // A typo'd flag must not become the instance directory and yield a
            // plausible but wrong published table; one leading dash, so `-x` too.
            std::fprintf(stderr, "Unknown or incomplete option: %s (see --help)\n", s.c_str());
            std::exit(2);
        } else if (inst_dir_set) {
            // Same hazard: with last-one-wins, `cbls_uc_chped dirA dirB` reads a
            // roster nobody named.
            std::fprintf(stderr, "Only one instance directory may be given (got '%s' and '%s')\n",
                         a.inst_dir.c_str(), s.c_str());
            std::exit(2);
        } else {
            a.inst_dir = s;
            inst_dir_set = true;
        }
    }
    check_numeric_flags(a);
    resolve_out_csv(a);
    return a;
}

/// The scratch file must not outlive a failed run: it sits in a tracked,
/// unignored directory where `git add -A` would commit a truncated shadow of the
/// table. Every path this program controls cleans up; a kill -9 can still leave
/// one, which is why the table is written to it and renamed.
void remove_temp(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

/// Removes the scratch file unless the run reached its rename. RAII, not a call
/// per error path, because the solve loop can throw (Model::var out_of_range,
/// make_subinstance invalid_argument) past every such call.
class TempFileGuard {
public:
    explicit TempFileGuard(std::string path) : path_(std::move(path)) {}
    TempFileGuard(const TempFileGuard&) = delete;
    TempFileGuard& operator=(const TempFileGuard&) = delete;
    TempFileGuard(TempFileGuard&&) = delete;
    TempFileGuard& operator=(TempFileGuard&&) = delete;
    ~TempFileGuard() {
        if (armed_) {
            remove_temp(path_);
        }
    }
    void release() { armed_ = false; }

private:
    std::string path_;
    bool armed_ = true;
};

/// One CSV line. The first nine fields are the pre-#103 schema in its original
/// order, so a reader of the old file reads the same columns; the rest append.
struct Row {
    std::string instance;
    int periods = 0;
    std::string method;
    double objective = kNaN;
    double lb = kNaN;
    double gap_pct = kNaN;
    double time_s = kNaN;
    std::string source;
    std::string note;
    double ub = kNaN;
    double time_limit_s = kNaN;
    std::string seed;      // empty on cited rows: they are not our measurement
    std::string feasible;  // "true" / "false" / "" (not applicable)
    std::string verified;  // "true" / "false" / "" (--verify not requested)
    double max_violation = kNaN;
    double feas_tol = kNaN;
    std::string commit_sha;
    // Empty on the cited Pedroso rows, which are not measurements of this engine
    // (as they carry no seed or tolerance); every MEASURED row carries it (#136).
    std::string search_config;
};

/// A NaN prints as an empty cell rather than "NaN": the published table already
/// spelled "not applicable" that way (the cited rows carry no time_s), and an
/// empty cell reads as missing in every spreadsheet and dataframe loader.
std::string num(double v, int precision = 10) {
    if (std::isnan(v)) {
        return "";
    }
    std::ostringstream os;
    os.precision(precision);
    os << v;
    return os.str();
}

std::string fixed2(double v) {
    if (std::isnan(v)) {
        return "";
    }
    std::array<char, 64> buf{};
    std::snprintf(buf.data(), buf.size(), "%.2f", v);
    return {buf.data()};
}

// The CSV cell rule, shared with the anytime trace (trace_recorder.h) so the
// two output files cannot diverge in what they substitute.
using cbls::uc_chped::csv_text;

const char* const kCsvHeader =
    "instance,periods,method,objective,lb,gap_pct,time_s,source,note,"
    "ub,time_limit_s,seed,feasible,verified,max_violation,feas_tol,commit_sha,search_config";

void write_row(std::ostream& csv, const Row& r) {
    csv << csv_text(r.instance) << "," << r.periods << "," << csv_text(r.method) << ","
        << num(r.objective) << "," << num(r.lb) << "," << fixed2(r.gap_pct) << ","
        << num(r.time_s, 4) << "," << csv_text(r.source) << "," << csv_text(r.note) << ","
        << num(r.ub) << "," << num(r.time_limit_s, 4) << "," << r.seed << "," << r.feasible << ","
        << r.verified << "," << num(r.max_violation, 4) << "," << num(r.feas_tol, 3) << ","
        << csv_text(r.commit_sha) << "," << csv_text(r.search_config) << "\n";
}

/// The provenance block. Emitted by the generator so that regenerating the table
/// cannot lose it; the copy committed alongside the cited rows says the same
/// thing.
void write_header_comment(std::ostream& csv, const Args& args) {
    csv << "# UC-CHPED comparison table. GENERATED by benchmarks/uc-chped/uc_chped.cpp\n"
           "# (`cbls_uc_chped --out <this file> --commit <preflight.py's SHA> --verify`;\n"
           "# the full command is in docs/benchmarks/uc-chped/README.md).\n"
           "# Do not hand-edit the rows; re-run the generator.\n"
           "#\n"
           "# Model fidelity: our CBLS model and the SCIP reference (reference_solve.py)\n"
           "# both solve a ramp-free unit commitment -- no |P[t]-P[t-1]| constraints -- and\n"
           "# so does the source. Pedroso, Kubo & Viana 2014 states power balance, spinning\n"
           "# reserve, unit initial conditions and minimum up/down times only, with no\n"
           "# ramp-rate constraints\n"
           "# (https://web.fc.up.pt/dcc/Pubs/TReports/TR14/dcc-2014-05.pdf), and their\n"
           "# public instance-generation code carries no ramp data. The ramp question the\n"
           "# #73 audit left open is therefore settled (#77, closed as not planned).\n"
           "#\n"
           "# Instance identity (#148, #193): ucp13 and ucp40 are field-for-field the\n"
           "# authors' instances (their GPL ucp_data.py) and the three proven optima are\n"
           "# reproduced by an exact solve, so gap_pct is against bounds for these\n"
           "# instances. Our demand row is >= where the source has ==, which does not\n"
           "# move the proven optima. The SCIP reference's piecewise-linear valve-point\n"
           "# term is not the ~0.1% approximation once claimed -- uniform breakpoints\n"
           "# straddle the valve-point cusps. See benchmarks/uc-chped/FIDELITY.md section 7.\n"
           "#\n"
           "# Two tolerances, not one: `feasible` is the engine's own verdict at the\n"
           "# `feas_tol` recorded on the row, while `verified` is an independent re-check\n"
           "# by verify_uc_chped() (benchmarks/uc-chped/verify_uc_chped.h) at ITS OWN\n"
           "# tolerances -- 1e-4 absolute on the UC-semantic checks, 1e-6 on the generic\n"
           "# cbls::verify_model() pass it runs first, and a relative band on the objective\n"
           "# recomputation. None of the three is `feas_tol`. `verified` is empty when the\n"
           "# run was made without --verify, and on a row the engine called infeasible:\n"
           "# there is no solution to re-check. A row that failed verification publishes no\n"
           "# objective and no gap: those columns would describe a solution we do not stand\n"
           "# behind.\n"
           "#\n"
           "# `lb`/`ub` are the Pedroso Table 2 bounds carried in the instance jsonl, and\n"
           "# the `Pedroso MIP (1hr)` rows are those bounds restated as cited reference\n"
           "# results -- they are not measurements of this engine and so carry no seed,\n"
           "# tolerance or commit. They are re-emitted from each instance's bounds map on\n"
           "# every run, and the generator refuses to write at all unless every instance\n"
           "# it was asked to run loaded, so a regeneration cannot drop them. Note that\n"
           "# scope is the roster it was asked for: a --instance-filtered run emits the\n"
           "# cited rows for the instances it names and no others, which is why such a\n"
           "# run also refuses the published path.\n"
           "#\n";
    csv << "# Run: commit " << csv_text(args.commit_sha) << ", seed " << args.seed << ", feas-tol "
        << num(args.feas_tol, 3) << ", time limit ";
    if (args.search.no_time_limit) {
        csv << "none (--no-time-limit; the iteration budget is the only one)";
    } else if (args.time_limit_set) {
        csv << num(args.time_limit, 4) << "s (uniform override)";
    } else {
        csv << "per-horizon default map";
    }
    csv << ", verify " << (args.do_verify ? "on" : "off");
    csv << ", search config " << csv_text(args.search_config);
    // A filtered table otherwise looks exactly like a full one minus rows.
    if (!args.instances.empty()) {
        csv << ", roster filtered to";
        for (const auto& name : args.instances) {
            csv << " " << csv_text(name);
        }
    }
    csv << ".\n";
}

std::string base_name(const std::string& filename) {
    const std::string suffix = ".jsonl";
    if (filename.size() > suffix.size() &&
        filename.compare(filename.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return filename.substr(0, filename.size() - suffix.size());
    }
    return filename;
}

struct Tally {
    int solved = 0;
    int feasible = 0;
    int verified_pass = 0;
    int verified_fail = 0;
    // Solves whose portfolio completed fewer workers than `--threads` asked
    // for (#170): the row is written, but it is not the N-thread run it says.
    int lost_workers = 0;
};

/// Restate an instance's published Table 2 bounds as cited reference rows,
/// regenerated from `known_bounds` each run so the ten cited rows survive
/// without the writer parsing its own output. run_benchmark() refuses to write
/// unless every rostered instance loaded, else a wrong-directory run would drop
/// them. Instances with no published bounds (ucp100/ucp200) get none: an
/// invented citation would be worse than none.
void write_reference_rows(std::ostream& csv, const cbls::uc_chped::UCInstance& base,
                          const std::vector<int>& periods) {
    for (int horizon : periods) {
        auto it = base.known_bounds.find(horizon);
        if (it == base.known_bounds.end()) {
            continue;
        }
        const double lb = it->second.first;
        const double ub = it->second.second;
        Row r;
        r.instance = base.name;
        r.periods = horizon;
        r.method = "Pedroso MIP (1hr)";
        r.objective = ub;
        r.lb = lb;
        r.ub = ub;
        r.gap_pct = lb != 0.0 ? 100.0 * (ub - lb) / lb : kNaN;
        r.source = "Pedroso et al. 2014 Table 2";
        r.note = "cited reference; ramp-free (see header)";
        write_row(csv, r);
    }
}

/// Build, warm-start and search one model: the measured part of the row,
/// independent of how it is scored or printed.
cbls::SearchResult solve_instance(const Args& args, const cbls::uc_chped::UCInstance& inst,
                                  cbls::uc_chped::UCModel& ucm, double tlim,
                                  cbls::SolveCallback* callback) {
    // Greedy initialization + short FJ polish.
    cbls::uc_chped::greedy_uc_initialize(ucm.model, inst, ucm);
    {
        cbls::RNG init_rng(args.seed);
        cbls::ViolationManager init_vm(ucm.model);
        // The 200 iterations bound the polish; 1s is only a safety cap. Dropped
        // (0 = no wall clock) under --no-time-limit: a clock anywhere lets a slow
        // machine truncate the warm start of an iteration-budgeted arm (#136).
        const double init_limit = args.search.no_time_limit ? 0.0 : 1.0;
        cbls::fj_nl_initialize(ucm.model, init_vm, 200, &init_rng, init_limit);
    }

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    cbls::SearchConfig cfg;
    cfg.skip_init = true;
    cfg.feasibility_tolerance = args.feas_tol;
    // The ablation arm. Everything the flags touch is applied here and nowhere
    // else, so the `search_config` cell on the row describes exactly the
    // configuration this call received (#136).
    cbls::bench::apply_search_flags(args.search, cfg);
    const double time_limit = cbls::bench::time_limit_argument(args.search, tlim);
    if (args.threads == 1) {
        return cbls::solve(ucm.model, time_limit, args.seed, false,
                           cbls::bench::hook_argument(args.search, hook),
                           cbls::bench::lns_argument(args.search, lns),
                           cbls::bench::lns_interval_argument(args.search), callback, cfg);
    }
    // The master overload, as cbls_cvrp: workers copy the warm-started model
    // (skip_init keeps that start). One hook and one LNS per worker -- the search
    // locks neither, so a shared instance would be a data race (pool.h).
    std::function<std::shared_ptr<cbls::InnerSolverHook>(cbls::Model&)> hook_factory;
    if (args.search.float_hook) {
        hook_factory = [](cbls::Model&) { return std::make_shared<cbls::FloatIntensifyHook>(); };
    }
    std::function<std::shared_ptr<cbls::LNS>()> lns_factory;
    if (args.search.lns) {
        lns_factory = [] { return std::make_shared<cbls::LNS>(0.3); };
    }
    cfg.lns_interval = cbls::bench::lns_interval_argument(args.search);
    cbls::ParallelConfig par;
    par.n_threads = args.threads;
    cbls::ParallelSearch ps(args.threads);
    cbls::SearchResult result =
        ps.solve(ucm.model, time_limit, args.seed, cfg, hook_factory, lns_factory, callback, par);
    // The workers searched copies, so the master still holds its start: restore
    // the returned best, which is the state run_one scores and verifies. Guarded:
    // a portfolio whose deadline expired before any worker ran returns no state.
    if (result.best_state.values.size() == ucm.model.num_vars()) {
        ucm.model.restore_state(result.best_state);
        cbls::full_evaluate(ucm.model);
    }
    return result;
}

/// A solve result scored against the instance's published bounds, before the
/// verifier has had its say. `gap` is held apart from `row.gap_pct` because the
/// console prints it either way while the CSV publishes it only for a row we
/// stand behind.
struct Scored {
    Row row;
    double gap = kNaN;
    bool have_bounds = false;
    bool have_objective = false;
};

/// Scoring policy: which of the result's numbers are usable, deciding the row's
/// numeric columns; deliberately separate from the verifier's veto below.
Scored score_result(const Args& args, const cbls::uc_chped::UCInstance& inst,
                    const std::string& instance_name, double tlim,
                    const cbls::SearchResult& result) {
    Scored s;
    Row& row = s.row;
    row.instance = instance_name;
    row.periods = inst.n_periods;
    row.method = "CBLS ViolationLS";
    row.source = "this work";
    row.time_s = result.time_seconds;
    row.time_limit_s = tlim;
    row.seed = std::to_string(args.seed);
    row.feasible = result.feasible ? "true" : "false";
    row.max_violation = result.best_violation;
    row.feas_tol = args.feas_tol;
    row.commit_sha = args.commit_sha;
    row.search_config = args.search_config;

    auto it = inst.known_bounds.find(inst.n_periods);
    s.have_bounds = it != inst.known_bounds.end();
    if (s.have_bounds) {
        row.lb = it->second.first;
        row.ub = it->second.second;
    }

    // `feasible` does not imply a finite objective: +inf is a feasibility
    // witness whose objective overflowed (#100, SearchResult), and "inf" in a
    // numeric column would read as a solve result.
    s.have_objective = result.feasible && std::isfinite(result.objective);
    if (s.have_bounds && s.have_objective && row.lb != 0.0) {
        s.gap = 100.0 * (result.objective - row.lb) / row.lb;
    }
    return s;
}

void print_result_line(const cbls::SearchResult& result, const Scored& s) {
    if (s.have_bounds && s.have_objective) {
        std::printf("%12.1f %12.1f %7.2f%% %7.1fs", result.objective, s.row.lb, s.gap,
                    result.time_seconds);
    } else {
        std::printf("%12.1f %12s %8s %7.1fs", result.feasible ? result.objective : -1.0, "-",
                    result.feasible ? "-" : "INFEAS", result.time_seconds);
    }
}

/// Runs the independent checker and records its verdict. Returns true when the
/// row must not publish its objective or gap: a row the checker rejected would
/// otherwise describe a solution we do not stand behind.
bool verification_withholds(const Args& args, const cbls::uc_chped::UCModel& ucm,
                            const cbls::uc_chped::UCInstance& inst,
                            const cbls::SearchResult& result, Row& row, Tally& tally) {
    if (!args.do_verify || !result.feasible) {
        return false;
    }
    auto vr = cbls::uc_chped::verify_uc_chped(ucm, inst, kVerifierTolerance);
    std::printf("  %s", vr.ok ? "VERIFIED" : "VERIFY FAIL");
    row.verified = vr.ok ? "true" : "false";
    if (vr.ok) {
        ++tally.verified_pass;
        return false;
    }
    ++tally.verified_fail;
    row.note = "verify-fail (" + std::to_string(vr.errors.size()) + " errors)";
    vr.print_diagnostics(stdout);
    return true;
}

/// Solve one (instance, horizon) pair and append its row.
void run_one(std::ostream& csv, std::ofstream& trace, const Args& args,
             const cbls::uc_chped::UCInstance& inst, const std::string& instance_name, double tlim,
             Tally& tally) {
    auto ucm = cbls::uc_chped::build_uc_model(inst);
    const int horizon = inst.n_periods;

    std::printf("%-20s %6d %6d ", inst.name.c_str(), inst.n_units, horizon);
    std::fflush(stdout);

    // Trace rows carry the horizon too: one row per (instance, horizon), so the
    // instance alone cannot key a trace row to its solve (#147).
    cbls::uc_chped::TraceRecorder recorder(trace, instance_name, horizon, tlim, args.commit_sha);
    const cbls::SearchResult result =
        solve_instance(args, inst, ucm, tlim, trace.is_open() ? &recorder : nullptr);
    const bool lost_workers = cbls::mipfeas::lost_workers(result, args.threads);
    if (lost_workers) {
        std::fputs(cbls::mipfeas::lost_workers_message(inst.name, args.threads, result).c_str(),
                   stderr);
        ++tally.lost_workers;
    }
    // The closing row: every started (instance, horizon) appears once at its
    // final time (TraceRecorder::record_final, #147).
    if (trace.is_open()) {
        recorder.record_final(result);
    }
    ++tally.solved;
    if (result.feasible) {
        ++tally.feasible;
    }

    Scored scored = score_result(args, inst, instance_name, tlim, result);
    Row& row = scored.row;
    print_result_line(result, scored);

    const bool withhold = verification_withholds(args, ucm, inst, result, row, tally);
    if (scored.have_objective && !withhold) {
        row.objective = result.objective;
        row.gap_pct = scored.gap;
    }
    if (row.note.empty()) {
        // The exit code says the job failed; the note says which row, for a
        // scorer that only sees the table (#363).
        if (lost_workers) {
            row.note = "lost-workers";
        } else if (!result.feasible) {
            row.note = "infeasible at feas_tol";
        } else if (!scored.have_objective) {
            row.note = "feasible; no finite objective";
        }
    }

    std::printf("  (%s, %ld vars, %ld nodes, %ld iters)\n",
                result.feasible ? "feasible" : "INFEASIBLE",
                static_cast<long>(ucm.model.num_vars()), static_cast<long>(ucm.model.num_nodes()),
                static_cast<long>(result.iterations));

    write_row(csv, row);
    csv.flush();
}

/// Apply `--instance`. False when ANY requested name is not in the roster: an
/// unmatched name is a typo, and dropping it would silently publish a table
/// short a row. Keeps roster order; a repeated name contributes one entry.
bool filter_specs(const Args& args, std::vector<InstanceSpec>& specs) {
    if (args.instances.empty()) {
        return true;
    }
    bool all_matched = true;
    for (const auto& want : args.instances) {
        const bool found = std::any_of(specs.begin(), specs.end(), [&](const InstanceSpec& spec) {
            return base_name(spec.filename) == want;
        });
        if (!found) {
            std::fprintf(stderr, "--instance '%s' is not in the roster\n", want.c_str());
            all_matched = false;
        }
    }
    if (!all_matched) {
        return false;
    }
    std::vector<InstanceSpec> filtered;
    for (const auto& spec : specs) {
        const std::string name = base_name(spec.filename);
        if (std::find(args.instances.begin(), args.instances.end(), name) != args.instances.end()) {
            filtered.push_back(spec);
        }
    }
    specs = filtered;
    return true;
}

/// Apply `--periods` to the files `--instance` left. Refuses, as `--instance`
/// does, a horizon that no selected file has: a run asked for one row and
/// finding none would otherwise write an empty table at exit 0. A file left
/// with no horizon is dropped.
bool filter_periods(const Args& args, std::vector<InstanceSpec>& specs) {
    if (args.periods.empty()) {
        return true;
    }
    bool all_matched = true;
    for (const int64_t want : args.periods) {
        const bool found = std::any_of(specs.begin(), specs.end(), [&](const InstanceSpec& spec) {
            return std::find(spec.periods.begin(), spec.periods.end(), want) != spec.periods.end();
        });
        if (!found) {
            std::fprintf(stderr, "--periods %lld is not a horizon of the selected instances\n",
                         static_cast<long long>(want));
            all_matched = false;
        }
    }
    if (!all_matched) {
        return false;
    }
    std::vector<InstanceSpec> filtered;
    for (const auto& spec : specs) {
        InstanceSpec kept{spec.filename, {}};
        for (const int horizon : spec.periods) {
            if (std::find(args.periods.begin(), args.periods.end(), horizon) !=
                args.periods.end()) {
                kept.periods.push_back(horizon);
            }
        }
        if (!kept.periods.empty()) {
            filtered.push_back(std::move(kept));
        }
    }
    specs = filtered;
    return true;
}

void print_tally(const Args& args, const Tally& tally) {
    std::printf("=== Tally ===\n");
    std::printf("wrote:                %s\n", args.out_csv.c_str());
    if (!args.trace_csv.empty()) {
        std::printf("trace:                %s\n", args.trace_csv.c_str());
    }
    std::printf("commit:               %s\n", args.commit_sha.c_str());
    std::printf("seed:                 %llu\n", static_cast<unsigned long long>(args.seed));
    std::printf("feas-tol:             %g   (engine 'feasible')\n", args.feas_tol);
    std::printf("search config:        %s\n", args.search_config.c_str());
    std::printf("verifier tolerance:   %g   (independent 'verified'; its\n", kVerifierTolerance);
    std::printf("                              verify_model() pass uses 1e-6)\n");
    std::printf("solved:               %d\n", tally.solved);
    std::printf("feasible:             %d\n", tally.feasible);
    if (args.do_verify) {
        std::printf("verify passed:        %d\n", tally.verified_pass);
        std::printf("verify failed:        %d\n", tally.verified_fail);
    } else {
        std::printf("verify:               not run (--verify off)\n");
    }
}

/// The per-horizon wall-clock budget, unless `--time-limit` overrides it.
/// Every published gap is measured AT one of these, so a horizon still
/// improving at expiry publishes a budget artifact, not an engine result:
/// collect `--trace` evidence (#147) before changing a number.
double horizon_budget(const Args& args, int horizon) {
    if (args.time_limit_set) {
        return args.time_limit;
    }
    static const std::map<int, double> kBudgets = {
        {1, 10.0}, {3, 30.0}, {6, 60.0}, {12, 120.0}, {24, 300.0}, {48, 600.0}, {168, 600.0},
    };
    constexpr double kDefaultBudget = 300.0;
    auto it = kBudgets.find(horizon);
    return it != kBudgets.end() ? it->second : kDefaultBudget;
}

/// Reads every rostered instance, reporting each failure by name; false if ANY
/// failed. Runs BEFORE any output file opens: the cited rows come from
/// known_bounds, so a wrong-directory run would otherwise publish a table with
/// none. The instances are committed and nothing downloads them, so a missing
/// .jsonl is a setup error: report it and leave the existing files untouched.
bool load_roster(const Args& args, const std::vector<InstanceSpec>& specs,
                 std::vector<cbls::uc_chped::UCInstance>& loaded) {
    loaded.assign(specs.size(), cbls::uc_chped::UCInstance{});
    bool all_loaded = true;
    for (size_t i = 0; i < specs.size(); ++i) {
        try {
            loaded[i] = cbls::uc_chped::load_jsonl(args.inst_dir + "/" + specs[i].filename);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Cannot load %s: %s\n", specs[i].filename.c_str(), e.what());
            all_loaded = false;
        }
    }
    if (!all_loaded) {
        std::fprintf(stderr, "Refusing to write %s from an incomplete roster.\n",
                     args.out_csv.c_str());
    }
    return all_loaded;
}

/// Opens the anytime trace (#147) and writes its header; without `--trace` the
/// stream stays closed, which run_one reads as "pass no callback". Called only
/// after the roster loads and the table's temp file opens, since `--trace`
/// truncates: a run that cannot proceed leaves the trace as it found it.
/// Written in place, not renamed: an interrupted run's value is the rows it
/// already flushed.
bool open_trace(const Args& args, std::ofstream& trace) {
    if (args.trace_csv.empty()) {
        return true;
    }
    trace.open(args.trace_csv);
    if (!trace.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", args.trace_csv.c_str());
        return false;
    }
    trace << cbls::uc_chped::kTraceHeader << "\n";
    trace.flush();
    return true;
}

/// The measurement: every (instance, horizon) pair in roster order. The files
/// (which may be written, opening, publishing) stay with run_benchmark.
void solve_roster(std::ostream& csv, std::ofstream& trace, const Args& args,
                  const std::vector<InstanceSpec>& specs,
                  const std::vector<cbls::uc_chped::UCInstance>& loaded, Tally& tally) {
    for (size_t i = 0; i < specs.size(); ++i) {
        const cbls::uc_chped::UCInstance& base = loaded[i];
        for (int horizon : specs[i].periods) {
            if (horizon > base.n_periods) {
                std::printf("%-20s %6d %6d  (skipped: T > n_periods)\n", base.name.c_str(),
                            base.n_units, horizon);
                Row r;
                r.instance = base.name;
                r.periods = horizon;
                r.method = "CBLS ViolationLS";
                r.note = "skipped: T > n_periods";
                r.commit_sha = args.commit_sha;
                r.search_config = args.search_config;
                write_row(csv, r);
                continue;
            }
            // `horizon == base.n_periods` is already the right size.
            const cbls::uc_chped::UCInstance inst =
                horizon < base.n_periods ? cbls::uc_chped::make_subinstance(base, horizon) : base;
            run_one(csv, trace, args, inst, base.name, horizon_budget(args, horizon), tally);
        }
        std::printf("\n");
    }
}

int run_benchmark(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // Instance specs: filename + period options.
    std::vector<InstanceSpec> specs = {
        {"ucp13.jsonl", {1, 3, 6, 12, 24}},  {"ucp40.jsonl", {1, 3, 6, 12, 24}},
        {"ucp100.jsonl", {1, 3, 6, 12, 24}}, {"ucp100-48p.jsonl", {48}},
        {"ucp100-168p.jsonl", {168}},        {"ucp200.jsonl", {1, 3, 6, 12, 24}},
        {"ucp200-48p.jsonl", {48}},          {"ucp200-168p.jsonl", {168}},
    };
    if (!filter_specs(args, specs)) {
        std::fprintf(stderr, "--instance: unknown name(s) listed above (see --help)\n");
        return 2;
    }
    if (!filter_periods(args, specs)) {
        return 2;
    }

    std::vector<cbls::uc_chped::UCInstance> loaded;
    if (!load_roster(args, specs, loaded)) {
        return 2;
    }

    // Write-then-rename, as benchmarks/mipfeas does: this run takes over an hour
    // at the documented budgets, and a job killed mid-write must leave either
    // the previous table or none, never a truncated one.
    const std::string tmp_csv = args.out_csv + ".tmp";
    TempFileGuard tmp_guard(tmp_csv);
    std::ofstream csv(tmp_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", tmp_csv.c_str());
        return 2;
    }

    // Opened after the table's temp file: `--trace` truncates in place, so a
    // run that fails the table's open must not have emptied it.
    std::ofstream trace;
    if (!open_trace(args, trace)) {
        return 2;
    }
    write_header_comment(csv, args);
    csv << kCsvHeader << "\n";

    std::printf("%-20s %6s %6s %12s %12s %8s %8s\n", "Instance", "Units", "Periods", "Objective",
                "Known LB", "Gap%", "Time(s)");
    std::printf("%-20s %6s %6s %12s %12s %8s %8s\n", "--------", "-----", "-------", "---------",
                "--------", "----", "-------");

    Tally tally;

    // Cited rows first, so the published bounds head the table.
    for (size_t i = 0; i < specs.size(); ++i) {
        write_reference_rows(csv, loaded[i], specs[i].periods);
    }

    solve_roster(csv, trace, args, specs, loaded, tally);

    // A stream error means the table on disk is short rows nobody can see are
    // missing, so it must not be published or reported as success.
    csv.flush();
    if (!csv) {
        std::fprintf(stderr, "Error writing %s; %s not replaced\n", tmp_csv.c_str(),
                     args.out_csv.c_str());
        return 2;
    }
    csv.close();
    if (!cbls::bench::rename_into_place(tmp_csv, args.out_csv)) {
        return 2;
    }
    tmp_guard.release();  // it is the published table now, not scratch

    print_tally(args, tally);

    // A trace stream error does not unpublish the table (already in place and
    // valid alone) but exits nonzero: a truncated trace read as complete would
    // defend a budget on evidence never written.
    if (!args.trace_csv.empty() && !trace) {
        std::fprintf(stderr, "Error writing %s; the trace is incomplete\n", args.trace_csv.c_str());
        return 2;
    }
    // As cbls_mipfeas and cbls_cvrp: a portfolio that lost workers measured
    // fewer threads than its rows claim, so the job failed (#170).
    if (tally.lost_workers > 0) {
        std::fprintf(stderr, "%d solve(s) lost portfolio workers; exiting 1\n", tally.lost_workers);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    // Line-buffered even into a pipe: the documented command tees this to
    // comparison.log, the only record of WHICH checks failed on a VERIFY FAIL
    // row, and a fully buffered stdout loses its tail to a kill (#146). A few
    // lines per row, so no cost.
    (void)std::setvbuf(stdout, nullptr, _IOLBF, BUFSIZ);
    // An exception escaping main is std::terminate, an abort with no message
    // that looks like a crash: report it and exit 1 (bugprone-exception-escape).
    try {
        return run_benchmark(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
