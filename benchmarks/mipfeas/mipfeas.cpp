// MIPfeas benchmark runner (CBLS side).
//
// ONE instance per process: the driver (run_benchmark.py) parallelises, caps
// memory and resumes, and a dying instance takes only its own process.
//
// Writes per instance:
//   <out-dir>/<instance>.json       result record (schema shared with cpsat_solve.py)
//   <out-dir>/<instance>.trace.csv  incumbent objective vs wall time, for
//                                   primal_integral.py
//   <solution-dir>/<instance>.sol   when given and feasible: for verify_solution.py
//                                   to check against the original file with a
//                                   third-party reader, the only check not
//                                   downstream of the MPS adapter (#138)
//
// Writes nothing when the instance file is absent, so it cannot be scored as
// "found nothing" (#103).

#include <algorithm>
#include <array>
#include <atomic>
#include <benchmarks/common/runner_args.h>
#include <benchmarks/common/runner_io.h>
#include <benchmarks/common/search_config_flags.h>
#include <benchmarks/mipfeas/original_check.h>
#include <benchmarks/mipfeas/worker_accounting.h>
#include <cbls/cbls.h>
#include <cbls/io_mps.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace {

// LNS destroy fraction, shared by the single-threaded path and the portfolio's
// per-worker factory: they must differ only in concurrency, else a --threads A/B
// measures two engines. The interval is `--lns-interval` (#267).
constexpr double kLnsFraction = 0.3;

// The shared search flags (search_config_flags.h) this runner takes, for
// `xk tune` (#267). The rest of the families duplicate this runner's own flags
// (--compound-moves, --no-objective-tie-break: own JSON keys) or misfit its
// protocol (--no-time-limit, --max-iterations against a wall-budget Primal
// Integral; --no-float-hook, unasked). Unlisted ones hit "Unknown argument".
constexpr std::array<const char*, 5> kSharedSearchFlags = {"--no-lns", "--lns-interval",
                                                           "--novelty-prob", "--unproductive-iters",
                                                           "--perturbation-period"};

/// The families the flags above belong to, for validate_search_flags().
constexpr unsigned kSharedSearchFamilies = cbls::bench::kScalarRunnerSearchFlags;

struct Args {
    std::string instance;
    std::string inst_dir = "benchmarks/instances/mipfeas";
    std::string out_dir;
    // MIPfeas scores the Primal Integral over a 600s budget.
    double budget = 600.0;
    uint64_t seed = 42;
    // Stated, not inherited: a published result must not move with an engine
    // default (#103).
    double feas_tol = cbls::kDefaultFeasibilityTolerance;
    // Novelty Jump (compound moves), stated for #103 though the engine default.
    // CP-SAT's `ls_restart_*compound*` subsolvers give 45-67% of improving
    // solutions on binkar10_1 and pk1; without it the gap would be FJ vs FJ +
    // Novelty Jump, not a reimplementation gap.
    bool compound_moves = true;
    // OR-Tools' objective tie-break/scan (#218), on as the engine default since
    // #226. Recorded per row so the scorer never averages on- with off-arms.
    bool objective_tie_break = true;
    // Finite-bound fallback for a column no constraint bounds (#120). 1e7, not
    // the engine's 1e9: it measured better on the smoke roster. CP-SAT clamps no
    // domains (`mip_max_bound` is not a clamp), so this is CBLS-only: the table
    // publishes `n_clamped_bounds` beside `n_unbounded_columns` (pre-propagation).
    double inf_clamp = 1.0e7;
    // Implied bounds from the rows before the clamp. Off disables propagation
    // only; honouring a wide finite bound (#120) is unconditional.
    bool propagate_bounds = true;
    // Changes the derived box, so recorded per result.
    int max_propagation_passes = 10;
    // 1 = single-threaded engine; >1 = `ParallelSearch` on that many threads.
    // Match the baseline's `num_workers` (CP-SAT runs N fj and N ls subsolvers),
    // else a CPU advantage reads as an implementation gap. A scorer config key.
    int threads = 1;
    // Shared objective bound (#179), on like the engine; off is the control arm.
    // A scorer config key; recorded at --threads 1 too (no pool) so rows share keys.
    bool share_objective_bound = true;
    // The kSharedSearchFlags arms (#267); other fields stay default. Each is a
    // scorer config key, dropped at its default so older rows read as default.
    cbls::bench::SearchFlags search;
    // Solution-vector output for #138's verification; empty writes none.
    std::string solution_dir;
    std::string commit_sha = "unknown";
};

void print_usage() {
    std::printf(
        "Usage: cbls_mipfeas --instance NAME --out-dir DIR [--inst-dir DIR]\n"
        "                    [--budget SECONDS] [--seed N] [--feas-tol T]\n"
        "                    [--inf-clamp B] [--no-propagate-bounds]\n"
        "                    [--max-propagation-passes N]\n"
        "                    [--no-compound-moves]\n"
        "                    [--no-objective-tie-break]\n"
        "                    [--no-lns] [--lns-interval N] [--novelty-prob P]\n"
        "                    [--unproductive-iters N] [--perturbation-period N]\n"
        "                    [--threads N]\n"
        "                    [--share-bound | --no-share-bound]\n"
        "                    [--solution-dir DIR]\n"
        "                    [--commit SHA]\n");
}

// Value policy (runner_args.h): a bad double is NaN for the positivity guards
// below (exit 2); a bad integer exits 2 directly.
using cbls::bench::file_exists;
using cbls::bench::parse_double;
using cbls::bench::parse_int64;
using cbls::bench::parse_thread_count;
using cbls::bench::peak_rss_kib;
using cbls::bench::seconds_since;

/// Range-checked, not cast: a wrapped value would silently change propagation
/// and the derived box at exit 0.
int parse_propagation_passes(const char* text) {
    const int64_t passes = parse_int64("--max-propagation-passes", text);
    if (passes < 0 || passes > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "--max-propagation-passes must be in [0, %d]\n",
                     std::numeric_limits<int>::max());
        std::exit(2);
    }
    return static_cast<int>(passes);
}

/// The search-behaviour arms: the flags that set an engine option rather than
/// the run (instance, budget, output). True when the current argument was one.
bool parse_search_flag(cbls::bench::ArgCursor& c, Args& a) {
    const std::string s = c.arg();
    const bool shared = std::any_of(kSharedSearchFlags.begin(), kSharedSearchFlags.end(),
                                    [&s](const char* flag) { return s == flag; });
    // The shared parse, so the value policy is the other runners'.
    if (shared) {
        return cbls::bench::match_search_flag(c, s, a.search);
    }
    if (s == "--compound-moves") {
        a.compound_moves = true;
    } else if (s == "--no-compound-moves") {
        a.compound_moves = false;
    } else if (s == "--objective-tie-break") {
        a.objective_tie_break = true;
    } else if (s == "--no-objective-tie-break") {
        a.objective_tie_break = false;
    } else if (s == "--share-bound") {
        a.share_objective_bound = true;
    } else if (s == "--no-share-bound") {
        a.share_objective_bound = false;
    } else {
        return false;
    }
    return true;
}

Args parse_args(int argc, char** argv) {
    Args a;
    cbls::bench::ArgCursor c(argc, argv);
    const char* v = nullptr;
    while (c.advance()) {
        const std::string s = c.arg();
        if (c.value_flag("--instance", v)) {
            a.instance = v;
        } else if (c.value_flag("--inst-dir", v)) {
            a.inst_dir = v;
        } else if (c.value_flag("--out-dir", v)) {
            a.out_dir = v;
        } else if (c.value_flag("--budget", v)) {
            a.budget = parse_double("--budget", v);
        } else if (c.value_flag("--seed", v)) {
            a.seed = static_cast<uint64_t>(parse_int64("--seed", v));
        } else if (c.value_flag("--feas-tol", v)) {
            a.feas_tol = parse_double("--feas-tol", v);
        } else if (c.value_flag("--inf-clamp", v)) {
            a.inf_clamp = parse_double("--inf-clamp", v);
        } else if (s == "--no-propagate-bounds") {
            a.propagate_bounds = false;
        } else if (c.value_flag("--max-propagation-passes", v)) {
            a.max_propagation_passes = parse_propagation_passes(v);
        } else if (parse_search_flag(c, a)) {
            continue;
        } else if (c.value_flag("--threads", v)) {
            a.threads = parse_thread_count(v);
        } else if (c.value_flag("--solution-dir", v)) {
            a.solution_dir = v;
        } else if (c.value_flag("--commit", v)) {
            a.commit_sha = v;
        } else if (s == "--help" || s == "-h") {
            print_usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown argument: %s\n", s.c_str());
            print_usage();
            std::exit(2);
        }
    }
    return a;
}

// Records the incumbent objective against wall time: the step function the
// Primal Integral integrates. Keys on a finite objective ("an incumbent
// exists"), not on `p.feasible` (the current point); pinned by
// tests/test_mipfeas.cpp. Only strict improvements are written: the ~1 Hz
// no-new-best rows would be the bulk of the file at 233 instances x 600s.
class TraceRecorder : public cbls::SolveCallback {
public:
    explicit TraceRecorder(std::ofstream& out) : out_(out) {
        // Round-trip precision, so the trace's last value matches the result
        // file's objective in the same comparison row (6 digits would not).
        out_ << std::setprecision(17);
    }

    void on_progress(const cbls::SolveProgress& p) override {
        if (!std::isfinite(p.objective) || p.objective >= last_written_) {
            return;
        }
        last_written_ = p.objective;
        ++n_points_;
        // Buffered, not flushed per row: an interrupted run keeps only the
        // prefix the stream has already written out.
        out_ << p.time_seconds << "," << p.objective << '\n';
    }

    /// Incumbents recorded. Feeds `trace_source`, so a scorer tells a real
    /// anytime profile from one invented from the final objective (a silent
    /// callback would otherwise read as a bad search).
    [[nodiscard]] long n_points() const { return n_points_; }

private:
    std::ofstream& out_;
    double last_written_ = std::numeric_limits<double>::infinity();
    long n_points_ = 0;
};

int count_int_vars(const cbls::MpsProblem& prob) {
    int n = 0;
    for (const auto& v : prob.vars) {
        if (v.kind != cbls::MpsVarKind::Continuous) {
            ++n;
        }
    }
    return n;
}

// Columns the MPS leaves unbounded on a side, which CBLS must bound (after
// propagation: `n_clamped_bounds`). Recorded so "both engines solved the same
// program" is checkable.
int count_unbounded_columns(const cbls::MpsProblem& prob) {
    int n = 0;
    for (const auto& v : prob.vars) {
        if (cbls::is_unbounded_below(v.lb) || cbls::is_unbounded_above(v.ub)) {
            ++n;
        }
    }
    return n;
}

/// Writes the solution vector in the MIPLIB-style format verify_solution.py
/// reads: `=obj= <value>` followed by one `<name> <value>` line per MPS column,
/// at full round-trip precision.
///
/// Keyed by name, not position, so an independent reader catches a mis-ordered
/// column. Returns false (having said why) when unwritable; the caller then
/// publishes no objective: an unverifiable row carries no number (#138).
/// `values` is one per MPS column: `MpsToModelResult::original_values`.
bool write_solution_to(const std::string& tmp_path, const Args& args, const cbls::MpsProblem& prob,
                       const std::vector<double>& values, double objective) {
    if (values.size() != prob.vars.size()) {
        std::fprintf(stderr, "%s: %zu values for %zu columns; no solution written\n",
                     args.instance.c_str(), values.size(), prob.vars.size());
        return false;
    }
    {
        std::ofstream out(tmp_path);
        if (!out.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", tmp_path.c_str());
            return false;
        }
        // 17 digits: a rounded value can violate a row the true one satisfies.
        out << std::setprecision(17);
        out << "# instance " << args.instance << "\n# engine cbls\n";
        out << "=obj= " << objective << '\n';
        for (size_t i = 0; i < prob.vars.size(); ++i) {
            const std::string& name = prob.vars[i].name;
            // Refuse only what the format cannot carry: whitespace, a leading
            // '#' (comment line), `=obj=`. An inner '#' is fine (MIPLIB `x#1#1`).
            if (name.empty() || name.find_first_of(" \t") != std::string::npos ||
                name.front() == '#' || name == "=obj=") {
                std::fprintf(stderr,
                             "%s: column name '%s' is not representable in the solution "
                             "format; no solution written\n",
                             args.instance.c_str(), name.c_str());
                return false;
            }
            out << name << ' ' << values[i] << '\n';
        }
        if (!out.good()) {
            std::fprintf(stderr, "Failed while writing %s\n", tmp_path.c_str());
            return false;
        }
    }
    return true;
}

/// Writes the solution through a temp file and rename, leaving no partial file
/// on any failure path: a truncated solution would verify as infeasible.
bool write_solution(const Args& args, const cbls::MpsProblem& prob,
                    const std::vector<double>& values, double objective) {
    const std::string path = args.solution_dir + "/" + args.instance + ".sol";
    const std::string tmp_path = path + ".tmp";
    std::error_code ec;
    if (!write_solution_to(tmp_path, args, prob, values, objective)) {
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    if (!cbls::bench::rename_into_place(tmp_path, path)) {
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

void write_result(const Args& args, const nlohmann::json& extra) {
    nlohmann::json j = extra;
    j["engine"] = "cbls";
    j["instance"] = args.instance;
    j["peak_rss_kib"] = peak_rss_kib();
    j["budget_seconds"] = args.budget;
    j["seed"] = args.seed;
    j["feasibility_tolerance"] = args.feas_tol;
    j["compound_moves"] = args.compound_moves;
    j["objective_tie_break"] = args.objective_tie_break;
    j["inf_clamp"] = args.inf_clamp;
    j["propagate_bounds"] = args.propagate_bounds;
    j["max_propagation_passes"] = args.max_propagation_passes;
    j["threads"] = args.threads;
    j["share_objective_bound"] = args.share_objective_bound;
    j["lns"] = args.search.lns;
    j["lns_interval"] = args.search.lns_interval;
    j["novelty_prob"] = args.search.novelty_prob;
    j["unproductive_iters"] = args.search.unproductive_iters;
    j["perturbation_period"] = args.search.perturbation_period;
    j["commit_sha"] = args.commit_sha;

    // Atomic: the driver resumes on file existence without revalidating, so a
    // truncated result would be permanent.
    const std::string path = args.out_dir + "/" + args.instance + ".json";
    if (!cbls::bench::write_atomically(path, j.dump(2) + "\n")) {
        std::exit(2);
    }
}

/// 0 when the arguments name a runnable run, else the exit code. The parse
/// layer's other half: it turns parse_double's NaN into the exit code the
/// driver's tests pin.
int validate_args(const Args& args) {
    if (args.instance.empty() || args.out_dir.empty()) {
        std::fprintf(stderr, "--instance and --out-dir are required\n");
        print_usage();
        return 2;
    }
    // Load-bearing: catches parse_double's NaN with the pinned message. Without
    // it solve() does nothing and a typo scores a whole roster "no_solution"
    // (Primal Integral 2.0) at exit 0, which resume treats as done.
    if (!(args.budget > 0.0)) {
        std::fprintf(stderr, "--budget must be a positive number of seconds\n");
        return 2;
    }
    if (!(args.feas_tol > 0.0)) {
        std::fprintf(stderr, "--feas-tol must be positive\n");
        return 2;
    }
    // Same hazard and NaN contract as --budget: a clamp of 0 collapses every
    // column to [0, 0].
    if (!(args.inf_clamp > 0.0)) {
        std::fprintf(stderr, "--inf-clamp must be a positive bound\n");
        return 2;
    }
    // Shared ranges and no-effect refusals (#267). This runner's own
    // compound-moves flag is copied in for the --novelty-prob rule; no
    // `--time-limit` exists here.
    cbls::bench::SearchFlags search = args.search;
    search.compound_moves = args.compound_moves;
    std::string error;
    if (!cbls::bench::validate_search_flags(search, kSharedSearchFamilies,
                                            /*time_limit_set=*/false, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    return 0;
}

/// The re-check's status: a publication policy deciding whether a row may carry
/// an objective.
struct Verdict {
    const char* status = "no_solution";
    bool have_solution = false;
    int n_fractional_int = 0;
    double obj_drift = 0.0;
};

/// Independent re-checks of the returned point (solve() restores best_state and
/// full-evaluates, so the model holds it), as minlplib.cpp does:
///   * residual: the engine's verdict, on its own DAG;
///   * integrality: a fractional Int is no MIP solution;
///   * objective drift: the published number must be the model's value at the
///     returned point, not the running best.
Verdict assess_result(const cbls::MpsToModelResult& built, const cbls::SearchResult& result,
                      double feas_tol) {
    Verdict v;
    if (result.feasible) {
        for (const auto& var : built.model.variables()) {
            if (var.type == cbls::VarType::Int &&
                std::abs(var.value - std::round(var.value)) > 1e-9) {
                ++v.n_fractional_int;
            }
        }
    }
    const double model_obj = built.objective_node_id >= 0
                                 ? built.model.node_value(built.objective_node_id)
                                 : result.objective;
    // Finite objectives only: a non-finite one (#100) gives NaN drift, which
    // would mislabel it `violation_mismatch`; `have_solution` refuses it as
    // `no_solution`, matching minlplib.
    v.obj_drift = result.feasible && std::isfinite(result.objective)
                      ? std::abs(model_obj - result.objective)
                      : 0.0;
    const bool verdict_consistent =
        !result.feasible || (result.best_violation <= feas_tol && v.n_fractional_int == 0 &&
                             v.obj_drift <= 1e-6 * (std::abs(result.objective) + 1.0));
    v.have_solution = result.feasible && std::isfinite(result.objective) && verdict_consistent;
    if (!verdict_consistent) {
        v.status = "violation_mismatch";
    } else if (v.have_solution) {
        v.status = "feasible";
    }
    return v;
}

/// The point as the MPS columns see it, checked against the parsed file before
/// anything is written (#325): assess_result judged the model the search ran
/// on, and this check shares nothing with that build but the parsed file. A
/// point that fails is withdrawn as `original_mismatch`. `original` receives the
/// point; it stays empty when there was none to check.
cbls::mipfeas::OriginalCheck check_against_file(const Args& args, const cbls::MpsProblem& prob,
                                                const cbls::MpsToModelResult& built,
                                                const cbls::SearchResult& result, Verdict& verdict,
                                                std::vector<double>& original) {
    cbls::mipfeas::OriginalCheck check;
    if (!verdict.have_solution) {
        return check;
    }
    try {
        original = built.original_values(built.model);
        check =
            cbls::mipfeas::check_original_point(prob, original, result.objective, args.feas_tol);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: point read-back failed: %s\n", args.instance.c_str(), e.what());
        check.size_mismatch = true;
    }
    if (!check.passes(result.objective)) {
        verdict.have_solution = false;
        verdict.status = "original_mismatch";
    }
    return check;
}

/// The portfolio arm of the solve: N workers over one shared structure, plus
/// the reconciliation the single-threaded path gets for free.
cbls::SearchResult solve_portfolio(const Args& args, cbls::MpsToModelResult& built,
                                   const cbls::SearchConfig& cfg,
                                   std::vector<cbls::Model>& replicas,
                                   cbls::SolveCallback* recorder) {
    std::atomic<size_t> next_replica{0};
    // Called once per worker on its own thread, hence atomic. Hands over a
    // pre-built replica by move; the copy past the end is unreachable under
    // ParallelSearch's contract and guards against an out-of-bounds read.
    auto model_factory = [&replicas, &next_replica, &built]() -> cbls::Model {
        const size_t i = next_replica.fetch_add(1, std::memory_order_relaxed);
        if (i < replicas.size()) {
            return std::move(replicas[i]);
        }
        return built.model;
    };
    // One hook and one LNS per worker: the search locks neither, so sharing is
    // a data race (include/cbls/pool.h).
    auto hook_factory = [](cbls::Model&) -> std::shared_ptr<cbls::InnerSolverHook> {
        return std::make_shared<cbls::FloatIntensifyHook>();
    };
    // Empty under --no-lns: ParallelSearch then gives no worker an LNS, as
    // lns_argument() gives the single-threaded solve() a null one.
    std::function<std::shared_ptr<cbls::LNS>()> lns_factory;
    if (args.search.lns) {
        lns_factory = []() -> std::shared_ptr<cbls::LNS> {
            return std::make_shared<cbls::LNS>(kLnsFraction);
        };
    }
    cbls::ParallelConfig par_config;
    par_config.n_threads = args.threads;
    par_config.share_objective_bound = args.share_objective_bound;
    cbls::ParallelSearch ps(args.threads);
    // The factory overload, so the pre-built replicas' cost lands in
    // `setup_seconds`, not the search deadline; they share one DAG either way
    // (#157).
    cbls::SearchResult result = ps.solve(model_factory, args.budget, args.seed, cfg, hook_factory,
                                         lns_factory, recorder, par_config);
    // The master still holds its initial point: restore the returned one and
    // re-evaluate, as a single-threaded solve() leaves it. No
    // `set_objective_bound` (unlike ViolationLSLoop::finish()): no search ran on
    // the master, so its per-model bound is still +inf.
    // Guarded: an empty `best_state` is reachable (the time limit expired during
    // thread creation), and restore_state's size-mismatch throw would record
    // "did no work" as `solve_error`.
    if (result.best_state.values.size() == built.model.num_vars()) {
        built.model.restore_state(result.best_state);
        cbls::full_evaluate(built.model);
    }
    return result;
}

int run_benchmark(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (const int rc = validate_args(args); rc != 0) {
        return rc;
    }

    std::error_code ec;
    std::filesystem::create_directories(args.out_dir, ec);
    if (!args.solution_dir.empty()) {
        // Before the solve: an unwritable --solution-dir must fail in a
        // millisecond, not after a 600s run.
        std::error_code sol_ec;
        std::filesystem::create_directories(args.solution_dir, sol_ec);
        if (sol_ec && !std::filesystem::is_directory(args.solution_dir)) {
            std::fprintf(stderr, "Cannot create --solution-dir %s: %s\n", args.solution_dir.c_str(),
                         sol_ec.message().c_str());
            return 2;
        }
    }

    const std::string mps_path = args.inst_dir + "/" + args.instance + ".mps.gz";
    if (!file_exists(mps_path)) {
        // No result file: an absent instance is an incomplete run, not a zero score.
        std::fprintf(stderr,
                     "%s not found. Fetch the roster first:\n"
                     "  python %s/download.py\n",
                     mps_path.c_str(), args.inst_dir.c_str());
        return 2;
    }

    // Read and build are timed apart from the solve bracket (`wall_seconds`):
    // the largest models spend minutes there. Reported, never subtracted:
    // differenced whole-program timings are not a measurement (CLAUDE.md).
    const auto t_read = std::chrono::steady_clock::now();
    cbls::MpsProblem prob;
    try {
        prob = cbls::read_mps(mps_path);
    } catch (const std::exception& e) {
        write_result(args, {{"status", "read_error"},
                            {"message", e.what()},
                            {"read_seconds", seconds_since(t_read)}});
        std::fprintf(stderr, "%s: read error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }
    const double read_seconds = seconds_since(t_read);

    cbls::MpsToModelOptions mps_opts;
    mps_opts.inf_clamp = args.inf_clamp;
    mps_opts.propagate_bounds = args.propagate_bounds;
    mps_opts.max_propagation_passes = args.max_propagation_passes;
    const auto t_build = std::chrono::steady_clock::now();
    cbls::MpsToModelResult built;
    try {
        built = cbls::mps_to_model(prob, mps_opts);
    } catch (const std::exception& e) {
        write_result(args, {{"status", "build_error"},
                            {"message", e.what()},
                            {"n_vars", prob.vars.size()},
                            {"n_cons", prob.rows.size()},
                            {"n_int_vars", count_int_vars(prob)},
                            {"read_seconds", read_seconds},
                            {"build_seconds", seconds_since(t_build)}});
        std::fprintf(stderr, "%s: build error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }
    // Bound propagation runs inside mps_to_model, so it is inside this number.
    const double build_seconds = seconds_since(t_build);
    // Sampled before freeze(), which adds the two-node `obj <= bound` row at
    // every thread count (#157, #271), so this is the DAG the reader built.
    const std::size_t n_nodes_built = built.model.num_nodes();

    const std::string trace_path = args.out_dir + "/" + args.instance + ".trace.csv";
    std::ofstream trace(trace_path);
    if (!trace.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", trace_path.c_str());
        return 2;
    }
    trace << "time_seconds,objective\n";

    cbls::SearchConfig cfg;
    cfg.feasibility_tolerance = args.feas_tol;
    cfg.use_compound_moves = args.compound_moves;
    cfg.fj_objective_tie_break = args.objective_tie_break;
    // The kSharedSearchFlags arms (#267), field by field: apply_search_flags()
    // would overwrite this runner's own flags above. validate_args() range-checked
    // the narrowed ones. `cfg.lns_interval` feeds both solve paths.
    cfg.lns_interval = cbls::bench::lns_interval_argument(args.search);
    cfg.novelty_jump_probability = args.search.novelty_prob;
    cfg.unproductive_iterations = args.search.unproductive_iters;
    cfg.perturbation_period = static_cast<int>(args.search.perturbation_period);

    // One model per portfolio worker, replicated before the solve bracket
    // (replicas share the immutable structure, #157), because:
    //  - the copies' cost belongs in `setup_seconds`, not the search budget;
    //  - an over-sized `--threads` must fail here under the driver's `ulimit -v`,
    //    in a millisecond as `replicate_error`, not on a worker thread, where it
    //    costs the whole budget for a row the scorer refuses (#170).
    // No cost at the peak: the solve holds this plus per-worker tables, and sets
    // `peak_rss_kib`.
    std::vector<cbls::Model> replicas;
    const auto t_replicate = std::chrono::steady_clock::now();
    // Caught and recorded: a job dying without a result is indistinguishable
    // to the driver's resume from one never run.
    try {
        // Frozen here at every thread count (objective-row rebuild plus cone
        // cache, #271): the copies share it, and a 1-thread row's search time
        // must not carry a cost an N-thread row reports as setup.
        built.model.freeze(cfg.cone_cache_bytes);
        if (args.threads > 1) {
            replicas.reserve(static_cast<size_t>(args.threads));
            for (int i = 0; i < args.threads; ++i) {
                replicas.push_back(built.model);
            }
        }
    } catch (const std::exception& e) {
        write_result(args, {{"status", "replicate_error"},
                            {"message", e.what()},
                            {"read_seconds", read_seconds},
                            {"build_seconds", build_seconds},
                            {"replicate_seconds", seconds_since(t_replicate)}});
        std::fprintf(stderr, "%s: replicate error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }
    const double replicate_seconds = seconds_since(t_replicate);

    const auto t0 = std::chrono::steady_clock::now();
    cbls::SearchResult result;
    long trace_points = 0;
    try {
        TraceRecorder recorder(trace);
        if (args.threads > 1) {
            result = solve_portfolio(args, built, cfg, replicas, &recorder);
        } else {
            cbls::FloatIntensifyHook hook;
            cbls::LNS lns(kLnsFraction);
            result = cbls::solve(built.model, args.budget, args.seed, &hook,
                                 cbls::bench::lns_argument(args.search, lns), cfg.lns_interval,
                                 &recorder, cfg);
        }
        trace_points = recorder.n_points();
    } catch (const std::exception& e) {
        write_result(args, {{"status", "solve_error"},
                            {"message", e.what()},
                            {"read_seconds", read_seconds},
                            {"build_seconds", build_seconds}});
        std::fprintf(stderr, "%s: solve error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }
    const double wall = seconds_since(t0);

    Verdict verdict = assess_result(built, result, args.feas_tol);
    std::vector<double> original;
    const cbls::mipfeas::OriginalCheck original_check =
        check_against_file(args, prob, built, result, verdict, original);
    // Solution before result: the driver resumes on the result file and
    // verifies the solution beside it, so the reverse order opens a window.
    bool solution_write_failed = false;
    if (verdict.have_solution && !args.solution_dir.empty() &&
        !write_solution(args, prob, original, result.objective)) {
        // No solution file, no independent verdict, no number (#138).
        verdict.have_solution = false;
        verdict.status = "solution_write_error";
        solution_write_failed = true;
    }
    nlohmann::json j{
        {"status", verdict.status},
        // The solve bracket only; published tables are scored against it.
        {"wall_seconds", wall},
        {"read_seconds", read_seconds},
        {"build_seconds", build_seconds},
        // DAG size, what evaluation cost scales with: a nonzero costs 0 to 2
        // nodes by coefficient (src/io/mps_to_model.cpp), so the ratio varies
        // (square47: 125k nodes from a matrix ten times larger).
        {"n_nodes", n_nodes_built},
        {"n_nonzeros", prob.nonzeros.size()},
        // Includes the freeze at every thread count, so not zero at 1 thread;
        // grows with --threads (per-worker mutable side, #157).
        {"replicate_seconds", replicate_seconds},
        {"setup_seconds", read_seconds + build_seconds + replicate_seconds},
        // What the search did with its budget: tells "never diversified" from
        // "diversified and could not escape" (binkar10_1: last improvement at
        // 5.1s of 60s, 150x above optimum).
        {"perturbations", result.perturbations},
        {"lns_repairs", result.lns_repairs},
        {"lns_repairs_accepted", result.lns_repairs_accepted},
        // Batch split and Novelty activity (#209), summed over workers: a
        // Novelty batch charges `iterations` 3 per move and 1 per bump, so
        // `iterations` alone cannot separate Novelty from FJ work.
        {"fj_batches", result.counters.fj_batches},
        {"novelty_batches", result.counters.novelty_batches},
        {"novelty_moves", result.counters.novelty_moves},
        {"novelty_commits", result.counters.novelty_commits},
        {"novelty_weight_bumps", result.counters.novelty_weight_bumps},
        // Shared-bound engagement (#179), summed over workers: a null A/B vs a
        // mechanism that never fired. Zero at --threads 1. With sharing off,
        // `bound_behind_global_*` is the opportunity it would take.
        {"portfolio_restarts", result.counters.portfolio_restarts},
        {"shared_bound_tightenings", result.counters.shared_bound_tightenings},
        {"own_best_behind_global", result.counters.own_best_behind_global},
        {"bound_behind_global_batches", result.counters.bound_behind_global_batches},
        {"bound_behind_global_seconds", result.counters.bound_behind_global_seconds},
        // null, not NaN (JSON) or 0 ("instant"), when never feasible.
        {"time_to_first_feasible", std::isfinite(result.time_to_first_feasible)
                                       ? nlohmann::json(result.time_to_first_feasible)
                                       : nlohmann::json(nullptr)},
        {"first_feasible_objective", std::isfinite(result.first_feasible_objective)
                                         ? nlohmann::json(result.first_feasible_objective)
                                         : nlohmann::json(nullptr)},
        // Mirrors cpsat_solve.py: `callback` is a real anytime profile,
        // `final_only` a scorer-invented end point (a harness condition). The
        // scorer keys on `trace_source`; the count is for reading by hand.
        {"trace_points", trace_points},
        {"trace_source", trace_points > 0 ? "callback" : "final_only"},
        {"iterations", result.iterations},
        {"max_violation", result.best_violation},
        {"n_fractional_int", verdict.n_fractional_int},
        {"objective_drift", verdict.obj_drift},
        {"n_vars", prob.vars.size()},
        {"n_cons", prob.rows.size()},
        {"n_int_vars", count_int_vars(prob)},
        {"n_unbounded_columns", count_unbounded_columns(prob)},
        {"n_clamped_bounds", built.n_clamped_columns},
        {"n_bounds_tightened", built.bound_stats.n_tightened},
        {"n_bounds_finitized", built.bound_stats.n_finitized},
        {"n_bounds_fixed", built.bound_stats.n_fixed},
        {"bound_propagation_passes", built.bound_stats.passes},
        // The verdict too: on infeasible the adapter zeroes the counts, which
        // would read like "propagation found nothing".
        {"bound_propagation_infeasible", built.bound_stats.infeasible},
        {"bound_propagation_hit_pass_limit", built.bound_stats.hit_pass_limit},
        // The point's check against the file (#325), as ratios to the
        // tolerance: above 1 fails. All zero when there was no point to check.
        {"original_max_row_ratio", original_check.max_row_ratio},
        {"original_max_bound_ratio", original_check.max_bound_ratio},
        {"original_max_integrality", original_check.max_integrality},
        {"original_objective_drift", original_check.objective_drift},
        // Why an `original_mismatch` with zero ratios was withdrawn: a
        // wrong-sized point (or failed read-back), or a non-finite value or
        // row activity.
        {"original_size_mismatch", original_check.size_mismatch},
        {"original_non_finite", original_check.non_finite},
    };
    j["objective"] =
        verdict.have_solution ? nlohmann::json(result.objective) : nlohmann::json(nullptr);
    // How many portfolio workers ran to the end (#170); see worker_accounting.h.
    cbls::mipfeas::add_worker_accounting(result, j);
    write_result(args, j);
    const std::string lost =
        cbls::mipfeas::lost_workers_message(args.instance, args.threads, result);
    std::fputs(lost.c_str(), stderr);

    std::printf("%-28s %-12s obj=%-16.8g viol=%-10.3g %8.2fs\n", args.instance.c_str(),
                verdict.status,
                verdict.have_solution ? result.objective : std::numeric_limits<double>::quiet_NaN(),
                result.best_violation, wall);
    return cbls::mipfeas::runner_exit_code(solution_write_failed,
                                           cbls::mipfeas::lost_workers(result, args.threads));
}

}  // namespace

int main(int argc, char** argv) {
    // An escaping exception is a silent std::terminate; report it and exit 1
    // instead (bugprone-exception-escape).
    try {
        return run_benchmark(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
