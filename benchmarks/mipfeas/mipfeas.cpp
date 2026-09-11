// MIPfeas benchmark runner (CBLS side).
//
// Runs ONE instance per process: the driver (run_benchmark.py) parallelises,
// caps memory and resumes across invocations, and a single instance dying takes
// its own process with it rather than the whole run.
//
// Writes two files per instance into --out-dir:
//   <instance>.json       result record (schema shared with cpsat_solve.py)
//   <instance>.trace.csv  incumbent objective vs wall time, the input to the
//                         Primal Integral (primal_integral.py)
//   <instance>.sol        the solution vector, when --solution-dir is given and
//                         the run reports a feasible solution. Written for
//                         verify_solution.py to check against the ORIGINAL
//                         instance file with a third-party reader (#138) --
//                         every other check this runner performs is downstream
//                         of the MPS-to-model adapter and so cannot see an
//                         adapter defect at all.
//
// Deliberately refuses to write anything when the instance file is absent: a
// missing instance must not be scored as "found nothing" (see issue #103, where
// a runner emptied a published table by skipping every absent instance).

#include <benchmarks/common/runner_args.h>
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
#include <iomanip>
#include <limits>
#include <nlohmann/json.hpp>
#include <string>
#include <sys/resource.h>

namespace {

struct Args {
    std::string instance;
    std::string inst_dir = "benchmarks/instances/mipfeas";
    std::string out_dir;
    // MIPfeas scores the Primal Integral over a 600s budget.
    double budget = 600.0;
    uint64_t seed = 42;
    // Stated explicitly rather than inherited from the engine default: a published
    // result must not silently change when an engine default moves (issue #103).
    double feas_tol = cbls::kDefaultFeasibilityTolerance;
    // Novelty Jump (compound moves), on for this benchmark although the engine
    // default is off. That default exists because the per-batch cost was not
    // bounded tightly enough for the large *continuous* benchmarks, which is not
    // this roster; and roughly half of CP-SAT's incumbents here come from its own
    // compound-move subsolvers (`ls_restart_*compound*` — 45-67% of improving
    // solutions on binkar10_1 and pk1). Running without it would compare our
    // Feasibility Jump against their Feasibility Jump plus Novelty Jump and call
    // the difference a reimplementation gap.
    bool compound_moves = true;
    // CBLS variables need finite bounds. Implied bounds supply most of them
    // (#120); this is the fallback for a column no constraint bounds. 1e7 rather
    // than the engine's 1e9 because it measured better on the smoke roster — NOT
    // because it matches CP-SAT, which does not truncate variable domains at all
    // (`mip_max_bound` is not a domain clamp: an integer column bounded at 1e12 is
    // solved to 1e12). This is a CBLS-side restriction, so `n_clamped_bounds`
    // records how many columns it still narrows and the comparison table
    // publishes it next to `n_unbounded_columns`, the exposure before propagation.
    double inf_clamp = 1.0e7;
    // Derive implied bounds from the rows first, so the clamp above is reached
    // only on columns no constraint bounds. Off disables *propagation* only —
    // #120's other half, honouring a finite bound however wide, is
    // unconditional — so this is an A/B on propagation, not on the old engine.
    bool propagate_bounds = true;
    // Changes the derived box, so it is recorded per result like every other
    // setting that does.
    int max_propagation_passes = 10;
    // Where to write the solution vector of a feasible run, for independent
    // verification against the original instance file (#138). Empty means "do
    // not write one" -- the driver passes it, a bare invocation need not.
    std::string solution_dir;
    std::string commit_sha = "unknown";
};

void print_usage() {
    std::printf(
        "Usage: cbls_mipfeas --instance NAME --out-dir DIR [--inst-dir DIR]\n"
        "                    [--budget SECONDS] [--seed N] [--feas-tol T]\n"
        "                    [--inf-clamp B] [--no-propagate-bounds]\n"
        "                    [--max-propagation-passes N]\n"
        "                    [--no-compound-moves] [--solution-dir DIR]\n"
        "                    [--commit SHA]\n");
}

// Flag-value parsing lives in benchmarks/common/runner_args.h, shared with the
// other runners so that a fix to one cannot silently diverge from the rest. The
// reporting policy is documented there and is unchanged: a bad double is
// reported and returned as NaN for the positivity guards below to turn into
// exit 2, and a bad integer is reported and exits 2 directly.
using cbls::bench::parse_double;
using cbls::bench::parse_int64;

/// Range-checked rather than cast: parse_int64 validates the syntax, but an
/// out-of-range value would wrap to a small or negative pass count and silently
/// disable propagation -- a different derived box, published at exit code 0.
int parse_propagation_passes(const char* text) {
    const int64_t passes = parse_int64("--max-propagation-passes", text);
    if (passes < 0 || passes > std::numeric_limits<int>::max()) {
        std::fprintf(stderr, "--max-propagation-passes must be in [0, %d]\n",
                     std::numeric_limits<int>::max());
        std::exit(2);
    }
    return static_cast<int>(passes);
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
        } else if (s == "--compound-moves") {
            a.compound_moves = true;
        } else if (s == "--no-compound-moves") {
            a.compound_moves = false;
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

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

// Records the incumbent objective against wall time — the step function the
// Primal Integral integrates.
//
// Filters on a finite objective rather than on `p.feasible`: SolveProgress
// carries `best_feasible_obj`, which is +inf until a real-feasible solution has
// been recorded, whereas `p.feasible` reports whether the *current* assignment
// is feasible. The two agree on the rows this recorder keeps — solve() only
// emits a new best from a feasible point — so this is the more direct statement
// of the invariant the step function needs ("an incumbent exists"), not a
// correction of a bug. tests/test_mipfeas.cpp pins that invariant down.
//
// Only strict improvements are written. solve() also emits progress roughly once
// a second with no new best; those rows repeat a value the step function already
// holds, and at 233 instances x 600s the repetition is the bulk of the file.
class TraceRecorder : public cbls::SolveCallback {
public:
    explicit TraceRecorder(std::ofstream& out) : out_(out) {
        // Full round-trip precision. The default 6 significant digits rounds the
        // objective the Primal Integral integrates (1010195.19 -> 1.0102e+06), so
        // the trace's last value would stop matching the objective the result file
        // publishes — and the two are read into the same comparison row.
        out_ << std::setprecision(17);
    }

    void on_progress(const cbls::SolveProgress& p) override {
        if (!std::isfinite(p.objective) || p.objective >= last_written_) {
            return;
        }
        last_written_ = p.objective;
        ++n_points_;
        // Flushed per row: a run is minutes long and an interrupted one must
        // still leave a scorable prefix behind.
        out_ << p.time_seconds << "," << p.objective << '\n';
    }

    /// How many incumbents the callback actually recorded. Published as
    /// `trace_source`, which is how a scorer tells a genuine anytime profile
    /// from one the scorer had to invent from the final objective: a callback
    /// that stopped firing would otherwise score every instance near the
    /// no-solution penalty, indistinguishable from "the search is bad".
    [[nodiscard]] long n_points() const { return n_points_; }

private:
    std::ofstream& out_;
    double last_written_ = std::numeric_limits<double>::infinity();
    long n_points_ = 0;
};

// Seconds elapsed since `start`, on the monotonic clock.
double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// Peak resident set of this process, in KiB. Reported per result so the
// concurrency for a full-roster run can be sized from measurement rather than
// guessed: the roster spans models from tens of KB to millions of nonzeros.
long peak_rss_kib() {
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0;
    }
    return usage.ru_maxrss;
}

int count_int_vars(const cbls::MpsProblem& prob) {
    int n = 0;
    for (const auto& v : prob.vars) {
        if (v.kind != cbls::MpsVarKind::Continuous) {
            ++n;
        }
    }
    return n;
}

// Columns the MPS leaves unbounded on at least one side. Every one of these is
// a column CBLS would have had to invent a bound for; how many it still has to
// after propagation is `n_clamped_bounds` below. Recorded per result because
// "the two engines solved the same program" is otherwise an assumption a reader
// cannot check.
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
/// Keyed by the MPS column name rather than by position, because the point of
/// the file is to be re-read against the ORIGINAL instance by a reader that
/// shares no code with this one: a mis-ordered column would otherwise verify
/// clean. Returns false (having said why) when the file cannot be written, and
/// the caller then publishes no objective -- an unverifiable row must not carry
/// a number, which is the whole of issue #138.
bool write_solution_to(const std::string& tmp_path, const Args& args, const cbls::MpsProblem& prob,
                       const cbls::MpsToModelResult& built, double objective) {
    if (built.var_handles.size() != prob.vars.size()) {
        std::fprintf(stderr, "%s: %zu handles for %zu columns; no solution written\n",
                     args.instance.c_str(), built.var_handles.size(), prob.vars.size());
        return false;
    }
    {
        std::ofstream out(tmp_path);
        if (!out.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", tmp_path.c_str());
            return false;
        }
        // 17 significant digits: a rounded value can violate a row the true one
        // satisfies, which would read as an engine defect rather than as a
        // lossy dump.
        out << std::setprecision(17);
        out << "# instance " << args.instance << "\n# engine cbls\n";
        out << "=obj= " << objective << '\n';
        const auto& vars = built.model.variables();
        for (size_t i = 0; i < prob.vars.size(); ++i) {
            const std::string& name = prob.vars[i].name;
            // Only a name the format genuinely cannot carry is refused: the
            // lines are whitespace-separated, a whole line starting with '#' is
            // a comment, and `=obj=` is the objective sentinel. A '#' *inside* a
            // name is fine and common in MIPLIB (`x#1#1`), which is why the
            // reader treats comments as whole lines.
            if (name.empty() || name.find_first_of(" \t") != std::string::npos ||
                name.front() == '#' || name == "=obj=") {
                std::fprintf(stderr,
                             "%s: column name '%s' is not representable in the solution "
                             "format; no solution written\n",
                             args.instance.c_str(), name.c_str());
                return false;
            }
            const int32_t var_id = cbls::handle_to_var_id(built.var_handles[i]);
            if (var_id < 0 || var_id >= static_cast<int32_t>(vars.size())) {
                std::fprintf(stderr, "%s: column %s has no variable\n", args.instance.c_str(),
                             name.c_str());
                return false;
            }
            out << name << ' ' << vars[static_cast<size_t>(var_id)].value << '\n';
        }
        if (!out.good()) {
            std::fprintf(stderr, "Failed while writing %s\n", tmp_path.c_str());
            return false;
        }
    }
    return true;
}

/// Writes the solution and renames it into place, leaving no partial file behind
/// on any failure path. Renamed for the same reason the result file is: the
/// driver reads this file only after the result appears, and a truncated
/// solution would verify as an infeasible one.
bool write_solution(const Args& args, const cbls::MpsProblem& prob,
                    const cbls::MpsToModelResult& built, double objective) {
    const std::string path = args.solution_dir + "/" + args.instance + ".sol";
    const std::string tmp_path = path + ".tmp";
    std::error_code ec;
    if (!write_solution_to(tmp_path, args, prob, built, objective)) {
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    std::filesystem::rename(tmp_path, path, ec);
    if (ec) {
        std::fprintf(stderr, "Failed to rename %s -> %s: %s\n", tmp_path.c_str(), path.c_str(),
                     ec.message().c_str());
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
    j["inf_clamp"] = args.inf_clamp;
    j["propagate_bounds"] = args.propagate_bounds;
    j["max_propagation_passes"] = args.max_propagation_passes;
    j["commit_sha"] = args.commit_sha;

    // Write-then-rename: a job killed mid-write must leave either the previous
    // result or none, never a truncated one. The driver resumes on file existence
    // and does not revalidate, so a half-written result is otherwise permanent.
    const std::string path = args.out_dir + "/" + args.instance + ".json";
    const std::string tmp_path = path + ".tmp";
    {
        std::ofstream out(tmp_path);
        if (!out.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", tmp_path.c_str());
            std::exit(2);
        }
        out << j.dump(2) << "\n";
    }
    std::error_code rename_ec;
    std::filesystem::rename(tmp_path, path, rename_ec);
    if (rename_ec) {
        std::fprintf(stderr, "Failed to rename %s -> %s: %s\n", tmp_path.c_str(), path.c_str(),
                     rename_ec.message().c_str());
        std::exit(2);
    }
}

/// Whether the arguments name a run that can produce a result at all. Returns 0
/// when they do, otherwise the process exit code. Separate from run_benchmark
/// because these guards are the parse layer's other half rather than part of
/// running: parse_double reports a bad --budget and hands back NaN, and it is
/// this function that turns that into an exit code the driver's tests pin.
int validate_args(const Args& args) {
    if (args.instance.empty() || args.out_dir.empty()) {
        std::fprintf(stderr, "--instance and --out-dir are required\n");
        print_usage();
        return 2;
    }
    // This guard is load-bearing, not defensive: parse_double reports a bad
    // --budget and returns NaN, and NaN is rejected here rather than at the
    // parse, so that the exit code and message stay the ones the driver's tests
    // pin. solve() with a non-positive time limit and no iteration budget
    // returns having done nothing, so unchecked, one typo'd flag scores an
    // entire roster "no_solution" (Primal Integral 2.0) at exit code 0, and the
    // driver's resume then treats that as work completed.
    if (!(args.budget > 0.0)) {
        std::fprintf(stderr, "--budget must be a positive number of seconds\n");
        return 2;
    }
    if (!(args.feas_tol > 0.0)) {
        std::fprintf(stderr, "--feas-tol must be positive\n");
        return 2;
    }
    // Same hazard as --budget, and the same NaN-reaches-this-guard contract: a
    // clamp of 0 collapses every column to [0, 0] rather than erroring — so one
    // typo'd flag scores an entire roster "no_solution" at exit code 0, which
    // resume then treats as work completed.
    if (!(args.inf_clamp > 0.0)) {
        std::fprintf(stderr, "--inf-clamp must be a positive bound\n");
        return 2;
    }
    return 0;
}

/// The independent re-check of the assignment solve() returned, and the status
/// it earns. Its own function because it is a publication policy, not part of
/// running the search: it decides whether a row may carry an objective at all.
struct Verdict {
    const char* status = "no_solution";
    bool have_solution = false;
    int n_fractional_int = 0;
    double obj_drift = 0.0;
};

/// Independent re-checks of the assignment solve() actually returned. It
/// restores best_state and full-evaluates before returning, so the model holds
/// that point now. Mirrors benchmarks/minlplib/minlplib.cpp, which already
/// refuses to publish a row failing any of these:
///
///   * residual — the engine's verdict, recomputed, on its own DAG;
///   * integrality — an Int variable left fractional means the point is not a
///     solution of the MIP at all;
///   * objective drift — result.objective is the search's *running best*, taken
///     when the incumbent was recorded. The number published has to be what the
///     model evaluates to at the point being returned.
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
                                 ? built.model.node(built.objective_node_id).value
                                 : result.objective;
    // Only meaningful for a finite objective: a feasible point on which the
    // objective is +inf/NaN (issue #100) makes this |inf - inf| = NaN, and
    // `NaN <= tol` is false — which would report a perfectly consistent verdict
    // as `violation_mismatch`. `have_solution` below already refuses such a
    // point via isfinite, so skipping the drift check just gets it the right
    // label (`no_solution`), matching minlplib's non-finite handling.
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

int run_benchmark(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (const int rc = validate_args(args); rc != 0) {
        return rc;
    }

    std::error_code ec;
    std::filesystem::create_directories(args.out_dir, ec);
    if (!args.solution_dir.empty()) {
        // Created before the solve, not after it: an unwritable --solution-dir
        // must cost a millisecond, not a 600s run whose solution then has
        // nowhere to go.
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

    // Read and build are timed separately from the solve. The solve bracket
    // below is the only thing `wall_seconds` has ever measured, so every second
    // spent parsing the MPS, lowering it into a DAG and propagating bounds was
    // invisible in the published table -- on a roster whose largest models spend
    // minutes there. Reported, never subtracted: two whole-program timings
    // differenced is not a measurement (CLAUDE.md).
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

    const std::string trace_path = args.out_dir + "/" + args.instance + ".trace.csv";
    std::ofstream trace(trace_path);
    if (!trace.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", trace_path.c_str());
        return 2;
    }
    trace << "time_seconds,objective\n";

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    cbls::SearchConfig cfg;
    cfg.feasibility_tolerance = args.feas_tol;
    cfg.use_compound_moves = args.compound_moves;

    const auto t0 = std::chrono::steady_clock::now();
    cbls::SearchResult result;
    long trace_points = 0;
    try {
        TraceRecorder recorder(trace);
        result = cbls::solve(built.model, args.budget, args.seed, /*use_fj=*/true, &hook, &lns,
                             /*lns_interval=*/3, &recorder, cfg);
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
    // The solution goes out before the result does. The driver resumes on the
    // result file's existence and verifies what it finds next to it, so a
    // result that appeared first would leave a window in which the row looks
    // complete and unverifiable at once.
    bool solution_write_failed = false;
    if (verdict.have_solution && !args.solution_dir.empty() &&
        !write_solution(args, prob, built, result.objective)) {
        // No solution file means no independent verdict, and a row with no
        // verdict must not publish a number (#138).
        verdict.have_solution = false;
        verdict.status = "solution_write_error";
        solution_write_failed = true;
    }
    nlohmann::json j{
        {"status", verdict.status},
        // The solve bracket, unchanged: `wall_seconds` has always meant this and
        // published tables are scored against it. The setup keys beside it are
        // the part that used to go unreported.
        {"wall_seconds", wall},
        {"read_seconds", read_seconds},
        {"build_seconds", build_seconds},
        {"setup_seconds", read_seconds + build_seconds},
        // Mirrors cpsat_solve.py's key of the same name. `callback` is this
        // runner's analogue of CP-SAT's log: a genuine anytime profile. Anything
        // else means the scorer had to stand in a single end point, which is a
        // harness condition rather than a search result. The scorer keys its
        // trace-health counts on `trace_source`; the raw count beside it is for
        // reading a single result by hand, where "one point" and "four thousand"
        // are the difference between a profile and a coincidence.
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
        // The verdict, not just the counts: on `infeasible` the adapter discards
        // the derived bounds and zeroes the counts, so without this a false
        // infeasibility reads exactly like "propagation found nothing".
        {"bound_propagation_infeasible", built.bound_stats.infeasible},
        {"bound_propagation_hit_pass_limit", built.bound_stats.hit_pass_limit},
    };
    j["objective"] =
        verdict.have_solution ? nlohmann::json(result.objective) : nlohmann::json(nullptr);
    write_result(args, j);

    std::printf("%-28s %-12s obj=%-16.8g viol=%-10.3g %8.2fs\n", args.instance.c_str(),
                verdict.status,
                verdict.have_solution ? result.objective : std::numeric_limits<double>::quiet_NaN(),
                result.best_violation, wall);
    // Non-zero when the solution could not be written: the job did run, but it
    // produced a row nothing can verify, and the driver has to see that.
    return solution_write_failed ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    // A benchmark run is long, and an exception escaping main is std::terminate
    // -- an abort with no message, indistinguishable from a crash. Say what
    // failed and exit non-zero instead (bugprone-exception-escape).
    try {
        return run_benchmark(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
