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
//
// Deliberately refuses to write anything when the instance file is absent: a
// missing instance must not be scored as "found nothing" (see issue #103, where
// a runner emptied a published table by skipping every absent instance).

#include <cbls/cbls.h>
#include <cbls/io_mps.h>
#include <nlohmann/json.hpp>

#include <sys/resource.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

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
    std::string commit_sha = "unknown";
};

void print_usage() {
    std::printf(
        "Usage: cbls_mipfeas --instance NAME --out-dir DIR [--inst-dir DIR]\n"
        "                    [--budget SECONDS] [--seed N] [--feas-tol T]\n"
        "                    [--commit SHA]\n");
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--instance" && i + 1 < argc) {
            a.instance = argv[++i];
        } else if (s == "--inst-dir" && i + 1 < argc) {
            a.inst_dir = argv[++i];
        } else if (s == "--out-dir" && i + 1 < argc) {
            a.out_dir = argv[++i];
        } else if (s == "--budget" && i + 1 < argc) {
            a.budget = std::atof(argv[++i]);
        } else if (s == "--seed" && i + 1 < argc) {
            a.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (s == "--feas-tol" && i + 1 < argc) {
            a.feas_tol = std::atof(argv[++i]);
        } else if (s == "--commit" && i + 1 < argc) {
            a.commit_sha = argv[++i];
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
        // Flushed per row: a run is minutes long and an interrupted one must
        // still leave a scorable prefix behind.
        out_ << p.time_seconds << "," << p.objective << std::endl;
    }

private:
    std::ofstream& out_;
    double last_written_ = std::numeric_limits<double>::infinity();
};

// Peak resident set of this process, in KiB. Reported per result so the
// concurrency for a full-roster run can be sized from measurement rather than
// guessed: the roster spans models from tens of KB to millions of nonzeros.
long peak_rss_kib() {
    struct rusage usage {};
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

void write_result(const Args& args, const nlohmann::json& extra) {
    nlohmann::json j = extra;
    j["engine"] = "cbls";
    j["instance"] = args.instance;
    j["peak_rss_kib"] = peak_rss_kib();
    j["budget_seconds"] = args.budget;
    j["seed"] = args.seed;
    j["feasibility_tolerance"] = args.feas_tol;
    j["commit_sha"] = args.commit_sha;

    const std::string path = args.out_dir + "/" + args.instance + ".json";
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", path.c_str());
        std::exit(2);
    }
    out << j.dump(2) << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (args.instance.empty() || args.out_dir.empty()) {
        std::fprintf(stderr, "--instance and --out-dir are required\n");
        print_usage();
        return 2;
    }
    // std::atof returns 0 on a parse failure, and solve() with a non-positive time
    // limit and no iteration budget returns having done nothing. Unchecked, one
    // typo'd flag scores an entire roster "no_solution" (Primal Integral 2.0) at
    // exit code 0, and the driver's resume then treats that as work completed.
    if (!(args.budget > 0.0)) {
        std::fprintf(stderr, "--budget must be a positive number of seconds\n");
        return 2;
    }
    if (!(args.feas_tol > 0.0)) {
        std::fprintf(stderr, "--feas-tol must be positive\n");
        return 2;
    }

    std::error_code ec;
    std::filesystem::create_directories(args.out_dir, ec);

    const std::string mps_path = args.inst_dir + "/" + args.instance + ".mps.gz";
    if (!file_exists(mps_path)) {
        // No result file: an absent instance is an incomplete run, not a zero score.
        std::fprintf(stderr,
                     "%s not found. Fetch the roster first:\n"
                     "  python %s/download.py\n",
                     mps_path.c_str(), args.inst_dir.c_str());
        return 2;
    }

    cbls::MpsProblem prob;
    try {
        prob = cbls::read_mps(mps_path);
    } catch (const std::exception& e) {
        write_result(args, {{"status", "read_error"}, {"message", e.what()}});
        std::fprintf(stderr, "%s: read error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }

    cbls::MpsToModelResult built;
    try {
        built = cbls::mps_to_model(prob);
    } catch (const std::exception& e) {
        write_result(args, {{"status", "build_error"},
                            {"message", e.what()},
                            {"n_vars", prob.vars.size()},
                            {"n_cons", prob.rows.size()},
                            {"n_int_vars", count_int_vars(prob)}});
        std::fprintf(stderr, "%s: build error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }

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

    const auto t0 = std::chrono::steady_clock::now();
    cbls::SearchResult result;
    try {
        TraceRecorder recorder(trace);
        result = cbls::solve(built.model, args.budget, args.seed, /*use_fj=*/true, &hook, &lns,
                             /*lns_interval=*/3, &recorder, cfg);
    } catch (const std::exception& e) {
        write_result(args, {{"status", "solve_error"}, {"message", e.what()}});
        std::fprintf(stderr, "%s: solve error: %s\n", args.instance.c_str(), e.what());
        return 1;
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Cross-check the engine's own feasibility verdict against the residual of the
    // assignment it actually returned, which solve() recomputes with a full
    // evaluate after restoring the best state. The two can only disagree if
    // incremental and full evaluation have drifted — and publishing an objective
    // for an infeasible point is the one thing this table must never contain.
    const bool verdict_consistent = !result.feasible || result.best_violation <= args.feas_tol;
    const bool have_solution =
        result.feasible && std::isfinite(result.objective) && verdict_consistent;
    const char* status = !verdict_consistent ? "violation_mismatch"
                         : have_solution     ? "feasible"
                                             : "no_solution";
    nlohmann::json j{
        {"status", status},
        {"wall_seconds", wall},
        {"iterations", result.iterations},
        {"max_violation", result.best_violation},
        {"n_vars", prob.vars.size()},
        {"n_cons", prob.rows.size()},
        {"n_int_vars", count_int_vars(prob)},
    };
    j["objective"] = have_solution ? nlohmann::json(result.objective) : nlohmann::json(nullptr);
    write_result(args, j);

    std::printf("%-28s %-12s obj=%-16.8g viol=%-10.3g %8.2fs\n", args.instance.c_str(), status,
                have_solution ? result.objective : std::numeric_limits<double>::quiet_NaN(),
                result.best_violation, wall);
    return 0;
}
