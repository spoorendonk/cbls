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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    // CBLS variables need finite bounds, so an infinite one is clamped. 1e7 rather
    // than the engine's 1e9 because it measured better on the smoke roster — NOT
    // because it matches CP-SAT, which does not truncate variable domains at all
    // (`mip_max_bound` is not a domain clamp: an integer column bounded at 1e12 is
    // solved to 1e12). This is a CBLS-side restriction, so `n_clamped_bounds`
    // records how many columns it narrows and the comparison table publishes it.
    double inf_clamp = 1.0e7;
    std::string commit_sha = "unknown";
};

void print_usage() {
    std::printf(
        "Usage: cbls_mipfeas --instance NAME --out-dir DIR [--inst-dir DIR]\n"
        "                    [--budget SECONDS] [--seed N] [--feas-tol T]\n"
        "                    [--inf-clamp B] [--no-compound-moves] [--commit SHA]\n");
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
        } else if (s == "--inf-clamp" && i + 1 < argc) {
            a.inf_clamp = std::atof(argv[++i]);
        } else if (s == "--compound-moves") {
            a.compound_moves = true;
        } else if (s == "--no-compound-moves") {
            a.compound_moves = false;
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
        out_ << p.time_seconds << "," << p.objective << '\n';
    }

private:
    std::ofstream& out_;
    double last_written_ = std::numeric_limits<double>::infinity();
};

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

// Columns whose domain the clamp narrows. CBLS variables need finite bounds, so
// the model it searches is a restriction of the MPS on these: it can lose
// solutions, never invent them. Recorded per result because "the two engines
// solved the same program" is otherwise an assumption a reader cannot check.
int count_clamped_bounds(const cbls::MpsProblem& prob, double inf_clamp) {
    int n = 0;
    for (const auto& v : prob.vars) {
        if (!(v.lb >= -inf_clamp) || !(v.ub <= inf_clamp)) {
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
    j["compound_moves"] = args.compound_moves;
    j["inf_clamp"] = args.inf_clamp;
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
    // Same hazard as --budget: atof returns 0 on a parse failure, and a clamp of 0
    // collapses every column to [0, 0] rather than erroring — so one typo'd flag
    // scores an entire roster "no_solution" at exit code 0, which resume then
    // treats as work completed.
    if (!(args.inf_clamp > 0.0)) {
        std::fprintf(stderr, "--inf-clamp must be a positive bound\n");
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

    cbls::MpsToModelOptions mps_opts;
    mps_opts.inf_clamp = args.inf_clamp;
    cbls::MpsToModelResult built;
    try {
        built = cbls::mps_to_model(prob, mps_opts);
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
    cfg.use_compound_moves = args.compound_moves;

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
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Independent re-checks of the assignment solve() actually returned. It
    // restores best_state and full-evaluates before returning, so the model holds
    // that point now. Mirrors benchmarks/minlplib/minlplib.cpp, which already
    // refuses to publish a row failing any of these:
    //
    //   * residual — the engine's verdict, recomputed, on its own DAG;
    //   * integrality — an Int variable left fractional means the point is not a
    //     solution of the MIP at all;
    //   * objective drift — result.objective is the search's *running best*, taken
    //     when the incumbent was recorded. The number published has to be what the
    //     model evaluates to at the point being returned.
    int n_fractional_int = 0;
    if (result.feasible) {
        for (const auto& v : built.model.variables()) {
            if (v.type == cbls::VarType::Int && std::abs(v.value - std::round(v.value)) > 1e-9) {
                ++n_fractional_int;
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
    const double obj_drift = result.feasible && std::isfinite(result.objective)
                                 ? std::abs(model_obj - result.objective)
                                 : 0.0;
    const bool verdict_consistent =
        !result.feasible || (result.best_violation <= args.feas_tol && n_fractional_int == 0 &&
                             obj_drift <= 1e-6 * (std::abs(result.objective) + 1.0));
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
        {"n_fractional_int", n_fractional_int},
        {"objective_drift", obj_drift},
        {"n_vars", prob.vars.size()},
        {"n_cons", prob.rows.size()},
        {"n_int_vars", count_int_vars(prob)},
        {"n_clamped_bounds", count_clamped_bounds(prob, args.inf_clamp)},
    };
    j["objective"] = have_solution ? nlohmann::json(result.objective) : nlohmann::json(nullptr);
    write_result(args, j);

    std::printf("%-28s %-12s obj=%-16.8g viol=%-10.3g %8.2fs\n", args.instance.c_str(), status,
                have_solution ? result.objective : std::numeric_limits<double>::quiet_NaN(),
                result.best_violation, wall);
    return 0;
}
