// UC-CHPED benchmark runner.
//
// Loads each `ucp*.jsonl` from `benchmarks/instances/uc-chped/`, builds the
// unit-commitment-with-valve-point model, runs a fixed-time ViolationLS pass per
// (instance, horizon), and writes a per-row table to
// `benchmarks/instances/uc-chped/comparison.csv`.
//
// The feasibility tolerance is stated explicitly (`--feas-tol`, default
// `cbls::kDefaultFeasibilityTolerance`) and recorded on every row, rather than
// inherited from the engine default. That inheritance is what made the
// previously published rows uninterpretable when the default moved from 1e-9 to
// 1e-6, and they had to be deleted (issue #103): a published result must not
// silently change when an engine default moves.
//
// Two different tolerances appear in the output and they are not the same
// number: `feasible` is the engine's verdict at `--feas-tol`, `verified` is an
// independent re-check by verify_uc_chped() at its own 1e-4.

#include "data.h"
#include "greedy_init.h"
#include "uc_model.h"
#include "verify_uc_chped.h"

#include <algorithm>
#include <array>
#include <cbls/cbls.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// The verifier's own absolute tolerance, distinct from the engine's
/// `--feas-tol`. Mirrors the default argument of `verify_uc_chped()`; kept here
/// only so the tally can name it.
constexpr double kVerifierTolerance = 1e-4;

struct InstanceSpec {
    std::string filename;
    std::vector<int> periods;
};

struct Args {
    std::string inst_dir = "benchmarks/instances/uc-chped";
    bool do_verify = false;
    // Unset means "use the per-horizon budget map". A uniform override exists so
    // the roster can be smoke-tested in seconds; whichever budget applied is
    // written to the row's time_limit_s, so a short run is self-describing
    // rather than indistinguishable from a full one.
    bool time_limit_set = false;
    double time_limit = 0.0;
    // Seed for both the FJ warm-start RNG and solve(). One knob, so the two
    // cannot drift apart.
    uint64_t seed = 42;
    // Absolute constraint-violation tolerance for "feasible". Stated explicitly
    // rather than inherited, because it is a published property of these
    // results. Same value as the engine default, which matches SCIP's
    // numerics/feastol.
    double feas_tol = cbls::kDefaultFeasibilityTolerance;
    std::vector<std::string> instances;  // optional roster filter (base names)
    std::string commit_sha = "unknown";
    std::string out_csv;  // default: <inst_dir>/comparison.csv
};

// Numeric flags are parsed with std::stod / std::stoll rather than std::atof /
// std::atoll: the ato* family has no error path, so a typo'd value silently
// became 0 and a run that never searched would report like a solver result
// (bugprone-unchecked-string-to-number-conversion). Trailing characters are
// rejected too, so `--time-limit 60s` no longer quietly means 60.
//
// A bad double yields NaN rather than exiting here: both double flags have a
// `> 0.0` guard below that NaN fails, so the parse layer adds a diagnostic
// without moving where the failure is reported. Integer flags have no such
// guard and so report and exit 2 directly.
double parse_double(const char* flag, const std::string& text) {
    size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        used = 0;  // not a number at all; reported just below
    }
    if (text.empty() || used != text.size()) {
        std::fprintf(stderr, "%s: '%s' is not a number\n", flag, text.c_str());
        return kNaN;
    }
    return value;
}

int64_t parse_int64(const char* flag, const std::string& text) {
    size_t used = 0;
    int64_t value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        used = 0;  // not a number at all; reported just below
    }
    if (text.empty() || used != text.size()) {
        std::fprintf(stderr, "%s: '%s' is not an integer\n", flag, text.c_str());
        std::exit(2);
    }
    return value;
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--verify") {
            a.do_verify = true;
        } else if (s == "--time-limit" && i + 1 < argc) {
            a.time_limit = parse_double("--time-limit", argv[++i]);
            a.time_limit_set = true;
        } else if (s == "--seed" && i + 1 < argc) {
            a.seed = static_cast<uint64_t>(parse_int64("--seed", argv[++i]));
        } else if (s == "--feas-tol" && i + 1 < argc) {
            a.feas_tol = parse_double("--feas-tol", argv[++i]);
        } else if (s == "--instance" && i + 1 < argc) {
            a.instances.emplace_back(argv[++i]);
        } else if (s == "--commit" && i + 1 < argc) {
            a.commit_sha = argv[++i];
        } else if (s == "--out" && i + 1 < argc) {
            a.out_csv = argv[++i];
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_uc_chped [inst-dir] [--verify] [--time-limit S] [--seed N]"
                " [--feas-tol T] [--instance NAME ...] [--commit SHA] [--out CSV]\n");
            std::exit(0);
        } else if (s.rfind("--", 0) == 0) {
            // A typo'd flag must not silently become the instance directory:
            // this tool's output is published, so a misparsed argument would
            // produce a plausible-looking but wrong results table.
            std::fprintf(stderr, "Unknown or incomplete option: %s (see --help)\n", s.c_str());
            std::exit(2);
        } else {
            a.inst_dir = s;
        }
    }
    // A NaN from parse_double fails this guard, as does a literal 0 or a
    // negative: solve() with a non-positive budget returns having searched
    // nothing, which would publish a full-looking table of empty results.
    if (a.time_limit_set && !(a.time_limit > 0.0)) {
        std::fprintf(stderr, "--time-limit must be > 0 (got %g)\n", a.time_limit);
        std::exit(2);
    }
    if (!(a.feas_tol > 0.0)) {
        std::fprintf(stderr, "--feas-tol must be > 0 (got %g)\n", a.feas_tol);
        std::exit(2);
    }
    if (a.out_csv.empty()) {
        if (!a.instances.empty()) {
            // A partial roster must never overwrite the published results table:
            // the default --out is the file the README publishes, and a
            // single-instance debugging run would silently truncate it.
            std::fprintf(stderr,
                         "--instance requires an explicit --out (refusing to overwrite %s)\n",
                         (a.inst_dir + "/comparison.csv").c_str());
            std::exit(2);
        }
        a.out_csv = a.inst_dir + "/comparison.csv";
    }
    return a;
}

/// One CSV line. The first nine fields are the schema the pre-#103 table
/// published, in their original order, so a reader of the old file reads the
/// same columns here; everything the old file could not record is appended
/// after them.
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
    return std::string(buf.data());
}

/// A comma inside a free-text cell would shift every column after it, so the
/// writer substitutes rather than quoting: these cells are short diagnostics,
/// and a quoted field would need escaping rules the readers of this table
/// (grep, awk, the pandas one-liner in the README) do not all implement.
std::string csv_text(std::string s) {
    std::replace(s.begin(), s.end(), ',', ';');
    std::replace(s.begin(), s.end(), '\n', ' ');
    return s;
}

const char* const kCsvHeader =
    "instance,periods,method,objective,lb,gap_pct,time_s,source,note,"
    "ub,time_limit_s,seed,feasible,verified,max_violation,feas_tol,commit_sha";

void write_row(std::ostream& csv, const Row& r) {
    csv << csv_text(r.instance) << "," << r.periods << "," << csv_text(r.method) << ","
        << num(r.objective) << "," << num(r.lb) << "," << fixed2(r.gap_pct) << ","
        << num(r.time_s, 4) << "," << csv_text(r.source) << "," << csv_text(r.note) << ","
        << num(r.ub) << "," << num(r.time_limit_s, 4) << "," << r.seed << "," << r.feasible << ","
        << r.verified << "," << num(r.max_violation, 4) << "," << num(r.feas_tol, 3) << ","
        << csv_text(r.commit_sha) << "\n";
}

/// The provenance block. Emitted by the generator so that regenerating the table
/// cannot lose it; the copy committed alongside the cited rows says the same
/// thing.
void write_header_comment(std::ostream& csv, const Args& args) {
    csv << "# UC-CHPED comparison table. GENERATED by benchmarks/uc-chped/uc_chped.cpp\n"
           "# (`cbls_uc_chped --out <this file> --commit $(git rev-parse --short=7 HEAD)`).\n"
           "# Do not hand-edit the rows; re-run the generator.\n"
           "#\n"
           "# Model fidelity: our CBLS model and the SCIP reference (reference_solve.py)\n"
           "# both solve a ramp-free unit commitment -- no |P[t]-P[t-1]| constraints -- and\n"
           "# so does the source. Pedroso, Kubo & Viana 2014 states power balance, spinning\n"
           "# reserve, unit initial conditions and minimum up/down times only, with no\n"
           "# ramp-rate constraints\n"
           "# (https://web.fc.up.pt/dcc/Pubs/TReports/TR14/dcc-2014-05.pdf), and their\n"
           "# public instance-generation code carries no ramp data. The ramp question the\n"
           "# #73 audit left open is therefore settled (#77, closed as not planned): the\n"
           "# Table 2 bounds and our results describe the same problem, so gap_pct against\n"
           "# them compares like with like. The SCIP reference additionally uses a\n"
           "# 50-segment piecewise-linear approximation of the |d*sin(e*(Pmin-P))|\n"
           "# valve-point term (objective error bounded ~0.1%). See\n"
           "# benchmarks/uc-chped/FIDELITY.md.\n"
           "#\n"
           "# Two tolerances, not one: `feasible` is the engine's own verdict at the\n"
           "# `feas_tol` recorded on the row, while `verified` is an independent re-check\n"
           "# by verify_uc_chped() at ITS OWN absolute tolerance of 1e-4\n"
           "# (benchmarks/uc-chped/verify_uc_chped.h). `verified` is empty when the run was\n"
           "# made without --verify. A row that failed verification publishes no objective\n"
           "# and no gap: those columns would describe a solution we do not stand behind.\n"
           "#\n"
           "# `lb`/`ub` are the Pedroso Table 2 bounds carried in the instance jsonl, and\n"
           "# the `Pedroso MIP (1hr)` rows are those bounds restated as cited reference\n"
           "# results -- they are not measurements of this engine and so carry no seed,\n"
           "# tolerance or commit. They are re-emitted from the same bounds map on every\n"
           "# run, so a regeneration cannot drop them.\n"
           "#\n";
    csv << "# Run: commit " << csv_text(args.commit_sha) << ", seed " << args.seed << ", feas-tol "
        << num(args.feas_tol, 3) << ", time limit ";
    if (args.time_limit_set) {
        csv << num(args.time_limit, 4) << "s (uniform override)";
    } else {
        csv << "per-horizon default map";
    }
    csv << ", verify " << (args.do_verify ? "on" : "off") << ".\n";
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
    int not_found = 0;
};

/// Restate an instance's published Table 2 bounds as cited reference rows.
/// Regenerated from `known_bounds` rather than copied forward from the previous
/// file, so the ten cited rows survive every regeneration without the writer
/// having to parse its own output. Instances with no published bounds
/// (ucp100/ucp200) contribute none -- inventing a citation for them would be
/// worse than omitting it.
void write_reference_rows(std::ostream& csv, const cbls::uc_chped::UCInstance& base,
                          const std::vector<int>& periods) {
    for (int T : periods) {
        auto it = base.known_bounds.find(T);
        if (it == base.known_bounds.end()) {
            continue;
        }
        const double lb = it->second.first;
        const double ub = it->second.second;
        Row r;
        r.instance = base.name;
        r.periods = T;
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

/// Solve one (instance, horizon) pair and append its row.
void run_one(std::ostream& csv, const Args& args, const cbls::uc_chped::UCInstance& inst,
             const std::string& instance_name, double tlim, Tally& tally) {
    auto ucm = cbls::uc_chped::build_uc_model(inst);
    const int T = inst.n_periods;

    std::printf("%-20s %6d %6d ", inst.name.c_str(), inst.n_units, T);
    std::fflush(stdout);

    // Greedy initialization + short FJ polish.
    cbls::uc_chped::greedy_uc_initialize(ucm.model, inst, ucm);
    {
        cbls::RNG init_rng(args.seed);
        cbls::ViolationManager init_vm(ucm.model);
        cbls::fj_nl_initialize(ucm.model, init_vm, 200, &init_rng, 1.0);
    }

    cbls::FloatIntensifyHook hook;
    cbls::LNS lns(0.3);
    cbls::SearchConfig cfg;
    cfg.skip_init = true;
    cfg.feasibility_tolerance = args.feas_tol;
    auto result = cbls::solve(ucm.model, tlim, args.seed, false, &hook, &lns, 3, nullptr, cfg);
    ++tally.solved;
    if (result.feasible) {
        ++tally.feasible;
    }

    Row row;
    row.instance = instance_name;
    row.periods = T;
    row.method = "CBLS ViolationLS";
    row.time_s = result.time_seconds;
    row.time_limit_s = tlim;
    row.seed = std::to_string(args.seed);
    row.feasible = result.feasible ? "true" : "false";
    row.max_violation = result.best_violation;
    row.feas_tol = args.feas_tol;
    row.commit_sha = args.commit_sha;

    auto it = inst.known_bounds.find(T);
    const bool have_bounds = it != inst.known_bounds.end();
    if (have_bounds) {
        row.lb = it->second.first;
        row.ub = it->second.second;
    }

    double gap = kNaN;
    if (have_bounds && result.feasible && row.lb != 0.0) {
        gap = 100.0 * (result.objective - row.lb) / row.lb;
    }

    // Console.
    if (have_bounds && result.feasible) {
        std::printf("%12.1f %12.1f %7.2f%% %7.1fs", result.objective, row.lb, gap,
                    result.time_seconds);
    } else {
        std::printf("%12.1f %12s %8s %7.1fs", result.feasible ? result.objective : -1.0, "-",
                    result.feasible ? "-" : "INFEAS", result.time_seconds);
    }

    bool withhold = false;
    if (args.do_verify && result.feasible) {
        auto vr = cbls::uc_chped::verify_uc_chped(ucm, inst);
        std::printf("  %s", vr.ok ? "VERIFIED" : "VERIFY FAIL");
        row.verified = vr.ok ? "true" : "false";
        if (vr.ok) {
            ++tally.verified_pass;
        } else {
            ++tally.verified_fail;
            // A row the independent checker rejected must not publish the
            // objective or gap it was rejected for: those columns would describe
            // a solution we do not stand behind.
            withhold = true;
            row.note = "verify-fail (" + std::to_string(vr.errors.size()) + " errors)";
            vr.print_diagnostics(stdout);
        }
    }

    if (result.feasible && !withhold) {
        row.objective = result.objective;
        row.gap_pct = gap;
    }
    if (!result.feasible && row.note.empty()) {
        row.note = "infeasible at feas_tol";
    }

    std::printf("  (%s, %ld vars, %ld nodes, %ld iters)\n",
                result.feasible ? "feasible" : "INFEASIBLE", (long)ucm.model.num_vars(),
                (long)ucm.model.num_nodes(), (long)result.iterations);

    write_row(csv, row);
    csv.flush();
}

/// Apply `--instance`. Returns false when the filter matched nothing, which is
/// an error rather than an empty run: an unmatched name is a typo, and silently
/// writing a table with no measured rows is exactly the failure mode the
/// not-found rows below exist to prevent.
bool filter_specs(const Args& args, std::vector<InstanceSpec>& specs) {
    if (args.instances.empty()) {
        return true;
    }
    std::vector<InstanceSpec> filtered;
    for (const auto& spec : specs) {
        const std::string name = base_name(spec.filename);
        if (std::find(args.instances.begin(), args.instances.end(), name) != args.instances.end()) {
            filtered.push_back(spec);
        }
    }
    if (filtered.empty()) {
        return false;
    }
    specs = filtered;
    return true;
}

void write_not_found_rows(std::ostream& csv, const Args& args, const InstanceSpec& spec,
                          Tally& tally) {
    // Emit a row per rostered horizon instead of skipping silently: the roster
    // is the list of record, so an absent .jsonl must show up as a missing
    // result rather than as a table that quietly shrank.
    for (int T : spec.periods) {
        Row r;
        r.instance = base_name(spec.filename);
        r.periods = T;
        r.method = "CBLS ViolationLS";
        r.note = "not-found: " + args.inst_dir + "/" + spec.filename;
        r.feasible = "false";
        r.seed = std::to_string(args.seed);
        r.feas_tol = args.feas_tol;
        r.commit_sha = args.commit_sha;
        write_row(csv, r);
        ++tally.not_found;
    }
}

void print_tally(const Args& args, const Tally& tally) {
    std::printf("=== Tally ===\n");
    std::printf("wrote:                %s\n", args.out_csv.c_str());
    std::printf("commit:               %s\n", args.commit_sha.c_str());
    std::printf("seed:                 %llu\n", static_cast<unsigned long long>(args.seed));
    std::printf("feas-tol:             %g   (engine 'feasible')\n", args.feas_tol);
    std::printf("verifier tolerance:   %g   (independent 'verified')\n", kVerifierTolerance);
    std::printf("solved:               %d\n", tally.solved);
    std::printf("feasible:             %d\n", tally.feasible);
    if (args.do_verify) {
        std::printf("verify passed:        %d\n", tally.verified_pass);
        std::printf("verify failed:        %d\n", tally.verified_fail);
    } else {
        std::printf("verify:               not run (--verify off)\n");
    }
    std::printf("instances not found:  %d\n", tally.not_found);
}

int run_benchmark(int argc, char** argv) {
    const Args args = parse_args(argc, argv);

    // Time limits per number of periods, used unless --time-limit overrides.
    std::map<int, double> time_limits = {
        {1, 10.0}, {3, 30.0}, {6, 60.0}, {12, 120.0}, {24, 300.0}, {48, 600.0}, {168, 600.0},
    };
    const double default_time_limit = 300.0;

    // Instance specs: filename + period options.
    std::vector<InstanceSpec> specs = {
        {"ucp13.jsonl", {1, 3, 6, 12, 24}},  {"ucp40.jsonl", {1, 3, 6, 12, 24}},
        {"ucp100.jsonl", {1, 3, 6, 12, 24}}, {"ucp100-48p.jsonl", {48}},
        {"ucp100-168p.jsonl", {168}},        {"ucp200.jsonl", {1, 3, 6, 12, 24}},
        {"ucp200-48p.jsonl", {48}},          {"ucp200-168p.jsonl", {168}},
    };
    if (!filter_specs(args, specs)) {
        std::fprintf(stderr, "--instance matched nothing in the roster (see --help)\n");
        return 2;
    }

    std::ofstream csv(args.out_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", args.out_csv.c_str());
        return 2;
    }
    write_header_comment(csv, args);
    csv << kCsvHeader << "\n";

    std::printf("%-20s %6s %6s %12s %12s %8s %8s\n", "Instance", "Units", "Periods", "Objective",
                "Known LB", "Gap%", "Time(s)");
    std::printf("%-20s %6s %6s %12s %12s %8s %8s\n", "--------", "-----", "-------", "---------",
                "--------", "----", "-------");

    Tally tally;

    // Load everything up front and emit the cited rows first, so the published
    // bounds head the table as they did before the measured rows.
    std::vector<cbls::uc_chped::UCInstance> loaded(specs.size());
    std::vector<bool> loaded_ok(specs.size(), false);
    for (size_t i = 0; i < specs.size(); ++i) {
        try {
            loaded[i] = cbls::uc_chped::load_jsonl(args.inst_dir + "/" + specs[i].filename);
            loaded_ok[i] = true;
        } catch (const std::exception& e) {
            std::printf("Skipping %s: %s\n", specs[i].filename.c_str(), e.what());
            continue;
        }
        write_reference_rows(csv, loaded[i], specs[i].periods);
    }

    for (size_t i = 0; i < specs.size(); ++i) {
        const InstanceSpec& spec = specs[i];
        if (!loaded_ok[i]) {
            write_not_found_rows(csv, args, spec, tally);
            continue;
        }
        const cbls::uc_chped::UCInstance& base = loaded[i];

        for (int T : spec.periods) {
            cbls::uc_chped::UCInstance inst;
            if (T == base.n_periods) {
                inst = base;  // already the right size
            } else if (T < base.n_periods) {
                inst = cbls::uc_chped::make_subinstance(base, T);
            } else {
                std::printf("%-20s %6d %6d  (skipped: T > n_periods)\n", base.name.c_str(),
                            base.n_units, T);
                Row r;
                r.instance = base.name;
                r.periods = T;
                r.method = "CBLS ViolationLS";
                r.note = "skipped: T > n_periods";
                r.commit_sha = args.commit_sha;
                write_row(csv, r);
                continue;
            }

            double tlim = args.time_limit;
            if (!args.time_limit_set) {
                auto lit = time_limits.find(T);
                tlim = lit != time_limits.end() ? lit->second : default_time_limit;
            }
            run_one(csv, args, inst, base.name, tlim, tally);
        }
        std::printf("\n");
    }

    print_tally(args, tally);
    return 0;
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
