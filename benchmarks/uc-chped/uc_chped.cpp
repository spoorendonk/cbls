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
// number: `feasible` is the engine's verdict at `--feas-tol`; `verified` is an
// independent re-check by verify_uc_chped(), whose UC-semantic checks use an
// absolute 1e-4 and whose leading cbls::verify_model() pass uses 1e-6.

#include "data.h"
#include "greedy_init.h"
#include "uc_model.h"
#include "verify_uc_chped.h"

#include <algorithm>
#include <array>
#include <benchmarks/common/runner_args.h>
#include <cbls/cbls.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// The tolerance `verify_uc_chped()` applies to its UC-semantic checks,
/// distinct from the engine's `--feas-tol`. Passed explicitly at the call site
/// so this constant and the verifier cannot drift apart. Not the whole story:
/// that function also runs `cbls::verify_model()` at its own 1e-6, and checks
/// the objective recomputation on a relative band rather than this absolute
/// one, so `verified` is a verdict over all three.
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
    bool commit_set = false;
    std::string commit_sha = "unknown";
    std::string out_csv;  // default: <inst_dir>/comparison.csv
};

// The parse rule itself lives in include/cbls/arg_parse.h and the runners'
// report-and-exit policy in benchmarks/common/runner_args.h; what follows is
// only why this runner cares. Numeric flags go through std::stod / std::stoll rather than std::atof
// / std::atoll: the ato* family has no error path, so a typo'd value silently became 0 and a run
// that never searched would report like a solver result
// (bugprone-unchecked-string-to-number-conversion). Trailing characters are
// rejected too, so `--time-limit 60s` no longer quietly means 60.
//
// A bad double yields NaN rather than exiting: both double flags have a
// `> 0.0` guard below that NaN fails, so the parse layer adds a diagnostic
// without moving where the failure is reported. Integer flags have no such
// guard and so report and exit 2 directly.
//
// The rule and this policy are shared with the other runners rather than copied
// a fourth time -- a copy is where a fix to one silently diverges from the
// rest, which is what #130 factored out.
using cbls::bench::parse_double;
using cbls::bench::parse_int64;
/// Whether two paths name the same file. Canonicalised when both exist, so that
/// `./benchmarks/x` and an absolute path to it compare equal; a path that does
/// not exist yet cannot be the published table, so falling back to the lexical
/// form is safe rather than merely convenient.
bool same_file(const std::string& a, const std::string& b) {
    std::error_code ec;
    const std::filesystem::path pa = std::filesystem::weakly_canonical(a, ec);
    if (ec) {
        return a == b;
    }
    const std::filesystem::path pb = std::filesystem::weakly_canonical(b, ec);
    if (ec) {
        return a == b;
    }
    return pa == pb;
}

/// True when a double flag names a usable positive quantity. Both failure modes
/// of the parse layer fall out here: NaN (the token was not a number) and inf
/// (std::stod accepts "inf" happily).
bool is_positive_finite(double v) {
    return v > 0.0 && std::isfinite(v);
}

Args parse_args(int argc, char** argv) {
    Args a;
    bool inst_dir_set = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--verify") {
            a.do_verify = true;
        } else if (s == "--time-limit" && i + 1 < argc) {
            a.time_limit = parse_double("--time-limit", argv[++i]);
            a.time_limit_set = true;
        } else if (s == "--seed" && i + 1 < argc) {
            const int64_t seed = parse_int64("--seed", argv[++i]);
            // The seed is published on every measured row so a run can be
            // repeated. A negative one wraps to a uint64_t that --seed itself
            // then rejects as out of range, so the recorded value would not be
            // usable as an input.
            if (seed < 0) {
                std::fprintf(stderr, "--seed must be >= 0 (got %lld)\n",
                             static_cast<long long>(seed));
                std::exit(2);
            }
            a.seed = static_cast<uint64_t>(seed);
        } else if (s == "--feas-tol" && i + 1 < argc) {
            a.feas_tol = parse_double("--feas-tol", argv[++i]);
        } else if (s == "--instance" && i + 1 < argc) {
            a.instances.emplace_back(argv[++i]);
        } else if (s == "--commit" && i + 1 < argc) {
            a.commit_sha = argv[++i];
            a.commit_set = true;
        } else if (s == "--out" && i + 1 < argc) {
            a.out_csv = argv[++i];
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_uc_chped [inst-dir] [--verify] [--time-limit S] [--seed N]"
                " [--feas-tol T] [--instance NAME ...] [--commit SHA] [--out CSV]\n");
            std::exit(0);
        } else if (!s.empty() && s[0] == '-') {
            // A typo'd flag must not silently become the instance directory:
            // this tool's output is published, so a misparsed argument would
            // produce a plausible-looking but wrong results table. The test is
            // on a single leading dash, not two, so `-x` is caught as well.
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
    // A NaN from parse_double fails this guard, as does a literal 0 or a
    // negative: solve() with a non-positive budget returns having searched
    // nothing, which would publish a full-looking table of empty results.
    // std::stod also accepts "inf", which would never terminate, so the guard
    // tests isfinite as well as positivity.
    if (a.time_limit_set && !is_positive_finite(a.time_limit)) {
        std::fprintf(stderr, "--time-limit must be > 0 (got %g)\n", a.time_limit);
        std::exit(2);
    }
    // isfinite as well as > 0: std::stod accepts "inf", and an infinite
    // tolerance calls every assignment feasible, publishing a full-looking table
    // of rows the verifier would reject.
    if (!is_positive_finite(a.feas_tol)) {
        std::fprintf(stderr, "--feas-tol must be > 0 (got %g)\n", a.feas_tol);
        std::exit(2);
    }
    const std::string published = a.inst_dir + "/comparison.csv";
    if (a.out_csv.empty()) {
        a.out_csv = published;
    }
    // The guards below are about the FILE, not about which flags were typed. An
    // earlier cut tested `--out` for emptiness, which meant the documented
    // command -- which passes `--out <the published table>` explicitly --
    // satisfied every guard while doing exactly the damage they exist to stop:
    // a `--time-limit 60` pass over one instance rewrote the table and deleted
    // the cited reference rows for every instance it did not run, at exit 0.
    // So compare the resolved paths instead. Canonicalised where the file
    // exists, because `./benchmarks/...` and an absolute path are the same file
    // and a string compare says otherwise; a path that does not exist yet
    // cannot be the published table, so the lexical fallback is safe.
    if (same_file(a.out_csv, published)) {
        // A run that is not the full published measurement must never overwrite
        // the published results table. Both a partial roster and a shortened
        // budget qualify -- `--time-limit 2` is the obvious smoke test, and it
        // would otherwise replace the table with two-second results (the #88
        // hazard).
        const char* why = nullptr;
        if (!a.instances.empty()) {
            why = "--instance";
        } else if (a.time_limit_set) {
            why = "--time-limit";
        }
        if (why != nullptr) {
            std::fprintf(stderr,
                         "%s cannot write the published table %s "
                         "(pass --out elsewhere for an unpublished run)\n",
                         why, published.c_str());
            std::exit(2);
        }
        // A full-roster run at the documented budgets is the only thing allowed
        // to land here, and it still has to say which engine it measured:
        // "unknown" is exactly the provenance a reader cannot tell engine drift
        // from a bug with.
        if (!a.commit_set) {
            std::fprintf(stderr,
                         "writing %s requires an explicit --commit SHA "
                         "(pass --out elsewhere for an unpublished run)\n",
                         published.c_str());
            std::exit(2);
        }
    }
    return a;
}

/// The scratch file must not outlive a failed run: it sits in a tracked
/// directory, is not ignored, and a later `git add -A` would commit a truncated
/// shadow of the published table beside it. A kill -9 can still leave one, which
/// is the case the rename is for; every path this program controls cleans up.
void remove_temp(const std::string& path) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

/// Removes the scratch file unless the run reached its rename. RAII rather than
/// a call on each error path, because the solve loop can throw -- Model::var
/// raises out_of_range, make_subinstance raises invalid_argument -- and an
/// exception unwinds past every such call, leaving the scratch file behind in a
/// tracked directory that does not ignore it.
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
    return {buf.data()};
}

/// A comma inside a free-text cell would shift every column after it, so the
/// writer substitutes rather than quoting: these cells are short diagnostics,
/// and a quoted field would need escaping rules the readers of this table
/// (grep, awk, a spreadsheet import) do not all implement.
std::string csv_text(std::string s) {
    // Nothing here quotes, so every character that would end a field or a
    // record has to be substituted rather than escaped -- including '"', which
    // a reader that does honour quoting would otherwise treat as opening one,
    // and '\r', which turns a row into two on a CRLF-aware reader.
    std::replace(s.begin(), s.end(), ',', ';');
    std::replace(s.begin(), s.end(), '"', '\'');
    std::replace(s.begin(), s.end(), '\n', ' ');
    std::replace(s.begin(), s.end(), '\r', ' ');
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
    if (args.time_limit_set) {
        csv << num(args.time_limit, 4) << "s (uniform override)";
    } else {
        csv << "per-horizon default map";
    }
    csv << ", verify " << (args.do_verify ? "on" : "off");
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
};

/// Restate an instance's published Table 2 bounds as cited reference rows.
/// Regenerated from `known_bounds` rather than copied forward from the previous
/// file, so the ten cited rows survive every regeneration without the writer
/// having to parse its own output. The bounds live in the instance jsonl, so
/// run_benchmark() refuses to write anything unless every rostered instance
/// loaded -- otherwise a run from the wrong directory would replace the
/// published table with one missing these rows. Instances with no published bounds
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
    row.source = "this work";
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

    // `feasible` does not imply a finite objective: SearchResult documents
    // `feasible == true` with `objective == +inf` as a feasibility witness whose
    // objective overflowed (#100), and tells callers to test isfinite before
    // using the value. An "inf" in a numeric column would read as a solve result.
    const bool have_objective = result.feasible && std::isfinite(result.objective);
    double gap = kNaN;
    if (have_bounds && have_objective && row.lb != 0.0) {
        gap = 100.0 * (result.objective - row.lb) / row.lb;
    }

    // Console.
    if (have_bounds && have_objective) {
        std::printf("%12.1f %12.1f %7.2f%% %7.1fs", result.objective, row.lb, gap,
                    result.time_seconds);
    } else {
        std::printf("%12.1f %12s %8s %7.1fs", result.feasible ? result.objective : -1.0, "-",
                    result.feasible ? "-" : "INFEAS", result.time_seconds);
    }

    bool withhold = false;
    if (args.do_verify && result.feasible) {
        auto vr = cbls::uc_chped::verify_uc_chped(ucm, inst, kVerifierTolerance);
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

    if (have_objective && !withhold) {
        row.objective = result.objective;
        row.gap_pct = gap;
    }
    if (row.note.empty()) {
        if (!result.feasible) {
            row.note = "infeasible at feas_tol";
        } else if (!have_objective) {
            row.note = "feasible; no finite objective";
        }
    }

    std::printf("  (%s, %ld vars, %ld nodes, %ld iters)\n",
                result.feasible ? "feasible" : "INFEASIBLE", (long)ucm.model.num_vars(),
                (long)ucm.model.num_nodes(), (long)result.iterations);

    write_row(csv, row);
    csv.flush();
}

/// Apply `--instance`. Returns false when ANY requested name is absent from the
/// roster, not merely when none matched: an unmatched name is a typo, and a run
/// that dropped it silently would publish a table short a row with nothing to
/// say so. Roster order is preserved and a repeated name contributes one entry.
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

void print_tally(const Args& args, const Tally& tally) {
    std::printf("=== Tally ===\n");
    std::printf("wrote:                %s\n", args.out_csv.c_str());
    std::printf("commit:               %s\n", args.commit_sha.c_str());
    std::printf("seed:                 %llu\n", static_cast<unsigned long long>(args.seed));
    std::printf("feas-tol:             %g   (engine 'feasible')\n", args.feas_tol);
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
        std::fprintf(stderr, "--instance: unknown name(s) listed above (see --help)\n");
        return 2;
    }

    // Load the whole roster BEFORE opening the output file. The cited Pedroso
    // rows are regenerated from each instance's known_bounds, so a run started
    // from the wrong directory would otherwise replace the published table with
    // one that has no cited rows at all -- the failure that emptied the
    // miplib-fj table. Every uc-chped instance is committed in this repo and
    // nothing downloads them, so a missing .jsonl is a setup error, not a
    // result: report it and leave the existing file untouched.
    std::vector<cbls::uc_chped::UCInstance> loaded(specs.size());
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
    write_header_comment(csv, args);
    csv << kCsvHeader << "\n";

    std::printf("%-20s %6s %6s %12s %12s %8s %8s\n", "Instance", "Units", "Periods", "Objective",
                "Known LB", "Gap%", "Time(s)");
    std::printf("%-20s %6s %6s %12s %12s %8s %8s\n", "--------", "-----", "-------", "---------",
                "--------", "----", "-------");

    Tally tally;

    // Cited rows first, so the published bounds head the table as they did
    // before the measured rows.
    for (size_t i = 0; i < specs.size(); ++i) {
        write_reference_rows(csv, loaded[i], specs[i].periods);
    }

    for (size_t i = 0; i < specs.size(); ++i) {
        const InstanceSpec& spec = specs[i];
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

    // A stream error means the table on disk is short rows nobody can see are
    // missing, so it must not be published or reported as success.
    csv.flush();
    if (!csv) {
        std::fprintf(stderr, "Error writing %s; %s not replaced\n", tmp_csv.c_str(),
                     args.out_csv.c_str());
        return 2;
    }
    csv.close();
    std::error_code rename_ec;
    std::filesystem::rename(tmp_csv, args.out_csv, rename_ec);
    if (!rename_ec) {
        tmp_guard.release();  // it is the published table now, not scratch
    }
    if (rename_ec) {
        std::fprintf(stderr, "Failed to rename %s -> %s: %s\n", tmp_csv.c_str(),
                     args.out_csv.c_str(), rename_ec.message().c_str());
        return 2;
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
