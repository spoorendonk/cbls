// MINLPLib non-convex benchmark runner.
//
// Loads each `.nl` from `benchmarks/instances/minlplib/`, builds a CBLS model
// via the NL-to-Model adapter, runs a fixed-time ViolationLS pass, and writes a
// per-instance row to `benchmarks/instances/minlplib/comparison.csv`. Unsupported
// operators and non-finite blowups are reported (not crashed). The roster and
// published bounds come from `bounds.csv` (written by download.py).

#include <algorithm>
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct Args {
    std::string inst_dir = "benchmarks/instances/minlplib";
    // 60s per instance. The previously published run used 5s, which is a stingy
    // budget for a general-purpose non-convex MINLP heuristic (issue #88).
    double time_limit = 60.0;
    uint64_t seed = 1;
    // Absolute constraint-violation tolerance for "feasible". Stated explicitly
    // rather than inherited, because it is a published property of these
    // results: it matches SCIP's numerics/feastol default, which keeps the SCIP
    // baseline (#89) comparable. Same value as the engine default.
    double feas_tol = cbls::kDefaultFeasibilityTolerance;
    std::vector<std::string> instances;  // optional override
    std::string commit_sha = "unknown";
    std::string out_csv;    // default: <inst_dir>/comparison.csv
    std::string trace_csv;  // optional: anytime profile (best objective vs time)
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--time-limit" && i + 1 < argc) {
            a.time_limit = std::atof(argv[++i]);
        } else if (s == "--seed" && i + 1 < argc) {
            a.seed = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (s == "--feas-tol" && i + 1 < argc) {
            a.feas_tol = std::atof(argv[++i]);
        } else if (s == "--instance" && i + 1 < argc) {
            a.instances.emplace_back(argv[++i]);
        } else if (s == "--commit" && i + 1 < argc) {
            a.commit_sha = argv[++i];
        } else if (s == "--out" && i + 1 < argc) {
            a.out_csv = argv[++i];
        } else if (s == "--trace" && i + 1 < argc) {
            a.trace_csv = argv[++i];
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_minlplib [inst-dir] [--time-limit S] [--seed N]"
                " [--feas-tol T] [--instance NAME ...] [--commit SHA] [--out CSV]"
                " [--trace CSV]\n");
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
    if (!(a.time_limit > 0.0)) {
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
            // single-instance debugging run would silently truncate it to one row.
            std::fprintf(stderr,
                         "--instance requires an explicit --out (refusing to overwrite %s)\n",
                         (a.inst_dir + "/comparison.csv").c_str());
            std::exit(2);
        }
        a.out_csv = a.inst_dir + "/comparison.csv";
    }
    return a;
}

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

struct Bounds {
    bool have = false;
    double primal = std::numeric_limits<double>::quiet_NaN();
    double dual = std::numeric_limits<double>::quiet_NaN();
    // Catalogue integer-variable count (nbinvars + nintvars); -1 when absent.
    // Cross-checks the NL reader's recovered integrality.
    int n_disc = -1;
};

// Parse bounds.csv:
// instance,structure,nvars,ncons,objsense,primal_bks,dual_bound[,n_disc_vars_bks].
std::unordered_map<std::string, Bounds> load_bounds(const std::string& path) {
    std::unordered_map<std::string, Bounds> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        return out;
    }
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> cells;
        while (std::getline(ss, cell, ',')) {
            cells.push_back(cell);
        }
        if (cells.size() < 7) {
            continue;
        }
        Bounds b;
        b.have = true;
        auto parse = [](const std::string& s) -> double {
            try {
                return std::stod(s);
            } catch (...) {
                return std::numeric_limits<double>::quiet_NaN();
            }
        };
        b.primal = parse(cells[5]);
        b.dual = parse(cells[6]);
        if (cells.size() >= 8) {
            double n = parse(cells[7]);
            b.n_disc = std::isnan(n) ? -1 : static_cast<int>(n);
        }
        out[cells[0]] = b;
    }
    return out;
}

// Split one CSV line, honouring "quoted" fields (which may contain commas and
// doubled "" escapes). Needed for analysis_notes.csv, whose note column is prose.
std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    bool in_quotes = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (in_quotes) {
            if (c == '"') {
                if (i + 1 < line.size() && line[i + 1] == '"') {
                    cur += '"';
                    ++i;
                } else {
                    in_quotes = false;
                }
            } else {
                cur += c;
            }
        } else if (c == '"') {
            in_quotes = true;
        } else if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    out.push_back(cur);
    return out;
}

// Curated root-cause annotations: instance -> "classification: note". Optional;
// an absent file just means no annotations. Records are one per physical line:
// a quoted field may contain commas but not a newline.
std::unordered_map<std::string, std::string> load_analysis_notes(const std::string& path) {
    std::unordered_map<std::string, std::string> out;
    std::ifstream f(path);
    if (!f.is_open()) {
        return out;
    }
    std::string line;
    bool header_seen = false;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!header_seen) {
            header_seen = true;  // "instance,classification,note"
            continue;
        }
        auto cells = split_csv_line(line);
        if (cells.size() < 3) {
            continue;
        }
        out[cells[0]] = cells[1] + ": " + cells[2];
    }
    return out;
}

// Records the incumbent objective against wall time, so the value of the tail
// of the budget can be measured rather than guessed. solve() emits progress on
// each new best and roughly once a second.
class TraceRecorder : public cbls::SolveCallback {
public:
    TraceRecorder(std::ofstream& out, std::string instance)
        : out_(out), instance_(std::move(instance)) {}

    void on_progress(const cbls::SolveProgress& p) override {
        if (!p.feasible || !std::isfinite(p.objective)) {
            return;  // no incumbent yet
        }
        // Flushed per row, as the results CSV is: a run is tens of minutes long
        // and an interrupted one must not lose its buffered profile.
        // NOTE: p.objective is the internally *minimised* value, so a maximize
        // instance appears negated here relative to comparison.csv. Improvement
        // ratios are invariant under that, but raw values are not.
        out_ << instance_ << "," << p.time_seconds << "," << p.objective << ","
             << (p.new_best ? 1 : 0) << '\n';
    }

private:
    std::ofstream& out_;
    std::string instance_;
};

// Roster order: instances listed in bounds.csv, in file order. If bounds.csv is
// missing, the caller falls back to scanning for *.nl (handled in main).
std::vector<std::string> roster_from_bounds(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream f(path);
    if (!f.is_open()) {
        return names;
    }
    std::string line;
    bool header = true;
    while (std::getline(f, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (line.empty()) {
            continue;
        }
        auto comma = line.find(',');
        names.push_back(comma == std::string::npos ? line : line.substr(0, comma));
    }
    return names;
}

struct Tally {
    int parsed = 0;
    int closed = 0;
    int feasible = 0;
    int better = 0;
    int matches = 0;     // equal to BKS within the tie band
    int within_tol = 0;  // better than BKS, but by less than the tolerance slack
    int worse = 0;
    int mixed_integer = 0;  // instances with >=1 integer column (integrality enforced)
    int failed_nonfinite = 0;
    int skipped_unsupported = 0;
    int not_found = 0;
    int errored = 0;  // read/build/solve exceptions
    int integrality_mismatch = 0;
    int verify_failed = 0;  // reported feasible but failed the independent re-check
    int near_miss = 0;      // infeasible, but residual within kNearMiss of feasible
    int nonfinite_obj = 0;  // infeasible with a non-finite objective at the closest approach
};

// An infeasible run whose closest approach is this small is a numerical
// near-miss (a tolerance/conditioning story), not a search failure to find the
// feasible region. Used only to classify the note, never to claim feasibility.
constexpr double kNearMiss = 1e-4;

// Where the closest-approach assignment is still violated. `solve()` leaves the
// model at that assignment when it reports infeasible.
struct Residual {
    double worst = 0.0;
    int nl_row = -1;  // NL constraint row index, -1 if it maps to no recorded row
    cbls::NlBoundType row_type = cbls::NlBoundType::Free;
    int n_violated = 0;
};

Residual worst_residual(const cbls::NlProblem& prob, const cbls::NlToModelResult& built,
                        double tol) {
    // Always the model `built` owns; taking it separately invited passing a
    // different one.
    const cbls::Model& model = built.model;
    // Map constraint node id -> NL row so the worst offender can be named. Range
    // rows record only their upper node, so the lower half maps to -1.
    std::unordered_map<int32_t, int> node_to_row;
    for (size_t i = 0; i < built.constraint_node_ids.size(); ++i) {
        if (built.constraint_node_ids[i] >= 0) {
            node_to_row[built.constraint_node_ids[i]] = static_cast<int>(i);
        }
    }
    Residual r;
    const int32_t obj_ci = model.objective_constraint_idx();
    const auto& cids = model.constraint_ids();
    for (size_t i = 0; i < cids.size(); ++i) {
        if (static_cast<int32_t>(i) == obj_ci) {
            continue;  // artificial objective bound, not a real constraint
        }
        double v = model.node(cids[i]).value;
        if (std::isnan(v)) {
            v = std::numeric_limits<double>::infinity();
        }
        if (v > tol) {
            ++r.n_violated;
        }
        if (v > r.worst) {
            r.worst = v;
            auto it = node_to_row.find(cids[i]);
            r.nl_row = it != node_to_row.end() ? it->second : -1;
            r.row_type = cbls::NlBoundType::Free;
            if (r.nl_row >= 0 && r.nl_row < static_cast<int>(prob.constraints.size())) {
                r.row_type = prob.constraints[static_cast<size_t>(r.nl_row)].bound.type;
            }
        }
    }
    return r;
}

const char* bound_type_name(cbls::NlBoundType t) {
    switch (t) {
        case cbls::NlBoundType::Range:
            return "range";
        case cbls::NlBoundType::Upper:
            return "<=";
        case cbls::NlBoundType::Lower:
            return ">=";
        case cbls::NlBoundType::Equal:
            return "==";
        case cbls::NlBoundType::Free:
            return "free";
    }
    return "?";
}

// Signed gap in percent, oriented so a POSITIVE value always means "worse than
// the reference" regardless of objective sense: for a minimize instance a larger
// objective is worse, for a maximize one a smaller one is. Without the flip a
// maximize row reads like a 64% improvement when it is a 64% miss.
//
// NOTE: when |ref| < 1e-12 the return value is an ABSOLUTE residual, not a
// percent — a percentage against zero is meaningless. Consumers must not bucket
// those rows as percentages; the CSV does not distinguish them, so the README
// records which instances they are (mathopt1, prob09, least).
double safe_gap(double obj, double ref, bool maximizing) {
    if (std::isnan(obj) || std::isnan(ref)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double diff = maximizing ? (ref - obj) : (obj - ref);
    double denom = std::abs(ref);
    if (denom < 1e-12) {
        return diff;  // ref ~ 0: absolute residual
    }
    return 100.0 * diff / denom;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::string bounds_path = args.inst_dir + "/bounds.csv";
    auto bounds = load_bounds(bounds_path);
    auto analysis_notes = load_analysis_notes(args.inst_dir + "/analysis_notes.csv");

    std::vector<std::string> insts = args.instances;
    if (insts.empty()) {
        insts = roster_from_bounds(bounds_path);
        if (insts.empty()) {
            std::printf("WARNING: %s missing and no --instance given; nothing to run.\n",
                        bounds_path.c_str());
            return 1;
        }
    }

    std::ofstream trace;
    if (!args.trace_csv.empty()) {
        trace.open(args.trace_csv);
        if (!trace.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", args.trace_csv.c_str());
            return 2;
        }
        trace << "instance,time_seconds,objective,new_best\n";
    }

    std::ofstream csv(args.out_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", args.out_csv.c_str());
        return 2;
    }
    csv << "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,"
           "wall_seconds,feasible,note,commit_sha,max_violation,n_int_vars\n";

    std::printf("\n%-22s %12s %12s %10s %9s  %s\n", "Instance", "Objective", "BKS", "Gap%",
                "Time(s)", "Note");
    std::printf("%-22s %12s %12s %10s %9s  %s\n", "--------", "---------", "---", "----", "-------",
                "----");

    Tally t;

    for (const std::string& name : insts) {
        std::string nl_path = args.inst_dir + "/" + name + ".nl";
        if (!file_exists(nl_path)) {
            std::printf("%-22s  (skipped: %s not found)\n", name.c_str(), nl_path.c_str());
            ++t.not_found;
            // Emit a row anyway: bounds.csv is the roster of record, so a
            // silently absent row makes the results table disagree with the
            // roster it claims to cover, visible only on stdout.
            csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false,not-found," << args.commit_sha
                << ",NaN,NaN\n";
            continue;
        }

        cbls::NlProblem prob;
        try {
            prob = cbls::read_nl(nl_path);
        } catch (const std::exception& e) {
            // An unknown opcode or unsupported segment is a coverage gap, not a
            // malformed file: bucket it as skipped(unsupported), not an error.
            std::string what = e.what();
            bool unsupported = what.find("NL_UNKNOWN_OPCODE") != std::string::npos ||
                               what.find("not supported by this reader") != std::string::npos;
            std::replace(what.begin(), what.end(), ',', ';');
            if (unsupported) {
                std::printf("%-22s  (skipped: %s)\n", name.c_str(), what.c_str());
                ++t.skipped_unsupported;
                csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false,unsupported: " << what << ","
                    << args.commit_sha << ",NaN,NaN\n";
            } else {
                std::printf("%-22s  ERROR reading: %s\n", name.c_str(), what.c_str());
                ++t.errored;
                csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false,read-error," << args.commit_sha
                    << ",NaN,NaN\n";
            }
            continue;
        }
        ++t.parsed;

        cbls::NlToModelResult built;
        std::string note;
        double max_violation = 0.0;  // closest-approach residual on infeasible rows
        try {
            built = cbls::nl_to_model(prob);
        } catch (const std::exception& e) {
            std::printf("%-22s  ERROR building model: %s\n", name.c_str(), e.what());
            ++t.errored;
            csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false,build-error," << args.commit_sha
                << ",NaN,NaN\n";
            continue;
        }

        auto bit = bounds.find(name);
        Bounds b = bit != bounds.end() ? bit->second : Bounds{};

        if (!built.supported) {
            note = built.skipped_reasons.empty() ? "unsupported"
                                                 : "unsupported: " + built.skipped_reasons[0];
            // Sanitise commas in the note so the CSV stays well-formed.
            std::replace(note.begin(), note.end(), ',', ';');
            std::printf("%-22s  (skipped: %s)\n", name.c_str(), note.c_str());
            ++t.skipped_unsupported;
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN,0,false," << note
                << "," << args.commit_sha << ",NaN," << prob.n_discrete_vars << "\n";
            continue;
        }
        ++t.closed;
        // Counted here, not earlier, so the tally's mixed-integer count is a
        // subset of the instances actually built (as the printout implies).
        if (prob.n_discrete_vars > 0) {
            ++t.mixed_integer;
        }

        // Integrality cross-check: the NL header declares how many columns are
        // discrete, and Gay's variable ordering places them. If that disagrees
        // with MINLPLib's own nbinvars+nintvars, the model we just built is not
        // the instance the published bound refers to — say so rather than
        // reporting a gap against a bound for a different problem. Counted here,
        // after the supported check, because only this path writes a row that can
        // carry the note; counting earlier would report a mismatch no row explains.
        std::string integrality_note;
        if (b.n_disc >= 0 && b.n_disc != prob.n_discrete_vars) {
            ++t.integrality_mismatch;
            integrality_note = "integrality-mismatch(nl=" + std::to_string(prob.n_discrete_vars) +
                               " catalogue=" + std::to_string(b.n_disc) + ")";
            std::printf("%-22s  WARNING: %s\n", name.c_str(), integrality_note.c_str());
        }

        std::printf("%-22s ", name.c_str());
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        cbls::FloatIntensifyHook hook;
        cbls::LNS lns(0.3);
        cbls::SearchConfig cfg;
        cfg.feasibility_tolerance = args.feas_tol;
        cbls::SearchResult result;
        try {
            TraceRecorder recorder(trace, name);
            result = cbls::solve(built.model, args.time_limit, args.seed,
                                 /*use_fj=*/true, &hook, &lns, /*lns_interval=*/3,
                                 trace.is_open() ? &recorder : nullptr, cfg);
        } catch (const std::exception& e) {
            std::printf(" ERROR solving: %s\n", e.what());
            ++t.errored;
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN,0,false,solve-error,"
                << args.commit_sha << ",NaN," << prob.n_discrete_vars << "\n";
            continue;
        }
        auto t1 = std::chrono::steady_clock::now();
        double wall = std::chrono::duration<double>(t1 - t0).count();

        // solve() reports the *minimised* objective. For a maximize instance the
        // model objective was negated, so un-negate to recover the true value and
        // make gap-to-BKS comparable to the published (max-sense) bound.
        double obj = result.feasible ? result.objective : std::numeric_limits<double>::quiet_NaN();
        if (result.feasible && built.model.is_maximizing()) {
            obj = -obj;
        }

        // A feasible-but-non-finite objective means the guard fired: count it as
        // failed(non-finite), not feasible.
        bool nonfinite = result.feasible && !std::isfinite(obj);
        if (nonfinite) {
            ++t.failed_nonfinite;
            note = "non-finite";
            if (!integrality_note.empty()) {
                // else the tally reports a mismatch that no CSV row explains
                note += "; " + integrality_note;
            }
            std::printf("%12s %12.4g %10s %8.2fs  %s\n", "NONFIN", b.primal, "N/A", wall,
                        note.c_str());
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN," << wall << ",false,"
                << note << "," << args.commit_sha << ",NaN," << prob.n_discrete_vars << "\n";
            continue;
        }

        const bool maximizing = built.model.is_maximizing();
        double gap_bks = safe_gap(obj, b.primal, maximizing);
        double gap_dual = safe_gap(obj, b.dual, maximizing);

        // Independent re-check of the returned assignment. solve() restores
        // best_state and full-evaluates, so this re-derives feasibility and
        // integrality from the model rather than trusting the search's own
        // bookkeeping. A reported-feasible row that fails here is a solver bug,
        // and must not be published as a solved instance.
        bool verified = result.feasible;
        if (result.feasible) {
            Residual r = worst_residual(prob, built, args.feas_tol);
            max_violation = r.worst;
            int frac = 0;
            for (const auto& v : built.model.variables()) {
                if (v.type == cbls::VarType::Int &&
                    std::abs(v.value - std::round(v.value)) > 1e-9) {
                    ++frac;
                }
            }
            // The published objective must also be the one the model reports at
            // the returned assignment, not just the search's running best.
            double model_obj = built.objective_node_id >= 0
                                   ? built.model.node(built.objective_node_id).value
                                   : obj;
            if (built.model.is_maximizing()) {
                model_obj = -model_obj;  // same un-negation applied to `obj` above
            }
            const double obj_drift = std::abs(model_obj - obj);
            const bool obj_mismatch = obj_drift > 1e-6 * (std::abs(obj) + 1.0);

            if (r.worst > args.feas_tol || frac > 0 || obj_mismatch) {
                verified = false;
                ++t.verify_failed;
                char buf[192];
                std::snprintf(buf, sizeof(buf),
                              "VERIFY-FAILED(residual=%.2g; %d fractional int; obj drift %.2g)",
                              r.worst, frac, obj_drift);
                std::printf("%-22s  WARNING: %s\n", name.c_str(), buf);
                note = buf;
            }
        }

        if (verified) {
            ++t.feasible;
            if (!std::isnan(gap_bks)) {
                // Claiming to beat a published MINLPLib bound needs a margin
                // that is actually meaningful. Two things set the floor:
                //   * relative floating-point noise on the objective value, and
                //   * the feasibility tolerance itself — we accept solutions
                //     violating a constraint by up to feas_tol, and that slack
                //     buys a small objective gain. A "win" at that scale is a
                //     tolerance artifact, not a better solution.
                // Anything inside the band is reported as a tie, so the
                // better-than-bks count means something and warrants scrutiny.
                // Two different bands, deliberately not the same number.
                //
                // `win_slack` is the margin an improvement must exceed to be
                // claimed: we accept solutions violating a constraint by up to
                // feas_tol, and that slack buys a small objective gain, so a
                // "win" at that scale is a tolerance artifact.
                //
                // `tie_band` is much tighter and purely relative — it is what it
                // takes to call two objectives *equal*. Reusing win_slack for
                // both published ex8_4_5 (BKS 3.07e-4) as "matches-bks" when it
                // was in fact 1.38% worse: there the absolute floor of
                // 10*feas_tol = 1e-5 dwarfs the objective's own magnitude.
                const double win_slack =
                    std::max(1e-6 * (std::abs(b.primal) + 1.0), 10.0 * args.feas_tol);
                const double tie_band = 1e-6 * (std::abs(b.primal) + 1.0);
                const double diff = obj - b.primal;                    // signed, in objective units
                const double improvement = maximizing ? diff : -diff;  // >0 is better
                // Three outcomes, not two. A row can improve on BKS by more than
                // the tie band yet less than win_slack: we will not claim that as a
                // win (the feasibility slack alone could buy it), but calling it
                // "worse than BKS" when its objective is better is simply false.
                // It gets its own label so the worse count means what it says.
                if (improvement > win_slack) {
                    ++t.better;
                    note = "better-than-bks";
                } else if (std::abs(diff) <= tie_band) {
                    ++t.matches;
                    note = "matches-bks";
                } else if (improvement > 0.0) {
                    ++t.within_tol;
                    note = "within-tolerance-of-bks";
                } else {
                    ++t.worse;
                    note = "feasible";
                }
            } else {
                note = "feasible";
            }
        } else if (result.feasible) {
            // Verify failed; `note` is already the VERIFY-FAILED string.
        } else {
            // Infeasible: report *where* the closest approach is still violated
            // and by how much, so the row distinguishes a numerical near-miss
            // from a search that never reached the feasible region. solve()
            // leaves the model at that closest-approach assignment.
            Residual r = worst_residual(prob, built, args.feas_tol);
            char buf[192];
            // nl_row is -1 both when the worst offender is a range row's
            // unrecorded lower half AND when nothing is violated at all — solve()
            // can report infeasible on a feasible point whose objective is
            // non-finite, since record_best refuses those. Don't name a range
            // row in the latter case.
            std::string row_label = "no violated row (non-finite objective)";
            if (r.nl_row >= 0) {
                row_label = "row" + std::to_string(r.nl_row) + " " + bound_type_name(r.row_type);
            } else if (r.worst > 0.0) {
                row_label = "range-lower-half";
            }
            // A non-finite objective at the closest approach is its own failure
            // mode, not generic hardness: the objective is folded in as an
            // `obj <= bound` soft constraint, so an infinite objective makes
            // that constraint's violation swamp the real ones and the search
            // loses the feasibility signal entirely. Call it out by name.
            if (built.objective_node_id >= 0 &&
                !std::isfinite(built.model.node(built.objective_node_id).value)) {
                ++t.nonfinite_obj;
                row_label += "; obj non-finite here";
            }
            if (r.worst <= kNearMiss) {
                ++t.near_miss;
                std::snprintf(buf, sizeof(buf),
                              "infeasible(near-miss residual=%.2g; %d viol; worst %s)", r.worst,
                              r.n_violated, row_label.c_str());
            } else {
                std::snprintf(buf, sizeof(buf), "infeasible(residual=%.2g; %d viol; worst %s)",
                              r.worst, r.n_violated, row_label.c_str());
            }
            note = buf;
            max_violation = r.worst;
        }
        if (!integrality_note.empty()) {
            note += "; " + integrality_note;
        }
        // Curated root-cause annotation, if this instance has one. Scoped to
        // rows we could NOT solve: analysis_notes.csv explains why an *unsolved*
        // instance is a solver defect or genuine hardness, so pasting that
        // verdict onto a row that now solves would publish a stale claim the
        // data itself contradicts. Warn loudly instead, so the note gets retired.
        {
            auto an = analysis_notes.find(name);
            // Only for rows that are actually infeasible: a VERIFY-FAILED row is
            // also !verified, but its failure is a solver-bookkeeping mismatch,
            // not the infeasibility mechanism the note describes.
            if (an != analysis_notes.end() && !result.feasible) {
                if (verified) {
                    std::printf("%-22s  WARNING: stale analysis note (now solved)\n", name.c_str());
                    note += "; stale-analysis-note";
                } else {
                    note += " | " + an->second;
                }
            }
        }
        std::replace(note.begin(), note.end(), ',', ';');

        // Console.
        if (verified) {
            std::printf("%12.4g ", obj);
        } else {
            std::printf("%12s ", "INFEAS");
        }
        if (b.have && !std::isnan(b.primal)) {
            std::printf("%12.4g ", b.primal);
        } else {
            std::printf("%12s ", "?");
        }
        if (std::isnan(gap_bks)) {
            std::printf("%10s ", "N/A");
        } else {
            std::printf("%9.2f%% ", gap_bks);
        }
        std::printf("%8.2fs  %s\n", wall, note.c_str());

        // CSV row.
        auto cell = [](double v) -> std::string {
            if (std::isnan(v)) {
                return "NaN";
            }
            std::ostringstream os;
            os << v;
            return os.str();
        };
        // A row that failed verification must not publish the objective or gaps
        // it was rejected for: those columns describe a solution we do not stand
        // behind. The note and max_violation still record what happened.
        const double pub_obj = verified ? obj : std::numeric_limits<double>::quiet_NaN();
        const double pub_gap_bks = verified ? gap_bks : std::numeric_limits<double>::quiet_NaN();
        const double pub_gap_dual = verified ? gap_dual : std::numeric_limits<double>::quiet_NaN();
        csv << name << "," << cell(pub_obj) << "," << cell(b.primal) << "," << cell(b.dual) << ","
            << cell(pub_gap_bks) << "," << cell(pub_gap_dual) << "," << wall << ","
            << (verified ? "true" : "false") << "," << note << "," << args.commit_sha << ","
            << cell(max_violation) << "," << prob.n_discrete_vars << "\n";
        csv.flush();
    }

    std::printf("\n=== Tally ===\n");
    std::printf("time limit:           %.0fs/instance, seed %llu, feas-tol %.0e\n", args.time_limit,
                static_cast<unsigned long long>(args.seed), args.feas_tol);
    std::printf("parsed:               %d\n", t.parsed);
    std::printf("closed (built):       %d\n", t.closed);
    std::printf("  mixed-integer:      %d  (integrality enforced)\n", t.mixed_integer);
    std::printf("feasible:             %d\n", t.feasible);
    std::printf("  better-than-BKS:    %d\n", t.better);
    std::printf("  matches BKS:        %d\n", t.matches);
    std::printf("  within tolerance:   %d  (better, but inside the tolerance slack)\n",
                t.within_tol);
    std::printf("  worse than BKS:     %d\n", t.worse);
    std::printf("infeasible:           %d\n", t.closed - t.feasible - t.failed_nonfinite);
    std::printf("  near-miss (<=%.0e): %d\n", kNearMiss, t.near_miss);
    std::printf("  non-finite obj:     %d  (objective +inf/NaN at closest approach)\n",
                t.nonfinite_obj);
    std::printf("failed(non-finite):   %d\n", t.failed_nonfinite);
    std::printf("skipped(unsupported): %d\n", t.skipped_unsupported);
    std::printf("read/build errors:    %d\n", t.errored);
    std::printf("not found:            %d\n", t.not_found);
    std::printf("integrality mismatch: %d  (NL header vs MINLPLib catalogue)\n",
                t.integrality_mismatch);
    std::printf("verify failed:        %d  (reported feasible; re-check disagreed)\n",
                t.verify_failed);
    // Closed-model rate over everything we attempted to read (present .nl files):
    // parsed + skipped-unsupported + errors. not_found excluded (no file).
    int attempted = t.parsed + t.skipped_unsupported + t.errored;
    if (attempted > 0) {
        std::printf("closed-model rate:    %.0f%% of %d attempted (%d not found)\n",
                    100.0 * t.closed / attempted, attempted, t.not_found);
    }
    std::printf("\nWrote %s\n", args.out_csv.c_str());
    return 0;
}
