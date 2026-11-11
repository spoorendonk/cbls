// MINLPLib non-convex benchmark runner.
//
// Loads each `.nl` from `benchmarks/instances/minlplib/`, builds a CBLS model
// via the NL-to-Model adapter, runs a fixed-time ViolationLS pass, and writes a
// per-instance row to `--out`. Unsupported operators and non-finite blowups are
// reported (not crashed). The roster and published bounds come from `bounds.csv`
// (written by download.py). The published `comparison.csv` is assembled by
// `run_benchmark.py`, never written here: it checks each verified row's
// `--solution-dir` assignment in SCIP first (#205; see resolve_out_csv).

#include <algorithm>
#include <array>
#include <benchmarks/common/published_guard.h>
#include <benchmarks/common/runner_args.h>
#include <benchmarks/common/runner_io.h>
#include <benchmarks/common/search_config_flags.h>
#include <benchmarks/minlplib/nl_refusal.h>
#include <benchmarks/minlplib/note_policy.h>
#include <benchmarks/mipfeas/worker_accounting.h>
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// The search-flag families this runner's scalar models can use (#198): with no
// List or Set variable a structural flag would be recorded and read by nothing,
// so validate_search_flags() refuses one.
constexpr unsigned kSearchFlagFamilies = cbls::bench::kScalarRunnerSearchFlags;

// The thread count every MINLPLib campaign runs at: Simon's decision of
// 2026-11-01, recorded on https://github.com/spoorendonk/cbls/issues/255 and
// applied to the full campaign, https://github.com/spoorendonk/cbls/issues/256.
constexpr int kCampaignThreads = 16;

struct Args {
    std::string inst_dir = "benchmarks/instances/minlplib";
    // 60s per instance: the previously published 5s is a stingy budget for a
    // general-purpose non-convex MINLP heuristic (#88).
    double time_limit = 60.0;
    bool time_limit_set = false;
    uint64_t seed = 1;
    bool seed_set = false;
    // Absolute violation tolerance for "feasible", stated because it is a
    // published property: the engine default, matching SCIP's numerics/feastol
    // so the SCIP baseline (#89) stays comparable.
    double feas_tol = cbls::kDefaultFeasibilityTolerance;
    bool feas_tol_set = false;
    std::vector<std::string> instances;  // optional override
    // Portfolio workers (#262): 1 is the committed table's single-threaded
    // `solve()`, >1 runs `ParallelSearch`; campaigns run at kCampaignThreads.
    // Another count is a different search, so each row records it in
    // `search_config` and a table never mixes counts.
    int threads = 1;
    bool commit_set = false;
    std::string commit_sha = "unknown";
    std::string out_csv;    // default: <inst_dir>/comparison.csv
    std::string trace_csv;  // optional: anytime profile (best objective vs time)
    // optional: one row of search counters and CPU time per searched instance
    // (#341), beside the table so the table's test-checked schema stays put
    std::string counters_csv;
    // optional: where each verified row's assignment is written, one
    // `<instance>.sol` per instance, for the DAG-independent check (#205)
    std::string solution_dir;
    // The ablation arm (#136); its canonical spelling is recorded on every row
    // this run writes, so a results file states its own configuration.
    cbls::bench::SearchFlags search;
    std::string search_config;  // search_config_string(search); filled in by parse_args
};

// Flag-value parsing: benchmarks/common/runner_args.h, shared with the other
// runners so a fix cannot diverge. A bad double is reported and returned as NaN
// for the guards below to exit 2 on; a bad integer is reported and exits 2.
using cbls::bench::parse_double;
using cbls::bench::parse_int64;

/// The numeric and search-configuration flags' value guards, apart from the
/// flag table: the parse layer reports a bad token and hands over NaN, the
/// guard decides the exit code.
void check_flag_values(Args& a) {
    std::string flag_error;
    if (!cbls::bench::validate_search_flags(a.search, kSearchFlagFamilies, a.time_limit_set,
                                            flag_error)) {
        std::fprintf(stderr, "%s\n", flag_error.c_str());
        std::exit(2);
    }
    a.search_config = cbls::bench::search_config_string(a.search, kSearchFlagFamilies);
    // The portfolio size is part of the row's configuration; written only off
    // 1, as the opt-in search flags are, so a `--threads 1` row's cell is the
    // one the committed tables hold.
    if (a.threads != 1) {
        a.search_config += (a.search_config.empty() ? "" : ";");
        a.search_config += "threads=" + std::to_string(a.threads);
    }
    if (a.search.no_time_limit) {
        // The wall-clock-free arm (#136): solve() disables its clock at
        // `time_limit <= 0`, making an iteration-budgeted run deterministic;
        // validate_search_flags has required a positive --max-iterations, so
        // the run still has a budget.
        a.time_limit = 0.0;
    } else if (!(a.time_limit > 0.0)) {
        std::fprintf(stderr, "--time-limit must be > 0 (got %g)\n", a.time_limit);
        std::exit(2);
    }
    if (!(a.feas_tol > 0.0)) {
        std::fprintf(stderr, "--feas-tol must be > 0 (got %g)\n", a.feas_tol);
        std::exit(2);
    }
}

/// The first flag that makes this run other than the full published
/// measurement, or nullptr: what `resolve_out_csv` refuses to publish under. A
/// partial roster would truncate the table; an ablation arm would republish it
/// as the default configuration's (#136); `--time-limit 1` would publish
/// one-second results (#88); `--seed`/`--feas-tol` change rows in a way no
/// column records. (`--commit` is required by check_published_write: an
/// "unknown" provenance cannot tell engine drift from a bug.)
const char* unpublishable_flag(const Args& a) {
    if (!a.instances.empty()) {
        return "--instance";
    }
    if (a.time_limit_set) {
        return "--time-limit";
    }
    if (a.seed_set) {
        return "--seed";
    }
    if (a.feas_tol_set) {
        return "--feas-tol";
    }
    if (a.threads != 1 && a.threads != kCampaignThreads) {
        // 1 is the committed table's count and kCampaignThreads the campaign
        // protocol's (#255, #256); any other count is neither, and a table
        // holds one thread count.
        return "--threads";
    }
    return cbls::bench::first_non_default_search_flag(a.search);
}

/// Fills in `--out` and enforces what may overwrite the published comparison
/// table: the shared guards (published_guard.h), then a refusal of every
/// publish from here.
void resolve_out_csv(Args& a) {
    if (a.out_csv.empty()) {
        a.out_csv = a.inst_dir + "/comparison.csv";
    }
    const std::optional<cbls::bench::PublishedTarget> target = cbls::bench::check_published_write(
        a.out_csv, a.trace_csv, a.inst_dir, unpublishable_flag(a), a.commit_set);
    if (!target.has_value()) {
        return;
    }
    // Every publish from here is refused (#205): `feasible` is this engine's
    // verdict on its own DAG (`worst_residual` reads the node values the search
    // optimised), so an evaluation bug both share goes uncaught.
    // `run_benchmark.py` checks each verified row's assignment in SCIP from the
    // `.nl` (independent_check.py) before it publishes.
    std::fprintf(stderr,
                 "the published %s %s is written only by benchmarks/minlplib/run_benchmark.py, "
                 "which checks every verified row outside this engine's DAG "
                 "(pass %s elsewhere for an unpublished run)\n",
                 target->artifact(), target->path.c_str(), target->redirect());
    std::exit(2);
}

/// Whether `file`, symlinks resolved, lies anywhere under directory `dir`.
bool lies_under(const std::string& file, const std::string& dir) {
    std::error_code ec;
    const std::filesystem::path f = std::filesystem::weakly_canonical(file, ec);
    if (ec) {
        return false;
    }
    const std::filesystem::path d = std::filesystem::weakly_canonical(dir, ec);
    if (ec) {
        return false;
    }
    auto fi = f.begin();
    for (auto di = d.begin(); di != d.end(); ++di, ++fi) {
        if (di->empty()) {  // a trailing separator's empty element
            continue;
        }
        if (fi == f.end() || *fi != *di) {
            return false;
        }
    }
    return fi != f.end();
}

/// `--counters` names a file of its own: not the table, its temp file, the
/// trace, nor anything under the instance or solution directory, which it would
/// truncate on open or have renamed over. The instance directory is refused
/// whole: the files read there (published table and trace, `bounds.csv`, the
/// SCIP baseline, the notes, `heldout/`) cannot be listed here without drifting
/// from the loaders. Symlinks are resolved; a hardlink under another name is
/// not caught.
void check_counters_path(const Args& a) {
    if (a.counters_csv.empty()) {
        return;
    }
    for (const std::string& dir : {a.inst_dir, a.solution_dir}) {
        if (!dir.empty() && lies_under(a.counters_csv, dir)) {
            std::fprintf(stderr, "--counters names %s, under %s, which the run owns\n",
                         a.counters_csv.c_str(), dir.c_str());
            std::exit(2);
        }
    }
    const std::vector<std::string> taken = {a.out_csv, a.out_csv + ".tmp"};
    for (const std::string& other : taken) {
        if (cbls::bench::same_file(a.counters_csv, other)) {
            std::fprintf(stderr, "--counters names %s, which another output owns\n", other.c_str());
            std::exit(2);
        }
    }
    if (!a.trace_csv.empty() && cbls::bench::same_file(a.counters_csv, a.trace_csv)) {
        std::fprintf(stderr, "--counters and --trace name the same file %s\n",
                     a.counters_csv.c_str());
        std::exit(2);
    }
}

Args parse_args(int argc, char** argv) {
    Args a;
    cbls::bench::ArgCursor c(argc, argv);
    const char* v = nullptr;
    while (c.advance()) {
        const std::string s = c.arg();
        if (c.value_flag("--time-limit", v)) {
            a.time_limit = parse_double("--time-limit", v);
            a.time_limit_set = true;
        } else if (c.value_flag("--seed", v)) {
            a.seed = static_cast<uint64_t>(parse_int64("--seed", v));
            a.seed_set = true;
        } else if (c.value_flag("--feas-tol", v)) {
            a.feas_tol = parse_double("--feas-tol", v);
            a.feas_tol_set = true;
        } else if (c.value_flag("--instance", v)) {
            a.instances.emplace_back(v);
        } else if (c.value_flag("--threads", v)) {
            a.threads = cbls::bench::parse_thread_count(v);
        } else if (c.value_flag("--commit", v)) {
            a.commit_sha = v;
            a.commit_set = true;
        } else if (c.value_flag("--out", v)) {
            a.out_csv = v;
        } else if (c.value_flag("--trace", v)) {
            a.trace_csv = v;
        } else if (c.value_flag("--counters", v)) {
            a.counters_csv = v;
        } else if (c.value_flag("--solution-dir", v)) {
            a.solution_dir = v;
        } else if (cbls::bench::match_search_flag(c, s, a.search)) {
            // A search-configuration flag (#136); the table, the parse and the
            // recorded spelling all live in benchmarks/common/search_config_flags.h.
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_minlplib [inst-dir] [--time-limit S] [--seed N]"
                " [--feas-tol T] [--instance NAME ...] [--threads N] [--commit SHA]"
                " [--out CSV] [--trace CSV] [--counters CSV] [--solution-dir DIR]\n"
                "       %s\n",
                cbls::bench::search_flags_usage(kSearchFlagFamilies).c_str());
            std::exit(0);
        } else if (s.rfind("--", 0) == 0) {
            // A typo'd flag must not become the instance directory and yield a
            // plausible but wrong published table.
            std::fprintf(stderr, "Unknown or incomplete option: %s (see --help)\n", s.c_str());
            std::exit(2);
        } else {
            a.inst_dir = s;
        }
    }
    check_flag_values(a);
    resolve_out_csv(a);
    check_counters_path(a);
    return a;
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
        // Flushed per row: a run is tens of minutes, and an interrupted one must
        // keep its profile. p.objective is the internally *minimised* value, so
        // a maximize instance is negated vs comparison.csv: improvement ratios
        // are invariant, raw values are not.
        out_ << instance_ << "," << p.time_seconds << "," << p.objective << ","
             << (p.new_best ? 1 : 0) << '\n';
        out_.flush();
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
    int errored = 0;  // read/build/solve exceptions; decides the exit status
    // Of `errored`, those that threw INSIDE solve(): the model was closed
    // first, so they are in `closed` too and must not be reported as
    // infeasibilities, the claim the nonzero exit exists to deny.
    int solve_errored = 0;
    // Solves whose portfolio completed fewer workers than `--threads` asked
    // for (#170): the row is written, but it is not the N-thread run it says.
    int lost_workers = 0;
    // Instances whose .nl file was present, counted once each: summing
    // `parsed + skipped_unsupported + errored` double-counts every outcome
    // after a successful read (build error, solve error, adapter-declined).
    int with_file = 0;
    int integrality_mismatch = 0;
    int verify_failed = 0;  // reported feasible but failed the independent re-check
    int near_miss = 0;      // infeasible, but residual within kNearMiss of feasible
    int nonfinite_obj = 0;  // infeasible with a non-finite objective at the closest approach
};

// Exit status for a run that reached the end of its roster with errors: 1 (no
// roster to run, or an exception escaping main) and 2 (a bad flag or an
// unwritable output file, benchmarks/common/runner_args.h) are taken, so a
// consumer can tell the cases apart. Mirrored as Python
// `runner.RUNNER_EXIT_ERRORED`, pinned to this literal by a test because the
// drivers branch on the value.
constexpr int kExitErrored = 3;

/// The process exit status for a run that reached the end of its roster.
/// `errored` (`Tally::errored`: a read, build or solve that THREW) is
/// deliberately the only input: `skipped_unsupported` and `not_found` are
/// coverage gaps, bucketed apart so a gap is not reported as a failure, and
/// `failed_nonfinite`/`verify_failed` are measurements of a completed search.
int run_exit_status(int errored) {
    return errored > 0 ? kExitErrored : 0;
}

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
    // Always `built`'s own model, so a mismatched one cannot be passed.
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
        double v = model.node_value(cids[i]);
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

// Signed gap in percent, POSITIVE meaning "worse than the reference" in either
// objective sense (unflipped, a maximize row's 64% miss reads as a 64% gain).
// When |ref| < 1e-12 it returns an ABSOLUTE residual, not a percent; the CSV does
// not mark these, so consumers must not bucket them as percentages and the
// README lists them (mathopt1, prob09, least).
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

using BoundsMap = std::unordered_map<std::string, Bounds>;
using NotesMap = std::unordered_map<std::string, std::string>;

// Per-instance work is split by the question each step answers, each with its
// own policy and failure mode: can it be read and built, is the returned
// assignment really a solution, where does a verified objective stand against
// the published bound, why is an unverified one not one. run_instance asks
// them in order; run_benchmark owns only the roster, output files and tally.

/// A CSV cell for a double: NaN spelled the way the readers of this table
/// expect, rather than the stream's locale-dependent "nan".
std::string cell(double v) {
    if (std::isnan(v)) {
        return "NaN";
    }
    std::ostringstream os;
    os << v;
    return os.str();
}

/// `cell` at a precision that round-trips a double, for a value the analysis
/// COMPARES: the #149 analysis buckets an instance `arrival-invariant` (evidence
/// against the effect) when every seed's first feasible objective is EQUAL, and
/// at six figures 1234567.1 and 1234567.9 would print alike and manufacture it.
std::string precise_cell(double v) {
    if (std::isnan(v)) {
        return "NaN";
    }
    std::ostringstream os;
    os << std::setprecision(17) << v;
    return os.str();
}

/// The row's two LNS counter cells: destroy-repairs ATTEMPTED and the subset the
/// accept rule KEPT. Both default to "NaN" for a row where no solve completed:
/// a 0 would claim "LNS ran and never repaired" (or "all rolled back"), the
/// reading that gates the #143 LNS arm. EVERY row writer, write_preread_row
/// included, goes through this pair, so neither cell gets a number alone.
struct LnsCells {
    std::string attempted = "NaN";
    std::string accepted = "NaN";
};

/// The counters of a completed solve, as cells.
LnsCells lns_cells(const cbls::SearchResult& result) {
    return {std::to_string(result.lns_repairs), std::to_string(result.lns_repairs_accepted)};
}

/// The row's first-feasible pair (#149): the objective at the first feasible
/// point the search recorded and the seconds to reach it; a pair through every
/// writer for LnsCells' reason. Both default to "NaN" where no search
/// completed: a 0 would claim "feasible instantly, at objective zero" and enter
/// the #149 correlation. `seconds` says whether a point was recorded at all;
/// `objective` can be NaN or inf on a row that was (#100), so a reader must
/// exclude non-finite objectives.
struct FirstFeasibleCells {
    std::string objective = "NaN";
    std::string seconds = "NaN";
};

/// The first-feasible pair of a completed solve, as cells. `maximizing`
/// un-negates the objective as the row's `objective` cell is (`solve()`
/// minimises). No `isfinite` guard, so a blown-up objective keeps the sign its
/// magnitude implies; no `result.feasible` guard: a run that never reached
/// feasibility latched nothing, both cells are NaN, and `-NaN` is NaN.
FirstFeasibleCells first_feasible_cells(const cbls::SearchResult& result, bool maximizing) {
    double obj = result.first_feasible_objective;
    if (maximizing) {
        obj = -obj;
    }
    // The objective at full precision, the time at display precision: only the
    // objective is compared for equality by the bucket rule.
    return {precise_cell(obj), cell(result.time_to_first_feasible)};
}

/// A row for an instance whose bounds have not been looked up: every numeric
/// cell NaN. Still written: bounds.csv is the roster of record, and the table
/// must not silently disagree with it.
void write_preread_row(std::ostream& csv, const Args& args, const std::string& name,
                       const std::string& note) {
    // Through the cell types like every writer (LnsCells), not literal NaNs.
    const LnsCells lns;
    const FirstFeasibleCells ff;
    csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false," << note << "," << args.commit_sha << ",NaN,NaN,"
        << lns.attempted << "," << lns.accepted << "," << ff.objective << "," << ff.seconds << ","
        << args.search_config << "\n";
}

/// The returned assignment of a DAG-verified row, for the DAG-independent check
/// (#205): `benchmarks/minlplib/independent_check.py` loads the `.nl` in SCIP and
/// checks these values, so an evaluation bug shared with `worst_residual` (same
/// DAG node values) cannot publish a `feasible` row. One full-precision value per
/// NL column in order; `objective` is the true-sense value at full precision (the
/// CSV cell has six digits). Temp file then rename, so a crash leaves no half
/// file; an unwritable directory exits 2, as an unwritable `--out` does, since a
/// verified row without its solution cannot be checked.
void write_solution(const Args& args, const std::string& name, const cbls::NlToModelResult& built,
                    double obj) {
    if (args.solution_dir.empty()) {
        return;
    }
    const std::string path = args.solution_dir + "/" + name + ".sol";
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        out << "cbls-minlplib-solution 1\n"
            << "instance " << name << "\n"
            << "objective " << precise_cell(obj) << "\n"
            << "columns " << built.var_handles.size() << "\n";
        for (const int32_t h : built.var_handles) {
            out << precise_cell(built.model.var(cbls::handle_to_var_id(h)).value) << "\n";
        }
        out.flush();
        if (!out) {
            std::fprintf(stderr, "cannot write solution file %s\n", tmp.c_str());
            std::exit(2);
        }
    }
    if (!cbls::bench::rename_into_place(tmp, path)) {
        std::exit(2);
    }
}

/// A row for an instance that was built but produced no publishable objective.
void write_unsolved_row(std::ostream& csv, const Args& args, const std::string& name,
                        const Bounds& b, double wall, const std::string& note, int n_discrete,
                        const LnsCells& lns, const FirstFeasibleCells& ff) {
    csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN," << wall << ",false," << note
        << "," << args.commit_sha << ",NaN," << n_discrete << "," << lns.attempted << ","
        << lns.accepted << "," << ff.objective << "," << ff.seconds << "," << args.search_config
        << "\n";
}

/// Reads the instance's .nl. Returns false having written the row and bumped the
/// tally when it cannot be read: an unknown opcode or unsupported segment is a
/// coverage gap, not a malformed file, so it is bucketed as skipped(unsupported)
/// rather than as an error.
bool read_instance(std::ostream& csv, const Args& args, const std::string& name,
                   const std::string& nl_path, cbls::NlProblem& prob, Tally& t) {
    try {
        prob = cbls::read_nl(nl_path);
    } catch (const std::exception& e) {
        std::string what = e.what();
        const bool unsupported = cbls::minlplib::is_reader_coverage_gap(what);
        std::replace(what.begin(), what.end(), ',', ';');
        if (unsupported) {
            std::printf("%-22s  (skipped: %s)\n", name.c_str(), what.c_str());
            ++t.skipped_unsupported;
            write_preread_row(csv, args, name, "unsupported: " + what);
        } else {
            std::printf("%-22s  ERROR reading: %s\n", name.c_str(), what.c_str());
            ++t.errored;
            write_preread_row(csv, args, name, "read-error");
        }
        return false;
    }
    ++t.parsed;
    return true;
}

/// Integrality cross-check: if the NL header's discrete-column count (placed by
/// Gay's variable ordering) disagrees with MINLPLib's nbinvars+nintvars, the model
/// is not the instance the published bound refers to; say so rather than report
/// a gap. Called only on a row that will be written, so every counted mismatch
/// has a row explaining it.
std::string integrality_check(const cbls::NlProblem& prob, const Bounds& b, const std::string& name,
                              Tally& t) {
    if (b.n_disc < 0 || b.n_disc == prob.n_discrete_vars) {
        return "";
    }
    ++t.integrality_mismatch;
    std::string note = "integrality-mismatch(nl=" + std::to_string(prob.n_discrete_vars) +
                       " catalogue=" + std::to_string(b.n_disc) + ")";
    std::printf("%-22s  WARNING: %s\n", name.c_str(), note.c_str());
    return note;
}

/// Independent re-check of the returned assignment. solve() restores best_state
/// and full-evaluates, so this re-derives feasibility and integrality from the
/// model rather than trusting the search's own bookkeeping. A reported-feasible
/// row that fails here is a solver bug, and must not be published as a solved
/// instance. Returns false having set `note` in that case.
bool verify_assignment(const cbls::NlProblem& prob, const cbls::NlToModelResult& built,
                       const Args& args, double obj, const std::string& name, Tally& t,
                       std::string& note, double& max_violation) {
    Residual r = worst_residual(prob, built, args.feas_tol);
    max_violation = r.worst;
    int frac = 0;
    for (const auto& v : built.model.variables()) {
        if (v.type == cbls::VarType::Int && std::abs(v.value - std::round(v.value)) > 1e-9) {
            ++frac;
        }
    }
    // The published objective must also be the one the model reports at the
    // returned assignment, not just the search's running best.
    double model_obj =
        built.objective_node_id >= 0 ? built.model.node_value(built.objective_node_id) : obj;
    if (built.model.is_maximizing()) {
        model_obj = -model_obj;  // same un-negation applied to `obj` by the caller
    }
    const double obj_drift = std::abs(model_obj - obj);
    const bool obj_mismatch = obj_drift > 1e-6 * (std::abs(obj) + 1.0);

    if (r.worst <= args.feas_tol && frac == 0 && !obj_mismatch) {
        return true;
    }
    ++t.verify_failed;
    std::array<char, 192> buf{};
    std::snprintf(buf.data(), buf.size(),
                  "VERIFY-FAILED(residual=%.2g; %d fractional int; obj drift %.2g)", r.worst, frac,
                  obj_drift);
    std::printf("%-22s  WARNING: %s\n", name.c_str(), buf.data());
    note = buf.data();
    return false;
}

/// Where a verified objective stands against the published BKS. A claimed win
/// must exceed `win_slack`, floored by relative float noise and by feas_tol (an
/// accepted violation buys a small objective gain): inside it a "win" is a
/// tolerance artifact, so better-than-bks stays worth scrutiny. `tie_band` is
/// much tighter and purely relative: what it takes to call two objectives
/// *equal*. Must not reuse win_slack as the tie band, else ex8_4_5 (BKS
/// 3.07e-4, 1.38% worse) reads matches-bks under the 10*feas_tol = 1e-5 floor.
/// An improvement inside win_slack but beyond tie_band is not claimed as a win,
/// nor is it "worse": it is within-tolerance-of-bks, so the worse count is true.
std::string classify_against_bks(double obj, const Bounds& b, bool maximizing, double feas_tol,
                                 Tally& t) {
    const double win_slack = std::max(1e-6 * (std::abs(b.primal) + 1.0), 10.0 * feas_tol);
    const double tie_band = 1e-6 * (std::abs(b.primal) + 1.0);
    const double diff = obj - b.primal;                    // signed, in objective units
    const double improvement = maximizing ? diff : -diff;  // >0 is better
    if (improvement > win_slack) {
        ++t.better;
        return "better-than-bks";
    }
    if (std::abs(diff) <= tie_band) {
        ++t.matches;
        return "matches-bks";
    }
    if (improvement > 0.0) {
        ++t.within_tol;
        return "within-tolerance-of-bks";
    }
    ++t.worse;
    return "feasible";
}

/// Why an infeasible row is infeasible: *where* the closest approach is still
/// violated and by how much, so the row distinguishes a numerical near-miss from
/// a search that never reached the feasible region. solve() leaves the model at
/// that closest-approach assignment.
std::string describe_infeasible(const cbls::NlProblem& prob, const cbls::NlToModelResult& built,
                                const Args& args, Tally& t, double& max_violation) {
    Residual r = worst_residual(prob, built, args.feas_tol);
    max_violation = r.worst;
    // nl_row is -1 both when the worst offender is a range row's unrecorded
    // lower half AND when nothing is violated at all -- solve() can report
    // infeasible on a feasible point whose objective is non-finite, since
    // record_best refuses those. Don't name a range row in the latter case.
    std::string row_label = "no violated row (non-finite objective)";
    if (r.nl_row >= 0) {
        row_label = "row" + std::to_string(r.nl_row) + " " + bound_type_name(r.row_type);
    } else if (r.worst > 0.0) {
        row_label = "range-lower-half";
    }
    // A non-finite objective at the closest approach is its own failure mode:
    // folded in as `obj <= bound`, an infinite objective swamps the real
    // violations and the search loses the feasibility signal. Named in the row.
    if (built.objective_node_id >= 0 &&
        !std::isfinite(built.model.node_value(built.objective_node_id))) {
        ++t.nonfinite_obj;
        row_label += "; obj non-finite here";
    }
    std::array<char, 192> buf{};
    if (r.worst <= kNearMiss) {
        ++t.near_miss;
        std::snprintf(buf.data(), buf.size(),
                      "infeasible(near-miss residual=%.2g; %d viol; worst %s)", r.worst,
                      r.n_violated, row_label.c_str());
    } else {
        std::snprintf(buf.data(), buf.size(), "infeasible(residual=%.2g; %d viol; worst %s)",
                      r.worst, r.n_violated, row_label.c_str());
    }
    return buf.data();
}

/// Curated root-cause annotation, if any. analysis_notes.csv explains why an
/// *unsolved* instance is a solver defect or genuine hardness, so a row that now
/// solves gets a loud stale warning, not a verdict its own data contradicts.
void merge_analysis_note(const NotesMap& analysis_notes, const std::string& name,
                         bool result_feasible, bool verified, std::string& note) {
    auto an = analysis_notes.find(name);
    // The three-way split is note_policy.h's, testable without a solve.
    using cbls::minlplib::NoteAction;
    switch (cbls::minlplib::note_action(an != analysis_notes.end(), result_feasible, verified)) {
        case NoteAction::kStale:
            std::printf("%-22s  WARNING: stale analysis note (now solved)\n", name.c_str());
            note += "; stale-analysis-note";
            break;
        case NoteAction::kMerge:
            note += " | " + an->second;
            break;
        case NoteAction::kNone:
            // No note, or feasible but VERIFY-FAILED (see note_policy.h).
            break;
    }
}

void print_console_row(const Bounds& b, bool verified, double obj, double gap_bks, double wall,
                       const std::string& note) {
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
}

/// Everything before a solve: the .nl present, read, built and supported.
/// Returns false having written the row and bumped the tally when one fails;
/// each outcome is its own tally bucket and row shape, hence answered here
/// rather than in the scoring below.
bool prepare_instance(std::ostream& csv, const Args& args, const std::string& name,
                      const BoundsMap& bounds, cbls::NlProblem& prob, cbls::NlToModelResult& built,
                      Bounds& b, Tally& t) {
    std::string nl_path = args.inst_dir + "/" + name + ".nl";
    if (!cbls::bench::file_exists(nl_path)) {
        std::printf("%-22s  (skipped: %s not found)\n", name.c_str(), nl_path.c_str());
        ++t.not_found;
        write_preread_row(csv, args, name, "not-found");
        return false;
    }
    ++t.with_file;

    if (!read_instance(csv, args, name, nl_path, prob, t)) {
        return false;
    }

    try {
        built = cbls::nl_to_model(prob);
    } catch (const std::exception& e) {
        std::printf("%-22s  ERROR building model: %s\n", name.c_str(), e.what());
        ++t.errored;
        write_preread_row(csv, args, name, "build-error");
        return false;
    }

    auto bit = bounds.find(name);
    b = bit != bounds.end() ? bit->second : Bounds{};

    if (!built.supported) {
        std::string note = built.skipped_reasons.empty()
                               ? "unsupported"
                               : "unsupported: " + built.skipped_reasons[0];
        // Sanitise commas in the note so the CSV stays well-formed.
        std::replace(note.begin(), note.end(), ',', ';');
        std::printf("%-22s  (skipped: %s)\n", name.c_str(), note.c_str());
        ++t.skipped_unsupported;
        write_unsolved_row(csv, args, name, b, 0.0, note, prob.n_discrete_vars, LnsCells{},
                           FirstFeasibleCells{});
        return false;
    }
    ++t.closed;
    // Counted here, not earlier, so the tally's mixed-integer count is a subset
    // of the instances actually built (as the printout implies).
    if (prob.n_discrete_vars > 0) {
        ++t.mixed_integer;
    }

    return true;
}

/// The search itself: the single-threaded `solve()` at `--threads 1`, the
/// `ParallelSearch` master overload above it. Either way the model is left
/// holding the state the row scores.
cbls::SearchResult search_instance(const Args& args, cbls::Model& model,
                                   cbls::SolveCallback* callback) {
    cbls::SearchConfig cfg;
    cfg.feasibility_tolerance = args.feas_tol;
    // The ablation arm. Everything the flags touch is applied here and nowhere
    // else, so the `search_config` cell on the row describes exactly the
    // configuration this call received (#136).
    cbls::bench::apply_search_flags(args.search, cfg);
    const double time_limit = cbls::bench::time_limit_argument(args.search, args.time_limit);
    if (args.threads == 1) {
        cbls::FloatIntensifyHook hook;
        cbls::LNS lns(0.3);
        return cbls::solve(model, time_limit, args.seed, /*use_fj=*/true,
                           cbls::bench::hook_argument(args.search, hook),
                           cbls::bench::lns_argument(args.search, lns),
                           cbls::bench::lns_interval_argument(args.search), callback, cfg);
    }
    // One hook and one LNS per worker: the search locks neither, so a shared
    // instance would be a data race (pool.h). The callback is the portfolio's
    // ordered stream, invoked under the pool's mutex.
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
        ps.solve(model, time_limit, args.seed, cfg, hook_factory, lns_factory, callback, par);
    // The workers searched copies, so the master still holds its start: restore
    // the returned best, which is the state the row's residual check and `.sol`
    // read. Guarded: a portfolio whose deadline expired before any worker ran
    // returns no state.
    if (result.best_state.values.size() == model.num_vars()) {
        model.restore_state(result.best_state);
        cbls::full_evaluate(model);
    }
    return result;
}

/// Process CPU seconds: at `--threads 1` the search's own CPU time, which is
/// what #341's throughput ratios divide by (wall time is noisy on a shared
/// machine). At more threads it sums the workers.
double process_cpu_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return static_cast<double>(ts.tv_sec) + (1e-9 * static_cast<double>(ts.tv_nsec));
}

/// What a search cost the process, read as it returns: CPU seconds around it
/// and the peak RSS so far (KiB).
struct SearchCost {
    double cpu_seconds = 0.0;
    long peak_rss_kb = 0;
};

/// `search_instance`, measured (#341).
cbls::SearchResult measured_search(const Args& args, cbls::Model& model,
                                   cbls::SolveCallback* callback, SearchCost& cost) {
    const double cpu0 = process_cpu_seconds();
    cbls::SearchResult result = search_instance(args, model, callback);
    cost.cpu_seconds = process_cpu_seconds() - cpu0;
    cost.peak_rss_kb = cbls::bench::peak_rss_kib();
    return result;
}

/// Under `--counters`, one row of what the search did (#341): iterations and
/// batches, the Novelty counters, CPU time and peak RSS
/// (per instance only with one instance per call). Counters are summed over
/// workers. Written only once the solve bracket has succeeded, so a
/// solve-error row has no counters row.
void write_counters_row(std::ofstream& counters, const Args& args, const std::string& name,
                        const cbls::SearchResult& result, const SearchCost& cost) {
    if (!counters.is_open()) {
        return;
    }
    const cbls::SearchCounters& k = result.counters;
    counters << name << "," << result.iterations << "," << k.fj_batches << "," << k.novelty_batches
             << "," << k.novelty_moves << "," << cost.cpu_seconds << "," << cost.peak_rss_kb << ","
             << args.search_config << "\n";
    counters.flush();
}

/// Read, build, solve and score one instance, and write its row. Every exit
/// writes exactly one row.
void run_instance(std::ostream& csv, std::ofstream& trace, std::ofstream& counters,
                  const Args& args, const BoundsMap& bounds, const NotesMap& analysis_notes,
                  const std::string& name, Tally& t) {
    cbls::NlProblem prob;
    cbls::NlToModelResult built;
    Bounds b;
    if (!prepare_instance(csv, args, name, bounds, prob, built, b, t)) {
        return;
    }
    std::string note;
    double max_violation = 0.0;  // closest-approach residual on infeasible rows

    const std::string integrality_note = integrality_check(prob, b, name, t);

    std::printf("%-22s ", name.c_str());
    std::fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    auto t1 = t0;
    cbls::SearchResult result;
    SearchCost cost;
    try {
        // Test seam for the solve-error path, which needs a throwing solve (the
        // engine no longer throws on an empty model, #202). Only
        // `tests/python/test_minlplib_run_benchmark.py` sets it.
        if (const char* boom = std::getenv("CBLS_MINLPLIB_THROW_IN_SOLVE");
            boom != nullptr && name == boom) {
            throw std::runtime_error("CBLS_MINLPLIB_THROW_IN_SOLVE names this instance");
        }
        TraceRecorder recorder(trace, name);
        result = measured_search(args, built.model, trace.is_open() ? &recorder : nullptr, cost);
        t1 = std::chrono::steady_clock::now();
        if (cbls::mipfeas::lost_workers(result, args.threads)) {
            std::fputs(cbls::mipfeas::lost_workers_message(name, args.threads, result).c_str(),
                       stderr);
            ++t.lost_workers;
        }
    } catch (const std::exception& e) {
        std::printf(" ERROR solving: %s\n", e.what());
        ++t.errored;
        ++t.solve_errored;
        write_unsolved_row(csv, args, name, b, 0.0, "solve-error", prob.n_discrete_vars, LnsCells{},
                           FirstFeasibleCells{});
        return;
    }
    write_counters_row(counters, args, name, result, cost);
    double wall = std::chrono::duration<double>(t1 - t0).count();

    // solve() reports the *minimised* objective. For a maximize instance the
    // model objective was negated, so un-negate to recover the true value and
    // make gap-to-BKS comparable to the published (max-sense) bound.
    const bool maximizing = built.model.is_maximizing();
    double obj = result.feasible ? result.objective : std::numeric_limits<double>::quiet_NaN();
    if (result.feasible && maximizing) {
        obj = -obj;
    }
    // Filled once, from the one `result`, and handed to whichever row writer
    // below turns out to be the one that runs.
    const FirstFeasibleCells ff = first_feasible_cells(result, maximizing);

    // A feasible-but-non-finite objective means the guard fired: count it as
    // failed(non-finite), not feasible.
    if (result.feasible && !std::isfinite(obj)) {
        ++t.failed_nonfinite;
        note = "non-finite";
        if (!integrality_note.empty()) {
            // else the tally reports a mismatch that no CSV row explains
            note += "; " + integrality_note;
        }
        std::printf("%12s %12.4g %10s %8.2fs  %s\n", "NONFIN", b.primal, "N/A", wall, note.c_str());
        write_unsolved_row(csv, args, name, b, wall, note, prob.n_discrete_vars, lns_cells(result),
                           ff);
        return;
    }

    double gap_bks = safe_gap(obj, b.primal, maximizing);
    double gap_dual = safe_gap(obj, b.dual, maximizing);

    const bool verified =
        result.feasible && verify_assignment(prob, built, args, obj, name, t, note, max_violation);

    if (verified) {
        ++t.feasible;
        note = std::isnan(gap_bks) ? "feasible"
                                   : classify_against_bks(obj, b, maximizing, args.feas_tol, t);
    } else if (result.feasible) {
        // Verify failed; `note` is already the VERIFY-FAILED string.
    } else {
        note = describe_infeasible(prob, built, args, t, max_violation);
    }
    if (!integrality_note.empty()) {
        note += "; " + integrality_note;
    }
    merge_analysis_note(analysis_notes, name, result.feasible, verified, note);
    std::replace(note.begin(), note.end(), ',', ';');

    print_console_row(b, verified, obj, gap_bks, wall, note);

    // A row that failed verification must not publish the objective or gaps it
    // was rejected for: those columns describe a solution we do not stand behind.
    // The note and max_violation still record what happened.
    const double pub_obj = verified ? obj : std::numeric_limits<double>::quiet_NaN();
    const double pub_gap_bks = verified ? gap_bks : std::numeric_limits<double>::quiet_NaN();
    const double pub_gap_dual = verified ? gap_dual : std::numeric_limits<double>::quiet_NaN();
    if (verified) {
        write_solution(args, name, built, obj);  // before the row: no row without its solution
    }
    const LnsCells solved = lns_cells(result);
    csv << name << "," << cell(pub_obj) << "," << cell(b.primal) << "," << cell(b.dual) << ","
        << cell(pub_gap_bks) << "," << cell(pub_gap_dual) << "," << wall << ","
        << (verified ? "true" : "false") << "," << note << "," << args.commit_sha << ","
        << cell(max_violation) << "," << prob.n_discrete_vars << "," << solved.attempted << ","
        << solved.accepted << "," << ff.objective << "," << ff.seconds << "," << args.search_config
        << "\n";
    csv.flush();
}

void print_tally(const Args& args, const Tally& t) {
    std::printf("\n=== Tally ===\n");
    // "0s/instance" would read as a bug rather than as the clock-free arm, and
    // the tally is what a reader checks a run against.
    if (args.search.no_time_limit) {
        std::printf("time limit:           none (--no-time-limit), seed %llu, feas-tol %.0e\n",
                    static_cast<unsigned long long>(args.seed), args.feas_tol);
    } else {
        std::printf("time limit:           %.0fs/instance, seed %llu, feas-tol %.0e\n",
                    args.time_limit, static_cast<unsigned long long>(args.seed), args.feas_tol);
    }
    std::printf("search config:        %s\n", args.search_config.c_str());
    std::printf("parsed:               %d\n", t.parsed);
    std::printf("closed (built):       %d\n", t.closed);
    std::printf("  mixed-integer:      %d  (integrality enforced)\n", t.mixed_integer);
    std::printf("feasible:             %d\n", t.feasible);
    std::printf("  better-than-BKS:    %d\n", t.better);
    std::printf("  matches BKS:        %d\n", t.matches);
    std::printf("  within tolerance:   %d  (better, but inside the tolerance slack)\n",
                t.within_tol);
    std::printf("  worse than BKS:     %d\n", t.worse);
    // A solve that threw is subtracted: its model was closed, but no search
    // ran, so calling it infeasible would contradict the row's own note and the
    // nonzero exit status this run reports.
    std::printf("infeasible:           %d\n",
                t.closed - t.feasible - t.failed_nonfinite - t.solve_errored);
    std::printf("  near-miss (<=%.0e): %d\n", kNearMiss, t.near_miss);
    std::printf("  non-finite obj:     %d  (objective +inf/NaN at closest approach)\n",
                t.nonfinite_obj);
    std::printf("failed(non-finite):   %d\n", t.failed_nonfinite);
    std::printf("skipped(unsupported): %d\n", t.skipped_unsupported);
    std::printf("read/build/solve err: %d  (exit %d when nonzero)\n", t.errored, kExitErrored);
    std::printf("not found:            %d\n", t.not_found);
    std::printf("integrality mismatch: %d  (NL header vs MINLPLib catalogue)\n",
                t.integrality_mismatch);
    std::printf("verify failed:        %d  (reported feasible; re-check disagreed)\n",
                t.verify_failed);
    // Closed-model rate over everything we attempted to read (present .nl
    // files), counted once per instance. not_found excluded (no file).
    int attempted = t.with_file;
    if (attempted > 0) {
        std::printf("closed-model rate:    %.0f%% of %d attempted (%d not found)\n",
                    100.0 * t.closed / attempted, attempted, t.not_found);
    }
    std::printf("\nWrote %s\n", args.out_csv.c_str());
}

int run_benchmark(int argc, char** argv) {
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

    std::ofstream counters;
    if (!args.counters_csv.empty()) {
        counters.open(args.counters_csv);
        if (!counters.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", args.counters_csv.c_str());
            return 2;
        }
        counters << "instance,iterations,fj_batches,novelty_batches,novelty_moves,"
                    "cpu_seconds,"
                    "peak_rss_kb,search_config\n";
    }

    std::ofstream csv(args.out_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", args.out_csv.c_str());
        return 2;
    }
    // `search_config` stays last and is on EVERY row, early exits included: a
    // short row makes the column count depend on how the instance failed, and
    // a row with no configuration cannot be reproduced (#136). The run's
    // counters go before it: `lns_repairs`/`lns_repairs_accepted` say whether
    // LNS did anything (#143) and whether any was kept (#150);
    // `first_feasible_objective`/`time_to_first_feasible` (#149) say where the
    // run ARRIVED in the feasible region, so the first-feasible/final
    // correlation is computable from an ordinary campaign. Recording them
    // changes no trajectory (see `SearchResult::first_feasible_objective`).
    csv << "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,"
           "wall_seconds,feasible,note,commit_sha,max_violation,n_int_vars,lns_repairs,"
           "lns_repairs_accepted,first_feasible_objective,time_to_first_feasible,"
           "search_config\n";

    std::printf("\n%-22s %12s %12s %10s %9s  %s\n", "Instance", "Objective", "BKS", "Gap%",
                "Time(s)", "Note");
    std::printf("%-22s %12s %12s %10s %9s  %s\n", "--------", "---------", "---", "----", "-------",
                "----");

    Tally t;
    for (const std::string& name : insts) {
        run_instance(csv, trace, counters, args, bounds, analysis_notes, name, t);
    }

    print_tally(args, t);
    const int status = run_exit_status(t.errored);
    if (status != 0) {
        // On stderr and not only in the tally: the exit status is what an
        // automated consumer reads, and a driver that records the row still has
        // to be able to say WHY the process failed.
        std::fprintf(stderr,
                     "%d instance(s) threw while being read, built or solved; exiting %d. "
                     "Those rows carry read-error/build-error/solve-error and record no "
                     "search -- they are not infeasibility results.\n",
                     t.errored, status);
    }
    // As cbls_mipfeas and cbls_cvrp: a portfolio that lost workers measured
    // fewer threads than its rows claim, so the job failed (#170).
    if (status == 0 && t.lost_workers > 0) {
        std::fprintf(stderr, "%d solve(s) lost portfolio workers; exiting 1\n", t.lost_workers);
        return 1;
    }
    return status;
}

}  // namespace

int main(int argc, char** argv) {
    // An exception escaping main is std::terminate, an abort with no message
    // that looks like a crash: report it and exit 1 (bugprone-exception-escape).
    try {
        return run_benchmark(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
