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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct Args {
    std::string inst_dir = "benchmarks/instances/minlplib";
    double time_limit = 10.0;
    std::vector<std::string> instances;  // optional override
    std::string commit_sha = "unknown";
    std::string out_csv;  // default: <inst_dir>/comparison.csv
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--time-limit" && i + 1 < argc) {
            a.time_limit = std::atof(argv[++i]);
        } else if (s == "--instance" && i + 1 < argc) {
            a.instances.emplace_back(argv[++i]);
        } else if (s == "--commit" && i + 1 < argc) {
            a.commit_sha = argv[++i];
        } else if (s == "--out" && i + 1 < argc) {
            a.out_csv = argv[++i];
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_minlplib [inst-dir] [--time-limit S]"
                " [--instance NAME ...] [--commit SHA] [--out CSV]\n");
            std::exit(0);
        } else {
            a.inst_dir = s;
        }
    }
    if (a.out_csv.empty()) {
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
};

// Parse bounds.csv: instance,structure,nvars,ncons,objsense,primal_bks,dual_bound.
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
        out[cells[0]] = b;
    }
    return out;
}

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
    int worse = 0;
    int failed_nonfinite = 0;
    int skipped_unsupported = 0;
    int not_found = 0;
};

double safe_gap(double obj, double ref) {
    if (std::isnan(obj) || std::isnan(ref)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double denom = std::abs(ref);
    if (denom < 1e-12) {
        return obj - ref;  // ref ~ 0: report absolute residual
    }
    return 100.0 * (obj - ref) / denom;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    std::string bounds_path = args.inst_dir + "/bounds.csv";
    auto bounds = load_bounds(bounds_path);

    std::vector<std::string> insts = args.instances;
    if (insts.empty()) {
        insts = roster_from_bounds(bounds_path);
        if (insts.empty()) {
            std::printf("WARNING: %s missing and no --instance given; nothing to run.\n",
                        bounds_path.c_str());
            return 1;
        }
    }

    std::ofstream csv(args.out_csv);
    if (!csv.is_open()) {
        std::fprintf(stderr, "Failed to open %s for writing\n", args.out_csv.c_str());
        return 2;
    }
    csv << "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,"
           "wall_seconds,feasible,note,commit_sha\n";

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
            continue;
        }

        cbls::NlProblem prob;
        try {
            prob = cbls::read_nl(nl_path);
        } catch (const std::exception& e) {
            std::printf("%-22s  ERROR reading: %s\n", name.c_str(), e.what());
            csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false,read-error," << args.commit_sha << "\n";
            continue;
        }
        ++t.parsed;

        cbls::NlToModelResult built;
        std::string note;
        try {
            built = cbls::nl_to_model(prob);
        } catch (const std::exception& e) {
            std::printf("%-22s  ERROR building model: %s\n", name.c_str(), e.what());
            csv << name << ",NaN,NaN,NaN,NaN,NaN,0,false,build-error," << args.commit_sha << "\n";
            continue;
        }

        Bounds b = bounds.count(name) ? bounds[name] : Bounds{};

        if (!built.supported) {
            note = built.skipped_reasons.empty() ? "unsupported"
                                                 : "unsupported: " + built.skipped_reasons[0];
            // Sanitise commas in the note so the CSV stays well-formed.
            std::replace(note.begin(), note.end(), ',', ';');
            std::printf("%-22s  (skipped: %s)\n", name.c_str(), note.c_str());
            ++t.skipped_unsupported;
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN,0,false," << note
                << "," << args.commit_sha << "\n";
            continue;
        }
        ++t.closed;

        std::printf("%-22s ", name.c_str());
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        cbls::FloatIntensifyHook hook;
        cbls::LNS lns(0.3);
        cbls::SearchResult result;
        try {
            result = cbls::solve(built.model, args.time_limit, /*seed=*/42,
                                 /*use_fj=*/true, &hook, &lns);
        } catch (const std::exception& e) {
            std::printf(" ERROR solving: %s\n", e.what());
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN,0,false,solve-error,"
                << args.commit_sha << "\n";
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
            std::printf("%12s %12.4g %10s %8.2fs  %s\n", "NONFIN", b.primal, "N/A", wall,
                        note.c_str());
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN," << wall << ",false,"
                << note << "," << args.commit_sha << "\n";
            continue;
        }

        double gap_bks = safe_gap(obj, b.primal);
        double gap_dual = safe_gap(obj, b.dual);

        if (result.feasible) {
            ++t.feasible;
            if (!std::isnan(gap_bks)) {
                // "Better than BKS" is sense-aware: a smaller objective beats a
                // min-sense BKS, a larger one beats a max-sense BKS. gap_bks is
                // signed obj-vs-BKS, so the beat condition flips with the sense.
                const bool beats_bks =
                    built.model.is_maximizing() ? (gap_bks > 1e-6) : (gap_bks < -1e-6);
                if (beats_bks) {
                    ++t.better;
                    note = "better-than-bks";
                } else {
                    ++t.worse;
                    note = "feasible";
                }
            } else {
                note = "feasible";
            }
        } else {
            note = "infeasible";
        }

        // Console.
        if (result.feasible) {
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
        csv << name << "," << cell(obj) << "," << cell(b.primal) << "," << cell(b.dual) << ","
            << cell(gap_bks) << "," << cell(gap_dual) << "," << wall << ","
            << (result.feasible ? "true" : "false") << "," << note << "," << args.commit_sha
            << "\n";
        csv.flush();
    }

    std::printf("\n=== Tally ===\n");
    std::printf("parsed:               %d\n", t.parsed);
    std::printf("closed (built):       %d\n", t.closed);
    std::printf("feasible:             %d\n", t.feasible);
    std::printf("  better-than-BKS:    %d\n", t.better);
    std::printf("  worse/equal:        %d\n", t.worse);
    std::printf("failed(non-finite):   %d\n", t.failed_nonfinite);
    std::printf("skipped(unsupported): %d\n", t.skipped_unsupported);
    std::printf("not found:            %d\n", t.not_found);
    int considered = t.parsed + t.not_found;
    if (considered > 0) {
        std::printf("closed-model rate:    %.0f%% of %d considered\n",
                    100.0 * t.closed / considered, considered);
    }
    std::printf("\nWrote %s\n", args.out_csv.c_str());
    return 0;
}
