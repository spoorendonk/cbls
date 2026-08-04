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
    // 60s per instance. The previously published run used 5s, which is a stingy
    // budget for a general-purpose non-convex MINLP heuristic (issue #88).
    double time_limit = 60.0;
    uint64_t seed = 1;
    // Absolute constraint-violation tolerance for "feasible". 1e-6 matches
    // SCIP's numerics/feastol default, which is the right reference point for a
    // continuous/nonlinear roster and keeps the SCIP baseline (#89) comparable.
    // The engine default (1e-9) is unreachable for equality rows whose bodies
    // are large in magnitude, since the violation is the absolute |lhs - rhs|.
    double feas_tol = 1e-6;
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
        } else if (s == "--help" || s == "-h") {
            std::printf(
                "Usage: cbls_minlplib [inst-dir] [--time-limit S] [--seed N]"
                " [--feas-tol T] [--instance NAME ...] [--commit SHA] [--out CSV]\n");
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
    int mixed_integer = 0;  // instances with >=1 integer column (integrality enforced)
    int failed_nonfinite = 0;
    int skipped_unsupported = 0;
    int not_found = 0;
    int errored = 0;         // read/build/solve exceptions
    int integrality_mismatch = 0;
    int verify_failed = 0;  // reported feasible but failed the independent re-check
    int near_miss = 0;       // infeasible, but residual within kNearMiss of feasible
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

Residual worst_residual(const cbls::Model& model, const cbls::NlProblem& prob,
                        const cbls::NlToModelResult& built, double tol) {
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

        // Integrality cross-check: the NL header declares how many columns are
        // discrete, and Gay's variable ordering places them. If that disagrees
        // with MINLPLib's own nbinvars+nintvars, the model we just built is not
        // the instance the published bound refers to — say so rather than
        // reporting a gap against a bound for a different problem.
        const bool mixed_integer = prob.n_discrete_vars > 0;
        if (mixed_integer) {
            ++t.mixed_integer;
        }
        std::string integrality_note;
        if (b.n_disc >= 0 && b.n_disc != prob.n_discrete_vars) {
            ++t.integrality_mismatch;
            integrality_note = "integrality-mismatch(nl=" + std::to_string(prob.n_discrete_vars) +
                               " catalogue=" + std::to_string(b.n_disc) + ")";
            std::printf("%-22s  WARNING: %s\n", name.c_str(), integrality_note.c_str());
        }

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

        std::printf("%-22s ", name.c_str());
        std::fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        cbls::FloatIntensifyHook hook;
        cbls::LNS lns(0.3);
        cbls::SearchConfig cfg;
        cfg.feasibility_tolerance = args.feas_tol;
        cbls::SearchResult result;
        try {
            result = cbls::solve(built.model, args.time_limit, args.seed,
                                 /*use_fj=*/true, &hook, &lns, /*lns_interval=*/3,
                                 /*callback=*/nullptr, cfg);
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
            std::printf("%12s %12.4g %10s %8.2fs  %s\n", "NONFIN", b.primal, "N/A", wall,
                        note.c_str());
            csv << name << ",NaN," << b.primal << "," << b.dual << ",NaN,NaN," << wall << ",false,"
                << note << "," << args.commit_sha << ",NaN," << prob.n_discrete_vars << "\n";
            continue;
        }

        double gap_bks = safe_gap(obj, b.primal);
        double gap_dual = safe_gap(obj, b.dual);

        // Independent re-check of the returned assignment. solve() restores
        // best_state and full-evaluates, so this re-derives feasibility and
        // integrality from the model rather than trusting the search's own
        // bookkeeping. A reported-feasible row that fails here is a solver bug,
        // and must not be published as a solved instance.
        bool verified = result.feasible;
        if (result.feasible) {
            Residual r = worst_residual(built.model, prob, built, args.feas_tol);
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
        } else if (result.feasible) {
            // Verify failed; `note` is already the VERIFY-FAILED string.
        } else {
            // Infeasible: report *where* the closest approach is still violated
            // and by how much, so the row distinguishes a numerical near-miss
            // from a search that never reached the feasible region. solve()
            // leaves the model at that closest-approach assignment.
            Residual r = worst_residual(built.model, prob, built, args.feas_tol);
            char buf[192];
            std::string row_label =
                r.nl_row >= 0 ? "row" + std::to_string(r.nl_row) + " " + bound_type_name(r.row_type)
                              : "range-lower-half";
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
        csv << name << "," << cell(obj) << "," << cell(b.primal) << "," << cell(b.dual) << ","
            << cell(gap_bks) << "," << cell(gap_dual) << "," << wall << ","
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
    std::printf("  worse/equal:        %d\n", t.worse);
    std::printf("infeasible:           %d\n", t.closed - t.feasible - t.failed_nonfinite);
    std::printf("  near-miss (<=%.0e): %d\n", kNearMiss, t.near_miss);
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
