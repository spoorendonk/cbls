// CVRP runner (#197): one CVRPLIB X instance per process, the way mipfeas runs.
//
//   cbls_cvrp --instance X-n101-k25 --budget 10 --out r.json --solution-out r.sol
//   cbls_cvrp --instance X-n101-k25 --threads 4 --neighbours 20
//             --structural-selection violation_guided
//
// The model is Hexaly's List encoding (cvrp_model.h); the data and its pinned
// BKS are benchmarks/instances/cvrp/. Every reported solution is re-checked by
// verify_cvrp.h, which recomputes the routes' cost from the instance rather than
// trusting the DAG, and only a verified solution is written to --solution-out.
//
// Smoke and campaign runs only. No result table is written by this runner;
// publishing is #204's job. The engine's default structural arm is not feasible
// within 10 s on the largest smoke instances; the smoke check uses
// `--structural-selection violation_guided --neighbours 20` (see
// benchmarks/instances/cvrp/README.md, "Search arm for the smoke check").

#include "cvrp_model.h"
#include "data.h"
#include "verify_cvrp.h"

#include <benchmarks/common/runner_args.h>
#include <benchmarks/common/search_config_flags.h>
#include <benchmarks/mipfeas/worker_accounting.h>
#include <cbls/cbls.h>
#include <cbls/lns.h>
#include <cbls/pool.h>
#include <cbls/search.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The search-flag families this runner's models can use (#198). A CVRP model is
// Lists only, so FJ has nothing to jump: the float hook, compound moves and FJ's
// unproductive exit would be recorded and read by nothing, and are refused. The
// runner supplies the distance as the neighbour cost.
constexpr unsigned kSearchFlagFamilies = cbls::bench::kCommonSearchFlags |
                                         cbls::bench::kStructuralSearchFlags |
                                         cbls::bench::kNeighbourSearchFlags;

/// Upper bound on `--threads`; mipfeas's reasoning: a typo'd 1000 would be an
/// out-of-memory kill rather than an error message.
constexpr int kMaxThreads = 256;

struct Args {
    std::string instance;
    std::string inst_dir = "benchmarks/instances/cvrp";
    std::string out_path;
    std::string solution_out;
    double budget = 10.0;
    bool budget_set = false;
    uint64_t seed = 1;
    int threads = 1;
    int fleet = 0;  // 0 = default_fleet(k)
    std::string commit_sha = "unknown";
    cbls::bench::SearchFlags search;
    std::string search_config;
};

void print_usage() {
    std::printf(
        "Usage: cbls_cvrp --instance NAME [--inst-dir DIR] [--out JSON] [--solution-out SOL]\n"
        "                 [--budget SECONDS] [--seed N] [--threads N] [--fleet R]\n"
        "                 [--commit SHA]\n"
        "                 %s\n",
        cbls::bench::search_flags_usage(kSearchFlagFamilies).c_str());
}

using cbls::bench::parse_double;
using cbls::bench::parse_int64;

int parse_bounded_int(const char* flag, const char* text, int lo, int hi) {
    const int64_t v = parse_int64(flag, text);
    if (v < lo || v > hi) {
        std::fprintf(stderr, "%s must be in [%d, %d]\n", flag, lo, hi);
        std::exit(2);
    }
    return static_cast<int>(v);
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
        } else if (c.value_flag("--out", v)) {
            a.out_path = v;
        } else if (c.value_flag("--solution-out", v)) {
            a.solution_out = v;
        } else if (c.value_flag("--budget", v)) {
            a.budget = parse_double("--budget", v);
            a.budget_set = true;
        } else if (c.value_flag("--seed", v)) {
            a.seed = static_cast<uint64_t>(parse_int64("--seed", v));
        } else if (c.value_flag("--threads", v)) {
            a.threads = parse_bounded_int("--threads", v, 1, kMaxThreads);
        } else if (c.value_flag("--fleet", v)) {
            a.fleet = parse_bounded_int("--fleet", v, 1, std::numeric_limits<int>::max());
        } else if (c.value_flag("--commit", v)) {
            a.commit_sha = v;
        } else if (cbls::bench::match_search_flag(c, s, a.search)) {
            // A search-configuration flag; see search_config_flags.h.
        } else if (s == "--help" || s == "-h") {
            print_usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s (see --help)\n", s.c_str());
            std::exit(2);
        }
    }
    return a;
}

/// The value guards. Returns 0 when the arguments name a runnable job, else the
/// exit code.
int validate_args(Args& a) {
    if (a.instance.empty()) {
        std::fprintf(stderr, "--instance is required\n");
        return 2;
    }
    std::string error;
    if (!cbls::bench::validate_search_flags(a.search, kSearchFlagFamilies, a.budget_set, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    a.search_config = cbls::bench::search_config_string(a.search, kSearchFlagFamilies);
    // NaN (a malformed --budget) fails this as well as a non-positive value: a
    // solve with no budget returns at once with an empty result.
    if (!a.search.no_time_limit && !(a.budget > 0.0)) {
        std::fprintf(stderr, "--budget must be a positive number of seconds\n");
        return 2;
    }
    return 0;
}

struct Bks {
    double value = std::numeric_limits<double>::quiet_NaN();
    std::string kind;
};

/// The pinned BKS from `<inst-dir>/roster.csv`, or NaN when the instance is not
/// on it (an instance from elsewhere is still runnable; it reports no gap).
Bks read_bks(const std::string& inst_dir, const std::string& instance) {
    Bks out;
    std::ifstream in(inst_dir + "/roster.csv");
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream ss(line);
        std::string name;
        std::string bks;
        std::string kind;
        if (std::getline(ss, name, ',') && name == instance && std::getline(ss, bks, ',') &&
            std::getline(ss, kind, ',')) {
            out.value = std::strtod(bks.c_str(), nullptr);
            out.kind = kind;
        }
    }
    return out;
}

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

/// Writes `text` to `path` through a temp file and a rename, so a killed job
/// leaves the previous file or none, never half of one.
bool write_atomically(const std::string& path, const std::string& text) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream out(tmp);
        if (!out.is_open()) {
            std::fprintf(stderr, "Failed to open %s for writing\n", tmp.c_str());
            return false;
        }
        out << text;
        if (!out.good()) {
            std::fprintf(stderr, "Failed to write %s\n", tmp.c_str());
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::fprintf(stderr, "Failed to rename %s -> %s: %s\n", tmp.c_str(), path.c_str(),
                     ec.message().c_str());
        return false;
    }
    return true;
}

/// CVRPLIB's `.sol` format: one `Route #i:` line per non-empty route, customer
/// ids 1-based (TSPLIB node minus one, the depot being node 1), then `Cost N`.
std::string solution_text(const std::vector<std::vector<int>>& routes, double cost) {
    std::ostringstream out;
    int k = 0;
    for (const auto& route : routes) {
        if (route.empty()) {
            continue;
        }
        out << "Route #" << ++k << ":";
        for (const int e : route) {
            out << ' ' << (e + 1);
        }
        out << '\n';
    }
    out << "Cost " << std::llround(cost) << '\n';
    return out.str();
}

/// The solve, single-threaded or through the portfolio. Either way the model is
/// left holding the returned best state, which is what read_routes reads.
cbls::SearchResult run_search(const Args& args, cbls::cvrp::CvrpModel& cm,
                              const cbls::SearchConfig& cfg) {
    const double time_limit = cbls::bench::time_limit_argument(args.search, args.budget);
    cbls::SearchResult result;
    if (args.threads == 1) {
        cbls::LNS lns;
        result = cbls::solve(cm.model, time_limit, args.seed, /*use_fj=*/true, /*hook=*/nullptr,
                             cbls::bench::lns_argument(args.search, lns),
                             cbls::bench::lns_interval_argument(args.search), nullptr, cfg);
    } else {
        // The master overload: workers share the frozen model's structure (#157).
        // LNS is per worker -- the search locks neither it nor the hook.
        std::function<std::shared_ptr<cbls::LNS>()> lns_factory;
        if (args.search.lns) {
            lns_factory = [] { return std::make_shared<cbls::LNS>(); };
        }
        cbls::ParallelConfig par;
        par.n_threads = args.threads;
        cbls::ParallelSearch ps(args.threads);
        cbls::SearchConfig worker_cfg = cfg;
        worker_cfg.lns_interval = cbls::bench::lns_interval_argument(args.search);
        result = ps.solve(cm.model, time_limit, args.seed, worker_cfg, nullptr, lns_factory,
                          nullptr, par);
    }
    // The portfolio searched copies, so the master still holds its start; the
    // single-threaded solve leaves its final point, not necessarily the best.
    // Restore the returned best into the model either way. Guarded: a portfolio
    // whose deadline expired before any worker ran returns an empty state.
    if (result.best_state.values.size() == cm.model.num_vars()) {
        cm.model.restore_state(result.best_state);
        cbls::full_evaluate(cm.model);
    }
    return result;
}

nlohmann::json nullable(double v) {
    return std::isfinite(v) ? nlohmann::json(v) : nlohmann::json(nullptr);
}

int run_benchmark(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (const int rc = validate_args(args); rc != 0) {
        return rc;
    }

    const std::string vrp_path = args.inst_dir + "/X/" + args.instance + ".vrp";
    cbls::cvrp::CvrpInstance inst;
    try {
        inst = cbls::cvrp::read_vrp_file(vrp_path);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", vrp_path.c_str(), e.what());
        return 2;
    }
    const int n = inst.customers();
    std::string error;
    if (!cbls::bench::validate_neighbours(args.search, n, error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    const int k = cbls::cvrp::name_k(inst.name);
    if (args.fleet == 0 && k <= 0) {
        std::fprintf(stderr, "%s: the name carries no k; pass --fleet R\n", inst.name.c_str());
        return 2;
    }
    const int fleet = args.fleet > 0 ? args.fleet : cbls::cvrp::default_fleet(k);
    const Bks bks = read_bks(args.inst_dir, args.instance);

    // A previous run's solution at this path must not survive a run that
    // produces none: a `.sol` beside a record saying `verified=false` would be
    // checked as if this run had written it.
    if (!args.solution_out.empty()) {
        std::error_code rm_ec;
        std::filesystem::remove(args.solution_out, rm_ec);
        if (rm_ec) {
            std::fprintf(stderr, "Cannot remove the old %s: %s\n", args.solution_out.c_str(),
                         rm_ec.message().c_str());
            return 2;
        }
    }

    const auto t_build = std::chrono::steady_clock::now();
    cbls::cvrp::CvrpModel cm = cbls::cvrp::build_model(inst, fleet);
    const double build_seconds = seconds_since(t_build);

    cbls::SearchConfig cfg;
    cbls::bench::apply_search_flags(args.search, cfg);
    const std::shared_ptr<const cbls::cvrp::DistanceMatrix> dist = cm.dist;
    cfg.structural_neighbours = cbls::bench::neighbours_argument(
        args.search, n, [dist](int a, int b) { return dist->customers(a, b); });

    const auto t0 = std::chrono::steady_clock::now();
    const cbls::SearchResult result = run_search(args, cm, cfg);
    const double wall = seconds_since(t0);

    const std::vector<std::vector<int>> routes = cbls::cvrp::read_routes(cm);
    const cbls::cvrp::RouteCheck check = cbls::cvrp::check_routes(inst, routes, result.objective);
    const bool verified = result.feasible && check.ok;
    for (const auto& e : check.errors) {
        if (result.feasible) {
            std::fprintf(stderr, "%s: VERIFY FAILED: %s\n", args.instance.c_str(), e.c_str());
        }
    }

    bool write_failed = false;
    if (verified && !args.solution_out.empty()) {
        write_failed = !write_atomically(args.solution_out, solution_text(routes, check.cost));
    }

    const double objective = verified ? check.cost : std::numeric_limits<double>::quiet_NaN();
    nlohmann::json j{
        {"engine", "cbls"},
        {"instance", args.instance},
        {"customers", n},
        {"capacity", inst.capacity},
        {"bks", nullable(bks.value)},
        {"bks_kind", bks.kind},
        {"objective", nullable(objective)},
        {"gap_percent", nullable(100.0 * (objective - bks.value) / bks.value)},
        {"feasible", result.feasible},
        {"verified", verified},
        {"verify_errors", result.feasible ? check.errors : std::vector<std::string>{}},
        {"routes_used", check.routes_used},
        {"fleet", fleet},
        {"k_name", k},
        {"budget_seconds", args.search.no_time_limit ? 0.0 : args.budget},
        {"seed", args.seed},
        {"threads", args.threads},
        {"wall_seconds", wall},
        {"build_seconds", build_seconds},
        {"time_to_first_feasible", nullable(result.time_to_first_feasible)},
        {"iterations", result.iterations},
        {"batches", result.counters.batches},
        {"fj_batches", result.counters.fj_batches},
        {"novelty_batches", result.counters.novelty_batches},
        {"structural_batches", result.counters.structural_batches},
        {"structural_moves_tried", result.counters.structural_moves_tried},
        {"structural_moves_accepted", result.counters.structural_moves_accepted},
        {"perturbations", result.perturbations},
        {"lns_repairs", result.lns_repairs},
        {"lns_repairs_accepted", result.lns_repairs_accepted},
        {"commit_sha", args.commit_sha},
        {"search_config", args.search_config},
    };
    cbls::mipfeas::add_worker_accounting(result, j);
    if (!args.out_path.empty() && !write_atomically(args.out_path, j.dump(2) + "\n")) {
        write_failed = true;
    }
    std::fputs(cbls::mipfeas::lost_workers_message(args.instance, args.threads, result).c_str(),
               stderr);

    std::printf(
        "%-14s fleet=%-4d feasible=%d verified=%d obj=%-10.0f bks=%-10.0f gap=%7.2f%% "
        "routes=%d %.2fs\n",
        args.instance.c_str(), fleet, result.feasible ? 1 : 0, verified ? 1 : 0, objective,
        bks.value, 100.0 * (objective - bks.value) / bks.value, check.routes_used, wall);
    // A claimed-feasible solution that fails the check is the failure this
    // verifier exists for. An infeasible run is a result, not an error.
    const bool verify_failed = result.feasible && !check.ok;
    return (verify_failed || write_failed || cbls::mipfeas::lost_workers(result, args.threads)) ? 1
                                                                                                : 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_benchmark(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
