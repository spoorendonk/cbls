// cbls_scip_traj: solve one problem file with the pinned SCIP and log every
// incumbent as (CLOCK_MONOTONIC seconds, original-space objective), so a
// comparison can score SCIP by primal integral (#442). Concurrent at N > 1
// threads (SCIPsolveConcurrent, master's defaults), SCIPsolve at N = 1.
//
//   cbls_scip_traj PROBLEM --threads N --time-limit S --traj-out TRAJ.jsonl
//       --result-out RESULT.json [--set PARAMS.set] [--partial-sol START.sol]
//       [--sol-out BEST.sol]
//
// Refuses to run unless libscip's git hash is the commit CMake read from
// SCIP_COMMIT beside this file (CBLS_SCIP_PINNED_COMMIT).

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <new>
#include <nlohmann/json.hpp>
#include <optional>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>
#include <scip/scipgithash.h>
#include <string>
#include <vector>

#ifndef CBLS_SCIP_PINNED_COMMIT
#error "CBLS_SCIP_PINNED_COMMIT must name the SCIP commit (CMake reads SCIP_COMMIT)"
#endif

namespace {

constexpr const char* kEventName = "cbls_traj";
constexpr const char* kEventDesc = "logs every incumbent with its monotonic time (#442)";

double monotonic_seconds() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<double>(ts.tv_sec) + (static_cast<double>(ts.tv_nsec) * 1e-9);
}

// Shared by the main SCIP's handler and every copy's: one file, one lock.
struct TrajLog {
    std::mutex mu;
    FILE* out = nullptr;
    SCIP* main = nullptr;
    bool concurrent = false;  // N > 1: SCIPsolveConcurrent
    std::atomic<bool> solvers_copied{false};
    int next_solver = 1;
    int records = 0;
};

constexpr int kSilent = -1;

struct HandlerData {
    TrajLog* log = nullptr;
    int solver = kSilent;  // 0: the main SCIP; k >= 1: the k-th concurrent solver; else silent
};

HandlerData* handler_data(SCIP_EVENTHDLR* eventhdlr) {
    return reinterpret_cast<HandlerData*>(SCIPeventhdlrGetData(eventhdlr));
}

// Called from SCIP's C frames, which no exception may cross: printf, not json.
// Stamped under the lock, so the file is in time order across threads.
SCIP_RETCODE write_record(TrajLog& log, int solver, std::optional<double> objective) {
    try {
        const std::scoped_lock lock(log.mu);
        const double mono_s = monotonic_seconds();
        const int n =
            objective.has_value()
                ? std::fprintf(log.out, "{\"mono_s\":%.17g,\"solver\":%d,\"objective\":%.17g}\n",
                               mono_s, solver, *objective)
                : std::fprintf(log.out, "{\"mono_s\":%.17g,\"solver\":%d,\"objective\":null}\n",
                               mono_s, solver);
        if (n < 0 || std::fflush(log.out) != 0) {
            return SCIP_WRITEERROR;
        }
        ++log.records;
    } catch (...) {
        return SCIP_ERROR;
    }
    return SCIP_OKAY;
}

SCIP_RETCODE include_traj_handler(SCIP* scip, TrajLog* log, int solver);

// A copy logs only if it is a concurrent solver: copied from the main while the
// main sits at PRESOLVED, which is SCIPsolveConcurrent setting its solvers up
// (scip_solve.c:3269 on). Depth cannot tell: a concurrent solver's is reset to
// 0 (concurrent.c:91), and dualval builds its sub-SCIP at depth 0 too.
SCIP_DECL_EVENTCOPY(event_copy_traj) {
    const HandlerData* source = handler_data(eventhdlr);
    TrajLog* log = source->log;
    int solver = kSilent;
    if (source->solver == 0 && log->concurrent && SCIPgetStage(log->main) == SCIP_STAGE_PRESOLVED) {
        try {
            const std::scoped_lock lock(log->mu);
            solver = log->next_solver++;
            log->solvers_copied = true;
        } catch (...) {
            return SCIP_ERROR;
        }
    }
    SCIP_CALL(include_traj_handler(scip, log, solver));
    return SCIP_OKAY;
}

SCIP_DECL_EVENTFREE(event_free_traj) {
    (void)scip;
    delete handler_data(eventhdlr);
    SCIPeventhdlrSetData(eventhdlr, nullptr);
    return SCIP_OKAY;
}

// Caught at INIT, not INITSOL, so solutions found while presolving count too.
SCIP_DECL_EVENTINIT(event_init_traj) {
    if (handler_data(eventhdlr)->solver != kSilent) {
        SCIP_CALL(SCIPcatchEvent(scip, SCIP_EVENTTYPE_BESTSOLFOUND, eventhdlr, nullptr, nullptr));
    }
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXIT(event_exit_traj) {
    if (handler_data(eventhdlr)->solver != kSilent) {
        SCIP_CALL(SCIPdropEvent(scip, SCIP_EVENTTYPE_BESTSOLFOUND, eventhdlr, nullptr, -1));
    }
    return SCIP_OKAY;
}

SCIP_DECL_EVENTEXEC(event_exec_traj) {
    (void)eventdata;
    const HandlerData* data = handler_data(eventhdlr);
    SCIP* main = data->log->main;
    // Once the concurrent solvers exist the main stays silent: after their solve
    // it re-adds their solutions (concsolver_scip.c:925), already logged.
    if (scip == main && data->log->solvers_copied) {
        return SCIP_OKAY;
    }
    // A copy's original space is the main's transformed space (the concurrent
    // solvers copy the transformed problem), so its objective is retransformed.
    // Infinity is tested in the copy's own space: retransforming scales it.
    double objective = SCIPgetSolOrigObj(scip, SCIPeventGetSol(event));
    const bool finite = std::isfinite(objective) && !SCIPisInfinity(scip, std::abs(objective));
    if (scip != main) {
        objective = SCIPretransformObj(main, objective);
    }
    return write_record(*data->log, data->solver,
                        finite ? std::optional<double>(objective) : std::nullopt);
}

SCIP_RETCODE include_traj_handler(SCIP* scip, TrajLog* log, int solver) {
    auto* data = new (std::nothrow) HandlerData{log, solver};
    if (data == nullptr) {
        return SCIP_NOMEMORY;
    }
    SCIP_EVENTHDLR* eventhdlr = nullptr;
    const SCIP_RETCODE rc =
        SCIPincludeEventhdlrBasic(scip, &eventhdlr, kEventName, kEventDesc, event_exec_traj,
                                  reinterpret_cast<SCIP_EVENTHDLRDATA*>(data));
    if (rc != SCIP_OKAY) {
        delete data;
        return rc;
    }
    SCIP_CALL(SCIPsetEventhdlrCopy(scip, eventhdlr, event_copy_traj));
    SCIP_CALL(SCIPsetEventhdlrFree(scip, eventhdlr, event_free_traj));
    SCIP_CALL(SCIPsetEventhdlrInit(scip, eventhdlr, event_init_traj));
    SCIP_CALL(SCIPsetEventhdlrExit(scip, eventhdlr, event_exit_traj));
    return SCIP_OKAY;
}

// The names match PySCIPOpt's getStatus(), so both backends report alike.
const char* status_name(SCIP_STATUS status) {
    switch (status) {
        case SCIP_STATUS_UNKNOWN:
            return "unknown";
        case SCIP_STATUS_USERINTERRUPT:
            return "userinterrupt";
        case SCIP_STATUS_NODELIMIT:
            return "nodelimit";
        case SCIP_STATUS_TOTALNODELIMIT:
            return "totalnodelimit";
        case SCIP_STATUS_STALLNODELIMIT:
            return "stallnodelimit";
        case SCIP_STATUS_TIMELIMIT:
            return "timelimit";
        case SCIP_STATUS_MEMLIMIT:
            return "memlimit";
        case SCIP_STATUS_GAPLIMIT:
            return "gaplimit";
        case SCIP_STATUS_PRIMALLIMIT:
            return "primallimit";
        case SCIP_STATUS_DUALLIMIT:
            return "duallimit";
        case SCIP_STATUS_SOLLIMIT:
            return "sollimit";
        case SCIP_STATUS_BESTSOLLIMIT:
            return "bestsollimit";
        case SCIP_STATUS_RESTARTLIMIT:
            return "restartlimit";
        case SCIP_STATUS_OPTIMAL:
            return "optimal";
        case SCIP_STATUS_INFEASIBLE:
            return "infeasible";
        case SCIP_STATUS_UNBOUNDED:
            return "unbounded";
        case SCIP_STATUS_INFORUNBD:
            return "inforunbd";
        case SCIP_STATUS_TERMINATE:
            return "terminate";
    }
    return "unknown";
}

struct Args {
    std::string problem;
    std::string settings;
    std::string partial_sol;
    std::string sol_out;
    std::string traj_out;
    std::string result_out;
    int threads = 0;
    double time_limit = -1.0;
};

// 0 (refused by parse_args) unless the whole string is a count in [1, 1024].
int parse_threads(const std::string& s) {
    char* end = nullptr;
    const long n = std::strtol(s.c_str(), &end, 10);
    return (*end == '\0' && n >= 1 && n <= 1024) ? static_cast<int>(n) : 0;
}

// -1 (refused by parse_args) unless the whole, non-empty string is a finite number.
double parse_seconds(const std::string& s) {
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    return (end != s.c_str() && *end == '\0' && std::isfinite(v)) ? v : -1.0;
}

std::optional<Args> parse_args(int argc, char** argv) {
    Args a;
    const std::vector<std::string> v(argv + 1, argv + argc);
    for (size_t i = 0; i < v.size(); ++i) {
        const std::string& k = v[i];
        if (k.rfind("--", 0) != 0) {
            if (!a.problem.empty()) {
                std::cerr << "second problem file " << k << "\n";
                return std::nullopt;
            }
            a.problem = k;
            continue;
        }
        if (i + 1 >= v.size()) {
            std::cerr << k << " needs a value\n";
            return std::nullopt;
        }
        const std::string& val = v[++i];
        if (k == "--set") {
            a.settings = val;
        } else if (k == "--partial-sol") {
            a.partial_sol = val;
        } else if (k == "--sol-out") {
            a.sol_out = val;
        } else if (k == "--traj-out") {
            a.traj_out = val;
        } else if (k == "--result-out") {
            a.result_out = val;
        } else if (k == "--threads") {
            a.threads = parse_threads(val);
        } else if (k == "--time-limit") {
            a.time_limit = parse_seconds(val);
        } else {
            std::cerr << "unknown option " << k << "\n";
            return std::nullopt;
        }
    }
    if (a.problem.empty() || a.traj_out.empty() || a.result_out.empty() || a.threads < 1 ||
        a.time_limit < 0.0) {
        std::cerr << "usage: cbls_scip_traj PROBLEM --threads N --time-limit S --traj-out F "
                     "--result-out F [--set F] [--partial-sol F] [--sol-out F]\n";
        return std::nullopt;
    }
    return a;
}

SCIP_RETCODE load(SCIP* scip, const Args& a) {
    SCIP_CALL(SCIPincludeDefaultPlugins(scip));
    if (!a.settings.empty()) {
        SCIP_CALL(SCIPreadParams(scip, a.settings.c_str()));
    }
    SCIP_CALL(SCIPreadProb(scip, a.problem.c_str(), nullptr));
    return SCIP_OKAY;
}

// Reads a .sol file into a partial solution: variables it does not list stay
// unknown, as PySCIPOpt's createPartialSol leaves them.
SCIP_RETCODE add_partial_sol(SCIP* scip, const std::string& path) {
    SCIP_SOL* sol = nullptr;
    SCIP_CALL(SCIPcreatePartialSol(scip, &sol, nullptr));
    SCIP_Bool partial = FALSE;
    SCIP_Bool error = FALSE;
    const SCIP_RETCODE rc = SCIPreadSolFile(scip, path.c_str(), sol, FALSE, &partial, &error);
    if (rc != SCIP_OKAY || error != FALSE) {
        SCIP_CALL(SCIPfreeSol(scip, &sol));
        std::cerr << "cannot read partial solution " << path << "\n";
        return rc != SCIP_OKAY ? rc : SCIP_READERROR;
    }
    SCIP_Bool stored = FALSE;
    SCIP_CALL(SCIPaddSolFree(scip, &sol, &stored));
    return SCIP_OKAY;
}

// A concurrent solver gets only the main's complete best solution
// (concsolver_scip.c:636), never a partial one, and the main never runs
// completesol at N > 1. So the start is completed first by SCIP's own
// completesol: an auxiliary single-thread SCIP, the other before-presolve
// heuristics off, runs it before presolving (presolving/maxrounds 0), and its best solution,
// matched by variable name, goes to the main as a complete solution (#442).
// Checked by heuristic: completesol skips a start past maxunknownrate with only
// a warning, so "a solution exists" does not mean "the start was completed".
SCIP_SOL* completesol_best(SCIP* aux) {
    const SCIP_HEUR* completesol = SCIPfindHeur(aux, "completesol");
    SCIP_SOL** sols = SCIPgetSols(aux);  // best first
    const int nsols = SCIPgetNSols(aux);
    for (int i = 0; completesol != nullptr && i < nsols; ++i) {
        if (SCIPsolGetHeur(sols[i]) == completesol) {
            return sols[i];
        }
    }
    return nullptr;
}

// Every other before-presolve heuristic's freq to -1: one storing the same
// point first would leave completesol's copy dropped as a duplicate and the
// start unreported. Only those: completesol's sub-MIP copies these params and
// keeps every other heuristic (these six or so lose their node runs too).
// Not SCIPsetHeuristics(OFF): it first resets heuristics/* to defaults,
// maxunknownrate from --set too (paramset.c:3092).
SCIP_RETCODE only_completesol_before_presolve(SCIP* aux) {
    SCIP_HEUR** heurs = SCIPgetHeurs(aux);
    const int nheurs = SCIPgetNHeurs(aux);
    for (int i = 0; i < nheurs; ++i) {
        const std::string name = SCIPheurGetName(heurs[i]);
        if (name != "completesol" &&
            (SCIPheurGetTimingmask(heurs[i]) & SCIP_HEURTIMING_BEFOREPRESOL) != 0) {
            SCIP_CALL(SCIPsetIntParam(aux, ("heuristics/" + name + "/freq").c_str(), -1));
        }
    }
    return SCIP_OKAY;
}

SCIP_RETCODE complete_start(SCIP* main, const Args& a, double deadline, bool* completed) {
    *completed = false;
    SCIP* aux = nullptr;
    SCIP_CALL(SCIPcreate(&aux));
    SCIP_RETCODE rc = load(aux, a);
    if (rc == SCIP_OKAY) {
        rc = SCIPsetIntParam(aux, "parallel/maxnthreads", 1);
    }
    if (rc == SCIP_OKAY) {
        rc = SCIPsetIntParam(aux, "presolving/maxrounds", 0);
    }
    if (rc == SCIP_OKAY) {
        rc = only_completesol_before_presolve(aux);
    }
    if (rc == SCIP_OKAY) {
        rc = SCIPsetRealParam(aux, "limits/time", std::max(deadline - monotonic_seconds(), 0.0));
    }
    if (rc == SCIP_OKAY) {
        rc = add_partial_sol(aux, a.partial_sol);
    }
    if (rc == SCIP_OKAY) {
        rc = SCIPpresolve(aux);
    }
    SCIP_SOL* best = rc == SCIP_OKAY ? completesol_best(aux) : nullptr;
    if (best != nullptr) {
        SCIP_SOL* sol = nullptr;
        rc = SCIPcreateOrigSol(main, &sol, nullptr);
        SCIP_VAR** vars = SCIPgetOrigVars(aux);
        const int nvars = SCIPgetNOrigVars(aux);
        for (int i = 0; rc == SCIP_OKAY && i < nvars; ++i) {
            SCIP_VAR* target = SCIPfindVar(main, SCIPvarGetName(vars[i]));
            if (target == nullptr) {
                std::cerr << "variable " << SCIPvarGetName(vars[i]) << " not in the main SCIP\n";
                rc = SCIP_INVALIDDATA;
                break;
            }
            rc = SCIPsetSolVal(main, sol, target, SCIPgetSolVal(aux, best, vars[i]));
        }
        SCIP_Bool stored = FALSE;
        if (rc == SCIP_OKAY) {
            rc = SCIPaddSolFree(main, &sol, &stored);
            *completed = stored != FALSE;
        } else if (sol != nullptr) {
            (void)SCIPfreeSol(main, &sol);
        }
    }
    SCIP_CALL(SCIPfree(&aux));
    return rc;
}

std::optional<double> finite(SCIP* scip, double v) {
    if (SCIPisInfinity(scip, std::abs(v))) {
        return std::nullopt;
    }
    return v;
}

nlohmann::json opt_json(std::optional<double> v) {
    return v.has_value() ? nlohmann::json(*v) : nlohmann::json(nullptr);
}

SCIP_RETCODE run(SCIP* scip, const Args& a, TrajLog& log, double start, nlohmann::json& result) {
    const double deadline = start + a.time_limit;
    SCIP_CALL(load(scip, a));
    SCIP_CALL(include_traj_handler(scip, &log, 0));
    SCIP_CALL(SCIPsetIntParam(scip, "parallel/maxnthreads", a.threads));
    result["start_completed"] = nullptr;
    if (!a.partial_sol.empty()) {
        if (a.threads > 1) {
            bool completed = false;
            const double t0 = monotonic_seconds();
            SCIP_CALL(complete_start(scip, a, deadline, &completed));
            result["start_completed"] = completed;
            result["completion_s"] = monotonic_seconds() - t0;
        } else {
            SCIP_CALL(add_partial_sol(scip, a.partial_sol));
        }
    }
    SCIP_CALL(SCIPsetRealParam(scip, "limits/time", std::max(deadline - monotonic_seconds(), 0.0)));
    if (a.threads > 1) {
        SCIP_CALL(SCIPsolveConcurrent(scip));
    } else {
        SCIP_CALL(SCIPsolve(scip));
    }
    const double end = monotonic_seconds();
    const bool has_sol = SCIPgetNSols(scip) > 0;
    result["status"] = status_name(SCIPgetStatus(scip));
    result["sense"] = SCIPgetObjsense(scip) == SCIP_OBJSENSE_MAXIMIZE ? "maximize" : "minimize";
    result["n_sols"] = SCIPgetNSols(scip);
    result["primal_bound"] = has_sol ? nlohmann::json(SCIPgetPrimalbound(scip)) : nullptr;
    result["dual_bound"] = opt_json(finite(scip, SCIPgetDualbound(scip)));
    result["solving_time_s"] = SCIPgetSolvingTime(scip);
    result["wall_s"] = end - start;
    if (!a.sol_out.empty() && has_sol) {
        FILE* f = std::fopen(a.sol_out.c_str(), "w");
        if (f == nullptr) {
            std::cerr << "cannot write " << a.sol_out << ": " << std::strerror(errno) << "\n";
            return SCIP_WRITEERROR;
        }
        const SCIP_RETCODE rc = SCIPprintBestSol(scip, f, TRUE);
        std::fclose(f);
        SCIP_CALL(rc);
    }
    return SCIP_OKAY;
}

// Replaced by rename, so a reader never sees half a file, even of a killed driver.
bool write_result(const Args& a, nlohmann::json& result, SCIP_RETCODE rc, bool freed) {
    result["ok"] = rc == SCIP_OKAY;
    result["scip_retcode"] = static_cast<int>(rc);
    result["freed"] = freed;
    const std::string tmp = a.result_out + ".tmp";
    {
        std::ofstream out(tmp);
        out << result.dump(2) << "\n";
        out.close();
        if (!out) {
            std::cerr << "cannot write " << tmp << "\n";
            std::remove(tmp.c_str());
            return false;
        }
    }
    if (std::rename(tmp.c_str(), a.result_out.c_str()) != 0) {
        std::cerr << "cannot rename " << tmp << ": " << std::strerror(errno) << "\n";
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

int run_main(int argc, char** argv) {
    const double start = monotonic_seconds();
    const std::optional<Args> args = parse_args(argc, argv);
    if (!args.has_value()) {
        return 2;
    }
    const std::string githash = SCIPgetGitHash();
    const std::string pinned = CBLS_SCIP_PINNED_COMMIT;
    if (githash.size() < 7 || pinned.compare(0, githash.size(), githash) != 0) {
        std::cerr << "libscip is " << githash << ", not the pinned commit " << pinned << "\n";
        return 2;
    }
    TrajLog log;
    log.out = std::fopen(args->traj_out.c_str(), "w");
    if (log.out == nullptr) {
        std::cerr << "cannot write " << args->traj_out << ": " << std::strerror(errno) << "\n";
        return 2;
    }
    nlohmann::json result = {
        {"problem", args->problem},
        {"threads", args->threads},
        {"time_limit_s", args->time_limit},
        {"start_mono_s", start},
        {"scip_version", std::to_string(SCIPmajorVersion()) + "." +
                             std::to_string(SCIPminorVersion()) + "." +
                             std::to_string(SCIPtechVersion())},
        {"scip_git_hash", githash},
    };
    SCIP* scip = nullptr;
    SCIP_RETCODE rc = SCIPcreate(&scip);
    if (rc == SCIP_OKAY) {
        log.main = scip;
        log.concurrent = args->threads > 1;
        rc = run(scip, *args, log, start, result);
    }
    // Written before SCIPfree too, which can outlast the caller's kill grace on a
    // large model: a killed driver leaves this result, "freed": false (#442).
    result["records"] = log.records;
    const bool early_written = write_result(*args, result, rc, false);
    if (scip != nullptr) {
        const SCIP_RETCODE free_rc = SCIPfree(&scip);
        if (rc == SCIP_OKAY) {
            rc = free_rc;
        }
    }
    std::fclose(log.out);
    if (!write_result(*args, result, rc, true) || !early_written) {
        return 2;
    }
    return rc == SCIP_OKAY ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run_main(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cbls_scip_traj: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "cbls_scip_traj: unknown exception\n");
    }
    return 2;
}
