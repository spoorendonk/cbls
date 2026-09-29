// MINLPLib through the PORTFOLIO, for #179's shared-bound A/B.
//
// `cbls_minlplib` runs `cbls::solve()` single-threaded and has no --threads, so
// it cannot run either arm of a portfolio comparison. This runs ONE instance
// through `ParallelSearch` on the master model, built exactly as that runner
// builds it (`read_nl` -> `nl_to_model`), with the runner's per-worker
// `FloatIntensifyHook` and `LNS(0.3)` at `lns_interval = 3`, and prints one JSON
// line: the result, the #179 counters, and the portfolio's incumbent trace
// (every new-best row of the progress stream, on the portfolio clock). The
// objective is the MINIMISED one -- a maximize instance is built negated.
//
// Not a published-table runner: it writes nothing but stdout, and
// `portfolio_ab.py` beside it is the driver that pairs the arms and scores them.
//
//   cbls_minlplib_portfolio INSTANCE.nl BUDGET_SECONDS SEED THREADS SHARE(0|1)

#include <benchmarks/common/runner_args.h>
#include <cbls/cbls.h>
#include <cbls/io_nl.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {

// Collects the portfolio's new-best rows. Called from every worker thread
// through ParallelSearch's own serialising wrapper; the lock is for the vector
// against the final read, which happens after the solve has joined them.
class TraceRecorder : public cbls::SolveCallback {
public:
    void on_progress(const cbls::SolveProgress& p) override {
        if (p.new_best && std::isfinite(p.objective)) {
            const std::scoped_lock lock(mutex_);
            points_.emplace_back(p.time_seconds, p.objective);
        }
    }
    std::vector<std::pair<double, double>> take() {
        const std::scoped_lock lock(mutex_);
        return std::move(points_);
    }

private:
    std::mutex mutex_;
    std::vector<std::pair<double, double>> points_;
};

int usage() {
    std::fprintf(stderr,
                 "Usage: cbls_minlplib_portfolio INSTANCE.nl BUDGET_SECONDS SEED THREADS "
                 "SHARE(0|1)\n");
    return 2;
}

int run(int argc, char** argv) {
    if (argc != 6) {
        return usage();
    }
    const std::string path = argv[1];
    // The shared reporting policy (benchmarks/common/runner_args.h): a bad double
    // comes back NaN for the positivity guard, a bad integer reports and exits 2.
    // Positional rather than ArgCursor flags because the driver is the only caller.
    const double budget = cbls::bench::parse_double("BUDGET_SECONDS", argv[2]);
    if (!(budget > 0.0)) {
        return usage();
    }
    const int64_t seed = cbls::bench::parse_int64("SEED", argv[3]);
    if (seed < 0) {
        return usage();  // strtoull would have wrapped a leading '-' silently
    }
    const int64_t threads = cbls::bench::parse_int64("THREADS", argv[4]);
    if (threads < 1 || threads > 256) {
        return usage();
    }
    const std::string share = argv[5];
    if (share != "0" && share != "1") {
        return usage();
    }

    cbls::NlProblem prob = cbls::read_nl(path);
    cbls::NlToModelResult built = cbls::nl_to_model(prob);
    if (!built.supported) {
        std::printf(R"({"unsupported": true})"
                    "\n");
        return 0;
    }
    cbls::SearchConfig cfg;
    cfg.lns_interval = 3;
    cbls::ParallelConfig pc;
    pc.n_threads = static_cast<int>(threads);
    pc.share_objective_bound = share == "1";
    auto hook_factory = [](cbls::Model&) -> std::shared_ptr<cbls::InnerSolverHook> {
        return std::make_shared<cbls::FloatIntensifyHook>();
    };
    auto lns_factory = []() -> std::shared_ptr<cbls::LNS> {
        return std::make_shared<cbls::LNS>(0.3);
    };
    TraceRecorder recorder;
    cbls::ParallelSearch ps(pc.n_threads);
    const cbls::SearchResult r = ps.solve(built.model, budget, static_cast<uint64_t>(seed), cfg,
                                          hook_factory, lns_factory, &recorder, pc);
    const auto trace = recorder.take();

    std::printf(R"({"feasible": %s, "objective": )", r.feasible ? "true" : "false");
    if (r.feasible && std::isfinite(r.objective)) {
        std::printf("%.17g", r.objective);
    } else {
        std::printf("null");
    }
    std::printf(R"(, "workers_completed": %d, "iterations": %lld)", r.workers_completed,
                static_cast<long long>(r.iterations));
    std::printf(R"(, "shared_bound_tightenings": %lld, "own_best_behind_global": %lld)",
                static_cast<long long>(r.counters.shared_bound_tightenings),
                static_cast<long long>(r.counters.own_best_behind_global));
    std::printf(R"(, "bound_behind_global_batches": %lld, "bound_behind_global_seconds": %.6f)",
                static_cast<long long>(r.counters.bound_behind_global_batches),
                r.counters.bound_behind_global_seconds);
    std::printf(R"(, "trace": [)");
    for (size_t i = 0; i < trace.size(); ++i) {
        std::printf("%s[%.6f, %.17g]", i > 0 ? ", " : "", trace[i].first, trace[i].second);
    }
    std::printf("]}\n");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "cbls_minlplib_portfolio: %s\n", e.what());
        return 1;
    }
}
