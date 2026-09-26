// The host's cancellation channel (#169): `SearchConfig::stop`,
// `ParallelConfig::stop`, and `TerminationReason::Cancelled`.
//
// Every test here is written the #104 way: it proves that the STOP is what
// ended the run, not that the run happened to be short. An iteration budget is
// set far above what the run can reach before the cancel, and the assertion is
// `termination == Cancelled` together with `iterations` strictly inside that
// budget -- so a stop check that was deleted would either run the budget out
// (failing on iterations) or report a different reason (failing on
// termination). None of them asserts on a duration.

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <chrono>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace cbls;

namespace {

// A continuous model with an objective, so the search can never exit on
// TerminationReason::Feasible and keeps finding new bests (the objective bound
// is tightened on each), which is what keeps the progress callback firing.
Model quadratic_model() {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));  // x + y >= 1
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
    return m;
}

// Raise the token from inside the search's own progress callback rather than
// from a sleeping thread: the cancellation then lands at a deterministic point
// and the test needs no wall-clock assertion. tests/CMakeLists.txt's #104 note
// asks for exactly that restraint.
struct StopOnProgress : SolveCallback {
    StopToken* token;
    int seen = 0;
    explicit StopOnProgress(StopToken* t) : token(t) {}
    void on_progress(const SolveProgress& /*p*/) override {
        if (++seen >= 2) {
            token->request();
        }
    }
};

// Big enough that a search which ignored the stop would run visibly past the
// cancellation, small enough that such a regression fails in seconds rather
// than parking the suite.
constexpr int64_t kUnreachableIterations = 2000000;

}  // namespace

TEST_CASE("a host stop cancels a search that still has budget", "[stop]") {
    StopToken token;
    token.request();

    SearchConfig config;
    config.max_iterations = kUnreachableIterations;
    config.stop = token;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/9, true, nullptr, nullptr, 3, nullptr, config);

    // Not TimeLimit: this run had no wall clock at all. Not Stopped either --
    // no peer worker exists, and a host that cancelled needs to read back that
    // it was the cause.
    REQUIRE(r.termination == TerminationReason::Cancelled);
    REQUIRE(r.iterations < config.max_iterations);
}

TEST_CASE("a host stop raised mid-search ends the run where it stands", "[stop]") {
    // The distinguishing case: cancellation at a batch boundary part-way
    // through a search that is doing work. The test above pre-raises the flag,
    // so the loop condition is false on its first evaluation and no batch ever
    // runs.
    StopToken token;
    StopOnProgress cb(&token);

    SearchConfig config;
    config.max_iterations = kUnreachableIterations;
    config.batch_iterations = 100;
    config.stop = token;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/5, true, nullptr, nullptr, 3, &cb, config);

    REQUIRE(r.termination == TerminationReason::Cancelled);
    // Cancelled MID-FLIGHT, not before it started. Without this the test passes
    // on the pre-raised case too.
    REQUIRE(r.iterations > 0);
    REQUIRE(r.iterations < config.max_iterations);
}

TEST_CASE("an unattached stop leaves the run to its own budgets", "[stop]") {
    // The control for the two above. A `StopRef` that names nothing must never
    // read as requested, or every unconfigured run in the tree would end as
    // Cancelled -- which is the failure mode a default-constructed function
    // pointer invites.
    SearchConfig config;
    config.max_iterations = 5000;
    REQUIRE_FALSE(config.stop.attached());
    REQUIRE_FALSE(config.stop.requested());

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/5, true, nullptr, nullptr, 3, nullptr, config);

    REQUIRE(r.termination == TerminationReason::IterationLimit);
    REQUIRE(r.iterations >= config.max_iterations);
}

TEST_CASE("an attached but unraised stop leaves the run to its own budgets", "[stop]") {
    // Stronger than the unattached control above, and the case that separates
    // `StopRef::requested()` from `StopRef::attached()` AT THE POLL SITE: a
    // `cancel_requested()` that asked whether a source is ATTACHED would end every
    // configured run as Cancelled, and every other solve in this file attaches a
    // token it then raises -- so nothing else here would notice.
    StopToken token;
    SearchConfig config;
    config.max_iterations = 5000;
    config.stop = token;
    REQUIRE(config.stop.attached());
    REQUIRE_FALSE(config.stop.requested());

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/5, true, nullptr, nullptr, 3, nullptr, config);

    REQUIRE(r.termination == TerminationReason::IterationLimit);
    REQUIRE(r.iterations >= config.max_iterations);
    // And the search never wrote to the token.
    REQUIRE_FALSE(token.requested());
}

TEST_CASE("an attached stop is not a budget", "[stop]") {
    // Documented on `StopRef::attached()` and in docs/architecture.md, and worth
    // pinning because "run until the host cancels me" is the first thing a host
    // tries: with neither a wall clock nor an iteration limit the run returns at
    // once with NoBudget, attached stop or not. Treating an attachment as a budget
    // would hang forever the first time a host forgot to cancel.
    StopToken token;
    SearchConfig config;  // no max_iterations
    config.stop = token;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/5, true, nullptr, nullptr, 3, nullptr, config);

    REQUIRE(r.termination == TerminationReason::NoBudget);
    REQUIRE(r.iterations == 0);
}

TEST_CASE("a stop source is adapted by shape, not by type", "[stop]") {
    // StopRef takes anything with `bool requested() const`, which is what lets a
    // host hand over its own flag wrapper (or a C++20 std::stop_token) without
    // an adapter class.
    struct HostFlag {
        bool raised = false;
        [[nodiscard]] bool requested() const { return raised; }
    };

    HostFlag flag;
    const StopRef ref(flag);
    REQUIRE(ref.attached());
    REQUIRE_FALSE(ref.requested());
    flag.raised = true;
    REQUIRE(ref.requested());

    // And a default-constructed one is inert, with no source to read.
    const StopRef none;
    REQUIRE_FALSE(none.attached());
    REQUIRE_FALSE(none.requested());
}

TEST_CASE("a token can be reset and reused", "[stop]") {
    StopToken token;
    REQUIRE_FALSE(token.requested());
    token.request();
    REQUIRE(token.requested());
    token.reset();
    REQUIRE_FALSE(token.requested());
}

TEST_CASE("a host stop cancels every portfolio worker", "[stop][parallel]") {
    // ParallelConfig::stop reaches the workers through the combined StopRef in
    // solve_portfolio. The portfolio has NO wall clock here, so a worker that
    // ignored the cancel would grind its whole iteration budget and the
    // aggregate would come back IterationLimit.
    StopToken token;
    StopOnProgress cb(&token);

    SearchConfig config;
    config.max_iterations = kUnreachableIterations;
    config.batch_iterations = 100;

    ParallelConfig par_config;
    par_config.n_threads = 2;
    par_config.stop = token;

    ParallelSearch ps(2);
    const SearchResult r =
        ps.solve([] { return quadratic_model(); }, /*time_limit=*/0.0, /*seed=*/11, config,
                 /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, &cb, par_config);

    REQUIRE(r.termination == TerminationReason::Cancelled);
    REQUIRE(r.iterations > 0);
    // Summed over both workers, so the budget to beat is twice one worker's.
    REQUIRE(r.iterations < 2 * config.max_iterations);
}

TEST_CASE("a stop raised from another thread reaches a running portfolio", "[stop][parallel]") {
    // The shape a host actually has: the cancel arrives from a thread that is
    // neither a worker nor the caller, while the solve is genuinely IN FLIGHT.
    //
    // "In flight" is the hard part, and an earlier version of this test did not
    // get it: opening the gate before `ps.solve` let the canceller fire before any
    // worker existed, so every worker broke out of its restart loop without ever
    // searching and the result came from the no-result fallback -- the same path
    // the pre-raised cases above already cover. The gate is now a PROGRESS ROW,
    // which only a running worker can produce, and `iterations > 0` is the
    // assertion that says the cancel landed on a search that had started.
    //
    // The wait is bounded so a regression cannot hang the suite: if no row ever
    // arrives the canceller fires anyway, inside the 30-second budget, so the run
    // is still mid-flight. That bound is not a duration assertion -- nothing here
    // asserts on elapsed time. The 30 seconds are there so that `Cancelled` is not
    // satisfiable by the clock expiring, which would report TimeLimit.
    struct GateOnProgress : SolveCallback {
        std::atomic<bool>* gate;
        explicit GateOnProgress(std::atomic<bool>* g) : gate(g) {}
        void on_progress(const SolveProgress& /*p*/) override {
            gate->store(true, std::memory_order_release);
        }
    };

    StopToken token;
    std::atomic<bool> searching{false};
    GateOnProgress cb(&searching);

    ParallelConfig par_config;
    par_config.n_threads = 2;
    par_config.stop = token;

    std::thread canceller([&token, &searching]() {
        const auto give_up = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!searching.load(std::memory_order_acquire) &&
               std::chrono::steady_clock::now() < give_up) {
            std::this_thread::yield();
        }
        token.request();
    });

    ParallelSearch ps(2);
    const SearchResult r =
        ps.solve([] { return quadratic_model(); }, /*time_limit=*/30.0, /*seed=*/13, SearchConfig{},
                 /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, &cb, par_config);
    canceller.join();

    REQUIRE(r.termination == TerminationReason::Cancelled);
    REQUIRE(r.iterations > 0);
}

TEST_CASE("a cancel seen only after a worker's last solve still reports Cancelled",
          "[stop][parallel]") {
    // The window `with_host_cancel` exists for: a worker finishes a restart on its
    // ITERATION budget and the cancel is only visible afterwards. Reporting
    // IterationLimit there says the engine chose to stop, when the host took the
    // budget away.
    //
    // ONE worker, and that is load-bearing. With two, the first worker to finish
    // raises the token while its PEER is still mid-solve, the peer's own
    // `past_deadline()` sees it and returns Cancelled, and the aggregate is then
    // Cancelled for a completely different reason -- which is exactly how this test
    // passed with the guard neutered when it was first written. One worker has no
    // peer to see it.
    //
    // Reverting `with_host_cancel` to `return aggregate;` makes this report
    // IterationLimit.
    // A `Tracer`'s DESTRUCTOR runs on the worker's thread after its last solve and
    // before the portfolio aggregates, which is the only seam a host has into that
    // moment: callback, hook, LNS and tracer EVENTS all fire inside `cbls::solve`.
    struct RaiseOnDestruction : Tracer {
        StopToken* token;
        explicit RaiseOnDestruction(StopToken* t) : token(t) {}
        RaiseOnDestruction(const RaiseOnDestruction&) = delete;
        RaiseOnDestruction& operator=(const RaiseOnDestruction&) = delete;
        RaiseOnDestruction(RaiseOnDestruction&&) = delete;
        RaiseOnDestruction& operator=(RaiseOnDestruction&&) = delete;
        ~RaiseOnDestruction() override { token->request(); }
    };

    StopToken token;
    SearchConfig config;
    // No wall clock, so the worker runs exactly ONE solve and ends on this budget.
    config.max_iterations = 2000;
    config.batch_iterations = 100;

    ParallelConfig par_config;
    par_config.n_threads = 1;
    par_config.stop = token;
    par_config.tracer_factory = [&token](int /*worker*/) -> std::unique_ptr<Tracer> {
        return std::make_unique<RaiseOnDestruction>(&token);
    };

    ParallelSearch ps(1);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.0, /*seed=*/17, config,
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(token.requested());
    // The worker did a full solve, so this is not the no-result fallback (which
    // reports Cancelled by a different route -- see empty_portfolio_reason).
    REQUIRE(r.iterations >= config.max_iterations);
    REQUIRE(r.termination == TerminationReason::Cancelled);
}

TEST_CASE("a wall clock that expires is not relabelled as a cancel", "[stop][parallel]") {
    // The other half of `with_host_cancel`, and why it is restricted to the
    // iteration budget: a portfolio whose SHARED CLOCK ran out is time-limited, and
    // a host cancel arriving in the same instant must not rewrite that. Same order
    // `ViolationLSLoop::run` applies by asking the clock first.
    //
    // One worker, for the same reason as the test above.
    // A `Tracer`'s DESTRUCTOR runs on the worker's thread after its last solve and
    // before the portfolio aggregates, which is the only seam a host has into that
    // moment: callback, hook, LNS and tracer EVENTS all fire inside `cbls::solve`.
    struct RaiseOnDestruction : Tracer {
        StopToken* token;
        explicit RaiseOnDestruction(StopToken* t) : token(t) {}
        RaiseOnDestruction(const RaiseOnDestruction&) = delete;
        RaiseOnDestruction& operator=(const RaiseOnDestruction&) = delete;
        RaiseOnDestruction(RaiseOnDestruction&&) = delete;
        RaiseOnDestruction& operator=(RaiseOnDestruction&&) = delete;
        ~RaiseOnDestruction() override { token->request(); }
    };

    StopToken token;
    ParallelConfig par_config;
    par_config.n_threads = 1;
    par_config.stop = token;
    par_config.tracer_factory = [&token](int /*worker*/) -> std::unique_ptr<Tracer> {
        return std::make_unique<RaiseOnDestruction>(&token);
    };

    ParallelSearch ps(1);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/0.25, /*seed=*/19, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);

    REQUIRE(token.requested());
    REQUIRE(r.termination == TerminationReason::TimeLimit);
}

TEST_CASE("SearchConfig::stop and ParallelConfig::stop are OR-ed", "[stop][parallel]") {
    // Either channel alone cancels. A portfolio that read only one of them
    // would silently ignore whichever the caller happened to set.
    auto run = [](bool on_search_config, bool on_parallel_config) {
        StopToken token;
        token.request();
        SearchConfig config;
        config.max_iterations = kUnreachableIterations;
        if (on_search_config) {
            config.stop = token;
        }
        ParallelConfig par_config;
        par_config.n_threads = 1;
        if (on_parallel_config) {
            par_config.stop = token;
        }
        ParallelSearch ps(1);
        return ps
            .solve([] { return quadratic_model(); }, /*time_limit=*/0.0, /*seed=*/3, config,
                   nullptr, nullptr, nullptr, par_config)
            .termination;
    };

    REQUIRE(run(/*on_search_config=*/true, /*on_parallel_config=*/false) ==
            TerminationReason::Cancelled);
    REQUIRE(run(/*on_search_config=*/false, /*on_parallel_config=*/true) ==
            TerminationReason::Cancelled);
    REQUIRE(run(/*on_search_config=*/true, /*on_parallel_config=*/true) ==
            TerminationReason::Cancelled);
}
