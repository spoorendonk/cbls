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
#include <cstdint>
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
    // not a worker, while the solve is in flight. A wall-clock budget is set
    // here -- generously, 30 seconds -- precisely so that the assertion on
    // `termination` is not satisfiable by the budget expiring: a run that
    // reached its deadline would report TimeLimit.
    StopToken token;
    std::atomic<bool> running{false};

    ParallelConfig par_config;
    par_config.n_threads = 2;
    par_config.stop = token;

    std::thread canceller([&token, &running]() {
        while (!running.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        token.request();
    });

    ParallelSearch ps(2);
    running.store(true, std::memory_order_release);
    const SearchResult r = ps.solve(
        [] { return quadratic_model(); }, /*time_limit=*/30.0, /*seed=*/13, SearchConfig{},
        /*hook_factory=*/nullptr, /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);
    canceller.join();

    REQUIRE(r.termination == TerminationReason::Cancelled);
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
