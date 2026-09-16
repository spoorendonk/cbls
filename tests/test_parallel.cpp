// Tests for the cooperative portfolio: live solution sharing, restart from the
// shared pool, and the "no idle worker while budget remains" property.
//
// The first test here is the load-bearing one. Everything the portfolio gained
// is reached through `SearchCoordination*`, and every benchmark runner -- and
// the MIPfeas same-algorithm comparison that rests on them -- calls
// `cbls::solve()` with that pointer null. So the first thing to pin is that a
// null pointer leaves the trajectory exactly what it was.

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

using namespace cbls;

namespace {

// A continuous model with an objective, so no worker can ever exit on
// TerminationReason::Feasible: every run here is budget-bounded.
Model quadratic_model() {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);
    // x + y >= 1, minimise x^2 + y^2.
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
    return m;
}

// A pure-feasibility model: no objective, so the first feasible assignment IS
// the answer and solve() returns TerminationReason::Feasible.
Model satisfaction_model() {
    Model m;
    auto x = m.int_var(0, 100);
    auto y = m.int_var(0, 100);
    auto neg1 = m.constant(-1.0);
    auto ten = m.constant(10.0);
    m.add_constraint(m.sum({ten, m.prod(neg1, x), m.prod(neg1, y)}));  // x + y >= 10
    m.close();
    return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// The invariant that protects every benchmark
// ---------------------------------------------------------------------------

TEST_CASE("a null SearchCoordination leaves the search bit-identical", "[parallel][coord]") {
    // Iteration-budgeted, no wall clock: solve() reads no clock at all on this
    // path, so two runs at the same seed are bit-reproducible and any
    // divergence is the code change, not the machine.
    SearchConfig config;
    config.max_iterations = 20000;

    Model a = quadratic_model();
    const SearchResult ra = solve(a, /*time_limit=*/0.0, /*seed=*/7, true, nullptr, nullptr, 3,
                                  nullptr, config, /*coord=*/nullptr);

    // A coordination object that is present but carries neither channel: the
    // guards inside the search must key on the CHANNELS, not on the pointer, or
    // ParallelSearch's own null-pool paths would diverge from this.
    SearchCoordination empty;
    Model b = quadratic_model();
    const SearchResult rb = solve(b, /*time_limit=*/0.0, /*seed=*/7, true, nullptr, nullptr, 3,
                                  nullptr, config, &empty);

    REQUIRE(ra.iterations == rb.iterations);
    REQUIRE(ra.feasible == rb.feasible);
    REQUIRE(ra.objective == rb.objective);
    REQUIRE(ra.termination == rb.termination);
    REQUIRE(ra.best_state.values == rb.best_state.values);
    REQUIRE(ra.perturbations == rb.perturbations);
}

// ---------------------------------------------------------------------------
// Submit when found
// ---------------------------------------------------------------------------

TEST_CASE("a solve submits every incumbent as it finds it", "[parallel][coord]") {
    // The point of the requirement is that sharing happens DURING the search,
    // not once at the end. One solve, one pool: if the pool holds more than one
    // solution when the call returns, they were submitted mid-run -- a
    // submit-at-the-end implementation can only ever put one there.
    SolutionPool pool(10);
    SearchCoordination coord;
    coord.pool = &pool;

    SearchConfig config;
    config.max_iterations = 50000;

    Model m = quadratic_model();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/11, true, nullptr, nullptr, 3,
                                 nullptr, config, &coord);

    REQUIRE(r.feasible);
    REQUIRE(pool.size() > 1);
    auto best = pool.best();
    if (!best.has_value()) {
        FAIL("pool empty after a feasible solve");
        return;
    }
    REQUIRE(best->feasible);
    // The pool's best is the run's best: the last thing record_best shared was
    // the incumbent the run goes on to return.
    REQUIRE(best->objective == r.objective);
}

// ---------------------------------------------------------------------------
// Restart from the shared pool
// ---------------------------------------------------------------------------

TEST_CASE("a stalled search adopts a peer's solution from the pool", "[parallel][coord]") {
    // The model is chosen so the gift is genuinely OUT OF REACH of the search
    // itself inside the budget. `sum(x) >= kTarget` over kVars integer columns,
    // minimising sum of squares: reaching the target is easy and FJ does it in
    // one batch, but the OPTIMUM is the balanced assignment, and getting there
    // means moving mass between columns one jump at a time. A run that starts
    // by piling the target onto a few columns spends a long way climbing down.
    //
    // The control arm below is not decoration. Without it this test passes
    // whether or not adoption works the moment the search gets fast enough to
    // reach the gift on its own -- exactly how it went vacuous the first time
    // it was written.
    constexpr int kVars = 20;
    constexpr double kTarget = 60.0;
    auto build = []() {
        Model m;
        std::vector<int32_t> xs;
        std::vector<int32_t> squares;
        auto two = m.constant(2);
        auto neg1 = m.constant(-1.0);
        xs.reserve(kVars);
        for (int i = 0; i < kVars; ++i) {
            xs.push_back(m.int_var(0, 10));
            squares.push_back(m.pow_expr(xs.back(), two));
        }
        std::vector<int32_t> row;
        row.push_back(m.constant(kTarget));
        for (int32_t x : xs) {
            row.push_back(m.prod(neg1, x));
        }
        m.add_constraint(m.sum(row));  // kTarget - sum(x) <= 0
        m.minimize(m.sum(squares));
        m.close();
        return m;
    };

    // The balanced assignment: every column at 3 -- sum 60, objective 20*9 = 180,
    // which is this model's optimum.
    Model donor = build();
    Model::State balanced = donor.copy_state();
    for (int i = 0; i < kVars; ++i) {
        balanced.values[i] = 3.0;
    }
    constexpr double kGiftObjective = 180.0;

    SearchConfig config;
    config.max_iterations = 20000;
    // Small batches so the run gets several of them, and a short
    // stagnation window so the full-period kick -- the only route adoption is
    // wired to -- is actually reached inside that many. Leaving
    // perturbation_period at its default of 100 would end the run before a
    // single kick, and shrinking max_iterations instead hits the OTHER cap in
    // budget_exhausted (`batches >= max_iterations`), which with the
    // unproductive exit ending batches early makes the batch count the binding
    // one and the run far longer than the iteration number suggests.
    config.batch_iterations = 100;
    config.perturbation_period = 2;

    // Control arm: same model, same seed, same budget, NO pool.
    Model control = build();
    const SearchResult rc = solve(control, /*time_limit=*/0.0, /*seed=*/3, true, nullptr, nullptr,
                                  3, nullptr, config, /*coord=*/nullptr);
    REQUIRE(rc.feasible);
    // If this ever stops holding the test below has gone vacuous, and this is
    // the assertion that says so instead of quietly passing.
    INFO("control objective " << rc.objective << " must be worse than the gift's "
                              << kGiftObjective);
    REQUIRE(rc.objective > kGiftObjective);

    // Treatment arm. Capacity ONE, deliberately: the search shares its own
    // incumbents into this same pool, and get_restart_point draws from the
    // better half rather than the best, so with room for ten the draw would
    // usually be one of the search's own. At capacity one the pool keeps only
    // the best solution submitted -- the gift, since nothing this run finds
    // beats it -- and the draw is deterministic.
    Solution gift;
    gift.state = balanced;
    gift.objective = kGiftObjective;
    gift.feasible = true;
    SolutionPool pool(1);
    pool.submit(gift);

    SearchCoordination coord;
    coord.pool = &pool;

    Model m = build();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/3, true, nullptr, nullptr, 3,
                                 nullptr, config, &coord);

    REQUIRE(r.feasible);
    // Having been handed the gift, the run must not come back worse than it.
    INFO("pooled objective " << r.objective);
    REQUIRE(r.objective <= kGiftObjective + 1e-6);

    // A self-draw must NOT count as a kick. Once this run has adopted the gift
    // it is sitting on it, and the capacity-1 pool holds nothing else, so every
    // later full-period kick draws the assignment the worker already holds.
    // Those kicks fall through to diversify(), which leaves the Float escape
    // probe that maybe_diversify just armed in place -- whereas an adoption
    // disarms it. So the latch sampled at exit is exactly the difference
    // between "the later kicks perturbed" and "the later kicks restored the
    // state we were already on and moved nothing". Measured: armed with the
    // guard, unarmed without it.
    REQUIRE(r.escape_probe_armed);

    // Adoption re-grounds the DAG, the violation manager and FJ, so what comes
    // back must be a real assignment of this model and not a half-restored one.
    // Verify it independently rather than trusting the search's own bookkeeping.
    Model check = build();
    check.restore_state(r.best_state);
    full_evaluate(check);
    const VerifyResult v = verify_model(check);
    INFO("verify errors: " << v.errors.size());
    REQUIRE(v.ok);
}

TEST_CASE("a pool solution that does not fit the model is refused", "[parallel][coord]") {
    // The factory is caller-supplied and nothing makes it return the same model
    // twice. A state of the wrong width restored element-wise into the wrong
    // variables would be searched from in silence, so the guard refuses it --
    // and the run must simply carry on, not throw and not hang.
    Solution alien;
    alien.state.values = {1.0, 2.0, 3.0, 4.0, 5.0};  // wider than the 2-var model
    alien.state.elements.resize(5);
    alien.objective = -1e9;  // sorts first, so get_restart_point will draw it
    alien.feasible = true;

    SolutionPool pool(10);
    pool.submit(alien);

    // Same shape of budget as the adoption test above, and for the same
    // reason: the guard is only exercised on a batch that actually reaches the
    // full-period kick, so a run that never stagnates would pass this whether
    // or not the guard exists. Model::restore_state throws on a size mismatch,
    // so with the guard gone the REQUIRE_NOTHROW below is what fails.
    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;

    SearchCoordination coord;
    coord.pool = &pool;

    Model m = quadratic_model();
    SearchResult r;
    REQUIRE_NOTHROW(r = solve(m, /*time_limit=*/0.0, /*seed=*/5, true, nullptr, nullptr, 3, nullptr,
                              config, &coord));
    REQUIRE(r.feasible);
    REQUIRE(r.best_state.values.size() == 2);
    // The bogus objective never became the run's answer.
    REQUIRE(r.objective > 0.0);
}

// ---------------------------------------------------------------------------
// The stop flag
// ---------------------------------------------------------------------------

TEST_CASE("a raised stop flag ends a search that still has budget", "[parallel][coord]") {
    std::atomic<bool> stop{true};
    SearchCoordination coord;
    coord.stop = &stop;

    SearchConfig config;
    // Large enough that a broken stop check runs for ~20s -- comfortably past
    // the 5s assertion below -- and no larger, so a regression fails in half a
    // minute rather than parking a test run for three.
    config.max_iterations = 10000000;

    Model m = quadratic_model();
    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/9, true, nullptr, nullptr, 3,
                                 nullptr, config, &coord);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    REQUIRE(elapsed < 5.0);
    // Not TimeLimit: this run had no wall clock at all, and reporting one would
    // make `time_seconds` read as a budget that was never set.
    REQUIRE(r.termination == TerminationReason::Stopped);
}

// ---------------------------------------------------------------------------
// ParallelSearch: no idle workers, and a global stop
// ---------------------------------------------------------------------------

TEST_CASE("the portfolio stops every worker once one solves the model", "[parallel]") {
    // The factory hands the FIRST worker to call it a satisfiable model and
    // every other worker an unsatisfiable one of the same shape. (First to
    // call, not index 0: the factory runs on the worker threads.) That asymmetry is what makes the
    // test discriminate: give every worker the same easy model and they all
    // finish in milliseconds on their own, so the call returns fast whether or
    // not the stop flag exists -- which is exactly how the first version of
    // this test passed with the mechanism removed.
    //
    // Here, one worker finds its feasible point at once and the question is
    // settled. The others are grinding a model with no feasible point at all
    // and have no budget reason to stop, so without the flag this call blocks
    // until the wall clock runs out.
    std::atomic<int> handed{0};
    auto factory = [&handed]() {
        const int which = handed.fetch_add(1);
        Model m;
        auto x = m.int_var(0, 100);
        auto y = m.int_var(0, 100);
        auto neg1 = m.constant(-1.0);
        // x + y >= 10 is trivially satisfiable; x + y >= 1000 is not, the
        // domains cap the sum at 200. Same shape either way, so the pool's
        // entries stay mutually restorable.
        auto rhs = m.constant(which == 0 ? 10.0 : 1000.0);
        m.add_constraint(m.sum({rhs, m.prod(neg1, x), m.prod(neg1, y)}));
        m.close();
        return m;
    };

    const double budget = 30.0;
    ParallelSearch ps(4);
    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult r = ps.solve(factory, budget, 42);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    REQUIRE(r.feasible);
    REQUIRE(r.termination == TerminationReason::Feasible);
    // Generous against a solve that takes milliseconds: this is a "did the
    // other three workers hear about it at all" detector, not a performance
    // floor. Without the stop flag it is `budget`.
    INFO("elapsed " << elapsed << "s of a " << budget << "s budget");
    REQUIRE(elapsed < budget / 3.0);
}

TEST_CASE("a worker out of iterations is restarted, not left idle", "[parallel]") {
    // Each solve() is capped at a few thousand iterations while the portfolio
    // holds a wall-clock budget. Without the restart loop every worker returns
    // almost immediately and the cores sit out the rest of the run; with it,
    // each worker keeps restarting until the clock stops, so the total
    // iteration count runs far past n_threads * max_iterations.
    constexpr int kThreads = 2;
    constexpr int64_t kCap = 2000;
    const double budget = 2.0;

    SearchConfig config;
    config.max_iterations = kCap;

    ParallelConfig pc;
    pc.n_threads = kThreads;

    ParallelSearch ps(kThreads);
    const auto t0 = std::chrono::steady_clock::now();
    const SearchResult r = ps.solve([]() { return quadratic_model(); }, budget, 42, config, nullptr,
                                    nullptr, nullptr, pc);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    REQUIRE(r.feasible);
    // A single solve per worker could account for at most this many iterations
    // (plus up to batch_iterations-1 of overshoot each). Exceeding it several
    // times over is the restart loop, and nothing else.
    REQUIRE(r.iterations > int64_t{4} * kThreads * kCap);
    // And the budget was actually used rather than returned early.
    REQUIRE(elapsed > budget * 0.5);
}

TEST_CASE("an iteration-only portfolio returns instead of restarting forever", "[parallel]") {
    // `time_limit <= 0` disables the wall clock, leaving SearchConfig::
    // max_iterations as the only budget -- a supported call shape, and the one
    // the removed epoch-sync mode used. The restart loop must NOT restart there:
    // there is no "time the predecessor left" to run on, so every restart would
    // hand the worker its whole iteration budget again. It hung.
    //
    // The TIMEOUT on this test in tests/CMakeLists.txt is what makes the
    // regression report itself; without it a failure is an inconclusive
    // never-finishing run.
    SearchConfig config;
    config.max_iterations = 500;

    ParallelConfig pc;
    pc.n_threads = 2;

    ParallelSearch ps(2);
    SearchResult r;
    REQUIRE_NOTHROW(r = ps.solve([]() { return quadratic_model(); }, /*time_limit=*/0.0, 42, config,
                                 nullptr, nullptr, nullptr, pc));
    REQUIRE(r.feasible);
    // One solve per worker, so the iteration count is bounded by the budget each
    // was given (plus a batch of overshoot apiece). This is the assertion that
    // distinguishes "returned" from "restarted a few times and then returned".
    REQUIRE(r.iterations <= int64_t{2} * pc.n_threads * config.max_iterations);
    REQUIRE(r.termination == TerminationReason::IterationLimit);
}

TEST_CASE("the portfolio returns a verifiable assignment", "[parallel]") {
    // The returned state can now come from a DIFFERENT worker's model than the
    // one that recorded it last, so "is this a valid assignment of the model"
    // is no longer implied by the single-solve tests. Check it independently.
    ParallelSearch ps(4);
    const SearchResult r = ps.solve([]() { return quadratic_model(); }, 1.0, 42);
    REQUIRE(r.feasible);

    Model check = quadratic_model();
    REQUIRE(r.best_state.values.size() == check.num_vars());
    check.restore_state(r.best_state);
    full_evaluate(check);
    const VerifyResult v = verify_model(check);
    INFO("verify errors: " << v.errors.size());
    REQUIRE(v.ok);
}
