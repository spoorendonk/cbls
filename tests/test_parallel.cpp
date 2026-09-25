// Tests for the cooperative portfolio: live solution sharing, restart from the
// shared pool, and the "no idle worker while budget remains" property.
//
// The first test here is the load-bearing one. Everything the portfolio gained
// is reached through `SearchCoordination*`, and every benchmark runner -- and
// the MIPfeas same-algorithm comparison that rests on them -- calls
// `cbls::solve()` with that pointer null. So the first thing to pin is that a
// null pointer leaves the trajectory exactly what it was.

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <cbls/cbls.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
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

// A 20-column integer model: reaching the target is easy, but the optimum is
// the balanced assignment, so the search keeps finding new incumbents for a
// long time -- which is what fills a shared pool.
Model integer_model() {
    Model m;
    std::vector<int32_t> xs;
    std::vector<int32_t> squares;
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    for (int i = 0; i < 20; ++i) {
        xs.push_back(m.int_var(0, 10));
        squares.push_back(m.pow_expr(xs.back(), two));
    }
    std::vector<int32_t> row;
    row.push_back(m.constant(60.0));
    for (int32_t x : xs) {
        row.push_back(m.prod(neg1, x));
    }
    m.add_constraint(m.sum(row));
    m.minimize(m.sum(squares));
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
    //
    // It caught that a second time, and the size below is the repair. At 20
    // columns the control arm returned exactly the gift's 180 once a
    // diversification kick started from the incumbent rather than from wherever
    // the previous kick left the search (#158): the single-threaded run simply
    // finds this model's optimum now, so there was no gift left to be out of
    // reach. 80 columns puts it back out of reach -- the control converges to
    // 734 against the balanced optimum's 720 -- at the same budget. Raise the
    // column count, not the budget, if it ever goes vacuous again: the control's
    // 734 is where the search CONVERGES here, not where the budget stops it, so
    // a longer run does not widen the margin.
    constexpr int kVars = 80;
    constexpr double kTarget = 240.0;
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

    // The balanced assignment: every column at 3 -- sum 240, objective 80*9 = 720,
    // which is this model's optimum.
    Model donor = build();
    Model::State balanced = donor.copy_state();
    for (int i = 0; i < kVars; ++i) {
        balanced.values[i] = 3.0;
    }
    constexpr double kGiftObjective = 720.0;

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

    // A self-draw must NOT count as a kick. The capacity-1 pool holds nothing
    // but the gift, so a later full-period kick can only draw that -- and when
    // it does, holds_assignment stands the draw down and the kick falls through
    // to diversify(), which leaves the Float escape probe that maybe_diversify
    // just armed in place, whereas an adoption disarms it. So the latch sampled
    // at exit distinguishes "the last full-period kick was a stood-down
    // self-draw" from "it was an adoption".
    //
    // Since #158 the worker is NOT reliably sitting on the gift when that test
    // runs: diversify() restores kick_origin() -- here the gift, which IS
    // best_state_ because it improved on the control's 734 -- and then perturb()
    // is guaranteed to move at least one variable (#109/#111). So whether the
    // draw is a self-draw now depends on FJ having come back to the gift, which
    // is the same parity dependence the paragraph below already describes rather
    // than a new one. Re-measured on this branch at kVars = 80: both assertions
    // hold. The pre-#158 note said "armed with the guard, unarmed without it",
    // measured at kVars = 20 on a trajectory this change replaced; it is
    // restated here rather than carried over.
    //
    // The control is asserted too, and that is the point of asserting it: this
    // proxy is parity-dependent -- it really says "the LAST full-period kick of
    // this run was a self-draw" -- so an engine change that shifts the batch
    // count by one could flip it. With both arms pinned, such a drift shows up
    // as "the control changed as well", which points at the model or the
    // budget, rather than as a bare red pointing at the guard.
    REQUIRE(rc.escape_probe_armed);
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
TEST_CASE("a portfolio result describes the state it returns", "[parallel]") {
    // Everything on a SearchResult that is a statement ABOUT the returned point
    // has to be assembled with it. `best_violation` was not: the portfolio built
    // its result from the pool -- which carried the state, the objective and the
    // feasibility flag and nothing else -- and left the residual at its +inf
    // default. Any caller that gates publication on the residual then rejects
    // every parallel row it is handed, which is what benchmarks/mipfeas did.
    ParallelSearch ps(4);
    const SearchResult r = ps.solve(quadratic_model, /*time_limit=*/0.5, /*seed=*/42);

    REQUIRE(r.feasible);
    REQUIRE(std::isfinite(r.best_violation));
    REQUIRE(r.best_violation <= SearchConfig{}.feasibility_tolerance);
}

TEST_CASE("a portfolio result counts the work its workers did", "[parallel]") {
    // The same defect one field over: counters left at zero describe a run that
    // perturbed nothing and repaired nothing, and an ablation reading them would
    // conclude the mechanisms are never used.
    auto factory = [] { return integer_model(); };

    ParallelSearch ps(4);
    ParallelConfig par_config;
    par_config.n_threads = 4;
    SearchConfig cfg;
    cfg.perturbation_period = 1;  // kick on every non-improving batch
    cfg.batch_iterations = 50;
    auto lns_factory = []() -> std::shared_ptr<LNS> { return std::make_shared<LNS>(0.3); };
    const SearchResult r =
        ps.solve(factory, /*time_limit=*/0.5, /*seed=*/42, cfg, /*hook_factory=*/nullptr,
                 lns_factory, /*callback=*/nullptr, par_config);

    REQUIRE(r.iterations > 0);
    REQUIRE(r.perturbations > 0);
}

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

TEST_CASE("an LNS-due kick stands adoption down entirely", "[parallel][coord]") {
    // The structural version of `lns_calls > 0` in tests/test_search.cpp: with
    // lns_interval = 1 every full-period kick is LNS-due, so `lns_kick_due()`
    // short-circuits and adopt_from_pool() is NEVER CALLED -- not even to draw
    // from the RNG. A run with a pool must therefore be bit-identical to one
    // without, which is a far stronger statement than "some LNS ran".
    //
    // Delete `lns_kick_due() ||` from maybe_diversify and the pooled run adopts
    // the gift instead: the assignment moves, the RNG advances, and every
    // REQUIRE below parts company with the control.
    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;
    config.lns_interval = 1;

    LNS lns_a(0.3);
    Model a = quadratic_model();
    const SearchResult ra =
        solve(a, /*time_limit=*/0.0, /*seed=*/13, true, nullptr, &lns_a, 1, nullptr, config,
              /*coord=*/nullptr);

    // A distinct, feasible, better-than-reachable point, so adoption WOULD
    // succeed on the first kick if it were ever consulted. Capacity 1 keeps it
    // there against the run's own submissions.
    Model donor = quadratic_model();
    Solution gift;
    gift.state = donor.copy_state();
    gift.state.values[0] = 0.5;
    gift.state.values[1] = 0.5;
    gift.objective = 0.5;
    gift.feasible = true;
    SolutionPool pool(1);
    pool.submit(gift);
    SearchCoordination coord;
    coord.pool = &pool;

    LNS lns_b(0.3);
    Model b = quadratic_model();
    const SearchResult rb = solve(b, /*time_limit=*/0.0, /*seed=*/13, true, nullptr, &lns_b, 1,
                                  nullptr, config, &coord);

    // The cadence was actually reached, or the rest of this proves nothing.
    REQUIRE(ra.lns_repairs > 0);
    REQUIRE(ra.lns_repairs == rb.lns_repairs);
    REQUIRE(ra.iterations == rb.iterations);
    REQUIRE(ra.objective == rb.objective);
    REQUIRE(ra.best_state.values == rb.best_state.values);
}

TEST_CASE("adoption judges feasibility against its own model", "[parallel][coord]") {
    // `Solution::feasible` describes whatever model the SUBMITTER searched, and
    // nothing makes two workers' models identical. This gift is the right SHAPE
    // -- so the size guard passes it through -- flagged feasible, with an
    // objective better than anything reachable, and it flatly violates this
    // model's `x + y >= 1`.
    //
    // Replace `real_feasible()` with `sol->feasible` in adopt_from_pool and the
    // run adopts it as an incumbent, best_state_ becomes the violating point,
    // and the independent verify below goes red on a result reporting
    // feasible = true.
    // The point has to be TEMPTING as well as infeasible, or the test is
    // vacuous: adoption only takes a drawn point as its incumbent when the
    // objective RECOMPUTED at that point improves on what the worker holds, and
    // Solution::objective is used for nothing but the pool's sort order. So
    // x = y = 0 -- objective 0, below the model's feasible optimum of 0.5, and
    // flatly violating x + y >= 1.
    Model donor = quadratic_model();
    Solution liar;
    liar.state = donor.copy_state();
    liar.state.values[0] = 0.0;
    liar.state.values[1] = 0.0;
    liar.objective = -1e9;  // sorts first, so get_restart_point draws it
    liar.feasible = true;   // ...and lies about it

    SolutionPool pool(1);
    pool.submit(liar);
    SearchCoordination coord;
    coord.pool = &pool;

    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;

    Model m = quadratic_model();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/21, true, nullptr, nullptr, 3,
                                 nullptr, config, &coord);

    REQUIRE(r.feasible);
    // The lie never became the answer.
    REQUIRE(r.objective > 0.0);
    Model check = quadratic_model();
    check.restore_state(r.best_state);
    full_evaluate(check);
    const VerifyResult v = verify_model(check);
    INFO("verify errors: " << v.errors.size());
    REQUIRE(v.ok);
}

TEST_CASE("a worker that throws after sharing still propagates", "[parallel]") {
    // record_best() -- and with it share() -- runs BEFORE hook->solve(), so each
    // worker puts an incumbent in the pool and only then dies. The pool is
    // therefore NOT empty when the aggregation runs, and `pool.best()` alone can
    // no longer tell "every worker died" from "a worker succeeded". Drop the
    // all_failed test in src/pool.cpp back to a bare `if (!best)` and this
    // returns a dead worker's mid-run snapshot as a result instead of throwing.
    struct ThrowingHook : InnerSolverHook {
        void solve(Model& /*model*/, ViolationManager& /*vm*/,
                   const std::vector<int32_t>& /*last_changed_vars*/ = {}) override {
            throw std::runtime_error("hook failed");
        }
    };
    auto hook_factory = [](Model&) -> std::shared_ptr<InnerSolverHook> {
        return std::make_shared<ThrowingHook>();
    };
    std::function<std::shared_ptr<LNS>()> no_lns;

    ParallelSearch ps(2);
    ParallelConfig pc;
    pc.n_threads = 2;
    REQUIRE_THROWS_MATCHES(ps.solve([]() { return quadratic_model(); }, 2.0, 42, SearchConfig{},
                                    hook_factory, no_lns, nullptr, pc),
                           std::runtime_error, Catch::Matchers::Message("hook failed"));
}

TEST_CASE("a stop flag raised mid-search ends the run where it stands", "[parallel][coord]") {
    // The other stop-flag test pre-raises the flag, so the loop condition is
    // false on its first evaluation and no batch ever runs. This covers the
    // case the mechanism actually exists for: cancellation at a batch boundary,
    // part-way through a search that is doing work.
    //
    // The flag is raised from inside the search's own progress callback rather
    // than from a sleeping thread, so the test is deterministic and carries no
    // wall-clock assertion (tests/CMakeLists.txt's #104 note asks for exactly
    // that restraint).
    struct StopOnProgress : SolveCallback {
        std::atomic<bool>* flag;
        int seen = 0;
        explicit StopOnProgress(std::atomic<bool>* f) : flag(f) {}
        void on_progress(const SolveProgress& /*p*/) override {
            if (++seen >= 2) {
                flag->store(true, std::memory_order_relaxed);
            }
        }
    };

    std::atomic<bool> stop{false};
    SearchCoordination coord;
    coord.stop = &stop;
    StopOnProgress cb(&stop);

    SearchConfig config;
    config.max_iterations = 10000000;  // would run far past the cancellation
    config.batch_iterations = 100;

    Model m = quadratic_model();
    const SearchResult r =
        solve(m, /*time_limit=*/0.0, /*seed=*/5, true, nullptr, nullptr, 3, &cb, config, &coord);

    REQUIRE(r.termination == TerminationReason::Stopped);
    // The distinguishing assertion: the run was cancelled MID-FLIGHT, not before
    // it started. Without it this passes on the pre-raised case too.
    REQUIRE(r.iterations > 0);
    REQUIRE(r.iterations < config.max_iterations);
}

TEST_CASE("a one-worker portfolio is still a correct portfolio", "[parallel]") {
    // n_threads == 1 is reachable from C++ and from Python (the CLI routes it to
    // the single-threaded path instead, so nothing else covers it). It is not
    // equivalent to a bare solve(): the lone worker's pool holds only its own
    // incumbents, so every adoption it makes is a self-restart. Whatever that
    // does to search quality, the result must still be a valid assignment.
    ParallelSearch ps(1);
    const SearchResult r = ps.solve([]() { return quadratic_model(); }, 1.0, 42);
    REQUIRE(r.feasible);

    Model check = quadratic_model();
    REQUIRE(r.best_state.values.size() == check.num_vars());
    check.restore_state(r.best_state);
    full_evaluate(check);
    REQUIRE(verify_model(check).ok);
}

TEST_CASE("SolutionPool clamps a degenerate capacity", "[pool]") {
    // A direct caller reaches this constructor unvalidated, so the clamp is the
    // guard. A capacity of 0 with no clamp makes submit() resize to 0 and best()
    // return nullopt on a pool that was just handed a solution. (A portfolio no
    // longer gets here with 0 -- effective_pool_capacity intercepts it first --
    // but SolutionPool is public API and this is its own contract.)
    SolutionPool pool(0);
    for (int i = 0; i < 3; ++i) {
        Solution s;
        s.objective = static_cast<double>(i);
        s.feasible = true;
        pool.submit(s);
    }
    REQUIRE(pool.size() == 1);
    auto best = pool.best();
    if (!best.has_value()) {
        FAIL("clamped pool dropped every solution");
        return;
    }
    REQUIRE(best->objective == 0.0);
    // And the read side's own clamp: a negative k is not a buffer underrun.
    REQUIRE(pool.top_k(-1).empty());
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

// ---------------------------------------------------------------------------
// Worker seeding (#135 A3)
// ---------------------------------------------------------------------------

TEST_CASE("adjacent base seeds do not share worker streams", "[parallel][seed]") {
    // The property the old `base + worker + restart * n_threads` scheme did NOT
    // have. It was a correct bijection within one run -- which is all it
    // claimed -- but at 12 workers `--seed 42` and `--seed 43` shared 11 of
    // their 12 base streams, so bumping the seed to draw an independent sample
    // barely changed the portfolio. Now the default worker count.
    constexpr int kWorkers = 12;
    std::set<uint64_t> a;
    std::set<uint64_t> b;
    for (int w = 0; w < kWorkers; ++w) {
        a.insert(portfolio_worker_seed(42, w, 0));
        b.insert(portfolio_worker_seed(43, w, 0));
    }
    REQUIRE(a.size() == kWorkers);  // no collisions within a run
    REQUIRE(b.size() == kWorkers);
    std::vector<uint64_t> shared;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(shared));
    INFO("shared streams between seed 42 and 43: " << shared.size());
    REQUIRE(shared.empty());
}

TEST_CASE("worker seeds are distinct across workers and restarts", "[parallel][seed]") {
    // Within one run the scheme must still be injective over (worker, restart),
    // which is what stops a restart replaying the run that just stalled. This
    // is the property the old scheme had and the new one must not lose.
    std::set<uint64_t> seen;
    int count = 0;
    for (int w = 0; w < 32; ++w) {
        for (int r = 0; r < 32; ++r) {
            seen.insert(portfolio_worker_seed(7, w, r));
            ++count;
        }
    }
    REQUIRE(static_cast<int>(seen.size()) == count);
    // ...and it is a pure function, or a restart would not be reproducible at
    // all for a fixed seed on a fixed schedule.
    REQUIRE(portfolio_worker_seed(7, 3, 5) == portfolio_worker_seed(7, 3, 5));
}

// ---------------------------------------------------------------------------
// A worker that throws is not abandoned (#135 A1)
// ---------------------------------------------------------------------------

namespace {

// Throws on its first `fail_times` invocations, then behaves. Models a
// transient failure inside the search -- a bad_alloc under memory pressure --
// as against a factory that cannot build its model at all.
struct FlakyHook : InnerSolverHook {
    explicit FlakyHook(int fail_times) : remaining_failures(fail_times) {}
    void solve(Model& /*model*/, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/ = {}) override {
        if (remaining_failures > 0) {
            --remaining_failures;
            throw std::runtime_error("transient hook failure");
        }
        ++successes;
    }
    int remaining_failures;
    int successes = 0;
};

}  // namespace

TEST_CASE("a worker whose search throws is restarted, not abandoned", "[parallel]") {
    // Before this, one try/catch wrapped the whole worker: any throw parked the
    // exception and the thread exited, leaving its core idle for the rest of the
    // run while its peers continued -- and the exception was discarded unless
    // EVERY worker had failed. Remove the retry (break instead of continue in
    // run_worker's catch) and this goes red: no hook ever gets past its
    // failures, so `successes` stays 0 and the run is not feasible.
    std::atomic<int> total_successes{0};
    auto hook_factory = [&total_successes](Model&) -> std::shared_ptr<InnerSolverHook> {
        // One failure per worker, then fine.
        struct Reporting : FlakyHook {
            explicit Reporting(std::atomic<int>& sink) : FlakyHook(1), out(sink) {}
            ~Reporting() override { out.fetch_add(successes); }
            std::atomic<int>& out;
        };
        return std::make_shared<Reporting>(total_successes);
    };
    std::function<std::shared_ptr<LNS>()> no_lns;

    ParallelSearch ps(2);
    ParallelConfig pc;
    pc.n_threads = 2;
    SearchResult r;
    REQUIRE_NOTHROW(r = ps.solve([]() { return quadratic_model(); }, 2.0, 42, SearchConfig{},
                                 hook_factory, no_lns, nullptr, pc));
    REQUIRE(r.feasible);
    // The workers got past their failures and kept working, rather than dying
    // on the first throw.
    REQUIRE(total_successes.load() > 0);
}

TEST_CASE("a completed attempt resets a worker's consecutive-failure count", "[parallel]") {
    // run_worker gives up after kMaxWorkerRetries (3) CONSECUTIVE throws; a
    // solve that returns normally resets the count. The sequence here is
    // throw, throw, completed attempt(s), throw: with the reset that last throw
    // is failure 1 of a fresh run and the worker restarts; without it
    // (`consecutive_failures = 0;` deleted) it is failure 3 and the worker stops
    // for good, so the hook is never called again. Red-checked: with that line
    // deleted this fails on `calls > kLateThrowCall + 1` (54 > 54).
    //
    // No clock decides where the late throw lands. An attempt is capped at
    // kCap batches (SearchConfig::max_iterations also bounds batches, and the
    // loop checks it before each one) and the hook runs at most once per batch
    // -- only on a feasible one -- so kCap + 1 uneventful hook calls after the
    // second throw cannot all belong to one attempt: at least one attempt ended
    // between them. The only other ways an attempt ends are this hook throwing
    // (it does not, in that window) and the deadline, after which nothing runs.
    constexpr int kCap = 50;
    constexpr int kEarlyThrows = 2;
    constexpr int kLateThrowCall = kEarlyThrows + kCap + 1;

    struct Scripted : InnerSolverHook {
        void solve(Model& /*model*/, ViolationManager& /*vm*/,
                   const std::vector<int32_t>& /*last_changed_vars*/ = {}) override {
            const int call = calls++;
            if (call < kEarlyThrows || call == kLateThrowCall) {
                late_thrown = late_thrown || call == kLateThrowCall;
                throw std::runtime_error("scripted hook failure");
            }
        }
        int calls = 0;
        bool late_thrown = false;
    };
    // Built on the worker thread and read here only after solve() has joined it.
    std::shared_ptr<Scripted> hook;
    auto hook_factory = [&hook](Model&) -> std::shared_ptr<InnerSolverHook> {
        hook = std::make_shared<Scripted>();
        return hook;
    };
    std::function<std::shared_ptr<LNS>()> no_lns;

    SearchConfig config;
    config.max_iterations = kCap;
    ParallelConfig pc;
    pc.n_threads = 1;
    ParallelSearch ps(1);
    // A model with an objective, so no attempt ends Feasible and stops the run.
    // The wall clock only has to outlast the scripted sequence -- kLateThrowCall
    // + 2 hook calls, each on a batch of a two-variable model -- and the run uses
    // all of it, so it is kept short.
    SearchResult r;
    REQUIRE_NOTHROW(r = ps.solve([]() { return quadratic_model(); }, 1.0, 42, config, hook_factory,
                                 no_lns, nullptr, pc));
    if (hook == nullptr) {
        FAIL("the hook factory was never called");
        return;
    }
    REQUIRE(hook->late_thrown);
    // The worker restarted after the late throw instead of giving up.
    REQUIRE(hook->calls > kLateThrowCall + 1);
    REQUIRE(r.feasible);
}

// ---------------------------------------------------------------------------
// The objective bound after adoption (#135 A2, B3)
// ---------------------------------------------------------------------------

namespace {

// Reads the model's objective bound every time the search polishes a feasible
// point. The hook is the only extension point handed the Model itself, so it is
// the one place a test can watch a value the SearchResult does not carry.
struct BoundWatcher : InnerSolverHook {
    void solve(Model& model, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/ = {}) override {
        bounds.push_back(model.objective_bound());
    }
    std::vector<double> bounds;
};

}  // namespace

TEST_CASE("adopting a worse point does not relax objective pressure", "[parallel][coord]") {
    // record_best rewrites the bound ONLY on a strict improvement over
    // best_feasible_obj_, so a bound loosened by an adoption can never tighten
    // back: the worker searches with the artificial `obj <= bound` row satisfied
    // and no objective signal until it beats its own all-time best. Adoption
    // therefore derives the bound from the TIGHTER of the adopted point and its
    // own incumbent. Delete the `std::min` against best_feasible_obj_ and the
    // observed bound sequence stops being monotone, which is what this asserts.
    //
    // No planted gift: a pool seeded with a deliberately WORSE solution is the
    // wrong setup, because `submit` sorts best-first and trims, so the plant is
    // evicted by the worker's own first incumbent and never drawn. The loosening
    // case arises on its own -- the worker fills the pool with its own improving
    // trajectory, and `get_restart_point` draws from the better HALF, which is
    // its best five, four of which are older and worse than its current best.
    //
    // The model keeps producing new incumbents for a long time, which is what
    // fills a pool with distinct entries.
    SolutionPool pool(10);
    SearchCoordination coord;
    coord.pool = &pool;

    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;

    BoundWatcher watcher;
    Model m = integer_model();
    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/17, true, &watcher, nullptr, 3,
                                 nullptr, config, &coord);

    REQUIRE(r.feasible);
    // The run reached the code path at all: many feasible points polished, and a
    // full pool of its own incumbents to draw worse ones from.
    REQUIRE(watcher.bounds.size() > 5);
    REQUIRE(pool.size() > 1);
    // Monotone non-increasing over every finite bound observed. A loosening
    // adoption shows up here as a step back up.
    double previous = std::numeric_limits<double>::infinity();
    for (double b : watcher.bounds) {
        if (!std::isfinite(b)) {
            continue;  // the pre-incumbent sentinel; see record_best
        }
        INFO("bound went from " << previous << " to " << b);
        REQUIRE(b <= previous + 1e-9);
        previous = b;
    }
}

TEST_CASE("drawing an infeasible pool entry is safe", "[parallel][coord]") {
    // The pool holds infeasible entries: `solve_portfolio` submits each worker's
    // closest approach at the end of a run that never reached feasibility, so a
    // peer can draw one. This pins that doing so is safe -- the adoption
    // re-grounds the DAG, leaves no NaN in the violation machinery, and the run
    // still terminates with a diagnosable result.
    //
    // The model has NO feasible point (x, y in [0, 1] cannot sum to 5), which is
    // what makes the plant reachable at all: `share()` is only called from
    // `record_best`, so a run that never becomes feasible never submits, and the
    // plant is never outranked. That matters -- an earlier version of this test
    // used a solvable model and a capacity-1 pool, and the worker's first
    // feasible submit evicted the plant before any draw, so the test exercised
    // nothing. `submit` sorts feasible-first, so an infeasible entry sorts LAST
    // and the better-half draw can only ever reach it when the pool is entirely
    // infeasible.
    //
    // What this does NOT claim: that the `feasible_here` guard on the bound
    // derivation changes an outcome. It does not have a distinct behavioural
    // observable -- `max_real_violation` excludes the artificial objective row,
    // so even an unmeetable bound leaves feasibility reporting unchanged. The
    // guard is argued from record_best's invariant, not measured, and the
    // sentinel arm that used to sit beside it was deleted for exactly that
    // reason.
    auto build = []() {
        Model m;
        auto x = m.float_var(0, 1);
        auto y = m.float_var(0, 1);
        auto neg1 = m.constant(-1.0);
        auto two = m.constant(2);
        m.add_constraint(m.sum({m.constant(5.0), m.prod(neg1, x), m.prod(neg1, y)}));
        m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
        m.close();
        return m;
    };

    Model donor = build();
    Solution approach;
    approach.state = donor.copy_state();
    approach.state.values[0] = 1.0;
    approach.state.values[1] = 1.0;  // the closest approach: violation 3
    approach.objective = 2.0;
    approach.feasible = false;  // honestly flagged, as solve_portfolio submits it

    SolutionPool pool(10);
    pool.submit(approach);
    SearchCoordination coord;
    coord.pool = &pool;

    SearchConfig config;
    config.max_iterations = 5000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;

    Model m = build();
    SearchResult r;
    REQUIRE_NOTHROW(r = solve(m, /*time_limit=*/0.0, /*seed=*/23, true, nullptr, nullptr, 3,
                              nullptr, config, &coord));

    REQUIRE_FALSE(r.feasible);
    // The diagnosable part: a NaN anywhere in the bound or the re-grounding
    // poisons the violation machinery, and this is where it would surface.
    REQUIRE(std::isfinite(r.best_violation));
    REQUIRE(r.best_violation > 0.0);
    // ...and the state handed back is still a state of this model.
    REQUIRE(r.best_state.values.size() == m.num_vars());
    // The plant survived to be drawable, which is what makes the above non-vacuous.
    REQUIRE(pool.size() == 1);
}

TEST_CASE("the portfolio pool capacity scales with the worker count", "[parallel]") {
    // The auto rule behind ParallelConfig::pool_capacity = 0. A pool smaller
    // than the worker count cannot represent the portfolio at all: `submit`
    // sorts globally and applies no per-worker quota, so at 32 workers a
    // capacity of 10 holds the ten best objectives and nothing else.
    //
    // Replace effective_pool_capacity's body with `return requested;` and every
    // portfolio pool falls back to SolutionPool's own max(1, ...) clamp, i.e.
    // capacity 1 -- which still produces feasible, verifiable results, so only a
    // direct test of the rule catches it.
    REQUIRE(effective_pool_capacity(0, 32) == 64);
    REQUIRE(effective_pool_capacity(0, 12) == 24);
    // Floored at 10, so a small portfolio is not handed a pool too small to hold
    // a useful spread.
    REQUIRE(effective_pool_capacity(0, 1) == 10);
    REQUIRE(effective_pool_capacity(0, 4) == 10);
    // An explicit request wins, however small.
    REQUIRE(effective_pool_capacity(7, 32) == 7);
    REQUIRE(effective_pool_capacity(1, 32) == 1);
    // Negative is a caller error and folds into auto rather than silently
    // becoming a one-solution pool.
    REQUIRE(effective_pool_capacity(-1, 12) == 24);
}

// ---------------------------------------------------------------------------
// TerminationReason::Stopped, end to end (#135 B)
// ---------------------------------------------------------------------------

TEST_CASE("every TerminationReason has a distinct stable token", "[parallel][search]") {
    // The tokens are the JSONL `termination` field, so they are a machine
    // contract, and `Stopped` was added to the enum by the cooperative
    // portfolio. A missing `case` is a -Wswitch warning rather than a test
    // failure, and a token that silently duplicated another would not even be
    // that.
    const std::vector<TerminationReason> all = {
        TerminationReason::TimeLimit, TerminationReason::IterationLimit,
        TerminationReason::Feasible,  TerminationReason::NoBudget,
        TerminationReason::Stopped,
    };
    std::set<std::string> tokens;
    for (TerminationReason t : all) {
        const char* name = termination_reason_name(t);
        REQUIRE(name != nullptr);
        REQUIRE(std::string(name) != "unknown");
        tokens.insert(name);
    }
    REQUIRE(tokens.size() == all.size());
    REQUIRE(std::string(termination_reason_name(TerminationReason::Stopped)) == "stopped");
}

// ---------------------------------------------------------------------------
// ParallelConfig::pool_capacity actually reaches the pool (#135 B)
// ---------------------------------------------------------------------------

TEST_CASE("pool_capacity reaches the pool the workers share", "[parallel]") {
    // The field was plumbed from Python through ParallelConfig to the
    // SolutionPool constructor and exercised by nothing: deleting the parameter
    // from solve_portfolio left the whole suite green.
    //
    // One worker and an iteration-only budget make the run fully deterministic
    // (no wall clock, and the no-deadline break means exactly one solve), so the
    // only thing separating the two arms is how many solutions the pool keeps --
    // which changes what get_restart_point can draw, and so the trajectory.
    SearchConfig config;
    config.max_iterations = 20000;
    config.batch_iterations = 100;
    config.perturbation_period = 2;

    auto run = [&config](int capacity) {
        ParallelConfig pc;
        pc.n_threads = 1;
        pc.pool_capacity = capacity;
        ParallelSearch ps(1);
        return ps.solve([]() { return integer_model(); }, /*time_limit=*/0.0, /*seed=*/31, config,
                        nullptr, nullptr, nullptr, pc);
    };

    const SearchResult tight = run(1);
    const SearchResult loose = run(10);

    REQUIRE(tight.feasible);
    REQUIRE(loose.feasible);
    // Same seed, same budget, same model, one worker: if the capacity never
    // reached the pool these two runs would be bit-identical.
    //
    // The assertion is on ITERATIONS, not on the objective or the state: both
    // arms converge to this model's optimum, so those agree while the
    // trajectories differ. The iteration count is where the difference shows --
    // measured 20 029 against 20 038 -- and it is exact, because an
    // iteration-only budget makes a one-worker portfolio fully deterministic
    // (no wall clock, and the no-deadline break means exactly one solve).
    INFO("capacity 1 -> " << tight.iterations << " iterations, capacity 10 -> "
                          << loose.iterations);
    REQUIRE(tight.iterations != loose.iterations);
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

// ---------------------------------------------------------------------------
// Move generators are per worker (#165)
// ---------------------------------------------------------------------------

namespace {

/// Records, through a mutex-guarded registry, every clone this prototype
/// produced and how many commits each of those clones saw.
///
/// The counter that matters is `own_commits_`, which is per INSTANCE: if the
/// portfolio handed every worker the same generator object, the prototype's own
/// counter would move and the per-clone counts would not add up. A generator is
/// free to hold exactly this kind of state -- a cache, a cursor, a counter --
/// which is why cloning rather than sharing is the contract.
struct CloneRegistry {
    std::mutex mu;
    /// Identity TOKENS, not generator addresses. A clone dies when its worker
    /// restarts, and the allocator hands the very same address straight back to
    /// the next clone on that thread -- so comparing raw `this` pointers reports
    /// a duplicate for a perfectly correct restart, and whether it does depends
    /// on whether the budget produced one. The registry keeps each token alive,
    /// which is what makes its address unique for the whole run.
    std::vector<std::shared_ptr<const int>> identities;
    std::vector<int> per_clone_commits;
    int total_commits = 0;
};

class CountingGenerator final : public MoveGenerator {
public:
    CountingGenerator(int32_t var_id, std::shared_ptr<CloneRegistry> registry)
        : var_id_(var_id),
          registry_(std::move(registry)),
          identity_(std::make_shared<const int>(0)) {}

    ~CountingGenerator() override {
        if (registered_) {
            const std::scoped_lock lock(registry_->mu);
            registry_->per_clone_commits.push_back(own_commits_);
        }
    }
    CountingGenerator(const CountingGenerator&) = delete;
    CountingGenerator& operator=(const CountingGenerator&) = delete;
    CountingGenerator(CountingGenerator&&) = delete;
    CountingGenerator& operator=(CountingGenerator&&) = delete;

    [[nodiscard]] std::string_view name() const override { return "counting"; }
    [[nodiscard]] ConstSpan<int32_t> scope() const override { return {&var_id_, 1}; }

    void generate(MoveContext& ctx, std::vector<Move>& out) override {
        generate_standard_moves(ctx.model.var(var_id_), ctx.rng, out, nullptr);
    }

    void on_commit(const Move& /*move*/) override {
        ++own_commits_;
        const std::scoped_lock lock(registry_->mu);
        ++registry_->total_commits;
    }

    [[nodiscard]] int own_commits() const { return own_commits_; }

    [[nodiscard]] std::unique_ptr<MoveGenerator> clone() const override {
        auto copy = std::make_unique<CountingGenerator>(var_id_, registry_);
        copy->registered_ = true;
        const std::scoped_lock lock(registry_->mu);
        registry_->identities.push_back(copy->identity_);
        return copy;
    }

private:
    int32_t var_id_;
    std::shared_ptr<CloneRegistry> registry_;
    std::shared_ptr<const int> identity_;
    int own_commits_ = 0;
    bool registered_ = false;  // only a clone publishes its count on destruction
};

Model set_cover_toy() {
    Model m;
    Expr chosen = m.Set(12, 1, 6, "chosen");
    for (int r = 0; r < 4; ++r) {
        const int base = r;
        Expr covered(
            &m, m.lambda_sum(chosen.handle, [base](int e) { return (e % 4 == base) ? 1.0 : 0.0; }));
        m.add_constraint(covered >= m.Constant(1.0));
    }
    m.minimize(m.lambda_sum(chosen.handle, [](int e) { return 1.0 + (0.1 * e); }));
    m.close();
    return m;
}

}  // namespace

TEST_CASE("each portfolio worker gets its own move generator", "[parallel][structural]") {
    // A generator registered on SearchConfig is SHARED by every worker's config
    // copy -- `SearchConfig cfg = ctx.config` copies a vector of shared_ptr, not
    // the generators. What keeps that from being a data race is that each
    // worker's StructuralBatch clones what it was given, so the registered
    // instance is never touched by the search.
    auto registry = std::make_shared<CloneRegistry>();
    Model master = set_cover_toy();
    const int32_t set_var = 0;
    auto prototype = std::make_shared<CountingGenerator>(set_var, registry);

    SearchConfig cfg;
    cfg.default_structural_generators = false;  // only ours, so the counts are ours
    cfg.structural_batch_probability = 1.0;
    cfg.batch_iterations = 50;
    cfg.move_generators.push_back(prototype);

    constexpr int kThreads = 4;
    ParallelSearch ps(kThreads);
    ParallelConfig par_config;
    par_config.n_threads = kThreads;
    const SearchResult r =
        ps.solve(master, /*time_limit=*/0.5, /*seed=*/42, cfg, /*hook_factory=*/nullptr,
                 /*lns_factory=*/nullptr, /*callback=*/nullptr, par_config);
    REQUIRE(r.iterations >= 0);

    const std::scoped_lock lock(registry->mu);
    // One clone per worker at least (a restarted worker builds another).
    REQUIRE(registry->identities.size() >= static_cast<size_t>(kThreads));
    std::vector<const void*> sorted;
    sorted.reserve(registry->identities.size());
    for (const std::shared_ptr<const int>& token : registry->identities) {
        sorted.push_back(token.get());
    }
    std::sort(sorted.begin(), sorted.end());
    REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
    // The instance the caller registered was never run, so its own state is
    // untouched -- the property that makes sharing the registration safe.
    REQUIRE(prototype->own_commits() == 0);
    // Every commit was attributed to exactly one clone: none lost, none double
    // counted, which is what a shared mutable generator would break.
    REQUIRE(registry->total_commits > 0);
    int summed = 0;
    for (int c : registry->per_clone_commits) {
        summed += c;
    }
    REQUIRE(summed == registry->total_commits);
}
