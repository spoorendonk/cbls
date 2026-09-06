#include "test_helpers.h"

#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace cbls;

// Violation tests
TEST_CASE("No violation when feasible", "[violation]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    m.add_constraint(m.sum({x, neg5}));  // x - 5 <= 0
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.total_violation() == 0.0);
    REQUIRE(vm.is_feasible());
}

TEST_CASE("Violation when infeasible", "[violation]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    m.add_constraint(m.sum({x, neg5}));
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE(vm.total_violation() == 3.0);
    REQUIRE_FALSE(vm.is_feasible());
}

TEST_CASE("Augmented objective", "[violation]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto neg5 = m.constant(-5.0);
    m.add_constraint(m.sum({x, neg5}));
    m.minimize(m.sum({x}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    full_evaluate(m);
    ViolationManager vm(m);
    // Penalty-method objective with unit weights: f + V = 8 + 3 = 11.
    REQUIRE(vm.augmented_objective() == 11.0);
}

// FJ-NL tests
TEST_CASE("FJ-NL finds feasibility simple", "[search]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    // constraint: 4 - x - y <= 0 (i.e. x + y >= 4)
    auto neg1_node = m.constant(-1.0);
    auto xy_sum = m.sum({x, y});
    auto neg_xy = m.prod(neg1_node, xy_sum);
    auto c = m.sum({neg1_node, neg_xy, m.constant(5.0)});
    m.add_constraint(c);
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE_FALSE(vm.is_feasible());

    RNG rng(42);
    // time_limit 0: bounded by the iteration budget alone, so a loaded machine
    // cannot cut the repair short and turn the assertion below into a flake.
    fj_nl_initialize(m, vm, 1000, &rng, /*time_limit=*/0.0);
    full_evaluate(m);
    REQUIRE(vm.is_feasible());
}

TEST_CASE("FJ-NL finds feasibility bool", "[search]") {
    Model m;
    auto x = m.bool_var();
    auto y = m.bool_var();
    auto neg1 = m.constant(-1.0);
    auto neg_x = m.prod(neg1, x);
    auto neg_y = m.prod(neg1, y);
    auto one = m.constant(1.0);
    m.add_constraint(m.sum({one, neg_x, neg_y}));  // 1 - x - y <= 0
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);
    REQUIRE_FALSE(vm.is_feasible());

    RNG rng(42);
    fj_nl_initialize(m, vm, 100, &rng, /*time_limit=*/0.0);
    full_evaluate(m);
    REQUIRE(vm.is_feasible());
}

// SA solver tests
TEST_CASE("SA unconstrained minimum", "[search]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    auto result = solve_deterministic(m, 947000, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 1.0);
}

TEST_CASE("SA constrained problem", "[search]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto three = m.constant(3.0);
    m.add_constraint(m.sum({three, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    auto result = solve_deterministic(m, 1110000, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 5.0);
}

TEST_CASE("SA integer problem", "[search]") {
    Model m;
    auto x = m.int_var(0, 10);
    auto neg7 = m.constant(-7.0);
    m.minimize(m.abs_expr(m.sum({x, neg7})));
    m.close();

    auto result = solve_deterministic(m, 895000, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 2.0);
}

TEST_CASE("SA Rosenbrock 2D", "[search]") {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto one = m.constant(1.0);
    auto neg1 = m.constant(-1.0);
    auto two = m.constant(2.0);
    auto hundred = m.constant(100.0);

    auto one_minus_x = m.sum({one, m.prod(neg1, x)});
    auto term1 = m.pow_expr(one_minus_x, two);

    auto y_minus_x2 = m.sum({y, m.prod(neg1, m.pow_expr(x, two))});
    auto term2 = m.prod(hundred, m.pow_expr(y_minus_x2, two));

    m.minimize(m.sum({term1, term2}));
    m.close();

    auto result = solve_deterministic(m, 1117000, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 50.0);
}

TEST_CASE("SA returns result", "[search]") {
    Model m;
    auto x = m.float_var(0, 1);
    m.minimize(m.sum({x}));
    m.close();

    auto result = solve(m, 0.5, 42);
    REQUIRE(result.iterations > 0);
    REQUIRE(result.time_seconds > 0);
}

// LNS test
TEST_CASE("LNS basic", "[lns]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto five = m.constant(5.0);
    m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 8.0;
    m.var_mut(vid(y)).value = 8.0;
    full_evaluate(m);
    ViolationManager vm(m);

    LNS lns(0.5);
    RNG rng(42);
    lns.destroy_repair_cycle(m, vm, rng, 5);
    full_evaluate(m);
    // Just check it doesn't crash
}

// LNS integration in solve() test
TEST_CASE("solve with LNS param", "[search][lns]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto five = m.constant(5.0);
    m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    LNS lns(0.5);
    auto result = solve_deterministic(m, 746000, 42, nullptr, &lns);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 10.0);
}

TEST_CASE("solve with hook and LNS", "[search][lns]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto five = m.constant(5.0);
    m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    FloatIntensifyHook hook;
    LNS lns(0.3);
    auto result = solve_deterministic(m, 745000, 42, &hook, &lns);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 8.0);
}

// This test was inert (#104). Its model — 50 vars in [0, 100] summing to 2500 —
// is satisfiable and converged in ~2ms against a 50ms cap, so FJ returned long
// before the deadline could matter and the `elapsed < 0.5` assertion held
// whether or not time_limit was honoured at all. Two changes make it real:
//
//   * the domain is [0, 10], so 50 variables sum to at most 500 and the equality
//     is unreachable. FJ can now never converge, and only a budget can stop it.
//   * the assertion is on the iteration count fj_nl_initialize returns, not on
//     elapsed time. Stopping far short of the iteration budget is something only
//     the clock can have done.
//
// What remains machine-dependent is the ratio between the two budgets: the
// machine must not be able to spend 30k GLS iterations in 20ms. Measured for
// *this* test: 192 iterations, so ~156x of margin, and a regression takes ~4.9s
// to go red. (The batch test above is a different model and has its own
// numbers — 128 iterations, ~312x, ~10s red. Don't copy one set to the other.)
// A slower or loaded machine does fewer iterations and passes more comfortably.
TEST_CASE("fj_nl_initialize respects time_limit", "[search][fj]") {
    Model m;
    std::vector<int32_t> vars;
    vars.reserve(50);
    for (int i = 0; i < 50; ++i) {
        vars.push_back(m.int_var(0, 10));
    }
    std::vector<int32_t> sum_args(vars.begin(), vars.end());
    sum_args.push_back(m.constant(-2500.0));
    m.add_constraint(m.abs_expr(m.sum(sum_args)));
    m.close();

    ViolationManager vm(m);
    RNG rng(42);
    initialize_random(m, rng);
    full_evaluate(m);

    constexpr int kIterationBudget = 30000;
    int64_t iterations = fj_nl_initialize(m, vm, kIterationBudget, &rng, /*time_limit=*/0.02);

    REQUIRE(iterations > 0);                 // not inert: it really did run
    REQUIRE(iterations < kIterationBudget);  // and the clock, not the budget, stopped it
}

TEST_CASE("fj_nl_initialize respects max_iterations", "[search][fj]") {
    // The complement of the test above, and what makes its `<` assertion mean
    // something: with the wall clock switched off the same unreachable model runs
    // exactly as far as the iteration budget allows, so the count is not merely
    // "small" — it is the budget, deterministically.
    Model m;
    std::vector<int32_t> vars;
    vars.reserve(50);
    for (int i = 0; i < 50; ++i) {
        vars.push_back(m.int_var(0, 10));
    }
    std::vector<int32_t> sum_args(vars.begin(), vars.end());
    sum_args.push_back(m.constant(-2500.0));
    m.add_constraint(m.abs_expr(m.sum(sum_args)));
    m.close();

    ViolationManager vm(m);
    RNG rng(42);
    initialize_random(m, rng);
    full_evaluate(m);

    REQUIRE(fj_nl_initialize(m, vm, /*max_iterations=*/500, &rng, /*time_limit=*/0.0) == 500);
}

TEST_CASE("lns_interval=0 disables LNS in solve", "[search][lns]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto five = m.constant(5.0);
    m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    LNS lns(0.5);
    // lns_interval=0 should disable LNS entirely (no division by zero)
    auto result = solve_deterministic(m, 376000, 42, nullptr, &lns, 0);
    REQUIRE(result.feasible);
    REQUIRE(result.iterations > 0);
}

TEST_CASE("lns_interval gates LNS frequency", "[search][lns]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto five = m.constant(5.0);
    m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
    m.minimize(m.sum({x, y}));
    m.close();

    // With high lns_interval, LNS rarely fires — solve should still work
    LNS lns(0.5);
    auto result = solve_deterministic(m, 378000, 42, nullptr, &lns, 100);
    REQUIRE(result.feasible);
    REQUIRE(result.iterations > 0);
}

// Solution pool test
TEST_CASE("SolutionPool ordering", "[pool]") {
    SolutionPool pool(3);
    Model::State empty_state;

    pool.submit({empty_state, 10.0, true});
    pool.submit({empty_state, 5.0, true});
    pool.submit({empty_state, 20.0, false});
    pool.submit({empty_state, 3.0, true});

    auto best = pool.best();
    // Guarded with FAIL + return rather than REQUIRE(best.has_value()): both
    // abort the test, but REQUIRE is a Catch2 expression template that
    // bugprone-unchecked-optional-access cannot see through, so the accesses
    // below would read as unchecked -- and .value() reads that way to it too.
    if (!best.has_value()) {
        FAIL("SolutionPool::best() returned nothing after four submissions");
        return;
    }
    REQUIRE(best->objective == 3.0);
    REQUIRE(best->feasible);
    REQUIRE(pool.size() == 3);
}

// Error-path tests
TEST_CASE("Out-of-range var throws", "[model][error]") {
    Model m;
    m.float_var(0, 1);
    REQUIRE_THROWS_AS(m.var(999), std::out_of_range);
    REQUIRE_THROWS_AS(m.var(-1), std::out_of_range);
}

TEST_CASE("Out-of-range node throws", "[model][error]") {
    Model m;
    REQUIRE_THROWS_AS(m.node(0), std::out_of_range);
    m.constant(1.0);
    REQUIRE_THROWS_AS(m.node(999), std::out_of_range);
}

TEST_CASE("add_constraint rejects var handle", "[model][error]") {
    Model m;
    auto x = m.float_var(0, 10);
    REQUIRE_THROWS_AS(m.add_constraint(x), std::invalid_argument);
}

TEST_CASE("minimize rejects var handle", "[model][error]") {
    Model m;
    auto x = m.float_var(0, 10);
    REQUIRE_THROWS_AS(m.minimize(x), std::invalid_argument);
}

// State snapshot/restore tests
TEST_CASE("copy_state and restore_state", "[model]") {
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    m.minimize(m.sum({x, y}));
    m.close();

    m.var_mut(vid(x)).value = 3.0;
    m.var_mut(vid(y)).value = 7.0;
    auto state = m.copy_state();

    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 2.0;
    REQUIRE(m.var(vid(x)).value == 1.0);
    REQUIRE(m.var(vid(y)).value == 2.0);

    m.restore_state(state);
    REQUIRE(m.var(vid(x)).value == 3.0);
    REQUIRE(m.var(vid(y)).value == 7.0);
}

// Helper: factory for a simple x^2 + y^2 model
static std::function<Model()> simple_model_factory() {
    return []() {
        Model m;
        auto x = m.float_var(-5, 5);
        auto y = m.float_var(-5, 5);
        auto two = m.constant(2);
        m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
        m.close();
        return m;
    };
}

// ParallelSearch test
TEST_CASE("ParallelSearch basic", "[pool]") {
    ParallelSearch ps(2);
    auto result = ps.solve(simple_model_factory(), 1.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 5.0);
    // The portfolio composes a fresh SearchResult from its workers', so it has to
    // aggregate their termination reasons rather than report the default. This
    // model has an objective, so no worker can break out on Feasible, and the
    // only budget any of them was given is the 1s wall clock.
    REQUIRE(result.termination == TerminationReason::TimeLimit);
}

TEST_CASE("ParallelSearch default threads uses hardware_concurrency", "[pool]") {
    ParallelSearch ps;  // n_threads=0 -> hardware_concurrency()
    auto result = ps.solve(simple_model_factory(), 1.0, 42);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 5.0);
}

TEST_CASE("ParallelSearch with hook and LNS factories", "[pool]") {
    auto factory = []() {
        Model m;
        auto x = m.float_var(0, 10);
        auto y = m.float_var(0, 10);
        auto neg1 = m.constant(-1.0);
        auto five = m.constant(5.0);
        m.add_constraint(m.sum({five, m.prod(neg1, x), m.prod(neg1, y)}));
        m.minimize(m.sum({x, y}));
        m.close();
        return m;
    };

    // Counting subclasses rather than the stock types: the factory contract is
    // an OWNERSHIP contract -- one object per worker, every one released before
    // solve() returns -- and #129 changed the pointer type precisely so Python
    // could share in it. Nothing asserted the C++ half, so a change that
    // hoisted the shared_ptr out of the worker, reused one object for every
    // worker, or leaked them into a static would have stayed green here.
    //
    // Build/destroy counts alone are not enough: they stay green on a search
    // that constructs both objects and then wires in neither. The overrides
    // below count *dispatches*, so the object the factory returned has to be
    // the one the search actually runs.
    std::atomic<int> hooks_built{0};
    std::atomic<int> hooks_destroyed{0};
    std::atomic<int> hook_calls{0};
    std::atomic<int> lns_built{0};
    std::atomic<int> lns_destroyed{0};
    std::atomic<int> lns_calls{0};

    struct CountingHook : FloatIntensifyHook {
        CountingHook(std::atomic<int>& d, std::atomic<int>& c) : destroyed(d), calls(c) {}
        ~CountingHook() override { destroyed.fetch_add(1); }
        void solve(Model& model, ViolationManager& vm,
                   const std::vector<int32_t>& last_changed_vars = {}) override {
            calls.fetch_add(1);
            FloatIntensifyHook::solve(model, vm, last_changed_vars);
        }
        std::atomic<int>& destroyed;
        std::atomic<int>& calls;
    };
    struct CountingLNS : LNS {
        CountingLNS(std::atomic<int>& d, std::atomic<int>& c) : LNS(0.3), destroyed(d), calls(c) {}
        ~CountingLNS() override { destroyed.fetch_add(1); }
        bool destroy_repair(Model& model, ViolationManager& vm, RNG& rng,
                            double repair_time_limit) override {
            calls.fetch_add(1);
            return LNS::destroy_repair(model, vm, rng, repair_time_limit);
        }
        std::atomic<int>& destroyed;
        std::atomic<int>& calls;
    };

    auto hook_factory = [&](Model&) -> std::shared_ptr<InnerSolverHook> {
        hooks_built.fetch_add(1);
        return std::make_shared<CountingHook>(hooks_destroyed, hook_calls);
    };
    auto lns_factory = [&]() -> std::shared_ptr<LNS> {
        lns_built.fetch_add(1);
        return std::make_shared<CountingLNS>(lns_destroyed, lns_calls);
    };

    ParallelSearch ps(2);
    ParallelConfig pc;
    pc.n_threads = 2;
    auto result = ps.solve(factory, 2.0, 42, {}, hook_factory, lns_factory, nullptr, pc);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 15.0);
    // Read after solve() returned, i.e. after join_all(): one object per worker,
    // each destroyed exactly once, none of them outliving the call.
    REQUIRE(hooks_built.load() == 2);
    REQUIRE(hooks_destroyed.load() == 2);
    REQUIRE(lns_built.load() == 2);
    REQUIRE(lns_destroyed.load() == 2);
    // The object the factory returned is the one the search ran, not merely one
    // it built and dropped. Without this the suite passes on a solve() call
    // that hands nullptr to both slots.
    REQUIRE(hook_calls.load() > 0);
}

// A portfolio worker cannot let an exception escape its thread function, so
// solve_portfolio catches everything. Swallowing it outright made a run in which
// *every* worker failed indistinguishable from one that searched and found
// nothing: the aggregate was the default SearchResult, i.e. feasible=false with
// an infinite objective -- a model-loading failure reported as an infeasible
// model. A total failure now propagates; a partial one still does not.
TEST_CASE("ParallelSearch propagates a factory that fails in every worker", "[pool]") {
    auto factory = []() -> Model { throw std::runtime_error("model factory failed"); };

    ParallelSearch ps(2);
    // Match the message, not just the type: solve_portfolio's "no result and no
    // error" guard on the same path is a std::runtime_error too, so a type-only
    // assertion would still pass with the rethrow loop deleted.
    REQUIRE_THROWS_MATCHES(ps.solve(factory, 0.5, 42), std::runtime_error,
                           Catch::Matchers::Message("model factory failed"));
}

// The other half of that contract: one worker failing must not lose the
// others' work, which is what catching per worker is for.
TEST_CASE("ParallelSearch absorbs a factory that fails in one worker", "[pool]") {
    std::atomic<int> calls{0};
    auto base = simple_model_factory();
    std::function<Model()> factory = [&calls, base]() -> Model {
        if (calls.fetch_add(1) == 0) {
            throw std::runtime_error("model factory failed");
        }
        return base();
    };

    ParallelSearch ps(2);
    SearchResult result;
    REQUIRE_NOTHROW(result = ps.solve(factory, 1.0, 42));
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 5.0);
}

TEST_CASE("Deterministic mode produces identical results", "[pool][deterministic]") {
    auto factory = simple_model_factory();

    ParallelConfig pc;
    pc.n_threads = 2;
    pc.deterministic = true;
    pc.epoch_iterations = 5000;
    pc.max_epochs = 3;
    pc.elite_pool_size = 2;

    ParallelSearch ps1(2);
    auto r1 = ps1.solve(factory, 999.0, 42, {}, nullptr, nullptr, nullptr, pc);

    ParallelSearch ps2(2);
    auto r2 = ps2.solve(factory, 999.0, 42, {}, nullptr, nullptr, nullptr, pc);

    REQUIRE(r1.feasible);
    REQUIRE(r2.feasible);
    REQUIRE(r1.objective == r2.objective);
    REQUIRE(r1.iterations == r2.iterations);
    // Epoch-sync runs every epoch with no wall clock at all, by design, so the
    // run is iteration-bounded and this can never come back TimeLimit — which is
    // the property that keeps the mode deterministic in the first place.
    REQUIRE(r1.termination == TerminationReason::IterationLimit);
    REQUIRE(r1.termination == r2.termination);
}

TEST_CASE("max_iterations stops SA by iteration count", "[search]") {
    Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    SearchConfig config;
    config.max_iterations = 1000;
    // time_limit 0 = no wall clock at all: the iteration budget is the only
    // thing that can stop this, which is exactly what the test asserts.
    auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);
    // The budget is checked at batch boundaries, so the count lands in
    // [max_iterations, max_iterations + batch_iterations). Both ends matter: the
    // lower proves the budget was actually spent rather than an early break, the
    // upper that it stopped near the budget rather than running unbounded.
    REQUIRE(result.iterations >= 1000);
    REQUIRE(result.iterations < 1000 + config.batch_iterations);
    REQUIRE(result.termination == TerminationReason::IterationLimit);
}

// Issue #104: the sibling of the test above, with the wall clock switched on.
// The original version of that test passed a generous time_limit alongside
// max_iterations and so covered the *interaction* — both budgets armed, the
// iteration budget wins. Converting the suite to time_limit = 0 dropped that,
// leaving nothing to catch a change that let a live clock cut an
// iteration-budgeted run short (or that stopped honouring max_iterations once a
// deadline existed).
TEST_CASE("max_iterations wins over a live wall clock", "[search]") {
    auto build = [](Model& m) {
        auto x = m.float_var(-5, 5);
        auto y = m.float_var(-5, 5);
        auto two = m.constant(2);
        m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
        m.close();
    };

    SearchConfig config;
    config.max_iterations = 1000;

    // 30s is unreachable: this model spends its 1000 iterations in well under a
    // millisecond, so the clock is armed but can never be the thing that fires.
    Model timed;
    build(timed);
    auto result = solve(timed, /*time_limit=*/30.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr,
                        3, nullptr, config);

    REQUIRE(result.termination == TerminationReason::IterationLimit);
    // Same bounds as the no-clock case: the lower proves the budget was actually
    // spent rather than an early break, the upper that it stopped near the budget.
    REQUIRE(result.iterations >= 1000);
    REQUIRE(result.iterations < 1000 + config.batch_iterations);

    // Stronger than the bounds: arming a clock this run cannot exhaust must not
    // perturb the search. Reading the clock consumes no randomness, so the two
    // runs follow the same path and land on the same answer.
    //
    // Read that as scoped to *this* configuration rather than as a general
    // guarantee, which it stopped being in #117: the escape probe now also arms
    // once a quarter of the budget passes with no new best, so a timed and an
    // untimed run of the same model diverge once the budget is short relative to
    // the run. Here the budget is 30s against a sub-millisecond run, so the
    // wall-clock arming condition is never reached and the two stay identical.
    Model untimed;
    build(untimed);
    auto reference = solve(untimed, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr,
                           nullptr, 3, nullptr, config);
    REQUIRE(result.iterations == reference.iterations);
    REQUIRE(result.objective == reference.objective);
}

// The two TerminationReason branches the deadline tests below never reach.
// Together with those, this covers every value the enum can take, so none of
// them is an untested claim on SearchResult.
TEST_CASE("termination reason covers the non-budget exits", "[search]") {
    SECTION("a pure-feasibility model stops on its first solution") {
        // No objective, so there is nothing to improve after the first feasible
        // point and solve() returns it immediately — neither budget is consulted.
        Model m;
        auto x = m.float_var(0, 10);
        m.add_constraint(m.leq(m.constant(3.0), x));  // 3 <= x
        m.close();

        SearchConfig config;
        config.max_iterations = 100000;
        auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr,
                            3, nullptr, config);

        REQUIRE(result.feasible);
        REQUIRE(result.termination == TerminationReason::Feasible);
        REQUIRE(result.iterations < config.max_iterations);
    }

    SECTION("no budget at all returns immediately having done nothing") {
        // Neither a wall clock nor an iteration budget. The loop must return
        // rather than spin forever, and must say so rather than claim a limit it
        // was never given.
        Model m;
        auto x = m.float_var(-5, 5);
        auto two = m.constant(2);
        m.minimize(m.sum({m.pow_expr(x, two)}));
        m.close();

        auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr,
                            3, nullptr, SearchConfig{});

        REQUIRE(result.termination == TerminationReason::NoBudget);
        REQUIRE(result.iterations == 0);
    }
}

// ---------------------------------------------------------------------------
// Wall-clock budget enforcement (#104)
//
// `solve(model, time_limit)` is a promise to return within time_limit. Four
// sub-steps can overrun it when entered just before the deadline, and each is
// bounded separately in search.cpp:
//
//   1. a Feasibility Jump batch — handed the same absolute deadline, so it stops
//      mid-batch instead of finishing its 1000 GLS iterations;
//   2. the InnerSolverHook — not started at all when the budget is already spent
//      (a hook is arbitrary user code and unbounded in time);
//   3. the LNS repair — handed whatever budget is left, not its own 2s;
//   4. the STRUCTURAL sweep — checked between variables (#105, tested below).
//
// A bug in this class cost a 60s budget 87s and was found only by reading a
// benchmark's wall_seconds column by hand. Epic #87 publishes per-instance wall
// times, so a silently overrunning budget corrupts published data.
//
// NONE OF THE TESTS BELOW ASSERTS ON ELAPSED TIME. The suite was deliberately
// converted to be deterministic — iteration-bounded, time_limit = 0 — and a
// duration assertion reintroduces machine-speed coupling. The STRUCTURAL test is
// the single grandfathered exception and is quarantined behind [timing]. Each
// test here instead observes its bound directly:
//
//   1. work done: one batch cannot have run to completion;
//   2. the call count of a test InnerSolverHook: was it started at all?
//   3. the argument a test LNS was handed: the remaining budget, or 2s?
//
// Each also asserts SearchResult::termination, so a test cannot quietly go inert
// by converging before the clock ever mattered. That is not hypothetical: it is
// what had happened to "fj_nl_initialize respects time_limit", whose model
// converged in ~2ms against a 50ms cap, so it passed whether or not the limit
// was honoured.
// ---------------------------------------------------------------------------

namespace {

// A model no assignment satisfies, so no batch can exit early on feasibility and
// the wall clock is the only thing that can end the run. 50 variables capped at
// 10 sum to at most 500, so |sum - 2500| = 0 is unreachable.
void build_unsatisfiable(Model& m) {
    std::vector<int32_t> vars;
    vars.reserve(50);
    for (int i = 0; i < 50; ++i) {
        vars.push_back(m.int_var(0, 10));
    }
    std::vector<int32_t> args(vars.begin(), vars.end());
    args.push_back(m.constant(-2500.0));
    m.add_constraint(m.abs_expr(m.sum(args)));
    m.minimize(m.sum(vars));  // an objective, so the pure-feasibility break cannot fire
    m.close();
}

// Records that solve() started it, and does nothing else. What is under test is
// solve()'s refusal to *begin* hook work with no budget left — the only thing it
// can guarantee about an arbitrary user hook, whose running time it cannot know.
class CountingHook : public InnerSolverHook {
public:
    int calls = 0;
    void solve(Model& /*model*/, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/) override {
        ++calls;
    }
};

// Records the repair budget solve() hands it. Deliberately does not destroy or
// repair anything: the argument is what is under test, and a real repair would
// mutate the search state and make the run harder to reason about.
class BudgetRecordingLNS : public LNS {
public:
    std::vector<double> repair_limits;
    bool destroy_repair(Model& /*model*/, ViolationManager& /*vm*/, RNG& /*rng*/,
                        double repair_time_limit) override {
        repair_limits.push_back(repair_time_limit);
        return false;  // no improvement, so the caller restores nothing
    }
};

}  // namespace

TEST_CASE("solve stops a Feasibility Jump batch at the deadline", "[search][deadline]") {
    Model m;
    build_unsatisfiable(m);

    SearchConfig config;
    // Sized so one batch cannot possibly finish inside the budget, which makes
    // "fewer than batch_iterations iterations were done" mean "the clock stopped
    // the batch from the inside". Remove the bound and the first batch runs to
    // completion, landing on exactly batch_iterations, and the run then exits at
    // the top of the loop having already overrun.
    //
    // This is a counting assertion, not a timing one, but it is not free of
    // machine speed either: what it needs is that the machine cannot do 40k GLS
    // iterations in 20ms. Measured here: 128 iterations, identical across
    // repetitions, so the margin is ~300x, and the trade-off is fixed —
    // margin = (time a regression takes to go red) / budget, so buying more
    // margin costs proportionally more time on the failing path. 40k iterations
    // is ~10s of red against a 20ms green.
    //
    // Note the flake direction is the safe one: a loaded or slower machine does
    // *fewer* iterations and passes more comfortably. Only a machine ~300x faster
    // than this one at GLS iterations could turn this red spuriously.
    config.batch_iterations = 40000;
    config.max_iterations = 0;  // no iteration budget: the clock is the only bound

    auto result = solve(m, /*time_limit=*/0.02, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::TimeLimit);
    REQUIRE(result.iterations > 0);  // not inert: the batch really did run
    REQUIRE(result.iterations < config.batch_iterations);
}

// Issue #113: bound 1 above says a batch is "handed the same absolute deadline",
// but the deadline used to be consulted only every 64 GLS iterations, and one
// GLS iteration is O(sampled vars x candidate values x constraints touched) —
// unbounded in the model size. So the bound held in iterations and not at all in
// time: on 400 Int vars with 20 000 rows of 8, budgets of 0.05s, 1s and 3s all
// took ~7s, each after exactly 64 iterations. The stride is now sized in time
// (see test_feasibility_jump.cpp, which tests the sizing rule itself); this is
// the end-to-end statement of it through solve().
//
// The test above cannot catch this one. Its model has 50 cheap variables, so its
// 64-iteration overrun is microseconds and its "fewer than batch_iterations"
// assertion passes with the stride fixed at 64. What is needed is a model whose
// single iteration is expensive, which is also the shape of the benchmark
// instances whose wall times epic #87 publishes.
TEST_CASE("solve stops a Feasibility Jump batch inside a fraction of the budget",
          "[search][deadline]") {
    // 60 Int vars in 1500 rows of 8, so each variable sits in ~200 constraints
    // and each of its 21 domain values must be scored against all of them. The
    // rows are satisfiable at the all-zeros start, so — exactly as in the hook
    // test below — the objective bound immediately tightens to something
    // unreachable and every later batch grinds against it, which is what makes
    // the iterations expensive.
    Model m;
    RNG rng(7);
    constexpr int kVars = 60;
    constexpr int kRows = 1500;
    std::vector<int32_t> vars;
    vars.reserve(kVars);
    for (int i = 0; i < kVars; ++i) {
        vars.push_back(m.int_var(0, 20));
    }
    for (int r = 0; r < kRows; ++r) {
        std::vector<int32_t> args;
        args.reserve(8);
        for (int k = 0; k < 8; ++k) {
            args.push_back(vars[static_cast<size_t>(rng.integers(0, kVars))]);
        }
        m.add_constraint(m.leq(m.sum(args), m.constant(3.0)));
    }
    m.minimize(m.sum(vars));
    m.close();

    SearchConfig config;
    config.batch_iterations = 1000;
    config.max_iterations = 0;  // no iteration budget: the clock is the only bound

    auto result = solve(m, /*time_limit=*/0.02, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::TimeLimit);
    REQUIRE(result.iterations > 0);  // not inert: the batch really did run
    // A counting assertion, not a timing one, but as with the test above it is
    // not free of machine speed: it needs the machine to be unable to do 64 GLS
    // iterations on this model inside 20ms. Measured here (Release): the run
    // does 1 iteration and takes 0.035s, while restoring the fixed stride of 64
    // makes it take 12.5s — a ~600x margin on the failing path, and the flake
    // direction is the safe one.
    REQUIRE(result.iterations < 64);
}

TEST_CASE("solve does not start the inner-solver hook past the deadline", "[search][deadline]") {
    // x's optimum is the value FeasibilityJump starts it at (the domain value
    // closest to zero), which pins the run to exactly two batches:
    //
    //   batch 1  nothing is violated yet — the objective bound is still +inf — so
    //            FJ returns immediately. The point is feasible, so it is recorded,
    //            which tightens the bound to obj - eps. The hook is then called
    //            with budget still in hand: this is the one legitimate call.
    //   batch 2  the objective row `obj <= -1e-3` is now violated and unreachable
    //            (x >= 0), so FJ cannot exit early and runs until the deadline.
    //            The *real* constraint is still satisfied, so the hook site is
    //            reached a second time — now with the budget spent.
    //
    // Remove the guard and batch 2 calls the hook too, so the count goes 1 -> 2.
    Model m;
    auto x = m.float_var(0, 10);
    m.add_constraint(m.leq(x, m.constant(10.0)));  // real, and satisfied everywhere
    m.minimize(m.sum({x}));
    m.close();

    SearchConfig config;
    // Batch 2 must not finish inside the budget, or the hook site is reached with
    // budget still in hand and the count goes to 2 for an innocent reason. This
    // model runs ~0.3us per iteration, so 20M iterations is ~300x what fits in the
    // budget. Unlike the batch test above, a large value is nearly free here: the
    // FJ deadline bound is what actually ends batch 2, so the number only matters
    // as a ceiling. It is not unbounded only because if *that* bound were also
    // removed this test would run the full 20M (~6s) instead of hanging.
    config.batch_iterations = 20000000;
    config.max_iterations = 0;

    CountingHook hook;
    auto result = solve(m, /*time_limit=*/0.02, /*seed=*/42, /*use_fj=*/true, &hook, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::TimeLimit);
    REQUIRE(hook.calls == 1);
}

TEST_CASE("solve bounds the LNS repair by its own remaining budget", "[search][deadline][lns]") {
    Model m;
    build_unsatisfiable(m);

    SearchConfig config;
    // Kick on the very first stagnant batch, and with lns_interval = 1 make every
    // kick an LNS kick. The model can never improve, so batch 1 stagnates and the
    // first repair happens one batch into the run. Deliberately not "let the
    // default 100-batch perturbation_period elapse": that would make "did a kick
    // happen at all?" depend on machine speed, and in the *unsafe* direction — a
    // slow machine would see no kick and the test would go red for no reason.
    config.batch_iterations = 1;
    config.max_iterations = 0;
    config.perturbation_period = 1;

    BudgetRecordingLNS lns;
    constexpr double kBudget = 0.05;
    auto result = solve(m, kBudget, /*seed=*/42, /*use_fj=*/true, nullptr, &lns,
                        /*lns_interval=*/1, nullptr, config);

    REQUIRE(result.termination == TerminationReason::TimeLimit);
    REQUIRE_FALSE(lns.repair_limits.empty());  // not inert: kicks really happened
    for (double limit : lns.repair_limits) {
        // The whole run only has kBudget to spend, so no single repair may ever be
        // handed more than that. Remove the clamp and every entry is LNS's own 2s
        // default — 10x this run's entire budget, and the shape of the bug that
        // made a 60s budget take 87s.
        REQUIRE(limit <= kBudget);
        // ...and never exactly 0. Downstream in fj_nl_initialize, 0 means "no wall
        // clock at all", so clamping an exhausted budget to 0 would hand the
        // repair an *unbounded* run — the precise opposite of the intent. That is
        // what the 1e-9 floor in search.cpp is for. Honest caveat: this pins the
        // property, not the floor's own branch. Reaching that branch needs the
        // clock to cross the deadline in the window between the loop's
        // past_deadline() check and the kick, which is a race no test can force.
        REQUIRE(limit > 0.0);
    }
}

TEST_CASE("solve hands the LNS repair no wall clock when it has none", "[search][deadline][lns]") {
    // The other half of the clamp: with no deadline, the repair must be handed 0
    // — "bounded by your iteration budget alone" — because that is what keeps an
    // iteration-budgeted run deterministic. Passing a positive limit here would
    // make every such run machine-speed-dependent again.
    Model m;
    build_unsatisfiable(m);

    SearchConfig config;
    config.batch_iterations = 1;
    config.max_iterations = 200;
    config.perturbation_period = 1;  // kick on the first stagnant batch, as above

    BudgetRecordingLNS lns;
    auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, &lns,
                        /*lns_interval=*/1, nullptr, config);

    REQUIRE(result.termination == TerminationReason::IterationLimit);
    REQUIRE_FALSE(lns.repair_limits.empty());
    for (double limit : lns.repair_limits) {
        REQUIRE(limit == 0.0);
    }
}

// Issue #117: `perturbation_period` counts BATCHES, and a batch is
// `batch_iterations` GLS iterations -- microseconds on a small model, seconds on
// an expensive one. Measured on MINLPLib elec25 at a 60s budget, a batch costs
// ~1.2s, so the run completes 52 batches against the default threshold of 100
// and the Float escape probe is never armed at all: the stagnation gate that
// makes the probe a last resort (#107) is dead code exactly on the models that
// need it. The cases below pin the two arming conditions, the guard that keeps
// the clock out of an iteration-budgeted run, the disarm on a new best, and that
// diversification survives a run with a wall clock armed.
//
// They observe `SearchResult::escape_probe_armed` -- the engine's own armed
// state at exit -- rather than timing the call or inferring arming from the
// trajectory, so nothing here asserts on elapsed wall clock.
//
// Scope limit worth stating: `build_unsatisfiable` is Int-only, and the probe
// changes nothing but Float candidate generation. So these pin the arming
// DECISION and its plumbing onto SearchResult -- not the probe's effect on a
// search, which test_feasibility_jump.cpp covers under [stationary]. Only the
// first case can fail on the unfixed engine; the rest are guards and
// characterization pins, and are labelled as such.

TEST_CASE("solve arms the escape probe on wall-clock stagnation", "[search][deadline][escape]") {
    Model m;
    build_unsatisfiable(m);  // never improves, so the probe is never disarmed

    SearchConfig config;
    // One GLS iteration per batch: the arming condition is only ever tested at a
    // batch boundary, so cheap batches give the run thousands of boundaries
    // inside the window between a quarter of the budget and the deadline (225ms
    // wide here). Be honest about the flake direction -- unlike the batch-count
    // tests further up this file, which get *safer* on a loaded box, the only way
    // to miss this window is one contiguous scheduler stall spanning all of it,
    // and that turns the test red. It fails loudly (armed == false), not inertly.
    config.batch_iterations = 1;
    config.max_iterations = 0;  // wall clock is the only budget
    // The batch counter can never arm the probe here, so an armed probe at exit
    // can only have come from the wall-clock condition. It also holds the
    // projected-batch gate open, since any projected count is below INT_MAX.
    // Not elec25's mechanism reproduced, to be precise: there the threshold is
    // reachable in principle and the budget cannot pay for it, here it is
    // unreachable outright. Same effect on the arming path, different cause.
    config.perturbation_period = std::numeric_limits<int>::max();

    auto result = solve(m, /*time_limit=*/0.3, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::TimeLimit);
    REQUIRE_FALSE(result.feasible);  // premise: nothing improves, so nothing disarms
    REQUIRE(result.escape_probe_armed);
}

TEST_CASE("solve does not arm the escape probe off the clock without a deadline",
          "[search][escape][determinism]") {
    // The determinism half of the fix. With no wall-clock budget, no clock read
    // may influence control flow, or an iteration-budgeted run stops being
    // bit-reproducible.
    //
    // This pins the PROPERTY, not either guard that delivers it, and the two are
    // redundant: `has_deadline` skips the block, and `now < deadline` is false
    // anyway because `budget_seconds` is 0 with no deadline, so `deadline ==
    // start`. Dropping either alone leaves this green -- confirmed by mutation.
    // Dropping BOTH arms the probe from batch one ("elapsed >= 0.25 * 0" holds
    // immediately) and turns it red.
    Model m;
    build_unsatisfiable(m);

    SearchConfig config;
    config.batch_iterations = 1;
    config.max_iterations = 500;
    config.perturbation_period = std::numeric_limits<int>::max();  // no batch arming either

    auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::IterationLimit);
    REQUIRE_FALSE(result.feasible);  // premise: nothing improves, so nothing disarms
    REQUIRE_FALSE(result.escape_probe_armed);
}

TEST_CASE("solve still arms the escape probe on the batch count", "[search][escape]") {
    // The pre-existing condition, unchanged: on a model where the batch
    // threshold is reachable the probe arms exactly as before, with no wall
    // clock involved at all. A characterization pin -- green before #117 and
    // after -- so it catches a regression in the old route, not this fix.
    Model m;
    build_unsatisfiable(m);

    SearchConfig config;
    config.batch_iterations = 1;
    config.max_iterations = 500;
    config.perturbation_period = 1;  // arm on the first stagnant batch

    auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::IterationLimit);
    REQUIRE_FALSE(result.feasible);  // premise: nothing improves, so nothing disarms
    REQUIRE(result.escape_probe_armed);
}

TEST_CASE("solve keeps diversifying when a wall clock is armed",
          "[search][deadline][escape][lns]") {
    // The starvation guard. The clock route arms WITHOUT an accompanying kick,
    // which the batch route never did, so the hazard it introduces is a probe
    // that drips tiny improvements, resets `stagnation` before it reaches
    // `perturbation_period`, and thereby stops diversify() -- and with it LNS,
    // which hangs off the same counter -- from ever firing. The projected-batch
    // gate in solve() confines the clock route to runs that cannot reach the
    // batch threshold anyway; this pins the other side of that gate: a run with a
    // wall clock armed and a reachable threshold still kicks.
    //
    // Honest about its strength: this model is Int-only and never improves, so it
    // cannot produce the drip itself. It is a floor -- it goes red if the clock
    // route ever suppresses kicks outright -- not a proof that the drip case is
    // handled. That case needs a Float model that improves under an armed probe.
    Model m;
    build_unsatisfiable(m);

    SearchConfig config;
    config.batch_iterations = 1;
    config.max_iterations = 0;       // wall clock is the only budget
    config.perturbation_period = 5;  // reachable within the budget, unlike the case above

    BudgetRecordingLNS lns;
    auto result = solve(m, /*time_limit=*/0.3, /*seed=*/42, /*use_fj=*/true, nullptr, &lns,
                        /*lns_interval=*/1, nullptr, config);

    REQUIRE(result.termination == TerminationReason::TimeLimit);
    REQUIRE_FALSE(lns.repair_limits.empty());  // diversification still fires
}

TEST_CASE("solve disarms the escape probe on a new best", "[search][escape]") {
    // `SearchResult::escape_probe_armed` is documented as "true only for a run
    // that ended stuck", which rests entirely on the disarm in the improvement
    // branch. Nothing pinned that. A pure-feasibility model does it without a
    // clock: it arms on the first stagnant batch, then breaks out the moment it
    // becomes feasible, so a run ending Feasible must report the probe disarmed.
    Model m;
    std::vector<int32_t> vars;
    vars.reserve(50);
    for (int i = 0; i < 50; ++i) {
        vars.push_back(m.int_var(0, 10));
    }
    std::vector<int32_t> args(vars.begin(), vars.end());
    args.push_back(m.constant(-250.0));
    // Satisfiable, unlike build_unsatisfiable: 50 vars capped at 10 reach 250
    // exactly. No objective, so the first feasible point ends the search.
    m.add_constraint(m.abs_expr(m.sum(args)));
    m.close();

    SearchConfig config;
    config.batch_iterations = 1;     // one GLS iteration per batch, so...
    config.perturbation_period = 1;  // ...batch 1 stagnates and arms the probe
    // Keep the kick minimal so it cannot undo FJ's progress: at p = 0 perturb()
    // still forces exactly one variable (#109), which counts as a kick without
    // randomising a tenth of the model on every stagnant batch.
    config.perturbation_probability = 0.0;
    config.max_iterations = 20000;  // ample: the model needs ~25 improving jumps

    auto result = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3,
                        nullptr, config);

    REQUIRE(result.termination == TerminationReason::Feasible);
    REQUIRE(result.feasible);
    // Not vacuous: more than one batch ran, so batch 1 was stagnant and (at
    // perturbation_period = 1) armed the probe before the improvement disarmed
    // it. Every batch is a Feasibility Jump here -- the model is Int-only, so
    // structural_probability is 0 and compound moves are off by default -- which
    // is what makes the iteration count a batch count.
    REQUIRE(result.iterations > 1);
    REQUIRE_FALSE(result.escape_probe_armed);
}

// Issue #105: the STRUCTURAL batch used to be the one sub-step with no deadline
// check, so a batch entered just before the deadline ran its whole sweep and
// overran the budget by however long that took. Unbounded in the model size --
// on the model below a 0.5s budget took 1.19-1.25s unbounded.
//
// structural_batch_probability = 1.0 makes every batch structural, so the very
// first batch is the one under test and the assertion does not depend on which
// batch kind the RNG happened to pick.
//
// [timing] MARKS THE EXCEPTION (issue #104): this is still the suite's only
// assertion on wall-clock DURATION. Everything else asserts on iteration counts
// or values -- the three throughput floors added under #125 share the [timing]
// tag but assert a floor on work completed inside a budget, which is a count.
// Run this one with `cbls_tests "structural batch respects the wall-clock
// deadline"`, or the whole class with `cbls_tests "[timing]"`. It is not known
// to flake -- measured at 0.116s against its 0.6s threshold with 48 busy
// processes on 12 cores -- but a wall-clock assertion is a standing liability,
// so keep it greppable and do not add more without a reason as concrete as
// this one.
TEST_CASE("structural batch respects the wall-clock deadline", "[search][structural][timing]") {
    constexpr int kLists = 1500;
    constexpr int kStops = 100;
    // The sweep rescans every constraint once per move (weighted_delta_from),
    // so these inert filler constraints are what make one sweep expensive — at
    // almost no model-build cost.
    constexpr int kFiller = 40000;

    Model m;
    std::vector<int32_t> lists;
    lists.reserve(kLists);
    for (int i = 0; i < kLists; ++i) {
        lists.push_back(m.list_var(kStops));
    }
    for (int i = 0; i < kLists; ++i) {
        // Order-dependent (so list moves can change it) and unsatisfiable (so the
        // search never ends early on feasibility and always spends the budget).
        auto len = m.pair_lambda_sum(lists[static_cast<size_t>(i)],
                                     [](int a, int b) { return 1.0 + 0.5 * std::abs(a - b); });
        m.add_constraint(m.leq(len, m.constant(0.5)));
    }
    auto z = m.float_var(0.0, 1.0);
    for (int i = 0; i < kFiller; ++i) {
        m.add_constraint(m.leq(z, m.constant(2.0)));
    }
    m.close();

    SearchConfig config;
    config.structural_batch_probability = 1.0;

    constexpr double kBudget = 0.10;
    auto before = std::chrono::steady_clock::now();
    solve(m, kBudget, /*seed=*/42, /*use_fj=*/true, nullptr, nullptr, 3, nullptr, config);
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - before).count();

    // Measured on this model: +0.003s overrun with the bound, +1.134s without
    // (one full unbounded sweep). The threshold sits between the two with room
    // on both sides, so it stays green on a loaded machine but still catches a
    // regression that removes the bound.
    REQUIRE(elapsed < kBudget + 0.5);
}

TEST_CASE("SolutionPool top_k", "[pool]") {
    SolutionPool pool(10);
    Model::State empty_state;
    pool.submit({empty_state, 10.0, true});
    pool.submit({empty_state, 5.0, true});
    pool.submit({empty_state, 3.0, true});
    pool.submit({empty_state, 20.0, false});

    auto top2 = pool.top_k(2);
    REQUIRE(top2.size() == 2);
    REQUIRE(top2[0].objective == 3.0);
    REQUIRE(top2[1].objective == 5.0);
}
