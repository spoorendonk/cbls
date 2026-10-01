#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cbls/cbls.h>
#include <chrono>
#include <cmath>
#include <vector>

using namespace cbls;

TEST_CASE("solve with nullptr hook is regression-safe", "[inner_solver]") {
    Model m;
    auto x = m.float_var(-10, 10);
    auto y = m.float_var(-10, 10);
    auto two = m.constant(2);
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    auto result = solve_deterministic(m, 954000, 42, nullptr);
    REQUIRE(result.feasible);
    REQUIRE(result.objective < 1.0);
}

TEST_CASE("FloatIntensifyHook improves Float vars", "[inner_solver]") {
    // min x^2 + y^2 s.t. x + y >= 1  (i.e. 1 - x - y <= 0)
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);

    // constraint: 1 - x - y <= 0
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));
    // objective: x^2 + y^2
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();

    // Start at a feasible but suboptimal point
    m.var_mut(vid(x)).value = 5.0;
    m.var_mut(vid(y)).value = 5.0;
    full_evaluate(m);

    ViolationManager vm(m);
    double before_aug = vm.augmented_objective();

    FloatIntensifyHook hook;
    hook.max_sweeps = 5;
    hook.solve(m, vm);

    double after_aug = vm.augmented_objective();
    // Hook should improve the augmented objective
    REQUIRE(after_aug < before_aug);
}

TEST_CASE("FloatIntensifyHook with infeasible start (no objective)", "[inner_solver]") {
    // Pure feasibility: x + y >= 3 (no objective)
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto three = m.constant(3.0);

    m.add_constraint(m.sum({three, m.prod(neg1, x), m.prod(neg1, y)}));
    m.close();

    // Start infeasible: x=0, y=0, constraint = 3 > 0
    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    full_evaluate(m);

    ViolationManager vm(m);
    REQUIRE_FALSE(vm.is_feasible());
    double before_aug = vm.augmented_objective();

    FloatIntensifyHook hook;
    hook.solve(m, vm);

    double after_aug = vm.augmented_objective();
    // Newton steps should reduce violation (no objective to counterbalance)
    REQUIRE(after_aug < before_aug);
}

TEST_CASE("Backtracking line search finds better step than fixed", "[inner_solver]") {
    // min (x - 3)^2, starting at x = 0
    Model m;
    auto x = m.float_var(-10, 10);
    auto three = m.constant(3.0);
    auto neg1 = m.constant(-1.0);
    auto two = m.constant(2.0);
    auto x_minus_3 = m.sum({x, m.prod(neg1, three)});
    m.minimize(m.pow_expr(x_minus_3, two));
    m.close();

    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    ViolationManager vm(m);

    FloatIntensifyHook hook;
    hook.max_sweeps = 1;
    hook.max_line_search_steps = 5;
    hook.solve(m, vm);

    // Should move x toward 3.0
    REQUIRE(m.var(vid(x)).value > 0.5);
}

TEST_CASE("Multi-var Newton moves multiple vars", "[inner_solver]") {
    // constraint: x^2 + y^2 - 25 <= 0  (x^2 + y^2 >= 25)
    // Start at x=1, y=1 (violation = 23). Single-var Newton on x alone
    // would need x=sqrt(24)~4.9 but multi-var distributes the correction.
    // objective: min (x-10)^2 + (y-10)^2 to push toward (10,10)
    // Multi-var Newton should move both vars toward the constraint boundary.
    Model m;
    auto x = m.float_var(0, 10);
    auto y = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto two = m.constant(2.0);
    auto ten = m.constant(10.0);
    auto twentyfive = m.constant(25.0);

    // constraint: 25 - x^2 - y^2 <= 0
    m.add_constraint(
        m.sum({twentyfive, m.prod(neg1, m.pow_expr(x, two)), m.prod(neg1, m.pow_expr(y, two))}));
    // objective: (x-10)^2 + (y-10)^2
    auto xm10 = m.sum({x, m.prod(neg1, ten)});
    auto ym10 = m.sum({y, m.prod(neg1, ten)});
    m.minimize(m.sum({m.pow_expr(xm10, two), m.pow_expr(ym10, two)}));
    m.close();

    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 1.0;
    full_evaluate(m);

    ViolationManager vm(m);
    double before_viol = vm.total_violation();
    REQUIRE(before_viol > 0.0);

    FloatIntensifyHook hook;
    hook.max_sweeps = 3;
    hook.max_multi_var_constraints = 5;
    hook.solve(m, vm);

    // Should reduce total violation
    REQUIRE(vm.total_violation() < before_viol);
    // Both vars should have moved from initial value of 1.0
    REQUIRE(m.var(vid(x)).value > 1.5);
    REQUIRE(m.var(vid(y)).value > 1.5);
}

TEST_CASE("solve with FloatIntensifyHook improves mixed problem", "[inner_solver]") {
    // Bool b, Float x in [0,10], constraint: b + x >= 3, min x
    Model m;
    auto b = m.bool_var();
    auto x = m.float_var(0, 10);
    auto neg1 = m.constant(-1.0);
    auto three = m.constant(3.0);

    m.add_constraint(m.sum({three, m.prod(neg1, b), m.prod(neg1, x)}));
    m.minimize(m.sum({x}));
    m.close();

    FloatIntensifyHook hook;
    auto result = solve_deterministic(m, 945000, 42, &hook);
    REQUIRE(result.feasible);
    // With hook, should find good solution (x=2 when b=1, or x=3 when b=0)
    REQUIRE(result.objective <= 3.5);
}

// ---------------------------------------------------------------------------
// #191: the hook runs inside the search's wall-clock budget.
// ---------------------------------------------------------------------------

namespace {

// min sum x_i^2 over n Floats in [1, 10], each with a row x_i - 20 <= 0 that
// always holds.
Model many_float_model(int n) {
    Model m;
    std::vector<int32_t> squares;
    squares.reserve(n);
    const auto two = m.constant(2.0);
    const auto minus20 = m.constant(-20.0);
    for (int i = 0; i < n; ++i) {
        const auto x = m.float_var(1.0, 10.0);
        m.add_constraint(m.sum({x, minus20}));
        squares.push_back(m.pow_expr(x, two));
    }
    m.minimize(m.sum(squares));
    m.close();
    return m;
}

// min sum -x_i over n Floats in [0, 1e9], each with a row x_i - 2e9 <= 0 that
// always holds. Feasible from the first batch, so the search calls the hook at
// once, and every FloatIntensifyHook sweep improves the objective -- the line
// search climbs each x_i by its full initial step -- so a pass runs every one
// of the sweeps it is allowed.
Model climbing_model(int n) {
    Model m;
    std::vector<int32_t> negated;
    negated.reserve(n);
    const auto minus1 = m.constant(-1.0);
    const auto minus_cap = m.constant(-2.0e9);
    for (int i = 0; i < n; ++i) {
        const auto x = m.float_var(0.0, 1.0e9);
        m.add_constraint(m.sum({x, minus_cap}));
        negated.push_back(m.prod(minus1, x));
    }
    m.minimize(m.sum(negated));
    m.close();
    return m;
}
constexpr int kClimbVars = 1000;
constexpr int kClimbSweeps = 200;

std::vector<double> float_values(const Model& m) {
    std::vector<double> values;
    values.reserve(m.num_vars());
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        values.push_back(m.var(v).value);
    }
    return values;
}

}  // namespace

// [timing], hand-registered as timing_inner_solver_hook_deadline in
// tests/CMakeLists.txt: a wall-clock-duration assertion, kept because its
// subject is the user-facing budget contract #191 broke. The deterministic
// discriminator for the polls themselves is the poll-sequence tests below.
TEST_CASE("solve returns within its budget when an intensification pass would outlast it",
          "[inner_solver][timing]") {
    // Red before #191 (6.7 s against this 0.5 s budget): the hook got no stop,
    // so the solve returned only when the pass had run all its sweeps.
    Model m = climbing_model(kClimbVars);
    FloatIntensifyHook hook;
    hook.max_sweeps = kClimbSweeps;

    constexpr double kBudget = 0.5;
    const auto started = std::chrono::steady_clock::now();
    const SearchResult result = solve(m, kBudget, 42, true, &hook);
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();

    REQUIRE(result.feasible);
    REQUIRE(result.counters.inner_solver_calls >= 1);
    // The tolerance covers setup, finish() and a loaded machine, not the hook:
    // this model polls every 65 Float variables, microseconds of descent apart.
    CHECK(result.time_seconds < kBudget + 0.5);
    CHECK(wall < kBudget + 0.5);
}

TEST_CASE("FloatIntensifyHook returns before any work on a raised stop", "[inner_solver]") {
    Model m = many_float_model(50);
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        m.var_mut(v).value = 5.0;
    }
    full_evaluate(m);
    ViolationManager vm(m);

    StopToken token;
    token.request();
    FloatIntensifyHook hook;
    hook.solve(m, vm, {}, token);
    REQUIRE(float_values(m) == std::vector<double>(m.num_vars(), 5.0));
}

namespace {

// A stop source that counts its polls and turns true from poll `raise_at` on
// (never, when 0). The poll sequence is the discriminator: each multi-variable
// Newton step is preceded by exactly one poll, and the coordinate pass polls
// before its first Float variable and then every vars_per_poll() of them.
struct CountingStop {
    int raise_at = 0;
    mutable int polls = 0;
    [[nodiscard]] bool requested() const {
        ++polls;
        return raise_at > 0 && polls >= raise_at;
    }
};

// n Floats in [0, 1] with a row 5 - x_i <= 0 each, which no assignment
// satisfies: every row stays violated, so a multi-variable pass always runs its
// full max_multi_var_constraints steps (each a no-op here -- one variable per
// row -- but polled all the same).
Model always_violated_model(int n) {
    Model m;
    const auto five = m.constant(5.0);
    const auto minus1 = m.constant(-1.0);
    for (int i = 0; i < n; ++i) {
        const auto x = m.float_var(0.0, 1.0);
        m.add_constraint(m.sum({five, m.prod(minus1, x)}));
    }
    m.close();
    return m;
}

int count_changed(const std::vector<double>& before, const std::vector<double>& after) {
    int changed = 0;
    for (size_t i = 0; i < before.size(); ++i) {
        changed += before[i] != after[i] ? 1 : 0;
    }
    return changed;
}

// Puts every variable at `value` and re-evaluates, so a test does not depend on
// how a fresh model initialises its Floats.
void start_all_at(Model& m, double value) {
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        m.var_mut(v).value = value;
    }
    full_evaluate(m);
}

}  // namespace

TEST_CASE("FloatIntensifyHook stops the coordinate pass at the poll that is raised",
          "[inner_solver]") {
    // 1024 rows -> one poll per 64 Float variables. Every descended variable
    // climbs, so the number that moved is the number descended.
    Model m = climbing_model(1024);
    const int stride = FloatIntensifyHook::vars_per_poll(m);
    REQUIRE(stride == 64);
    start_all_at(m, 0.0);
    const std::vector<double> before = float_values(m);
    ViolationManager vm(m);

    const CountingStop stop{3};  // polls 1 and 2 pass, poll 3 stops
    FloatIntensifyHook hook;
    hook.solve(m, vm, {}, stop);
    CHECK(stop.polls == 3);
    CHECK(count_changed(before, float_values(m)) == 2 * stride);
}

TEST_CASE("FloatIntensifyHook polls before every multi-variable Newton step", "[inner_solver]") {
    Model m = always_violated_model(1024);
    const int stride = FloatIntensifyHook::vars_per_poll(m);
    REQUIRE(stride == 64);
    const int coordinate_polls = 1024 / stride;
    start_all_at(m, 0.0);
    FloatIntensifyHook hook;
    hook.max_sweeps = 1;
    REQUIRE(hook.max_multi_var_constraints == 5);

    SECTION("never raised: one poll per stride, then one per multi-variable step") {
        ViolationManager vm(m);
        const CountingStop stop{0};
        hook.solve(m, vm, {}, stop);
        CHECK(stop.polls == coordinate_polls + 5);
    }
    SECTION("raised at the first multi-variable poll: no multi-variable step runs") {
        // Poll coordinate_polls + 1 is the first one the multi-variable pass
        // makes; with that poll gone the pass runs unpolled, and max_sweeps = 1
        // ends the call one poll short.
        ViolationManager vm(m);
        const CountingStop stop{coordinate_polls + 1};
        hook.solve(m, vm, {}, stop);
        CHECK(stop.polls == coordinate_polls + 1);
    }
}

namespace {

// Spins until the search's stop is raised, with a 1 s cap -- many times the
// 50 ms budget below -- so a search that never raises it fails the test fast
// rather than hanging it.
struct WaitForStopHook : InnerSolverHook {
    int calls = 0;
    int stopped_calls = 0;
    void solve(Model& /*model*/, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/, StopRef stop) override {
        ++calls;
        const auto cap = std::chrono::steady_clock::now() + std::chrono::seconds(1);
        while (std::chrono::steady_clock::now() < cap) {
            if (stop.requested()) {
                ++stopped_calls;
                return;
            }
        }
    }
};

// Raises the host's token from inside the hook, then reports whether the stop
// the search handed it saw that.
struct CancelInsideHook : InnerSolverHook {
    explicit CancelInsideHook(StopToken& t) : token(t) {}
    StopToken& token;
    int calls = 0;
    int raised_on_entry = 0;
    int stopped_calls = 0;
    void solve(Model& /*model*/, ViolationManager& /*vm*/,
               const std::vector<int32_t>& /*last_changed_vars*/, StopRef stop) override {
        ++calls;
        if (stop.requested()) {
            ++raised_on_entry;
            return;
        }
        token.request();
        if (stop.requested()) {
            ++stopped_calls;
        }
    }
};

}  // namespace

TEST_CASE("search hands a custom hook a stop raised at its deadline", "[inner_solver]") {
    Model m = many_float_model(5);
    WaitForStopHook hook;
    // No duration assertion: what is pinned is that the stop the hook holds
    // turns true once the deadline passes, not how promptly the solve returns.
    const SearchResult result = solve(m, 0.05, 42, true, &hook);
    REQUIRE(hook.calls >= 1);
    REQUIRE(hook.stopped_calls == hook.calls);
    REQUIRE(result.termination == TerminationReason::TimeLimit);
}

TEST_CASE("search hands a custom hook a stop raised by a host cancel", "[inner_solver]") {
    // No wall clock: an iteration-budgeted run, whose stop reads no clock. The
    // hook raises the host's token itself, so the only route by which it can
    // see stop.requested() is the search's own cancel check -- the route a
    // host cancelling from another thread mid-hook relies on.
    Model m = many_float_model(5);
    StopToken token;
    CancelInsideHook hook(token);
    SearchConfig cfg;
    cfg.max_iterations = 100000;
    cfg.stop = token;
    const SearchResult result = solve(m, 0.0, 42, true, &hook, nullptr, 3, nullptr, cfg);
    REQUIRE(hook.calls == 1);
    REQUIRE(hook.raised_on_entry == 0);
    REQUIRE(hook.stopped_calls == 1);
    REQUIRE(result.termination == TerminationReason::Cancelled);
}
