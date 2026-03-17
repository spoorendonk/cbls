#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>

using namespace cbls;

TEST_CASE("SolveCallback fires during solve", "[callback]") {
    Model m;
    auto x = m.float_var(0, 10, "x");
    auto y = m.float_var(0, 10, "y");
    auto obj = m.sum({x, y});
    m.minimize(obj);
    m.close();

    struct CountCallback : SolveCallback {
        int count = 0;
        bool saw_new_best = false;
        void on_progress(const SolveProgress& p) override {
            count++;
            if (p.new_best) saw_new_best = true;
        }
    };

    CountCallback cb;
    auto result = solve(m, 0.5, 42, true, nullptr, nullptr, 3, &cb);

    REQUIRE(cb.count > 0);
    REQUIRE(cb.saw_new_best);
    REQUIRE(result.iterations > 0);
}

TEST_CASE("SolveCallback receives valid progress data", "[callback]") {
    Model m;
    auto x = m.float_var(0, 10, "x");
    auto c5 = m.constant(5.0);
    m.add_constraint(m.geq(x, c5));
    m.minimize(m.sum({x}));
    m.close();

    struct ValidateCallback : SolveCallback {
        bool all_valid = true;
        void on_progress(const SolveProgress& p) override {
            if (p.time_seconds < 0) all_valid = false;
            if (p.iteration < 0) all_valid = false;
            if (p.temperature < 0) all_valid = false;
            if (p.total_violation < 0) all_valid = false;
        }
    };

    ValidateCallback cb;
    solve(m, 0.5, 42, true, nullptr, nullptr, 3, &cb);
    REQUIRE(cb.all_valid);
}
