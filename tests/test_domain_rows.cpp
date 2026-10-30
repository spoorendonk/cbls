// #248: the implicit domain rows `close()` appends for `Log`, `Sqrt` and a
// fractional-exponent `Pow`.
//
// MINLPLib's ex8_5_6 has only free variables; the signs of its mole fractions
// are fixed by nothing but the domain of the `log` in its objective. Its search
// ended on a constraint-feasible point outside that domain, where the objective
// is NaN and the objective row reads a flat 1e30: no direction back in, and no
// finite answer in 60 s. The search test below is that situation in two
// variables.

#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cbls/cbls.h>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

using namespace cbls;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

}  // namespace

TEST_CASE("a log of a free variable gets a hidden row at SCIP's floor", "[domain-rows]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    m.add_constraint(m.leq(x, m.constant(5.0)));
    m.minimize(m.log_expr(x));
    m.close();

    REQUIRE(m.num_user_constraints() == 1);
    REQUIRE(m.num_domain_rows() == 1);
    REQUIRE(m.constraint_ids().size() == 2);

    // The row is graded: further outside, more violated.
    m.var_mut(vid(x)).value = -1.0;
    full_evaluate(m);
    const double at_minus_one = m.node_value(m.constraint_ids()[1]);
    m.var_mut(vid(x)).value = -3.0;
    full_evaluate(m);
    const double at_minus_three = m.node_value(m.constraint_ids()[1]);
    REQUIRE(at_minus_one > 0.0);
    REQUIRE(at_minus_three > at_minus_one);
    REQUIRE_FALSE(verify_model(m).ok);

    // Violated beyond the feasibility tolerance, which is what the engine
    // reads, at 0 and inside (0, kLogDomainFloor), as SCIP cuts it off...
    const int32_t row = m.constraint_ids()[1];
    for (const double v : {0.0, kLogDomainFloor / 2.0}) {
        m.var_mut(vid(x)).value = v;
        full_evaluate(m);
        CAPTURE(v);
        REQUIRE(m.node_value(row) > kDefaultFeasibilityTolerance);
    }
    // ...and satisfied just above the floor.
    m.var_mut(vid(x)).value = 2.0 * kLogDomainFloor;
    full_evaluate(m);
    REQUIRE(m.node_value(row) <= kDefaultFeasibilityTolerance);
}

TEST_CASE("an argument exactly at its floor adds no violation", "[domain-rows]") {
    // #258: with the tolerance folded into the RHS, `sqrt(x^2 + y^2)` at the
    // origin read 1e-6 on its row, and the weighted sums charged it.
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    const int32_t y = m.float_var(-kInf, kInf, "y");
    const int32_t z = m.float_var(-kInf, kInf, "z");
    const int32_t two = m.constant(2.0);
    m.add_constraint(
        m.leq(m.sqrt_expr(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)})), m.constant(1.0)));
    m.add_constraint(m.leq(m.log_expr(z), m.constant(1.0)));
    m.close();
    REQUIRE(m.num_domain_rows() == 2);
    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 0.0;
    m.var_mut(vid(z)).value = kLogDomainFloor;
    full_evaluate(m);

    const ViolationManager vm(m);
    for (size_t c = 0; c < m.constraint_ids().size(); ++c) {
        CAPTURE(c);
        REQUIRE(vm.constraint_violation(static_cast<int>(c)) == 0.0);
    }
    REQUIRE(vm.total_violation() == 0.0);
}

TEST_CASE("the solve's feasibility tolerance decides a domain row, not the default",
          "[domain-rows]") {
    // #258: the RHS carried the default 1e-6 from close(). At 1e-9 that made
    // sqrt's optimum x = 0 infeasible; at 1e-4 it let a log's argument down to
    // about -1e-4 pass.
    const double tol = GENERATE(1e-9, 1e-4);
    CAPTURE(tol);
    SearchConfig cfg;
    cfg.feasibility_tolerance = tol;
    cfg.skip_init = true;
    cfg.max_iterations = 200;
    cfg.perturbation_period = 1 << 30;

    SECTION("minimize sqrt(x) over a free x is feasible at its optimum x = 0") {
        Model m;
        const int32_t x = m.float_var(-kInf, kInf, "x");
        m.minimize(m.sqrt_expr(x));
        m.close();
        m.var_mut(vid(x)).value = 0.0;
        const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/1, /*use_fj=*/true, nullptr,
                                     nullptr, 3, nullptr, cfg);
        REQUIRE(r.feasible);
        REQUIRE(r.objective == 0.0);
    }

    SECTION("a log's argument at 0 or below reads infeasible") {
        Model m;
        const int32_t x = m.float_var(-kInf, kInf, "x");
        m.add_constraint(m.leq(m.log_expr(x), m.constant(1.0)));
        m.close();
        m.var_mut(vid(x)).value = 1.0;
        (void)solve(m, /*time_limit=*/0.0, /*seed=*/1, /*use_fj=*/true, nullptr, nullptr, 3,
                    nullptr, cfg);
        for (const double v : {0.0, -tol / 2.0, -tol}) {
            CAPTURE(v);
            m.var_mut(vid(x)).value = v;
            full_evaluate(m);
            REQUIRE_FALSE(ViolationManager(m).is_feasible(tol));
            REQUIRE_FALSE(verify_model(m, tol).ok);
        }
        m.var_mut(vid(x)).value = kLogDomainFloor;
        full_evaluate(m);
        REQUIRE(ViolationManager(m).is_feasible(tol));
    }
}

TEST_CASE("verify_model holds a domain row to 0 whatever its tolerance", "[domain-rows]") {
    // The model's margin is the default 1e-6; a check at a looser tolerance
    // still sees an argument below the floor.
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    m.minimize(m.sqrt_expr(x));
    m.close();
    m.var_mut(vid(x)).value = -1e-12;
    full_evaluate(m);
    REQUIRE_FALSE(verify_model(m, 1e-3).ok);
    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    REQUIRE(verify_model(m, 1e-3).ok);
}

TEST_CASE("a sqrt's row reads every negative argument as violated past the tolerance",
          "[domain-rows]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    m.minimize(m.sqrt_expr(x));
    m.close();
    REQUIRE(m.num_domain_rows() == 1);
    const int32_t row = m.constraint_ids()[0];

    m.var_mut(vid(x)).value = -kDefaultFeasibilityTolerance / 2.0;
    full_evaluate(m);
    REQUIRE(m.node_value(row) > kDefaultFeasibilityTolerance);
    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    REQUIRE(m.node_value(row) <= kDefaultFeasibilityTolerance);
}

TEST_CASE("a negative fractional exponent's pole at 0 is cut off as a log's", "[domain-rows]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    m.minimize(m.pow_expr(x, m.constant(-0.5)));
    m.close();
    REQUIRE(m.num_domain_rows() == 1);

    m.var_mut(vid(x)).value = 0.0;
    full_evaluate(m);
    REQUIRE(m.node_value(m.constraint_ids()[0]) > kDefaultFeasibilityTolerance);
    m.var_mut(vid(x)).value = 2.0 * kLogDomainFloor;
    full_evaluate(m);
    REQUIRE(m.node_value(m.constraint_ids()[0]) <= kDefaultFeasibilityTolerance);
}

TEST_CASE("the objective row comes after the domain rows", "[domain-rows]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    m.minimize(m.sqrt_expr(x));
    m.close();
    m.add_objective_soft_constraint();

    REQUIRE(m.num_user_constraints() == 0);
    REQUIRE(m.num_domain_rows() == 1);
    REQUIRE(m.objective_constraint_idx() == 1);
}

TEST_CASE("an unclosed model handed to the objective row gets its domain rows first",
          "[domain-rows]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    m.minimize(m.log_expr(x));
    m.add_objective_soft_constraint();

    REQUIRE(m.num_domain_rows() == 1);
    REQUIRE(m.objective_constraint_idx() == 1);
}

TEST_CASE("only domain-restricted ops a row or the objective reads get a domain row",
          "[domain-rows]") {
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    const int32_t y = m.float_var(-kInf, kInf, "y");
    const int32_t z = m.float_var(-kInf, kInf, "z");
    const int32_t pos = m.float_var(1.0, 2.0, "pos");
    const int32_t w = m.float_var(-kInf, kInf, "w");
    std::vector<int32_t> terms;
    // Integer exponent: defined everywhere.
    terms.push_back(m.pow_expr(x, m.constant(3.0)));
    // A lower bound that already meets the floor.
    terms.push_back(m.log_expr(pos));
    // A Const argument: no move changes it.
    terms.push_back(m.log_expr(m.constant(2.0)));
    // Under an If branch only: `if(y > 0, log(y), 0)` is defined at y <= 0.
    terms.push_back(m.if_then_else(m.gt(y, m.constant(0.0)), m.log_expr(y), m.constant(0.0)));
    // Read by nothing.
    m.log_expr(z);
    // A fractional exponent: one row. Two logs of the same argument: one row.
    terms.push_back(m.pow_expr(w, m.constant(0.5)));
    terms.push_back(m.log_expr(w));
    terms.push_back(m.log_expr(w));
    m.add_constraint(m.leq(m.sum(terms), m.constant(10.0)));
    m.close();

    REQUIRE(m.num_user_constraints() == 1);
    REQUIRE(m.num_domain_rows() == 2);  // w >= 0 and w >= kLogDomainFloor
}

TEST_CASE("save_model leaves the domain rows to the reload's close", "[domain-rows][io]") {
    Model m;
    // Finite bounds: the JSON format has no infinity to write.
    const int32_t x = m.float_var(-5.0, 5.0, "x");
    m.add_constraint(m.leq(m.log_expr(x), m.constant(1.0)));
    m.close();
    REQUIRE(m.num_domain_rows() == 1);

    std::stringstream ss;
    save_model(m, ss);
    const Model back = load_model(ss);

    REQUIRE(back.num_user_constraints() == 1);
    REQUIRE(back.num_domain_rows() == 1);
    REQUIRE(back.constraint_ids().size() == m.constraint_ids().size());
    REQUIRE(back.num_nodes() == m.num_nodes());
}

TEST_CASE("a feasible start outside a log's domain reaches a finite objective",
          "[domain-rows][nonfinite-objective]") {
    // x + y = 1 over free variables, minimise x log x + y log y: optimum
    // -log 2 at x = y = 1/2. Both starts satisfy the row with an objective that
    // is not a number: (1.5, -0.5) well outside the domain, and (1, 0) at its
    // edge, where ex8_5_6's seed 2 ended (0 log 0 is NaN) and where a floor
    // inside the feasibility tolerance would read as met. Diversification is
    // off, as in test_nonfinite_guard.cpp, so a random kick cannot rescue the
    // search: the only route back into the domain is a row that says which way
    // it is.
    const auto [x0, y0] =
        GENERATE(std::pair<double, double>{1.5, -0.5}, std::pair<double, double>{1.0, 0.0});
    CAPTURE(x0, y0);
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    const int32_t y = m.float_var(-kInf, kInf, "y");
    m.add_constraint(m.eq_expr(m.sum({x, y}), m.constant(1.0)));
    m.minimize(m.sum({m.prod(x, m.log_expr(x)), m.prod(y, m.log_expr(y))}));
    m.close();
    m.var_mut(vid(x)).value = x0;
    m.var_mut(vid(y)).value = y0;
    full_evaluate(m);
    REQUIRE(std::isnan(m.node_value(m.objective_id())));

    SearchConfig cfg;
    cfg.skip_init = true;  // keep the start outside the domain
    cfg.max_iterations = 20000;
    cfg.perturbation_period = 1 << 30;

    const SearchResult r = solve(m, /*time_limit=*/0.0, /*seed=*/42, /*use_fj=*/true, nullptr,
                                 nullptr, 3, nullptr, cfg);

    CAPTURE(r.objective);
    REQUIRE(r.feasible);
    REQUIRE(std::isfinite(r.objective));
    REQUIRE(r.objective >= -std::log(2.0) - 1e-6);
}
