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
#include <cbls/cbls.h>
#include <cmath>
#include <limits>
#include <sstream>

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

    // Inside (0, kLogDomainFloor) is cut off, as SCIP cuts it off.
    m.var_mut(vid(x)).value = kLogDomainFloor / 2.0;
    full_evaluate(m);
    REQUIRE(m.node_value(m.constraint_ids()[1]) > 0.0);
    m.var_mut(vid(x)).value = 1.0;
    full_evaluate(m);
    REQUIRE(m.node_value(m.constraint_ids()[1]) <= 0.0);
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
    // -log 2 at x = y = 1/2. The start (1.5, -0.5) satisfies the row, and the
    // objective there is NaN. Diversification is off, as in
    // test_nonfinite_guard.cpp, so a random kick cannot rescue the search: the
    // only route back into the domain is a row that says which way it is.
    Model m;
    const int32_t x = m.float_var(-kInf, kInf, "x");
    const int32_t y = m.float_var(-kInf, kInf, "y");
    m.add_constraint(m.eq_expr(m.sum({x, y}), m.constant(1.0)));
    m.minimize(m.sum({m.prod(x, m.log_expr(x)), m.prod(y, m.log_expr(y))}));
    m.close();
    m.var_mut(vid(x)).value = 1.5;
    m.var_mut(vid(y)).value = -0.5;
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
