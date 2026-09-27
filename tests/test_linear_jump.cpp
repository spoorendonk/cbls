// The closed-form linear jump scorer (include/cbls/linear_jump.h): its scores
// against Model::weighted_violation_delta, its fallback conditions, and its
// upkeep across Model::extend.
#include "test_helpers.h"

#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/custom_invariant.h>
#include <cbls/linear_jump.h>
#include <cbls/model_extension.h>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cbls;

namespace {

// Tolerance scale for comparing a closed-form score with the DAG's: the summed
// weighted magnitude of every comparison child the move reads, before and after.
// `p + r D` and a re-summed row agree to a few ulp of THAT, not of the delta,
// which can be a small difference of large sides.
double score_magnitude(Model& m, const std::vector<double>& w, int32_t v, double j) {
    const auto& cids = m.constraint_ids();
    auto side_sum = [&]() {
        double s = 0.0;
        for (const int32_t c : m.constraints_of_var(v)) {
            const ExprNode& nd = m.nodes()[static_cast<size_t>(cids[static_cast<size_t>(c)])];
            for (const ChildRef& ch : m.children(nd)) {
                const double x =
                    ch.is_var ? m.var(ch.id).value : m.node_values()[static_cast<size_t>(ch.id)];
                if (std::isfinite(x)) {
                    s += w[static_cast<size_t>(c)] * std::abs(x);
                }
            }
        }
        return s;
    };
    const double before = side_sum();
    const double x0 = m.var(v).value;
    m.var_mut(v).value = j;
    full_evaluate(m);
    const double after = side_sum();
    m.var_mut(v).value = x0;
    full_evaluate(m);
    return before + after;
}

// One comparison row of a random kind over (lhs, rhs), or a range row over lhs.
void add_random_comparison(Model& m, RNG& rng, int32_t lhs, int32_t rhs) {
    switch (rng.integers(0, 6)) {
        case 0:
            m.add_constraint(m.leq(lhs, rhs));
            break;
        case 1:
            m.add_constraint(m.geq(lhs, rhs));
            break;
        case 2:
            m.add_constraint(m.lt(lhs, rhs));
            break;
        case 3:
            m.add_constraint(m.gt(lhs, rhs));
            break;
        case 4:
            m.add_constraint(m.eq_expr(lhs, rhs));
            break;
        default:  // a range row: two constraints over one lhs
            m.add_constraint(m.geq(lhs, m.constant(-2.0)));
            m.add_constraint(m.leq(lhs, m.constant(3.0)));
            break;
    }
}

struct RandomLinearModel {
    Model m;
    std::vector<int32_t> handles;
};

// A random model of linear rows over mixed scalars. Every shape the scorer has a
// branch for: Leq/Geq/Lt/Gt/Eq, a range (two rows sharing an lhs), negative,
// zero and fractional coefficients, Prod with the constant on either side, Neg,
// a variable directly as a comparison child, a variable twice in one row, and
// rows with a variable on BOTH sides. `integral` keeps every coefficient,
// constant and domain integral, so the arithmetic is exact and scores must match
// to the bit.
void build_random_linear(RandomLinearModel& r, uint64_t seed, bool integral) {
    RNG rng(seed);
    Model& m = r.m;
    const int nv = 9;
    for (int i = 0; i < nv; ++i) {
        switch (i % (integral ? 3 : 4)) {
            case 0:
                r.handles.push_back(m.int_var(-5, 5));
                break;
            case 1:
                r.handles.push_back(m.bool_var());
                break;
            case 2:
                r.handles.push_back(m.int_var(0, 300));  // > 256 wide: the grid path
                break;
            default:
                r.handles.push_back(m.float_var(-3.0, 3.0));
                break;
        }
    }
    const std::vector<double> int_coefs = {-3.0, -1.0, 0.0, 2.0, 5.0, 1.0};
    const std::vector<double> frac_coefs = {-2.5, 0.5, 1.75, 0.1, 0.0, -0.3};
    const std::vector<double>& coefs = integral ? int_coefs : frac_coefs;
    auto pick_var = [&]() {
        return r.handles[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(nv)))];
    };
    auto pick_coef = [&]() {
        return coefs[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(coefs.size())))];
    };
    auto term = [&](int32_t v) {
        switch (rng.integers(0, 5)) {
            case 0:
                return m.prod(m.constant(pick_coef()), v);
            case 1:
                return m.neg(v);
            case 2:
                return m.prod(v, m.constant(pick_coef()));
            case 3:  // affine / const; a power-of-two divisor keeps it exact
                return m.div_expr(v, m.constant(rng.integers(0, 2) == 0 ? 2.0 : -0.5));
            default:
                return v;
        }
    };
    auto side = [&]() {
        const int64_t k = rng.integers(1, 5);
        if (k == 1 && rng.integers(0, 2) == 0) {
            return pick_var();  // a variable directly as the comparison child
        }
        std::vector<int32_t> terms;
        const int32_t first = pick_var();
        terms.push_back(term(first));
        for (int64_t t = 1; t < k; ++t) {
            terms.push_back(term(pick_var()));
        }
        if (rng.integers(0, 3) == 0) {
            terms.push_back(term(first));  // the same variable twice
        }
        return m.sum(terms);
    };
    auto rhs_const = [&]() {
        return m.constant(integral ? static_cast<double>(rng.integers(-6, 7))
                                   : rng.uniform(-6.0, 6.0));
    };
    for (int row = 0; row < 14; ++row) {
        const int32_t lhs = side();
        const int32_t rhs = rng.integers(0, 4) == 0 ? side() : rhs_const();
        add_random_comparison(m, rng, lhs, rhs);
    }
    // Reverse the operand order on one Eq, so a literal on the LEFT is covered.
    m.add_constraint(m.eq_expr(m.constant(1.0), side()));
    std::vector<int32_t> obj_terms;
    obj_terms.reserve(r.handles.size());
    for (const int32_t h : r.handles) {
        obj_terms.push_back(m.prod(m.constant(pick_coef()), h));
    }
    m.minimize(m.sum(obj_terms));
    m.close();
    m.add_objective_soft_constraint();  // obj <= bound, as solve() folds it in
}

void randomise_assignment(Model& m, RNG& rng) {
    for (size_t v = 0; v < m.num_vars(); ++v) {
        Variable& var = m.var_mut(static_cast<int32_t>(v));
        if (var.type == VarType::Float) {
            var.value = rng.uniform(var.lb, var.ub);
        } else {
            var.value = static_cast<double>(
                rng.integers(static_cast<int64_t>(var.lb), static_cast<int64_t>(var.ub) + 1));
        }
    }
    full_evaluate(m);
}

std::vector<double> random_weights(size_t n, RNG& rng) {
    const std::vector<double> choices = {0.0, 1.0, 2.5, 0.3, 17.0};
    std::vector<double> w(n);
    for (double& x : w) {
        x = choices[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(choices.size())))];
    }
    return w;
}

std::vector<double> candidates_for(const Variable& var, RNG& rng) {
    std::vector<double> js;
    if (var.type == VarType::Float) {
        js = {var.lb, var.ub, 0.5 * (var.lb + var.ub)};
        for (int i = 0; i < 6; ++i) {
            js.push_back(rng.uniform(var.lb, var.ub));
        }
        return js;
    }
    const auto lb = static_cast<int64_t>(var.lb);
    const auto ub = static_cast<int64_t>(var.ub);
    const int64_t step = ub - lb > 20 ? 37 : 1;
    for (int64_t x = lb; x <= ub; x += step) {
        js.push_back(static_cast<double>(x));
    }
    js.push_back(var.ub);
    return js;
}

// Every (variable, candidate) of a model under one assignment and weight vector:
// the closed form must be taken, and must agree with the probe. Returns how many
// comparisons ran, so a caller can insist the check was not vacuous.
int check_scores(Model& m, LinearJumpScorer& sc, const std::vector<double>& w, RNG& rng,
                 bool exact) {
    int compared = 0;
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        const Variable var = m.var(v);
        for (const double j : candidates_for(var, rng)) {
            REQUIRE(sc.prepare(v, w));
            const double fast = sc.delta(j);
            const double ref = m.weighted_violation_delta(v, j, w);
            if (exact) {
                REQUIRE(fast == ref);
            } else {
                const double tol = 1e-12 * (1.0 + score_magnitude(m, w, v, j));
                INFO("var " << v << " j " << j << " fast " << fast << " ref " << ref);
                REQUIRE(std::abs(fast - ref) <= tol);
            }
            ++compared;
        }
    }
    return compared;
}

void mark_all_rows(const Model& m, LinearJumpScorer& sc) {
    sc.resize_rows(m.constraint_ids().size());
    for (size_t c = 0; c < m.constraint_ids().size(); ++c) {
        sc.set_row_eligible(static_cast<int32_t>(c), true);
    }
}

// A scalar input sum, as user code: affine in fact, never eligible by rule.
class InputSum : public CustomInvariant {
public:
    double evaluate(const InvariantInputs& in) override {
        double s = 0.0;
        for (int32_t i = 0; i < in.size(); ++i) {
            s += in.value(i);
        }
        return s;
    }
    [[nodiscard]] std::unique_ptr<CustomInvariant> clone() const override {
        return std::make_unique<InputSum>(*this);
    }
};

}  // namespace

TEST_CASE("linear jump scores equal the probe's exactly on integral rows", "[fj][linear_jump]") {
    for (uint64_t seed = 1; seed <= 12; ++seed) {
        RandomLinearModel r;
        build_random_linear(r, seed, /*integral=*/true);
        Model& m = r.m;
        // A FeasibilityJump's own classification must call every row eligible.
        ViolationManager vm(m);
        RNG fj_rng(seed);
        FeasibilityJump fj(m, vm, fj_rng);
        for (size_t c = 0; c < m.constraint_ids().size(); ++c) {
            REQUIRE(fj.linear_scorer().row_eligible(static_cast<int32_t>(c)));
        }
        LinearJumpScorer& sc = fj.linear_scorer();
        RNG rng(seed * 7919);
        for (int round = 0; round < 4; ++round) {
            randomise_assignment(m, rng);
            // Both the +inf sentinel bound the row opens with, and a finite one.
            m.set_objective_bound(round % 2 == 0 ? std::numeric_limits<double>::infinity()
                                                 : static_cast<double>(rng.integers(-20, 20)));
            const std::vector<double> w = random_weights(m.constraint_ids().size(), rng);
            REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);
            // Same candidates, same selection: the whole jump is unchanged.
            for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
                const JumpResult a = compute_var_jump(m, w, v);
                const JumpResult b = compute_var_jump(m, w, v, false, &sc);
                REQUIRE(a.jump_value == b.jump_value);
                REQUIRE(a.score == b.score);
            }
        }
    }
}

TEST_CASE("linear jump scores match the probe to rounding on fractional rows",
          "[fj][linear_jump]") {
    for (uint64_t seed = 101; seed <= 112; ++seed) {
        RandomLinearModel r;
        build_random_linear(r, seed, /*integral=*/false);
        Model& m = r.m;
        LinearJumpScorer sc(m);
        mark_all_rows(m, sc);
        RNG rng(seed * 104729);
        for (int round = 0; round < 3; ++round) {
            randomise_assignment(m, rng);
            if (round == 1) {
                m.set_objective_bound(rng.uniform(-10.0, 10.0));
            }
            const std::vector<double> w = random_weights(m.constraint_ids().size(), rng);
            REQUIRE(check_scores(m, sc, w, rng, /*exact=*/false) > 50);
        }
    }
}

TEST_CASE("cached row partials are bit-identical to compute_partial where claimed",
          "[fj][linear_jump]") {
    int claimed = 0;
    int declined = 0;
    for (uint64_t seed = 201; seed <= 210; ++seed) {
        RandomLinearModel r;
        build_random_linear(r, seed, /*integral=*/false);
        Model& m = r.m;
        LinearJumpScorer sc(m);
        mark_all_rows(m, sc);
        RNG rng(seed);
        randomise_assignment(m, rng);
        const auto& cids = m.constraint_ids();
        for (int32_t c = 0; c < static_cast<int32_t>(cids.size()); ++c) {
            for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
                double g = 0.0;
                if (sc.residual_partial(c, v, g)) {
                    ++claimed;
                    REQUIRE(g == compute_partial(m, cids[static_cast<size_t>(c)], v));
                } else {
                    ++declined;
                }
            }
        }
    }
    REQUIRE(claimed > 500);
    REQUIRE(declined > 0);  // Eq rows with a computed side on both ends decline
}

TEST_CASE("a row clamped to kInfPenalty cancels exactly in the closed form (#100)",
          "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.int_var(1, 4);
    const int32_t y = m.int_var(1, 4);
    // Row 0: 1e40 x + 1e40 y <= 0 -- clamped to 1e30 before and after any move
    // of x inside its domain. Row 1: x <= 0.5, the O(1) signal.
    m.add_constraint(
        m.leq(m.sum({m.prod(m.constant(1e40), x), m.prod(m.constant(1e40), y)}), m.constant(0.0)));
    m.add_constraint(m.leq(x, m.constant(0.5)));
    m.close();
    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 1.0;
    full_evaluate(m);

    LinearJumpScorer sc(m);
    mark_all_rows(m, sc);
    const std::vector<double> w = {1.0, 1.0};
    REQUIRE(sc.prepare(vid(x), w));
    // Differencing whole sums would give 1e30 + 1.5 - (1e30 + 0.5) == 0.
    REQUIRE(sc.delta(2.0) == 1.0);
    REQUIRE(sc.delta(4.0) == 3.0);
    REQUIRE(sc.delta(2.0) == m.weighted_violation_delta(vid(x), 2.0, w));
}

TEST_CASE("the linear scorer declines rows it cannot model", "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.float_var(-2.0, 2.0);
    const int32_t y = m.float_var(-2.0, 2.0);
    const int32_t z = m.float_var(-2.0, 2.0);
    const int32_t u = m.int_var(0, 3);
    const int32_t t = m.int_var(0, 3);
    const int32_t s = m.float_var(-std::numeric_limits<double>::infinity(),
                                  std::numeric_limits<double>::infinity());
    m.add_constraint(m.leq(m.sum({x, z}), m.constant(1.0)));                  // 0: linear
    m.add_constraint(m.leq(m.prod(x, y), m.constant(1.0)));                   // 1: bilinear
    m.add_constraint(m.neq(u, m.constant(2.0)));                              // 2: Neq, a step
    m.add_constraint(m.leq(m.custom({t}, std::make_unique<InputSum>(), "c"),  // 3: Custom
                           m.constant(1.0)));
    m.add_constraint(m.leq(m.sum({s, z}), m.constant(0.0)));  // 4: linear
    m.close();
    for (size_t v = 0; v < m.num_vars(); ++v) {
        m.var_mut(static_cast<int32_t>(v)).value = 0.0;
    }
    full_evaluate(m);

    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng);
    LinearJumpScorer& sc = fj.linear_scorer();
    REQUIRE(sc.row_eligible(0));
    REQUIRE_FALSE(sc.row_eligible(1));
    REQUIRE_FALSE(sc.row_eligible(2));
    REQUIRE_FALSE(sc.row_eligible(3));
    REQUIRE(sc.row_eligible(4));

    std::vector<double> w(m.constraint_ids().size(), 1.0);
    SECTION("a variable in a weighted nonlinear row falls back") {
        REQUIRE_FALSE(sc.prepare(vid(x), w));
        REQUIRE_FALSE(sc.prepare(vid(y), w));
        REQUIRE_FALSE(sc.prepare(vid(u), w));
        REQUIRE_FALSE(sc.prepare(vid(t), w));
        REQUIRE(sc.prepare(vid(z), w));
        REQUIRE(sc.fallback_prepares() == 4);
        REQUIRE(sc.fast_prepares() == 1);
    }
    SECTION("a masked (weight 0) nonlinear row does not force the fallback") {
        w[1] = 0.0;
        REQUIRE(sc.prepare(vid(x), w));
        REQUIRE(sc.delta(1.5) == m.weighted_violation_delta(vid(x), 1.5, w));
    }
    SECTION("a non-finite computed side falls back") {
        m.var_mut(vid(s)).value = std::numeric_limits<double>::infinity();
        full_evaluate(m);
        REQUIRE_FALSE(sc.prepare(vid(z), w));
        REQUIRE_FALSE(sc.prepare(vid(s), w));
    }
    SECTION("compute_var_jump takes the probe for a declined variable") {
        const int64_t before = sc.fallback_prepares();
        (void)compute_var_jump(m, w, vid(y), false, &sc);
        REQUIRE(sc.fallback_prepares() == before + 1);
    }
}

TEST_CASE("FeasibilityJump scores through the linear scorer", "[fj][linear_jump]") {
    RandomLinearModel r;
    build_random_linear(r, 7, /*integral=*/true);
    Model& m = r.m;
    ViolationManager vm(m);
    RNG rng(7);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    (void)fj.batch(200);
    // Every row is linear, so no jump took the probe (the objective row opens at
    // the +inf literal bound, which the closed form models).
    REQUIRE(fj.linear_scorer().fast_prepares() > 0);
    REQUIRE(fj.linear_scorer().fallback_prepares() == 0);
}

TEST_CASE("novelty jump scores both its W'-argmin and its W score in closed form",
          "[fj][linear_jump]") {
    // One jumpable variable, so exactly one is sampled; select_novelty_var then
    // prepares twice -- once under W' (the argmin) and once under W (the score).
    // A count, not "grew": either call site reverting to the probe loses one.
    Model m;
    const int32_t x = m.int_var(0, 5);
    m.add_constraint(m.leq(x, m.constant(0.0)));
    m.close();
    m.var_mut(vid(x)).value = 3.0;
    full_evaluate(m);
    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(false);
    const int64_t before = fj.linear_scorer().fast_prepares();
    REQUIRE(fj.apply_novelty_jump());
    REQUIRE(fj.linear_scorer().fast_prepares() == before + 2);
    REQUIRE(m.var(vid(x)).value == 0.0);
}

TEST_CASE("a Float's Newton step reads the cached row partial", "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.float_var(-10.0, 10.0);
    // Row 0 is satisfied and has a different slope, so reading the partial of the
    // wrong row would move the Newton candidate.
    m.add_constraint(m.leq(m.prod(m.constant(3.0), x), m.constant(100.0)));
    m.add_constraint(m.leq(m.prod(m.constant(2.0), x), m.constant(3.0)));  // 2x <= 3
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    LinearJumpScorer sc(m);
    mark_all_rows(m, sc);
    const std::vector<double> w = {1.0, 1.0};
    // Newton lands exactly on the root and is considered first; the midpoint and
    // lb tie with it at zero violation, so a wrong partial picks another value.
    const JumpResult r = compute_var_jump(m, w, vid(x), false, &sc);
    REQUIRE(r.jump_value == 1.5);
    REQUIRE(sc.cached_partials() == 1);
}

TEST_CASE("a zero slope through Div by a near-zero constant still falls back",
          "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.int_var(0, 2);
    const int32_t y = m.int_var(-1, 1);
    // Affine by rule (affine / const) with local derivative 0, but the value is
    // +inf for y >= 0 and -inf for y < 0: moving y flips the row.
    m.add_constraint(m.leq(m.sum({x, m.div_expr(y, m.constant(0.0))}), m.constant(1.0)));
    m.close();
    m.var_mut(vid(x)).value = 0.0;
    m.var_mut(vid(y)).value = 1.0;
    full_evaluate(m);
    LinearJumpScorer sc(m);
    mark_all_rows(m, sc);
    const std::vector<double> w = {1.0};
    REQUIRE(m.weighted_violation_delta(vid(y), -1.0, w) == -kInfPenalty);
    REQUIRE_FALSE(sc.prepare(vid(y), w));
    REQUIRE(compute_var_jump(m, w, vid(y), false, &sc).score == kInfPenalty);
}

TEST_CASE("an infinite candidate takes the probe, not the closed form", "[fj][linear_jump]") {
    // s unbounded, so -inf and +inf are candidates. Five violated rows s <= -10k:
    // Newton serves only the first four roots, so -inf is the one candidate that
    // satisfies all five. But row 5 reads 0 * s, which the DAG evaluates to NaN
    // at s = -inf -- a maximal violation -- while `p + r D` never sees it (zero
    // slope). Scored in closed form, -inf would win; the probe says it loses.
    const double inf = std::numeric_limits<double>::infinity();
    Model m;
    const int32_t s = m.float_var(-inf, inf);
    const int32_t z = m.int_var(0, 1);
    for (int k = 1; k <= 5; ++k) {
        m.add_constraint(m.leq(s, m.constant(-10.0 * k)));
    }
    m.add_constraint(m.leq(m.sum({m.prod(m.constant(0.0), s), z}), m.constant(1.0)));
    m.close();
    m.var_mut(vid(s)).value = 0.0;
    m.var_mut(vid(z)).value = 0.0;
    full_evaluate(m);
    LinearJumpScorer sc(m);
    mark_all_rows(m, sc);
    const std::vector<double> w(m.constraint_ids().size(), 1.0);
    REQUIRE(sc.prepare(vid(s), w));
    const JumpResult probe = compute_var_jump(m, w, vid(s));
    REQUIRE(probe.jump_value == -40.0);
    const JumpResult fast = compute_var_jump(m, w, vid(s), false, &sc);
    REQUIRE(fast.jump_value == probe.jump_value);
    REQUIRE(fast.score == probe.score);
}

TEST_CASE("a scorer out of step with the model refuses to prepare", "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.int_var(0, 2);
    m.add_constraint(m.leq(x, m.constant(1.0)));
    m.close();
    LinearJumpScorer sc(m);  // never sized
    const std::vector<double> w = {1.0};
    REQUIRE_THROWS_AS(sc.prepare(vid(x), w), std::logic_error);
}

TEST_CASE("the linear scorer follows Model::extend through on_extended", "[fj][linear_jump]") {
    Model m;
    const int32_t a = m.int_var(0, 4);
    const int32_t b = m.int_var(0, 4);
    const int32_t lhs = m.sum({m.prod(m.constant(1.0), a), m.prod(m.constant(2.0), b)});
    m.add_constraint(m.leq(lhs, m.constant(3.0)));  // row 0
    m.add_constraint(m.geq(lhs, m.constant(1.0)));  // row 1, sharing the lhs
    const int32_t other = m.sum({m.prod(m.constant(-1.0), b)});
    m.add_constraint(m.eq_expr(other, m.constant(-2.0)));  // row 2
    m.close();

    ViolationManager vm(m);
    RNG rng(11);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    (void)fj.batch(20);
    // Build every row's slopes BEFORE the extension, so a stale cache would exist
    // to be caught.
    LinearJumpScorer& sc = fj.linear_scorer();
    std::vector<double> w(m.constraint_ids().size(), 1.0);
    REQUIRE(sc.prepare(vid(a), w));
    REQUIRE(sc.prepare(vid(b), w));

    ModelExtension ext(m);
    const int32_t c = ext.int_var(0, 3);
    ext.set_initial(c, 1.0);
    // A new column into the shared lhs, and -- the case a slope cache can get
    // wrong -- an EXISTING variable appended again, which changes a's slope in
    // rows 0 and 1 from 1 to 4.
    ext.append_to_sum(lhs, ext.prod(ext.constant(1.0), c));
    ext.append_to_sum(lhs, ext.prod(ext.constant(3.0), a));
    // Row 2 grows a bilinear term: it must stop being eligible.
    ext.append_to_sum(other, ext.prod(c, a));
    // And a new linear row over old and new variables.
    ext.add_constraint(ext.leq(ext.sum({ext.prod(ext.constant(2.0), c), b}), ext.constant(2.0)));
    const ExtensionResult res = m.extend(ext);
    vm.on_extended(res);
    fj.on_extended(res);

    REQUIRE(sc.num_rows() == 4);
    REQUIRE(sc.row_eligible(0));
    REQUIRE(sc.row_eligible(1));
    REQUIRE_FALSE(sc.row_eligible(2));
    REQUIRE(sc.row_eligible(3));

    w.assign(m.constraint_ids().size(), 1.0);
    w[2] = 0.0;  // mask the now-bilinear row, so every variable takes the closed form
    for (const int32_t h : {a, b, c}) {
        const int32_t v = vid(h);
        for (int64_t k = 0; k <= static_cast<int64_t>(m.var(v).ub); ++k) {
            const auto j = static_cast<double>(k);
            REQUIRE(sc.prepare(v, w));
            REQUIRE(sc.delta(j) == m.weighted_violation_delta(v, j, w));
        }
    }
    // With the bilinear row weighted again, every variable reading it falls back.
    w[2] = 1.0;
    REQUIRE_FALSE(sc.prepare(vid(a), w));
    REQUIRE_FALSE(sc.prepare(vid(b), w));
    REQUIRE_FALSE(sc.prepare(vid(c), w));
}
