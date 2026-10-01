// The closed-form linear jump scorer (include/cbls/linear_jump.h): its scores
// against Model::weighted_violation_delta, its fallback conditions, and its
// slope-table upkeep.
#include "test_helpers.h"

#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cbls/cbls.h>
#include <cbls/custom_invariant.h>
#include <cbls/linear_jump.h>
#include <cmath>
#include <functional>
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

// Bare bodies, read as `body <= 0` (#190): every affine node kind as the row's
// own node, a constant term, and a variable on both signs. None unless `bare`,
// so a model built without them draws the same random stream as before.
void add_bare_rows(Model& m, bool bare, const std::function<int32_t()>& side,
                   const std::function<double()>& pick_coef,
                   const std::function<int32_t()>& rhs_const, bool integral) {
    if (!bare) {
        return;
    }
    for (int row = 0; row < 3; ++row) {
        m.add_constraint(m.sum({side(), m.neg(rhs_const())}));
        m.add_constraint(m.neg(side()));
        m.add_constraint(m.prod(m.constant(pick_coef()), side()));
        m.add_constraint(m.div_expr(side(), m.constant(integral ? 2.0 : 0.3)));
        m.add_constraint(m.sum({side(), m.neg(side()), rhs_const()}));
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
// to the bit. Closed, but without the objective row -- build_random_linear adds
// it; a test that adds it later calls this.
void build_random_linear_no_objective_row(RandomLinearModel& r, uint64_t seed, bool integral,
                                          bool bare = false) {
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
    add_bare_rows(m, bare, side, pick_coef, rhs_const, integral);
    std::vector<int32_t> obj_terms;
    obj_terms.reserve(r.handles.size());
    for (const int32_t h : r.handles) {
        obj_terms.push_back(m.prod(m.constant(pick_coef()), h));
    }
    m.minimize(m.sum(obj_terms));
    m.close();
}

void build_random_linear(RandomLinearModel& r, uint64_t seed, bool integral, bool bare = false) {
    build_random_linear_no_objective_row(r, seed, integral, bare);
    r.m.add_objective_soft_constraint();  // obj <= bound, as solve() folds it in
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

// Row ci's partial in v through the positional API; false when ci is not in G_v.
bool partial_of_row(const Model& m, LinearJumpScorer& sc, int32_t ci, int32_t v, double& out) {
    const ConstSpan<int32_t> gv = m.constraints_of_var(v);
    const auto* it = std::lower_bound(gv.begin(), gv.end(), ci);
    return it != gv.end() && *it == ci &&
           sc.residual_partial_at(v, static_cast<size_t>(it - gv.begin()), out);
}

// Every (variable, G_v position) partial the scorer claims equals compute_partial.
// Returns how many were claimed.
int check_partials(Model& m, LinearJumpScorer& sc) {
    int claimed = 0;
    const auto& cids = m.constraint_ids();
    for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
        const ConstSpan<int32_t> gv = m.constraints_of_var(v);
        for (size_t k = 0; k < gv.size(); ++k) {
            double g = 0.0;
            if (!sc.residual_partial_at(v, k, g)) {
                continue;
            }
            ++claimed;
            REQUIRE(g == compute_partial(m, cids[static_cast<size_t>(gv[k])], v));
        }
    }
    return claimed;
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
        const auto& cids = m.constraint_ids();
        // Rows are built on the first pass; the later passes run at new
        // assignments, where an Eq row's sign has typically flipped -- so a
        // partial that cached the sign along with the slope goes red.
        for (int pass = 0; pass < 3; ++pass) {
            randomise_assignment(m, rng);
            for (int32_t c = 0; c < static_cast<int32_t>(cids.size()); ++c) {
                for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
                    double g = 0.0;
                    if (partial_of_row(m, sc, c, v, g)) {
                        ++claimed;
                        REQUIRE(g == compute_partial(m, cids[static_cast<size_t>(c)], v));
                    } else {
                        ++declined;
                    }
                }
            }
        }
    }
    REQUIRE(claimed > 500);
    REQUIRE(declined > 0);  // Eq rows with a computed side on both ends decline
}

TEST_CASE("a built row is never reclassified", "[fj][linear_jump]") {
    // A closed model's rows do not change, so a row's slopes are built once and
    // kept. Reclassifying a built row would leave its slopes in the table, so
    // it is refused instead; an unbuilt
    // row can still be (re)classified.
    RandomLinearModel r;
    build_random_linear(r, 31, /*integral=*/true);
    Model& m = r.m;
    LinearJumpScorer sc(m);
    mark_all_rows(m, sc);
    RNG rng(31);
    randomise_assignment(m, rng);
    const std::vector<double> w(m.constraint_ids().size(), 1.0);
    REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);
    const size_t cached = sc.cached_slopes();  // every row built once
    REQUIRE(cached > 0);
    REQUIRE_THROWS_AS(sc.set_row_eligible(0, false), std::logic_error);
    REQUIRE(sc.row_eligible(0));
    REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);
    REQUIRE(sc.cached_slopes() == cached);  // scoring again builds nothing new

    LinearJumpScorer fresh(m);
    fresh.resize_rows(m.constraint_ids().size());
    fresh.set_row_eligible(0, true);
    REQUIRE_NOTHROW(fresh.set_row_eligible(0, false));  // not built yet
    REQUIRE_FALSE(fresh.row_eligible(0));

    // The row table only grows: a shrink would orphan built rows the same way.
    const size_t nc = m.constraint_ids().size();
    REQUIRE_THROWS_AS(sc.resize_rows(nc - 1), std::logic_error);
    REQUIRE(sc.num_rows() == nc);
    REQUIRE_NOTHROW(sc.resize_rows(nc));
    REQUIRE_NOTHROW(fresh.resize_rows(nc + 1));
    REQUIRE(fresh.num_rows() == nc + 1);
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

TEST_CASE("slopes are right whichever variable or path first builds a row (#176)",
          "[fj][linear_jump]") {
    // The slope table is written per row, into every variable's G_v slot, by
    // whichever read builds the row first: another variable's prepare, a Newton
    // step's residual_partial, or this variable's own prepare. Reads in a random
    // order across all three, then the exact checks.
    for (uint64_t seed = 401; seed <= 410; ++seed) {
        RandomLinearModel r;
        build_random_linear(r, seed, /*integral=*/true);
        Model& m = r.m;
        LinearJumpScorer sc(m);
        mark_all_rows(m, sc);
        RNG rng(seed);
        randomise_assignment(m, rng);
        std::vector<std::pair<int32_t, size_t>> reads;
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            for (size_t k = 0; k < m.constraints_of_var(v).size(); ++k) {
                reads.emplace_back(v, k);
            }
        }
        for (size_t i = reads.size(); i > 1; --i) {
            std::swap(reads[i - 1],
                      reads[static_cast<size_t>(rng.integers(0, static_cast<int64_t>(i)))]);
        }
        const size_t half = reads.size() / 2;  // the rest are built by check_scores
        for (size_t i = 0; i < half; ++i) {
            const auto [v, k] = reads[i];
            if (rng.integers(0, 2) == 0) {
                double g = 0.0;
                (void)sc.residual_partial_at(v, k, g);
            } else {
                (void)sc.prepare(v, random_weights(m.constraint_ids().size(), rng));
            }
        }
        const std::vector<double> w = random_weights(m.constraint_ids().size(), rng);
        REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);
        REQUIRE(check_partials(m, sc) > 20);
    }
}

TEST_CASE("a row demoted by its build counts no slopes and stays ineligible", "[fj][linear_jump]") {
    const double inf = std::numeric_limits<double>::infinity();
    Model m;
    const int32_t x = m.int_var(0, 3);
    const int32_t y = m.int_var(0, 3);
    // Row 0: inf * x + y <= 0 -- an infinite slope, demoted at build. Row 1: x + y <= 1.
    m.add_constraint(m.leq(m.sum({m.prod(m.constant(inf), x), y}), m.constant(0.0)));
    m.add_constraint(m.leq(m.sum({x, y}), m.constant(1.0)));
    m.close();
    m.var_mut(vid(x)).value = 1.0;
    m.var_mut(vid(y)).value = 2.0;
    full_evaluate(m);
    LinearJumpScorer sc(m);
    mark_all_rows(m, sc);
    const std::vector<double> w = {1.0, 1.0};
    REQUIRE_FALSE(sc.prepare(vid(y), w));  // builds row 0, which demotes
    REQUIRE_FALSE(sc.row_eligible(0));
    REQUIRE(sc.cached_slopes() == 0);
    const std::vector<double> masked = {0.0, 1.0};
    REQUIRE(sc.prepare(vid(y), masked));
    REQUIRE(sc.cached_slopes() == 2);  // row 1's two, and nothing of row 0's
    REQUIRE(sc.delta(0.0) == m.weighted_violation_delta(vid(y), 0.0, masked));
    double g = 0.0;
    REQUIRE_FALSE(sc.residual_partial_at(vid(x), 0, g));  // row 0 is x's first row
    REQUIRE(sc.residual_partial_at(vid(x), 1, g));
    REQUIRE(g == 1.0);
    // Past the end of x's G_v: refused. The incidence array is flat, so without
    // the guard k = |G_x| + 1 lands on y's SECOND row -- row 1, built and exact --
    // and would be answered with y's slope as though it were x's.
    const size_t gv_size = m.constraints_of_var(vid(x)).size();
    REQUIRE_FALSE(sc.residual_partial_at(vid(x), gv_size, g));
    REQUIRE_FALSE(sc.residual_partial_at(vid(x), gv_size + 1, g));
}

TEST_CASE("slopes follow the G_v layout when the objective row is added (#176)",
          "[fj][linear_jump]") {
    // add_objective_soft_constraint rebuilds every G_v: the objective's variables
    // gain a row, and every later variable's offset in the incidence array moves.
    // Slopes written before that sit at stale positions, so the scorer must not
    // read them -- and after resize_rows it must rebuild, not reuse, them.
    for (uint64_t seed = 501; seed <= 506; ++seed) {
        RandomLinearModel r;
        build_random_linear_no_objective_row(r, seed, /*integral=*/true);
        Model& m = r.m;
        LinearJumpScorer sc(m);
        mark_all_rows(m, sc);
        RNG rng(seed);
        randomise_assignment(m, rng);
        std::vector<double> w(m.constraint_ids().size(), 1.0);
        REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);  // every row built
        REQUIRE(check_partials(m, sc) > 20);
        const size_t built_before = sc.cached_slopes();
        const size_t table_before = sc.slope_table_size();
        REQUIRE(table_before == m.num_var_constraint_incidences());

        m.add_objective_soft_constraint();
        const size_t nc = m.constraint_ids().size();
        w.assign(nc, 1.0);
        REQUIRE_THROWS_AS(sc.prepare(0, w), std::logic_error);
        double g = 0.0;
        REQUIRE_FALSE(sc.residual_partial_at(0, 0, g));

        sc.resize_rows(nc);
        sc.set_row_eligible(static_cast<int32_t>(nc - 1), true);
        REQUIRE(sc.cached_slopes() == 0);  // the layout moved: everything is rebuilt
        for (int round = 0; round < 2; ++round) {
            randomise_assignment(m, rng);
            m.set_objective_bound(round == 0 ? std::numeric_limits<double>::infinity()
                                             : static_cast<double>(rng.integers(-20, 20)));
            REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);
            REQUIRE(check_partials(m, sc) > 20);
        }
        REQUIRE(sc.cached_slopes() > built_before);  // plus the objective row's
        // The table was reallocated for the NEW layout, not kept: reusing the old
        // block would write the grown layout's slopes past its end.
        REQUIRE(sc.slope_table_size() == m.num_var_constraint_incidences());
        REQUIRE(sc.slope_table_size() > table_before);
    }
}

// ---------------------------------------------------------------------------
// Bare constraint bodies (#190): `add_constraint(expr)` means expr <= 0.

TEST_CASE("a bare body's violation is comparison_residual(body, 0) for every value",
          "[fj][linear_jump]") {
    // The scorer reads a bare body as the residual `body - 0` with a literal 0;
    // the engine reads it as clamped(body). They must agree on every input the
    // DAG can produce, non-finite included.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> values = {0.0,    -0.0,     1.0,  -1.0, 0.1,  -2.5,
                                        1e-300, 4.9e-324, 1e29, 1e30, 1e31, -1e40,
                                        1e300,  -1e300,   inf,  -inf, nan,  -nan};
    for (const double v : values) {
        INFO("v = " << v);
        for (const bool p_literal : {false, true}) {
            const double residual = comparison_residual(v, 0.0, p_literal, true);
            const double a = clamped_node_violation(v);
            const double b = clamped_node_violation(residual);
            REQUIRE(((a == b) || (std::isnan(a) && std::isnan(b))));
            // Not just after the clamp: the residual is the value itself.
            REQUIRE(((residual == v) || (std::isnan(residual) && std::isnan(v))));
        }
    }

    // And through the model: a bare row's violation, for a body that is finite,
    // +inf, -inf and NaN, against the same row posted as `body <= 0`.
    Model m;
    const int32_t y = m.float_var(-inf, inf);
    const int32_t x = m.float_var(-inf, inf);
    const int32_t body = m.sum({x, y});
    m.add_constraint(body);
    m.add_constraint(m.leq(body, m.constant(0.0)));
    m.close();
    for (const auto& [xv, yv] : std::vector<std::pair<double, double>>{
             {1.5, -0.25}, {inf, 0.0}, {-inf, 0.0}, {inf, -inf}, {nan, 1.0}, {-3.0, 3.0}}) {
        m.var_mut(vid(x)).value = xv;
        m.var_mut(vid(y)).value = yv;
        full_evaluate(m);
        ViolationManager vm(m);
        INFO("x = " << xv << ", y = " << yv);
        REQUIRE(vm.constraint_violation(0) == vm.constraint_violation(1));
    }
}

TEST_CASE("bare affine bodies are closed-form eligible, non-affine ones are not",
          "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.float_var(-2.0, 2.0);
    const int32_t y = m.float_var(-2.0, 2.0);
    const int32_t u = m.int_var(0, 3);
    m.add_constraint(m.sum({x, m.prod(m.constant(2.0), y), m.constant(-1.0)}));  // 0: Sum
    m.add_constraint(m.neg(x));                                                  // 1: Neg
    m.add_constraint(m.prod(m.constant(0.5), m.sum({x, u})));                    // 2: Prod by const
    m.add_constraint(m.div_expr(m.sum({x, y}), m.constant(4.0)));                // 3: Div by const
    m.add_constraint(m.sum({x, m.prod(x, y)}));                                  // 4: bilinear term
    m.add_constraint(m.prod(x, y));                                              // 5: bilinear
    m.add_constraint(m.abs_expr(x));                                             // 6: Abs
    m.add_constraint(m.neq(u, m.constant(2.0)));                                 // 7: Neq, a step
    m.add_constraint(m.custom({u}, std::make_unique<InputSum>(), "c"));          // 8: Custom
    m.add_constraint(m.sum({m.custom({u}, std::make_unique<InputSum>(), "d"), y}));  // 9
    m.close();
    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng);
    const LinearJumpScorer& sc = fj.linear_scorer();
    for (int32_t c = 0; c <= 3; ++c) {
        INFO("row " << c);
        REQUIRE(sc.row_eligible(c));
    }
    for (int32_t c = 4; c <= 9; ++c) {
        INFO("row " << c);
        REQUIRE_FALSE(sc.row_eligible(c));
    }
}

TEST_CASE("bare-body scores equal the probe's exactly on integral rows", "[fj][linear_jump]") {
    for (uint64_t seed = 601; seed <= 612; ++seed) {
        RandomLinearModel r;
        build_random_linear(r, seed, /*integral=*/true, /*bare=*/true);
        Model& m = r.m;
        ViolationManager vm(m);
        RNG fj_rng(seed);
        FeasibilityJump fj(m, vm, fj_rng);
        // Every row, bare or comparison, through the owner's own classification.
        for (size_t c = 0; c < m.constraint_ids().size(); ++c) {
            REQUIRE(fj.linear_scorer().row_eligible(static_cast<int32_t>(c)));
        }
        LinearJumpScorer& sc = fj.linear_scorer();
        RNG rng(seed * 31);
        for (int round = 0; round < 3; ++round) {
            randomise_assignment(m, rng);
            const std::vector<double> w = random_weights(m.constraint_ids().size(), rng);
            REQUIRE(check_scores(m, sc, w, rng, /*exact=*/true) > 50);
            for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
                const JumpResult a = compute_var_jump(m, w, v);
                const JumpResult b = compute_var_jump(m, w, v, false, &sc);
                REQUIRE(a.jump_value == b.jump_value);
                REQUIRE(a.score == b.score);
            }
        }
        REQUIRE(check_partials(m, sc) > 20);
    }
}

TEST_CASE("bare-body scores match the probe to rounding on fractional rows", "[fj][linear_jump]") {
    for (uint64_t seed = 701; seed <= 712; ++seed) {
        RandomLinearModel r;
        build_random_linear(r, seed, /*integral=*/false, /*bare=*/true);
        Model& m = r.m;
        ViolationManager vm(m);
        RNG fj_rng(seed);
        FeasibilityJump fj(m, vm, fj_rng);
        LinearJumpScorer& sc = fj.linear_scorer();
        RNG rng(seed * 104729);
        for (int round = 0; round < 3; ++round) {
            randomise_assignment(m, rng);
            const std::vector<double> w = random_weights(m.constraint_ids().size(), rng);
            REQUIRE(check_scores(m, sc, w, rng, /*exact=*/false) > 50);
        }
    }
}

TEST_CASE("FeasibilityJump scores a bare-body model in closed form", "[fj][linear_jump]") {
    RandomLinearModel r;
    build_random_linear(r, 9, /*integral=*/true, /*bare=*/true);
    Model& m = r.m;
    ViolationManager vm(m);
    RNG rng(9);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(true);
    (void)fj.batch(200);
    REQUIRE(fj.linear_scorer().fast_prepares() > 0);
    REQUIRE(fj.linear_scorer().fallback_prepares() == 0);
}

TEST_CASE("a non-finite bare body falls back to the probe", "[fj][linear_jump]") {
    const double inf = std::numeric_limits<double>::infinity();
    SECTION("an infinite variable in the body") {
        Model m;
        const int32_t z = m.float_var(-2.0, 2.0);
        const int32_t s = m.float_var(-inf, inf);
        m.add_constraint(m.sum({s, z}));  // s + z <= 0
        m.add_constraint(m.sum({z, m.constant(-1.0)}));
        m.close();
        m.var_mut(vid(z)).value = 0.5;
        m.var_mut(vid(s)).value = inf;
        full_evaluate(m);
        ViolationManager vm(m);
        RNG rng(1);
        FeasibilityJump fj(m, vm, rng);
        LinearJumpScorer& sc = fj.linear_scorer();
        REQUIRE(sc.row_eligible(0));
        const std::vector<double> w = {1.0, 1.0};
        REQUIRE_FALSE(sc.prepare(vid(z), w));
        REQUIRE_FALSE(sc.prepare(vid(s), w));
        for (const int32_t v : {vid(z), vid(s)}) {
            const JumpResult probe = compute_var_jump(m, w, v);
            const JumpResult fast = compute_var_jump(m, w, v, false, &sc);
            REQUIRE(fast.jump_value == probe.jump_value);
            REQUIRE(fast.score == probe.score);
        }
        // Back to finite: the same rows take the closed form, and agree.
        m.var_mut(vid(s)).value = -1.0;
        full_evaluate(m);
        REQUIRE(sc.prepare(vid(z), w));
        REQUIRE(sc.delta(-2.0) == m.weighted_violation_delta(vid(z), -2.0, w));
    }
    SECTION("a zero slope through Div by a near-zero constant") {
        // Affine by rule, local derivative 0, but +inf for y >= 0 and -inf for
        // y < 0: moving y flips the row from maximal violation to satisfied.
        Model m;
        const int32_t x = m.int_var(0, 2);
        const int32_t y = m.int_var(-1, 1);
        m.add_constraint(m.sum({x, m.div_expr(y, m.constant(0.0)), m.constant(-1.0)}));
        m.close();
        m.var_mut(vid(x)).value = 0.0;
        m.var_mut(vid(y)).value = 1.0;
        full_evaluate(m);
        ViolationManager vm(m);
        RNG rng(1);
        FeasibilityJump fj(m, vm, rng);
        LinearJumpScorer& sc = fj.linear_scorer();
        REQUIRE(sc.row_eligible(0));
        const std::vector<double> w = {1.0};
        REQUIRE(m.weighted_violation_delta(vid(y), -1.0, w) == -kInfPenalty);
        REQUIRE_FALSE(sc.prepare(vid(y), w));
        REQUIRE(compute_var_jump(m, w, vid(y), false, &sc).score == kInfPenalty);
    }
    SECTION("a NaN body") {
        Model m;
        const int32_t z = m.float_var(-2.0, 2.0);
        const int32_t s = m.float_var(-inf, inf);
        const int32_t t = m.float_var(-inf, inf);
        m.add_constraint(m.sum({s, t, z}));  // inf + -inf: NaN, maximally violated
        m.close();
        m.var_mut(vid(z)).value = 0.0;
        m.var_mut(vid(s)).value = inf;
        m.var_mut(vid(t)).value = -inf;
        full_evaluate(m);
        ViolationManager vm(m);
        REQUIRE(vm.constraint_violation(0) == kInfPenalty);
        RNG rng(1);
        FeasibilityJump fj(m, vm, rng);
        LinearJumpScorer& sc = fj.linear_scorer();
        const std::vector<double> w = {1.0};
        REQUIRE_FALSE(sc.prepare(vid(z), w));
        const JumpResult probe = compute_var_jump(m, w, vid(z));
        const JumpResult fast = compute_var_jump(m, w, vid(z), false, &sc);
        REQUIRE(fast.jump_value == probe.jump_value);
        REQUIRE(fast.score == probe.score);
    }
}

TEST_CASE("a bare body's Newton step reads the cached row partial", "[fj][linear_jump]") {
    Model m;
    const int32_t x = m.float_var(-10.0, 10.0);
    // Row 0 satisfied with another slope; row 1 is 2x - 3 <= 0, a bare body.
    m.add_constraint(m.sum({m.prod(m.constant(3.0), x), m.constant(-100.0)}));
    m.add_constraint(m.sum({m.prod(m.constant(2.0), x), m.constant(-3.0)}));
    m.close();
    m.var_mut(vid(x)).value = 5.0;
    full_evaluate(m);
    ViolationManager vm(m);
    RNG rng(1);
    FeasibilityJump fj(m, vm, rng);
    LinearJumpScorer& sc = fj.linear_scorer();
    const std::vector<double> w = {1.0, 1.0};
    const JumpResult r = compute_var_jump(m, w, vid(x), false, &sc);
    REQUIRE(r.jump_value == 1.5);
    REQUIRE(sc.cached_partials() == 1);
    REQUIRE(sc.fast_prepares() == 1);
}
