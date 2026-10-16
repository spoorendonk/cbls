// Correctness audit on generated models (#225).
//
// Seeded random models -- linear MIP rows (<=, >=, ==, ranged) over Bool, Int
// and Float columns with narrow and wide domains; nonlinear rows (products,
// squares, division, exp/log/sin/sqrt, abs/min/max); and Set/List variables
// read through lambda sums -- each built around a planted assignment, so every
// instance is feasible. Each runs under the default search and each opt-in
// FJ setting, and three things are checked:
//
//  1. INCREMENTAL CONSISTENCY. A `SearchAuditor` calls
//     `FeasibilityJump::audit` before every batch -- after single moves,
//     Novelty compound moves, perturbations, LNS repairs, the inner-solver
//     polish and weight resets -- and after every FJ and Novelty batch. It
//     compares every node value, V and its counts, the jump table's scores
//     and Q against a from-scratch recompute; its doc states the tolerances.
//  2. REPORTED FEASIBILITY. Every assignment the engine reports as a new
//     feasible best (the callback, which runs while the model holds that
//     assignment) and the returned `best_state` pass an independent check that
//     shares no code with the DAG: each row, bound, integrality and Set/List
//     rule is re-derived from the generator's own plain-C++ formula. The
//     reported objective equals the formula's.
//  3. OBJECTIVE BOOKKEEPING. The returned objective is the best of the
//     reported stream, for minimise and maximise (whose objective the engine
//     reports in its internal, minimised sense: -f), with a constant offset.
//
// Deterministic: iteration budgets only, no wall clock.
#include "test_helpers.h"

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <cbls/audit.h>
#include <cbls/cbls.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace cbls;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

using Fn = std::function<double(const Model::State&)>;

// lo <= body <= hi, either side possibly infinite.
struct Row {
    Fn body;
    double lo = -kInf;
    double hi = kInf;
    double scale = 1.0;  // sum of |terms| at the planted point, for the check's slack
    // `body < hi` / `body > lo`: over an integral body the engine reads it as
    // `body <= hi - 1` (dag.h's kIntegralStrictMargin), so the check does too.
    bool strict = false;
    bool integral = false;
    // `body != neq`: satisfied when |body - neq| >= 1e-9, NodeOp::Neq's rule.
    bool is_neq = false;
    double neq = 0.0;
};

struct Structured {
    int32_t var = -1;
    int universe = 0;
    int min_len = 0;
    int max_len = 0;
};

struct Instance {
    std::string family;
    Model m;
    std::vector<Row> rows;
    std::vector<Structured> structured;
    Fn objective;  // user sense; empty on a pure feasibility model
    bool maximize = false;
};

// ---- the independent check: no DAG, no engine evaluation ------------------

void check_columns(const Instance& inst, const Model::State& st, std::vector<std::string>& errs) {
    for (size_t i = 0; i < inst.m.num_vars(); ++i) {
        const Variable& v = inst.m.var(static_cast<int32_t>(i));
        if (v.type != VarType::Bool && v.type != VarType::Int && v.type != VarType::Float) {
            continue;
        }
        const double x = st.values[i];
        const std::string name = "var " + std::to_string(i) + " = " + std::to_string(x);
        if (x < v.lb || x > v.ub || std::isnan(x)) {
            errs.push_back(name + " outside its bounds");
        }
        if (v.type != VarType::Float && x != std::round(x)) {
            errs.push_back(name + " not integral");
        }
    }
}

void check_structured(const Instance& inst, const Model::State& st,
                      std::vector<std::string>& errs) {
    for (const Structured& s : inst.structured) {
        const std::vector<int32_t>& e = st.elements[static_cast<size_t>(s.var)];
        const std::set<int32_t> distinct(e.begin(), e.end());
        const std::string name = "structured var " + std::to_string(s.var);
        if (distinct.size() != e.size()) {
            errs.push_back(name + " repeats an element");
        }
        if (!distinct.empty() && (*distinct.begin() < 0 || *distinct.rbegin() >= s.universe)) {
            errs.push_back(name + " leaves its universe");
        }
        const auto n = static_cast<int>(e.size());
        if (n < s.min_len || n > s.max_len) {
            errs.push_back(name + " has size " + std::to_string(n));
        }
    }
}

void check_rows(const Instance& inst, const Model::State& st, double tol,
                std::vector<std::string>& errs) {
    for (size_t r = 0; r < inst.rows.size(); ++r) {
        const Row& row = inst.rows[r];
        const double b = row.body(st);
        bool ok = false;
        if (row.is_neq) {
            ok = std::fabs(b - row.neq) >= 1e-9;
        } else {
            const double slack = tol + (1e-9 * row.scale);
            const double gap = (row.strict && row.integral) ? 1.0 : 0.0;
            ok = b >= row.lo + gap - slack && b <= row.hi - gap + slack;
        }
        if (!ok) {
            errs.push_back("row " + std::to_string(r) + " fails at body " + std::to_string(b));
        }
    }
}

// Bounds, integrality, Set/List rules and every row, from the generator's own
// formulas.
std::vector<std::string> check_assignment(const Instance& inst, const Model::State& st,
                                          double tol) {
    std::vector<std::string> errs;
    check_columns(inst, st, errs);
    check_structured(inst, st, errs);
    check_rows(inst, st, tol, errs);
    return errs;
}

// ---- the generator ----------------------------------------------------------

struct Gen {
    std::mt19937_64 rng;
    explicit Gen(uint64_t seed) : rng(seed) {}
    double uniform(double lo, double hi) { return std::uniform_real_distribution<>(lo, hi)(rng); }
    int integer(int lo, int hi) { return std::uniform_int_distribution<>(lo, hi)(rng); }
    bool coin(double p = 0.5) { return uniform(0.0, 1.0) < p; }
};

struct Column {
    Expr e;
    int32_t id;
    double planted;
};

// A scalar column of a random type and domain width, and a planted value in it.
Column add_column(Instance& inst, Gen& g, bool narrow) {
    Model& m = inst.m;
    const int kind = g.integer(0, narrow ? 2 : 5);
    Expr e{&m, 0};
    double planted = 0.0;
    switch (kind) {
        case 0:
            e = m.Bool();
            planted = g.coin() ? 1.0 : 0.0;
            break;
        case 1:
            e = m.Int(-5, 5);
            planted = g.integer(-5, 5);
            break;
        case 2:
            e = m.Float(-4.0, 4.0);
            planted = g.uniform(-4.0, 4.0);
            break;
        case 3:
            e = m.Int(-100000, 100000);
            planted = g.integer(-60, 60);
            break;
        case 4:
            e = m.Float(-1.0e4, 1.0e4);
            planted = g.uniform(-50.0, 50.0);
            break;
        default:  // magnitudes where a fractional row's running sum rounds
            e = m.Float(-1.0e9, 1.0e9);
            planted = g.uniform(-1.0e3, 1.0e3);
            break;
    }
    return {e, e.var_id(), planted};
}

Model::State planted_state(const Instance& inst, const std::vector<Column>& cols) {
    Model::State st = inst.m.copy_state();
    for (const Column& c : cols) {
        st.values[static_cast<size_t>(c.id)] = c.planted;
    }
    return st;
}

// Add `body` as a row of random sense around its planted value.
void add_row(Instance& inst, Gen& g, const Expr& body, Fn fn, const Model::State& planted,
             bool allow_eq, bool integral = false) {
    const double at = fn(planted);
    Row row;
    row.body = std::move(fn);
    row.scale = 1.0 + std::fabs(at);
    const double s1 = g.coin(0.3) ? 0.0 : g.uniform(0.0, 3.0);
    const double s2 = g.coin(0.3) ? 0.0 : g.uniform(0.0, 3.0);
    row.integral = integral;
    // Strict and != rows a fifth of the time; the planted point satisfies both.
    int sense = g.integer(0, allow_eq ? 3 : 2);
    if (g.coin(0.2)) {
        sense = g.integer(4, 6);
    }
    const double step = integral ? static_cast<double>(g.integer(1, 3)) : g.uniform(0.01, 3.0);
    Model& m = inst.m;
    switch (sense) {
        case 4:
            row.strict = true;
            row.hi = at + step;
            m.add_constraint(body < row.hi);
            break;
        case 5:
            row.strict = true;
            row.lo = at - step;
            m.add_constraint(body > row.lo);
            break;
        case 6:
            row.is_neq = true;
            row.neq = at + (g.coin() ? step : -step);
            m.add_constraint(body.neq(m.Constant(row.neq)));
            break;
        case 0:
            row.hi = at + s1;
            m.add_constraint(body <= row.hi);
            break;
        case 1:
            row.lo = at - s1;
            m.add_constraint(body >= row.lo);
            break;
        case 2:  // ranged: two rows over one body, as the MPS reader builds it
            row.lo = at - s1;
            row.hi = at + s2;
            m.add_constraint(body >= row.lo);
            m.add_constraint(body <= row.hi);
            break;
        default:
            row.lo = at;
            row.hi = at;
            m.add_constraint(body.eq(m.Constant(at)));
            break;
    }
    inst.rows.push_back(std::move(row));
}

// A linear objective plus a constant offset, minimised or maximised.
void add_linear_objective(Instance& inst, Gen& g, const std::vector<Column>& cols) {
    Model& m = inst.m;
    const double offset = g.uniform(-100.0, 100.0);
    Expr obj = m.Constant(offset);
    std::vector<std::pair<int32_t, double>> terms;
    for (const Column& c : cols) {
        if (g.coin(0.6)) {
            const double a = static_cast<double>(g.integer(-4, 4)) + (g.coin() ? 0.5 : 0.0);
            obj = obj + (a * c.e);
            terms.emplace_back(c.id, a);
        }
    }
    inst.objective = [terms, offset](const Model::State& st) {
        double v = offset;
        for (const auto& [id, a] : terms) {
            v += a * st.values[static_cast<size_t>(id)];
        }
        return v;
    };
    inst.maximize = g.coin();
    if (inst.maximize) {
        m.maximize(obj);
    } else {
        m.minimize(obj);
    }
}

std::unique_ptr<Instance> linear_instance(uint64_t seed) {
    auto inst = std::make_unique<Instance>();
    inst->family = "linear";
    Gen g(seed);
    std::vector<Column> cols;
    const int n = g.integer(5, 12);
    cols.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        cols.push_back(add_column(*inst, g, /*narrow=*/false));
    }
    const Model::State planted = planted_state(*inst, cols);
    const int rows = g.integer(3, 9);
    for (int r = 0; r < rows; ++r) {
        const int k = g.integer(2, std::min(4, n));
        std::vector<std::pair<int32_t, double>> terms;
        Expr body = inst->m.Constant(0.0);
        std::set<int32_t> used;
        const bool fractional = g.coin(0.4);
        bool integral = !fractional;
        while (static_cast<int>(terms.size()) < k) {
            const Column& c = cols[static_cast<size_t>(g.integer(0, n - 1))];
            if (!used.insert(c.id).second) {
                continue;
            }
            double a = g.integer(1, 5) * (g.coin() ? 1.0 : -1.0);
            if (fractional) {
                a *= 0.1 * g.integer(1, 9);
            }
            body = body + (a * c.e);
            terms.emplace_back(c.id, a);
            integral = integral && c.e.var().type != VarType::Float;
        }
        Fn fn = [terms](const Model::State& st) {
            double v = 0.0;
            for (const auto& [id, a] : terms) {
                v += a * st.values[static_cast<size_t>(id)];
            }
            return v;
        };
        add_row(*inst, g, body, std::move(fn), planted, /*allow_eq=*/true, integral);
    }
    if (g.coin(0.85)) {
        add_linear_objective(*inst, g, cols);
    }
    inst->m.close();
    return inst;
}

// A larger pure-integer model: 15-30 Bool/Int columns, longer rows with
// integer data, equalities and ranges common -- where incremental Sums
// and the exact Int jump engage.
std::unique_ptr<Instance> mip_instance(uint64_t seed) {
    auto inst = std::make_unique<Instance>();
    inst->family = "mip";
    Gen g(seed);
    Model& m = inst->m;
    std::vector<Column> cols;
    const int n = g.integer(15, 30);
    for (int i = 0; i < n; ++i) {
        if (g.coin(0.4)) {
            Expr e = m.Bool();
            cols.push_back({e, e.var_id(), g.coin() ? 1.0 : 0.0});
        } else {
            const int ub = g.coin(0.2) ? 1000 : 10;
            Expr e = m.Int(0, ub);
            cols.push_back({e, e.var_id(), static_cast<double>(g.integer(0, 10))});
        }
    }
    const Model::State planted = planted_state(*inst, cols);
    const int rows = g.integer(6, 14);
    for (int r = 0; r < rows; ++r) {
        const int k = g.integer(3, 8);
        std::vector<std::pair<int32_t, double>> terms;
        std::set<int32_t> used;
        Expr body = m.Constant(0.0);
        while (static_cast<int>(terms.size()) < k) {
            const Column& c = cols[static_cast<size_t>(g.integer(0, n - 1))];
            if (!used.insert(c.id).second) {
                continue;
            }
            const double a = g.integer(1, 6) * (g.coin(0.7) ? 1.0 : -1.0);
            body = body + (a * c.e);
            terms.emplace_back(c.id, a);
        }
        Fn fn = [terms](const Model::State& st) {
            double v = 0.0;
            for (const auto& [id, a] : terms) {
                v += a * st.values[static_cast<size_t>(id)];
            }
            return v;
        };
        add_row(*inst, g, body, std::move(fn), planted, /*allow_eq=*/true, /*integral=*/true);
    }
    add_linear_objective(*inst, g, cols);
    m.close();
    return inst;
}

// One nonlinear term over narrow-domain columns, as an Expr and as a formula.
std::pair<Expr, Fn> nonlinear_term(Instance& inst, Gen& g, const std::vector<Column>& cols,
                                   const std::vector<Column>& positive) {
    Model& m = inst.m;
    auto pick = [&](const std::vector<Column>& from) -> const Column& {
        return from[static_cast<size_t>(g.integer(0, static_cast<int>(from.size()) - 1))];
    };
    const Column& x = pick(cols);
    const Column& y = pick(cols);
    const Column& p = pick(positive);
    const auto xi = static_cast<size_t>(x.id);
    const auto yi = static_cast<size_t>(y.id);
    const auto pi = static_cast<size_t>(p.id);
    const auto c = static_cast<double>(g.integer(-2, 2));
    switch (g.integer(0, 14)) {
        case 10:
            return {floor(0.5 * x.e),
                    [xi](const Model::State& s) { return std::floor(0.5 * s.values[xi]); }};
        case 11:
            return {round(x.e * y.e), [xi, yi](const Model::State& s) {
                        return std::round(s.values[xi] * s.values[yi]);
                    }};
        case 12:
            return {ceil(x.e - 0.3),
                    [xi](const Model::State& s) { return std::ceil(s.values[xi] - 0.3); }};
        case 13: {
            // An index read the way Element reads it: truncated toward zero,
            // 0.0 outside the table.
            std::vector<double> table(5);
            for (double& t : table) {
                t = static_cast<double>(g.integer(-6, 6));
            }
            return {element(table, p.e), [table, pi](const Model::State& s) {
                        const double i = std::trunc(s.values[pi]);
                        return (i >= 0.0 && i < static_cast<double>(table.size()))
                                   ? table[static_cast<size_t>(i)]
                                   : 0.0;
                    }};
        }
        case 14:
            return {if_then_else(x.e - c, y.e, -y.e), [xi, yi, c](const Model::State& s) {
                        return s.values[xi] - c > 0.0 ? s.values[yi] : -s.values[yi];
                    }};
        case 0:
            return {x.e * y.e,
                    [xi, yi](const Model::State& s) { return s.values[xi] * s.values[yi]; }};
        case 1:
            return {pow(x.e, m.Constant(2.0)),
                    [xi](const Model::State& s) { return s.values[xi] * s.values[xi]; }};
        case 2:
            return {x.e / (p.e + 1.0), [xi, pi](const Model::State& s) {
                        return s.values[xi] / (s.values[pi] + 1.0);
                    }};
        case 3:
            return {exp(0.3 * x.e),
                    [xi](const Model::State& s) { return std::exp(0.3 * s.values[xi]); }};
        case 4:
            return {log(p.e + 1.0),
                    [pi](const Model::State& s) { return std::log(s.values[pi] + 1.0); }};
        case 5:
            return {sin(x.e), [xi](const Model::State& s) { return std::sin(s.values[xi]); }};
        case 6:
            return {sqrt(p.e), [pi](const Model::State& s) { return std::sqrt(s.values[pi]); }};
        case 7:
            return {abs(x.e - c),
                    [xi, c](const Model::State& s) { return std::fabs(s.values[xi] - c); }};
        case 8:
            return {min({x.e, y.e}), [xi, yi](const Model::State& s) {
                        return std::min(s.values[xi], s.values[yi]);
                    }};
        default:
            return {max({x.e, y.e}), [xi, yi](const Model::State& s) {
                        return std::max(s.values[xi], s.values[yi]);
                    }};
    }
}

std::unique_ptr<Instance> nonlinear_instance(uint64_t seed) {
    auto inst = std::make_unique<Instance>();
    inst->family = "nonlinear";
    Gen g(seed);
    std::vector<Column> cols;
    std::vector<Column> positive;
    const int n = g.integer(3, 7);
    cols.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        cols.push_back(add_column(*inst, g, /*narrow=*/true));
    }
    for (int i = 0; i < 2; ++i) {
        Expr e = g.coin() ? inst->m.Float(0.0, 5.0) : inst->m.Int(0, 6);
        const bool is_int = e.var().type == VarType::Int;
        positive.push_back(
            {e, e.var_id(), is_int ? static_cast<double>(g.integer(0, 6)) : g.uniform(0.0, 5.0)});
    }
    std::vector<Column> all = cols;
    all.insert(all.end(), positive.begin(), positive.end());
    const Model::State planted = planted_state(*inst, all);
    const int rows = g.integer(2, 6);
    for (int r = 0; r < rows; ++r) {
        auto [body, fn] = nonlinear_term(*inst, g, cols, positive);
        if (g.coin()) {
            auto [b2, f2] = nonlinear_term(*inst, g, cols, positive);
            const double a = g.uniform(-2.0, 2.0);
            body = body + (a * b2);
            fn = [f = fn, f2 = f2, a](const Model::State& s) { return f(s) + (a * f2(s)); };
        }
        const Column& lin =
            all[static_cast<size_t>(g.integer(0, static_cast<int>(all.size()) - 1))];
        const auto a = static_cast<double>(g.integer(-3, 3));
        body = body + (a * lin.e);
        const auto li = static_cast<size_t>(lin.id);
        fn = [f = fn, a, li](const Model::State& s) { return f(s) + (a * s.values[li]); };
        add_row(*inst, g, body, std::move(fn), planted, /*allow_eq=*/g.coin(0.3));
    }
    if (g.coin(0.85)) {
        auto [term, fn] = nonlinear_term(*inst, g, cols, positive);
        const double offset = g.uniform(-50.0, 50.0);
        Expr obj = term + offset;
        inst->objective = [f = fn, offset](const Model::State& s) { return f(s) + offset; };
        inst->maximize = g.coin();
        if (inst->maximize) {
            inst->m.maximize(obj);
        } else {
            inst->m.minimize(obj);
        }
    }
    inst->m.close();
    return inst;
}

std::unique_ptr<Instance> structured_instance(uint64_t seed) {
    auto inst = std::make_unique<Instance>();
    inst->family = "structured";
    Gen g(seed);
    Model& m = inst->m;
    const int universe = g.integer(5, 9);
    const int min_size = g.integer(1, 2);
    const int max_size = g.integer(min_size + 1, universe - 1);
    const int32_t set = m.set_var(universe, min_size, max_size);
    inst->structured.push_back({vid(set), universe, min_size, max_size});
    const int perm_n = g.integer(4, 7);
    const int32_t list = m.list_var(perm_n);
    inst->structured.push_back({vid(list), perm_n, perm_n, perm_n});

    std::vector<double> w(static_cast<size_t>(universe));
    for (double& x : w) {
        x = static_cast<double>(g.integer(1, 9));
    }
    std::vector<std::vector<double>> d(static_cast<size_t>(perm_n),
                                       std::vector<double>(static_cast<size_t>(perm_n)));
    for (auto& r : d) {
        for (double& x : r) {
            x = static_cast<double>(g.integer(1, 20));
        }
    }
    const Expr weight{&m, m.lambda_sum(set, [w](int e) { return w[static_cast<size_t>(e)]; })};
    const Expr length{&m, m.pair_lambda_sum(list, [d](int a, int b) {
                          return d[static_cast<size_t>(a)][static_cast<size_t>(b)];
                      })};
    const Expr count{&m, m.count(set)};
    const auto set_id = static_cast<size_t>(vid(set));
    const auto list_id = static_cast<size_t>(vid(list));
    Fn weight_fn = [w, set_id](const Model::State& s) {
        double v = 0.0;
        for (const int32_t e : s.elements[set_id]) {
            v += w[static_cast<size_t>(e)];
        }
        return v;
    };
    Fn length_fn = [d, list_id](const Model::State& s) {
        const std::vector<int32_t>& e = s.elements[list_id];
        double v = 0.0;
        for (size_t k = 0; k + 1 < e.size(); ++k) {
            v += d[static_cast<size_t>(e[k])][static_cast<size_t>(e[k + 1])];
        }
        return v;
    };
    Fn count_fn = [set_id](const Model::State& s) {
        return static_cast<double>(s.elements[set_id].size());
    };

    // Planted: a random subset of the right size, a random permutation, and a
    // couple of scalars tied to them.
    std::vector<Column> cols;
    cols.push_back(add_column(*inst, g, /*narrow=*/true));
    cols.push_back(add_column(*inst, g, /*narrow=*/true));
    Model::State planted = planted_state(*inst, cols);
    std::vector<int32_t> pool(static_cast<size_t>(universe));
    for (int i = 0; i < universe; ++i) {
        pool[static_cast<size_t>(i)] = i;
    }
    std::shuffle(pool.begin(), pool.end(), g.rng);
    pool.resize(static_cast<size_t>(g.integer(min_size, max_size)));
    planted.elements[set_id] = pool;
    std::vector<int32_t> perm(static_cast<size_t>(perm_n));
    for (int i = 0; i < perm_n; ++i) {
        perm[static_cast<size_t>(i)] = i;
    }
    std::shuffle(perm.begin(), perm.end(), g.rng);
    planted.elements[list_id] = perm;

    add_row(*inst, g, weight, weight_fn, planted, /*allow_eq=*/false);
    add_row(*inst, g, count, count_fn, planted, /*allow_eq=*/false);
    const auto c0 = static_cast<size_t>(cols[0].id);
    add_row(
        *inst, g, weight + (2.0 * cols[0].e),
        [weight_fn, c0](const Model::State& s) { return weight_fn(s) + (2.0 * s.values[c0]); },
        planted, /*allow_eq=*/false);
    const auto c1 = static_cast<size_t>(cols[1].id);
    add_row(
        *inst, g, length - cols[1].e,
        [length_fn, c1](const Model::State& s) { return length_fn(s) - s.values[c1]; }, planted,
        /*allow_eq=*/false);

    const double offset = g.uniform(-30.0, 30.0);
    inst->objective = [length_fn, weight_fn, c0, offset](const Model::State& s) {
        return length_fn(s) - weight_fn(s) + s.values[c0] + offset;
    };
    const Expr obj = length - weight + cols[0].e + offset;
    inst->maximize = g.coin(0.3);
    if (inst->maximize) {
        m.maximize(obj);
    } else {
        m.minimize(obj);
    }
    m.close();
    return inst;
}

// ---- the search side ----------------------------------------------------------

struct Arm {
    const char* name;
    void (*apply)(SearchConfig&);
};

const std::vector<Arm>& arms() {
    static const std::vector<Arm> kArms = {
        {"default", [](SearchConfig&) {}},
        {"fj_exact_int_jump", [](SearchConfig& c) { c.fj_exact_int_jump = true; }},
        {"perturb_near_value", [](SearchConfig& c) { c.perturb_near_value = true; }},
        {"fj_objective_tie_break", [](SearchConfig& c) { c.fj_objective_tie_break = true; }},
        {"all_opt_in",
         [](SearchConfig& c) {
             c.fj_exact_int_jump = true;
             c.perturb_near_value = true;
             c.fj_objective_tie_break = true;
         }},
    };
    return kArms;
}

// Collects every finding, labelled with where the search was.
struct Auditor : SearchAuditor, Tracer, SolveCallback {
    const Instance* inst = nullptr;
    std::vector<std::string> findings;
    std::string last_event = "start";
    int audits = 0;
    int kicks = 0;
    int lns_repairs = 0;
    std::vector<double> reported;  // objective of every reported feasible best

    void note(const std::string& where, const std::vector<std::string>& found) {
        for (const std::string& f : found) {
            if (findings.size() < 20) {
                std::string line = where;
                line += ": ";
                line += f;
                findings.push_back(std::move(line));
            }
        }
    }
    void audit(AuditPoint point, const FeasibilityJump& fj) override {
        ++audits;
        const char* at = "after Novelty batch";
        if (point == AuditPoint::BeforeBatch) {
            at = "before batch";
        } else if (point == AuditPoint::AfterFeasibilityJump) {
            at = "after FJ batch";
        }
        note(std::string(at) + " (last: " + last_event + ")", fj.audit());
        last_event = at;
    }
    void batch_end(BatchKind kind, int64_t /*iterations*/, bool improved) override {
        last_event =
            std::string(batch_kind_name(kind)) + (improved ? " batch, new best" : " batch");
    }
    void kick(KickKind kind) override {
        ++kicks;
        last_event = kick_kind_name(kind);
    }
    void lns(bool accepted) override {
        ++lns_repairs;
        last_event = accepted ? "lns accepted" : "lns rejected";
    }
    void hook(double /*seconds*/) override { last_event = "inner solver"; }
    void on_progress(const SolveProgress& p) override {
        if (!p.new_best || !p.feasible) {
            return;
        }
        // The callback runs while the model holds the assignment just recorded.
        const Model::State st = inst->m.copy_state();
        note("reported best", check_assignment(*inst, st, kDefaultFeasibilityTolerance));
        if (inst->objective) {
            const double f = inst->objective(st);
            const double internal = inst->maximize ? -f : f;
            if (!(std::fabs(p.objective - internal) <= 1e-7 * (1.0 + std::fabs(internal)))) {
                findings.push_back("reported best objective " + std::to_string(p.objective) +
                                   " != recomputed " + std::to_string(internal));
            }
        }
        reported.push_back(p.objective);
    }
};

struct Outcome {
    int audits = 0;
    int kicks = 0;
    int lns = 0;
    bool feasible = false;
};

Outcome audit_run(Instance& inst, const Arm& arm, uint64_t seed, int64_t iterations) {
    Auditor a;
    a.inst = &inst;
    SearchConfig cfg;
    arm.apply(cfg);
    cfg.max_iterations = iterations;
    cfg.batch_iterations = 40;
    cfg.perturbation_period = 4;
    cfg.auditor = &a;
    cfg.tracer = &a;
    LNS lns;
    FloatIntensifyHook hook;
    const SearchResult r = solve(inst.m, /*time_limit=*/0.0, seed, /*use_fj=*/true, &hook, &lns,
                                 /*lns_interval=*/2, &a, cfg);

    const std::string ctx = inst.family + " seed " + std::to_string(seed) + " arm " + arm.name;
    INFO(ctx);
    for (const std::string& f : a.findings) {
        UNSCOPED_INFO(f);
    }
    CHECK(a.findings.empty());

    // The returned assignment, and the model solve() left at it.
    if (r.feasible) {
        CHECK(check_assignment(inst, r.best_state, kDefaultFeasibilityTolerance).empty());
        CHECK(check_assignment(inst, inst.m.copy_state(), kDefaultFeasibilityTolerance).empty());
        if (inst.objective) {
            const double f = inst.objective(r.best_state);
            const double internal = inst.maximize ? -f : f;
            CHECK(std::fabs(r.objective - internal) <= 1e-7 * (1.0 + std::fabs(internal)));
            // The best is the best of the reported stream (each strictly better).
            REQUIRE_FALSE(a.reported.empty());
            double best = kInf;
            for (const double o : a.reported) {
                best = std::min(best, o);
            }
            CHECK(r.objective == best);
            CHECK(r.first_feasible_objective == a.reported.front());
        }
    } else {
        CHECK(a.reported.empty());
    }
    return {a.audits, a.kicks, a.lns_repairs, r.feasible};
}

// CBLS_AUDIT_SEEDS=N widens every family to N seeds, for a soak run outside
// the fast suite.
void audit_family(std::unique_ptr<Instance> (*make)(uint64_t), uint64_t first_seed, int count,
                  int64_t iterations) {
    if (const char* soak = std::getenv("CBLS_AUDIT_SEEDS")) {
        int64_t n = 0;
        if (cbls::try_parse_int64(soak, n) && n > 0) {
            count = static_cast<int>(n);
        }
    }
    Outcome total;
    int runs = 0;
    for (uint64_t s = first_seed; s < first_seed + static_cast<uint64_t>(count); ++s) {
        for (const Arm& arm : arms()) {
            std::unique_ptr<Instance> inst = make(s);
            const Outcome o = audit_run(*inst, arm, s, iterations);
            total.audits += o.audits;
            total.kicks += o.kicks;
            total.lns += o.lns;
            total.feasible = total.feasible || o.feasible;
            ++runs;
        }
    }
    // The audit is only evidence if it ran where it says it did.
    CHECK(total.audits > 20 * runs);
    CHECK(total.kicks > runs);
    CHECK(total.lns > 0);
    CHECK(total.feasible);
}

// The portfolio: every worker audited, and the merged result checked
// independently. A small iteration cap makes each worker's solve() return and
// restart on the time left, and its stagnation kicks may adopt a peer's pool
// solution: both paths only exist here, so the tracers count them and the
// test case requires each to have been audited at least once. Wall-clock
// bounded, so how often varies; what it asserts does not.
struct SharedAuditor : SearchAuditor {
    std::mutex mu;
    std::vector<std::string> findings;
    int audits = 0;
    std::atomic<int> adoptions{0};
    std::atomic<int> restarts{0};
    void audit(AuditPoint /*point*/, const FeasibilityJump& fj) override {
        std::vector<std::string> found = fj.audit();
        const std::scoped_lock lock(mu);
        ++audits;
        for (std::string& f : found) {
            if (findings.size() < 20) {
                findings.push_back(std::move(f));
            }
        }
    }
};

// One per worker. A restart starts a fresh solve(), whose cumulative
// iteration count starts again from zero.
struct WorkerTracer : Tracer {
    SharedAuditor* shared = nullptr;
    int64_t last_iterations = 0;
    void batch_end(BatchKind /*kind*/, int64_t iterations, bool /*improved*/) override {
        if (iterations < last_iterations) {
            ++shared->restarts;
        }
        last_iterations = iterations;
    }
    void kick(KickKind kind) override {
        if (kind == KickKind::Adopt) {
            ++shared->adoptions;
        }
    }
};

struct PortfolioOutcome {
    int adoptions = 0;
    int restarts = 0;
};

PortfolioOutcome audit_portfolio(Instance& inst, uint64_t seed) {
    SharedAuditor a;
    SearchConfig cfg;
    cfg.batch_iterations = 40;
    cfg.perturbation_period = 4;
    cfg.max_iterations = 1500;
    cfg.auditor = &a;
    ParallelConfig par;
    par.n_threads = 2;
    par.tracer_factory = [&a](int /*worker*/) {
        auto t = std::make_unique<WorkerTracer>();
        t->shared = &a;
        return std::unique_ptr<Tracer>(std::move(t));
    };
    ParallelSearch ps(2);
    const SearchResult r = ps.solve(
        inst.m, /*time_limit=*/0.15, seed, cfg, nullptr, [] { return std::make_shared<LNS>(); },
        nullptr, par);
    INFO(inst.family + " seed " + std::to_string(seed) + " portfolio");
    for (const std::string& f : a.findings) {
        UNSCOPED_INFO(f);
    }
    CHECK(a.findings.empty());
    CHECK(a.audits > 0);
    if (r.feasible) {
        CHECK(check_assignment(inst, r.best_state, kDefaultFeasibilityTolerance).empty());
        if (inst.objective) {
            const double f = inst.objective(r.best_state);
            const double internal = inst.maximize ? -f : f;
            CHECK(std::fabs(r.objective - internal) <= 1e-7 * (1.0 + std::fabs(internal)));
        }
    }
    return {a.adoptions.load(), a.restarts.load()};
}

}  // namespace

TEST_CASE("audit: linear MIP rows over Bool/Int/Float columns", "[audit]") {
    audit_family(linear_instance, 1000, 8, 3000);
}

TEST_CASE("audit: larger integer models with long equality rows", "[audit]") {
    // Three seeds: with #222's equality-balance arm removed (#226), seed 4000
    // alone reaches no feasible point in any arm, and the audit then has no
    // evidence that it checked a reported feasible result.
    audit_family(mip_instance, 4000, 3, 2000);
}

TEST_CASE("audit: nonlinear DAG rows", "[audit]") {
    audit_family(nonlinear_instance, 2000, 8, 3000);
}

TEST_CASE("audit: Set and List variables through lambda sums", "[audit]") {
    audit_family(structured_instance, 3000, 6, 4000);
}

TEST_CASE("audit: portfolio workers and the merged result", "[audit]") {
    PortfolioOutcome total;
    for (uint64_t seed = 5000; seed < 5003; ++seed) {
        for (const PortfolioOutcome o : {audit_portfolio(*linear_instance(seed), seed),
                                         audit_portfolio(*mip_instance(seed), seed)}) {
            total.adoptions += o.adoptions;
            total.restarts += o.restarts;
        }
    }
    // The audit is only evidence for the paths it actually reached.
    INFO("adoptions " << total.adoptions << ", restarts " << total.restarts);
    CHECK(total.adoptions > 0);
    CHECK(total.restarts > 0);
}

TEST_CASE("audit: FeasibilityJump::audit reports a corrupted cache", "[audit]") {
    // The audit is only evidence if it can fail: corrupt each kind of cached
    // state the search keeps and check it is named.
    std::unique_ptr<Instance> inst = mip_instance(4000);
    Model& m = inst->m;
    m.add_objective_soft_constraint();
    ViolationManager vm(m);
    RNG rng(7);
    FeasibilityJump fj(m, vm, rng);
    fj.begin(/*set_initial_x=*/true);
    fj.batch(1);  // one move: sampled entries stay valid, V stays non-empty
    REQUIRE(fj.audit().empty());

    SECTION("a node value") {
        const int32_t row = m.constraint_ids().front();
        const double was = m.node_value(row);
        m.set_node_value_unchecked(row, was + 1.0);
        CHECK_FALSE(fj.audit().empty());
        m.set_node_value_unchecked(row, was);
        CHECK(fj.audit().empty());
    }
    SECTION("a weight under a cached score") {
        bool scored = false;
        for (int32_t v = 0; v < static_cast<int32_t>(m.num_vars()); ++v) {
            const std::optional<JumpResult> j = fj.cached_jump(v);
            scored = scored || (j.has_value() && j->score != 0.0);
        }
        REQUIRE(scored);
        for (double& w : vm.weights) {
            w *= 3.0;
        }
        CHECK_FALSE(fj.audit().empty());
    }
    SECTION("a weight under V's counts") {
        const std::vector<int32_t> v = fj.violated_rows();
        REQUIRE_FALSE(v.empty());
        vm.weights[static_cast<size_t>(v.front())] = 0.0;
        CHECK_FALSE(fj.audit().empty());
    }
}
