#pragma once

// Small models and probes that several test files share (#374). Each was a
// verbatim copy in every file that used it.

#include <algorithm>
#include <cbls/dag_ops.h>
#include <cbls/model.h>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

namespace cbls_test {

// x + y >= 1 over [-5, 5]^2, minimise x^2 + y^2: an objective model, so the
// search keeps tightening its bound and never exits on `Feasible`.
inline cbls::Model quadratic_model() {
    cbls::Model m;
    auto x = m.float_var(-5, 5);
    auto y = m.float_var(-5, 5);
    auto two = m.constant(2);
    auto neg1 = m.constant(-1.0);
    auto one = m.constant(1.0);
    m.add_constraint(m.sum({one, m.prod(neg1, x), m.prod(neg1, y)}));  // x + y >= 1
    m.minimize(m.sum({m.pow_expr(x, two), m.pow_expr(y, two)}));
    m.close();
    return m;
}

// A 20-column integer model: reaching the target is easy, but the optimum is
// the balanced assignment, so the search keeps finding new incumbents for a
// long time (which fills a shared pool) and plateaus (which makes a
// diversification kick fire). The quadratic model above improves on nearly
// every batch and takes almost no kicks.
inline cbls::Model integer_model() {
    cbls::Model m;
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

// The trajectory fence's random over-constrained integer model: rows that
// cannot all hold, so FJ and Novelty both reach local minima, backtrack and
// bump, and propagation legs narrow domains with some fixes conflicting. With
// `nonlinear`, every fifth row also multiplies two of its variables (no extra
// draws, so the linear part is the same model). Returns the variable handles;
// the model is left open.
inline std::vector<int32_t> build_random_int_model(cbls::Model& m, uint32_t seed, int num_vars,
                                                   int num_rows, bool nonlinear = false) {
    std::mt19937 gen(seed);
    std::uniform_int_distribution<int> pick_var(0, num_vars - 1);
    std::uniform_int_distribution<int> pick_coef(1, 3);
    std::uniform_int_distribution<int> pick_sign(0, 1);
    std::uniform_int_distribution<int> pick_rhs(-2, 6);
    std::vector<int32_t> handles;
    handles.reserve(static_cast<size_t>(num_vars));
    for (int i = 0; i < num_vars; ++i) {
        handles.push_back(m.int_var(0, 4));
    }
    for (int r = 0; r < num_rows; ++r) {
        std::vector<int> used;
        while (used.size() < 4) {
            const int v = pick_var(gen);
            if (std::find(used.begin(), used.end(), v) == used.end()) {
                used.push_back(v);
            }
        }
        std::vector<int32_t> terms;
        for (const int v : used) {
            const double coef = (pick_sign(gen) != 0 ? 1.0 : -1.0) * pick_coef(gen);
            terms.push_back(m.prod(m.constant(coef), handles[static_cast<size_t>(v)]));
        }
        if (nonlinear && r % 5 == 0) {
            terms.push_back(m.prod(handles[static_cast<size_t>(used[0])],
                                   handles[static_cast<size_t>(used[1])]));
        }
        const int32_t sum = m.sum(terms);
        const double rhs = pick_rhs(gen);
        m.add_constraint(r % 2 == 0 ? m.leq(sum, m.constant(rhs)) : m.geq(sum, m.constant(rhs)));
    }
    return handles;
}

// The bit pattern of a double, for bit-for-bit comparisons.
inline uint64_t bits_of(double d) {
    uint64_t u = 0;
    std::memcpy(&u, &d, sizeof u);
    return u;
}

// Fully evaluate `m` and read one node's value.
inline double eval_at(cbls::Model& m, int32_t node) {
    cbls::full_evaluate(m);
    return m.node_value(node);
}

}  // namespace cbls_test
