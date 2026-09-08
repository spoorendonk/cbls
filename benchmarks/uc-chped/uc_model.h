#pragma once

#include "data.h"

#include <algorithm>
#include <cbls/cbls.h>
#include <vector>

namespace cbls::uc_chped {

struct UCModel {
    Model model;
    std::vector<std::vector<int32_t>> y;  // [unit][period] commitment (bool var handles)
    std::vector<std::vector<int32_t>> p;  // [unit][period] dispatch (float var handles)
};

// The model builder is split along the parts of the formulation, not into
// arbitrary halves: the variables, the constant pool the expressions read, the
// objective's two cost terms, and the two constraint groups (per-period market
// constraints, per-unit operating constraints). The order in which nodes and
// constraints are created is part of the model -- constraint indices carry the
// GLS weights and the search trajectory follows them -- so the helpers are
// called in exactly the order the single body used to run in.
namespace model_detail {

/// Literal nodes the expressions reuse, created once per model.
struct Literals {
    int32_t zero;
    int32_t neg1;
    int32_t neg_half;
    int32_t one;
    int32_t two;
};

/// Per-unit constant nodes, created once and reused across periods.
struct UnitConstants {
    std::vector<int32_t> a, b, c, d, e;
    std::vector<int32_t> pmin, pmax;
    std::vector<int32_t> a_hot, a_cold;
    std::vector<int32_t> y_prev;
};

inline Literals make_literals(Model& m) {
    Literals k{};
    k.zero = m.constant(0.0);
    k.neg1 = m.constant(-1.0);
    k.neg_half = m.constant(-0.5);
    k.one = m.constant(1.0);
    k.two = m.constant(2.0);
    return k;
}

inline UnitConstants make_unit_constants(Model& m, const UCInstance& inst) {
    UnitConstants k;
    const int n = inst.n_units;
    k.a.resize(n);
    k.b.resize(n);
    k.c.resize(n);
    k.d.resize(n);
    k.e.resize(n);
    k.pmin.resize(n);
    k.pmax.resize(n);
    k.a_hot.resize(n);
    k.a_cold.resize(n);
    k.y_prev.resize(n);
    for (int u = 0; u < n; ++u) {
        k.a[u] = m.constant(inst.a[u]);
        k.b[u] = m.constant(inst.b[u]);
        k.c[u] = m.constant(inst.c[u]);
        k.d[u] = m.constant(inst.d[u]);
        k.e[u] = m.constant(inst.e[u]);
        k.pmin[u] = m.constant(inst.P_min[u]);
        k.pmax[u] = m.constant(inst.P_max[u]);
        k.a_hot[u] = m.constant(inst.a_hot[u]);
        k.a_cold[u] = m.constant(inst.a_cold[u]);
        k.y_prev[u] = m.constant(static_cast<double>(inst.y_prev[u]));
    }
    return k;
}

inline void make_variables(UCModel& result, const UCInstance& inst) {
    Model& m = result.model;
    result.y.resize(inst.n_units);
    result.p.resize(inst.n_units);
    for (int u = 0; u < inst.n_units; ++u) {
        result.y[u].resize(inst.n_periods);
        result.p[u].resize(inst.n_periods);
        for (int t = 0; t < inst.n_periods; ++t) {
            const std::string suffix = std::to_string(u) + "_" + std::to_string(t);
            result.y[u][t] = m.bool_var("y_" + suffix);
            // Bounds [0, Pmax]: when y=0, dispatch should be 0; Pmin enforced via constraint
            result.p[u][t] = m.float_var(0.0, inst.P_max[u], "p_" + suffix);
        }
    }
}

/// The commitment one period before t, as an expression handle: the unit's
/// y_prev constant at the start of the horizon, the previous period's variable
/// after it.
inline int32_t commitment_before(const UCModel& result, const UnitConstants& k, int u, int t) {
    return (t == 0) ? k.y_prev[u] : result.y[u][t - 1];
}

/// Fuel cost: y * (a + b*p + c*p^2 + |d*sin(e*(Pmin-p))|).
inline int32_t fuel_cost(Model& m, const Literals& lit, const UnitConstants& k, int32_t y_h,
                         int32_t p_h, int u) {
    auto base_cost = m.sum({k.a[u], m.prod(k.b[u], p_h), m.prod(k.c[u], m.pow_expr(p_h, lit.two))});
    auto pmin_minus_p = m.sum({k.pmin[u], m.prod(lit.neg1, p_h)});
    auto valve_point = m.abs_expr(m.prod(k.d[u], m.sin_expr(m.prod(k.e[u], pmin_minus_p))));
    return m.prod(y_h, m.sum({base_cost, valve_point}));
}

/// Startup cost: max(0, y[t] - y[t-1]) priced hot if the unit ran anywhere in
/// the t_cold periods before t, cold otherwise. With no lookback window the
/// startup is always cold.
inline int32_t startup_cost(Model& m, const UCModel& result, const Literals& lit,
                            const UnitConstants& k, const UCInstance& inst, int u, int t) {
    auto su = m.max_expr(
        {lit.zero, m.sum({result.y[u][t], m.prod(lit.neg1, commitment_before(result, k, u, t))})});

    // was_on = max(y[tau] for tau in the lookback window)
    int lookback_start = (t == 0) ? -inst.t_cold[u] : t - inst.t_cold[u];
    std::vector<int32_t> lookback_ys;
    for (int tau = lookback_start; tau < t; ++tau) {
        if (tau >= 0) {
            lookback_ys.push_back(result.y[u][tau]);
        } else if (lookback_ys.empty() || lookback_ys.back() != k.y_prev[u]) {
            // Before the horizon: reuse the single y_prev constant for this unit
            lookback_ys.push_back(k.y_prev[u]);
        }
    }

    if (lookback_ys.empty()) {
        return m.prod(k.a_cold[u], su);
    }
    // if_then_else(cond, then, else): cond > 0 -> then, else -> else.
    // We want: was_on > 0.5 -> hot, else -> cold.
    auto was_on = m.max_expr(lookback_ys);
    auto cond = m.sum({was_on, lit.neg_half});
    return m.if_then_else(cond, m.prod(k.a_hot[u], su), m.prod(k.a_cold[u], su));
}

inline void set_objective(UCModel& result, const Literals& lit, const UnitConstants& k,
                          const UCInstance& inst) {
    Model& m = result.model;
    std::vector<int32_t> cost_terms;
    for (int u = 0; u < inst.n_units; ++u) {
        for (int t = 0; t < inst.n_periods; ++t) {
            cost_terms.push_back(fuel_cost(m, lit, k, result.y[u][t], result.p[u][t], u));
            cost_terms.push_back(startup_cost(m, result, lit, k, inst, u, t));
        }
    }
    m.minimize(m.sum(cost_terms));
}

/// Per-period market constraints: demand must be met, and where a reserve margin
/// is given the committed capacity must cover demand + reserve.
inline void add_market_constraints(UCModel& result, const Literals& lit, const UnitConstants& k,
                                   const UCInstance& inst) {
    Model& m = result.model;
    for (int t = 0; t < inst.n_periods; ++t) {
        // demand[t] - sum(p[u][t]) <= 0
        std::vector<int32_t> supply_terms;
        supply_terms.reserve(static_cast<size_t>(inst.n_units));
        for (int u = 0; u < inst.n_units; ++u) {
            supply_terms.push_back(result.p[u][t]);
        }
        // Node creation order is preserved exactly as it was when this lived in
        // one body: node ids drive the topological order the DAG evaluates in.
        auto supply = m.sum(supply_terms);
        auto demand_t = m.constant(inst.demand[t]);
        m.add_constraint(m.sum({demand_t, m.prod(lit.neg1, supply)}));

        // (demand[t] + reserve[t]) - sum(Pmax[u]*y[u][t]) <= 0
        if (inst.reserve[t] > 0) {
            std::vector<int32_t> cap_terms;
            cap_terms.reserve(static_cast<size_t>(inst.n_units));
            for (int u = 0; u < inst.n_units; ++u) {
                cap_terms.push_back(m.prod(k.pmax[u], result.y[u][t]));
            }
            auto capacity = m.sum(cap_terms);
            auto reserve_t = m.constant(inst.reserve[t]);
            m.add_constraint(m.sum({demand_t, reserve_t, m.prod(lit.neg1, capacity)}));
        }
    }
}

/// Dispatch is zero unless committed and inside [Pmin, Pmax] when it is:
/// Pmin*y - p <= 0 and p - Pmax*y <= 0.
inline void add_dispatch_bounds(UCModel& result, const Literals& lit, const UnitConstants& k,
                                const UCInstance& inst, int u) {
    Model& m = result.model;
    for (int t = 0; t < inst.n_periods; ++t) {
        auto y_h = result.y[u][t];
        auto p_h = result.p[u][t];
        m.add_constraint(m.sum({m.prod(k.pmin[u], y_h), m.prod(lit.neg1, p_h)}));
        m.add_constraint(m.sum({p_h, m.prod(lit.neg1, m.prod(k.pmax[u], y_h))}));
    }
}

/// Minimum uptime. A unit that starts up at t (y[t]=1, y[t-1]=0) must stay on
/// for min_on periods, posted pairwise as y[t] - y[t-1] - y[tau] <= 0 for each
/// tau in [t+1, min(t+min_on-1, T-1)].
inline void add_min_uptime(UCModel& result, const Literals& lit, const UnitConstants& k,
                           const UCInstance& inst, int u) {
    Model& m = result.model;
    for (int t = 0; t < inst.n_periods; ++t) {
        int32_t y_prev_h = commitment_before(result, k, u, t);
        int end = std::min(t + inst.min_on[u], inst.n_periods);
        for (int tau = t + 1; tau < end; ++tau) {
            m.add_constraint(m.sum(
                {result.y[u][t], m.prod(lit.neg1, y_prev_h), m.prod(lit.neg1, result.y[u][tau])}));
        }
    }
}

/// Minimum downtime. A unit that shuts down at t (y[t]=0, y[t-1]=1) must stay
/// off for min_off periods, posted pairwise as y[t-1] - y[t] + y[tau] - 1 <= 0.
inline void add_min_downtime(UCModel& result, const Literals& lit, const UnitConstants& k,
                             const UCInstance& inst, int u) {
    Model& m = result.model;
    for (int t = 0; t < inst.n_periods; ++t) {
        int32_t y_prev_h = commitment_before(result, k, u, t);
        int end = std::min(t + inst.min_off[u], inst.n_periods);
        for (int tau = t + 1; tau < end; ++tau) {
            m.add_constraint(
                m.sum({y_prev_h, m.prod(lit.neg1, result.y[u][t]), result.y[u][tau], lit.neg1}));
        }
    }
}

/// Initial conditions: a unit that entered the horizon mid-run is pinned for the
/// remainder of its minimum on- or off-time.
inline void add_initial_conditions(UCModel& result, const Literals& lit, const UCInstance& inst,
                                   int u) {
    Model& m = result.model;
    if (inst.y_prev[u] == 1) {
        int remaining_on = std::max(0, inst.min_on[u] - inst.n_init[u]);
        for (int t = 0; t < std::min(remaining_on, inst.n_periods); ++t) {
            // y[t] >= 1  ->  1 - y[t] <= 0
            m.add_constraint(m.sum({lit.one, m.prod(lit.neg1, result.y[u][t])}));
        }
    }
    if (inst.y_prev[u] == 0) {
        int remaining_off = std::max(0, inst.min_off[u] - inst.n_init[u]);
        for (int t = 0; t < std::min(remaining_off, inst.n_periods); ++t) {
            // y[t] <= 0
            m.add_constraint(result.y[u][t]);
        }
    }
}

}  // namespace model_detail

inline UCModel build_uc_model(const UCInstance& inst) {
    UCModel result;
    Model& m = result.model;

    model_detail::make_variables(result, inst);
    const model_detail::Literals lit = model_detail::make_literals(m);
    const model_detail::UnitConstants k = model_detail::make_unit_constants(m, inst);

    model_detail::set_objective(result, lit, k, inst);

    model_detail::add_market_constraints(result, lit, k, inst);
    for (int u = 0; u < inst.n_units; ++u) {
        model_detail::add_dispatch_bounds(result, lit, k, inst, u);
        model_detail::add_min_uptime(result, lit, k, inst, u);
        model_detail::add_min_downtime(result, lit, k, inst, u);
        model_detail::add_initial_conditions(result, lit, inst, u);
    }

    // Register commitment var sequences for block moves
    for (int u = 0; u < inst.n_units; ++u) {
        m.add_var_sequence(result.y[u], inst.min_on[u], inst.min_off[u]);
    }

    m.close();
    return result;
}

}  // namespace cbls::uc_chped
