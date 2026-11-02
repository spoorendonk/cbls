#pragma once

// Shared by the presolve tests (tests/test_presolve*.cpp): a problem builder
// and a feasibility check with the 1e-9 scaled tolerance they all use.

#include <algorithm>
#include <cbls/presolve.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace cbls::presolve_test {

inline constexpr double kInf = std::numeric_limits<double>::infinity();

struct Builder {
    LinearProblem p;

    int32_t col(double lb, double ub, bool integral = false, double obj = 0.0,
                bool frozen = false) {
        p.col_lb.push_back(lb);
        p.col_ub.push_back(ub);
        p.integral.push_back(integral ? 1 : 0);
        p.frozen.push_back(frozen ? 1 : 0);
        p.obj.push_back(obj);
        return static_cast<int32_t>(p.col_lb.size() - 1);
    }

    void row(const std::vector<std::pair<int32_t, double>>& terms, double lo, double hi) {
        for (const auto& [c, a] : terms) {
            p.cols.push_back(c);
            p.coefs.push_back(a);
        }
        p.row_start.push_back(p.cols.size());
        p.row_lo.push_back(lo);
        p.row_hi.push_back(hi);
    }
};

inline PresolveOptions only(bool propagate) {
    PresolveOptions o;
    o.enabled = true;
    o.propagate_bounds = propagate;
    return o;
}

inline double tol_of(double scale) {
    return 1e-9 * std::max(1.0, std::abs(scale));
}

/// x satisfies p's bounds, integrality and rows within 1e-9 (scaled).
inline bool satisfies(const LinearProblem& p, const std::vector<double>& x) {
    if (x.size() != p.n_cols()) {
        return false;
    }
    for (std::size_t j = 0; j < p.n_cols(); ++j) {
        if (!std::isfinite(x[j]) || x[j] < p.col_lb[j] - tol_of(x[j]) ||
            x[j] > p.col_ub[j] + tol_of(x[j])) {
            return false;
        }
        if (p.integral[j] != 0 && std::abs(x[j] - std::round(x[j])) > 1e-9) {
            return false;
        }
    }
    for (std::size_t i = 0; i < p.n_rows(); ++i) {
        double act = 0.0;
        double mag = 0.0;
        for (std::size_t k = p.row_start[i]; k < p.row_start[i + 1]; ++k) {
            const double t = p.coefs[k] * x[static_cast<std::size_t>(p.cols[k])];
            act += t;
            mag += std::abs(t);
        }
        const double lo = p.row_lo[i];
        const double hi = p.row_hi[i];
        const double scale = mag + (std::isfinite(lo) ? std::abs(lo) : 0.0) +
                             (std::isfinite(hi) ? std::abs(hi) : 0.0);
        if (act < lo - tol_of(scale) || act > hi + tol_of(scale)) {
            return false;
        }
    }
    return true;
}

inline double objective(const LinearProblem& p, const std::vector<double>& x) {
    double v = p.obj_offset;
    for (std::size_t j = 0; j < p.n_cols(); ++j) {
        v += p.obj[j] * x[j];
    }
    return v;
}

}  // namespace cbls::presolve_test
