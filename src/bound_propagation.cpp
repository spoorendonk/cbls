// Activity-based bound propagation. See include/cbls/bound_propagation.h.

#include "cbls/bound_propagation.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cbls {

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr double kPosInf = std::numeric_limits<double>::infinity();

/// Bounds are read through these, so that a sentinel magnitude (`1e20` and up,
/// the MPS/CPLEX/SCIP convention) behaves like the infinity it stands for.
/// Both map *outward* only — never turning a real bound into a tighter one.
double eff_lo(double b, double infinity) {
    return b <= -infinity ? kNegInf : b;
}
double eff_hi(double b, double infinity) {
    return b >= infinity ? kPosInf : b;
}

/// Running min or max activity of a row, kept as the sum of its finite terms
/// plus a count of the infinite ones. The count is what lets a row tighten the
/// single column that made it infinite: with exactly one infinite contribution,
/// the activity of *everything else* is still finite and still bounds it.
struct Activity {
    double sum = 0.0;
    int n_infinite = 0;

    void add(double contribution) {
        if (std::isfinite(contribution)) {
            sum += contribution;
        } else {
            ++n_infinite;
        }
    }

    /// The activity of the row excluding one term, or `unbounded` when two or
    /// more terms are infinite in that direction.
    [[nodiscard]] double without(double contribution, double unbounded) const {
        if (std::isfinite(contribution)) {
            return n_infinite == 0 ? sum - contribution : unbounded;
        }
        return n_infinite == 1 ? sum : unbounded;
    }
};

/// The term's contribution to the row's minimum activity (and its maximum),
/// given the term's effective column bounds.
double min_contribution(double coef, double lo, double hi) {
    return coef > 0.0 ? coef * lo : coef * hi;
}
double max_contribution(double coef, double lo, double hi) {
    return coef > 0.0 ? coef * hi : coef * lo;
}

double relaxation(double b, const BoundPropagationOptions& opts) {
    return std::max(opts.safety_absolute, opts.safety_relative * std::abs(b));
}

/// True if `candidate` is a materially tighter upper bound than `incumbent`.
/// Lower bounds are compared through the same function with both signs flipped.
bool improves(double candidate, double incumbent, const BoundPropagationOptions& opts) {
    if (!std::isfinite(incumbent)) {
        return incumbent > 0.0;  // +inf: any finite candidate improves it.
    }
    const double threshold = std::max(opts.min_absolute_improvement,
                                      opts.min_relative_improvement * std::abs(incumbent));
    return candidate < incumbent - threshold;
}

void validate(const std::vector<LinearRow>& rows, const std::vector<double>& lb,
              const std::vector<double>& ub) {
    const std::size_t n_cols = lb.size();
    for (std::size_t j = 0; j < n_cols; ++j) {
        if (std::isnan(lb[j]) || std::isnan(ub[j])) {
            throw std::invalid_argument("propagate_bounds: column " + std::to_string(j) +
                                        " has a NaN bound");
        }
    }
    for (const LinearRow& row : rows) {
        if (row.cols.size() != row.coefs.size()) {
            throw std::invalid_argument("propagate_bounds: LinearRow has " +
                                        std::to_string(row.cols.size()) + " columns but " +
                                        std::to_string(row.coefs.size()) + " coefficients");
        }
        for (int32_t col : row.cols) {
            if (col < 0 || static_cast<std::size_t>(col) >= n_cols) {
                throw std::invalid_argument("propagate_bounds: LinearRow references column " +
                                            std::to_string(col) + " outside [0, " +
                                            std::to_string(n_cols) + ")");
            }
        }
    }
}

/// State threaded through one sweep, so the two bound directions share a single
/// implementation instead of two near-copies that can drift apart.
struct Tightener {
    std::vector<double>& lb;
    std::vector<double>& ub;
    const std::vector<uint8_t>& integral;
    const BoundPropagationOptions& opts;
    bool changed = false;
    bool infeasible = false;

    /// Apply `x[col] <= value` (or `>=` when `upper` is false), relaxed outward
    /// and then rounded inward on an integral column.
    void apply(int32_t col, double value, bool upper) {
        if (!std::isfinite(value) || std::abs(value) >= opts.infinity) {
            return;
        }
        const double eps = relaxation(value, opts);
        value = upper ? value + eps : value - eps;
        const std::size_t j = static_cast<std::size_t>(col);
        if (integral[j] != 0) {
            value = upper ? std::floor(value + eps) : std::ceil(value - eps);
        }
        if (upper) {
            if (!improves(value, eff_hi(ub[j], opts.infinity), opts)) {
                return;
            }
            ub[j] = value;
        } else {
            if (!improves(-value, -eff_lo(lb[j], opts.infinity), opts)) {
                return;
            }
            lb[j] = value;
        }
        changed = true;
        if (lb[j] > ub[j] + relaxation(lb[j], opts)) {
            infeasible = true;
        }
    }

    /// Derive both bounds on one term of a row from the activity of the rest.
    void tighten_term(double coef, int32_t col, double row_lo, double row_hi, double min_rest,
                      double max_rest) {
        // coef * x <= row_hi - min_rest  and  coef * x >= row_lo - max_rest.
        const double upper_slack =
            (std::isfinite(row_hi) && std::isfinite(min_rest)) ? row_hi - min_rest : kPosInf;
        const double lower_slack =
            (std::isfinite(row_lo) && std::isfinite(max_rest)) ? row_lo - max_rest : kNegInf;
        if (coef > 0.0) {
            apply(col, upper_slack / coef, /*upper=*/true);
            apply(col, lower_slack / coef, /*upper=*/false);
        } else {
            apply(col, upper_slack / coef, /*upper=*/false);
            apply(col, lower_slack / coef, /*upper=*/true);
        }
    }

    /// One row against the current bounds, updating them in place.
    void tighten_row(const LinearRow& row) {
        const double infinity = opts.infinity;
        const std::size_t nnz = row.cols.size();
        Activity min_act;
        Activity max_act;
        for (std::size_t k = 0; k < nnz; ++k) {
            const double a = row.coefs[k];
            if (a == 0.0) {
                continue;
            }
            const std::size_t j = static_cast<std::size_t>(row.cols[k]);
            const double lo = eff_lo(lb[j], infinity);
            const double hi = eff_hi(ub[j], infinity);
            min_act.add(min_contribution(a, lo, hi));
            max_act.add(max_contribution(a, lo, hi));
        }
        const double row_lo = eff_lo(row.lo, infinity);
        const double row_hi = eff_hi(row.hi, infinity);
        for (std::size_t k = 0; k < nnz; ++k) {
            const double a = row.coefs[k];
            if (a == 0.0) {
                continue;
            }
            const int32_t col = row.cols[k];
            const std::size_t j = static_cast<std::size_t>(col);
            // Contributions are recomputed here rather than cached: an earlier
            // term of this same row may have moved a bound this one reads, if a
            // column occurs in the row more than once.
            const double lo = eff_lo(lb[j], infinity);
            const double hi = eff_hi(ub[j], infinity);
            const double min_rest = min_act.without(min_contribution(a, lo, hi), kNegInf);
            const double max_rest = max_act.without(max_contribution(a, lo, hi), kPosInf);
            tighten_term(a, col, row_lo, row_hi, min_rest, max_rest);
            if (infeasible) {
                return;
            }
        }
    }
};

void record_stats(const std::vector<double>& lb, const std::vector<double>& ub,
                  const std::vector<double>& lb0, const std::vector<double>& ub0,
                  const BoundPropagationOptions& opts, BoundPropagationStats& stats) {
    const double infinity = opts.infinity;
    for (std::size_t j = 0; j < lb.size(); ++j) {
        const bool moved = lb[j] != lb0[j] || ub[j] != ub0[j];
        if (moved) {
            ++stats.n_tightened;
        }
        const bool was_infinite =
            !std::isfinite(eff_lo(lb0[j], infinity)) || !std::isfinite(eff_hi(ub0[j], infinity));
        const bool is_infinite =
            !std::isfinite(eff_lo(lb[j], infinity)) || !std::isfinite(eff_hi(ub[j], infinity));
        if (was_infinite && !is_infinite) {
            ++stats.n_finitized;
        }
        if (moved && lb[j] == ub[j] && lb0[j] != ub0[j]) {
            ++stats.n_fixed;
        }
    }
}

}  // namespace

BoundPropagationStats propagate_bounds(const std::vector<LinearRow>& rows,
                                       const std::vector<uint8_t>& integral,
                                       std::vector<double>& lb, std::vector<double>& ub,
                                       const BoundPropagationOptions& opts) {
    const std::size_t n_cols = lb.size();
    if (ub.size() != n_cols || integral.size() != n_cols) {
        throw std::invalid_argument("propagate_bounds: lb, ub and integral must be the same size");
    }
    validate(rows, lb, ub);

    BoundPropagationStats stats;
    if (opts.max_passes <= 0 || rows.empty() || n_cols == 0) {
        return stats;
    }

    const std::vector<double> lb0 = lb;
    const std::vector<double> ub0 = ub;

    Tightener t{lb, ub, integral, opts};
    for (int pass = 0; pass < opts.max_passes; ++pass) {
        t.changed = false;
        ++stats.passes;
        for (const LinearRow& row : rows) {
            // A row bounded on neither side implies nothing about any column.
            if (!std::isfinite(eff_lo(row.lo, opts.infinity)) &&
                !std::isfinite(eff_hi(row.hi, opts.infinity))) {
                continue;
            }
            t.tighten_row(row);
            if (t.infeasible) {
                stats.infeasible = true;
                break;
            }
        }
        if (stats.infeasible || !t.changed) {
            break;
        }
        stats.hit_pass_limit = (pass + 1 == opts.max_passes);
    }

    record_stats(lb, ub, lb0, ub0, opts, stats);
    return stats;
}

}  // namespace cbls
