#pragma once

#include "data.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

// Valve-point economic dispatch of one UC period, for the opt-in UC hook
// (#445, #455). A pure function of the instance and an on-set; not used by the
// published protocol.
namespace cbls::uc_chped {

inline constexpr double kDispatchInf = std::numeric_limits<double>::infinity();

/// min sum F_u(p_u) s.t. sum p_u = D and Pmin <= p_u <= Pmax over the on-set.
/// F is separable and non-convex, with local minima at the valve points
/// Pmin + k*pi/e. The fast path is lambda dispatch of the quadratic part,
/// snapped to valve points, rebalanced by whole valve steps and closed by one
/// swing unit; the polish is a pair-transfer descent (one unit onto a valve
/// point, one unit absorbing). Holds scratch buffers: one per thread.
class Dispatcher {
public:
    explicit Dispatcher(UCInstance&&) = delete;  // holds a reference
    explicit Dispatcher(const UCInstance& inst) : inst_(inst) {
        constexpr double kPi = 3.14159265358979323846;
        const int n = inst.n_units;
        valves_.resize(n);
        for (int u = 0; u < n; ++u) {
            auto& v = valves_[u];
            const double lo = inst.P_min[u];
            const double hi = inst.P_max[u];
            if (inst.e[u] > 0.0 && inst.d[u] != 0.0) {
                const double step = kPi / inst.e[u];
                for (int k = 0;; ++k) {
                    const double x = lo + (k * step);
                    if (x > hi + 1e-9) {
                        break;
                    }
                    v.push_back(std::min(x, hi));
                }
            } else {
                v.push_back(lo);
            }
            if (v.back() < hi - 1e-9) {
                v.push_back(hi);
            }
        }
    }

    [[nodiscard]] double fuel(int u, double p) const {
        const UCInstance& in = inst_;
        return in.a[u] + (in.b[u] * p) + (in.c[u] * p * p) +
               std::abs(in.d[u] * std::sin(in.e[u] * (in.P_min[u] - p)));
    }

    [[nodiscard]] const std::vector<double>& valves(int u) const { return valves_[u]; }

    /// Fast dispatch of `on` at `demand` into `p`, resized to n_units with every
    /// off unit at 0. Returns the on-set's fuel cost, or +inf (p all 0) when
    /// sum Pmin > demand or sum Pmax < demand.
    double fast(const std::vector<int>& on, double demand, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        p.assign(in.n_units, 0.0);
        double smin = 0.0;
        double smax = 0.0;
        for (int u : on) {
            smin += in.P_min[u];
            smax += in.P_max[u];
        }
        if (on.empty() || demand < smin - 1e-9 || demand > smax + 1e-9) {
            return kDispatchInf;
        }
        lambda_dispatch(on, demand, p);
        snap_and_balance(on, demand, p);
        return cost_of(on, p);
    }

    /// Pair-transfer descent from a dispatch of `on` (one fast() returned
    /// finite for), at most `max_sweeps` sweeps; keeps sum p and the bounds,
    /// never raises the cost.
    /// `window` > 0 tries only the valve points within `window` steps of p_i
    /// (the LS's cheap variant); 0 tries all of them. `jcap` > 0 tries about
    /// that many absorbing units per move, a stride through the on-set rotated
    /// by (i, k); 0 tries all of them.
    double polish(const std::vector<int>& on, std::vector<double>& p, int max_sweeps,
                  int window = 0, int jcap = 0) const {
        const UCInstance& in = inst_;
        const size_t stride =
            jcap > 0 ? std::max<size_t>(1, (on.size() + jcap - 1) / static_cast<size_t>(jcap)) : 1;
        fcache_.resize(in.n_units);
        for (int u : on) {
            fcache_[u] = fuel(u, p[u]);
        }
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            bool improved = false;
            for (int i : on) {
                improved = polish_unit(on, i, p, window, stride) || improved;
            }
            if (!improved) {
                break;
            }
        }
        return cost_of(on, p);
    }

    [[nodiscard]] double cost_of(const std::vector<int>& on, const std::vector<double>& p) const {
        double c = 0.0;
        for (int u : on) {
            c += fuel(u, p[u]);
        }
        return c;
    }

private:
    // The best single move of unit i onto one of its valve points, with one
    // other on unit absorbing the change; applied if it lowers the cost.
    bool polish_unit(const std::vector<int>& on, int i, std::vector<double>& p, int window,
                     size_t stride) const {
        const UCInstance& in = inst_;
        const double fi = fcache_[i];
        double best_gain = -1e-7;
        int best_j = -1;
        double best_q = 0.0;
        const auto& vi = valves_[i];
        size_t lo = 0;
        size_t hi = vi.size();
        if (window > 0) {
            const auto at =
                static_cast<size_t>(std::lower_bound(vi.begin(), vi.end(), p[i]) - vi.begin());
            // `window` points each side; p_i on vi[at] is itself skipped below.
            const bool on_valve = at < vi.size() && std::abs(vi[at] - p[i]) < 1e-9;
            lo = at > static_cast<size_t>(window) ? at - window : 0;
            hi = std::min(vi.size(), at + window + (on_valve ? 1 : 0));
        }
        for (size_t k = lo; k < hi; ++k) {
            const double q = vi[k];
            const double delta = q - p[i];
            if (std::abs(delta) < 1e-9) {
                continue;
            }
            const double di = fuel(i, q) - fi;
            for (size_t jj = (static_cast<size_t>(i) + k) % stride; jj < on.size(); jj += stride) {
                const int j = on[jj];
                if (j == i) {
                    continue;
                }
                const double pj = p[j] - delta;
                if (pj < in.P_min[j] - 1e-12 || pj > in.P_max[j] + 1e-12) {
                    continue;
                }
                const double g = di + fuel(j, pj) - fcache_[j];
                if (g < best_gain) {
                    best_gain = g;
                    best_j = j;
                    best_q = q;
                }
            }
        }
        if (best_j < 0) {
            return false;
        }
        const double delta = best_q - p[i];
        p[i] = best_q;
        p[best_j] = std::clamp(p[best_j] - delta, in.P_min[best_j], in.P_max[best_j]);
        fcache_[i] = fuel(i, p[i]);
        fcache_[best_j] = fuel(best_j, p[best_j]);
        return true;
    }

    // Quadratic-part optimum: p_u = clamp((lambda - b)/(2c)), bisection on lambda.
    void lambda_dispatch(const std::vector<int>& on, double demand, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        auto at = [&](int u, double lam) {
            if (in.c[u] <= 0.0) {
                return lam >= in.b[u] ? in.P_max[u] : in.P_min[u];
            }
            return std::clamp((lam - in.b[u]) / (2.0 * in.c[u]), in.P_min[u], in.P_max[u]);
        };
        double lo = kDispatchInf;
        double hi = -kDispatchInf;
        for (int u : on) {
            lo = std::min(lo, in.b[u] + (2.0 * in.c[u] * in.P_min[u]));
            hi = std::max(hi, in.b[u] + (2.0 * in.c[u] * in.P_max[u]));
        }
        lo -= 1.0;
        hi += 1.0;
        for (int it = 0; it < 60; ++it) {
            const double mid = 0.5 * (lo + hi);
            double s = 0.0;
            for (int u : on) {
                s += at(u, mid);
            }
            (s < demand ? lo : hi) = mid;
        }
        for (int u : on) {
            p[u] = at(u, hi);
        }
    }

    // Each on unit to its nearest valve point (index into idx_); returns sum p.
    double snap_to_valves(const std::vector<int>& on, std::vector<double>& p) const {
        idx_.assign(inst_.n_units, 0);
        double sum = 0.0;
        for (int u : on) {
            const auto& v = valves_[u];
            size_t best = 0;
            for (size_t k = 1; k < v.size(); ++k) {
                if (std::abs(v[k] - p[u]) < std::abs(v[best] - p[u])) {
                    best = k;
                }
            }
            idx_[u] = static_cast<int>(best);
            p[u] = v[best];
            sum += p[u];
        }
        return sum;
    }

    // Snap to valve points, then move whole valve steps until the residual is
    // under a step, then close it. Each step picks the lowest fuel change per
    // MW moved: the cheapest unit to raise, or the unit that saves the most to
    // lower (#455, review Medium 5).
    void snap_and_balance(const std::vector<int>& on, double demand, std::vector<double>& p) const {
        double r = demand - snap_to_valves(on, p);
        for (;;) {
            const bool up = r > 0.0;
            int bu = -1;
            double bscore = kDispatchInf;
            double bstep = 0.0;
            for (int u : on) {
                const auto& v = valves_[u];
                const int k = idx_[u] + (up ? 1 : -1);
                if (k < 0 || k >= static_cast<int>(v.size())) {
                    continue;
                }
                const double step = std::abs(v[k] - p[u]);
                if (step > std::abs(r) || step <= 0.0) {
                    continue;
                }
                const double score = (fuel(u, v[k]) - fuel(u, p[u])) / step;
                if (score < bscore) {
                    bscore = score;
                    bu = u;
                    bstep = step;
                }
            }
            if (bu < 0) {
                break;
            }
            idx_[bu] += up ? 1 : -1;
            p[bu] = valves_[bu][idx_[bu]];
            r += up ? -bstep : bstep;
        }
        close_residual(on, r, p);
    }

    // The swing: the one unit whose cost rises least on absorbing r; if none
    // can absorb all of it, move the unit with the most room to its bound and
    // repeat. Terminates: each fill zeroes one unit's room. fast() checked
    // sum Pmin <= demand <= sum Pmax, so the fills close r.
    void close_residual(const std::vector<int>& on, double r, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        while (std::abs(r) > 1e-10) {
            int bu = -1;
            double bd = kDispatchInf;
            for (int u : on) {
                const double q = p[u] + r;
                if (q < in.P_min[u] || q > in.P_max[u]) {
                    continue;
                }
                const double d = fuel(u, q) - fuel(u, p[u]);
                if (d < bd) {
                    bd = d;
                    bu = u;
                }
            }
            if (bu >= 0) {
                p[bu] += r;
                return;
            }
            int mu = -1;
            double room = 0.0;
            for (int u : on) {
                const double rr = r > 0.0 ? in.P_max[u] - p[u] : p[u] - in.P_min[u];
                if (rr > room) {
                    room = rr;
                    mu = u;
                }
            }
            if (mu < 0) {
                return;
            }
            const double mv = r > 0.0 ? room : -room;
            p[mu] += mv;
            r -= mv;
        }
    }

    const UCInstance& inst_;
    std::vector<std::vector<double>> valves_;
    mutable std::vector<int> idx_;
    mutable std::vector<double> fcache_;
};

}  // namespace cbls::uc_chped
