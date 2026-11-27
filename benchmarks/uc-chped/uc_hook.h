#pragma once

#include "data.h"
#include "uc_model.h"

#include <algorithm>
#include <atomic>
#include <cbls/cbls.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <unordered_map>
#include <vector>

// Opt-in UC inner solver for the valve-point UC (`--uc-hook`, #445). Plugged in
// as cbls's InnerSolverHook: on each feasible point the search reaches, it
// re-dispatches every period (valve-point economic dispatch) and runs a
// commitment local search scored by that dispatch, then hands the point back
// to the search. Off by default; the published protocol never runs it.
namespace cbls::uc_chped {

namespace hook_detail {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kPi = 3.14159265358979323846;

/// One period's on-set, packed: the memo key of a dispatch.
struct ColumnKey {
    int t = 0;
    std::vector<uint64_t> words;
    bool operator==(const ColumnKey& o) const { return t == o.t && words == o.words; }
};

struct ColumnKeyHash {
    size_t operator()(const ColumnKey& k) const {
        uint64_t h = 1469598103934665603ULL ^ static_cast<uint64_t>(k.t);
        for (uint64_t w : k.words) {
            h ^= w + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
            h *= 1099511628211ULL;
        }
        return static_cast<size_t>(h);
    }
};

/// A period's dispatch: cost of the committed units' fuel, and p for every unit
/// (0 for an off unit). cost is +inf when the on-set cannot meet demand.
struct Dispatch {
    double cost = kInf;
    std::vector<double> p;
};

/// Valve-point economic dispatch of one period: min sum F_u(p_u) subject to
/// sum p_u = D and Pmin <= p_u <= Pmax over the on-set. F is separable and
/// non-convex, with its local minima at the valve points Pmin + k*pi/e. The
/// fast path is lambda dispatch of the quadratic part, snapped to valve points,
/// rebalanced by whole valve steps and closed by one swing unit; the polish is
/// a pair-transfer descent (one unit onto a valve point, one unit absorbing).
class Dispatcher {
public:
    explicit Dispatcher(const UCInstance& inst) : inst_(inst) {
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

    /// Fast dispatch of `on` at demand `demand`, written into `p` (dense).
    double fast(const std::vector<int>& on, double demand, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        double smin = 0.0;
        double smax = 0.0;
        for (int u : on) {
            smin += in.P_min[u];
            smax += in.P_max[u];
        }
        if (on.empty() || demand < smin - 1e-9 || demand > smax + 1e-9) {
            return kInf;
        }
        lambda_dispatch(on, demand, p);
        snap_and_balance(on, demand, p);
        return cost_of(on, p);
    }

    /// Pair-transfer descent from a feasible `p`, at most `max_sweeps` sweeps.
    // `window` > 0 tries only the valve points within `window` steps of p_i
    // (the LS's cheap variant); 0 tries all of them.
    // `jcap` > 0 tries about that many absorbing units per move, a stride
    // through the on-set rotated by (i, k); 0 tries all of them.
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
                const double fi = fcache_[i];
                double best_gain = -1e-7;
                int best_j = -1;
                double best_q = 0.0;
                const auto& vi = valves_[i];
                size_t lo = 0;
                size_t hi = vi.size();
                if (window > 0) {
                    const size_t at = static_cast<size_t>(
                        std::lower_bound(vi.begin(), vi.end(), p[i]) - vi.begin());
                    lo = at > static_cast<size_t>(window) ? at - window : 0;
                    hi = std::min(vi.size(), at + window);
                }
                for (size_t k = lo; k < hi; ++k) {
                    const double q = vi[k];
                    const double delta = q - p[i];
                    if (std::abs(delta) < 1e-9) {
                        continue;
                    }
                    const double di = fuel(i, q) - fi;
                    for (size_t jj = (static_cast<size_t>(i) + k) % stride; jj < on.size();
                         jj += stride) {
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
                if (best_j >= 0) {
                    const double delta = best_q - p[i];
                    p[i] = best_q;
                    p[best_j] = std::clamp(p[best_j] - delta, in.P_min[best_j], in.P_max[best_j]);
                    fcache_[i] = fuel(i, p[i]);
                    fcache_[best_j] = fuel(best_j, p[best_j]);
                    improved = true;
                }
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
    // Quadratic-part optimum: p_u = clamp((lambda - b)/(2c)), bisection on lambda.
    void lambda_dispatch(const std::vector<int>& on, double demand, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        auto at = [&](int u, double lam) {
            if (in.c[u] <= 0.0) {
                return lam >= in.b[u] ? in.P_max[u] : in.P_min[u];
            }
            return std::clamp((lam - in.b[u]) / (2.0 * in.c[u]), in.P_min[u], in.P_max[u]);
        };
        double lo = kInf;
        double hi = -kInf;
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

    // Snap each unit to its nearest valve point, move whole valve steps on the
    // cheapest units per MW until the residual is under a step, then close it.
    void snap_and_balance(const std::vector<int>& on, double demand, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        idx_.assign(in.n_units, 0);
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
        double r = demand - sum;
        for (;;) {
            const bool up = r > 0.0;
            int bu = -1;
            double bscore = kInf;
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
                const double score = (fuel(u, v[k]) - fuel(u, p[u])) / step * (up ? 1.0 : -1.0);
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

    // The swing: the one unit whose cost rises least on absorbing r; if none can
    // absorb all of it, fill units to their bound in that order.
    void close_residual(const std::vector<int>& on, double r, std::vector<double>& p) const {
        const UCInstance& in = inst_;
        while (std::abs(r) > 1e-10) {
            int bu = -1;
            double bd = kInf;
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
            // No single unit can: move the one with the most room to its bound.
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

}  // namespace hook_detail

/// Process-wide counters over every worker's hook, for the run's log.
struct UCHookStats {
    std::atomic<int64_t> calls{0};
    std::atomic<int64_t> skipped{0};
    std::atomic<int64_t> improvements{0};
    std::atomic<int64_t> nanos{0};
    std::atomic<int64_t> throttled{0};
    std::atomic<int64_t> infeasible_calls{0};
    std::atomic<int64_t> ils_kicks{0};
    std::atomic<int64_t> ils_wins{0};
    static UCHookStats& get() {
        static UCHookStats s;
        return s;
    }
};

/// The hook. `commit_ls` turns the commitment search on; with it off the hook
/// only re-dispatches the search's commitment.
class UCHook : public InnerSolverHook {
public:
    UCHook(const UCInstance& inst, const UCModel& ucm, bool commit_ls, uint64_t seed)
        : inst_(inst),
          ucm_(ucm),
          ed_(inst),
          commit_ls_(commit_ls),
          rng_(seed),
          born_(std::chrono::steady_clock::now()) {}

    int polish_sweeps = 4;
    int ls_polish_sweeps = 0;  // > 0: the LS scores periods by polished dispatch
    int ls_polish_window = 0;  // its valve window (Dispatcher::polish)
    int ls_polish_jcap = 0;    // its absorbing-unit cap (Dispatcher::polish)
    size_t memo_cap = 200000;
    // Also run on infeasible batches whose commitment is feasible (re-dispatch
    // repairs p). Those calls are skipped once the hook has used more than
    // `infeasible_share` of this worker's wall time, or the commitment repeats.
    bool on_infeasible = false;
    double infeasible_share = 0.5;
    // Iterated local search after the descent: random kicks, re-descend, keep
    // if cheaper; at most `ils_seconds` per call and `ils_stall` kicks in a row
    // without improvement. 0 seconds = off.
    double ils_seconds = 0.0;
    int ils_stall = 100;
    int ils_kick = 3;
    double elite_prob = 0.0;  // chance an ILS starts from the hook's elite y
    // > 0: the descent after a kick tries only the kicked units plus this many
    // random ones (a full pass is the cost on 100+ unit rows).
    int ils_local = 0;
    // Repair a commitment that breaks min up/down or capacity (repair_row,
    // repair_capacity) instead of skipping the call.
    bool repair = false;

    [[nodiscard]] bool wants_infeasible() const override { return on_infeasible; }

    void solve(Model& model, ViolationManager& vm, const std::vector<int32_t>& /*last*/ = {},
               StopRef stop = {}) override {
        const auto t0 = std::chrono::steady_clock::now();
        UCHookStats& st = UCHookStats::get();
        const bool feasible_in = vm.is_feasible();
        if (!feasible_in) {
            const double alive = std::chrono::duration<double>(t0 - born_).count();
            if (own_seconds_ > infeasible_share * alive) {
                ++st.throttled;
                return;
            }
        }
        ++st.calls;
        run(model, vm, stop, feasible_in);
        const auto dt = std::chrono::steady_clock::now() - t0;
        own_seconds_ += std::chrono::duration<double>(dt).count();
        st.nanos += std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
    }

private:
    void run(Model& model, ViolationManager& vm, StopRef stop, bool feasible_in) {
        UCHookStats& st = UCHookStats::get();
        read_state(model);
        if (!feasible_in && y_ == last_y_) {
            ++st.skipped;
            return;
        }
        last_y_ = y_;
        if (repair) {
            repair_state();
        }
        if (!state_feasible()) {
            ++st.skipped;
            return;
        }
        // An infeasible point has no cost to beat: write back only what beats
        // the best this hook has seen.
        double before = best_seen_;
        if (feasible_in) {
            before = current_cost(model);
            best_seen_ = std::min(best_seen_, before);
        } else {
            ++st.infeasible_calls;
        }
        init_costs();
        if (commit_ls_) {
            commitment_search(stop);
            if (ils_seconds > 0.0) {
                // With probability elite_prob, kick from this hook's elite
                // commitment rather than the search's descended one.
                std::uniform_real_distribution<double> coin(0.0, 1.0);
                if (!elite_y_.empty() && elite_total_ < total_cost() - 1e-6 &&
                    coin(rng_) < elite_prob) {
                    y_ = elite_y_;
                    init_costs();
                }
                iterated_search(stop);
                const double c = total_cost();
                if (c < elite_total_) {
                    elite_total_ = c;
                    elite_y_ = y_;
                }
            }
        }
        // Final dispatch: polished, from the fast dispatch and from the model's
        // own p where the column is unchanged; the cheaper is kept.
        std::vector<std::vector<double>> pnew(inst_.n_periods);
        double total = final_dispatch(pnew);
        if (!(total < before - 1e-6) && y_ != y0_) {
            // The fast dispatch misjudged the recommitment: re-dispatch only.
            y_ = y0_;
            total = final_dispatch(pnew);
        }
        if (!(total < before - 1e-6)) {
            return;
        }
        best_seen_ = std::min(best_seen_, total);
        ++st.improvements;
        for (int u = 0; u < inst_.n_units; ++u) {
            for (int t = 0; t < inst_.n_periods; ++t) {
                model.var_mut(-(ucm_.y[u][t] + 1)).value = static_cast<double>(y_[u][t]);
                model.var_mut(-(ucm_.p[u][t] + 1)).value = pnew[t][u];
            }
        }
        full_evaluate(model);
        vm.invalidate_cache();
    }

    using Row = std::vector<int8_t>;

    void read_state(const Model& model) {
        const int n = inst_.n_units;
        const int T = inst_.n_periods;
        y_.assign(n, Row(T, 0));
        p0_.assign(n, std::vector<double>(T, 0.0));
        for (int u = 0; u < n; ++u) {
            for (int t = 0; t < T; ++t) {
                y_[u][t] = model.var(-(ucm_.y[u][t] + 1)).value > 0.5 ? 1 : 0;
                p0_[u][t] = model.var(-(ucm_.p[u][t] + 1)).value;
            }
        }
        y0_ = y_;
    }

    double current_cost(const Model& model) const { return model.node_value(model.objective_id()); }

    [[nodiscard]] std::vector<int> on_set(int t) const {
        std::vector<int> on;
        for (int u = 0; u < inst_.n_units; ++u) {
            if (y_[u][t] != 0) {
                on.push_back(u);
            }
        }
        return on;
    }

    hook_detail::ColumnKey key_of(int t) const {
        hook_detail::ColumnKey k;
        k.t = t;
        k.words.assign((static_cast<size_t>(inst_.n_units) + 63) / 64, 0);
        for (int u = 0; u < inst_.n_units; ++u) {
            if (y_[u][t] != 0) {
                k.words[static_cast<size_t>(u) / 64] |= uint64_t{1} << (u % 64);
            }
        }
        return k;
    }

    // Exactly the model's commitment rows for unit u: initial conditions, then
    // min up and min down (uc_model.h).
    [[nodiscard]] bool row_feasible(int u, const Row& r) const {
        const UCInstance& in = inst_;
        const int T = in.n_periods;
        if (in.y_prev[u] == 1 || in.y_prev[u] == 0) {
            const int held = in.y_prev[u];
            const int min_run = held == 1 ? in.min_on[u] : in.min_off[u];
            const int rem = std::max(0, min_run - in.n_init[u]);
            for (int t = 0; t < std::min(rem, T); ++t) {
                if (r[t] != held) {
                    return false;
                }
            }
        }
        for (int t = 0; t < T; ++t) {
            const int prev = t == 0 ? in.y_prev[u] : r[t - 1];
            if (r[t] == prev) {
                continue;
            }
            const int held = r[t];
            const int run = held == 1 ? in.min_on[u] : in.min_off[u];
            const int end = std::min(t + run, T);
            for (int tau = t + 1; tau < end; ++tau) {
                if (r[tau] != held) {
                    return false;
                }
            }
        }
        return true;
    }

    // Forward pass making row r feasible for unit u, only ever switching units
    // on outside the initial-condition span: that span is forced to its held
    // state; a startup's min_on run is extended; a shutdown whose off run is
    // cut short by a restart inside min_off is cancelled (the gap filled).
    void repair_row(int u, Row& r) const {
        const UCInstance& in = inst_;
        const int T = in.n_periods;
        int forced = 0;
        if (in.y_prev[u] == 1 || in.y_prev[u] == 0) {
            const int held = in.y_prev[u];
            const int min_run = held == 1 ? in.min_on[u] : in.min_off[u];
            forced = std::min(T, std::max(0, min_run - in.n_init[u]));
            std::fill(r.begin(), r.begin() + forced, static_cast<int8_t>(held));
        }
        for (int t = forced; t < T; ++t) {
            const int prev = t == 0 ? in.y_prev[u] : r[t - 1];
            if (r[t] == prev) {
                continue;
            }
            if (r[t] == 1) {
                const int end = std::min(t + in.min_on[u], T);
                std::fill(r.begin() + t, r.begin() + end, 1);
            } else {
                const int end = std::min(t + in.min_off[u], T);
                int restart = -1;
                for (int tau = t + 1; tau < end && restart < 0; ++tau) {
                    if (r[tau] == 1) {
                        restart = tau;
                    }
                }
                if (restart >= 0) {
                    std::fill(r.begin() + t, r.begin() + restart, 1);
                }
            }
        }
    }

    // Each row repaired, then every period short of demand + reserve gets the
    // largest idle unit that may start there, run for min_on, until covered.
    void repair_state() {
        const UCInstance& in = inst_;
        for (int u = 0; u < in.n_units; ++u) {
            if (!row_feasible(u, y_[u])) {
                repair_row(u, y_[u]);
            }
        }
        for (int t = 0; t < in.n_periods; ++t) {
            for (;;) {
                double cap = 0.0;
                for (int u = 0; u < in.n_units; ++u) {
                    cap += in.P_max[u] * y_[u][t];
                }
                if (cap >= in.demand[t] + std::max(0.0, in.reserve[t]) - 1e-9) {
                    break;
                }
                int best = -1;
                for (int u = 0; u < in.n_units; ++u) {
                    if (y_[u][t] == 0 && (best < 0 || in.P_max[u] > in.P_max[best])) {
                        Row r = y_[u];
                        r[t] = 1;
                        repair_row(u, r);
                        if (r[t] == 1 && row_feasible(u, r)) {
                            best = u;
                        }
                    }
                }
                if (best < 0) {
                    break;
                }
                y_[best][t] = 1;
                repair_row(best, y_[best]);
            }
        }
    }

    [[nodiscard]] double startup_cost(int u, const Row& r) const {
        const UCInstance& in = inst_;
        double c = 0.0;
        for (int t = 0; t < in.n_periods; ++t) {
            const int prev = t == 0 ? in.y_prev[u] : r[t - 1];
            if (r[t] == 1 && prev == 0) {
                bool hot = false;
                for (int tau = t - in.t_cold[u]; tau < t && !hot; ++tau) {
                    hot = (tau < 0 ? in.y_prev[u] : r[tau]) == 1;
                }
                c += hot ? in.a_hot[u] : in.a_cold[u];
            }
        }
        return c;
    }

    [[nodiscard]] bool reserve_ok(int t) const {
        if (inst_.reserve[t] <= 0) {
            return true;
        }
        double cap = 0.0;
        for (int u = 0; u < inst_.n_units; ++u) {
            cap += inst_.P_max[u] * y_[u][t];
        }
        return cap >= inst_.demand[t] + inst_.reserve[t] - 1e-9;
    }

    [[nodiscard]] bool state_feasible() const {
        for (int u = 0; u < inst_.n_units; ++u) {
            if (!row_feasible(u, y_[u])) {
                return false;
            }
        }
        for (int t = 0; t < inst_.n_periods; ++t) {
            if (!reserve_ok(t)) {
                return false;
            }
        }
        return true;
    }

    // Fast-dispatch cost of period t under the current y_, memoised.
    double period_cost(int t) {
        auto k = key_of(t);
        auto it = fast_memo_.find(k);
        if (it != fast_memo_.end()) {
            return it->second;
        }
        if (fast_memo_.size() >= memo_cap) {
            fast_memo_.clear();
        }
        if (!reserve_ok(t)) {
            fast_memo_.emplace(std::move(k), hook_detail::kInf);
            return hook_detail::kInf;
        }
        scratch_.assign(inst_.n_units, 0.0);
        const std::vector<int> on = on_set(t);
        double c = ed_.fast(on, inst_.demand[t], scratch_);
        if (ls_polish_sweeps > 0 && c < hook_detail::kInf) {
            c = ed_.polish(on, scratch_, ls_polish_sweeps, ls_polish_window, ls_polish_jcap);
        }
        fast_memo_.emplace(std::move(k), c);
        return c;
    }

    // Polished dispatch of period t into p (dense), memoised by column.
    double polished(int t, std::vector<double>& p) {
        auto k = key_of(t);
        auto it = polish_memo_.find(k);
        bool unchanged = true;
        for (int u = 0; u < inst_.n_units; ++u) {
            unchanged = unchanged && y_[u][t] == y0_[u][t];
        }
        const std::vector<int> on = on_set(t);
        double best = hook_detail::kInf;
        if (it != polish_memo_.end()) {
            p = it->second.p;
            best = it->second.cost;
        } else {
            p.assign(inst_.n_units, 0.0);
            best = ed_.fast(on, inst_.demand[t], p);
            if (best < hook_detail::kInf) {
                best = ed_.polish(on, p, polish_sweeps);
            }
        }
        if (unchanged) {
            // The search's own dispatch, if it balances, polished as well.
            std::vector<double> q(inst_.n_units, 0.0);
            double s = 0.0;
            for (int u : on) {
                q[u] = p0_[u][t];
                s += q[u];
            }
            if (std::abs(s - inst_.demand[t]) < 1e-7) {
                const double c = ed_.polish(on, q, polish_sweeps);
                if (c < best) {
                    best = c;
                    p = q;
                }
            }
        }
        if (polish_memo_.size() >= memo_cap / 20) {
            polish_memo_.clear();
        }
        polish_memo_[k] = hook_detail::Dispatch{best, p};
        return best;
    }

    double final_dispatch(std::vector<std::vector<double>>& pnew) {
        double total = 0.0;
        for (int t = 0; t < inst_.n_periods; ++t) {
            total += polished(t, pnew[t]);
        }
        for (int u = 0; u < inst_.n_units; ++u) {
            total += startup_cost(u, y_[u]);
        }
        return total;
    }

    void init_costs() {
        pc_.assign(inst_.n_periods, 0.0);
        for (int t = 0; t < inst_.n_periods; ++t) {
            pc_[t] = period_cost(t);
        }
        su_.assign(inst_.n_units, 0.0);
        for (int u = 0; u < inst_.n_units; ++u) {
            su_[u] = startup_cost(u, y_[u]);
        }
    }

    // Try unit rows `units` set to `rows` (already feasible rows). Returns the
    // cost change over the affected periods; leaves y_ as it was.
    double try_rows(const std::vector<int>& units, const std::vector<Row>& rows,
                    std::vector<int>& touched, std::vector<double>& new_pc) {
        touched.clear();
        for (size_t i = 0; i < units.size(); ++i) {
            const Row& r = rows[i];
            const Row& cur = y_[units[i]];
            for (int t = 0; t < inst_.n_periods; ++t) {
                if (r[t] != cur[t] &&
                    std::find(touched.begin(), touched.end(), t) == touched.end()) {
                    touched.push_back(t);
                }
            }
        }
        std::vector<Row> saved(units.size());
        double delta = 0.0;
        for (size_t i = 0; i < units.size(); ++i) {
            saved[i] = y_[units[i]];
            delta += startup_cost(units[i], rows[i]) - su_[units[i]];
            y_[units[i]] = rows[i];
        }
        new_pc.assign(touched.size(), 0.0);
        for (size_t k = 0; k < touched.size() && delta < hook_detail::kInf; ++k) {
            new_pc[k] = period_cost(touched[k]);
            delta += new_pc[k] - pc_[touched[k]];
        }
        for (size_t i = 0; i < units.size(); ++i) {
            y_[units[i]] = saved[i];
        }
        return delta;
    }

    void apply_rows(const std::vector<int>& units, const std::vector<Row>& rows,
                    const std::vector<int>& touched, const std::vector<double>& new_pc) {
        for (size_t i = 0; i < units.size(); ++i) {
            y_[units[i]] = rows[i];
            su_[units[i]] = startup_cost(units[i], rows[i]);
        }
        for (size_t k = 0; k < touched.size(); ++k) {
            pc_[touched[k]] = new_pc[k];
        }
    }

    // Blocks of 1s in a row, as [start, end] inclusive.
    static std::vector<std::pair<int, int>> blocks(const Row& r) {
        std::vector<std::pair<int, int>> b;
        const int T = static_cast<int>(r.size());
        for (int t = 0; t < T; ++t) {
            if (r[t] == 1 && (t == 0 || r[t - 1] == 0)) {
                int e = t;
                while (e + 1 < T && r[e + 1] == 1) {
                    ++e;
                }
                b.emplace_back(t, e);
            }
        }
        return b;
    }

    // Single-unit candidate rows: drop, shrink or extend a block, fill a gap,
    // or add a min_on-long block in a gap. Only rows feasible for the unit.
    void candidate_rows(int u, std::vector<Row>& out) {
        out.clear();
        const Row& r = y_[u];
        const int T = inst_.n_periods;
        auto push = [&](Row&& c) {
            if (row_feasible(u, c)) {
                out.push_back(std::move(c));
            }
        };
        const auto bl = blocks(r);
        for (size_t b = 0; b < bl.size(); ++b) {
            const auto [s, e] = bl[b];
            Row c = r;
            std::fill(c.begin() + s, c.begin() + e + 1, 0);
            push(std::move(c));
            if (e > s) {
                Row c1 = r;
                c1[s] = 0;
                push(std::move(c1));
                Row c2 = r;
                c2[e] = 0;
                push(std::move(c2));
            }
            if (s > 0) {
                Row c3 = r;
                c3[s - 1] = 1;
                push(std::move(c3));
            }
            if (e + 1 < T) {
                Row c4 = r;
                c4[e + 1] = 1;
                push(std::move(c4));
            }
            if (b + 1 < bl.size()) {
                Row c5 = r;
                std::fill(c5.begin() + e + 1, c5.begin() + bl[b + 1].first, 1);
                push(std::move(c5));
            }
        }
        // New blocks in the off gaps, at a few sampled starts.
        const int L = std::max(1, inst_.min_on[u]);
        std::vector<int> starts;
        for (int t = 0; t < T; ++t) {
            if (r[t] == 0) {
                starts.push_back(t);
            }
        }
        std::shuffle(starts.begin(), starts.end(), rng_);
        const size_t cap = std::min<size_t>(starts.size(), 4);
        for (size_t i = 0; i < cap; ++i) {
            Row c = r;
            for (int t = starts[i]; t < std::min(T, starts[i] + L); ++t) {
                c[t] = 1;
            }
            push(std::move(c));
        }
    }

    // First-improvement descent over single-unit rows, then unit swaps.
    // With `subset`, only those units' rows (and swaps out of them) are tried.
    void commitment_search(StopRef stop, const std::vector<int>* subset = nullptr) {
        std::vector<int> order;
        if (subset != nullptr) {
            order = *subset;
        } else {
            order.resize(inst_.n_units);
            for (int u = 0; u < inst_.n_units; ++u) {
                order[u] = u;
            }
        }
        std::vector<Row> cands;
        std::vector<int> touched;
        std::vector<double> new_pc;
        std::vector<int> one(1);
        std::vector<Row> one_row(1);
        for (int pass = 0; pass < 50; ++pass) {
            bool improved = false;
            std::shuffle(order.begin(), order.end(), rng_);
            for (int u : order) {
                if (stop.requested()) {
                    return;
                }
                candidate_rows(u, cands);
                one[0] = u;
                for (auto& c : cands) {
                    one_row[0] = c;
                    const double d = try_rows(one, one_row, touched, new_pc);
                    if (d < -1e-6) {
                        apply_rows(one, one_row, touched, new_pc);
                        improved = true;
                        break;
                    }
                }
            }
            if (stop.requested()) {
                return;
            }
            if (swap_pass(stop, order)) {
                improved = true;
            }
            if (!improved) {
                return;
            }
        }
    }

    // Swap: unit u's block off, an idle unit v on over the same span.
    bool swap_pass(StopRef stop, const std::vector<int>& from) {
        bool improved = false;
        std::vector<int> two(2);
        std::vector<Row> rows(2);
        std::vector<int> touched;
        std::vector<double> new_pc;
        std::vector<int> others(inst_.n_units);
        for (int u = 0; u < inst_.n_units; ++u) {
            others[u] = u;
        }
        for (int u : from) {
            for (const auto& [s, e] : blocks(y_[u])) {
                if (stop.requested()) {
                    return improved;
                }
                Row off = y_[u];
                std::fill(off.begin() + s, off.begin() + e + 1, 0);
                if (!row_feasible(u, off)) {
                    continue;
                }
                std::shuffle(others.begin(), others.end(), rng_);
                int tried = 0;
                for (int v : others) {
                    if (v == u || tried >= 8) {
                        continue;
                    }
                    bool idle = true;
                    for (int t = s; t <= e && idle; ++t) {
                        idle = y_[v][t] == 0;
                    }
                    if (!idle) {
                        continue;
                    }
                    Row on = y_[v];
                    std::fill(on.begin() + s, on.begin() + e + 1, 1);
                    if (!row_feasible(v, on)) {
                        continue;
                    }
                    ++tried;
                    two[0] = u;
                    two[1] = v;
                    rows[0] = off;
                    rows[1] = on;
                    const double d = try_rows(two, rows, touched, new_pc);
                    if (d < -1e-6) {
                        apply_rows(two, rows, touched, new_pc);
                        improved = true;
                        break;
                    }
                }
            }
        }
        return improved;
    }

    [[nodiscard]] double total_cost() const {
        double c = 0.0;
        for (double x : pc_) {
            c += x;
        }
        for (double x : su_) {
            c += x;
        }
        return c;
    }

    // One kick: `ils_kick` random single-unit rows (from candidate_rows) or
    // unit swaps, applied whatever they cost, so long as dispatch stays finite.
    void kick(std::vector<int>& kicked) {
        kicked.clear();
        std::vector<Row> cands;
        std::vector<int> touched;
        std::vector<double> new_pc;
        std::vector<int> one(1);
        std::vector<Row> one_row(1);
        std::uniform_int_distribution<int> unit(0, inst_.n_units - 1);
        for (int k = 0; k < ils_kick; ++k) {
            for (int attempt = 0; attempt < 10; ++attempt) {
                const int u = unit(rng_);
                candidate_rows(u, cands);
                if (cands.empty()) {
                    continue;
                }
                std::uniform_int_distribution<size_t> pick(0, cands.size() - 1);
                one[0] = u;
                one_row[0] = cands[pick(rng_)];
                const double d = try_rows(one, one_row, touched, new_pc);
                if (d < hook_detail::kInf) {
                    apply_rows(one, one_row, touched, new_pc);
                    kicked.push_back(u);
                    break;
                }
            }
        }
    }

    // ILS from the descended y_: kick, re-descend, keep the better; ends on
    // the time slice, the stall cap or `stop`, with y_/pc_/su_ the best found.
    void iterated_search(StopRef stop) {
        UCHookStats& st = UCHookStats::get();
        const auto end = std::chrono::steady_clock::now() +
                         std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                             std::chrono::duration<double>(ils_seconds));
        double best = total_cost();
        std::vector<Row> by = y_;
        std::vector<double> bpc = pc_;
        std::vector<double> bsu = su_;
        int stall = 0;
        std::vector<int> kicked;
        std::vector<int> local;
        std::uniform_int_distribution<int> unit(0, inst_.n_units - 1);
        const bool restrict = ils_local > 0 && ils_local < inst_.n_units;
        while (stall < ils_stall && !stop.requested() && std::chrono::steady_clock::now() < end) {
            ++st.ils_kicks;
            kick(kicked);
            if (restrict) {
                local = kicked;
                for (int i = 0; i < ils_local; ++i) {
                    const int u = unit(rng_);
                    if (std::find(local.begin(), local.end(), u) == local.end()) {
                        local.push_back(u);
                    }
                }
                commitment_search(stop, &local);
            } else {
                commitment_search(stop);
            }
            const double c = total_cost();
            if (c < best - 1e-6) {
                ++st.ils_wins;
                best = c;
                by = y_;
                bpc = pc_;
                bsu = su_;
                stall = 0;
            } else {
                ++stall;
                y_ = by;
                pc_ = bpc;
                su_ = bsu;
            }
        }
        y_ = std::move(by);
        pc_ = std::move(bpc);
        su_ = std::move(bsu);
    }

    const UCInstance& inst_;
    const UCModel& ucm_;
    hook_detail::Dispatcher ed_;
    bool commit_ls_;
    std::mt19937_64 rng_;
    std::chrono::steady_clock::time_point born_;
    double own_seconds_ = 0.0;
    double best_seen_ = hook_detail::kInf;
    double elite_total_ = hook_detail::kInf;  // fast-dispatch cost of elite_y_
    std::vector<std::vector<int8_t>> elite_y_;
    std::vector<Row> y_, y0_, last_y_;
    std::vector<std::vector<double>> p0_;
    std::vector<double> pc_, su_, scratch_;
    std::unordered_map<hook_detail::ColumnKey, double, hook_detail::ColumnKeyHash> fast_memo_;
    std::unordered_map<hook_detail::ColumnKey, hook_detail::Dispatch, hook_detail::ColumnKeyHash>
        polish_memo_;
};

}  // namespace cbls::uc_chped
