#pragma once

#include "data.h"
#include "uc_commitment.h"
#include "uc_dispatch.h"

#include <algorithm>
#include <cbls/rng.h>
#include <cbls/stop.h>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

// Commitment local search and ILS for the opt-in UC hook (#445, #460): moves
// on y scored by per-period dispatch, with no tie to the cbls model. Not used
// by the published protocol.
namespace cbls::uc_chped {

struct CommitmentSearchConfig {
    int polish_sweeps = 4;     // final dispatch's polish (Dispatcher::polish)
    int ls_polish_sweeps = 0;  // > 0: the LS scores periods by polished dispatch
    int ls_polish_window = 0;  // its valve window
    int ls_polish_jcap = 0;    // its absorbing-unit cap
    size_t memo_cap = 200000;  // fast memo entries; the polish memo holds 1/20 of it
    int ils_stall = 100;       // ILS ends after this many kicks in a row without a win
    int ils_kick = 3;          // single-unit moves per kick
    // > 0: the descent after a kick tries only the kicked units plus this many
    // random ones (a full pass is the cost on 100+ unit rows).
    int ils_local = 0;
};

struct CommitmentSearchStats {
    int64_t ils_kicks = 0;
    int64_t ils_wins = 0;
};

namespace search_detail {

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

/// A period's polished dispatch: fuel cost (+inf when the on-set cannot meet
/// demand) and p for every unit, 0 when off.
struct PeriodDispatch {
    double cost = kDispatchInf;
    std::vector<double> p;
};

}  // namespace search_detail

/// Holds y_ with its cached costs: pc_[t] is period t's fast-dispatch cost and
/// su_[u] unit u's startup cost, both of y_ after init_costs and every move.
/// Memos are keyed on a column alone and survive load(). One per thread.
class CommitmentSearch {
public:
    using Dispatches = std::vector<std::vector<double>>;  // [t][u]

    CommitmentSearch(UCInstance&&, const CommitmentSearchConfig&, uint64_t) = delete;
    CommitmentSearch(const UCInstance& inst, const CommitmentSearchConfig& cfg, uint64_t seed)
        : inst_(inst), cfg_(cfg), ed_(inst), rng_(seed) {}

    /// Start from commitment y and its dispatch p[u][t]; y is also the loaded
    /// commitment whose columns final_dispatch may take p from.
    void load(const Commitment& y, const std::vector<std::vector<double>>& p) {
        y_ = y;
        y0_ = y;
        p0_ = p;
    }

    /// repair_state on y_; costs then need init_costs.
    bool repair() { return repair_state(inst_, y_); }
    [[nodiscard]] bool feasible() const { return commitment_feasible(inst_, y_); }

    /// Replace y_ (an elite restart); costs then need init_costs.
    void set_commitment(const Commitment& y) { y_ = y; }

    /// Cache pc_ and su_ for y_. False when stopped part-way: costs invalid.
    bool init_costs(StopRef stop = {}) {
        pc_.assign(inst_.n_periods, 0.0);
        for (int t = 0; t < inst_.n_periods; ++t) {
            if (stop.requested()) {
                return false;
            }
            pc_[t] = period_cost(t);
        }
        su_.assign(inst_.n_units, 0.0);
        for (int u = 0; u < inst_.n_units; ++u) {
            su_[u] = startup_cost(inst_, u, y_[u]);
        }
        return true;
    }

    /// First-improvement descent over single-unit rows, then unit swaps, until
    /// no move improves, 50 passes, or `stop`. With `subset`, only those
    /// units' rows (and swaps out of them) are tried.
    void descend(StopRef stop = {}, const std::vector<int>* subset = nullptr) {
        std::vector<int> order;
        if (subset != nullptr) {
            order = *subset;
        } else {
            order.resize(inst_.n_units);
            for (int u = 0; u < inst_.n_units; ++u) {
                order[u] = u;
            }
        }
        constexpr int kMaxPasses = 50;
        for (int pass = 0; pass < kMaxPasses; ++pass) {
            rng_.shuffle(order);
            bool improved = false;
            for (int u : order) {
                if (stop.requested()) {
                    return;
                }
                improved = improve_unit(u) || improved;
            }
            if (stop.requested()) {
                return;
            }
            improved = swap_pass(stop, order) || improved;
            if (!improved) {
                return;
            }
        }
    }

    /// ILS from y_: kick, re-descend, keep the better. Ends on `end`, after
    /// `ils_stall` kicks without a win, or on `stop`, with y_/pc_/su_ the best.
    void iterate(StopRef stop, std::chrono::steady_clock::time_point end) {
        double best = total_cost();
        Commitment by = y_;
        std::vector<double> bpc = pc_;
        std::vector<double> bsu = su_;
        int stall = 0;
        std::vector<int> kicked;
        std::vector<int> local;
        const bool restrict = cfg_.ils_local > 0 && cfg_.ils_local < inst_.n_units;
        while (stall < cfg_.ils_stall && !stop.requested() &&
               std::chrono::steady_clock::now() < end) {
            ++stats_.ils_kicks;
            kick(kicked);
            if (restrict) {
                local = kicked;
                for (int i = 0; i < cfg_.ils_local; ++i) {
                    const int u = static_cast<int>(rng_.integers(0, inst_.n_units));
                    if (std::find(local.begin(), local.end(), u) == local.end()) {
                        local.push_back(u);
                    }
                }
                descend(stop, &local);
            } else {
                descend(stop);
            }
            const double c = total_cost();
            if (c < best - 1e-6) {
                ++stats_.ils_wins;
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

    /// One kick: `ils_kick` random single-unit rows from candidate_rows,
    /// applied whatever they cost, so long as dispatch stays finite.
    void kick(std::vector<int>& kicked) {
        kicked.clear();
        std::vector<CommitRow> cands;
        std::vector<int> touched;
        std::vector<double> new_pc;
        std::vector<int> one(1);
        std::vector<CommitRow> one_row(1);
        constexpr int kAttempts = 10;
        for (int k = 0; k < cfg_.ils_kick; ++k) {
            for (int attempt = 0; attempt < kAttempts; ++attempt) {
                const int u = static_cast<int>(rng_.integers(0, inst_.n_units));
                candidate_rows(u, cands);
                if (cands.empty()) {
                    continue;
                }
                one[0] = u;
                one_row[0] = cands[rng_.integers(0, static_cast<int64_t>(cands.size()))];
                const double d = try_rows(one, one_row, touched, new_pc);
                if (d < kDispatchInf) {
                    apply_rows(one, one_row, touched, new_pc);
                    kicked.push_back(u);
                    break;
                }
            }
        }
    }

    /// Cost change of setting units[i]'s row to rows[i] (each row-feasible),
    /// over the periods it touches; leaves y_ as it was. +inf when a touched
    /// period becomes undispatchable, so no inf - inf NaN (#455, Low 9).
    double try_rows(const std::vector<int>& units, const std::vector<CommitRow>& rows,
                    std::vector<int>& touched, std::vector<double>& new_pc) {
        touched.clear();
        for (size_t i = 0; i < units.size(); ++i) {
            const CommitRow& r = rows[i];
            const CommitRow& cur = y_[units[i]];
            for (int t = 0; t < inst_.n_periods; ++t) {
                if (r[t] != cur[t] &&
                    std::find(touched.begin(), touched.end(), t) == touched.end()) {
                    touched.push_back(t);
                }
            }
        }
        std::vector<CommitRow> saved(units.size());
        double delta = 0.0;
        for (size_t i = 0; i < units.size(); ++i) {
            saved[i] = y_[units[i]];
            delta += startup_cost(inst_, units[i], rows[i]) - su_[units[i]];
            y_[units[i]] = rows[i];
        }
        new_pc.assign(touched.size(), 0.0);
        for (size_t k = 0; k < touched.size(); ++k) {
            new_pc[k] = period_cost(touched[k]);
            if (new_pc[k] == kDispatchInf) {
                delta = kDispatchInf;
                break;
            }
            delta += new_pc[k] - pc_[touched[k]];
        }
        for (size_t i = 0; i < units.size(); ++i) {
            y_[units[i]] = saved[i];
        }
        return delta;
    }

    /// Single-unit candidate rows: drop, shrink or extend a block, fill a gap,
    /// or add a min_on-long block at up to 4 sampled off periods. Only rows
    /// feasible for the unit.
    void candidate_rows(int u, std::vector<CommitRow>& out) {
        out.clear();
        block_moves(u, out);
        new_block_moves(u, out);
    }

    /// Sum of pc_ and su_: y_'s fast-dispatch cost.
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

    /// Polished dispatch of every period of y_ into pnew[t][u], plus startup
    /// cost; a column equal to the loaded one also tries the loaded p,
    /// polished, if it balances within bounds. +inf when stopped part-way.
    double final_dispatch(Dispatches& pnew, StopRef stop = {}) {
        pnew.assign(inst_.n_periods, {});
        double total = 0.0;
        for (int t = 0; t < inst_.n_periods; ++t) {
            if (stop.requested()) {
                return kDispatchInf;
            }
            total += polished(t, pnew[t]);
        }
        for (int u = 0; u < inst_.n_units; ++u) {
            total += startup_cost(inst_, u, y_[u]);
        }
        return total;
    }

    [[nodiscard]] const Commitment& commitment() const { return y_; }
    [[nodiscard]] const Commitment& loaded() const { return y0_; }
    [[nodiscard]] const std::vector<double>& period_costs() const { return pc_; }
    [[nodiscard]] const std::vector<double>& startup_costs() const { return su_; }
    [[nodiscard]] const CommitmentSearchStats& stats() const { return stats_; }
    [[nodiscard]] size_t fast_memo_size() const { return fast_memo_.size(); }
    [[nodiscard]] size_t polish_memo_size() const { return polish_memo_.size(); }

private:
    [[nodiscard]] std::vector<int> on_set(int t) const {
        std::vector<int> on;
        for (int u = 0; u < inst_.n_units; ++u) {
            if (y_[u][t] != 0) {
                on.push_back(u);
            }
        }
        return on;
    }

    [[nodiscard]] search_detail::ColumnKey key_of(int t) const {
        search_detail::ColumnKey k;
        k.t = t;
        k.words.assign((static_cast<size_t>(inst_.n_units) + 63) / 64, 0);
        for (int u = 0; u < inst_.n_units; ++u) {
            if (y_[u][t] != 0) {
                k.words[static_cast<size_t>(u) / 64] |= uint64_t{1} << (u % 64);
            }
        }
        return k;
    }

    // Fast-dispatch cost of period t under y_ (polished if ls_polish_sweeps),
    // memoised by column; +inf unless capacity_ok.
    double period_cost(int t) {
        auto k = key_of(t);
        auto it = fast_memo_.find(k);
        if (it != fast_memo_.end()) {
            return it->second;
        }
        if (fast_memo_.size() >= cfg_.memo_cap) {
            fast_memo_.clear();
        }
        double c = kDispatchInf;
        if (capacity_ok(inst_, y_, t)) {
            scratch_.assign(inst_.n_units, 0.0);
            const std::vector<int> on = on_set(t);
            c = ed_.fast(on, inst_.demand[t], scratch_);
            if (cfg_.ls_polish_sweeps > 0 && c < kDispatchInf) {
                c = ed_.polish(on, scratch_, cfg_.ls_polish_sweeps, cfg_.ls_polish_window,
                               cfg_.ls_polish_jcap);
            }
        }
        fast_memo_.emplace(std::move(k), c);
        return c;
    }

    // Polished dispatch of period t into p. The memo holds only the dispatch
    // from scratch, a function of the column; the loaded p is tried on top, so
    // no entry depends on p0_ (#460).
    double polished(int t, std::vector<double>& p) {
        const std::vector<int> on = on_set(t);
        auto k = key_of(t);
        auto it = polish_memo_.find(k);
        double best = kDispatchInf;
        if (it != polish_memo_.end()) {
            p = it->second.p;
            best = it->second.cost;
        } else {
            p.assign(inst_.n_units, 0.0);
            best = ed_.fast(on, inst_.demand[t], p);
            if (best < kDispatchInf) {
                best = ed_.polish(on, p, cfg_.polish_sweeps);
            }
            if (polish_memo_.size() >= cfg_.memo_cap / 20) {
                polish_memo_.clear();
            }
            polish_memo_.emplace(std::move(k), search_detail::PeriodDispatch{best, p});
        }
        std::vector<double> q;
        if (loaded_dispatch(t, on, q)) {
            const double c = ed_.polish(on, q, cfg_.polish_sweeps);
            if (c < best) {
                best = c;
                p = std::move(q);
            }
        }
        return best;
    }

    // The loaded p of period t into q, when column t is the loaded one and that
    // p balances demand with every on unit within its bounds.
    bool loaded_dispatch(int t, const std::vector<int>& on, std::vector<double>& q) const {
        for (int u = 0; u < inst_.n_units; ++u) {
            if (y_[u][t] != y0_[u][t]) {
                return false;
            }
        }
        q.assign(inst_.n_units, 0.0);
        double s = 0.0;
        for (int u : on) {
            const double x = p0_[u][t];
            if (x < inst_.P_min[u] - 1e-9 || x > inst_.P_max[u] + 1e-9) {
                return false;
            }
            q[u] = x;
            s += x;
        }
        return std::abs(s - inst_.demand[t]) < 1e-7;
    }

    void apply_rows(const std::vector<int>& units, const std::vector<CommitRow>& rows,
                    const std::vector<int>& touched, const std::vector<double>& new_pc) {
        for (size_t i = 0; i < units.size(); ++i) {
            y_[units[i]] = rows[i];
            su_[units[i]] = startup_cost(inst_, units[i], rows[i]);
        }
        for (size_t k = 0; k < touched.size(); ++k) {
            pc_[touched[k]] = new_pc[k];
        }
    }

    // Apply unit u's first improving candidate row; whether one was found.
    bool improve_unit(int u) {
        candidate_rows(u, cands_);
        one_[0] = u;
        const auto hit = std::find_if(cands_.begin(), cands_.end(), [&](const CommitRow& c) {
            one_row_[0] = c;
            return try_rows(one_, one_row_, touched_, new_pc_) < -1e-6;
        });
        if (hit == cands_.end()) {
            return false;
        }
        apply_rows(one_, one_row_, touched_, new_pc_);  // touched_/new_pc_ are hit's
        return true;
    }

    // Blocks of 1s in a row, as [start, end] inclusive.
    static std::vector<std::pair<int, int>> blocks(const CommitRow& r) {
        std::vector<std::pair<int, int>> b;
        const int n_periods = static_cast<int>(r.size());
        for (int t = 0; t < n_periods; ++t) {
            if (r[t] == 1 && (t == 0 || r[t - 1] == 0)) {
                int e = t;
                while (e + 1 < n_periods && r[e + 1] == 1) {
                    ++e;
                }
                b.emplace_back(t, e);
            }
        }
        return b;
    }

    void push_if_feasible(int u, CommitRow&& c, std::vector<CommitRow>& out) const {
        if (row_feasible(inst_, u, c)) {
            out.push_back(std::move(c));
        }
    }

    // Per block of 1s: drop it, shrink either end, extend either end, fill
    // the gap to the next block.
    void block_moves(int u, std::vector<CommitRow>& out) const {
        const CommitRow& r = y_[u];
        const int n_periods = inst_.n_periods;
        const auto bl = blocks(r);
        for (size_t b = 0; b < bl.size(); ++b) {
            const auto [s, e] = bl[b];
            CommitRow c = r;
            std::fill(c.begin() + s, c.begin() + e + 1, 0);
            push_if_feasible(u, std::move(c), out);
            if (e > s) {
                CommitRow c1 = r;
                c1[s] = 0;
                push_if_feasible(u, std::move(c1), out);
                CommitRow c2 = r;
                c2[e] = 0;
                push_if_feasible(u, std::move(c2), out);
            }
            if (s > 0) {
                CommitRow c3 = r;
                c3[s - 1] = 1;
                push_if_feasible(u, std::move(c3), out);
            }
            if (e + 1 < n_periods) {
                CommitRow c4 = r;
                c4[e + 1] = 1;
                push_if_feasible(u, std::move(c4), out);
            }
            if (b + 1 < bl.size()) {
                CommitRow c5 = r;
                std::fill(c5.begin() + e + 1, c5.begin() + bl[b + 1].first, 1);
                push_if_feasible(u, std::move(c5), out);
            }
        }
    }

    // A min_on-long block of 1s from each of up to 4 sampled off periods.
    void new_block_moves(int u, std::vector<CommitRow>& out) {
        const CommitRow& r = y_[u];
        const int n_periods = inst_.n_periods;
        const int len = std::max(1, inst_.min_on[u]);
        std::vector<int> starts;
        for (int t = 0; t < n_periods; ++t) {
            if (r[t] == 0) {
                starts.push_back(t);
            }
        }
        rng_.shuffle(starts);
        constexpr size_t kStarts = 4;
        const size_t cap = std::min(starts.size(), kStarts);
        for (size_t i = 0; i < cap; ++i) {
            CommitRow c = r;
            std::fill(c.begin() + starts[i], c.begin() + std::min(n_periods, starts[i] + len), 1);
            push_if_feasible(u, std::move(c), out);
        }
    }

    // Swaps out of each unit in `from`: a block of u off, an idle unit v on
    // over the same span; up to 8 row-feasible v per block, first improving
    // swap applied.
    bool swap_pass(StopRef stop, const std::vector<int>& from) {
        bool improved = false;
        std::vector<int> others(inst_.n_units);
        for (int u = 0; u < inst_.n_units; ++u) {
            others[u] = u;
        }
        for (int u : from) {
            for (const auto& [s, e] : blocks(y_[u])) {
                if (stop.requested()) {
                    return improved;
                }
                improved = swap_block(u, s, e, others) || improved;
            }
        }
        return improved;
    }

    bool swap_block(int u, int s, int e, std::vector<int>& others) {
        CommitRow off = y_[u];
        std::fill(off.begin() + s, off.begin() + e + 1, 0);
        if (!row_feasible(inst_, u, off)) {
            return false;
        }
        rng_.shuffle(others);
        constexpr int kTries = 8;
        int tried = 0;
        std::vector<int> two{u, 0};
        std::vector<CommitRow> rows{off, {}};
        for (int v : others) {
            if (tried >= kTries) {
                break;
            }
            if (v == u || !idle_over(v, s, e)) {
                continue;
            }
            CommitRow on = y_[v];
            std::fill(on.begin() + s, on.begin() + e + 1, 1);
            if (!row_feasible(inst_, v, on)) {
                continue;
            }
            ++tried;
            two[1] = v;
            rows[1] = std::move(on);
            if (try_rows(two, rows, touched_, new_pc_) < -1e-6) {
                apply_rows(two, rows, touched_, new_pc_);
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool idle_over(int v, int s, int e) const {
        for (int t = s; t <= e; ++t) {
            if (y_[v][t] != 0) {
                return false;
            }
        }
        return true;
    }

    const UCInstance& inst_;
    CommitmentSearchConfig cfg_;
    Dispatcher ed_;
    RNG rng_;
    CommitmentSearchStats stats_;
    Commitment y_, y0_;
    std::vector<std::vector<double>> p0_;  // loaded dispatch, [u][t]
    std::vector<double> pc_, su_, scratch_;
    // Scratch for improve_unit and swap_block.
    std::vector<CommitRow> cands_;
    std::vector<int> touched_;
    std::vector<double> new_pc_;
    std::vector<int> one_ = std::vector<int>(1);
    std::vector<CommitRow> one_row_ = std::vector<CommitRow>(1);
    std::unordered_map<search_detail::ColumnKey, double, search_detail::ColumnKeyHash> fast_memo_;
    std::unordered_map<search_detail::ColumnKey, search_detail::PeriodDispatch,
                       search_detail::ColumnKeyHash>
        polish_memo_;
};

}  // namespace cbls::uc_chped
