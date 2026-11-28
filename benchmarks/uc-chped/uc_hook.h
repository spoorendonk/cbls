#pragma once

#include "data.h"
#include "uc_commitment.h"
#include "uc_model.h"
#include "uc_search.h"

#include <algorithm>
#include <cbls/cbls.h>
#include <chrono>
#include <cstdint>
#include <vector>

// Opt-in UC inner solver for the valve-point UC (#445, #461): on each point it
// is handed, re-dispatch every period and run the commitment search, then write
// back what beats the point. Off unless constructed; the published protocol
// never runs it.
namespace cbls::uc_chped {

struct UCHookConfig {
    CommitmentSearchConfig search;
    // false: re-dispatch the search's commitment only (no descent, ILS or elite).
    bool commit_ls = true;
    // Repair a commitment breaking min up/down or capacity instead of skipping.
    bool repair = false;
    // Also run on infeasible batches (wants_infeasible); such a call is skipped
    // once this hook has used more than `infeasible_share` of its wall time
    // since construction, or when its commitment repeats the last call's.
    bool on_infeasible = false;
    double infeasible_share = 0.5;
    // ILS after the descent, at most this long per call; 0 = off. Never on the
    // hook's first descent, so the first incumbent is not held up by it.
    double ils_seconds = 0.0;
    double elite_prob = 0.0;  // chance an ILS starts from the hook's elite y
    // Must equal the search's SearchConfig::feasibility_tolerance: a smaller one
    // sends points the search calls feasible down the infeasible path.
    double feasibility_tolerance = kDefaultFeasibilityTolerance;
};

/// One hook's counters; the runner sums them over workers.
struct UCHookStats {
    int64_t calls = 0;
    int64_t skipped = 0;           // repeated infeasible commitment, or not repaired
    int64_t throttled = 0;         // infeasible call over the time share
    int64_t infeasible_calls = 0;  // run on an infeasible point
    int64_t improvements = 0;      // write-backs kept
    int64_t rejected = 0;          // write-backs the model refused, rolled back
    int64_t nanos = 0;             // wall time in run()
    int64_t ils_nanos = 0;         // of which in the ILS
};

/// One per worker (ParallelSearch's hook_factory); holds var handles, valid on
/// every copy of the model built for `inst`. Across pool restarts it keeps all
/// its state (#461): best_seen_ stays at least this worker's incumbent cost, so
/// it is only a stricter bar for infeasible-input write-backs after a restart;
/// elite_y_ stays a feasible commitment of the same instance; descended_, the
/// throttle clock and last_y_ are about the hook's own work, not the run's.
/// The throttle and the ILS read the wall clock, so with either on a run is
/// not bit-reproducible.
class UCHook : public InnerSolverHook {
public:
    UCHook(UCInstance&&, const UCModel&, const UCHookConfig&, uint64_t) = delete;
    UCHook(const UCInstance& inst, const UCModel& ucm, const UCHookConfig& cfg, uint64_t seed)
        : inst_(inst),
          y_handles_(ucm.y),
          p_handles_(ucm.p),
          cfg_(cfg),
          search_(inst, cfg.search, seed),
          rng_(seed ^ 0x9e3779b97f4a7c15ULL),
          born_(Clock::now()) {}

    [[nodiscard]] bool wants_infeasible() const override { return cfg_.on_infeasible; }

    void solve(Model& model, ViolationManager& vm, const std::vector<int32_t>& /*last*/ = {},
               StopRef stop = {}) override {
        const auto t0 = Clock::now();
        // The search's own test (#454): is_feasible counts the objective row.
        const bool feasible_in = vm.max_real_violation() <= cfg_.feasibility_tolerance;
        if (!feasible_in) {
            const double alive = std::chrono::duration<double>(t0 - born_).count();
            if (own_seconds_ > cfg_.infeasible_share * alive) {
                ++stats_.throttled;
                return;
            }
        }
        ++stats_.calls;
        run(model, vm, stop, feasible_in);
        const auto dt = Clock::now() - t0;
        own_seconds_ += std::chrono::duration<double>(dt).count();
        stats_.nanos += std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
    }

    [[nodiscard]] const UCHookStats& stats() const { return stats_; }
    [[nodiscard]] const CommitmentSearchStats& search_stats() const { return search_.stats(); }
    [[nodiscard]] double best_seen() const { return best_seen_; }
    /// The search's cost of the last kept write-back; +inf before one.
    [[nodiscard]] double last_kept_total() const { return last_kept_total_; }

private:
    using Clock = std::chrono::steady_clock;

    // The outer stop or the ILS deadline: iterate() reads its end only between
    // kicks, the re-descent polls this (#461).
    struct DeadlineStop {
        StopRef outer;
        Clock::time_point end;
        [[nodiscard]] bool requested() const { return outer.requested() || Clock::now() >= end; }
    };

    void run(Model& model, ViolationManager& vm, StopRef stop, bool feasible_in) {
        read_state(model);
        if (!feasible_in && y_in_ == last_y_) {
            ++stats_.skipped;
            return;
        }
        search_.load(y_in_, p_in_);
        if (cfg_.repair) {
            search_.repair(stop);
        }
        if (stop.requested()) {
            return;
        }
        last_y_ = y_in_;  // after the stop: a stopped call does not skip the next
        if (!search_.feasible()) {
            ++stats_.skipped;
            return;
        }
        // The post-repair commitment: the fallback must not be the loaded y,
        // which may break min up/down or capacity (#452 review, High 2).
        const Commitment start = search_.commitment();
        // An infeasible point has no cost to beat: beat the best seen instead.
        double before = best_seen_;
        if (feasible_in) {
            before = model.node_value(model.objective_id());
            best_seen_ = std::min(best_seen_, before);
        } else {
            ++stats_.infeasible_calls;
        }
        // final_dispatch needs no cached costs, so re-dispatch alone skips them.
        if (cfg_.commit_ls && (!search_.init_costs(stop) || !commitment_search(stop))) {
            last_y_.clear();
            return;
        }
        CommitmentSearch::Dispatches pnew;
        double total = search_.final_dispatch(pnew, stop);
        if (!(total < before - 1e-6) && search_.commitment() != start) {
            // The fast dispatch misjudged the recommitment: re-dispatch only.
            search_.set_commitment(start);
            total = search_.final_dispatch(pnew, stop);
        }
        if (stop.requested()) {
            last_y_.clear();  // not searched in full: the next call may retry it
        }
        // +inf also when stopped part-way.
        if (!(total < before - 1e-6)) {
            return;
        }
        write_back(model, vm, search_.commitment(), pnew, total, before);
    }

    // Descent, then the ILS when on and not the first descent. False when
    // stopped with costs invalid.
    bool commitment_search(StopRef stop) {
        search_.descend(stop);
        const bool first = !descended_;
        descended_ = descended_ || !stop.requested();  // a whole descent only
        if (cfg_.ils_seconds <= 0.0 || first) {
            return true;
        }
        if (!elite_y_.empty() && elite_total_ < search_.total_cost() - 1e-6 &&
            rng_.random() < cfg_.elite_prob) {
            search_.set_commitment(elite_y_);
            if (!search_.init_costs(stop)) {
                return false;
            }
        }
        const auto end = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                            std::chrono::duration<double>(cfg_.ils_seconds));
        const DeadlineStop ils_stop{stop, end};
        const auto t0 = Clock::now();
        search_.iterate(ils_stop, end);
        stats_.ils_nanos +=
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
        const double c = search_.total_cost();
        if (c < elite_total_) {
            elite_total_ = c;
            elite_y_ = search_.commitment();
        }
        return true;
    }

    void read_state(const Model& model) {
        const int n = inst_.n_units;
        const int periods = inst_.n_periods;
        y_in_.assign(n, CommitRow(periods, 0));
        p_in_.assign(n, std::vector<double>(periods, 0.0));
        for (int u = 0; u < n; ++u) {
            for (int t = 0; t < periods; ++t) {
                y_in_[u][t] = model.var(handle_to_var_id(y_handles_[u][t])).value > 0.5 ? 1 : 0;
                p_in_[u][t] = model.var(handle_to_var_id(p_handles_[u][t])).value;
            }
        }
    }

    void assign(Model& model, int u, int t, double y, double p) const {
        model.var_mut(handle_to_var_id(y_handles_[u][t])).value = y;
        model.var_mut(handle_to_var_id(p_handles_[u][t])).value = p;
    }

    // Writes (y, pnew) and keeps it only if the model holds it real-feasible
    // and below `before`; else restores the point read, value for value. Of
    // write-backs, only a kept one lowers best_seen_ (#452 review, High 2).
    void write_back(Model& model, ViolationManager& vm, const Commitment& y,
                    const CommitmentSearch::Dispatches& pnew, double total, double before) {
        const int n = inst_.n_units;
        const int periods = inst_.n_periods;
        std::vector<std::vector<double>> y_raw(n, std::vector<double>(periods));
        for (int u = 0; u < n; ++u) {
            for (int t = 0; t < periods; ++t) {
                y_raw[u][t] = model.var(handle_to_var_id(y_handles_[u][t])).value;
                assign(model, u, t, static_cast<double>(y[u][t]), pnew[t][u]);
            }
        }
        full_evaluate(model);
        vm.invalidate_cache();
        const double after = model.node_value(model.objective_id());
        if (vm.max_real_violation() <= cfg_.feasibility_tolerance && after < before) {
            best_seen_ = std::min(best_seen_, after);
            last_kept_total_ = total;
            ++stats_.improvements;
            return;
        }
        ++stats_.rejected;
        for (int u = 0; u < n; ++u) {
            for (int t = 0; t < periods; ++t) {
                assign(model, u, t, y_raw[u][t], p_in_[u][t]);
            }
        }
        full_evaluate(model);
        vm.invalidate_cache();
    }

    const UCInstance& inst_;
    std::vector<std::vector<int32_t>> y_handles_;  // [u][t], as UCModel::y
    std::vector<std::vector<int32_t>> p_handles_;  // [u][t], as UCModel::p
    UCHookConfig cfg_;
    CommitmentSearch search_;
    RNG rng_;
    Clock::time_point born_;
    UCHookStats stats_;
    double own_seconds_ = 0.0;
    double best_seen_ = kDispatchInf;
    double last_kept_total_ = kDispatchInf;
    bool descended_ = false;
    double elite_total_ = kDispatchInf;  // fast-dispatch cost of elite_y_
    Commitment elite_y_;
    Commitment y_in_, last_y_;
    std::vector<std::vector<double>> p_in_;  // [u][t], the point read
};

}  // namespace cbls::uc_chped
