#pragma once

#include "data.h"
#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

namespace cbls {
namespace nuclear_outage {

/// Compute availability matrix: available[period] = bitset of which units are online.
/// A unit is offline during [outage_start, outage_start + duration).
inline std::vector<std::vector<bool>> compute_availability(
    const NuclearInstance& inst,
    const std::vector<int>& outage_starts)
{
    // available[period][unit] = true if unit is online
    std::vector<std::vector<bool>> avail(inst.n_periods,
        std::vector<bool>(inst.n_units, true));

    for (int o = 0; o < inst.n_outages; ++o) {
        int u = inst.outage_unit[o];
        int start = outage_starts[o];
        int end = std::min(start + inst.outage_duration[o], inst.n_periods);
        for (int t = start; t < end; ++t) {
            avail[t][u] = false;
        }
    }
    return avail;
}

/// Merit-order dispatch for a single period.
/// Returns the generation cost for serving the given demand.
/// Units are dispatched cheapest-first up to their capacity.
/// Unserved energy incurs a penalty.
inline double dispatch_period(
    const NuclearInstance& inst,
    const std::vector<bool>& unit_available,
    double demand,
    const std::vector<int>& merit_order)  // pre-sorted by fuel_cost
{
    double cost = 0.0;
    double remaining = demand;

    for (int u : merit_order) {
        if (remaining <= 0.0) break;
        if (!unit_available[u]) continue;

        double gen = std::min(inst.capacity[u], remaining);
        cost += gen * inst.fuel_cost[u];
        remaining -= gen;
    }

    // Penalty for unserved energy
    if (remaining > 0.0) {
        cost += remaining * inst.penalty_unserved;
    }
    return cost;
}

/// Expected cost across all scenarios for a given outage schedule.
/// This is the main evaluation function called by the hook.
inline double expected_cost(
    const NuclearInstance& inst,
    const std::vector<int>& outage_starts,
    int n_scenarios = -1,  // -1 = all scenarios
    int scenario_offset = 0)
{
    if (n_scenarios < 0) n_scenarios = inst.n_scenarios;
    n_scenarios = std::min(n_scenarios, inst.n_scenarios - scenario_offset);

    auto avail = compute_availability(inst, outage_starts);

    // Pre-compute merit order (sorted by fuel cost, ascending)
    std::vector<int> merit_order(inst.n_units);
    std::iota(merit_order.begin(), merit_order.end(), 0);
    std::sort(merit_order.begin(), merit_order.end(),
              [&](int a, int b) { return inst.fuel_cost[a] < inst.fuel_cost[b]; });

    double total_cost = 0.0;
    for (int s = scenario_offset; s < scenario_offset + n_scenarios; ++s) {
        for (int t = 0; t < inst.n_periods; ++t) {
            total_cost += dispatch_period(inst, avail[t],
                                          inst.demand[s][t], merit_order);
        }
    }

    return total_cost / n_scenarios;
}

/// Compute resource constraint violation penalty.
/// Penalizes: (1) exceeding max simultaneous outages per site,
///            (2) violating min_spacing_same_site between outages at the same site.
inline double resource_violation_penalty(
    const NuclearInstance& inst,
    const std::vector<int>& outage_starts,
    double penalty_weight = 1e6)
{
    double violation = 0.0;

    // (1) Max simultaneous outages per site
    for (int t = 0; t < inst.n_periods; ++t) {
        std::vector<int> site_count(inst.n_sites, 0);
        for (int o = 0; o < inst.n_outages; ++o) {
            int start = outage_starts[o];
            if (t >= start && t < start + inst.outage_duration[o]) {
                site_count[inst.site[inst.outage_unit[o]]]++;
            }
        }
        for (int s = 0; s < inst.n_sites; ++s) {
            int excess = site_count[s] - inst.max_outages_per_site[s];
            if (excess > 0) {
                violation += excess;
            }
        }
    }

    // (2) Min spacing between outages at the same site
    // Group outages by site
    std::vector<std::vector<int>> site_outages(inst.n_sites);
    for (int o = 0; o < inst.n_outages; ++o) {
        site_outages[inst.site[inst.outage_unit[o]]].push_back(o);
    }
    for (int s = 0; s < inst.n_sites; ++s) {
        auto& outages = site_outages[s];
        if (outages.size() < 2) continue;
        // Sort by start time
        std::sort(outages.begin(), outages.end(),
                  [&](int a, int b) { return outage_starts[a] < outage_starts[b]; });
        for (size_t i = 0; i + 1 < outages.size(); ++i) {
            int o1 = outages[i];
            int o2 = outages[i + 1];
            int end1 = outage_starts[o1] + inst.outage_duration[o1];
            int gap = outage_starts[o2] - end1;
            if (gap < inst.min_spacing_same_site) {
                violation += (inst.min_spacing_same_site - gap);
            }
        }
    }

    return violation * penalty_weight;
}

// -------------------------------------------------------------------------
// DispatchEvaluator: allocation-free dispatch + resource penalty evaluation
// Pre-allocates all scratch buffers; uses flat contiguous availability array;
// caches merit order; supports delta resource penalty.
// -------------------------------------------------------------------------

class DispatchEvaluator {
public:
    explicit DispatchEvaluator(const NuclearInstance& inst)
        : inst_(inst),
          avail_(inst.n_periods * inst.n_units, 1),
          merit_order_(inst.n_units),
          site_count_(inst.n_sites, 0),
          site_outages_(inst.n_sites),
          outage_site_(inst.n_outages)
    {
        // 5c: cache merit order once (fuel costs never change)
        std::iota(merit_order_.begin(), merit_order_.end(), 0);
        std::sort(merit_order_.begin(), merit_order_.end(),
                  [&](int a, int b) { return inst.fuel_cost[a] < inst.fuel_cost[b]; });

        // Pre-compute outage→site mapping
        for (int o = 0; o < inst.n_outages; ++o) {
            outage_site_[o] = inst.site[inst.outage_unit[o]];
        }

        // Build static site→outages grouping (indices only, sorted later by start)
        for (int o = 0; o < inst.n_outages; ++o) {
            site_outages_[outage_site_[o]].push_back(o);
        }
    }

    /// Compute expected dispatch cost. No heap allocation on the hot path.
    /// If changed_outages is non-empty and the scenario window matches the cache,
    /// uses incremental dispatch (5d): only re-dispatches affected periods.
    double expected_cost(const std::vector<int>& outage_starts,
                         int n_scenarios, int scenario_offset,
                         const std::vector<int>& changed_outages = {}) {
        int n_sc = std::min(n_scenarios, inst_.n_scenarios - scenario_offset);

        // 5d: try incremental path if cached state matches.
        // Fall back to full if estimated affected periods > half the total.
        if (!changed_outages.empty() && dispatch_cached_ &&
            cached_n_sc_ == n_sc && cached_offset_ == scenario_offset) {
            int est_affected = 0;
            for (int o : changed_outages) {
                est_affected += 2 * inst_.outage_duration[o];
            }
            if (est_affected >= inst_.n_periods) {
                return full_dispatch(outage_starts, n_sc, scenario_offset);
            }
            return multi_incremental_dispatch(outage_starts, changed_outages);
        }

        return full_dispatch(outage_starts, n_sc, scenario_offset);
    }

    /// Find which outages changed vs cached state.
    /// Returns number of changed outages (0 if cache invalid). Fills changed_outages.
    int find_changes(const std::vector<int>& outage_starts,
                     std::vector<int>& changed_outages) const {
        changed_outages.clear();
        if (!dispatch_cached_) return 0;
        for (int o = 0; o < inst_.n_outages; ++o) {
            if (outage_starts[o] != cached_starts_[o]) {
                changed_outages.push_back(o);
            }
        }
        return static_cast<int>(changed_outages.size());
    }

    /// Compute resource violation penalty using pre-allocated scratch.
    /// If changed_outage >= 0, only rechecks the affected site (5e: delta).
    double resource_penalty(const std::vector<int>& outage_starts,
                            double penalty_weight = 1e6,
                            int changed_outage = -1) {
        if (changed_outage >= 0) {
            return delta_resource_penalty(outage_starts, penalty_weight, changed_outage);
        }
        return full_resource_penalty(outage_starts, penalty_weight);
    }

private:
    const NuclearInstance& inst_;

    // 5b: flat contiguous availability array [period * n_units + unit]
    std::vector<uint8_t> avail_;

    // 5c: cached merit order (sorted once at construction)
    std::vector<int> merit_order_;

    // 5a: pre-allocated scratch buffers
    std::vector<int> site_count_;

    // 5e: static site→outage grouping (outage indices, re-sorted by start on use)
    std::vector<std::vector<int>> site_outages_;
    std::vector<int> outage_site_;  // outage → site

    // 5d: cached dispatch state for incremental evaluation
    std::vector<double> cost_per_period_;   // [n_periods] total cost across cached scenarios
    std::vector<int> cached_starts_;        // outage starts for the cached avail/cost state
    double cached_total_ = 0.0;             // sum of cost_per_period_
    int cached_n_sc_ = 0;                   // scenario count for cached state
    int cached_offset_ = 0;                 // scenario offset for cached state
    bool dispatch_cached_ = false;

    // 5e: cached per-site penalty contributions
    std::vector<double> site_capacity_penalty_;
    std::vector<double> site_spacing_penalty_;
    double total_penalty_ = 0.0;
    bool penalty_valid_ = false;

    /// Full dispatch: reset avail, mark all outages, dispatch all periods.
    double full_dispatch(const std::vector<int>& outage_starts,
                         int n_sc, int scenario_offset) {
        int P = inst_.n_periods;
        int U = inst_.n_units;

        // Reset flat avail array
        std::memset(avail_.data(), 1, P * U);

        // Mark outage periods offline
        for (int o = 0; o < inst_.n_outages; ++o) {
            int u = inst_.outage_unit[o];
            int start = outage_starts[o];
            int end = std::min(start + inst_.outage_duration[o], P);
            for (int t = start; t < end; ++t) {
                avail_[t * U + u] = 0;
            }
        }

        // Dispatch across scenarios and periods, caching per-period costs
        cost_per_period_.resize(P);
        cached_total_ = 0.0;
        for (int t = 0; t < P; ++t) {
            double period_cost = 0.0;
            for (int s = scenario_offset; s < scenario_offset + n_sc; ++s) {
                period_cost += dispatch_flat(t, inst_.demand[s][t]);
            }
            cost_per_period_[t] = period_cost;
            cached_total_ += period_cost;
        }

        // Save cache state
        cached_starts_ = outage_starts;
        cached_n_sc_ = n_sc;
        cached_offset_ = scenario_offset;
        dispatch_cached_ = true;

        return cached_total_ / n_sc;
    }

    /// 5d: Multi-outage incremental dispatch — apply each change sequentially.
    double multi_incremental_dispatch(const std::vector<int>& outage_starts,
                                       const std::vector<int>& changed_outages) {
        for (int o : changed_outages) {
            incremental_dispatch_one(outage_starts, o);
        }
        return cached_total_ / cached_n_sc_;
    }

    /// 5d: Incremental dispatch for a single outage change.
    void incremental_dispatch_one(const std::vector<int>& outage_starts,
                                  int changed_outage) {
        int P = inst_.n_periods;
        int U = inst_.n_units;
        int u = inst_.outage_unit[changed_outage];
        int dur = inst_.outage_duration[changed_outage];
        int old_start = cached_starts_[changed_outage];
        int new_start = outage_starts[changed_outage];

        if (old_start == new_start) return;  // no actual change

        // Toggle avail: restore old position (unit back online)
        int old_end = std::min(old_start + dur, P);
        for (int t = old_start; t < old_end; ++t) {
            avail_[t * U + u] = 1;
        }
        // Toggle avail: mark new position (unit offline)
        int new_end = std::min(new_start + dur, P);
        for (int t = new_start; t < new_end; ++t) {
            avail_[t * U + u] = 0;
        }

        // Collect affected periods (union of old and new ranges)
        int lo = std::min(old_start, new_start);
        int hi = std::max(old_end, new_end);

        // Re-dispatch only affected periods
        for (int t = lo; t < hi; ++t) {
            double period_cost = 0.0;
            for (int s = cached_offset_; s < cached_offset_ + cached_n_sc_; ++s) {
                period_cost += dispatch_flat(t, inst_.demand[s][t]);
            }
            cached_total_ -= cost_per_period_[t];
            cost_per_period_[t] = period_cost;
            cached_total_ += period_cost;
        }

        cached_starts_[changed_outage] = new_start;
    }

    /// Merit-order dispatch on flat avail array. No allocation.
    double dispatch_flat(int period, double demand) const {
        double cost = 0.0;
        double remaining = demand;
        int U = inst_.n_units;
        const uint8_t* row = avail_.data() + period * U;

        for (int u : merit_order_) {
            if (remaining <= 0.0) break;
            if (!row[u]) continue;
            double gen = std::min(inst_.capacity[u], remaining);
            cost += gen * inst_.fuel_cost[u];
            remaining -= gen;
        }

        if (remaining > 0.0) {
            cost += remaining * inst_.penalty_unserved;
        }
        return cost;
    }

    /// Full resource penalty computation (first call or fallback).
    double full_resource_penalty(const std::vector<int>& outage_starts,
                                 double penalty_weight) {
        int n_sites = inst_.n_sites;
        site_capacity_penalty_.assign(n_sites, 0.0);
        site_spacing_penalty_.assign(n_sites, 0.0);

        // (1) Max simultaneous outages per site
        for (int t = 0; t < inst_.n_periods; ++t) {
            std::fill(site_count_.begin(), site_count_.end(), 0);
            for (int o = 0; o < inst_.n_outages; ++o) {
                int start = outage_starts[o];
                if (t >= start && t < start + inst_.outage_duration[o]) {
                    site_count_[outage_site_[o]]++;
                }
            }
            for (int s = 0; s < n_sites; ++s) {
                int excess = site_count_[s] - inst_.max_outages_per_site[s];
                if (excess > 0) {
                    site_capacity_penalty_[s] += excess;
                }
            }
        }

        // (2) Min spacing
        for (int s = 0; s < n_sites; ++s) {
            auto& outages = site_outages_[s];
            if (outages.size() < 2) continue;
            std::sort(outages.begin(), outages.end(),
                      [&](int a, int b) { return outage_starts[a] < outage_starts[b]; });
            for (size_t i = 0; i + 1 < outages.size(); ++i) {
                int o1 = outages[i];
                int o2 = outages[i + 1];
                int end1 = outage_starts[o1] + inst_.outage_duration[o1];
                int gap = outage_starts[o2] - end1;
                if (gap < inst_.min_spacing_same_site) {
                    site_spacing_penalty_[s] += (inst_.min_spacing_same_site - gap);
                }
            }
        }

        total_penalty_ = 0.0;
        for (int s = 0; s < n_sites; ++s) {
            total_penalty_ += site_capacity_penalty_[s] + site_spacing_penalty_[s];
        }
        penalty_valid_ = true;
        return total_penalty_ * penalty_weight;
    }

    /// 5e: Delta resource penalty — only recompute site of changed outage.
    double delta_resource_penalty(const std::vector<int>& outage_starts,
                                  double penalty_weight,
                                  int changed_outage) {
        if (!penalty_valid_) {
            return full_resource_penalty(outage_starts, penalty_weight);
        }

        int site = outage_site_[changed_outage];

        // Subtract old site contribution
        total_penalty_ -= site_capacity_penalty_[site] + site_spacing_penalty_[site];

        // Recompute capacity penalty for this site only
        site_capacity_penalty_[site] = 0.0;
        for (int t = 0; t < inst_.n_periods; ++t) {
            int count = 0;
            for (int o : site_outages_[site]) {
                int start = outage_starts[o];
                if (t >= start && t < start + inst_.outage_duration[o]) {
                    count++;
                }
            }
            int excess = count - inst_.max_outages_per_site[site];
            if (excess > 0) {
                site_capacity_penalty_[site] += excess;
            }
        }

        // Recompute spacing penalty for this site only
        site_spacing_penalty_[site] = 0.0;
        auto& outages = site_outages_[site];
        if (outages.size() >= 2) {
            std::sort(outages.begin(), outages.end(),
                      [&](int a, int b) { return outage_starts[a] < outage_starts[b]; });
            for (size_t i = 0; i + 1 < outages.size(); ++i) {
                int o1 = outages[i];
                int o2 = outages[i + 1];
                int end1 = outage_starts[o1] + inst_.outage_duration[o1];
                int gap = outage_starts[o2] - end1;
                if (gap < inst_.min_spacing_same_site) {
                    site_spacing_penalty_[site] += (inst_.min_spacing_same_site - gap);
                }
            }
        }

        // Add new site contribution
        total_penalty_ += site_capacity_penalty_[site] + site_spacing_penalty_[site];
        return total_penalty_ * penalty_weight;
    }
};

}  // namespace nuclear_outage
}  // namespace cbls
