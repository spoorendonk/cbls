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
    double expected_cost(const std::vector<int>& outage_starts,
                         int n_scenarios, int scenario_offset) {
        int n_sc = std::min(n_scenarios, inst_.n_scenarios - scenario_offset);
        int P = inst_.n_periods;
        int U = inst_.n_units;

        // 5b: reset flat avail array with memset (all units online)
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

        // Dispatch across scenarios and periods
        double total_cost = 0.0;
        for (int s = scenario_offset; s < scenario_offset + n_sc; ++s) {
            const auto& demand_s = inst_.demand[s];
            for (int t = 0; t < P; ++t) {
                total_cost += dispatch_flat(t, demand_s[t]);
            }
        }

        return total_cost / n_sc;
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

    // 5e: cached per-site penalty contributions
    std::vector<double> site_capacity_penalty_;
    std::vector<double> site_spacing_penalty_;
    double total_penalty_ = 0.0;
    bool penalty_valid_ = false;

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
