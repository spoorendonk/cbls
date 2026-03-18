#pragma once

#include "data.h"
#include <algorithm>
#include <numeric>
#include <vector>
#include <cmath>

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

/// Incremental cost update: recompute cost for only the periods affected
/// by moving one outage from old_start to new_start.
/// Returns the delta in expected cost (new - old).
inline double incremental_cost_delta(
    const NuclearInstance& inst,
    const std::vector<int>& outage_starts,
    int outage_idx,
    int old_start,
    int new_start,
    const std::vector<int>& merit_order,
    int n_scenarios = -1,
    int scenario_offset = 0)
{
    if (n_scenarios < 0) n_scenarios = inst.n_scenarios;
    n_scenarios = std::min(n_scenarios, inst.n_scenarios - scenario_offset);

    int u = inst.outage_unit[outage_idx];
    int dur = inst.outage_duration[outage_idx];

    // Periods affected: union of [old_start, old_start+dur) and [new_start, new_start+dur)
    int t_min = std::min(old_start, new_start);
    int t_max = std::min(std::max(old_start + dur, new_start + dur), inst.n_periods);

    // Compute current availability for affected periods
    // We need to rebuild availability for these periods only
    auto avail_old = compute_availability(inst, outage_starts);

    // New availability: same as old but with outage moved
    // In affected periods, only unit u changes
    double delta = 0.0;
    for (int s = scenario_offset; s < scenario_offset + n_scenarios; ++s) {
        for (int t = t_min; t < t_max; ++t) {
            double old_cost = dispatch_period(inst, avail_old[t],
                                              inst.demand[s][t], merit_order);

            // Flip unit u availability for new schedule
            auto avail_new_t = avail_old[t];
            // Was unit u offline in old schedule at period t?
            bool was_offline = (t >= old_start && t < old_start + dur);
            // Will unit u be offline in new schedule at period t?
            bool will_be_offline = (t >= new_start && t < new_start + dur);

            if (was_offline != will_be_offline) {
                avail_new_t[u] = !was_offline;  // flip
            }

            double new_cost = dispatch_period(inst, avail_new_t,
                                              inst.demand[s][t], merit_order);
            delta += (new_cost - old_cost);
        }
    }

    return delta / n_scenarios;
}

/// Compute resource constraint violation penalty.
/// Penalizes exceeding max simultaneous outages per site.
inline double resource_violation_penalty(
    const NuclearInstance& inst,
    const std::vector<int>& outage_starts,
    double penalty_weight = 1e6)
{
    double violation = 0.0;

    // For each period, count outages per site
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

    return violation * penalty_weight;
}

}  // namespace nuclear_outage
}  // namespace cbls
