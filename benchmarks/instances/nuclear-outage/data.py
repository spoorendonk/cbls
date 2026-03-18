"""ROADEF 2010 Nuclear Outage Scheduling instance data.

Problem: Schedule refueling outages for EDF's French nuclear fleet over a
multi-year horizon (~260 weekly periods). Two-stage stochastic: integer
outage dates (first stage) coupled to continuous production dispatch across
multiple demand scenarios (second stage).

Reference: ROADEF/EURO 2010 Challenge, "Scheduling of Outages of Nuclear
Power Plants" (EDF).

Instance format follows the competition specification (sujetEDFv22.pdf).
Mini instances are synthetic but structurally faithful for dev/test.
"""

import random
import math

# ---------------------------------------------------------------------------
# Mini instance: 10 units, 52 periods (1 year), 20 scenarios
# Designed for fast development iteration and testing
# ---------------------------------------------------------------------------

def _make_mini():
    """Create a small synthetic instance for development."""
    n_units = 10
    n_periods = 52  # 1 year of weeks
    n_scenarios = 20

    random.seed(42)

    # Units: mix of large (900MW) and medium (600MW) reactors
    capacities = [900, 900, 900, 600, 600, 600, 600, 300, 300, 300]
    min_powers = [c * 0.3 for c in capacities]

    # Each unit has a fuel cost rate (EUR/MWh) — merit order determines dispatch
    # Lower cost = dispatched first
    fuel_costs = [8.0, 8.5, 9.0, 10.0, 10.5, 11.0, 11.5, 15.0, 16.0, 17.0]

    # Sites: group units into sites (affects resource constraints)
    sites = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # 3 sites

    # Each unit has 1 outage to schedule (simplified from real instances)
    # Outage durations in weeks
    outage_durations = [6, 7, 5, 6, 5, 7, 6, 4, 5, 4]

    # Time windows: [earliest_start, latest_start] for each outage
    outage_earliest = [1, 5, 10, 1, 8, 15, 20, 1, 12, 25]
    outage_latest = [30, 35, 40, 28, 33, 38, 42, 30, 36, 42]

    # Minimum spacing between outages at the same site (weeks)
    # This is a simplification — real problem has more complex resource constraints
    min_spacing_same_site = 3

    # Max simultaneous outages per site
    max_outages_per_site = [2, 2, 2]

    # Demand scenarios: base demand + noise
    # Base demand follows seasonal pattern (higher in winter)
    base_demand = []
    for w in range(n_periods):
        # Seasonal: peak in winter (weeks 0-12, 40-52), lower in summer
        season = math.cos(2 * math.pi * (w - 6) / 52)
        base = 35000 + 15000 * season  # MW, 35-50 GW range (France-scale)
        base_demand.append(base)

    # Generate demand scenarios
    demand = []
    for s in range(n_scenarios):
        scenario = []
        for w in range(n_periods):
            noise = random.gauss(0, 2000)  # +/- 2 GW noise
            scenario.append(max(0.0, base_demand[w] + noise))
        demand.append(scenario)

    # Penalty for unserved energy (very high — drives feasibility)
    penalty_unserved = 5000.0  # EUR/MWh

    return {
        "name": "mini",
        "n_units": n_units,
        "n_periods": n_periods,
        "n_scenarios": n_scenarios,
        "n_outages": n_units,  # 1 outage per unit
        "n_sites": 3,
        # Unit data
        "capacity": capacities,
        "min_power": min_powers,
        "fuel_cost": fuel_costs,
        "site": sites,
        # Outage data (indexed by outage, 1 outage per unit here)
        "outage_unit": list(range(n_units)),
        "outage_duration": outage_durations,
        "outage_earliest": outage_earliest,
        "outage_latest": outage_latest,
        # Resource constraints
        "min_spacing_same_site": min_spacing_same_site,
        "max_outages_per_site": max_outages_per_site,
        # Demand scenarios: [n_scenarios][n_periods]
        "demand": demand,
        # Cost parameters
        "penalty_unserved": penalty_unserved,
        # Best known solution (hand-tuned for synthetic instance)
        "known_bounds": {},
    }


# ---------------------------------------------------------------------------
# Small instance: 20 units, 104 periods (2 years), 50 scenarios
# ---------------------------------------------------------------------------

def _make_small():
    """Create a small but more realistic instance."""
    n_units = 20
    n_periods = 104  # 2 years
    n_scenarios = 50
    n_outages = 30  # some units have 2 outages over 2 years

    random.seed(123)

    # Mix of reactor types
    capacities = (
        [1300] * 4 +   # N4 reactors
        [900] * 8 +    # P4/P'4 reactors
        [600] * 8      # CP0/CP1 reactors
    )
    min_powers = [c * 0.3 for c in capacities]
    fuel_costs = (
        [7.0, 7.2, 7.5, 7.8] +           # N4: cheapest
        [9.0, 9.2, 9.5, 9.8, 10.0, 10.2, 10.5, 10.8] +  # P4
        [12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5]  # CP
    )
    sites = (
        [0, 0, 1, 1] +          # N4: 2 sites
        [2, 2, 3, 3, 4, 4, 5, 5] +  # P4: 4 sites
        [6, 6, 7, 7, 8, 8, 9, 9]    # CP: 4 sites
    )
    n_sites = 10

    # Outages: first 20 outages = 1 per unit, next 10 = second outage for
    # the first 10 units (these must be spaced from first outage)
    outage_unit = list(range(n_units)) + list(range(10))
    outage_duration = (
        [8, 7, 8, 6, 6, 7, 5, 6, 7, 5,
         6, 5, 6, 7, 5, 6, 4, 5, 6, 4] +
        [7, 6, 7, 5, 5, 6, 5, 5, 6, 4]
    )

    # Time windows
    outage_earliest = []
    outage_latest = []
    for i in range(n_outages):
        if i < 20:
            # First outage in first 80 weeks
            e = random.randint(1, 40)
            l = min(e + 30, n_periods - outage_duration[i])
        else:
            # Second outage must come after first (roughly)
            e = random.randint(50, 80)
            l = min(e + 20, n_periods - outage_duration[i])
        outage_earliest.append(e)
        outage_latest.append(l)

    min_spacing_same_site = 4
    max_outages_per_site = [1] * n_sites  # 1 at a time per site

    # Demand
    base_demand = []
    for w in range(n_periods):
        season = math.cos(2 * math.pi * (w - 6) / 52)
        base = 40000 + 18000 * season
        base_demand.append(base)

    demand = []
    for s in range(n_scenarios):
        scenario = []
        for w in range(n_periods):
            noise = random.gauss(0, 3000)
            scenario.append(max(0.0, base_demand[w] + noise))
        demand.append(scenario)

    return {
        "name": "small",
        "n_units": n_units,
        "n_periods": n_periods,
        "n_scenarios": n_scenarios,
        "n_outages": n_outages,
        "n_sites": n_sites,
        "capacity": capacities,
        "min_power": min_powers,
        "fuel_cost": fuel_costs,
        "site": sites,
        "outage_unit": outage_unit,
        "outage_duration": outage_duration,
        "outage_earliest": outage_earliest,
        "outage_latest": outage_latest,
        "min_spacing_same_site": min_spacing_same_site,
        "max_outages_per_site": max_outages_per_site,
        "demand": demand,
        "penalty_unserved": 5000.0,
        "known_bounds": {},
    }


# ---------------------------------------------------------------------------
# Medium instance: ~56 units, 260 periods, 500 scenarios
# Approximates competition instance A1
# ---------------------------------------------------------------------------

def _make_medium():
    """Create an instance approximating ROADEF A1 scale."""
    n_units = 56
    n_periods = 260  # 5 years
    n_scenarios = 500
    n_sites = 19

    random.seed(456)

    # French nuclear fleet composition (simplified)
    # 4 × N4 (1300MW), 20 × P4 (1300MW), 16 × CP1 (900MW), 16 × CP0 (900MW)
    capacities = (
        [1300] * 4 +    # N4
        [1300] * 20 +   # P4/P'4
        [900] * 16 +    # CP1
        [900] * 16      # CP0
    )
    min_powers = [c * 0.3 for c in capacities]

    # Fuel costs: slight variation within type
    fuel_costs = []
    for i, c in enumerate(capacities):
        base = 7.0 if c == 1300 else 10.0
        fuel_costs.append(base + random.uniform(0, 3.0))

    # Sites: group 2-4 units per site
    sites = []
    site_id = 0
    units_assigned = 0
    units_per_site = []
    while units_assigned < n_units:
        n = min(random.choice([2, 3, 4]), n_units - units_assigned)
        for _ in range(n):
            sites.append(site_id)
        units_per_site.append(n)
        units_assigned += n
        site_id += 1
    n_sites = site_id

    # ~3 outages per unit over 5 years
    n_outages_per_unit = [random.choice([2, 3, 3, 4]) for _ in range(n_units)]
    outage_unit = []
    outage_duration = []
    outage_earliest = []
    outage_latest = []

    for u in range(n_units):
        spacing = n_periods // (n_outages_per_unit[u] + 1)
        for j in range(n_outages_per_unit[u]):
            outage_unit.append(u)
            dur = random.randint(4, 10)
            outage_duration.append(dur)
            center = spacing * (j + 1)
            e = max(1, center - 20)
            l = min(center + 20, n_periods - dur)
            outage_earliest.append(e)
            outage_latest.append(l)

    n_outages = len(outage_unit)

    min_spacing_same_site = 3
    max_outages_per_site = [min(2, n) for n in units_per_site]

    # Demand scenarios
    base_demand = []
    for w in range(n_periods):
        season = math.cos(2 * math.pi * ((w % 52) - 6) / 52)
        base = 45000 + 20000 * season
        base_demand.append(base)

    demand = []
    for s in range(n_scenarios):
        scenario = []
        for w in range(n_periods):
            noise = random.gauss(0, 4000)
            scenario.append(max(0.0, base_demand[w] + noise))
        demand.append(scenario)

    return {
        "name": "medium",
        "n_units": n_units,
        "n_periods": n_periods,
        "n_scenarios": n_scenarios,
        "n_outages": n_outages,
        "n_sites": n_sites,
        "capacity": capacities,
        "min_power": min_powers,
        "fuel_cost": fuel_costs,
        "site": sites,
        "outage_unit": outage_unit,
        "outage_duration": outage_duration,
        "outage_earliest": outage_earliest,
        "outage_latest": outage_latest,
        "min_spacing_same_site": min_spacing_same_site,
        "max_outages_per_site": max_outages_per_site,
        "demand": demand,
        "penalty_unserved": 5000.0,
        "known_bounds": {},
    }


MINI = _make_mini()
SMALL = _make_small()
MEDIUM = _make_medium()
