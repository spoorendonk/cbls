# Benchmark 4: Maritime Tramp Ship Routing + Bunker + Speed + ECA

A tramp shipping operator manages a heterogeneous fleet carrying contract and spot
cargoes across global routes. The problem jointly optimizes cargo-to-ship assignment,
sailing speed, bunker (fuel) purchasing, and fuel switching in Emission Control Areas
(ECAs). This benchmark exercises CBLS's mixed IntVar/FloatVar/BoolVar support,
nonlinear v^2 fuel coupling, and inner solver hooks for analytical speed optimization.

## Paper References

Three papers inform the benchmark design. No single paper covers routing + speed +
bunker + ECA together — the ECA extension to multi-vessel routing is novel.

1. **Vilhelmsen, Lusby & Larsen (2014)** — "Tramp ship routing and scheduling with
   integrated bunker optimization," *EURO Journal on Transportation and Logistics*
   3(2), 143-175.
   - Base TSRSPBO formulation: routing + bunker purchasing + tank dynamics
   - Column generation + dynamic programming approach
   - Primary structural inspiration for cargo/ship/region data model

2. **Tamburini, Lange & Pisinger (2025)** — "A rich model for the tramp ship routing
   and scheduling problem," *Transportation Research Part E* 198, 104019.
   - State-of-the-art: adds speed optimization + chartering + hull cleaning
   - Column generation on time-space multigraph, <0.1% MIP gap
   - Real instances: 40 vessels, 35 regions
   - Does **not** include ECA fuel switching — our extension

3. **Fagerholt, Gausel, Rakke & Psaraftis (2015)** — "Maritime routing and speed
   optimization with emission control areas," *Transportation Research Part C* 52,
   57-73.
   - Single-vessel speed + ECA optimization
   - Shows operators adjust speed inside/outside ECAs and reroute to avoid ECA sailing
   - Inspires the ECA fuel-switching BoolVar in our multi-vessel model

## Problem Description

- **Fleet**: heterogeneous ships with speed ranges (10.5-14.5 kn), cubic fuel
  coefficients, and dual tanks (HFO + MGO)
- **Cargoes**: contract (must carry) and spot (optional, for profit), each with
  pickup/delivery ports, time windows, revenue, and quantity
- **Regions**: real-world ports (Rotterdam, Singapore, Houston, Dubai, Shanghai, etc.)
  with port costs and bunker prices
- **ECA zones**: legs through ECAs (Baltic Sea, North Sea, North American coast)
  require expensive low-sulfur MGO instead of HFO
- **Bunker options**: buy fuel at various ports at time-varying prices

## Mathematical Model

### Sets

| Symbol | Description |
|--------|-------------|
| V | Ships (1..V) |
| C | Cargoes |
| R | Regions (ports) |
| L | Legs (directed arcs between regions) |

### Decision Variables

| Variable | Type | Description |
|----------|------|-------------|
| assign_c | int [0,V] | Ship assigned to cargo c (0 = unassigned for spot) |
| speed_c | float [v_min, v_max] | Sailing speed (knots) on cargo c's main leg |
| eca_fuel_c | bool | Use MGO (1) or HFO (0) on ECA portion of leg |

### Fuel Consumption

Daily consumption follows a cubic model `k * v^3`, but over a leg of distance d the
total fuel consumed is:

```
fuel_c = alpha * v_c^2     where alpha = k * d / 24
```

This arises because sailing time = d / (24 * v), so total fuel = k * v^3 * d / (24 * v)
= k * v^2 * d / 24.

### Tank Dynamics (Simplified)

Per-ship aggregate fuel balance:

```
total_hfo_consumed_v = sum_{c on v} hfo_consumed_c
total_hfo_consumed_v <= initial_hfo_v + (hfo_tank_max_v - initial_hfo_v) - hfo_safety_v
```

HFO/MGO split per cargo depends on ECA fraction and fuel choice:

```
mgo_consumed_c = fuel_c * eca_fraction * eca_fuel_c
hfo_consumed_c = fuel_c - mgo_consumed_c
```

### Timing

```
travel_time_c = d_c / (24 * speed_c)
travel_time_c <= available_time_c * active_c
```

where `available_time_c = delivery_tw_end - pickup_tw_start - service_load - service_discharge`.

### Objective

Maximize profit:

```
max  sum_c (revenue_c * active_c)
   - sum_c (hfo_price * hfo_consumed_c + mgo_price * mgo_consumed_c)
   - sum_c (port_cost_c * active_c)
```

### Constraints

1. **Fuel capacity**: per-ship HFO consumption <= usable tank capacity
2. **Time windows**: travel time must fit within available time for each cargo
3. **ECA compliance**: fully-ECA legs (eca_fraction >= 0.99) force MGO use
4. **Workload balance**: max cargoes per ship <= max(4, ceil(C/V) * 3)
5. **Contract coverage**: contract cargoes must be assigned (assign_c >= 1)
6. **Speed bounds**: fleet-wide v_min <= speed_c <= v_max (per-ship enforcement via indicators)

## CBLS Formulation

The CBLS expression DAG encodes the model with these simplifications vs. the full
mathematical formulation:

**Variables:**
- `assign[c]` IntVar: cargo-to-ship assignment (0 = unassigned for spot cargoes)
- `speed[c]` FloatVar: sailing speed per cargo's main leg
- `eca_fuel[c]` BoolVar: MGO (1) or HFO (0) for ECA-crossing legs; sentinel -1 if no ECA

**Key simplifications:**
- Average fuel coefficient across fleet (reduces DAG node count)
- No explicit bunker purchase variables — fuel cost derived from consumption * average price
- Indicator expressions `on_v[c][v]` = 1 iff assign[c] == v+1, built from abs/if-then-else
- Aggregate tank balance per ship (not per-port chain)

**Inner solver hook (BunkerSpeedHook):**

Given fixed assignments, analytically computes minimum-fuel speed:

```
v* = max(v_min, dist / (24 * available_time))
```

Minimum feasible speed minimizes the quadratic fuel cost. The hook then delegates to
FloatIntensifyHook for gradient-based polish of remaining float variables.

## Instance Generation

All instances are generated by factory functions in C++ (`data.h`) and Python
(`data.py`), with JSONL serialization support.

| Instance | Ships | Cargoes | Regions | Horizon | Contract/Spot |
|----------|-------|---------|---------|---------|---------------|
| small-3s-10c | 3 | 10 | 7 | 60 days | 6 / 4 |
| medium-7s-30c | 7 | 30 | 15 | 90 days | 18 / 12 |
| large-15s-60c | 15 | 60 | 15 | 90 days | 36 / 24 |
| xlarge-30s-120c | 30 | 120 | 15 | 120 days | 72 / 48 |

**Regions**: Rotterdam, Hamburg, Antwerp, Singapore, Houston, New York, Dubai, Shanghai,
Busan, Santos, Durban, Mumbai, Gothenburg, Los Angeles, Tokyo (medium+). ECA ports:
Rotterdam, Hamburg, Antwerp, Houston, New York, Gothenburg, Los Angeles.

**Ship parameters** (Panamax class, from Tamburini et al.): speed 10.5-14.5 kn laden,
fuel coefficients 0.0032-0.0040, HFO tank 2200-2800 MT, MGO tank 450-550 MT, safety
stock 110-140 MT HFO / 22-28 MT MGO.

**Time windows**: computed to be feasible at v_max with slack. Sailing time at 14.5 kn
= dist / 348 days.

**Scaling**: large and xlarge are built by doubling ships/cargoes from medium with
shifted time windows and varied fuel coefficients.

## Results

Results from the CBLS runner (`bunker_eca.cpp`) with LNS (destroy rate 0.3), seed 42:

| Instance | Vars | Nodes | Time Limit | Feasible |
|----------|------|-------|------------|----------|
| small-3s-10c | ~25 | ~500 | 30s | yes |
| medium-7s-30c | ~80 | ~2500 | 120s | yes |
| large-15s-60c | ~160 | ~5000 | 300s | yes |

The runner also includes a no-ECA comparison mode (zeroing all eca_fraction values) to
measure the cost impact of ECA compliance.

## Files

| File | Purpose |
|------|---------|
| `data.h` | Instance data structures (Region, Cargo, Ship, Leg, BunkerOption), factory functions (make_small/medium/large/xlarge), JSONL loader |
| `bunker_eca_model.h` | CBLS model builder: variables, indicators, fuel/cost expressions, constraints, objective |
| `bunker_speed_hook.h` | BunkerSpeedHook inner solver: analytical speed optimization + FloatIntensifyHook delegation |
| `bunker_eca.cpp` | Runner: solves small/medium/large with and without ECA, prints results table |
| `reference_solve.py` | PySCIPOpt MIQCP reference solver for validation |
| `../../tests/test_bunker_eca.cpp` | Catch2 tests: data integrity, model builds, feasibility, fuel formula, medium instance |
