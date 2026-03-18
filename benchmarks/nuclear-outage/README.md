# Benchmark 3: Nuclear Outage Scheduling (ROADEF/EURO 2010)

Schedule maintenance/refueling outages for EDF's French nuclear fleet over a
multi-year horizon. First-stage integer decisions (outage start weeks) are
coupled to second-stage continuous dispatch across hundreds of demand scenarios.

## Problem Description

The ROADEF/EURO 2010 Challenge posed the problem of scheduling planned
maintenance outages for Electricite de France's 56-unit nuclear fleet over a
5-year (~260 weekly periods) horizon. Outage dates determine which units are
available in each week; available capacity is then dispatched against uncertain
demand via merit-order. The objective is to minimize expected production cost
plus a penalty for unserved energy.

Key characteristics:
- **Two-stage stochastic:** integer outage dates (1st stage), continuous
  dispatch across demand scenarios (2nd stage)
- **Combinatorial 1st stage:** each outage must fall within a time window,
  outages on the same unit must not overlap, and site-level resource
  constraints limit simultaneous outages
- **Large 2nd stage:** up to 500 demand scenarios x 260 periods x 56 units
  (~7.3M implicit float variables)

## References

- **Competition:** ROADEF/EURO 2010 Challenge — "Scheduling of Outages of
  Nuclear Power Plants" (EDF).
  https://www.roadef.org/challenge/2010/en/

- **Problem specification:** `sujetEDFv22.pdf` (downloadable via
  `benchmarks/instances/nuclear-outage/download.sh`)

- **1st prize:** Jost, V. & Savourey, D. (2013). "A 0-1 integer linear
  programming approach to schedule outages of nuclear power plants."
  *Journal of Scheduling*, 16, 551-566.

- **2nd prize:** Gorge, A., Lisser, A., & Zorgati, R. (2012). "Stochastic
  nuclear outage scheduling problem." *European Journal of Operational
  Research*, 216(2), 344-354.

- **Survey:** Froger, A., Gendreau, M., Mendoza, J.E., Pinson, E., &
  Rousseau, L.M. (2016). "Maintenance scheduling in the electricity industry:
  A literature review." *European Journal of Operational Research*, 251(3),
  695-706.

## Mathematical Model

### Sets

| Symbol | Description |
|--------|-------------|
| U | Units (reactors), indexed by u |
| O | Outages, indexed by o |
| L | Sites (locations), indexed by l |
| T | Periods (weeks), indexed by t |
| Xi | Demand scenarios, indexed by xi |

### Parameters

| Symbol | Description |
|--------|-------------|
| cap_u | Capacity of unit u (MW) |
| fc_u | Fuel cost of unit u (EUR/MWh) |
| dur_o | Duration of outage o (weeks) |
| e_o, l_o | Earliest / latest start period for outage o |
| unit(o) | Unit that outage o belongs to |
| site(u) | Site that unit u belongs to |
| d(xi, t) | Demand in scenario xi, period t (MW) |
| gamma | Penalty for unserved energy (EUR/MWh) |
| delta | Minimum spacing between outages at the same site (weeks) |
| K_l | Maximum simultaneous outages at site l |

### Variables

| Symbol | Domain | Description |
|--------|--------|-------------|
| s_o | Z, [e_o, l_o] | Start period of outage o |
| p(u,t,xi) | R+ | Production of unit u in period t, scenario xi (implicit) |
| q(t,xi) | R+ | Unserved energy in period t, scenario xi (implicit) |

The dispatch variables p and q are not explicitly modeled -- they are
determined by merit-order dispatch given the outage schedule.

### Objective

Minimize expected dispatch cost plus unserved energy penalty:

```
min  (1/|Xi|) * sum_{xi} sum_t [ sum_u fc_u * p(u,t,xi)  +  gamma * q(t,xi) ]
```

### Constraints

**Unit non-overlap:** consecutive outages on the same unit do not overlap:
```
s_{o1} + dur_{o1} <= s_{o2}    for consecutive outages o1, o2 on the same unit
```

**Site capacity:** at most K_l outages active simultaneously per site:
```
|{o : site(unit(o)) = l, s_o <= t < s_o + dur_o}| <= K_l    for all l, t
```

**Site spacing:** minimum gap between outages at the same site:
```
s_{o1} + dur_{o1} + delta <= s_{o2}    or    s_{o2} + dur_{o2} + delta <= s_{o1}
    for all o1, o2 at the same site
```

**Demand satisfaction:** (enforced implicitly by dispatch)
```
sum_u p(u,t,xi) + q(t,xi) >= d(xi,t)    for all t, xi
```

## CBLS Model Architecture

### Why Two-Stage Decomposition

The full problem has ~7.3M float variables (56 units x 260 periods x 500
scenarios). Putting these in the DAG would be impractical. Instead, the CBLS
model uses a decomposition:

1. **DAG (1st stage):** One `IntVar` per outage for its start period.
   The SA loop explores the combinatorial space of outage schedules.

2. **Hook (2nd stage):** An `InnerSolverHook` reads the current outage starts,
   runs merit-order dispatch across demand scenarios, and writes the expected
   cost back into a constant objective node.

### DAG Structure (`nuclear_model.h`)

- **Variables:** `s[o]` -- one `IntVar` per outage, domain `[earliest, latest]`
- **Objective:** a constant node (value set by the hook, not by DAG evaluation)
- **Constraints:** unit non-overlap only (`s[o1] + dur[o1] - s[o2] <= 0`).
  Site constraints are evaluated as penalty in the hook to avoid
  O(n_periods * n_sites) DAG nodes.

### Dispatch Hook (`nuclear_hook.h`)

The `NuclearDispatchHook` implements `InnerSolverHook::solve()`:

1. Read current outage start values from the model
2. Compute merit-order dispatch cost over a rotating window of scenarios
3. Add resource violation penalty (site capacity + site spacing)
4. Write total cost into the objective constant node

**Scenario rotation:** To keep each move evaluation fast, only a subset of
scenarios (`scenarios_per_move`, default 50) is evaluated per move. The window
rotates through the full scenario set to avoid bias toward early scenarios.

### Dispatch Algorithm (`dispatch.h`)

Merit-order dispatch: units sorted by fuel cost (cheapest first), dispatched
greedily up to capacity. Unserved energy incurs a penalty. This is equivalent
to the LP relaxation of the dispatch subproblem when there are no minimum
power constraints binding.

## Instance Data

### Synthetic Instances

Three synthetic instances are provided for development and testing. They are
structurally faithful to the competition format but smaller:

| Instance | Units | Outages | Periods | Scenarios | Sites | Description |
|----------|-------|---------|---------|-----------|-------|-------------|
| mini | 10 | 10 | 52 | 20 | 3 | 1 year, 1 outage/unit |
| small | 20 | 30 | 104 | 50 | 10 | 2 years, some units have 2 outages |
| medium | 56 | 163 | 260 | 500 | 19 | 5 years, approximates competition A1 |

Generation logic is in `benchmarks/instances/nuclear-outage/data.py`. The unit
mix reflects the French fleet: N4 (1300 MW), P4/P'4 (1300 MW), CP0/CP1
(900 MW). Demand follows a seasonal cosine pattern (winter peak) with Gaussian
noise.

### Competition Instances

Official ROADEF 2010 instances (sets A and B) can be downloaded:

```bash
cd benchmarks/instances/nuclear-outage
bash download.sh
```

This fetches instance sets A1-A5 (qualification) and B1-B10 (final round),
plus the official checker and problem specification PDF.

### JSONL Format

Each instance is a single JSON object with fields:

```
name, n_units, n_periods, n_scenarios, n_outages, n_sites,
capacity[], min_power[], fuel_cost[], site[],
outage_unit[], outage_duration[], outage_earliest[], outage_latest[],
min_spacing_same_site, max_outages_per_site[],
demand[n_scenarios][n_periods],
penalty_unserved, known_bounds{}
```

## Results

Comparison on synthetic instances (from `comparison.csv`):

| Instance | Method | Objective | Gap vs Greedy |
|----------|--------|----------:|:-------------:|
| mini | Greedy | 7,711,651,923 | -- |
| mini | MIP (SAA 20sc) | 7,711,651,923 | 0.00% |
| mini | CBLS (SA+hook) | 7,648,754,817 | -0.82% |
| small | Greedy | 12,736,677,594 | -- |
| small | MIP (SAA 20sc) | 12,724,583,264 | -0.10% |
| small | CBLS (SA+hook) | 12,702,032,099 | -0.27% |

CBLS outperforms both the greedy heuristic and the MIP reference solver on
these synthetic instances. Note that these are not the published competition
instances, so direct comparison with competition results is not applicable.

## Usage

```bash
# Build
cmake -B build && cmake --build build

# Run CBLS benchmark (mini + small instances)
./build/cbls_nuclear_outage
./build/cbls_nuclear_outage benchmarks/instances/nuclear-outage  # explicit path

# Run reference solver
pip install pyscipopt numpy
python benchmarks/nuclear-outage/reference_solve.py              # mini instance
python benchmarks/nuclear-outage/reference_solve.py --instance small
python benchmarks/nuclear-outage/reference_solve.py --all        # all instances

# Run tests
ctest --test-dir build
```
