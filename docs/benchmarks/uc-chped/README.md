# UC-CHPED: Unit Commitment with Combined Heat and Power Economic Dispatch

## Problem Description

Unit Commitment with Combined Heat and Power Economic Dispatch (UC-CHPED) is a power system optimization problem that combines two classical subproblems:

1. **Unit Commitment (UC):** Decide which generating units to turn on/off over a planning horizon (binary decisions), subject to minimum uptime/downtime constraints and startup costs.
2. **Economic Dispatch (ED):** For committed units, determine the optimal power output to meet demand at minimum fuel cost, where the cost function includes **valve-point effects** — sinusoidal ripples that model the non-smooth behavior of steam admission valves.

**Real-world motivation:** Day-ahead scheduling of thermal power plants. The system operator must commit enough generation capacity to serve hourly demand plus a spinning reserve margin, while minimizing total fuel cost and startup cost. Valve-point effects make the cost function non-convex and non-smooth, ruling out simple LP/QP dispatch.

## References

- **Pedroso, Kubo & Viana (2014)** — *"Pricing and unit commitment in combined energy and reserve markets using valve-point effects"*. Source of instance data and known MIP bounds (Table 2). Original code at `http://www.dcc.fc.up.pt/~jpp/code/valve/`.
- **Kazarlis, Bakirtzis & Petridis (1996)** — 10-unit UC system. Source of commitment parameters (min up/down times, startup costs, initial state).
- **Sinha, Chakrabarti & Chattopadhyay (2003)** — 13-unit system cost coefficients with valve-point effects.
- **Taipower 40-unit system** — Cost coefficients for the 40-unit base, extended to 100/200-unit instances.
- **Basu (2011), Vasebi, Fesanghary & Bathaee (2007)** — CHPED formulation background.

## Mathematical Model

### Sets and Indices

- $\mathcal{U} = \{1, \ldots, N\}$ — set of generating units
- $\mathcal{T} = \{1, \ldots, T\}$ — set of time periods (hours)

### Decision Variables

- $y_{u,t} \in \{0, 1\}$ — commitment status of unit $u$ at period $t$
- $p_{u,t} \in \mathbb{R}$ — power dispatch of unit $u$ at period $t$

### Objective

Minimize total fuel cost plus startup cost:

$$\min \sum_{u \in \mathcal{U}} \sum_{t \in \mathcal{T}} \left[ y_{u,t} \cdot F_u(p_{u,t}) + SC_{u,t} \right]$$

**Fuel cost with valve-point effects:**

$$F_u(P) = a_u + b_u P + c_u P^2 + |d_u \sin(e_u (P_u^{\min} - P))|$$

The sinusoidal term models valve-point loading effects — discontinuities in the heat-rate curve caused by steam admission valve openings.

**Startup cost:**

$$SC_{u,t} = \begin{cases}
A_u^{\text{hot}} \cdot su_{u,t} & \text{if unit was on within last } \tau_u^{\text{cold}} \text{ periods} \\
A_u^{\text{cold}} \cdot su_{u,t} & \text{otherwise}
\end{cases}$$

where $su_{u,t} = \max(0,\; y_{u,t} - y_{u,t-1})$ detects startups.

### Constraints

**Demand:**

$$\sum_{u \in \mathcal{U}} p_{u,t} \geq D_t \quad \forall t$$

**Spinning reserve:**

$$\sum_{u \in \mathcal{U}} P_u^{\max} \cdot y_{u,t} \geq D_t + R_t \quad \forall t$$

**Dispatch bounds:**

$$P_u^{\min} \cdot y_{u,t} \leq p_{u,t} \leq P_u^{\max} \cdot y_{u,t} \quad \forall u, t$$

**Minimum uptime:** If unit $u$ starts at period $t$, it must stay on for at least $\text{MinOn}_u$ periods:

$$y_{u,t} - y_{u,t-1} \leq y_{u,\tau} \quad \forall \tau \in [t+1, \min(t + \text{MinOn}_u - 1, T)]$$

**Minimum downtime:** If unit $u$ shuts down at period $t$, it must stay off for at least $\text{MinOff}_u$ periods:

$$y_{u,t-1} - y_{u,t} + y_{u,\tau} \leq 1 \quad \forall \tau \in [t+1, \min(t + \text{MinOff}_u - 1, T)]$$

**Initial conditions:** Units that were on/off before the horizon must respect their remaining min up/down time obligations.

## Instance Construction

### Base: Kazarlis 10-Unit UC Parameters

All instances share UC parameters (min uptime, min downtime, cold start threshold, startup costs, initial state) drawn from the Kazarlis 10-unit system, mapped cyclically to larger fleets.

### UCP_13UNIT

- **Cost coefficients:** Sinha et al. 13-unit system (with valve-point: $d_u, e_u \neq 0$)
- **UC parameters:** Mapped from Kazarlis via index `[0,1,2, 0,1,2,3,4,5,6,7,8,9]`
- **Demand:** 24-hour profile, peak 2670 MW
- **Reserve:** 10% of demand

### UCP_40UNIT

- **Cost coefficients:** Taipower 40-unit system (with valve-point)
- **UC parameters:** Kazarlis `i % 10`
- **Demand:** 24-hour profile, peak 11480 MW
- **Reserve:** 10% of demand

### UCP_100UNIT

- 2.5x scaling of the 40-unit system
- **Cost coefficients:** Cycle from 40-unit (`i % 40`)
- **UC parameters:** Kazarlis `i % 10`
- **Demand:** 2.5x the 40-unit demand (peak 28700 MW)

### UCP_200UNIT

- 5x scaling of the 40-unit system
- **Cost coefficients:** Cycle from 40-unit (`i % 40`)
- **UC parameters:** Kazarlis `i % 10`
- **Demand:** 5x the 40-unit demand (peak 57400 MW)

### Extended Horizons

Instances can be extended to 48h and 168h (1 week) via `extend_horizon()`, which repeats the 24h demand profile with a ±3% daily sinusoidal variation to avoid perfect periodicity.

## Instance Summary

| Instance | Units | Periods | Binary Vars | Continuous Vars | Total Vars | Known Bounds |
|----------|------:|--------:|------------:|----------------:|-----------:|:-------------|
| ucp13    |    13 |    1–24 |       13–312 |          13–312 |     26–624 | Pedroso 2014 |
| ucp40    |    40 |    1–24 |      40–960 |         40–960 |   80–1920 | Pedroso 2014 |
| ucp100   |   100 |   1–168 |    100–16800 |      100–16800 |  200–33600 | None         |
| ucp200   |   200 |   1–168 |    200–33600 |      200–33600 |  400–67200 | None         |

## Known Bounds

From Pedroso et al. (2014), Table 2 — MIP with 1-hour time limit:

### 13-Unit System

| Periods | Lower Bound | Upper Bound | Gap (%) |
|--------:|------------:|------------:|--------:|
|       1 |      11,701 |      11,701 |    0.00 |
|       3 |      38,850 |      38,850 |    0.00 |
|       6 |      91,406 |      91,784 |    0.41 |
|      12 |     231,587 |     232,537 |    0.41 |
|      24 |     464,053 |     466,187 |    0.46 |

### 40-Unit System

| Periods | Lower Bound | Upper Bound | Gap (%) |
|--------:|------------:|------------:|--------:|
|       1 |      55,645 |      55,645 |    0.00 |
|       3 |     178,396 |     178,547 |    0.08 |
|       6 |     416,108 |     416,606 |    0.12 |
|      12 |   1,112,371 |   1,113,801 |    0.13 |
|      24 |   2,235,971 |   2,238,504 |    0.11 |

## CBLS Model Implementation

The CBLS model (`uc_model.h`) maps the mathematical formulation to the solver's expression DAG:

- **`BoolVar`** for each $y_{u,t}$ (commitment decisions) — explored by the SA flip move
- **`FloatVar`** for each $p_{u,t}$ (dispatch levels) — explored by the SA float perturbation move, with a `FloatIntensifyHook` that refines dispatch once commitment is partially fixed
- **Expression DAG nodes** for the objective:
  - `sum`, `prod`, `pow_expr` for the quadratic cost terms
  - `sin_expr`, `abs_expr` for the valve-point term
  - `if_then_else` for hot/cold startup cost selection
  - `max_expr` over a lookback window to detect recent ON status
- **Constraint nodes** (penalty-based): demand, reserve, dispatch bounds, min uptime, min downtime, initial conditions — all expressed as `expr <= 0` violations
- **LNS** with 30% destroy rate for large neighborhood search over commitment variables

See [`benchmarks/uc-chped/uc_model.h`](../../../benchmarks/uc-chped/uc_model.h) for the full implementation.

## Reference Solver

The reference solver (`benchmarks/chped/reference_solve.py --uc`) uses PySCIPOpt to solve a MIP formulation:

- Binary variables for commitment ($y$), startup ($su$), and shutdown ($sd$)
- **Piecewise-linear approximation** of the valve-point cost function using 50 segments per unit with an incremental (SOS2-like) formulation
- Hot/cold startup cost modeled via auxiliary binary indicator variables
- Standard min uptime/downtime constraints

The PWL approximation is necessary because SCIP cannot directly handle the `|sin(...)|` term in a MIP. With 50 segments, the approximation error is negligible.

See [`benchmarks/chped/reference_solve.py`](../../../benchmarks/chped/reference_solve.py) for the full implementation.

## Results

Results from `comparison.csv`:

### 13-Unit System

| Instance | Periods | Method            | Objective  | LB      | Gap (%) | Time (s) |
|----------|--------:|-------------------|------------|--------:|--------:|---------:|
| ucp13    |       1 | Pedroso MIP (1hr) | 11,701     |  11,701 |    0.00 |        — |
| ucp13    |       3 | Pedroso MIP (1hr) | 38,850     |  38,850 |    0.00 |        — |
| ucp13    |       6 | Pedroso MIP (1hr) | 91,784     |  91,406 |    0.41 |        — |
| ucp13    |      12 | Pedroso MIP (1hr) | 232,537    | 231,587 |    0.41 |        — |
| ucp13    |      24 | Pedroso MIP (1hr) | 466,187    | 464,053 |    0.46 |        — |
| ucp13    |       1 | CBLS SA (10s)     | 13,864.8   |  11,701 |   18.49 |     10.0 |
| ucp13    |       3 | CBLS SA (30s)     | 48,531.9   |  38,850 |   24.92 |     30.0 |
| ucp13    |       6 | CBLS SA (60s)     | INFEASIBLE |  91,406 |       — |     60.0 |
| ucp13    |      12 | CBLS SA (120s)    | INFEASIBLE | 231,587 |       — |    123.5 |
| ucp13    |      24 | CBLS SA (300s)    | INFEASIBLE | 464,053 |       — |    308.9 |

### 40-Unit System

| Instance | Periods | Method            | Objective  |       LB | Gap (%) | Time (s) |
|----------|--------:|-------------------|------------|----------:|--------:|---------:|
| ucp40    |       1 | Pedroso MIP (1hr) | 55,645     |    55,645 |    0.00 |        — |
| ucp40    |       3 | Pedroso MIP (1hr) | 178,547    |   178,396 |    0.08 |        — |
| ucp40    |       6 | Pedroso MIP (1hr) | 416,606    |   416,108 |    0.12 |        — |
| ucp40    |      12 | Pedroso MIP (1hr) | 1,113,801  | 1,112,371 |    0.13 |        — |
| ucp40    |      24 | Pedroso MIP (1hr) | 2,238,504  | 2,235,971 |    0.11 |        — |
| ucp40    |       1 | CBLS SA (10s)     | 77,964.4   |    55,645 |   40.11 |     10.0 |
| ucp40    |       3 | CBLS SA (30s)     | INFEASIBLE |   178,396 |       — |     30.9 |
| ucp40    |       6 | CBLS SA (60s)     | INFEASIBLE |   416,108 |       — |     63.2 |
| ucp40    |      12 | CBLS SA (120s)    | INFEASIBLE | 1,112,371 |       — |    130.7 |
| ucp40    |      24 | CBLS SA (300s)    | INFEASIBLE | 2,235,971 |       — |    391.3 |

### Scaled Systems

| Instance | Periods | Method        | Result     | Time (s) |
|----------|--------:|---------------|------------|:--------:|
| ucp100   |       1 | CBLS SA (10s) | INFEASIBLE |     10.5 |
| ucp200   |       1 | CBLS SA (10s) | INFEASIBLE |     10.0 |

### Discussion

The SA-based solver currently struggles with feasibility on multi-period UC instances. The core challenge is the tight coupling between commitment decisions across time — min uptime/downtime constraints create long-range dependencies that are difficult for local search moves (single-variable flips) to satisfy simultaneously with demand and reserve constraints. The 1-period instances are feasible but show significant gaps (18–40%) versus MIP bounds, largely due to the valve-point non-convexity making it hard for float perturbation to find good dispatch points.

Key areas for improvement:
- **Commitment-aware moves:** Multi-variable moves that flip a unit's commitment across a block of consecutive periods, respecting min up/down constraints by construction
- **Feasibility-first strategies:** Relaxed initial states or greedy commitment initialization
- **Dispatch sub-optimization:** Once commitment is fixed, dispatch is a separable non-convex NLP per period that could be solved more aggressively

## Code Locations

| File | Description |
|------|-------------|
| [`benchmarks/instances/uc-chped/data.py`](../../../benchmarks/instances/uc-chped/data.py) | Instance definitions (Python) |
| [`benchmarks/instances/uc-chped/*.jsonl`](../../../benchmarks/instances/uc-chped/) | Serialized instances (JSONL) |
| [`benchmarks/instances/uc-chped/comparison.csv`](../../../benchmarks/instances/uc-chped/comparison.csv) | Results comparison table |
| [`benchmarks/uc-chped/data.h`](../../../benchmarks/uc-chped/data.h) | C++ data structures and JSONL loader |
| [`benchmarks/uc-chped/uc_model.h`](../../../benchmarks/uc-chped/uc_model.h) | CBLS model builder |
| [`benchmarks/uc-chped/uc_chped.cpp`](../../../benchmarks/uc-chped/uc_chped.cpp) | Benchmark runner executable |
| [`benchmarks/chped/reference_solve.py`](../../../benchmarks/chped/reference_solve.py) | Reference solver (SCIP MIP + scipy) |
| [`tests/test_uc_chped.cpp`](../../../tests/test_uc_chped.cpp) | Catch2 unit tests |
