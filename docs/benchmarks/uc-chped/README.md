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

- **`BoolVar`** for each $y_{u,t}$ (commitment decisions) — driven by Feasibility Jump's bool flips
- **`FloatVar`** for each $p_{u,t}$ (dispatch levels) — driven by FJ's Newton-derived float jump values, with a `FloatIntensifyHook` that refines dispatch once commitment is partially fixed
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

**These are archived pre-#103 numbers, not the current contents of
`comparison.csv`.** The CBLS SA rows below were produced under the engine's
old `1e-9` feasibility tolerance and under the simulated-annealing search that
preceded the ViolationLS port; they were removed from `comparison.csv` because
the file recorded no tolerance, which left them uninterpretable once the
default moved to `1e-6`. They are kept here as the historical record and remain
in git history. `comparison.csv` currently carries the 10 cited Pedroso
reference rows only; regenerating the measured rows is tracked in issue #131.

The table now has a generator — `benchmarks/uc-chped/uc_chped.cpp` writes it,
stating the feasibility tolerance explicitly and recording it, the seed, the
time budget and the engine commit on every measured row:

```bash
./build/cbls_uc_chped --out benchmarks/instances/uc-chped/comparison.csv \
                      --commit "$(git rev-parse --short=7 HEAD)" --verify
```

`--feas-tol T` overrides the tolerance, `--time-limit S` overrides the
per-horizon budget map, `--seed N` the seed, and `--instance NAME` restricts the
roster. The last two are refused outright when the resolved `--out` path *is*
the published table above — a partial roster or a shortened budget cannot become
the published result, whether the path was defaulted or spelled out. Writing
that file at all requires `--commit`, so a published row always names the engine
it measured. Point `--out` at a scratch path and none of this applies — but see
the anytime-trace section below, where `--trace` is held to the same rule
against both published names. Note that `feasible` is
the engine's verdict at the row's `feas_tol` while `verified` is an independent
re-check by `verify_uc_chped()` at its own tolerances — `1e-4` on its
UC-semantic checks and `1e-6` on the generic `cbls::verify_model()` pass it runs
first — so the two columns answer different questions.

### The anytime trace

`--trace PATH` writes an incumbent-versus-wall-time profile beside the results
(issue #147):

```bash
./build/cbls_uc_chped --instance ucp13 --out /tmp/ucp13.csv \
                      --trace /tmp/ucp13-trace.csv \
                      --commit "$(git rev-parse --short=7 HEAD)"
```

Columns are
`instance,periods,time_limit_s,time_seconds,batches,objective,new_best,commit_sha`.
Four of those are not in the sibling MINLPLib runner's trace and each is here
for a reason:

- **`periods`** — this runner solves one row per (instance, horizon) pair, so
  without the horizon a trace row cannot be read back to the row it describes.
- **`time_limit_s`** — the question is whether the incumbent flattened *before
  the budget*, so a trace that does not carry its budget is not self-contained.
  The per-horizon map is exactly what these traces exist to argue about, so it
  is not a stable external reference an archived trace can point at.
- **`batches`** — `SolveProgress::iteration`, the ViolationLS batch count (not
  `comparison.csv`'s `iterations`, which is the GLS count). It separates a flat
  tail meaning "converged" from one meaning "barely got going" — opposite
  arguments about the budget.
- **`commit_sha`** — a search-trajectory change silently invalidates a profile,
  and without the commit the next reader cannot tell drift from a bug.

`new_best` is 1 on an improvement and 0 on the periodic (~1s) sample the search
emits regardless. **Read flatness off the `objective` column, not off
`new_best`**: the engine accepts an improvement at a 1e-12 *relative*
threshold, which on a six-figure UC objective is far below the tenth
significant digit the trace prints, so a tail of `new_best = 1` rows can all
carry the same printed objective. A run of identical objective values is a flat
tail whatever the flag says.

A row is written whenever a finite-objective incumbent
exists — **not** only while the current assignment is feasible. That
distinction matters: the search moves off each feasible point as soon as the
objective bound is tightened below it, so filtering on the current assignment
would delete most of the periodic samples and with them the flat tail, which is
the one thing the trace is for.

Every solve the runner starts writes a **closing row** at the time the run
actually ended, with `new_best = 0`. Without it the record stops at the last
periodic sample — measured at 0.65–1.45 s short of the budget — and a solve
that improved inside that window ends on a rising incumbent with budget
apparently to spare, which reads as "still improving when the clock stopped"
when what followed was a flat tail nobody sampled. So `max(time_seconds)`
within a horizon's block is the end of the run, not the last thing that
happened to be sampled.

The closing row is also how a horizon that **never** found a valued incumbent
states itself: it is the block's only row and its `objective` cell is **empty**.
No periodic row can produce an empty objective, so that cell distinguishes
"there was never anything to flatten" — a legitimate and important answer here
— from a filtered roster, an interrupted run, or a callback that regressed to
`nullptr`. Its `batches` cell is empty too when `solve()` never reported.

Two caveats on reading `time_seconds`. It is measured from `solve()` entry, and
the greedy commitment plus the 200-iteration FJ polish run *before* that call,
so `t = 0` is not the start of work. And the first *periodic* row of a horizon's
block is its time to first feasible incumbent **with a finite objective** — a
feasibility witness whose objective overflowed (#100) is feasible but has no
value to plot, so it does not appear.

`<inst-dir>/anytime_trace.csv` is reserved as the published profile, exactly as
`comparison.csv` is, and both names are refused to any run that is not the full
published measurement. The guard is about the **files**, not about which flag
names them, and it has two layers.

A **crossed** artifact is refused outright, for every run: only `--out` ever
writes the table and only `--trace` ever writes the trace. `--trace` pointed at
`comparison.csv` truncates the published table on open, before any solving, and
a run that *is* the published protocol — full roster, default budgets, explicit
`--commit` — satisfies every protocol rung, so nothing but the file-to-flag
pairing can refuse it. Beyond that, each published name is refused to any run
that is not the full published measurement.

Naming one path for both flags is refused too — the table's closing rename
would land on the trace — as is aiming `--trace` at the table's `<out>.tmp`
staging path, where two truncating writers would interleave and the result be
published under the table's name. Point both flags at scratch
paths and none of this applies; that is the shape a smoke run or an ablation
arm should take.

### Budget defence: what the traces establish (issue #147)

Every gap percentage this benchmark publishes is a measurement **at** a
per-horizon budget (`horizon_budget()` in the runner). Those budgets were
asserted, never derived, and until #147 nothing in the repository recorded what
the incumbent was doing when the clock stopped — which made a bad gap
uninterpretable in both directions at once.

The traces below settle that, per horizon, for the two families carrying
published Pedroso bounds. They are committed at
[`benchmarks/instances/uc-chped/traces/`](../../../benchmarks/instances/uc-chped/traces/),
every row carries `commit_sha`, and every number in this section is
reproducible from them.

**Measurement**: engine commit `4a320c5`, seed 42, `feas-tol 1e-6`, `--verify`
on, per-horizon map budgets, serial on an otherwise idle machine. ucp13
verified 5/5; ucp40 verified 3/5 (two horizons found nothing to verify).

*Idle tail* is the budget minus the time of the last change in the incumbent,
as a fraction of the budget. Read off the `objective` column rather than
`new_best`, per the warning above — on this data the two agree exactly at every
horizon, no horizon having a tail of same-valued `new_best` rows, but that is a
property of this run and not something to assume next time.

| Family | T | Budget | First feasible | Last gain | Idle tail | Gap | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| ucp13 |   1 |  10s | <0.01s |   2.17s | 78% | 0.03% | **defended**, generous |
| ucp13 |   3 |  30s | <0.01s |   9.66s | 68% | 2.11% | **defended**, generous |
| ucp13 |   6 |  60s | <0.01s |   6.67s | 89% | 3.20% | **defended**, generous |
| ucp13 |  12 | 120s | <0.01s | 105.86s | 12% | 2.99% | **defended**, the one marginal case |
| ucp13 |  24 | 300s | <0.01s |  14.78s | 95% | 4.97% | **defended**, generous |
| ucp40 |   1 |  10s | <0.01s |   4.90s | 51% | 4.17% | **defended**, generous |
| ucp40 |   3 |  30s | <0.01s |   0.35s | 99% | 3.68% | **defended**, generous |
| ucp40 |   6 |  60s | <0.01s |   2.74s | 95% | 3.15% | **defended**, generous |
| ucp40 |  12 | 120s | never | — | — | — | **unchanged**; the clock is not what binds |
| ucp40 |  24 | 300s | never | — | — | — | **unchanged**; the clock is not what binds |

**No budget was changed.** Seven horizons are flat with idle tails of 51–99%;
at ucp40/3p the incumbent stops moving after 0.35s of a 30s budget. The
sharpest single statement is the convergence time: **every horizon that reaches
feasibility is within 1% of its own final objective in under 13 seconds**, including
the two carrying 120s and 300s budgets. The map is not tight anywhere it works.

The headroom is kept deliberately rather than trimmed to the observed
convergence times. A budget cut to where the search converges *today* becomes
binding after any trajectory change, and would do so silently — turning a
future engine regression into what looks like a budget artifact, which is the
exact confusion this issue exists to remove. Generous budgets cost machine time;
tight ones cost interpretability.

**ucp13/12p is the one genuinely marginal case, and it is marginal in time
rather than in value.** Its last improvement lands at 105.9s of 120s — an idle
tail of only 12% — so on the clock alone it reads as "still improving". But that
improvement is worth 384.6 absolute: **1.97% of the run's total gain and 0.166%
of the lower bound**, against a published gap of 2.99%. The run was already
within 1% of its final objective at 9.99s. A longer budget here buys tenths of a
percentage point, not the gap. The budget stands; this is the horizon to re-check
first if the map is ever revisited.

**What the traces establish about the gap numbers.** This is the reading nobody
could make before #147. A 4.97% gap at ucp13/24p is **a statement about the
search stagnating, not about the budget**: the incumbent stopped moving at 14.8s
of 300s and the remaining 95% of the budget bought nothing. The same holds at
ucp13/6p and across ucp40's three feasible horizons. So these gaps are evidence
about *search quality* — diversification, LNS, structural moves — and any work
aimed at closing them should target the search, not the clock. Conversely,
nobody may now explain these gaps away as "it needed more time"; the traces
refuse that explanation.

**ucp40/12p and 24p never reach a feasible incumbent at all**, so there is no
incumbent to flatten and raising the budget is unjustified without evidence.
Two independent signals say the clock is not the binding constraint:

1. Both end at a maximum real violation of **exactly 6**, at budgets differing
   by 2.5x (120s and 300s). Had the clock been binding, the longer run should
   have closed more residual than the shorter one. It closed none.
2. They are not short of search. ucp40/12p completed **115** ViolationLS
   batches and 24p completed **178** — *more* than the **97** and **98** with
   which ucp40/3p and 6p reached feasibility and then converged. These horizons
   get more batches than the siblings that succeed, and still find nothing, so
   "too few iterations" does not describe the failure. (Batch throughput does
   fall with horizon — 11.6, 3.23, 1.63, 0.96, 0.59 batches/s across ucp40 — but
   the batch *counts* are what the comparison rests on.)

**Open, and not tested — an honest limitation.** A longer-budget probe (ucp40 at
a uniform 600s) was attempted to settle whether these two horizons are
budget-limited or feasibility-limited. It was killed three times by the
measurement machine's low-memory reaper before producing usable rows, so **that
question is open**. The two signals above are suggestive, not conclusive: neither
rules out a feasible point sitting just beyond a 300s budget. Do not read this
section as having tested a longer budget on ucp40/12p and 24p. Nothing here
should be cited as evidence that a longer budget *cannot* help — only that
nothing measured so far suggests it would.

Archived results:

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
