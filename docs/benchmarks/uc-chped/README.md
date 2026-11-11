# UC-CHPED: Unit Commitment with Combined Heat and Power Economic Dispatch

## Problem Description

Unit Commitment with Combined Heat and Power Economic Dispatch (UC-CHPED) is a power system optimization problem that combines two classical subproblems:

1. **Unit Commitment (UC):** Decide which generating units to turn on/off over a planning horizon (binary decisions), subject to minimum uptime/downtime constraints and startup costs.
2. **Economic Dispatch (ED):** For committed units, determine the optimal power output to meet demand at minimum fuel cost, where the cost function includes **valve-point effects** — sinusoidal ripples that model the non-smooth behavior of steam admission valves.

**Real-world motivation:** Day-ahead scheduling of thermal power plants. The system operator must commit enough generation capacity to serve hourly demand plus a spinning reserve margin, while minimizing total fuel cost and startup cost. Valve-point effects make the cost function non-convex and non-smooth, ruling out simple LP/QP dispatch.

## References

- **Pedroso, Kubo & Viana (2014)** — *"Unit commitment with valve-point loading effect"*, DCC-2014-05 / [arXiv:1404.4944](https://arxiv.org/abs/1404.4944). Source of instance data and known MIP bounds (Table 2). Original code at `http://www.dcc.fc.up.pt/~jpp/code/valve/`.
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

### Provenance (#148, #193)

`ucp13` and `ucp40` were checked field by field against the authors' GPL
instance code,
[`ucp_data.py`](http://www.dcc.fc.up.pt/~jpp/code/valve/ucp_data.py)
(2013-12-09); method and numbers in
[`FIDELITY.md` §7](../../../benchmarks/uc-chped/FIDELITY.md#7-instance-data--do-we-solve-the-published-instances-148).

| Field | `ucp13` | `ucp40` |
|---|---|---|
| Cost coefficients `a b c d e`, `P_min` | sourced — identical to the authors' | sourced — identical to the authors' |
| `P_max` | sourced — identical | sourced — identical since #193 (units 19-20 were 500 here until then, against the source's 550: an error in `benchmarks/chped/data.py`) |
| Kazarlis mapping (min up/down, `t_cold`, startup costs, initial state) | sourced — the authors' own assignment | sourced — the authors' own `i % 10` |
| Demand and reserve profile | sourced — the authors' table verbatim | sourced — the authors' table verbatim |

`ucp100`/`ucp200` and the 48h/168h extensions are this repository's
construction; the authors publish nothing for them.

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

Instances can be extended to 48h and 168h (1 week) via `extend_horizon()`, which repeats the 24h demand and reserve profile, scaling day `d` (0-indexed) by the weekly factor `1 + 0.03·(sin(2πd/7) − 1)` and rounding to the nearest MW. Its seven values (0.970, 0.993, 0.999, 0.983, 0.957, 0.941, 0.947) are distinct, so no two days of a week are exact copies, which a search could exploit. (A cosine would make day 0 the base day exactly, but it is symmetric about mid-week and repeats days 1-3 as days 6-4.)

**The construction rule (#194): no extended day's demand or reserve exceeds the base profile's in the same hour.** So an extended instance has at least its base instance's capacity slack, `sum(P_max) − demand[t] − reserve[t]`, in every period. The base slack is 699 MW for `ucp100` and 470 MW for `ucp200`, both at hour 12; the extended instances' minimum slack is 906/723 MW (`ucp100` 48/168 periods) and 884/517 MW (`ucp200`). Rounding cannot break the rule, because the base profile is integral and the factor is below 1. (The factor before #194, `1 + 0.03·sin(2πd/7)`, exceeded 1 and made every extended instance infeasible.) `tests/python/test_uc_chped_instances.py` checks `demand + reserve <= sum(P_max)` in every period of every committed instance, and checks the rule itself on `extend_horizon()`.

## Instance Summary

| Instance | Units | Periods | Binary Vars | Continuous Vars | Total Vars | Known Bounds |
|----------|------:|--------:|------------:|----------------:|-----------:|:-------------|
| ucp13    |    13 |    1–24 |       13–312 |          13–312 |     26–624 | Pedroso 2014 |
| ucp40    |    40 |    1–24 |      40–960 |         40–960 |   80–1920 | Pedroso 2014 |
| ucp100   |   100 |   1–168 |    100–16800 |      100–16800 |  200–33600 | None         |
| ucp200   |   200 |   1–168 |    200–33600 |      200–33600 |  400–67200 | None         |

## Known Bounds

From Pedroso et al. (2014), Table 2 — MIP with 1-hour time limit. These are
bounds on the **true** valve-point cost, not on a piecewise-linear surrogate:
the authors' code refines its linearisation until the upper bound, priced at
the true cost, meets the lower one.

**The 13-unit rows describe this repository's `ucp13`.** It is the authors'
instance field for field, and both of its proven optima are reproduced by an
exact solve (11701.28 and 38849.84, §7 of `FIDELITY.md`).

**The 40-unit rows describe this repository's `ucp40` since #193.** #148
found that ours capped units 19-20 at 500 MW where the authors' instance has
550; #193 corrected the two values (in `benchmarks/chped/data.py`, which
`ucp40`, `ucp100` and `ucp200` all draw from, and its C++ twin `data.h`).
`ucp40` is now the authors' instance field for field, and an exact solve
reproduces its proven 1-period optimum (55644.79). A `ucp40` result measured
before #193, such as the budget-defence traces, ran the old instance and has
no gap against these bounds.

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

### Closed-form jump scoring does not reach the search (#190)

Every row here is a bare affine body (`add_constraint(sum(...))`, read as
`expr <= 0`), so since #190 it is eligible for FJ's closed-form
`LinearJumpScorer`. But a column takes the closed form only when *every*
weighted row it reads is eligible, and every `y` and `p` column also reads the
objective row `obj <= bound`, whose fuel-cost body is nonlinear and always
weighted in `solve()`. So the batch API scores every uc-chped candidate with the
DAG probe; only the `run()` path — the runner's 200-iteration warm start and the
LNS repair — takes the closed form, on a small share of its prepares
(`docs/benchmarks/linear-jump-pingpong.md`, #190 addendum). The before/after
measurement, on the pre-#193 instances, is on #190.

## Reference Solver

The reference solver (`benchmarks/chped/reference_solve.py --uc`) uses PySCIPOpt to solve a MIP formulation:

- Binary variables for commitment ($y$), startup ($su$), and shutdown ($sd$)
- **Piecewise-linear approximation** of the valve-point cost function using 50 segments per unit with an incremental (SOS2-like) formulation
- Hot/cold startup cost modeled via auxiliary binary indicator variables
- Standard min uptime/downtime constraints

The PWL approximation is what makes the reference a MIP: a MIP cannot carry the `|sin(...)|` term (SCIP's MINLP mode can, which is what the exact check below uses). It is **not** negligible at 50 segments: uniformly spaced breakpoints straddle the valve-point cusps, which are exactly where optimal dispatch sits, and the chord over a cusp overestimates the cost by up to about `d·e·Δ/2` per unit. Summed over units, the worst case for one period of `ucp13` is 213 at 50 segments and 47 at 200 (1.8% and 0.4% of the optimum); the realised error is smaller. Two further differences from the source: demand is `>=` where the source has `=`, and a startup of a `t_cold = 0` unit is priced hot, where the source and our model price every startup on these instances cold (§7 of `FIDELITY.md`).

For an exact check, #148's `instance_identity.py` (in git history at `047f2d4c`; FIDELITY.md §7) solves the proven-optimum cases as a SCIP MINLP with the true `sin` term (no linearisation) and compares the instance data against the authors' code.

See [`benchmarks/chped/reference_solve.py`](../../../benchmarks/chped/reference_solve.py) for the full implementation.

## Results

`comparison.csv` carries the 10 cited Pedroso reference rows only; no measured
CBLS row is published yet. The full campaign, #257, writes them. (The
simulated-annealing rows the file once held recorded no feasibility tolerance
and were deleted when the default moved from `1e-9` to `1e-6`; they are in git
history.)

`benchmarks/uc-chped/uc_chped.cpp` writes the table, stating the feasibility
tolerance explicitly and recording it, the seed, the time budget, the search
configuration and the engine commit on every measured row. The search
configuration is the engine's defaults with compound moves (Novelty Jump)
**off** (`published_search_flags()` in the runner): a single-seed smoke at
`b8f52f4` found them worse on 16 of the 19 rows that differed here, so
`--compound-moves` is an ablation arm the runner refuses to write onto the
published table.

That choice was measured on the same instances the table publishes, so on
them it is in-sample (#233). The `xk tune` study (#266) kept the same
protocol, tuned on the constructed instances only, and declared Pedroso's ten
report-only; the published rows are still not out-of-sample, since their
protocol was chosen on them. The objective tie-break (#220) needs an affine
objective and does nothing here, because the valve-point term is not affine;
the `search_config` cell records it only when off. When #257 regenerates the
measured rows, the text quoting them names the engine commit (also on every row
as `commit_sha`) and says the tie-break is inert on this objective.

The command that writes it:

```bash
set -o pipefail
cmake --build build --target cbls_uc_chped &&
sha=$(.venv/bin/python benchmarks/uc-chped/preflight.py) &&
./build/cbls_uc_chped --out benchmarks/instances/uc-chped/comparison.csv \
                      --commit "$sha" --verify 2>&1 |
    tee benchmarks/instances/uc-chped/comparison.log.tmp &&
mv benchmarks/instances/uc-chped/comparison.log.tmp benchmarks/instances/uc-chped/comparison.log
```

The rebuild comes first so the binary is the `HEAD` the preflight then names.
`benchmarks/uc-chped/preflight.py` refuses, before any solving starts, a working
tree with modified tracked files, and a `build/` that is not `Release`, is
configured with `CBLS_SANITIZE` or `CBLS_PROFILE`, or was configured from a
different checkout. On success it prints the commit SHA, so the recorded
`--commit` comes from the check that found the tree clean rather than from a
free-form string (issue #146). The runner separately refuses to write either
published artifact from a binary that is not a Release, uninstrumented build,
so a caller who skips the preflight is still stopped on that half.

`comparison.log` is committed alongside the table. The `verified` column carries
each row's verdict; the log carries *which* checks failed on a `VERIFY FAIL`
row, which the runner prints to standard output and nowhere else. `tee` keeps
refusals on the terminal, and `pipefail` keeps the runner's exit status. The log
is renamed into place only on a run that exited 0, so a refused or killed run
leaves the committed log matching the committed table.

`--feas-tol T` overrides the tolerance, `--time-limit S` overrides the
per-horizon budget map, `--seed N` the seed, `--instance NAME` restricts the
roster and `--periods N` its horizons (one (file, horizon) per run is the shape a
tuning study scores, #262; a horizon no selected file has is refused).
`--threads N` runs the `ParallelSearch` portfolio instead of the single-threaded
`solve()` the table measures, and the row records it in `search_config` as
`threads=N` (at 1 the cell stays as published). `--instance`, `--periods`, `--threads` and
`--time-limit` are refused outright when the resolved `--out` path *is*
the published table above — a partial roster or a shortened budget cannot become
the published result, whether the path was defaulted or spelled out. Writing
that file at all requires `--commit`, so a published row always names the engine
it measured, and `--verify`, so no published row leaves `verified` empty. Point
`--out` at a scratch path and none of this applies — but see
the anytime-trace section below, where `--trace` is held to the same rule
against both published names. Note that `feasible` is
the engine's verdict at the row's `feas_tol` while `verified` is an independent
re-check by `verify_uc_chped()` at its own tolerances — `1e-4` on its
UC-semantic checks and `1e-6` on the generic `cbls::verify_model()` pass it runs
first — so the two columns answer different questions.

### Scoring a multi-seed campaign (#257)

`benchmarks/uc-chped/score.py RESULTS/seed*/comparison.csv [--out table.md]`
reads one scratch `comparison.csv` per seed, written by `cbls_uc_chped --out
RESULTS/seed<N>/comparison.csv --seed N --commit SHA --verify`, and gives one line
per (instance, horizon): the budget, the seeds that solved it, and the gap to
Pedroso's lower bound and one-hour incumbent as median [min, max] over those
seeds, beside Pedroso's own gap. Instances Pedroso did not publish show their
objective and no gap. There is no SCIP column, because `reference_solve.py --uc`
writes no table. The scorer refuses an incomplete roster, a row failing
verification or never verified, mixed commits, search configurations (thread count
included) or tolerances, and one horizon at two budgets; the module's docstring
has the full list. `--roster NAME:T,T ...` scores a smoke's subset and says so on
its first line.

### The anytime trace

`--trace PATH` writes an incumbent-versus-wall-time profile beside the results
(issue #147):

```bash
cmake --build build --target cbls_uc_chped &&
sha=$(.venv/bin/python benchmarks/uc-chped/preflight.py) &&
./build/cbls_uc_chped --instance ucp13 --out /tmp/ucp13.csv \
                      --trace /tmp/ucp13-trace.csv \
                      --commit "$sha"
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
  arguments about the budget. Single-threaded only: at `--threads N > 1` the
  portfolio rewrites a report's time and objective but not its iteration, so
  the cell is the reporting worker's own count and is not monotone.
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

Every gap this benchmark publishes is a measurement **at** a per-horizon budget
(`horizon_budget()` in the runner). #147 traced ucp13 and ucp40, the two
families with published bounds, to test whether those budgets bind. The traces
are committed at
[`benchmarks/instances/uc-chped/traces/`](../../../benchmarks/instances/uc-chped/traces/),
every row carries `commit_sha`, and every number here is reproducible from
them: engine commit `4a320c5`, seed 42, `feas-tol 1e-6`, `--verify`, map
budgets, serial on an idle machine. `4a320c5` predates later changes that act
on this model (the float-hook sweep fix #207, the perf work #270/#271), so this
is a record of that commit, not of today's engine; #257 re-measures. The ucp40
runs predate #193 too, so they solved the old instance: their budget verdicts
hold for it, and their gaps are not gaps against Table 2.

- **No budget binds, and none was changed.** Every horizon that reached
  feasibility was within 1% of its own final objective in under 13 s, including
  the 120 s and 300 s ones; seven of the eight had idle tails (budget after the
  last change in the `objective` column) of 51-99%.
- **ucp13/12p is the one marginal case, in time rather than value.** Its last
  gain lands at 105.9 s of 120 s, but is worth 0.166% of the lower bound against
  a 2.99% gap. It is the horizon to re-check first if the map is revisited.
- **The headroom is deliberate.** A budget cut to today's convergence time
  becomes binding after a trajectory change, silently, and turns an engine
  regression into what looks like a budget artifact.
- **So the gaps measure search quality, not the clock.** ucp13/24p (4.97%)
  stopped moving at 14.8 s of 300 s. Work on closing gaps targets
  diversification, LNS and structural moves.
- **ucp40/12p and 24p never found a feasible point because none existed before
  #193 (#152):** period 12 needs 12628 MW of demand plus reserve against 12622 MW
  of total `P_max`, a shortfall of exactly the residual of 6 both horizons ended
  at, also after 600 s at `e9b28a6`
  ([`traces/ucp40-600s-trace.csv`](../../../benchmarks/instances/uc-chped/traces/ucp40-600s-trace.csv)).
  ucp200 at 12/24 periods was short too (fixed by #193); the four extended
  horizons, for a different reason, until #194. Every committed instance now has capacity slack
  in every period, and a test pins it.
- **The repaired instances have only smoke runs** (10 s, uniform budget, seed
  42, `--verify`, shared machines): after #193, `ucp40` and `ucp200` were
  feasible and verified at every 1-24 period horizon; after #194, all four
  extended instances were. Those settle satisfiability and the end-to-end flow,
  not results.
- **The 48h/168h budgets have never been defended.** Defending them, and
  re-establishing ucp40's on the corrected instance, needs traces at the map
  budget (#257).

The per-horizon table, the 600 s probe and the smoke objectives are on
[#147](https://github.com/spoorendonk/cbls/issues/147#issuecomment-6632898096).

## Code Locations

| File | Description |
|------|-------------|
| [`benchmarks/instances/uc-chped/data.py`](../../../benchmarks/instances/uc-chped/data.py) | Instance definitions (Python) |
| [`benchmarks/instances/uc-chped/gen_jsonl.py`](../../../benchmarks/instances/uc-chped/gen_jsonl.py) | Serializes `data.py` into the `*.jsonl` below — from the repo root, `.venv/bin/python benchmarks/instances/uc-chped/gen_jsonl.py` |
| [`benchmarks/instances/uc-chped/*.jsonl`](../../../benchmarks/instances/uc-chped/) | Serialized instances (JSONL), derived from `data.py` |
| [`benchmarks/instances/uc-chped/comparison.csv`](../../../benchmarks/instances/uc-chped/comparison.csv) | Results comparison table |
| [`benchmarks/uc-chped/data.h`](../../../benchmarks/uc-chped/data.h) | C++ data structures and JSONL loader |
| [`benchmarks/uc-chped/uc_model.h`](../../../benchmarks/uc-chped/uc_model.h) | CBLS model builder |
| [`benchmarks/uc-chped/uc_chped.cpp`](../../../benchmarks/uc-chped/uc_chped.cpp) | Benchmark runner executable |
| [`benchmarks/chped/reference_solve.py`](../../../benchmarks/chped/reference_solve.py) | Reference solver (SCIP MIP + scipy) |
| [`tests/test_uc_chped.cpp`](../../../tests/test_uc_chped.cpp) | Catch2 unit tests |
