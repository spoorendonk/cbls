# Nuclear-outage model fidelity audit vs ROADEF/EURO 2010 specification

Tracking issue: #74. Epic: #26.

## TL;DR

- **Two distinct models** live side-by-side in this directory:
  1. A **synthetic / "legacy" model** (`nuclear_model.h::build_nuclear_model`,
     `dispatch.h`, `nuclear_hook.h`) over a stripped-down formulation
     (single-stage merit-order dispatch, no fuel dynamics, no CT13–CT21).
     This is the only one driven by `comparison.csv` today.
  2. A **ROADEF 2010 model** (`build_roadef_model`, `roadef_dispatch.h`,
     `roadef_hook.h`) over the actual ROADEF data format with fuel dynamics,
     decreasing profiles, and CT13–CT21 evaluated as hook penalties.
- **Severity vs ROADEF 2010 spec: qualitative.**
  - The synthetic model solves a *different problem* (not ROADEF). Comparison
    against any ROADEF 2010 BKS is meaningless.
  - The ROADEF model is structurally close to the spec but has known gaps
    (CT6 greedy lower bound #45, CT12 not enforced #44, CT14–CT18 only as
    hook penalties #43, scenario subsampling, max-RMAX reload heuristic #42),
    and the official competition data is **not present in the repo** (the
    `download.sh` script has never been run here — only synthetic JSONL
    instances ship under `benchmarks/instances/nuclear-outage/`).
- **`comparison.csv` decision:** *self-consistent only*. The current rows
  (mini, small) are CBLS-vs-greedy / CBLS-vs-SCIP-SAA on synthetic instances,
  with no relation to ROADEF 2010 BKS. The CSV should not claim "gap vs BKS"
  until (a) competition data is downloaded, (b) the ROADEF model is run on
  it, and (c) at minimum #43, #44, #45 close.

## 1. Source specification

### Reference

- **Competition:** ROADEF/EURO 2010 Challenge, "Scheduling of Outages of
  Nuclear Power Plants" (EDF). https://www.roadef.org/challenge/2010/en/
- **Subject PDF:** `sujetEDFv22.pdf` (downloaded by
  `benchmarks/instances/nuclear-outage/download.sh`; **not present in this
  repo**, so no page citations are possible).
- **Winning paper (1st prize, integer LP approach):** Jost & Savourey (2013),
  *J. Scheduling* 16, 551–566.
- **Stochastic LP approach (2nd prize):** Gorge, Lisser & Zorgati (2012),
  *EJOR* 216, 344–354.
- **Survey:** Froger et al. (2016), *EJOR* 251, 695–706.

### Constraint catalogue (CT1–CT21)

The numbering below follows the EDF subject document conventions as used in
this codebase (parser/data-struct names, `roadef_hook.h` comments, README).
Direct page citations are not possible without the PDF; everything below is
**inferred from code, parser, and data-struct comments** unless tagged "spec".

| ID | Name (per code/comments) | Spec area |
|----|--------------------------|-----------|
| CT1 | Type 1 plant power bounds (`pmin <= p1 <= pmax`) per scenario / timestep | dispatch |
| CT2 | Type 1 production cost (linear in `p1`) | dispatch |
| CT3 | Type 2 plant offline during outage (`p2 = 0` while in outage) | dispatch |
| CT4 | Type 2 power upper bound (`p2 <= pmax_t`) when online | dispatch |
| CT5 | Type 2 nominal-cycle power (`p = pmax_t` when fuel >= BO) | dispatch |
| CT6 | Type 2 decreasing profile when fuel < BO: `(1-eps)*PB(x)*pmax <= p <= PB(x)*pmax` | dispatch |
| CT7 | Refueling at start of outage; reload <= RMAX, >= RMIN | dispatch |
| CT8 | Fuel balance during cycle: `x(t+1) = x(t) - p(t)*dt` | dispatch |
| CT9 | Refueling stock relation `x_after = ((q-1)/q)*(x_before - bo_prev) + r + bo_k`; `<= SMAX` | dispatch |
| CT10 | Stock at start of refueling `<= AMAX`; reload "instantaneous" at first timestep of outage | dispatch |
| CT11 | End-of-horizon fuel credit `-fuel_price_end * residual_stock` (sign baked into objective) | dispatch |
| CT12 | Modulation limit MMAX per cycle (max number of `p` changes) | dispatch |
| CT13 | Outage start window `[TO_{i,k}, TA_{i,k}]` for plant i, cycle k | scheduling |
| CT14 | Min spacing within plant set `C_m` (any pair) `Se_m`; spec uses `max(h_i - h_j - DA_j, h_j - h_i - DA_i) >= Se_m` | scheduling |
| CT15 | Min spacing within plant set within window `[ID_m, IF_m]` | scheduling |
| CT16 | Min `\|h_i - h_j\| >= Se_m` (start-time spacing) | scheduling |
| CT17 | Min spacing on coupling times `\|h_i+DA_i - h_j-DA_j\| >= Se_m` | scheduling |
| CT18 | Min spacing on each coupling vs other start `\|h_i+DA_i - h_j\|` and `\|h_j+DA_j - h_i\|` | scheduling |
| CT19 | Resource (manpower) constraint: at most Q_m parallel usages per week, with per-plant per-cycle (start, duration) within outage | scheduling |
| CT20 | Max number of overlapping outages at week h_m within plant set | scheduling |
| CT21 | Max offline capacity within `[IT_start, IT_end]` per plant set | scheduling |

### Objective (per ROADEF spec)

Minimise the **expected scenario cost**:

`E_s [ sum_t (sum_j cost_j(p1[j,t,s])*dt + sum_i refuel_cost_{i,k} * r_{i,k}) - sum_i fuel_price_end_i * residual_fuel_i(s) ]`

(refueling costs are scenario-independent; the residual-fuel credit
implicitly depends on the scenario through fuel dynamics.)

## 2. Our model — point-by-point coverage

### 2a. Synthetic / "legacy" model (`nuclear_model.h::build_nuclear_model`)

This is what `cbls_nuclear_outage` runs by default and what
`comparison.csv` measures. **It is not the ROADEF problem.** The relationship
to ROADEF CT1–CT21:

| ROADEF id | Status in synthetic model | Where |
|-----------|---------------------------|-------|
| CT1 | **Replaced** by simple `capacity[u]` upper bound, no `pmin`. `dispatch.h:46-53` |
| CT2 | **Replaced** by single linear `fuel_cost[u]` per unit. `dispatch.h:51` |
| CT3 | **Replaced** by 0/1 availability `avail[t][u]`. `dispatch.h:14-31` |
| CT4 | Implicit in `capacity[u]`. No per-timestep `pmax_t`. |
| CT5 | **Missing** (no fuel concept). |
| CT6 | **Missing** (no fuel concept). |
| CT7 | **Missing** (no refueling). |
| CT8 | **Missing** (no fuel dynamics). |
| CT9 | **Missing**. |
| CT10 | **Missing**. |
| CT11 | **Missing** (no end-of-horizon credit). |
| CT12 | **Missing** (no modulation). |
| CT13 | **Replaced** by `[outage_earliest, outage_latest]` per outage. `nuclear_model.h:31` |
| CT14–18 | **Collapsed** into one parameter `min_spacing_same_site` (single-site, simple-pair). `dispatch.h:120-141`, `nuclear_model.h:42-56` |
| CT19 | **Missing** (no resource/manpower). |
| CT20 | **Replaced** by per-site `max_outages_per_site[s]`. `dispatch.h:103-116` |
| CT21 | **Missing**. |

DAG side: only **unit non-overlap** is in the DAG
(`nuclear_model.h:42-56`); site capacity, site spacing, and dispatch are all
hook-side penalties. Demand satisfaction is implicit (unserved energy is
penalised at `inst.penalty_unserved`, `dispatch.h:56-58`).

### 2b. ROADEF model (`build_roadef_model`, `roadef_hook.h`, `roadef_dispatch.h`)

This is the model that *would* compare against ROADEF 2010 BKS, but it is
not what `comparison.csv` measures today and the competition data is not
checked in.

| ROADEF id | Status | Location | Notes |
|-----------|--------|----------|-------|
| CT1 | **Implemented** (greedy/heuristic) | `roadef_dispatch.h:199-228` | Type 1 dispatched merit-order; `gen = max(pmin, min(pmax, remaining))`. **Deviation:** when `remaining < pmin` the unit is forced to `pmin`, producing **overproduction** (line 217 comment: "may overproduce when remaining < pmin"). The spec requires unit commitment to determine whether a unit runs at all; the greedy hook commits every unit cheaper than necessary. |
| CT2 | **Implemented** (linear cost) | `roadef_dispatch.h:218` | `cost += gen * cost[s][t] * dt`. |
| CT3 | **Implemented** | `roadef_dispatch.h:145-161` | `t2_prod_step[i] = 0` while in outage. |
| CT4 | **Implemented** | `roadef_dispatch.h:171,175,182` | `prod <= pmax_t * (PB factor) * 1`, plus `prod <= fuel/dt`. |
| CT5 | **Implemented** | `roadef_dispatch.h:170-171` | `if fuel >= bo: max_prod = pmax_t`. |
| CT6 | **Partial** | `roadef_dispatch.h:172-179` | Upper bound `PB(x)*pmax` is enforced, but the **lower bound** `(1-eps)*PB(x)*pmax <= p` is **NOT** checked in the greedy dispatcher (#45). The greedy may serve less than the lower bound when total demand is lower than what the plant must produce — this would fail the official checker on CT6 in some scenarios. |
| CT7 | **Implemented** (heuristic) | `roadef_hook.h:67-73`, `roadef_dispatch.h:81-95`, `roadef_dispatch.h:154-160` | `compute_reloads` always returns `RMAX[k]` (#42 — no cost-aware optimisation, RMIN/RMAX bounds are observed only because RMAX is by definition admissible). |
| CT8 | **Implemented** | `roadef_dispatch.h:186-188` | `fuel -= prod * dt`. |
| CT9 | **Implemented** | `roadef_dispatch.h:154-160` | Stock relation with `q`, `bo_prev`, `bo_k`, capped at `smax`. |
| CT10 | **Partial** | `roadef_dispatch.h:151-152` | Refuel happens at first timestep of outage, but `amax` (max stock **before** refueling) is **not enforced**: there is no penalty if `fuel[i] > amax[k]` immediately before the refuel event. Solver writes the value into the spec output but does not penalise infeasibility. |
| CT11 | **Implemented** | `roadef_dispatch.h:235-237` | `total_cost -= fuel_price_end * fuel[i]`. The spec also implies a *minimum* end-of-horizon fuel constraint per plant in some instance variants — **not enforced**. |
| CT12 | **Missing** (#44) | — | MMAX modulation is parsed (`Type2Plant::mmax`) but never read in dispatch / hook / model. |
| CT13 | **Implemented in DAG** | `nuclear_model.h:89-100` | `ha[o]` IntVar with domain `[TO, TA]`; `TA<0` falls back to `inst.H-1`. **Cycle ordering** also in DAG (line 117–132): `ha[o1] + DA[o1] - ha[o2] <= 0` for consecutive cycles on the same plant. |
| CT14 | **Hook penalty only** (#43) | `roadef_hook.h:141-162` | All-pairs `O(n^2)` over plant set; `max(h1-h2-DA2, h2-h1-DA1) >= spacing`. Not in DAG → no gradient signal to SA. |
| CT15 | **Hook penalty only** (#43) | `roadef_hook.h:164-189` | Same as CT14 with extra `[ID, IF]` window filter. **Parser fragility:** `data.h:566-577` treats the second token of an `end <number>` line as `period_end` and continues, breaking only on a plain `end constraint`. This works for the documented two-line form but is undocumented and easy to break by re-ordering lines. CT15 instances that omit a closing `end <number>` line leave `period_end = -1`, making `o2_intersects` always false on positive `period_start`, so no penalty fires — silent acceptance, not a hard bug today. |
| CT16 | **Hook penalty only** (#43) | `roadef_hook.h:191-203` | `\|h1 - h2\| >= spacing`. |
| CT17 | **Hook penalty only** (#43) | `roadef_hook.h:205-222` | `\|coupling1 - coupling2\| >= spacing`. |
| CT18 | **Hook penalty only** (#43) | `roadef_hook.h:224-242` | `\|coupling_i - h_j\|` plus `\|coupling_j - h_i\|`. **Note:** the inner `gap1`/`gap2` use `outages[j]`/`outages[i]` as the "h" side which mixes indices; this matches the spec's intended both-directions check, but is fragile to read — see follow-up below. |
| CT19 | **Hook penalty** | `roadef_hook.h:244-268` | Per-week count of usages-active intervals; viol = `count - quantity`. |
| CT20 | **Hook penalty** | `roadef_hook.h:270-283` | At fixed `week h_m`, count outages active there. |
| CT21 | **Hook penalty** | `roadef_hook.h:285-302` | Sum of offline `pmax_t` across plant set during `[IT_start, IT_end]`. **Deviation:** uses `t = h * timesteps_per_week` (only the first timestep of the week), not the per-week `pmax_t` aggregate; the spec's IMAX is on weekly offline capacity. |

### 2c. Objective coverage (ROADEF model)

`roadef_hook.h:31-84` writes:

```
cost = refuel_cost(scenario-independent, RMAX) +
       avg_prod_cost(over rotating window of n_sc scenarios) +
       sum(spacing_penalty * 1e13)
```

Deviations vs spec:

- **Scenario subsampling.** When `inst.S > 50`, only `scenarios_per_move = 50`
  scenarios are evaluated per hook call (`nuclear_outage.cpp:94-96`, default in
  `roadef_hook.h:24` is `-1` = all). The window rotates every `epoch_size = 10`
  hook calls. **Effect on objective:** the value written into the model is a
  rotating SAA estimator, not the true 500-scenario expectation. Expected-cost
  estimate has `sigma / sqrt(n_sc)` noise. This biases SA decisions toward
  schedules cheap on the *current* window. The final reported objective at
  `result.objective` is whatever the hook wrote on the last accept — also a
  subsampled estimate. **No final full re-evaluation across all 500
  scenarios** is performed before logging the result.
- **Penalty weight `1e13`.** Treats infeasibility as ~10³ × the typical
  objective magnitude (~1e10–1e12 EUR). This is a soft penalty; nothing
  prevents the solver from reporting an objective that is "feasible" by SA
  but fails the official checker on CT14–CT21.
- **Type 1 overproduction (CT1).** The merit-order greedy can `gen >= pmin`
  even if total demand is met by cheaper plants → solution may have
  positive surplus that becomes (effectively free) energy. In the spec's
  LP formulation a unit can be off (`y=0`); here every Type 1 plant runs
  at >= pmin if its merit rank is hit, regardless of whether the system
  actually needs that energy. This **inflates** the reported cost vs the
  LP optimum but matches ROADEF checker arithmetic if the checker treats
  the over-produced energy as paid-for. **Not verified against checker.**

### 2d. Constraint counts

`nuclear_outage.cpp:82-84` prints the parsed constraint counts. For `data0.txt`
(per epic #26 description), the file would expose CT13×N, CT14–18×M, CT19×P,
CT20×Q, CT21×R; we have **no record of running the official checker** on
`data0.txt` from this worktree (no `CHECKER/` directory present), but the
epic description claims data0.txt passes all 21 constraint checks at
commit `72c3671` from a previous worktree — that result is **not
reproducible from this branch's working tree** until `download.sh` is run.

## 3. Greedy / hook fidelity

### 3a. Synthetic hook (`NuclearDispatchHook`)

Fully described above (only computes merit-order dispatch + simple site
penalty + scenario rotation). It is faithful to **its own** synthetic model
but not to ROADEF.

### 3b. ROADEF hook (`ROADEFDispatchHook`) — formal-model deviations

| Hook does | Formal model says | Notes |
|-----------|-------------------|-------|
| Forces every Type 1 plant on its merit run to `pmin` once committed (CT1 line 217) | Allows `y_j = 0` (commitment binary); LP minimises over commitments | Inflates Type 1 cost; can also help CT21 (more offline cap from Type 2). Net effect on reported objective: probably **higher than LP optimum**. |
| Reload = RMAX always | Cost = `c_refuel * r + (-c_residual * residual_fuel)`; trade-off; #42 | Smaller end-of-horizon credit, possibly higher refuel cost. |
| Skips CT12 modulation entirely | Modulation count `<= MMAX_k` | Penalty function never penalises high-modulation profiles; greedy runs at `pmax_t` so changes are smooth in practice but not guaranteed below MMAX. |
| CT6 lower-bound: greedy picks `prod = max(0, max_prod)` then clamps to `fuel/dt` | Spec: `(1-eps)*PB(x)*pmax <= p <= PB(x)*pmax` | Greedy may dispatch less than `(1-eps)*PB*pmax` (e.g. when fuel/dt limits bind), violating CT6 lower bound (#45). |
| CT10 / AMAX not penalised | `stock_before_refuel <= AMAX` | Solution can be reported feasible by SA but rejected by checker. |
| Spacing penalties evaluated at hook time only | Spacing in DAG | Hook adds `1e13 *` violation; SA only sees the **objective** changing, not the constraint structure. ViolationManager has no record of these as "constraints" → can't drive Feasibility-Jump or adaptive-lambda properly. |
| CT13 cycle-ordering in DAG | Same | OK. |
| Scenario rotation `epoch_size=10`, `scenarios_per_move=50` | All scenarios per evaluation | Bias / noise in objective writes. |

### 3c. Hook contract violations (engine-level)

- **Penalty addition into `objective_node` instead of `ViolationManager`**:
  `roadef_hook.h:78-83` writes `cost + scheduling_penalty(...)` into
  `model.node_mut(rm.objective_node).value`. This bundles
  feasibility into the objective, defeating the engine's adaptive-lambda
  separation. It is "OK" for SA's accept/reject (ratio is preserved) but
  breaks the moment ViolationLS (#67) lands — that path expects violations
  to live in `ViolationManager`.
- **No `last_changed_vars` use**: the hook ignores its third parameter; every
  call recomputes lookups, status, reloads, and dispatch from scratch.
  #41 tracks the optimisation; for fidelity the issue is that
  `compute_plant_status` allocates `n_type2 * H` bool/int matrices on every
  call.

## 4. 500-scenario handling

| Aspect | Synthetic hook | ROADEF hook |
|--------|----------------|-------------|
| Default `scenarios_per_move` | 50 (capped at `n_scenarios`) | -1 = all when `S<=50`, else 50 |
| Strategy | Rotating window | Rotating window |
| Window size | `min(scenarios_per_move, n_scenarios)` | same |
| Rotation period | `epoch_size = 10` hook calls | `epoch_size = 10` |
| Final re-eval over full set? | **No** | **No** |
| Effect on reported objective | If `n_sc >= n_scenarios`: exact. Else: SAA estimator with `O(1/sqrt(n_sc))` noise; logged result is whatever the last accept saw on its window — **not the true expectation** | Same |

Implications:

- For `mini` (S=20) with `scenarios_per_move = 5`: very high noise; the
  reported objective for `mini` in `comparison.csv` is a 5-scenario estimate,
  not the 20-scenario truth.
- For `small` (S=50) with `scenarios_per_move = 20`: 40% sample.
- For `medium` / ROADEF B-set instances with S=500 and `scenarios_per_move=50`:
  10% sample. `comparison.csv` does **not** flag this.

A faithful comparison requires a **terminal full-scenario re-evaluation**
of the best schedule — not done today.

## 5. Severity & decision

### Severity

- **Synthetic model:** **qualitative** divergence from ROADEF 2010. It is a
  *different problem*. Cannot meaningfully compare to ROADEF BKS.
- **ROADEF model:** **quantitative** divergence from ROADEF 2010. The
  scheduling structure (CT13 windows, cycle ordering, all spacing
  constraints, CT19/20/21) is present, and the dispatch covers CT1–CT11
  (with the CT6-lower-bound hole, CT10-AMAX hole, and Type 1 commitment
  approximation). CT12 is entirely missing. The objective is a scenario-
  subsampled estimator. The competition data is not checked into this branch.

### Decision for `comparison.csv`

The current `comparison.csv` lists **mini** and **small** synthetic
instances and reports CBLS-vs-Greedy gap. These are valid as
*self-consistency* numbers for the synthetic model. They are **not** valid
as a claim against ROADEF 2010 BKS — none of the synthetic instances are
ROADEF instances and the synthetic model doesn't even try to encode CT5–
CT12.

**Action:** annotate `comparison.csv` with a `notes` column declaring that
all rows are synthetic-model only and no ROADEF BKS comparison is implied.
Until #46 (BKS lookup) lands and the ROADEF model runs the data0–dataB10
files through the official checker, this benchmark may report **only
self-consistent SA-vs-Greedy / SA-vs-ViolationLS** results.

A future ROADEF-grade row (e.g., `data0`, `dataB6`) requires:

1. `download.sh` executed (or instances vendored in some other way).
2. `reference_solve.py` updated for ROADEF format (#47).
3. CT6 lower bound (#45) and CT12 (#44) closed.
4. CT14–18 ideally promoted to DAG (#43) so SA gets a real gradient.
5. Final full-scenario re-evaluation in `nuclear_outage.cpp` before
   reporting.
6. Official `checker.jar` run on the produced `solution.txt` and a
   "checker-passed" column added.

## 6. Cross-reference index

| Gap | Existing issue | Status |
|-----|----------------|--------|
| Incremental dispatch (perf, also affects whether full S=500 re-eval is feasible) | #41 | Open |
| Cost-aware reload (CT7/CT11 trade-off) | #42 | Open |
| CT14–18 in DAG instead of hook penalty | #43 | Open |
| CT12 max-modulation enforcement | #44 | Open |
| CT6 lower-bound check in greedy | #45 | Open |
| BKS comparison for ROADEF data | #46 | Open |
| `reference_solve.py` rewrite for ROADEF format | #47 | Open |
| SIMD/bitmask dispatch (deferred perf) | #48 | Open |

## 7. New gaps surfaced by this audit (not already filed)

The follow-ups below are **new** — they are not previously covered by
#41–#48. Filed during this audit:

| New issue | Title |
|-----------|-------|
| #80 | Nuclear: CT10 AMAX (max stock before refuel) not enforced |
| #82 | Nuclear: CT1 Type 1 commitment relaxation inflates objective |
| #83 | Nuclear: terminal full-scenario re-evaluation of best schedule |
| #84 | Nuclear: route scheduling penalties through ViolationManager (not objective) |
| #85 | Nuclear: CT21 weekly aggregate uses single-timestep pmax_t |
| #86 | Nuclear: vendor / fetch ROADEF 2010 competition data into the repo |

Note also: **CT11 minimum end-of-horizon fuel** (per-plant minimum
residual stock floor implied by some ROADEF instance variants) — the
simulator applies the credit but not the floor. Not filed separately
since `data0.txt`-style instances may not exercise it; revisit after #86
lets us scan the actual ROADEF instance set for this constraint.

Note: **CT15 parser may mis-set `period_end`** — the dual-meaning of an
`end <token>` line in the parser (line ~568 of `data.h`) is fragile;
flagged in section 2b but not filed because no instance is currently
loaded that exercises it. Revisit after #86.
