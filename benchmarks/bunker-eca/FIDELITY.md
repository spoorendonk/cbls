# Bunker-ECA Model Fidelity Audit

Fidelity audit conducted under epic #64 (measurement quality). Companion audits:
#73 (uc-chped), #74 (nuclear-outage), #76 (pharma-glsp). Tracking issue: #75.

This benchmark is unique among the four because **both sides** of the comparison
(CBLS model and SCIP reference) are under suspicion. The audit's purpose is to
decide what `comparison.csv` may credibly claim about Bunker-ECA results.

## TL;DR (verdict)

| Aspect | Verdict |
|--------|---------|
| Source formulation | **Inferred** — no single paper covers routing + speed + bunker + ECA together. Three cited papers (Vilhelmsen 2014; Tamburini 2025; Fagerholt 2015) inspire pieces; the synthesis is ours. |
| Published BKS | **None exist** for these instances — the instances are synthetic factory functions in `data.h` / `data.py`, not literature instances. |
| CBLS model fidelity vs intended formulation | **Quantitative** simplifications on top of an inferred formulation: avg fuel coeff (#53), single per-cargo speed var (#54), avg prices, aggregate per-ship tank balance, 0.99 ECA threshold. |
| SCIP reference fidelity vs intended formulation | **Qualitative** errors: wrong fuel formula (#49, linear in v), missing constraints (#50), different ECA semantics (#51), different price averaging (#52). |
| Cross-validation tooling | **Not implemented** — `cross_validate.py` is open as #55. |
| `comparison.csv` claim allowed | **Self-consistency only.** Cannot claim "gap vs published BKS" (no BKS exist) and cannot claim "gap vs SCIP" (SCIP is solving a different, easier problem). The single SCIP row that has a number (small-3s-10c, $1,350,768) is from a known-broken formulation and must be annotated as such. |
| Severity classification | **Qualitative** — both sides differ in *what problem* they solve, not only in objective magnitude. |

## 1. Source formulation (inferred)

No single paper specifies "routing + speed + bunker + ECA fuel switching".
The benchmark synthesises features from three sources (see `README.md`):

- Vilhelmsen, Lusby, Larsen (2014) — TSRSPBO base structure, tank dynamics.
- Tamburini, Lange, Pisinger (2025) — ship parameters, speed optimisation.
- Fagerholt et al. (2015) — single-vessel ECA fuel-switching, the inspiration
  for the BoolVar `eca_fuel[c]`.

Because no canonical formulation exists, the "intended" model is itself a design
choice that has shifted over the implementation. Below is a best-effort
reconstruction from `data.h`, `bunker_eca_model.h`, and the issue history.

### 1.1 Decision variables (intended)

- `assign[c] ∈ {0..V}`: ship for cargo c (0 = unassigned, only allowed for
  spot cargoes).
- `speed[c,v] ∈ [v_min_v, v_max_v]`: sailing speed when cargo c is on ship v.
  *(per-(cargo, ship) — see #54.)*
- `eca_fuel[c] ∈ {0,1}`: MGO (1) vs HFO (0) on ECA portion of cargo c's leg.
  Sentinel "no var needed" if `eca_fraction == 0`.
- *(Optionally: `bunker[v,r,t]` purchase quantities, like Vilhelmsen — but the
  current benchmark abstracts these into "average price" terms.)*

### 1.2 Fuel consumption (intended)

Daily consumption is **cubic** in speed: `daily_fuel(v) = k_v · v³` MT/day.
A leg of distance `d` nm at speed `v` (knots) takes `d/(24·v)` days, so
total fuel for the leg:

```
fuel_leg = k_v · v³ · d / (24 · v) = k_v · v² · d / 24      [MT]
```

This is the **quadratic-in-speed** form used in `bunker_eca_model.h:121-126`
and is the formula `README.md` advertises. SCIP's reference (#49) currently
uses `k · v · d / 24` (linear in v) which is wrong.

### 1.3 ECA fuel switching (intended)

Each leg has `eca_fraction ∈ [0,1]` (fraction of distance inside an ECA).
Within the ECA, regulation requires low-sulfur MGO. Outside, HFO is allowed
(and cheaper). Two reasonable semantics exist:

- **Strict (Fagerholt-like)**: any positive `eca_fraction > 0` forces MGO on
  the ECA portion. The decision is *de facto* not a decision — it is forced.
- **Soft / threshold**: a leg with very small `eca_fraction` (e.g. <0.01) may
  be treated as "no ECA" because the data is approximate.

The current code uses the threshold semantics with cutoff 0.99 (CBLS model)
or 0.0 (SCIP reference) — see #51 for the mismatch.

### 1.4 Fuel split (intended)

```
mgo_consumed[c] = fuel_leg[c] · eca_fraction[c] · eca_fuel[c]
hfo_consumed[c] = fuel_leg[c] − mgo_consumed[c]
```

That is, when `eca_fuel = 1`, MGO covers exactly the ECA fraction, HFO covers
the rest. When `eca_fuel = 0`, HFO covers the whole leg (regulatory
non-compliance, penalised only via the `eca_fraction ≥ 0.99` constraint).

### 1.5 Workload / capacity caps (intended)

- **Per-ship fuel capacity:** `Σ hfo_consumed[c] over c on v ≤ initial_hfo_v +
  (hfo_tank_max_v − initial_hfo_v) − hfo_safety_v`. Equivalently: ship cannot
  consume more HFO than max usable tank.
- **MGO tank** (intended but not modeled): analogous cap on
  `Σ mgo_consumed[c] over c on v`.
- **Workload balance:** `count(cargoes on v) ≤ max(4, ⌈C/V⌉·3)`.

### 1.6 Price aggregation (intended)

Bunker prices vary by region and time (`bunker_options[]` and per-region
`hfo_price` / `mgo_price`). Realistic costing requires linking
`hfo_consumed[c]` to the price at the port where the fuel was purchased.
The current benchmark abstracts this into a **single fleet-wide average
price** for HFO and MGO — but the two solvers disagree on what to average
(#52: SCIP averages over regions, CBLS averages over bunker_options).

### 1.7 Objective (intended)

```
maximise   Σ_c revenue[c] · active[c]
         − avg_hfo_price · Σ_c hfo_consumed[c]
         − avg_mgo_price · Σ_c mgo_consumed[c]
         − Σ_c port_cost[c] · active[c]
```

## 2. CBLS model — line-by-line

Source: `benchmarks/bunker-eca/bunker_eca_model.h`.

### 2.1 Variables (lines 56-84)

| Item | Code | Notes |
|------|------|-------|
| `assign[c]` IntVar `[lb,V]` | 56-60 | `lb=1` if contract, `0` if spot. Matches intent. |
| `speed[c]` FloatVar | 62-72 | **Single var per cargo** with **fleet-wide bounds** (min over all ships' v_min, max over all ships' v_max). Cannot enforce per-(cargo, ship) bounds — see #54. For small/medium instances all ships share `[11.0, 14.5]`, so no impact in practice. Heterogeneous fleets (Vessel-D=`[11.5,14.0]`, Vessel-E=`[10.5,14.5]`, Vessel-F=`[11.0,14.0]`) in medium+ allow speeds outside individual ship envelopes. |
| `eca_fuel[c]` BoolVar | 75-84 | Created only when `eca_fraction > 0`; sentinel `-1` otherwise. Matches intent. |

### 2.2 Indicators (lines 88-105)

`on_v[c][v] = 1 ⟺ assign[c] == v+1`, built from
`if_then_else(|assign − (v+1)| − 0.5, 0, 1)`. `active[c] = 1 − on_v[c][0]` for
spot, `1` for contract. Standard CBLS pattern, matches intent.

### 2.3 Fuel consumption (lines 107-141)

| Item | Code | Notes |
|------|------|-------|
| `avg_fuel_coeff` | 110-114 | **Simplification** — fleet-wide average instead of per-ship `k_v`. Issue #53 tracks impact. Fuel-cost gradient is dominated by `v²` not `k`, so the impact on solution structure is likely small, but **unquantified**. Verify also uses the average (verify_bunker_eca.h:32-36) so the verifier doesn't catch this mismatch. |
| `alpha = avg_fuel_coeff · dist / 24` | 121-124 | Matches the cubic-derived `k·v²·d/24`. ✓ |
| `fuel_per_cargo[c]` | 125-126 | `alpha · speed² · active[c]`. ✓ |
| `mgo_consumed[c]` | 134-135 | `fuel · eca_frac · eca_fuel[c]`. ✓ |
| `hfo_consumed[c]` | 136 | `fuel − mgo`. ✓ |

### 2.4 Price aggregation (lines 143-160)

Lines 144-160. Uses **bunker_options average if any, else regions average**.
For all current instances bunker_options is non-empty so the bunker_options
branch is taken. Issue #52 is about the SCIP side (regions); the CBLS side is
self-consistent. In medium+ instances bunker_options carry small price
variations (`1 + 0.02·sin(...)`), so the CBLS average sits very close to the
nominal regional prices but is not identical.

### 2.5 Objective (lines 163-187)

Revenue + fuel cost + port cost. Matches intent §1.7. ✓

### 2.6 Constraints

| # | Code | Notes |
|---|------|-------|
| Fuel capacity (HFO) | 193-205 | `Σ on_v[c][v] · hfo_consumed[c] ≤ initial + (max − initial) − safety`. Note `initial + (max − initial) − safety = max − safety`, i.e. max usable tank. Matches intent §1.5. ✓ |
| Time windows | 208-227 | `dist/(24·speed) ≤ available · active[c]`. ✓ Note: when `active=0` the constraint becomes `dist/(24·speed) ≤ 0`, but speed is bounded ≥ v_min > 0 and dist > 0, so the LHS is positive. In CBLS this is handled by penalty rather than by requiring speed = 0; speed for unassigned cargoes is meaningless and the BunkerSpeedHook drives it to v_min, so the violation can be small but **non-zero**. This is a subtle modelling artifact worth tracking. |
| ECA compliance | 230-236 | `eca_fraction ≥ 0.99 ⇒ eca_fuel = 1`. **Threshold mismatch with SCIP** (#51, SCIP enforces for any > 0). Also: only enforced when `eca_fraction ≥ 0.99`, so a leg with 0.5 ECA fraction has no compliance constraint at all — `eca_fuel = 0` is legal. This is the **likely root cause of #57** (ECA vs noECA producing identical profits): for non-fully-ECA legs, the solver picks `eca_fuel = 0` (HFO), so MGO usage is always zero on those legs. Setting all `eca_fraction = 0` then changes nothing. |
| Workload | 238-247 | `Σ on_v[c][v] ≤ max(4, ⌈C/V⌉·3)`. ✓ |
| MGO tank cap | — | **Missing.** Per intent §1.5 there should be a per-ship `Σ mgo_consumed[c] ≤ mgo_max_v − mgo_safety_v` constraint analogous to HFO. The data carries `mgo_tank_max` / `mgo_safety` but they are unused. |

### 2.7 BunkerSpeedHook (`bunker_speed_hook.h`)

For each cargo: if assigned, set `speed = clip(dist/(24·available), v_min, v_max)`
— minimum feasible speed minimises the convex `α·v²` fuel cost. If unassigned,
set `speed = v_min`. Then delegates to `FloatIntensifyHook`.

This is correct under the intended model; the inner solver is not where
fidelity issues arise.

### 2.8 Verifier (`verify_bunker_eca.h`)

Mirrors the model exactly: same `avg_fuel_coeff`, same average prices, same
0.99 ECA threshold. **Therefore the verifier cannot detect any of the
simplifications listed in this audit** — it only catches deltas between the
DAG evaluation and an independent recomputation under the same simplified
formula. A "passes verification" stamp does **not** imply the solution is
feasible under the intended formulation.

## 3. SCIP reference — line-by-line

Source: `benchmarks/bunker-eca/reference_solve.py`. All issues below are open.

| Issue | Location | Description | Status |
|-------|----------|-------------|--------|
| #49 fuel formula | 195-196 | `fc ≥ k·d/24·blend·speed − M·(1−x)` is **linear in speed**, modelling daily consumption ≈ `k·v²` (instead of `k·v³`). Intended `k·v²·d/24` requires a quadratic constraint `fc ≥ k·d/24·blend·speed² − M·(1−x)`. **Open.** |
| #50 missing constraints | n/a | Per-ship HFO capacity and workload balance are not in the SCIP model (#50 scope). SCIP's feasible region is therefore strictly larger than CBLS's, biasing SCIP toward higher objectives. (MGO tank cap is missing on **both** sides — see #78.) **Open.** |
| #51 ECA threshold | 111-118 | SCIP forces `eca_mgo[c,v] ≥ x[c,v]` for **any** `eca_fraction > 0`. CBLS only forces it for `≥ 0.99`. Different problems. **Open.** |
| #52 price averaging | 183-184 | SCIP averages prices **over regions**: `sum(r["hfo_price"]) / n_regions`. CBLS averages **over bunker_options** (when any exist). Numerically close but not equal. **Open.** |

Additional observations from re-reading `reference_solve.py`:

- **S1. Linearised fuel cost drives speed to lower bound** (filed as #81).
  Lines 102-105 do `speed ≥ v_min·x` and `speed ≤ v_max·x`, forcing
  `speed = 0` when `x = 0`. The linearised fuel cost (line 196)
  `fc ≥ coeff·speed − M(1−x)` has `coeff > 0`, so the LP relaxation drives
  `speed` to its lower bound to minimise fuel. Combined with #49 (linear-in-v
  fuel), SCIP's "optimal" speed is always `v_min`. This is likely why the
  small-instance "optimal" exists in 60 s — the formulation is borderline
  trivial.

- **S2. No travel-time-via-speed constraint** (filed as #81). Line 150 uses
  `min_sailing = dist / (24 · v_max)` as a *constant*. Actual sailing time
  depends on the `speed[c,v]` variable, but SCIP linearises away this
  dependency. So SCIP assumes travel at `v_max` (for time-window
  feasibility) but pays fuel for a free `speed` (which the LP drives to
  `v_min`). The two speeds are inconsistent within a single solution —
  a serious modelling error.

- **S3. Spot vs contract revenue accounting (no issue — correct).** Lines
  211-215 sum `revenue[c] · x[c,v]` over all `(c,v)`. Combined with the
  contract constraint `Σ_v x[c,v] = 1` (line 95) and spot constraint
  `Σ_v x[c,v] ≤ 1` (line 98), this is correct.

S1 and S2 are two facets of the same modelling error and have been filed
together as **#81**.

## 4. Cross-validation status (#55)

- `cross_validate.py` does not exist in this repo.
- Once both formulations agree, cross-validation would (a) load a SCIP
  solution into CBLS and verify feasibility, (b) fix all CBLS variables in a
  SCIP model and check objective parity. Tolerance on objective should be
  ~1 e-3 once #49–#52, #78, #79, #81 are all aligned.
- Until both sides resolve their formulation issues, cross-validation will
  detect *known* mismatches and provide no new information.

## 5. Severity & decision

### Severity classification

- **CBLS model** vs intended formulation: **quantitative**. Each
  simplification (avg fuel coefficient, single speed var, fleet-wide speed
  bounds, missing MGO cap, 0.99 ECA threshold) shifts objective and possibly
  feasibility but does not change the *problem class*. Bounded by #53, #54,
  #78, #79.
- **SCIP reference** vs intended formulation: **qualitative**. The fuel
  formula is the wrong polynomial degree (#49); the time-window formulation
  decouples sailing time from speed (#81); fuel cost is linearised so the LP
  drives speed to its lower bound (also #81). SCIP is solving a *different
  and easier* problem.
- **Both vs each other**: **qualitative** — they enforce different ECA rules
  (#51), different fuel-degree models (#49), different price aggregation
  (#52), and SCIP is missing per-ship caps (#50).

### Decision for `comparison.csv`

**Allowed claims (annotated as such):**

- CBLS-with-ECA vs CBLS-noECA self-consistency. Currently equal on all
  instances (#57), confirming the ECA constraint is not biting under the
  current data + 0.99 threshold.
- CBLS feasibility (verified by `verify_bunker_eca.h`, with the caveat in
  §2.8: same formulation as the model).
- Engine throughput numbers (iters/s) are valid as engineering metrics.

**Not allowed claims:**

- "CBLS gap vs published BKS" — there is no BKS for these synthetic
  instances.
- "CBLS gap vs SCIP optimum" — SCIP is solving a different, weaker problem
  (see §3). Reporting the SCIP number as a baseline would invert the
  apparent quality (SCIP looks better because it was given an easier
  problem). The single existing SCIP row (small-3s-10c, $1,350,768) must be
  annotated as "buggy formulation, not comparable".
- "CBLS-noECA shows ECA cost impact" — false, currently zero impact for the
  reasons in §2.6 row "ECA compliance".

**Once #49, #50, #51, #52, #81 (SCIP side) and #78, #79 (CBLS side) are
resolved, `cross_validate.py` (#55) can be implemented and run. After that —
and only after that — the comparison can claim a meaningful SA-vs-SCIP gap.**

## 6. Follow-ups (new gaps not covered by #49–#57)

The audit identified three new gaps not already tracked in #49–#57. They
have been filed under epic #27:

### #78 (CBLS): missing per-ship MGO tank cap

`Ship.mgo_tank_max` and `Ship.mgo_safety` are populated in `data.h` for every
instance but `bunker_eca_model.h` adds no per-ship MGO consumption ≤ usable
tank constraint. The HFO equivalent is at lines 193-205. Severity: low —
MGO usage is currently always zero (#57), so the missing cap never bites.
Severity rises to "must fix" if/when #51 (ECA threshold alignment) lands and
MGO becomes non-zero.

### #79 (CBLS): time-window constraint on inactive cargoes

`bunker_eca_model.h:208-226` writes the time-window constraint as
`dist/(24·speed) ≤ available · active[c]`. When `active[c] = 0` the RHS is
zero, but `speed[c] ≥ v_min > 0` and `dist > 0`, so the LHS is strictly
positive. The constraint is therefore violated for any unassigned cargo
unless the solver drives `speed[c]` to infinity (which it cannot — it is
bounded above by `v_max`). The `BunkerSpeedHook` mitigates this by setting
`speed` to v_min for unassigned cargoes, but this still leaves the constraint
violated. Visible as a small residual penalty in the SA loop.

Cleaner formulations:
- Big-M: `dist/(24·speed) − available ≤ M·(1 − active[c])`.
- Or replace `available · active[c]` with `available + LARGE·(1 − active[c])`.

### #81 (SCIP): linearised fuel cost decouples speed from sailing time

`reference_solve.py:135-150` uses `min_sailing = dist/(24·v_max)` as a
*constant* in the time-window constraint, but the SCIP `speed[c,v]` variable
is free in `[0, v_max]`. Sailing time and fuel cost should depend on the
same `speed` variable. The current formulation lets SCIP pin `speed` to a
fuel-minimising value while pretending sailing time used `v_max`. Will
require a quadratic constraint
`travel_time[c,v] · speed[c,v] · 24 ≥ dist · x[c,v]` with the consequent
nonlinearity, or a piecewise-linear approximation with breakpoints in
`[v_min, v_max]`.


## 7. Audit metadata

- Audit date: 2026-05-04.
- Branch: `audit/bunker-eca-fidelity` (worktree at
  `~/code/my/cbls/.claude/worktrees/agent-aa4c33e7f063c1b20`).
- Reviewer: see commit history.
- Re-audit triggers: any change to the cubic fuel model, the ECA threshold
  semantics, or the data factories in `data.h`.
