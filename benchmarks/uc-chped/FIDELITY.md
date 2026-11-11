# UC-CHPED Model Fidelity Audit

Filed against issue #73 (epic #25). Decides whether
`benchmarks/instances/uc-chped/comparison.csv` may legitimately claim
"gap vs published BKS" or only self-consistency. It was written while the
engine was the SA solver, which the ViolationLS port (#64) has since
replaced, so the old "CBLS-SA vs CBLS-ViolationLS" alternative no longer
arises; the verdict below is a gap against Pedroso's bounds.

## Verdict (TL;DR)

| Item | Verdict |
|------|---------|
| Source-vs-implementation severity | **Quantitative** (objective form matches verbatim, startup costs match the source — every startup on these instances is cold in both (§7.4), and only the SCIP reference diverges, pricing `t_cold = 0` startups hot; the ramp-rate question is closed — the source is ramp-free too, §1.7) |
| Instance data vs source (§7, #148, #193) | **`ucp13` and `ucp40` identical.** Checked field by field against the authors' `ucp_data.py`. #148 found `ucp40` capping units 19-20 at 500 MW where the source has 550 (exact 1-period optimum 55704.72 against their 55644.79); #193 corrected it, and the exact solve now reproduces 55644.79. `ucp13` reproduces both proven optima exactly. |
| SCIP reference vs source | **Quantitative** (§7.4: the PWL chord error at the valve-point cusps is up to 1.8 % of a 1-period `ucp13` at 50 segments, not the ≈ 0.1 % first claimed; a `t_cold = 0` startup is priced hot where the source prices it cold; demand is `>=` where the source has `=`. Same ramp-free problem otherwise) |
| Solver-internal-feasibility vs verifier | **Resolved** (§2.8): the SA-era mismatch (#32, #33) is closed; #32's hook-level coupling gap is real and untouched, but the coupling is a scored constraint, and a probe found no verifier rejection. Any current claim must still come from a `--verify` run rather than from this row. |
| `comparison.csv` may claim | **A gap against the Pedroso Table 2 bounds for `ucp13` and `ucp40`.** #77 settled that those bounds describe the same ramp-free problem (§1.7), #148 that `ucp13` is the same instance, and #193 that `ucp40` is, once its data was corrected (§7.5). A `ucp40` result measured before #193 solved a different instance and carries no valid gap. Each measured row must carry the feasibility tolerance it was produced at and should be `--verify`-checked, because the verifier is the independent check and the engine's own `feasible` flag is not. |

The remainder of this document records the equation-by-equation evidence.

## 1. Source formulation

Primary reference: J. P. Pedroso, M. Kubo, A. Viana,
*Unit commitment with valve-point loading effect*, DCC-2014-05 /
arXiv:1404.4944, 2014 (Pedroso 2014; sources in §7.1). The cost coefficients
trace back to:

- 13-unit: Sinha, Chakrabarti, Chattopadhyay,
  *Evolutionary programming techniques for economic load dispatch*,
  IEEE Trans. EC 7(1), 2003.
- 40-unit: Niu et al. Taipower system, also reused widely in the valve-point
  ED literature.
- UC parameters (`min_on`, `min_off`, `t_cold`, hot/cold startup, initial
  state) come from Kazarlis, Bakirtzis, Petridis,
  *A Genetic Algorithm Solution to the Unit Commitment Problem*,
  IEEE Trans. PWRS 11(1), 1996.

The unit-commitment-with-valve-point formulation as published in Pedroso
2014 (and reproduced in Niu/Sinha-style instances) is:

### 1.1 Decision variables

- `y[i,t] ∈ {0,1}` — commitment of unit `i` in period `t`.
- `P[i,t] ∈ [0, P_max_i]` — dispatch (continuous).

### 1.2 Objective

Minimise total cost = fuel cost + startup cost.

Per-unit fuel cost when committed:

```
F_i(P) = a_i + b_i·P + c_i·P^2 + |e_i · sin(f_i · (P_min_i − P))|
```

The two valve-point terms `(e_i, f_i)` are stored in the instance as
`(d, e)` in our codebase — i.e. our `d` is the *amplitude* and our `e` is
the *frequency*. (See `benchmarks/instances/uc-chped/data.py`,
docstring and column `[d, e]`, and the cost expression in
`reference_solve.py`'s `_unit_breakpoints` and in our `verify_uc_chped.h:176`.)

Total fuel = `Σ_t Σ_i y[i,t] · F_i(P[i,t])`.

### 1.3 Startup cost

Hot/cold startup model:

```
S_i(t) = a_hot_i  if unit was on within the last t_cold_i periods
       = a_cold_i otherwise
```

charged whenever `y[i,t]=1 ∧ y[i,t−1]=0` (and analogously vs the
pre-horizon initial state). With `t_cold = 0` there is no lookback window,
and conventions vary (§2.3). The paper's text leaves it open, but the
authors' `ucp_valve.py` forbids a hot start whenever `t_cold < min_off` or
`t_cold = 0` (§7.4). Every Kazarlis unit has
`t_cold < min_off`, so every startup on these instances is cold — as in our
model and verifier. Only the SCIP reference prices `t_cold = 0` startups
hot (`a_hot = a_cold/2`, so 30 instead of 60 per startup on the smallest
units).

### 1.4 Demand and spinning reserve

```
∀t: Σ_i P[i,t] = demand[t]
∀t: Σ_i P_max_i · y[i,t] ≥ demand[t] + reserve[t]
```

Demand is an **equality** in the source (paper §2; `ucp_valve.py`'s
`demand(t)` row). Our model, verifier and SCIP reference use `≥`: a
relaxation, since a non-monotone valve-point cost can make overproduction
cheaper. It leaves the 1- and 3-period proven optima unchanged (§7.4).

The reserve constraint is *committed-capacity-based*, not
dispatch-based: it counts every committed unit's full capacity, not its
current dispatch.

### 1.5 Min up / min down

```
y[i,t] = 1  ∧  y[i,t−1] = 0   ⇒   y[i,τ] = 1 for τ ∈ [t, t + min_on_i − 1]
y[i,t] = 0  ∧  y[i,t−1] = 1   ⇒   y[i,τ] = 0 for τ ∈ [t, t + min_off_i − 1]
```

The "rolling-window" form. Pedroso 2014 uses this form exactly. Initial
condition: if the unit starts the horizon ON for `n_init_i` periods, it
must remain ON for `max(0, min_on_i − n_init_i)` more periods, and
symmetrically for OFF.

### 1.6 Dispatch limits

```
y[i,t] = 1  ⇒  P_min_i ≤ P[i,t] ≤ P_max_i
y[i,t] = 0  ⇒  P[i,t] = 0
```

Equivalently: `P_min_i · y[i,t] ≤ P[i,t] ≤ P_max_i · y[i,t]`.

### 1.7 Ramp rates — absent from the source

Standard UC formulations (Carrión & Arroyo 2006; Kazarlis 1996; many
papers in the valve-point ED literature) include ramp-rate limits
`|P[i,t] − P[i,t−1]| ≤ ramp_i` when committed, plus separate startup
and shutdown ramp limits. **Pedroso 2014 does not.** Its formulation
states power balance, spinning reserve, unit initial conditions and
minimum up/down times only
(<https://web.fc.up.pt/dcc/Pubs/TReports/TR14/dcc-2014-05.pdf>), and the
GPL instance-generation code behind the shipped data
(`http://www.dcc.fc.up.pt/~jpp/code/valve/ucp_data.py`, mirrored into
`benchmarks/instances/uc-chped/data.py`) carries **no ramp-rate
fields**. Our model and the SCIP reference are ramp-free too, so the bounds
we quote and the results we measure describe the same problem (#77, closed
as not planned; §2.7).

## 2. Our model — equation by equation

Source file: `benchmarks/uc-chped/uc_model.h` (217 lines).

### 2.1 Variables (`uc_model.h:24–35`)

`y[u][t]` as `m.bool_var(...)`, `p[u][t]` as `m.float_var(0, P_max_u, ...)`.

Matches §1.1.

### 2.2 Fuel cost (`uc_model.h:69–77`)

```cpp
auto base_cost   = a + b·P + c·P^2;
auto pmin_minus_p = P_min − P;
auto valve_point  = |d · sin(e · (P_min − P))|;
auto fuel_cost    = y · (base_cost + valve_point);
```

Matches §1.2 verbatim. The `(d, e)` ↔ `(amplitude, frequency)`
convention is consistent with `data.py` and with the verifier
(`verify_uc_chped.h:176`).

### 2.3 Startup cost (`uc_model.h:79–113`)

Detection: `su = max(0, y[t] − y_prev)`. Correct (rolling-window
startup indicator). The hot/cold dispatch logic walks `[t − t_cold, t−1]`
and flags `was_on = max(y[τ] for τ in window) > 0.5`.

**Deviation #1 — `t_cold = 0` semantics.**
- Our model (line 102–104): empty window ⇒ always cold cost.
- SCIP reference (`reference_solve.py`'s `_add_startup_cost_constraints`): empty window ⇒ always
  hot cost.
- Our verifier (`verify_uc_chped.h:184–193`): empty window ⇒ `was_on`
  defaults to false ⇒ cold cost (matches our model).

The model and verifier agree, but disagree with the SCIP reference. For
Kazarlis units 7/8/9 (1-indexed 8/9/10), `t_cold = 0` and
`a_hot = 30 = a_cold / 2`, so the per-startup discrepancy is at most
30 currency units. Across a 24-period horizon and the 3 affected units in
ucp13 / 12 in ucp40 / 30 in ucp100 / 60 in ucp200, the cumulative
discrepancy is bounded by `30 · n_starts`. **Severity: quantitative, in the
SCIP reference only.** Our model and verifier match the source, which prices
every startup cold (§7.4). On the proven `ucp13`-3p and `ucp40`-1p optima the
difference is 90 and 240 (§5.1).

**Deviation #2 — pre-horizon lookback for `y_prev = 0` units.**
- SCIP reference (`reference_solve.py`'s `_hot_start_lookback`): if a unit was OFF for
  `n_init` periods but `n_init + t < t_cold`, treats the unit as
  potentially hot-startable.
- Our model and verifier ignore this and always treat
  pre-horizon-OFF as cold-eligible only.

In the published instances, `n_init` for off units always equals their
`min_off` (8/8/5/5/6/3/3/1/1/1) which already meets or exceeds `t_cold`
(5/5/4/4/4/2/2/0/0/0) for every unit, so this divergence is **vacuous
on our shipped instances**. Documenting it for completeness.

### 2.4 Demand (`uc_model.h:121–129`)

```cpp
demand[t] − Σ_u p[u][t] ≤ 0
```

`≥`, rewritten as `≤`. The source has `=` (§1.4). This is a
relaxation, and it leaves the proven 1- and 3-period optima unchanged (§7.3).

### 2.5 Spinning reserve (`uc_model.h:131–141`)

```cpp
demand[t] + reserve[t] − Σ_u P_max_u · y[u][t] ≤ 0
```

Matches §1.4.

### 2.6 Dispatch limits and min up/down (`uc_model.h:143–204`)

- `P_min_u · y − P ≤ 0` and `P − P_max_u · y ≤ 0` — matches §1.6.
- Min up: `y[t] − y[t−1] − y[τ] ≤ 0` for `τ ∈ (t, t + min_on)`.
  Matches §1.5.
- Min down: `y[t−1] − y[t] + y[τ] − 1 ≤ 0` for `τ ∈ (t, t + min_off)`.
  Matches §1.5.
- Initial conditions on `y_prev`: matches §1.5 closing paragraph.

### 2.7 Ramp rates — absent, matching the source

There is no ramp-rate constraint anywhere in `uc_model.h`. Neither the
source formulation (§1.7) nor the instance data (`data.py`, traceable to
Pedroso's GPL ucp_data.py) has one, so this is **not a deviation**: our
problem is not a relaxation of Pedroso's, and "% gap to Pedroso LB / UB"
compares like with like. **Severity: none.**

### 2.8 Cross-cutting solver-quality issues

Not formulation deviations: five issues filed when the engine was the SA solver,
as corrupting the meaning of `comparison.csv`'s "feasible". All are closed.

- **#32 (closed, not planned)** — `FloatIntensifyHook` does not zero `p[u][t]`
  when `y[u][t]` flips 1→0. **The hook-level gap is real and still present; the
  consequence claimed (verifier errors on feasible rows) is not.**
  `P − P_max · y ≤ 0` is a *scored* constraint (`uc_model.h`), and the engine
  calls an assignment feasible only when every constraint node is within
  `feas_tol`, so `p ≤ 1e-6` wherever `y = 0`, inside the verifier's `1e-4`.
  `solve()` re-runs `full_evaluate` on the best state before returning, so the
  published `max_violation` is a fresh recomputation, not an incremental one; a
  stale-value false positive would show as `verify_model()`'s
  `ConstraintViolation` on the `p − P_max·y` node and as the UC check's
  `dispatch above Pmax*y`. A probe at `6f9d419` (ucp40/100/200, horizons 1-24,
  five seeds, pre-#193 data) found **55 feasible rows, 55 verified, 0
  rejections**, residuals ≤ 2.91e-11. It covers only rows the engine called
  feasible and leaves out `ucp13`.
- **#33 (fixed)** — the default tolerance was an absolute `1e-9`; it is now
  `1e-6` (`cbls::kDefaultFeasibilityTolerance`, `include/cbls/violation.h`),
  SCIP's `numerics/feastol` and the verifier's scale, and the runner records it
  on every measured row (#103).
- **#34 (fixed)** — min up/down had only a global adaptive lambda; the
  ViolationLS port's per-constraint weights supply what it asked for.
- **#35, #36 (closed as not planned)** — LNS destroying feasibility on
  24-period instances, and structure-aware destroy as the fix: observed on the
  SA search only. Re-file against a current measurement.

## 3. Verifier — what it checks

Source: `benchmarks/uc-chped/verify_uc_chped.h`. Defaults `tol = 1e-4`.

| # | Check | Source map | Faithful? |
|---|-------|------------|-----------|
| 1 | `y ∈ {0,1}` | §1.1 | yes |
| 2 | `P_min·y ≤ p ≤ P_max·y` | §1.6 | yes |
| 3 | `Σ_u p[u,t] ≥ demand[t]` | §1.4 | yes |
| 4 | `Σ_u P_max_u · y[u,t] ≥ demand[t] + reserve[t]` | §1.4 | yes |
| 5 | min up rolling window | §1.5 | yes |
| 6 | min down rolling window | §1.5 | yes |
| 7 | initial on/off remainder | §1.5 | yes |
| 8 | objective recomputation: `Σ_t Σ_i y[i,t]·F_i(P[i,t]) + Σ S_i(t)` | §1.2, §1.3 | yes (matches our model's `t_cold=0` convention, §2.3) |

**Not checked:** ramp rates (§2.7) — consistent with the model.

The verifier is therefore consistent with our model. It is *not* a check
against the source formulation; it is a check against
"what we said we built". That distinction matters for any "VERIFIED"
column in comparison output.

## 4. SCIP reference — what it actually solves

Source: `benchmarks/uc-chped/reference_solve.py` (`solve_uc_scip`).

### 4.1 Approximation level

- Valve-point cost is approximated by a piecewise-linear envelope with
  `n_pwl_segments=50` breakpoints over `[P_min, P_max]` per
  (unit, period). Encoded as the incremental SOS2-like formulation
  with binary indicators (`reference_solve.py`'s `_add_pwl_cost`).
- Min up/down: same rolling-window form as our model.
- Startup cost: hot/cold via auxiliary binary `w[u,t]`, with the
  pre-horizon lookback handling described in §2.3 (Deviation #1, #2).
- Demand and reserve: same as ours.
- Time limits per period count: 60 s (1p), 120 s (3p), 300 s (6p),
  600 s (12p), 3600 s (24p).
- Ramp rates: **also not modelled** — consistent with our omission, so
  the SCIP reference and our CBLS model solve the *same* relaxed
  problem.

### 4.2 Worst-case bound on the SCIP "optimum"

A curvature bound (`(1/8)·d·(e·Δ_seg)^2` per segment, ~0.1 % of a 24-period
objective) is wrong here: `|sin|` has a cusp at every valve point, uniform
breakpoints straddle the cusps, the chord over one overestimates by up to
about `d·e·Δ/2`, and optimal dispatch sits at cusps (#148, §7.4). Measured as
each unit's worst chord error, summed: 213 for one period of `ucp13` at 50
segments (1.8 %), 47 at 200.

The Pedroso 2014 bounds `LB / UB` are **not** conditioned on a surrogate:
`ucp_valve.py` refines its breakpoints until the linearised lower bound
meets an upper bound priced at the true cost, so Table 2 bounds the true
valve-point objective. Only the SCIP reference column is conditioned on a
piecewise-linear surrogate at some segment count.

### 4.3 Conclusion on the SCIP reference

`reference_solve.py` solves the same relaxed-no-ramps problem we do,
modulo:
- a PWL approximation whose worst case is 1.8 % of a 1-period `ucp13` at 50
  segments (realised far less, §7.4), and
- the `t_cold = 0` startup-cost convention difference (§2.3: 30 currency
  units per affected startup, 90 and 240 on the proven `ucp13`-3p and
  `ucp40`-1p optima, §7.4).

It is therefore **not** a bound on the true optimum of our problem in
either direction (§7.4): the chord can sit above or below the cost, and the
hot pricing of `t_cold = 0` startups undercuts the source. At 200 segments it
lands 0.17 % *below* the proven `ucp13`-3p optimum and 0.38 % below the
proven `ucp40`-1p optimum (55434.65 against 55644.79, on the data as corrected
by #193; 0.44 % below the old instance's 55704.72 before it). For an exact value use the MINLP in
`instance_identity.py` (§7; in git history at `047f2d4c`).

## 5. Severity & decision

### 5.1 Severity classification

| Aspect | Severity | Rationale |
|--------|----------|-----------|
| Valve-point cost form | Cosmetic | Equation matches verbatim. |
| Min up/down semantics | Cosmetic | Rolling-window matches Pedroso. |
| Demand & reserve | See below | Reserve matches; demand is `≥` here, `=` in the source. |
| Hot/cold `t_cold = 0` (§2.3) | None in our model; quantitative in the SCIP reference | Our model, verifier and the source all price every startup cold on these instances (§7.4). The SCIP reference prices `t_cold = 0` startups hot: 30 per startup, 90 (0.23 %) and 240 (0.43 %) on the proven `ucp13`-3p and `ucp40`-1p optima. |
| Pre-horizon `y_prev=0` lookback (§2.3) | Cosmetic | Vacuous on shipped instances (n_init ≥ t_cold for all off units). |
| Ramp rates (§2.7) | None | Pedroso 2014 states no ramp constraints and their public instance code carries no ramp data. Our model, the SCIP reference and the source all solve the same ramp-free problem; #77 closed as not planned. |
| Solver-feasibility vs verifier (#32) | **Resolved** | #33 and #34 are fixed, the tolerance is recorded per row, and #32 closed as not planned after a probe found no verifier rejection in 55 feasible rows. **The `FloatIntensifyHook` coupling gap itself is untouched** — it still does not zero `p` when `y` flips — but no engine-feasible solution has ever shown the consequence, because the coupling is a scored constraint. A published row should still carry a `--verify` verdict rather than the engine's `feasible` flag alone. |
| SCIP PWL approximation (§4.2) | Quantitative | Not ~0.1 %: cusp chord error up to 1.8 % of a 1-period `ucp13` at 50 segments (§7.4). Not a bound either way. |
| Demand `≥` vs source `=` (§1.4) | Cosmetic on the measured horizons | A relaxation; leaves all three proven optima unchanged (§7.4). |
| `ucp40` instance data (§7) | **Resolved** (#193) | `P_max` of units 19-20 was 500, not 550. Corrected; Table 2's `ucp40` rows are bounds for this instance again. |

### 5.2 Decision for `comparison.csv`

The Pedroso "1hr MIP" rows and our solver attack the same ramp-free problem
(§1.7, §2.7), so "gap vs Pedroso LB / UB" is a like-for-like comparison —
**for `ucp13` and `ucp40`**, the instances §7 found identical to the source
(`ucp40` since #193). The SCIP reference is not a bound (§4.3). What remains
is a reporting question.

**Decision:**

1. Every measured row records the feasibility tolerance it was produced
   at, plus the seed, the time budget and the engine commit. That is what
   `benchmarks/uc-chped/uc_chped.cpp` writes (#103), so a row cannot become
   uninterpretable when an engine default moves.
2. `feasible` and `verified` are separate columns because they are
   separate tolerances: the engine's recorded `feas_tol`, against the
   verifier's own `1e-4` on its UC-semantic checks plus the `1e-6` of the
   `cbls::verify_model()` pass it runs first. `verified` is the column to
   trust, and a row that fails it publishes no objective and no gap.
3. The Pedroso numbers stay in the file as cited reference rows. The
   generator re-emits them from each instance's `known_bounds` map, and
   refuses to write at all unless every rostered instance loaded, so a
   regeneration cannot silently drop them.

## 6. Follow-up issues filed

All closed: #77 (ramps; not planned, §1.7), #103 (the `comparison.csv` writer;
its measurement run is #257), #32-#36 (§2.8), #148 and #193 (instance
identity, §7), #194 (extended horizons within capacity; README, **Extended
Horizons**). The other deviations are cosmetic or vacuous on the shipped
instances.

## 7. Instance data — do we solve the published instances? (#148)

Sections 1-6 audit the *formulation*. This one audits the *instance data*: are
`ucp13` and `ucp40` the instances Table 2's bounds were computed on? It was
settled against the authors' own artefacts, not against the paper's prose.

### 7.1 Sources

§1's citation once gave a wrong title. The report at
<https://web.fc.up.pt/dcc/Pubs/TReports/TR14/dcc-2014-05.pdf> is
J. P. Pedroso, M. Kubo, A. Viana, *Unit commitment with valve-point loading
effect*, DCC-2014-05, also arXiv:1404.4944 (<https://arxiv.org/abs/1404.4944>,
v1 2014-04-19). That PDF URL returned 404 on 2026-10-02; the arXiv copy is
the one read here. Its Table 2 is the source of the ten bounds, which it gives
to two decimals (`ucp13` 1p 11701.28, 3p 38849.84; `ucp40` 1p 55644.79). The
paper says the data and programs are at its Internet page,
<http://www.dcc.fc.up.pt/~jpp/code/valve/>, which is still live and holds,
fetched 2026-10-02:

| File | Server date | Bytes | sha256 |
|---|---|---:|---|
| `ucp_data.py` (instance data, GPL-3) | 2013-12-09 | 36510 | `3d5b8f078550fea50c98a42389257db9f7cda891190a9676b43b29fcb18934c2` |
| `ucp_valve.py` (the solver, Gurobi) | 2013-12-09 | 17563 | `c1f6e8d9ba316c6291e0eab447a3df65a773adacb1c996a610f68c62d46d9104` |
| `RESULTS/ucp13-1.txt` | 2013-12-04 | — | `96660c8a6ede9671a32e29b22def1a5b0f04ee99835adc9198b338eaa19eea86` |
| `RESULTS/ucp13-3.txt` | 2013-12-04 | — | `bc35a0d0f259b53a6096539993e9f06357b20a824128c85548fba2fa91da3920` |
| `RESULTS/ucp40-1.txt` | 2013-12-04 | — | `e74c19e14cfc7637529605aa841d09c08d2ff0a5006e6c607b49535a9e8a7706` |
| `RESULTS/eld40-10500.txt` (fetched 2026-10-03, #193; cited, not loaded by the script) | — | 51985 | `d9b4abc16a696d52c0d7f3c410bcd8d288bd1b37798acca3ef39a9643b80c320` |

None of it is vendored (GPL, and the check needs it once). The script records
these hashes (`UPSTREAM_SHA256`) and refuses any `ucp_data.py` or log that
does not match them, so a revised upstream file cannot silently change the
comparison. The `RESULTS/` logs
are the runs behind Table 2: the last `CURRENT ERROR: LB UP` line of each of
the ten `ucp13`/`ucp40` logs carries that row's UB to the printed digits and
its LB to within 0.5 (on the multi-period rows the log's final bound differs
from the table's in the first or second decimal), and each log ends with the
final schedule (`y`, startups, dispatch).

### 7.2 Provenance, field by field

`instance_identity.py` (removed once #148 closed; `git show
047f2d4c:benchmarks/uc-chped/instance_identity.py`) loads `ucp_data.py` (a stub supplies
the one `gurobipy` name it imports, `multidict`) and compares every field of the
authors' `ucp13(24)` and `ucp40(24)` with ours, exactly:

| Field | `ucp13` | `ucp40` |
|---|---|---|
| `a b c`, valve `d e` (theirs `e f`), `P_min` | identical | identical |
| `P_max` | identical | identical since #193 (**units 19 and 20 were 500 here, 550 there**) |
| `y_prev`, `n_init`, `t_cold`, `min_on`, `min_off`, `a_hot`, `a_cold` | identical | identical |
| `demand`, `reserve` (24 periods) | identical | identical |

The **unit-parameter mapping** is the authors' own: `ucp_data.py` annotates
each `ucp13`/`ucp40` unit with its "corresp. in kazarlis" unit, and our
`_UCP13_MAP` and `i % 10` reproduce those annotations. The **demand and
reserve profiles** are the authors' tables verbatim.

`ucp40` takes its costs and limits from `CHPED_40UNIT` in
`benchmarks/uc-chped/chped_data.py`, which
had `P_max = 500` for units 19-20 until #193 set them to the source's 550. The
authors' `eld40` optimum at 10500 MW (their Table 1, 121412.53, `CHPED_40UNIT`'s
`known_optimum`) dispatches those units at 511.28 MW each
(`RESULTS/eld40-10500.txt`). `CHPED_40UNIT` is now field-identical to the
authors' `eld40()` as well. `ucp100`/`ucp200` inherit the same values through
their `i % 40` cycle; they and `extend_horizon()` are this repository's
construction.

### 7.3 The check

Three independent ways, on the three proven-optimum rows:

1. **Re-price the authors' schedules on our data.** Parse each log's final
   schedule, price it with our coefficients (startups cold, see 7.4), and check
   it against our constraints (dispatch limits, demand to the logs' 3-decimal
   print rounding, reserve, min up/down).
2. **Exact solve, no linearisation.** SCIP MINLP with the true
   `|d·sin(e·(P_min − P))|` term (SCIP handles `sin` by spatial
   branch-and-bound, so the bound is global), `limits/gap 0`, one thread, a
   600 s cap. The authors' demand row is an **equality** (paper §2 and
   `ucp_valve.py`); ours is `>=` (§1.4), so both were
   solved. Off units are kept out of the fuel term without a bilinear product:
   `p = 0` when off, and the valve auxiliary `v >= ±d·sin(…) − d·(1 − y)` can
   reach 0 because `|d·sin(…)| <= d`.
3. **The existing PWL reference** (`reference_solve.py`'s `solve_uc_scip`,
   unchanged) at 200 segments, four times the default.

As a control, 2 and 3 were also run on the **authors' own** `ucp40` (loaded from
`ucp_data.py`), which separates "our data differs" from "our method misses".

Machine: the shared 12-core box, other single-threaded jobs running.
PySCIPOpt 6.2.1, SCIP 10.0.2. Engine commit irrelevant (no CBLS solve);
branch commit of the script: see `git log 047f2d4c -- benchmarks/uc-chped/instance_identity.py`.

| Case | Table 2 | Re-priced authors' schedule | MINLP `=` | MINLP `>=` | PWL ref (200 seg) |
|---|---:|---:|---:|---:|---:|
| `ucp13` 1p | 11701.28 | 11701.285, feasible | **11701.280** (optimal, 0.5 s) | 11701.280 (bound 11701.280) | 11704.077 (8 s) |
| `ucp13` 3p | 38849.84 | 38849.872, feasible | **38849.841** (bound 38849.839) | 38849.841 (bound 38849.841) | 38784.832 (60 s) |
| `ucp40` 1p, ours before #193 | 55644.79 | 55644.848, **infeasible: reserve short by 31 MW** | **55704.724** (bound 55704.724) | 55704.724 (bound 55704.724) | 55458.427 (31 s) |
| `ucp40` 1p, authors' data | 55644.79 | 55644.848, feasible | **55644.793** (bound 55644.793) | 55644.793 (bound 55644.793) | 55434.648 (46 s) |
| `ucp40` 1p, ours after #193 | 55644.79 | 55644.848, feasible | **55644.793** (bound 55638.420, 300 s cap) | 55644.793 (bound 55644.793, 300 s cap) | 55434.648 (91 s) |

Every MINLP row of the #148 run but the first stopped at the 600 s cap with
the incumbent and the dual bound equal to the printed three decimals (absolute
gap < 0.01): SCIP cannot certify a literal zero gap on `sin`, so those are
optimal to that resolution rather than by status. The last row is #193's
re-run on the corrected data (2026-10-03, same PySCIPOpt 6.2.1, `--cases
ucp40-1 --time-limit 300 --pwl-segments 200`, a shared machine at a load
average near 14): the data diff reports `ucp40: IDENTICAL`, the authors'
schedule re-prices feasible, and both MINLPs find the published optimum to the
printed decimals. At the halved cap the `=` solve's bound stopped 6.4 short;
the `>=` relaxation closed, and its optimum is a lower bound on the `=` one,
so 55644.79 is certified for both. Every objective matches the authors'-data
control row.

**Tolerance.** The MINLP has no linearisation, so the only slack is SCIP's
(feasibility `1e-6`, the gap above) and Table 2's two-decimal rounding. A
match is `|Δ| < 0.01` on the objective, and that is what the authors' data
gives on all three rows (0.000, +0.001, +0.003). A ~0.1 % band, right for a
PWL comparison, could not have caught the pre-#193 `ucp40` (+59.93, +0.108 %):
a data difference can move an optimum by less than a linearisation error
does. The field diff (7.2) and the reserve infeasibility of the authors'
schedule (check 1) were the decisive evidence.

### 7.4 Formulation differences the check surfaced

Recorded here because the comparison exercised them; none moves a proven
optimum.

- **Demand is `=` in the source, `>=` here** (our model, verifier and SCIP
  reference). With a non-monotone valve-point cost, overproducing can be
  cheaper, so `>=` is a relaxation that *could* lower the optimum. On all five
  MINLP rows it does not (the two columns' incumbents agree), so the 1- and 3-period
  bounds are unaffected; longer horizons are unmeasured.
- **Startups.** `ucp_valve.py` forbids a hot start whenever `t_cold < min_off`
  or `t_cold = 0`, and every Kazarlis unit has `t_cold < min_off`. Since a unit
  must then stay off at least `min_off > t_cold` periods, every startup is
  cold in the source, and also in our model and verifier (hot needs a run
  inside the last `t_cold` periods, which `min_off` rules out; `t_cold = 0` is
  cold there by §2.3), so we match. The **SCIP reference does not**: it prices a
  `t_cold = 0` startup hot (30, not 60). The authors' `ucp13`-3p optimum starts
  three such units and their `ucp40`-1p optimum eight, so the reference
  undercuts those optima by 90 and 240 — which, with the chord error below, is
  why its column sits below Table 2.
- **The PWL reference is neither ~0.1 % nor an upper bound.** §4.2 bounded the
  chord error by curvature, but `|sin|` has a cusp at every valve point, and
  uniformly spaced breakpoints straddle them; over a cusp the chord sits above
  the cost by up to about `d·e·Δ/2`. Summing each unit's worst chord error,
  one period of `ucp13` is exposed to 213 at 50 segments and 47 at 200 (1.8 %
  and 0.4 % of 11701), `ucp40` to 564 and 167 on the data as corrected by #193 (552 and 164
  before it). Realised: +2.8 on `ucp13`-1p at
  200 segments; +25.0 on `ucp13`-3p and +29.9 on the authors'-data
  `ucp40`-1p, each measured from the proven optimum less the startup-pricing
  difference (38849.84 − 90 and 55644.79 − 240). Between cusps the chord can sit *below* the
  cost, and together with the startup convention that makes the reference's
  value a number near the optimum, not a bound on it (§4.3).
- **Table 2 is not conditioned on a PWL surrogate** (§4.2):
  `ucp_valve.py` refines breakpoints until the lower bound of its
  linearised model meets the upper bound priced at the *true* cost, so the
  bounds are bounds on the true valve-point objective.

### 7.5 Verdict and what changed

| Instance | Identity | Table 2 rows |
|---|---|---|
| `ucp13` | **Confirmed.** Field-identical, both proven optima reproduced to `< 0.01` by an exact solve. | Stand: bounds for this instance. |
| `ucp40` | **Confirmed since #193.** #148 found `P_max` of units 19-20 at 500 against the source's 550 (the authors' optimum infeasible here, the exact optimum 55704.72 against their 55644.79). #193 corrected it: field-identical, and the exact solve reproduces 55644.79. | Stand: bounds for this instance. #148's relabel is removed. |

A second consequence of the same 100 MW (#152): `ucp40` at 12 and 24 periods
was infeasible before #193 (period 12's demand plus reserve, 12628 MW, against
12622 MW of total `P_max`), and so was `ucp200`; with the source's limits
`ucp40` has 94 MW of slack there. The extended 48h/168h instances were short
for an unrelated reason until #194.

#193 regenerated every `ucp*.jsonl` built from the 40-unit data (only `P_max`
changed) and removed #148's interim relabelling of the `ucp40` rows as bounds
for a related system. `tests/python/test_uc_chped_instances.py` pins
the corrected state: 550 at units 19-20 in every 40-unit copy, and no
capacity-short period in any committed instance. The runner scores `ucp40`
against its own bounds as it does `ucp13`; a `ucp40` result measured before
#193 solved the old instance and has no gap against them.

The same data was the `chped` 40-unit dispatch instance (removed by #385; in git
history at `73ee2690`). Its `known_optimum`
(121412, rounded down from the authors' 121412.53, whose `eld40` log closes
LB/UB to 0.003) was not attainable on the old data, because its dispatch puts
units 19-20 at 511.28 MW. That dispatch is now feasible here, so the value is
attainable; it is the proven optimum of the authors' `=` instance, while our
model's demand row is `>=`, a relaxation not certified to share it (§7.4). The `chped` 40-unit test's iteration-bounded run did not attain
it (121635.52, +0.18 %); the test asserted only `objective >= known_optimum`
and `< 140000`.

Reproduce (from the repository root; fetch the upstream files and restore the
script from git history first):

```
git show 047f2d4c:benchmarks/uc-chped/instance_identity.py > benchmarks/uc-chped/instance_identity.py
mkdir -p U/RESULTS && cd U && curl -sLO https://www.dcc.fc.up.pt/~jpp/code/valve/ucp_data.py
for c in ucp13-1 ucp13-3 ucp40-1; do curl -sL -o RESULTS/$c.txt https://www.dcc.fc.up.pt/~jpp/code/valve/RESULTS/$c.txt; done; cd ..
.venv/bin/python benchmarks/uc-chped/instance_identity.py --upstream-dir U --time-limit 600 --pwl-segments 200
.venv/bin/python benchmarks/uc-chped/instance_identity.py --upstream-dir U --data upstream --cases ucp40-1
```

`--skip-solves` runs checks 7.2 and 7.3(1) alone, in about a second.

At HEAD the first command (600 s cap) reproduces the objectives of the `ours
after #193` row of §7.3; the row itself, including the `=` bound 55638.420,
which depends on the cap, needs `--time-limit 300 --cases ucp40-1`. The `ours
before #193` row needs `benchmarks/chped/data.py` as of `79ae972^`.
