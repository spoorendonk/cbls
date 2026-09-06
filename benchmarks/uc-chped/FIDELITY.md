# UC-CHPED Model Fidelity Audit

Filed against issue #73 (epic #25). Decides whether
`benchmarks/instances/uc-chped/comparison.csv` may legitimately claim
"gap vs published BKS" or only "CBLS-SA vs CBLS-ViolationLS"
self-consistency.

## Verdict (TL;DR)

| Item | Verdict |
|------|---------|
| Source-vs-implementation severity | **Quantitative** (objective form matches verbatim, hot/cold `t_cold=0` divergence is bounded; the ramp-rate question is closed — the source is ramp-free too, §1.7) |
| SCIP reference vs source | **Cosmetic** (PWL valve-point bound ≈ 0.1 % at 50 segments; same ramp-free relaxation as our model) |
| Solver-internal-feasibility vs verifier | **Qualitative when this audit was written** (#32 + #33 meant the SA reported "feasible" while the verifier counted 44 / 134 / 166 violations). #33 and #34 have since been fixed; #32 is still open, so any current claim must come from a `--verify` run rather than from this row. |
| `comparison.csv` may claim | **A gap against the Pedroso Table 2 bounds**, now that #77 has settled that those bounds describe the same ramp-free problem (§1.7). Each measured row must carry the feasibility tolerance it was produced at and should be `--verify`-checked, because #32 is still open. |

The remainder of this document records the equation-by-equation evidence.

## 1. Source formulation

Primary reference: J. P. Pedroso, M. Kubo, A. Viana,
*Pricing and unit commitment in combined energy and reserve markets using
valve-point effects*, 2014 (Pedroso 2014). The cost coefficients trace back to:

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
`reference_solve.py:45` and in our `verify_uc_chped.h:176`.)

Total fuel = `Σ_t Σ_i y[i,t] · F_i(P[i,t])`.

### 1.3 Startup cost

Hot/cold startup model:

```
S_i(t) = a_hot_i  if unit was on within the last t_cold_i periods
       = a_cold_i otherwise
```

charged whenever `y[i,t]=1 ∧ y[i,t−1]=0` (and analogously vs the
pre-horizon initial state). Some sources (and our SCIP reference) treat
`t_cold = 0` specially: with no lookback window the unit can never be
"recently on" so the published convention varies — see §2.3 below. This
is *the* genuine ambiguity in the source: Pedroso 2014 does not pin it
down, and Kazarlis-style instances avoid the edge by giving the
`t_cold = 0` units `a_hot = a_cold/2` (so the choice is at most a 2× cost
on the cheapest small units).

### 1.4 Demand and spinning reserve

```
∀t: Σ_i P[i,t] ≥ demand[t]
∀t: Σ_i P_max_i · y[i,t] ≥ demand[t] + reserve[t]
```

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
fields**. The two agree, and both agree with our model.

When this audit was first written the paper text had not been read
directly, so the possibility that the Table 2 bounds assumed ramps was
left open as follow-up #77. Reading it settled the question and #77 was
closed as *not planned*: there is nothing to add. Our model is
ramp-free, the SCIP reference is ramp-free, and the source is ramp-free,
so the bounds we quote and the results we measure describe the same
problem. See §2.7.

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
- SCIP reference (`reference_solve.py:233–236`): empty window ⇒ always
  hot cost.
- Our verifier (`verify_uc_chped.h:184–193`): empty window ⇒ `was_on`
  defaults to false ⇒ cold cost (matches our model).

The model and verifier agree, but disagree with the SCIP reference. For
Kazarlis units 7/8/9 (1-indexed 8/9/10), `t_cold = 0` and
`a_hot = 30 = a_cold / 2`, so the per-startup discrepancy is at most
30 currency units. Across a 24-period horizon and the 9 affected units in
ucp40 / 30 units in ucp100 / 60 units in ucp200, the cumulative
discrepancy is bounded by `30 · n_starts`. **Severity: cosmetic on
ucp13/ucp40 (subdominant), quantitative on ucp100/ucp200.**

**Deviation #2 — pre-horizon lookback for `y_prev = 0` units.**
- SCIP reference (`reference_solve.py:251–255`): if a unit was OFF for
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

Matches §1.4 (≥ rewritten as ≤).

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
compares like with like.

**Severity: none.** This was the largest single open question of the
audit. It was resolved by reading Pedroso 2014 directly, and follow-up
#77 was closed as *not planned* — the model already matches.

### 2.8 Cross-cutting solver-quality issues

These are not formulation deviations but they corrupt the meaning of the
"feasible" annotation in `comparison.csv`:

- **#32** — `FloatIntensifyHook` does not enforce indicator/float
  coupling. When `y[u][t]` flips 1→0 the dispatch `p[u][t]` is not zeroed.
  This is a *solver* bug, not a model bug, but it produces solutions that
  the verifier rejects (44 / 134 / 166 errors on
  ucp13-3p / ucp13-12p / ucp13-24p). The constraint `P − P_max · y ≤ 0`
  *exists* in the model and is checked at every step; the issue is that
  the SA's "feasible" flag uses a tolerance loose enough (and a
  delta-evaluation order subtle enough) that the violation accumulates
  across moves.

- **#33 (fixed)** — the default `is_feasible` tolerance was `1e-9` when
  this audit was written. The complaint was that it is an *absolute*
  residual on constraint bodies of dispatch-times-Pmax magnitude (an
  effective `1e-9 · P_max` ≈ `4.55e-7` MW on Kazarlis unit 1), so
  combined with #32 the cumulative violation routinely exceeded the
  verifier's `1e-4`. The default is now `1e-6`
  (`cbls::kDefaultFeasibilityTolerance`, `include/cbls/violation.h`),
  matching SCIP's `numerics/feastol` and the verifier's own scale. The
  runner also states the tolerance explicitly per run and records it on
  every measured `comparison.csv` row (#103), so a published row can no longer
  become uninterpretable when that default moves again.

- **#34 (fixed)** — min up / down constraints had only the global
  adaptive lambda. Per-constraint weight bumping (a la GLS /
  ViolationLS) was expected to remove the chronic late-stage violations
  that drove `comparison.csv` rows to "INFEASIBLE"; the ViolationLS port
  supplies exactly that, and #34 is closed.

- **#35, #36 (closed as not planned)** — LNS destroy/repair was
  destroying feasibility on 24-period instances when this audit was
  written, and structure-aware destroy was proposed as the fix. Both were
  closed unimplemented: the observation was made on the SA search that
  the ViolationLS port (#64) replaced. Re-file against a current
  measurement, not against this bullet.

The fidelity audit does not propose changes to these — they are tracked
under #25 already, and the ViolationLS port (#64) has since landed. Of
the five, only #32 is still open.

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

Source: `benchmarks/chped/reference_solve.py:138–348` (`solve_uc_scip`).

### 4.1 Approximation level

- Valve-point cost is approximated by a piecewise-linear envelope with
  `n_pwl_segments=50` breakpoints over `[P_min, P_max]` per
  (unit, period). Encoded as the incremental SOS2-like formulation
  with binary indicators (`reference_solve.py:269–331`).
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

Let `Δ_seg = (P_max − P_min) / n_pwl_segments`. The PWL envelope can
deviate from the true cost on each segment by at most a quadratic
remainder term in the curvature. The valve-point sinusoid
`|d · sin(e · (P_min − P))|` has period `2π/e` and `e ≈ 0.04` for
Sinha-13/Taipower-40 units, so one cosine cycle spans `≈ 157 MW`. With
`Δ_seg ≈ (P_max − P_min)/50` — about `(455 − 150)/50 = 6.1 MW` for a
Kazarlis 455 MW unit — the PWL has ~25 segments per cycle. Per-segment
maximum-curvature error is bounded by `(1/8) · d · (e · Δ_seg)^2`, i.e.
`(1/8) · d · (0.04 · 6.1)^2 ≈ d · 0.0074` ≈ `5.2` currency units for
Kazarlis-large `d = 700`. Across a 24-period horizon with ~13 committed
units this accumulates to a few hundred currency units, i.e. **~0.1 %**
of a 466 k objective. The quadratic-base term `c · P^2` is also PWL'd
but its curvature is much smaller, so it contributes far less.

This is a worst-case envelope. The expected error is smaller because
the PWL is *exact* at every breakpoint and the cosine peaks/troughs do
not all align with mid-segment.

In other words, the SCIP "optimum" reported in
`comparison.csv` is an approximation but the bound is small (~0.1 % at
50 segments). The Pedroso 2014 bounds `LB / UB` are computed with a
different MIP package; the encoding details (PWL count, model form) are
not given in their Table 2. **Every gap percentage in this benchmark is
conditioned on a piecewise-linear surrogate at some segment count.**

### 4.3 Conclusion on the SCIP reference

`reference_solve.py` solves the same relaxed-no-ramps problem we do,
modulo:
- a PWL approximation worth a few percent of the objective, and
- the `t_cold = 0` startup-cost convention difference (§2.3, bounded by
  ~30 currency units per startup on the affected small units).

It is therefore a defensible *upper bound* on the true optimum of our
problem, not a published BKS.

## 5. Severity & decision

### 5.1 Severity classification

| Aspect | Severity | Rationale |
|--------|----------|-----------|
| Valve-point cost form | Cosmetic | Equation matches verbatim. |
| Min up/down semantics | Cosmetic | Rolling-window matches Pedroso. |
| Demand & reserve | Cosmetic | Matches Pedroso exactly. |
| Hot/cold `t_cold = 0` (§2.3) | Cosmetic to quantitative | ≤30 currency units per affected startup; <0.1 % of objective on shipped instances. |
| Pre-horizon `y_prev=0` lookback (§2.3) | Cosmetic | Vacuous on shipped instances (n_init ≥ t_cold for all off units). |
| Ramp rates (§2.7) | None | Pedroso 2014 states no ramp constraints and their public instance code carries no ramp data. Our model, the SCIP reference and the source all solve the same ramp-free problem; #77 closed as not planned. |
| Solver-feasibility vs verifier (#32) | **Qualitative, reduced** | #33 and #34 are fixed and the tolerance is now recorded per row; #32 is still open, so a published row should carry a `--verify` verdict rather than the engine's `feasible` flag alone. |
| SCIP PWL approximation (§4.2) | Cosmetic | ~0.1 % objective error bound at 50 segments. |

### 5.2 Decision for `comparison.csv`

The ramp question this section was originally written around is closed
(§1.7, §2.7): the Pedroso "1hr MIP" rows and our solver attack the same
ramp-free problem, so "gap vs Pedroso LB / UB" is a like-for-like
comparison, bounded on the SCIP side by the ~0.1 % PWL error of §4.2.

What remains is a reporting question rather than a formulation one. The
"INFEASIBLE" rows of the original table were partly real and partly an
artefact of #32 + #33; #33 and #34 are fixed, #32 is still open.

**Decision (as revised):**

1. Every measured row records the feasibility tolerance it was produced
   at, plus the seed, the time budget and the engine commit. That is what
   `benchmarks/uc-chped/uc_chped.cpp` now writes (#103). The previous
   rows carried none of it and had to be deleted when the engine default
   moved from `1e-9` to `1e-6`.
2. `feasible` and `verified` are separate columns because they are
   separate tolerances: the engine's recorded `feas_tol`, against the
   verifier's own `1e-4` on its UC-semantic checks plus the `1e-6` of the
   `cbls::verify_model()` pass it runs first. While #32 is open,
   `verified` is the column to
   trust, and a row that fails it publishes no objective and no gap.
3. The Pedroso numbers stay in the file as cited reference rows. The
   generator re-emits them from each instance's `known_bounds` map, and
   refuses to write at all unless every rostered instance loaded, so a
   regeneration cannot silently drop them.

This audit does **not** delete the Pedroso rows — that would lose
information. It annotates them.

## 6. Follow-up issues filed

- **#77** — UC-CHPED: add ramp-rate constraints to match Pedroso 2014.
  **Closed as not planned.** Reading the paper showed it has no ramp
  constraints, so §2.7 is not a deviation and there is nothing to add.
- **#103** — give the runner a `comparison.csv` writer and an explicit
  feasibility tolerance, so §2.8's failure mode — a published row that
  becomes uninterpretable when an engine default moves — cannot recur.
  The measurement pass it unblocks is tracked separately as **#131**.
- All other deviations either are already tracked (**#32**, still open;
  **#33** and **#34**, fixed; **#35** and **#36**, closed as not
  planned) or are cosmetic / vacuous on shipped instances (no issue
  filed).
