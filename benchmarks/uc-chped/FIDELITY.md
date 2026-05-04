# UC-CHPED Model Fidelity Audit

Filed against issue #73 (epic #25). Decides whether
`benchmarks/instances/uc-chped/comparison.csv` may legitimately claim
"gap vs published BKS" or only "CBLS-SA vs CBLS-ViolationLS"
self-consistency.

## Verdict (TL;DR)

| Item | Verdict |
|------|---------|
| Source-vs-implementation severity | **Quantitative + one open question** (objective form matches verbatim, hot/cold `t_cold=0` divergence is bounded, ramp-rate question unresolved) |
| SCIP reference vs source | **Cosmetic** (PWL valve-point bound ≈ 0.1 % at 50 segments; same ramp-free relaxation as our model) |
| Solver-internal-feasibility vs verifier | **Qualitative** under current code (#32 + #33 mean SA reports "feasible" while verifier counts 44 / 134 / 166 violations) |
| `comparison.csv` may claim | **CBLS-SA vs CBLS-ViolationLS self-consistency only** until #32, #33, #34, and #77 (ramp rates) are resolved. Pedroso bounds remain quoted as reference values; whether they apply to the same problem we solve is open. |

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

### 1.7 Ramp rates

Standard UC formulations (Carrión & Arroyo 2006; Kazarlis 1996; many
papers in the valve-point ED literature) include ramp-rate limits
`|P[i,t] − P[i,t−1]| ≤ ramp_i` when committed, plus separate startup
and shutdown ramp limits. Whether Pedroso 2014 specifically uses ramp
constraints when computing the Table 2 bounds we cite as "known LB / UB"
**could not be verified** in the course of this audit — we have access
to Pedroso's GPL instance-generation code
(`http://www.dcc.fc.up.pt/~jpp/code/valve/ucp_data.py`, mirrored into
`benchmarks/instances/uc-chped/data.py`), and that code carries **no
ramp-rate fields**. This suggests one of:

1. Pedroso's published bounds were computed on a *ramp-free* UC, in
   which case our model is faithful and `comparison.csv` gap rows are
   honest.
2. Ramp data was distributed separately from the instance file, and
   the bounds in Table 2 *do* assume ramps. In that case our model is a
   relaxation.

Without direct verification of the Pedroso paper text, we treat this as
unresolved — the conservative position is that ramps may apply and our
gap numbers may understate the real gap. See §2.7.

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

### 2.7 Ramp rates — **MISSING (Deviation #3, conditional)**

There is no ramp-rate constraint anywhere in `uc_model.h`. The instance
data (`data.py`, traceable to Pedroso's GPL ucp_data.py) does not carry
ramp-rate fields either. *If* Pedroso 2014's Table 2 bounds were
computed with ramp constraints, our problem is a *relaxation* of theirs
— the true LB for our problem is ≤ Pedroso's LB and the true UB is
≤ Pedroso's UB. Reporting "% gap to Pedroso UB" would then understate
the gap. *If* the bounds are ramp-free (consistent with their own
public instance-generation code), our model is faithful on this axis.

**Severity: qualitative if ramps apply, cosmetic otherwise.** This is
the largest single open question of the audit. Resolved either by
(a) reading Pedroso 2014 directly and either confirming the ramp-free
reading or (b) sourcing ramp data and adding the constraint to our
model + SCIP reference + verifier.

Filed as follow-up #77.

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

- **#33** — Default `is_feasible` tolerance is `1e-9`
  (`src/violation.cpp:85`). On the surface this is tight, but the
  violation magnitudes that leak through are dispatch-times-Pmax-scale
  (so an effective tolerance of `1e-9 · P_max` ≈ `4.55e-7` MW on
  Kazarlis unit 1). Combined with #32 the cumulative violation routinely
  exceeds the verifier tolerance of `1e-4`.

- **#34** — Min up / down constraints have only the global adaptive
  lambda. Per-constraint weight bumping (a la GLS / ViolationLS) would
  remove the chronic late-stage violations that drive
  `comparison.csv` rows to "INFEASIBLE".

- **#35, #36** — LNS destroy/repair currently destroys feasibility on
  24-period instances; structural awareness would help.

The fidelity audit does not propose changes to these — they are tracked
under #25 already and #32, #33, #34, #35, #36 will be re-evaluated under
the ViolationLS port (#64).

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
| Ramp rates (§2.7) | **Qualitative if applicable, cosmetic otherwise** | Pedroso 2014 paper text not directly verified; their public instance code lacks ramp data. Worst case: we solve a strict relaxation. |
| Solver-feasibility vs verifier (#32, #33) | **Qualitative** | "Feasible" rows in `comparison.csv` are unverified by our own checker on multi-period instances. |
| SCIP PWL approximation (§4.2) | Cosmetic | ~0.1 % objective error bound at 50 segments. |

### 5.2 Decision for `comparison.csv`

The Pedroso "1hr MIP" rows in `comparison.csv` may have been obtained
on a tighter (ramp-constrained) problem than the one our solver and
our SCIP reference attack. Reporting "gap vs Pedroso LB" is therefore
**potentially misleading** until either (a) the Pedroso paper text is
read and the ramp question resolved, or (b) ramp rates are added to our
model and our SCIP reference and we re-run the comparison.

In addition, the "INFEASIBLE" rows are partly real and partly an
artefact of #32 + #33; under ViolationLS (#64) they will be re-measured.

**Decision (this audit):**

1. Annotate `comparison.csv` so that:
   - the `note` column flags rows produced by our (possibly relaxed)
     formulation as "ramp-free", and
   - the `note` on Pedroso rows flags them as "ramp-constrained
     reference" pending verification.
2. Until #77 resolves the ramp question, `comparison.csv` may
   **only** claim self-consistency between CBLS-SA and (eventual)
   CBLS-ViolationLS, plus a bounded-PWL gap to our own SCIP reference.
   The Pedroso numbers stay in the file as historical reference rows
   but are not the basis of a published gap.

This audit does **not** delete the Pedroso rows — that would lose
information. It annotates them.

## 6. Follow-up issues filed

- **#77** — UC-CHPED: add ramp-rate constraints to match Pedroso 2014.
  Resolves Deviation #3 (§2.7) and the qualitative-severity finding.
- All other deviations either are already tracked (#32, #33, #34,
  #35, #36) or are cosmetic / vacuous on shipped instances (no issue
  filed).
