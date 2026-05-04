# Pharma-GLSP — Model Fidelity Audit vs Goerler et al. (2020)

Status: **qualitative deviation**. CBLS solves a simplified macro-period
formulation. Numerical objectives are **not directly comparable** to the paper's
GLSP-RP results. This document supersedes / closes #63 (which raised the same
concern in narrative form) and formalises the gap, the consequences for
`comparison.csv`, and what re-running the audit will require post-#59.

Tracking issue: #76. Parent epic: #28. Related sub-issues: #59 (micro-period
rework simulation), #60 (cross-boundary changeover cost), #61 (structure-aware
LNS), #62 (Class C feasibility), #63 (comparison.csv mismatch — superseded by
this audit).

## 1. Source formulation

**Reference**

> A. Goerler, E. Lalla-Ruiz & S. Voß (2020). *A Late Acceptance Hill Climbing
> Metaheuristic for the General Lot-Sizing and Scheduling Problem with Rich
> Constraints and Rework.* Algorithms 13(6):138.
> DOI: <https://doi.org/10.3390/a13060138>

The paper's GLSP-RP is an extension of the General Lot-Sizing and Scheduling
Problem (Fleischmann & Meyr 1997) augmented with **rich constraints** and
**rework / shelf-life dynamics**. The decision horizon is a sequence of macro
periods T, each split into |M_t| micro-periods. Production decisions are taken
at micro-period granularity; demand and capacity are macro-period quantities.

### 1.1 Decision variables (paper)

| Symbol | Index | Type | Meaning |
|---|---|---|---|
| `x_{j,m}` | product j, micro-period m | binary | product j is set up in micro-period m |
| `y_{j,m}` | product j, micro-period m | continuous | production amount of j in m |
| `z_{i,j,m}` | i, j, m | binary | changeover from i to j at the start of micro-period m |
| `I_{j,t}` | j, t | continuous | serviceable end-of-macro-period inventory |
| `R_{j,m}` | j, m | continuous | reworked quantity of j in m |
| `D_{j,t}` | j, t | continuous | disposed defectives in macro period t |

The "GLSP-CS" variant preserves the setup state across micro-period and
macro-period boundaries — the next slot inherits the previous setup unless an
explicit `z_{i,j,m}` changeover fires.

### 1.2 Constraint set (paper, Eqs 1-25 — labels approximate)

| # | Constraint | Granularity | Role |
|---|---|---|---|
| C1 | Exactly one product set up per micro-period: `Σ_j x_{j,m} = 1` | micro | sequencing |
| C2 | Production only when set up: `y_{j,m} ≤ B · x_{j,m}` | micro | linking |
| C3 | Changeover detection: `z_{i,j,m} ≥ x_{i,m-1} + x_{j,m} - 1`, applied across **micro** boundaries **and across macro boundaries** | micro/macro-boundary | sequencing |
| C4 | Capacity per macro: `Σ_{j,m∈M_t} tp_j y_{j,m} + Σ_{i≠j,m∈M_t} st_{ij} z_{i,j,m} + (rework time) ≤ b_t` | macro | resource |
| C5 | Inventory balance with defect: `I_{j,t} = I_{j,t-1} + Σ_{m∈M_t}(1−Θ_{j,t}) y_{j,m} + R^{ok}_{j,t} − d_{j,t}` | macro | demand |
| C6 | Rework window: defectives produced in `m` may be reworked only within `Ω_j` micro-periods; otherwise disposed | **micro** | shelf-life |
| C7 | Rework capacity per micro-period uses `tp^R_j` per unit reworked (consumes machine time, competes with production) | micro | resource |
| C8 | Disposal accounts for the **excess of defectives over what could be reworked within window** (computed micro-period by micro-period, not lump-sum) | micro→macro | shelf-life |
| C9 | Min lot size: if a campaign is opened at all (one or more consecutive micros set up to j), the **total** must be ≥ κ_j | campaign | rich-constraint |
| C10 | GLSP-CS: setup state preserved across boundaries unless `z` fires | micro/macro | sequencing |

### 1.3 Objective (paper)

```
min   Σ_{i≠j,m} f_{ij} z_{i,j,m}                                     # changeover
    + Σ_{j,t}   h_j  I_{j,t}                                         # inventory holding
    + Σ_{j,m}   h^R_j · age(defect, m) · (defective in flight)       # rework holding
    + Σ_{j,t}   λ_j  D_{j,t}                                         # disposal
```

The **rework holding** term is the integral of defective inventory over the
micro-periods between production and rework completion. The disposal term is
**non-zero in the paper's CPLEX/LAHCM solutions** because tight micro-period
rework capacity (machine is busy producing; only `Ω_j` slots to rework before
expiry) routinely forces some fraction of defectives to disposal.

### 1.4 What makes GLSP-RP harder than vanilla GLSP

Three coupled mechanics, all keyed on **micro-periods**:

1. **Micro-period setup decisions** — a macro-period is not a single campaign;
   it can switch products at every micro-period, paying changeover cost each
   time. The sequence is a length-`|M_t|` word, not a length-J permutation.
2. **Cross-boundary changeover** — `z_{i,j,m}` fires whenever the active product
   changes, including at the macro-period boundary `t → t+1`.
3. **Rework window Ω_j coupled to micro-period schedule** — defectives must
   slot into rework time within `Ω_j` micros; if the machine is too busy
   producing within that window, the surplus is **disposed** at cost `λ_j`.

These three together drive the paper's reported objectives upward and are the
reason CPLEX takes 1,800 s on Class A.

## 2. Our model — constraint-by-constraint mapping

CBLS encoding lives in `benchmarks/pharma-glsp/glsp_model.h` and
`benchmarks/pharma-glsp/glsp_hook.h`. The decision space is **drastically
simplified**:

- One `ListVar(J)` per macro-period (length-J permutation, **not** length-|M_t|
  micro-period sequence)
- One `FloatVar` per (product, macro-period) for the **macro-period total** lot
  size
- Micro-periods exist only as scalar parameters used inside cost approximations

### 2.1 Mapping table

| Paper element | CBLS encoding | Status |
|---|---|---|
| `x_{j,m}` (micro setup) | replaced by `seq[t]` permutation of products | **simplified** — no notion of multiple slots per product per macro |
| `y_{j,m}` (micro production) | replaced by `lot[j][t]` macro lot size | **simplified** — micro-period production aggregated to macro |
| `z_{i,j,m}` (changeover) | implicit from adjacent pairs in `seq[t]` via `pair_lambda_sum` | **simplified** — only J−1 changeovers per macro, fixed at one campaign per product per macro |
| `I_{j,t}` (inventory) | `cum_supply − cum_demand` clamped at 0 | implemented |
| `R_{j,m}` / rework dynamics | not represented | **missing** |
| `D_{j,t}` (disposal) | static `max(0, Θ·lot − ReworkCap)` with closed-form `ReworkCap` | **simplified, structurally inert** (see §2.3) |
| C1 one-product-per-micro | not modelled (no micros) | **missing** |
| C2 production iff setup | not modelled (no micros) | **missing** (replaced by lot ≥ 0) |
| C3 changeover incl macro boundary | within-macro only via `pair_lambda_sum` | **simplified** — see §2.4 (issue #60) |
| C4 capacity | macro-level: `Σ tp_j lot[j][t] + setup_time(seq[t]) ≤ b_t` | implemented (macro-level only) |
| C5 inventory balance with defect | `(1−Θ_{j,t}) · lot[j][t]` aggregated cumulatively | implemented |
| C6 rework window | not modelled | **missing** (issue #59) |
| C7 rework capacity competes with production | not modelled | **missing** (issue #59) |
| C8 disposal from micro-period excess | static formula, see §2.3 | **simplified — disposal ≈ 0 in practice** |
| C9 min lot size | soft constraint on macro lot | implemented (per-macro, not per-campaign) |
| C10 GLSP-CS setup preservation | not applicable (no micros) | **missing** |

### 2.2 Objective — term-by-term

| Paper term | CBLS term | Status |
|---|---|---|
| Changeover `Σ f_{ij} z_{i,j,m}` over **all** micro-periods including macro boundaries | `Σ_t pair_lambda_sum(seq[t], f)` over within-macro adjacencies of one J-permutation | **simplified** — undercounts by both (a) micro-period switches and (b) macro-boundary changeovers |
| Inventory holding `Σ h_j I_{j,t}` | `Σ h_j max(0, cum_supply − cum_demand)` | implemented; agrees with paper at macro level |
| Rework holding `Σ h^R_j · age · defective` | `Σ h^R_j · (M/2) · Θ_{j,t} · lot[j][t]` — assumes mid-window age, no actual scheduling | **simplified approximation** |
| Disposal `Σ λ_j D_{j,t}` | `Σ λ_j max(0, Θ·lot − ReworkCap)` with `ReworkCap = Ω·b/(T·M)/tp^R` | **simplified, structurally near-zero** |

### 2.3 Why disposal is effectively dead in our model

For Class A (J=5, T=4, M=7, capacity ≈ 2·Σd):

```
ReworkCap_{j,t} = Ω_j · (b_t / T / M) / tp^R_j
                ≈ 3 · (b_t / 28) / 0.5
                ≈ b_t / 4.7
```

Defectives `Θ_{j,t} · lot[j][t]` are bounded above by `0.03 · 120 ≈ 3.6` per
macro lot at JIT-feasible sizes (Θ ≤ 0.03, demand ≤ 120). `ReworkCap` is on
the order of `b_t / 4.7` — hundreds for tight instances, low thousands for
loose ones — two to three orders of magnitude above defective mass. Disposal
trips only when `Θ · lot > ReworkCap`, which never happens in any class
A/B/D/E instance and basically never in C either. **Disposal cost is
structurally 0 in CBLS solutions** — confirmed by `verify_glsp.h`'s breakdown.

In the paper's CPLEX solutions, disposal is **non-zero** because the
micro-period-level rework capacity is shared with production and the rework
window is enforced exactly, not via a closed-form per-macro estimate. This is
the single largest objective-mass discrepancy and the core reason Class A/B
CBLS objectives are 2–3× lower than the paper.

### 2.4 Cross-boundary changeover (issue #60)

`pair_lambda_sum` in `src/dag.cpp` (line 145) iterates only adjacent pairs
within one `ListVar` — `for k in [0, |seq|-1) sum f(seq[k], seq[k+1])`. The
DAG has no operator that joins `seq[t].back()` to `seq[t+1].front()`, so the
T−1 macro-boundary changeovers are **uncosted**. For Class A this is
~$T-1 = 3$ extra changeovers · ~$E[f] \approx 250$ ≈ ~750 cost-mass missing.

### 2.5 Min lot semantics

Paper: a "campaign" is a maximal run of consecutive micros set up to one
product; min lot κ_j applies to the **campaign total**. In CBLS we apply κ_j
to the **macro lot total**, which is at most one campaign per (j, t) by
construction (one slot per product per macro). The two coincide only when the
paper's optimum likewise opens at most one campaign per (j, t) — generally not
the case for Class C.

## 3. JIT hook (`GLSPInnerSolverHook`)

Defined in `glsp_hook.h`. Triggered by the engine after each accepted SA move
or on reheat. It is a **macro-period heuristic**, not a faithful inner solver
for the paper's micro-period MIP. It **only writes lot sizes**; it never
touches the sequence.

### 3.1 Algorithm

1. **Setup-time read** — for each macro `t`, sum `setup_time[seq[t][k]][seq[t][k+1]]`
   for `k = 0..J-2`. (Same in-macro-only summation as the DAG.)
2. **JIT allocation** — `lot[j][t] ← demand[j][t] / (1 − Θ_{j,t})`. Just-enough
   production, zero serviceable holding, ignores capacity.
3. **Capacity spill** — for `t = T-1 .. 1`, if total time at `t` exceeds
   `capacity[t]`, push production to earlier `s < t` (latest first), in product
   order **by ascending holding cost** (cheapest-to-hold first), respecting
   per-period spare capacity. Defect-rate ratio scales the moved quantity so
   serviceable conservation holds.
4. **Min lot rounding** — if `0 < lot[j][t] < κ_j`, round up to κ_j when there
   was demand; else zero out.
5. **Apply** — write changed lots, trigger `delta_evaluate`.

### 3.2 Divergence from the source formulation

| Aspect | Paper / GLSP-RP | Hook |
|---|---|---|
| Granularity | micro-period MIP | macro-period heuristic |
| Rework window Ω_j | enforced exactly per micro | not modelled — disposal computed from a closed-form macro estimate elsewhere |
| Rework capacity vs production | shared on the same machine | rework time ignored in capacity check |
| Cross-macro setup carryover (GLSP-CS) | preserved unless `z` fires | not represented |
| Cost optimised | full paper objective on micros | macro-level proxy: serviceable holding + macro setup + closed-form disposal |
| Reasoning about defective lifetime | per-batch | none — defectives appear and disappear in the same macro |

### 3.3 What the hook gets right

- Macro-level capacity feasibility under the simplified model (the spill phase)
- Serviceable inventory minimisation under JIT semantics
- Stable interaction with SA: pure local repair, idempotent, cheap

### 3.4 What it does not get right

- It cannot create or evaluate disposal cost truthfully — issue #59 is exactly
  the rework-timing simulation that would fix this
- It cannot account for cross-boundary changeover — issue #60 lives in the DAG,
  not the hook, but the hook's setup-time reading would need updating too
- It cannot exploit GLSP-CS state carry-over to avoid extra changeover cost
- It does not consider rework processing time consuming machine capacity

## 4. Post-#59 preview

Issue #59 proposes simulating micro-period rework timing **inside the JIT
hook**, returning a corrected objective contribution. After it lands the model
will:

| Feature | Pre-#59 | Post-#59 (planned) | Paper |
|---|---|---|---|
| Disposal cost | ≈ 0 | computed from simulated rework window | exact in MIP |
| Rework holding age | mid-window (M/2) | per-defective from sim | exact |
| Rework time on machine | ignored | accounted in spill | exact |
| Micro-period sequencing decisions | absent | absent (still macro-level seq) | full micros |
| Cross-boundary changeover | missing | still missing unless #60 also lands | exact |

**Verdict for post-#59 audit re-run**: with #59 alone the disposal mass becomes
realistic and the objectives move into the paper's range. With #59 **and** #60
the changeover cost mass also matches. At that point a paper-aligned
`comparison.csv` row could be claimed, but the audit must be **re-executed**
with fresh runs and an apples-to-apples breakdown — promotion to "gap vs BKS"
language is not automatic.

The remaining **structural** gap (no actual micro-period sequencing decision)
will still mean CBLS picks at most one campaign per (j, t). For Class C
(M = 8, very tight capacity) this likely remains a meaningful simplification
that the audit should acknowledge in the post-#59 run.

## 5. Severity & decision

| Question | Answer |
|---|---|
| Is the comparison apples-to-apples today? | **No.** |
| Is the deviation quantitative (tunable) or qualitative (structural)? | **Qualitative.** Disposal is structurally absent; cross-boundary changeover is structurally absent; sequencing granularity is structurally simpler. |
| Can `comparison.csv` claim "gap vs published BKS" today? | **No.** |
| What may `comparison.csv` claim today? | Self-consistency of the simplified CBLS formulation across SA configurations only. Paper rows must be marked as **reference-only / different model**. |
| What needs to change for paper-aligned claims? | At minimum #59 (rework simulation). Strongly recommended also #60 (cross-boundary changeover). After both, re-run audit. |

## 6. Action taken

- `benchmarks/instances/pharma-glsp/comparison.csv` updated:
  - paper rows annotated `reference-only — different model`
  - CBLS row annotated `simplified-model — disposal & cross-boundary changeover absent`
  - file header note pointing at this document and at #76
- Issue #63 superseded by this audit and referenced for closure in the commit.

## 7. Follow-ups

All known concrete gaps already have open issues:

- #59 — micro-period rework timing simulation (largest gap)
- #60 — cross-boundary changeover cost (DAG / hook coordination)
- #61 — structure-aware LNS (search quality, not fidelity)
- #62 — Class C feasibility (search quality)
- #63 — `comparison.csv` framing — **superseded by this audit**

No new follow-ups filed: the structural gaps that remain after #59 + #60
(notably full micro-period sequencing à la GLSP-CS) would constitute a
different benchmark, not a fidelity fix; documenting them here suffices unless
the project later commits to that scope. If/when post-#59 the audit re-run
flags new gaps not covered above, file at that point.
