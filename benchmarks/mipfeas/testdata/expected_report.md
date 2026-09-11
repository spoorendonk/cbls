# MIPfeas parity report

Roster `roster.csv` (13 instances), 2.0s per instance-engine pair. Table: `expected_comparison.csv`.

CBLS against CP-SAT restricted to its `fj` + `ls` subsolvers under
`num_violation_ls` -- the reference implementation of the same jump-based
algorithm. This is a correctness-and-parity sweep against that one worker
pairing, never a claim against any solver's default portfolio (epic #87).

> **WIRING CHECK, NOT A PUBLISHABLE RESULT.** The MIPfeas roster is 233 instances; this report covers 13. These numbers are not comparable to a MIPfeas score, and the two engines' relative standing on a subset need not hold on the full roster.

> **INCOMPLETE RUN** — every aggregate below covers only the jobs that ran: cpsat: 1 not run.

## 1. Defects

| counter | cbls | cpsat | total |
|---|---|---|---|
| verification_failed | 1 | 0 | 1 |
| unverified | 1 | 0 | 1 |
| below_reference | 1 | 0 | 1 |
| invalid_model | 0 | 1 | 1 |
| errored | 1 | 0 | 1 |
| not_run | 0 | 1 | 1 |
| shape_flagged | - | - | 1 |
| trace_degraded | - | - | 1 |

**8 defect finding(s) in this run.**

`verification_marginal` is not a defect and is not counted above: cbls=0, cpsat=0.

## 2. Feasibility parity

| engine | feasible | of roster |
|---|---|---|
| cbls | 8 | 13 |
| cpsat | 9 | 13 |

Comparable on **8 of 13** instances -- the denominator for every set below. An instance drops out when either engine's row never ran, did not search, or had its solution withheld: such a row cannot say whether feasibility was reached.

- Agreement: **6/8** (5 both feasible, 1 neither).
- **cbls only** (1) -- feasible for cbls, not for cpsat: only-cbls
- **cpsat only** (1) -- feasible for cpsat, not for cbls: only-cpsat

Both feasible (5): degraded-trace, gen-ip054, mad, shape-mismatch, slow-start

Neither feasible (1): neither

Excluded from the parity sets:

| instance | engine | why |
|---|---|---|
| invalid-for-cpsat | cpsat | did not search (status invalid_model) |
| killed-job | cbls | did not search (status killed) |
| missing-job | cpsat | not run |
| rejected-solution | cbls | withheld (fail: row_violation) |
| unverified-row | cbls | withheld (unverified: no_verdict_file) |

## 3. Job failures

| instance | engine | kind | reason |
|---|---|---|---|
| invalid-for-cpsat | cpsat | invalid_model | MODEL_INVALID |
| killed-job | cbls | killed | exceeded 902.0s wall clock |
| missing-job | cpsat | not_run | no result file was written for this job |
| rejected-solution | cbls | verification fail | row_violation -- row 4.5 (4.5e+06x tol at R7) |
| unverified-row | cbls | verification unverified | no_verdict_file |

## 4. Model-shape cross-check

Variable counts must agree exactly: every reader enumerates the same MPS COLUMNS section, so a difference there is a reader defect, never benign. Constraint counts may differ by exactly the free rows the baseline keeps -- the MPS `N` rows after the first, which is the objective. OR-Tools' ModelBuilder holds each remaining `N` row as a linear constraint with infinite bounds; the CBLS adapter drops them, and so does SCIP. A free row constrains nothing, so the two programs have the same feasible set and the difference is benign. Any other difference, in either direction, is flagged. The excuse also needs a THIRD reading to stand: the free-row count is self-reported by the baseline, so it is certified by the reader under test. A difference is benign only when the independent checker recorded a count and that count agrees with the CBLS adapter. Where no verdict exists -- neither engine reported a solution on that instance, or the checker was killed -- the excuse is unchecked and the row is flagged, because an instance both engines failed is exactly where a translation defect is the likeliest explanation for the double failure.

Cross-checked 11 of 13 instance(s): the rest carry no shape from at least one engine (a killed job, a job that never ran, or a runner that died before publishing one) and are reported `not_compared`, not `agree`.

| instance | kind | counts | verdict | explanation |
|---|---|---|---|---|
| mad | constraints | cbls=51 cpsat=52 checker=51 free_rows=1 | benign | cpsat keeps 1 free row(s) the CBLS adapter drops |
| shape-mismatch | constraints | cbls=40 cpsat=44 checker=40 free_rows=1 | FLAGGED | not accounted for by the free rows the baseline keeps |

## 5. Trace health

| engine | genuine | of which single-point | degraded | unrecorded | denominator |
|---|---|---|---|---|---|
| cbls | 10 | 9 | 0 | 0 | 10 |
| cpsat | 8 | 6 | 1 | 0 | 9 |

Denominator: rows the engine **reported feasible** -- including any whose objective was later withheld, which section 2 excludes, so this count can exceed the feasible count there. The profile exists either way, and its health is a fact about the harness rather than about the verdict. A run that found nothing has no incumbent profile to have, so it is not counted here. A single-point column counts genuine profiles that nonetheless record at most one incumbent: not a defect (one improvement is one improvement), but a run where it is most of the denominator is one to look at before quoting the anytime aggregate. A degraded row is one whose profile collapsed to the final objective alone -- a harness condition (a changed log format, a callback that stopped firing), not a search result, and it scores near the no-solution penalty either way.

- cpsat degraded: degraded-trace

## 6. Anytime quality vs CP-SAT's fj + ls subsolvers under num_violation_ls (Primal Integral; shifted geometric mean is the primary ranking)

| engine | sgm | mean | median | iqr | scored |
|---|---|---|---|---|---|
| cbls | 0.8827 | 1.1918 | 1.0455 | [0.6386, 2.0000] | 11 |
| cpsat | 0.9828 | 1.2148 | 1.1325 | [0.6277, 2.0000] | 12 |

Lower is better; the Primal Integral runs from 0 (optimal immediately) to 2 (never feasible) and is budget-relative, so this is comparable only to another table scored at the same budget. `scored` excludes rows that never ran and rows whose objective was withheld. It does **not** exclude a job the driver killed or a model the baseline rejected: those score the full no-solution penalty of 2.0 here, where section 2 leaves them out entirely. The two sections answer different questions -- parity asks who reached feasibility, the aggregate asks what a run of this budget delivered -- and the defect counters in section 1 are where such a row is meant to be read.

## 7. Where the time went

| engine | setup measured | setup median | setup max | max / budget | overruns |
|---|---|---|---|---|---|
| cbls | 12/12 | 0.002 | 3.204 (slow-start) | 160.2% | 1 (max +6.531s on slow-start) |
| cpsat | 11/11 | 0.002 | 1.440 (slow-start) | 72.0% | 0 |

Two different effects, kept in two columns because a single wall-clock number cannot tell them apart:

- **setup** (`setup_seconds`) is instance read + model build + bound propagation. It happens before the solve bracket, so no table this harness has published has ever measured it; it is not charged against the search, but it is charged against the wall clock a run has to be scheduled for.
- **overrun** is `solve_seconds` past the budget. Search initialisation is not bounded by the deadline, so the first batch of a large model runs to completion whatever the clock says. A row can overrun with a negligible setup time and vice versa.


## 8. Machine and run record

| field | value |
|---|---|
| host | fixture-host |
| platform | fixture-platform |
| cores | 8 (4 available to the process) |
| memory | 16.0 GiB |
| concurrency | **2 job(s) at a time**, large instances 1 at a time, CP-SAT 1 worker(s), address-space cap 6.0 GB |
| budget | 2.0s per instance-engine pair |
| engine commit | fixture |
| solver versions | ortools 9.15.6755, PySCIPOpt 6.2.1, Python 3.12.0 |
| reference file | `miplib2017-v36.solu` (sha256 9236602294c1a5ac) |
| started | 2026-01-01T00:00:00+00:00 |
| finished | 2026-01-01T00:01:00+00:00 |
| status | complete |

