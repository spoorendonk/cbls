# Does the closed-form jump scorer ping-pong on Float plateaus? (#178)

`LinearJumpScorer` scores a jump over linear comparison rows as
`cmp(p + r·Δ, q)` rather than by re-summing the row through the DAG. On
fractional data that is exact only to rounding, so, unlike the DAG probe, it is
not exactly antisymmetric: on an exactly balanced plateau both `x → j` and
`j → x` can score `+1 ulp`, and `apply_jump` accepts any score `> 0`. FJ could
then bounce between two points inside a batch where the probe would have bumped
weights. The header of `include/cbls/linear_jump.h` documents the hazard. This
file records the measurement of whether it happens.

## Pre-registration

Written and committed before the pilot and the campaign were run.

**Engine.** Main at `c09a8ca`. Both arms are built from that tree with the same
uncommitted instrumentation patch (counters only, below). Builds are Release with
an empty `CBLS_SANITIZE`, checked in each `CMakeCache.txt`.

- **Treatment**: the engine as it is.
- **Control**: the same tree compiled with `-DCBLS_PP_CONTROL`, which makes
  `LinearJumpScorer::prepare` return false (fall back to the DAG probe) for any
  Float variable. Bool/Int columns keep the closed form, and so does the Newton
  step's cached row partial, which is bit-identical to `compute_partial` by
  construction.

**Zero trajectory change from the patch.** Unpatched main and the instrumented
treatment are run at a fixed iteration budget with no wall clock
(`--no-time-limit --max-iterations 20000 --seed 1`). This covers all 50 MINLPLib
instances and uc-chped `ucp13` and `ucp40` at every horizon. Every result row and
every anytime-trace row must be identical apart from the timing columns. The
same run between treatment and control is reported as a deterministic auxiliary.
It does not enter the verdict. It was started while this file was being written,
and only its row and trace equality was inspected before the commit.

**Counters** (per `solve()` call, written as one JSON line):

- `api_batches`: FJ batches (the batch API that `solve()` drives) with at least
  one GLS iteration.
- `api_flat_nobump`: of those, batches that ended with **no weight bump** and an
  **unchanged unweighted violation**. The violation is the exact sum over the
  real, active rows that `refresh_unweighted_violation` computes, taken at batch
  entry and exit. "Unchanged" means within `kProgressRelEps · max(1, U0)` (1e-9
  relative), the same floor `track_batch_progress` uses. This is the issue's metric.
- **P (primary)** = `api_flat_nobump / api_batches`, per run.
- `rev_float_nobump_tiny` = **D (direct)**: an accepted jump of one Float
  variable from B back to A, following its own accepted A → B jump in the same
  batch with no weight bump in between, where both accepted scores are *tiny*.
  Tiny means `0 < s ≤ 1e-12 · Σ_{c ∈ G_v, w_c ≠ 0} w_c · max(1, Σ |child values
  of row c|)`: positive, but within rounding of the rows' magnitudes. Scores and
  weights are both read in the lazy-decay scaled space, so the ratio is
  scale-free. The probe is exactly antisymmetric, so on a probe-scored column this
  counter can only be nonzero through a change of candidates, not through
  rounding.
- Engagement: `float_prepare_fast` (Float prepares that took the closed form;
  zero in the control by construction) and `accepted_float_cf` (accepted Float
  jumps whose column is closed-form eligible under the current weights). The
  second is arm-independent.
- Also reported: `tiny_float`, `rev_float`, `rev_float_nobump`, `bumps`,
  `accepted_float`.

**Secondary metrics.** The runner's feasibility verdict and objective (gap to
BKS on MINLPLib, gap to the Pedroso bound on uc-chped).

**Pilot (control only, for admission).** Seed 1 at a 3 s budget. It runs all 50
MINLPLib instances (the runner's roster; the held-out roster of #144 does not
exist) and all 8 uc-chped instance files at every rostered horizon. An instance
is **admitted** to the campaign iff the control pilot shows `accepted_float_cf ≥
1` on it. For uc-chped the unit is the file, because the runner cannot select a
horizon. A Float column that never takes an eligible jump cannot ping-pong,
because treatment and control score it identically. A benchmark that admits
nothing has its answer from the pilot plus the fixed-iteration treat/control
auxiliary.

**Campaign.** Every admitted instance, seeds 1, 2, 3, 10 s per solve (uc-chped:
10 s per horizon via `--time-limit 10`), one thread, both arms. The worst case,
with everything admitted, is (50 + 24 solves) × 10 s × 2 arms × 3 seeds ≈ 74 min.
Pairs run serially under an exclusive machine lock, with the arm order
alternating per pair. Each pair waits for a 1-minute load average below 1.5 and
records the load at start.

**Supplementary: uc-chped's linear phase.** `solve()` and `fj_nl_initialize` both
set `GFJConfig::two_phase = false`. The masked linear phase the issue names is
therefore reachable only through `FeasibilityJump::run()` with the default
config, and no benchmark runner, the CLI and the Python bindings do not call it.
It is measured anyway, by an uncommitted harness. The harness builds the uc-chped
model, adds the objective row at bound `+inf` as `solve()` does, and runs
`FeasibilityJump::run()` with the default `GFJConfig` (`two_phase = true`,
`set_initial_x = true`) at `max_iterations = 200000`. It covers all 24
(file, horizon) pairs with seeds 1-5 in both arms. It is deterministic, so no
timing is involved. Metrics: D, `float_prepare_fast`, the status (feasible or
not), and the iterations used.

**Verdict rule.** Ping-pong is **real** if at least one instance meets all
three of these conditions:

1. **Engaged**: treatment `float_prepare_fast > 0` in every seed.
2. **Direct**: `D_treat ≥ 1` in at least 2 of 3 seeds, and `D_control = 0` in
   every seed. A nonzero control would mean the detector is catching something
   other than the closed form's asymmetry.
3. **Material**: either the median over seeds of `P_treat − P_control` is at least
   0.05, or the treatment is infeasible in at least 2 of 3 seeds where the
   control is feasible in at least 2 of 3.

In the supplementary, the treatment counts as real on a (file, horizon) if
`D_treat ≥ 1` in at least 3 of 5 seeds and either of these holds in at least 3 of
5 seeds:

- the treatment ends unsolved where the control reaches feasibility;
- the treatment uses at least twice the control's iterations.

The other outcomes are:

- (1) and (2) hold somewhere but (3) nowhere: ping-pong is **present but
  immaterial**, and the issue treats it as not real.
- (2) holds nowhere: **absent**.
- No instance is engaged: **uninformative** for that benchmark. The
  fixed-iteration auxiliary and the supplementary are then its answer.

### Addendum: MIPfeas extension (post hoc, requested by review)

Written and committed after the MINLPLib and uc-chped results above, and before
any MIPfeas run of either arm. The issue's claim that MIPfeas is unaffected
rests on a #106 throughput probe that had no reversal counter.
Pure-feasibility models with continuous columns in fractional rows are the
regime most exposed to this hazard.

**Roster.** The five instances #106 names: `cbs-cta`, `neos-662469`,
`neos-860300`, `swath3` and `uccase12`. Three more MIPfeas-roster instances were
added for having many fractional coefficients on continuous columns, as counted
by a scan of the MPS `COLUMNS` section: `binkar10_1`, `app1-1` and
`neos-3754480-nidda`. That scan found no fractional continuous coefficients on
`cbs-cta`, `neos-662469` or `neos-860300`. They stay in the roster because
#106 named them.

**Protocol.** The same two arms, counters and verdict rule (1)-(3) as above,
with no admission pilot: every instance runs. `cbls_mipfeas` runs at
`--budget 10 --threads 1` with seeds 1, 2 and 3. Pairs run serially under the
exclusive lock, with the same load gate. Feasibility is the runner's verdict.

**Also reported, outside the rule.** `rev_float_nobump`, the reversal count with
no tiny-score threshold, and `tiny`, the tiny accepted scores over all variable
types. The hazard comes from fractional coefficients, not from the Float type,
so Int columns in fractional rows can also produce it; D counts only Float
reversals.

## Results

Run on 2026-09-30, engine `c09a8ca`, on an AMD Ryzen 5 5600H (12 threads) under
Linux 7.0.0. The instrumentation patch, drivers and raw per-run records were
kept outside the tree and are not committed.

**Verdict: present but immaterial, so not real in the issue's sense.** The
closed form does produce the ping-pong the header predicts, and it has been
observed directly on MINLPLib `alkylation`. It never reaches the pre-registered
materiality bar, it costs no feasibility, and on `alkylation` the treatment's
objective is *better* than the control's in all three seeds. uc-chped cannot be
affected at all, because none of its rows is closed-form eligible.

### Zero trajectory change from the instrumentation

Every model was fingerprinted at the end of each `solve()`. The fingerprint is
an FNV-1a hash of every variable's value bits, the iteration count and the
objective, and the identical snippet was added to both the unpatched and the
patched tree. The runs used `--no-time-limit --max-iterations 20000 --seed 1`
on 49 MINLPLib instances and on uc-chped `ucp13` and `ucp40` at all five
horizons each. They used 300 iterations on `eg_all_s`, whose 20 000 did not
finish in 120 s. Unpatched main and the instrumented treatment agree on every
fingerprint, every result row, and every new-best trace row: 52 of 52 units.
Only the wall-clock-paced liveness ticks, the trace rows with `new_best = 0`,
differ in count.

The timed campaign ran on binaries built before the fingerprint snippet was
added. That snippet is the only difference between those binaries and the
checked ones.

### Pilot and admission

Only 4 of the 50 MINLPLib instances were admitted: `alkylation`,
`kall_ellipsoids_tc02b`, `minlphi` and `process`. On every other instance,
no accepted Float jump in the control pilot had a closed-form-eligible column.
Each such column has a nonlinear or otherwise ineligible row with a nonzero
weight, which forces the probe in both arms.

No uc-chped file was admitted: `accepted_float_cf = 0` on all 8. This is
structural. Every uc-chped row is posted as `add_constraint(sum(...))`, a bare
`Sum` read as `expr <= 0`, and not as a `Leq`/`Geq`/`Lt`/`Gt`/`Eq` node.
`comparison_of_affine_children` therefore classifies every row as ineligible,
and `LinearJumpScorer` never scores a uc-chped column in closed form, whether
the column is Float, Bool, masked or unmasked. The fixed-iteration check agrees:
treatment and control fingerprints are identical on `ucp13` and `ucp40` at all
horizons.

### Campaign (MINLPLib, 10 s, seeds 1-3, 12 pairs)

The pairs ran at 10:18-10:27. The 1-minute load at pair start was 0.79-1.45
(median 1.19). The arm order alternated per pair, but with an even roster it
stayed fixed per instance: `alkylation` and `minlphi` ran control first,
`kall_ellipsoids_tc02b` and `process` treatment first. All 24 runs were
feasible.

The table gives per-seed values for seeds 1, 2 and 3, treatment / control:

- **D**: tiny no-bump Float reversals.
- **P**: flat, no-bump FJ batches as a fraction of all FJ batches.
- **Share**: D divided by FJ iterations. Each reversal is paired with its
  forward jump, so ping-pong occupies about twice this fraction of the run.
- **Gap**: gap to BKS, in %.

| Instance | D (treat) | D (control) | P (treat) | P (control) | Share (treat) | Gap (treat) | Gap (control) |
|---|---|---|---|---|---|---|---|
| alkylation | 27991, 6776, 28586 | 0, 0, 0 | .012, .004, .014 | 0, 0, 0 | 2.4%, 0.6%, 2.7% | .0004, .0003, .047 | .051, .077, .079 |
| kall_ellipsoids_tc02b | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0 | 48.2, 37.2, 156.1 | 71.2, 67.5, 139.0 |
| minlphi | 368, 7, 288 | 0, 3, 1 | .005, 0, 0 | 0, 0, 0 | <0.1% | 0, 0, 0 | 0, 0, 0 |
| process | 3, 5, 197 | 3, 5, 0 | 0, 0, 0 | 0, 0, 0 | <0.1% | .014, .040, .018 | .014, .040, .044 |

The closed form engaged on all four instances, with `float_prepare_fast` between
0.33M and 0.82M per treatment run. The treatment also ran more FJ iterations in the
same 10 s in 11 of 12 pairs: from -7% to +25%, median +8%.

The instances fall under the rule as follows:

- **`alkylation`** meets engagement (1) and direct evidence (2): up to 28.6k tiny
  reversals per run, against 0 in the control. It fails materiality (3): the
  median ΔP is +0.012, against a bar of 0.05, and there is no feasibility loss.
  This is the one instance classified **present but immaterial**.
- **`minlphi`** and **`process`** fail (2), because the control also records
  reversals.
- **`kall_ellipsoids_tc02b`** records none.

Totals over the 12 runs of each arm:

- Flat no-bump batches: 75 of 21,696 in the treatment and 0 of 19,853 in the
  control.
- Tiny no-bump reversals: 64,221 in the treatment against 12 in the control.

The control's small counts cannot be rounding asymmetry, because the probe is
exactly antisymmetric. They were not traced further; the likely source is a
candidate set that changes between the two jumps.

### Supplementary: uc-chped under two-phase `FeasibilityJump::run()`

The pre-registered 200 000-iteration runs took too long on the 48- and
168-period instances. They completed 77 of the 120 (file, horizon, seed) pairs:
`ucp13`, `ucp40` and `ucp100` at every horizon, plus 2 seeds of `ucp100-48p`.
In every one of those pairs, both arms returned the same status and iteration
count, with `float_prepare_fast = 0` and D = 0. All 120 pairs were then re-run at
2 000 iterations, which is a deviation from the pre-registration; the result is
given here. In all 120 pairs, both arms gave an identical status, iteration
count and maximum violation, with `float_prepare_fast = 0` and D = 0. The zero follows from the
`Sum`-row structure described above: masking the nonlinear objective row does
not make the linear rows eligible, because none of them is a comparison node.
The two-phase path is therefore as unaffected as `solve()`.

## What this settles, and what it does not

- #178's question is answered for both benchmarks at `c09a8ca`. On MINLPLib the
  hazard is real in mechanism, but it does not move feasibility, the batch
  metric, or the objective. On uc-chped it is unreachable.
- None of the issue's three options is needed on this evidence. Option 1, the
  Bool/Int-only restriction, would give up the closed form on exactly the
  columns where it bought the treatment's extra iterations. Option 2, a relative score tolerance, changes
  the selection rule, and nothing measured here asks for it.
- The roster is small: four engaged instances, three seeds, 10 s. A model whose
  Float columns sit on large balanced linear plateaus, where MINLPLib has only
  `alkylation`, could still find the effect material. The counters above are the
  way to check such a model.
- A side finding outside #178's scope: every row `uc_model.h` posts is a `Sum`
  or a bare expression, and never a comparison node. By the eligibility rule, its
  Bool columns therefore never get the closed form either and are always scored
  by the DAG probe. This was read from the code, not measured.
