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
