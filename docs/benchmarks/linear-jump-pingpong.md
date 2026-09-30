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

Run on 2026-09-30 with engine `c09a8ca`, on an AMD Ryzen 5 5600H (12 threads)
under Linux 7.0.0. The instrumentation, the fingerprint and the two-phase harness
are committed beside this file as `linear-jump-pingpong.patch` (see
**Reproduce**). The drivers and raw per-run records are not committed.

### Verdict

**Present but immaterial, and so not real in the issue's sense, on MINLPLib and
MIPfeas. At `c09a8ca` it is unreachable on uc-chped.**

The mechanistic bound comes first. Tiny-score A→B→A reversals of one Float
variable, with no weight bump in between, took at most 2.7% of a run's FJ
iterations, which with the forward jumps is ≤ ~5% of the iterations spent going
back and forth. That was on `alkylation`; everywhere else it was ≤ 0.23%.
Against this, the treatment ran a median +8% more FJ iterations on MINLPLib and
2.6-29× more on MIPfeas. Those are iteration counts across trajectories that
diverge, not a controlled throughput A/B.

No arm lost feasibility on any instance, and no objective loss is measurable. P
rose from 0 to 0.012 on `alkylation`, under the pre-registered 0.05 bar, and
stayed at ≈ 0 everywhere else.

### Where the cycle can run, and for how long (from the code)

A tiny-score reversal can only be the winning jump where no other sampled
variable has a positive score, which is at a local minimum of the weighted
violation. There it is **absorbing**. After A→B→A the assignment is bit-for-bit
the one before. Every jump value, Newton targets included, is a pure function of
the assignment and the weights, and the weights move only on a bump. The
detector requires an exact return to A. All 75 flat no-bump MINLPLib batches had
U0 > 0: they were stalls at positive violation, not descent on the objective
row.

**What ends it depends on the entry point:**

- **Batch API (`solve()`, every benchmark runner here):** the cycle is bounded
  by `batch_iterations`, and by the unproductive-streak exit
  (`track_batch_progress`, armed through `watch_progress`) once the outer loop's
  stagnation count arms it. The perturbation kick follows.
- **`gls()` / `run()`:** there is no progress exit. `watch_progress` requires
  `batch_iter_limit > 0`, which only the batch API passes, so the loop ends only
  on feasibility, `max_iterations` or `time_limit`. A cycle entered there can
  consume the caller's whole cap:
  - the 2 000 iterations of the LNS repair, `fj_nl_initialize(model, vm, 2000,
    …)` in `src/lns.cpp`;
  - Python's `fj_nl_initialize`, 10 000 by default;
  - `FeasibilityJump::run()` with a default `GFJConfig`, where `max_iterations`
    and `time_limit` are both 0. That path is genuinely unbounded.

  None of these paths was measured for D here: the counters bucket them as
  `run_*` loops, and D is not split by path.

**The hazard is fractional coefficients, not the Float type.** An Int column in a
row with fractional coefficients goes through the same `p + r·Δ` arithmetic and
has the same asymmetry. D counts Float reversals only, so Int columns were not
measured.

### Zero trajectory change from the instrumentation

Every model was fingerprinted at the end of each `solve()`. The fingerprint is
an FNV-1a hash of every variable's value bits, the iteration count and the
objective, and the identical snippet was added to both the unpatched and the
patched tree. The runs used `--no-time-limit --max-iterations 20000 --seed 1`
on 49 MINLPLib instances and on uc-chped `ucp13` and `ucp40` at all five
horizons. They used 300 iterations on `eg_all_s`, whose 20 000 did not finish in
120 s. Unpatched main and the instrumented treatment agree on every fingerprint,
every result row and every new-best trace row: 52 of 52 units.

The same fixed-iteration run between treatment and control (the auxiliary)
matched on 48 of 50 MINLPLib instances and on both uc-chped files. It differed
on `kall_ellipsoids_tc02b` and `minlphi`.

### Pilot and admission

- **MINLPLib:** 4 of 50 instances admitted: `alkylation`,
  `kall_ellipsoids_tc02b`, `minlphi` and `process`.
- **uc-chped:** 0 of 8 files admitted.

The pre-registration's rationale, that a column never eligible scores
identically in both arms, is wrong in principle. `accepted_float_cf` counts the
eligibility of *accepted* jumps only, so a column can be closed-form scored
without ever winning a jump. The real support is on the treatment side: in the
fixed-iteration run, `float_prepare_fast = 0` on every non-admitted MINLPLib
instance and on both uc-chped files. The treatment never took the closed form
there, so both arms scored those columns by the same probe.

**uc-chped, at `c09a8ca`.** Every row `uc_model.h` posts is `add_constraint` of a
`Sum` or of a bare expression, read as `expr <= 0`. None is a
`Leq`/`Geq`/`Lt`/`Gt`/`Eq` node, and `comparison_of_affine_children` accepts
only comparison nodes. So no uc-chped row is closed-form eligible, whatever the
column type and whether masked or not. This holds only while the eligibility
rule stays as it is: a rule that accepted a bare affine body would expose
uc-chped's Float columns, and the counters would need re-running.

### Campaign: MINLPLib (10 s, seeds 1-3, 12 pairs)

The pairs ran at 10:18-10:27. The 1-minute load at pair start was 0.79-1.45
(median ≈ 1.2). All 24 runs were feasible.

The table gives per-seed values for seeds 1, 2 and 3, treatment / control:

- **D**: tiny no-bump Float reversals.
- **P**: flat, no-bump FJ batches as a fraction of all FJ batches.
- **Share**: D divided by FJ iterations.
- **Gap**: gap to BKS, in %.

| Instance | D (treat) | D (control) | P (treat) | P (control) | Share (treat) | Gap (treat) | Gap (control) |
|---|---|---|---|---|---|---|---|
| alkylation | 27991, 6776, 28586 | 0, 0, 0 | .012, .004, .014 | 0, 0, 0 | 2.4%, 0.6%, 2.7% | .0004, .0003, .047 | .051, .077, .079 |
| kall_ellipsoids_tc02b | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0, 0, 0 | 0 | 48.2, 37.2, 156.1 | 71.2, 67.5, 139.0 |
| minlphi | 368, 7, 288 | 0, 3, 1 | .005, 0, 0 | 0, 0, 0 | ≤ 0.08% | 0, 0, 0 | 0, 0, 0 |
| process | 3, 5, 197 | 3, 5, 0 | 0, 0, 0 | 0, 0, 0 | ≤ 0.01% | .014, .040, .018 | .014, .040, .044 |

The instances fall under the rule as follows:

- **`alkylation`** meets (1) and (2) and fails (3). The median ΔP is +0.012
  against a bar of 0.05, and there is no feasibility loss. This is **present but
  immaterial**.
  - Its treatment gap is better in all three seeds. That is descriptive only: a
    sign test gives p = 0.25, the arm order was fixed for this instance, and the
    treatment ran 12-25% more iterations.
- **`minlphi`** fails (2) on a technicality: its control is 0, 3 and 1. Its
  treatment total of 663 is still ~100× the control's 4.
- **`process`** fails (2), with near-equal control counts.
- **`kall_ellipsoids_tc02b`** records none.

Totals over the 12 runs per arm:

- Flat no-bump batches: 75 of 21,696 in the treatment and 0 of 19,853 in the
  control.
- Tiny no-bump reversals: 64,221 in the treatment against 12 in the control.

The control's small counts cannot be rounding asymmetry, because the probe is
exactly antisymmetric. They were not traced further; the likely source is a
candidate set that changes between the two jumps.

### Campaign: MIPfeas (post-hoc extension, 10 s, seeds 1-3, 24 pairs)

The pairs ran at 11:09-11:17, with the 1-minute load at pair start 1.00-1.45
(median ≈ 1.1). The arm order alternated by instance and seed parity. The table
sums D and `rev_float_nobump` over the three seeds.

| Instance | D treat (per seed) | D control (per seed) | `rev_float_nobump` treat / control | Status (both arms, all seeds) | FJ iterations treat/control |
|---|---|---|---|---|---|
| cbs-cta | 1, 2, 3 | 0, 0, 0 | 60 / 27 | no solution | 2.6-2.7× |
| neos-662469 | 0, 0, 0 | 0, 0, 0 | 0 / 0 | no solution | 1.0× (no continuous column) |
| neos-860300 | 0, 0, 0 | 0, 0, 0 | 0 / 0 | no solution | 1.0× |
| swath3 | 0, 0, 0 | 0, 0, 0 | 826 / 16 | no solution | 27.8-28.8× |
| uccase12 | 14, 10, 38 | 0, 0, 0 | 292 / 48 | no solution | 10.9-11.5× |
| binkar10_1 | 22, 11, 7 | 1, 0, 0 | 40,313 / 3,072 | feasible | 9.7-11.9× |
| app1-1 | 0, 0, 0 | 0, 0, 0 | 19,038 / 1,811 | feasible | 10.5-12.5× |
| neos-3754480-nidda | 1150, 1360, 1403 | 479, 112, 165 | 18,240 / 2,909 | feasible | 3.3-8.1× |

P was at most 0.0053 in any run of either arm (`binkar10_1` seed 1, control).
Over all 24 runs per arm there was exactly 1 flat no-bump batch in each.

- **Rule outcome:** `cbs-cta` and `uccase12` meet (1) and (2) and fail (3), so
  they are **present but immaterial**. `binkar10_1` and `neos-3754480-nidda` fail
  (2), because the control also has counts. No instance is **real**.
- **Largest rates:** D per FJ iteration is at most 0.03% on `uccase12`. On
  `neos-3754480-nidda` the treatment's rate (0.12-0.22%) is comparable to the
  control's (0.08-0.30%).
- **Feasibility:** unchanged on every instance and seed. The five #106
  instances find no solution in 10 s in either arm; the three added instances
  are feasible in both.
- **Objective:** on `binkar10_1` and `neos-3754480-nidda` the treatment's
  objective is better in all three seeds. As on `alkylation`, that is
  descriptive only, at 3-12× the control's iterations.

The threshold-free `rev_float_nobump` is 2.2-52× the control's on the engaged
instances. Since the treatment also runs 2.6-29× more iterations, it scales
roughly with the iteration count.

### Supplementary: uc-chped under two-phase `FeasibilityJump::run()`

In all 120 pairs at 2 000 iterations, both arms gave an identical status,
iteration count and maximum violation, with `float_prepare_fast = 0` and D = 0.
The 77 pairs that finished at 200 000 iterations were likewise identical. This
follows from the `Sum`-row structure above: masking the nonlinear objective row
cannot make a linear row eligible when that row is not a comparison node.

### Deviations from the pre-registration

1. **Two-phase supplementary at 2 000 iterations.** It was pre-registered at
   200 000, which finished only 77 of 120 pairs; all 120 were re-run at 2 000.
2. **Fixed arm order on MINLPLib.** The order alternated per pair, but with a
   4-instance roster it ended up fixed per instance. D and P are counted inside
   each process, so the order does not affect them; it can affect only the
   secondary objective comparison.
3. **Fingerprint added after the campaign.** The fingerprint snippet was added
   and the arms rebuilt after the MINLPLib campaign. That snippet, which runs
   once at the end of `solve()`, is the only difference from the campaign's
   binaries. The MIPfeas extension ran on the rebuilt binaries.
4. **`eg_all_s` at 300 iterations.** The identity check ran it at 300 fixed
   iterations instead of 20 000, which did not finish in 120 s.
5. **Liveness ticks excluded.** Trace rows with `new_best = 0` are paced by the
   wall clock and were excluded from the identity comparison.
6. **Auxiliary reported.** The pre-registered MINLPLib treatment-vs-control
   fixed-iteration auxiliary is reported above: identical on 48 of 50.
7. **Admission rationale corrected.** The rationale is replaced by the
   treatment-side evidence, as described under **Pilot and admission**.
8. **MIPfeas extension.** It is post hoc, requested by review, and its
   addendum was committed before it ran.

## What this settles, and what it does not

- #178 is answered at `c09a8ca` for MINLPLib, uc-chped and MIPfeas. The
  asymmetry does produce ping-pong. It costs at most ~5% of FJ iterations, on one
  instance, and nowhere costs feasibility or measurable objective.
- None of the issue's three options is needed on this evidence.
  - **Option 1**, restricting the closed form to Bool/Int columns, would give up
    the Float closed form. That is where MIPfeas's 2.6-29× iteration counts come
    from.
  - **Option 2**, a relative score tolerance, changes the selection rule.
  - **A guard proposed in review:** when the winning score is tiny and the column
    is closed-form, re-score that one candidate with `weighted_violation_delta`
    and treat a probe score ≤ 0 as non-improving. This would make the
    accept/reject decision for those moves exactly the probe's, which is the
    reference semantics, and it costs one probe per tiny winner (2.9-3.3% of
    accepted jumps here). It needs its own tiny threshold, and it does not
    reproduce the probe's choice *among* candidates in a near tie. It is not
    needed on this evidence.
- **What was not measured:**
  - the unbounded or cap-bounded `gls()`/`run()` paths, split out from the batch
    API;
  - Int columns in fractional rows;
  - budgets beyond 10 s;
  - rosters beyond the admitted ones.

  A model with large balanced linear Float plateaus, or a heavy user of
  `FeasibilityJump::run()`, is where the counters in the patch should be run
  first.
- A side finding outside #178's scope, read from the code and not measured:
  uc-chped's Bool columns never get the closed form either, for the same reason.

## Reproduce

**Patch.** `docs/benchmarks/linear-jump-pingpong.patch` applies to `c09a8ca`
with `git apply`. It adds four things, with no change to the search:

- the counters (`pp178::Stats`);
- the control switch `CBLS_PP_CONTROL` in `LinearJumpScorer::prepare`;
- the end-of-solve fingerprint;
- the harness `benchmarks/uc-chped/uc_twophase_178.cpp`.

For the unpatched arm, copy by hand only the fingerprint from the patch's
`src/search.cpp` hunk: the `hash178_dump` helper and its call.

**Builds.** Three build directories, all `Release` with `CBLS_SANITIZE` empty:

```
cmake -S <unpatched+fingerprint> -B build-pristine -DCMAKE_BUILD_TYPE=Release \
  -DCBLS_BUILD_TESTS=OFF -DCBLS_BUILD_EXAMPLES=OFF -DCBLS_INSTALL_GIT_HOOKS=OFF
cmake -S <patched> -B build-treat   <same flags>
cmake -S <patched> -B build-control <same flags> -DCMAKE_CXX_FLAGS=-DCBLS_PP_CONTROL
cmake --build build-<arm> -j4 --target cbls_minlplib cbls_uc_chped cbls_mipfeas
```

The harness is built outside CMake, against each arm's library:

```
c++ -O3 -DNDEBUG -std=gnu++17 -I<patched> -I<patched>/include \
  -I build-<arm>/_deps/json-src/include \
  <patched>/benchmarks/uc-chped/uc_twophase_178.cpp build-<arm>/libcbls.a -lz \
  -o build-<arm>/uc_twophase
```

**Environment.** `CBLS_PP_OUT=<file>` appends one JSON line of counters per
`solve()` call, and one at process exit if any counts remain.
`CBLS_HASH_OUT=<file>` appends the fingerprint and the iteration count per
`solve()`.

**Commands.**

- Fixed-iteration identity, per instance and arm:

  ```
  cbls_minlplib <minlplib dir> --no-time-limit --max-iterations 20000 --seed 1 \
    --instance I --out O.csv --trace T.csv
  ```

  and the same with `cbls_uc_chped <uc-chped dir> ... --instance ucp13`.
- Timed runs:

  ```
  cbls_minlplib <dir> --time-limit 10 --seed S --instance I --out O.csv
  cbls_mipfeas --instance I --inst-dir <mipfeas dir> --out-dir D --budget 10 \
    --seed S --threads 1 --commit X
  ```
- Harness:

  ```
  uc_twophase <file.jsonl> <horizon> <seed> <max_iterations>
  ```

**Protocol.** A pair is the control run and the treatment run of one
(instance, seed). Each pair holds an exclusive machine lock
(`flock -x ~/.cache/cbls-bench.lock`). It waits inside the lock until the
1-minute load average is below 1.5, polling every 10 s. It records that load and
then runs both arms back to back. The lock is released between pairs.

The arm order alternates between pairs. Scoring follows the pre-registered rule
above:

- D is `rev_float_nobump_tiny`;
- P is `api_flat_nobump / api_batches`;
- engagement is `float_prepare_fast`.

Counters are summed over the `solve()` lines of a run.
