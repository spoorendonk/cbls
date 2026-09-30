# Bounded-drift incremental Sum: pre-registration and record (#188)

This file was written and committed before the parameter sweep below ran,
and before any roster A/B. Results were appended under their own headings.
Review round 1 then edited it in five ways, all visible in the git history
of `docs/prereg-188.md`, where the file lived until then:

- it re-cites the sweep's provenance;
- it relabels one instance;
- it adds the disclosures under **What was seen before pre-registration**;
- it records the identity hashes;
- it pre-registers the MINLPLib re-check at the end.

No pre-registered rule or result was changed.

## Deviations from #188's proposal

- **The re-sum bound.** The issue proposed bounding a fresh re-sum by
  gamma_{n-1} * sum(|t_i|). The bound used here is the sum of the re-sum's own
  rounding errors, which TwoSum computes exactly. It is never larger than the
  issue's form, and it is 0 when the re-sum was exact. The issue's form is
  never 0, so under it every re-summed row would count as drifting and would
  be gated.
- **No margin factor.** The bound is rigorous as computed: every addition,
  and the gate's own comparison, is rounded up. A factor above 1 would only
  make the gate fire more often, and a factor below 1 would be unsound.
- **Verdicts are exact relative to a re-sum, not to the real sum.** On a Sum
  that is drifting, the gate decides against the real sum of the stored terms,
  within the bound. A Sum that was just re-summed with a nonzero bound is
  decided on the re-sum's own value, as the pre-#177 engine decided every
  row. That is the baseline the landing criterion measures against.

## The parameter

The design has one parameter, `kIncSumPeriod` (`include/cbls/dag_ops.h`).
The inexact update that would be a Sum's `kIncSumPeriod`-th since its last
re-sum re-sums it instead, so a Sum carries at most `kIncSumPeriod - 1`
inexact updates. Exact updates do not count. Candidates: **16, 64, 256, and 0
(never)**.

The period is a safety cap on how far the bound can grow between
re-groundings. The sweep below prices that cap; it does not show the chosen
value is better than the alternatives.

## The sweep

- **Instances**: thor50dday, neos-873061 and neos-4763324-toguru, #177's
  held-out sweep set. None of them is on the A/B roster below.
- **Arms**: this design with `kIncSumPeriod` patched to each candidate. Each
  arm is its own Release build (`CBLS_SANITIZE` empty) driving one
  single-threaded `solve()`, set up as `benchmarks/mipfeas` does it
  (FloatIntensifyHook, LNS 0.3 every 3 kicks).
- **Runs**: 10 s, seeds 1 and 2, serial, under the machine-wide lock. Each
  instance/seed group runs all four arms back to back in a rotated order,
  after waiting for a 1-minute load below 1.5. The load is recorded per run.
- **Metric**: GLS iterations in the budget, plus the gate's counters (local
  minima, gated minima, gate re-sums, flips, batch-end re-sums).
- **Rule**: take the geometric mean of iterations over the six runs for each
  candidate, and find the highest. Among the candidates within 1% of it,
  choose the smallest nonzero period, because a tighter drift bound fires
  the gate less. Choose 0 only if it alone is within 1%.

## The roster A/B

- **Arms**: base is main at `bf4979f` (exact-only #177 plus #186). New is the
  #188 branch's final engine head, with the chosen period.
- **Roster**: the 44 instances #177 pre-registered:
  b1c1s1 binkar10_1 cbs-cta co-100 cost266-UUE cvs16r128-89 drayage-100-23
  eilA101-2 exp-1-500-5-5 fiball gmu-35-50 irish-electricity lectsched-5-obj
  mcsched mushroom-best n3div36 neos-1582420 neos-3083819-nubu
  neos-3555904-turama neos-4300652-rahue neos-4532248-waihi neos-5093327-huahum
  neos-5188808-nattai neos8 neos-860300 neos-957323 ns1116954 ns1830653
  nursesched-sprint02 peg-solitaire-a3 rail01 ran14x18-disj-8 rmatr200-p5
  rococoC10-001000 s100 savsched1 sing326 sp150x300d supportcase33 swath3
  traininstance6 uccase12 uccase9 wachplan
- **Subsets**, exactly #177's. An instance is *integral* when at least half of
  its commit re-sum work in #177's pilot (`634001b` plus instrumentation, 5 s,
  seed 42) was in exact-eligible Sums; otherwise it is *fractional*.
  - integral (18): co-100 cvs16r128-89 eilA101-2 fiball lectsched-5-obj n3div36
    neos-1582420 neos-3083819-nubu neos-4532248-waihi neos8 neos-860300
    nursesched-sprint02 peg-solitaire-a3 rococoC10-001000 sp150x300d
    supportcase33 traininstance6 wachplan
  - fractional (26): the rest, rail01 among them.
- **Runs**: `cbls_mipfeas --budget 20 --threads 1 --seed {1,2} --commit <sha>`,
  serial, one base/new pair per instance and seed under the machine-wide lock.
  Each pair starts only once the 1-minute load is below 1.5, and the load is
  recorded. The order inside a pair alternates from one pair to the next.
  Every record is commit-stamped, and each binary's sha256 is recorded.
- **MINLPLib** (no-regression check): all 50 `.nl` instances, 20 s, seed 42,
  both arms, under the same pairing rules. *Disclosed in review round 1:* this
  check had no quantified pass rule at the time; see the re-check at the end.
- **Reporting**, at the instance level, where each instance is the mean of its
  two seeds:
  - the geometric-mean iteration ratio new/base, with a 95% interval (normal
    approximation on the log ratios), overall and per subset;
  - a two-sided sign test on iteration wins;
  - quality scored per instance by #177's rule, with a two-sided sign test.
    Per seed, feasibility counts first, then the objective when both runs are
    feasible, else the best violation; the instance's score is the sign of
    the sum.
- **Landing rule**: it lands only if the fractional subset's iteration geomean
  has a 95% interval above 1 and the integral subset's interval reaches 1 (no
  significant loss), and neither subset's quality sign test is significantly
  worse (p < 0.05). Otherwise the negative result is reported as measured.
  Nothing is re-tuned on this roster.

## What was seen before pre-registration (disclosed in review round 1)

Before this file was committed, one exploratory run looked at the roster:

- **Setup**: 10 s, seed 1, 10 roster instances (swath3, rail01, cbs-cta,
  co-100, b1c1s1, s100, neos-957323, n3div36, rmatr200-p5, savsched1).
- **Arms**: exact-only (`7f8c5a2`) against the design's first two commits.

Iteration ratios ranged from 1.03x to 1.67x, except b1c1s1 at **0.105x**.
b1c1s1 had reached feasibility, and its run was dominated by inner-solver
walks, which then paid a checked re-sum (a TwoSum per term) on every
incremental Sum they touched.

In response, the design was changed before any pre-registered run: walks that
do not know the old values re-sum plainly and leave the Sum untracked. That
change was first committed as `a50af3c` and is `94f019b` on this branch.

The roster was therefore not untouched when the design was fixed. Nothing was
tuned on it, but one design choice was made after seeing it.

## Sweep result (appended after the sweep)

The sweep ran on 2026-09-30, at a load of 0.80-1.27 per run, on the
machine-wide lock of a 12-core machine.

- **Engine**: `94f019b`'s patch applied on `7f8c5a2` (#186's pre-final head),
  with only `kIncSumPeriod` patched per arm.
- **Why that base is acceptable**: #186's last review round changed only
  breakpoint code, which these instances do not use.
- **Provenance**: the commit it was first built from, on a branch since
  rebuilt onto `bf4979f`, is not retained.

Geometric mean of GLS iterations over the six runs:

| period | geomean | vs best |
|---|---|---|
| 16 | 105,591 | -1.95% |
| 64 | 106,975 | -0.67% |
| 256 | 107,140 | -0.51% |
| 0 (never) | 107,694 | best |

64 and 256 are within 1% of the best, and the rule takes the smallest
nonzero: **`kIncSumPeriod = 64`**, the value the code already carries. Its
measured cost against never re-summing is 0.67%.

The gate barely fires here:

- thor50dday: 10-17 gated minima out of 592-955;
- neos-873061: at most 1 of about 240;
- neos-4763324-toguru: none of about 44,000-48,000.

The batch-end re-grounding re-sums 15,700-157,000 Sums per run.

## Integral trajectories and the identity hashes

At a fixed iteration count, main `bf4979f` and this branch end on the same
final-assignment hashes. The same hashes were recorded on the heads
`7bb0201`, `bfe9826` and `301ac22`.

| instance | subset | iterations | final-assignment hash |
|---|---|---|---|
| neos-860300 | integral | 5,000 | `db36cea8e3f52e39` |
| rail01 | fractional | 20,630 | `055ddb905574bf51` |

neos-860300 carries the claim that integral trajectories are unchanged.

rail01 is a fractional-subset instance, and its match is the stronger
evidence. Its fractional objective took 4,561 inexact updates in those
iterations, and drift still flipped no decision.

## First roster A/B, and why it was re-run (appended before the re-run)

The pre-registered A/B first ran with `bf4979f` against `7bb0201`
(2026-09-30, 12:13-13:47, load 0.94-1.48 per pair, median 1.01).

**MIPfeas**, per instance:

| subset | n | iterations, geomean | 95% CI |
|---|---|---|---|
| all | 44 | 1.078x | [1.034, 1.124] |
| fractional | 26 | 1.116x | [1.044, 1.193] |
| integral | 18 | 1.026x | [1.004, 1.048] |

Quality was 8 better and 6 worse (p = 0.79). Feasible runs went from 31 to 32.
It met the landing rule.

**MINLPLib failed the no-regression check.** Both arms had 48 feasible, but
the objective was worse on 7 instances and better on none (sign p = 0.016).

At a fixed iteration count the two arms end on the same objective, so the loss
is speed, not trajectory. `7bb0201` was 4-11% slower per iteration on nvs14,
nvs02 and shiporig, and 19-23% slower on gear4.

gprof and elimination on throwaway builds located four causes:

- the incremental-Sum wrapper was no longer inlined per dirty node;
- every probe stashed by scanning its whole cone;
- models with no incremental Sum at all (gear4, nvs14, chain50) still built
  the rules on every walk;
- the new per-model state sat among `Model`'s hot members.

Commits `0a9fa86`, `9dc5047` and `bfe9826` fix these. No parameter changed
and no trajectory moved. Fixed-iteration wall times of `bfe9826` against
`bf4979f`:

| instance | `bfe9826` | `bf4979f` |
|---|---|---|
| gear4 | 8.02-8.31 s | 7.96-8.05 s |
| nvs14 | 11.41 s | 11.79-11.90 s |
| nvs02 | 11.36-11.38 s | 11.75-11.85 s |
| shiporig, eq6_1 | within noise | within noise |

The fixes were diagnosed on the losing MINLPLib instances and re-run on the
same 50 instances and seed. The re-check at the end of this file answers
that with a fresh seed.

The whole pre-registered A/B was therefore re-run, unchanged, with `bf4979f`
against `bfe9826`. That run decides the landing rule. Both runs are reported.

## The deciding A/B: `bf4979f` -> `bfe9826`

Run 2026-09-30, 14:23-16:00. Pairs started at a 1-minute load of 0.81-1.52
(median 1.02). A pair waits for a load below 1.5, and the second run of a pair
may start just above it. Binary sha256 prefixes:

| binary | `bf4979f` | `bfe9826` |
|---|---|---|
| `cbls_mipfeas` | `b793a538` | `9f435c21` |
| `cbls_minlplib` | `d4e71580` | `1c1ccaf7` |

MIPfeas, per instance (the mean of seeds 1 and 2):

| subset | n | iterations, geomean | 95% CI | up / down (sign p) | quality better / worse / tied (sign p) |
|---|---|---|---|---|---|
| all | 44 | 1.082x | [1.038, 1.127] | 31 / 12 (0.005) | 8 / 6 / 30 (0.79) |
| fractional | 26 | 1.124x | [1.053, 1.199] | 23 / 3 (9e-5) | 8 / 4 / 14 (0.39) |
| integral | 18 | 1.024x | [1.004, 1.044] | 8 / 9 (1.0) | 0 / 2 / 16 (0.50) |

Feasible runs went from 31 to 32, the extra one being neos-4300652-rahue at
seed 1.

Two outliers:

- **rahue (0.70x)**: the loss is where the time went, not slower commits.
  - Seed 2: the new arm reaches its first feasible solution at 9.5 s, against
    19.8 s for base, and then spends the rest of the budget in objective
    search. Its iterations are 305k against 613k, and its objective is
    18.34 against 19.46.
  - Seed 1: the new arm is feasible at 16.7 s, where base never is.
- **ns1116954 (0.945x; 0.949x in the first run)**: this repeats. Over 50,000
  iterations its batch-end re-grounding re-sums 39,413 Sums, about 0.8 per
  iteration, against 12,610 inexact updates. Its batches are short and touch
  long fractional rows. The gate never fires there, so the cost is the
  batch-end re-sums, which exact-only does not pay.

MINLPLib, 50 instances: 48 feasible in both arms. The objective was better on
1, worse on 2 and the same on 47 (sign p = 1).

**Landing rule met.** The fractional interval lies above 1. The integral
interval lies above 1 as well. Neither quality sign test is significantly
worse.

The review-round-1 engine head `301ac22` changes the gate's margin arithmetic,
the fast-math guard and comments only. At a fixed iteration count it ends on
the same final-assignment hashes as `bfe9826` on swath3 (20,189 iterations,
608 gated minima), cbs-cta (50,162 iterations, 43 gated) and sp150x300d
(50,721 iterations, 200 gated). The same holds against `bf4979f` on the two
identity instances above.

## MINLPLib re-check (pre-registered in review round 1, before any of its data)

The first MINLPLib check had no quantified pass rule. Its fixes were diagnosed
on the losing instances and re-measured on the same seed. This re-check
answers that with a fresh seed and a stated criterion. Both parts below must
pass. If either fails, the result is reported and nothing is iterated.

**Arms**: base is `bf4979f`; new is `301ac22`. Both are Release builds with
`CBLS_SANITIZE` empty.

**Part 1: fresh-seed A/B**

- **Runs**: all 50 `.nl` instances, `cbls_minlplib --time-limit 20 --seed 43`,
  serial, under the pairing rules of the roster A/B above.
- **Pass** requires both:
  1. new has at least as many feasible runs as base;
  2. the per-instance quality sign test is not significantly worse at 0.05,
     two-sided. Quality is scored as follows: feasibility first; then, when
     both are feasible, the objective with a relative tolerance of 1e-6,
     lower being better. A significant result counts against the check only
     when new has more "worse" than "better" instances.

**Part 2: fixed-iteration timing (no wall clock in the search)**

- **Runs**: all 50 instances,
  `cbls_minlplib --no-time-limit --max-iterations 10000 --seed 1`. Each arm
  runs 3 times per instance in the order base, new, new, base, base, new.
  All 6 runs of an instance go under one exclusive-lock hold, after the load
  falls below 1.5.
- **Measure**: wall time. Per instance, the ratio is the median new time over
  the median base time. The ratios are reported as a distribution (quartiles,
  the extremes, and the geometric mean), together with whether both arms end
  on the same objective.
- **Pass** requires both:
  1. the geometric mean ratio over all 50 is at most 1.02;
  2. no instance whose base median is at least 0.5 s has a ratio above 1.10.

  Shorter runs are reported but not held to the 1.10 cap, because process
  start and model build dominate them.

### MINLPLib re-check result (appended after the run): Part 1 passes, Part 2 FAILS

The re-check ran on 2026-09-30, 16:28-17:19, at a 1-minute load of 0.52-1.14
(median 1.00). `cbls_minlplib` sha256 prefixes: `d4e71580` for `bf4979f` and
`55e96f6e` for `301ac22`. Both builds are Release with `CBLS_SANITIZE` empty.

**Part 1, fresh seed 43: PASS.**

- Feasible: 48 against 48.
- Objective: new better on 2 (ex8_6_1, nvs05), worse on 3 (maxmin,
  nvs01, shiporig), the same on 45. Sign test p = 1.

**Part 2, fixed-iteration timing (10,000 iterations, 3 repeats per arm): FAIL.**

- All 50 instances end on the same objective in both arms.
- Geomean ratio new/base: 1.0078, which is within the 1.02 limit.
- Distribution: min 0.672, first quartile 0.996, median 1.000, third
  quartile 1.003, max 1.498.
- Two of the 18 instances with a base median of at least 0.5 s exceed the
  1.10 cap. That fails the rule.

| instance | base median | new median | ratio | verdict |
|---|---|---|---|---|
| eg_all_s | 86.40 s | 98.70 s | 1.142 | fails the 1.10 cap |
| ex8_4_5 | 0.809 s | 0.911 s | 1.126 | fails the 1.10 cap |
| maxmin | 1.211 s | 1.312 s | 1.083 | under the cap |
| nvs01 | 2.612 s | 2.812 s | 1.077 | under the cap |
| gear4 | 0.207 s | 0.311 s | 1.498 | below 0.5 s, not capped |
| ex14_2_4 | 0.311 s | 0.409 s | 1.318 | below 0.5 s, not capped |

What these numbers show:

- **eg_all_s is a real per-iteration regression of about 14%.** Its six runs
  are tight within each arm: 85.5-87.0 s against 98.3-99.3 s.
- **The sub-second ratios are mostly one quantization step.** Every short run
  in both arms lands on a grid of about 0.1 s: 0.207/0.310, 0.809/0.910,
  1.211/1.312, 2.612/2.812. The ex8_4_5, maxmin, nvs01, gear4 and ex14_2_4
  differences are each one or two steps of that grid. Where the grid comes
  from was not identified. The pre-registered rule does not exempt it, so the
  verdict stands as measured.

Per the pre-registration, nothing is iterated on this result. The MINLPLib
no-regression check is **not met** at `301ac22`.

### Round 2 (appended before its run)

Round 2 re-checks Part 2 after an overhead fix. It uses the same rule and
thresholds, all 50 instances, 10,000 iterations and 3 repeats, with
`bf4979f` against `eadd414`. Nothing about the rule changes.

Part 1 is not re-run. At a fixed iteration count `eadd414` ends on the same
final-assignment hash as `301ac22` on all 50 MINLPLib instances (5,000
iterations, seed 1) and on swath3, cbs-cta, sp150x300d, neos-860300 and
rail01. Part 1's fresh-seed result therefore carries over unchanged.

Each run's CPU time (`/usr/bin/time`, user seconds) is recorded next to its
wall time as context only. The verdict stays on wall time.
