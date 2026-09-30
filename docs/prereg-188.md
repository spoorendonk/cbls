# #188 pre-registration: bounded-drift incremental Sum

Written and committed before the parameter sweep below ran, and before any
roster A/B. Results are appended under their headings, never edited in.

## The parameter

One: `kIncSumPeriod` (`include/cbls/dag_ops.h`), the number of INEXACT updates
an incremental Sum may carry before a commit re-sums it instead. Exact updates
do not count. Candidates: **16, 64, 256, and 0 (never)**.

There is no margin factor. The drift bound is the sum of the rounding errors
TwoSum computes exactly, rounded up at every addition, so it is an upper bound
as it stands: a factor above 1 could only make the local-minimum gate fire more
often, and one below 1 would be unsound.

## The sweep

- **Instances**: thor50dday, neos-873061, neos-4763324-toguru -- #177's
  held-out K-sweep set, none of them on the A/B roster below.
- **Arms**: this design with `kIncSumPeriod` patched to each candidate, one
  Release build per candidate (`CBLS_SANITIZE` empty), each driving one
  single-threaded `solve()` set up as `benchmarks/mipfeas` does
  (FloatIntensifyHook, LNS 0.3 every 3 kicks).
- **Runs**: 10 s, seeds 1 and 2. Serial, under the machine-wide lock, waiting
  for a 1-minute load below 1.5 before each instance/seed group, which runs all
  four arms back to back in a rotated order. Load recorded per run.
- **Metric**: GLS iterations in the budget, plus the gate's counters (local
  minima, gated minima, gate re-sums, flips, batch-end re-sums).
- **Rule**: per candidate, the geometric mean of iterations over the six runs.
  Choose the candidate with the highest geomean; among candidates within 1% of
  it, choose the smallest nonzero period (a tighter drift bound fires the gate
  less), and 0 only if it alone is within 1%.

## The roster A/B

- **Arms**: base = main at `bf4979f` (exact-only #177 plus #186); new = the
  final head of the #188 branch, with the chosen period.
- **Roster**: the 44 instances #177 pre-registered:
  b1c1s1 binkar10_1 cbs-cta co-100 cost266-UUE cvs16r128-89 drayage-100-23
  eilA101-2 exp-1-500-5-5 fiball gmu-35-50 irish-electricity lectsched-5-obj
  mcsched mushroom-best n3div36 neos-1582420 neos-3083819-nubu
  neos-3555904-turama neos-4300652-rahue neos-4532248-waihi neos-5093327-huahum
  neos-5188808-nattai neos8 neos-860300 neos-957323 ns1116954 ns1830653
  nursesched-sprint02 peg-solitaire-a3 rail01 ran14x18-disj-8 rmatr200-p5
  rococoC10-001000 s100 savsched1 sing326 sp150x300d supportcase33 swath3
  traininstance6 uccase12 uccase9 wachplan
- **Subsets**, exactly #177's: an instance is *integral* when at least half of
  its commit re-sum work in #177's pilot (`634001b` + instrumentation, 5 s,
  seed 42) was in exact-eligible Sums, else *fractional*.
  - integral (18): co-100 cvs16r128-89 eilA101-2 fiball lectsched-5-obj n3div36
    neos-1582420 neos-3083819-nubu neos-4532248-waihi neos8 neos-860300
    nursesched-sprint02 peg-solitaire-a3 rococoC10-001000 sp150x300d
    supportcase33 traininstance6 wachplan
  - fractional (26): the rest.
- **Runs**: `cbls_mipfeas --budget 20 --threads 1 --seed {1,2} --commit <sha>`,
  serial, one base/new pair per instance and seed under the machine-wide lock,
  each pair started only once the 1-minute load is below 1.5 (recorded), the
  order inside a pair alternating from one pair to the next. Every record
  commit-stamped; each binary's sha256 recorded.
- **MINLPLib** (no-regression check): all 50 `.nl` instances, 20 s, seed 42,
  both arms, same pairing rules.
- **Reporting**, at the instance level (each instance: mean of its two seeds):
  geometric-mean iteration ratio new/base with a 95% interval (normal
  approximation on the log ratios), overall and per subset; a two-sided sign
  test on iteration wins; quality scored per instance by #177's rule (per seed:
  feasibility first, then objective when both feasible, else best violation;
  the instance's score is the sign of the sum) with a two-sided sign test.
- **Landing rule**: it lands only if, on the fractional subset, the iteration
  geomean's 95% interval lies above 1, and on the integral subset the interval
  reaches 1 (no significant loss) -- with neither subset's quality sign test
  significantly worse (p < 0.05). Otherwise the negative result is reported
  as measured. Nothing is re-tuned on this roster.

## Sweep result (appended after the sweep)

Run 2026-09-30, load 0.80-1.27 per run (machine-wide lock, 12 cores). Arms were
built from branch commit `a50af3c` with only `kIncSumPeriod` patched, on the
pre-rebase base `7f8c5a2` (#186 before its last review round, which touched only
breakpoint code these instances do not use). Geometric mean of GLS iterations
over the six runs:

| period | geomean | vs best |
|---|---|---|
| 16 | 105,591 | -1.95% |
| 64 | 106,975 | -0.67% |
| 256 | 107,140 | -0.51% |
| 0 (never) | 107,694 | best |

64 and 256 are within 1% of the best; the rule takes the smallest nonzero:
**`kIncSumPeriod = 64`**, the value the code already carries. The gate barely
fires here: 10-17 gated minima out of 592-955 on thor50dday, at most 1 of ~240
on neos-873061, none of ~44,000-48,000 on neos-4763324-toguru. The batch-end
re-grounding re-sums 15,700-157,000 Sums per run.

## Branch note (appended before the roster A/B)

The sweep above ran on branch `worktree-agent-abbee2e52063f74b1`, kept as a
record; its commits were then cherry-picked onto main `bf4979f` as branch
`feat/188-bounded-drift-sum`. Matched by `git patch-id --stable`:

| old branch | new branch | commit |
|---|---|---|
| `d9b2f34` | `5997147` | perf(dag): bounded-drift incremental Sum (conflicts resolved against #186's final `model.h` and `test_element_rounding.cpp`, so its patch-id differs) |
| `2c56754` | `2fcf994` | refactor(fj): gate counters |
| `3899dd6` | `a4b5eca` | test(dag): rounding of `new - old` |
| `a6503b1` | `090c931` | refactor(model): split the classifier |
| `a50af3c` | `94f019b` | perf(dag): plain re-sum on walks without old values (the sweep's arms) |
| `8f8ed78` | `721485a` | this file |
| `eb5f167` | `7bb0201` | the sweep result |

The roster A/B's new arm is `7bb0201`; later commits on the branch change only
docs. Before it runs, fixed-iteration runs of `bf4979f` and `7bb0201` end on
identical final-assignment hashes on neos-860300 (5,000 iterations) and rail01
(20,630 iterations, of which the objective row's Sum took 4,561 inexact
updates).

## First roster A/B, and why it is re-run (appended before the re-run)

The pre-registered A/B ran with `bf4979f` against `7bb0201` (2026-09-30,
12:13-13:47, load 0.94-1.48 per pair, median 1.01). MIPfeas, per instance:
all 44 1.078x [1.034, 1.124], fractional 26 1.116x [1.044, 1.193], integral
18 1.026x [1.004, 1.048]. Quality 8 better and 6 worse (p = 0.79). Feasible
runs went from 31 to 32. It met the landing rule.

MINLPLib failed the no-regression check. Feasible 48 against 48, but the
objective was worse on 7 instances and better on none (sign p = 0.016). At a
fixed iteration count the two arms end on the same objective, so the loss is
speed, not trajectory. `7bb0201` was 4-11% slower per iteration on
nvs14/nvs02/shiporig and 19-23% on gear4. The cause was located by gprof and
by elimination on throwaway builds:

- the incremental-Sum wrapper was no longer inlined per dirty node;
- every probe stashed by scanning its whole cone;
- models with no incremental Sum at all (gear4, nvs14, chain50) still built
  the rules on every walk;
- the new per-model state sat among `Model`'s hot members.

Commits `0a9fa86`, `9dc5047` and `bfe9826` fix these. No parameter changed
and no trajectory moved: fixed-iteration final hashes still match `bf4979f` on
neos-860300 and rail01, and the MINLPLib objectives match at fixed iterations.
The fixed-iteration wall times of `bfe9826` against `bf4979f`: gear4
8.02-8.31 s against 7.96-8.05 s, nvs14 11.41 against 11.79-11.90, nvs02 11.36-11.38
against 11.75-11.85, shiporig and eq6_1 within noise.

The whole pre-registered A/B is therefore re-run, unchanged, with `bf4979f`
against `bfe9826`. That run decides the landing rule. Both runs are reported.
