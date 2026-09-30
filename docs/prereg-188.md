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
