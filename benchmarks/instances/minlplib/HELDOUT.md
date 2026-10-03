# MINLPLib held-out roster (#144)

A second set of fifty MINLPLib instances that **no default was fitted to and no
result has been published against** (#145's transfer check below is the only
result recorded on it, and it publishes no table). It exists so the parameter transfer check
in #145 has unseen instances to run on. The shipped unproductive-batch exit
threshold was set by a grid over the published roster in `../bounds.csv`, and
the headline result is published against that same roster. Without a held-out
set, "the engine is good on non-convex MINLP" cannot be told apart from "the
engine was fitted to these fifty instances".

### Provenance

The seed, `HELDOUT_SEED = 144`, was fixed before any membership was computed and
has not changed since. The draw rule was revised twice after a membership had
been computed, both times on review, and both times before any measurement of
any kind existed on any version of the set:

1. Instances the published walk had already found to be binary NL are removed
   before the draw instead of being skipped during it. This changed one
   instance.
2. The per-class mix changed from an even round-robin over the three shared
   classes (17 / 17 / 16) to quotas proportional to the published roster's mix
   in those classes (21 / 20 / 9; see Method). This replaced 7 of the 50
   instances.

No solver had been run on any version of this set when either revision was
made. At no point did a result
exist that could have informed either change. When this was written, none of
the fifty names appeared anywhere else in the repository.

## Files

| File | What it is |
|---|---|
| `heldout/bounds.csv` | The held-out roster, in draw order, in `../bounds.csv`'s schema. It is the roster of record for this set. |
| `heldout/*.nl` | The fifty text-NL files, committed (1.2 MB, about the size of the published roster's 1.2 MB). |
| `heldout/pool.csv` | Snapshot of the catalogue rows the filter admits (all 397), limited to the columns the filter reads. Both rosters are re-derivable from it offline. |
| `heldout/unfetchable.csv` | The instances each fetch walk skipped because MINLPLib serves them as binary NL. A walk can only be replayed offline with this list. |

The `.nl` files are committed, not fetched on demand. #145 must run on exactly
these bytes. MINLPLib revises instances, and a check that silently ran on a
re-served file would not be the check that was registered. This is the same
choice the published roster made.

## The candidate pool

The pool is every row of the MINLPLib catalogue that passes `download.py`'s
filter: non-convex, an accepted problem type, text NL offered, only supported
operators, `nvars <= 150`, `ncons <= 150` and a finite primal bound.

- Catalogue: `https://www.minlplib.org/instancedata.csv`, fetched 2026-10-01,
  1633 rows, sha256 `0ec2cb1e766f6ee04b5d7e1aa8deee91c5eaab5b2eeb9c7fbaa45bc28dcc8283`.
- **Pool size: 397.** Of these, 347 lie outside the published roster. 8 of
  those 347 are known to be served only as binary NL (the published walk's
  skips), which leaves **339 drawable**. Those 339 were not all fetched to
  check them; the held-out walk fetched only the instances it took.
- This catalogue still rebuilds the published `bounds.csv` byte for byte (pinned
  by the test below). The 8 instances that roster's walk skipped (`ex8_1_2` and
  seven `kriging_peaks-red*`) are all still served as binary NL. Each was
  re-fetched to check: if any were text NL now, a rebuild would admit it and
  change the roster.

### Precondition check: is the pool big enough for a comparable second set?

**In total, yes; class for class, no.** That finding drives the rest of this
design.

| Structure class | Pool | Published | Drawable after the published roster |
|---|---:|---:|---:|
| bilinear | 5 | 5 | **0** |
| polynomial | 10 | 10 | **0** |
| mixed-integer | 53 | 15 | 38 |
| other | 263 | 14 | 249 |
| transcendental | 66 | 6 | 52 (8 binary-NL excluded) |
| **total** | **397** | **50** | **339** |

The published roster used up the whole bilinear and polynomial classes. While
that roster stays fixed, no held-out set can contain any instance of either
class. So 15 of the published roster's 50 slots (30%) have no held-out
counterpart, whatever split is used.

The pool is also unevenly covered by size. The published roster is the
**smallest** end of each large class. Size here means `nvars + ncons`:

| Class | Pool median | Published (min / median / max) | Drawable remainder (min / median / max) |
|---|---:|---|---|
| mixed-integer | 79 | 4 / 11 / 36 | 36 / 101 / 200 |
| other | 26 | 1 / 3 / 4 | 4 / 29 / 281 |
| transcendental | 10 | 1 / 3.5 / 4 | 4 / 13.5 / 228 |
| whole set | 25 | 1 / 6.5 / 252 | 4 / 30 / 281 |

In "other" and "transcendental", the published instances are the 14 and 6 smallest
drawable instances in the pool, none larger than 4. Every drawable instance
left in those classes is at least size 4, so it is at least as large as every
published one in its class. **Because of this, no set
disjoint from the published roster can match its size profile.** The only
choice is which way to differ from it (next section).

## Method

1. Start from the pool. Remove the published roster and the instances its walk
   already found unfetchable.
2. Within each structure class, order instances by
   `sha256("cbls-minlplib-heldout:144:<name>")`. This is a seeded shuffle that
   ignores size completely. A hash is used rather than `random.shuffle`
   because Python only guarantees `random()` and seeding to stay stable across
   versions, not `shuffle`. sha256 of a fixed string is stable everywhere.
3. Give each class still present in the remainder a quota proportional to the
   published roster's count in that class. The published counts in the three
   shared classes are 15 mixed-integer, 14 other and 6 transcendental (35 in
   all). Scaled to 50 they are 150/7 ≈ 21.43, 20 and 60/7 ≈ 8.57. Largest remainder,
   computed in exact fractions with ties broken by class name, gives
   **21 / 20 / 9**.
4. Fill the quotas from each class's hash order, interleaving the classes in
   sorted class order. An instance served as binary NL is passed over without
   costing its class a seat, and is recorded in `unfetchable.csv`. A network
   failure or an HTML error page aborts the run instead of being recorded as a
   skip. Otherwise a transient outage would change the membership.

The quotas keep the held-out set's class mix as close to the published
roster's as the remaining classes allow. An even split over three classes would
have given transcendental instances 16 seats against the published roster's 6
of 35.

All of this is `download.py`'s `select_heldout`, `heldout_quotas`, `quota_walk`
and `build_heldout`. The seed is a module constant with no command-line flag,
because changing it re-draws the set.

### Composition of the two halves

| Class | Published (`../bounds.csv`) | Held-out (`heldout/bounds.csv`) |
|---|---:|---:|
| bilinear | 5 | 0 (class exhausted) |
| polynomial | 10 | 0 (class exhausted) |
| mixed-integer | 15 | 21 |
| other | 14 | 20 |
| transcendental | 6 | 9 |
| **total** | **50** | **50** |

| Size (`nvars + ncons`) | Published (min / median / max) | Held-out (min / median / max) |
|---|---|---|
| mixed-integer | 4 / 11 / 36 | 36 / 96 / 196 |
| other | 1 / 3 / 4 | 6 / 33.5 / 220 |
| transcendental | 1 / 3.5 / 4 | 4 / 12 / 130 |
| whole set | 1 / 6.5 / 252 | 4 / 60.5 / 220 |

In both rosters the instances with integer variables are exactly the
mixed-integer class (15 and 21). The published roster has one maximisation
instance (`alkylation`); the held-out set has two (`pointpack06`,
`pointpack10`).

## Why not the "next fifty"

The obvious construction is to keep walking the published order and take the
next fifty survivors. It was **rejected**. Within each class, that order is
smallest first, so the next fifty are by construction the next-largest
instances in every class. Measured on this pool, they form a narrow band just
above the published roster: "other" sizes 4-6, "transcendental" 4-9,
mixed-integer 36-96.

This is an explicit trade-off against #144's own objection, so it is stated as
one. With the published roster fixed, **every** set disjoint from it is larger
than it (see the precondition check), and the seeded draw is further from it in
size than the next fifty would be: whole-set median 60.5, against 7.0 for the
next fifty and 6.5 for the published roster. Taken at face value, the
"systematically harder" objection applies to the chosen set even more strongly.

It does not decide the matter, because #145 is not a comparison between this
set and the published roster. It is an A/B of arms *within* the held-out set:
the shipped threshold against its neighbours, on the same instances, seeds and
budget. That comparison needs no size-matching to the tuning roster. What it
does need, to tell a size effect from a failure to transfer, is variation in
size inside the set:

- **The next fifty has no size spread to diagnose with.** The band is so
  narrow that a size effect would show up as a shift of the whole set, and
  that shift looks exactly like non-transfer. Nothing inside the set could tell
  them apart.
- **The seeded draw spans the remainder's size range.** Sizes run from 4 to 220
  overall (spread 216, against 92 for the next fifty), and within each class
  (mixed-integer 36-196, other 6-220, transcendental 4-130). #145 can therefore
  report the arm effect per size band, and that is the only way to separate a
  size effect from non-transfer. If the shipped value holds in the small band
  and loses in the large one, the finding is a size dependence: the value was
  fitted to the small end of the pool. If it loses across bands, that is
  non-transfer.

**Pre-registered size bands for #145.** These are fixed now, at the held-out
set's size tertiles, before anything is run:

| Band | `nvars + ncons` | Instances | mixed-integer / other / transcendental |
|---|---|---:|---|
| small | <= 32 | 16 | 0 / 9 / 7 |
| medium | 33-99 | 18 | 11 / 7 / 0 |
| large | >= 100 | 16 | 10 / 4 / 2 |

#145 reports the arm effect per band **and** per class, not only pooled. Size
and class are confounded, as the table shows: the small band has no
mixed-integer instance and the medium band no transcendental one. So no single
band-or-class reading isolates size, and a trend in one view should be checked
against the other. With 0 to 11 instances per class-band cell (two cells empty), a null trend
makes a size explanation less likely but does not rule it out.

So #144's acceptance criterion, "both halves have comparable size and
structure-class composition", is **not met and cannot be met** while the
published roster stays fixed. This document records that as the precondition
finding, rather than presenting the draw as comparable. The class mix is as
close as the remaining classes allow; the size profile is deliberately not
matched.

`test_heldout_is_not_the_size_ordered_next_fifty` pins that the committed set is
not that construction.

### Why not a randomised split of the whole pool

#144's first suggestion was to shuffle the whole pool and split it into halves.
That would produce two halves with matched class and size profiles, but it
would **change the published roster**. Every published table (comparison,
anytime trace, SCIP baseline, the ablation campaign) would then describe a
roster that no longer exists. The acceptance criterion that those tables stay
valid rules this out. The draw above is the closest thing to it that keeps the
published roster fixed.

## What this licenses #145 to claim

- **Valid: comparing arms within the held-out set.** For example, the shipped
  threshold against its neighbours on the same instances, seeds and budget.
  This is #145's question as written: does the shipped value still win, or is
  the difference inside the noise floor?
- **Not valid: comparing absolute numbers across the two sets** (for example,
  the published roster's mean gap against the held-out set's). The two sets
  differ in class mix (no bilinear or polynomial in the held-out set) and in
  size (median 6.5 against 60.5). A difference in absolute numbers is not
  evidence of over-fitting. If any cross-set comparison is wanted, use the
  published roster's 35 instances in the three shared classes, and still treat
  size as a covariate.
- **Families cluster.** The held-out set holds 12 `graphpart_*` instances (12
  of its 21 mixed-integer ones) and 7 `ex8_*` instances (5 transcendental, 2
  other). That follows from the pool, where graphpart is 22 of the 38 drawable
  mixed-integer instances. But instances in one family behave alike, so the
  effective sample is smaller than fifty. A count-of-wins statement should be
  reported per class, and a family should be treated as one cluster rather than
  as a dozen independent votes.
- **`nvars + ncons` misses expression size.** `eg_disc2_s` counts as size 36
  but its `.nl` is 925 KB, 78% of the held-out bytes. Treat it as an outlier
  in any size-covariate analysis.
- **Pre-registered handling of unloadable instances.** These instances were
  admitted on catalogue metadata alone and have never been through the NL
  reader. An instance the runner reports as `skipped(unsupported)` **stays in
  the roster and is reported that way**. It is not replaced, because replacing
  it after seeing a result is exactly the revisiting the commitment forbids.
  The same applies to an instance no arm solves.

A side finding about the headline result: the published roster samples the
small end of the pool (median size 6.5 against the pool's 25). Its claims are
claims about that end.

## Running on it

The runner and both drivers take the instance directory as a parameter, and
they read the roster from `<dir>/bounds.csv`. Pointing them at `heldout/` is all
the targeting needed:

    build/cbls_minlplib benchmarks/instances/minlplib/heldout \
        --instance <name> --seed <s> --time-limit 60 \
        --unproductive-iters <N> --commit <sha> --out <scratch>/<name>-<N>-<s>.csv

    .venv/bin/python3 benchmarks/minlplib/run_ablation.py \
        --inst-dir benchmarks/instances/minlplib/heldout ...

Output must go to a scratch path outside `benchmarks/instances/`, because the
drivers treat anything under it as published: `run_ablation.py` refuses an
`--out-dir` there, and `run_benchmark.py --inst-dir .../heldout` refuses to run
without scratch `--out`, `--trace-out` (or `--no-trace`) and `--staging-dir`.
Always pass `--out`: since #205 the runner refuses to write
`heldout/comparison.csv` (or any `<inst-dir>/comparison.csv`) at all, default
flags and `--commit` included, so a run without `--out` exits 2. A row from a
direct runner call carries only the runner's own DAG check; to check it
independently, pass `--solution-dir <dir>` and run
`.venv/bin/python3 -m benchmarks.minlplib.independent_check <out.csv> --inst-dir
benchmarks/instances/minlplib/heldout --solution-dir <dir>` (both drivers do
this themselves). `heldout/` has no
`comparison.csv`, `scip_baseline.csv` or `analysis_notes.csv`, and the runner
does not need them. #145's grid is `run_ablation.py --campaign transfer-145`;
see the next section.

## #145: does the shipped unproductive-exit threshold transfer?

### Result (smoke scale, 10 s)

**Measured at 10 s only, this leaves #145's question at the published 60 s
budget unanswered.** At 10 s the answer splits by neighbour. **300 is not
beaten.** Against 1000 the shipped value transfers: 1000 is worse. Against 100
it is not resolvable at 10 s. That is closest to #145's third outcome but not
its strict form: 100 moved instances outside their own floor, so the difference
is not inside the noise floor either.
100 shows no consistent direction, and the data shows only that 100 is not
detectably better. The original grid's only reason to reject 100 (lost
feasibility on `kall_ellipsoids_tc02b`) has no counterpart here: that instance
is not in this set, and the three `kall_*` instances that are all held. No
better value is
identified, and the shipped default is unchanged.

**Run.** Engine commit `1de6a7a`, `--time-limit 10`, seeds 1, 2 and 3, all 50
held-out instances x {300 (control), 100, 1000}. That is 450 solves, all of
which completed a search (mean wall 10.0 s per run, max 10.38 s). The run was
serial, with arms interleaved per instance. It ran under the machine-wide
wall-clock lock, with a one-minute load of at most 0.4 at start (the driver
refuses to start above that). Machine: `simon-Legion-5-Pro-16ACH6H`, AMD Ryzen
5 5600H (6 cores, 12 threads). The numbers live in scratch, not in this repository, as every
held-out result does. To regenerate them, run the command under "Protocol:
smoke scale" below from `1de6a7a`, then rescore with `--report-only --no-build`
on the same `--campaign` and `--out-dir`. That rescore reproduces the report
quoted here exactly.

**Noise floor, measured on the held-out roster.** It is each instance's own
control spread across the three seeds, as a two-sided 95% Student band.
Median control s = 0.59 gap points; typical per-instance floor +/-2.07 points;
31 instances measured. Of those, 5 were raised to the runner's tie band. The
other comparable instances (17 for the 100 arm, 16 for 1000) have no measurable
floor and are held to the tie band instead. In `tloss`, and for the 100 arm
also in `tln6`, the control was feasible on only one seed. (For the 1000 arm
`tln6` is control-only-feasible, so it is not comparable.) The rest returned
the same control gap on every seed, and most of those are solved exactly: 7 of the 12
`graphpart_*`, plus `st_bpv1`, `st_z`, both `ex14_*` and `ex2_1_2`. With three
seeds the Student factor is 4.30, so this floor is wide. A small real effect
can hide inside it.

| Neighbour | Worse | Better | Held | Unscored, beyond tie band | Feasibility (control-only / arm-only) | Feasible runs / BKS-matching runs (control: 141 / 72) | Sign test |
|---|---:|---:|---:|---|---|---|---:|
| 100 | 2 | 1 | 28 of 31 | `tloss`, `tln6` (mixed direction) | 1 / 0 (`ex8_5_6`) | 141 / 75 | p = 1.000 |
| 1000 | 9 | 0 | 22 of 31 | `ex14_2_1`, `ex14_2_6`, `tloss` (all worse) | 2 / 0 (`ex8_5_6`, `tln6`) | 139 / 61 | p = 0.004 (families 8/0, p = 0.0078) |

**1000 against 300: worse.** The nine scored movers are `kriging_peaks-full030`,
`hybriddynamic_fixedcc`, `ex8_5_4`, `wastewater02m1`, `st_fp7b`,
`kall_circles_c6a`, `hydro`, `tltr` and `tln4`. The median delta over them is
+14.47 points, against a median own floor of +/-1.49. The direction is
consistent but the magnitude is not. Three of the nine are moves of under 0.4
points over tiny floors: `hybriddynamic_fixedcc` +0.010, `kall_circles_c6a`
+0.0009 and `hydro` +0.38.

- **Families.** Treat each family as one vote (see "Families cluster" above).
  The nine still come from eight distinct families, with only `tln4` and `tltr`
  sharing one. All eight point the same way. A family-level sign test gives
  8 worse / 0 better, p = 0.0078 (0.016 after Bonferroni over the two
  neighbours). This is the count to quote under this document's clustering
  rule; the instance-level p = 0.004 overstates the sample. The three unscored
  movers add `ex14_*` worse and another `tl*` worse. No `graphpart_*` instance moved
  outside its floor, so the result is not driven by the family that dominates
  the mixed-integer class.
- **Size and class.** Worse in every pre-registered band (small 2, medium 4,
  large 3) and every class (mixed-integer 2, other 5, transcendental 2), with
  no "better" anywhere. That is consistent with no size dependence. With 2-4
  movers per band it cannot rule one out.
- **Multiplicity.** Bonferroni over the two neighbours tested against one
  control leaves the instance-level p at 0.008 and the family-level p at
  0.016.
- **Echo of the original grid.** The original grid recorded that 1000 stopped
  solving `st_e40`, which is not in this set. Here there is a similar
  regression. 1000 loses BKS matches on instances the control solves on every
  seed: `hybriddynamic_fixedcc` (3/3 -> 1/3), and `kall_circles_c6a`,
  `ex14_2_1` and `ex14_2_6` (each 3/3 -> 2/3). Several of these are at
  sub-0.1-point gaps. It also has one seed on `ex14_2_1` at objective 2.5e-5
  against a BKS of 3.8e-11 and a dual bound of 0: that prints as a 6.6e7% gap
  only because the BKS is numerically zero (the main README's `|BKS| >= 1e-4`
  caveat), so read its direction, not its size. And it
  loses `tln6`'s only feasible seed. Over the roster, 1000 matches BKS on 61
  runs against the control's 72.

**100 against 300: not resolvable at 10 s.** It is **not** inside the noise
floor. Three scored instances moved outside their own floor, two unscored
instances moved beyond the tie band, and two instances changed feasibility.
For that reason the report's READING declined to state the third outcome. What it shows
is no consistent direction: 2 worse, 1 better (and that 1 is float residue),
sign test p = 1.0. The three scored movers:

- `ex9_2_3` is better by 1.2e-6, just beyond the 1e-6 tie band. That is float
  residue.
- `hybriddynamic_fixedcc` is worse by 6e-4, from one seed at 0.0018.
- `wastewater02m1` is worse by +94 points, from two of three seeds at 118 and
  171 against a control of about 2-4. It is the one sizeable move. It is a
  single instance, and the 1000 arm moved it the same way.

Feasibility nets out to zero. `ex8_5_6` lost the control's single feasible
seed, `tloss` gained a seed, and `tln6` was feasible on one seed in each arm,
on different seeds. Per band: small 0 worse / 1 better, medium 1 / 0, large
1 / 0. Per class: only "other" has movers (2 worse / 1 better). No family or
class signal. Some instances favour 100 on BKS counts, notably `tln4` (3/3
against 1/3). Its delta (-1.61) sits inside its floor (+/-4.89), so it is not
read as a win. The original grid's 100-side finding (lost feasibility on
`kall_ellipsoids_tc02b`) is **not echoed**. That instance is not in this set,
and the three `kall_*` instances here all held. The engagement counters show
that the arm did change the search. LNS repairs went from 1157 attempted / 282
accepted under the control to 2131 / 383 under 100, and to 464 / 121 under
1000. So "held" here means the outcome did not move, not that the exit never
fired.

**What this does not establish.**

- Nothing about 60 s. At 10 s the exit gets fewer chances to fire. A 100-vs-300
  difference that builds up over a longer search could not show here, and
  1000's deficit could shrink or grow.
- That 300 is better than 100, or that 300 is an optimum. The grid has three
  points, and the lower neighbour is not resolvable against it.
- Anything off MINLPLib. The held-out set is unseen MINLPLib instances, not a
  different library. The scale-invariance objection recorded on
  `GFJConfig::unproductive_iterations` is untouched.

The report's own READING line is "NOT STATED MECHANICALLY (budget 10s)",
because neighbours moved instances. The outcome above is the judgement that
section hands to this write-up. It is not a rule's output. The default stays
300. A change to it would need its own justification and regression evidence,
and nothing here suggests one.

### What is run

The grid the shipped value came from, exactly: `--unproductive-iters` 100, 300
and 1000 (the provenance comment on `GFJConfig::unproductive_iterations`). 300
is the shipped default and is the **control**. The other two are scored against
it. The driver is `run_ablation.py` with `--campaign transfer-145`. It is the
same experiment shape as the #143 ablation (arms against a control from the same
sitting, interleaved per instance, three seeds, scored against a measured noise
floor), so it reuses that driver's locks, resume rule and refusals rather than
adding a second driver. The stamp records the campaign and each arm's flags, so
an ablation out-dir and a transfer out-dir cannot resume into each other.

Every row records the engine commit, seed, budget (`time_limit`) and the arm's
value. The value appears twice: in `arm_flags`, which the control also carries
as an explicit `--unproductive-iters 300` so a crashed run's row still has it,
and in the runner's own `search_config` cell. The driver stops the campaign if
`search_config` disagrees with the arm it ran.

### Protocol: smoke scale

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --campaign transfer-145 \
        --out-dir "$HOME/.cache/cbls-145/smoke" --time-limit 10

`--inst-dir` defaults to `heldout/` for this campaign, and the published roster
is refused for it. Seeds default to 1, 2 and 3. That is 50 instances x 3 arms x 3
seeds = 450 solves, about 1.25 h of solving. The run is serial on an otherwise
idle machine, under the same machine-wide lock as every timed MINLPLib driver.
Add `--dry-run` to print the plan without solving. Add `--report-only` to
re-score the out-dir.

**This is a smoke-scale check, not the run #145 describes.** #145 asks for the
published 60 s budget. This runs at 10 s, by decision, at about a sixth of the
cost. What that limits:

- The answer is about **10 s**. The threshold is an iteration count. A shorter
  budget runs fewer batches, so it gives the exit fewer chances to fire and the
  arms fewer chances to diverge. A difference that only builds up over a 60 s
  search cannot show here. "Inside the noise at 10 s" does not mean "inside the
  noise at 60 s".
- The noise floor is measured at 10 s too. It is the held-out control's own
  across-seed spread from this sitting, and it applies only to this budget.
- The original grid ran at 2 s, contended, on one seed. 10 s on three seeds is
  a better measurement than that, but it is not the published configuration.
  It cannot be cited as evidence about the 60 s headline result.
- So **#145's own question, at the published 60 s budget, is not answered**
  by this run, and nothing tracks it: #145 was closed on this 10 s result by
  decision. Answering it needs the same command at `--time-limit 60`, about
  7.5 h, under its own issue.

### How the report states the outcome

The report closes with a #145 section. For each neighbour it lists the
instances that moved worse or better than their own floor, the instances that
held, and the feasibility buckets. Each of these is broken down by the
pre-registered size bands above and by structure class, with the movers named
so a family such as `graphpart_*` can be read as one cluster. The floor comes
from this run's control only. Nothing is carried over from the published
roster.

The report header names the budget and the engine commit. A results file that
mixes budgets is refused outright.

Only the third outcome is stated by rule, and only in its strict form. The
report prints "INSIDE THE MEASURED NOISE FLOOR" only when every one of these
holds:

- **The planned campaign is all there.** Every roster instance in
  `heldout/bounds.csv` has, for the control and both neighbours, a completed
  search at exactly each planned seed: the seeds the out-dir's stamp records,
  else `--seeds`. No cell may hold a crashed or unscored row (for example
  `solve-error`). There are no rows for instances outside the roster or for arms
  outside the grid, and the roster metadata was found. One exemption follows
  the pre-registered rule for unloadable instances above. If the runner could
  not load an instance under any arm at any seed, and every one of those rows
  carries the same `unsupported` or `not-found` note, that instance is listed
  as "REPORTED, NOT MEASURED" and does not block the reading. If it fails to
  load under only some arms, it still blocks.
- **One budget and one commit** across the rows.
- **No neighbour moved anything.** No instance is outside its own floor. No
  instance whose control spread could not be measured differs by more than the
  runner's own tie band. That is the smallest difference the runner itself
  calls a difference; any nonzero delta would count float residue on exactly
  solved instances. Where the control returned the same gap on every seed,
  every single arm seed is held to that band, not only the arm's mean.
- **No feasibility change on any instance**, judged per instance, so a loss on
  one instance and a gain on another do not cancel.

In every other case it prints "NOT STATED MECHANICALLY" with the reasons
listed, and the outcome is decided in the write-up above. Each arm's own
`VERDICT` line above the section looks at gaps only and ignores all of these
checks, so in this report it is labelled "gap only". Quote the READING, not the
VERDICT. Naming a winner
by rule would mean choosing a threshold across two neighbours tested against
one control, with no multiplicity correction, over clustered instances and a
three-seed floor. Such a rule would sometimes pick winners from noise.

## Reproducing and checking

    .venv/bin/pytest tests/python/test_minlplib_heldout.py -q

That test re-derives both rosters from `heldout/pool.csv` and
`heldout/unfetchable.csv`, with no network access, and compares them byte for
byte against the two committed `bounds.csv` files. That comparison is
circular by construction: the committed files were written by the same code. It
pins the rosters against later changes to that code, not the method itself. The
test also pins the quota rule's 21/20/9 against hard-coded values, checks that the two
sets are disjoint and that the draw depends on the seed, and checks that every
held-out row has its text-NL file. Fake-server tests cover the live draw's
refusals: a published skip that is text NL today, a network failure, and an
HTML error page each abort the run without writing the roster files. (A
outage mid-walk can leave `.nl` files already fetched.)

To re-run the draw itself (it refuses to replace a committed roster without
`--force`). It keeps the committed `.nl` files rather than re-fetching them, but
it does fetch the published walk's eight skips again to re-check them:

    .venv/bin/python3 benchmarks/instances/minlplib/download.py --heldout \
        --catalogue benchmarks/instances/minlplib/heldout/pool.csv --force

A future catalogue refresh is a **new pool**. The committed held-out membership
stays as it is, because re-drawing it from a newer catalogue would be the
revision the commitment exists to prevent.
