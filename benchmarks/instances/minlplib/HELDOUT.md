# MINLPLib held-out roster (#144)

A second set of fifty MINLPLib instances that **no default was fitted to, except
presolve and compound propagation on the 25 of #331's sample, and no table has
been published against**. It was drawn so the parameter transfer check in #145
had unseen instances to run on. It is no longer unseen. Runs recorded on it, none publishing a table, include:

- #145's transfer check (below);
- #244's held-out smoke of the fitted defaults (10 s, seeds 1-2; every default
  kept);
- #228's laptop smoke beside the published roster (engine `0c0a945`, 60 s,
  seeds 1-2, `elec25`/`elec50` left out): a
  cbls/SCIP GM(gap+1) ratio of 1.81-1.84 held-out against 1.75-1.88 published,
  no overfit at that scale (posted on #228);
- #256's dev run, compound moves on vs off (60 s, seeds 1-3, posted on #256);
- the #265 `xk tune` study (`benchmarks/tune/minlplib.study.toml`, in git
  history at `047f2d4c`): it
  searched on 35 of the fifty and re-checked its elites on the other 15,
  confirmed no setting and changed no default;
- presolve smokes (#343, #347, #354);
- #331's presolve-322 campaign, run by #352 (engine `ef1bbad`, 60 s, seeds 1-3,
  four arms, 300 runs on the 25 of the sample below; posted on #331).

**Fitted in part since #331.** #331 decided the presolve and
compound-propagation defaults (both kept off) on its presolve-322 sample, 25 of
these fifty (below), so those 25 are in-sample for those two defaults. The other
25 are still un-fitted. No open issue
plans a held-out score beside the published roster's at full scale; #256 runs
the full library instead. The shipped unproductive-batch exit
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
the fifty names appeared anywhere else in the repository; they now also appear
in the #265 study's configuration and in #331's sample list.

## Files

| File | What it is |
|---|---|
| `heldout/bounds.csv` | The held-out roster, in draw order, in `../bounds.csv`'s schema. It is the roster of record for this set. |
| `heldout/*.nl` | The fifty text-NL files, committed (1.2 MB, about the size of the published roster's 1.2 MB). |
| `heldout/pool.csv` | Snapshot of the catalogue rows the filter admits (all 397), limited to the columns the filter reads. Both rosters are re-derivable from it offline. |
| `heldout/unfetchable.csv` | The instances each fetch walk skipped because MINLPLib serves them as binary NL. A walk can only be replayed offline with this list. |
| `heldout/presolve_322_sample.txt` | #331's presolve-322 sample: the 25 instances presolve reduces (see "#331: the presolve-322 sample"). |

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

Taking the next fifty survivors in the published order was **rejected**:
within each class that order is smallest first, so they form a narrow size band
just above the published roster ("other" 4-6, "transcendental" 4-9,
mixed-integer 36-96). #145 compares arms *within* the held-out set, which needs
no size-matching to the tuning roster but does need size spread inside the set,
to tell a size effect from a failure to transfer. The seeded draw has it: sizes
4-220 overall (mixed-integer 36-196, other 6-220, transcendental 4-130). The
price is that every set disjoint from the published roster is larger than it
(whole-set median 60.5, against 6.5), so #144's "comparable size and
structure-class composition" is **not met and cannot be met** while the
published roster stays fixed; the class mix is as close as the remaining
classes allow, and size is deliberately not matched. A randomised split of the
whole pool would match both but change the published roster, invalidating every
published table. The full rationale is in
[#144](https://github.com/spoorendonk/cbls/issues/144#issuecomment-6633092805).

**Pre-registered size bands for #145.** These are fixed now, at the held-out
set's size tertiles, before anything is run:

| Band | `nvars + ncons` | Instances | mixed-integer / other / transcendental |
|---|---|---:|---|
| small | <= 32 | 16 | 0 / 9 / 7 |
| medium | 33-99 | 18 | 11 / 7 / 0 |
| large | >= 100 | 16 | 10 / 4 / 2 |

#145 reports the arm effect per band **and** per class, not only pooled: size
and class are confounded (the small band has no mixed-integer instance, the
medium band no transcendental one), so a trend in one view should be checked
against the other.

`test_heldout_is_not_the_size_ordered_next_fifty` pins that the committed set is
not that construction.

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
  admitted on catalogue metadata alone and, when drawn, had never been through
  the NL reader. An instance the runner reports as `skipped(unsupported)` **stays in
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

**Outcome (10 s only; the published 60 s budget is unanswered).** The grid the
shipped value came from, `--unproductive-iters` {100, 300 (control), 1000}, ran
on all 50 held-out instances at seeds 1-3 and `--time-limit 10`, engine commit
`1de6a7a`. **300 is not beaten**, and the default stays 300:

- **1000 is worse**: 9 instances worse outside their own floor and 0 better
  (family-level sign test 8/0, p = 0.0078), in every size band and class.
- **100 is not resolvable at 10 s**: 2 worse, 1 better (float residue), and
  feasibility changes netting to zero. No consistent direction, but not inside
  the noise floor either, so the report's READING is "NOT STATED
  MECHANICALLY".

#145 was closed on this result by decision. Answering it at 60 s needs the same
command at `--time-limit 60` (about 7.5 h), under its own issue. The full
write-up (noise floor, per-neighbour table, movers, what it does not establish,
and how the report states an outcome) is in
[#145](https://github.com/spoorendonk/cbls/issues/145#issuecomment-6633092217).

The campaign, at smoke scale (450 solves, about 1.25 h, serial on an idle
machine under the machine-wide lock):

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --campaign transfer-145 \
        --out-dir "$HOME/.cache/cbls-145/smoke" --time-limit 10

`--inst-dir` defaults to `heldout/` for this campaign and the published roster
is refused for it. `--dry-run` prints the plan; `--report-only --no-build` on
the same `--campaign` and `--out-dir` re-scores it, and reproduces the report
exactly. The report prints "INSIDE THE MEASURED NOISE FLOOR" only when the
campaign is complete at one budget and one commit, no instance moved outside
its floor or the tie band, and no instance changed feasibility; otherwise
"NOT STATED MECHANICALLY" with its reasons. Quote the READING, not the per-arm
`VERDICT` (gap only).

## #331: the presolve-322 sample

#331 asks whether presolve and compound-move propagation should be on by
default. Its MINLPLib arms run with
`run_ablation.py --campaign presolve-322` (control, `--presolve`,
`--compound-propagation`, both) on a 25-instance dev sample of this roster,
`heldout/presolve_322_sample.txt`.

**Rule, fixed before any solve:** the held-out instances the A4-NL presolve
reduces, filled up to 25 with instances it leaves untouched if fewer reduce.
The census reads and presolves each model without solving it:

    build/cbls_minlplib_nl_check --presolve benchmarks/instances/minlplib/heldout/*.nl

At commit 408a4bf exactly 25 of the 50 print `presolve=reduced`, so the
sample is those 25 and no fill was needed. A presolve change that alters the
census does not alter the sample: it is a committed list, and the campaign's
stamp records its hash. The sample is a dev set, chosen because presolve acts
on it, so it says nothing about instances presolve leaves untouched (there the
`presolve` arm is the control up to the presolve pass's own cost).

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --campaign presolve-322 \
        --out-dir <scratch> [--dry-run]

defaults to this sample and to `heldout/`, refuses the published roster, and
takes `--instances` for a subset (the laptop smoke). `--lock-command 'xk lock --'`
takes the machine lock per solve, as #331 asks, so the campaign's hours do not
hold it against other sessions' builds. Pair it with `--no-build`, since the
driver's own build runs outside that prefix: build at this commit first, under
the lock, and do not rebuild that `--build-dir` until the campaign ends. The
lock is free between solves and this driver has no binary check, so rows from a
rebuilt binary would land under the same commit. If the prefix does not start a
solve, the campaign stops with nothing recorded for it, and a resume goes on.
The load check still refuses a start on a busy machine rather than queueing on
the lock; `--allow-busy` passes it. Arms of one block stay adjacent in order, but
another session's locked job can run between two of them.

**How the report reads.** After the shared per-arm scoring, the report adds
#331's section (`benchmarks/minlplib/presolve_report.py`): each arm's
GM(gap+1) against the control's, per seed and pooled over seeds, with feasible
counts. A ratio is taken over the instances feasible, with a finite gap, in
both the arm and the control at that seed, so an instance only one of them
solves moves the feasible count, not the ratio; below 1 is the arm ahead. The
noise floor is the control's GM per seed over one instance set (feasible in
every arm at every seed), max over min; a pooled ratio inside
[1/floor, floor] is not a difference. The section opens with `INCOMPLETE`,
naming the missing (instance, arm, seed) runs and those recorded without a
completed search (a crash), when the rows fall short of the stamped seeds and
roster size: #331's decision is read off a complete run only.

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
