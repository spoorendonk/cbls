# MINLPLib held-out roster (#144)

A second set of fifty MINLPLib instances that **no default was fitted to and no
result has been published against**. It exists so the parameter transfer check
in #145 has unseen instances to run on. The shipped unproductive-batch exit
threshold was set by a grid over the published roster in `../bounds.csv`, and
the headline result is published against that same roster. Without a held-out
set, "the engine is good on non-convex MINLP" cannot be told apart from "the
engine was fitted to these fifty instances".

The membership and the seed were committed before any run on the set:
`HELDOUT_SEED = 144` landed in the commit "feat(minlplib): add a seeded held-out
draw to the selection tooling", and the membership in the commit after it.
That second commit also settled one rule of the draw: the instances the
published walk found unfetchable are removed before the round-robin rather than
skipped during the walk, so a binary-NL instance does not cost its class a turn.
Under the earlier rule the set differs by one instance (`graphpart_3g-0333-0333`
in place of `ex14_2_7`, giving 18/17/15 instead of 17/17/16). No solver had been
run on either set when the rule changed. Nothing has been solved on these
instances. When this was written, none of
the fifty names appeared anywhere else in the repository.

## Files

| File | What it is |
|---|---|
| `heldout/bounds.csv` | The held-out roster, in draw order, in `../bounds.csv`'s schema. It is the roster of record for this set. |
| `heldout/*.nl` | The fifty text-NL files, committed (1.1 MB, about the size of the published roster's 1.2 MB). |
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
  skips), which leaves **339 drawable**. The other 339 were not each fetched;
  the held-out walk fetched only the instances it took.
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
   already found unfetchable. Removing those before step 3 keeps a known-binary
   instance from costing its class a turn.
2. Within each structure class, order instances by
   `sha256("cbls-minlplib-heldout:144:<name>")`. This is a seeded shuffle that
   ignores size completely. A hash is used rather than `random.shuffle`
   because Python only guarantees `random()` and seeding to stay stable across
   versions, not `shuffle`. sha256 of a fixed string is stable everywhere.
3. Interleave the classes round-robin in sorted class order. This is the same
   rule `select()` uses for the published roster, so both sets are stratified
   the same way.
4. Fetch in that order until 50 text-NL files are in hand, recording any skip
   in `unfetchable.csv`. A network failure aborts the run instead of being
   recorded as a skip. Otherwise a transient outage would change the
   membership.

All four steps are `download.py`'s `select_heldout` and `build_heldout`. The
seed is a module constant with no command-line flag, because changing it
re-draws the set.

### Composition of the two halves

| Class | Published (`../bounds.csv`) | Held-out (`heldout/bounds.csv`) |
|---|---:|---:|
| bilinear | 5 | 0 (class exhausted) |
| polynomial | 10 | 0 (class exhausted) |
| mixed-integer | 15 | 17 |
| other | 14 | 17 |
| transcendental | 6 | 16 |
| **total** | **50** | **50** |

| Size (`nvars + ncons`) | Published (min / median / max) | Held-out (min / median / max) |
|---|---|---|
| mixed-integer | 4 / 11 / 36 | 36 / 96 / 196 |
| other | 1 / 3 / 4 | 6 / 33 / 220 |
| transcendental | 1 / 3.5 / 4 | 4 / 11 / 210 |
| whole set | 1 / 6.5 / 252 | 4 / 56.5 / 220 |

In both rosters the instances with integer variables are exactly the
mixed-integer class (15 and 17), and each roster has one maximisation instance. The held-out set gets 17 mixed-integer slots because the
round-robin runs over three classes, not five.

## Why not the "next fifty"

The obvious construction is to keep walking the published order and take the
next fifty survivors. It was **rejected**. Within each class, that order is
smallest first, so the next fifty are by construction the next-largest
instances in every class. Measured on this pool, they form a narrow band just
above the published roster: "other" sizes 4-6, "transcendental" 4-9,
mixed-integer 36-96.

Be clear about what that rejection does and does not buy. With the published
roster fixed, **every** set disjoint from it is larger than it (previous
section), and the seeded draw is further from it in size than the next fifty
would be: whole-set median 56.5, against 7.0 for the next fifty and 6.5 for the
published roster. So the "systematically harder" objection in #144 applies to
the chosen set even more strongly. Choosing the seeded draw does not remove the
size difference. It trades a small size difference for a useful property:

- **The next fifty has no size spread to diagnose with.** The band is so
  narrow that a size effect would show up as a shift of the whole set, and
  that shift looks exactly like over-fitting. Nothing inside the set could tell
  them apart.
- **The seeded draw is a sample of the remainder's own size distribution.**
  Sizes run from 4 to 220 overall, with a wide spread inside each class
  (mixed-integer 36-196, other 6-220, transcendental 4-210). That lets #145 ask
  whether an arm's effect *varies with size within the held-out set*. If no
  size trend is visible, a size explanation becomes less likely. It is not
  ruled out: there are only fifty instances, and size is partly confounded with
  class (every mixed-integer instance is size 36 or more). If a trend is
  visible, the size dependence is itself the finding: the shipped value was
  fitted to the small end of the pool.

So #144's acceptance criterion, "both halves have comparable size and
structure-class composition", is **not met and cannot be met** while the
published roster stays fixed. This document records that as the precondition
finding, rather than presenting the draw as comparable.

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
  size (median 6.5 against 56.5). A difference in absolute numbers is not
  evidence of over-fitting. If any cross-set comparison is wanted, use the
  published roster's 35 instances in the three shared classes, and still treat
  size as a covariate.
- **Families cluster.** The held-out set holds 9 `graphpart_*` instances (9 of
  its 17 mixed-integer ones) and 9 `ex8_*` instances (7 of them
  transcendental). That is proportional to the pool, but instances in one
  family behave alike, so the effective sample is smaller than fifty. A
  count-of-wins statement should be reported per class, and a family should be
  treated as one cluster rather than as nine independent votes.
- **`nvars + ncons` misses expression size.** `eg_disc2_s` counts as size 36
  but its `.nl` is 925 KB, 81% of the held-out bytes. Treat it as an outlier
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
drivers treat anything under it as published. Always pass `--out`: a
whole-roster `cbls_minlplib .../heldout` run at default flags with `--commit`
writes `heldout/comparison.csv`. `heldout/` has no
`comparison.csv`, `scip_baseline.csv` or `analysis_notes.csv`, and the runner
does not need them. `run_ablation.py`'s arm set is fixed in its `ARMS` table,
so a three-point grid around the shipped `--unproductive-iters 300` needs those
arms added there, or a loop over the runner command above.

## Reproducing and checking

    .venv/bin/pytest tests/python/test_minlplib_heldout.py -q

That test re-derives both rosters from `heldout/pool.csv` and
`heldout/unfetchable.csv`, with no network access, and compares them byte for
byte against the two committed `bounds.csv` files. It also pins the class
counts above, checks that the two sets are disjoint, checks that the draw
depends on the seed, and checks that every held-out row has its text-NL file.

To re-run the draw itself (it refuses to replace a committed roster without
`--force`). It keeps the committed `.nl` files rather than re-fetching them, but
it does fetch the published walk's eight skips again to re-check them:

    .venv/bin/python3 benchmarks/instances/minlplib/download.py --heldout \
        --catalogue benchmarks/instances/minlplib/heldout/pool.csv --force

A future catalogue refresh is a **new pool**. The committed held-out membership
stays as it is, because re-drawing it from a newer catalogue would be the
revision the commitment exists to prevent.
