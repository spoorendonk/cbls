# Set covering — the `Set`-variable coverage check

Scoped validation of the collection-typed **`Set`** variable, not a benchmark
epic (issue #93). Until this was added, `Model::set_var` had **zero**
real-instance usage anywhere in the repo: the three Set move generators
(`set_add` / `set_remove` / `set_swap`) were reached only from synthetic 3-10
element unit tests, so any claim that the engine "generalises to List/Set
structured variables" was substantiated for List (pharma-glsp) and unsupported
for Set. **That List half has since been withdrawn**: pharma-glsp was retired
(#28) once its model turned out to be a relaxation of the source paper, so
neither structured type has a positive result behind it now.

Set covering is the natural workload: the decision *is* a subset of a universe,
so the model is one `Set` variable and nothing else.

## What this validates, and what it does not

| Question | Answer |
|---|---|
| Can a standard set-based problem be *expressed* with a `Set` variable? | **Yes** — one `Set` over the columns, one `lambda_sum` coverage row per row. No new DAG op was needed. |
| Does the search produce genuine, verified solutions? | **Yes** — every run on the roster returns a real cover, re-checked against the instance file. |
| Is the `Set` encoding *competitive*? | **No.** It never beats the plain Bool encoding of the same instance (it ties on two unicost instances), and on the weighted instances it costs 8.6-11.0x the optimum where Bool is within 9-20%. See [Result](#result). |
| Does #165's cost-aware selection fix it? | **Not demonstrably.** At 20 seeds per arm and a 10s budget, `ViolationGuided` beats the default by 2.6% on the weighted instances with a 95% CI of [−5.6%, +0.3%] — it crosses zero, and the estimate shrank as seeds were added. See [Cost-aware structural selection](#cost-aware-structural-selection-165--measured-twice-still-not-established). |

So the honest scope of the structured-variable claim today is: **`Set`
variables are validated for expressiveness only, and `List` variables are not
validated at all** — the benchmark that once carried the List claim
(pharma-glsp) was retired in #28. Making the Set search competitive is future
work, and [Why the Set encoding loses](#why-the-set-encoding-loses) says exactly
what is missing.

## Roster

Ten instances from J.E. Beasley's OR-Library set-covering collection
(<https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html>), the smallest
standard files in the collection, covering both cost regimes:

| Instances | Size | Costs | Optimum |
|---|---|---|---|
| `scp41`-`scp45` | 200 rows x 1000 cols | integer, 1-100 | 429, 512, 516, 494, 512 |
| `scpe1`-`scpe5` | 50 rows x 500 cols | all 1 (unicost) | 5 each |

Every optimum is **proven**, so scoring a result needs no reference solver run
— which is why this directory has no `reference_solve.py`.

- Primary source for the optima: J.E. Beasley, *An algorithm for set covering
  problems*, EJOR 31 (1987) 85-93, which the OR-Library page names as the source
  of the optimal values for problem sets 4-6 and A-E. Sets 4-6 originate with
  Balas & Ho and are distributed through OR-Library.
- Cross-checked against the public machine-readable table in
  [`fontanf/setcoveringsolver`](https://github.com/fontanf/setcoveringsolver/blob/master/data/data.csv),
  whose *Best known solution value* and *Best known bound* columns agree for all
  ten (i.e. optimality is proven, not merely best-known). That table's nonzero
  counts also match the vendored files exactly (4009, 3982, 3984, 4009, 3939 /
  4914, 5013, 5040, 4952, 5017), confirming these are the same files the optima
  refer to.

The files are **vendored** here (200 KiB total) so the C++ tests and the runner
need no network; `download.py --check` re-verifies dimensions, index ranges and
SHA-256 offline, and `download.py --force` re-fetches from OR-Library.

### File format

Whitespace-separated, line breaks insignificant:

```
m n
c(1) ... c(n)
for each row i: k(i) followed by k(i) 1-based column indices covering row i
```

## The two encodings

`../../setcover/setcover_model.h` builds the same instance twice:

- **`set`** — one `Set` variable over the column universe, cardinality bounded
  by `min(cols, rows)` (valid, not tuned: no *minimal* cover holds more columns
  than there are rows). Each row is `lambda_sum(chosen, covers_row_i) >= 1`; the
  objective is `lambda_sum(chosen, cost)`. The search moves only through the
  STRUCTURAL batch's Set add/remove/swap generators.
- **`bool`** — one Bool per column, `sum(x_j : j covers row i) >= 1`, objective
  `sum c_j x_j`. This is the ordinary linear encoding, which CP-SAT's
  violation-based LS worker also accepts, and it runs through Generalised
  Feasibility Jump like any scalar model.

Running both is the point: the interesting question is not "does the Set
variable work" but "does it buy anything the scalar encoding does not".

## Result

Best of seeds 42-44, 10s wall clock per run, single thread, default
`SearchConfig`, Release build. Every one of the 60 runs returned a **verified
cover** — feasibility recomputed from the instance file — so the expressiveness
half of the claim holds outright.

**Measured at engine commit `adc8ee4`** (#118, the per-constraint structural-pass
fix). Commits after it do touch engine code, but not these numbers: #114's
changes are all Int-gated, and these models declare no Int variable — the `Set`
encoding is one `Set`, the Bool encoding is N Bools. Re-derive the per-seed form
with the runner's `--csv` output. No summary table is committed; the table above
is the record.

| Instance | Optimum | `set` best | gap | `bool` best | gap |
|---|---|---|---|---|---|
| scp41 | 429 | 4739 | +1005% | 469 | +9% |
| scp42 | 512 | 4445 | +768% | 613 | +20% |
| scp43 | 516 | 4596 | +791% | 615 | +19% |
| scp44 | 494 | 4287 | +768% | 573 | +16% |
| scp45 | 512 | 4392 | +758% | 596 | +16% |
| scpe1 | 5 | 7 | +40% | 6 | +20% |
| scpe2 | 5 | 6 | +20% | 6 | +20% |
| scpe3 | 5 | 5 | **+0%** | 5 | **+0%** |
| scpe4 | 5 | 7 | +40% | 6 | +20% |
| scpe5 | 5 | 7 | +40% | 6 | +20% |
| **mean gap, weighted (scp4x)** | | | **+818%** | | **+16.1%** |
| **mean gap, unicost (scpex)** | | | **+28%** | | **+16%** |

These are best-of-3 at a fixed wall-clock budget, so they are samples, not
constants. Every row above comes from one serial sweep on an idle machine. What
repeated measurement has shown:

- The `bool` rows cannot move for a *code* reason across the #118 fix at all —
  `build_bool_model` creates no List or Set, so `has_structural` is false and
  the structural pass never runs — which makes them a control on measurement
  rather than on the engine. A pre-fix sweep and this post-fix sweep give the
  same best on all five weighted instances, and the same per-seed triple on four
  of the five; the exception is `scp41`, whose pre-fix `bool` block was itself
  run at reduced throughput.
- The `set` rows only compare within the post-fix engine. `scp41` has been swept
  three times post-fix, with a best of 4739 every time (seeds 42 and 43 are
  stable to the unit, seed 44 spans 4902-4963). The rest of the roster has one
  clean post-fix sweep each, so treat those as single samples.
- One earlier sweep was taken while the machine was loaded, and its `scp41` and
  `scp44` blocks ran at ~2/3 the iteration throughput of the rest. That did not
  move `scp41`/`set` — this search stalls well inside 10s — but it did move
  `scp44`/`bool`, which this table reported as 607 until the clean re-runs put
  it back at 573. Check `uptime` and the per-run iteration counts before
  trusting a sweep; the runner writes both to its `--csv` output.

Read it as two different results, because they are:

- **Weighted costs (`scp4x`)**: the `Set` encoding is *not usable*. It lands at
  8.6-11.0x the optimal cost while the Bool encoding of the identical instance is
  within 9-20%. Both select a comparable *number* of columns (100-102 vs 65-73) —
  the Set search is simply blind to which ones are cheap, because nothing in its
  move generator looks at cost or violation before proposing an element.
- **Unicost (`scpex`)**: the two encodings nearly converge (5-7 vs 5-6), and on
  `scpe3` the Set encoding reaches the proven optimum. With all costs equal,
  "which column" matters far less, and the objective reduces to cardinality —
  the one thing a random add/remove/swap can optimise.

That contrast is the sharpest available evidence for what is missing: not the
`Set` type, but a violation-guided choice of *which* element to move.

The one-instance headline: on `scp41`, `Set` reaches 4739 against a proven
optimum of 429, while the ordinary Bool encoding of the same data reaches 469.

## Cost-aware structural selection (#165) — measured twice, still not established

CLAUDE.md has long named **cost-aware structural move selection** as the
prerequisite for any renewed structured-variable claim, and #165 built it:
registrable move generators, granular neighbour lists, and a
`StructuralSelection` policy with a violation-guided arm. This roster is its A/B
harness.

**Result: not established.** Engine commit `8dc906b`, 20 seeds (42-61) per arm,
`Set` encoding, one solve at a time on an idle machine, via
`benchmarks/setcover/ab_selection.sh --time 10 --seeds 20`. 400 runs, every one
feasible and verified. This run is *after* #164's position-based move
representation, so the per-candidate element-vector copy that confounded the
first attempt is gone.

Paired per seed, objective difference `violation_guided − first_improving`:

| family | n (paired) | mean diff | 95% CI | distance from 0 | VG better |
|---|---|---|---|---|---|
| weighted, scp41-45 | 100 | **−76.2** on a base of ~2900 (−2.6%) | **[−161.4, +9.0]** | 1.75 SE | 60/100 |
| unicost, scpe1-5 | 100 | **−0.18** on a base of ~6.7 | [−0.4, −0.0] | 2.07 SE | 34/100 (45 ties) |

Per-instance means (± sd over 20 seeds):

| instance | `first_improving` | `violation_guided` | rel. |
|---|---|---|---|
| scp41 | 2989.1 ± 229.8 | 2902.6 ± 386.1 | −2.9% |
| scp42 | 2815.4 ± 270.5 | 2825.1 ± 312.7 | +0.3% |
| scp43 | 2956.4 ± 316.6 | 2806.8 ± 290.9 | −5.1% |
| scp44 | 2941.4 ± 331.4 | 2821.0 ± 311.8 | −4.1% |
| scp45 | 2919.5 ± 289.5 | 2885.4 ± 314.7 | −1.2% |
| scpe1-5 | 6.4-6.9 | 6.3-6.8 | −6% to 0% |

**Read this as "not established", not as "it works".** On the weighted instances
the confidence interval crosses zero. A one-sided reading would clear 0.05 and a
two-sided one would not — precisely the regime where choosing the favourable
framing is how a null becomes a claim, so the interval is quoted instead. The
direction is consistently non-worse (9 of 10 instances), which is weak positive
evidence and nothing more; no individual instance separates.

**The decisive detail is that the estimate SHRANK with more data.** An earlier
five-seed run of the same comparison, at the same budget and on the same
representation, put the weighted difference at −131.9 (−4.5%). Four times the
data moved it to −76.2 (−2.6%), interval still spanning zero. That is regression
toward the mean, and it is why the five-seed reading is not recorded here as a
result. Per-seed spread is ±8-11% of the objective — larger than the effect being
looked for — and run-to-run variation within the *same* arm is comparable:
scp44's `first_improving` mean was 2703 over seeds 42-46 and 2941 over 42-61.

**What would settle it**, in preference order: a longer per-run budget, which
attacks the noise at its source instead of averaging it down; then more seeds
(sd 434.7 implies ~740 paired runs for a 3-SE reading of a −76 effect); then a
harder roster, since 10s is already past the knee here. None of that is worth
doing before there is a reason to care about `Set`-encoded set covering
specifically. The honest summary is that cost-aware selection is **implemented,
measured, and not demonstrably better** on the one roster that can test it.

It does not rescue the headline result below either: the `Set` encoding's
8.6-11.0x remains what it was, because a prerequisite being *implemented* is not
the same as its being *effective*.

**On the representation, separately.** #164 made a structured candidate carry
positional edits rather than the whole element vector, so scoring one costs an
allocation-free `assign` per variable its predecessor changed where it used to
cost at least two heap allocations and three O(|elements|) copies. (Not "no
allocation at all" — an `EditKind::Replace` allocates by construction, and the
per-call membership scans still do.) The one place the effect was measured
directly is the diversification kick, which does **not** go through the
structural batch: over a 20 000-element `List` it now runs in 0.022 s where the
pre-#164 code did not finish inside 0.2 s, i.e. at least ~9x. **No
search-throughput A/B has been run**, and the table above is the only evidence
about the batch.

## Why the Set encoding loses

On a model whose only variable is a `Set`, most of the engine is inert:

| Mechanism | On a Set-only model |
|---|---|
| Feasibility Jump batch | no jumpable variable, so `apply_jump` fails every iteration and the batch degenerates into a pure GLS weight pump |
| Novelty Jump | same — compound moves are built from scalar jumps (and it is off by default) |
| `perturb` diversification kick | reaches the Set since #111 (a kick applies `clamp(round(p*|elements|), 1, |elements|)` random structural moves to it), but those are the same unguided add/remove/swap, so it lands somewhere arbitrary rather than somewhere better — measured ~30% *worse* on the weighted instances than the pre-#111 no-op |
| LNS destroy-repair | destroys the single Set variable wholesale (a random restart) and repairs with FJ, which has nothing to jump |
| STRUCTURAL batch | the only mechanism that moves anything |

and the STRUCTURAL batch is a **first-improvement hill climber over three
randomly sampled moves**: per pass, `set_moves` proposes one random add, one
random remove and one random swap, each kept only if it strictly lowers weighted
violation. There is no violation-guided choice of *which* element to add or drop
(the scalar path has exactly that, in FJ's jump table and best-of-N scan-set
sampling). Progress once the sampled neighbourhood dries up is slow rather than
absent: 6x the budget buys ~18% (scp41 `set`, best of seeds 42-44 at the current
engine commit: 4739 at 10s, 3876 at 60s — the 60s sweep is 4275 / 4438 / 3876),
so the table above is a floor set by the budget, not a converged result.

The 3x4 fixture in `tests/test_setcover.cpp` shows the failure in miniature:
column 0 covers all three rows for 5, columns 1-3 cover one row each for 1, so
the optimum is 3. From `{0}` every single add, remove or swap either uncovers a
row or costs more, so the Set encoding stalls at 5 while the Bool encoding
reaches 3.

Spending the whole batch budget on structural passes is better in **both** cost
regimes — the idle FJ batches buy nothing that outweighs the passes they
displace. Both columns below are one 10s sweep at engine commit `adc8ee4`,
seeds 42-44 in that order:

| Instance | default | `--struct-prob 1.0` |
|---|---|---|
| scpe1 (unicost) | 7, 7, 7 | 6, 6, 6 |
| scp41 (weighted) | 4917, 4739, 4902 | 2593, 2727, 2916 |

Scored **best-to-best**, the way the roster table above is scored, that is
`1 - 2593/4739` = a **~45%** improvement on scp41 — still ~6x the optimum, so it
changes nothing about the conclusion: the encoding is not competitive at either
setting. (Mean-to-mean the same sweep gives ~43%; an earlier revision of this
file quoted the two methods against each other, so the number is stated with its
method from here on.) It is reported here rather than in the roster table
because the roster uses the engine default throughout, which is the
configuration a user gets.

(An earlier revision of this file recorded the opposite for the unicost column
and explained it as a weight-pump-as-diversification effect. That reading was an
artefact of measuring before the structural kick landed; the effect does not
reproduce at engine HEAD.)

## Reproducing

```bash
python benchmarks/instances/setcover/download.py --check   # verify the vendored files
cmake -B build-rel -DCMAKE_BUILD_TYPE=Release && cmake --build build-rel -j4
./build-rel/cbls_setcover --time 10 --seeds 3 --csv comparison_raw.csv
./build-rel/cbls_setcover --instance benchmarks/instances/setcover/scpe1.txt --encoding set
ctest --test-dir build-rel -R setcover
```

The runner verifies every solution against the instance file (not against the
DAG) and exits non-zero if any run is infeasible or unverified.

The selection A/B above is one command, and it refuses to run its arms
concurrently:

```bash
./benchmarks/setcover/ab_selection.sh --time 10 --seeds 5
# --arms first_improving,best_of_sample,violation_guided  for all three
```

It writes per-arm and merged per-seed CSVs plus a `run.txt` carrying the engine
commit, the host and `uptime`, into a timestamped directory under `results/`
(gitignored). Quote the commit it prints.

The runner's `--csv` output is the per-seed raw form. **No summary table is
committed.** One used to be, and it went stale silently when #118 changed the
search trajectory — nothing in the suite reads a committed CSV, so there is no
gate that would have caught it. The Result table above is the record instead,
and re-measuring is the way to check it.

## Provenance and licensing

- Instance data: OR-Library (J.E. Beasley), <https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html>.
  OR-Library states its data sets are freely available for research use.
- Optima: Beasley, EJOR 31 (1987) 85-93; cross-checked as described above.
- No third-party source code is vendored; the parser is written from the format
  description on the OR-Library page.
