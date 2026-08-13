# Set covering — the `Set`-variable coverage check

Scoped validation of the collection-typed **`Set`** variable, not a benchmark
epic (issue #93). Until this was added, `Model::set_var` had **zero**
real-instance usage anywhere in the repo: the three Set move generators
(`set_add` / `set_remove` / `set_swap`) were reached only from synthetic 3-10
element unit tests, so any claim that the engine "generalises to List/Set
structured variables" was substantiated for List (pharma-glsp) and unsupported
for Set.

Set covering is the natural workload: the decision *is* a subset of a universe,
so the model is one `Set` variable and nothing else.

## What this validates, and what it does not

| Question | Answer |
|---|---|
| Can a standard set-based problem be *expressed* with a `Set` variable? | **Yes** — one `Set` over the columns, one `lambda_sum` coverage row per row. No new DAG op was needed. |
| Does the search produce genuine, verified solutions? | **Yes** — every run on the roster returns a real cover, re-checked against the instance file. |
| Is the `Set` encoding *competitive*? | **No.** It never beats the plain Bool encoding of the same instance, and on the weighted instances it costs 5-7x the optimum where Bool is within 9-29%. See [Result](#result). |

So the honest scope of the structured-variable claim today is: **List variables
are validated for quality (pharma-glsp); `Set` variables are validated for
expressiveness only.** Making the Set search competitive is future work, and
[Why the Set encoding loses](#why-the-set-encoding-loses) says exactly what is
missing.

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

The files are **vendored** here (236 KiB total) so the C++ tests and the runner
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
half of the claim holds outright. `comparison.csv` carries the same numbers in
the repo-standard schema; the per-seed values are in the runner's `--csv` output.

| Instance | Optimum | `set` best | gap | `bool` best | gap |
|---|---|---|---|---|---|
| scp41 | 429 | 3260 | +660% | 469 | +9% |
| scp42 | 512 | 3045 | +495% | 646 | +26% |
| scp43 | 516 | 3387 | +556% | 651 | +26% |
| scp44 | 494 | 3825 | +674% | 636 | +29% |
| scp45 | 512 | 3053 | +496% | 596 | +16% |
| scpe1 | 5 | 7 | +40% | 6 | +20% |
| scpe2 | 5 | 7 | +40% | 6 | +20% |
| scpe3 | 5 | 7 | +40% | 5 | **+0%** |
| scpe4 | 5 | 7 | +40% | 7 | +40% |
| scpe5 | 5 | 7 | +40% | 6 | +20% |
| **mean gap, weighted (scp4x)** | | | **+576%** | | **+21%** |
| **mean gap, unicost (scpex)** | | | **+40%** | | **+20%** |

Read it as two different results, because they are:

- **Weighted costs (`scp4x`)**: the `Set` encoding is *not usable*. It lands at
  5-7x the optimal cost while the Bool encoding of the identical instance is
  within 9-29%. Both select a similar *number* of columns (78-97 vs 63-80) —
  the Set search is simply blind to which ones are cheap, because nothing in its
  move generator looks at cost or violation before proposing an element.
- **Unicost (`scpex`)**: the two encodings nearly converge (7 vs 5-7). With all
  costs equal, "which column" matters far less, and the objective reduces to
  cardinality — the one thing a random add/remove/swap can optimise.

That contrast is the sharpest available evidence for what is missing: not the
`Set` type, but a violation-guided choice of *which* element to move.

The one-instance headline: on `scp41`, `Set` reaches 3260 against a proven
optimum of 429, while the ordinary Bool encoding of the same data reaches 469.

## Why the Set encoding loses

On a model whose only variable is a `Set`, most of the engine is inert:

| Mechanism | On a Set-only model |
|---|---|
| Feasibility Jump batch | no jumpable variable, so `apply_jump` fails every iteration and the batch degenerates into a pure GLS weight pump |
| Novelty Jump | same — compound moves are built from scalar jumps (and it is off by default) |
| `perturb` diversification kick | documented no-op: it randomises jumpable variables only |
| LNS destroy-repair | destroys the single Set variable wholesale (a random restart) and repairs with FJ, which has nothing to jump |
| STRUCTURAL batch | the only mechanism that moves anything |

and the STRUCTURAL batch is a **first-improvement hill climber over three
randomly sampled moves**: per pass, `set_moves` proposes one random add, one
random remove and one random swap, each kept only if it strictly lowers weighted
violation. There is no violation-guided choice of *which* element to add or drop
(the scalar path has exactly that, in FJ's jump table and best-of-N scan-set
sampling), and no escape once no sampled move improves.

The 3x4 fixture in `tests/test_setcover.cpp` shows the failure in miniature:
column 0 covers all three rows for 5, columns 1-3 cover one row each for 1, so
the optimum is 3. From `{0}` every single add, remove or swap either uncovers a
row or costs more, so the Set encoding stalls at 5 while the Bool encoding
reaches 3.

One second-order effect is worth recording because it makes tuning
counter-intuitive: those idle FJ batches are *not* harmless. Their weight pump
inflates the GLS weight of the persistently-violated objective row, and a
sufficiently skewed weight vector is what lets the structural pass accept a move
that breaks a coverage row — the closest thing the Set search has to
diversification. Trading the pump for more structural passes therefore cuts both
ways (10s, seeds 42-44, `--struct-prob 1.0` vs the default 0.33):

| Instance | default 0.33 | `--struct-prob 1.0` |
|---|---|---|
| scpe1 (unicost) | 8, 7, 7 | 10, 10, 9 |
| scp41 (weighted) | 3877, 3260, 3625 | 2187, 2303, 2517 |

Neither setting is the fix, and neither is reported above as the headline: the
roster table uses the engine default throughout.

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

`comparison.csv` holds the summary in the repo-standard schema
(`instance,method,objective,gap,source`); the runner's `--csv` output is the
per-seed raw form.

## Provenance and licensing

- Instance data: OR-Library (J.E. Beasley), <https://people.brunel.ac.uk/~mastjjb/jeb/orlib/scpinfo.html>.
  OR-Library states its data sets are freely available for research use.
- Optima: Beasley, EJOR 31 (1987) 85-93; cross-checked as described above.
- No third-party source code is vendored; the parser is written from the format
  description on the OR-Library page.
