# MIPfeas benchmark

Head-to-head against the reference implementation of the algorithm this engine
reimplements. **This is an implementation sanity check, not a MIP-competitiveness
claim** — see epic #87 for why a MIP shootout is deliberately out of scope.

## The question

CP-SAT ships a violation-based local search directly descended from Feasibility
Jump — the `fj` and `ls` workers described in Davies, Didier & Perron,
*ViolationLS: Constraint-Based Local Search in CP-SAT*, CPAIOR 2024
([paper](https://link.springer.com/chapter/10.1007/978-3-031-60597-0_16)). CBLS
implements the same algorithm from scratch. Running both on the same instances at
the same budget answers one narrow question: **does the reimplementation hold up
against the reference implementation?**

A gap in either direction is informative. What would *not* be informative is
comparing against CP-SAT's default portfolio, or against Xpress/Gurobi/CPLEX:
those bring presolve, cuts and decades of MIP heuristics to a fight this engine
is not in. They are excluded by design, not by oversight.

## Roster: MIPfeas, 233 instances

[MIPfeas](https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/)
is the MIPLIB 2017 *benchmark set* (240 instances) minus the ones known to be
infeasible, which are dropped so the metric stays well defined.

The roster is not published as a name list, so `download.py` derives it:

```
benchmark-v2.test (240 names)  -  {names tagged =inf= in miplib2017-v36.solu}  =  233
```

and asserts the count. A MIPLIB revision that moves either input fails the
download rather than silently redefining the benchmark. The seven excluded
instances are `bnatt500`, `cryptanalysiskb128n5obj14`, `fhnw-binpack4-4`,
`neos-2075418-temuka`, `neos-3402454-bohle`, `neos-3988577-wolgan` and
`neos859080`.

Of the 233, 232 have a proven optimum in the solution file and one carries a
best-known value; `roster.csv` records which per instance.

## Metric: Primal Integral

MIPfeas ranks on the Primal Integral rather than the gap at the time limit,
which is the right question for a heuristic — reaching a good solution *early*
is the thing it is for. With `x*` the reference value and `x(t)` the incumbent,

```
p(t) = |x(t) - x*| / max(|x(t)|, |x*|)      P(T) = (1/T) * integral of p over [0, T]
```

with `p = 2` while no feasible solution exists, `p = 1` across a sign flip, and
`p = 0` when both values are below `1e-6`. So `P` runs from 0 (optimal
immediately) to 2 (never feasible), and lower is better. The primary ranking is
the shifted geometric mean (shift 0.001).

## Layout

```
benchmarks/instances/mipfeas/
  download.py     roster derivation + instance fetch
  roster.csv           233 instances with reference values  (committed)
  smoke.csv            11-instance subset for wiring checks (committed)
  manifest.csv         sha256 + byte size per instance      (committed)
  smoke_comparison.csv the wiring check's output            (committed)
  comparison.csv       scored results                       (committed once a run is published)
  *.mps.gz             instances, ~546 MiB                   (gitignored)

benchmarks/mipfeas/
  mipfeas.cpp        CBLS runner, one instance per process
  cpsat_solve.py     CP-SAT fj+ls baseline, same result schema
  run_benchmark.py   driver: parallel, resumable, memory-capped
  primal_integral.py scoring -> comparison.csv
```

The instances are too large to vendor, so `manifest.csv` pins the exact bytes the
roster refers to. Because they can be absent, both runners **refuse to write a
result for a missing instance** rather than recording it as "found nothing" —
that failure mode is what emptied a published table in #103.

## Running it

```bash
# 1. Fetch the roster (~546 MiB via benchmark.zip; --subset smoke for the 11)
python benchmarks/instances/mipfeas/download.py

# 2. Build the CBLS runner
cmake -B build && cmake --build build -j$(nproc) --target cbls_mipfeas

# 3. Install the baseline
pip install -e '.[benchmarks]'

# 4. Wiring check: 11 instances, both engines, short budget
python benchmarks/mipfeas/run_benchmark.py --roster smoke --budget 60 --jobs 2

# 5. The publishable run (233 x 600s x 2 engines is ~78 CPU-hours)
python benchmarks/mipfeas/run_benchmark.py --roster full --budget 600 \
    --jobs 4 --mem-limit-gb 6

# 6. Score it
python benchmarks/mipfeas/primal_integral.py \
    --results-dir results/mipfeas --roster benchmarks/instances/mipfeas/roster.csv \
    --budget 600 --out benchmarks/instances/mipfeas/comparison.csv
```

The driver is resumable: a job whose result file exists is skipped, so an
interrupted run continues where it stopped.

Sizing `--jobs` is a memory question rather than a core-count one. Every result
records its `peak_rss_kib`. From the wiring check, `neos-5114902-kasavu` (710k
columns, 961k rows, 4.9M nonzeros) peaked at **1.2 GB under CBLS and 3.2 GB
under CP-SAT**, and CP-SAT carries a ~100 MB floor on even the smallest models.

It is not the roster's largest model, though: `square47` carries 27.4M nonzeros
(5.6x kasavu) and `supportcase19` 1.43M columns (2x). `square47` also spends
~100-150s on model build and un-deadlined search initialisation before its first
iteration, so budget wall-clock accordingly — the driver allows 900s of slack on
top of the budget for exactly this. Note also that
`--mem-limit-gb` caps *address space* (`ulimit -v`), not resident set. CBLS runs
about 1.4x its RSS; CP-SAT adds a roughly constant ~0.7 GB of reservations on top
of its own, so the ratio is ~1.4x on a large model like `neos-5114902-kasavu` but
~4.4x on a small one like `atlanta-ip`. Confirm the cap on the target machine
rather than reading 6 GB off the RSS figures above.

## What the wiring check found

`smoke_comparison.csv` is 11 instances at 60s — enough to prove the harness end
to end, and **not** a result. Both engines honoured the budget, the largest
instance in the subset ran without OOM, and the driver resumed correctly after
being interrupted.

The first run of it found two defects in the harness rather than in either
solver, which is what a wiring check is for. The MPS reader was binarising
general-integer columns carrying an `LI` bound — under that restriction
`gen-ip054` and `enlight_hard` have no feasible point at all, and `gen-ip002`'s
optimum moves, so all three read as CBLS failing. And CP-SAT was being given two
workers on a mistaken belief that one would not run its `ls` worker.

It also settled the two configuration choices above by measurement rather than
argument. Each was run over the whole smoke roster, CBLS only:

| Novelty Jump | clamp | shifted geomean | mean | feasible |
|---|---|---|---|---|
| off | 1e9 | 0.654 | 1.109 | 7/11 |
| on | 1e9 | 0.614 | 0.961 | 8/11 |
| off | 1e7 | 0.638 | 1.089 | 7/11 |
| **on** | **1e7** | **0.561** | **0.937** | **8/11** |

Novelty Jump turns `mas76` from no-solution into feasible and improves
`binkar10_1` (PI 0.80 → 0.60), `pk1` and `neos5`, at a small cost on `gen-ip054`
and `gen-ip002`. The two changes compose, and together they move `binkar10_1`'s
objective from 1,010,195 to 9,865 against an optimum of 6,741.

**Both rows of the clamp column predate #120** and no longer describe the
engine. Implied bounds now clear the clamp entirely on 5 of these 11 instances —
`binkar10_1` among them, which is where the 1e9-vs-1e7 difference was largest —
so on those the setting no longer reaches anything. It still binds on `mad`,
`pk1`, `gen-ip002` and `gen-ip054`; the remaining two, `mas76` and `neos5`, had
no clamped column to begin with. The numbers below were measured before that
change and have not been re-run; treat them as the pre-propagation baseline they
are.

At that configuration, on this subset at 60s: both engines reach feasibility on
8 of 11, and CP-SAT leads on the shifted geometric mean (0.258 vs 0.566) — it
gets to good solutions much earlier, which is what the metric is built to
reward. CBLS ends ahead on `pk1` (74 vs 473), `mad` (1.79 vs 2.00) and `neos5`
(15.5 vs 16.0), and is close on `gen-ip054` (2.0% gap). It remains far behind on
`markshare2` and `binkar10_1`. Dropping CP-SAT from two workers to one moved its
aggregate by 0.0004, so that asymmetry was never what the gap was made of.

Two caveats on reading any of these numbers. The A/B rows other than the chosen
one were measured before the harness reached its current state and are not
reproducible from what is committed here — only the bottom row is. And a
time-limited `solve()` is not bit-reproducible even at a fixed seed (the engine
is deterministic only when bounded by iterations, `time_limit = 0`), so re-running
this roster moves the aggregate in the third decimal. Recording the commit SHA and
seed pins the code and the configuration, not the exact objective.

Whether any of this holds over 233 instances at 600s is exactly what the full
run is for. One thing to expect there: `neos-5114902-kasavu` (961k rows) took
87s against a 60s budget, because model build and search initialisation are not
bounded by the deadline. The Primal Integral is unaffected — it integrates over
[0, budget] whatever the wall clock does — but the run takes longer than the
budget times the job count implies.

## Configuration, and what is recorded

Both engines are run at a stated configuration, recorded per result rather than
inherited, so a published number cannot silently change when a default moves:

| | CBLS | CP-SAT |
|---|---|---|
| Threads | 1 | 1 (`num_workers`) |
| Algorithm | Feasibility Jump + ViolationLS + Novelty Jump | `fj` + `ls` workers only (`filter_subsolvers`) |
| Presolve | implied variable bounds only (activity-based propagation) | default, i.e. on |
| Feasibility tolerance | `1e-6`, stated explicitly | CP-SAT's own |
| Unbounded column falls back to | `1e7` (`--inf-clamp`), where propagation derives nothing | not clamped |
| Recorded per result | commit SHA, seed, tolerance, clamp + columns it still narrows, columns declared unbounded, columns tightened, propagation verdict and pass cap, compound-move and propagation flags, peak RSS | OR-Tools version, seed, full parameter string, solver verdict, peak RSS |

Two of those are deliberate departures from the engine's own defaults, both made
to keep the two sides comparable rather than to flatter either:

* **Novelty Jump is on**, though `SearchConfig::use_compound_moves` defaults to
  off. That default exists because the per-batch cost was not bounded tightly
  enough for the large *continuous* benchmarks — not this roster. Roughly half of
  CP-SAT's incumbents here come from its own compound-move subsolvers
  (`ls_restart_*compound*`: 45–67% of improving solutions on binkar10_1 and pk1),
  so running without it would compare our Feasibility Jump against their
  Feasibility Jump *plus* Novelty Jump and read the difference as a
  reimplementation gap.
* **A column no constraint bounds falls back to 1e7**, not the engine's 1e9 —
  because it measured better on the smoke roster, *not* because it matches the
  baseline. CP-SAT does **not** truncate variable domains: `mip_max_bound` is not
  a domain clamp, and on ortools 9.15 an integer column bounded at 1e12 is solved
  to 1e12. So this remains a CBLS-side restriction the baseline does not share.
  It can lose solutions, never invent them, so an objective CBLS reports stays
  valid for the original program — but where it bites, CBLS searches a strictly
  smaller box, and the comparison table publishes `n_clamped_bounds` per row so a
  reader can see where. Note `n_clamped_bounds` also counts the int32 clip on an
  integer column the file bounded finitely, so it is **not** a subset of
  `n_unbounded_columns` and the difference of the two is not "what propagation
  removed".

### Implied bounds shrink that restriction (#120)

Before #120 the fallback *was* the whole story: every unbounded column got `1e7`
substituted, and so did every column declared wider than it. Activity-based
propagation over the linear rows now derives bounds that the constraints
actually imply, and the fallback is consulted only on what is left. Two rules
changed together:

* An implied bound is used wherever one exists. Unlike the substitute, it cannot
  put a feasible point outside the box.
* A bound that *exists* — declared in the file or derived — is honoured however
  wide. Narrowing a declared bound has the same defect as inventing one; across
  the 233 instances it reached 110 columns on 6 of them, so stopping gives up
  nothing measurable.

Measured over all 233 roster instances, model build only (no search). First taken
at `aec705b` and re-derived after the review fixes that followed it; every count
below is identical across the two runs:

| | before propagation | after |
|---|---:|---:|
| Instances with a clamped column | 116 | **52** |
| Columns the clamp supplies | 2,766,548 | **356,827** |

64 instances are cleared entirely, `binkar10_1` (2128 of 2298 columns clamped,
and the largest CBLS-vs-reference gap on the smoke roster) and
`neos-5114902-kasavu` (695,604 of 710,164) among them. A further 519,524 columns
across 82 instances are fixed outright to a single value.

Cost, timed in isolation rather than by differencing two model builds (that
difference is smaller than the run-to-run noise): **~1.1s worst case**, on
`square47` — 0.38s assembling the rows plus 0.73s propagating over its 27.4M
nonzeros. `neos-5114902-kasavu`, the largest by rows, costs 0.26s over 4.9M
nonzeros in 3 passes. `supportcase12` is the only instance that reaches the
10-pass cap, at 0.31s. Roughly 7s summed over the whole roster, against a 600s
per-instance budget.

Two caveats a reader should have:

* **"116 of 233" is not the "120 of 233" this README used to state.** The old
  count was relative to the `1e7` clamp and so included columns merely *wider*
  than it; this one counts columns with no bound at all, at the `1e20`
  MPS/CPLEX/SCIP sentinel. The definition changed, not the roster.
* **Soundness is argued and tested, not verified against known optima.** The
  `.solu` file carries objective values, not solution vectors, so "the optimum
  is still inside the box" cannot be checked per instance — an excluded optimum
  would read as search weakness, not as infeasibility. What is checked: a unit
  test where a fixed clamp excludes the only solution and propagation keeps it,
  and 233/233 instances propagating without a single false infeasibility.

`--no-propagate-bounds` turns off propagation only — the second rule change,
honouring a finite bound however wide, is unconditional — so it is an A/B on
propagation rather than a return to the pre-#120 engine. It exists for A/B work;
it is not a configuration to publish, because the bounds it restores are not
implied by the constraints.

Two configuration facts are worth knowing before changing anything here, both
established empirically against ortools 9.15:

* `filter_subsolvers` is the **only** parameter that accepts `fj`/`ls`.
  `subsolvers` and `ignore_subsolvers` validate against full-problem subsolver
  names and reject both, so they cannot express "LS only".
* `num_workers: 1` runs the whole algorithm — the log reports `1 first solution
  subsolver: [fj]` and `1 interleaved subsolver: [ls]` — so both engines get one
  thread. Raising it multiplies *both* workers (`num_workers: 2` gives `fj(2)`
  and `ls(2)`, roughly twice the CPU in the same wall time), so the count is
  recorded per result. `ls` without `fj` never bootstraps a first solution at any
  worker count, which is why both are enabled.

One asymmetry is not configuration and cannot be tuned away: CBLS's clock starts
inside `solve()`, so its MPS read and model build are free, while CP-SAT's log
timestamps include its presolve. On the largest instances that is a few seconds
against a 600s budget — under 1%, but it favours CBLS and is stated here rather
than left to be discovered.

## Provenance

- Instances and reference values: MIPLIB 2017, <https://miplib.zib.de/>
  (`benchmark-v2.test`, `miplib2017-v36.solu`, `benchmark.zip`).
- Metric and roster definition: the MIPfeas announcement linked above.
- Baseline: OR-Tools CP-SAT (Apache-2.0), version recorded per result.
