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
pip install ortools

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
columns, 961k rows, 4.9M nonzeros) peaked at **1.2 GB under CBLS and 3.3 GB
under CP-SAT**, and CP-SAT carries a ~100 MB floor on even the smallest models.

It is not the roster's largest model, though: `square47` carries 27.4M nonzeros
(5.6x kasavu) and `supportcase19` 1.43M columns (2x). Note also that
`--mem-limit-gb` caps *address space* (`ulimit -v`), which runs roughly 30% above
resident set — so confirm the cap on the target machine before trusting
`--jobs 4` rather than reading 6 GB off the RSS figures above.

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

With both fixed, on this subset at 60s: CP-SAT reaches feasibility on 8 of 11
against CBLS's 7, and leads on the shifted geometric mean (0.256 vs 0.652). CBLS
is close on `gen-ip054` (5.0% gap) and ahead on `pk1` (PI 0.911 vs 0.978) and
`neos5`; it is far behind on `binkar10_1` (1.01e6 against an optimum of 6741)
and `markshare2`. Dropping CP-SAT from two workers to one moved its aggregate by
0.0004, so the earlier asymmetry was not what the gap was made of.

Whether any of this holds over 233 instances at 600s is exactly what the full
run is for.

## Configuration, and what is recorded

Both engines are run at a stated configuration, recorded per result rather than
inherited, so a published number cannot silently change when a default moves:

| | CBLS | CP-SAT |
|---|---|---|
| Threads | 1 | 1 (`num_workers`) |
| Algorithm | Feasibility Jump + ViolationLS | `fj` + `ls` workers only (`filter_subsolvers`) |
| Presolve | none | default, i.e. on |
| Feasibility tolerance | `1e-6`, stated explicitly | CP-SAT's own |
| Recorded per result | commit SHA, seed, tolerance, peak RSS | OR-Tools version, seed, full parameter string, peak RSS |

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
