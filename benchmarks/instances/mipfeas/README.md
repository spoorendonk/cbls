# MIPfeas benchmark

CBLS against OR-Tools CP-SAT **at its default parameters**, which holds the
reference implementation of the algorithm this engine reimplements. Every
comparison solver runs at its own defaults, with no setting changed and no
attempt to even out solver differences (Simon, 2026-11-01,
[#87](https://github.com/spoorendonk/cbls/issues/87#issuecomment-6558356494)).
Until then the baseline was CP-SAT restricted to its `fj` + `ls` workers; that
mode is still available (`--cpsat-workers N`), but no longer the campaign's.

## The question

CP-SAT ships a violation-based local search directly descended from Feasibility
Jump — the `fj` and `ls` workers described in Davies, Didier & Perron,
*ViolationLS: Constraint-Based Local Search in CP-SAT*, CPAIOR 2024
([paper](https://link.springer.com/chapter/10.1007/978-3-031-60597-0_16)). CBLS
implements the same algorithm from scratch. (Same jump and weighting; what each
does on stagnation differs, see **Restart versus kick** below.)

Restricted to those workers, CP-SAT answered one narrow question: **does the
reimplementation hold up against the reference implementation?** At its
defaults it answers a wider one: how CBLS fares against CP-SAT as it ships,
with presolve, its full portfolio and the worker count it picks for the
machine. The table states that baseline, each engine's version, and the
threads, CPU and wall time of every solve (**Where the time went**), and
reports the result without correcting for those differences. Xpress, Gurobi
and CPLEX stay out (#87).

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

## What the scorer reports

`primal_integral.py` writes two files: the comparison table, and a parity report
beside it (`--report`, default `<out stem>_report.md`). The report leads with what
the correctness sweep is for and keeps the anytime score as a co-equal section:

1. **defects** — rejected solutions, unverified rows, runs that beat a proven
   optimum, rejected models, killed jobs, jobs that never ran, flagged shape
   disagreements, degraded traces;
2. **feasibility parity** — each engine's feasible count over the roster, the
   instances both engines can be compared on, the agreement count, and **both
   asymmetric difference sets named instance by instance**;
3. **job failures** — the driver's kill message and the verifier's refusal
   reason;
4. **the model-shape cross-check** — see below;
5. **trace health** — how many anytime profiles are genuine, over a stated
   denominator (rows the engine reported feasible);
6. **anytime quality** — the Primal Integral aggregate, labelled with the exact
   baseline it measures (CP-SAT at its default parameters, or, for an fj+ls
   table, its `fj` + `ls` subsolvers under `num_violation_ls`), never "solvers"
   in general;
7. **where the time went** — setup and solve time in separate columns;
8. **the machine and run record** — host, cores, memory, the concurrency the run
   used, the memory cap, the engine commit, the solver versions, the budget and
   which yardstick file the gaps were scored against. A results directory with no
   record says so in this section rather than quietly omitting it.

A compact form of 1, 2, 4 and 5 also leads the table's own header, because that
is what a reader quoting the CSV will see. The table additionally carries
`verification_row_tolerance` / `verification_loosest_row` per row, read off the
verdict where the verifier recorded them: the row check is
`1e-6 + 1e-9 * max(|lower|, |upper|, sum|a_ij x_j|)`, which is unbounded in that
scale, so a pass says less on a model with large coefficients than on a small
one, and the number is published rather than argued about. Both cells are blank
for a verdict reached before the verifier recorded the figure.

### The model-shape cross-check

"Both engines solved the same program" is this benchmark's entire admission
ground, so each side's model shape is compared against the other's and against
the count SCIP recorded while verifying the row. The rule:

* **Variable counts must agree exactly.** Every reader enumerates the same MPS
  `COLUMNS` section, so a difference is a reader defect and never benign.
* **Constraint counts may differ by exactly the free rows the baseline keeps** —
  the `N` rows after the first, which is the objective. OR-Tools' `ModelBuilder`
  holds each remaining `N` row as a linear constraint with infinite bounds; the
  CBLS adapter drops them, and so does SCIP as the verifier reads it (observed on
  SCIP 10.0.2 / PySCIPOpt 6.2.1, the pair the verdicts record: `mad` has two `N`
  rows and `getNConss()` returns 51, matching the adapter). A free row constrains
  nothing, so the feasible sets are identical. `cpsat_solve.py` records the count
  as `n_free_cons`, so the rule subtracts a measured number rather than an argued
  one.
* **Anything else is flagged**, in either direction, including the third reader
  disagreeing with both engines.

`mad` is the live case: CBLS and SCIP read 51 rows, ModelBuilder 52, and the
instance has two `N` rows.

### Where the time went

The wall-clock column brackets the solve call only, and two different effects
hide behind that:

* **setup** (`setup_seconds`) — instance read, model build and bound propagation,
  all of which happen before the bracket. Both runners measure it: CBLS as
  `read_seconds` + `build_seconds`, CP-SAT as the `import_from_mps_file` call
  that does both at once. It is not charged against the search, but it is charged
  against the wall clock a run must be scheduled for.
* **overrun** — `solve_seconds` past the budget, because search initialisation is
  not bounded by the deadline: a large model's first batch runs to completion
  whatever the clock says. This is why the driver allows 900s of slack.

They are separate columns because a row can have one without the other.

Four more columns say what each solve was given and what it used (#285), since
every comparison solver runs at its own defaults and nothing evens the CPU out:

* **`threads_used`** — the threads the solve actually ran: CBLS's `threads`,
  CP-SAT's `workers_started`, read off its log (#284). At its defaults CP-SAT
  picks its own count, so this is the only record of it. The scorer refuses a
  table whose rows of one engine differ here, so a directory resumed on a host
  with other cores cannot mix two counts under one `workers=0` cell.
* **`driver_wall_seconds`**, **`driver_cpu_seconds`**, **`cpu_wall_ratio`** —
  the driver's own `driver_measured` numbers for the whole job (setup and solve),
  taken from outside the solver with `wait4`. The ratio is the CPU share the
  row actually got.

Each is blank where the row did not record it: a job that never ran, or a
result written before the field existed. `threads_used` alone is also blank on
a row the driver wrote for a killed job (its arm is nested under `driver_arm`)
and on a CP-SAT row whose log printed no worker count.

No aggregate setup magnitude is quoted for the roster: no per-row figure from a
real run is committed. Propagation alone costs ~1.1s worst case and ~7s summed
over the roster (**Implied bounds shrink that restriction**).

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
  download.py     roster derivation + instance fetch + pin verification
  roster.csv           233 instances with reference values  (committed, pinned)
  smoke.csv            11-instance subset for wiring checks (committed, pinned)
  miplib2017-v36.solu  the MIPLIB solution file it derives from (committed, pinned)
  references.csv       sha256 + byte size of the three above (committed)
  tune_confirm.csv     the tuning study's held-out 10, outside the roster, with
                       their .solu optima (committed, not in references.csv; a
                       pytest checks them against the .solu; #264)
  tuned.txt            the 22 roster instances settings were tuned on (the tune
                       study's screen + intermediate sets); scoring roster.csv
                       with --held-out tuned.txt reports the other 211 (#255)
  manifest.csv         sha256 + byte size per instance      (committed)
  smoke_comparison.csv the wiring check's output            (committed)
  comparison.csv       scored results                       (committed once a run is published)
  *.mps.gz             instances, ~546 MiB                   (gitignored)

benchmarks/mipfeas/
  mipfeas.cpp        CBLS runner, one instance per process
  cpsat_solve.py     CP-SAT baseline (defaults, or fj+ls), same result schema
  run_benchmark.py   driver: parallel, resumable, memory-capped
  verify_solution.py independent feasibility check against the instance file
  primal_integral.py scoring -> comparison.csv + <stem>_report.md
  testdata/          a frozen results directory and the two artifacts scored
                     from it, compared byte for byte by the test suite

results/mipfeas/
  run_record.json       machine + concurrency, one entry per invocation

results/mipfeas/<engine>/
  <instance>.json       result record
  <instance>.trace.csv  incumbent profile
  <instance>.sol        solution vector of a feasible run (MIPLIB format)
  <instance>.verify.json  the independent verdict on that solution
```

The instances are too large to vendor, so `manifest.csv` pins the exact bytes the
roster refers to. Because they can be absent, both runners **refuse to write a
result for a missing instance** rather than recording it as "found nothing" —
that failure mode is what emptied a published table in #103.

## Preconditions: the pins are checked, not merely written

Three things are verified before a run rather than discovered at scoring time
(#137).

### The bytes are what the pins say

`manifest.csv` pins every instance and `references.csv` pins the yardstick — the
MIPLIB solution file and both roster tables derived from it. `download.py
--verify` checks all of them offline and exits non-zero naming each mismatch, and
`run_benchmark.py` runs the same check before it dispatches the first job. A
corrupted, truncated or upstream-revised file therefore stops the run instead of
being measured under a published row's name.

Overwriting a pin takes an explicit flag, and the update prints what it changes:

* `download.py --update-references` accepts a moved solution file, rewrites
  `roster.csv` / `smoke.csv` and re-pins all three, printing the per-instance
  reference value and kind changes first. Without it a moved solution file
  **fails the download and nothing is written**. MIPLIB does revise its solution
  file — the shipped one carries a version in its name — and a silent revision
  moves every gap in the published table at once, which is strictly worse than a
  changed instance moving one row.
* `download.py --update-manifest` accepts instance bytes that no longer match.
  An instance the manifest has never seen is simply added: an acquisition is not
  an overwrite.

The `reference_kind` column is never dropped or flattened by a rewrite, and a
change to it is reported like a changed value — because a gap against the one
best-known value does not mean what a gap against a proven optimum means. It is
re-derived from the solution file's tag rather than frozen, so an upstream
`=best=` → `=opt=` does move it, visibly.

### The baseline is the one documented

Every parameter and log-format fact about CP-SAT in this harness was established
empirically against ortools 9.15, and the whole incumbent trace is scraped from a
debug log that carries no stability guarantee. A release that renames a subsolver
flag or reformats a log line does not crash — it produces a degraded or empty
baseline across all 233 instances at exit 0.

`cpsat_solve.py --preflight` solves one tiny model built in memory (no instance,
no network, ~2s) and asserts what the mode depends on, naming which broke.
With `--workers N` (fj+ls) it checks both halves below. At CP-SAT's defaults
(`--workers 0`, the default) there is no restriction to assert, so it checks a
**default configuration** half instead (CP-SAT accepted the one parameter it
sets, and finds a solution), plus the log format without the announcement lines:

* **worker restriction** (fj+ls mode only) — CP-SAT accepted the parameter string, the log's
  subsolver announcement lists exactly `fj` as first-solution and `ls` as
  interleaved, the thread count is the one asked for, and the configuration can
  actually find a solution;
* **log format** — the search header (`Starting search at Xs with N workers.`,
  which `workers_started` is read from) and, in the fj+ls mode, the announcement
  lines are present in the shape the harness parses, and the improving-solution
  lines still match the regex the incumbent trace is recovered from.

The split is by what moved: the *shape* of a line is the log format, its
*content* is the restriction. `pyproject.toml` bounds `ortools` to the range the
parsing was established against (`>=9.7,<9.16`); raise it by running the
preflight, not by editing the bound. The driver runs the preflight once before
dispatching any job, so a broken release fails at second zero rather than after
the roster has burned its budget.

### The machine is recorded

`run_benchmark.py` writes `run_record.json` into the results directory before the
first job — host, platform, core count and the cores actually available to the
process, total memory, **the concurrency the run used** (jobs at a time, large
instances at a time, CP-SAT workers, address-space cap), the budget, the engine
commit, the OR-Tools / PySCIPOpt / Python versions, the hashes of the pinned
reference files, and the path and hash of the roster CSV actually read — `--roster`
accepts any file, and the reference values a run is scored against are the ones in
the file it read. It also records whether the preconditions above were checked at
all, since `--skip-preconditions` is what makes a run unpublishable. A resumed run **appends** an
entry rather than overwriting the first, because such a directory was produced by
two machines and two concurrencies.

The report quotes the latest entry as its section 8, so a published table carries
its machine record rather than leaving it in a results directory nobody publishes.

`--skip-preconditions` turns off the byte check and the preflight. It is for
harness debugging, and a run made with it is not publishable.

### The recorded commit describes the DEFAULT binary only

`engine_commit` claims the checkout's HEAD for `build/cbls_mipfeas` and for
nothing else. Point `--cbls-bin` somewhere else — as an A/B between two builds
must — and the rows are attributed to that binary's own content hash (`bin:<12
hex>`) instead, because HEAD is not evidence about a binary it did not build
(#158). Pass
`--engine-commit <sha>` when you do know which commit built it.

## Verification: every reported solution is checked against the instance file

This is the correctness benchmark, so a reported solution is not taken on trust.
Both runners write the solution vector of any run they report feasible, and
`verify_solution.py` re-reads the **original** `.mps.gz` and checks that point
against it: row activities, variable bounds, integrality, and that the published
objective is the objective *of that point*.

The check is deliberately independent of both engines. It reads the instance
with **SCIP** (PySCIPOpt), which neither engine uses: CBLS reads MPS with
`src/io/mps_reader.cpp` and CP-SAT with OR-Tools' own reader, so verifying
either with its own reader would let a shared defect cancel out. This matters
because every *other* check in the harness is downstream of the engine's reader —
CBLS's residual, integrality and objective-drift checks all run against the model
the MPS-to-model adapter built, so a misread bound, a dropped row or a flipped
sign is invisible to all of them. The `.solu` reference values catch only the
other direction, an objective that is *too good*.

### Tolerances, and what "borderline" means

All four are constants of `verify_solution.py`, and every verdict file records
the set it was reached under (the comparison table quotes it back). Each is
`absolute + relative * scale`, because MIPLIB coefficient ranges span many orders
of magnitude and a purely absolute rule is either vacuous on a large row or
unmeetable on a small one:

| check | violation | tolerance |
|---|---|---|
| row activity | `max(0, lower - a.x, a.x - upper)` | `1e-6 + 1e-9 * row_scale` |
| variable bound | `max(0, lower - x, x - upper)` | `1e-6 + 1e-9 * \|bound\|` |
| integrality | `\|x - round(x)\|` | `1e-6` |
| objective | `\|reported - (c.x + offset)\|` | `1e-6 + 1e-6 * \|objective\|` |

`row_scale` is `max(\|lower\|, \|upper\|, sum \|a_ij x_j\|)` over the finite sides — the
sum of absolute terms rather than the activity, so a row whose terms cancel is
still judged against the magnitudes actually added up.

Each absolute term is the engine's own stated feasibility tolerance, which keeps
the two checks *comparable* without sharing anything. The objective's relative
term is likewise the runner's own drift gate, `1e-6 * (|objective| + 1)` — this
check re-measures exactly the quantity that gate accepts, so a tighter rule
downstream of it would reject runs the engine was entitled to publish rather than
being more independent. The row and bound relative terms are `1e-9`, ~7 orders
above double round-off; activities here are summed with `math.fsum`, which is
exact, so that margin covers the *engine's* accumulation and not the checker's.

Integrality is judged at the point as written, and the rows are evaluated on
those same unrounded values — a point cannot buy row feasibility by being rounded
to integers after the fact.

**The rule is the threshold, and nothing else.** A solution fails if and only if
some violation exceeds its tolerance; at or below it, it passes. There is no
judgement call at scoring time. A pass whose worst violation reaches 10% of its
tolerance is published as `pass` with `verification_marginal=1` — a signal to
look, never a reason to withhold. (CBLS routinely lands ~1e-9 inside a row, which
is 0.001x the tolerance and not marginal.)

### What a failure does

A row without a `pass` publishes **no objective and no Primal Integral**, and is
left out of the aggregates rather than scored 2.0 — the run did find a point, it
was rejected, and scoring it would be publishing a derived number of its own. The
`verification` and `verification_reason` columns say which rows and why, the
scorer prints a `DEFECT:` line naming the counts, and both the driver and the
scorer exit non-zero. Three verdicts exist: `pass`, `fail` (checked and wrong)
and `error` (could not check — a missing solution file, an unreadable instance, a
constraint type the checker does not model). Only `pass` publishes.

`--no-verify` on the driver turns the whole thing off, and `--allow-unverified`
on the scorer publishes rows that carry no verdict. Both are for harness
debugging and for results directories filled before this existed; a table scored
with either is not publishable.

### What it holds on real instances

Swept over the vendored `benchmarks/instances/miplib-fj/` set — 11 real MIPLIB
2017 instances, both engines, short budgets — **19 solutions, all `pass`**, and
none close to a threshold. The worst row violation observed is `1.7e-10`
(9e-5 of its tolerance, CBLS on `pk1`) and the worst objective mismatch `5.2e-09`
(7e-7 of its tolerance, CP-SAT on `binkar10_1`). So the tolerances are nowhere
near tight enough to manufacture a false defect, and CP-SAT's integer scaling of
continuous columns — the one plausible source of systematic *baseline* failures —
does not produce violations at this scale. Reproduce by pointing either runner's
`--inst-dir` at that directory and running `verify_solution.py` over the results;
`tests/python/test_verify_solution.py` pins the `pk1` case in the suite.

### Two costs worth knowing

Verification reads each instance a second time and sums every nonzero in Python,
inside the job slot and under the same memory cap as the solve. It checks one row
at a time and never holds the matrix: PySCIPOpt builds a fresh `str` key per
nonzero, measured at ~245 bytes each, so retaining it would cost ~6.7 GB on
`square47`'s 27.4M nonzeros — more than a `--mem-limit-gb 6` cap. Peak is
SCIP's own model plus a single row. The `peak_rss_kib` figures recorded per
result are the **solve's**, so they do not bound a job's peak on their own.

And a resumed run whose results predate the solution dump re-**solves** those
instances: nothing but the search can produce the solution vector. A verification
the driver had to kill (its timeout or the memory cap) is retried on the next
resume rather than being treated as a final verdict — once, for two attempts in
all, after which the verdict stands and the row stays withheld.

A row in that state is counted and the driver **exits non-zero** on it, over every
planned job rather than only the ones an invocation ran. Otherwise the second
resume of such a directory prints "0 jobs to run" and exits 0, reporting as a
clean run a row that was never successfully checked and whose objective the
scorer withholds.

## Running it

```bash
# 1. Fetch the roster (~546 MiB via benchmark.zip; --subset smoke for the 11)
.venv/bin/python benchmarks/instances/mipfeas/download.py

# 1b. Check the pinned bytes at any later point (offline, no download)
.venv/bin/python benchmarks/instances/mipfeas/download.py --verify

# 2. Build the CBLS runner
cmake -B build && cmake --build build -j$(nproc) --target cbls_mipfeas

# 3. Install the baseline (OR-Tools) and the verifier's reader (PySCIPOpt)
uv sync --extra dev --extra benchmarks
.venv/bin/python benchmarks/mipfeas/cpsat_solve.py --preflight   # ~2s, no instance, no network

# 4. Wiring check: 11 instances, both engines, short budget. CP-SAT at its
#    defaults takes the machine's cores, so the driver runs one job at a time.
.venv/bin/python benchmarks/mipfeas/run_benchmark.py --roster smoke --budget 60 --jobs 1

# 5. The full roster: 233 x 60s x 2 engines, one job at a time, ~8 wall-hours.
#    #255 ran the 211 outside tuned.txt at 60s: cbls at seed 1 on 16 threads
#    pinned to cores 0-15, CP-SAT unpinned at its defaults (the runner refuses
#    --cpuset with CP-SAT at its defaults, so the two halves ran separately).
#    It scored them with --held-out tuned.txt in step 6; its table waits on #253.
.venv/bin/python benchmarks/mipfeas/run_benchmark.py --roster full --budget 60 --jobs 1

# 6. Score it — into the results directory, beside what it scores
.venv/bin/python benchmarks/mipfeas/primal_integral.py \
    --results-dir results/mipfeas --roster benchmarks/instances/mipfeas/roster.csv \
    --budget 60 --out results/mipfeas/comparison.csv
```

Step 6 writes `comparison_report.md` beside the table; see **What the scorer
reports** above. It exits non-zero when a solution was rejected.

`--cpsat-ls-results-dir DIR` also writes `comparison_three_arm.md` (#321): cbls
and CP-SAT at its defaults from `--results-dir`, CP-SAT `fj` + `ls` from `DIR`'s
`cpsat/`, whose rows also get their own `comparison_fj_ls.csv` and
`comparison_fj_ls_report.md` against cbls. It refuses an incomplete arm, a row with
no usable verdict, `--allow-unverified`, arms that are not in the modes they are
labelled, fj + ls workers other than cbls's thread count, different seeds, and two
OR-Tools builds. It exits non-zero when the fj + ls arm has a rejected solution, a
shape disagreement with cbls, or an objective below a proven optimum.

A roster short of the 233 is stamped a wiring check. A table that leaves out
the instances settings were tuned on is not one: score the full `roster.csv`
with `--held-out benchmarks/instances/mipfeas/tuned.txt`, and the scorer drops
those names itself, scores the rest, and labels the table and report a held-out
roster naming what it left out (#255). It refuses the flag unless `--roster` has
all 233 instances and every held-out name is among them, so the scored set is
derived rather than hand-made.

**The table is an output, not an artifact to protect.** It is regenerable from a
results directory at any time and at any budget, so there is nothing in it to
defend and no reason to copy it into this directory as a checked-in record. The
numbers that *are* derived once and pinned live in `roster.csv` and
`references.csv`: the per-instance optimum or best-known value, taken from
MIPLIB's own `.solu` file, with `references.csv` recording the hash and byte
count of the file they came from so a MIPLIB revision cannot silently redefine
them. That is the only thing here with a provenance worth freezing; a comparison
between engines is something to re-derive from result records whenever the
question is asked.

The driver is resumable: a job whose result file exists is skipped, so an
interrupted run continues where it stopped.

### Comparing CBLS flag sets (#331)

#331 compared flag-set arms (presolve, compound propagation, both) with
`run_arms.py`, which ran each solve through `run_benchmark.py` under the
machine lock. The campaign is closed and the script is in git history:
`git show 047f2d4c:benchmarks/mipfeas/run_arms.py`.

### Running the portfolio (`--cbls-threads`)

At `--cbls-threads 1` CBLS takes the single-threaded engine path, which every
published figure here was measured on. Above 1 it runs `ParallelSearch`, the
cooperative portfolio: N workers, each on a replica that shares the model's
immutable DAG by reference (#157) and duplicates only what a search writes,
sharing incumbents through the solution pool and restarting from it when they
stall. Three things follow; the harness checks two, and the third is sizing
that stays the operator's:

* **In the fj+ls mode, the baseline gets the same number.** CP-SAT at
  `num_workers: N` runs N `fj` and N `ls` subsolvers, so an N-thread CBLS
  against a 1-worker CP-SAT is an N-fold CPU advantage the table would report as
  an implementation gap. The driver refuses `--cbls-threads != --cpsat-workers`
  unless `--allow-asymmetric-cpu` says the asymmetry is the measurement. At
  CP-SAT's defaults (`--cpsat-workers 0`) it picks its own count and nothing is
  matched: the table records the threads each side used (#255, #285).
* **Memory still grows with the thread count, and nothing checks it.** Size a
  run as **1-worker + N x a per-worker constant**: the `peak_rss_kib` figures
  below are the single-threaded term, and the constant is a floor rather than a
  bound, its largest term being the variable array at 104 B per column. On
  `neos-5114902-kasavu` (710k columns, 4.30M DAG nodes) it measures **0.34-0.39
  GiB** at `06eb3e5`; `supportcase19` has 2x the columns. `--mem-limit-gb` is a
  per-JOB `ulimit -v` and knows nothing about `--cbls-threads`, so `--jobs` x
  `--cbls-threads` against that cap is arithmetic to do before starting.
  A run out of memory is reported, not silent. All N replicas are built before
  the solve bracket opens, so a failed replication writes the row with
  `status: replicate_error` (on the worker threads `ParallelSearch` would park
  the `bad_alloc` and return the survivors' result). The rest of a worker's
  footprint (FJ tables, violation vectors, snapshots) is allocated on its own
  thread, so a cap can fit every replica and still starve a worker once it
  searches (#170): every row that reached the end of the solve carries
  `workers_launched`, `workers_completed` and `worker_failures` (index, whether
  it had produced a result, the reason); a row with `workers_completed <
  threads` makes the runner exit 1, and the scorer **refuses** it, as it refuses
  a multi-threaded row without the column. A worker is *completed* unless an
  exception ended it or the deadline was gone before its first solve; one that
  never started because a peer had already solved a pure-feasibility model
  counts as completed. A resume re-runs a refused row up to twice
  (`MAX_WORKER_LOSS_ATTEMPTS`, stamped as `worker_loss_attempts`), and the
  driver exits 1 with a `LOST WORKERS` line while any remains. A loss that
  survives both is the cap: raise `--mem-limit-gb` or lower `--cbls-threads`,
  then re-run with `--force`.
* **The two arms are separate results directories.** `threads` is a scorer
  config key, so a directory mixing thread counts is refused, and the driver
  refuses to *resume* a directory recorded at another thread count (resume
  skips jobs that already have a result, so the scorer would never see the
  mixture).

"N threads each" is a flag until it is measured (#240). `--cpuset 0,2,4,6` pins
every solve to those CPUs with `taskset`; the driver refuses a set whose size is
not the thread count, two SMT siblings of one core (read from
`/sys/devices/system/cpu/cpu*/topology/thread_siblings_list`), a CPU outside its
own affinity, and `--jobs` above 1. The set is recorded in `run_record.json`.
Pinned or not, every result carries a nested `driver_measured` key, read by the
driver with `wait4`: `wall_seconds` (the driver's clock, model read and build
included), `cpu_seconds` (user plus system, every process it reaped),
`cpu_wall_ratio` (the cores the solve actually used) and `peak_rss_kib`. It is
nested, like `driver_arm`, so the scorer never reads it as configuration: it
copies three of its numbers into `comparison.csv` (**Where the time went**) and
compares none of them across rows.

Replication is charged to setup, not search: the runner freezes the built model
(the objective row and the cone cache, #271, at every thread count) and copies
it once per worker before the solve bracket opens, reporting
`replicate_seconds` beside `read_seconds` and `build_seconds`. It grows slowly
with `--cbls-threads` and stays outside `wall_seconds`; a 1-thread row's figure
is the freeze alone.

The anytime trace is the portfolio's own: every worker reports through one
serialized stream on the portfolio clock, with the global incumbent as its
objective, so the profile the Primal Integral integrates is monotone and ends
at the objective the run returns.

At CP-SAT's defaults `--jobs` is 1 (it takes every core, so the driver refuses
more). In the fj+ls mode, or a CBLS-only run, sizing `--jobs` is a memory
question rather than a core-count one. Every result
records its `peak_rss_kib` — of the **solve only**; verification runs afterwards
in the same job slot (see above). `neos-5114902-kasavu` (710k columns, 961k
rows, 4.9M nonzeros) measures **0.78 GiB single-threaded under CBLS at
`06eb3e5`** (the wiring check's 1.2 GB predates #156's flat arrays) and peaked
at **3.2 GB under CP-SAT**, which carries a ~100 MB floor on even the smallest
models. Re-derive either number before multiplying it.

It is not the roster's largest model: `square47` carries 27.4M nonzeros (5.6x
kasavu) and `supportcase19` 1.43M columns (2x). Model build and search
initialisation are not deadlined (read `setup_seconds` in a recent run), so the
driver allows 900s of slack on top of the budget. `--mem-limit-gb` caps
*address space* (`ulimit -v`), not resident set: CBLS runs about 1.4x its RSS;
CP-SAT adds a roughly constant ~0.7 GB of reservations, so its ratio is ~1.4x on
a large model like `neos-5114902-kasavu` but ~4.4x on a small one like
`atlanta-ip`. Confirm the cap on the target machine rather than reading it off
the RSS figures above.

## What the wiring check found

`smoke_comparison.csv` is 11 instances at 60s, with CP-SAT in the fj+ls mode —
enough to prove the harness end to end, and **not** a result. It predates the
verification columns and cannot be regenerated (its results directory was never
committed and its rows carry no independent verdict); a fresh wiring check is
the only way to a current table. What *is* pinned is the report itself:
`benchmarks/mipfeas/testdata/` holds a frozen results directory with the two
artifacts scored from it, and `tests/python/test_mipfeas_parity.py` regenerates
both and compares them byte for byte. Two of its thirteen instances are a real
run, the rest constructed (that directory's README says which); no aggregate in
it is a measurement.

The check settled the two configuration choices above by measurement. Each was
run over the whole smoke roster, CBLS only:

| Novelty Jump | clamp | shifted geomean | mean | feasible |
|---|---|---|---|---|
| off | 1e9 | 0.654 | 1.109 | 7/11 |
| on | 1e9 | 0.614 | 0.961 | 8/11 |
| off | 1e7 | 0.638 | 1.089 | 7/11 |
| **on** | **1e7** | **0.561** | **0.937** | **8/11** |

Only the bottom row is reproducible from what is committed, and a time-limited
`solve()` is not bit-reproducible even at a fixed seed (the engine is
deterministic only when bounded by iterations, `time_limit = 0`), so a re-run
moves the aggregate in the third decimal. **Both clamp rows predate #120**:
implied bounds now clear the clamp entirely on 5 of these 11 instances
(`binkar10_1`, where the 1e9-vs-1e7 difference was largest, among them); it
still binds on `mad`, `pk1`, `gen-ip002` and `gen-ip054`, and `mas76` and
`neos5` had no clamped column. Whether any of it holds over the roster is what
the full run is for (#255; its table waits on #253).

## Choices tuned on the smoke roster (#233)

All 11 instances in `smoke.csv` — `enlight_hard`, `markshare2`, `gen-ip054`,
`gen-ip002`, `pk1`, `mas76`, `neos5`, `mad`, `binkar10_1`, `atlanta-ip` and
`neos-5114902-kasavu` — are also in `roster.csv`. Several defaults were chosen
by measurement on exactly those 11, so on them a row is in-sample:

| Choice | Decided by | Shipped in |
|---|---|---|
| Novelty Jump on, and the `1e7` clamp | the A/B table under "What the wiring check found" | the runner's flags |
| FJ-or-Novelty redrawn every batch | #209's pre-registered A/B, 10 seeds at 30s | `143c8da` |
| objective tie-break on | #226 Run A, 5 seeds at 10s; gap at first feasible 0.748 | `9e1cce9` (#220) |
| near-value kick off | #226 Run B on MINLPLib; Run A on this roster passed it | default kept since #217 (#219) |

#209 and #226 carry the numbers. The `xk tune` study (#264) then tuned on 11 more
roster instances, and it confirmed no setting, so no default changed. #331 then
decided `--presolve` and `--compound-propagation` on all 22, in a pre-registered
four-arm run at 60s with 3 seeds: pooled PI ratios 0.934 (presolve), 1.142
(propagation) and 0.993 (both), all inside the control's 1.30 noise floor, so
both stayed off. Presolve was then removed (#379; in git history at `b10da9d2`),
and so was compound propagation (#380; in git history at `57cdbfba`).
`tuned.txt` lists all 22. A table scored with `--held-out tuned.txt` reports the
other 211, none of them tuned on. One scored over the whole roster names the 22:
the effect on a roster-wide aggregate is small, but it is not zero.

Every CBLS result record carries the tie-break's state as `objective_tie_break`
(a record from before `6492993`, #218, has no such key, and one from before
`9e1cce9` either lacks it or carries `false`; both ran it off), and the scorer
puts `objective_tie_break=True` into the configuration cell, so an on row never
scores together with an off one. A published table names its engine commit and
states whether the tie-break was on.

## Configuration, and what is recorded

CBLS, and CP-SAT in the fj+ls mode, run at a stated configuration, recorded per
result rather than inherited, so a published number cannot silently change when
a default moves. CP-SAT at its defaults inherits them on purpose: there the
recorded OR-Tools version, parameter string and `workers_started` are what pin
the run:

| | CBLS | CP-SAT |
|---|---|---|
| Threads | 1 (`--cbls-threads`, > 1 runs the cooperative portfolio) | CP-SAT's own count (`--cpsat-workers 0`, the default; recorded as `workers_started`), or N in the fj+ls mode (`--cpsat-workers N` -> `num_workers`) |
| Algorithm | Feasibility Jump + ViolationLS + Novelty Jump | its full default portfolio; in the fj+ls mode `fj` + `ls` workers only (`filter_subsolvers`) |
| Presolve | implied variable bounds only (activity-based propagation); the opt-in `--presolve` (#325, kept off by #331's measurement) was removed (#379; in git history at `b10da9d2`) | default, i.e. on |
| Propagation during search | none; the opt-in `--compound-propagation` (#340, kept off by #331's measurement) was removed (#380; in git history at `57cdbfba`) | its own, as part of its default portfolio |
| Feasibility tolerance | `1e-6`, stated explicitly | CP-SAT's own |
| Unbounded column falls back to | `1e7` (`--inf-clamp`), where propagation derives nothing | not clamped |
| Recorded per result | commit SHA, seed, tolerance, clamp + columns it still narrows, columns declared unbounded, columns tightened, propagation verdict and pass cap, compound-move, exact-Int-jump (#216, until #381 removed it; in git history at `1330cd2c`), near-value-perturbation (#217, off by default), objective tie-break (#218, on by default since #226), propagation flags, the point's own check against the instance file (`original_*` ratios), peak RSS, read/build/setup seconds, trace point count and trace source | OR-Tools version, seed, full parameter string, workers started (at its defaults), solver verdict, peak RSS, read/setup seconds, free-row count, trace source |
| Solution verified against the instance file | yes, by SCIP | yes, by SCIP |

Two of those are deliberate choices, made to keep the two sides comparable
rather than to flatter either:

* **Novelty Jump is on**, stated explicitly by the runner (it is also the
  engine default). Roughly half of
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

Activity-based propagation over the linear rows derives bounds that the
constraints actually imply, and the fallback is consulted only on what is left.
Two rules hold together:

* An implied bound is used wherever one exists. Unlike the substitute, it cannot
  put a feasible point outside the box.
* A bound that *exists* — declared in the file or derived — is honoured however
  wide. Narrowing a declared bound has the same defect as inventing one; across
  the 233 instances it reached 110 columns on 6 of them, so stopping gives up
  nothing measurable.

Measured over all 233 roster instances at `aec705b`, model build only (no
search); "before" counts columns with no bound at all (the `1e20`
MPS/CPLEX/SCIP sentinel):

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
10-pass cap, at 0.31s. Roughly 7s summed over the whole roster, against a
per-instance budget of 60s in #255.

**Soundness is argued and tested, not verified against known optima.** The
`.solu` file carries objective values, not solution vectors, so "the optimum is
still inside the box" cannot be checked per instance — an excluded optimum
would read as search weakness, not as infeasibility. What is checked: a unit
test where a fixed clamp excludes the only solution and propagation keeps it,
and 233/233 instances propagating without a single false infeasibility.

`--no-propagate-bounds` turns off propagation only — the second rule, honouring
a finite bound however wide, is unconditional — so it is an A/B on
propagation rather than a return to the pre-#120 engine. It exists for A/B work;
it is not a configuration to publish, because the bounds it restores are not
implied by the constraints.

Two configuration facts are worth knowing before changing anything here, both
established empirically against ortools 9.15 and both re-checked by
`cpsat_solve.py --preflight` (see **Preconditions** above) rather than assumed:

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
timestamps include its presolve. It favours CBLS, by a share of the budget that
grows as the budget shrinks. Since #139 both runners
measure it and the table carries it as `setup_seconds`, so it is a number a
reader can check per row rather than a caveat taken on trust.

### Restart versus kick: an algorithmic difference (#234)

Besides the setup-time asymmetry above, a second difference is in the algorithm itself, so "the same algorithm" holds
for the jump and the weighting, not for what happens on stagnation (OR-Tools
cited at
[`4336f9f`](https://github.com/google/or-tools/tree/4336f9f3aeedd5462ee0008116b1a2a1e7d633ef),
`ortools/sat/feasibility_jump.{h,cc}`):

* **CP-SAT restarts.** Each subsolver interleaves a set of LS states (8 to
  start, one more only when all are taken). At a batch boundary after a new
  best, or when a state's batch allowance runs out, the state re-randomises its
  options (`LsOptions::Randomize`: among them decay, compound moves, objective
  use and the perturbation probability) and resets its weights. A restart state
  then restarts on a shared Luby schedule: `ls` from a pool solution, `fj` by
  `ResetCurrentSolution`, which sets each column to its smallest value or, at
  the perturbation probability, near zero, and, when the state uses the
  objective, objective columns to the bound their coefficient favours. Only a no-restart state perturbs near the
  current value (`PerturbateCurrentSolution`), and the first one is state 15
  (`use_restart = index % 16 != 15`), so in the fj+ls mode, at fewer than 16
  workers on one state set, the CP-SAT arm never takes that path. At its
  defaults CP-SAT picks its own worker count and portfolio, so it may.
* **CBLS kicks.** One search state, fixed options, no reset to zero or the bound,
  and by default no Luby schedule. On stagnation it perturbs from `kick_origin()`: the
  incumbent once one exists, an adopted pool solution in a portfolio, in place
  before the first feasible point. Portfolio workers restart from the pool.
  `luby_restart_unit` and `restart_randomization` are default-off arms, and the
  runner takes `--luby-restart-unit` and `--restart-randomization`; neither
  separated in #239's smoke A/B.

## Provenance

- Instances and reference values: MIPLIB 2017, <https://miplib.zib.de/>
  (`benchmark-v2.test`, `miplib2017-v36.solu`, `benchmark.zip`).
- Metric and roster definition: the MIPfeas announcement linked above.
- Baseline: OR-Tools CP-SAT (Apache-2.0), version recorded per result.
- Verification reader: SCIP via PySCIPOpt (Apache-2.0), version recorded in every
  `.verify.json`. Used only to re-read the instance and check a solution against
  it; it never solves anything here.
