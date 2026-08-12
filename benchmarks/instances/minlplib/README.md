# MINLPLib non-convex benchmark subset

External yardstick for the CBLS continuous machinery (reverse-mode AD on a
transcendental DAG + Newton jump values). Where MIPLIB-FJ (`../miplib-fj/`)
exercises the pure-linear search core, this subset targets **non-convex
mixed-integer non-linear** instances — the regime where standalone CBLS
competitors (Yuck, fzn-oscar-cbls) cannot even *encode* most cases, because
FlatZinc has no transcendental float constraints. Coverage is the headline
metric here; gap-to-BKS is secondary (we are a primal heuristic).

Source: [MINLPLib](https://www.minlplib.org/). Metadata and bounds come from
the published catalogue CSV:

    https://www.minlplib.org/instancedata.csv   (semicolon-separated)

Each instance's text NL file is fetched individually from
`https://www.minlplib.org/nl/<name>.nl` (we do **not** pull the multi-hundred-MB
archive). The CBLS NL reader (`src/io/nl_reader.cpp`) handles the **text** ('g'
header) format; instances served only as **binary** NL ('b' header) are rejected
by the downloader and excluded (e.g. the `kriging_peaks-*` family).

## Selection method (CSV-driven, reproducible)

`download.py` applies a metadata-only filter — no NL parsing needed for
selection — then stratifies:

1. `convex == False` (non-convex only).
2. `probtype` in {NLP, MINLP, QCP, QCQP, QP, MIQCP, MIQCQP, MIQP, BQP, BQCP}.
3. The instance advertises the `nl` format.
4. **Operator subset check**: no operator column outside the set CBLS can
   express (see table below) is flagged `True`.
5. Size budget: `nvars <= 150` and `ncons <= 150`.
6. A finite `primalbound` (so gap-to-BKS is defined).
7. Stratify the survivors round-robin across structure classes
   (bilinear / polynomial / transcendental / mixed-integer / other), smallest
   first.

The downloader then walks that stratified order fetching `.nl` files until
`--limit` instances are **successfully fetched**, so an instance the catalogue
advertises as `nl` but serves as binary NL (the `kriging_peaks-*` family) is
replaced from the candidate pool rather than shrinking the roster.
`bounds.csv` is written from the fetched set only, so the roster the runner
reads is exactly the set of `.nl` files on disk.

Reproduce with the project venv:

    .venv/bin/python3 benchmarks/instances/minlplib/download.py --select-only
    .venv/bin/python3 benchmarks/instances/minlplib/download.py --limit 50

The downloader validates every fetched body (rejects HTML/404 and binary-NL
headers) and prints a sha256 for provenance.

`bounds.csv` columns: `instance,structure,nvars,ncons,objsense,primal_bks,
dual_bound,n_disc_vars_bks`. The trailing `n_disc_vars_bks` is the catalogue's
`nbinvars + nintvars`; the runner cross-checks it against the integrality the
NL reader recovers and reports any mismatch (see below).

## Operator support (CBLS DAG ↔ MINLPLib op columns)

| MINLPLib op column | CBLS DAG op            | Supported |
|--------------------|------------------------|-----------|
| `opmul`            | `Prod`                 | yes       |
| `opdiv`            | `Div`                  | yes       |
| `oppower` / `opsqr`| `Pow`                  | yes       |
| `opsqrt`           | `Sqrt`                 | yes       |
| `opabs`            | `Abs`                  | yes       |
| `opexp`            | `Exp`                  | yes       |
| `oplog` / `oplog10`| `Log` (+ scale)        | yes       |
| `opsin` / `opcos`  | `Sin` / `Cos`          | yes       |
| `opmin`            | `Min` / `Max`          | yes       |
| `opsignpower` / `oprpower` | `Pow` (NL emits these as standard `OPPOW`; the `SignPower` DAG op added in #72 is available for direct model building / JSONL) | yes |
| `optanh`           | `Tanh` (added #72)     | yes       |
| `opcvpower` / `opvcpower` | —               | no (skipped) |
| `operrorf` (erf)   | —                      | no (skipped) |
| `opgamma`          | —                      | no (skipped) |
| `opcentropy`       | —                      | no (skipped) |
| `opmod`            | —                      | no (skipped) |

Piecewise (`OPPLTERM`), function calls (`OPFUNCALL`), and inverse-trig opcodes
are parsed structurally but rejected by the adapter with a skip reason; the
runner records these as `skipped(unsupported)`.

The reader also rejects NL `V` (defined-variable / common-subexpression), `S`
(suffix), `F` (function), and `d` (dual) segments — instances using them are
reported as `skipped(unsupported)`. The curated roster avoids these; supporting
`V` (inlining defined variables) is the natural next step to widen coverage.

## Yuck / fzn-oscar-cbls coverage

These FlatZinc-based local-search solvers are the natural CBLS comparators, but
FlatZinc's float layer has no `exp`/`log`/`sin`/`signpower` constraints, so the
**transcendental and signpower instances in this roster are not expressible**
for them at all. That non-expressibility is the differentiator this benchmark
documents. No published Yuck numbers exist for these instances.

## Provenance

- Instance data, primal/dual bounds: MINLPLib, https://www.minlplib.org/ (CSV
  above). MINLPLib is a curated public benchmark library; bounds are from
  BARON/SCIP/ANTIGONE runs reported there.
- NL format: David M. Gay, *Writing .nl Files*
  (https://ampl.github.io/nlwrite.pdf) and *Hooking Your Solver to AMPL*
  (https://ampl.com/REFS/hooking2.pdf). The CBLS NL reader and opcode table are
  an original implementation from those public specs (opcode numbers cross-
  checked against the ASL `opcode.hd`); no third-party source is vendored.

## Files

- `download.py` — CSV-driven selection + per-instance `.nl` fetch + validation.
- `bounds.csv` — fetched roster with published primal/dual bounds and the
  catalogue integer-variable count.
- `comparison.csv` — written by the `cbls_minlplib` runner: CBLS objective,
  gap-to-BKS, gap-to-dual, feasibility, notes, commit SHA, closest-approach
  residual (`max_violation`) and integer-variable count (`n_int_vars`).
- `analysis_notes.csv` — curated per-instance root-cause verdicts
  (`bug` vs `hard`) for instances the runner cannot solve. Merged into
  `comparison.csv`'s note column, so the data carries its own explanation.
- `anytime_trace.csv` — incumbent objective against wall time for the published
  run (`instance,time_seconds,objective,new_best`), written by `--trace`. The
  `objective` column is the internally *minimised* value, so a maximize instance
  appears negated relative to `comparison.csv`.
- `scip_baseline.csv` — written by `../../minlplib/reference_solve.py`: the SCIP
  baseline's objective, gaps against the same published bounds, feasibility,
  wall time, plus the dual bound, gap and status only a complete solver
  produces, and the exact `SCIP x / PySCIPOpt y` pair per row.
- `comparison_all.csv` — the three-way comparison in long format, one row per
  `(instance, method)` with `method` in `published-bks` / `cbls` / `scip`. Also
  written by `reference_solve.py`, by joining the two CSVs above with
  `bounds.csv`.
- `*.nl` — fetched text NL instance files.

Regenerate `comparison.csv` with:

    cmake --build build -j$(nproc)
    ./build/cbls_minlplib --time-limit 60 --seed 1 --commit "$(git rev-parse --short HEAD)"

Regenerate `scip_baseline.csv` and `comparison_all.csv` with (needs the
`benchmarks` extra — `pip install -e '.[benchmarks]'`):

    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --time-limit 60

Run the CBLS side first: the merge reads whatever `comparison.csv` holds. To
rebuild only the merge after a fresh CBLS run, add `--merge-only`.

## Results

Latest run: **60s per instance, seed 1, feasibility tolerance 1e-6**, commit
recorded per row in `comparison.csv`. The tally below, the gap buckets and the
anytime profile all come from that **one** run; its incumbent trace is committed
as `anytime_trace.csv`, so every number in this section is reproducible from a
checkout without re-running anything.

**These are single-sample numbers.** The budget is wall-clock, so a fixed seed
does not pin the iteration count and consecutive runs of the same binary differ:
two runs at one earlier commit and seed gave 46 and 45 feasible, and an
independent replication moved two gap values materially (`nvs05` 453%→477%).
The spread is wider than that on some rows: re-running the *unmodified* binary
at the same seed and budget moved `kall_ellipsoids_tc02b` from 55.1% to 78.2%,
and `eq6_1` spans 7.6–28.7% across four seeds. Treat any single row as one draw,
not a measurement. Reporting a median over
several seeds is the fix; it is not done here, and the runner has no flag for a
deterministic budget yet (the engine supports one — `time_limit = 0` plus
`SearchConfig::max_iterations` — but `cbls_minlplib` requires `--time-limit > 0`).

| | count |
|---|---|
| roster | 50 |
| parsed and built (closed-model rate) | 50 (100%) |
| of which mixed-integer (integrality enforced) | 15 |
| **feasible** | **46** |
| — matching BKS (within the tie band) | 18 |
| — better than BKS, but inside the tolerance slack | 1 |
| — worse than BKS | 27 |
| — better than BKS | 0 |
| infeasible | 4 |
| unsupported / read errors / non-finite | 0 |
| integrality mismatches vs catalogue | 0 |
| verification failures | 0 |

Gap distribution over the feasible instances: **21 within 0.01% of BKS, 22
within 1%, 26 within 10%.**

Five rows have a numerically zero BKS (`|BKS| < 1e-12`), for which the runner
writes an *absolute* residual into the `gap_to_bks%` column rather than a
meaningless percentage against zero: `mathopt1`, `prob09`, `least`, `ex14_2_4`
and `ex14_2_5`. Those values are not percentages. The buckets above exclude the
first three, whose residual is non-zero, and retain `ex14_2_4`/`ex14_2_5`, where
objective and BKS are both exactly 0 and so are exact matches at any threshold.
Excluding all five instead gives 19 / 20 / 24 over 41 rows. Counting the
excluded three *as* percentages would have put `mathopt1` — objective 1.0
against a BKS of 3.3e-18 — inside the "within 1%" bucket.

Nothing in this roster beats a published bound. Under the runner's earlier
margin rule — which compared a *percentage* against 1e-6, i.e. 1e-8 relative —
two rows of this run would have been flagged `better-than-bks`: `ex6_2_6` at
8.3e-5 percent and `prob06` at 3.2e-4 percent. Those are ties, not improvements.

Two bands are used, deliberately different. An improvement is only *claimed*
when it exceeds `max(1e-6·(|BKS|+1), 10·feas_tol)`: we accept solutions
violating a constraint by up to `feas_tol`, and that slack itself buys a small
objective gain. A *tie* requires the much tighter, purely relative
`1e-6·(|BKS|+1)` — using one band for both would have published `ex8_4_5`
(BKS 3.07e-4) as matching BKS when it was 1.38% worse, because the absolute
floor dwarfs an objective that small. A row that improves on BKS by more than
the tie band but less than the claim threshold falls between the two and is
labelled `within-tolerance-of-bks` rather than being miscounted as worse.

### Why 60s

Measured from the committed trace, not assumed. Cumulative instances with a
feasible solution by time t (of 50):

| by | 1s | 5s | 10s | 20s | 30s | 45s | 60s |
|----|----|----|-----|-----|-----|-----|-----|
| feasible | 41 | 41 | 41 | 42 | 44 | 46 | 46 |

**This is the load-bearing argument.** Five instances reach feasibility only
long after 5s — `chain50` (17.6s), `ex8_4_5` (24.9s), `tln2` (26.9s), `spring`
(31.4s), `minlphi` (36.8s) — so a 5s budget would publish all five as
infeasible, 41 solved instead of 46. Which five varies between draws; that
several exist does not.

Solution *quality* over time is a weaker argument than it first appears, and is
recorded here with that caveat. Of the 46 instances that become feasible, 46%
stop improving within the first second while 22% are still improving in the
final 15 seconds. But the incumbent trace cannot be read as pure search
progress: `record_best` tightens the objective bound by `1e-3·(|obj|+1)` per
accepted solution, so improvements are *floored* at roughly 0.1% steps. The
measured median consecutive-incumbent ratio on `eg_all_s` is 0.9989993 — exactly
`1 - 1e-3` — and it takes 15931 such steps to walk from 1e9 down to 8.46. That
instance is therefore evidence about the bound-tightening step size, not about
how long the search needs. Read the quality column as a lower bound on what a
larger step (or a direct objective descent) might achieve sooner.

### The four unsolved instances

Every one is root-caused, and the verdict is recorded per row in
`comparison.csv` (merged from `analysis_notes.csv`):

| Instance | Verdict | Cause |
|----------|---------|-------|
| `elec25`, `elec50` | **bug** ([#110](https://github.com/spoorendonk/cbls/issues/110)) | Thomson problem: points on the unit sphere, Coulomb objective `+inf` wherever two coincide. **The objective-encoding defects (#100) are fixed and are no longer the blocker.** What remains: the `.nl` declares no finite variable bounds, so the box is the ±1e9 inf-clamp; random init starts ~1e9 out, and shrinking each variable toward 0 is a huge row improvement — which parks the search on the origin, a stationary point of every row `x²+y²+z²=1`. The Float jump offers a single *undamped* Newton step (`x0 - residual/grad`) plus `lb`/`ub`/midpoint; near the origin that step overshoots wildly and is rejected, and because a candidate was nonetheless *generated* the #107 escape probe is suppressed — so the variable freezes at score 0. Measured: escape probe fires only at exactly `x0 = 0`; at `x0 = 0.001/0.01/0.1` the score is 0 with the probe armed or not. Infeasible at violation ≈1 **both with the objective present and with it neutralised**, so it is not objective-related — the earlier "dropping the objective makes elec25 feasible in 20s" claim no longer reproduces. Tightening `inf_clamp` to 1 makes `elec25` feasible at violation 0 post-#100 (pre-#100 it was infeasible at *every* clamp), because clamping accidentally supplies the missing damping. |
| `nvs01` ([#101](https://github.com/spoorendonk/cbls/issues/101)) | hard | `420.169·√(x0²+900) == x2·x0·x1` needs `x0` and the product `x1·x2` changed together. While `x0 = 0` the product term vanishes, so `x1` and `x2` receive no gradient signal and no single-variable jump improves — escaping requires a compound move (Novelty Jump implements exactly this, but is off by default). Verified analytically and reproduced across seeds 1–7. |
| `st_e40` ([#102](https://github.com/spoorendonk/cbls/issues/102)) | hard, **mechanism unidentified** | Rows C1–C3 are degree-7 polynomials `(x-1)(x-2)(x-3)(x-5)(x-8)(x-10)(x-12) == 0` restricting each integer to `{1,2,3,5,8,10,12}`; C0 pins the free `x3` to a bilinear function of them, and four linear rows bound the combination. The search satisfies C1–C3 but misses a linear row by ~2. An earlier revision of this table claimed a violation barrier between allowed integer values — that was **wrong**: `int_jump_candidates` enumerates the entire domain when that domain spans at most 256 values, and these are `[1,12]`, so every allowed value is one jump away. The real mechanism is still open. |

## SCIP baseline

An independently-run open-source yardstick for the roster (issue #89), so the
numbers above sit against a solver we ran ourselves rather than only against
bounds MINLPLib publishes. Written by `../../minlplib/reference_solve.py`; per-row
results in `scip_baseline.csv`, the labelled three-way join in
`comparison_all.csv`.

**Why SCIP.** BARON is commercial and reachable only through the NEOS job queue,
so it cannot be batch-run reproducibly. Couenne is free but has seen little
development since ~2018 and is generally outperformed on this family. SCIP's
nonconvex spatial branch-and-bound is purpose-built, separately benchmarked on
MINLPLib in *Global Optimization of Mixed-Integer Nonlinear Programs with SCIP
8.0* ([PDF](https://optimization-online.org/wp-content/uploads/2022/12/scip8_minlp.pdf)),
and already a repository dependency — this adds none.

**What is matched.** SCIP reads the **same `.nl` files** through its own AMPL
reader, so neither solver sees a re-modelled instance and no formulation drift
can enter the comparison. Same roster and order (`bounds.csv`), same 60s
per-instance wall-clock budget, one thread each, and the same feasibility
tolerance — CBLS defaults to 1e-6 and SCIP's `numerics/feastol` default is 1e-6.
That parameter is left unset rather than assigned, and the runner reads the live
value back on every solve and writes a `FEASTOL-MISMATCH` note if SCIP's default
ever moves, so the shared tolerance is checked rather than assumed. SCIP rows are
scored by ports of the runner's `safe_gap` and its two-band BKS classification,
so the `gap_to_bks%` column means the same thing on both; the ports are pinned by
`tests/python/test_minlplib_scip_baseline.py`, one of whose tests recomputes them
against the gap column the C++ binary actually wrote, at that file's
six-significant-digit resolution.

**What is not matched, by construction.** SCIP is a complete global solver and
proves a dual bound; CBLS is a primal heuristic and proves none. Only the primal
columns are like-for-like. `comparison_all.csv`'s `dual_bound` therefore holds
what *that method* proved — NaN on CBLS rows — rather than repeating the
published dual on all three. Verification is also asymmetric: the C++ runner
re-checks its assignment against the model it built, whereas the SCIP side uses
`Model.checkSol(original=True)`, i.e. SCIP validating its own solution against
the pre-presolve problem. A solution SCIP cannot re-validate is not published as
feasible; zero rows in this run failed that check.

Run at **SCIP 10.0.2 / PySCIPOpt 6.2.1**, 60s per instance, with every parameter
other than `limits/time` and `randomization/randomseedshift` at its shipped
default — presolve, cuts, symmetry handling and the full primal-heuristic set are
all on, i.e. SCIP as a user would get it. The seed shift is 0, which is already
SCIP's default, so this uses SCIP's own seed sequence; the flag exists so a
multi-seed re-run is possible, not because a seed was chosen. The full
configuration (versions, budget, seed) is recorded in every row's `scip_version`
column, so a re-run at a different budget cannot be mistaken for this one.

Three properties of the budget were measured rather than assumed:
`timing/clocktype` defaults to 2 (wall clock), so `limits/time` is the same kind
of budget the CBLS runner imposes; instance reading falls outside it on both
sides (SCIP's `.nl` reads total well under a second across the roster, recorded
per row as `read_seconds`); and the solve stays single-threaded (CPU/wall
measured at ~1.0). Like the CBLS run this is a wall-clock budget, so these are
single-sample numbers.

Where SCIP proved no dual bound it reports its `1e20` infinity sentinel, which is
a *finite* float; the runner folds that to `NaN` at capture, so an unproved bound
is never published as a proof. Rows with no dual bound therefore read `NaN` in
`scip_dual_bound` and `scip_gap%`, the same spelling the CBLS rows use.

**Hardware.** The SCIP run was executed on an AMD Ryzen 5 5600H (12 logical
cores, Linux 7.0), one core in use. **The CBLS run's hardware is not recorded** —
`comparison.csv` has no machine column and that run predates this one. Recording
the machine per row is worth doing on the next re-run of either side.

That gap matters less than it first appears, and it bites the opposite way round
from the obvious guess. Both sides run a fixed 60s per instance, so:

- The **wall-clock totals are the robust number.** CBLS never terminates early —
  its 3001s is 50 × 60s by construction, and is therefore independent of the
  machine entirely. SCIP's 1011s is dominated by proving optimality on 34
  instances and stopping, not by clock rate; even a 2x hardware advantage would
  leave 505s against 3001s.
- The **counts are what a hardware difference would actually move.** Both
  "feasible within 60s" and "proved optimal within 60s" scale with machine speed,
  so those are the numbers a faster or slower box would change — in either
  direction, for either solver.

| | CBLS | SCIP |
|---|---|---|
| feasible | 46 / 50 | **49 / 50** |
| proved optimal | n/a (primal heuristic) | 34 / 50 |
| hit the 60s limit | 50 | 16 |
| total wall over the roster | 3001s | 1011s (median 0.28s; 31 instances under 1s) |
| integrality mismatches vs catalogue | 0 | 0 |
| verification failures | 0 | 0 |

**The failures are almost disjoint, and that is the useful part.** SCIP reaches
a feasible solution on all four instances CBLS cannot solve — two of them proved
optimal in under a quarter of a second:

| Instance | CBLS | SCIP | What it settles |
|---|---|---|---|
| `nvs01` | infeasible | optimal in 0.11s | The instance is not hard; #101 is an engine gap (single-variable jumps cannot move a product term pinned at zero). |
| `st_e40` | infeasible | optimal in 0.22s | Same — #102's mechanism is still unidentified, but "genuinely hard instance" is now ruled out as the explanation. |
| `elec25` | infeasible | 243.859 vs BKS 243.813 (0.02%) | Confirms an engine gap, not hardness: a feasible point of near-BKS quality is easy to reach. Originally attributed to #100; that is fixed, and the remaining cause is the undamped Newton jump ([#110](https://github.com/spoorendonk/cbls/issues/110), see the root-cause table above). |
| `elec50` | infeasible | 1422.3 vs BKS 1055.2 (34.8%) | Same mechanism at 50 points; SCIP does not close it either, but it does reach the feasible region. |
| `st_e36` | −147 (BKS −246) | **no feasible solution in 60s** | The one row the other way. SCIP spends the full budget and returns only a dual bound of −304.5. |

**Solution quality where both are feasible.** Buckets over the 38 instances that
both solve and whose `|BKS| >= 1e-4` (below that a percentage against the bound
is not informative — see the zero-BKS discussion above):

| | ≤0.01% | ≤1% | ≤10% |
|---|---|---|---|
| CBLS | 17 | 18 | 22 |
| SCIP | 32 | 32 | 33 |

SCIP is clearly ahead on quality, as expected of a mature global solver on a
roster capped at 150 variables and 150 constraints. Five instances go the other
way by a margin far larger than any rounding effect, and every one is a row
where SCIP exhausted the 60s budget:

| Instance | CBLS gap | SCIP gap |
|---|---|---|
| `eg_all_s` | 10.5% | 2324% |
| `ex8_1_5` | **matches BKS** | 100% |
| `ex8_6_1` | 49.1% | 99.6% |
| `eq6_1` | 20.4% | 27.0% |
| `maxmin` | 0.07% | 2.18% |

`ex8_1_5` is the sharpest of these: SCIP cannot make progress on it at all (its
two variables are unbounded, so the dual bound diverges), while CBLS now reaches
the published optimum exactly. It was CBLS's *worst* row before #107 was fixed.

### What #107 accounted for

#107 was found by noticing that instances with at least one **free (unbounded)**
variable did far worse, and asked how much of that gap the fix actually explains.
Measured before and after, over the rows each group solves whose `|BKS| >= 1e-4`:

| | instances | eligible | within 10% before | after |
|---|---|---|---|---|
| ≥1 free variable | 16 | 12 | 1 | **3** |
| no free variables | 34 | 27 | 20 | 19 |

So it explains **2 of the 11** free-variable misses — `ex8_1_5` and `shiporig`
join `maxmin`. A real but minority share: the correlation that motivated the
issue is only partly this bug, and the rest is still open. The no-free group's
−1 is `eq6_1` crossing the 10% line, which an A/B shows is *not* attributable to
the change (it is bit-identical between arms; that instance spans 20.5–36.9%
across seeds at one budget).

No finer-grained win/loss tally is published: `comparison.csv` writes objectives
at six significant digits, which is below the tie band on several rows, so a
per-instance head-to-head count would be reporting output precision rather than
search quality.

**One catalogue row looks stale.** On `ex6_2_6` SCIP proves optimality at
−3.51174e−06, better than MINLPLib's published primal bound (−2.60253e−06) and
marginally past its published dual (−3.49783e−06), which a valid dual bound for a
minimize instance cannot be. The absolute difference is 9e−07, so the two-band
rule correctly labels it `matches-bks` — but the `gap_to_bks%` column reads
−34.9%, which is the tiny-objective artifact rather than a real improvement. The
same caution applies to `ex6_2_11`, `least`, `mathopt1`, `prob09`, `ex14_2_4`
and `ex14_2_5`.

## Integrality

NL columns flagged integer/binary are built as CBLS `Int` variables, so the
mixed-integer instances are solved as genuine MINLPs rather than continuous
relaxations.

The NL header gives integer *counts* per category, not positions; the positions
follow Gay's variable ordering ("Hooking Your Solver to AMPL"), where columns
are laid out as

| Order | Category                                | Count         | Integers            |
|-------|-----------------------------------------|---------------|---------------------|
| 1     | nonlinear in both constraints and objs  | `nlvb`        | last `nlvbi`        |
| 2     | nonlinear in constraints only           | `nlvc - nlvb` | last `nlvci`        |
| 3     | nonlinear in objectives only            | `nlvo - nlvc` | last `nlvoi`        |
| 4     | linear arc variables                    | `nwv`         | —                   |
| 5     | other linear                            | remainder     | —                   |
| 6     | binary                                  | `nbv`         | all                 |
| 7     | other integer                           | `niv`         | all                 |

Three checks guard this mapping. The first two are *count* checks — they catch a
miscount but would not catch a correctly-sized set placed at the wrong offset:

- the reader fails loudly if the positions it derives don't account for exactly
  the count the header declares;
- the runner compares the recovered count against MINLPLib's own
  `nbinvars + nintvars` (the `n_disc_vars_bks` column) per instance and reports
  `integrality mismatch` in the tally. The published run has **zero mismatches
  across the roster**.

The positions themselves are pinned by the unit tests in `tests/test_minlplib.cpp`
("NL reader recovers discrete-variable count and positions"), which assert the
exact `var_is_discrete` vector for each block. That third check matters: an
earlier revision had the objective-only block length wrong and *both* count
checks still passed.

## Feasibility tolerance

A constraint counts as satisfied when its violation is `<= 1e-6`, matching
SCIP's default `numerics/feastol` — the right reference point for a
continuous/nonlinear roster, and the tolerance the SCIP baseline above runs at
(it is left at SCIP's default rather than set, so a future SCIP release changing
it surfaces as a mismatch instead of being masked). This is also the
engine-wide default
(`cbls::kDefaultFeasibilityTolerance`); the runner states it explicitly because
it is a published property of these results, and `--feas-tol` overrides it.

The violation is an *absolute* residual (for an equality row, the raw
`|lhs - rhs|`), so a much tighter tolerance is not meaningful: on a row whose
body is of magnitude 1e4, `1e-9` would demand ~13 significant digits.

Infeasible rows report the closest approach the search made, so a numerical
near-miss is distinguishable from a search that never reached the feasible
region: `max_violation` holds the residual, and the `note` column names the
worst-violated NL row and its sense.

## Caveats

- CBLS is a primal heuristic; large gap-to-BKS on hard multimodal instances is
  expected and acceptable (per issue #72).
- The roster is reproducible from the CSV but will drift as MINLPLib updates its
  catalogue; re-run `download.py` to refresh.
- Both runs are single samples on a wall-clock budget. The SCIP side is the more
  stable of the two — 34 of its 50 rows terminate with a proof rather than at
  the limit — but the 16 that hit the limit are as draw-dependent as the CBLS
  numbers.
- The roster's size budget (`nvars <= 150`, `ncons <= 150`) is set by what the
  NL reader and the selection filter admit, not chosen to favour either solver.
  It does mean the baseline runs SCIP on instances well inside its comfortable
  range, which is the right way round: the comparison should not flatter the
  engine under test.
- **The roster is the subset CBLS can express**, and this is the largest
  confounder in the SCIP comparison — it points the *opposite* way to the size
  caveat above, so the two belong together. The selection filter (see "Selection
  method") admits an instance only if every operator column it uses falls inside
  the CBLS DAG's op set, and the NL reader additionally rejects `V`/`S`/`F`/`d`
  segments. SCIP has no such restriction and would run the excluded instances.
  This is therefore a comparison on CBLS's expressible domain, not on MINLPLib.
