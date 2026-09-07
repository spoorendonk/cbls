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
- `../../minlplib/run_benchmark.py` — the CBLS re-run driver. See
  "Re-running the CBLS rows" below; that is the supported way to regenerate
  `comparison.csv`.

Regenerate `scip_baseline.csv` and `comparison_all.csv` with (needs the
`benchmarks` extra — `pip install -e '.[benchmarks]'`):

    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --time-limit 60

Run the CBLS side first: the merge reads whatever `comparison.csv` holds. To
rebuild only the merge after a fresh CBLS run, add `--merge-only` — which is
what `run_benchmark.py` does for you, so the SCIP baseline is never re-solved
by a CBLS re-run.

## Re-running the CBLS rows

**One command**, from a configured Release build directory and a clean checkout:

    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py

That rebuilds `cbls_minlplib`, solves the 50-instance `bounds.csv` roster
**serially** at 60s each, rewrites `comparison.csv` and `anytime_trace.csv`, and
re-merges the **CBLS rows** of `comparison_all.csv`. Add `--dry-run` first to
print the exact commands, the roster size and the estimate without running
anything.

**Budget ~50 minutes of solving** (50 × 60s), plus model build and read time on
top. Read
`--time-limit`, `--seed 1` and the roster as *fixed*: the point of a re-run is
that only the engine differs from the previous table.

**The box must be quiet.** These are wall-clock-budgeted solves, so a concurrent
build, test suite or second benchmark changes how many iterations each instance
gets and produces numbers comparable neither to each other nor to the committed
table. Check `uptime` before starting, and do not run this alongside anything
else — that is why the driver never parallelises the solves and why
`--build-jobs` (build only) defaults to 4.

What the driver refuses, and why each refusal matters:

| Refusal | Why |
|---|---|
| dirty working tree | rows would carry a plain SHA whose code is not what ran |
| build dir not `CMAKE_BUILD_TYPE=Release` | an unoptimised build measures a different engine |
| unconfigured build dir | nothing to rebuild the runner from |
| a roster instance with no `.nl` | a hole in the table found 40 minutes in |
| build dir configured from another checkout | the rows would name one checkout and measure another |
| `--no-build` with no runner binary | nothing to run, found at the first solve |
| `--instances` without `--out`/`--trace-out`/`--staging-dir` | a debug subset would truncate a fifty-row table, or leave short-budget rows for the next run's resume to publish |
| `--instances` with `--out` resolving to `comparison.csv` | same, via a relative path the explicit-`--out` guard would otherwise wave through |
| a staging directory stamped with another commit, budget or seed | resuming it would mix two configurations into one table |
| missing `scip_baseline.csv` (unless `--no-merge`) | the merge would publish `comparison_all.csv` without SCIP rows |

It also rebuilds the runner target itself, so the binary cannot lag the SHA it
is about to be labelled with. `--dry-run` prints the plan and exits non-zero if
any refusal applies, so it works as a precheck.

**Resumable.** Each instance is solved in its own process into
`build/minlplib-rerun/<instance>.csv` (plus `.trace.csv` and a `.log` holding
that instance's runner output and tally). A re-invocation skips instances that
already have a *complete* staged row. Incomplete means any of: a header-only
file (what a killed job leaves behind), a torn last line, a row stamped with a
different `commit_sha`, or a missing `.trace.csv` — all are re-solved.
`build/minlplib-rerun/stamp.txt` records the commit, budget and seed the
directory's rows belong to, and a resume against a different one is refused
outright rather than silently mixed. `--no-resume` forces a full re-solve.

`comparison.csv` and `anytime_trace.csv` are only replaced at the end, by an
atomic rename of a fully-assembled file, so an interrupted run leaves them
byte-for-byte intact. `comparison_all.csv` is the exception — the merge step
rewrites it in place. If the merge fails after `comparison.csv` is written the
driver says so and exits 1, leaving `comparison_all.csv` on its *previous* CBLS
rows; just re-run, which reuses the staged rows and goes straight back to the
merge.

`elec25`/`elec50` **stay in the roster** — `bounds.csv` is the roster of record
and #123 asks for 50 instances — but their rows are published as documented
failures and are excluded from every aggregate and quality claim, per
[#87](https://github.com/spoorendonk/cbls/issues/87) ("Do not publish `elec`
rows until #110 lands and #116's criterion can actually be checked"). The
driver's summary holds them apart for that reason. **If an `elec` row comes back
feasible with a finite objective, stop and check #110/#116 before publishing
anything about it** — that would be a result, not a routine table refresh.

### After the run

1. `git diff benchmarks/instances/minlplib/` — expect changes confined to
   `comparison.csv`, `anytime_trace.csv` and the `cbls` rows of
   `comparison_all.csv`. The `published-bks` and `scip` rows are engine-independent
   and must be byte-identical; if they moved, something re-solved SCIP and the
   run must be redone.
2. Update every run-derived number below. The per-instance runner tallies in
   `build/minlplib-rerun/*.log` carry the tie/improvement-band counts, and the
   driver's own summary gives the verdict counts — but note its `counted` line
   excludes `elec`, whereas the **Results** tally counts the whole roster, so
   take its `rows written` figure as that table's `roster`. The full list,
   because it is longer than it looks and no test checks any of it:
   - **Results**: the tally table; the gap-distribution sentence; the zero-BKS
     paragraph (`21/22/26`, `19/20/24 over 41 rows`, and which instances have a
     numerically zero BKS); the worked examples in the two-band paragraphs
     (`ex6_2_6`, `prob06`, `ex8_4_5`).
   - **Why 60s**: the cumulative-feasibility table, recomputed from the new
     `anytime_trace.csv`, *and* the prose under it — the named late-feasible
     instances with their times, the 46%/22% split, and the `eg_all_s`
     bound-tightening analysis.
   - **The four instances the published run left infeasible**: the heading
     count and the table, if the set changed. `st_e40` is expected to leave it
     on this re-run (#102); drop its row here and in `analysis_notes.csv`
     together, and fix the heading count, only once the regenerated
     `comparison.csv` actually shows it feasible.
   - **SCIP baseline**: the CBLS column of the head-to-head table — `feasible`,
     `hit the 60s limit`, and `total wall over the roster` (step 4's #113 moves
     `wall_seconds`, so this one will change); the "failures are almost
     disjoint" table; the CBLS row of the quality buckets; and the "What #107
     accounted for" table.
   - The SCIP side of all of the above is engine-independent and must not move.
3. Delete the "**The table predates #120**" staleness paragraph below once the
   numbers are the new run's.
4. **Attribute any material movement to the engine changes, not to an
   algorithmic improvement.** Five landed between the old table and this one:
   [#111](https://github.com/spoorendonk/cbls/issues/111) (the diversification
   kick gained a structural half),
   [#112](https://github.com/spoorendonk/cbls/issues/112) (guarded randomisation
   on unbounded domains), [#113](https://github.com/spoorendonk/cbls/issues/113)
   (the FJ deadline stride is bounded in time and iterations, so `wall_seconds`
   moves and instances that were silently overrunning now stop at the budget),
   [#114](https://github.com/spoorendonk/cbls/issues/114) (an unbounded Int gets
   FJ jump candidates instead of freezing) and
   [#120](https://github.com/spoorendonk/cbls/issues/120) (implied variable
   bounds by propagation, replacing the fixed inf-clamp). Note also that these
   are single-sample numbers — the spread quoted in **Results** is wide enough
   that a single row moving is not evidence of anything.
5. `analysis_notes.csv` is merged into the note column by the runner. A root
   cause that no longer applies has to be edited there, not in `comparison.csv`.
   **Nothing warns you**: the runner merges the curated note only onto rows that
   came back infeasible, and its "stale analysis note (now solved)" warning sits
   inside that same guard, so it can never fire. Diff the noted instances
   (`elec25`, `elec50`, `nvs01`, `st_e40`) against the new table by hand —
   #114, #120 and #102 are exactly the kind of change that could retire one.
   `st_e40`'s note is the live case: the engine solves that instance now
   (see the root-cause table below), so this re-run is expected to retire it.
   Its row is deliberately still in `analysis_notes.csv` until then — if the
   row comes back infeasible under the published protocol after all, that note
   is the only thing that would flag it.
6. Confirm the provenance the whole re-run is for:
   `.venv/bin/python -c 'import csv,sys;print(sorted({r["commit_sha"] for r in csv.DictReader(open("comparison.csv"))}))'` must print exactly one
   SHA, and that SHA must be the checkout you built. Two SHAs mean a resumed run
   spanned a commit; discard the staging directory and re-run.
7. Record the machine in the **Results** preamble — CPU model and core count.
   The "Hardware" note in the SCIP baseline section below asks for this on the
   next re-run of either side, and this is it.
8. Run the Python suite:
   `.venv/bin/pytest tests/python/test_minlplib_scip_baseline.py`.
   `test_safe_gap_reproduces_the_cpp_runners_published_gap_column` reads the
   regenerated `comparison.csv` and requires at least 20 rows with enough
   resolution to cross-check, so a new table can legitimately turn it red. Then
   run the full gated suite from `CLAUDE.md`'s **Build & Test** before pushing.

## Results

Latest run: **60s per instance, seed 1, feasibility tolerance 1e-6**, commit
recorded per row in `comparison.csv`. The tally below, the gap buckets and the
anytime profile all come from that **one** run; its incumbent trace is committed
as `anytime_trace.csv`, so every number in this section is reproducible from the
checkout it was measured at, which each row's `commit_sha` names.

**The table predates #120 and #102, and does not describe the current engine.**
`st_e40` is the change with a named row below: at engine commit `167116e` on
branch `issue-102-st-e40` it reaches its BKS of 30.4142135 at both 10s and 60s,
seed 42, where this table records it infeasible. #123 is the re-run that will
move it (and whatever else the trajectory change moved); until then the tally,
the gap buckets and the "four instances" section below all describe the
**published run**, not the engine. The `.nl`
adapter now derives implied variable bounds from the purely linear rows instead
of substituting `inf_clamp`/`int_inf_clamp` for an infinite one, and it is on by
default — so any instance whose linear rows imply something is searched over a
different box than the one these numbers came from. How many that is has not been
measured; the `elec` family is *not* among them (its rows are nonlinear, so the
adapter skips them and the `±1e9` box noted below still stands). Re-run before
comparing anything new against this table, and see
`benchmarks/instances/mipfeas/README.md` for what the same change did to the
MIPLIB roster.

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

These are the counts of the **published run**, not of the current engine. The
`infeasible` row is 4 because `st_e40` was infeasible at the commit the table
was measured at; it is not at engine commit `167116e` (see below). Re-derive the
tally from a regenerated `comparison.csv` on #123 rather than editing it here —
a hand-adjusted count would no longer match the table it is a summary of.

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

### The four instances the published run left infeasible

Every one is root-caused, and the verdict is recorded per row in
`comparison.csv` (merged from `analysis_notes.csv`). One of the four —
`st_e40` — has since been fixed and is no longer unsolved at engine commit
`167116e`; it stays in this table, with its verdict updated, until #123
regenerates the rows:

| Instance | Verdict | Cause |
|----------|---------|-------|
| `elec25`, `elec50` | **bug** ([#110](https://github.com/spoorendonk/cbls/issues/110)) | Thomson problem: points on the unit sphere, Coulomb objective `+inf` wherever two coincide. **The objective-encoding defects (#100) are fixed and are no longer the blocker.** What remains: the `.nl` declares no finite variable bounds, so the box is the ±1e9 inf-clamp; random init starts ~1e9 out, and shrinking each variable toward 0 is a huge row improvement — which parks the search on the origin, a stationary point of every row `x²+y²+z²=1`. The Float jump offers a single *undamped* Newton step (`x0 - residual/grad`) plus `lb`/`ub`/midpoint; near the origin that step overshoots wildly and is rejected, and because a candidate was nonetheless *generated* the #107 escape probe is suppressed — so the variable freezes at score 0. Measured: escape probe fires only at exactly `x0 = 0`; at `x0 = 0.001/0.01/0.1` the score is 0 with the probe armed or not. Infeasible at violation ≈1 **both with the objective present and with it neutralised**, so it is not objective-related — the earlier "dropping the objective makes elec25 feasible in 20s" claim no longer reproduces. Tightening `inf_clamp` to 1 makes `elec25` feasible at violation 0 post-#100 (pre-#100 it was infeasible at *every* clamp), because clamping accidentally supplies the missing damping. |
| `nvs01` ([#101](https://github.com/spoorendonk/cbls/issues/101)) | hard | `420.169·√(x0²+900) == x2·x0·x1` needs `x0` and the product `x1·x2` changed together. While `x0 = 0` the product term vanishes, so `x1` and `x2` receive no gradient signal and no single-variable jump improves — escaping requires a compound move (Novelty Jump implements exactly this, but is off by default). Verified analytically and reproduced across seeds 1–7. |
| `st_e40` ([#102](https://github.com/spoorendonk/cbls/issues/102)) | **fixed** — was an engine gap, not hardness | Rows C1–C3 are degree-7 polynomials `(x-1)(x-2)(x-3)(x-5)(x-8)(x-10)(x-12) == 0` restricting each integer to `{1,2,3,5,8,10,12}`; C0 pins the free `x3` to a bilinear function of them, and four linear rows bound the combination. That leaves 343 integer combinations, **52 of them feasible**. The search reached only 3 of the 343: it fell into a limit cycle within ~20 GLS iterations — `x1` flipping between two values and `x3` hopping between the roots of the four rows containing it — and neither trapped value appears in any feasible combination (all 52 need `x0 >= 5` and `x1 >= 5`). It then spent **92% of a 155 000-iteration run on GLS weight bumps that changed nothing**, because diversification was gated on 100 non-improving *batches* of 1000 iterations and, before a first feasible solution, no batch ever improves — a fixed cadence of one kick per 100 000 iterations with no feedback from the search. Two earlier explanations in this table were both wrong: a violation barrier between allowed integer values (`int_jump_candidates` enumerates the whole domain below 256 values, and these are `[1,12]`), and "the feasible combination" singular (there are 52). Fixed by `GFJConfig::unproductive_iterations`: a batch that stops reducing the real rows' violation ends, and `solve()` takes the diversification kick as due (without arming the Float escape probe, which stays gated on `perturbation_period` — see #107). `solve()` arms that exit only once its own stagnation count reaches `perturbation_period / 20` batches, which is what keeps the mechanism out of the objective-descent phase where its measure is blind — see the objective-quality section below for what that cost before it was gated. Measured at engine commit `167116e`, branch `issue-102-ex8_6_1`: reaches the BKS 30.4142135 at 10s and 60s, seed 42, and at 10s on seeds 1, 2 and 3 as well. The `comparison.csv` row still reads infeasible and is left alone deliberately; #123 is the re-run that regenerates it. |

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
a feasible solution on all four instances CBLS did not solve in this run — two
of them proved optimal in under a quarter of a second. (`st_e40` is no longer
one of CBLS's failures at engine commit `167116e`; it is kept in the table
because the CBLS column is this run's, and #123 regenerates it.)

| Instance | CBLS | SCIP | What it settles |
|---|---|---|---|
| `nvs01` | infeasible | optimal in 0.11s | The instance is not hard; #101 is an engine gap (single-variable jumps cannot move a product term pinned at zero). |
| `st_e40` | infeasible in this table → **BKS 30.4142135** at engine commit `167116e` | optimal in 0.22s | Was an engine gap, now closed (#102): the outer loop only diversified once per 100 000 GLS iterations, so a limit cycle established after ~20 iterations ran essentially unchallenged. The `infeasible` reading is this table's, measured before the fix; see the root-cause row above for the mechanism and the measurement. |
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

### Objective quality under the #102 unproductive-batch exit

The #102 change ends a GLS batch that has stopped reducing the real rows'
violation and takes the diversification kick as due. That alters the search
trajectory on every instance, not only the one it was found on, so it was
measured on objective quality and not only on feasibility — a feasibility-only
tally cannot see a row that stayed feasible and got worse.

That measurement is what caught the one real defect in the change, so the
numbers below are given in **three** arms: `main`, the first cut of the fix
(`before`), and the fix as it now stands (`after`, engine commit `167116e`).
Eight-instance probe, `--time-limit 10 --seed 42`, three separately built
binaries, each arm run serially and interleaved per instance on an idle box,
gap to BKS. This is an **indicative probe, not the published protocol** — one
seed, a 10s budget, and a subset chosen to include the instances an earlier arm
had regressed:

| instance | main | before | after |
|---|---|---|---|
| `alkylation` | 99.98% | 0.07% | **0.03%** |
| `st_e36` | 40.24% | **0.87%** | 3.38% |
| `maxmin` | 0.01% | 0.10% | 0.20% |
| `kall_ellipsoids_tc02b` | 161.76% | 159.73% | **149.36%** |
| `st_e40` | infeasible | **0.00%** (BKS) | **0.00%** (BKS) |
| `nvs01` | infeasible | 21.94% | 30.16% |
| `ex4_1_8` | 0.00% | 0.00% | 0.00% |
| `ex8_6_1` | 69.67% | **89.39%** | **69.71%** |

**`ex8_6_1` was the reason this change was held, and it is fixed.** The `before`
arm lost about 20 gap points there on all four paired seeds tried; the `after`
arm is level with `main` on all four (`-8.63 / -6.01 / -10.20 / -11.04` against
main's `-8.62 / -6.01 / -10.05 / -9.82`).

Checked at the published 60s budget too, since the regression was a *rate* effect
and a longer budget is where a residual would show. Gap to BKS, `main` against
`after`, seeds 42/1/2/3: **39.58/64.86/53.14/43.48** against
**46.97/57.78/55.98/30.11** — two seeds each way, `after` ahead on the mean
(47.7% against 50.3%). That is the two arms being indistinguishable, which is
what "recovered" means here; contrast the `before` arm, which was worse on 4/4.

The mechanism, which took instrumentation rather than argument to find: the
exit's progress measure sums the **real** rows and deliberately cannot see the
artificial `obj <= bound` row. Before the first feasible solution that is the
point of it. After it, the search's work is a *trade* between the two — the
bound is tightened on every new best, FJ pulls the assignment off the
real-feasible set to chase the objective row, and the real rows settle at a
strictly **positive** equilibrium. Since the measure's reference is a running
*minimum* over the batch, "no new all-time low in `unproductive_iterations`"
is then the normal state of a batch that is working perfectly, and the detector
is unconditionally true. Measured on `ex8_6_1`: the run improved its incumbent
on **152 of its first 162 batches** and was declared stuck on nine of the rest,
with the measure between 0.016 and 0.51 — three orders above the tolerance at
which this engine calls a row satisfied, so no zero-test could ever have seen
it. Three of the nine kicks drew LNS (`lns_interval = 3`), each bounded by
`min(2.0, remaining())`, and those three alone consumed **4.7s of the 10s
budget**. The run managed 8.2k GLS iterations instead of 32k.

That also explains the "identical objective at 2s and 10s" that the issue asked
someone to pull on: the eight missing seconds were spent inside LNS repairs the
search never benefited from.

Two things were wrong, and both are fixed:

1. `progress_residual` counted any residual `> 0.0`, while `is_violated` calls a
   row satisfied at `<= 1e-9`. On inequality rows over integers the two agree; on
   **equality** rows over continuous variables a satisfied row's residual
   (`|body(x)|`) lands a few ulp off zero rather than on it, so the measure could
   never reach its floor and the earlier "the real rows are all satisfied" guard
   was dead code. `ex8_6_1` is 45 nonlinear equalities over 75 continuous
   variables — the equality fraction tracks this table closely: `maxmin` 0/78 and
   `alkylation` 3/11 are the big wins, `kall_ellipsoids_tc02b` 109/128 barely
   moves, `ex8_6_1` 45/45 was the regression.
2. The exit had no witness other than its own blind measure. `solve()` now arms
   it only once its own count of consecutive non-improving batches reaches
   `perturbation_period / 20`, which makes the mechanism an **acceleration** of
   the stagnation window (one kick per ~6 batches at the default) rather than a
   replacement for it (one per ~300 GLS iterations). A search improving its
   incumbent every few batches never arms it and is then bit-identical to a run
   with the mechanism off. Arming, rather than merely suppressing the kick, is
   what makes that exact: an armed exit still *ends* batches early, and on
   `ex8_6_1` those early ends alone still cost about six gap points with every
   kick suppressed.

**The `20` is tuned and says so.** A sweep of `{5, 10, 20}` batches over four
paired seeds: all three remove the `ex8_6_1` regression completely, because a
search improving that often never reaches any of them. They differ on the
instance the mechanism exists for — `st_e40` reaches its BKS on 4/4 seeds at a
threshold of 5 batches and on 2/4 at 10 or 20 — so the smallest is chosen.
Nothing here establishes that it transfers off MINLPLib; it is the same standing
complaint `GFJConfig::unproductive_iterations` records against its own `300`.

**What the fix trades, stated plainly.** Over four seeds (42, 1, 2, 3):

- `ex8_6_1` recovers on 4/4 and `st_e40` keeps its BKS on 4/4 — the two the
  change is judged on.
- `alkylation` is *better* than the `before` arm on 4/4 (0.02–0.10% against
  0.07–0.11%), and `maxmin` is level (mean 0.12% against 0.17%; `main` is 0.59%).
- `st_e36` is the one loss: 3.38% against `before`'s 0.87% on three of the four
  seeds, level on the fourth. Still far better than `main`'s 32–40%, but it is a
  loss and not noise.
- `nvs01` becomes feasible on 4/4 where `main` never does, and its objective is
  too unstable in **both** arms to separate them (`before` 21.9/414.7/662.6/319.1
  against `after` 30.2/12.2/734.9/769.3). Read it as a feasibility result only.

`maxmin`'s `main` column moved from 3.90% in an earlier revision of this table to
0.01% here at the same seed and the same binary, which is the honest calibration
for how much of a sub-1% difference on these rows is run-to-run variance.

The published rows below are **not** regenerated from this probe. That is #123's
job, at the documented protocol, and it is where a proper win/loss tally over
the full roster belongs.

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
