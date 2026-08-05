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
- `*.nl` — fetched text NL instance files.

Regenerate `comparison.csv` with:

    cmake --build build -j$(nproc)
    ./build/cbls_minlplib --time-limit 60 --seed 1 --commit "$(git rev-parse --short HEAD)"

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
Treat any single row as one draw, not a measurement. Reporting a median over
several seeds is the fix; it is not done here, and the runner has no flag for a
deterministic budget yet (the engine supports one — `time_limit = 0` plus
`SearchConfig::max_iterations` — but `cbls_minlplib` requires `--time-limit > 0`).

| | count |
|---|---|
| roster | 50 |
| parsed and built (closed-model rate) | 50 (100%) |
| of which mixed-integer (integrality enforced) | 15 |
| **feasible** | **46** |
| — matching BKS (within the tie band) | 17 |
| — better than BKS, but inside the tolerance slack | 1 |
| — worse than BKS | 28 |
| — better than BKS | 0 |
| infeasible | 4 |
| unsupported / read errors / non-finite | 0 |
| integrality mismatches vs catalogue | 0 |
| verification failures | 0 |

Gap distribution over the feasible instances: **20 within 0.01% of BKS, 21
within 1%, 25 within 10%.**

Five rows have a numerically zero BKS (`|BKS| < 1e-12`), for which the runner
writes an *absolute* residual into the `gap_to_bks%` column rather than a
meaningless percentage against zero: `mathopt1`, `prob09`, `least`, `ex14_2_4`
and `ex14_2_5`. Those values are not percentages. The buckets above exclude the
first three, whose residual is non-zero, and retain `ex14_2_4`/`ex14_2_5`, where
objective and BKS are both exactly 0 and so are exact matches at any threshold.
Excluding all five instead gives 18 / 19 / 23 over 41 rows. Counting the
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
long after 5s — `chain50` (17.6s), `ex8_4_5` (25.1s), `tln2` (26.6s), `spring`
(30.9s), `minlphi` (36.5s) — so a 5s budget would publish all five as
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
| `elec25`, `elec50` | **bug** (#100) | Thomson problem. The feasible region contains coincident-point configurations where the Coulomb objective is `+inf`. Since the objective is folded in as an `obj <= bound` soft constraint, that violation clamps to ~1e30 and absorbs the real constraints' O(1) contributions in floating point, so the search loses its feasibility signal. Verified by construction: dropping the objective makes `elec25` feasible in 20s at violation 0; keeping it pins the search at the origin at violation exactly 1. |
| `nvs01` ([#101](https://github.com/spoorendonk/cbls/issues/101)) | hard | `420.169·√(x0²+900) == x2·x0·x1` needs `x0` and the product `x1·x2` changed together. While `x0 = 0` the product term vanishes, so `x1` and `x2` receive no gradient signal and no single-variable jump improves — escaping requires a compound move (Novelty Jump implements exactly this, but is off by default). Verified analytically and reproduced across seeds 1–7. |
| `st_e40` ([#102](https://github.com/spoorendonk/cbls/issues/102)) | hard, **mechanism unidentified** | Rows C1–C3 are degree-7 polynomials `(x-1)(x-2)(x-3)(x-5)(x-8)(x-10)(x-12) == 0` restricting each integer to `{1,2,3,5,8,10,12}`; C0 pins the free `x3` to a bilinear function of them, and four linear rows bound the combination. The search satisfies C1–C3 but misses a linear row by ~2. An earlier revision of this table claimed a violation barrier between allowed integer values — that was **wrong**: `int_jump_candidates` enumerates the entire domain when that domain spans at most 256 values, and these are `[1,12]`, so every allowed value is one jump away. The real mechanism is still open. |

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
continuous/nonlinear roster, and the same tolerance the SCIP baseline (#89)
will use. This is also the engine-wide default
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
