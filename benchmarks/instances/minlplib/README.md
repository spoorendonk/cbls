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
documents; per-instance "expressible? Y/N" is noted in `comparison.csv`'s `note`
column where relevant. No published Yuck numbers exist for these instances.

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
- `*.nl` — fetched text NL instance files.

Regenerate `comparison.csv` with:

    cmake --build build -j$(nproc)
    ./build/cbls_minlplib --time-limit 60 --seed 1 --commit "$(git rev-parse --short HEAD)"

## Results

Latest run: **60s per instance, seed 1, feasibility tolerance 1e-6**, commit
recorded per row in `comparison.csv`. The previously published numbers used a
5s budget and solved the mixed-integer instances as continuous relaxations;
both are fixed here, so the two runs are not comparable row-by-row.

| | count |
|---|---|
| roster | 50 |
| parsed and built (closed-model rate) | 50 (100%) |
| of which mixed-integer (integrality enforced) | 15 |
| **feasible** | **46** |
| — matching BKS (within the tie band) | 19 |
| — worse than BKS | 27 |
| — better than BKS | 0 |
| infeasible | 4 |
| unsupported / read errors / non-finite | 0 |
| integrality mismatches vs catalogue | 0 |
| verification failures | 0 |

Gap distribution over the 46 feasible instances: 20 within 0.01% of BKS, 23
within 1%, 27 within 10%.

Nothing in this roster beats a published bound. Three instances previously
appeared to, on margins of ~1e-5 percent; those are ties, not improvements, and
are now classified as `matches-bks` (see the tie band in the runner).

### The four unsolved instances

Every one is root-caused, and the verdict is recorded per row in
`comparison.csv` (merged from `analysis_notes.csv`):

| Instance | Verdict | Cause |
|----------|---------|-------|
| `elec25`, `elec50` | **bug** (#100) | Thomson problem. The feasible region contains coincident-point configurations where the Coulomb objective is `+inf`. Since the objective is folded in as an `obj <= bound` soft constraint, that violation clamps to ~1e30 and absorbs the real constraints' O(1) contributions in floating point, so the search loses its feasibility signal. Verified by construction: dropping the objective makes `elec25` feasible in 20s at violation 0; keeping it pins the search at the origin at violation exactly 1. |
| `nvs01` | hard | `420.169·√(x0²+900) == x2·x0·x1` needs `x0` and the product `x1·x2` changed together. While `x0 = 0` the product term vanishes, so `x1` and `x2` receive no gradient signal and no single-variable jump improves — escaping requires a compound move. |
| `st_e40` | hard | Rows C1–C3 are degree-7 polynomials `(x-1)(x-2)(x-3)(x-5)(x-8)(x-10)(x-12) == 0` restricting each integer to `{1,2,3,5,8,10,12}`. Moving between allowed values crosses forbidden integers where the violation spikes into the thousands. |

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
| 3     | nonlinear in objectives only            | `nlvo - nlvb` | last `nlvoi`        |
| 4     | linear arc variables                    | `nwv`         | —                   |
| 5     | other linear                            | remainder     | —                   |
| 6     | binary                                  | `nbv`         | all                 |
| 7     | other integer                           | `niv`         | all                 |

Two independent checks guard this mapping:

- the reader itself fails loudly if the positions it derives don't account for
  exactly the count the header declares;
- the runner compares the recovered count against MINLPLib's own
  `nbinvars + nintvars` (the `n_disc_vars_bks` column) per instance and reports
  `integrality mismatch` in the tally. The published run has **zero mismatches
  across the roster**.

## Feasibility tolerance

A constraint counts as satisfied when its violation is `<= 1e-6`, matching
SCIP's default `numerics/feastol` — the right reference point for a
continuous/nonlinear roster, and the same tolerance the SCIP baseline (#89)
will use.

The engine default is `1e-9`, which is not a reasonable requirement here: the
violation is measured as an *absolute* residual (for an equality row, the raw
`|lhs - rhs|`), so on a row whose body is of magnitude 1e4 it demands ~13
significant digits. The runner overrides it via `--feas-tol`.

Infeasible rows report the closest approach the search made, so a numerical
near-miss is distinguishable from a search that never reached the feasible
region: `max_violation` holds the residual, and the `note` column names the
worst-violated NL row and its sense.

## Caveats

- CBLS is a primal heuristic; large gap-to-BKS on hard multimodal instances is
  expected and acceptable (per issue #72).
- The roster is reproducible from the CSV but will drift as MINLPLib updates its
  catalogue; re-run `download.py` to refresh.
