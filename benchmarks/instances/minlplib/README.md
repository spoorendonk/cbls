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
- `comparison.csv` — the `cbls_minlplib` runner's rows, assembled by
  `run_benchmark.py` after SCIP has checked every verified row (see
  "Verification" below; the runner refuses to write this file itself): CBLS objective,
  gap-to-BKS, gap-to-dual, feasibility, notes, commit SHA, closest-approach
  residual (`max_violation`), integer-variable count (`n_int_vars`), the LNS
  destroy-repair count of the run (`lns_repairs`), how many of those repairs the
  accept rule kept (`lns_repairs_accepted`), where the run first reached
  feasibility (`first_feasible_objective`, `time_to_first_feasible` — issue
  #149) and the search configuration the row was produced under
  (`search_config`, a canonical `key=value;...` cell — see below). Both LNS
  counters read `NaN` on a row where no solve completed: 0 there would be the
  different — and false — claim that LNS ran and repaired nothing, and on the
  accepted cell that it repaired and rolled every one back. The first-feasible
  pair obeys the same rule, with `time_to_first_feasible` as the cell that says
  whether a feasible point was recorded at all — its objective can be `NaN` or
  `inf` on a row that *did* reach feasibility, because the point of arrival can
  be the non-finite-objective witness of #100.
  The trailing columns are also how a reader dates a `comparison.csv`: each
  names the issue that added it, so the trailing set *is* the provenance. In
  order — `search_config` (#136), `lns_repairs` (#143), `lns_repairs_accepted`
  (#150), then the first-feasible pair (#149). A header stopping at
  `n_int_vars` predates all four; the committed table, regenerated by #123,
  carries all five.
- `analysis_notes.csv` — curated per-instance root-cause verdicts
  (`bug` vs `hard`) for instances the runner cannot solve. Merged into
  `comparison.csv`'s note column, so the data carries its own explanation.
- `anytime_trace.csv` — incumbent objective against wall time for the published
  run (`instance,time_seconds,objective,new_best`), written by `--trace`. The
  `objective` column is the internally *minimised* value, so a maximize instance
  appears negated relative to `comparison.csv`.
- `campaign_summary.json` — the published campaign's machine-readable summary,
  written by `../../minlplib/campaign_report.py --json` (README step 2): every
  aggregate under the stated aggregation rule, the provenance, and the
  per-instance figures including each anytime score and its reference. A test
  requires it to equal a fresh rendering of the committed tables.
- `scip_baseline.csv` — written by `../../minlplib/reference_solve.py`: the SCIP
  baseline's objective, gaps against the same published bounds, feasibility,
  wall time, plus the dual bound, gap and status only a complete solver
  produces, and the exact `SCIP x / PySCIPOpt y` pair per row.
- `comparison_all.csv` — the three-way comparison in long format, one row per
  `(instance, method)` with `method` in `published-bks` / `cbls` / `scip`. Also
  written by `reference_solve.py`, by joining the two CSVs above with
  `bounds.csv`.
- `comparison_seeds.csv` — every seed's CBLS rows, `seed` first and then
  `comparison.csv`'s columns, written by `run_benchmark.py` (#141).
  `comparison.csv` stays the single pre-registered seed 1. The committed one
  holds seeds 1, 2 and 3 of #123's campaign. See "Seeds, and what each publish
  records".
- `comparison.run.json`, `comparison_seeds.run.json` — the run record beside
  each published table (commit, budget, seed, machine, concurrency), written by
  `run_benchmark.py` at publish time. Both are committed for #123's campaign.
- `*.nl` — fetched text NL instance files.
- `../../minlplib/run_benchmark.py` — the CBLS re-run driver. See
  "Re-running the CBLS rows" below; that is the supported way to regenerate
  `comparison.csv`.
- `../../minlplib/campaign_report.py` — regenerates every run-derived number in
  **Results** and **SCIP baseline** below from the four tables above (and the
  `.nl` bounds, for #107's free-variable split), plus anytime scores, as Markdown
  and as a JSON summary (#142), and rewrites this README's
  `campaign_report` blocks (`--write-readme`). It solves nothing. See "After the
  run" step 2.
- `../../minlplib/first_feasible_report.py` — scores issue #149's question
  (does the first feasible point determine the final objective?) from the
  `first_feasible_objective` column of a multi-seed run. Reads CSVs and writes
  at most the per-instance table you name; it solves nothing. See "Is the final
  objective set by the first feasible point?" below.
- `../../minlplib/run_ablation.py`, `../../minlplib/ablation_report.py` — the
  ablation campaign driver and its scoring (issue #143). See "The ablation
  campaign" below. With `--campaign transfer-145` the same driver runs issue
  #145's unproductive-exit transfer check on the held-out roster; see
  [HELDOUT.md](HELDOUT.md). It writes only to a scratch `--out-dir` and refuses one
  anywhere inside `benchmarks/instances/`, so it can never touch the files
  listed above.

Regenerate `scip_baseline.csv` and `comparison_all.csv` with (needs the
`benchmarks` extra — `pip install -e '.[benchmarks]'`):

    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --time-limit 60

Run the CBLS side first: the merge reads whatever `comparison.csv` holds. To
rebuild only the merge after a fresh CBLS run, add `--merge-only` — which is
what `run_benchmark.py` does for you, so the SCIP baseline is never re-solved
by a CBLS re-run.

### Ablation arms

The runner's search configuration is settable per run rather than per build, and
whatever was set is recorded in each row's `search_config` cell, so a results
file states the configuration it was produced under (#136):

```
--no-float-hook            drop the FloatIntensifyHook
--no-lns / --lns-interval N   drop LNS, or change how often a kick is a repair
--compound-moves / --no-compound-moves, --novelty-prob P
--unproductive-iters N     the #102 unproductive-batch exit (<= 0 = old cadence)
--perturbation-period N    batches without improvement before a kick
--max-iterations N         GLS iteration budget (0 = unlimited)
--no-time-limit            disable the wall clock; requires --max-iterations
```

Two combinations are refused rather than accepted: `--lns-interval` with
`--no-lns`, and `--novelty-prob` without `--compound-moves`. The engine
short-circuits past the second flag in each pair, so accepting them would record
an arm that the run did not have.

Any non-default value refuses to write the published `comparison.csv` (and
`anytime_trace.csv`): an arm's rows would look exactly like the published ones
while describing a different search. `cbls_uc_chped` carries the same flags, and
`benchmarks/common/search_config_flags.h` is the single definition of all of
them.

### The ablation campaign

`benchmarks/minlplib/run_ablation.py` runs the arms above as one interleaved
campaign (issue #143) and `ablation_report.py` scores it:

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir /scratch/ablation

The same command resumes after an interruption; `--dry-run` prints the plan and
the cost, `--report-only` re-scores an existing `results.csv`. Four properties
are the point of it and each is pinned by a test in
`tests/python/test_minlplib_ablation.py`:

* **Interleaved per instance.** Every (arm, seed) for one instance runs back to
  back before the next instance, with the arms innermost, so a control and its
  arm are minutes apart rather than a roster pass apart. Arm-major order would
  make every comparison a comparison across an hour of machine drift.
* **Serial, and asserted.** One solve process at a time, under an exclusive lock
  on the out-dir plus a load-average refusal at start.
* **Scratch only.** `--out-dir` is refused anywhere inside
  `benchmarks/instances/`, and the runner is always invoked with `--instance`,
  which is its own refusal to write the published table. The control arm needs
  both: its flags are all default, so the runner's arm guard alone would let it
  through.
* **Resumable.** Each completed run is appended to `<out-dir>/results.csv` and
  fsynced before the next starts; a restart skips exactly the recorded
  `(instance, arm, seed)` triples and refuses an out-dir stamped with another
  commit, budget, seed set or arm set.

The `--no-lns` arm is **gated on data, not on judgement**: a one-seed control
pass over the roster is run first purely to read the `lns_repairs` column, and
the arm runs only if some instance recorded a repair. With no repair anywhere
the engine takes the same branch at every diversification kick with or without
LNS, so the arm would measure nothing; the reading and the verdict are written
to `<out-dir>/lns_gate.json` either way.

The gate reads repairs **attempted**, not repairs accepted, even though
`lns_repairs_accepted` publishes the latter. A rejected repair randomises
variables, runs a repair pass and rolls back, so it consumes both the RNG stream
and seconds of the budget — the arm still diverges from the control, and that
budget cost is the thing the arm exists to price. Acceptance is reported instead:
`ablation_report.py` prints both totals per arm, which is what makes "LNS is
working" separable from "LNS is only spending".

The **noise floor is measured**, from the control's own across-seed spread —
per instance `t * s_i * sqrt(1/k_arm + 1/k_control)`, a two-sided 95% Student
band (`t` = 4.30 at three seeds, not 2.0), bounded below by the runner's own
tie band for that instance's published bound so that a control agreeing to
~1e-7 on every seed cannot hand an arm a floor it clears by rounding. One floor
per instance and no aggregate floor: on a roster whose gaps span six orders of
magnitude the only aggregate worth quoting is a rank statistic, which has no
gap-point floor to be read against. `ablation_report.py`'s module docstring is
the definition of record. Any effect at or inside its floor is
reported as "inside the noise" with the floor quoted; an instance where one arm
is feasible and the other is not contributes no gap delta at all and is counted
in its own bucket instead.

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
| dirty working tree (the tables the driver itself publishes excepted) | rows would carry a plain SHA whose code is not what ran |
| build dir not `CMAKE_BUILD_TYPE=Release` | an unoptimised build measures a different engine |
| unconfigured build dir | nothing to rebuild the runner from |
| a roster instance with no `.nl` | a hole in the table found 40 minutes in |
| build dir configured from another checkout | the rows would name one checkout and measure another |
| `--no-build` with no runner binary | nothing to run, found at the first solve |
| `--instances` without `--out`/`--trace-out`/`--staging-dir` | a debug subset would truncate a fifty-row table, or leave short-budget rows for the next run's resume to publish |
| `--instances` with `--out` resolving to `comparison.csv` | same, via a relative path the explicit-`--out` guard would otherwise wave through |
| a whole-roster run writing exactly one of the two published artifacts | `--out` moved off `comparison.csv` with `--trace-out` left at its default would replace the published anytime trace at exit 0 while reporting a scratch table; the converse republishes `comparison.csv` at this engine beside a trace from the previous one. Refused in both directions (#149) |
| build dir configured with `CBLS_SANITIZE` or `CBLS_PROFILE` | a sanitizer or profiling build measures a different engine, and both are sticky cache entries a later flag-less `cmake -B build` keeps |
| a staging directory stamped with another commit, budget, seed or host | resuming it would mix two configurations into one table |
| a `--seed` other than 1 naming `comparison.csv` or `anytime_trace.csv` | the published table is the pre-registered seed alone; another seed written there would make the published seed whichever ran last (#141) |
| a `--time-limit` other than 60 onto the default paths | it would publish into `comparison.csv` or `comparison_seeds.csv`, replacing a 60s table or seed block with a smoke run; a non-default budget needs a scratch `--out` |
| `--inst-dir` naming a roster directory under `benchmarks/instances/` other than this one (e.g. `heldout/`) without scratch `--out` and `--staging-dir`, or any scratch output landing under `benchmarks/instances/` | the held-out roster publishes nothing (HELDOUT.md), and another roster's scratch output must not reach this directory's tables (#144) |
| another timed driver (`run_benchmark.py` or `run_ablation.py`) holding the wall-clock lock | two timed runs sharing the machine halve each other's iteration counts with nothing in either record saying so |
| `--trace-out` naming `anytime_trace.csv` while `--out` is scratch | the published trace would describe a run the published table does not |
| a `comparison_seeds.csv` with another runner's columns, or an unreadable `comparison_seeds.run.json` | this run's seed could not be added beside them, found only after the solves |
| missing `scip_baseline.csv` (unless `--no-merge`) | the merge would publish `comparison_all.csv` without SCIP rows |
| a roster instance whose read, build or solve **throws** (runner exit 3) | its row carries `read-error`/`build-error`/`solve-error` and measures nothing, so publishing it would put a non-result in the table. The staged row is *structurally* complete, so the resume check refuses it by name (#153) and a re-run hits the same throw rather than skipping past it. A coverage gap — `unsupported`, or a missing `.nl` — exits 0 and does **not** trip this |

It also rebuilds the runner target itself, so the binary cannot lag the SHA it
is about to be labelled with. `--dry-run` prints the plan and exits non-zero if
any refusal applies, so it works as a precheck.

**Resumable.** Each instance is solved in its own process into
`$XDG_STATE_HOME/cbls/minlplib-rerun/<commit>/<budget>s/seed<N>/<instance>.csv`
— by default `~/.local/state/cbls/minlplib-rerun/<commit>/60s/seed1/...` (plus `.trace.csv` and a `.log` holding
that instance's runner output and tally). A re-invocation skips instances that
already have a *complete* staged row. Incomplete means any of: a header-only
file (what a killed job leaves behind), a torn last line, a row stamped with a
different `commit_sha`, a missing `.trace.csv`, or a row whose note says the
runner **threw** (`read-error`/`build-error`/`solve-error`) — all are re-solved.
The last is there because such a row is whole in every other respect: the run
that produced it aborted the driver, and without this the next resume would skip
the instance and publish a row that measured nothing.
The directory's `stamp.txt` records the commit, budget, seed, host and a hash
of `bounds.csv` (the roster) its rows belong to, and a resume against a different one is refused outright rather than
silently mixed. `--no-resume` forces a full re-solve.

**The staging directory lives outside the checkout** (#141). It used to default
to `build/minlplib-rerun/`, which pre-push's clean step (`rm -rf build`) deletes
— a push in the middle of a campaign threw away every staged instance and its
log. A gitignored directory inside the checkout would not do either: the
worktree workflow deletes whole checkouts after a merge. XDG's *state*
directory is the one meant for data that must survive a restart (its *cache*
directory may be emptied at any time). It is keyed by commit and budget, so a
new engine commit starts a fresh directory rather than tripping the stamp; old
ones accumulate there and are safe to delete once their run is published.
`--staging-dir` still overrides it. **One timed driver runs at a time on the
machine**: this driver and the ablation driver (`run_ablation.py`) both hold
`benchmarks/common/jobs.py`'s `wallclock_lock` — one fixed file,
`~/.local/state/cbls/wallclock-<host>.lock`, whatever `--staging-dir` or
`$XDG_STATE_HOME` say — for the whole run, and a second one refuses with
"refusing to run" and exit 2. So two seeds started in two terminals cannot share
the machine or race on the per-seed table. The lock is per user; another user's
run on the same box is not excluded by it, and the host in the name keeps a
home directory shared between machines from making one machine's run refuse
another's.

`comparison.csv` and `anytime_trace.csv` are only replaced at the end, by an
atomic rename of a fully-assembled file, so an interrupted run leaves them
byte-for-byte intact. `comparison_all.csv` is the exception — the merge step
rewrites it in place. If the merge fails after `comparison.csv` is written the
driver says so and exits 1, leaving `comparison_all.csv` on its *previous* CBLS
rows; just re-run, which reuses the staged rows and goes straight back to the
merge.

`elec25`/`elec50` **stay in the roster** — `bounds.csv` is the roster of record
and #123 asks for 50 instances — but their rows are published as documented
failures and are excluded from every quality aggregate and quality claim, per
[#87](https://github.com/spoorendonk/cbls/issues/87) ("Do not publish `elec`
rows until #110 lands and #116's criterion can actually be checked" — #110 has
since closed as not planned and #116/#117 as completed, with the elec criterion
retired rather than met, so no open issue tracks the failure). They are
still counted in **roster counts** — roster size, built, feasible, infeasible,
wall-clock totals, the SCIP head-to-head counts — because the roster of record is
the whole table. That rule is stated once, as `AGGREGATION_RULE` in
`benchmarks/minlplib/campaign_report.py`, and both the driver's summary and the
report generator apply and print it (#142; before that the two used different
denominators). **If an `elec` row comes back
feasible with a finite objective, stop and read the closed history in
#110/#116/#117 before publishing anything about it** — that would be a result, not a routine table refresh.

### Seeds, and what each publish records

**`comparison.csv` is one pre-registered seed, seed 1, and only seed 1 may write
it** (#141). Its job is an A/B against the previous table in which only the
engine differs, so the seed is fixed before the run by the driver itself: a
`--seed` other than 1 is refused if it names `comparison.csv` or
`anytime_trace.csv` by `--out`/`--trace-out`, and by default writes neither. A
median-of-N table would change the selection rule along with the engine, and
"median" is not even defined for a whole table (by feasible count? by aggregate
gap? by anytime score?), so there is none.

**Every seed is published beside it, in `comparison_seeds.csv`** — the runner's
columns with `seed` prepended, one block per seed. Every whole-roster run onto
the default paths adds its seed's rows there, seed 1's included, replacing that
seed's earlier block and leaving the others' rows intact. **Only the published
protocol's 60s budget may publish**, into either table: a `--time-limit` other
than 60 onto the default paths is refused (pass a scratch `--out`), so a smoke
run cannot replace a seed's 60s block. The driver runs one seed per invocation;
check `uptime` before starting the loop, since every seed's record captures the
load average at its own start:

    for SEED in 1 2 3; do
        .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --seed "$SEED" || break
    done

**Run every seed at one commit, and edit nothing between them.** Seed 1's
publish leaves `comparison.csv`, `anytime_trace.csv` and `comparison_all.csv`
modified in the working tree. The driver's dirty-tree refusal does not count the
tables it publishes and their run records (`run_benchmark.run_commit_sha`), so
seed 2 runs at the same plain SHA. Any other modified tracked file still makes it refuse: a
regenerated README or `campaign_summary.json`, an edited `analysis_notes.csv`
(which the runner merges into every row's note), a test, any source. So do not
start "After the run" steps 2-5 until the last seed has published, and do not
commit between seeds: a commit changes the SHA, and the multi-seed summary only
aggregates the seeds published at one commit. A refusal names the files that
tripped it.

A seed other than 1 also skips the `comparison_all.csv` merge, and assembles its
own table and trace inside its staging directory
(`comparison.assembled.csv`, `anytime_trace.assembled.csv`). **Only rows are
published per seed**, not traces: the per-seed anytime traces stay in the
staging directory, and nothing aggregates anytime scores across seeds yet. A
scratch `--out` or `--trace-out` that resolves to any published file — or whose
run record would — is refused at every seed.

**Three seeds is the floor** for quoting a spread — no power calculation
supports a larger number, and the ablation campaign on this roster settled on
three as well; more are welcome but not a blocker. After each publish the
summary prints, across the per-seed table's seeds at this run's commit, budget
and host, the feasible count's min / median / max, each instance's `gap_to_bks%`
median and range, and any seed left out and why (another commit, another
budget, another host, no run record, a record for another commit than its
rows, or rows changed since their record was written). The definitions are `campaign_report.py`'s, applied per
seed (`summarize_seeds`, rule `SEED_AGGREGATION_RULE`): a seed that did not
reach feasibility counts as worse than any gap rather than being dropped, and
`elec25`/`elec50` count in the feasible spread and stay out of quality claims as
everywhere else. `campaign_report.py --seeds` prints the same summary from the
committed files, at the (commit, budget, host) the most seeds share — ties to
the most recently published — not merely at the latest record's. **Report medians and ranges in prose**, beside the published
table; never substitute one for it.

**Every published set carries a run record**: `comparison.run.json` beside
`comparison.csv`, and `comparison_seeds.run.json` beside the per-seed table
(one entry per seed). Each names the commit, budget, seed and roster size, the
machine — host, CPU model, CPU count and usable cores, memory, load average at
the start of the publishing invocation (`benchmarks/common/provenance.py`'s
`machine_record()`) — the concurrency (one solve at a time, one thread each,
held true by the wall-clock lock), `resumed`: how many rows were staged by an
earlier invocation, whose machine and load the record did not see, and
`table_sha256`: the hash of the table's bytes (of the seed's block, in the
per-seed record). A reader refuses a single-seed record whose hash no longer
matches its table, and leaves out a seed whose block does not match — a table
rewritten after its record, or a kill between the two writes. A scratch `--out`
gets a record too, beside it. `campaign_report.py` reads the
budget, seed and machine from `comparison.run.json` when it exists, so
`--budget`/`--seed`/`--machine` are needed only for a table published before
it. The committed table has one (#123), so its provenance block below is read
from the record.

### After the run

These steps are for after the **last** seed (see "Seeds, and what each publish
records"): steps 2-5 modify tracked files the driver does not own, and a later
seed refuses the dirty tree that leaves.

1. `git diff benchmarks/instances/minlplib/` — expect changes confined to
   `comparison.csv`, `anytime_trace.csv`, the `cbls` rows of
   `comparison_all.csv`, `comparison.run.json`, and seed 1's block of
   `comparison_seeds.csv` with its `comparison_seeds.run.json` entry (plus each
   further seed's, once run). The `published-bks` and `scip` rows are engine-independent
   and must be byte-identical; if they moved, something re-solved SCIP and the
   run must be redone.
2. Regenerate every run-derived number below. Nobody transcribes them (the
   one exception is the multi-seed spread, which step 7 writes into prose): every
   derived table and paragraph in this README sits between
   `<!-- campaign_report:begin NAME -->` / `end` markers, and the report
   generator rewrites those blocks from the committed tables and solves
   nothing:

       .venv/bin/python3 benchmarks/minlplib/campaign_report.py \
           --write-readme benchmarks/instances/minlplib/README.md \
           --json benchmarks/instances/minlplib/campaign_summary.json

   Review the diff. The budget (also the anytime score's horizon), the seed and
   the machine are read from `comparison.run.json`, which the driver writes at
   publish time (#141). Pass no `--budget`, `--seed` or `--machine` then: a
   stated value the record contradicts, and any `--machine` at all, is a
   provenance warning; the record wins, and the rewrite is refused, not warned
   about, on any provenance warning or when a documented failure came back
   feasible. A record for another commit than the table's rows is refused
   outright. Tables published before the record existed had to be rendered
   with `--budget 60 --seed 1` stated by hand; the output says which values
   were stated. `--check-readme` instead of `--write-readme`
   exits 1 naming any stale block, and
   `test_the_committed_readme_blocks_are_the_generators_rendering` fails the same
   way, so a hand edit inside a block, or a regenerated table without a
   regenerated README, is caught; `test_the_committed_summary_json_is_the_generators`
   does the same for `campaign_summary.json`. `README_BUDGET` / `README_SEED` /
   `README_MACHINE` in that test file are `None` since #123, so it renders from
   the record, as the command above does; keep them so. Prose outside the
   blocks interprets the numbers rather than restating them; anything numeric
   left there is a separate measurement or a reference value, of a kind the generator's "Not
   regenerated" list names. Re-read that prose against the new blocks — which
   instances moved, and why, is still yours to write.
   - **The instances the published run left infeasible**: the root-cause
     table, if the generated list above it changed. Drop a row there and in
     `analysis_notes.csv` together, and only once the regenerated
     `comparison.csv` actually shows the instance feasible — as #123 did for
     `nvs01` and `st_e40`.
   - The SCIP side of the blocks is engine-independent and must not move.
3. Delete any staleness paragraph in **Results** that the new numbers
   retire (#123 deleted "The table predates #120 and #102").
4. **Attribute any material movement to the engine changes, not to an
   algorithmic improvement.** List them with
   `git log --oneline <old commit>..<new commit> -- src include` and name the
   trajectory-changing ones; **Results** carries #123's attribution against
   `21086c2+107` as the worked example. A row that moved by less than its own
   across-seed range is not evidence of anything.
5. `analysis_notes.csv` is merged into the note column by the runner. A root
   cause that no longer applies has to be edited there, not in `comparison.csv`.
   The runner merges the curated note only onto rows that came back infeasible;
   on a noted instance that came back feasible and verified it warns "stale
   analysis note (now solved)" and marks the row instead (`note_policy.h`, made
   reachable in `09097de`). That warning is per run and easy to miss in a
   50-instance log, so still diff the noted instances
   (every instance with a row in `analysis_notes.csv` — since #123 only
   `elec25` and `elec50`) against **every seed's** rows by hand: any engine
   change can retire one. A noted instance that came back feasible carries
   `stale-analysis-note` in its note cell; #123 retired `nvs01` and `st_e40`
   that way, feasible on all three seeds.
6. Confirm the provenance the whole re-run is for:
   `.venv/bin/python -c 'import csv,sys;print(sorted({r["commit_sha"] for r in csv.DictReader(open("comparison.csv"))}))'` must print exactly one
   SHA, and that SHA must be the checkout you built. Two SHAs mean a resumed run
   spanned a commit; discard the staging directory and re-run.
7. Check the machine in `comparison.run.json`, and in **every** seed's entry of
   `comparison_seeds.run.json` — host, CPU model, cores, memory, and a
   start-of-run load average near zero — or, for a seed started straight after
   another, about the previous solve's 1.0 decaying, which is that solve rather
   than a tenant; say which in **Results**. Nothing to transcribe: step 2's
   provenance block reads seed 1's from `comparison.run.json` (`--machine`
   exists only for a table with no record). If a load average says the box was
   not quiet, that seed is not publishable; a nonzero `resumed` means earlier
   invocations solved some rows, and their load is not in the record. The other
   seeds ("Seeds, and what each publish records"; at least two more) have
   already run by now; write each instance's median and range from the printed spread into the prose
   of **Results** — beside the published table, never in place of it.
8. Run the Python suite:
   `.venv/bin/pytest tests/python/test_minlplib_scip_baseline.py tests/python/test_minlplib_campaign_report.py`.
   The second goes red on any regenerated table until step 2's README blocks
   are rewritten and `test_the_committed_tables_reproduce_the_readme`'s pinned
   numbers are updated to match them.
   `test_safe_gap_reproduces_the_cpp_runners_published_gap_column` reads every
   seed's rows of the regenerated `comparison_seeds.csv` and requires at least
   15 distinct instances with enough resolution to cross-check (a row within
   0.01% of BKS has none), a zero-BKS one among them, so a new table can
   legitimately turn it red. Then
   run the full gated suite from `CLAUDE.md`'s **Build & Test** before pushing.

## Results

<!-- campaign_report:begin provenance -->
Latest run: **60s per instance, seed 1, feasibility tolerance 1e-6**, engine commit `4524460` (recorded per row in `comparison.csv`; the budget, the seed and the machine are read from `comparison.run.json`, which the driver writes at publish time; machine: simon-Legion-5-Pro-16ACH6H (AMD Ryzen 5 5600H with Radeon Graphics), 12 CPUs (12 usable), 13.0 GiB RAM, load 0.73 at start; 1 solve(s) at a time, 1 thread(s) each).
<!-- campaign_report:end provenance -->

The tally below, the gap buckets and the
anytime profile all come from that **one** run; its incumbent trace is committed
as `anytime_trace.csv`, so every number in this section is reproducible from the
checkout it was measured at, which each row's `commit_sha` names.

**The published table is one pre-registered seed, seed 1, of a three-seed
campaign** (#123) at engine commit `4524460`; seeds 2 and 3 are in
`comparison_seeds.csv`, and their spread is written out under "Across seeds"
below, beside this table and never in place of it. The budget is wall-clock, so a
fixed seed does not pin the iteration count and consecutive runs of the same
binary differ: at an earlier commit two runs at one seed gave 46 and 45
feasible, and re-running the *unmodified* binary at the same seed and budget
moved `kall_ellipsoids_tc02b` from 55.1% to 78.2%. Treat any single row as one
draw, not a measurement. A deterministic budget IS available:
`--no-time-limit --max-iterations N` disables the wall clock and bounds the run
by GLS iterations alone, so a given seed reproduces bit for bit (#136). It is not
what the published rows were measured under — they are wall-clock runs at 60s —
so an iteration-budgeted re-run is a different measurement, not a replication of
this table.

<!-- campaign_report:begin tally -->
| | count |
|---|---|
| roster | 50 |
| parsed and built (closed-model rate) | 50 (100%) |
| of which mixed-integer (integrality enforced) | 15 |
| **feasible** | **48** |
| — matching BKS (within the tie band) | 28 |
| — better than BKS, but inside the tolerance slack | 0 |
| — worse than BKS | 20 |
| — better than BKS | 0 |
| infeasible | 2 |
| unsupported / read errors / non-finite | 0 |
| integrality mismatches vs catalogue | 0 |
| verification failures | 0 |

The four verdict rows are over the 48 feasible claim-set rows; `elec25` and `elec50` are excluded from them per the aggregation rule below.
<!-- campaign_report:end tally -->

<!-- campaign_report:begin rule -->
**Aggregation rule.** Documented-failure instances (elec25, elec50) are INCLUDED in roster counts (roster, built, mixed-integer, feasible, infeasible and the infeasible list, coverage gaps, errors, non-finite, integrality mismatches, verification failures, wall-clock totals, the cumulative-feasibility profile and its late-feasible list, the SCIP head-to-head counts and the disjoint-failure list) and EXCLUDED from quality aggregates (verdict-vs-BKS breakdown, gap buckets and the zero-BKS split, the earlier-margin and single-band examples, improvement timing, anytime scores, both-solved quality buckets, the CBLS-ahead-of-SCIP list, and the eligible and within-10% columns of the free-variable split), per #87. Their per-instance rows are still listed, marked excluded.
<!-- campaign_report:end rule -->

**Denominators.** Every count and percentage in this README follows the rule
above, which is stated once, as `AGGREGATION_RULE` in
`benchmarks/minlplib/campaign_report.py`, and printed by both the driver's
summary and the report generator: `elec25`/`elec50` are in every *roster* count,
so "48 feasible" is of 50, and out of every *quality* figure, so the verdict rows
are over the 48 feasible claim-set rows and the anytime score is over 48
instances. The across-seed figures below apply the same rule to each seed
(`SEED_AGGREGATION_RULE`).

Every number inside a `campaign_report` block in this README is generated by
`benchmarks/minlplib/campaign_report.py` from the committed tables, and
`tests/python/test_minlplib_campaign_report.py` fails if a block differs from
what the generator renders. Numbers outside the blocks are separate measurements
(other seeds, other commits, probes), reference values or historical tables; the
generator's "Not regenerated" list names each kind. Denominators follow the
aggregation rule stated above.

<!-- campaign_report:begin gap-buckets -->
Gap distribution over 45 of the 48 feasible claim-set instances: **29 within 0.01% of BKS, 33 within 1%, 36 within 10%.**

Five rows have a numerically zero BKS (`|BKS| < 1e-12`), for which the runner writes an *absolute* residual into the `gap_to_bks%` column rather than a meaningless percentage against zero: `ex14_2_5`, `ex14_2_4`, `mathopt1`, `least` and `prob09`. Those values are not percentages. The buckets above exclude `mathopt1`, `least` and `prob09`, whose residual is non-zero, and retain `ex14_2_5` and `ex14_2_4`, where objective and BKS are both exactly 0 and so are exact matches at any threshold. Excluding all of them instead gives 27 / 31 / 34 over 43 rows. Counting the excluded rows *as* percentages would have put `mathopt1` (gap cell 2.107e-15) and `prob09` (gap cell -3.339e-10) inside the "within 1%" bucket.
<!-- campaign_report:end gap-buckets -->

<!-- campaign_report:begin anytime -->
**Anytime score.** The MIPfeas Primal Integral (`benchmarks/mipfeas/primal_integral.py`) of the committed trace against BKS over the 60s budget — 0 is "at BKS from the first instant", 2 is "never feasible" — over the 48 instances outside the documented failures (`elec25` and `elec50`): **mean 0.180, median 0.011, shifted geometric mean 0.018**. A maximize row's trace is negated, so its reference is −BKS (at catalogue precision, from `bounds.csv`). BKS is not a proven optimum, so an incumbent past it scores a positive gap; and the zero-BKS rows left out of the gap buckets (`mathopt1`, `least` and `prob09`) are *in* this score, because the scorer's own zero test is 1e-6 absolute. The per-instance scores, with the reference each was scored against, are in `campaign_summary.json`; with one seed, each is one draw.
<!-- campaign_report:end anytime -->

### Across seeds, and what moved since the previous table (#123)

**How the campaign ran.** Seeds 1, 2 and 3, all at engine commit `4524460`,
60s per instance, one solve at a time with one thread, published by
`run_benchmark.py` with `resumed: 0` in every record. A first attempt at
`f0ee7b1` published seed 1 and then refused seed 2 on a dirty tree: seed 1's own
publish had modified the tracked tables, so the documented seed loop could not
work. `658f32b` and `4524460` fixed that — the dirty-tree guard now ignores only
the files the driver itself publishes — and that attempt's output was discarded
and every seed re-run at `4524460`. There, seed 1 ran 10:41-11:31 (local time,
2026-10-02). No other tenant is known in that window — the other project's
session described next reports starting around 11:30 — but the window was not
load-sampled, so that rests on its start-of-run load and that session's own
account rather than on a measurement. Seed 2's first run, 11:31-12:21, shared the
machine with that session (unlocked commit-hook test runs at 11:42 and 11:44, a
CMake configure around 12:00), so it was discarded and re-run with
`--no-resume`, 13:20-14:11. Seed 3 ran 12:30-13:20; its first two attempts were
refused by the machine-wide wall-clock lock. The holder was verified with
`fuser` at the time: the other session's `flock <lock> git push origin main`,
waiting since 11:44, followed by its pre-push `ctest` — the lock working as
designed.

**Machine and load.** One host for all three seeds, as `comparison.run.json`
and every entry of `comparison_seeds.run.json` record it: AMD Ryzen 5 5600H, 12
logical CPUs (12 usable), 13.0 GiB RAM. The 1-minute load average at the start of
each seed's run was 0.73 (seed 1), 0.92 (seed 2) and 0.49 (seed 3); seed 3's
5-minute figure was still 2.64 from the activity that preceded it. None of those
is the "near zero" step 7 asks for. Only seed 2's falls under step 7's
back-to-back case: its re-run started the moment seed 3 finished, so its 0.92 is
seed 3's last solve decaying. Seed 1's 0.73 followed the preflight and the 4-job
runner build, and seed 3's 0.49 (5-minute 2.64) followed the other session's
pre-push `ctest`, which held the wall-clock lock until shortly before it started.
Neither is a tenant during the run, but both are judgements from the timeline,
not records.
During seeds 3 and 2 the load was also sampled every 15s: the 1-minute figure was
above 1.10 for about three minutes of seed 3's window (12:59:50-13:02:50, peak
1.27; by roster order roughly `st_e01`, `chain50`, `st_e40` and `st_e09`, and
no claim rests on seed 3 alone there: `chain50`'s other two seeds are already
worse than the old table) and otherwise at most 1.10, and peaked at 1.36 in seed 2's (around 14:04),
where about 1.0 is the solve itself. Seed 1's window was not sampled.

**Spread.** Feasible: **48 on every seed** (min 48, median 48, max 48).
Matching BKS, over the 48 claim-set rows: **26 / 27 / 28** (min / median / max;
seed 1 28, seed 2 27, seed 3 26). `elec25` and `elec50` are infeasible on all
three seeds, so no documented failure came back feasible.

Per instance below: `gap_to_bks%` of the previous published table, of the
published seed, and the median and range over the three seeds as the driver's
across-seed summary (`campaign_report.py --seeds`) prints them, where a seed that
did not reach feasibility counts as worse than any gap. `(abs)` marks a zero-BKS
row, whose cell is an absolute residual, not a percentage. The last column is
the LNS counters per seed, `lns_repairs` with `lns_repairs_accepted` in brackets.

| instance | old (`21086c2+107`) | seed 1 (published) | median | range | feasible | LNS repairs (accepted), seeds 1 / 2 / 3 |
|---|---:|---:|---:|---|---|---|
| `process` | 10.3 | 0.014 | 0.014 | 0.0125 – 0.0163 | 3/3 | 60 (1) / 63 (1) / 64 (0) |
| `st_e36` | 40.2 | 0.871 | 0.871 | 0.871 – 0.871 | 3/3 | 10 (0) / 10 (2) / 10 (3) |
| `ex4_1_1` | 1.30e-9 | 1.30e-9 | 1.30e-9 | 1.30e-9 – 1.30e-9 | 3/3 | 79 (0) / 64 (1) / 65 (0) |
| `ex8_1_6` | 49.8 | 49.8 | 49.8 | 49.8 – 49.8 | 3/3 | 32 (0) / 32 (0) / 32 (0) |
| `alkylation` | 66.1 | 2.30e-6 | 3.20e-4 | 2.30e-6 – 0.0121 | 3/3 | 54 (0) / 56 (0) / 52 (1) |
| `nvs21` | 100 | 100 | 100 | 100 – 100 | 3/3 | 1 (0) / 1 (0) / 1 (0) |
| `ex4_1_2` | 3.51e-4 | 0.00104 | 8.33e-5 | 7.13e-7 – 0.00104 | 3/3 | 12 (0) / 12 (0) / 12 (1) |
| `ex14_2_5` (abs) | 0 | 0 | 0 | 0 – 0 | 3/3 | 183 (83) / 181 (50) / 183 (41) |
| `mathopt5_3` | -1.81e-8 | -1.81e-8 | -1.81e-8 | -1.81e-8 – -1.81e-8 | 3/3 | 115 (0) / 115 (0) / 115 (0) |
| `ex8_4_5` | 1.2 | 0.231 | 0.0655 | 0.0497 – 0.231 | 3/3 | 7 (1) / 4 (3) / 5 (1) |
| `nvs01` | infeasible | 1.26e-8 | 266 | 1.26e-8 – 266 | 3/3 | 1 (1) / 1 (0) / 1 (0) |
| `ex4_1_3` | -9.26e-9 | -9.26e-9 | -9.26e-9 | -9.26e-9 – -9.26e-9 | 3/3 | 100 (0) / 100 (0) / 101 (0) |
| `ex14_2_4` (abs) | 0 | 0 | 0 | 0 – 0 | 3/3 | 104 (55) / 92 (16) / 34 (7) |
| `ex8_1_1` | -1.78e-8 | -1.78e-8 | -1.78e-8 | -1.78e-8 – -1.78e-8 | 3/3 | 70 (0) / 83 (0) / 83 (0) |
| `shiporig` | 1.9 | 1.64 | 1.64 | 0.00218 – 2.98 | 3/3 | 5 (0) / 4 (0) / 3 (0) |
| `nvs08` | 1.61 | -9.64e-8 | -9.64e-8 | -9.65e-8 – 1.61 | 3/3 | 1 (1) / 1 (1) / 1 (1) |
| `ex4_1_7` | 0 | 0 | 0 | 0 – 0 | 3/3 | 113 (0) / 113 (0) / 114 (0) |
| `eq6_1` | 20.4 | 2.53 | 2.53 | 2.5 – 2.56 | 3/3 | 0 (0) / 0 (0) / 1 (0) |
| `minlphi` | -7.97e-7 | -8.64e-7 | -8.64e-7 | -1.55e-6 – -7.94e-7 | 3/3 | 17 (0) / 17 (0) / 16 (0) |
| `gear4` | 1.65e6 | 4.29e6 | 4.29e6 | 1.77e6 – 9.48e6 | 3/3 | 10 (2) / 10 (1) / 10 (4) |
| `mathopt5_7` | -1.01e-8 | -1.01e-8 | -1.01e-8 | -1.01e-8 – -1.01e-8 | 3/3 | 86 (0) / 87 (0) / 86 (0) |
| `elec25` (excluded) | infeasible | infeasible | — | — | 0/3 | 22 (0) / 22 (0) / 23 (0) |
| `st_e38` | 1.05 | -6.61e-9 | -6.61e-9 | -6.61e-9 – -6.61e-9 | 3/3 | 9 (1) / 10 (0) / 9 (0) |
| `ex8_1_5` | -4.75e-8 | -4.75e-8 | -4.75e-8 | -4.75e-8 – -4.75e-8 | 3/3 | 40 (0) / 41 (0) / 40 (0) |
| `maxmin` | 0.0678 | 0.00896 | 0.00706 | 5.90e-4 – 0.00896 | 3/3 | 3 (0) / 2 (0) / 2 (0) |
| `nvs02` | 35.2 | 1.17e-9 | 1.17e-9 | 1.17e-9 – 1.57 | 3/3 | 0 (0) / 0 (0) / 2 (0) |
| `ex4_1_8` | 3.40e-8 | 4.34e-7 | -2.31e-9 | -6.69e-9 – 4.34e-7 | 3/3 | 133 (0) / 133 (1) / 133 (0) |
| `ex8_6_1` | 49.1 | 23.9 | 26.9 | 23.9 – 38.7 | 3/3 | 0 (0) / 0 (0) / 0 (0) |
| `nvs14` | 53.4 | 1.73e-9 | 1.73e-9 | 1.73e-9 – 1.73e-9 | 3/3 | 0 (0) / 0 (0) / 3 (0) |
| `st_e01` | 5.00e-9 | 5.00e-9 | 5.00e-9 | 5.00e-9 – 5.00e-9 | 3/3 | 140 (0) / 155 (1) / 158 (0) |
| `chain50` | 14.4 | 19.5 | 35 | 19.5 – 35.7 | 3/3 | 0 (0) / 1 (1) / 2 (0) |
| `st_e40` | infeasible | 1.36 | 0 | 0 – 1.36 | 3/3 | 22 (0) / 20 (0) / 21 (1) |
| `st_e09` | 2.00e-8 | 2.00e-8 | 2.00e-8 | 2.00e-8 – 2.00e-8 | 3/3 | 73 (0) / 74 (0) / 73 (0) |
| `elec50` (excluded) | infeasible | infeasible | — | — | 0/3 | 13 (0) / 13 (0) / 13 (0) |
| `nvs05` | 453 | 1483 | 574 | 177 – 1483 | 3/3 | 0 (0) / 0 (0) / 0 (0) |
| `ex4_1_9` | 0.0086 | 0.00401 | 0.0259 | 0.00401 – 0.0317 | 3/3 | 41 (0) / 48 (0) / 34 (1) |
| `kall_ellipsoids_tc02b` | 54.3 | 37.3 | 37.3 | 0.0756 – 95.3 | 3/3 | 9 (0) / 10 (1) / 9 (1) |
| `nvs22` | 355 | 122 | 173 | 122 – 311 | 3/3 | 0 (0) / 0 (0) / 0 (0) |
| `mathopt1` (abs) | 1 | 2.11e-15 | 2.06e-14 | 2.11e-15 – 1.87e-13 | 3/3 | 39 (19) / 36 (0) / 36 (0) |
| `least` (abs) | 2.33e4 | 1.41e4 | 1.41e4 | 1.41e4 – 1.41e4 | 3/3 | 11 (3) / 11 (2) / 11 (3) |
| `tln2` | 0 | 0 | 0 | 0 – 0 | 3/3 | 10 (0) / 9 (0) / 11 (1) |
| `prob06` | -3.19e-4 | -6.49e-5 | -3.18e-4 | -3.19e-4 – -6.49e-5 | 3/3 | 90 (0) / 90 (0) / 87 (0) |
| `ex6_2_11` | 9.55e-6 | 0.0771 | 2.94e-4 | 7.54e-5 – 0.0771 | 3/3 | 7 (1) / 6 (6) / 5 (3) |
| `spring` | 39.7 | 2.61e-4 | 2.61e-4 | 1.85e-4 – 2.61e-4 | 3/3 | 8 (0) / 8 (0) / 8 (0) |
| `prob09` (abs) | 0.0459 | -3.34e-10 | -3.34e-10 | -7.98e-10 – 1.00e-13 | 3/3 | 68 (12) / 68 (10) / 69 (18) |
| `ex6_2_6` | -8.30e-5 | -5.78e-5 | -6.55e-5 | -8.20e-5 – -5.78e-5 | 3/3 | 4 (3) / 7 (7) / 4 (4) |
| `windfac` | 195 | 1.41e-8 | 1.41e-8 | -2.53e-4 – 1.41e-8 | 3/3 | 16 (0) / 15 (2) / 15 (2) |
| `st_e08` | 2.17e-4 | 0.00102 | 0.00219 | 0.00102 – 0.00219 | 3/3 | 100 (0) / 98 (0) / 100 (0) |
| `ex6_2_8` | 7.62e-5 | 1.27e-4 | 1.27e-4 | 1.05e-4 – 1.81e-4 | 3/3 | 6 (0) / 6 (0) / 6 (0) |
| `eg_all_s` | 10.5 | 88.3 | 3.43 | 3.43 – 88.3 | 3/3 | 0 (0) / 0 (0) / 0 (0) |

**What moved since the previous table.** The previous table was one seed at
`21086c2+107` — the engine at `21086c2` with #107 applied, committed in
`0648067`. Against it:

One rule sorts the claim-set rows, applied to `gap_to_bks%` with an infeasible
row counting as worse than any gap: a row is **better** when the old value is
worse than every one of the three new seeds, **worse** when it is better than
every one, and otherwise **inside the new range**, which says nothing either way.
Rows labelled `matches-bks` in the old table and on all three seeds are left out
as unchanged, whatever their sub-tie-band digits did.

- **Feasible 46 → 48.** `nvs01` and `st_e40` are feasible on every seed. Both had
  already been measured solving after #102 (`st_e40` at `b7f8a50`, `nvs01` at
  `1559786` and `eb9e1a5`).
- **Better:** `process` (10.3% → 0.012-0.016%), `st_e36` (40.2% → 0.87%),
  `alkylation` (66.1% → at most 0.012%), `nvs14` (53.4% → BKS), `st_e38`
  (1.05% → BKS), `windfac` (195% → BKS), `spring` (39.7% → at most 2.6e-4%),
  `nvs02` (35.2% → BKS on two seeds, 1.57% on the third), `eq6_1`
  (20.4% → 2.5-2.6%), `ex8_6_1` (49.1% → 23.9-38.7%), `nvs22` (355% → 122-311%),
  `ex8_4_5` (1.20% → 0.05-0.23%), `maxmin` (0.068% → 5.9e-4-9.0e-3%), the two
  newly feasible rows above, and the zero-BKS rows `mathopt1` and `prob09`
  (absolute residuals 1 and 0.046 → at most 1e-9) and `least` (23349 → ~14086).
  Some of these moved far further than the new seeds spread (`process`,
  `alkylation`, `windfac`); others, such as `ex8_6_1` and `nvs22`, sit beyond the
  new worst seed by less than the new seeds differ among themselves, so the old
  single draw is the weaker side of that comparison.
- **Worse:** `chain50` (14.4% → 19.5-35.7%), `gear4` (1.65e6% → 1.77e6-9.48e6%)
  and `st_e08` (2.2e-4% → 1.0e-3-2.2e-3%, which no longer matches BKS).
- **Inside the new range:** `eg_all_s` (10.5% → 3.43 / 3.43 / 88.3%; the
  published seed is the outlier), `nvs05` (453% → 177-1483%),
  `kall_ellipsoids_tc02b` (54.3% → 0.08-95%), `nvs08`, `shiporig`, `ex4_1_2`,
  `ex4_1_9` and `prob06`. `ex8_1_6` and `nvs21` are identical on every seed and
  to the old row.
- **Wall time:** `eg_all_s` 61.1s → 60.0s, so the roster total is 3000s rather
  than 3001s. Two changes in the range would each stop an instance that used to
  overrun its budget — #113's bounded deadline stride and #191's stop of the
  inner-solver hook at the deadline — and nothing here separates them.

**Attribution: engine changes, not an algorithmic improvement.** 291 commits
touched `src/` or `include/` between `0648067` and `4524460`
(`git log 0648067..4524460 -- src include`). Those that change the search
trajectory on this runner's single-threaded path:
[#111](https://github.com/spoorendonk/cbls/issues/111) (the diversification kick
gained a structural half), [#112](https://github.com/spoorendonk/cbls/issues/112)
(one guarded randomisation helper),
[#113](https://github.com/spoorendonk/cbls/issues/113) (the FJ deadline stride is
bounded in time and iterations), [#114](https://github.com/spoorendonk/cbls/issues/114)
(an unbounded Int gets FJ jump candidates instead of freezing),
[#120](https://github.com/spoorendonk/cbls/issues/120) (implied variable bounds by
propagation, replacing the fixed inf-clamp) — the five this issue named — plus
#100/#116 (infinite bounds and a non-finite first feasible point), #105 (the
STRUCTURAL batch is bounded by the wall clock), #108/#109 (one initialisation
path; no no-op kicks), #117 (escape-probe arming on wall-clock stagnation), #102
(the unproductive-batch exit), #158 (kick from the incumbent), #186 (breakpoint
jump candidates through `Ceil`/`Floor`/`Round`/`Element` and differentiable ops),
#188 (incremental Sums within a tracked drift bound), #190 (closed-form scoring
of bare affine bodies) and #191 (the inner-solver hook stops at the deadline).
Separately, throughput work — a sublinear `delta_evaluate`, closed-form scoring
over linear rows (which agrees with the DAG probe to rounding, not to the bit),
an incremental violated-row set, lazy GLS decay (#175) and cone-restricted
reverse-mode AD — changes how many iterations 60s buys, which by itself moves
every row of a wall-clock-limited table. No per-change ablation was run, so no
row here is attributed to a single change except `st_e40` and `nvs01`, where
#102 was measured beforehand. Read the table as the sum of fixes, bound handling
and throughput between two commits, measured at one budget over three seeds —
not as evidence that the search got better in the sense #87 asks about.

**LNS repairs** (the counters #123 asked for; the `lns_repairs` and
`lns_repairs_accepted` columns of `comparison.csv` and `comparison_seeds.csv`,
per seed in the table above). Over the 150 runs, LNS ran 5716 destroy-repairs
and its accept rule kept 382 (6.7%). Acceptance is concentrated: 292 of the 382
are on three zero-BKS instances (`ex14_2_5` 174, `ex14_2_4` 78, `prob09` 40),
25 instances kept at least one repair on some seed, and four — `ex8_6_1`,
`nvs05`, `nvs22` and `eg_all_s` — ran no repair on any seed. A kept repair is
one the lexicographic (violation, objective) key accepted, not one shown to have
improved the final row: these counts say where LNS is engaged at 60s, not what it
contributes, which is the ablation campaign's question (#143).

<!-- campaign_report:begin two-band -->
Nothing in this roster beats a published bound. Under the runner's earlier margin rule — which compared a *percentage* against 1e-6, i.e. 1e-8 relative — two rows of this run would have been flagged `better-than-bks`: `ex6_2_6` at 5.8e-5 percent and `prob06` at 6.5e-5 percent. Those are ties, not improvements.

Two bands are used, deliberately different. An improvement is only *claimed* when it exceeds `max(1e-6·(|BKS|+1), 10·feas_tol)`: we accept solutions violating a constraint by up to `feas_tol`, and that slack itself buys a small objective gain. A *tie* requires the much tighter, purely relative `1e-6·(|BKS|+1)` — using one band for both would have published `spring` (BKS 8.46e-1) as matching BKS when it was 2.6e-4% worse and `st_e08` (BKS 7.42e-1) as matching BKS when it was 1.0e-3% worse, because at that magnitude the absolute floor dwarfs the relative tie band. A row that improves on BKS by more than the tie band but less than the claim threshold falls between the two and is labelled `within-tolerance-of-bks` rather than being miscounted as worse.
<!-- campaign_report:end two-band -->

### Why 60s

<!-- campaign_report:begin feasibility -->
Measured from the committed trace, not assumed. Cumulative instances with a feasible solution by time t (of 50):

| by | 1s | 5s | 10s | 20s | 30s | 45s | 60s |
|----|----|----|----|----|----|----|----|
| feasible | 44 | 47 | 48 | 48 | 48 | 48 | 48 |

**What a 5s budget would lose.** One instance reaches feasibility only after 5s — `eg_all_s` (5.5s) — so a 5s budget would publish it as infeasible, 47 solved instead of 48. Which instance varies between draws.
<!-- campaign_report:end feasibility -->

<!-- campaign_report:begin improvement -->
Solution *quality* over time is a weaker argument than it first appears, and is recorded here with that caveat. Of the 48 claim-set instances that become feasible, 40% stop improving within the first second while 23% are still improving in the final 15 seconds — reading an improvement as a strict decrease of the trace's *printed*, six-significant-digit objective. Read off the engine's own `new_best` flag instead, the split is 33% / 23%. The two part because 747 of the trace's `new_best` rows print the same objective as the row before: improvements below the trace's print resolution, which the printed reading cannot see and the flag counts. 686 of the 747 are on `least`.

But the incumbent trace cannot be read as pure search progress: `record_best` tightens the objective bound by `1e-3·(|obj|+1)` per accepted solution, so improvements are *floored* at roughly 0.1% steps. The measured median consecutive-incumbent ratio on `eg_all_s` (the instance with the most improvements) is 0.9989993 — 1 − 0.001001, against the bound step's 1 − 1e-3 — and it takes 15892 such steps (15893 incumbents) to walk from 1e9 down to 14.4.
<!-- campaign_report:end improvement -->

That instance is therefore evidence about the bound-tightening step size, not about
how long the search needs. Read the quality column as a lower bound on what a
larger step (or a direct objective descent) might achieve sooner.

**At `4524460` the feasibility argument for 60s has largely gone.** Under the
previous table five instances first became feasible between 17s and 37s, and that
was the load-bearing reason for the budget. Now every instance that becomes
feasible does so within 6.5s on every seed (`time_to_first_feasible`: last are
`eg_all_s` at 5.5s on seed 1, `chain50` at 1.5s on seed 2, `nvs02` at 6.5s on
seed 3). 60s stays the protocol budget for reasons that do not depend on this
run: #123's table is an A/B against the previous one in which only the engine may
differ, the SCIP baseline is also 60s, and quality is still moving late (the
block above). Choosing a shorter budget from this run's own numbers would fit the
protocol to the result; a budget change is a separate, pre-registered experiment.

### The instances the published run left infeasible

<!-- campaign_report:begin infeasible -->
Left infeasible in this table: `elec25` and `elec50` (2 of 50).
<!-- campaign_report:end infeasible -->

Both are root-caused, and the verdict is recorded per row in `comparison.csv`
(merged from `analysis_notes.csv`). The published note cells, in `comparison.csv`
and `comparison_seeds.csv`, carry the note **as it was merged at run time**,
which still calls #110 the tracker; `analysis_notes.csv` and the table below say
#110 is closed. The cells are left as published because each run record pins its
table's sha256, and `campaign_report.py` refuses a table (or leaves out a seed)
whose bytes no longer match — editing a cell would unpublish the run. They are infeasible on all three of #123's
seeds, as they were in the previous table.

**`nvs01` and `st_e40` left this list in #123's regeneration**: feasible on all
three seeds (see "Across seeds" above), as #102 had already been measured to
make them. Their root-cause rows are retired from the table below and from
`analysis_notes.csv` — the run marked both rows `stale-analysis-note` on every
seed — and the mechanisms are kept under "Retired root causes" after it, since
the `nvs01` section and the #102 section below still lean on them.

| Instance | Verdict | Cause |
|----------|---------|-------|
| `elec25`, `elec50` | **bug**, untracked ([#110](https://github.com/spoorendonk/cbls/issues/110) closed as not planned on 2026-09-07; its elec criterion had been retired to #116 and #117, both closed) | Thomson problem: points on the unit sphere, Coulomb objective `+inf` wherever two coincide. **The objective-encoding defects (#100) are fixed and are no longer the blocker.** What remains: the `.nl` declares no finite variable bounds, so the box is the ±1e9 inf-clamp; random init starts ~1e9 out, and shrinking each variable toward 0 is a huge row improvement — which parks the search on the origin, a stationary point of every row `x²+y²+z²=1`. The Float jump offers a single *undamped* Newton step (`x0 - residual/grad`) plus `lb`/`ub`/midpoint; near the origin that step overshoots wildly and is rejected, and because a candidate was nonetheless *generated* the #107 escape probe is suppressed — so the variable freezes at score 0. Measured: escape probe fires only at exactly `x0 = 0`; at `x0 = 0.001/0.01/0.1` the score is 0 with the probe armed or not. Infeasible at violation ≈1 **both with the objective present and with it neutralised**, so it is not objective-related — the earlier "dropping the objective makes elec25 feasible in 20s" claim no longer reproduces. Tightening `inf_clamp` to 1 makes `elec25` feasible at violation 0 post-#100 (pre-#100 it was infeasible at *every* clamp), because clamping accidentally supplies the missing damping. |

#### Retired root causes

**`nvs01`** ([#101](https://github.com/spoorendonk/cbls/issues/101); was `hard`,
then **superseded**). `420.169·√(x0²+900) == x2·x0·x1` needs `x0` and the
product `x1·x2` changed together. While `x0 = 0` the product term vanishes, so
`x1` and `x2` receive no gradient signal and no single-variable jump improves —
the analysis that concluded escaping requires a compound move. **That conclusion
does not survive #102.** Re-checked at engine commit `1559786` with
`use_compound_moves` still `false`: `nvs01` is feasible on all four of seeds
42/1/2/3 at 10s (gaps 30.16 / 12.23 / 734.94 / 792.13 %), and at the issue's own
60s budget it reaches the BKS exactly on seeds 42/1 and lands at 169.83 / 266.00
% on seeds 2/3. A single-variable path reaches the feasible region, so the
compound-move account is **refuted**, not merely incomplete, and #101 is closed
as superseded. What is left is objective quality and its seed-to-seed spread,
now tracked as [#134](https://github.com/spoorendonk/cbls/issues/134) — not an
escape failure, and not fast-verifiable: the noise floor recorded below is 3-4
gap points on wall-clock runs, so any criterion there is a multi-seed timed run.
The incumbent on seeds 2/3 *walks* (104.1→33.6 and 111.2→45.6 between 10s and
60s) while the 10s figures are seed-deterministic to the digit across two engine
commits, which points at a convergence-rate limit rather than at run-to-run
randomness. **That hypothesis has since been tested and partly refuted — see the
dedicated `nvs01` section below**: measured over eight seeds at `09097de`, the
spread is seed-deterministic and is set at *first feasibility* (r = 0.945), not
accumulated during the descent. Note on provenance: `b7f8a50` is the pre-rebase
tip of `issue-102-ex8_6_1` and `1559786` is its rebased equivalent, so `git diff
b7f8a50 1559786 -- src include benchmarks/minlplib` is **empty** — the two arms
share identical engine source and running both proves nothing about drift. An
earlier revision of this row claimed otherwise. The determinism that *is*
evidenced comes from re-running seeds 42/2/3 at `6f9d419`, a genuinely different
tree, which reproduced 30.1611 / 734.942 / 792.13 to the digit.

**`st_e40`** ([#102](https://github.com/spoorendonk/cbls/issues/102); was an engine
gap, not hardness — **fixed**). Rows C1–C3 are degree-7 polynomials
`(x-1)(x-2)(x-3)(x-5)(x-8)(x-10)(x-12) == 0` restricting each integer to
`{1,2,3,5,8,10,12}`; C0 pins the free `x3` to a bilinear function of them, and
four linear rows bound the combination. That leaves 343 integer combinations,
**52 of them feasible**. The search reached only 3 of the 343: it fell into a
limit cycle within ~20 GLS iterations — `x1` flipping between two values and
`x3` hopping between the roots of the four rows containing it — and neither
trapped value appears in any feasible combination (all 52 need `x0 >= 5` and `x1
>= 5`). It then spent **92% of a 155 000-iteration run on GLS weight bumps that
changed nothing**, because diversification was gated on 100 non-improving
*batches* of 1000 iterations and, before a first feasible solution, no batch
ever improves — a fixed cadence of one kick per 100 000 iterations with no
feedback from the search. Two earlier explanations in this table were both
wrong: a violation barrier between allowed integer values (`int_jump_candidates`
enumerates the whole domain below 256 values, and these are `[1,12]`), and "the
feasible combination" singular (there are 52). Fixed by
`GFJConfig::unproductive_iterations`: a batch that stops reducing the real rows'
violation ends, and `solve()` takes the diversification kick as due (without
arming the Float escape probe, which stays gated on `perturbation_period` — see
#107). Two things bound it, and the second is what took two rounds to get right:
`solve()` arms the exit only once its own stagnation count reaches
`perturbation_period / 20` batches, and — because that count is monotone across
an unproductive kick, so a threshold only DELAYS a misfire rather than
preventing it — once a feasible solution exists the kick draws only a perturb
and never an LNS destroy-repair. The measure is blind to the artificial
objective row in that phase, and an LNS repair launched on a blind signal is
bounded in seconds and usually rejected; see the objective-quality section below
for what it cost before it was bounded. Measured at engine commit `b7f8a50`,
branch `issue-102-ex8_6_1`: reaches the BKS 30.4142135 at 10s on all four of
seeds 42/1/2/3, and at 60s on seed 42 (both re-measured at that commit, not
carried over from an earlier arm). #123's regeneration confirms it under
the published protocol: feasible on all three seeds, at the BKS on seeds 2 and 3
and 1.36% above it on seed 1.

### `nvs01`: the seed spread is real, and it is set at first feasibility

Measured at engine commit `09097de`, eight seeds (42/1/2/3/7/11/13/17) at the
60s published budget, run serially on an idle machine through the driver's
non-destructive subset mode. This is issue #134's measurement.

**Feasibility is completely reliable; objective quality is not.** All 8 seeds
reach feasibility. Three reach the BKS 12.4697 exactly; the rest spread from
17% to 266%.

| seed | first feasible obj | final obj | gap to BKS |
|---|---:|---:|---:|
| 1 | 16.49 | 12.4697 | **BKS** |
| 17 | 16.76 | 12.4697 | **BKS** |
| 42 | 17.03 | 12.4697 | **BKS** |
| 11 | 18.60 | 14.62 | 17.2% |
| 7 | 117.0 | 22.09 | 77.1% |
| 2 | 186.8 | 33.65 | 169.9% |
| 13 | 236.8 | 41.64 | 233.9% |
| 3 | 176.9 | 45.64 | 266.0% |

Median gap 47.2%, range 0.0-266.0%, BKS on 3 of 8.

**The outcome is determined by where the first feasible solution lands**, not by
what happens afterwards: Pearson `r = 0.945` between the first feasible objective
and the final one, with a clean separation — every seed whose first feasible
point is under ~19 finishes at or near the BKS, and every seed above ~100 never
recovers within the budget.

The anytime traces show why. The incumbent walks down in small, near-constant
decrements, so the number of improvements needed scales with how far out the
search starts: the three BKS seeds needed 16-20 improvements, while the seeds
starting above 100 needed 72-141 and still did not arrive. Three seeds were
still improving when the clock stopped (last improvement at 57.5-58.4s of 60s);
the ones that stalled early either had already converged to the BKS or, in seed
3's case, stalled at 45.64 with 26.5s left.

**This refutes the run-to-run-noise reading and narrows the hypothesis this
benchmark carried.** The spread is not noise — it is seed-deterministic and
reproduces to the digit across engine commits (169.826 and 266.002 appear at
`1559786`, `5f33c59` and `09097de`). It is also not an escape failure: #101
established that, and feasibility is 8/8 here. What remains is a **convergence
rate** limit, and the lever is the quality of the first feasible point rather
than the descent that follows it.

Worth noting for anyone reading this before a campaign:

- A single-seed row for this instance is close to meaningless — it is a draw from
  a distribution spanning the optimum to 266% above it. This is the sharpest
  case in the roster for publishing per-seed results (#141).
- **#123's regeneration, at `4524460`:** seed 1 at the BKS (12.4697), seeds 2 and
  3 both at 45.6392 (266.0%). Seeds 1 and 3 reproduce their `09097de` values; seed
  2 does not — it lands on seed 3's 266.0% rather than its own earlier 169.83% —
  so the to-the-digit reproduction above held across the commits it names, not
  across the engine changes between `09097de` and `4524460`.
- With 3 of 8 seeds reaching the optimum, a portfolio of independent seeds would
  very likely find it. That is a concrete prediction for the scaling study in
  #135, and one of the few places in this roster where the portfolio's value
  should be visible rather than assumed.

### Is the final objective set by the first feasible point? (#149)

The section above measured that on **one** instance. Issue #149 asked whether it
generalises, and forbade any engine change until that was answered — the effect
could easily be an artefact of `nvs01`, which is a three-variable instance with a
product term.

> **Status: #149 is closed with the roster-wide question deferred, and the
> campaign below was NOT run.** Re-measured at `eb9e1a5`, the effect reproduces
> on `nvs01` itself — 8 of 8 seeds feasible, Pearson **r = 0.9399** against
> #134's 0.945 at `09097de`, 106 commits earlier — so it is real and stable on
> that instance, not a one-commit artefact.
>
> **Whether `nvs01` is unusual or typical is undetermined.** Saying it is "an
> outlier" would be a claim about the other 49 instances, and not one of them has
> been measured: the first-feasible columns did not exist when this table was
> written. One instance is one instance, in both directions — it supports no
> claim about the engine in general, and equally no claim that the engine is
> generally fine. No engine change may be made on it either way.
>
> That question was deferred to the campaign that regenerates this table (#123),
> where the per-seed columns come for free rather than costing a dedicated
> 6.7-hour run. **It is still open**: #123 ran three seeds, and the pre-registered
> rule needs at least four usable seeds per instance, so
> `first_feasible_report.py` over seeds 1-3 of `comparison_seeds.csv` returns
> `INCONCLUSIVE` with 0 eligible instances. The protocol below stays as the
> recipe for whoever runs it.

Everything needed is in the tree. The runner publishes
`first_feasible_objective` and `time_to_first_feasible` on every row (recording
them changes no trajectory — see `SearchResult::first_feasible_objective`), and
`benchmarks/minlplib/first_feasible_report.py` scores the roster from those
columns. **What is missing is the campaign**: it is wall-clock-limited, so it
has to be run serially on an idle machine and cannot be run beside anything
else.

**Protocol.** Engine commit: whatever `git rev-parse --short=7 HEAD` reports on
a clean tree — the driver refuses a dirty one, and `--short=7` matches
`benchmarks.common.provenance.commit_sha()`, so the recorded verdict names the same string the
rows' `commit_sha` column carries. Budget: **60s per instance**, the published one.
Seeds: **1 2 3 7 11 13 17 42** — the eight #134 used, so the roster result and
the `nvs01` result are the same measurement on different instances rather than
two protocols. Roster: all 50 from `bounds.csv`; `elec25`/`elec50` are dropped by
the scorer, as they are from every other claim. Serial, one process at a time:
**about 6.7 hours** (50 × 60s × 8, plus per-instance overhead).

Run it from a configured Release build directory on an otherwise idle machine —
check `uptime` first, and re-run anything measured while the load average was
not near zero. **Staging goes outside `build/`**, which pre-push deletes.

```bash
FF=/var/tmp/cbls-149            # anywhere outside the repo and outside build/
mkdir -p "$FF"
for SEED in 1 2 3 7 11 13 17 42; do
    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py \
        --seed "$SEED" \
        --time-limit 60 \
        --out "$FF/seed$SEED.csv" \
        --staging-dir "$FF/stage-seed$SEED" \
        --no-trace || { echo "seed $SEED failed; stopping" >&2; break; }
done
```

The `|| break` is not decoration: the driver exits 2 on a refusal and 1 on an
instance that throws, and without it a dirty tree fires eight identical refusals
in a second, while a throw at seed 3 burns the remaining five hours before the
scoring step fails on a missing file.

Each invocation resumes: a killed run picks up from its staging directory, and
the loop can simply be re-run. Nothing here touches a published table —
`--out` keeps the results in `$FF`, `--no-trace` keeps `anytime_trace.csv`
alone (and the driver refuses either half of that pair without the other,
in both directions). The runner's *own* published-path guard is **not** a second
layer here: every instance is staged to `$FF`, so `--out` never names a published
file and the guard never fires. The driver's `usage_error` is the only thing
between this recipe and the published tables.

Then score it, which takes a second and solves nothing. Invoked **by path**,
like the driver above: the module puts the repo root on `sys.path` itself so that
form works from any directory, whereas `-m benchmarks.minlplib.…` needs the repo
root to already be the working directory.

```bash
.venv/bin/python3 benchmarks/minlplib/first_feasible_report.py \
    --table 1="$FF/seed1.csv"   --table 2="$FF/seed2.csv" \
    --table 3="$FF/seed3.csv"   --table 7="$FF/seed7.csv" \
    --table 11="$FF/seed11.csv" --table 13="$FF/seed13.csv" \
    --table 17="$FF/seed17.csv" --table 42="$FF/seed42.csv" \
    --csv "$FF/first_feasible_per_instance.csv"
```

The report prints, per instance, the across-seed Pearson `r` between the first
feasible objective and the final one and the Spearman rank correlation beside it,
its bucket, and the reason where there is no `r`. Above the table it prints a
global tally of the rows it dropped and why; below it, the verdict.

**The decision rule is pre-registered** — it is written into
`first_feasible_report.py` as `R_DETERMINED`, `MIN_ELIGIBLE_INSTANCES` and
`MIN_SEEDS_PER_INSTANCE`, each with the argument for its value, and it is fixed
before the campaign rather than chosen once the numbers are in:

- an instance is **eligible** when it has at least **4** usable seeds and
  *something* varied across them. Only `no-spread` — the same arrival and the
  same outcome on every seed — is outside the count, because there the
  experiment never varied its input *and* never varied its output;
- an eligible instance where exactly one end held still has no defined `r`
  (zero variance on one axis) and is **not** silent. `final-invariant` — arrival
  varied, every seed finished at the same objective — says the search got to the
  same answer however far out it started. `arrival-invariant` is its mirror:
  every seed arrived at the same point and they still finished apart, so arrival
  explains none of the outcome. Both count as not-determined, so a roster full of
  either refutes rather than abstains;
- an instance is **determined** when it reaches **0.7** on *both* Pearson and
  Spearman. Pearson alone is the statistic #134 used, but on this roster the
  common shape is several seeds tied at the published optimum and one far out,
  and a Pearson `r` over that is carried by the single outlying seed. Requiring
  the rank correlation too is what stops the count being built from one-point
  correlations, and it errs toward *not* finding an effect;
- the effect **generalises** when the median `r` is at least **0.7** and a strict
  majority of eligible instances are determined;
- fewer than **10** eligible instances is **inconclusive**, which is not the same
  answer as "does not generalise".

Two things to hold while reading the output. First, "varied" means "differs in
the cell the runner published". The **final** objective is six significant
figures — `cell()` streams a double through a default-precision
`std::ostringstream` — so eight seeds finishing within ~1e-6 relative read as
`final-invariant`. That is the intended reading (a 1e-6 spread is not a descent)
but it is a property of the table's precision, and it belongs in any write-up.
The **first-feasible** objective is written by `precise_cell()` at 17 digits
instead, and deliberately: `arrival-invariant` is an *eligible* bucket that
counts as evidence **against** the effect, so a rounding artefact there would
manufacture a refutation rather than merely withhold evidence. Second, much of
this roster is solved to the same objective on every seed at 60s, so a large
`no-spread`, `final-invariant` or `arrival-invariant` count is not a failed
campaign. Read the bucket line the report prints before reading the verdict.

**Before spending the 6.7 hours, buy the eligibility floor cheaply.** The
verdict is `INCONCLUSIVE` below 10 eligible instances, and eligibility needs a
spread at one end — which much of this roster does not have. Run a three-seed,
10-second pilot over the whole roster first (~25 minutes, same commands with
`--time-limit 10` and seeds `1 42 7` into a separate scratch directory), and read
**only the bucket line** from it: if `determined + not-determined +
final-invariant + arrival-invariant` is already at or above 10 there, the real
campaign will clear the floor comfortably. The pilot's own verdict is **not** the
campaign's verdict and must never be quoted as one — a 10-second budget is a
different experiment.

Sanity check before believing any of it: the report prints `nvs01`'s own `r`
beside #134's 0.945. A large disagreement is a reason to stop and find out why
before reading the roster line — either the measurement is wrong, or the engine
has drifted since `09097de`, which is worth knowing on its own. It is not
evidence about the roster either way.

Record the verdict — and the commit, seed set and budget it was measured at — in
this README, beside the `nvs01` section above. Whatever it says, **it does not
authorise an engine change**: #149's fourth criterion is that any change gets its
own issue, its own hypothesis and its own regression test.

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
published dual on all three. Verification: the SCIP side uses
`Model.checkSol(original=True)`, i.e. SCIP validating its own solution against
the pre-presolve problem. A solution SCIP cannot re-validate is not published as
feasible. A CBLS row is checked twice. The C++ runner re-checks its assignment
against the model it built -- but that reads the very DAG node values the search
optimised, so an evaluation bug the two share agrees with itself by
construction (#205: `sqrt` of a negative read as 0.0, and rows were marked
feasible on it). So since #205 the runner also writes each verified row's
assignment (`--solution-dir`), and `run_benchmark.py` has SCIP read the `.nl`
and check that assignment, and the objective the row publishes, on SCIP's own
expression evaluation (`benchmarks/minlplib/independent_check.py`) before
anything is published. A row SCIP rejects is published as
`VERIFY-FAILED(independent: ...)` with its objective blanked. This needs the
`benchmarks` extra (pyscipopt) at publish time.

<!-- campaign_report:begin scip-verification -->
Zero rows in this run failed that check.
<!-- campaign_report:end scip-verification -->

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
sides (recorded per row as `read_seconds`; the total is below); and the solve
stays single-threaded (CPU/wall measured at ~1.0). Like the CBLS run this is a
wall-clock budget, so these are single-sample numbers.

<!-- campaign_report:begin scip-read -->
SCIP's `.nl` reads total 0.076s across the roster (`read_seconds`).
<!-- campaign_report:end scip-read -->

Where SCIP proved no dual bound it reports its `1e20` infinity sentinel, which is
a *finite* float; the runner folds that to `NaN` at capture, so an unproved bound
is never published as a proof. Rows with no dual bound therefore read `NaN` in
`scip_dual_bound` and `scip_gap%`, the same spelling the CBLS rows use.

**Hardware.** The SCIP run was executed on an AMD Ryzen 5 5600H (12 logical
cores, Linux 7.0), one core in use. The committed CBLS run's machine is recorded
in `comparison.run.json` (#123) and printed in the provenance block under
**Results**: the same CPU model, one solve at a time, one thread. The SCIP side
still names its machine only here, and the two runs were made at different
times, so read the head-to-head as two runs on the same hardware class rather
than one controlled measurement.

That gap matters less than it first appears, and it bites the opposite way round
from the obvious guess. Both sides run a fixed 60s per instance, so:

<!-- campaign_report:begin hardware -->
- The **wall-clock totals are the robust number.** CBLS never terminates early — its 3000s is 50 × 60s by construction, and is therefore independent of the machine entirely. SCIP's 1011s is dominated by proving optimality on 34 instances and stopping, not by clock rate; even a 2x hardware advantage would leave 505s against 3000s.
- The **counts are what a hardware difference would actually move.** Both
  "feasible within the budget" and "proved optimal within the budget" scale with
  machine speed, so those are the numbers a faster or slower box would change — in
  either direction, for either solver.
<!-- campaign_report:end hardware -->

<!-- campaign_report:begin head-to-head -->
| | CBLS | SCIP |
|---|---|---|
| feasible | 48 / 50 | **49 / 50** |
| proved optimal | n/a (primal heuristic) | 34 / 50 |
| hit the 60s limit | 50 | 16 |
| total wall over the roster | 3000s | 1011s (median 0.28s; 31 instances under 1s) |
| integrality mismatches vs catalogue | 0 | 0 |
| verification failures | 0 | 0 |
<!-- campaign_report:end head-to-head -->

<!-- campaign_report:begin disjoint -->
**The failures are almost disjoint, and that is the useful part.** SCIP reaches a feasible solution on both instances CBLS did not solve in this run. One row goes the other way: CBLS feasible where SCIP found no feasible solution in 60s.

| Instance | CBLS | SCIP |
|---|---|---|
| `st_e36` | -243.857 (0.87% from BKS) | **no feasible solution in 60s** (timelimit); dual bound -304.5 |
| `elec25` | infeasible | 243.859 (0.02% from BKS), timelimit in 60.00s |
| `elec50` | infeasible | 1422.33 (34.79% from BKS), timelimit in 60.00s |
<!-- campaign_report:end disjoint -->

What each row settles — the numbers are in the block above:

- `nvs01` and `st_e40` — on this list in the previous table, where SCIP proved
  both optimal in under a quarter of a second and CBLS found no feasible point.
  #123's table has CBLS feasible on both, on every seed; see "Retired root
  causes" above for what #102 changed.
- `elec25` — confirms an engine gap, not hardness: a feasible point of near-BKS
  quality is easy to reach. Originally attributed to #100; that is fixed, and the
  remaining cause is the undamped Newton jump (see the root-cause table above;
  [#110](https://github.com/spoorendonk/cbls/issues/110), which tracked it, is
  closed as not planned and nothing open replaces it). Still infeasible on all three of #123's seeds.
- `elec50` — same mechanism at 50 points; SCIP does not close it either, but it
  does reach the feasible region.
- `st_e36` — the row the other way: SCIP spends the full budget and returns only
  a dual bound.

<!-- campaign_report:begin quality -->
**Solution quality where both are feasible.** Buckets over the 40 instances that both solve and whose `|BKS| >= 0.0001` (below that a percentage against the bound is not informative — see the zero-BKS discussion above), outside the documented failures:

| | ≤0.01% | ≤1% | ≤10% |
|---|---|---|---|
| CBLS | 26 | 28 | 31 |
| SCIP | 34 | 34 | 35 |
<!-- campaign_report:end quality -->

SCIP is clearly ahead on quality, as expected of a mature global solver on a
roster capped at 150 variables and 150 constraints.

<!-- campaign_report:begin cbls-ahead -->
Five instances go the other way by a margin larger than both the claim band and the table's six-significant-digit resolution, and every one is a row where SCIP exhausted the 60s budget:

| Instance | CBLS gap | SCIP gap | SCIP status |
|---|---|---|---|
| `eg_all_s` | 88.3% | 2324% | timelimit |
| `ex8_1_5` | **matches BKS** | 100% | timelimit |
| `ex8_6_1` | 23.9% | 99.6% | timelimit |
| `eq6_1` | 2.53% | 27.0% | timelimit |
| `maxmin` | 9.0e-3% | 2.18% | timelimit |
<!-- campaign_report:end cbls-ahead -->

`ex8_1_5` is the sharpest of these: SCIP cannot make progress on it at all (its
two variables are unbounded, so the dual bound diverges), while CBLS now reaches
the published optimum exactly. It was CBLS's *worst* row before #107 was fixed.

### Objective quality under the #102 unproductive-batch exit

The #102 change ends a Feasibility-Jump batch that has stopped reducing the real
rows' violation and takes the diversification kick as due. That alters the search
trajectory on every instance, not only the one it was found on, so it is measured
on objective quality and not only on feasibility — a feasibility-only tally
cannot see a row that stayed feasible and got worse.

**Arms.** `main` at engine commit `0ffbf9d`; `after` at `b7f8a50` on branch
`issue-102-ex8_6_1`. Both built Release from their own checkout, sanitizer and
profiling off. Eight instances x four seeds (42, 1, 2, 3) at `--time-limit 10`,
the two arms **interleaved per instance** and run serially on an idle box, gap to
BKS. This is an **indicative probe, not the published protocol** — a 10s budget
and a subset chosen to include every instance an earlier arm had regressed. The
published rows were regenerated separately, at the documented protocol, by #123
(see "Across seeds" under **Results**).

**The budget is wall-clock, so a fixed seed does not pin the iteration count.**
Read this table with the noise floor below, not as exact quantities.

| instance | main (`0ffbf9d`) | after (`b7f8a50`) |
|---|---|---|
| `alkylation` | 99.98 / 99.99 / 99.99 / 99.98 | **0.08 / 0.07 / 0.14 / 0.08** |
| `st_e36` | 40.24 / 32.34 / 40.24 / 40.24 | **3.38 / 3.38 / 3.38 / 3.38** |
| `maxmin` | 3.90 / 0.01 / 0.29 / 1.97 | 0.33 / 0.10 / 0.17 / 0.14 |
| `kall_ellipsoids_tc02b` | 161.76 / 110.61 / **infeasible** / 170.73 | 71.30 / 110.61 / 133.14 / 147.58 |
| `st_e40` | infeasible on all four | **0.00 on all four** (BKS 30.4142135) |
| `nvs01` | infeasible on all four | feasible on all four: 30.16 / 12.23 / 734.94 / 792.13 |
| `ex4_1_8` | ~0 on all four | ~0 on all four |
| `ex8_6_1` | 69.67 / 78.87 / 64.17 / 65.48 | 69.67 / 78.87 / 64.17 / 61.16 |

**The noise floor is about 3-4 gap points on these rows, not "sub-1%".** Measured,
not assumed: between two rounds against a **bit-identical** `main` binary,
`ex8_6_1`'s own gaps moved 76.55 -> 78.85, 67.90 -> 64.64 and 67.97 -> 65.45 on
seeds 1, 2 and 3, and `maxmin`'s seed-42 figure moved 3.90 -> 0.01 -> 3.90 across
three separate measurements. Differences smaller than that band are not results.
Read `maxmin`, `ex4_1_8` and `ex8_6_1` seeds 42/1/2 as unchanged.

What survives that floor is the four rows where the movement is an order of
magnitude larger than it: `alkylation` (~99.98 -> ~0.1), `st_e36` (32-40 -> 0.9-3.4),
`st_e40` (infeasible -> BKS) and `nvs01` (infeasible -> feasible). `kall_ellipsoids_tc02b`
gains a feasible point on seed 2 where `main` has none, and is level or better on
the rest; its remaining spread is wide and single-run, and this README elsewhere
records that instance moving 55.1 -> 78.2 on an unmodified binary at a fixed seed,
so read only the feasibility change there.

**`ex8_6_1` was the reason this change was held, and it is fixed** — the `after`
arm is level with `main` on every seed, where the arm that held the branch lost
6-20 gap points on all four. Seeds 42, 1 and 2 come back at `main`'s figure to
the digit, which is consistent with the exit never arming on those runs; the
bit-identity that implies is asserted where it can actually be tested, on the
iteration-bounded path, by `tests/test_search.cpp` — a wall-clock A/B cannot
establish it, since the two arms need not get the same number of iterations from
the same 10s.

**Two rows moved between two runs of this arm** and the table above is the
later, so nothing here is carried across arms: `st_e36` seed 3 (0.87 -> 3.38,
which brings it into line with its other three seeds rather than away from them)
and `ex8_6_1` seed 2 (64.12 -> 64.17). Both are inside the noise floor below. The
other 30 of 32 runs are identical between the two runs.

An earlier revision of this paragraph named a commit pair, but the re-pointing
edit that introduced it wrote `b7f8a50` on both sides, so the pair it meant is
not recoverable from the text. What is checkable is that `b7f8a50` (the
pre-rebase tip of `issue-102-ex8_6_1`) and `1559786` (its rebased equivalent on
main) have **identical engine source** — `git diff b7f8a50 1559786 -- src
include benchmarks/minlplib` is empty — so any movement between measurements
taken across that pair is run-to-run noise, which is what the noise floor below
is for, and not engine drift.

**The mechanism, and what it cost before it was bounded.** The exit's progress
measure sums the **real** rows and deliberately cannot see the artificial
`obj <= bound` row. Before the first feasible solution that is the point of it.
After it, the search's work is a *trade* between the two — the bound is tightened
on every new best, FJ pulls the assignment off the real-feasible set to chase the
objective row, and the real rows settle at a strictly **positive** equilibrium.
Because the measure's reference is a running *minimum* over the batch, "no new
all-time low in `unproductive_iterations`" is then the normal state of a batch
that is working perfectly, and the detector is unconditionally true.

An earlier arm gated that on the outer loop's stagnation count. That is a one-shot
**delay**, not a bound: the kick site deliberately does not reset `stagnation`, so
once the count first crosses the threshold every later batch is armed again. On a
synthetic plateau model — proven optimum reached in the first batch, strictly
positive real violation for the rest of the budget — it took **76 kicks and 25 LNS
repairs** where the correct answer is zero of each. `ex8_6_1` escaped it only
because it improves on most of its batches and so rarely strings five
non-improving ones together; the gate held for that instance, not for the class.
That arm lost 6-20 gap points on `ex8_6_1` across four paired seeds
(89.39 / 85.90 / 78.90 / 74.37 against a `main` of 69.67 / 76.55 / 67.90 / 67.97),
so "about 20 points" describes its worst seed, not all four.

**What is bounded is the LNS half, not the kick.** A perturb is microseconds; an
LNS destroy-repair is bounded by `min(2.0, remaining())` **seconds**, and
`state_key` accepts on raw violation with no tolerance, so against an incumbent
sitting at ~1e-7 essentially every repair is rejected and rolled back. So once a
feasible solution exists the unproductive route draws only the perturb. Keeping
the kick matters: `st_e40` uses exactly those post-feasible kicks to move between
its 52 feasible integer combinations, and suppressing the whole kick drops it from
its BKS on 4 of 4 seeds to 2 of 4. Rate-limiting the kick instead starves it the
other way — infeasible at seed 2 on the 15 000-iteration regression budget.

Pinned by `tests/test_search.cpp`, which asserts that on that plateau model the
run performs no more LNS repairs than one with the exit compiled out; red at 25
against 0.

**Provenance of the mechanism figures.** The per-batch counts quoted above for
`ex8_6_1` — 152 of its first 162 batches improving, nine declared stuck, the
measure between 0.016 and 0.51, three LNS repairs taking 4.7s of a 10s budget —
were read from temporary instrumentation that is **not committed**, on `ex8_6_1`
at seed 42, 10s, against the delayed arm. They are the reason the mechanism was
found; they are not reproducible from this checkout.

The published rows are **not** regenerated from this probe: #123 regenerated
them at the documented protocol, and its old-against-new movement is under
"Across seeds" in **Results**.

### What #107 accounted for

#107 was found by noticing that instances with at least one **free (unbounded)**
variable did far worse, and asked how much of that gap the fix actually explains.
Measured from the committed tables and the `.nl` files' variable bounds, over the
rows each group solves whose `|BKS| >= 1e-4`:

<!-- campaign_report:begin free-variables -->
| | instances | eligible | within 10% |
|---|---|---|---|
| ≥1 free variable | 16 | 13 | 6 |
| no free variables | 34 | 28 | 26 |

Within 10% with at least one free variable: `shiporig`, `ex8_1_5`, `maxmin`, `st_e40`, `spring` and `windfac`.

**Before #107** — a table that is not committed, so these counts are fixed constants in `campaign_report.py` — within 10% were 1 of 12 eligible rows with a free variable (`maxmin` only) and 20 of 27 without. Against this table: **1 of 12 → 6 of 13** with a free variable (`shiporig`, `ex8_1_5`, `st_e40`, `spring` and `windfac` join `maxmin`), and 20 of 27 → 26 of 28 without. That difference spans every engine change between the pre-#107 table and this table's commit, not #107 alone.
<!-- campaign_report:end free-variables -->

**#107's own share was measured against the previous table**, which was #107's
AFTER run (`21086c2+107`): within 10% were 3 of 12 eligible free-variable rows
and 19 of 27 without, so #107 explained **2 of the 11** free-variable misses
(`shiporig` and `ex8_1_5` joining `maxmin`), and the no-free group's change was
−1 — `eq6_1` crossing the 10% line, which an A/B showed is *not* attributable to
#107 (bit-identical between arms). A real but minority share. The block above is
against #123's table, so its larger difference belongs to every engine change
since (see "Across seeds" under **Results**), not to #107.

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
  `integrality mismatch` in the tally; the published run's count is the
  Results tally's "integrality mismatches" row.

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
  stable of the two — most of its rows terminate with a proof rather than at the
  limit (the head-to-head table has the counts) — but the rows that hit the limit
  are as draw-dependent as the CBLS numbers.
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

## Portfolio A/B: the shared objective bound (#179)

`cbls_minlplib` is single-threaded, so a portfolio comparison runs through
`cbls_minlplib_portfolio` (`benchmarks/minlplib/portfolio_ab.cpp`). It builds the
model exactly as the runner does and runs `ParallelSearch` with the runner's
per-worker hook and LNS. `benchmarks/minlplib/portfolio_ab.py` pairs the arms and
scores them:

```
.venv/bin/python3 benchmarks/minlplib/portfolio_ab.py run --instances nvs22 eq6_1 ... \
    --seeds 1001 1002 1003 1004 --budget 20 --threads 4 --out ab.jsonl \
    --lock ~/.cache/cbls-bench.lock
.venv/bin/python3 benchmarks/minlplib/portfolio_ab.py analyze ab.jsonl
```

The first campaign ran through a scratch build of the same harness, which was
later committed unchanged in behaviour. The engine was a pre-rebase commit whose
on-main equivalent is `321d896`. Production code on main differs from that binary
in two places:
- #176's slope lookup, which gives bit-identical scores;
- the shared-bound sync's dropped `vm_.invalidate_cache()` (`bfd578d`), which is
  value-correct but not bit-neutral. Setup: 4 threads, 20 s,
seeds 1001-1004. The roster was 10 instances admitted by a pre-registered
control-only pilot, which selected instances where the control arm runs behind
the portfolio's best bound. The results therefore say how the mechanism does
where it has something to act on; they are not a roster average.

The primal integral improved on all 10 instances (sign test p = 0.002, instance
mean -0.083, t_9 95% CI [-0.156, -0.010]). The final gap improved on 8, got worse
on 1 and was flat on 1 (p = 0.039; CI [-0.183, +0.003]). All 40 runs were
feasible in both arms. The protocol and the MIPfeas half are on #179.

The MIPfeas half of the same A/B was run without a committed driver. Each arm pair
is two direct `cbls_mipfeas` invocations:
`--instance I --inst-dir benchmarks/instances/mipfeas --out-dir D --budget 30
--seed S --threads 4 --commit <sha>`, the control arm adding `--no-share-bound`.
The runs used seeds 1001-1006 over the 10 pilot-admitted instances. They ran
serially, with both arms of a pair back to back under the machine lock and the
arm order alternating with seed parity. Each run was scored with
`primal_integral.primal_integral` against the roster optimum over [0, 30 s].
`portfolio_ab.py analyze` accepts the resulting rows (`instance`, `seed`,
`share`, `pi`, `gap`).

A held-out roster the shipped defaults were never fitted to lives in `heldout/`; method, pool size and composition are in [HELDOUT.md](HELDOUT.md) (#144).
