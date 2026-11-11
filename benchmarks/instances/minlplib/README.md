# MINLPLib non-convex benchmark subset

External yardstick for the CBLS continuous machinery (reverse-mode AD on a
transcendental DAG + Newton jump values). Where MIPfeas (`../mipfeas/`; it
superseded the retired MIPLIB-FJ, #90) exercises the pure-linear search core,
this subset targets **non-convex
mixed-integer non-linear** instances — the regime where standalone CBLS
competitors (Yuck, fzn-oscar-cbls) cannot even *encode* most cases, because
FlatZinc has no transcendental float constraints. Coverage is the headline
metric here; gap-to-BKS is secondary (we are a primal heuristic).

Source: [MINLPLib](https://www.minlplib.org/). Metadata and bounds come from
the published catalogue CSV:

    https://www.minlplib.org/instancedata.csv   (semicolon-separated)

Each instance's text NL file is fetched individually from
`https://www.minlplib.org/nl/<name>.nl` (we do **not** pull the multi-hundred-MB
archive). The CBLS NL reader (`src/io/nl_reader.cpp`) reads both the text ('g'
header) and, since #282, the binary ('b' header) format. The downloader still
passes over instances served only as binary NL (e.g. the `kriging_peaks-red*`
family): the published and held-out rosters were drawn under that rule, and
changing it would change their membership.

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
| `opvcpower` (x^c) / `opcvpower` (c^x) | `Pow` (NL emits both as standard `OPPOW`) | full library only: the roster filter still skips them, as its membership is pinned (#283) |
| `operrorf` (erf)   | —                      | no (skipped) |
| `opgamma`          | —                      | no (skipped) |
| `opcentropy`       | —                      | no (skipped) |
| `opmod`            | —                      | no (skipped) |

Piecewise (`OPPLTERM`), function calls (`OPFUNCALL`), and inverse-trig opcodes
are parsed structurally but rejected by the adapter with a skip reason; the
runner records these as `skipped(unsupported)`.

The reader inlines `V` (defined-variable) segments and skips `d` (dual initial
guess) segments (#281). It rejects `S` (suffix; in MINLPLib only the SOS
suffixes of the `water*` family, which skipping would relax) and `F` (function)
segments; instances using them are reported as `skipped(unsupported)`.

## Full-library status (#283)

Beside the pinned roster filter, `download.full_instance` is the full-library
selection: every problem type, convex or not, no size budget, `opvcpower` and
`opcvpower` accepted; only `operrorf`, `opgamma`, `opcentropy` and `opmod`
exclude, plus the lack of a finite primal bound. `status.py` gives every
catalogue instance one status, the first that applies: `unsupported-op`,
`missing-nl`, `no-primal-bound`, `reader-error`, `reader-unsupported` (a construct the reader lacks, e.g. an SOS suffix), `adapter-unsupported`, `checker-failed` (crashed, killed, timed out or threw),
`modelled`. It reads the NL files from a local directory (the unpacked
`https://www.minlplib.org/minlplib_nl.zip`) and runs the
`cbls_minlplib_nl_check` target (read and model, no solve) over them:

    .venv/bin/python3 benchmarks/instances/minlplib/status.py \
        --catalogue instancedata.csv --nl-dir minlplib/nl \
        --out status.csv --bounds minlplib/nl/bounds.csv

`--bounds` writes the modelled set as a `bounds.csv` for `cbls_minlplib`,
which takes one instance directory, so it goes next to the NL files. Counts per
status for a given catalogue and engine commit are recorded on #280.

### Presolve (removed)

`--presolve` built each model with a linear-part presolve (#326, #342, #343),
with doubleton-equality aggregation (#328, #347) and a census mode in
`cbls_minlplib_nl_check`. #331 measured it no better than the plain engine and
it stayed off; it was removed (#379; in git history at `b10da9d2`). Census
counts for its commits are on #326, #343 and #328.

### Search counters (#341)

`cbls_minlplib --counters CSV` writes one row per instance the search ran on:
`instance, iterations, fj_batches, novelty_batches, novelty_moves, cpu_seconds,
peak_rss_kb, search_config` (#380 dropped the propagated-leg counters with the
feature). `cpu_seconds` is process CPU time around the
search: the search's own at `--threads 1`, summed over workers above it.
`peak_rss_kb` is the process's peak as the search returns, so it is per
instance only with one `--instance` per call. The table is unchanged; the
sidecar never takes the table's or the trace's path, nor any path under the
instance directory (where `bounds.csv` and the published files live) or
`--solution-dir`, symlinks resolved. A solve
that throws, postsolve included, writes no counters row.

### Full-library campaign (#256)

Both halves read that `bounds.csv`; neither writes under `benchmarks/instances/`.

- SCIP: `../../minlplib/scip_shards.py --nl-dir NL --shards DIR` runs
  `reference_solve.py` once per instance, `--jobs` single-thread solves at a
  time under one machine lock. Each instance that reached a solve leaves `<instance>.csv` or an
  `<instance>.failed` marker (one `exit <rc> <utc>` line per failed attempt; 124
  is the 15-min wrapper cap); a re-run retries only the missing and
  failed ones. The #256 rule: one
  retry with the memory cap raised, then SCIP no-solution, named with its exit.
  A marker with two attempts is left alone on a re-run unless `--retry-final
  NAME` asks for it, and the run exits 1 (#323). A process that exits by itself
  with nothing written never reached a solve and records no attempt. `stamp.txt` records budget and seed and refuses a
  run under others; `--restart` deletes the `.failed` markers first.
- cbls: `../../minlplib/run_benchmark.py`, one instance per call, at the
  campaign's thread count. A nonzero runner exit sets the row aside as
  `<instance>.csv.rejected` and appends `exit <rc> <utc>` to
  `<instance>.failures`; the next call solves it again (one retry, then cbls
  no-solution). Exit 0 without a publishable row counts the same; an exit before
  a row is staged records nothing; a success clears the record, and a stamp
  change clears every record. With two attempts recorded, a re-run leaves the
  instance alone, says it is final and assembles no table (exit 1), unless
  `--retry-final NAME` asks for it; `--instances NAME` and `--no-resume` do not
  (#323). Once SCIP has checked a staged row the driver
  writes `<instance>.checked` with the row's sha256. `../../minlplib/check_staged.py
  --inst-dir NL --staging-dir STAGE` does the check and marking without solving
  (for rows a killed or pre-#307 driver left unchecked); it refuses a stamp for
  another roster or host, lists a row whose check raised (exit 1), and
  leaves header-only, torn and unstageable rows unmarked: re-solve those with
  the driver at the stamp's commit (`--instances NAME`).
- Score: `../../minlplib/gm_ratio.py --bounds NL/bounds.csv --cbls
  comparison.csv --scip-shards DIR` prints the GM(gap+1) ratio cbls/SCIP with
  feasible and head-to-head counts and names every no-solution instance. It
  refuses an incomplete roster. `--cbls-staging STAGE` scores the staging
  directory instead: every staged row needs a matching `.checked`, a lone
  `.rejected` counts as cbls no-solution. In
  either mode a lone `.rejected` or a `.failed` also needs two recorded failed
  attempts, and fewer is refused by name (#312). A shard stamp whose budget or
  seed differs from SCIP's rows is refused too. `--partial` gives an interim
  score without the attempts and `.checked` requirements, and is the only way to
  score a campaign staged before #312.

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

In this directory:

- `download.py` (selection, fetch, validation); `status.py` (full-library
  status); `heldout/` (the held-out roster, [HELDOUT.md](HELDOUT.md)); `*.nl`.
- `bounds.csv`: the roster with published primal/dual bounds and the
  catalogue integer-variable count.
- `comparison.csv`: the `cbls_minlplib` runner's rows, assembled by
  `run_benchmark.py` once SCIP has checked every verified row (the runner
  refuses to write it): objective, gaps to BKS and dual, feasibility, notes,
  commit SHA, `max_violation` (closest approach), `n_int_vars`, then columns
  named for the issue that added them, which date a table: `search_config`
  (#136, `key=value;...`), `lns_repairs` (#143), `lns_repairs_accepted` (#150),
  `first_feasible_objective` and `time_to_first_feasible` (#149). Both LNS
  counters read `NaN` where no solve completed (0 would claim LNS ran);
  `time_to_first_feasible` says whether a feasible point was recorded, since
  its objective can be `NaN` or `inf` (#100).
- `comparison_seeds.csv`: every seed's rows, `seed` first (#141); the
  committed one holds seeds 1-3 of #123. `comparison.run.json`,
  `comparison_seeds.run.json`: their run records.
- `analysis_notes.csv`: curated root-cause verdicts (`bug` vs `hard`) for
  unsolved instances, merged into `comparison.csv`'s note column.
- `anytime_trace.csv`: incumbent against wall time for the published run
  (`instance,time_seconds,objective,new_best`), the internally *minimised*
  objective, so a maximize instance appears negated.
- `campaign_summary.json`: the campaign's machine-readable summary
  (`campaign_report.py --json`); a test requires it to equal a fresh rendering.
- `scip_baseline.csv`: SCIP's objective, gaps, feasibility, wall time, dual
  bound, status and `SCIP x / PySCIPOpt y` per row. `comparison_all.csv`: the
  long-format join of both with `bounds.csv`, `method` in `published-bks` /
  `cbls` / `scip`.

In `../../minlplib/`: `run_benchmark.py` (the re-run driver, the supported way
to regenerate `comparison.csv`); `campaign_report.py` (every run-derived number
in **Results** and **SCIP baseline**, the JSON summary and the README blocks,
#142; solves nothing);
`run_ablation.py` and `ablation_report.py` (#143's ablation campaign; scratch
`--out-dir` only, refused under `benchmarks/instances/`. The held-out
`--campaign transfer-145` and `--campaign presolve-322` arm sets and
`presolve_report.py`, HELDOUT.md, are removed; in git history at `ef795b55`); `nl_check.cpp` (`cbls_minlplib_nl_check`: read and
model, no solve); `independent_check.py` (#205's SCIP check of a row's `.sol`);
`scip_shards.py`, `check_staged.py`, `gm_ratio.py` (the full-library campaign);
`runner.py` (the
runner's row, exit-status and note contract).

Regenerate `scip_baseline.csv` and `comparison_all.csv` (needs the `benchmarks`
extra) after the CBLS side, since the merge reads whatever `comparison.csv`
holds:

    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --time-limit 60

`--merge-only` rebuilds only the merge, which is what `run_benchmark.py` does,
so a CBLS re-run never re-solves SCIP.

### Ablation arms

The runner's search configuration is settable per run, and whatever was set is
recorded in each row's `search_config` cell (#136):

```
--no-float-hook            drop the FloatIntensifyHook
--no-lns / --lns-interval N   drop LNS, or change how often a kick is a repair
--compound-moves / --no-compound-moves, --novelty-prob P
--unproductive-iters N     the #102 unproductive-batch exit (<= 0 = old cadence)
--perturbation-period N    batches without improvement before a kick
--max-iterations N         GLS iteration budget (0 = unlimited)
--no-time-limit            disable the wall clock; requires --max-iterations
--no-objective-tie-break   drop the FJ objective tie-break (#218; on since #226)
```

`--no-objective-tie-break` is recorded only off the engine default, so a
default row's cell is the one the committed tables hold (they predate #226 and
ran the tie-break off; only the commit tells them apart). Compound moves are on by default since
2026-10-07; the published tables record `compound_moves=off` and so are an
off-default arm until regenerated. Two pairs are refused, because the engine
never reads the second flag and the row would record an arm the run did not
have: `--lns-interval` with `--no-lns`, and `--novelty-prob` with
`--no-compound-moves`.

Any non-default value refuses to write the published `comparison.csv` (and
`anytime_trace.csv`). `cbls_uc_chped` carries the same flags against its own
protocol (compound moves off, so there `--compound-moves` is the arm and
`--novelty-prob` alone is refused); `benchmarks/common/search_config_flags.h`
defines them all. `--threads N` (#262) runs the `ParallelSearch` portfolio, one
hook and one LNS per worker, and adds `threads=N` to the cell (left out at 1).
Every campaign runs at 16 threads (Simon, 2026-11-01,
[#255](https://github.com/spoorendonk/cbls/issues/255),
[#256](https://github.com/spoorendonk/cbls/issues/256)); the committed tables
ran the single-threaded `solve()`. One table holds one thread count: see the
refusals under "Re-running the CBLS rows".

### The ablation campaign

`benchmarks/minlplib/run_ablation.py` runs the arms above as one interleaved
campaign (issue #143) and `ablation_report.py` scores it:

    .venv/bin/python3 -m benchmarks.minlplib.run_ablation --out-dir /scratch/ablation

The same command resumes; `--dry-run` prints the plan and the cost,
`--report-only` re-scores an existing `results.csv`. Four properties, each
pinned in `tests/python/test_minlplib_ablation.py`:

* **Interleaved per instance**: every (arm, seed) of one instance runs back to
  back, arms innermost, so a control and its arm are minutes apart, not a
  roster pass.
* **Serial, and asserted**: one solve at a time, under an exclusive lock on the
  out-dir plus a load-average refusal at start.
* **Scratch only**: `--out-dir` is refused inside `benchmarks/instances/`, and
  the runner always gets `--instance` (its own refusal to write the published
  table, which the all-default control arm needs).
* **Resumable**: each run is appended to `<out-dir>/results.csv` and fsynced; a
  restart skips the recorded `(instance, arm, seed)` triples and refuses an
  out-dir stamped with another commit, budget, seed set or arm set.

The `--no-lns` arm is **gated on data**: a one-seed control pass reads
`lns_repairs`, and the arm runs only if some instance recorded a repair
(attempted, not accepted: a rejected repair still spends RNG and budget). The
verdict goes to `<out-dir>/lns_gate.json`; `ablation_report.py` prints both
totals per arm. The **noise floor is measured** per instance from the control's
across-seed spread, `t * s_i * sqrt(1/k_arm + 1/k_control)` (two-sided 95%
Student, `t` = 4.30 at three seeds), bounded below by the runner's tie band for
that instance's bound; there is no aggregate floor. `ablation_report.py`'s
module docstring is the definition of record. An effect inside its floor is
reported as "inside the noise" with the floor quoted, and an instance feasible
in one arm only is counted in its own bucket.

## Re-running the CBLS rows

**One command**, from a configured Release build directory and a clean checkout:

    .venv/bin/python3 benchmarks/minlplib/run_benchmark.py

It rebuilds `cbls_minlplib` (so the binary cannot lag the SHA it is labelled
with), solves the 50-instance `bounds.csv` roster **serially** at 60s each
(~50 minutes of solving, plus build and read time), rewrites `comparison.csv`
and `anytime_trace.csv`, and re-merges the **CBLS rows** of `comparison_all.csv`.
`--dry-run` prints the commands, roster size and estimate, and exits non-zero if
any refusal applies, so it is a precheck. `--time-limit`, `--seed 1` and the
roster are *fixed*: only the engine may differ from the previous table. **The
box must be quiet**: on a wall-clock budget a concurrent build, test suite or
benchmark changes every row's iteration count. Check `uptime` first; the driver
never parallelises solves, and `--build-jobs` (build only) defaults to 4.

**It refuses** (the module docstring gives some reasons; the table with each
one is on [#358](https://github.com/spoorendonk/cbls/issues/358#issuecomment-6652522933)): a dirty tree (its own
published tables excepted); a build dir that is unconfigured, not
`CMAKE_BUILD_TYPE=Release`, configured from another checkout, or with
`CBLS_SANITIZE` or `CBLS_PROFILE` (sticky cache entries a flag-less `cmake -B
build` keeps); `--no-build` with no runner binary; a roster instance with no
`.nl`; `--instances` without scratch `--out`, `--trace-out` and
`--staging-dir`, or with `--out` resolving to `comparison.csv`; a run writing
exactly one of `comparison.csv` and `anytime_trace.csv`, either way round
(#149), or `--trace-out` naming the published trace with a scratch `--out`; a
staging directory stamped with another commit, budget, seed, `--threads`, host
or roster; a `--seed` other than 1 naming either published file (#141); a
`--time-limit` other than 60 onto the default paths; a `--threads` other than 1
or 16 onto them (16 is the campaign protocol, #255, #256), above the CPUs the
affinity mask allows, or unlike the other seeds' in `comparison_seeds.csv`;
`--inst-dir` naming another roster under `benchmarks/instances/` (e.g.
`heldout/`, which publishes nothing) without scratch `--out` and
`--staging-dir`, or scratch output under `benchmarks/instances/` (#144);
another timed driver holding the wall-clock lock; a `comparison_seeds.csv` with
another runner's columns or an unreadable `comparison_seeds.run.json`; a
missing `scip_baseline.csv` (unless `--no-merge`); and a roster instance whose
read, build or solve **throws** (runner exit 3: its row measures nothing, is set
aside as `<instance>.csv.rejected` and is refused by name on resume, #153). A
coverage gap (`unsupported`, a missing `.nl`) exits 0 and does not trip that.

**Resumable.** Each instance is solved in its own process into
`$XDG_STATE_HOME/cbls/minlplib-rerun/<commit>/<budget>s/threads<T>/seed<N>/<instance>.csv`
(default `~/.local/state/cbls/minlplib-rerun/<commit>/60s/threads1/seed1/...`),
with a `.trace.csv` and a `.log`. A re-invocation skips *complete* staged rows
and re-solves a header-only file, a torn last line, a row with another
`commit_sha` or thread count, a missing `.trace.csv`, or a row whose note says
the runner threw. A nonzero runner exit renames row and trace `*.rejected`: a
solve that lost portfolio workers (#170) says so only in its exit status.
`stamp.txt` records commit, budget, seed, host and a hash of `bounds.csv`.
`--no-resume` forces a full re-solve, final no-solutions excepted
(`--retry-final NAME`); `--staging-dir` moves it. It lives
**outside the checkout** (#141), where neither pre-push's `rm -rf build` nor
the worktree workflow reaches; old commit directories are safe to delete once
published. **One timed driver runs at a time**: this one and `run_ablation.py`
hold `benchmarks/common/jobs.py`'s `wallclock_lock`
(`~/.local/state/cbls/wallclock-<host>.lock`, per user and host) for the whole
run, and a second refuses with exit 2.

`comparison.csv` and `anytime_trace.csv` are replaced only at the end, by an
atomic rename. The merge rewrites `comparison_all.csv` in place: if it fails,
the driver exits 1 leaving the *previous* CBLS rows there, and a re-run goes
straight back to the merge.

`elec25`/`elec50` **stay in the roster** as documented failures: in every
roster count, out of every quality aggregate and claim, per
[#87](https://github.com/spoorendonk/cbls/issues/87) (`AGGREGATION_RULE` in
`benchmarks/minlplib/campaign_report.py`, printed by the driver and the report,
#142). **If an `elec` row comes back feasible with a finite objective, stop and
read the closed history in #110/#116/#117 before publishing anything about
it**: that is a result, not a routine refresh.

### Seeds, and what each publish records

**`comparison.csv` is one pre-registered seed, seed 1, and only seed 1 may write
it** (#141): it is an A/B against the previous table in which only the engine
differs, and a median-of-N table is not even defined for a whole table. **Every
seed is published beside it, in `comparison_seeds.csv`**: the runner's columns
with `seed` prepended, one block per seed. Every whole-roster 60s run onto the
default paths replaces its seed's block there and leaves the others intact. A
seed other than 1 writes neither published file by default, skips the
`comparison_all.csv` merge and assembles its table and trace in its staging
directory (`comparison.assembled.csv`, `anytime_trace.assembled.csv`); per-seed
traces are not published. A scratch `--out` or `--trace-out` that resolves to a
published file, or whose run record would, is refused at every seed. One seed
per invocation; check `uptime` first, since each record captures the load at its
own start:

    for SEED in 1 2 3; do
        .venv/bin/python3 benchmarks/minlplib/run_benchmark.py --seed "$SEED" || break
    done

**Run every seed at one commit, and edit nothing between them.** The dirty-tree
refusal ignores the tables the driver publishes and their run records
(`run_benchmark.run_commit_sha`), so seed 2 runs at seed 1's SHA, but any other
modified tracked file refuses (a regenerated README or `campaign_summary.json`,
an edited `analysis_notes.csv`, a test, any source), naming the files. So do
not start "After the run" steps 2-5 until the last seed has published, and do
not commit between seeds: the multi-seed summary aggregates only seeds published
at one commit.

**Three seeds is the floor** for quoting a spread. After each publish the summary
prints, across the seeds at this run's commit, budget and host, the feasible
count's min / median / max, each instance's `gap_to_bks%` median and range, and
any seed left out and why (another commit, budget or host, no run record, a
record for another commit than its rows, or rows changed since their record).
The definitions are `campaign_report.py`'s (`summarize_seeds`,
`SEED_AGGREGATION_RULE`): a seed that did not reach feasibility counts as worse
than any gap, and `elec25`/`elec50` count in the feasible spread only.
`campaign_report.py --seeds` prints the same from the committed files, at the
(commit, budget, host) the most seeds share, ties to the most recent. **Report
medians and ranges in prose**, beside the published table, never in place of it.

**Every published set carries a run record**: `comparison.run.json` and
`comparison_seeds.run.json` (one entry per seed). Each names the commit, budget,
seed, roster size, the machine (`benchmarks/common/provenance.py`'s
`machine_record()`: host, CPU model, CPU count and usable cores, memory, load at
the start of the publishing invocation), the concurrency, `resumed` (rows staged
by an earlier invocation, whose machine and load the record did not see), and
`table_sha256` of the table's (or seed block's) bytes. A reader refuses a
single-seed record whose hash no longer matches and leaves out a mismatching
seed. A scratch `--out` gets a record too. `campaign_report.py` reads the
budget, seed and machine from `comparison.run.json`, so
`--budget`/`--seed`/`--machine` are needed only for a table published before
records existed.

### After the run

After the **last** seed only: steps 2-5 modify tracked files the driver does not
own, and a later seed refuses that dirty tree.

1. `git diff benchmarks/instances/minlplib/`: expect changes only in
   `comparison.csv`, `anytime_trace.csv`, the `cbls` rows of
   `comparison_all.csv`, `comparison.run.json`, and each seed's block of
   `comparison_seeds.csv` with its `comparison_seeds.run.json` entry. The
   `published-bks` and `scip` rows must be byte-identical; if they moved,
   something re-solved SCIP and the run must be redone.
2. Regenerate every run-derived number. Nobody transcribes them (except the
   multi-seed spread, step 7): each sits between `<!-- campaign_report:begin
   NAME -->` / `end` markers, and the generator rewrites them from the committed
   tables, solving nothing:

       .venv/bin/python3 benchmarks/minlplib/campaign_report.py \
           --write-readme benchmarks/instances/minlplib/README.md \
           --json benchmarks/instances/minlplib/campaign_summary.json

   Review the diff. Pass no `--budget`, `--seed` or `--machine`: they are read
   from `comparison.run.json`, a stated value the record contradicts (and any
   `--machine`) is a provenance warning, and the rewrite is refused on any
   provenance warning or when a documented failure came back feasible. A record
   naming another commit than the rows is refused outright, in every mode. `--check-readme` instead exits 1
   naming any stale block, as `test_the_committed_readme_blocks_are_the_generators_rendering`
   and `test_the_committed_summary_json_is_the_generators` fail; `README_BUDGET` /
   `README_SEED` / `README_MACHINE` in that test file stay `None`, matching the
   command above. Then re-read the prose outside the blocks against them: which
   instances moved, and why, is yours to write.
   - **The instances the published run left infeasible**: if the generated list
     changed, drop a root-cause row and its `analysis_notes.csv` row together,
     only once `comparison.csv` shows the instance feasible.
   - The SCIP side of the blocks is engine-independent and must not move.
3. Delete any staleness paragraph in **Results** the new numbers retire.
4. **Attribute material movement to the engine changes, not to an algorithmic
   improvement**: `git log --oneline <old commit>..<new commit> -- src include`,
   naming the trajectory-changing ones (#123's attribution against `21086c2+107`
   is the worked example). A row that moved by less than its across-seed range is
   not evidence.
5. `analysis_notes.csv` is merged into the note column by the runner, onto rows
   that came back infeasible; a noted instance that came back feasible and
   verified gets `stale-analysis-note` and a per-run warning (`note_policy.h`),
   easy to miss in the log. So diff every noted instance (since #123 `elec25`
   and `elec50`) against **every seed's** rows by hand, and edit a retired root
   cause in `analysis_notes.csv`, not in `comparison.csv`.
6. Confirm the provenance:
   `.venv/bin/python -c 'import csv,sys;print(sorted({r["commit_sha"] for r in csv.DictReader(open("comparison.csv"))}))'` must print exactly one
   SHA, the checkout you built. Two mean a resumed run spanned a commit: discard
   the staging directory and re-run.
7. Check the machine in `comparison.run.json` and in **every** entry of
   `comparison_seeds.run.json`: host, CPU model, cores, memory, and a
   start-of-run load near zero, or, for a seed started straight after another,
   the previous solve's 1.0 decaying (say which in **Results**). A load that says
   the box was not quiet makes that seed unpublishable; a nonzero `resumed` means
   some rows' load is not in the record. Write each instance's median and range
   from the printed spread into **Results**, beside the published table.
8. Run `.venv/bin/pytest tests/python/test_minlplib_scip_baseline.py tests/python/test_minlplib_campaign_report.py`.
   The second is red on any regenerated table until step 2 has rewritten the
   blocks and `test_the_committed_tables_reproduce_the_readme`'s pinned numbers
   are updated by hand to match them. `test_safe_gap_reproduces_the_cpp_runners_published_gap_column`
   needs at least 15 distinct instances across the seeds with resolution to
   cross-check (a row within 0.01% of BKS has none), a zero-BKS one among them,
   so a new table can legitimately turn it red. Then run the full gated suite
   (`CLAUDE.md`, **Build & Test**) before pushing.

## Results

<!-- campaign_report:begin provenance -->
Latest run: **60s per instance, seed 1, feasibility tolerance 1e-6**, engine commit `4524460` (recorded per row in `comparison.csv`; the budget, the seed and the machine are read from `comparison.run.json`, which the driver writes at publish time; machine: simon-Legion-5-Pro-16ACH6H (AMD Ryzen 5 5600H with Radeon Graphics), 12 CPUs (12 usable), 13.0 GiB RAM, load 0.73 at start; 1 solve(s) at a time, 1 thread(s) each).
<!-- campaign_report:end provenance -->

**These are not the current engine's numbers.** `4524460` (2026-10-02)
predates compound moves on by default (`79a694a`), #201's structural-batch
fallback, #226's defaults (among them the objective tie-break on, `9e1cce9`)
and the #270-#273 search performance changes. No open issue regenerates this
table: #256's full-library campaign will give the newer cbls/SCIP score over the
full-library `bounds.csv` and writes nothing under `benchmarks/instances/`
(#306 scores it, #253 publishes it in the docs).

The published table is seed 1, pre-registered, of #123's three-seed campaign;
seeds 2 and 3 are in `comparison_seeds.csv` and their spread under "Across
seeds", beside this table, never in place of it. The tally, the gap buckets
and the anytime profile come from that one run and its committed
`anytime_trace.csv`, reproducible
from the commit each row's `commit_sha` names. The budget is wall-clock, so a
fixed seed does not pin the iteration count: re-running an *unmodified* binary
at one seed moved `kall_ellipsoids_tc02b` from 55.1% to 78.2%. Treat a row as
one draw. `--no-time-limit --max-iterations N` bounds a run by GLS iterations
alone and reproduces bit for bit (#136), a different measurement from this
table.

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

**Denominators** follow the rule above, stated once as `AGGREGATION_RULE` in
`benchmarks/minlplib/campaign_report.py` and printed by the driver's summary and
the report generator: "48 feasible" is of 50, and the verdict rows and the
anytime score are over the 48 feasible claim-set rows. The across-seed figures
apply it per seed (`SEED_AGGREGATION_RULE`). Every number inside a
`campaign_report` block is generated from the committed tables, and
`tests/python/test_minlplib_campaign_report.py` fails if a block differs from
the generator's rendering. Numbers outside the blocks are separate measurements
(other seeds, other commits, probes) or reference values; the generator's "Not
regenerated" list names each kind.

<!-- campaign_report:begin gap-buckets -->
Gap distribution over 45 of the 48 feasible claim-set instances: **29 within 0.01% of BKS, 33 within 1%, 36 within 10%.**

Five rows have a numerically zero BKS (`|BKS| < 1e-12`), for which the runner writes an *absolute* residual into the `gap_to_bks%` column rather than a meaningless percentage against zero: `ex14_2_5`, `ex14_2_4`, `mathopt1`, `least` and `prob09`. Those values are not percentages. The buckets above exclude `mathopt1`, `least` and `prob09`, whose residual is non-zero, and retain `ex14_2_5` and `ex14_2_4`, where objective and BKS are both exactly 0 and so are exact matches at any threshold. Excluding all of them instead gives 27 / 31 / 34 over 43 rows. Counting the excluded rows *as* percentages would have put `mathopt1` (gap cell 2.107e-15) and `prob09` (gap cell -3.339e-10) inside the "within 1%" bucket.
<!-- campaign_report:end gap-buckets -->

<!-- campaign_report:begin anytime -->
**Anytime score.** The MIPfeas Primal Integral (`benchmarks/mipfeas/primal_integral.py`) of the committed trace against BKS over the 60s budget — 0 is "at BKS from the first instant", 2 is "never feasible" — over the 48 instances outside the documented failures (`elec25` and `elec50`): **mean 0.180, median 0.011, shifted geometric mean 0.018**. A maximize row's trace is negated, so its reference is −BKS (at catalogue precision, from `bounds.csv`). BKS is not a proven optimum, so an incumbent past it scores a positive gap; and the zero-BKS rows left out of the gap buckets (`mathopt1`, `least` and `prob09`) are *in* this score, because the scorer's own zero test is 1e-6 absolute. The per-instance scores, with the reference each was scored against, are in `campaign_summary.json`; with one seed, each is one draw.
<!-- campaign_report:end anytime -->

### Across seeds, and what moved since the previous table (#123)

Seeds 1, 2 and 3, all at engine commit `4524460`, 60s per instance, one solve at
a time with one thread, `resumed: 0` in every record, on one host (AMD Ryzen 5
5600H, 12 logical CPUs, 13.0 GiB, per the run records). The start-of-run
1-minute loads (0.73 / 0.92 / 0.49) were not the "near zero" step 7 asks for.
Only seed 2's is step 7's back-to-back case (seed 3's last solve decaying);
seed 1's followed the preflight and runner build, seed 3's another session's
pre-push `ctest`. Neither is a tenant during the run, but both are judgements
from the timeline, not records. The run diary is on
[#123](https://github.com/spoorendonk/cbls/issues/123).

**Spread.** Feasible: **48 on every seed**. Matching BKS, over the 48 claim-set
rows: **26 / 27 / 28** (min / median / max; seeds 1/2/3: 28, 27, 26). `elec25`
and `elec50` are infeasible on all three seeds.

Per instance: `gap_to_bks%` of the published seed, and the median and range over
the three seeds as `campaign_report.py --seeds` prints them (a seed that did not
reach feasibility counts as worse than any gap; every row is feasible on all
three seeds except the two excluded ones, on none). `(abs)` marks a zero-BKS
row, whose cell is an absolute residual. The last column is `lns_repairs`
(`lns_repairs_accepted`) per seed. The previous table's column is on #123.

| instance | seed 1 (published) | median | range | LNS repairs (accepted), seeds 1 / 2 / 3 |
|---|---:|---:|---|---|
| `process` | 0.014 | 0.014 | 0.0125 – 0.0163 | 60 (1) / 63 (1) / 64 (0) |
| `st_e36` | 0.871 | 0.871 | 0.871 – 0.871 | 10 (0) / 10 (2) / 10 (3) |
| `ex4_1_1` | 1.30e-9 | 1.30e-9 | 1.30e-9 – 1.30e-9 | 79 (0) / 64 (1) / 65 (0) |
| `ex8_1_6` | 49.8 | 49.8 | 49.8 – 49.8 | 32 (0) / 32 (0) / 32 (0) |
| `alkylation` | 2.30e-6 | 3.20e-4 | 2.30e-6 – 0.0121 | 54 (0) / 56 (0) / 52 (1) |
| `nvs21` | 100 | 100 | 100 – 100 | 1 (0) / 1 (0) / 1 (0) |
| `ex4_1_2` | 0.00104 | 8.33e-5 | 7.13e-7 – 0.00104 | 12 (0) / 12 (0) / 12 (1) |
| `ex14_2_5` (abs) | 0 | 0 | 0 – 0 | 183 (83) / 181 (50) / 183 (41) |
| `mathopt5_3` | -1.81e-8 | -1.81e-8 | -1.81e-8 – -1.81e-8 | 115 (0) / 115 (0) / 115 (0) |
| `ex8_4_5` | 0.231 | 0.0655 | 0.0497 – 0.231 | 7 (1) / 4 (3) / 5 (1) |
| `nvs01` | 1.26e-8 | 266 | 1.26e-8 – 266 | 1 (1) / 1 (0) / 1 (0) |
| `ex4_1_3` | -9.26e-9 | -9.26e-9 | -9.26e-9 – -9.26e-9 | 100 (0) / 100 (0) / 101 (0) |
| `ex14_2_4` (abs) | 0 | 0 | 0 – 0 | 104 (55) / 92 (16) / 34 (7) |
| `ex8_1_1` | -1.78e-8 | -1.78e-8 | -1.78e-8 – -1.78e-8 | 70 (0) / 83 (0) / 83 (0) |
| `shiporig` | 1.64 | 1.64 | 0.00218 – 2.98 | 5 (0) / 4 (0) / 3 (0) |
| `nvs08` | -9.64e-8 | -9.64e-8 | -9.65e-8 – 1.61 | 1 (1) / 1 (1) / 1 (1) |
| `ex4_1_7` | 0 | 0 | 0 – 0 | 113 (0) / 113 (0) / 114 (0) |
| `eq6_1` | 2.53 | 2.53 | 2.5 – 2.56 | 0 (0) / 0 (0) / 1 (0) |
| `minlphi` | -8.64e-7 | -8.64e-7 | -1.55e-6 – -7.94e-7 | 17 (0) / 17 (0) / 16 (0) |
| `gear4` | 4.29e6 | 4.29e6 | 1.77e6 – 9.48e6 | 10 (2) / 10 (1) / 10 (4) |
| `mathopt5_7` | -1.01e-8 | -1.01e-8 | -1.01e-8 – -1.01e-8 | 86 (0) / 87 (0) / 86 (0) |
| `elec25` (excluded) | infeasible | — | — | 22 (0) / 22 (0) / 23 (0) |
| `st_e38` | -6.61e-9 | -6.61e-9 | -6.61e-9 – -6.61e-9 | 9 (1) / 10 (0) / 9 (0) |
| `ex8_1_5` | -4.75e-8 | -4.75e-8 | -4.75e-8 – -4.75e-8 | 40 (0) / 41 (0) / 40 (0) |
| `maxmin` | 0.00896 | 0.00706 | 5.90e-4 – 0.00896 | 3 (0) / 2 (0) / 2 (0) |
| `nvs02` | 1.17e-9 | 1.17e-9 | 1.17e-9 – 1.57 | 0 (0) / 0 (0) / 2 (0) |
| `ex4_1_8` | 4.34e-7 | -2.31e-9 | -6.69e-9 – 4.34e-7 | 133 (0) / 133 (1) / 133 (0) |
| `ex8_6_1` | 23.9 | 26.9 | 23.9 – 38.7 | 0 (0) / 0 (0) / 0 (0) |
| `nvs14` | 1.73e-9 | 1.73e-9 | 1.73e-9 – 1.73e-9 | 0 (0) / 0 (0) / 3 (0) |
| `st_e01` | 5.00e-9 | 5.00e-9 | 5.00e-9 – 5.00e-9 | 140 (0) / 155 (1) / 158 (0) |
| `chain50` | 19.5 | 35 | 19.5 – 35.7 | 0 (0) / 1 (1) / 2 (0) |
| `st_e40` | 1.36 | 0 | 0 – 1.36 | 22 (0) / 20 (0) / 21 (1) |
| `st_e09` | 2.00e-8 | 2.00e-8 | 2.00e-8 – 2.00e-8 | 73 (0) / 74 (0) / 73 (0) |
| `elec50` (excluded) | infeasible | — | — | 13 (0) / 13 (0) / 13 (0) |
| `nvs05` | 1483 | 574 | 177 – 1483 | 0 (0) / 0 (0) / 0 (0) |
| `ex4_1_9` | 0.00401 | 0.0259 | 0.00401 – 0.0317 | 41 (0) / 48 (0) / 34 (1) |
| `kall_ellipsoids_tc02b` | 37.3 | 37.3 | 0.0756 – 95.3 | 9 (0) / 10 (1) / 9 (1) |
| `nvs22` | 122 | 173 | 122 – 311 | 0 (0) / 0 (0) / 0 (0) |
| `mathopt1` (abs) | 2.11e-15 | 2.06e-14 | 2.11e-15 – 1.87e-13 | 39 (19) / 36 (0) / 36 (0) |
| `least` (abs) | 1.41e4 | 1.41e4 | 1.41e4 – 1.41e4 | 11 (3) / 11 (2) / 11 (3) |
| `tln2` | 0 | 0 | 0 – 0 | 10 (0) / 9 (0) / 11 (1) |
| `prob06` | -6.49e-5 | -3.18e-4 | -3.19e-4 – -6.49e-5 | 90 (0) / 90 (0) / 87 (0) |
| `ex6_2_11` | 0.0771 | 2.94e-4 | 7.54e-5 – 0.0771 | 7 (1) / 6 (6) / 5 (3) |
| `spring` | 2.61e-4 | 2.61e-4 | 1.85e-4 – 2.61e-4 | 8 (0) / 8 (0) / 8 (0) |
| `prob09` (abs) | -3.34e-10 | -3.34e-10 | -7.98e-10 – 1.00e-13 | 68 (12) / 68 (10) / 69 (18) |
| `ex6_2_6` | -5.78e-5 | -6.55e-5 | -8.20e-5 – -5.78e-5 | 4 (3) / 7 (7) / 4 (4) |
| `windfac` | 1.41e-8 | 1.41e-8 | -2.53e-4 – 1.41e-8 | 16 (0) / 15 (2) / 15 (2) |
| `st_e08` | 0.00102 | 0.00219 | 0.00102 – 0.00219 | 100 (0) / 98 (0) / 100 (0) |
| `ex6_2_8` | 1.27e-4 | 1.27e-4 | 1.05e-4 – 1.81e-4 | 6 (0) / 6 (0) / 6 (0) |
| `eg_all_s` | 88.3 | 3.43 | 3.43 – 88.3 | 0 (0) / 0 (0) / 0 (0) |

**What moved since the previous table** (one seed at `21086c2+107`, committed
in `0648067`), sorting claim-set rows by `gap_to_bks%` (infeasible worst): a
row is better when the old value is worse than all three new seeds, worse when
it is better than all three, else inside the new range; rows `matches-bks` in
the old table and on all three seeds are left out as unchanged. Feasible 46 →
48 (`nvs01` and `st_e40`); 18 better (those two and three zero-BKS rows among
them), 3 worse (`chain50`, `gear4`, `st_e08`, which no longer matches BKS), the
rest inside. This sums 291 engine commits between `0648067` and `4524460` with no
per-change ablation, so no row is attributed to a single change except `st_e40`
and `nvs01` (#102, measured beforehand), and it is not evidence that the search
got better in the sense #87 asks about. The per-row comparison and the
trajectory-changing issues are on #123.

**LNS repairs.** Over the 150 runs LNS ran 5716 destroy-repairs and kept 382
(6.7%), 292 of them on three zero-BKS instances (`ex14_2_5` 174, `ex14_2_4` 78,
`prob09` 40); `ex8_6_1`, `nvs05`, `nvs22` and `eg_all_s` ran none on any seed. A
kept repair is one the lexicographic (violation, objective) key accepted: these
counts say where LNS is engaged at 60s, not what it contributes (#143).

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

But the incumbent trace cannot be read as pure search progress: `record_best` tightens the objective bound by `1e-3·(|obj|+1)` per accepted solution — the relative step was fixed at 1e-3 until #241, which shrinks it toward 1e-6 only at a stall — so improvements are *floored* at roughly 0.1% steps. The measured median consecutive-incumbent ratio on `eg_all_s` (the instance with the most improvements) is 0.9989993 — 1 − 0.001001, against the bound step's 1 − 1e-3 — and it takes 15892 such steps (15893 incumbents) to walk from 1e9 down to 14.4.
<!-- campaign_report:end improvement -->

That instance is evidence about the bound-tightening step, not about how long the
search needs: read the quality column as a lower bound on what a larger step (or
a direct objective descent) might reach sooner.

**At `4524460` the feasibility argument for 60s has largely gone**: every
instance that becomes feasible does so within 6.5s on every seed (last:
`eg_all_s` 5.5s on seed 1, `chain50` 1.5s on seed 2, `nvs02` 6.5s on seed 3),
where the previous table had five arriving between 17s and 37s. 60s stays the
protocol budget for reasons independent of this run: #123's table is an A/B
against the previous one in which only the engine may differ, the SCIP baseline
is 60s, and quality still moves late (the block above). A budget change is a
separate, pre-registered experiment.

### The instances the published run left infeasible

<!-- campaign_report:begin infeasible -->
Left infeasible in this table: `elec25` and `elec50` (2 of 50).
<!-- campaign_report:end infeasible -->

Both are root-caused per row in `comparison.csv` (merged from
`analysis_notes.csv`) and infeasible on all three of #123's seeds. The published
note cells in `comparison.csv` and `comparison_seeds.csv` still call #110 the
tracker, as merged at run time; they stay, because each run record pins its
table's sha256 and `campaign_report.py` refuses a table (or leaves out a seed)
whose bytes no longer match, so editing a cell would unpublish the run.

| Instance | Verdict | Cause |
|----------|---------|-------|
| `elec25`, `elec50` | **bug**, untracked ([#110](https://github.com/spoorendonk/cbls/issues/110) closed as not planned on 2026-09-07; its elec criterion had been retired to #116 and #117, both closed) | Thomson problem: points on the unit sphere, Coulomb objective `+inf` wherever two coincide. The objective-encoding defects (#100) are fixed and no longer the blocker. The `.nl` declares no finite variable bounds, so the box is the ±1e9 inf-clamp; random init starts ~1e9 out, shrinking each variable toward 0 is a huge row improvement, and the search parks on the origin, a stationary point of every row `x²+y²+z²=1`. The Float jump offers a single *undamped* Newton step (`x0 - residual/grad`) plus `lb`/`ub`/midpoint; near the origin that step overshoots and is rejected, and because a candidate was *generated* the #107 escape probe is suppressed, so the variable freezes at score 0 (the probe fires only at exactly `x0 = 0`). Infeasible at violation ≈1 with the objective present and neutralised alike, so it is not objective-related. Tightening `inf_clamp` to 1 makes `elec25` feasible post-#100, because the clamp supplies the missing damping. |

`nvs01` and `st_e40` left this list in #123's regeneration, feasible on all
three seeds as #102 had made them; their rows were retired from the table and
`analysis_notes.csv` together. `nvs01`'s compound-move account
([#101](https://github.com/spoorendonk/cbls/issues/101)) was refuted: a
single-variable path reaches feasibility once #102 is in. `st_e40`'s limit cycle,
and the GLS weight bumps that changed nothing for 92% of a run, were fixed by
#102's unproductive-batch exit. Both retired root causes are written out on
[#102](https://github.com/spoorendonk/cbls/issues/102).

### Is the final objective set by the first feasible point? (#149)

On `nvs01` it is. [#134](https://github.com/spoorendonk/cbls/issues/134)
measured eight seeds at `09097de`, 60 s: all eight feasible, three at the BKS
12.4697 and the rest 17-266% above it, with Pearson `r = 0.945` between the
first feasible objective and the final one. The spread is seed-deterministic,
not noise, and not an escape failure (#101): the lever is the first feasible
point, not the descent after it. `eb9e1a5` re-measured `r = 0.9399`. A
single-seed `nvs01` row is one draw from that spread.

**Whether the roster behaves the same is open.** #149 closed with it deferred to
a multi-seed campaign, and #123's three seeds are short of the rule's four, so
its scorer over them returned `INCONCLUSIVE` with 0 eligible instances. One instance supports no claim about the engine either way, and no
engine change may be made on it.

**The campaign.** The runner publishes `first_feasible_objective` and
`time_to_first_feasible` on every row (recording them changes no trajectory).
The pre-registered scorer, `first_feasible_report.py`, and its campaign recipe
(eight seeds, 60 s, all 50 instances) are in git history: `git show
047f2d4c:benchmarks/minlplib/first_feasible_report.py`, and this section at
`047f2d4c`. A verdict from it would not authorise an engine change, which #149
sends to its own issue, hypothesis and regression test.

## SCIP baseline

An independently run open-source yardstick for the roster (#89), written by
`../../minlplib/reference_solve.py`: per-row results in `scip_baseline.csv`, the
labelled three-way join in `comparison_all.csv`. **Why SCIP:** BARON is
commercial and reachable only through the NEOS queue, so it cannot be batch-run
reproducibly; Couenne has seen little development since ~2018 and is generally
outperformed on this family; SCIP's nonconvex spatial branch-and-bound is
purpose-built, benchmarked on MINLPLib in *Global Optimization of Mixed-Integer
Nonlinear Programs with SCIP 8.0*
([PDF](https://optimization-online.org/wp-content/uploads/2022/12/scip8_minlp.pdf)),
and already a repository dependency.

**What is matched.** SCIP reads the **same `.nl` files** through its own AMPL
reader, so no formulation drift can enter. Same roster and order (`bounds.csv`),
same 60s per-instance wall-clock budget, one thread each, and the same
feasibility tolerance: CBLS defaults to 1e-6 and SCIP's `numerics/feastol`
default is 1e-6. That parameter is left unset; the runner reads the live value
back on every solve and writes a `FEASTOL-MISMATCH` note if SCIP's default ever
moves. SCIP rows are scored by ports of the runner's `safe_gap` and its two-band
BKS classification, so `gap_to_bks%` means the same on both; the ports are
pinned by `tests/python/test_minlplib_scip_baseline.py`, which recomputes them
against the gap column the C++ binary wrote, at its six-significant-digit
resolution.

**What is not matched, by construction.** SCIP proves a dual bound; CBLS, a
primal heuristic, proves none, so only the primal columns are like-for-like and
`comparison_all.csv`'s `dual_bound` holds what *that method* proved (NaN on CBLS
rows). SCIP's solution is checked with `Model.checkSol(original=True)`; one it
cannot re-validate is not published as feasible. A CBLS row is checked twice
since #205 (the committed `comparison.csv` predates it): the runner's own
re-check reads the DAG values the search optimised, so a shared evaluation bug
passes it (#205: `sqrt` of a negative read as 0.0), and so `run_benchmark.py`
has SCIP check each verified row's assignment (`--solution-dir`) and objective
on its own evaluation of the `.nl` (`benchmarks/minlplib/independent_check.py`)
before publishing. A rejected row is published as `VERIFY-FAILED(independent:
...)` with objective, gaps and first-feasible pair blanked. This needs the
`benchmarks` extra (pyscipopt).

<!-- campaign_report:begin scip-verification -->
Zero rows in this run failed that check.
<!-- campaign_report:end scip-verification -->

Run at **SCIP 10.0.2 / PySCIPOpt 6.2.1**, 60s per instance, every parameter
other than `limits/time` and `randomization/randomseedshift` at its shipped
default (presolve, cuts, symmetry and the full primal-heuristic set on). The
seed shift is 0, SCIP's default; the flag exists for a multi-seed re-run. The
configuration (versions, budget, seed) is in every row's `scip_version` column.
Measured, not assumed: `timing/clocktype` defaults to 2 (wall clock), so
`limits/time` is the same kind of budget the CBLS runner imposes; instance
reading is outside it on both sides (`read_seconds`, total below); the solve
stays single-threaded (CPU/wall ~1.0).

<!-- campaign_report:begin scip-read -->
SCIP's `.nl` reads total 0.076s across the roster (`read_seconds`).
<!-- campaign_report:end scip-read -->

Where SCIP proves no dual bound it reports its `1e20` infinity sentinel, a
*finite* float; the runner folds that to `NaN` at capture, so `scip_dual_bound`
and `scip_gap%` read `NaN` there, as on the CBLS rows.

**Hardware.** The SCIP run was on an AMD Ryzen 5 5600H (12 logical cores, Linux
7.0), one core in use; the CBLS run's machine (the same CPU model) is in the
provenance block under **Results**. The runs were made at different times: two
runs on one hardware class, not one controlled measurement. Both run a fixed
60s per instance, so:

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

`elec25` reaching a near-BKS feasible point in SCIP confirms an engine gap, not
hardness (the undamped Newton jump; see the root-cause table above); at 50
points SCIP does not close `elec50` either, but reaches the feasible region.
`st_e36` is the row the other way: SCIP spends the budget and returns only a
dual bound. `nvs01` and `st_e40`, on this list in the previous table, are
feasible on every seed since #102.

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

`ex8_1_5` is the sharpest: its two variables are unbounded, so SCIP's dual bound
diverges and it makes no progress, while CBLS reaches the published optimum. It
was CBLS's *worst* row before #107.

### Objective quality under the #102 unproductive-batch exit

#102's exit (`GFJConfig::unproductive_iterations`) ends an FJ batch that has
stopped reducing the real rows' violation and takes the diversification kick as
due, which changes the trajectory on every instance. A paired probe (`0ffbf9d`
against `b7f8a50`; eight instances, seeds 42/1/2/3, 10 s, interleaved, serial;
not the published protocol) moved four rows by far more than its measured
noise floor of 3-4 gap points: `alkylation` ~99.98 → ~0.1%, `st_e36` 32-40 →
0.9-3.4%, `st_e40` infeasible → BKS (30.4142135 at 10 s on all four seeds, and
at 60 s on seed 42) and `nvs01` infeasible → feasible. `ex8_6_1`, which had held
the change, came back level with main on every seed (69.67 / 78.87 / 64.17 /
61.16 against 69.67 / 78.87 / 64.17 / 65.48), where the arm gated only on the
stagnation count had lost 6-20 gap points to LNS repairs. Once a feasible
solution exists the exit draws only a perturb, never an LNS destroy-repair:
`tests/test_search.cpp` pins that on a plateau model (red at 25 repairs against
0). The per-seed table, the noise-floor measurement and the mechanism are on
[#102](https://github.com/spoorendonk/cbls/issues/102); #123 regenerated the
published rows.

### What #107 accounted for

#107 was found by noticing that instances with at least one **free (unbounded)**
variable did far worse. From the committed tables and the `.nl` bounds, over the
rows each group solves whose `|BKS| >= 1e-4`:

<!-- campaign_report:begin free-variables -->
| | instances | eligible | within 10% |
|---|---|---|---|
| ≥1 free variable | 16 | 13 | 6 |
| no free variables | 34 | 28 | 26 |

Within 10% with at least one free variable: `shiporig`, `ex8_1_5`, `maxmin`, `st_e40`, `spring` and `windfac`.

**Before #107** — a table that is not committed, so these counts are fixed constants in `campaign_report.py` — within 10% were 1 of 12 eligible rows with a free variable (`maxmin` only) and 20 of 27 without. Against this table: **1 of 12 → 6 of 13** with a free variable (`shiporig`, `ex8_1_5`, `st_e40`, `spring` and `windfac` join `maxmin`), and 20 of 27 → 26 of 28 without. That difference spans every engine change between the pre-#107 table and this table's commit, not #107 alone.
<!-- campaign_report:end free-variables -->

**#107's own share**, measured against its AFTER run (`21086c2+107`, the previous
table): 3 of 12 free-variable rows within 10% and 19 of 27 without, so #107
explained **2 of the 11** free-variable misses (`shiporig` and `ex8_1_5`); the
no-free group's −1 is `eq6_1`, which an A/B showed is not #107's
(bit-identical between arms). No finer win/loss tally is published:
`comparison.csv`'s six significant digits are below the tie band on several
rows.

**One catalogue row looks stale.** On `ex6_2_6` SCIP proves optimality at
−3.51174e−06, better than MINLPLib's primal bound (−2.60253e−06) and marginally
past its dual (−3.49783e−06), which a valid dual bound for a minimize instance
cannot be. The absolute difference is 9e−07, so the two-band rule labels it
`matches-bks`, but `gap_to_bks%` reads −34.9%, the tiny-objective artifact. The
same caution applies to `ex6_2_11`, `least`, `mathopt1`, `prob09`, `ex14_2_4`
and `ex14_2_5`.

## Integrality

NL columns flagged integer/binary are built as CBLS `Int` variables, so the
mixed-integer instances are solved as genuine MINLPs. The NL header gives
integer *counts* per category, not positions; the positions follow Gay's
variable ordering ("Hooking Your Solver to AMPL"):

| Order | Category                                | Count         | Integers            |
|-------|-----------------------------------------|---------------|---------------------|
| 1     | nonlinear in both constraints and objs  | `nlvb`        | last `nlvbi`        |
| 2     | nonlinear in constraints only           | `nlvc - nlvb` | last `nlvci`        |
| 3     | nonlinear in objectives only            | `nlvo - nlvc` | last `nlvoi`        |
| 4     | linear arc variables                    | `nwv`         | —                   |
| 5     | other linear                            | remainder     | —                   |
| 6     | binary                                  | `nbv`         | all                 |
| 7     | other integer                           | `niv`         | all                 |

Three checks guard it. Two are *count* checks, blind to a correctly sized set at
the wrong offset: the reader fails if its positions do not account for exactly
the header's count, and the runner compares the recovered count with
MINLPLib's `nbinvars + nintvars` (`n_disc_vars_bks`) and reports `integrality
mismatch` (the Results tally's row). The positions themselves are pinned by
`tests/test_minlplib.cpp` ("NL reader recovers discrete-variable count and
positions"), which asserts the exact `var_is_discrete` vector per block: an
earlier revision had the objective-only block wrong and both count checks
passed.

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

## Defaults fitted on this roster (#233)

A parameter is fitted on one roster and reported on another
([docs/measuring.md](../../../docs/measuring.md), "Performance and measurement"). These
defaults were chosen by measurement on these fifty instances, so a table
published on them is in-sample for each one:

| Default | Chosen by | Measurement on this roster |
|---|---|---|
| unproductive-batch exit at 300 iterations | #102 | a grid {100, 300, 1000} over the fifty, one seed at 2s; #145 checked the transfer at 10s only (HELDOUT.md) |
| compound moves (Novelty Jump) on | `79a694a` | one seed at 5s at `b8f52f4`: 16 instances better, 7 worse |
| near-value kick off (removed by #382; in git history at `8803b404`) | #226 Run B, #219 | 3 seeds at 5s: gap ratio 1.225, `nvs01` lost on seed 1 |
| quadratic roots and equality balance removed | #226 Run B, #223, #224 | 3 seeds at 5s: neutral or worse |

The objective tie-break (on since `9e1cce9`, #220) was decided on MIPfeas's
smoke roster (#226 Run A). #211's changes to the FJ tolerance,
`sample_size_general` and the weight carry-over were correctness fixes or moves
to the reference algorithm's values, not fitted.

The held-out roster in `heldout/` was fitted to by none of these, but it is no
longer unseen (the runs on it are listed in HELDOUT.md); #331 decided the presolve
and compound-propagation defaults (both kept off) on 25 of its instances, which
are in-sample for them. #228's smoke (`0c0a945`,
60 s, seeds 1-2) found no in-sample gap between the two rosters (cbls/SCIP
GM(gap+1) 1.81-1.84 held-out, 1.75-1.88 published); #256 runs the full library
instead of a full-scale repeat.

**The tie-break and the `search_config` cell.** The cell records
`objective_tie_break` only when it is off (see "Ablation arms"), so a default
row's cell is the same before and after `9e1cce9`. The committed tables, at
`4524460`, ran the tie-break **off**; a table regenerated now runs it **on** with
an unchanged cell, and only `commit_sha` tells them apart. Any text quoting a
table names its engine commit and the tie-break's state.

## Caveats

- CBLS is a primal heuristic; large gap-to-BKS on hard multimodal instances is
  expected and acceptable (per issue #72).
- The roster is reproducible from the CSV but will drift as MINLPLib updates its
  catalogue; re-run `download.py` to refresh.
- Both runs are single samples on a wall-clock budget. Most SCIP rows end with a
  proof rather than at the limit (head-to-head table), but those that hit the
  limit are as draw-dependent as the CBLS numbers.
- The size budget (`nvars <= 150`, `ncons <= 150`) is a roster choice pinned
  with its membership, not a reader limit (the full-library selection has none,
  #283). It runs SCIP well inside its comfortable range, which is the right way
  round: the comparison should not flatter the engine under test.
- **The roster is the subset CBLS can express**, the largest confounder in the
  SCIP comparison and pointing the *opposite* way to the size caveat: the
  selection filter admits an instance only if every operator column it uses is
  in the CBLS DAG's op set, and the NL reader rejects `S` and `F` segments (see
  "Selection method" and "Operator support"). The published table compares on
  CBLS's expressible domain, not on MINLPLib. The full-library status (#283)
  drops the size, problem-type and convexity filters and the
  `opvcpower`/`opcvpower` exclusion; #256's campaign runs both solvers on its
  modelled set, its score not yet published (#306, #253).

## Portfolio A/B: the shared objective bound (#179)

`cbls_minlplib --threads N` has no shared-bound switch, so #179 ran its A/B
through `cbls_minlplib_portfolio` (`portfolio_ab.cpp`, which ran
`ParallelSearch` with the runner's per-worker hook and LNS) and paired and
scored the arms with `portfolio_ab.py`. Both are in git history at `047f2d4c`
(`benchmarks/minlplib/`).

Result (4 threads, 20 s, seeds 1001-1004, on 10 instances a pre-registered
control-only pilot admitted because the control runs behind the portfolio's
best bound, so where the mechanism has something to act on, not a roster
average): the primal integral improved on all 10 (sign test p = 0.002, instance
mean -0.083, t_9 95% CI [-0.156, -0.010]); the final gap improved on 8, got
worse on 1 and was flat on 1 (p = 0.039; CI [-0.183, +0.003]). All 40 runs were
feasible in both arms. The engine binary and how main differs from it are on
#179.

The MIPfeas half had no committed driver: each pair is two `cbls_mipfeas
--instance I --inst-dir benchmarks/instances/mipfeas --out-dir D --budget 30
--seed S --threads 4 --commit <sha>` runs, the control adding
`--no-share-bound`, seeds 1001-1006 over the 10 pilot-admitted instances, run
serially under the machine lock, both arms of a pair back to back and the arm
order alternating with seed parity, each scored with `primal_integral.primal_integral` against the roster
optimum over [0, 30 s], and scored with `portfolio_ab.py analyze` (`instance`,
`seed`, `share`, `pi`, `gap` rows).

A held-out roster lives in `heldout/`. No shipped default was fitted to it except #331's presolve and compound propagation (both off), which were decided on 25 of its instances; method, pool size and composition are in [HELDOUT.md](HELDOUT.md) (#144).
