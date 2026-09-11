# Assumptions — issue #139

- **"The committed smoke results"** (acceptance criterion 8): the issue and the
  brief both speak of regenerating the report over them, but **no results
  directory is committed anywhere in the tree** — only `smoke_comparison.csv`,
  which is the scorer's *output*, not its input, and which cannot be regenerated
  from itself. Resolved as: commit a frozen results directory of my own,
  `benchmarks/mipfeas/testdata/`, together with the two artifacts scored from it,
  and make `tests/python/test_mipfeas_parity.py` regenerate both and compare byte
  for byte. Because: that is the reproduction the criterion asks for, and a
  fixture is the only way to hold a defect still — a clean run pins none of the
  counters the report exists to show.

- **Leaving the published `smoke_comparison.csv` alone.** Resolved as: not
  regenerated, and the roster README now says why it cannot be. Because: the
  scorer's inputs for it were never committed, and since #138 its rows carry no
  independent verdict, so re-scoring what survives would need
  `--allow-unverified` and would publish no objective for any row. The README
  already recorded the same staleness for two earlier column additions; I
  extended that paragraph rather than inventing a measurement.

- **Fixture realism.** Resolved as: two of the thirteen fixture instances (`mad`,
  `gen-ip054`) are a genuine 2-second run of both engines over the vendored
  `benchmarks/instances/miplib-fj/` copies, verified by `verify_solution.py`; the
  other eleven are constructed, one per branch of the report. Because: `mad`
  carries the real shape disagreement the issue points at, and a real run carries
  no defects at all, so the constructed rows are what make the test able to fail.
  `benchmarks/mipfeas/testdata/README.md` says which is which and that none of it
  is a measurement.

- **Setup-time magnitude** (criterion 7). Resolved as: the mechanism is reported
  and the columns are emitted, but **no magnitude is claimed** for the full
  roster. Because: the brief forbids a timed campaign, the published smoke
  results predate the measurement, and the only figures I could produce are
  fixture numbers and two tiny instances where setup is ~2 ms against a 2 s
  budget. The report states "No setup time was recorded in this results
  directory" rather than guessing when a directory carries none.

- **Trace-source vocabulary.** `cpsat_solve.py` already wrote `trace_source`
  ∈ {`log`, `final_only`}; the CBLS runner wrote nothing. Resolved as: CBLS now
  writes `callback` (its progress callback is the analogue of CP-SAT's log) or
  `final_only`, and the scorer treats `{log, callback}` as healthy. Because: a
  shared `log` would be untrue for CBLS, and a blank column made the trace-health
  denominator uncomputable on half the table.

- **Trace-health denominator.** Resolved as: rows the engine *reported* feasible,
  stated in the report. Because: a run that found nothing has no incumbent
  profile to have, and CP-SAT marks every such row `final_only` — counting them
  would make the degraded count read as the no-solution count.

- **Parity exclusions.** Resolved as: a row that never ran, did not search, or had
  its objective withheld is `excluded` and drops its whole instance from the
  comparable set, which is reported with its own denominator and an
  instance-by-instance reason table. Because: calling a killed job "did not reach
  feasibility" charges a harness failure to the search, the same mistake `not_run`
  has always been kept out of the aggregates to avoid.

- **Free-row count measured on the CP-SAT side.** The benign-shape rule needs to
  know how many free rows the baseline keeps. Resolved as: `cpsat_solve.py`
  counts linear constraints with both bounds infinite. Because: the CBLS MPS
  reader discards additional `N` rows without counting them, and adding a counter
  there means editing `src/io/mps_reader.cpp`, which CLAUDE.md holds diff-clean
  against its vendored upstream.

- **Report lives in `primal_integral.py`** rather than a sibling module.
  Resolved as: one file. Because: a sibling would need the repo-root `sys.path`
  shim `benchmarks/minlplib/run_ablation.py` carries, and `Scored` plus the
  verification constants would have had to move down the dependency chain to
  avoid a cycle — more moving parts than one consumer justifies.

- **Orchestrator-relayed #138 review findings.** Taken on this branch as asked.
  `verifier_died` is now capped at `MAX_VERIFY_ATTEMPTS` retries rather than
  classified by signal alone, because a pyscipopt segfault arrives as a signal too
  and signal-classification alone would still never converge; the signal is
  recorded in the message, where it distinguishes "try again with more memory"
  from "this checker crashes on this model".

## Added after self-review

- **The fixture results directory was gitignored.** `.gitignore`'s blanket
  `results/` swallowed `benchmarks/mipfeas/testdata/results/`, so criterion 8's
  test passed only on this machine and failed on every fresh clone — while the
  guard test beside it stayed green, because it reads the *tracked*
  `expected_report.md`. Resolved as: a `!benchmarks/mipfeas/testdata/results/`
  negation with the reason written beside it, and the 67 files committed.
  Because: a fixture is a test *input*, not a run artifact, and this is exactly
  the failure the criterion exists to make impossible.

- **The shape cross-check did not implement the rule it prints.** All three
  reviewers found it independently: when the two engines' constraint counts
  differed by the free-row count, the verdict was `benign` without consulting the
  checker at all, so a SCIP reading that matched neither engine was published as
  benign. Resolved as: a benign verdict now requires the checker to agree with the
  adapter, and two verdicts disagreeing with *each other* about the same file are
  themselves flagged. Three new tests, each shown red on the pre-fix code.

- **`verification_row_tolerance` has no producer on this branch.** The keys it
  reads (`max_row_tolerance`, `loosest_row`) are written by `f42f883`, which is on
  `feat/138`'s tip and **not** in this branch's history — the merge base is
  `eee18fd`. Resolved as: keep the columns, read them defensively, and say in both
  READMEs that they are blank for a verdict reached before the verifier recorded
  the figure. Because: the orchestrator has said it will rebase onto that tip, and
  a column that fills in on rebase is better than one deleted and re-added. **The
  orchestrator must not resolve `verify_solution.py` in this branch's favour** —
  a diff against `feat/138`'s tip reads as though 139 reverted `f42f883`.

- **Two "feasible" counts in one report.** Parity counts rows that published a
  feasible objective; trace health counts rows the engine *reported* feasible,
  withheld ones included. Resolved as: keep both denominators and say in §5 that
  it can exceed §2's, rather than narrowing trace health. Because: a withheld
  row's profile still exists, and its health is a fact about the harness rather
  than about the verdict.

- **Parity excludes a killed job; the anytime aggregate still scores it 2.0.**
  Pre-existing aggregate behaviour that the new §2 made visibly contradictory.
  Resolved as: a sentence in §6 saying so and pointing at the defect counters,
  not a change to the metric. Because: changing what `scored` covers moves a
  published number, and the two sections genuinely answer different questions.

- **Criterion 7's "separate columns".** Read as one column for read+build+
  propagate (`setup_seconds`) and one for solve (`solve_seconds`), which is what
  the criterion's wording asks for. Both runners record the halves in the result
  JSON (`read_seconds`, `build_seconds`) for diagnosis; propagation is inside
  `build_seconds` and is not separable without timing it inside `mps_to_model`.
