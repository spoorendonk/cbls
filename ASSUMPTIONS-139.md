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
