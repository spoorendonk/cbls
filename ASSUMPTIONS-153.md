# Assumptions — issue #153

- **Which tally decides the exit status**: the issue says "make the exit status
  reflect the error tally" but the runner keeps seven non-feasible counters.
  Resolved as: `Tally::errored` alone (read/build/solve exceptions). Because:
  `skipped_unsupported` and `not_found` are coverage gaps, which the issue
  explicitly requires to stay exit 0; `failed_nonfinite`, `verify_failed`,
  `near_miss` and `nonfinite_obj` are *measurements* — a search ran to
  completion and the row says what it reported — so failing the process on them
  would make a legitimate result look like a crash.

- **Exit code value**: resolved as 3. Because: the runner already uses 1 ("no
  roster to run", and an exception escaping `main`) and 2 (bad flag or an
  output file that would not open, per `benchmarks/common/runner_args.h`), so a
  consumer has to be able to tell an error tally from those. Mirrored in Python
  as `run_benchmark.RUNNER_EXIT_ERRORED` with a test pinning it against the C++
  literal.

- **Provoking a solve that throws**: the brief asks for the cheapest honest way
  and forbids production code that exists only to be broken. Resolved as: an
  `.nl` declaring **zero variables and one objective**. It reads and it builds;
  `cbls::solve` then throws `var id out of range`, which is the runner's real
  `solve-error` path, reached with nothing added to `src/` or the runner. Found
  by probing candidate fixtures against the built binary, not by reading.

- **`not-found` is exit 0**: the issue names only `unsupported` as the coverage
  gap that must not fail the run. Resolved as: `not-found` stays exit 0 too.
  Because: the runner buckets it apart from `errored` for the same reason — an
  instance nobody downloaded is a gap in the roster on disk, not a failure of
  the solve — and collapsing it into the error status would make a partial
  instance directory indistinguishable from a crash.

- **Denylist vs allowlist**: the brief allows either inversion or a drift test,
  and asks for the weighing. Resolved as: **both**. Inverted
  `ablation_report` to `COMPLETED_SEARCH_NOTES` (the seven notes a completed
  search writes), so an unknown note is held out, counted and named on the
  disclosure line — fails safe. The inversion's own risk is the mirror one (an
  allowlist that falls behind the runner silently discards real measurements),
  so `test_every_completed_search_note_the_runner_writes_is_allowlisted` sweeps
  `minlplib.cpp`'s four note-composing call sites and holds each literal against
  the list. `NO_SEARCH_NOTES` survives, no longer as a classifier but as the
  labels the report collapses known held-out notes to.

- **Re-runnability over an already-collected `results.csv`**: the inversion
  changes the verdict for a row whose note is not on the allowlist. Resolved as:
  accepted, and checked — every note in the published roster's rows
  (`feasible`, `matches-bks`, `better-than-bks`, `within-tolerance-of-bks`,
  `infeasible(...)`) is allowlisted, and a row that is newly held out is
  disclosed by name rather than dropped. What did have to change is the *test
  fixtures*: `run_rows`/`write_results`/`_campaign_csv` wrote no note at all
  (or `"0"`), which the runner never does, so they now carry the note the runner
  would have written. No assertion was weakened.

- **The driver's row for a failed run**: the brief asks that the row produced
  after the change not be worse than the runner's own. Resolved as:
  `execute_runs` records the *runner's* row when the exit status is
  `RUNNER_EXIT_ERRORED`, the row is readable, and its note is not one a
  completed search writes; otherwise `failed_row`. Because: `failed_row` NaNs
  `primal_bks`, `dual_bound` and `n_int_vars`, which the runner reads from
  `bounds.csv` and the driver never does, and `solve-error` is a more precise
  cause than `exit-3`. Both rows are held out identically, so the scoring is
  unaffected. A non-`3` exit (a signal) is never trusted with its leftover row,
  and a nonzero exit beside a row claiming a result is recorded as
  `runner-failed-row-claims-a-result`.

- **`run_benchmark.py` (the publish driver) now stops on a thrown instance**:
  its `run_roster` already raises on any nonzero exit, so the exit-code change
  means a solve that throws halts the publish run instead of staging a
  `solve-error` row into the published table. Resolved as: kept, with the
  message extended to say what happened and that a re-run will hit the same
  throw. Because: publishing a comparison table containing a row that measured
  nothing is the defect one level up, and the committed
  `comparison.csv` contains no such row today.

- **Test counts**: the Python suite went 411 -> 423 (12 new tests, none of them
  binding tests). README.md line 75 updated; the C++ roster is untouched at 361,
  so the other six hard-coded places are unchanged.
