# Assumption log — issue #138 (MIPfeas solution verification)

- **Which third-party reader**: the issue said "both third-party solvers already
  available as benchmark dependencies can read the instance format; pick one".
  Resolved as: **PySCIPOpt** (`benchmarks/mipfeas/verify_solution.py`). Because:
  `cpsat_solve.py` builds the CP-SAT model with OR-Tools' own MPS reader, so
  verifying a CP-SAT solution with OR-Tools would let a shared reader defect
  cancel out on exactly the half this check exists to cover. SCIP's reader is
  independent of *both* engines, so one verifier covers both engines' solutions.

- **Tolerances**: the issue asked for "a stated tolerance" and hinted at
  absolute+relative. Resolved as, all constants of `verify_solution.py`:
  row activity `1e-6 + 1e-9 * row_scale` (row_scale = max(|lhs|, |rhs|,
  sum |a_ij x_j|) over the finite sides); variable bound `1e-6 + 1e-9 * |bound|`;
  integrality `1e-6` absolute; objective consistency `1e-6 + 1e-6 * |objective|`
  (the relative term is the CBLS runner's own objective-drift gate, since this
  check re-measures the very quantity that gate accepts — a tighter rule
  downstream of it would reject runs the engine was entitled to publish).
  Because: the `1e-6` absolute terms are the engine's own stated feasibility
  tolerance (`--feas-tol`, `kDefaultFeasibilityTolerance`), which keeps the two
  checks comparable without sharing anything; the `1e-9` relative terms sit ~7
  orders above double round-off, which is what a row of millions of nonzeros can
  accumulate, and MIPLIB coefficient ranges make a purely absolute rule either
  vacuous or unmeetable.

- **The borderline rule**: the issue demanded "a defined threshold, not a
  judgement call at scoring time". Resolved as: fail **iff** some violation
  exceeds its tolerance; at or below it, pass. A pass whose worst
  violation/tolerance ratio reaches `MARGINAL_FRACTION = 0.1` is published as
  `pass` with `marginal=1` — a signal to look, never a reason to withhold.

- **Objective consistency is part of verification**: not named in the issue.
  Resolved as: the verifier recomputes `c.x + offset` from the instance file's own
  cost row and fails the row on a mismatch. Because: an adapter that dropped a
  cost coefficient or lost the objective constant publishes a number that is not
  the objective of the point it published, and the reference-value check cannot
  see that.

- **What "publishes no objective and no derived score" means for the
  aggregates**: resolved as: a failing row's `objective`, `final_gap` and
  `primal_integral` are all blank/NaN and the row is **excluded** from the
  engine's aggregates, counted separately as `verification_failed`. Because:
  scoring it 2.0 (never-feasible) would itself be publishing a derived score, and
  a misleading one — the run did find a point, it was just rejected.

- **A feasible row with no verdict**: resolved as: `primal_integral.py` requires a
  verdict by default (`--allow-unverified` opts out, and is what an older results
  directory needs). Because: acceptance criterion 1 is "every row reported
  feasible has an independent verdict", which only a default-on rule guarantees.

- **Unsupported constraint types**: resolved as: any SCIP constraint handler other
  than `linear` makes the verdict `error` (never `pass`), and the row is withheld
  like a failure. Because: a constraint nobody checked must not read as a
  constraint that held. Surveyed the roster — all 233 instances are pure linear
  MPS (no SOS/INDICATOR/quadratic/OBJSENSE sections; 3 use RANGES), so this path
  is a guard rather than a live limitation.

- **Solution file format**: not specified. Resolved as: the MIPLIB-style
  `=obj= <value>` + `<name> <value>` text format, `#` comments ignored, written by
  both runners at 17 significant digits. Because: it is a published convention a
  third-party checker could also consume, and it is cheap to write from C++ and
  from Python.

- **Where verification runs**: resolved as: a separate step inside the driver's
  per-job unit (`run_benchmark.py`), writing `<engine>/<instance>.verify.json`
  beside the result. Because: it keeps verification resumable and independent of
  the runner processes (so the CBLS runner never verifies itself in-process), and
  it keeps `cpsat_solve.py`/`run_benchmark.py` edits small and separable for #137.

- **Test instance**: the issue asked for "a small instance with a known solution".
  Resolved as: a hand-written 3-column / 3-row MPS (integer + continuous columns,
  L/G/E rows, a RANGES entry, an objective constant) with a known optimum,
  written by the test itself — its integer column named `x#1`, because 13 roster
  instances name columns that way and a `#`-anywhere comment rule would make
  every solution on them unparseable. Plus an end-to-end pass over the vendored
  `benchmarks/instances/miplib-fj/pk1.mps.gz` (86 columns, 45 rows) through the
  CBLS runner, which is the only place the two readers' agreement on a real
  fixed-format MPS is pinned. Because: the 233 roster instances are gitignored
  (~546 MiB) and a default-suite test cannot depend on them.

- **`smoke_comparison.csv` is not regenerated**: the committed wiring-check table
  predates these columns. Resolved as: left as-is with a note in the benchmark
  README. Because: regenerating it means an 11-instance timed run, which this
  session is forbidden to do (and which would be invalid on a shared machine).

- **The verifier streams the constraint matrix**: not called for by the issue.
  Resolved as: `Instance.rows` is a factory yielding one row at a time, never a
  list. Because: PySCIPOpt builds a fresh `str` key per nonzero (~245 bytes,
  measured on `supportcase7`: 2.85M nonzeros, 698 MiB RSS), so holding the matrix
  would cost ~6.7 GB on `square47` — more than the `--mem-limit-gb 6` the README's
  own run command sets, and the verifier runs inside that cap.

- **A verification the driver killed is retried; one the verifier itself reached
  is final**: resolved as `DRIVER_WRITTEN_VERDICT_REASONS` in `run_benchmark.py`.
  Because: a timeout or a memory cap under load is transient, and treating it as
  final would withhold the row for good — recoverable only by `--force`, which
  pays for the whole 600s search again. Re-running a check that cannot succeed
  (an unsupported constraint type) never converges, so those stay sticky.
