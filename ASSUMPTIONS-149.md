# Assumptions — issue #149

Scratch file for the orchestrator; delete before merging.

- **What "first feasible" means**: the issue says "where the first feasible
  solution lands" without saying whether that is before or after the inner
  solver's polish, or whether a later feasible point can displace it. Resolved
  as: the moment `have_feasible_` first turns true in `record_best`, i.e. before
  the `FloatIntensifyHook` polish in the same batch and before any bound
  tightening; a latch, never re-armed. Because: #149's whole contrast is
  arrival against descent, and the polish is already descent. It also matches
  what #134's own measurement could have observed through the trace callback.

- **The sentinel for "never recorded"**: `SearchResult::objective` uses `+inf`
  for "nothing to report", but `+inf` is a value the #100 witness path genuinely
  produces for a first-feasible objective. Resolved as: NaN for both new fields,
  with `time_to_first_feasible` as the authoritative latch — NaN iff no feasible
  point was recorded — and `first_feasible_objective` allowed to be NaN/±inf on a
  run that DID reach feasibility. Because: the two readings must not collide, and
  the time is the only one that can never legitimately be non-finite.

- **ParallelSearch**: `solve_portfolio`/`solve_deterministic` compose their
  result field by field and drop such fields (the documented `escape_probe_armed`
  caveat). Resolved as: leave them dropped, document it on the fields, pin it
  with a test. Because: the pool carries only state + objective per solution, so
  a meaningful aggregate would need a new pool field and a decision about what
  "the first feasible point of eight workers" even means — out of scope for an
  instrumentation issue, and NaN ("not recorded") is the honest reading.

- **Column position in `comparison.csv`**: appended after `lns_repairs_accepted`
  and before `search_config`, which the runner's own comment requires to stay
  last. Because: they describe the run, not the arm, which is the same reason the
  LNS counters sit there.

- **The pair-type rule (commit `84d87f9`)**: resolved as a `FirstFeasibleCells`
  struct that every row writer goes through, exactly like `LnsCells`, both cells
  defaulting to `"NaN"`. Because: the two are known together or not at all, and
  a 0 in `time_to_first_feasible` would be the most favourable possible reading
  ("arrived instantly") of a row where nothing ran.

- **Which campaign the orchestrator should run**: the issue says "a serial run on
  an idle machine" without naming a driver. Resolved as: eight runs of
  `run_benchmark.py` (one per seed, all to scratch paths), not `run_ablation.py`.
  Because: the ablation's arm set is fixed at 4-5 arms, so it would cost ~12h and
  mix in configurations the correlation does not ask about; `run_benchmark.py`
  already carries the dirty-tree, Release and resume guards and takes `--seed`.

- **The seed set**: eight seeds (1 2 3 7 11 13 17 42), not #141's "three is the
  defensible floor". Because: three points is not a correlation — one of them
  decides it — and these are exactly the seeds #134 used, so the roster result
  and the `nvs01` result are one protocol rather than two. Cost: ~6.7h serial.

- **The decision rule**: the issue asks for "a verdict" but states no criterion.
  Resolved as a pre-registered rule in `first_feasible_report.py` (median
  r >= 0.7 over eligible instances AND a strict majority at or above it; fewer
  than 10 eligible instances is INCONCLUSIVE; an instance needs >= 4 usable seeds
  and actual across-seed variance). Because: a threshold chosen after seeing the
  numbers is worth nothing, and #149's own framing ("check whether the effect
  generalises **before** changing anything") is exactly that concern.

- **Instances with no across-seed spread at one end**: `r` is undefined when
  either column is constant (`statistics.correlation` raises). Resolved, after
  review, as a three-way split rather than one "ineligible" bucket. Arrival
  varied and outcome constant (`final-invariant`), or the mirror
  (`arrival-invariant`), are both EVIDENCE AGAINST the effect: one end moved and
  the other did not. They are eligible, carry no `r`, and count as
  not-determined. Only `no-spread` — nothing moved at either end — is dropped.
  Because: filing a refutation with the uninformative instances biases the
  verdict toward GENERALISES, which is the direction #149 exists to guard; and
  an end-to-end smoke on four instances put three of them in
  `arrival-invariant`, so the bucket is not hypothetical.

- **Which statistic decides `determined`**: Pearson is what #134 used, but this
  roster's common shape is several seeds tied at the published optimum and one
  far out, where Pearson is carried by the single outlier. Resolved as: an
  instance is determined only at `>= 0.7` on BOTH Pearson and Spearman. Because:
  it makes the Spearman column load-bearing rather than decorative, and it errs
  toward not finding an effect, which is the cheaper error here.

- **`elec25`/`elec50`**: excluded, via `run_benchmark.CLAIM_EXCLUDED`. Because:
  a roster-wide correlation is a quality claim and those rows are published as
  documented failures. Counted separately in the report so the drop is visible.

- **Editing `benchmarks/instances/minlplib/README.md`**: the brief says
  `git diff <merge-base> HEAD -- benchmarks/instances/` must be empty, and also
  says to write the campaign command sequence into the benchmark's documentation
  — which lives at exactly that path. Resolved as: edit the README, change no
  `.csv` under `benchmarks/instances/`. The diff over that directory is therefore
  README-only; verify with
  `git diff 41a90fc HEAD --stat -- benchmarks/instances/` (`41a90fc` is this
  branch's base — it was cut from `fix/153-minlplib-exit-code`, not from main).
  Flagged in the report.

- **The published table's header now disagrees with the schema**: it already did
  (it predates `search_config`, `lns_repairs`, `lns_repairs_accepted`). Resolved
  as: do NOT regenerate; extend the existing
  `test_the_committed_table_uses_the_columns_the_driver_assembles` to name the
  two new columns, and state in the README how a reader dates a table from its
  header. Because: regenerating is a 50-minute measurement this run may not make.

- **A scratch `--out` with a defaulted `--trace-out`**: found while writing the
  campaign command — `resolve_paths` defaults the trace to the *published*
  `anytime_trace.csv` however `--out` is set, and `publish` assembles into it, so
  a scratch per-seed run would have replaced the published anytime profile at
  exit 0. Resolved as: fixed inline in `run_benchmark.usage_error` with two
  tests, rather than filed. Because: it is directly in the path of the command
  sequence this issue asks me to document, and brief rule 8 prefers an inline fix.

- **Python bindings**: not extended. Because: `SearchResult`'s other counters
  (`escape_probe_armed`, `lns_repairs`, ...) are not bound either, so binding
  these two alone would be an inconsistency, and no Python consumer needs them.
