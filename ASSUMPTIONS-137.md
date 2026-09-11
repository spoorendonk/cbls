# Assumptions — issue #137

- **Where the pinned reference hashes live**: the issue asks for the solution file
  and the roster table to be "pinned by hash", without saying where. Resolved as:
  a committed `benchmarks/instances/mipfeas/references.csv` (`file,sha256,bytes`),
  mirroring `manifest.csv`'s shape. Because: a CSV pin table is a reviewable diff
  in the same style the directory already uses, and putting the hashes in Python
  constants would make an update a code change rather than a data change.
- **What `--verify` does about absent instances**: an absent instance could be
  read as a failure ("the roster is incomplete") or as a legitimate state.
  Resolved as: absent instances are counted and printed but do not fail `--verify`.
  Because: a checkout that fetched only the 11-instance smoke subset is normal,
  and the run driver already refuses a roster with missing instances before it
  starts — so making `--verify` fail on absence would only duplicate that check
  while making the common case noisy.
- **Two update flags rather than one**: resolved as `--update-references` (the
  yardstick: solution file + roster tables) and `--update-manifest` (instance
  bytes). Because: they have different blast radii — a moved reference invalidates
  every gap in the table at once, a moved instance invalidates one row — and a
  single flag would let an accepted instance re-fetch silently carry a yardstick
  revision with it.
- **Which of the two a missing log line is**: the issue asks the preflight to name
  "which of the two broke", but a subsolver announcement that is simply absent
  could be either. Resolved as a stated rule: the *shape* of a line the harness
  parses is the log format, its *content* is the worker restriction. So a missing
  announcement is a log-format failure and an announcement listing `default_lp` is
  a restriction failure. Because: any other split would have the preflight guess,
  and a guess in the failure message is worse than a rule a reader can check.
- **Where the preflight runs**: per-solve or once per run. Resolved as
  `cpsat_solve.py --preflight` (a standalone mode), invoked once by the driver
  before it dispatches any job. Because: the check costs ~2.3s, and 233 instances
  x 2 engines would pay it 233 times for one answer — while the driver is the only
  place that can refuse to start the run.
- **ortools upper bound**: `<9.16`, i.e. the next minor after the 9.15 every
  parameter and log fact was established against. Because: OR-Tools ships
  subsolver and log changes in minor releases, and the preflight (not the bound)
  is what allows raising it after a check.
- **How strict the driver's reference-pin check is**: `--inst-dir` may legitimately
  point at a bare instance directory (the vendored `miplib-fj` set) that has no
  roster tables to pin. Resolved as: the instance manifest is always required, and
  the reference pins are required only when the directory actually holds one of
  the three reference files. Because: demanding `references.csv` of every
  directory would make the documented ad-hoc sweeps impossible, while a directory
  holding `roster.csv` and nothing pinning it is exactly the unpinned-yardstick
  state this issue exists to end.
- **What counts as "never checked"** (orchestrator finding 1): resolved as any
  feasible row whose verdict is neither `pass` nor `fail` — an exhausted driver
  retry, a verifier error, or no verdict file at all. Because: all three mean the
  scorer withholds the row, and `fail` is excluded precisely so the loudest signal
  the benchmark has is not buried under a harness counter.
- **Where the CP-SAT "did not search" messages live** (orchestrator finding 2):
  resolved as a pure `status_note()` helper rather than inline in `solve()`.
  Because: the two verdicts it classifies cannot be produced on demand in a test,
  and a pure function is the only way to pin the message a published row will
  carry without provoking the solver into an invalid state.
