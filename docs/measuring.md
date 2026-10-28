# Measuring and testing engine changes

The full rules; CLAUDE.md's `## Measuring and testing engine changes` is the short list. Profiling, heap attribution and sanitizer recipes are in `docs/profiling.md`.

## Performance and measurement

**Performance matters.** Most of our code is solvers — don't sacrifice perf
for "clean code". `docs/profiling.md` is how to measure: heap attribution, CPU
profiling and the sanitizer build, with a dated tool-availability table naming
the machine it was checked on. Numbers reached by subtracting two whole-program
timings are not measurements.

**Profile before MICRO-optimizing; a complexity win does not need permission.**
The two are different claims. Replacing an O(n²) dedup with an O(1) stamp, or
an O(|nodes|) scan with an O(d log d) sort, rests on a cost argument that holds
independently of any machine — make it in a comment at the site, stating the
regime it wins in *and* the regime it loses in, and land it. Guessing that a
hand-unrolled loop or a reordered branch is faster is the thing that needs a
profile first, because nothing but the profile decides it.

Measurement is still required for three things, and a plausible argument is
never a substitute: any number that goes into a doc, a README or a published
table; any choice of a *parameter*, which is never fitted on the instances it
is reported on (fit on one roster, report on a held-out one, as
`benchmarks/instances/minlplib/HELDOUT.md` does); and any claim that a change made something faster,
which means an A/B at one budget on one idle machine, not two runs at different
budgets compared to each other.

## Engine changes

**A regression test must be shown to fail before the fix.** A plausible-looking
test frequently passes on the unfixed code — LNS maps a NaN constraint to `+inf`
and rolls back, for instance, so a NaN-poisoned repair can never win and a test
written against that path is green either way. Verify by reverting *only* the
production change in a throwaway copy (`git archive HEAD | tar -x -C /tmp/...`,
then restore the changed files from `main`) and confirming the new test goes red.
A test that cannot fail is not a regression test.

**Benchmark comparisons are time-limited, so never run them concurrently.**
Objective quality at a fixed wall-clock budget depends on how many iterations the
process gets, so two runs sharing cores produce numbers that are not comparable
to each other *or* to the committed tables. Run A/B comparisons serially, check
`uptime` first, and re-run anything anomalous. A bare `cmake -B build` now
configures Release, but an existing `build/` keeps whatever type it was first
configured with — and the same goes for `CBLS_SANITIZE`/`CBLS_PROFILE`, which are
cache entries too: a `build/` once configured with a sanitizer stays that way
through every later flag-less `cmake -B build`. So
for a timing comparison either check `CMAKE_BUILD_TYPE` and `CBLS_SANITIZE` in
`build/CMakeCache.txt` or configure a fresh directory. The measurement recipes —
heap attribution, CPU profiling, sanitizers — are in `docs/profiling.md`.

**Read sub-second runs on CPU time, not wall time.** #188's MINLPLib timing
check saw short runs land on a ~0.1 s wall-clock grid, attributed on one
reproduction to the `timeout` wrapper. The grid turns small differences into
whole steps (gear4 read as 1.50x) (`docs/benchmarks/incremental-sum-drift.md`).
Take CPU time (`/usr/bin/time`) alongside wall, or drop the wrapper for short
runs. And a per-iteration regression with unchanged work can be code shape, not
cost: #188's walk lost 14% on eg_all_s with the same per-walk cycle counts, and
got it back by keeping cold paths out of line with `[[gnu::noinline]]`.

**A runner that writes a published table must not truncate it before it has
results.** `std::ofstream out(path)` truncates on open, so a run started from
the wrong directory replaces the table with a header and nothing else, at exit
code 0 — this emptied the miplib-fj table once already. Load the roster first
and refuse to write on an incomplete one, and write to a temp path then
`std::filesystem::rename`, as `benchmarks/mipfeas/` and
`benchmarks/uc-chped/` do. The rename also covers the job killed mid-write.
Rows regenerated from the instance files — uc-chped's cited reference rows are
— make this sharper, not softer: the rows most worth protecting are exactly the
ones a missing instance file deletes.

**When regenerating a `comparison.csv`, record the engine commit in it.**
Search-trajectory changes silently invalidate published tables, and without the
commit the next reader cannot tell drift from a bug. `minlplib`'s table carries a
per-row `commit_sha` column; a header comment naming the commit does the job too.

**Nothing checks this outside MINLPLib.** Only MINLPLib's tables are read by
tests: `tests/python/test_minlplib_scip_baseline.py`, and
`tests/python/test_minlplib_campaign_report.py`, which pins the README's
generator-owned blocks and `campaign_summary.json` to the committed tables
(#142). Every other committed table drifts silently and is caught only when
someone re-measures. Prefer stating
results in a benchmark README with the engine commit named in the text — the
shape `benchmarks/instances/setcover/README.md` uses — unless a test is actually
going to read the table.

**A comparison table is an OUTPUT; the reference values are the artifact.**
Distinguish the two, because only one of them is worth freezing:

- **Best-known and optimal values** are derived ONCE, from the paper or the
  benchmark library that publishes them, and pinned — `mipfeas` keeps them in
  `roster.csv` with `references.csv` recording the hash and byte count of the
  MIPLIB `.solu` file they came from, so a library revision cannot silently
  redefine the yardstick. This is the thing to protect, and to re-derive only
  when the upstream source itself changes.
- **An engine-vs-engine comparison** is re-derived from result records whenever
  the question is asked. It is a function of a results directory, a roster and a
  budget, all of which are inputs the run already keeps — so checking one in as
  a record buys nothing but a file that goes stale the next time the engine
  moves. Score it into the results directory, quote it with the engine commit
  and the machine record beside it, and regenerate rather than defend it.

`mipfeas` follows this; the older tables predate it. Don't add machinery to
protect a comparison table from being overwritten — the protection worth having
is on the reference values and on the per-instance result records the table is
computed from.
