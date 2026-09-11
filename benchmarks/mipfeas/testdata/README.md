# MIPfeas parity fixture

A frozen MIPfeas results directory and the two artifacts the scorer produces from
it. `tests/python/test_mipfeas_parity.py` regenerates both and compares them byte
for byte, which is what pins every number the parity report publishes (issue
#139): the defect counters, both asymmetric feasibility difference sets, the
model-shape verdicts, the trace-health counts and the timing split.

```
roster.csv                13 instances with reference values
results/<engine>/         result, incumbent trace and independent verdict per job
expected_comparison.csv   what primal_integral.py writes from results/
expected_report.md        the parity report it writes beside that table
```

Regenerate both after any deliberate change to the report — the command the test
runs, and the one whose output is committed here:

```bash
python benchmarks/mipfeas/primal_integral.py \
    --results-dir benchmarks/mipfeas/testdata/results \
    --roster benchmarks/mipfeas/testdata/roster.csv --budget 2 \
    --out benchmarks/mipfeas/testdata/expected_comparison.csv \
    --report benchmarks/mipfeas/testdata/expected_report.md
```

It exits 1, because the fixture deliberately contains a rejected solution.

## What is in it, and how real it is

**`mad` and `gen-ip054` are a real run**: both engines at a 2-second budget over
the vendored `benchmarks/instances/miplib-fj/` copies of those two instances,
each solution checked by `verify_solution.py` against the instance file. `mad`
carries the constraint-count disagreement this benchmark actually has — CBLS and
SCIP read 51 rows, OR-Tools' ModelBuilder 52 — and is the case the free-row rule
in the report exists to explain.

**The other eleven instances are constructed**, because a clean run pins nothing.
Each exists to make one branch of the report produce output, and a fixture is the
only way to hold a defect still: `only-cbls` and `only-cpsat` are the two
difference sets, `neither` and the five clean rows the agreement count,
`killed-job` a driver kill with its message, `rejected-solution` a solution the
checker refused (and one that beats the proven optimum, which is what such a row
usually looks like), `unverified-row` a feasible row nobody checked,
`invalid-for-cpsat` a model the baseline rejected, `missing-job` a job with no
result file, `shape-mismatch` a constraint-count difference the free-row rule
does **not** account for, `degraded-trace` an anytime profile that collapsed to a
single end point, and `slow-start` the two timing effects at once — a long model
build and a solve that ran past the budget because search initialisation is not
bounded by the deadline.

These are fixture numbers. Nothing here is a measurement of either engine, and
nothing here may be quoted as one: the roster is 13 instances at 2 seconds where
the benchmark is 233 at 600, and eleven of the thirteen never ran at all.
