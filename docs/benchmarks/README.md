# Benchmarks

Layout, priority, evidence position and workflow for every benchmark. CLAUDE.md's `## Benchmarks` keeps the priority table and the hard rules; per-benchmark detail is in each `benchmarks/instances/<name>/README.md` (and `docs/benchmarks/`), shared runner and driver code in `benchmarks/common/README.md`.

## Layout

Each benchmark follows the same pattern:
- `data.h` — C++ data structures + constexpr data arrays
- `*_model.h` — model builder function
- `*.cpp` — runner executable
- `reference_solve.py` — SCIP/PySCIPOpt baseline
- `verify_*.h` — solution correctness checker

Not every benchmark carries every file: `minlplib` and `mipfeas` read instances
from disk rather than from a `data.h`, and no benchmark ships a custom
`InnerSolverHook`. `benchmarks/chped/` is header-only — a model pattern consumed
by `examples/chped.cpp` and `tests/test_chped.cpp`, with no runner of its own.

## Benchmark priority

Three benchmarks carry the comparative story, in this order. Work the list top
down; do not start lower-priority benchmark work while a higher one is open.

**The criterion for carrying a comparative quality claim**: the model contains
terms CP-SAT cannot express at all — transcendental or non-convex over
continuous variables. A benchmark whose faithful formulation is a MILP cannot
carry such a claim however interesting the application is, because CP-SAT, CPLEX
and Gurobi own that regime and epic #87 already rejects competing there.

Row 1 is deliberately exempt, on a different rationale: `mipfeas` is pure MILP
and CP-SAT expresses every instance natively. It earns its place as an
*implementation* check — same algorithm, two implementations, identical
formulation — not as a claim that this engine is better there. Read the two
admission grounds as separate: expressiveness (rows 2-3) or same-algorithm
head-to-head (row 1). Nothing else qualifies.

| # | Benchmark | Compared against | Purpose | Tracking |
|---|-----------|------------------|---------|----------|
| 1 | `mipfeas` | OR-Tools CP-SAT `num_violation_ls` worker **only** | head-to-head correctness *and* performance against the reference implementation of the same jump-based algorithm | #87, #106 |
| 2 | `minlplib` | SCIP (nonlinear) | performance on non-convex MINLP — the regime CP-SAT's LS worker cannot express | #87 |
| 3 | `uc-chped` | Pedroso 2014 | a good result on a published unit-commitment formulation | #25, #92, #131 |

**Priority 1 is deliberately not a MIP shootout.** Compare only against CP-SAT's
`num_violation_ls` worker (same algorithm, different implementation). Do **not**
compare against CP-SAT's default full portfolio, Xpress, Gurobi or CPLEX — see
epic #87 for why that framing is rejected.

Nothing outside this table is a comparative benchmark. Beside it sit **one
worked example per structured variable type** (user decision, 2026-10-03). Their
job is to exercise and measure that type on real instances, not to carry a
comparative claim, so the table's ordering rule does not gate them:

- **`Set` → `setcover`.** Ten small OR-Library set-covering instances, run under
  both a `Set` and a Bool encoding, whose result is a documented limitation
  (see `benchmarks/instances/setcover/README.md`). Keep it as the `Set` example;
  don't grow it into an epic.
- **`List` → CVRP** (CVRPLIB X set, Hexaly's List encoding, scored as gap to
  the pinned best-known values, with PyVRP at equal budget and threads as
  context). Tracked under epic #182 (#196-#204). **In the tree, unpublished**:
  `cbls_cvrp`, its driver, verifier, scorer and PyVRP baseline landed in
  #196-#200, but until #204 publishes results `List` still has no evidence (next
  section), and the single-seed smoke numbers quoted in #201-#203 are not
  results.

Engine-wide (cross-cutting) work is tracked under epic #24.

## Structured variables: the claim is currently unsupported

Do not write, in docs or issues or commit messages, that List variables are
validated on a published formulation. The evidence position is:

- **`Set`** — measured and **negative**. The `Set` encoding lands at
  3.4-5.1x the proven optimum where the same data in Bools is within 4-9%
  (engine `35ec07c`, after #201; it was 8.6-11.0x against 9-20% at `adc8ee4`).
  `35ec07c` predates two defaults that act on the Bool encoding, compound moves
  (on since 2026-10-07) and the FJ objective tie-break (on since #226, and
  setcover's Bool objective is affine); the Bool figure was measured without
  either.
- **`List`** — **no evidence either way.** No `List` result is published yet;
  CVRP (#204) will be the first.

The mechanical cause of the `Set` result is generic and applies to `List`
equally, and it has **two independent halves** — fixing either alone changes
nothing:

1. `FeasibilityJump::jumpable` (`src/feasibility_jump.cpp`) is a deliberate
   Bool/Int/Float whitelist, so no structured variable ever gets a jump-table
   entry regardless of its derivative. Its comment says the whitelist is
   intentional.
2. `local_derivative` (`src/dag.cpp`) returns `0.0` for *every* structural op —
   `At`, `Count`, `Lambda`, `PairLambda`, `LambdaExtra`, `PairLambdaExtra` — so there is no AD signal to build a
   jump value from either. (`setcover`'s `Set` model reads through `Lambda`.)

What is left, for a `Set` or `List` searched without a neighbour list (every
model `setcover` runs), is `set_moves`/`list_moves` drawing uniformly at random
under the default `FirstImprovingSample` hill climber, where the scalar path has
FJ's jump table and best-of-N scan-set sampling to steer it. Note the contrast
is between the two *paths*, not the individual generators — `int_rand` and
`float_perturb` are themselves uniform.

Two things have since changed for a `ListPartition`. With a neighbour list
(`SearchConfig::structural_neighbours`) the default `Auto` selection runs
`GranularBest` (#203): one element's granular complete neighbourhood, best
improving move committed. And whenever LNS is on (opt-in in `solve()` and the
CLI's `--lns`, on by default in the cvrp, mipfeas, minlplib and uc-chped
runners) it ruins and recreates every `ListPartition` with SISR, neighbour list
or not (#202). Both change how a `List` is searched; neither is evidence, and
the claim rule above stands until #204. (A `Set`, or a `List` outside any
partition, given a neighbour list is sampled as under `ViolationGuided`.)

**Cost-aware structural move selection was the stated prerequisite** for any
renewed structured-variable claim. #165 built it — registrable `MoveGenerator`s,
granular neighbour lists, and `BestOfSample` / `ViolationGuided` selection
policies — and `setcover` was its ready-made A/B harness.

**It has now been measured twice, and it is still not established.** The
confounder is gone: #164 landed the position-based move representation, so a
structured candidate carries positional `ElementEdit`s and scoring one costs an
allocation-free `assign` per variable its predecessor changed, where it used to
cost at least two heap allocations and three O(|elements|) copies. (Not "no
allocation at all": an `EditKind::Replace` allocates by construction, and the
per-call membership scans still do.)

Measured under that representation, at engine commit `8dc906b`, **20 seeds per
arm** at a 10s budget, paired per seed: on the weighted instances
`ViolationGuided` is **−76.2** against `FirstImprovingSample` on a base of ~2900
(−2.6%), 95% CI **[−161.4, +9.0]** — 1.75 SE from zero, crossing it, 60/100
paired wins. Unicost is −0.18 on ~6.7 (2.07 SE, 34/100 with 45 ties). Table and
caveats in `benchmarks/instances/setcover/README.md`.

**The methodological point is worth more than the number.** A five-seed run of
the same comparison put the weighted difference at −131.9 (−4.5%); quadrupling
the data moved it to −76.2 with the interval still spanning zero. Regression
toward the mean, on a roster where per-seed spread (±8-11%) exceeds the effect
being looked for. So: do not read a five-seed structural A/B on this roster as a
result in either direction, and do not pick the one-sided framing because it
clears 0.05 when the two-sided one does not.

So the prerequisite is **implemented, measured, and not demonstrably better**,
and for `Set` the guidance stands: **don't build another `Set` benchmark.**
(`List`'s example is CVRP; see **Benchmark priority**.) What would settle the
`Set` reading, in order: a longer per-run budget (attacks the noise at source),
then more seeds (~740 paired runs for a 3-SE reading), then a harder roster. None of that is worth doing without a reason to care about
`Set`-encoded set covering specifically.

Note also that #164 generalised `List` itself — a variable-length ordered subset
of a universe, with an optional partition across several Lists whose cover the
moves maintain. That is **new capability, not new evidence**: until CVRP
publishes (#204) there is no measured `List` result, the evidence position above
stands, and nothing in the tree may claim otherwise.

**Epic #182 (compact route-network models) is where List work lives, and it has
resumed with CVRP as the `List` example** (2026-10-03). Its first sub-issues —
the CVRP benchmark #183, guided List search #184, expensive-invariant screening
#185, repeated List elements #187 and O(edits) lambda deltas #189 — were closed
as not planned in the 2026-09-30 triage, because each was speculative with no
benchmark to measure it against. That rule still holds: the CVRP benchmark comes
first (#196, #198, #197, #199), and List engine work is filed against measured
CVRP data, not ahead of it. #201-#203 cite smoke runs and qualify; #201 is a
prerequisite of the campaign (#204). Don't reopen the closed ones.

## Benchmark workflow

Each benchmark session must follow these steps in order:

0. **Plan first** — Read the benchmark's epic (linked above) and its open sub-issues, plus `docs/architecture.md` (solver internals). Investigate what already exists for your benchmark (data, reference solver, model code). Propose an approach and wait for approval before implementing. This is the Plan phase — do not skip it.

1. **Download/prepare instance data** — Write a download/generation script into `benchmarks/instances/{name}/`. Follow the pattern of `benchmarks/instances/uc-chped/` (Python data definitions + JSONL serialization). Source data from public benchmark libraries, competition archives, or papers.

2. **Find a reference solver** — Implement a reference solver in `benchmarks/{name}/reference_solve.py` using SCIP (PySCIPOpt) or another open-source solver if SCIP can't handle the formulation. Follow the pattern in `benchmarks/chped/reference_solve.py`.

3. **Collect best-known results** — Find published results from papers/competitions. Record them per `docs/measuring.md`: a README with the engine commit named in the text, or a `comparison.csv` (columns: instance, method, objective, gap, source) if a test will read it.

4. **Implement CBLS model** — Create `benchmarks/{name}/data.h` (C++ data structs + loaders) and `benchmarks/{name}/{name}_model.h` (model builder). Follow the pattern of `benchmarks/chped/chped_model.h`. **Critical rules:**
   - Implement features generically in `include/cbls/` and `src/` — not benchmark-specific hacks
   - If the solver needs new ops, moves, or hooks — implement them in the core library so all benchmarks benefit
   - Add a runner executable in `benchmarks/{name}/{name}.cpp`
   - Add Catch2 tests in `tests/test_{name}.cpp`
   - Update the root `CMakeLists.txt` and `tests/CMakeLists.txt` for new executables and tests

5. **Run comparison** — Compare CBLS vs reference solver vs best-known results. Report objective, gap %, and solve time.

6. **Verify correctness** — Check CBLS solutions against best-known solutions. Feasibility must be verified (all constraints satisfied). Objective should be within reasonable gap of BKS.

7. **Commit often** — Commit after each meaningful step. Use descriptive commit messages.

8. **Self-review loop** — After each commit, review your own changes for issues/nits. Fix and commit again. Repeat until clean.

9. **Do not interrupt the user** — No exceptions. Keep going until the benchmark is fully implemented, running, and producing correct results. Only stop if you hit a fundamental blocker that requires API/architecture changes.

## Retiring a benchmark

If a benchmark is ever retired, it is a tracker job as well as a tree job:

- Grep **every** open issue body for the epic number *and* the benchmark slug — GitHub's sub-issue list is not the full set of issues that point at an epic, because many are body-linked with "Part of #N" instead. Grep for hard-coded test counts too; those go stale the same way.
- Close the epic's sub-issues **explicitly** and as **not planned** — a closing keyword in a commit message closes the epic only, and the work was abandoned, not delivered. `gh issue close --reason` is silently a no-op on an already-closed issue; correct one after the fact with
  `gh api --method PATCH /repos/:owner/:repo/issues/<num> -f state=closed -f state_reason=not_planned`.
- The epic itself closes as **completed** — retiring it was the job.
- An issue that is still valid and merely *cites* the retired code (a hook name, a file path, a list of affected benchmarks) needs its body **edited** to name a surviving example instead. Closing it would drop live work.
