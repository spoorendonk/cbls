# Benchmarks

Layout, priority, evidence position and workflow for every benchmark. CLAUDE.md's `## Benchmarks` keeps the priority table and the hard rules; per-benchmark detail is in each `benchmarks/instances/<name>/README.md` (and `docs/benchmarks/`), shared runner and driver code in `benchmarks/common/README.md`.

## Layout

Each benchmark follows the same pattern:
- `data.h` — C++ data structures + constexpr data arrays
- `*_model.h` — model builder function
- `*.cpp` — runner executable
- a reference solver script, per benchmark:
  - `minlplib`: `reference_solve.py` (SCIP/PySCIPOpt)
  - `uc-chped`: `benchmarks/chped/reference_solve.py --uc` (SCIP/PySCIPOpt)
  - `mipfeas`: `cpsat_solve.py` (CP-SAT)
  - `cvrp`: `reference_solve.py` (PyVRP) and `ortools_solve.py` (OR-Tools)
  - `setcover`: none; its optima are proven and pinned
- `verify_*.h` — solution correctness checker

Not every benchmark carries every file: `minlplib` and `mipfeas` read instances
from disk rather than from a `data.h`, and no benchmark ships a custom
`InnerSolverHook`. `benchmarks/chped/` has no runner of its own: its model
pattern is consumed by `examples/chped.cpp` and `tests/test_chped.cpp`, and its
`reference_solve.py` is UC-CHPED's live SCIP reference.

## Benchmark priority

Three benchmarks carry the comparative story, in this order. Work the list top
down; do not start lower-priority benchmark work while a higher one is open.

**The criterion for carrying a comparative quality claim**: the model contains
terms CP-SAT cannot express at all — transcendental or non-convex over
continuous variables. A benchmark whose faithful formulation is a MILP cannot
carry such a claim however interesting the application is, because CP-SAT, CPLEX
and Gurobi own that regime and epic #87 already rejects competing there.

Row 1 is admitted on a different ground: `mipfeas` is pure MILP and CP-SAT
expresses every instance natively. It earns its place because CP-SAT carries
the same jump-based algorithm (its `fj` and `ls` workers) on an identical
formulation, not as an expressiveness claim. Read the two admission grounds
as separate: expressiveness (rows 2-3) or the same algorithm in the baseline
(row 1). Nothing else qualifies.

| # | Benchmark | Compared against | Purpose | Tracking |
|---|-----------|------------------|---------|----------|
| 1 | `mipfeas` | OR-Tools CP-SAT at its default parameters | correctness *and* performance against CP-SAT as it ships, which holds the reference implementation of the same jump-based algorithm | #87; campaign #255 is closed and its table waits on #253; #321 adds a CP-SAT `fj` + `ls` arm |
| 2 | `minlplib` | SCIP (nonlinear) | performance on non-convex MINLP — the regime CP-SAT's LS worker cannot express | #87, campaign #256, scored by #306 |
| 3 | `uc-chped` | Pedroso 2014 | a good result on a published unit-commitment formulation | #87, campaign #257 |

#253 publishes the full-benchmark tables of all four sets: these three and CVRP
(#204).

**Every comparison solver runs at its own default parameters** (Simon,
2026-11-01, [#87](https://github.com/spoorendonk/cbls/issues/87#issuecomment-6558356494)):
CP-SAT and SCIP with no setting changed, and HiGHS likewise should a benchmark
ever run it (none does today). CP-SAT runs as it ships — presolve on, its full
default portfolio — not restricted to its `fj` + `ls` workers under
`num_violation_ls`, as row 1 was until #284 (#106). The comparison reports what each solver does and does not try to even out their
differences; the MIPfeas table records each engine's version and the threads,
CPU and wall time of every solve (#285). Xpress, Gurobi and CPLEX stay out of
every table (#87).

Nothing outside this table is a comparative benchmark. Beside it sit **one
worked example per structured variable type** (user decision, 2026-10-03). Their
job is to exercise and measure that type on real instances, not to carry a
comparative claim, so the table's ordering rule does not gate them:

- **`Set` → `setcover`.** Ten small OR-Library set-covering instances, run under
  both a `Set` and a Bool encoding, whose result is a documented limitation
  (see `benchmarks/instances/setcover/README.md`). Keep it as the `Set` example;
  don't grow it into an epic.
- **`List` → CVRP** (CVRPLIB X set, Hexaly's List encoding, scored as gap to
  the pinned best-known values, with PyVRP and OR-Tools' routing library at
  equal budget and one thread as context, and an 8-thread `cbls` row set beside
  Hexaly's 8-core figures). Built under epic #182, now closed; its
  campaign, #204, moved to #87.
  **In the tree, unpublished**: `cbls_cvrp`, its driver, verifier, scorer and
  PyVRP baseline landed in #196-#200, the OR-Tools baseline in #204, but until #204 publishes results `List` still has no evidence (next
  section), and the single-seed smoke numbers quoted in #201-#203 are not
  results.

## Structured variables: the claim is currently unsupported

Do not write, in docs or issues or commit messages, that List variables are
validated on a published formulation. The evidence position is:

- **`Set`** — measured and **negative**. The `Set` encoding lands at
  3.4-5.1x the proven optimum where the same data in Bools is within 4-9%
  (engine `35ec07c`; history in the setcover README). `35ec07c` predates two defaults that act on the Bool encoding, compound moves
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

**Cost-aware structural move selection** (#165: registrable `MoveGenerator`s,
granular neighbour lists, `BestOfSample` / `ViolationGuided`), the stated
prerequisite for any renewed structured-variable claim, is **implemented,
measured, and not demonstrably better**: on `setcover`, under #164's
position-based move representation, `ViolationGuided` against
`FirstImprovingSample` has a 95% CI spanning zero at 20 seeds per arm (table,
caveats and why a five-seed run misled in `benchmarks/instances/setcover/README.md`).
Do not read a five-seed structural A/B on that roster as a result in either
direction. For `Set` the guidance stands: **don't build another `Set`
benchmark.** (`List`'s example is CVRP; see **Benchmark priority**.) #164's
generalised `List` is **new capability, not new evidence**: until CVRP
publishes (#204) nothing in the tree may claim a measured `List` result.

**Epic #182 (compact route-network models) is closed as not planned**, and its
CVRP benchmark moved to #87 as #204. List engine work is filed against measured
CVRP data, not ahead of it; #201-#203 cite smoke runs and qualify. Don't reopen
#183-#185, #187 or #189.

## Benchmark workflow

Adding a benchmark follows CLAUDE.md's **Working Through Issues** for the issue
loop, review and pushing. The steps specific to a benchmark:

0. **Plan first.** Read the benchmark's issue and `docs/architecture.md` (solver
   internals), and check what already exists (data, reference solver, model
   code). Put the plan on the issue and wait for approval before implementing.

1. **Prepare instance data.** Write a download or generation script into
   `benchmarks/instances/{name}/`, from public benchmark libraries, competition
   archives or papers. `benchmarks/instances/setcover/download.py` is the pattern
   for a download; `benchmarks/instances/uc-chped/gen_jsonl.py` for generated
   data.

2. **Find a reference solver.** Implement it in `benchmarks/{name}/`, using SCIP
   (PySCIPOpt), or another open-source solver if SCIP can't handle the
   formulation. Follow `benchmarks/chped/reference_solve.py`.

3. **Collect best-known results** from papers and competitions, and pin them per
   `docs/measuring.md`.

4. **Implement the CBLS model.** `benchmarks/{name}/data.h` (structs and loaders),
   `benchmarks/{name}/{name}_model.h` (model builder, following
   `benchmarks/chped/chped_model.h`), a runner `benchmarks/{name}/{name}.cpp` and
   Catch2 tests `tests/test_{name}.cpp`, all registered per CLAUDE.md's **CMake**.
   New ops, moves or hooks go in `include/cbls/` and `src/` so every benchmark
   benefits, never as benchmark-specific hacks.

5. **Run the comparison** of CBLS against the reference solver and the best-known
   results, per `docs/measuring.md`: objective, gap % and solve time.

6. **Verify correctness** against the instance: every constraint satisfied,
   recomputed independently of the engine, and the objective compared with the
   best-known value.

## Experiment records

Engine experiments, not benchmarks, kept beside them:

- `incremental-sum-drift.md`: bounded-drift incremental `Sum`, pre-registration
  and record (#188).
- `linear-jump-pingpong.md` and `linear-jump-pingpong.patch`: whether the
  closed-form jump scorer ping-pongs on Float plateaus (#178).
- the `xk tune` study files (#261-#266), formerly `benchmarks/tune/`: in git
  history at `047f2d4c`.

## Retiring a benchmark

If a benchmark is ever retired, it is a tracker job as well as a tree job:

- Grep **every** open issue body for the epic number *and* the benchmark slug — GitHub's sub-issue list is not the full set of issues that point at an epic, because many are body-linked with "Part of #N" instead. Grep for hard-coded test counts too; those go stale the same way.
- Close the epic's sub-issues **explicitly** and as **not planned** — a closing keyword in a commit message closes the epic only, and the work was abandoned, not delivered. `gh issue close --reason` is silently a no-op on an already-closed issue; correct one after the fact with
  `gh api --method PATCH /repos/:owner/:repo/issues/<num> -f state=closed -f state_reason=not_planned`.
- The epic itself closes as **completed** — retiring it was the job.
- An issue that is still valid and merely *cites* the retired code (a hook name, a file path, a list of affected benchmarks) needs its body **edited** to name a surviving example instead. Closing it would drop live work.
