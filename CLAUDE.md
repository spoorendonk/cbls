# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Standards

The rules below are the project's own. Where a generic house style would say
otherwise, what is written here wins.

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions. `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol` answer "where is X / what's its signature" in a few tokens. Use before `Read`.
2. **Grep with `head_limit` (small) + `-n`** to locate lines. Start with `head_limit: 20`; raise only if inconclusive.
3. **Read with `offset`/`limit`** to fetch a slice around the hit. Full-file `Read` is fine for files under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile. This is a preference, not a prohibition: shelling out to `grep`/`rg` is fine when the built-in can't do the job (filtering a pipe like `git log | grep`, or a session without the `Grep` tool). What matters is bounding output, not which binary produces it.

## C++

- **C++17** (`CMAKE_CXX_STANDARD 17`) — not C++23. `std::expected`, concepts and ranges are *not* available here; don't reach for them.
- Style: Google-based, enforced by `.clang-format` and `.clang-tidy` at the repo root. Both tools are **pinned PyPI wheels in `.venv/`** (`clang-format==22.1.8`, `clang-tidy==22.1.8`), not system packages — `.venv/bin/pip install -e '.[dev]'` installs them, and the hooks prefer `.venv/bin` over `PATH`. Formatter output changes between major releases, so the pin is what keeps one person's commit from reformatting another's.
- Naming is **not** Google's default — `.clang-tidy` encodes what this codebase actually does:

  | Kind | Style | Example |
  |---|---|---|
  | Functions (free and member) | `lower_case` | `full_evaluate`, `weighted_violation_delta` |
  | Locals, parameters, `const` locals | `lower_case` | `total_violation`, `node_id` |
  | Private members | trailing `_` | `out_`, `model_` |
  | File-scope / static / class / `constexpr` constants | `k` + `CamelCase` | `kVersion`, `kEps`, `kMpsInf` |
  | Classes | `CamelCase` | `Model`, `JumpTable` |
  | Namespaces | `lower_case` | `cbls` |

  Two documented exceptions: `Model::Bool/Int/Float/List/Set/Constant` stay CamelCase because they name the type they create (and `bool`/`int`/`float` cannot be lowercased without hitting a keyword), and `src/io/` disables the naming check entirely because its two readers are vendored from `spoorendonk/mipx` and must stay diff-clean against upstream.
- Use `#pragma once` for include guards.
- Minimize includes in headers. Forward-declare where possible.

### LSP

Install `clangd-lsp@claude-plugins-official` plus `clangd` itself (`apt install clangd` or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`. Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`) over `Read` for symbol questions.

## Python

- Style: enforced by `ruff` (format + lint) and `mypy --strict`, configured in `pyproject.toml`.
- All functions must have full type annotations (mypy strict mode).
- Use built-in generics (`list[int]`, `dict[str, Any]`) and `|` union syntax.
- Pin dependencies with `>=` lower bounds in `pyproject.toml`. Use `uv` or `pip`.

### LSP

Install `pyright-lsp@claude-plugins-official`. Pyright reads `[tool.mypy]` and project layout from `pyproject.toml` — no extra config needed.

## nanobind Bindings

- Bindings live in `python/bindings.cpp` — a single `NB_MODULE(_cbls_core, ...)` built by `nanobind_add_module` in `python/CMakeLists.txt`, gated on `CBLS_BUILD_PYTHON`. Keep binding glue out of `src/`; a new binding source must be added to that call or it is never compiled.
- C++ `camelCase` methods → Python `snake_case` via nanobind. Use `nb::arg("name")` for Python-friendly parameter names.

### Ownership and Lifetime

- Default: nanobind manages ownership. Use `nb::rv_policy::reference` only when C++ retains ownership and guarantees the object outlives Python references.
- Never return raw pointers without explicit lifetime annotation.
- Prefer returning by value or `std::shared_ptr`. Document ownership on each binding that transfers or shares it.

### Type Conversions

- Use automatic conversions for standard types (`std::string` ↔ `str`, `std::vector` ↔ `list`).
- Use `nb::ndarray` for NumPy interop — specify dtype and shape constraints.

## Testing

C++ tests use **Catch2** (not GoogleTest): files in `tests/`, registered in `CMakeLists.txt`, run via `ctest`. Python tests use **pytest**: `test_<module>.py` under `tests/python/`, `conftest.py` for shared fixtures, `pytest.mark.parametrize` for data-driven cases.

- Name tests descriptively — `returns_optimal_for_feasible_input`, `test_solver_returns_optimal_for_feasible_input`.
- Test nanobind bindings from Python with pytest, not from C++ — the binding is an implementation detail. Include round-trip tests: create in Python → pass to C++ → get result back.
- Terse output is not configured anywhere — pass the flags. `ctest --progress` collapses the running list, `--tb=short -q` keeps pytest failures short, as the `## Build & Test` blocks below already do. `pyproject.toml` sets `testpaths`, ruff's rule set and `mypy strict`, but no `addopts`, `pretty` or `output-format`.

## CMake

- `set(CMAKE_EXPORT_COMPILE_COMMANDS ON)` for clang-tidy.
- Use FetchContent for dependencies.
- Targets live in the root `CMakeLists.txt` — library, CLI, examples and every benchmark runner. Only `tests/` and `python/` carry their own. Add new benchmark and example targets to the root file.

## Development Workflow

```
plan (non-trivial) → implement → test → /review → push to main
```

Run tests locally before considering work done — don't skip the suite even on changes that look trivial. The pre-push hook is the final gate for build and tests; `/review` is a discipline nothing enforces, so it is on you to actually run it.

## Git Hooks

The hooks live in **`.githooks/`, tracked in this repo** — that directory is the source, edit it there. Git runs them only once `core.hooksPath` points at that directory, and that setting is per-checkout local config which cannot be committed, so `cmake -B build` sets it (`CBLS_INSTALL_GIT_HOOKS`, default ON). Configuring the project once is all a fresh clone needs; to wire one up by hand, `git config core.hooksPath .githooks`. If `core.hooksPath` already points somewhere else, CMake warns rather than overwriting it.

`core.hooksPath` is shared across every worktree of a checkout, but `.githooks/` is resolved from the working tree — so a branch that does not carry the directory runs **no hooks at all**, silently. Merge main into any long-lived branch before relying on the gates.

- `pre-commit` — auto-formats staged C++/Python/shell (clang-format, ruff, shfmt), applies safe clang-tidy fixes, re-stages, then runs the affected test suite. Hard block on failure. The clang tools come from `.venv/bin`; if neither the venv nor `PATH` has them it says so rather than skipping quietly.
- `commit-msg` — Conventional Commits format.
- `pre-push` — the clean build + **full** suite from `## Build & Test` below, then clang-tidy/ruff-complexity/shellcheck/mypy as warnings. Both the ```build and ```test fences must resolve or the push is blocked; there is no auto-detect fallback, because guessing a build would gate a different one than the documented build.

Nothing enforces `/review` at push time, by design — a gate keyed on gitignored local tooling can only be satisfied in whichever checkout happens to carry it, and passes silently everywhere else. `/review` is still expected on every change (see **Agent Self-Review**); running it is on you.

### Fast vs. slow tests

22 of the 270 C++ tests are multi-minute benchmark solves carrying the Catch2 `[slow]` tag; they account for ~1470s of the suite's aggregate (summed per-test) time, which `-j$(nproc)` compresses to a ~304s wall-clock full run. They are registered by their own `catch_discover_tests` call in `tests/CMakeLists.txt` with `LABELS "slow"`, so:

- `ctest -LE slow` — the other 248 tests, ~7s with `-j`. This is what **pre-commit** runs.
- `ctest` — everything. This is what **pre-push** and CI run.

These counts are hard-coded in **five** places and nothing checks that they
agree: this section, the comment above `catch_discover_tests` in
`tests/CMakeLists.txt`, the build section of `README.md`, the comment above the
`ctest` call in `.githooks/pre-commit`, and — for the pytest side — the
`.venv/bin/pytest` line in `README.md` (175 tests, 73 of them binding tests,
echoed in prose by `pyproject.toml` and `tests/python/conftest.py`). Update all
of them in the same commit as any change to the test roster. Leaving one behind
is not hypothetical: removing a benchmark missed `README.md` once and
`.githooks/pre-commit` twice running, so pre-commit spent two removals claiming
33 slow tests of a ~1010s run.

Tag a new test `[slow]` if it takes more than ~10s. Don't tag one just to get a green commit — pre-push will still run it.

**Never use `git push --no-verify` or `git commit --no-verify`** unless explicitly asked. A failing hook is a signal — fix the root cause.

Gating lives in git hooks only — `.claude/settings.json` carries no `hooks` block, and none should be added. A `PostToolUse` formatter cannot see which file was edited, so it silently formats nothing; formatting belongs at commit time. Don't hand-tune formatting.

Three conventions therefore rest on you rather than on a tool: branch only from main (**Git Workflow**), never invoke `python`/`pip`/`pytest`/`mypy` outside `.venv/bin/` (**Build & Test**), and prefer the `Grep` tool to shelling out (**Code Navigation**).

`.claude/` is gitignored local agent tooling (settings, statusline, the `/review` command). Nothing in it is part of the repo.

## Git Workflow

Trunk-based development with linear history on main. Commit directly to main and push when local gates pass.

Feature branches are optional for larger changes:
- Always branch from main. Run `git checkout main && git pull` first.
- Never branch from another feature branch.
- Keep branches short-lived; rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`. Do this for every issue covered by the push.
- **Delete the feature branch** if one was used: `git branch -d <branch>` locally, plus `git push origin --delete <branch>` if it was pushed. Don't leave stale branches behind.

## Issue Tracking

GitHub Issues is the tracker. Use the `gh` CLI.

- **Default to HTTPS** for GitHub remotes (`https://github.com/...`), not SSH.
- **Read an issue** with `gh issue view <num> --json title,body,labels,state,comments`. Plain `gh issue view <num>` is deprecated for programmatic use.
- Don't propose deferring work via a new gh issue unless it is substantial. Small follow-ups should be either fixed inline or left alone — don't open an issue just because you noticed something.

### Writing Issues

Issues get picked up later in fresh sessions, often by a different agent with no access to the author's machine. Write them to be picked up cold:

- **Self-contained.** Body must carry all needed context: problem, motivation, acceptance criteria, repro steps. Don't assume the reader has the current conversation.
- **No local references.** No local file paths, local repo paths, or machine-specific locations (`/home/user/...`, `~/code/foo/bar.py`, "see my other checkout"). Dead links in a fresh session.
- **Prefer stable external links.** GitHub permalinks, paper URLs, RFCs, official docs.
- **Be vague about local code context.** Describe the concept rather than the path; hint that the agent can search under `..`, `../..`, or `~/code/`.

## Agent Self-Review

**Any agent or agent team that produces code must run `/review` on its own changes before that code can merge to main.** No subagent returns unreviewed work; no orchestrator merges unreviewed work. This applies to every agent team, not just the parallel-issue workflow.

## Parallel Issue Workflow

When the user brings multiple gh issues to work on at once:

1. **Propose parallelism first.** Offer it explicitly and wait for confirmation — don't silently start serial work.
2. **Orchestrator role.** Spawn one subagent per issue (Agent tool with `isolation: "worktree"`). Subagents branch from main, not from the orchestrator's working branch, and work in their own git worktree. Pass each subagent its gh issue number and any plan file path.
3. **Subagents self-review** per the Agent Self-Review rule above. Subagents commit locally in their worktree and **do not push** — worktrees share `.git`, so the orchestrator sees their commits via `git log <branch>` with no network round-trip.
4. **No merging without user OK.** Subagents never merge into main; the orchestrator never merges a subagent's branch without explicit user approval.
5. **Final combined review, then push.** The orchestrator merges all approved branches into local main, runs `/review` over the merged result, and only then runs `git push origin main`. No pushes — of main or feature branches — happen before that final review.

## Commit Messages

Conventional Commits. The commit-msg hook enforces format.

- `type: description` or `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`, `build`, `ci`
- Subject ≤72 chars. Focus on **why**, not what.

## CLAUDE.md Discipline

When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask: what are the independent responsibilities here? Split along those boundaries. If the function is genuinely complex because the domain is, add a comment explaining why and suppress the warning.

## Plan Adherence

**Follow the agreed plan.** If you think a plan should change, stop and discuss — don't silently diverge. The same goes outside a written plan: if your current approach isn't working, say so out loud — don't quietly switch strategies. Implement everything specified; don't leave TODO placeholders or stub implementations unless explicitly asked.

## Reference Correctness

When implementing from papers, pseudocode, or open-source references:
- Match the reference algorithm exactly. No early exits, iteration limits, size caps, or "optimization" shortcuts that change behavior.
- Only introduce heuristic approximations when explicitly asked.
- Implement edge cases and special handling — don't simplify them away.
- When in doubt, be faithful to the reference and let tests verify correctness.

## Common Mistakes

- **Don't invent APIs — verify they exist.** Check that functions, flags, and methods actually exist before using them.
- **Don't ignore type errors.** If mypy/clang-tidy flags something, fix the root cause — don't suppress.
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of our code is solvers — profile before micro-optimizing, but don't sacrifice perf for "clean code".

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python" && cmake --build build -j$(nproc)
```

```test
ctest --test-dir build --output-on-failure -j$(nproc) && (CBLS_REQUIRE_BINDINGS=1 .venv/bin/pytest --tb=short -q; rc=$?; [ $rc -eq 0 ] || [ $rc -eq 5 ])
```

**The gated build turns the Python bindings on, and the gated test run requires
them.** `CBLS_BUILD_PYTHON` defaults to `OFF`, so the fences used to build no
bindings, and `tests/python/conftest.py` then dropped every test that imports
`_cbls_core` without saying so — a clean `cmake -B build` produced `102 passed`
while 73 binding tests had not run at all, and no gate anywhere would have caught
a binding regression. Building them costs ~2.4s of build and ~6s of pytest
against a suite that already spends ~340s in `ctest`, which is not a price worth
trading coverage for. `CBLS_REQUIRE_BINDINGS=1` makes the skip a hard error, so
a build that silently produced no module fails the gate instead of passing it.

`pytest` is **only** in `.venv/` — a bare `pytest` is not on PATH and
`python3 -m pytest` has no module, so always spell out `.venv/bin/pytest`.
A git worktree has no `.venv` of its own; symlink the main checkout's in before
building or running the Python suite there. The build now needs it too — the
```build fence locates nanobind through `$PWD/.venv/bin/python`, so without the
symlink a worktree builds no bindings and the ```test fence fails on
`CBLS_REQUIRE_BINDINGS=1`:

```bash
ln -sfn /path/to/main/checkout/.venv .venv && .venv/bin/pytest --tb=short -q; rm -f .venv
```

Run a single C++ test by name (`catch_discover_tests` registers each `TEST_CASE` as its own ctest test, so `-R` matches the test-case name):
```bash
ctest --test-dir build -R "test_name_substring"
```

Run a single Python test:
```bash
.venv/bin/pytest tests/python/test_foo.py::test_specific -x
```

Python bindings (off by default):
```bash
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE=$(which python3)
cmake --build build -j$(nproc)
```

## Measuring and testing engine changes

Two disciplines this repo has been burned by repeatedly. Both cost a full re-run
when skipped.

**A regression test must be shown to fail before the fix.** Several issues here
warn that a plausible-looking test passes on the unfixed code — #112 says so
explicitly, because LNS maps a NaN constraint to `+inf` and rolls back, so a
NaN-poisoned repair can never win and a test written against that path is green
either way. Verify by reverting *only* the production change in a throwaway copy
(`git archive HEAD | tar -x -C /tmp/...`, then restore the changed files from
`main`) and confirming the new test goes red. A test that cannot fail is not a
regression test, and three separate changes in one recent batch shipped with one.

**Benchmark comparisons are time-limited, so never run them concurrently.**
Objective quality at a fixed wall-clock budget depends on how many iterations the
process gets, so two runs sharing cores produce numbers that are not comparable
to each other *or* to the committed tables. Run A/B comparisons serially, check
`uptime` first, and re-run anything anomalous. Pin the build type explicitly
(`-DCMAKE_BUILD_TYPE=Release`) — a bare `cmake -B build` is unoptimized and an
existing `build/` may be cached at a different type.

**When regenerating a `comparison.csv`, record the engine commit in its header.**
Search-trajectory changes silently invalidate published tables, and without the
commit the next reader cannot tell drift from a bug. `benchmarks/instances/setcover/comparison.csv`
is the pattern to copy.

## Architecture

CBLS = constraint-based local search. ViolationLS (guided local search over single- and compound-variable jumps) on an expression DAG with penalty-method feasibility. Full details in `docs/architecture.md`.

### Core pipeline

1. **Model building** (`include/cbls/model.h`, `include/cbls/expr.h`) — Declare typed variables (Bool, Int, Float, List, Set), build expressions via operator overloading, add constraints, set objective, call `close()` which topologically sorts the DAG.

2. **Expression DAG** (`include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`) — Variables use negative handles `-(id+1)`, nodes use non-negative `id`. 23 operation types. Two evaluation modes:
   - `full_evaluate`: evaluate all nodes in topo order (initialization)
   - `delta_evaluate`: BFS dirty-marking from changed variables, recompute only affected nodes (moves)
   - Reverse-mode AD via `compute_all_partials` for the continuous (Newton) jump-value engine

3. **Search** (`src/search.cpp`) — ViolationLS batch outer loop (Davies et al. CPAIOR 2024, Algorithm 6). The objective is folded into the constraints as `obj <= bound`; each batch is a Feasibility Jump, Novelty Jump, or STRUCTURAL batch (selected by config probabilities). The objective bound is tightened on each new real-feasible solution; on stagnation the assignment is perturbed or diversified via LNS.

4. **Feasibility Jump** (`src/feasibility_jump.cpp`) — Generalised Feasibility Jump: a `JumpTable` of cached per-variable best jumps (score = `-W·δ_G`), best-of-N scan-set sampling, GLS weight dynamics (bump violated + ρ-decay), and Novelty Jump compound moves (Algorithms 4–5). Float jump values come from Newton-toward-violated-root candidates via reverse-mode AD.

5. **Moves** (`src/moves.cpp`) — typed move generators by variable type (bool flip, int ±1/rand, float perturb, list swap/2opt/relocate/or-opt, set add/remove/swap). Scalar moves are subsumed by FJ's jump values; the list/set moves feed the STRUCTURAL batch.

6. **Inner solver** (`src/inner_solver.cpp`) — `FloatIntensifyHook`: coordinate descent over float variables using Newton steps on violated constraints + backtracking line search on objective. Triggered on each new feasible solution.

7. **LNS** (`src/lns.cpp`) — Destroy 30% of variables, repair via FJ. Fires every N diversification kicks. Accepts on a lexicographic (real-violation, objective) key.

8. **Violation & penalty** (`src/violation.cpp`) — `total_violation = Σ W[c]·max(0, viol_c)` with per-constraint GLS weights `W`. `weighted_violation_delta` is the no-commit counterfactual δ_G. `augmented_objective() = obj + total_violation()` is the penalty-method metric the inner solver descends.

9. **Parallel search** (`src/pool.cpp`) — `SolutionPool` + `ParallelSearch` with opportunistic (independent seeds) or deterministic (epoch-sync) modes.

10. **I/O** (`src/io.cpp`) — JSONL `.cbls` model format. CLI in `src/cli.cpp`.

### Key extension points

- **`InnerSolverHook`** — subclass to provide domain-specific continuous optimization. The reference implementation is the built-in `FloatIntensifyHook` (`include/cbls/inner_solver.h`, `src/inner_solver.cpp`); no benchmark currently ships a custom one, the worked example having gone with pharma-glsp (#28).
- **New operations** — add to `NodeOp` enum in `dag.h`, implement `evaluate()` in `dag.cpp`, `local_derivative()` for AD, `delta_evaluate` support in `dag_ops.cpp`

### Build targets

| Target | Source | Description |
|--------|--------|-------------|
| `cbls_lib` | `src/*.cpp` | Core library (static) |
| `cbls_cli` | `src/cli.cpp` | CLI executable |
| `cbls_tests` | `tests/*.cpp` | Catch2 test suite |
| `cbls_uc_chped` | `benchmarks/uc-chped/` | UC-CHPED benchmark runner |
| `cbls_mipfeas` | `benchmarks/mipfeas/` | MIPfeas runner (one instance per process) |
| `cbls_minlplib` | `benchmarks/minlplib/` | MINLPLib benchmark runner |
| `cbls_setcover` | `benchmarks/setcover/` | OR-Library set-covering runner (Set-variable coverage check, #93) |

### Dependencies

- **nlohmann/json** v3.11.3 — JSONL I/O (FetchContent)
- **Catch2** v3.5.2 — C++ tests (FetchContent)
- **nanobind** — Python bindings (optional, via scikit-build-core)

### C++ standard

C++17 (`CMAKE_CXX_STANDARD 17`). See the C++ subsection under **Standards** above.

## Benchmarks

Each benchmark follows the same pattern:
- `data.h` — C++ data structures + constexpr data arrays
- `*_model.h` — model builder function
- `*_hook.h` — optional custom `InnerSolverHook`; no benchmark ships one today (see **Key extension points**)
- `*.cpp` — runner executable
- `reference_solve.py` — SCIP/PySCIPOpt baseline
- `verify_*.h` — solution correctness checker

The `benchmarks/chped/` directory is reference-only (Benchmark 1 was dropped).

### Benchmark priority

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
| 1 | `mipfeas` | OR-Tools CP-SAT `num_violation_ls` worker **only** | head-to-head correctness *and* performance against the reference implementation of the same jump-based algorithm | #90 |
| 2 | `minlplib` | SCIP (nonlinear) | performance on non-convex MINLP — the regime CP-SAT's LS worker cannot express | #89, #87 |
| 3 | `uc-chped` | Pedroso 2014 | a good result on a published unit-commitment formulation | #25, #91, #92 |

**Priority 1 is deliberately not a MIP shootout.** Compare only against CP-SAT's
`num_violation_ls` worker (same algorithm, different implementation). Do **not**
compare against CP-SAT's default full portfolio, Xpress, Gurobi or CPLEX — see
epic #87 for why that framing is rejected.

`nuclear-outage` (#26), `bunker-eca` (#27) and `pharma-glsp` (#28, #94) are
**removed**. Their benchmark code, instance data, tests, CMake targets and docs
are gone from the tree. Nothing should be restored from history; if a
ROADEF-style outage-scheduling, maritime bunker/ECA or pharmaceutical
lot-sizing benchmark is ever wanted again it starts from a new epic.

**Why pharma-glsp went**, since it sat on this list until now and was not
obviously doomed: its
sole justification was being the one List-variable use case, and that did not
survive scrutiny. The committed model was a macro-period *relaxation* (audit
#76) — C1, C2, C6, C7 and C10 unmodelled, disposal structurally inert — which
is why its objectives ran 2-15x *below* the paper's. Expressing the paper's
actual GLSP-RP faithfully needs one Bool per (product, micro-slot) plus Floats
and nothing else; the permutation `List` was never the natural object, it was a
workaround for the engine not having a sequence-with-repeats type. And the
faithful formulation is a MILP — Goerler et al. solved it with CPLEX — so it
fails the criterion above. The List claim now rests on nothing; see the
**Structured variables** note below before reasserting it.

**Retiring a benchmark is a tracker job as well as a tree job.** An epic's own
sub-issue list is not the full set of things that point at it: grep *every* open
issue body for the epic number and the benchmark slug before declaring it
closed. Removing bunker-eca turned up three issues its sub-issue list missed — a
generic engine issue mis-filed under the epic, a cross-benchmark maintenance
issue naming it in passing, and a test issue citing one of its hooks. Removing
nuclear-outage was worse: GitHub's sub-issue list for #26 held 8 of the 14
issues that pointed at it, the other 6 being body-linked with "Part of #26". A
closing keyword in a commit message closes the epic only, never its sub-issues,
so close those explicitly.

Not every hit is closeable, though. An issue that is still valid and merely
*cites* the retired benchmark's code — a hook name, a file path, a list of
affected benchmarks — needs its body **edited** to name a surviving example
instead; closing it would drop live work. Retiring nuclear-outage needed
closure on fifteen issues and an edit on four — #103, #38, #31, and #119, an
unrelated engine issue whose acceptance criterion hard-coded a suite size the
removal invalidated. Grep for stale test counts as well as for the slug; #119
named neither the benchmark nor the epic.

Close a retired benchmark's sub-issues as **not planned**, not completed — the
work was abandoned, not delivered, and a later query filtering closed-as-
completed to reconstruct what shipped would otherwise be wrong by a dozen or
more. `gh issue close --reason` is silently a no-op on an already-closed issue;
to correct one after the fact use
`gh api --method PATCH /repos/:owner/:repo/issues/<num> -f state=closed -f state_reason=not_planned`.
The epic itself stays *completed* — retiring it was the job, and it got done.

`setcover` is **not a fourth benchmark**. It is the scoped coverage check the
`Set` variable type had been missing (#93): ten small OR-Library set-covering
instances, run under both a `Set` and a Bool encoding, whose result is a
documented limitation rather than a comparative claim (see
`benchmarks/instances/setcover/README.md`). Don't grow it into an epic.

### Structured variables: the claim is currently unsupported

Do not write, in docs or issues or commit messages, that List variables are
validated on a published formulation. They are not, as of the pharma-glsp
retirement. The evidence position is:

- **`Set`** — measured and **negative** (#93). The `Set` encoding lands at
  8.5-9.9x the proven optimum where the same data in Bools is within 9-20%.
- **`List`** — **no evidence either way**. The only List benchmark was
  pharma-glsp, whose model was a relaxation, and whose faithful form does not
  need a List at all.

The mechanical cause of the `Set` result is generic and applies to `List`
equally, and it has **two independent halves** — fixing either alone changes
nothing:

1. `FeasibilityJump::jumpable` (`src/feasibility_jump.cpp`) is a deliberate
   Bool/Int/Float whitelist, so no structured variable ever gets a jump-table
   entry regardless of its derivative. Its comment says the whitelist is
   intentional.
2. `local_derivative` (`src/dag.cpp`) returns `0.0` for *every* structural op —
   `At`, `Count`, `Lambda`, `PairLambda` — so there is no AD signal to build a
   jump value from either. (`setcover`'s `Set` model reads through `Lambda`;
   `PairLambda` was the retired benchmark's op.)

What is left is `set_moves`/`list_moves` drawing uniformly at random, where the
scalar path has FJ's jump table and best-of-N scan-set sampling to steer it.
Note the contrast is between the two *paths*, not the individual generators —
`int_rand` and `float_perturb` are themselves uniform. **Cost-aware
structural move selection is the prerequisite** for any renewed structured-variable
claim, and `setcover` is its ready-made A/B harness — same instances, both
encodings, a published baseline already committed. Fix the guidance, re-run
setcover, and only then consider whether a new List/Set benchmark is worth
building. Adding one first just reproduces #93's negative result in a new domain.

### Benchmark worktrees

Active work happens in sibling git worktrees under `~/code/my/cbls/`. Each session works ONLY on its assigned benchmark.

| Worktree | Problem | Epic |
|----------|---------|------|
| `uc-chped/` | UC-CHPED: unit commitment + valve-point dispatch | #25 |

Engine-wide (cross-cutting) work is tracked under epic #24.

### Common benchmark workflow

Each benchmark session must follow these steps in order:

0. **Plan first** — Read the benchmark's epic (linked above) and its open sub-issues, plus `docs/architecture.md` (solver internals). Investigate what already exists for your benchmark (data, reference solver, model code). Propose an approach and wait for approval before implementing. This is the Plan phase — do not skip it.

1. **Download/prepare instance data** — Write a download/generation script into `benchmarks/instances/{name}/`. Follow the pattern of `benchmarks/instances/uc-chped/` (Python data definitions + JSONL serialization). Source data from public benchmark libraries, competition archives, or papers.

2. **Find a reference solver** — Implement a reference solver in `benchmarks/{name}/reference_solve.py` using SCIP (PySCIPOpt) or another open-source solver if SCIP can't handle the formulation. Follow the pattern in `benchmarks/chped/reference_solve.py`.

3. **Collect best-known results** — Find published results from papers/competitions. Write to `benchmarks/instances/{name}/comparison.csv` with columns: instance, method, objective, gap, source.

4. **Implement CBLS model** — Create `benchmarks/{name}/data.h` (C++ data structs + loaders) and `benchmarks/{name}/{name}_model.h` (model builder). Follow the pattern of `benchmarks/chped/chped_model.h`. **Critical rules:**
   - Implement features generically in `include/cbls/` and `src/` — not benchmark-specific hacks
   - You may READ files in other worktree sibling folders (e.g., `../cbls/`, `../uc-chped/`) to understand patterns, but NEVER WRITE to them or to their git branches
   - If the solver needs new ops, moves, or hooks — implement them in the core library so all benchmarks benefit
   - Add a runner executable in `benchmarks/{name}/{name}.cpp`
   - Add Catch2 tests in `tests/test_{name}.cpp`
   - Update `CMakeLists.txt` for new executables and tests

5. **Run comparison** — Compare CBLS vs reference solver vs best-known results. Report objective, gap %, and solve time. Update `comparison.csv`.

6. **Verify correctness** — Check CBLS solutions against best-known solutions. Feasibility must be verified (all constraints satisfied). Objective should be within reasonable gap of BKS.

7. **Commit often** — Commit to your worktree branch (`bench/{name}`) after each meaningful step. Use descriptive commit messages.

8. **Self-review loop** — After each commit, review your own changes for issues/nits. Fix and commit again. Repeat until clean.

9. **Do not interrupt the user** — No exceptions. Keep going until the benchmark is fully implemented, running, and producing correct results. Only stop if you hit a fundamental blocker that requires API/architecture changes.

### Cross-worktree rules

- Each session works ONLY on its assigned benchmark
- READ other worktrees for reference — never write to them
- Merge to main via squash-merge when done
- Pull main into your branch before final merge to pick up other benchmarks' core changes
