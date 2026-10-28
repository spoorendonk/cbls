<!-- wcp: CLAUDE.md base=32c47ef20d45 type=cpp-python -->
<!-- wcp-declined: e9feadcddcb5 Testing (GoogleTest) -->
<!-- wcp-declined: cc3df5ee540d Testing (pytest) -->
<!-- wcp-declined: 004504610dd6 Dependencies -->
<!-- wcp-declined: c4fc130d1235 Ownership and Lifetime -->
<!-- wcp-declined: cf8d0fc1773f Type Conversions -->
<!-- wcp-declined: 7e7de94f1273 Testing Bindings -->
<!-- wcp-declined: 9070a09f6181 LSP -->
<!-- wcp-declined: aa64ca55c171 Development Workflow -->
<!-- wcp-declined: 2f2d1638a52b Hooks Are Files In This Repository -->
<!-- wcp-declined: a66bab4a00b4 This Document Carries A Provenance Line -->
<!-- wcp-declined: 527ae62c7a65 Running wcp -->
<!-- wcp-declined: 2b4541d688e6 Commit Messages -->
# Project: cbls

CBLS = constraint-based local search: ViolationLS (Davies et al., CPAIOR 2024)
over an expression DAG. Generalised Feasibility Jump with GLS weights drives
feasibility, the objective is folded in as `obj <= bound`, an inner solver
polishes floats, perturbation/LNS diversify. C++17 core, nanobind bindings.

This file holds what every session needs. The rest is read on demand:

| Read | When |
|---|---|
| `docs/architecture.md` (its **Source Map** first) | engine internals: pipeline, files, defaults, extension points, build targets |
| `docs/git-hooks.md` | a hook refuses; editing a hook, `.clang-tidy` or the Python gate; review receipts; wcp's files |
| `docs/development.md` | why the fences look as they do, venv setup, the test roster and its six pinned counts, git workflow, agent-team runs, writing issues |
| `docs/measuring.md` | any timing, A/B, published number or table |
| `docs/profiling.md` | heap/CPU profiling, sanitizers |
| `docs/benchmarks/README.md` | benchmark layout, priority rationale, structured-variable evidence, benchmark workflow, retiring one |
| `benchmarks/common/README.md` | shared runner headers and driver package |

Issues, reviews, pushing, scratch and runs follow wcp's shared and group rules.
Where this file is more specific, it wins.

## Communication Style

Be terse. No preamble. No filler.

## Code Navigation

LSP (`goToDefinition`, `hover`, `documentSymbol`) for symbol questions; Grep
with a small `head_limit` and `-n` for strings; `Read` with `offset`/`limit`
for the slice. Bound the output, whichever tool produces it.

## C++

- **C++17.** No `std::expected`, concepts or ranges.
- Google-based style via `.clang-format` and `.clang-tidy`, both **pinned
  wheels in `.venv/`** (`clang-format`/`clang-tidy` 22.1.8), not system tools.
- Naming (encoded in `.clang-tidy`): functions, locals, parameters
  `lower_case`; private members trailing `_`; constants `kCamelCase`; classes
  `CamelCase`; namespaces `lower_case`. Exceptions: `Model::Bool/Int/Float/List/Set/Constant`;
  `src/io/` disables the naming check because `mps_reader.cpp`/`solu_reader.cpp`
  are vendored from `spoorendonk/mipx` and stay diff-clean (watch new code there).
- `src/io/.clang-tidy` **must** keep `InheritParentConfig: true`. Verify any
  edit to either `.clang-tidy` with `cd src/io && ../../.venv/bin/clang-tidy --dump-config`.
  The root `Checks:` is a `>` folded scalar: `#` there is literal text that
  corrupts the list. Both failures fail open.
- `#pragma once`; minimal includes in headers, forward-declare.
- New first-party target: call `cbls_enable_warnings()` (root `CMakeLists.txt`).

## Python

- `ruff` (format + lint) and `mypy --strict`, configured and **pinned exactly**
  in `pyproject.toml`; full annotations, `list[int]`, `X | None`.
- Dependencies get `>=` lower bounds; use `uv`.
- pre-commit gates the **whole tree**. The only mypy exclusion is `^build/`;
  never widen it. Naming waivers are per file and per rule in
  `[tool.ruff.lint.per-file-ignores]` (published-formulation symbols); a new
  one needs the same justification. History: `docs/git-hooks.md`.
- Never run `python`/`pytest`/`mypy` outside `.venv/bin/`; change the venv with
  `uv sync`, not `pip`.

## nanobind Bindings

- One `NB_MODULE(_cbls_core, ...)` in `python/bindings.cpp`, built by
  `nanobind_add_module` in `python/CMakeLists.txt` under `CBLS_BUILD_PYTHON`.
  No binding glue in `src/`; a new binding source must join that call.
- Python names `snake_case`, `nb::arg("name")`. nanobind owns by default;
  `rv_policy::reference` only when C++ owns and outlives; no raw pointers
  without a lifetime annotation; prefer value or `shared_ptr`, document
  ownership. `nb::ndarray` with dtype/shape for NumPy.
- **A `def_rw` on a vector the engine indexes unchecked, or a raw `int32_t`
  handle, is a segfault** (#156). Validate where Python hands it over (a
  `def_prop_rw` setter, a check at node creation), and pin it with a
  child-process test (`tests/python/test_model_handles.py`).
- **Never `reference_internal` to an element of a growable container.**
  `keep_alive` keeps the owner, not the element. Return an (owner, index)
  handle that resolves on each access (`VariableRef`, bound as
  `cbls.Variable`) or a copy (#167).

## Testing

- C++: **Catch2** in `tests/`, registered in `tests/CMakeLists.txt`, run via
  `ctest`. Python: pytest, `tests/python/test_<module>.py`, `conftest.py`,
  `parametrize`. Descriptive names (`returns_optimal_for_feasible_input`).
- Bindings are tested from Python, with round trips, not from C++.
- `bugprone-unchecked-optional-access` cannot see through `REQUIRE`: write
  `if (!o.has_value()) { FAIL("..."); return; }`. No NOLINT, no disabling.
- A Python test that could deadlock the interpreter (a Python callable handed
  to C++ threads) runs in a **child process** under a timeout
  (`tests/python/test_pool.py`).
- Tests spawning `git` must not inherit hook `GIT_*` variables; `conftest.py`
  strips them (#160). Don't work around it.
- Pass terse flags yourself: `ctest --progress`, pytest `--tb=short -q`.
- `ctest -LE slow` is the fast loop (run it before committing; no hook does);
  `ctest` runs everything. Don't tag `[timing]` without a concrete reason.
- **Test counts are hard-coded in six places** (`docs/development.md`, **Fast
  vs. slow tests**). Any roster change updates all six in the same commit,
  re-derived from `ctest -N` / `pytest --collect-only`.

## CMake

`CMAKE_EXPORT_COMPILE_COMMANDS ON`; FetchContent for dependencies; every
target (library, CLI, examples, benchmark runners) in the root
`CMakeLists.txt` — only `tests/` and `python/` carry their own.

## Git Hooks

`.githooks/` (tracked; `cmake -B build` sets `core.hooksPath`). `pre-commit`
checks, never rewrites: ruff, mypy, clang-format over the tree; runs no tests.
`commit-msg`: Conventional Commits, subject ≤72, no `fixup!`/`squash!`.
`pre-push`: no merge commits on main, review receipt required, then the
`clean`/`build`/`test` fences below as local CI. Internals: `docs/git-hooks.md`.

**Checked by hand before every push**, because no hook enforces it now: zero
clang-tidy warnings (`.venv/bin/clang-tidy -p build <changed files>`) and zero
first-party compiler warnings in the build log.

- A clang-tidy finding is new code's doing: **fix it**. Never add a check to
  `.clang-tidy`'s disabled list or tune an option to silence it.
- `NOLINT` only as `NOLINTNEXTLINE(readability-function-cognitive-complexity)`
  with the reason at the site (see **Complexity**), plus one pre-existing
  `bugprone-float-loop-counter`; all five are listed in `docs/git-hooks.md`.
- No `-Werror`, `-w` or `-Wno-*` on first-party targets. Never add `default:`
  to a `NodeOp` switch.
- **Never `--no-verify`** unless explicitly asked. Fix the cause.
- No `hooks` block in `.claude/settings.json`; `.claude/` is local tooling.

## Git Workflow

Trunk-based, linear history: commit to `main`, push when gates pass. A
feature branch is optional, cut from an up-to-date `main` (never from another
feature branch, except serialized parallel issues: `docs/development.md`),
short-lived, landed by rebase or squash. After a push, close each issue it resolved
(`gh issue close N -c "..."`) and delete any branch used.

Commits: Conventional Commits (`feat`, `fix`, `refactor`, `test`, `docs`,
`style`, `perf`, `chore`, `build`, `ci`), subject ≤72, saying **why**.

## Working Through Issues

Follow wcp's rules for a goal's issues, one at a time: issue check (plan as an
issue comment when unclear; re-plan if a later issue's plan changes it) →
implement → gate (the full fences, even for a trivial-looking change) and
commit → **one cold reviewer** (a fresh subagent running `/code-review` on
unpushed commits; a fresh one again for round two), at most two rounds, fix
between them → record the receipt on a clean tree, push, close. A real finding left after round two, or out of
scope, becomes its own issue. File only what has a concrete failure scenario
or a missing acceptance criterion; fix or drop nits. Agent-team and parallel
runs: `docs/development.md`; no subagent returns unreviewed work and no
orchestrator merges it (`docs/git-hooks.md`, **Agent Self-Review**). If `wcp`
is not on PATH, say so; don't install it unasked.

A hygiene goal (`wcp hygiene`) is a task worked from its note: findings are
edits to issues or new `tech-debt` issues, nothing is fixed in the round, one
cold reviewer checks them (at most two rounds), and it ends with the summary
comment the note specifies, whose first line is the marker wcp looks for.

## Self-Review

Review happens on **commits**, never uncommitted changes. Then, on a clean
tree, record and push the receipt pre-push requires:

```sh
git update-ref "refs/reviews/$(git rev-parse HEAD)" HEAD
git push --atomic origin HEAD "refs/reviews/$(git rev-parse HEAD)"
```

A new commit needs its own receipt. The fix after round two is confirmed by
that round's reviewer (`git show <sha>` against its findings), not reviewed
again. Meaning and limits: `docs/git-hooks.md`.

## Plan Adherence

Follow the agreed plan. To change it, stop and say so; never diverge
silently, and say out loud when an approach is not working. No TODOs or stubs
unless asked.

## Reference Correctness

Implementing from a paper, pseudocode or reference code: match it exactly —
no early exits, caps or shortcuts that change behaviour, no heuristic
approximations unless asked, every edge case kept. When in doubt, be faithful
and let tests decide.

## Complexity

On a complexity warning, split along real responsibilities, not mechanically.
If the domain makes the function irreducibly complex, say why in a comment
above it and add `NOLINTNEXTLINE(readability-function-cognitive-complexity)`.

## Common Mistakes

- Don't invent APIs, flags or methods; verify they exist. Check current docs.
- Don't suppress type errors or lint; fix the root cause.
- **Performance matters** — these are solvers. A complexity win (O(n²) → O(1)
  stamp) lands on a cost argument written at the site, naming the regime it
  wins and loses in. A micro-optimisation needs a profile first.

## Build & Test

Builds, ctest, clang-tidy and pushes run under the machine lock
(`xk lock -- CMD`) with at most 4 jobs (group rules). The fences below are
what pre-push runs, pinned to `git config hooks.localCiCpus` so `$(nproc)` is
4: a new clone sets it (`git config hooks.localCiCpus 0-3`) before its first
push. By hand, run them as `xk lock -- taskset -c 0-3 sh -c '…'`.
Why they look this way: `docs/development.md`.

```clean
rm -rf build
```

```build
py="$PWD/.venv/bin/python"; [ -x "$py" ] || py=$(command -v python); cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$py" && cmake --build build -j$(nproc)
```

```test
ctest --test-dir build --output-on-failure -j$(nproc) && (py="$PWD/.venv/bin/python"; [ -x "$py" ] || py=$(command -v python); CBLS_REQUIRE_BINDINGS=1 CBLS_REQUIRE_BENCHMARKS=1 "$py" -m pytest --tb=short -q || [ $? -eq 5 ])
```

- No `-DCMAKE_BUILD_TYPE`: `CMakeLists.txt` defaults to Release. An existing
  `build/` keeps its first build type and `CBLS_SANITIZE`/`CBLS_PROFILE`.
- A full clone makes its own venv, never a symlink to another checkout's:
  `uv sync --extra dev --extra benchmarks` (the `benchmarks` extra is required
  by the test fence). Only a subagent worktree under this checkout symlinks
  the checkout's `.venv`, and removes it after.
- Single tests: `ctest --test-dir build -R "name_substring"`;
  `.venv/bin/pytest tests/python/test_foo.py::test_bar -x`.

## Measuring and testing engine changes

Full rules: `docs/measuring.md`. The short list:

- **A regression test must be shown red before the fix**: revert only the
  production change in a throwaway copy and watch it fail.
- **Never run time-limited comparisons concurrently.** Serial A/B, one budget,
  idle machine (`uptime`), Release, no sanitizer. Sub-second runs: read CPU
  time, not wall.
- Measurement is required for any number in a doc or table, for any
  **parameter choice** (fit on one roster, report on a held-out one), and for
  any "faster" claim. Subtracted whole-program timings are not measurements.
- A runner writing a published table loads its roster first, refuses an
  incomplete one, and writes to a temp path then renames.
- **A search-trajectory change invalidates every published table**: record the
  engine commit with any regenerated result. Only MINLPLib's tables are
  test-checked.
- Reference values (best-known, optimal) are derived once and pinned; an
  engine-vs-engine comparison is an output, regenerated, not defended.

## Benchmarks

Detail: `docs/benchmarks/README.md`. Comparative benchmarks, worked top down:

| # | Benchmark | Compared against | Purpose |
|---|---|---|---|
| 1 | `mipfeas` | CP-SAT `num_violation_ls` worker **only** | same-algorithm implementation check (#87, #106) |
| 2 | `minlplib` | SCIP (nonlinear) | non-convex MINLP, beyond CP-SAT's LS (#87) |
| 3 | `uc-chped` | Pedroso 2014 | published unit-commitment formulation (#25, #92, #131) |

- Priority 1 is **not a MIP shootout**: never compare against CP-SAT's full
  portfolio, Xpress, Gurobi or CPLEX (#87).
- A comparative quality claim needs terms CP-SAT cannot express (transcendental
  or non-convex continuous). Nothing outside the table is comparative.
- Worked examples per structured type, not comparative: `Set` → `setcover`
  (measured, negative; don't build another `Set` benchmark), `List` → CVRP
  (epic #182; unpublished until #204).
- **Never write that `List` is validated on a published formulation**: there
  is no `List` evidence until #204. List engine work is filed against measured
  CVRP data; don't reopen #183-#185, #187, #189.
- Generic features go in `include/cbls/` and `src/`, never benchmark hacks.

## Issue Tracking

GitHub Issues via `gh`; HTTPS remotes. Read with
`gh issue view N --json title,body,labels,state,comments`. Write issues to be
picked up cold: self-contained, no local paths, stable links. Don't file small
follow-ups; fix them inline or leave them.

## CLAUDE.md Discipline

When a better instruction would have prevented a mistake, fix this file in the
same commit. Keep it under ~15 KB: reference material goes to `docs/`, with a
pointer here. Leave the provenance lines at the top in place.
