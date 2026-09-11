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

- **C++17** (`CMAKE_CXX_STANDARD 17`). `std::expected`, concepts and ranges are *not* available here; don't reach for them.
- Style: Google-based, enforced by `.clang-format` and `.clang-tidy` at the repo root. Both tools are **pinned PyPI wheels in `.venv/`** (`clang-format==22.1.8`, `clang-tidy==22.1.8`), not system packages — `.venv/bin/pip install -e '.[dev]'` installs them, and the hooks prefer `.venv/bin` over `PATH`. The exact pin keeps formatter and tidy output identical across checkouts.
- Naming is **not** Google's default — `.clang-tidy` encodes what this codebase actually does:

  | Kind | Style | Example |
  |---|---|---|
  | Functions (free and member) | `lower_case` | `full_evaluate`, `weighted_violation_delta` |
  | Locals, parameters, `const` locals | `lower_case` | `total_violation`, `node_id` |
  | Private members | trailing `_` | `out_`, `model_` |
  | File-scope / static / class / `constexpr` constants | `k` + `CamelCase` | `kVersion`, `kEps`, `kMpsInf` |
  | Classes | `CamelCase` | `Model`, `JumpTable` |
  | Namespaces | `lower_case` | `cbls` |

  Two exceptions: `Model::Bool/Int/Float/List/Set/Constant` stay CamelCase because they name the type they create (and `bool`/`int`/`float` cannot be lowercased without hitting a keyword), and `src/io/.clang-tidy` disables the naming check for that directory because `mps_reader.cpp` and `solu_reader.cpp` are vendored from `spoorendonk/mipx` and must stay diff-clean against upstream. The three native adapters there (`mps_to_model.cpp`, `nl_reader.cpp`, `nl_to_model.cpp`) lose naming coverage as a side effect — keep an eye on new code in that directory.

  That directory config **must** carry `InheritParentConfig: true`. Without it clang-tidy *replaces* the root `Checks:` instead of appending to it, leaving only the always-on `clang-diagnostic-*` — the directory then runs no registered check at all and clang-tidy exits 1 with `Error: no checks enabled.` Verify any edit to **either** `.clang-tidy` with `cd src/io && ../../.venv/bin/clang-tidy --dump-config` and confirm the root's check list is actually present. This is the same verification **Git Hooks** below demands for the root file's folded-scalar hazard, and for the same reason: both failures fail open.
- Use `#pragma once` for include guards.
- Minimize includes in headers. Forward-declare where possible.

### LSP

Install `clangd-lsp@claude-plugins-official` plus `clangd` itself (`apt install clangd` or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`. Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`) over `Read` for symbol questions.

## Python

- Style: enforced by `ruff` (format + lint) and `mypy --strict`, configured in `pyproject.toml`.
- All functions must have full type annotations (mypy strict mode).
- Use built-in generics (`list[int]`, `dict[str, Any]`) and `|` union syntax.
- Pin dependencies with `>=` lower bounds in `pyproject.toml`. Use `uv` or `pip`.

### Where that standard is actually enforced

`pre-push` runs `ruff check`, `ruff format --check` and `mypy --strict` over
**every tracked `.py` outside `benchmarks/`** as a **hard block** — the same
standing as clang-tidy, and for the same reason. The scope itself is the
`PY_GATE_EXCLUDE` regex in `.githooks/pre-push` — that string is the source of
truth and the one place to edit; this section carries the reasoning and the
measurements behind it, and issue #155 is what removes the exclusion. Because
they block, `ruff` and `mypy` are **pinned exactly** in `pyproject.toml`, on the same
argument the clang tools are: a floating version turns "these files are at zero"
into "at zero on whichever version this checkout resolved".

It is written as an *exclusion*, never an allowlist. An allowlist would be
fail-open in the one direction that matters: a Python directory nobody has
created yet would be gated by nothing and could rot exactly as the five
binding-test modules did.

`benchmarks/` is excluded on a measurement, not a preference. It carries 22 ruff
findings, 5 unformatted files and 37 `mypy --strict` errors — that last counted
over a file list with `uc-chped` removed, because
`benchmarks/instances/uc-chped/` cannot be type-checked *at all*: an
`__init__.py` inside a hyphenated directory makes mypy abort the whole run with
`not a valid Python package name` before checking anything, so no single
invocation spanning the tree even completes. Cost was never the objection —
`ruff` is ~0.01s and `mypy --strict` ~4.5s cold / ~0.15s warm, on a gate that
also does a clean rebuild plus the full `ctest` suite.

**But `benchmarks/` is not actually unchecked, and the trap is worth knowing.**
`ruff` sees only the gated list, so `benchmarks/` really is outside it — but
`mypy` **follows imports**, and eight test modules import benchmark packages, so
every benchmark module reachable from a gated file is hard-gated too. The tree is
green today only because the three benchmark files carrying `mypy` errors are
ones nothing gated imports. Add one test that imports
`benchmarks/chped/data.py` and the next push blocks on 16 pre-existing
errors nobody in that push wrote — fix them, or fix #155, rather than widening
`PY_GATE_EXCLUDE`.

Both escape routes that let the five modules rot are closed for the gated set,
and both mattered: pre-commit lints only **staged** files — and its ruff step
auto-fixes rather than blocks, so a non-fixable finding like `E741` survives
staging however often the file is edited — while pre-push's other Python steps
see only the pushed file list and merely warn. That is how 139 `mypy --strict`
errors and 9 ruff errors accumulated with every gate green throughout (#154).

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

C++ tests use **Catch2** (not GoogleTest): files in `tests/`, registered in `tests/CMakeLists.txt`, run via `ctest`. Python tests use **pytest**: `test_<module>.py` under `tests/python/`, `conftest.py` for shared fixtures, `pytest.mark.parametrize` for data-driven cases.

- Name tests descriptively — `returns_optimal_for_feasible_input`, `test_solver_returns_optimal_for_feasible_input`.
- Test nanobind bindings from Python with pytest, not from C++ — the binding is an implementation detail. Include round-trip tests: create in Python → pass to C++ → get result back.
- An `std::optional` guarded by `REQUIRE(o.has_value())` still trips `bugprone-unchecked-optional-access`, which is enabled: the check cannot see through Catch2's expression templates, and `.value()` reads the same way to it. Write `if (!o.has_value()) { FAIL("..."); return; }` instead — the `return` is what makes the flow analysis see the guard, even though `FAIL` throws. Do **not** reach for a `NOLINT` or add the check to `.clang-tidy`'s disabled list; both routes are closed (see **Git Hooks**).
- A Python test that could **deadlock the interpreter** — anything handing a Python callable to C++ that spawns threads — must run its scenario in a **child process** under a wall-clock timeout. No in-process deadline can report it: `pytest-timeout`'s `thread` method runs its timer on a Python thread that needs the very GIL the test is starving, and its `signal` method only takes effect at the next bytecode in a main thread parked inside a C++ `join()`. `tests/python/test_pool.py` is the pattern — scenarios in a `__main__` block, `subprocess.run(..., timeout=...)` in the test.
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

- `pre-commit` — auto-formats staged C++/Python/shell (clang-format, ruff, shfmt), applies safe clang-tidy fixes, re-stages, then runs the affected test suite. The test step is a hard block; the ruff step auto-fixes and does **not** block, so a non-fixable finding commits and is caught at push instead. The clang tools come from `.venv/bin` (resolved by `.githooks/resolve-venv.sh`); if neither the venv nor `PATH` has them it says so rather than skipping quietly.
- `commit-msg` — Conventional Commits format.
- `pre-push` — the clean build + **full** suite from `## Build & Test` below, then **clang-tidy as a hard block** — with **the Python lint/type gate, also a hard block, running first** because it needs nothing the build produces (scope and rationale in **Where that standard is actually enforced** above, the one place they are stated) — and ruff-complexity/shellcheck/mypy over the pushed file list remaining warnings. Both the ```build and ```test fences must resolve or the push is blocked; there is no auto-detect fallback, because guessing a build would gate a different one than the documented build.

  clang-tidy blocks because the tree is held at **zero** warnings. Getting there
  took a **ratchet** — a list in `.clang-tidy` of checks disabled by name, each
  with its unfixed finding count — and **that list is now empty** (issue #121,
  closed). Every `bugprone-*`/`modernize-*`/`performance-*`/`readability-*` check
  is enabled except six structural exemptions, and each of those is off because
  the *check* is wrong for this codebase, with the measurement that says so
  written beside it in the file.

  So a warning at push is new code's doing. **Fix it — do not add the check to
  the disabled list.** There is no backlog list left to append to: putting a
  name there now has to clear the same bar as the six, a structural cause plus a
  measurement, which is a case to argue rather than a way to get a push through.
  `.clang-tidy` is the source of truth and carries its own sweep recipe — read
  the file rather than a count restated somewhere else.

  In-tree suppression is **sanctioned only where the reason is written at the
  site**, and the tree has exactly five, across two checks —
  `grep -rn NOLINT src/ include/ benchmarks/ tests/ python/` should return
  nothing else, and a sixth needs the same standard as `.clang-tidy`'s six
  exemptions. That grep prints **six** lines, not five: one is `mps_reader.cpp`
  naming the directive in prose in a comment, not applying it. Count the
  directives, not the matches. Four are `readability-function-cognitive-complexity`: on
  `src/dag.cpp`'s two 28-case `NodeOp` dispatch tables, where the score counts a
  table a human reads as one unit and any split would have to stay inlinable on
  the delta-evaluation and reverse-mode-AD hot paths; and on the two
  `src/io/mps_reader.cpp` functions that the vendored port-the-diff-upstream
  contract keeps whole. That is the route **Complexity** below describes — split
  along real responsibilities, or record why the function is irreducibly complex
  — and it is specific to that check, not a general licence. The fifth is
  `bugprone-float-loop-counter` in `src/feasibility_jump.cpp`, on an
  integer-valued enumeration whose `steppable` guard on the line above is
  exactly what makes `v += 1.0` exact and the loop finite.

  One **check option** is tuned, which is otherwise the same fail-open move as
  disabling a check: `readability-function-cognitive-complexity.IgnoreMacros`.
  It is there because clang-tidy names a Catch2 `TEST_CASE` body after the macro
  that generates it and charges it for every `SECTION`'s `if` and every
  `REQUIRE`'s `try`/`catch`. `.clang-tidy` carries the measurement and the probe
  showing the check still fires on a genuinely convoluted test body. Tuning any
  other option to make a finding go away is not open — see the
  `bugprone-easily-swappable-parameters` note in the same file for why.

  Note `.clang-tidy`'s `Checks:` is a YAML `>` folded scalar, where `#` is *not*
  a comment — it is literal text that silently corrupts the check list, and a
  trailing `# 25` on a check line swallows the line after it. Comments go above
  the key. Verify any edit with `clang-tidy --dump-config`, because a corrupted
  list fails open: checks silently stop running and the sweep goes green.

  Because both config hazards fail open, the pre-push clang-tidy step matches
  `(warning|error):` **case-insensitively** and treats any **nonzero clang-tidy
  exit** as a finding: a misconfigured directory reports `Error: no checks
  enabled.` with a capital E, which a lower-case-only pattern misses. It also
  lints a canary translation unit per config directory whenever a `.clang-tidy`
  is itself part of the push, since such a change otherwise touches no source
  file and would skip the step that polices it.

Nothing enforces `/review` at push time, by design — a gate keyed on gitignored local tooling can only be satisfied in whichever checkout happens to carry it, and passes silently everywhere else. `/review` is still expected on every change (see **Agent Self-Review**); running it is on you.

Gating lives in git hooks only — `.claude/settings.json` carries no `hooks` block, and none should be added. A `PostToolUse` formatter cannot see which file was edited, so it silently formats nothing; formatting belongs at commit time. Don't hand-tune formatting.

Three conventions therefore rest on you rather than on a tool: branch only from main (**Git Workflow**), never invoke `python`/`pip`/`pytest`/`mypy` outside `.venv/bin/` (**Build & Test**), and prefer the `Grep` tool to shelling out (**Code Navigation**).

`.claude/` is gitignored local agent tooling (settings, statusline, the `/review` command). Nothing in it is part of the repo.

### Fast vs. slow tests

The C++ suite is **365 `TEST_CASE`s**: 361 registered by `catch_discover_tests`
plus the **4 `[timing]` cases registered by hand**. Of the 361, **6 carry the
Catch2 `[slow]` tag** — the CHPED and UC-CHPED benchmark solves, ~103s of
aggregate (summed per-test) time, which `-j$(nproc)` compresses to a ~40s
wall-clock full run. `tests/CMakeLists.txt` discovers them in a second
`catch_discover_tests` call with `LABELS "slow"`, so:

- `ctest -LE slow` — the other 356 tests, ~9s with `-j`. This is what **pre-commit** runs.
- `ctest` — everything. This is what **pre-push** and CI run.
- `ctest -L timing` — 4 tests: `timing_structural_batch_deadline` plus the three
  `timing_throughput_*` floors added for #125. Each is registered by an explicit
  `add_test` (naming its Catch2 test case) so it can carry a `TIMEOUT` and be
  quarantined individually. Don't add tests to this class without a concrete
  reason.

**The Catch2 `[slow]` tag and the ctest `slow` label are not the same set here**,
and both numbers appear above: 6 tests carry the *tag*, 9 carry the *label*
(`ctest -N -L slow` says 9). The three throughput floors are tagged `[timing]`
— which is what keeps them out of both `catch_discover_tests` calls, since the
second spec is `[slow]~[timing]`; retagging one `[slow]` *instead of* `[timing]`
would get it discovered *and* hand-registered, i.e. run twice (adding `[slow]`
alongside `[timing]` changes nothing) —
and are given `LABELS "timing;slow"` by hand so their wall-clock seconds stay
out of the pre-commit set. Derive each count with the matching command rather
than by arithmetic: `ctest -N`, `ctest -N -L slow`, `ctest -N -LE slow`,
`ctest -N -L timing`.

These counts are hard-coded in **seven** places and nothing checks that they
agree:

1. this section,
2. the comment above `catch_discover_tests` in `tests/CMakeLists.txt`,
3. the build section of `README.md`,
4. the comment above the `ctest` call in `.githooks/pre-commit`,
5. the `.venv/bin/pytest` line in `README.md` for the Python side (478 tests, 81
   of them binding tests, echoed in prose by `pyproject.toml` and
   `tests/python/conftest.py`),
6. the `-LE slow` guidance and the ~40s/~304s figures in `docs/profiling.md`.
   Note its "302/302 green" sanitizer line is a **dated record of one run at a
   named commit**, not a current count — it says so inline. Leave it alone.
7. the binding count in **`## Build & Test`** below, in the paragraph explaining
   why the gated build turns `CBLS_BUILD_PYTHON` on ("81 binding tests silently
   unrun"). It is in this file, but not in this section, so a search that stops
   at the enumeration above misses it.

Update all seven in the same commit as any change to the test roster, and
re-derive every number from `ctest -N` / `pytest --collect-only` rather than
incrementing the one you find.

All timings here are **Release**, the build type `CMakeLists.txt` defaults to.
The tag was re-derived at `-O3`: 16 tests that earned it at `-O0` no longer come
close to the threshold, and carrying them cost pre-push a roster it did not need
while hiding them from the inner loop. That move is what took pre-commit from
~2s to ~8s — the fast set is no longer trivially fast, and the 6 that remain are
the only ones where the split still pays.

Measure both ways before tagging. These tests run under `-j$(nproc)`, and
contention inflates a solve by 1.4-1.7x: `UC-CHPED 40-unit 1-period feasibility`
is 9.95s alone and 17.1s under `-j12`. Tag if it exceeds ~10s on **either**
basis. Don't tag one just to get a green commit — pre-push will still run it.

**Never use `git push --no-verify` or `git commit --no-verify`** unless explicitly asked. A failing hook is a signal — fix the root cause.

## Git Workflow

Trunk-based development with linear history on main. Commit directly to main and push when local gates pass. There are currently no other worktrees or long-lived branches.

Feature branches are optional for larger changes:
- Always branch from main. Run `git checkout main && git pull` first.
- Never branch from another feature branch, with one exception: serialized
  issues under **Parallel Issue Workflow**, where the second issue's branch is
  cut from the first's so the pair lands as one fast-forward.
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

### Retiring a benchmark

If a benchmark is ever retired, it is a tracker job as well as a tree job:

- Grep **every** open issue body for the epic number *and* the benchmark slug — GitHub's sub-issue list is not the full set of issues that point at an epic, because many are body-linked with "Part of #N" instead. Grep for hard-coded test counts too; those go stale the same way.
- Close the epic's sub-issues **explicitly** and as **not planned** — a closing keyword in a commit message closes the epic only, and the work was abandoned, not delivered. `gh issue close --reason` is silently a no-op on an already-closed issue; correct one after the fact with
  `gh api --method PATCH /repos/:owner/:repo/issues/<num> -f state=closed -f state_reason=not_planned`.
- The epic itself closes as **completed** — retiring it was the job.
- An issue that is still valid and merely *cites* the retired code (a hook name, a file path, a list of affected benchmarks) needs its body **edited** to name a surviving example instead. Closing it would drop live work.

## Agent Self-Review

**Any agent or agent team that produces code must run `/review` on its own changes before that code can merge to main.** No subagent returns unreviewed work; no orchestrator merges unreviewed work. This applies to every agent team, not just the parallel-issue workflow.

## Parallel Issue Workflow

When the user brings multiple gh issues to work on at once:

1. **Propose parallelism first**, unless a goal already settled it. Offer it
   explicitly and wait for confirmation — don't silently start serial work. A
   `/goal` that names the issues and says how to take them has already answered
   this; don't ask again.
2. **Serialize issues that touch the same files or interfaces.** Two agents
   editing one file, or one row schema, is a merge conflict dressed up as
   parallelism. Decide this before spawning anything: name the shared files and
   say which issue goes first and why. The second issue's branch is then cut
   **from the first issue's branch**, not from main — that is the one sanctioned
   exception to **Git Workflow**'s "always branch from main", and it is what
   keeps the final merge a single fast-forward. Genuinely independent issues
   still branch from main and run concurrently.
3. **Orchestrator role.** Spawn one subagent per issue (Agent tool with
   `isolation: "worktree"`). Pass each subagent its gh issue number, its
   worktree path, and any plan file path.
4. **Subagents self-review** per the Agent Self-Review rule above. Subagents
   commit locally in their worktree and **do not push, merge, or touch another
   branch** — worktrees share `.git`, so the orchestrator sees their commits via
   `git log <branch>` with no network round-trip.
5. **Merging and pushing are the orchestrator's own call when the work is part
   of a stated goal.** A goal that names the issues to take *is* the
   authorisation to land them; stopping to ask again mid-run breaks the "no
   interruptions" half of the same instruction and buys nothing, because the
   thing actually standing between a bad merge and main is the pre-push hook and
   the cold review, not a second yes. **Outside a goal** — an ad-hoc "have a look
   at these two issues" — ask before merging.
6. **Final combined review, then push.** The orchestrator merges every approved
   branch into local main, runs `/review` over the merged result, and only then
   runs `git push origin main`. No pushes — of main or of a feature branch —
   happen before that final review. Then close the issues the push resolved and
   delete the branches and worktrees, per **Git Workflow**.

If the harness denies the merge or the push, that is a permission layer, not a
review finding: say so plainly, hand over the exact commands, and do not report
the work as shipped. Committed-on-a-branch is not pushed, and the final report
must not say otherwise.

A subagent worktree may READ sibling worktrees to understand patterns, but must never WRITE to one or to its git branch. A fresh worktree has no `.venv`; see **Build & Test** for the symlink it needs before building or testing.

## Commit Messages

Conventional Commits. The commit-msg hook enforces format.

- `type: description` or `type(scope): description`
- Types: `feat`, `fix`, `refactor`, `test`, `docs`, `style`, `perf`, `chore`, `build`, `ci`
- Subject ≤72 chars. Focus on **why**, not what.

## CLAUDE.md Discipline

When Claude gets something wrong, fix CLAUDE.md in the same commit. It's a living document — update it whenever better instructions would have prevented the mistake.

## Complexity

When a complexity warning fires, don't extract methods mechanically. Ask: what are the independent responsibilities here? Split along those boundaries. If the function is genuinely complex because the domain is, say why in a comment directly above it and suppress that one warning with `NOLINTNEXTLINE(readability-function-cognitive-complexity)`.

This is the **only** check a `NOLINT` is sanctioned for — plus one pre-existing `bugprone-float-loop-counter` — and only with the reason written at the site; **Git Hooks** above names all five in-tree examples. Every other clang-tidy finding gets fixed, not suppressed, and adding the check to `.clang-tidy`'s disabled list is not an option either.

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
- **Don't ignore type errors.** If mypy/clang-tidy flags something, fix the root cause — don't suppress. Both are hard blocks at push; silencing either — by disabling a clang-tidy check or by adding to `PY_GATE_EXCLUDE` — is the route explicitly closed (see **Git Hooks**).
- **Don't use deprecated patterns.** Check current docs, not training data.
- **Performance matters.** Most of our code is solvers — profile before micro-optimizing, but don't sacrifice perf for "clean code". `docs/profiling.md` is how: heap attribution, CPU profiling and the sanitizer build, with a dated tool-availability table naming the machine it was checked on. Numbers reached by subtracting two whole-program timings are not measurements.

## Build & Test

```clean
rm -rf build
```

```build
cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python" && cmake --build build -j$(nproc)
```

**No `-DCMAKE_BUILD_TYPE` here on purpose.** `CMakeLists.txt` defaults it to
Release when the caller sets none, so this fence, CI and a plain `cmake -B build`
all gate the same binaries from one place. An explicit `-DCMAKE_BUILD_TYPE=Debug`
still overrides it. The suite is mostly real solver runs, so the type is not
cosmetic: the full `ctest` was ~304s at the old empty default and is ~40s at
Release.

`CMakeLists.txt` also picks up `ccache` as a compiler launcher when the machine
has it (`apt install ccache`), which matters because pre-push's ```clean fence is
`rm -rf build`. Rebuilding the same path with a warm cache is **0.9s** against
**23s** cold. It is optional — a system package cannot be committed, so a machine
without it just builds normally.

```test
ctest --test-dir build --output-on-failure -j$(nproc) && (CBLS_REQUIRE_BINDINGS=1 .venv/bin/pytest --tb=short -q; rc=$?; [ $rc -eq 0 ] || [ $rc -eq 5 ])
```

**The gated build turns the Python bindings on, and the gated test run requires
them.** `CBLS_BUILD_PYTHON` defaults to `OFF` and `tests/python/conftest.py`
skips every test that imports `_cbls_core` when the module is missing, so a build
without the flag would leave 81 binding tests silently unrun.
`CBLS_REQUIRE_BINDINGS=1` turns that skip into a hard error. Bindings cost ~2.4s
of build and ~6s of pytest against a suite that already spends ~340s in `ctest` —
always build them.

`pytest` is **only** in `.venv/` — a bare `pytest` is not on PATH and
`python3 -m pytest` has no module, so always spell out `.venv/bin/pytest`.
A git worktree has no `.venv` of its own; symlink the main checkout's in before
building or running the Python suite there. The build needs it too — the
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

## Measuring and testing engine changes

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
through every later flag-less `cmake -B build`, and pre-commit gates on it. So
for a timing comparison either check `CMAKE_BUILD_TYPE` and `CBLS_SANITIZE` in
`build/CMakeCache.txt` or configure a fresh directory. The measurement recipes —
heap attribution, CPU profiling, sanitizers — are in `docs/profiling.md`.

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

**Nothing checks this.** `tests/python/test_minlplib_scip_baseline.py` is the
only test that reads a `comparison.csv` at all, so every other committed table
drifts silently and is caught only when someone re-measures. Prefer stating
results in a benchmark README with the engine commit named in the text — the
shape `benchmarks/instances/setcover/README.md` uses — unless a test is actually
going to read the table.

## Architecture

CBLS = constraint-based local search. ViolationLS (guided local search over single- and compound-variable jumps) on an expression DAG with penalty-method feasibility. Full details in `docs/architecture.md`.

### Core pipeline

1. **Model building** (`include/cbls/model.h`, `include/cbls/expr.h`) — Declare typed variables (Bool, Int, Float, List, Set), build expressions via operator overloading, add constraints, set objective, call `close()` which topologically sorts the DAG.

2. **Expression DAG** (`include/cbls/dag.h`, `src/dag.cpp`, `src/dag_ops.cpp`) — Variables use negative handles `-(id+1)`, nodes use non-negative `id`. 28 `NodeOp` operation types. Two evaluation modes:
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

10. **I/O** (`src/io.cpp`, `src/io/`) — JSONL `.cbls` model format in `src/io.cpp`; `src/io/` holds the MPS reader (`mps_reader.cpp` + `mps_to_model.cpp`, gzip via zlib, optional bzip2), the AMPL `.nl` reader (`nl_reader.cpp` + `nl_to_model.cpp`) and the MIPLIB `.solu` reader. CLI in `src/cli.cpp`.

### Key extension points

- **`InnerSolverHook`** — subclass to provide domain-specific continuous optimization. The reference implementation is the built-in `FloatIntensifyHook` (`include/cbls/inner_solver.h`, `src/inner_solver.cpp`); no benchmark currently ships a custom one.
- **New operations** — add to `NodeOp` enum in `dag.h`, implement `evaluate()` in `dag.cpp`, `local_derivative()` for AD, `delta_evaluate` support in `dag_ops.cpp`

### Build targets

| Target | Source | Description |
|--------|--------|-------------|
| `cbls_lib` | `src/*.cpp`, `src/io/*.cpp` | Core library (static) |
| `cbls_cli` | `src/cli.cpp` | CLI executable (output name `cbls`) |
| `cbls_tests` | `tests/*.cpp` | Catch2 test suite |
| `cbls_simple` | `examples/simple.cpp` | Minimal modeling example |
| `cbls_chped` | `examples/chped.cpp` | CHPED modeling example over `benchmarks/chped/` |
| `cbls_uc_chped` | `benchmarks/uc-chped/` | UC-CHPED benchmark runner |
| `cbls_mipfeas` | `benchmarks/mipfeas/` | MIPfeas runner (one instance per process) |
| `cbls_minlplib` | `benchmarks/minlplib/` | MINLPLib benchmark runner |
| `cbls_setcover` | `benchmarks/setcover/` | OR-Library set-covering runner (`Set`-variable coverage check) |

Examples, tests, benchmarks and bindings are each behind an option:
`CBLS_BUILD_EXAMPLES`/`CBLS_BUILD_TESTS`/`CBLS_BUILD_BENCHMARKS` default ON,
`CBLS_BUILD_PYTHON` defaults OFF (but the gated build turns it on).

### Dependencies

- **zlib** — required (`find_package(ZLIB REQUIRED)`), gzip-compressed MPS input
- **bzip2** — optional, `-DCBLS_USE_BZIP2=ON`, adds `.bz2` MPS support
- **nlohmann/json** v3.11.3 — JSONL I/O (FetchContent)
- **Catch2** v3.5.2 — C++ tests (FetchContent)
- **nanobind** — Python bindings (optional, via scikit-build-core)

## Benchmarks

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

`benchmarks/common/` is **not** a benchmark. It holds headers shared across
runners: `runner_args.h` and `search_config_flags.h`.

`runner_args.h` holds three things. First the flag-value *reporting policy* (a
bad double reports and returns NaN for a later positivity guard to turn into
exit 2; a bad integer reports and exits 2), layered over the pure parsing *rule*
in `include/cbls/arg_parse.h`. Second `ArgCursor`, the value-flag matching rule:
a flag that takes a value matches only when a value actually follows it, so a
trailing `--budget` falls through to the runner's unknown-argument path instead
of reading past `argv` or silently keeping a default that a published table would
then be built on. Third `same_file`, the published-table guard's path
comparison — canonicalised where the file exists, so that naming the published
table explicitly (which the documented commands do) is held to the same rule as
defaulting into it. `mipfeas`, `minlplib` and `uc-chped` share the first two;
`setcover` deliberately does not, and says why at its own `parse_args`. The
policy split is deliberate: the rule is a library concern, the policy writes to
stderr and calls `std::exit` and is pinned by
`tests/python/test_run_benchmark.py`, so it lives next to the programs those
tests run.

`search_config_flags.h` is the per-arm search configuration (#136): the flag
table, its parse, its validation, its application onto `cbls::SearchConfig` and
`solve()`'s arguments, and the canonical `search_config` cell every row carries.
`minlplib` and `uc-chped` share it. Two rules it exists to enforce: **a flag
whose field the engine ignores must not ship** (the header names the two
rejected — the GLS ρ and `use_fj`, whose argument `solve()` opens by discarding),
and **a non-default arm may never write a published comparison table**, which is
why `first_non_default_search_flag` feeds both runners' `resolve_out_csv`. Every
flag has a behavioural probe in `tests/test_bench_flags.cpp`; add one with the
next flag, and check it fails when the apply step is neutered.

Neither header has a CMake target — runners reach them through the
`target_include_directories(<target> PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})` line
each already carries, so a new runner target that omits that line will not
compile `<benchmarks/common/...>`. `cbls_tests` carries the same line for
`${CMAKE_SOURCE_DIR}`, which is what lets a Catch2 test include them without a
runner `main()`.

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
| 1 | `mipfeas` | OR-Tools CP-SAT `num_violation_ls` worker **only** | head-to-head correctness *and* performance against the reference implementation of the same jump-based algorithm | #87, #106 |
| 2 | `minlplib` | SCIP (nonlinear) | performance on non-convex MINLP — the regime CP-SAT's LS worker cannot express | #87 |
| 3 | `uc-chped` | Pedroso 2014 | a good result on a published unit-commitment formulation | #25, #92, #131 |

**Priority 1 is deliberately not a MIP shootout.** Compare only against CP-SAT's
`num_violation_ls` worker (same algorithm, different implementation). Do **not**
compare against CP-SAT's default full portfolio, Xpress, Gurobi or CPLEX — see
epic #87 for why that framing is rejected.

Nothing outside this table is a benchmark. `setcover` is the scoped coverage
check the `Set` variable type had been missing: ten small OR-Library
set-covering instances, run under both a `Set` and a Bool encoding, whose result
is a documented limitation rather than a comparative claim (see
`benchmarks/instances/setcover/README.md`). Don't grow it into an epic.

Engine-wide (cross-cutting) work is tracked under epic #24.

### Structured variables: the claim is currently unsupported

Do not write, in docs or issues or commit messages, that List variables are
validated on a published formulation. The evidence position is:

- **`Set`** — measured and **negative**. The `Set` encoding lands at
  8.6-11.0x the proven optimum where the same data in Bools is within 9-20%.
- **`List`** — **no evidence either way.** No benchmark in the tree uses a
  `List` variable.

The mechanical cause of the `Set` result is generic and applies to `List`
equally, and it has **two independent halves** — fixing either alone changes
nothing:

1. `FeasibilityJump::jumpable` (`src/feasibility_jump.cpp`) is a deliberate
   Bool/Int/Float whitelist, so no structured variable ever gets a jump-table
   entry regardless of its derivative. Its comment says the whitelist is
   intentional.
2. `local_derivative` (`src/dag.cpp`) returns `0.0` for *every* structural op —
   `At`, `Count`, `Lambda`, `PairLambda` — so there is no AD signal to build a
   jump value from either. (`setcover`'s `Set` model reads through `Lambda`.)

What is left is `set_moves`/`list_moves` drawing uniformly at random, where the
scalar path has FJ's jump table and best-of-N scan-set sampling to steer it.
Note the contrast is between the two *paths*, not the individual generators —
`int_rand` and `float_perturb` are themselves uniform. **Cost-aware
structural move selection is the prerequisite** for any renewed structured-variable
claim, and `setcover` is its ready-made A/B harness — same instances, both
encodings, a published baseline. Fix the guidance, re-run setcover, and only then
consider whether a new List/Set benchmark is worth building. Adding one first
just reproduces the negative result in a new domain.

### Benchmark workflow

Each benchmark session must follow these steps in order:

0. **Plan first** — Read the benchmark's epic (linked above) and its open sub-issues, plus `docs/architecture.md` (solver internals). Investigate what already exists for your benchmark (data, reference solver, model code). Propose an approach and wait for approval before implementing. This is the Plan phase — do not skip it.

1. **Download/prepare instance data** — Write a download/generation script into `benchmarks/instances/{name}/`. Follow the pattern of `benchmarks/instances/uc-chped/` (Python data definitions + JSONL serialization). Source data from public benchmark libraries, competition archives, or papers.

2. **Find a reference solver** — Implement a reference solver in `benchmarks/{name}/reference_solve.py` using SCIP (PySCIPOpt) or another open-source solver if SCIP can't handle the formulation. Follow the pattern in `benchmarks/chped/reference_solve.py`.

3. **Collect best-known results** — Find published results from papers/competitions. Record them per **Measuring and testing engine changes** above: a README with the engine commit named in the text, or a `comparison.csv` (columns: instance, method, objective, gap, source) if a test will read it.

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
