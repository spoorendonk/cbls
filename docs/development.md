# Development reference

What CLAUDE.md points at for building, testing and working in this repository: the reasoning behind the gated fences, the test roster and where its counts are pinned, git workflow and agent-team runs.

## Build & Test, in detail

The fences themselves stay in CLAUDE.md's `## Build & Test`, because `.githooks/pre-push` reads them from there. This is why they look the way they do.

**No `-DCMAKE_BUILD_TYPE` here on purpose.** `CMakeLists.txt` defaults it to
Release when the caller sets none, so the ```build fence (pre-push's local CI) and a plain `cmake -B build`
both gate the same binaries from one place. An explicit `-DCMAKE_BUILD_TYPE=Debug`
still overrides it. The suite is mostly real solver runs, so the type is not
cosmetic: the full `ctest` is ~490s at the old empty default and ~25s at
Release (both re-measured 2026-09-26 at 566 tests, `-j12`, on an idle machine).

`CMakeLists.txt` also picks up `ccache` as a compiler launcher when the machine
has it (`apt install ccache`), which matters because pre-push's ```clean fence is
`rm -rf build`. Rebuilding the same path with a warm cache is **0.9s** against
**23s** cold. It is optional — a system package cannot be committed, so a machine
without it just builds normally. pre-push builds in a worktree under
`~/.cache/local-ci`, a different path for every push, so ccache only hits there
with `base_dir` set (e.g. `ccache -o base_dir=$HOME`); without it each push is a
cold build.

**Why the fences look for `python` on PATH when `.venv/` is missing.** `.githooks/pre-push`
runs these fences (`clean`, `build`, `test`) as a local CI in a fresh worktree
*outside* this checkout, which has no `.venv/` of its own; it puts this
checkout's `.venv/bin` first on PATH instead. So each fence uses
`$PWD/.venv/bin/python` when it exists (the checkout itself) and otherwise the
`python` PATH resolves to (the hook's worktree), and runs pytest as
`python -m pytest` from that interpreter. `|| [ $? -eq 5 ]` keeps pytest's
"nothing collected" exit a pass, and is written that way because the hook
evaluates each fence under `set -e`. Keep the fences runnable in both places.

**The gated build turns the Python bindings on, and the gated test run requires
them.** `CBLS_BUILD_PYTHON` defaults to `OFF` and `tests/python/conftest.py`
skips every test that imports `_cbls_core` when the module is missing, so a build
without the flag would leave 255 binding tests silently unrun.
`CBLS_REQUIRE_BINDINGS=1` turns that skip into a hard error. Bindings cost ~2.4s
of build and ~6s of pytest against a suite that already spends ~25s in `ctest` —
always build them. The cost argument is the weaker one: the reason is that
without the flag those binding tests do not run at all.

**`CBLS_REQUIRE_BENCHMARKS=1` is the same rule for the `benchmarks` extra.**
`ortools` and `pyscipopt` are not in `[dev]`, so a venv synced with
`--extra dev` alone leaves every test of the CP-SAT
baseline and of the independent solution verifier on a module-level
`pytest.importorskip` — dozens of tests, including the one that is #138's whole
acceptance criterion, disappearing into a green summary. Sync the extra
(`uv sync --extra dev --extra benchmarks`); the flag makes the skip a hard
error instead of a silent one.

`pytest` is **only** in `.venv/` — a bare `pytest` is not on PATH and
`python3 -m pytest` has no module, so always spell out `.venv/bin/pytest`.
The build needs the venv too — the ```build fence locates nanobind through
`$PWD/.venv/bin/python` (falling back to whatever `python` is on PATH, usually
one without nanobind), so a checkout without one builds no bindings and the ```test
fence fails on `CBLS_REQUIRE_BINDINGS=1`.

**A full clone — a `wcp session start` clone included — makes its own venv**,
and never symlinks another checkout's:

```bash
uv sync --extra dev --extra benchmarks
```

`--extra benchmarks` is what the ```test fence requires; leave it out only for a
clone that runs no gate. This builds `cbls` into `.venv` once (under the machine
lock, like any C++ build) and writes an untracked `uv.lock`, which `.gitignore`
ignores. Verified in a fresh clone (#227): the ```build and ```test fences then
pass with the full roster. A symlink into another clone points outside the
session's working directory, so every command touching `.venv` raises a
permission prompt and the session stalls until a human answers; it also forces
one set of extras on every clone that shares it (#226 needed `benchmarks` where
the main checkout's env lacked it).

**Only a subagent git worktree under this checkout** borrows the checkout's venv
by symlink, since it has none and is torn down with the run:

```bash
ln -sfn /path/to/this/checkout/.venv .venv && .venv/bin/pytest --tb=short -q; rm -f .venv
```


### Fast vs. slow tests

The C++ suite is **828 ctest tests**: 826 Catch2 ones over **823 `TEST_CASE`s**
— 818 registered by `catch_discover_tests` plus **8 registered by hand**, the 5
`[timing]` cases and the three hang guards `hang_guard_iteration_only_portfolio`,
`hang_guard_novelty_zero_move_batch` and `hang_guard_nonpositive_batch_iterations`,
each hand-registered *as well as*
discovered (a hang only reports through a `TIMEOUT`, but all three are cheap enough
to belong in the fast set), so each of those `TEST_CASE`s accounts for two
ctest tests — plus **2 shell tests that are not Catch2 at all**,
`clang_tidy_gate_probe` and `gate_lib_shell_test`, registered in the root
`CMakeLists.txt` (they pin the clang-tidy gate and the hook filters, #171). Of the 818,
**6 carry the
Catch2 `[slow]` tag** — the CHPED and UC-CHPED benchmark solves, ~46s of
aggregate (summed per-test) time, which `-j$(nproc)` compresses to a ~25s
wall-clock full run. `tests/CMakeLists.txt` discovers them in a second
`catch_discover_tests` call with `LABELS "slow"`, so:

- `ctest -LE slow` — the other 818 tests, ~12s with `-j`. The fast inner loop; run it yourself before committing (the hook no longer does).
- `ctest` — everything. This is what **pre-push**'s local CI runs.
- `ctest -L timing` — 5 tests: `timing_structural_batch_deadline`, the three
  `timing_throughput_*` floors added for #125, and
  `timing_inner_solver_hook_deadline` (#191's budget contract). Each is
  registered by an explicit `add_test` (naming its Catch2 test case) so it can carry a `TIMEOUT` and be
  quarantined individually. Don't add tests to this class without a concrete
  reason.

**The Catch2 `[slow]` tag and the ctest `slow` label are not the same set here**,
and both numbers appear above: 6 tests carry the *tag*, 10 carry the *label*
(`ctest -N -L slow` says 10). The three throughput floors and #191's hook
deadline test are tagged `[timing]`
— which is what keeps them out of both `catch_discover_tests` calls, since the
second spec is `[slow]~[timing]`; retagging one `[slow]` *instead of* `[timing]`
would get it discovered *and* hand-registered, i.e. run twice (adding `[slow]`
alongside `[timing]` changes nothing) —
and are given `LABELS "timing;slow"` by hand so their wall-clock seconds stay
out of the fast set. Derive each count with the matching command rather
than by arithmetic: `ctest -N`, `ctest -N -L slow`, `ctest -N -LE slow`,
`ctest -N -L timing`.

These counts are hard-coded in **six** places and nothing checks that they
agree:

1. this section,
2. the comment above `catch_discover_tests` in `tests/CMakeLists.txt`,
3. the build section of `README.md`,
4. the `.venv/bin/pytest` line in `README.md` for the Python side (1364 tests, 255
   of them binding tests, echoed in prose by `pyproject.toml` and
   `tests/python/conftest.py`),
5. the `-LE slow` guidance and the ~25s/~490s figures in `docs/profiling.md`.
   Note its "302/302 green" sanitizer line is a **dated record of one run at a
   named commit**, not a current count — it says so inline. Leave it alone
   apart from the parenthetical restating the current fast-set size.
6. the binding count in **Build & Test, in detail** above, in the paragraph explaining
   why the gated build turns `CBLS_BUILD_PYTHON` on ("255 binding tests silently
   unrun"). It is in this file, but not in this section, so a search that stops
   at this enumeration misses it.

Update all six in the same commit as any change to the test roster, and
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

## LSP

### C++


Install `clangd-lsp@claude-plugins-official` plus `clangd` itself (`apt install clangd` or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`. Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`) over `Read` for symbol questions.

### Python


Install `pyright-lsp@claude-plugins-official`. Pyright reads `[tool.mypy]` and project layout from `pyproject.toml` — no extra config needed.

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
4. **One cold reviewer per issue**, at most two rounds, per the Agent
   Self-Review rule (`docs/git-hooks.md`) — a different agent from the implementer. Subagents
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
   branch into local main, has **one** reviewer make a single pass over the merged
   result (not `/review`'s three-agent fan-out), and only then
   runs `git push origin main`. No pushes — of main or of a feature branch —
   happen before that final review. Then close the issues the push resolved and
   delete the branches and worktrees, per **Git Workflow**.

If the harness denies the merge or the push, that is a permission layer, not a
review finding: say so plainly, hand over the exact commands, and do not report
the work as shipped. Committed-on-a-branch is not pushed, and the final report
must not say otherwise.

A subagent worktree may READ sibling worktrees to understand patterns, but must never WRITE to one or to its git branch. A fresh worktree has no `.venv`; see **Build & Test, in detail** above for the symlink it needs before building or testing (a worktree only — a full clone makes its own).

## Conventions in full

CLAUDE.md states these rules in short form; this is the original text, with the reasons.

### C++

- **C++17** (`CMAKE_CXX_STANDARD 17`). `std::expected`, concepts and ranges are *not* available here; don't reach for them.
- Style: Google-based, enforced by `.clang-format` and `.clang-tidy` at the repo root. Both tools are **pinned PyPI wheels in `.venv/`** (`clang-format==22.1.8`, `clang-tidy==22.1.8`), not system packages — `uv sync --extra dev --extra benchmarks` (**Build & Test, in detail** above) installs them, and the hooks prefer `.venv/bin` over `PATH`. The `benchmarks` extra is in that command because the gated test fence requires it. The exact pin keeps formatter and tidy output identical across checkouts.
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

  That directory config **must** carry `InheritParentConfig: true`. Without it clang-tidy *replaces* the root `Checks:` instead of appending to it, leaving only the always-on `clang-diagnostic-*` — the directory then runs no registered check at all and clang-tidy exits 1 with `Error: no checks enabled.` Verify any edit to **either** `.clang-tidy` with `cd src/io && ../../.venv/bin/clang-tidy --dump-config` and confirm the root's check list is actually present. This is the same verification `docs/git-hooks.md` demands for the root file's folded-scalar hazard, and for the same reason: both failures fail open.
- Use `#pragma once` for include guards.
- Minimize includes in headers. Forward-declare where possible.

### Python

- Style: enforced by `ruff` (format + lint) and `mypy --strict`, configured in `pyproject.toml`.
- All functions must have full type annotations (mypy strict mode).
- Use built-in generics (`list[int]`, `dict[str, Any]`) and `|` union syntax.
- Pin dependencies with `>=` lower bounds in `pyproject.toml`. Use `uv`; change the venv with `uv sync`, never `pip`.

### nanobind Bindings

- Bindings live in `python/bindings.cpp` — a single `NB_MODULE(_cbls_core, ...)` built by `nanobind_add_module` in `python/CMakeLists.txt`, gated on `CBLS_BUILD_PYTHON`. Keep binding glue out of `src/`; a new binding source must be added to that call or it is never compiled.
- C++ `camelCase` methods → Python `snake_case` via nanobind. Use `nb::arg("name")` for Python-friendly parameter names.

#### Ownership and Lifetime

- Default: nanobind manages ownership. Use `nb::rv_policy::reference` only when C++ retains ownership and guarantees the object outlives Python references.
- Never return raw pointers without explicit lifetime annotation.
- Prefer returning by value or `std::shared_ptr`. Document ownership on each binding that transfers or shares it.

#### Type Conversions

- Use automatic conversions for standard types (`std::string` ↔ `str`, `std::vector` ↔ `list`).
- Use `nb::ndarray` for NumPy interop — specify dtype and shape constraints.
- **A `def_rw` on a vector the engine indexes unchecked is a segfault, and so is
  a raw `int32_t` handle.** Python supplies both without a type system in the
  way, and the hot paths index by constraint or variable id without bounds
  tests. Three crashes of exactly this shape were found in one review pass
  (#156): `ModelState.elements`, `ViolationManager.weights`, and child handles
  into every builder. Validate where the caller hands the value over — a
  `def_prop_rw` setter that checks the length, or a check at node creation —
  rather than per read on a hot path, and pin it with a child-process test
  (`tests/python/test_model_handles.py`), since the unguarded state crashes the
  interpreter rather than failing a test.
- **Never return `reference_internal` to an element of a container the owner
  can grow.** `keep_alive` keeps the *owner* alive, not the element: before
  `close()` every variable builder appends to the model's variable array and
  can reallocate it,
  so a `m.var_mut(i)` held across one writes into freed heap and the write is
  silently lost (first found through the since-removed `Model.extend`, #167's
  cold review). Return an
  (owner, index) handle that resolves on every access — `VariableRef`, bound as
  `cbls.Variable`, is the pattern — or a copy.

### Testing

C++ tests use **Catch2** (not GoogleTest): files in `tests/`, registered in `tests/CMakeLists.txt`, run via `ctest` (the two shell tests pinning the git-hook gates, in the root `CMakeLists.txt`, are the only non-Catch2 ctest entries). Python tests use **pytest**: `test_<module>.py` under `tests/python/`, `conftest.py` for shared fixtures, `pytest.mark.parametrize` for data-driven cases.

- Name tests descriptively — `returns_optimal_for_feasible_input`, `test_solver_returns_optimal_for_feasible_input`.
- Test nanobind bindings from Python with pytest, not from C++ — the binding is an implementation detail. Include round-trip tests: create in Python → pass to C++ → get result back.
- An `std::optional` guarded by `REQUIRE(o.has_value())` still trips `bugprone-unchecked-optional-access`, which is enabled: the check cannot see through Catch2's expression templates, and `.value()` reads the same way to it. Write `if (!o.has_value()) { FAIL("..."); return; }` instead — the `return` is what makes the flow analysis see the guard, even though `FAIL` throws. Do **not** reach for a `NOLINT` or add the check to `.clang-tidy`'s disabled list; both routes are closed (see `docs/git-hooks.md`).
- A Python test that could **deadlock the interpreter** — anything handing a Python callable to C++ that spawns threads — must run its scenario in a **child process** under a wall-clock timeout. No in-process deadline can report it: `pytest-timeout`'s `thread` method runs its timer on a Python thread that needs the very GIL the test is starving, and its `signal` method only takes effect at the next bytecode in a main thread parked inside a C++ `join()`. `tests/python/test_pool.py` is the pattern — scenarios in a `__main__` block, `subprocess.run(..., timeout=...)` in the test.
- A test that spawns `git` must not inherit a hook's `GIT_*` variables: pre-commit exports `GIT_DIR` and `GIT_INDEX_FILE`, and a `git init` in a tmp dir once re-initialised the real repository (`core.bare = true` for every worktree, #160). `tests/python/conftest.py` strips every `GIT_*` variable before each test; don't work around it.
- Terse output is not configured anywhere — pass the flags. `ctest --progress` collapses the running list, `--tb=short -q` keeps pytest failures short, as CLAUDE.md's `## Build & Test` blocks already do. `pyproject.toml` sets `testpaths`, ruff's rule set and `mypy strict`, but no `addopts`, `pretty` or `output-format`.

### Code Navigation

Prefer narrow queries over full-file reads:

1. **LSP** for symbol questions. `goToDefinition`, `hover`, `documentSymbol`, `workspaceSymbol` answer "where is X / what's its signature" in a few tokens. Use before `Read`.
2. **Grep with `head_limit` (small) + `-n`** to locate lines. Start with `head_limit: 20`; raise only if inconclusive.
3. **Read with `offset`/`limit`** to fetch a slice around the hit. Full-file `Read` is fine for files under ~200 lines or when structure matters.

Know the symbol → LSP. Know a string, not its location → Grep. Full-file Read is the last mile. This is a preference, not a prohibition: shelling out to `grep`/`rg` is fine when the built-in can't do the job (filtering a pipe like `git log | grep`, or a session without the `Grep` tool). What matters is bounding output, not which binary produces it.

### Complexity

When a complexity warning fires, don't extract methods mechanically. Ask: what are the independent responsibilities here? Split along those boundaries. If the function is genuinely complex because the domain is, say why in a comment directly above it and suppress that one warning with `NOLINTNEXTLINE(readability-function-cognitive-complexity)`.

This is the **only** check a `NOLINT` is sanctioned for — plus one pre-existing `bugprone-float-loop-counter` — and only with the reason written at the site; `docs/git-hooks.md` names all five in-tree examples. Every other clang-tidy finding gets fixed, not suppressed, and adding the check to `.clang-tidy`'s disabled list is not an option either.

### Plan Adherence

**Follow the agreed plan.** If you think a plan should change, stop and discuss — don't silently diverge. The same goes outside a written plan: if your current approach isn't working, say so out loud — don't quietly switch strategies. Implement everything specified; don't leave TODO placeholders or stub implementations unless explicitly asked.

### Reference Correctness

When implementing from papers, pseudocode, or open-source references:
- Match the reference algorithm exactly. No early exits, iteration limits, size caps, or "optimization" shortcuts that change behavior.
- Only introduce heuristic approximations when explicitly asked.
- Implement edge cases and special handling — don't simplify them away.
- When in doubt, be faithful to the reference and let tests verify correctness.

### Common Mistakes

- **Don't invent APIs — verify they exist.** Check that functions, flags, and methods actually exist before using them.
- **Don't ignore type errors.** If mypy/clang-tidy flags something, fix the root cause — don't suppress. Silencing either — by disabling a clang-tidy check or by widening `[tool.mypy] exclude` — is the route explicitly closed (see `docs/git-hooks.md`).
- **Don't use deprecated patterns.** Check current docs, not training data.
