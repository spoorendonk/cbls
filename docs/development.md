# Development reference

What CLAUDE.md points at for building, testing and working in this repository: the reasoning behind the gated fences, the test roster and where its counts are pinned, git workflow and agent-team runs.

## Build & Test, in detail

The fences themselves stay in CLAUDE.md's `## Build & Test`, because `.githooks/pre-push` reads them from there. This is why they look the way they do.

**No `-DCMAKE_BUILD_TYPE` here on purpose.** `CMakeLists.txt` defaults it to
Release when the caller sets none, so the ```build fence (pre-push's local CI) and a plain `cmake -B build`
both gate the same binaries from one place. An explicit `-DCMAKE_BUILD_TYPE=Debug`
still overrides it. The suite is mostly real solver runs, so the type is not
cosmetic: the full `ctest` is ~490s at the old empty default and ~25s at
Release (both re-measured 2026-09-26 at 566 tests, `-j12`, on an idle machine;
the gate runs at `-j$(nproc)` over the CPUs the push inherits: at most 4
where the host pins pushes, or with the `hooks.localCiCpus` override
(CLAUDE.md's **Build & Test**), so expect its wall-clock to be longer than
the `-j12` figures in this file).

`CMakeLists.txt` also picks up `ccache` as a compiler launcher when the machine
has it (`apt install ccache`), which matters because pre-push's ```clean fence is
`rm -rf build`. Rebuilding the same path with a warm cache is **0.9s** against
**23s** cold. It is optional — a system package cannot be committed, so a machine
without it just builds normally. pre-push builds in a worktree under
`~/.cache/local-ci`, a different path for every push, so ccache only hits there
with `base_dir` set (e.g. `ccache -o base_dir=$HOME`); without it each push is a
cold build. `git config hooks.localCiWorktree <path>` instead keeps one warm
worktree between pushes and skips its ```clean fence (`.githooks/pre-push`).

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
skips every test that imports `cbls` when its compiled `cbls._cbls_core` is missing, so a build
without the flag would leave 345 binding tests silently unrun.
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
clone that runs no gate. It installs the tools and dependencies only, not
`cbls` itself (`[tool.uv] package = false`): the tests import the gated build's
`build/python/cbls`, and the package is installed only as a wheel (#463). A
binding change regenerates the committed stub `python/cbls/_cbls_core.pyi`: the
command is in `tests/python/test_stub.py`, whose drift test fails otherwise. `uv sync`
writes an untracked `uv.lock`, which `.gitignore` ignores. No symlink to another clone's venv: it points outside the session's
working directory, so every command touching `.venv` raises a permission prompt,
and it forces one set of extras on every clone that shares it.

**Only a subagent git worktree under this checkout** borrows the checkout's venv
by symlink, since it has none and is torn down with the run:

```bash
ln -sfn /path/to/this/checkout/.venv .venv && .venv/bin/pytest --tb=short -q; rm -f .venv
```


### Fast vs. slow tests

The C++ suite is **901 ctest tests**: 895 Catch2 ones over **891 `TEST_CASE`s**
— 885 registered by `catch_discover_tests` plus **10 registered by hand**, the 6
`[timing]` cases and the four hang guards `hang_guard_iteration_only_portfolio`,
`hang_guard_novelty_zero_move_batch`, `hang_guard_nonpositive_batch_iterations`
and `hang_guard_nl_placement_counts`, each hand-registered *as well as*
discovered (a hang only reports through a `TIMEOUT`, but all four are cheap enough
to belong in the fast set), so each of those `TEST_CASE`s accounts for two
ctest tests — plus **2 shell tests that are not Catch2 at all**,
`clang_tidy_gate_probe` and `gate_lib_shell_test`, registered in the root
`CMakeLists.txt` (they pin the clang-tidy gate and the hook filters, #171) —
plus **2 consumer tests**, `consumer_find_package` and
`consumer_add_subdirectory`, also registered there with `LABELS "slow"`: each
builds `tests/consumer/` against cbls as a downstream project would (#447) —
plus **1 wheel test**, `wheel_smoke`, registered in `python/CMakeLists.txt` with
`LABELS "slow"`: it builds the wheel, installs it into a fresh venv and imports
and solves there (#463, `tests/wheel/smoke.py`). Like the consumer tests it is a
ctest script, not pytest: a test that builds a whole project belongs in the
`slow` label, which pytest has no counterpart for — plus **1 docs test**,
`docs_build` (label `docs`), registered in the root `CMakeLists.txt` when
configure finds doxygen and a Python with the `docs` group (**Documentation
site** below). Of the 885,
**4 carry the
Catch2 `[slow]` tag** — the UC-CHPED benchmark solves. `tests/CMakeLists.txt` discovers them in a second
`catch_discover_tests` call with `LABELS "slow"`, so:

- `ctest -LE slow` — the other 889 tests, ~12s with `-j`. The fast inner loop; run it yourself before committing (the hook no longer does).
- `ctest` — everything. This is what **pre-push**'s local CI runs.
- `ctest -L timing` — 6 tests: `timing_structural_batch_deadline`, the three
  `timing_throughput_*` floors added for #125,
  `timing_inner_solver_hook_deadline` (#191's budget contract) and
  `timing_uc_hook_ils_deadline` (the UC hook's ILS deadline, #461). Each is
  registered by an explicit `add_test` (naming its Catch2 test case) so it can carry a `TIMEOUT` and be
  quarantined individually. Don't add tests to this class without a concrete
  reason.

**The Catch2 `[slow]` tag and the ctest `slow` label are not the same set here**,
and both numbers appear above: 4 tests carry the *tag*, 12 carry the *label*
(`ctest -N -L slow` says 12): the 4 tagged ones, `wheel_smoke`, the 2 consumer tests and 5
`[timing]` ones. The three throughput floors and #191's and
#461's deadline tests are tagged `[timing]`
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
4. the `.venv/bin/pytest` line in `README.md` for the Python side (1863 tests, 345
   of them binding tests, echoed in prose by `pyproject.toml`),
5. the `-LE slow` guidance and the ~25s/~490s figures in `docs/profiling.md`.
   Note its "302/302 green" sanitizer line is a **dated record of one run at a
   named commit**, not a current count — it says so inline. Leave it alone
   apart from the parenthetical restating the current fast-set size.
6. the binding count in **Build & Test, in detail** above, in the paragraph explaining
   why the gated build turns `CBLS_BUILD_PYTHON` on ("345 binding tests silently
   unrun"). It is in this file, but not in this section, so a search that stops
   at this enumeration misses it.

Update all six in the same commit as any change to the test roster, and
re-derive every number from `ctest -N` / `pytest --collect-only` rather than
incrementing the one you find.

All timings here are **Release**, the build type `CMakeLists.txt` defaults to,
and `[slow]` is earned at Release: the 4 tagged tests are the only ones where
the fast/slow split pays.

Measure both ways before tagging. These tests run under `-j$(nproc)`, and
contention inflates a solve (by 1.4-1.7x at `-j12`: `UC-CHPED 40-unit 1-period
feasibility` was 9.95s alone and 17.1s contended). Tag if it exceeds ~10s on
**either** basis. Don't tag one just to get a green commit — pre-push will still run it.

### Off Linux (macOS)

Linux with libstdc++ is the platform that gates; on it nothing below skips. The
fences also pass on macOS (libc++, arm64), with tests that cannot hold there
skipped, not rewritten (#355):

- C++ tests pinned to a seeded trajectory call `skip_unless_libstdcxx_rng()`
  (`tests/libstdcxx_rng.h`): `<random>`'s distributions are
  implementation-defined, and libc++ draws differently. A new test that pins a
  seeded trajectory, fingerprint or seed-built scenario calls it too.
- Python tests of `ulimit -v`, `taskset`, `/proc` or a seeded trajectory carry
  `pytest.mark.skipif(sys.platform != "linux", ...)`.
- `os.sched_getaffinity` is Linux-only in typeshed, so mypy fails off Linux on
  a bare call: use `benchmarks.common.jobs.allowed_cpus()`.

Skips change no roster count: a skipped test is still listed.

The benchmark runners themselves stay Linux-only, as runs belong on Linux
machines: a memory cap (`--mem-limit-gb`, `ulimit -v`) fails every job on
macOS, a CPU pin needs `taskset`, and off Linux `allowed_cpus()` is every CPU, not a measured affinity.

### Documentation site

The site at <https://spoorendonk.dk/cbls/> is built with Sphinx (MyST, Furo)
from `docs/` (#451): `docs/index.md` includes the README, beside the results,
a C++ reference (Doxygen XML of the supported headers, `<cbls/cbls.h>`, what
it includes and `<cbls/io_nl.h>`, read from `cbls.h` into `CPP_HEADERS` in
`docs/conf.py`, rendered by Breathe) and a Python reference (sphinx-autoapi, read statically
from `python/cbls/` and the committed `_cbls_core.pyi`, so nothing is
compiled). It also writes `llms.txt` and `llms-full.txt`. The other files in
`docs/` are for working on cbls: `exclude_patterns` keeps them off the site,
so a site page or the README links them by absolute GitHub URL, as it does any
other repository file. Build it with `doxygen` on PATH:

```bash
uv run --only-group docs sphinx-build -W -b html docs build/docs/html
```

`-W` turns warnings into errors. The `docs` dependency group holds the Sphinx
packages; `default-groups` in `pyproject.toml` makes the usual `uv sync` keep it.

The gate keeps it warning-free: the ctest test `docs_build` (label `docs`) runs
the same build into the build directory, when configure finds `doxygen` (on
macOS `brew install doxygen`) and a Python with the `docs` group; without
either, configure warns and the test is not registered, and the roster above
is one short (900, and 888 in the fast set). A push that changes
the README or a file under `docs/` runs the gate. The `docs` workflow
(`.github/workflows/docs.yml`) builds and deploys the site to GitHub Pages on a
push to `main` that changes what the site is built from (`docs/`, the README,
`CMakeLists.txt`, `include/cbls/`, `python/cbls/`, `pyproject.toml`, the
workflow), on a release tag (`v[0-9]*`), or by hand (`gh workflow run
docs.yml`). It runs no tests, and never on another branch or a pull request. A
tag deploys only because the `github-pages` environment has a `v*` tag rule
beside its default-branch one (set when Pages was enabled, #466); without it
GitHub refuses the deploy. The sidebar and `llms.txt`
name the version from `git describe` over the release tags: the tag, or
`main (after v0.1.0, abc1234)`, or `main (abc1234)` before the first tag.

The landing page's figure, the sidebar logo and the favicon are committed SVGs
in `docs/figures/`, beside `routes-card.png`, the 1200×627 image of the Open
Graph tags (`sphinxext-opengraph`) that gives a link to the site its preview
card on LinkedIn and elsewhere, which render no SVG. `docs/figures/make_routes.py`
draws them all from a CVRP solve (`X-n115-k10`, the model of
`benchmarks/cvrp/cvrp_model.h`) through the Python API, single-threaded under an
iteration cap, so one build on one platform redraws the same files; the docs
build does not run it. The committed files were drawn at engine commit
`3ef800c9` on macOS. Redraw them after an engine change that moves those
routes, after the build fence, and update that commit here:
`PYTHONPATH=build/python .venv/bin/python docs/figures/make_routes.py`. It prints
the cost, its gap to the best known and the route count; the captions in
`docs/index.md` state none of them. The script reads the accent colours from
`docs/conf.py`.

## LSP

### C++


Install `clangd-lsp@claude-plugins-official` plus `clangd` itself (`apt install clangd` or from LLVM). `.clangd` points at `build/compile_commands.json`, produced by `CMAKE_EXPORT_COMPILE_COMMANDS ON`. Prefer `LSP` tool queries (`goToDefinition`, `hover`, `documentSymbol`) over `Read` for symbol questions.

### Python


Install `pyright-lsp@claude-plugins-official`. Pyright reads `[tool.mypy]` and project layout from `pyproject.toml` — no extra config needed.

## Git Workflow

Trunk-based development with linear history on main. Commit directly to main and push when local gates pass.

Feature branches are optional for larger changes:
- Always branch from main. Run `git checkout main && git pull` first.
- Never branch from another feature branch, with one exception: serialized
  issues under **Parallel Issue Workflow**, where the second issue's branch is
  cut from the first's so the pair lands as one fast-forward.
- Keep branches short-lived; rebase or squash merge — no merge commits on main.

After a successful push:
- **Close any gh issue the work resolved**: `gh issue close <num> -c "<one-line note>"`. Do this for every issue covered by the push.
- **Delete the feature branch** if one was used: `git branch -d <branch>` locally, plus `git push origin --delete <branch>` if it was pushed. Don't leave stale branches behind.

## Writing Issues

Issues get picked up later in fresh sessions, often by a different agent with no access to the author's machine. Write them to be picked up cold:

- **Self-contained.** Body must carry all needed context: problem, motivation, acceptance criteria, repro steps. Don't assume the reader has the current conversation.
- **No local references.** No local file paths, local repo paths, or machine-specific locations (`/home/user/...`, `~/code/foo/bar.py`, "see my other checkout"). Dead links in a fresh session.
- **Prefer stable external links.** GitHub permalinks, paper URLs, RFCs, official docs.
- **Name code by repository path, not machine path.** `src/search.cpp` or a GitHub permalink is stable; a checkout's absolute path is not.

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
   result (not a multi-agent fan-out), and only then
   runs `git push origin main`. No pushes — of main or of a feature branch —
   happen before that final review. Then close the issues the push resolved and
   delete the branches and worktrees, per **Git Workflow**.

If the harness denies the merge or the push, that is a permission layer, not a
review finding: say so plainly, hand over the exact commands, and do not report
the work as shipped. Committed-on-a-branch is not pushed, and the final report
must not say otherwise.

A subagent worktree may READ sibling worktrees to understand patterns, but must never WRITE to one or to its git branch. A fresh worktree has no `.venv`; see **Build & Test, in detail** above for the symlink it needs before building or testing (a worktree only — a full clone makes its own).

## Conventions: the reasons

CLAUDE.md states the conventions; this section keeps only the reasons it
leaves out.

- **`src/io/` naming.** `mps_reader.cpp` and `solu_reader.cpp` are vendored
  from `spoorendonk/mipx` and stay diff-clean, so `src/io/.clang-tidy` disables
  the naming check for the directory. The native adapters there
  (`mps_to_model.cpp`, `nl_reader.cpp`, `nl_to_model.cpp`) lose naming coverage
  as a side effect: watch new code in that directory.
- **`InheritParentConfig: true` in `src/io/.clang-tidy`.** Without it
  clang-tidy *replaces* the root `Checks:` instead of appending to it, leaving
  only `clang-diagnostic-*`: the directory runs no registered check and
  clang-tidy exits 1 with `Error: no checks enabled.` Verify an edit to either
  `.clang-tidy` with `cd src/io && ../../.venv/bin/clang-tidy --dump-config`.
  Like the root file's folded-scalar hazard (`docs/git-hooks.md`), it fails
  open.
- **Validate Python-supplied indices where they are handed over.** The hot
  paths index by constraint or variable id without bounds tests, and Python
  supplies both with no type system in the way. One review pass (#156) found
  three crashes of this shape: `ModelState.elements`,
  `ViolationManager.weights`, and child handles into every builder. Check in a
  `def_prop_rw` setter or at node creation, not per read on a hot path, and pin
  it with a child-process test (`tests/python/test_model_handles.py`), since the
  unguarded state crashes the interpreter rather than failing a test.
- **No `reference_internal` to an element of a growable container.**
  `keep_alive` keeps the *owner* alive, not the element: before `close()`
  every variable builder appends to the model's variable array and can
  reallocate it, so a `m.var_mut(i)` held across one writes into freed heap and
  the write is silently lost (#167). `VariableRef`, bound as `cbls.Variable`,
  resolves (owner, index) on every access.
- **`bugprone-unchecked-optional-access` and `REQUIRE`.** The check cannot see
  through Catch2's expression templates. In
  `if (!o.has_value()) { FAIL("..."); return; }` the `return` is what makes the
  flow analysis see the guard, even though `FAIL` throws.
- **Deadlock-prone Python tests run in a child process.** No in-process
  deadline can report a deadlock: `pytest-timeout`'s `thread` method needs the
  GIL the test is starving, and its `signal` method only takes effect at the
  next bytecode in a main thread parked in a C++ `join()`. `tests/python/test_pool.py`
  is the pattern.
- **Tests spawning `git` strip `GIT_*`.** pre-commit exports `GIT_DIR` and
  `GIT_INDEX_FILE`; a `git init` in a tmp dir once re-initialised the real
  repository (`core.bare = true` for every worktree, #160).
- **Terse output is not configured anywhere.** `pyproject.toml` sets
  `testpaths`, ruff's rules and mypy strict, but no `addopts`; pass
  `ctest --progress` and pytest `--tb=short -q` yourself.
- **Comment share.** `.venv/bin/python tools/comment_share.py [--by-file AREA]`
  prints comment lines over non-blank lines per area; `include/cbls` stood at
  66% when the comment rule landed (#372, #394).
