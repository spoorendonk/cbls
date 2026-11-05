# Git hooks, gates and review receipts

Reference for the hooks in `.githooks/` and the gates they enforce. CLAUDE.md keeps the short form (**Git Hooks**, **Self-Review**); this is the full account, read when a hook refuses something or when editing a hook, a `.clang-tidy` or the Python gate.

## Git Hooks

The hooks live in **`.githooks/`, tracked in this repo** — that directory is the source, edit it there. Git runs them only once `core.hooksPath` points at that directory, and that setting is per-checkout local config which cannot be committed, so `cmake -B build` sets it (`CBLS_INSTALL_GIT_HOOKS`, default ON). Configuring the project once is all a fresh clone needs; to wire one up by hand, `git config core.hooksPath .githooks`. If `core.hooksPath` already points somewhere else, CMake warns rather than overwriting it.

`core.hooksPath` is shared across every worktree of a checkout, but `.githooks/` is resolved from the working tree — so a branch that does not carry the directory runs **no hooks at all**, silently. Branches are short-lived and cut from an up-to-date main, which carries it.

The three hooks are wcp's (see **wcp's files in this repository**); `gate-lib.sh`, `tidy-probe.sh`, `tidy-probe/` and `tests/` beside them are this project's own and are still exercised by ctest (`gate_lib_shell_test`, `clang_tidy_gate_probe`).

- `pre-commit` — **checks, never rewrites**: `ruff format --check .`, `ruff check .` and `mypy .` (strict, from `pyproject.toml`) over the whole tree, each run as `.venv/bin/ruff` and `.venv/bin/mypy` (this checkout's, then, from a linked worktree, the main checkout's; never `uv run`, never PATH's), and `clang-format --dry-run -Werror` over every tracked C/C++ file, preferring the pinned tools in `.venv/bin`. Any finding blocks the commit. So does a ruff or mypy that is missing from the venv or off its `pyproject.toml` pin, naming the `uv sync` to run; a missing or wrong-major clang-format is reported on stderr and skipped. It runs **no tests**, and **no clang-tidy**. Why not `uv run` (#371): it syncs first, and after any `pyproject.toml` change that rebuilds the editable C++ extension, a full C++ build inside the hook, outside the build lock; when that build failed, every later commit retried it and ruff and mypy were skipped. The venv is kept current by `uv sync` (CLAUDE.md's **Build & Test**), not by the hook. `gate_lib_shell_test` pins this behaviour.
- `commit-msg` — Conventional Commits, subject at most 72 characters. Unlike the hook it replaced, it does **not** pass `fixup!`/`squash!`/`amend!` subjects, so `git commit --fixup` is refused.
- `pre-push` — refuses a merge commit pushed to `main` and a tip without its review receipt, then runs the ```clean, ```build and ```test fences from CLAUDE.md's `## Build & Test` as a local CI in a fresh worktree outside the checkout (skipped for a documentation-only push), each step timed. It runs on the CPUs the push inherits, as the host sets them (`wcp build`'s CPU pinning, `taskset`); on a host that does not pin pushes, the optional override `git config hooks.localCiCpus 0-3` pins it to those CPUs instead. This local CI is the only gate: there is no GitHub Actions workflow (an unused one was deleted, #246).

**What the hooks before wcp's enforced and these do not** (replaced 2026-10-14; the old scripts are in git history): the `ctest -LE slow` / pytest run at commit; auto-formatting and safe clang-tidy fixes at commit; at push, the **clang-tidy hard block** over the changed files plus a canary per config directory, `tidy-probe.sh` whenever a `.clang-tidy`, a `CMakeLists.txt` or the probe changed, the **block on any first-party compiler warning** in the gated build's log (configured with `CBLS_GATE_PROBE_IN_ALL=1` as its control), the advisory ruff-complexity/shellcheck/mypy pass, and the **Python gate over every tracked `.py`** (now `pre-commit`'s, over the tree). Until those are restored — a ```lint fence under CLAUDE.md's `## Build & Test` is where wcp's pre-push would run them — the clang-tidy and compiler-warning policy below is **policy you check by hand** before pushing (`.venv/bin/clang-tidy -p build <files>`, and the build log), not something a hook enforces.

  The tree is held at **zero** clang-tidy warnings. Every
  `bugprone-*`/`modernize-*`/`performance-*`/`readability-*` check is enabled
  except six structural exemptions, each off because the *check* is wrong for
  this codebase, with the measurement written beside it in `.clang-tidy` (the
  ratchet list that got the tree there is empty, #121). So a warning is new
  code's doing: **fix it — do not add the check to the disabled list**; a
  seventh exemption needs a structural cause plus a measurement. `.clang-tidy`
  is the source of truth and carries its own sweep recipe.

  In-tree suppression is **sanctioned only where the reason is written at the
  site**, and the tree has exactly five, across two checks —
  `grep -rn NOLINT src/ include/ benchmarks/ tests/ python/` should return
  nothing else, and a sixth needs the same standard as `.clang-tidy`'s six
  exemptions. That grep prints **six** lines, not five: one is `mps_reader.cpp`
  naming the directive in prose in a comment, not applying it. Count the
  directives, not the matches. Four are `readability-function-cognitive-complexity`: on
  `src/dag.cpp`'s two 35-case `NodeOp` dispatch tables, where the score counts a
  table a human reads as one unit and any split would have to stay inlinable on
  the delta-evaluation and reverse-mode-AD hot paths; and on the two
  `src/io/mps_reader.cpp` functions that the vendored port-the-diff-upstream
  contract keeps whole. That is the route CLAUDE.md's **Complexity** describes — split
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

  Because both config hazards fail open, check clang-tidy's output for
  `(warning|error):` **case-insensitively** and treat any **nonzero exit** as a
  finding: a misconfigured directory reports `Error: no checks enabled.` with a
  capital E. When a push changes a `.clang-tidy`, lint a file in each config
  directory too, since such a change touches no source file of its own.

  **The probe.** `clang-diagnostic-*` is named **after** the wildcards in
  `Checks:` (a leading `-*` switches it off, #171), every first-party target gets `-Wall -Wextra`
  through `cbls_enable_warnings()` in the root `CMakeLists.txt` (a new target
  needs the call), and **`.githooks/tidy-probe.sh`** proves it. It lints a
  deliberately dirty TU, `.githooks/tidy-probe/incomplete_switch.cpp`, copied
  into every config directory of a mirror of the configs, and fails unless its
  incomplete `enum class` switch, unused local and unused parameter are all
  reported. It also fails on any first-party compile command missing the flags
  or cancelling them with `-Wno-*`, `-w` or `-Werror` (flags from the
  environment are exempt: the rule polices the project, not the machine). ctest
  runs it as `clang_tidy_gate_probe` (Skipped without the pinned clang-tidy
  in this tree's or the main checkout's `.venv`), and `gate_lib_shell_test`
  tests the text filters in `.githooks/gate-lib.sh` that the old pre-push
  gates used. Never add a `default:` to a `NodeOp`
  switch: it is what would hide the case.

  **No `-Werror`**: with it in `compile_commands.json` every clang-diagnostic
  finding becomes a compiler error and clang-tidy exits 1. GCC, the build
  compiler, has warnings clang lacks (two `-Wrange-loop-construct` at #171), so
  read the build log for first-party warnings instead. A GCC upgrade can then
  surface new ones without breaking anyone's build, which `-Werror` would.

`pre-push` refuses a push whose tip has no review receipt, `refs/reviews/<sha>`, written by a line of plain git (**Self-Review** below), so the check works in every checkout. It proves a review was recorded, not that it was done well; see **Agent Self-Review**.

Gating lives in git hooks only — `.claude/settings.json` carries no `hooks` block, and none should be added. A `PostToolUse` formatter cannot see which file was edited, so it silently formats nothing; formatting belongs at commit time. Don't hand-tune formatting.

These conventions therefore rest on you rather than on a tool: zero clang-tidy warnings and zero first-party compiler warnings (above), branch only from main (**Git Workflow** in `docs/development.md`), never invoke `python`/`pytest`/`mypy` outside `.venv/bin/` (a `uv sync` venv has no `pip`; change it with `uv sync`; CLAUDE.md's **Python**), and bounded searches (CLAUDE.md's **Code Navigation**).

`.claude/` is gitignored local agent tooling (settings, statusline). Nothing in it is part of the repo.

## The Python gate


`pre-commit` runs `ruff format --check .`, `ruff check .` and `mypy .` (strict,
from `pyproject.toml`) over **the whole tree** as a **hard block**. There is no
exclusion beyond `[tool.mypy] exclude = ['^build/']`, which keeps mypy out of the
vendored Python FetchContent unpacks under `build/_deps`; a tree that genuinely
cannot be clean gets its own entry there, with the reason written beside it,
never a silent widening. (Until the move to wcp's hooks this gate lived in
`pre-push`, over every tracked `.py`, with a `PY_GATE_EXCLUDE` that was empty.)
Because they block, `ruff` and `mypy` are **pinned exactly** in
`pyproject.toml`, on the same argument the clang tools are: a floating version
turns "these files are at zero" into "at zero on whichever version this checkout
resolved".

**Two naming waivers exist, scoped per file and per rule** in
`pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]`: `N803`/`N806` on
`benchmarks/chped/reference_solve.py` and `N806` on
`benchmarks/instances/uc-chped/data.py`. `P`, `A`, `T` and `base_T` are the
published unit-commitment formulation's own symbols, and those solvers exist to
be audited line-by-line against the paper whose numbers they reproduce. Same
argument, and the same shape, as `src/io/.clang-tidy` waiving the naming check
for the vendored readers. A third waiver needs the same standard. A directory
exclusion is no substitute: `mypy` follows imports, so an excluded module that a
gated file imports is checked anyway (#154, #155).

## Self-Review

**Code you produced gets reviewed before it lands on main.** A fresh, cold
subagent runs `/code-review` on your unpushed commits, and you act on what it
finds; the review is not a formality you
report having done. The pre-push hook is the backstop, not the review.

**Then record the review, and push it with the commit**, or `pre-push` will
refuse the push:

```sh
git update-ref "refs/reviews/$(git rev-parse HEAD)" HEAD
git push --atomic origin HEAD "refs/reviews/$(git rev-parse HEAD)"
```

The receipt is a ref named by the reviewed commit, `refs/reviews/<sha>`, and it
travels with the push, so anyone can check after the fact that a commit on the
trunk has one (this repository has no CI that does). `--atomic` lands the
branch and its receipt together or not at all.

Plain git, on purpose. No wcp, no harness, nothing installed — so a reviewer on
any machine can write it, and a clone that has never heard of wcp is still able
to record a review and push. Every worktree of a clone sees the same receipts,
and each names one commit, so a review counts for that commit wherever it is
checked out and for nothing else.

**What it means, and what it does not.** It records that a review happened at
this commit — not what was reviewed. It carries exactly the trust of whoever ran
it, and it is a line anybody can write by hand; what it buys is that nobody can
push work no one has looked at *without noticing they are doing it*.

A later commit has no receipt, and the next push is refused again. That is
deliberate: a new commit is work nobody reviewed, including the commit that
fixes what the review found.

**The fix after the last round.** Reviews stop after two rounds, so the commit
that fixes the second round's findings gets no third review. That round's
reviewer reads the fix commit alone (`git show <sha>`) against its own
findings, and the receipt is recorded for the fix once it confirms they are
addressed. It looks for nothing new. What it does not confirm is taken out of
the fix, which makes a new commit: the same reviewer confirms that one before
its receipt is recorded, and what was taken out becomes its own issue.

**Record it only when the tree is clean.** With nothing new committed, a review
falls back to reading uncommitted changes — so a record written then names a
commit that does not contain what was read, and the push it clears is of work
nobody reviewed. Commit first, then review, then record. Untracked scratch files
do not matter; staged and unstaged changes to tracked files do.

## Agent Self-Review

**Code produced by any agent or agent team must be reviewed before it can merge to main** — solo work by one cold reviewer running `/code-review` on the commits, a team run by one cold reviewer per issue (below). No subagent returns unreviewed work; no orchestrator merges unreviewed work. This applies to every agent team, not just the parallel-issue workflow.

**One reviewer per issue** (user decision, 2026-10-03). In a team run, each
issue gets exactly **one** cold reviewer: a separate agent that did not write
the code, reading the issue's acceptance criteria literally. That reviewer *is*
the issue's review — the implementer does not also fan out further
reviewers, and the orchestrator does not add a second reviewer per issue. The
cold review runs **at most two rounds**: review, a fix round, re-review. A
finding still open after round two goes to the orchestrator to fix or to file;
it does not start a third round. The final combined review of the merged result
(step 6 of **Parallel Issue Workflow** in `docs/development.md`) is a separate single pass over the whole batch, not a per-issue
reviewer.

## wcp's files in this repository

The hooks in `.githooks/` and the top line of CLAUDE.md were written by wcp
(`https://github.com/spoorendonk/wcp`), a command-line tool installed once per
machine. What matters here:

- **The hooks call no wcp.** They are plain shell scripts committed like any
  other file, so a clone without wcp is still gated once `core.hooksPath` is set
  (the first `cmake -B build` sets it, **Git Hooks** above). A hook that shelled
  out to a missing tool would exit 127, which git reads as a refusal.
- **They are ours to edit**, as is CLAUDE.md's prose. Each hook's `# wcp:` line
  and CLAUDE.md's top comment record the bytes wcp wrote; leave them in place so
  `wcp doctor` can tell a local edit from a newer wcp version.
- **Run `wcp doctor`** after pulling or updating wcp and before editing the
  hooks or CLAUDE.md. It changes nothing; `wcp --help` lists the rest.
- **If `wcp` is not on PATH**, say so rather than installing it unasked.
