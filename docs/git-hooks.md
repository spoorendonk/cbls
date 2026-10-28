# Git hooks, gates and review receipts

Reference for the hooks in `.githooks/` and the gates they enforce. CLAUDE.md keeps the short form (**Git Hooks**, **Self-Review**); this is the full account, read when a hook refuses something or when editing a hook, a `.clang-tidy` or the Python gate.

## Git Hooks

The hooks live in **`.githooks/`, tracked in this repo** — that directory is the source, edit it there. Git runs them only once `core.hooksPath` points at that directory, and that setting is per-checkout local config which cannot be committed, so `cmake -B build` sets it (`CBLS_INSTALL_GIT_HOOKS`, default ON). Configuring the project once is all a fresh clone needs; to wire one up by hand, `git config core.hooksPath .githooks`. If `core.hooksPath` already points somewhere else, CMake warns rather than overwriting it.

`core.hooksPath` is shared across every worktree of a checkout, but `.githooks/` is resolved from the working tree — so a branch that does not carry the directory runs **no hooks at all**, silently. Merge main into any long-lived branch before relying on the gates.

The three hooks are wcp's (see **Hooks Are Files In This Repository**); `gate-lib.sh`, `tidy-probe.sh`, `tidy-probe/` and `tests/` beside them are this project's own and are still exercised by ctest (`gate_lib_shell_test`, `clang_tidy_gate_probe`); `resolve-venv.sh` was used only by the old hooks and nothing calls it now.

- `pre-commit` — **checks, never rewrites**: `ruff format --check .`, `ruff check .` and `mypy .` (strict, from `pyproject.toml`) over the whole tree, each through `uv run`, and `clang-format --dry-run -Werror` over every tracked C/C++ file, preferring the pinned tools in `.venv/bin`. Any finding blocks the commit; a tool that cannot run is reported on stderr and skipped. It runs **no tests**. A venv made by `uv sync` (CLAUDE.md's **Build & Test**) is already locked and synced, so `uv run` uses it as it is. An older pip-made `.venv/` has no `uv.lock`, and there `uv run` would lock and sync it (writing `uv.lock` and rebuilding `cbls` into the venv) — commit with `UV_NO_SYNC=1` set to have it use the venv as it is.
- `commit-msg` — Conventional Commits, subject at most 72 characters. Unlike the hook it replaced, it does **not** pass `fixup!`/`squash!`/`amend!` subjects, so `git commit --fixup` is refused.
- `pre-push` — refuses a merge commit pushed to `main` and a tip without its review receipt, then runs the ```clean, ```build and ```test fences from CLAUDE.md's `## Build & Test` as a local CI in a fresh worktree outside the checkout (skipped for a documentation-only push), each step timed. `git config hooks.localCiCpus 0-3` pins it to fewer CPUs.

**What the hooks before wcp's enforced and these do not** (replaced 2026-10-14; the old scripts are in git history): the `ctest -LE slow` / pytest run at commit; auto-formatting and safe clang-tidy fixes at commit; at push, the **clang-tidy hard block** over the changed files plus a canary per config directory, `tidy-probe.sh` whenever a `.clang-tidy`, a `CMakeLists.txt` or the probe changed, the **block on any first-party compiler warning** in the gated build's log (configured with `CBLS_GATE_PROBE_IN_ALL=1` as its control), the advisory ruff-complexity/shellcheck/mypy pass, and the **Python gate over every tracked `.py`** (now `pre-commit`'s, over the tree). Until those are restored — a ```lint fence under CLAUDE.md's `## Build & Test` is where wcp's pre-push would run them — the clang-tidy and compiler-warning policy below is **policy you check by hand** before pushing (`.venv/bin/clang-tidy -p build <files>`, and the build log), not something a hook enforces.

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

  Because both config hazards fail open, the pre-push clang-tidy step matches
  `(warning|error):` **case-insensitively** and treats any **nonzero clang-tidy
  exit** as a finding: a misconfigured directory reports `Error: no checks
  enabled.` with a capital E, which a lower-case-only pattern misses. It also
  lints a canary translation unit per config directory whenever a `.clang-tidy`
  is itself part of the push, since such a change otherwise touches no source
  file and would skip the step that polices it.

  A clean canary cannot tell a gate that sees compiler warnings from one that
  does not, and for a long time this one did not (#171): clang-tidy prepends
  `clang-diagnostic-*`, the leading `-*` switched it off, and no target compiled
  with `-Wall`. Now `clang-diagnostic-*` is named **after** the wildcards, every
  first-party target gets `-Wall -Wextra` through `cbls_enable_warnings()` in the
  root `CMakeLists.txt` (a new target needs the call), and
  **`.githooks/tidy-probe.sh`** proves it: it lints a deliberately dirty TU,
  `.githooks/tidy-probe/incomplete_switch.cpp`, copied into every config
  directory of a mirror of the configs (so clang-tidy's real directory lookup
  applies, and a new config directory is probed automatically), and fails
  unless its incomplete `enum class` switch, unused local and unused parameter
  are all reported. It also fails on any first-party compile command missing
  the flags or cancelling them — **any** `-Wno-*`, `-w` or `-Werror`: a
  per-target suppression is a NOLINT without a written reason, so fix the code.
  Flags from the environment (`CXXFLAGS`, which CMake seeds into the cache's
  `CMAKE_CXX_FLAGS[_<CONFIG>]`) are exempt: the rule polices the project, not
  the machine. That flag check needs no clang-tidy and always runs. The mirror
  sits inside the checkout, so each probe copy's `--dump-config` must equal
  the real directory's — otherwise a missing copy would let the lookup climb
  into the real tree and pass on the wrong config. Pre-push runs the probe
  (pinned clang-tidy only, and blocking if it cannot run) when a `.clang-tidy`,
  a `CMakeLists.txt` or the probe changes; ctest runs it as
  `clang_tidy_gate_probe` (Skipped, not Passed, without the pinned clang-tidy),
  and `gate_lib_shell_test` tests the gates' text filters in
  `.githooks/gate-lib.sh`. Never add a `default:` to a `NodeOp` switch — it is
  what would hide the case.

  **No `-Werror`**: with it in `compile_commands.json` every clang-diagnostic
  finding becomes a compiler error and clang-tidy exits 1, which the step above
  reports as a config fault. GCC — the build compiler — has warnings clang lacks
  (two `-Wrange-loop-construct` at #171), so instead pre-push **blocks on any
  compiler warning in first-party code in the gated build's output**, and
  proves that scan can see: it configures with `CBLS_GATE_PROBE_IN_ALL=1`, which
  puts the probe TU into the fenced build, and blocks unless the scan finds its
  `-Wswitch` in that same log (colour codes, a translated `warning:`, an
  unexpected path spelling or a fence that stops sending compiler output there
  would otherwise make it match nothing). The three probe warnings in pre-push's
  build output are that control, not findings. That gate
  follows the local GCC version, so an upgrade can block a push; it cannot
  break a user's build, which is what `-Werror` would.

`pre-push` refuses a push whose tip has no review receipt, `refs/reviews/<sha>`, written by a line of plain git (**Self-Review** below), so the check works in every checkout. It proves a review was recorded, not that it was done well; see **Agent Self-Review**.

Gating lives in git hooks only — `.claude/settings.json` carries no `hooks` block, and none should be added. A `PostToolUse` formatter cannot see which file was edited, so it silently formats nothing; formatting belongs at commit time. Don't hand-tune formatting.

Three conventions therefore rest on you rather than on a tool: branch only from main (**Git Workflow** in `docs/development.md`), never invoke `python`/`pytest`/`mypy` outside `.venv/bin/` (a `uv sync` venv has no `pip`; change it with `uv sync`) (CLAUDE.md's **Build & Test**), and prefer the `Grep` tool to shelling out (**Code Navigation** in `docs/development.md`).

`.claude/` is gitignored local agent tooling (settings, statusline, the `/review` command). Nothing in it is part of the repo.

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

It was scoped narrower once. #154 gated everything except `benchmarks/`, because
that tree carried 22 ruff findings, 5 unformatted files and 37 `mypy --strict`
errors, plus an `__init__.py` inside the hyphenated `benchmarks/instances/uc-chped/`
which made mypy abort the whole run on the package name before checking anything.
#155 cleared all of it and emptied the exclusion. Two things are worth keeping
from that episode:

- **A partial exclusion was never the safety it looked like.** `mypy` follows
  imports, so every benchmark module reachable from a gated test file was
  hard-gated anyway. The tree was green only because the three files carrying
  errors happened to be ones nothing gated imported — and one new test importing
  any of them would have blocked a push on errors nobody in that push wrote.
- **Two naming waivers survive, scoped per file and per rule** in
  `pyproject.toml`'s `[tool.ruff.lint.per-file-ignores]`: `N803`/`N806` on
  `benchmarks/chped/reference_solve.py` and `N806` on
  `benchmarks/instances/uc-chped/data.py`. `P`, `A`, `T` and `base_T` are the
  published unit-commitment formulation's own symbols, and those solvers exist to
  be audited line-by-line against the paper whose numbers they reproduce. Same
  argument, and the same shape, as `src/io/.clang-tidy` waiving the naming check
  for the vendored readers. A third waiver needs the same standard.

Both escape routes that let the five binding-test modules rot are closed, and
both mattered: the pre-commit of the time linted only **staged** files — and its
ruff step auto-fixed rather than blocked, so a non-fixable finding like `E741`
survived staging however often the file was edited — while pre-push's other
Python steps saw only the pushed file list and merely warned. That is how 139 `mypy --strict`
errors and 9 ruff errors accumulated with every gate green throughout (#154).

## Self-Review

**Code you produced gets reviewed before it lands on main.** Run `/code-review`
on your own changes and act on what it finds; the review is not a formality you
report having done. The pre-push hook is the backstop, not the review.

**Then record the review, and push it with the commit**, or `pre-push` will
refuse the push:

```sh
git update-ref "refs/reviews/$(git rev-parse HEAD)" HEAD
git push --atomic origin HEAD "refs/reviews/$(git rev-parse HEAD)"
```

The receipt is a ref named by the reviewed commit, `refs/reviews/<sha>`, and it
travels with the push, so a CI step can check every commit that reaches the
trunk — including one pushed with `--no-verify`, or from a clone whose hooks
never ran. wcp's README has the step to paste into the project's CI.
`--atomic` lands the branch and its receipt together or not at all.

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

**Code produced by any agent or agent team must be reviewed before it can merge to main** — solo work by `/review`, a team run by one cold reviewer per issue (below). No subagent returns unreviewed work; no orchestrator merges unreviewed work. This applies to every agent team, not just the parallel-issue workflow.

**One reviewer per issue** (user decision, 2026-10-03). In a team run, each
issue gets exactly **one** cold reviewer: a separate agent that did not write
the code, reading the issue's acceptance criteria literally. That reviewer *is*
the issue's review — the implementer does not also fan out `/review`'s three
reviewers, and the orchestrator does not add a second reviewer per issue. The
cold review runs **at most two rounds**: review, a fix round, re-review. A
finding still open after round two goes to the orchestrator to fix or to file;
it does not start a third round. The final combined review of the merged result
(step 6 of **Parallel Issue Workflow** in `docs/development.md`) is a separate single pass over the whole batch, not a per-issue
reviewer.

## wcp's files in this repository

### Hooks Are Files In This Repository

The git hooks live in `.githooks/` as ordinary shell scripts, committed like any other file. `wcp init` copies them there and points `core.hooksPath` at them; wcp installs no Claude Code hooks at all. The gate is those files and nothing else, so a clone without wcp installed is still gated — including the review receipt, which is a line of plain git the Self-Review section documents rather than anything wcp runs.

- **They are yours to read, and yours to edit.** Each hook carries a `# wcp:` line recording the digest of the bytes wcp wrote. That is what lets `wcp doctor` tell an edit you made from a version wcp has moved past: an untouched hook is refreshed silently, an edited one is left alone, and one that moved on both sides is reported rather than overwritten. Edit them when the project needs something different — and say so upstream if the default was wrong for everyone.
- **A clone is not gated until someone runs `wcp init` in it.** git clones neither a hooks directory nor local config, so the files arriving with the clone is not the same as git running them. `wcp doctor` reports an unset `core.hooksPath` as an error for exactly this reason.
- **They call no wcp.** A hook that shelled out to a missing tool would exit 127, which git reads as a refusal — so a machine without wcp could not commit, and could not commit the fix either. What the hooks do call is the project's own toolchain, and there they fail open and say so on stderr.
- **Never use `git push --no-verify` or `git commit --no-verify`** unless explicitly asked. A failing hook is a signal — fix the root cause, or change the hook deliberately and commit that.

### CLAUDE.md Carries A Provenance Line

The comment at the top of CLAUDE.md is wcp's record of the bytes it wrote, in the same scheme the hooks use. `wcp doctor` reads it to tell an edit you made from a version wcp has moved past, and it is where this project's type is recorded — there is no `.wcp/` directory and no config file beside it.

**The prose is yours.** wcp writes CLAUDE.md once and never rewrites it unasked; edit it freely. When wcp adds or changes a section you have not edited, `wcp doctor` names it and `wcp init --interactive` offers it, one section at a time; answering `n` declines that version for good. Leave the top line in place, and the tooling stays able to tell your changes from its own.

### Running wcp

wcp is a command-line tool installed once per machine, outside any repository. The hooks never call it, so a machine without it is still gated.

```sh
uv tool install git+https://github.com/spoorendonk/wcp   # once per machine; @vX.Y.Z for a release
uv tool upgrade wcp                                       # to update an install without a tag
uv tool install --force git+https://github.com/spoorendonk/wcp@vX.Y.Z   # to move a tagged one
wcp --help                                                # the commands; `wcp <command> --help` for each one
```

- **Run `wcp doctor`** after pulling, after updating wcp, and before editing the hooks or CLAUDE.md. It changes nothing, and every finding names the command that fixes it. It exits 1 only when the gate does not work, e.g. `core.hooksPath` is unset, which is how every fresh clone starts until someone runs `wcp init` in it.
- **Files wcp wrote carry a `wcp:` provenance line**: `# wcp:` in each hook, and the comment at the top of CLAUDE.md. Edit the files when the project needs to, but leave that line in place. It is how `wcp doctor` tells your edits apart from a newer wcp version.
- **If `wcp` is not on PATH**, say so rather than installing it unasked. Installing it is a machine-wide change.
