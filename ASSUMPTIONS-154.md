# Assumption log — issue #154

- **The "five files"**: the issue named a count, not paths, and told me to
  re-derive. Resolved as: derived from `mypy --strict tests/python/` on
  b4be81f — `test_expr.py` (63 errors), `test_dag.py` (28),
  `test_moves.py` (25), `test_violation_delta.py` (15), `test_search.py` (8),
  summing to the 139 the issue reports across exactly 5 files. Because: it
  matches the "imports `_cbls_core`" hint minus `test_pool.py`, which is
  already clean. **Note for whoever closes #154**: the per-file split in the
  issue body (109/36/33/18/13) is wrong — it counts all mypy output lines,
  `note:` lines included, and sums to 209, not 139. The file *selection* it
  implies is correct; only the arithmetic is not.

- **How precise the annotations can be**: `_cbls_core` is a compiled nanobind
  extension with `ignore_missing_imports = true` in `pyproject.toml`, so every
  name reached through `cbls.` is `Any` to mypy and a model/variable/node local
  cannot be given a checkable type. Resolved as: annotated the things that *are*
  checkable — `vid(handle: int) -> int`, `-> None` on every test function, and
  `_naive_total(m: "cbls.Model", vm: "cbls.ViolationManager", var_id: int,
  j: float) -> float` — and used the string forms `"cbls.Model"` /
  `"cbls.ViolationManager"` for the dynamic ones. Because: that is already the
  house pattern (`test_pool.py::_feasible_model() -> "cbls.Model"`), it
  documents intent at the boundary, and it is honest about where the type
  information stops. No `Any` was written anywhere and no `# type: ignore` was
  added.

- **`warn_return_any` on a helper returning a binding value**: `_naive_total`
  returns `vm.total_violation()`, which is `Any`, so declaring `-> float` would
  trip `no-any-return`. Resolved as: `total: float = vm.total_violation()` then
  `return total`. Because: the narrowing is stated once, at the boundary, rather
  than suppressed — and the declared return type stays the true one.

- **`ruff format` was not in the issue's finding list**: `ruff check` reported 9
  errors, but `ruff format --check` separately wanted `test_expr.py`
  reformatted (`x ** 2.0` -> `x**2.0`, 3 lines). Resolved as: ran
  `ruff format` over `tests/python/` as part of the fix. Because: CLAUDE.md
  says style is enforced by ruff "format + lint", pre-commit would have
  reformatted these files the moment they were staged anyway, and leaving it
  would have made the new blocking gate red on its own first push.

- **Renaming the ambiguous `l`**: E741 says rename, not what to. Resolved as:
  `l = m.log_expr(x)` -> `log_node`, `l = m.lt(x, y)` -> `lt_node`. Because:
  the surrounding tests use one-letter names for the node under test (`t`, `e`,
  `s`, `g`, `n`) but `lg`/`lt` would read as the operator rather than the node;
  `_node` is unambiguous and local to two tests.

- **THE ENFORCEMENT QUESTION — answered YES, and implemented, but scoped**:
  the issue asked whether the Python gate should become a hard block at push
  over the whole tree. Resolved as: **hard block, over whole directories, but
  over `tests/python/` and `python/` only** — implemented in
  `.githooks/pre-push` as `PY_GATE_DIRS`, running `ruff check`,
  `ruff format --check` and `mypy --strict`, with a missing tool treated as a
  finding rather than a silent skip. The existing changed-files ruff-C901 and
  mypy steps stay advisory and are what still covers `benchmarks/`. Because:
  (a) cost is not the objection — ruff ~0.02s and mypy ~4.5s cold / ~0.15s warm
  against a gate that already does a clean rebuild plus the full `ctest` suite;
  (b) the whole tree *cannot* be gated today: `benchmarks/` has 22 ruff
  findings, 5 unformatted files and 37 mypy errors, and
  `benchmarks/instances/uc-chped/` has an `__init__.py` inside a hyphenated
  directory, which makes `mypy --strict` abort the entire run with "not a valid
  Python package name" before checking anything — so not one whole-tree
  invocation succeeds; (c) `benchmarks/` was explicitly off-limits this run
  (other agents working there), and several of its findings are not mechanical
  (a C901 of 39, and `N803`/`N806` single-capital names that match the
  unit-commitment paper's notation — a judgement call, not a rename).
  The split is written down in CLAUDE.md under "Where that standard is actually
  enforced", and the remaining work is filed as issue #155 with the three
  problems separated.

- **Filing #155 rather than fixing `benchmarks/` inline**: the brief says not to
  defer acceptance criteria, but also to stay out of `benchmarks/`. Resolved as:
  filed #155. Because: criterion 3 asks for the question to be *answered in
  CLAUDE.md*, which it is, and the benchmark cleanup is neither small nor
  mechanical — it needs a decision on paper-notation naming and on an
  `__init__.py` that may be imported. It is squarely the "substantial follow-up"
  the issue-tracking rule allows.

- **Blocking-step ordering**: clang-tidy's block `exit 1`s before the new Python
  report prints. Resolved as: left it that way and said so in the hook comment.
  Because: restructuring the clang-tidy block into a shared exit path touches
  heavily-commented code for a cosmetic gain, and either gate's message tells
  you what to fix.

- **No new tests**: this issue adds none, so the fail-before-fix rule has no
  test to apply to. Demonstrated the equivalent for the gate instead: stashing
  the `tests/python/` fix and running the new `PY_GATE_DIRS` step against
  b4be81f's tree makes it exit 1 on `test_dag.py:3 I001`; with the fix in place
  it prints "ruff + mypy: clean" and exits 0. The gate also exits 1 when pointed
  at `benchmarks/chped`, so it is not vacuously green.

## Review round 2 — decisions taken on reviewer findings

- **Gate scope: allowlist -> exclusion**. A reviewer showed `PY_GATE_DIRS`
  (`tests/python python`) is fail-open for a top-level Python directory that
  does not exist yet. Resolved as: `PY_GATE_EXCLUDE='^benchmarks/'` plus
  `git ls-tree -r HEAD` for the file list. Because: verified to select the exact
  same 18 files today, so it costs nothing and is fail-closed for directories
  nobody has created — which is the failure class the whole change is about.
  Rejected `[tool.mypy] exclude` / `[tool.ruff] extend-exclude`: those would
  also blind pre-commit's auto-format and the advisory steps for `benchmarks/`,
  which is strictly worse than today. The exclusion belongs in the hook, where
  only the blocking step sees it.

- **The scope claim in CLAUDE.md was FALSE and is corrected**. A reviewer proved
  `mypy` follows imports out of the gated set: appending an untyped function to
  `benchmarks/mipfeas/primal_integral.py` makes the gate red, because eight test
  modules import benchmark packages. So `benchmarks/` is *not* "covered only by
  the advisory steps" — most of it is already hard-gated transitively, and the
  tree is green only because the three benchmark files carrying `mypy` errors
  are ones nothing gated imports. Adding one test that imports
  `benchmarks/chped/data.py` blocks the next push on 16 pre-existing errors
  (measured). CLAUDE.md now says this and names the trap. Reviewer B's
  supporting count of "thirteen test modules" was itself wrong — re-derived as
  **eight** (`grep -rlE '^(from|import) benchmarks' tests/python/*.py`).

- **`ruff` and `mypy` pinned exactly** (`ruff==0.16.1`, `mypy==2.3.0`).
  Resolved as: adopted. Because: `pyproject.toml` already pins clang-format and
  clang-tidy exactly, with the reason written there — a tool that *blocks* must
  not float, or the gate goes red on a fresh clone with no code change and the
  only escape is the `--no-verify` CLAUDE.md forbids. That argument now applies
  verbatim to ruff and mypy. This changes no installed version today; it
  constrains future bootstraps.

- **Gate moved ahead of the build**. Resolved as: the Python gate is now Step 1,
  before the clean rebuild and the full suite; build+test is Step 2, advisory
  issues Step 3. Because: unlike clang-tidy it needs nothing the build produces,
  and reported from Step 3 a missing `-> None` costs a full rebuild plus ~40s of
  `ctest` before the developer hears about it — twice, once for the report and
  once for the fix. As a side effect the "clang-tidy exits before the Python
  report prints" ordering complaint disappears, so no restructuring of the
  heavily-commented clang-tidy block was needed.

- **Tool fault vs. code finding**. Resolved as: mirrored clang-tidy's
  `TIDY_CONFIG_ERR` with `PY_GATE_TOOL_ERR` — exit 1 is a finding, exit >= 2 is
  a fault, and the two print different advice. Also prefixed every finding with
  the step that produced it. Because: without the split, "mypy: cannot read
  file" printed "an untyped test function wants `-> None`", and a tool failing
  with empty output produced an empty report box naming nothing. Exercised all
  five paths (clean / findings-only / empty `VENV_BIN` / missing tool / empty
  file list) — clean exits 0, the other four exit 1 with the right message.

- **`[ -n "$VENV_BIN" ]` outer guard removed**. Resolved as: an empty
  `VENV_BIN` is now a finding. Because: the block's own comment argues a missing
  tool is a finding, not a skip, and then the guard skipped the whole block in
  silence. Unreachable today (`resolve-venv.sh` exits 2 when `pyproject.toml`
  exists and no venv resolves), but it was the exact shape the comment rejects.

- **`ruff check --no-fix`** in the blocking step, matching the advisory one.
  Because: inert while `pyproject.toml` sets no `fix = true`, but a gate that
  rewrites the working tree mid-push and then reports clean is fail-open.

- **Duplicated measurements removed from the hook comment**. The hook now names
  the *reason* for the exclusion and points at CLAUDE.md; the counts live in one
  place. Because: CLAUDE.md said "don't restate the split anywhere else" and the
  same commit restated all three counts in the hook — the seven-places pathology
  its own "Fast vs. slow tests" section documents.

- **Declined: deleting `ASSUMPTIONS-154.md`** (one reviewer asked for it).
  Because: the orchestrator's brief requires it committed and states the
  orchestrator deletes it before merging. It is not intended to reach main.

- **Declined: making pre-commit's ruff step block.** A reviewer is right that
  `ruff check --fix --quiet ... 2>/dev/null || true` means a non-fixable finding
  commits silently. Resolved as: documented the fact in CLAUDE.md (both the
  pre-commit bullet and the enforcement section) rather than changing the hook.
  Because: two other agents are committing into this repo concurrently and
  several `benchmarks/` files carry non-fixable findings, so flipping pre-commit
  to blocking mid-run would block their unrelated commits. Recorded on #155,
  which is where the benchmark tree gets cleaned.

- **Declined: de-duplicating the advisory ruff/mypy steps** now that the
  blocking gate covers the same files. Because: with the gate at Step 1 the
  advisory steps only ever run when it is green, they cost ~0.2s, and they are
  the only thing covering `benchmarks/` until #155.
