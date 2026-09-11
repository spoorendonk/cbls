# Assumption log — issue #154

- **The "five files"**: the issue named a count, not paths, and told me to
  re-derive. Resolved as: derived from `mypy --strict tests/python/` on
  b4be81f — `test_expr.py` (109), `test_dag.py` (36), `test_moves.py` (33),
  `test_violation_delta.py` (18), `test_search.py` (13). Because: that is
  exactly the 139/5 the issue reports, and it matches the "imports `_cbls_core`"
  hint minus `test_pool.py`, which is already clean.

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
