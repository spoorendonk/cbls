"""Tests for the conftest gates that stop a short run from looking complete.

Two suites in this repo disappear silently when an optional dependency is
missing: the binding tests when `_cbls_core` is not built, and every test of the
CP-SAT baseline and the independent solution verifier when the `benchmarks`
extra is not installed. Both leave a green summary behind, which is
indistinguishable from a run where they all passed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import (
    BENCHMARK_IMPORTS,
    BINDING_MODULE,
    binding_test_files,
    files_skipping_on,
    imports_module,
    missing_imports,
)


def test_an_absent_dependency_is_reported_missing() -> None:
    assert missing_imports(("definitely_not_a_package_xyz",)) == ["definitely_not_a_package_xyz"]


def test_an_importable_dependency_is_not_reported_missing() -> None:
    assert missing_imports(("json", "pathlib")) == []


def test_the_files_that_skip_on_a_missing_dependency_are_named(tmp_path: Path) -> None:
    (tmp_path / "test_guarded.py").write_text('pytest.importorskip("pyscipopt", reason="x")\n')
    (tmp_path / "test_plain.py").write_text("def test_x() -> None:\n    pass\n")
    assert files_skipping_on(["pyscipopt"], tmp_path) == ["test_guarded.py"]


def test_nothing_is_named_when_nothing_is_missing(tmp_path: Path) -> None:
    (tmp_path / "test_guarded.py").write_text('pytest.importorskip("pyscipopt")\n')
    assert files_skipping_on([], tmp_path) == []


def test_every_package_in_the_benchmarks_extra_is_watched() -> None:
    """Read off pyproject rather than restated here.

    A third package added to the extra would otherwise leave this green while a
    test guarded on it vanished with no announcement -- which is the drift the
    gate exists to stop.
    """
    import re
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    extra = pyproject["project"]["optional-dependencies"]["benchmarks"]
    declared = {re.split(r"[<>=!~\[ ]", requirement, maxsplit=1)[0] for requirement in extra}
    assert declared == set(BENCHMARK_IMPORTS)


def test_every_importorskipped_dependency_is_one_this_gate_watches() -> None:
    """A new `importorskip` on a third package would skip silently again.

    The gate can only announce what `BENCHMARK_IMPORTS` names, so the names in
    use and the names watched have to be the same set.
    """
    from pathlib import Path

    from conftest import IMPORTORSKIP

    guarded = {
        name
        for path in sorted(Path(__file__).parent.glob("test_*.py"))
        for name in IMPORTORSKIP.findall(path.read_text())
    }
    assert guarded <= set(BENCHMARK_IMPORTS), sorted(guarded - set(BENCHMARK_IMPORTS))


# --- the binding gate ----------------------------------------------------------
#
# The rule that decides which files are collect-ignored when `_cbls_core` is not
# built had no test at all, and shipped two opposite defects in turn: a whole-file
# substring search that ignored this very suite because its docstring names the
# module, and a narrower one that missed `from _cbls_core import X` -- which does
# not merely run the test, it raises ImportError during collection and aborts the
# entire session.


def test_a_file_that_only_mentions_the_module_in_prose_is_not_a_binding_test() -> None:
    text = '"""Prose that says import _cbls_core without importing it."""\n'
    assert not imports_module(text, BINDING_MODULE)


def test_a_comment_naming_the_module_is_not_a_binding_test() -> None:
    assert not imports_module("# import _cbls_core one day\nx = 1\n", BINDING_MODULE)


def test_a_plain_import_is_a_binding_test() -> None:
    assert imports_module("import _cbls_core\n", BINDING_MODULE)


def test_an_aliased_import_is_a_binding_test() -> None:
    assert imports_module("import _cbls_core as core\n", BINDING_MODULE)


def test_a_from_import_is_a_binding_test() -> None:
    """The spelling that aborts collection for the whole directory when missed."""
    assert imports_module("from _cbls_core import Model\n", BINDING_MODULE)


def test_a_dynamic_import_is_a_binding_test() -> None:
    assert imports_module('importlib.import_module("_cbls_core")\n', BINDING_MODULE)


def test_an_unparseable_file_is_not_claimed_as_a_binding_test() -> None:
    """pytest reports the syntax error itself; guessing from a broken parse is
    how a real binding test gets ignored for a typo."""
    assert not imports_module("def (:\n", BINDING_MODULE)


def test_binding_test_files_names_only_the_importers(tmp_path: Path) -> None:
    (tmp_path / "test_binding.py").write_text("from _cbls_core import Model\n")
    (tmp_path / "test_prose.py").write_text('"""about _cbls_core."""\n')
    (tmp_path / "not_a_test.py").write_text("import _cbls_core\n")
    assert binding_test_files(tmp_path) == ["test_binding.py"]


def test_a_prose_only_test_still_runs_when_the_bindings_are_absent(tmp_path: Path) -> None:
    """End to end, in a directory where `_cbls_core` genuinely cannot be imported.

    The unit tests above pin the predicate; this pins what the predicate is FOR.
    conftest adds `../../build/python` to `sys.path` relative to its own location,
    so a copy of it two levels below `tmp_path` finds no such directory and the
    bindings really are missing -- the situation the gate exists for.
    """
    root = tmp_path / "tests" / "python"
    root.mkdir(parents=True)
    source = Path(__file__).parent / "conftest.py"
    (root / "conftest.py").write_text(source.read_text())
    (root / "test_prose.py").write_text(
        '"""This suite mentions from _cbls_core import Model in prose only."""\n\n'
        "def test_it_runs() -> None:\n    assert True\n"
    )
    (root / "test_real_binding.py").write_text(
        "from _cbls_core import Model\n\ndef test_needs_bindings() -> None:\n    assert Model\n"
    )
    # Without stripping it, the gated run's own CBLS_REQUIRE_BINDINGS=1 is
    # inherited and the child raises UsageError before collecting anything --
    # which is the gate working, but not the thing under test here.
    env = {k: v for k, v in os.environ.items() if k != "CBLS_REQUIRE_BINDINGS"}
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", str(root), "-q", "-p", "no:cacheprovider"],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=env,
    )
    # The prose file ran; the real binding file was ignored rather than aborting
    # collection for both.
    assert "1 passed" in completed.stdout, completed.stdout + completed.stderr
    assert "test_real_binding.py" in completed.stdout
    # Collection was not aborted: the missed-import failure mode exits 2 having
    # run nothing at all.
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Interrupted" not in completed.stdout
