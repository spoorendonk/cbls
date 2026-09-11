"""Tests for the conftest gates that stop a short run from looking complete.

Two suites in this repo disappear silently when an optional dependency is
missing: the binding tests when `_cbls_core` is not built, and every test of the
CP-SAT baseline and the independent solution verifier when the `benchmarks`
extra is not installed. Both leave a green summary behind, which is
indistinguishable from a run where they all passed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conftest import BENCHMARK_IMPORTS, files_skipping_on, missing_imports

if TYPE_CHECKING:
    from pathlib import Path


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


def test_both_benchmark_extras_are_watched() -> None:
    # The `benchmarks` extra is exactly these two, and a test guarded on one that
    # this tuple does not list would vanish with no announcement.
    assert set(BENCHMARK_IMPORTS) == {"ortools", "pyscipopt"}


def test_every_importorskipped_dependency_is_one_this_gate_watches() -> None:
    """A new `importorskip` on a third package would skip silently again.

    The gate can only announce what `BENCHMARK_IMPORTS` names, so the names in
    use and the names watched have to be the same set.
    """
    import re
    from pathlib import Path

    pattern = re.compile(r"""importorskip\(\s*["']([A-Za-z_][\w.]*)["']""")
    guarded = {
        match.group(1)
        for path in sorted(Path(__file__).parent.glob("test_*.py"))
        for match in pattern.finditer(path.read_text())
    }
    assert guarded <= set(BENCHMARK_IMPORTS), sorted(guarded - set(BENCHMARK_IMPORTS))
