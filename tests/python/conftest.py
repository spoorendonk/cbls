import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

# Add the build directory to the path so we can import _cbls_core
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Add the repo root so tests can import benchmark modules (benchmarks.<name>.<module>).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Without compiled bindings, skip the tests that import them (found from the
# source, so a new one is covered and pure-Python tests still run) instead of
# failing collection with ImportError. A silent skip reads as a green run, so the
# terminal summary names what was dropped, and CBLS_REQUIRE_BINDINGS=1 (set by
# the ```test fence in CLAUDE.md) makes it a hard error.
_ignored_binding_tests: list[str] = []

#: The compiled module a binding test needs.
BINDING_MODULE = "_cbls_core"


@pytest.fixture(autouse=True)
def _no_inherited_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every GIT_* variable before a test can spawn git.

    A git hook exports GIT_DIR and GIT_INDEX_FILE, and any git a test runs
    inherits them whatever its `cwd`, so a tmp-dir `git init` re-initialises the
    repository being committed to (#160). A test that needs a GIT_* variable sets
    its own with `monkeypatch.setenv`.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


def imports_module(text: str, module: str) -> bool:
    """Whether this source really imports `module` -- by AST, not by substring.

    A substring match ignores a pure-Python test whose docstring names the
    module, or misses `from _cbls_core import X` /
    `importlib.import_module("_cbls_core")`, and a missed binding test raises
    ImportError at collection, aborting the whole session. Docstrings and
    comments are not in the AST; all three import spellings are recognised.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Let pytest report an unparseable file itself; guessing from a broken
        # parse is how a binding test would be ignored for a typo.
        return False
    prefix = f"{module}."
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name == module or a.name.startswith(prefix) for a in node.names):
                return True
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if name == module or name.startswith(prefix):
                return True
        elif isinstance(node, ast.Call):
            # importlib.import_module("_cbls_core"), which no `import` statement
            # spelling would catch.
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if called == "import_module" and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and first.value == module:
                    return True
    return False


def binding_test_files(directory: Path) -> list[str]:
    """The test files in `directory` that import the compiled bindings."""
    return [
        path.name
        for path in sorted(directory.glob("test_*.py"))
        if imports_module(path.read_text(), BINDING_MODULE)
    ]


try:
    import _cbls_core  # noqa: F401
except ImportError:
    _ignored_binding_tests = binding_test_files(Path(__file__).parent)
    collect_ignore = _ignored_binding_tests

    if os.environ.get("CBLS_REQUIRE_BINDINGS") == "1":
        raise pytest.UsageError(
            "CBLS_REQUIRE_BINDINGS=1, but _cbls_core is not importable, so these "
            f"binding test files would not run: {', '.join(_ignored_binding_tests)}. "
            "Build them with:\n"
            '  cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"\n'
            "  cmake --build build -j$(nproc)\n"
            "If nanobind is missing from the venv, "
            "`uv sync --extra dev --extra benchmarks` installs it."
        ) from None


# `ortools`, `pyscipopt` and `pyvrp` live in the `benchmarks` extra, not `dev`,
# and the tests needing them `pytest.importorskip` them, so the same rule holds:
# the skip is named in the summary and CBLS_REQUIRE_BENCHMARKS=1 (the ```test
# fence) makes it a hard error. Files are found from the source; a guard on a
# new package needs this tuple extended, which
# `test_every_importorskipped_dependency_is_one_this_gate_watches` enforces.
BENCHMARK_IMPORTS = ("ortools", "pyscipopt", "pyvrp")

#: How a test file says it needs one of them. One pattern, shared with the audit
#: test, because the gate matching a narrower shape than the audit is how a
#: single-quoted `importorskip` would pass the audit and still skip in silence.
IMPORTORSKIP = re.compile(r"""importorskip\(\s*["']([A-Za-z_][\w.]*)["']""")


def missing_imports(names: tuple[str, ...]) -> list[str]:
    """Which of `names` this interpreter cannot import."""
    return [name for name in names if importlib.util.find_spec(name) is None]


def files_skipping_on(missing: list[str], directory: Path) -> list[str]:
    """Test files in `directory` that skip themselves when `missing` is absent.

    Matched on the `pytest.importorskip("<name>"` call itself rather than a list
    kept here, so a new test guarded the same way is covered without an edit.
    """
    if not missing:
        return []
    wanted = set(missing)
    # One read per file, not one per missing name.
    return [
        path.name
        for path in sorted(directory.glob("test_*.py"))
        if wanted & set(IMPORTORSKIP.findall(path.read_text()))
    ]


_missing_benchmark_imports = missing_imports(BENCHMARK_IMPORTS)
_ignored_benchmark_tests = files_skipping_on(_missing_benchmark_imports, Path(__file__).parent)

if _ignored_benchmark_tests and os.environ.get("CBLS_REQUIRE_BENCHMARKS") == "1":
    raise pytest.UsageError(
        f"CBLS_REQUIRE_BENCHMARKS=1, but {', '.join(_missing_benchmark_imports)} "
        f"{'is' if len(_missing_benchmark_imports) == 1 else 'are'} not importable, so "
        f"tests in these files would be skipped: {', '.join(_ignored_benchmark_tests)}. "
        "Install them with (uv sync drops any extra it is not given, so name both):\n"
        "  uv sync --extra dev --extra benchmarks"
    )


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Announce every silently skipped suite so a short run cannot look complete.

    Here, not at import or in `pytest_report_header`: pytest discards conftest
    import output on a passing run and `-q` (the gate's flag) hides the header.
    `get_terminal_writer()` asserts in `pytest_configure`, before the reporter
    exists.
    """
    if _ignored_binding_tests:
        terminalreporter.write_line(
            f"WARNING: _cbls_core not built — skipped {len(_ignored_binding_tests)} binding "
            f"test file(s): {', '.join(_ignored_binding_tests)}. "
            "Set CBLS_REQUIRE_BINDINGS=1 to make this an error.",
            yellow=True,
            bold=True,
        )
    if _ignored_benchmark_tests:
        terminalreporter.write_line(
            f"WARNING: {', '.join(_missing_benchmark_imports)} not installed — skipped tests "
            f"in {len(_ignored_benchmark_tests)} file(s): "
            f"{', '.join(_ignored_benchmark_tests)}. "
            "Set CBLS_REQUIRE_BENCHMARKS=1 to make this an error.",
            yellow=True,
            bold=True,
        )
