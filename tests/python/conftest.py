import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

# The build tree's package (build/python/cbls, with _cbls_core inside, #463).
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Add the repo root so tests can import benchmark modules (benchmarks.<name>.<module>).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Without compiled bindings (CBLS_BUILD_PYTHON defaults to OFF), skip the tests that
# import them (found from the source, so a new one is covered and pure-Python tests
# still run) instead of failing collection with ImportError. A silent skip reads as a
# green run, so the terminal summary names what was dropped, and
# CBLS_REQUIRE_BINDINGS=1 (set by the ```test fence in CLAUDE.md) makes it a hard error.
_ignored_binding_tests: list[str] = []

#: The package a binding test needs: `cbls` and its compiled `cbls._cbls_core`.
BINDING_MODULE = "cbls"


@pytest.fixture(autouse=True)
def _no_inherited_git_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip every GIT_* variable before a test can spawn git.

    A git hook exports GIT_DIR and GIT_INDEX_FILE, and any git a test runs
    inherits them whatever its `cwd`, so a tmp-dir `git init` re-initialises the
    repository being committed to and its `git add` stages into the commit's
    index (#160). A test that needs a GIT_* variable sets its own with
    `monkeypatch.setenv`.
    """
    for name in [name for name in os.environ if name.startswith("GIT_")]:
        monkeypatch.delenv(name)


def imports_module(text: str, module: str) -> bool:
    """Whether this source really imports `module` -- by AST, not by substring.

    A whole-file substring match ignores a pure-Python test whose docstring names
    the module; the narrower `import cbls` misses `from cbls import X`
    and `importlib.import_module("cbls")`, and a missed binding test raises
    ImportError at collection, aborting the whole session. Docstrings and
    comments are not in the AST; all three import spellings are recognised, of
    the module or a submodule.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # Let pytest report an unparseable file itself; guessing from a broken
        # parse is how a binding test would be ignored for a typo.
        return False
    return any(
        name == module or name.startswith(f"{module}.")
        for node in ast.walk(tree)
        for name in imported_names(node)
    )


def imported_names(node: ast.AST) -> list[str]:
    """The module names `node` imports, if it is an import of any spelling."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [node.module or ""]
    if isinstance(node, ast.Call) and node.args:
        # importlib.import_module("cbls"), which no `import` statement
        # spelling would catch.
        func = node.func
        called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        first = node.args[0]
        if called != "import_module" or not isinstance(first, ast.Constant):
            return []
        if isinstance(first.value, str):
            return [first.value]
    return []


def binding_test_files(directory: Path) -> list[str]:
    """The test files in `directory` that import the compiled bindings."""
    return [
        path.name
        for path in sorted(directory.glob("test_*.py"))
        if imports_module(path.read_text(), BINDING_MODULE)
    ]


def check_cbls_is_the_build_trees(build_package: Path) -> None:
    """Refuse a `cbls` that resolves anywhere but the build tree, once it has one.

    A venv synced before #463 holds an editable cbls whose import hook runs
    ahead of sys.path and maps the package to the source tree, where there is
    no _cbls_core; the tests would then skip, or test a stale install.
    """
    if not (build_package / "__init__.py").exists():
        return
    spec = importlib.util.find_spec("cbls")
    origin = Path(spec.origin).resolve() if spec and spec.origin else None
    if origin is None or build_package.resolve() not in origin.parents:
        raise pytest.UsageError(
            f"`import cbls` resolves to {origin}, not to the build tree's {build_package}: "
            "a cbls installed in this venv shadows it. "
            "`uv sync --extra dev --extra benchmarks` removes it."
        )


check_cbls_is_the_build_trees(Path(build_dir) / "cbls")

try:
    import cbls._cbls_core  # noqa: F401
except ImportError:
    _ignored_binding_tests = binding_test_files(Path(__file__).parent)
    collect_ignore = _ignored_binding_tests

    if os.environ.get("CBLS_REQUIRE_BINDINGS") == "1":
        raise pytest.UsageError(
            "CBLS_REQUIRE_BINDINGS=1, but cbls._cbls_core is not importable, so these "
            f"binding test files would not run: {', '.join(_ignored_binding_tests)}. "
            "Build them with:\n"
            '  cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"\n'
            "  cmake --build build -j$(nproc)\n"
            "If nanobind is missing from the venv, "
            "`uv sync --extra dev --extra benchmarks` installs it."
        ) from None


# `ortools`, `pyscipopt` and `pyvrp` live in the `benchmarks` extra, not `dev`, so a
# `--extra dev` venv lacks them and the tests that `pytest.importorskip` them vanish.
# The same rule holds: the skip is named in the summary and CBLS_REQUIRE_BENCHMARKS=1
# (the ```test fence) makes it a hard error. Files are found from the source; a guard on a
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


# `cbls_scip_traj` (#442) is built only with -DCBLS_SCIP_DIR, which the gate does not
# set, so a test asking for the `scip_driver` fixture skips without it -- named in the
# summary, and an error under CBLS_REQUIRE_SCIP_DRIVER=1. Not CBLS_REQUIRE_BENCHMARKS:
# the ```test fence sets that one.
_skipped_scip_driver_tests: list[str] = []


def scip_driver_path() -> Path:
    """Where the driver is looked for: $CBLS_SCIP_DRIVER, else the root build's."""
    default = Path(__file__).resolve().parents[2] / "build" / "cbls_scip_traj"
    return Path(os.environ.get("CBLS_SCIP_DRIVER", str(default)))


@pytest.fixture
def scip_driver(request: pytest.FixtureRequest) -> Path:
    """The built `cbls_scip_traj`, or a skip (an error under CBLS_REQUIRE_SCIP_DRIVER=1)."""
    path = scip_driver_path()
    if os.access(path, os.X_OK):
        return path
    why = f"no cbls_scip_traj at {path} (benchmarks/common/scip_driver/README.md)"
    if os.environ.get("CBLS_REQUIRE_SCIP_DRIVER") == "1":
        pytest.fail(f"CBLS_REQUIRE_SCIP_DRIVER=1, but {why}")
    _skipped_scip_driver_tests.append(request.node.nodeid)
    pytest.skip(why)


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Announce every silently skipped suite so a short run cannot look complete.

    Here, not at import or in `pytest_report_header`: pytest discards conftest
    import output on a passing run and `-q` (the gate's flag) hides the header.
    `get_terminal_writer()` asserts in `pytest_configure`, before the reporter
    exists.
    """
    if _ignored_binding_tests:
        terminalreporter.write_line(
            f"WARNING: cbls._cbls_core not built — skipped {len(_ignored_binding_tests)} binding "
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
    if _skipped_scip_driver_tests:
        terminalreporter.write_line(
            f"WARNING: cbls_scip_traj not built — skipped {len(_skipped_scip_driver_tests)} "
            f"test(s) needing it (looked at {scip_driver_path()}). "
            "Set CBLS_REQUIRE_SCIP_DRIVER=1 to make this an error.",
            yellow=True,
            bold=True,
        )
