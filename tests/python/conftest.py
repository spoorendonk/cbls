import importlib.util
import os
import sys
from pathlib import Path

import pytest

# Add the build directory to the path so we can import _cbls_core
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# Add the repo root so tests can import benchmark modules (benchmarks.<name>.<module>).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# If the compiled bindings aren't built, skip the tests that need them rather than
# failing with ImportError during collection. Detected from the source rather than
# hardcoded, so a new binding test is covered without touching this file — and so a
# pure-Python test still runs when the bindings are absent.
#
# The skip is announced rather than silent. It used to be silent, and because
# CBLS_BUILD_PYTHON defaults to OFF, the documented build produced a run that
# reported a green summary while never executing any of the 81 binding tests — a
# result indistinguishable from one where they all passed. Nothing in the gates
# would have caught a binding regression. So: the terminal summary names what was
# dropped, and CBLS_REQUIRE_BINDINGS=1 (set by the ```test fence in CLAUDE.md)
# turns the skip into a hard error, on the same principle as the hooks refusing
# to lint quietly when ruff is missing from the venv.
_ignored_binding_tests: list[str] = []

try:
    import _cbls_core  # noqa: F401
except ImportError:
    _ignored_binding_tests = [
        path.name
        for path in sorted(Path(__file__).parent.glob("test_*.py"))
        if "_cbls_core" in path.read_text()
    ]
    collect_ignore = _ignored_binding_tests

    if os.environ.get("CBLS_REQUIRE_BINDINGS") == "1":
        raise pytest.UsageError(
            "CBLS_REQUIRE_BINDINGS=1, but _cbls_core is not importable, so these "
            f"binding test files would not run: {', '.join(_ignored_binding_tests)}. "
            "Build them with:\n"
            '  cmake -B build -DCBLS_BUILD_PYTHON=ON -DPython_EXECUTABLE="$PWD/.venv/bin/python"\n'
            "  cmake --build build -j$(nproc)\n"
            "If nanobind is missing from the venv, `.venv/bin/pip install -e '.[dev]'` "
            "installs it."
        ) from None


# The same hazard one layer out. `ortools` and `pyscipopt` live in the
# `benchmarks` extra, not `dev`, so a checkout bootstrapped with the documented
# `.venv/bin/pip install -e '.[dev]'` has neither -- and every test of the CP-SAT
# baseline and of the independent solution verifier opens with
# `pytest.importorskip`, so all of them vanish into a green summary. That is the
# same failure shape as the 81 binding tests, and gets the same answer: the skip
# is named in the summary, and CBLS_REQUIRE_BENCHMARKS=1 (set by the ```test
# fence in CLAUDE.md) turns it into a hard error.
#
# Detected from the source rather than hardcoded, so a new benchmark-dependency
# test is covered without touching this file.
BENCHMARK_IMPORTS = ("ortools", "pyscipopt")


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
    return [
        path.name
        for path in sorted(directory.glob("test_*.py"))
        if any(f'importorskip("{name}"' in path.read_text() for name in missing)
    ]


_missing_benchmark_imports = missing_imports(BENCHMARK_IMPORTS)
_ignored_benchmark_tests = files_skipping_on(_missing_benchmark_imports, Path(__file__).parent)

if _ignored_benchmark_tests and os.environ.get("CBLS_REQUIRE_BENCHMARKS") == "1":
    raise pytest.UsageError(
        f"CBLS_REQUIRE_BENCHMARKS=1, but {', '.join(_missing_benchmark_imports)} "
        f"{'is' if len(_missing_benchmark_imports) == 1 else 'are'} not importable, so "
        f"tests in these files would be skipped: {', '.join(_ignored_benchmark_tests)}. "
        "Install them with:\n"
        "  .venv/bin/pip install -e '.[benchmarks]'"
    )


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Announce every silently skipped suite so a short run cannot look complete.

    Reported here, next to the pass count, rather than printed at import or via
    `pytest_report_header`: pytest captures stdout/stderr during conftest import
    and discards it for a passing session, and `-q` suppresses the header --
    which is what the gate and every documented command run, so both stayed
    invisible in the one case that matters. `get_terminal_writer()` is not usable
    from `pytest_configure` either; it asserts before the reporter exists.
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
