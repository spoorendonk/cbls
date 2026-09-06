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
# reported a green summary while never executing any of the 77 binding tests — a
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


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Announce skipped binding tests so a short run cannot look complete.

    Reported here, next to the pass count, rather than printed at import or via
    `pytest_report_header`: pytest captures stdout/stderr during conftest import
    and discards it for a passing session, and `-q` suppresses the header --
    which is what the gate and every documented command run, so both stayed
    invisible in the one case that matters. `get_terminal_writer()` is not usable
    from `pytest_configure` either; it asserts before the reporter exists.
    """
    if not _ignored_binding_tests:
        return
    terminalreporter.write_line(
        f"WARNING: _cbls_core not built — skipped {len(_ignored_binding_tests)} binding "
        f"test file(s): {', '.join(_ignored_binding_tests)}. "
        "Set CBLS_REQUIRE_BINDINGS=1 to make this an error.",
        yellow=True,
        bold=True,
    )
