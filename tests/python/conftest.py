import os
import sys
from pathlib import Path

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
try:
    import _cbls_core  # noqa: F401
except ImportError:
    collect_ignore = [
        path.name
        for path in sorted(Path(__file__).parent.glob("test_*.py"))
        if "_cbls_core" in path.read_text()
    ]
