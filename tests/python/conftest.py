import os
import sys

# Add the build directory to the path so we can import _cbls_core
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build", "python")
if os.path.exists(build_dir):
    sys.path.insert(0, build_dir)

# If the compiled bindings aren't built, skip all Python tests rather than
# failing with ImportError during collection.
try:
    import _cbls_core  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
