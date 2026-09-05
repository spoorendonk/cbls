"""src/io/.clang-tidy must APPEND to the root check list, not replace it.

Without ``InheritParentConfig: true`` clang-tidy drops the root's ``Checks:``,
leaves nothing enabled, and exits 1 with "Error: no checks enabled." -- every
translation unit under ``src/io/`` then goes silently unlinted while the
pre-push gate reports success. That was the live state until issue #121's
``src/io`` half; this test is what stops it coming back.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
IO_DIR = REPO_ROOT / "src" / "io"
PINNED = REPO_ROOT / ".venv" / "bin" / "clang-tidy"


def _tidy() -> str:
    """Prefer the pinned wheel: a system clang-tidy of another vintage enables
    a different default set, which would make the comparison below meaningless."""
    if PINNED.is_file():
        return str(PINNED)
    found = shutil.which("clang-tidy")
    if found is None:
        pytest.skip("clang-tidy not available (venv or PATH)")
    return found


def _enabled_checks(cwd: Path) -> set[str]:
    proc = subprocess.run(
        [_tidy(), "--list-checks"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        f"clang-tidy --list-checks failed in {cwd} -- a config there most likely "
        f"replaced the root check list instead of inheriting it:\n"
        f"{proc.stdout}{proc.stderr}"
    )
    # First line is the "Enabled checks:" banner.
    return {ln.strip() for ln in proc.stdout.splitlines()[1:] if ln.strip()}


def test_src_io_inherits_the_root_check_list() -> None:
    root = _enabled_checks(REPO_ROOT)
    # Guard the guard: if the root config were itself broken, an equality
    # assertion between two empty sets would pass and prove nothing.
    assert len(root) > 100, "root .clang-tidy enables almost nothing -- it is broken"

    io_checks = _enabled_checks(IO_DIR)

    # src/io/ deliberately subtracts the naming check (vendored readers spell
    # their helpers camelBack). Subtract it from both sides so this stays true
    # whichever way #121's ratchet moves that check.
    naming = {"readability-identifier-naming"}
    assert io_checks - naming == root - naming
