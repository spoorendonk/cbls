"""The UC-CHPED docs page's commands run as written (#468).

docs/uc-chped.md's `bash` blocks are read from the page and run in order, from
a directory that links the checkout's build/, benchmarks/ and venv, so the
files they write stay out of the tree.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "docs" / "uc-chped.md"
UC_CHPED_BINARY = ROOT / "build" / "cbls_uc_chped"


def _bash_blocks(text: str) -> list[str]:
    return re.findall(r"^```bash\n(.*?)^```$", text, re.M | re.S)


def test_uc_chped_page_commands_run_as_written(tmp_path: Path) -> None:
    blocks = _bash_blocks(PAGE.read_text())
    # The build, the solve and the checker: a block lost to a fence typo fails here.
    assert len(blocks) == 3, blocks
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    (tmp_path / "build").symlink_to(ROOT / "build")
    (tmp_path / "benchmarks").symlink_to(ROOT / "benchmarks")
    # The venv this test runs in, so .venv/bin/python resolves in any checkout.
    (tmp_path / ".venv").symlink_to(sys.prefix)

    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", "\n".join(blocks)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERIFIED" in result.stdout, result.stdout
    checked = [ln for ln in result.stdout.splitlines() if ln.startswith(("OK", "FAIL"))]
    assert len(checked) == 1, result.stdout
    assert checked[0].startswith("OK"), result.stdout
