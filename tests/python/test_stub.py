"""The committed `_cbls_core.pyi` matches the built module (#464).

The stub is nanobind.stubgen's output, unedited, so mypy checks every caller
against the real bindings. A binding change regenerates it:

    PYTHONPATH=build/python .venv/bin/python -m nanobind.stubgen -q -P \\
        -m cbls._cbls_core -o python/cbls/_cbls_core.pyi

`-P` keeps `EditKind.None_`, whose trailing underscore stubgen reads as private.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cbls

PACKAGE = Path(__file__).resolve().parents[2] / "python" / "cbls"


def test_the_committed_stub_matches_the_built_module(tmp_path: Path) -> None:
    generated = tmp_path / "_cbls_core.pyi"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    subprocess.run(
        [sys.executable, "-m", "nanobind.stubgen", "-q", "-P", "-m", "cbls._cbls_core"]
        + ["-o", str(generated)],
        env=env,
        check=True,
        timeout=120,
    )
    committed = (PACKAGE / "_cbls_core.pyi").read_text()
    assert generated.read_text() == committed, (
        "python/cbls/_cbls_core.pyi is stale; regenerate it (see this file's docstring)"
    )


def test_the_package_is_marked_typed() -> None:
    assert (PACKAGE / "py.typed").is_file()
    assert Path(cbls.__file__).with_name("py.typed").is_file()
