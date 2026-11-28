"""The committed `_cbls_core.pyi` matches the built module (#464).

The stub is nanobind.stubgen's output put through `ruff format` and ruff's
import sort, so mypy checks every caller against the real bindings. A binding
change regenerates it:

    PYTHONPATH=build/python .venv/bin/python -m nanobind.stubgen -q -P \\
        -m cbls._cbls_core -o python/cbls/_cbls_core.pyi
    .venv/bin/ruff format python/cbls/_cbls_core.pyi
    .venv/bin/ruff check --select I --fix python/cbls/_cbls_core.pyi

`-P` keeps `EditKind.None_`, whose trailing underscore stubgen reads as private.
The output depends on the nanobind release, which the dev extra pins.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cbls

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "cbls"


def _generate(out: Path) -> None:
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    stubgen = [sys.executable, "-m", "nanobind.stubgen", "-q", "-P", "-m", "cbls._cbls_core"]
    subprocess.run([*stubgen, "-o", str(out)], env=env, check=True, timeout=120)
    ruff = str(Path(sys.executable).with_name("ruff"))
    config = ["--config", str(ROOT / "pyproject.toml")]
    subprocess.run([ruff, "format", "-q", *config, str(out)], check=True, timeout=60)
    sort = [ruff, "check", "-q", *config, "--select", "I", "--fix", str(out)]
    subprocess.run(sort, check=True, timeout=60)


def test_the_committed_stub_matches_the_built_module(tmp_path: Path) -> None:
    generated = tmp_path / "_cbls_core.pyi"
    _generate(generated)
    committed = (PACKAGE / "_cbls_core.pyi").read_text()
    assert generated.read_text() == committed, (
        "python/cbls/_cbls_core.pyi is stale; regenerate it (see this file's docstring)"
    )


def test_the_package_is_marked_typed() -> None:
    assert (PACKAGE / "py.typed").is_file()
    assert Path(cbls.__file__).with_name("py.typed").is_file()
