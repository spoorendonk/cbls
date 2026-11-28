"""The Python halves of the docs' example pages run as the pages show (#470).

Each case is a page's command line, run from the repository root in a child
process, as a reader would run it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import cbls
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("command", "reference"),
    [
        # MINLPLib's best known (= optimal) objective, bounds.csv; cbls reaches it.
        ("examples/minlplib.py benchmarks/instances/minlplib/st_e38.nl", 7197.727149),
        # MIPLIB's optimum is 15; the page asks for a feasible point only.
        ("examples/mip.py benchmarks/instances/miplib-fj/neos5.mps.gz", None),
    ],
)
def test_an_example_page_command_runs_as_written(command: str, reference: float | None) -> None:
    env = dict(os.environ, PYTHONPATH=str(Path(cbls.__file__).resolve().parents[1]))
    result = subprocess.run(
        [sys.executable, *command.split()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "feasible  = True\n" in result.stdout
    assert "verify: PASS\n" in result.stdout
    if reference is not None:
        found = re.search(r"^objective = (\S+)$", result.stdout, re.M)
        assert found, result.stdout
        # cbls reaches it in under 0.2 s, but a loaded host may stop the 2 s
        # solve short of it: allow 1 % above, and only feasibility noise below.
        assert reference * (1 - 1e-6) <= float(found[1]) <= reference * 1.01
