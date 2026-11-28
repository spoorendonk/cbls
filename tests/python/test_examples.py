"""The Python and CLI halves of the docs' example pages run as the pages show (#470, #471).

Each case is a page's command line, run from the repository root in a child
process, as a reader would run it.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import cbls
import pytest

ROOT = Path(__file__).resolve().parents[2]

# MINLPLib's best known (= optimal) objective for st_e38, bounds.csv; cbls reaches it.
ST_E38_BEST = 7197.727149


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """`argv` under this interpreter, importing the cbls the tests import."""
    env = dict(os.environ, PYTHONPATH=str(Path(cbls.__file__).resolve().parents[1]))
    return subprocess.run(
        [sys.executable, *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )


def _assert_near(objective: float, reference: float) -> None:
    # cbls reaches it in under 0.2 s, but a loaded host may stop the 2 s
    # solve short of it: allow 1 % above, and only feasibility noise below.
    assert reference * (1 - 1e-6) <= objective <= reference * 1.01


@pytest.mark.parametrize(
    ("command", "reference"),
    [
        ("examples/minlplib.py benchmarks/instances/minlplib/st_e38.nl", ST_E38_BEST),
        # MIPLIB's optimum is 15; the page asks for a feasible point only.
        ("examples/mip.py benchmarks/instances/miplib-fj/neos5.mps.gz", None),
    ],
)
def test_an_example_page_command_runs_as_written(command: str, reference: float | None) -> None:
    result = _run(command.split())

    assert result.returncode == 0, result.stdout + result.stderr
    assert "feasible  = True\n" in result.stdout
    assert "verify: PASS\n" in result.stdout
    if reference is not None:
        found = re.search(r"^objective = (\S+)$", result.stdout, re.M)
        assert found, result.stdout
        _assert_near(float(found[1]), reference)


_ST_E38_CLI = "cbls benchmarks/instances/minlplib/st_e38.nl --time-limit 2 --threads 1"
_NEOS5_CLI = "cbls benchmarks/instances/miplib-fj/neos5.mps.gz --time-limit 2 --threads 1"


@pytest.mark.parametrize(
    ("page", "command", "reference"),
    [
        ("minlplib.md", f"{_ST_E38_CLI} --quiet", ST_E38_BEST),
        ("minlplib.md", f"{_ST_E38_CLI} --format jsonl --quiet", ST_E38_BEST),
        ("mip.md", f"{_NEOS5_CLI} --quiet", None),
        ("mip.md", f"{_NEOS5_CLI} --format jsonl --quiet", None),
    ],
)
def test_an_example_page_cli_command_runs_as_written(
    page: str, command: str, reference: float | None
) -> None:
    # The page shows this exact line; `python -m cbls` is the console script's entry.
    assert f"\n{command}\n" in (ROOT / "docs" / "examples" / page).read_text()
    program, *args = command.split()
    assert program == "cbls"
    result = _run(["-m", "cbls", *args])

    assert result.returncode == 0, result.stdout + result.stderr
    if "--format jsonl" in command:
        lines = result.stdout.splitlines()
        assert len(lines) == 1, result.stdout
        record = json.loads(lines[0])
        assert record["event"] == "result"
        assert record["feasible"] is True
        objective = float(record["objective"])
    else:
        assert re.search(r"^Status:\s+feasible$", result.stdout, re.M), result.stdout
        found = re.search(r"^Objective:\s+(\S+)$", result.stdout, re.M)
        assert found, result.stdout
        objective = float(found[1])
    if reference is not None:
        _assert_near(objective, reference)
