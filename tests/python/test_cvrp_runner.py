"""Tests for the `cbls_cvrp` runner (#197), through its binary.

What a Catch2 test cannot reach: the runner's flag families (#198) -- a scalar
search flag must be refused on a List-only model, where FJ has nothing to jump
-- and the files it writes, read back through the same `.sol` parser that pins
the BKS. Every solve here is iteration-budgeted with the wall clock off, so it
is quick and the same on every machine.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from benchmarks.instances.cvrp.download import parse_sol

REPO = Path(__file__).resolve().parents[2]
CVRP_BINARY = REPO / "build" / "cbls_cvrp"
INST_DIR = REPO / "benchmarks" / "instances" / "cvrp"


def _run(*args: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    if not CVRP_BINARY.exists():
        pytest.skip("cbls_cvrp not built")
    return subprocess.run(
        [str(CVRP_BINARY), "--inst-dir", str(INST_DIR), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


#: Every flag of the scalar family. On a List-only model each would be read by
#: nothing, so the runner refuses it rather than recording it.
SCALAR_SEARCH_FLAGS = [
    ["--no-float-hook"],
    ["--compound-moves"],
    ["--no-compound-moves"],
    ["--novelty-prob", "0.5"],
    ["--unproductive-iters", "300"],
]


@pytest.mark.parametrize("flag", SCALAR_SEARCH_FLAGS)
def test_cvrp_refuses_a_scalar_search_flag(flag: list[str], tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    result = _run("--instance", "X-n101-k25", "--out", str(out), *flag)
    assert result.returncode == 2, result.stdout
    assert f"{flag[0]} is a scalar" in result.stderr, result.stderr
    assert not out.exists()


def test_cvrp_refuses_more_neighbours_than_the_instance_has(tmp_path: Path) -> None:
    """X-n101-k25 has 100 customers, so at most 99 others: a larger K would be
    clamped by the engine and record one run under several cells."""
    result = _run("--instance", "X-n101-k25", "--neighbours", "100")
    assert result.returncode == 2, result.stdout
    assert "--neighbours" in result.stderr, result.stderr


def test_cvrp_help_lists_its_families_only() -> None:
    result = _run("--help")
    assert result.returncode == 0
    for flag in ("--structural-prob", "--structural-selection", "--neighbours", "--no-lns"):
        assert flag in result.stdout, flag
    for refused in SCALAR_SEARCH_FLAGS:
        assert refused[0] not in result.stdout, refused[0]


@pytest.mark.parametrize("threads", ["1", "2"])
def test_cvrp_writes_a_verified_solution_its_record_agrees_with(
    threads: str, tmp_path: Path
) -> None:
    out = tmp_path / "r.json"
    sol_path = tmp_path / "r.sol"
    result = _run(
        "--instance",
        "X-n101-k25",
        "--out",
        str(out),
        "--solution-out",
        str(sol_path),
        "--threads",
        threads,
        "--no-time-limit",
        "--max-iterations",
        "200",
        "--seed",
        "3",
        "--structural-selection",
        "violation_guided",
        "--neighbours",
        "10",
        "--commit",
        "test",
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text())
    assert record["feasible"] and record["verified"], record
    assert record["bks"] == 27591
    assert record["fleet"] == 28  # default_fleet(25)
    assert record["batches"] == (
        record["fj_batches"] + record["novelty_batches"] + record["structural_batches"]
    )
    assert record["threads"] == int(threads)
    assert record["workers_completed"] == int(threads)
    assert "structural_selection=violation_guided" in record["search_config"]
    assert "neighbours=10" in record["search_config"]
    assert "novelty_prob" not in record["search_config"]  # the scalar family is not recorded

    sol = parse_sol(sol_path.read_text())
    assert sol.cost == record["objective"]
    assert sum(1 for r in sol.routes if r) == record["routes_used"]
    customers = sorted(c for r in sol.routes for c in r)
    assert customers == list(range(1, 101))  # CVRPLIB ids, every customer once


def test_cvrp_writes_no_solution_without_a_verified_one(tmp_path: Path) -> None:
    """A fleet too small to carry the demand can never be feasible, so no `.sol`
    may appear: a solution file is a claim the instance's checker would test."""
    sol_path = tmp_path / "r.sol"
    sol_path.write_text("Route #1: 1\nCost 1\n")  # a previous run's, which must not survive
    out = tmp_path / "r.json"
    result = _run(
        "--instance",
        "X-n101-k25",
        "--fleet",
        "2",
        "--no-time-limit",
        "--max-iterations",
        "50",
        "--out",
        str(out),
        "--solution-out",
        str(sol_path),
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text())
    assert not record["feasible"]
    assert record["objective"] is None
    assert not sol_path.exists()
