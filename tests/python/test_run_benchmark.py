"""Tests for the MIPfeas run driver."""

from __future__ import annotations

import csv
import subprocess
from typing import TYPE_CHECKING

import pytest

from benchmarks.mipfeas.primal_integral import NO_SOLUTION_GAP, score_instance, summarize
from benchmarks.mipfeas.run_benchmark import (
    Job,
    plan_jobs,
    read_roster,
    read_sizes,
    resolve_roster,
    with_memory_limit,
    write_failure_result,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def test_resolve_roster_maps_the_named_rosters(tmp_path: Path) -> None:
    assert resolve_roster("smoke", tmp_path) == tmp_path / "smoke.csv"
    assert resolve_roster("full", tmp_path) == tmp_path / "roster.csv"


def test_resolve_roster_passes_a_path_through(tmp_path: Path) -> None:
    custom = tmp_path / "mine.csv"
    assert resolve_roster(str(custom), tmp_path) == custom


def test_read_roster_returns_instance_names(tmp_path: Path) -> None:
    path = tmp_path / "roster.csv"
    _write_csv(
        path,
        ["instance", "reference_value", "reference_kind"],
        [["a", 1.0, "opt"], ["b", 2.0, "best"]],
    )
    assert read_roster(path) == ["a", "b"]


def test_read_sizes_of_an_absent_manifest_is_empty(tmp_path: Path) -> None:
    assert read_sizes(tmp_path / "manifest.csv") == {}


def test_plan_jobs_pairs_every_instance_with_every_engine() -> None:
    normal, large = plan_jobs(["a", "b"], ("cbls", "cpsat"), {}, large_bytes=1000)
    assert large == []
    assert set(normal) == {Job("cbls", "a"), Job("cpsat", "a"), Job("cbls", "b"), Job("cpsat", "b")}


def test_plan_jobs_sets_large_instances_aside() -> None:
    # The roster spans four orders of magnitude in size; the big ones must not run
    # four-up against a memory limit.
    sizes = {"small": 100, "huge": 50_000}
    normal, large = plan_jobs(["small", "huge"], ("cbls",), sizes, large_bytes=10_000)
    assert normal == [Job("cbls", "small")]
    assert large == [Job("cbls", "huge")]


def test_plan_jobs_treats_an_unknown_size_as_small() -> None:
    normal, large = plan_jobs(["mystery"], ("cbls",), {}, large_bytes=10_000)
    assert normal == [Job("cbls", "mystery")]
    assert large == []


def test_result_path_is_per_engine(tmp_path: Path) -> None:
    assert Job("cbls", "inst").result_path(tmp_path) == tmp_path / "cbls" / "inst.json"


def test_a_killed_job_scores_as_a_failure_not_as_unrun(tmp_path: Path) -> None:
    # A job the driver had to kill did happen, so it scores 2 like any other run
    # that produced nothing — but it must stay distinguishable from a job that was
    # never scheduled, which is excluded from the aggregate instead.
    job = Job("cbls", "inst")
    write_failure_result(job, tmp_path, "killed", "exceeded wall clock", budget=60.0)

    scored = score_instance("inst", "cbls", 100.0, "opt", tmp_path, budget=60.0)
    assert scored.status == "killed"
    assert scored.primal_integral == pytest.approx(NO_SOLUTION_GAP)

    summary = summarize([scored], "cbls")
    assert summary.scored == 1
    assert summary.not_run == 0
    assert summary.feasible == 0
    assert summary.errored == 1


def test_with_memory_limit_is_a_no_op_without_a_limit() -> None:
    assert with_memory_limit(["prog", "--flag"], None) == ["prog", "--flag"]
    assert with_memory_limit(["prog"], 0) == ["prog"]


def test_with_memory_limit_actually_caps_the_child() -> None:
    # Asserting on the wrapper's shape would pass even if the quoting were wrong,
    # so run it and ask the child what its own limit is. 2 GB in KiB.
    command = with_memory_limit(["/bin/sh", "-c", "ulimit -v"], 2.0)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(2 * 1024 * 1024)


def test_with_memory_limit_preserves_arguments_containing_spaces() -> None:
    command = with_memory_limit(["/bin/echo", "two words", "--flag=a b"], 1.0)
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout.rstrip("\n") == "two words --flag=a b"


def test_write_failure_result_creates_the_engine_directory(tmp_path: Path) -> None:
    job = Job("cpsat", "inst")
    write_failure_result(job, tmp_path / "fresh", "killed", "oom", budget=60.0)
    assert (tmp_path / "fresh" / "cpsat" / "inst.json").exists()
