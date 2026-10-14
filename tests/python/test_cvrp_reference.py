"""Tests for the PyVRP CVRP baseline (#200).

The two engines must solve the same instance. That is pinned through the shared
Python reader (`benchmarks/cvrp/verify_solution.py`): PyVRP's distance matrix is
checked equal to the reader's here, and every `cbls_cvrp` row's cost is
recomputed under that same matrix by the verifier, which `score.py` requires to
equal the runner's reported objective. So both engines' costs are in one metric,
and a drift on either side fails a check rather than skewing a gap.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np
import pytest

pyvrp = pytest.importorskip("pyvrp")

from benchmarks.common import jobs  # noqa: E402
from benchmarks.cvrp import reference_solve as ref  # noqa: E402
from benchmarks.cvrp import run_benchmark as rb  # noqa: E402
from benchmarks.cvrp import score  # noqa: E402
from benchmarks.cvrp.verify_solution import distance_matrix, read_vrp_file, verify  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

INST_DIR = ref.DEFAULT_INST_DIR


def test_pyvrp_solves_the_matrix_the_verifier_scores_with() -> None:
    inst = read_vrp_file(INST_DIR / "X" / "X-n101-k25.vrp")
    data = ref.problem_data(inst)
    np.testing.assert_array_equal(np.asarray(data.distance_matrix(0)), distance_matrix(inst))
    assert data.num_clients == inst.customers == 100
    assert data.num_depots == 1
    # Customer id c (CVRPLIB numbering) sits at location c, demand and all.
    clients = data.clients()
    assert all(clients[c - 1].location == c for c in range(1, 101))
    assert [clients[c - 1].delivery[0] for c in range(1, 101)] == inst.demands[1:]
    assert data.vehicle_type(0).capacity[0] == inst.capacity
    # The fleet never binds: one vehicle per customer.
    assert data.vehicle_type(0).num_available == inst.customers


def test_reference_solve_writes_a_record_and_sol_the_verifier_accepts(tmp_path: Path) -> None:
    out, sol = tmp_path / "r.json", tmp_path / "r.sol"
    rc = ref.main(
        [
            "--instance",
            "X-n101-k25",
            "--budget",
            "0.5",
            "--seed",
            "3",
            "--out",
            str(out),
            "--solution-out",
            str(sol),
            "--commit",
            "abc1234",
        ]
    )
    assert rc == 0
    record = json.loads(out.read_text())
    # `status` is the driver's to add, as it is for `cbls_cvrp`'s JSON; the
    # record the driver files is complete.
    job = rb.Job("X-n101-k25", rb.ARMS["pyvrp"], 0.5, 3, 1)
    filed = rb.job_record(job, jobs.Outcome(0, "", "", 0.6), record, "abc1234")
    assert set(score.RECORD_FIELDS) <= set(filed)
    assert filed["status"] == rb.STATUS_OK
    assert record["engine"] == "pyvrp" and record["threads"] == 1
    assert record["feasible"] is True and record["verified"] is True
    assert record["commit_sha"] == "abc1234"
    verdict = verify(INST_DIR / "X" / "X-n101-k25.vrp", sol, record["objective"])
    assert verdict.ok, verdict.errors


def test_the_budget_and_ttff_include_pyvrps_construction_phase() -> None:
    """PyVRP's own clocks start after its exhaustive start-solution search."""
    inst = read_vrp_file(INST_DIR / "X" / "X-n1001-k43.vrp")
    data = ref.problem_data(inst)
    started = time.monotonic()
    result = pyvrp.solve(data, stop=ref.Deadline(0.3), seed=1, collect_stats=True)
    wall = time.monotonic() - started
    setup = wall - result.runtime
    assert setup > 0.01  # the construction phase PyVRP's own clocks leave out
    # The deadline covers it: MaxRuntime would give the search itself >= 0.3 s.
    assert result.runtime < 0.3
    assert ref.time_to_first_feasible(result, wall) >= setup


def test_reference_solve_refuses_more_than_one_thread(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    rc = ref.main(["--instance", "X-n101-k25", "--threads", "2", "--out", str(out)])
    assert rc == 2
    assert not out.exists()
