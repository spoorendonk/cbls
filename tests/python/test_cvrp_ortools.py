"""Tests for the OR-Tools routing CVRP baseline (#204).

As for PyVRP (`test_cvrp_reference.py`), the baseline must solve the instance
the verifier scores: its arc costs are checked equal to the shared reader's
matrix, and its solution file is re-checked by `verify_solution.verify`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

pytest.importorskip("ortools")

from benchmarks.common import jobs  # noqa: E402
from benchmarks.cvrp import ortools_solve as ort  # noqa: E402
from benchmarks.cvrp import run_benchmark as rb  # noqa: E402
from benchmarks.cvrp import score  # noqa: E402
from benchmarks.cvrp.verify_solution import distance_matrix, read_vrp_file, verify  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

INST_DIR = ort.DEFAULT_INST_DIR


def test_ortools_costs_the_arcs_the_verifier_scores_with() -> None:
    inst = read_vrp_file(INST_DIR / "X" / "X-n101-k25.vrp")
    routing = ort.build_routing(inst, fleet=2)
    dist = distance_matrix(inst)
    manager, model = routing.manager, routing.model
    # Arc costs are only readable from a closed model (an open one segfaults).
    model.CloseModel()
    for a, b in [(0, 1), (1, 0), (5, 77), (100, 3), (42, 42)]:
        cost = model.GetArcCostForVehicle(manager.NodeToIndex(a), manager.NodeToIndex(b), 0)
        assert cost == dist[a][b]
    # Node c is CVRPLIB customer c, with its demand.
    load = model.GetDimensionOrDie("load")
    assert load.GetUnaryTransitEvaluator(0) is not None
    assert model.vehicles() == 2


@pytest.mark.parametrize(
    ("instance", "k", "fleet"),
    [("X-n101-k25", 25, 28), ("X-n1001-k43", 43, 48), ("X-n106-k14", 14, 17)],
)
def test_the_fleet_is_cbls_cvrps_default(instance: str, k: int, fleet: int) -> None:
    """`k + max(3, ceil(0.1 k))`, `cvrp_model.h`'s `default_fleet`."""
    assert ort.name_k(instance) == k
    assert ort.default_fleet(k) == fleet


def test_a_name_without_k_is_refused() -> None:
    assert ort.name_k("X-n101") is None
    assert ort.main(["--instance", "X-n101", "--budget", "1"]) == 2


def test_ortools_is_single_threaded_only() -> None:
    assert ort.main(["--instance", "X-n101-k25", "--threads", "2"]) == 2


def test_ortools_solve_writes_a_record_and_sol_the_verifier_accepts(tmp_path: Path) -> None:
    out, sol = tmp_path / "r.json", tmp_path / "r.sol"
    rc = ort.main(
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
    job = rb.Job("X-n101-k25", rb.ARMS["ortools"], 0.5, 3, 1)
    filed = rb.job_record(job, jobs.Outcome(0, "", "", 0.6), record, "abc1234")
    assert set(score.RECORD_FIELDS) <= set(filed)
    assert filed["status"] == rb.STATUS_OK
    assert record["engine"] == "ortools" and record["threads"] == 1
    assert record["feasible"] is True and record["verified"] is True
    assert record["fleet"] == 28 and record["routes_used"] <= 28
    assert record["solutions_found"] >= 1
    ttff = record["time_to_first_feasible"]
    assert ttff is not None and 0 < ttff <= record["wall_seconds"]
    verdict = verify(INST_DIR / "X" / "X-n101-k25.vrp", sol, record["objective"])
    assert verdict.ok, verdict.errors
