"""ucp_valve_scip.py's `--backend driver` (#442) on ucp13, one period: seconds a case.

The runner is GPL and nothing outside its directory imports it, so it runs as a
child process. Skipped unless `cbls_scip_traj` is built (the `scip_driver` fixture).
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("pyscipopt")

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks" / "uc-chped" / "pedroso" / "ucp_valve_scip.py"
SCIP_COMMIT = ROOT / "benchmarks" / "common" / "scip_driver" / "SCIP_COMMIT"
LIMIT_S = 30.0


def run_driver(tmp_path: Path, driver: Path, method: str, threads: int) -> dict[str, Any]:
    out = tmp_path / "out.jsonl"
    cmd = [sys.executable, str(RUNNER), "--instance", "ucp13", "--periods", "1"]
    cmd += ["--method", method, "--backend", "driver", "--threads", str(threads)]
    cmd += ["--time-limit", str(LIMIT_S), "--driver", str(driver), "--out", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    record: dict[str, Any] = json.loads(out.read_text())
    return record


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("method", ["exact", "pwl"])
def test_driver_trajectory_ends_at_the_runs_bound_within_its_limit(
    tmp_path: Path, scip_driver: Path, method: str, threads: int
) -> None:
    r = run_driver(tmp_path, scip_driver, method, threads)
    assert (r["backend"], r["threads"], r["time_limit_s"]) == ("driver", threads, LIMIT_S)
    assert SCIP_COMMIT.read_text().strip().startswith(r["scip_git_hash"])
    assert r["check"] == []
    times = [t for t, _ in r["trajectory"]]
    objs = [obj for _, obj in r["trajectory"]]
    assert times, "no incumbent logged"
    assert times == sorted(times)
    assert times[0] >= 0.0 and times[-1] <= r["wall_s"] <= LIMIT_S
    assert all(a > b for a, b in zip(objs, objs[1:], strict=False)), "not the running best"
    if method == "exact":
        # SCIP's objective, an upper bound on each incumbent's true cost.
        assert objs[-1] == pytest.approx(r["scip_objective"], rel=1e-12)
        assert objs[-1] >= r["ub"] - 1e-6 * abs(r["ub"])
    else:
        assert objs[-1] == pytest.approx(r["ub"], rel=1e-12)
        # Only a start completed by the driver (N > 1) is reported; the first
        # iteration has none. completesol may fail at SCIP's defaults (False).
        completed = [it["start_completed"] for it in r["iterations"] if "ub" in it]
        assert completed[0] is None
        assert all(isinstance(c, bool) if threads > 1 else c is None for c in completed[1:])
