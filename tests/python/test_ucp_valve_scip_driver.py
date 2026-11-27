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


def run_runner(
    tmp_path: Path, driver: Path, method: str, threads: int, limit: float, *extra: str
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(RUNNER), "--instance", "ucp13", "--periods", "1"]
    cmd += ["--method", method, "--backend", "driver", "--threads", str(threads)]
    cmd += ["--time-limit", str(limit), "--driver", str(driver)]
    cmd += ["--out", str(tmp_path / "out.jsonl"), *extra]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)


def run_driver(
    tmp_path: Path, driver: Path, method: str, threads: int, limit: float = LIMIT_S, *extra: str
) -> dict[str, Any]:
    proc = run_runner(tmp_path, driver, method, threads, limit, *extra)
    assert proc.returncode == 0, proc.stderr
    record: dict[str, Any] = json.loads((tmp_path / "out.jsonl").read_text())
    return record


#: A stand-in for the driver: runs the real one with its arguments, then `after`
#: (Python with `args`, `start`, `limit` and `rc` bound), then exits with its code.
WRAPPER = """#!{python}
import json, subprocess, sys, time
args = sys.argv[1:]
start = time.monotonic()
limit = float(args[args.index("--time-limit") + 1])
rc = subprocess.run([{driver!r}, *args], check=False).returncode
{after}
sys.exit(rc)
"""


def wrap(tmp_path: Path, driver: Path, after: str) -> Path:
    path = tmp_path / "wrapped_driver"
    path.write_text(WRAPPER.format(python=sys.executable, driver=str(driver), after=after))
    path.chmod(0o755)
    return path


# Sleeps as a long SCIPfree would, after the driver has written its result.
IN_FREE = "time.sleep(60)"
# Moves the last incumbent past the budget, as SCIP's overshoot would.
LATE_INCUMBENT = """
traj = args[args.index("--traj-out") + 1]
recs = [json.loads(line) for line in open(traj) if line.strip()]
last = max((i for i, r in enumerate(recs) if r["objective"] is not None), default=None)
if last is not None:
    recs[last]["mono_s"] = start + limit + 5.0
open(traj, "w").writelines(json.dumps(r) + "\\n" for r in recs)
time.sleep(max(start + limit + 5.5 - time.monotonic(), 0.0))  # no find after the return
"""
# Returns past the budget, within the kill grace.
LATE_RETURN = "time.sleep(max(start + limit + 1.0 - time.monotonic(), 0.0))"


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


@pytest.mark.parametrize("salvage", [False, True])
@pytest.mark.parametrize("method", ["exact", "pwl"])
def test_a_driver_killed_after_its_result_counts_only_with_salvage_killed(
    tmp_path: Path, scip_driver: Path, method: str, salvage: bool
) -> None:
    driver = wrap(tmp_path, scip_driver, IN_FREE)
    extra = ["--driver-grace", "1"] + (["--salvage-killed"] if salvage else [])
    proc = run_runner(tmp_path, driver, method, 1, 5.0, *extra)
    if not salvage:
        assert proc.returncode != 0
        assert "killed" in proc.stderr
        return
    assert proc.returncode == 0, proc.stderr
    r = json.loads((tmp_path / "out.jsonl").read_text())
    assert r["driver_killed_in_free"] >= 1
    # Killed past the budget, but SCIP found the solution within it.
    assert r["ub"] is not None and r["check"] == []


def test_exact_ub_counts_only_incumbents_within_the_budget(
    tmp_path: Path, scip_driver: Path
) -> None:
    limit = 5.0
    r = run_driver(tmp_path, wrap(tmp_path, scip_driver, LATE_INCUMBENT), "exact", 1, limit)
    assert len(r["trajectory_after_budget"]) == 1
    assert r["trajectory_after_budget"][0][0] > limit
    assert all(t <= limit for t, _ in r["trajectory"])
    # The best solution is kept apart; ub is the last in-budget point's SCIP objective.
    late = r["trajectory_after_budget"][0][1]
    assert r["ub_after_budget"] <= late + 1e-6 * abs(late)
    assert r["check_after_budget"] == [] and "check" not in r
    assert r["ub_is_scip_objective"] is True and r["status"] == "timelimit"
    assert r["ub"] == (r["trajectory"][-1][1] if r["trajectory"] else None)


def test_pwl_ub_found_only_after_the_budget_is_not_the_runs(
    tmp_path: Path, scip_driver: Path
) -> None:
    limit = 5.0
    r = run_driver(tmp_path, wrap(tmp_path, scip_driver, LATE_INCUMBENT), "pwl", 1, limit)
    assert r["iterations"] and all(it["found_s"] > limit for it in r["iterations"])
    assert all(it["in_budget"] is False for it in r["iterations"])
    assert (r["ub"], r["trajectory"], r["converged"]) == (None, [], False)
    # LBs are SCIP's at its stop, late or not.
    assert r["lb"] == max(it["lb"] for it in r["iterations"])
    assert r["status"] == "time limit reached" and r["out_of_budget"] is True


def test_pwl_ub_found_within_the_budget_counts_though_returned_after_it(
    tmp_path: Path, scip_driver: Path
) -> None:
    limit = 5.0
    r = run_driver(tmp_path, wrap(tmp_path, scip_driver, LATE_RETURN), "pwl", 1, limit)
    first = r["iterations"][0]
    assert first["wall_s"] > limit >= first["found_s"] and first["in_budget"] is True
    assert r["ub"] == first["ub"] and r["check"] == []
    # Stamped at the find, so a score over [0, limit] sees it.
    assert r["trajectory"] == [[first["found_s"], first["ub"]]]
    assert r["converged"] is False and r["out_of_budget"] is True
