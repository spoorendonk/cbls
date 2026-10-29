"""Tests for the CVRP campaign tooling (#199): driver wiring, verifier, scorer.

The `benchmarks/common` mechanics are pinned in `test_benchmark_common.py` and
not re-tested here. What is pinned is this driver's use of them -- which jobs
resume, what a killed job records, which timeout a call carries -- plus the
independent verifier against hand-corrupted solutions and the scorer's tables
and refusals. No solver runs: the driver is driven with a stand-in runner.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

from benchmarks.common import jobs
from benchmarks.cvrp import run_benchmark as rb
from benchmarks.cvrp import score
from benchmarks.cvrp.verify_solution import (
    check_routes,
    distance_matrix,
    nint_distance,
    read_vrp,
    read_vrp_file,
    verify,
)
from benchmarks.instances.cvrp.download import parse_sol, read_roster

INST_DIR = rb.DEFAULT_INST_DIR
SMOKE = [e.instance for e in read_roster(INST_DIR / "smoke.csv")]
BKS = {e.instance: e.bks for e in read_roster(INST_DIR / "roster.csv")}
GUIDED = rb.ARMS["guided"]
DEFAULT = rb.ARMS["default"]


# --------------------------------------------------------------------------
# Verifier
# --------------------------------------------------------------------------


@pytest.mark.parametrize("instance", SMOKE)
def test_verifier_accepts_every_smoke_bks(instance: str) -> None:
    verdict = verify(INST_DIR / "X" / f"{instance}.vrp", INST_DIR / "X" / f"{instance}.sol")
    assert verdict.ok, verdict.errors
    assert verdict.cost == BKS[instance]


def _bks_routes() -> tuple[list[list[int]], float]:
    sol = parse_sol((INST_DIR / "X" / "X-n101-k25.sol").read_text())
    return [list(r) for r in sol.routes], sol.cost


def _check(routes: list[list[int]], cost: float, expected: float | None = None) -> list[str]:
    inst = read_vrp_file(INST_DIR / "X" / "X-n101-k25.vrp")
    return check_routes(inst, routes, cost, expected).errors


def test_verifier_rejects_a_duplicate_customer() -> None:
    routes, cost = _bks_routes()
    routes[1].append(routes[0][0])
    assert any("more than once" in e for e in _check(routes, cost))


def test_verifier_rejects_a_missing_customer() -> None:
    routes, cost = _bks_routes()
    routes[0].pop()
    assert any("not visited" in e for e in _check(routes, cost))


def test_verifier_rejects_an_overload() -> None:
    routes, cost = _bks_routes()
    merged = [routes[0] + routes[1], *routes[2:]]
    assert any("capacity" in e for e in _check(merged, cost))


def test_verifier_rejects_a_wrong_cost() -> None:
    routes, cost = _bks_routes()
    assert any("the file's Cost" in e for e in _check(routes, cost + 1))
    assert any("the record's objective" in e for e in _check(routes, cost, cost - 1))


@pytest.mark.parametrize("bad", [0, 101, -3])
def test_verifier_rejects_the_depot_or_an_unknown_id(bad: int) -> None:
    routes, cost = _bks_routes()
    routes[0].append(bad)
    assert any("outside 1..100" in e for e in _check(routes, cost))


def test_verifier_does_not_cap_the_fleet() -> None:
    """X-n101-k25's BKS uses 26 routes: the name's k is a minimum."""
    routes, cost = _bks_routes()
    assert len([r for r in routes if r]) == 26
    assert _check(routes, cost) == []


def test_nint_rounds_halves_up_not_to_even() -> None:
    assert nint_distance((0.0, 0.0), (0.5, 0.0)) == 1
    assert nint_distance((0.0, 0.0), (2.5, 0.0)) == 3


def test_distance_matrix_is_square_symmetric_and_zero_on_the_diagonal() -> None:
    inst = read_vrp_file(INST_DIR / "X" / "X-n101-k25.vrp")
    d = distance_matrix(inst)
    assert len(d) == inst.customers + 1 == 101
    assert all(d[i][i] == 0 for i in range(101))
    assert all(d[i][j] == d[j][i] for i in range(101) for j in range(i))


def test_read_vrp_refuses_a_duplicate_node_and_a_missing_header() -> None:
    text = (INST_DIR / "X" / "X-n101-k25.vrp").read_text()
    doubled = text.replace("DEMAND_SECTION", "2\t1\t1\nDEMAND_SECTION", 1)
    with pytest.raises(ValueError, match="twice"):
        read_vrp(doubled)
    with pytest.raises(ValueError, match="CAPACITY"):
        read_vrp("\n".join(ln for ln in text.splitlines() if not ln.startswith("CAPACITY")))


def test_read_vrp_refuses_another_metric() -> None:
    text = (INST_DIR / "X" / "X-n101-k25.vrp").read_text().replace("EUC_2D", "GEO")
    with pytest.raises(ValueError, match="EUC_2D"):
        read_vrp(text)


def test_an_unreadable_sol_is_a_failed_verdict(tmp_path: Path) -> None:
    sol = tmp_path / "x.sol"
    sol.write_text("Route #1: 1 2\n")
    verdict = verify(INST_DIR / "X" / "X-n101-k25.vrp", sol)
    assert not verdict.ok


# --------------------------------------------------------------------------
# Driver wiring
# --------------------------------------------------------------------------


def _job(**kw: object) -> rb.Job:
    fields: dict[str, object] = {
        "instance": "X-n101-k25",
        "arm": GUIDED,
        "budget": 10.0,
        "seed": 1,
        "threads": 1,
    }
    fields.update(kw)
    return rb.Job(**fields)  # type: ignore[arg-type]


def test_record_path_is_the_resume_key(tmp_path: Path) -> None:
    job = _job(budget=2.5, seed=7, threads=4)
    assert rb.record_path(job, tmp_path) == tmp_path / "cbls/guided/t4/b2.5/X-n101-k25.s7.json"
    assert rb.solution_path(job, tmp_path).name == "X-n101-k25.s7.sol"


def test_a_job_with_its_record_is_skipped_and_others_run(tmp_path: Path) -> None:
    job = _job()
    assert rb.needs_run(job, tmp_path, force=False, retry_failed=False)
    rec = {**job.key(), "status": rb.STATUS_OK}
    path = rb.record_path(job, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec))
    assert not rb.needs_run(job, tmp_path, force=False, retry_failed=False)
    assert rb.needs_run(job, tmp_path, force=True, retry_failed=False)
    # Another seed, budget or arm is another job.
    for other in (_job(seed=2), _job(budget=20.0), _job(arm=DEFAULT), _job(threads=2)):
        assert rb.needs_run(other, tmp_path, force=False, retry_failed=False)


def test_a_record_naming_another_job_does_not_count(tmp_path: Path) -> None:
    job = _job()
    path = rb.record_path(job, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**job.key(), "seed": 9, "status": rb.STATUS_OK}))
    assert rb.needs_run(job, tmp_path, force=False, retry_failed=False)


def test_a_torn_record_is_rerun(tmp_path: Path) -> None:
    job = _job()
    path = rb.record_path(job, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text('{"instance": "X-n1')
    assert rb.needs_run(job, tmp_path, force=False, retry_failed=False)


def test_a_failed_job_is_done_until_retry_failed(tmp_path: Path) -> None:
    job = _job()
    path = rb.record_path(job, tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({**job.key(), "status": rb.STATUS_TIMEOUT}))
    assert not rb.needs_run(job, tmp_path, force=False, retry_failed=False)
    assert rb.needs_run(job, tmp_path, force=False, retry_failed=True)


def test_plan_runs_the_arms_of_one_instance_and_seed_back_to_back() -> None:
    plan = rb.plan_jobs(["A", "B"], [DEFAULT, GUIDED], [10.0], [1, 2], [1])
    assert [(j.instance, j.seed, j.arm.name) for j in plan[:4]] == [
        ("A", 1, "default"),
        ("A", 1, "guided"),
        ("B", 1, "default"),
        ("B", 1, "guided"),
    ]
    assert len(plan) == 8


def test_a_killed_job_records_a_timeout_and_its_key() -> None:
    job = _job()
    outcome = jobs.Outcome(-9, "", "", 70.0, timed_out=True)
    runner: dict[str, object] = {"objective": 1.0, "feasible": True}
    rec = rb.job_record(job, outcome, runner, "abc1234")
    assert rec["status"] == rb.STATUS_TIMEOUT
    assert rec["feasible"] is False and rec["objective"] is None
    assert {k: rec[k] for k in job.key()} == job.key()
    assert rec["commit_sha"] == "abc1234"
    assert set(score.RECORD_FIELDS) <= set(rec)


def test_a_crash_with_no_runner_json_records_a_failure() -> None:
    rec = rb.job_record(_job(), jobs.Outcome(-11, "", "", 1.0), None, "abc1234")
    assert rec["status"] == rb.STATUS_FAILED
    assert rec["returncode"] == -11


def test_a_nonzero_exit_with_runner_json_is_failed_but_keeps_the_fields() -> None:
    runner: dict[str, object] = {
        "objective": 5.0,
        "feasible": True,
        "workers_completed": 1,
        "threads": 2,
    }
    rec = rb.job_record(_job(threads=2), jobs.Outcome(1, "", "", 1.0), runner, "x")
    assert rec["status"] == rb.STATUS_FAILED
    assert rec["workers_completed"] == 1


#: A stand-in for `cbls_cvrp`: writes a runner JSON and a `.sol` from argv,
#: or sleeps past its timeout when the instance is called `SLOW`.
FAKE_RUNNER = """\
import json, sys, time
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
if a["--instance"] == "SLOW":
    time.sleep(30)
json.dump({"engine": "cbls", "instance": a["--instance"], "objective": 1.0,
           "feasible": True, "time_to_first_feasible": 0.1,
           "commit_sha": a["--commit"], "argv": sys.argv[1:]}, open(a["--out"], "w"))
open(a["--solution-out"], "w").write("Route #1: 1\\nCost 1\\n")
"""


def _fake_binary(tmp_path: Path) -> Path:
    script = tmp_path / "fake_runner.py"
    script.write_text(FAKE_RUNNER)
    binary = tmp_path / "cbls_cvrp"
    binary.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    binary.chmod(0o755)
    return binary


def _args(results: Path, **kw: object) -> argparse.Namespace:
    args = rb.parse_args(["--results", str(results)])
    for k, v in kw.items():
        setattr(args, k, v)
    return args


def test_run_job_passes_the_budget_timeout_cap_and_own_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    def fake_run(command: list[str], **kw: object) -> jobs.Outcome:
        seen.update(kw, command=command)
        return jobs.Outcome(3, "", "", 0.1)

    monkeypatch.setattr(jobs, "run_process", fake_run)
    job = _job(budget=12.0, seed=3, threads=2)
    _line, ok = rb.run_job(job, Path("/bin/cbls_cvrp"), _args(tmp_path), "abc1234")
    assert not ok
    assert seen["timeout"] == 12.0 + rb.TIMEOUT_MARGIN_SECONDS
    assert seen["own_session"] is True
    assert seen["mem_limit_gb"] == rb.DEFAULT_MEM_LIMIT_GB
    assert seen["log"] == rb.log_path(job, tmp_path)
    command = seen["command"]
    assert isinstance(command, list)
    for flag, value in (("--budget", "12"), ("--seed", "3"), ("--threads", "2")):
        assert command[command.index(flag) + 1] == value
    assert command[-4:] == list(GUIDED.flags)


def test_run_job_end_to_end_with_a_stand_in_runner(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    results = tmp_path / "results"
    job = _job()
    _line, ok = rb.run_job(job, binary, _args(results), "abc1234")
    assert ok
    rec = json.loads(rb.record_path(job, results).read_text())
    assert rec["status"] == rb.STATUS_OK and rec["arm"] == "guided"
    assert rec["arm_flags"] == list(GUIDED.flags)
    assert rb.solution_path(job, results).exists()
    assert not rb.needs_run(job, results, force=False, retry_failed=False)


def test_a_job_killed_by_its_timeout_leaves_a_timeout_record_and_no_old_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = _fake_binary(tmp_path)
    results = tmp_path / "results"
    job = _job(instance="SLOW", budget=0.5)
    # A previous run's files must not outlive this one's kill.
    stale = rb.solution_path(job, results)
    stale.parent.mkdir(parents=True)
    stale.write_text("old")
    monkeypatch.setattr(rb, "TIMEOUT_MARGIN_SECONDS", 0.5)
    _line, ok = rb.run_job(job, binary, _args(results), "abc1234")
    assert not ok
    rec = json.loads(rb.record_path(job, results).read_text())
    assert rec["status"] == rb.STATUS_TIMEOUT
    assert not stale.exists()


@pytest.mark.parametrize(
    ("arm", "script"), [("pyvrp", "reference_solve.py"), ("ortools", "ortools_solve.py")]
)
def test_a_baseline_arm_runs_its_script_with_the_same_flags(
    tmp_path: Path, arm: str, script: str
) -> None:
    job = _job(arm=rb.ARMS[arm])
    command = rb.build_command(job, Path("/bin/cbls_cvrp"), INST_DIR, tmp_path, "abc1234")
    assert command[0] == sys.executable
    assert Path(command[1]).name == script
    assert command[command.index("--threads") + 1] == "1"
    assert rb.record_path(job, tmp_path).parts[-5:-3] == (arm, arm)


@pytest.mark.parametrize("arm", ["pyvrp", "ortools"])
def test_a_baseline_arm_is_single_threaded_only(tmp_path: Path, arm: str) -> None:
    complaint = rb.check_arguments(_args(tmp_path, arms=[arm, "default"], threads=[1, 4]))
    assert complaint is not None and "single-threaded" in complaint
    assert rb.check_arguments(_args(tmp_path, arms=[arm, "default"], threads=[1])) is None


@pytest.mark.parametrize(
    "overrides",
    [{"seeds": [1, 1]}, {"arms": ["guided", "guided"]}, {"budget": [10.0, 10.0]}],
)
def test_a_repeated_value_is_refused(tmp_path: Path, overrides: dict[str, object]) -> None:
    complaint = rb.check_arguments(_args(tmp_path, **overrides))
    assert complaint is not None and "repeats" in complaint


def test_a_budget_the_path_cannot_spell_is_refused(tmp_path: Path) -> None:
    complaint = rb.check_arguments(_args(tmp_path, budget=[1234567.0]))
    assert complaint is not None and "%g" in complaint


def test_jobs_above_one_need_not_comparable(tmp_path: Path) -> None:
    complaint = rb.check_arguments(_args(tmp_path, jobs=4))
    assert complaint is not None and "--not-comparable" in complaint
    assert rb.check_arguments(_args(tmp_path, jobs=4, not_comparable=True)) is None
    assert rb.check_arguments(_args(tmp_path, arms=["nope"])) is not None


def test_a_directory_from_another_commit_is_refused(tmp_path: Path) -> None:
    (tmp_path / rb.STAMP_NAME).write_text(rb.stamp_text("aaaaaaa", comparable=True))
    rc = rb.execute(_args(tmp_path), "bbbbbbb", Path("/nonexistent"))
    assert rc == 2


def test_the_machine_record_is_written_once(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    results = tmp_path / "results"
    args = _args(results, roster="smoke", arms=["guided"], inst_dir=_one_instance_dir(tmp_path))
    assert rb.execute(args, "abc1234", binary) == 0
    first = (results / rb.MACHINE_NAME).read_text()
    assert rb.execute(args, "abc1234", binary) == 0
    assert (results / rb.MACHINE_NAME).read_text() == first
    assert rb.stamp_comparable(results)


def _one_instance_dir(tmp_path: Path) -> Path:
    inst = tmp_path / "inst"
    inst.mkdir()
    (inst / "smoke.csv").write_text("instance,bks,bks_kind,bks_routes\nX-n101-k25,1,opt,1\n")
    return inst


# --------------------------------------------------------------------------
# Scorer
# --------------------------------------------------------------------------


def _write_run(
    results: Path,
    instance: str,
    *,
    arm: rb.Arm = GUIDED,
    seed: int = 1,
    budget: float = 10.0,
    threads: int = 1,
    objective: float | None = None,
    sol: str | None = None,
    **extra: object,
) -> Path:
    """A record as the driver would leave it, its `.sol` the instance's BKS."""
    job = rb.Job(instance, arm, budget, seed, threads)
    path = rb.record_path(job, results)
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        **job.key(),
        "engine": "cbls",
        "status": rb.STATUS_OK,
        "feasible": True,
        "objective": BKS[instance] if objective is None else objective,
        "time_to_first_feasible": 0.5,
        "commit_sha": "abc1234",
        "workers_completed": threads,
    }
    record.update(extra)
    path.write_text(json.dumps(record))
    sol_path = rb.solution_path(job, results)
    if sol is None:
        shutil.copy(INST_DIR / "X" / f"{instance}.sol", sol_path)
    else:
        sol_path.write_text(sol)
    return path


def _worse_solution(instance: str) -> tuple[str, int]:
    """The BKS with its first route split in two: still valid, strictly costlier."""
    sol = parse_sol((INST_DIR / "X" / f"{instance}.sol").read_text())
    first = next(r for r in sol.routes if len(r) >= 2)
    routes = [first[:1], first[1:], *(r for r in sol.routes if r is not first)]
    inst = read_vrp_file(INST_DIR / "X" / f"{instance}.vrp")
    cost = check_routes(inst, routes, 0.0).cost
    text = "".join(f"Route #{i}: {' '.join(map(str, r))}\n" for i, r in enumerate(routes, 1))
    return text + f"Cost {cost}\n", cost


def _stamped(results: Path, comparable: bool = True) -> Path:
    results.mkdir(parents=True, exist_ok=True)
    (results / rb.STAMP_NAME).write_text(rb.stamp_text("abc1234", comparable))
    return results


def _score(results: Path, *argv: object) -> str:
    args = argparse.Namespace(
        results=results,
        roster="smoke",
        budget=None,
        ab=None,
        allow_partial=False,
        inst_dir=INST_DIR,
        out=None,
    )
    for flag, value in zip(argv[::2], argv[1::2], strict=True):
        setattr(args, str(flag), value)
    return score.score(args)


def test_score_builds_a_table_from_a_smoke_directory(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    text = _score(results)
    assert "| cbls/guided t1 | X-n101-k25 | 1 | 1 | 0.00 | 0.00 | 0.00 | 0.50 |" in text
    assert "| cbls/guided t1 | 800-1000 | 2 | 2 | 0.00 | 0.9 |" in text
    assert "NOT COMPARABLE" not in text and "PARTIAL" not in text


def test_score_refuses_a_mixed_budget_directory(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    _write_run(results, SMOKE[0], budget=60.0)
    with pytest.raises(score.ScoreError, match="budgets"):
        _score(results)
    assert "Budget 60 s" in _score(results, "budget", 60.0, "allow_partial", True)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"status": "timeout"}, "status"),
        ({"workers_completed": 1, "threads": 2}, "portfolio workers"),
    ],
)
def test_score_refuses_a_defective_row(
    tmp_path: Path, extra: dict[str, object], match: str
) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    path = _write_run(results, SMOKE[1], seed=2)
    rec = json.loads(path.read_text())
    rec.update(extra)
    path.write_text(json.dumps(rec))
    with pytest.raises(score.ScoreError, match=match):
        _score(results)


def test_score_refuses_a_row_failing_the_verifier(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    _write_run(results, SMOKE[0], seed=2, objective=1.0, sol="Route #1: 1\nCost 1\n")
    with pytest.raises(score.ScoreError, match="failed verification"):
        _score(results)


def test_score_refuses_a_feasible_row_without_its_sol(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    path = _write_run(results, SMOKE[0], seed=2)
    path.with_name(path.name.replace(".json", ".sol")).unlink()
    with pytest.raises(score.ScoreError, match="no .*sol"):
        _score(results)


def test_score_refuses_a_partial_roster_unless_bannered(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    _write_run(results, SMOKE[0])
    with pytest.raises(score.ScoreError, match="partial roster"):
        _score(results)
    assert "**PARTIAL**" in _score(results, "allow_partial", True)


def test_score_banners_a_not_comparable_directory(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r", comparable=False)
    for instance in SMOKE:
        _write_run(results, instance)
    assert "**NOT COMPARABLE**" in _score(results)


def test_score_refuses_the_published_table_for_a_non_default_arm(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    with pytest.raises(score.ScoreError, match="non-publishable"):
        _score(results, "out", score.PUBLISHED_TABLE)


def test_score_refuses_the_published_table_from_the_smoke_roster(tmp_path: Path) -> None:
    """Complete, clean and default: still not a table to publish."""
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance, arm=DEFAULT)
    with pytest.raises(score.ScoreError, match="smoke roster"):
        _score(results, "out", score.PUBLISHED_TABLE)


def test_score_refuses_the_published_table_across_thread_counts(tmp_path: Path) -> None:
    """Multi-thread CBLS is never set beside single-thread PyVRP (#200)."""
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance, arm=DEFAULT, threads=4)
        _write_run(results, instance, arm=rb.ARMS["pyvrp"], threads=1)
    with pytest.raises(score.ScoreError, match="thread"):
        _score(results, "roster", "smoke", "out", score.PUBLISHED_TABLE)


def test_score_refuses_the_published_table_from_a_modified_tree(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance, arm=DEFAULT, commit_sha="abc1234-dirty")
    assert "**MODIFIED TREE**" in _score(results)
    with pytest.raises(score.ScoreError, match="banners"):
        _score(results, "out", score.PUBLISHED_TABLE)


def test_a_modified_tree_stamps_the_directory_not_comparable(tmp_path: Path) -> None:
    binary = _fake_binary(tmp_path)
    results = tmp_path / "results"
    args = _args(results, roster="smoke", arms=["guided"], inst_dir=_one_instance_dir(tmp_path))
    assert rb.execute(args, "abc1234-dirty", binary) == 0
    assert not rb.stamp_comparable(results)


def test_score_treats_uneven_seed_sets_as_partial(tmp_path: Path) -> None:
    """A run killed between seeds: every instance present, not every seed."""
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance, seed=1)
    _write_run(results, SMOKE[0], seed=2)
    with pytest.raises(score.ScoreError, match=r"lacks seed\(s\) \[2\]"):
        _score(results)
    assert "**PARTIAL**" in _score(results, "allow_partial", True)


def test_score_refuses_a_row_whose_instance_file_is_missing(tmp_path: Path) -> None:
    inst = tmp_path / "inst"
    inst.mkdir()
    for name in ("smoke.csv", "manifest.csv"):
        shutil.copy(INST_DIR / name, inst / name)
    results = _stamped(tmp_path / "r")
    for instance in SMOKE:
        _write_run(results, instance)
    with pytest.raises(score.ScoreError, match="cannot read"):
        _score(results, "inst_dir", inst)


def test_score_refuses_the_published_table_from_a_not_comparable_run(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r", comparable=False)
    for instance in SMOKE:
        _write_run(results, instance, arm=DEFAULT)
    with pytest.raises(score.ScoreError, match="banners"):
        _score(results, "out", score.PUBLISHED_TABLE)


def _ab_directory(results: Path) -> dict[str, float]:
    """guided at the BKS, default on a worse valid solution, on three instances
    x two seeds. Returns each instance's guided-minus-default gap difference."""
    diffs: dict[str, float] = {}
    for instance in SMOKE[:3]:
        text, cost = _worse_solution(instance)
        diffs[instance] = -100.0 * (cost - BKS[instance]) / BKS[instance]
        for seed in (1, 2):
            _write_run(results, instance, arm=GUIDED, seed=seed)
            _write_run(results, instance, arm=DEFAULT, seed=seed, objective=cost, sol=text)
    return diffs


def test_paired_ab_differences_by_instance_and_seed(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    diffs = _ab_directory(results)
    text = _score(results, "allow_partial", True, "ab", ["guided", "default"])
    mean = sum(diffs.values()) / 3
    assert f"pairs: n=6 mean={mean:+.3f}" in text
    assert f"instances: n=3 mean={mean:+.3f}" in text
    assert "A better=3 worse=0 tied=0 sign-test p=0.250" in text


def test_paired_ab_counts_a_pair_with_an_infeasible_side(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    _ab_directory(results)
    path = rb.record_path(rb.Job(SMOKE[0], DEFAULT, 10.0, 1, 1), results)
    rec = json.loads(path.read_text())
    rec.update(feasible=False, objective=None)
    path.write_text(json.dumps(rec))
    text = _score(results, "allow_partial", True, "ab", ["guided", "default"])
    assert "1 pair(s) where at least one side was infeasible" in text
    assert "pairs: n=5" in text


def test_ab_labels_name_the_engine_when_arms_collide() -> None:
    groups = [("cbls", "default", 1), ("pyvrp", "default", 1), ("cbls", "default", 4)]
    assert score.parse_group("pyvrp/default", groups) == ("pyvrp", "default", 1)
    assert score.parse_group("cbls/default@4", groups) == ("cbls", "default", 4)
    with pytest.raises(score.ScoreError, match="engine/arm@threads"):
        score.parse_group("default@1", groups)


def test_paired_ab_refuses_different_thread_counts(tmp_path: Path) -> None:
    results = _stamped(tmp_path / "r")
    _write_run(results, SMOKE[0], arm=GUIDED, threads=1)
    _write_run(results, SMOKE[0], arm=DEFAULT, threads=4)
    with pytest.raises(score.ScoreError, match="thread counts"):
        _score(results, "allow_partial", True, "ab", ["guided", "default"])


def test_sign_test_is_two_sided_and_exact() -> None:
    assert score.sign_test_p(0, 0) == 1.0
    assert score.sign_test_p(5, 0) == pytest.approx(2 / 32)
    assert score.sign_test_p(3, 3) == 1.0
