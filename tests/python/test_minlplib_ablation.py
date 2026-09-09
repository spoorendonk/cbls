"""Unit tests for the MINLPLib ablation campaign driver (issue #143).

Every test here is offline. The campaign itself is ten hours of solving, so what
is checkable without solving anything is exactly what the driver is: a run
order, a set of refusals, and a resume rule. `execute_runs` is exercised against
a fake runner, so even its loop costs no solve.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib.ablation_report import CONTROL_ARM, PROBE_ARM_NAME
from benchmarks.minlplib.run_ablation import (
    ARMS,
    GATED_ARM,
    LOCK_NAME,
    MAX_LOAD_AVERAGE,
    PROBE_ARM,
    RESULT_COLUMNS,
    RESULTS_NAME,
    STAMP_NAME,
    Arm,
    Run,
    campaign_lock,
    campaign_plan,
    campaign_stamp,
    decide_lns_gate,
    estimate_hours,
    execute_runs,
    load_refusal,
    probe_plan,
    read_runner_row,
    recorded_keys,
    repair_torn_tail,
    runner_command,
    scratch_refusal,
    stamp_conflict,
    usage_error,
)
from benchmarks.minlplib.run_benchmark import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

RUNNER_HEADER = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,"
    "wall_seconds,feasible,note,commit_sha,max_violation,n_int_vars,lns_repairs,search_config"
)
DEFAULT_ARM_CELL = (
    "float_hook=on;lns=on;lns_interval=3;compound_moves=off;novelty_prob=0.5;"
    "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on"
)


def make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "out_dir": tmp_path / "scratch",
        "inst_dir": tmp_path / "inst",
        "build_dir": tmp_path / "build",
        "time_limit": 60.0,
        "seeds": [1, 2, 3],
        "instances": [],
        "lns_arm": "auto",
        "allow_busy": False,
        "resume": True,
        "build": True,
        "dry_run": False,
        "report_only": False,
        "merge": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class FakeCompleted:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = "runner tally\n"
        self.stderr = ""


def fake_runner(
    *, repairs: int = 0, returncode: int = 0, seen: list[str] | None = None
) -> Callable[..., FakeCompleted]:
    """Stand in for `subprocess.run(cbls_minlplib ...)` without solving anything."""

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        name = cmd[cmd.index("--instance") + 1]
        sha = cmd[cmd.index("--commit") + 1]
        out = Path(cmd[cmd.index("--out") + 1])
        if seen is not None:
            seen.append(out.stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            f"{RUNNER_HEADER}\n"
            f"{name},1,1,1,5,5,60,true,feasible,{sha},0,0,{repairs},{DEFAULT_ARM_CELL}\n"
        )
        return FakeCompleted(returncode)

    return run


# --- the interleaving contract -------------------------------------------------


def test_the_plan_finishes_every_arm_and_seed_on_one_instance_before_the_next() -> None:
    """Instance-major is the protocol: drift between a control and its arm is
    minutes, not the hour a whole roster pass would put between them."""
    plan = campaign_plan(["a", "b"], [1, 2], [Arm("ctl", ()), Arm("x", ("--x",))])
    assert [(r.instance, r.arm.name, r.seed) for r in plan] == [
        ("a", "ctl", 1),
        ("a", "x", 1),
        ("a", "ctl", 2),
        ("a", "x", 2),
        ("b", "ctl", 1),
        ("b", "x", 1),
        ("b", "ctl", 2),
        ("b", "x", 2),
    ]


def test_the_plan_visits_every_triple_exactly_once() -> None:
    plan = campaign_plan(["a", "b", "c"], [1, 2, 3], [*ARMS, GATED_ARM])
    keys = [r.key for r in plan]
    assert len(keys) == 3 * 3 * 5
    assert len(set(keys)) == len(keys)


def test_the_arms_rotate_innermost_so_a_paired_comparison_is_adjacent() -> None:
    """The control and any one arm at the same seed are never separated by
    another seed -- the tightest pairing this campaign can produce."""
    plan = campaign_plan(["a"], [1, 2, 3], list(ARMS))
    positions = {(r.arm.name, r.seed): i for i, r in enumerate(plan)}
    for arm in ARMS[1:]:
        for seed in (1, 2, 3):
            assert abs(positions[(arm.name, seed)] - positions[(CONTROL_ARM, seed)]) < len(ARMS)


def test_the_gate_probe_is_one_control_pass_at_one_seed() -> None:
    plan = probe_plan(["a", "b"], 7)
    assert [(r.instance, r.arm.name, r.seed) for r in plan] == [
        ("a", PROBE_ARM_NAME, 7),
        ("b", PROBE_ARM_NAME, 7),
    ]
    assert PROBE_ARM.flags == ()  # the probe IS the control configuration


# --- serial enforcement --------------------------------------------------------


def test_a_second_campaign_on_the_same_out_dir_is_refused(tmp_path: Path) -> None:
    """Timed comparisons must never share the machine, and the driver says so
    itself rather than trusting the caller to remember."""
    out_dir = tmp_path / "scratch"
    with campaign_lock(out_dir):
        assert (out_dir / LOCK_NAME).exists()
        with pytest.raises(RuntimeError, match="another campaign holds"), campaign_lock(out_dir):
            pass


def test_the_lock_is_released_when_the_campaign_ends(tmp_path: Path) -> None:
    out_dir = tmp_path / "scratch"
    with campaign_lock(out_dir):
        pass
    with campaign_lock(out_dir):
        pass  # a clean second acquisition, so the first really let go


def test_execute_runs_never_has_two_solves_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serial by construction: a blocking call in a plain loop. The fake asserts
    it is never re-entered, which is what a thread pool or a `&` would break."""
    inflight: list[str] = []

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        assert not inflight, f"a second solve started while {inflight} was running"
        inflight.append(cmd[cmd.index("--instance") + 1])
        result = fake_runner()(cmd, **kwargs)
        inflight.pop()
        return result

    monkeypatch.setattr(subprocess, "run", run)
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    execute_runs(
        make_args(tmp_path),
        "abc1234",
        campaign_plan(["a", "b"], [1], list(ARMS)),
        out_dir,
        "campaign",
    )
    assert not inflight


def test_a_busy_machine_is_refused_and_the_refusal_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("os.getloadavg", lambda: (MAX_LOAD_AVERAGE + 1.0, 0.0, 0.0))
    refusal = load_refusal(allow_busy=False)
    assert refusal is not None
    assert "load average" in refusal
    assert load_refusal(allow_busy=True) is None
    monkeypatch.setattr("os.getloadavg", lambda: (0.0, 0.0, 0.0))
    assert load_refusal(allow_busy=False) is None


# --- the published tables are out of reach -------------------------------------


@pytest.mark.parametrize(
    "relative",
    [
        "benchmarks/instances",
        "benchmarks/instances/minlplib",
        "benchmarks/instances/minlplib/scratch",
    ],
)
def test_an_out_dir_that_could_reach_a_published_table_is_refused(relative: str) -> None:
    """Refused, not warned: the campaign writes several files and the runner
    writes more underneath, so the durable rule is that the whole tree is off
    limits rather than that three filenames are."""
    refusal = scratch_refusal(REPO_ROOT / relative)
    assert refusal is not None
    assert "published" in refusal


@pytest.mark.parametrize("relative", ["", "benchmarks"])
def test_an_out_dir_containing_the_published_tables_is_refused(relative: str) -> None:
    refusal = scratch_refusal(REPO_ROOT / relative if relative else REPO_ROOT)
    assert refusal is not None


def test_a_scratch_out_dir_is_accepted(tmp_path: Path) -> None:
    assert scratch_refusal(tmp_path / "campaign") is None


def test_usage_error_carries_the_scratch_refusal(tmp_path: Path) -> None:
    args = make_args(tmp_path, out_dir=REPO_ROOT / "benchmarks" / "instances" / "minlplib")
    refusal = usage_error(args)
    assert refusal is not None
    assert "published" in refusal


@pytest.mark.parametrize("seeds", [[1], [1, 2]])
def test_fewer_than_three_seeds_is_refused(tmp_path: Path, seeds: list[int]) -> None:
    """The floor is measured from the control's across-seed spread, so a
    two-seed campaign cannot both estimate an effect and its noise."""
    refusal = usage_error(make_args(tmp_path, seeds=seeds))
    assert refusal is not None
    assert "at least three seeds" in refusal


def test_a_repeated_seed_is_refused(tmp_path: Path) -> None:
    refusal = usage_error(make_args(tmp_path, seeds=[1, 1, 2]))
    assert refusal is not None
    assert "distinct" in refusal


def test_the_runner_command_carries_the_arm_flags_the_seed_and_a_scratch_out(
    tmp_path: Path,
) -> None:
    cmd = runner_command(
        make_args(tmp_path), "abc1234", Run("nvs01", GATED_ARM, 3), tmp_path / "scratch"
    )
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--instance") + 1] == "nvs01"
    assert cmd[-1] == "--no-lns"
    # `--instance` is the runner's own second lock on the published table: it
    # refuses a subset run onto comparison.csv whatever else is passed, which is
    # what covers the control arm, whose flags are all default.
    assert "--instance" in cmd
    assert Path(cmd[cmd.index("--out") + 1]).is_relative_to(tmp_path / "scratch")


# --- resume --------------------------------------------------------------------


def write_results(path: Path, keys: Sequence[tuple[str, str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_COLUMNS)
        for instance, arm, seed in keys:
            row = dict.fromkeys(RESULT_COLUMNS, "0")
            row.update({"instance": instance, "arm": arm, "seed": str(seed)})
            writer.writerow([row[column] for column in RESULT_COLUMNS])


def test_resume_skips_exactly_the_recorded_triples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_dir = tmp_path / "scratch"
    write_results(out_dir / RESULTS_NAME, [("a", CONTROL_ARM, 1), ("b", "no-float-hook", 2)])
    seen: list[str] = []
    monkeypatch.setattr(subprocess, "run", fake_runner(seen=seen))
    plan = campaign_plan(["a", "b"], [1, 2], [ARMS[0], ARMS[1]])
    execute_runs(make_args(tmp_path), "abc1234", plan, out_dir, "campaign")
    ran = {tuple(slug.split("__")) for slug in seen}
    assert ("a", CONTROL_ARM, "seed1") not in ran
    assert ("b", "no-float-hook", "seed2") not in ran
    assert len(seen) == len(plan) - 2


def test_no_resume_re_runs_every_triple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "scratch"
    write_results(out_dir / RESULTS_NAME, [("a", CONTROL_ARM, 1)])
    seen: list[str] = []
    monkeypatch.setattr(subprocess, "run", fake_runner(seen=seen))
    plan = campaign_plan(["a"], [1], [ARMS[0]])
    execute_runs(make_args(tmp_path, resume=False), "abc1234", plan, out_dir, "campaign")
    assert seen == ["a__control__seed1"]


def test_every_completed_run_is_on_disk_before_the_next_one_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ten-hour campaign killed at hour nine must keep hour nine's work."""
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    results = out_dir / RESULTS_NAME
    counts: list[int] = []

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        counts.append(len(recorded_keys(results)))
        return fake_runner()(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    plan = campaign_plan(["a", "b"], [1], [ARMS[0]])
    execute_runs(make_args(tmp_path), "abc1234", plan, out_dir, "campaign")
    assert counts == [0, 1]  # the second solve began with the first already durable
    assert len(recorded_keys(results)) == 2


def test_a_torn_final_line_is_dropped_before_it_can_be_resumed(tmp_path: Path) -> None:
    results = tmp_path / RESULTS_NAME
    write_results(results, [("a", CONTROL_ARM, 1)])
    with results.open("a") as fh:
        fh.write("b,control,,")  # killed mid-append
    assert repair_torn_tail(results) is True
    assert recorded_keys(results) == {("a", CONTROL_ARM, 1)}
    assert repair_torn_tail(results) is False


def test_the_stamp_refuses_a_resume_from_another_budget(tmp_path: Path) -> None:
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    stamp = campaign_stamp("abc1234", 60.0, [1, 2, 3], list(ARMS))
    assert stamp_conflict(out_dir, stamp, resume=True) is None
    assert (out_dir / STAMP_NAME).exists()
    other = campaign_stamp("abc1234", 5.0, [1, 2, 3], list(ARMS))
    conflict = stamp_conflict(out_dir, other, resume=True)
    assert conflict is not None
    assert "different configuration" in conflict
    # --no-resume is starting over, so the stamp is rewritten rather than checked.
    assert stamp_conflict(out_dir, other, resume=False) is None


def test_a_runner_row_for_another_instance_is_refused(tmp_path: Path) -> None:
    """A stale file from an earlier invocation would otherwise be recorded under
    this run's arm and seed."""
    path = tmp_path / "row.csv"
    path.write_text(f"{RUNNER_HEADER}\nother,1,1,1,5,5,60,true,feasible,abc1234,0,0,0,x\n")
    with pytest.raises(RuntimeError, match="is for other"):
        read_runner_row(path, Run("nvs01", ARMS[0], 1), "abc1234")


def test_a_runner_row_from_another_commit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "row.csv"
    path.write_text(f"{RUNNER_HEADER}\nnvs01,1,1,1,5,5,60,true,feasible,old0000,0,0,0,x\n")
    with pytest.raises(RuntimeError, match="written at old0000"):
        read_runner_row(path, Run("nvs01", ARMS[0], 1), "abc1234")


def test_a_header_only_runner_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "row.csv"
    path.write_text(RUNNER_HEADER + "\n")
    with pytest.raises(RuntimeError, match="expected exactly 1"):
        read_runner_row(path, Run("nvs01", ARMS[0], 1), "abc1234")


# --- the LNS gate --------------------------------------------------------------


def probe_row(instance: str, repairs: str) -> dict[str, str]:
    return {"instance": instance, "arm": PROBE_ARM_NAME, "lns_repairs": repairs}


def test_the_gate_runs_the_arm_when_any_instance_repaired() -> None:
    decision = decide_lns_gate([probe_row("a", "0"), probe_row("b", "4")])
    assert decision.run_arm is True
    assert decision.instances_with_repairs == 1
    assert decision.total_repairs == 4
    assert "real arm" in decision.reason


def test_the_gate_skips_the_arm_when_nothing_repaired() -> None:
    """With no repair anywhere, `diversify()` takes the perturb branch at every
    kick with or without LNS, so the arm is provably a no-op."""
    decision = decide_lns_gate([probe_row("a", "0"), probe_row("b", "0")])
    assert decision.run_arm is False
    assert decision.total_repairs == 0
    assert "would measure nothing" in decision.reason


def test_a_row_where_no_solve_ran_does_not_vote_in_the_gate() -> None:
    """`lns_repairs` is NaN on a row the runner wrote without solving. It must
    not read as a repair, and it must not crash the gate either."""
    decision = decide_lns_gate([probe_row("a", "NaN"), probe_row("b", "")])
    assert decision.run_arm is False
    assert decision.total_repairs == 0


def test_the_gate_decision_is_recorded_as_data() -> None:
    """A driver decision in the output, not a human one in a shell."""
    recorded = decide_lns_gate([probe_row("a", "2")]).as_dict()
    assert recorded["run_arm"] is True
    assert recorded["min_repairs_per_instance"] == 1
    assert recorded["min_instances"] == 1
    assert isinstance(recorded["reason"], str)


def test_the_estimate_is_derived_from_the_budget() -> None:
    assert estimate_hours(750, 60.0) == pytest.approx(12.5)
    assert estimate_hours(50, 60.0) == pytest.approx(50 / 60)
