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

from benchmarks.minlplib.ablation_report import (
    CONTROL_ARM,
    PROBE_ARM_NAME,
    render_report,
    sign_test_p,
    t_multiplier,
)
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
    drop_partial_block,
    estimate_hours,
    execute,
    execute_runs,
    failed_row,
    load_refusal,
    main,
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
    "wall_seconds,feasible,note,commit_sha,max_violation,n_int_vars,lns_repairs,"
    "lns_repairs_accepted,search_config"
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
            f"{name},1,1,1,5,5,60,true,feasible,{sha},0,0,{repairs},0,{DEFAULT_ARM_CELL}\n"
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


def test_the_refused_driver_can_still_name_the_holder(tmp_path: Path) -> None:
    """A truncating open would wipe the pid line before flock had even failed,
    so the refusal would destroy the one diagnostic it needs to quote."""
    out_dir = tmp_path / "scratch"
    with campaign_lock(out_dir):
        holder = (out_dir / LOCK_NAME).read_text()
        assert holder.startswith("pid=")
        with pytest.raises(RuntimeError, match=r"pid=\d+"), campaign_lock(out_dir):
            pass
        assert (out_dir / LOCK_NAME).read_text() == holder, "the holder's pid was overwritten"


def test_the_lock_refusal_does_not_advise_deleting_the_lock(tmp_path: Path) -> None:
    """flock is held on the inode. Deleting a LIVE lock file lets the next
    driver create a fresh inode and lock that -- two wall-clock-budgeted
    campaigns on one machine, invited by the driver's own error message. A file
    left by a crashed driver is already unlocked, so the advice is never needed
    either."""
    out_dir = tmp_path / "scratch"
    with (
        campaign_lock(out_dir),
        pytest.raises(RuntimeError) as caught,
        campaign_lock(out_dir),
    ):
        pass
    message = str(caught.value)
    assert "Do NOT delete the lock file" in message
    assert "already unlocked" in message


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


def test_resume_skips_recorded_triples_outside_the_interrupted_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Resume skips what is recorded -- EXCEPT the block the interruption split.

    `("a", control, 1)` sits in a block that was left half-finished, so its
    whole block is re-run: the interleave is the protocol, and finishing a block
    after however long the campaign was down compares its arms across the
    interruption. `("b", no-float-hook, 2)` is in the last recorded block, which
    is likewise partial and likewise redone. What must NOT happen is redoing
    blocks that completed in one sitting.
    """
    out_dir = tmp_path / "scratch"
    write_results(out_dir / RESULTS_NAME, [("a", CONTROL_ARM, 1), ("b", "no-float-hook", 2)])
    seen: list[str] = []
    monkeypatch.setattr(subprocess, "run", fake_runner(seen=seen))
    plan = campaign_plan(["a", "b"], [1, 2], [ARMS[0], ARMS[1]])
    execute_runs(make_args(tmp_path), "abc1234", plan, out_dir, "campaign")
    ran = {tuple(slug.split("__")) for slug in seen}
    # ("a", control, 1) is in an EARLIER block, so it is still skipped.
    assert ("a", CONTROL_ARM, "seed1") not in ran
    # ("b", no-float-hook, 2) is in the trailing partial block, so it is redone
    # together with the rest of that block rather than finished in isolation.
    assert ("b", "no-float-hook", "seed2") in ran
    assert ("b", CONTROL_ARM, "seed2") in ran
    assert len(seen) == len(plan) - 1


def test_no_resume_re_runs_every_triple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "scratch"
    write_results(out_dir / RESULTS_NAME, [("a", CONTROL_ARM, 1)])
    seen: list[str] = []
    monkeypatch.setattr(subprocess, "run", fake_runner(seen=seen))
    plan = campaign_plan(["a"], [1], [ARMS[0]])
    execute_runs(make_args(tmp_path, resume=False), "abc1234", plan, out_dir, "campaign")
    assert seen == ["a__control__seed1"]


def test_no_resume_moves_the_previous_rows_aside_instead_of_adding_to_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--no-resume` is what the stamp conflict offers as the way out of a
    configuration mismatch, so it must not leave the mismatched rows in the file
    for the scorer to average in. Two rows for one triple would also shrink the
    measured noise floor, which decides every verdict."""
    out_dir = tmp_path / "scratch"
    results = out_dir / RESULTS_NAME
    write_results(results, [("a", CONTROL_ARM, 1)])
    monkeypatch.setattr(subprocess, "run", fake_runner())
    args = make_args(tmp_path, resume=False, lns_arm="off")
    assert execute(args, "abc1234", ["a"], out_dir) == 0
    with results.open(newline="") as fh:
        triples = [(r["instance"], r["arm"], r["seed"]) for r in csv.DictReader(fh)]
    assert len(triples) == len(set(triples)), f"a triple was recorded twice: {triples}"
    superseded = list(out_dir.glob("results.superseded-*.csv"))
    assert len(superseded) == 1, "the old rows must be kept, not deleted"
    assert recorded_keys(superseded[0]) == {("a", CONTROL_ARM, 1)}


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


def test_report_only_survives_the_torn_file_it_exists_to_read(tmp_path: Path) -> None:
    """Killed mid-append is the normal state of a ten-hour campaign, and
    --report-only is the only read-only way to look at what it produced."""
    out_dir = tmp_path / "scratch"
    results = out_dir / RESULTS_NAME
    write_results(results, [("a", CONTROL_ARM, 1)])
    with results.open("a") as fh:
        fh.write("b,control,,")
    assert main(["--out-dir", str(out_dir), "--report-only"]) == 0


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
    path.write_text(f"{RUNNER_HEADER}\nother,1,1,1,5,5,60,true,feasible,abc1234,0,0,0,0,x\n")
    with pytest.raises(RuntimeError, match="is for other"):
        read_runner_row(path, Run("nvs01", ARMS[0], 1), "abc1234")


def test_a_runner_row_from_another_commit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "row.csv"
    path.write_text(f"{RUNNER_HEADER}\nnvs01,1,1,1,5,5,60,true,feasible,old0000,0,0,0,0,x\n")
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
    """`lns_repairs` is NaN on a row the runner wrote without solving.

    It must not read as a repair, it must not crash the gate -- and it must not
    read as a reading of ZERO repairs either, which is what an earlier cut did.
    A skip assembled from rows where nothing ran is not "the counter reading
    that justified skipping"; with no reading at all the arm runs and the
    campaign spends the time.
    """
    decision = decide_lns_gate([probe_row("a", "NaN"), probe_row("b", "")])
    assert decision.run_arm is True
    assert decision.total_repairs == 0
    assert decision.probed_runs == 0
    assert decision.unread_runs == 2


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


def _campaign_csv(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    """A results.csv holding exactly `rows`, every other cell a benign default."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_COLUMNS)
        for row in rows:
            cells = dict.fromkeys(RESULT_COLUMNS, "0")
            cells["feasible"] = "true"
            cells["lns_repairs"] = "0"
            cells.update({k: str(v) for k, v in row.items()})
            writer.writerow([cells[column] for column in RESULT_COLUMNS])
    return path


def _flat_campaign(
    huge_gap: float, huge_noise: float, arm_shift: float = 0.0
) -> list[dict[str, object]]:
    """45 small instances plus one enormous one, with NO true arm effect.

    This is the shape of the real roster: in the committed comparison.csv
    `gear4` is 1.65e6 gap points and the median instance is 1.00, so one
    instance owns 98.5% of the sum. `arm_shift` adds a uniform regression to
    every small instance, which is the effect a roster-wide statistic has to be
    able to see past the big one.
    """
    rows: list[dict[str, object]] = []
    for i in range(45):
        for seed in (1, 2, 3):
            base = 1.0 + 0.01 * ((i + seed) % 3)
            rows.append(
                {"instance": f"small{i}", "arm": "control", "seed": seed, "gap_to_bks%": base}
            )
            rows.append(
                {
                    "instance": f"small{i}",
                    "arm": "no-float-hook",
                    "seed": seed,
                    "gap_to_bks%": base + arm_shift,
                }
            )
    for seed in (1, 2, 3):
        # The big instance's own run-to-run noise. The two sides differ by one
        # noise step, which is well INSIDE this instance's own floor (its
        # across-seed spread is the same size) -- but in raw gap points it is
        # four orders of magnitude larger than anything the small instances do,
        # so it is the whole of any unweighted mean.
        rows.append(
            {
                "instance": "gear4",
                "arm": "control",
                "seed": seed,
                "gap_to_bks%": huge_gap + huge_noise * (seed - 2),
            }
        )
        rows.append(
            {
                "instance": "gear4",
                "arm": "no-float-hook",
                "seed": seed,
                "gap_to_bks%": huge_gap + huge_noise * (seed - 1),
            }
        )
    return rows


def test_one_enormous_instance_cannot_decide_the_verdict(tmp_path: Path) -> None:
    """The roster's gaps span six orders of magnitude, so an unweighted mean is
    not an average over the roster -- it is the largest instance's seed draw.

    Against the cut that took `statistics.fmean` of raw per-instance deltas,
    this exact campaign printed `mean gap delta -1140.74 points is INSIDE THE
    NOISE (measured floor +/-3733.20 points)`, where both numbers were entirely
    `gear4`. Here there is no arm effect at all, so the only honest verdict is
    that nothing moved.
    """
    results = _campaign_csv(
        tmp_path / RESULTS_NAME, _flat_campaign(huge_gap=1.65e6, huge_noise=8.0e4)
    )
    report = render_report(results)

    assert "INSIDE THE NOISE" in report
    # And the mean, which is still printed, has to disclose that it is one
    # instance's number wearing the roster's name.
    assert "NOT the verdict statistic" in report
    assert "largest single-instance |delta|" in report
    assert "gear4" in report


def test_a_uniform_regression_is_not_hidden_by_the_enormous_instance(tmp_path: Path) -> None:
    """The other half of the same defect: a real, uniform regression on every
    small instance moved the old mean by 9.8 points against a floor of 3733,
    and was reported as inside the noise. It must now be seen."""
    results = _campaign_csv(
        tmp_path / RESULTS_NAME,
        _flat_campaign(huge_gap=1.65e6, huge_noise=8.0e4, arm_shift=10.0),
    )
    report = render_report(results)

    assert "INSIDE THE NOISE" not in report
    assert "WORSE than the control" in report
    assert "45 worse" in report


def test_an_instance_with_one_control_run_is_not_scored(tmp_path: Path) -> None:
    """No spread, no floor, no score -- and it is counted, not dropped.

    Imputing the median absolute spread (~1 gap point on this roster) onto an
    instance whose gap is ~1e6 hands it a floor it clears automatically, which
    manufactures a significant result. `gear4` is exactly the instance most
    likely to be feasible on only one seed.
    """
    rows: list[dict[str, object]] = []
    rows += [
        {"instance": "small0", "arm": "control", "seed": s, "gap_to_bks%": 1.0 + 0.01 * s}
        for s in (1, 2, 3)
    ]
    rows += [
        {"instance": "small0", "arm": "no-float-hook", "seed": s, "gap_to_bks%": 1.0}
        for s in (1, 2, 3)
    ]
    # One comparable control run, an enormous delta on the arm side.
    rows.append({"instance": "gear4", "arm": "control", "seed": 1, "gap_to_bks%": 1.0e6})
    rows += [
        {"instance": "gear4", "arm": "no-float-hook", "seed": s, "gap_to_bks%": 2.0e6}
        for s in (1, 2, 3)
    ]
    report = render_report(_campaign_csv(tmp_path / RESULTS_NAME, rows))

    assert "NOT SCORED" in report
    assert "fewer than two comparable runs" in report
    # It must not have been scored as a move despite its 1e6 delta.
    assert "1 worse" not in report


def test_the_floor_uses_a_student_multiplier_at_three_seeds() -> None:
    """df = 2 at three seeds, so the two-sided 95% multiplier is 4.30, not 2.0.

    An earlier cut used 2.0 flat and said in its docstring that it had "no
    degrees of freedom for" a t-interval, which is wrong in both directions:
    there are two, and the band it printed was about half its nominal width.
    """
    assert t_multiplier(2) == pytest.approx(4.303)
    assert t_multiplier(1) == pytest.approx(12.71)
    # Off the tabulated points the NEXT LOWER df's multiplier is used, which is
    # the wider band: a floor that errs generous keeps a marginal move from
    # being called a result. Never narrower than the normal value.
    assert t_multiplier(50) >= 1.96
    assert t_multiplier(7) == pytest.approx(2.365)


def test_the_sign_test_needs_more_than_a_bare_majority() -> None:
    """Two instances one way and one the other is not a direction."""
    assert sign_test_p(0, 0) == pytest.approx(1.0)
    assert sign_test_p(2, 1) > 0.05
    assert sign_test_p(10, 0) < 0.01
    assert sign_test_p(5, 5) == pytest.approx(1.0)


def test_a_row_with_no_reading_cannot_vote_to_skip_the_lns_arm() -> None:
    """ "NaN" is what the runner writes when no solve ran, and it exits 0 doing
    it -- for an unsupported instance and for a solve that threw.

    Folding those to zero let rows where nothing happened vote for SKIP, which
    is the opposite of what the runner's own comment on that cell says. A skip
    assembled from runs that never solved would not be the "counter reading
    that justified skipping" the acceptance criterion asks for.
    """
    unread = [{"instance": f"i{i}", "arm": PROBE_ARM.name, "lns_repairs": "NaN"} for i in range(20)]
    decision = decide_lns_gate([dict.fromkeys(RESULT_COLUMNS, "0") | r for r in unread])

    # Nothing was read, so nothing may be skipped on it: the arm runs.
    assert decision.run_arm
    assert decision.probed_runs == 0
    assert decision.unread_runs == 20
    assert "produced an lns_repairs reading" in decision.reason


def test_a_genuine_zero_reading_still_skips_the_arm() -> None:
    """The gate must still be able to fire: a real reading of zero repairs
    across the roster means --no-lns is trajectory-identical to control."""
    rows = [
        dict.fromkeys(RESULT_COLUMNS, "0")
        | {"instance": f"i{i}", "arm": PROBE_ARM.name, "lns_repairs": "0"}
        for i in range(20)
    ]
    decision = decide_lns_gate(rows)

    assert not decision.run_arm
    assert decision.probed_runs == 20
    assert decision.unread_runs == 0


def test_the_load_threshold_is_below_one_busy_core() -> None:
    """The runner is single-threaded, so a running campaign holds the 1-minute
    average at about 1.00 and oscillates either side of it. A threshold of 1.0
    let a second campaign start about half the time it was tried, and the lock
    is per out-dir so nothing else stands between them."""
    assert MAX_LOAD_AVERAGE < 1.0


def test_resume_re_runs_the_block_the_interruption_split(tmp_path: Path) -> None:
    """The interleave is the protocol, and a plain resume breaks it.

    Every arm for one (instance, seed) runs back to back so the control and the
    arms meet the same machine. An interruption lands inside such a block with
    probability (k-1)/k -- 80% at five arms -- and a resume that just skips what
    is recorded finishes that block after however long the campaign was down,
    with nothing downstream able to see it.
    """
    arms = [Arm("control", ()), Arm("a", ("--x",)), Arm("b", ("--y",))]
    runs = campaign_plan(["i1", "i2"], [7], arms)
    results = tmp_path / RESULTS_NAME
    # i1's block is 2 of 3 complete: the interruption split it.
    recorded = [("i1", "control", 7), ("i1", "a", 7)]
    _campaign_csv(
        results,
        [{"instance": i, "arm": a, "seed": s} for i, a, s in recorded],
    )

    done = drop_partial_block(results, runs, set(recorded))

    # The whole block is re-run, not just its missing arm.
    assert done == set()
    # And the discarded rows are kept, not deleted -- they are real solves.
    aside = tmp_path / "results.split-block.csv"
    assert aside.exists()
    assert len(aside.read_text().strip().splitlines()) == 3  # header + 2 rows
    assert len(results.read_text().strip().splitlines()) == 1  # header only


def test_resume_leaves_a_complete_block_alone(tmp_path: Path) -> None:
    """A block that finished in one sitting is not redone -- that would burn
    hours re-measuring something already measured correctly."""
    arms = [Arm("control", ()), Arm("a", ("--x",))]
    runs = campaign_plan(["i1", "i2"], [7], arms)
    results = tmp_path / RESULTS_NAME
    recorded = [("i1", "control", 7), ("i1", "a", 7)]
    _campaign_csv(results, [{"instance": i, "arm": a, "seed": s} for i, a, s in recorded])

    done = drop_partial_block(results, runs, set(recorded))

    assert done == set(recorded)
    assert not (tmp_path / "results.split-block.csv").exists()


def test_report_only_does_not_repair_the_file_a_campaign_is_appending_to(
    tmp_path: Path,
) -> None:
    """The README invites watching a running campaign from a second terminal.

    The row that looks torn from there is very often one the running driver has
    already fsynced and already struck off its in-memory resume set, so
    repairing the live file would delete that run from the record permanently.
    """
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    results = _campaign_csv(
        out_dir / RESULTS_NAME,
        [{"instance": "small0", "arm": "control", "seed": 1, "gap_to_bks%": 1.0}],
    )
    torn = results.read_text() + "small0,no-float-hook,,2,60,abc"  # no trailing newline
    results.write_text(torn)

    rc = main(["--out-dir", str(out_dir), "--report-only"])

    assert rc == 0
    assert results.read_text() == torn, "the live results file was modified"


def test_a_failed_run_is_recorded_rather_than_ending_the_campaign(tmp_path: Path) -> None:
    """An instance that crashes only under one arm, at hour nine, must not make
    the campaign unfinishable -- and a resume must not retry the same crash
    forever. The row carries no measurement, so nothing can read it as one."""
    args = make_args(tmp_path)
    row = failed_row(Run("i1", ARMS[0], 7), args, "abc1234", 139)

    assert row["note"] == "runner-failed-exit-139"
    assert row["feasible"] == "false"
    assert row["gap_to_bks%"] == "NaN"
    assert row["lns_repairs"] == "NaN"
    # A NaN repair cell is "no reading", which is what keeps a crashed run out
    # of the LNS gate's denominator.
    assert decide_lns_gate([{**row, "arm": PROBE_ARM.name}]).unread_runs == 1
