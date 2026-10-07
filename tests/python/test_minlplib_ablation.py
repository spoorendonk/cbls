"""Unit tests for the MINLPLib ablation campaign driver (issue #143).

Every test here is offline. The campaign itself is ten hours of solving, so what
is checkable without solving anything is exactly what the driver is: a run
order, a set of refusals, and a resume rule. `execute_runs` is exercised against
a fake runner, so even its loop costs no solve. The durable-append and torn-tail
mechanics it is built on are pinned in `test_benchmark_common.py`.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.common import jobs
from benchmarks.common.jobs import wallclock_lock
from benchmarks.common.provenance import REPO_ROOT
from benchmarks.minlplib import independent_check
from benchmarks.minlplib.ablation_report import (
    CONTROL_ARM,
    PROBE_ARM_NAME,
    render_report,
    sign_test_p,
)
from benchmarks.minlplib.run_ablation import (
    ARMS,
    GATED_ARM,
    HELDOUT_INST_DIR,
    LOCK_NAME,
    MAX_LOAD_AVERAGE,
    PROBE_ARM,
    RESULT_COLUMNS,
    RESULTS_NAME,
    SHIPPED_UNPRODUCTIVE_ITERS,
    STAMP_NAME,
    TRANSFER_ARMS,
    TRANSFER_CAMPAIGN,
    TRANSFER_GRID,
    Arm,
    ArmConfigMismatchError,
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
    header_conflict,
    load_refusal,
    main,
    parse_args,
    probe_plan,
    read_runner_row,
    recorded_keys,
    recorded_seeds,
    report_for,
    resolve_campaign,
    run_locked,
    runner_command,
    scratch_refusal,
    stamp_conflict,
    usage_error,
)
from benchmarks.minlplib.run_benchmark import DEFAULT_INST_DIR, _data_problems, roster_from_bounds
from benchmarks.minlplib.runner import RUNNER_COLUMNS, RUNNER_EXIT_ERRORED, completed_search

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

RUNNER_HEADER = ",".join(RUNNER_COLUMNS)
DEFAULT_ARM_CELL = (
    "float_hook=on;lns=on;lns_interval=3;compound_moves=on;novelty_prob=0.5;"
    "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on"
)


#: The real DAG-independent check, kept before `_no_independent_check` stubs it.
REAL_CHECK_ROWS = independent_check.check_rows


@pytest.fixture(autouse=True)
def _no_independent_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip the SCIP check (#205) where the test is about something else.

    The fake runner's instances have no `.nl`. The test that is about the check
    restores the real one and stubs only SCIP's verdict.
    """
    monkeypatch.setattr(independent_check, "check_rows", lambda *_a, **_k: [])


def make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "out_dir": tmp_path / "scratch",
        "inst_dir": tmp_path / "inst",
        "build_dir": tmp_path / "build",
        "time_limit": 60.0,
        "seeds": [1, 2, 3],
        "instances": [],
        "lns_arm": "auto",
        "campaign": "ablation",
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
    *,
    repairs: int = 0,
    returncode: int = 0,
    seen: list[str] | None = None,
    row: str | None = None,
    write_row: bool = True,
) -> Callable[..., FakeCompleted]:
    """Stand in for `subprocess.run(cbls_minlplib ...)` without solving anything.

    `row` is a row template with `{name}` and `{sha}` placeholders, for the tests
    that care what the runner left behind next to its exit status;
    `write_row=False` is the process that died before writing one.
    """

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        name = cmd[cmd.index("--instance") + 1]
        sha = cmd[cmd.index("--commit") + 1]
        out = Path(cmd[cmd.index("--out") + 1])
        if seen is not None:
            seen.append(out.stem)
        out.parent.mkdir(parents=True, exist_ok=True)
        template = (
            row
            if row is not None
            else f"{{name}},1,1,1,5,5,60,true,feasible,{{sha}},0,0,{repairs},0,3,0.5,"
            f"{DEFAULT_ARM_CELL}"
        )
        text = f"{RUNNER_HEADER}\n"
        if write_row:
            text += template.format(name=name, sha=sha) + "\n"
        out.write_text(text)
        return FakeCompleted(returncode)

    return run


#: What `minlplib.cpp`'s `write_unsolved_row` leaves behind when `cbls::solve`
#: throws: `feasible=false`, every measured cell NaN -- but the published bounds
#: and the discrete-variable count filled in, because the runner looked them up
#: before it ever tried to solve.
SOLVE_ERROR_ROW = (
    "{name},NaN,-1161.34,-1161.34,NaN,NaN,0,false,solve-error,{sha},NaN,4,NaN,NaN,NaN,NaN,"
    + DEFAULT_ARM_CELL
)


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


def test_a_second_campaign_on_the_same_out_dir_is_refused_and_told_why(tmp_path: Path) -> None:
    """Timed comparisons must never share the machine, and the driver says so
    itself rather than trusting the caller to remember.

    The refusal quotes the holder's pid, which a truncating open would wipe
    before flock had even failed. And it must not advise deleting the lock:
    flock is held on the inode, so deleting a LIVE lock file lets the next
    driver lock a fresh inode and run beside the first -- two wall-clock-
    budgeted campaigns on one machine, invited by the driver's own message.
    """
    out_dir = tmp_path / "scratch"
    with campaign_lock(out_dir):
        holder = (out_dir / LOCK_NAME).read_text()
        assert holder.startswith("pid=")
        with (
            pytest.raises(RuntimeError, match="another campaign holds") as caught,
            campaign_lock(out_dir),
        ):
            pass
        assert (out_dir / LOCK_NAME).read_text() == holder, "the holder's pid was overwritten"
    message = str(caught.value)
    assert holder.strip() in message
    assert "Do NOT delete the lock file" in message
    assert "already unlocked" in message
    with campaign_lock(out_dir):
        pass  # a clean acquisition once the campaign ended: the first really let go


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
    plan = campaign_plan(["a", "b"], [1], list(ARMS))
    execute_runs(make_args(tmp_path), "abc1234", plan, out_dir, "campaign")
    assert not inflight


def test_a_busy_machine_is_refused_and_the_refusal_is_overridable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner is single-threaded, so a running campaign holds the 1-minute
    average at about 1.00 and oscillates either side of it. A threshold of 1.0
    let a second campaign start about half the time it was tried, and the lock
    is per out-dir so nothing else stands between them."""
    assert MAX_LOAD_AVERAGE < 1.0
    monkeypatch.setattr("os.getloadavg", lambda: (MAX_LOAD_AVERAGE + 1.0, 0.0, 0.0))
    refusal = load_refusal(allow_busy=False)
    assert refusal is not None
    assert "load average" in refusal
    assert load_refusal(allow_busy=True) is None
    monkeypatch.setattr("os.getloadavg", lambda: (0.0, 0.0, 0.0))
    assert load_refusal(allow_busy=False) is None


# --- refusals: the published tables, and a campaign that cannot measure --------


@pytest.mark.parametrize(
    ("out_dir", "said"),
    [
        # Refused, not warned: the campaign writes several files and the runner
        # writes more underneath, so the durable rule is that the whole tree is off
        # limits rather than that three filenames are.
        ("benchmarks/instances", "published"),
        ("benchmarks/instances/minlplib", "published"),
        ("benchmarks/instances/minlplib/scratch", "published"),
        # ... and so is anything containing it.
        ("", "contains"),
        ("benchmarks", "contains"),
        (None, None),
    ],
    ids=["instances", "minlplib", "under-minlplib", "repo-root", "benchmarks", "scratch"],
)
def test_an_out_dir_that_could_reach_a_published_table_is_refused(
    tmp_path: Path, out_dir: str | None, said: str | None
) -> None:
    path = tmp_path / "campaign" if out_dir is None else REPO_ROOT / out_dir
    refusal = scratch_refusal(path)
    assert refusal is None if said is None else (refusal is not None and said in refusal)


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        # The floor is measured from the control's across-seed spread, so a
        # two-seed campaign cannot both estimate an effect and its noise.
        ({"seeds": [1]}, "at least three seeds"),
        ({"seeds": [1, 2]}, "at least three seeds"),
        ({"seeds": [1, 1, 2]}, "distinct"),
        ({"out_dir": REPO_ROOT / "benchmarks" / "instances" / "minlplib"}, "published"),
        ({}, None),
    ],
    ids=["one-seed", "two-seeds", "repeated-seed", "published-out-dir", "accepted"],
)
def test_usage_error(tmp_path: Path, overrides: dict[str, object], refusal: str | None) -> None:
    message = usage_error(make_args(tmp_path, **overrides))
    assert message is None if refusal is None else (message is not None and refusal in message)


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


def _torn(results: Path) -> str:
    """A results.csv whose last append was killed half-way through the row."""
    write_results(results, [("a", CONTROL_ARM, 1)])
    with results.open("a") as fh:
        fh.write("b,control,,")
    return results.read_text()


def test_a_campaign_drops_a_torn_final_line_before_it_can_be_resumed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A torn row's key is missing from the resume set while its bytes stay in the
    file, so without the repair the run is redone AND the half-row stays behind,
    carried into the record as a row with no seed."""
    out_dir = tmp_path / "scratch"
    _torn(out_dir / RESULTS_NAME)
    monkeypatch.setattr(subprocess, "run", fake_runner())

    assert execute(make_args(tmp_path, lns_arm="off"), "abc1234", ["a"], out_dir) == 0

    with (out_dir / RESULTS_NAME).open(newline="") as fh:
        seeds = [row["seed"] for row in csv.DictReader(fh)]
    assert seeds and all(seed.isdigit() for seed in seeds), seeds


def test_report_only_scores_a_torn_file_without_repairing_the_live_one(tmp_path: Path) -> None:
    """Killed mid-append is the normal state of a ten-hour campaign, and
    --report-only is the only read-only way to look at what it produced.

    The README invites watching a running campaign from a second terminal. The
    row that looks torn from there is very often one the running driver has
    already fsynced and already struck off its in-memory resume set, so
    repairing the live file would delete that run from the record permanently.
    """
    out_dir = tmp_path / "scratch"
    torn = _torn(out_dir / RESULTS_NAME)

    assert main(["--out-dir", str(out_dir), "--report-only"]) == 0
    assert (out_dir / RESULTS_NAME).read_text() == torn, "the live results file was modified"


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
    # Each field is a way to mix two campaigns into one results.csv, not only the budget.
    for variant in (
        campaign_stamp("abc1234", 60.0, [1, 2, 7], list(ARMS)),
        campaign_stamp("abc1234", 60.0, [1, 2, 3], list(ARMS), roster=["i1"]),
        campaign_stamp("abc1234", 60.0, [1, 2, 3], list(ARMS), lns_arm="off"),
    ):
        assert stamp_conflict(out_dir, variant, resume=True) is not None
    # --no-resume is starting over, so the stamp is rewritten rather than checked.
    assert stamp_conflict(out_dir, other, resume=False) is None


def test_a_results_file_written_under_another_schema_is_refused(tmp_path: Path) -> None:
    """The campaign stamp usually catches a schema change first, because the
    schema moves with the engine commit and the stamp records the commit. It
    cannot when the stamp file is absent -- deleted, or an out-dir from a driver
    that predates it -- and then `open_results` leaves the old header in place
    while `append_result` writes a row of the NEW width beneath it, at exit 0.

    `csv.DictReader` binds by position from there on: under a pre-#150 header a
    post-#150 row hands `lns_repairs_accepted` to `search_config` and drops the
    last cell into `restkey`. The campaign then scores columns that never
    described it, and nothing says so.
    """
    results = tmp_path / RESULTS_NAME
    assert header_conflict(results) is None  # absent: nothing to conflict with

    # Spelled out, because it is derived: `header_conflict` compares the header
    # in order, so a reordered derivation would make every existing campaign
    # directory unresumable -- and nothing else says what `results.csv` looks like.
    assert RESULT_COLUMNS == (
        "instance",
        "arm",
        "arm_flags",
        "seed",
        "time_limit",
        "commit_sha",
        "objective",
        "primal_bks",
        "dual_bound",
        "gap_to_bks%",
        "gap_to_dual%",
        "wall_seconds",
        "feasible",
        "note",
        "max_violation",
        "n_int_vars",
        "lns_repairs",
        "lns_repairs_accepted",
        "first_feasible_objective",
        "time_to_first_feasible",
        "search_config",
    )
    results.write_text(",".join(RESULT_COLUMNS) + "\n")
    assert header_conflict(results) is None  # this schema: appendable

    older = [c for c in RESULT_COLUMNS if c != "lns_repairs_accepted"]
    results.write_text(",".join(older) + "\n")
    conflict = header_conflict(results)
    assert conflict is not None
    assert "lns_repairs_accepted" in conflict
    assert "--report-only" in conflict  # the way to read it where it stands


@pytest.mark.parametrize(
    ("row", "match"),
    [
        # A stale file from an earlier invocation would otherwise be recorded under
        # this run's arm and seed.
        ("other,1,1,1,5,5,60,true,feasible,abc1234,0,0,0,0,x", "is for other"),
        ("nvs01,1,1,1,5,5,60,true,feasible,old0000,0,0,0,0,x", "written at old0000"),
        # A runner that wrote the header and died leaves no row.
        (None, "expected exactly 1"),
    ],
    ids=["another-instance", "another-commit", "header-only"],
)
def test_a_runner_row_that_is_not_this_run_is_refused(
    tmp_path: Path, row: str | None, match: str
) -> None:
    path = tmp_path / "row.csv"
    path.write_text(RUNNER_HEADER + "\n" + ("" if row is None else row + "\n"))
    with pytest.raises(RuntimeError, match=match):
        read_runner_row(path, Run("nvs01", ARMS[0], 1), "abc1234")


def _campaign_csv(path: Path, rows: Sequence[dict[str, object]]) -> Path:
    """A results.csv holding exactly `rows`, every other cell a benign default."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_COLUMNS)
        for row in rows:
            cells = dict.fromkeys(RESULT_COLUMNS, "0")
            cells["feasible"] = "true"
            cells["lns_repairs"] = "0"
            # The scorer scores only the notes a COMPLETED search writes (#153);
            # "0" is not one, so a benign default has to be a real note.
            cells["note"] = "feasible"
            cells.update({k: str(v) for k, v in row.items()})
            writer.writerow([cells[column] for column in RESULT_COLUMNS])
    return path


@pytest.mark.parametrize(
    ("arms", "recorded", "left"),
    [
        # The interleave is the protocol, and a plain resume breaks it. Every arm
        # for one (instance, seed) runs back to back so the control and the arms
        # meet the same machine; an interruption lands inside such a block with
        # probability (k-1)/k -- 80% at five arms -- and a resume that just skips
        # what is recorded finishes that block after the campaign was down, with
        # nothing downstream able to see it. So the whole block is re-run, and the
        # discarded rows are kept aside -- they are real solves.
        (["control", "a", "b"], [("i1", "control", 7), ("i1", "a", 7)], []),
        # A block that finished in one sitting is not redone -- that would burn
        # hours re-measuring something already measured correctly.
        (["control", "a"], [("i1", "control", 7), ("i1", "a", 7)], None),
    ],
    ids=["split-block-re-run-whole", "complete-block-left-alone"],
)
def test_resume_re_runs_only_the_block_the_interruption_split(
    tmp_path: Path,
    arms: list[str],
    recorded: list[tuple[str, str, int]],
    left: list[tuple[str, str, int]] | None,
) -> None:
    runs = campaign_plan(["i1", "i2"], [7], [Arm(name, (f"--{name}",)) for name in arms])
    results = _campaign_csv(
        tmp_path / RESULTS_NAME, [{"instance": i, "arm": a, "seed": s} for i, a, s in recorded]
    )

    done = drop_partial_block(results, runs, set(recorded))

    aside = tmp_path / "results.split-block.csv"
    if left is None:
        assert done == set(recorded)
        assert not aside.exists()
    else:
        assert done == set(left)
        assert len(aside.read_text().strip().splitlines()) == 1 + len(recorded)
        assert len(results.read_text().strip().splitlines()) == 1 + len(left)


# --- a run whose process failed ------------------------------------------------


def _recorded(out_dir: Path) -> list[dict[str, str]]:
    with (out_dir / RESULTS_NAME).open(newline="") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize(
    ("runner", "note", "primal_bks"),
    [
        # #153: a thrown solve exits nonzero, so it arrives on the failed-run path
        # -- and must not be downgraded on the way in. The runner knows the
        # published bounds (it reads `bounds.csv` before it builds anything) and
        # which of read, build or solve threw; the driver's own row would NaN all
        # of that. Held out of every count either way.
        (
            fake_runner(returncode=RUNNER_EXIT_ERRORED, row=SOLVE_ERROR_ROW),
            "solve-error",
            "-1161.34",
        ),
        # Exit 139 is a segfault: a row a dead process left behind is not evidence
        # of anything. Only the runner's own error-tally status buys its row trust.
        (fake_runner(returncode=139, row=SOLVE_ERROR_ROW), "runner-failed-exit-139", "NaN"),
        # The error status is not a promise that a readable row exists -- a full
        # disk reaches this -- so the fallback is the row that needs nothing.
        (
            fake_runner(returncode=RUNNER_EXIT_ERRORED, write_row=False),
            "runner-failed-exit-3",
            "NaN",
        ),
        # The process failed and the row says a search ran: both cannot be true,
        # and a campaign may lose a measurement but not gain one from a process
        # that reported failure.
        (
            fake_runner(returncode=RUNNER_EXIT_ERRORED),
            "runner-failed-row-claims-a-result",
            "NaN",
        ),
        # Exit 0 with no readable row: raising instead would wedge the campaign,
        # every resume re-dropping the block and dying at the same run.
        (fake_runner(write_row=False), "runner-failed-unreadable-row", "NaN"),
    ],
    ids=[
        "error-tally-row-kept",
        "crash",
        "error-without-row",
        "error-claiming-a-result",
        "unreadable",
    ],
)
def test_a_failed_run_is_recorded_rather_than_ending_the_campaign(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner: Callable[..., FakeCompleted],
    note: str,
    primal_bks: str,
) -> None:
    out_dir = tmp_path / "scratch"
    monkeypatch.setattr(subprocess, "run", runner)
    plan = campaign_plan(["i1", "i2"], [7], [ARMS[0]])
    execute_runs(make_args(tmp_path), "abc1234", plan, out_dir, "c")

    rows = _recorded(out_dir)
    # Recorded, and the campaign moved on: the next instance ran too.
    assert [row["instance"] for row in rows] == ["i1", "i2"]
    assert rows[0]["note"] == note
    assert rows[0]["primal_bks"] == primal_bks
    assert rows[0]["n_int_vars"] == ("4" if note == "solve-error" else "NaN")
    # ... and the failure's log is kept beside it, since it is the one that gets read.
    assert (out_dir / "runs" / "i1__control__seed7.log").exists()
    # Still not a measurement: the scorer holds every non-completed note out.
    assert not completed_search(rows[0]["note"])


def test_the_drivers_failed_row_carries_no_measurement(tmp_path: Path) -> None:
    """An instance that crashes only under one arm, at hour nine, must not make
    the campaign unfinishable -- and a resume must not retry the same crash
    forever. The row carries no measurement, so nothing can read it as one."""
    row = failed_row(Run("i1", ARMS[0], 7), make_args(tmp_path), "abc1234", 139)

    assert row["note"] == "runner-failed-exit-139"
    assert row["feasible"] == "false"
    assert row["gap_to_bks%"] == "NaN"
    assert row["lns_repairs"] == "NaN"
    # Both cells, not just the first: a half-read row is the one shape the
    # scorer cannot represent -- it would sum an accepted count against an
    # attempted count that was never taken.
    assert row["lns_repairs_accepted"] == "NaN"
    # A NaN repair cell is "no reading", which is what keeps a crashed run out
    # of the LNS gate's denominator.
    assert decide_lns_gate([{**row, "arm": PROBE_ARM.name}]).unread_runs == 1


# --- the LNS gate --------------------------------------------------------------


@pytest.mark.parametrize(
    ("repairs", "run_arm", "hits", "total", "probed", "reason"),
    [
        (["0", "4"], True, 1, 4, 2, "real arm"),
        # With no repair anywhere, `diversify()` takes the perturb branch at every
        # kick with or without LNS, so the arm is provably a no-op.
        (["0"] * 20, False, 0, 0, 20, "would measure nothing"),
        # `lns_repairs` is NaN on a row the runner wrote without solving -- for an
        # unsupported instance (exit 0) and for a solve that threw (exit 3). It
        # must not read as a repair, must not crash the gate -- and must not read
        # as a reading of ZERO repairs either, which is what an earlier cut did. A
        # skip assembled from rows where nothing ran is not "the counter reading
        # that justified skipping"; with no reading at all the arm runs.
        (["NaN", ""], True, 0, 0, 0, "produced an lns_repairs reading"),
        (["NaN"] * 20, True, 0, 0, 0, "produced an lns_repairs reading"),
        # A skip may stand on a reading with a few holes in it, and says so...
        (["NaN"] + ["0"] * 19, False, 0, 0, 19, "excluded from the denominator"),
        # ... but not on one below GATE_MIN_READABLE_FRACTION.
        (["NaN"] * 3 + ["0"] * 17, True, 0, 0, 17, "below the 90%"),
    ],
    ids=[
        "an-instance-repaired",
        "nothing-repaired",
        "no-solve-ran",
        "no-reading-at-all",
        "reading-with-holes",
        "reading-too-thin",
    ],
)
def test_the_lns_gate(
    repairs: list[str], run_arm: bool, hits: int, total: int, probed: int, reason: str
) -> None:
    rows = [
        {"instance": f"i{k}", "arm": PROBE_ARM_NAME, "lns_repairs": cell}
        for k, cell in enumerate(repairs)
    ]
    decision = decide_lns_gate(rows)
    assert decision.run_arm is run_arm
    assert (decision.instances_with_repairs, decision.total_repairs) == (hits, total)
    assert (decision.probed_runs, decision.unread_runs) == (probed, len(repairs) - probed)
    assert reason in decision.reason
    # A driver decision recorded in the output, not a human one in a shell.
    recorded = decision.as_dict()
    assert recorded["run_arm"] is run_arm
    assert recorded["reason"] == decision.reason
    assert (recorded["min_repairs_per_instance"], recorded["min_instances"]) == (1, 1)


def test_the_estimate_is_derived_from_the_budget() -> None:
    assert estimate_hours(750, 60.0) == pytest.approx(12.5)
    assert estimate_hours(50, 60.0) == pytest.approx(50 / 60)


# --- the report the campaign ends with -----------------------------------------


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


def test_the_sign_test_needs_more_than_a_bare_majority() -> None:
    """Two instances one way and one the other is not a direction."""
    assert sign_test_p(0, 0) == pytest.approx(1.0)
    assert sign_test_p(2, 1) > 0.05
    assert sign_test_p(10, 0) < 0.01
    assert sign_test_p(5, 5) == pytest.approx(1.0)


def test_a_held_wallclock_lock_refuses_the_campaign_before_it_touches_the_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`run_benchmark.py` and this driver share one machine-wide lock; the loser exits 2."""
    monkeypatch.setattr(jobs, "WALLCLOCK_LOCK", tmp_path / "lock" / "wallclock.lock")
    out_dir = tmp_path / "out"
    with wallclock_lock("run_benchmark.py"):
        assert run_locked(argparse.Namespace(build=False), "abc1234", ["a"], out_dir) == 2
    assert "run_benchmark.py" in capsys.readouterr().err
    assert not out_dir.exists()


# --- #145: the unproductive-exit transfer campaign ------------------------------


def echoing_runner(*, ignore_flag: bool = False) -> Callable[..., FakeCompleted]:
    """A fake runner whose `search_config` cell reports the `--unproductive-iters`
    it was given, as the real runner's does -- or, with `ignore_flag`, a binary
    that drops the flag and reports the shipped 300 whatever it was asked."""

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        value = "300"
        if "--unproductive-iters" in cmd and not ignore_flag:
            value = cmd[cmd.index("--unproductive-iters") + 1]
        cell = DEFAULT_ARM_CELL.replace("unproductive_iters=300", f"unproductive_iters={value}")
        row = f"{{name}},1,1,1,5,5,10,true,feasible,{{sha}},0,0,0,0,3,0.5,{cell}"
        return fake_runner(row=row)(cmd, **kwargs)

    return run


def test_the_transfer_arms_are_the_provenance_grid_with_the_shipped_value_as_control() -> None:
    """#145 re-runs exactly the grid the shipped value came from, centred on it.

    Pinned against the C++ header rather than restated: if the default moves,
    this campaign would otherwise go on calling 300 the control while the engine
    ships something else, and the "transfer" it measured would be of a value
    nobody ships.
    """
    header = (REPO_ROOT / "include" / "cbls" / "feasibility_jump.h").read_text()
    shipped = re.search(r"int64_t unproductive_iterations = (\d+);", header)
    assert shipped is not None
    assert int(shipped.group(1)) == SHIPPED_UNPRODUCTIVE_ITERS == 300
    assert "{100, 300, 1000}" in header  # the provenance comment's grid
    # ... and the default the RUNNER actually applies: search_config_flags.h
    # seeds `--unproductive-iters` from SearchConfig, which forwards it to
    # GFJConfig. Either could move alone.
    flags = (REPO_ROOT / "benchmarks" / "common" / "search_config_flags.h").read_text()
    assert "unproductive_iters = SearchConfig{}.unproductive_iterations;" in flags
    search = (REPO_ROOT / "include" / "cbls" / "search.h").read_text()
    runner_default = re.findall(r"int64_t unproductive_iterations = (\d+);", search)
    assert runner_default == [str(SHIPPED_UNPRODUCTIVE_ITERS)]
    assert TRANSFER_GRID == (100, 300, 1000)

    assert [(a.name, a.flags, a.config) for a in TRANSFER_ARMS] == [
        (CONTROL_ARM, ("--unproductive-iters", "300"), ("unproductive_iters=300",)),
        ("unproductive-100", ("--unproductive-iters", "100"), ("unproductive_iters=100",)),
        ("unproductive-1000", ("--unproductive-iters", "1000"), ("unproductive_iters=1000",)),
    ]
    # No name is shared with an ablation arm other than the control, so the two
    # campaigns' rows can never be mistaken for one another by arm name alone.
    assert {a.name for a in TRANSFER_ARMS} & {a.name for a in (*ARMS, GATED_ARM)} == {CONTROL_ARM}


def test_the_transfer_campaign_interleaves_its_arms_per_instance() -> None:
    """The same protocol as the ablation: every grid point for one (instance,
    seed) back to back, control first, before the next block starts."""
    plan = campaign_plan(["a", "b"], [1, 2, 3], TRANSFER_ARMS)
    assert len(plan) == 2 * 3 * 3
    names = [a.name for a in TRANSFER_ARMS]
    for start in range(0, len(plan), len(names)):
        block = plan[start : start + len(names)]
        assert [r.arm.name for r in block] == names
        assert len({(r.instance, r.seed) for r in block}) == 1
    assert [(r.instance, r.seed) for r in plan[:: len(names)]] == [
        ("a", 1),
        ("a", 2),
        ("a", 3),
        ("b", 1),
        ("b", 2),
        ("b", 3),
    ]


def test_the_stamp_refuses_a_resume_across_campaigns(tmp_path: Path) -> None:
    """Both campaigns have a `control`, and the two are different configurations.
    The stamp names the campaign and each arm's flags, so neither can resume into
    the other's rows."""
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    ablation = campaign_stamp("abc1234", 10.0, [1, 2, 3], [*ARMS, GATED_ARM])
    transfer = campaign_stamp("abc1234", 10.0, [1, 2, 3], TRANSFER_ARMS, campaign=TRANSFER_CAMPAIGN)
    assert "campaign=transfer-145\n" in transfer
    assert "control[--unproductive-iters 300]" in transfer
    assert stamp_conflict(out_dir, ablation, resume=True) is None
    conflict = stamp_conflict(out_dir, transfer, resume=True)
    assert conflict is not None
    assert "different configuration" in conflict
    # ... and the other way round.
    other = tmp_path / "other"
    other.mkdir()
    assert stamp_conflict(other, transfer, resume=True) is None
    assert stamp_conflict(other, ablation, resume=True) is not None
    # Same arm NAMES under different flags is a different arm set too.
    renamed = campaign_stamp(
        "abc1234",
        10.0,
        [1, 2, 3],
        [Arm(CONTROL_ARM, ()), *TRANSFER_ARMS[1:]],
        campaign=TRANSFER_CAMPAIGN,
    )
    assert stamp_conflict(other, renamed, resume=True) is not None


def test_a_transfer_run_cannot_resume_into_an_ablation_out_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """End to end through `execute`: refused before a single solve."""
    seen: list[str] = []
    monkeypatch.setattr(subprocess, "run", fake_runner(seen=seen))
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()  # `campaign_lock` creates it in a real run
    assert execute(make_args(tmp_path, lns_arm="off"), "abc1234", ["a"], out_dir) == 0
    ablation_runs = len(seen)
    assert ablation_runs == len(ARMS) * 3

    transfer = make_args(tmp_path, campaign=TRANSFER_CAMPAIGN)
    assert execute(transfer, "abc1234", ["a"], out_dir) == 2
    assert len(seen) == ablation_runs, "a transfer solve ran in an ablation out-dir"
    assert "campaign=ablation" in capsys.readouterr().err


def test_a_transfer_campaign_runs_its_grid_and_every_row_records_the_arms_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """#145: "every row records engine commit, seed, budget and the arm's value".
    The value is in `arm_flags` (so a crashed run's row has it too) and in the
    runner's own `search_config` cell (what the binary actually applied)."""
    monkeypatch.setattr(subprocess, "run", echoing_runner())
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()  # `campaign_lock` creates it in a real run
    args = make_args(tmp_path, campaign=TRANSFER_CAMPAIGN, time_limit=10.0)
    (tmp_path / "inst").mkdir()
    (tmp_path / "inst" / "bounds.csv").write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks\n"
        "a,other,3,4,min,1,1,0\nb,mixed-integer,60,50,min,1,1,0\n"
    )
    assert execute(args, "abc1234", ["a", "b"], out_dir) == 0

    rows = _recorded(out_dir)
    assert [(r["instance"], r["arm"], r["seed"]) for r in rows[:3]] == [
        ("a", CONTROL_ARM, "1"),
        ("a", "unproductive-100", "1"),
        ("a", "unproductive-1000", "1"),
    ]
    assert len(rows) == 2 * 3 * 3
    for row in rows:
        value = {"control": "300", "unproductive-100": "100", "unproductive-1000": "1000"}[
            row["arm"]
        ]
        assert row["arm_flags"] == f"--unproductive-iters {value}"
        assert f"unproductive_iters={value};" in row["search_config"]
        assert row["commit_sha"] == "abc1234"
        assert row["time_limit"] == "10"
        assert row["seed"] in ("1", "2", "3")
    # No gate probe and no LNS arm: those are the ablation's.
    assert not any(r["arm"] in (PROBE_ARM.name, GATED_ARM.name) for r in rows)
    assert not (out_dir / "lns_gate.json").exists()
    assert "campaign=transfer-145" in (out_dir / STAMP_NAME).read_text()
    out = capsys.readouterr().out
    assert "issue #145" in out
    assert "by pre-registered size band" in out


def test_a_row_whose_search_config_contradicts_its_arm_stops_the_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A binary that ignores `--unproductive-iters` would otherwise fill the
    campaign with three copies of the control filed under three grid points --
    and report, correctly by its own lights, that the value does not matter."""
    monkeypatch.setattr(subprocess, "run", echoing_runner(ignore_flag=True))
    out_dir = tmp_path / "scratch"
    plan = campaign_plan(["a"], [1], TRANSFER_ARMS)
    with pytest.raises(ArmConfigMismatchError, match="unproductive_iters=100"):
        execute_runs(make_args(tmp_path), "abc1234", plan, out_dir, "campaign")
    # The control (whose cell does say 300) was recorded; the mismatched run was not.
    assert [r["arm"] for r in _recorded(out_dir)] == [CONTROL_ARM]


def test_the_transfer_campaign_defaults_to_the_heldout_roster_and_refuses_the_published(
    tmp_path: Path,
) -> None:
    out = ["--out-dir", str(tmp_path / "scratch")]
    args = parse_args([*out, "--campaign", TRANSFER_CAMPAIGN])
    assert resolve_campaign(args) is None
    assert args.inst_dir == HELDOUT_INST_DIR
    assert usage_error(args) is None

    ablation = parse_args(out)
    assert resolve_campaign(ablation) is None
    assert ablation.campaign == "ablation"
    assert ablation.inst_dir == DEFAULT_INST_DIR

    published = parse_args(
        [*out, "--campaign", TRANSFER_CAMPAIGN, "--inst-dir", str(DEFAULT_INST_DIR)]
    )
    assert resolve_campaign(published) is None
    refusal = usage_error(published)
    assert refusal is not None
    assert "fitted on" in refusal

    gated = parse_args([*out, "--campaign", TRANSFER_CAMPAIGN, "--lns-arm", "on"])
    assert resolve_campaign(gated) is None
    refusal = usage_error(gated)
    assert refusal is not None
    assert "no LNS arm" in refusal


def test_the_heldout_roster_passes_the_drivers_roster_preflight(tmp_path: Path) -> None:
    """The roster comes from `heldout/bounds.csv`, every instance has its `.nl`,
    and the out-dir refusal is still about `benchmarks/instances/` as a whole --
    a scratch dir is fine and `heldout/` itself is refused."""
    roster = roster_from_bounds(HELDOUT_INST_DIR / "bounds.csv")
    assert len(roster) == 50
    args = make_args(tmp_path, campaign=TRANSFER_CAMPAIGN, inst_dir=HELDOUT_INST_DIR)
    assert _data_problems(args, roster) == []
    assert usage_error(args) is None
    inside = usage_error(make_args(tmp_path, out_dir=HELDOUT_INST_DIR / "scratch"))
    assert inside is not None
    assert "published" in inside


def test_report_only_scores_the_campaign_its_stamp_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    (out_dir / STAMP_NAME).write_text(
        campaign_stamp("abc1234", 10.0, [1, 2, 3], TRANSFER_ARMS, campaign=TRANSFER_CAMPAIGN)
    )
    write_results(out_dir / RESULTS_NAME, [("a", CONTROL_ARM, 1), ("a", "unproductive-100", 1)])
    assert main(["--out-dir", str(out_dir), "--report-only"]) == 0
    assert "issue #145" in capsys.readouterr().out

    assert main(["--out-dir", str(out_dir), "--report-only", "--campaign", "ablation"]) == 2
    assert "contradicts" in capsys.readouterr().err


def test_report_only_checks_completeness_against_the_stamped_seeds(tmp_path: Path) -> None:
    """`--report-only` does not repeat `--seeds`, so the plan the rows are checked
    against takes its seeds from the stamp, not from the default list."""
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    assert recorded_seeds(out_dir) is None
    (out_dir / STAMP_NAME).write_text(
        campaign_stamp("abc1234", 10.0, [4, 5, 6], TRANSFER_ARMS, campaign=TRANSFER_CAMPAIGN)
    )
    assert recorded_seeds(out_dir) == [4, 5, 6]

    (tmp_path / "inst").mkdir()
    (tmp_path / "inst" / "bounds.csv").write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks\n"
        "a,other,3,4,min,1,1,0\n"
    )
    results = _campaign_csv(
        out_dir / RESULTS_NAME,
        [
            {"instance": "a", "arm": arm.name, "seed": seed, "gap_to_bks%": 1.0 + 0.1 * seed}
            for seed in (4, 5, 6)
            for arm in TRANSFER_ARMS
        ],
    )
    args = make_args(tmp_path, campaign=TRANSFER_CAMPAIGN, out_dir=out_dir)
    assert args.seeds == [1, 2, 3]  # what a --report-only invocation carries
    stamped = report_for(args, results, gate=None)
    assert "lack a completed search" not in stamped
    # ... and nothing else blocks it either: the stamped seeds make it read clean.
    assert "INSIDE THE MEASURED NOISE FLOOR" in stamped
    (out_dir / STAMP_NAME).unlink()  # no stamp: falls back to --seeds, and says so
    assert "lack a completed search" in report_for(args, results, gate=None)


def test_the_transfer_dry_run_prints_the_grid_and_no_gate_probe(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dry run touches nothing and needs no build; it may refuse (no build
    dir in a test), but it still prints the plan it would run."""
    out_dir = tmp_path / "scratch"
    main(
        [
            "--out-dir",
            str(out_dir),
            "--campaign",
            TRANSFER_CAMPAIGN,
            "--time-limit",
            "10",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "campaign: transfer-145" in out
    assert "control[--unproductive-iters 300]" in out
    assert "unproductive-1000[--unproductive-iters 1000]" in out
    assert "campaign: 450 run(s)" in out
    assert "gate probe" not in out
    assert "estimated 1.2h of solving" in out  # 450 x 10s
    assert not out_dir.exists()


# --- the DAG-independent check (#205) -----------------------------------------


def test_a_run_scip_rejects_is_recorded_as_verify_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every run's verified row is checked outside the DAG before it is recorded."""
    monkeypatch.setattr(independent_check, "check_rows", REAL_CHECK_ROWS)
    monkeypatch.setattr(
        independent_check, "check_solution", lambda nl, sol: "SCIP rejects the assignment (c1)"
    )
    inner = fake_runner()
    seen_dirs: list[Path] = []

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        name = cmd[cmd.index("--instance") + 1]
        sol_dir = Path(cmd[cmd.index("--solution-dir") + 1])
        seen_dirs.append(sol_dir)
        (sol_dir / f"{name}.sol").write_text(
            f"{independent_check.SOLUTION_MAGIC}\ninstance {name}\nobjective 1\ncolumns 1\n0\n"
        )
        return inner(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    assert execute(make_args(tmp_path, lns_arm="off", seeds=[1]), "abc1234", ["a"], out_dir) == 0
    # One solution directory per run: arms of one instance must not share a file.
    assert len(set(seen_dirs)) == len(seen_dirs) == len(ARMS)
    with (out_dir / RESULTS_NAME).open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert all(r["feasible"] == "false" for r in rows)
    assert all(r["note"].startswith("VERIFY-FAILED(independent: ") for r in rows)


def test_a_run_the_check_cannot_apply_is_held_out_not_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ValueError from the check (an unrecognised SCIP model) must not end the campaign."""

    def refuse(*_a: object, **_k: object) -> list[str]:
        raise ValueError("SCIP sees 17 columns; the solution has 16")

    monkeypatch.setattr(independent_check, "check_rows", refuse)
    monkeypatch.setattr(subprocess, "run", fake_runner())
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    assert execute(make_args(tmp_path, lns_arm="off", seeds=[1]), "abc1234", ["a"], out_dir) == 0
    with (out_dir / RESULTS_NAME).open(newline="") as fh:
        notes = [r["note"] for r in csv.DictReader(fh)]
    assert notes and all("independent-check-error" in n for n in notes)
    assert not any(completed_search(n) for n in notes)
