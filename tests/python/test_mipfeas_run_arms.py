"""The MIPfeas arm driver (#350, for #331): plan, commands, stamp and report."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from benchmarks.mipfeas import run_arms
from benchmarks.mipfeas.run_arms import CAMPAIGNS, CONTROL, Arm, Cell

if TYPE_CHECKING:
    from pathlib import Path

ARMS = CAMPAIGNS["presolve-322"]

#: Stands in for `run_benchmark.py` behind the lock prefix: logs its argv, and
#: writes a verified feasible row whose objective depends on the arm, so the
#: report has something to rank. `--presolve` halves the gap to the reference;
#: an instance named `bad` gets a failing verdict.
FAKE_RUNNER = """
import json, sys
from pathlib import Path
argv = sys.argv[1:]
log = Path(sys.argv[0]).with_name("calls.log")
with log.open("a") as fh:
    fh.write(json.dumps(argv) + "\\n")
def value(flag):
    return argv[argv.index(flag) + 1]
inst = value("--roster").rsplit("/", 1)[1].removesuffix(".csv")
out = Path(value("--results-dir")) / "cbls"
out.mkdir(parents=True, exist_ok=True)
seed = int(value("--seed"))
presolve = "--presolve" in argv
objective = 100.0 + (10.0 if presolve else 20.0) + seed
record = {
    "status": "feasible", "objective": objective, "wall_seconds": 1.0,
    "commit_sha": "abc1234", "seed": seed, "presolve": presolve,
    "compound_propagation": "--compound-propagation" in argv,
    "setup_seconds": 0.5, "peak_rss_kib": 2048,
}
(out / f"{inst}.json").write_text(json.dumps(record))
(out / f"{inst}.sol").write_text("x 1\\n")
(out / f"{inst}.trace.csv").write_text(f"time_seconds,objective\\n0.5,{objective}\\n")
verdict = "fail" if inst == "bad" else "pass"
(out / f"{inst}.verify.json").write_text(json.dumps({
    "verdict": verdict, "reason": "" if verdict == "pass" else "row_violation",
    "marginal": False, "tolerances": {"row_absolute": 1e-06},
}))
"""


def _campaign(tmp_path: Path, instances: list[str]) -> tuple[Path, Path, Path]:
    """An instance dir with a reference roster, a names roster and the fake runner."""
    inst_dir = tmp_path / "inst"
    inst_dir.mkdir()
    lines = ["instance,reference_value,reference_kind"]
    lines += [f"{i},100.0,opt" for i in instances]
    (inst_dir / "roster.csv").write_text("\n".join(lines) + "\n")
    names = tmp_path / "tuned.txt"
    names.write_text("# dev set\n" + "\n".join(instances) + "\n")
    fake = tmp_path / "fake_runner.py"
    fake.write_text(FAKE_RUNNER)
    return inst_dir, names, fake


def _main(tmp_path: Path, inst_dir: Path, names: Path, fake: Path, *extra: str) -> int:
    return run_arms.main(
        [
            "--out-dir",
            str(tmp_path / "out"),
            "--budget",
            "2",
            "--seeds",
            "1",
            "2",
            "--roster",
            str(names),
            "--inst-dir",
            str(inst_dir),
            "--lock-command",
            f"{sys.executable} {fake}",
            *extra,
            "--",
            "--engine-commit",
            "abc1234",
        ]
    )


def test_the_plan_is_instance_major_then_seed_then_arm() -> None:
    cells = run_arms.plan(["a", "b"], [1, 2], ARMS)
    assert len(cells) == 2 * 2 * 4
    assert [(c.instance, c.seed) for c in cells[:4]] == [("a", 1)] * 4
    assert [c.arm.name for c in cells[:4]] == [CONTROL, "presolve", "compound-propagation", "both"]
    assert (cells[4].instance, cells[4].seed) == ("a", 2)
    assert cells[8].instance == "b"


def test_the_presolve_campaign_arms_carry_their_flags() -> None:
    flags = {arm.name: arm.flags for arm in ARMS}
    assert flags == {
        CONTROL: (),
        "presolve": ("--presolve",),
        "compound-propagation": ("--compound-propagation",),
        "both": ("--presolve", "--compound-propagation"),
    }


def test_a_cell_runs_one_locked_cbls_solve_into_its_arm_and_seed_directory(
    tmp_path: Path,
) -> None:
    cell = Cell("neos5", 3, Arm("both", ("--presolve", "--compound-propagation")))
    cmd = run_arms.runner_command(
        cell,
        tmp_path / "r.csv",
        tmp_path,
        60.0,
        tmp_path / "inst",
        ["--cbls-threads", "4"],
        ["xk", "lock", "--"],
    )
    assert cmd[:3] == ["xk", "lock", "--"]
    assert cmd[4] == str(run_arms.RUN_BENCHMARK)
    assert cmd[cmd.index("--engines") + 1] == "cbls"
    assert cmd[cmd.index("--results-dir") + 1] == str(tmp_path / "both" / "seed3")
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--budget") + 1] == "60"
    assert cmd[-4:] == ["--presolve", "--compound-propagation", "--cbls-threads", "4"]


@pytest.mark.parametrize(
    "flag", ["--presolve", "--seed=5", "--no-presolve-aggregation", "--results-dir", "--force"]
)
def test_a_flag_the_arms_own_is_refused_in_the_pass_through(
    tmp_path: Path, flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    inst_dir, names, fake = _campaign(tmp_path, ["a"])
    code = run_arms.main(
        ["--out-dir", str(tmp_path / "out"), "--roster", str(names), "--inst-dir", str(inst_dir)]
        + ["--lock-command", f"{sys.executable} {fake}", "--", flag]
    )
    assert code == 2
    assert "cannot be passed through" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_a_resume_with_other_seeds_is_refused_and_a_report_needs_only_the_campaign(
    tmp_path: Path,
) -> None:
    stamp = run_arms.campaign_stamp("presolve-322", ["a"], [1, 2], 60.0, ["--x"], "abc")
    assert run_arms.check_stamp(tmp_path, stamp) is None
    assert run_arms.check_stamp(tmp_path, stamp) is None
    other = run_arms.campaign_stamp("presolve-322", ["a"], [1, 3], 60.0, ["--x"], "abc")
    refusal = run_arms.check_stamp(tmp_path, other)
    assert refusal is not None and "seeds" in refusal
    # A report ignores the pass-through and the commit, but not the seeds.
    rescored = run_arms.campaign_stamp("presolve-322", ["a"], [1, 2], 60.0, [], "def")
    assert run_arms.check_stamp(tmp_path, rescored, report_only=True) is None
    refusal = run_arms.check_stamp(tmp_path, other, report_only=True)
    assert refusal is not None and "seeds" in refusal
    assert run_arms.check_stamp(tmp_path / "none", stamp, report_only=True) is not None


def test_the_campaign_runs_every_cell_once_resumes_and_reports_against_control(
    tmp_path: Path,
) -> None:
    inst_dir, names, fake = _campaign(tmp_path, ["a", "b"])
    assert _main(tmp_path, inst_dir, names, fake) == 0
    calls = [json.loads(line) for line in (tmp_path / "calls.log").read_text().splitlines()]
    assert len(calls) == 2 * 2 * 4
    assert all(c[1] == str(run_arms.RUN_BENCHMARK) for c in calls)
    # A second invocation finds every cell verified and runs nothing.
    assert _main(tmp_path, inst_dir, names, fake) == 0
    assert len((tmp_path / "calls.log").read_text().splitlines()) == len(calls)

    report = (tmp_path / "out" / run_arms.REPORT_NAME).read_text()
    assert "Engine commit: abc1234" in report
    assert "| control | 1 | 2/2 | 0 | 0 | 0 |" in report
    # Every arm is feasible everywhere, presolve arms closer to the reference.
    pooled = report.split("## Pooled over seeds", 1)[1]
    rows = {line.split(" | ")[0].strip("| "): line for line in pooled.splitlines()[4:]}
    assert set(rows) == {CONTROL, "presolve", "compound-propagation", "both"}
    assert " 1.0000 " in rows[CONTROL] and " 1.0000 " in rows["compound-propagation"]
    ratio = float(rows["presolve"].split(" | ")[4])
    assert ratio < 1.0
    assert all("| none |" in row for row in rows.values())


def test_a_rejected_solution_fails_the_campaign(tmp_path: Path) -> None:
    inst_dir, names, fake = _campaign(tmp_path, ["bad"])
    assert _main(tmp_path, inst_dir, names, fake) == 1
    report = (tmp_path / "out" / run_arms.REPORT_NAME).read_text()
    assert "| control | 1 | 0/1 | 0 | 1 | 0 |" in report


def test_rows_from_two_engine_commits_are_refused(tmp_path: Path) -> None:
    inst_dir, names, fake = _campaign(tmp_path, ["a"])
    assert _main(tmp_path, inst_dir, names, fake) == 0
    row = tmp_path / "out" / "both" / "seed2" / "cbls" / "a.json"
    record = json.loads(row.read_text())
    record["commit_sha"] = "fff0000"
    row.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="2 engine commits"):
        _main(tmp_path, inst_dir, names, fake, "--report-only")
