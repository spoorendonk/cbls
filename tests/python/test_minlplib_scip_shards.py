"""The sharded SCIP driver (#305): the shard layout it writes is the one gm_ratio reads.

A stub stands in for `reference_solve.py` (no SCIP, no `.nl` files): it takes
(instance, out, merged, mode) and writes a row, nothing, or hangs.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.common.records import append_attempt, recorded_attempts
from benchmarks.minlplib.gm_ratio import load_scip_shards
from benchmarks.minlplib.reference_solve import load_bounds
from benchmarks.minlplib.scip_shards import (
    capped,
    main,
    reference_command,
    reset_on_new_stamp,
    run_shards,
    shard_stamp,
    solve_shard,
)

if TYPE_CHECKING:
    from collections.abc import Callable

SCIP_COLUMNS = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,wall_seconds,feasible,"
    "note,scip_dual_bound,scip_gap%,status,n_int_vars,read_seconds,solving_seconds,scip_version"
)

STUB = f'''
import os, sys, time
name, out, merged, mode = sys.argv[1:5]
if mode == "abort":
    os.abort()
if mode == "slow":
    time.sleep(0.3)
    mode = "ok"
row = (name + ",1,1,NaN,0,NaN,60,true,feasible,NaN,NaN,timelimit,0,0,60,"
       "SCIP 10.0.2 / PySCIPOpt 6.3.0 / 60s / seed 0")
if mode == "torn":
    open(out, "w").write("{SCIP_COLUMNS}\\n" + name + ",1,")
if mode in ("ok", "row-then-exit-3"):
    open(out, "w").write("{SCIP_COLUMNS}\\n" + row + "\\n")
    open(merged, "w").write("merged\\n")
if mode == "hang":
    open(out, "w").write("{SCIP_COLUMNS}\\n" + row + "\\n")
    time.sleep(60)
sys.exit(3 if mode == "row-then-exit-3" else 1 if mode == "nothing" else 0)
'''


def stub_command(tmp_path: Path, modes: dict[str, str]) -> Callable[[str, Path, Path], list[str]]:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)

    def command(name: str, out: Path, merged: Path) -> list[str]:
        return [sys.executable, str(stub), name, str(out), str(merged), modes[name]]

    return command


def write_bounds(path: Path, names: list[str]) -> Path:
    lines = ["instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks"]
    lines += [f"{name},other,1,1,min,1,,0" for name in names]
    path.write_text("\n".join(lines) + "\n")
    return path


def test_clean_exit_publishes_the_row_and_its_merged_csv(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    rc = solve_shard(shards, "a", stub_command(tmp_path, {"a": "ok"}), cap_seconds=30)
    assert rc == 0
    assert (shards / "a.csv").read_text().startswith("instance,")
    assert (shards / "a.merged.csv").exists()
    assert not (shards / "a.csv.tmp").exists()
    assert not (shards / "a.failed").exists()
    assert (shards / "a.log").exists()


def test_no_row_leaves_a_failure_marker_with_the_exit(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    rc = solve_shard(shards, "a", stub_command(tmp_path, {"a": "nothing"}), cap_seconds=30)
    assert rc == 1
    assert not (shards / "a.csv").exists()
    assert recorded_attempts(shards / "a.failed") == ["exit 1"]


def test_a_retry_that_fails_again_appends_a_second_attempt(tmp_path: Path) -> None:
    """#312: the scorer counts these lines, so a retry must not overwrite the first."""
    shards = tmp_path / "shards"
    shards.mkdir()
    for outcome, cap in (("abort", 30), ("hang", 1)):
        solve_shard(shards, "a", stub_command(tmp_path, {"a": outcome}), cap_seconds=cap)
    assert recorded_attempts(shards / "a.failed") == ["exit 134", "exit 124"]


def test_another_budget_or_seed_clears_the_attempts_and_the_same_one_keeps_them(
    tmp_path: Path,
) -> None:
    """#312: a smoke run's failure into the campaign's directory is no retry of it."""
    shards = tmp_path / "shards"
    shards.mkdir()
    assert reset_on_new_stamp(shards, shard_stamp(5, 0)) == []
    append_attempt(shards / "a.failed", 124)
    assert reset_on_new_stamp(shards, shard_stamp(5, 0)) == []
    assert recorded_attempts(shards / "a.failed") == ["exit 124"], "same stamp: a real retry"
    assert reset_on_new_stamp(shards, shard_stamp(60, 0)) == ["a"]
    assert not (shards / "a.failed").exists()
    append_attempt(shards / "a.failed", 134)
    assert reset_on_new_stamp(shards, shard_stamp(60, 1)) == ["a"]


def test_a_torn_attempt_line_is_dropped_not_glued_onto_the_next(tmp_path: Path) -> None:
    """A kill mid-append loses that attempt, never the next one with it."""
    marker = tmp_path / "a.failed"
    marker.write_text("exit 134 2026-11-02T08:00:00Z\nexit 13")
    append_attempt(marker, 124)
    assert recorded_attempts(marker) == ["exit 134", "exit 124"]


def test_a_capped_solve_is_a_failure_even_with_a_row_written(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    rc = solve_shard(shards, "a", stub_command(tmp_path, {"a": "hang"}), cap_seconds=1)
    assert rc == 124
    assert not (shards / "a.csv").exists()
    assert recorded_attempts(shards / "a.failed") == ["exit 124"]


def test_a_solve_dying_by_signal_is_recorded_as_sh_reports_it(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    rc = solve_shard(shards, "a", stub_command(tmp_path, {"a": "abort"}), cap_seconds=30)
    assert rc == 134
    assert recorded_attempts(shards / "a.failed") == ["exit 134"]


def test_ctrl_c_starts_no_further_solve(tmp_path: Path) -> None:
    names = [f"i{k}" for k in range(30)]
    shards = tmp_path / "shards"
    script = tmp_path / "drive.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from benchmarks.minlplib.scip_shards import run_shards\n"
        f"stub = {str(tmp_path / 'stub.py')!r}\n"
        "def cmd(name, out, merged):\n"
        "    return [sys.executable, stub, name, str(out), str(merged), 'slow']\n"
        f"run_shards(Path({str(shards)!r}), {names!r}, cmd, jobs=1, cap_seconds=30)\n"
    )
    (tmp_path / "stub.py").write_text(STUB)
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONPATH": str(repo)}
    child = subprocess.Popen([sys.executable, str(script)], env=env, cwd=repo)
    try:
        deadline = time.monotonic() + 20
        while not list(shards.glob("*.csv")) and time.monotonic() < deadline:
            time.sleep(0.05)
        child.send_signal(signal.SIGINT)
        child.wait(timeout=20)
    finally:
        child.kill()
    started = len(list(shards.glob("*.log")))
    assert child.returncode != 0
    assert 1 <= started < len(names)


def test_a_row_from_a_process_that_exited_by_itself_is_kept(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    cmd = stub_command(tmp_path, {"a": "row-then-exit-3"})
    assert solve_shard(shards, "a", cmd, cap_seconds=30) == 3
    assert (shards / "a.csv").exists()
    assert not (shards / "a.failed").exists()


def test_resume_skips_a_row_and_a_retry_clears_the_failure_marker(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    kept = "h1,h2\nkept,row\n"
    (shards / "done.csv").write_text(kept)
    (shards / "torn.csv").write_text("h1,h2\ntorn,")
    (shards / "retry.failed").write_text("exit 134\n")
    cmd = stub_command(tmp_path, {"done": "nothing", "torn": "ok", "retry": "ok"})
    exits = run_shards(shards, ["done", "torn", "retry"], cmd, jobs=2, cap_seconds=30)
    assert exits == {"done": None, "torn": 0, "retry": 0}
    assert (shards / "done.csv").read_text() == kept
    assert (shards / "torn.csv").read_text().startswith("instance,")
    assert (shards / "retry.csv").exists()
    assert not (shards / "retry.failed").exists()


def test_a_torn_row_is_not_published(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    rc = solve_shard(shards, "a", stub_command(tmp_path, {"a": "torn"}), cap_seconds=30)
    assert rc == 0
    assert not (shards / "a.csv").exists()
    assert recorded_attempts(shards / "a.failed") == ["exit 0"]


def test_the_scorer_reads_the_shards_the_driver_writes(tmp_path: Path) -> None:
    names = ["good", "dead"]
    roster = {b.instance: b for b in load_bounds(write_bounds(tmp_path / "bounds.csv", names))}
    shards = tmp_path / "shards"
    cmd = stub_command(tmp_path, {"good": "ok", "dead": "nothing"})
    run_shards(shards, names, cmd, jobs=2, cap_seconds=30)
    side = load_scip_shards(shards, roster)
    assert set(side.rows) == {"good"}
    assert side.rows["good"].feasible
    assert side.failures == {"dead": "exit 1"}
    assert side.attempts == {"dead": 1}


def test_the_real_command_solves_one_instance_into_the_shard_paths(tmp_path: Path) -> None:
    cmd = reference_command(tmp_path / "nl", 60, 0)("x", tmp_path / "x.csv.tmp", tmp_path / "m")
    assert cmd[cmd.index("--instances") + 1] == "x"
    assert cmd[cmd.index("--time-limit") + 1] == "60"
    assert cmd[cmd.index("--seed") + 1] == "0"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "x.csv.tmp")
    assert capped(cmd, 900)[:4] == ["timeout", "-k", "10", "900"]


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        (["--cap", "60"], "--cap must exceed"),
        (["--jobs", "0"], "--jobs must be"),
        (["--time-limit", "0"], "--time-limit must be positive"),
        (["--seed", "-1"], "--seed must be non-negative"),
    ],
)
def test_bad_limits_are_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], why: str
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--nl-dir", str(tmp_path), "--shards", str(tmp_path / "s"), *extra])
    assert exc.value.code == 2
    assert why in capsys.readouterr().err


def test_an_instance_outside_the_roster_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_bounds(tmp_path / "bounds.csv", ["a"])
    argv = ["--nl-dir", str(tmp_path), "--shards", str(tmp_path / "s"), "--instances", "zz"]
    assert main(argv) == 2
    assert "not in the roster: zz" in capsys.readouterr().err
    assert not (tmp_path / "s").exists()
