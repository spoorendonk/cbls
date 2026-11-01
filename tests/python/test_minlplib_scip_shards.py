"""The sharded SCIP driver (#305): the shard layout it writes is the one gm_ratio reads.

A stub stands in for `reference_solve.py` (no SCIP, no `.nl` files): it takes
(instance, out, merged, mode) and writes a row, nothing, or hangs.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib.gm_ratio import load_scip_shards
from benchmarks.minlplib.reference_solve import load_bounds
from benchmarks.minlplib.scip_shards import (
    capped,
    main,
    reference_command,
    run_shards,
    solve_shard,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

SCIP_COLUMNS = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,wall_seconds,feasible,"
    "note,scip_dual_bound,scip_gap%,status,n_int_vars,read_seconds,solving_seconds,scip_version"
)

STUB = f'''
import sys, time
name, out, merged, mode = sys.argv[1:5]
row = (name + ",1,1,NaN,0,NaN,60,true,feasible,NaN,NaN,timelimit,0,0,60,"
       "SCIP 10.0.2 / PySCIPOpt 6.3.0 / 60s / seed 0")
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
    assert (shards / "a.failed").read_text() == "exit 1\n"


def test_a_capped_solve_is_a_failure_even_with_a_row_written(tmp_path: Path) -> None:
    shards = tmp_path / "shards"
    shards.mkdir()
    rc = solve_shard(shards, "a", stub_command(tmp_path, {"a": "hang"}), cap_seconds=1)
    assert rc == 124
    assert not (shards / "a.csv").exists()
    assert (shards / "a.failed").read_text() == "exit 124\n"


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
    (shards / "done.csv").write_text("kept\n")
    (shards / "retry.failed").write_text("exit 134\n")
    cmd = stub_command(tmp_path, {"done": "nothing", "retry": "ok"})
    exits = run_shards(shards, ["done", "retry"], cmd, jobs=2, cap_seconds=30)
    assert exits == {"done": None, "retry": 0}
    assert (shards / "done.csv").read_text() == "kept\n"
    assert (shards / "retry.csv").exists()
    assert not (shards / "retry.failed").exists()


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


def test_the_real_command_solves_one_instance_into_the_shard_paths(tmp_path: Path) -> None:
    cmd = reference_command(tmp_path / "nl", 60, 0)("x", tmp_path / "x.csv.tmp", tmp_path / "m")
    assert cmd[cmd.index("--instances") + 1] == "x"
    assert cmd[cmd.index("--time-limit") + 1] == "60"
    assert cmd[cmd.index("--seed") + 1] == "0"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "x.csv.tmp")
    assert capped(cmd, 900)[:4] == ["timeout", "-k", "10", "900"]


@pytest.mark.parametrize(
    ("extra", "why"),
    [(["--cap", "60"], "--cap must exceed"), (["--jobs", "0"], "--jobs must be")],
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
