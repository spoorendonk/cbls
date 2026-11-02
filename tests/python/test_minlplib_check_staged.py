"""Tests for `benchmarks/minlplib/check_staged.py` (issue #307).

Offline: the staging directory is synthetic, and SCIP's verdict on a solution
is stubbed (`independent_check.check_solution`); the rest of the check is the
real `independent_check.check_rows`. Nothing is solved.
"""

from __future__ import annotations

import socket
import subprocess
from typing import TYPE_CHECKING

import pytest

from benchmarks.common import jobs
from benchmarks.common.jobs import wallclock_lock
from benchmarks.minlplib import independent_check
from benchmarks.minlplib.check_staged import EXIT_CHECK_ERRORS, EXIT_REFUSED, main
from benchmarks.minlplib.run_benchmark import (
    STAMP_NAME,
    checked_marker,
    mark_checked,
    roster_fingerprint,
    staged_checked,
)
from benchmarks.minlplib.runner import RUNNER_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

HEADER = ",".join(RUNNER_COLUMNS)


@pytest.fixture(autouse=True)
def _lock_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jobs, "WALLCLOCK_LOCK", tmp_path / "lock" / "wallclock.lock")


def row(name: str, *, feasible: bool = True, note: str = "feasible", sha: str = "old0000") -> str:
    cells = {c: "NaN" for c in RUNNER_COLUMNS}
    cells.update(
        instance=name,
        objective="1",
        feasible="true" if feasible else "false",
        note=note,
        commit_sha=sha,
        search_config="threads=16",
    )
    return ",".join(cells[c] for c in RUNNER_COLUMNS)


def stage_row(stage: Path, name: str, text: str, *, solution: bool = True) -> None:
    (stage / f"{name}.csv").write_text(text)
    if solution:
        (stage / f"{name}.sol").write_text(
            f"{independent_check.SOLUTION_MAGIC}\ninstance {name}\nobjective 1\ncolumns 1\n0\n"
        )


def make_dirs(tmp_path: Path, names: list[str]) -> tuple[Path, Path]:
    inst = tmp_path / "nl"
    inst.mkdir()
    lines = ["instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks"]
    lines += [f"{n},other,1,1,min,1,,0" for n in names]
    (inst / "bounds.csv").write_text("\n".join(lines) + "\n")
    stage = tmp_path / "stage"
    stage.mkdir()
    write_stamp(stage, roster=roster_fingerprint(inst), host=socket.gethostname())
    return inst, stage


def write_stamp(stage: Path, *, roster: str, host: str) -> None:
    """A staging stamp as an older driver wrote it: another commit than this checkout's."""
    (stage / STAMP_NAME).write_text(
        f"commit=old0000\ntime-limit=60\nseed=1\nthreads=16\nhost={host}\nroster={roster}\n"
    )


def test_rows_staged_at_another_commit_are_checked_and_marked_without_a_solve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The #256 case: rows staged by a driver older than the marker, at another commit."""
    names = ["a", "b", "header", "torn", "threw", "rejected", "marked"]
    inst, stage = make_dirs(tmp_path, names)
    stage_row(stage, "a", f"{HEADER}\n{row('a')}\n")
    stage_row(stage, "b", f"{HEADER}\n{row('b')}\n")
    stage_row(stage, "header", f"{HEADER}\n", solution=False)
    stage_row(stage, "torn", f"{HEADER}\n{row('torn')}", solution=False)
    stage_row(stage, "threw", f"{HEADER}\n{row('threw', feasible=False, note='solve-error(x)')}\n")
    (stage / "rejected.csv.rejected").write_text(f"{HEADER}\n{row('rejected')}\n")
    stage_row(stage, "marked", f"{HEADER}\n{row('marked')}\n")
    mark_checked(stage, "marked")
    rejects = {"b", "marked"}  # `marked` would be demoted if it were read again
    monkeypatch.setattr(
        independent_check,
        "check_solution",
        lambda nl, sol: "SCIP rejects the assignment" if sol.instance in rejects else None,
    )

    def no_solve(*_a: object, **_k: object) -> object:
        raise AssertionError("check_staged ran a process")

    monkeypatch.setattr(subprocess, "run", no_solve)

    assert main(["--inst-dir", str(inst), "--staging-dir", str(stage)]) == 0

    assert staged_checked(stage, "a")
    assert staged_checked(stage, "b")
    assert "VERIFY-FAILED" in (stage / "b.csv").read_text()
    assert "VERIFY-FAILED" not in (stage / "marked.csv").read_text()
    assert staged_checked(stage, "marked")
    for name in ("header", "torn", "threw", "rejected"):
        assert not checked_marker(stage, name).exists(), name
    out = capsys.readouterr().out
    assert "checked and marked: 2\n  of them demoted to VERIFY-FAILED: 1 (b)\n" in out
    assert "already marked: 1\n" in out
    assert "no <instance>.csv (rejected-only or not staged): 1\n" in out
    assert "left unmarked: 3\n" in out
    assert "  header: no whole row (header-only or torn)\n" in out
    assert "  torn: no whole row (header-only or torn)\n" in out
    assert "  threw: unstageable note\n" in out
    assert "check raised, left unmarked: 0\n" in out
    assert "to clear a row left unmarked: re-solve it with run_benchmark.py" in out


def test_instances_limits_the_check_to_the_names_given(tmp_path: Path) -> None:
    inst, stage = make_dirs(tmp_path, ["a", "b"])
    stage_row(stage, "a", f"{HEADER}\n{row('a', feasible=False, note='infeasible')}\n")
    stage_row(stage, "b", f"{HEADER}\n{row('b', feasible=False, note='infeasible')}\n")
    argv = ["--inst-dir", str(inst), "--staging-dir", str(stage), "--instances", "b"]
    assert main(argv) == 0
    assert not checked_marker(stage, "a").exists()
    assert staged_checked(stage, "b")


def test_it_refuses_while_a_driver_holds_the_wall_clock_lock(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inst, stage = make_dirs(tmp_path, ["a"])
    stage_row(stage, "a", f"{HEADER}\n{row('a', feasible=False, note='infeasible')}\n")
    with wallclock_lock("run_benchmark.py"):
        assert main(["--inst-dir", str(inst), "--staging-dir", str(stage)]) == EXIT_REFUSED
    assert "run_benchmark.py" in capsys.readouterr().err
    assert not checked_marker(stage, "a").exists()


def test_a_missing_staging_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inst, _ = make_dirs(tmp_path, ["a"])
    argv = ["--inst-dir", str(inst), "--staging-dir", str(tmp_path / "nowhere")]
    assert main(argv) == EXIT_REFUSED
    assert "not a directory" in capsys.readouterr().err


@pytest.mark.parametrize("field", ["roster", "host"])
def test_a_stamp_for_another_roster_or_host_is_refused_before_anything_is_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    field: str,
) -> None:
    """Checked against another roster's models, a valid row would be demoted for good."""
    inst, stage = make_dirs(tmp_path, ["a"])
    stage_row(stage, "a", f"{HEADER}\n{row('a')}\n")
    stamp = {"roster": roster_fingerprint(inst), "host": socket.gethostname()}
    stamp[field] = "elsewhere"
    write_stamp(stage, **stamp)
    monkeypatch.setattr(independent_check, "check_solution", lambda nl, sol: "wrong model")
    assert main(["--inst-dir", str(inst), "--staging-dir", str(stage)]) == EXIT_REFUSED
    assert f"{field}=elsewhere" in capsys.readouterr().err
    assert "VERIFY-FAILED" not in (stage / "a.csv").read_text()
    assert not checked_marker(stage, "a").exists()


@pytest.mark.parametrize("missing", ["bounds", "stamp"])
def test_no_roster_or_no_stamp_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    """A mistyped --inst-dir must not report "checked and marked: 0" and exit 0."""
    inst, stage = make_dirs(tmp_path, ["a"])
    stage_row(stage, "a", f"{HEADER}\n{row('a', feasible=False, note='infeasible')}\n")
    if missing == "bounds":
        (inst / "bounds.csv").unlink()
    else:
        (stage / STAMP_NAME).unlink()
    assert main(["--inst-dir", str(inst), "--staging-dir", str(stage)]) == EXIT_REFUSED
    assert ("bounds.csv" if missing == "bounds" else STAMP_NAME) in capsys.readouterr().err
    assert not checked_marker(stage, "a").exists()


def test_a_row_whose_check_raises_is_listed_and_the_rest_are_still_marked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inst, stage = make_dirs(tmp_path, ["a", "nosol", "z"])
    stage_row(stage, "a", f"{HEADER}\n{row('a')}\n")
    stage_row(stage, "nosol", f"{HEADER}\n{row('nosol')}\n", solution=False)
    stage_row(stage, "z", f"{HEADER}\n{row('z')}\n")
    monkeypatch.setattr(independent_check, "check_solution", lambda nl, sol: None)
    argv = ["--inst-dir", str(inst), "--staging-dir", str(stage)]
    assert main(argv) == EXIT_CHECK_ERRORS
    assert staged_checked(stage, "a")
    assert staged_checked(stage, "z")
    assert not checked_marker(stage, "nosol").exists()
    out = capsys.readouterr().out
    assert "checked and marked: 2\n" in out
    assert "check raised, left unmarked: 1\n  nosol: FileNotFoundError: " in out
