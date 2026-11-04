"""The three-arm table of #321: cbls, CP-SAT fj + ls and CP-SAT at its defaults."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from benchmarks.mipfeas.primal_integral import FULL_ROSTER_SIZE, main

if TYPE_CHECKING:
    from pathlib import Path

_PASS: dict[str, object] = {"verdict": "pass", "reason": "ok", "tolerances": {"row": 1e-6}}
_BUDGET = 60.0


def _row(
    root: Path,
    engine: str,
    instance: str,
    *,
    objective: float | None,
    build: dict[str, str],
    verified: dict[str, object] | None = _PASS,
    **config: object,
) -> None:
    """One job's outputs, as the driver lays them out."""
    directory = root / engine
    directory.mkdir(parents=True, exist_ok=True)
    feasible = objective is not None
    record: dict[str, object] = {
        "status": "feasible" if feasible else "no_solution",
        "wall_seconds": _BUDGET,
        "budget_seconds": _BUDGET,
        **build,
        **config,
    }
    if feasible:
        record["objective"] = objective
    (directory / f"{instance}.json").write_text(json.dumps(record))
    if feasible and verified is not None:
        (directory / f"{instance}.verify.json").write_text(json.dumps(verified))


def _campaign(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two instances: all three arms solve `a`, only CP-SAT at defaults solves `b`."""
    main_dir = tmp_path / "defaults"
    ls_dir = tmp_path / "ls"
    roster = tmp_path / "roster.csv"
    roster.write_text("instance,reference_value,reference_kind\na,100,opt\nb,100,opt\n")
    cbls = {"commit_sha": "abc1234"}
    tools = {"ortools_version": "9.15"}
    _row(main_dir, "cbls", "a", objective=110.0, build=cbls, threads=4, workers_completed=4)
    _row(main_dir, "cbls", "b", objective=None, build=cbls, threads=4, workers_completed=4)
    _row(main_dir, "cpsat", "a", objective=100.0, build=tools, workers=0)
    _row(main_dir, "cpsat", "b", objective=100.0, build=tools, workers=0)
    _row(ls_dir, "cpsat", "a", objective=105.0, build=tools, workers=4)
    _row(ls_dir, "cpsat", "b", objective=None, build=tools, workers=4)
    return main_dir, ls_dir, roster


def _score(
    monkeypatch: pytest.MonkeyPatch,
    main_dir: Path,
    ls_dir: Path,
    roster: Path,
    out: Path,
    *extra: str,
) -> int:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "primal_integral.py",
            "--results-dir",
            str(main_dir),
            "--cpsat-ls-results-dir",
            str(ls_dir),
            "--roster",
            str(roster),
            "--budget",
            str(_BUDGET),
            "--out",
            str(out),
            *extra,
        ],
    )
    return main()


def _table_rows(text: str, first_cell: str) -> list[str]:
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if cells[0] == first_cell:
            return cells
    raise AssertionError(f"no row {first_cell!r} in:\n{text}")


def test_three_arm_table_reports_every_arm_and_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    out = tmp_path / "comparison.csv"
    assert _score(monkeypatch, main_dir, ls_dir, roster, out) == 0

    text = (tmp_path / "comparison_three_arm.md").read_text()
    assert _table_rows(text, "cbls")[1:4] == ["4", "1/2", "0"]
    assert _table_rows(text, "CP-SAT fj+ls")[1:4] == ["4", "1/2", "0"]
    assert _table_rows(text, "CP-SAT defaults")[1:4] == ["own choice (not recorded)", "2/2", "2"]
    # Compared, both, neither, A only, B only.
    assert _table_rows(text, "cbls / CP-SAT fj+ls")[1:] == ["2", "1", "1", "0", "0"]
    assert _table_rows(text, "cbls / CP-SAT defaults")[1:] == ["2", "1", "0", "0", "1"]
    assert _table_rows(text, "CP-SAT fj+ls / CP-SAT defaults")[1:] == ["2", "1", "0", "0", "1"]
    assert "WIRING CHECK" in text
    assert "9.15" in text


def test_three_arm_table_refuses_an_incomplete_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    (ls_dir / "cpsat" / "b.json").unlink()
    with pytest.raises(SystemExit):
        _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv")
    assert "cpsat-fj-ls: 1 not run: b" in capsys.readouterr().err
    assert not (tmp_path / "comparison_three_arm.md").exists()


def test_three_arm_table_refuses_an_unchecked_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    (ls_dir / "cpsat" / "a.verify.json").unlink()
    with pytest.raises(SystemExit):
        _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv")
    assert "cpsat-fj-ls: 1 with no usable verdict: a (no_verdict_file)" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("main_workers", "ls_workers", "message"),
    [
        (4, 4, "not CP-SAT at its defaults"),
        (0, 0, "not the fj + ls mode"),
        (0, 8, "symmetric rule"),
    ],
)
def test_three_arm_table_refuses_arms_that_are_not_what_they_say(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    main_workers: int,
    ls_workers: int,
    message: str,
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    tools = {"ortools_version": "9.15"}
    for instance, objective in (("a", 100.0), ("b", 100.0)):
        _row(main_dir, "cpsat", instance, objective=objective, build=tools, workers=main_workers)
    for instance, ls_objective in (("a", 105.0), ("b", None)):
        _row(ls_dir, "cpsat", instance, objective=ls_objective, build=tools, workers=ls_workers)
    with pytest.raises(SystemExit):
        _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv")
    assert message in capsys.readouterr().err


def test_three_arm_table_refuses_two_cp_sat_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    tools = {"ortools_version": "9.14"}
    _row(ls_dir, "cpsat", "a", objective=105.0, build=tools, workers=4)
    _row(ls_dir, "cpsat", "b", objective=None, build=tools, workers=4)
    with pytest.raises(SystemExit):
        _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv")
    assert "different builds" in capsys.readouterr().err


def test_a_rejected_ls_row_is_written_and_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    rejected: dict[str, object] = {
        "verdict": "fail",
        "reason": "row_violated",
        "tolerances": {"row": 1e-6},
    }
    _row(
        ls_dir,
        "cpsat",
        "a",
        objective=105.0,
        build={"ortools_version": "9.15"},
        verified=rejected,
        workers=4,
    )
    assert _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv") == 1
    text = (tmp_path / "comparison_three_arm.md").read_text()
    assert _table_rows(text, "CP-SAT fj+ls")[6] == "1"


def test_three_arm_table_refuses_arms_at_different_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    tools = {"ortools_version": "9.15"}
    cbls = {"commit_sha": "abc1234"}
    _row(main_dir, "cbls", "a", objective=110.0, build=cbls, threads=4, workers_completed=4, seed=1)
    _row(main_dir, "cbls", "b", objective=None, build=cbls, threads=4, workers_completed=4, seed=1)
    _row(main_dir, "cpsat", "a", objective=100.0, build=tools, workers=0)
    _row(main_dir, "cpsat", "b", objective=100.0, build=tools, workers=0)
    _row(ls_dir, "cpsat", "a", objective=105.0, build=tools, workers=4, seed=42)
    _row(ls_dir, "cpsat", "b", objective=None, build=tools, workers=4, seed=42)
    with pytest.raises(SystemExit):
        _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv")
    assert "different seeds" in capsys.readouterr().err


def test_three_arm_table_refuses_allow_unverified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    (ls_dir / "cpsat" / "a.verify.json").unlink()
    with pytest.raises(SystemExit):
        _score(
            monkeypatch,
            main_dir,
            ls_dir,
            roster,
            tmp_path / "comparison.csv",
            "--allow-unverified",
        )
    assert "--allow-unverified" in capsys.readouterr().err
    assert not (tmp_path / "comparison_three_arm.md").exists()


def test_three_arm_table_refuses_verdicts_under_two_tolerance_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    loose: dict[str, object] = {"verdict": "pass", "reason": "ok", "tolerances": {"row": 1e-3}}
    tools = {"ortools_version": "9.15"}
    _row(ls_dir, "cpsat", "a", objective=105.0, build=tools, verified=loose, workers=4)
    with pytest.raises(ValueError, match="tolerance sets"):
        _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv")


def test_an_errored_row_is_named_in_the_table_and_out_of_the_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    (ls_dir / "cpsat" / "b.json").write_text(
        json.dumps(
            {
                "status": "killed",
                "wall_seconds": _BUDGET,
                "budget_seconds": _BUDGET,
                "ortools_version": "9.15",
                "workers": 4,
            }
        )
    )
    assert _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv") == 0
    text = (tmp_path / "comparison_three_arm.md").read_text()
    assert _table_rows(text, "CP-SAT fj+ls")[5] == "1"
    assert _table_rows(text, "cbls / CP-SAT fj+ls")[1] == "1"


def test_the_ls_arm_gets_its_own_report_and_its_defects_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    main_dir, ls_dir, roster = _campaign(tmp_path)
    _row(ls_dir, "cpsat", "a", objective=50.0, build={"ortools_version": "9.15"}, workers=4)
    assert _score(monkeypatch, main_dir, ls_dir, roster, tmp_path / "comparison.csv") == 1
    assert (tmp_path / "comparison_fj_ls.csv").exists()
    report = (tmp_path / "comparison_fj_ls_report.md").read_text()
    assert "fj` + `ls" in report


def test_a_held_out_table_carries_no_wiring_stamp_and_shows_the_chosen_worker_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    names = [f"i{n}" for n in range(FULL_ROSTER_SIZE)]
    roster = tmp_path / "roster.csv"
    roster.write_text(
        "instance,reference_value,reference_kind\n" + "".join(f"{n},100,opt\n" for n in names)
    )
    held = tmp_path / "tuned.txt"
    held.write_text("i0\n")
    main_dir = tmp_path / "defaults"
    ls_dir = tmp_path / "ls"
    tools = {"ortools_version": "9.15"}
    cbls = {"commit_sha": "abc1234"}
    for name in names[1:]:
        _row(main_dir, "cbls", name, objective=100.0, build=cbls, threads=4, workers_completed=4)
        _row(main_dir, "cpsat", name, objective=100.0, build=tools, workers=0, workers_started=16)
        _row(ls_dir, "cpsat", name, objective=100.0, build=tools, workers=4)
    out = tmp_path / "comparison.csv"
    assert _score(monkeypatch, main_dir, ls_dir, roster, out, "--held-out", str(held)) == 0
    text = (tmp_path / "comparison_three_arm.md").read_text()
    assert "WIRING CHECK" not in text
    assert _table_rows(text, "CP-SAT defaults")[1] == "own choice (16)"
