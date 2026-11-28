"""The uc-chped runner's UC hook flags and its evidence chain (#462).

`--solution-out` writes each measured row's (y, p); `check_solutions.py`, the
independent Pedroso-2014 checker from #445, recomputes feasibility and cost
from the instance file alone. Both run as built, from the shell.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
UC_CHPED_BINARY = ROOT / "build" / "cbls_uc_chped"
CHECKER = ROOT / "benchmarks" / "uc-chped" / "check_solutions.py"
INSTANCES = ROOT / "benchmarks" / "instances" / "uc-chped"

# #445's candidate (its report), one horizon, a 1 s budget.
CANDIDATE = [
    "--uc-hook",
    "full",
    "--uc-infeasible",
    "--uc-repair",
    "--uc-ils",
    "0.05",
    "--uc-ils-stall",
    "300",
    "--uc-ils-local",
    "16",
    "--uc-ls-polish",
    "2",
    "--uc-ls-window",
    "2",
    "--uc-ls-jcap",
    "24",
    "--uc-elite",
    "0.5",
]


def _run(
    tmp_path: Path, *args: str, inst_dir: Path = INSTANCES
) -> subprocess.CompletedProcess[str]:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    return subprocess.run(
        [str(UC_CHPED_BINARY), str(inst_dir), "--instance", "ucp13", "--periods", "3"]
        + ["--out", str(tmp_path / "o.csv"), *args],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _check(solutions: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CHECKER), str(INSTANCES), str(solutions)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def _measured(tmp_path: Path) -> list[dict[str, str]]:
    lines = [ln for ln in (tmp_path / "o.csv").read_text().splitlines() if not ln.startswith("#")]
    return [r for r in csv.DictReader(lines) if r["method"] == "CBLS ViolationLS"]


@pytest.mark.parametrize(
    "arm",
    [
        # Clock-free, so the verdict cannot depend on load.
        ["--no-time-limit", "--max-iterations", "20000"],
        ["--time-limit", "1", *CANDIDATE],
        # --no-float-hook turns off the generic hook only, not the UC one.
        ["--time-limit", "1", "--no-float-hook", "--uc-hook", "ed", "--uc-repair"],
        ["--time-limit", "1", "--threads", "2", *CANDIDATE],
    ],
    ids=["default", "candidate", "ed-no-float-hook", "candidate-threads2"],
)
def test_check_solutions_accepts_every_written_solution(arm: list[str], tmp_path: Path) -> None:
    sols = tmp_path / "s.jsonl"
    result = _run(tmp_path, "--demand-equality", "--verify", "--solution-out", str(sols), *arm)
    assert result.returncode == 0, result.stderr

    measured = _measured(tmp_path)
    assert [r["verified"] for r in measured] == ["true"], measured
    lines = sols.read_text().splitlines()
    assert len(lines) == 1
    assert float(json.loads(lines[0])["objective"]) == pytest.approx(
        float(measured[0]["objective"]), rel=1e-9
    )
    checked = _check(sols)
    assert checked.returncode == 0, checked.stdout
    assert checked.stdout.startswith("OK"), checked.stdout

    if "--uc-hook" in arm:
        cell = measured[0]["search_config"].split(";")
        assert f"uc_hook={arm[arm.index('--uc-hook') + 1]}" in cell, cell
        assert "uc_repair=on" in cell, cell
        workers = "2" if "--threads" in arm else "1"
        stats = [ln for ln in result.stdout.splitlines() if "uc-hook (" in ln]
        assert stats and stats[0].strip().startswith(f"uc-hook ({workers}):"), result.stdout
        calls = int(stats[0].split("):")[1].split()[0])
        assert calls > 0, stats[0]
    else:
        assert "uc_hook" not in measured[0]["search_config"]


def test_check_solutions_rejects_a_tampered_solution(tmp_path: Path) -> None:
    """The checker is not vacuous: one unit's dispatch moved fails demand."""
    sols = tmp_path / "s.jsonl"
    result = _run(
        tmp_path,
        *["--demand-equality", "--no-time-limit", "--max-iterations", "20000"],
        *["--solution-out", str(sols)],
    )
    assert result.returncode == 0, result.stderr
    sol = json.loads(sols.read_text())
    on = next(u for u, row in enumerate(sol["y"]) if row[0] == 1)
    sol["p"][on][0] += 1.0
    tampered = tmp_path / "t.jsonl"
    tampered.write_text(json.dumps(sol) + "\n")
    checked = _check(tampered)
    assert checked.returncode == 1, checked.stdout
    assert checked.stdout.startswith("FAIL") and "demand[0]" in checked.stdout, checked.stdout


def test_check_solutions_fails_on_a_file_with_no_solutions(tmp_path: Path) -> None:
    """Every row vetoed leaves an empty file: no evidence, not a pass."""
    empty = tmp_path / "s.jsonl"
    empty.write_text("\n")
    checked = _check(empty)
    assert checked.returncode == 1, checked.stdout
    assert "no solutions to check" in checked.stdout, checked.stdout


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--uc-ils", "0.1"], "--uc-ils requires --uc-hook"),
        (["--uc-repair"], "--uc-repair requires --uc-hook"),
        (["--uc-hook", "ed", "--uc-elite", "0.5"], "--uc-elite requires --uc-hook full"),
        (["--uc-hook", "dispatch"], "--uc-hook must be ed or full"),
        (
            ["--uc-hook", "full", "--no-time-limit", "--max-iterations", "100"],
            "cannot run under --no-time-limit",
        ),
        (["--uc-hook", "full", "--uc-ils-stall", "0"], "--uc-ils-stall must be in [1,"),
        (["--uc-hook", "full", "--uc-ls-jcap", "-1"], "--uc-ls-jcap must be in [0,"),
        (["--uc-hook", "full", "--uc-ils-local", "4294967296"], "--uc-ils-local must be in"),
        (["--uc-hook", "full", "--uc-ils", "inf"], "--uc-ils must be in [0, 1]"),
        (["--uc-hook", "full", "--uc-elite", "1.5"], "--uc-elite must be in [0, 1]"),
        (["--uc-hook", "full", "--uc-elite", "nan"], "--uc-elite must be in [0, 1]"),
        # The message's 15-digit DBL_MAX overflows a read-back.
        (["--uc-hook", "full", "--uc-elite", "1.7976931348623157e308"], "--uc-elite must be in"),
    ],
)
def test_uc_chped_refuses_a_bad_uc_flag(args: list[str], message: str, tmp_path: Path) -> None:
    result = _run(tmp_path, *args)  # refused while parsing, before any solve
    assert result.returncode == 2, result.stdout
    assert message in result.stderr, result.stderr
    assert not (tmp_path / "o.csv").exists()


@pytest.mark.parametrize("other", ["--out", "--trace"])
def test_uc_chped_refuses_a_solution_file_that_names_another_output(
    other: str, tmp_path: Path
) -> None:
    target = tmp_path / "o.csv" if other == "--out" else tmp_path / "t.csv"
    args = ["--time-limit", "1", "--demand-equality", "--solution-out", str(target)]
    if other == "--trace":
        args += ["--trace", str(target)]
    result = _run(tmp_path, *args)
    assert result.returncode == 2, result.stdout
    assert "--solution-out" in result.stderr and "names another output" in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        # The checker holds `=` demand: a `>=` row it would fail is refused.
        ([], "requires --demand-equality"),
        (["--demand-equality", "--feas-tol", "1e-5"], "requires --feas-tol <= 1e-6"),
        # Inside the instance directory: the published table, trace or an instance.
        (["--demand-equality", "--inst-file", "comparison.csv"], "inside the instance directory"),
        (["--demand-equality", "--inst-file", "ucp13.jsonl"], "inside the instance directory"),
        (["--demand-equality", "--inst-file", "traces/x.csv"], "inside the instance directory"),
        # Outside it by path, but a hard link to the published table.
        (["--demand-equality", "--link-file", "comparison.csv"], "inside the instance directory"),
    ],
)
def test_uc_chped_refuses_a_solution_file_its_checker_cannot_stand_behind(
    args: list[str], message: str, tmp_path: Path
) -> None:
    target = str(tmp_path / "s.jsonl")
    inst_dir = INSTANCES
    flag = next((f for f in ("--inst-file", "--link-file") if f in args), None)
    if flag is not None:
        # A scratch copy: a regression here would truncate the file it names.
        inst_dir = tmp_path / "uc-chped"
        inst_dir.mkdir()
        for f in INSTANCES.iterdir():
            if f.is_file():
                (inst_dir / f.name).write_bytes(f.read_bytes())
        i = args.index(flag)
        inside = inst_dir / args[i + 1]
        inside.parent.mkdir(exist_ok=True)
        target = str(inside)
        if flag == "--link-file":
            target = str(tmp_path / "s.jsonl")
            Path(target).hardlink_to(inside)
        args = args[:i]
    before = Path(target).read_bytes() if Path(target).exists() else None
    result = _run(tmp_path, "--time-limit", "1", "--solution-out", target, *args, inst_dir=inst_dir)
    assert result.returncode == 2, result.stdout
    assert message in result.stderr, result.stderr
    if before is not None:
        assert Path(target).read_bytes() == before


def test_a_failed_solution_open_leaves_the_trace_as_found(tmp_path: Path) -> None:
    trace = tmp_path / "t.csv"
    trace.write_text("previous run\n")
    result = _run(
        tmp_path,
        *["--time-limit", "1", "--demand-equality", "--trace", str(trace)],
        *["--solution-out", str(tmp_path / "missing" / "s.jsonl")],
    )
    assert result.returncode == 2, result.stdout
    assert trace.read_text() == "previous run\n"
