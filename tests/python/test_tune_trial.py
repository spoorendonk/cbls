"""`benchmarks/common/tune_trial.py`: the one-run cost an `xk tune` study reads (#262).

Each scorer is pinned on canned runner outputs against the table's own scoring
function, so a tuned configuration is chosen on the measure it is reported on.
One end-to-end run through the real UC runner pins the wrapper's plumbing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.common import tune_trial
from benchmarks.mipfeas.primal_integral import NO_SOLUTION_GAP, primal_integral

REPO = Path(__file__).resolve().parents[2]
UC_CHPED_BINARY = REPO / "build" / "cbls_uc_chped"

# --- cvrp -------------------------------------------------------------------


def _cvrp_result(tmp_path: Path, **record: object) -> Path:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(record))
    return path


def test_cvrp_scores_a_verified_runs_gap(tmp_path: Path) -> None:
    path = _cvrp_result(tmp_path, gap_percent=18.5, verified=True, feasible=True)
    assert tune_trial.score_cvrp(path) == 18.5


def test_cvrp_leaves_an_unverified_run_unscored(tmp_path: Path) -> None:
    path = _cvrp_result(tmp_path, gap_percent=None, verified=False, feasible=True)
    assert tune_trial.score_cvrp(path) is None


def test_cvrp_leaves_a_missing_result_unscored(tmp_path: Path) -> None:
    assert tune_trial.score_cvrp(tmp_path / "result.json") is None


# --- uc ---------------------------------------------------------------------

UC_HEADER = (
    "instance,periods,method,objective,lb,gap_pct,time_s,source,note,ub,time_limit_s,seed,"
    "feasible,verified,max_violation,feas_tol,commit_sha,search_config"
)
UC_CITED = "ucp13,3,Pedroso MIP (1hr),38850,38850,0.00,,Pedroso,cited,38850,,,,,,,,"


def _uc_measured(objective: str, feasible: str, verified: str) -> str:
    return (
        f"ucp13,3,CBLS ViolationLS,{objective},38850,3.4,0.3,cbls,,,0.3,1,"
        f"{feasible},{verified},0,1e-06,unknown,cfg"
    )


def _uc_table(tmp_path: Path, *rows: str) -> Path:
    path = tmp_path / "table.csv"
    path.write_text("\n".join(["# a comment line", UC_HEADER, *rows]) + "\n")
    return path


def test_uc_scores_the_one_measured_verified_row(tmp_path: Path) -> None:
    path = _uc_table(tmp_path, UC_CITED, _uc_measured("40176.2", "true", "true"))
    assert tune_trial.score_uc(path) == 40176.2


@pytest.mark.parametrize(("feasible", "verified"), [("true", "false"), ("false", "")])
def test_uc_leaves_an_unverified_row_unscored(tmp_path: Path, feasible: str, verified: str) -> None:
    path = _uc_table(tmp_path, _uc_measured("", feasible, verified))
    assert tune_trial.score_uc(path) is None


def test_uc_refuses_a_table_with_more_than_one_measured_row(tmp_path: Path) -> None:
    row = _uc_measured("40176.2", "true", "true")
    with pytest.raises(ValueError, match="2 measured rows"):
        tune_trial.score_uc(_uc_table(tmp_path, row, row))


# --- minlplib ---------------------------------------------------------------

MINLPLIB_HEADER = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,wall_seconds,feasible,"
    "note,commit_sha,max_violation,n_int_vars,lns_repairs,lns_repairs_accepted,"
    "first_feasible_objective,time_to_first_feasible,search_config"
)
BOUNDS = (
    "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks\n"
    "alkylation,bilinear,10,11,max,1768.806964,1768.807015,0\n"
)


def _minlplib_files(tmp_path: Path, feasible: str, note: str) -> tuple[Path, Path, Path]:
    table = tmp_path / "table.csv"
    table.write_text(
        MINLPLIB_HEADER + f"\nalkylation,1767.56,1768.81,1768.81,0.07,0.07,0.5,{feasible},{note},"
        "unknown,0,0,0,0,1,0.003,cfg\n"
    )
    # The runner's trace holds the internally minimised value: a maximize row
    # is negated.
    trace = tmp_path / "trace.csv"
    trace.write_text(
        "instance,time_seconds,objective,new_best\n"
        "alkylation,0.1,-1000,1\n"
        "alkylation,0.4,-1767.56,1\n"
    )
    bounds = tmp_path / "bounds.csv"
    bounds.write_text(BOUNDS)
    return table, trace, bounds


def test_minlplib_scores_the_tables_primal_integral_sign_normalised(tmp_path: Path) -> None:
    table, trace, bounds = _minlplib_files(tmp_path, "true", "feasible")
    expected = primal_integral([(0.1, -1000.0), (0.4, -1767.56)], -1768.806964, 0.5)
    assert tune_trial.score_minlplib(table, trace, bounds, 0.5) == pytest.approx(expected)


def test_minlplib_scores_an_infeasible_run_as_never_feasible(tmp_path: Path) -> None:
    table, trace, bounds = _minlplib_files(tmp_path, "false", "infeasible")
    assert tune_trial.score_minlplib(table, trace, bounds, 0.5) == NO_SOLUTION_GAP


def test_minlplib_leaves_a_missing_table_unscored(tmp_path: Path) -> None:
    _, trace, bounds = _minlplib_files(tmp_path, "true", "feasible")
    assert tune_trial.score_minlplib(tmp_path / "none.csv", trace, bounds, 0.5) is None


# --- mipfeas ----------------------------------------------------------------


def _mipfeas_files(tmp_path: Path, status: str, objective: float | None) -> Path:
    engine_dir = tmp_path / tune_trial.MIPFEAS_ENGINE
    engine_dir.mkdir()
    record = {"status": status, "objective": objective, "budget_seconds": 2.0}
    (engine_dir / "inst.json").write_text(json.dumps(record))
    (engine_dir / "inst.trace.csv").write_text("time_seconds,objective\n0.5,110\n1.0,100\n")
    roster = tmp_path / "roster.csv"
    roster.write_text("instance,reference_value,reference_kind\ninst,100.0,opt\n")
    return roster


def test_mipfeas_scores_the_tables_primal_integral(tmp_path: Path) -> None:
    roster = _mipfeas_files(tmp_path, "feasible", 100.0)
    expected = primal_integral([(0.5, 110.0), (1.0, 100.0)], 100.0, 2.0)
    assert tune_trial.score_mipfeas(tmp_path, "inst", roster, 2.0) == pytest.approx(expected)


def test_mipfeas_scores_no_solution_as_never_feasible(tmp_path: Path) -> None:
    roster = _mipfeas_files(tmp_path, "no_solution", None)
    assert tune_trial.score_mipfeas(tmp_path, "inst", roster, 2.0) == NO_SOLUTION_GAP


def test_mipfeas_leaves_an_errored_run_unscored(tmp_path: Path) -> None:
    roster = _mipfeas_files(tmp_path, "read_error", None)
    assert tune_trial.score_mipfeas(tmp_path, "inst", roster, 2.0) is None


def test_mipfeas_refuses_an_instance_the_roster_lacks(tmp_path: Path) -> None:
    roster = _mipfeas_files(tmp_path, "feasible", 100.0)
    with pytest.raises(ValueError, match="no instance 'other'"):
        tune_trial.score_mipfeas(tmp_path, "other", roster, 2.0)


# --- arguments --------------------------------------------------------------

BASE = ["--build-dir", "b", "--out-dir", "o", "--instance", "i", "--seed", "1"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["cvrp", *BASE], "cvrp needs --budget"),
        (["uc", *BASE, "--budget", "1"], "uc needs --periods"),
        (["minlplib", *BASE, "--budget", "1", "--periods", "3"], "--periods is uc's"),
        (["mipfeas", *BASE, "--budget", "0"], "--budget must be a positive"),
    ],
)
def test_refuses_an_incomplete_or_crossed_call(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        tune_trial.parse_args(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("runner", tune_trial.RUNNERS)
def test_every_runner_gets_a_repo_anchored_instance_dir_and_the_flags_last(runner: str) -> None:
    periods = ["--periods", "3"] if runner == "uc" else []
    args = tune_trial.parse_args(
        [runner, *BASE, "--budget", "1", *periods, "--", "--novelty-prob", "0.2"]
    )
    cmd = tune_trial.runner_command(args, Path("o"))
    assert str(tune_trial.DEFAULT_INST_DIRS[runner]) in cmd
    assert tune_trial.DEFAULT_INST_DIRS[runner].is_dir()
    assert cmd[-2:] == ["--novelty-prob", "0.2"]
    assert cmd[cmd.index("--threads") + 1] == "1"


# --- end to end -------------------------------------------------------------


def _uc_call(out_dir: Path, *flags: str) -> list[str]:
    return [
        "uc",
        "--build-dir",
        str(UC_CHPED_BINARY.parent),
        "--out-dir",
        str(out_dir),
        "--instance",
        "ucp13",
        "--periods",
        "3",
        "--seed",
        "1",
        "--threads",
        "2",
        "--budget",
        "0.3",
        "--",
        *flags,
    ]


def test_a_uc_run_leaves_a_cost_the_study_reads(tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    assert tune_trial.main(_uc_call(tmp_path, "--lns-interval", "5")) == 0
    record = json.loads((tmp_path / "cost.json").read_text())
    assert isinstance(record["cost"], float) and record["cost"] > 0
    assert record["returncode"] == 0
    assert "--lns-interval" in record["command"]


def test_a_runner_refusal_is_passed_on_with_no_cost(tmp_path: Path) -> None:
    if not UC_CHPED_BINARY.exists():
        pytest.skip("cbls_uc_chped not built")
    assert tune_trial.main(_uc_call(tmp_path, "--novelty-prob", "2")) == 2
    record = json.loads((tmp_path / "cost.json").read_text())
    assert record["cost"] is None
    assert record["returncode"] == 2


# --- review round 1 ---------------------------------------------------------


@pytest.mark.parametrize("flag", ["--seed", "--time-limit=3"])
def test_refuses_a_study_flag_that_overrides_the_wrappers_own(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runners keep the last value given, so a `{config}` carrying the seed
    or the budget would run one thing and be scored as another."""
    with pytest.raises(SystemExit) as exc:
        tune_trial.parse_args(["cvrp", *BASE, "--budget", "1", "--", flag, "7"])
    assert exc.value.code == 2
    assert "would override this script's own" in capsys.readouterr().err


def test_an_instance_the_roster_lacks_is_refused_before_the_run(tmp_path: Path) -> None:
    roster = tmp_path / "roster.csv"
    roster.write_text("instance,reference_value,reference_kind\ninst,100.0,opt\n")
    out = tmp_path / "run"
    argv = ["mipfeas", "--build-dir", str(tmp_path / "no-build"), "--out-dir", str(out)]
    argv += ["--instance", "other", "--seed", "1", "--budget", "1", "--roster", str(roster)]
    # The build dir holds no runner: reaching the subprocess would raise.
    assert tune_trial.main(argv) == 2
    assert not (out / "cost.json").exists()


def test_an_unreadable_result_still_leaves_a_null_cost(tmp_path: Path) -> None:
    """A run whose output the scorer refuses is a failed trial with its reason
    on disk, not a traceback and no file."""
    build = tmp_path / "build"
    build.mkdir()
    fake = build / "cbls_cvrp"
    fake.write_text(
        '#!/bin/sh\nwhile [ $# -gt 0 ]; do [ "$1" = --out ] && printf "{" > "$2"; shift; done\n'
    )
    fake.chmod(0o755)
    out = tmp_path / "run"
    argv = ["cvrp", "--build-dir", str(build), "--out-dir", str(out), "--instance", "X-n101-k25"]
    assert tune_trial.main([*argv, "--seed", "1", "--budget", "1"]) == 1
    record = json.loads((out / "cost.json").read_text())
    assert record["cost"] is None
    assert record["returncode"] == 1
    assert record["error"]
