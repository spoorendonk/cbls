"""Unit tests for the MINLPLib re-run driver.

Every test here is offline: the driver's job is to refuse bad invocations, to
resume only from rows it may trust, and to assemble staged rows into the
published tables — all checkable without solving anything. `run_roster` is
exercised against a fake runner, so even its loop costs no solve. The one thing
not covered is the search itself, which is a 50-minute campaign.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib.ablation_report import COMPLETED_SEARCH_NOTES
from benchmarks.minlplib.run_benchmark import (
    CLAIM_EXCLUDED,
    REPO_ROOT,
    RUNNER_EXIT_ERRORED,
    RUNNER_TARGET,
    STAGEABLE_NOTES,
    STAMP_NAME,
    assemble,
    cmake_build_type,
    describe_plan,
    merge_command,
    preflight,
    resolve_paths,
    roster_from_bounds,
    run_roster,
    runner_command,
    staged_complete,
    staged_row_complete,
    staging_stamp_conflict,
    summarize,
    usage_error,
    verdict_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

HEADER = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,"
    "wall_seconds,feasible,note,commit_sha,max_violation,n_int_vars,lns_repairs,"
    "lns_repairs_accepted,first_feasible_objective,time_to_first_feasible,"
    "search_config"
)
DEFAULT_ARM = (
    "float_hook=on;lns=on;lns_interval=3;compound_moves=off;novelty_prob=0.5;"
    "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on"
)
ROW = "nvs01,1,1,1,0,0,60,true,feasible,abc1234,0,3,7,2,9,0.25," + DEFAULT_ARM
TRACE_HEADER = "instance,time_seconds,objective,new_best"


def make_args(tmp_path: Path, **overrides: object) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "inst_dir": tmp_path / "inst",
        "build_dir": tmp_path / "build",
        "time_limit": 60.0,
        "seed": 1,
        "build_jobs": 4,
        "instances": [],
        "out": None,
        "trace_out": None,
        "staging_dir": None,
        "trace": True,
        "merge": True,
        "resume": True,
        "build": True,
        "dry_run": False,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def make_inst_dir(tmp_path: Path, names: list[str], *, scip: bool = True) -> Path:
    inst = tmp_path / "inst"
    inst.mkdir(exist_ok=True)
    rows = ["instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks"]
    rows += [f"{name},bilinear,3,2,min,1.0,1.0,0" for name in names]
    (inst / "bounds.csv").write_text("\n".join(rows) + "\n")
    for name in names:
        (inst / f"{name}.nl").write_text("stub\n")
    if scip:
        (inst / "scip_baseline.csv").write_text("instance,scip_version\n")
    return inst


def make_build_dir(
    tmp_path: Path,
    build_type: str = "Release",
    home: str | None = None,
    extra: str = "",
) -> Path:
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    (build / "CMakeCache.txt").write_text(
        "// a comment line that is not a cache entry\n"
        "CMAKE_PROJECT_NAME:STATIC=cbls\n"
        f"CMAKE_BUILD_TYPE:STRING={build_type}\n"
        f"CMAKE_HOME_DIRECTORY:INTERNAL={home or REPO_ROOT}\n" + extra
    )
    return build


# --- roster and cache parsing ------------------------------------------------


def test_roster_comes_from_bounds_csv_in_file_order(tmp_path: Path) -> None:
    inst = make_inst_dir(tmp_path, ["process", "st_e36", "elec25"])
    assert roster_from_bounds(inst / "bounds.csv") == ["process", "st_e36", "elec25"]


def test_an_absent_bounds_csv_is_an_empty_roster_not_a_traceback(tmp_path: Path) -> None:
    """Preflight turns the empty roster into the refusal that names download.py."""
    assert roster_from_bounds(tmp_path / "nope" / "bounds.csv") == []


def test_cmake_build_type_reads_the_cache(tmp_path: Path) -> None:
    assert cmake_build_type(make_build_dir(tmp_path, "Release")) == "Release"


def test_cmake_build_type_of_an_unconfigured_dir_is_none(tmp_path: Path) -> None:
    assert cmake_build_type(tmp_path / "nope") is None


def test_paths_default_to_the_published_tables_and_a_build_staging_dir(tmp_path: Path) -> None:
    paths = resolve_paths(make_args(tmp_path))
    assert paths.out == tmp_path / "inst" / "comparison.csv"
    assert paths.out == paths.published_out
    assert paths.trace_out == tmp_path / "inst" / "anytime_trace.csv"
    assert paths.stage == tmp_path / "build" / "minlplib-rerun"


# --- preflight ---------------------------------------------------------------


def test_preflight_accepts_a_clean_release_checkout(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path)
    assert preflight(make_args(tmp_path), "abc1234", ["process"]) == []


def test_preflight_refuses_a_dirty_working_tree(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path)
    problems = preflight(make_args(tmp_path), "abc1234-dirty", ["process"])
    assert any("dirty" in p for p in problems)


def test_preflight_refuses_a_non_release_build(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path, "Debug")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("not Release" in p for p in problems)


def test_preflight_refuses_an_unconfigured_build_dir(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("CMakeCache.txt not found" in p for p in problems)


def test_preflight_refuses_a_build_dir_from_another_checkout(tmp_path: Path) -> None:
    """The SHA is read from this checkout; the binary must come from it too."""
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path, home=str(tmp_path / "elsewhere"))
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("was configured from" in p for p in problems)


def test_preflight_refuses_a_sanitizer_build_dir(tmp_path: Path) -> None:
    """CBLS_SANITIZE is a sticky cache entry that leaves CMAKE_BUILD_TYPE=Release.

    So the Release check alone would pass a build dir configured once with a
    sanitizer, and publish wall-clock-budgeted rows measured several-fold slow.
    """
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path, extra="CBLS_SANITIZE:STRING=address,undefined\n")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("CBLS_SANITIZE=address,undefined" in p for p in problems)


def test_preflight_refuses_a_frame_pointer_build_dir(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path, extra="CBLS_PROFILE:BOOL=ON\n")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("CBLS_PROFILE=ON" in p for p in problems)


def test_preflight_accepts_the_options_turned_off(tmp_path: Path) -> None:
    """An ordinary gated build records both options empty/OFF; that must pass."""
    make_inst_dir(tmp_path, ["process"])
    build = make_build_dir(tmp_path, extra="CBLS_SANITIZE:STRING=\nCBLS_PROFILE:BOOL=OFF\n")
    (build / RUNNER_TARGET).write_text("")
    assert preflight(make_args(tmp_path), "abc1234", ["process"]) == []


def test_preflight_refuses_no_build_when_the_runner_is_absent(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path)
    problems = preflight(make_args(tmp_path, build=False), "abc1234", ["process"])
    assert any("--no-build" in p for p in problems)


def test_preflight_accepts_no_build_when_the_runner_exists(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"])
    build = make_build_dir(tmp_path)
    (build / RUNNER_TARGET).write_text("")
    assert preflight(make_args(tmp_path, build=False), "abc1234", ["process"]) == []


def test_preflight_refuses_an_empty_roster_and_names_download_py(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, [])
    make_build_dir(tmp_path)
    problems = preflight(make_args(tmp_path), "abc1234", [])
    assert any("download.py" in p for p in problems)


def test_preflight_refuses_a_roster_with_a_missing_instance_file(tmp_path: Path) -> None:
    inst = make_inst_dir(tmp_path, ["process", "st_e36"])
    make_build_dir(tmp_path)
    (inst / "st_e36.nl").unlink()
    problems = preflight(make_args(tmp_path), "abc1234", ["process", "st_e36"])
    assert any("no .nl file" in p and "st_e36" in p for p in problems)


def test_preflight_refuses_a_merge_with_no_scip_baseline(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"], scip=False)
    make_build_dir(tmp_path)
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("scip_baseline.csv" in p for p in problems)


def test_preflight_without_the_merge_does_not_need_a_scip_baseline(tmp_path: Path) -> None:
    make_inst_dir(tmp_path, ["process"], scip=False)
    make_build_dir(tmp_path)
    assert preflight(make_args(tmp_path, merge=False), "abc1234", ["process"]) == []


# --- argument guards ---------------------------------------------------------


def test_a_subset_run_must_name_a_scratch_output(tmp_path: Path) -> None:
    args = make_args(tmp_path, instances=["nvs01"])
    message = usage_error(args, tmp_path / "comparison.csv")
    assert message is not None and "--out" in message


def test_a_subset_run_may_not_aim_out_at_the_published_table(tmp_path: Path) -> None:
    """An explicit --out is not enough if it resolves to the fifty-row table."""
    published = tmp_path / "comparison.csv"
    args = make_args(
        tmp_path,
        instances=["nvs01"],
        out=tmp_path / "sub" / ".." / "comparison.csv",
        trace_out=tmp_path / "t.csv",
        staging_dir=tmp_path / "stage",
    )
    (tmp_path / "sub").mkdir()
    message = usage_error(args, published)
    assert message is not None and "would truncate" in message


def test_a_traced_subset_run_must_name_a_scratch_trace(tmp_path: Path) -> None:
    args = make_args(tmp_path, instances=["nvs01"], out=tmp_path / "scratch.csv")
    message = usage_error(args, tmp_path / "comparison.csv")
    assert message is not None and "--trace-out" in message


def test_a_subset_run_must_name_a_scratch_staging_dir(tmp_path: Path) -> None:
    """Otherwise its short-budget rows sit where the next full run resumes from."""
    args = make_args(
        tmp_path,
        instances=["nvs01"],
        out=tmp_path / "scratch.csv",
        trace_out=tmp_path / "scratch_trace.csv",
    )
    message = usage_error(args, tmp_path / "comparison.csv")
    assert message is not None and "--staging-dir" in message


def test_a_fully_redirected_subset_run_is_accepted(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        instances=["nvs01"],
        out=tmp_path / "scratch.csv",
        trace_out=tmp_path / "scratch_trace.csv",
        staging_dir=tmp_path / "stage",
    )
    assert usage_error(args, tmp_path / "comparison.csv") is None


def test_no_trace_is_rejected_when_it_would_publish_a_stale_trace(tmp_path: Path) -> None:
    """A whole-roster --no-trace publishes comparison.csv at this engine while
    anytime_trace.csv keeps the previous one, and nothing in either file says so."""
    published = tmp_path / "comparison.csv"
    message = usage_error(make_args(tmp_path, trace=False), published)
    assert message is not None and "--no-trace" in message


def test_no_trace_is_allowed_when_the_table_goes_to_scratch(tmp_path: Path) -> None:
    args = make_args(tmp_path, trace=False, out=tmp_path / "scratch.csv")
    assert usage_error(args, tmp_path / "comparison.csv") is None


def test_a_scratch_table_with_a_defaulted_trace_is_rejected(tmp_path: Path) -> None:
    """`--out` moved off the published table does NOT move the trace with it.

    `resolve_paths` defaults `trace_out` to the published `anytime_trace.csv`
    regardless of `--out`, and `publish` assembles into it unconditionally -- so
    a scratch run at another seed or budget would replace the published anytime
    profile at exit 0 while reporting that it wrote a scratch table. The runner's
    own guard cannot catch it: every instance is staged, so the published path
    never reaches the runner at all.

    This is the shape the #149 campaign uses (whole roster, per-seed scratch
    outputs), which is how it was found.
    """
    args = make_args(tmp_path, out=tmp_path / "scratch.csv")
    message = usage_error(args, tmp_path / "comparison.csv")
    assert message is not None
    assert "--trace-out" in message and "anytime_trace.csv" in message


def test_a_scratch_table_with_an_explicit_trace_out_is_accepted(tmp_path: Path) -> None:
    args = make_args(
        tmp_path, out=tmp_path / "scratch.csv", trace_out=tmp_path / "scratch.trace.csv"
    )
    assert usage_error(args, tmp_path / "comparison.csv") is None


def test_a_whole_roster_run_with_tracing_on_is_accepted(tmp_path: Path) -> None:
    assert usage_error(make_args(tmp_path), tmp_path / "comparison.csv") is None


def test_a_nonpositive_budget_is_rejected(tmp_path: Path) -> None:
    message = usage_error(make_args(tmp_path, time_limit=0.0), tmp_path / "comparison.csv")
    assert message is not None and "--time-limit" in message


def test_a_nonpositive_build_job_count_is_rejected(tmp_path: Path) -> None:
    message = usage_error(make_args(tmp_path, build_jobs=0), tmp_path / "comparison.csv")
    assert message is not None and "--build-jobs" in message


# --- staging: what may be resumed from ---------------------------------------


def test_a_header_only_staging_file_is_not_a_completed_instance(tmp_path: Path) -> None:
    """The runner writes its header before it solves, so existence proves nothing."""
    path = tmp_path / "nvs01.csv"
    path.write_text(HEADER + "\n")
    assert not staged_row_complete(path, "abc1234")


def test_a_staging_file_with_a_row_is_complete(tmp_path: Path) -> None:
    path = tmp_path / "nvs01.csv"
    path.write_text(HEADER + "\n" + ROW + "\n")
    assert staged_row_complete(path, "abc1234")


def test_an_absent_staging_file_is_not_complete(tmp_path: Path) -> None:
    assert not staged_row_complete(tmp_path / "absent.csv", "abc1234")


def test_a_row_staged_by_another_commit_is_re_solved(tmp_path: Path) -> None:
    """Resuming onto it would publish a table whose rows name two engines."""
    path = tmp_path / "nvs01.csv"
    path.write_text(HEADER + "\n" + ROW + "\n")
    assert not staged_row_complete(path, "def5678")


def test_a_torn_final_line_is_not_a_completed_instance(tmp_path: Path) -> None:
    """A job killed mid-write leaves a short line that still reads as a line."""
    path = tmp_path / "nvs01.csv"
    path.write_text(HEADER + "\nnvs01,1,1")
    assert not staged_row_complete(path, "abc1234")


def test_a_staged_csv_without_its_trace_is_not_complete_when_tracing(tmp_path: Path) -> None:
    """A --no-trace run must not let a later traced run skip straight to assembly."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "nvs01.csv").write_text(HEADER + "\n" + ROW + "\n")
    assert not staged_complete(make_args(tmp_path), "abc1234", "nvs01", stage)
    assert staged_complete(make_args(tmp_path, trace=False), "abc1234", "nvs01", stage)


def test_a_fresh_staging_dir_is_stamped_with_the_configuration(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    assert staging_stamp_conflict(stage, make_args(tmp_path), "abc1234") is None
    assert "commit=abc1234" in (stage / STAMP_NAME).read_text()


def test_resuming_a_staging_dir_from_another_budget_is_refused(tmp_path: Path) -> None:
    """Only wall_seconds would betray a 5s smoke run resumed into a 60s publish."""
    stage = tmp_path / "stage"
    stage.mkdir()
    staging_stamp_conflict(stage, make_args(tmp_path, time_limit=5.0), "abc1234")
    conflict = staging_stamp_conflict(stage, make_args(tmp_path), "abc1234")
    assert conflict is not None and "--no-resume" in conflict


def test_resuming_a_staging_dir_from_another_commit_is_refused(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    staging_stamp_conflict(stage, make_args(tmp_path), "abc1234")
    assert staging_stamp_conflict(stage, make_args(tmp_path), "def5678") is not None


def test_no_resume_restamps_instead_of_refusing(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    staging_stamp_conflict(stage, make_args(tmp_path, time_limit=5.0), "abc1234")
    assert staging_stamp_conflict(stage, make_args(tmp_path, resume=False), "abc1234") is None
    assert "time-limit=60" in (stage / STAMP_NAME).read_text()


# --- command construction ----------------------------------------------------


def test_the_runner_is_asked_for_one_instance_and_a_staged_output(tmp_path: Path) -> None:
    cmd = runner_command(make_args(tmp_path), "abc1234", "nvs01", tmp_path / "stage")
    assert cmd[0].endswith(RUNNER_TARGET)
    assert cmd[1] == str(tmp_path / "inst")
    assert cmd[cmd.index("--instance") + 1] == "nvs01"
    assert cmd[cmd.index("--commit") + 1] == "abc1234"
    assert cmd[cmd.index("--time-limit") + 1] == "60"
    assert cmd[cmd.index("--seed") + 1] == "1"
    assert cmd[cmd.index("--out") + 1] == str(tmp_path / "stage" / "nvs01.csv")
    assert cmd[cmd.index("--trace") + 1] == str(tmp_path / "stage" / "nvs01.trace.csv")


def test_the_runner_gets_no_trace_flag_when_tracing_is_off(tmp_path: Path) -> None:
    cmd = runner_command(make_args(tmp_path, trace=False), "abc1234", "nvs01", tmp_path)
    assert "--trace" not in cmd


def test_the_merge_step_solves_nothing(tmp_path: Path) -> None:
    cmd = merge_command(tmp_path / "inst")
    assert "--merge-only" in cmd
    assert cmd[cmd.index("--inst-dir") + 1] == str(tmp_path / "inst")


# --- the dry run -------------------------------------------------------------


def test_a_dry_run_over_an_empty_roster_reports_rather_than_crashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_build_dir(tmp_path)
    args = make_args(tmp_path)
    rc = describe_plan(args, "abc1234", [], resolve_paths(args), ["bounds.csv is missing"])
    assert rc == 2
    assert "WOULD REFUSE" in capsys.readouterr().out


def test_a_clean_dry_run_prints_the_solve_command_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = make_args(tmp_path)
    rc = describe_plan(args, "abc1234", ["nvs01"], resolve_paths(args), [])
    assert rc == 0
    out = capsys.readouterr().out
    assert "--instance nvs01" in out
    assert "--merge-only" in out


# --- run_roster --------------------------------------------------------------


class FakeCompleted:
    """The shape of `subprocess.CompletedProcess` that `run_roster` reads."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.stdout = "runner tally\n"
        self.stderr = ""


def path_after(cmd: Sequence[str], flag: str) -> Path:
    return Path(cmd[cmd.index(flag) + 1])


def fake_runner(
    returncode: int = 0,
    *,
    write_row: bool = True,
    note: str = "feasible",
    feasible: str = "true",
) -> Callable[..., FakeCompleted]:
    """Stand in for `subprocess.run(cbls_minlplib ...)` without solving anything.

    `note`/`feasible` are for the tests that pair a row with an exit status --
    the runner writes a row for a thrown instance too, and what the driver does
    next depends on what that row says.
    """

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        text = HEADER + "\n"
        if write_row:
            name = cmd[cmd.index("--instance") + 1]
            sha = cmd[cmd.index("--commit") + 1]
            text += f"{name},1,1,1,0,0,60,{feasible},{note},{sha},0,0,0,0,1,0.5,{DEFAULT_ARM}\n"
        path_after(cmd, "--out").write_text(text)
        if "--trace" in cmd:
            path_after(cmd, "--trace").write_text(TRACE_HEADER + "\n")
        return FakeCompleted(returncode)

    return run


def test_run_roster_solves_each_instance_and_keeps_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(subprocess, "run", fake_runner())
    run_roster(make_args(tmp_path), "abc1234", ["a", "b"], stage)
    assert staged_complete(make_args(tmp_path), "abc1234", "a", stage)
    assert (stage / "b.log").read_text() == "runner tally\n"


def test_run_roster_skips_an_instance_already_staged_at_this_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.csv").write_text(
        HEADER + f"\na,1,1,1,0,0,60,true,feasible,abc1234,0,0,0,0,1,0.5,{DEFAULT_ARM}\n"
    )
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    monkeypatch.setattr(subprocess, "run", fake_runner())
    run_roster(make_args(tmp_path), "abc1234", ["a", "b"], stage)
    assert "a: staged already, skipping" in capsys.readouterr().out
    assert not (stage / "a.log").exists()


def test_run_roster_re_solves_an_instance_staged_at_another_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.csv").write_text(
        HEADER + f"\na,1,1,1,0,0,60,true,feasible,old0000,0,0,0,0,1,0.5,{DEFAULT_ARM}\n"
    )
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    monkeypatch.setattr(subprocess, "run", fake_runner())
    run_roster(make_args(tmp_path), "abc1234", ["a"], stage)
    assert "abc1234" in (stage / "a.csv").read_text()


def test_run_roster_raises_and_keeps_the_log_when_the_runner_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(subprocess, "run", fake_runner(returncode=2))
    with pytest.raises(RuntimeError, match="Re-running resumes"):
        run_roster(make_args(tmp_path), "abc1234", ["a"], stage)
    assert (stage / "a.log").exists()


def test_run_roster_raises_when_the_runner_exits_zero_without_a_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A runner that wrote its header and died leaves no row to record.

    Not a read or build error: those write a row and, since #153, exit nonzero.
    This is the exit-0-with-nothing-to-stage case, which the row check catches.
    """
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(subprocess, "run", fake_runner(write_row=False))
    with pytest.raises(RuntimeError, match="exit 0"):
        run_roster(make_args(tmp_path), "abc1234", ["a"], stage)


# --- assembly ----------------------------------------------------------------


def _stage_two(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "b.csv").write_text(HEADER + "\nb,2,2,2,0,0,60,true,feasible,abc1234,0,0\n")
    (stage / "a.csv").write_text(HEADER + "\na,1,1,1,0,0,60,true,matches-bks,abc1234,0,0\n")
    return stage


def test_assemble_emits_one_header_and_roster_order(tmp_path: Path) -> None:
    stage = _stage_two(tmp_path)
    out = tmp_path / "comparison.csv"
    assemble(stage, ["a", "b"], out, ".csv")
    lines = out.read_text().splitlines()
    assert lines[0] == HEADER
    assert [line.split(",")[0] for line in lines[1:]] == ["a", "b"]


def test_assemble_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    stage = _stage_two(tmp_path)
    out = tmp_path / "comparison.csv"
    assemble(stage, ["a", "b"], out, ".csv")
    assert not (tmp_path / "comparison.csv.partial").exists()


def test_assemble_refuses_a_staging_file_with_a_different_header(tmp_path: Path) -> None:
    stage = _stage_two(tmp_path)
    (stage / "b.csv").write_text("instance,objective\nb,2\n")
    with pytest.raises(RuntimeError, match="header differs"):
        assemble(stage, ["a", "b"], tmp_path / "comparison.csv", ".csv")


def test_assemble_refuses_an_empty_staging_file(tmp_path: Path) -> None:
    stage = _stage_two(tmp_path)
    (stage / "b.csv").write_text("")
    with pytest.raises(RuntimeError, match="is empty"):
        assemble(stage, ["a", "b"], tmp_path / "comparison.csv", ".csv")


def test_assemble_keeps_a_header_only_trace_file(tmp_path: Path) -> None:
    """An instance that never reaches feasibility contributes no trace rows."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\na,1.0,5,true\n")
    (stage / "b.trace.csv").write_text(TRACE_HEADER + "\n")
    out = tmp_path / "anytime_trace.csv"
    assemble(stage, ["a", "b"], out, ".trace.csv")
    assert out.read_text() == TRACE_HEADER + "\na,1.0,5,true\n"


def test_assemble_does_not_replace_the_output_when_a_row_is_missing(tmp_path: Path) -> None:
    """A failed assembly must leave the previously published table in place."""
    stage = _stage_two(tmp_path)
    out = tmp_path / "comparison.csv"
    out.write_text("previous table\n")
    with pytest.raises(FileNotFoundError):
        assemble(stage, ["a", "b", "c"], out, ".csv")
    assert out.read_text() == "previous table\n"


# --- the derived summary -----------------------------------------------------


def test_a_verdict_drops_the_analysis_note_the_runner_glued_on() -> None:
    """Otherwise one annotated row becomes its own histogram bucket."""
    assert verdict_of("feasible | bug: Thomson problem") == "feasible"
    assert verdict_of("infeasible(residual=1; 25 viol) | hard") == "infeasible"
    assert verdict_of("matches-bks; int-mismatch") == "matches-bks"


def test_summary_holds_elec_out_of_the_counted_rows(tmp_path: Path) -> None:
    out = tmp_path / "comparison.csv"
    rows = [
        HEADER,
        "a,1,1,1,0,0,60,true,matches-bks,abc1234,0,0",
        "b,1,1,1,0,0,60,true,feasible | bug: x,abc1234,0,0",
        f"{CLAIM_EXCLUDED[0]},NaN,1,1,NaN,NaN,60,false,infeasible(residual=1),abc1234,1,0",
    ]
    out.write_text("\n".join(rows) + "\n")
    text = summarize(out)
    assert "rows written:         3" in text
    assert "counted (excl. elec): 2" in text
    assert "feasible:             2" in text
    assert "  feasible            1" in text
    assert f"excluded from claims: {CLAIM_EXCLUDED[0]} -> infeasible" in text


def test_the_published_header_still_matches_what_the_runner_writes() -> None:
    """Every HEADER column name appears in `minlplib.cpp`, delimited as a cell.

    A substring search per name, so it catches a *renamed* or deleted column but
    not a reordered or inserted one — several of these names also occur in that
    file's prose. The exact pin is the next test, which compares the header
    against the committed table field for field.

    The delimiter is load-bearing and not decoration: a bare `column in source`
    is satisfied for `lns_repairs` by the presence of `lns_repairs_accepted`, so
    deleting the shorter column would leave this green. Requiring the trailing
    comma (or, for the last column, the literal's closing newline) separates the
    two. It works only because the header literal is split BETWEEN cells in that
    file, never mid-name — which is itself a thing this assertion pins.
    """
    source = (REPO_ROOT / "benchmarks" / "minlplib" / "minlplib.cpp").read_text()
    columns = HEADER.split(",")
    assert all(f"{column}," in source for column in columns[:-1])
    assert f'{columns[-1]}\\n"' in source
    assert 'trace << "' + TRACE_HEADER + '\\n"' in source


def test_the_committed_table_uses_the_columns_the_driver_assembles() -> None:
    """The assembled table must slot into the published one column-for-column.

    Five exceptions, all dated rather than permanent: `search_config` (#136),
    `lns_repairs` (#143), `lns_repairs_accepted` (#150),
    `first_feasible_objective` and `time_to_first_feasible` (#149) were added to
    the runner after this table was measured, so the committed rows do not carry
    them. They cannot be given them retroactively either -- nobody recorded what
    configuration those rows were produced under, how often LNS repaired during
    them, how many of those repairs were kept, or where each run first reached
    feasibility, which is the whole reason the columns now exist. The next full
    regeneration writes all five, and this assertion goes back to a plain
    equality then.

    The disagreement is also how a reader tells an old table from a new one: a
    `comparison.csv` whose header stops at `n_int_vars` predates #136, and one
    that carries the two first-feasible columns is from #149 or later. The
    column count IS the provenance, which is why this test states the trailing
    set explicitly rather than allowing any suffix.
    """
    published = REPO_ROOT / "benchmarks" / "instances" / "minlplib" / "comparison.csv"
    with published.open(newline="") as fh:
        columns = next(csv.reader(fh))
    assert HEADER.split(",") == [
        *columns,
        "lns_repairs",
        "lns_repairs_accepted",
        "first_feasible_objective",
        "time_to_first_feasible",
        "search_config",
    ]


# --- the runner's exit status (issue #153) ------------------------------------
#
# These run the REAL binary. The exit status is a property of the process
# contract every automated consumer depends on -- `run_roster` above raises on
# it and `run_ablation.execute_runs` records a failed run from it -- not a
# property of any function, and the whole defect was that the contract said
# "success" while the tally said otherwise.


def nl_header(nvars: int, ncons: int, nobjs: int) -> str:
    """A minimal but valid `g3` NL header for `nvars`/`ncons`/`nobjs`.

    Mirrors the fixture layout in `tests/test_minlplib.cpp`; the counts past the
    first line are cosmetic to the reader.
    """
    return (
        "g3 0 1 0\t# header\n"
        f" {nvars} {ncons} {nobjs} 0 0\t# vars, cons, objs, ranges, eqns\n"
        " 0 0\n 0 0\n 0 0 0\n 0 0 0 1\n 0 0 0 0 0\n 0 0\n 0 0\n 0 0 0 0 0\n"
    )


#: One variable in [0,10], one constraint `x <= 7`, minimise `x`. Solves.
SOLVABLE_NL = nl_header(1, 1, 1) + "b\n0 0 10\nr\n1 7\nO0 0\nn0\nG0 1\n0 1\nC0\nn0\nJ0 1\n0 1\n"

#: Zero variables, one objective. It reads and it builds -- and then `cbls::solve`
#: throws `var id out of range` reaching for a variable that is not there, which
#: is the runner's `solve-error` path with nothing added to production code to
#: provoke it. If the engine ever handles an empty model gracefully this stops
#: being a solve-error and the fixture needs replacing -- loudly, since the
#: tally and the exit status both move.
THROWS_ON_SOLVE_NL = nl_header(0, 0, 1) + "b\nr\nO0 0\nn0\n"

#: An opcode outside the adapter's set. A COVERAGE GAP, not an error: the runner
#: buckets it as skipped(unsupported) and it must not move the exit status.
UNSUPPORTED_NL = nl_header(1, 0, 1) + "b\n0 0 10\nr\nO0 0\no999\nv0\n"

#: Not an NL file at all. `read_nl` throws something that is not an unsupported
#: opcode, so the runner counts it in the same error tally as a thrown solve.
UNREADABLE_NL = "g3 0 1 0\n 1 1 1 0 0\nb\nZZZ not a bound\n"


#: Every verdict `classify_against_bks` can return for a row that solved. The
#: exit-status tests assert membership rather than a value: which of the four a
#: one-second solve earns is a property of the engine, and these tests are about
#: the process contract.
COMPLETED_VERDICTS = frozenset(
    {"feasible", "matches-bks", "better-than-bks", "within-tolerance-of-bks"}
)


def minlplib_binary() -> Path:
    binary = REPO_ROOT / "build" / RUNNER_TARGET
    if not binary.exists():
        pytest.skip(f"{RUNNER_TARGET} not built")
    return binary


def run_the_runner(tmp_path: Path, fixtures: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run the real binary over a roster built from `fixtures`.

    A name mapped to the empty string gets a `bounds.csv` entry and no `.nl`
    file, which is the runner's `not-found` bucket.
    """
    binary = minlplib_binary()
    inst_dir = tmp_path / "instances"
    inst_dir.mkdir(exist_ok=True)
    bounds = ["instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks"]
    for name, text in fixtures.items():
        if text:
            (inst_dir / f"{name}.nl").write_text(text)
        # BKS 0.0: the solvable fixture minimises x over x<=7, x>=0, so a
        # completed search lands on the published bound and the note is stable.
        bounds.append(f"{name},linear,1,1,min,0.0,0.0,0")
    (inst_dir / "bounds.csv").write_text("\n".join(bounds) + "\n")
    return subprocess.run(
        [
            str(binary),
            str(inst_dir),
            "--time-limit",
            "1",
            "--commit",
            "abc1234",
            "--out",
            str(tmp_path / "out.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def notes_of(out: Path) -> dict[str, str]:
    with out.open(newline="") as fh:
        return {row["instance"]: row["note"] for row in csv.DictReader(fh)}


def test_a_solve_that_throws_makes_the_runner_exit_nonzero(tmp_path: Path) -> None:
    """The #153 defect: `run_benchmark` returned 0 whatever its error tally said.

    A thrown solve still writes a well-formed row -- `feasible=false`,
    `note=solve-error`, measured cells NaN -- so the row alone cannot stop a
    consumer that checks `$?`, and every one of them saw success.
    """
    result = run_the_runner(tmp_path, {"ok": SOLVABLE_NL, "boom": THROWS_ON_SOLVE_NL})

    assert result.returncode == RUNNER_EXIT_ERRORED
    assert "ERROR solving" in result.stdout
    assert "exiting 3" in result.stderr
    # ... and the tally agrees with that message. The thrown instance's model
    # WAS closed, so without holding it apart the tally printed "infeasible: 1"
    # two lines above a stderr line saying these are not infeasibility results,
    # and counted the instance twice in the closed-model rate's denominator.
    #
    # Asserted on a `boom`-ONLY roster: with `ok` in it, a healthy instance that
    # misses its one-second budget would print "infeasible: 1" and red a
    # process-contract test for a reason that has nothing to do with the
    # contract. Alone, the two numbers can only move if the arithmetic regresses.
    alone_dir = tmp_path / "alone"
    alone_dir.mkdir()
    alone = run_the_runner(alone_dir, {"boom": THROWS_ON_SOLVE_NL})
    assert alone.returncode == RUNNER_EXIT_ERRORED
    assert "infeasible:           0" in alone.stdout
    assert "closed-model rate:    100% of 1 attempted" in alone.stdout
    # The row is still written: the exit status is an addition to the record,
    # not a replacement for it.
    notes = notes_of(tmp_path / "out.csv")
    assert notes["boom"] == "solve-error"
    # The healthy instance still solved and was still scored. Its exact verdict
    # is the engine's business, not this test's -- pinning `matches-bks` would
    # red a process-contract test on a tie-band change.
    assert notes["ok"] in COMPLETED_VERDICTS


def test_an_unreadable_instance_makes_the_runner_exit_nonzero(tmp_path: Path) -> None:
    """`read-error` and `build-error` feed the same tally as `solve-error`."""
    result = run_the_runner(tmp_path, {"ok": SOLVABLE_NL, "junk": UNREADABLE_NL})

    assert result.returncode == RUNNER_EXIT_ERRORED
    assert notes_of(tmp_path / "out.csv")["junk"] == "read-error"


def test_a_coverage_gap_is_not_an_error_and_the_runner_still_exits_zero(
    tmp_path: Path,
) -> None:
    """An instance skipped as unsupported, and one whose `.nl` is not there, are
    gaps in what this adapter covers -- not failures of the run.

    The runner has always bucketed them apart from `Tally::errored`, and the
    exit status must keep that distinction rather than collapsing every
    non-result into one failure signal.
    """
    result = run_the_runner(tmp_path, {"ok": SOLVABLE_NL, "exotic": UNSUPPORTED_NL, "absent": ""})

    assert result.returncode == 0
    assert "exiting" not in result.stderr
    notes = notes_of(tmp_path / "out.csv")
    assert notes["ok"] in COMPLETED_VERDICTS
    assert notes["exotic"].startswith("unsupported")
    assert notes["absent"] == "not-found"


def test_a_staged_row_from_a_thrown_instance_cannot_stand_in_for_a_solve(
    tmp_path: Path,
) -> None:
    """The abort alone would only delay the bad publish by one invocation.

    The row a thrown instance leaves is COMPLETE by every structural check --
    header, one whole line, the right commit -- and `--resume` is the default.
    So `run_roster` aborts on the first run, and without this the next run would
    skip the instance and `publish` would assemble a row that measured nothing
    into `comparison.csv`.
    """
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    errored = (
        HEADER
        + f"\na,NaN,1,1,NaN,NaN,0,false,solve-error,abc1234,NaN,0,NaN,NaN,NaN,NaN,{DEFAULT_ARM}\n"
    )
    (stage / "a.csv").write_text(errored)

    # Structurally whole -- which is exactly why it has to be named separately.
    assert staged_row_complete(stage / "a.csv", "abc1234")
    assert not staged_complete(make_args(tmp_path), "abc1234", "a", stage)


def test_a_coverage_gap_staged_row_still_stands_in_for_a_solve(tmp_path: Path) -> None:
    """`unsupported` and `not-found` exit 0 and are documented rows, not errors.

    The staging refusal must key on the notes that come with the nonzero exit,
    or a roster with one unsupported instance would abort every publish run.
    """
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    note = "unsupported: NL_UNKNOWN_OPCODE 42"
    row = f"a,NaN,NaN,NaN,NaN,NaN,0,false,{note},abc1234,NaN,NaN,NaN,NaN,NaN,NaN,{DEFAULT_ARM}"
    (stage / "a.csv").write_text(f"{HEADER}\n{row}\n")

    assert staged_complete(make_args(tmp_path), "abc1234", "a", stage)


def test_a_staged_row_whose_note_is_unrecognised_cannot_stand_in_for_a_solve(
    tmp_path: Path,
) -> None:
    """The guard is an allowlist, so a note added later fails safe.

    Naming the three notes a throw writes today would fail open the moment the
    runner grows a fourth `++t.errored` site: that row is structurally whole, so
    the next resume would skip the instance and `publish` would assemble a row
    that measured nothing. `convert-error` stands in for any such future note.
    """
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    row = f"a,NaN,1,1,NaN,NaN,0,false,convert-error,abc1234,NaN,0,NaN,NaN,{DEFAULT_ARM}"
    (stage / "a.csv").write_text(f"{HEADER}\n{row}\n")

    assert staged_row_complete(stage / "a.csv", "abc1234")
    assert not staged_complete(make_args(tmp_path), "abc1234", "a", stage)


def test_a_staged_row_with_no_note_at_all_cannot_stand_in_for_a_solve(
    tmp_path: Path,
) -> None:
    """An empty note is not a measurement either, and the allowlist says so."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    row = f"a,NaN,1,1,NaN,NaN,0,false,,abc1234,NaN,0,NaN,NaN,{DEFAULT_ARM}"
    (stage / "a.csv").write_text(f"{HEADER}\n{row}\n")

    assert not staged_complete(make_args(tmp_path), "abc1234", "a", stage)


def test_the_stageable_notes_are_the_scorer_s_completed_set_plus_the_two_gaps() -> None:
    """Pins the driver's allowlist to the scorer's, which is swept against the runner.

    `run_benchmark.py` is run as a script and has no package context to import
    `ablation_report` through, so the seven completed-search prefixes are spelled
    out in both files. That duplication is only safe while something fails when
    the two drift, and this is that something: `ablation_report`'s own sweep test
    keeps `COMPLETED_SEARCH_NOTES` honest against every note literal in
    `minlplib.cpp`, and this carries that guarantee across to the driver.
    """
    scorer_plus_gaps = (*COMPLETED_SEARCH_NOTES, "unsupported", "not-found")
    assert scorer_plus_gaps == STAGEABLE_NOTES


def test_run_roster_says_what_to_do_when_the_runner_reports_its_error_tally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exit 3 is not the generic failure, and its message must not say it is.

    "Re-running resumes from here" is true of a killed job and false of a
    deterministic throw; the operator needs to be told the difference.
    """
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(
        subprocess,
        "run",
        fake_runner(returncode=RUNNER_EXIT_ERRORED, note="solve-error", feasible="false"),
    )
    with pytest.raises(RuntimeError, match="measures nothing"):
        run_roster(make_args(tmp_path), "abc1234", ["a"], stage)


def test_the_drivers_exit_constant_is_the_runners_own() -> None:
    """`RUNNER_EXIT_ERRORED` is mirrored from C++, and both drivers branch on it.

    Pinned against the literal rather than against a run, so a change to one side
    fails here instead of being discovered by a campaign that stopped recording
    failed runs.
    """
    source = (REPO_ROOT / "benchmarks" / "minlplib" / "minlplib.cpp").read_text()
    assert f"constexpr int kExitErrored = {RUNNER_EXIT_ERRORED};" in source
    # ... and it is what the roster loop actually returns.
    # The tally the status is computed FROM, not just the constant: a status
    # read off `skipped_unsupported` or `verify_failed` would be the same
    # integer for the wrong reason.
    assert "run_exit_status(t.errored)" in source
