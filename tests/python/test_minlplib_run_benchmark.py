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

from benchmarks.minlplib.run_benchmark import (
    CLAIM_EXCLUDED,
    REPO_ROOT,
    RUNNER_TARGET,
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
    "wall_seconds,feasible,note,commit_sha,max_violation,n_int_vars"
)
ROW = "nvs01,1,1,1,0,0,60,true,feasible,abc1234,0,3"
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


def fake_runner(returncode: int = 0, *, write_row: bool = True) -> Callable[..., FakeCompleted]:
    """Stand in for `subprocess.run(cbls_minlplib ...)` without solving anything."""

    def run(cmd: Sequence[str], **kwargs: object) -> FakeCompleted:
        text = HEADER + "\n"
        if write_row:
            name = cmd[cmd.index("--instance") + 1]
            sha = cmd[cmd.index("--commit") + 1]
            text += f"{name},1,1,1,0,0,60,true,feasible,{sha},0,0\n"
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
    (stage / "a.csv").write_text(HEADER + "\na,1,1,1,0,0,60,true,feasible,abc1234,0,0\n")
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
    (stage / "a.csv").write_text(HEADER + "\na,1,1,1,0,0,60,true,feasible,old0000,0,0\n")
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
    """The runner exits 0 on a read/build error, having written a header only."""
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
    """Every HEADER column name appears in `minlplib.cpp`.

    A substring search per name, so it catches a *renamed* or deleted column but
    not a reordered or inserted one — several of these names also occur in that
    file's prose. The exact pin is the next test, which compares the header
    against the committed table field for field.
    """
    source = (REPO_ROOT / "benchmarks" / "minlplib" / "minlplib.cpp").read_text()
    assert all(column in source for column in HEADER.split(","))
    assert 'trace << "' + TRACE_HEADER + '\\n"' in source


def test_the_committed_table_uses_the_columns_the_driver_assembles() -> None:
    """The assembled table must slot into the published one column-for-column."""
    published = REPO_ROOT / "benchmarks" / "instances" / "minlplib" / "comparison.csv"
    with published.open(newline="") as fh:
        assert next(csv.reader(fh)) == HEADER.split(",")
