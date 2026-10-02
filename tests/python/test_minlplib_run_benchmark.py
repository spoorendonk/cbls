"""Unit tests for the MINLPLib re-run driver.

Every test here is offline: the driver's job is to refuse bad invocations, to
resume only from rows it may trust, and to assemble staged rows into the
published tables — all checkable without solving anything. `run_roster` is
exercised against a fake runner, so even its loop costs no solve. The one thing
not covered is the search itself, which is a 50-minute campaign. The build-dir
refusals it shares with any driver are pinned in `test_benchmark_common.py`.
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.common import jobs
from benchmarks.common.jobs import wallclock_lock
from benchmarks.common.provenance import REPO_ROOT
from benchmarks.minlplib import run_benchmark
from benchmarks.minlplib.campaign_report import (
    AGGREGATION_RULE,
    SEEDS_TABLE_NAME,
    file_sha256,
    seed_block_hashes,
    verdict_of,
)
from benchmarks.minlplib.run_benchmark import (
    ASSEMBLED_TABLE,
    ASSEMBLED_TRACE,
    SEEDS_TABLE_COLUMNS,
    STAMP_NAME,
    assemble,
    common_preflight,
    default_staging_root,
    describe_plan,
    execute,
    merge_command,
    preflight,
    publish_seed_rows,
    resolve_paths,
    roster_from_bounds,
    run_roster,
    runner_command,
    staged_complete,
    staged_row_complete,
    staging_stamp_conflict,
    summarize,
    usage_error,
)
from benchmarks.minlplib.runner import (
    CLAIM_EXCLUDED,
    RUNNER_COLUMNS,
    RUNNER_EXIT_ERRORED,
    RUNNER_TARGET,
    TRACE_COLUMNS,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

HEADER = ",".join(RUNNER_COLUMNS)
DEFAULT_ARM = (
    "float_hook=on;lns=on;lns_interval=3;compound_moves=off;novelty_prob=0.5;"
    "unproductive_iters=300;perturbation_period=100;max_iterations=0;time_limit=on"
)
ROW = "nvs01,1,1,1,0,0,60,true,feasible,abc1234,0,3,7,2,9,0.25," + DEFAULT_ARM
TRACE_HEADER = ",".join(TRACE_COLUMNS)


@pytest.fixture(autouse=True)
def _state_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the staging root and the wall-clock lock into the test's tmp dir, never home."""
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setattr(jobs, "WALLCLOCK_LOCK", tmp_path / "lock" / "wallclock.lock")


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


def make_build_dir(tmp_path: Path, build_type: str = "Release") -> Path:
    build = tmp_path / "build"
    build.mkdir(exist_ok=True)
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_BUILD_TYPE:STRING={build_type}\nCMAKE_HOME_DIRECTORY:INTERNAL={REPO_ROOT}\n"
    )
    return build


# --- roster ------------------------------------------------------------------


def test_roster_comes_from_bounds_csv_in_file_order(tmp_path: Path) -> None:
    inst = make_inst_dir(tmp_path, ["process", "st_e36", "elec25"])
    assert roster_from_bounds(inst / "bounds.csv") == ["process", "st_e36", "elec25"]
    # Preflight turns an empty roster into the refusal that names download.py.
    assert roster_from_bounds(tmp_path / "nope" / "bounds.csv") == []


def test_paths_default_to_the_published_tables_and_a_persistent_staging_dir(
    tmp_path: Path,
) -> None:
    paths = resolve_paths(make_args(tmp_path), "abc1234")
    assert paths.out == tmp_path / "inst" / "comparison.csv"
    assert paths.out == paths.published_out
    assert paths.trace_out == tmp_path / "inst" / "anytime_trace.csv"
    root = tmp_path / "state" / "cbls" / "minlplib-rerun"
    assert paths.stage == root / "abc1234" / "60s" / "seed1"
    assert paths.seeds_out == tmp_path / "inst" / SEEDS_TABLE_NAME


def test_the_default_staging_dir_survives_the_pre_push_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#141: `<build-dir>/minlplib-rerun` died with every `rm -rf build`.

    The default is outside the checkout altogether -- not just outside `build/`,
    since the worktree workflow deletes whole checkouts after a merge.
    """
    stage = resolve_paths(make_args(tmp_path, build_dir=REPO_ROOT / "build"), "abc1234").stage
    assert not stage.is_relative_to(REPO_ROOT)
    monkeypatch.delenv("XDG_STATE_HOME")
    assert default_staging_root() == Path.home() / ".local" / "state" / "cbls" / "minlplib-rerun"
    # The XDG spec says a relative value is invalid and is to be ignored.
    monkeypatch.setenv("XDG_STATE_HOME", "relative/state")
    assert default_staging_root() == Path.home() / ".local" / "state" / "cbls" / "minlplib-rerun"


def test_each_seed_stages_apart_and_only_seed_1_defaults_to_the_published_table(
    tmp_path: Path,
) -> None:
    """Two seeds' rows accumulate in two staging dirs; neither stamp refuses the other."""
    one, two = (
        resolve_paths(make_args(tmp_path), "abc1234"),
        resolve_paths(make_args(tmp_path, seed=2), "abc1234"),
    )
    assert one.stage != two.stage
    assert two.out == two.stage / ASSEMBLED_TABLE
    assert two.trace_out == two.stage / ASSEMBLED_TRACE
    assert two.seeds_out == one.seeds_out == tmp_path / "inst" / SEEDS_TABLE_NAME
    # Scratch and subset runs publish nothing.
    assert resolve_paths(make_args(tmp_path, out=tmp_path / "s.csv"), "abc1234").seeds_out is None
    subset = make_args(tmp_path, instances=["a"], out=tmp_path / "s.csv")
    assert resolve_paths(subset, "abc1234").seeds_out is None


def test_the_staging_stamp_names_the_host(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    staging_stamp_conflict(stage, make_args(tmp_path), "abc1234")
    assert f"host={socket.gethostname()}\n" in (stage / STAMP_NAME).read_text()


# --- preflight ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("sha", "build_type", "runner", "overrides", "roster", "scip", "refusal"),
    [
        ("abc1234", "Release", False, {}, ["process"], True, None),
        ("abc1234-dirty", "Release", False, {}, ["process"], True, ("dirty",)),
        # The build-dir refusals are `build_dir_problems`, pinned where it lives;
        # these two pin that preflight carries them.
        ("abc1234", "Debug", False, {}, ["process"], True, ("not Release",)),
        ("abc1234", None, False, {}, ["process"], True, ("CMakeCache.txt not found",)),
        ("abc1234", "Release", False, {"build": False}, ["process"], True, ("--no-build",)),
        ("abc1234", "Release", True, {"build": False}, ["process"], True, None),
        ("abc1234", "Release", False, {}, [], True, ("download.py",)),
        ("abc1234", "Release", False, {}, ["process", "st_e36"], True, ("no .nl file", "st_e36")),
        ("abc1234", "Release", False, {}, ["process"], False, ("scip_baseline.csv",)),
        ("abc1234", "Release", False, {"merge": False}, ["process"], False, None),
    ],
    ids=[
        "clean-release-checkout",
        "dirty-tree",
        "non-release-build",
        "unconfigured-build",
        "no-build-without-a-runner",
        "no-build-with-a-runner",
        "empty-roster-names-download-py",
        "missing-instance-file",
        "merge-without-scip-baseline",
        "no-merge-needs-no-scip-baseline",
    ],
)
def test_preflight(
    tmp_path: Path,
    sha: str,
    build_type: str | None,
    runner: bool,
    overrides: dict[str, object],
    roster: list[str],
    scip: bool,
    refusal: tuple[str, ...] | None,
) -> None:
    make_inst_dir(tmp_path, ["process"] if roster else [], scip=scip)
    if build_type is not None:
        build = make_build_dir(tmp_path, build_type)
        if runner:
            (build / RUNNER_TARGET).write_text("")
    problems = preflight(make_args(tmp_path, **overrides), sha, roster)
    if refusal is None:
        assert problems == []
    else:
        assert any(all(word in p for word in refusal) for p in problems), problems


# --- argument guards ---------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        # A subset rewrites the same whole files a full run does, and its rows would
        # be resumed by the next full run, so it must name scratch paths throughout.
        ({"instances": ["nvs01"]}, ("--out",)),
        # An explicit --out is not enough if it resolves to the fifty-row table.
        (
            {
                "instances": ["nvs01"],
                "out": "sub/../comparison.csv",
                "trace_out": "t.csv",
                "staging_dir": "stage",
            },
            ("would truncate",),
        ),
        ({"instances": ["nvs01"], "out": "scratch.csv"}, ("--trace-out",)),
        # Otherwise its short-budget rows sit where the next full run resumes from.
        (
            {"instances": ["nvs01"], "out": "scratch.csv", "trace_out": "scratch_trace.csv"},
            ("--staging-dir",),
        ),
        (
            {
                "instances": ["nvs01"],
                "out": "scratch.csv",
                "trace_out": "scratch_trace.csv",
                "staging_dir": "stage",
            },
            None,
        ),
        # A whole-roster --no-trace publishes comparison.csv at this engine while
        # anytime_trace.csv keeps the previous one, and nothing in either file says so.
        ({"trace": False}, ("--no-trace",)),
        ({"trace": False, "out": "scratch.csv"}, None),
        # `--out` moved off the published table does NOT move the trace with it:
        # `resolve_paths` defaults `trace_out` to the published `anytime_trace.csv`
        # and `publish` assembles into it unconditionally, so a scratch run at
        # another seed or budget would replace the published anytime profile at
        # exit 0 while reporting that it wrote a scratch table. The runner's own
        # guard cannot catch it: every instance is staged, so the published path
        # never reaches the runner. The #149 campaign's shape, which found it.
        ({"out": "scratch.csv"}, ("--trace-out", "anytime_trace.csv")),
        # The mirror, which slips past both other guards: `--out` defaulted with
        # `--trace-out` at scratch republishes comparison.csv at this engine beside
        # an anytime_trace.csv from the previous one.
        ({"trace_out": "scratch.trace.csv"}, ("--trace-out", "anytime_trace.csv")),
        ({"out": "scratch.csv", "trace_out": "scratch.trace.csv"}, None),
        ({}, None),
        # #141's seed policy: comparison.csv is the pre-registered seed alone. Any
        # other seed defaults into its own staging dir and the per-seed table...
        ({"seed": 2}, None),
        ({"seed": 2, "out": "scratch.csv"}, None),
        # ...and may not reach either published artifact by naming it.
        ({"seed": 2, "out": "sub/../comparison.csv"}, ("--seed 2", "pre-registered seed 1")),
        (
            {"seed": 2, "out": "s.csv", "trace_out": "inst/anytime_trace.csv"},
            ("--seed 2", "pre-registered seed 1"),
        ),
        (
            {
                "seed": 2,
                "instances": ["nvs01"],
                "out": "comparison.csv",
                "trace_out": "t.csv",
                "staging_dir": "stage",
            },
            ("--seed 2",),
        ),
        # Any other spelling that lands a scratch output on a published file: the
        # trace over the table, a table over the trace, the per-seed table (its
        # record would replace every seed's), and an --out whose run record is
        # the published comparison.run.json.
        (
            {"seed": 2, "out": "s.csv", "trace_out": "inst/comparison.csv"},
            ("--trace-out resolves to the published comparison.csv",),
        ),
        ({"seed": 2, "out": "inst/anytime_trace.csv", "trace": False}, ("anytime_trace.csv",)),
        (
            {"out": "inst/comparison_seeds.csv", "trace_out": "t.csv"},
            ("--out resolves to the published comparison_seeds.csv",),
        ),
        (
            {"seed": 2, "out": "inst/comparison.tsv", "trace": False},
            ("--out's run record resolves to the published comparison.run.json",),
        ),
        ({"out": "inst/comparison_all.csv", "trace_out": "t.csv"}, ("comparison_all.csv",)),
        # The run's own inputs are not scratch space either.
        ({"out": "inst/bounds.csv", "trace_out": "t.csv"}, ("--out resolves", "bounds.csv")),
        (
            {"out": "s.csv", "trace_out": "inst/analysis_notes.csv"},
            ("--trace-out resolves", "analysis_notes.csv"),
        ),
        ({"out": "s.csv", "trace_out": "s.run.json"}, ("--out's run record",)),
        (
            {"out": "s.csv", "trace_out": "inst/scip_baseline.csv"},
            ("--trace-out resolves", "scip_baseline.csv"),
        ),
        # The explicit spelling of the #149 hazard, at the published seed.
        (
            {"out": "scratch.csv", "trace_out": "inst/anytime_trace.csv"},
            ("--trace-out names the published",),
        ),
        # A non-default budget may not publish: a smoke run replaced seed 2's 60s
        # block in comparison_seeds.csv, and seed 1's comparison.csv likewise.
        ({"time_limit": 5.0}, ("--time-limit 5", "comparison.csv")),
        ({"seed": 2, "time_limit": 5.0}, ("--time-limit 5", "comparison_seeds.csv")),
        ({"time_limit": 5.0, "out": "s.csv", "trace_out": "t.csv"}, None),
        ({"seed": 2, "time_limit": 5.0, "out": "s.csv"}, None),
        ({"time_limit": 0.0}, ("--time-limit",)),
        ({"build_jobs": 0}, ("--build-jobs",)),
    ],
    ids=[
        "subset-without-scratch-out",
        "subset-out-resolving-to-the-published-table",
        "traced-subset-without-scratch-trace",
        "subset-without-scratch-staging",
        "fully-redirected-subset",
        "no-trace-publishing-a-stale-trace",
        "no-trace-to-scratch",
        "scratch-table-with-defaulted-trace",
        "published-table-with-scratch-trace",
        "scratch-table-with-scratch-trace",
        "whole-roster-traced",
        "another-seed-onto-the-default-paths",
        "another-seed-to-scratch",
        "another-seed-naming-the-published-table",
        "another-seed-naming-the-published-trace",
        "another-seed-subset-naming-the-published-table",
        "trace-over-the-published-table",
        "table-over-the-published-trace",
        "table-over-the-per-seed-table",
        "run-record-over-the-published-record",
        "table-over-comparison-all",
        "table-over-bounds",
        "trace-over-analysis-notes",
        "trace-over-the-out-run-record",
        "trace-over-scip-baseline",
        "scratch-table-naming-the-published-trace",
        "smoke-budget-onto-the-published-table",
        "smoke-budget-onto-the-per-seed-table",
        "smoke-budget-to-scratch",
        "smoke-budget-another-seed-to-scratch",
        "nonpositive-budget",
        "nonpositive-build-jobs",
    ],
)
def test_usage_error(
    tmp_path: Path, overrides: dict[str, object], refusal: tuple[str, ...] | None
) -> None:
    (tmp_path / "sub").mkdir()
    paths = {k: tmp_path / v for k, v in overrides.items() if isinstance(v, str)}
    message = usage_error(
        make_args(tmp_path, **{**overrides, **paths}), tmp_path / "comparison.csv"
    )
    if refusal is None:
        assert message is None
    else:
        assert message is not None and all(word in message for word in refusal), message


# --- staging: what may be resumed from ---------------------------------------


@pytest.mark.parametrize(
    ("text", "sha", "complete"),
    [
        # The runner writes its header before it solves, so existence proves nothing.
        (HEADER + "\n", "abc1234", False),
        (HEADER + "\n" + ROW + "\n", "abc1234", True),
        (None, "abc1234", False),
        # Resuming onto it would publish a table whose rows name two engines.
        (HEADER + "\n" + ROW + "\n", "def5678", False),
        # A job killed mid-write leaves a short line that still reads as a line.
        (HEADER + "\nnvs01,1,1", "abc1234", False),
        # Torn inside the last cell: every column is present, only the newline is not.
        (HEADER + "\n" + ROW[:-1], "abc1234", False),
    ],
    ids=[
        "header-only",
        "whole-row",
        "absent",
        "another-commit",
        "torn-final-line",
        "torn-inside-the-last-cell",
    ],
)
def test_staged_row_complete(tmp_path: Path, text: str | None, sha: str, complete: bool) -> None:
    path = tmp_path / "nvs01.csv"
    if text is not None:
        path.write_text(text)
    assert staged_row_complete(path, sha) is complete


def _stage(
    tmp_path: Path, note: str = "feasible", sha: str = "abc1234", trace: bool = True
) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    row = f"a,NaN,1,1,NaN,NaN,0,false,{note},{sha},NaN,0,NaN,NaN,NaN,NaN,{DEFAULT_ARM}"
    (stage / "a.csv").write_text(f"{HEADER}\n{row}\n")
    if trace:
        (stage / "a.trace.csv").write_text(TRACE_HEADER + "\n")
    return stage


@pytest.mark.parametrize(
    ("note", "stands_in"),
    [
        # The #153 hole: the row a thrown instance leaves is COMPLETE by every
        # structural check, and `--resume` is the default. `run_roster` aborts on
        # the first run, but without this the next would skip the instance and
        # `publish` would assemble a row that measured nothing.
        ("solve-error", False),
        ("read-error", False),
        ("build-error", False),
        # The guard is an allowlist, so a note added later fails safe:
        # `convert-error` stands in for any future `++t.errored` site.
        ("convert-error", False),
        ("", False),
        # A coverage gap exits 0 and is a documented row, not an error: keying the
        # refusal on anything else would abort every publish run on a roster with
        # one unsupported instance.
        ("unsupported: NL_UNKNOWN_OPCODE 42", True),
        ("not-found", True),
    ],
    ids=[
        "thrown",
        "read-error",
        "build-error",
        "unrecognised",
        "empty",
        "unsupported",
        "not-found",
    ],
)
def test_only_a_measurement_or_a_coverage_gap_stands_in_for_a_solve(
    tmp_path: Path, note: str, stands_in: bool
) -> None:
    stage = _stage(tmp_path, note)
    assert staged_row_complete(stage / "a.csv", "abc1234")
    assert staged_complete(make_args(tmp_path), "abc1234", "a", stage) is stands_in


def test_a_staging_dir_from_another_seed_is_refused(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    assert staging_stamp_conflict(stage, make_args(tmp_path, seed=7), "abc1234") is None
    assert staging_stamp_conflict(stage, make_args(tmp_path), "abc1234") is not None


def test_a_staged_csv_without_its_trace_is_not_complete_when_tracing(tmp_path: Path) -> None:
    """A --no-trace run must not let a later traced run skip straight to assembly."""
    stage = _stage(tmp_path, trace=False)
    assert not staged_complete(make_args(tmp_path), "abc1234", "a", stage)
    assert staged_complete(make_args(tmp_path, trace=False), "abc1234", "a", stage)


@pytest.mark.parametrize(
    ("first", "second", "refused", "stamped"),
    [
        (None, (60.0, "abc1234", True), False, "commit=abc1234"),
        # Only wall_seconds would betray a 5s smoke run resumed into a 60s publish.
        ((5.0, "abc1234", True), (60.0, "abc1234", True), True, "time-limit=5"),
        ((60.0, "abc1234", True), (60.0, "def5678", True), True, "commit=abc1234"),
        ((5.0, "abc1234", True), (60.0, "abc1234", False), False, "time-limit=60"),
    ],
    ids=["fresh-dir-stamped", "another-budget", "another-commit", "no-resume-restamps"],
)
def test_a_staging_dir_is_resumed_only_under_its_own_configuration(
    tmp_path: Path,
    first: tuple[float, str, bool] | None,
    second: tuple[float, str, bool],
    refused: bool,
    stamped: str,
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    if first is not None:
        staging_stamp_conflict(stage, make_args(tmp_path, time_limit=first[0]), first[1])
    time_limit, sha, resume = second
    conflict = staging_stamp_conflict(
        stage, make_args(tmp_path, time_limit=time_limit, resume=resume), sha
    )
    assert (conflict is not None) is refused
    assert conflict is None or "--no-resume" in conflict
    assert stamped in (stage / STAMP_NAME).read_text()


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
    assert "--trace" not in runner_command(make_args(tmp_path, trace=False), "abc1", "n", tmp_path)


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
    rc = describe_plan(
        args, "abc1234", [], resolve_paths(args, "abc1234"), ["bounds.csv is missing"]
    )
    assert rc == 2
    assert "WOULD REFUSE" in capsys.readouterr().out


def test_a_clean_dry_run_prints_the_solve_command_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = make_args(tmp_path)
    rc = describe_plan(args, "abc1234", ["nvs01"], resolve_paths(args, "abc1234"), [])
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


@pytest.mark.parametrize(
    ("staged_sha", "skipped"),
    [(None, False), ("abc1234", True), ("old0000", False)],
    ids=["nothing-staged", "staged-at-this-commit", "staged-at-another-commit"],
)
def test_run_roster_solves_what_is_not_staged_and_keeps_its_log(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    staged_sha: str | None,
    skipped: bool,
) -> None:
    stage = _stage(tmp_path, sha=staged_sha) if staged_sha else tmp_path / "stage"
    stage.mkdir(exist_ok=True)
    monkeypatch.setattr(subprocess, "run", fake_runner())

    run_roster(make_args(tmp_path), "abc1234", ["a", "b"], stage)

    assert ("a: staged already, skipping" in capsys.readouterr().out) is skipped
    assert (stage / "a.log").exists() is not skipped
    assert (stage / "b.log").read_text() == "runner tally\n"
    assert staged_complete(make_args(tmp_path), "abc1234", "a", stage)


@pytest.mark.parametrize(
    ("runner", "said"),
    [
        (fake_runner(returncode=2), "Re-running resumes"),
        # A runner that wrote its header and died leaves no row to record. Not a
        # read or build error: those write a row and, since #153, exit nonzero.
        (fake_runner(write_row=False), "exit 0"),
        # Exit 3 is not the generic failure, and its message must not say it is:
        # "Re-running resumes from here" is true of a killed job and false of a
        # deterministic throw.
        (
            fake_runner(returncode=RUNNER_EXIT_ERRORED, note="solve-error", feasible="false"),
            "measures nothing",
        ),
    ],
    ids=["runner-failed", "exit-zero-without-a-row", "runner-error-tally"],
)
def test_run_roster_stops_on_a_failed_instance_and_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runner: Callable[..., object], said: str
) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(subprocess, "run", runner)
    with pytest.raises(RuntimeError, match=said):
        run_roster(make_args(tmp_path), "abc1234", ["a", "b"], stage)
    assert (stage / "a.log").exists()
    assert not (stage / "b.csv").exists(), "the roster went on past a failed instance"


# --- assembly ----------------------------------------------------------------


def _stage_two(tmp_path: Path) -> Path:
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "b.csv").write_text(HEADER + "\nb,2,2,2,0,0,60,true,feasible,abc1234,0,0\n")
    (stage / "a.csv").write_text(HEADER + "\na,1,1,1,0,0,60,true,matches-bks,abc1234,0,0\n")
    return stage


def test_assemble_emits_one_header_in_roster_order_and_nothing_beside_it(tmp_path: Path) -> None:
    stage = _stage_two(tmp_path)
    out = tmp_path / "comparison.csv"
    assemble(stage, ["a", "b"], out, ".csv")
    lines = out.read_text().splitlines()
    assert lines[0] == HEADER
    assert [line.split(",")[0] for line in lines[1:]] == ["a", "b"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["comparison.csv", "stage"]


@pytest.mark.parametrize(
    ("b_csv", "roster", "error", "match"),
    [
        ("instance,objective\nb,2\n", ["a", "b"], RuntimeError, "header differs"),
        ("", ["a", "b"], RuntimeError, "is empty"),
        # A failed assembly must leave the previously published table in place.
        (None, ["a", "b", "c"], FileNotFoundError, None),
    ],
    ids=["different-header", "empty-staging-file", "missing-row"],
)
def test_a_failed_assembly_leaves_the_published_table_alone(
    tmp_path: Path, b_csv: str | None, roster: list[str], error: type[Exception], match: str | None
) -> None:
    stage = _stage_two(tmp_path)
    if b_csv is not None:
        (stage / "b.csv").write_text(b_csv)
    out = tmp_path / "comparison.csv"
    out.write_text("previous table\n")
    with pytest.raises(error, match=match):
        assemble(stage, roster, out, ".csv")
    assert out.read_text() == "previous table\n"


def test_assemble_keeps_a_header_only_trace_file(tmp_path: Path) -> None:
    """An instance that never reaches feasibility contributes no trace rows."""
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "a.trace.csv").write_text(TRACE_HEADER + "\na,1.0,5,true\n")
    (stage / "b.trace.csv").write_text(TRACE_HEADER + "\n")
    out = tmp_path / "anytime_trace.csv"
    assemble(stage, ["a", "b"], out, ".trace.csv")
    assert out.read_text() == TRACE_HEADER + "\na,1.0,5,true\n"


# --- per-seed publication and the run record (#141) ---------------------------


REAL_RUN = subprocess.run


def seeded_runner(offset: float = 0.0) -> Callable[..., object]:
    """`fake_runner`, with instance `a`'s gap set by the seed so seeds are told apart.

    Anything that is not a runner invocation goes to the real `subprocess.run`:
    `machine_record` reaches it through `platform.processor()`.
    """

    def run(cmd: Sequence[str], **kwargs: object) -> object:
        if "--instance" not in cmd:
            return REAL_RUN(cmd, **kwargs)  # type: ignore[call-overload]
        name = cmd[cmd.index("--instance") + 1]
        sha = cmd[cmd.index("--commit") + 1]
        gap = 10.0 * int(cmd[cmd.index("--seed") + 1]) + offset if name == "a" else 0.0
        row = f"{name},1,1,1,{gap:g},0,60,true,feasible,{sha},0,0,0,0,1,0.5,{DEFAULT_ARM}"
        path_after(cmd, "--out").write_text(f"{HEADER}\n{row}\n")
        path_after(cmd, "--trace").write_text(TRACE_HEADER + "\n")
        return FakeCompleted(0)

    return run


def _seeds_in(table: Path) -> list[tuple[str, str, str]]:
    with table.open(newline="") as fh:
        return [(r["seed"], r["instance"], r["gap_to_bks%"]) for r in csv.DictReader(fh)]


def _execute(tmp_path: Path, seed: int) -> int:
    args = make_args(tmp_path, seed=seed, build=False, merge=False)
    return execute(args, "abc1234", ["a", "b"], resolve_paths(args, "abc1234"))


def test_seeds_accumulate_and_only_the_preregistered_seed_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The issue's acceptance criteria end to end, against a fake runner.

    Seed 2 runs FIRST and must not create `comparison.csv`; seed 1 then writes
    it; seed 3 must not replace it. Every seed lands in the per-seed table with
    its seed on the row, each published set gets a run record, and the summary
    reports the spread across all three.
    """
    inst = make_inst_dir(tmp_path, ["a", "b"])
    monkeypatch.setattr(subprocess, "run", seeded_runner())
    assert _execute(tmp_path, 2) == 0
    assert not (inst / "comparison.csv").exists()
    assert not (inst / "anytime_trace.csv").exists()
    assert _execute(tmp_path, 1) == 0
    published = (inst / "comparison.csv").read_text()
    assert _execute(tmp_path, 3) == 0
    assert (inst / "comparison.csv").read_text() == published
    with (inst / "comparison.csv").open(newline="") as fh:
        assert [r["gap_to_bks%"] for r in csv.DictReader(fh)] == ["10", "0"]

    table = inst / SEEDS_TABLE_NAME
    with table.open(newline="") as fh:
        assert tuple(next(csv.reader(fh))) == SEEDS_TABLE_COLUMNS
    assert _seeds_in(table) == [
        ("1", "a", "10"),
        ("1", "b", "0"),
        ("2", "a", "20"),
        ("2", "b", "0"),
        ("3", "a", "30"),
        ("3", "b", "0"),
    ]

    single = json.loads((inst / "comparison.run.json").read_text())
    assert (single["seed"], single["budget_seconds"], single["commit"]) == (1, 60.0, "abc1234")
    assert single["resumed"] == 0
    assert "cpu_model" in single["machine"]
    assert single["table_sha256"] == file_sha256(inst / "comparison.csv")
    assert single["machine"]["host"] == socket.gethostname()
    assert single["concurrency"]["parallel_solves"] == 1
    seeds = json.loads((inst / "comparison_seeds.run.json").read_text())["seeds"]
    assert seeds["2"]["table_sha256"] == seed_block_hashes(table)[2]
    assert sorted(seeds) == ["1", "2", "3"]
    assert all("memory_total_kib" in record["machine"] for record in seeds.values())
    stage2 = resolve_paths(make_args(tmp_path, seed=2), "abc1234").stage
    assert (stage2 / ASSEMBLED_TABLE).exists() and (stage2 / ASSEMBLED_TRACE).exists()
    assert (stage2 / "comparison.assembled.run.json").exists()

    out = capsys.readouterr().out
    assert "seeds aggregated: 1, 2, 3 (commit abc1234, 60s per instance)" in out
    assert "  a: 20 [10, 30], 3/3" in out
    assert "feasible (roster count): min 2, median 2, max 2" in out


def test_rerunning_a_seed_replaces_its_rows_and_keeps_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inst = make_inst_dir(tmp_path, ["a", "b"])
    monkeypatch.setattr(subprocess, "run", seeded_runner())
    assert _execute(tmp_path, 1) == 0
    assert _execute(tmp_path, 2) == 0
    monkeypatch.setattr(subprocess, "run", seeded_runner(offset=5.0))
    args = make_args(tmp_path, seed=2, build=False, merge=False, resume=False)
    assert execute(args, "abc1234", ["a", "b"], resolve_paths(args, "abc1234")) == 0
    assert _seeds_in(inst / SEEDS_TABLE_NAME) == [
        ("1", "a", "10"),
        ("1", "b", "0"),
        ("2", "a", "25"),
        ("2", "b", "0"),
    ]


def test_a_seeds_table_of_another_shape_is_refused_and_left_alone(tmp_path: Path) -> None:
    table = tmp_path / SEEDS_TABLE_NAME
    table.write_text("seed,instance,objective\n7,a,1\n")
    out = tmp_path / "comparison.csv"
    full = f"{HEADER}\na,1,1,1,0,0,60,true,feasible,abc1234,0,0,0,0,1,0.5,{DEFAULT_ARM}\n"
    out.write_text(full)
    with pytest.raises(RuntimeError, match="has columns"):
        publish_seed_rows(table, 1, out)
    assert table.read_text() == "seed,instance,objective\n7,a,1\n"


def test_a_resumed_run_says_its_record_did_not_see_the_resumed_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    inst = make_inst_dir(tmp_path, ["a", "b"])
    monkeypatch.setattr(subprocess, "run", seeded_runner())
    assert _execute(tmp_path, 1) == 0
    assert _execute(tmp_path, 1) == 0
    assert json.loads((inst / "comparison.run.json").read_text())["resumed"] == 2
    assert "2 of 2 row(s) were staged by an earlier invocation" in capsys.readouterr().err


def test_a_second_driver_on_the_machine_is_refused_with_exit_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two seeds in two terminals would share the machine and race on the per-seed table.

    `main` must refuse in one line and exit 2, not raise a traceback.
    """
    inst = make_inst_dir(tmp_path, ["a"])
    build = make_build_dir(tmp_path)
    (build / RUNNER_TARGET).write_text("")
    monkeypatch.setattr(run_benchmark, "run_commit_sha", lambda *_args: "abc1234")
    monkeypatch.setattr(subprocess, "run", seeded_runner())
    argv = ["--inst-dir", str(inst), "--build-dir", str(build), "--no-build", "--no-merge"]
    with wallclock_lock("another driver"):
        assert run_benchmark.main(argv) == 2
    err = capsys.readouterr().err
    assert "refusing to run: another wall-clock benchmark holds" in err
    assert "another driver" in err
    assert not (inst / "comparison.csv").exists()


def test_the_ablation_driver_takes_the_same_machine_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """It runs serial timed MINLPLib solves too, so it must not run beside this driver."""
    from benchmarks.minlplib import run_ablation

    inst = make_inst_dir(tmp_path, ["a"])
    build = make_build_dir(tmp_path)
    (build / RUNNER_TARGET).write_text("")
    monkeypatch.setattr(run_ablation, "commit_sha", lambda: "abc1234")
    monkeypatch.setattr(run_ablation, "execute", lambda *a, **k: 0)
    argv = [
        "--out-dir",
        str(tmp_path / "abl"),
        "--inst-dir",
        str(inst),
        "--build-dir",
        str(build),
        "--no-build",
        "--allow-busy",
    ]
    with wallclock_lock("run_benchmark.py"):
        assert run_ablation.main(argv) == 2
    assert "refusing to run: another wall-clock benchmark holds" in capsys.readouterr().err
    assert run_ablation.main(argv) == 0


def test_a_changed_roster_is_a_stamp_conflict(tmp_path: Path) -> None:
    """The staging root is shared: rows staged for one bounds.csv must not resume into another."""
    inst = make_inst_dir(tmp_path, ["a", "b"])
    stage = tmp_path / "stage"
    stage.mkdir()
    assert staging_stamp_conflict(stage, make_args(tmp_path), "abc1234") is None
    assert staging_stamp_conflict(stage, make_args(tmp_path), "abc1234") is None
    (inst / "bounds.csv").write_text((inst / "bounds.csv").read_text().replace("1.0,1.0", "2,2"))
    conflict = staging_stamp_conflict(stage, make_args(tmp_path), "abc1234")
    assert conflict is not None and "roster=" in conflict


def test_preflight_refuses_a_per_seed_row_it_could_not_upsert(tmp_path: Path) -> None:
    """`publish_seed_rows` sorts by int(seed): a bad cell would raise after the solves."""
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path)
    table = tmp_path / "inst" / SEEDS_TABLE_NAME
    good = ",".join(["1", "process", *["0"] * (len(SEEDS_TABLE_COLUMNS) - 2)])
    table.write_text(",".join(SEEDS_TABLE_COLUMNS) + "\n" + good + "\n")
    assert preflight(make_args(tmp_path), "abc1234", ["process"]) == []
    table.write_text(table.read_text() + good.replace("1,", "x,", 1) + "\n")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("line 3 is not a per-seed row" in p for p in problems), problems


@pytest.mark.parametrize("cell", ["--1", "\u00b2", "1.0", " 1"])
def test_preflight_refuses_a_seed_cell_int_would_not_read_back(tmp_path: Path, cell: str) -> None:
    """`isdigit` admits `²` and a stripped `--1`; the upsert would raise after the solves."""
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path)
    table = tmp_path / "inst" / SEEDS_TABLE_NAME
    row = ",".join([cell, "process", *["0"] * (len(SEEDS_TABLE_COLUMNS) - 2)])
    table.write_text(",".join(SEEDS_TABLE_COLUMNS) + "\n" + row + "\n")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("line 2 is not a per-seed row" in p for p in problems), problems


def test_the_upsert_replaces_a_seed_written_with_a_leading_zero(tmp_path: Path) -> None:
    """Seeds compare as integers: a hand-written `01` block is seed 1's, not a second one."""
    table = tmp_path / SEEDS_TABLE_NAME
    cells = ["0"] * (len(SEEDS_TABLE_COLUMNS) - 2)
    table.write_text(",".join(SEEDS_TABLE_COLUMNS) + "\n" + ",".join(["01", "a", *cells]) + "\n")
    out = tmp_path / "comparison.csv"
    out.write_text(f"{HEADER}\na,1,1,1,0,0,60,true,feasible,abc1234,0,0,0,0,1,0.5,{DEFAULT_ARM}\n")
    publish_seed_rows(table, 1, out)
    assert [r[0] for r in csv.reader(table.open(newline=""))][1:] == ["1"]


def test_a_failed_merge_makes_the_publish_exit_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The table is written but comparison_all.csv still holds the previous rows: not a success."""
    make_inst_dir(tmp_path, ["a", "b"])
    runner = seeded_runner()

    def run(cmd: Sequence[str], **kwargs: object) -> object:
        if "--merge-only" in cmd:
            return FakeCompleted(2)
        return runner(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    args = make_args(tmp_path, build=False, merge=True)
    assert execute(args, "abc1234", ["a", "b"], resolve_paths(args, "abc1234")) == 1


def test_the_ablation_drivers_preflight_still_runs(tmp_path: Path) -> None:
    """`run_ablation.py` shares the common preflight with a namespace that has no --seed."""
    from benchmarks.minlplib import run_ablation

    make_inst_dir(tmp_path, ["process"])
    args = run_ablation.parse_args(
        [
            "--out-dir",
            str(tmp_path / "abl"),
            "--inst-dir",
            str(tmp_path / "inst"),
            "--build-dir",
            str(tmp_path / "nobuild"),
        ]
    )
    problems = common_preflight(args, "abc1234", ["process"])
    assert any("CMakeCache.txt not found" in p for p in problems)


def test_preflight_refuses_a_seeds_table_the_run_could_not_join(tmp_path: Path) -> None:
    """Found after fifty minutes of solving otherwise: `publish_seed_rows` refuses the same."""
    make_inst_dir(tmp_path, ["process"])
    make_build_dir(tmp_path)
    table = tmp_path / "inst" / SEEDS_TABLE_NAME
    table.write_text("seed,instance,objective\n")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("another runner version" in p for p in problems), problems
    table.write_text(",".join(SEEDS_TABLE_COLUMNS) + "\n")
    assert preflight(make_args(tmp_path), "abc1234", ["process"]) == []
    (tmp_path / "inst" / "comparison_seeds.run.json").write_text("{broken")
    problems = preflight(make_args(tmp_path), "abc1234", ["process"])
    assert any("not a readable JSON object" in p for p in problems), problems
    # A scratch run publishes nothing, so it does not care.
    scratch = make_args(tmp_path, out=tmp_path / "s.csv", trace_out=tmp_path / "t.csv")
    assert preflight(scratch, "abc1234", ["process"]) == []


# --- the derived summary -----------------------------------------------------


def test_a_verdict_drops_the_analysis_note_the_runner_glued_on() -> None:
    """Otherwise one annotated row becomes its own histogram bucket."""
    assert verdict_of("feasible | bug: Thomson problem") == "feasible"
    assert verdict_of("infeasible(residual=1; 25 viol) | hard") == "infeasible"
    assert verdict_of("matches-bks; int-mismatch") == "matches-bks"


def _summary_inputs(tmp_path: Path, excluded_row: str) -> tuple[Path, Path]:
    out = tmp_path / "comparison.csv"
    rows = [
        HEADER,
        "a,1,1,1,0,0,60,true,matches-bks,abc1234,0,0",
        "b,2,1,1,100,100,60,true,feasible | bug: x,abc1234,0,0",
        excluded_row,
    ]
    out.write_text("\n".join(rows) + "\n")
    bounds = tmp_path / "bounds.csv"
    bounds.write_text(
        "instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks\n"
        + "".join(f"{n},other,1,1,min,1,1,0\n" for n in ("a", "b", CLAIM_EXCLUDED[0]))
    )
    return out, bounds


def test_summary_counts_elec_in_the_roster_and_out_of_the_quality_aggregates(
    tmp_path: Path,
) -> None:
    """#142: one rule for both denominators, and the summary states it.

    Before #142 the driver dropped `elec` from every line while the README's tally
    counted the whole roster. Now the roster counts include it and only the
    quality aggregates hold it out, under `campaign_report.AGGREGATION_RULE`.
    """
    excluded = f"{CLAIM_EXCLUDED[0]},NaN,1,1,NaN,NaN,60,false,infeasible(residual=1),abc1234,1,0"
    out, bounds = _summary_inputs(tmp_path, excluded)
    text = summarize(out, bounds)
    assert f"rule: {AGGREGATION_RULE}" in text
    assert "  roster:               3" in text
    assert "  feasible:             2" in text
    assert "  infeasible:           1" in text
    assert "quality aggregates (2 feasible claim-set rows):" in text
    assert "  matches-bks:          1" in text
    assert "  worse than BKS:       1" in text
    assert f"excluded from quality aggregates: {CLAIM_EXCLUDED[0]} -> infeasible" in text
    assert "WARNING" not in text
    # Failure kinds the old verdict histogram showed must still be visible.
    assert "  verification failed:  0" in text
    assert "  integrality mismatch: 0" in text
    assert "  no published bound:   0" in text


def test_summary_warns_when_a_documented_failure_comes_back_feasible(tmp_path: Path) -> None:
    """README: a feasible `elec` row is a result to check, not a table refresh."""
    excluded = f"{CLAIM_EXCLUDED[0]},1,1,1,0,0,60,true,matches-bks,abc1234,0,0"
    out, bounds = _summary_inputs(tmp_path, excluded)
    text = summarize(out, bounds)
    assert "  feasible:             3" in text
    assert "quality aggregates (2 feasible claim-set rows):" in text
    assert f"WARNING: documented failure {CLAIM_EXCLUDED[0]} came back FEASIBLE" in text


# --- the runner contract, pinned against minlplib.cpp ---------------------------


def test_the_published_header_still_matches_what_the_runner_writes() -> None:
    """Every `RUNNER_COLUMNS` name appears in `minlplib.cpp`, delimited as a cell.

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
    assert all(f"{column}," in source for column in RUNNER_COLUMNS[:-1])
    assert f'{RUNNER_COLUMNS[-1]}\\n"' in source
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
    assert list(RUNNER_COLUMNS) == [
        *columns,
        "lns_repairs",
        "lns_repairs_accepted",
        "first_feasible_objective",
        "time_to_first_feasible",
        "search_config",
    ]


def test_the_drivers_exit_constant_is_the_runners_own() -> None:
    """`RUNNER_EXIT_ERRORED` is mirrored from C++, and both drivers branch on it.

    Pinned against the literal rather than against a run, so a change to one side
    fails here instead of being discovered by a campaign that stopped recording
    failed runs.
    """
    source = (REPO_ROOT / "benchmarks" / "minlplib" / "minlplib.cpp").read_text()
    assert f"constexpr int kExitErrored = {RUNNER_EXIT_ERRORED};" in source
    # The tally the status is computed FROM, not just the constant: a status
    # read off `skipped_unsupported` or `verify_failed` would be the same
    # integer for the wrong reason.
    assert "run_exit_status(t.errored)" in source


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
            *("--time-limit", "1", "--commit", "abc1234", "--out", str(tmp_path / "out.csv")),
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


# --- the held-out roster (#144) ------------------------------------------------

HELDOUT_DIR = REPO_ROOT / "benchmarks" / "instances" / "minlplib" / "heldout"
PUBLISHED_DIR = REPO_ROOT / "benchmarks" / "instances" / "minlplib"


@pytest.mark.parametrize(
    "overrides",
    [{}, {"merge": False}, {"seed": 2}, {"out": "s.csv", "trace_out": "t.csv"}],
    ids=["default-paths", "no-merge", "another-seed", "scratch-out-defaulted-staging"],
)
def test_the_heldout_roster_publishes_nothing(tmp_path: Path, overrides: dict[str, object]) -> None:
    """HELDOUT.md: every output of a held-out run goes outside benchmarks/instances/."""
    paths = {k: tmp_path / v for k, v in overrides.items() if isinstance(v, str)}
    args = make_args(tmp_path, **{"inst_dir": HELDOUT_DIR, **overrides, **paths})
    message = usage_error(args, HELDOUT_DIR / "comparison.csv")
    assert message is not None and "--inst-dir" in message, message


def test_a_heldout_run_onto_scratch_paths_is_allowed(tmp_path: Path) -> None:
    args = make_args(
        tmp_path,
        inst_dir=HELDOUT_DIR,
        out=tmp_path / "s.csv",
        trace_out=tmp_path / "t.csv",
        staging_dir=tmp_path / "stage",
    )
    assert usage_error(args, HELDOUT_DIR / "comparison.csv") is None


@pytest.mark.parametrize(
    ("inst_dir", "out"),
    [
        (HELDOUT_DIR, PUBLISHED_DIR / "comparison.csv"),
        (PUBLISHED_DIR, HELDOUT_DIR / "bounds.csv"),
    ],
    ids=["heldout-run-over-the-published-table", "published-run-over-the-heldout-roster"],
)
def test_a_scratch_output_may_not_reach_another_roster_directory(
    tmp_path: Path, inst_dir: Path, out: Path
) -> None:
    args = make_args(
        tmp_path,
        inst_dir=inst_dir,
        out=out,
        trace_out=tmp_path / "t.csv",
        staging_dir=tmp_path / "stage",
    )
    message = usage_error(args, inst_dir / "comparison.csv")
    assert message is not None and "--out" in message, message


# --- the dirty guard across a multi-seed campaign (#123) ----------------------

GIT_IDENTITY = ("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false")


def _git(repo: Path, *argv: str) -> str:
    """Run git in `repo`. Safe only because conftest strips a hook's GIT_* variables."""
    done = subprocess.run(["git", *argv], cwd=repo, check=True, capture_output=True, text=True)
    return done.stdout.strip()


def _checkout(tmp_path: Path) -> tuple[Path, str]:
    """`tmp_path` as a real repository: a source file, and an instance directory holding
    the previous campaign's three tracked tables. Returns the instance directory and SHA.
    """
    _git(tmp_path, "init", "-q")
    inst = make_inst_dir(tmp_path, ["a", "b"])
    (tmp_path / "engine.cpp").write_text("int main() {}\n")
    for name in ("comparison.csv", "anytime_trace.csv", "comparison_all.csv"):
        (inst / name).write_text("the previous campaign\n")
    _git(tmp_path, "add", "engine.cpp", "inst")
    _git(tmp_path, *GIT_IDENTITY, "-c", "core.hooksPath=/dev/null", "commit", "-qm", "c")
    return inst, _git(tmp_path, "rev-parse", "--short=7", "HEAD")


def _publish_seed_1(tmp_path: Path, inst: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Seed 1's publish, as the README's seed loop leaves the checkout; returns its SHA.

    Through `execute` with the fake runner; the merge's rewrite of
    `comparison_all.csv` is stood in for by hand.
    """
    sha = run_benchmark.run_commit_sha(inst, tmp_path)
    assert not sha.endswith("-dirty")
    args = make_args(tmp_path, seed=1, build=False, merge=False)
    with monkeypatch.context() as patched:
        patched.setattr(subprocess, "run", seeded_runner())
        assert execute(args, sha, ["a", "b"], resolve_paths(args, sha)) == 0
    (inst / "comparison_all.csv").write_text("the merge's rewrite\n")
    modified = _git(tmp_path, "status", "--porcelain", "--untracked-files=no").splitlines()
    assert len(modified) == 3, modified  # all three tracked tables moved
    return sha


def _seed_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *extra: str) -> int:
    """`main` at seed 2, reading its SHA from the temporary checkout."""
    build = make_build_dir(tmp_path)
    (build / RUNNER_TARGET).write_text("")
    monkeypatch.setattr(run_benchmark, "REPO_ROOT", tmp_path)
    argv = ["--inst-dir", str(tmp_path / "inst"), "--build-dir", str(build), "--no-build"]
    return run_benchmark.main([*argv, "--seed", "2", *extra])


def test_seed_2_runs_at_seed_1s_sha_over_the_tables_seed_1_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The README's `for SEED in 1 2 3` loop, which #123's campaign found refused at seed 2.

    Seed 1's publish modifies three tracked tables. They are this driver's own
    output, not the code that ran, so seed 2 must see the same plain SHA --
    committing between seeds instead would split the campaign across two
    commits, and the seed summary keeps one.
    """
    inst, _ = _checkout(tmp_path)
    sha = _publish_seed_1(tmp_path, inst, monkeypatch)
    assert run_benchmark.run_commit_sha(inst, tmp_path) == sha
    assert _seed_2(tmp_path, monkeypatch, "--dry-run") == 0
    out = capsys.readouterr().out
    assert f"commit {sha}, " in out
    assert "WOULD REFUSE" not in out


@pytest.mark.parametrize("seed_1_published", [False, True], ids=["source-only", "with-published"])
def test_a_modified_source_file_still_refuses_seed_2(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seed_1_published: bool,
) -> None:
    """Ignoring the published tables must not ignore anything beside them."""
    inst, sha = _checkout(tmp_path)
    if seed_1_published:
        _publish_seed_1(tmp_path, inst, monkeypatch)
    (tmp_path / "engine.cpp").write_text("int main() { return 1; }\n")
    assert run_benchmark.run_commit_sha(inst, tmp_path) == f"{sha}-dirty"
    assert _seed_2(tmp_path, monkeypatch) == 2
    refusal = f"refusing to run: working tree is dirty ({sha}-dirty): engine.cpp; commit"
    assert refusal in capsys.readouterr().err


def test_a_modified_instance_directory_input_still_refuses_seed_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An instance file sits beside the published tables but is an input, not their output."""
    inst, sha = _checkout(tmp_path)
    _publish_seed_1(tmp_path, inst, monkeypatch)
    (inst / "a.nl").write_text("edited\n")
    assert _seed_2(tmp_path, monkeypatch, "--dry-run") == 2
    refusal = f"WOULD REFUSE: working tree is dirty ({sha}-dirty): inst/a.nl; commit"
    assert refusal in capsys.readouterr().out
