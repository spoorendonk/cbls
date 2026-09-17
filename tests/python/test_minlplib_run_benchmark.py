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
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks.common.provenance import REPO_ROOT
from benchmarks.minlplib.run_benchmark import (
    STAMP_NAME,
    assemble,
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


def test_paths_default_to_the_published_tables_and_a_build_staging_dir(tmp_path: Path) -> None:
    paths = resolve_paths(make_args(tmp_path))
    assert paths.out == tmp_path / "inst" / "comparison.csv"
    assert paths.out == paths.published_out
    assert paths.trace_out == tmp_path / "inst" / "anytime_trace.csv"
    assert paths.stage == tmp_path / "build" / "minlplib-rerun"


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
