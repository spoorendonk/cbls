"""Tests for `benchmarks/minlplib/gm_ratio.py` (issue #300).

Every fixture is SYNTHETIC and small: a roster, a cbls table or staging
directory, and a SCIP shard directory, all under `tmp_path`. The GM(gap+1)
arithmetic is pinned against hand-computed numbers, not against the module.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib import gm_ratio
from benchmarks.minlplib.gm_ratio import (
    EXIT_REFUSED,
    CblsSide,
    RefusedError,
    Score,
    gap_plus_one,
    gm_gap_plus_one,
    head_to_head,
    load_cbls_staging,
    load_cbls_table,
    load_roster,
    load_scip_shards,
    main,
    reason_category,
    score,
)
from benchmarks.minlplib.run_benchmark import STAMP_NAME, checked_marker, mark_checked
from benchmarks.minlplib.runner import CLAIM_EXCLUDED, RUNNER_COLUMNS

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from benchmarks.minlplib.reference_solve import Bound

SCIP_COLUMNS = (
    "instance,objective,primal_bks,dual_bound,gap_to_bks%,gap_to_dual%,wall_seconds,feasible,"
    "note,scip_dual_bound,scip_gap%,status,n_int_vars,read_seconds,solving_seconds,scip_version"
)
SCIP_VERSION = "SCIP 10.0.2 / PySCIPOpt 6.3.0 / 60s / seed 0"
SHA = "abc1234"


def write_bounds(path: Path, rows: list[tuple[str, str]]) -> Path:
    """`rows` is (instance, primal_bks); an empty primal_bks is no BKS."""
    lines = ["instance,structure,nvars,ncons,objsense,primal_bks,dual_bound,n_disc_vars_bks"]
    lines += [f"{name},other,1,1,min,{bks},,0" for name, bks in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


def cbls_line(
    name: str,
    gap: float | str,
    *,
    feasible: bool = True,
    note: str = "feasible",
    sha: str = SHA,
    config: str = "threads=16",
) -> str:
    cells = {c: "NaN" for c in RUNNER_COLUMNS}
    cells.update(
        instance=name,
        objective="1",
        primal_bks="1",
        wall_seconds="60",
        feasible="true" if feasible else "false",
        note=note,
        commit_sha=sha,
        n_int_vars="0",
        search_config=config,
    )
    cells["gap_to_bks%"] = str(gap)
    return ",".join(cells[c] for c in RUNNER_COLUMNS)


def write_cbls(path: Path, lines: list[str]) -> Path:
    path.write_text(",".join(RUNNER_COLUMNS) + "\n" + "".join(f"{x}\n" for x in lines))
    return path


def write_scip(
    shards: Path,
    name: str,
    gap: float | str,
    *,
    feasible: bool = True,
    version: str = SCIP_VERSION,
    note: str = "",
    status: str = "timelimit",
) -> None:
    """A note left empty is the solve's own: `feasible`, or `no-solution(<status>)`."""
    shards.mkdir(exist_ok=True)
    flag = "true" if feasible else "false"
    note = note or ("feasible" if feasible else f"no-solution({status})")
    row = f"{name},1,1,NaN,{gap},NaN,60,{flag},{note},NaN,NaN,{status},0,0,60,{version}"
    (shards / f"{name}.csv").write_text(f"{SCIP_COLUMNS}\n{row}\n")


def write_failed(shards: Path, name: str, detail: str = "exit 134") -> None:
    shards.mkdir(exist_ok=True)
    (shards / f"{name}.failed").write_text(detail + "\n")


def campaign(
    tmp_path: Path,
    rows: list[tuple[str, float | str, float | str]],
    *,
    bks: dict[str, str] | None = None,
) -> tuple[Path, Path, Path]:
    """Roster, cbls table and SCIP shards from (instance, cbls gap, SCIP gap).

    A gap of "-" means that solver found nothing (feasible=false).
    """
    bks = bks or {}
    bounds = write_bounds(tmp_path / "bounds.csv", [(n, bks.get(n, "1")) for n, _, _ in rows])
    shards = tmp_path / "scip"
    shards.mkdir()
    lines = []
    for name, gc, gs in rows:
        found = gc != "-"
        note = "feasible" if found else "infeasible"
        lines.append(cbls_line(name, gc if found else "NaN", feasible=found, note=note))
        write_scip(shards, name, "NaN" if gs == "-" else gs, feasible=gs != "-")
    table = write_cbls(tmp_path / "comparison.csv", lines)
    return bounds, table, shards


def scored(bounds: Path, table: Path, shards: Path, *, partial: bool = False) -> Score:
    roster_list = load_roster(bounds)
    roster = {b.instance: b for b in roster_list}
    return score(
        roster_list,
        CblsSide(rows=load_cbls_table(table, roster)),
        load_scip_shards(shards, roster),
        partial=partial,
    )


def write_stamp(stage: Path, time_limit: str = "60") -> None:
    """The fields `run_benchmark.staging_stamp` writes; only time-limit is read."""
    stage.mkdir(exist_ok=True)
    (stage / STAMP_NAME).write_text(
        f"commit={SHA}\ntime-limit={time_limit}\nseed=0\nthreads=16\nhost=h\nroster=r\n"
    )


def staging_argv(bounds: Path, stage: Path, shards: Path) -> list[str]:
    return ["--bounds", str(bounds), "--cbls-staging", str(stage), "--scip-shards", str(shards)]


# --------------------------------------------------------------------------
# The arithmetic.
# --------------------------------------------------------------------------


def test_gm_of_gaps_zero_three_minus_one_is_the_cube_root_of_four() -> None:
    # gap+1 = 1, 4, 1 (the -1 clamps to 0): GM = (1 * 4 * 1) ** (1/3).
    assert gm_gap_plus_one([0.0, 3.0, -1.0]) == pytest.approx(4 ** (1 / 3))


def test_gm_of_an_empty_pool_is_nan() -> None:
    assert math.isnan(gm_gap_plus_one([]))


def test_a_better_than_bks_gap_clamps_to_one() -> None:
    assert gap_plus_one(-50.0) == 1.0
    assert gap_plus_one(0.0) == 1.0
    assert gap_plus_one(2.5) == 3.5


@pytest.mark.parametrize(
    ("gap_cbls", "gap_scip", "expected"),
    [
        (0.0, 0.0101, 1),  # 1 < 1.0101 / 1.01 = 1.0000990...: a win
        (0.0, 0.0099, 0),  # 1.0099 / 1.01 < 1: inside 1%, a tie
        (0.0101, 0.0, -1),
        (0.0099, 0.0, 0),
        (-5.0, -1.0, 0),  # both clamp to 1
        (9.0, 0.0, -1),
    ],
)
def test_head_to_head_needs_more_than_one_percent_on_gap_plus_one(
    gap_cbls: float, gap_scip: float, expected: int
) -> None:
    assert head_to_head(gap_cbls, gap_scip) == expected


def test_head_to_head_at_exactly_one_percent_is_a_tie() -> None:
    # (gs+1) = 1.01 * (gc+1) exactly, in binary: 2.02 / 1.01 == 2.0.
    assert (1.02 + 1.0) / 1.01 == 2.0
    assert head_to_head(1.0, 1.02) == 0
    assert head_to_head(1.02, 1.0) == 0


def test_ratio_is_gm_cbls_over_gm_scip_over_the_both_solved_pool(tmp_path: Path) -> None:
    bounds, table, shards = campaign(
        tmp_path, [("a", 0.0, 1.0), ("b", 3.0, 0.0), ("c", -1.0, 8.0), ("d", 2.0, "-")]
    )
    s = scored(bounds, table, shards)
    assert s.gm_pool == 3
    assert s.gm_cbls == pytest.approx(4 ** (1 / 3))  # 1, 4, 1
    assert s.gm_scip == pytest.approx(18 ** (1 / 3))  # 2, 1, 9
    assert s.ratio == pytest.approx((4 / 18) ** (1 / 3))
    assert (s.wins, s.losses, s.ties) == (2, 1, 0)


# --------------------------------------------------------------------------
# Counts.
# --------------------------------------------------------------------------


def test_only_one_feasible_instances_are_bracketed_not_won(tmp_path: Path) -> None:
    bounds, table, shards = campaign(
        tmp_path,
        [("a", 0.0, 0.0), ("b", 1.0, "-"), ("c", 2.0, "-"), ("d", "-", 0.0), ("e", "-", "-")],
    )
    s = scored(bounds, table, shards)
    assert (s.cbls_feasible, s.scip_feasible, s.both_feasible) == (3, 2, 1)
    assert (s.wins, s.losses, s.ties) == (0, 0, 1)
    assert (s.only_cbls, s.only_scip) == (2, 1)


def test_claim_excluded_instances_are_out_of_every_count_and_listed_by_name(
    tmp_path: Path,
) -> None:
    # #228's smoke: n=48 of 50, the feasible counts without elec too.
    excluded = CLAIM_EXCLUDED[0]
    bounds, table, shards = campaign(
        tmp_path, [("a", 0.0, 0.0), (excluded, 99.0, 0.0), (CLAIM_EXCLUDED[1], 1.0, "-")]
    )
    s = scored(bounds, table, shards)
    assert (s.n_roster, s.n_counted, s.n_scored) == (3, 1, 1)
    assert (s.cbls_feasible, s.scip_feasible, s.both_feasible) == (1, 1, 1)
    assert s.gm_pool == 1
    assert s.gm_cbls == 1.0
    assert (s.wins, s.losses, s.only_cbls) == (0, 0, 0)
    assert s.scip_no_solution == {}
    assert s.claim_excluded == [excluded, CLAIM_EXCLUDED[1]]


def test_instances_without_a_finite_bks_are_listed_and_kept_out_of_the_gm(
    tmp_path: Path,
) -> None:
    bounds, table, shards = campaign(
        tmp_path,
        [("a", 0.0, 1.0), ("nobks", "NaN", "NaN"), ("nobks2", "NaN", "-")],
        bks={"nobks": "", "nobks2": ""},
    )
    s = scored(bounds, table, shards)
    assert s.both_feasible == 2
    assert s.gm_pool == 1
    assert s.no_bks == ["nobks", "nobks2"]
    assert s.only_cbls == 1  # an only-one bracket needs no gap


def test_a_scip_failure_marker_is_scip_no_solution_listed_with_its_exit(
    tmp_path: Path,
) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0)])
    write_bounds(bounds, [("a", "1"), ("crashed", "1")])
    write_cbls(table, [cbls_line("a", 0.0), cbls_line("crashed", 0.0)])
    write_failed(shards, "crashed", "exit 134")
    s = scored(bounds, table, shards)
    assert s.scip_feasible == 1
    assert s.only_cbls == 1
    assert s.scip_no_solution == {"crashed": "exit 134"}


def test_a_verify_failed_cbls_row_is_no_solution(tmp_path: Path) -> None:
    # As the driver demotes it: feasible=false, the independent check's note.
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    demoted = cbls_line("b", "NaN", feasible=False, note="VERIFY-FAILED(independent: x)")
    write_cbls(table, [cbls_line("a", 0.0), demoted])
    s = scored(bounds, table, shards)
    assert s.cbls_feasible == 1
    assert s.only_scip == 1
    assert s.cbls_no_solution == {"b": "VERIFY-FAILED(independent: x)"}


def test_every_cbls_no_solution_row_is_listed_with_its_note(tmp_path: Path) -> None:
    names = ["a", "rd", "inf", "uns"]
    bounds, table, shards = campaign(tmp_path, [(n, 0.0, 0.0) for n in names])
    write_cbls(
        table,
        [
            cbls_line("a", 0.0),
            cbls_line("rd", "NaN", feasible=False, note="read-error(bad .nl header)"),
            cbls_line("inf", "NaN", feasible=False, note="infeasible"),
            cbls_line("uns", "NaN", feasible=False, note="unsupported: erf"),
        ],
    )
    s = scored(bounds, table, shards)
    assert s.cbls_no_solution == {
        "rd": "read-error(bad .nl header)",
        "inf": "infeasible",
        "uns": "unsupported: erf",
    }
    assert len(s.cbls_no_solution) == s.n_scored - s.cbls_feasible


def test_scip_read_error_and_check_failed_rows_are_listed_with_their_notes(
    tmp_path: Path,
) -> None:
    names = ["a", "rd", "chk", "lim"]
    bounds, table, shards = campaign(tmp_path, [(n, 0.0, 0.0) for n in names])
    write_scip(
        shards, "rd", "NaN", feasible=False, note="read-error(OSError: x)", status="read-error"
    )
    write_scip(
        shards, "chk", "NaN", feasible=False, note="CHECK-FAILED(SCIP could not re-validate)"
    )
    write_scip(shards, "lim", "NaN", feasible=False)
    s = scored(bounds, table, shards)
    assert s.scip_no_solution == {
        "rd": "read-error(OSError: x)",
        "chk": "CHECK-FAILED(SCIP could not re-validate)",
        "lim": "no-solution(timelimit)",
    }
    assert len(s.scip_no_solution) == s.n_scored - s.scip_feasible


def test_markdown_groups_no_solution_instances_by_reason_category(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = ["a", "x1", "x2", "w", "y", "z"]
    bounds, table, shards = campaign(tmp_path, [(n, 0.0, 0.0) for n in names])
    for name in ("x1", "x2"):
        write_scip(shards, name, "NaN", feasible=False)
    write_scip(shards, "w", "NaN", feasible=False, status="infeasible")
    (shards / "y.csv").unlink()
    write_failed(shards, "y", "exit 134")
    write_scip(shards, "z", "NaN", feasible=False, note="read-error(OSError: x)")
    argv = ["--bounds", str(bounds), "--cbls", str(table), "--scip-shards", str(shards)]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert "- SCIP no solution: 5\n" in text
    assert (
        "  - no-solution(timelimit): 2 (x1, x2)\n"
        "  - exit 134: 1 (y)\n"
        "  - no-solution(infeasible): 1 (w)\n"
        "  - read-error: 1 (z)\n"
    ) in text
    assert reason_category("read-error(bad .nl header)") == "read-error"
    assert reason_category("rejected: lost workers") == "rejected"
    annotated = "no-solution(timelimit); integrality mismatch: SCIP 3 vs catalogue 4"
    assert reason_category(annotated) == "no-solution(timelimit)"
    assert "- cbls no solution: 0\n" in text


# --------------------------------------------------------------------------
# Completeness.
# --------------------------------------------------------------------------


def test_a_roster_instance_without_a_cbls_row_is_refused(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    write_cbls(table, [cbls_line("a", 0.0)])
    with pytest.raises(RefusedError, match="1 without a cbls row: b"):
        scored(bounds, table, shards)


def test_a_roster_instance_without_a_scip_shard_is_refused(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    (shards / "b.csv").unlink()
    with pytest.raises(RefusedError, match="without a SCIP row or failure marker: b"):
        scored(bounds, table, shards)


def test_a_refusal_names_ten_missing_instances_and_counts_the_rest(tmp_path: Path) -> None:
    names = [f"i{k:02d}" for k in range(13)]
    bounds, table, shards = campaign(tmp_path, [(n, 0.0, 0.0) for n in names])
    write_cbls(table, [])
    with pytest.raises(
        RefusedError, match=r"13 without a cbls row: i00, .*i09, \.\.\. \(\+3 more\)"
    ):
        scored(bounds, table, shards)


def test_a_result_for_an_instance_outside_the_roster_is_refused(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0)])
    write_scip(shards, "stranger", 0.0)
    with pytest.raises(RefusedError, match="'stranger' is not in the roster"):
        scored(bounds, table, shards, partial=True)
    (shards / "stranger.csv").unlink()
    write_cbls(table, [cbls_line("a", 0.0), cbls_line("stranger", 0.0)])
    with pytest.raises(RefusedError, match="not in bounds.csv"):
        scored(bounds, table, shards, partial=True)


def test_a_duplicated_instance_is_refused(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0)])
    write_cbls(table, [cbls_line("a", 0.0), cbls_line("a", 1.0)])
    with pytest.raises(RefusedError, match="appears twice"):
        scored(bounds, table, shards, partial=True)
    write_cbls(table, [cbls_line("a", 0.0)])
    write_failed(shards, "a")
    with pytest.raises(RefusedError, match="both a result row and a failure marker"):
        scored(bounds, table, shards, partial=True)


def test_partial_scores_only_instances_present_on_both_sides(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 3.0), ("b", 0.0, 0.0), ("c", 0.0, 0.0)])
    write_cbls(table, [cbls_line("a", 0.0), cbls_line("b", 0.0)])
    (shards / "b.csv").unlink()
    s = scored(bounds, table, shards, partial=True)
    assert (s.n_roster, s.n_scored, s.partial) == (3, 1, True)
    assert s.missing_cbls == ["c"]
    assert s.missing_scip == ["b"]
    assert s.gm_scip == 4.0


def test_main_exits_two_on_an_incomplete_campaign(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    (shards / "b.csv").unlink()
    argv = ["--bounds", str(bounds), "--cbls", str(table), "--scip-shards", str(shards)]
    assert main(argv) == EXIT_REFUSED
    assert "incomplete campaign" in capsys.readouterr().err


def test_main_partial_prints_the_banner_first_and_records_it_in_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    (shards / "b.csv").unlink()
    out = tmp_path / "score.json"
    argv = ["--bounds", str(bounds), "--cbls", str(table), "--scip-shards", str(shards)]
    assert main([*argv, "--partial", "--json", str(out), "--budget", "60 s"]) == 0
    text = capsys.readouterr().out
    assert text.splitlines()[0] == "PARTIAL: 1 of 2 instances, not a result"
    assert "- budget: 60s per instance (from SCIP scip_version)" in text
    record = json.loads(out.read_text())
    assert record["partial"] is True
    assert record["n_scored"] == 1
    assert record["provenance"]["cbls_threads"] == 16


def test_main_on_a_complete_campaign_prints_no_banner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0)])
    argv = ["--bounds", str(bounds), "--cbls", str(table), "--scip-shards", str(shards)]
    assert main(argv) == 0
    text = capsys.readouterr().out
    assert text.startswith("## MINLPLib GM(gap+1)")
    assert "PARTIAL" not in text


# --------------------------------------------------------------------------
# Provenance.
# --------------------------------------------------------------------------


def test_mixed_engine_commits_are_refused(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    write_cbls(table, [cbls_line("a", 0.0), cbls_line("b", 0.0, sha="def5678")])
    with pytest.raises(RefusedError, match="mixed engine commits"):
        scored(bounds, table, shards)


def test_mixed_thread_counts_are_refused_and_an_absent_count_reads_as_one(
    tmp_path: Path,
) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    write_cbls(table, [cbls_line("a", 0.0), cbls_line("b", 0.0, config="weights=gls")])
    with pytest.raises(RefusedError, match="mixed cbls thread counts: '1', '16'"):
        scored(bounds, table, shards)
    write_cbls(table, [cbls_line("a", 0.0, config=""), cbls_line("b", 0.0, config="x=1")])
    assert scored(bounds, table, shards).provenance.cbls_threads == 1


def test_mixed_scip_versions_are_refused(tmp_path: Path) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("b", 0.0, 0.0)])
    write_scip(shards, "b", 0.0, version="SCIP 9.0.0 / PySCIPOpt 5.0.0 / 60s / seed 0")
    with pytest.raises(RefusedError, match="mixed SCIP versions"):
        scored(bounds, table, shards)


def test_a_scip_budget_unlike_the_staging_stamp_is_refused_and_agreement_printed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bounds, table, shards = campaign(tmp_path, [("a", 0.0, 0.0)])
    stage = tmp_path / "stage"
    write_stamp(stage, "30")
    write_cbls(stage / "a.csv", [cbls_line("a", 0.0)])
    argv = [*staging_argv(bounds, stage, shards), "--partial"]
    assert main(argv) == EXIT_REFUSED
    err = capsys.readouterr().err
    assert "budgets differ: SCIP scip_version 60s, cbls staging stamp 30s" in err

    write_stamp(stage, "60")
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "- budget: 60s per instance (from SCIP scip_version, cbls staging stamp)" in out
    assert main([*argv, "--budget", "45"]) == EXIT_REFUSED
    assert "--budget 45s, but the data records 60s" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Staging and shard directories.
# --------------------------------------------------------------------------


def test_staging_skips_traces_rejected_and_in_progress_files(tmp_path: Path) -> None:
    bounds = write_bounds(tmp_path / "bounds.csv", [("a", "1"), ("b", "1"), ("c", "1")])
    roster = {b.instance: b for b in load_roster(bounds)}
    stage = tmp_path / "stage"
    stage.mkdir()
    write_cbls(stage / "a.csv", [cbls_line("a", 0.0)])
    (stage / "a.trace.csv").write_text("instance,time_seconds,objective,new_best\n")
    write_cbls(stage / "b.csv", [])  # header only: the solve is still running
    write_cbls(stage / "c.trace.csv.rejected", [])
    write_cbls(stage / "comparison.assembled.csv", [cbls_line("a", 0.0)])
    side = load_cbls_staging(stage, roster)
    assert list(side.rows) == ["a"]
    assert side.rejected == {}


def test_a_rejected_only_instance_is_cbls_no_solution_with_its_note_not_missing(
    tmp_path: Path,
) -> None:
    bounds = write_bounds(tmp_path / "bounds.csv", [("a", "1"), ("c", "1"), ("t", "1")])
    shards = tmp_path / "scip"
    for name in ("a", "c", "t"):
        write_scip(shards, name, 0.0)
    roster_list = load_roster(bounds)
    roster = {b.instance: b for b in roster_list}
    stage = tmp_path / "stage"
    stage.mkdir()
    write_cbls(stage / "a.csv", [cbls_line("a", 0.0)])
    # exit 3: the runner threw; the row it staged says why.
    write_cbls(
        stage / "c.csv.rejected",
        [cbls_line("c", "NaN", feasible=False, note="solve-error(bad_alloc)")],
    )
    # lost workers: an ordinary-looking row, rejected on the exit status alone.
    write_cbls(stage / "t.csv.rejected", [cbls_line("t", 0.0)])
    side = load_cbls_staging(stage, roster)
    assert side.rejected == {"c": "rejected: solve-error(bad_alloc)", "t": "rejected: feasible"}
    s = score(roster_list, side, load_scip_shards(shards, roster))
    assert s.missing_cbls == []
    assert s.cbls_feasible == 1
    assert s.only_scip == 2
    assert s.cbls_no_solution == side.rejected


def test_a_whole_staged_row_beside_a_rejected_one_is_a_re_solve_and_wins(
    tmp_path: Path,
) -> None:
    bounds = write_bounds(tmp_path / "bounds.csv", [("c", "1")])
    roster = {b.instance: b for b in load_roster(bounds)}
    stage = tmp_path / "stage"
    stage.mkdir()
    write_cbls(stage / "c.csv.rejected", [cbls_line("c", "NaN", feasible=False, note="x")])
    write_cbls(stage / "c.csv", [cbls_line("c", 2.0)])
    side = load_cbls_staging(stage, roster)
    assert side.rejected == {}
    assert side.rows["c"].gap_pct == 2.0


def checked_staging(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A complete campaign, staged: `a` solved and checked, `c` rejected twice over."""
    bounds, _, shards = campaign(tmp_path, [("a", 0.0, 0.0), ("c", "-", 0.0)])
    stage = tmp_path / "stage"
    write_stamp(stage)
    write_cbls(stage / "a.csv", [cbls_line("a", 0.0)])
    mark_checked(stage, "a")
    write_cbls(
        stage / "c.csv.rejected",
        [cbls_line("c", "NaN", feasible=False, note="solve-error(bad_alloc)")],
    )
    return bounds, stage, shards


def test_a_checked_staging_dir_with_a_rejected_instance_scores_without_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#307: a cbls instance that kept failing is no-solution in a final score."""
    bounds, stage, shards = checked_staging(tmp_path)
    out = tmp_path / "score.json"
    assert main([*staging_argv(bounds, stage, shards), "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "PARTIAL" not in text
    assert "  - rejected: 1 (c)" in text
    record = json.loads(out.read_text())
    assert record["partial"] is False
    assert record["cbls_no_solution"] == {"c": "rejected: solve-error(bad_alloc)"}
    assert (record["cbls_feasible"], record["only_scip"]) == (1, 1)


@pytest.mark.parametrize(
    "spoil",
    ["no-marker", "stale-marker", "re-solve-in-progress"],
)
def test_an_unchecked_staged_row_is_refused_by_name_unless_partial(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spoil: str
) -> None:
    bounds, stage, shards = checked_staging(tmp_path)
    if spoil == "no-marker":
        checked_marker(stage, "a").unlink()
        named = "a"
    elif spoil == "stale-marker":
        write_cbls(stage / "a.csv", [cbls_line("a", 5.0)])
        named = "a"
    else:  # a killed re-solve left its header beside the old rejected row
        write_cbls(stage / "c.csv", [])
        named = "c"
    argv = staging_argv(bounds, stage, shards)
    assert main(argv) == EXIT_REFUSED
    err = capsys.readouterr().err
    assert "1 staged cbls row(s) without a matching .checked marker" in err
    assert f"check: {named}. Run check_staged.py" in err
    assert main([*argv, "--partial"]) == 0
    assert capsys.readouterr().out.startswith("PARTIAL")


def test_a_row_rewritten_while_it_is_read_is_refused_even_with_a_matching_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A driver's check demotes and marks `a` between the parse and the hash."""
    bounds, stage, shards = checked_staging(tmp_path)
    checked_marker(stage, "a").unlink()
    real_load = gm_ratio.load_cbls_staging

    def load_then_demote(path: Path, roster: Mapping[str, Bound]) -> CblsSide:
        side = real_load(path, roster)
        write_cbls(stage / "a.csv", [cbls_line("a", "NaN", feasible=False, note="VERIFY-FAILED")])
        mark_checked(stage, "a")
        return side

    monkeypatch.setattr(gm_ratio, "load_cbls_staging", load_then_demote)
    assert main(staging_argv(bounds, stage, shards)) == EXIT_REFUSED
    assert "staged rows changed while they were read" in capsys.readouterr().err


def test_a_staged_row_filed_under_another_instance_is_refused(tmp_path: Path) -> None:
    bounds = write_bounds(tmp_path / "bounds.csv", [("a", "1"), ("b", "1")])
    roster = {b.instance: b for b in load_roster(bounds)}
    stage = tmp_path / "stage"
    stage.mkdir()
    write_cbls(stage / "a.csv", [cbls_line("b", 0.0)])
    with pytest.raises(RefusedError, match="the row is for 'b'"):
        load_cbls_staging(stage, roster)


def test_scip_shards_ignore_merged_files_and_logs(tmp_path: Path) -> None:
    bounds = write_bounds(tmp_path / "bounds.csv", [("a", "1")])
    roster = {b.instance: b for b in load_roster(bounds)}
    shards = tmp_path / "scip"
    write_scip(shards, "a", 0.0)
    (shards / "a.merged.csv").write_text("instance,method\na,scip\n")
    (shards / "a.log").write_text("log\n")
    side = load_scip_shards(shards, roster)
    assert list(side.rows) == ["a"]
    assert side.failures == {}


def test_a_roster_listing_an_instance_twice_is_refused(tmp_path: Path) -> None:
    bounds = write_bounds(tmp_path / "bounds.csv", [("a", "1"), ("a", "1")])
    with pytest.raises(RefusedError, match="listed twice in the roster: a"):
        load_roster(bounds)
