"""Unit tests for #331's GM(gap+1) section of the presolve-322 report (#354).

Offline: rows are built directly, or written as a `results.csv` for the
end-to-end `--report-only` test. What is pinned is what #331's decision is read
off: the pool each ratio is taken over, the noise floor's one instance set, and
the banner a partial campaign carries.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib import independent_check, run_ablation
from benchmarks.minlplib.ablation_report import CONTROL_ARM, RunRow
from benchmarks.minlplib.presolve_report import (
    PRESOLVE_SECTION,
    PresolvePlan,
    missing_triples,
    presolve_lines,
)
from benchmarks.minlplib.run_ablation import (
    PRESOLVE_ARMS,
    PRESOLVE_CAMPAIGN,
    RESULT_COLUMNS,
    RESULTS_NAME,
    STAMP_NAME,
    campaign_stamp,
    main,
    recorded_roster_size,
    stamped_presolve_roster,
)
from benchmarks.minlplib.runner import RUNNER_FAILED_NOTE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

ARMS = tuple(arm.name for arm in PRESOLVE_ARMS)
SEEDS = (1, 2, 3)


@pytest.fixture(autouse=True)
def _no_independent_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """The rows here have no `.nl` for SCIP's check (#205) to read."""
    monkeypatch.setattr(independent_check, "check_rows", lambda *_a, **_k: [])


def row(instance: str, arm: str, seed: int, gap: float, *, feasible: bool = True) -> RunRow:
    """A completed search; `feasible=False` is a search that found nothing."""
    return RunRow(
        instance=instance,
        arm=arm,
        seed=seed,
        feasible=feasible,
        gap=gap if feasible else float("nan"),
        primal_bks=1.0,
        lns_repairs=0.0,
        lns_repairs_accepted=0.0,
        wall=1.0,
        note="feasible" if feasible else "infeasible",
    )


def grid(instances: Sequence[str], gap: Callable[[str, str, int], float | None]) -> list[RunRow]:
    """Every (instance, arm, seed); `gap` returns None for an infeasible run."""
    rows = []
    for instance in instances:
        for seed in SEEDS:
            for arm in ARMS:
                g = gap(instance, arm, seed)
                rows.append(
                    row(instance, arm, seed, 0.0 if g is None else g, feasible=g is not None)
                )
    return rows


def plan(instances: Sequence[str], roster_size: int | None = None) -> PresolvePlan:
    return PresolvePlan(ARMS, SEEDS, tuple(instances), roster_size)


def table_row(lines: list[str], arm: str, seed: int) -> list[str]:
    """The cells of the per-seed table's line for (arm, seed)."""
    for line in lines:
        cells = line.split()
        if len(cells) == 8 and cells[0] == arm and cells[1] == str(seed):
            return cells
    raise AssertionError(f"no line for {arm} seed {seed}")


def pooled_row(lines: list[str], arm: str) -> str:
    start = lines.index("pooled over seeds:")
    return next(line for line in lines[start + 2 :] if line.split()[:1] == [arm])


def test_the_ratio_is_below_one_when_the_arm_is_ahead() -> None:
    # gap+1 = 1 under presolve, 4 (4.4 at seed 2) elsewhere: a floor of 1.1,
    # and the arm about 4x ahead.
    def gap(_instance: str, arm: str, seed: int) -> float:
        return 0.0 if arm == "presolve" else (3.4 if seed == 2 else 3.0)

    lines = presolve_lines(grid(["a", "b"], gap), plan(["a", "b"]))
    assert lines[0] == PRESOLVE_SECTION
    assert lines[1].startswith("complete:")
    assert table_row(lines, "presolve", 1)[-1] == "0.2500"
    assert pooled_row(lines, "presolve").endswith("arm ahead, outside the floor")
    assert pooled_row(lines, "both").endswith("inside the floor")


def test_a_floor_of_one_is_no_noise_measured_not_any_difference_counts() -> None:
    def gap(_instance: str, arm: str, _seed: int) -> float:
        return 2.9999 if arm == "presolve" else 3.0

    lines = presolve_lines(grid(["a"], gap), plan(["a"]))
    assert pooled_row(lines, "presolve").endswith("no seed noise measured; read the ratio itself")


def test_the_pool_is_the_instances_feasible_in_both_the_arm_and_the_control() -> None:
    # b is feasible only under the control, at gap+1 = 10. Pooled in, it would
    # put the control's GM at sqrt(10) and the arm far ahead; the arm never
    # found b, so it must not be credited with the control's bad gap there.
    def gap(instance: str, arm: str, _seed: int) -> float | None:
        if instance == "b":
            return 9.0 if arm == CONTROL_ARM else None
        return 0.0

    lines = presolve_lines(grid(["a", "b"], gap), plan(["a", "b"]))
    cells = table_row(lines, "presolve", 1)
    assert cells[2:] == ["1/2", "2/2", "1", "1.0000", "1.0000", "1.0000"]


def test_the_floor_is_taken_over_one_instance_set_for_every_seed() -> None:
    # The control's gaps on a and c are the same at every seed: no seed noise.
    # b is perfect under the control at seeds 1 and 3 but lost by `both` at
    # seed 2. A per-seed set would drop b at seed 2 only and read a floor of
    # 2 / 4**(1/3) = 1.26 out of an arm's loss. One common set drops b at
    # every seed and reads 1.
    def gap(instance: str, arm: str, seed: int) -> float | None:
        if instance == "b":
            return None if (arm == "both" and seed == 2) else 0.0
        return 1.0

    lines = presolve_lines(grid(["a", "b", "c"], gap), plan(["a", "b", "c"]))
    floor = next(line for line in lines if line.startswith("noise floor:"))
    assert "the 2 instance(s)" in floor
    assert "max / min over seeds = 1.0000" in floor


def test_a_partial_campaign_is_flagged_and_its_missing_runs_named() -> None:
    rows = [
        r
        for r in grid(["a", "b"], lambda *_: 1.0)
        if (r.instance, r.arm, r.seed) != ("a", "both", 2)
    ]
    assert missing_triples(rows, plan(["a", "b"])) == [("a", "both", 2)]
    lines = presolve_lines(rows, plan(["a", "b"]))
    assert any(line.startswith("INCOMPLETE") for line in lines)
    assert any("1 run(s) missing: a/both/seed2" in line for line in lines)

    complete = presolve_lines(grid(["a", "b"], lambda *_: 1.0), plan(["a", "b"]))
    assert not any("INCOMPLETE" in line for line in complete)
    assert any(
        line.startswith("complete: 2 instance(s) x 4 arm(s) x 3 seed(s)") for line in complete
    )


def test_an_instance_with_no_row_at_all_is_caught_by_the_stamped_roster_size() -> None:
    """Rows name only the instances that ran; the stamp says how many should have."""
    rows = grid(["a", "b"], lambda *_: 1.0)
    lines = presolve_lines(rows, plan([], roster_size=25))
    assert any(line.startswith("INCOMPLETE") for line in lines)
    assert any("the stamp records 25 instance(s); the rows name 2" in line for line in lines)


def test_a_planned_instance_with_no_row_is_named_missing() -> None:
    lines = presolve_lines(grid(["a"], lambda *_: 1.0), plan(["a", "z"]))
    assert any("12 run(s) missing: z/control/seed1" in line for line in lines)
    assert any(" and 2 more" in line for line in lines)


def test_a_stray_instance_cannot_stand_in_for_a_missing_one() -> None:
    rows = grid(["a", "b", "d"], lambda *_: 1.0)
    counted = presolve_lines(rows, plan([], roster_size=3))  # names unknown: count only
    assert any(line.startswith("complete:") for line in counted)  # the count alone cannot tell
    named = presolve_lines(rows, plan(["a", "b", "c"], roster_size=3))
    assert any("the stamp records 3 instance(s); the rows name 4" in line for line in named)
    assert any("outside the planned roster: d" in line for line in named)
    assert any("12 run(s) missing: c/control/seed1" in line for line in named)


def test_a_run_recorded_twice_is_flagged() -> None:
    rows = grid(["a"], lambda *_: 1.0)
    lines = presolve_lines([*rows, row("a", "presolve", 2, 5.0)], plan(["a"]))
    assert any(line.startswith("INCOMPLETE") for line in lines)
    assert any("1 run(s) recorded more than once: a/presolve/seed2" in line for line in lines)


def test_a_crashed_run_is_neither_a_lost_feasibility_nor_a_recorded_run() -> None:
    """The driver records a crash, so a resume skips it; it is no measurement."""
    rows = grid(["a", "b"], lambda *_: 1.0)
    crashed = [
        replace(r, feasible=False, note=f"{RUNNER_FAILED_NOTE}: segfault")
        if (r.instance, r.arm, r.seed) == ("b", CONTROL_ARM, 2)
        else r
        for r in rows
    ]
    lines = presolve_lines(crashed, plan(["a", "b"]))
    assert table_row(lines, "presolve", 2)[2:5] == ["2/2", "1/1", "1"]
    assert "6/6" in pooled_row(lines, "presolve")
    assert "5/5" in pooled_row(lines, "presolve")
    assert any(line.startswith("INCOMPLETE") for line in lines)
    assert any("1 run(s) recorded without a completed search" in line for line in lines)
    assert any(line.endswith(": b/control/seed2") for line in lines)


def test_a_claim_excluded_instance_enters_no_count_or_ratio() -> None:
    # elec25 is far worse under the control; pooled in, it would put the arm ahead.
    def gap(instance: str, arm: str, _seed: int) -> float | None:
        return 99.0 if (instance == "elec25" and arm == CONTROL_ARM) else 0.0

    lines = presolve_lines(grid(["a", "elec25"], gap), plan(["a", "elec25"], roster_size=2))
    assert table_row(lines, "presolve", 1)[2:] == ["1/1", "1/1", "1", "1.0000", "1.0000", "1.0000"]
    assert any(line.startswith("complete: 1 instance(s)") for line in lines)


def _results_csv(path: Path, rows: Sequence[RunRow]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RESULT_COLUMNS)
        for r in rows:
            cells = dict.fromkeys(RESULT_COLUMNS, "0")
            cells.update(
                instance=r.instance,
                arm=r.arm,
                seed=str(r.seed),
                feasible="true" if r.feasible else "false",
                note=r.note,
            )
            cells["gap_to_bks%"] = str(r.gap) if r.feasible else ""
            writer.writerow([cells[column] for column in RESULT_COLUMNS])


def test_report_only_reads_the_seeds_and_roster_size_off_the_stamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    (out_dir / STAMP_NAME).write_text(
        campaign_stamp(
            "abc1234", 10.0, list(SEEDS), PRESOLVE_ARMS, ["a", "b", "c"], campaign=PRESOLVE_CAMPAIGN
        )
    )
    assert recorded_roster_size(out_dir) == 3
    _results_csv(out_dir / RESULTS_NAME, grid(["a", "b"], lambda *_: 1.0))
    assert main(["--out-dir", str(out_dir), "--report-only"]) == 0
    out = capsys.readouterr().out
    assert PRESOLVE_SECTION in out
    assert "the stamp records 3 instance(s); the rows name 2" in out

    (out_dir / STAMP_NAME).write_text(
        campaign_stamp(
            "abc1234", 10.0, list(SEEDS), PRESOLVE_ARMS, ["a", "b"], campaign=PRESOLVE_CAMPAIGN
        )
    )
    assert main(["--out-dir", str(out_dir), "--report-only"]) == 0
    out = capsys.readouterr().out
    assert "complete: 2 instance(s) x 4 arm(s) x 3 seed(s), every run recorded" in out


def test_report_only_names_the_sample_the_stamp_hashes_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stamp holds a count and a hash; where the committed sample has that
    hash, the report names the instance that never ran, and the stray one."""
    sample = tmp_path / "sample.txt"
    sample.write_text("a\nb\nc\n")
    monkeypatch.setattr(run_ablation, "PRESOLVE_SAMPLE", sample)
    out_dir = tmp_path / "scratch"
    out_dir.mkdir()
    (out_dir / STAMP_NAME).write_text(
        campaign_stamp(
            "abc1234", 10.0, list(SEEDS), PRESOLVE_ARMS, ["a", "b", "c"], campaign=PRESOLVE_CAMPAIGN
        )
    )
    assert stamped_presolve_roster(out_dir) == ("a", "b", "c")
    _results_csv(out_dir / RESULTS_NAME, grid(["a", "b", "d"], lambda *_: 1.0))
    assert main(["--out-dir", str(out_dir), "--report-only"]) == 0
    out = capsys.readouterr().out
    assert "outside the planned roster: d" in out
    assert "12 run(s) missing: c/control/seed1" in out

    sample.write_text("a\nb\n")  # a sample edited since: not the stamp's, count only
    assert stamped_presolve_roster(out_dir) == ()
