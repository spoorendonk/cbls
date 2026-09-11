"""Tests for `benchmarks/minlplib/first_feasible_report.py` (issue #149).

Every fixture here is SYNTHETIC. Nothing in this file solves anything, and
nothing reads a published table: the module under test is a scorer, and a test
that needed a real campaign to run could only be run by whoever had six hours of
idle machine.

The anchor fixture is `ISSUE_134_PAIRS` -- the eight (first feasible, final)
pairs issue #134 published for `nvs01`, with the r it reported. A scorer that
does not reproduce that number cannot be trusted with the roster, and no amount
of internal consistency would show it up.
"""

from __future__ import annotations

import csv
import math
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib.first_feasible_report import (
    MIN_ELIGIBLE_INSTANCES,
    R_DETERMINED,
    REFERENCE_INSTANCE,
    InstanceResult,
    collect,
    group,
    main,
    parse_args,
    read_table,
    score_instance,
    usage_error,
    verdict,
)

if TYPE_CHECKING:
    from pathlib import Path

RUNNER_HEADER = [
    "instance",
    "objective",
    "primal_bks",
    "dual_bound",
    "gap_to_bks%",
    "gap_to_dual%",
    "wall_seconds",
    "feasible",
    "note",
    "commit_sha",
    "max_violation",
    "n_int_vars",
    "lns_repairs",
    "lns_repairs_accepted",
    "first_feasible_objective",
    "time_to_first_feasible",
    "search_config",
]

#: #134's table for `nvs01`: (first feasible objective, final objective) at
#: seeds 1, 17, 42, 11, 7, 2, 13, 3. The issue reports Pearson r = 0.945.
ISSUE_134_PAIRS = (
    (16.49, 12.4697),
    (16.76, 12.4697),
    (17.03, 12.4697),
    (18.60, 14.62),
    (117.0, 22.09),
    (186.8, 33.65),
    (236.8, 41.64),
    (176.9, 45.64),
)


def runner_row(
    instance: str,
    *,
    objective: float | str,
    first_feasible: float | str,
    seconds: float | str = 0.25,
    feasible: bool = True,
    note: str = "feasible",
) -> list[str]:
    """One `comparison.csv`-shaped row, in the runner's column order."""
    return [
        instance,
        str(objective),
        "1",
        "1",
        "0",
        "0",
        "60",
        "true" if feasible else "false",
        note,
        "abc1234",
        "0",
        "0",
        "0",
        "0",
        str(first_feasible),
        str(seconds),
        "arm",
    ]


def write_table(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RUNNER_HEADER)
        writer.writerows(rows)
    return path


def seed_tables(tmp_path: Path, per_seed: dict[int, list[list[str]]]) -> list[str]:
    """Write one table per seed and return the `--table SEED=PATH` arguments."""
    args = []
    for seed, rows in per_seed.items():
        path = write_table(tmp_path / f"seed{seed}.csv", rows)
        args += ["--table", f"{seed}={path}"]
    return args


def nvs01_issue_134(tmp_path: Path) -> list[str]:
    """#134's eight seeds, one table each, as `--table` arguments."""
    return seed_tables(
        tmp_path,
        {
            seed: [runner_row(REFERENCE_INSTANCE, objective=final, first_feasible=first)]
            for seed, (first, final) in enumerate(ISSUE_134_PAIRS, start=1)
        },
    )


def score(argv: list[str]) -> list[InstanceResult]:
    args = parse_args(argv)
    assert usage_error(args) is None
    observations, _skipped, _rows = collect(args)
    return [
        score_instance(instance, found, args.min_seeds)
        for instance, found in sorted(group(observations).items())
    ]


# --- the anchor: #134's own measurement ----------------------------------------


def test_the_scorer_reproduces_the_correlation_issue_134_published(tmp_path: Path) -> None:
    """The eight nvs01 seeds must come back at r = 0.945, as #134 reported.

    This is the one assertion in the file that is checked against a number
    measured OUTSIDE it. Everything else here tests the scorer against its own
    conventions; this tests it against the measurement that caused the issue.
    """
    results = score(nvs01_issue_134(tmp_path))
    assert len(results) == 1
    assert results[0].instance == REFERENCE_INSTANCE
    assert results[0].seeds == 8
    assert results[0].pearson == pytest.approx(0.945, abs=5e-4)
    assert results[0].determined


def test_a_perfectly_correlated_instance_scores_one(tmp_path: Path) -> None:
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=float(seed), first_feasible=10.0 * seed)]
            for seed in range(1, 6)
        },
    )
    (result,) = score(argv)
    assert result.pearson == pytest.approx(1.0)
    assert result.spearman == pytest.approx(1.0)


def test_an_anticorrelated_instance_scores_minus_one_and_is_not_determined(
    tmp_path: Path,
) -> None:
    """A strong correlation of the WRONG SIGN must not count as the effect.

    `abs(r)` would score this instance 1.0 and read it as "the first feasible
    point determines the final objective", when what it says is the opposite:
    the runs that arrive worst finish best. The threshold is on r, not on its
    magnitude, and this pins that.
    """
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=float(-seed), first_feasible=10.0 * seed)]
            for seed in range(1, 6)
        },
    )
    (result,) = score(argv)
    assert result.pearson == pytest.approx(-1.0)
    assert result.eligible
    assert not result.determined


# --- which rows are usable -----------------------------------------------------


def test_a_row_with_no_first_feasible_reading_is_dropped_not_read_as_zero(
    tmp_path: Path,
) -> None:
    """The NaN rule, where it actually bites.

    A pre-#149 table, or a row the runner wrote without solving, carries NaN in
    both first-feasible cells. Read as 0 it would be the single most favourable
    observation possible -- "arrived instantly at objective zero" -- and on this
    fixture it drags a flat instance into a spurious r of 1.
    """
    rows = [
        runner_row("a", objective=5.0, first_feasible="NaN", seconds="NaN"),
    ]
    path = write_table(tmp_path / "t.csv", rows)
    found, skipped = read_table(path, seed=1, arm=None)
    assert found == []
    assert skipped.no_first_feasible == 1
    assert skipped.total() == 1


def test_an_infeasible_row_is_dropped_and_counted(tmp_path: Path) -> None:
    path = write_table(
        tmp_path / "t.csv",
        [
            runner_row(
                "a",
                objective="NaN",
                first_feasible="NaN",
                seconds="NaN",
                feasible=False,
                note="infeasible(residual=1)",
            )
        ],
    )
    found, skipped = read_table(path, seed=1, arm=None)
    assert found == []
    assert skipped.infeasible == 1


def test_a_feasible_row_with_a_non_finite_first_objective_is_dropped(tmp_path: Path) -> None:
    """#100's witness: feasible, recorded, but with no objective to correlate.

    `time_to_first_feasible` is a number here, which is the whole reason the
    engine keeps the two cells apart -- so this row must be bucketed as
    "non-finite", not as "no first-feasible reading".
    """
    path = write_table(
        tmp_path / "t.csv",
        [runner_row("a", objective=1.0, first_feasible="inf", seconds=0.5)],
    )
    found, skipped = read_table(path, seed=1, arm=None)
    assert found == []
    assert skipped.non_finite == 1
    assert skipped.no_first_feasible == 0


def test_the_rows_published_as_documented_failures_are_excluded(tmp_path: Path) -> None:
    """`elec25`/`elec50` are published as failures and excluded from every claim.

    A correlation over the roster is a quality claim, so they stay out of it --
    counted apart, so a reader can see they were dropped on purpose rather than
    wonder where they went.
    """
    path = write_table(
        tmp_path / "t.csv",
        [
            runner_row("elec25", objective=1.0, first_feasible=2.0),
            runner_row("a", objective=1.0, first_feasible=2.0),
        ],
    )
    found, skipped = read_table(path, seed=1, arm=None)
    assert [o.instance for o in found] == ["a"]
    assert skipped.claim_excluded == 1


def test_a_campaign_results_file_is_filtered_to_one_arm(tmp_path: Path) -> None:
    """An ablation `results.csv` carries every arm; only one may be scored.

    Mixing them would correlate two different engines' trajectories as though
    they were repeated draws of one.
    """
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "arm", "seed", *RUNNER_HEADER[1:]])
        for seed in range(1, 6):
            control = runner_row("a", objective=float(seed), first_feasible=10.0 * seed)
            other = runner_row("a", objective=99.0, first_feasible=1.0)
            writer.writerow([control[0], "control", str(seed), *control[1:]])
            writer.writerow([other[0], "no-lns", str(seed), *other[1:]])
    (result,) = score(["--results", str(path), "--arm", "control"])
    assert result.seeds == 5
    assert result.pearson == pytest.approx(1.0)


def test_a_seed_appearing_twice_is_not_weighted_double(tmp_path: Path) -> None:
    """One observation per seed, whatever the inputs contain.

    A `results.csv` with a duplicated `(instance, arm, seed)` triple -- the
    exact hazard `run_ablation`'s `--no-resume` note describes -- would
    otherwise weight that seed twice and shrink the spread the correlation sits
    on.
    """
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "arm", "seed", *RUNNER_HEADER[1:]])
        for seed in range(1, 6):
            row = runner_row("a", objective=float(seed), first_feasible=10.0 * seed)
            writer.writerow([row[0], "control", str(seed), *row[1:]])
        duplicate = runner_row("a", objective=1.0, first_feasible=10.0)
        writer.writerow([duplicate[0], "control", "1", *duplicate[1:]])
    (result,) = score(["--results", str(path), "--arm", "control"])
    assert result.seeds == 5


# --- eligibility ---------------------------------------------------------------


def test_an_instance_with_too_few_seeds_gets_no_correlation(tmp_path: Path) -> None:
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=float(seed), first_feasible=10.0 * seed)]
            for seed in range(1, 4)
        },
    )
    (result,) = score([*argv, "--min-seeds", "4"])
    assert not result.eligible
    assert math.isnan(result.pearson)
    assert "below --min-seeds 4" in result.ineligible


def test_an_instance_that_lands_on_the_same_objective_every_seed_is_ineligible(
    tmp_path: Path,
) -> None:
    """Zero spread is not r = 0.

    An instance solved to the same objective on every seed has no across-seed
    variance to explain, so it can neither support nor refute the effect.
    Scoring it 0 would drag the roster median down with instances that are
    evidence of nothing; and `statistics.correlation` raises on it rather than
    returning NaN, so this also pins that the report survives one.
    """
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=7.0, first_feasible=10.0 * seed)]
            for seed in range(1, 6)
        },
    )
    (result,) = score(argv)
    assert not result.eligible
    assert math.isnan(result.pearson)
    assert "final objective is identical on every seed" in result.ineligible


def test_an_instance_that_always_arrives_at_the_same_point_is_ineligible(
    tmp_path: Path,
) -> None:
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=float(seed), first_feasible=3.0)]
            for seed in range(1, 6)
        },
    )
    (result,) = score(argv)
    assert not result.eligible
    assert "first feasible objective is identical on every seed" in result.ineligible


# --- the pre-registered verdict ------------------------------------------------


def _results(pearsons: list[float]) -> list[InstanceResult]:
    return [InstanceResult(f"i{n}", 8, r, r, "") for n, r in enumerate(pearsons)]


def test_too_few_eligible_instances_is_inconclusive_not_a_refutation() -> None:
    call = verdict(_results([0.99] * (MIN_ELIGIBLE_INSTANCES - 1)))
    assert call.word == "INCONCLUSIVE"
    assert "has not measured the effect" in call.detail


def test_a_strong_majority_above_the_threshold_generalises() -> None:
    call = verdict(_results([0.95] * MIN_ELIGIBLE_INSTANCES))
    assert call.word == "GENERALISES"


def test_a_high_median_without_a_majority_does_not_generalise() -> None:
    """Both halves of the rule are load-bearing, so each is pinned alone.

    Ten instances, five at 0.99 and five at 0.69: the median lands above the
    threshold (it is the mean of the two middle values, 0.84) while only half
    the instances reach it -- and "half" is not a strict majority.
    """
    call = verdict(_results([0.99] * 5 + [0.69] * 5))
    assert call.word == "DOES NOT GENERALISE"
    assert f"only 5 of 10 eligible instances reach r >= {R_DETERMINED}" in call.detail


def test_a_majority_with_a_low_median_does_not_generalise() -> None:
    call = verdict(_results([0.71] * 6 + [-0.9] * 6))
    assert call.word == "DOES NOT GENERALISE"
    assert "median r" in call.detail


def test_ineligible_instances_do_not_count_toward_the_minimum() -> None:
    """The floor is on the ELIGIBLE set, not on the rows the campaign wrote.

    A roster of flat instances must read as "not measured", not as a verdict
    assembled from instances that carry no correlation at all.
    """
    flat = [InstanceResult(f"f{n}", 8, math.nan, math.nan, "no spread") for n in range(40)]
    call = verdict([*_results([0.99] * 3), *flat])
    assert call.word == "INCONCLUSIVE"


# --- the command line ----------------------------------------------------------


def test_no_inputs_is_refused() -> None:
    assert (
        usage_error(parse_args([])) == "nothing to score: pass --table SEED=PATH or --results PATH"
    )


def test_two_tables_at_one_seed_are_refused(tmp_path: Path) -> None:
    a = write_table(tmp_path / "a.csv", [])
    b = write_table(tmp_path / "b.csv", [])
    refusal = usage_error(parse_args(["--table", f"1={a}", "--table", f"1={b}"]))
    assert refusal is not None
    assert "repeats a seed" in refusal


def test_a_missing_input_is_refused(tmp_path: Path) -> None:
    refusal = usage_error(parse_args(["--table", f"1={tmp_path / 'nope.csv'}"]))
    assert refusal is not None
    assert "no such file" in refusal


def test_a_min_seeds_below_two_is_refused(tmp_path: Path) -> None:
    path = write_table(tmp_path / "a.csv", [])
    refusal = usage_error(parse_args(["--table", f"1={path}", "--min-seeds", "1"]))
    assert refusal is not None
    assert "a correlation needs two points" in refusal


def test_the_report_names_the_reference_instance_and_states_a_verdict(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(nvs01_issue_134(tmp_path)) == 0
    text = capsys.readouterr().out
    assert "the #134 instance" in text
    assert "0.945" in text
    # One instance is far below the ten the rule needs, so the report must say
    # inconclusive rather than declare a roster-wide finding from nvs01 -- which
    # is the exact mistake #149 exists to prevent.
    assert "VERDICT: INCONCLUSIVE" in text


def test_the_per_instance_csv_records_the_ineligible_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "per_instance.csv"
    argv = seed_tables(
        tmp_path,
        {
            seed: [
                runner_row("varies", objective=float(seed), first_feasible=10.0 * seed),
                runner_row("flat", objective=7.0, first_feasible=10.0 * seed),
            ]
            for seed in range(1, 6)
        },
    )
    assert main([*argv, "--csv", str(out)]) == 0
    capsys.readouterr()
    with out.open(newline="") as fh:
        rows = {row["instance"]: row for row in csv.DictReader(fh)}
    assert rows["varies"]["pearson_r"] == "1.000000"
    assert rows["varies"]["ineligible_reason"] == ""
    assert rows["flat"]["pearson_r"] == "NaN"
    assert "identical on every seed" in rows["flat"]["ineligible_reason"]


def test_a_bad_table_spec_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--table", "no-equals-sign"])
    with pytest.raises(SystemExit):
        parse_args(["--table", "notaseed=/tmp/x.csv"])
