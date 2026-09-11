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
import statistics
from typing import TYPE_CHECKING

import pytest

from benchmarks.minlplib import ablation_report
from benchmarks.minlplib.first_feasible_report import (
    ARRIVAL_INVARIANT,
    DETERMINED,
    FINAL_INVARIANT,
    MIN_ELIGIBLE_INSTANCES,
    NO_SPREAD,
    NOT_DETERMINED,
    R_DETERMINED,
    REFERENCE_INSTANCE,
    RUNNER_FAILED_NOTE,
    TOO_FEW_SEEDS,
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
    # Deliberately NOT a perfect line: on a perfect line a repeated point leaves
    # r at 1.0, so the test would pass with the dedup removed. `seeds` is a set
    # either way, so `pearson` against the five intended pairs is what actually
    # pins the behaviour.
    firsts = [10.0, 20.0, 30.0, 40.0, 100.0]
    finals = [1.0, 2.0, 3.0, 4.0, 4.5]
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "arm", "seed", *RUNNER_HEADER[1:]])
        for seed, (first, final) in enumerate(zip(firsts, finals, strict=True), start=1):
            row = runner_row("a", objective=final, first_feasible=first)
            writer.writerow([row[0], "control", str(seed), *row[1:]])
        # An exact copy of seed 1's row, which is what a re-run without
        # `--no-resume` leaves behind. Last-wins, so the pair it carries is
        # unchanged -- only its WEIGHT would change.
        duplicate = runner_row("a", objective=finals[0], first_feasible=firsts[0])
        writer.writerow([duplicate[0], "control", "1", *duplicate[1:]])
    (result,) = score(["--results", str(path), "--arm", "control"])
    assert result.seeds == 5
    assert result.pearson == pytest.approx(statistics.correlation(firsts, finals))


def test_a_crashed_run_is_not_tallied_as_a_search_that_found_nothing(tmp_path: Path) -> None:
    """#153 made the runner exit 3 on a throw; the driver records those rows too.

    Both carry `feasible=false`, so folding them together makes the tally a
    reader uses to judge campaign health say "the search found nothing" where
    the truth is "the process died".
    """
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "seed", *RUNNER_HEADER[1:]])
        crashed = runner_row(
            "a",
            objective="NaN",
            first_feasible="NaN",
            seconds="NaN",
            feasible=False,
            note="runner-failed-exit-3",
        )
        writer.writerow([crashed[0], "1", *crashed[1:]])
    _, skipped = read_table(path, seed=1, arm=None)

    assert skipped.runner_failed == 1
    assert skipped.infeasible == 0
    assert skipped.total() == 1


def test_the_runner_failed_prefix_matches_the_scorer_s(tmp_path: Path) -> None:
    """Spelled in two modules because one of them is run as a script."""
    assert RUNNER_FAILED_NOTE == ablation_report.RUNNER_FAILED_NOTE


def test_a_table_without_the_first_feasible_columns_is_refused_by_name(tmp_path: Path) -> None:
    """A pre-#149 table does not fail -- it returns a clean INCONCLUSIVE.

    Every row lands in the drop tally and the verdict comes back well-formed at
    exit 0, which after a six-hour campaign cannot be told apart from a campaign
    that genuinely measured nothing. So the missing column is named instead.
    """
    path = tmp_path / "old.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "objective", "feasible"])
        writer.writerow(["a", "1.0", "true"])
    refusal = usage_error(parse_args(["--table", f"1={path}"]))
    if refusal is None:
        pytest.fail("a table with no first-feasible columns was accepted")
    assert "first_feasible_objective" in refusal
    assert "time_to_first_feasible" in refusal


def test_the_same_results_file_passed_twice_is_refused(tmp_path: Path) -> None:
    """`--table` repeats are refused by seed; this is the same mistake, other flag."""
    path = tmp_path / "results.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "seed", *RUNNER_HEADER[1:]])
        row = runner_row("a", objective=1.0, first_feasible=10.0)
        writer.writerow([row[0], "1", *row[1:]])
    refusal = usage_error(parse_args(["--results", str(path), "--results", str(path)]))
    if refusal is None:
        pytest.fail("the same results file was accepted twice")
    assert "counted twice" in refusal


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
    assert result.bucket == TOO_FEW_SEEDS
    assert math.isnan(result.pearson)
    assert "below --min-seeds 4" in result.note


def test_an_instance_whose_outcome_never_moves_is_evidence_against_the_effect(
    tmp_path: Path,
) -> None:
    """Arrival varied; the final objective did not. That is a refutation.

    Pearson r is formally undefined here -- zero variance in the final objective,
    and `statistics.correlation` RAISES on it rather than returning NaN -- but the
    instance is not silent: it says the search reached the same answer however far
    out it started, which is exactly the opposite of #149's claim. Filing it with
    the uninformative instances would let a roster full of them come back
    "inconclusive" when it had actually refuted the effect, so it is eligible and
    counts as not-determined.
    """
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=7.0, first_feasible=10.0 * seed)]
            for seed in range(1, 6)
        },
    )
    (result,) = score(argv)
    assert result.bucket == FINAL_INVARIANT
    assert result.eligible
    assert not result.determined
    assert math.isnan(result.pearson)
    assert "every seed finished at the same objective" in result.note


def test_an_instance_where_nothing_moved_at_all_is_ineligible(tmp_path: Path) -> None:
    """The genuinely uninformative case: neither end varied.

    Distinguished from the two invariant buckets on purpose. Here nothing can be
    concluded; there, something can.
    """
    argv = seed_tables(
        tmp_path,
        {seed: [runner_row("a", objective=7.0, first_feasible=3.0)] for seed in range(1, 6)},
    )
    (result,) = score(argv)
    assert result.bucket == NO_SPREAD
    assert not result.eligible
    assert "neither the first feasible objective nor the final one varied" in result.note


def test_an_instance_that_always_arrives_at_the_same_point_but_finishes_apart(
    tmp_path: Path,
) -> None:
    """The mirror of FINAL_INVARIANT, and a refutation just as directly.

    Every seed arrived at the same objective and they still finished apart, so
    the arrival explains none of the outcome's variance. Bucketing this with the
    uninformative instances would drop it from the denominator and bias the
    verdict toward GENERALISES -- the one direction #149 is written to guard
    against, since a false positive there buys engine work on a premise that was
    never there.
    """
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=float(seed), first_feasible=3.0)]
            for seed in range(1, 6)
        },
    )
    (result,) = score(argv)
    assert result.bucket == ARRIVAL_INVARIANT
    assert result.eligible
    assert not result.determined
    assert math.isnan(result.pearson)
    assert "arrived at the same objective and still finished apart" in result.note


# --- the pre-registered verdict ------------------------------------------------


def _results(pearsons: list[float]) -> list[InstanceResult]:
    return [
        InstanceResult(f"i{n}", 8, r, r, DETERMINED if r >= R_DETERMINED else NOT_DETERMINED, "")
        for n, r in enumerate(pearsons)
    ]


def _final_invariant(count: int) -> list[InstanceResult]:
    return [
        InstanceResult(f"flat{n}", 8, math.nan, math.nan, FINAL_INVARIANT, "flat")
        for n in range(count)
    ]


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
    flat = [
        InstanceResult(f"f{n}", 8, math.nan, math.nan, NO_SPREAD, "no spread") for n in range(40)
    ]
    call = verdict([*_results([0.99] * 3), *flat])
    assert call.word == "INCONCLUSIVE"


def test_a_roster_whose_outcomes_never_move_refutes_rather_than_abstains() -> None:
    """Every eligible instance FINAL_INVARIANT: no r anywhere, and that is an answer.

    The median is NaN here, so the rule's first half cannot be evaluated -- and it
    must not fall through to "inconclusive", because the campaign did measure the
    thing and found the outcome independent of the arrival.
    """
    call = verdict(_final_invariant(MIN_ELIGIBLE_INSTANCES))
    assert call.word == "DOES NOT GENERALISE"
    assert "no eligible instance has a defined r" in call.detail


def test_final_invariant_instances_count_against_the_majority() -> None:
    """Six instances at r = 0.99 and six flat ones is not a majority.

    Without the FINAL_INVARIANT bucket in the denominator this would be 6 of 6
    and would read as a clean generalisation.
    """
    call = verdict([*_results([0.99] * 6), *_final_invariant(6)])
    assert call.word == "DOES NOT GENERALISE"
    assert "only 6 of 12 eligible instances" in call.detail


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


def test_the_per_instance_csv_records_the_bucket_and_its_reason(
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
    assert rows["varies"]["bucket"] == DETERMINED
    assert rows["varies"]["note"] == ""
    assert rows["flat"]["pearson_r"] == "NaN"
    assert rows["flat"]["bucket"] == FINAL_INVARIANT
    assert "same objective" in rows["flat"]["note"]


def test_a_bad_table_spec_is_rejected_by_the_parser() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--table", "no-equals-sign"])
    with pytest.raises(SystemExit):
        parse_args(["--table", "notaseed=/tmp/x.csv"])


def test_a_pearson_carried_by_one_outlying_seed_is_not_determined(tmp_path: Path) -> None:
    """The Spearman half of the threshold, which is what makes it load-bearing.

    Seven seeds tied at the published optimum and one far out is the common
    shape on this roster -- it is `nvs01`'s own shape, less extreme. Pearson over
    that is carried entirely by the outlying seed; the rank correlation is not.
    Requiring both at the threshold is what stops a `DETERMINED` count being
    built from one-point correlations.
    """
    finals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 500.0]
    # Arrivals deliberately in an order that disagrees with the finals' ranking
    # among the tied seeds, so Spearman is well below Pearson.
    firsts = [40.0, 10.0, 30.0, 20.0, 60.0, 50.0, 70.0, 900.0]
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=final, first_feasible=first)]
            for seed, (first, final) in enumerate(zip(firsts, finals, strict=True), start=1)
        },
    )
    (result,) = score(argv)
    # The premise: Pearson alone would have called this determined.
    assert result.pearson >= R_DETERMINED
    assert result.spearman < R_DETERMINED
    assert result.bucket == NOT_DETERMINED
    assert result.eligible


def test_a_results_file_without_a_seed_column_is_refused(tmp_path: Path) -> None:
    """`--results` on a per-run `comparison.csv` is the likeliest single mistake.

    The documented campaign produces eight seedless tables and `--results` is one
    word from `--table`. Without this it dies on a raw `KeyError: 'seed'`, when
    every other input mistake gets a clean refusal.
    """
    path = write_table(
        tmp_path / "comparison.csv", [runner_row("a", objective=1, first_feasible=2)]
    )
    refusal = usage_error(parse_args(["--results", str(path)]))
    assert refusal is not None
    assert "no `seed` column" in refusal
    assert "--table SEED=PATH" in refusal


def _campaign_results(path: Path, arms: list[str]) -> Path:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "arm", "seed", *RUNNER_HEADER[1:]])
        for arm in arms:
            for seed in range(1, 6):
                row = runner_row("a", objective=float(seed), first_feasible=10.0 * seed)
                writer.writerow([row[0], arm, str(seed), *row[1:]])
    return path


def test_an_arm_that_matches_no_row_is_refused(tmp_path: Path) -> None:
    """Otherwise a typo yields a well-formed INCONCLUSIVE.

    After six hours of solving, a verdict indistinguishable from a real one is
    the worst available failure mode: the arm filter drops every row, and the
    report prints a complete, correctly formatted answer about nothing.
    """
    path = _campaign_results(tmp_path / "results.csv", ["control", "no-lns"])
    refusal = usage_error(parse_args(["--results", str(path), "--arm", "contorl"]))
    assert refusal is not None
    assert "matches no row" in refusal
    assert "control, no-lns" in refusal


def test_the_registered_arm_is_accepted(tmp_path: Path) -> None:
    path = _campaign_results(tmp_path / "results.csv", ["control"])
    assert usage_error(parse_args(["--results", str(path)])) is None


def test_an_overridden_min_seeds_is_flagged_as_not_the_registered_rule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--min-seeds` must not silently rewrite the block headed "fixed before".

    `usage_error` allows anything >= 2, so a sensitivity check at 2 would
    otherwise print a report that reads exactly like the campaign's verdict at a
    threshold nobody registered.
    """
    argv = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=float(seed), first_feasible=10.0 * seed)]
            for seed in range(1, 4)
        },
    )
    assert main([*argv, "--min-seeds", "3"]) == 0
    text = capsys.readouterr().out
    assert "NOT THE REGISTERED RULE" in text
    assert "overrides the pre-registered" in text

    assert main(argv) == 0
    assert "NOT THE REGISTERED RULE" not in capsys.readouterr().out
