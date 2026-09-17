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

from benchmarks.minlplib.first_feasible_report import (
    ARRIVAL_INVARIANT,
    DETERMINED,
    FINAL_INVARIANT,
    MIN_ELIGIBLE_INSTANCES,
    NO_SPREAD,
    NOT_DETERMINED,
    R_DETERMINED,
    REFERENCE_INSTANCE,
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
from benchmarks.minlplib.runner import RUNNER_COLUMNS

if TYPE_CHECKING:
    from pathlib import Path

RUNNER_HEADER = list(RUNNER_COLUMNS)

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


#: `expected` value meaning a NaN: no correlation was computed.
NAN = "nan"


@pytest.mark.parametrize(
    ("pairs", "argv", "expected"),
    [
        (
            [(10.0 * s, float(s)) for s in range(1, 6)],
            [],
            {"pearson": 1.0, "spearman": 1.0, "bucket": DETERMINED},
        ),
        # A strong correlation of the WRONG SIGN must not count as the effect.
        # `abs(r)` would score this 1.0 and read it as "the first feasible point
        # determines the final objective", when it says the opposite: the runs that
        # arrive worst finish best. The threshold is on r, not on its magnitude.
        (
            [(10.0 * s, float(-s)) for s in range(1, 6)],
            [],
            {"pearson": -1.0, "eligible": True, "determined": False},
        ),
        (
            [(10.0 * s, float(s)) for s in range(1, 4)],
            ["--min-seeds", "4"],
            {
                "bucket": TOO_FEW_SEEDS,
                "eligible": False,
                "pearson": NAN,
                "note": "below --min-seeds 4",
            },
        ),
        # Arrival varied; the final objective did not. That is a refutation: r is
        # formally undefined (`statistics.correlation` RAISES on zero variance), but
        # the instance says the search reached the same answer however far out it
        # started. Filed with the uninformative instances, a roster full of them
        # would come back "inconclusive" when it had refuted the effect.
        (
            [(10.0 * s, 7.0) for s in range(1, 6)],
            [],
            {
                "bucket": FINAL_INVARIANT,
                "eligible": True,
                "determined": False,
                "pearson": NAN,
                "note": "every seed finished at the same objective",
            },
        ),
        # The genuinely uninformative case: neither end varied.
        (
            [(3.0, 7.0) for _ in range(1, 6)],
            [],
            {
                "bucket": NO_SPREAD,
                "eligible": False,
                "note": "neither the first feasible objective nor the final one varied",
            },
        ),
        # The mirror of FINAL_INVARIANT, and as direct a refutation: bucketing it with
        # the uninformative would drop it from the denominator and bias the verdict
        # toward GENERALISES -- the direction #149 is written to guard against.
        (
            [(3.0, float(s)) for s in range(1, 6)],
            [],
            {
                "bucket": ARRIVAL_INVARIANT,
                "eligible": True,
                "determined": False,
                "pearson": NAN,
                "note": "arrived at the same objective and still finished apart",
            },
        ),
    ],
    ids=[
        "perfectly-correlated",
        "anticorrelated",
        "too-few-seeds",
        "final-invariant",
        "nothing-moved",
        "arrival-invariant",
    ],
)
def test_an_instance_is_bucketed_by_what_its_seeds_can_say(
    tmp_path: Path, pairs: list[tuple[float, float]], argv: list[str], expected: dict[str, object]
) -> None:
    tables = seed_tables(
        tmp_path,
        {
            seed: [runner_row("a", objective=final, first_feasible=first)]
            for seed, (first, final) in enumerate(pairs, start=1)
        },
    )
    (result,) = score([*tables, *argv])
    for field, value in expected.items():
        actual = getattr(result, field)
        if field == "note":
            assert str(value) in actual
        elif value == NAN:
            assert math.isnan(actual), field
        elif isinstance(value, float):
            assert actual == pytest.approx(value), field
        else:
            assert actual == value, field


# --- which rows are usable -----------------------------------------------------


@pytest.mark.parametrize(
    ("rows", "seed_column", "found", "dropped"),
    [
        # The NaN rule, where it bites: a pre-#149 table, or a row the runner wrote
        # without solving, carries NaN in both first-feasible cells. Read as 0 it is
        # the most favourable observation possible -- "arrived instantly at zero".
        (
            [runner_row("a", objective=5.0, first_feasible="NaN", seconds="NaN")],
            False,
            [],
            {"no_first_feasible": 1},
        ),
        (
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
            False,
            [],
            {"infeasible": 1},
        ),
        # #100's witness: feasible, recorded, with no objective to correlate.
        # `time_to_first_feasible` is a number here, which is the whole reason the
        # engine keeps the two cells apart -- so this is "non-finite", not "no reading".
        (
            [runner_row("a", objective=1.0, first_feasible="inf", seconds=0.5)],
            False,
            [],
            {"non_finite": 1, "no_first_feasible": 0},
        ),
        # `elec25`/`elec50` are published as failures and excluded from every claim,
        # counted apart so a reader can see they were dropped on purpose.
        (
            [
                runner_row("elec25", objective=1.0, first_feasible=2.0),
                runner_row("a", objective=1.0, first_feasible=2.0),
            ],
            False,
            ["a"],
            {"claim_excluded": 1},
        ),
        # #153: the driver records a crashed run with `feasible=false` too, and
        # folding it into "the search found nothing" misstates campaign health.
        (
            [
                runner_row(
                    "a",
                    objective="NaN",
                    first_feasible="NaN",
                    seconds="NaN",
                    feasible=False,
                    note="runner-failed-exit-3",
                )
            ],
            True,
            [],
            {"runner_failed": 1, "infeasible": 0},
        ),
    ],
    ids=["no-first-feasible-reading", "infeasible", "non-finite", "claim-excluded", "crashed"],
)
def test_a_row_that_cannot_be_an_observation_is_dropped_and_counted(
    tmp_path: Path,
    rows: list[list[str]],
    seed_column: bool,
    found: list[str],
    dropped: dict[str, int],
) -> None:
    path = tmp_path / "t.csv"
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", *(["seed"] if seed_column else []), *RUNNER_HEADER[1:]])
        writer.writerows([[row[0], *(["1"] if seed_column else []), *row[1:]] for row in rows])
    observations, skipped = read_table(path, seed=1, arm=None)
    assert [o.instance for o in observations] == found
    for field, count in dropped.items():
        assert getattr(skipped, field) == count, field
    assert skipped.total() == len(rows) - len(found)


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


@pytest.mark.parametrize(
    ("results", "word", "detail"),
    [
        (
            _results([0.99] * (MIN_ELIGIBLE_INSTANCES - 1)),
            "INCONCLUSIVE",
            "has not measured the effect",
        ),
        (_results([0.95] * MIN_ELIGIBLE_INSTANCES), "GENERALISES", ""),
        # Both halves of the rule are load-bearing, so each is pinned alone. Five at
        # 0.99 and five at 0.69: the median (0.84) clears the threshold while only
        # half the instances do -- and half is not a strict majority.
        (
            _results([0.99] * 5 + [0.69] * 5),
            "DOES NOT GENERALISE",
            f"only 5 of 10 eligible instances reach r >= {R_DETERMINED}",
        ),
        (_results([0.71] * 6 + [-0.9] * 6), "DOES NOT GENERALISE", "median r"),
        # The floor is on the ELIGIBLE set, not on the rows the campaign wrote: a
        # roster of flat instances reads as "not measured".
        (
            [
                *_results([0.99] * 3),
                *[InstanceResult(f"f{n}", 8, math.nan, math.nan, NO_SPREAD, "") for n in range(40)],
            ],
            "INCONCLUSIVE",
            "",
        ),
        # Every eligible instance FINAL_INVARIANT: the median is NaN, and that must
        # not fall through to "inconclusive" -- the campaign measured the thing and
        # found the outcome independent of the arrival.
        (
            _final_invariant(MIN_ELIGIBLE_INSTANCES),
            "DOES NOT GENERALISE",
            "no eligible instance has a defined r",
        ),
        # Without FINAL_INVARIANT in the denominator, 6 at r = 0.99 beside 6 flat
        # would be 6 of 6 and read as a clean generalisation.
        (
            [*_results([0.99] * 6), *_final_invariant(6)],
            "DOES NOT GENERALISE",
            "only 6 of 12 eligible instances",
        ),
    ],
    ids=[
        "too-few-eligible",
        "strong-majority",
        "high-median-without-majority",
        "majority-with-low-median",
        "ineligible-do-not-count",
        "outcomes-never-move",
        "final-invariant-counts-against",
    ],
)
def test_the_pre_registered_verdict(results: list[InstanceResult], word: str, detail: str) -> None:
    call = verdict(results)
    assert call.word == word
    assert detail in call.detail


# --- the command line ----------------------------------------------------------


def _refusal_argv(tmp_path: Path, case: str) -> list[str]:
    """The command line for one of the input mistakes the report refuses by name."""
    empty = write_table(tmp_path / "a.csv", [])
    results = tmp_path / "results.csv"
    with results.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "arm", "seed", *RUNNER_HEADER[1:]])
        for arm in ("control", "no-lns"):
            for seed in range(1, 6):
                row = runner_row("a", objective=float(seed), first_feasible=10.0 * seed)
                writer.writerow([row[0], arm, str(seed), *row[1:]])
    old = tmp_path / "old.csv"
    old.write_text("instance,objective,feasible\na,1.0,true\n")
    seedless = write_table(
        tmp_path / "comparison.csv", [runner_row("a", objective=1, first_feasible=2)]
    )
    return {
        "no-inputs": [],
        "two-tables-one-seed": [
            "--table",
            f"1={empty}",
            "--table",
            f"1={write_table(tmp_path / 'b.csv', [])}",
        ],
        "missing-input": ["--table", f"1={tmp_path / 'nope.csv'}"],
        "min-seeds-below-two": ["--table", f"1={empty}", "--min-seeds", "1"],
        "results-twice": ["--results", str(results), "--results", str(results)],
        "no-first-feasible-columns": ["--table", f"1={old}"],
        "results-without-seed": ["--results", str(seedless)],
        "arm-matches-no-row": ["--results", str(results), "--arm", "contorl"],
        "registered-arm": ["--results", str(results)],
    }[case]


@pytest.mark.parametrize(
    ("case", "refusal"),
    [
        ("no-inputs", ("nothing to score: pass --table SEED=PATH or --results PATH",)),
        ("two-tables-one-seed", ("repeats a seed",)),
        ("missing-input", ("no such file",)),
        ("min-seeds-below-two", ("a correlation needs two points",)),
        # `--table` repeats are refused by seed; this is the same mistake, other flag.
        ("results-twice", ("counted twice",)),
        # A pre-#149 table does not fail -- every row lands in the drop tally and
        # the verdict comes back a well-formed INCONCLUSIVE at exit 0, which after a
        # six-hour campaign cannot be told apart from one that measured nothing.
        ("no-first-feasible-columns", ("first_feasible_objective", "time_to_first_feasible")),
        # `--results` on a per-run comparison.csv is the likeliest single mistake:
        # the campaign produces eight seedless tables and the flag is one word from
        # `--table`. Without this it dies on a raw `KeyError: 'seed'`.
        ("results-without-seed", ("no `seed` column", "--table SEED=PATH")),
        # A typo otherwise yields a complete, correctly formatted answer about nothing.
        ("arm-matches-no-row", ("matches no row", "control, no-lns")),
        ("registered-arm", None),
    ],
    ids=[
        "no-inputs",
        "two-tables-one-seed",
        "missing-input",
        "min-seeds-below-two",
        "results-twice",
        "no-first-feasible-columns",
        "results-without-seed",
        "arm-matches-no-row",
        "registered-arm",
    ],
)
def test_an_input_mistake_is_refused_by_name(
    tmp_path: Path, case: str, refusal: tuple[str, ...] | None
) -> None:
    message = usage_error(parse_args(_refusal_argv(tmp_path, case)))
    if refusal is None:
        assert message is None
    elif case == "no-inputs":
        assert message == refusal[0]  # the whole sentence: it is all the user gets
    else:
        assert message is not None and all(text in message for text in refusal), message


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
