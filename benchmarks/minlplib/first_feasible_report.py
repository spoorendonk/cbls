"""Score issue #149: does the first feasible point determine the final objective?

Issue #134 measured, on the MINLPLib instance `nvs01` alone, a Pearson
correlation of **r = 0.945** between the objective at the first feasible point a
run reaches and the objective it finishes with, over eight seeds at the
published 60s budget. #149 asks the only question that decides whether that is
an engine finding or a curiosity: **does it hold across the roster?**

This module answers it from the columns the runner now publishes
(`first_feasible_objective`, `time_to_first_feasible` -- see
`benchmarks/minlplib/minlplib.cpp`), so the campaign is an ordinary multi-seed
run rather than a bespoke instrumented one. It solves nothing itself.

Usage -- one seed per table, the seed stated on the command line:

    .venv/bin/python3 -m benchmarks.minlplib.first_feasible_report \\
        --table 1=/scratch/ff/seed1.csv --table 2=/scratch/ff/seed2.csv ...

or straight off an ablation campaign, whose `results.csv` carries the seed and
the arm on every row:

    .venv/bin/python3 -m benchmarks.minlplib.first_feasible_report \\
        --results /scratch/ablation/results.csv --arm control

The full campaign recipe -- commit, seed set, budget, and the exact commands --
is in `benchmarks/instances/minlplib/README.md`, under "Is the final objective
set by the first feasible point? (#149)".

THE DECISION RULE IS PRE-REGISTERED, and it is stated here rather than chosen
after looking at the output. The three numbers are `R_DETERMINED`,
`MIN_ELIGIBLE_INSTANCES` and `MIN_SEEDS_PER_INSTANCE` below, each with the
argument for its value beside it. A verdict read off a threshold picked once the
numbers were in would be worth nothing, which is the whole reason #149 says
"check whether the effect generalises **before changing anything**".

What this module does NOT do: it makes no engine change, recommends none, and
its verdict is an input to that decision rather than the decision. #149's fourth
acceptance criterion -- "no engine change made on the strength of one instance"
-- is the standing constraint on whatever follows.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks.minlplib.run_benchmark import CLAIM_EXCLUDED

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: Pearson r at or above which one instance counts as "its final objective is
#: determined by where it first became feasible".
#:
#: 0.7 because r**2 is then at least 0.49: the first feasible objective accounts
#: for half the across-seed variance in the final one. That is the plain reading
#: of #149's claim, and it is well below the 0.945 the one measured instance
#: produced, so the bar is not set at that instance's own result.
R_DETERMINED = 0.7

#: Instances needed before a verdict is allowed to stand at all.
#:
#: A majority over a handful of instances is one instance wide. Ten is a fifth
#: of the 50-instance roster, and it is a floor on the ELIGIBLE set -- instances
#: that reached feasibility on enough seeds AND whose outcomes actually varied
#: -- not on the roster, because a roster where nothing varies has no
#: correlation to report and should say so rather than produce a verdict.
MIN_ELIGIBLE_INSTANCES = 10

#: Seeds an instance needs before its r is computed.
#:
#: A correlation over three points is decided by one of them; over eight (the
#: seed set #134 used, and the one the documented campaign runs) it is a
#: measurement. Four is the floor rather than eight so that an instance that
#: lost a few seeds to infeasibility still contributes, and `--min-seeds` can
#: raise it for a sensitivity check.
MIN_SEEDS_PER_INSTANCE = 4

#: The instance #134 measured. Its r is printed on its own line so the campaign
#: can be checked against the result that raised the issue before its verdict is
#: believed: a roster-wide r that disagrees with 0.945 HERE is a bug in the
#: measurement, not a finding about the roster.
REFERENCE_INSTANCE = "nvs01"

#: #134's Pearson r on `REFERENCE_INSTANCE`, at engine commit 09097de, eight
#: seeds, 60s. Quoted for comparison only -- nothing branches on it.
REFERENCE_R = 0.945


@dataclass(frozen=True)
class Observation:
    """One solve: where it arrived in the feasible region, and where it ended."""

    instance: str
    seed: int
    first_feasible: float
    final: float
    seconds_to_first_feasible: float


@dataclass(frozen=True)
class Skipped:
    """Rows that carried no usable (first feasible, final) pair, by reason.

    Counted and reported rather than silently dropped: a correlation computed
    over a quarter of the roster is a different claim from one computed over all
    of it, and the reader cannot tell which they have without these.
    """

    infeasible: int = 0
    no_first_feasible: int = 0
    non_finite: int = 0
    claim_excluded: int = 0

    def total(self) -> int:
        return self.infeasible + self.no_first_feasible + self.non_finite + self.claim_excluded


@dataclass(frozen=True)
class InstanceResult:
    """One instance's across-seed correlation, or the reason there is none."""

    instance: str
    seeds: int
    #: Pearson r, or NaN when the instance is not eligible for one.
    pearson: float
    #: Spearman rank correlation, or NaN. Reported beside Pearson as a
    #: robustness check: Pearson is the statistic #134 used and the one #149
    #: asks about, but it is sensitive to a single far-out seed, which is
    #: exactly the shape these runs produce.
    spearman: float
    #: Why `pearson` is NaN, or "" when it is a number.
    ineligible: str

    @property
    def eligible(self) -> bool:
        return not self.ineligible

    @property
    def determined(self) -> bool:
        return self.eligible and self.pearson >= R_DETERMINED


def _number(text: str | None) -> float:
    """A CSV cell as a float, with anything unparseable reading as NaN.

    Same rule as `ablation_report._number`: "NaN" is what the runner writes for
    a cell it has no value for, an empty cell means the same, and neither may
    become a 0 that an aggregate would treat as a measurement.
    """
    try:
        return float(text) if text is not None and text.strip() else math.nan
    except ValueError:
        return math.nan


def read_table(path: Path, seed: int | None, arm: str | None) -> tuple[list[Observation], Skipped]:
    """Observations from one results table, and a tally of the rows it dropped.

    `seed` is the seed every row belongs to, for a table that does not record
    one (the driver's per-run `comparison.csv`); pass None for a campaign
    `results.csv`, which carries `seed` -- and `arm`, which `arm` then filters
    on so the control rows are not averaged with an ablation arm's.
    """
    observations: list[Observation] = []
    infeasible = no_first = non_finite = excluded = 0
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if arm is not None and "arm" in row and row["arm"] != arm:
                continue
            instance = row["instance"]
            if instance in CLAIM_EXCLUDED:
                excluded += 1
                continue
            if row.get("feasible") != "true":
                infeasible += 1
                continue
            first = _number(row.get("first_feasible_objective"))
            seconds = _number(row.get("time_to_first_feasible"))
            final = _number(row.get("objective"))
            if math.isnan(seconds):
                # The authoritative "was a feasible point recorded" cell; see
                # `SearchResult::first_feasible_objective`. A row that is
                # feasible and has no time is one written by an engine older
                # than #149 -- a pre-#149 table, in other words, which must not
                # be scored as though it had produced no correlation.
                no_first += 1
                continue
            if not (math.isfinite(first) and math.isfinite(final)):
                # The #100 witness: feasible, but with no objective value at the
                # point of arrival, so there is nothing to correlate.
                non_finite += 1
                continue
            observations.append(
                Observation(
                    instance=instance,
                    seed=int(row["seed"]) if seed is None else seed,
                    first_feasible=first,
                    final=final,
                    seconds_to_first_feasible=seconds,
                )
            )
    return observations, Skipped(infeasible, no_first, non_finite, excluded)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r, or NaN when either side has no spread.

    `statistics.correlation` raises on a constant input rather than returning
    NaN, and a constant input is the normal case here: an instance the search
    solves to the same objective on every seed has nothing to explain, and must
    be reported as such rather than crash the report or be counted as r = 0.
    """
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return math.nan
    return statistics.correlation(xs, ys)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r over the ranks -- ties averaged, as the usual definition has it."""
    if len(xs) < 2 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return math.nan
    return _pearson(_ranks(xs), _ranks(ys))


def _ranks(values: Sequence[float]) -> list[float]:
    """Fractional ranks, averaging ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        stop = index
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
            stop += 1
        shared = (index + stop) / 2.0 + 1.0
        for position in range(index, stop + 1):
            ranks[order[position]] = shared
        index = stop + 1
    return ranks


def score_instance(
    instance: str, observations: Sequence[Observation], min_seeds: int
) -> InstanceResult:
    """One instance's correlation across the seeds that produced a usable pair."""
    seeds = sorted({o.seed for o in observations})
    if len(seeds) < min_seeds:
        return InstanceResult(
            instance,
            len(seeds),
            math.nan,
            math.nan,
            f"only {len(seeds)} usable seed(s), below --min-seeds {min_seeds}",
        )
    # One observation per seed, so a table accidentally supplied twice cannot
    # weight a seed double and shrink the spread the correlation is computed on.
    by_seed = {o.seed: o for o in observations}
    firsts = [by_seed[s].first_feasible for s in seeds]
    finals = [by_seed[s].final for s in seeds]
    pearson = _pearson(firsts, finals)
    if math.isnan(pearson):
        reason = (
            "the final objective is identical on every seed"
            if len(set(finals)) < 2
            else "the first feasible objective is identical on every seed"
        )
        return InstanceResult(instance, len(seeds), math.nan, math.nan, reason)
    return InstanceResult(instance, len(seeds), pearson, _spearman(firsts, finals), "")


def group(observations: Iterable[Observation]) -> dict[str, list[Observation]]:
    grouped: dict[str, list[Observation]] = {}
    for observation in observations:
        grouped.setdefault(observation.instance, []).append(observation)
    return grouped


@dataclass(frozen=True)
class Verdict:
    """The pre-registered rule's answer, and the sentence that states it."""

    word: str
    detail: str


def verdict(results: Sequence[InstanceResult]) -> Verdict:
    """Apply the pre-registered rule to the scored instances.

    Three outcomes, and "inconclusive" is a real one: a campaign that produced
    too few eligible instances has not measured the thing, and saying so is
    the honest report. It is NOT the same as "does not generalise".
    """
    eligible = [r for r in results if r.eligible]
    if len(eligible) < MIN_ELIGIBLE_INSTANCES:
        return Verdict(
            "INCONCLUSIVE",
            f"{len(eligible)} eligible instance(s), below the {MIN_ELIGIBLE_INSTANCES} the rule "
            "requires; the campaign has not measured the effect either way",
        )
    determined = [r for r in eligible if r.determined]
    median = statistics.median(r.pearson for r in eligible)
    majority = len(determined) * 2 > len(eligible)
    if median >= R_DETERMINED and majority:
        return Verdict(
            "GENERALISES",
            f"median r = {median:.3f} (>= {R_DETERMINED}) and {len(determined)} of "
            f"{len(eligible)} eligible instances are at or above it",
        )
    why = []
    if median < R_DETERMINED:
        why.append(f"median r = {median:.3f} is below {R_DETERMINED}")
    if not majority:
        why.append(
            f"only {len(determined)} of {len(eligible)} eligible instances reach r >= "
            f"{R_DETERMINED}"
        )
    return Verdict("DOES NOT GENERALISE", "; ".join(why))


def render(results: Sequence[InstanceResult], skipped: Skipped, rows: int, min_seeds: int) -> str:
    """The whole report, as text."""
    eligible = [r for r in results if r.eligible]
    lines = [
        "=== #149: first feasible objective vs final objective ===",
        "",
        f"rows read:              {rows}",
        f"rows dropped:           {skipped.total()}"
        f"  (infeasible {skipped.infeasible}, no first-feasible reading "
        f"{skipped.no_first_feasible}, non-finite objective {skipped.non_finite}, "
        f"excluded from claims {skipped.claim_excluded})",
        f"instances scored:       {len(results)}",
        f"instances eligible:     {len(eligible)}  "
        f"(>= {min_seeds} usable seeds and a spread in both columns)",
        "",
        f"  {'instance':<22} {'seeds':>5} {'pearson r':>10} {'spearman':>9}  note",
    ]
    for result in sorted(
        results, key=lambda r: (not r.eligible, -r.pearson if r.eligible else 0.0)
    ):
        r_cell = "-" if math.isnan(result.pearson) else f"{result.pearson:10.3f}"
        rho_cell = "-" if math.isnan(result.spearman) else f"{result.spearman:9.3f}"
        lines.append(
            f"  {result.instance:<22} {result.seeds:>5} {r_cell:>10} {rho_cell:>9}  "
            f"{result.ineligible}"
        )
    lines.append("")
    reference = next((r for r in results if r.instance == REFERENCE_INSTANCE), None)
    if reference is None:
        lines.append(
            f"{REFERENCE_INSTANCE}: not in this run -- #134's r = {REFERENCE_R} cannot be "
            "cross-checked, so read the verdict with that caveat"
        )
    elif reference.eligible:
        lines.append(
            f"{REFERENCE_INSTANCE} (the #134 instance): r = {reference.pearson:.3f} here "
            f"against {REFERENCE_R} in #134, over {reference.seeds} seed(s)"
        )
    else:
        lines.append(
            f"{REFERENCE_INSTANCE} (the #134 instance): no r this time -- {reference.ineligible}"
        )
    if eligible:
        lines += [
            f"median r over eligible: {statistics.median(r.pearson for r in eligible):.3f}",
            f"instances at r >= {R_DETERMINED}:  "
            f"{sum(1 for r in eligible if r.determined)} of {len(eligible)}",
        ]
    call = verdict(results)
    lines += [
        "",
        f"VERDICT: {call.word}",
        f"  {call.detail}",
        "",
        "  Rule, fixed before the campaign: an instance counts when it has at least",
        f"  {min_seeds} usable seeds and its outcomes vary; the effect GENERALISES when the",
        f"  median Pearson r over those instances is >= {R_DETERMINED} and a strict majority of",
        f"  them reach it; fewer than {MIN_ELIGIBLE_INSTANCES} such instances is INCONCLUSIVE.",
        "",
        "  A verdict is not a licence to change the engine. #149's fourth criterion",
        "  stands: any change needs its own issue, hypothesis and regression test.",
    ]
    return "\n".join(lines)


def write_csv(path: Path, results: Sequence[InstanceResult]) -> None:
    """The per-instance numbers, for a reader who wants to re-plot them."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "seeds", "pearson_r", "spearman_rho", "ineligible_reason"])
        for result in sorted(results, key=lambda r: r.instance):
            writer.writerow(
                [
                    result.instance,
                    result.seeds,
                    "NaN" if math.isnan(result.pearson) else f"{result.pearson:.6f}",
                    "NaN" if math.isnan(result.spearman) else f"{result.spearman:.6f}",
                    result.ineligible,
                ]
            )


def parse_table(spec: str) -> tuple[int, Path]:
    """`--table SEED=PATH`, the form for a table that records no seed of its own."""
    seed, sep, path = spec.partition("=")
    if not sep or not path:
        raise argparse.ArgumentTypeError(f"--table wants SEED=PATH (got {spec!r})")
    try:
        return int(seed), Path(path)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--table seed must be an integer (got {seed!r})"
        ) from None


def collect(args: argparse.Namespace) -> tuple[list[Observation], Skipped, int]:
    """Every observation the inputs carry, the drop tally, and the rows read."""
    observations: list[Observation] = []
    infeasible = no_first = non_finite = excluded = 0
    rows = 0
    sources: list[tuple[int | None, Path]] = [(None, p) for p in args.results]
    sources += [(seed, path) for seed, path in args.table]
    for seed, path in sources:
        found, dropped = read_table(path, seed, args.arm)
        observations += found
        infeasible += dropped.infeasible
        no_first += dropped.no_first_feasible
        non_finite += dropped.non_finite
        excluded += dropped.claim_excluded
        rows += len(found) + dropped.total()
    return observations, Skipped(infeasible, no_first, non_finite, excluded), rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--table",
        type=parse_table,
        action="append",
        default=[],
        metavar="SEED=PATH",
        help="a results table with no seed column; repeat once per seed",
    )
    parser.add_argument(
        "--results",
        type=Path,
        action="append",
        default=[],
        help="a campaign results.csv, which carries seed and arm per row",
    )
    parser.add_argument(
        "--arm",
        default="control",
        help="arm to score in a --results file (default: control); ignored by --table",
    )
    parser.add_argument("--min-seeds", type=int, default=MIN_SEEDS_PER_INSTANCE)
    parser.add_argument("--csv", type=Path, default=None, help="also write the per-instance table")
    return parser.parse_args(argv)


def usage_error(args: argparse.Namespace) -> str | None:
    """The reason to reject the argument combination outright, or None."""
    if not args.table and not args.results:
        return "nothing to score: pass --table SEED=PATH or --results PATH"
    if args.min_seeds < 2:
        return f"--min-seeds must be >= 2 (got {args.min_seeds}); a correlation needs two points"
    seeds = [seed for seed, _ in args.table]
    if len(set(seeds)) != len(seeds):
        return (
            f"--table repeats a seed ({sorted(seeds)}); two tables at one seed are two draws of "
            "the same run and would weight it double"
        )
    missing = [str(p) for _, p in args.table] + [str(p) for p in args.results]
    absent = [p for p in missing if not Path(p).exists()]
    if absent:
        return f"no such file: {', '.join(absent)}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    refusal = usage_error(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    observations, skipped, rows = collect(args)
    results = [
        score_instance(instance, found, args.min_seeds)
        for instance, found in sorted(group(observations).items())
    ]
    if args.csv is not None:
        write_csv(args.csv, results)
    print(render(results, skipped, rows, args.min_seeds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
