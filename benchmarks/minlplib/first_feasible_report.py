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

    .venv/bin/python3 benchmarks/minlplib/first_feasible_report.py \\
        --table 1=/scratch/ff/seed1.csv --table 2=/scratch/ff/seed2.csv ...

or straight off an ablation campaign, whose `results.csv` carries the seed and
the arm on every row:

    .venv/bin/python3 benchmarks/minlplib/first_feasible_report.py \\
        --results /scratch/ablation/results.csv --arm control

By path rather than `-m`, deliberately: the `sys.path` shim below makes the
by-path form work from any directory, while `-m benchmarks.minlplib....` needs
the repo root to be the working directory before Python can find the package at
all. Both forms work from the repo root.

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
from statistics import StatisticsError
from typing import TYPE_CHECKING

# Same shim `run_ablation.py` carries, and for the same reason: this module
# imports its siblings by package path, so without the repo root on sys.path it
# runs only from the repo root and only via `-m`. An operator who has just spent
# seven hours on a campaign should not lose the scoring step to a
# ModuleNotFoundError over which directory they are standing in.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.minlplib.ablation_report import CONTROL_ARM  # noqa: E402
from benchmarks.minlplib.run_benchmark import CLAIM_EXCLUDED  # noqa: E402

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

    #: Rows whose `feasible` cell is not "true". Wider than its name: the runner
    #: writes `feasible=false` on a verify-failed row and on its `NONFIN` row
    #: too, both of which DID reach feasibility. Right to drop either way --
    #: neither publishes an objective we stand behind -- but do not read this
    #: count as "runs that never found a feasible point".
    infeasible: int = 0
    #: Feasible rows with no `time_to_first_feasible` -- a table written by an
    #: engine older than #149, or a row the runner wrote without solving.
    no_first_feasible: int = 0
    non_finite: int = 0
    claim_excluded: int = 0

    def total(self) -> int:
        return self.infeasible + self.no_first_feasible + self.non_finite + self.claim_excluded


#: One instance's verdict, and the whole reason this is a four-way bucket rather
#: than "has an r / does not".
#:
#: `FINAL_INVARIANT` is the bucket that matters. An instance whose runs ARRIVED
#: at different objectives and all FINISHED at the same one has a Pearson r that
#: is formally undefined -- zero variance in y -- but it is not silent about
#: #149's question: it says the effect is absent there, and loudly. Filing it
#: with the genuinely uninformative instances would let a roster full of them
#: come back "inconclusive" when what it actually showed was a refutation. So it
#: is eligible, and it counts as not-determined.
#:
#: `ARRIVAL_INVARIANT` is its mirror and gets the same treatment for the same
#: reason: every seed arrived at the SAME objective and they still finished
#: apart, so the arrival explains none of the outcome's variance. Excluding it
#: would bias the verdict toward GENERALISES, which is the one direction #149 is
#: written to guard against.
#:
#: `NO_SPREAD` is the genuinely uninformative one: nothing moved at either end,
#: so the instance cannot speak either way.
DETERMINED = "determined"
NOT_DETERMINED = "not-determined"
FINAL_INVARIANT = "final-invariant"
ARRIVAL_INVARIANT = "arrival-invariant"
NO_SPREAD = "no-spread"
TOO_FEW_SEEDS = "too-few-seeds"

#: The buckets that count toward the majority test and the eligible floor.
ELIGIBLE_BUCKETS = (DETERMINED, NOT_DETERMINED, FINAL_INVARIANT, ARRIVAL_INVARIANT)

#: "Varied" here means "differ in the cell the runner published", which is six
#: significant digits (`cell()` streams a double through a default-precision
#: `std::ostringstream`). So an instance whose eight seeds finish within ~1e-6
#: relative of each other reads as FINAL_INVARIANT. That is the intended
#: reading -- a 1e-6 spread is not a descent -- but it is a property of the
#: table's precision, not of the engine, and it belongs in any write-up of the
#: result.


@dataclass(frozen=True)
class InstanceResult:
    """One instance's across-seed correlation, or the reason there is none."""

    instance: str
    seeds: int
    #: Pearson r, or NaN where the instance has no defined one -- which includes
    #: the eligible `FINAL_INVARIANT` bucket, so NaN here does NOT mean "not
    #: eligible". Read `bucket`.
    pearson: float
    #: Spearman rank correlation, or NaN. Reported beside Pearson as a
    #: robustness check: Pearson is the statistic #134 used and the one #149
    #: asks about, but it is sensitive to a single far-out seed, which is
    #: exactly the shape these runs produce.
    spearman: float
    #: One of the five constants above.
    bucket: str
    #: What the bucket means for this instance, in words; "" for a plain r.
    note: str

    @property
    def eligible(self) -> bool:
        return self.bucket in ELIGIBLE_BUCKETS

    @property
    def determined(self) -> bool:
        return self.bucket == DETERMINED


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
    try:
        return statistics.correlation(xs, ys)
    except StatisticsError:
        # The set-based guard above is about DISTINCT values; this is about the
        # deviations. At ~1e-200 they underflow to zero and `correlation` calls
        # the input constant, and at ~1e200 the sums of squares overflow and it
        # returns NaN instead. Neither magnitude occurs in this roster, but a
        # traceback six hours into a campaign is the wrong way to find that out.
        return math.nan


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
            TOO_FEW_SEEDS,
            f"only {len(seeds)} usable seed(s), below --min-seeds {min_seeds}",
        )
    # One observation per seed, so a table accidentally supplied twice cannot
    # weight a seed double and shrink the spread the correlation is computed on.
    by_seed = {o.seed: o for o in observations}
    firsts = [by_seed[s].first_feasible for s in seeds]
    finals = [by_seed[s].final for s in seeds]
    if len(set(firsts)) < 2:
        if len(set(finals)) < 2:
            # Nothing moved at either end: the experiment never varied its input
            # and the outcome never varied either, so this instance cannot speak
            # to #149 in either direction.
            return InstanceResult(
                instance,
                len(seeds),
                math.nan,
                math.nan,
                NO_SPREAD,
                "neither the first feasible objective nor the final one varied",
            )
        # Arrival was identical on every seed and the outcome still moved, so
        # arrival explains none of the outcome's variance -- the mirror of
        # FINAL_INVARIANT, and a refutation just as directly.
        return InstanceResult(
            instance,
            len(seeds),
            math.nan,
            math.nan,
            ARRIVAL_INVARIANT,
            "every seed arrived at the same objective and still finished apart",
        )
    if len(set(finals)) < 2:
        # Arrival varied; the outcome did not. No r exists, but the reading is
        # unambiguous -- see FINAL_INVARIANT above.
        return InstanceResult(
            instance,
            len(seeds),
            math.nan,
            math.nan,
            FINAL_INVARIANT,
            "arrival varied but every seed finished at the same objective",
        )
    pearson = _pearson(firsts, finals)
    if math.isnan(pearson):
        # Both sides varied, so the guards above passed, and still no r came
        # back -- the deviations overflowed or underflowed. Bucketed explicitly
        # rather than left to fall through the `>=` below (NaN compares false),
        # because that route produces the one row in the table with no r AND no
        # reason given for its absence.
        return InstanceResult(
            instance,
            len(seeds),
            math.nan,
            math.nan,
            NOT_DETERMINED,
            "no correlation is computable; these objectives over/underflow it",
        )
    spearman = _spearman(firsts, finals)
    # BOTH statistics, not Pearson alone. On this roster the common shape is
    # several seeds tied at the published optimum and one or two elsewhere, and
    # a Pearson r over that is carried by the outlying seed -- which is exactly
    # what a rank correlation is robust to. Requiring both at the same threshold
    # makes the Spearman column load-bearing instead of decorative, and it
    # tightens the rule in the conservative direction: #149's cost of a false
    # GENERALISES is engine work on a premise that was never there.
    bucket = DETERMINED if pearson >= R_DETERMINED and spearman >= R_DETERMINED else NOT_DETERMINED
    return InstanceResult(instance, len(seeds), pearson, spearman, bucket, "")


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


def median_r(results: Sequence[InstanceResult]) -> float:
    """Median Pearson r over the eligible instances that HAVE one, or NaN.

    `FINAL_INVARIANT` instances are eligible and have no r, so they are absent
    here and present in the majority test. That asymmetry is deliberate: the
    median is a descriptive statistic about the instances where a correlation
    exists, while the majority test is the rule, and it is the majority test
    that must see an instance whose outcome did not move.
    """
    values = [r.pearson for r in results if r.eligible and not math.isnan(r.pearson)]
    return statistics.median(values) if values else math.nan


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
    median = median_r(results)
    majority = len(determined) * 2 > len(eligible)
    if median >= R_DETERMINED and majority:
        return Verdict(
            "GENERALISES",
            f"median r = {median:.3f} (>= {R_DETERMINED}) and {len(determined)} of "
            f"{len(eligible)} eligible instances are at or above it",
        )
    why = []
    if math.isnan(median):
        # No eligible instance has a defined r: every one of them is
        # FINAL_INVARIANT or ARRIVAL_INVARIANT, i.e. one end held still. Both
        # are refutations, so this is an answer, not a missing measurement.
        why.append(
            f"no eligible instance has a defined r -- on all {len(eligible)} either the arrival "
            "or the outcome held still across seeds"
        )
    elif median < R_DETERMINED:
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
    buckets = {bucket: sum(1 for r in results if r.bucket == bucket) for bucket in _BUCKET_ORDER}
    lines = [
        "=== #149: first feasible objective vs final objective ===",
        "",
        f"rows considered:        {rows}  (rows filtered out by --arm are not counted)",
        f"rows dropped:           {skipped.total()}"
        f"  (infeasible {skipped.infeasible}, no first-feasible reading "
        f"{skipped.no_first_feasible}, non-finite objective {skipped.non_finite}, "
        f"excluded from claims {skipped.claim_excluded})",
        f"instances scored:       {len(results)}",
        f"instances eligible:     {len(eligible)}  (>= {min_seeds} usable seeds and a spread "
        "at one end or the other)",
        "  " + ", ".join(f"{bucket} {count}" for bucket, count in buckets.items()),
        "",
        f"  {'instance':<22} {'seeds':>5} {'pearson r':>10} {'spearman':>9}  {'bucket':<18} note",
    ]
    for result in sorted(results, key=_report_order):
        r_cell = "-" if math.isnan(result.pearson) else f"{result.pearson:10.3f}"
        rho_cell = "-" if math.isnan(result.spearman) else f"{result.spearman:9.3f}"
        lines.append(
            f"  {result.instance:<22} {result.seeds:>5} {r_cell:>10} {rho_cell:>9}  "
            f"{result.bucket:<18} {result.note}"
        )
    lines.append("")
    reference = next((r for r in results if r.instance == REFERENCE_INSTANCE), None)
    if reference is None:
        lines.append(
            f"{REFERENCE_INSTANCE}: not in this run -- #134's r = {REFERENCE_R} cannot be "
            "cross-checked, so read the verdict with that caveat"
        )
    elif math.isnan(reference.pearson):
        lines.append(
            f"{REFERENCE_INSTANCE} (the #134 instance): no r this time -- {reference.note}"
        )
    else:
        lines.append(
            f"{REFERENCE_INSTANCE} (the #134 instance): r = {reference.pearson:.3f} here "
            f"against {REFERENCE_R} in #134, over {reference.seeds} seed(s)"
        )
    if eligible:
        median = median_r(results)
        lines += [
            f"median r over eligible: {'-' if math.isnan(median) else f'{median:.3f}'}  "
            f"(over the {sum(1 for r in eligible if not math.isnan(r.pearson))} with a "
            "defined r)",
            f"instances at r >= {R_DETERMINED}:  {buckets[DETERMINED]} of {len(eligible)} eligible",
        ]
    call = verdict(results)
    lines += [
        "",
        f"VERDICT: {call.word}",
        f"  {call.detail}",
        "",
        "  Rule, fixed before the campaign: an instance is ELIGIBLE when it has at",
        f"  least {MIN_SEEDS_PER_INSTANCE} usable seeds and SOMETHING varied across them. One",
        "  whose arrival varied but whose outcome held still, or the mirror of that,",
        "  has no defined r and counts as not-determined -- one end moved and the other",
        "  did not, which is evidence against the effect rather than an absence of it.",
        f"  The effect GENERALISES when the median Pearson r is >= {R_DETERMINED}, a strict",
        f"  majority of eligible instances reach {R_DETERMINED} on BOTH Pearson and Spearman,",
        f"  and at least {MIN_ELIGIBLE_INSTANCES} instances are eligible; below that it is",
        "  INCONCLUSIVE.",
        "",
        "  A verdict is not a licence to change the engine. #149's fourth criterion",
        "  stands: any change needs its own issue, hypothesis and regression test.",
    ]
    if min_seeds != MIN_SEEDS_PER_INSTANCE:
        # Otherwise --min-seeds silently rewrites the block above and the report
        # reads as the registered rule at a threshold nobody registered.
        lines += [
            "",
            f"  NOT THE REGISTERED RULE: --min-seeds {min_seeds} overrides the pre-registered",
            f"  {MIN_SEEDS_PER_INSTANCE}. Read this run as a sensitivity check, not as the",
            "  campaign's verdict.",
        ]
    return "\n".join(lines)


#: Bucket order for the summary line and the per-instance listing: the two that
#: answer the question, then the two that cannot, then the unrun.
_BUCKET_ORDER = (
    DETERMINED,
    NOT_DETERMINED,
    FINAL_INVARIANT,
    ARRIVAL_INVARIANT,
    NO_SPREAD,
    TOO_FEW_SEEDS,
)


def _report_order(result: InstanceResult) -> tuple[int, float, str]:
    """Bucket first, then r descending within a bucket, then name."""
    rank = _BUCKET_ORDER.index(result.bucket)
    return (rank, -result.pearson if not math.isnan(result.pearson) else 0.0, result.instance)


def write_csv(path: Path, results: Sequence[InstanceResult]) -> None:
    """The per-instance numbers, for a reader who wants to re-plot them."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["instance", "seeds", "pearson_r", "spearman_rho", "bucket", "note"])
        for result in sorted(results, key=lambda r: r.instance):
            writer.writerow(
                [
                    result.instance,
                    result.seeds,
                    "NaN" if math.isnan(result.pearson) else f"{result.pearson:.6f}",
                    "NaN" if math.isnan(result.spearman) else f"{result.spearman:.6f}",
                    result.bucket,
                    result.note,
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
        default=CONTROL_ARM,
        help="arm to score; applies to any input carrying an `arm` column",
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
    for path in args.results:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        # The documented campaign produces eight seedless per-run tables and
        # --results is one word away from --table, so this is the likeliest
        # single user error here. Refuse it by name rather than dying on the
        # KeyError that reading `row["seed"]` would raise.
        if "seed" not in header:
            return (
                f"--results {path} has no `seed` column, so every row would belong to an "
                "unknown seed; a per-run comparison.csv is passed as --table SEED=PATH"
            )
        if "arm" in header:
            with path.open(newline="") as fh:
                arms = {row["arm"] for row in csv.DictReader(fh)}
            # Otherwise a mistyped arm drops every row and the report comes back
            # a well-formed INCONCLUSIVE -- indistinguishable, after six hours of
            # solving, from a campaign that genuinely measured nothing.
            if args.arm not in arms:
                return (
                    f"--arm {args.arm!r} matches no row in {path}; it carries "
                    f"{', '.join(sorted(arms))}"
                )
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
