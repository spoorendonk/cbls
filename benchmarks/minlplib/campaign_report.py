"""Regenerate every run-derived number the MINLPLib README states (issue #142).

The benchmark README used to carry ~15 derived quantities recomputed by hand
after each campaign -- tallies, gap buckets, the cumulative-feasibility table,
the SCIP head-to-head -- with nothing checking any of them. This module derives
them from the campaign's own committed outputs and nothing else:

* `comparison.csv`    -- the CBLS results table (one row per roster instance);
* `anytime_trace.csv` -- the CBLS incumbent trace (feasible incumbents only);
* `bounds.csv`        -- the published bounds and objective sense, by instance;
* `scip_baseline.csv` -- the SCIP reference rows;
* `<instance>.nl`     -- read only for the free-variable split (#107's table).

It solves nothing. Usage, from the repository root:

    .venv/bin/python3 benchmarks/minlplib/campaign_report.py --budget 60 --seed 1 \\
        [--machine "..."] [--json summary.json] [--markdown report.md]

With neither output flag the Markdown report goes to stdout. `--budget` is
required because no committed table records it: it is the horizon the anytime
score integrates over, and the output says it was supplied rather than read.

THE AGGREGATION RULE (`AGGREGATION_RULE`, printed in every output). The instances
in `runner.CLAIM_EXCLUDED` are published as documented failures (#87). They are
INCLUDED in roster counts -- how many instances the roster has, how many were
built, feasible, infeasible, wall-clock totals, the SCIP head-to-head counts --
because the roster of record is the whole table. They are EXCLUDED from quality
aggregates -- anything that scores an objective: the verdict-vs-BKS breakdown,
the gap buckets, improvement timing, the anytime scores and the both-solved
quality buckets. Before #142 the README's tally used the first denominator and
the driver's summary the second, with nothing saying which a percentage used.

FOR #141 (per-seed reporting). `summarize_results(rows)` is the per-table entry
point: one results table in, every table-level aggregate out, under the rule
above. Call it once per seed; `load_results` reads a table at any path. Every
definition it applies is a named constant or function here, so a per-seed median
is a median of the same quantity this report publishes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.minlplib.reference_solve import (  # noqa: E402
    Bound,
    claim_band,
    load_bounds,
    tie_band,
)
from benchmarks.minlplib.runner import CLAIM_EXCLUDED, completed_search  # noqa: E402
from benchmarks.mipfeas.primal_integral import (  # noqa: E402
    NO_SOLUTION_GAP,
    primal_gap,
    primal_integral,
    shifted_geometric_mean,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

DEFAULT_INST_DIR = Path(__file__).resolve().parents[1] / "instances" / "minlplib"

# --------------------------------------------------------------------------
# Definitions. Each aggregate below is defined once, here.
# --------------------------------------------------------------------------

AGGREGATION_RULE = (
    "Documented-failure instances ({excluded}) are INCLUDED in roster counts "
    "(roster, built, mixed-integer, feasible, infeasible, coverage gaps, errors, "
    "integrality mismatches, verification failures, wall-clock totals, the "
    "cumulative-feasibility profile and the SCIP head-to-head counts) and EXCLUDED "
    "from quality aggregates (verdict-vs-BKS breakdown, gap buckets, improvement "
    "timing, anytime scores, both-solved quality buckets), per #87. Their "
    "per-instance rows are still listed, marked excluded."
).format(excluded=", ".join(CLAIM_EXCLUDED))

#: Gap-bucket thresholds, in percent. A row is in a bucket when its signed
#: `gap_to_bks%` is AT MOST the threshold, so a row better than BKS is within
#: every bucket rather than outside them.
GAP_THRESHOLDS_PCT: tuple[float, ...] = (0.01, 1.0, 10.0)

#: Below this |BKS| the runner's `safe_gap` writes an ABSOLUTE residual into the
#: gap column, not a percentage (`reference_solve.safe_gap`, `minlplib.cpp`).
ZERO_BKS = 1e-12

#: The both-solved CBLS/SCIP quality buckets admit only |BKS| at least this: below
#: it a percentage against the bound is not informative (README, "SCIP baseline").
HEAD_TO_HEAD_MIN_ABS_BKS = 1e-4

#: The runner's default feasibility tolerance (`cbls::kDefaultFeasibilityTolerance`).
#: Not recorded per row; overridable with --feas-tol for a run that set it.
DEFAULT_FEAS_TOL = 1e-6

#: The runner's EARLIER margin rule compared a gap PERCENTAGE against this, i.e.
#: 1e-8 relative. Rows it would have flagged better-than-bks are listed so the
#: README's "those are ties, not improvements" example is regenerated too.
LEGACY_MARGIN_PCT = 1e-6

#: Times (seconds) at which the cumulative-feasibility profile is read. Points
#: past the budget are dropped and the budget itself is always included.
FEASIBILITY_CHECKPOINTS: tuple[float, ...] = (1.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0)

#: An instance first feasible after this many seconds is named as late-feasible:
#: a budget this short would have published it infeasible.
LATE_FEASIBLE_AFTER = 5.0

#: Improvement timing. An IMPROVEMENT is a strict decrease of the recorded
#: (internally minimised) trace objective; the first incumbent counts as one.
#: The trace's `new_best` flag is NOT used: the engine sets it on any
#: improvement over 1e-12 relative (`record_best` in `src/search.cpp`), but the
#: trace writes six significant digits, so a flagged row can print the SAME
#: objective as the row before it -- an improvement the committed trace cannot
#: show and the tie band would not count. An instance "stopped improving early" when its last
#: improvement is at or before EARLY_STOP_SECONDS, and is "still improving" when
#: its last improvement falls inside the final LATE_WINDOW_SECONDS of the budget.
EARLY_STOP_SECONDS = 1.0
LATE_WINDOW_SECONDS = 15.0

#: The CBLS results table and trace write objectives at six significant digits
#: (default `std::ostream` precision), so an objective difference below half a
#: unit in the last place is not a difference. Used when one solver is called
#: AHEAD of the other.
CELL_RELATIVE_RESOLUTION = 5e-6

#: A run "hit the limit" when its wall time reaches the budget, less this
#: relative slack (one rule for both solvers; SCIP's own `status` column is
#: reported beside it as `proved_optimal`).
LIMIT_RELATIVE_SLACK = 1e-3

#: Notes of a runner exception or a non-finite objective (no usable result).
ERROR_NOTES: tuple[str, ...] = ("read-error", "build-error", "solve-error", "non-finite")

#: Notes of a coverage gap (instance declined or not downloaded).
COVERAGE_NOTES: tuple[str, ...] = ("unsupported", "not-found")

NOT_RECORDED = "not recorded"


def verdict_of(note: str) -> str:
    """The runner's own verdict word, with its appended annotations stripped.

    The runner glues `analysis_notes.csv`'s curated root cause onto the note with
    ` | `, and an integrality remark with `; `. Without stripping them, one
    annotated row becomes its own histogram bucket.
    """
    return note.split("(")[0].split(" | ")[0].split(";")[0].strip()


def _float(cell: str | None) -> float:
    try:
        return float(cell) if cell not in (None, "") else math.nan
    except ValueError:
        return math.nan


# --------------------------------------------------------------------------
# Inputs.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Row:
    """One results-table row, joined with its `bounds.csv` sense and catalogue count."""

    instance: str
    objective: float
    primal_bks: float
    gap_pct: float
    wall_seconds: float
    feasible: bool
    note: str
    commit_sha: str
    n_int_vars: int
    maximizing: bool
    n_disc_vars_bks: int
    search_config: str = ""

    @property
    def verdict(self) -> str:
        return verdict_of(self.note)

    @property
    def excluded(self) -> bool:
        """A documented-failure instance: out of every quality aggregate."""
        return self.instance in CLAIM_EXCLUDED

    @property
    def built(self) -> bool:
        """The model was read, built and searched (a completed search note)."""
        return completed_search(self.note)

    @property
    def zero_bks(self) -> bool:
        return abs(self.primal_bks) < ZERO_BKS


def load_bounds_index(path: Path) -> dict[str, Bound]:
    """`bounds.csv` by instance. Extra rows (instances not in a table) are harmless."""
    return {b.instance: b for b in load_bounds(path)}


def load_results(path: Path, bounds: Mapping[str, Bound]) -> list[Row]:
    """Read one CBLS results table (the published one, or any seed's), in file order.

    Refuses a row whose instance `bounds.csv` does not know: without its sense the
    trace normalisation cannot be done, and a silent default of `min` would invert
    every maximize row.
    """
    rows: list[Row] = []
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            name = raw["instance"]
            if name not in bounds:
                raise ValueError(f"{path}: instance {name!r} is not in bounds.csv")
            bound = bounds[name]
            n_int = _float(raw.get("n_int_vars"))
            rows.append(
                Row(
                    instance=name,
                    objective=_float(raw["objective"]),
                    primal_bks=_float(raw["primal_bks"]),
                    gap_pct=_float(raw["gap_to_bks%"]),
                    wall_seconds=_float(raw["wall_seconds"]),
                    feasible=raw["feasible"] == "true",
                    note=raw["note"],
                    commit_sha=raw.get("commit_sha", ""),
                    n_int_vars=-1 if math.isnan(n_int) else int(n_int),
                    maximizing=bound.maximizing,
                    n_disc_vars_bks=bound.n_disc_vars_bks,
                    search_config=raw.get("search_config", "") or "",
                )
            )
    return rows


@dataclass(frozen=True)
class ScipRow:
    instance: str
    objective: float
    gap_pct: float
    wall_seconds: float
    feasible: bool
    status: str
    note: str
    dual_bound: float
    n_int_vars: int
    version: str


def load_scip(path: Path) -> dict[str, ScipRow]:
    rows: dict[str, ScipRow] = {}
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            n_int = _float(raw.get("n_int_vars"))
            rows[raw["instance"]] = ScipRow(
                instance=raw["instance"],
                objective=_float(raw["objective"]),
                gap_pct=_float(raw["gap_to_bks%"]),
                wall_seconds=_float(raw["wall_seconds"]),
                feasible=raw["feasible"] == "true",
                status=raw.get("status", ""),
                note=raw.get("note", ""),
                dual_bound=_float(raw.get("scip_dual_bound")),
                n_int_vars=-1 if math.isnan(n_int) else int(n_int),
                version=raw.get("scip_version", ""),
            )
    return rows


@dataclass(frozen=True)
class TracePoint:
    time_seconds: float
    #: The engine's internally MINIMISED objective: negated on a maximize row.
    objective: float


def load_trace(path: Path) -> dict[str, list[TracePoint]]:
    """The anytime trace by instance, each list sorted by time."""
    trace: dict[str, list[TracePoint]] = defaultdict(list)
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            trace[raw["instance"]].append(
                TracePoint(float(raw["time_seconds"]), float(raw["objective"]))
            )
    return {name: sorted(points, key=lambda p: p.time_seconds) for name, points in trace.items()}


def free_variable_instances(inst_dir: Path, instances: Iterable[str]) -> dict[str, bool] | None:
    """Whether each instance has at least one FREE variable, from its `.nl` bounds.

    A free variable is NL bound type 3 (no lower and no upper bound) in the `b`
    segment -- the "free (unbounded)" split #107's table uses. Returns None when
    any `.nl` is missing, so the split is reported as not derivable rather than
    computed over a partial roster.
    """
    out: dict[str, bool] = {}
    for name in instances:
        path = inst_dir / f"{name}.nl"
        if not path.exists():
            return None
        lines = path.read_text().splitlines()
        n_vars = int(lines[1].split()[0])
        start = next((i for i, line in enumerate(lines) if line.startswith("b")), None)
        if start is None:
            raise ValueError(f"{path}: no 'b' (variable bounds) segment")
        types = [line.split()[0] for line in lines[start + 1 : start + 1 + n_vars]]
        out[name] = "3" in types
    return out


# --------------------------------------------------------------------------
# Table-level aggregates (one results table; #141 calls these per seed).
# --------------------------------------------------------------------------


@dataclass
class RosterCounts:
    """Over EVERY row, documented failures included (`AGGREGATION_RULE`)."""

    roster: int
    built: int
    built_pct: float
    mixed_integer: int
    feasible: int
    infeasible: int
    infeasible_instances: list[str]
    coverage_gaps: int
    errors: int
    integrality_mismatches: int
    verification_failures: int
    #: Documented-failure instances present in the table.
    documented_failures: list[str]
    #: A documented failure that came back FEASIBLE: README says stop and check
    #: #110/#116 before publishing anything about it.
    documented_failures_feasible: list[str]
    total_wall_seconds: float


def roster_counts(rows: Sequence[Row]) -> RosterCounts:
    built = [r for r in rows if r.built]
    infeasible = [r for r in built if not r.feasible and r.verdict == "infeasible"]
    return RosterCounts(
        roster=len(rows),
        built=len(built),
        built_pct=100.0 * len(built) / len(rows) if rows else math.nan,
        mixed_integer=sum(1 for r in built if r.n_int_vars > 0),
        feasible=sum(1 for r in rows if r.feasible),
        infeasible=len(infeasible),
        infeasible_instances=[r.instance for r in infeasible],
        coverage_gaps=sum(1 for r in rows if r.note.startswith(COVERAGE_NOTES)),
        errors=sum(1 for r in rows if r.note.startswith(ERROR_NOTES)),
        integrality_mismatches=sum(
            1
            for r in built
            if r.n_disc_vars_bks >= 0
            and (r.n_int_vars != r.n_disc_vars_bks or "integrality-mismatch" in r.note)
        ),
        verification_failures=sum(1 for r in rows if r.note.startswith("VERIFY-FAILED")),
        documented_failures=[r.instance for r in rows if r.excluded],
        documented_failures_feasible=[r.instance for r in rows if r.excluded and r.feasible],
        total_wall_seconds=math.fsum(r.wall_seconds for r in rows if math.isfinite(r.wall_seconds)),
    )


@dataclass
class BksVerdicts:
    """The runner's own verdicts over the FEASIBLE rows of the claim set."""

    denominator: int
    matches_bks: int
    within_tolerance: int
    worse: int
    better: int
    #: Feasible rows with no published bound to classify against.
    no_bks: int


def bks_verdicts(rows: Sequence[Row]) -> BksVerdicts:
    feasible = [r for r in rows if r.feasible and not r.excluded]
    by_verdict = [r.verdict for r in feasible]
    no_bks = sum(1 for r in feasible if r.verdict == "feasible" and math.isnan(r.primal_bks))
    return BksVerdicts(
        denominator=len(feasible),
        matches_bks=by_verdict.count("matches-bks"),
        within_tolerance=by_verdict.count("within-tolerance-of-bks"),
        worse=by_verdict.count("feasible") - no_bks,
        better=by_verdict.count("better-than-bks"),
        no_bks=no_bks,
    )


@dataclass
class GapBuckets:
    """Counts of rows with `gap_pct <= t` for each threshold in `thresholds`."""

    thresholds_pct: list[float]
    counts: list[int]
    denominator: int
    #: Zero-BKS rows left out (their gap column is an absolute residual).
    excluded_zero_bks: list[str]
    #: Zero-BKS rows kept because objective and BKS are both exactly zero: an
    #: exact match at any threshold.
    retained_zero_bks: list[str]


def _bucket(rows: Sequence[Row], thresholds: Sequence[float]) -> list[int]:
    return [sum(1 for r in rows if r.gap_pct <= t) for t in thresholds]


def gap_buckets(
    rows: Sequence[Row],
    thresholds: Sequence[float] = GAP_THRESHOLDS_PCT,
    *,
    keep_exact_zero: bool = True,
) -> GapBuckets:
    """Gap buckets over the feasible claim-set rows.

    A zero-BKS row's gap cell is an absolute residual, so it is excluded -- unless
    `keep_exact_zero` and its objective is exactly zero too, which is an exact
    match whatever the threshold. `keep_exact_zero=False` is the stricter variant
    the README quotes beside the headline buckets.
    """
    feasible = [r for r in rows if r.feasible and not r.excluded and not math.isnan(r.gap_pct)]
    zero = [r for r in feasible if r.zero_bks]
    retained = [r for r in zero if keep_exact_zero and r.objective == 0.0 and r.primal_bks == 0.0]
    counted = [r for r in feasible if not r.zero_bks or r in retained]
    return GapBuckets(
        thresholds_pct=list(thresholds),
        counts=_bucket(counted, thresholds),
        denominator=len(counted),
        excluded_zero_bks=[r.instance for r in zero if r not in retained],
        retained_zero_bks=[r.instance for r in retained],
    )


@dataclass
class BandExample:
    instance: str
    gap_pct: float
    primal_bks: float


def legacy_margin_ties(rows: Sequence[Row]) -> list[BandExample]:
    """Feasible claim rows the EARLIER margin rule would have called better-than-bks.

    That rule flagged a gap percentage below `-LEGACY_MARGIN_PCT`; every row it
    catches here is a tie under the current two-band rule, which is the point the
    README makes with them. Zero-BKS rows are skipped (their cell is not a percent).
    """
    return [
        BandExample(r.instance, r.gap_pct, r.primal_bks)
        for r in rows
        if r.feasible
        and not r.excluded
        and not r.zero_bks
        and r.gap_pct < -LEGACY_MARGIN_PCT
        and r.verdict != "better-than-bks"
    ]


def single_band_false_ties(rows: Sequence[Row], feas_tol: float) -> list[BandExample]:
    """Rows WORSE than BKS that one band (the claim band) would have called a tie.

    The two-band rule exists because the claim band's absolute floor
    (`10*feas_tol`) dwarfs a small objective; these rows are the evidence.
    """
    out: list[BandExample] = []
    for r in rows:
        if not (r.feasible and not r.excluded and r.verdict == "feasible"):
            continue
        if math.isnan(r.primal_bks) or math.isnan(r.objective):
            continue
        diff = abs(r.objective - r.primal_bks)
        if tie_band(r.primal_bks) < diff <= claim_band(r.primal_bks, feas_tol):
            out.append(BandExample(r.instance, r.gap_pct, r.primal_bks))
    return out


@dataclass
class ResultsSummary:
    """Every aggregate of ONE results table, under `AGGREGATION_RULE`."""

    rule: str
    commit_shas: list[str]
    search_configs: list[str]
    counts: RosterCounts
    verdicts: BksVerdicts
    gap_buckets: GapBuckets
    gap_buckets_strict: GapBuckets
    zero_bks_instances: list[str]
    legacy_margin_ties: list[BandExample]
    single_band_false_ties: list[BandExample]


def summarize_results(rows: Sequence[Row], feas_tol: float = DEFAULT_FEAS_TOL) -> ResultsSummary:
    """The per-table entry point: call once per results table (per seed, for #141)."""
    return ResultsSummary(
        rule=AGGREGATION_RULE,
        commit_shas=sorted({r.commit_sha for r in rows}),
        search_configs=sorted({r.search_config for r in rows if r.search_config}),
        counts=roster_counts(rows),
        verdicts=bks_verdicts(rows),
        gap_buckets=gap_buckets(rows),
        gap_buckets_strict=gap_buckets(rows, keep_exact_zero=False),
        zero_bks_instances=[r.instance for r in rows if r.zero_bks],
        legacy_margin_ties=legacy_margin_ties(rows),
        single_band_false_ties=single_band_false_ties(rows, feas_tol),
    )


# --------------------------------------------------------------------------
# Trace-derived aggregates.
# --------------------------------------------------------------------------


def improvement_times(points: Sequence[TracePoint]) -> list[float]:
    """Times of strict decreases of the recorded objective (the first point included)."""
    best = math.inf
    times: list[float] = []
    for p in points:
        if p.objective < best:
            best = p.objective
            times.append(p.time_seconds)
    return times


@dataclass
class FeasibilityProfile:
    """Cumulative instances first feasible by each checkpoint (a roster count)."""

    roster: int
    checkpoints: list[float]
    counts: list[int]
    late_feasible: dict[str, float]


def feasibility_profile(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]], budget: float
) -> FeasibilityProfile:
    checkpoints = sorted({*(t for t in FEASIBILITY_CHECKPOINTS if t <= budget), budget})
    first = {name: pts[0].time_seconds for name, pts in trace.items() if pts}
    names = {r.instance for r in rows}
    first = {k: v for k, v in first.items() if k in names}
    return FeasibilityProfile(
        roster=len(rows),
        checkpoints=checkpoints,
        counts=[sum(1 for t in first.values() if t <= c) for c in checkpoints],
        late_feasible={
            k: v for k, v in sorted(first.items(), key=lambda kv: kv[1]) if v > LATE_FEASIBLE_AFTER
        },
    )


@dataclass
class ImprovementTiming:
    """When feasible claim-set instances last improved (a quality aggregate)."""

    denominator: int
    stopped_early: int
    still_improving: int
    last_improvement: dict[str, float]
    #: The instance with the most improvements, and its consecutive-incumbent ratio:
    #: the README's evidence that the trace measures the bound-tightening step.
    most_steps_instance: str | None
    most_steps_incumbents: int
    most_steps_median_ratio: float
    most_steps_first: float
    most_steps_last: float


def improvement_timing(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]], budget: float
) -> ImprovementTiming:
    claim = {r.instance for r in rows if r.feasible and not r.excluded}
    last: dict[str, float] = {}
    longest: tuple[str, list[float]] | None = None
    for name in sorted(claim):
        points = trace.get(name, [])
        times = improvement_times(points)
        if not times:
            continue
        last[name] = times[-1]
        values: list[float] = []
        best = math.inf
        for p in points:
            if p.objective < best:
                best = p.objective
                values.append(p.objective)
        if longest is None or len(values) > len(longest[1]):
            longest = (name, values)
    ratios: list[float] = []
    if longest is not None:
        values = longest[1]
        ratios = [b / a for a, b in zip(values, values[1:], strict=False) if a != 0.0]
    return ImprovementTiming(
        denominator=len(last),
        stopped_early=sum(1 for t in last.values() if t <= EARLY_STOP_SECONDS),
        still_improving=sum(1 for t in last.values() if t > budget - LATE_WINDOW_SECONDS),
        last_improvement=last,
        most_steps_instance=longest[0] if longest else None,
        most_steps_incumbents=len(longest[1]) if longest else 0,
        most_steps_median_ratio=statistics.median(ratios) if ratios else math.nan,
        most_steps_first=longest[1][0] if longest else math.nan,
        most_steps_last=longest[1][-1] if longest else math.nan,
    )


@dataclass
class AnytimeInstance:
    instance: str
    excluded: bool
    #: MIPfeas primal integral over [0, budget], in [0, 2]; NaN with no BKS.
    primal_integral: float
    #: MIPfeas primal gap of the final incumbent (2.0 when none).
    final_primal_gap: float


@dataclass
class AnytimeScores:
    """MIPfeas Primal Integral against the published primal bound (BKS).

    Normalisation: the trace objective is internally minimised, so the reference
    is BKS on a minimize row and -BKS on a maximize row. The gap at time t is
    `benchmarks.mipfeas.primal_integral.primal_gap` (|x - x*| / max(|x|, |x*|),
    2 before the first incumbent, 1 across a sign change, 0 when both are below
    1e-6), and the score is its time average over [0, budget]. 0 is "at BKS
    immediately", 2 is "never feasible". An incumbent better than BKS scores a
    positive gap: the measure is distance from the bound.
    """

    budget_seconds: float
    per_instance: list[AnytimeInstance]
    denominator: int
    mean: float
    median: float
    shifted_geometric_mean: float


def anytime_scores(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]], budget: float
) -> AnytimeScores:
    per: list[AnytimeInstance] = []
    for r in rows:
        if not r.built or math.isnan(r.primal_bks):
            per.append(AnytimeInstance(r.instance, r.excluded, math.nan, math.nan))
            continue
        reference = -r.primal_bks if r.maximizing else r.primal_bks
        points = [(p.time_seconds, p.objective) for p in trace.get(r.instance, [])]
        final = primal_gap(points[-1][1], reference) if points else NO_SOLUTION_GAP
        per.append(
            AnytimeInstance(
                r.instance, r.excluded, primal_integral(points, reference, budget), final
            )
        )
    scored = [a.primal_integral for a in per if not a.excluded and math.isfinite(a.primal_integral)]
    return AnytimeScores(
        budget_seconds=budget,
        per_instance=per,
        denominator=len(scored),
        mean=statistics.fmean(scored) if scored else math.nan,
        median=statistics.median(scored) if scored else math.nan,
        shifted_geometric_mean=shifted_geometric_mean(scored),
    )


# --------------------------------------------------------------------------
# SCIP head-to-head.
# --------------------------------------------------------------------------


def hit_limit(wall_seconds: float, budget: float) -> bool:
    return wall_seconds >= budget * (1.0 - LIMIT_RELATIVE_SLACK)


@dataclass
class SolverCounts:
    """Roster counts for one solver (documented failures included)."""

    roster: int
    feasible: int
    proved_optimal: int | None
    hit_limit: int
    total_wall_seconds: float
    median_wall_seconds: float
    under_one_second: int
    integrality_mismatches: int
    verification_failures: int


@dataclass
class DisjointFailure:
    instance: str
    cbls_feasible: bool
    scip_feasible: bool
    cbls_objective: float
    cbls_gap_pct: float
    scip_objective: float
    scip_gap_pct: float
    scip_status: str
    scip_wall_seconds: float
    scip_dual_bound: float


@dataclass
class AheadRow:
    instance: str
    cbls_gap_pct: float
    scip_gap_pct: float
    scip_status: str


@dataclass
class HeadToHead:
    cbls: SolverCounts
    scip: SolverCounts
    disjoint_failures: list[DisjointFailure]
    quality_denominator: int
    quality_thresholds_pct: list[float]
    cbls_quality: list[int]
    scip_quality: list[int]
    cbls_ahead: list[AheadRow]


def _solver_counts(
    walls: Sequence[float],
    feasible: int,
    proved: int | None,
    budget: float,
    mismatches: int,
    failures: int,
) -> SolverCounts:
    finite = [w for w in walls if math.isfinite(w)]
    return SolverCounts(
        roster=len(walls),
        feasible=feasible,
        proved_optimal=proved,
        hit_limit=sum(1 for w in finite if hit_limit(w, budget)),
        total_wall_seconds=math.fsum(finite),
        median_wall_seconds=statistics.median(finite) if finite else math.nan,
        under_one_second=sum(1 for w in finite if w < 1.0),
        integrality_mismatches=mismatches,
        verification_failures=failures,
    )


def _better_by(a: float, b: float, maximizing: bool, margin: float) -> bool:
    """Whether objective `a` beats `b` by more than `margin` in the instance's sense."""
    return (a - b if maximizing else b - a) > margin


def cbls_ahead_of_scip(
    rows: Sequence[Row], scip: Mapping[str, ScipRow], feas_tol: float
) -> list[AheadRow]:
    """Both-solved claim rows where CBLS's objective beats SCIP's by a real margin.

    The margin is the claim band (`reference_solve.claim_band`) or the CBLS cell's
    six-significant-digit resolution, whichever is larger, so neither the
    feasibility slack nor the table's rounding can produce an entry.
    """
    out: list[AheadRow] = []
    for r in _both_solved(rows, scip):
        s = scip[r.instance]
        margin = max(
            claim_band(r.primal_bks, feas_tol), CELL_RELATIVE_RESOLUTION * abs(r.objective)
        )
        if _better_by(r.objective, s.objective, r.maximizing, margin):
            out.append(AheadRow(r.instance, r.gap_pct, s.gap_pct, s.status))
    return out


def _both_solved(rows: Sequence[Row], scip: Mapping[str, ScipRow]) -> list[Row]:
    return [
        r
        for r in rows
        if r.feasible
        and not r.excluded
        and r.instance in scip
        and scip[r.instance].feasible
        and abs(r.primal_bks) >= HEAD_TO_HEAD_MIN_ABS_BKS
    ]


def head_to_head(
    rows: Sequence[Row], scip: Mapping[str, ScipRow], budget: float, feas_tol: float
) -> HeadToHead:
    counts = roster_counts(rows)
    scip_rows = [scip[r.instance] for r in rows if r.instance in scip]
    if len(scip_rows) != len(rows):
        missing = sorted({r.instance for r in rows} - set(scip))
        raise ValueError(f"scip_baseline.csv has no row for {missing}")
    bounds_disc = {r.instance: r.n_disc_vars_bks for r in rows}
    disjoint = [
        DisjointFailure(
            r.instance,
            r.feasible,
            scip[r.instance].feasible,
            r.objective,
            r.gap_pct,
            scip[r.instance].objective,
            scip[r.instance].gap_pct,
            scip[r.instance].status,
            scip[r.instance].wall_seconds,
            scip[r.instance].dual_bound,
        )
        for r in rows
        if r.feasible != scip[r.instance].feasible
    ]
    both = _both_solved(rows, scip)
    return HeadToHead(
        cbls=_solver_counts(
            [r.wall_seconds for r in rows],
            counts.feasible,
            None,
            budget,
            counts.integrality_mismatches,
            counts.verification_failures,
        ),
        scip=_solver_counts(
            [s.wall_seconds for s in scip_rows],
            sum(1 for s in scip_rows if s.feasible),
            sum(1 for s in scip_rows if s.status == "optimal"),
            budget,
            sum(
                1
                for s in scip_rows
                if "integrality mismatch" in s.note
                or (
                    bounds_disc[s.instance] >= 0
                    and s.n_int_vars not in (-1, bounds_disc[s.instance])
                )
            ),
            sum(1 for s in scip_rows if "CHECK-FAILED" in s.note),
        ),
        disjoint_failures=disjoint,
        quality_denominator=len(both),
        quality_thresholds_pct=list(GAP_THRESHOLDS_PCT),
        cbls_quality=[sum(1 for r in both if r.gap_pct <= t) for t in GAP_THRESHOLDS_PCT],
        scip_quality=[
            sum(1 for r in both if scip[r.instance].gap_pct <= t) for t in GAP_THRESHOLDS_PCT
        ],
        cbls_ahead=cbls_ahead_of_scip(rows, scip, feas_tol),
    )


@dataclass
class FreeVariableSplit:
    """#107's table, AFTER column only: within 10% of BKS by free-variable group.

    Eligible means feasible, in the claim set, and |BKS| >= HEAD_TO_HEAD_MIN_ABS_BKS.
    The BEFORE column came from a pre-#107 run that is not committed, so it cannot
    be regenerated and is not reported.
    """

    with_free: int
    with_free_eligible: int
    with_free_within_10pct: int
    without_free: int
    without_free_eligible: int
    without_free_within_10pct: int
    free_instances: list[str] = field(default_factory=list)


def free_variable_split(rows: Sequence[Row], free: Mapping[str, bool]) -> FreeVariableSplit:
    def eligible(r: Row) -> bool:
        return r.feasible and not r.excluded and abs(r.primal_bks) >= HEAD_TO_HEAD_MIN_ABS_BKS

    groups = {
        True: [r for r in rows if free[r.instance]],
        False: [r for r in rows if not free[r.instance]],
    }
    elig = {k: [r for r in v if eligible(r)] for k, v in groups.items()}
    within = {k: sum(1 for r in v if r.gap_pct <= 10.0) for k, v in elig.items()}
    return FreeVariableSplit(
        with_free=len(groups[True]),
        with_free_eligible=len(elig[True]),
        with_free_within_10pct=within[True],
        without_free=len(groups[False]),
        without_free_eligible=len(elig[False]),
        without_free_within_10pct=within[False],
        free_instances=[r.instance for r in groups[True]],
    )


# --------------------------------------------------------------------------
# The whole report.
# --------------------------------------------------------------------------


@dataclass
class Provenance:
    engine_commit: str
    budget_seconds: float
    budget_source: str
    seed: str
    machine: str
    feas_tol: float
    feas_tol_source: str
    scip_configuration: str
    scip_machine: str


@dataclass
class CampaignReport:
    provenance: Provenance
    results: ResultsSummary
    feasibility: FeasibilityProfile
    improvement: ImprovementTiming
    anytime: AnytimeScores
    head_to_head: HeadToHead
    free_variables: FreeVariableSplit | None
    not_regenerated: list[str]


#: Numbers the README states that this report deliberately does NOT regenerate,
#: and why. Printed in every output so a reader can tell "not derived" from
#: "derived and agreeing".
NOT_REGENERATED: tuple[str, ...] = (
    "#107 'within 10% before' column: measured on a pre-#107 table that is not committed.",
    "Per-seed and multi-commit measurements (nvs01's eight seeds, the #102 probe, the "
    "portfolio A/B): separate campaigns, not the committed tables.",
)


def _one_or_flag(values: Sequence[str]) -> str:
    if not values:
        return NOT_RECORDED
    if len(values) == 1:
        return values[0]
    return "MIXED: " + ", ".join(values)


def build_report(
    inst_dir: Path,
    *,
    budget: float,
    seed: int | None,
    machine: str | None,
    feas_tol: float | None,
) -> CampaignReport:
    if budget <= 0:
        raise ValueError("budget must be positive")
    bounds = load_bounds_index(inst_dir / "bounds.csv")
    rows = load_results(inst_dir / "comparison.csv", bounds)
    trace = load_trace(inst_dir / "anytime_trace.csv")
    scip = load_scip(inst_dir / "scip_baseline.csv")
    tol = DEFAULT_FEAS_TOL if feas_tol is None else feas_tol
    results = summarize_results(rows, tol)
    free = free_variable_instances(inst_dir, [r.instance for r in rows])
    provenance = Provenance(
        engine_commit=_one_or_flag(results.commit_shas),
        budget_seconds=budget,
        budget_source="--budget (comparison.csv does not record the budget)",
        seed=f"{seed} (--seed; comparison.csv does not record the seed)"
        if seed is not None
        else NOT_RECORDED,
        machine=f"{machine} (--machine; comparison.csv does not record the machine)"
        if machine
        else NOT_RECORDED,
        feas_tol=tol,
        feas_tol_source="--feas-tol"
        if feas_tol is not None
        else "runner default (comparison.csv does not record it)",
        scip_configuration=_one_or_flag(sorted({s.version for s in scip.values() if s.version})),
        scip_machine=NOT_RECORDED,
    )
    return CampaignReport(
        provenance=provenance,
        results=results,
        feasibility=feasibility_profile(rows, trace, budget),
        improvement=improvement_timing(rows, trace, budget),
        anytime=anytime_scores(rows, trace, budget),
        head_to_head=head_to_head(rows, scip, budget, tol),
        free_variables=free_variable_split(rows, free) if free is not None else None,
        not_regenerated=list(NOT_REGENERATED),
    )


def _jsonable(value: object) -> object:
    """NaN/inf become None: JSON has no spelling for them."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def to_json(report: CampaignReport) -> str:
    return json.dumps(_jsonable(asdict(report)), indent=2, sort_keys=False) + "\n"


# --------------------------------------------------------------------------
# Markdown.
# --------------------------------------------------------------------------


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "n/a"


def _g(value: float, digits: int = 3) -> str:
    return "NaN" if math.isnan(value) else f"{value:.{digits}g}"


def _names(names: Iterable[str]) -> str:
    listed = [f"`{n}`" for n in names]
    return ", ".join(listed) if listed else "none"


def _thresholds(values: Sequence[float]) -> list[str]:
    return [f"≤{t:g}%" for t in values]


def render_markdown(report: CampaignReport) -> str:
    p = report.provenance
    res = report.results
    c = res.counts
    v = res.verdicts
    gb = res.gap_buckets
    gs = res.gap_buckets_strict
    fp = report.feasibility
    it = report.improvement
    at = report.anytime
    h2h = report.head_to_head
    out: list[str] = ["# MINLPLib campaign report", ""]
    out += [
        "## Provenance",
        "",
        f"- engine commit: {p.engine_commit}",
        f"- budget: {p.budget_seconds:g}s per instance ({p.budget_source})",
        f"- seed: {p.seed}",
        f"- machine: {p.machine}",
        f"- feasibility tolerance: {p.feas_tol:g} ({p.feas_tol_source})",
        f"- SCIP configuration: {p.scip_configuration}; SCIP machine: {p.scip_machine}",
        "",
        f"**Aggregation rule.** {res.rule}",
        "",
    ]
    if c.documented_failures_feasible:
        out += [
            f"**WARNING:** documented-failure instance(s) {_names(c.documented_failures_feasible)} "
            "came back FEASIBLE. Check #110/#116 before publishing anything about them.",
            "",
        ]
    out += [
        "## Results tally",
        "",
        "| | count |",
        "|---|---|",
        f"| roster | {c.roster} |",
        f"| parsed and built (closed-model rate) | {c.built} ({c.built_pct:.0f}%) |",
        f"| of which mixed-integer (integrality enforced) | {c.mixed_integer} |",
        f"| **feasible** | **{c.feasible}** |",
        f"| — matching BKS (within the tie band) | {v.matches_bks} |",
        f"| — better than BKS, but inside the tolerance slack | {v.within_tolerance} |",
        f"| — worse than BKS | {v.worse} |",
        f"| — better than BKS | {v.better} |",
        f"| infeasible | {c.infeasible} ({_names(c.infeasible_instances)}) |",
        f"| unsupported / read errors / non-finite | {c.coverage_gaps + c.errors} |",
        f"| integrality mismatches vs catalogue | {c.integrality_mismatches} |",
        f"| verification failures | {c.verification_failures} |",
        "",
        f"Verdict rows are over the {v.denominator} feasible claim-set rows"
        + (f"; {v.no_bks} had no published bound." if v.no_bks else "."),
        "",
        "## Gap distribution",
        "",
        f"Over {gb.denominator} feasible rows: "
        + ", ".join(
            f"**{n} within {t:g}%**" for n, t in zip(gb.counts, gb.thresholds_pct, strict=True)
        )
        + ".",
        f"Zero-BKS rows (|BKS| < {ZERO_BKS:g}, gap cell is an absolute residual): "
        f"{_names(res.zero_bks_instances)}; excluded {_names(gb.excluded_zero_bks)}, "
        f"retained as exact zeros {_names(gb.retained_zero_bks)}.",
        f"Excluding all of them: {' / '.join(str(n) for n in gs.counts)} "
        f"over {gs.denominator} rows.",
        "",
        "Earlier margin rule (gap % below -1e-6) would have flagged as better-than-bks: "
        + (
            ", ".join(
                f"`{e.instance}` at {_g(-e.gap_pct, 2)} percent" for e in res.legacy_margin_ties
            )
            or "none"
        )
        + ".",
        "Worse than BKS but inside the claim band (one band would have called them ties): "
        + (
            ", ".join(
                f"`{e.instance}` (BKS {_g(e.primal_bks)}, {e.gap_pct:.2f}% worse)"
                for e in res.single_band_false_ties
            )
            or "none"
        )
        + ".",
        "",
        "## Cumulative feasibility (roster count)",
        "",
        "| by | " + " | ".join(f"{t:g}s" for t in fp.checkpoints) + " |",
        "|----|" + "|".join("----" for _ in fp.checkpoints) + "|",
        "| feasible | " + " | ".join(str(n) for n in fp.counts) + " |",
        "",
        f"Of {fp.roster}. First feasible after {LATE_FEASIBLE_AFTER:g}s: "
        + (", ".join(f"`{k}` ({t:.1f}s)" for k, t in fp.late_feasible.items()) or "none")
        + ".",
        "",
        "## Improvement timing",
        "",
        f"Of the {it.denominator} feasible claim-set instances, {it.stopped_early} "
        f"({_pct(it.stopped_early, it.denominator)}) last improved within "
        f"{EARLY_STOP_SECONDS:g}s and {it.still_improving} "
        f"({_pct(it.still_improving, it.denominator)}) were still improving in the final "
        f"{LATE_WINDOW_SECONDS:g}s.",
    ]
    if it.most_steps_instance is not None:
        out.append(
            f"Most improvements: `{it.most_steps_instance}`, {it.most_steps_incumbents} "
            f"incumbents ({it.most_steps_incumbents - 1} steps) from "
            f"{_g(it.most_steps_first, 6)} to {_g(it.most_steps_last, 6)}, median "
            f"consecutive ratio {it.most_steps_median_ratio:.7f}."
        )
    out += [
        "",
        "## Anytime score (primal integral vs BKS)",
        "",
        f"Budget {at.budget_seconds:g}s. Over {at.denominator} claim-set instances: mean "
        f"{_g(at.mean, 4)}, median {_g(at.median, 4)}, shifted geometric mean "
        f"{_g(at.shifted_geometric_mean, 4)} (0 = at BKS immediately, "
        f"{NO_SOLUTION_GAP:g} = never feasible).",
        "",
        "| instance | primal integral | final primal gap | |",
        "|---|---|---|---|",
    ]
    out += [
        f"| `{a.instance}` | {_g(a.primal_integral, 4)} | {_g(a.final_primal_gap, 4)} | "
        f"{'excluded' if a.excluded else ''} |"
        for a in at.per_instance
    ]
    cb, sc = h2h.cbls, h2h.scip
    out += [
        "",
        "## SCIP head-to-head (roster counts)",
        "",
        "| | CBLS | SCIP |",
        "|---|---|---|",
        f"| feasible | {cb.feasible} / {cb.roster} | {sc.feasible} / {sc.roster} |",
        f"| proved optimal | n/a (primal heuristic) | {sc.proved_optimal} / {sc.roster} |",
        f"| hit the {p.budget_seconds:g}s limit | {cb.hit_limit} | {sc.hit_limit} |",
        f"| total wall over the roster | {cb.total_wall_seconds:.0f}s | "
        f"{sc.total_wall_seconds:.0f}s (median {sc.median_wall_seconds:.2f}s; "
        f"{sc.under_one_second} instances under 1s) |",
        f"| integrality mismatches vs catalogue | {cb.integrality_mismatches} | "
        f"{sc.integrality_mismatches} |",
        f"| verification failures | {cb.verification_failures} | {sc.verification_failures} |",
        "",
        "Disjoint failures (one solver feasible, the other not):",
        "",
        "| instance | CBLS | SCIP |",
        "|---|---|---|",
    ]
    for d in h2h.disjoint_failures:
        cbls_cell = (
            f"{_g(d.cbls_objective, 6)} ({d.cbls_gap_pct:.2f}%)"
            if d.cbls_feasible
            else "infeasible"
        )
        scip_cell = (
            f"{_g(d.scip_objective, 6)} ({d.scip_gap_pct:.2f}%), {d.scip_status} in "
            f"{d.scip_wall_seconds:.2f}s"
            if d.scip_feasible
            else f"no feasible solution ({d.scip_status}); dual bound {_g(d.scip_dual_bound, 4)}"
        )
        out.append(f"| `{d.instance}` | {cbls_cell} | {scip_cell} |")
    out += [
        "",
        f"Quality where both are feasible, over the {h2h.quality_denominator} claim-set instances "
        f"with |BKS| >= {HEAD_TO_HEAD_MIN_ABS_BKS:g}:",
        "",
        "| | " + " | ".join(_thresholds(h2h.quality_thresholds_pct)) + " |",
        "|---|" + "|".join("---" for _ in h2h.quality_thresholds_pct) + "|",
        "| CBLS | " + " | ".join(str(n) for n in h2h.cbls_quality) + " |",
        "| SCIP | " + " | ".join(str(n) for n in h2h.scip_quality) + " |",
        "",
        "CBLS ahead of SCIP by more than the claim band and the cell resolution:",
        "",
        "| instance | CBLS gap | SCIP gap | SCIP status |",
        "|---|---|---|---|",
    ]
    out += [
        f"| `{a.instance}` | {a.cbls_gap_pct:.4g}% | {a.scip_gap_pct:.4g}% | {a.scip_status} |"
        for a in h2h.cbls_ahead
    ]
    out += ["", "## Free variables (#107, after column)", ""]
    fv = report.free_variables
    if fv is None:
        out.append("Not derivable: a roster `.nl` file is missing.")
    else:
        out += [
            "| | instances | eligible | within 10% |",
            "|---|---|---|---|",
            f"| ≥1 free variable | {fv.with_free} | {fv.with_free_eligible} | "
            f"{fv.with_free_within_10pct} |",
            f"| no free variables | {fv.without_free} | {fv.without_free_eligible} | "
            f"{fv.without_free_within_10pct} |",
        ]
    out += ["", "## Not regenerated", ""]
    out += [f"- {line}" for line in report.not_regenerated]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument(
        "--budget",
        type=float,
        required=True,
        help="per-instance budget in seconds the campaign ran at (no table records it)",
    )
    parser.add_argument("--seed", type=int, default=None, help="the campaign's seed, if known")
    parser.add_argument("--machine", default=None, help="the campaign's machine, if known")
    parser.add_argument("--feas-tol", type=float, default=None)
    parser.add_argument("--json", type=Path, default=None, help="write the summary JSON here")
    parser.add_argument("--markdown", type=Path, default=None, help="write the report here")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.budget <= 0:
        print("--budget must be positive", file=sys.stderr)
        return 2
    report = build_report(
        args.inst_dir,
        budget=args.budget,
        seed=args.seed,
        machine=args.machine,
        feas_tol=args.feas_tol,
    )
    markdown = render_markdown(report)
    if args.json is not None:
        args.json.write_text(to_json(report))
    if args.markdown is not None:
        args.markdown.write_text(markdown)
    if args.json is None and args.markdown is None:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
