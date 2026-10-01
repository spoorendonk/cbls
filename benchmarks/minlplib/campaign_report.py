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
        [--machine "..."] [--json summary.json] [--markdown report.md] \\
        [--write-readme README.md | --check-readme README.md]

THE README IS GENERATOR-OWNED where it states a derived number: each such table
or paragraph sits between `<!-- campaign_report:begin NAME -->` / `end` markers,
`README_RENDERERS` renders every block's whole body, `--write-readme` rewrites
them and `--check-readme` (and the test suite) fails on a stale one.

With neither output flag the Markdown report goes to stdout. `--budget` is
required unless `comparison.run.json` records it (#141; the committed table
predates the record): it is the horizon the anytime
score integrates over, and the output says it was supplied rather than read. It
is checked against the evidence the tables do carry (the CBLS wall times and the
budget in SCIP's configuration cell) and a disagreement is printed as a warning.

THE AGGREGATION RULE is `AGGREGATION_RULE` below, stated once and printed in
every output: the documented-failure instances (`runner.CLAIM_EXCLUDED`, #87)
count in roster counts and are held out of quality aggregates. Before #142 the
README's tally used the first denominator and the driver's summary the second,
with nothing saying which a percentage used.

FOR #141 (per-seed reporting). Two entry points, each taking one seed's inputs:
`summarize_results(rows)` for every aggregate of a results table, and
`summarize_trace(rows, trace, budget)` for every aggregate of that table's
anytime trace (cumulative feasibility, improvement timing, anytime scores).
`load_results` and `load_trace` read them at any path. Every definition they
apply is a named constant or function here, so a per-seed median is a median of
the same quantity this report publishes.

#141 built on those: `summarize_seeds` runs `summarize_results` once per seed of
the per-seed table (`SEEDS_TABLE_NAME`, read by `load_seed_results`) and takes
the spread, under `SEED_AGGREGATION_RULE`; `--seeds` prints it. The driver also
writes a run record beside every table it publishes (`RunRecord`,
`run_record_path`), and when `comparison.csv` has one the budget, seed and
machine are read from it rather than stated -- `--budget` is then optional.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import read_json_object  # noqa: E402
from benchmarks.minlplib.reference_solve import (  # noqa: E402
    SAFE_GAP_ABSOLUTE_BELOW,
    Bound,
    claim_band,
    load_bounds,
    tie_band,
)
from benchmarks.minlplib.runner import (  # noqa: E402
    CLAIM_EXCLUDED,
    COVERAGE_GAP_NOTES,
    completed_search,
)
from benchmarks.mipfeas.primal_integral import (  # noqa: E402
    NO_SOLUTION_GAP,
    primal_gap,
    primal_integral,
    shifted_geometric_mean,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

DEFAULT_INST_DIR = Path(__file__).resolve().parents[1] / "instances" / "minlplib"

# --------------------------------------------------------------------------
# Definitions. Each aggregate below is defined once, here.
# --------------------------------------------------------------------------

AGGREGATION_RULE = (
    "Documented-failure instances ({excluded}) are INCLUDED in roster counts "
    "(roster, built, mixed-integer, feasible, infeasible and the infeasible list, "
    "coverage gaps, errors, non-finite, integrality mismatches, verification "
    "failures, wall-clock totals, the cumulative-feasibility profile and its "
    "late-feasible list, the SCIP head-to-head counts and the disjoint-failure "
    "list) and EXCLUDED from quality aggregates (verdict-vs-BKS breakdown, gap "
    "buckets and the zero-BKS split, the earlier-margin and single-band examples, "
    "improvement timing, anytime scores, both-solved quality buckets, the "
    "CBLS-ahead-of-SCIP list, and the eligible and within-10% columns of the "
    "free-variable split), per #87. Their per-instance rows are still listed, "
    "marked excluded."
).format(excluded=", ".join(CLAIM_EXCLUDED))

#: Gap-bucket thresholds, in percent. A row is in a bucket when its signed
#: `gap_to_bks%` is AT MOST the threshold, so a row better than BKS is within
#: every bucket rather than outside them.
GAP_THRESHOLDS_PCT: tuple[float, ...] = (0.01, 1.0, 10.0)

#: Below this |BKS| the runner's `safe_gap` writes an ABSOLUTE residual into the
#: gap column, not a percentage. The same constant `reference_solve.safe_gap` uses.
ZERO_BKS = SAFE_GAP_ABSOLUTE_BELOW

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
#: That is the PRINTED-OBJECTIVE reading, and it depends on the trace's print
#: precision (six significant digits). The report also gives the NEW-BEST
#: reading -- the engine's own `new_best` flag, which `record_best` in
#: `src/search.cpp` sets on any improvement over 1e-12 relative -- and counts the
#: flagged rows that print the same objective as the row before (improvements
#: below the print resolution), which is exactly where the two readings part.
#: An instance "stopped improving early"
#: when its last improvement is at or before EARLY_STOP_SECONDS, and is "still
#: improving" when its last improvement falls inside the final
#: LATE_WINDOW_SECONDS of the budget -- a window that never reaches back into
#: the early one, so on a short budget the two cannot overlap.
EARLY_STOP_SECONDS = 1.0
LATE_WINDOW_SECONDS = 15.0

#: The CBLS results table and trace write objectives at six significant digits
#: (default `std::ostream` precision), so an objective difference below half a
#: unit in the last place is not a difference. Used when one solver is called
#: AHEAD of the other, and when the trace is checked against the table.
CELL_RELATIVE_RESOLUTION = 5e-6

#: A run "hit the limit" when its wall time reaches the budget, less this
#: relative slack (one rule for both solvers; SCIP's own `status` column is
#: reported beside it as `proved_optimal`).
LIMIT_RELATIVE_SLACK = 1e-3

#: How far the median CBLS wall time may sit from `--budget` before the report
#: warns that the budget it was given is probably not the campaign's. The runner
#: spends its whole budget on this roster (every committed row is 60.00xs).
BUDGET_EVIDENCE_TOLERANCE = 0.05

NOT_RECORDED = "not recorded"

#: The committed machine-readable summary of the published campaign, beside the
#: tables it summarises. README step 2 writes it with `--json`.
SUMMARY_JSON_NAME = "campaign_summary.json"


def verdict_of(note: str) -> str:
    """The runner's own verdict word, with its appended annotations stripped.

    The runner glues `analysis_notes.csv`'s curated root cause onto the note with
    ` | `, and an integrality remark with `; `. Without stripping them, one
    annotated row becomes its own histogram bucket.
    """
    return note.split("(")[0].split(" | ")[0].split(";")[0].strip()


#: The verdicts a FEASIBLE row can carry (`classify_against_bks` in minlplib.cpp).
#: Anything else on a feasible row is reported as unclassified, never dropped.
FEASIBLE_VERDICTS: tuple[str, ...] = (
    "matches-bks",
    "within-tolerance-of-bks",
    "feasible",
    "better-than-bks",
)


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
    #: `bounds.csv`'s primal bound at catalogue precision; `primal_bks` is the
    #: runner's six-significant-digit cell of the same value.
    catalogue_bks: float = math.nan

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
    def coverage_gap(self) -> bool:
        return self.note.startswith(COVERAGE_GAP_NOTES)

    @property
    def verification_failed(self) -> bool:
        """The runner rejected its own incumbent on an independent re-check."""
        return self.note.startswith("VERIFY-FAILED")

    @property
    def zero_bks(self) -> bool:
        return abs(self.primal_bks) < ZERO_BKS


def _refuse_duplicate(path: Path, seen: set[str], name: str) -> None:
    if name in seen:
        raise ValueError(f"{path}: instance {name!r} appears twice")
    seen.add(name)


def load_bounds_index(path: Path) -> dict[str, Bound]:
    """`bounds.csv` by instance. Extra rows (instances not in a table) are harmless."""
    return {b.instance: b for b in load_bounds(path)}


def load_results(path: Path, bounds: Mapping[str, Bound]) -> list[Row]:
    """Read one CBLS results table (the published one, or any seed's), in file order.

    Refuses a row whose instance `bounds.csv` does not know -- without its sense
    the trace normalisation cannot be done, and a silent default of `min` would
    invert every maximize row -- and an instance listed twice, which would inflate
    every roster count.
    """
    rows: list[Row] = []
    seen: set[str] = set()
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            rows.append(_row_from_cells(path, raw, bounds, seen))
    return rows


def _row_from_cells(
    path: Path, raw: Mapping[str, str], bounds: Mapping[str, Bound], seen: set[str]
) -> Row:
    """One results-table row, read by `load_results` and `load_seed_results` alike."""
    name = raw["instance"]
    _refuse_duplicate(path, seen, name)
    if name not in bounds:
        raise ValueError(f"{path}: instance {name!r} is not in bounds.csv")
    bound = bounds[name]
    n_int = _float(raw.get("n_int_vars"))
    return Row(
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
        catalogue_bks=bound.primal_bks,
    )


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
    read_seconds: float = math.nan


def load_scip(path: Path) -> dict[str, ScipRow]:
    rows: dict[str, ScipRow] = {}
    seen: set[str] = set()
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            _refuse_duplicate(path, seen, raw["instance"])
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
                read_seconds=_float(raw.get("read_seconds")),
            )
    return rows


@dataclass(frozen=True)
class TracePoint:
    time_seconds: float
    #: The engine's internally MINIMISED objective: negated on a maximize row.
    objective: float
    #: The engine's own `new_best` flag for this row.
    new_best: bool = True


def load_trace(path: Path) -> dict[str, list[TracePoint]]:
    """The anytime trace by instance, each list sorted by time.

    Refuses a non-finite entry, as `mipfeas.primal_integral.load_trace` does: one
    NaN turns an integral into NaN and drops the instance from every aggregate
    without a word. (The runner never writes one -- its recorder skips a
    non-finite incumbent -- so one here means the file is not the runner's.)
    """
    trace: dict[str, list[TracePoint]] = defaultdict(list)
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            point = TracePoint(
                float(raw["time_seconds"]),
                float(raw["objective"]),
                raw.get("new_best", "1").strip() in ("1", "true"),
            )
            if not (math.isfinite(point.time_seconds) and math.isfinite(point.objective)):
                raise ValueError(f"{path}: non-finite trace entry for {raw['instance']!r}")
            trace[raw["instance"]].append(point)
    return {name: sorted(points, key=lambda p: p.time_seconds) for name, points in trace.items()}


def check_trace_matches_table(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]]
) -> None:
    """Refuse a trace that is not the results table's own run.

    Every aggregate of the trace -- the feasibility profile, improvement timing,
    the anytime score -- is silently wrong if the trace is stale or partial, so
    the two are cross-checked: no trace instance outside the table; every row
    published feasible with a finite objective has a trace whose best incumbent
    (un-negated on a maximize row) is that objective to the cells' resolution;
    and no row published infeasible has one, except a VERIFY-FAILED row, whose
    incumbents the runner recorded before rejecting them.
    """
    by_name = {r.instance: r for r in rows}
    stray = sorted(set(trace) - set(by_name))
    if stray:
        raise ValueError(f"anytime trace names instances not in the results table: {stray}")
    for r in rows:
        points = trace.get(r.instance, [])
        if not r.feasible:
            if points and not r.verification_failed:
                raise ValueError(f"{r.instance}: infeasible in the table but traced")
            continue
        if not math.isfinite(r.objective):
            continue  # the non-finite-objective witness (#100) is never traced
        if not points:
            raise ValueError(f"{r.instance}: feasible in the table but absent from the trace")
        best = min(p.objective for p in points)
        best = -best if r.maximizing else best
        slack = 2 * CELL_RELATIVE_RESOLUTION * max(abs(best), abs(r.objective)) + 1e-300
        if abs(best - r.objective) > slack:
            raise ValueError(
                f"{r.instance}: trace's best incumbent {best:g} is not the table's "
                f"objective {r.objective:g}; the trace is not this table's run"
            )


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
        if not lines or not lines[0].startswith("g"):
            raise ValueError(f"{path}: not a text ('g' header) NL file")
        n_vars = int(lines[1].split()[0])
        # The segment header is a line that is exactly "b"; line 0 is the file
        # header and never the segment.
        start = next((i for i, line in enumerate(lines) if i > 0 and line.strip() == "b"), None)
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
    """Over EVERY row, documented failures included (`AGGREGATION_RULE`).

    `built + coverage_gaps + errors == roster` always: an error is any row whose
    note is neither a completed search nor a coverage gap -- the allowlist
    complement, so a note nobody has heard of lands here rather than nowhere.
    """

    roster: int
    built: int
    built_pct: float
    mixed_integer: int
    feasible: int
    infeasible: int
    infeasible_instances: list[str]
    coverage_gaps: int
    errors: int
    error_instances: list[str]
    #: A completed search whose objective was not finite.
    non_finite: int
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
    errors = [r for r in rows if not r.built and not r.coverage_gap]
    return RosterCounts(
        roster=len(rows),
        built=len(built),
        built_pct=100.0 * len(built) / len(rows) if rows else math.nan,
        mixed_integer=sum(1 for r in built if r.n_int_vars > 0),
        feasible=sum(1 for r in rows if r.feasible),
        infeasible=len(infeasible),
        infeasible_instances=[r.instance for r in infeasible],
        coverage_gaps=sum(1 for r in rows if r.coverage_gap),
        errors=len(errors),
        error_instances=[r.instance for r in errors],
        non_finite=sum(1 for r in built if r.verdict == "non-finite"),
        integrality_mismatches=sum(
            1
            for r in built
            if r.n_disc_vars_bks >= 0
            and (r.n_int_vars != r.n_disc_vars_bks or "integrality-mismatch" in r.note)
        ),
        verification_failures=sum(1 for r in rows if r.verification_failed),
        documented_failures=[r.instance for r in rows if r.excluded],
        documented_failures_feasible=[r.instance for r in rows if r.excluded and r.feasible],
        total_wall_seconds=math.fsum(r.wall_seconds for r in rows if math.isfinite(r.wall_seconds)),
    )


@dataclass
class BksVerdicts:
    """The runner's own verdicts over the FEASIBLE rows of the claim set.

    The five counts add up to `denominator`: a verdict outside
    `FEASIBLE_VERDICTS` is named in `unclassified`, never dropped.
    """

    denominator: int
    matches_bks: int
    within_tolerance: int
    worse: int
    better: int
    #: Feasible rows with no published bound to classify against.
    no_bks: int
    unclassified: list[str] = field(default_factory=list)


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
        unclassified=[r.instance for r in feasible if r.verdict not in FEASIBLE_VERDICTS],
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
# Trace-derived aggregates (one trace; #141 calls `summarize_trace` per seed).
# --------------------------------------------------------------------------


def improvement_steps(points: Sequence[TracePoint]) -> list[TracePoint]:
    """The points where the recorded objective strictly decreased (the first included).

    The one definition of an improvement (see `EARLY_STOP_SECONDS`'s comment).
    """
    best = math.inf
    steps: list[TracePoint] = []
    for p in points:
        if p.objective < best:
            best = p.objective
            steps.append(p)
    return steps


def improvement_times(points: Sequence[TracePoint]) -> list[float]:
    return [p.time_seconds for p in improvement_steps(points)]


def _trace_of(r: Row, trace: Mapping[str, Sequence[TracePoint]]) -> Sequence[TracePoint]:
    """A row's incumbents, or none when the table does not publish it feasible.

    The runner records the trace before its independent re-check, so a
    VERIFY-FAILED row carries incumbents the runner then rejected; nothing here
    may count them.
    """
    return trace.get(r.instance, []) if r.feasible else []


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
    """First-feasible times, clamped to the budget as the primal integral clamps them.

    A run can record its first incumbent a few milliseconds past the budget while
    the clock is being read; clamping keeps the last column equal to the
    feasible count of the instances that have a trace.
    """
    checkpoints = sorted({*(t for t in FEASIBILITY_CHECKPOINTS if t <= budget), budget})
    first = {
        r.instance: min(points[0].time_seconds, budget)
        for r in rows
        if (points := _trace_of(r, trace))
    }
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
    #: The printed-objective reading (see `EARLY_STOP_SECONDS`).
    stopped_early: int
    still_improving: int
    #: The new-best reading: the same split off the engine's `new_best` flag.
    new_best_stopped_early: int
    new_best_still_improving: int
    #: `new_best` rows that print the same objective as the row before them --
    #: improvements below the trace's print resolution, where the readings part.
    sub_resolution_new_best_rows: int
    #: The same count per instance (instances with none are omitted).
    sub_resolution_by_instance: dict[str, int]
    first_feasible: dict[str, float]
    last_improvement: dict[str, float]
    #: The instance with the most improvements, and its consecutive-incumbent ratio:
    #: the README's evidence that the trace measures the bound-tightening step.
    #: First/last are in the instance's own sense (un-negated on a maximize row).
    most_steps_instance: str | None
    most_steps_incumbents: int
    most_steps_median_ratio: float
    most_steps_first: float
    most_steps_last: float


def improvement_timing(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]], budget: float
) -> ImprovementTiming:
    late_start = max(budget - LATE_WINDOW_SECONDS, EARLY_STOP_SECONDS)
    first: dict[str, float] = {}
    last: dict[str, float] = {}
    last_flagged: dict[str, float] = {}
    sub_resolution = 0
    sub_resolution_by_instance: dict[str, int] = {}
    longest: tuple[Row, list[TracePoint]] | None = None
    for r in sorted((r for r in rows if r.feasible and not r.excluded), key=lambda r: r.instance):
        points = _trace_of(r, trace)
        steps = improvement_steps(points)
        if not steps:
            continue
        first[r.instance] = steps[0].time_seconds
        last[r.instance] = steps[-1].time_seconds
        # The arrival counts in both readings; after it, only flagged rows.
        flagged = [points[0].time_seconds] + [p.time_seconds for p in points if p.new_best]
        last_flagged[r.instance] = max(flagged)
        unseen = sum(
            1
            for prev, p in zip(points, points[1:], strict=False)
            if p.new_best and p.objective >= prev.objective
        )
        sub_resolution += unseen
        if unseen:
            sub_resolution_by_instance[r.instance] = unseen
        if longest is None or len(steps) > len(longest[1]):
            longest = (r, steps)
    if longest is None:
        return ImprovementTiming(
            0, 0, 0, 0, 0, 0, {}, {}, {}, None, 0, math.nan, math.nan, math.nan
        )
    row, steps = longest
    values = [p.objective for p in steps]
    ratios = [b / a for a, b in zip(values, values[1:], strict=False) if a != 0.0]
    sign = -1.0 if row.maximizing else 1.0
    return ImprovementTiming(
        denominator=len(last),
        stopped_early=sum(1 for t in last.values() if t <= EARLY_STOP_SECONDS),
        still_improving=sum(1 for t in last.values() if t > late_start),
        new_best_stopped_early=sum(1 for t in last_flagged.values() if t <= EARLY_STOP_SECONDS),
        new_best_still_improving=sum(1 for t in last_flagged.values() if t > late_start),
        sub_resolution_new_best_rows=sub_resolution,
        sub_resolution_by_instance=sub_resolution_by_instance,
        first_feasible=first,
        last_improvement=last,
        most_steps_instance=row.instance,
        most_steps_incumbents=len(steps),
        most_steps_median_ratio=statistics.median(ratios) if ratios else math.nan,
        most_steps_first=sign * values[0],
        most_steps_last=sign * values[-1],
    )


@dataclass
class AnytimeInstance:
    instance: str
    excluded: bool
    #: MIPfeas primal integral over [0, budget], in [0, 2]; NaN when unscored.
    primal_integral: float
    #: MIPfeas primal gap of the final incumbent (2.0 when none); NaN when unscored.
    final_primal_gap: float
    #: The reference the trace was scored against, in the trace's minimised sense.
    reference: float = math.nan
    #: Where `reference` came from, and at what precision.
    reference_source: str = ""


@dataclass
class AnytimeScores:
    """MIPfeas Primal Integral against the published primal bound (BKS).

    Normalisation: the trace objective is internally minimised, so the reference
    is BKS on a minimize row and -BKS on a maximize row. The gap at time t is
    `benchmarks.mipfeas.primal_integral.primal_gap` (|x - x*| / max(|x|, |x*|),
    2 before the first incumbent, 1 across a sign change, 0 when both are below
    1e-6), and the score is its time average over [0, budget]. 0 is "at BKS
    immediately", 2 is "never feasible". Two consequences to read the mean with:
    BKS is not a proven optimum, so an incumbent better than BKS scores a
    positive gap (the measure is distance from the bound); and a zero-BKS row
    whose objective is not within 1e-6 of zero scores 1.0 throughout -- such
    rows are left out of the gap buckets but are IN this score.

    Unscored (NaN, out of the aggregates, named in `unscored`): a row with no
    completed search, no published bound, or a VERIFY-FAILED verdict -- the last
    as `mipfeas` withholds a failed-verification row rather than scoring it.
    """

    budget_seconds: float
    per_instance: list[AnytimeInstance]
    denominator: int
    unscored: list[str]
    mean: float
    median: float
    shifted_geometric_mean: float


def _reference_source(r: Row) -> str:
    """Where a row's anytime reference came from, and at what precision."""
    source = (
        "bounds.csv primal_bks (catalogue precision)"
        if math.isfinite(r.catalogue_bks)
        else "comparison.csv primal_bks (6 significant digits)"
    )
    return f"-1 x {source}; maximize row" if r.maximizing else source


def anytime_scores(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]], budget: float
) -> AnytimeScores:
    per: list[AnytimeInstance] = []
    unscored: list[str] = []
    for r in rows:
        if not r.built or math.isnan(r.primal_bks) or r.verification_failed:
            per.append(AnytimeInstance(r.instance, r.excluded, math.nan, math.nan))
            if not r.excluded:
                unscored.append(r.instance)
            continue
        bks = r.catalogue_bks if math.isfinite(r.catalogue_bks) else r.primal_bks
        reference = -bks if r.maximizing else bks
        points = [(p.time_seconds, p.objective) for p in _trace_of(r, trace)]
        final = primal_gap(points[-1][1], reference) if points else NO_SOLUTION_GAP
        per.append(
            AnytimeInstance(
                r.instance,
                r.excluded,
                primal_integral(points, reference, budget),
                final,
                reference,
                _reference_source(r),
            )
        )
    scored = [a.primal_integral for a in per if not a.excluded and math.isfinite(a.primal_integral)]
    return AnytimeScores(
        budget_seconds=budget,
        per_instance=per,
        denominator=len(scored),
        unscored=unscored,
        mean=statistics.fmean(scored) if scored else math.nan,
        median=statistics.median(scored) if scored else math.nan,
        shifted_geometric_mean=shifted_geometric_mean(scored),
    )


@dataclass
class TraceSummary:
    """Every aggregate of ONE anytime trace against its results table."""

    feasibility: FeasibilityProfile
    improvement: ImprovementTiming
    anytime: AnytimeScores


def summarize_trace(
    rows: Sequence[Row], trace: Mapping[str, Sequence[TracePoint]], budget: float
) -> TraceSummary:
    """The per-trace entry point: call once per seed's trace (for #141).

    Cross-checks the trace against the table first (`check_trace_matches_table`).
    """
    if not (math.isfinite(budget) and budget > 0):
        raise ValueError("budget must be positive and finite")
    check_trace_matches_table(rows, trace)
    return TraceSummary(
        feasibility=feasibility_profile(rows, trace, budget),
        improvement=improvement_timing(rows, trace, budget),
        anytime=anytime_scores(rows, trace, budget),
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
    #: Instance read time summed over the roster (SCIP only; NaN for CBLS).
    total_read_seconds: float = math.nan


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
    cbls_verdict: str
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
            out.append(AheadRow(r.instance, r.verdict, r.gap_pct, s.gap_pct, s.status))
    return out


def head_to_head(
    rows: Sequence[Row], scip: Mapping[str, ScipRow], budget: float, feas_tol: float
) -> HeadToHead:
    counts = roster_counts(rows)
    missing = sorted({r.instance for r in rows} - set(scip))
    if missing:
        raise ValueError(f"scip_baseline.csv has no row for {missing}")
    scip_rows = [scip[r.instance] for r in rows]
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
    scip_mismatches = sum(
        1
        for s in scip_rows
        if "integrality mismatch" in s.note
        or (bounds_disc[s.instance] >= 0 and s.n_int_vars not in (-1, bounds_disc[s.instance]))
    )
    scip_counts = _solver_counts(
        [s.wall_seconds for s in scip_rows],
        sum(1 for s in scip_rows if s.feasible),
        sum(1 for s in scip_rows if s.status == "optimal"),
        budget,
        scip_mismatches,
        sum(1 for s in scip_rows if "CHECK-FAILED" in s.note),
    )
    scip_counts.total_read_seconds = math.fsum(
        s.read_seconds for s in scip_rows if math.isfinite(s.read_seconds)
    )
    return HeadToHead(
        cbls=_solver_counts(
            [r.wall_seconds for r in rows],
            counts.feasible,
            None,
            budget,
            counts.integrality_mismatches,
            counts.verification_failures,
        ),
        scip=scip_counts,
        disjoint_failures=disjoint,
        quality_denominator=len(both),
        quality_thresholds_pct=list(GAP_THRESHOLDS_PCT),
        cbls_quality=[sum(1 for r in both if r.gap_pct <= t) for t in GAP_THRESHOLDS_PCT],
        scip_quality=[
            sum(1 for r in both if scip[r.instance].gap_pct <= t) for t in GAP_THRESHOLDS_PCT
        ],
        cbls_ahead=cbls_ahead_of_scip(rows, scip, feas_tol),
    )


#: #107's BEFORE column, from the pre-#107 run, which is not committed: the
#: within-10% count of each group and the free-variable instances it named. Fixed
#: constants (the run cannot be regenerated); the README's AFTER - BEFORE
#: sentences are rendered from these and the generated AFTER column.
FREE_VARIABLES_BEFORE_WITHIN_10PCT = 1
NO_FREE_VARIABLES_BEFORE_WITHIN_10PCT = 20
FREE_VARIABLES_BEFORE_WITHIN_10PCT_INSTANCES: tuple[str, ...] = ("maxmin",)


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
    #: The free-variable instances within 10% of BKS, by name.
    with_free_within_10pct_instances: list[str] = field(default_factory=list)


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
        with_free_within_10pct_instances=[r.instance for r in elig[True] if r.gap_pct <= 10.0],
    )


# --------------------------------------------------------------------------
# Run records and the per-seed table (#141).
# --------------------------------------------------------------------------

#: The run record `run_benchmark.py` writes beside every results table it
#: publishes, named after the table: `comparison.csv` -> `comparison.run.json`.
#: A wall-clock-budgeted table that does not say what machine produced it, at
#: what budget and seed, is an anecdote; this is where it says so.
RUN_RECORD_SUFFIX = ".run.json"

#: Bumped when a record's shape changes, so a reader refuses one it does not know.
RUN_RECORD_SCHEMA = 1

#: The per-seed table, beside `comparison.csv`: every seed's rows of the
#: published protocol, one block per seed, `seed` first and then the runner's
#: own columns. The pre-registered seed's rows are in it too, so it is the
#: whole record; `comparison.csv` stays the single pre-registered seed (#141).
SEEDS_TABLE_NAME = "comparison_seeds.csv"
SEED_COLUMN = "seed"

#: Below this many aggregated seeds a spread is printed with a note that it is
#: under the floor (#141: three seeds is the defensible minimum, not five).
MIN_SEEDS_FOR_SPREAD = 3

#: How the per-seed table is aggregated. Stated once, printed beside every
#: multi-seed summary, and layered ON `AGGREGATION_RULE` rather than beside it.
SEED_AGGREGATION_RULE = (
    "Each seed's table is summarized on its own under the aggregation rule, and "
    "a count's spread is the min / median / max of those per-seed counts. Per "
    "instance, the median and range are of gap_to_bks% over EVERY aggregated "
    "seed, a seed that did not reach feasibility counting as +inf (worse than any "
    "gap) rather than being dropped -- so a median is finite only when more than "
    "half the seeds are feasible, and a range's upper end is inf when any seed is "
    "not. A zero-BKS instance's cells are absolute residuals, not percentages, "
    "and are marked; a documented failure is listed, marked excluded, and stays "
    "out of every quality claim. Only seeds run at one commit, one budget and on "
    "one host are "
    "aggregated; any other seed in the table is named with the reason it was "
    "left out. The published comparison.csv is the pre-registered seed alone and "
    "is never replaced by an aggregate."
)


@dataclass(frozen=True)
class RunRecord:
    """What produced one seed's results, as the driver recorded it at publish time."""

    commit: str
    budget_seconds: float
    seed: int
    roster: int
    #: Rows staged by an EARLIER invocation and reused: `machine` did not see them.
    resumed: int
    published_at: str
    #: `benchmarks.common.provenance.machine_record()` at the start of the
    #: invocation that published.
    machine: dict[str, object]
    #: How many solves shared the machine, and with how many threads each.
    concurrency: dict[str, object]


def run_record_path(table: Path) -> Path:
    """Where the run record of the results table at `table` lives."""
    return table.with_name(table.stem + RUN_RECORD_SUFFIX)


def _record_field(obj: Mapping[str, object], key: str, where: str) -> object:
    if key not in obj:
        raise ValueError(f"{where}: run record has no {key!r}")
    return obj[key]


def _record_str(obj: Mapping[str, object], key: str, where: str) -> str:
    value = _record_field(obj, key, where)
    if not isinstance(value, str):
        raise ValueError(f"{where}: run record's {key!r} is not a string")
    return value


def _record_int(obj: Mapping[str, object], key: str, where: str) -> int:
    value = _record_field(obj, key, where)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: run record's {key!r} is not an integer")
    return value


def _record_dict(obj: Mapping[str, object], key: str, where: str) -> dict[str, object]:
    value = _record_field(obj, key, where)
    if not isinstance(value, dict):
        raise ValueError(f"{where}: run record's {key!r} is not an object")
    return {str(k): v for k, v in value.items()}


def parse_run_record(obj: Mapping[str, object], where: str) -> RunRecord:
    """A `RunRecord` from its JSON object; refuses a malformed one rather than guessing."""
    budget = _record_field(obj, "budget_seconds", where)
    if isinstance(budget, bool) or not isinstance(budget, int | float):
        raise ValueError(f"{where}: run record's 'budget_seconds' is not a number")
    if not (math.isfinite(budget) and budget > 0):
        raise ValueError(f"{where}: run record's budget {budget} is not positive and finite")
    return RunRecord(
        commit=_record_str(obj, "commit", where),
        budget_seconds=float(budget),
        seed=_record_int(obj, "seed", where),
        roster=_record_int(obj, "roster", where),
        resumed=_record_int(obj, "resumed", where),
        published_at=_record_str(obj, "published_at", where),
        machine=_record_dict(obj, "machine", where),
        concurrency=_record_dict(obj, "concurrency", where),
    )


def _read_record_file(path: Path) -> dict[str, object] | None:
    """The record file's object, None when absent; a present but unreadable one is refused.

    Absent is the honest state of every table published before #141 and reads
    as "not recorded". A file that exists and does not parse is damage, and
    reading it as absent would quietly turn a recorded machine into an
    unrecorded one.
    """
    if not path.exists():
        return None
    obj = read_json_object(path)
    if obj is None:
        raise ValueError(f"{path} exists but is not a readable JSON object")
    schema = obj.get("schema")
    if schema != RUN_RECORD_SCHEMA:
        raise ValueError(f"{path}: run record schema {schema!r}, expected {RUN_RECORD_SCHEMA}")
    return obj


def load_run_record(table: Path) -> RunRecord | None:
    """The run record beside a single-seed results table, or None if it has none."""
    path = run_record_path(table)
    obj = _read_record_file(path)
    return None if obj is None else parse_run_record(obj, str(path))


def load_seed_run_records(seeds_table: Path) -> dict[int, RunRecord]:
    """The per-seed table's run records, by seed; empty when it has none."""
    path = run_record_path(seeds_table)
    obj = _read_record_file(path)
    if obj is None:
        return {}
    seeds = obj.get("seeds")
    if not isinstance(seeds, dict):
        raise ValueError(f"{path}: has no 'seeds' object")
    records: dict[int, RunRecord] = {}
    for key, value in seeds.items():
        if not isinstance(value, dict):
            raise ValueError(f"{path}: seed {key!r} is not an object")
        record = parse_run_record({str(k): v for k, v in value.items()}, f"{path} seed {key}")
        if str(record.seed) != str(key):
            raise ValueError(f"{path}: the record under seed {key!r} is seed {record.seed}")
        records[record.seed] = record
    return records


def describe_machine(record: RunRecord) -> str:
    """One line naming the machine and the concurrency a run was measured under."""
    m, c = record.machine, record.concurrency

    def known(value: object) -> str:
        return "?" if value is None else str(value)

    memory = m.get("memory_total_kib")
    memory_text = (
        f"{memory / 1024**2:.1f} GiB RAM"
        if isinstance(memory, int | float) and not isinstance(memory, bool)
        else "RAM ?"
    )
    load = m.get("load_average")
    load_text = (
        f"load {load[0]:.2f} at start"
        if isinstance(load, list) and load and isinstance(load[0], int | float)
        else "load ?"
    )
    model = m.get("cpu_model")
    return (
        f"{known(m.get('host'))}{f' ({model})' if model else ''}, "
        f"{known(m.get('cpu_count'))} CPUs "
        f"({known(m.get('cpu_affinity'))} usable), {memory_text}, {load_text}; "
        f"{known(c.get('parallel_solves'))} solve(s) at a time, "
        f"{known(c.get('threads_per_solve'))} thread(s) each"
    )


def load_seed_results(path: Path, bounds: Mapping[str, Bound]) -> dict[int, list[Row]]:
    """Read the per-seed table: each seed's rows, in file order, by seed.

    The same row reader as `load_results`, so a seed's rows are exactly what that
    seed's single-seed table would read as; a duplicate instance is refused
    within a seed, not across seeds.
    """
    by_seed: dict[int, list[Row]] = {}
    seen: dict[int, set[str]] = defaultdict(set)
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            try:
                seed = int(raw[SEED_COLUMN])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: a row without an integer {SEED_COLUMN!r}") from exc
            by_seed.setdefault(seed, []).append(_row_from_cells(path, raw, bounds, seen[seed]))
    return by_seed


def comparable_seeds(
    by_seed: Mapping[int, Sequence[Row]],
    records: Mapping[int, RunRecord],
    *,
    commit: str,
    budget: float,
    host: str | None = None,
) -> tuple[dict[int, list[Row]], dict[int, str]]:
    """Split the per-seed table into the seeds to aggregate and the ones left out, with why.

    A seed is aggregated only when every one of its rows names `commit` and its
    run record is for that commit, at `budget`, on `host` (when given). A seed
    with no run record is left out: the table does not carry the budget, so
    nothing says its rows are comparable. A record for another commit than its
    rows is what a kill between the driver's two writes leaves behind. Another
    host would mix machine variance into a seed spread.
    """
    kept: dict[int, list[Row]] = {}
    left_out: dict[int, str] = {}
    for seed in sorted(by_seed):
        rows = list(by_seed[seed])
        shas = sorted({r.commit_sha for r in rows})
        record = records.get(seed)
        if shas != [commit]:
            left_out[seed] = f"commit {', '.join(shas)}, not {commit}"
        elif record is None:
            left_out[seed] = "no run record, so its budget is unknown"
        elif record.commit != commit:
            left_out[seed] = f"its run record is for commit {record.commit}, not its rows' {commit}"
        elif record.budget_seconds != budget:
            left_out[seed] = f"budget {record.budget_seconds:g}s, not {budget:g}s"
        elif host is not None and record.machine.get("host") != host:
            left_out[seed] = f"host {record.machine.get('host')}, not {host}"
        else:
            kept[seed] = rows
    return kept, left_out


@dataclass
class CountSpread:
    """One roster or quality count, per seed and its spread across seeds."""

    per_seed: dict[int, int]
    min: int
    median: float
    max: int


def _spread(per_seed: Mapping[int, int]) -> CountSpread:
    values = list(per_seed.values())
    return CountSpread(
        per_seed=dict(per_seed),
        min=min(values),
        median=float(statistics.median(values)),
        max=max(values),
    )


@dataclass
class InstanceSpread:
    """One instance's gap across the aggregated seeds (`SEED_AGGREGATION_RULE`)."""

    instance: str
    excluded: bool
    zero_bks: bool
    feasible_seeds: int
    seeds: int
    #: Seeds on which no search completed (a coverage gap or an error): counted
    #: as +inf like an infeasible seed, but marked, since nothing was searched.
    not_built_seeds: int
    gap_median_pct: float
    gap_min_pct: float
    gap_max_pct: float


def instance_spread(rows: Sequence[Row]) -> InstanceSpread:
    """One instance's gap median and range over `rows` (one per seed), infeasible as +inf.

    NaN throughout when a feasible seed has no gap (no published bound): there is
    no ordering to take a median over.
    """
    values = [r.gap_pct if r.feasible else math.inf for r in rows]
    defined = not any(math.isnan(v) for v in values)
    return InstanceSpread(
        instance=rows[0].instance,
        excluded=rows[0].excluded,
        zero_bks=any(r.zero_bks for r in rows),
        feasible_seeds=sum(1 for r in rows if r.feasible),
        seeds=len(rows),
        not_built_seeds=sum(1 for r in rows if not r.built),
        gap_median_pct=float(statistics.median(values)) if defined else math.nan,
        gap_min_pct=min(values) if defined else math.nan,
        gap_max_pct=max(values) if defined else math.nan,
    )


@dataclass
class SeedsSummary:
    """Every multi-seed aggregate, computed from per-seed `summarize_results`."""

    rule: str
    seed_rule: str
    seeds: list[int]
    commit: str
    budget_seconds: float
    #: Roster count: feasible rows per seed, documented failures included.
    feasible: CountSpread
    #: Quality count: matches-bks among the feasible claim-set rows per seed.
    matches_bks: CountSpread
    per_instance: list[InstanceSpread]
    #: Seeds in the table that are not aggregated, and why.
    left_out: dict[int, str]
    #: Each aggregated seed's machine and concurrency, from its run record.
    machines: dict[int, str]


def summarize_seeds(
    by_seed: Mapping[int, Sequence[Row]],
    records: Mapping[int, RunRecord],
    *,
    commit: str,
    budget: float,
    host: str | None = None,
    feas_tol: float = DEFAULT_FEAS_TOL,
) -> SeedsSummary:
    """The multi-seed entry point: `summarize_results` per seed, then the spread.

    Refuses when no seed is comparable, and when the comparable seeds disagree
    about the roster -- an instance missing from one seed would otherwise move
    every count spread by a row nobody ran.
    """
    kept, left_out = comparable_seeds(by_seed, records, commit=commit, budget=budget, host=host)
    if not kept:
        raise ValueError(
            f"no seed in the per-seed table ran at commit {commit} and {budget:g}s"
            + (f" on {host}" if host else "")
            + "".join(f"; seed {s}: {why}" for s, why in left_out.items())
        )
    seeds = sorted(kept)
    roster = [r.instance for r in kept[seeds[0]]]
    differing = [s for s in seeds if sorted(r.instance for r in kept[s]) != sorted(roster)]
    if differing:
        raise ValueError(f"seeds {differing} do not share the roster of seed {seeds[0]}")
    summaries = {seed: summarize_results(kept[seed], feas_tol) for seed in seeds}
    by_instance: dict[str, list[Row]] = defaultdict(list)
    for seed in seeds:
        for r in kept[seed]:
            by_instance[r.instance].append(r)
    return SeedsSummary(
        rule=AGGREGATION_RULE,
        seed_rule=SEED_AGGREGATION_RULE,
        seeds=seeds,
        commit=commit,
        budget_seconds=budget,
        feasible=_spread({s: summaries[s].counts.feasible for s in seeds}),
        matches_bks=_spread({s: summaries[s].verdicts.matches_bks for s in seeds}),
        per_instance=[instance_spread(by_instance[name]) for name in roster],
        left_out=left_out,
        machines={s: describe_machine(records[s]) for s in seeds},
    )


def _gap_cell(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return "infeasible" if math.isinf(value) else f"{value:.4g}"


def render_seeds_text(summary: SeedsSummary) -> str:
    """The multi-seed summary as plain text, for the driver and `--seeds`."""
    f, m = summary.feasible, summary.matches_bks

    def per_seed(spread: CountSpread) -> str:
        return ", ".join(f"seed {s}: {n}" for s, n in spread.per_seed.items())

    lines = [
        f"rule: {summary.rule}",
        f"seed rule: {summary.seed_rule}",
        f"seeds aggregated: {', '.join(map(str, summary.seeds))} "
        f"(commit {summary.commit}, {summary.budget_seconds:g}s per instance)",
    ]
    if len(summary.seeds) < MIN_SEEDS_FOR_SPREAD:
        lines.append(
            f"NOTE: {len(summary.seeds)} seed(s) aggregated; {MIN_SEEDS_FOR_SPREAD} is the "
            "floor before a spread is worth quoting"
        )
    lines += [f"left out: seed {s} ({why})" for s, why in summary.left_out.items()]
    lines += [f"machine, seed {s}: {text}" for s, text in summary.machines.items()]
    lines += [
        f"feasible (roster count): min {f.min}, median {f.median:g}, max {f.max} [{per_seed(f)}]",
        f"matches-bks (claim set): min {m.min}, median {m.median:g}, max {m.max} [{per_seed(m)}]",
        "per instance: gap_to_bks% median [min, max], feasible seeds",
    ]
    for s in summary.per_instance:
        marks = [mark for mark, on in (("excluded", s.excluded), ("zero-BKS", s.zero_bks)) if on]
        if s.not_built_seeds:
            marks.append(f"no search on {s.not_built_seeds}")
        lines.append(
            f"  {s.instance}: {_gap_cell(s.gap_median_pct)} "
            f"[{_gap_cell(s.gap_min_pct)}, {_gap_cell(s.gap_max_pct)}], "
            f"{s.feasible_seeds}/{s.seeds}" + (f" ({', '.join(marks)})" if marks else "")
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The whole report.
# --------------------------------------------------------------------------


@dataclass
class Provenance:
    """What produced the tables. Values are null when the tables do not record them.

    Each `*_source` says where the value came from, so a reader can tell a value
    the tables carry from one the caller typed.
    """

    engine_commit: str
    search_config: str
    budget_seconds: float
    budget_source: str
    seed: int | None
    seed_source: str
    machine: str | None
    machine_source: str
    feas_tol: float
    feas_tol_source: str
    observed_median_cbls_wall_seconds: float
    scip_configuration: str
    scip_budget_seconds: float | None
    scip_machine: str
    #: Disagreements between `--budget` and the evidence in the tables.
    warnings: list[str] = field(default_factory=list)
    #: The run record the budget, seed and machine were read from, if any (#141).
    run_record: str | None = None


@dataclass
class InstanceFigures:
    """One instance's figures side by side, so a reader sees what moves the aggregates."""

    instance: str
    excluded: bool
    verdict: str
    gap_pct: float
    first_feasible_seconds: float
    last_improvement_seconds: float
    primal_integral: float
    final_primal_gap: float


@dataclass
class CampaignReport:
    provenance: Provenance
    results: ResultsSummary
    trace: TraceSummary
    head_to_head: HeadToHead
    free_variables: FreeVariableSplit | None
    per_instance: list[InstanceFigures]
    not_regenerated: list[str]


#: Numbers the README states that this report deliberately does NOT regenerate,
#: and why. Printed in every output so a reader can tell "not derived" from
#: "derived and agreeing".
NOT_REGENERATED: tuple[str, ...] = (
    "#107's BEFORE column itself: a pre-#107 table that is not committed, held as "
    "the FREE_VARIABLES_BEFORE_* / NO_FREE_VARIABLES_BEFORE_* constants (the "
    "differences from the generated AFTER column are rendered).",
    "Per-seed and multi-commit measurements (nvs01's eight seeds, the #102 probe, the "
    "portfolio A/B, the replication spreads quoted in Results, the st_e40/nvs01 "
    "re-checks): separate campaigns, not the committed tables.",
    "The ex6_2_6 'stale catalogue row' note under SCIP baseline: a prose reading of "
    "one scip_baseline.csv row, not an aggregate.",
    "SCIP's CPU/wall ratio (~1.0), its clock type and its version string as quoted "
    "in prose: measured or read from SCIP, not from the tables' aggregates.",
    "Machine descriptions (the SCIP run's hardware): no table records a machine.",
    "Published bounds quoted in prose (e.g. st_e40's BKS): reference values from "
    "bounds.csv, not run-derived.",
    "Single cells quoted inside the root-cause prose (nvs01's published residual): "
    "the table's own text, not an aggregate.",
)

_SCIP_BUDGET = re.compile(r"/\s*([0-9.]+)s\s*/")


def _scip_budget(configuration: str) -> float | None:
    match = _SCIP_BUDGET.search(configuration)
    return float(match.group(1)) if match else None


def _one_or_flag(values: Sequence[str]) -> str:
    if not values:
        return NOT_RECORDED
    if len(values) == 1:
        return values[0]
    return "MIXED: " + ", ".join(values)


def _budget_warnings(budget: float, cbls_median: float, scip_budget: float | None) -> list[str]:
    out: list[str] = []
    if (
        math.isfinite(cbls_median)
        and abs(cbls_median - budget) > BUDGET_EVIDENCE_TOLERANCE * budget
    ):
        out.append(
            f"--budget {budget:g}s but the median CBLS wall time is {cbls_median:.2f}s: "
            "the budget is probably not the campaign's"
        )
    if scip_budget is not None and scip_budget != budget:
        out.append(
            f"--budget {budget:g}s but SCIP's configuration records {scip_budget:g}s; "
            "the head-to-head compares runs at different budgets"
        )
    return out


@dataclass(frozen=True)
class _Stated:
    """The budget, seed and machine, each with where it came from (`build_report`)."""

    budget: float
    budget_source: str
    seed: int | None
    seed_source: str
    machine: str | None
    machine_source: str
    run_record: str | None
    warnings: list[str]


def _stated(
    table: Path,
    record: RunRecord | None,
    *,
    budget: float | None,
    seed: int | None,
    machine: str | None,
    commit_shas: Sequence[str],
) -> _Stated:
    """Read the budget, seed and machine from the table's run record, else from the caller.

    The record is read, the caller's values are typed, so the record wins: a
    caller value that disagrees with it is a warning, not an override. Without a
    record -- every table published before #141 -- the caller's values are used
    and labelled as stated, exactly as before.
    """
    if record is None:
        if budget is None:
            raise ValueError(
                f"{run_record_path(table).name} not found beside {table}; pass the budget"
            )
        return _Stated(
            budget=budget,
            budget_source="--budget (comparison.csv does not record the budget)",
            seed=seed,
            seed_source="--seed (comparison.csv does not record the seed)"
            if seed is not None
            else NOT_RECORDED,
            machine=machine,
            machine_source="--machine (comparison.csv does not record the machine)"
            if machine
            else NOT_RECORDED,
            run_record=None,
            warnings=[],
        )
    name = run_record_path(table).name
    warnings: list[str] = []
    if budget is not None and budget != record.budget_seconds:
        warnings.append(f"--budget {budget:g}s but {name} records {record.budget_seconds:g}s")
    if seed is not None and seed != record.seed:
        warnings.append(f"--seed {seed} but {name} records seed {record.seed}")
    if machine:
        warnings.append(f"--machine is ignored: {name} records the machine")
    if list(commit_shas) != [record.commit]:
        # Refused, not warned: the record would otherwise supply the budget (the
        # anytime horizon), the seed and the machine of ANOTHER run, and
        # `--write-readme` would publish them as read. A kill between the
        # driver's table and record writes leaves exactly this.
        raise ValueError(
            f"{name} is for commit {record.commit} but the table's rows name "
            f"{', '.join(commit_shas) or 'none'}: the record is not this table's. Re-run "
            "the publish, or move the record aside and state --budget/--seed by hand"
        )
    return _Stated(
        budget=record.budget_seconds,
        budget_source=name,
        seed=record.seed,
        seed_source=name,
        machine=describe_machine(record),
        machine_source=name,
        run_record=name,
        warnings=warnings,
    )


def build_report(
    inst_dir: Path,
    *,
    budget: float | None,
    seed: int | None,
    machine: str | None,
    feas_tol: float | None,
    results_csv: Path | None = None,
    trace_csv: Path | None = None,
    scip_csv: Path | None = None,
    bounds_csv: Path | None = None,
) -> CampaignReport:
    """The whole report. Each input file defaults to its published name in
    `inst_dir`; pass one seed's table and trace to report on that seed (#141).
    `inst_dir` is still where the `.nl` files are read from. `budget`, `seed` and
    `machine` are read from the run record beside the results table when there is
    one (#141); otherwise `budget` is required.
    """
    if budget is not None and not (math.isfinite(budget) and budget > 0):
        raise ValueError("budget must be positive and finite")
    if feas_tol is not None and not (math.isfinite(feas_tol) and feas_tol > 0):
        raise ValueError("feas_tol must be positive and finite")
    table = results_csv or inst_dir / "comparison.csv"
    bounds = load_bounds_index(bounds_csv or inst_dir / "bounds.csv")
    rows = load_results(table, bounds)
    stated = _stated(
        table,
        load_run_record(table),
        budget=budget,
        seed=seed,
        machine=machine,
        commit_shas=sorted({r.commit_sha for r in rows}),
    )
    budget = stated.budget
    trace = load_trace(trace_csv or inst_dir / "anytime_trace.csv")
    scip = load_scip(scip_csv or inst_dir / "scip_baseline.csv")
    tol = DEFAULT_FEAS_TOL if feas_tol is None else feas_tol
    results = summarize_results(rows, tol)
    trace_summary = summarize_trace(rows, trace, budget)
    free = free_variable_instances(inst_dir, [r.instance for r in rows])
    built_walls = [r.wall_seconds for r in rows if r.built and math.isfinite(r.wall_seconds)]
    cbls_median = statistics.median(built_walls) if built_walls else math.nan
    scip_configuration = _one_or_flag(sorted({s.version for s in scip.values() if s.version}))
    scip_budget = _scip_budget(scip_configuration)
    provenance = Provenance(
        engine_commit=_one_or_flag(results.commit_shas),
        search_config=_one_or_flag(results.search_configs),
        budget_seconds=budget,
        budget_source=stated.budget_source,
        seed=stated.seed,
        seed_source=stated.seed_source,
        machine=stated.machine,
        machine_source=stated.machine_source,
        feas_tol=tol,
        feas_tol_source="--feas-tol"
        if feas_tol is not None
        else "runner default (comparison.csv does not record it)",
        observed_median_cbls_wall_seconds=cbls_median,
        scip_configuration=scip_configuration,
        scip_budget_seconds=scip_budget,
        scip_machine="not in scip_baseline.csv (the README's SCIP 'Hardware' note names one)",
        warnings=stated.warnings + _budget_warnings(budget, cbls_median, scip_budget),
        run_record=stated.run_record,
    )
    anytime = {a.instance: a for a in trace_summary.anytime.per_instance}
    timing = trace_summary.improvement
    per_instance = [
        InstanceFigures(
            instance=r.instance,
            excluded=r.excluded,
            verdict=r.verdict,
            gap_pct=r.gap_pct,
            first_feasible_seconds=timing.first_feasible.get(r.instance, math.nan),
            last_improvement_seconds=timing.last_improvement.get(r.instance, math.nan),
            primal_integral=anytime[r.instance].primal_integral,
            final_primal_gap=anytime[r.instance].final_primal_gap,
        )
        for r in rows
    ]
    return CampaignReport(
        provenance=provenance,
        results=results,
        trace=trace_summary,
        head_to_head=head_to_head(rows, scip, budget, tol),
        free_variables=free_variable_split(rows, free) if free is not None else None,
        per_instance=per_instance,
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
    return json.dumps(_jsonable(asdict(report)), indent=2, allow_nan=False) + "\n"


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


def _sourced(value: object, source: str) -> str:
    return NOT_RECORDED if value is None else f"{value} ({source})"


def _provenance_lines(p: Provenance, rule: str) -> list[str]:
    out = [
        "## Provenance",
        "",
        f"- engine commit: {p.engine_commit}",
        f"- search config: {p.search_config}",
        f"- budget: {p.budget_seconds:g}s per instance ({p.budget_source}); observed "
        f"median CBLS wall {_g(p.observed_median_cbls_wall_seconds, 5)}s",
        f"- seed: {_sourced(p.seed, p.seed_source)}",
        f"- machine: {_sourced(p.machine, p.machine_source)}",
        f"- feasibility tolerance: {p.feas_tol:g} ({p.feas_tol_source})",
        f"- SCIP configuration: {p.scip_configuration}; SCIP machine: {p.scip_machine}",
        "",
    ]
    for warning in p.warnings:
        out += [f"**WARNING:** {warning}", ""]
    return [*out, f"**Aggregation rule.** {rule}", ""]


def _results_lines(res: ResultsSummary) -> list[str]:
    c, v, gb, gs = res.counts, res.verdicts, res.gap_buckets, res.gap_buckets_strict
    out: list[str] = []
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
        f"| unsupported / read errors / non-finite | {c.coverage_gaps + c.errors + c.non_finite} |",
        f"| integrality mismatches vs catalogue | {c.integrality_mismatches} |",
        f"| verification failures | {c.verification_failures} |",
        "",
        f"Verdict rows are over the {v.denominator} feasible claim-set rows"
        + (f"; {v.no_bks} had no published bound" if v.no_bks else "")
        + (f"; unclassified: {_names(v.unclassified)}" if v.unclassified else "")
        + ".",
    ]
    if c.errors:
        out.append(f"Errors (no search completed): {_names(c.error_instances)}.")
    out += [
        "",
        "## Gap distribution",
        "",
        f"Over {gb.denominator} of the {res.verdicts.denominator} feasible claim-set rows: "
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
        f"Earlier margin rule (gap % below -{LEGACY_MARGIN_PCT:g}) would have flagged as "
        "better-than-bks: "
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
    ]
    return out


def _trace_lines(ts: TraceSummary) -> list[str]:
    fp, it, at = ts.feasibility, ts.improvement, ts.anytime
    out = [
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
        f"{LATE_WINDOW_SECONDS:g}s (printed-objective reading). Off the engine's "
        f"`new_best` flag: {it.new_best_stopped_early} "
        f"({_pct(it.new_best_stopped_early, it.denominator)}) and "
        f"{it.new_best_still_improving} ({_pct(it.new_best_still_improving, it.denominator)}); "
        f"{it.sub_resolution_new_best_rows} `new_best` rows print the same objective as "
        "the row before.",
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
        f"{NO_SOLUTION_GAP:g} = never feasible). Unscored: {_names(at.unscored)}.",
        "",
    ]
    return out


def _per_instance_lines(figures: Sequence[InstanceFigures]) -> list[str]:
    out = [
        "## Per instance",
        "",
        "| instance | verdict | gap % | first feasible | last improvement | primal integral "
        "| final primal gap | |",
        "|---|---|---|---|---|---|---|---|",
    ]
    out += [
        f"| `{f.instance}` | {f.verdict} | {_g(f.gap_pct, 4)} | "
        f"{_g(f.first_feasible_seconds, 3)} | {_g(f.last_improvement_seconds, 3)} | "
        f"{_g(f.primal_integral, 4)} | {_g(f.final_primal_gap, 4)} | "
        f"{'excluded' if f.excluded else ''} |"
        for f in figures
    ]
    return [*out, ""]


def _head_to_head_lines(h2h: HeadToHead, budget: float) -> list[str]:
    cb, sc = h2h.cbls, h2h.scip
    out = [
        "## SCIP head-to-head (roster counts)",
        "",
        "| | CBLS | SCIP |",
        "|---|---|---|",
        f"| feasible | {cb.feasible} / {cb.roster} | {sc.feasible} / {sc.roster} |",
        f"| proved optimal | n/a (primal heuristic) | {sc.proved_optimal} / {sc.roster} |",
        f"| hit the {budget:g}s limit | {cb.hit_limit} | {sc.hit_limit} |",
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
    return [*out, ""]


def render_markdown(report: CampaignReport) -> str:
    out: list[str] = ["# MINLPLib campaign report", ""]
    out += _provenance_lines(report.provenance, report.results.rule)
    out += _results_lines(report.results)
    out += _trace_lines(report.trace)
    out += _head_to_head_lines(report.head_to_head, report.provenance.budget_seconds)
    out += ["## Free variables (#107, after column)", ""]
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
    out += [""]
    out += _per_instance_lines(report.per_instance)
    out += ["## Not regenerated", ""]
    out += [f"- {line}" for line in report.not_regenerated]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------
# README blocks: the README's derived text, owned by this module.
# --------------------------------------------------------------------------

#: Every derived number in the benchmark README lives between a pair of these
#: markers, and `readme_blocks` renders each block's whole body. A README edit
#: inside a block that this module would not have written makes
#: `check_readme` fail; a number left outside the blocks must be one of the
#: kinds `NOT_REGENERATED` names.
README_BEGIN = "<!-- campaign_report:begin {name} -->"
README_END = "<!-- campaign_report:end {name} -->"
_README_BLOCK = re.compile(
    r"<!-- campaign_report:begin (?P<name>[a-z0-9-]+) -->\n(?P<body>.*?)"
    r"<!-- campaign_report:end (?P=name) -->",
    re.DOTALL,
)

_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]


def _word(n: int, *, capital: bool = False) -> str:
    """A small count spelled out, as the README's prose writes it."""
    text = _WORDS[n] if 0 <= n < len(_WORDS) else str(n)
    return text.capitalize() if capital else text


def _sci(value: float, digits: int = 2) -> str:
    """Scientific notation without exponent padding: 8.3e-5, 3.07e-4."""
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _plain(value: float) -> str:
    """`:g` without the exponent padding: 1e-6, 1e9, 8.46."""
    return re.sub(
        r"e([+-])0*(\d)", lambda m: "e" + ("-" if m[1] == "-" else "") + m[2], f"{value:g}"
    )


def _pct_cell(value: float) -> str:
    # Decided on the ROUNDED value: 99.96 at three significant digits is 100, and
    # `:#.3g` would print it as "100.".
    if abs(float(f"{value:.3g}")) >= 100:
        return f"{value:.0f}%"
    if abs(value) >= 0.01:
        return f"{value:#.3g}%"
    return f"{_sci(value)}%"


def _and_list(items: Sequence[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _tick(names: Iterable[str]) -> list[str]:
    return [f"`{n}`" for n in names]


def _readme_provenance(r: CampaignReport) -> str:
    p = r.provenance
    seed = f"seed {p.seed}" if p.seed is not None else "seed not recorded"
    machine = p.machine if p.machine else "not recorded"
    where = (
        f"the budget, the seed and the machine are read from `{p.run_record}`, which the "
        "driver writes at publish time"
        if p.run_record
        else "the budget and the seed are recorded in no table and are stated to the generator"
    )
    return (
        f"Latest run: **{p.budget_seconds:g}s per instance, {seed}, feasibility tolerance "
        f"{_plain(p.feas_tol)}**, engine commit `{p.engine_commit}` (recorded per row in "
        f"`comparison.csv`; {where}; machine: "
        f"{machine})."
    )


def _readme_tally(r: CampaignReport) -> str:
    c, v = r.results.counts, r.results.verdicts
    return "\n".join(
        [
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
            f"| infeasible | {c.infeasible} |",
            f"| unsupported / read errors / non-finite | "
            f"{c.coverage_gaps + c.errors + c.non_finite} |",
            f"| integrality mismatches vs catalogue | {c.integrality_mismatches} |",
            f"| verification failures | {c.verification_failures} |",
            "",
            _claim_set_sentence(r),
        ]
    )


def _claim_set_sentence(r: CampaignReport) -> str:
    c, v = r.results.counts, r.results.verdicts
    text = (
        f"The four verdict rows are over the {v.denominator} feasible claim-set rows; "
        f"{_and_list(_tick(c.documented_failures)) or 'no instance'} "
        "are excluded from them per the aggregation rule below"
    )
    if c.documented_failures_feasible:
        text += (
            f", and {_and_list(_tick(c.documented_failures_feasible))} came back feasible, "
            "so the feasible row counts it and the verdict rows do not"
        )
    if v.no_bks:
        text += f". {v.no_bks} feasible rows have no published bound"
    if v.unclassified:
        text += f". Unclassified verdicts: {_and_list(_tick(v.unclassified))}"
    if c.errors:
        text += f". No search completed on: {_and_list(_tick(c.error_instances))}"
    return text + "."


def _readme_rule(r: CampaignReport) -> str:
    return f"**Aggregation rule.** {r.results.rule}"


def _readme_gap_buckets(r: CampaignReport) -> str:
    res = r.results
    gb, gs = res.gap_buckets, res.gap_buckets_strict
    buckets = ", ".join(
        f"{n} within {t:g}%{' of BKS' if i == 0 else ''}"
        for i, (n, t) in enumerate(zip(gb.counts, gb.thresholds_pct, strict=True))
    )
    zero = _tick(res.zero_bks_instances)
    excluded = _tick(gb.excluded_zero_bks)
    retained = _tick(gb.retained_zero_bks)
    by_name = {f.instance: f for f in r.per_instance}
    one_pct = gb.thresholds_pct.index(1.0) if 1.0 in gb.thresholds_pct else None
    inside = [
        f"`{n}` (gap cell {_g(by_name[n].gap_pct, 4)})"
        for n in gb.excluded_zero_bks
        if one_pct is not None and by_name[n].gap_pct <= 1.0
    ]
    lines = [
        f"Gap distribution over {gb.denominator} of the {res.verdicts.denominator} feasible "
        f"claim-set instances: **{buckets}.**",
        "",
        f"{_word(len(zero), capital=True)} rows have a numerically zero BKS "
        f"(`|BKS| < {_plain(ZERO_BKS)}`), for which the runner writes an *absolute* residual into "
        "the `gap_to_bks%` column rather than a meaningless percentage against zero: "
        f"{_and_list(zero)}. Those values are not percentages. The buckets above exclude "
        f"{_and_list(excluded) or 'none of them'}, whose residual is non-zero, and retain "
        f"{_and_list(retained) or 'none'}, where objective and BKS are both exactly 0 and "
        "so are exact matches at any threshold. Excluding all of them instead gives "
        f"{' / '.join(str(n) for n in gs.counts)} over {gs.denominator} rows.",
    ]
    if inside:
        lines[-1] += (
            " Counting the excluded rows *as* percentages would have put "
            f'{_and_list(inside)} inside the "within 1%" bucket.'
        )
    return "\n".join(lines)


def _readme_two_band(r: CampaignReport) -> str:
    res = r.results
    v = res.verdicts
    if v.better == 0:
        opening = "Nothing in this roster beats a published bound."
    else:
        opening = f"{_word(v.better, capital=True)} rows beat a published bound."
    ties = sorted(res.legacy_margin_ties, key=lambda e: -e.gap_pct)
    if ties:
        listed = _and_list([f"`{e.instance}` at {_sci(-e.gap_pct)} percent" for e in ties])
        legacy = (
            " Under the runner's earlier margin rule — which compared a *percentage* "
            f"against {_plain(LEGACY_MARGIN_PCT)}, i.e. 1e-8 relative — {_word(len(ties))} "
            f"rows of this run would have been flagged `better-than-bks`: {listed}. "
            "Those are ties, not improvements."
        )
    else:
        legacy = ""
    false_ties = res.single_band_false_ties
    if false_ties:
        example = _and_list(
            [
                f"`{e.instance}` (BKS {_sci(e.primal_bks, 3)}) as matching BKS when it was "
                f"{e.gap_pct:.2f}% worse"
                for e in false_ties
            ]
        )
    else:
        example = "a small-objective row worse than BKS as matching it"
    return "\n".join(
        [
            opening + legacy,
            "",
            "Two bands are used, deliberately different. An improvement is only *claimed* "
            "when it exceeds `max(1e-6·(|BKS|+1), 10·feas_tol)`: we accept solutions "
            "violating a constraint by up to `feas_tol`, and that slack itself buys a small "
            "objective gain. A *tie* requires the much tighter, purely relative "
            f"`1e-6·(|BKS|+1)` — using one band for both would have published {example}, "
            "because the absolute floor dwarfs an objective that small. A row that improves "
            "on BKS by more than the tie band but less than the claim threshold falls between "
            "the two and is labelled `within-tolerance-of-bks` rather than being miscounted "
            "as worse.",
        ]
    )


def _readme_anytime(r: CampaignReport) -> str:
    at = r.trace.anytime
    zero_in = [
        n
        for n in r.results.gap_buckets.excluded_zero_bks
        if any(a.instance == n and math.isfinite(a.primal_integral) for a in at.per_instance)
    ]
    excluded = _and_list(_tick(r.results.counts.documented_failures)) or "none"
    text = (
        "**Anytime score.** The MIPfeas Primal Integral "
        "(`benchmarks/mipfeas/primal_integral.py`) of the committed trace against BKS over "
        f'the {at.budget_seconds:g}s budget — 0 is "at BKS from the first instant", '
        f'{NO_SOLUTION_GAP:g} is "never feasible" — over the {at.denominator} instances '
        f"outside the documented failures ({excluded}): **mean {at.mean:.3f}, median "
        f"{at.median:.3f}, shifted geometric mean {at.shifted_geometric_mean:.3g}**. A "
        "maximize row's trace is negated, so its reference is −BKS (at catalogue "
        "precision, from `bounds.csv`). BKS is not a proven optimum, so an incumbent "
        "past it scores a positive gap"
    )
    if zero_in:
        text += (
            f"; and the zero-BKS rows left out of the gap buckets ({_and_list(_tick(zero_in))}) "
            "are *in* this score, because the scorer's own zero test is 1e-6 absolute"
        )
    if at.unscored:
        text += f". Unscored: {_and_list(_tick(at.unscored))}"
    return text + (
        ". The per-instance scores, with the reference each was scored against, are in "
        f"`{SUMMARY_JSON_NAME}`; with one seed, each is one draw."
    )


def _readme_feasibility(r: CampaignReport) -> str:
    fp = r.trace.feasibility
    late = list(fp.late_feasible.items())
    at_late = (
        fp.counts[fp.checkpoints.index(LATE_FEASIBLE_AFTER)]
        if LATE_FEASIBLE_AFTER in fp.checkpoints
        else None
    )
    lines = [
        "Measured from the committed trace, not assumed. Cumulative instances with a "
        f"feasible solution by time t (of {fp.roster}):",
        "",
        "| by | " + " | ".join(f"{t:g}s" for t in fp.checkpoints) + " |",
        "|----|" + "|".join("----" for _ in fp.checkpoints) + "|",
        "| feasible | " + " | ".join(str(n) for n in fp.counts) + " |",
        "",
    ]
    if late and at_late is not None:
        names = ", ".join(f"`{k}` ({t:.1f}s)" for k, t in late)
        n = _word(len(late))
        lines.append(
            f"**This is the load-bearing argument.** {_word(len(late), capital=True)} "
            f"instances reach feasibility only long after {LATE_FEASIBLE_AFTER:g}s — "
            f"{names} — so a {LATE_FEASIBLE_AFTER:g}s budget would publish all {n} as "
            f"infeasible, {at_late} solved instead of {fp.counts[-1]}. Which {n} varies "
            "between draws; that several exist does not."
        )
    else:
        lines.append(f"No instance reaches feasibility after {LATE_FEASIBLE_AFTER:g}s in this run.")
    return "\n".join(lines)


def _readme_improvement(r: CampaignReport) -> str:
    it = r.trace.improvement
    d = it.denominator
    text = (
        "Solution *quality* over time is a weaker argument than it first appears, and is "
        f"recorded here with that caveat. Of the {d} claim-set instances that become "
        "feasible, "
        f"{_pct(it.stopped_early, d)} stop improving within the first "
        f"{'second' if EARLY_STOP_SECONDS == 1.0 else f'{EARLY_STOP_SECONDS:g}s'} while "
        f"{_pct(it.still_improving, d)} are still improving in the final "
        f"{LATE_WINDOW_SECONDS:g} seconds — reading an improvement as a strict decrease of "
        "the trace's *printed*, six-significant-digit objective. Read off the engine's own "
        f"`new_best` flag instead, the split is {_pct(it.new_best_stopped_early, d)} / "
        f"{_pct(it.new_best_still_improving, d)}. The two part because "
        f"{it.sub_resolution_new_best_rows} of the trace's `new_best` rows print the same "
        "objective as the row before: improvements below the trace's print resolution, "
        "which the printed reading cannot see and the flag counts."
    )
    if it.sub_resolution_by_instance:
        top, count = max(it.sub_resolution_by_instance.items(), key=lambda kv: (kv[1], kv[0]))
        text += f" {count} of the {it.sub_resolution_new_best_rows} are on `{top}`."
    if it.most_steps_instance is not None:
        text += (
            "\n\nBut the incumbent trace cannot be read as pure search progress: "
            "`record_best` tightens the objective bound by `1e-3·(|obj|+1)` per accepted "
            "solution, so improvements are *floored* at roughly 0.1% steps. The measured "
            f"median consecutive-incumbent ratio on `{it.most_steps_instance}` (the instance "
            f"with the most improvements) is {it.most_steps_median_ratio:.7f} — "
            f"1 − {1 - it.most_steps_median_ratio:.4g}, against the bound step's 1 − 1e-3 — "
            f"and it takes {it.most_steps_incumbents - 1} such steps "
            f"({it.most_steps_incumbents} incumbents) to walk from "
            f"{_plain(float(f'{it.most_steps_first:.3g}'))} down to "
            f"{_plain(float(f'{it.most_steps_last:.3g}'))}."
        )
    return text


def _readme_infeasible(r: CampaignReport) -> str:
    c = r.results.counts
    return (
        f"Left infeasible in this table: {_and_list(_tick(c.infeasible_instances)) or 'none'} "
        f"({c.infeasible} of {c.roster})."
    )


def _readme_scip_verification(r: CampaignReport) -> str:
    n = r.head_to_head.scip.verification_failures
    return f"{_word(n, capital=True)} rows in this run failed that check."


def _readme_scip_read(r: CampaignReport) -> str:
    return (
        "SCIP's `.nl` reads total "
        f"{r.head_to_head.scip.total_read_seconds:.3f}s across the roster (`read_seconds`)."
    )


def _readme_hardware(r: CampaignReport) -> str:
    cb, sc = r.head_to_head.cbls, r.head_to_head.scip
    budget = r.provenance.budget_seconds
    if cb.hit_limit == cb.roster:
        cbls = (
            f"CBLS never terminates early — its {cb.total_wall_seconds:.0f}s is "
            f"{cb.roster} × {budget:g}s by construction, and is therefore independent of "
            "the machine entirely."
        )
    else:
        cbls = (
            f"CBLS stopped before its budget on {cb.roster - cb.hit_limit} instances; its "
            f"total is {cb.total_wall_seconds:.0f}s."
        )
    return "\n".join(
        [
            f"- The **wall-clock totals are the robust number.** {cbls} SCIP's "
            f"{sc.total_wall_seconds:.0f}s is dominated by proving optimality on "
            f"{sc.proved_optimal} instances and stopping, not by clock rate; even a 2x "
            f"hardware advantage would leave {sc.total_wall_seconds / 2:.0f}s against "
            f"{cb.total_wall_seconds:.0f}s.",
            "- The **counts are what a hardware difference would actually move.** Both",
            '  "feasible within the budget" and "proved optimal within the budget" scale with',
            "  machine speed, so those are the numbers a faster or slower box would change — in",
            "  either direction, for either solver.",
        ]
    )


def _readme_head_to_head(r: CampaignReport) -> str:
    cb, sc = r.head_to_head.cbls, r.head_to_head.scip
    budget = r.provenance.budget_seconds
    return "\n".join(
        [
            "| | CBLS | SCIP |",
            "|---|---|---|",
            f"| feasible | {cb.feasible} / {cb.roster} | **{sc.feasible} / {sc.roster}** |",
            f"| proved optimal | n/a (primal heuristic) | {sc.proved_optimal} / {sc.roster} |",
            f"| hit the {budget:g}s limit | {cb.hit_limit} | {sc.hit_limit} |",
            f"| total wall over the roster | {cb.total_wall_seconds:.0f}s | "
            f"{sc.total_wall_seconds:.0f}s (median {sc.median_wall_seconds:.2f}s; "
            f"{sc.under_one_second} instances under 1s) |",
            f"| integrality mismatches vs catalogue | {cb.integrality_mismatches} | "
            f"{sc.integrality_mismatches} |",
            f"| verification failures | {cb.verification_failures} | {sc.verification_failures} |",
        ]
    )


def _readme_disjoint(r: CampaignReport) -> str:
    h = r.head_to_head
    budget = r.provenance.budget_seconds
    cbls_failed = h.cbls.roster - h.cbls.feasible
    scip_rescues = [d for d in h.disjoint_failures if d.scip_feasible]
    quick = [d for d in scip_rescues if d.scip_status == "optimal" and d.scip_wall_seconds < 0.25]
    other_way = [d for d in h.disjoint_failures if d.cbls_feasible]
    howmany = (
        f"all {_word(cbls_failed)}"
        if len(scip_rescues) == cbls_failed
        else f"{_word(len(scip_rescues))} of the {_word(cbls_failed)}"
    )
    lead = (
        "**The failures are almost disjoint, and that is the useful part.** SCIP reaches a "
        f"feasible solution on {howmany} instances CBLS did not solve in this run"
    )
    if quick:
        lead += f" — {_word(len(quick))} of them proved optimal in under a quarter of a second"
    lead += "."
    if other_way:
        lead += (
            f" {_word(len(other_way), capital=True)} row{'s go' if len(other_way) > 1 else ' goes'}"
            " the other way: CBLS feasible where SCIP found no feasible solution in "
            f"{budget:g}s."
        )
    rows = ["| Instance | CBLS | SCIP |", "|---|---|---|"]
    for d in h.disjoint_failures:
        cbls_cell = (
            f"{_g(d.cbls_objective, 6)} ({d.cbls_gap_pct:.2f}% from BKS)"
            if d.cbls_feasible
            else "infeasible"
        )
        if d.scip_feasible:
            status = "proved optimal" if d.scip_status == "optimal" else d.scip_status
            scip_cell = (
                f"{_g(d.scip_objective, 6)} ({d.scip_gap_pct:.2f}% from BKS), {status} in "
                f"{d.scip_wall_seconds:.2f}s"
            )
        else:
            scip_cell = (
                f"**no feasible solution in {budget:g}s** ({d.scip_status}); dual bound "
                f"{_g(d.scip_dual_bound, 4)}"
            )
        rows.append(f"| `{d.instance}` | {cbls_cell} | {scip_cell} |")
    return "\n".join([lead, "", *rows])


def _readme_quality(r: CampaignReport) -> str:
    h = r.head_to_head
    return "\n".join(
        [
            "**Solution quality where both are feasible.** Buckets over the "
            f"{h.quality_denominator} instances that both solve and whose "
            f"`|BKS| >= {_plain(HEAD_TO_HEAD_MIN_ABS_BKS)}` (below that a percentage against the "
            "bound is not informative — see the zero-BKS discussion above), outside the "
            "documented failures:",
            "",
            "| | " + " | ".join(_thresholds(h.quality_thresholds_pct)) + " |",
            "|---|" + "|".join("---" for _ in h.quality_thresholds_pct) + "|",
            "| CBLS | " + " | ".join(str(n) for n in h.cbls_quality) + " |",
            "| SCIP | " + " | ".join(str(n) for n in h.scip_quality) + " |",
        ]
    )


def _readme_cbls_ahead(r: CampaignReport) -> str:
    ahead = r.head_to_head.cbls_ahead
    budget = r.provenance.budget_seconds
    if not ahead:
        return "No instance goes the other way by more than the claim band and the cell resolution."
    lead = (
        f"{_word(len(ahead), capital=True)} instances go the other way by a margin larger "
        "than both the claim band and the table's six-significant-digit resolution"
    )
    if all(a.scip_status == "timelimit" for a in ahead):
        lead += f", and every one is a row where SCIP exhausted the {budget:g}s budget"
    rows = ["| Instance | CBLS gap | SCIP gap | SCIP status |", "|---|---|---|---|"]
    for a in sorted(ahead, key=lambda a: -a.scip_gap_pct):
        cbls = "**matches BKS**" if a.cbls_verdict == "matches-bks" else _pct_cell(a.cbls_gap_pct)
        rows.append(f"| `{a.instance}` | {cbls} | {_pct_cell(a.scip_gap_pct)} | {a.scip_status} |")
    return "\n".join([lead + ":", "", *rows])


def _readme_free_variables(r: CampaignReport) -> str:
    fv = r.free_variables
    if fv is None:
        return "Not derivable: a roster `.nl` file is missing."
    return "\n".join(
        [
            "| | instances | eligible | within 10% |",
            "|---|---|---|---|",
            f"| ≥1 free variable | {fv.with_free} | {fv.with_free_eligible} | "
            f"{fv.with_free_within_10pct} |",
            f"| no free variables | {fv.without_free} | {fv.without_free_eligible} | "
            f"{fv.without_free_within_10pct} |",
            "",
            "Within 10% with at least one free variable: "
            f"{_and_list(_tick(fv.with_free_within_10pct_instances)) or 'none'}.",
            "",
            _free_variable_delta(fv),
        ]
    )


def _signed(n: int) -> str:
    return f"+{n}" if n > 0 else ("−" + str(-n) if n < 0 else "±0")


def _free_variable_delta(fv: FreeVariableSplit) -> str:
    """#107's effect: the generated AFTER column minus the fixed BEFORE constants."""
    misses = fv.with_free_eligible - FREE_VARIABLES_BEFORE_WITHIN_10PCT
    explained = fv.with_free_within_10pct - FREE_VARIABLES_BEFORE_WITHIN_10PCT
    joined = [
        n
        for n in fv.with_free_within_10pct_instances
        if n not in FREE_VARIABLES_BEFORE_WITHIN_10PCT_INSTANCES
    ]
    no_free = fv.without_free_within_10pct - NO_FREE_VARIABLES_BEFORE_WITHIN_10PCT
    before = _and_list(_tick(FREE_VARIABLES_BEFORE_WITHIN_10PCT_INSTANCES))
    return (
        "**Before #107** — a table that is not committed, so these two counts are fixed "
        "constants in `campaign_report.py` — the within-10% counts were "
        f"{FREE_VARIABLES_BEFORE_WITHIN_10PCT} ({before} only) and "
        f"{NO_FREE_VARIABLES_BEFORE_WITHIN_10PCT}. So #107 explains **{explained} of the "
        f"{misses}** free-variable misses — {_and_list(_tick(joined)) or 'none'} "
        f"join{'s' if len(joined) == 1 else ''} {before}. The no-free group's change is "
        f"{_signed(no_free)}."
    )


#: Block name -> renderer, in README order.
README_RENDERERS: dict[str, Callable[[CampaignReport], str]] = {
    "provenance": _readme_provenance,
    "tally": _readme_tally,
    "rule": _readme_rule,
    "gap-buckets": _readme_gap_buckets,
    "two-band": _readme_two_band,
    "anytime": _readme_anytime,
    "feasibility": _readme_feasibility,
    "improvement": _readme_improvement,
    "infeasible": _readme_infeasible,
    "scip-verification": _readme_scip_verification,
    "scip-read": _readme_scip_read,
    "hardware": _readme_hardware,
    "head-to-head": _readme_head_to_head,
    "disjoint": _readme_disjoint,
    "quality": _readme_quality,
    "cbls-ahead": _readme_cbls_ahead,
    "free-variables": _readme_free_variables,
}


def readme_blocks(report: CampaignReport) -> dict[str, str]:
    """Every README block's body, rendered from the report."""
    return {name: render(report) for name, render in README_RENDERERS.items()}


def apply_readme_blocks(text: str, blocks: Mapping[str, str]) -> str:
    """`text` with every block's body replaced by its rendering.

    Refuses a README missing a block, carrying a block twice, or carrying one this
    module does not render: each would let a derived number escape the check.
    """
    found = [m.group("name") for m in _README_BLOCK.finditer(text)]
    duplicates = sorted({n for n in found if found.count(n) > 1})
    unknown = sorted(set(found) - set(blocks))
    missing = sorted(set(blocks) - set(found))
    if duplicates or unknown or missing:
        raise ValueError(
            f"README blocks: duplicated {duplicates}, unknown {unknown}, missing {missing}"
        )

    def body(match: re.Match[str]) -> str:
        name = match.group("name")
        return (
            README_BEGIN.format(name=name)
            + "\n"
            + blocks[name]
            + "\n"
            + README_END.format(name=name)
        )

    return _README_BLOCK.sub(body, text)


def stale_readme_blocks(text: str, blocks: Mapping[str, str]) -> list[str]:
    """The names of the blocks whose committed body differs from the rendering."""
    current = {m.group("name"): m.group("body") for m in _README_BLOCK.finditer(text)}
    apply_readme_blocks(text, blocks)  # the structural refusals
    return [name for name in blocks if current[name] != blocks[name] + "\n"]


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="per-instance budget in seconds the campaign ran at; required when no "
        "comparison.run.json records it (every table published before #141)",
    )
    parser.add_argument("--seed", type=int, default=None, help="the campaign's seed, if known")
    parser.add_argument("--machine", default=None, help="the campaign's machine, if known")
    parser.add_argument("--feas-tol", type=float, default=None)
    parser.add_argument("--json", type=Path, default=None, help="write the summary JSON here")
    parser.add_argument("--markdown", type=Path, default=None, help="write the report here")
    parser.add_argument(
        "--seeds",
        action="store_true",
        help=f"print the multi-seed summary of {SEEDS_TABLE_NAME} instead (#141)",
    )
    readme = parser.add_mutually_exclusive_group()
    readme.add_argument(
        "--write-readme",
        type=Path,
        default=None,
        help="rewrite this README's campaign_report blocks from the tables",
    )
    readme.add_argument(
        "--check-readme",
        type=Path,
        default=None,
        help="exit 1 naming every campaign_report block in this README that is stale",
    )
    return parser.parse_args(argv)


def readme_write_refusal(report: CampaignReport) -> str | None:
    """Why the README may not be rewritten from this report, or None.

    A documented failure that came back feasible is a result to check (#110,
    #116), not a table refresh; and a budget the tables contradict would publish
    blocks computed over the wrong horizon. Both are refused, not warned about.
    """
    feasible = report.results.counts.documented_failures_feasible
    if feasible:
        return (
            f"documented failure(s) {', '.join(feasible)} came back FEASIBLE; check "
            "#110/#116 before publishing anything about them"
        )
    if report.provenance.warnings:
        return "; ".join(report.provenance.warnings)
    return None


def _print_warnings(report: CampaignReport) -> None:
    for warning in report.provenance.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for name in report.results.counts.documented_failures_feasible:
        print(
            f"WARNING: documented failure {name} came back FEASIBLE; check #110/#116 "
            "before publishing anything about it",
            file=sys.stderr,
        )


def _readme_action(args: argparse.Namespace, report: CampaignReport) -> int | None:
    """Run --check-readme / --write-readme; an exit status ends `main`, None continues."""
    blocks = readme_blocks(report)
    if args.check_readme is not None:
        stale = stale_readme_blocks(args.check_readme.read_text(), blocks)
        if stale:
            print(f"stale README blocks: {', '.join(stale)}", file=sys.stderr)
            return 1
        print("README blocks are current")
        return 0
    if args.write_readme is not None:
        refusal = readme_write_refusal(report)
        if refusal:
            print(f"refusing --write-readme: {refusal}", file=sys.stderr)
            return 2
        text = args.write_readme.read_text()
        args.write_readme.write_text(apply_readme_blocks(text, blocks))
        print(f"rewrote {len(blocks)} blocks in {args.write_readme}")
    return None


def _host(record: RunRecord) -> str | None:
    host = record.machine.get("host")
    return host if isinstance(host, str) else None


def seeds_report(inst_dir: Path, feas_tol: float | None) -> SeedsSummary:
    """`summarize_seeds` over the per-seed table at the latest-published seed's configuration.

    The commit, budget and host to aggregate at are those of the run record published
    most recently, so a table holding a stale campaign's seeds aggregates the
    current one and names the rest as left out.
    """
    table = inst_dir / SEEDS_TABLE_NAME
    records = load_seed_run_records(table)
    if not records:
        raise ValueError(f"{run_record_path(table)} not found: no seed has been published")
    latest = max(records.values(), key=lambda r: r.published_at)
    return summarize_seeds(
        load_seed_results(table, load_bounds_index(inst_dir / "bounds.csv")),
        records,
        commit=latest.commit,
        budget=latest.budget_seconds,
        host=_host(latest),
        feas_tol=DEFAULT_FEAS_TOL if feas_tol is None else feas_tol,
    )


def _usage_error(args: argparse.Namespace) -> str | None:
    if args.budget is not None and not (math.isfinite(args.budget) and args.budget > 0):
        return "--budget must be positive and finite"
    if args.feas_tol is not None and not (math.isfinite(args.feas_tol) and args.feas_tol > 0):
        return "--feas-tol must be positive and finite"
    if args.seeds:
        ignored = [
            flag
            for flag, value in (
                ("--budget", args.budget),
                ("--seed", args.seed),
                ("--machine", args.machine),
                ("--json", args.json),
                ("--markdown", args.markdown),
                ("--write-readme", args.write_readme),
                ("--check-readme", args.check_readme),
            )
            if value is not None
        ]
        if ignored:
            return (
                f"--seeds prints the multi-seed summary only; {', '.join(ignored)} would be ignored"
            )
    record = run_record_path(args.inst_dir / "comparison.csv")
    if not args.seeds and args.budget is None and not record.exists():
        return f"--budget is required: {record} does not exist to record the budget"
    return None


def _seeds_main(args: argparse.Namespace) -> int:
    try:
        print(render_seeds_text(seeds_report(args.inst_dir, args.feas_tol)))
    except (ValueError, KeyError, OSError) as exc:
        print(f"no multi-seed summary: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    error = _usage_error(args)
    if error:
        print(error, file=sys.stderr)
        return 2
    if args.seeds:
        return _seeds_main(args)
    report = build_report(
        args.inst_dir,
        budget=args.budget,
        seed=args.seed,
        machine=args.machine,
        feas_tol=args.feas_tol,
    )
    _print_warnings(report)
    status = _readme_action(args, report)
    if status is not None:
        return status
    markdown = render_markdown(report)
    if args.json is not None:
        args.json.write_text(to_json(report))
    if args.markdown is not None:
        args.markdown.write_text(markdown)
    if args.json is None and args.markdown is None and args.write_readme is None:
        sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
