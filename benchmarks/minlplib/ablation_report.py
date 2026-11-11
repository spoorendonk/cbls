"""Score the MINLPLib ablation campaign of issue #143.

Reads `run_ablation.py`'s `results.csv` and reports each arm's per-instance and
aggregate effect against **the control from the same sitting**, never the
published table, measured under other machine load.

THE NOISE FLOOR IS MEASURED from the control's across-seed spread, not the
"3-4 gap point" figure #143 quotes from an earlier campaign:

* `s_i`: sample standard deviation (ddof=1) of the control's gaps, for an
  instance with at least two feasible runs with a finite gap.
* The effect is a difference of per-arm mean gaps, so its scale is
  `se_i = s_i * sqrt(1/k_arm + 1/k_control)`, `k` the comparable runs per side.
* **Per-instance floor** `t * se_i`: a two-sided 95% band, `t` the Student
  multiplier at `k_control - 1` degrees of freedom (4.30 at three seeds, not
  1.96; 2.0 would print a band about half its nominal width).
* The band is RAISED, where below it, to the runner's tie band
  (`min_move_points`, the one unmeasured number here): a ~1e-7 control spread
  is not one a 1e-5 delta can be scored against. Raised instances are counted
  and disclosed on their own line. Per-instance, since a gap point is a
  percentage of the bound and so scales differently on every instance.
* Fewer than two comparable control runs, or an identical gap on every seed:
  no measured spread, so **not scored**, but counted and listed. Not imputed:
  gaps span six orders of magnitude, and the median spread (~1 gap point) on a
  ~1e6 gap would manufacture significance.

THE VERDICT IS SCALE-FREE. In the committed table `gear4` alone is 1.65e6 gap
points, 98.5% of the sum over all finite gaps (median instance 1.00), so an
unweighted mean delta is `gear4`'s seed draw: a uniform 10-point regression on
all 45 other instances moves it by less than its own noise. So:

* Each instance is judged against **its own** floor; `moved worse` and
  `moved better` count instances whose delta exceeds it.
* The counts get an exact two-sided **sign test**. Nothing outside the floor is
  "inside the noise" (typical floor quoted); comparable moves both ways are
  "mixed", not an effect.
* The **median** delta is quoted, never the mean. The mean is printed with the
  largest single instance's share of it.

WHAT DOES NOT GET AVERAGED. A gap needs a feasible run AND a finite published
bound, so an arm feasible where control is not has no delta. Every instance
lands in exactly one of six buckets, all counted; only `both-feasible` enters a
mean. `no-runs-recorded` holds a cell whose every row crashed or errored: its
`feasible_runs == 0` is not a measurement, so not a side losing feasibility.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from benchmarks.common.records import csv_number
from benchmarks.common.stats import sign_test_p, t975
from benchmarks.minlplib.runner import (
    CLAIM_EXCLUDED,
    RUNNER_FAILED_NOTE,
    completed_search,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

#: The arm every other arm is measured against.
CONTROL_ARM = "control"

#: The report's first line.
ABLATION_HEADING = "=== MINLPLib ablation campaign (issue #143) ==="

#: Labels `no_search_label` collapses a held-out row's known note to, in report
#: order; classification is `completed_search`'s alone. Any other note keeps its
#: text and is called unrecognised.
#:
#: `RUNNER_FAILED_NOTE` is first so `note_order` lists driver crashes ahead of
#: runner notes; `build_cells` filters those rows out before labelling. The
#: runner's five, by writer:
#:
#:   * `not-found`, `read-error`, `build-error`  -- `write_preread_row`, before
#:     a model exists;
#:   * `unsupported`/`unsupported: <reason>`     -- reader or adapter declined;
#:   * `solve-error`                             -- `cbls::solve` threw.
NO_SEARCH_NOTES: tuple[str, ...] = (
    RUNNER_FAILED_NOTE,
    "solve-error",
    "read-error",
    "build-error",
    "not-found",
    "unsupported",
)

#: Prefix for a note no list recognises; the third disclosure clause keys off it
#: and a test asserts on it.
UNRECOGNISED_NOTE = "unrecognised-note"

#: Characters of an unrecognised note the disclosure line reproduces: enough to
#: identify it, since `infeasible(...)` notes run to several hundred characters.
UNRECOGNISED_NOTE_CHARS = 60


def no_search_label(note: str) -> str:
    """A held-out row's note collapsed to the label the report lists it under.

    A known note collapses to its prefix, so `unsupported: <reason>` and
    `runner-failed-exit-139` share buckets; anything else keeps its text, marked
    unrecognised, so the reader sees what an unscored row said.
    """
    for prefix in NO_SEARCH_NOTES:
        if note.startswith(prefix):
            return prefix
    text = note.strip() or "<empty>"
    if len(text) > UNRECOGNISED_NOTE_CHARS:
        text = f"{text[:UNRECOGNISED_NOTE_CHARS]}..."
    return f"{UNRECOGNISED_NOTE}({text})"


def note_order(label: str) -> tuple[int, str]:
    """Sort key for the disclosure line: the known labels first, in list order."""
    if label in NO_SEARCH_NOTES:
        return (NO_SEARCH_NOTES.index(label), "")
    return (len(NO_SEARCH_NOTES), label)


#: The runner's own relative tie band: `classify_against_bks` (minlplib.cpp)
#: returns `matches-bks` within `1e-6 * (|primal_bks| + 1)` objective units.
#: Reused here, not reinvented; minlplib.cpp says why it is relative.
OBJ_TIE_RELATIVE = 1e-6

#: Where `safe_gap` stops dividing. Below it a "gap point" is a raw objective
#: residual rather than a percentage, so the floor is in objective units too.
GAP_ABSOLUTE_BRANCH = 1e-12

#: The tie band in GAP POINTS as |primal_bks| grows (its infimum), and the floor
#: when the bound cannot say better; see `min_move_points`.
MIN_MOVE_POINTS = 1e-4


def min_move_points(primal_bks: float) -> float:
    """The smallest gap-point delta that can be a move on this instance.

    The spread guard rejects only an EXACT zero, so a control at gap ~1e-7 on
    every seed gives a ~3.5e-7 band that a 1e-5 delta clears (#151). The floor
    is the runner's tie band in OBJECTIVE units pushed through `safe_gap`:
    `100*(obj-ref)/|ref|` for `|ref| >= GAP_ABSOLUTE_BRANCH`, so
    `OBJ_TIE_RELATIVE * (|ref| + 1)` becomes `1e-4 * (1 + 1/|ref|)` gap points,
    from the recorded `primal_bks` with no re-measurement.

    #151's alternatives fail: a relative-spread test passes ~1e-7 gaps with
    ~1e-7 spreads (ratio order 1); a floor at a multiple of the gap's numerical
    resolution is cleared, since ~1e-7 is ~1e16 ulps at ~1e-23 resolution. A
    flat `MIN_MOVE_POINTS` (the infimum, safe but loose below `|ref|` 1) is five
    orders of magnitude too permissive on `ex6_2_6`/`ex6_2_11` (`|primal_bks|`
    ~2.6e-6, tie band ~38 gap points, gaps -8.3e-5 and +9.5e-6).

    Missing or non-finite `primal_bks` falls back to the infimum: such a row has
    no finite gap and never reaches a floor, so it errs wide rather than raising.
    """
    if not math.isfinite(primal_bks):
        return MIN_MOVE_POINTS
    reference = abs(primal_bks)
    if reference < GAP_ABSOLUTE_BRANCH:
        # `safe_gap`'s absolute-residual branch: a gap point IS an objective
        # unit here, so the tie band needs no conversion.
        return OBJ_TIE_RELATIVE * (reference + 1.0)
    return 100.0 * OBJ_TIE_RELATIVE * (reference + 1.0) / reference


#: The gate probe: control's configuration, run hours earlier only to read
#: `lns_repairs`; held out of every effect estimate, reported as a drift check.
PROBE_ARM_NAME = "gate-probe"

#: Unanimous movers required to name a direction without a significant sign
#: test; below it movers are listed individually, not called an arm effect.
MIN_UNANIMOUS_MOVERS = 4


def format_points(value: float, *, signed: bool = True) -> str:
    """A gap-point quantity with enough digits to be visible.

    Two decimals from 0.1 up, three significant figures below, so a floor (1e-4
    and up) never prints as `0.00`. An exact zero, negative too, prints `+0.00`
    (`0.00` unsigned), so it cannot read as a direction.
    """
    if value == 0.0:
        return "+0.00" if signed else "0.00"
    if abs(value) >= 0.1:
        return f"{value:+.2f}" if signed else f"{value:.2f}"
    return f"{value:+.3g}" if signed else f"{value:.3g}"


#: Per-instance comparison buckets. Every compared instance lands in exactly one.
BOTH_FEASIBLE = "both-feasible"
ARM_ONLY_FEASIBLE = "arm-only-feasible"
CONTROL_ONLY_FEASIBLE = "control-only-feasible"
NEITHER_FEASIBLE = "neither-feasible"
NO_COMPARABLE_GAP = "no-comparable-gap"
#: One side has no completed run (every row crashed or carries a no-search or
#: unrecognised note): an absence of runs is not an absence of feasibility.
NO_RUNS_RECORDED = "no-runs-recorded"

BUCKETS = (
    BOTH_FEASIBLE,
    ARM_ONLY_FEASIBLE,
    CONTROL_ONLY_FEASIBLE,
    NEITHER_FEASIBLE,
    NO_COMPARABLE_GAP,
    NO_RUNS_RECORDED,
)


@dataclass(frozen=True)
class RunRow:
    """One completed solve, as `results.csv` records it."""

    instance: str
    arm: str
    seed: int
    feasible: bool
    gap: float
    #: Published best-known primal value behind this row's gap. Read only to
    #: size the floor, since `safe_gap` divides by it.
    primal_bks: float
    lns_repairs: float
    #: Repairs LNS KEPT (#150): attempts say LNS spent budget, this says whether
    #: it bought a better state. NaN means no reading (no solve ran).
    lns_repairs_accepted: float
    wall: float
    #: The note cell: the only thing telling a driver-written crash row
    #: (`feasible=false`) from an infeasible run.
    note: str = ""
    #: Budget and engine commit, so the report states what it measured and
    #: refuses a file mixing budgets.
    time_limit: float = math.nan
    commit: str = ""

    @property
    def runner_failed(self) -> bool:
        """Whether this row records a crashed process rather than a solve."""
        return self.note.startswith(RUNNER_FAILED_NOTE)

    @property
    def no_search(self) -> bool:
        """Whether NO SEARCH COMPLETED on this row, crash or exit-0 alike.

        A superset of `runner_failed`, counted apart but held out identically.
        Decided by the ALLOWLIST (#153), so an unknown note is held out and
        disclosed, not read as lost feasibility.
        """
        return not completed_search(self.note)


def load_rows(path: Path) -> list[RunRow]:
    """Every row of a campaign's `results.csv`."""
    with path.open(newline="") as fh:
        return [
            RunRow(
                instance=row["instance"],
                arm=row["arm"],
                seed=int(row["seed"]),
                feasible=row["feasible"] == "true",
                gap=csv_number(row.get("gap_to_bks%")),
                primal_bks=csv_number(row.get("primal_bks")),
                lns_repairs=csv_number(row.get("lns_repairs")),
                lns_repairs_accepted=csv_number(row.get("lns_repairs_accepted")),
                wall=csv_number(row.get("wall_seconds")),
                note=row.get("note") or "",
                time_limit=csv_number(row.get("time_limit")),
                commit=row.get("commit_sha") or "",
            )
            for row in csv.DictReader(fh)
        ]


@dataclass(frozen=True)
class Cell:
    """One arm's runs on one instance."""

    instance: str
    arm: str
    runs: int
    feasible_runs: int
    #: Finite gaps of the feasible runs: the only values a mean may be taken of.
    gaps: tuple[float, ...]
    repairs: tuple[float, ...]
    #: Crashed-process rows: excluded from `runs`/`feasible_runs`, surfaced in
    #: the report.
    failed_runs: int = 0
    #: Runner rows for a solve that never completed (`NO_SEARCH_NOTES` or an
    #: unrecognised note). Held out like `failed_runs`, counted apart.
    no_search_runs: int = 0
    #: Those rows' `no_search_label`s, in `note_order`: `not-found` and
    #: `unsupported` hit every arm alike, `solve-error` can be the arm's own.
    no_search_notes: tuple[str, ...] = ()
    #: The instance's published primal bound, for sizing the floor. NaN where no
    #: row recorded one.
    primal_bks: float = math.nan
    #: `repairs`' accepted halves, same order. Defaulted: only the repair line
    #: reads it, reporting an empty one as "no accepted reading", not zero
    #: (`_repair_cell`).
    repairs_accepted: tuple[float, ...] = ()
    #: Gaps keyed by seed, for the drift check's same-seed comparison (averaging
    #: seeds away measures seed variance). Defaulted: only that check reads it.
    seed_gaps: dict[int, float] = field(default_factory=dict)

    @property
    def mean_gap(self) -> float | None:
        return statistics.fmean(self.gaps) if self.gaps else None


def build_cells(rows: Iterable[RunRow]) -> dict[tuple[str, str], Cell]:
    """Group runs into `(instance, arm)` cells."""
    grouped: dict[tuple[str, str], list[RunRow]] = {}
    for row in rows:
        grouped.setdefault((row.instance, row.arm), []).append(row)
    return {
        key: Cell(
            instance=key[0],
            arm=key[1],
            # No-search rows (crashes, or exceptions via the failed-run path,
            # #153) are not runs, else they read as `control-only-feasible`.
            runs=sum(1 for r in group if not r.no_search),
            feasible_runs=sum(1 for r in group if r.feasible and not r.no_search),
            failed_runs=sum(1 for r in group if r.runner_failed),
            no_search_runs=sum(1 for r in group if r.no_search and not r.runner_failed),
            no_search_notes=tuple(
                sorted(
                    {no_search_label(r.note) for r in group if r.no_search and not r.runner_failed},
                    key=note_order,
                )
            ),
            primal_bks=next((r.primal_bks for r in group if math.isfinite(r.primal_bks)), math.nan),
            gaps=tuple(
                r.gap for r in group if r.feasible and not r.no_search and math.isfinite(r.gap)
            ),
            repairs=tuple(r.lns_repairs for r in group if not r.no_search),
            repairs_accepted=tuple(r.lns_repairs_accepted for r in group if not r.no_search),
            seed_gaps={
                r.seed: r.gap
                for r in group
                if r.feasible and not r.no_search and math.isfinite(r.gap)
            },
        )
        for key, group in grouped.items()
    }


@dataclass(frozen=True)
class Comparison:
    """One instance's verdict for one arm against the control."""

    instance: str
    arm: str
    bucket: str
    control: Cell
    treatment: Cell
    #: arm mean gap minus control mean gap, in gap points. POSITIVE MEANS WORSE:
    #: `gap_to_bks%` is a distance from the best known solution. None unless the
    #: bucket is `both-feasible`.
    delta: float | None
    #: Feasible-run count difference, a result where no gap comparison exists.
    #: **None where either side recorded no run**, as `classify` refuses.
    feasibility_delta: int | None


def classify(control: Cell, treatment: Cell) -> str:
    """Which bucket an instance falls in for this arm."""
    if control.runs == 0 or treatment.runs == 0:
        # NO RUNS IS NOT NO FEASIBILITY, so this test comes first: crashes on
        # either side would otherwise read as one side losing feasibility.
        return NO_RUNS_RECORDED
    if control.feasible_runs == 0 and treatment.feasible_runs == 0:
        return NEITHER_FEASIBLE
    if control.feasible_runs == 0:
        return ARM_ONLY_FEASIBLE
    if treatment.feasible_runs == 0:
        return CONTROL_ONLY_FEASIBLE
    if not control.gaps or not treatment.gaps:
        # Feasible both sides, but a side lacks a finite gap (no published
        # bound, or a non-finite objective).
        return NO_COMPARABLE_GAP
    return BOTH_FEASIBLE


def compare(control: Cell, treatment: Cell) -> Comparison:
    """One instance's comparison, with a delta only where one is defined."""
    bucket = classify(control, treatment)
    control_mean = control.mean_gap
    treatment_mean = treatment.mean_gap
    delta: float | None = None
    if bucket == BOTH_FEASIBLE and control_mean is not None and treatment_mean is not None:
        delta = treatment_mean - control_mean
    return Comparison(
        instance=control.instance,
        arm=treatment.arm,
        bucket=bucket,
        control=control,
        treatment=treatment,
        delta=delta,
        feasibility_delta=(
            None if bucket == NO_RUNS_RECORDED else treatment.feasible_runs - control.feasible_runs
        ),
    )


@dataclass(frozen=True)
class NoiseFloor:
    """The measured noise floor for one arm's comparison against the control."""

    per_instance: dict[str, float]
    #: Instances with a real control `s_i`, and those without one (NOT scored;
    #: module docstring).
    measured: int
    unmeasured: int
    #: Measured instances whose Student band was raised to `min_move_points`;
    #: reported, as their floor is not purely measured.
    floored: int
    #: The typical per-instance floor: the headline number to quote.
    median_floor: float
    #: Median control standard deviation over every instance, not only floored
    #: ones: the sitting's run-to-run noise.
    median_spread: float


def control_spreads(
    cells: dict[tuple[str, str], Cell], instances: Sequence[str]
) -> dict[str, float]:
    """Per-instance control standard deviation, where two or more runs allow one.

    Over every control instance, not only compared ones, so the quoted median
    describes the campaign. Never borrowed: an instance without one is not scored.
    """
    spreads: dict[str, float] = {}
    for instance in instances:
        control = cells.get((instance, CONTROL_ARM))
        if control is None or len(control.gaps) < 2:
            continue
        spread = statistics.stdev(control.gaps)
        # An exactly-zero spread is seeds landing on one value (normal for an
        # instance solved to its bound: the published table has four at gap 0
        # and around fourteen within 1e-7), not "no noise"; a 0.0 floor would
        # let any nonzero delta count. Unmeasured: counted, listed, not scored.
        if spread > 0.0:
            spreads[instance] = spread
    return spreads


def noise_floor(comparisons: Sequence[Comparison], spreads: dict[str, float]) -> NoiseFloor:
    """The floor for this arm, from the control's own across-seed spread.

    Per instance only: across six orders of magnitude the only aggregate worth
    quoting is a rank statistic, which has no gap-point floor.
    """
    compared = [c for c in comparisons if c.bucket == BOTH_FEASIBLE]
    per_instance: dict[str, float] = {}
    measured = unmeasured = floored = 0
    for comparison in compared:
        spread = spreads.get(comparison.instance)
        if spread is None:
            # No measured spread: counted and listed, never scored or imputed.
            unmeasured += 1
            continue
        measured += 1
        scale = math.sqrt(1.0 / len(comparison.treatment.gaps) + 1.0 / len(comparison.control.gaps))
        # Two-sided 95% Student multiplier at `k_control - 1` df (4.30 at three
        # seeds; 1.96 or 2.0 would print a band about half its nominal width).
        band = t975(len(comparison.control.gaps) - 1) * spread * scale
        # Raised to the runner's tie band: a finer spread is not meaningful in
        # the gap's units (`min_move_points`).
        reference = comparison.control.primal_bks
        if not math.isfinite(reference):
            reference = comparison.treatment.primal_bks
        smallest = min_move_points(reference)
        if band < smallest:
            floored += 1
            band = smallest
        per_instance[comparison.instance] = band
    return NoiseFloor(
        per_instance=per_instance,
        measured=measured,
        unmeasured=unmeasured,
        floored=floored,
        median_floor=statistics.median(per_instance.values()) if per_instance else math.nan,
        median_spread=statistics.median(spreads.values()) if spreads else math.nan,
    )


@dataclass(frozen=True)
class ArmSummary:
    """One arm's whole story: buckets, aggregate effect, floor and verdict."""

    arm: str
    comparisons: tuple[Comparison, ...]
    counts: dict[str, int]
    #: Printed beside `largest_delta`, never used for the verdict (module
    #: docstring: the mean is one instance's number).
    mean_delta: float | None
    #: The largest |delta| and its instance, beside the mean. NOT a share:
    #: `max|d| / sum|d|` reassures exactly when signs cancel (+100 and -99 give
    #: mean +0.5 and a "50%" share, though that instance is 200x the mean).
    largest_delta: float
    largest_delta_instance: str
    median_delta: float | None
    #: Median delta over the MOVERS, quoted with any named direction so it
    #: carries a nonzero number (every mover exceeds a floor of at least
    #: `MIN_MOVE_POINTS`); the roster median, possibly a true `+0.00`, is
    #: reported too.
    mover_median_delta: float | None
    #: Median of the MOVERS' OWN floors, beside the typical floor whenever
    #: anything moved: the two can differ by orders of magnitude (#151).
    mover_median_floor: float | None
    #: Instances whose delta exceeds THEIR OWN floor, by direction.
    moved_worse: int
    moved_better: int
    #: Exact two-sided sign-test p-value over those two counts.
    sign_p: float
    feasibility_delta: int
    #: Instances the balanced delta sums over, so `+0` reads as "over N".
    feasibility_balanced: int
    #: Feasible-run delta over equal-run-count instances, and how many are left
    #: out: an unequal-`k` one (interrupted campaign) is bookkeeping, not a result.
    feasibility_delta_balanced: int
    feasibility_unbalanced: int
    floor: NoiseFloor
    #: Scored instances with rows on only one side, so in no bucket (e.g. an
    #: interrupted campaign's last instance). Shown so bucket counts reconcile
    #: with "instances scored".
    uncompared: int = 0

    @property
    def comparable(self) -> int:
        """Instances with a delta: the denominator of `mean_delta`.

        `scored` is `median_delta`'s; both are printed so the two statistics'
        different sets are visible.
        """
        return sum(1 for c in self.comparisons if c.delta is not None)

    @property
    def scored(self) -> int:
        """Instances with both a delta and a measured floor to judge it by."""
        return self.moved_worse + self.moved_better + self.held

    @property
    def held(self) -> int:
        """Scored instances whose delta stayed inside their own floor."""
        return sum(
            1
            for c in self.comparisons
            if c.delta is not None
            and c.instance in self.floor.per_instance
            and abs(c.delta) <= self.floor.per_instance[c.instance]
        )

    @property
    def verdict(self) -> str:
        """The arm's effect, judged instance by instance and then by sign test.

        Never from the mean (module docstring).
        """
        comparable = sum(1 for c in self.comparisons if c.delta is not None)
        if comparable == 0:
            return "no instance is comparable on gap; read the feasibility counts instead"
        if not self.floor.per_instance:
            # Comparable, but no floor to judge by; "no instance is comparable"
            # would be false here.
            return (
                f"{comparable} instance(s) are comparable on gap, but none has a measurable "
                "noise floor -- every control either produced fewer than two comparable runs "
                "or returned an identical gap on every seed, which is not a measurement that "
                "there is no noise. The effect cannot be called"
            )
        if self.median_delta is None:
            return "no scored instance has a delta; read the buckets below"
        band = format_points(self.floor.median_floor, signed=False)
        typical = f"typical per-instance floor +/-{band} points"
        moved = self.moved_worse + self.moved_better
        if moved == 0:
            return (
                f"INSIDE THE NOISE: none of {self.scored} scored instance(s) moved outside its "
                f"own measured floor ({typical}); median gap delta over those {self.scored} "
                f"instance(s) {format_points(self.median_delta)} points"
            )
        # A named direction quotes the MOVERS' median: a verdict naming a
        # direction may not sit beside a roster median of +0.00 (#151), which is
        # still reported below with its denominator. The movers' median cannot
        # be zero: each |delta| exceeds a floor at least the tie band, and a
        # direction needs unanimity or a resolved sign test, so most share a sign.
        mover_delta = self.mover_median_delta
        summary = (
            f"{moved} of {self.scored} scored instance(s) moved outside their own floor "
            f"({self.moved_worse} worse, {self.moved_better} better; sign test p = "
            f"{self.sign_p:.3f}, {typical}); median gap delta over the {moved} mover(s) "
            f"{format_points(mover_delta if mover_delta is not None else 0.0)} points against "
            f"their own median floor "
            f"+/-{format_points(self.mover_median_floor or 0.0, signed=False)}; the other "
            f"{self.held} scored instance(s) held inside their own floor"
        )
        # A roster-level direction needs a significant sign test or
        # MIN_UNANIMOUS_MOVERS unanimous movers; those give p = 0.125, short of
        # significance, so the line says so.
        unanimous = self.moved_worse == 0 or self.moved_better == 0
        if not unanimous and self.sign_p > 0.05:
            return f"MIXED, no consistent direction: {summary}"
        if not unanimous or moved >= MIN_UNANIMOUS_MOVERS or self.sign_p <= 0.05:
            direction = "WORSE than" if self.moved_worse > self.moved_better else "BETTER than"
            qualifier = "" if self.sign_p <= 0.05 else " (unanimous, but too few movers for the "
            if qualifier:
                qualifier += "sign test to resolve)"
            return f"the arm is {direction} the control{qualifier}: {summary}"
        return (
            f"ISOLATED MOVERS, not a roster-level effect: {summary}. Fewer than "
            f"{MIN_UNANIMOUS_MOVERS} instances moved; read them individually below"
        )


def summarize_arm(
    arm: str, cells: dict[tuple[str, str], Cell], instances: Sequence[str]
) -> ArmSummary:
    """Compare one arm against the control over `instances`."""
    comparisons: list[Comparison] = []
    uncompared = 0
    for instance in instances:
        control = cells.get((instance, CONTROL_ARM))
        treatment = cells.get((instance, arm))
        if control is None or treatment is None:
            uncompared += 1
            continue
        comparisons.append(compare(control, treatment))
    deltas = [c.delta for c in comparisons if c.delta is not None]
    floor = noise_floor(comparisons, control_spreads(cells, instances))
    moved_worse = moved_better = 0
    for comparison in comparisons:
        band = floor.per_instance.get(comparison.instance)
        if comparison.delta is None or band is None:
            continue
        if comparison.delta > band:
            moved_worse += 1
        elif comparison.delta < -band:
            moved_better += 1
    # The mean is reported only beside the largest single delta.
    mean_delta = statistics.fmean(deltas) if deltas else None
    scored_deltas = [
        c.delta for c in comparisons if c.delta is not None and c.instance in floor.per_instance
    ]
    movers = [
        c
        for c in comparisons
        if c.delta is not None
        and c.instance in floor.per_instance
        and abs(c.delta) > floor.per_instance[c.instance]
    ]
    mover_deltas = [c.delta for c in movers if c.delta is not None]
    mover_floors = [floor.per_instance[c.instance] for c in movers]
    largest = max(
        ((abs(c.delta), c.instance) for c in comparisons if c.delta is not None),
        default=(math.nan, ""),
    )
    balanced = [
        c
        for c in comparisons
        if c.bucket != NO_RUNS_RECORDED
        and c.control.runs == c.treatment.runs
        and c.control.runs > 0
    ]
    return ArmSummary(
        arm=arm,
        comparisons=tuple(comparisons),
        counts={bucket: sum(1 for c in comparisons if c.bucket == bucket) for bucket in BUCKETS},
        mean_delta=mean_delta,
        largest_delta=largest[0],
        largest_delta_instance=largest[1],
        # Over SCORED instances only, matching the verdict's "N scored" wording.
        median_delta=statistics.median(scored_deltas) if scored_deltas else None,
        mover_median_delta=statistics.median(mover_deltas) if mover_deltas else None,
        mover_median_floor=statistics.median(mover_floors) if mover_floors else None,
        moved_worse=moved_worse,
        moved_better=moved_better,
        sign_p=sign_test_p(moved_worse, moved_better),
        # No completed run on one side: no feasibility comparison either way.
        feasibility_delta=sum(
            c.feasibility_delta for c in comparisons if c.feasibility_delta is not None
        ),
        feasibility_delta_balanced=sum(
            c.feasibility_delta for c in balanced if c.feasibility_delta is not None
        ),
        feasibility_balanced=len(balanced),
        feasibility_unbalanced=sum(1 for c in comparisons if c.bucket != NO_RUNS_RECORDED)
        - len(balanced),
        floor=floor,
        uncompared=uncompared,
    )


def scored_instances(rows: Sequence[RunRow]) -> list[str]:
    """Roster order, minus the instances published as documented failures.

    `elec25`/`elec50` are excluded from quality claims (#87) and reported
    separately: a change there is a finding, not an aggregate input.
    """
    seen: list[str] = []
    for row in rows:
        if row.instance not in seen and row.instance not in CLAIM_EXCLUDED:
            seen.append(row.instance)
    return seen


def drift_check(cells: dict[tuple[str, str], Cell], instances: Sequence[str]) -> str:
    """The probe against the campaign control -- one configuration, two times.

    The campaign's only direct drift measurement: one configuration hours
    apart, so the disagreement is drift plus seed-free noise, read against the
    arms' floor.
    """
    pairs: list[float] = []
    for instance in instances:
        probe = cells.get((instance, PROBE_ARM_NAME))
        control = cells.get((instance, CONTROL_ARM))
        if probe is None or control is None:
            continue
        # SAME SEED ON BOTH SIDES: against the control's three-seed mean the
        # one-seed probe differs by ~s*sqrt(2/3) from seeds alone, not drift.
        shared = set(probe.seed_gaps) & set(control.seed_gaps)
        for seed in sorted(shared):
            pairs.append(abs(probe.seed_gaps[seed] - control.seed_gaps[seed]))
    if not pairs:
        return (
            "drift check: no instance has a probe row and a control row at the same seed, "
            "so this campaign measured no drift"
        )
    return (
        f"drift check: {len(pairs)} same-seed pair(s) ran the control configuration twice at "
        f"different points in the sitting (gate probe vs campaign control). Mean absolute gap "
        f"difference {format_points(statistics.fmean(pairs), signed=False)} points, median "
        f"{format_points(statistics.median(pairs), signed=False)}, "
        f"max {format_points(max(pairs), signed=False)}. Same configuration, same seed, "
        "hours apart: the difference is machine drift. It is descriptive -- no verdict above "
        "is computed from it."
    )


def _repair_totals(cells: Iterable[Cell]) -> tuple[float, float]:
    """LNS destroy-repairs summed over cells: attempted, then accepted.

    NaN (no solve completed) is skipped, not read as zero, in each sum
    independently: dropping a row from both would turn a missing COLUMN into
    "0 attempted, 0 accepted". Accepted could then exceed attempted only on a
    row with one reading, which `LnsCells` never writes (both cells or
    neither). A side with no reading totals NaN, not `math.fsum(())`'s 0.0.
    """
    materialised = list(cells)
    attempted = [r for cell in materialised for r in cell.repairs if math.isfinite(r)]
    accepted = [r for cell in materialised for r in cell.repairs_accepted if math.isfinite(r)]
    return (
        math.fsum(attempted) if attempted else math.nan,
        math.fsum(accepted) if accepted else math.nan,
    )


def _repair_cell(total: float, noun: str) -> str:
    """One half of the repair line, or the fact that nothing measured it.

    NaN is not zero: no row carried that counter (a campaign predating the
    column, re-scored by `--report-only`; the two columns arrived at different
    times, #150). 0 would claim "LNS repaired/kept nothing" unmeasured.
    """
    return f"no {noun} reading" if math.isnan(total) else f"{total:g} {noun}"


def _instance_lines(summary: ArmSummary) -> list[str]:
    lines = [
        f"  {'instance':<22} {'ctl feas':>8} {'ctl gap':>11} {'arm feas':>8} "
        f"{'arm gap':>11} {'delta':>11} {'floor':>10}  bucket"
    ]
    for comparison in summary.comparisons:
        control_mean = comparison.control.mean_gap
        treatment_mean = comparison.treatment.mean_gap
        floor = summary.floor.per_instance.get(comparison.instance, math.nan)
        lines.append(
            f"  {comparison.instance:<22} "
            f"{comparison.control.feasible_runs:>3}/{comparison.control.runs:<4} "
            f"{'-' if control_mean is None else format_points(control_mean, signed=False):>11} "
            f"{comparison.treatment.feasible_runs:>3}/{comparison.treatment.runs:<4} "
            f"{'-' if treatment_mean is None else format_points(treatment_mean, signed=False):>11} "
            f"{'-' if comparison.delta is None else format_points(comparison.delta):>11} "
            f"{'-' if math.isnan(floor) else format_points(floor, signed=False):>10}  "
            f"{comparison.bucket}"
        )
    return lines


def _floor_lines(summary: ArmSummary) -> list[str]:
    floor = summary.floor
    if not floor.per_instance:
        # Both reasons, matching the verdict: fewer than two comparable runs OR
        # an identical gap on every seed.
        return [
            "  noise floor: not measurable -- every control either produced fewer than two "
            "comparable runs or returned an identical gap on every seed"
        ]
    lines = [
        f"  noise floor: typical per-instance "
        f"+/-{format_points(floor.median_floor, signed=False)} points",
        "    from the control's own across-seed spread (median s = "
        f"{format_points(floor.median_spread, signed=False)} "
        f"gap points, two-sided 95% Student band); {floor.measured} instance(s) measured",
    ]
    if floor.floored:
        lines.append(
            f"    {floor.floored} instance(s) measured a band finer than the runner's own tie "
            "band for their published bound, and were raised to it: a control that returned "
            "~1e-7 on every seed measures a spread, but not one the gap it bounds can resolve. "
            "That bound is the only assumed number here -- see `min_move_points`"
        )
    if floor.unmeasured:
        lines.append(
            f"    {floor.unmeasured} comparable instance(s) NOT SCORED: their control either "
            "produced fewer than two comparable runs or returned an identical gap on every "
            "seed, so no floor could be measured. Their deltas are listed below with a '-' "
            "floor. Imputing one from other instances would be unsound here -- the roster's "
            "gaps span six orders of magnitude, so a borrowed ~1-point spread lent to a "
            "1e6-point instance is a floor it clears automatically."
        )
    return lines


def _excluded_lines(cells: dict[tuple[str, str], Cell], arms: Sequence[str]) -> list[str]:
    lines: list[str] = []
    for instance in CLAIM_EXCLUDED:
        for arm in arms:
            cell = cells.get((instance, arm))
            if cell is None:
                continue
            mean = cell.mean_gap
            lines.append(
                f"  {instance:<10} {arm:<16} feasible {cell.feasible_runs}/{cell.runs}  "
                f"gap {'-' if mean is None else format_points(mean, signed=False)}"
            )
    return lines or ["  (no rows)"]


def _arm_section(summary: ArmSummary) -> list[str]:
    """One arm's block of the report: counts, held-out rows, aggregates, floor, verdict."""
    lines: list[str] = []
    lines.append(f"--- {summary.arm} ---")
    lines.append("  " + "  ".join(f"{k}={v}" for k, v in summary.counts.items()))
    # The denominator is printed even at zero: "+0 over 0 instances" is an
    # empty sum, not a measurement that the arm changed nothing.
    lines.append(
        f"  feasible-run delta over the roster: {summary.feasibility_delta_balanced:+d} "
        f"over the {summary.feasibility_balanced} instance(s) with equal run counts on "
        "both sides"
        + (
            ""
            if not summary.feasibility_unbalanced
            else f" ({summary.feasibility_unbalanced} instance(s) left out for unequal run "
            f"counts; counting them gives {summary.feasibility_delta:+d}, which is "
            "bookkeeping on run counts that do not correspond -- either an incomplete "
            "campaign or rows held out above -- rather than a result)"
        )
    )
    # Repair counts for both sides of every arm (#143); the control's answers
    # "is LNS working at this budget" over all seeds. Accepted (#150) separates
    # working from spending; reported, not gated on (run_ablation.py's
    # LNS_GATE_MIN_REPAIRS).
    control_attempted, control_accepted = _repair_totals(c.control for c in summary.comparisons)
    arm_attempted, arm_accepted = _repair_totals(c.treatment for c in summary.comparisons)
    lines.append(
        "  LNS repairs over the roster: control "
        f"{_repair_cell(control_attempted, 'attempted')}, "
        f"{_repair_cell(control_accepted, 'accepted')}; "
        f"arm {_repair_cell(arm_attempted, 'attempted')}, "
        f"{_repair_cell(arm_accepted, 'accepted')}"
    )
    # SPLIT BY SIDE: whether a solve crashes can depend on the configuration,
    # which a total would hide.
    crashed_control = sum(c.control.failed_runs for c in summary.comparisons)
    crashed_arm = sum(c.treatment.failed_runs for c in summary.comparisons)
    if crashed_control or crashed_arm:
        lines.append(
            f"  {crashed_control + crashed_arm} run(s) crashed (control {crashed_control}, "
            f"arm {crashed_arm}) and are held out of every count above -- they record no "
            "measurement, so reading them as infeasible would score a process failure as an "
            "arm losing feasibility"
        )
    unsearched_control = sum(c.control.no_search_runs for c in summary.comparisons)
    unsearched_arm = sum(c.treatment.no_search_runs for c in summary.comparisons)
    observed = sorted(
        {
            note
            for c in summary.comparisons
            for note in c.control.no_search_notes + c.treatment.no_search_notes
        },
        key=note_order,
    )
    if unsearched_control or unsearched_arm:
        # The justification differs by note: `solve-error` can depend on the
        # arm; `not-found`/`read-error`/`build-error`/`unsupported` are roster
        # problems hitting every arm alike.
        arm_dependent = [n for n in observed if n == "solve-error"]
        unrecognised = [n for n in observed if n.startswith(UNRECOGNISED_NOTE)]
        roster_wide = [
            n for n in observed if n != "solve-error" and not n.startswith(UNRECOGNISED_NOTE)
        ]
        why = "; ".join(
            part
            for part in (
                "whether a solve throws can depend on the arm, so scoring "
                f"{', '.join(arm_dependent)} would read an exception as a lost feasibility"
                if arm_dependent
                else "",
                f"{', '.join(roster_wide)} is a roster problem that hits every arm alike, "
                "so scoring it would read a missing or unsupported instance as one"
                if roster_wide
                else "",
                # An unknown note is held out and SAID, not scored as
                # infeasible from its `feasible=false` cell.
                f"{', '.join(unrecognised)} is a note this report does not recognise, and it "
                "scores only the notes a completed search is known to write -- so an "
                "unfamiliar one is held out and named here rather than counted"
                if unrecognised
                else "",
            )
            if part
        )
        # "not scored", not "completed no search": an UNRECOGNISED-note row may
        # have completed a search the allowlist does not yet know.
        lines.append(
            f"  {unsearched_control + unsearched_arm} run(s) not scored "
            f"(control {unsearched_control}, arm {unsearched_arm}; {', '.join(observed)}) and "
            "are held out of every count above. The row is well-formed and "
            f"`feasible=false`, so nothing but the note says so -- and {why}"
        )
    held_out = [c.instance for c in summary.comparisons if c.bucket == NO_RUNS_RECORDED]
    if held_out:
        lines.append(
            f"  {len(held_out)} instance(s) recorded no completed run on at least one side "
            f"and are in no feasibility bucket and in no feasible-run delta: "
            f"{', '.join(held_out)}"
        )
    if summary.uncompared:
        lines.append(
            f"  {summary.uncompared} scored instance(s) have rows on only one side and are "
            "in no bucket -- the campaign is incomplete for this arm"
        )
    if summary.median_delta is not None:
        lines.append(
            f"  median per-instance gap delta: {format_points(summary.median_delta)} points "
            f"over the {summary.scored} SCORED instance(s)"
        )
    if summary.mean_delta is not None:
        largest = (
            ""
            if math.isnan(summary.largest_delta)
            else "; largest single-instance |delta| "
            f"{format_points(summary.largest_delta, signed=False)} points "
            f"({summary.largest_delta_instance})"
        )
        lines.append(
            f"  mean per-instance gap delta: {format_points(summary.mean_delta)} points over "
            f"the {summary.comparable} COMPARABLE instance(s){largest} "
            "-- NOT the verdict statistic, see the module docstring"
        )
    lines += _floor_lines(summary)
    lines.append(f"  VERDICT: {summary.verdict}")
    lines.append("")
    lines += _instance_lines(summary)
    lines.append("")
    return lines


def render_report(
    results: Path,
    gate: dict[str, object] | None = None,
) -> str:
    """The whole campaign report, as text.

    A file mixing budgets is refused: it would compare budgets, not only arms.
    """
    rows = load_rows(results)
    if not rows:
        return f"{results} holds no rows"
    budgets = sorted({r.time_limit for r in rows if math.isfinite(r.time_limit)})
    commits = sorted({r.commit for r in rows if r.commit})
    # A row with no recorded budget counts as a budget of its own.
    unrecorded = any(not math.isfinite(r.time_limit) for r in rows)
    if len(budgets) > 1 or (budgets and unrecorded):
        named = [f"{b:g}s" for b in budgets] + (["not recorded"] if unrecorded else [])
        return (
            f"{results} mixes budgets {', '.join(named)}; refusing to "
            "score it -- an arm compared against a control run at another budget measures the "
            "budget. Split the rows by time_limit and score each file on its own."
        )
    cells = build_cells(rows)
    instances = scored_instances(rows)
    # Name only the documented failures present; a subset run may have neither.
    present = {r.instance for r in rows}
    excluded = [name for name in CLAIM_EXCLUDED if name in present]
    arms = [a for a in dict.fromkeys(r.arm for r in rows) if a not in (CONTROL_ARM, PROBE_ARM_NAME)]
    # No-search rows carry a 0.0 wall (`write_preread_row`, `solve-error`'s
    # `write_unsolved_row`), which would drag the mean toward zero.
    walls = [r.wall for r in rows if math.isfinite(r.wall) and not r.no_search]
    mean_wall = statistics.fmean(walls) if walls else math.nan
    lines = [
        ABLATION_HEADING,
        f"rows recorded:        {len(rows)} "
        f"({sum(1 for r in rows if not r.no_search)} completed a search)",
        f"instances scored:     {len(instances)}"
        + (
            f" (excluding {', '.join(excluded)}, published as documented failures)"
            if excluded
            else ""
        ),
        f"seeds:                {sorted({r.seed for r in rows})}",
        "budget (time_limit):  "
        + (", ".join(f"{b:g}s" for b in budgets) if budgets else "not recorded"),
        f"engine commit(s):     {', '.join(commits) if commits else 'not recorded'}",
        f"arms:                 {', '.join([CONTROL_ARM, *arms])}",
        "mean wall per run:    "
        + (
            f"{mean_wall:.1f}s"
            if math.isfinite(mean_wall)
            else "not measured (no row completed a search)"
        ),
        "",
        "Sign convention: delta = arm mean gap-to-BKS minus control mean gap-to-BKS,",
        "in gap points. POSITIVE IS WORSE. Only `both-feasible` instances have one.",
        "",
    ]
    if gate is not None:
        lines += [
            f"LNS gate: {'ran the arm' if gate.get('run_arm') else 'skipped the arm'}",
            f"  {gate.get('reason')}",
            "",
        ]
    lines += [drift_check(cells, instances), ""]
    for arm in arms:
        lines += _arm_section(summarize_arm(arm, cells, instances))
    lines.append(
        "NOTE ON UNITS: `gap_to_bks%` is a percentage where the reference value is nonzero "
        "and an ABSOLUTE residual where it is not (see the runner's safe_gap). Per-instance "
        "floors and per-instance deltas are unit-consistent, so the moved/held verdict above "
        "is sound; the roster-level median and mean pool the two and are quoted as 'points' "
        "for want of a better word."
    )
    lines.append("")
    lines.append("--- instances excluded from every claim ---")
    lines += _excluded_lines(cells, [CONTROL_ARM, *arms])
    return "\n".join(lines)
