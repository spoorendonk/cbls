"""Score the MINLPLib ablation campaign of issue #143.

Reads the `results.csv` that `run_ablation.py` appends to and reports, for each
arm, its per-instance and aggregate effect against **the control from the same
sitting** -- never against the published table, which was measured at another
time on a differently-loaded machine.

THE NOISE FLOOR IS MEASURED, NOT ASSUMED. Issue #143 quotes a "3-4 gap point"
noise floor from an earlier campaign; nothing here uses that number. The only
measurement of run-to-run noise this campaign contains is the control's own
spread across its seeds, and that is what the floor is built from:

* For each instance whose control produced at least two feasible runs with a
  finite gap, `s_i` is the sample standard deviation (ddof=1) of those gaps.
* The effect statistic for an instance is a difference of per-arm mean gaps, so
  its noise scale is `se_i = s_i * sqrt(1/k_arm + 1/k_control)`, with `k` the
  number of comparable runs on each side.
* The **per-instance floor** is `NOISE_Z * se_i`, a two-sided ~95% normal band.
  At three seeds that band is coarse and this file says so rather than dressing
  it up as a t-interval it has no degrees of freedom for.
* The **aggregate floor** is the floor on the mean of the per-instance deltas:
  `NOISE_Z * sqrt(sum(se_i^2)) / n`.
* An instance whose control produced fewer than two comparable runs has no
  measured spread. Its `s_i` is imputed as the median `s_i` over the instances
  that do have one, and the report says how many were imputed -- dropping them
  instead would quietly shrink the floor by discarding the noisiest cases.

An effect whose magnitude is at or below its floor is reported as "inside the
noise", with the floor quoted, which is the acceptance criterion.

WHAT DOES NOT GET AVERAGED. A gap is defined only for a run that was feasible
AND has a finite published bound to be scored against, so an arm that is
feasible where the control is not has no delta at all -- there is no control gap
to subtract. Silently averaging that instance in (as a NaN, or by pretending the
missing side scored zero) is the failure this module is written to avoid: every
instance lands in exactly one of five buckets, the buckets are counted in the
report, and only `both-feasible` contributes to a mean.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarks.minlplib.run_benchmark import CLAIM_EXCLUDED

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

#: The arm every other arm is measured against.
CONTROL_ARM = "control"

#: The gate probe's rows. Same configuration as the control, run over the roster
#: hours earlier for the sole purpose of reading `lns_repairs`, so they are held
#: out of every effect estimate and reported on their own as a drift check.
PROBE_ARM_NAME = "gate-probe"

#: Two-sided ~95% normal band. Named so the report can quote it and a reader can
#: recompute the floor at another confidence without guessing what was used.
NOISE_Z = 2.0

#: Per-instance comparison buckets. Every compared instance lands in exactly one.
BOTH_FEASIBLE = "both-feasible"
ARM_ONLY_FEASIBLE = "arm-only-feasible"
CONTROL_ONLY_FEASIBLE = "control-only-feasible"
NEITHER_FEASIBLE = "neither-feasible"
NO_COMPARABLE_GAP = "no-comparable-gap"

BUCKETS = (
    BOTH_FEASIBLE,
    ARM_ONLY_FEASIBLE,
    CONTROL_ONLY_FEASIBLE,
    NEITHER_FEASIBLE,
    NO_COMPARABLE_GAP,
)


@dataclass(frozen=True)
class RunRow:
    """One completed solve, as `results.csv` records it."""

    instance: str
    arm: str
    seed: int
    feasible: bool
    gap: float
    lns_repairs: float
    wall: float


def _number(text: str | None) -> float:
    """A CSV cell as a float, with anything unparseable reading as NaN.

    "NaN" is what the runner writes for a cell it has no value for, and an empty
    cell means the same thing; neither may become a 0 that an aggregate would
    then treat as a measurement.
    """
    try:
        return float(text) if text is not None and text.strip() else math.nan
    except ValueError:
        return math.nan


def load_rows(path: Path) -> list[RunRow]:
    """Every row of a campaign's `results.csv`."""
    with path.open(newline="") as fh:
        return [
            RunRow(
                instance=row["instance"],
                arm=row["arm"],
                seed=int(row["seed"]),
                feasible=row["feasible"] == "true",
                gap=_number(row.get("gap_to_bks%")),
                lns_repairs=_number(row.get("lns_repairs")),
                wall=_number(row.get("wall_seconds")),
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
    #: Finite gaps of the feasible runs -- the only values a mean may be taken of.
    gaps: tuple[float, ...]
    repairs: tuple[float, ...]

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
            runs=len(group),
            feasible_runs=sum(1 for r in group if r.feasible),
            gaps=tuple(r.gap for r in group if r.feasible and math.isfinite(r.gap)),
            repairs=tuple(r.lns_repairs for r in group),
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
    #: Feasible-run count difference, which is a result in its own right on the
    #: instances where no gap comparison exists.
    feasibility_delta: int


def classify(control: Cell, treatment: Cell) -> str:
    """Which bucket an instance falls in for this arm."""
    if control.feasible_runs == 0 and treatment.feasible_runs == 0:
        return NEITHER_FEASIBLE
    if control.feasible_runs == 0:
        return ARM_ONLY_FEASIBLE
    if treatment.feasible_runs == 0:
        return CONTROL_ONLY_FEASIBLE
    if not control.gaps or not treatment.gaps:
        # Feasible on both sides, but at least one has no finite gap -- the
        # instance has no published bound, or the objective was not finite there.
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
        feasibility_delta=treatment.feasible_runs - control.feasible_runs,
    )


@dataclass(frozen=True)
class NoiseFloor:
    """The measured noise floor for one arm's comparison against the control."""

    per_instance: dict[str, float]
    aggregate: float
    #: Instances whose control supplied a real `s_i`, and those whose `s_i` was
    #: imputed from the median of those.
    measured: int
    imputed: int
    #: The typical per-instance floor -- the headline number to quote.
    median_floor: float
    #: The median control standard deviation the imputation used.
    median_spread: float


def control_spreads(
    cells: dict[tuple[str, str], Cell], instances: Sequence[str]
) -> dict[str, float]:
    """Per-instance control standard deviation, where two or more runs allow one.

    Computed over every instance the control covers, not only the compared ones,
    because the median of these is what an instance with too few control runs
    borrows.
    """
    spreads: dict[str, float] = {}
    for instance in instances:
        control = cells.get((instance, CONTROL_ARM))
        if control is not None and len(control.gaps) >= 2:
            spreads[instance] = statistics.stdev(control.gaps)
    return spreads


def noise_floor(comparisons: Sequence[Comparison], spreads: dict[str, float]) -> NoiseFloor:
    """The floor for this arm, from the control's own across-seed spread."""
    compared = [c for c in comparisons if c.bucket == BOTH_FEASIBLE]
    median_spread = statistics.median(spreads.values()) if spreads else math.nan
    per_instance: dict[str, float] = {}
    squares: list[float] = []
    measured = imputed = 0
    for comparison in compared:
        spread = spreads.get(comparison.instance, median_spread)
        if comparison.instance in spreads:
            measured += 1
        else:
            imputed += 1
        scale = math.sqrt(1.0 / len(comparison.treatment.gaps) + 1.0 / len(comparison.control.gaps))
        standard_error = spread * scale
        per_instance[comparison.instance] = NOISE_Z * standard_error
        squares.append(standard_error * standard_error)
    if not squares or math.isnan(median_spread):
        return NoiseFloor(per_instance, math.nan, measured, imputed, math.nan, median_spread)
    aggregate = NOISE_Z * math.sqrt(math.fsum(squares)) / len(squares)
    return NoiseFloor(
        per_instance=per_instance,
        aggregate=aggregate,
        measured=measured,
        imputed=imputed,
        median_floor=statistics.median(per_instance.values()),
        median_spread=median_spread,
    )


@dataclass(frozen=True)
class ArmSummary:
    """One arm's whole story: buckets, aggregate effect, floor and verdict."""

    arm: str
    comparisons: tuple[Comparison, ...]
    counts: dict[str, int]
    mean_delta: float | None
    median_delta: float | None
    feasibility_delta: int
    floor: NoiseFloor
    #: Scored instances with rows on only one side, so in no bucket at all. An
    #: interrupted campaign always ends mid-instance-block, so its last instance
    #: has control rows and not every arm's. It has to be visible: without it the
    #: bucket counts sit quietly short of the "instances scored" line above, and
    #: this module's own promise that every instance lands in exactly one bucket
    #: is false for precisely the instances a reader would want to ask about.
    uncompared: int = 0

    @property
    def verdict(self) -> str:
        if self.mean_delta is None:
            return "no instance is comparable on gap; read the feasibility counts instead"
        if math.isnan(self.floor.aggregate):
            return (
                f"mean gap delta {self.mean_delta:+.2f} points, but the control produced no "
                "instance with two comparable runs, so this campaign measured no noise floor "
                "and the effect cannot be called"
            )
        if abs(self.mean_delta) <= self.floor.aggregate:
            return (
                f"mean gap delta {self.mean_delta:+.2f} points is INSIDE THE NOISE "
                f"(measured floor +/-{self.floor.aggregate:.2f} points)"
            )
        direction = "WORSE than" if self.mean_delta > 0 else "BETTER than"
        return (
            f"mean gap delta {self.mean_delta:+.2f} points is outside the measured floor "
            f"(+/-{self.floor.aggregate:.2f}); the arm is {direction} the control"
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
    return ArmSummary(
        arm=arm,
        comparisons=tuple(comparisons),
        counts={bucket: sum(1 for c in comparisons if c.bucket == bucket) for bucket in BUCKETS},
        mean_delta=statistics.fmean(deltas) if deltas else None,
        median_delta=statistics.median(deltas) if deltas else None,
        feasibility_delta=sum(c.feasibility_delta for c in comparisons),
        floor=noise_floor(comparisons, control_spreads(cells, instances)),
        uncompared=uncompared,
    )


def scored_instances(rows: Sequence[RunRow]) -> list[str]:
    """Roster order, minus the instances published as documented failures.

    `elec25`/`elec50` are excluded from every quality claim under epic #87. They
    are reported separately rather than dropped: an arm that changes them is
    still a finding, just not one that can be folded into an aggregate.
    """
    seen: list[str] = []
    for row in rows:
        if row.instance not in seen and row.instance not in CLAIM_EXCLUDED:
            seen.append(row.instance)
    return seen


def drift_check(cells: dict[tuple[str, str], Cell], instances: Sequence[str]) -> str:
    """The probe against the campaign control -- one configuration, two times.

    This is the only direct measurement of machine drift the campaign contains,
    and it is why the probe's rows are kept rather than thrown away: the two sets
    are the same configuration run hours apart, so their disagreement is drift
    plus seed-free run-to-run noise, and it can be read against the same floor
    the arms are judged by.
    """
    pairs: list[float] = []
    for instance in instances:
        probe = cells.get((instance, PROBE_ARM_NAME))
        control = cells.get((instance, CONTROL_ARM))
        if probe is None or control is None or not probe.gaps or not control.gaps:
            continue
        control_mean = control.mean_gap
        probe_mean = probe.mean_gap
        if control_mean is None or probe_mean is None:
            continue
        pairs.append(abs(probe_mean - control_mean))
    if not pairs:
        return "drift check: no instance has both a probe row and a control row"
    return (
        f"drift check: {len(pairs)} instance(s) ran the control configuration twice at "
        f"different points in the sitting (gate probe vs campaign control). Mean absolute "
        f"gap difference "
        f"{statistics.fmean(pairs):.2f} points, median {statistics.median(pairs):.2f}. This is "
        "drift plus run-to-run noise for one configuration; an arm effect smaller than it is "
        "not an arm effect."
    )


def _repair_total(cells: Iterable[Cell]) -> float:
    """LNS destroy-repairs summed over cells.

    A NaN is skipped rather than read as zero: the runner writes it on a row
    where no solve completed, and "no reading" is not "no repairs" -- that is
    the same distinction the gate turns on.
    """
    return math.fsum(r for cell in cells for r in cell.repairs if math.isfinite(r))


def _instance_lines(summary: ArmSummary) -> list[str]:
    lines = [
        f"  {'instance':<22} {'ctl feas':>8} {'ctl gap':>9} {'arm feas':>8} "
        f"{'arm gap':>9} {'delta':>9} {'floor':>8}  bucket"
    ]
    for comparison in summary.comparisons:
        control_mean = comparison.control.mean_gap
        treatment_mean = comparison.treatment.mean_gap
        floor = summary.floor.per_instance.get(comparison.instance, math.nan)
        lines.append(
            f"  {comparison.instance:<22} "
            f"{comparison.control.feasible_runs:>3}/{comparison.control.runs:<4} "
            f"{'-' if control_mean is None else f'{control_mean:.2f}':>9} "
            f"{comparison.treatment.feasible_runs:>3}/{comparison.treatment.runs:<4} "
            f"{'-' if treatment_mean is None else f'{treatment_mean:.2f}':>9} "
            f"{'-' if comparison.delta is None else f'{comparison.delta:+.2f}':>9} "
            f"{'-' if math.isnan(floor) else f'{floor:.2f}':>8}  {comparison.bucket}"
        )
    return lines


def _floor_lines(summary: ArmSummary) -> list[str]:
    floor = summary.floor
    if math.isnan(floor.aggregate):
        return ["  noise floor: not measurable -- no instance has two comparable control runs"]
    return [
        f"  noise floor: typical per-instance +/-{floor.median_floor:.2f} points, "
        f"aggregate +/-{floor.aggregate:.2f} points",
        f"    from the control's own across-seed spread (median s = {floor.median_spread:.2f} "
        f"gap points, z = {NOISE_Z:g}); {floor.measured} instance(s) measured, "
        f"{floor.imputed} imputed from that median",
    ]


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
                f"gap {'-' if mean is None else f'{mean:.2f}'}"
            )
    return lines or ["  (no rows)"]


def render_report(results: Path, gate: dict[str, object] | None = None) -> str:
    """The whole campaign report, as text."""
    rows = load_rows(results)
    if not rows:
        return f"{results} holds no rows"
    cells = build_cells(rows)
    instances = scored_instances(rows)
    arms = [a for a in dict.fromkeys(r.arm for r in rows) if a not in (CONTROL_ARM, PROBE_ARM_NAME)]
    walls = [r.wall for r in rows if math.isfinite(r.wall)]
    mean_wall = statistics.fmean(walls) if walls else math.nan
    lines = [
        "=== MINLPLib ablation campaign (issue #143) ===",
        f"runs recorded:        {len(rows)}",
        f"instances scored:     {len(instances)} "
        f"(excluding {', '.join(CLAIM_EXCLUDED)}, published as documented failures)",
        f"seeds:                {sorted({r.seed for r in rows})}",
        f"arms:                 {', '.join([CONTROL_ARM, *arms])}",
        f"mean wall per run:    {mean_wall:.1f}s",
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
        summary = summarize_arm(arm, cells, instances)
        lines.append(f"--- {arm} ---")
        lines.append("  " + "  ".join(f"{k}={v}" for k, v in summary.counts.items()))
        lines.append(f"  feasible-run delta over the roster: {summary.feasibility_delta:+d}")
        # Issue #143 asks for the repair counts wherever the LNS arm is RUN, not
        # only for the reading that would justify skipping it. Both sides are
        # printed for every arm: the control's is the campaign's own answer to
        # "is LNS doing work at this budget", measured over all three seeds
        # rather than the gate probe's one.
        lines.append(
            "  LNS repairs over the roster: control "
            f"{_repair_total(c.control for c in summary.comparisons):g}, "
            f"arm {_repair_total(c.treatment for c in summary.comparisons):g}"
        )
        if summary.uncompared:
            lines.append(
                f"  {summary.uncompared} scored instance(s) have rows on only one side and are "
                "in no bucket -- the campaign is incomplete for this arm"
            )
        if summary.median_delta is not None:
            lines.append(f"  median per-instance gap delta: {summary.median_delta:+.2f} points")
        lines += _floor_lines(summary)
        lines.append(f"  VERDICT: {summary.verdict}")
        lines.append("")
        lines += _instance_lines(summary)
        lines.append("")
    lines.append("--- instances excluded from every claim ---")
    lines += _excluded_lines(cells, [CONTROL_ARM, *arms])
    return "\n".join(lines)
