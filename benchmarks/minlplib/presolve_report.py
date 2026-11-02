"""The #331 section of the presolve-322 campaign report: GM(gap+1) per arm.

#331 decides whether presolve and compound-move propagation go on by default.
`ablation_report.render_report` scores each arm in gap points per instance;
this section adds the measure the rest of #331 is read in (#256's GM(gap+1),
`gm_ratio.gm_gap_plus_one`) and the feasible count, per arm against the
same-sitting control, per seed and pooled over seeds.

THE MEASURE:

* gap+1 of a row is `max(gap_to_bks%, 0) + 1` (`gm_ratio.gap_plus_one`).
* At one seed, an arm's POOL is the instances feasible, with a finite gap, in
  both the arm and the control at that seed. The arm's GM and the control's GM
  are taken over that pool, so the ratio compares like with like; below 1 is
  the arm ahead. Pooled over seeds, the pool is every such (instance, seed).
* The feasible count is over every recorded row of that arm and seed.
* NOISE FLOOR: the control's GM per seed over ONE instance set -- the
  instances feasible with a finite gap in every arm at every seed -- and the
  floor is max / min of those per-seed GMs. One common set, because a per-seed
  set would let an instance an arm loses at one seed widen the floor, which is
  not seed noise (#350's round 2). A pooled ratio inside [1/floor, floor] is
  not a difference.
* COMPLETENESS: the rows alone cannot say what is missing, so the plan (arms,
  stamped seeds, stamped roster size) is handed in, and every missing
  (instance, arm, seed) is named. A report over a partial campaign says so
  first: #331's decision must not be read off one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarks.minlplib.ablation_report import CONTROL_ARM, RunRow
from benchmarks.minlplib.gm_ratio import gm_gap_plus_one

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The section's first line.
PRESOLVE_SECTION = "--- issue #331: GM(gap+1) per arm against the control ---"

#: How many missing triples are named before the rest are counted.
MISSING_NAMED = 10


@dataclass(frozen=True)
class PresolvePlan:
    """What the campaign was supposed to contain, handed in by the driver."""

    #: Every arm, control included, in run order.
    arms: tuple[str, ...]
    seeds: tuple[int, ...]
    #: The instances the campaign was planned over, where known (a run, or
    #: `--instances`); the report adds any instance the rows name.
    instances: tuple[str, ...]
    #: The roster size the stamp records, or None where there is no stamp.
    roster_size: int | None


def _scorable(row: RunRow) -> bool:
    """Feasible with a finite gap: a row GM(gap+1) can score."""
    return row.feasible and not row.no_search and math.isfinite(row.gap)


def _fmt(value: float, spec: str = ".4f") -> str:
    return format(value, spec) if math.isfinite(value) else "n/a"


def _ratio(arm: float, control: float) -> float:
    if math.isfinite(arm) and math.isfinite(control) and control > 0.0:
        return arm / control
    return math.nan


def missing_triples(rows: Sequence[RunRow], plan: PresolvePlan) -> list[tuple[str, str, int]]:
    """Every planned (instance, arm, seed) with no recorded row."""
    recorded = {(r.instance, r.arm, r.seed) for r in rows}
    instances = list(dict.fromkeys([*plan.instances, *(r.instance for r in rows)]))
    return [
        (instance, arm, seed)
        for instance in instances
        for seed in plan.seeds
        for arm in plan.arms
        if (instance, arm, seed) not in recorded
    ]


def _completeness_lines(rows: Sequence[RunRow], plan: PresolvePlan) -> list[str]:
    instances = {r.instance for r in rows} | set(plan.instances)
    missing = missing_triples(rows, plan)
    short = plan.roster_size is not None and len(instances) < plan.roster_size
    if not missing and not short:
        return [
            f"complete: {len(instances)} instance(s) x {len(plan.arms)} arm(s) x "
            f"{len(plan.seeds)} seed(s), every run recorded"
        ]
    lines = ["INCOMPLETE -- do not read #331's decision off this report:"]
    if short:
        lines.append(
            f"  the stamp records {plan.roster_size} instance(s); the rows name {len(instances)}"
        )
    if missing:
        named = ", ".join(f"{i}/{a}/seed{s}" for i, a, s in missing[:MISSING_NAMED])
        more = f" and {len(missing) - MISSING_NAMED} more" if len(missing) > MISSING_NAMED else ""
        lines.append(f"  {len(missing)} run(s) missing: {named}{more}")
    return lines


def presolve_lines(rows: Sequence[RunRow], plan: PresolvePlan) -> list[str]:
    """The #331 section, as report lines."""
    by_key = {(r.instance, r.arm, r.seed): r for r in rows}
    instances = list(dict.fromkeys([*plan.instances, *(r.instance for r in rows)]))
    arms = [a for a in plan.arms if a != CONTROL_ARM]

    def gap(instance: str, arm: str, seed: int) -> float | None:
        row = by_key.get((instance, arm, seed))
        return row.gap if row is not None and _scorable(row) else None

    def pool(arm: str, seed: int) -> list[str]:
        return [
            i
            for i in instances
            if gap(i, arm, seed) is not None and gap(i, CONTROL_ARM, seed) is not None
        ]

    def gm_over(arm: str, pairs: Sequence[tuple[str, int]]) -> float:
        values = [gap(i, arm, s) for i, s in pairs]
        return gm_gap_plus_one([v for v in values if v is not None])

    floor_set = [
        i for i in instances if all(gap(i, a, s) is not None for a in plan.arms for s in plan.seeds)
    ]
    floor_gm = {s: gm_over(CONTROL_ARM, [(i, s) for i in floor_set]) for s in plan.seeds}
    finite = [v for v in floor_gm.values() if math.isfinite(v) and v > 0.0]
    floor = max(finite) / min(finite) if len(finite) > 1 else math.nan

    lines = [
        PRESOLVE_SECTION,
        "gap+1 = max(gap_to_bks%, 0) + 1; GM over the instances feasible (finite gap) in BOTH the",
        "arm and the control at that seed; ratio = arm / control, BELOW 1 IS THE ARM AHEAD.",
        "Feasible counts are over every recorded row of that arm and seed.",
        *_completeness_lines(rows, plan),
        f"noise floor: control GM over the {len(floor_set)} instance(s) feasible in every arm at "
        f"every seed, max / min over seeds = {_fmt(floor)}"
        + (
            f"; a pooled ratio inside [{_fmt(1.0 / floor)}, {_fmt(floor)}] is not a difference"
            if math.isfinite(floor)
            else " (needs two seeds with a finite GM)"
        ),
        "",
        f"{'arm':<22}{'seed':>6}{'feasible':>10}{'ctl feas':>10}{'pool':>6}"
        f"{'GM arm':>10}{'GM ctl':>10}{'ratio':>9}",
    ]

    def feasible(arm: str, seed: int) -> str:
        recorded = [r for i in instances if (r := by_key.get((i, arm, seed))) is not None]
        return f"{sum(1 for r in recorded if r.feasible)}/{len(recorded)}"

    for arm in arms:
        for seed in plan.seeds:
            pairs = [(i, seed) for i in pool(arm, seed)]
            gm_arm, gm_ctl = gm_over(arm, pairs), gm_over(CONTROL_ARM, pairs)
            lines.append(
                f"{arm:<22}{seed:>6}{feasible(arm, seed):>10}{feasible(CONTROL_ARM, seed):>10}"
                f"{len(pairs):>6}{_fmt(gm_arm):>10}{_fmt(gm_ctl):>10}"
                f"{_fmt(_ratio(gm_arm, gm_ctl)):>9}"
            )
    lines += [
        "",
        "pooled over seeds:",
        f"{'arm':<22}{'feasible':>10}{'ctl feas':>10}{'pool':>6}{'GM arm':>10}{'GM ctl':>10}"
        f"{'ratio':>9}{'seed range':>17}  reading",
    ]
    for arm in arms:
        pairs = [(i, s) for s in plan.seeds for i in pool(arm, s)]
        gm_arm, gm_ctl = gm_over(arm, pairs), gm_over(CONTROL_ARM, pairs)
        ratio = _ratio(gm_arm, gm_ctl)
        per_seed = [
            _ratio(gm_over(arm, ps), gm_over(CONTROL_ARM, ps))
            for ps in ([(i, s) for i in pool(arm, s)] for s in plan.seeds)
        ]
        per_seed = [r for r in per_seed if math.isfinite(r)]
        spread = f"{min(per_seed):.4f}-{max(per_seed):.4f}" if per_seed else "n/a"
        lines.append(
            f"{arm:<22}{_count(rows, arm):>10}{_count(rows, CONTROL_ARM):>10}{len(pairs):>6}"
            f"{_fmt(gm_arm):>10}{_fmt(gm_ctl):>10}{_fmt(ratio):>9}{spread:>17}  "
            f"{_reading(ratio, floor)}"
        )
    return lines


def _count(rows: Sequence[RunRow], arm: str) -> str:
    recorded = [r for r in rows if r.arm == arm]
    return f"{sum(1 for r in recorded if r.feasible)}/{len(recorded)}"


def _reading(ratio: float, floor: float) -> str:
    """Where a pooled ratio sits against the floor, in words."""
    if not math.isfinite(ratio):
        return "no pool"
    if not math.isfinite(floor):
        return "no floor"
    if ratio < 1.0 / floor:
        return "arm ahead, outside the floor"
    if ratio > floor:
        return "arm behind, outside the floor"
    return "inside the floor"
