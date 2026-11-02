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
* The feasible count is over the MEASURED runs of that arm and seed: a row
  with no completed search (`RunRow.no_search`: a crash, a held-out check) is
  not a lost feasibility, so it is out of the count, as it is out of the
  shared scoring above, and named under completeness instead.
* `runner.CLAIM_EXCLUDED` instances are out of every count and aggregate here
  too (`gm_ratio`'s rule).
* NOISE FLOOR: the control's GM per seed over ONE instance set -- the
  instances feasible with a finite gap in every arm at every seed -- and the
  floor is max / min of those per-seed GMs. One common set, because a per-seed
  set would let an instance an arm loses at one seed widen the floor, which is
  not seed noise (#350's round 2). A pooled ratio inside [1/floor, floor] is
  not a difference.
* COMPLETENESS: the rows alone cannot say what is missing, so the plan (arms,
  stamped seeds, stamped roster size, and the instances where they are known:
  `--instances`, or the committed sample whose hash the stamp records) is
  handed in. Named under INCOMPLETE, right under the heading: every missing
  (instance, arm, seed), every one recorded without a completed search (the
  driver records a crash, so a resume does not retry it), every one recorded
  twice, any instance outside the plan, and an instance count other than the
  stamp's. #331's decision must not be read off a partial campaign.
* A floor of exactly 1 is no seed noise measured, and is read as such.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from benchmarks.minlplib.ablation_report import CONTROL_ARM, RunRow
from benchmarks.minlplib.gm_ratio import gm_gap_plus_one
from benchmarks.minlplib.runner import CLAIM_EXCLUDED

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


def _fmt(value: float) -> str:
    """At most 10 characters: fixed point where it is readable, else scientific.

    GMs of gap+1 span orders of magnitude (a 1e9-point gap is on the roster),
    and a ratio of 1e-5 printed fixed would read as 0.
    """
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}" if 1e-3 <= abs(value) < 1e4 else f"{value:.3e}"


def _ratio(arm: float, control: float) -> float:
    if math.isfinite(arm) and math.isfinite(control) and control > 0.0:
        return arm / control
    return math.nan


Triple = tuple[str, str, int]


def _all_names(rows: Sequence[RunRow], plan: PresolvePlan) -> list[str]:
    """Every instance planned or recorded, in plan order, claim-excluded included."""
    return list(dict.fromkeys([*plan.instances, *(r.instance for r in rows)]))


def _instances(rows: Sequence[RunRow], plan: PresolvePlan) -> list[str]:
    """The instances the section scores: `_all_names` less `CLAIM_EXCLUDED`."""
    return [name for name in _all_names(rows, plan) if name not in CLAIM_EXCLUDED]


def _planned(rows: Sequence[RunRow], plan: PresolvePlan) -> list[Triple]:
    return [
        (instance, arm, seed)
        for instance in _instances(rows, plan)
        for seed in plan.seeds
        for arm in plan.arms
    ]


def _by_key(rows: Sequence[RunRow]) -> dict[Triple, RunRow]:
    return {(r.instance, r.arm, r.seed): r for r in rows}


def missing_triples(rows: Sequence[RunRow], plan: PresolvePlan) -> list[Triple]:
    """Every planned (instance, arm, seed) with no recorded row."""
    recorded = _by_key(rows)
    return [t for t in _planned(rows, plan) if t not in recorded]


def unmeasured_triples(rows: Sequence[RunRow], plan: PresolvePlan) -> list[Triple]:
    """Every planned (instance, arm, seed) whose row records no completed search."""
    recorded = _by_key(rows)
    return [t for t in _planned(rows, plan) if t in recorded and recorded[t].no_search]


def _named(triples: Sequence[Triple]) -> str:
    named = ", ".join(f"{i}/{a}/seed{s}" for i, a, s in triples[:MISSING_NAMED])
    more = f" and {len(triples) - MISSING_NAMED} more" if len(triples) > MISSING_NAMED else ""
    return f"{named}{more}"


def _completeness_lines(rows: Sequence[RunRow], plan: PresolvePlan) -> list[str]:
    # The stamp's count includes any claim-excluded instance, so it is checked
    # against every name, not the scored ones.
    present = len(_all_names(rows, plan))
    missing = missing_triples(rows, plan)
    unmeasured = unmeasured_triples(rows, plan)
    # Counted both ways: a stray instance can stand in for a missing one, and
    # where the plan names its instances the stray is named too.
    miscounted = plan.roster_size is not None and present != plan.roster_size
    outside = sorted({r.instance for r in rows} - set(plan.instances)) if plan.instances else []
    # The shared scoring reads every row; this section, one per triple. A
    # repeated triple (a hand merge) would make the two disagree silently.
    counts = Counter((r.instance, r.arm, r.seed) for r in rows)
    repeated = sorted(t for t, n in counts.items() if n > 1)
    if not (missing or unmeasured or miscounted or outside or repeated):
        return [
            f"complete: {len(_instances(rows, plan))} instance(s) x {len(plan.arms)} arm(s) x "
            f"{len(plan.seeds)} seed(s), every run recorded once with a completed search"
        ]
    lines = ["INCOMPLETE -- do not read #331's decision off this report:"]
    if miscounted:
        lines.append(f"  the stamp records {plan.roster_size} instance(s); the rows name {present}")
    if outside:
        lines.append(f"  instance(s) outside the planned roster: {', '.join(outside)}")
    if repeated:
        lines.append(f"  {len(repeated)} run(s) recorded more than once: {_named(repeated)}")
    if missing:
        lines.append(f"  {len(missing)} run(s) missing: {_named(missing)}")
    if unmeasured:
        lines.append(
            f"  {len(unmeasured)} run(s) recorded without a completed search (crashed or "
            f"held out; a resume does not retry them): {_named(unmeasured)}"
        )
    return lines


def presolve_lines(rows: Sequence[RunRow], plan: PresolvePlan) -> list[str]:
    """The #331 section, as report lines."""
    by_key = _by_key(rows)
    instances = _instances(rows, plan)
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
        *_completeness_lines(rows, plan),
        "gap+1 = max(gap_to_bks%, 0) + 1; GM over the instances feasible (finite gap) in BOTH the",
        "arm and the control at that seed; ratio = arm / control, BELOW 1 IS THE ARM AHEAD.",
        "Feasible counts are feasible / measured: a run with no completed search is out of",
        "both, and named under completeness. Claim-excluded instances are out of everything.",
        f"noise floor: control GM over the {len(floor_set)} instance(s) feasible in every arm at "
        f"every seed, max / min over seeds = {_fmt(floor)}"
        + (
            f"; a pooled ratio inside [{_fmt(1.0 / floor)}, {_fmt(floor)}] is not a difference"
            if math.isfinite(floor)
            else " (needs two seeds with a finite GM)"
        ),
        "",
        f"{'arm':<22}{'seed':>6}{'feasible':>10}{'ctl feas':>10}{'pool':>6}"
        f"{'GM arm':>11}{'GM ctl':>11}{'ratio':>11}",
    ]

    def feasible(arm: str, *seeds: int) -> str:
        """Feasible / measured over the planned runs of `arm` at `seeds`."""
        measured = [
            r
            for s in seeds
            for i in instances
            if (r := by_key.get((i, arm, s))) is not None and not r.no_search
        ]
        return f"{sum(1 for r in measured if r.feasible)}/{len(measured)}"

    for arm in arms:
        for seed in plan.seeds:
            pairs = [(i, seed) for i in pool(arm, seed)]
            gm_arm, gm_ctl = gm_over(arm, pairs), gm_over(CONTROL_ARM, pairs)
            lines.append(
                f"{arm:<22}{seed:>6}{feasible(arm, seed):>10}{feasible(CONTROL_ARM, seed):>10}"
                f"{len(pairs):>6}{_fmt(gm_arm):>11}{_fmt(gm_ctl):>11}"
                f"{_fmt(_ratio(gm_arm, gm_ctl)):>11}"
            )
    lines += [
        "",
        "pooled over seeds:",
        f"{'arm':<22}{'feasible':>10}{'ctl feas':>10}{'pool':>6}{'GM arm':>11}{'GM ctl':>11}"
        f"{'ratio':>11}{'seed range':>23}  reading",
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
        spread = f"{_fmt(min(per_seed))}..{_fmt(max(per_seed))}" if per_seed else "n/a"
        lines.append(
            f"{arm:<22}{feasible(arm, *plan.seeds):>10}"
            f"{feasible(CONTROL_ARM, *plan.seeds):>10}{len(pairs):>6}"
            f"{_fmt(gm_arm):>11}{_fmt(gm_ctl):>11}{_fmt(ratio):>11}{spread:>23}  "
            f"{_reading(ratio, floor)}"
        )
    return lines


def _reading(ratio: float, floor: float) -> str:
    """Where a pooled ratio sits against the floor, in words."""
    if not math.isfinite(ratio):
        return "no pool"
    if not math.isfinite(floor):
        return "no floor"
    # A floor of 1 is no seed noise measured, not a licence to read any
    # rounding-level ratio as a difference.
    if floor <= 1.0:
        return "no seed noise measured; read the ratio itself"
    if ratio < 1.0 / floor:
        return "arm ahead, outside the floor"
    if ratio > floor:
        return "arm behind, outside the floor"
    return "inside the floor"
