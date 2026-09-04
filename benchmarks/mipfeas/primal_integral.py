"""Score MIPfeas runs by Primal Integral and write the comparison table.

The metric follows the MIPfeas methodology
(https://www.gams.com/blog/2026/03/expanding-the-focus-introducing-the-mipfeas-benchmark/):
the primal gap of the incumbent at time t is

    p(t) = |x(t) - x*| / max(|x(t)|, |x*|)

with p = 2 while no feasible solution has been found, p = 1 when incumbent and
reference have opposite signs, and p = 0 when both are below 1e-6. The Primal
Integral is its time average over the budget,

    P(T) = (1/T) * integral of p(t) from 0 to T,

so P lies in [0, 2]: 0 is "optimal immediately", 2 is "never feasible". Rewarding
*early* good solutions is the point — it is the question a local-search heuristic
is built to answer, unlike gap-to-optimal at the time limit.

Usage:
    python primal_integral.py --results-dir results --budget 600 \
        --roster benchmarks/instances/mipfeas/roster.csv \
        --out benchmarks/instances/mipfeas/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

#: Below this magnitude an objective counts as zero, so that 0 vs 1e-12 does not
#: score as a 100% gap.
ZERO_TOLERANCE = 1e-6

#: Penalty for a run that never found a feasible solution.
NO_SOLUTION_GAP = 2.0

#: Penalty when incumbent and reference have opposite signs; the relative formula
#: is not meaningful across a sign change.
SIGN_FLIP_GAP = 1.0

#: An incumbent below a *proven* optimum is not a good result, it is a bug — a
#: violated constraint, a wrong objective, a mis-transcribed row. `primal_gap` takes
#: an absolute value, so it scores as an ordinary positive gap and would publish
#: silently. 232 of the 233 references are proven optima.
BELOW_REFERENCE_TOLERANCE = 1e-6

#: Shift for the geometric mean, so that a PI of exactly 0 does not collapse it.
GEOMETRIC_MEAN_SHIFT = 0.001

ENGINES = ("cbls", "cpsat")

#: Size of the full MIPfeas roster. Anything smaller is a wiring check, and the
#: table says so in its header — a partial run read as the published result is
#: the recurring way this repo has published a wrong number.
FULL_ROSTER_SIZE = 233

#: Result keys describing *how* a run was configured. Two results disagreeing on any
#: of them are not comparable. The driver resumes on file existence alone and
#: defaults to one results directory whatever the flags, so a flag changed between
#: invocations would otherwise be averaged into a single table with nothing to show
#: for it — and these are exactly the flags measured to move the aggregate.
CONFIG_KEYS = (
    "seed",
    "feasibility_tolerance",
    "compound_moves",
    "inf_clamp",
    "propagate_bounds",
    "max_propagation_passes",
    "workers",
    "parameters",
)


class Scored(NamedTuple):
    instance: str
    engine: str
    status: str
    objective: float | None
    reference_value: float
    reference_kind: str
    final_gap: float
    #: True when a proven optimum was beaten — a bug signal, not an achievement.
    below_reference: bool
    primal_integral: float
    wall_seconds: float | None
    n_vars: int | None
    n_cons: int | None
    #: Peak resident set of the job, so a full-roster run's concurrency can be
    #: sized from measurement rather than guessed.
    peak_rss_kib: int | None
    #: Columns the adapter narrowed for a CBLS-side reason, *after* implied bounds
    #: were derived: the `inf_clamp` fallback on a column nothing bounds, OR the
    #: int32 clip on an integer column whose (finite) bounds exceed what
    #: `Model::int_var` can hold. Both are restrictions the baseline does not
    #: share, so "both engines solved the same program" is only checkable if this
    #: is published next to the score.
    n_clamped_bounds: int | None
    #: Columns the MPS left unbounded on at least one side — the exposure the
    #: clamp would have covered in full had propagation not run. NOT a superset
    #: of `n_clamped_bounds`, which also counts the int32 clip on a column the
    #: file bounded, so the difference of the two is not "what propagation
    #: removed".
    n_unbounded_columns: int | None
    #: Columns whose bounds propagation tightened, unbounded or not.
    n_bounds_tightened: int | None
    #: The run's configuration, so a table cannot silently mix two of them.
    config: str
    #: Whether the incumbent profile came from the engine's log or is a single end
    #: point. A systematic log-format change would otherwise score every CP-SAT
    #: instance near 2.0, indistinguishable from "CP-SAT is bad".
    trace_source: str
    #: The engine's own verdict, where it has one finer than `status` (CP-SAT's
    #: OPTIMAL / FEASIBLE / NOT_SOLVED / MODEL_INVALID). An INFEASIBLE here on a
    #: roster of known-feasible instances is a red flag about CP-SAT's integer
    #: scaling, and would otherwise be indistinguishable from a plain timeout.
    solver_status: str
    provenance: str


def primal_gap(incumbent: float | None, reference: float) -> float:
    """MIPfeas primal gap of one incumbent against the reference value."""
    if incumbent is None:
        return NO_SOLUTION_GAP
    if abs(incumbent) < ZERO_TOLERANCE and abs(reference) < ZERO_TOLERANCE:
        return 0.0
    if incumbent * reference < 0:
        return SIGN_FLIP_GAP
    denominator = max(abs(incumbent), abs(reference))
    if denominator == 0.0:
        return 0.0
    return abs(incumbent - reference) / denominator


def primal_integral(trace: list[tuple[float, float]], reference: float, budget: float) -> float:
    """Time-average the primal gap over [0, budget].

    `trace` is the incumbent step function as (seconds, objective) pairs. Before the
    first entry the gap is NO_SOLUTION_GAP; each entry holds until the next one, and
    the last holds to the budget.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")

    area = 0.0
    previous_time = 0.0
    current_gap = NO_SOLUTION_GAP
    # Sorted by time, then objective descending: CP-SAT logs to 0.01s and routinely
    # reports several improvements inside one tick. Plain tuple ordering would apply
    # the *worst* of a tied group last and hold it until the next distinct time — a
    # small, one-directional penalty against whichever engine has the denser trace,
    # which is systematically CP-SAT.
    for raw_time, objective in sorted(trace, key=lambda entry: (entry[0], -entry[1])):
        time = min(max(raw_time, 0.0), budget)
        if time > previous_time:
            area += current_gap * (time - previous_time)
            previous_time = time
        current_gap = primal_gap(objective, reference)
    area += current_gap * (budget - previous_time)
    return area / budget


def shifted_geometric_mean(values: list[float], shift: float = GEOMETRIC_MEAN_SHIFT) -> float:
    if not values:
        return math.nan
    return math.exp(statistics.fmean(math.log(v + shift) for v in values)) - shift


def load_trace(path: Path) -> list[tuple[float, float]]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        trace = [
            (float(row["time_seconds"]), float(row["objective"])) for row in csv.DictReader(fh)
        ]
    # One NaN or infinity turns every aggregate into NaN with no warning: the
    # geometric mean, the arithmetic mean and the median all propagate it.
    bad = [entry for entry in trace if not (math.isfinite(entry[0]) and math.isfinite(entry[1]))]
    if bad:
        raise ValueError(f"{path} has non-finite entries (e.g. {bad[0]})")
    return trace


def _provenance(result: dict[str, object]) -> str:
    """Whatever identifies the build that produced a result: commit SHA or version."""
    for key in ("commit_sha", "ortools_version"):
        value = result.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def score_instance(
    instance: str,
    engine: str,
    reference_value: float,
    reference_kind: str,
    results_dir: Path,
    budget: float,
) -> Scored:
    result_path = results_dir / engine / f"{instance}.json"
    if not result_path.exists():
        # Not run is not the same as ran-and-failed: scoring it 2 would report a
        # gap in the harness as a gap in the solver. Excluded from the aggregates.
        return Scored(
            instance=instance,
            engine=engine,
            status="not_run",
            objective=None,
            reference_value=reference_value,
            reference_kind=reference_kind,
            final_gap=math.nan,
            below_reference=False,
            primal_integral=math.nan,
            wall_seconds=None,
            n_vars=None,
            n_cons=None,
            peak_rss_kib=None,
            n_clamped_bounds=None,
            n_unbounded_columns=None,
            n_bounds_tightened=None,
            config="",
            trace_source="",
            solver_status="",
            provenance="n/a",
        )

    try:
        result: dict[str, object] = json.loads(result_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{result_path} is not valid JSON (a job killed mid-write?)") from exc

    # A result produced under a different budget cannot be scored against this one:
    # holding its last incumbent over a longer budget silently improves its Primal
    # Integral. Reachable by ordinary use — the driver resumes on file existence and
    # defaults to one results directory whatever the budget, so the README's own
    # smoke-then-full sequence would otherwise fold 60s results into a 600s table.
    recorded = result.get("budget_seconds")
    if isinstance(recorded, (int, float)) and abs(float(recorded) - budget) > 1e-9:
        raise ValueError(
            f"{result_path} was produced at a {recorded}s budget but is being scored "
            f"at {budget}s. Re-run those jobs, or score at the budget they used."
        )
    status = str(result.get("status", "unknown"))
    raw_objective = result.get("objective")
    objective = float(raw_objective) if isinstance(raw_objective, (int, float)) else None

    trace = load_trace(results_dir / engine / f"{instance}.trace.csv")
    if status == "feasible" and objective is not None and not trace:
        # A result without a profile still has a known end state; treat the solution
        # as arriving at the buzzer rather than dropping the instance.
        trace = [(budget, objective)]
    if status != "feasible":
        trace = []

    wall = result.get("wall_seconds")
    n_vars = result.get("n_vars")
    n_cons = result.get("n_cons")
    peak_rss = result.get("peak_rss_kib")
    clamped = result.get("n_clamped_bounds")
    unbounded = result.get("n_unbounded_columns")
    tightened = result.get("n_bounds_tightened")
    return Scored(
        instance=instance,
        engine=engine,
        status=status,
        objective=objective,
        reference_value=reference_value,
        reference_kind=reference_kind,
        final_gap=primal_gap(objective, reference_value),
        below_reference=(
            reference_kind == "opt"
            and objective is not None
            and objective < reference_value - BELOW_REFERENCE_TOLERANCE * (abs(reference_value) + 1)
        ),
        primal_integral=primal_integral(trace, reference_value, budget),
        wall_seconds=float(wall) if isinstance(wall, (int, float)) else None,
        n_vars=int(n_vars) if isinstance(n_vars, int) else None,
        n_cons=int(n_cons) if isinstance(n_cons, int) else None,
        peak_rss_kib=int(peak_rss) if isinstance(peak_rss, int) else None,
        n_clamped_bounds=int(clamped) if isinstance(clamped, int) else None,
        n_unbounded_columns=int(unbounded) if isinstance(unbounded, int) else None,
        n_bounds_tightened=int(tightened) if isinstance(tightened, int) else None,
        config=";".join(f"{k}={result[k]}" for k in CONFIG_KEYS if k in result),
        trace_source=str(result.get("trace_source", "")),
        solver_status=str(result.get("cpsat_status", "")),
        provenance=_provenance(result),
    )


class Summary(NamedTuple):
    engine: str
    scored: int
    not_run: int
    feasible: int
    matched_reference: int
    #: Runs that beat a proven optimum. Any non-zero value is a defect to chase,
    #: not a result to publish.
    below_reference: int
    invalid_model: int
    #: Jobs that neither found a solution nor honestly searched for one — killed by
    #: the driver's timeout or memory cap, or failed to read the instance. Scored
    #: as no-solution, but surfaced separately: a harness failure and a search
    #: failure look identical in the aggregate otherwise.
    errored: int
    shifted_geomean: float
    arithmetic_mean: float
    median: float
    q1: float
    q3: float


def summarize(rows: list[Scored], engine: str) -> Summary:
    mine = [r for r in rows if r.engine == engine]
    ran = [r for r in mine if r.status != "not_run"]
    integrals = [r.primal_integral for r in ran]
    # "inclusive": the default extrapolates on small samples and can report an IQR
    # bound outside the metric's own [0, 2] range — a negative Primal Integral in a
    # file whose whole job is to be quoted.
    quartiles = (
        statistics.quantiles(integrals, n=4, method="inclusive")
        if len(integrals) > 1
        else [math.nan] * 3
    )
    return Summary(
        engine=engine,
        scored=len(ran),
        not_run=len(mine) - len(ran),
        feasible=sum(1 for r in ran if r.status == "feasible"),
        matched_reference=sum(1 for r in ran if r.final_gap < ZERO_TOLERANCE),
        below_reference=sum(1 for r in ran if r.below_reference),
        invalid_model=sum(1 for r in ran if r.status == "invalid_model"),
        errored=sum(1 for r in ran if r.status not in ("feasible", "no_solution", "invalid_model")),
        shifted_geomean=shifted_geometric_mean(integrals),
        arithmetic_mean=statistics.fmean(integrals) if integrals else math.nan,
        median=statistics.median(integrals) if integrals else math.nan,
        q1=quartiles[0],
        q3=quartiles[2],
    )


def check_uniform_configuration(rows: list[Scored]) -> None:
    """Refuse to score results produced under more than one configuration.

    The budget guard in `score_instance` catches only the budget. Everything else —
    the seed, the tolerance, Novelty Jump, the bound clamp, CP-SAT's worker count —
    is a CLI flag, so two invocations into the same results directory would average
    two configurations into one table and look entirely normal doing it.
    """
    for engine in ENGINES:
        configs = {r.config for r in rows if r.engine == engine and r.config}
        if len(configs) > 1:
            raise ValueError(
                f"{engine} results span {len(configs)} configurations: {sorted(configs)}. "
                f"The driver resumes on file existence alone, so a flag changed between "
                f"invocations lands in the same results directory. Re-run the odd ones "
                f"out with --force, or score them into separate tables."
            )


def read_roster(path: Path) -> list[tuple[str, float, str]]:
    with open(path, newline="") as fh:
        return [
            (row["instance"], float(row["reference_value"]), row["reference_kind"])
            for row in csv.DictReader(fh)
        ]


def write_comparison(
    path: Path, rows: list[Scored], summaries: list[Summary], budget: float, roster_path: Path
) -> None:
    header = [
        "# MIPfeas comparison: CBLS vs CP-SAT's violation_ls worker.",
        "#",
        "# This is an IMPLEMENTATION SANITY CHECK, not a MIP-competitiveness claim.",
        "# The only baseline is CP-SAT restricted to its fj + ls workers — the reference",
        "# implementation of the same jump-based algorithm CBLS reimplements. CP-SAT's",
        "# default portfolio, and every other MIP solver, are deliberately out of scope",
        "# (epic #87). A gap in either direction is informative about the reimplementation.",
        "#",
        f"# Roster:  {roster_path.name} ({len(rows) // max(len(summaries), 1)} instances)",
        f"# Budget:  {budget}s per instance-solver pair — the budget this table was scored at.",
        "#          Primal Integrals are budget-relative, so a table is comparable only",
        "#          to another scored at the same budget.",
        "# Metric:  Primal Integral over the budget, in [0, 2]; lower is better.",
        "#          0 = optimal immediately, 2 = never feasible.",
        "#",
        "# NOT a MIPfeas leaderboard entry: the published MIPfeas runs give each",
        "# solver 24 threads, and both engines here get 1. These numbers are",
        "# comparable to each other and never to one from plato.asu.edu or the",
        "# GAMS blog.",
        "#",
    ]
    instances = len(rows) // max(len(summaries), 1)
    partial = [s for s in summaries if s.not_run]
    if partial:
        # Fires independently of the wiring-check banner: a *full* roster with half
        # its jobs unfinished would otherwise read as a clean result, with the
        # shortfall visible only in a stderr warning nobody sees months later.
        header += [
            "# *** INCOMPLETE RUN — AGGREGATES COVER ONLY THE JOBS THAT RAN ***",
            "# " + "; ".join(f"{s.engine}: {s.not_run} of {instances} not run" for s in partial),
            "#",
        ]
    if instances != FULL_ROSTER_SIZE:
        # The budget is not part of this test (#126): the budget to score at is a
        # choice the run makes and the header records, so a full roster at any one
        # budget is a result. A partial roster is a wiring check at every budget.
        header += [
            "# *** WIRING CHECK, NOT A PUBLISHABLE RESULT ***",
            f"# The MIPfeas roster is {FULL_ROSTER_SIZE} instances; this table used {instances}.",
            "# These numbers are not comparable to a MIPfeas score, and the two engines'",
            "# relative standing on a subset need not hold on the full roster.",
            "#",
        ]
    header.append("# Aggregates (shifted geometric mean is the primary ranking):")
    for s in summaries:
        header.append(
            f"#   {s.engine:<6} sgm={s.shifted_geomean:.4f} mean={s.arithmetic_mean:.4f} "
            f"median={s.median:.4f} iqr=[{s.q1:.4f},{s.q3:.4f}] "
            f"feasible={s.feasible}/{s.scored} matched_reference={s.matched_reference} "
            f"invalid_model={s.invalid_model} errored={s.errored} "
            f"below_reference={s.below_reference} not_run={s.not_run}"
        )
    header.append("#")

    with open(path, "w", newline="") as fh:
        fh.write("\n".join(header) + "\n")
        writer = csv.writer(fh)
        writer.writerow(
            [
                "instance",
                "engine",
                "status",
                "objective",
                "reference_value",
                "reference_kind",
                "final_gap",
                "below_reference",
                "primal_integral",
                "wall_seconds",
                "n_vars",
                "n_cons",
                "peak_rss_kib",
                "n_clamped_bounds",
                "n_unbounded_columns",
                "n_bounds_tightened",
                "trace_source",
                "solver_status",
                "provenance",
                "config",
            ]
        )
        for r in rows:
            writer.writerow(
                [
                    r.instance,
                    r.engine,
                    r.status,
                    "" if r.objective is None else repr(r.objective),
                    repr(r.reference_value),
                    r.reference_kind,
                    f"{r.final_gap:.6g}",
                    int(r.below_reference),
                    f"{r.primal_integral:.6g}",
                    "" if r.wall_seconds is None else f"{r.wall_seconds:.4f}",
                    "" if r.n_vars is None else r.n_vars,
                    "" if r.n_cons is None else r.n_cons,
                    "" if r.peak_rss_kib is None else r.peak_rss_kib,
                    "" if r.n_clamped_bounds is None else r.n_clamped_bounds,
                    "" if r.n_unbounded_columns is None else r.n_unbounded_columns,
                    "" if r.n_bounds_tightened is None else r.n_bounds_tightened,
                    r.trace_source,
                    r.solver_status,
                    r.provenance,
                    r.config,
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, help="dir holding cbls/ and cpsat/")
    parser.add_argument("--roster", default="benchmarks/instances/mipfeas/roster.csv")
    parser.add_argument("--budget", type=float, required=True, help="seconds the runs were given")
    parser.add_argument("--out", required=True, help="comparison.csv to write")
    args = parser.parse_args()

    roster_path = Path(args.roster)
    roster = read_roster(roster_path)
    results_dir = Path(args.results_dir)

    rows = [
        score_instance(name, engine, value, kind, results_dir, args.budget)
        for name, value, kind in roster
        for engine in ENGINES
    ]
    summaries = [summarize(rows, engine) for engine in ENGINES]
    check_uniform_configuration(rows)

    out_path = Path(args.out)
    write_comparison(out_path, rows, summaries, args.budget, roster_path)

    print(
        f"Scored {len(roster)} instances x {len(ENGINES)} engines at {args.budget}s -> {out_path}"
    )
    for s in summaries:
        print(
            f"  {s.engine:<6} sgm={s.shifted_geomean:.4f} mean={s.arithmetic_mean:.4f} "
            f"median={s.median:.4f} feasible={s.feasible}/{s.scored} "
            f"matched_reference={s.matched_reference} not_run={s.not_run}"
        )
    incomplete = [s for s in summaries if s.not_run]
    if incomplete:
        print(
            "\nWARNING: the roster is not fully covered "
            f"({', '.join(f'{s.engine}: {s.not_run} not run' for s in incomplete)}). "
            "Aggregates cover only the instances that ran and are not comparable to a "
            "published MIPfeas score.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
