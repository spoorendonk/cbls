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

Every row reported feasible must also carry an independent verdict from
`verify_solution.py` — its solution checked against the original instance file
with a reader neither engine uses (issue #138). A row whose solution fails that
check, or that carries no verdict at all, publishes **no objective and no Primal
Integral**, and is left out of the aggregates rather than scored as a failure:
the run did find a point, it was simply rejected, and scoring it 2.0 would be
publishing a derived number of its own. `--allow-unverified` relaxes the
"carries no verdict" half for an older results directory; nothing relaxes a
failed check.

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

#: The only verification verdict that lets a feasible row publish its objective.
#: `fail` is a rejected solution and `error` a check that could not run; both
#: withhold, and both stay distinguishable in the published row.
VERIFICATION_PASS = "pass"

#: A solution the checker read and rejected, as opposed to one it could not check.
VERIFICATION_FAIL = "fail"

#: What a row that had nothing to verify reports: the run found no solution, so
#: there is no point to check and no objective to withhold. Kept distinct from
#: `unverified` so a defect counter cannot mistake one for the other.
NOT_APPLICABLE = "not_applicable"

#: What a feasible row with no verdict file at all reports.
UNVERIFIED = "unverified"

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


class Verification(NamedTuple):
    """One row's independent verdict, as `verify_solution.py` recorded it."""

    #: `pass`, `fail`, `error`, or `unverified` when no verdict file exists.
    verdict: str
    #: Stable machine-readable code: `row_violation`, `missing_solution_file`, ...
    reason: str
    #: A pass whose worst violation reached 10% of its tolerance. A flag to look
    #: at, never a reason to withhold.
    marginal: bool
    #: The tolerances the verdict was reached under, so a table cannot silently
    #: mix two of them.
    tolerances: str
    #: The human sentence behind `reason`. Published in the failure section of the
    #: report: a stable code groups, a sentence is what tells the reader what to do.
    message: str = ""
    #: Columns SCIP found in the instance file, and constraints it held. A third
    #: reading of the same program, from a reader neither engine uses, which is
    #: what makes the engines' own shape counts checkable rather than merely
    #: disclosed (issue #139).
    n_columns: int | None = None
    n_rows: int | None = None


#: The verdict of a row nobody checked. Not a NamedTuple default, because a row
#: that was never *reported feasible* has nothing to verify and gets this too.
NO_VERDICT = Verification(UNVERIFIED, "no_verdict_file", False, "")


def read_verification(results_dir: Path, engine: str, instance: str) -> Verification:
    """Read `<engine>/<instance>.verify.json`, or report that there is none."""
    path = results_dir / engine / f"{instance}.verify.json"
    if not path.exists():
        return NO_VERDICT
    try:
        record: dict[str, object] = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON (a job killed mid-write?)") from exc
    tolerances = record.get("tolerances")
    columns = record.get("n_columns")
    rows = record.get("n_rows")
    return Verification(
        verdict=str(record.get("verdict", UNVERIFIED)),
        reason=str(record.get("reason", "")),
        marginal=bool(record.get("marginal", False)),
        tolerances=(
            ";".join(f"{k}={v}" for k, v in sorted(tolerances.items()))
            if isinstance(tolerances, dict)
            else ""
        ),
        message=str(record.get("message", "")),
        # A verdict the driver wrote for a checker that died carries no shape at
        # all, and the verifier's own error verdicts carry zeroes where it never
        # got as far as reading the file. Neither is a reading of the program, so
        # neither may enter the cross-check.
        n_columns=int(columns) if isinstance(columns, int) and columns > 0 else None,
        n_rows=int(rows) if isinstance(rows, int) and rows > 0 else None,
    )


def withholds(status: str, verification: Verification, require_verification: bool) -> bool:
    """Whether this row must publish no objective and no derived score.

    Only a feasible row can be withheld — the others have no objective to begin
    with. A `fail` always withholds; a missing verdict withholds unless the
    caller opted out, which is what an older results directory needs.
    """
    if status == "solution_write_error":
        # The search found a point and only the dump failed. Publishing it would
        # score the row 2.0 -- a derived number for a row nothing could check,
        # which is exactly what this rule exists to prevent -- and charge a disk
        # error to the search.
        return True
    if status != "feasible":
        return False
    if verification.verdict == VERIFICATION_PASS:
        return False
    return not (verification.verdict == UNVERIFIED and not require_verification)


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
    #: The solve call only -- what `wall_seconds` has always meant in a result
    #: file, renamed here because the table now carries the other half beside it.
    #: It can exceed the budget: search initialisation is not bounded by the
    #: deadline, so a large model's first batch runs to completion whatever the
    #: clock says. That is a different effect from `setup_seconds` below and the
    #: two are separate columns so a reader can tell them apart (issue #139).
    solve_seconds: float | None
    #: Instance read + model build + bound propagation: everything outside the
    #: solve bracket, which no published MIPfeas table had ever measured. Blank
    #: for a results directory written before the runners recorded it.
    setup_seconds: float | None
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
    #: Linear constraints the baseline holds that bound nothing on either side --
    #: the MPS `N` rows after the objective. The one known-benign way the two
    #: engines' constraint counts differ, recorded so the cross-check can subtract
    #: it instead of arguing about it after the fact.
    n_free_cons: int | None
    #: Whatever the job recorded about why it did not search: the driver's kill
    #: message, the reader's exception, the adapter's rejection. Recorded in every
    #: result since the driver was written and published by nothing until now, so
    #: a dead job showed up in the table as a gap.
    message: str
    #: The independent verdict on this row's solution: `pass`, `fail`, `error`,
    #: or `unverified`. Only `pass` lets the objective above be a number.
    verification: str
    #: Why, as a stable code a counter can group on.
    verification_reason: str
    #: A pass that came within 10% of a tolerance. Published, never withheld.
    verification_marginal: bool
    #: True when the objective and the score were withheld because of the verdict.
    withheld: bool
    #: The tolerances the verdict was reached under; not a column, but compared
    #: across rows so one table cannot mix two of them.
    verification_tolerances: str
    #: The verifier's sentence for this row, published in the report's failure
    #: section. Empty for a row it accepted.
    verification_message: str
    #: Columns and constraints SCIP read from the instance file while verifying
    #: this row: the third opinion the model-shape cross-check needs. None when
    #: the row carries no verdict.
    checker_n_vars: int | None
    checker_n_cons: int | None


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
    require_verification: bool = True,
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
            solve_seconds=None,
            setup_seconds=None,
            n_vars=None,
            n_cons=None,
            peak_rss_kib=None,
            n_clamped_bounds=None,
            n_unbounded_columns=None,
            n_bounds_tightened=None,
            n_free_cons=None,
            message="no result file was written for this job",
            config="",
            trace_source="",
            solver_status="",
            provenance="n/a",
            verification=UNVERIFIED,
            verification_reason="not_run",
            verification_marginal=False,
            withheld=False,
            verification_tolerances="",
            verification_message="",
            checker_n_vars=None,
            checker_n_cons=None,
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
    # Kept across the withholding below. `below_reference` is a defect flag, not a
    # published number, and 232 of the 233 references are proven optima -- dropping
    # it for a row the checker rejected would disable the cheapest gate this
    # benchmark has on precisely the rows most likely to trip it.
    reported_objective = objective

    verification = read_verification(results_dir, engine, instance)
    if status not in ("feasible", "solution_write_error"):
        # Nothing was reported, so nothing is unchecked. Said explicitly, because a
        # counter grouping on this column would otherwise read every no-solution
        # row as one nobody verified.
        verification = verification._replace(verdict=NOT_APPLICABLE, reason="", marginal=False)
    withheld = withholds(status, verification, require_verification)
    if withheld:
        # The whole of the rule: no objective, and therefore no gap, no Primal
        # Integral and no place in the aggregates. An empty trace below makes the
        # profile unscorable rather than scoring it as never-feasible, which
        # would be a derived number of its own.
        objective = None

    trace = load_trace(results_dir / engine / f"{instance}.trace.csv")
    if status == "feasible" and objective is not None and not trace:
        # A result without a profile still has a known end state; treat the solution
        # as arriving at the buzzer rather than dropping the instance.
        trace = [(budget, objective)]
    if status != "feasible" or withheld:
        trace = []

    wall = result.get("wall_seconds")
    # `setup_seconds` where the runner wrote it; otherwise its two halves, so a
    # result that recorded only the read (a build that threw) still reports what
    # it measured. Absent entirely in a directory written before #139.
    setup = result.get("setup_seconds")
    if not isinstance(setup, (int, float)):
        halves = [result.get(k) for k in ("read_seconds", "build_seconds")]
        measured = [float(h) for h in halves if isinstance(h, (int, float))]
        setup = sum(measured) if measured else None
    n_vars = result.get("n_vars")
    n_cons = result.get("n_cons")
    peak_rss = result.get("peak_rss_kib")
    clamped = result.get("n_clamped_bounds")
    unbounded = result.get("n_unbounded_columns")
    tightened = result.get("n_bounds_tightened")
    free_cons = result.get("n_free_cons")
    return Scored(
        instance=instance,
        engine=engine,
        status=status,
        objective=objective,
        reference_value=reference_value,
        reference_kind=reference_kind,
        final_gap=math.nan if withheld else primal_gap(objective, reference_value),
        below_reference=(
            reference_kind == "opt"
            and reported_objective is not None
            and reported_objective
            < reference_value - BELOW_REFERENCE_TOLERANCE * (abs(reference_value) + 1)
        ),
        primal_integral=(math.nan if withheld else primal_integral(trace, reference_value, budget)),
        solve_seconds=float(wall) if isinstance(wall, (int, float)) else None,
        setup_seconds=float(setup) if isinstance(setup, (int, float)) else None,
        n_vars=int(n_vars) if isinstance(n_vars, int) else None,
        n_cons=int(n_cons) if isinstance(n_cons, int) else None,
        peak_rss_kib=int(peak_rss) if isinstance(peak_rss, int) else None,
        n_clamped_bounds=int(clamped) if isinstance(clamped, int) else None,
        n_unbounded_columns=int(unbounded) if isinstance(unbounded, int) else None,
        n_bounds_tightened=int(tightened) if isinstance(tightened, int) else None,
        n_free_cons=int(free_cons) if isinstance(free_cons, int) else None,
        message=str(result.get("message", "")),
        config=";".join(f"{k}={result[k]}" for k in CONFIG_KEYS if k in result),
        trace_source=str(result.get("trace_source", "")),
        solver_status=str(result.get("cpsat_status", "")),
        provenance=_provenance(result),
        verification=verification.verdict,
        verification_reason=verification.reason,
        verification_marginal=verification.marginal,
        withheld=withheld,
        verification_tolerances=verification.tolerances,
        verification_message=verification.message,
        checker_n_vars=verification.n_columns,
        checker_n_cons=verification.n_rows,
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
    #: Rows whose solution the independent checker rejected. Any non-zero value
    #: is a defect to chase before anything else in this table means anything.
    verification_failed: int
    #: Feasible rows with no usable verdict — never checked, or the checker
    #: itself could not run. Withheld like a failure, counted apart from one.
    unverified: int
    #: Passing rows whose worst violation came within 10% of its tolerance.
    verification_marginal: int
    shifted_geomean: float
    arithmetic_mean: float
    median: float
    q1: float
    q3: float


def summarize(rows: list[Scored], engine: str) -> Summary:
    mine = [r for r in rows if r.engine == engine]
    ran = [r for r in mine if r.status != "not_run"]
    # A withheld row ran, but published nothing: it is excluded from the
    # aggregates exactly as a not-run row is, and counted where a reader will see
    # it instead of averaged into a number that would look like a search result.
    published = [r for r in ran if not r.withheld]
    integrals = [r.primal_integral for r in published]
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
        scored=len(published),
        not_run=len(mine) - len(ran),
        feasible=sum(1 for r in published if r.status == "feasible"),
        matched_reference=sum(1 for r in published if r.final_gap < ZERO_TOLERANCE),
        # Over every row that ran, withheld or not: beating a proven optimum is a
        # defect signal rather than a published number, and a rejected solution is
        # the likeliest place for one.
        below_reference=sum(1 for r in ran if r.below_reference),
        invalid_model=sum(1 for r in published if r.status == "invalid_model"),
        errored=sum(
            1 for r in published if r.status not in ("feasible", "no_solution", "invalid_model")
        ),
        verification_failed=sum(1 for r in ran if r.verification == VERIFICATION_FAIL),
        # Keyed on the verdict, not on `withheld`: --allow-unverified withholds
        # nothing, and that is the run where a count of unchecked rows matters most.
        unverified=sum(
            1
            for r in ran
            if r.verification not in (VERIFICATION_PASS, VERIFICATION_FAIL, NOT_APPLICABLE)
        ),
        verification_marginal=sum(1 for r in published if r.verification_marginal),
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
    # The same hazard one layer down: verdicts reached under two different
    # tolerance sets are two different claims, and the table states only one.
    tolerances = {r.verification_tolerances for r in rows if r.verification_tolerances}
    if len(tolerances) > 1:
        raise ValueError(
            f"the verdicts span {len(tolerances)} tolerance sets: {sorted(tolerances)}. "
            f"Re-verify them under one, or score them into separate tables."
        )


# ---------------------------------------------------------------------------
# Parity, defects and cross-checks.
#
# This benchmark is admitted as a head-to-head against the reference
# implementation of the same jump-based algorithm -- correctness first, and
# performance against that one worker pairing (CLAUDE.md, "Benchmark priority").
# The Primal Integral aggregate answers the second half. Everything below is the
# first half: who reached feasibility where, what the two engines disagree about,
# and which rows are defects rather than results (issue #139).


#: A row that reached feasibility and published it.
PARITY_FEASIBLE = "feasible"

#: A row that searched honestly and found nothing.
PARITY_NO_SOLUTION = "no_solution"

#: A row that cannot answer the parity question either way: it never ran, it was
#: killed, its model was rejected, or its solution was withheld. Counting one of
#: these as "did not reach feasibility" would charge a harness failure to the
#: search, which is the same mistake `not_run` has always been kept out of the
#: aggregates to avoid.
PARITY_EXCLUDED = "excluded"

#: Statuses that mean the engine searched. Anything else is a job that did not
#: get as far as answering the question.
SEARCHED_STATUSES = (PARITY_FEASIBLE, PARITY_NO_SOLUTION)

#: Trace sources that mean the engine reported a genuine anytime profile: CP-SAT's
#: solver log, and the CBLS runner's progress callback. Anything else means the
#: scorer stood in a single end point, which is a harness condition and not a
#: search result -- a log-format change would otherwise average in silently.
HEALTHY_TRACE_SOURCES = ("log", "callback")

#: How far past the budget a solve may run before the report names it. Search
#: initialisation is not bounded by the deadline, so an overrun is expected on a
#: large model; a second of slack keeps ordinary scheduling jitter out of the list
#: while still catching the effect the column exists to show.
BUDGET_OVERRUN_TOLERANCE = 1.0


def parity_verdict(row: Scored) -> str:
    """Where `row` stands on the feasibility question: one of the PARITY_* codes."""
    if row.status == "not_run" or row.withheld or row.status not in SEARCHED_STATUSES:
        return PARITY_EXCLUDED
    return PARITY_FEASIBLE if row.status == PARITY_FEASIBLE else PARITY_NO_SOLUTION


class Parity(NamedTuple):
    """Feasibility parity between the two engines, instance by instance."""

    #: Instances where both engines answered the question. The denominator of
    #: every set below, and stated wherever they are.
    considered: list[str]
    both_feasible: list[str]
    neither_feasible: list[str]
    #: engine -> the instances only that engine reached feasibility on. The two
    #: asymmetric difference sets, which are the central finding of a parity check.
    only: dict[str, list[str]]
    #: (instance, engine, why) for every row that could not take part.
    excluded: list[tuple[str, str, str]]
    #: engine -> feasible rows over the whole roster, excluded rows included in
    #: the denominator. Not the same denominator as the sets above, and said so.
    feasible: dict[str, int]
    roster_size: int

    @property
    def agreement(self) -> int:
        """Instances the two engines agree about, feasible or not."""
        return len(self.both_feasible) + len(self.neither_feasible)


def _exclusion_reason(row: Scored) -> str:
    if row.status == "not_run":
        return "not run"
    if row.withheld:
        return f"withheld ({row.verification}: {row.verification_reason or 'no reason recorded'})"
    return f"did not search (status {row.status})"


def compare_feasibility(rows: list[Scored], engines: tuple[str, ...] = ENGINES) -> Parity:
    """Who reached feasibility where, and where the two engines differ."""
    by_instance: dict[str, dict[str, Scored]] = {}
    for row in rows:
        by_instance.setdefault(row.instance, {})[row.engine] = row

    considered: list[str] = []
    both: list[str] = []
    neither: list[str] = []
    only: dict[str, list[str]] = {engine: [] for engine in engines}
    excluded: list[tuple[str, str, str]] = []
    feasible = {engine: 0 for engine in engines}

    for instance, per_engine in by_instance.items():
        verdicts: dict[str, str] = {}
        for engine in engines:
            scored_row = per_engine.get(engine)
            verdict = PARITY_EXCLUDED if scored_row is None else parity_verdict(scored_row)
            verdicts[engine] = verdict
            if verdict == PARITY_FEASIBLE:
                feasible[engine] += 1
            elif verdict == PARITY_EXCLUDED:
                excluded.append(
                    (
                        instance,
                        engine,
                        "no row in this table"
                        if scored_row is None
                        else _exclusion_reason(scored_row),
                    )
                )
        if any(v == PARITY_EXCLUDED for v in verdicts.values()):
            continue
        considered.append(instance)
        reached = [e for e in engines if verdicts[e] == PARITY_FEASIBLE]
        if len(reached) == len(engines):
            both.append(instance)
        elif not reached:
            neither.append(instance)
        else:
            for engine in reached:
                only[engine].append(instance)

    return Parity(
        considered=sorted(considered),
        both_feasible=sorted(both),
        neither_feasible=sorted(neither),
        only={engine: sorted(names) for engine, names in only.items()},
        excluded=sorted(excluded),
        feasible=feasible,
        roster_size=len(by_instance),
    )


class ShapeDisagreement(NamedTuple):
    """One instance the two readers -- or the checker -- disagree about."""

    instance: str
    #: `variables` or `constraints`.
    kind: str
    #: The counts, rendered, so the line stands on its own in a report.
    counts: str
    #: Whether the stated rule accounts for the difference.
    benign: bool
    explanation: str


#: The rule, stated once and quoted into the report so the table and the prose
#: cannot drift apart.
SHAPE_RULE = (
    "Variable counts must agree exactly: every reader enumerates the same MPS "
    "COLUMNS section, so a difference there is a reader defect, never benign. "
    "Constraint counts may differ by exactly the free rows the baseline keeps -- "
    "the MPS `N` rows after the first, which is the objective. OR-Tools' "
    "ModelBuilder holds each remaining `N` row as a linear constraint with "
    "infinite bounds; the CBLS adapter drops them, and so does SCIP. A free row "
    "constrains nothing, so the two programs have the same feasible set and the "
    "difference is benign. Any other difference, in either direction, is flagged."
)


def _variable_disagreement(
    instance: str, counted: dict[str, int | None]
) -> ShapeDisagreement | None:
    """A variable-count difference, which the rule never excuses."""
    known = [v for v in counted.values() if v is not None]
    if len(set(known)) <= 1:
        return None
    return ShapeDisagreement(
        instance=instance,
        kind="variables",
        counts=" ".join(f"{k}={v}" for k, v in counted.items() if v is not None),
        benign=False,
        explanation="variable counts must agree exactly",
    )


def _constraint_disagreement(
    instance: str,
    engines: tuple[str, str],
    counted: dict[str, int | None],
    free: int | None,
) -> ShapeDisagreement | None:
    """A constraint-count difference, benign only where the free rows explain it."""
    first, second = engines
    mine, theirs, checker = counted[first], counted[second], counted["checker"]
    if mine is None or theirs is None:
        return None
    counts = " ".join(f"{k}={v}" for k, v in counted.items() if v is not None)
    if free is not None:
        counts += f" free_rows={free}"
    if mine != theirs:
        benign = free is not None and free > 0 and theirs - mine == free
        return ShapeDisagreement(
            instance=instance,
            kind="constraints",
            counts=counts,
            benign=benign,
            explanation=(
                f"{second} keeps {free} free row(s) the CBLS adapter drops"
                if benign
                else "not accounted for by the free rows the baseline keeps"
            ),
        )
    if checker is not None and checker != mine:
        # The engines agree and the third reader does not, which the free-row rule
        # cannot explain in either direction: SCIP drops free rows too.
        return ShapeDisagreement(
            instance=instance,
            kind="constraints",
            counts=counts,
            benign=False,
            explanation="the checker read a different constraint count from both engines",
        )
    return None


def cross_check_shapes(
    rows: list[Scored], engines: tuple[str, ...] = ENGINES
) -> list[ShapeDisagreement]:
    """Compare what each engine -- and the checker -- read from the same file.

    "Both engines solved the same program" is this benchmark's entire admission
    ground, and until now each side merely *disclosed* its own shape. A real
    disagreement and a benign one looked identical.
    """
    if len(engines) != 2:
        raise ValueError(f"the cross-check compares exactly two engines, got {engines}")
    pair = (engines[0], engines[1])
    by_instance: dict[str, dict[str, Scored]] = {}
    for row in rows:
        by_instance.setdefault(row.instance, {})[row.engine] = row

    found: list[ShapeDisagreement] = []
    for instance in sorted(by_instance):
        per_engine = by_instance[instance]
        a, b = per_engine.get(pair[0]), per_engine.get(pair[1])
        if a is None or b is None:
            continue
        variables = {
            pair[0]: a.n_vars,
            pair[1]: b.n_vars,
            "checker": _first_known(a.checker_n_vars, b.checker_n_vars),
        }
        constraints = {
            pair[0]: a.n_cons,
            pair[1]: b.n_cons,
            "checker": _first_known(a.checker_n_cons, b.checker_n_cons),
        }
        free = b.n_free_cons if b.n_free_cons is not None else a.n_free_cons
        found += [
            d
            for d in (
                _variable_disagreement(instance, variables),
                _constraint_disagreement(instance, pair, constraints, free),
            )
            if d is not None
        ]
    return found


def _first_known(*values: int | None) -> int | None:
    return next((v for v in values if v is not None), None)


class TraceHealth(NamedTuple):
    """How many of an engine's profiles are real, over a stated denominator."""

    engine: str
    #: Rows the engine reported feasible. A run that found nothing has no
    #: incumbent profile to have, so it is not in the denominator.
    reported_feasible: int
    healthy: int
    #: Rows whose profile degraded to a single end point.
    degraded: int
    #: Rows from a results directory that recorded no `trace_source` at all.
    unrecorded: int
    degraded_instances: list[str]
    unrecorded_instances: list[str]


def trace_health(rows: list[Scored], engine: str) -> TraceHealth:
    """Trace-health counts for one engine, with the denominator carried along."""
    mine = [r for r in rows if r.engine == engine and r.status == PARITY_FEASIBLE]
    degraded = sorted(
        r.instance for r in mine if r.trace_source and r.trace_source not in HEALTHY_TRACE_SOURCES
    )
    unrecorded = sorted(r.instance for r in mine if not r.trace_source)
    return TraceHealth(
        engine=engine,
        reported_feasible=len(mine),
        healthy=len(mine) - len(degraded) - len(unrecorded),
        degraded=len(degraded),
        unrecorded=len(unrecorded),
        degraded_instances=degraded,
        unrecorded_instances=unrecorded,
    )


class JobFailure(NamedTuple):
    """A job that did not produce an honest search result, and what it said."""

    instance: str
    engine: str
    #: `not_run`, `killed`, `read_error`, `verification`, ...
    kind: str
    reason: str


def job_failures(rows: list[Scored]) -> list[JobFailure]:
    """Every reason a row is missing or withheld, from the records that hold them.

    The driver has always written a message on the job it killed, and the verifier
    on the solution it refused. Neither reached the table, so a dead job appeared
    there as a blank and the reason lived on only in a results directory nobody
    publishes.
    """
    failures: list[JobFailure] = []
    for row in sorted(rows, key=lambda r: (r.instance, r.engine)):
        if row.status not in SEARCHED_STATUSES:
            failures.append(
                JobFailure(
                    instance=row.instance,
                    engine=row.engine,
                    kind=row.status,
                    reason=row.message or "no message recorded",
                )
            )
        if row.verification in (VERIFICATION_FAIL, "error") or (
            row.withheld and row.verification == UNVERIFIED
        ):
            failures.append(
                JobFailure(
                    instance=row.instance,
                    engine=row.engine,
                    kind=f"verification {row.verification}",
                    reason=" ".join(
                        part
                        for part in (
                            row.verification_reason,
                            f"-- {row.verification_message}" if row.verification_message else "",
                        )
                        if part
                    )
                    or "no reason recorded",
                )
            )
    return failures


class TimingSummary(NamedTuple):
    """Setup and solve time for one engine, kept apart on purpose.

    Two different effects hide behind a single wall-clock column. `setup` is work
    the old column never bracketed at all -- reading the instance, building the
    model, propagating bounds. `overrun` is solve time past the budget, which
    happens because search initialisation is not bounded by the deadline. A reader
    has to be able to tell them apart, so they are never summed here.
    """

    engine: str
    #: Rows whose result recorded a setup time. Zero for a results directory
    #: written before the runners measured it -- in which case no magnitude for
    #: setup may be stated at all, and the report says so instead of guessing.
    setup_measured: int
    searched: int
    setup_median: float
    setup_max: float
    setup_max_instance: str
    #: The largest setup time as a fraction of the budget. The quantity the
    #: benchmark actually cares about: setup is not charged against the search,
    #: but it is charged against the wall clock a run is scheduled for.
    setup_max_budget_share: float
    overruns: int
    overrun_max: float
    overrun_max_instance: str


def timing_summary(rows: list[Scored], engine: str, budget: float) -> TimingSummary:
    mine = [r for r in rows if r.engine == engine and r.status in SEARCHED_STATUSES]
    setups = [(r.setup_seconds, r.instance) for r in mine if r.setup_seconds is not None]
    overruns = [
        (r.solve_seconds - budget, r.instance)
        for r in mine
        if r.solve_seconds is not None and r.solve_seconds - budget > BUDGET_OVERRUN_TOLERANCE
    ]
    worst_setup = max(setups, default=(math.nan, ""))
    worst_overrun = max(overruns, default=(math.nan, ""))
    return TimingSummary(
        engine=engine,
        setup_measured=len(setups),
        searched=len(mine),
        setup_median=statistics.median(v for v, _ in setups) if setups else math.nan,
        setup_max=worst_setup[0],
        setup_max_instance=worst_setup[1],
        setup_max_budget_share=worst_setup[0] / budget if setups and budget > 0 else math.nan,
        overruns=len(overruns),
        overrun_max=worst_overrun[0],
        overrun_max_instance=worst_overrun[1],
    )


class Defects(NamedTuple):
    """Everything in a run that is a bug rather than a result.

    Reported as a block ahead of the score, because every one of these makes the
    score below it mean less, and a column is not where a reader looking at a
    233-row table will notice one.
    """

    #: engine -> count, for the per-engine counters `summarize` already produces.
    verification_failed: dict[str, int]
    unverified: dict[str, int]
    below_reference: dict[str, int]
    invalid_model: dict[str, int]
    errored: dict[str, int]
    not_run: dict[str, int]
    verification_marginal: dict[str, int]
    #: Shape disagreements the stated rule does not account for. Not per engine:
    #: a disagreement is a property of the pair.
    shape_flagged: int
    #: Rows whose anytime profile degraded to a single point, summed over engines.
    trace_degraded: int

    @property
    def total(self) -> int:
        """How many defect findings this run carries, over every counter above."""
        per_engine = (
            self.verification_failed,
            self.unverified,
            self.below_reference,
            self.invalid_model,
            self.errored,
            self.not_run,
        )
        return (
            sum(sum(counter.values()) for counter in per_engine)
            + self.shape_flagged
            + self.trace_degraded
        )


def collect_defects(
    rows: list[Scored], summaries: list[Summary], engines: tuple[str, ...] = ENGINES
) -> Defects:
    by_engine = {s.engine: s for s in summaries}
    shapes = cross_check_shapes(rows, engines)
    return Defects(
        verification_failed={e: by_engine[e].verification_failed for e in by_engine},
        unverified={e: by_engine[e].unverified for e in by_engine},
        below_reference={e: by_engine[e].below_reference for e in by_engine},
        invalid_model={e: by_engine[e].invalid_model for e in by_engine},
        errored={e: by_engine[e].errored for e in by_engine},
        not_run={e: by_engine[e].not_run for e in by_engine},
        verification_marginal={e: by_engine[e].verification_marginal for e in by_engine},
        shape_flagged=sum(1 for d in shapes if not d.benign),
        trace_degraded=sum(trace_health(rows, e).degraded for e in engines),
    )


#: How the anytime aggregate is labelled everywhere it appears. Named for the
#: exact pairing it is measured against rather than "solvers": the only baseline
#: is CP-SAT's `fj` + `ls` subsolvers under `num_violation_ls`, which is the
#: reference implementation of the algorithm CBLS reimplements. Anything broader
#: -- CP-SAT's default portfolio, another MIP solver -- is a different and
#: rejected question (epic #87).
ANYTIME_LABEL = (
    "Anytime quality vs CP-SAT's fj + ls subsolvers under num_violation_ls "
    "(Primal Integral; shifted geometric mean is the primary ranking)"
)


def _names(instances: list[str], limit: int = 12) -> str:
    """Instance names for a one-line summary, truncated with a count."""
    if not instances:
        return "none"
    if len(instances) <= limit:
        return ", ".join(instances)
    return ", ".join(instances[:limit]) + f", ... (+{len(instances) - limit} more)"


def headline_lines(
    rows: list[Scored], summaries: list[Summary], engines: tuple[str, ...] = ENGINES
) -> list[str]:
    """The parity-and-defects block that leads the comparison table's header.

    A compact form of the report: the counts, and the difference sets by name
    while they are short. The report file carries them in full.
    """
    parity = compare_feasibility(rows, engines)
    defects = collect_defects(rows, summaries, engines)
    shapes = cross_check_shapes(rows, engines)
    lines = [
        "# DEFECTS (a non-zero count here makes every number below it mean less):",
        "#   "
        + "  ".join(
            f"{name}={sum(counter.values())}"
            for name, counter in (
                ("verification_failed", defects.verification_failed),
                ("unverified", defects.unverified),
                ("below_reference", defects.below_reference),
                ("invalid_model", defects.invalid_model),
                ("errored", defects.errored),
                ("not_run", defects.not_run),
            )
        )
        + f"  shape_flagged={defects.shape_flagged}"
        + f"  trace_degraded={defects.trace_degraded}"
        + f"  (total {defects.total})",
        "#",
        "# FEASIBILITY PARITY:",
    ]
    lines += [
        f"#   {engine} reached feasibility on {parity.feasible[engine]}/{parity.roster_size} "
        f"roster instances"
        for engine in engines
    ]
    lines += [
        f"#   comparable on {len(parity.considered)}/{parity.roster_size} instances "
        f"(the rest are excluded: a row that never ran, was killed or was withheld "
        f"cannot answer the question)",
        f"#   agree on {parity.agreement}/{len(parity.considered)} "
        f"({len(parity.both_feasible)} both feasible, "
        f"{len(parity.neither_feasible)} neither)",
    ]
    for engine in engines:
        others = [e for e in engines if e != engine]
        lines.append(
            f"#   {engine} only ({len(parity.only[engine])}, i.e. not "
            f"{'/'.join(others)}): {_names(parity.only[engine])}"
        )
    flagged = [d for d in shapes if not d.benign]
    benign = [d for d in shapes if d.benign]
    lines += [
        "#",
        "# MODEL SHAPE (both engines must have solved the same program):",
        f"#   {len(shapes)} disagreement(s): {len(benign)} benign, {len(flagged)} flagged",
    ]
    lines += [f"#   benign: {d.instance} {d.kind} {d.counts} -- {d.explanation}" for d in benign]
    lines += [f"#   FLAGGED: {d.instance} {d.kind} {d.counts} -- {d.explanation}" for d in flagged]
    lines += ["#", "# TRACE HEALTH (denominator: rows the engine reported feasible):"]
    for engine in engines:
        health = trace_health(rows, engine)
        lines.append(
            f"#   {engine}: {health.healthy}/{health.reported_feasible} genuine profiles, "
            f"{health.degraded} degraded to a single end point, "
            f"{health.unrecorded} with no trace_source recorded"
        )
    lines.append("#")
    return lines


def _md_table(headers: list[str], body: list[list[str]]) -> list[str]:
    if not body:
        return []
    return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"] + [
        "| " + " | ".join(cells) + " |" for cells in body
    ]


def _num(value: float, digits: int = 3) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.{digits}f}"


def render_report(
    rows: list[Scored],
    summaries: list[Summary],
    budget: float,
    roster_path: Path,
    table_path: Path,
    engines: tuple[str, ...] = ENGINES,
) -> str:
    """The parity report that is published beside the comparison table.

    Everything the re-scoped correctness sweep is for, in the order a reader
    needs it: defects first, then who reached feasibility where, then the
    cross-checks that say the two engines were asked the same question, and only
    then the anytime score the benchmark also carries.
    """
    parity = compare_feasibility(rows, engines)
    defects = collect_defects(rows, summaries, engines)
    shapes = cross_check_shapes(rows, engines)
    failures = job_failures(rows)

    out: list[str] = [
        "# MIPfeas parity report",
        "",
        f"Roster `{roster_path.name}` ({parity.roster_size} instances), "
        f"{budget}s per instance-engine pair. Table: `{table_path.name}`.",
        "",
        "CBLS against CP-SAT restricted to its `fj` + `ls` subsolvers under",
        "`num_violation_ls` -- the reference implementation of the same jump-based",
        "algorithm. This is a correctness-and-parity sweep against that one worker",
        "pairing, never a claim against any solver's default portfolio (epic #87).",
        "",
        "## 1. Defects",
        "",
    ]
    defect_rows = [
        [name, *[str(counter.get(e, 0)) for e in engines], str(sum(counter.values()))]
        for name, counter in (
            ("verification_failed", defects.verification_failed),
            ("unverified", defects.unverified),
            ("below_reference", defects.below_reference),
            ("invalid_model", defects.invalid_model),
            ("errored", defects.errored),
            ("not_run", defects.not_run),
        )
    ]
    defect_rows.append(["shape_flagged", *["-" for _ in engines], str(defects.shape_flagged)])
    defect_rows.append(["trace_degraded", *["-" for _ in engines], str(defects.trace_degraded)])
    out += _md_table(["counter", *engines, "total"], defect_rows)
    out += [
        "",
        f"**{defects.total} defect finding(s) in this run.**"
        if defects.total
        else "**No defect findings in this run.**",
        "",
        "`verification_marginal` is not a defect and is not counted above: "
        + ", ".join(f"{e}={defects.verification_marginal.get(e, 0)}" for e in engines)
        + ".",
        "",
        "## 2. Feasibility parity",
        "",
    ]
    out += _md_table(
        ["engine", "feasible", "of roster"],
        [[e, str(parity.feasible[e]), str(parity.roster_size)] for e in engines],
    )
    out += [
        "",
        f"Comparable on **{len(parity.considered)} of {parity.roster_size}** instances -- "
        "the denominator for every set below. An instance drops out when either "
        "engine's row never ran, did not search, or had its solution withheld: such a "
        "row cannot say whether feasibility was reached.",
        "",
        f"- Agreement: **{parity.agreement}/{len(parity.considered)}** "
        f"({len(parity.both_feasible)} both feasible, {len(parity.neither_feasible)} neither).",
    ]
    for engine in engines:
        others = "/".join(e for e in engines if e != engine)
        out.append(
            f"- **{engine} only** ({len(parity.only[engine])}) -- feasible for {engine}, "
            f"not for {others}: {_names(parity.only[engine], limit=10**6)}"
        )
    out += [
        "",
        f"Both feasible ({len(parity.both_feasible)}): {_names(parity.both_feasible, limit=10**6)}",
        "",
        f"Neither feasible ({len(parity.neither_feasible)}): "
        f"{_names(parity.neither_feasible, limit=10**6)}",
        "",
    ]
    if parity.excluded:
        out += ["Excluded from the parity sets:", ""]
        out += _md_table(
            ["instance", "engine", "why"], [[i, e, why] for i, e, why in parity.excluded]
        )
        out.append("")
    out += ["## 3. Job failures", ""]
    if failures:
        out += _md_table(
            ["instance", "engine", "kind", "reason"],
            [[f.instance, f.engine, f.kind, f.reason] for f in failures],
        )
    else:
        out.append("No job failed and no solution was refused.")
    out += [
        "",
        "## 4. Model-shape cross-check",
        "",
        SHAPE_RULE,
        "",
    ]
    if shapes:
        out += _md_table(
            ["instance", "kind", "counts", "verdict", "explanation"],
            [
                [d.instance, d.kind, d.counts, "benign" if d.benign else "FLAGGED", d.explanation]
                for d in shapes
            ],
        )
    else:
        out.append("Every instance reads the same shape from every reader.")
    out += ["", "## 5. Trace health", ""]
    out += _md_table(
        ["engine", "genuine", "degraded", "unrecorded", "denominator"],
        [
            [
                h.engine,
                str(h.healthy),
                str(h.degraded),
                str(h.unrecorded),
                str(h.reported_feasible),
            ]
            for h in (trace_health(rows, e) for e in engines)
        ],
    )
    out += [
        "",
        "Denominator: rows the engine **reported feasible**. A run that found nothing "
        "has no incumbent profile to have, so it is not counted here. A degraded row "
        "is one whose profile collapsed to the final objective alone -- a harness "
        "condition (a changed log format, a callback that stopped firing), not a "
        "search result, and it scores near the no-solution penalty either way.",
        "",
    ]
    named: list[str] = []
    for engine in engines:
        health = trace_health(rows, engine)
        if health.degraded_instances:
            named.append(f"- {engine} degraded: {_names(health.degraded_instances, limit=10**6)}")
        if health.unrecorded_instances:
            named.append(
                f"- {engine} with no `trace_source` recorded: "
                f"{_names(health.unrecorded_instances, limit=10**6)}"
            )
    out += [*named, ""] if named else []
    out += [f"## 6. {ANYTIME_LABEL}", ""]
    out += _md_table(
        ["engine", "sgm", "mean", "median", "iqr", "scored"],
        [
            [
                s.engine,
                _num(s.shifted_geomean, 4),
                _num(s.arithmetic_mean, 4),
                _num(s.median, 4),
                f"[{_num(s.q1, 4)}, {_num(s.q3, 4)}]",
                str(s.scored),
            ]
            for s in summaries
        ],
    )
    out += [
        "",
        "Lower is better; the Primal Integral runs from 0 (optimal immediately) to 2 "
        "(never feasible) and is budget-relative, so this is comparable only to "
        "another table scored at the same budget. `scored` excludes rows that never "
        "ran and rows whose objective was withheld.",
        "",
        "## 7. Where the time went",
        "",
    ]
    timings = [timing_summary(rows, e, budget) for e in engines]
    out += _md_table(
        ["engine", "setup measured", "setup median", "setup max", "max / budget", "overruns"],
        [
            [
                t.engine,
                f"{t.setup_measured}/{t.searched}",
                _num(t.setup_median),
                f"{_num(t.setup_max)} ({t.setup_max_instance})" if t.setup_measured else "n/a",
                f"{_num(100 * t.setup_max_budget_share, 1)}%" if t.setup_measured else "n/a",
                f"{t.overruns} (max +{_num(t.overrun_max)}s on {t.overrun_max_instance})"
                if t.overruns
                else "0",
            ]
            for t in timings
        ],
    )
    out += [
        "",
        "Two different effects, kept in two columns because a single wall-clock "
        "number cannot tell them apart:",
        "",
        "- **setup** (`setup_seconds`) is instance read + model build + bound "
        "propagation. It happens before the solve bracket, so no published MIPfeas "
        "table has ever measured it; it is not charged against the search, but it is "
        "charged against the wall clock a run has to be scheduled for.",
        "- **overrun** is `solve_seconds` past the budget. Search initialisation is "
        "not bounded by the deadline, so the first batch of a large model runs to "
        "completion whatever the clock says. A row can overrun with a negligible "
        "setup time and vice versa.",
        "",
    ]
    if all(t.setup_measured == 0 for t in timings):
        out.append(
            "**No setup time was recorded in this results directory**, so no magnitude "
            "for it is stated here. Only the mechanism above is reported. Re-run with "
            "runners that record `setup_seconds` to measure it."
        )
    elif any(t.setup_measured < t.searched for t in timings):
        out.append(
            "Some rows recorded no setup time (a results directory written before the "
            "runners measured it). The figures above cover only the rows that did, and "
            "the `setup measured` column states how many that is."
        )
    out.append("")
    return "\n".join(out) + "\n"


def read_roster(path: Path) -> list[tuple[str, float, str]]:
    with open(path, newline="") as fh:
        return [
            (row["instance"], float(row["reference_value"]), row["reference_kind"])
            for row in csv.DictReader(fh)
        ]


def _tolerance_line(rows: list[Scored]) -> str:
    """The tolerance set the verdicts were reached under, as the verdicts record it.

    Read back off the rows rather than restated here: a table must quote the
    thresholds its own verdicts used, not the ones this scorer was written
    against. `check_uniform_configuration` has already refused a mixture.
    """
    for row in rows:
        if row.verification_tolerances:
            return row.verification_tolerances
    return "none recorded (no row carried a verdict)"


def shape_notes_by_instance(
    rows: list[Scored], engines: tuple[str, ...] = ENGINES
) -> dict[str, str]:
    """Per-instance cell for the table's `shape_agreement` column.

    `agree` where every reader read the same program, `benign:<kind>` where the
    stated rule accounts for the difference, `FLAGGED:<kind>` where it does not.
    """
    notes: dict[str, str] = {r.instance: "agree" for r in rows}
    for disagreement in cross_check_shapes(rows, engines):
        prefix = "benign" if disagreement.benign else "FLAGGED"
        notes[disagreement.instance] = f"{prefix}:{disagreement.kind}"
    return notes


def _failure_reason(row: Scored) -> str:
    """The one-cell reason this row is not an ordinary search result.

    Empty for a row that searched and was accepted. The driver and the verifier
    have always recorded these; nothing published them, so a dead job reached the
    table as a blank (issue #139).
    """
    if row.status not in SEARCHED_STATUSES:
        return f"{row.status}: {row.message}" if row.message else row.status
    if row.verification in (VERIFICATION_FAIL, "error"):
        detail = row.verification_reason or "no reason recorded"
        return f"verification {row.verification}: {detail}"
    if row.withheld:
        return f"withheld: {row.verification_reason or row.verification}"
    return ""


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
        "# Verified: every row reported feasible was checked against the ORIGINAL instance",
        "#          file by benchmarks/mipfeas/verify_solution.py, which reads it with SCIP",
        "#          -- a reader neither engine uses, so a shared reader defect cannot cancel",
        "#          out. Row activities, variable bounds, integrality and the objective are",
        "#          all re-derived there. A row that fails, or that carries no verdict,",
        "#          publishes no objective and no Primal Integral and is left out of the",
        "#          aggregates; the verification / verification_reason columns say why.",
        "#          A `marginal` pass came within 10% of a tolerance: a flag, not a failure.",
        f"#          Tolerances: {_tolerance_line(rows)}",
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
    unverified_published = [
        r
        for r in rows
        if r.status == "feasible" and not r.withheld and r.verification != VERIFICATION_PASS
    ]
    if unverified_published:
        # The "Verified:" note above is unconditional, and --allow-unverified makes
        # it untrue for these rows while leaving them in the aggregates. Banner it
        # for the same reason a partial roster is bannered: the per-row column is
        # not what a reader quoting this table will look at.
        header += [
            "# *** SCORED WITH --allow-unverified — NOT A PUBLISHABLE RESULT ***",
            f"# {len(unverified_published)} feasible row(s) are published with no independent",
            "# verdict; the 'Verified:' note above does not hold for them.",
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
    # Parity and defects lead, the anytime score follows. The re-scoped run is a
    # correctness sweep, and a reader quoting this file reads the header.
    header += headline_lines(rows, summaries)
    header.append(f"# {ANYTIME_LABEL}:")
    for s in summaries:
        header.append(
            f"#   {s.engine:<6} sgm={s.shifted_geomean:.4f} mean={s.arithmetic_mean:.4f} "
            f"median={s.median:.4f} iqr=[{s.q1:.4f},{s.q3:.4f}] "
            f"feasible={s.feasible}/{s.scored} matched_reference={s.matched_reference} "
            f"invalid_model={s.invalid_model} errored={s.errored} "
            f"below_reference={s.below_reference} verification_failed={s.verification_failed} "
            f"unverified={s.unverified} verification_marginal={s.verification_marginal} "
            f"not_run={s.not_run}"
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
                "solve_seconds",
                "setup_seconds",
                "n_vars",
                "n_cons",
                "n_free_cons",
                "shape_agreement",
                "peak_rss_kib",
                "n_clamped_bounds",
                "n_unbounded_columns",
                "n_bounds_tightened",
                "trace_source",
                "solver_status",
                "verification",
                "verification_reason",
                "verification_marginal",
                "failure_reason",
                "provenance",
                "config",
            ]
        )
        # One lookup per instance rather than per row: the cross-check is a
        # property of the pair, and both of an instance's rows carry its verdict.
        shape_notes = shape_notes_by_instance(rows)
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
                    "" if r.solve_seconds is None else f"{r.solve_seconds:.4f}",
                    "" if r.setup_seconds is None else f"{r.setup_seconds:.4f}",
                    "" if r.n_vars is None else r.n_vars,
                    "" if r.n_cons is None else r.n_cons,
                    "" if r.n_free_cons is None else r.n_free_cons,
                    shape_notes[r.instance],
                    "" if r.peak_rss_kib is None else r.peak_rss_kib,
                    "" if r.n_clamped_bounds is None else r.n_clamped_bounds,
                    "" if r.n_unbounded_columns is None else r.n_unbounded_columns,
                    "" if r.n_bounds_tightened is None else r.n_bounds_tightened,
                    r.trace_source,
                    r.solver_status,
                    r.verification,
                    r.verification_reason,
                    int(r.verification_marginal),
                    _failure_reason(r),
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
    parser.add_argument(
        "--report",
        default=None,
        help="parity report to write (default: <out stem>_report.md beside the table). "
        "Leads with the defect counters and the feasibility difference sets, which is "
        "what the re-scoped correctness sweep is for; the anytime aggregate is kept "
        "beside them, not replaced by them",
    )
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="publish a feasible row that carries no verdict from verify_solution.py. "
        "For an older results directory only: a run made since #138 verifies as it "
        "goes, and a row nobody checked is what that issue exists to stop publishing. "
        "Never relaxes a verdict that says `fail`",
    )
    args = parser.parse_args()

    roster_path = Path(args.roster)
    roster = read_roster(roster_path)
    results_dir = Path(args.results_dir)

    rows = [
        score_instance(
            name,
            engine,
            value,
            kind,
            results_dir,
            args.budget,
            require_verification=not args.allow_unverified,
        )
        for name, value, kind in roster
        for engine in ENGINES
    ]
    summaries = [summarize(rows, engine) for engine in ENGINES]
    check_uniform_configuration(rows)

    out_path = Path(args.out)
    write_comparison(out_path, rows, summaries, args.budget, roster_path)
    report_path = (
        Path(args.report) if args.report else out_path.with_name(f"{out_path.stem}_report.md")
    )
    report_path.write_text(render_report(rows, summaries, args.budget, roster_path, out_path))

    parity = compare_feasibility(rows)
    defects = collect_defects(rows, summaries)
    print(
        f"Scored {len(roster)} instances x {len(ENGINES)} engines at {args.budget}s -> {out_path}"
    )
    print(f"Parity report -> {report_path}")
    print(
        f"  defects={defects.total}  comparable={len(parity.considered)}/{parity.roster_size}  "
        f"agreement={parity.agreement}/{len(parity.considered)}  "
        + "  ".join(f"{e}_only={len(parity.only[e])}" for e in ENGINES)
    )
    for s in summaries:
        print(
            f"  {s.engine:<6} sgm={s.shifted_geomean:.4f} mean={s.arithmetic_mean:.4f} "
            f"median={s.median:.4f} feasible={s.feasible}/{s.scored} "
            f"matched_reference={s.matched_reference} "
            f"verification_failed={s.verification_failed} unverified={s.unverified} "
            f"not_run={s.not_run}"
        )
    rejected = [s for s in summaries if s.verification_failed]
    if rejected:
        # Louder than the incomplete-run warning below, because this is the one
        # thing the benchmark exists to detect: a solution the engine reported
        # feasible that is not feasible for the program in the file.
        print(
            "\nDEFECT: "
            + ", ".join(
                f"{s.engine}: {s.verification_failed} solution(s) rejected by the independent check"
                for s in rejected
            )
            + ". Their objectives are withheld; see the verification_reason column "
            "and the .verify.json files.",
            file=sys.stderr,
        )
    unverified = [s for s in summaries if s.unverified]
    if unverified:
        consequence = (
            "Those rows publish no objective; re-run them so they are checked."
            if not args.allow_unverified
            else "They were PUBLISHED UNCHECKED because --allow-unverified was given."
        )
        print(
            "\nWARNING: "
            + ", ".join(
                f"{s.engine}: {s.unverified} feasible row(s) with no verdict" for s in unverified
            )
            + ". "
            + consequence,
            file=sys.stderr,
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
    # Non-zero when a solution was rejected: a correctness benchmark whose
    # checker refused a published point must not score at exit 0.
    return 1 if rejected else 0


if __name__ == "__main__":
    sys.exit(main())
