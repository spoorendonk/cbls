"""MINLPLib baseline: SCIP spatial branch-and-bound on the CBLS roster.

Independently-run open-source yardstick for the headline nonlinear result. The
roster of record is `bounds.csv` — the same file `cbls_minlplib` reads — and SCIP
consumes the **same `.nl` files** through its own AMPL reader, so neither solver
sees a re-modelled instance and no formulation drift can enter the comparison.

Why SCIP (issue #89): BARON is commercial and only reachable through the NEOS
queue, so it cannot be batch-run reproducibly; Couenne is free but has seen
little development since ~2018 and is generally outperformed on this family.
SCIP's nonconvex spatial branch-and-bound is purpose-built and separately
benchmarked on MINLPLib in *Global Optimization of Mixed-Integer Nonlinear
Programs with SCIP 8.0*
(https://optimization-online.org/wp-content/uploads/2022/12/scip8_minlp.pdf).

What is and is not matched between the two runs:

* **Matched** — instance files, roster and its order, per-instance wall-clock
  budget, one thread, and the feasibility tolerance (CBLS defaults to 1e-6;
  SCIP's `numerics/feastol` default is also 1e-6).
* **Not matched, by construction** — SCIP is a complete global solver and proves
  a dual bound; CBLS is a primal heuristic and proves none. Only the *primal*
  columns are a like-for-like comparison. `dual_bound` is reported per method so
  that asymmetry is visible in the data rather than hidden by it.
* **Asymmetric verification** — the C++ runner re-verifies a returned assignment
  against the model it built. Here the check is `Model.checkSol(original=True)`,
  i.e. SCIP validating its own solution against the pre-presolve problem. A
  solution SCIP cannot validate is not published as feasible.

Like the CBLS run, the budget is wall-clock, so these are single-sample numbers:
a fixed seed does not pin the node count and consecutive runs differ.

Requires the `benchmarks` extra (`pip install -e '.[benchmarks]'`). Run with the
project venv:

    .venv/bin/python3 benchmarks/minlplib/reference_solve.py
    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --time-limit 60
    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --merge-only

A subset run must name scratch outputs — it refuses the published paths, so a
three-instance debug run cannot replace a fifty-row table:

    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --instances nvs01 tln2 \
        --out /tmp/scip_subset.csv --merged-out /tmp/comparison_subset.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyscipopt import Model

#: Default roster/output directory: the same one `cbls_minlplib` reads.
DEFAULT_INST_DIR = Path(__file__).resolve().parents[1] / "instances" / "minlplib"

#: Seconds per instance. Matches the published CBLS run (issue #88); the 60s
#: figure is argued from that run's committed anytime trace, not assumed.
DEFAULT_TIME_LIMIT = 60.0

#: SCIP's own default. Left unset rather than assigned, so that a SCIP release
#: changing it surfaces as a mismatch instead of silently loosening one side of
#: the comparison — `solve_instance` reads the live value back and flags it.
SCIP_FEASTOL = 1e-6

CBLS_METHOD = "cbls"
SCIP_METHOD = "scip"
BKS_METHOD = "published-bks"


def safe_gap(obj: float, ref: float, maximizing: bool) -> float:
    """Signed gap in percent, positive == worse than `ref` in either sense.

    Port of `safe_gap` in `benchmarks/minlplib/minlplib.cpp`; kept numerically
    identical so a SCIP row and a CBLS row for the same instance are comparable.

    NOTE: when ``|ref| < 1e-12`` the result is an ABSOLUTE residual, not a
    percent — a percentage against zero is meaningless. The roster's five such
    instances are listed in the instances README.
    """
    if math.isnan(obj) or math.isnan(ref):
        return math.nan
    diff = (ref - obj) if maximizing else (obj - ref)
    if abs(ref) < 1e-12:
        return diff
    return 100.0 * diff / abs(ref)


def classify_vs_bks(obj: float, bks: float, maximizing: bool, feas_tol: float) -> str:
    """Label a feasible objective against the published primal bound.

    Same two-band rule as the C++ runner, so the note columns mean the same
    thing for both methods. An improvement is only *claimed* when it clears
    ``max(1e-6·(|BKS|+1), 10·feas_tol)`` — a solver may violate a constraint by
    up to `feas_tol`, and that slack alone buys a small objective gain. Calling
    two objectives *equal* needs the much tighter, purely relative band; a row
    between the two is neither a win nor "worse than BKS", so it gets its own
    label.
    """
    if math.isnan(bks):
        return "feasible"
    win_slack = max(1e-6 * (abs(bks) + 1.0), 10.0 * feas_tol)
    tie_band = 1e-6 * (abs(bks) + 1.0)
    diff = obj - bks
    improvement = diff if maximizing else -diff
    if improvement > win_slack:
        return "better-than-bks"
    if abs(diff) <= tie_band:
        return "matches-bks"
    if improvement > 0.0:
        return "within-tolerance-of-bks"
    return "feasible"


def _reads_as_max(objsense: str) -> bool:
    """The one place a sense string is interpreted.

    Both the catalogue's spelling and SCIP's go through here, so the two can
    never disagree about the identical string — a divergence would invert every
    gap on a maximize row *and* silence the mismatch note, since both sides
    would then agree on the wrong answer.
    """
    return objsense.strip().lower().startswith("max")


@dataclass
class Bound:
    """One `bounds.csv` row: the roster of record."""

    instance: str
    structure: str
    objsense: str
    primal_bks: float
    dual_bound: float
    n_disc_vars_bks: int

    @property
    def maximizing(self) -> bool:
        return _reads_as_max(self.objsense)


@dataclass
class ScipResult:
    """One SCIP solve, in the instance's original objective sense."""

    instance: str
    status: str = "not-run"
    objective: float = math.nan
    dual_bound: float = math.nan
    scip_gap: float = math.nan
    feasible: bool = False
    wall_seconds: float = 0.0
    read_seconds: float = 0.0
    solving_seconds: float = 0.0
    n_int_vars: int = -1
    #: The sense SCIP read from the instance, cross-checked against the catalogue.
    objsense: str = ""
    #: `SCIP x / PySCIPOpt y / Ts / seed n` this row came from. Empty on a fresh
    #: solve, where the run-wide string applies; restored per row by
    #: `read_scip_csv` so a results file spanning two runs keeps each row's own
    #: provenance instead of inheriting the first row's.
    version: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def note(self) -> str:
        return "; ".join(self.notes) if self.notes else ""


def _to_float(cell: str) -> float:
    try:
        return float(cell)
    except ValueError:
        return math.nan


def load_bounds(path: Path) -> list[Bound]:
    """Read `bounds.csv` in file order — the roster and its order of record."""
    rows: list[Bound] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            n_disc = _to_float(row.get("n_disc_vars_bks", ""))
            rows.append(
                Bound(
                    instance=row["instance"],
                    structure=row.get("structure", ""),
                    objsense=row.get("objsense", "min"),
                    primal_bks=_to_float(row.get("primal_bks", "")),
                    dual_bound=_to_float(row.get("dual_bound", "")),
                    n_disc_vars_bks=-1 if math.isnan(n_disc) else int(n_disc),
                )
            )
    return rows


def load_cbls_rows(path: Path) -> dict[str, dict[str, str]]:
    """Read the CBLS `comparison.csv`, if the runner has been run. Optional."""
    if not path.exists():
        return {}
    with path.open(newline="") as fh:
        return {row["instance"]: row for row in csv.DictReader(fh)}


def solve_instance(nl_path: Path, time_limit: float, seed: int, log: bool) -> ScipResult:
    """Read one `.nl` with SCIP's AMPL reader and solve it to the time limit."""
    # Imported here, not at module scope: pyscipopt lives in the optional
    # `benchmarks` extra, and --merge-only must work without it installed.
    from pyscipopt import Model

    result = ScipResult(instance=nl_path.stem)
    model = Model()
    if not log:
        model.hideOutput()

    t0 = time.perf_counter()
    try:
        model.readProblem(str(nl_path))
    except Exception as exc:  # any reader failure becomes one CSV row, not a crash
        result.status = "read-error"
        result.notes.append(f"read-error({type(exc).__name__}: {exc})")
        result.wall_seconds = time.perf_counter() - t0
        return result
    result.read_seconds = time.perf_counter() - t0

    # `getVars()` is the ORIGINAL problem (PySCIPOpt defaults transformed=False),
    # which is what the catalogue's nbinvars+nintvars counts. The transformed
    # problem carries presolve's aggregations instead. SCIP's .nl reader also adds
    # a continuous `nlobjvar`, so a raw variable count would not match the
    # catalogue even where the discrete count does.
    result.n_int_vars = sum(1 for v in model.getVars() if v.vtype() in ("BINARY", "INTEGER"))
    result.objsense = model.getObjectiveSense()

    model.setParam("limits/time", time_limit)
    model.setParam("randomization/randomseedshift", seed)
    # feastol is deliberately left at SCIP's default so both runs share one
    # tolerance. That only holds while the default still *is* the value the BKS
    # classification is computed from, so read it back rather than assume it.
    live_feastol = float(model.getParam("numerics/feastol"))
    if live_feastol != SCIP_FEASTOL:
        result.notes.append(f"FEASTOL-MISMATCH(SCIP {live_feastol:g} vs assumed {SCIP_FEASTOL:g})")

    t1 = time.perf_counter()
    try:
        model.optimize()
    except Exception as exc:  # one bad instance must not end a 50-minute run
        result.wall_seconds = time.perf_counter() - t1
        result.status = "solve-error"
        result.notes.append(f"solve-error({type(exc).__name__}: {exc})")
        return result
    result.wall_seconds = time.perf_counter() - t1
    result.status = model.getStatus()
    result.solving_seconds = model.getSolvingTime()
    result.dual_bound = _finite_or_nan(model, model.getDualbound())
    result.scip_gap = _finite_or_nan(model, model.getGap())

    _record_solution(model, result)
    model.freeProb()
    return result


def _finite_or_nan(model: Model, value: float) -> float:
    """Normalise SCIP's "no bound proved" sentinel to NaN.

    SCIP reports an unproved bound, and an undefined gap, as ±SCIPinfinity —
    which is 1e20, a *finite* float. `math.isfinite` accepts it, so without this
    it would be published as a bound the solver never proved, in the very column
    the merged CSV documents as "what that method proves".

    The 0.9 factor is not slack for its own sake: `getDualbound` reports in the
    original objective space, so an instance whose objective carries an offset
    comes back as e.g. -9.999999980870551e19 rather than exactly -1e20, and
    `Model.isInfinity` misses it. Genuine bounds in this roster peak around
    7e10, nine orders below the threshold.
    """
    return math.nan if abs(value) >= 0.9 * model.infinity() else value


def _record_solution(model: Model, result: ScipResult) -> None:
    """Publish SCIP's incumbent only if SCIP can re-validate it."""
    # An unbounded problem still carries a solution — an arbitrary point on the
    # ray, which `checkSol` accepts — and publishing it would assert a finite
    # optimum the instance does not have.
    if result.status in ("unbounded", "inforunbd"):
        result.notes.append(f"no-published-objective({result.status})")
        return
    if model.getNSols() == 0:
        result.notes.append(f"no-solution({result.status})")
        return
    best = model.getBestSol()
    # `original=True` checks against the pre-presolve problem, so a solution
    # that only satisfies a presolved relaxation cannot slip through.
    if not model.checkSol(best, printreason=False, completely=True, original=True):
        result.notes.append("CHECK-FAILED(SCIP could not re-validate its own solution)")
        return
    objective = model.getObjVal()
    if not math.isfinite(objective):
        result.notes.append("non-finite objective")
        return
    result.objective = objective
    result.feasible = True


def annotate(result: ScipResult, bound: Bound) -> None:
    """Attach the BKS classification and the cross-checks worth failing loudly."""
    if result.feasible:
        result.notes.insert(
            0,
            classify_vs_bks(
                result.objective, bound.primal_bks, sense_is_max(bound, result), SCIP_FEASTOL
            ),
        )
    if result.status == "optimal":
        result.notes.append("proved-optimal")
    # `sense_is_max` scores this row by the instance's sense, matching the C++
    # runner, which takes it from the model it built. A disagreement therefore no
    # longer flips the gap's sign against the C++ row — but it does mean the
    # catalogue's primal/dual here are oriented for the other sense, so both gaps
    # are measured against a reference one of the two sides has wrong.
    if result.objsense and _reads_as_max(result.objsense) != bound.maximizing:
        result.notes.append(
            f"objsense mismatch: instance {result.objsense} vs catalogue {bound.objsense}"
        )
    if (
        result.n_int_vars >= 0
        and bound.n_disc_vars_bks >= 0
        and result.n_int_vars != bound.n_disc_vars_bks
    ):
        result.notes.append(
            f"integrality mismatch: SCIP {result.n_int_vars} vs catalogue {bound.n_disc_vars_bks}"
        )
    # SCIP's own dual bound crossing the catalogue's primal bound means one of
    # the two published numbers is wrong; that is worth surfacing, not hiding.
    if not math.isnan(bound.primal_bks) and math.isfinite(result.dual_bound):
        crossed = (
            result.dual_bound < bound.primal_bks - 1e-6 * (abs(bound.primal_bks) + 1.0)
            if sense_is_max(bound, result)
            else result.dual_bound > bound.primal_bks + 1e-6 * (abs(bound.primal_bks) + 1.0)
        )
        if crossed:
            result.notes.append(
                f"SCIP dual {result.dual_bound:.6g} crosses the published primal "
                f"{bound.primal_bks:.6g}"
            )


def solver_version(time_limit: float, seed: int) -> str:
    """`SCIP <v> / PySCIPOpt <v> / <T>s / seed <n>` — the run configuration.

    Budget and seed belong in the provenance string, not just the version: both
    CSVs are rewritten whole, so a re-run at a different budget would otherwise
    produce rows indistinguishable from the published ones, and silently
    comparing unequal budgets is the one mistake this table cannot survive.
    """
    from pyscipopt import Model  # optional `benchmarks` extra; see solve_instance

    try:
        binding = pkg_version("pyscipopt")
    except PackageNotFoundError:  # pragma: no cover — source checkouts only
        binding = "unknown"
    scip = Model()
    # Not `version()`: it returns a float, so 10.0.2 renders as "10.0" and a
    # future 10.10 would render as "10.1", colliding with 10.1.x.
    core = f"{scip.getMajorVersion()}.{scip.getMinorVersion()}.{scip.getTechVersion()}"
    return f"SCIP {core} / PySCIPOpt {binding} / {time_limit:g}s / seed {seed}"


SCIP_CSV_HEADER = [
    "instance",
    "objective",
    "primal_bks",
    "dual_bound",
    "gap_to_bks%",
    "gap_to_dual%",
    "wall_seconds",
    "feasible",
    "note",
    "scip_dual_bound",
    "scip_gap%",
    "status",
    "n_int_vars",
    "read_seconds",
    "solving_seconds",
    "scip_version",
]


def _cell(value: float) -> str:
    """`NaN` for undefined, matching what the C++ runner writes.

    SCIP never hands back a float `inf` — it uses a 1e20 sentinel, which
    `_finite_or_nan` folds into NaN at capture time, so "not proved" and "not
    computed" share the one spelling the C++ runner already uses.
    """
    return "NaN" if math.isnan(value) else repr(value)


def _published_objective(result: ScipResult) -> float:
    """The objective only if we stand behind it, else NaN.

    Same rule the C++ runner applies: a row that failed its feasibility check
    must not publish the number it was rejected for.
    """
    return result.objective if result.feasible else math.nan


def sense_is_max(bound: Bound, result: ScipResult) -> bool:
    """The objective sense to score by: the instance's, else the catalogue's.

    The C++ runner takes the sense from the model it built, not from
    `bounds.csv`, and the instance file is the ground truth — the catalogue is
    derived metadata from a library that drifts. Scoring by the catalogue would
    let a one-word error there invert every gap and label on the row *before*
    `annotate`'s cross-check could report the disagreement.

    Falls back to the catalogue whenever SCIP never read the instance: a
    `not-found` or `read-error` row, and every `--merge-only` reload, since
    `scip_baseline.csv` has no `objsense` column to restore. On a row where the
    two senses disagree the merge therefore scores it differently from the fresh
    run — `_run_merge_only` refuses rather than publish that contradiction.
    """
    return _reads_as_max(result.objsense) if result.objsense else bound.maximizing


def _gap_to_bks(result: ScipResult, bound: Bound) -> float:
    """Gap only for a row we publish as feasible; NaN otherwise.

    Companion to `_published_objective`: a row that failed its check must not
    carry the gap it was rejected for, in either CSV or on the console.
    """
    if not result.feasible:
        return math.nan
    return safe_gap(result.objective, bound.primal_bks, sense_is_max(bound, result))


def write_scip_csv(path: Path, rows: Sequence[tuple[Bound, ScipResult]], version: str) -> None:
    """SCIP-only results. Columns mirror `comparison.csv` where they mean the
    same thing, then add the bound/gap/status only a complete solver produces.
    """
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(SCIP_CSV_HEADER)
        for bound, result in rows:
            gap_bks = _gap_to_bks(result, bound)
            gap_dual = (
                safe_gap(result.objective, bound.dual_bound, sense_is_max(bound, result))
                if result.feasible
                else math.nan
            )
            out.writerow(
                [
                    result.instance,
                    _cell(_published_objective(result)),
                    _cell(bound.primal_bks),
                    _cell(bound.dual_bound),
                    _cell(gap_bks),
                    _cell(gap_dual),
                    repr(result.wall_seconds),
                    "true" if result.feasible else "false",
                    # No comma scrubbing: csv.writer quotes the field. The C++
                    # runner has to scrub because it writes raw CSV, and a
                    # scrubbed comma would corrupt "; " — the note separator.
                    result.note,
                    _cell(result.dual_bound),
                    _cell(100.0 * result.scip_gap),
                    result.status,
                    result.n_int_vars,
                    repr(result.read_seconds),
                    repr(result.solving_seconds),
                    version,
                ]
            )


MERGED_CSV_HEADER = [
    "instance",
    "method",
    "version",
    "objective",
    "feasible",
    "wall_seconds",
    "gap_to_bks%",
    "dual_bound",
    "note",
]


def _bks_row(bound: Bound) -> list[str]:
    # An absent catalogue primal is not a feasible solution. Every consumer here
    # filters on `feasible`, so a NaN objective on a "true" row would be counted
    # as a solved instance with no objective.
    have_primal = not math.isnan(bound.primal_bks)
    return [
        bound.instance,
        BKS_METHOD,
        "minlplib-catalogue",
        _cell(bound.primal_bks),
        "true" if have_primal else "false",
        "NaN",  # catalogue values carry no budget comparable to a 60s run
        "0.0" if have_primal else "NaN",
        _cell(bound.dual_bound),
        f"published primal/dual bound; structure={bound.structure}",
    ]


def _cbls_row(bound: Bound, row: dict[str, str]) -> list[str]:
    return [
        bound.instance,
        CBLS_METHOD,
        f"cbls@{row.get('commit_sha', 'unknown')}",
        row.get("objective", "NaN"),
        row.get("feasible", "false"),
        row.get("wall_seconds", "NaN"),
        row.get("gap_to_bks%", "NaN"),
        "NaN",  # a primal heuristic proves no dual bound
        row.get("note", ""),
    ]


def _scip_row(bound: Bound, result: ScipResult, version: str) -> list[str]:
    return [
        bound.instance,
        SCIP_METHOD,
        # The row's own provenance wins: a results CSV can span more than one run.
        result.version or version,
        _cell(_published_objective(result)),
        "true" if result.feasible else "false",
        repr(result.wall_seconds),
        _cell(_gap_to_bks(result, bound)),
        _cell(result.dual_bound),
        result.note,
    ]


def write_merged_csv(
    path: Path,
    bounds: Sequence[Bound],
    cbls_rows: dict[str, dict[str, str]],
    scip_rows: dict[str, ScipResult],
    version: str,
) -> None:
    """Long-format three-way comparison, one row per (instance, method).

    `dual_bound` is the bound *that method proves*: the catalogue's for
    `published-bks`, SCIP's own for `scip`, and NaN for `cbls`, which as a primal
    heuristic proves none. It is deliberately not the published dual repeated on
    every row — that would read as if all three had proved the same thing.
    """
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(MERGED_CSV_HEADER)
        for bound in bounds:
            out.writerow(_bks_row(bound))
            if bound.instance in cbls_rows:
                out.writerow(_cbls_row(bound, cbls_rows[bound.instance]))
            if bound.instance in scip_rows:
                out.writerow(_scip_row(bound, scip_rows[bound.instance], version))


def read_scip_csv(path: Path) -> dict[str, ScipResult]:
    """Reload a previous `scip_baseline.csv` so `--merge-only` needs no solve.

    Restores the fields the merge publishes, plus `status` for the coverage
    warning; the SCIP-specific columns (`scip_gap%`, `n_int_vars`, the two timing
    splits) are not read back — they live only in that file.
    """
    if not path.exists():
        return {}
    rows: dict[str, ScipResult] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            result = ScipResult(instance=row["instance"])
            result.objective = _to_float(row["objective"])
            result.dual_bound = _to_float(row["scip_dual_bound"])
            result.feasible = row["feasible"] == "true"
            result.wall_seconds = _to_float(row["wall_seconds"])
            result.status = row["status"]
            result.version = row.get("scip_version", "")
            if row["note"]:
                result.notes.append(row["note"])
            rows[result.instance] = result
    return rows


def _print_row(bound: Bound, result: ScipResult) -> None:
    obj = f"{result.objective:12.4g}" if result.feasible else f"{'INFEAS':>12}"
    bks = f"{bound.primal_bks:12.4g}" if not math.isnan(bound.primal_bks) else f"{'?':>12}"
    gap = _gap_to_bks(result, bound)
    gap_s = f"{'N/A':>10}" if math.isnan(gap) else f"{gap:9.2f}%"
    print(f"{result.instance:<22} {obj} {bks} {gap_s} {result.wall_seconds:8.2f}s  {result.note}")
    sys.stdout.flush()


def _print_tally(rows: Sequence[tuple[Bound, ScipResult]], time_limit: float, seed: int) -> None:
    feasible = [r for _, r in rows if r.feasible]
    notes = [n for _, r in rows for n in r.notes]
    print("\n=== Tally (SCIP) ===")
    print(f"time limit:           {time_limit:.0f}s/instance, seed shift {seed}")
    print(f"roster:               {len(rows)}")
    print(f"feasible:             {len(feasible)}")
    for label in ("matches-bks", "better-than-bks", "within-tolerance-of-bks", "proved-optimal"):
        print(f"  {label + ':':<24} {sum(1 for n in notes if n == label)}")
    print(f"read/solve errors:    {sum(1 for _, r in rows if r.status.endswith('-error'))}")
    print(f"not found:            {sum(1 for _, r in rows if r.status == 'not-found')}")
    print(f"check failures:       {sum(1 for n in notes if n.startswith('CHECK-FAILED'))}")
    print(f"feastol mismatches:   {sum(1 for n in notes if n.startswith('FEASTOL-MISMATCH'))}")
    print(f"objsense mismatches:  {sum(1 for n in notes if n.startswith('objsense mismatch'))}")


def _select(bounds: Sequence[Bound], wanted: Sequence[str]) -> list[Bound]:
    if not wanted:
        return list(bounds)
    by_name = {b.instance: b for b in bounds}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        # rc 2 for a usage error, kept distinct from main()'s rc 1, which means
        # "the run completed but some instance failed to read or solve".
        print(f"not in bounds.csv: {', '.join(missing)}", file=sys.stderr)
        raise SystemExit(2)
    return [by_name[name] for name in wanted]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument(
        "--seed", type=int, default=0, help="SCIP randomization/randomseedshift (default 0)"
    )
    # nargs="+", not "*": a bare `--instances` would otherwise yield [], slip past
    # the subset guard, and run the whole roster into the published paths.
    parser.add_argument("--instances", nargs="+", default=[], help="subset; default whole roster")
    parser.add_argument(
        "--out", type=Path, default=None, help="default <inst-dir>/scip_baseline.csv"
    )
    parser.add_argument(
        "--merged-out", type=Path, default=None, help="default <inst-dir>/comparison_all.csv"
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="rebuild the merged CSV from existing results without solving",
    )
    parser.add_argument("--log", action="store_true", help="show SCIP's solver log")
    return parser.parse_args(argv)


def read_versions(path: Path) -> list[str]:
    """Distinct `scip_version` values in a results CSV, in first-seen order."""
    if not path.exists():
        return []
    seen: list[str] = []
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            value = row.get("scip_version", "")
            if value and value not in seen:
                seen.append(value)
    return seen


def _usage_error(args: argparse.Namespace) -> str | None:
    """The reason to refuse this invocation outright, or None to proceed.

    Every case here would otherwise publish something misleading rather than
    fail: a budget SCIP cannot use, a seed it rejects deep in the loop, or a
    subset run aimed at the roster-wide output paths.
    """
    if args.time_limit <= 0.0:
        # A non-positive budget makes SCIP return before it starts, publishing a
        # full roster of "infeasible" rows that say nothing about the solver and
        # are indistinguishable from a real result. The C++ runner refuses it too.
        return f"--time-limit must be > 0 (got {args.time_limit})"
    if args.seed < 0:
        # SCIP's own range check would fire mid-loop, as a bare traceback.
        return f"--seed must be >= 0 (got {args.seed})"
    if args.instances and (args.out is None or args.merged_out is None):
        # Both CSVs are rewritten whole, so a three-instance debug run would
        # silently replace fifty rows with three. Make the caller name a scratch
        # path rather than defaulting into the published one.
        return (
            "--instances is a subset run; pass explicit --out and --merged-out so it "
            "cannot overwrite the published roster-wide CSVs."
        )
    return None


def _run_merge_only(
    out_csv: Path,
    merged_csv: Path,
    bounds: Sequence[Bound],
    cbls_rows: dict[str, dict[str, str]],
) -> int:
    """Rebuild the merged CSV from an existing results file, without solving."""
    scip_rows = read_scip_csv(out_csv)
    if not scip_rows:
        print(f"{out_csv} not found or empty; nothing to merge.", file=sys.stderr)
        return 2
    # `scip_baseline.csv` has no objsense column, so a reload scores by the
    # catalogue while the fresh run scored by the instance. On a row where the
    # two disagree that flips the gap's sign, and the two published files would
    # contradict each other on the same instance. Refuse instead: the fresh run
    # already recorded which rows are affected.
    conflicted = sorted(n for n, r in scip_rows.items() if "objsense mismatch" in r.note)
    if conflicted:
        print(
            f"{out_csv} has {len(conflicted)} row(s) whose instance and catalogue objective "
            f"senses disagree ({', '.join(conflicted)}). The merge cannot recover the "
            "instance's sense from this file, so it would publish a differently-signed gap "
            "than the run that produced it. Fix bounds.csv and re-run the roster.",
            file=sys.stderr,
        )
        return 2
    version = next(iter(read_versions(out_csv)), "unknown")
    write_merged_csv(merged_csv, bounds, cbls_rows, scip_rows, version)
    # A crashed run leaves a truncated results file, and the merge would
    # otherwise publish a silently shorter table at exit 0.
    absent = [b.instance for b in bounds if b.instance not in scip_rows]
    if absent:
        shown = ", ".join(absent[:5]) + ("..." if len(absent) > 5 else "")
        print(
            f"WARNING: {len(absent)} roster instances have no SCIP row ({shown})", file=sys.stderr
        )
    print(f"wrote {merged_csv} ({len(bounds)} roster rows, {len(scip_rows)} with SCIP results)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    refusal = _usage_error(args)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    inst_dir: Path = args.inst_dir
    out_csv: Path = args.out or inst_dir / "scip_baseline.csv"
    merged_csv: Path = args.merged_out or inst_dir / "comparison_all.csv"

    bounds_path = inst_dir / "bounds.csv"
    if not bounds_path.exists():
        print(f"{bounds_path} not found. Fetch the roster first:", file=sys.stderr)
        print(f"  python {inst_dir}/download.py", file=sys.stderr)
        return 2
    bounds = load_bounds(bounds_path)
    if not bounds:
        # Both CSVs are rewritten whole, so an empty roster would replace the
        # published files with header-only stubs and still exit 0.
        print(f"{bounds_path} has no rows; refusing to rewrite the CSVs.", file=sys.stderr)
        return 2
    cbls_rows = load_cbls_rows(inst_dir / "comparison.csv")
    if not cbls_rows:
        print(
            f"no CBLS rows in {inst_dir / 'comparison.csv'}; the merged CSV will "
            "carry two methods, not three. Run cbls_minlplib first.",
            file=sys.stderr,
        )

    if args.merge_only:
        return _run_merge_only(out_csv, merged_csv, bounds, cbls_rows)

    version = solver_version(args.time_limit, args.seed)
    selected = _select(bounds, args.instances)
    print(f"{version}, {args.time_limit:.0f}s/instance, seed shift {args.seed}")
    print(f"{'instance':<22} {'objective':>12} {'BKS':>12} {'gap':>10} {'wall':>9}  note")

    results: list[tuple[Bound, ScipResult]] = []
    for bound in selected:
        nl_path = inst_dir / f"{bound.instance}.nl"
        if not nl_path.exists():
            result = ScipResult(instance=bound.instance, status="not-found")
            result.notes.append("not-found")
        else:
            result = solve_instance(nl_path, args.time_limit, args.seed, args.log)
            annotate(result, bound)
        results.append((bound, result))
        _print_row(bound, result)
        # Written every iteration: a 50-minute run must not lose everything to a
        # crash on the last instance.
        write_scip_csv(out_csv, results, version)

    _print_tally(results, args.time_limit, args.seed)
    # Only on a complete run, unlike the per-instance CSV above: the merged file
    # is the published artifact, so a crashed run should leave the last complete
    # one in place rather than half-replace it. `--merge-only` rebuilds it from
    # whatever the per-instance CSV holds.
    write_merged_csv(merged_csv, bounds, cbls_rows, {r.instance: r for _, r in results}, version)
    print(f"\nwrote {out_csv}\nwrote {merged_csv}")
    return 1 if any(r.status.endswith("-error") for _, r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
