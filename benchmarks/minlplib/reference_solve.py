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
    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --instances nvs01 tln2
    .venv/bin/python3 benchmarks/minlplib/reference_solve.py --merge-only
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

#: SCIP's own default. Stated rather than set, so a future SCIP release changing
#: it shows up as a mismatch against the CBLS tolerance instead of being masked.
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
        return self.objsense.strip().lower().startswith("max")


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

    # Count integrality on the ORIGINAL problem: after optimize() getVars()
    # returns transformed columns, which presolve may have fixed or aggregated.
    result.n_int_vars = sum(1 for v in model.getVars() if v.vtype() in ("BINARY", "INTEGER"))

    model.setParam("limits/time", time_limit)
    model.setParam("randomization/randomseedshift", seed)

    t1 = time.perf_counter()
    model.optimize()
    result.wall_seconds = time.perf_counter() - t1
    result.status = model.getStatus()
    result.solving_seconds = model.getSolvingTime()
    result.dual_bound = model.getDualbound()
    result.scip_gap = model.getGap()

    _record_solution(model, result)
    model.freeProb()
    return result


def _record_solution(model: Model, result: ScipResult) -> None:
    """Publish SCIP's incumbent only if SCIP can re-validate it."""
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
            0, classify_vs_bks(result.objective, bound.primal_bks, bound.maximizing, SCIP_FEASTOL)
        )
    if result.status == "optimal":
        result.notes.append("proved-optimal")
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
            if bound.maximizing
            else result.dual_bound > bound.primal_bks + 1e-6 * (abs(bound.primal_bks) + 1.0)
        )
        if crossed:
            result.notes.append(
                f"SCIP dual {result.dual_bound:.6g} crosses the published primal "
                f"{bound.primal_bks:.6g}"
            )


def solver_version() -> str:
    """`SCIP <v> / PySCIPOpt <v>` — recorded per row for provenance."""
    from pyscipopt import Model  # optional `benchmarks` extra; see solve_instance

    try:
        binding = pkg_version("pyscipopt")
    except PackageNotFoundError:  # pragma: no cover — source checkouts only
        binding = "unknown"
    return f"SCIP {Model().version()} / PySCIPOpt {binding}"


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

    Infinity is kept as `inf`: an unbounded SCIP gap is a real, distinct outcome
    from "not computed", and collapsing the two would hide it.
    """
    return "NaN" if math.isnan(value) else repr(value)


def _published_objective(result: ScipResult) -> float:
    """The objective only if we stand behind it, else NaN.

    Same rule the C++ runner applies: a row that failed its feasibility check
    must not publish the number it was rejected for.
    """
    return result.objective if result.feasible else math.nan


def write_scip_csv(path: Path, rows: Sequence[tuple[Bound, ScipResult]], version: str) -> None:
    """SCIP-only results. Columns mirror `comparison.csv` where they mean the
    same thing, then add the bound/gap/status only a complete solver produces.
    """
    with path.open("w", newline="") as fh:
        out = csv.writer(fh)
        out.writerow(SCIP_CSV_HEADER)
        for bound, result in rows:
            gap_bks = (
                safe_gap(result.objective, bound.primal_bks, bound.maximizing)
                if result.feasible
                else math.nan
            )
            gap_dual = (
                safe_gap(result.objective, bound.dual_bound, bound.maximizing)
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
                    result.note.replace(",", ";"),
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
    return [
        bound.instance,
        BKS_METHOD,
        "minlplib-catalogue",
        _cell(bound.primal_bks),
        "true",
        "NaN",  # catalogue values carry no budget comparable to a 60s run
        "0.0",
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
    gap_bks = (
        safe_gap(result.objective, bound.primal_bks, bound.maximizing)
        if result.feasible
        else math.nan
    )
    return [
        bound.instance,
        SCIP_METHOD,
        version,
        _cell(_published_objective(result)),
        "true" if result.feasible else "false",
        repr(result.wall_seconds),
        _cell(gap_bks),
        _cell(result.dual_bound),
        result.note.replace(",", ";"),
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

    Restores only the fields the merge consumes; the SCIP-specific columns
    (`scip_gap%`, `n_int_vars`, the two timing splits) stay in that file.
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
            if row["note"]:
                result.notes.append(row["note"])
            rows[result.instance] = result
    return rows


def _print_row(bound: Bound, result: ScipResult) -> None:
    obj = f"{result.objective:12.4g}" if result.feasible else f"{'INFEAS':>12}"
    bks = f"{bound.primal_bks:12.4g}" if not math.isnan(bound.primal_bks) else f"{'?':>12}"
    gap = (
        safe_gap(result.objective, bound.primal_bks, bound.maximizing)
        if result.feasible
        else math.nan
    )
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
    print(f"read errors:          {sum(1 for _, r in rows if r.status == 'read-error')}")
    print(f"check failures:       {sum(1 for n in notes if n.startswith('CHECK-FAILED'))}")


def _select(bounds: Sequence[Bound], wanted: Sequence[str]) -> list[Bound]:
    if not wanted:
        return list(bounds)
    by_name = {b.instance: b for b in bounds}
    missing = [name for name in wanted if name not in by_name]
    if missing:
        raise SystemExit(f"not in bounds.csv: {', '.join(missing)}")
    return [by_name[name] for name in wanted]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT)
    parser.add_argument(
        "--seed", type=int, default=0, help="SCIP randomization/randomseedshift (default 0)"
    )
    parser.add_argument("--instances", nargs="*", default=[], help="subset; default whole roster")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    inst_dir: Path = args.inst_dir
    # A subset run must not overwrite the published roster-wide results: both
    # CSVs are rewritten whole, so a three-instance debug run would silently
    # replace fifty rows with three. Make the caller name a scratch path.
    if args.instances and (args.out is None or args.merged_out is None):
        print(
            "--instances is a subset run; pass explicit --out and --merged-out so it "
            "cannot overwrite the published roster-wide CSVs.",
            file=sys.stderr,
        )
        return 2
    out_csv: Path = args.out or inst_dir / "scip_baseline.csv"
    merged_csv: Path = args.merged_out or inst_dir / "comparison_all.csv"

    bounds_path = inst_dir / "bounds.csv"
    if not bounds_path.exists():
        print(f"{bounds_path} not found. Fetch the roster first:", file=sys.stderr)
        print(f"  python {inst_dir}/download.py", file=sys.stderr)
        return 2
    bounds = load_bounds(bounds_path)
    cbls_rows = load_cbls_rows(inst_dir / "comparison.csv")

    if args.merge_only:
        scip_rows = read_scip_csv(out_csv)
        if not scip_rows:
            print(f"{out_csv} not found or empty; nothing to merge.", file=sys.stderr)
            return 2
        version = next(iter(read_versions(out_csv)), "unknown")
        write_merged_csv(merged_csv, bounds, cbls_rows, scip_rows, version)
        print(f"wrote {merged_csv} ({len(bounds)} instances)")
        return 0

    version = solver_version()
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
    return 1 if any(r.status == "read-error" for _, r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
