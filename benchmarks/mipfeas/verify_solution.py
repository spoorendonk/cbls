"""Independent feasibility check of a MIPfeas solution against its instance file.

MIPfeas is this repo's correctness benchmark (CLAUDE.md, "Benchmark priority"),
so the one thing it must not do is take a reported solution on trust. Every check
either engine runs on itself is downstream of its own MPS reader: CBLS's
residual, integrality and objective-drift checks all run against the model
`src/io/mps_to_model.cpp` built, so a misread bound, a dropped row or a flipped
sign is invisible to all of them, and the reference `.solu` values catch only an
objective that is *too good* (issue #138).

This module closes that half. It re-reads the ORIGINAL `.mps.gz` with a
third-party reader and checks the written solution vector against it.

**Why PySCIPOpt and not OR-Tools.** Both are already benchmark dependencies and
both read MPS. But `benchmarks/mipfeas/cpsat_solve.py` builds the CP-SAT model
with OR-Tools' own reader, so verifying a CP-SAT solution with OR-Tools would let
a shared reader defect cancel out on exactly the half this check exists to cover.
SCIP's reader is independent of *both* engines, so one verifier covers both. No
CBLS code is involved at any point: not `src/io/mps_reader.cpp`, not
`mps_to_model`.

Tolerances
----------

All of them are constants of this module, stated here rather than decided at
scoring time, and all are `absolute + relative * scale` because MIPLIB
coefficient ranges span many orders of magnitude and a purely absolute rule is
either vacuous on a large row or unmeetable on a small one:

==================  ===================================  ==============================
check               violation                            tolerance
==================  ===================================  ==============================
row activity        ``max(0, lower - a.x, a.x - upper)``  ``1e-6 + 1e-9 * row_scale``
variable bound      ``max(0, lower - x, x - upper)``      ``1e-6 + 1e-9 * |bound|``
integrality         ``|x - round(x)|``                    ``1e-6``
objective           ``|reported - (c.x + offset)|``       ``1e-6 + 1e-9 * |objective|``
==================  ===================================  ==============================

``row_scale`` is ``max(|lower|, |upper|, sum |a_ij * x_j|)`` over the finite
sides — the sum of absolute terms rather than the activity itself, so that a row
whose terms cancel is still judged against the magnitudes that were actually
added up.

The `1e-6` absolute terms are the engine's own stated feasibility tolerance
(`--feas-tol`, `cbls::kDefaultFeasibilityTolerance`), so the two checks are
*comparable* — a point the engine accepts at its limit is not rejected here for
being at that limit — while remaining *independent*, since nothing else is
shared. The objective's relative term is likewise the runner's own drift gate
(`1e-6 * (|objective| + 1)`); the row and bound ones are `1e-9`, about seven
orders of magnitude above double-precision round-off (`2.2e-16`) — this module
sums each row with `math.fsum`, which is exact, so that margin covers only the
*engine's* accumulation over a row of millions of nonzeros.

Integrality is judged at the point as written and rows are evaluated on those
same unrounded values: a point cannot buy row feasibility by being rounded to
integers after the fact.

The borderline rule
-------------------

There is no judgement call at scoring time. A solution **fails** if and only if
some violation exceeds its tolerance above; anything at or below it **passes**.
To keep "just inside" visible rather than invisible, a passing solution whose
worst violation reaches `MARGINAL_FRACTION` (10%) of its tolerance is published
as `pass` with `marginal` set — a signal to look, never a reason to withhold.

Verdicts are machine-readable: `pass`, `fail` (the solution is wrong) or `error`
(the check could not be performed — a missing solution file, an unreadable
instance, a constraint type this checker does not model). Only `pass` lets the
scorer publish the row's objective; see `benchmarks/mipfeas/primal_integral.py`.

Usage:
    python verify_solution.py --instance pk1 --result-dir results/mipfeas/cbls
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pyscipopt import Model

#: Absolute and relative slack on a row activity. See the module docstring.
ROW_ABS_TOLERANCE = 1e-6
ROW_REL_TOLERANCE = 1e-9

#: Absolute and relative slack on a variable bound.
BOUND_ABS_TOLERANCE = 1e-6
BOUND_REL_TOLERANCE = 1e-9

#: Absolute slack on |x - round(x)| for an integer column. No relative term: a
#: double holds every integer below 2^53 exactly, so distance-to-integer does not
#: scale with the value.
INTEGRALITY_TOLERANCE = 1e-6

#: Absolute and relative slack between the objective the engine published and the
#: one this module computes from the solution vector and the instance file.
#:
#: The relative term is 1e-6, NOT the 1e-9 the row and bound checks use, and for
#: the same reason those use 1e-6 absolutely: this check re-measures exactly the
#: quantity the CBLS runner's own objective-drift gate accepts, at
#: `1e-6 * (|objective| + 1)` (benchmarks/mipfeas/mipfeas.cpp, assess_result).
#: A tighter rule downstream of that gate would not be more independent -- it
#: would reject runs the engine was entitled to publish, and at |obj| = 1e6 a
#: 1e-9 relative term is a thousand times tighter than the gate upstream of it.
OBJECTIVE_ABS_TOLERANCE = 1e-6
OBJECTIVE_REL_TOLERANCE = 1e-6

#: A pass whose worst violation reaches this fraction of its tolerance is marked
#: `marginal`. It is a flag on a passing row, never a reason to withhold one.
MARGINAL_FRACTION = 0.1

#: Verdicts. Only PASS lets a row publish its objective.
PASS = "pass"
FAIL = "fail"
ERROR = "error"

#: SCIP's infinity sentinel; a bound at or beyond it is not a bound.
SCIP_INFINITY = 1e20

#: The only SCIP constraint handler this module knows how to check. Everything
#: else (SOS, indicator, nonlinear) becomes an `error` verdict rather than being
#: skipped: a constraint nobody checked must not read as a constraint that held.
LINEAR_HANDLER = "linear"


@dataclass(frozen=True)
class Column:
    lower: float
    upper: float
    integral: bool


@dataclass(frozen=True)
class Row:
    name: str
    lower: float
    upper: float
    coefficients: dict[str, float]


@dataclass(frozen=True)
class Instance:
    """The original program, as a third-party reader sees it.

    `rows` is a factory rather than a list, and that is load-bearing rather than
    stylistic. PySCIPOpt's `getValsLinear` builds a fresh `str` key per nonzero,
    measured at ~245 bytes each (`supportcase7`: 2.85M nonzeros, 698 MiB RSS), so
    holding the whole matrix would cost ~6.7 GB on `square47`'s 27.4M nonzeros --
    on top of SCIP's own copy, inside the job's `--mem-limit-gb` slot. Streaming
    one row at a time keeps the peak at SCIP's model plus a single row. A factory
    rather than a bare generator so the caller may iterate more than once.
    """

    columns: dict[str, Column]
    rows: Callable[[], Iterator[Row]]
    #: Constraints SCIP holds, linear or not. Informational; `rows()` yields only
    #: the linear ones and names the rest.
    n_rows: int
    objective: dict[str, float]
    objective_offset: float


@dataclass
class Verification:
    """One solution's verdict, in the shape the scorer and its consumers read.

    `verdict` and `reason` are the machine-readable pair: `reason` is a stable
    code (`row_violation`, `missing_solution_file`, ...), `message` is the human
    sentence, and `failed_checks` lists every check that failed rather than only
    the one that named the reason.
    """

    instance: str = ""
    engine: str = ""
    verdict: str = ERROR
    reason: str = "not_run"
    message: str = ""
    marginal: bool = False
    failed_checks: list[str] = field(default_factory=list)
    n_columns: int = 0
    n_rows: int = 0
    max_row_violation: float = 0.0
    max_row_ratio: float = 0.0
    worst_row: str = ""
    max_bound_violation: float = 0.0
    max_bound_ratio: float = 0.0
    worst_bound_column: str = ""
    max_integrality_violation: float = 0.0
    max_integrality_ratio: float = 0.0
    worst_integral_column: str = ""
    objective_reported: float | None = None
    objective_recomputed: float | None = None
    objective_violation: float = 0.0
    objective_ratio: float = 0.0
    checker: str = ""

    def to_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = dict(vars(self))
        record["tolerances"] = {
            "row_absolute": ROW_ABS_TOLERANCE,
            "row_relative": ROW_REL_TOLERANCE,
            "bound_absolute": BOUND_ABS_TOLERANCE,
            "bound_relative": BOUND_REL_TOLERANCE,
            "integrality_absolute": INTEGRALITY_TOLERANCE,
            "objective_absolute": OBJECTIVE_ABS_TOLERANCE,
            "objective_relative": OBJECTIVE_REL_TOLERANCE,
            "marginal_fraction": MARGINAL_FRACTION,
        }
        return record


def _error(reason: str, message: str, **fields: Any) -> Verification:
    return Verification(verdict=ERROR, reason=reason, message=message, **fields)


# ---------------------------------------------------------------------------
# Reading the original instance (PySCIPOpt only; no CBLS code).


def checker_provenance() -> str:
    """Which reader produced the verdict, recorded per result like every other tool."""
    from pyscipopt import Model

    scip = Model()
    # Silenced: this runs inside the driver, which slices the verifier's stdout
    # into its own log, and a chattier SCIP build prints a banner on construction.
    scip.hideOutput()
    try:
        binding = pkg_version("pyscipopt")
    except PackageNotFoundError:  # pragma: no cover - source checkouts only
        binding = "unknown"
    scip = Model()
    core = f"{scip.getMajorVersion()}.{scip.getMinorVersion()}.{scip.getTechVersion()}"
    return f"SCIP {core} / PySCIPOpt {binding}"


def _read_columns(model: Model) -> dict[str, Column]:
    columns: dict[str, Column] = {}
    for var in model.getVars():
        columns[var.name] = Column(
            lower=var.getLbOriginal(),
            upper=var.getUbOriginal(),
            integral=var.vtype() != "CONTINUOUS",
        )
    return columns


def _row_stream(model: Model, unsupported: list[str]) -> Iterator[Row]:
    """Yield one linear row at a time, appending anything else to `unsupported`.

    The caller must consume the whole stream before reading `unsupported`: a
    constraint type nobody checked must not read as a constraint that held, and
    that is only knowable once every constraint has been seen.
    """
    unsupported.clear()
    for cons in model.getConss():
        handler = cons.getConshdlrName()
        if handler != LINEAR_HANDLER:
            unsupported.append(f"{cons.name}({handler})")
            continue
        yield Row(
            name=cons.name,
            lower=model.getLhs(cons),
            upper=model.getRhs(cons),
            coefficients=model.getValsLinear(cons),
        )


def read_instance(mps_path: Path) -> tuple[Instance, list[str]]:
    """Parse `mps_path` with SCIP. Raises whatever SCIP raises on a bad file.

    Returns the instance and the list constraint handlers this module cannot
    check accumulate into; the list is empty until `Instance.rows()` has been
    consumed, and `check` reads it afterwards.

    The objective sense is deliberately not carried: the identity checked here is
    `c.x + offset`, the same number under either sense, and SCIP returns
    original-problem coefficients. No roster instance carries an OBJSENSE section
    (surveyed, 233/233) and the CBLS adapter rejects one outright.
    """
    from pyscipopt import Model

    model = Model()
    model.hideOutput()
    model.readProblem(str(mps_path))
    unsupported: list[str] = []
    return (
        Instance(
            columns=_read_columns(model),
            rows=lambda: _row_stream(model, unsupported),
            n_rows=model.getNConss(),
            objective={var.name: var.getObj() for var in model.getVars() if var.getObj() != 0.0},
            # `getObjoffset()` only: the transformed-problem variant segfaults in
            # the problem-creation stage on PySCIPOpt 6.2.1, which is where a
            # model that was read but never solved sits.
            objective_offset=model.getObjoffset(),
        ),
        unsupported,
    )


# ---------------------------------------------------------------------------
# The checks. Pure arithmetic over an Instance and a value per column.


def _finite(value: float) -> bool:
    return math.isfinite(value) and abs(value) < SCIP_INFINITY


def bound_violation(value: float, lower: float, upper: float) -> tuple[float, float]:
    """`(violation, tolerance)` of `value` against a two-sided bound."""
    violation = 0.0
    scale = 0.0
    if _finite(lower) and value < lower:
        violation = lower - value
        scale = abs(lower)
    if _finite(upper) and value - upper > violation:
        violation = value - upper
        scale = abs(upper)
    return violation, BOUND_ABS_TOLERANCE + BOUND_REL_TOLERANCE * scale


def _check_columns(instance: Instance, values: dict[str, float], out: Verification) -> None:
    """Variable bounds and integrality, against the file's own declarations."""
    for name, column in instance.columns.items():
        value = values[name]
        violation, tolerance = bound_violation(value, column.lower, column.upper)
        ratio = violation / tolerance
        if ratio > out.max_bound_ratio:
            out.max_bound_ratio = ratio
            out.max_bound_violation = violation
            out.worst_bound_column = name
        if not column.integral:
            continue
        distance = abs(value - round(value))
        if distance > out.max_integrality_violation:
            out.max_integrality_violation = distance
            out.max_integrality_ratio = distance / INTEGRALITY_TOLERANCE
            out.worst_integral_column = name


def _check_rows(instance: Instance, values: dict[str, float], out: Verification) -> None:
    """Row activities against the ranges the file declares for them.

    One row at a time, and each activity summed with `math.fsum`, which is exact:
    the tolerance below then has to cover only the *engine's* accumulation error,
    not this checker's as well.
    """
    for row in instance.rows():
        terms = [coefficient * values[name] for name, coefficient in row.coefficients.items()]
        activity = math.fsum(terms)
        absolute_terms = math.fsum(abs(term) for term in terms)
        violation, _ = bound_violation(activity, row.lower, row.upper)
        scale = absolute_terms
        for side in (row.lower, row.upper):
            if _finite(side):
                scale = max(scale, abs(side))
        tolerance = ROW_ABS_TOLERANCE + ROW_REL_TOLERANCE * scale
        ratio = violation / tolerance
        if ratio > out.max_row_ratio:
            out.max_row_ratio = ratio
            out.max_row_violation = violation
            out.worst_row = row.name


def _check_objective(
    instance: Instance, values: dict[str, float], reported: float | None, out: Verification
) -> None:
    """Whether the published number is the objective of the published point.

    Recomputed from the instance file's own cost row, so an adapter that dropped
    a cost coefficient or lost the objective constant is caught here rather than
    read as search quality.
    """
    recomputed = math.fsum(
        [instance.objective_offset]
        + [coefficient * values[name] for name, coefficient in instance.objective.items()]
    )
    out.objective_recomputed = recomputed
    out.objective_reported = reported
    if reported is None:
        # Unreachable from either runner -- both always write `=obj=` -- but a row
        # whose objective nobody stated cannot have it checked, and the scorer
        # would then publish an unchecked number.
        out.objective_violation = math.inf
        out.objective_ratio = math.inf
        return
    if not (math.isfinite(reported) and math.isfinite(recomputed)):
        # An objective of inf or NaN is not a number a row can publish, and every
        # comparison below would be False — so the check would pass on it.
        out.objective_violation = math.inf
        out.objective_ratio = math.inf
        return
    out.objective_violation = abs(reported - recomputed)
    tolerance = OBJECTIVE_ABS_TOLERANCE + OBJECTIVE_REL_TOLERANCE * max(
        abs(reported), abs(recomputed)
    )
    out.objective_ratio = out.objective_violation / tolerance


#: Check name -> the field on Verification carrying its worst violation/tolerance
#: ratio. Ordered: the first failing check names the verdict's `reason`.
_RATIO_CHECKS = (
    ("row_violation", "max_row_ratio"),
    ("bound_violation", "max_bound_ratio"),
    ("integrality_violation", "max_integrality_ratio"),
    ("objective_mismatch", "objective_ratio"),
)


def check(
    instance: Instance,
    values: dict[str, float],
    reported_objective: float | None,
    unsupported: list[str] | None = None,
) -> Verification:
    """Check `values` against `instance`; the verdict is the whole story of it.

    `unsupported` is the list `read_instance` returned alongside the instance:
    iterating the rows fills it with any constraint type this module cannot
    model, and a non-empty one can only be an `error`.
    """
    unsupported = [] if unsupported is None else unsupported
    out = Verification(
        verdict=PASS,
        reason="",
        n_columns=len(instance.columns),
        n_rows=instance.n_rows,
    )
    # Before anything else: every comparison against a NaN is False, so a NaN
    # value would sail through every check below and verify as `pass`. This is
    # the one module where failing open is the cardinal sin, so a value that is
    # not a real number makes the file unusable rather than acceptable.
    unreal = sorted(name for name, value in values.items() if not math.isfinite(value))
    if unreal:
        return _error(
            "non_finite_solution",
            f"{len(unreal)} values are not finite (e.g. {unreal[:3]}); the file does "
            f"not describe a point",
            n_columns=len(instance.columns),
            n_rows=instance.n_rows,
        )
    missing = sorted(set(instance.columns) - set(values))
    extra = sorted(set(values) - set(instance.columns))
    if missing or extra:
        # Not a failed solution but an unusable one: a name set that does not
        # match the instance means the two files are not about the same program.
        return _error(
            "solution_variable_mismatch",
            f"solution covers {len(values)} of {len(instance.columns)} columns "
            f"(missing e.g. {missing[:3]}, unknown e.g. {extra[:3]})",
            n_columns=len(instance.columns),
            n_rows=instance.n_rows,
        )
    _check_columns(instance, values, out)
    _check_rows(instance, values, out)
    # Read only after the row stream has been consumed, which is what fills it.
    if unsupported:
        return _error(
            "unsupported_constraint",
            f"{len(unsupported)} constraints this checker does not model "
            f"(e.g. {unsupported[:3]}); a constraint nobody checked must not read "
            f"as one that held",
            n_columns=len(instance.columns),
            n_rows=instance.n_rows,
        )
    _check_objective(instance, values, reported_objective, out)

    out.failed_checks = [name for name, attr in _RATIO_CHECKS if getattr(out, attr) > 1.0]
    if out.failed_checks:
        out.verdict = FAIL
        out.reason = out.failed_checks[0]
        out.message = describe(out)
        return out
    out.marginal = max(getattr(out, attr) for _, attr in _RATIO_CHECKS) >= MARGINAL_FRACTION
    out.message = describe(out)
    return out


def describe(out: Verification) -> str:
    """One line naming every violation as a fraction of the tolerance it is judged by."""
    return (
        f"row {out.max_row_violation:.3g} ({out.max_row_ratio:.3g}x tol"
        f"{f' at {out.worst_row}' if out.worst_row else ''}), "
        f"bound {out.max_bound_violation:.3g} ({out.max_bound_ratio:.3g}x tol"
        f"{f' at {out.worst_bound_column}' if out.worst_bound_column else ''}), "
        f"integrality {out.max_integrality_violation:.3g}"
        f"{f' at {out.worst_integral_column}' if out.worst_integral_column else ''}, "
        f"objective {out.objective_violation:.3g} ({out.objective_ratio:.3g}x tol)"
    )


# ---------------------------------------------------------------------------
# The solution file: `=obj= value` plus one `name value` line per column.


def parse_solution(text: str) -> tuple[dict[str, float], float | None]:
    """Parse the MIPLIB-style solution format both runners write.

    Whole-line `#` comments and blank lines are ignored; `=obj= <value>` carries
    the objective the engine published, and every other line is `<name> <value>`.
    """
    values: dict[str, float] = {}
    objective: float | None = None
    for lineno, raw in enumerate(text.splitlines(), start=1):
        # A comment is a WHOLE line. MIPLIB column names contain `#` -- 13 of the
        # 233 roster instances name columns `x#1#1`, `delay#1`, `P#0#0` -- so
        # stripping from the first `#` anywhere on the line would turn every
        # solution on those instances into a parse error and withhold the row for
        # both engines, for a format bug.
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"line {lineno}: expected '<name> <value>', got {raw!r}")
        name, value = fields
        if name == "=obj=":
            objective = float(value)
            continue
        if name in values:
            raise ValueError(f"line {lineno}: {name} appears twice")
        values[name] = float(value)
    return values, objective


def write_solution(
    path: Path, instance: str, engine: str, objective: float, values: dict[str, float]
) -> None:
    """Write a solution file. Used by the tests; the runners write their own."""
    lines = [f"# instance {instance}", f"# engine {engine}", f"=obj= {objective!r}"]
    lines += [f"{name} {value!r}" for name, value in values.items()]
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Driver entry point: verify one instance's result directory.


def verify_result(instance: str, inst_dir: Path, result_dir: Path) -> Verification:
    """Verify `<result_dir>/<instance>.sol` against `<inst_dir>/<instance>.mps.gz`.

    Never raises: every failure becomes a verdict, because the driver runs this
    unattended over hundreds of instances and a crash there would leave a row with
    no verdict at all — which the scorer then cannot tell from a row nobody
    checked.
    """
    mps_path = inst_dir / f"{instance}.mps.gz"
    solution_path = result_dir / f"{instance}.sol"
    result_path = result_dir / f"{instance}.json"
    engine = ""
    reported: float | None = None
    if result_path.exists():
        try:
            record = json.loads(result_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return _error("unreadable_result", f"{result_path}: {exc}", instance=instance)
        engine = str(record.get("engine", ""))
        raw = record.get("objective")
        reported = float(raw) if isinstance(raw, (int, float)) else None
    if not mps_path.exists():
        return _error("missing_instance", f"{mps_path} not found", instance=instance, engine=engine)
    if not solution_path.exists():
        # A feasible row whose solution was never written cannot be checked, and
        # an unchecked row is exactly what issue #138 exists to stop publishing.
        return _error(
            "missing_solution_file",
            f"{solution_path} not found; re-run this job with --solution-dir",
            instance=instance,
            engine=engine,
        )
    try:
        values, file_objective = parse_solution(solution_path.read_text())
    except (ValueError, OSError) as exc:
        return _error(
            "unreadable_solution", f"{solution_path}: {exc}", instance=instance, engine=engine
        )
    if reported is None:
        reported = file_objective
    try:
        parsed, unsupported = read_instance(mps_path)
        out = check(parsed, values, reported, unsupported)
    except Exception as exc:  # any reader failure becomes one verdict, not a crash
        return _error(
            "instance_read_error",
            f"{mps_path}: {type(exc).__name__}: {exc}",
            instance=instance,
            engine=engine,
        )
    out.instance = instance
    out.engine = engine
    out.checker = checker_provenance()
    return out


def write_verification(path: Path, out: Verification) -> None:
    """Write the verdict, via a rename so a killed job leaves no half-verdict."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(out.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--inst-dir", default="benchmarks/instances/mipfeas")
    parser.add_argument(
        "--result-dir",
        required=True,
        help="the per-engine results directory holding <instance>.json and <instance>.sol",
    )
    parser.add_argument("--out", default=None, help="default: <result-dir>/<instance>.verify.json")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    out = verify_result(args.instance, Path(args.inst_dir), result_dir)
    write_verification(
        Path(args.out) if args.out else result_dir / f"{args.instance}.verify.json", out
    )
    print(f"{args.instance:<28} {out.verdict:<6} {out.reason or 'ok':<24} {out.message}")
    # Exit 1 on a rejected solution and 2 when the check could not run: a
    # correctness benchmark must not report either at exit 0.
    return {PASS: 0, FAIL: 1}.get(out.verdict, 2)


if __name__ == "__main__":
    sys.exit(main())
