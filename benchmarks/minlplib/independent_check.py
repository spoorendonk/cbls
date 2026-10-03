"""Check a MINLPLib row's assignment in SCIP, independently of this engine's DAG (#205).

The runner's own re-check (`worst_residual` in `minlplib.cpp`) reads the very
node values the search optimised. It catches a search that misreports its own
bookkeeping, and nothing else: an evaluation bug the DAG and its re-check share
agrees with itself by construction. #205 was one -- `sqrt` of a negative read
as 0.0, `pow(neg, 0.5)` as +inf -- and rows built on it were marked `feasible`.

This module is the check that does not share the DAG. The runner writes each
verified row's assignment (`--solution-dir`, one `<instance>.sol` per instance:
the NL column values in column order and the true-sense objective, both at full
precision), and here SCIP reads the instance's `.nl` with its own AMPL reader,
loads the values and decides:

* **feasibility** -- `Model.checkSol` on the ORIGINAL problem, bounds,
  integrality and every constraint, with SCIP's own expression evaluation. A
  function evaluated outside its domain is infeasible to SCIP, as it is to the
  MINLPLib solution checker;
* **the objective** -- the value the row publishes must be the objective SCIP
  computes at that assignment, within `objective_tolerance`.

A row that fails either is demoted in place: `feasible=false`, the objective
and both gaps blanked to `NaN` (the runner's rule for a row it does not stand
behind), and the note replaced by `VERIFY-FAILED(independent: ...)` -- inside
the existing `VERIFY-FAILED` vocabulary, so every consumer already buckets it as
a verification failure.

`run_benchmark.py` applies this to every staged row before it publishes, and
`run_ablation.py` to every run before it records it. The runner refuses to
write the published table itself, so no row reaches it unchecked.

SCIP's objective, for the record: its `.nl` reader keeps a linear objective as
variable coefficients and moves a nonlinear part g(x) into a constraint
`objcons: g(x) - nlobjvar <= 0` (`>= 0` when maximising) over an auxiliary free
column `nlobjvar`. The check sets `nlobjvar` from the published objective minus
SCIP's own linear part and asks `checkSol` twice -- once just above that value,
where `objcons` must hold, and once just below, where it must not -- which
brackets g(x) without reading it from anywhere but SCIP.

Standalone use, over a runner CSV:

    .venv/bin/python3 -m benchmarks.minlplib.independent_check OUT.csv \\
        --inst-dir benchmarks/instances/minlplib --solution-dir SOLDIR
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import io
import math
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

#: First line of every solution file the runner writes (`write_solution` in
#: `minlplib.cpp`); a test pins the two together.
SOLUTION_MAGIC = "cbls-minlplib-solution 1"

#: The note prefix a demoted row carries. Starts with the runner's own
#: `VERIFY-FAILED`, which `runner.COMPLETED_SEARCH_NOTES` already allowlists.
DEMOTED_NOTE_PREFIX = "VERIFY-FAILED(independent: "

#: The columns a demoted row blanks: what it would otherwise publish about a
#: solution nobody stands behind (`run_instance` in `minlplib.cpp`, same rule).
BLANKED_COLUMNS = ("objective", "gap_to_bks%", "gap_to_dual%")

#: SCIP's name for the auxiliary objective column of its `.nl` reader.
NLOBJVAR = "nlobjvar"

_COLUMN_NAME = re.compile(r"^[a-z](\d+)$")

#: An `nlobjvar` value no finite objective in the roster comes near (the largest
#: published bound is ~7e10), so `objcons` holds at it unless g(x) is undefined.
#: Far below SCIP's infinity (1e20), which `setSolVal` would treat specially.
_FAR = 1e15


@dataclass(frozen=True)
class Solution:
    """One `<instance>.sol` file: the assignment and the objective it claims."""

    instance: str
    objective: float
    values: tuple[float, ...]


def objective_tolerance(objective: float) -> float:
    """How far the published objective may sit from SCIP's: the runner's own drift bound."""
    return 1e-6 * (abs(objective) + 1.0)


def read_solution(path: Path) -> Solution:
    """Parse a runner solution file, refusing anything but the exact format."""
    lines = path.read_text().splitlines()
    if len(lines) < 4 or lines[0] != SOLUTION_MAGIC:
        raise ValueError(f"{path}: not a {SOLUTION_MAGIC!r} file")
    fields: dict[str, str] = {}
    for line in lines[1:4]:
        key, _, value = line.partition(" ")
        fields[key] = value
    if set(fields) != {"instance", "objective", "columns"}:
        raise ValueError(f"{path}: header is {sorted(fields)}")
    n = int(fields["columns"])
    body = lines[4:]
    if len(body) != n:
        raise ValueError(f"{path}: {len(body)} values for {n} columns")
    return Solution(
        instance=fields["instance"],
        objective=float(fields["objective"]),
        values=tuple(float(v) for v in body),
    )


@contextlib.contextmanager
def _captured_stdout() -> Iterator[io.StringIO]:
    """Capture file descriptor 1, which SCIP's C message handler writes to."""
    buffer = io.StringIO()
    libc = ctypes.CDLL(None)
    sys.stdout.flush()
    libc.fflush(None)
    saved = os.dup(1)
    with tempfile.TemporaryFile(mode="w+b") as sink:
        os.dup2(sink.fileno(), 1)
        try:
            yield buffer
        finally:
            libc.fflush(None)
            os.dup2(saved, 1)
            os.close(saved)
            sink.seek(0)
            buffer.write(sink.read().decode(errors="replace"))


def _first_reason(text: str) -> str:
    """The first violated constraint SCIP names, short and comma-free for a CSV cell."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith("violation:"):
            head = lines[i - 1] if i > 0 else ""
            name = re.search(r"<([^>]+)>", head)
            what = f"{name.group(1)}: " if name else ""
            reason = what + line.removeprefix("violation:").strip()
            break
    else:
        reason = lines[0] if lines else "rejected"
    return reason.replace(",", ";")[:160]


def _check(model: Any, sol: Any) -> tuple[bool, str]:
    model.hideOutput(False)
    try:
        with _captured_stdout() as out:
            ok = model.checkSol(sol, printreason=True, completely=True, original=True)
    finally:
        model.hideOutput(True)
    return bool(ok), _first_reason(out.getvalue())


def _columns(model: Any, n: int) -> tuple[list[Any], Any | None]:
    """SCIP's variables in NL column order, and its `nlobjvar` if it made one."""
    nlobj = None
    columns: list[Any] = []
    for var in model.getVars():
        if var.name == NLOBJVAR:
            nlobj = var
        else:
            columns.append(var)
    if len(columns) != n:
        raise ValueError(f"SCIP sees {len(columns)} columns; the solution has {n}")
    # Without a .col file the reader names column j `x<j>`/`i<j>`/`b<j>`; check
    # the order rather than assume it.
    for j, var in enumerate(columns):
        match = _COLUMN_NAME.match(var.name)
        if match is not None and int(match.group(1)) != j:
            raise ValueError(f"SCIP column {j} is named {var.name}")
    return columns, nlobj


class _Probe:
    """One instance read into SCIP, and the assignment under test loaded into solutions."""

    def __init__(self, nl_path: Path, solution: Solution) -> None:
        from pyscipopt import Model  # the optional `benchmarks` extra, as in reference_solve.py

        self.model: Any = Model()
        self.model.hideOutput()
        self.model.readProblem(str(nl_path))
        self.columns, self.nlobj = _columns(self.model, len(solution.values))
        self.values = solution.values
        self.maximizing = self.model.getObjectiveSense() == "maximize"

    def sol(self, nlobj_value: float = 0.0) -> Any:
        """The assignment, with SCIP's `nlobjvar` (when it made one) at `nlobj_value`."""
        sol = self.model.createSol()
        for var, value in zip(self.columns, self.values, strict=True):
            self.model.setSolVal(sol, var, value)
        if self.nlobj is not None:
            self.model.setSolVal(sol, self.nlobj, nlobj_value)
        return sol

    def objective(self, sol: Any) -> float:
        return float(self.model.getSolObjVal(sol, original=True))


def _check_linear_objective(probe: _Probe, claimed: float) -> str | None:
    """No `nlobjvar`: SCIP's objective at the assignment is its own value directly."""
    sol = probe.sol()
    ok, reason = _check(probe.model, sol)
    if not ok:
        return f"SCIP rejects the assignment ({reason})"
    scip_obj = probe.objective(sol)
    if not abs(scip_obj - claimed) <= objective_tolerance(claimed):
        return f"objective {claimed:.10g} but SCIP computes {scip_obj:.10g}"
    return None


def _check_nonlinear_objective(probe: _Probe, claimed: float) -> str | None:
    """The objective is SCIP's linear part plus `nlobjvar`, which `objcons` ties to g(x).

    First the assignment alone: `nlobjvar` far on objcons' slack side, so that
    the row holds for every finite g(x) and fails only where g is undefined.
    Then bracket g(x). Minimising, objcons is g - v <= 0: it must hold just
    above the target (else the claim is better than the assignment achieves)
    and fail just below (else it is worse). Maximising flips both.
    """
    ok, reason = _check(probe.model, probe.sol(-_FAR if probe.maximizing else _FAR))
    if not ok:
        return f"SCIP rejects the assignment ({reason})"
    target = claimed - probe.objective(probe.sol(0.0))  # what g(x) must be
    tol = objective_tolerance(claimed)
    feastol = float(probe.model.getParam("numerics/feastol"))
    loose, tight = (
        (target - tol, target + tol + 2.0 * feastol)
        if probe.maximizing
        else (target + tol, target - tol - 2.0 * feastol)
    )
    if not _check(probe.model, probe.sol(loose))[0]:
        return f"objective {claimed:.10g} is better than SCIP's value at the assignment"
    if _check(probe.model, probe.sol(tight))[0]:
        return f"objective {claimed:.10g} is worse than SCIP's value at the assignment"
    return None


def check_solution(nl_path: Path, solution: Solution) -> str | None:
    """None when SCIP accepts the assignment and its objective; otherwise why not."""
    if not math.isfinite(solution.objective):
        return "non-finite objective"
    probe = _Probe(nl_path, solution)
    if probe.nlobj is None:
        return _check_linear_objective(probe, solution.objective)
    return _check_nonlinear_objective(probe, solution.objective)


def demote(row: dict[str, str], reason: str) -> dict[str, str]:
    """The row as it must be published once the independent check has rejected it."""
    out = dict(row)
    out["feasible"] = "false"
    for column in BLANKED_COLUMNS:
        out[column] = "NaN"
    out["note"] = f"{DEMOTED_NOTE_PREFIX}{reason}; was {row.get('note', '')})".replace(",", ";")
    return out


def check_rows(csv_path: Path, inst_dir: Path, solution_dir: Path) -> list[str]:
    """Check every `feasible=true` row of a runner CSV; demote the ones SCIP rejects.

    Rewrites `csv_path` atomically when anything is demoted and returns the
    demoted instances. Idempotent: a demoted row is `feasible=false` and is not
    looked at again. A verified row with no solution file is an error, not a
    pass -- nothing would have been checked.
    """
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = list(reader)
    demoted: list[str] = []
    for i, row in enumerate(rows):
        if row.get("feasible") != "true":
            continue
        name = row["instance"]
        sol_path = solution_dir / f"{name}.sol"
        if not sol_path.exists():
            raise FileNotFoundError(
                f"{csv_path}: {name} is marked feasible but has no {sol_path}; "
                "the runner was not given --solution-dir, so the row cannot be checked"
            )
        solution = read_solution(sol_path)
        if solution.instance != name:
            raise ValueError(f"{sol_path} is for {solution.instance}, not {name}")
        reason = check_solution(inst_dir / f"{name}.nl", solution)
        if reason is not None:
            rows[i] = demote(row, reason)
            demoted.append(name)
    if demoted:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        atomic_write(csv_path, buffer.getvalue())
    return demoted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("csv", type=Path, help="a cbls_minlplib --out file")
    parser.add_argument("--inst-dir", type=Path, required=True)
    parser.add_argument("--solution-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    demoted = check_rows(args.csv, args.inst_dir, args.solution_dir)
    for name in demoted:
        print(f"demoted {name}: the independent SCIP check rejected it")
    return 1 if demoted else 0


if __name__ == "__main__":
    sys.exit(main())
