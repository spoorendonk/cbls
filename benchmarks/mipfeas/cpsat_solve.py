"""MIPfeas baseline: OR-Tools CP-SAT restricted to its violation-based local search.

This is the reference implementation of the algorithm CBLS reimplements — the
`fj` (Feasibility Jump) and `ls` (ViolationLS) workers described in Davies, Didier
& Perron, *ViolationLS: Constraint-Based Local Search in CP-SAT*, CPAIOR 2024
(https://link.springer.com/chapter/10.1007/978-3-031-60597-0_16). Comparing against
them answers "is the reimplementation competent"; comparing against CP-SAT's default
portfolio would be a different (and rejected) question — see epic #87.

Configuration notes, all established empirically against ortools 9.15:

* `filter_subsolvers` is the only parameter that accepts `fj`/`ls`. `subsolvers` and
  `ignore_subsolvers` validate against full-problem subsolver names only and reject
  both, so they cannot express "LS only".
* `ls` alone never bootstraps a first solution (status stays UNKNOWN); it needs `fj`.
  That pairing mirrors CBLS, which runs Feasibility Jump to reach feasibility and then
  ViolationLS with the objective folded in as `obj <= bound`.
* `num_workers: 1` runs both of them — the log reports `1 first solution subsolver: [fj]`
  and `1 interleaved subsolver: [ls]`. One worker is therefore the default here, so the
  baseline gets the same single thread CBLS does. Raising it multiplies *both* workers
  (`num_workers: 2` gives `fj(2)` and `ls(2)`, ~2x the CPU in the same wall time), so
  the count is recorded per result rather than assumed.
* Presolve is left at its default (on), i.e. the worker as it actually ships.
* `ModelSolver.log_callback` raises TypeError on this ortools/Python combination
  (pybind11 std::function caster unregistered), so the log is captured by redirecting
  fd 1 around the solve call. That works precisely because this script runs one
  instance per process.

With --solution-dir, a feasible run also writes its solution vector as
`<instance>.sol`, for `verify_solution.py` to check against the original
instance file with a reader that is not OR-Tools' (issue #138). The objective
this script reports is otherwise taken entirely on trust.

Usage:
    python cpsat_solve.py --instance pk1 --out-dir results/cpsat --budget 600
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import resource
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

from ortools.linear_solver.python import model_builder

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

#: CP-SAT logs each improving solution as e.g.
#:   `#12      3.30s best:6908.97 next:[5726.32,6908.97] ls_restart_compound(...)`
#: Bound-only lines (`#Bound`) and model lines (`#Model`) do not match.
SOLUTION_LINE = re.compile(r"^#(\d+)\s+([0-9.]+)s\s+best:(-?[0-9.eE+-]+)\b")

DEFAULT_WORKERS = 1


def build_parameters(workers: int, seed: int) -> str:
    """CP-SAT parameter string restricting the solve to the fj + ls workers.

    The budget is not set here: `set_time_limit_in_seconds` already populates
    `max_time_in_seconds`, and this string merges on top of it rather than
    replacing it, so stating the limit twice would only invite the two to drift.
    """
    return (
        f"num_workers:{workers},"
        f"num_violation_ls:1,"
        f"filter_subsolvers:'fj',"
        f"filter_subsolvers:'ls',"
        f"random_seed:{seed},"
        f"log_search_progress:true"
    )


def parse_trace(log_text: str) -> list[tuple[float, float]]:
    """Extract (seconds, objective) for each improving solution in a CP-SAT log."""
    return _parse_lines(log_text.splitlines())


def parse_trace_file(path: Path) -> list[tuple[float, float]]:
    """Same, streaming the file rather than loading it.

    A 600s run on an instance that keeps improving writes tens of thousands of log
    lines (mas76 manages 3.3 MB in 5s), and this job runs under an address-space cap
    alongside a multi-GB model.
    """
    with open(path) as fh:
        return _parse_lines(fh)


def _parse_lines(lines: Iterable[str]) -> list[tuple[float, float]]:
    trace: list[tuple[float, float]] = []
    for line in lines:
        match = SOLUTION_LINE.match(line.strip())
        if match is None:
            continue
        trace.append((float(match.group(2)), float(match.group(3))))
    return trace


#: Magnitude at or past which ModelBuilder reports a constraint bound as infinite.
#: Its own sentinel is 1e30; compared with a margin rather than for equality so a
#: bound the reader scaled or rounded still reads as infinite.
MODEL_BUILDER_INFINITY = 1e30


def count_free_constraints(model: model_builder.ModelBuilder) -> int:
    """Linear constraints CP-SAT holds that bound nothing on either side.

    These are the MPS `N` rows after the first: the first is the objective, and
    every reader treats it as such, but ModelBuilder keeps the rest as
    unconstrained linear constraints where the CBLS adapter (and SCIP) drop them.
    That is the whole of the known-benign way the two engines' constraint counts
    differ, so it is recorded rather than reasoned about after the fact -- the
    scorer's model-shape cross-check subtracts exactly this and flags whatever is
    left (issue #139).
    """
    return sum(
        1
        for con in model.get_linear_constraints()
        if con.lower_bound <= -MODEL_BUILDER_INFINITY and con.upper_bound >= MODEL_BUILDER_INFINITY
    )


@contextlib.contextmanager
def capture_stdout_fd(sink_path: Path) -> Iterator[None]:
    """Redirect fd 1 (including writes from C++) to `sink_path` for the block's duration.

    The caller owns `sink_path`. Holding it in a TemporaryDirectory is what makes the
    log disappear on every exit path, including a solve that raises; deleting it only
    after a clean return leaks one log per failed job, and these logs are not small.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        with open(sink_path, "w") as sink:
            os.dup2(sink.fileno(), 1)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def solve(
    mps_path: Path, budget: float, workers: int, seed: int
) -> tuple[dict[str, object], list[tuple[float, float]], dict[str, float]]:
    """Run the LS-only CP-SAT configuration.

    Returns (result record, incumbent trace, solution values). The values are
    keyed by the MPS column name, empty when the run found nothing; they are what
    `verify_solution.py` checks against the instance file.
    """
    # ortools ships no annotations for ModelBuilder's constructor; the rest of the
    # model_builder surface used here is typed.
    model = model_builder.ModelBuilder()  # type: ignore[no-untyped-call]
    # Read and build are one call here -- ModelBuilder parses the MPS straight
    # into its own protobuf -- so this is the whole of the setup CBLS splits into
    # `read_seconds` + `build_seconds`. Timed on both sides because the published
    # wall-clock column has only ever bracketed the solve.
    setup_started = time.monotonic()
    if not model.import_from_mps_file(str(mps_path)):
        # Carries every key main() and the scorer read unconditionally. A key
        # missing here crashes the job *after* write_outputs has written its
        # result, and the driver — seeing a result file — reports that as a clean
        # run. The scorer .get()s the rest.
        return (
            {
                "status": "read_error",
                "message": f"CP-SAT could not import {mps_path.name}",
                "wall_seconds": 0.0,
                "read_seconds": time.monotonic() - setup_started,
                "setup_seconds": time.monotonic() - setup_started,
                "objective": None,
            },
            [],
            {},
        )

    read_seconds = time.monotonic() - setup_started
    free_cons = count_free_constraints(model)

    solver = model_builder.ModelSolver("SAT")
    solver.enable_output(True)
    solver.set_time_limit_in_seconds(budget)
    solver.set_solver_specific_parameters(build_parameters(workers, seed))
    setup_seconds = time.monotonic() - setup_started

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cpsat-log-") as tmpdir:
        log_path = Path(tmpdir) / "solve.log"
        with capture_stdout_fd(log_path):
            status = solver.solve(model)
        wall = time.monotonic() - started
        trace = parse_trace_file(log_path)
    has_solution = status in (
        model_builder.SolveStatus.OPTIMAL,
        model_builder.SolveStatus.FEASIBLE,
    )

    record: dict[str, object] = {
        "status": "feasible" if has_solution else "no_solution",
        "cpsat_status": status.name,
        "wall_seconds": wall,
        "read_seconds": read_seconds,
        "setup_seconds": setup_seconds,
        "n_vars": model.num_variables,
        "n_cons": model.num_constraints,
        "n_free_cons": free_cons,
        "objective": None,
    }
    if status == model_builder.SolveStatus.INVALID_SOLVER_PARAMETERS:
        record["status"] = "invalid_parameters"
    elif not has_solution and status in (
        model_builder.SolveStatus.MODEL_INVALID,
        model_builder.SolveStatus.ABNORMAL,
    ):
        # CP-SAT scales continuous columns to integers and rejects what it cannot
        # express (MODEL_INVALID), and ABNORMAL means it errored out. Both are "did
        # not search", not "searched and found nothing", so they are tallied apart
        # rather than counted against the baseline as a search failure.
        #
        # NOT_SOLVED deliberately stays `no_solution`: it is the ordinary outcome of
        # a time-limited run that found nothing, which is exactly what the metric is
        # asking about. The precise verdict survives in `cpsat_status` and reaches
        # the comparison table as `solver_status`.
        record["status"] = "invalid_model"

    # Whether the incumbent profile came from the log or is a single end-point.
    # A systematic regex miss after an OR-Tools log-format change would otherwise
    # score every CP-SAT instance ~2.0, indistinguishable from "CP-SAT is bad".
    record["trace_source"] = "log" if trace else "final_only"

    values: dict[str, float] = {}
    if has_solution:
        objective = float(solver.objective_value)
        record["objective"] = objective
        # Keyed by name, not by index: the verifier re-reads the instance with a
        # different reader, so a positional dump would verify clean against a
        # differently ordered parse of the same file.
        values = {var.name: float(solver.value(var)) for var in model.get_variables()}
        # The log is the source of truth for *when* each incumbent appeared, but it
        # prints rounded values and the final solution can land after the last logged
        # line. Append the exact final objective so the tail of the profile is right.
        if not trace or trace[-1][1] != objective:
            trace.append((min(wall, budget), objective))
    return record, trace, values


def solution_text(instance: str, objective: object, values: dict[str, float]) -> str:
    """The MIPLIB-style solution format, shared with the CBLS runner.

    `repr` rather than a format string: a rounded value can violate a row the true
    one satisfies, which would read as a solver defect rather than a lossy dump.
    """
    lines = [f"# instance {instance}", "# engine cpsat", f"=obj= {objective!r}"]
    lines += [f"{name} {value!r}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def write_outputs(
    out_dir: Path,
    instance: str,
    record: dict[str, object],
    trace: list[tuple[float, float]],
    args: argparse.Namespace,
    values: dict[str, float] | None = None,
) -> dict[str, object]:
    """Write the trace, the solution vector and the result; return what was written.

    Returns the record rather than mutating the caller's, because a solution the
    job could not write downgrades the status and withholds the objective — and
    main() has to report and exit on what actually landed on disk.
    """
    record = dict(record)
    record.update(
        engine="cpsat",
        instance=instance,
        # Reported per result so a full-roster run's concurrency can be sized from
        # measurement rather than guessed.
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        budget_seconds=args.budget,
        seed=args.seed,
        workers=args.workers,
        ortools_version=version("ortools"),
        parameters=build_parameters(args.workers, args.seed),
    )
    lines = ["time_seconds,objective"]
    lines += [f"{t},{obj}" for t, obj in trace]
    (out_dir / f"{instance}.trace.csv").write_text("\n".join(lines) + "\n")

    if args.solution_dir and values:
        solution_dir = Path(args.solution_dir)
        solution_dir.mkdir(parents=True, exist_ok=True)
        try:
            # Temp-then-rename, like every other file this benchmark writes: a job
            # killed mid-write must leave no solution rather than a truncated one,
            # which would verify as an infeasible point.
            tmp_solution = solution_dir / f"{instance}.sol.tmp"
            tmp_solution.write_text(solution_text(instance, record["objective"], values))
            tmp_solution.replace(solution_dir / f"{instance}.sol")
        except OSError as exc:
            # No solution file means no independent verdict, and a row with no
            # verdict must not publish a number (#138).
            record["status"] = "solution_write_error"
            record["message"] = f"could not write the solution vector: {exc}"
            record["objective"] = None

    # Result last, and via a rename. The driver resumes on the result file's
    # existence, so it must not appear before the trace it is scored with, and it
    # must never appear truncated.
    tmp = out_dir / f"{instance}.json.tmp"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    tmp.replace(out_dir / f"{instance}.json")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--inst-dir", default="benchmarks/instances/mipfeas")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budget", type=float, default=600.0, help="seconds (MIPfeas uses 600)")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="CP-SAT threads; 1 runs both the fj and ls workers, matching CBLS",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--solution-dir",
        default=None,
        help="write the solution vector of a feasible run here, for "
        "verify_solution.py to check against the original instance file",
    )
    args = parser.parse_args()

    mps_path = Path(args.inst_dir) / f"{args.instance}.mps.gz"
    if not mps_path.exists():
        # No result file: an absent instance is an incomplete run, not a zero score.
        print(
            f"{mps_path} not found. Fetch the roster first:\n  python {args.inst_dir}/download.py",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record, trace, values = solve(mps_path, args.budget, args.workers, args.seed)
    record = write_outputs(out_dir, args.instance, record, trace, args, values)

    objective = record.get("objective")
    print(
        f"{args.instance:<28} {str(record['status']):<12} "
        f"obj={objective if objective is not None else 'n/a':<16} "
        f"{float(record['wall_seconds']):8.2f}s"  # type: ignore[arg-type]
    )
    # A rejected parameter string or an unreadable instance is a harness fault, not
    # a search outcome. An OR-Tools release renaming `filter_subsolvers` would
    # otherwise score every CP-SAT instance at 2.0 across a 39 CPU-hour run, at exit
    # 0. Matches the CBLS runner, which also exits 1 on a read error — and on a
    # solution it could not write, which leaves a row nothing can verify.
    return (
        1 if record["status"] in ("invalid_parameters", "read_error", "solution_write_error") else 0
    )


if __name__ == "__main__":
    sys.exit(main())
