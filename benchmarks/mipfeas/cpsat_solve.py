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
    from collections.abc import Iterator

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
    trace: list[tuple[float, float]] = []
    for line in log_text.splitlines():
        match = SOLUTION_LINE.match(line.strip())
        if match is None:
            continue
        trace.append((float(match.group(2)), float(match.group(3))))
    return trace


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
) -> tuple[dict[str, object], list[tuple[float, float]]]:
    """Run the LS-only CP-SAT configuration; return (result record, incumbent trace)."""
    # ortools ships no annotations for ModelBuilder's constructor; the rest of the
    # model_builder surface used here is typed.
    model = model_builder.ModelBuilder()  # type: ignore[no-untyped-call]
    if not model.import_from_mps_file(str(mps_path)):
        # Same keys as the solved path. A key missing here crashes the job *after*
        # write_outputs has already written its result, and the driver — seeing a
        # result file — then reports the crash as a clean run.
        return {
            "status": "read_error",
            "message": f"CP-SAT could not import {mps_path.name}",
            "wall_seconds": 0.0,
            "objective": None,
        }, []

    solver = model_builder.ModelSolver("SAT")
    solver.enable_output(True)
    solver.set_time_limit_in_seconds(budget)
    solver.set_solver_specific_parameters(build_parameters(workers, seed))

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="cpsat-log-") as tmpdir:
        log_path = Path(tmpdir) / "solve.log"
        with capture_stdout_fd(log_path):
            status = solver.solve(model)
        wall = time.monotonic() - started
        log_text = log_path.read_text()

    trace = parse_trace(log_text)
    has_solution = status in (
        model_builder.SolveStatus.OPTIMAL,
        model_builder.SolveStatus.FEASIBLE,
    )

    record: dict[str, object] = {
        "status": "feasible" if has_solution else "no_solution",
        "cpsat_status": status.name,
        "wall_seconds": wall,
        "n_vars": model.num_variables,
        "n_cons": model.num_constraints,
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

    if has_solution:
        objective = float(solver.objective_value)
        record["objective"] = objective
        # The log is the source of truth for *when* each incumbent appeared, but it
        # prints rounded values and the final solution can land after the last logged
        # line. Append the exact final objective so the tail of the profile is right.
        if not trace or trace[-1][1] != objective:
            trace.append((min(wall, budget), objective))
    return record, trace


def write_outputs(
    out_dir: Path,
    instance: str,
    record: dict[str, object],
    trace: list[tuple[float, float]],
    args: argparse.Namespace,
) -> None:
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
    (out_dir / f"{instance}.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")

    lines = ["time_seconds,objective"]
    lines += [f"{t},{obj}" for t, obj in trace]
    (out_dir / f"{instance}.trace.csv").write_text("\n".join(lines) + "\n")


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

    record, trace = solve(mps_path, args.budget, args.workers, args.seed)
    write_outputs(out_dir, args.instance, record, trace, args)

    objective = record.get("objective")
    print(
        f"{args.instance:<28} {str(record['status']):<12} "
        f"obj={objective if objective is not None else 'n/a':<16} "
        f"{float(record['wall_seconds']):8.2f}s"  # type: ignore[arg-type]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
