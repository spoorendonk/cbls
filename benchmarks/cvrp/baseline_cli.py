"""The CLI and record shared by the CVRP baselines (`reference_solve.py`,
`ortools_solve.py`): argument parsing, the refusals, clearing a previous run's
files, verifying the routes, writing the solution and the JSON record, and
the exit code. Each baseline supplies only its solver call and its own
record fields, so `run_benchmark.py` reads one record shape from both.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write  # noqa: E402
from benchmarks.cvrp.verify_solution import VrpInstance, check_routes  # noqa: E402
from benchmarks.instances.cvrp.download import read_roster  # noqa: E402

DEFAULT_INST_DIR = Path(__file__).resolve().parents[1] / "instances" / "cvrp"


def solution_text(routes: list[list[int]], cost: int) -> str:
    lines = [
        f"Route #{i}: {' '.join(map(str, r))}" for i, r in enumerate((r for r in routes if r), 1)
    ]
    return "\n".join([*lines, f"Cost {cost}"]) + "\n"


def bks_of(inst_dir: Path, instance: str) -> float | None:
    for entry in read_roster(inst_dir / "roster.csv"):
        if entry.instance == instance:
            return entry.bks
    return None


def parse_args(doc: str | None, argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--instance", required=True)
    parser.add_argument("--inst-dir", type=Path, default=DEFAULT_INST_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--solution-out", type=Path, default=None)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--commit", default="unknown")
    return parser.parse_args(argv)


def refuse_args(args: argparse.Namespace, single_thread_reason: str) -> bool:
    """Reports and returns True when `args` cannot run."""
    if args.threads != 1:
        print(f"--threads must be 1: {single_thread_reason}", file=sys.stderr)
        return True
    if not args.budget > 0:
        print("--budget must be a positive number of seconds", file=sys.stderr)
        return True
    return False


def clear_outputs(args: argparse.Namespace) -> None:
    """As cbls_cvrp does: a previous run's files must not be read as this one's."""
    for path in (args.out, args.solution_out):
        if path is not None:
            path.unlink(missing_ok=True)


@dataclass
class Outcome:
    """What a baseline's solve produced, before verification."""

    engine: str
    feasible: bool
    routes: list[list[int]]
    cost: int
    fleet: int
    wall: float
    #: The baseline's own record fields, beside the shared ones.
    extra: dict[str, object] = field(default_factory=dict)


def finish(args: argparse.Namespace, inst: VrpInstance, outcome: Outcome) -> int:
    """Verifies `outcome`, writes the solution and the record, prints the
    summary line and returns the exit code."""
    feasible = outcome.feasible
    check = check_routes(inst, outcome.routes, outcome.cost)
    verified = feasible and check.ok
    for error in check.errors if feasible else []:
        print(f"{args.instance}: VERIFY FAILED: {error}", file=sys.stderr)
    if verified and args.solution_out is not None:
        atomic_write(args.solution_out, solution_text(outcome.routes, check.cost))
    bks = bks_of(args.inst_dir, args.instance)
    objective = check.cost if verified else None
    gap = None if objective is None or bks is None else 100 * (objective - bks) / bks
    record: dict[str, object] = {
        "engine": outcome.engine,
        "arm": outcome.engine,
        "instance": args.instance,
        "customers": inst.customers,
        "capacity": inst.capacity,
        "bks": bks,
        "objective": objective,
        "gap_percent": gap,
        "feasible": feasible,
        "verified": verified,
        "verify_errors": check.errors if feasible else [],
        "routes_used": check.routes_used,
        "fleet": outcome.fleet,
        "budget_seconds": args.budget,
        "seed": args.seed,
        "threads": 1,
        "wall_seconds": outcome.wall,
        "workers_launched": 1,
        "workers_completed": 1,
        "commit_sha": args.commit,
        **outcome.extra,
    }
    if args.out is not None:
        atomic_write(args.out, json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"{args.instance:14s} feasible={int(feasible)} verified={int(verified)} "
        f"obj={objective} gap={gap if gap is None else f'{gap:.2f}'}% {outcome.wall:.2f}s"
    )
    # Same contract as cbls_cvrp: a claimed-feasible solution that fails the
    # check is an error; an infeasible run is a result.
    return 1 if feasible and not check.ok else 0
