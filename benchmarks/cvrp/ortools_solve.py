"""OR-Tools routing baseline for the CVRP benchmark (#204).

OR-Tools' vehicle routing library (`pywrapcp.RoutingModel`) with its guided
local search metaheuristic is the second open-source baseline beside PyVRP
(`reference_solve.py`), run on the same machine, at the same budget, over the
same instance: `ortools` is already a `benchmarks` dependency (CP-SAT's
MIPfeas arm), so this adds no package.

THE SAME INSTANCE. The arc costs are `verify_solution.distance_matrix`, TSPLIB
EUC_2D `nint` rounded per edge, registered as a transit matrix: the matrix the
verifier recomputes every cost with. Capacity is a dimension with zero slack
over the CVRPLIB demands.

THE FLEET. `cbls_cvrp`'s own default, `R = k + max(3, ceil(0.1 k))`
(`default_fleet` in `cvrp_model.h`), from the `k` in the X name. PyVRP gets a
fleet of `n`; OR-Tools allocates per-vehicle state, so a fleet of 1000 on the
largest X instances is a different model size, not a looser constraint, and
`R` is the fleet the engine under test solves with. Unused vehicles cost
nothing, so `R` binds only where it binds for `cbls_cvrp` too.

THE SEARCH. First solution `PATH_CHEAPEST_ARC`, then `GUIDED_LOCAL_SEARCH`,
both OR-Tools' documented CVRP defaults; every other parameter is the library
default. The routing search parameters carry no random seed, so the search is
deterministic up to where the time limit cuts it: `--seed` is recorded but
changes nothing, and the spread across seeds is timing alone.

THREADS. The routing search is single-threaded, so `--threads` must be 1, as
for PyVRP, and its rows pair only with `cbls_cvrp --threads 1`.

THE CLOCK. Budget and `time_to_first_feasible` run from before the model is
built, as `cbls_cvrp`'s cover its whole search: the routing time limit is what
is left of the budget once the model exists. Every solution the routing search
reports satisfies the capacity dimension, so the first one is the first
feasible.

THE CLI AND THE RECORD mirror `reference_solve.py`'s, so `run_benchmark.py`
runs it as the `ortools` arm and `score.py` scores it alike; `engine` is
`ortools`. Usage:

    .venv/bin/python benchmarks/cvrp/ortools_solve.py --instance X-n101-k25 \\
        --budget 10 --seed 1 --out r.json --solution-out r.sol
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write  # noqa: E402
from benchmarks.cvrp.reference_solve import bks_of, solution_text  # noqa: E402
from benchmarks.cvrp.verify_solution import (  # noqa: E402
    VrpInstance,
    check_routes,
    distance_matrix,
    read_vrp_file,
)

DEFAULT_INST_DIR = Path(__file__).resolve().parents[1] / "instances" / "cvrp"
ENGINE = "ortools"
ARM = "ortools"


def name_k(instance: str) -> int | None:
    """The `k` an X name carries (`X-n101-k25` -> 25), or None."""
    match = re.search(r"-k(\d+)$", instance)
    return int(match.group(1)) if match else None


def default_fleet(k: int) -> int:
    """`cbls_cvrp`'s default fleet, `cvrp_model.h`'s `default_fleet`."""
    return k + max(3, math.ceil(0.1 * k))


@dataclass
class Routing:
    """The routing model of one instance, node `c` = CVRPLIB customer `c`."""

    manager: pywrapcp.RoutingIndexManager
    model: pywrapcp.RoutingModel


def build_routing(inst: VrpInstance, fleet: int) -> Routing:
    nodes = inst.customers + 1
    manager = pywrapcp.RoutingIndexManager(nodes, fleet, 0)
    model = pywrapcp.RoutingModel(manager)
    arc = model.RegisterTransitMatrix(distance_matrix(inst))
    model.SetArcCostEvaluatorOfAllVehicles(arc)
    demand = model.RegisterUnaryTransitVector(list(inst.demands))
    model.AddDimensionWithVehicleCapacity(demand, 0, [inst.capacity] * fleet, True, "load")
    return Routing(manager, model)


def search_parameters(seconds: float) -> object:
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    # Milliseconds: the proto's Duration takes whole seconds and nanos, and a
    # sub-second remainder would otherwise be lost.
    params.time_limit.FromMilliseconds(max(int(seconds * 1000), 1))
    return params


def routes_of(routing: Routing, solution: pywrapcp.Assignment, fleet: int) -> list[list[int]]:
    """Each vehicle's customers, in visiting order, as CVRPLIB ids."""
    routes: list[list[int]] = []
    for vehicle in range(fleet):
        route: list[int] = []
        index = solution.Value(routing.model.NextVar(routing.model.Start(vehicle)))
        while not routing.model.IsEnd(index):
            route.append(routing.manager.IndexToNode(index))
            index = solution.Value(routing.model.NextVar(index))
        routes.append(route)
    return routes


def run(args: argparse.Namespace) -> int:
    inst = read_vrp_file(args.inst_dir / "X" / f"{args.instance}.vrp")
    k = name_k(args.instance)
    if k is None:
        print(f"{args.instance}: the name carries no k", file=sys.stderr)
        return 2
    fleet = default_fleet(k)
    started = time.monotonic()
    routing = build_routing(inst, fleet)
    # Seconds from `started` of every solution the search reports.
    found: list[float] = []
    routing.model.AddAtSolutionCallback(lambda: found.append(time.monotonic() - started))
    remaining = args.budget - (time.monotonic() - started)
    solution = routing.model.SolveWithParameters(search_parameters(remaining))
    wall = time.monotonic() - started
    feasible = solution is not None
    routes = routes_of(routing, solution, fleet) if feasible else []
    cost = int(solution.ObjectiveValue()) if feasible else 0
    check = check_routes(inst, routes, cost)
    verified = feasible and check.ok
    for error in check.errors if feasible else []:
        print(f"{args.instance}: VERIFY FAILED: {error}", file=sys.stderr)
    if verified and args.solution_out is not None:
        atomic_write(args.solution_out, solution_text(routes, check.cost))
    bks = bks_of(args.inst_dir, args.instance)
    objective = check.cost if verified else None
    record: dict[str, object] = {
        "engine": ENGINE,
        "arm": ARM,
        "instance": args.instance,
        "customers": inst.customers,
        "capacity": inst.capacity,
        "bks": bks,
        "objective": objective,
        "gap_percent": None if objective is None or bks is None else 100 * (objective - bks) / bks,
        "feasible": feasible,
        "verified": verified,
        "verify_errors": check.errors if feasible else [],
        "routes_used": check.routes_used,
        "fleet": fleet,
        "budget_seconds": args.budget,
        "seed": args.seed,
        "threads": 1,
        "wall_seconds": wall,
        "time_to_first_feasible": found[0] if feasible and found else None,
        "solutions_found": len(found),
        "routing_status": int(routing.model.status()),
        "workers_launched": 1,
        "workers_completed": 1,
        "commit_sha": args.commit,
        "ortools_version": importlib.metadata.version("ortools"),
    }
    if args.out is not None:
        atomic_write(args.out, json.dumps(record, indent=2, sort_keys=True) + "\n")
    gap = record["gap_percent"]
    print(
        f"{args.instance:14s} feasible={int(feasible)} verified={int(verified)} "
        f"obj={objective} gap={gap if gap is None else f'{gap:.2f}'}% {wall:.2f}s"
    )
    # Same contract as cbls_cvrp: a claimed-feasible solution that fails the
    # check is an error; an infeasible run is a result.
    return 1 if feasible and not check.ok else 0


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads != 1:
        print(
            "--threads must be 1: the OR-Tools routing search is single-threaded", file=sys.stderr
        )
        return 2
    if not args.budget > 0:
        print("--budget must be a positive number of seconds", file=sys.stderr)
        return 2
    # As cbls_cvrp does: a previous run's files must not be read as this one's.
    for path in (args.out, args.solution_out):
        if path is not None:
            path.unlink(missing_ok=True)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
