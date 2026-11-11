"""OR-Tools routing baseline for the CVRP benchmark (#204).

OR-Tools' routing library (`pywrapcp.RoutingModel`) with guided local search:
the second open-source baseline beside PyVRP (`reference_solve.py`), on the
same machine, budget and instance. `ortools` is already a `benchmarks`
dependency (CP-SAT's MIPfeas arm), so this adds no package.

THE SAME INSTANCE. The arc costs are `verify_solution.distance_matrix`, TSPLIB
EUC_2D `nint` rounded per edge, registered as a transit matrix: the matrix the
verifier recomputes every cost with. Capacity is a dimension with zero slack
over the CVRPLIB demands.

THE FLEET. `cbls_cvrp`'s own default, `R = k + max(3, ceil(0.1 k))`
(`default_fleet` in `cvrp_model.h`), from the `k` in the X name. PyVRP gets a
fleet of `n`, but OR-Tools allocates per-vehicle state, so a fleet of 1000 on
the largest X instances is a different model size; `R` is the fleet the engine
under test solves with, and unused vehicles cost nothing. `R` can bind harder
than for `cbls_cvrp`, which searches through capacity-infeasible states:
OR-Tools' greedy construction must place every customer within `R` routes, or
the solve fails before any search with `feasible=false` and `routing_status`
ROUTING_FAIL_TIMEOUT (measured on X-n101-k25 at fleets 1 and 10), the same
status a budget too short for the construction gives. Few customers per route
and a tight `k` are likeliest; `score.py` lists every such row, so read them
on a roster before publishing its OR-Tools rows.

THE SEARCH. First solution `PATH_CHEAPEST_ARC`, then `GUIDED_LOCAL_SEARCH`,
both OR-Tools' documented CVRP defaults; every other parameter is the library
default. The routing search parameters carry no random seed, so the search is
deterministic up to where the time limit cuts it: `--seed` is recorded but
changes nothing, and the spread across seeds is timing alone.

THREADS. The routing search is single-threaded, so `--threads` must be 1, as
for PyVRP, and its rows pair only with `cbls_cvrp --threads 1`.

THE CLOCK. Budget and `time_to_first_feasible` run from the
`SolveWithParameters` call, as PyVRP's run from `solve()` and `cbls_cvrp`'s from
the start of its search: reading the instance and building and closing the
routing model sit outside, recorded as `build_seconds` as `cbls_cvrp` records its model
build. The first-solution heuristic runs inside the time limit, as PyVRP's
construction does not, so the first solution is charged to the budget. Every
solution the routing search reports satisfies the capacity dimension, so the
first one is the first feasible.

THE CLI AND THE RECORD are `reference_solve.py`'s (`baseline_cli.py`), so `run_benchmark.py`
runs it as the `ortools` arm and `score.py` scores it alike; `engine` is
`ortools`. Usage:

    .venv/bin/python benchmarks/cvrp/ortools_solve.py --instance X-n101-k25 \\
        --budget 10 --seed 1 --out r.json --solution-out r.sol
"""

from __future__ import annotations

import importlib.metadata
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from typing import TYPE_CHECKING

from benchmarks.cvrp import baseline_cli  # noqa: E402
from benchmarks.cvrp.baseline_cli import DEFAULT_INST_DIR as DEFAULT_INST_DIR  # noqa: E402
from benchmarks.cvrp.verify_solution import (  # noqa: E402
    VrpInstance,
    distance_matrix,
    read_vrp_file,
)

if TYPE_CHECKING:
    import argparse

ENGINE = "ortools"


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


def run(args: argparse.Namespace, fleet: int) -> int:
    inst = read_vrp_file(args.inst_dir / "X" / f"{args.instance}.vrp")
    build_started = time.monotonic()
    routing = build_routing(inst, fleet)
    # Seconds from `started` of every solution the search reports.
    found: list[float] = []
    started: float  # set below, before the search can call back
    routing.model.AddAtSolutionCallback(lambda: found.append(time.monotonic() - started))
    params = search_parameters(args.budget)
    # Closing builds the search's structures (vehicle classes, filters,
    # neighbourhoods): model build, so outside the clock, where Solve would do it.
    routing.model.CloseModelWithParameters(params)
    started = time.monotonic()
    build_seconds = started - build_started
    solution = routing.model.SolveWithParameters(params)
    wall = time.monotonic() - started
    feasible = solution is not None
    outcome = baseline_cli.Outcome(
        engine=ENGINE,
        feasible=feasible,
        routes=routes_of(routing, solution, fleet) if feasible else [],
        cost=int(solution.ObjectiveValue()) if feasible else 0,
        fleet=fleet,
        wall=wall,
        extra={
            "build_seconds": build_seconds,
            "time_to_first_feasible": found[0] if feasible and found else None,
            "solutions_found": len(found),
            "routing_status": int(routing.model.status()),
            "ortools_version": importlib.metadata.version("ortools"),
        },
    )
    return baseline_cli.finish(args, inst, outcome)


def main(argv: list[str] | None = None) -> int:
    args = baseline_cli.parse_args(__doc__, argv)
    if baseline_cli.refuse_args(args, "the OR-Tools routing search is single-threaded"):
        return 2
    k = name_k(args.instance)
    if k is None:
        print(f"{args.instance}: the name carries no k", file=sys.stderr)
        return 2
    baseline_cli.clear_outputs(args)
    return run(args, default_fleet(k))


if __name__ == "__main__":
    sys.exit(main())
