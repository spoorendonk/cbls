"""PyVRP reference baseline for the CVRP benchmark (#200).

PyVRP (Wouda, Lan & Kool, "PyVRP: a high-performance VRP solver package",
INFORMS J. Computing 2024, https://doi.org/10.1287/ijoc.2023.0055;
https://github.com/PyVRP/PyVRP) is an open-source solver of the HGS-CVRP
lineage. It is the reference solver this benchmark runs on the same machine, at
the same budget, so a gap to BKS can be split into what the budget costs and
what the engine costs. SCIP cannot handle CVRP at n = 100..1000 within these
budgets, hence another open-source solver (the benchmark workflow's step 2).

THE SAME INSTANCE. The distance matrix is `verify_solution.distance_matrix`:
TSPLIB EUC_2D `nint`, rounded per edge, from the shared Python `.vrp` reader --
the same matrix the verifier recomputes every `cbls_cvrp` cost with, and whose
agreement with `cbls_cvrp`'s own cost the scorer enforces on every row. The
coordinates are passed to PyVRP too but its search reads only the matrices.
Durations are all zero.

THE FLEET. One vehicle type with `num_available = n`, the customer count: a
fleet that can never bind, larger than `cbls_cvrp`'s default
`R = k + max(3, ceil(0.1 k))` on every X instance. The `k` in an X name is a
minimum, and neither engine is held to it.

THREADS. PyVRP is single-threaded, so `--threads` must be 1. Its rows are
compared only against `cbls_cvrp --threads 1`; `score.py` refuses a paired A/B
across thread counts, and a multi-thread CBLS row is reported on its own.

THE CLI AND THE RECORD mirror `cbls_cvrp`'s, so `run_benchmark.py` runs it as
the `pyvrp` arm and `verify_solution.py` / `score.py` score both engines alike:
`--out` receives a JSON object carrying `score.RECORD_FIELDS` (`engine` is
`pyvrp`), `--solution-out` a CVRPLIB `.sol`, written only when the best solution
is feasible.

THE CLOCK. Budget and `time_to_first_feasible` both run from the `solve()`
call, as `cbls_cvrp`'s run from the start of its search: PyVRP's own
`MaxRuntime` and statistics start only after its construction phase (`Deadline`,
`time_to_first_feasible`). Reading the instance and building `ProblemData` sit
outside, as `cbls_cvrp`'s model build does (its `build_seconds`).

    .venv/bin/python benchmarks/cvrp/reference_solve.py --instance X-n101-k25 \\
        --budget 10 --seed 1 --out r.json --solution-out r.sol
"""

from __future__ import annotations

import argparse
import importlib.metadata
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from pyvrp import Client, Depot, Location, ProblemData, Result, VehicleType, solve

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.common.records import atomic_write  # noqa: E402
from benchmarks.cvrp.verify_solution import (  # noqa: E402
    VrpInstance,
    check_routes,
    distance_matrix,
    read_vrp_file,
)
from benchmarks.instances.cvrp.download import read_roster  # noqa: E402

DEFAULT_INST_DIR = Path(__file__).resolve().parents[1] / "instances" / "cvrp"
ENGINE = "pyvrp"
ARM = "pyvrp"


def problem_data(inst: VrpInstance) -> ProblemData:
    """The instance as PyVRP's `ProblemData`, depot at location 0 and customer
    id `c` (CVRPLIB numbering) at location `c`."""
    dist = np.asarray(distance_matrix(inst), dtype=np.int64)
    # PyVRP 0.14's stub types each matrix as `np.ndarray[int]`, the dtype in
    # numpy's shape parameter, which no array type satisfies and which cannot
    # even be spelled under current numpy stubs. The lists are passed as
    # `list[Any]`; what PyVRP needs at runtime is a 2-D int64 array, as here.
    distances: list[Any] = [dist]
    durations: list[Any] = [np.zeros_like(dist)]
    return ProblemData(
        locations=[Location(x=x, y=y) for x, y in inst.coords],
        clients=[
            Client(location=c, delivery=[inst.demands[c]]) for c in range(1, inst.customers + 1)
        ],
        depots=[Depot(location=0)],
        vehicle_types=[VehicleType(num_available=inst.customers, capacity=[inst.capacity])],
        distance_matrices=distances,
        duration_matrices=durations,
    )


def routes_of(data: ProblemData, result: Result) -> list[list[int]]:
    """The best solution's routes as CVRPLIB customer ids (= location indices)."""
    clients = data.clients()
    return [
        [clients[a.idx].location for a in route if a.is_client()] for route in result.best.routes()
    ]


class Deadline:
    """A PyVRP stopping criterion on a clock started before `solve()` is called.

    `MaxRuntime` starts its clock at the first stop check inside the iterated
    local search, after `solve()` has already built the neighbourhoods and run
    an exhaustive local search from a random start -- real search, outside the
    budget. `cbls_cvrp`'s budget covers its whole search, so this one must too.
    """

    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds

    def __call__(self, best_cost: int) -> bool:
        del best_cost  # the protocol's argument; a deadline does not read it
        return time.monotonic() >= self.deadline


def time_to_first_feasible(result: Result, wall: float) -> float:
    """Seconds from the `solve()` call to the first feasible best solution.

    PyVRP's statistics start with the iterated local search, so its setup --
    the time `solve()` spent before the first iteration, `wall` less the
    search's own `runtime` -- is added: the start solution comes out of that
    setup and is often feasible already. Resolution is one iteration; a
    feasible best with no iteration recorded reports the setup alone.
    """
    setup = max(wall - result.runtime, 0.0)
    elapsed = itertools.accumulate(result.stats.runtimes)
    for datum, t in zip(result.stats.data, elapsed, strict=False):
        if datum.best_feas:
            return setup + t
    return setup


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


def run(args: argparse.Namespace) -> int:
    inst = read_vrp_file(args.inst_dir / "X" / f"{args.instance}.vrp")
    data = problem_data(inst)
    started = time.monotonic()
    result = solve(data, stop=Deadline(args.budget), seed=args.seed, collect_stats=True)
    wall = time.monotonic() - started
    routes = routes_of(data, result)
    feasible = bool(result.best.is_feasible())
    cost = int(result.best.distance())
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
        "fleet": inst.customers,
        "budget_seconds": args.budget,
        "seed": args.seed,
        "threads": 1,
        "wall_seconds": wall,
        "solver_runtime": result.runtime,
        "time_to_first_feasible": time_to_first_feasible(result, wall) if feasible else None,
        "iterations": result.num_iterations,
        "workers_launched": 1,
        "workers_completed": 1,
        "commit_sha": args.commit,
        "pyvrp_version": importlib.metadata.version("pyvrp"),
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
        print("--threads must be 1: PyVRP is single-threaded", file=sys.stderr)
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
