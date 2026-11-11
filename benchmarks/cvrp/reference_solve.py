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
across thread counts, and a multi-thread CBLS row set is labelled as one.

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

import importlib.metadata
import itertools
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pyvrp import Client, Depot, Location, ProblemData, Result, VehicleType, solve

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.cvrp import baseline_cli  # noqa: E402
from benchmarks.cvrp.baseline_cli import DEFAULT_INST_DIR as DEFAULT_INST_DIR  # noqa: E402
from benchmarks.cvrp.verify_solution import (  # noqa: E402
    VrpInstance,
    distance_matrix,
    read_vrp_file,
)

if TYPE_CHECKING:
    import argparse

ENGINE = "pyvrp"


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


def run(args: argparse.Namespace) -> int:
    inst = read_vrp_file(args.inst_dir / "X" / f"{args.instance}.vrp")
    data = problem_data(inst)
    started = time.monotonic()
    result = solve(data, stop=Deadline(args.budget), seed=args.seed, collect_stats=True)
    wall = time.monotonic() - started
    feasible = bool(result.best.is_feasible())
    outcome = baseline_cli.Outcome(
        engine=ENGINE,
        feasible=feasible,
        routes=routes_of(data, result),
        cost=int(result.best.distance()),
        fleet=inst.customers,
        wall=wall,
        extra={
            "solver_runtime": result.runtime,
            "time_to_first_feasible": time_to_first_feasible(result, wall) if feasible else None,
            "iterations": result.num_iterations,
            "pyvrp_version": importlib.metadata.version("pyvrp"),
        },
    )
    return baseline_cli.finish(args, inst, outcome)


def main(argv: list[str] | None = None) -> int:
    args = baseline_cli.parse_args(__doc__, argv)
    if baseline_cli.refuse_args(args, "PyVRP is single-threaded"):
        return 2
    baseline_cli.clear_outputs(args)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
