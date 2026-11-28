"""Model a CVRP with List variables, solve it and read the routes (docs/examples/cvrp.md).

Usage: python examples/cvrp.py benchmarks/instances/cvrp/X/X-n115-k10.vrp
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cbls
import numpy as np


@dataclass
class Instance:
    """Node 0 is the depot; customer element e (0-based) is node e + 1."""

    name: str
    capacity: int
    coords: list[tuple[float, float]]
    demands: list[int]


def read_vrp(path: str) -> Instance:
    """A TSPLIB CVRP file as CVRPLIB's X set writes it: EUC_2D, depot at node 1."""
    header: dict[str, str] = {}
    coords: list[tuple[float, float]] = []
    demands: list[int] = []
    section = ""
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if line in ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION", "EOF"):
            section = line
        elif not line or section in ("DEPOT_SECTION", "EOF"):
            continue
        elif not section:
            key, _, value = line.partition(":")
            header[key.strip()] = value.strip()
        elif section == "NODE_COORD_SECTION":
            _, x, y = line.split()
            coords.append((float(x), float(y)))
        else:
            demands.append(int(line.split()[1]))
    if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D" or demands[0] != 0:
        raise ValueError(f"{path}: expected EUC_2D distances and the depot at node 1")
    return Instance(header["NAME"], int(header["CAPACITY"]), coords, demands)


def distances(inst: Instance) -> np.ndarray:
    """TSPLIB's EUC_2D: the Euclidean distance rounded to the nearest integer."""
    return np.array(
        [[math.floor(math.dist(a, b) + 0.5) for b in inst.coords] for a in inst.coords],
        dtype=np.float64,
    )


def build(inst: Instance, dist: np.ndarray, fleet: int) -> tuple[cbls.Model, list[int]]:
    """`fleet` routes partitioning the customers, each within capacity, total length minimised."""
    n = len(inst.demands) - 1
    model = cbls.Model()
    # One List per route: distinct customers 0..n-1 in visiting order.
    routes = [model.list_var(n, 0, n, cbls.ListInit.Empty, f"route{r}") for r in range(fleet)]
    # Every customer on exactly one route: the search starts from a random
    # complete partition and its moves keep it so.
    model.add_list_partition(routes, cbls.Cover.Exact)
    demand = np.array(inst.demands[1:], dtype=np.float64)
    capacity = model.constant(float(inst.capacity))
    between = np.ascontiguousarray(dist[1:, 1:])
    from_depot = np.ascontiguousarray(dist[0, 1:])
    to_depot = np.ascontiguousarray(dist[1:, 0])
    lengths = []
    for route in routes:
        model.add_constraint(model.leq(model.lambda_table_sum(route, demand), capacity))
        # The legs between consecutive customers, plus the depot legs; an
        # empty route costs 0.
        lengths.append(model.pair_table_sum(route, between, head=from_depot, tail=to_depot))
    model.minimize(model.sum(lengths))
    model.close()
    return model, routes


def verify(inst: Instance, dist: np.ndarray, routes: list[list[int]], cost: float) -> list[str]:
    """Each customer on one route, each route within capacity, and the cost as stated."""
    errors: list[str] = []
    visits = sorted(c for route in routes for c in route)
    if visits != list(range(1, len(inst.demands))):
        errors.append("the routes do not visit every customer exactly once")
    for r, route in enumerate(routes, 1):
        load = sum(inst.demands[c] for c in route)
        if load > inst.capacity:
            errors.append(f"route {r} carries {load} > {inst.capacity}")
    length = sum(
        dist[a, b] for route in routes for a, b in zip([0, *route], [*route, 0], strict=True)
    )
    if length != cost:
        errors.append(f"the routes are {length:g} long, the solve reported {cost:g}")
    return errors


def main(path: str) -> int:
    inst = read_vrp(path)
    dist = distances(inst)
    n = len(inst.demands) - 1
    k = int(inst.name.rsplit("-k", 1)[1])  # an X name's minimum vehicle count
    fleet = k + 3  # room to spare: the search may leave routes empty
    model, routes = build(inst, dist, fleet)
    print(f"{inst.name}: {n} customers, capacity {inst.capacity}, fleet {fleet}")

    config = cbls.SearchConfig()
    # Moves relocate a customer next to one of its 20 nearest customers.
    config.structural_neighbours = cbls.nearest_neighbours(
        n, 20, lambda a, b: float(dist[a + 1, b + 1])
    )
    result = cbls.solve(model, time_limit=5.0, seed=1, threads=1, lns=0.2, config=config)

    print(f"feasible  = {result.feasible}")
    print(f"objective = {result.objective:.0f}")
    # A List handle h is variable -(h + 1); its elements are the customers in
    # visiting order, printed here as node numbers (CVRPLIB's .sol numbering).
    visited = [[e + 1 for e in result.elements[-(h + 1)]] for h in routes]
    visited = [route for route in visited if route]
    for r, route in enumerate(visited, 1):
        print(f"Route #{r}: {' '.join(map(str, route))}")
    errors = verify(inst, dist, visited, result.objective)
    print("verify: PASS" if not errors else f"verify: FAIL ({len(errors)} errors)")
    for error in errors:
        print("  " + error)
    return 0 if result.feasible and not errors else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} INSTANCE.vrp", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
