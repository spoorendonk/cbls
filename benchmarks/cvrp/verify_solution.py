"""Independent check of a CVRP `.sol` against its `.vrp` (#199).

The CVRP analogue of `benchmarks/mipfeas/verify_solution.py`. It shares no code
with the C++ runner: the `.vrp` is read here, the distances are recomputed here,
and the only thing taken from elsewhere is `parse_sol`, the Python `.sol` parser
that pinned the BKS in the first place (`benchmarks/instances/cvrp/download.py`).
`cbls_cvrp` checks its own routes in `verify_cvrp.h` before writing them; this is
the second, independent opinion the scorer relies on, and the one a PyVRP
solution (#200) gets too.

A solution passes when:

* every route entry is a customer id in `1..n` (CVRPLIB's numbering: customer
  `c` is TSPLIB node `c + 1`, the depot is node 1 / id 0) -- so the depot never
  appears inside a route;
* every customer is visited exactly once;
* every route's load is at most the capacity `Q`;
* the cost recomputed as depot -> route -> depot over TSPLIB's EUC_2D `nint`
  distances equals the file's `Cost` line, and equals `expected_cost` when the
  caller has one (the record's reported objective).

There is no fleet check: the `k` in an X name is a minimum, not a limit, and
the BKS themselves use more routes (`benchmarks/instances/cvrp/README.md`).

`read_vrp` and `distance_matrix` are also the shared Python reader the PyVRP
baseline (#200) builds its distances from, so the test pinning PyVRP's matrix
pins it against the very matrix this verifier scores with.

    .venv/bin/python benchmarks/cvrp/verify_solution.py X-n101-k25.vrp X-n101-k25.sol
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.instances.cvrp.download import parse_sol  # noqa: E402


@dataclass(frozen=True)
class VrpInstance:
    """A TSPLIB CVRP instance. Node 0 is the depot; customer id `c` is node `c`."""

    name: str
    capacity: int
    #: (x, y) per node, depot first.
    coords: list[tuple[float, float]]
    #: Demand per node; `demands[0] == 0`, the depot.
    demands: list[int]

    @property
    def customers(self) -> int:
        return len(self.coords) - 1


_SECTIONS = ("NODE_COORD_SECTION", "DEMAND_SECTION", "DEPOT_SECTION")


def _scan(text: str) -> tuple[dict[str, str], dict[str, list[list[str]]]]:
    """Split a TSPLIB file into its `KEY : value` header and each section's
    token rows. Only the tokenising; what the values mean is `read_vrp`'s."""
    header: dict[str, str] = {}
    sections: dict[str, list[list[str]]] = {name: [] for name in _SECTIONS}
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        first = line.split()[0].upper() if line else ""
        if first == "EOF":
            break
        if first in _SECTIONS:
            section = first
        elif ":" in line and not line[0].isdigit() and line[0] != "-":
            key, _, value = line.partition(":")
            header[key.strip().upper()] = value.strip()
            section = ""
        elif line and section:
            sections[section].append(line.split())
    return header, sections


def _numbered(rows: list[list[str]], width: int, label: str) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for row in rows:
        if len(row) != width:
            raise ValueError(f"bad {label} line: {' '.join(row)!r}")
        out[int(row[0])] = row[1:]
    return out


def read_vrp(text: str) -> VrpInstance:
    """Parse a TSPLIB `.vrp` as the X set uses it, or raise `ValueError`.

    Only `TYPE: CVRP`, `EDGE_WEIGHT_TYPE: EUC_2D` and a single depot at TSPLIB
    node 1 are accepted: the BKS are costs in exactly that metric and numbering,
    so anything else is refused rather than guessed at.
    """
    header, sections = _scan(text)
    if header.get("TYPE", "").upper() != "CVRP":
        raise ValueError(f"TYPE {header.get('TYPE')!r} is not CVRP")
    if header.get("EDGE_WEIGHT_TYPE", "").upper() != "EUC_2D":
        raise ValueError(f"EDGE_WEIGHT_TYPE {header.get('EDGE_WEIGHT_TYPE')!r} is not EUC_2D")
    dimension = int(header["DIMENSION"])
    coords = _numbered(sections["NODE_COORD_SECTION"], 3, "NODE_COORD_SECTION")
    demands = _numbered(sections["DEMAND_SECTION"], 2, "DEMAND_SECTION")
    depots = [int(row[0]) for row in sections["DEPOT_SECTION"]]
    nodes = list(range(1, dimension + 1))
    if sorted(coords) != nodes or sorted(demands) != nodes:
        raise ValueError(f"coordinates or demands do not cover nodes 1..{dimension}")
    if depots != [1, -1]:
        raise ValueError(f"expected a single depot at node 1, got {depots}")
    if int(demands[1][0]) != 0:
        raise ValueError(f"the depot has demand {demands[1][0]}")
    return VrpInstance(
        name=header.get("NAME", ""),
        capacity=int(header["CAPACITY"]),
        coords=[(float(coords[i][0]), float(coords[i][1])) for i in nodes],
        demands=[int(demands[i][0]) for i in nodes],
    )


def read_vrp_file(path: Path) -> VrpInstance:
    return read_vrp(path.read_text())


def nint_distance(a: tuple[float, float], b: tuple[float, float]) -> int:
    """TSPLIB's EUC_2D distance: `nint` = floor(d + 0.5), rounded per edge.

    Not Python's `round`, which rounds halves to even and so differs on an exact
    half -- the same reason the C++ reader uses `floor` over `nearbyint`.
    """
    return math.floor(math.hypot(a[0] - b[0], a[1] - b[1]) + 0.5)


def distance_matrix(inst: VrpInstance) -> list[list[int]]:
    """The full (n+1) x (n+1) integer distance matrix, depot at index 0."""
    pts = inst.coords
    return [[nint_distance(p, q) for q in pts] for p in pts]


@dataclass
class Verdict:
    """The outcome of one check. `ok` holds only when `errors` is empty."""

    cost: int = 0
    reported_cost: float = math.nan
    routes_used: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _check_route(
    inst: VrpInstance, number: int, route: list[int], visits: list[int], out: Verdict
) -> int:
    """One non-empty route: its ids, its load, its visits. Returns its cost."""
    n = inst.customers
    load = 0
    cost = 0
    previous = 0
    valid = True
    for c in route:
        if not 1 <= c <= n:
            where = "the depot (id 0) inside the route" if c == 0 else f"customer {c}"
            out.errors.append(f"route {number} names {where}, outside 1..{n}")
            valid = False
            continue
        visits[c] += 1
        load += inst.demands[c]
        cost += nint_distance(inst.coords[previous], inst.coords[c])
        previous = c
    if valid and load > inst.capacity:
        out.errors.append(f"route {number} carries {load} > capacity {inst.capacity}")
    return cost + nint_distance(inst.coords[previous], inst.coords[0])


def check_routes(
    inst: VrpInstance,
    routes: list[list[int]],
    reported_cost: float,
    expected_cost: float | None = None,
) -> Verdict:
    """Check `routes` (CVRPLIB customer ids) against `inst`; see the module doc."""
    out = Verdict(reported_cost=reported_cost)
    n = inst.customers
    visits = [0] * (n + 1)
    for number, route in enumerate(routes, 1):
        if route:
            out.routes_used += 1
            out.cost += _check_route(inst, number, route, visits, out)
    missing = [c for c in range(1, n + 1) if visits[c] == 0]
    repeated = [c for c in range(1, n + 1) if visits[c] > 1]
    if missing:
        out.errors.append(f"{len(missing)} customer(s) not visited, first {missing[:5]}")
    if repeated:
        out.errors.append(
            f"{len(repeated)} customer(s) visited more than once, first {repeated[:5]}"
        )
    if out.errors:
        return out
    if out.cost != reported_cost:
        out.errors.append(f"recomputed cost {out.cost} != the file's Cost {reported_cost:g}")
    if expected_cost is not None and out.cost != expected_cost:
        out.errors.append(f"recomputed cost {out.cost} != the record's objective {expected_cost:g}")
    return out


def verify(vrp_path: Path, sol_path: Path, expected_cost: float | None = None) -> Verdict:
    """Read both files and check them. An unreadable `.sol` is a failed verdict,
    not an exception: the scorer refuses the row either way."""
    inst = read_vrp_file(vrp_path)
    try:
        sol = parse_sol(sol_path.read_text())
    except (OSError, ValueError) as exc:
        return Verdict(errors=[f"{sol_path}: {exc}"])
    return check_routes(inst, sol.routes, sol.cost, expected_cost)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("vrp", type=Path)
    parser.add_argument("sol", type=Path)
    parser.add_argument("--expected-cost", type=float, default=None)
    args = parser.parse_args(argv)
    verdict = verify(args.vrp, args.sol, args.expected_cost)
    if verdict.ok:
        print(f"OK cost={verdict.cost} routes={verdict.routes_used}")
        return 0
    for error in verdict.errors:
        print(f"FAIL {error}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
