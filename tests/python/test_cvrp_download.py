"""Tests for the CVRPLIB X-set snapshot (#196).

The BKS are the yardstick every CVRP gap is measured against, and CVRPLIB moves
them through its BKS challenge, so the snapshot is pinned. These tests pin the
parsers the pins rest on, the committed snapshot itself, and the refusals that
keep a moved file from silently replacing it.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pytest

from benchmarks.instances.cvrp.download import (
    DEV_FILENAME,
    DEV_PER_BAND,
    INSTANCE_SUBDIR,
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    ROSTER_FILENAME,
    SIZE_BANDS,
    SMOKE_FILENAME,
    SMOKE_INSTANCES,
    Snapshot,
    parse_listing,
    parse_sol,
    parse_vrp_header,
    read_manifest,
    read_roster,
    refusal,
    roster_entry,
    size_band,
    verify,
)

#: The committed snapshot.
CVRP_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "instances" / "cvrp"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_parse_sol_reads_a_trailing_cost() -> None:
    sol = parse_sol("Route #1: 1 2 3\nRoute #2: 4 5\nCost 27591\n")
    assert sol.cost == 27591
    assert sol.routes == [[1, 2, 3], [4, 5]]


def test_parse_sol_reads_a_mid_file_cost_with_trailer_lines() -> None:
    """X-n1001-k43's shape: `Cost` is not last, `NB_ROUTES` and `CPU_TIME` follow."""
    sol = parse_sol("Route #1: 7 8\nRoute #2: 9\nCost 72346\nNB_ROUTES: 2\nCPU_TIME: 1667.85\n")
    assert sol.cost == 72346
    assert len(sol.routes) == 2


@pytest.mark.parametrize(
    "text",
    [
        "Route #1: 1 2\n",  # no cost
        "Route #1: 1 2\nCost 5\nCost 6\n",  # two costs
        "Cost 5\n",  # no routes
    ],
)
def test_parse_sol_refuses_an_ambiguous_file(text: str) -> None:
    with pytest.raises(ValueError):
        parse_sol(text)


@pytest.mark.parametrize(
    ("name", "cost", "routes"),
    [("X-n101-k25", 27591, 26), ("X-n1001-k43", 72346, 43)],
)
def test_parse_sol_on_both_committed_shapes(name: str, cost: int, routes: int) -> None:
    """The two shapes #196 names, on the pinned files themselves. X-n101-k25's
    BKS uses 26 routes although its name says k25: the `k` is a minimum."""
    text = (CVRP_DIR / INSTANCE_SUBDIR / f"{name}.sol").read_text()
    sol = parse_sol(text)
    assert sol.cost == cost
    assert sum(1 for r in sol.routes if r) == routes
    if name == "X-n1001-k43":
        assert not text.rstrip().splitlines()[-1].startswith("Cost")


def _vrp(type_: str = "CVRP", weight: str = "EUC_2D") -> str:
    return (
        f"NAME : t\nTYPE : \t{type_}\t\nDIMENSION : 4\nEDGE_WEIGHT_TYPE : {weight}\n"
        "CAPACITY : 10\nNODE_COORD_SECTION\n1 0 0\n2 1 1\n3 2 2\n4 3 3\n"
    )


def test_parse_vrp_header_reads_customers_and_capacity() -> None:
    header = parse_vrp_header(_vrp())
    assert header.customers == 3  # DIMENSION counts the depot
    assert header.capacity == 10


@pytest.mark.parametrize(("type_", "weight"), [("TSP", "EUC_2D"), ("CVRP", "ATT"), ("CVRP", "")])
def test_parse_vrp_header_refuses_what_the_bks_metric_does_not_cover(
    type_: str, weight: str
) -> None:
    with pytest.raises(ValueError):
        parse_vrp_header(_vrp(type_, weight))


def test_parse_listing_reads_a_row_and_refuses_a_partial_one() -> None:
    row = (
        '<tr class="collapse border-bottom" id="set-17"><td>'
        '<a href="/cvrplib/index.php/en/download/instance/158" title="Instance File">'
        " X-n101-k25 </a></td>"
        '<td class="text-end">$100$</td><td class="text-end">$25$</td>'
        '<td class="text-end">$206$</td>'
        '<td class="text-end"><span class="pe-none">$27{,}591.00$</span></td>'
        '<td><span class="badge text-success-emphasis">yes</span></td></tr>'
    )
    other_set = row.replace('id="set-17"', 'id="set-3"')
    rows = parse_listing(row + other_set)
    assert len(rows) == 1
    assert rows[0].instance == "X-n101-k25"
    assert (rows[0].cvrplib_id, rows[0].customers, rows[0].k_min) == (158, 100, 25)
    assert (rows[0].capacity, rows[0].bks, rows[0].optimal) == (206, 27591.0, True)
    with pytest.raises(ValueError):
        parse_listing(row.replace("pe-none", "gone"))


# ---------------------------------------------------------------------------
# The committed snapshot
# ---------------------------------------------------------------------------


def test_the_committed_snapshot_verifies() -> None:
    """Every pin holds and every roster row is what its pinned `.sol` says."""
    assert verify(CVRP_DIR) == []


def test_roster_agrees_with_the_pinned_sol_files() -> None:
    """The issue's acceptance criterion, spelled out rather than left inside
    verify(): BKS and route count are read back from each `.sol`."""
    for entry in read_roster(CVRP_DIR / ROSTER_FILENAME):
        sol = parse_sol((CVRP_DIR / INSTANCE_SUBDIR / f"{entry.instance}.sol").read_text())
        assert roster_entry(entry.instance, sol, entry.bks_kind == "opt") == entry


def test_the_split_is_stratified_and_disjoint() -> None:
    full = {e.instance for e in read_roster(CVRP_DIR / ROSTER_FILENAME)}
    dev = {e.instance for e in read_roster(CVRP_DIR / DEV_FILENAME)}
    report = {e.instance for e in read_roster(CVRP_DIR / REPORT_FILENAME)}
    assert len(full) == 100
    assert dev.isdisjoint(report)
    assert dev | report == full
    customers = {n: int(r["n"]) for n, r in read_manifest(CVRP_DIR / MANIFEST_FILENAME).items()}
    for band in range(len(SIZE_BANDS)):
        assert sum(1 for n in dev if size_band(customers[n]) == band) == DEV_PER_BAND
    smoke = [e.instance for e in read_roster(CVRP_DIR / SMOKE_FILENAME)]
    assert smoke == list(SMOKE_INSTANCES)


@pytest.fixture
def snapshot_copy(tmp_path: Path) -> Path:
    dst = tmp_path / "cvrp"
    shutil.copytree(CVRP_DIR, dst, ignore=shutil.ignore_patterns("__pycache__", "*.py"))
    return dst


def test_a_tampered_instance_is_refused(snapshot_copy: Path) -> None:
    path = snapshot_copy / INSTANCE_SUBDIR / "X-n101-k25.vrp"
    path.write_bytes(path.read_bytes().replace(b"365\t689", b"366\t689"))
    problems = verify(snapshot_copy)
    assert any("X-n101-k25.vrp" in p for p in problems), problems


def test_a_tampered_solution_is_refused(snapshot_copy: Path) -> None:
    path = snapshot_copy / INSTANCE_SUBDIR / "X-n101-k25.sol"
    path.write_text(path.read_text().replace("Cost 27591", "Cost 27000"))
    problems = verify(snapshot_copy)
    assert any("X-n101-k25.sol" in p for p in problems), problems


def test_a_hand_edited_roster_is_refused(snapshot_copy: Path) -> None:
    path = snapshot_copy / ROSTER_FILENAME
    path.write_text(path.read_text().replace("X-n101-k25,27591,", "X-n101-k25,27500,"))
    problems = verify(snapshot_copy)
    assert any(ROSTER_FILENAME in p for p in problems), problems


# ---------------------------------------------------------------------------
# Refusing a moved download
# ---------------------------------------------------------------------------


def _committed_snapshot() -> Snapshot:
    """A download that returned exactly the committed files."""
    manifest = read_manifest(CVRP_DIR / MANIFEST_FILENAME)
    inst = CVRP_DIR / INSTANCE_SUBDIR
    return Snapshot(
        vrps={n: (inst / f"{n}.vrp").read_bytes() for n in manifest},
        sols={n: (inst / f"{n}.sol").read_bytes() for n in manifest},
        manifest=manifest,
        entries=read_roster(CVRP_DIR / ROSTER_FILENAME),
    )


def _args(*, references: bool = False, manifest: bool = False) -> argparse.Namespace:
    return argparse.Namespace(update_references=references, update_manifest=manifest)


def test_an_unchanged_download_is_accepted() -> None:
    assert refusal(_committed_snapshot(), CVRP_DIR, _args()) is None


def test_a_moved_bks_is_refused_without_the_flag() -> None:
    snap = _committed_snapshot()
    first = snap.entries[0]
    snap.entries[0] = first._replace(bks=first.bks - 1)
    reason = refusal(snap, CVRP_DIR, _args())
    assert reason is not None
    assert first.instance in reason
    assert refusal(snap, CVRP_DIR, _args(references=True)) is None


def test_a_reserialised_sol_is_a_moved_reference() -> None:
    """Same cost and routes, different bytes: still not the pinned file."""
    snap = _committed_snapshot()
    snap.sols["X-n101-k25"] += b"\n"
    reason = refusal(snap, CVRP_DIR, _args())
    assert reason is not None
    assert "X-n101-k25.sol" in reason


def test_moved_instance_bytes_are_refused_without_the_flag() -> None:
    snap = _committed_snapshot()
    snap.manifest["X-n101-k25"] = dict(snap.manifest["X-n101-k25"], sha256="0" * 64)
    reason = refusal(snap, CVRP_DIR, _args(references=True))
    assert reason is not None
    assert MANIFEST_FILENAME in reason
    assert refusal(snap, CVRP_DIR, _args(references=True, manifest=True)) is None
