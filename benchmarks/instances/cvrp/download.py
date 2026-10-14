"""Download CVRPLIB's X set and pin its best-known solutions (#196).

The X set (Uchoa, Pecin, Pessoa, Poggi, Subramanian and Vidal, EJOR 2017,
https://doi.org/10.1016/j.ejor.2016.08.012) is 100 CVRP instances with 100 to
1000 customers. CVRPLIB serves each instance's TSPLIB `.vrp` file and its
best-known solution (`.sol`). Those BKS move: CVRPLIB runs an open BKS challenge,
so the yardstick a gap is measured against is a snapshot, and this script pins
it the way `benchmarks/instances/mipfeas/download.py` pins MIPLIB's `.solu`:

* `manifest.csv` pins every `.vrp` by sha256 and byte count, beside the header
  fields a reader needs to sanity-check it (customers, capacity, minimum fleet).
* `references.csv` pins every `.sol`, every roster table derived from them, and
  the manifest itself (its `k_min` column comes from the listing, not a file).
* `--verify` re-checks all of it offline, and also re-derives each roster row
  from its `.sol` file, so a hand-edited roster or a swapped solution fails.
* A download whose instance bytes moved fails unless `--update-manifest`; one
  whose BKS moved fails unless `--update-references`, which prints the
  per-instance changes first. Nothing is written on a refusal.

The instance and solution files are committed (`X/`, about 1.1 MB), so a
checkout can verify, test and run without the network; this script is how they
were obtained and how a later snapshot would be taken.

Usage:
    python download.py                      # fetch, check against the pins
    python download.py --verify             # offline check of every pin
    python download.py --update-references  # accept moved BKS (prints the diff)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import re
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

# Run as a script (`python download.py`, the documented form), only this file's
# own directory is on sys.path; the repository root makes `benchmarks.common`
# importable.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from benchmarks.common.pins import (  # noqa: E402
    pin_mismatch,
    read_pin_table,
    sha256_of,
    write_pin_table,
)

#: CVRPLIB moved here in 2025; the old `vrp.galgos.inf.puc-rio.br/media/...`
#: paths return 404. The listing page carries every set's instance ids, the
#: published BKS and CVRPLIB's own optimality marking.
BASE_URL = "https://galgos.inf.puc-rio.br/cvrplib/index.php/en/"
INSTANCE_URL = BASE_URL + "download/instance/{id}"
SOLUTION_URL = BASE_URL + "download/bks/{id}"
#: The X set's id on the listing page.
X_SET_ID = 17
USER_AGENT = "cbls-cvrp/0.1"

INSTANCE_SUBDIR = "X"
MANIFEST_FILENAME = "manifest.csv"
REFERENCES_FILENAME = "references.csv"
ROSTER_FILENAME = "roster.csv"
SMOKE_FILENAME = "smoke.csv"
DEV_FILENAME = "dev.csv"
REPORT_FILENAME = "report.csv"
ROSTER_TABLES: tuple[str, ...] = (ROSTER_FILENAME, SMOKE_FILENAME, DEV_FILENAME, REPORT_FILENAME)
#: Pinned in references.csv beside the `.sol` files. The manifest's hash and
#: header columns are re-derived from each `.vrp`, but `k_min` comes from
#: CVRPLIB's listing alone, so only a pin on the table itself protects it.
PINNED_TABLES: tuple[str, ...] = (MANIFEST_FILENAME, *ROSTER_TABLES)

EXPECTED_SET_SIZE = 100

#: One instance per size band, spanning the set: the runner check of #197.
SMOKE_INSTANCES: tuple[str, ...] = (
    "X-n101-k25",
    "X-n251-k28",
    "X-n502-k39",
    "X-n801-k40",
    "X-n1001-k43",
)

#: The held-out split (see README.md, "Dev and report rosters"). Fixed before
#: any membership was computed, and before any CVRP result existed.
HELDOUT_SEED = 196
#: Hexaly's size bands, by customer count: [lo, hi). The last band is closed.
SIZE_BANDS: tuple[tuple[int, int], ...] = (
    (100, 200),
    (200, 400),
    (400, 600),
    (600, 800),
    (800, 1001),
)
DEV_PER_BAND = 4


class ListingRow(NamedTuple):
    """One X instance as CVRPLIB's listing page states it."""

    instance: str
    cvrplib_id: int
    customers: int
    k_min: int
    capacity: int
    bks: float
    optimal: bool


class VrpHeader(NamedTuple):
    customers: int
    capacity: int


class Solution(NamedTuple):
    cost: float
    routes: list[list[int]]


class RosterEntry(NamedTuple):
    instance: str
    bks: float
    #: `opt` when CVRPLIB marks the BKS proven optimal, `ub` otherwise.
    bks_kind: str
    #: Non-empty routes in the BKS. The `k` in an X name is the MINIMUM fleet,
    #: and the BKS may use more (X-n101-k25's uses 26).
    bks_routes: int


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

_SOL_COST = re.compile(r"^\s*Cost\s+([0-9]+(?:\.[0-9]+)?)\s*$")
_SOL_ROUTE = re.compile(r"^\s*Route\s*#\s*\d+\s*:(.*)$")


def parse_sol(text: str) -> Solution:
    """A CVRPLIB `.sol`: `Route #i: c c c` lines and one `Cost N` line.

    The format is not uniform. The `Cost` line is usually last, but some files
    put it mid-file and end with e.g. `CPU_TIME:`, so it is searched for rather
    than read from the end. Exactly one is required. Customer ids are as CVRPLIB
    writes them: 1-based, the depot being node 0 / TSPLIB node 1.
    """
    costs: list[float] = []
    routes: list[list[int]] = []
    for line in text.splitlines():
        m = _SOL_ROUTE.match(line)
        if m is not None:
            routes.append([int(tok) for tok in m.group(1).split()])
            continue
        m = _SOL_COST.match(line)
        if m is not None:
            costs.append(float(m.group(1)))
    if len(costs) != 1:
        raise ValueError(f"expected exactly one 'Cost' line, found {len(costs)}")
    if not routes:
        raise ValueError("no 'Route #' lines")
    return Solution(costs[0], routes)


def parse_vrp_header(text: str) -> VrpHeader:
    """The fields of a TSPLIB `.vrp` header that the roster records.

    Only what the X set uses is accepted: `TYPE: CVRP` and
    `EDGE_WEIGHT_TYPE: EUC_2D`. Anything else is refused rather than guessed,
    because the BKS costs are in that one metric.
    """
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().upper()
        if key.endswith("_SECTION"):
            break
        fields[key] = value.strip()
    if fields.get("TYPE") != "CVRP":
        raise ValueError(f"TYPE is {fields.get('TYPE')!r}, expected 'CVRP'")
    if fields.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
        raise ValueError(
            f"EDGE_WEIGHT_TYPE is {fields.get('EDGE_WEIGHT_TYPE')!r}, expected 'EUC_2D'"
        )
    return VrpHeader(customers=int(fields["DIMENSION"]) - 1, capacity=int(fields["CAPACITY"]))


_ROW = re.compile(
    r'<tr class="collapse border-bottom" id="set-(?P<set>\d+)">(?P<body>.*?)</tr>', re.S
)
_ROW_INSTANCE = re.compile(r'download/instance/(\d+)" title="Instance File">\s*(\S+)\s*</a>')
_ROW_NUMBER = re.compile(r'<td class="text-end">\s*\$([0-9{},.]+)\$\s*</td>')
_ROW_BKS = re.compile(r'<span class="pe-none">\$([0-9{},.]+)\$</span>')
_ROW_OPTIMAL = re.compile(r'text-(?:success|danger)-emphasis">(yes|no)<')


def _latex_number(text: str) -> float:
    return float(text.replace("{,}", "").replace(",", ""))


def parse_listing(page: str, set_id: int = X_SET_ID) -> list[ListingRow]:
    """The rows of one set on CVRPLIB's listing page.

    Every field is required; a row the patterns do not fully match fails the
    parse, so a redesign of the page stops the download instead of yielding a
    roster with silently missing columns.
    """
    rows: list[ListingRow] = []
    for m in _ROW.finditer(page):
        if int(m.group("set")) != set_id:
            continue
        body = m.group("body")
        inst = _ROW_INSTANCE.search(body)
        numbers = _ROW_NUMBER.findall(body)
        bks = _ROW_BKS.search(body)
        optimal = _ROW_OPTIMAL.search(body)
        if inst is None or len(numbers) < 3 or bks is None or optimal is None:
            raise ValueError(f"unparseable listing row in set {set_id}: {body[:200]!r}")
        rows.append(
            ListingRow(
                instance=html.unescape(inst.group(2)),
                cvrplib_id=int(inst.group(1)),
                customers=int(_latex_number(numbers[0])),
                k_min=int(_latex_number(numbers[1])),
                capacity=int(_latex_number(numbers[2])),
                bks=_latex_number(bks.group(1)),
                optimal=optimal.group(1) == "yes",
            )
        )
    return rows


# --------------------------------------------------------------------------
# Rosters
# --------------------------------------------------------------------------


def roster_entry(instance: str, sol: Solution, optimal: bool) -> RosterEntry:
    return RosterEntry(
        instance=instance,
        bks=sol.cost,
        bks_kind="opt" if optimal else "ub",
        bks_routes=sum(1 for r in sol.routes if r),
    )


def size_band(customers: int) -> int:
    for i, (lo, hi) in enumerate(SIZE_BANDS):
        if lo <= customers < hi:
            return i
    raise ValueError(f"{customers} customers is outside every size band")


def draw_key(instance: str) -> str:
    """The seeded order of the dev draw. sha256 of a fixed string, not
    `random.shuffle`, whose output Python does not promise across versions."""
    return hashlib.sha256(f"cbls-cvrp-dev:{HELDOUT_SEED}:{instance}".encode()).hexdigest()


def draw_dev(customers: dict[str, int]) -> list[str]:
    """`DEV_PER_BAND` instances from each size band, in seeded hash order."""
    dev: list[str] = []
    for band in range(len(SIZE_BANDS)):
        members = sorted(
            (name for name, c in customers.items() if size_band(c) == band), key=draw_key
        )
        if len(members) < DEV_PER_BAND:
            raise ValueError(f"size band {SIZE_BANDS[band]} has only {len(members)} instances")
        dev.extend(members[:DEV_PER_BAND])
    return sorted(dev, key=_name_order)


def _name_order(instance: str) -> tuple[int, str]:
    m = re.fullmatch(r"X-n(\d+)-k\d+", instance)
    return (int(m.group(1)) if m else 0, instance)


def write_roster(path: Path, entries: list[RosterEntry]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(RosterEntry._fields)
        for e in sorted(entries, key=lambda e: _name_order(e.instance)):
            writer.writerow([e.instance, f"{e.bks:.15g}", e.bks_kind, e.bks_routes])
    tmp.replace(path)


def read_roster(path: Path) -> list[RosterEntry]:
    with open(path, newline="") as fh:
        return [
            RosterEntry(r["instance"], float(r["bks"]), r["bks_kind"], int(r["bks_routes"]))
            for r in csv.DictReader(fh)
        ]


def roster_tables(
    full: list[RosterEntry], customers: dict[str, int]
) -> dict[str, list[RosterEntry]]:
    """Every roster table, derived from the full roster and the customer counts."""
    by_name = {e.instance: e for e in full}
    missing = [n for n in SMOKE_INSTANCES if n not in by_name]
    if missing:
        raise ValueError(f"smoke instances not in the set: {missing}")
    dev = set(draw_dev(customers))
    return {
        ROSTER_FILENAME: full,
        SMOKE_FILENAME: [by_name[n] for n in SMOKE_INSTANCES],
        DEV_FILENAME: [e for e in full if e.instance in dev],
        REPORT_FILENAME: [e for e in full if e.instance not in dev],
    }


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

MANIFEST_FIELDS = ("instance", "n", "capacity", "k_min", "sha256", "bytes")


def manifest_row(instance: str, vrp: bytes, k_min: int) -> dict[str, str]:
    """`k_min` is CVRPLIB's listed minimum fleet, which the `.vrp` does not carry.
    It is NOT always the `k` in the name: CVRPLIB lists X-n247-k50, X-n524-k153
    and X-n670-k130 with 47, 137 and 126 (see README.md)."""
    header = parse_vrp_header(vrp.decode())
    return {
        "instance": instance,
        "n": str(header.customers),
        "capacity": str(header.capacity),
        "k_min": str(k_min),
        "sha256": sha256_of(vrp),
        "bytes": str(len(vrp)),
    }


def read_manifest(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with open(path, newline="") as fh:
        return {r["instance"]: r for r in csv.DictReader(fh)}


def write_manifest(path: Path, rows: dict[str, dict[str, str]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        for name in sorted(rows, key=_name_order):
            writer.writerow(rows[name])
    tmp.replace(path)


# --------------------------------------------------------------------------
# Offline verification
# --------------------------------------------------------------------------


def verify_pins(here: Path, manifest: dict[str, dict[str, str]]) -> list[str]:
    """Every `.vrp` against the manifest, every `.sol` and roster table against
    `references.csv`, and no pin left over that nothing reads."""
    problems: list[str] = []
    if len(manifest) != EXPECTED_SET_SIZE:
        problems.append(f"{MANIFEST_FILENAME}: {len(manifest)} rows, expected {EXPECTED_SET_SIZE}")
    for name, row in manifest.items():
        path = here / INSTANCE_SUBDIR / f"{name}.vrp"
        if not path.exists():
            problems.append(f"{name}.vrp: pinned but absent")
            continue
        data = path.read_bytes()
        complaint = pin_mismatch(f"{name}.vrp", data, (row["sha256"], int(row["bytes"])))
        if complaint is None and manifest_row(name, data, int(row["k_min"])) != row:
            complaint = f"{MANIFEST_FILENAME}: header fields of {name} disagree with the file"
        if complaint is not None:
            problems.append(complaint)

    refs = read_pin_table(here / REFERENCES_FILENAME, "file")
    pinned_files = [f"{INSTANCE_SUBDIR}/{n}.sol" for n in manifest] + list(PINNED_TABLES)
    for rel in pinned_files:
        path = here / rel
        complaint = (
            f"{rel}: pinned but absent"
            if not path.exists()
            else pin_mismatch(rel, path.read_bytes(), refs.get(rel))
        )
        if complaint is not None:
            problems.append(complaint)
    extra = sorted(set(refs) - set(pinned_files))
    problems.extend(f"{REFERENCES_FILENAME}: pins {rel}, which nothing reads" for rel in extra)
    return problems


def verify_derivations(here: Path, manifest: dict[str, dict[str, str]]) -> list[str]:
    """The roster's BKS and route count are what each pinned `.sol` says, and
    the subset tables are what the draw says."""
    full = read_roster(here / ROSTER_FILENAME)
    if sorted(e.instance for e in full) != sorted(manifest):
        return [f"{ROSTER_FILENAME}: instances differ from {MANIFEST_FILENAME}"]
    problems: list[str] = []
    for entry in full:
        sol = parse_sol((here / INSTANCE_SUBDIR / f"{entry.instance}.sol").read_text())
        derived = roster_entry(entry.instance, sol, entry.bks_kind == "opt")
        if derived != entry:
            problems.append(f"{ROSTER_FILENAME}: {entry} disagrees with its .sol ({derived})")
    customers = {n: int(r["n"]) for n, r in manifest.items()}
    for table, entries in roster_tables(full, customers).items():
        if table != ROSTER_FILENAME and read_roster(here / table) != entries:
            problems.append(f"{table}: does not match its derivation from {ROSTER_FILENAME}")
    return problems


def verify(here: Path) -> list[str]:
    """Every pin, then every derivation, checked against the files on disk. An
    empty list means clean."""
    manifest = read_manifest(here / MANIFEST_FILENAME)
    return verify_pins(here, manifest) or verify_derivations(here, manifest)


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        data: bytes = resp.read()
    return data


def _looks_like_html(data: bytes) -> bool:
    head = data[:512].lstrip().lower()
    return head.startswith((b"<!doctype", b"<html"))


class Snapshot(NamedTuple):
    """One download: the files, and what the page and the files say together."""

    vrps: dict[str, bytes]
    sols: dict[str, bytes]
    manifest: dict[str, dict[str, str]]
    entries: list[RosterEntry]


def download_snapshot() -> Snapshot | str:
    """Fetch the listing and every file, and cross-check them. The page and the
    files are one source, so a disagreement means one of them is not what it
    claims to be; the reason is returned instead of a snapshot."""
    print(f"[fetch] {BASE_URL}")
    listing = parse_listing(_http_get(BASE_URL).decode())
    if len(listing) != EXPECTED_SET_SIZE:
        return f"listing has {len(listing)} X instances, expected {EXPECTED_SET_SIZE}"
    snap = Snapshot({}, {}, {}, [])
    problems: list[str] = []
    for row in listing:
        vrp = _http_get(INSTANCE_URL.format(id=row.cvrplib_id))
        sol_bytes = _http_get(SOLUTION_URL.format(id=row.cvrplib_id))
        if _looks_like_html(vrp) or _looks_like_html(sol_bytes):
            return f"{row.instance}: the server returned an HTML page, not a file"
        mrow = manifest_row(row.instance, vrp, row.k_min)
        if (int(mrow["n"]), int(mrow["capacity"])) != (row.customers, row.capacity):
            problems.append(f"{row.instance}: .vrp header disagrees with the listing")
        sol = parse_sol(sol_bytes.decode())
        if sol.cost != row.bks:
            problems.append(f"{row.instance}: .sol cost {sol.cost:g} != listed BKS {row.bks:g}")
        snap.vrps[row.instance] = vrp
        snap.sols[row.instance] = sol_bytes
        snap.manifest[row.instance] = mrow
        snap.entries.append(roster_entry(row.instance, sol, row.optimal))
    return "\n".join(problems) if problems else snap


def refusal(snap: Snapshot, here: Path, args: argparse.Namespace) -> str | None:
    """Why this snapshot may not replace the pinned one, or None.

    A first acquisition of an instance is not an overwrite and needs no flag;
    the first snapshot of the BKS does, since it is the yardstick.
    """
    old_manifest = read_manifest(here / MANIFEST_FILENAME)
    moved = [n for n, r in snap.manifest.items() if n in old_manifest and old_manifest[n] != r]
    if moved and not args.update_manifest:
        return (
            f"{len(moved)} instance file(s) no longer match {MANIFEST_FILENAME}: {moved[:5]}. "
            "Re-run with --update-manifest to accept them."
        )
    roster_path = here / ROSTER_FILENAME
    if not roster_path.exists():
        return (
            None
            if args.update_references
            else f"{ROSTER_FILENAME} is absent; the first snapshot needs --update-references."
        )
    old = {e.instance: e for e in read_roster(roster_path)}
    changes = [f"  {old[e.instance]} -> {e}" for e in snap.entries if old.get(e.instance) != e]
    # A `.sol` whose bytes moved with the same cost and routes (a re-serialised
    # file, a new trailer line) is still a moved reference file.
    old_refs = read_pin_table(here / REFERENCES_FILENAME, "file")
    changes += [
        f"  {n}.sol: bytes changed"
        for n, data in snap.sols.items()
        if old_refs.get(f"{INSTANCE_SUBDIR}/{n}.sol") != (sha256_of(data), len(data))
    ]
    if changes and not args.update_references:
        return (
            "the BKS this benchmark is scored against have moved:\n"
            + "\n".join(changes)
            + "\nRe-run with --update-references to accept them as a reviewable diff."
        )
    if changes:
        print("[update] BKS changes:\n" + "\n".join(changes))
    return None


def write_snapshot(snap: Snapshot, here: Path) -> None:
    inst_dir = here / INSTANCE_SUBDIR
    inst_dir.mkdir(exist_ok=True)
    refs: dict[str, tuple[str, int]] = {}
    for name in snap.manifest:
        (inst_dir / f"{name}.vrp").write_bytes(snap.vrps[name])
        (inst_dir / f"{name}.sol").write_bytes(snap.sols[name])
        refs[f"{INSTANCE_SUBDIR}/{name}.sol"] = (sha256_of(snap.sols[name]), len(snap.sols[name]))
    write_manifest(here / MANIFEST_FILENAME, snap.manifest)
    customers = {n: int(r["n"]) for n, r in snap.manifest.items()}
    for table, rows in roster_tables(snap.entries, customers).items():
        write_roster(here / table, rows)
    for table in PINNED_TABLES:
        data = (here / table).read_bytes()
        refs[table] = (sha256_of(data), len(data))
    write_pin_table(here / REFERENCES_FILENAME, "file", refs)


def fetch(args: argparse.Namespace, here: Path) -> int:
    snap = download_snapshot()
    if isinstance(snap, str):
        print(f"FAILED: {snap}", file=sys.stderr)
        return 2
    reason = refusal(snap, here, args)
    if reason is not None:
        print(f"FAILED: {reason}", file=sys.stderr)
        return 3
    write_snapshot(snap, here)
    problems = verify(here)
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    n_opt = sum(1 for e in snap.entries if e.bks_kind == "opt")
    print(f"Done. {len(snap.entries)} instances ({n_opt} with proven-optimal BKS), pins verified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--verify", action="store_true", help="check every pin and derivation offline, and exit"
    )
    parser.add_argument(
        "--update-references",
        action="store_true",
        help="accept moved BKS and re-pin them (prints the per-instance changes first)",
    )
    parser.add_argument(
        "--update-manifest",
        action="store_true",
        help="accept instance files that no longer match manifest.csv",
    )
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    if args.verify:
        problems = verify(here)
        for p in problems:
            print(p, file=sys.stderr)
        if not problems:
            print("All pins and roster derivations verified.")
        return 1 if problems else 0
    return fetch(args, here)


if __name__ == "__main__":
    sys.exit(main())
