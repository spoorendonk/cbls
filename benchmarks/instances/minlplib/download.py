"""Download and select a non-convex MINLPLib subset for the CBLS benchmark.

Pipeline:
  1. Fetch the master metadata CSV (``instancedata.csv``, semicolon-separated).
  2. Filter to non-convex instances whose nonzero operator columns are a subset
     of the operator set CBLS can express today, with a size budget and a finite
     primal bound.
  3. Stratify the survivors across structure types into a candidate order.
  4. Walk that order fetching each instance's text ``.nl`` file (validating the
     ``g3`` header, rejecting HTML/404 bodies, printing a sha256) until
     ``--limit`` instances have been fetched successfully, so an instance served
     as binary NL is replaced rather than shrinking the roster.
  5. Write ``bounds.csv`` from the *fetched* set, so every row has a .nl on disk.

Network access is required only for steps 1 and 4; everything is logged so a
failed fetch is visible. Run with the project venv:

    .venv/bin/python3 benchmarks/instances/minlplib/download.py
    .venv/bin/python3 benchmarks/instances/minlplib/download.py --force
    .venv/bin/python3 benchmarks/instances/minlplib/download.py --limit 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import sys
import urllib.error
import urllib.request
from pathlib import Path

CSV_URL = "https://www.minlplib.org/instancedata.csv"
NL_URL_TEMPLATE = "https://www.minlplib.org/nl/{name}.nl"

# Operator columns CBLS can express today. An instance is selectable only if all
# of its remaining (nonzero) operator columns fall in this set. In the NL text
# format these are realised by standard opcodes (e.g. signpower/rpower appear as
# OPPOW); the SignPower DAG op added in #72 is available for direct model
# building. optanh maps onto Tanh; the rest onto existing DAG ops (see
# src/io/nl_to_model.cpp).
SUPPORTED_OP_COLUMNS: frozenset[str] = frozenset(
    {
        "opabs",
        "opcos",
        "opdiv",
        "opexp",
        "oplog",
        "oplog10",
        "opmin",
        "opmul",
        "oppower",
        "opsin",
        "opsqr",
        "opsqrt",
        "opsignpower",
        "oprpower",
        "optanh",
    }
)

# Operator columns CBLS cannot express; an instance using any of these is skipped
# at selection time. Listed explicitly for the README provenance table.
UNSUPPORTED_OP_COLUMNS: frozenset[str] = frozenset(
    {
        "opcentropy",
        "opcvpower",
        "operrorf",
        "opgamma",
        "opmod",
        "opvcpower",
    }
)

ALL_OP_COLUMNS: frozenset[str] = SUPPORTED_OP_COLUMNS | UNSUPPORTED_OP_COLUMNS

# Problem types we accept (continuous + mixed-integer nonlinear / quadratic).
ACCEPTED_PROBTYPES: frozenset[str] = frozenset(
    {"NLP", "MINLP", "QCP", "QCQP", "QP", "MIQCP", "MIQCQP", "MIQP", "BQP", "BQCP"}
)

# Size budget (variables and constraints).
MAX_VARS = 150
MAX_CONS = 150

# Target roster size after stratification. Must match the published roster:
# bounds.csv is written from the fetched set, so a smaller default would silently
# shrink the roster the runner reads while the extra .nl files sit unused on disk.
DEFAULT_ROSTER = 50


def _is_true(cell: str) -> bool:
    return cell.strip().lower() == "true"


def _to_int(cell: str) -> int | None:
    try:
        return int(float(cell))
    except (ValueError, TypeError):
        return None


def _to_float(cell: str) -> float | None:
    try:
        return float(cell)
    except (ValueError, TypeError):
        return None


def fetch_bytes(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cbls-minlplib/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data: bytes = resp.read()
    return data


def _looks_like_html(data: bytes) -> bool:
    head = data[:128].lstrip().lower()
    return head.startswith(b"<!doctype") or head.startswith(b"<html")


class Instance:
    """A selected instance with its published bounds and structure tag."""

    def __init__(
        self,
        name: str,
        nvars: int,
        ncons: int,
        primalbound: float | None,
        dualbound: float | None,
        objsense: str,
        structure: str,
        ndiscvars: int,
    ) -> None:
        self.name = name
        self.nvars = nvars
        self.ncons = ncons
        self.primalbound = primalbound
        self.dualbound = dualbound
        self.objsense = objsense
        self.structure = structure
        # Catalogue ground truth (nbinvars + nintvars). The runner cross-checks
        # the NL reader's recovered integrality against this: the NL header gives
        # integer *counts* per category and Gay's variable ordering gives their
        # positions, so an off-by-one in that mapping shows up as a mismatch here.
        self.ndiscvars = ndiscvars


def classify_structure(row: dict[str, str]) -> str:
    """Coarse structure tag used for stratified sampling and the README."""
    transcendental = any(
        _is_true(row.get(c, "")) for c in ("opexp", "oplog", "oplog10", "opsin", "opcos", "optanh")
    )
    has_int = (_to_int(row.get("nintvars", "0")) or 0) > 0 or (
        _to_int(row.get("nbinvars", "0")) or 0
    ) > 0
    polynomial = any(
        _is_true(row.get(c, "")) for c in ("oppower", "opsignpower", "oprpower", "opsqr")
    )
    bilinear = _is_true(row.get("opmul", ""))

    if has_int:
        return "mixed-integer"
    if transcendental:
        return "transcendental"
    if polynomial:
        return "polynomial"
    if bilinear:
        return "bilinear"
    return "other"


def row_to_instance(row: dict[str, str]) -> Instance | None:
    """Apply the per-row filter; return an Instance if it qualifies, else None."""
    if row.get("convex", "").strip() != "False":
        return None
    if row.get("probtype", "").strip() not in ACCEPTED_PROBTYPES:
        return None
    # The instance must offer the text NL format.
    if "nl" not in row.get("formats", ""):
        return None
    # Operator subset check: reject if any unsupported op column is True.
    if any(_is_true(row.get(c, "")) for c in UNSUPPORTED_OP_COLUMNS):
        return None

    nvars = _to_int(row.get("nvars", ""))
    ncons = _to_int(row.get("ncons", ""))
    if nvars is None or ncons is None:
        return None
    if nvars > MAX_VARS or ncons > MAX_CONS:
        return None

    primal = _to_float(row.get("primalbound", ""))
    if primal is None or not math.isfinite(primal):
        return None  # need a *finite* primal BKS for gap reporting ("inf" parses)

    return Instance(
        name=row["name"].strip(),
        nvars=nvars,
        ncons=ncons,
        primalbound=primal,
        dualbound=_to_float(row.get("dualbound", "")),
        objsense=row.get("objsense", "").strip(),
        structure=classify_structure(row),
        ndiscvars=(_to_int(row.get("nbinvars", "0")) or 0)
        + (_to_int(row.get("nintvars", "0")) or 0),
    )


def select(rows: list[dict[str, str]]) -> list[Instance]:
    """Apply the CSV filter and return every survivor in stratified order.

    The caller walks this order and stops once enough instances have been
    *fetched successfully*, so that instances the catalogue advertises as ``nl``
    but serves as binary NL (the ``kriging_peaks-*`` family) are replaced rather
    than silently shrinking the roster.
    """
    survivors: list[Instance] = []
    for row in rows:
        inst = row_to_instance(row)
        if inst is not None:
            survivors.append(inst)

    # Stratify: round-robin across structure classes, smallest first within each.
    by_structure: dict[str, list[Instance]] = {}
    for inst in survivors:
        by_structure.setdefault(inst.structure, []).append(inst)
    for bucket in by_structure.values():
        bucket.sort(key=lambda i: (i.nvars + i.ncons, i.name))

    roster_list: list[Instance] = []
    order = sorted(by_structure.keys())
    idx = 0
    while any(by_structure.values()):
        cls = order[idx % len(order)]
        bucket = by_structure.get(cls, [])
        if bucket:
            roster_list.append(bucket.pop(0))
        idx += 1
    return roster_list


def write_bounds(path: Path, instances: list[Instance]) -> None:
    """Write bounds.csv for `instances`.

    Schema is the original seven columns plus a trailing `n_disc_vars_bks`
    (appended, so positional readers of columns 0-6 are unaffected).
    """
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "instance",
                "structure",
                "nvars",
                "ncons",
                "objsense",
                "primal_bks",
                "dual_bound",
                "n_disc_vars_bks",
            ]
        )
        for inst in instances:
            writer.writerow(
                [
                    inst.name,
                    inst.structure,
                    inst.nvars,
                    inst.ncons,
                    inst.objsense,
                    inst.primalbound,
                    inst.dualbound,
                    inst.ndiscvars,
                ]
            )


def parse_csv(data: bytes) -> list[dict[str, str]]:
    text = data.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text), delimiter=";")
    return list(reader)


def validate_nl(data: bytes) -> str | None:
    """Return an error string if `data` is not a valid text NL file, else None."""
    if _looks_like_html(data):
        return "server returned HTML (likely 404)"
    head = data[:64].lstrip()
    if not head.startswith(b"g"):
        return f"not a text NL file (header: {head[:8]!r})"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download even if the file exists")
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_ROSTER,
        help=f"roster size after stratification (default {DEFAULT_ROSTER})",
    )
    parser.add_argument(
        "--select-only",
        action="store_true",
        help="print the selected roster and exit without fetching .nl files",
    )
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    print("=== MINLPLib download ===")
    print(f"target dir: {here}")

    print(f"[fetch] {CSV_URL}")
    try:
        csv_bytes = fetch_bytes(CSV_URL)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[fail]  could not fetch metadata CSV: {exc}")
        return 1
    rows = parse_csv(csv_bytes)
    print(f"        {len(rows)} instances in metadata")

    candidates = select(rows)
    print(
        f"\n{len(candidates)} instances pass the filter (budget nvars<={MAX_VARS}, "
        f"ncons<={MAX_CONS}, non-convex, supported ops); "
        f"target roster {args.limit}."
    )

    bounds_path = here / "bounds.csv"

    if args.select_only:
        roster = candidates[: args.limit]
        for inst in roster:
            print(
                f"  {inst.name:30s} {inst.structure:14s} "
                f"nvars={inst.nvars:4d} ncons={inst.ncons:4d} "
                f"primal={inst.primalbound}"
            )
        # Deliberately does NOT write bounds.csv: this is the pre-fetch roster and
        # still contains instances the catalogue advertises as `nl` but serves as
        # binary NL. Writing it would desynchronise bounds.csv from the .nl files
        # on disk, and the runner would report those rows as not-found.
        print(f"\n(selection only — no .nl fetched, {bounds_path.name} left unchanged)")
        return 0

    # Walk the stratified order, fetching until `limit` instances are in hand.
    # bounds.csv is written from the fetched set only, so the roster the runner
    # reads is exactly the set of .nl files on disk.
    fetched: list[Instance] = []
    fail = 0
    for inst in candidates:
        if len(fetched) >= args.limit:
            break
        dest = here / f"{inst.name}.nl"
        if dest.exists() and not args.force and dest.stat().st_size > 0:
            print(f"[skip]  {dest.name} (exists, {dest.stat().st_size} bytes)")
            fetched.append(inst)
            continue
        url = NL_URL_TEMPLATE.format(name=inst.name)
        print(f"[fetch] {url}")
        try:
            data = fetch_bytes(url)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[fail]  {url}: {exc}")
            fail += 1
            continue
        err = validate_nl(data)
        if err is not None:
            print(f"[fail]  {url}: {err}")
            fail += 1
            continue
        dest.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        print(
            f"        -> {dest.name} ({len(data)} bytes, sha256 {digest[:12]}..., "
            f"{inst.structure}, {inst.ndiscvars} int vars)"
        )
        fetched.append(inst)

    write_bounds(bounds_path, fetched)
    by_class: dict[str, int] = {}
    for inst in fetched:
        by_class[inst.structure] = by_class.get(inst.structure, 0) + 1
    n_mip = sum(1 for inst in fetched if inst.ndiscvars > 0)

    print(f"\nWrote {bounds_path.name} ({len(fetched)} fetched instances)")
    print(f"Done. fetched={len(fetched)} fail={fail} (target {args.limit}).")
    print(f"  structure mix: {dict(sorted(by_class.items()))}")
    print(f"  mixed-integer: {n_mip} of {len(fetched)}")
    # A short roster is the only real failure: individual fetch failures are
    # expected (binary-NL instances) and are replaced from the candidate pool.
    return 0 if len(fetched) >= min(args.limit, len(candidates)) else 1


if __name__ == "__main__":
    sys.exit(main())
