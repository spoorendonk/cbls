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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

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

# The held-out roster (#144; HELDOUT.md). The seed was fixed before the draw was
# looked at and is committed with the membership it produced; changing it
# re-draws the set, which is exactly what committing it exists to prevent. There
# is deliberately no flag for it.
HELDOUT_SEED = 144
HELDOUT_DIRNAME = "heldout"
# The catalogue rows the split was drawn from: every filter survivor, restricted
# to the columns the filter and classifier read. With it the published roster
# and the held-out membership are both re-derivable offline (and are, by
# tests/python/test_minlplib_heldout.py).
POOL_FILENAME = "pool.csv"
# Instances a fetch walk skipped because MINLPLib served them as something other
# than text NL, per walk. Needed to replay a walk offline: `walk()` drops these.
UNFETCHABLE_FILENAME = "unfetchable.csv"
POOL_COLUMNS: tuple[str, ...] = (
    "name",
    "probtype",
    "convex",
    "formats",
    "nvars",
    "ncons",
    "nbinvars",
    "nintvars",
    "objsense",
    "primalbound",
    "dualbound",
    *sorted(ALL_OP_COLUMNS),
)


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


def survivors(rows: list[dict[str, str]]) -> list[Instance]:
    """Every catalogue row the per-row filter admits: the candidate pool."""
    pool: list[Instance] = []
    for row in rows:
        inst = row_to_instance(row)
        if inst is not None:
            pool.append(inst)
    return pool


def _round_robin(
    pool: list[Instance], within: Callable[[Instance], tuple[int | str, ...]]
) -> list[Instance]:
    """Round-robin across structure classes (sorted by name), `within` order inside each."""
    by_structure: dict[str, list[Instance]] = {}
    for inst in pool:
        by_structure.setdefault(inst.structure, []).append(inst)
    for bucket in by_structure.values():
        bucket.sort(key=within)

    ordered: list[Instance] = []
    order = sorted(by_structure.keys())
    idx = 0
    while any(by_structure.values()):
        cls = order[idx % len(order)]
        bucket = by_structure.get(cls, [])
        if bucket:
            ordered.append(bucket.pop(0))
        idx += 1
    return ordered


def select(rows: list[dict[str, str]]) -> list[Instance]:
    """Apply the CSV filter and return every survivor in stratified order.

    The caller walks this order and stops once enough instances have been
    *fetched successfully*, so that instances the catalogue advertises as ``nl``
    but serves as binary NL (the ``kriging_peaks-*`` family) are replaced rather
    than silently shrinking the roster.

    This order defines the published roster: bounds.csv is its first
    DEFAULT_ROSTER fetchable entries, and tests/python/test_minlplib_heldout.py
    pins that against a committed snapshot of the pool. Do not change it.
    """
    # Smallest first within each class.
    return _round_robin(survivors(rows), lambda i: (i.nvars + i.ncons, i.name))


def heldout_key(seed: int, name: str) -> str:
    """The held-out draw's order key: a seeded hash of the instance name.

    A hash rather than ``random.Random(seed).shuffle``, because Python guarantees
    only ``random()`` and seeding to be stable across versions, not ``shuffle``;
    sha256 of a fixed string is stable everywhere, so the committed membership
    can be re-derived by any interpreter. The key ignores size entirely: that is
    the point of the draw (HELDOUT.md, "Why not the next fifty").
    """
    return hashlib.sha256(f"cbls-minlplib-heldout:{seed}:{name}".encode()).hexdigest()


def select_heldout(rows: list[dict[str, str]], exclude: set[str], seed: int) -> list[Instance]:
    """The held-out candidate order: the pool minus `exclude`, seeded-shuffled.

    Same round-robin across structure classes as `select`, so the held-out set is
    stratified by the same rule as the published roster; the only change is that
    within a class the order is `heldout_key` instead of smallest first.
    `exclude` is the published roster -- the instances the shipped defaults were
    fitted on.
    """
    pool = [inst for inst in survivors(rows) if inst.name not in exclude]
    return _round_robin(pool, lambda i: (heldout_key(seed, i.name),))


def walk(candidates: list[Instance], limit: int, unavailable: set[str]) -> list[Instance]:
    """The first `limit` candidates not in `unavailable` -- the fetch walk, offline.

    `main` fetches in candidate order and skips an instance whose body is not a
    text NL file; given the set of skipped names, this reproduces its result
    without the network, which is what lets the tests pin both rosters.
    """
    return [inst for inst in candidates if inst.name not in unavailable][:limit]


def write_bounds(path: Path, instances: list[Instance]) -> None:
    """Write bounds.csv for `instances`."""
    with open(path, "w", newline="") as fh:
        fh.write(bounds_text(instances))


def bounds_text(instances: list[Instance]) -> str:
    """bounds.csv's contents for `instances`.

    Schema is the original seven columns plus a trailing `n_disc_vars_bks`
    (appended, so positional readers of columns 0-6 are unaffected).
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
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
    return buf.getvalue()


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


def pool_text(rows: list[dict[str, str]]) -> str:
    """The pool snapshot: every filter survivor's row, POOL_COLUMNS only, in catalogue order.

    Semicolon-separated like the catalogue itself, so `parse_csv` reads it back
    and `select`/`select_heldout` run on it unchanged.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";", lineterminator="\n")
    writer.writerow(POOL_COLUMNS)
    for row in rows:
        if row_to_instance(row) is not None:
            writer.writerow([row.get(col, "") for col in POOL_COLUMNS])
    return buf.getvalue()


def read_bounds_names(path: Path) -> list[str]:
    """Instance names in a bounds.csv, in file order."""
    with open(path, newline="") as fh:
        return [row["instance"] for row in csv.DictReader(fh)]


def roster_walk_skips(candidates: list[Instance], roster: list[str]) -> list[str]:
    """The candidates the published roster's fetch walk must have skipped.

    The published roster is the first len(roster) fetchable entries of `select`'s
    order, so it must be an in-order subsequence of `candidates`, and every
    candidate ahead of its last member that is not in it was skipped as
    unfetchable. Raises ValueError if the roster is not such a subsequence: the
    catalogue has drifted and the roster could no longer be rebuilt from it.
    """
    position = {inst.name: k for k, inst in enumerate(candidates)}
    missing = [name for name in roster if name not in position]
    if missing:
        raise ValueError(f"roster instance(s) no longer pass the filter: {missing}")
    indices = [position[name] for name in roster]
    if indices != sorted(indices):
        raise ValueError("roster order no longer matches the stratified selection order")
    members = set(roster)
    return [inst.name for inst in candidates[: indices[-1] + 1] if inst.name not in members]


def unfetchable_text(entries: list[tuple[str, str, str]]) -> str:
    """unfetchable.csv's contents: (instance, walk, reason) rows."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["instance", "walk", "reason"])
    writer.writerows(entries)
    return buf.getvalue()


def read_unfetchable(path: Path) -> dict[str, set[str]]:
    """unfetchable.csv as {walk: names skipped by that walk}."""
    walks: dict[str, set[str]] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            walks.setdefault(row["walk"], set()).add(row["instance"])
    return walks


def class_counts(instances: list[Instance]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for inst in instances:
        counts[inst.structure] = counts.get(inst.structure, 0) + 1
    return dict(sorted(counts.items()))


def _write_atomic(path: Path, text: str) -> None:
    """Write `text` to `path` via a sibling temp file, so a kill leaves old or new."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        fh.write(text)
    tmp.replace(path)


def _fetch_text_nl(name: str) -> tuple[bytes | None, str | None]:
    """(body, None) for a text NL file, (None, reason) for any other body.

    A network failure raises: the held-out walk must not record a transient
    outage as "unfetchable", because that would change the committed membership.
    """
    data = fetch_bytes(NL_URL_TEMPLATE.format(name=name))
    err = validate_nl(data)
    return (None, err) if err is not None else (data, None)


Skip = tuple[str, str, str]  # (instance, walk, reason): one unfetchable.csv row


def _recheck_published_skips(candidates: list[Instance], roster: list[str]) -> list[Skip] | None:
    """The published walk's skips, each re-fetched to confirm it is still not text NL.

    If one is text NL today, a rebuild of the published roster would admit it and
    change the roster -- report that rather than record it. None on any failure.
    """
    try:
        names = roster_walk_skips(candidates, roster)
    except ValueError as exc:
        print(f"[fail]  published roster is not reproducible from this catalogue: {exc}")
        return None
    skips: list[Skip] = []
    for name in names:
        try:
            body, reason = _fetch_text_nl(name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[fail]  {name}: {exc} (network; not recorded as unfetchable)")
            return None
        if body is not None:
            print(f"[fail]  {name} is text NL now; the published roster would not rebuild")
            return None
        print(f"[skip]  {name} (published walk): {reason}")
        skips.append((name, "published", str(reason)))
    return skips


def _fetch_heldout(
    order: list[Instance], out_dir: Path, limit: int, force: bool
) -> tuple[list[Instance], list[Skip]] | None:
    """Walk `order` fetching text NL into `out_dir` until `limit` are in hand.

    Unlike the published walk, a network failure aborts instead of skipping: a
    transient outage recorded as a skip would change the committed membership.
    """
    out_dir.mkdir(exist_ok=True)
    fetched: list[Instance] = []
    skips: list[Skip] = []
    for inst in order:
        if len(fetched) >= limit:
            break
        dest = out_dir / f"{inst.name}.nl"
        if dest.exists() and not force and dest.stat().st_size > 0:
            fetched.append(inst)
            continue
        try:
            body, reason = _fetch_text_nl(inst.name)
        except (urllib.error.URLError, OSError) as exc:
            print(f"[fail]  {inst.name}: {exc} (network; aborting, roster not written)")
            return None
        if body is None:
            print(f"[skip]  {inst.name} (held-out walk): {reason}")
            skips.append((inst.name, "heldout", str(reason)))
            continue
        dest.write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        print(f"        -> {dest.name} ({len(body)} bytes, sha256 {digest[:12]}...)")
        fetched.append(inst)
    if len(fetched) < limit:
        print(f"[fail]  only {len(fetched)} held-out instances fetchable (target {limit})")
        return None
    return fetched, skips


def build_heldout(here: Path, rows: list[dict[str, str]], limit: int, force: bool) -> int:
    """Draw the held-out roster into `here/heldout/` (HELDOUT.md has the method).

    Refuses to replace a committed held-out roster without --force: the
    membership is fixed before any run, and re-drawing it from a newer catalogue
    would silently change it.
    """
    out_dir = here / HELDOUT_DIRNAME
    bounds_path = out_dir / "bounds.csv"
    if bounds_path.exists() and not force:
        print(f"[refuse] {bounds_path} exists; the held-out membership is committed (--force)")
        return 2

    roster = read_bounds_names(here / "bounds.csv")
    candidates = select(rows)
    published_skips = _recheck_published_skips(candidates, roster)
    if published_skips is None:
        return 1
    walked = _fetch_heldout(select_heldout(rows, set(roster), HELDOUT_SEED), out_dir, limit, force)
    if walked is None:
        return 1
    fetched, heldout_skips = walked

    _write_atomic(out_dir / POOL_FILENAME, pool_text(rows))
    _write_atomic(out_dir / UNFETCHABLE_FILENAME, unfetchable_text(published_skips + heldout_skips))
    _write_atomic(bounds_path, bounds_text(fetched))

    members = set(roster)
    print(f"\npool {len(candidates)}: {class_counts(candidates)}")
    published = [inst for inst in candidates if inst.name in members]
    print(f"published {len(published)}: {class_counts(published)}")
    print(f"held-out  {len(fetched)} (seed {HELDOUT_SEED}): {class_counts(fetched)}")
    print(f"Wrote {bounds_path}")
    return 0


def fetch_published(here: Path, candidates: list[Instance], limit: int, force: bool) -> int:
    """Walk the stratified order, fetching until `limit` instances are in hand.

    bounds.csv is written from the fetched set only, so the roster the runner
    reads is exactly the set of .nl files on disk.
    """
    bounds_path = here / "bounds.csv"
    fetched: list[Instance] = []
    fail = 0
    for inst in candidates:
        if len(fetched) >= limit:
            break
        dest = here / f"{inst.name}.nl"
        if dest.exists() and not force and dest.stat().st_size > 0:
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
    n_mip = sum(1 for inst in fetched if inst.ndiscvars > 0)

    print(f"\nWrote {bounds_path.name} ({len(fetched)} fetched instances)")
    print(f"Done. fetched={len(fetched)} fail={fail} (target {limit}).")
    print(f"  structure mix: {class_counts(fetched)}")
    print(f"  mixed-integer: {n_mip} of {len(fetched)}")
    # A short roster is the only real failure: individual fetch failures are
    # expected (binary-NL instances) and are replaced from the candidate pool.
    return 0 if len(fetched) >= min(limit, len(candidates)) else 1


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--heldout",
        action="store_true",
        help=f"draw the held-out roster into {HELDOUT_DIRNAME}/ (seed {HELDOUT_SEED}; HELDOUT.md)",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=None,
        help=f"read the metadata CSV from this file instead of {CSV_URL} "
        f"(e.g. {HELDOUT_DIRNAME}/{POOL_FILENAME})",
    )
    args = parser.parse_args()
    if args.heldout and args.select_only:
        parser.error("--heldout and --select-only are exclusive")
    return args


def _read_catalogue(path: Path | None) -> list[dict[str, str]] | None:
    """The metadata rows, from `path` or the network; None (after reporting) on failure."""
    print(f"[fetch] {path or CSV_URL}")
    try:
        data = path.read_bytes() if path else fetch_bytes(CSV_URL)
    except (urllib.error.URLError, OSError) as exc:
        print(f"[fail]  could not read metadata CSV: {exc}")
        return None
    rows = parse_csv(data)
    print(f"        {len(rows)} instances in metadata, sha256 {hashlib.sha256(data).hexdigest()}")
    return rows


def main() -> int:
    args = _parse_args()
    here = Path(__file__).resolve().parent
    print("=== MINLPLib download ===")
    print(f"target dir: {here}")

    rows = _read_catalogue(args.catalogue)
    if rows is None:
        return 1
    if args.heldout:
        return build_heldout(here, rows, args.limit, args.force)

    candidates = select(rows)
    print(
        f"\n{len(candidates)} instances pass the filter (budget nvars<={MAX_VARS}, "
        f"ncons<={MAX_CONS}, non-convex, supported ops); "
        f"target roster {args.limit}."
    )

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
        print("\n(selection only — no .nl fetched, bounds.csv left unchanged)")
        return 0

    return fetch_published(here, candidates, args.limit, args.force)


if __name__ == "__main__":
    sys.exit(main())
